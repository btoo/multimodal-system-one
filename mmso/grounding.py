"""Label-blind Apple Vision OCR proposals and two frozen lexical ranking controls."""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from .artifacts import ROOT, sha256

STOPWORDS = set("a an and at be button by check choose click do for from in into is it of on open please press select show the this to use with your".split())
VARIANTS = ("token_f1", "idf_coverage")


def tokens(text):
    # Fixed lightweight plural normalization; no learned or target-specific dictionary.
    values = re.findall(r"[a-z0-9]+", text.lower())
    return [word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word for word in values]


def query_tokens(instruction):
    all_tokens = tokens(instruction)
    useful = [word for word in all_tokens if word not in STOPWORDS]
    return set(useful or all_tokens)


def center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def clean_proposals(proposals, width, height):
    output = []
    seen = {}
    for proposal in sorted(proposals, key=lambda p: (p["kind"] != "word", p["bbox_xyxy"][1], p["bbox_xyxy"][0])):
        if not tokens(proposal["text"]):
            continue
        box = np.asarray(proposal["bbox_xyxy"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Invalid OCR box")
        box[[0, 2]] = box[[0, 2]].clip(0, width)
        box[[1, 3]] = box[[1, 3]].clip(0, height)
        if not box[0] < box[2] or not box[1] < box[3]:
            continue
        point = center(box)
        signature = tuple(tokens(proposal["text"]))
        if any(np.linalg.norm(np.asarray(previous) - point) <= 2 for previous in seen.get(signature, [])):
            continue
        seen.setdefault(signature, []).append(point)
        output.append({"text": proposal["text"], "bbox_xyxy": box.tolist(),
                       "kind": proposal["kind"], "confidence": float(proposal["confidence"])})
    return output


def rank_proposals(proposals, instruction, variant="token_f1"):
    if variant not in VARIANTS:
        raise ValueError("Unknown frozen ranking variant")
    query = query_tokens(instruction)
    sets = [set(tokens(p["text"])) for p in proposals]
    scores = []
    idf = {word: math.log((len(sets) + 1) / (1 + sum(word in words for words in sets))) + 1 for word in query}
    for words in sets:
        common = query & words
        if not query or not common:
            scores.append(0.0)
        elif variant == "token_f1":
            scores.append(2 * len(common) / (len(query) + len(words)))
        else:
            coverage = sum(idf[w] for w in common) / sum(idf.values())
            precision = len(common) / len(words)
            scores.append(coverage * (0.75 + 0.25 * precision))
    order = sorted(range(len(proposals)), key=lambda i: (-scores[i], -proposals[i]["confidence"],
                    len(sets[i]), proposals[i]["bbox_xyxy"][1], proposals[i]["bbox_xyxy"][0]))
    best = order[0] if order and scores[order[0]] > 0 else None
    return {"point": None if best is None else center(proposals[best]["bbox_xyxy"]),
            "winning_proposal": None if best is None else best,
            "score": 0.0 if best is None else scores[best],
            "abstained": best is None, "zero_proposals": not bool(proposals),
            "tie_count": sum(abs(score - scores[best]) < 1e-12 for score in scores) if best is not None else 0,
            "top_indices": order[:10], "top_scores": [scores[i] for i in order[:10]]}


class VisionOCR:
    def __init__(self):
        if platform.system() != "Darwin":
            raise ValueError("Apple Vision OCR requires macOS")
        self.source = ROOT / "scripts/screen_ocr.swift"
        self.binary = ROOT / "data/tools" / ("screen-ocr-" + sha256(self.source)[:16])
        self.binary.parent.mkdir(parents=True, exist_ok=True)
        if not self.binary.exists():
            subprocess.run(["swiftc", "-O", str(self.source), "-o", str(self.binary)], check=True, capture_output=True, text=True)

    def proposals(self, image_path):
        # Only a local image path crosses the subprocess boundary. No instruction, type, or target.
        start = time.perf_counter()
        result = subprocess.run([str(self.binary), str(image_path)], check=True, capture_output=True, text=True, timeout=60)
        data = json.loads(result.stdout)
        proposals = clean_proposals(data.pop("proposals"), data["width"], data["height"])
        data["complete_ocr_seconds"] = time.perf_counter() - start
        return proposals, data

    def identity(self):
        return {"name": "Apple Vision VNRecognizeTextRequest", "revision": 3,
                "recognition_level": "fast", "language": "en-US", "language_correction": False,
                "cpu_only_requested": True, "source_sha256": sha256(self.source), "binary_sha256": sha256(self.binary),
                "macos": subprocess.check_output(["sw_vers"], text=True).strip(),
                "swift": subprocess.check_output(["swift", "--version"], text=True).strip(),
                "limitation": "OCR weights are bundled with macOS; OS build/revision recorded, no independently hashed OCR weight file"}
