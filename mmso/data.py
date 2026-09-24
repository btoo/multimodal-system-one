"""Pinned public-data adapters. Raw media stays under ignored data/.

The subsets are implementation pilots, not official benchmark submissions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen
import csv
import hashlib
import json
import wave
import zipfile

from PIL import Image

from .artifacts import ROOT, sha256, validate_manifest, write_json, write_manifest

MINI_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
MINI_MD5 = "4b8a67bae2973844e84fa7ac988d1a44"
ESC_REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
SCREEN_REVISION = "210e78d3844251110bff86c95835ebd37a6930fa"
SCREEN_APPS = ["macos_common_macos", "vscode_macos", "excel_macos", "word_macos", "powerpoint_windows", "pycharm_macos"]


def download(url, path, expected_sha=None, max_bytes=32 << 20):
    path = Path(path)
    if path.exists():
        if expected_sha and sha256(path) != expected_sha:
            raise ValueError(f"Cached content mismatch: {path.name}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    try:
        with urlopen(Request(url, headers={"User-Agent": "multimodal-system-one/0.1"}), timeout=60) as response, temp.open("wb") as out:
            size = 0
            while block := response.read(1 << 20):
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"Download exceeds declared cap: {path.name}")
                out.write(block)
        if expected_sha and sha256(temp) != expected_sha:
            raise ValueError(f"Downloaded content mismatch: {path.name}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def rank(value):
    return hashlib.sha256(("mmso-pilot-v1:" + value).encode()).hexdigest()


def speaker_split(speaker):
    bucket = int(rank(speaker)[:8], 16) % 100
    return "train" if bucket < 70 else "dev" if bucket < 80 else "calibration" if bucket < 90 else "test"


def audio_media(path):
    with wave.open(str(path)) as w:
        sr, channels, frames = w.getframerate(), w.getnchannels(), w.getnframes()
    return {"modality": "audio", "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "sample_rate": sr, "channels": channels, "duration_seconds": frames / sr}


def finish(name, rows, source):
    path = ROOT / "evals/manifests" / f"{name}.jsonl"
    audit = validate_manifest(rows)
    if path.exists():
        previous = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if previous != rows:
            raise ValueError("Frozen manifest changed; use a new version rather than silently replacing it")
    write_manifest(path, rows)
    write_json(ROOT / "evals/acquisition" / f"{name}.json", {
        "dataset": name, "source": source, "manifest_sha256": sha256(path), "audit": audit,
        "selection": "Deterministic pilot; split before sample selection. See adapter source.",
        "official_benchmark_result": False})
    print(json.dumps({"dataset": name, "manifest": str(path.relative_to(ROOT)), "audit": audit}), flush=True)
    return path


def prepare_speech():
    archive = download(MINI_URL, ROOT / "data/downloads/mini_speech_commands.zip", max_bytes=190_000_000)
    with archive.open("rb") as f:
        actual_md5 = hashlib.file_digest(f, "md5").hexdigest()
    if actual_md5 != MINI_MD5:
        raise ValueError("Mini Speech Commands differs from the inspected source ETag; do not silently update")
    labels = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
    quotas = {"train": 256, "dev": 32, "calibration": 32, "test": 32}
    rows = []
    with zipfile.ZipFile(archive) as z:
        candidates = []
        for name in z.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) != 3 or parts[0] != "mini_speech_commands" or parts[1] not in labels or not name.endswith(".wav"):
                continue
            speaker = parts[2].split("_nohash_")[0]
            if speaker == parts[2]:
                raise ValueError("Missing speaker ID in source recording")
            candidates.append((speaker_split(speaker), parts[1], name, speaker))
        for split, count in quotas.items():
            for label in labels:
                selected = sorted((x for x in candidates if x[:2] == (split, label)), key=lambda x: rank(x[2]))[:count]
                if len(selected) != count:
                    raise ValueError(f"Not enough grouped records for {split}/{label}")
                for _, _, name, speaker in selected:
                    destination = ROOT / "data/speech_keywords" / label / PurePosixPath(name).name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    payload = z.read(name)
                    if len(payload) > 100_000:
                        raise ValueError("Unexpectedly large speech clip")
                    if destination.exists() and destination.read_bytes() != payload:
                        raise ValueError("Cached speech data changed")
                    destination.write_bytes(payload)
                    rows.append({"id": "mini:"+label+":"+destination.name, "dataset": "speech_keywords",
                                 "kind": "categorical", "split": split, "group_id": "speaker:"+speaker,
                                 "labels": labels, "target": label, "media": [audio_media(destination)]})
        readme = [n for n in z.namelist() if n == "mini_speech_commands/README.md"]
        if readme:
            (ROOT / "data/speech_keywords/SOURCE_README.md").write_bytes(z.read(readme[0]))
    return finish("speech_keywords", rows, {"url": MINI_URL, "archive_sha256": sha256(archive),
        "archive_md5": actual_md5, "license": "CC BY 4.0; source notice retained locally",
        "split_policy": "Custom SHA256-speaker split 70/10/10/10; per-class caps 256/32/32/32; not official splits",
        "scope": "8-word recognition; does not measure sentence intent or unknown commands"})


def prepare_sounds():
    base = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC_REVISION}/"
    csv_path = download(base+"meta/esc50.csv", ROOT / "data/sounds/esc50.csv")
    license_path = download(base+"LICENSE", ROOT / "data/sounds/SOURCE_LICENSE")
    selected = [r for r in csv.DictReader(csv_path.open()) if r["esc10"].lower() == "true"]
    labels = sorted({r["category"] for r in selected})
    split_map = {"1": "train", "2": "train", "3": "dev", "4": "calibration", "5": "test"}
    def one(r):
        filename = r["filename"]
        if PurePosixPath(filename).name != filename:
            raise ValueError("Invalid sound source filename")
        path = download(base+"audio/"+filename, ROOT / "data/sounds/audio" / filename, max_bytes=1_000_000)
        return {"id": "esc10:"+filename, "dataset": "sound_events", "kind": "categorical",
                "split": split_map[r["fold"]], "group_id": "source:"+r["src_file"], "source_fold": int(r["fold"]),
                "labels": labels, "target": r["category"], "media": [audio_media(path)]}
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, row in enumerate(pool.map(one, selected), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"Downloaded/verified ESC-10 {i}/{len(selected)}", flush=True)
    return finish("sound_events", rows, {"repository": "https://github.com/karolpiczak/ESC-50",
        "revision": ESC_REVISION, "metadata_sha256": sha256(csv_path), "license_sha256": sha256(license_path),
        "license": "ESC-10 subset marked CC BY in source README; per-clip notices retained locally",
        "split_policy": "Preserved source-recording folds: 1+2 train, 3 dev, 4 calibration, 5 test; not five-fold CV",
        "scope": "Single-label environmental sound control; not overlapping-event or streaming evaluation"})


def prepare_screens():
    base = f"https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/{SCREEN_REVISION}/"
    card = download(base+"README.md", ROOT / "data/screens/SOURCE_README.md")
    selected = []
    for app in SCREEN_APPS:
        annotation = download(base+f"annotations/{app}.json", ROOT / "data/screens/annotations" / f"{app}.json")
        values = json.loads(annotation.read_text())
        # Two text and two icon targets per app; target size/position never influences selection.
        for ui_type in ["text", "icon"]:
            pool = sorted((r for r in values if r["ui_type"] == ui_type), key=lambda r: rank(r["id"]))
            if len(pool) < 2:
                raise ValueError(f"Insufficient {ui_type} targets in {app}")
            selected.extend(pool[:2])
    def one(r):
        name = PurePosixPath(r["img_filename"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid screen source path")
        image = download(base+"images/"+quote(str(name)), ROOT / "data/screens/images" / name, max_bytes=25 << 20)
        with Image.open(image) as im:
            actual = list(im.size)
        if actual != r["img_size"]:
            raise ValueError("Source image dimensions disagree with annotation")
        return {"id": "screenspot:"+r["id"], "dataset": "screen_grounding", "kind": "grounding",
                "split": "external_probe", "group_id": "screen:"+str(name), "instruction": r["instruction"],
                "application": r["application"], "platform": r["platform"], "ui_type": r["ui_type"],
                "image_size": actual, "target_bbox_xyxy": r["bbox"],
                "media": [{"modality": "image", "path": str(image.relative_to(ROOT)), "sha256": sha256(image)}]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, selected))
    return finish("screen_grounding", rows, {"repository": "https://huggingface.co/datasets/likaixin/ScreenSpot-Pro",
        "revision": SCREEN_REVISION, "card_sha256": sha256(card), "license": "Author dataset card: MIT",
        "split_policy": "24-case app/type-stratified external implementation probe; no training/tuning, not full benchmark",
        "scope": "Six apps; 2 icon and 2 text cases each. Not browser workflows or screen-state classification."})


PREPARERS = {"speech_keywords": prepare_speech, "sound_events": prepare_sounds, "screen_grounding": prepare_screens}
