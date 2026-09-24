"""Deterministic fresh ScreenSpot-Pro identities; target geometry never selects rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from urllib.parse import quote

from PIL import Image

from .artifacts import ROOT, read_manifest, sha256, validate_manifest, write_json, write_manifest
from .data import download, SCREEN_REVISION


def prepare_confirmation():
    protocol_path = ROOT / "evals/screen-protocol-v2.json"
    protocol = json.loads(protocol_path.read_text())
    output = ROOT / "evals/manifests/screen_grounding_v2.jsonl"
    if output.exists():
        raise ValueError("Confirmation manifest is immutable; it already exists")
    old = read_manifest(ROOT / protocol["development"]["manifest"])
    excluded_names = {r["group_id"].removeprefix("screen:") for r in old}
    excluded_hashes = {m["sha256"] for r in old for m in r["media"]}
    selected_names, selected_hashes = set(), set()
    limit = protocol["confirmation"]["download_cap_bytes"]
    total = 0
    base = f"https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/{SCREEN_REVISION}/"
    cache = ROOT / "data/screens_v2"
    source_hashes, rows, duplicate_skips = {}, [], []

    def acquire(relative, maximum=25 << 20):
        nonlocal total
        path = cache / relative
        was_present = path.exists()
        path = download(base + quote(relative, safe="/"), path, max_bytes=min(maximum, limit - total))
        if not was_present:total += path.stat().st_size
        if total > limit:raise ValueError("Exceeded preregistered download budget")
        return path

    card = acquire("README.md", 1 << 20)
    for application in protocol["confirmation"]["applications"]:
        annotation = acquire(f"annotations/{application}.json", 1 << 20)
        source_hashes[application] = sha256(annotation)
        values = json.loads(annotation.read_text())
        for kind in ("text", "icon"):
            pool = sorted((r for r in values if r["ui_type"] == kind),
                key=lambda r: hashlib.sha256(("mmso-screen-v2-confirmation:" + r["id"]).encode()).hexdigest())
            count = 0
            for record in pool:
                name = PurePosixPath(record["img_filename"])
                if name.is_absolute() or ".." in name.parts:raise ValueError("Invalid source image path")
                if str(name) in excluded_names or str(name) in selected_names:continue
                path = acquire("images/" + str(name))
                fingerprint = sha256(path)
                if fingerprint in excluded_hashes or fingerprint in selected_hashes:
                    duplicate_skips.append(record["id"])
                    continue
                with Image.open(path) as image:shape = list(image.size)
                if shape != record["img_size"]:raise ValueError("Image dimensions differ from source metadata")
                selected_names.add(str(name));selected_hashes.add(fingerprint)
                rows.append({"id": "screenspot:" + record["id"], "dataset": "screen_grounding_v2",
                    "kind": "grounding", "split": "test", "group_id": "screen:" + str(name),
                    "instruction": record["instruction"], "application": record["application"],
                    "source_annotation": application, "platform": record["platform"], "ui_type": kind,
                    "image_size": shape, "target_bbox_xyxy": record["bbox"],
                    "media": [{"modality": "image", "path": str(path.relative_to(ROOT)), "sha256": fingerprint}]})
                count += 1
                if count == protocol["confirmation"]["per_application"][kind]:break
            if count != protocol["confirmation"]["per_application"][kind]:
                raise ValueError(f"Insufficient disjoint {kind} screens for {application}")
        print(f"Selected {len(rows)}/64 fresh screens; downloaded {total / 1e6:.1f} MB", flush=True)
    if len(rows) != protocol["confirmation"]["examples"]:raise ValueError("Incomplete confirmation set")
    audit = validate_manifest(rows)
    combined_audit = validate_manifest([*old, *rows])
    assert excluded_names.isdisjoint(selected_names) and excluded_hashes.isdisjoint(selected_hashes)
    write_manifest(output, rows)
    write_json(ROOT / "evals/acquisition/screen_grounding_v2.json", {
        "repository": "https://huggingface.co/datasets/likaixin/ScreenSpot-Pro", "revision": SCREEN_REVISION,
        "license": "MIT per pinned author dataset card", "card_sha256": sha256(card),
        "annotation_sha256": source_hashes, "manifest_sha256": sha256(output),
        "protocol_sha256": sha256(protocol_path), "audit": audit, "combined_old_new_audit": combined_audit,
        "development_disjoint_image_names": True, "development_disjoint_content_hashes": True,
        "all_confirmation_image_names_and_hashes_unique": True, "download_bytes_this_invocation": total,
        "cache_bytes": sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()), "duplicate_skips": duplicate_skips,
        "selection": protocol["confirmation"]["selection"], "official_benchmark_result": False,
        "freshness": "Unseen by this project's previous probes; not a claim about upstream OCR/CLIP pretraining"})
    return output
