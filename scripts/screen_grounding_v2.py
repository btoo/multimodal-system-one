"""Frozen development selection and single fresh confirmation for real GUI controls."""
from __future__ import annotations

import argparse
import json
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from mmso.artifacts import ROOT, local_media_path, provenance, read_manifest, sha256, validate_manifest, write_json, write_manifest
from mmso.grounding import VisionOCR, VARIANTS, center, rank_proposals
from mmso.metrics import point_hit, clustered_accuracy_interval
from mmso.screen import ClipControl, region_center, MODEL_ID, MODEL_REVISION


def counts(predictions, variant):
    values = [p["methods"][variant] for p in predictions]
    return {"examples": len(values), "hits": sum(p["hit"] for p in values),
            "hit_rate": float(np.mean([p["hit"] for p in values])),
            "proposal_center_hits": sum(p["proposal_recall"] for p in values),
            "proposal_center_recall": float(np.mean([p["proposal_recall"] for p in values])),
            "abstentions": sum(p.get("abstained", False) for p in values),
            "zero_proposals": sum(p.get("zero_proposals", False) for p in values),
            "pipeline_p50_ms": float(np.median([p["pipeline_seconds"] for p in values]) * 1000),
            "pipeline_p95_ms": float(np.quantile([p["pipeline_seconds"] for p in values], .95) * 1000)}


def source_snapshot(manifest):
    record = provenance(manifest)
    record["additional_source_hashes"] = {str(p.relative_to(ROOT)): sha256(p) for p in [
        ROOT / "scripts/screen_ocr.swift", ROOT / "scripts/screen_grounding_v2.py", ROOT / "scripts/screen_prepare.py",
        ROOT / "evals/screen-protocol-v2.json"]}
    return record


def evaluate(phase):
    protocol = ROOT / "evals/screen-protocol-v2.json"
    nomination_path = ROOT / "evals/screen-nomination-v2.json"
    if phase == "development":
        manifest = ROOT / "evals/manifests/screen_grounding.jsonl"
        variants = VARIANTS
    else:
        nomination = json.loads(nomination_path.read_text())
        if sha256(protocol) != nomination["protocol_sha256"]:raise ValueError("Frozen protocol changed")
        for name, expected in nomination["inference_source_sha256"].items():
            if sha256(ROOT / name) != expected:raise ValueError(f"Frozen inference source changed: {name}")
        manifest = ROOT / "evals/manifests/screen_grounding_v2.jsonl"
        if sha256(manifest) != nomination["confirmation_manifest_sha256"]:raise ValueError("Frozen confirmation identities changed")
        variants = [nomination["selected_variant"]]
    output = ROOT / ("reports/screens-ocr-" + phase + "-v2")
    if output.exists():raise ValueError("Run IDs are immutable; this evaluation already exists")
    rows = read_manifest(manifest)
    audit = validate_manifest(rows)
    snapshot = source_snapshot(manifest)
    setup = time.perf_counter()
    ocr = VisionOCR()
    model = None
    if phase == "confirmation":
        torch.set_num_threads(2)
        weights = ROOT / "data/models/clip-vit-base-patch32/pytorch_model.bin"
        original = json.loads((ROOT / "reports/screens-clip-v1/report.json").read_text())
        if sha256(weights) != original["model"]["weights_sha256"]:raise ValueError("CLIP fingerprint changed")
        with patch("huggingface_hub.snapshot_download", return_value=str(weights.parent)):
            model = ClipControl("cpu")
        model.score([Image.new("RGB", (224, 224))], ["a computer interface"])
    setup_seconds = time.perf_counter() - setup
    predictions, proposal_rows = [], []
    for row in rows:
        path = local_media_path(ROOT, row["media"][0]["path"])
        start = time.perf_counter()
        proposals, ocr_info = ocr.proposals(path)
        ocr_seconds = time.perf_counter() - start
        methods = {}
        for variant in variants:
            start = time.perf_counter()
            prediction = rank_proposals(proposals, row["instruction"], variant)
            prediction["pipeline_seconds"] = ocr_seconds + time.perf_counter() - start
            methods[variant] = prediction
        if model:
            start = time.perf_counter()
            with Image.open(path) as opened:image = opened.convert("RGB")
            grid_point, regions, scores = model.ground(image, row["instruction"])
            methods["clip_grid"] = {"point": grid_point, "pipeline_seconds": time.perf_counter() - start,
                                    "abstained": False, "zero_proposals": False}
            methods["center"] = {"point": [image.width / 2, image.height / 2], "pipeline_seconds": 0.0,
                                 "abstained": False, "zero_proposals": False}
        # Annotation geometry is consumed only after all methods produce their output.
        box = row["target_bbox_xyxy"]
        proposal_recall = any(point_hit(center(p["bbox_xyxy"]), box) for p in proposals)
        for variant in variants:
            methods[variant]["hit"] = methods[variant]["point"] is not None and point_hit(methods[variant]["point"], box)
            methods[variant]["proposal_recall"] = proposal_recall
        if model:
            for name in ["clip_grid", "center"]:methods[name]["hit"] = point_hit(methods[name]["point"], box)
            methods["clip_grid"]["proposal_recall"] = any(point_hit(region_center(r), box) for r in regions)
            methods["center"]["proposal_recall"] = methods["center"]["hit"]
        predictions.append({"id": row["id"], "application": row["application"], "ui_type": row["ui_type"],
            "image_sha256": row["media"][0]["sha256"], "target_bbox_xyxy": box,
            "proposal_count": len(proposals), "ocr_pipeline_seconds": ocr_seconds, "methods": methods})
        proposal_rows.append({"id": row["id"], "proposals": proposals, "ocr_info": ocr_info})
        print(f"OCR {phase} {len(predictions)}/{len(rows)}", flush=True)
    names = list(predictions[0]["methods"])
    results = {name: counts(predictions, name) for name in names}
    slices = {field: {value: {name: counts([p for p in predictions if p[field] == value], name) for name in names}
                     for value in sorted({p[field] for p in predictions})} for field in ["ui_type", "application"]}
    report = {"status": "completed", "phase": phase, "scope": "Separate pretrained OCR/CLIP controls on public real screenshots",
        "provenance": snapshot, "audit": audit, "protocol_sha256": sha256(protocol), "ocr": ocr.identity(),
        "results": results, "slices": slices, "setup_seconds": setup_seconds,
        "timing": {"ocr_includes": ["subprocess startup", "image file decode", "Vision CPU OCR", "word/line box extraction", "JSON serialization", "proposal cleanup", "lexical ranking"],
                   "grid_includes": ["image read/decode", "fixed crops", "CLIP preprocessing", "CPU scoring", "argmax"],
                   "excludes": ["model/compiler setup", "manifest hash audit", "report serialization"],
                   "center_timing": "not benchmarked; zero is an explicit sentinel"},
        "limitations": ["Balanced custom subset, not the official full benchmark", "Lexical OCR misses icons and semantic paraphrases",
                        "No trained joint/native model or browser execution evidence", "OCR dependency includes the recorded macOS build"]}
    if model:
        selected = variants[0]
        groups = [p["application"] for p in predictions]
        report["paired_delta_vs_grid"] = {"difference": results[selected]["hit_rate"] - results["clip_grid"]["hit_rate"],
            "application_cluster_95ci": clustered_accuracy_interval(
                [float(p["methods"][selected]["hit"]) - float(p["methods"]["clip_grid"]["hit"]) for p in predictions], groups,
                seed=20260925, repeats=5000)}
        report["clip"] = {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": sha256(weights), "device": "cpu"}
    write_manifest(output / "predictions.jsonl", predictions)
    write_manifest(output / "proposals.jsonl", proposal_rows)
    write_json(output / "report.json", report)
    if phase == "development":
        selected = max(VARIANTS, key=lambda name: (results[name]["hits"], -VARIANTS.index(name)))
        confirmation = ROOT / "evals/manifests/screen_grounding_v2.jsonl"
        write_json(nomination_path, {"selected_variant": selected, "selection": "Development hit count; tie favors token_f1",
            "development_hits": {name: results[name]["hits"] for name in VARIANTS},
            "protocol_sha256": sha256(protocol), "confirmation_manifest_sha256": sha256(confirmation),
            "inference_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
                ROOT / "mmso/grounding.py", ROOT / "scripts/screen_ocr.swift", ROOT / "scripts/screen_grounding_v2.py"]},
            "development_report_sha256": sha256(output / "report.json"),
            "confirmation_status_at_nomination": "not evaluated"})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development", "confirmation"])
    evaluate(parser.parse_args().phase)
