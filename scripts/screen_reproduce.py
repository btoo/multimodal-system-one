"""Re-run the original fixed-grid control using the exact pinned, local weights."""
from unittest.mock import patch
import time
import numpy as np
from PIL import Image
import torch

from mmso.artifacts import ROOT, local_media_path, provenance, read_manifest, sha256, validate_manifest, write_json, write_manifest
from mmso.metrics import point_hit
from mmso.screen import ClipControl, MODEL_ID, MODEL_REVISION, region_center


def reproduce():
    output = ROOT / "reports/screens-grid-reproduced-v2"
    if output.exists():raise ValueError("Immutable run already exists")
    manifest = ROOT / "evals/manifests/screen_grounding.jsonl"
    rows = read_manifest(manifest)
    audit = validate_manifest(rows)
    torch.set_num_threads(2)
    weights = ROOT / "data/models/clip-vit-base-patch32/pytorch_model.bin"
    original = __import__('json').loads((ROOT / "reports/screens-clip-v1/report.json").read_text())
    if sha256(weights) != original["model"]["weights_sha256"]:raise ValueError("Pinned CLIP weights differ")
    # Loading only: identical pinned model/processor and existing ground() implementation.
    with patch("huggingface_hub.snapshot_download", return_value=str(weights.parent)):
        model = ClipControl("cpu")
    model.score([Image.new("RGB", (224, 224))], ["a computer interface"])
    predictions = []
    for row in rows:
        start = time.perf_counter()
        with Image.open(local_media_path(ROOT, row["media"][0]["path"])) as opened:image = opened.convert("RGB")
        point, regions, scores = model.ground(image, row["instruction"])
        elapsed = time.perf_counter() - start
        # Ground truth is first accessed after all predictions/proposals are fixed.
        box = row["target_bbox_xyxy"]
        predictions.append({"id": row["id"], "point": point, "hit": point_hit(point, box),
            "center_hit": point_hit([image.width / 2, image.height / 2], box),
            "proposal_recall": any(point_hit(region_center(region), box) for region in regions),
            "pipeline_seconds": elapsed})
        print(f"Legacy grid reproduction {len(predictions)}/{len(rows)}", flush=True)
    write_manifest(output / "predictions.jsonl", predictions)
    report = {"status": "completed", "scope": "Reproduction on 24 exposed cases, now development only",
        "provenance": provenance(manifest), "audit": audit, "source_script_sha256": sha256(__file__),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": sha256(weights), "device": "cpu"},
        "examples": len(rows), "grid_hits": sum(p["hit"] for p in predictions),
        "center_hits": sum(p["center_hit"] for p in predictions),
        "grid_point_coverage_hits": sum(p["proposal_recall"] for p in predictions),
        "pipeline_p50_ms": float(np.median([p["pipeline_seconds"] for p in predictions]) * 1000),
        "pipeline_p95_ms": float(np.quantile([p["pipeline_seconds"] for p in predictions], .95) * 1000),
        "includes": ["image decode", "grid crops", "CLIP preprocessing", "CPU model scoring", "argmax"],
        "excludes": ["model setup and warmup"], "oracle_crops_used": False}
    write_json(output / "report.json", report)
    print({key: report[key] for key in ["examples", "grid_hits", "center_hits", "grid_point_coverage_hits"]})


if __name__ == "__main__":reproduce()
