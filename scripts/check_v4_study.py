"""Audit retained v4 results against immutable cases and model revisions."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = ROOT / "evals/manifests/v4_selection_v1.jsonl"
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    ids = {r["id"]: r for r in rows}
    assert len(ids) == len(rows)
    registry = json.loads((ROOT / "evals/v4-candidates-v1.json").read_text())
    models = {m["key"]: m for m in registry["models"]}
    counts = {"complete_extractions": 0, "failed_attempts": 0, "predictions": 0, "fitted_candidates": 0}
    for path in (ROOT / "reports/v4-selection-v1/attempts").glob("*/result.json"):
        result = json.loads(path.read_text())
        if result.get("status") != "completed": continue
        spec = models[result["key"]]
        assert result["model"]["revision"] == spec["revision"]
        assert result["source_hashes"]["evals/manifests/v4_selection_v1.jsonl"] == sha(manifest)
        predictions_path = path.parent / "predictions.jsonl"
        assert sha(predictions_path) == result["predictions_sha256"]
        predictions = [json.loads(s) for s in predictions_path.read_text().splitlines()]
        assert len(predictions) == len({r["id"] for r in predictions}) == result["completed"] == result["expected"]
        phase = result["phase"]
        if phase in {"development", "adapter-development"}:
            assert {r["id"] for r in predictions} == {r["id"] for r in rows if r["split"] != "confirmation"}
        if phase in {"confirmation", "adapter-confirmation"}:
            assert (ROOT / "evals/v4-nomination-v1.json").exists()
            assert {r["id"] for r in predictions} == {r["id"] for r in rows if r["split"] == "confirmation"}
        for p in predictions:
            assert p["id"] in ids
            if p["status"] != "ok": continue
            expected = len(ids[p["id"]]["choices"])
            probabilities = np.asarray(p["probabilities"])
            assert probabilities.shape == (expected,)
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            assert abs(probabilities.sum() - 1) < 1e-5
            assert all(np.isfinite(p["timings"][k]) and p["timings"][k] >= 0 for k in ["decode_ms", "processor_ms", "forward_ms", "request_ms"])
            measured = sum(p["timings"][k] for k in ["decode_ms", "processor_ms", "h2d_ms", "forward_ms", "score_and_d2h_ms"])
            assert abs(measured - p["timings"]["request_ms"]) < .001
        feature_path = ROOT / ".research/v4/features" / (path.parent.name + ".npz")
        if result["features_sha256"]:
            assert sha(feature_path) == result["features_sha256"]
            features = np.load(feature_path, allow_pickle=False)
            assert set(features["ids"].tolist()) == {p["id"] for p in predictions if p["status"] == "ok"}
            assert np.isfinite(features["features"]).all()
        counts["complete_extractions"] += 1
        counts["predictions"] += len(predictions)
    counts["failed_attempts"] = len(list((ROOT / "reports/v4-selection-v1/attempts").glob("*/failure.json")))
    for path in (ROOT / "reports/v4-selection-v1/candidates").glob("*/selection.json"):
        report = json.loads(path.read_text())
        assert report["revision"] == models[report.get("base_key", report["key"])]["revision"]
        config_path = ROOT / "artifacts/v4-selection" / report["key"] / "config.json"
        config = json.loads(config_path.read_text())
        assert config["temperature"] == report["temperature"]
        assert config["selected_method"] == report["selected_method"]
        if config["readout_sha256"]:
            assert sha(config_path.parent / "readout.safetensors") == config["readout_sha256"]
        counts["fitted_candidates"] += 1
    figures_path = ROOT / "reports/v4-selection-v1/figures.json"
    if figures_path.exists():
        figures = json.loads(figures_path.read_text())
        assert sha(ROOT / "reports/v4-selection-v1/summary.json") == figures["summary_sha256"]
        for name, entry in figures["figures"].items():
            assert sha(ROOT / "docs/assets" / (name + ".svg")) == entry["svg_sha256"]
            assert sha(ROOT / "docs/assets" / (name + ".png")) == entry["png_sha256"]
    costs_path = ROOT / "reports/v4-selection-v1/cost-and-shutdown.json"
    if costs_path.exists():
        costs = json.loads(costs_path.read_text())
        assert costs["gpu_jobs_reserved"] <= 24
        assert costs["maximum_reserved_compute_proxy_usd"] <= costs["study_ceiling_usd"]
        assert costs["all_study_apps_stopped"] and not costs["active_study_containers"]
    parallel_path = ROOT / "reports/v4-selection-v1/attempts/minicpmo45-parallel-v1/parallel-summary.json"
    if parallel_path.exists():
        parallel = json.loads(parallel_path.read_text())
        for measurement in parallel["measurements"]:
            assert measurement["argmax_agreement"] == measurement["questions"]
            assert measurement["maximum_probability_difference"] < .01
            a = np.median(measurement["sequential_shared_media_ms"])
            b = np.median(measurement["packed_ms"])
            assert abs(a / b - measurement["median_speedup"]) < 1e-9
    print(json.dumps({"audit": "passed", **counts}, indent=2))


if __name__ == "__main__":
    main()
