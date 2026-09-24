"""Verify every retained varied-input HTTP timing round and source snapshot."""
import hashlib
import json
import math
import statistics
import subprocess

from benchmark_api import percentile
from mmso.artifacts import ROOT, read_manifest, sha256
from mmso.api.registry import MODEL_REGISTRY
from mmso.optimization_study import MANIFEST


scenes = {s["id"]: s for s in read_manifest(MANIFEST)}
expected_ids = None
for name in ("api-v2-varied-v1", "api-v3-varied-v1", "api-v2-varied-v2", "api-v3-varied-v2", "api-v3-mps-v1"):
    directory = ROOT / "reports" / name
    report = json.loads((directory / "report.json").read_text())
    rows = read_manifest(directory / "responses.jsonl")
    ids = [r["scene_id"] for r in rows]
    assert ids == report["scene_ids"] and len(set(ids)) == len(rows) == report["samples"] == 32
    if expected_ids is None: expected_ids = ids
    assert ids == expected_ids
    registration = MODEL_REGISTRY[report["model"]]
    assert report["model_card"]["checkpoint_sha256"] == registration.checkpoint_sha256
    assert report["model_card"]["config_sha256"] == registration.config_sha256
    assert report["max_native_http_probability_error"] < 1e-5 and report["native_parity_examples"] == 3
    values = [r["elapsed_ms"] for r in rows]
    assert all(math.isfinite(x) and x > 0 for x in values)
    recomputed = {"min": min(values), "max": max(values), "mean": statistics.mean(values), "p50": percentile(values, .5), "p95": percentile(values, .95)}
    for key, value in recomputed.items(): assert math.isclose(value, report["latency_ms"][key], abs_tol=1e-10)
    for row in rows:
        scene = scenes[row["scene_id"]]
        assert row["audio_sha256"] == scene["audio"]["sha256"] and row["image_sha256"] == scene["image"]["sha256"]
        response = row["response"]
        assert response["model"] == report["model"] and response["checkpoint_sha256"] == registration.checkpoint_sha256
        assert {r["type"] for r in response["results"].values()} == {"choice", "ranking", "noul", "score"}
        for result in response["results"].values():
            p = result["probabilities"]
            assert all(0 <= value <= 1 for value in p.values()) and math.isclose(sum(p.values()), 1, abs_tol=1e-6)
    provenance = report["provenance"]
    assert provenance["manifest_sha256"] == sha256(MANIFEST)
    for path, digest in provenance["source_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{provenance['git_revision']}:{path}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, path
    print(f"Verified {name}:32 identical-case comparisons, raw timing statistics, model identity, response distributions and source fingerprints")
