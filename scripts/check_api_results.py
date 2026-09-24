"""Check published API schema, parity, timings, and recorded source identity."""
import hashlib
import json
import math
import statistics
import subprocess

from mmso.api.app import create_app
from mmso.api.runtime import CHECKPOINT_SHA256, CONFIG_SHA256
from mmso.artifacts import ROOT, sha256


def read(path):
    return json.loads((ROOT / path).read_text())


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower) if lower != upper else ordered[lower]


def main():
    assert create_app(device="cpu", api_key="").openapi() == read("docs/openapi.json")
    assert sha256(ROOT / "artifacts/joint-full-v2/model.safetensors") == CHECKPOINT_SHA256
    assert sha256(ROOT / "artifacts/joint-full-v2/config.json") == CONFIG_SHA256
    demo = read("reports/api-v1/demo-verification.json")
    assert demo["status"] == "passed" and demo["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert demo["reference_output_sha256"] == sha256(ROOT / "examples/joint-demo-output.json")
    reference = read("examples/joint-demo-output.json")["answers"]
    results = demo["response"]["results"]
    errors = [abs(value - reference[index]["probabilities"][key])
              for name, index in [("color", 0), ("opposite_ranked", 1), ("present", 3)]
              for key, value in results[name]["probabilities"].items()]
    errors.append(abs(results["presence_score"]["expected_value"] - reference[3]["probabilities"]["true"]))
    assert max(errors) == demo["max_probability_absolute_error"] < 1e-5
    assert all(item["abstained"] and item["decision"] is None for item in demo["gated_response"]["results"].values())
    benchmark = read("reports/api-v1/http-benchmark.json")
    samples = benchmark["elapsed_ms"]
    assert len(samples) == benchmark["samples"] and all(math.isfinite(t) and t > 0 for t in samples)
    assert benchmark["model_card"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert benchmark["model_card"]["config_sha256"] == CONFIG_SHA256
    measured = {"min": min(samples), "max": max(samples), "mean": statistics.mean(samples),
                "p50": percentile(samples, .5), "p95": percentile(samples, .95), "p99": percentile(samples, .99)}
    for key, value in measured.items():
        assert math.isclose(value, benchmark["latency_ms"][key], rel_tol=1e-12, abs_tol=1e-12)
    provenance = benchmark["provenance"]
    for path, digest in provenance["source_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{provenance['git_revision']}:{path}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, path
    print("Verified live-generated OpenAPI snapshot, weight/config hashes, real-fixture parity, raw HTTP timings, and benchmark source blobs.")


if __name__ == "__main__":
    main()
