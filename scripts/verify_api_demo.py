"""Verify the four typed HTTP views against the saved, real-recording demo.

The demonstration is one preselected scene and three distinct questions. It
establishes transport/checkpoint parity, not population accuracy.
"""
import argparse
import json
from pathlib import Path

from mmso.api.client import Client
from mmso.artifacts import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON written by examples/api_client.py --write-request")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh evidence path")
    payload = json.loads(args.request.read_text())
    response = Client(args.base_url).decide(**payload)
    recorded = json.loads((ROOT / "examples/joint-demo-output.json").read_text())["answers"]
    result = response["results"]
    assert response["checkpoint_sha256"] == sha256(ROOT / "artifacts/joint-full-v2/model.safetensors")
    assert set(result) == {"color", "opposite_ranked", "present", "presence_score"}
    errors = []
    for name, index in [("color", 0), ("opposite_ranked", 1), ("present", 3)]:
        assert set(result[name]["probabilities"]) == set(recorded[index]["probabilities"])
        errors.extend(abs(p - recorded[index]["probabilities"][k]) for k, p in result[name]["probabilities"].items())
        assert result[name]["prediction"] == recorded[index]["prediction"]
    errors.append(abs(result["presence_score"]["expected_value"] - recorded[3]["probabilities"]["true"]))
    assert max(errors) < 1e-5, errors
    assert result["opposite_ranked"]["ranking"][0]["id"] == "color-red"
    assert result["present"]["value"] is False
    assert result["presence_score"]["range"] == [0., 1.]
    # Same actual media, gated through the HTTP path.
    gated_payload = {**payload, "abstain_threshold": 1.0}
    gated = Client(args.base_url).decide(**gated_payload)
    assert all(answer["abstained"] and answer["decision"] is None for answer in gated["results"].values())
    report = {"status": "passed", "kind": "real_fixture_http_parity", "model": response["model"],
              "checkpoint_sha256": response["checkpoint_sha256"],
              "request_sha256": sha256(args.request), "reference_output_sha256": sha256(ROOT / "examples/joint-demo-output.json"),
              "max_probability_absolute_error": max(errors), "typed_views": 4, "distinct_questions": 3,
              "confidence_one_abstention_verified": True,
              "scope": "One fixed real-keyword/generated-panel demo; numerical and serialization parity, not an accuracy benchmark.",
              "response": response, "gated_response": gated}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({key: report[key] for key in ["status", "max_probability_absolute_error", "typed_views", "distinct_questions"]}))


if __name__ == "__main__":
    main()
