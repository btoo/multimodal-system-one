"""Jev text-only development reference, separate from H100 latency rankings."""
from pathlib import Path
import json
import time

from mmso.artifacts import ROOT, sha256
from mmso.backbone_selection import summarize
from mmso.workflow_benchmark import JevAdapter, digest


def main():
    output = ROOT / "reports/v4-selection-v1/jev-text-reference-v1"
    if output.exists(): raise ValueError("Immutable reference run already exists")
    output.mkdir(parents=True)
    manifest = ROOT / "evals/manifests/v4_selection_v1.jsonl"
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "development" and r["track"] == "text_rules"]
    adapter = JevAdapter(max_spend_usd=.05, key_file=Path.home() / ".config/miso/typesafe-brian-test.json")
    results, probabilities = [], {}
    for row in rows:
        criteria = {chr(65 + i): choice for i, choice in enumerate(row["choices"])}
        request = {"state": row["question"], "questions": {"decision": {
            "type": "choice", "instructions": "Choose the action required by the stated policy and record.", "criteria": criteria}}}
        try:
            answer = adapter.predict(request)
            p = answer["answers"]["decision"]["probabilities"]
            probabilities[row["id"]] = [p[k] for k in criteria]
            result = {"id": row["id"], "status": "ok", "request_sha256": digest(request),
                      "probabilities": probabilities[row["id"]], "usage": answer["usage"], "cost_usd": answer["cost_usd"],
                      "raw": answer["raw"], "timings": {"request_ms": answer["latency_seconds"] * 1000}}
        except Exception as error:
            result = {"id": row["id"], "status": "error", "error_type": type(error).__name__,
                      "error": str(error), "cost_usd": None}
            results.append(result)
            with (output / "predictions.jsonl").open("a") as stream: stream.write(json.dumps(result) + "\n")
            raise
        results.append(result)
        with (output / "predictions.jsonl").open("a") as stream: stream.write(json.dumps(result) + "\n")
        if len(results) % 16 == 0: print(json.dumps({"completed": len(results), "cost_usd": adapter.spent}), flush=True)
    report = {"model": adapter.name, "manifest_sha256": sha256(manifest), "cases": len(rows),
              "metrics": summarize(rows, results, probabilities), "inference_cost_usd": adapter.spent,
              "scope": "Original synthetic text-rule development controls only; not official TypeSafe workflows",
              "latency_boundary": "Client-visible API latency from this Mac, unlike in-container H100 pipeline timing",
              "eligible_for_same_hardware_frontier": False, "calibration_fitted_here": False,
              "reference_used_for_training": False, "openai_reference_executed": False}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accuracy": report["metrics"]["macro_accuracy"], "cost_usd": adapter.spent}, indent=2))


if __name__ == "__main__":
    main()
