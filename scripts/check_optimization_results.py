"""Verify optimization study lineage, checkpoint selection and all saved metrics."""
import hashlib
import json
import math
import subprocess

import numpy as np

from mmso.artifacts import ROOT, read_manifest, sha256
from mmso.joint_data import expand_scenes
from mmso.joint_training import decision_metrics
from mmso.metrics import aligned_predictions
from mmso.optimization_study import MANIFEST, PROTOCOL, NOMINATION, audit_optimization, selection_key
from summarize_optimization_results import build_summary


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: close(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): close(x, y)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8), (a, b)
    else: assert a == b, (a, b)


def source_check(prov):
    assert prov["manifest_sha256"] == sha256(MANIFEST)
    assert prov["protocol_sha256"] == sha256(PROTOCOL)
    for path, digest in {**prov["source_hashes"], str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
                         str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL)}.items():
        blob = subprocess.check_output(["git", "show", f"{prov['git_revision']}:{path}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, path


scenes = read_manifest(MANIFEST); audit = audit_optimization(scenes, verify_media=False)
assert audit["confirmation_audio"] == 256 and audit["confirmation_speakers"] == 58
protocol = json.loads(PROTOCOL.read_text()); nomination = json.loads(NOMINATION.read_text())
assert nomination["manifest_sha256"] == sha256(MANIFEST) and nomination["protocol_sha256"] == sha256(PROTOCOL)
development = expand_scenes([s for s in scenes if s["split"] == "dev"])
test = expand_scenes([s for s in scenes if s["split"] == "test"])
for run in protocol["conditions"]:
    directory = ROOT / "reports" / run
    training = json.loads((directory / "training.json").read_text())
    source_check(training["provenance"])
    assert sha256(ROOT / training["checkpoint"]) == training["checkpoint_sha256"]
    assert sha256((ROOT / training["checkpoint"]).with_name("final.safetensors")) == training["final_checkpoint_sha256"]
    assert training["configuration"]["actual_steps"] == protocol["training"]["optimizer_steps"]
    for filename, key in [("development-predictions.jsonl", "development"), ("final-development-predictions.jsonl", "final_development")]:
        prediction = aligned_predictions(development, read_manifest(directory / filename))
        metrics, _ = decision_metrics(development, [np.log(p["probabilities"]) for p in prediction])
        close(metrics, training[key])
    close(list(selection_key(training["development"])), training["selection_key"])
    assert max(training["history"], key=lambda row: tuple(row["selection_key"]))["epoch"] == training["configuration"]["selected_epoch"]
    assert next(r for r in nomination["conditions"] if r["run_id"] == run)["checkpoint_sha256"] == training["checkpoint_sha256"]
for run in [*protocol["conditions"], "optimization-served-reference-v1"]:
    directory = ROOT / "reports" / run
    evaluation = json.loads((directory / "evaluation.json").read_text()); source_check(evaluation["provenance"])
    blob = subprocess.check_output(["git", "show", f"{evaluation['provenance']['git_revision']}:{NOMINATION.relative_to(ROOT)}"], cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest() == sha256(NOMINATION)
    prediction = aligned_predictions(test, read_manifest(directory / "predictions.jsonl"))
    logs = [np.log(p["probabilities"]) for p in prediction]
    calibrated, _ = decision_metrics(test, logs); raw, _ = decision_metrics(test, [z * evaluation["temperature"] for z in logs])
    close(calibrated, evaluation["calibrated"]); close(raw, evaluation["raw"])
    checkpoint_run = "joint-full-v2" if run == "optimization-served-reference-v1" else run
    assert sha256(ROOT / "artifacts" / checkpoint_run / "model.safetensors") == evaluation["checkpoint_sha256"]
close(build_summary(), json.loads((ROOT / "reports/optimization-v1/summary.json").read_text()))
print("Verified all selected/final checkpoints, exact predictions, metrics, seed effects, shortcut slices, speaker intervals, source/data lineage and pre-test nomination.")
