"""Recompute scale-study scores and verify frozen source/data/checkpoint lineage."""
import hashlib
import json
import math
import subprocess

import numpy as np

from mmso.artifacts import ROOT, read_manifest, sha256
from mmso.joint_data import expand_scenes
from mmso.joint_training import decision_metrics
from mmso.metrics import aligned_predictions
from mmso.scale_study import MANIFEST, NOMINATION, PROTOCOL, audit_scale, selection_score


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8), (a, b)
    else:
        assert a == b, (a, b)


def source_check(prov):
    assert prov["manifest_sha256"] == sha256(MANIFEST)
    assert prov["protocol_sha256"] == sha256(PROTOCOL)
    for path, digest in prov["source_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{prov['git_revision']}:{path}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, path
    for path in [PROTOCOL, MANIFEST]:
        blob = subprocess.check_output(["git", "show", f"{prov['git_revision']}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == sha256(path), path


scenes = read_manifest(MANIFEST)
audit = audit_scale(scenes, verify_media=False)
assert audit["confirmation_speakers"] == 80
assert audit["confirmation_audio"] == 117
protocol = json.loads(PROTOCOL.read_text())
nomination = json.loads(NOMINATION.read_text())
development = expand_scenes([s for s in scenes if s["split"] == "dev"])
test = expand_scenes([s for s in scenes if s["split"] == "test"])
steps = []
for run in protocol["conditions"]:
    report_dir = ROOT / "reports" / run
    training = json.loads((report_dir / "training.json").read_text())
    assert sha256(ROOT / training["checkpoint"]) == training["checkpoint_sha256"]
    source_check(training["provenance"])
    pred = aligned_predictions(development, read_manifest(report_dir / "development-predictions.jsonl"))
    metrics, _ = decision_metrics(development, [np.log(p["probabilities"]) for p in pred])
    close(metrics, training["development"])
    close(selection_score(metrics), training["selection_normalized_nll"])
    steps.append(training["configuration"]["actual_steps"])
    assert training["configuration"]["matched_step_budget_completed"] == (steps[-1] == protocol["training"]["optimizer_steps"])
    nominated = next(r for r in nomination["conditions"] if r["run_id"] == run)
    assert nominated["checkpoint_sha256"] == training["checkpoint_sha256"]
for run in [*protocol["conditions"], "scale-served-reference-v1"]:
    report_dir = ROOT / "reports" / run
    evaluation = json.loads((report_dir / "evaluation.json").read_text())
    source_check(evaluation["provenance"])
    pred = aligned_predictions(test, read_manifest(report_dir / "predictions.jsonl"))
    log_p = [np.log(p["probabilities"]) for p in pred]
    metrics, _ = decision_metrics(test, log_p)
    raw, _ = decision_metrics(test, [z * evaluation["temperature"] for z in log_p])
    close(metrics, evaluation["calibrated"])
    close(raw, evaluation["raw"])
    checkpoint_run = "joint-full-v2" if run == "scale-served-reference-v1" else run
    assert sha256(ROOT / "artifacts" / checkpoint_run / "model.safetensors") == evaluation["checkpoint_sha256"]
eligible = [r for r in nomination["conditions"] if r["matched_step_budget_completed"]]
assert min(eligible, key=lambda r: r["development_score"])["run_id"] == nomination["selected_run"]
print(f"Verified scale-study source/data/checkpoint hashes, fresh speaker disjointness, nomination, exact prediction coverage and recomputed metrics. Steps: {steps}")
