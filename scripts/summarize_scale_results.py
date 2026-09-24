"""Derive the scale-study table and paired speaker intervals from saved predictions."""
from collections import defaultdict
import json

import numpy as np

from mmso.artifacts import ROOT, read_manifest, sha256, write_json
from mmso.joint_data import expand_scenes
from mmso.joint_world import JOINT_TASKS
from mmso.scale_study import MANIFEST, NOMINATION, PROTOCOL


def paired_interval(records, first, second, slice_name):
    first = {r["id"]: r for r in first}
    second = {r["id"]: r for r in second}
    groups = defaultdict(list)
    for record in records:
        if record["task"] not in JOINT_TASKS or record["slice"] != slice_name:
            continue
        identifier = record["id"]
        groups[record["speaker"]].append(first[identifier]["correct"] - second[identifier]["correct"])
    groups = [np.array(rows) for _, rows in sorted(groups.items())]
    rng = np.random.default_rng(20260925)
    samples = []
    for _ in range(2000):
        selected = rng.integers(len(groups), size=len(groups))
        samples.append(float(np.concatenate([groups[i] for i in selected]).mean()) * 100)
    return {"mean_percentage_points": float(np.concatenate(groups).mean()) * 100,
            "interval_percentage_points": np.quantile(samples, [0.025, 0.975]).tolist(),
            "speaker_clusters": len(groups), "bootstrap_resamples": len(samples),
            "method": "Paired speaker-cluster percentile bootstrap for fixed checkpoints; not training-seed variance"}


protocol = json.loads(PROTOCOL.read_text())
nomination = json.loads(NOMINATION.read_text())
scenes = read_manifest(MANIFEST)
records = expand_scenes([s for s in scenes if s["split"] == "test"])
runs = [*protocol["conditions"], "scale-served-reference-v1"]
evaluations = {}
predictions = {}
rows = {}
for run in runs:
    report_dir = ROOT / "reports" / run
    evaluation = json.loads((report_dir / "evaluation.json").read_text())
    training_id = "joint-full-v2" if run == "scale-served-reference-v1" else run
    training = json.loads((ROOT / "reports" / training_id / "training.json").read_text())
    evaluations[run] = evaluation
    predictions[run] = read_manifest(report_dir / "predictions.jsonl")
    rows[run] = {"parameters": training["parameters"], "temperature": evaluation["temperature"],
                 "selected_epoch": training["configuration"]["selected_epoch"],
                 "selected_steps": next(h["steps"] for h in training["history"] if h["epoch"] == training["configuration"]["selected_epoch"]),
                 "actual_steps": training["configuration"]["actual_steps"],
                 "train_seconds": training.get("train_seconds"),
                 "selection_normalized_nll": training.get("selection_normalized_nll"),
                 "by_slice": evaluation["calibrated"]["by_slice"],
                 "checkpoint_sha256": evaluation["checkpoint_sha256"]}
comparisons = {}
for candidate in protocol["conditions"]:
    comparisons[candidate] = {}
    for reference in ["scale-small-v1", "scale-served-reference-v1"]:
        if candidate == reference:
            continue
        comparisons[candidate][reference] = {
            slice_name: paired_interval(records, predictions[candidate], predictions[reference], slice_name)
            for slice_name in ["in_distribution", "compositional"]}
summary = {"protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(MANIFEST),
           "selected_run": nomination["selected_run"], "conditions": rows,
           "paired_comparisons": comparisons, "fresh_confirmation_speakers": 80,
           "fresh_confirmation_recordings": 117, "composition_gap_previously_exposed": True,
           "serving_default_changed": False, "training_seeds_per_condition": 1,
           "real_browser_generalization_established": False}
write_json(ROOT / "reports/scale-v1/summary.json", summary)
print(json.dumps(summary, indent=2))
