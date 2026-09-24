"""Recompute recipe effects, shortcut diagnostics and the declared promotion gates."""
from collections import defaultdict
import json

import numpy as np

from mmso.artifacts import ROOT, read_manifest, sha256, write_json
from mmso.joint_data import expand_scenes
from mmso.joint_world import JOINT_TASKS
from mmso.optimization_study import MANIFEST, NOMINATION, PROTOCOL, audit_optimization


def paired_interval(records, first, second, slice_name):
    first = {r["id"]: r for r in first}; second = {r["id"]: r for r in second}
    groups = defaultdict(list)
    for r in records:
        if r["task"] in JOINT_TASKS and r["slice"] == slice_name:
            groups[r["speaker"]].append(first[r["id"]]["correct"] - second[r["id"]]["correct"])
    values = [np.asarray(rows) for _, rows in sorted(groups.items())]
    rng = np.random.default_rng(20260924)
    samples = [np.concatenate([values[i] for i in rng.integers(len(values), size=len(values))]).mean() * 100 for _ in range(2000)]
    return {"mean_percentage_points": float(np.concatenate(values).mean() * 100),
            "interval_percentage_points": np.quantile(samples, [.025, .975]).tolist(),
            "speaker_clusters": len(values), "bootstrap_resamples": 2000,
            "scope": "Paired speaker-cluster percentile interval conditional on these checkpoints; not seed variance or a simultaneous interval"}


def shortcut_metrics(records, predictions):
    by_id = {p["id"]: p for p in predictions}; groups = defaultdict(list)
    for r in records:
        if r["task"] in {"color", "opposite_color", "position", "opposite_position"}:
            group = "absent_target" if r["target_text"] == "not present" else "present_target"
            groups[(r["slice"], group)].append(by_id[r["id"]])
    return {f"{s}:{g}": {"examples": len(rows), "accuracy": float(np.mean([r["correct"] for r in rows])),
                         "not_present_prediction_rate": float(np.mean([r["predicted_text"] == "not present" for r in rows]))}
            for (s, g), rows in groups.items()}


def build_summary():
    protocol = json.loads(PROTOCOL.read_text()); nomination = json.loads(NOMINATION.read_text())
    scenes = read_manifest(MANIFEST); audit = audit_optimization(scenes, verify_media=False)
    records = expand_scenes([s for s in scenes if s["split"] == "test"])
    development = expand_scenes([s for s in scenes if s["split"] == "dev"])
    rows = {}; predictions = {}
    reference = "optimization-served-reference-v1"
    for run in [*protocol["conditions"], reference]:
        directory = ROOT / "reports" / run
        evaluation = json.loads((directory / "evaluation.json").read_text())
        training = json.loads((ROOT / "reports" / ("joint-full-v2" if run == reference else run) / "training.json").read_text())
        predictions[run] = read_manifest(directory / "predictions.jsonl")
        config = training["configuration"]
        selected = next(h for h in training["history"] if h["epoch"] == config["selected_epoch"])
        rows[run] = {"parameters": training["parameters"], "seed": config["seed"], "recipe": config.get("recipe", "served_reference"),
                     "actual_steps": config["actual_steps"], "selected_steps": selected["steps"],
                     "train_seconds": training.get("train_seconds"), "temporary_auxiliary_parameters": training.get("temporary_auxiliary_parameters", 0),
                     "temperature": evaluation["temperature"], "raw_by_slice": evaluation["raw"]["by_slice"],
                     "by_slice": evaluation["calibrated"]["by_slice"], "checkpoint_sha256": evaluation["checkpoint_sha256"],
                     "shortcut_slices": shortcut_metrics(records, predictions[run]),
                     "development_gate_passed": training.get("selection_key", [None])[0] == 1 if run != reference else None,
                     "matched_step_budget_completed": training.get("matched_step_budget_completed")}
        if run != reference:
            rows[run]["development_shortcut_slices"] = shortcut_metrics(development, read_manifest(directory / "development-predictions.jsonl"))
    comparisons = {}; seed_effects = {}
    for seed in (20260924, 20260925):
        control = next(r for r, c in protocol["conditions"].items() if c["seed"] == seed and c["recipe"] == "joint_only")
        candidate = next(r for r, c in protocol["conditions"].items() if c["seed"] == seed and c["recipe"] == "primitive_supervision")
        comparisons[candidate] = {baseline: {s: paired_interval(records, predictions[candidate], predictions[baseline], s)
                                           for s in ("in_distribution", "compositional")}
                                  for baseline in (control, reference)}
        delta = float(np.mean([rows[candidate]["by_slice"][s]["joint_macro_accuracy"] - rows[control]["by_slice"][s]["joint_macro_accuracy"]
                               for s in ("in_distribution", "compositional")]))
        seed_effects[str(seed)] = {"candidate": candidate, "control": control, "balanced_accuracy_difference_percentage_points": delta * 100}
    candidate = nomination["candidate_run"]
    treatment_rows = [r for r in rows.values() if r["recipe"] == "primitive_supervision"]
    gates = {"both_treatment_runs_complete": all(r["matched_step_budget_completed"] for r in treatment_rows),
             "both_development_gates_pass": all(r["development_gate_passed"] for r in treatment_rows),
             "balanced_accuracy_gain_in_both_seeds": all(x["balanced_accuracy_difference_percentage_points"] > 0 for x in seed_effects.values()),
             "candidate_composition_interval_positive_vs_served": candidate is not None and comparisons[candidate][reference]["compositional"]["interval_percentage_points"][0] > 0,
             "candidate_id_loss_no_more_than_two_points_vs_served": candidate is not None and comparisons[candidate][reference]["in_distribution"]["mean_percentage_points"] >= -2.}
    return {"protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(MANIFEST), "nomination_sha256": sha256(NOMINATION),
            "conditions": rows, "paired_comparisons": comparisons, "paired_seed_effects": seed_effects,
            "candidate_run": candidate, "promotion_gates": gates, "accuracy_gates_passed": all(gates.values()),
            "deployment_verification_required": True, "serving_default_changed": False,
            "fresh_confirmation_speakers": audit["confirmation_speakers"], "fresh_confirmation_recordings": audit["confirmation_audio"],
            "training_seeds_per_recipe": 2, "composition_gap_previously_exposed": True, "real_browser_generalization_established": False}


if __name__ == "__main__":
    summary = build_summary()
    write_json(ROOT / "reports/optimization-v1/summary.json", summary)
    print(json.dumps({"candidate": summary["candidate_run"], "gates": summary["promotion_gates"], "seed_effects": summary["paired_seed_effects"]}, indent=2))
