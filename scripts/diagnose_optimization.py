"""Read-only training/development diagnosis; optional CPU gradients, zero updates.

This script never opens calibration or final prediction files. It diagnoses
existing selected checkpoints, not the unretained later weights in old runs.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file

from mmso.artifacts import ROOT, read_manifest, sha256, write_json
from mmso.joint_data import expand_scenes
from mmso.joint_model import Vocabulary
from mmso.joint_training import PairedCache
from mmso.joint_world import JOINT_TASKS
from mmso.scale_study import make_model

RUNS = ["joint-full-v2", "scale-small-v1", "scale-large-v1", "scale-factorized-v1"]
MANIFESTS = {run: ROOT / "evals/manifests" / ("joint_panels_v1.jsonl" if run == "joint-full-v2" else "scale_panels_v1.jsonl") for run in RUNS}
OUTPUT = ROOT / "reports/optimization-diagnosis-v1"


def read_inputs():
    results = {}
    fingerprints = {}
    for run in RUNS:
        training_path = ROOT / "reports" / run / "training.json"
        predictions_path = ROOT / "reports" / run / "development-predictions.jsonl"
        manifest = MANIFESTS[run]
        training = json.loads(training_path.read_text())
        scenes = read_manifest(manifest)
        records = expand_scenes([s for s in scenes if s["split"] == "dev"])
        predictions = read_manifest(predictions_path)
        predicted = {p["id"]: p for p in predictions}
        if len(predicted) != len(predictions) or set(predicted) != {r["id"] for r in records}:
            raise ValueError("Development prediction coverage changed")
        for r in records:
            p = predicted[r["id"]]
            probabilities = np.asarray(p["probabilities"])
            if len(probabilities) != len(r["candidates"]) or not np.isclose(probabilities.sum(), 1):
                raise ValueError("Invalid saved probabilities")
            if p["target_text"] != r["target_text"] or p["predicted_text"] != r["candidates"][int(probabilities.argmax())]["text"]:
                raise ValueError("Saved target or prediction disagrees with manifest")
        results[run] = (training, scenes, records, predicted)
        for path in [training_path, predictions_path, manifest, ROOT / training["checkpoint"]]:
            fingerprints[str(path.relative_to(ROOT))] = sha256(path)
    return results, fingerprints


def summarize(records, predictions):
    groups = defaultdict(list)
    for record in records:
        groups[(record["slice"], record["task"])].append((record, predictions[record["id"]]))
    result = {}
    for (slice_name, task), rows in groups.items():
        correct = [float(p["predicted_text"] == r["target_text"]) for r, p in rows]
        confidence = [max(p["probabilities"]) for _, p in rows]
        entropy = [-sum(p * np.log(max(p, 1e-30)) for p in row["probabilities"]) / np.log(len(row["probabilities"])) for _, row in rows]
        nll = [-np.log(max(p["probabilities"][r["target_index"]], 1e-30)) / np.log(len(r["candidates"])) for r, p in rows]
        targets = Counter(r["target_text"] for r, _ in rows)
        labels = [c["text"] for c in rows[0][0]["candidates"]]
        f1 = []
        for label in labels:
            tp = sum(r["target_text"] == label and p["predicted_text"] == label for r, p in rows)
            fp = sum(r["target_text"] != label and p["predicted_text"] == label for r, p in rows)
            fn = sum(r["target_text"] == label and p["predicted_text"] != label for r, p in rows)
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        item = {"examples": len(rows), "accuracy": float(np.mean(correct)), "macro_f1": float(np.mean(f1)),
                "mean_confidence": float(np.mean(confidence)), "normalized_nll": float(np.mean(nll)),
                "normalized_entropy": float(np.mean(entropy)), "target_counts": dict(targets),
                "predicted_counts": dict(Counter(p["predicted_text"] for _, p in rows))}
        if task.endswith("color") or task.endswith("position"):
            for present in [True, False]:
                subset = [(r, p) for r, p in rows if (r["target_text"] != "not present") == present]
                item["present" if present else "absent"] = {
                    "examples": len(subset),
                    "accuracy": float(np.mean([r["target_text"] == p["predicted_text"] for r, p in subset])),
                    "predict_not_present_rate": float(np.mean([p["predicted_text"] == "not present" for _, p in subset]))}
        result.setdefault(slice_name, {})[task] = item
    return result


def training_history(training):
    history = training["history"]
    selected = next(row for row in history if row["epoch"] == training["configuration"]["selected_epoch"])
    return {"selected": selected, "first": history[0], "last": history[-1],
            "all_epochs": history, "selection_used_calibrated_nll": False,
            "later_checkpoint_logits_retained": False,
            "scope": "The original run retained only its selected checkpoint and selected development predictions; later aggregate histories cannot recover logits or branch accuracies."}


def gradient_probe(config, checkpoint, scenes):
    """Three mean-loss gradient observations on one fixed training-only batch."""
    torch.set_num_threads(2)
    torch.manual_seed(20260925)
    vocabulary = Vocabulary(config["vocabulary"])
    selected = [s for s in scenes if s["split"] == "train"][:32]
    cache = PairedCache(selected, vocabulary, "train")
    model = make_model({"vision": "rgb", **config}, len(vocabulary.tokens))
    if checkpoint:
        model.load_state_dict(load_file(str(checkpoint)))
    else:
        # The same pretrained acoustic representation as the controlled study.
        model = make_model({"vision": "rgb", **config}, len(vocabulary.tokens), ROOT / "artifacts/speech-cnn-v1/model.safetensors")
    model.eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inputs, targets = cache.batch(torch.arange(len(cache.records)), torch.device("cpu"), pairing_generator=torch.Generator().manual_seed(20260926))
    logits = model(*inputs)
    losses = nn.functional.cross_entropy(logits, targets, reduction="none")
    params = [(name, parameter) for name, parameter in model.named_parameters()]
    branches = {"audio": [], "image": [], "shared": []}
    for i, (name, _) in enumerate(params):
        branch = "audio" if name.startswith(("audio.", "audio_projection.")) else "image" if name.startswith("image.") else "shared"
        branches[branch].append(i)
    groups = {"joint": [i for i, r in enumerate(cache.records) if r["task"] in JOINT_TASKS],
              "heard_word": [i for i, r in enumerate(cache.records) if r["task"] == "heard_word"],
              "tile_word": [i for i, r in enumerate(cache.records) if r["task"] == "tile_word"]}
    vectors = {}
    result = {"scene_ids": [s["id"] for s in selected], "train_scenes": len(selected),
              "requests": len(cache.records), "paired_source_pool_scenes": len(selected), "device": "cpu",
              "optimizer_updates": 0, "weights_saved": False, "groups": {}, "gradient_cosines": {}}
    for name, indices in groups.items():
        loss = losses[indices].mean()
        gradients = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
        vectors[name] = {}
        row = {"examples": len(indices), "mean_ce": float(loss.detach()),
               "accuracy": float((logits[indices].argmax(-1) == targets[indices]).float().mean()), "branches": {}}
        for branch, members in branches.items():
            g = torch.cat([(gradients[i] if gradients[i] is not None else torch.zeros_like(params[i][1])).reshape(-1) for i in members])
            vectors[name][branch] = g
            weights = torch.cat([params[i][1].detach().reshape(-1) for i in members])
            row["branches"][branch] = {"parameters": len(g), "l2": float(g.norm()),
                                       "rms": float(g.square().mean().sqrt()),
                                       "gradient_to_weight_norm": float(g.norm() / weights.norm().clamp_min(1e-12))}
        result["groups"][name] = row
    for aux in ["heard_word", "tile_word"]:
        result["gradient_cosines"]["joint_vs_" + aux] = {
            branch: float(nn.functional.cosine_similarity(vectors["joint"][branch], vectors[aux][branch], dim=0)) for branch in branches}
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items()), "Probe mutated weights"
    result["limits"] = "Single selected-checkpoint snapshot on 32 training scenes; not a training trajectory, causal test, or estimate of population gradient imbalance. Auxiliary losses here are existing question-conditioned tasks, not the proposed direct primitive heads."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradients", action="store_true", help="Compute CPU gradients on 32 training scenes, with no parameter updates")
    args = parser.parse_args()
    if (OUTPUT / "report.json").exists():
        raise ValueError("Diagnosis report is immutable; use a new version for a new analysis")
    inputs, fingerprints = read_inputs()
    run_results = {}
    for run, (training, scenes, records, predictions) in inputs.items():
        row = {"development": summarize(records, predictions), "history": training_history(training),
               "checkpoint_sha256": training["checkpoint_sha256"]}
        if args.gradients:
            row["training_gradient_snapshot"] = gradient_probe(training["configuration"], ROOT / training["checkpoint"], scenes)
        run_results[run] = row
    script = Path(__file__).resolve()
    fingerprints[str(script.relative_to(ROOT))] = sha256(script)
    for source in [*sorted((ROOT / "mmso").glob("*.py")), ROOT / "uv.lock"]:
        fingerprints[str(source.relative_to(ROOT))] = sha256(source)
    result = {"kind": "read_only_optimization_diagnosis", "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_hashes": fingerprints, "python": platform.python_version(), "torch": torch.__version__,
              "consumed_partitions": ["train", "dev"], "new_test_inference": False, "optimizer_updates": 0,
              "runs": run_results,
              "cautions": ["Old selected tile_word accuracy conflates visual perception, position routing, question interpretation and answer scoring.",
                           "Raw normalized NLL selected early checkpoints; a later calibration rescue cannot be tested without the unretained logits.",
                           "A positive scalar temperature cannot change argmax accuracy or solve a compositional ranking failure.",
                           "Gradient norms across differently sized encoders are not directly comparable; RMS and parameter-relative values are provided."]}
    write_json(OUTPUT / "report.json", result)
    print(json.dumps({"report": str((OUTPUT / "report.json").relative_to(ROOT)), "runs": list(run_results), "optimizer_updates": 0, "gradients": args.gradients}))


if __name__ == "__main__":
    main()
