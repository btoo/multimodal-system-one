"""Fit fixed CPU ridge readouts on training-only frozen model representations.

These diagnostic classifiers are discarded. No neural parameter is updated,
and no calibration or final example is scored. Heads are equally fitted for
every condition rather than comparing a trained head with a random head.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from safetensors.torch import load_file

from mmso.artifacts import ROOT, read_manifest, sha256, write_json
from mmso.joint_model import audio_tensor, image_tensor
from mmso.joint_world import COLORS, WORDS
from mmso.scale_study import make_model


def inputs(scenes):
    images = torch.stack([image_tensor(ROOT / s["image"]["path"]) for s in scenes])
    cache = {}
    audio = []
    for scene in scenes:
        media = scene["audio"]
        if media["sha256"] not in cache:
            cache[media["sha256"]] = audio_tensor(ROOT / media["path"])
        audio.append(cache[media["sha256"]])
    return images, torch.stack(audio)


@torch.inference_mode()
def representations(model, data):
    visual, acoustic = [], []
    for index in torch.arange(len(data[0])).split(32):
        v, a = model.encode_observations(data[0][index], data[1][index])
        visual.append(v.numpy())
        acoustic.append(a.numpy())
    return np.concatenate(visual), np.concatenate(acoustic)


def labels(scenes):
    return {"glyph": np.array([WORDS.index(tile["word"]) for s in scenes for tile in s["panel"]["tiles"]]),
            "color": np.array([list(COLORS).index(tile["color"]) for s in scenes for tile in s["panel"]["tiles"]]),
            "audio_word": np.array([WORDS.index(s["audio_word"]) for s in scenes])}


def fit(train, target, classes):
    train = np.asarray(train, dtype=np.float64)
    mean, std = train.mean(0), train.std(0).clip(1e-6)
    x = np.column_stack([(train - mean) / std, np.ones(len(train))])
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ np.eye(classes)[target])
    return mean, std, weights


def score(features, targets, probe):
    mean, std, weights = probe
    x = np.column_stack([(np.asarray(features, dtype=np.float64) - mean) / std, np.ones(len(features))])
    predictions = (x @ weights).argmax(-1)
    return {"examples": len(targets), "accuracy": float((predictions == targets).mean()),
            "per_class_accuracy": {str(c): float((predictions[targets == c] == c).mean()) for c in sorted(set(targets.tolist()))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", dest="runs")
    parser.add_argument("--output", default="reports/optimization-diagnosis-v1/primitive-probes.json")
    args = parser.parse_args()
    runs = args.runs or ["joint-full-v2", "scale-small-v1", "scale-large-v1", "scale-factorized-v1"]
    output = ROOT / args.output
    if output.exists():
        raise ValueError("Probe report is immutable; use a new output path")
    manifest = ROOT / "evals/manifests/scale_panels_v1.jsonl"
    all_scenes = read_manifest(manifest)
    training = [s for s in all_scenes if s["split"] == "train"][:256]
    development = {slice_name: [s for s in all_scenes if s["split"] == "dev" and s["slice"] == slice_name]
                   for slice_name in ["in_distribution", "compositional"]}
    assert not {s["speaker"] for s in training} & {s["speaker"] for rows in development.values() for s in rows}
    torch.set_num_threads(2)
    train_inputs = inputs(training)
    dev_inputs = {slice_name: inputs(rows) for slice_name, rows in development.items()}
    train_labels = labels(training)
    dev_labels = {slice_name: labels(rows) for slice_name, rows in development.items()}
    results = {}
    for run in runs:
        report_path = ROOT / "reports" / run / "training.json"
        report = json.loads(report_path.read_text())
        config = report["configuration"]
        checkpoint = ROOT / report["checkpoint"]
        if sha256(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("Checkpoint digest changed")
        model = make_model({"vision": "rgb", **config}, len(config["vocabulary"]))
        model.load_state_dict(load_file(str(checkpoint)))
        model.eval()
        train_visual, train_audio = representations(model, train_inputs)
        features = {"glyph": train_visual.reshape(-1, config["width"]),
                    "color": train_visual.reshape(-1, config["width"]), "audio_word": train_audio}
        probes = {name: fit(features[name], train_labels[name], 4 if name == "color" else 8) for name in features}
        result = {"checkpoint_sha256": sha256(checkpoint), "training_report_sha256": sha256(report_path),
                  "training": {name: score(features[name], train_labels[name], probes[name]) for name in features}, "development": {}}
        for slice_name, data in dev_inputs.items():
            visual, audio = representations(model, data)
            dev_features = {"glyph": visual.reshape(-1, config["width"]), "color": visual.reshape(-1, config["width"]), "audio_word": audio}
            result["development"][slice_name] = {name: score(dev_features[name], dev_labels[slice_name][name], probes[name]) for name in dev_features}
        results[run] = result
    script = Path(__file__).resolve()
    write_json(output, {"kind": "frozen_representation_ridge_diagnostic", "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        "script_sha256": sha256(script), "manifest_sha256": sha256(manifest),
                        "neural_optimizer_updates": 0, "closed_form_probe_fits": len(runs) * 3,
                        "device": "cpu", "torch_threads": 2, "training_panels": len(training),
                        "training_scene_ids": [s["id"] for s in training], "training_glyphs": len(training) * 4,
                        "ridge_lambda": 1.0, "standardization": "training feature mean/std only; unpenalized intercept",
                        "readout_weights_retained": False, "consumed_partitions": ["train", "dev"],
                        "runs": results,
                        "limits": "Fixed small training subset and linear readouts. These are diagnostic classifiers on frozen features, not the served model's question-conditioned outputs. Development has already been exposed. They do not establish real-screenshot or open-vocabulary recognition."})
    print(json.dumps({"output": str(output.relative_to(ROOT)), "neural_updates": 0, "probe_fits": len(runs) * 3}))


if __name__ == "__main__":
    main()
