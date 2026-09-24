"""A preregistered size-versus-visual-prior experiment; old runs stay immutable."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .artifacts import ROOT, provenance, read_manifest, sha256, write_json, write_manifest
from .audio import choose_device, synchronize
from .data import MINI_SHA256, audio_media, rank
from .joint_data import MANIFEST as OLD_MANIFEST, audit_joint, expand_scenes, seed_for
from .joint_model import NativeDecisionModel, Vocabulary
from .joint_training import PairedCache, decision_metrics, evaluate_logits, fit_joint_temperature
from .joint_world import COLORS, JOINT_TASKS, WORDS, held_pair, make_panel, render_panel

MANIFEST = ROOT / "evals/manifests/scale_panels_v1.jsonl"
PROTOCOL = ROOT / "evals/scale-protocol-v1.json"
NOMINATION = ROOT / "evals/scale-nomination-v1.json"


class FactorizedVision(nn.Module):
    """Pixel-only renderer-specific dark-shape prior; no scene labels at inference."""
    def __init__(self, width):
        super().__init__()
        shape_width = width * 3 // 4
        self.shape = nn.Sequential(
            nn.Conv2d(1, 24, 3, 2, 1), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.GroupNorm(6, 48), nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, shape_width))
        self.color = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, width - shape_width))

    @staticmethod
    def ink_map(patches):
        rgb = (patches + 1) / 2
        return ((0.4 - rgb.amax(dim=1, keepdim=True)) / 0.2).clamp(0, 1)

    def forward(self, patches):
        shape = self.shape(self.ink_map(patches))
        color = self.color(patches.mean(dim=(-2, -1)))
        return torch.cat([shape, color], dim=-1)


def make_model(config, vocabulary_size, audio_initialization=None):
    model = NativeDecisionModel(vocabulary_size, config["width"], config["layers"], audio_initialization)
    if config["vision"] == "factorized":
        model.image = FactorizedVision(config["width"])
    elif config["vision"] != "rgb":
        raise ValueError("Unknown visual encoder")
    return model


def previous_speakers():
    old = read_manifest(ROOT / "evals/manifests/speech_keywords.jsonl")
    joint = read_manifest(OLD_MANIFEST)
    return {r["group_id"].split(":", 1)[1] for r in old} | {s["speaker"] for s in joint}


def audit_scale(scenes, verify_media=True):
    audit = audit_joint(scenes, verify_media=verify_media)
    exposed = previous_speakers()
    fresh = {s["speaker"] for s in scenes if s["split"] == "test"}
    if fresh & exposed:
        raise ValueError("Confirmation speakers appeared in a prior manifest")
    for scene in scenes:
        expected = scene["slice"] == "compositional"
        if any(held_pair(t["word"], t["color"]) != expected for t in scene["panel"]["tiles"]):
            raise ValueError("Held composition policy violated")
    audit["confirmation_speakers_absent_from_all_previous_manifests"] = True
    audit["confirmation_speakers"] = len(fresh)
    audit["confirmation_audio"] = len({s["audio"]["sha256"] for s in scenes if s["split"] == "test"})
    audit["confirmation_word_counts"] = dict(Counter(s["audio_word"] for s in scenes if s["split"] == "test" and s["slice"] == "in_distribution"))
    audit["held_pair_types_known_from_prior_exposed_test"] = True
    return audit


def prepare_scale():
    if MANIFEST.exists():
        return audit_scale(read_manifest(MANIFEST))
    archive = ROOT / "data/downloads/mini_speech_commands.zip"
    if sha256(archive) != MINI_SHA256:
        raise ValueError("Pinned speech archive changed")
    prior = read_manifest(OLD_MANIFEST)
    scenes = [s for s in prior if s["split"] == "train"]
    sources = {split: [s for s in prior if s["split"] == split] for split in ["dev", "calibration"]}
    sources["test"] = []
    seen = previous_speakers()
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist(), key=rank):
            parts = name.split("/")
            if len(parts) != 3 or parts[0] != "mini_speech_commands" or not name.endswith(".wav") or parts[1] not in WORDS:
                continue
            speaker = parts[2].split("_nohash_")[0]
            if speaker in seen:
                continue
            path = ROOT / "data/scale-v1/audio" / parts[1] / parts[2]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = z.read(name)
            if path.exists() and path.read_bytes() != payload:
                raise ValueError("Cached confirmation audio changed")
            path.write_bytes(payload)
            sources["test"].append({"audio": audio_media(path), "audio_word": parts[1], "speaker": speaker})
    for split, rows in sources.items():
        for i, source in enumerate(rows):
            for slice_name in ["in_distribution", "compositional"]:
                identifier = f"scale-v1:{split}:{slice_name}:{i:05d}"
                panel = make_panel(seed_for(identifier), slice_name == "compositional")
                path = ROOT / "data/scale-v1/images" / (identifier.replace(":", "-") + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                render_panel(panel).save(path)
                scenes.append({"id": identifier, "family_id": f"scale-v1:{split}:{i:05d}",
                               "split": split, "slice": slice_name, "speaker": source["speaker"],
                               "audio_word": source["audio_word"], "audio": source["audio"], "panel": panel,
                               "image": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "modality": "image"}})
    audit = audit_scale(scenes)
    write_manifest(MANIFEST, scenes)
    write_json(ROOT / "evals/acquisition/scale_panels_v1.json", {
        "manifest_sha256": sha256(MANIFEST), "protocol_sha256": sha256(PROTOCOL), "audit": audit,
        "source_archive_sha256": MINI_SHA256, "previous_joint_manifest_sha256": sha256(OLD_MANIFEST),
        "scope": "Same known composition gap, fresh confirmation speakers and generated images. No real screenshot claim."})
    return audit


def selection_score(metrics):
    return float(np.mean([metrics["by_slice"][s]["joint_macro_normalized_nll"] for s in ["in_distribution", "compositional"]]))


def source_snapshot():
    snapshot = provenance(MANIFEST)
    for path, digest in snapshot["source_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{snapshot['git_revision']}:{path}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"Commit source before training or evaluation: {path}")
    protocol_blob = subprocess.check_output(["git", "show", f"{snapshot['git_revision']}:evals/scale-protocol-v1.json"], cwd=ROOT)
    if hashlib.sha256(protocol_blob).hexdigest() != sha256(PROTOCOL):
        raise ValueError("Commit protocol before starting")
    snapshot["protocol_sha256"] = sha256(PROTOCOL)
    return snapshot


def train_scale(run_id, device="mps"):
    protocol = json.loads(PROTOCOL.read_text())
    variant = protocol["conditions"][run_id]
    budget = protocol["training"]
    report_dir = ROOT / "reports" / run_id
    checkpoint_dir = ROOT / "artifacts" / run_id
    if report_dir.exists() or checkpoint_dir.exists():
        raise ValueError("Use a fresh immutable run ID")
    snapshot = source_snapshot()
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(budget["seed"])
    chosen = choose_device(device)
    scenes = read_manifest(MANIFEST)
    audit = audit_scale(scenes)
    vocabulary = Vocabulary.from_records(expand_scenes([s for s in scenes if s["split"] == "train"]))
    training = PairedCache(scenes, vocabulary, "train")
    development = PairedCache(scenes, vocabulary, "dev")
    initial = ROOT / "artifacts/speech-cnn-v1/model.safetensors"
    model = make_model(variant, len(vocabulary.tokens), initial).to(chosen)
    audio_params = list(model.audio.parameters())
    audio_ids = {id(p) for p in audio_params}
    other = [p for p in model.parameters() if id(p) not in audio_ids]
    optimizer = torch.optim.AdamW([
        {"params": audio_params, "lr": budget["audio_learning_rate"]},
        {"params": other, "lr": budget["other_learning_rate"]}], weight_decay=budget["weight_decay"])
    generator = torch.Generator().manual_seed(budget["seed"])
    pairing = torch.Generator().manual_seed(budget["pairing_seed"])
    spent = 0.0
    steps = 0
    history = []
    best = float("inf")
    best_state = None
    best_epoch = 0
    epoch = 0
    while steps < budget["optimizer_steps"] and spent < budget["max_train_seconds_per_condition"]:
        epoch += 1
        losses = []
        model.train()
        for indices in torch.randperm(len(training.records), generator=generator).split(budget["batch_size"]):
            if steps >= budget["optimizer_steps"] or spent >= budget["max_train_seconds_per_condition"]:
                break
            synchronize(chosen)
            t = time.perf_counter()
            inputs, targets = training.batch(indices, chosen, pairing_generator=pairing)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), budget["gradient_clip"])
            optimizer.step()
            synchronize(chosen)
            spent += time.perf_counter() - t
            steps += 1
            losses.append(float(loss.detach().cpu()))
        metrics, _ = decision_metrics(development.records, evaluate_logits(model, development, chosen))
        score = selection_score(metrics)
        row = {"epoch": epoch, "steps": steps, "train_seconds": spent, "train_loss": float(np.mean(losses)),
               "selection_normalized_nll": score, "development": metrics["by_slice"]}
        history.append(row)
        print(json.dumps({"run_id": run_id, **row}), flush=True)
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError("No update completed")
    # Check the executed source did not change during the experiment.
    if source_snapshot()["source_hashes"] != snapshot["source_hashes"]:
        raise ValueError("Source changed during training")
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.safetensors"
    save_file(best_state, str(checkpoint))
    model.load_state_dict(best_state)
    metrics, predictions = decision_metrics(development.records, evaluate_logits(model, development, chosen))
    config = {**variant, "architecture": "native-factorized-v1" if variant["vision"] == "factorized" else "native-decision-v1",
              "vocabulary": vocabulary.tokens, "temperature": 1.0, "mode": "full", "seed": budget["seed"],
              "actual_steps": steps, "selected_epoch": best_epoch, "matched_step_budget_completed": steps == budget["optimizer_steps"],
              "audio_initialization_sha256": sha256(initial),
              "scope": "Real one-word speech and generated 2x2 panels; limited vocabulary; research checkpoint, not serving default"}
    write_json(checkpoint_dir / "config.json", config)
    write_manifest(report_dir / "development-predictions.jsonl", predictions)
    report = {"run_id": run_id, "kind": "native_scale_training", "status": "development_complete",
              "configuration": config, "provenance": snapshot, "dataset_audit": audit,
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
              "parameters": sum(p.numel() for p in model.parameters()), "history": history, "development": metrics,
              "selection_normalized_nll": selection_score(metrics), "train_seconds": spent,
              "total_wall_seconds": time.perf_counter() - start, "test_evaluated": False}
    write_json(report_dir / "training.json", report)
    return report


def nominate_scale():
    if NOMINATION.exists():
        raise ValueError("Nomination already recorded")
    protocol = json.loads(PROTOCOL.read_text())
    rows = []
    for run_id in protocol["conditions"]:
        report = json.loads((ROOT / "reports" / run_id / "training.json").read_text())
        if report["test_evaluated"]:
            raise ValueError("Nominate before test")
        rows.append({"run_id": run_id, "development_score": report["selection_normalized_nll"],
                     "checkpoint_sha256": report["checkpoint_sha256"],
                     "matched_step_budget_completed": report["configuration"]["matched_step_budget_completed"]})
    eligible = [r for r in rows if r["matched_step_budget_completed"]]
    if not eligible:
        raise ValueError("No matched-budget candidate completed")
    result = {"protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(MANIFEST),
              "selected_run": min(eligible, key=lambda r: r["development_score"])["run_id"],
              "conditions": rows, "test_opened": False, "serving_default_changed": False}
    write_json(NOMINATION, result)
    return result


def evaluate_scale(run_id, device="mps"):
    snapshot = source_snapshot()
    nomination = json.loads(NOMINATION.read_text())
    blob = subprocess.check_output(["git", "show", f"{snapshot['git_revision']}:evals/scale-nomination-v1.json"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != sha256(NOMINATION):
        raise ValueError("Commit nomination before confirmation")
    frozen_reference = run_id == "joint-full-v2"
    report_dir = ROOT / "reports" / ("scale-served-reference-v1" if frozen_reference else run_id)
    if (report_dir / "evaluation.json").exists():
        raise ValueError("Confirmation already evaluated")
    training = json.loads((ROOT / "reports" / run_id / "training.json").read_text())
    config = training["configuration"]
    checkpoint = ROOT / training["checkpoint"]
    nominated = training if frozen_reference else next(r for r in nomination["conditions"] if r["run_id"] == run_id)
    if sha256(checkpoint) != nominated["checkpoint_sha256"]:
        raise ValueError("Nominated checkpoint changed")
    torch.set_num_threads(4)
    chosen = choose_device(device)
    scenes = read_manifest(MANIFEST)
    audit = audit_scale(scenes)
    vocabulary = Vocabulary(config["vocabulary"])
    model = make_model({"vision": "rgb", **config}, len(vocabulary.tokens)).to(chosen)
    model.load_state_dict(load_file(str(checkpoint)))
    if frozen_reference:
        temperature = json.loads(checkpoint.with_name("config.json").read_text())["temperature"]
    else:
        calibration = PairedCache(scenes, vocabulary, "calibration")
        temperature = fit_joint_temperature(calibration.records, evaluate_logits(model, calibration, chosen))
    test = PairedCache(scenes, vocabulary, "test")
    logits = evaluate_logits(model, test, chosen)
    raw, _ = decision_metrics(test.records, logits)
    calibrated, predictions = decision_metrics(test.records, logits, temperature)
    write_manifest(report_dir / "predictions.jsonl", predictions)
    result = {"run_id": run_id, "kind": "native_scale_confirmation", "status": "confirmation_complete",
              "checkpoint_sha256": sha256(checkpoint), "temperature": temperature, "provenance": snapshot,
              "dataset_audit": audit, "raw": raw, "calibrated": calibrated, "frozen_served_reference": frozen_reference,
              "scope": "Same-gap fresh-speaker confirmation; known held tuple identities; no real-browser result"}
    # Research config carries calibration for an explicit future load, not a registry promotion.
    if not frozen_reference:
        write_json(checkpoint.with_name("config.json"), {**config, "temperature": temperature})
    write_json(report_dir / "evaluation.json", result)
    return result
