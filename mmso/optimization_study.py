"""Primitive-supervision curriculum with matched seeds and fresh speech confirmation.

Temporary supervised heads exist only in training/diagnostics. The retained
inference checkpoint has exactly the original native candidate architecture.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
import subprocess
import tarfile
import time

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .artifacts import ROOT, provenance, read_manifest, sha256, write_json, write_manifest
from .audio import choose_device, synchronize
from .data import audio_media, download, rank
from .joint_data import audit_joint, expand_scenes, seed_for
from .joint_model import NativeDecisionModel, Vocabulary
from .joint_training import PairedCache, decision_metrics, evaluate_logits, fit_joint_temperature
from .joint_world import COLORS, WORDS, held_pair, make_panel, render_panel

MANIFEST = ROOT / "evals/manifests/optimization_panels_v1.jsonl"
PROTOCOL = ROOT / "evals/optimization-protocol-v1.json"
NOMINATION = ROOT / "evals/optimization-nomination-v1.json"
ARCHIVE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
ARCHIVE_SHA = "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df"


def prior_exposure():
    speech = read_manifest(ROOT / "evals/manifests/speech_keywords.jsonl")
    speakers = {r["group_id"].split(":", 1)[1] for r in speech}
    audio_hashes = {m["sha256"] for r in speech for m in r["media"]}
    images = set()
    for name in ("joint_panels_v1", "scale_panels_v1"):
        scenes = read_manifest(ROOT / f"evals/manifests/{name}.jsonl")
        speakers.update(s["speaker"] for s in scenes)
        audio_hashes.update(s["audio"]["sha256"] for s in scenes)
        images.update(s["image"]["sha256"] for s in scenes)
    return speakers, audio_hashes, images


def audit_optimization(scenes, verify_media=True):
    result = audit_joint(scenes, verify_media)
    exposed, old_audio, old_images = prior_exposure()
    test = [s for s in scenes if s["split"] == "test"]
    if any(s["speaker"] in exposed or s["audio"]["sha256"] in old_audio or s["image"]["sha256"] in old_images for s in test):
        raise ValueError("Confirmation overlaps prior speaker or media exposure")
    for scene in scenes:
        if any(held_pair(t["word"], t["color"]) != (scene["slice"] == "compositional") for t in scene["panel"]["tiles"]):
            raise ValueError("Held command/color policy violated")
    result.update(confirmation_speakers=len({s["speaker"] for s in test}),
                  confirmation_audio=len({s["audio"]["sha256"] for s in test}),
                  confirmation_word_counts=dict(Counter(s["audio_word"] for s in test if s["slice"] == "in_distribution")),
                  confirmation_absent_from_all_previous_manifests=True,
                  composition_gap_identities_previously_exposed=True)
    return result


def prepare_optimization():
    if MANIFEST.exists():
        return audit_optimization(read_manifest(MANIFEST))
    archive = download(ARCHIVE_URL, ROOT / "data/downloads/speech_commands_test_set_v0.02.tar.gz",
                       expected_sha=ARCHIVE_SHA, max_bytes=113_000_000)
    exposed, old_audio, old_images = prior_exposure()
    pools = defaultdict(list)
    with tarfile.open(archive) as source:
        for member in source:
            parts = member.name.removeprefix("./").split("/")
            if not member.isfile() or len(parts) != 2 or parts[0] not in WORDS:
                continue
            if not re.fullmatch(r"[0-9a-f]+_nohash_[0-9]+\.wav", parts[1]) or member.size > 128_000:
                continue
            speaker = parts[1].split("_nohash_")[0]
            if speaker in exposed:
                continue
            payload = source.extractfile(member).read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in old_audio:
                pools[parts[0]].append((parts[1], speaker, digest, payload))
        notice = source.extractfile("./README.md").read()
    raw_root = ROOT / "data/optimization-v1"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "SOURCE-README.md").write_bytes(notice)
    selected = []
    used_hashes = set()
    for word in WORDS:
        # One recording per speaker per word before considering repeats.
        used_speakers = set()
        for filename, speaker, digest, payload in sorted(pools[word], key=lambda x: rank("optimization:" + word + "/" + x[0])):
            if speaker in used_speakers or digest in used_hashes:
                continue
            path = raw_root / "audio" / word / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            selected.append({"audio_word": word, "speaker": speaker, "audio": audio_media(path)})
            used_speakers.add(speaker)
            used_hashes.add(digest)
            if len(used_speakers) == 32:
                break
        if len(used_speakers) != 32:
            raise ValueError(f"Fewer than 32 new speakers for {word}; revise protocol before sampling")
    scenes = [s for s in read_manifest(ROOT / "evals/manifests/scale_panels_v1.jsonl") if s["split"] != "test"]
    for index, row in enumerate(selected):
        for slice_name in ("in_distribution", "compositional"):
            identifier = f"optimization-v1:test:{slice_name}:{index:05d}"
            panel = make_panel(seed_for(identifier), slice_name == "compositional")
            path = raw_root / "images" / (identifier.replace(":", "-") + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            render_panel(panel).save(path)
            scenes.append({**row, "id": identifier, "family_id": f"optimization-v1:test:{index:05d}",
                           "split": "test", "slice": slice_name, "panel": panel,
                           "image": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "modality": "image"}})
    audit = audit_optimization(scenes)
    write_manifest(MANIFEST, scenes)
    write_json(ROOT / "evals/acquisition/optimization_panels_v1.json", {
        "source_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA, "archive_bytes": archive.stat().st_size,
        "upstream_checksum_blob": "https://api.github.com/repos/tensorflow/datasets/git/blobs/ebcb6fa83ebb4f8ab6e780290bf3f82b538b3287",
        "upstream_checksum_git_blob": "ebcb6fa83ebb4f8ab6e780290bf3f82b538b3287",
        "license": "Speech Commands, Pete Warden / TensorFlow, CC BY 4.0; source README retained locally",
        "source_is_official_test_archive": True, "official_benchmark_result": False,
        "selection": "Custom 8-word subset, 32 distinct new speakers per word, deterministic hashed order, no prior speaker/hash overlap",
        "eligible_recordings_by_word": {w: len(pools[w]) for w in WORDS},
        "manifest_sha256": sha256(MANIFEST), "protocol_sha256": sha256(PROTOCOL), "audit": audit,
        "scope": "Fresh human keyword voices plus generated panels; same known composition gap, no real screenshot or sentence claim"})
    return audit


def selection_key(metrics):
    slices = metrics["by_slice"]
    identity = slices["in_distribution"]["joint_macro_accuracy"]
    composition = slices["compositional"]["joint_macro_accuracy"]
    nll = np.mean([slices[s]["joint_macro_normalized_nll"] for s in ("in_distribution", "compositional")])
    return (int(identity >= .65), (identity + composition) / 2, -float(nll))


def snapshot():
    result = provenance(MANIFEST)
    result["protocol_sha256"] = sha256(PROTOCOL)
    for path, digest in {**result["source_hashes"], str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL),
                         str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST)}.items():
        blob = subprocess.check_output(["git", "show", f"{result['git_revision']}:{path}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"Commit exact source, data and protocol before execution: {path}")
    return result


class PrimitiveHeads(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.word = nn.Linear(width, len(WORDS))
        self.color = nn.Linear(width, len(COLORS))

    def losses(self, visual, acoustic, words, colors, heard):
        return {"visual_word": nn.functional.cross_entropy(self.word(visual).flatten(0, 1), words.flatten()),
                "visual_color": nn.functional.cross_entropy(self.color(visual).flatten(0, 1), colors.flatten()),
                "audio_word": nn.functional.cross_entropy(self.word(acoustic), heard)}


class PrimitiveCache(PairedCache):
    def __init__(self, scenes, vocabulary, split):
        super().__init__(scenes, vocabulary, split)
        self.visual_words = torch.tensor([[WORDS.index(t["word"]) for t in s["panel"]["tiles"]] for s in self.scenes])
        self.visual_colors = torch.tensor([[list(COLORS).index(t["color"]) for t in s["panel"]["tiles"]] for s in self.scenes])
        self.heard = torch.tensor([WORDS.index(s["audio_word"]) for s in self.scenes])

    def training_batch(self, indices, device, generator):
        if self.counterfactual_targets is None:
            raise ValueError("Primitive updates require the training split")
        scene_indices = self.index[indices]
        heard = torch.randint(len(WORDS), (len(indices),), generator=generator)
        draws = torch.randint(2**30, (len(indices),), generator=generator)
        audio_indices = torch.tensor([self.audio_pools[WORDS[int(w)]][int(d) % len(self.audio_pools[WORDS[int(w)]])]
                                      for w, d in zip(heard, draws)])
        inputs = [self.images[scene_indices], self.audio[audio_indices], self.question[indices], self.candidates[indices], self.mask[indices]]
        targets = self.counterfactual_targets[indices, heard]
        primitives = [self.visual_words[scene_indices], self.visual_colors[scene_indices], heard]
        return [x.to(device) for x in inputs], targets.to(device), [x.to(device) for x in primitives]


@torch.inference_mode()
def primitive_diagnostics(model, heads, data, device):
    model.eval(); heads.eval()
    correct = Counter(); counts = Counter()
    for indices in torch.arange(len(data.scenes)).split(64):
        visual, acoustic = model.encode_observations(data.images[indices].to(device), data.audio[indices].to(device))
        predictions = {"visual_word": heads.word(visual).argmax(-1).cpu(), "visual_color": heads.color(visual).argmax(-1).cpu(),
                       "audio_word": heads.word(acoustic).argmax(-1).cpu()}
        targets = {"visual_word": data.visual_words[indices], "visual_color": data.visual_colors[indices], "audio_word": data.heard[indices]}
        for j, index in enumerate(indices):
            slice_name = data.scenes[int(index)]["slice"]
            for task in predictions:
                match = predictions[task][j] == targets[task][j]
                correct[slice_name + ":" + task] += int(match.sum())
                counts[slice_name + ":" + task] += match.numel()
    return {key: {"accuracy": correct[key] / count, "examples": count} for key, count in counts.items()}


def train_optimization(run_id, device="mps"):
    protocol = json.loads(PROTOCOL.read_text())
    condition = protocol["conditions"][run_id]
    budget = protocol["training"]
    report_dir, checkpoint_dir = ROOT / "reports" / run_id, ROOT / "artifacts" / run_id
    if report_dir.exists() or checkpoint_dir.exists():
        raise ValueError("Run IDs are immutable")
    source = snapshot(); start = time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(condition["seed"])
    chosen = choose_device(device)
    scenes = read_manifest(MANIFEST); audit = audit_optimization(scenes)
    vocabulary = Vocabulary.from_records(expand_scenes([s for s in scenes if s["split"] == "train"]))
    training = PrimitiveCache(scenes, vocabulary, "train")
    development = PrimitiveCache(scenes, vocabulary, "dev")
    initial = ROOT / "artifacts/speech-cnn-v1/model.safetensors"
    model = NativeDecisionModel(len(vocabulary.tokens), 128, 2, initial).to(chosen)
    heads = PrimitiveHeads(128).to(chosen)
    audio_parameters = list(model.audio.parameters()); audio_ids = {id(p) for p in audio_parameters}
    other = [p for p in model.parameters() if id(p) not in audio_ids] + list(heads.parameters())
    optimizer = torch.optim.AdamW([{"params": audio_parameters, "lr": 1e-4}, {"params": other, "lr": 3e-4}], weight_decay=.01)
    generator = torch.Generator().manual_seed(condition["seed"])
    pairing = torch.Generator().manual_seed(condition["seed"] + 1)
    curriculum = condition["recipe"] == "primitive_supervision"
    diagnostics = {"before_training": primitive_diagnostics(model, heads, development, chosen)} if curriculum else {}
    history = []; steps = 0; spent = 0.; epoch = 0; best_key = None; best_state = None; best_heads = None; best_epoch = 0
    while steps < budget["optimizer_steps"] and spent < budget["max_train_seconds_per_run"]:
        epoch += 1; losses = []; primitive_losses = []
        model.train(); heads.train()
        for indices in torch.randperm(len(training.records), generator=generator).split(64):
            if steps >= budget["optimizer_steps"] or spent >= budget["max_train_seconds_per_run"]:
                break
            synchronize(chosen); tick = time.perf_counter()
            inputs, targets, primitive_targets = training.training_batch(indices, chosen, pairing)
            optimizer.zero_grad(set_to_none=True)
            visual, acoustic = model.encode_observations(inputs[0], inputs[1])
            warmup = curriculum and steps < budget["primitive_warmup_steps"]
            primitive_loss = sum(heads.losses(visual, acoustic, *primitive_targets).values()) if curriculum else torch.zeros((), device=chosen)
            if warmup:
                loss = primitive_loss
            else:
                logits = model.decide(visual, acoustic, *inputs[2:])
                loss = nn.functional.cross_entropy(logits, targets) + (budget["auxiliary_weight"] * primitive_loss if curriculum else 0)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            loss.backward(); nn.utils.clip_grad_norm_([*model.parameters(), *heads.parameters()], 1.)
            optimizer.step(); synchronize(chosen)
            spent += time.perf_counter() - tick; steps += 1
            losses.append(float(loss.detach().cpu())); primitive_losses.append(float(primitive_loss.detach().cpu()))
            if curriculum and steps == budget["primitive_warmup_steps"]:
                diagnostics["after_warmup"] = primitive_diagnostics(model, heads, development, chosen)
                model.train(); heads.train()
        metrics, _ = decision_metrics(development.records, evaluate_logits(model, development, chosen))
        key = selection_key(metrics)
        row = {"epoch": epoch, "steps": steps, "train_seconds": spent, "objective": float(np.mean(losses)),
               "primitive_loss_sum": float(np.mean(primitive_losses)), "selection_key": list(key), "development": metrics["by_slice"]}
        history.append(row); print(json.dumps({"run_id": run_id, **row}), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
            best_heads = {k: v.detach().cpu().clone().contiguous() for k, v in heads.state_dict().items()}
    if best_state is None:
        raise ValueError("No completed updates")
    if snapshot()["source_hashes"] != source["source_hashes"]:
        raise ValueError("Source changed during training")
    checkpoint_dir.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(checkpoint_dir / "final.safetensors"))
    final_metrics, final_predictions = decision_metrics(development.records, evaluate_logits(model, development, chosen))
    write_manifest(report_dir / "final-development-predictions.jsonl", final_predictions)
    if curriculum:
        diagnostics["final"] = primitive_diagnostics(model, heads, development, chosen)
        save_file({k: v.detach().cpu().contiguous() for k, v in heads.state_dict().items()}, str(checkpoint_dir / "final-auxiliary.safetensors"))
    save_file(best_state, str(checkpoint_dir / "model.safetensors"))
    if curriculum: save_file(best_heads, str(checkpoint_dir / "auxiliary.safetensors"))
    model.load_state_dict(best_state); heads.load_state_dict(best_heads)
    selected_metrics, selected_predictions = decision_metrics(development.records, evaluate_logits(model, development, chosen))
    write_manifest(report_dir / "development-predictions.jsonl", selected_predictions)
    if curriculum: diagnostics["selected"] = primitive_diagnostics(model, heads, development, chosen)
    config = {"architecture": "native-decision-v1", "width": 128, "layers": 2, "mode": "full", "vocabulary": vocabulary.tokens,
              "temperature": 1., **condition, "actual_steps": steps, "selected_epoch": best_epoch,
              "audio_initialization_sha256": sha256(initial),
              "scope": "Eight human spoken keywords and generated 2x2 panels; training-only primitive heads discarded for inference"}
    write_json(checkpoint_dir / "config.json", config)
    result = {"run_id": run_id, "kind": "native_optimization_training", "status": "development_complete", "configuration": config,
              "provenance": source, "dataset_audit": audit, "parameters": sum(p.numel() for p in model.parameters()),
              "temporary_auxiliary_parameters": sum(p.numel() for p in heads.parameters()) if curriculum else 0,
              "checkpoint": str((checkpoint_dir / "model.safetensors").relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint_dir / "model.safetensors"),
              "final_checkpoint_sha256": sha256(checkpoint_dir / "final.safetensors"), "history": history,
              "development": selected_metrics, "final_development": final_metrics, "selection_key": list(best_key),
              "primitive_diagnostics": diagnostics, "train_seconds": spent, "total_wall_seconds": time.perf_counter() - start,
              "test_evaluated": False, "matched_step_budget_completed": steps == budget["optimizer_steps"]}
    write_json(report_dir / "training.json", result)
    return result


def nominate_optimization():
    if NOMINATION.exists(): raise ValueError("Nomination is immutable")
    protocol = json.loads(PROTOCOL.read_text()); rows = []
    for run_id in protocol["conditions"]:
        report = json.loads((ROOT / "reports" / run_id / "training.json").read_text())
        rows.append({"run_id": run_id, "selection_key": report["selection_key"], "checkpoint_sha256": report["checkpoint_sha256"],
                     "matched_step_budget_completed": report["matched_step_budget_completed"], **protocol["conditions"][run_id]})
    eligible = [r for r in rows if r["matched_step_budget_completed"] and r["recipe"] == "primitive_supervision" and r["selection_key"][0] == 1]
    nominee = max(eligible, key=lambda r: tuple(r["selection_key"]))["run_id"] if eligible else None
    result = {"protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(MANIFEST), "conditions": rows,
              "candidate_run": nominee, "test_opened": False, "serving_default_changed": False}
    write_json(NOMINATION, result); return result


def evaluate_optimization(run_id, device="mps"):
    source = snapshot(); nomination = json.loads(NOMINATION.read_text())
    if nomination["manifest_sha256"] != sha256(MANIFEST) or nomination["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("Frozen nomination lineage changed")
    blob = subprocess.check_output(["git", "show", f"{source['git_revision']}:{NOMINATION.relative_to(ROOT)}"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != sha256(NOMINATION): raise ValueError("Commit nomination before test inference")
    reference = run_id == "joint-full-v2"
    directory = ROOT / "reports" / ("optimization-served-reference-v1" if reference else run_id)
    if (directory / "evaluation.json").exists(): raise ValueError("Confirmation already opened")
    training = json.loads((ROOT / "reports" / run_id / "training.json").read_text())
    config = training["configuration"]; checkpoint = ROOT / training["checkpoint"]
    expected = training if reference else next(r for r in nomination["conditions"] if r["run_id"] == run_id)
    if sha256(checkpoint) != expected["checkpoint_sha256"]: raise ValueError("Nominated weights changed")
    if not reference and training["provenance"]["manifest_sha256"] != sha256(MANIFEST): raise ValueError("Training data lineage changed")
    torch.set_num_threads(4); chosen = choose_device(device)
    scenes = read_manifest(MANIFEST); audit = audit_optimization(scenes)
    vocabulary = Vocabulary(config["vocabulary"])
    model = NativeDecisionModel(len(vocabulary.tokens), config["width"], config["layers"]).to(chosen)
    model.load_state_dict(load_file(str(checkpoint)))
    if reference:
        temperature = json.loads(checkpoint.with_name("config.json").read_text())["temperature"]
    else:
        calibration = PairedCache(scenes, vocabulary, "calibration")
        temperature = fit_joint_temperature(calibration.records, evaluate_logits(model, calibration, chosen))
    test = PairedCache(scenes, vocabulary, "test")
    logits = evaluate_logits(model, test, chosen)
    raw, _ = decision_metrics(test.records, logits)
    calibrated, predictions = decision_metrics(test.records, logits, temperature)
    write_manifest(directory / "predictions.jsonl", predictions)
    result = {"run_id": run_id, "kind": "native_optimization_confirmation", "status": "confirmation_complete", "provenance": source,
              "checkpoint_sha256": sha256(checkpoint), "temperature": temperature, "dataset_audit": audit,
              "raw": raw, "calibrated": calibrated, "frozen_served_reference": reference,
              "scope": "Fresh human speakers and generated panels for the previously exposed composition gap"}
    if not reference: write_json(checkpoint.with_name("config.json"), {**config, "temperature": temperature})
    write_json(directory / "evaluation.json", result); return result
