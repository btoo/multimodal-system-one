"""Bounded portability/resume smoke; never a model-selection or accuracy trial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import torch
from safetensors.torch import load_file

from .artifacts import ROOT, local_media_path, read_manifest, sha256, write_json
from .audio import choose_device, synchronize
from .joint_model import NativeDecisionModel, Vocabulary
from .joint_training import PairedCache

CHECKPOINT = "artifacts/optimization-primitive-s24-v1/model.safetensors"
SEED = 20260924
STEPS = 32
SPLIT_STEP = 16
BATCH_SIZE = 16


def configure(device):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def verified_bundle():
    bundle = json.loads((ROOT / "bundle.json").read_text())
    for relative, expected in bundle["files"].items():
        if sha256(local_media_path(ROOT, relative)) != expected:
            raise ValueError(f"Cloud bundle hash mismatch: {relative}")
    scenes = read_manifest(ROOT / "pilot-scenes.jsonl")
    if len(scenes) != 64 or any(s["split"] != "train" for s in scenes):
        raise ValueError("This smoke accepts exactly 64 training scenes")
    config = json.loads((ROOT / CHECKPOINT).with_name("config.json").read_text())
    return bundle, scenes, config


def new_model(config, device):
    model = NativeDecisionModel(len(config["vocabulary"]), config["width"], config["layers"])
    model.load_state_dict(load_file(str(ROOT / CHECKPOINT)))
    return model.to(device)


class DivisibleAveragePool(torch.nn.Module):
    """Equivalent nonoverlapping pooling for this model's fixed input shapes.

    CUDA adaptive pooling backward rejects strict determinism. Ordinary average
    pooling has a deterministic backward; only use it for divisible dimensions.
    This module has no parameters or buffers, so checkpoint keys are unchanged.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def forward(self, x):
        height, width = x.shape[-2:]
        out_height, out_width = self.output_size
        if height % out_height or width % out_width:
            raise ValueError("Deterministic pilot pooling requires divisible spatial sizes")
        return torch.nn.functional.avg_pool2d(x, (height // out_height, width // out_width))


def deterministic_pooling(model):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.AdaptiveAvgPool2d):
                setattr(module, name, DivisibleAveragePool(child.output_size))
    return model


def percentile(values):
    return {"samples": len(values), "p50_ms": float(np.median(values) * 1000),
            "p95_ms": float(np.quantile(values, .95) * 1000)}


@torch.inference_mode()
def inference_check(config, data, device):
    """Compare the same loaded checkpoint, then time warm batch-1 and batch-32."""
    model = new_model(config, torch.device("cpu")).eval()
    indices = torch.arange(64)
    inputs, _ = data.batch(indices, torch.device("cpu"))
    reference = (model(*inputs) / config["temperature"]).softmax(-1)
    deterministic_pooling(model).to(device)
    observed = (model(*[x.to(device) for x in inputs]) / config["temperature"]).softmax(-1).cpu()
    difference = float((reference - observed).abs().max())
    agreement = int((reference.argmax(-1) == observed.argmax(-1)).sum())
    if difference > 2e-4 or agreement != len(indices):
        raise ValueError(f"CPU/CUDA inference mismatch: {difference}, {agreement}/64")
    timings = {}
    for batch_size in (1, 32):
        elapsed = []
        # Prepared CPU tensors; include indexing, transfer, model, calibration,
        # output transfer and synchronization. Excludes WAV/PNG decoding and RPC.
        for iteration in range(5 + 32):
            batch_indices = (torch.arange(batch_size) + iteration * batch_size) % len(data.records)
            synchronize(device); start = time.perf_counter()
            batch, _ = data.batch(batch_indices, device)
            (model(*batch) / config["temperature"]).softmax(-1).cpu()
            synchronize(device)
            if iteration >= 5:
                elapsed.append(time.perf_counter() - start)
        timings[str(batch_size)] = {**percentile(elapsed), "batch_size": batch_size,
                                   "decisions_per_second": batch_size / float(np.mean(elapsed))}
    return {"compared_requests": len(indices), "top1_agreement": agreement,
            "maximum_probability_difference": difference, "warm_preprocessed_batches": timings,
            "cpu_probabilities": reference.tolist(), "cuda_probabilities": observed.tolist(),
            "timing_excludes": ["media decoding", "model loading", "container startup", "network/RPC"],
            "accuracy_claim": False, "data_role": "previously used training examples"}


def save_training(path, model, optimizer, generator, step, losses, device):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "generator": generator.get_state(), "cpu_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
               "step": step, "losses": losses}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_training(path, model, optimizer, generator, device):
    # Only this project's own checkpoint is loaded, with safe tensor-only loading.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["generator"])
    torch.set_rng_state(payload["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["step"], list(payload["losses"])


def tensor_tree_difference(first, second):
    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor) or first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError("Checkpoint tensor schema changed")
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("Nonfinite checkpoint tensor")
        return float((first.double() - second.double()).abs().max()) if first.numel() else 0.
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError("Checkpoint keys changed")
        return max((tensor_tree_difference(first[k], second[k]) for k in first), default=0.)
    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise ValueError("Checkpoint sequence changed")
        return max((tensor_tree_difference(a, b) for a, b in zip(first, second)), default=0.)
    if first != second:
        raise ValueError("Checkpoint scalar changed")
    return 0.


def train_segment(model, optimizer, generator, data, device, start_step, losses, output):
    model.train(); elapsed = []; gradient_norms = {}
    # Math SDPA makes the controlled resume comparison deterministic. This is
    # deliberately separate from an optimized throughput benchmark.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for step in range(start_step, STEPS):
            indices = torch.randint(len(data.records), (BATCH_SIZE,), generator=generator)
            inputs, targets = data.batch(indices, device)
            synchronize(device); start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(*inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite cloud training loss")
            loss.backward()
            if step == start_step:
                gradient_norms = {name: float(sum(p.grad.detach().square().sum() for p in module.parameters()
                                                 if p.grad is not None).sqrt().cpu())
                                  for name, module in {"image": model.image, "audio": model.audio,
                                                       "text": model.text}.items()}
                if not all(np.isfinite(v) and v > 0 for v in gradient_norms.values()):
                    raise ValueError("Training must reach every modality")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); synchronize(device)
            elapsed.append(time.perf_counter() - start)
            losses.append(float(loss.detach().cpu()))
            if step + 1 == SPLIT_STEP:
                save_training(output / "midpoint.pt", model, optimizer, generator, step + 1, losses, device)
    return {"updates": len(elapsed), "step_timing": percentile(elapsed),
            "training_seconds": sum(elapsed), "first_step_gradient_l2": gradient_norms,
            "losses": losses}


def run_phase(phase, output):
    if phase not in {"reference", "resume"}:
        raise ValueError("Unknown pilot phase")
    started = time.perf_counter(); device = choose_device("cuda"); configure(device)
    bundle, scenes, config = verified_bundle()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    completed = output / f"{phase}.json"
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous["bundle_sha256"] != sha256(ROOT / "bundle.json"):
            raise ValueError("Run ID reused with a different bundle")
        return previous
    data = PairedCache(scenes, Vocabulary(config["vocabulary"]), "train")
    result = {"phase": phase, "process_id": uuid.uuid4().hex, "source_commit": bundle["source_commit"],
              "bundle_sha256": sha256(ROOT / "bundle.json"), "checkpoint_sha256": sha256(ROOT / CHECKPOINT),
              "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "python": platform.python_version(),
              "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
              "training_scenes": len(scenes), "training_questions": len(data.records),
              "batch_size": BATCH_SIZE, "purpose": "portability and resume smoke only"}
    if phase == "reference":
        if (output / "midpoint.pt").exists():
            raise ValueError("Partial run exists; retain it and use a fresh run ID")
        result["inference"] = inference_check(config, data, device)
    model = deterministic_pooling(new_model(config, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    generator = torch.Generator().manual_seed(SEED)
    step, losses = (restore_training(output / "midpoint.pt", model, optimizer, generator, device)
                    if phase == "resume" else (0, []))
    if phase == "resume" and step != SPLIT_STEP:
        raise ValueError("Wrong resume step")
    result["training"] = train_segment(model, optimizer, generator, data, device, step, losses, output)
    final_path = output / f"{phase}.pt"
    save_training(final_path, model, optimizer, generator, STEPS, losses, device)
    result["checkpoint_files"] = {p.name: sha256(p) for p in (output / "midpoint.pt", final_path)}
    if phase == "resume":
        reference = torch.load(output / "reference.pt", map_location="cpu", weights_only=True)
        actual = torch.load(final_path, map_location="cpu", weights_only=True)
        differences = {k: tensor_tree_difference(reference[k], actual[k])
                       for k in ("model", "optimizer", "generator", "cpu_rng", "cuda_rng")}
        loss_difference = float(np.max(np.abs(np.array(reference["losses"]) - actual["losses"])))
        result["resume"] = {"from_step": step, "to_step": STEPS,
                            "maximum_differences": differences, "maximum_loss_difference": loss_difference,
                            "exact_match": max(*differences.values(), loss_difference) == 0.}
        if max(*differences.values(), loss_difference) > 1e-6:
            raise ValueError(f"Resume did not reproduce reference: {result['resume']}")
        reference_result = json.loads((output / "reference.json").read_text())
        if reference_result["process_id"] == result["process_id"]:
            raise ValueError("Resume must execute in another invocation")
    result["phase_wall_seconds"] = time.perf_counter() - started
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    write_json(completed, result)
    return result
