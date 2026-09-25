"""Small audited LoRA intervention without changing publisher base checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file


class DecisionLoRA(nn.Module):
    def __init__(self, frozen: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.frozen = frozen
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, frozen.in_features, device=frozen.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(frozen.out_features, rank, device=frozen.weight.device, dtype=torch.float32))
        nn.init.normal_(self.a, std=.01)
        for p in frozen.parameters(): p.requires_grad_(False)

    @property
    def weight(self): return self.frozen.weight

    @property
    def bias(self): return self.frozen.bias

    def forward(self, x):
        original = self.frozen(x)
        update = nn.functional.linear(nn.functional.linear(x.float(), self.a), self.b) * self.scale
        return original + update.to(original.dtype)

    @torch.no_grad()
    def merge(self):
        update = (self.b @ self.a) * self.scale
        self.frozen.weight.add_(update.to(self.frozen.weight.dtype))
        return self.frozen


def language_projection_names(model):
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or ".layers." not in name:
            continue
        if any(part in name.lower() for part in ("vision", "audio", "vpm", "apm")):
            continue
        stem = name.removesuffix(".base_layer")
        if stem.endswith((".q_proj", ".v_proj", ".qkv_proj")):
            names.append(name)
    if not names: raise ValueError("No supported language attention projections")
    return names


def replace(model, name, module):
    parent, attribute = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), attribute, module)


def train_adapter(backbone, root, rows, protocol, output):
    from .backbone_study import decode_media, move, prompt_for
    settings = protocol["training"]
    source = json.loads((root / "evals/v4-selection-protocol-v1.json").read_text())
    policy = source["input_policy"]
    torch.manual_seed(settings["seed"])
    torch.set_float32_matmul_precision("highest")
    model = backbone.model
    names = language_projection_names(model)
    wrappers = {}
    for name in names:
        layer = DecisionLoRA(model.get_submodule(name), settings["rank"], settings["alpha"])
        replace(model, name, layer); wrappers[name] = layer
    parameters = [p for layer in wrappers.values() for p in (layer.a, layer.b)]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    model.train()
    checkpoint_model = getattr(model, "llm", model)
    checkpoint_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Frozen media branches keep evaluation behavior throughout training.
    for name in backbone.encoder_names: model.get_submodule(name).eval()
    for name, module in model.named_modules():
        if any(term in name.lower() for term in ("vision", "audio", "vpm", "apm", "lora_dropout")):
            module.eval()
    tracks = sorted(source["tracks"])
    pools = {t: [r for r in rows if r["split"] == "train" and r["track"] == t] for t in tracks}
    randomizers = {t: random.Random(f"{settings['seed']}:{t}") for t in tracks}
    queues = {t: [] for t in tracks}
    records, gradient_observed = [], False
    started = time.perf_counter()
    for step in range(settings["steps"]):
        track = tracks[step % len(tracks)]
        if not queues[track]:
            queues[track] = pools[track].copy(); randomizers[track].shuffle(queues[track])
        row = queues[track].pop()
        safe = {k: row[k] for k in ("question", "choices", "media")}
        images, audios = decode_media(root, safe["media"], policy)
        inputs, _ = backbone.prepare(prompt_for(safe), images, audios)
        if inputs["input_ids"].shape[-1] > policy["max_input_tokens"]:
            raise ValueError("Training input exceeds frozen token ceiling")
        inputs = move(inputs, "cuda")
        optimizer.zero_grad(set_to_none=True)
        backbone.events = []
        result = backbone.forward(inputs)
        logits = result.logits[:, -1, backbone.token_ids[:len(row["choices"])]]
        target = torch.tensor([row["target"]], device="cuda", dtype=torch.long)
        loss = nn.functional.cross_entropy(logits.float(), target)
        if not torch.isfinite(loss): raise ValueError("Nonfinite adapter loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip_norm"])
        if not torch.isfinite(norm): raise ValueError("Nonfinite adapter gradient")
        gradient_observed |= float(norm) > 0
        optimizer.step()
        record = {"step": step + 1, "id": row["id"], "track": track, "split": row["split"],
                  "loss": float(loss.detach()), "gradient_norm": float(norm)}
        records.append(record)
        with (output / "training.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
        if (step + 1) % 32 == 0:
            print(json.dumps({"key": backbone.key, "training_step": step + 1, "steps": settings["steps"],
                              "recent_mean_loss": float(np.mean([r["loss"] for r in records[-32:]]))}), flush=True)
        if time.perf_counter() - started > 2200:
            raise TimeoutError("Adapter-training time bound reached; no partial adapter may qualify")
    if not gradient_observed: raise ValueError("No adapter gradient observed")
    destination = output / "adapter"
    destination.mkdir()
    state = {name + "." + attr: getattr(layer, attr).detach().cpu().contiguous() for name, layer in wrappers.items() for attr in ("a", "b")}
    save_file(state, str(destination / "adapter.safetensors"))
    metadata = {"base_model": backbone.spec["id"], "revision": backbone.spec["revision"], "modules": names,
                "rank": settings["rank"], "alpha": settings["alpha"], "steps": settings["steps"],
                "trainable_parameters": sum(p.numel() for p in parameters), "gradient_observed": gradient_observed,
                "protocol_sha256": hashlib.sha256((root / "evals/v4-adapter-protocol-v1.json").read_bytes()).hexdigest(),
                "training_seconds": time.perf_counter() - started, "research_only": True,
                "adapter_sha256": hashlib.sha256((destination / "adapter.safetensors").read_bytes()).hexdigest()}
    (destination / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, wrapper in wrappers.items(): replace(model, name, wrapper.merge())
    optimizer.zero_grad(set_to_none=True)
    del optimizer, wrappers, parameters
    checkpoint_model.gradient_checkpointing_disable()
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    torch.cuda.empty_cache()
    return metadata


def load_and_merge_adapter(model, folder, spec):
    config = json.loads((folder / "config.json").read_text())
    if config["base_model"] != spec["id"] or config["revision"] != spec["revision"]:
        raise ValueError("Adapter base checkpoint differs")
    path = folder / "adapter.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["adapter_sha256"]:
        raise ValueError("Adapter hash mismatch")
    state = load_file(str(path), device="cuda")
    torch.set_float32_matmul_precision("highest")
    for name in config["modules"]:
        wrapper = DecisionLoRA(model.get_submodule(name), config["rank"], config["alpha"])
        with torch.no_grad():
            wrapper.a.copy_(state[name + ".a"]); wrapper.b.copy_(state[name + ".b"])
        replace(model, name, wrapper.merge())
    return config
