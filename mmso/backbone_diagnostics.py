"""Controlled generation, output-head and hardware diagnostics after selection."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .backbone_adapters import load_and_merge_adapter
from .backbone_study import Backbone, decode_media, digest, move, prompt_for


def candidate_projection(original, token_ids):
    if not isinstance(original, nn.Linear): raise TypeError("Expected a linear language output projection")
    short = nn.Linear(original.in_features, len(token_ids), bias=original.bias is not None,
                      device=original.weight.device, dtype=original.weight.dtype)
    with torch.no_grad():
        short.weight.copy_(original.weight[token_ids])
        if original.bias is not None: short.bias.copy_(original.bias[token_ids])
    short.requires_grad_(False)
    return short


def trim_head(backbone):
    original = backbone.model.get_output_embeddings()
    before = sum(p.numel() for p in backbone.model.parameters())
    shared = original.weight.data_ptr() == backbone.model.get_input_embeddings().weight.data_ptr()
    short = candidate_projection(original, backbone.token_ids)
    backbone.head_hook.remove()
    backbone.model.set_output_embeddings(short)
    backbone.head = short
    backbone.head_hook = short.register_forward_pre_hook(backbone.capture_feature)
    backbone.candidate_only = True
    if not shared: original.to("cpu")
    del original
    gc.collect(); torch.cuda.empty_cache()
    return {"parameters_before": before, "parameters_after": sum(p.numel() for p in backbone.model.parameters()),
            "input_output_weights_shared": shared, "candidate_rows": len(backbone.token_ids)}


def generated(backbone, root, row, policy, json_mode):
    start = time.perf_counter()
    images, audios = decode_media(root, row["media"], policy)
    inputs, _ = backbone.prepare(prompt_for(row, generated_json=json_mode), images, audios)
    inputs = move(inputs, "cuda")
    torch.cuda.synchronize()
    generate_start = time.perf_counter()
    with torch.inference_mode():
        if backbone.family == "minicpmo45":
            inputs.pop("position_ids", None)
            texts, output = backbone.model.generate(**inputs, tokenizer=backbone.tokenizer,
                 max_new_tokens=128 if json_mode else 1, do_sample=False, repetition_penalty=1.0, use_cache=True)
            text = texts[0]
            token_count = int(output.sequences.shape[-1])
        else:
            output = backbone.model.generate(**inputs, max_new_tokens=128 if json_mode else 1,
                 do_sample=False, repetition_penalty=1.0, use_cache=True)
            generated_ids = output[0, inputs["input_ids"].shape[-1]:]
            text = backbone.tokenizer.decode(generated_ids, skip_special_tokens=True)
            token_count = len(generated_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    valid, parsed = False, None
    if json_mode:
        try:
            parsed = json.loads(text)
            p = np.asarray(parsed["probabilities"], dtype=float)
            choice = parsed["choice"]
            valid = bool(p.shape == (len(row["choices"]),) and np.isfinite(p).all() and (p >= 0).all()
                         and abs(p.sum() - 1) < 1e-5 and choice in "ABCDEFGHIJ"[:len(p)]
                         and p[ord(choice) - 65] >= p.max() - 1e-6)
        except (ValueError, TypeError, KeyError): pass
    else:
        valid = text.strip() in list("ABCDEFGHIJ"[:len(row["choices"])])
    return {"request_ms": elapsed * 1000, "model_generation_ms": (time.perf_counter() - generate_start) * 1000,
            "tokens": token_count, "text": text, "contract_valid": valid,
            "hit_token_limit": bool(json_mode and token_count >= 128)}


def batch_inputs(backbone, root, row, policy, count):
    if backbone.family != "minicpmo45": raise ValueError("Batch diagnostic is implemented for the nominated MiniCPM adapter")
    images, audios = decode_media(root, row["media"], policy)
    _, text = backbone.prepare(prompt_for(row), images, audios)
    data = backbone.processor([text] * count, [images] * count, [audios] * count,
           [[0] * len(audios)] * count, max_slice_nums=9, return_tensors="pt", max_length=8192)
    data.pop("image_sizes", None)
    if "position_ids" not in data:
        data["position_ids"] = torch.arange(data["input_ids"].shape[-1])[None].repeat(count, 1)
    return move(data, "cuda")


def run_diagnostics(root, key, output):
    torch.set_num_threads(4)
    torch.manual_seed(20260924)
    protocol = json.loads((root / "evals/v4-selection-protocol-v1.json").read_text())
    policy = protocol["input_policy"]
    registry = json.loads((root / "evals/v4-candidates-v1.json").read_text())
    spec = next(m for m in registry["models"] if m["key"] == key)
    rows = [json.loads(s) for s in (root / "evals/manifests/v4_selection_v1.jsonl").read_text().splitlines()]
    selected = []
    for track in protocol["tracks"]:
        selected += sorted((r for r in rows if r["track"] == track and r["split"] == "development"),
                           key=lambda r: hashlib.sha256(("diag:" + r["id"]).encode()).hexdigest())[:2]
    records, batch_records = [], []
    backbone = Backbone(spec)
    gpu_name = torch.cuda.get_device_name()
    started = time.perf_counter()
    for row in selected:
        safe = {k: row[k] for k in ("question", "choices", "media")}
        # Warm the exact shape before the generation comparison.
        backbone.infer(root, safe, policy)
        direct, _ = backbone.infer(root, safe, policy)
        one = generated(backbone, root, safe, policy, False)
        vector = generated(backbone, root, safe, policy, True)
        record = {"id": row["id"], "track": row["track"], "base_direct": direct,
                  "base_generated_label": one, "base_generated_json": vector}
        records.append(record)
        with (output / "generation.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
        print(json.dumps({"diagnostic": "generation", "key": key, "completed": len(records), "json_valid": vector["contract_valid"]}), flush=True)
    load_and_merge_adapter(backbone.model, root / "artifacts/v4-adapters" / key / "adapter", spec)
    backbone.load_readout(root / "artifacts/v4-selection" / (key + "-adapted"))
    full = {}
    for row in selected:
        safe = {k: row[k] for k in ("question", "choices", "media")}
        values = [backbone.infer(root, safe, policy)[0] for _ in range(5)]
        full[row["id"]] = {"probabilities": values[-1]["probabilities"],
                           "request_ms": [v["timings"]["request_ms"] for v in values],
                           "forward_ms": [v["timings"]["forward_ms"] for v in values]}
    trimmed = trim_head(backbone)
    parity, probability_differences = [], []
    for row in selected:
        safe = {k: row[k] for k in ("question", "choices", "media")}
        values = [backbone.infer(root, safe, policy)[0] for _ in range(5)]
        differences = np.abs(np.asarray(full[row["id"]]["probabilities"]) - values[-1]["probabilities"])
        probability_differences.append(float(differences.max()))
        same = int(np.argmax(full[row["id"]]["probabilities"])) == int(np.argmax(values[-1]["probabilities"]))
        parity.append(same)
        record = {"id": row["id"], "track": row["track"], "full": full[row["id"]],
                  "trimmed": {"probabilities": values[-1]["probabilities"],
                    "request_ms": [v["timings"]["request_ms"] for v in values],
                    "forward_ms": [v["timings"]["forward_ms"] for v in values]},
                  "same_argmax": same, "maximum_probability_difference": float(differences.max())}
        with (output / "head-projection.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
    for track in ("text_rules", "joint_control"):
        row = next(r for r in selected if r["track"] == track)
        safe = {k: row[k] for k in ("question", "choices", "media")}
        for count in (1, 4, 16):
            inputs = batch_inputs(backbone, root, safe, policy, count)
            times = []
            with torch.inference_mode():
                backbone.forward(inputs); torch.cuda.synchronize()
                for _ in range(5):
                    start = time.perf_counter()
                    result = backbone.forward(inputs)
                    logits = result.logits[:, -1, :len(row["choices"])] / backbone.temperature
                    probabilities = logits.softmax(-1).cpu()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
            record = {"track": track, "batch_size": count, "forward_score_d2h_ms": times,
                      "decisions_per_second": count / (np.median(times) / 1000),
                      "identical_repeated_example": True, "all_repeated_decisions_agree": bool((probabilities.argmax(-1) == probabilities[0].argmax()).all())}
            batch_records.append(record)
            (output / "batches.json").write_text(json.dumps(batch_records, indent=2) + "\n")
    report = {"status": "completed", "key": key, "phase": "diagnostics", "hardware": gpu_name,
              "cases": len(selected), "generation_json_valid": sum(r["base_generated_json"]["contract_valid"] for r in records),
              "head_trim": trimmed, "head_argmax_agreement": sum(parity),
              "maximum_probability_difference": max(probability_differences), "batch_results": batch_records,
              "seconds_excluding_load": time.perf_counter() - started,
              "scope": "Post-selection development diagnostics, not new confirmation or production throughput",
              "source_sha256": digest(root / "mmso/backbone_diagnostics.py")}
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
