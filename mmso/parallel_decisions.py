"""Independent causal question branches sharing one multimodal prefix.

This preserves the original per-question causal computation graph in real
arithmetic. GPU parity and timings must be checked; it is not a quality claim.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from .backbone_adapters import load_and_merge_adapter
from .backbone_diagnostics import trim_head
from .backbone_study import Backbone, decode_media, move, prompt_for


def common_prefix(sequences):
    minimum = min(len(s) for s in sequences)
    stacked = torch.stack([s[:minimum].cpu() for s in sequences])
    different = (~(stacked == stacked[0]).all(0)).nonzero()
    return int(different[0]) if len(different) else minimum


def branch_plan(sequences, prefix_length, *, device="cpu", dtype=torch.float32):
    if not 0 <= prefix_length < min(len(s) for s in sequences):
        raise ValueError("Every question needs a nonempty suffix")
    prefix = sequences[0][:prefix_length]
    if any(not torch.equal(s[:prefix_length].cpu(), prefix.cpu()) for s in sequences):
        raise ValueError("Question prefixes differ")
    suffixes = [s[prefix_length:] for s in sequences]
    ids = torch.cat([prefix, *suffixes]).to(device)[None]
    positions = torch.cat([torch.arange(prefix_length), *[torch.arange(prefix_length, len(s)) for s in sequences]]).to(device)[None]
    length = ids.shape[-1]
    mask = torch.full((length, length), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask[:prefix_length, :prefix_length] = torch.triu(torch.full((prefix_length, prefix_length), torch.finfo(dtype).min, device=device, dtype=dtype), diagonal=1)
    cursor, ends, spans = prefix_length, [], []
    for suffix in suffixes:
        end = cursor + len(suffix)
        mask[cursor:end, :prefix_length] = 0
        mask[cursor:end, cursor:end] = torch.triu(torch.full((len(suffix), len(suffix)), torch.finfo(dtype).min, device=device, dtype=dtype), diagonal=1)
        spans.append((cursor, end)); ends.append(end - 1); cursor = end
    return {"input_ids": ids, "position_ids": positions, "attention_mask": mask[None, None],
            "end_positions": torch.tensor(ends, device=device), "spans": spans}


def diagnostic_questions():
    colors = ["red", "blue", "green", "yellow"]
    words = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
    questions = [
        ("Which colored tile contains the spoken word?", [*colors, "not present"]),
        ("Which tile position contains the spoken word?", ["top left", "top right", "bottom left", "bottom right", "not present"]),
        ("Is the spoken word visible anywhere in the image?", ["yes", "no"]),
        ("What exact word is spoken in the recording?", words),
    ]
    for position in ("top left", "top right", "bottom left", "bottom right"):
        questions.append((f"What color is the {position} tile?", colors))
        questions.append((f"What word is printed in the {position} tile?", words))
    for word in ("left", "right", "up", "down"):
        questions.append((f"Is the word {word} printed anywhere in the image?", ["yes", "no"]))
    return [{"question": q, "choices": c} for q, c in questions]


def run_parallel_probe(root, key, output):
    if key != "minicpmo45": raise ValueError("This diagnostic targets the nominated MiniCPM candidate")
    torch.set_num_threads(4)
    registry = json.loads((root / "evals/v4-candidates-v1.json").read_text())
    spec = next(m for m in registry["models"] if m["key"] == key)
    protocol = json.loads((root / "evals/v4-selection-protocol-v1.json").read_text())
    rows = [json.loads(s) for s in (root / "evals/manifests/v4_selection_v1.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["split"] == "development" and r["track"] == "joint_control")
    backbone = Backbone(spec)
    load_and_merge_adapter(backbone.model, root / "artifacts/v4-adapters" / key / "adapter", spec)
    backbone.load_readout(root / "artifacts/v4-selection" / (key + "-adapted"))
    if backbone.readout is not None: raise ValueError("Packed diagnostic requires the selected label-logit readout")
    trim_head(backbone)
    images, audios = decode_media(root, row["media"], protocol["input_policy"])
    queries = diagnostic_questions()
    prepared = []
    for query in queries:
        data, _ = backbone.prepare(prompt_for(query), images, audios)
        prepared.append(move(data, "cuda"))
    ids = [data["input_ids"][0] for data in prepared]
    prefix_length = common_prefix(ids)
    if prefix_length <= 0: raise ValueError("Missing shared state prefix")
    with torch.inference_mode():
        actual_embeddings = []
        for data in prepared:
            value, _ = backbone.model.get_vllm_embedding(data)
            value = backbone.model.get_omni_embedding(data, input_embeddings=value, chunk_length=backbone.model.config.audio_chunk_length)
            actual_embeddings.append(value)
        prefix = actual_embeddings[0][:, :prefix_length]
        if any(not torch.equal(prefix, e[:, :prefix_length]) for e in actual_embeddings):
            raise ValueError("Media-prefix representations differ across questions")
        # These are the exact tensors used by independent original forwards.
        # Reusing them makes the baseline stronger by avoiding media re-encoding.
        def sequential(count):
            result = []
            for i in range(count):
                outputs = backbone.model.llm(inputs_embeds=actual_embeddings[i], position_ids=prepared[i]["position_ids"],
                           use_cache=False, return_dict=True, logits_to_keep=1)
                result.append((outputs.logits[0, -1, :len(queries[i]["choices"])].float() / backbone.temperature).softmax(-1).cpu().numpy())
            return result

        def packed(count, ordering=None):
            order = list(range(count)) if ordering is None else ordering
            plan = branch_plan([ids[i] for i in order], prefix_length, device="cuda", dtype=torch.bfloat16)
            if plan["input_ids"].shape[-1] > protocol["input_policy"]["max_input_tokens"]:
                raise ValueError("Packed request exceeds the token budget")
            embeddings = torch.cat([prefix, *[actual_embeddings[i][:, prefix_length:] for i in order]], dim=1)
            outputs = backbone.model.llm(inputs_embeds=embeddings, position_ids=plan["position_ids"],
                      attention_mask=plan["attention_mask"], use_cache=False, return_dict=True,
                      logits_to_keep=plan["end_positions"])
            results = [(outputs.logits[0, j, :len(queries[i]["choices"])].float() / backbone.temperature).softmax(-1).cpu().numpy() for j, i in enumerate(order)]
            return results, int(plan["input_ids"].shape[-1])

        records = []
        for count in (1, 4, 16):
            reference = sequential(count)
            values, total = packed(count)
            delta = max(float(np.max(np.abs(a - b))) for a, b in zip(reference, values))
            agrees = sum(int(a.argmax()) == int(b.argmax()) for a, b in zip(reference, values))
            sequential_ms, packed_ms = [], []
            for _ in range(5):
                torch.cuda.synchronize(); start = time.perf_counter(); sequential(count); torch.cuda.synchronize()
                sequential_ms.append((time.perf_counter() - start) * 1000)
                torch.cuda.synchronize(); start = time.perf_counter(); packed(count); torch.cuda.synchronize()
                packed_ms.append((time.perf_counter() - start) * 1000)
            record = {"questions": count, "shared_prefix_tokens": prefix_length, "packed_tokens": total,
                      "separate_tokens": sum(len(ids[i]) for i in range(count)),
                      "argmax_agreement": agrees, "maximum_probability_difference": delta,
                      "sequential_shared_media_ms": sequential_ms, "packed_ms": packed_ms,
                      "median_speedup": float(np.median(sequential_ms) / np.median(packed_ms)),
                      "reference_probabilities": [p.tolist() for p in reference], "packed_probabilities": [p.tolist() for p in values]}
            records.append(record)
            (output / "parallel.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({"questions": count, "speedup": record["median_speedup"], "argmax_agreement": agrees, "maximum_probability_difference": delta}), flush=True)
        order = list(reversed(range(16)))
        reversed_values, _ = packed(16, order)
        canonical = records[-1]["packed_probabilities"]
        reorder_delta = max(float(np.max(np.abs(np.asarray(canonical[i]) - reversed_values[j]))) for j, i in enumerate(order))
    report = {"status": "completed", "key": key, "phase": "parallel-probe", "gpu": torch.cuda.get_device_name(),
              "source_case": row["id"], "queries": queries, "measurements": records,
              "question_reorder_max_probability_difference": reorder_delta,
              "encoder_reuse": "Both timed paths reuse already computed image/audio embeddings. Packed path additionally shares language-prefix computation.",
              "timing_boundary": "Prepared GPU embeddings through language backbone, candidate scores and CPU probability transfer. Packed mask construction is included; file decode, processing and media encoders are excluded from BOTH paths.",
              "new_quality_result": False, "scope": "One fixed multimodal observation, sixteen distinct questions. Numerical agreement is a diagnostic, not a new task-accuracy benchmark."}
    (output / "parallel-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
