"""Train-only readout fitting, separate calibration, and quality-gated selection."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=-1, keepdims=True)


def summarize(rows, predictions, probabilities):
    by_id = {r["id"]: r for r in predictions}
    slices = {}
    for track in sorted({r["track"] for r in rows}):
        selected = [r for r in rows if r["track"] == track]
        correct, nll, brier, uniform_nll, uniform_brier, times, confidences = [], [], [], [], [], [], []
        errors = 0
        for row in selected:
            p = probabilities.get(row["id"])
            record = by_id.get(row["id"], {})
            if p is None or record.get("status") != "ok":
                errors += 1; correct.append(False); continue
            p = np.asarray(p, dtype=float)
            if len(p) != len(row["choices"]) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-6:
                raise ValueError("Invalid probability vector")
            y = row["target"]
            hit = int(p.argmax()) == y
            correct.append(hit); confidences.append((float(p.max()), hit))
            nll.append(-np.log(max(p[y], 1e-12)))
            target = np.zeros(len(p)); target[y] = 1
            brier.append(float(np.square(p - target).sum()))
            uniform_nll.append(np.log(len(p))); uniform_brier.append(1 - 1 / len(p))
            times.append(record["timings"]["request_ms"])
        ece = 0.0
        for lower in np.linspace(0, .9, 10):
            bucket = [(c, h) for c, h in confidences if lower <= c <= 1 if c < lower + .1 or lower >= .9]
            if bucket:
                ece += len(bucket) / max(1, len(confidences)) * abs(np.mean([x[0] for x in bucket]) - np.mean([x[1] for x in bucket]))
        slices[track] = {"examples": len(selected), "errors": errors, "correct": int(sum(correct)),
                         "accuracy": float(np.mean(correct)), "nll": float(np.mean(nll)) if nll else None,
                         "brier": float(np.mean(brier)) if brier else None,
                         "uniform_nll": float(np.mean(uniform_nll)) if nll else None,
                         "uniform_brier": float(np.mean(uniform_brier)) if brier else None,
                         "ece_10_bins": float(ece), "p50_ms": float(np.median(times)) if times else None,
                         "p95_ms": float(np.quantile(times, .95)) if times else None}
    valid = [r for r in slices.values() if r["nll"] is not None]
    return {"tracks": slices, "macro_accuracy": float(np.mean([r["accuracy"] for r in slices.values()])),
            "macro_nll": float(np.mean([r["nll"] for r in valid])) if valid else None,
            "macro_brier": float(np.mean([r["brier"] for r in valid])) if valid else None,
            "macro_track_p95_ms": float(np.mean([r["p95_ms"] for r in valid])) if valid else None,
            "errors": sum(r["errors"] for r in slices.values()),
            "proper_scores_conditioned_on_success": True}


def quality_reasons(summary, protocol):
    reasons = []
    expected = set(protocol["tracks"])
    if set(summary["tracks"]) != expected:
        reasons.append("missing_track")
    if summary["errors"]:
        reasons.append("incomplete_coverage")
    if summary["macro_accuracy"] < protocol["selection"]["macro_accuracy_floor"]:
        reasons.append("macro_accuracy_below_floor")
    for name, settings in protocol["tracks"].items():
        row = summary["tracks"].get(name)
        if row is None: continue
        if row["accuracy"] < settings["accuracy_floor"]:
            reasons.append(name + ":accuracy_below_floor")
        if row["nll"] is None or row["nll"] >= row["uniform_nll"]:
            reasons.append(name + ":nll_not_better_than_uniform")
        if row["brier"] is None or row["brier"] >= row["uniform_brier"]:
            reasons.append(name + ":brier_not_better_than_uniform")
    return reasons


def choose_winner(candidates, protocol):
    eligible = [c for c in candidates if not c["disqualification_reasons"]]
    if not eligible:
        return {"status": "no_qualified_winner", "winner": None}
    best = max(c["development"]["macro_accuracy"] for c in eligible)
    margin = protocol["selection"]["maximum_accuracy_gap_to_best_eligible"]
    close = [c for c in eligible if c["development"]["macro_accuracy"] >= best - margin]
    winner = min(close, key=lambda c: (c["development"]["macro_track_p95_ms"], -c["development"]["macro_accuracy"], c["key"]))
    reference = max(eligible, key=lambda c: (c["development"]["macro_accuracy"], -c["development"]["macro_nll"]))
    return {"status": "development_nomination", "winner": winner["key"], "quality_reference": reference["key"],
            "best_eligible_accuracy": best, "within_margin": [c["key"] for c in close],
            "statistical_noninferiority_established": False}


def fit_candidate(root, attempt):
    torch.set_num_threads(4)
    torch.manual_seed(20260924)
    folder = root / "reports/v4-selection-v1/attempts" / attempt
    result = json.loads((folder / "result.json").read_text())
    key = result["key"]
    if result["phase"] not in {"development", "adapter-development"} or result["status"] != "completed":
        raise ValueError("Readout fitting requires a complete development extraction")
    protocol = json.loads((root / "evals/v4-selection-protocol-v1.json").read_text())
    all_rows = [json.loads(s) for s in (root / "evals/manifests/v4_selection_v1.jsonl").read_text().splitlines()]
    rows = [r for r in all_rows if r["split"] != "confirmation"]
    predictions = [json.loads(s) for s in (folder / "predictions.jsonl").read_text().splitlines()]
    pred_by_id = {r["id"]: r for r in predictions}
    archive_path = root / ".research/v4/features" / (attempt + ".npz")
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != result["features_sha256"]:
        raise ValueError("Feature archive hash mismatch")
    archive = np.load(archive_path, allow_pickle=False)
    feature_by_id = dict(zip(archive["ids"].tolist(), archive["features"]))
    good = [r for r in rows if r["id"] in feature_by_id and pred_by_id[r["id"]]["status"] == "ok"]
    if not good:
        raise ValueError("No valid model features")
    x = torch.tensor(np.stack([feature_by_id[r["id"]] for r in good]), dtype=torch.float32)
    targets = torch.tensor([r["target"] for r in good], dtype=torch.long)
    mask = torch.arange(10)[None] < torch.tensor([len(r["choices"]) for r in good])[:, None]
    base = torch.full((len(good), 10), -1e4)
    for i, row in enumerate(good):
        base[i, :len(row["choices"])] = torch.tensor(pred_by_id[row["id"]]["logits"])
    train = torch.tensor([r["split"] == "train" for r in good])
    if not train.any(): raise ValueError("No training features")
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(.01)
    normalized = (x - mean) / std
    train_weights = torch.tensor([1 / sum(s["track"] == r["track"] and s["split"] == "train" for s in good) for r in good if r["split"] == "train"])
    train_weights /= train_weights.sum()
    variants = [("label_logits", None, base.clone())]
    for l2 in protocol["methods"]["readout_l2"]:
        layer = nn.Linear(x.shape[-1], 10)
        nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=protocol["methods"]["readout_learning_rate"], weight_decay=0)
        for _ in range(protocol["methods"]["readout_epochs"]):
            delta = layer(normalized[train])
            logits = (base[train] + delta).masked_fill(~mask[train], -1e4)
            loss = (nn.functional.cross_entropy(logits, targets[train], reduction="none") * train_weights).sum()
            loss += l2 * (delta[mask[train]] ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            logits = (base + layer(normalized)).masked_fill(~mask, -1e4)
        state = {"weight": layer.weight.detach().numpy(), "bias": layer.bias.detach().numpy(),
                 "mean": mean.numpy(), "std": std.numpy()}
        variants.append((f"readout-l2-{l2}", state, logits))
    summaries, states, probabilities_by_variant = [], {}, {}
    for name, state, logits in variants:
        values = logits.detach().numpy()
        calibration_rows = [r for r in rows if r["split"] == "calibration"]
        choices = []
        for temperature in protocol["methods"]["calibration_temperatures"]:
            probs = {r["id"]: softmax(values[i, :len(r["choices"])] / temperature).tolist() for i, r in enumerate(good)}
            scores = summarize(calibration_rows, predictions, probs)
            if scores["macro_nll"] is None: continue
            choices.append((scores["macro_nll"], temperature, probs))
        if not choices: raise ValueError("No calibration predictions")
        _, temperature, probabilities = min(choices, key=lambda r: (r[0], r[1]))
        dev = summarize([r for r in rows if r["split"] == "development"], predictions, probabilities)
        summary = {"method": name, "temperature": temperature, "development": dev,
                   "quality_reasons": quality_reasons(dev, protocol)}
        summaries.append(summary); states[name] = state; probabilities_by_variant[name] = probabilities
    qualifying = [s for s in summaries if not s["quality_reasons"]]
    selected = max(qualifying or summaries, key=lambda s: (s["development"]["macro_accuracy"], -s["development"]["macro_nll"], s["method"] == "label_logits"))
    registry = json.loads((root / "evals/v4-candidates-v1.json").read_text())
    spec = next(m for m in registry["models"] if m["key"] == key)
    reasons = selected["quality_reasons"] + ([] if spec["eligible_for_product"] else ["license_or_access_not_product_eligible"])
    adapted = result["phase"] == "adapter-development"
    report = {"key": key + ("-adapted" if adapted else ""), "base_key": key, "attempt": attempt, "model": spec["id"], "revision": spec["revision"],
              "selected_method": selected["method"], "temperature": selected["temperature"],
              "development": selected["development"], "disqualification_reasons": reasons,
              "variants": summaries, "feature_sha256": result["features_sha256"],
              "readout_training_scope": "Frozen backbone, training split only. This does not establish full adapter-training performance.",
              "latency_scope": "Measured label-logit pipeline; trained-readout incremental scoring overhead must be measured for finalists."}
    destination = root / "reports/v4-selection-v1/candidates" / report["key"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    chosen_probs = probabilities_by_variant[selected["method"]]
    (destination / "development-predictions.jsonl").write_text("".join(json.dumps({"id": r["id"], "probabilities": chosen_probs.get(r["id"])}) + "\n" for r in rows if r["split"] == "development"))
    artifact = root / "artifacts/v4-selection" / report["key"]
    artifact.mkdir(parents=True, exist_ok=True)
    state = states[selected["method"]]
    if state is not None:
        from safetensors.numpy import save_file
        save_file(state, str(artifact / "readout.safetensors"))
    config = {k: report[k] for k in ["key", "model", "revision", "selected_method", "temperature", "feature_sha256"]}
    config["readout_sha256"] = hashlib.sha256((artifact / "readout.safetensors").read_bytes()).hexdigest() if state is not None else None
    (artifact / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return report
