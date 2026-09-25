"""Render audited backbone-selection results and paired confirmation intervals."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from mmso.backbone_selection import summarize, quality_reasons
from mmso.parallel_decisions import branch_plan
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/v4-selection-v1"
ASSETS = ROOT / "docs/assets"
NAMES = {
    "gemma4-e2b": "Gemma 4 E2B", "gemma4-e4b": "Gemma 4 E4B", "gemma4-12b": "Gemma 4 12B",
    "qwen25-3b": "Qwen2.5 Omni 3B*", "qwen25-7b": "Qwen2.5 Omni 7B", "qwen3-30ba3b": "Qwen3 Omni 30B-A3B",
    "minicpmo45": "MiniCPM-o 4.5", "phi4mm": "Phi-4 multimodal",
    "qwen3-30ba3b-adapted": "Qwen3 Omni + MiSO adapter", "minicpmo45-adapted": "MiniCPM-o + MiSO adapter",
}
TRACKS = ["text_rules", "speech_intent", "sound_events", "joint_control", "screen_region"]
TRACK_NAMES = ["Text rules", "Spoken intent", "Sound events", "Audio + image", "Screen region"]


def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_intervals(cases, predictions, group_fields, repeats=5000):
    keys = list(predictions)
    by_id = {key: {r["id"]: r for r in value} for key, value in predictions.items()}
    rng = np.random.default_rng(20260925)
    samples = {key: [] for key in keys}
    group_counts = {}
    for track in TRACKS:
        groups = defaultdict(list)
        for case in cases:
            if case["track"] == track: groups[str(case[group_fields[track]])].append(case)
        values = list(groups.values())
        group_counts[track] = len(values)
        indices = rng.integers(0, len(values), size=(repeats, len(values)))
        counts = np.asarray([len(g) for g in values])
        for key in keys:
            hits = np.asarray([sum(int(np.argmax(by_id[key][c["id"]]["probabilities"])) == c["target"]
                                   if by_id[key].get(c["id"], {}).get("status") == "ok" else 0 for c in group) for group in values])
            samples[key].append(hits[indices].sum(-1) / counts[indices].sum(-1))
    combined = {key: np.mean(values, axis=0) for key, values in samples.items()}
    result = {"groups": group_counts, "repeats": repeats, "seed": 20260925,
              "macro_accuracy_95ci": {key: np.quantile(values, [.025, .975]).tolist() for key, values in combined.items()}}
    if len(keys) == 2:
        difference = combined[keys[0]] - combined[keys[1]]
        result["paired_difference"] = {"left": keys[0], "right": keys[1], "95ci": np.quantile(difference, [.025, .975]).tolist()}
    return result


def engineering_summaries():
    stage = {}
    for key in ("minicpmo45", "qwen3-30ba3b"):
        values = rows(REPORT / "attempts" / (key + "-confirmation-v1") / "predictions.jsonl")
        stage[key] = {}
        for track in TRACKS:
            selected = [r for r in values if r["track"] == track and r["status"] == "ok"]
            stage[key][track] = {name: float(np.median([r["timings"][name] for r in selected])) for name in
                                ["text_tokenizer_probe_ms", "decode_ms", "processor_ms", "forward_ms", "score_and_d2h_ms", "request_ms"]}
    (REPORT / "stage-timings.json").write_text(json.dumps({"boundary": "Median per component; component medians need not sum to median total. Plain-text tokenizer probe excludes media placeholders; actual tokenization is also included in processor time.", "confirmation": stage}, indent=2) + "\n")
    reference = {r["id"]: r for r in rows(REPORT / "attempts/minicpmo45-diagnostics-v2/head-projection.jsonl")}
    probe = rows(REPORT / "attempts/minicpmo45-l4-v1/hardware-predictions.jsonl")
    result = {"scope": "Ten fixed development cases, five warm repeats; no quality re-selection", "hardware": {}}
    for name, mode in (("full_projection", "full"), ("candidate_projection", "trimmed")):
        selected = [r for r in probe if r["mode"] == name]
        result["hardware"][name] = entry = {
            "l4_peak_allocated_gib": max(r["peak_allocated_bytes"] for r in selected) / 1024**3,
            "l4_peak_reserved_gib": max(r["peak_reserved_bytes"] for r in selected) / 1024**3,
            "argmax_agreement_with_h100": sum(int(np.argmax(r["probabilities"]) == np.argmax(reference[r["id"]][mode]["probabilities"])) for r in selected),
            "maximum_probability_difference_with_h100": max(float(np.max(np.abs(np.asarray(r["probabilities"]) - reference[r["id"]][mode]["probabilities"]))) for r in selected),
            "tracks": {}}
        for track in TRACKS:
            subset = [r for r in selected if r["track"] == track]
            lo = float(np.median([t for r in subset for t in r["request_ms"]]))
            hi = float(np.median([t for r in subset for t in reference[r["id"]][mode]["request_ms"]]))
            entry["tracks"][track] = {"h100_pipeline_median_ms": hi, "l4_pipeline_median_ms": lo, "l4_over_h100_latency": lo / hi,
                "l4_busy_resource_cost_proxy_per_1000_decisions": lo * (.000222 + 4 * .0000131 + 32 * .00000222),
                "h100_busy_resource_cost_proxy_per_1000_decisions": hi * (.001097 + 4 * .0000131 + 64 * .00000222)}
    result["cost_proxy_note"] = "Continuous sequential work only, using warm medians and requested CPU/RAM; excludes idle allocation, queueing, cold starts, network and billing reconciliation. Rates checked at study time."
    (REPORT / "hardware-comparison.json").write_text(json.dumps(result, indent=2) + "\n")


def make_figures(candidates, confirmation):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfdf9", "axes.facecolor": "#fcfdf9",
                         "text.color": "#24392d", "axes.labelcolor": "#24392d", "svg.hashsalt": "miso-v4-study"})
    ordered = sorted(candidates, key=lambda c: c["development"]["macro_accuracy"])
    matrix = np.array([[c["development"]["tracks"][t]["accuracy"] * 100 for t in TRACKS] for c in ordered])
    fig, ax = plt.subplots(figsize=(10.6, 6.3))
    ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(TRACKS)), TRACK_NAMES)
    ax.set_yticks(range(len(ordered)), [NAMES[c["key"]] for c in ordered])
    ax.tick_params(length=0, pad=10)
    for i in range(len(ordered)):
        for j in range(len(TRACKS)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="white" if value >= 70 else "#23362b", fontsize=10)
    ax.set_title("MiSO v4 development study: quality by input/task", loc="left", fontsize=16, weight="bold", pad=20)
    fig.text(.015, .015, "264 development cases · same H100/BF16 · best development-selected readout per checkpoint\n* Research-only base license. Screen task is coarse nine-region localization, not precise clicking.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1))
    for suffix in ("svg", "png"): fig.savefig(ASSETS / f"v4-quality.{suffix}", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    offsets = {"gemma4-e2b": (7, 3), "gemma4-e4b": (7, -5), "gemma4-12b": (7, 4),
               "qwen25-3b": (7, -12), "qwen25-7b": (-150, -14), "qwen3-30ba3b": (8, 0),
               "minicpmo45": (-122, 7), "phi4mm": (7, -4), "qwen3-30ba3b-adapted": (7, 6),
               "minicpmo45-adapted": (-65, 11)}
    for c in candidates:
        d = c["development"]
        adapted = c["key"].endswith("-adapted")
        x, y = d["macro_track_p95_ms"], d["macro_accuracy"] * 100
        ax.scatter(x, y, s=130 if adapted else 65, marker="D" if adapted else "o", color="#315e3f" if adapted else "#788773", zorder=3)
        ax.annotate(NAMES[c["key"]], (x, y), xytext=offsets[c["key"]], textcoords="offset points", fontsize=9)
    ax.axhline(75, color="#a89066", linestyle="--", alpha=.8)
    ax.set_ylim(33, 92)
    ax.set_xlim(min(c["development"]["macro_track_p95_ms"] for c in candidates) - 20,
                max(c["development"]["macro_track_p95_ms"] for c in candidates) + 55)
    ax.text(ax.get_xlim()[1] - 1, 75.8, "75% macro floor; per-track gates also apply", fontsize=8, color="#83704f", ha="right")
    ax.set_xlabel("Mean of five per-track p95 pipeline times (ms) — lower is better")
    ax.set_ylabel("Equal-track development accuracy (%)")
    ax.grid(alpha=.16)
    ax.set_title("Quality and latency on the same H100 hardware", loc="left", fontsize=16, weight="bold", pad=16)
    fig.text(.03, .035, "Every candidate still failed at least one preregistered quality gate. These are research tradeoffs, not qualified frontier points.\nDiamonds: 512-step attention adapters. Timings exclude loading, RPC and feature copies; adapted readout overhead is verified in confirmation.", fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, 1))
    for suffix in ("svg", "png"): fig.savefig(ASSETS / f"v4-quality-latency.{suffix}", dpi=170)
    plt.close(fig)

    keys = list(confirmation)
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    x = np.arange(len(TRACKS))
    for i, key in enumerate(keys):
        values = [confirmation[key]["metrics"]["tracks"][t]["accuracy"] * 100 for t in TRACKS]
        bars = ax.bar(x + (i - .5) * .35, values, .34, label=NAMES[key], color=("#315e3f", "#918466")[i])
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], fontsize=9, padding=3)
    ax.set_xticks(x, TRACK_NAMES)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Confirmation accuracy (%)")
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=2, frameon=False)
    ax.set_title("Fresh confirmation: screen understanding remains the bottleneck", loc="left", fontsize=15, weight="bold", pad=14)
    fig.text(.03, .015, "320 confirmation cases per model · nomination, adapters and temperatures frozen before these predictions", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    for suffix in ("svg", "png"): fig.savefig(ASSETS / f"v4-confirmation.{suffix}", dpi=170)
    plt.close(fig)

    plan = branch_plan([torch.tensor([1, 2, 3, 4, 5, 6, 7]), torch.tensor([1, 2, 3, 4, 8, 9, 10, 11]),
                        torch.tensor([1, 2, 3, 4, 12, 13, 14])], 4)
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    ax.imshow(plan["attention_mask"][0, 0].numpy() == 0, cmap=ListedColormap(["#edf1e6", "#315e3f"]), interpolation="nearest")
    centers = [1.5, 5, 8.5, 12]
    labels = ["Shared state", "Question 1", "Question 2", "Question 3"]
    ax.set_xticks(centers, labels, rotation=20, ha="right")
    ax.set_yticks(centers, labels)
    for edge in (3.5, 6.5, 10.5):
        ax.axvline(edge, color="white", linewidth=2); ax.axhline(edge, color="white", linewidth=2)
    ax.set_xlabel("Keys / values that a token may read")
    ax.set_ylabel("Query tokens")
    ax.set_title("Shared input, isolated question branches", loc="left", fontsize=14, weight="bold", pad=16)
    fig.text(.03, .015, "Green = allowed attention. This is a designed mask, not learned attention weights.\nEach question sees the shared state and its own earlier tokens; question positions reset.", fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for suffix in ("svg", "png"): fig.savefig(ASSETS / f"v4-shared-prefix.{suffix}", dpi=170)
    plt.close(fig)


def main():
    protocol = read(ROOT / "evals/v4-selection-protocol-v1.json")
    nomination = read(ROOT / "evals/v4-nomination-v1.json")
    all_cases = rows(ROOT / "evals/manifests/v4_selection_v1.jsonl")
    cases = [r for r in all_cases if r["split"] == "confirmation"]
    candidates = [read(p) for p in (REPORT / "candidates").glob("*/selection.json")]
    confirmation, predictions = {}, {}
    frozen_time = datetime.fromisoformat(nomination["created_at"]).timestamp()
    for key, nominee in nomination["candidates"].items():
        attempt = nominee["base_key"] + "-confirmation-v1"
        folder = REPORT / "attempts" / attempt
        reserve = read(folder / "reservation.json")
        assert reserve["started_unix"] > frozen_time
        selection = REPORT / "candidates" / key / "selection.json"
        assert sha(selection) == nominee["selection_sha256"]
        values = rows(folder / "predictions.jsonl")
        probabilities = {r["id"]: r["probabilities"] for r in values if r["status"] == "ok"}
        metrics = summarize(cases, values, probabilities)
        confirmation[key] = {"attempt": attempt, "metrics": metrics, "quality_reasons": quality_reasons(metrics, protocol),
                             "result_sha256": sha(folder / "result.json")}
        predictions[key] = values
    intervals = paired_intervals(cases, predictions, nomination["bootstrap_groups"])
    qualifies = bool(nomination.get("winner") and not confirmation[nomination["winner"]]["quality_reasons"])
    result = {"date": "2026-09-25", "status": "release_candidate_qualified" if qualifies else "no_release_qualified_winner", "research_lead": nomination["research_lead"],
              "quality_reference": nomination["quality_reference"], "nomination_sha256": sha(ROOT / "evals/v4-nomination-v1.json"),
              "primary_checkpoints_measured": 8, "adapter_followups": 2, "total_manifest_cases": len(all_cases),
              "development": {c["key"]: c["development"] for c in candidates}, "confirmation": confirmation,
              "confirmation_uncertainty": intervals, "confirmation_used_for_training_or_retuning": False,
              "jev_parity_established": False, "openai_parity_established": False}
    (REPORT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    engineering_summaries()
    make_figures(candidates, confirmation)
    figures = {"summary_sha256": sha(REPORT / "summary.json"), "figures": {}}
    for name in ("v4-quality", "v4-quality-latency", "v4-confirmation", "v4-shared-prefix"):
        svg = ASSETS / (name + ".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()).rstrip() + "\n")
        figures["figures"][name] = {"kind": "design" if name == "v4-shared-prefix" else "measured_result",
                                   "svg_sha256": sha(ASSETS / (name + ".svg")), "png_sha256": sha(ASSETS / (name + ".png"))}
    (REPORT / "figures.json").write_text(json.dumps(figures, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "research_lead": result["research_lead"],
                     "confirmation_accuracy": {k:v["metrics"]["macro_accuracy"] for k,v in confirmation.items()},
                     "intervals": intervals}, indent=2))


if __name__ == "__main__":
    main()
