"""Render the complete two-seed recipe comparison from verified saved evidence."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmso.artifacts import ROOT


OUT = ROOT / "reports/optimization-v1"
summary = json.loads((OUT / "summary.json").read_text())
names = ["optimization-control-s24-v1", "optimization-primitive-s24-v1", "optimization-control-s25-v1",
         "optimization-primitive-s25-v1", "optimization-served-reference-v1"]
labels = ["Control\nseed 24", "Primitive\nseed 24", "Control\nseed 25", "Primitive\nseed 25", "Served v2\nreference"]
rows = summary["conditions"]
BG, INK, BLUE, TEAL, GOLD = "#f8fafc", "#14283e", "#2e62bc", "#007e78", "#b26614"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "optimization-v1"})

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.subplots_adjust(left=.06, right=.98, top=.73, bottom=.22, wspace=.24)
fig.text(.06, .93, "Does primitive supervision improve joint decisions?", fontsize=22, weight="bold", color=INK)
fig.text(.06, .85, "Same 668k inference model · two matched seeds · 256 new human recordings from 58 new speakers", color=INK)
x = np.arange(len(names))
for offset, slice_name, label, color in [(-.18, "in_distribution", "Familiar combinations", BLUE), (.18, "compositional", "Known composition gap", TEAL)]:
    values = [rows[name]["by_slice"][slice_name]["joint_macro_accuracy"] * 100 for name in names]
    axes[0].bar(x + offset, values, width=.35, label=label, color=color)
    for i, value in enumerate(values): axes[0].text(i + offset, value + 1, f"{value:.1f}", ha="center", fontsize=8)
    values = [rows[name]["shortcut_slices"][slice_name + ":present_target"]["accuracy"] * 100 for name in names]
    axes[1].bar(x + offset, values, width=.35, color=color)
    for i, value in enumerate(values): axes[1].text(i + offset, value + 1, f"{value:.1f}", ha="center", fontsize=8)
for ax, title in zip(axes, ["All six joint question families", "Color/location when the target is present"]):
    ax.set(title=title, ylabel="Accuracy (%)", ylim=(0, 110))
    ax.set_xticks(x, labels, fontsize=9); ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
fig.text(.06, .10, "All attempts receive 8,192 updates; treatment includes 256 primitive-only warmup updates. Heads are discarded for inference.", fontsize=10, color=INK)
fig.text(.06, .05, "Generated panels and a previously known composition gap. Fixed-checkpoint speaker intervals and seed effects are in the report.", fontsize=10, color=INK)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"results.{ext}", dpi=160, facecolor=BG, metadata={"Date": None} if ext == "svg" else None)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.subplots_adjust(left=.065, right=.98, top=.73, bottom=.22, wspace=.25)
fig.text(.065, .93, "Accuracy gains and probability quality are separate", fontsize=22, weight="bold", color=INK)
fig.text(.065, .85, "Normalized NLL · lower is better · temperature fitted only on calibration data after selection", color=INK)
for ax, slice_name, title in zip(axes, ["in_distribution", "compositional"], ["Familiar combinations", "Known composition gap"]):
    for offset, key, label, color in [(-.18, "raw_by_slice", "Raw", GOLD), (.18, "by_slice", "Temperature adjusted", TEAL)]:
        values = [rows[name][key][slice_name]["joint_macro_normalized_nll"] for name in names]
        ax.bar(x + offset, values, width=.35, label=label, color=color)
    ax.axhline(1, color=INK, lw=1, ls=":", label="Uniform reference")
    ax.set(title=title, ylabel="Normalized negative log-likelihood")
    ax.set_xticks(x, labels, fontsize=9); ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=9)
fig.text(.065, .09, "Temperature does not change the top-ranked answer. Confidence on arbitrary rubrics, new task families, and real screens remains unvalidated.", fontsize=10, color=INK)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"probabilities.{ext}", dpi=160, facecolor=BG, metadata={"Date": None} if ext == "svg" else None)
plt.close(fig)

passed = summary["accuracy_gates_passed"]
lead = ("The primitive-supervision candidate passed the predeclared accuracy gates. Serving still requires separate compatibility and latency verification."
        if passed else "The predeclared accuracy gates did not all pass; no automatic serving change is supported by this study.")
lines = ["# Primitive-supervision optimization study", "", lead, "", "![Matched seed results and present-target behavior](results.svg)", "",
         "| Checkpoint | Budget updates | Selected updates | Training seconds | Familiar accuracy | Composition accuracy | Temperature |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for name, label in zip(names, ["Control, seed 24", "Primitive supervision, seed 24", "Control, seed 25", "Primitive supervision, seed 25", "Served v2 reference"]):
    row = rows[name]; timing = f"{row['train_seconds']:.1f}" if row["train_seconds"] is not None else "Earlier study"
    a, b = [row["by_slice"][s]["joint_macro_accuracy"] for s in ("in_distribution", "compositional")]
    lines.append(f"| {label} | {row['actual_steps']:,} | {row['selected_steps']:,} | {timing} | {a:.2%} | {b:.2%} | {row['temperature']:.3f} |")
lines += ["", "Both recipes retain the original 668,097-parameter inference architecture. The treatment uses 1,548 temporary training parameters, direct glyph/color/audio-word supervision, and a shared concept classifier. Its 256 warmup updates are included in the common budget. This compares a defined recipe, not each of its components in isolation; FLOPs and elapsed time differ.", "",
          "The same accuracy gate/selection rule applies to both new conditions. It differs from the previous scale study's raw-loss-only rule. The served-v2 checkpoint is an operational reference with a different historical budget and selection process, not the matched causal control.", "",
          "## Repeated-seed effects", "", "| Seed | Treatment minus control, mean familiar/composition accuracy |", "|---|---:|"]
for seed, row in summary["paired_seed_effects"].items(): lines.append(f"| {seed} | {row['balanced_accuracy_difference_percentage_points']:+.2f} percentage points |")
lines += ["", "Both seed outcomes are retained. Two seeds are a replication check; they do not precisely estimate training-seed variance.", "",
          "## Paired confirmation differences", "", "| Candidate | Reference | Slice | Difference | Speaker-cluster 95% interval |", "|---|---|---|---:|---:|"]
for candidate, references in summary["paired_comparisons"].items():
    for reference, slices in references.items():
        for slice_name, gap in slices.items():
            low, high = gap["interval_percentage_points"]
            lines.append(f"| {candidate} | {reference} | {slice_name} | {gap['mean_percentage_points']:+.2f} pp | [{low:+.2f}, {high:+.2f}] pp |")
lines += ["", "Intervals use 2,000 paired speaker-cluster resamples over 58 speakers. They condition on the trained checkpoints, are not simultaneous multiple-comparison guarantees, and do not measure variation over future training runs.", "",
          "## Probability quality", "", "![Raw and adjusted probability loss](probabilities.svg)", "",
          "Complete raw and temperature-adjusted NLL, Brier, reliability bins, and task results remain in each evaluation file. Higher accuracy can coexist with highly overconfident errors. Calibration uses only the separate calibration rows; it does not establish confidence reliability on a new distribution.", "",
          "## Shortcut and data checks", "",
          "The right-hand accuracy panel evaluates only color/location questions whose target actually exists. Predicting absence cannot pass this slice. The summary also retains absent-target accuracy and the rate of predicting `not present`, for development and confirmation separately.", "",
          "Confirmation contains 256 balanced keyword recordings from 58 speakers absent from all prior manifests, paired with 512 newly generated panels. Each primary slice has 1,536 related questions. Speaker grouping is used for uncertainty. The source is the official Speech Commands v0.02 test archive; this filtered custom task is not its official benchmark.", "",
          "Held word/color identities are the same previously exposed gap types used in development. Voice and image assets are fresh. This does not establish novel-gap, sentence-intent, real-screen, or browser-execution generalization.", "",
          "A numerical tie edge was corrected between seeds before confirmation: exact correct-count fractions now determine accuracy ties. Both completed seed-24 winners were checked and unchanged. The [precision audit](selection-precision-audit.json) and original source revisions are retained.", "",
          "## Gates and evidence", "", f"Pre-test development nominee: `{summary['candidate_run']}`.", "", "| Gate | Passed |", "|---|---|"]
for gate, value in summary["promotion_gates"].items(): lines.append(f"| {gate.replace('_', ' ')} | {value} |")
lines += ["", "- [Training design and reproduction](../../docs/research/primitive-supervision.md)",
          "- [Frozen protocol](../../evals/optimization-protocol-v1.json) and [nomination](../../evals/optimization-nomination-v1.json)",
          "- [Data audit](../../evals/acquisition/optimization_panels_v1.json) and [all comparisons/shortcut slices](summary.json)", "",
          "```bash", "uv run python scripts/check_optimization_results.py", "uv run --group docs python scripts/render_optimization_results.py", "```", ""]
(OUT / "README.md").write_text("\n".join(lines))
print("Rendered two-seed optimization evidence and probability-quality figures")
