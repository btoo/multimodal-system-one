"""Render measured size-versus-prior results from the verified study summary."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmso.artifacts import ROOT

OUT = ROOT / "reports/scale-v1"
summary = json.loads((OUT / "summary.json").read_text())
names = ["scale-small-v1", "scale-large-v1", "scale-factorized-v1", "scale-served-reference-v1"]
labels = ["Small RGB\n668k", "Larger RGB\n1.62M", "Shape + color\n667k", "Existing served\n668k"]
rows = summary["conditions"]
BG, INK, TEAL, BLUE, GOLD = "#f8fafc", "#14283e", "#007e78", "#2e62bc", "#b26614"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "scale-v1"})

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.subplots_adjust(left=.065, right=.975, top=.73, bottom=.22, wspace=.25)
fig.text(.065, .93, "Does size or visual structure help?", fontsize=24, weight="bold", color=INK)
fig.text(.065, .85, "Real keyword audio + generated panels · one seed per condition · fresh confirmation speakers", color=INK)
for name, label, color in zip(names[:3], ["Small RGB", "Larger RGB", "Shape + color"], [BLUE, GOLD, TEAL]):
    training = json.loads((ROOT / "reports" / name / "training.json").read_text())
    history = training["history"]
    axes[0].plot([h["steps"] for h in history], [h["selection_normalized_nll"] for h in history], label=label, color=color, lw=2)
    selected = next(h for h in history if h["epoch"] == training["configuration"]["selected_epoch"])
    axes[0].scatter([selected["steps"]], [selected["selection_normalized_nll"]], color=color, s=50, zorder=3)
axes[0].set(title="Development objective (lower is better)", xlabel="Optimizer updates", ylabel="Mean normalized NLL across ID + composition")
axes[0].legend(frameon=False)
x = np.arange(len(names))
for offset, slice_name, label, color in [(-.18, "in_distribution", "ID", BLUE), (.18, "compositional", "Known composition gap", TEAL)]:
    values = [rows[name]["by_slice"][slice_name]["joint_macro_accuracy"] * 100 for name in names]
    axes[1].bar(x + offset, values, width=.35, label=label, color=color)
    for i, v in enumerate(values):
        axes[1].text(i + offset, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
axes[1].set(title="Same confirmation rows for every checkpoint", ylabel="Six-family accuracy (%)", ylim=(0, 110))
axes[1].set_xticks(x, labels)
axes[1].legend(frameon=False, fontsize=9, loc="upper left")
for ax in axes:
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
fig.text(.065, .09, "80 previously unused speakers; 117 recordings per slice. Known held tuple types were used in development.", fontsize=10, color=INK)
fig.text(.065, .05, "Fixed-checkpoint intervals are in the report. Neither extra training seeds nor real-browser generalization are established.", fontsize=10, color=INK)
for ext in ["svg", "png"]:
    fig.savefig(OUT / f"results.{ext}", dpi=160, facecolor=BG, metadata={"Date": None} if ext == "svg" else None)
plt.close(fig)

lines = ["# Scale and visual-prior study", "",
         "**Neither intervention established a compositional gain, and no new checkpoint replaces the served model.** The 2.42× larger model and the small factorized visual model finish near the matched small baseline. All three selected checkpoints lose about 20 percentage points of ID accuracy against the existing served checkpoint on identical fresh confirmation rows.", "",
         "Three predeclared training conditions compare model size with a targeted visual prior. The existing served checkpoint is a frozen reference on the same new confirmation examples.", "",
         "![Measured scale-study results](results.svg)", "",
         "| Condition | Parameters | Budget updates (selected checkpoint) | Train seconds | Selected epoch | ID accuracy | Composition accuracy |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for name, label in zip(names, ["Small RGB", "Larger RGB", "Factorized shape/color", "Existing served checkpoint"]):
    r = rows[name]
    seconds = f"{r['train_seconds']:.1f}" if r["train_seconds"] is not None else "Earlier study"
    a, b = [r["by_slice"][s]["joint_macro_accuracy"] for s in ["in_distribution", "compositional"]]
    lines.append(f"| {label} | {r['parameters']:,} | {r['actual_steps']:,} ({r['selected_steps']:,}) | {seconds} | {r['selected_epoch']} | {a:.2%} | {b:.2%} |")
lines += ["", f"Development nominated **{summary['selected_run']}** by mean normalized NLL across the ID and compositional development slices before confirmation was opened. Every condition is retained, including inferior or incomplete outcomes. The default served model remains `joint-full-v2`.", "",
          "The new training attempts each consume the same 8,192-update budget and sampled data stream. Development selects among epoch checkpoints, so selected weights may represent different numbers of updates, shown in parentheses. FLOPs and time are not matched. The older served checkpoint used a different seed, an ID-only selection objective, and more updates; it is an operational reference, not the matched size control.", "",
          "The new selector chose early checkpoints because later training made the models increasingly overconfident on the known composition shift. The visual prior eventually improved ID development accuracy, but its shifted normalized NLL deteriorated. A color-invariant intermediate image view did not establish compositionally reliable final decisions. This does not prove scaling or factorization never helps; it is one bounded, single-seed outcome with this optimizer and representation.", "",
          "## Paired fixed-checkpoint differences", "",
          "| Candidate | Reference | Slice | Difference (percentage points) | Speaker-cluster 95% interval |",
          "|---|---|---|---:|---:|"]
for candidate, references in summary["paired_comparisons"].items():
    for reference, slices in references.items():
        for slice_name, gap in slices.items():
            low, high = gap["interval_percentage_points"]
            lines.append(f"| {candidate} | {reference} | {slice_name} | {gap['mean_percentage_points']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
lines += ["", "Intervals use 2,000 paired bootstrap resamples over 80 speaker clusters. These are exploratory comparisons of fixed checkpoints, without correction for multiple comparisons. They do not estimate training-seed variability.", "",
          "## What the test means", "",
          "Confirmation uses 117 human recordings from 80 speakers absent from every previous manifest. Each recording has a new ID panel and a new panel containing held command/color combinations, for 702 primary questions per slice. Related questions and repeated recordings are not independent samples. All held tuples are excluded from training.", "",
          "**The composition gap is already known.** Its eight tuple identities were exposed by the earlier 45.66% failure and now appear in development. This test measures the same gap on fresh speakers/images, not transfer to previously unexamined gap identities. It contains generated panels, not real screenshots. The old real-screen grounding failure remains unresolved.", "",
          "The factorized visual encoder imposes a fixed dark-ink threshold suitable for this renderer. It separates a learned shape CNN from a spatially pooled color MLP. No transcript, scene metadata, symbolic labels, or oracle enters inference. This controlled prior is not a general segmentation model.", "",
          "The next development question is how to train stable perception and cross-modal binding before committing more compute. Repeated development seeds and a perception/binding curriculum are testable follow-ups. A new real-screenshot confirmation set is still needed for browser claims; these results do not justify increasing production scope.", "",
          "## Evidence", "",
          "- [Design, sources and reproduction](../../docs/research/scale-study.md)",
          "- [Frozen protocol](../../evals/scale-protocol-v1.json) and [pre-test nomination](../../evals/scale-nomination-v1.json)",
          "- [Dataset audit](../../evals/acquisition/scale_panels_v1.json) and [summary with intervals](summary.json)",
          "- [Small training](../scale-small-v1/training.json) and [evaluation](../scale-small-v1/evaluation.json)",
          "- [Larger training](../scale-large-v1/training.json) and [evaluation](../scale-large-v1/evaluation.json)",
          "- [Factorized training](../scale-factorized-v1/training.json) and [evaluation](../scale-factorized-v1/evaluation.json)",
          "- [Frozen served reference evaluation](../scale-served-reference-v1/evaluation.json)", "",
          "```sh", "uv run python scripts/check_scale_results.py", "uv run python scripts/summarize_scale_results.py", "uv run --group docs python scripts/render_scale_results.py", "```", ""]
(OUT / "README.md").write_text("\n".join(lines))
print("Rendered scale-study report and measured figures")
