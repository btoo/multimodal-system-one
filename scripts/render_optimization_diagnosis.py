"""Render already-computed optimization diagnostics; no new model evaluation."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmso.artifacts import ROOT

OUT = ROOT / "reports/optimization-diagnosis-v1"
report = json.loads((OUT / "report.json").read_text())
probes = json.loads((OUT / "primitive-probes.json").read_text())
names = list(probes["runs"])
labels = ["Served v2", "Small RGB", "Larger RGB", "Factorized"]
BG, INK, BLUE, TEAL = "#f8fafc", "#14283e", "#2e62bc", "#007e78"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "optimization-diagnosis-v1"})
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.subplots_adjust(left=.07, right=.97, top=.70, bottom=.23, wspace=.30)
fig.text(.07, .93, "Decodable features do not guarantee usable decisions", fontsize=22, weight="bold", color=INK)
fig.text(.07, .84, "Frozen checkpoints · fixed training-only ridge probes · existing development examples", color=INK)
x = np.arange(len(names))
for offset, slice_name, label, color in [(-.18, "in_distribution", "ID glyph probe", BLUE), (.18, "compositional", "Held-color glyph probe", TEAL)]:
    values = [probes["runs"][name]["development"][slice_name]["glyph"]["accuracy"] * 100 for name in names]
    axes[0].bar(x + offset, values, .35, label=label, color=color)
    for i, value in enumerate(values):
        axes[0].text(i + offset, value + 2, f"{value:.1f}", ha="center", fontsize=9)
axes[0].set_xticks(x, labels)
axes[0].set(title="What a matched linear readout can extract", ylabel="Glyph accuracy (%)", ylim=(0, 118))
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
factor = report["runs"]["scale-factorized-v1"]["development"]
for offset, slice_name, label, color in [(-.18, "in_distribution", "ID joint decision", BLUE), (.18, "compositional", "Held-color joint decision", TEAL)]:
    rows = [factor[slice_name][task] for task in ["color", "opposite_color", "position", "opposite_position"]]
    values = [sum(row[group]["accuracy"] * row[group]["examples"] for row in rows) / sum(row[group]["examples"] for row in rows) * 100 for group in ["present", "absent"]]
    axes[1].bar(np.arange(2) + offset, values, .35, label=label, color=color)
    for i, value in enumerate(values):
        axes[1].text(i + offset, value + 2, f"{value:.0f}", ha="center", fontsize=9)
axes[1].set_xticks(np.arange(2), ["Target present", "Target absent"])
axes[1].set(title="What the factorized nominee actually answers", ylabel="Color / location accuracy (%)", ylim=(0, 118))
axes[1].legend(frameon=False, loc="upper left", fontsize=9)
for ax in axes:
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
fig.text(.07, .10, "Factorized glyph information is available to the probe, while the question-conditioned head always answers not present.", fontsize=10, color=INK)
fig.text(.07, .055, "Probes are diagnostic classifiers, not served outputs. Zero neural updates; no new final-test inference or real-screen claim.", fontsize=10, color=INK)
for ext in ["svg", "png"]:
    fig.savefig(OUT / f"diagnosis.{ext}", dpi=160, facecolor=BG, metadata={"Date": None} if ext == "svg" else None)
plt.close(fig)

lines = ["# Optimization diagnosis", "",
         "**The factorized encoder contains decodable glyph identity, but its selected joint head does not use it.** A matched linear probe scores 92.58% on held-color glyphs while the actual candidate scorer always answers “not present” on color/position questions. This distinguishes feature information from usable cross-modal binding.", "",
         "![Measured optimization diagnosis](diagnosis.svg)", "",
         "## Matched frozen-representation probes", "",
         "Each checkpoint receives the same fixed ridge procedure: 256 training panels, 1,024 glyphs, 256 recordings, training-only feature standardization, ridge strength 1 and an unpenalized intercept. Evaluation uses the same scale-v1 development rows for every checkpoint. There are 12 temporary closed-form classifier fits, zero neural optimizer updates, and no saved readout weights.", "",
         "| Checkpoint | Glyph ID | Glyph composition | Color ID / composition | Audio word |", "|---|---:|---:|---:|---:|"]
for name, label in zip(names, labels):
    row = probes["runs"][name]["development"]
    a, b = row["in_distribution"], row["compositional"]
    lines.append(f"| {label} | {a['glyph']['accuracy']:.2%} | {b['glyph']['accuracy']:.2%} | {a['color']['accuracy']:.2%} / {b['color']['accuracy']:.2%} | {a['audio_word']['accuracy']:.2%} |")
lines += ["", "Audio inputs are identical across the two development image slices. Probe feature width follows each model: 128 except the larger model's 192. These are trained diagnostic readouts on exposed development data, not a replacement for the learned question/candidate head or independent confirmation.", "",
          "## The absence shortcut in saved decision predictions", "",
          "| Checkpoint / native dev slice | Predict not present | Present-target accuracy | Absent-target accuracy | Heard-word question | Tile-word question |",
          "|---|---:|---:|---:|---:|---:|"]
for name, row in report["runs"].items():
    for slice_name, tasks in row["development"].items():
        rows = [tasks[t] for t in ["color", "opposite_color", "position", "opposite_position"]]
        rate = sum(r["predicted_counts"].get("not present", 0) for r in rows) / sum(r["examples"] for r in rows)
        accuracies = {group: sum(r[group]["accuracy"] * r[group]["examples"] for r in rows) / sum(r[group]["examples"] for r in rows) for group in ["present", "absent"]}
        lines.append(f"| {name} / {slice_name} | {rate:.2%} | {accuracies['present']:.2%} | {accuracies['absent']:.2%} | {tasks['heard_word']['accuracy']:.2%} | {tasks['tile_word']['accuracy']:.2%} |")
lines += ["", "This table recomputes each run's saved native development predictions. The served-v2 row uses its older development panels; the three scale rows share scale-v1 panels. The linear-probe table above instead scores all checkpoints on the same scale-v1 panels. The existing tile-word question also tests location routing and text/scorer behavior, so its failure is not a pure perception diagnosis.", "",
          "## Checkpoint-selection history", "",
          "The prior scale selector used raw normalized NLL. Temperature scaling was fitted afterward. Factorized epoch 32 reached 71.61% ID and 50.87% composition accuracy, but composition normalized NLL deteriorated to 2.0368, so epoch 2 was retained. Later weights/logits were not saved. We cannot reconstruct a calibrated late-checkpoint result. A positive temperature never changes argmax accuracy.", "",
          "## CPU gradient snapshots", "",
          "A deterministic re-paired batch uses the first 32 training scenes: 256 requests, comprising 192 joint, 32 heard-word and 32 tile-word losses. Gradients are computed on CPU with two threads and no optimizer step. The checkpoint parameters are compared before/after to verify they remain unchanged. The report includes L2, RMS and parameter-relative norms, and joint/auxiliary cosine similarity per branch.", "",
          "For the selected small model, image-branch joint/tile-word gradient cosine is -0.332; for the served model it is +0.726. Existing tile-word supervision also sends gradients through audio although the target is visually determined. These are single-batch observations, not causal evidence that gradient imbalance caused training failure. Different parameterizations and loss scales affect the norms.", "",
          "## Reproduce and inspect", "",
          "- [Primary-source review and falsification plan](../../docs/research/optimization-review.md)",
          "- [Saved-prediction diagnosis, gradient snapshots and source hashes](report.json)",
          "- [Matched frozen readouts and their source/input hashes](primitive-probes.json)", "",
          "```sh", "# Recompute saved metrics and frozen readouts without overwriting archived evidence.",
          "uv run python scripts/check_optimization_diagnosis.py --recompute-probes",
          "uv run python scripts/probe_primitive_representations.py --output .research/repeated-primitive-probes.json",
          "uv run --group docs python scripts/render_optimization_diagnosis.py", "```", "",
          "The diagnostics load versioned manifests and select train/dev records for computation; this is not a sandbox that hides other manifest rows. They perform no new calibration/final inference. The generated-panel setting and previously known composition gap remain explicit limitations.", ""]
(OUT / "README.md").write_text("\n".join(lines))
print("Rendered optimization diagnosis")
