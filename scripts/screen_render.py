"""Render measured fresh-screen results; never synthesize performance values."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmso.artifacts import ROOT


def render():
    directory = ROOT / "reports/screens-ocr-confirmation-v3"
    report = json.loads((directory / "report.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.2, 1]})
    figure.patch.set_facecolor("#f6f8fb")
    cases = [report["results"], report["slices"]["ui_type"]["text"], report["slices"]["ui_type"]["icon"]]
    locations = np.arange(3)
    for index, (method, label, color) in enumerate([
        ("token_f1", "Tiled OCR + lexical ranking", "#087f8c"), ("clip_grid", "Original CLIP grid", "#69798d")]):
        counts = [case[method] for case in cases]
        bars = axes[0].bar(locations + (index - .5) * .3, [c["hit_rate"] * 100 for c in counts],
                           width=.3, label=label, color=color)
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4,
                         f"{count['hits']}/{count['examples']}", ha="center", fontsize=11, weight="bold")
    axes[0].set_xticks(locations, ["All targets", "Text", "Icons"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Grounding hit rate (%)")
    axes[0].set_title("Fresh screenshots: progress on text", loc="left", fontsize=14, weight="bold", pad=15)
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    axes[0].grid(axis="y", color="#dde4ed", linewidth=.6)
    axes[0].set_axisbelow(True)
    recall = [case["token_f1"] for case in cases]
    bars = axes[1].bar(locations, [case["proposal_center_recall"] * 100 for case in recall],
                      color="#c2e3e2", width=.55, label="An OCR proposal could hit target")
    axes[1].bar(locations, [case["hit_rate"] * 100 for case in recall], color="#087f8c", width=.55,
                label="Selected proposal hit target")
    for bar, count in zip(bars, recall):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.7,
                     f"{count['proposal_center_hits']}/{count['examples']}", ha="center", weight="bold")
    axes[1].set_xticks(locations, ["All targets", "Text", "Icons"])
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Proposal-center coverage (%)")
    axes[1].set_title("Ranking still misses reachable targets", loc="left", fontsize=14, weight="bold", pad=15)
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    axes[1].grid(axis="y", color="#dde4ed", linewidth=.6)
    axes[1].set_axisbelow(True)
    figure.suptitle("Real-screen confirmation: 64 unique public screenshots · 8 applications",
                    x=.075, ha="left", fontsize=17, weight="bold", y=.98)
    timing = report["results"]["token_f1"]
    figure.text(.075, .04, f"Frozen once before evaluation · 32 text / 32 icon cases · OCR pipeline p50 / p95: {timing['pipeline_p50_ms']:.0f} / {timing['pipeline_p95_ms']:.0f} ms\n"
                "Separate pretrained OCR control. Zero icon successes; no native-model or browser-execution claim.",
                fontsize=10, color="#425167", linespacing=1.7)
    figure.subplots_adjust(left=.075, right=.98, bottom=.22, top=.83, wspace=.34)
    for suffix in ["svg", "png"]:
        figure.savefig(directory / ("results." + suffix), dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":render()
