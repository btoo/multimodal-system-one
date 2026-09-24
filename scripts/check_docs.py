"""Check documentation links, explanatory figures, and baseline evidence links."""
from pathlib import Path
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems = []
    markdown_files = [ROOT / "README.md", ROOT / "program.md", ROOT / "evals/README.md",
                      *sorted((ROOT / "docs/research").glob("*.md")), *sorted((ROOT / "reports").glob("*/README.md")),
                      *sorted((ROOT / "artifacts").glob("*/README.md"))]
    for path in markdown_files:
        text = path.read_text()
        if text.count("```") % 2:
            problems.append(f"Unclosed code fence: {path.relative_to(ROOT)}")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            relative = target.split("#")[0]
            if relative and not (path.parent / relative).exists():
                problems.append(f"Missing link in {path.relative_to(ROOT)}: {target}")
    manifest = json.loads((ROOT / "docs/assets/manifest.json").read_text())
    assert manifest["model_results"] is False
    for name, row in manifest["figures"].items():
        svg = ROOT / "docs/assets" / f"{name}.svg"
        ET.parse(svg)
        assert row["kind"] in {"design", "analytic", "simulation"}
        assert hashlib.sha256(svg.read_bytes()).hexdigest() == row["svg_sha256"]
        assert (svg.with_suffix(".png")).exists()
    campaign = json.loads((ROOT / "experiments/campaign.json").read_text())
    assert campaign["status"] == "planned" and campaign["execution_enabled"] is False
    c = campaign
    budget = len(c["architecture_screen"]["families"]) * c["architecture_screen"]["seeds_per_family"] * c["architecture_screen"]["training_seconds_per_run"]
    budget += c["refinement"]["max_runs"] * c["refinement"]["training_seconds_per_run"]
    budget += c["confirmation"]["configurations"] * c["confirmation"]["fresh_seeds_per_configuration"] * c["confirmation"]["training_seconds_per_run"]
    assert budget == campaign["total_allocated_training_seconds"] == 24000
    with (ROOT / "experiments/results.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows == [], "Synthetic-search campaign has not executed; baseline evidence lives under reports/."
    state = json.loads((ROOT / "docs/research/research-state.json").read_text())
    suites = json.loads((ROOT / "evals/suites.json").read_text())
    assert suites["execution_enabled"] is True and suites["data_populated"] is True
    assert state["baseline_runs_completed"] == len(suites["results"]) == 3
    assert state["joint_model_runs_completed"] == len(suites["joint_training_results"]) == 4
    assert state["model_runs_completed"] == 7 and state["native_joint_model_trained"] is True
    for report in suites["joint_training_results"]:
        assert json.loads((ROOT/report).read_text())["status"] == "development_complete"
    for report in suites["joint_evaluation_results"]:
        assert json.loads((ROOT/report).read_text())["status"] == "held_out_evaluated"
    for report in suites["results"]:
        assert json.loads((ROOT / report).read_text())["status"] == "completed"
    assert {"speech_intent", "acoustic_events", "screen_understanding", "paired_audio_screen"} <= {x["id"] for x in suites["suites"]}
    catalog = json.loads((ROOT / "evals/dataset-catalog.json").read_text())
    assert catalog["datasets_downloaded"] == sum(row["downloaded"] for row in catalog["records"]) == 3
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Checked {len(markdown_files)} Markdown files, {len(manifest['figures'])} explanatory figures, campaign budget, baseline reports, and joint-model evidence.")


if __name__ == "__main__":
    main()
