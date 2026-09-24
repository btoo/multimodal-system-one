"""Check local documentation links, figure provenance, and research-only status."""
from pathlib import Path
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems = []
    markdown_files = [ROOT / "README.md", ROOT / "program.md", *sorted((ROOT / "docs/research").glob("*.md"))]
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
    assert rows == [], "Research-only foundation must not contain fabricated model runs."
    state = json.loads((ROOT / "docs/research/research-state.json").read_text())
    assert state["model_runs_completed"] == 0
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Checked {len(markdown_files)} Markdown files, {len(manifest['figures'])} figures, campaign budget, and empty results ledger.")


if __name__ == "__main__":
    main()
