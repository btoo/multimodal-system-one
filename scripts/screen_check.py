"""Independently rescore saved real-screen outputs and verify their frozen source."""
import hashlib
import json
import subprocess

import numpy as np

from mmso.artifacts import ROOT, read_manifest, sha256, validate_manifest
from mmso.grounding import center, rank_proposals
from mmso.metrics import point_hit, clustered_accuracy_interval
from mmso.screen import grid_regions, region_center


def verify_source(record):
    revision = record["git_revision"]
    for name, expected in {**record["source_hashes"], **record.get("additional_source_hashes", {})}.items():
        content = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected, (revision, name)


def verify_run(phase, version):
    directory = ROOT / f"reports/screens-ocr-{phase}-v{version}"
    report = json.loads((directory / "report.json").read_text())
    manifest = ROOT / ("evals/manifests/screen_grounding_v2.jsonl" if phase == "confirmation" else "evals/manifests/screen_grounding.jsonl")
    rows = read_manifest(manifest)
    predictions = read_manifest(directory / "predictions.jsonl")
    archives = read_manifest(directory / "proposals.jsonl")
    assert report["status"] == "completed"
    assert sha256(manifest) == report["provenance"]["manifest_sha256"]
    expected_ids = [row["id"] for row in rows]
    assert [row["id"] for row in predictions] == expected_ids
    assert [row["id"] for row in archives] == expected_ids
    assert len(set(expected_ids)) == len(rows)
    verify_source(report["provenance"])
    validate_manifest(rows)
    for row, prediction, archive in zip(rows, predictions, archives):
        proposals = archive["proposals"]
        box = row["target_bbox_xyxy"]
        for name, result in prediction["methods"].items():
            expected_hit = result["point"] is not None and point_hit(result["point"], box)
            assert result["hit"] == expected_hit
            if name in {"token_f1", "idf_coverage"}:
                actual = rank_proposals(proposals, row["instruction"], name)
                assert actual["point"] == result["point"]
                assert actual["winning_proposal"] == result["winning_proposal"]
                assert actual["abstained"] == result["abstained"]
                recall = any(point_hit(center(p["bbox_xyxy"]), box) for p in proposals)
            elif name == "clip_grid":
                regions = grid_regions(*row["image_size"])
                assert result["point"] in [region_center(region) for region in regions]
                recall = any(point_hit(region_center(region), box) for region in regions)
            else:
                assert name == "center"
                assert result["point"] == [v / 2 for v in row["image_size"]]
                recall = expected_hit
            assert result["proposal_recall"] == recall
    groups = [(report["results"], predictions)]
    for field, slices in report["slices"].items():
        groups += [(metrics, [p for p in predictions if p[field] == value]) for value, metrics in slices.items()]
    for metrics, selected in groups:
        for name, values in metrics.items():
            cases = [p["methods"][name] for p in selected]
            for key, source in [("hits", "hit"), ("proposal_center_hits", "proposal_recall"),
                                ("abstentions", "abstained"), ("zero_proposals", "zero_proposals"),
                                ("inference_errors", "inference_error")]:
                if key in values:assert values[key] == sum(p.get(source, False) for p in cases)
            assert values["examples"] == len(cases)
            assert np.isclose(values["hit_rate"], np.mean([p["hit"] for p in cases]))
            assert np.isclose(values["proposal_center_recall"], np.mean([p["proposal_recall"] for p in cases]))
            assert np.isclose(values["pipeline_p50_ms"], np.median([p["pipeline_seconds"] for p in cases]) * 1000)
            assert np.isclose(values["pipeline_p95_ms"], np.quantile([p["pipeline_seconds"] for p in cases], .95) * 1000)
    if phase == "confirmation":
        nomination = json.loads((ROOT / f"evals/screen-nomination-v{version}.json").read_text())
        name = nomination["selected_variant"]
        delta = [float(p["methods"][name]["hit"]) - float(p["methods"]["clip_grid"]["hit"]) for p in predictions]
        expected = clustered_accuracy_interval(delta, [p["application"] for p in predictions], seed=20260925, repeats=5000)
        assert report["paired_delta_vs_grid"]["application_cluster_95ci"] == expected
        assert report["paired_delta_vs_grid"]["difference"] == np.mean(delta)
    print(f"Verified {phase} v{version}: {len(rows)} exact IDs, all slices/proposals/ranks/timings, committed source fingerprints")


def verify_reproduction():
    directory = ROOT / "reports/screens-grid-reproduced-v2"
    report = json.loads((directory / "report.json").read_text())
    rows = read_manifest(ROOT / "evals/manifests/screen_grounding.jsonl")
    predictions = read_manifest(directory / "predictions.jsonl")
    assert [p["id"] for p in predictions] == [r["id"] for r in rows]
    verify_source(report["provenance"])
    source = subprocess.check_output(["git", "show", report["provenance"]["git_revision"] + ":scripts/screen_reproduce.py"], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == report["source_script_sha256"]
    for row, prediction in zip(rows, predictions):
        box = row["target_bbox_xyxy"]
        assert prediction["hit"] == point_hit(prediction["point"], box)
        assert prediction["center_hit"] == point_hit([x / 2 for x in row["image_size"]], box)
        assert prediction["proposal_recall"] == any(point_hit(region_center(region), box) for region in grid_regions(*row["image_size"]))
    assert report["grid_hits"] == sum(p["hit"] for p in predictions) == 0
    assert report["center_hits"] == sum(p["center_hit"] for p in predictions) == 0
    assert report["grid_point_coverage_hits"] == sum(p["proposal_recall"] for p in predictions) == 1
    print("Verified original control reproduction: grid0/24, center0/24, proposal coverage1/24")


if __name__ == "__main__":
    old = read_manifest(ROOT / "evals/manifests/screen_grounding.jsonl")
    new = read_manifest(ROOT / "evals/manifests/screen_grounding_v2.jsonl")
    assert {r["group_id"] for r in old}.isdisjoint({r["group_id"] for r in new})
    assert {r["media"][0]["sha256"] for r in old}.isdisjoint({r["media"][0]["sha256"] for r in new})
    assert len({r["media"][0]["sha256"] for r in new}) == 64
    verify_reproduction()
    verify_run("development", 2)
    verify_run("development", 3)
    verify_run("confirmation", 3)
