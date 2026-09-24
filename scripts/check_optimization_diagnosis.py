"""Verify diagnosis provenance, media, saved scores and deterministic readouts."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile

from mmso.artifacts import ROOT, read_manifest, sha256
import diagnose_optimization as diagnosis


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            close(first, second)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8), (a, b)
    else:
        assert a == b, (a, b)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--recompute-probes", action="store_true")
args = parser.parse_args()
out = ROOT / "reports/optimization-diagnosis-v1"
report = json.loads((out / "report.json").read_text())
probes = json.loads((out / "primitive-probes.json").read_text())
assert report["optimizer_updates"] == probes["neural_optimizer_updates"] == 0
assert report["consumed_partitions"] == probes["consumed_partitions"] == ["train", "dev"]
for path, digest in report["source_hashes"].items():
    blob = subprocess.check_output(["git", "show", f"{report['git_revision']}:{path}"], cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest() == digest, path
blob = subprocess.check_output(["git", "show", f"{probes['git_revision']}:scripts/probe_primitive_representations.py"], cwd=ROOT)
assert hashlib.sha256(blob).hexdigest() == probes["script_sha256"]
manifest = ROOT / "evals/manifests/scale_panels_v1.jsonl"
assert sha256(manifest) == probes["manifest_sha256"]
scenes = read_manifest(manifest)
train = [s for s in scenes if s["split"] == "train"][:256]
dev = [s for s in scenes if s["split"] == "dev"]
assert probes["training_scene_ids"] == [s["id"] for s in train]
assert not {s["speaker"] for s in train} & {s["speaker"] for s in dev}
verified = set()
for scene in [*train, *dev]:
    for key in ["image", "audio"]:
        media = scene[key]
        if media["path"] not in verified:
            assert sha256(ROOT / media["path"]) == media["sha256"], media["path"]
            verified.add(media["path"])
inputs, _ = diagnosis.read_inputs()
for run, (training, scenes, records, predictions) in inputs.items():
    close(diagnosis.summarize(records, predictions), report["runs"][run]["development"])
    close(diagnosis.training_history(training), report["runs"][run]["history"])
    assert sha256(ROOT / training["checkpoint"]) == training["checkpoint_sha256"] == probes["runs"][run]["checkpoint_sha256"]
if args.recompute_probes:
    # New temporary destination keeps the archived report immutable.
    (ROOT / ".research").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mmso-primitive-probe-", dir=ROOT / ".research") as directory:
        destination = directory + "/probe.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/probe_primitive_representations.py"), "--output", destination], cwd=ROOT, check=True)
        actual = json.load(open(destination))
        close(actual["runs"], probes["runs"])
print(f"Verified source revisions, {len(verified)} media files, train/dev speaker separation and recomputed native metrics; probe recomputation={args.recompute_probes}.")
