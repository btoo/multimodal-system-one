"""Local artifacts, immutable provenance, and content checks."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))


def local_media_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("Media path escapes repository root")
    return path


def validate_manifest(rows, root=ROOT, verify_media=True):
    if not rows:
        raise ValueError("Manifest is empty")
    ids = set(); groups = {}; hashes = {}; files = {}; counts = Counter(); supports = defaultdict(Counter); schemas = {}
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate example {row['id']}")
        ids.add(row["id"])
        split = row["split"]
        if split not in {"train", "dev", "calibration", "test", "external_probe"}:
            raise ValueError(f"Invalid split: {split}")
        key = (row["dataset"], row["group_id"])
        if key in groups and groups[key] != split:
            raise ValueError(f"Group leakage: {key}")
        groups[key] = split
        for media in row["media"]:
            expected = media["sha256"]
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError("Invalid content hash")
            if expected in hashes and hashes[expected] != split:
                raise ValueError(f"Media-content leakage: {row['id']}")
            hashes[expected] = split
            path = local_media_path(root, media["path"])
            if verify_media:
                if path not in files:
                    files[path] = sha256(path)
                if files[path] != expected:
                    raise ValueError(f"Media hash mismatch: {row['id']}")
        if row["kind"] == "categorical":
            labels = row["labels"]
            if len(labels) < 2 or len(set(labels)) != len(labels) or row["target"] not in labels:
                raise ValueError("Invalid categorical target/schema")
            if row["dataset"] in schemas and schemas[row["dataset"]] != labels:
                raise ValueError("Inconsistent label order in dataset")
            schemas[row["dataset"]] = labels
            supports[split][row["target"]] += 1
        elif row["kind"] == "grounding":
            box = row["target_bbox_xyxy"]
            w, h = row["image_size"]
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError("Invalid screen bounds")
        else:
            raise ValueError("Unsupported populated manifest kind")
        counts[split] += 1
    return {"examples": len(rows), "splits": dict(counts), "groups": len(groups),
            "unique_media_hashes": len(hashes), "support": dict(supports),
            "media_verified": verify_media, "group_and_hash_disjoint": True}


def provenance(manifest):
    code_files = [*sorted((ROOT / "mmso").glob("*.py")), ROOT / "uv.lock"]
    fingerprints = {str(p.relative_to(ROOT)): sha256(p) for p in code_files}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "git_revision": revision,
            "working_tree_dirty": dirty, "source_hashes": fingerprints,
            "manifest_sha256": sha256(manifest), "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine()}
