"""Allowlist source, frozen protocol, and public benchmark media for Modal."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".research/v4/bundle"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    paths = ["mmso/__init__.py", "mmso/backbone_study.py", "cloud/modal_v4_study.py",
             "evals/v4-selection-protocol-v1.json", "evals/v4-candidates-v1.json", "evals/manifests/v4_selection_v1.jsonl"]
    allowlisted = set(paths)
    rows = [json.loads(x) for x in (ROOT / paths[-1]).read_text().splitlines()]
    for name in ["mmso/backbone_adapters.py", "evals/v4-adapter-protocol-v1.json", "evals/v4-adapter-nomination-v1.json", "evals/v4-nomination-v1.json"]:
        if (ROOT / name).exists(): paths.append(name); allowlisted.add(name)
    for path in (ROOT / "artifacts/v4-adapters").glob("*/adapter/*"):
        if path.is_file() and path.name in {"config.json", "adapter.safetensors"}:
            name = str(path.relative_to(ROOT)); paths.append(name); allowlisted.add(name)
    for path in (ROOT / "artifacts/v4-selection").glob("*/*"):
        if path.is_file() and path.name in {"config.json", "readout.safetensors"}:
            name = str(path.relative_to(ROOT)); paths.append(name); allowlisted.add(name)
    for row in rows:
        for item in row["media"]:
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Public media hash mismatch")
            paths.append(item["path"])
    hashes = {}
    for name in sorted(set(paths)):
        if not (name.startswith("data/") or name in allowlisted):
            raise ValueError("Outside upload allowlist")
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        hashes[name] = sha(target)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record = {"source_commit": source, "files": hashes,
              "bytes": sum((DEST / name).stat().st_size for name in hashes),
              "contains_private_data_or_credentials": False}
    (DEST / "bundle.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"files": len(hashes), "bytes": record["bytes"], "source_commit": source}))


if __name__ == "__main__":
    main()
