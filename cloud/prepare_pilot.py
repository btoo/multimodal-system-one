"""Stage only allowlisted code, a public checkpoint and 64 public training scenes."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmso.artifacts import local_media_path, read_manifest, sha256, write_json, write_manifest
from mmso.cloud_pilot import CHECKPOINT


def main():
    destination = ROOT / ".research/modal/bundle"
    if destination.exists():
        raise SystemExit("Bundle already exists; archive it before preparing another run")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip():
        raise SystemExit("Commit the exact pilot sources before staging an upload")
    scenes = [s for s in read_manifest(ROOT / "evals/manifests/joint_panels_v1.jsonl") if s["split"] == "train"][:64]
    paths = {CHECKPOINT, str(Path(CHECKPOINT).with_name("config.json")), "uv.lock", "pyproject.toml"}
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "mmso").glob("*.py"))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "cloud").glob("*.py"))
    for scene in scenes:
        for media in (scene["image"], scene["audio"]):
            if sha256(local_media_path(ROOT, media["path"])) != media["sha256"]:
                raise ValueError("Training media changed")
            paths.add(media["path"])
    for relative in sorted(paths):
        source = local_media_path(ROOT, relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_manifest(destination / "pilot-scenes.jsonl", scenes)
    paths.add("pilot-scenes.jsonl")
    bundle = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "files": {p: sha256(destination / p) for p in sorted(paths)},
              "bytes": sum((destination / p).stat().st_size for p in paths),
              "data_role": "64 previously used training scenes; no new evaluation",
              "speech_source": "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
              "speech_license": "CC BY 4.0; Speech Commands, Pete Warden / TensorFlow",
              "selection": "first 64 train scenes in frozen joint_panels_v1.jsonl, eight recordings per word",
              "excluded": ["credentials", "git history", "private media", "all other datasets", "API request logs"]}
    write_json(destination / "bundle.json", bundle)
    print(json.dumps({"files": len(paths), "bytes": bundle["bytes"], "source_commit": bundle["source_commit"]}))


if __name__ == "__main__":
    main()
