"""Restore the exact public v4 media without changing the frozen manifest."""
from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
from urllib.parse import quote

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import soundfile as sf

from mmso.artifacts import ROOT, sha256
from mmso.data import download, prepare_sounds, prepare_speech, SCREEN_REVISION


def main():
    manifest = ROOT / "evals/manifests/v4_selection_v1.jsonl"
    audit = json.loads((ROOT / "evals/acquisition/v4_selection_v1.json").read_text())
    if sha256(manifest) != audit["manifest_sha256"]: raise ValueError("Frozen manifest changed")
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    missing = [m for r in rows for m in r["media"] if not (ROOT / m["path"]).exists()]
    if any(m["path"].startswith("data/speech_keywords/") for m in missing): prepare_speech()
    if any(m["path"].startswith("data/sounds/") for m in missing): prepare_sounds()
    slurp = audit["slurp"]
    for source_split, source in slurp["source_shards"].items():
        selected = [r for r in rows if r["track"] == "speech_intent" and r["source_split"] == source_split
                    and not (ROOT / r["media"][0]["path"]).exists()]
        if not selected: continue
        relative = source["path"].removeprefix("data/v4/slurp/")
        shard = Path(hf_hub_download(slurp["repo"], relative, repo_type="dataset", revision=slurp["revision"],
                                    local_dir=ROOT / "data/v4/slurp"))
        if sha256(shard) != source["sha256"]: raise ValueError("SLURP shard hash mismatch")
        wanted = {int(r["id"].split(":")[1]): r for r in selected}
        for record in pq.read_table(shard, columns=["slurp_id", "audio"]).to_pylist():
            row = wanted.pop(record["slurp_id"], None)
            if row is None: continue
            waveform, sr = sf.read(io.BytesIO(record["audio"]["bytes"]), dtype="float32")
            target = ROOT / row["media"][0]["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, waveform, sr, subtype="PCM_16")
        if wanted: raise ValueError("SLURP identity not found in pinned source shard")
    base = f"https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/{SCREEN_REVISION}/"
    for row in rows:
        for item in row["media"]:
            path = Path(item["path"])
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "data":
                raise ValueError("Outside media restore allowlist")
            target = ROOT / path
            if not target.exists():
                if item["path"].startswith("data/screens_v2/"):
                    relative = item["path"].removeprefix("data/screens_v2/")
                    download(base + quote(relative, safe="/"), target, expected_sha=item["sha256"], max_bytes=25 << 20)
                elif item["path"].startswith("data/v4/media/joint/"):
                    source = ROOT / "evals/fixtures/v4-panels" / target.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                else:
                    raise ValueError(f"Missing source adapter for {item['path']}")
            if sha256(target) != item["sha256"]: raise ValueError(f"Media fingerprint differs: {item['path']}")
    print(json.dumps({"status": "restored_and_verified", "cases": len(rows), "manifest_unchanged": sha256(manifest) == audit["manifest_sha256"]}))


if __name__ == "__main__":
    main()
