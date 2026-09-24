"""Fetch only the exact pinned assets named by the checked-in fresh-screen manifest."""
import json
from pathlib import PurePosixPath
from urllib.parse import quote

from mmso.artifacts import ROOT, local_media_path, read_manifest, validate_manifest
from mmso.data import download, SCREEN_REVISION


def fetch():
    acquisition = json.loads((ROOT / "evals/acquisition/screen_grounding_v2.json").read_text())
    if acquisition["revision"] != SCREEN_REVISION:raise ValueError("Source revision changed")
    base = f"https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/{SCREEN_REVISION}/"
    remaining, fetched = 500 << 20, 0

    def one(source, destination, expected, maximum):
        nonlocal remaining, fetched
        path = local_media_path(ROOT, destination)
        existed = path.exists()
        download(base + quote(source, safe="/"), path, expected_sha=expected, max_bytes=min(maximum, remaining))
        if not existed:
            size = path.stat().st_size
            remaining -= size;fetched += size
            if remaining < 0:raise ValueError("Download budget exceeded")

    one("README.md", "data/screens_v2/README.md", acquisition["card_sha256"], 1 << 20)
    for application, expected in acquisition["annotation_sha256"].items():
        name = "annotations/" + application + ".json"
        one(name, "data/screens_v2/" + name, expected, 1 << 20)
    rows = read_manifest(ROOT / "evals/manifests/screen_grounding_v2.jsonl")
    for index, row in enumerate(rows):
        name = PurePosixPath(row["group_id"].removeprefix("screen:"))
        if name.is_absolute() or ".." in name.parts:raise ValueError("Invalid source media identity")
        one("images/" + str(name), row["media"][0]["path"], row["media"][0]["sha256"], 25 << 20)
        if (index + 1) % 8 == 0:print(f"Verified {index + 1}/{len(rows)} images", flush=True)
    print({"new_download_bytes": fetched, "audit": validate_manifest(rows)})


if __name__ == "__main__":fetch()
