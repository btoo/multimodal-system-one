"""Restore exactly the frozen study assets; never select new confirmation data."""
from contextlib import ExitStack
from io import BytesIO
import hashlib
import tarfile
import zipfile

from mmso.artifacts import ROOT, local_media_path, read_manifest, sha256
from mmso.data import MINI_URL, MINI_SHA256, download
from mmso.joint_world import render_panel
from mmso.optimization_study import ARCHIVE_URL, ARCHIVE_SHA, MANIFEST, audit_optimization


def restore_assets(scenes):
    restored = 0
    with ExitStack() as stack:
        mini = official = None
        for scene in scenes:
            for kind in ("audio", "image"):
                media = scene[kind]
                path = local_media_path(ROOT, media["path"])
                if not path.is_relative_to(ROOT / "data") or path.suffix != (".wav" if kind == "audio" else ".png"):
                    raise ValueError("Frozen assets must be WAV/PNG files under data/")
                if path.exists():
                    if sha256(path) != media["sha256"]: raise ValueError("Existing media differs from frozen content")
                    continue
                if kind == "image":
                    buffer = BytesIO(); render_panel(scene["panel"]).save(buffer, format="PNG")
                    payload = buffer.getvalue()
                elif media["path"].startswith("data/optimization-v1/audio/"):
                    if official is None:
                        archive = download(ARCHIVE_URL, ROOT / "data/downloads/speech_commands_test_set_v0.02.tar.gz",
                                           expected_sha=ARCHIVE_SHA, max_bytes=113_000_000)
                        official = stack.enter_context(tarfile.open(archive))
                    member = official.getmember("./" + scene["audio_word"] + "/" + path.name)
                    if not member.isfile() or member.size > 128_000: raise ValueError("Invalid frozen audio member")
                    payload = official.extractfile(member).read()
                else:
                    if mini is None:
                        archive = download(MINI_URL, ROOT / "data/downloads/mini_speech_commands.zip",
                                           expected_sha=MINI_SHA256, max_bytes=190_000_000)
                        mini = stack.enter_context(zipfile.ZipFile(archive))
                    payload = mini.read("mini_speech_commands/" + scene["audio_word"] + "/" + path.name)
                if hashlib.sha256(payload).hexdigest() != media["sha256"]:
                    raise ValueError("Restored bytes differ from the frozen asset")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload); restored += 1
    return restored


if __name__ == "__main__":
    scenes = read_manifest(MANIFEST)
    count = restore_assets(scenes)
    audit = audit_optimization(scenes)
    print({"restored_assets": count, "scenes": audit["scenes"], "confirmation_speakers": audit["confirmation_speakers"]})
