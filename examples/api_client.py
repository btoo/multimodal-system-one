"""Run after `uv run mmso joint-prepare` and `uv run mmso serve --device cpu`."""
import argparse
import json

from mmso.api.client import Client, media_block
from mmso.api import MODEL_ID
from mmso.artifacts import ROOT


def example_payload():
    demo = json.loads((ROOT / "examples/joint-demo-input.json").read_text())
    color, opposite, _, present, _ = demo["requests"]
    return {"model": MODEL_ID, "input": [media_block(ROOT / demo["image"], "image", "image/png"),
                      media_block(ROOT / demo["audio"], "audio", "audio/wav"),
                      {"type": "text", "id": "presence", "text": present["question"]}],
            "questions": {
                "color": {"type": "choice", "question": color["question"], "choices": color["candidates"]},
                "opposite_ranked": {"type": "ranking", "question": opposite["question"], "choices": opposite["candidates"]},
                "present": {"type": "noul", "question_ref": "presence"},
                "presence_score": {"type": "score", "question_ref": "presence", "levels": [
                    {"id": "absent", "text": "no", "value": 0},
                    {"id": "present", "text": "yes", "value": 1}]}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--write-request", help="Write inline JSON suitable for curl and exit")
    args = parser.parse_args()
    payload = example_payload()
    if args.write_request:
        from pathlib import Path
        Path(args.write_request).write_text(json.dumps(payload) + "\n")
    else:
        print(json.dumps(Client(args.base_url).decide(**payload), indent=2))
