"""Measure a registered model on 32 varied, already-opened confirmation scenes."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from benchmark_api import percentile
from mmso.api.client import media_block
from mmso.artifacts import ROOT, read_manifest, sha256, write_json, write_manifest
from mmso.data import rank
from mmso.joint_data import questions_for_scene
from mmso.joint_model import predict_joint
from mmso.joint_world import WORDS
from mmso.optimization_study import MANIFEST, PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    host = urlsplit(args.base_url)
    if host.scheme != "http" or host.hostname not in {"localhost", "127.0.0.1", "::1"} or host.username or host.password or host.query or host.fragment or host.path not in {"", "/"}:
        parser.error("Use a loopback HTTP API base URL")
    if args.output_dir.exists(): parser.error("Use a fresh immutable output directory")
    for run in json.loads(PROTOCOL.read_text())["conditions"]:
        if not (ROOT / "reports" / run / "evaluation.json").exists():
            parser.error("Complete the frozen accuracy confirmation before timing these scenes")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("MMSO_API_KEY"): headers["Authorization"] = "Bearer " + key
    def request(route, data=None):
        with urlopen(Request(args.base_url.rstrip("/") + route, data=data, headers=headers), timeout=30) as response:
            return json.load(response)
    card = request("/v1/models/" + args.model)
    if card["checkpoint_sha256"] != sha256(args.checkpoint): raise ValueError("Local and served checkpoints differ")
    scenes = [s for s in read_manifest(MANIFEST) if s["split"] == "test"]
    selected = []
    for word in WORDS:
        for slice_name in ("in_distribution", "compositional"):
            pool = sorted([s for s in scenes if s["audio_word"] == word and s["slice"] == slice_name], key=lambda s: rank("http-timing:" + s["id"]))
            selected.extend(pool[:2])
    assert len(selected) == 32 and len({s["id"] for s in selected}) == 32
    payloads = []; neural_requests = []
    for scene in selected:
        tasks = {r["task"]: r for r in questions_for_scene(scene)}
        color, position, present = [tasks[k] for k in ("color", "position", "present")]
        questions = {
            "color": {"type": "choice", "question": color["question"], "choices": color["candidates"]},
            "position": {"type": "ranking", "question": position["question"], "choices": position["candidates"]},
            "present": {"type": "noul", "question": present["question"]},
            "presence_score": {"type": "score", "question": present["question"], "levels": [
                {"id": "no", "text": "no", "value": 0.}, {"id": "yes", "text": "yes", "value": 1.}]} }
        payload = {"model": args.model, "input": [media_block(ROOT / scene["image"]["path"], "image", "image/png"),
                   media_block(ROOT / scene["audio"]["path"], "audio", "audio/wav")], "questions": questions}
        payloads.append(json.dumps(payload, allow_nan=False).encode())
        neural_requests.append([{"question": q["question"], "candidates": q["choices"]} for q in [questions["color"], questions["position"]]] +
            [{"question": present["question"], "candidates": [{"id": "true", "text": "yes"}, {"id": "false", "text": "no"}]},
             {"question": present["question"], "candidates": [{"id": "no", "text": "no"}, {"id": "yes", "text": "yes"}]}])
    for payload in payloads[:5]: request("/v1/decisions", payload)
    durations = []; responses = []
    for scene, payload in zip(selected, payloads):
        start = time.perf_counter_ns(); result = request("/v1/decisions", payload)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        assert result["model"] == args.model and result["checkpoint_sha256"] == card["checkpoint_sha256"]
        assert set(result["results"]) == {"color", "position", "present", "presence_score"}
        durations.append(elapsed)
        responses.append({"scene_id": scene["id"], "audio_sha256": scene["audio"]["sha256"], "image_sha256": scene["image"]["sha256"],
                          "elapsed_ms": elapsed, "response": result})
    # Compare every actual HTTP distribution to native inference on the first
    # three varied cases. This verification is outside the latency samples.
    import torch
    torch.set_num_threads(2)
    differences = []
    for scene, requests, response in zip(selected[:3], neural_requests[:3], responses[:3]):
        expected = predict_joint(args.checkpoint, ROOT / scene["image"]["path"], ROOT / scene["audio"]["path"], requests, device="cpu")
        for name, native in zip(("color", "position", "present", "presence_score"), expected["answers"]):
            actual = response["response"]["results"][name]
            differences.extend(abs(actual["probabilities"][key] - value) for key, value in native["probabilities"].items())
    assert max(differences) < 1e-5
    files = [*sorted((ROOT / "mmso").rglob("*.py")), ROOT / "uv.lock", Path(__file__).resolve(), ROOT / "scripts/benchmark_api.py"]
    report = {"kind": "varied_confirmation_http_benchmark", "model": args.model, "model_card": card,
              "samples": 32, "warmup_requests": 5, "typed_answers_per_request": 4, "distinct_questions_per_request": 3,
              "unique_images": len({s["image"]["sha256"] for s in selected}), "unique_audio": len({s["audio"]["sha256"] for s in selected}),
              "scene_ids": [s["id"] for s in selected], "latency_ms": {"mean": statistics.mean(durations), "p50": percentile(durations, .5),
                  "p95": percentile(durations, .95), "min": min(durations), "max": max(durations)},
              "max_native_http_probability_error": max(differences), "native_parity_examples": 3,
              "mean_recording_seconds": statistics.mean(s["audio"]["duration_seconds"] for s in selected),
              "includes": ["fresh loopback HTTP connection", "transmission", "server validation/media decoding/preprocessing",
                           "neural inference and synchronization", "typed result serialization", "client JSON decoding"],
              "excludes": ["recording time", "client file read/base64/request assembly", "server/model startup"],
              "concurrency": 1, "accuracy_evaluation": False, "timing_selection": "Two scenes per word and slice, deterministic hash order; 32 different images",
              "provenance": {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                  "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in files}, "manifest_sha256": sha256(MANIFEST),
                  "platform": platform.platform(), "torch_version": torch.__version__}}
    write_manifest(args.output_dir / "responses.jsonl", responses)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({"model": args.model, "latency_ms": report["latency_ms"], "native_parity_max_error": max(differences)}, indent=2))


if __name__ == "__main__":
    main()
