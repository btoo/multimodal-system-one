"""Measure a running local API using an explicit, already assembled request.

Example: uv run python scripts/benchmark_api.py /tmp/mmso-request.json \
    --output reports/api-v1/http-benchmark.json
The API process must be running. This is a warm HTTP measurement, not an
accuracy evaluation, load test, or microphone-to-decision measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 500 or not 1 <= args.warmup <= 100:
        parser.error("Use 1-500 measured requests and 1-100 warmup requests")
    parsed = urlsplit(args.base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        parser.error("This benchmark accepts only a loopback HTTP base URL")
    if args.output.exists():
        parser.error("Use a fresh output path; existing evidence is immutable")
    body = args.request.read_bytes()
    request_data = json.loads(body)
    model_id = request_data["model"]
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("MMSO_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    base_url = args.base_url.rstrip("/")

    def fetch(route, data=None):
        with urlopen(Request(base_url + route, data=data, headers=headers), timeout=30) as response:
            payload = response.read()
            return json.loads(payload), len(payload)

    card, _ = fetch(f"/v1/models/{model_id}")
    measured = []
    first_answer = None
    response_bytes = []
    for index in range(args.warmup + args.samples):
        start = time.perf_counter_ns()
        response, size = fetch("/v1/decisions", body)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if not response.get("results") or set(response["results"]) != set(request_data["questions"]):
            raise ValueError("Server returned incomplete question coverage")
        if first_answer is None:
            first_answer = response
        if index >= args.warmup:
            measured.append(elapsed_ms)
            response_bytes.append(size)
    source_files = [*sorted((ROOT / "mmso").rglob("*.py")), ROOT / "uv.lock", Path(__file__).resolve()]
    report = {
        "kind": "warm_local_http_benchmark",
        "schema_version": 1,
        "model": model_id,
        "model_card": card,
        "samples": args.samples,
        "warmup_requests": args.warmup,
        "questions_per_request": len(request_data["questions"]),
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "request_bytes": len(body),
        "response_bytes": response_bytes,
        "elapsed_ms": measured,
        "latency_ms": {
            "min": min(measured), "mean": statistics.mean(measured),
            "p50": percentile(measured, .5), "p95": percentile(measured, .95),
            "p99": percentile(measured, .99), "max": max(measured),
        },
        "timing_includes": ["fresh loopback HTTP connection per request", "request transmission",
                            "server validation and media decoding", "model inference and synchronization",
                            "response serialization and transmission", "client JSON decoding"],
        "timing_excludes": ["audio capture", "request assembly and base64 encoding",
                            "server startup and checkpoint loading"],
        "concurrency": 1,
        "interpretation": "Repeated fixture, warm loaded model; not an accuracy, streaming, capacity, or independent-input benchmark.",
        "first_response": first_answer,
        "provenance": {
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "python": sys.version.split()[0], "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "model": model_id, "samples": args.samples,
                      "questions_per_request": report["questions_per_request"], "latency_ms": report["latency_ms"]}, indent=2))


if __name__ == "__main__":
    main()
