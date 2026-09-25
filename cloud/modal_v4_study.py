"""Bounded MiSO v4 study: CPU downloads, one H100, no public endpoint.

Launch individual jobs with an immutable attempt ID. The local runner reserves
the maximum allocation proxy before each job and limits the whole study to 24
GPU jobs. Runtime limits do not replace Modal's own workspace budget.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / ".research/v4/bundle"
REGISTRY = ROOT / "evals/v4-candidates-v1.json"
app = modal.App("miso-v4-selection")
volume = modal.Volume.from_name("miso-v4-selection-v1", create_if_missing=True)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("huggingface-hub==1.32.0")
             .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/workspace:/workspace/cloud"})
             .add_local_file(REGISTRY, "/registry.json")
             .add_local_file(Path(__file__), "/workspace/cloud/modal_v4_study.py"))


def gpu_image(legacy=False):
    transformers_version, torch_version, torchvision_version = ("4.51.0", "2.8.0", "0.23.0") if legacy else ("5.17.0", "2.14.0", "0.29.0")
    hub_version, accelerate_version, peft_version = ("0.36.0", "1.6.0", "0.15.2") if legacy else ("1.32.0", "1.15.0", "0.21.0")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0", "libsndfile1", "ffmpeg")
            .uv_pip_install(f"torch=={torch_version}", f"torchvision=={torchvision_version}",
                            f"transformers=={transformers_version}", f"accelerate=={accelerate_version}",
                            "numpy==2.5.3", "Pillow==12.3.0", "scipy==1.18.1",
                            "soundfile==0.14.0", "librosa==0.11.0", "sentencepiece==0.2.2",
                            "einops==0.8.2", "safetensors==0.8.0", f"peft=={peft_version}",
                            "timm==1.0.30", f"huggingface-hub=={hub_version}", "backoff==2.2.1")
            # Only the package marker is needed by HF's recursive import check;
            # video and TTS utilities are never invoked in this decision study.
            # Preserve the common pinned Pillow/librosa/Torch runtime.
            .uv_pip_install("minicpmo-utils==1.0.6", extra_options="--no-deps")
            .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/workspace:/workspace/cloud",
                  "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
            .add_local_dir(BUNDLE, "/workspace"))


@app.function(image=cpu_image, volumes={"/cache": volume}, cpu=2, memory=4096,
              timeout=1800, startup_timeout=120, retries=0, max_containers=1,
              min_containers=0, scaledown_window=2, include_source=False)
def download_model(key):
    import hashlib
    from huggingface_hub import snapshot_download
    registry = json.loads(Path("/registry.json").read_text())
    spec = next(m for m in registry["models"] if m["key"] == key)
    if spec["gated"]:
        raise ValueError("Gated model has no authorized local credential")
    started = time.perf_counter()
    path = Path(snapshot_download(spec["id"], revision=spec["revision"],
                     allow_patterns=["*.json", "*.safetensors", "*.py", "*.model", "*.txt", "*.jinja", "*.tiktoken", "README.md", "LICENSE*"],
                     max_workers=8))
    files = []
    for file in sorted(path.rglob("*")):
        if file.is_file():
            files.append({"name": str(file.relative_to(path)), "bytes": file.stat().st_size})
    result = {"key": key, "model": spec["id"], "revision": spec["revision"],
              "snapshot": str(path), "download_seconds": time.perf_counter() - started,
              "total_bytes": sum(f["bytes"] for f in files), "files": files}
    meta = Path("/cache/study/downloads") / (key + ".json")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


def run_remote(key, phase, attempt):
    import sys
    sys.path.insert(0, "/workspace")
    from mmso.backbone_study import run_candidate
    volume.reload()
    output = Path("/cache/study/attempts") / attempt
    if output.exists():
        raise ValueError("Attempt ID already exists; never overwrite a run")
    output.mkdir(parents=True)
    try:
        result = run_candidate(Path("/workspace"), key, phase, output)
    except Exception as error:
        import traceback
        result = {"status": "failed", "key": key, "phase": phase,
                  "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        (output / "failure.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    files = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    return {"summary": result, "files": files}


@app.function(image=gpu_image(), gpu="H100!", cpu=4, memory=65536,
              volumes={"/cache": volume}, timeout=3600, startup_timeout=600,
              retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, single_use_containers=True, include_source=False)
def evaluate(key, phase, attempt):
    return run_remote(key, phase, attempt)


@app.function(image=gpu_image(legacy=True), gpu="H100!", cpu=4, memory=65536,
              volumes={"/cache": volume}, timeout=3600, startup_timeout=600,
              retries=0, max_containers=1, min_containers=0,
              scaledown_window=2, single_use_containers=True, include_source=False)
def evaluate_legacy(key, phase, attempt):
    return run_remote(key, phase, attempt)


@app.local_entrypoint()
def main(key: str = "gemma4-e2b", phase: str = "smoke", attempt: str = "gemma4-e2b-smoke-v1"):
    if phase not in {"download", "download-all", "smoke", "development", "confirmation", "adapter-development", "adapter-confirmation"}:
        raise ValueError("Invalid phase")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", attempt):
        raise ValueError("Invalid attempt ID")
    report = ROOT / "reports/v4-selection-v1/attempts" / attempt
    if report.exists():
        raise ValueError("Local attempt exists")
    report.mkdir(parents=True)
    started = time.time()
    if phase == "download-all":
        registry = json.loads(REGISTRY.read_text())
        for model in registry["models"]:
            if model["gated"]: continue
            result = download_model.remote(model["key"])
            (report / (model["key"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"phase": phase, "key": model["key"], "bytes": result["total_bytes"]}), flush=True)
        return
    if phase == "download":
        result = download_model.remote(key)
        (report / "download.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"phase": phase, "key": key, "bytes": result["total_bytes"]})); return
    attempts = list((ROOT / "reports/v4-selection-v1/attempts").glob("*/reservation.json"))
    if len(attempts) >= 24:
        raise ValueError("Study GPU job count exhausted")
    reserved = (3600 + 600) * (0.001097 + 4 * 0.0000131 + 64 * 0.00000222)
    spent_reservations = sum(json.loads(p.read_text())["maximum_compute_proxy_usd"] for p in attempts)
    if spent_reservations + reserved > 250:
        raise ValueError("Study allocation ceiling exhausted")
    if phase in {"confirmation", "adapter-confirmation"} and not (ROOT / "evals/v4-nomination-v1.json").exists():
        raise ValueError("Freeze a nomination before confirmation")
    (report / "reservation.json").write_text(json.dumps({"key": key, "phase": phase, "started_unix": started,
         "maximum_compute_proxy_usd": reserved, "note": "Conservative runtime reservation, not actual billed spend"}, indent=2) + "\n")
    runner = evaluate_legacy if key in {"minicpmo45", "phi4mm"} else evaluate
    call = runner.spawn(key, phase, attempt)
    try:
        result = call.get(timeout=4300)
    except BaseException:
        call.cancel(terminate_containers=True)
        raise
    for name, payload in result["files"].items():
        if name == "features.npz":
            path = ROOT / ".research/v4/features" / (attempt + ".npz")
        elif name.startswith("adapter/"):
            path = ROOT / "artifacts/v4-adapters" / key / name
        else:
            path = report / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (report / "rpc.json").write_text(json.dumps({"wall_seconds": time.time() - started,
          "maximum_compute_proxy_usd": reserved}, indent=2) + "\n")
    summary = result["summary"]
    print(json.dumps({k: summary.get(k) for k in ("status", "key", "phase", "completed", "errors", "error_type", "error")}), flush=True)
    if summary.get("status") != "completed": raise SystemExit(1)
