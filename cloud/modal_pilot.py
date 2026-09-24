"""Run two serial, single-use L4 containers; no public serving deployment."""
import json
from pathlib import Path
import re
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / ".research/modal/bundle"
app = modal.App("mmso-modal-pilot")
volume = modal.Volume.from_name("mmso-pilot-checkpoints-v1", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "numpy==2.5.3", "Pillow==12.3.0", "scipy==1.18.1", "safetensors==0.8.0")
         .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": "/workspace"})
         .add_local_dir(BUNDLE, remote_path="/workspace"))


@app.function(image=image, gpu="L4", cpu=(1, 2), memory=(2048, 4096),
              volumes={"/checkpoints": volume}, timeout=180, startup_timeout=120,
              max_containers=1, min_containers=0, scaledown_window=2, retries=0,
              single_use_containers=True, include_source=False)
def pilot(phase: str, run_id: str):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    sys.path.insert(0, "/workspace")
    from mmso.cloud_pilot import run_phase
    volume.reload()
    result = run_phase(phase, Path("/checkpoints") / run_id)
    volume.commit()
    return result


@app.local_entrypoint()
def main(run_id: str = "modal-l4-pilot-v1"):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    report = ROOT / "reports" / run_id
    report.mkdir(parents=True, exist_ok=True)
    if (report / "result.json").exists():
        raise ValueError("Completed local run ID is immutable")
    bundle = json.loads((BUNDLE / "bundle.json").read_text())
    (report / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    results = {}
    for phase in ("reference", "resume"):
        started = time.perf_counter()
        call = pilot.spawn(phase, run_id)
        try:
            result = call.get(timeout=330)
        except BaseException:
            call.cancel(terminate_containers=True)
            raise
        result["local_rpc_wall_seconds"] = time.perf_counter() - started
        results[phase] = result
        (report / f"{phase}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"phase": phase, "seconds": result["phase_wall_seconds"],
                          "resume": result.get("resume"), "training_seconds": result["training"]["training_seconds"]}), flush=True)
    summary = {"status": "cloud_portability_and_resume_verified", "phases": list(results),
               "source_commit": bundle["source_commit"], "accuracy_trial": False,
               "limits": {"gpu": "L4", "concurrent_containers": 1, "phases": 2,
                          "function_timeout_seconds": 180, "startup_timeout_seconds": 120, "retries": 0},
               "cost": {"budget_target_usd": 1, "actual_billed_usd": None,
                        "note": "Billing must be verified separately; runtime limits are not an account spending cap."}}
    (report / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved evidence to {report.relative_to(ROOT)}")
