"""Single-GPU bounded follow-up, reusing cached weights but separate evidence."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import modal
from modal_v4_study import gpu_image, volume

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / '.research/v4-iteration/bundle'
app = modal.App('miso-v4-iteration')


@app.function(image=gpu_image(legacy=True, bundle=BUNDLE), gpu='H100!', cpu=4, memory=65536,
              volumes={'/cache': volume}, timeout=3600, startup_timeout=600, retries=0,
              max_containers=1, min_containers=0, scaledown_window=2, single_use_containers=True, include_source=False)
def evaluate(phase, attempt):
    import traceback
    volume.reload()
    root = Path('/workspace')
    output = Path('/cache/iteration-v2') / attempt
    if output.exists(): raise ValueError('Immutable attempt already exists')
    output.mkdir(parents=True)
    (output / 'bundle.json').write_bytes((root / 'bundle.json').read_bytes())
    try:
        if phase == 'shared-observation':
            from mmso.shared_observation import run_full_pipeline_probe
            summary = run_full_pipeline_probe(root, output)
        elif phase == 'screens':
            from mmso.v4_iteration import run_screens
            summary = run_screens(root, output)
        else:
            from mmso.v4_iteration import run_frontier
            summary = run_frontier(root, phase, output)
    except Exception as error:
        summary = {'status': 'failed', 'phase': phase, 'error_type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc()}
        (output / 'failure.json').write_text(json.dumps(summary, indent=2) + '\n')
    volume.commit()
    return {'summary': summary, 'files': {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}}


@app.local_entrypoint()
def main(phase: str, attempt: str):
    if phase not in {'frontier-base', 'frontier-adapted', 'screens', 'shared-observation'}: raise ValueError('Invalid phase')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', attempt): raise ValueError('Invalid attempt name')
    protocol = json.loads((ROOT / 'evals/v4-iteration-protocol-v2.json').read_text())
    budget = protocol['budget']
    reports = ROOT / 'reports/v4-iteration-v2/attempts'
    report = reports / attempt
    if report.exists(): raise ValueError('Attempt already exists')
    reservations = [json.loads(p.read_text()) for p in reports.glob('*/reservation.json')]
    reserved = 4200 * (.001097 + 4 * .0000131 + 64 * .00000222)
    if len(reservations) >= budget['max_gpu_jobs'] or sum(r['maximum_compute_proxy_usd'] for r in reservations) + reserved > budget['max_additional_compute_reservation_usd']:
        raise ValueError('Iteration budget exhausted')
    report.mkdir(parents=True)
    started = time.time()
    (report / 'reservation.json').write_text(json.dumps({'phase': phase, 'started_unix': started,
        'maximum_compute_proxy_usd': reserved, 'note': 'Conservative reservation, not an invoice'}, indent=2) + '\n')
    call = evaluate.spawn(phase, attempt)
    try: result = call.get(timeout=4300)
    except BaseException:
        call.cancel(terminate_containers=True)
        raise
    for name, payload in result['files'].items():
        if Path(name).name != name: raise ValueError('Unexpected result path')
        (report / name).write_bytes(payload)
    (report / 'rpc.json').write_text(json.dumps({'wall_seconds': time.time()-started}) + '\n')
    print(json.dumps(result['summary']), flush=True)
    if result['summary']['status'] != 'completed': raise SystemExit(1)
