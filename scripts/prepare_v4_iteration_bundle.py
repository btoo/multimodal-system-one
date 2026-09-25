"""Upload only pinned code, approved public data and the existing adapter."""
from pathlib import Path
import json
import shutil
import subprocess

from mmso.backbone_study import digest

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.research/v4-iteration/bundle'


def main():
    names = ['mmso/__init__.py', 'mmso/backbone_study.py', 'mmso/backbone_adapters.py',
        'mmso/backbone_diagnostics.py', 'mmso/parallel_decisions.py', 'mmso/shared_observation.py',
        'mmso/frontier_evals.py', 'mmso/v4_iteration.py', 'cloud/modal_v4_iteration.py', 'cloud/modal_v4_study.py',
        'evals/v4-candidates-v1.json', 'evals/v4-iteration-protocol-v2.json', 'evals/manifests/v4_selection_v1.jsonl',
        'evals/acquisition/v4_frontier_v1.json', 'data/v4-frontier/manifest.jsonl',
        'artifacts/v4-adapters/minicpmo45/adapter/config.json', 'artifacts/v4-adapters/minicpmo45/adapter/adapter.safetensors',
        'artifacts/v4-selection/minicpmo45-adapted/config.json']
    rows = [json.loads(s) for s in (ROOT / 'data/v4-frontier/manifest.jsonl').read_text().splitlines()]
    rows += [json.loads(s) for s in (ROOT / 'evals/manifests/v4_selection_v1.jsonl').read_text().splitlines() if json.loads(s)['split'] == 'development']
    for name in ('evals/manifests/v4_fresh_screens_v2.jsonl', 'evals/v4-screen-nomination-v2.json', 'evals/v4-iteration-amendment-v1.json'):
        if (ROOT / name).exists(): names.append(name)
    if (ROOT / 'evals/manifests/v4_fresh_screens_v2.jsonl').exists():
        rows += [json.loads(s) for s in (ROOT / 'evals/manifests/v4_fresh_screens_v2.jsonl').read_text().splitlines()]
    for row in rows:
        for item in row['media']:
            if not item['path'].startswith('data/') or '..' in Path(item['path']).parts: raise ValueError('Unexpected media path')
            if digest(ROOT / item['path']) != item['sha256']: raise ValueError('Media hash changed')
            names.append(item['path'])
    hashes = {}
    for name in sorted(set(names)):
        target = DEST / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target); hashes[name] = digest(target)
    record = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'files': hashes, 'bytes': sum((DEST / name).stat().st_size for name in hashes),
        'contains_private_data_or_credentials': False}
    (DEST / 'bundle.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'files': len(hashes), 'bytes': record['bytes'], 'source_commit': record['source_commit']}))


if __name__ == '__main__': main()
