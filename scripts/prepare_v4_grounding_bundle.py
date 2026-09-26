"""Build immutable, content-addressed allowlisted Modal input snapshots."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from mmso.backbone_study import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=['mmso/__init__.py','mmso/backbone_study.py','mmso/backbone_adapters.py','mmso/frontier_evals.py',
        'mmso/grounding_tasks.py','mmso/grounding_training.py','cloud/modal_v4_study.py','cloud/modal_v4_grounding.py',
        'evals/v4-candidates-v1.json','evals/v4-grounding-training-protocol-v1.json',
        'evals/acquisition/v4_grounding_training_v1.json','data/v4-training/manifest.jsonl']
    for name in ['evals/v4-grounding-training-nomination-v1.json','evals/v4-grounding-confirmation-nomination-v1.json','evals/v4-native-coordinate-reference-v1.json','evals/manifests/v4_joint_regression_v1.jsonl','evals/v4-joint-regression-amendment-v1.json']:
        if (ROOT/name).exists():paths.append(name)
    for file in (ROOT/'artifacts/v4-grounding-training').glob('*/adapter/*'):
        if file.name in {'config.json','adapter.safetensors'}:paths.append(str(file.relative_to(ROOT)))
    for line in (ROOT/'data/v4-training/manifest.jsonl').read_text().splitlines():
        for item in json.loads(line)['media']:
            if not item['path'].startswith('data/') or '..' in Path(item['path']).parts:raise ValueError('Invalid upload path')
            if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Media changed')
            paths.append(item['path'])
    joint=ROOT/'evals/manifests/v4_joint_regression_v1.jsonl'
    if joint.exists():
        for line in joint.read_text().splitlines():
            for item in json.loads(line)['media']:
                if not item['path'].startswith('data/') or '..' in Path(item['path']).parts:raise ValueError('Invalid joint media path')
                if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Joint media changed')
                paths.append(item['path'])
    hashes={p:digest(ROOT/p) for p in sorted(set(paths))}
    identifier=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:20]
    base=ROOT/'.research/v4-grounding';dest=base/'bundles'/identifier
    size=sum((ROOT/p).stat().st_size for p in hashes)
    if not dest.exists() and shutil.disk_usage(ROOT).free<size+3*1024**3:raise ValueError('Insufficient free disk for snapshot')
    dest.mkdir(parents=True,exist_ok=True)
    for name,fingerprint in hashes.items():
        target=dest/name
        if target.exists():
            if digest(target)!=fingerprint:raise ValueError('Immutable bundle changed')
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'bundle_id':identifier,'files':hashes,'bytes':size,'credentials_included':False}
    if not (dest/'bundle.json').exists():(dest/'bundle.json').write_text(json.dumps(record,indent=2)+'\n')
    (base/'current-bundle.txt').write_text(identifier+'\n')
    print(json.dumps({'bundle_id':identifier,'files':len(hashes),'bytes':size}))


if __name__=='__main__':main()
