"""Verify joint reports from saved probabilities, generated requests, and Git source blobs."""
import hashlib
import json
import math
import subprocess
import numpy as np

from mmso.artifacts import ROOT,read_manifest,sha256
from mmso.joint_data import MANIFEST,expand_scenes
from mmso.joint_training import decision_metrics
from mmso.metrics import aligned_predictions


def close(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:close(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-8),(a,b)
    else:assert a==b,(a,b)


scenes=read_manifest(MANIFEST)
for name in ['joint-full-v1','joint-full-v2','joint-audio-only-v2','joint-image-only-v2']:
    train=json.loads((ROOT/'reports'/name/'training.json').read_text())
    assert sha256(ROOT/train['checkpoint'])==train['checkpoint_sha256']
    prov=train['provenance']
    assert prov['manifest_sha256']==sha256(MANIFEST)
    for path,digest in prov['source_hashes'].items():
        blob=subprocess.check_output(['git','show',f"{prov['git_revision']}:{path}"],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==digest,(name,path)
    development=expand_scenes([s for s in scenes if s['split']=='dev'])
    pred=aligned_predictions(development,read_manifest(ROOT/'reports'/name/'development-predictions.jsonl'))
    metrics,_=decision_metrics(development,[np.log(p['probabilities']) for p in pred])
    close(metrics,train['development'])
    evaluation=ROOT/'reports'/name/'evaluation.json'
    if evaluation.exists():
        final=json.loads(evaluation.read_text())
        test=expand_scenes([s for s in scenes if s['split']=='test'])
        pred=aligned_predictions(test,read_manifest(ROOT/'reports'/name/'predictions.jsonl'))
        logits=[np.log(p['probabilities']) for p in pred]
        calibrated,_=decision_metrics(test,logits)
        raw,_=decision_metrics(test,[z*final['temperature'] for z in logits])
        close(calibrated,final['calibrated']);close(raw,final['raw'])
        assert final['checkpoint_sha256']==train['checkpoint_sha256']
        for path,digest in final['provenance']['source_hashes'].items():
            blob=subprocess.check_output(['git','show',f"{final['provenance']['git_revision']}:{path}"],cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest()==digest,(name,path)
nomination=json.loads((ROOT/'evals/joint-nomination-v2.json').read_text())
for name in nomination['controls']:
    train=json.loads((ROOT/'reports'/name/'training.json').read_text())
    assert train['configuration']['actual_steps']==nomination['control_optimizer_steps']
    assert train['configuration']['resample_training_pairs'] is True
print('Verified joint checkpoint hashes, source revisions, exact prediction coverage, development/final metrics, and matched control steps.')
