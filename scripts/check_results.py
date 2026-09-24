"""Independently verify committed reports from predictions and recorded source revision."""
import hashlib
import json
import re
import subprocess
import numpy as np

from mmso.artifacts import ROOT, read_manifest, sha256
from mmso.metrics import aligned_predictions, categorical_metrics
from mmso.scoring import score_files


def main():
    runs=['speech-cnn-v1','sounds-cnn-v1','screens-clip-v1']
    for run in runs:
        report=json.loads((ROOT/'reports'/run/'report.json').read_text())
        manifest=ROOT/'evals/manifests'/f"{report['dataset']}.jsonl"
        predictions=ROOT/'reports'/run/'predictions.jsonl'
        assert sha256(manifest)==report['provenance']['manifest_sha256']
        revision=report['provenance']['git_revision']
        assert re.fullmatch('[0-9a-f]{40}',revision)
        for relative,digest in report['provenance']['source_hashes'].items():
            payload=subprocess.check_output(['git','show',f'{revision}:{relative}'],cwd=ROOT)
            assert hashlib.sha256(payload).hexdigest()==digest,(run,relative)
        if report['kind']=='categorical_audio':
            assert score_files(manifest,predictions,field='raw_probabilities',verify_media=False)['metrics']==report['test_raw']
            assert score_files(manifest,predictions,verify_media=False)['metrics']==report['test_calibrated']
            assert sha256(ROOT/report['checkpoint'])==report['checkpoint_sha256']
        else:
            for field,key in [('center_point','center_point'),('clip_grid_point','clip_grid')]:
                assert score_files(manifest,predictions,split='external_probe',field=field,verify_media=False)['metrics']==report[key]
            records=read_manifest(manifest)
            outputs=aligned_predictions(records,read_manifest(predictions))
            crop_targets=np.array([['text','icon'].index(r['ui_type']) for r in records])
            assert categorical_metrics(crop_targets,[p['oracle_crop_type_probabilities'] for p in outputs],['text','icon'])==report['oracle_crop_type']
    print('Verified three reports against complete saved predictions, pinned manifests, source commit blobs, and trained checkpoint hashes. Raw media not required for this offline check.')


if __name__=='__main__':main()
