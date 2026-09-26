"""Acquire pinned public sources without loading model weights locally."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
FILES={
    'omniact':['train.json','data.zip'],
    'chartqa':['data/train-00000-of-00003-49492f364babfa44.parquet','data/val-00000-of-00001-0f11003c77497969.parquet','data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet'],
    'boolq':['data/train-00000-of-00001.parquet','data/validation-00000-of-00001.parquet'],
    'screenspot_v2':['screenspot_desktop_v2.json','screenspot_web_v2.json','screenspotv2_image.zip'],
}


def main():
    protocol=json.loads((ROOT/'evals/v4-grounding-training-protocol-v1.json').read_text())
    jobs=[(protocol['sources'][key],file) for key,files in FILES.items() for file in files]
    def fetch(job):
        spec,name=job
        path=hf_hub_download(spec['repo'],name,repo_type='dataset',revision=spec['revision'],local_dir=ROOT/'data/v4-training/sources'/spec['repo'].replace('/','--'))
        print(json.dumps({'repo':spec['repo'],'file':name,'bytes':Path(path).stat().st_size}),flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(fetch,jobs))


if __name__=='__main__':main()
