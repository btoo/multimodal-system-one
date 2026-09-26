"""One-GPU coordinate/training study with immutable upload bundles."""
from pathlib import Path
import json
import re
import time

import modal
from modal_v4_study import gpu_image,volume

ROOT=Path(__file__).resolve().parents[1]
identifier=(ROOT/'.research/v4-grounding/current-bundle.txt').read_text().strip() if modal.is_local() else '0'*20
if not re.fullmatch('[0-9a-f]{20}',identifier):raise ValueError('Invalid immutable bundle identifier')
BUNDLE=ROOT/'.research/v4-grounding/bundles'/identifier
app=modal.App('miso-v4-grounding-training')


def remote(key,phase,attempt):
    import traceback
    from mmso.grounding_training import run
    volume.reload();root=Path('/workspace');out=Path('/cache/grounding-training-v1')/attempt
    if out.exists():raise ValueError('Attempt already exists')
    out.mkdir(parents=True);(out/'bundle.json').write_bytes((root/'bundle.json').read_bytes())
    try:summary=run(root,key,phase,out)
    except Exception as error:
        summary={'status':'failed','key':key,'phase':phase,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
        (out/'failure.json').write_text(json.dumps(summary,indent=2)+'\n')
    volume.commit()
    return {'summary':summary,'files':{str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}}


@app.function(image=gpu_image(legacy=True,bundle=BUNDLE),gpu='H100!',cpu=4,memory=65536,volumes={'/cache':volume},
    timeout=3600,startup_timeout=600,retries=0,max_containers=1,min_containers=0,scaledown_window=2,single_use_containers=True,include_source=False)
def mini(key,phase,attempt):return remote(key,phase,attempt)


@app.function(image=gpu_image(legacy=False,bundle=BUNDLE),gpu='H100!',cpu=4,memory=65536,volumes={'/cache':volume},
    timeout=3600,startup_timeout=600,retries=0,max_containers=1,min_containers=0,scaledown_window=2,single_use_containers=True,include_source=False)
def qwen(key,phase,attempt):return remote(key,phase,attempt)


@app.function(image=gpu_image(legacy=True,bundle=BUNDLE),gpu='H100!',cpu=4,memory=65536,volumes={'/cache':volume},
    timeout=600,startup_timeout=300,retries=0,max_containers=1,min_containers=0,scaledown_window=2,single_use_containers=True,include_source=False)
def ablation(key,phase,attempt):return remote(key,phase,attempt)


@app.local_entrypoint()
def main(key:str,phase:str,attempt:str):
    if key not in {'minicpmo45','qwen3-30ba3b'} or phase not in {'baseline','train','adapted-development','confirmation','native-coordinate-reference','image-ablation'}:raise ValueError('Unknown candidate/phase')
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,79}',attempt):raise ValueError('Invalid attempt name')
    report=ROOT/'reports/v4-grounding-training-v1/attempts'/attempt
    if report.exists():raise ValueError('Local attempt already exists')
    protocol=json.loads((ROOT/'evals/v4-grounding-training-protocol-v1.json').read_text());budget=protocol['budget']
    previous=[json.loads(p.read_text()) for p in report.parent.glob('*/reservation.json')]
    is_ablation=phase=='image-ablation'
    if is_ablation:
        amendment=json.loads((ROOT/'evals/v4-image-ablation-protocol-v1.json').read_text())
        if key!='minicpmo45' or any(r['phase']=='image-ablation' for r in previous):raise ValueError('Only one short ablation is reserved')
    regular=sum(r['phase']!='image-ablation' for r in previous)
    reserved=(900 if is_ablation else 4200)*(.001097+4*.0000131+64*.00000222)
    if (not is_ablation and regular>=budget['max_gpu_jobs']) or len(previous)>=9 or sum(r['maximum_compute_proxy_usd'] for r in previous)+reserved>budget['additional_reservation_ceiling_usd']:raise ValueError('Study budget exhausted')
    if phase in {'train','adapted-development','confirmation'}:
        name='evals/v4-grounding-'+('confirmation' if phase=='confirmation' else 'training')+'-nomination-v1.json'
        if json.loads((ROOT/name).read_text())['key']!=key:raise ValueError('Candidate not nominated')
    report.mkdir(parents=True);start=time.time()
    (report/'reservation.json').write_text(json.dumps({'key':key,'phase':phase,'started_unix':start,'maximum_compute_proxy_usd':reserved,'bundle_id':identifier},indent=2)+'\n')
    runner=ablation if is_ablation else mini if key=='minicpmo45' else qwen;call=runner.spawn(key,phase,attempt)
    try:result=call.get(timeout=4300)
    except BaseException:
        call.cancel(terminate_containers=True);raise
    for name,payload in result['files'].items():
        if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('Invalid result path')
        dest=ROOT/'artifacts/v4-grounding-training'/key/name if name.startswith('adapter/') else report/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    (report/'rpc.json').write_text(json.dumps({'wall_seconds':time.time()-start})+'\n')
    print(json.dumps({k:v for k,v in result['summary'].items() if k in ('status','key','phase','error','function_seconds')}),flush=True)
    if result['summary']['status']!='completed':raise SystemExit(1)
