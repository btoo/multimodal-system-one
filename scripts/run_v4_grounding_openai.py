"""Same-task coordinate/chart/BoolQ reference within the existing $2 cap."""
from pathlib import Path
import base64
import io
import json
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from PIL import Image

from mmso.backbone_study import digest
from mmso.grounding_tasks import safe_input,task_prompt,parse_point
from mmso.grounding_training import score_result,summarize
from mmso.reference_costs import openai_cost

ROOT=Path(__file__).resolve().parents[1]


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key-file',required=True);parser.add_argument('--attempt',default='openai-sol-development-v1');args=parser.parse_args()
    if Path(args.attempt).name!=args.attempt:raise ValueError('Invalid attempt')
    protocol=json.loads((ROOT/'evals/v4-grounding-training-protocol-v1.json').read_text())
    acquisition=json.loads((ROOT/'evals/acquisition/v4_grounding_training_v1.json').read_text())
    manifest=ROOT/'data/v4-training/manifest.jsonl'
    if digest(manifest)!=acquisition['manifest_sha256']:raise ValueError('Manifest changed')
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    rows=[r for r in rows if r['split']=='development' and r['dataset'] in {'screenspot_pro','chartqa','boolq'}]
    rows.sort(key=lambda r:__import__('hashlib').sha256(r['id'].encode()).hexdigest())
    folder=ROOT/'reports/v4-grounding-training-v1/attempts'/args.attempt
    if folder.exists():raise ValueError('Immutable attempt exists')
    path=Path(args.key_file).expanduser()
    if path.stat().st_mode & 0o077:raise ValueError('Credential must be owner-only')
    key=json.loads(path.read_text())['api_key'];folder.mkdir(parents=True)
    spec={'model':'gpt-6-sol','reasoning_effort':'none','input_usd_per_million':2,'output_usd_per_million':10}
    ledgers=[ROOT/'reports/v4-iteration-v2/openai-spend-ledger.jsonl',ROOT/'reports/v4-grounding-training-v1/openai-spend-ledger.jsonl']
    prior=sum(json.loads(s)['charged_upper_usd'] for p in ledgers if p.exists() for s in p.read_text().splitlines());spent=prior;stop=None;records=[]
    (folder/'request-protocol.json').write_text(json.dumps({'model':spec,'manifest_sha256':acquisition['manifest_sha256'],'cases':len(rows),
        'selection':'All development ScreenSpot-Pro, ChartQA and BoolQ cases. No training or confirmation requests.',
        'max_output_tokens':64,'image_max_side':1536,'image_detail':'high','concurrency':1,'previous_openai_upper_usd':prior,'cumulative_ceiling_usd':2},indent=2)+'\n')
    for row in rows:
        safe=safe_input(row);started=time.perf_counter();result={'status':'not_run_after_stop','id':row['id'],'task':row['task'],'dataset':row['dataset']}
        if stop is None:
            prompt=task_prompt(safe);content=[]
            for item in safe['media']:
                if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Media changed')
                with Image.open(ROOT/item['path']) as im:image=im.convert('RGB')
                image.thumbnail((1536,1536),Image.Resampling.LANCZOS);buf=io.BytesIO();image.save(buf,format='PNG')
                content.append({'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),'detail':'high'})
            if row['task']=='point':properties={k:{'type':'number','minimum':0,'maximum':1} for k in ('x','y')}
            elif row['task']=='chart':properties={'answer':{'type':'string'}};prompt+=' Return the value in the required JSON answer field.'
            else:properties={'choice':{'type':'string','enum':list('ABCDEFGHIJ'[:len(row['choices'])])}};prompt+=' Return the selected letter in the required JSON choice field.'
            content.append({'type':'input_text','text':prompt})
            payload={'model':spec['model'],'service_tier':'default','reasoning':{'effort':'none'},'max_output_tokens':64,'store':False,
                'input':[{'role':'user','content':content}], 'text':{'format':{'type':'json_schema','name':'answer','strict':True,
                'schema':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}}
            reservation=((len(prompt.encode())+2048+16384*len(safe['media']))*2*1.25+64*10)/1e6
            if spent+reservation>2:stop='budget';result['status']='not_run_budget'
            else:
                charge=reservation;usage=None
                try:
                    request=Request('https://api.openai.com/v1/responses',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                    with urlopen(request,timeout=90) as response:raw=json.load(response)
                    usage=raw['usage'];cost=openai_cost(usage,spec);charge=cost['conservative_upper_usd']
                    if charge>reservation:raise ValueError('Usage exceeded reservation')
                    texts=[part['text'] for item in raw.get('output',[]) if item.get('type')=='message' for part in item.get('content',[]) if part.get('type')=='output_text']
                    result.update(status='incomplete_or_refused',schema_valid=False,usage=usage,cost_upper_usd=charge,
                        standard_rate_estimate_usd=cost['standard_rate_estimate_usd'],model=raw.get('model'),service_tier=raw.get('service_tier'),
                        timings={'request_ms':(time.perf_counter()-started)*1000},response_id=raw.get('id'))
                    if raw.get('status')=='completed' and texts:
                        text=''.join(texts);value=json.loads(text)
                        if row['task']=='point':result.update(status='ok',text=text,point=parse_point(text),schema_valid=parse_point(text) is not None)
                        elif row['task']=='chart':result.update(status='ok',text=value['answer'],schema_valid=True)
                        else:result.update(status='ok',choice=ord(value['choice'])-65,schema_valid=True)
                except HTTPError as error:
                    result.update(status='http_error',http_status=error.code);stop='http_error'
                except Exception as error:
                    result.update(status='error',error_type=type(error).__name__);stop='error'
                spent+=charge
                with ledgers[-1].open('a') as stream:stream.write(json.dumps({'attempt':args.attempt,'id':row['id'],'charged_upper_usd':charge,'usage_known':usage is not None,'reservation_usd':reservation})+'\n')
        if row['task']=='choice' and result['status']=='ok':result['correct']=result['choice']==row['target']
        else:result.update(score_result(row,result))
        records.append(result)
        with (folder/'predictions.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
        if len(records)%25==0:print(json.dumps({'completed':len(records),'total':len(rows),'cumulative_upper_usd':spent,'stop':stop}),flush=True)
    summary={'status':'completed' if stop is None else 'stopped','stop_reason':stop,'model':spec,'metrics':summarize(rows,records),
        'new_cost_upper_usd':spent-prior,'cumulative_openai_upper_usd':spent,'manifest_sha256':acquisition['manifest_sha256'],
        'timing_boundary':'Full local media preparation and API client request through parsed answer, including network/provider work; warm/cold provider behavior is not controlled.'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
