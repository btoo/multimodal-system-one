"""Budgeted public-data reference; credentials stay outside the repository."""
from collections import defaultdict
from pathlib import Path
import argparse
import base64
import io
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from mmso.backbone_study import digest
from mmso.frontier_evals import stable_rank, summarize
from mmso.reference_costs import openai_cost

ROOT = Path(__file__).resolve().parents[1]


def chosen_rows(rows, configuration):
    groups = defaultdict(list)
    for row in rows:
        if row['benchmark'] in ('mmstar','mmlu_pro'): groups[(row['benchmark'], row['stratum'])].append(row)
    selected = []
    for (benchmark, _), cases in sorted(groups.items()):
        count = 1 if configuration == 'astra-medium' else 20 if benchmark == 'mmstar' else 32
        selected.extend(sorted(cases, key=lambda r: stable_rank(r['id']))[:count])
    return selected


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--key-file',required=True)
    parser.add_argument('--configuration',choices=['sol-none','astra-medium'],required=True)
    parser.add_argument('--attempt',required=True); args=parser.parse_args()
    if Path(args.attempt).name != args.attempt: raise ValueError('Invalid attempt ID')
    protocol=json.loads((ROOT/'evals/v4-openai-protocol-v1.json').read_text()); spec=protocol['configurations'][args.configuration]
    path=Path(args.key_file).expanduser()
    if path.stat().st_mode & 0o077: raise ValueError('API credential must be owner-only')
    key=json.loads(path.read_text())['api_key']
    manifest=ROOT/'data/v4-frontier/manifest.jsonl'
    acquisition=json.loads((ROOT/'evals/acquisition/v4_frontier_v1.json').read_text())
    if digest(manifest)!=acquisition['manifest_sha256']: raise ValueError('Manifest changed')
    rows=chosen_rows([json.loads(s) for s in manifest.read_text().splitlines()],args.configuration)
    folder=ROOT/'reports/v4-iteration-v2/attempts'/args.attempt
    if folder.exists(): raise ValueError('Attempt already exists')
    folder.mkdir(parents=True)
    # Unknown-charge reservations remain charged against the same $2 ceiling.
    ledger=ROOT/'reports/v4-iteration-v2/openai-spend-ledger.jsonl'
    spent=sum(json.loads(s)['charged_upper_usd'] for s in ledger.read_text().splitlines()) if ledger.exists() else 0.
    records=[]; stop=None; billed=0.; started=time.time()
    for row in rows:
        result={'id':row['id'],'status':'not_run_after_error'}
        if stop is None:
            start=time.perf_counter()
            labels=list('ABCDEFGHIJ'[:len(row['choices'])])
            prompt=row['question']+'\n\nOptions:\n'+'\n'.join(f'{l}. {c}' for l,c in zip(labels,row['choices']))+'\n\nChoose the best option. Return its letter in the required JSON object.'
            content=[]; media_count=0
            for media in row['media']:
                if media['modality']!='image': raise ValueError('Native audio is unsupported by this reference endpoint')
                if digest(ROOT/media['path']) != media['sha256']: raise ValueError('Source media changed')
                with Image.open(ROOT/media['path']) as opened: im=opened.convert('RGB')
                im.thumbnail((1536,1536),Image.Resampling.LANCZOS)
                buf=io.BytesIO(); im.save(buf,format='PNG')
                content.append({'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),'detail':'high'})
                media_count+=1
            content.append({'type':'input_text','text':prompt})
            payload={'model':spec['model'],'service_tier':'default','reasoning':{'effort':spec['reasoning_effort']},'max_output_tokens':spec['max_output_tokens'],
                'store':False,'input':[{'role':'user','content':content}],
                'text':{'format':{'type':'json_schema','name':'decision','strict':True,
                    'schema':{'type':'object','properties':{'choice':{'type':'string','enum':labels}},'required':['choice'],'additionalProperties':False}}}}
            # Text bytes upper-bound BPE token count, with schema/protocol margin.
            image_bound=4096 if args.configuration=='astra-medium' else 16384
            reservation=((len(prompt.encode())+2048+media_count*image_bound)*spec['input_usd_per_million']*1.25+
                         spec['max_output_tokens']*spec['output_usd_per_million'])/1e6
            if spent+reservation>protocol['budget_usd']:
                stop='budget';result['status']='not_run_budget'
            else:
                request=Request('https://api.openai.com/v1/responses',data=json.dumps(payload).encode(),
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                charge=reservation; usage=None
                try:
                    with urlopen(request,timeout=90) as response: raw=json.load(response)
                    duration=(time.perf_counter()-start)*1000
                    usage=raw.get('usage')
                    if not usage or not isinstance(usage.get('input_tokens'),int) or not isinstance(usage.get('output_tokens'),int): raise ValueError('Missing usage')
                    cost=openai_cost(usage,spec);charge=cost['conservative_upper_usd']
                    if charge>reservation: raise ValueError('Usage exceeded conservative reservation')
                    result.update(model=raw.get('model'),usage=usage,cost_upper_usd=charge,standard_rate_cost_estimate_usd=cost['standard_rate_estimate_usd'],service_tier=raw.get('service_tier'),timings={'request_ms':duration},response_id=raw.get('id'),provider_status=raw.get('status'))
                    texts=[p['text'] for o in raw.get('output',[]) if o.get('type')=='message' for p in o.get('content',[]) if p.get('type')=='output_text']
                    if raw.get('status')=='completed' and texts:
                        answer=json.loads(''.join(texts))
                        if set(answer)!= {'choice'} or answer['choice'] not in labels: raise ValueError('Invalid structured choice')
                        result.update(status='ok',choice=labels.index(answer['choice']),probabilities=None,probability_contract_valid=None)
                    else:
                        result.update(status='incomplete_or_refused',incomplete_details=raw.get('incomplete_details'))
                except HTTPError as error:
                    body=error.read(10000)
                    try: detail=json.loads(body).get('error',{})
                    except ValueError: detail={}
                    result.update(status='http_error',http_status=error.code,error_code=detail.get('code'),error_type=detail.get('type'),error_message=detail.get('message','')[:500])
                    stop='http_error'
                except Exception as error:
                    result.update(status='error',error_type=type(error).__name__);stop='error'
                spent+=charge;billed+=charge
                with ledger.open('a') as stream: stream.write(json.dumps({'attempt':args.attempt,'id':row['id'],'charged_upper_usd':charge,'reservation_usd':reservation,'usage_known':usage is not None,'status':result['status']})+'\n')
        records.append(result)
        with (folder/'predictions.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
        if len(records)%25==0: print(json.dumps({'completed':len(records),'total':len(rows),'spent_upper_usd':spent,'stop':stop}),flush=True)
    result={'status':'completed' if stop is None else 'stopped','stop_reason':stop,'configuration':spec,'cases':len(rows),
        'metrics':summarize(rows,records),'run_cost_upper_usd':billed,'all_openai_runs_cost_upper_usd':spent,
        'budget_usd':protocol['budget_usd'],'seconds':time.time()-started,'timing_boundary':protocol['timing'],
        'manifest_sha256':acquisition['manifest_sha256'],'protocol_sha256':digest(ROOT/'evals/v4-openai-protocol-v1.json')}
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if stop: raise SystemExit(1)


if __name__=='__main__': main()
