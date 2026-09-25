"""Capture provider-metered follow-up spend and verify every app stopped."""
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess

from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp
from modal.client import _Client
from modal_proto import api_pb2

ROOT = Path(__file__).resolve().parents[1]


async def main():
    ids=set()
    for file in (ROOT/'.research/v4-iteration').glob('*.log'):
        ids.update(re.findall(r'https://modal.com/apps/btoo/main/(ap-[A-Za-z0-9]+)',file.read_text(errors='replace')))
    reports=ROOT/'reports/v4-iteration-v2'
    reservations=[json.loads(p.read_text()) for p in (reports/'attempts').glob('*/reservation.json')]
    if not ids or not reservations: raise ValueError('No captured GPU attempts')
    client=await _Client.from_env(); start,end=Timestamp(),Timestamp()
    start.FromDatetime(datetime.fromtimestamp(min(r['started_unix'] for r in reservations),timezone.utc)-timedelta(hours=1));end.GetCurrentTime()
    request=api_pb2.WorkspaceBillingReportRequest(start_timestamp=start,end_timestamp=end,resolution='h',app_ids=sorted(ids))
    billing=[]
    async for row in client.stub.WorkspaceBillingReport.unary_stream(request): billing.append(MessageToDict(row,preserving_proto_field_name=True))
    async def state(app_id):
        raw=await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))
        value=MessageToDict(raw,preserving_proto_field_name=True)['lifecycle']
        return {'app_id':app_id,**{k:value.get(k) for k in ('app_state','created_at','stopped_at')}}
    states=await asyncio.gather(*(state(i) for i in sorted(ids)))
    containers=json.loads(subprocess.check_output(['uv','run','--locked','--group','cloud','modal','container','list','--json'],cwd=ROOT,text=True))
    containers=[c for c in containers if c['app_id'] in ids]
    stopped=not containers and all(s['app_state']=='APP_STATE_STOPPED' for s in states)
    api_ledger=reports/'openai-spend-ledger.jsonl'
    api=[json.loads(s) for s in api_ledger.read_text().splitlines()] if api_ledger.exists() else []
    jev=[json.loads(p.read_text()) for p in (reports/'attempts').glob('jev-*/summary.json')]
    diagnostic=json.loads((reports/'attempts/jev-mmlu-pro-v1/response-diagnostic.json').read_text())
    diagnostic_cost=diagnostic['raw']['usage']['input_tokens']*.042/1e6
    unknown_jev_calls=0
    for file in (reports/'attempts').glob('jev-*/summary.json'):
        result=json.loads(file.read_text())
        preds=[json.loads(s) for s in (file.parent/'predictions.jsonl').read_text().splitlines()]
        successful=[r for r in preds if r['status']=='ok']
        remaining=result['calls']-len(successful)
        # In the first attempt, usage was validated and charged before response
        # canonicalization failed. The aggregate includes both API calls.
        if remaining==1 and result['known_cost_usd']>sum(r.get('cost_usd',0) for r in successful): remaining=0
        unknown_jev_calls+=remaining
    result={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'app_states':states,'active_containers':containers,
        'all_apps_stopped':stopped,'modal_metered_usd':str(sum((Decimal(r.get('cost','0')) for r in billing),Decimal(0))),
        'billing_may_lag':True,'final_invoice_cash_charge':None,'billing_rows':billing,
        'gpu_jobs':len(reservations),'gpu_reserved_upper_usd':sum(r['maximum_compute_proxy_usd'] for r in reservations),
        'gpu_reservation_ceiling_usd':45,'openai_cost_upper_usd':sum(r['charged_upper_usd'] for r in api),
        'openai_ceiling_usd':2,'openai_usage_unknown_calls':sum(not r['usage_known'] for r in api),
        'jev_known_cost_usd':sum(r['known_cost_usd'] for r in jev)+diagnostic_cost,
        'jev_unknown_call_count':unknown_jev_calls,'jev_unknown_call_cost_upper_usd':unknown_jev_calls*65536*.042/1e6,
        'scope':'Research compute and public-data inference only. Credits/cash invoice allocation and production serving cost are not inferred.',
        'persistent_storage':'Pinned weights and experiment records remain on the existing Modal volume; no persistent GPU service created.'}
    (reports/'cost-and-shutdown.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('modal_metered_usd','gpu_jobs','all_apps_stopped','openai_cost_upper_usd','jev_known_cost_usd')}))
    if not stopped: raise SystemExit('Some study resources are still running')


if __name__=='__main__':asyncio.run(main())
