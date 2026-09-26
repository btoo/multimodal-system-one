"""Study-scoped provider billing and shutdown evidence."""
import asyncio
from datetime import datetime,timezone,timedelta
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess

from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp
from modal.client import _Client
from modal_proto import api_pb2

ROOT=Path(__file__).resolve().parents[1]


async def main():
    report=ROOT/'reports/v4-grounding-training-v1';ids=set()
    for path in (ROOT/'.research/v4-grounding').glob('*.log'):
        ids.update(re.findall(r'https://modal.com/apps/btoo/main/(ap-[A-Za-z0-9]+)',path.read_text(errors='replace')))
    reservations=[json.loads(p.read_text()) for p in (report/'attempts').glob('*/reservation.json')]
    if not ids or not reservations:raise ValueError('No recorded study apps')
    client=await _Client.from_env();start,end=Timestamp(),Timestamp()
    start.FromDatetime(datetime.fromtimestamp(min(r['started_unix'] for r in reservations),timezone.utc)-timedelta(hours=1));end.GetCurrentTime()
    request=api_pb2.WorkspaceBillingReportRequest(start_timestamp=start,end_timestamp=end,resolution='h',app_ids=sorted(ids))
    billing=[]
    async for row in client.stub.WorkspaceBillingReport.unary_stream(request):billing.append(MessageToDict(row,preserving_proto_field_name=True))
    async def state(app_id):
        raw=await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))
        value=MessageToDict(raw,preserving_proto_field_name=True)['lifecycle']
        return {'app_id':app_id,**{k:value.get(k) for k in ('app_state','created_at','stopped_at')}}
    states=await asyncio.gather(*(state(i) for i in sorted(ids)))
    containers=json.loads(subprocess.check_output(['uv','run','--locked','--group','cloud','modal','container','list','--json'],cwd=ROOT,text=True))
    containers=[c for c in containers if c['app_id'] in ids]
    stopped=not containers and all(s['app_state']=='APP_STATE_STOPPED' for s in states)
    ledger=report/'openai-spend-ledger.jsonl';api=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    result={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'app_states':states,'active_containers':containers,'all_apps_stopped':stopped,
        'modal_metered_usd':str(sum((Decimal(r.get('cost','0')) for r in billing),Decimal(0))),'billing_rows':billing,'billing_may_lag':True,
        'gpu_jobs_reserved':len(reservations),'gpu_reserved_upper_usd':sum(r['maximum_compute_proxy_usd'] for r in reservations),'gpu_reservation_ceiling_usd':45,
        'openai_new_cost_upper_usd':sum(r['charged_upper_usd'] for r in api),'openai_usage_unknown_calls':sum(not r['usage_known'] for r in api),
        'final_invoice_cash_charge':None,'persistent_endpoint':False,
        'persistent_storage':'Pinned publisher weights and experiment/adapter records remain on the existing Modal volume.',
        'scope':'Only the app IDs from this study; includes failed startup. Training, evaluation and loading are research compute, not an inference-serving price.'}
    (report/'cost-and-shutdown.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('modal_metered_usd','gpu_jobs_reserved','all_apps_stopped','openai_new_cost_upper_usd')}))
    if not stopped:raise SystemExit('A study app is still running')


if __name__=='__main__':asyncio.run(main())
