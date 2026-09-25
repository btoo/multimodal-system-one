"""Read metered study costs and verify study apps have stopped."""
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
    ids = set()
    for path in (ROOT / ".research/v4").glob("*.log"):
        ids.update(re.findall(r"https://modal.com/apps/btoo/main/(ap-[A-Za-z0-9]+)", path.read_text(errors="replace")))
    if not ids: raise ValueError("No recorded study app IDs")
    reservations = [json.loads(p.read_text()) for p in (ROOT / "reports/v4-selection-v1/attempts").glob("*/reservation.json")]
    client = await _Client.from_env()
    start, end = Timestamp(), Timestamp()
    start.FromDatetime(datetime.fromtimestamp(min(r["started_unix"] for r in reservations), timezone.utc) - timedelta(hours=1))
    end.GetCurrentTime()
    values = []
    request = api_pb2.WorkspaceBillingReportRequest(start_timestamp=start, end_timestamp=end, resolution="h", app_ids=sorted(ids))
    async for item in client.stub.WorkspaceBillingReport.unary_stream(request):
        values.append(MessageToDict(item, preserving_proto_field_name=True))
    async def lifecycle(app_id):
        response = await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))
        value = MessageToDict(response, preserving_proto_field_name=True)["lifecycle"]
        return {"app_id": app_id, **{k: value.get(k) for k in ("app_state", "created_at", "stopped_at")}}
    # The CLI intentionally drops older stopped apps. Query each recorded ID
    # directly instead of interpreting absence from that list as shutdown.
    states = await asyncio.gather(*(lifecycle(app_id) for app_id in sorted(ids)))
    missing = []
    containers = json.loads(subprocess.check_output(["uv", "run", "--locked", "--group", "cloud", "modal", "container", "list", "--json"], cwd=ROOT, text=True))
    containers = [r for r in containers if r["app_id"] in ids]
    all_stopped = not containers and all(r["app_state"] == "APP_STATE_STOPPED" for r in states)
    result = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "app_ids": sorted(ids),
              "metered_study_usd": str(sum((Decimal(r.get("cost", "0")) for r in values), Decimal(0))),
              "billing_rows": values, "billing_may_lag": True, "final_invoice_cash_charge": None,
              "scope": "Provider-metered resource use for these study apps; includes failed attempts. Credits and final invoice allocation are not inferred.",
              "gpu_jobs_reserved": len(reservations), "maximum_reserved_compute_proxy_usd": sum(r["maximum_compute_proxy_usd"] for r in reservations),
              "study_ceiling_usd": 250, "app_states": states, "missing_app_states": missing,
              "active_study_containers": containers, "all_study_apps_stopped": all_stopped,
              "persistent_storage": "Publisher checkpoints and research artifacts remain in miso-v4-selection-v1 Volume; no persistent GPU endpoint was created."}
    path = ROOT / "reports/v4-selection-v1/cost-and-shutdown.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["metered_study_usd", "gpu_jobs_reserved", "all_study_apps_stopped", "missing_app_states"]}, indent=2))
    if not all_stopped: raise SystemExit("Study shutdown has not been fully verified")


if __name__ == "__main__":
    asyncio.run(main())
