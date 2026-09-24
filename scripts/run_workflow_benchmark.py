"""Run the original workflow smoke or MiSO capability audit, not vendor evals."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmso.workflow_benchmark import run
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--provider',choices=['miso-v3','jev'],required=True)
parser.add_argument('--run-id',required=True)
parser.add_argument('--execute',action='store_true')
parser.add_argument('--max-spend-usd',type=float,default=.1)
args=parser.parse_args()
print(json.dumps(run(args.provider,args.run_id,args.execute,args.max_spend_usd),indent=2))
