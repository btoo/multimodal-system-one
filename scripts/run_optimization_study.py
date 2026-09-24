"""Run one explicit stage; source and nomination must be committed before use."""
import argparse
import json

from mmso.optimization_study import prepare_optimization, train_optimization, nominate_optimization, evaluate_optimization

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["prepare", "train", "nominate", "evaluate"])
parser.add_argument("--run-id")
parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
args = parser.parse_args()
if args.stage in {"train", "evaluate"} and not args.run_id:
    parser.error("--run-id is required")
if args.stage == "prepare":
    result = prepare_optimization()
elif args.stage == "train":
    result = train_optimization(args.run_id, args.device)
elif args.stage == "nominate":
    result = nominate_optimization()
else:
    result = evaluate_optimization(args.run_id, args.device)
if args.stage == "train":
    result = {"run_id": result["run_id"], "steps": result["configuration"]["actual_steps"],
              "selected_epoch": result["configuration"]["selected_epoch"], "development": result["development"]["by_slice"],
              "train_seconds": result["train_seconds"], "matched": result["matched_step_budget_completed"]}
elif args.stage == "evaluate":
    result = {"run_id": result["run_id"], "raw": result["raw"]["by_slice"], "calibrated": result["calibrated"]["by_slice"]}
elif args.stage == "prepare":
    result = {k: result[k] for k in ("scenes", "questions", "confirmation_speakers", "confirmation_audio")}
print(json.dumps(result, indent=2))
