"""Run only the explicitly selected stage of the preregistered local study."""
import argparse
import json

from mmso.scale_study import prepare_scale, train_scale, nominate_scale, evaluate_scale

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["prepare", "train", "nominate", "evaluate"])
parser.add_argument("--run-id")
parser.add_argument("--device", default="mps")
args = parser.parse_args()
if args.stage in {"train", "evaluate"} and not args.run_id:
    parser.error("--run-id is required")
if args.stage == "prepare":
    result = prepare_scale()
elif args.stage == "train":
    result = train_scale(args.run_id, args.device)
elif args.stage == "nominate":
    result = nominate_scale()
else:
    result = evaluate_scale(args.run_id, args.device)
print(json.dumps(result, indent=2))
