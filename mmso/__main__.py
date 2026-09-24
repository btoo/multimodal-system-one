"""CLI entry point: python -m mmso --help."""
import argparse
import json

from .artifacts import ROOT, read_manifest, validate_manifest


def main():
    parser=argparse.ArgumentParser(description="Real-data evaluation controls for Multimodal System One")
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare",help="Acquire pinned pilot data and write audited manifests")
    p.add_argument("suite",choices=["speech_keywords","sound_events","screen_grounding","all"])
    p=commands.add_parser("validate",help="Check IDs, split groups, media hashes and schemas")
    p.add_argument("manifest")
    p=commands.add_parser("audio-baseline",help="Train a bounded from-scratch audio CNN and evaluate once")
    p.add_argument("suite",choices=["speech_keywords","sound_events"])
    p.add_argument("--run-id",required=True)
    p.add_argument("--device",choices=["auto","cpu","mps"],default="auto")
    p.add_argument("--epochs",type=int,default=16)
    p.add_argument("--max-train-seconds",type=float,default=120)
    p.add_argument("--seed",type=int,default=20260923)
    p=commands.add_parser("screen-baseline",help="Run fixed CLIP/center controls on the screen probe")
    p.add_argument("--run-id",required=True)
    p.add_argument("--device",choices=["auto","cpu","mps"],default="auto")
    p=commands.add_parser("predict-audio",help="Use a saved local audio checkpoint on a WAV clip")
    p.add_argument("checkpoint");p.add_argument("audio")
    p.add_argument("--device",choices=["auto","cpu","mps"],default="auto")
    p=commands.add_parser("score",help="Score an exact-ID prediction file independently of the model")
    p.add_argument("manifest");p.add_argument("predictions")
    p.add_argument("--split",default="test")
    p.add_argument("--field",default="probabilities")
    p.add_argument("--metadata-only",action="store_true",help="Skip local media existence/hash checks; never claims media verification")
    args=parser.parse_args()
    if hasattr(args,"run_id") and (not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id)):
        parser.error("run-id must contain only letters, digits, hyphens, or underscores")
    if args.command=="prepare":
        from .data import PREPARERS
        for name in PREPARERS if args.suite=="all" else [args.suite]:
            PREPARERS[name]()
    elif args.command=="validate":
        print(json.dumps(validate_manifest(read_manifest(ROOT/args.manifest)),indent=2))
    elif args.command=="audio-baseline":
        from .audio import run_audio
        run_audio(args.suite,args.run_id,args.device,args.epochs,args.max_train_seconds,args.seed)
    elif args.command=="screen-baseline":
        from .screen import run_screen
        run_screen(args.run_id,args.device)
    elif args.command=="predict-audio":
        from .audio import predict_audio
        print(json.dumps(predict_audio(args.checkpoint,args.audio,args.device),indent=2))
    elif args.command=="score":
        from .scoring import score_files
        print(json.dumps(score_files(ROOT/args.manifest,ROOT/args.predictions,args.split,args.field,not args.metadata_only),indent=2))


if __name__=="__main__":
    main()
