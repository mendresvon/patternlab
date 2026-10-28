"""Launch the demo; reviewer access is a server setting, never a request parameter."""

import argparse


def parser():
    cli = argparse.ArgumentParser(description="PatternLab learning demo")
    cli.add_argument("--sample", action="store_true", help="Explicit prewritten examples, never real inference")
    cli.add_argument("--reviewer", action="store_true", help="Show English beside Chinese for review only")
    cli.add_argument("--sentence-bank", help="Private D-010 bank.jsonl path (85 sentences)")
    cli.add_argument("--checkpoint", help="Private trained Model 1 checkpoint")
    cli.add_argument("--adapter", help="Private fine-tuned Model 2 adapter folder")
    cli.add_argument("--background-model-load", action="store_true", help="Serve the web app while real models warm up")
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--threshold", type=float, default=0.5, help="Model 1 threshold from the chosen validation run")
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=8000)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if bool(args.checkpoint) != bool(args.adapter):
        cli.error("--checkpoint and --adapter must be supplied together")
    if args.background_model_load and (not args.checkpoint or args.sample):
        cli.error("--background-model-load needs real model artifacts")
    if not 0 <= args.threshold <= 1:
        cli.error("--threshold must be a finite number between 0 and 1")
    if args.sample and (args.checkpoint or args.adapter):
        cli.error("Sample mode cannot load real model artifacts")
    from .app import app
    app.state.mode = "sample" if args.sample else "live"
    app.state.reviewer = args.reviewer
    app.state.background_model_load = args.background_model_load
    app.state.bank_path = args.sentence_bank
    app.state.service_options = ({"checkpoint": args.checkpoint, "adapter": args.adapter,
                                  "device": args.device, "threshold": args.threshold,
                                  "local_files_only": True} if args.checkpoint else None)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
