import argparse
import getpass
import json
import sys
import warnings
from pathlib import Path

from video_gen_qc import __version__
from video_gen_qc.config import load_config
from video_gen_qc.environment import load_environment, save_qwen_key
from video_gen_qc.errors import VideoQCError
from video_gen_qc.pipeline import judge_video, run_generation


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Independent VLM-led video QC (offline mocks by default)"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    configure = commands.add_parser("configure", help="Save a Qwen API Key in a local .env file")
    configure.add_argument(
        "--env-file", type=Path, help="Local secret file; default = project .env"
    )
    for name, help_text in [
        ("run", "Generate an initial image (unless supplied), generate a video, then inspect"),
        ("judge", "Inspect an existing video without any generation calls"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--task", type=Path, required=True)
        command.add_argument("--initial-image", type=Path, help="Exact starting image; skips T2I")
        command.add_argument(
            "--reference-image",
            type=Path,
            help="Visual context; in full mode also passed to image generation for editing",
        )
        command.add_argument(
            "--config", type=Path, help="YAML config; omitted = built-in mock defaults"
        )
        command.add_argument("--env-file", type=Path, help="Explicit .env file; must exist")
        command.add_argument("--output-root", type=Path, help="Parent for a unique run directory")
        command.add_argument("--output-dir", type=Path, help="Exact run directory; must not exist")
        command.add_argument(
            "--allow-paid",
            action="store_true",
            help="Explicitly permit configured real VLM/image/video API calls",
        )
        if name == "judge":
            command.add_argument("--video", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "configure":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    key = getpass.getpass("DASHSCOPE_API_KEY (hidden input): ")
            except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
                raise VideoQCError(
                    "Key entry cancelled or unavailable; use an interactive terminal."
                ) from None
            path = save_qwen_key(key, args.env_file)
            print(f"Saved API Key to {path}.")
            if args.env_file is not None:
                print("Use --env-file with this path on future runs.")
            else:
                print("Future CLI runs from this project load it automatically.")
            return 0
        load_environment(args.config, args.env_file)
        config = load_config(args.config)
        common = dict(
            task_path=args.task,
            config=config,
            initial_image=args.initial_image,
            reference_image=args.reference_image,
            output_root=args.output_root,
            output_dir=args.output_dir,
            allow_paid=args.allow_paid,
        )
        run_dir = (
            run_generation(**common)
            if args.command == "run"
            else judge_video(video=args.video, **common)
        )
        report_path = run_dir / "qc_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        marker = " [MOCK: no VLM inspection performed]" if report["is_mock"] else ""
        print(f"{report['decision']}{marker}")
        print(f"Report: {report_path.resolve()}")
        return 0  # Completed judgments (including REJECT/REVIEW) are not runtime errors.
    except (VideoQCError, OSError) as exc:
        print(f"video-qc: {exc}", file=sys.stderr)
        return 2
