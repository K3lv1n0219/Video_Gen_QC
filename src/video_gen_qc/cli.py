import argparse
import json
import sys
from pathlib import Path

from video_gen_qc import __version__
from video_gen_qc.config import load_config
from video_gen_qc.errors import VideoQCError
from video_gen_qc.pipeline import judge_video, run_generation


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Independent VLM-led video QC (offline mocks by default)"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("run", "Generate an initial image (unless supplied), generate a video, then inspect"),
        ("judge", "Inspect an existing video without any generation calls"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--task", type=Path, required=True)
        command.add_argument("--initial-image", type=Path)
        command.add_argument("--reference-image", type=Path)
        command.add_argument(
            "--config", type=Path, help="YAML config; omitted = built-in mock defaults"
        )
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
