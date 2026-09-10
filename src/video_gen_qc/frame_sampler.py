"""Uniform sampling over actual decoded frame indices; no inference between samples."""

import math
from pathlib import Path

import av

from video_gen_qc.errors import OutputError, VideoError
from video_gen_qc.schemas import SampledFrame, SamplingResult


def uniform_indices(total: int, requested: int) -> list[int]:
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 2:
        raise VideoError("sample_frames must be an integer >= 2 to include both endpoints.")
    if total <= 0:
        raise VideoError("Video has zero decoded frames.")
    count = min(total, requested)
    if count == 1:
        return [0]
    return [index * (total - 1) // (count - 1) for index in range(count)]


def sample_video(video_path: Path, output_dir: Path, count: int = 16) -> SamplingResult:
    if not video_path.is_file():
        raise VideoError(f"Video file does not exist: {video_path}")
    try:
        # Count decoded frames instead of trusting container frame-count metadata.
        # Two sequential passes use bounded memory and work with variable frame rates.
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise VideoError("Input has no video stream.")
            total = sum(1 for _ in container.decode(video=0))
        wanted = set(uniform_indices(total, count))
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise OutputError(f"Sample output directory already exists: {output_dir}") from exc
        frames = []
        with av.open(str(video_path)) as container:
            for source_index, frame in enumerate(container.decode(video=0)):
                if source_index not in wanted:
                    continue
                timestamp = frame.time
                if timestamp is None or not math.isfinite(timestamp) or timestamp < 0:
                    raise VideoError("Video lacks usable nonnegative presentation timestamps.")
                if frames and timestamp < frames[-1].timestamp_seconds:
                    raise VideoError("Video presentation timestamps are not monotonic.")
                frame_id = len(frames)
                path = output_dir / f"frame_{frame_id:03d}.png"
                frame.to_image().save(path, format="PNG")
                frames.append(
                    SampledFrame(
                        frame_id=frame_id,
                        source_frame_index=source_index,
                        timestamp_seconds=float(timestamp),
                        path=f"{output_dir.name}/{path.name}",
                    )
                )
        if len(frames) != len(wanted):
            raise VideoError("Video decoded inconsistently between sampling passes.")
        return SamplingResult(decoded_frame_count=total, requested_frame_count=count, frames=frames)
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise VideoError(f"Cannot decode/sample video {video_path}: {exc}") from exc
