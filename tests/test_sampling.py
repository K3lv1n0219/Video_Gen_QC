from fractions import Fraction

import av
import pytest
from PIL import Image

from video_gen_qc.errors import OutputError, VideoError
from video_gen_qc.frame_sampler import sample_video, uniform_indices


def test_samples_actual_endpoints_and_timestamps(tmp_path, existing_video):
    with av.open(str(existing_video)) as container:
        actual_times = [frame.time for frame in container.decode(video=0)]
    result = sample_video(existing_video, tmp_path / "sampled_frames")
    assert result.decoded_frame_count == len(actual_times) == 32
    assert len(result.frames) == 16
    assert result.frames[0].source_frame_index == 0
    assert result.frames[-1].source_frame_index == 31
    assert [f.frame_id for f in result.frames] == list(range(16))
    indices = [frame.source_frame_index for frame in result.frames]
    assert len(set(indices)) == 16
    gaps = [right - left for left, right in zip(indices, indices[1:], strict=False)]
    assert max(gaps) - min(gaps) <= 1
    for frame in result.frames:
        assert frame.timestamp_seconds == actual_times[frame.source_frame_index]
        with Image.open(tmp_path / frame.path) as image:
            image.verify()


def test_variable_rate_uses_pts_not_index_divided_by_fps(tmp_path):
    path = tmp_path / "variable.mp4"
    timestamps = [0, 1, 4, 10, 11]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for pts in timestamps:
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), "gray"))
            frame.pts = pts
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = sample_video(path, tmp_path / "frames", count=16)
    assert [frame.timestamp_seconds for frame in result.frames] == [v / 10 for v in timestamps]
    assert len(result.frames) == 5


@pytest.mark.parametrize(
    "total,count,expected",
    [
        (1, 16, [0]),
        (3, 16, [0, 1, 2]),
        (10, 2, [0, 9]),
        (5, 3, [0, 2, 4]),
    ],
)
def test_short_video_sampling(total, count, expected):
    assert uniform_indices(total, count) == expected


@pytest.mark.parametrize("total,count", [(0, 16), (10, 1), (10, True), (10, 2.5)])
def test_invalid_sampling_requests(total, count):
    with pytest.raises(VideoError):
        uniform_indices(total, count)


@pytest.mark.parametrize("content", [b"", b"not a video"])
def test_unreadable_or_empty_video(tmp_path, content):
    path = tmp_path / "bad.mp4"
    path.write_bytes(content)
    with pytest.raises(VideoError, match="Cannot decode"):
        sample_video(path, tmp_path / "frames")


def test_missing_video(tmp_path):
    with pytest.raises(VideoError, match="does not exist"):
        sample_video(tmp_path / "missing.mp4", tmp_path / "frames")


def test_sample_output_conflict(tmp_path, existing_video):
    with pytest.raises(OutputError):
        sample_video(existing_video, tmp_path)
