import json

import pytest
from PIL import Image

from video_gen_qc import pipeline
from video_gen_qc.artifacts import sha256
from video_gen_qc.cli import main
from video_gen_qc.config import AppConfig, ProviderConfig
from video_gen_qc.errors import JudgmentError, OutputError, ProviderError
from video_gen_qc.providers.mock import MockVLM


def read(path):
    return json.loads(path.read_text())


def forbidden(*args, **kwargs):
    raise AssertionError("A skipped generator was constructed or called")


def test_mock_full_pipeline(tmp_path, task_path):
    directory = pipeline.run_generation(task_path, output_root=tmp_path)
    expected = {
        "task.json",
        "image_prompt.txt",
        "initial_image.png",
        "video_prompt.txt",
        "video.mp4",
        "sampled_frames",
        "sampled_frames.json",
        "qc_request.json",
        "vlm_raw_response.json",
        "qc_report.json",
        "run_metadata.json",
    }
    assert expected <= {path.name for path in directory.iterdir()}
    report = read(directory / "qc_report.json")
    assert report["decision"] == "REVIEW"
    assert report["is_mock"] is True
    assert len(report["sampling"]["frames"]) == 16
    for check in report["checks"].values():
        assert check["status"] == "uncertain"
        assert "MOCK" in check["reason"]
    metadata = read(directory / "run_metadata.json")
    assert metadata["status"] == "completed"
    assert metadata["mode"] == "full"
    assert metadata["artifacts_sha256"]["video.mp4"] == sha256(directory / "video.mp4")
    with Image.open(directory / "initial_image.png") as image:
        image.verify()


def test_existing_image_skips_image_provider(tmp_path, task_path, initial_image, monkeypatch):
    monkeypatch.setattr(pipeline, "create_image_generator", forbidden)
    config = AppConfig(image_generation=ProviderConfig(provider="http"))
    directory = pipeline.run_generation(
        task_path,
        config,
        initial_image=initial_image,
        output_root=tmp_path,
    )
    assert not (directory / "image_prompt.txt").exists()
    assert "image_generation" not in read(directory / "run_metadata.json")["active_providers"]
    with Image.open(initial_image) as source, Image.open(directory / "initial_image.png") as actual:
        assert source.tobytes() == actual.tobytes()


@pytest.mark.parametrize("with_images", [False, True])
def test_qc_only_never_constructs_generators(
    tmp_path,
    task_path,
    existing_video,
    initial_image,
    monkeypatch,
    with_images,
):
    monkeypatch.setattr(pipeline, "create_image_generator", forbidden)
    monkeypatch.setattr(pipeline, "create_video_generator", forbidden)
    config = AppConfig(
        image_generation=ProviderConfig(provider="http"),
        video_generation=ProviderConfig(provider="http"),
    )
    directory = pipeline.judge_video(
        task_path,
        existing_video,
        config,
        output_root=tmp_path,
        initial_image=initial_image if with_images else None,
        reference_image=initial_image if with_images else None,
    )
    assert read(directory / "qc_report.json")["decision"] == "REVIEW"
    assert not (directory / "image_prompt.txt").exists()
    assert not (directory / "video_prompt.txt").exists()
    request = read(directory / "qc_request.json")
    labels = request["payload"]["image_labels"]
    assert ("initial_image" in labels) == with_images
    assert ("reference_image" in labels) == with_images


def test_generation_context_never_reaches_qc(tmp_path, task_path, monkeypatch):
    requests = []

    class RecordingVLM(MockVLM):
        def complete(self, request):
            requests.append(request)
            if request.purpose != "qc":
                return "GENERATION_CONTEXT_SENTINEL " + super().complete(request)
            return super().complete(request)

    monkeypatch.setattr(pipeline, "create_vlm", lambda *a, **kw: RecordingVLM())
    directory = pipeline.run_generation(task_path, output_root=tmp_path)
    assert [request.purpose for request in requests] == ["image_prompt", "video_prompt", "qc"]
    assert len({request.system for request in requests}) == 3
    video_request = requests[1]
    assert video_request.images[0].path == directory / "initial_image.png"
    assert "GENERATION_CONTEXT_SENTINEL" not in video_request.text
    qc_request = requests[2]
    assert "GENERATION_CONTEXT_SENTINEL" not in qc_request.text + qc_request.system
    payload = json.loads(qc_request.text)
    assert payload["original_task"] == read(directory / "task.json")
    assert set(payload) == {"original_task", "sampling", "image_labels"}
    assert len([image for image in qc_request.images if image.label.startswith("frame_")]) == 16


@pytest.mark.parametrize("failure", ["invalid_json", "bad_evidence", "request_error"])
def test_qc_failure_is_runtime_error_with_no_report(
    tmp_path,
    task_path,
    existing_video,
    monkeypatch,
    failure,
):
    class BrokenVLM(MockVLM):
        def complete(self, request):
            if failure == "request_error":
                raise ProviderError("Service unavailable")
            if failure == "invalid_json":
                return "not JSON"
            data = json.loads(super().complete(request))
            data["checks"]["task_compliance"]["evidence_frames"] = [9999]
            return json.dumps(data)

    monkeypatch.setattr(pipeline, "create_vlm", lambda *a, **kw: BrokenVLM())
    directory = tmp_path / "failed-run"
    with pytest.raises((JudgmentError, ProviderError)):
        pipeline.judge_video(task_path, existing_video, output_dir=directory)
    assert not (directory / "qc_report.json").exists()
    metadata = read(directory / "run_metadata.json")
    assert metadata["status"] == "error"
    assert "decision" not in metadata
    if failure != "request_error":
        assert (directory / "vlm_raw_response.json").is_file()


def test_explicit_output_conflict_does_not_overwrite(tmp_path, task_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(OutputError):
        pipeline.run_generation(task_path, output_dir=tmp_path)
    assert sentinel.read_text() == "preserve me"
    assert not (tmp_path / "task.json").exists()


def test_cli_all_modes(tmp_path, task_path, capsys):
    full, existing, qc = [tmp_path / name for name in ("full", "existing", "qc")]
    assert main(["run", "--task", str(task_path), "--output-dir", str(full)]) == 0
    assert (
        main(
            [
                "run",
                "--task",
                str(task_path),
                "--initial-image",
                str(full / "initial_image.png"),
                "--output-dir",
                str(existing),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "judge",
                "--task",
                str(task_path),
                "--video",
                str(full / "video.mp4"),
                "--output-dir",
                str(qc),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.count("REVIEW [MOCK:") == 3


def test_cli_error_is_not_judgment(tmp_path, task_path, capsys):
    assert main(["judge", "--task", str(task_path), "--video", str(tmp_path / "missing")]) == 2
    output = capsys.readouterr()
    assert "Missing video file" in output.err
    assert "REJECT" not in output.out
