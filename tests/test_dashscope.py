import base64
import json

import httpx
import pytest
from pydantic import ValidationError

from video_gen_qc.config import ImageOptions, ProviderConfig, VideoOptions, load_config
from video_gen_qc.errors import ConfigError, InputError, ProviderError
from video_gen_qc.pipeline import run_generation
from video_gen_qc.prompts import QC_SYSTEM
from video_gen_qc.providers import dashscope
from video_gen_qc.providers.dashscope import (
    IMAGE_ENDPOINT,
    VIDEO_ENDPOINT,
    DashScopeClient,
    QwenImageGenerator,
    WanVideoGenerator,
)
from video_gen_qc.providers.factory import (
    create_image_generator,
    create_video_generator,
    create_vlm,
)

TASK_ID = "17ed7e50-00cf-4509-aea1-123456789abc"
MEDIA_URL = "https://media.oss-cn-beijing.aliyuncs.com/result?Signature=private"
TEST_KEY = "test-key-never-log"


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("TEST_DASHSCOPE_KEY", TEST_KEY)
    return ProviderConfig(
        provider="wan",
        model="wan3.0-video",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="TEST_DASHSCOPE_KEY",
    )


@pytest.fixture
def image_config(config):
    return config.model_copy(update={"provider": "qwen_image", "model": "qwen-image-3.0-pro"})


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(dashscope.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        dashscope.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    return now


def task_response(status, **fields):
    return {"output": {"task_id": TASK_ID, "task_status": status, **fields}}


def image_response():
    return {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": [{"image": MEDIA_URL}]},
                }
            ]
        }
    }


def completion(text):
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}


def test_same_key_three_distinct_models_and_paid_gates(
    config, image_config, task_path, monkeypatch
):
    full = load_config(task_path.parents[1] / "configs/aliyun-beijing.yaml")
    assert len({p.model for p in (full.vlm, full.image_generation, full.video_generation)}) == 3
    assert {p.api_key_env for p in (full.vlm, full.image_generation, full.video_generation)} == {
        "DASHSCOPE_API_KEY"
    }
    for factory, provider_config in (
        (create_vlm, full.vlm),
        (create_image_generator, image_config),
        (create_video_generator, config),
    ):
        with pytest.raises(ConfigError, match="--allow-paid"):
            factory(provider_config)
    monkeypatch.delenv("TEST_DASHSCOPE_KEY")
    with pytest.raises(ConfigError, match="TEST_DASHSCOPE_KEY"):
        create_video_generator(config, allow_paid=True)


def test_misrouted_models_fail_before_requests(config, image_config):
    for factory, provider_config in (
        (create_vlm, config),
        (create_vlm, image_config),
        (create_image_generator, config),
        (create_video_generator, image_config),
    ):
        with pytest.raises(ConfigError):
            factory(provider_config, allow_paid=True)
    with pytest.raises(ConfigError):
        WanVideoGenerator(config.model_copy(update={"model": "wan2.6-i2v"}), allow_paid=True)


def test_existing_or_missing_output_never_submits(config, image_config, initial_image, tmp_path):
    wan = WanVideoGenerator(config, allow_paid=True)
    image = QwenImageGenerator(image_config, allow_paid=True)
    existing = tmp_path / "existing.bin"
    existing.write_bytes(b"preserve user file")
    for output in (existing, tmp_path / "missing" / "file"):
        with pytest.raises(InputError):
            wan.generate(initial_image, "movement", output)
        with pytest.raises(InputError):
            image.generate("scene", output)
    assert existing.read_bytes() == b"preserve user file"


@pytest.mark.parametrize("size", ["1024x1024", "0*1024", "256*256", "8192*8192", "100*3000"])
def test_invalid_image_size(size):
    with pytest.raises(ValidationError):
        ImageOptions(size=size)


@pytest.mark.parametrize("options", [{"duration": 31}, {"duration": True}, {"resolution": "4K"}])
def test_video_cost_parameters_validated(options):
    with pytest.raises(ValidationError):
        VideoOptions(**options)


def test_image_generation_download(image_config, initial_image, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            assert request.url.path == "/api/v1" + IMAGE_ENDPOINT
            assert request.headers["Authorization"] == "Bearer " + TEST_KEY
            body = json.loads(request.content)
            assert body["model"] == "qwen-image-3.0-pro"
            assert body["input"] == {"messages": [{"role": "user", "content": [{"text": "scene"}]}]}
            assert body["parameters"] == {"n": 1, "size": "1024*1024", "prompt_extend": False}
            return httpx.Response(200, json=image_response())
        assert "authorization" not in request.headers
        return httpx.Response(200, content=initial_image.read_bytes())

    provider = QwenImageGenerator(
        image_config, allow_paid=True, transport=httpx.MockTransport(handler)
    )
    result = provider.generate("scene", tmp_path / "generated.png")
    assert result.read_bytes() == initial_image.read_bytes()
    assert len(calls) == 2


def test_wan_submits_once_actual_first_frame_and_polls(
    config, initial_image, existing_video, tmp_path, clock
):
    states = iter(["PENDING", "RUNNING", "SUCCEEDED"])
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            assert request.headers["X-DashScope-Async"] == "enable"
            body = json.loads(request.content)
            assert body["model"] == "wan3.0-video"
            frame = body["input"]["media"][0]
            assert frame["type"] == "first_frame"
            assert base64.b64decode(frame["url"].split(",", 1)[1]) == initial_image.read_bytes()
            assert body["parameters"] == {
                "resolution": "720P",
                "duration": 5,
                "audio": False,
                "prompt_extend": False,
                "ratio": "adaptive",
            }
            return httpx.Response(200, json=task_response("PENDING"))
        if request.url.path == "/api/v1/tasks/" + TASK_ID:
            assert request.headers["Authorization"] == "Bearer " + TEST_KEY
            assert (tmp_path / "video_generation_job.json").is_file()
            return httpx.Response(200, json=task_response(next(states), video_url=MEDIA_URL))
        assert "authorization" not in request.headers
        return httpx.Response(200, content=existing_video.read_bytes())

    provider = WanVideoGenerator(config, allow_paid=True, transport=httpx.MockTransport(handler))
    result = provider.generate(initial_image, "movement", tmp_path / "output.mp4")
    assert result.read_bytes() == existing_video.read_bytes()
    assert sum(r.method == "POST" for r in calls) == 1
    record = json.loads((tmp_path / "video_generation_job.json").read_text())
    assert record == {"model": "wan3.0-video", "task_id": TASK_ID, "task_status": "SUCCEEDED"}
    assert TEST_KEY not in json.dumps(record)
    assert "Signature" not in json.dumps(record)


@pytest.mark.parametrize("status", ["FAILED", "CANCELED", "UNKNOWN", "UNRECOGNIZED", [], None])
def test_failed_wan_never_downloads_or_resubmits(config, tmp_path, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=task_response(status, message=TEST_KEY))

    provider = WanVideoGenerator(config, allow_paid=True, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as error:
        provider.resume(TASK_ID, tmp_path / "output.mp4")
    assert TEST_KEY not in str(error.value)
    assert len(calls) == 1 and calls[0].method == "GET"
    assert not (tmp_path / "output.mp4").exists()


def test_timeout_retains_task_and_resume_only_gets(config, tmp_path, clock):
    config = config.model_copy(update={"video_options": VideoOptions(task_timeout_seconds=20.0)})
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=task_response("RUNNING"))

    provider = WanVideoGenerator(config, allow_paid=True, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="resume this task"):
        provider.resume(TASK_ID, tmp_path / "output.mp4")
    assert clock[0] == 20.0
    assert len(calls) == 2 and all(r.method == "GET" for r in calls)
    assert json.loads((tmp_path / "video_generation_job.json").read_text())["task_id"] == TASK_ID


def test_resume_success_never_submits(config, tmp_path, existing_video):
    def handler(request):
        assert request.method == "GET"
        if request.url.host == "dashscope.aliyuncs.com":
            return httpx.Response(200, json=task_response("SUCCEEDED", video_url=MEDIA_URL))
        return httpx.Response(200, content=existing_video.read_bytes())

    provider = WanVideoGenerator(config, allow_paid=True, transport=httpx.MockTransport(handler))
    output = tmp_path / "video.mp4"
    assert provider.resume(TASK_ID, output) == output
    with pytest.raises(InputError, match="already exists"):
        provider.resume(TASK_ID, output)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, text=TEST_KEY),
        httpx.Response(200, json={"code": "InvalidApiKey", "message": TEST_KEY}),
        httpx.Response(200, text=TEST_KEY),
        httpx.Response(200, json=[]),
    ],
)
def test_api_errors_are_sanitized_and_not_retried(config, response):
    calls = []

    def handler(request):
        calls.append(request)
        return response

    client = DashScopeClient(config, allow_paid=True, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as error:
        client.request("POST", VIDEO_ENDPOINT, {"model": config.model})
    assert TEST_KEY not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a",
        "https://localhost/a",
        "https://127.0.0.1/a",
        "https://user:pass@example.com/a",
    ],
)
def test_download_rejects_invalid_urls_before_network(config, tmp_path, url):
    client = DashScopeClient(config, allow_paid=True)
    with pytest.raises(ProviderError, match="URL"):
        client.download(url, tmp_path / "output", max_bytes=10)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "https://example.com/"}),
        httpx.Response(200, content=b"too-large"),
        httpx.Response(200, content=b""),
    ],
)
def test_failed_download_cleans_partial_file(config, tmp_path, response):
    client = DashScopeClient(
        config, allow_paid=True, transport=httpx.MockTransport(lambda r: response)
    )
    output = tmp_path / "output"
    with pytest.raises(ProviderError):
        client.download(MEDIA_URL, output, max_bytes=3)
    assert not output.exists() and not output.with_suffix(".part").exists()


@pytest.mark.parametrize(
    "data",
    [{}, {"output": {"choices": []}}, {"output": {"choices": [{"finish_reason": "length"}]}}],
)
def test_invalid_image_response(image_config, tmp_path, data):
    provider = QwenImageGenerator(
        image_config,
        allow_paid=True,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data)),
    )
    with pytest.raises(ProviderError):
        provider.generate("scene", tmp_path / "out.png")


def test_full_aliyun_pipeline_with_offline_transport(
    task_path, initial_image, existing_video, tmp_path, monkeypatch
):
    config = load_config(task_path.parents[1] / "configs/aliyun-beijing.yaml")
    monkeypatch.setenv("DASHSCOPE_API_KEY", TEST_KEY)
    vlm_calls = []
    image_generations = []
    video_generations = []

    def handler(request):
        if request.method == "POST":
            body = json.loads(request.content)
            assert request.headers["Authorization"] == "Bearer " + TEST_KEY
            if request.url.path.endswith("/chat/completions"):
                vlm_calls.append(body)
                assert body["model"] == "qwen3.8-max"
                assert len(body["messages"]) == 2
                if body["messages"][0]["content"] == QC_SYSTEM:
                    assert "GENERATION_SENTINEL" not in json.dumps(body)
                    images = [b for b in body["messages"][1]["content"] if b["type"] == "image_url"]
                    assert len(images) == 17
                    report = {
                        "checks": {
                            name: {
                                "status": "uncertain",
                                "reason": "Offline fixture.",
                                "evidence_frames": [0],
                            }
                            for name in ("task_compliance", "scene_consistency", "visual_anomalies")
                        }
                    }
                    return httpx.Response(200, json=completion(json.dumps(report)))
                return httpx.Response(200, json=completion("GENERATION_SENTINEL"))
            if request.url.path.endswith(IMAGE_ENDPOINT):
                image_generations.append(body)
                return httpx.Response(200, json=image_response())
            assert request.url.path.endswith(VIDEO_ENDPOINT)
            video_generations.append(body)
            return httpx.Response(200, json=task_response("PENDING"))
        if request.url.path.endswith("/tasks/" + TASK_ID):
            return httpx.Response(
                200, json=task_response("SUCCEEDED", video_url=MEDIA_URL + "video")
            )
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            content=existing_video.read_bytes()
            if str(request.url).endswith("video")
            else initial_image.read_bytes(),
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client(**kwargs):
        kwargs["transport"] = transport
        return original_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    output = run_generation(task_path, config, output_dir=tmp_path / "run", allow_paid=True)
    assert len(vlm_calls) == 3
    assert len(image_generations) == len(video_generations) == 1
    assert image_generations[0]["model"] == "qwen-image-3.0-pro"
    assert video_generations[0]["model"] == "wan3.0-video"
    first_frame = video_generations[0]["input"]["media"][0]["url"]
    assert (
        base64.b64decode(first_frame.split(",", 1)[1])
        == (output / "initial_image.png").read_bytes()
    )
    report = json.loads((output / "qc_report.json").read_text())
    assert report["is_mock"] is False and report["decision"] == "REVIEW"
    assert report["model"] == "qwen3.8-max"
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert "video_generation_job.json" in metadata["artifacts_sha256"]
    assert TEST_KEY not in json.dumps(metadata)
