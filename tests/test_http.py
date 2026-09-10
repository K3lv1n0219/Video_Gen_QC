import base64
import json

import httpx
import pytest
from PIL import Image

from video_gen_qc.config import AppConfig, ProviderConfig
from video_gen_qc.errors import ConfigError, ProviderError
from video_gen_qc.pipeline import run_generation
from video_gen_qc.providers.base import ImageInput, VLMRequest
from video_gen_qc.providers.http import HTTPVLM, HTTPBridge, HTTPImageGenerator, HTTPVideoGenerator


@pytest.fixture
def http_config(monkeypatch):
    monkeypatch.setenv("QC_TEST_ENDPOINT", "https://bridge.example.test/vlm")
    monkeypatch.setenv("QC_TEST_KEY", "test-token-never-log")
    return ProviderConfig(
        provider="http",
        model="test-model",
        endpoint_env="QC_TEST_ENDPOINT",
        api_key_env="QC_TEST_KEY",
    )


def test_explicit_paid_gate(http_config):
    with pytest.raises(ConfigError, match="--allow-paid"):
        HTTPBridge(http_config, allow_paid=False)


def test_missing_credentials(http_config, monkeypatch):
    monkeypatch.delenv("QC_TEST_KEY")
    with pytest.raises(ConfigError, match="QC_TEST_KEY"):
        HTTPBridge(http_config, allow_paid=True)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://remote.example.test/vlm",
        "https://bridge.example.test:bad/vlm",
        "https://user:secret@bridge.example.test/vlm",
        "https://bridge.example.test/vlm?key=secret",
    ],
)
def test_invalid_endpoint_is_a_clear_config_error(http_config, monkeypatch, endpoint):
    monkeypatch.setenv("QC_TEST_ENDPOINT", endpoint)
    with pytest.raises(ConfigError, match="Provider URL") as failure:
        HTTPBridge(http_config, allow_paid=True)
    assert "secret" not in str(failure.value)


def test_preflight_blocks_before_any_generation(tmp_path, task_path, http_config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No provider call may precede paid preflight")

    monkeypatch.setattr("video_gen_qc.providers.mock.MockVLM.complete", forbidden)
    with pytest.raises(ConfigError, match="--allow-paid"):
        run_generation(
            task_path, AppConfig(video_generation=http_config), output_dir=tmp_path / "run"
        )
    assert not (tmp_path / "run").exists()


def test_vlm_transmits_actual_image_and_separate_fields(http_config, initial_image):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-token-never-log"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["purpose"] == "video_prompt"
        assert body["system"] == "system instruction"
        assert body["text"] == "original task"
        assert "messages" not in body
        assert body["images"][0]["label"] == "initial_image"
        assert base64.b64decode(body["images"][0]["data_base64"]) == initial_image.read_bytes()
        return httpx.Response(200, json={"text": "Move from the visible starting state."})

    vlm = HTTPVLM(HTTPBridge(http_config, allow_paid=True, transport=httpx.MockTransport(handler)))
    result = vlm.complete(
        VLMRequest(
            "video_prompt",
            "system instruction",
            "original task",
            (ImageInput("initial_image", initial_image),),
        )
    )
    assert result.startswith("Move")
    assert vlm.is_mock is False


@pytest.mark.parametrize("kind", ["image", "video"])
def test_media_bridge_decodes_actual_content(
    http_config, initial_image, existing_video, tmp_path, kind
):
    source = initial_image if kind == "image" else existing_video

    def handler(request):
        body = json.loads(request.content)
        assert body["purpose"] == f"{kind}_generation"
        assert body["prompt"] == "prompt"
        return httpx.Response(
            200, json={f"{kind}_base64": base64.b64encode(source.read_bytes()).decode()}
        )

    bridge = HTTPBridge(http_config, allow_paid=True, transport=httpx.MockTransport(handler))
    if kind == "image":
        path = HTTPImageGenerator(bridge).generate("prompt", tmp_path / "http.png")
        with Image.open(path) as image:
            image.verify()
    else:
        path = HTTPVideoGenerator(bridge).generate(initial_image, "prompt", tmp_path / "http.mp4")
        assert path.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("kind", ["status", "timeout", "invalid_json", "empty_text"])
def test_service_errors_never_become_mock_success(http_config, kind):
    def handler(request):
        if kind == "status":
            return httpx.Response(401, text="test-token-never-log")
        if kind == "timeout":
            raise httpx.ReadTimeout("test-token-never-log", request=request)
        if kind == "invalid_json":
            return httpx.Response(200, text="not JSON")
        return httpx.Response(200, json={"text": ""})

    vlm = HTTPVLM(HTTPBridge(http_config, allow_paid=True, transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError) as failure:
        vlm.complete(VLMRequest("qc", "system", "task"))
    assert "test-token-never-log" not in str(failure.value)
