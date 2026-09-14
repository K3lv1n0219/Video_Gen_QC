import base64
import json

import httpx
import pytest

from video_gen_qc import pipeline
from video_gen_qc.config import ProviderConfig, load_config
from video_gen_qc.errors import ConfigError, JudgmentError, ProviderError
from video_gen_qc.prompts import QC_SYSTEM
from video_gen_qc.providers.base import ImageInput, VLMRequest
from video_gen_qc.providers.factory import (
    create_image_generator,
    create_video_generator,
    create_vlm,
)
from video_gen_qc.providers.qwen import QwenVLM


@pytest.fixture
def qwen_config(monkeypatch):
    monkeypatch.setenv("QWEN_TEST_API_KEY", "test-qwen-token-never-log")
    return ProviderConfig(
        provider="qwen",
        model="qwen3-vl-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="QWEN_TEST_API_KEY",
    )


def completion(text, finish_reason="stop"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": "internal-reasoning-sentinel",
                },
                "finish_reason": finish_reason,
            }
        ]
    }


def test_factory_and_paid_gate(qwen_config):
    with pytest.raises(ConfigError, match="--allow-paid"):
        create_vlm(qwen_config)
    vlm = create_vlm(qwen_config, allow_paid=True)
    assert isinstance(vlm, QwenVLM)
    assert vlm.provider_name == "qwen"
    assert vlm.is_mock is False


def test_beijing_example(task_path):
    config = load_config(task_path.parents[1] / "configs/qwen-beijing.yaml")
    assert config.vlm.provider == "qwen"
    assert config.vlm.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.vlm.api_key_env == "DASHSCOPE_API_KEY"
    assert config.image_generation.provider == config.video_generation.provider == "mock"


@pytest.mark.parametrize("factory", [create_image_generator, create_video_generator])
def test_qwen_cannot_be_misrouted_to_generators(qwen_config, factory):
    with pytest.raises(ConfigError, match="VLM calls only"):
        factory(qwen_config, allow_paid=True)


@pytest.mark.parametrize(
    "change",
    [
        {"model": None},
        {"base_url": None},
        {"endpoint_env": "ANOTHER_URL"},
        {"base_url": "https://example.test/chat/completions"},
    ],
)
def test_invalid_qwen_config(qwen_config, change):
    with pytest.raises(ConfigError):
        QwenVLM(qwen_config.model_copy(update=change), allow_paid=True)


def test_missing_key(qwen_config, monkeypatch):
    monkeypatch.delenv("QWEN_TEST_API_KEY")
    with pytest.raises(ConfigError, match="QWEN_TEST_API_KEY"):
        QwenVLM(qwen_config, allow_paid=True)


@pytest.mark.parametrize("purpose", ["image_prompt", "video_prompt", "qc"])
def test_actual_image_bytes_and_labels(qwen_config, initial_image, purpose):
    def handler(request):
        assert str(request.url) == qwen_config.base_url + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-qwen-token-never-log"
        body = json.loads(request.content)
        assert body["model"] == "qwen3-vl-plus"
        assert body["enable_thinking"] is False
        assert body["stream"] is False
        assert body["max_tokens"] == 4096
        assert ("response_format" in body) == (purpose == "qc")
        if purpose == "qc":
            assert body["response_format"] == {"type": "json_object"}
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert body["messages"][0]["content"] == "Distinct system instruction"
        blocks = body["messages"][1]["content"]
        assert blocks[0]["text"] == "Original task JSON"
        assert blocks[1]["text"] == "Image label: frame_7"
        uri = blocks[2]["image_url"]["url"]
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]) == initial_image.read_bytes()
        assert "purpose" not in body  # Vendor wire format, not the project's bridge contract.
        return httpx.Response(200, json=completion("final-content-only"))

    vlm = QwenVLM(qwen_config, allow_paid=True, transport=httpx.MockTransport(handler))
    assert (
        vlm.complete(
            VLMRequest(
                purpose,
                "Distinct system instruction",
                "Original task JSON",
                (ImageInput("frame_7", initial_image),),
            )
        )
        == "final-content-only"
    )


def test_endpoint_env_and_trailing_slash(qwen_config, monkeypatch):
    normal = QwenVLM(
        qwen_config.model_copy(update={"base_url": qwen_config.base_url + "/"}), allow_paid=True
    )
    monkeypatch.setenv("QWEN_TEST_ENDPOINT", normal.bridge.endpoint)
    alternate = QwenVLM(
        qwen_config.model_copy(
            update={
                "base_url": None,
                "endpoint_env": "QWEN_TEST_ENDPOINT",
            }
        ),
        allow_paid=True,
    )
    assert alternate.bridge.endpoint == normal.bridge.endpoint


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"choices": []},
        {"choices": [None]},
        completion("partial content", "length"),
        completion("filtered", "content_filter"),
        completion(""),
        completion(None),
        completion([{"text": "wrong type"}]),
        {"choices": [{"finish_reason": "stop", "message": {"refusal": "refused"}}]},
    ],
)
def test_invalid_vendor_response_is_runtime_error(qwen_config, data):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    vlm = QwenVLM(qwen_config, allow_paid=True, transport=transport)
    with pytest.raises(ProviderError):
        vlm.complete(VLMRequest("qc", QC_SYSTEM, "original task"))


def test_qwen_api_failure_does_not_expose_token(qwen_config):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, text="test-qwen-token-never-log")
    )
    vlm = QwenVLM(qwen_config, allow_paid=True, transport=transport)
    with pytest.raises(ProviderError, match="401") as failure:
        vlm.complete(VLMRequest("qc", QC_SYSTEM, "task"))
    assert "test-qwen-token-never-log" not in str(failure.value)


@pytest.mark.parametrize("valid_json", [True, False])
def test_qc_pipeline_isolated_and_validated(
    qwen_config,
    task_path,
    existing_video,
    initial_image,
    tmp_path,
    monkeypatch,
    valid_json,
):
    requests = []
    raw = (
        json.dumps(
            {
                "checks": {
                    name: {
                        "status": "uncertain",
                        "reason": "Offline HTTP test fixture.",
                        "evidence_frames": [0],
                    }
                    for name in ("task_compliance", "scene_consistency", "visual_anomalies")
                }
            }
        )
        if valid_json
        else "invalid model JSON"
    )

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body["messages"][0]["content"] == QC_SYSTEM:
            blocks = body["messages"][1]["content"]
            assert "GENERATION_SENTINEL" not in json.dumps(body)
            assert len([block for block in blocks if block["type"] == "image_url"]) == 18
            assert json.loads(blocks[0]["text"])["original_task"]["task_id"] == "box_lift_001"
            return httpx.Response(200, json=completion(raw))
        return httpx.Response(200, json=completion("GENERATION_SENTINEL"))

    vlm = QwenVLM(qwen_config, allow_paid=True, transport=httpx.MockTransport(handler))
    vlm.complete(VLMRequest("image_prompt", "Earlier generation system", "GENERATION_SENTINEL"))
    monkeypatch.setattr(pipeline, "create_vlm", lambda *a, **kw: vlm)
    output = tmp_path / "qwen-qc"

    def run():
        return pipeline.judge_video(
            task_path,
            existing_video,
            output_dir=output,
            initial_image=initial_image,
            reference_image=initial_image,
        )

    if valid_json:
        run()
        report = json.loads((output / "qc_report.json").read_text())
        assert report["decision"] == "REVIEW"
        assert report["provider"] == "qwen"
        assert report["is_mock"] is False  # Adapter provenance; transport is mocked only in tests.
    else:
        with pytest.raises(JudgmentError):
            run()
        assert not (output / "qc_report.json").exists()
    assert len(requests) == 2
    envelope = json.loads((output / "vlm_raw_response.json").read_text())
    assert envelope["response_text"] == raw
    assert "internal-reasoning-sentinel" not in json.dumps(envelope)
