"""Direct Alibaba Cloud Qwen vision adapter using its compatible Chat HTTP API.

No OpenAI SDK or bridge server is required. Each call has a fresh system/user
pair; prompt-generation history is never included in independent QC requests.
"""

import httpx

from video_gen_qc.config import ProviderConfig
from video_gen_qc.errors import ConfigError, ProviderError
from video_gen_qc.providers.base import VLM, VLMRequest
from video_gen_qc.providers.http import HTTPBridge, encode_image


class QwenVLM(VLM):
    provider_name = "qwen"

    def __init__(
        self,
        config: ProviderConfig,
        *,
        allow_paid: bool = False,
        transport: httpx.BaseTransport | None = None,
    ):
        if not allow_paid:
            raise ConfigError("Qwen API calls require explicit --allow-paid.")
        if not config.model:
            raise ConfigError("Qwen requires a vision-capable model name in vlm.model.")
        if config.base_url and config.endpoint_env:
            raise ConfigError("Qwen: configure either base_url or endpoint_env, not both.")
        if not config.base_url and not config.endpoint_env:
            raise ConfigError("Qwen requires a regional base_url or endpoint_env.")
        endpoint = None
        if config.base_url:
            base_url = config.base_url.rstrip("/")
            if not base_url.endswith("/compatible-mode/v1"):
                raise ConfigError("Qwen base_url must end with /compatible-mode/v1.")
            endpoint = base_url + "/chat/completions"
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.bridge = HTTPBridge(
            config, allow_paid=allow_paid, transport=transport, endpoint=endpoint
        )

    def complete(self, request: VLMRequest) -> str:
        content = [{"type": "text", "text": request.text}]
        for image in request.images:
            encoded = encode_image(image)
            # Adjacent labels tie evidence IDs to actual image bytes, not filenames.
            content.append({"type": "text", "text": f"Image label: {image.label}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded['data_base64']}"},
                }
            )
        payload = {
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": content},
            ],
            "stream": False,
            "enable_thinking": False,
            "max_tokens": self.max_tokens,
        }
        if request.purpose == "qc":
            payload["response_format"] = {"type": "json_object"}
        data = self.bridge.post(payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderError("Qwen returned an invalid choices response.")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ProviderError(
                "Qwen did not finish normally (truncation, filtering, or unsupported response). "
                "Check the model and max_tokens; no QC judgment was accepted."
            )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("refusal"):
            raise ProviderError("Qwen returned no usable assistant message or refused the request.")
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Qwen returned empty or non-text content.")
        # Return the unmodified final content. qc.py saves it, validates the checks,
        # and applies the decision policy; reasoning_content is never propagated.
        return text
