"""Optional real HTTP bridge; see README for the provider-neutral wire contract.

This is not a drop-in adapter for a particular vendor. It calls an explicitly
configured bridge service and never substitutes mock results on failure.
"""

import base64
import binascii
import io
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from video_gen_qc.config import ProviderConfig
from video_gen_qc.errors import ConfigError, ProviderError
from video_gen_qc.providers.base import VLM, ImageGenerator, ImageInput, VideoGenerator, VLMRequest
from video_gen_qc.schemas import parse_json


def encode_image(image: ImageInput) -> dict:
    with Image.open(image.path) as source:
        buffer = io.BytesIO()
        source.convert("RGB").save(buffer, format="PNG")
    return {
        "label": image.label,
        "mime_type": "image/png",
        "data_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


class HTTPBridge:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        allow_paid: bool,
        transport: httpx.BaseTransport | None = None,
        endpoint: str | None = None,
    ):
        if not allow_paid:
            raise ConfigError("Real HTTP providers require explicit --allow-paid (including VLM).")
        if (endpoint is None and not config.endpoint_env) or not config.api_key_env:
            raise ConfigError("HTTP provider requires an endpoint and api_key_env configuration.")
        if endpoint is None:
            endpoint = os.environ.get(config.endpoint_env, "").strip()
        token = os.environ.get(config.api_key_env, "").strip()
        if not endpoint:
            raise ConfigError(
                f"Missing provider endpoint environment variable: {config.endpoint_env}"
            )
        if not token:
            raise ConfigError(
                f"Missing provider credential environment variable: {config.api_key_env}"
            )
        try:
            url = urlsplit(endpoint)
            _ = url.port  # Validate malformed/out-of-range ports before any provider call.
            local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
            valid = url.hostname and (url.scheme == "https" or local_http)
            if not valid or url.username or url.password or url.query or url.fragment:
                raise ValueError
        except ValueError as exc:
            raise ConfigError(
                "Provider URL must be valid HTTPS (or local HTTP), without credentials/query."
            ) from exc
        self.config = config
        self.endpoint = endpoint
        self.token = token
        self.transport = transport

    def post(self, payload: dict) -> dict:
        try:
            # A fresh client/request has no generation conversation or persisted cookies.
            with httpx.Client(
                timeout=self.config.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={"model": self.config.model, **payload},
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Do not expose server bodies or URLs, which may echo secrets.
            raise ProviderError(
                f"HTTP provider returned status {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(f"HTTP provider request failed ({type(exc).__name__}).") from exc
        try:
            data = parse_json(response.text)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            return data
        except ValueError as exc:
            raise ProviderError(
                "HTTP provider returned invalid JSON; expected a JSON object."
            ) from exc


class HTTPVLM(VLM):
    provider_name = "http"

    def __init__(self, bridge: HTTPBridge):
        self.bridge = bridge
        self.model = bridge.config.model

    def complete(self, request: VLMRequest) -> str:
        data = self.bridge.post(
            {
                "purpose": request.purpose,
                "system": request.system,
                "text": request.text,
                "images": [encode_image(image) for image in request.images],
            }
        )
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("HTTP VLM response must contain a nonempty text string.")
        return text


def decode_media(data: dict, key: str) -> bytes:
    try:
        value = data[key]
        if not isinstance(value, str) or not value:
            raise ValueError
        decoded = base64.b64decode(value, validate=True)
        if not decoded:
            raise ValueError
        return decoded
    except (KeyError, ValueError, binascii.Error) as exc:
        raise ProviderError(f"HTTP provider must return valid, nonempty {key}.") from exc


class HTTPImageGenerator(ImageGenerator):
    def __init__(self, bridge: HTTPBridge):
        self.bridge = bridge

    def generate(self, prompt: str, output_path: Path) -> Path:
        data = self.bridge.post({"purpose": "image_generation", "prompt": prompt})
        content = decode_media(data, "image_base64")
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.convert("RGB").save(output_path, format="PNG")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ProviderError("HTTP image provider returned an unreadable image.") from exc
        return output_path


class HTTPVideoGenerator(VideoGenerator):
    def __init__(self, bridge: HTTPBridge):
        self.bridge = bridge

    def generate(self, image_path: Path, prompt: str, output_path: Path) -> Path:
        data = self.bridge.post(
            {
                "purpose": "video_generation",
                "prompt": prompt,
                "initial_image": encode_image(ImageInput("initial_image", image_path)),
            }
        )
        output_path.write_bytes(decode_media(data, "video_base64"))
        # The pipeline decoder validates media before any QC call is made.
        return output_path
