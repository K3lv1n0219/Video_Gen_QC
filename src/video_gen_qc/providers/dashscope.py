"""Native DashScope media APIs. One submission per generation; no hidden retries."""

import base64
import ipaddress
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from PIL import Image, UnidentifiedImageError

from video_gen_qc.artifacts import write_json
from video_gen_qc.config import ImageOptions, ProviderConfig, VideoOptions
from video_gen_qc.errors import ConfigError, InputError, ProviderError
from video_gen_qc.providers.base import ImageGenerator, ImageInput, VideoGenerator
from video_gen_qc.providers.http import HTTPBridge, encode_image
from video_gen_qc.schemas import parse_json

IMAGE_ENDPOINT = "/services/aigc/multimodal-generation/generation"
VIDEO_ENDPOINT = "/services/aigc/video-generation/video-synthesis"
IMAGE_MODELS = {"qwen-image-3.0-pro", "qwen-image-3.0"}
VIDEO_MODELS = {"wan3.0-video", "wan3.0-video-prime"}


def _require_new_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise InputError("Generation output already exists; refusing to overwrite it.")
    if not path.parent.is_dir():
        raise InputError("Generation output directory must exist before submitting a request.")


class DashScopeClient:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        allow_paid: bool = False,
        transport: httpx.BaseTransport | None = None,
    ):
        if not config.base_url or config.endpoint_env:
            raise ConfigError("DashScope media requires base_url ending in /api/v1.")
        self.base_url = config.base_url.rstrip("/")
        if not self.base_url.endswith("/api/v1"):
            raise ConfigError("DashScope media base_url must end with /api/v1.")
        # Reuse the environment-only credential and HTTPS validation of other providers.
        self.bridge = HTTPBridge(
            config, allow_paid=allow_paid, transport=transport, endpoint=self.base_url
        )
        self.config = config
        self.transport = transport

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        asynchronous: bool = False,
        timeout: float | None = None,
    ) -> dict:
        headers = {"Authorization": f"Bearer {self.bridge.token}"}
        if asynchronous:
            headers["X-DashScope-Async"] = "enable"
        try:
            with httpx.Client(
                timeout=timeout or self.config.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = client.request(
                    method, self.base_url + path, headers=headers, json=payload
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"DashScope returned HTTP {exc.response.status_code}; check model access, "
                "region, account balance, and request parameters."
            ) from None
        except httpx.RequestError as exc:
            raise ProviderError(
                f"DashScope request failed ({type(exc).__name__}); no submission was retried."
            ) from None
        try:
            data = parse_json(response.text)
        except ValueError:
            raise ProviderError("DashScope returned invalid JSON.") from None
        if not isinstance(data, dict) or data.get("code"):
            # Vendor errors can echo credentials or signed URLs; never propagate their text.
            raise ProviderError("DashScope returned an error or invalid response object.")
        return data

    def download(self, url: str, destination: Path, *, max_bytes: int) -> None:
        _require_new_output(destination)
        try:
            parsed = urlsplit(url)
            host = parsed.hostname or ""
            valid = (
                parsed.scheme == "https"
                and host
                and not parsed.username
                and not parsed.password
                and not parsed.fragment
                and parsed.port in {None, 443}
                and "." in host
                and not host.endswith((".localhost", ".local"))
            )
            try:
                valid = valid and ipaddress.ip_address(host).is_global
            except ValueError:
                pass
            if not valid:
                raise ValueError
        except (TypeError, ValueError):
            raise ProviderError("DashScope returned an invalid media download URL.") from None
        temporary = destination.with_name(destination.name + ".part")
        created = False
        try:
            # Signed media URLs authenticate themselves. Never send the API Key to storage.
            with httpx.Client(
                timeout=self.config.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    total = 0
                    with temporary.open("xb") as handle:
                        created = True
                        for chunk in response.iter_bytes(chunk_size=64 * 1024):
                            total += len(chunk)
                            if total > max_bytes:
                                raise ProviderError("Generated media exceeds the download limit.")
                            handle.write(chunk)
                    if not total:
                        raise ProviderError("DashScope returned empty media.")
            temporary.replace(destination)
        except httpx.HTTPError:
            raise ProviderError("Generated media download failed; URL omitted.") from None
        finally:
            if created:
                temporary.unlink(missing_ok=True)


class QwenImageGenerator(ImageGenerator):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        allow_paid: bool = False,
        transport: httpx.BaseTransport | None = None,
    ):
        if config.model not in IMAGE_MODELS:
            raise ConfigError("qwen_image supports qwen-image-3.0-pro and qwen-image-3.0.")
        self.client = DashScopeClient(config, allow_paid=allow_paid, transport=transport)
        self.options = config.image_options or ImageOptions()

    def generate(
        self, prompt: str, output_path: Path, *, reference_image: Path | None = None
    ) -> Path:
        _require_new_output(output_path)
        content = []
        if reference_image is not None:
            try:
                encoded = encode_image(ImageInput("reference_image", reference_image))[
                    "data_base64"
                ]
            except (OSError, ValueError, UnidentifiedImageError):
                raise InputError("Qwen Image reference must be a readable local image.") from None
            encoded_bytes = len(encoded) // 4 * 3 - (len(encoded) - len(encoded.rstrip("=")))
            if encoded_bytes > 10 * 1024 * 1024:
                raise InputError("Qwen Image reference exceeds 10 MB after PNG conversion.")
            content.append({"image": f"data:image/png;base64,{encoded}"})
        content.append({"text": prompt})
        parameters = self.options.model_dump(exclude_none=True)
        parameters["n"] = 1
        data = self.client.request(
            "POST",
            IMAGE_ENDPOINT,
            {
                "model": self.client.config.model,
                "input": {"messages": [{"role": "user", "content": content}]},
                "parameters": parameters,
            },
        )
        try:
            choices = data["output"]["choices"]
            if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
                raise ValueError
            content = choices[0]["message"]["content"]
            images = [item["image"] for item in content if "image" in item]
            if len(images) != 1 or not isinstance(images[0], str):
                raise ValueError
        except (KeyError, TypeError, ValueError, IndexError):
            raise ProviderError("Qwen Image did not return exactly one completed image.") from None
        downloaded = output_path.with_name(output_path.name + ".download")
        downloaded_ready = False
        try:
            self.client.download(images[0], downloaded, max_bytes=20 * 1024 * 1024)
            downloaded_ready = True
            with Image.open(downloaded) as source:
                source.convert("RGB").save(output_path, format="PNG")
        except (OSError, UnidentifiedImageError, ValueError):
            raise ProviderError("Qwen Image returned an unreadable image.") from None
        finally:
            if downloaded_ready:
                downloaded.unlink(missing_ok=True)
        return output_path


def _task_id(value: object) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise ProviderError("Wan returned an invalid task ID.") from None
    return value


class WanVideoGenerator(VideoGenerator):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        allow_paid: bool = False,
        transport: httpx.BaseTransport | None = None,
    ):
        if config.model not in VIDEO_MODELS:
            raise ConfigError("wan supports wan3.0-video and wan3.0-video-prime.")
        self.client = DashScopeClient(config, allow_paid=allow_paid, transport=transport)
        self.options = config.video_options or VideoOptions()

    def generate(self, image_path: Path, prompt: str, output_path: Path) -> Path:
        _require_new_output(output_path)
        if not prompt.strip() or len(prompt) > 20000:
            raise InputError("Wan prompt must be nonempty and at most 20000 characters.")
        try:
            with Image.open(image_path) as source:
                width, height = source.size
                source.verify()
                if source.format != "PNG" or source.mode != "RGB":
                    raise ValueError
            if not (240 <= min(width, height) and max(width, height) <= 8000):
                raise ValueError
            if max(width, height) > 8 * min(width, height):
                raise ValueError
            if not 0 < image_path.stat().st_size <= 20 * 1024 * 1024:
                raise ValueError
        except (OSError, ValueError, UnidentifiedImageError):
            raise InputError(
                "Wan needs an RGB PNG first frame, 240–8000 pixels per side, "
                "ratio ≤8:1, at most 20 MB."
            ) from None
        parameters = self.options.model_dump(
            exclude={"poll_interval_seconds", "task_timeout_seconds"}, exclude_none=True
        )
        parameters["ratio"] = "adaptive"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data = self.client.request(
            "POST",
            VIDEO_ENDPOINT,
            {
                "model": self.client.config.model,
                "input": {
                    "prompt": prompt,
                    "media": [{"type": "first_frame", "url": f"data:image/png;base64,{encoded}"}],
                },
                "parameters": parameters,
            },
            asynchronous=True,
        )
        output = data.get("output")
        if not isinstance(output, dict):
            raise ProviderError("Wan submission returned no task object; submission not retried.")
        task_id = _task_id(output.get("task_id"))
        record_path = output_path.with_name("video_generation_job.json")
        self._record(record_path, task_id, "SUBMITTED")
        return self.resume(task_id, output_path)

    def _record(self, path: Path, task_id: str, status: str) -> None:
        write_json(
            path,
            {"model": self.client.config.model, "task_id": task_id, "task_status": status},
        )

    def resume(self, task_id: str, output_path: Path) -> Path:
        """Fetch an existing job without submitting/charging for another generation."""
        _require_new_output(output_path)
        task_id = _task_id(task_id)
        record_path = output_path.with_name("video_generation_job.json")
        deadline = time.monotonic() + self.options.task_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError(
                    f"Wan polling timed out for task {task_id}. It may still be running; "
                    "resume this task instead of submitting another generation."
                )
            data = self.client.request(
                "GET",
                f"/tasks/{task_id}",
                timeout=min(remaining, self.client.config.timeout_seconds),
            )
            output = data.get("output")
            if not isinstance(output, dict) or output.get("task_id") != task_id:
                raise ProviderError("Wan returned a mismatched or invalid task response.")
            status = output.get("task_status")
            if not isinstance(status, str) or status not in {
                "PENDING",
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                "CANCELED",
                "UNKNOWN",
            }:
                raise ProviderError("Wan returned an unknown task status.")
            self._record(record_path, task_id, status)
            if status == "SUCCEEDED":
                url = output.get("video_url")
                if not isinstance(url, str) or not url:
                    raise ProviderError("Wan completed without a video URL.")
                self.client.download(url, output_path, max_bytes=200 * 1024 * 1024)
                return output_path
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                raise ProviderError(f"Wan task {task_id} ended with status {status}.")
            time.sleep(min(self.options.poll_interval_seconds, max(0, deadline - time.monotonic())))
