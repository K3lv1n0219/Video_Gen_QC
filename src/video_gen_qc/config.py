from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator

from video_gen_qc.errors import ConfigError
from video_gen_qc.schemas import NonEmpty, StrictModel


class ImageOptions(StrictModel):
    size: str = "1024*1024"
    prompt_extend: bool = False
    seed: int | None = Field(default=None, ge=0, le=2147483647)

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: str) -> str:
        try:
            width, height = (int(part) for part in value.split("*"))
            valid = (
                width > 0
                and height > 0
                and 512 * 512 <= width * height <= 2048 * 2048
                and max(width, height) <= 8 * min(width, height)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Image size must be WIDTH*HEIGHT, 512²–2048² pixels, ratio ≤8:1.")
        return value


class VideoOptions(StrictModel):
    resolution: Literal["480P", "720P", "1080P"] = "720P"
    duration: int = Field(default=5, ge=2, le=30)
    audio: bool = False
    prompt_extend: bool = False
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    poll_interval_seconds: float = Field(default=15.0, gt=0, le=60, allow_inf_nan=False)
    task_timeout_seconds: float = Field(default=900.0, gt=0, allow_inf_nan=False)


class ProviderConfig(StrictModel):
    provider: Literal["mock", "http", "qwen", "qwen_image", "wan"] = "mock"
    model: NonEmpty | None = None
    base_url: NonEmpty | None = None
    endpoint_env: NonEmpty | None = None
    api_key_env: NonEmpty | None = None
    timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=4096, gt=0)
    image_options: ImageOptions | None = None
    video_options: VideoOptions | None = None


class QCConfig(StrictModel):
    sample_frames: int = Field(default=16, ge=2)


class OutputConfig(StrictModel):
    root: NonEmpty = "outputs"


class AppConfig(StrictModel):
    vlm: ProviderConfig = Field(default_factory=lambda: ProviderConfig(model="mock-vlm"))
    image_generation: ProviderConfig = Field(default_factory=ProviderConfig)
    video_generation: ProviderConfig = Field(default_factory=ProviderConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return AppConfig.model_validate(data)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Cannot load configuration {path}: {exc}") from exc
