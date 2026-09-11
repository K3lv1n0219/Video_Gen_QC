from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from video_gen_qc.errors import ConfigError
from video_gen_qc.schemas import NonEmpty, StrictModel


class ProviderConfig(StrictModel):
    provider: Literal["mock", "http", "qwen"] = "mock"
    model: NonEmpty | None = None
    base_url: NonEmpty | None = None
    endpoint_env: NonEmpty | None = None
    api_key_env: NonEmpty | None = None
    timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=4096, gt=0)


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
