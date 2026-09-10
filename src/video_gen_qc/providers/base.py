from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ImageInput:
    label: str
    path: Path


@dataclass(frozen=True)
class VLMRequest:
    """A single isolated request, deliberately without conversation/history fields."""

    purpose: Literal["image_prompt", "video_prompt", "qc"]
    system: str
    text: str
    images: tuple[ImageInput, ...] = ()


class VLM(ABC):
    provider_name: str
    model: str | None
    is_mock: bool = False

    @abstractmethod
    def complete(self, request: VLMRequest) -> str:
        pass


class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, output_path: Path) -> Path:
        pass


class VideoGenerator(ABC):
    @abstractmethod
    def generate(self, image_path: Path, prompt: str, output_path: Path) -> Path:
        pass
