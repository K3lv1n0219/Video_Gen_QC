from video_gen_qc.config import ProviderConfig
from video_gen_qc.errors import ConfigError
from video_gen_qc.providers.base import VLM, ImageGenerator, VideoGenerator
from video_gen_qc.providers.dashscope import QwenImageGenerator, WanVideoGenerator
from video_gen_qc.providers.http import HTTPVLM, HTTPBridge, HTTPImageGenerator, HTTPVideoGenerator
from video_gen_qc.providers.mock import MockImageGenerator, MockVideoGenerator, MockVLM
from video_gen_qc.providers.qwen import QwenVLM


def create_vlm(config: ProviderConfig, *, allow_paid: bool = False) -> VLM:
    if config.provider == "mock":
        return MockVLM(config.model)
    if config.provider == "qwen":
        return QwenVLM(config, allow_paid=allow_paid)
    if config.provider != "http":
        raise ConfigError("VLM requires mock, http, or qwen provider.")
    return HTTPVLM(HTTPBridge(config, allow_paid=allow_paid))


def create_image_generator(config: ProviderConfig, *, allow_paid: bool = False) -> ImageGenerator:
    if config.provider == "qwen":
        raise ConfigError("The qwen provider supports VLM calls only, not image generation.")
    if config.provider == "mock":
        return MockImageGenerator()
    if config.provider == "qwen_image":
        return QwenImageGenerator(config, allow_paid=allow_paid)
    if config.provider != "http":
        raise ConfigError("Image generation requires mock, http, or qwen_image provider.")
    return HTTPImageGenerator(HTTPBridge(config, allow_paid=allow_paid))


def create_video_generator(config: ProviderConfig, *, allow_paid: bool = False) -> VideoGenerator:
    if config.provider == "qwen":
        raise ConfigError("The qwen provider supports VLM calls only, not video generation.")
    if config.provider == "mock":
        return MockVideoGenerator()
    if config.provider == "wan":
        return WanVideoGenerator(config, allow_paid=allow_paid)
    if config.provider != "http":
        raise ConfigError("Video generation requires mock, http, or wan provider.")
    return HTTPVideoGenerator(HTTPBridge(config, allow_paid=allow_paid))
