from video_gen_qc.config import ProviderConfig
from video_gen_qc.providers.base import VLM, ImageGenerator, VideoGenerator
from video_gen_qc.providers.http import HTTPVLM, HTTPBridge, HTTPImageGenerator, HTTPVideoGenerator
from video_gen_qc.providers.mock import MockImageGenerator, MockVideoGenerator, MockVLM


def create_vlm(config: ProviderConfig, *, allow_paid: bool = False) -> VLM:
    if config.provider == "mock":
        return MockVLM(config.model)
    return HTTPVLM(HTTPBridge(config, allow_paid=allow_paid))


def create_image_generator(config: ProviderConfig, *, allow_paid: bool = False) -> ImageGenerator:
    if config.provider == "mock":
        return MockImageGenerator()
    return HTTPImageGenerator(HTTPBridge(config, allow_paid=allow_paid))


def create_video_generator(config: ProviderConfig, *, allow_paid: bool = False) -> VideoGenerator:
    if config.provider == "mock":
        return MockVideoGenerator()
    return HTTPVideoGenerator(HTTPBridge(config, allow_paid=allow_paid))
