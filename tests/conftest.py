import socket
from pathlib import Path

import pytest

from video_gen_qc.providers.mock import MockImageGenerator, MockVideoGenerator


@pytest.fixture(autouse=True)
def ignore_personal_dotenv(monkeypatch):
    # Tests must not load the user's real local credentials. Environment tests opt back in.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.fixture
def task_path():
    return Path(__file__).resolve().parents[1] / "examples" / "box_lift.json"


@pytest.fixture
def initial_image(tmp_path):
    return MockImageGenerator().generate("initial-state fixture", tmp_path / "fixture.png")


@pytest.fixture
def existing_video(tmp_path, initial_image):
    return MockVideoGenerator().generate(initial_image, "fixture", tmp_path / "fixture.mp4")
