import json

import pytest

from video_gen_qc.config import AppConfig, load_config
from video_gen_qc.errors import ConfigError, InputError
from video_gen_qc.schemas import load_task


def test_valid_task(task_path):
    task = load_task(task_path)
    assert task.task_id == "box_lift_001"
    assert len(task.required_events) == 3


def test_yaml_task(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text('task_id: approach\ninstruction: "Approach but do not touch."\n')
    assert load_task(path).instruction == "Approach but do not touch."


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"task_id": "id"},
        {"instruction": "lift"},
        {"task_id": "id", "instruction": " "},
        {"task_id": "id", "instruction": 3},
        {"task_id": "id", "instruction": "lift", "required_events": "lift"},
        {"task_id": "id", "instruction": "lift", "extra_metric": 0.9},
    ],
)
def test_invalid_task(tmp_path, data):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(data))
    with pytest.raises(InputError):
        load_task(path)


@pytest.mark.parametrize("text", ["{", '{"task_id":"a","task_id":"b","instruction":"lift"}'])
def test_invalid_json(tmp_path, text):
    path = tmp_path / "task.json"
    path.write_text(text)
    with pytest.raises(InputError):
        load_task(path)


def test_missing_task(tmp_path):
    with pytest.raises(InputError, match="Cannot load task"):
        load_task(tmp_path / "missing.json")


def test_config_defaults_match_file(task_path):
    assert load_config(task_path.parents[1] / "configs" / "default.yaml") == AppConfig()


@pytest.mark.parametrize("text", ["[]", "qc:\n  sample_frames: 1", "vlm:\n  provider: typo"])
def test_invalid_config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)
