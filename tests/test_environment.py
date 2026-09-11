import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from video_gen_qc import cli
from video_gen_qc.environment import environment_path, load_environment, save_qwen_key
from video_gen_qc.errors import ConfigError


@pytest.fixture(autouse=True)
def enable_test_dotenv(monkeypatch):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    for name in ("DASHSCOPE_API_KEY", "VIDEO_QC_TEST_ENDPOINT", "VIDEO_QC_TEST_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src/video_gen_qc").mkdir(parents=True)
    (root / "pyproject.toml").touch()
    (root / "configs").mkdir()
    monkeypatch.chdir(root)
    return root


def test_configure_saves_once_without_echo(project, monkeypatch, capsys):
    key = "test-only-persistent-key"
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: key)
    assert cli.main(["configure"]) == 0
    path = project / ".env"
    assert dotenv_values(path)["DASHSCOPE_API_KEY"] == key
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    output = capsys.readouterr()
    assert key not in output.out + output.err
    assert str(path) in output.out
    load_environment()
    assert os.environ["DASHSCOPE_API_KEY"] == key


def test_save_preserves_settings_comments_and_literal_value(project):
    path = project / ".env"
    path.write_text(
        "# keep this comment\nVIDEO_QC_TEST_ENDPOINT=https://example.test\nDASHSCOPE_API_KEY=old\n"
    )
    path.chmod(0o644)
    key = "test-$LITERAL-${VIDEO_QC_TEST_TOKEN}\\'suffix"
    assert save_qwen_key(key) == path
    assert "# keep this comment" in path.read_text()
    values = dotenv_values(path, interpolate=False)
    assert values["DASHSCOPE_API_KEY"] == key
    assert values["VIDEO_QC_TEST_ENDPOINT"] == "https://example.test"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    load_environment()
    assert os.environ["DASHSCOPE_API_KEY"] == key


def test_shell_value_wins(project, monkeypatch):
    save_qwen_key("file-token")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "shell-token")
    load_environment()
    assert os.environ["DASHSCOPE_API_KEY"] == "shell-token"


def test_saved_key_loads_in_a_fresh_process(project):
    save_qwen_key("test-only-restart-token")
    child_environment = dict(os.environ)
    child_environment.pop("DASHSCOPE_API_KEY", None)
    child_environment.pop("PYTHON_DOTENV_DISABLED", None)
    child_environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; from video_gen_qc.environment import load_environment; "
                "load_environment(); "
                "print(os.environ.get('DASHSCOPE_API_KEY') == 'test-only-restart-token')"
            ),
        ],
        cwd=project,
        env=child_environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "True"
    assert "test-only-restart-token" not in result.stdout + result.stderr


def test_locates_project_from_config_and_subdirectory(project, monkeypatch, tmp_path):
    save_qwen_key("persistent-token")
    config = project / "configs" / "qwen.yaml"
    config.touch()
    monkeypatch.chdir(tmp_path)
    assert load_environment(config) == project / ".env"
    assert os.environ["DASHSCOPE_API_KEY"] == "persistent-token"
    monkeypatch.chdir(project / "configs")
    assert environment_path() == project / ".env"


def test_explicit_env_file_takes_precedence(project, tmp_path):
    save_qwen_key("project-token")
    selected = tmp_path / "selected.env"
    save_qwen_key("explicit-token", selected)
    assert load_environment(env_file=selected) == selected
    assert os.environ["DASHSCOPE_API_KEY"] == "explicit-token"


def test_missing_default_is_optional_but_explicit_is_required(project):
    assert load_environment() is None
    with pytest.raises(ConfigError, match="does not exist"):
        load_environment(env_file=project / "missing.env")


def test_does_not_load_parent_secrets_outside_project(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=parent-secret\n")
    child = tmp_path / "unrelated"
    child.mkdir()
    monkeypatch.chdir(child)
    assert load_environment() is None
    assert "DASHSCOPE_API_KEY" not in os.environ


def test_can_disable_loading(project, monkeypatch):
    save_qwen_key("file-token")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    assert load_environment() is None
    assert "DASHSCOPE_API_KEY" not in os.environ


@pytest.mark.parametrize("key", ["", " ", "with space", "with\nnewline", "非ASCII"])
def test_invalid_key_does_not_create_file(project, key):
    with pytest.raises(ConfigError):
        save_qwen_key(key)
    assert not (project / ".env").exists()


def test_configure_cancellation_leaves_existing_file(project, monkeypatch):
    path = save_qwen_key("keep-token")
    original = path.read_bytes()

    def cancel(prompt):
        raise EOFError

    monkeypatch.setattr(cli.getpass, "getpass", cancel)
    assert cli.main(["configure"]) == 2
    assert path.read_bytes() == original


def test_saving_rejects_symlink(project, tmp_path):
    target = tmp_path / "other-file"
    target.write_text("preserve me")
    (project / ".env").symlink_to(target)
    with pytest.raises(ConfigError, match="regular local file"):
        save_qwen_key("new-token")
    assert target.read_text() == "preserve me"


def test_invalid_encoding_does_not_leak_content(project):
    (project / ".env").write_bytes(b"DASHSCOPE_API_KEY=secret-sentinel\xff")
    with pytest.raises(ConfigError) as failure:
        load_environment()
    assert "secret-sentinel" not in str(failure.value)


def test_cli_loads_key_before_provider_and_keeps_paid_gate(project, task_path, capsys):
    save_qwen_key("test-only-secret")
    original_config = task_path.parents[1] / "configs/qwen-beijing.yaml"
    config = project / "configs/qwen.yaml"
    config.write_bytes(original_config.read_bytes())
    assert cli.main(["run", "--task", str(task_path), "--config", str(config)]) == 2
    assert os.environ["DASHSCOPE_API_KEY"] == "test-only-secret"
    output = capsys.readouterr()
    assert "--allow-paid" in output.err
    assert "test-only-secret" not in output.out + output.err
    assert not (project / "outputs").exists()


def test_mock_cli_loads_dotenv_without_leaking_into_artifacts(project, task_path):
    key = "test-only-artifact-secret"
    save_qwen_key(key)
    output = project / "run"
    assert cli.main(["run", "--task", str(task_path), "--output-dir", str(output)]) == 0
    assert os.environ["DASHSCOPE_API_KEY"] == key
    assert json.loads((output / "qc_report.json").read_text())["decision"] == "REVIEW"
    for path in output.rglob("*"):
        if path.suffix in {".txt", ".json"}:
            assert key not in path.read_text()
