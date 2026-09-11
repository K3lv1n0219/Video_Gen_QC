"""Project-local secret persistence. Providers still read only process variables."""

import io
import os
from pathlib import Path

from dotenv import load_dotenv, set_key

from video_gen_qc.errors import ConfigError


def environment_path(config_path: Path | None = None, env_file: Path | None = None) -> Path:
    if env_file is not None:
        return env_file.expanduser().absolute()
    start = config_path.resolve().parent if config_path is not None else Path.cwd()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src/video_gen_qc").is_dir():
            return parent / ".env"
    # No repository found: use only the selected directory, never search ancestors for secrets.
    return start / ".env"


def load_environment(config_path: Path | None = None, env_file: Path | None = None) -> Path | None:
    """Load one file without replacing shell variables or evaluating/interpolating values."""
    if os.environ.get("PYTHON_DOTENV_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    path = environment_path(config_path, env_file)
    if not path.exists() and env_file is None:
        return None
    if not path.is_file():
        raise ConfigError(f"Environment file does not exist or is not a regular file: {path}")
    try:
        content = path.read_text(encoding="utf-8")
        load_dotenv(stream=io.StringIO(content), override=False, interpolate=False)
    except (OSError, UnicodeError, ValueError):
        # Encoding errors can embed secret bytes in their message; never propagate them.
        raise ConfigError(
            f"Cannot load environment file {path}; check permissions and UTF-8."
        ) from None
    return path


def save_qwen_key(key: str, env_file: Path | None = None) -> Path:
    """Save only the Qwen key, preserving other settings/comments and owner-only permissions."""
    key = key.strip()
    if not key or not key.isascii() or any(character.isspace() for character in key):
        raise ConfigError("API Key must be nonempty ASCII text without whitespace.")
    path = environment_path(env_file=env_file)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ConfigError(f"Key configuration requires a regular local file: {path}")
    try:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            path.chmod(0o600)
        else:
            os.close(descriptor)
        # python-dotenv rewrites atomically; the existing file mode is preserved.
        set_key(path, "DASHSCOPE_API_KEY", key, quote_mode="always")
        path.chmod(0o600)
    except (OSError, UnicodeError, ValueError):
        raise ConfigError(
            f"Cannot save API Key to {path}; check the directory and permissions."
        ) from None
    return path
