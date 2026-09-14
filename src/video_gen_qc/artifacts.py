import hashlib
import json
import platform
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from video_gen_qc import __version__
from video_gen_qc.errors import InputError, OutputError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def new_run_dir(root: Path, task_id: str, explicit: Path | None = None) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_id).strip("_")[:64] or "task"
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = explicit if explicit is not None else root / f"{name}_{slug}_{uuid4().hex[:8]}"
    try:
        path.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise OutputError(f"Cannot create isolated run directory {path}: {exc}") from exc
    return path


def copy_image(source: Path, destination: Path) -> Path:
    try:
        with Image.open(source) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(destination, format="PNG")
        return destination
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise InputError(f"Cannot read initial/reference image {source}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def run_record(path: Path, task: dict, config: dict, mode: str, inputs: dict, active: list[str]):
    metadata = {
        "schema_version": "1.0",
        "package_version": __version__,
        "python_version": platform.python_version(),
        "dependency_versions": {
            name: version(name)
            for name in ("av", "Pillow", "pydantic", "PyYAML", "httpx", "python-dotenv")
        },
        "mode": mode,
        "started_at": utc_now(),
        "status": "running",
        "config": config,
        "inputs": inputs,
        "active_providers": active,
    }
    try:
        write_json(path / "task.json", task)
        write_json(path / "run_metadata.json", metadata)
        yield metadata
        metadata.update(status="completed", completed_at=utc_now())
        metadata["artifacts_sha256"] = {
            str(file.relative_to(path)): sha256(file)
            for file in sorted(path.rglob("*"))
            if file.is_file() and file.name != "run_metadata.json"
        }
        write_json(path / "run_metadata.json", metadata)
    except Exception as exc:
        metadata.update(
            status="error",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        try:
            write_json(path / "run_metadata.json", metadata)
        except OSError:
            pass  # Preserve the original error if output storage itself failed.
        raise
