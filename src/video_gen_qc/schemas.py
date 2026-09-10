"""Small, closed schemas: no downstream signals or numerical confidence fields."""

import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from video_gen_qc.errors import InputError

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Status = Literal["pass", "fail", "uncertain"]
Decision = Literal["PASS", "REVIEW", "REJECT"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskSpec(StrictModel):
    task_id: NonEmpty
    instruction: NonEmpty
    scene_constraints: list[NonEmpty] = Field(default_factory=list)
    initial_state_constraints: list[NonEmpty] = Field(default_factory=list)
    required_events: list[NonEmpty] = Field(default_factory=list)


class QCCheck(StrictModel):
    status: Status
    reason: NonEmpty
    evidence_frames: list[Annotated[int, Field(ge=0)]]


class CheckSet(StrictModel):
    task_compliance: QCCheck
    scene_consistency: QCCheck
    visual_anomalies: QCCheck

    def values(self) -> tuple[QCCheck, ...]:
        return (self.task_compliance, self.scene_consistency, self.visual_anomalies)


class VLMJudgment(StrictModel):
    checks: CheckSet


class SampledFrame(StrictModel):
    frame_id: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0, allow_inf_nan=False)
    path: str


class SamplingResult(StrictModel):
    decoded_frame_count: int = Field(gt=0)
    requested_frame_count: int = Field(ge=2)
    frames: list[SampledFrame]
    timestamp_basis: str = "source presentation timestamps (PTS), in seconds"


INSPECTION_SCOPE = (
    "Only the listed sampled video frames and supplied still images were provided. "
    "Unsampled moments were not inspected; pass means no explicit violation was found "
    "in the inspected evidence, not certification of the whole video or its physics."
)


class QCReport(StrictModel):
    schema_version: str = "1.0"
    task_id: str
    decision: Decision
    is_mock: bool
    provider: str
    model: str | None
    checks: CheckSet
    sampling: SamplingResult
    inspection_scope: str = INSPECTION_SCOPE


def parse_json(text: str) -> object:
    """Reject duplicate keys and nonstandard NaN/Infinity instead of silently accepting."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def load_task(path: Path) -> TaskSpec:
    try:
        text = path.read_text(encoding="utf-8")
        data = (
            yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else parse_json(text)
        )
        return TaskSpec.model_validate(data)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError, ValidationError) as exc:
        raise InputError(f"Cannot load task {path}: {exc}") from exc
