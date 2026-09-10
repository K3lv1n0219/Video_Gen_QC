"""Independent visual inspection; deliberately no imports of generation modules.

The QC system evaluates observable video quality and task compliance.
It does not estimate robot executability.
It does not use downstream manipulation success as a QC criterion.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from video_gen_qc.artifacts import write_json
from video_gen_qc.errors import JudgmentError
from video_gen_qc.prompts import QC_SYSTEM
from video_gen_qc.providers.base import VLM, ImageInput, VLMRequest
from video_gen_qc.schemas import (
    CheckSet,
    Decision,
    QCReport,
    SamplingResult,
    TaskSpec,
    VLMJudgment,
    parse_json,
)


def validate_judgment(raw: str, sampled_ids: set[int]) -> CheckSet:
    try:
        judgment = VLMJudgment.model_validate(parse_json(raw))
    except (ValueError, ValidationError) as exc:
        raise JudgmentError(f"VLM returned invalid QC JSON/schema: {exc}") from exc
    for name, check in judgment.checks:
        invalid = set(check.evidence_frames) - sampled_ids
        if invalid:
            raise JudgmentError(f"{name} cites nonexistent sampled frame IDs: {sorted(invalid)}")
        if len(set(check.evidence_frames)) != len(check.evidence_frames):
            raise JudgmentError(f"{name} contains duplicate evidence frame IDs.")
        if check.status != "uncertain" and not check.evidence_frames:
            raise JudgmentError(f"{name}: pass/fail requires at least one sampled evidence frame.")
    return judgment.checks


def aggregate(checks: CheckSet) -> Decision:
    statuses = {check.status for check in checks.values()}
    if "fail" in statuses:
        return "REJECT"
    if "uncertain" in statuses:
        return "REVIEW"
    return "PASS"


def judge(
    task: TaskSpec,
    sampling: SamplingResult,
    run_dir: Path,
    vlm: VLM,
    *,
    initial_image: Path | None = None,
    reference_image: Path | None = None,
) -> QCReport:
    """Build a fresh QC request from task + visual evidence only, never prompts."""
    images = []
    if reference_image is not None:
        images.append(ImageInput("reference_image", reference_image))
    if initial_image is not None:
        images.append(ImageInput("initial_image", initial_image))
    images.extend(
        ImageInput(f"frame_{frame.frame_id}", run_dir / frame.path) for frame in sampling.frames
    )
    request = VLMRequest(
        purpose="qc",
        system=QC_SYSTEM,
        text=json.dumps(
            {
                "original_task": task.model_dump(),
                "sampling": sampling.model_dump(),
                "image_labels": [image.label for image in images],
            },
            ensure_ascii=False,
        ),
        images=tuple(images),
    )
    # Reproducible request metadata uses local evidence paths, not duplicated image bytes.
    write_json(
        run_dir / "qc_request.json",
        {
            "purpose": request.purpose,
            "system": request.system,
            "payload": json.loads(request.text),
        },
    )
    raw = vlm.complete(request)
    write_json(
        run_dir / "vlm_raw_response.json",
        {
            "provider": vlm.provider_name,
            "model": vlm.model,
            "is_mock": vlm.is_mock,
            "response_text": raw,
        },
    )
    checks = validate_judgment(raw, {frame.frame_id for frame in sampling.frames})
    report = QCReport(
        task_id=task.task_id,
        decision=aggregate(checks),
        checks=checks,
        is_mock=vlm.is_mock,
        provider=vlm.provider_name,
        model=vlm.model,
        sampling=sampling,
    )
    write_json(run_dir / "qc_report.json", report.model_dump())
    return report
