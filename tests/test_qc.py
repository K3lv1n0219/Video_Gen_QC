import itertools
import json

import pytest

from video_gen_qc.errors import JudgmentError
from video_gen_qc.qc import aggregate, validate_judgment

NAMES = ("task_compliance", "scene_consistency", "visual_anomalies")


def response(statuses=("pass", "pass", "pass"), evidence=None):
    return {
        "checks": {
            name: {
                "status": status,
                "reason": "Visible sampled evidence.",
                "evidence_frames": [0, 2] if evidence is None else evidence,
            }
            for name, status in zip(NAMES, statuses, strict=True)
        }
    }


@pytest.mark.parametrize(
    "statuses", list(itertools.product(["pass", "fail", "uncertain"], repeat=3))
)
def test_all_policy_combinations(statuses):
    checks = validate_judgment(json.dumps(response(statuses)), {0, 2, 4})
    expected = "REJECT" if "fail" in statuses else "REVIEW" if "uncertain" in statuses else "PASS"
    assert aggregate(checks) == expected


@pytest.mark.parametrize("evidence", [[99], [-1], [True], ["0"], [0, 0], []])
def test_invalid_evidence(evidence):
    with pytest.raises(JudgmentError):
        validate_judgment(json.dumps(response(evidence=evidence)), {0, 2, 4})


def test_uncertain_may_have_no_evidence():
    checks = validate_judgment(json.dumps(response(("uncertain",) * 3, [])), {0})
    assert aggregate(checks) == "REVIEW"


@pytest.mark.parametrize("sampled_ids", [{0}, {0, 2, 4}])
def test_scene_consistency_pass_requires_temporal_evidence(sampled_ids):
    data = response(("uncertain", "pass", "uncertain"), evidence=[0])
    with pytest.raises(JudgmentError, match="at least two sampled frames"):
        validate_judgment(json.dumps(data), sampled_ids)


def test_visible_scene_violation_can_reject_even_when_action_succeeds():
    data = response()
    data["checks"]["scene_consistency"] = {
        "status": "fail",
        "reason": "Before contact, the table corner exits the view as the untouched box grows.",
        "evidence_frames": [0, 2],
    }
    checks = validate_judgment(json.dumps(data), {0, 2, 4})
    assert checks.task_compliance.status == "pass"
    assert aggregate(checks) == "REJECT"


def test_single_frame_scene_failure_is_still_valid():
    # Identity/background violations can be established from one frame and the
    # original task, unlike a claim of temporal consistency.
    checks = validate_judgment(json.dumps(response(("uncertain", "fail", "uncertain"), [0])), {0})
    assert aggregate(checks) == "REJECT"


@pytest.mark.parametrize("modification", ["decision", "missing_check", "confidence", "status"])
def test_rejects_unexpected_or_missing_fields(modification):
    data = response()
    if modification == "decision":
        data["decision"] = "PASS"
    elif modification == "missing_check":
        del data["checks"]["scene_consistency"]
    elif modification == "confidence":
        data["checks"]["task_compliance"]["confidence"] = 0.93
    else:
        data["checks"]["task_compliance"]["status"] = "probably"
    with pytest.raises(JudgmentError):
        validate_judgment(json.dumps(data), {0, 2})


@pytest.mark.parametrize("raw", ["not JSON", "```json\n{}\n```", '{"checks":{},"checks":{}}'])
def test_invalid_model_json(raw):
    with pytest.raises(JudgmentError):
        validate_judgment(raw, {0})
