"""Literature and local-resource gates for lightweight model candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "celltrack-alpha.model-candidate.v1"
POLICY_SCHEMA = "celltrack-alpha.compute-governance.v1"
PRIMARY_EVIDENCE_KIND = "primary_peer_reviewed"
FIT_SCOPE = "fit_only"


class CandidateGateError(ValueError):
    """Raised when a candidate does not satisfy the locked pre-training gates."""


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateGateError(f"{field} must be an object")
    return value


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateGateError(f"{field} must be non-empty text")
    return value.strip()


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateGateError(f"{field} must be numeric")
    number = float(value)
    if number <= 0:
        raise CandidateGateError(f"{field} must be positive")
    return number


def _nonempty_text_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CandidateGateError(f"{field} must be a non-empty list")
    return [_nonempty_text(item, f"{field}[{index}]") for index, item in enumerate(value)]


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateGateError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CandidateGateError(f"{path} must contain a JSON object")
    return payload


def validate_candidate(
    candidate: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a candidate manifest and return its normalized gate summary."""

    if candidate.get("schema") != SCHEMA:
        raise CandidateGateError(f"candidate schema must be {SCHEMA!r}")
    if policy.get("schema") != POLICY_SCHEMA or policy.get("status") != "locked":
        raise CandidateGateError("compute policy must be the locked v1 authority")

    candidate_id = _nonempty_text(candidate.get("candidate_id"), "candidate_id")
    model_family = _nonempty_text(candidate.get("model_family"), "model_family")

    evidence = candidate.get("literature_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise CandidateGateError("literature_evidence must be a non-empty list")
    primary_count = 0
    for index, raw_source in enumerate(evidence):
        source = _mapping(raw_source, f"literature_evidence[{index}]")
        kind = _nonempty_text(source.get("kind"), f"literature_evidence[{index}].kind")
        if kind == PRIMARY_EVIDENCE_KIND:
            primary_count += 1
        for field in ("citation", "url", "reported_result", "task_relevance"):
            _nonempty_text(source.get(field), f"literature_evidence[{index}].{field}")
    if primary_count < 1:
        raise CandidateGateError(
            f"literature_evidence needs at least one {PRIMARY_EVIDENCE_KIND!r} source"
        )

    hypothesis = _mapping(candidate.get("hypothesis"), "hypothesis")
    for field in ("observed_failure_mode", "proposed_mechanism", "falsification_test"):
        _nonempty_text(hypothesis.get(field), f"hypothesis.{field}")
    expected_uplift = _positive_number(
        hypothesis.get("expected_score_improvement_percent"),
        "hypothesis.expected_score_improvement_percent",
    )
    promotion = _mapping(policy.get("promotion"), "policy.promotion")
    minimum_uplift = _positive_number(
        promotion.get("hard_baseline_score_improvement_percent"),
        "policy.promotion.hard_baseline_score_improvement_percent",
    )
    if expected_uplift < minimum_uplift:
        raise CandidateGateError(
            f"expected score improvement {expected_uplift:g}% is below "
            f"the locked {minimum_uplift:g}% gate"
        )

    local_policy = _mapping(
        _mapping(policy.get("exploration_strategy"), "policy.exploration_strategy").get(
            "local_parallel_lane"
        ),
        "policy.exploration_strategy.local_parallel_lane",
    )
    local = _mapping(candidate.get("local_fast_fail"), "local_fast_fail")
    if local.get("m5_compatible") is not True:
        raise CandidateGateError("local_fast_fail.m5_compatible must be true")
    if local.get("training_scope") != FIT_SCOPE:
        raise CandidateGateError("local_fast_fail.training_scope must be 'fit_only'")
    if local.get("holdout_access") is not False:
        raise CandidateGateError("local_fast_fail.holdout_access must be false")
    if local.get("swap_growth_allowed") is not False:
        raise CandidateGateError("local_fast_fail.swap_growth_allowed must be false")

    projected_memory = _positive_number(
        local.get("projected_peak_memory_gib"),
        "local_fast_fail.projected_peak_memory_gib",
    )
    memory_cap = _positive_number(
        local_policy.get("maximum_training_memory_gib"),
        "policy.exploration_strategy.local_parallel_lane.maximum_training_memory_gib",
    )
    if projected_memory > memory_cap:
        raise CandidateGateError(
            f"projected peak memory {projected_memory:g} GiB exceeds {memory_cap:g} GiB"
        )

    reserve = _positive_number(
        local.get("background_reserve_gib"),
        "local_fast_fail.background_reserve_gib",
    )
    minimum_reserve = _positive_number(
        local_policy.get("minimum_background_reserve_gib"),
        "policy.exploration_strategy.local_parallel_lane.minimum_background_reserve_gib",
    )
    if reserve < minimum_reserve:
        raise CandidateGateError(
            f"background reserve {reserve:g} GiB is below {minimum_reserve:g} GiB"
        )
    minimum_free_percent = _positive_number(
        local_policy.get("minimum_system_memory_free_percent"),
        "policy.exploration_strategy.local_parallel_lane.minimum_system_memory_free_percent",
    )
    if minimum_free_percent > 100:
        raise CandidateGateError("minimum system memory free percent cannot exceed 100")

    wall_minutes = _positive_number(
        local.get("max_wall_minutes"), "local_fast_fail.max_wall_minutes"
    )
    maximum_wall_minutes = 60.0 * _positive_number(
        local_policy.get("maximum_unattended_stage_hours"),
        "policy.exploration_strategy.local_parallel_lane.maximum_unattended_stage_hours",
    )
    if wall_minutes > maximum_wall_minutes:
        raise CandidateGateError(
            f"max wall time {wall_minutes:g} minutes exceeds "
            f"{maximum_wall_minutes:g} minutes"
        )

    _nonempty_text_list(local.get("early_stop_conditions"), "local_fast_fail.early_stop_conditions")
    _nonempty_text_list(local.get("success_criteria"), "local_fast_fail.success_criteria")
    if candidate.get("decision") != "PASS":
        raise CandidateGateError("candidate.decision must be 'PASS' before training")

    return {
        "candidate_id": candidate_id,
        "model_family": model_family,
        "literature_primary_source_count": primary_count,
        "expected_score_improvement_percent": expected_uplift,
        "projected_peak_memory_gib": projected_memory,
        "maximum_training_memory_gib": memory_cap,
        "background_reserve_gib": reserve,
        "minimum_system_memory_free_percent": minimum_free_percent,
        "max_wall_minutes": wall_minutes,
        "training_scope": FIT_SCOPE,
        "holdout_access": False,
        "swap_growth_allowed": False,
        "decision": "PASS",
    }
