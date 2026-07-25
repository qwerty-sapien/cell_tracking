"""Literature and local-resource gates for lightweight model candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "celltrack-alpha.model-candidate.v1"
POLICY_SCHEMA = "celltrack-alpha.compute-governance.v1"
PRIMARY_EVIDENCE_KIND = "primary_peer_reviewed"
FIT_SCOPE = "fit_only"
CGL_D14_CANDIDATE_ID = "cgl-temporalunet-tail-rank-head-v1"
CGL_D14_SWAP_AMENDMENT_DATE = "2026-07-26"
CGL_D14_MAXIMUM_SWAP_GROWTH_GIB = 1.5
CGL_D14_RUNTIME_RSS_CEILING_GIB = 12.0


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
    swap_growth_allowed = local.get("swap_growth_allowed")
    if swap_growth_allowed is False:
        maximum_swap_growth_gib = 0.0
    elif swap_growth_allowed is True:
        if candidate_id != CGL_D14_CANDIDATE_ID:
            raise CandidateGateError(
                "local_fast_fail.swap_growth_allowed may be true only for the "
                "CGL D-14 candidate"
            )
        maximum_swap_growth_gib = _positive_number(
            local.get("maximum_swap_growth_gib"),
            "local_fast_fail.maximum_swap_growth_gib",
        )
        if maximum_swap_growth_gib != CGL_D14_MAXIMUM_SWAP_GROWTH_GIB:
            raise CandidateGateError(
                "CGL D-14 maximum swap growth must be exactly 1.5 GiB"
            )
        amendment = _mapping(
            candidate.get("runtime_guard_amendment"), "runtime_guard_amendment"
        )
        if amendment.get("authorized_by") != "goal_owner":
            raise CandidateGateError(
                "runtime_guard_amendment.authorized_by must be 'goal_owner'"
            )
        if amendment.get("authorized_on") != CGL_D14_SWAP_AMENDMENT_DATE:
            raise CandidateGateError(
                "runtime_guard_amendment.authorized_on must be '2026-07-26'"
            )
        if amendment.get("scope") != "CGL D-14 fast-fail only":
            raise CandidateGateError(
                "runtime_guard_amendment.scope must be 'CGL D-14 fast-fail only'"
            )
        if amendment.get("prior_maximum_swap_growth_gib") != 1.5:
            raise CandidateGateError(
                "runtime_guard_amendment.prior_maximum_swap_growth_gib must be 1.5"
            )
        if (
            amendment.get("maximum_swap_growth_gib")
            != CGL_D14_MAXIMUM_SWAP_GROWTH_GIB
        ):
            raise CandidateGateError(
                "runtime_guard_amendment.maximum_swap_growth_gib must be 1.5"
            )
        if (
            local.get("runtime_rss_ceiling_gib")
            != CGL_D14_RUNTIME_RSS_CEILING_GIB
        ):
            raise CandidateGateError(
                "local_fast_fail.runtime_rss_ceiling_gib must be 12"
            )
        if amendment.get("rss_ceiling_gib") != CGL_D14_RUNTIME_RSS_CEILING_GIB:
            raise CandidateGateError(
                "runtime_guard_amendment.rss_ceiling_gib must be 12"
            )
        if amendment.get("max_wall_minutes") != 360:
            raise CandidateGateError(
                "runtime_guard_amendment.max_wall_minutes must remain 360"
            )
        if amendment.get("outputs_deterministic_wrt_swap") is not True:
            raise CandidateGateError(
                "runtime_guard_amendment.outputs_deterministic_wrt_swap must be true"
            )
    else:
        raise CandidateGateError(
            "local_fast_fail.swap_growth_allowed must be boolean"
        )

    projected_memory = _positive_number(
        local.get("projected_peak_memory_gib"),
        "local_fast_fail.projected_peak_memory_gib",
    )
    memory_cap = _positive_number(
        local_policy.get("maximum_training_memory_gib"),
        "policy.exploration_strategy.local_parallel_lane.maximum_training_memory_gib",
    )
    runtime_memory_cap = _positive_number(
        local.get("runtime_rss_ceiling_gib", memory_cap),
        "local_fast_fail.runtime_rss_ceiling_gib",
    )
    if runtime_memory_cap > memory_cap:
        raise CandidateGateError(
            f"runtime RSS ceiling {runtime_memory_cap:g} GiB exceeds "
            f"the policy maximum {memory_cap:g} GiB"
        )
    if projected_memory > runtime_memory_cap:
        raise CandidateGateError(
            f"projected peak memory {projected_memory:g} GiB exceeds "
            f"the runtime RSS ceiling {runtime_memory_cap:g} GiB"
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
        "maximum_training_memory_gib": runtime_memory_cap,
        "policy_maximum_training_memory_gib": memory_cap,
        "background_reserve_gib": reserve,
        "minimum_system_memory_free_percent": minimum_free_percent,
        "max_wall_minutes": wall_minutes,
        "training_scope": FIT_SCOPE,
        "holdout_access": False,
        "swap_growth_allowed": swap_growth_allowed,
        "maximum_swap_growth_gib": maximum_swap_growth_gib,
        "decision": "PASS",
    }
