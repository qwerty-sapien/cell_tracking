#!/usr/bin/env python3
"""Measure fit-only precision and reachability ceilings for association reranking.

This is a zero-GPU diagnostic. It loads the parked temporal-rank gate without
training it, calibrates its proposal scores on the disjoint fit-validation12
partition, and evaluates an optimistic ceiling on the remaining fit-train105
videos. Unlabeled deployment-background proposals participate in the deployed
per-video cap but are never relabeled as negatives.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha.site_calibration import (  # noqa: E402
    BinaryProbabilityCalibration,
    fit_platt_calibration,
    fit_temperature_calibration,
)
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from fit_selected_graph_temporal_background_gate import (  # noqa: E402
    TemporalGate,
    _video_data,
)
from fit_selected_graph_temporal_rank_gate import (  # noqa: E402
    _model_scores,
    capped_video_metrics,
)


SCHEMA = "celltrack-alpha.achievable-headroom.v1"
DEFAULT_GATE_REPORT = (
    ROOT / "reports/g02/selected_graph_temporal_rank_gate_v1/report.json"
)
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
DEFAULT_REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
DEFAULT_APPEARANCE_FIT = (
    ROOT / "reports/g02/appearance_direct_policy_fit_v1/report.json"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/screening/achievable_headroom_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/g02/screening/achievable_headroom_v1.md"
EXPECTED_GATE_SCHEMA = "celltrack-alpha.selected-graph-temporal-rank-gate.v1"
EXPECTED_PREPARE_SCHEMA = (
    "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
)
EXPECTED_REACHABILITY_SCHEMA = (
    "celltrack-alpha.extended-radius-reachability-audit.v1"
)
FORBIDDEN_REFERENCE_TOKENS = ("cv57", "holdout")
FAMILIES = ("44b6", "6bba")

# Aggregate constants are published by scripts/analyze_appearance_residual.py.
# No residual report or per-video held-out artifact is opened by this diagnostic.
BANKED_SCORE = 0.8815220699597315
BANKED_EDGE_TP = 34320
BANKED_EDGE_FP = 2459
BANKED_EDGE_FN = 1948
BANKED_NODE_ADJUSTMENT = 0.9947175175795606
TARGET_SCORE = 0.90
TARGET_IDEAL_SWAPS = 378


class AchievableHeadroomError(RuntimeError):
    """A required artifact, fit-only contract, or arithmetic invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_safe_fit_only_path(path: Path) -> None:
    """Reject forbidden artifact identities before checking or opening a path."""

    lowered = str(path).lower()
    token = next(
        (value for value in FORBIDDEN_REFERENCE_TOKENS if value in lowered),
        None,
    )
    if token is not None:
        raise AchievableHeadroomError(
            f"forbidden {token!r} artifact reference: {path}"
        )


def _guard_embedded_references(value: Any, *, source: Path) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        token = next(
            (
                candidate
                for candidate in FORBIDDEN_REFERENCE_TOKENS
                if candidate in lowered
            ),
            None,
        )
        if token is not None:
            raise AchievableHeadroomError(
                f"{source} embeds forbidden {token!r} reference: {value}"
            )
    elif isinstance(value, Mapping):
        for child in value.values():
            _guard_embedded_references(child, source=source)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        for child in value:
            _guard_embedded_references(child, source=source)


def _load_json(path: Path) -> Mapping[str, Any]:
    assert_safe_fit_only_path(path)
    if not path.is_file():
        raise AchievableHeadroomError(f"missing required input artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AchievableHeadroomError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AchievableHeadroomError(f"expected JSON object: {path}")
    _guard_embedded_references(value, source=path)
    return value


def _assert_scope(scope: Any, *, source: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(scope, Mapping):
        raise AchievableHeadroomError(f"missing fit-only scope: {source}")
    train_ids = tuple(str(value) for value in scope.get("train_video_ids", ()))
    validation_ids = tuple(
        str(value) for value in scope.get("validation_video_ids", ())
    )
    if (
        scope.get("fit_only") is not True
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
        or len(train_ids) != 105
        or len(validation_ids) != 12
        or set(train_ids) & set(validation_ids)
        or len(set((*train_ids, *validation_ids))) != 117
        or any(not identity.startswith("train:") for identity in (*train_ids, *validation_ids))
    ):
        raise AchievableHeadroomError(
            f"artifact is not the exact fit-only 105/12 scope: {source}"
        )
    return train_ids, validation_ids


def _resolve_artifact(value: str, *, owner: Path) -> Path:
    assert_safe_fit_only_path(Path(value))
    path = Path(value)
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise AchievableHeadroomError(
            f"artifact escapes repository root: {value}"
        ) from exc
    assert_safe_fit_only_path(resolved)
    if not resolved.is_file():
        raise AchievableHeadroomError(
            f"missing required input artifact: {resolved} (referenced by {owner})"
        )
    return resolved


def _family(identity: str) -> str:
    try:
        return identity.split(":", 1)[1].split("_", 1)[0]
    except IndexError as exc:
        raise AchievableHeadroomError(
            f"cannot derive family from identity: {identity}"
        ) from exc


def _probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise AchievableHeadroomError("raw proposal scores must be finite")
    clipped = np.clip(values, 1.0e-7, 1.0 - 1.0e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _apply_calibration(
    scored: Sequence[Mapping[str, Any]],
    calibrator: BinaryProbabilityCalibration,
) -> list[dict[str, Any]]:
    result = []
    for video in scored:
        result.append(
            {
                "identity": str(video["identity"]),
                "selected_edges": int(video["selected_edges"]),
                "labeled_confidence": calibrator.predict_probabilities(
                    _probability_to_logit(
                        np.asarray(video["labeled_confidence"])
                    )
                ),
                "labels": np.asarray(video["labels"], dtype=np.int8),
                "background_confidence": calibrator.predict_probabilities(
                    _probability_to_logit(
                        np.asarray(video["background_confidence"])
                    )
                ),
            }
        )
    return result


def aggregate_capped_metrics(
    scored: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, int | float]:
    """Aggregate the existing deployed-cap metric without relabeling PU rows."""

    correct = harmful = background = selected = total_budget = 0
    for video in scored:
        budget = max(
            16, int(math.ceil(0.005 * int(video["selected_edges"])))
        )
        metrics = capped_video_metrics(
            np.asarray(video["labeled_confidence"]),
            np.asarray(video["labels"]),
            np.asarray(video["background_confidence"]),
            threshold=float(threshold),
            budget=budget,
        )
        correct += int(metrics["correct_rescues"])
        harmful += int(metrics["harmful_swaps"])
        background += int(metrics["background_accepted"])
        selected += int(metrics["selected"])
        total_budget += budget
    accepted_labeled = correct + harmful
    total_positives = sum(
        int(np.sum(np.asarray(video["labels"]) == 1)) for video in scored
    )
    return {
        "correct_swaps": correct,
        "harmful_swaps": harmful,
        "accepted_labeled": accepted_labeled,
        "precision": correct / max(accepted_labeled, 1),
        "recall": correct / max(total_positives, 1),
        "net_correct_swaps": correct - harmful,
        "conservative_utility": correct - 2 * harmful,
        "background_accepted_unlabeled": background,
        "total_selected_with_unlabeled": selected,
        "total_video_budget": total_budget,
    }


def precision_recall_curve(
    scored: Sequence[Mapping[str, Any]],
    *,
    sampled_points: int = 257,
) -> list[dict[str, int | float]]:
    """Return an exact-small/sampled-large threshold curve."""

    positives = [
        np.asarray(video["labeled_confidence"])[
            np.asarray(video["labels"]) == 1
        ]
        for video in scored
        if np.any(np.asarray(video["labels"]) == 1)
    ]
    if not positives:
        return []
    positive_scores = np.concatenate(positives)
    unique = np.unique(positive_scores)
    thresholds = (
        unique
        if len(unique) <= sampled_points
        else np.unique(
            np.quantile(
                positive_scores,
                np.linspace(0.0, 1.0, sampled_points),
            )
        )
    )
    return [
        {
            "threshold": float(threshold),
            **aggregate_capped_metrics(scored, float(threshold)),
        }
        for threshold in sorted(thresholds, reverse=True)
    ]


def select_p95_operating_point(
    curve: Sequence[Mapping[str, Any]],
    *,
    minimum_precision: float = 0.95,
) -> dict[str, Any] | None:
    candidates = [
        dict(point)
        for point in curve
        if int(point["accepted_labeled"]) > 0
        and float(point["precision"]) >= minimum_precision
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda point: (
            int(point["conservative_utility"]),
            int(point["net_correct_swaps"]),
            int(point["correct_swaps"]),
            -int(point["harmful_swaps"]),
            float(point["threshold"]),
        ),
    )


def _per_video_thresholds(
    scored: Sequence[Mapping[str, Any]],
    *,
    minimum_precision: float,
    stable_iqr_limit: float,
    stable_range_limit: float,
) -> dict[str, Any]:
    rows = []
    excluded_without_both_classes = 0
    for video in sorted(scored, key=lambda item: str(item["identity"])):
        labels = np.asarray(video["labels"])
        if not np.any(labels == 1) or not np.any(labels == 0):
            excluded_without_both_classes += 1
            continue
        curve = precision_recall_curve((video,), sampled_points=100000)
        selected = select_p95_operating_point(
            curve, minimum_precision=minimum_precision
        )
        if selected is not None:
            rows.append(
                {
                    "identity": str(video["identity"]),
                    "family": _family(str(video["identity"])),
                    **selected,
                }
            )

    def summarize(selected_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        thresholds = np.asarray(
            [float(row["threshold"]) for row in selected_rows],
            dtype=np.float64,
        )
        if not len(thresholds):
            return {
                "video_count": 0,
                "q1": None,
                "median": None,
                "q3": None,
                "iqr": None,
                "minimum": None,
                "maximum": None,
                "max_minus_min": None,
                "stable": False,
            }
        q1, median, q3 = np.quantile(thresholds, (0.25, 0.5, 0.75))
        minimum = float(np.min(thresholds))
        maximum = float(np.max(thresholds))
        iqr = float(q3 - q1)
        spread = maximum - minimum
        return {
            "video_count": len(thresholds),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "iqr": iqr,
            "minimum": minimum,
            "maximum": maximum,
            "max_minus_min": spread,
            "stable": (
                len(thresholds) >= 10
                and iqr <= stable_iqr_limit
                and spread <= stable_range_limit
            ),
        }

    overall = summarize(rows)
    families = {
        family: summarize(
            [row for row in rows if row["family"] == family]
        )
        for family in FAMILIES
    }
    return {
        "definition": (
            "For videos with both labeled classes, choose the threshold maximizing "
            "conservative utility subject to labeled precision >=0.95 under the "
            "existing joint labeled/unlabeled cap."
        ),
        "stability_rule": {
            "minimum_informative_videos": 10,
            "maximum_iqr_probability": stable_iqr_limit,
            "maximum_range_probability": stable_range_limit,
            "justification": (
                "A reusable global operating point should keep the central half of "
                "video-specific thresholds within one 0.10 probability band and "
                "avoid more than 0.30 end-to-end drift."
            ),
        },
        "excluded_without_both_labeled_classes": excluded_without_both_classes,
        "overall": overall,
        "families": families,
        "all_video_operating_points": rows,
        "stable_for_decision": (
            bool(overall["stable"])
            and all(bool(families[family]["stable"]) for family in FAMILIES)
        ),
    }


def _curve_sections(
    scored: Sequence[Mapping[str, Any]],
    *,
    sampled_points: int,
    minimum_precision: float,
) -> dict[str, Any]:
    result = {}
    for name in ("overall", *FAMILIES):
        selected = (
            list(scored)
            if name == "overall"
            else [
                video
                for video in scored
                if _family(str(video["identity"])) == name
            ]
        )
        curve = precision_recall_curve(
            selected, sampled_points=sampled_points
        )
        result[name] = {
            "video_count": len(selected),
            "known_positive_proposals": sum(
                int(np.sum(np.asarray(video["labels"]) == 1))
                for video in selected
            ),
            "known_harmful_proposals": sum(
                int(np.sum(np.asarray(video["labels"]) == 0))
                for video in selected
            ),
            "operating_point_p95": select_p95_operating_point(
                curve, minimum_precision=minimum_precision
            ),
            "curve": curve,
        }
    return result


def _load_parked_scores(
    gate_report_path: Path,
    prepare_path: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[dict[str, Any]],
    tuple[str, ...],
    tuple[str, ...],
    Path,
]:
    report = _load_json(gate_report_path)
    prepare = _load_json(prepare_path)
    report_train, report_validation = _assert_scope(
        report.get("scope"), source=gate_report_path
    )
    prepare_train, prepare_validation = _assert_scope(
        prepare.get("scope"), source=prepare_path
    )
    if (
        report.get("schema") != EXPECTED_GATE_SCHEMA
        or report.get("status") != "PARK"
        or prepare.get("schema") != EXPECTED_PREPARE_SCHEMA
        or (report_train, report_validation)
        != (prepare_train, prepare_validation)
    ):
        raise AchievableHeadroomError(
            "parked gate and prepared fit-only scope do not reconcile"
        )
    provenance = report.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("prepare_sha256") != _sha256(prepare_path)
    ):
        raise AchievableHeadroomError(
            "parked gate does not hash-pin the prepared fit-only artifact"
        )
    artifact = report.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AchievableHeadroomError("parked gate artifact section is missing")
    gate_path = _resolve_artifact(
        str(artifact.get("path", "")), owner=gate_report_path
    )
    if artifact.get("sha256") != _sha256(gate_path):
        raise AchievableHeadroomError("parked temporal gate SHA-256 drifted")

    rows_by_id = {
        str(row.get("identity")): row
        for row in prepare.get("videos", ())
        if isinstance(row, Mapping)
    }
    if set(rows_by_id) != set((*prepare_train, *prepare_validation)):
        raise AchievableHeadroomError(
            "prepared video rows do not cover exact fit117"
        )
    for identity in (*prepare_train, *prepare_validation):
        row = rows_by_id[identity]
        row_path = _resolve_artifact(str(row.get("path", "")), owner=prepare_path)
        if row.get("sha256") != _sha256(row_path):
            raise AchievableHeadroomError(
                f"prepared video SHA-256 drifted: {identity}"
            )

    value = torch.load(gate_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != EXPECTED_GATE_SCHEMA
        or int(value.get("feature_count", 0)) != 48
        or int(value.get("hidden", -1)) not in (0, 16)
    ):
        raise AchievableHeadroomError("parked temporal gate identity is invalid")
    mean = np.asarray(value.get("mean"), dtype=np.float32)
    scale = np.asarray(value.get("scale"), dtype=np.float32)
    if (
        mean.shape != (48,)
        or scale.shape != (48,)
        or np.any(scale <= 0.0)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
    ):
        raise AchievableHeadroomError("parked gate normalization is invalid")
    model = TemporalGate(48, int(value["hidden"]))
    model.load_state_dict(value["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.device.type != "cpu":
            raise AchievableHeadroomError("temporal gate is not on CPU")

    videos = [_video_data(rows_by_id[identity]) for identity in (*prepare_train, *prepare_validation)]
    with torch.no_grad():
        scored = _model_scores(model, mean, scale, videos)
    return (
        report,
        prepare,
        scored,
        prepare_train,
        prepare_validation,
        gate_path,
    )


def _reachability(
    path: Path,
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    post_swap_events: Mapping[str, Any],
) -> dict[str, Any]:
    report = _load_json(path)
    reach_train, reach_validation = _assert_scope(
        report.get("scope"), source=path
    )
    if (
        report.get("schema") != EXPECTED_REACHABILITY_SCHEMA
        or (reach_train, reach_validation)
        != (tuple(train_ids), tuple(validation_ids))
    ):
        raise AchievableHeadroomError(
            "reachability artifact does not share exact fit117 scope"
        )
    all_fit = report.get("all_fit")
    validation = report.get("validation")
    all_counts = (
        all_fit.get("counts") if isinstance(all_fit, Mapping) else None
    )
    validation_counts = (
        validation.get("counts")
        if isinstance(validation, Mapping)
        else None
    )
    if not isinstance(all_counts, Mapping) or not isinstance(
        validation_counts, Mapping
    ):
        raise AchievableHeadroomError("reachability fit-only counts are missing")
    all_excluded = int(all_counts.get("default_excluded_edges", -1))
    all_reachable = int(all_counts.get("default_reachable_edges", -1))
    all_matched = int(all_counts.get("matched_adjacent_edges", -1))
    validation_excluded = int(
        validation_counts.get("default_excluded_edges", -1)
    )
    validation_reachable = int(
        validation_counts.get("default_reachable_edges", -1)
    )
    validation_matched = int(
        validation_counts.get("matched_adjacent_edges", -1)
    )
    in_pool_residual = int(
        post_swap_events["remaining_conflicting_correct_swaps"]
    )
    if (
        all_excluded != 719
        or all_reachable + all_excluded != all_matched
        or validation_reachable + validation_excluded
        != validation_matched
        or validation_excluded != 70
        or in_pool_residual <= 0
    ):
        raise AchievableHeadroomError(
            "fit-only reachability or in-pool correction accounting drifted"
        )
    denominator = in_pool_residual + validation_excluded
    in_pool_fraction = in_pool_residual / denominator
    projected_in_pool = int(
        math.floor(TARGET_IDEAL_SWAPS * in_pool_fraction + 0.5)
    )
    projected_excluded = TARGET_IDEAL_SWAPS - projected_in_pool
    oracle_score = adjusted_score_after_perfect_swaps(projected_in_pool)
    return {
        "fit_validation_post_swap": dict(post_swap_events),
        "fit_validation_in_pool_residual_correction_events": in_pool_residual,
        "fit_validation_excluded_correct_edges": validation_excluded,
        "fit_validation_correction_support_total": denominator,
        "recall_bound_in_pool_fraction": in_pool_fraction,
        "target_ideal_swaps_from_published_aggregate": TARGET_IDEAL_SWAPS,
        "target_swap_allocation_method": (
            "nearest-integer proportional allocation using fit-only current-pool "
            "correct proposal events versus fit-only excluded correct edges"
        ),
        "target_swaps_in_current_pool": projected_in_pool,
        "target_swaps_excluded_from_current_pool": projected_excluded,
        "oracle_reachable_score_current_pool": oracle_score,
        "oracle_margin_over_banked": oracle_score - BANKED_SCORE,
        "raw_reachability_context": {
            "all_fit": {
                "matched_adjacent_correct_edges": all_matched,
                "default_reachable_correct_edges": all_reachable,
                "default_excluded_correct_edges": all_excluded,
            },
            "fit_validation12": {
                "matched_adjacent_correct_edges": validation_matched,
                "default_reachable_correct_edges": validation_reachable,
                "default_excluded_correct_edges": validation_excluded,
            },
        },
    }


def _post_swap_fit_validation_events(
    appearance_report_path: Path,
    prepare: Mapping[str, Any],
    *,
    validation_ids: Sequence[str],
) -> dict[str, Any]:
    """Reconcile fit-validation positive challengers with frozen banked graphs."""

    appearance = _load_json(appearance_report_path)
    selection = appearance.get("selection")
    control = appearance.get("control")
    policies = appearance.get("policies")
    if (
        appearance.get("schema")
        != "celltrack-alpha.appearance-direct-policy-fit-screen.v1"
        or not isinstance(selection, Mapping)
        or tuple(str(value) for value in selection.get("video_ids", ()))
        != tuple(validation_ids)
        or not isinstance(control, Mapping)
        or not isinstance(policies, Mapping)
        or abs(
            float(policies.get("control", {}).get("appearance_threshold"))
            - (-0.04768145084381103)
        )
        > 1.0e-12
    ):
        raise AchievableHeadroomError(
            "appearance fit report is not the frozen banked fit-validation12 control"
        )
    input_provenance = control.get("input_provenance")
    files = (
        input_provenance.get("files")
        if isinstance(input_provenance, Mapping)
        else None
    )
    if not isinstance(files, Sequence) or len(files) != 12:
        raise AchievableHeadroomError(
            "banked fit-validation graph provenance is incomplete"
        )
    fragment_by_id = {
        str(row.get("identity")): row
        for row in files
        if isinstance(row, Mapping)
    }
    prepare_by_id = {
        str(row.get("identity")): row
        for row in prepare.get("videos", ())
        if isinstance(row, Mapping)
    }
    if set(fragment_by_id) != set(validation_ids):
        raise AchievableHeadroomError(
            "banked fit-validation fragments do not match calibration split"
        )

    totals = {
        "known_correct_proposals": 0,
        "already_selected_correct_edges": 0,
        "remaining_conflicting_correct_swaps": 0,
        "correct_proposals_without_selected_conflict": 0,
    }
    rows = []
    for identity in validation_ids:
        fragment_row = fragment_by_id[identity]
        fragment = _resolve_artifact(
            str(fragment_row.get("path", "")), owner=appearance_report_path
        )
        if fragment_row.get("sha256") != _sha256(fragment):
            raise AchievableHeadroomError(
                f"banked fit-validation fragment SHA-256 drifted: {identity}"
            )
        graphs = read_submission(fragment)
        if len(graphs) != 1:
            raise AchievableHeadroomError(
                f"banked fit fragment has unexpected dataset count: {identity}"
            )
        selected_edges = {
            (int(edge.source_id), int(edge.target_id))
            for graph in graphs.values()
            for edge in graph.edges
        }

        prepared_path = _resolve_artifact(
            str(prepare_by_id[identity].get("path", "")), owner=appearance_report_path
        )
        prepared = torch.load(
            prepared_path, map_location="cpu", weights_only=False
        )
        pairs = np.asarray(prepared["labeled_challenger_pairs"], dtype=np.int64)
        labels = np.asarray(prepared["labeled_labels"], dtype=np.int8)
        grouped: dict[tuple[int, int], set[int]] = {}
        for pair, label in zip(pairs, labels, strict=True):
            grouped.setdefault(
                (int(pair[0]), int(pair[1])), set()
            ).add(int(label))
        if any(len(values) != 1 for values in grouped.values()):
            raise AchievableHeadroomError(
                f"prepared proposal has mixed labels: {identity}"
            )
        positive = [
            pair for pair, values in grouped.items() if values == {1}
        ]
        already = sum(pair in selected_edges for pair in positive)
        remaining = sum(
            pair not in selected_edges
            and any(
                edge[0] == pair[0] or edge[1] == pair[1]
                for edge in selected_edges
            )
            for pair in positive
        )
        no_conflict = len(positive) - already - remaining
        video = {
            "identity": identity,
            "known_correct_proposals": len(positive),
            "already_selected_correct_edges": already,
            "remaining_conflicting_correct_swaps": remaining,
            "correct_proposals_without_selected_conflict": no_conflict,
        }
        rows.append(video)
        for key in totals:
            totals[key] += int(video[key])
    if sum(totals.values()) - totals["known_correct_proposals"] != totals[
        "known_correct_proposals"
    ]:
        raise AchievableHeadroomError(
            "post-swap fit-validation correction categories do not partition"
        )
    return {**totals, "videos": rows}


def adjusted_score_after_perfect_swaps(swaps: int) -> float:
    if not 0 <= swaps <= min(BANKED_EDGE_FP, BANKED_EDGE_FN):
        raise AchievableHeadroomError("perfect-swap count is outside banked errors")
    return BANKED_NODE_ADJUSTMENT * (
        (BANKED_EDGE_TP + swaps)
        / (
            BANKED_EDGE_TP
            + swaps
            + BANKED_EDGE_FP
            - swaps
            + BANKED_EDGE_FN
            - swaps
        )
    )


def _calibration_record(
    calibrator: BinaryProbabilityCalibration,
    *,
    example_count: int,
    positives: int,
) -> dict[str, Any]:
    return {
        "name": calibrator.name,
        "intercept": calibrator.intercept,
        "slope": calibrator.slope,
        "temperature": (
            1.0 / calibrator.slope
            if calibrator.name == "temperature"
            else None
        ),
        "monotone_increasing": calibrator.slope > 0.0,
        "calibration_example_count": example_count,
        "calibration_positive_count": positives,
        "calibration_negative_count": example_count - positives,
        "metrics_before": calibrator.calibration_metrics_before,
        "metrics_after": calibrator.calibration_metrics_after,
    }


def _immutable_write(path: Path, content: bytes) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise AchievableHeadroomError(
                f"refusing to overwrite different report: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    recall = report["recall_side_ceiling"]
    operating = report["precision_side_ceiling"]["temperature"]["curves"][
        "overall"
    ]["operating_point_p95"]
    stability = report["precision_side_ceiling"]["per_video_threshold_stability"]
    return "\n".join(
        (
            "# Achievable headroom diagnostic",
            "",
            f"- In-pool net-correct swaps at labeled precision >=0.95: **{decision['in_pool_achievable_swaps_at_p95']}**.",
            f"- Per-video p95 threshold IQR/range: **{decision['per_video_threshold_iqr']:.6f} / {stability['overall']['max_minus_min']:.6f}**; stable = **{decision['per_video_thresholds_stable']}**.",
            f"- Fit-only recall allocation of the 378 target swaps: **{recall['target_swaps_in_current_pool']} in pool / {recall['target_swaps_excluded_from_current_pool']} excluded**.",
            f"- Current-pool perfect-ranker oracle score: **{decision['oracle_reachable_score_current_pool']:.9f}** versus banked **{BANKED_SCORE:.9f}**.",
            f"- Recommendation: **{decision['recommendation']}** — {decision['rationale']}",
            "",
            "The p95 precision is computed only over known labeled proposals. Unlabeled",
            "background participates in the deployed cap but is not treated as negative.",
            f"At the selected point it consumes {operating['background_accepted_unlabeled']} cap slots.",
            "",
        )
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    (
        gate_report,
        prepare,
        raw_scored,
        train_ids,
        calibration_ids,
        gate_path,
    ) = _load_parked_scores(args.gate_report, args.prepare)
    train_id_set = set(train_ids)
    calibration_id_set = set(calibration_ids)
    evaluation_raw = [
        video for video in raw_scored if video["identity"] in train_id_set
    ]
    calibration_raw = [
        video for video in raw_scored if video["identity"] in calibration_id_set
    ]
    if (
        len(evaluation_raw) != 105
        or len(calibration_raw) != 12
        or {video["identity"] for video in evaluation_raw}
        & {video["identity"] for video in calibration_raw}
    ):
        raise AchievableHeadroomError("calibration and ceiling partitions overlap")

    calibration_probabilities = np.concatenate(
        [
            np.asarray(video["labeled_confidence"], dtype=np.float64)
            for video in calibration_raw
        ]
    )
    calibration_labels = np.concatenate(
        [np.asarray(video["labels"], dtype=np.int8) for video in calibration_raw]
    )
    calibration_logits = _probability_to_logit(calibration_probabilities)
    calibrators = {
        "temperature": fit_temperature_calibration(
            calibration_logits, calibration_labels
        ),
        "platt": fit_platt_calibration(
            calibration_logits, calibration_labels
        ),
    }
    calibrated_evaluation = {
        name: _apply_calibration(evaluation_raw, calibrator)
        for name, calibrator in calibrators.items()
    }
    curves = {
        name: _curve_sections(
            scored,
            sampled_points=args.curve_points,
            minimum_precision=args.minimum_precision,
        )
        for name, scored in calibrated_evaluation.items()
    }
    primary = calibrated_evaluation["temperature"]
    stability = _per_video_thresholds(
        primary,
        minimum_precision=args.minimum_precision,
        stable_iqr_limit=args.stable_iqr_limit,
        stable_range_limit=args.stable_range_limit,
    )
    primary_operating = curves["temperature"]["overall"][
        "operating_point_p95"
    ]
    if primary_operating is None:
        achievable = 0
    else:
        achievable = int(primary_operating["net_correct_swaps"])

    post_swap_events = _post_swap_fit_validation_events(
        args.appearance_fit,
        prepare,
        validation_ids=calibration_ids,
    )
    recall = _reachability(
        args.reachability,
        train_ids=train_ids,
        validation_ids=calibration_ids,
        post_swap_events=post_swap_events,
    )
    oracle_margin_worthwhile = (
        float(recall["oracle_margin_over_banked"])
        >= args.minimum_worthwhile_margin
    )
    build = (
        achievable >= args.minimum_achievable_swaps
        and bool(stability["stable_for_decision"])
        and oracle_margin_worthwhile
    )
    recommendation = (
        "BUILD_RANKER" if build else "REDIRECT_DETERMINISTIC_AND_RECALL"
    )
    failed_reasons = []
    if achievable < args.minimum_achievable_swaps:
        failed_reasons.append(
            f"p95 net-correct swaps {achievable} < {args.minimum_achievable_swaps}"
        )
    if not stability["stable_for_decision"]:
        failed_reasons.append(
            "per-video thresholds exceed the preregistered IQR/range stability limits"
        )
    if not oracle_margin_worthwhile:
        failed_reasons.append(
            "current-pool oracle margin is below the local-run minimum"
        )
    rationale = (
        "All build gates pass."
        if build
        else "; ".join(failed_reasons)
        + ". Preserve the banked swap and prioritize deterministic calibration plus recall expansion."
    )

    report = {
        "schema": SCHEMA,
        "status": "completed_fit_only_zero_gpu_diagnostic",
        "seed": args.seed,
        "scope": {
            "fit_only": True,
            "train_video_count": len(train_ids),
            "held_fit_calibration_video_count": len(calibration_ids),
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
            "evaluation_note": (
                "Precision-side ceiling is optimistic resubstitution on fit-train105; "
                "calibration is fit only on disjoint fit-validation12."
            ),
        },
        "pu_contract": {
            "known_labels": "sparse correct/harmful conflict proposals only",
            "unlabeled_background_treated_as_negative": False,
            "unlabeled_background_role": (
                "competes for slots under the existing per-video deployment cap"
            ),
            "precision_denominator": "known labeled proposals only",
        },
        "current_pool_contract": {
            "radius_um": 10.0,
            "neighbor_cap": 8,
            "division_enabled": False,
            "swap_contract": "node-disjoint under existing per-video cap",
        },
        "calibration": {
            "partition": "fit-validation12",
            "target": "proposal-level parked temporal-rank confidence",
            "temperature_primary": True,
            "platt_alternative": True,
            "models": {
                name: _calibration_record(
                    calibrator,
                    example_count=len(calibration_labels),
                    positives=int(calibration_labels.sum()),
                )
                for name, calibrator in calibrators.items()
            },
        },
        "precision_side_ceiling": {
            name: {
                "calibration": _calibration_record(
                    calibrators[name],
                    example_count=len(calibration_labels),
                    positives=int(calibration_labels.sum()),
                ),
                "curves": curves[name],
            }
            for name in ("temperature", "platt")
        },
        "recall_side_ceiling": recall,
        "precision_side_ceiling_addendum": {
            "per_video_threshold_stability": stability,
            "curve_grid": (
                f"exact if <= {args.curve_points} unique positive scores, otherwise "
                f"{args.curve_points} positive-score quantiles"
            ),
        },
        "decision": {
            "in_pool_achievable_swaps_at_p95": achievable,
            "per_video_threshold_iqr": stability["overall"]["iqr"],
            "per_video_thresholds_stable": stability["stable_for_decision"],
            "recall_bound_in_pool_fraction": recall[
                "recall_bound_in_pool_fraction"
            ],
            "oracle_reachable_score_current_pool": recall[
                "oracle_reachable_score_current_pool"
            ],
            "recommendation": recommendation,
            "rationale": rationale,
            "build_rule": {
                "minimum_in_pool_achievable_swaps_at_p95": (
                    args.minimum_achievable_swaps
                ),
                "stable_thresholds_required": True,
                "minimum_oracle_margin_over_banked": (
                    args.minimum_worthwhile_margin
                ),
            },
        },
        "published_aggregate_context": {
            "banked_score": BANKED_SCORE,
            "target_score": TARGET_SCORE,
            "target_ideal_wrong_to_right_swaps": TARGET_IDEAL_SWAPS,
            "source": (
                "aggregate constants published in scripts/analyze_appearance_residual.py; "
                "no residual or per-video held-out artifact opened"
            ),
        },
        "provenance": {
            "gate_report": str(args.gate_report),
            "gate_report_sha256": _sha256(args.gate_report),
            "gate_artifact": str(gate_path.relative_to(ROOT)),
            "gate_artifact_sha256": _sha256(gate_path),
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
            "reachability": str(args.reachability),
            "reachability_sha256": _sha256(args.reachability),
            "appearance_fit": str(args.appearance_fit),
            "appearance_fit_sha256": _sha256(args.appearance_fit),
            "referenced_artifacts_fit_only_guarded": True,
        },
        "execution": {
            "device": "cpu",
            "gpu_action": False,
            "model_training": False,
            "transformer_built": False,
            "kaggle_action": False,
            "submission_action": False,
            "deterministic": True,
        },
    }
    # Preserve the exact verdict key requested by the decision record while also
    # keeping the full stability detail in the precision section.
    report["precision_side_ceiling"]["per_video_threshold_stability"] = stability

    json_bytes = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown_bytes = _markdown(report).encode("utf-8")
    _immutable_write(args.output, json_bytes)
    _immutable_write(args.markdown, markdown_bytes)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    value.add_argument(
        "--appearance-fit", type=Path, default=DEFAULT_APPEARANCE_FIT
    )
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    value.add_argument("--seed", type=int, default=907)
    value.add_argument("--curve-points", type=int, default=257)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--minimum-achievable-swaps", type=int, default=150)
    value.add_argument("--stable-iqr-limit", type=float, default=0.10)
    value.add_argument("--stable-range-limit", type=float, default=0.30)
    value.add_argument("--minimum-worthwhile-margin", type=float, default=0.005)
    return value


if __name__ == "__main__":
    result = analyze(parser().parse_args())
    print(json.dumps(result["decision"], sort_keys=True))
