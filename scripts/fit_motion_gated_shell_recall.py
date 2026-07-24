#!/usr/bin/env python3
"""Fit-only deterministic motion gate for 10--12 um shell recall recovery.

The only fitted value is a constant-velocity residual tolerance selected on the
grouped fit-train105 partition.  No weights are learned.  The fixed
fit-validation12 partition is used once for the falsification decision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from analyze_achievable_headroom import (  # noqa: E402
    BANKED_SCORE,
    TARGET_IDEAL_SWAPS,
    AchievableHeadroomError,
    _guard_embedded_references,
    adjusted_score_after_perfect_swaps,
    assert_safe_fit_only_path,
)
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.appearance_swap import (  # noqa: E402
    PATCH_SHAPE,
    _greedy_nonconflicting,
    load_contract,
)
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    _cache_identity,
    _load_base_scorer,
)
from prepare_extended_radius_shell import (  # noqa: E402
    SCALE_ZYX,
    _baseline,
    shell_candidate_indices,
)


SCHEMA = "celltrack-alpha.motion-gated-shell-recall.v1"
DEFAULT_SHELL_PREPARE = (
    ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json"
)
DEFAULT_SHELL_APPEARANCE = (
    ROOT / "reports/g02/extended_radius_shell_appearance_v1/report.json"
)
DEFAULT_REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
DEFAULT_HEADROOM = ROOT / "reports/g02/screening/achievable_headroom_v1.json"
DEFAULT_CONTRACT = ROOT / "reports/g02/appearance_direct_swap_contract_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/screening/motion_gated_shell_recall_v1.json"
TOLERANCE_GRID_UM = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
FAMILIES = ("44b6", "6bba")


class MotionGatedShellError(RuntimeError):
    """A fit-only identity, accounting, or deterministic gate invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        assert_safe_fit_only_path(path)
    except AchievableHeadroomError as exc:
        raise MotionGatedShellError(str(exc)) from exc
    if not path.is_file():
        raise MotionGatedShellError(f"missing required input artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MotionGatedShellError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MotionGatedShellError(f"expected JSON object: {path}")
    try:
        _guard_embedded_references(value, source=path)
    except AchievableHeadroomError as exc:
        raise MotionGatedShellError(str(exc)) from exc
    return value


def _resolve_artifact(raw: str, *, owner: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    try:
        assert_safe_fit_only_path(path)
    except AchievableHeadroomError as exc:
        raise MotionGatedShellError(str(exc)) from exc
    if not path.is_file():
        raise MotionGatedShellError(
            f"missing required input artifact referenced by {owner}: {path}"
        )
    return path


def _family(identity: str) -> str:
    try:
        family = identity.split(":", 1)[1].split("_", 1)[0]
    except IndexError as exc:
        raise MotionGatedShellError(f"invalid fit identity: {identity}") from exc
    if family not in FAMILIES:
        raise MotionGatedShellError(f"unknown fit family: {identity}")
    return family


def motion_gate_mask(
    candidate_displacements_um: np.ndarray,
    predicted_displacements_um: np.ndarray,
    history_available: np.ndarray,
    *,
    tolerance_um: float,
) -> np.ndarray:
    """Keep candidates within a constant-velocity residual tolerance."""

    candidate = np.asarray(candidate_displacements_um, dtype=np.float64)
    predicted = np.asarray(predicted_displacements_um, dtype=np.float64)
    history = np.asarray(history_available, dtype=bool)
    if (
        candidate.ndim != 2
        or candidate.shape[1:] != (3,)
        or predicted.shape != candidate.shape
        or history.shape != (len(candidate),)
        or not np.isfinite(candidate).all()
        or not np.isfinite(predicted).all()
        or not math.isfinite(tolerance_um)
        or tolerance_um < 0.0
    ):
        raise MotionGatedShellError("motion-gate inputs are invalid")
    residual = np.linalg.norm(candidate - predicted, axis=1)
    return history & (residual <= tolerance_um + 1.0e-12)


def recall_accounting(
    labels: np.ndarray,
    retained: np.ndarray,
    *,
    expected_positive_total: int | None = None,
) -> dict[str, int | float]:
    """Count positive shell edges retained by a gate."""

    labels = np.asarray(labels, dtype=np.int8)
    retained = np.asarray(retained, dtype=bool)
    if (
        labels.shape != retained.shape
        or np.any(~np.isin(labels, (-1, 0, 1)))
    ):
        raise MotionGatedShellError("recall-accounting arrays are invalid")
    total = int(np.sum(labels == 1))
    if expected_positive_total is not None and total != expected_positive_total:
        raise MotionGatedShellError(
            f"positive shell total drifted: {total} != {expected_positive_total}"
        )
    kept = int(np.sum(retained & (labels == 1)))
    return {
        "positive_total": total,
        "positive_retained": kept,
        "positive_removed": total - kept,
        "recall": kept / max(total, 1),
    }


def precision_counts(
    labels: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, int | float]:
    """Return conservative labeled/PU acceptance and utility accounting."""

    labels = np.asarray(labels, dtype=np.int8)
    accepted = np.asarray(accepted, dtype=bool)
    if (
        labels.shape != accepted.shape
        or np.any(~np.isin(labels, (-1, 0, 1)))
    ):
        raise MotionGatedShellError("precision-accounting arrays are invalid")
    correct = int(np.sum(accepted & (labels == 1)))
    known_negative = int(np.sum(accepted & (labels == 0)))
    unlabeled = int(np.sum(accepted & (labels == -1)))
    total = correct + known_negative + unlabeled
    definite = correct + known_negative
    return {
        "accepted": total,
        "correct_edges": correct,
        "known_negative_edges": known_negative,
        "accepted_unlabeled": unlabeled,
        "fit_precision": correct / max(total, 1),
        "definite_precision": correct / max(definite, 1),
        "capped_utility_delta": correct - 2 * known_negative - 2 * unlabeled,
    }


def _validate_scope(
    shell_prepare: Mapping[str, Any],
    shell_appearance: Mapping[str, Any],
    reachability: Mapping[str, Any],
    headroom: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scope = shell_prepare.get("scope")
    if (
        shell_prepare.get("schema")
        != "celltrack-alpha.extended-radius-shell-prepare.v1"
        or shell_appearance.get("schema")
        != "celltrack-alpha.extended-radius-shell-appearance.v1"
        or reachability.get("schema")
        != "celltrack-alpha.extended-radius-reachability-audit.v1"
        or headroom.get("schema")
        != "celltrack-alpha.achievable-headroom.v1"
        or not isinstance(scope, Mapping)
        or len(scope.get("train_video_ids", ())) != 105
        or len(scope.get("validation_video_ids", ())) != 12
        or scope.get("fit_only") is not True
        or any(
            scope.get(name) is not False
            for name in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise MotionGatedShellError("inputs do not preserve exact fit-only 105/12 scope")
    for other in (shell_appearance, reachability):
        if other.get("scope") != scope:
            raise MotionGatedShellError("fit-only input scopes disagree")
    headroom_scope = headroom.get("scope")
    if (
        not isinstance(headroom_scope, Mapping)
        or headroom_scope.get("fit_only") is not True
        or headroom_scope.get("held_out_used") is not False
    ):
        raise MotionGatedShellError("headroom input is not fit-only")
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    if (
        len(set((*train_ids, *validation_ids))) != 117
        or any(not identity.startswith("train:") for identity in (*train_ids, *validation_ids))
    ):
        raise MotionGatedShellError("fit identities are incomplete or malformed")
    return train_ids, validation_ids


def _load_examples(
    shell_prepare_path: Path,
    shell_prepare: Mapping[str, Any],
    shell_appearance_path: Path,
    shell_appearance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prepare_by_id = {
        str(row.get("identity")): row
        for row in shell_prepare.get("videos", ())
        if isinstance(row, Mapping)
    }
    appearance_rows = shell_appearance.get("videos")
    if (
        not isinstance(appearance_rows, list)
        or len(prepare_by_id) != 117
        or len(appearance_rows) != 117
    ):
        raise MotionGatedShellError("shell video manifests are incomplete")
    feature_names = tuple(str(value) for value in shell_appearance.get("feature_names", ()))
    required_features = {
        "source_history_available",
        "source_acceleration_residual_um",
        "conflict_count",
        "pair_minus_maximum_conflict_ncc",
    }
    if not required_features.issubset(feature_names):
        raise MotionGatedShellError("shell appearance features are incomplete")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in appearance_rows:
        if not isinstance(row, Mapping):
            raise MotionGatedShellError("shell appearance video row is invalid")
        identity = str(row.get("identity"))
        if identity in seen or identity not in prepare_by_id:
            raise MotionGatedShellError(f"shell video identity drifted: {identity}")
        seen.add(identity)
        path = _resolve_artifact(str(row.get("path", "")), owner=shell_appearance_path)
        if str(row.get("sha256")) != _sha256(path):
            raise MotionGatedShellError(f"shell appearance SHA-256 drifted: {identity}")
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise MotionGatedShellError(f"cannot load shell example: {path}") from exc
        if (
            not isinstance(value, Mapping)
            or tuple(str(name) for name in value.get("feature_names", ())) != feature_names
        ):
            raise MotionGatedShellError(f"invalid shell example: {path}")
        features = np.asarray(value["features"], dtype=np.float32)
        labels = np.asarray(value["labels"], dtype=np.int8)
        pairs = np.asarray(value["pairs"], dtype=np.int64)
        conflicts = np.asarray(value["conflict_pairs"], dtype=np.int64)
        conflict_count = np.asarray(value["conflict_count"], dtype=np.int8)
        if (
            features.shape != (len(labels), len(feature_names))
            or pairs.shape != (len(labels), 2)
            or conflicts.shape != (len(labels), 2, 2)
            or conflict_count.shape != labels.shape
            or np.any((conflict_count < 0) | (conflict_count > 2))
        ):
            raise MotionGatedShellError(f"shell example arrays disagree: {path}")
        prepared = prepare_by_id[identity]
        if (
            str(prepared.get("partition")) != str(row.get("partition"))
            or int(prepared.get("sample", {}).get("rows", -1)) != len(labels)
        ):
            raise MotionGatedShellError(f"shell example manifest drifted: {identity}")
        result.append(
            {
                "identity": identity,
                "family": _family(identity),
                "partition": str(row["partition"]),
                "feature_names": feature_names,
                "features": features,
                "labels": labels,
                "pairs": pairs,
                "conflict_pairs": conflicts,
                "conflict_count": conflict_count,
                "selected_edges": int(prepared["graph"]["raw_selected_edges"]),
                "shell_pairs": int(prepared["counts"]["shell_pairs"]),
                "path": str(path),
                "sha256": str(row["sha256"]),
            }
        )
    return result


def _motion_mask_from_features(
    video: Mapping[str, Any], tolerance_um: float
) -> np.ndarray:
    names = tuple(video["feature_names"])
    features = np.asarray(video["features"], dtype=np.float32)
    history = features[:, names.index("source_history_available")] > 0.5
    residual = features[:, names.index("source_acceleration_residual_um")]
    if not np.isfinite(residual).all():
        raise MotionGatedShellError("prepared motion residual contains nonfinite values")
    return history & (residual <= tolerance_um + 1.0e-12)


def _accepted_video(
    video: Mapping[str, Any],
    *,
    tolerance_um: float,
    appearance_threshold: float,
    cap_fraction: float,
    minimum_cap: int,
) -> dict[str, Any]:
    features = np.asarray(video["features"], dtype=np.float32)
    labels = np.asarray(video["labels"], dtype=np.int8)
    pairs = np.asarray(video["pairs"], dtype=np.int64)
    conflict_pairs = np.asarray(video["conflict_pairs"], dtype=np.int64)
    conflict_count = np.asarray(video["conflict_count"], dtype=np.int8)
    names = tuple(video["feature_names"])
    margin = features[:, names.index("pair_minus_maximum_conflict_ncc")]
    motion = _motion_mask_from_features(video, tolerance_um)
    eligible = motion & (conflict_count > 0) & (margin >= appearance_threshold)
    proposals = []
    for index in np.flatnonzero(eligible):
        count = int(conflict_count[index])
        confidence = float(margin[index])
        proposals.append(
            {
                "challenger": pairs[index].tolist(),
                "conflicts": conflict_pairs[index, :count].tolist(),
                "confidence": confidence,
                "appearance_margins": [confidence],
                "geometry_confidence": 1.0,
                "row_probabilities": [1.0],
            }
        )
    nonconflicting = _greedy_nonconflicting(proposals)
    budget = max(minimum_cap, int(math.ceil(cap_fraction * int(video["selected_edges"]))))
    accepted_rows = nonconflicting[:budget]
    accepted_pairs = {
        tuple(int(value) for value in row["challenger"]) for row in accepted_rows
    }
    accepted = np.asarray(
        [tuple(int(value) for value in pair) in accepted_pairs for pair in pairs],
        dtype=bool,
    )
    if int(np.sum(accepted)) != len(accepted_pairs):
        raise MotionGatedShellError("accepted shell pair identity is not unique")
    metrics = precision_counts(labels, accepted)
    return {
        "identity": str(video["identity"]),
        "family": str(video["family"]),
        "partition": str(video["partition"]),
        "motion_gated_sample_rows": int(np.sum(motion)),
        "appearance_threshold_rows": len(proposals),
        "accepted_before_cap": len(nonconflicting),
        "proposal_cap": budget,
        "cap_removed": max(0, len(nonconflicting) - budget),
        **metrics,
    }


def _aggregate_acceptance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "accepted",
        "correct_edges",
        "known_negative_edges",
        "accepted_unlabeled",
        "capped_utility_delta",
        "motion_gated_sample_rows",
        "appearance_threshold_rows",
        "accepted_before_cap",
        "cap_removed",
    )
    totals = {key: sum(int(row[key]) for row in rows) for key in keys}
    definite = totals["correct_edges"] + totals["known_negative_edges"]
    totals.update(
        {
            "video_count": len(rows),
            "fit_precision": totals["correct_edges"] / max(totals["accepted"], 1),
            "definite_precision": totals["correct_edges"] / max(definite, 1),
        }
    )
    return totals


def _evaluate_tolerance(
    videos: Sequence[Mapping[str, Any]],
    *,
    tolerance_um: float,
    appearance_threshold: float,
    cap_fraction: float,
    minimum_cap: int,
) -> dict[str, Any]:
    rows = [
        _accepted_video(
            video,
            tolerance_um=tolerance_um,
            appearance_threshold=appearance_threshold,
            cap_fraction=cap_fraction,
            minimum_cap=minimum_cap,
        )
        for video in videos
    ]
    return {
        "tolerance_um": tolerance_um,
        "overall": _aggregate_acceptance(rows),
        "families": {
            family: _aggregate_acceptance(
                [row for row in rows if row["family"] == family]
            )
            for family in FAMILIES
        },
        "videos": rows,
    }


def _training_admissible(
    row: Mapping[str, Any],
    *,
    minimum_precision: float,
    minimum_family_precision: float,
    maximum_unlabeled: int,
) -> bool:
    overall = row["overall"]
    families = row["families"]
    return bool(
        int(overall["accepted"]) > 0
        and float(overall["fit_precision"]) >= minimum_precision
        and int(overall["capped_utility_delta"]) > 0
        and int(overall["accepted_unlabeled"]) <= maximum_unlabeled
        and all(
            float(families[family]["fit_precision"]) >= minimum_family_precision
            and int(families[family]["capped_utility_delta"]) >= 0
            for family in FAMILIES
        )
    )


def _select_tolerance(
    grid: Sequence[Mapping[str, Any]],
    *,
    minimum_precision: float,
    minimum_family_precision: float,
    maximum_unlabeled: int,
) -> tuple[dict[str, Any], bool]:
    admissible = [
        row
        for row in grid
        if _training_admissible(
            row,
            minimum_precision=minimum_precision,
            minimum_family_precision=minimum_family_precision,
            maximum_unlabeled=maximum_unlabeled,
        )
    ]
    if admissible:
        return (
            max(
                admissible,
                key=lambda row: (
                    int(row["overall"]["capped_utility_delta"]),
                    int(row["overall"]["correct_edges"]),
                    -int(row["overall"]["accepted_unlabeled"]),
                    -float(row["tolerance_um"]),
                ),
            ),
            True,
        )
    diagnostic = max(
        grid,
        key=lambda row: (
            float(row["overall"]["fit_precision"]),
            int(row["overall"]["capped_utility_delta"]),
            int(row["overall"]["correct_edges"]),
            -int(row["overall"]["accepted_unlabeled"]),
            -float(row["tolerance_um"]),
        ),
    )
    return dict(diagnostic), False


def _detection_index(
    cache: bh_model.SparseInferenceCache,
) -> dict[int, tuple[int, np.ndarray]]:
    result: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            key = int(node_id)
            if key in result:
                raise MotionGatedShellError(f"duplicate detection node: {key}")
            result[key] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64) * SCALE_ZYX,
            )
    return result


def _exact_motion_counts(
    videos: Sequence[Mapping[str, Any]],
    *,
    cache_root: Path,
    scorer: Any,
    tolerances: Sequence[float],
) -> dict[str, Any]:
    totals = {float(tolerance): 0 for tolerance in tolerances}
    partition_totals = {
        partition: {float(tolerance): 0 for tolerance in tolerances}
        for partition in ("train", "validation")
    }
    rows = []
    total_before = 0
    for index, video in enumerate(videos, start=1):
        identity = str(video["identity"])
        slug = identity.replace(":", "__", 1)
        cache_path = cache_root / f"{slug}.bhc"
        metadata_path = cache_root / f"{slug}.json"
        for path in (cache_path, metadata_path):
            try:
                assert_safe_fit_only_path(path)
            except AchievableHeadroomError as exc:
                raise MotionGatedShellError(str(exc)) from exc
            if not path.is_file():
                raise MotionGatedShellError(f"missing required input artifact: {path}")
        cache = bh_model.load_sparse_cache(
            cache_path, expected_identity=_cache_identity(metadata_path)
        )
        selected, _probabilities, _no_link, graph = _baseline(cache, scorer)
        if int(graph["raw_selected_edges"]) != int(video["selected_edges"]):
            raise MotionGatedShellError(f"selected graph drifted: {identity}")
        selected_in = {target: source for source, target in selected}
        if len(selected_in) != len(selected):
            raise MotionGatedShellError(f"selected graph is not one-to-one: {identity}")
        detections = _detection_index(cache)
        local_before = 0
        local = {float(tolerance): 0 for tolerance in tolerances}
        for source_frame, target_frame in zip(
            cache.detections[:-1], cache.detections[1:], strict=True
        ):
            source_ids = np.asarray(source_frame.node_ids, dtype=np.int64)
            target_ids = np.asarray(target_frame.node_ids, dtype=np.int64)
            source_points = (
                np.asarray(source_frame.coords_zyx, dtype=np.float64) * SCALE_ZYX
            )
            target_points = (
                np.asarray(target_frame.coords_zyx, dtype=np.float64) * SCALE_ZYX
            )
            if not len(source_points) or not len(target_points):
                continue
            shell = shell_candidate_indices(
                source_ids, source_points, target_ids, target_points
            )
            if not shell:
                continue
            source_indices = np.asarray([row[0] for row in shell], dtype=np.int64)
            displacements = np.stack([row[4] for row in shell])
            prediction_by_source = np.zeros((len(source_ids), 3), dtype=np.float64)
            history_by_source = np.zeros(len(source_ids), dtype=bool)
            for source_index, source_id in enumerate(source_ids):
                predecessor = selected_in.get(int(source_id))
                predecessor_record = (
                    detections.get(predecessor) if predecessor is not None else None
                )
                if (
                    predecessor_record is not None
                    and predecessor_record[0] == int(source_frame.frame_index) - 1
                ):
                    prediction_by_source[source_index] = (
                        source_points[source_index] - predecessor_record[1]
                    )
                    history_by_source[source_index] = True
            candidate_prediction = prediction_by_source[source_indices]
            candidate_history = history_by_source[source_indices]
            local_before += len(shell)
            for tolerance in tolerances:
                mask = motion_gate_mask(
                    displacements,
                    candidate_prediction,
                    candidate_history,
                    tolerance_um=float(tolerance),
                )
                local[float(tolerance)] += int(np.sum(mask))
        if local_before != int(video["shell_pairs"]):
            raise MotionGatedShellError(
                f"shared shell construction drifted for {identity}: "
                f"{local_before} != {video['shell_pairs']}"
            )
        total_before += local_before
        for tolerance in tolerances:
            value = local[float(tolerance)]
            totals[float(tolerance)] += value
            partition_totals[str(video["partition"])][float(tolerance)] += value
        rows.append(
            {
                "identity": identity,
                "partition": str(video["partition"]),
                "candidates_before": local_before,
                "candidates_after_by_tolerance": {
                    str(tolerance): local[float(tolerance)]
                    for tolerance in tolerances
                },
            }
        )
        print(
            f"motion shell {index:03d}/117 {identity} before={local_before}",
            flush=True,
        )
    return {
        "candidates_before": total_before,
        "candidates_after_by_tolerance": {
            str(tolerance): totals[float(tolerance)] for tolerance in tolerances
        },
        "partitions": {
            partition: {
                str(tolerance): values[float(tolerance)]
                for tolerance in tolerances
            }
            for partition, values in partition_totals.items()
        },
        "videos": rows,
    }


def _recall_grid(
    videos: Sequence[Mapping[str, Any]], tolerances: Sequence[float]
) -> dict[str, Any]:
    labels = np.concatenate([np.asarray(video["labels"]) for video in videos])
    partitions = np.concatenate(
        [
            np.full(len(video["labels"]), str(video["partition"]), dtype=object)
            for video in videos
        ]
    )
    families = np.concatenate(
        [
            np.full(len(video["labels"]), str(video["family"]), dtype=object)
            for video in videos
        ]
    )
    result = {}
    for tolerance in tolerances:
        masks = np.concatenate(
            [_motion_mask_from_features(video, float(tolerance)) for video in videos]
        )
        result[str(tolerance)] = {
            "all_fit": recall_accounting(
                labels, masks, expected_positive_total=456
            ),
            "partitions": {
                partition: recall_accounting(
                    labels[partitions == partition],
                    masks[partitions == partition],
                    expected_positive_total=405 if partition == "train" else 51,
                )
                for partition in ("train", "validation")
            },
            "families": {
                family: recall_accounting(
                    labels[families == family], masks[families == family]
                )
                for family in FAMILIES
            },
        }
    return result


def _write_immutable(path: Path, content: str) -> None:
    data = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise MotionGatedShellError(f"refusing to overwrite differing output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise MotionGatedShellError(f"stale temporary output exists: {temporary}")
    temporary.write_bytes(data)
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    validation = report["validation"]
    gates = report["fast_fail_gates"]
    return "\n".join(
        (
            "# Motion-gated shell recall screen",
            "",
            f"- Candidates: **{decision['candidates_before']:,} → "
            f"{decision['candidates_after_motion_gate']:,}** after the frozen motion tolerance.",
            f"- Recoverable correct-edge recall: **{decision['recall_456_retained']}/456**.",
            f"- Fit-validation conservative precision: **{decision['fit_precision']:.6f}**; "
            f"accepted unlabeled: **{validation['overall']['accepted_unlabeled']}**.",
            f"- Capped utility delta: **{decision['capped_utility_delta']:+d}**; "
            f"all fast-fail gates pass: **{all(bool(row['passed']) for row in gates.values())}**.",
            f"- Recommendation: **{decision['recommendation']}** — {decision['rationale']}",
            "",
            "Precision treats sampled unlabeled background as adverse for the fast-fail gate; "
            "it is not asserted to be a true negative label.",
            "",
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for path in (
        args.shell_prepare,
        args.shell_appearance,
        args.reachability,
        args.headroom,
        args.contract,
        args.edge_artifact,
        args.edge_auxiliary,
        args.cache,
        args.support_root,
        args.output,
    ):
        try:
            assert_safe_fit_only_path(path)
        except AchievableHeadroomError as exc:
            raise MotionGatedShellError(str(exc)) from exc
    for path in (args.edge_artifact, args.edge_auxiliary):
        if not path.is_file():
            raise MotionGatedShellError(f"missing required input artifact: {path}")
    for path in (args.cache, args.support_root):
        if not path.is_dir():
            raise MotionGatedShellError(f"missing required input artifact: {path}")
    shell_prepare = _load_json(args.shell_prepare)
    shell_appearance = _load_json(args.shell_appearance)
    reachability = _load_json(args.reachability)
    headroom = _load_json(args.headroom)
    contract_json = _load_json(args.contract)
    train_ids, validation_ids = _validate_scope(
        shell_prepare, shell_appearance, reachability, headroom
    )
    if int(shell_prepare.get("totals", {}).get("shell_pairs", -1)) != 5_342_071:
        raise MotionGatedShellError("shell candidate total drifted from 5,342,071")
    if int(shell_prepare.get("totals", {}).get("positive_pairs", -1)) != 456:
        raise MotionGatedShellError("recoverable shell-positive total drifted from 456")
    if contract_json.get("schema") != "celltrack-alpha.appearance-direct-swap-contract.v1":
        raise MotionGatedShellError("appearance contract schema drifted")
    contract_sha = _sha256(args.contract)
    try:
        contract, _frozen_geometry_gate = load_contract(
            args.contract, expected_sha256=contract_sha, root=ROOT
        )
    except Exception as exc:
        raise MotionGatedShellError("frozen appearance contract failed validation") from exc
    if tuple(contract["appearance"]["patch_shape"]) != PATCH_SHAPE:
        raise MotionGatedShellError("frozen appearance patch shape drifted")
    appearance_threshold = float(contract["appearance"]["threshold"])
    cap_fraction = float(contract["proposal_cap"]["selected_edge_fraction"])
    minimum_cap = int(contract["proposal_cap"]["minimum_per_video"])
    videos = _load_examples(
        args.shell_prepare,
        shell_prepare,
        args.shell_appearance,
        shell_appearance,
    )
    if (
        tuple(video["identity"] for video in videos if video["partition"] == "train")
        != train_ids
        or tuple(
            video["identity"] for video in videos if video["partition"] == "validation"
        )
        != validation_ids
    ):
        raise MotionGatedShellError("shell video ordering differs from frozen 105/12 split")
    train_videos = [video for video in videos if video["partition"] == "train"]
    validation_videos = [
        video for video in videos if video["partition"] == "validation"
    ]
    train_grid = [
        _evaluate_tolerance(
            train_videos,
            tolerance_um=tolerance,
            appearance_threshold=appearance_threshold,
            cap_fraction=cap_fraction,
            minimum_cap=minimum_cap,
        )
        for tolerance in TOLERANCE_GRID_UM
    ]
    selected_train, train_admissible = _select_tolerance(
        train_grid,
        minimum_precision=args.minimum_precision,
        minimum_family_precision=args.minimum_family_precision,
        maximum_unlabeled=args.maximum_train_unlabeled,
    )
    selected_tolerance = float(selected_train["tolerance_um"])
    validation = _evaluate_tolerance(
        validation_videos,
        tolerance_um=selected_tolerance,
        appearance_threshold=appearance_threshold,
        cap_fraction=cap_fraction,
        minimum_cap=minimum_cap,
    )
    recall_grid = _recall_grid(videos, TOLERANCE_GRID_UM)
    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact,
        args.edge_auxiliary,
        args.support_root,
        "cpu",
    )
    exact_counts = _exact_motion_counts(
        videos,
        cache_root=args.cache,
        scorer=scorer,
        tolerances=TOLERANCE_GRID_UM,
    )
    if exact_counts["candidates_before"] != 5_342_071:
        raise MotionGatedShellError("exact shell count does not reconcile prepared evidence")
    selected_recall = recall_grid[str(selected_tolerance)]["all_fit"]
    recall_retained = int(selected_recall["positive_retained"])
    candidates_after = int(
        exact_counts["candidates_after_by_tolerance"][str(selected_tolerance)]
    )

    precision_pass = (
        int(validation["overall"]["accepted"]) > 0
        and float(validation["overall"]["fit_precision"]) >= args.minimum_precision
    )
    family_pass = all(
        float(validation["families"][family]["fit_precision"])
        >= args.minimum_family_precision
        and int(validation["families"][family]["capped_utility_delta"]) >= 0
        for family in FAMILIES
    )
    utility_pass = int(validation["overall"]["capped_utility_delta"]) > 0
    background_pass = (
        int(validation["overall"]["accepted_unlabeled"])
        <= args.maximum_validation_unlabeled
    )
    fast_fail = {
        "fit_precision_at_least_0p95": {
            "passed": precision_pass,
            "value": float(validation["overall"]["fit_precision"]),
            "minimum": args.minimum_precision,
        },
        "both_family_precision_and_utility": {
            "passed": family_pass,
            "minimum_precision": args.minimum_family_precision,
            "families": {
                family: {
                    "precision": float(
                        validation["families"][family]["fit_precision"]
                    ),
                    "capped_utility_delta": int(
                        validation["families"][family]["capped_utility_delta"]
                    ),
                }
                for family in FAMILIES
            },
        },
        "positive_capped_utility": {
            "passed": utility_pass,
            "value": int(validation["overall"]["capped_utility_delta"]),
            "minimum_exclusive": 0,
        },
        "bounded_added_background": {
            "passed": background_pass,
            "accepted_unlabeled": int(validation["overall"]["accepted_unlabeled"]),
            "maximum": args.maximum_validation_unlabeled,
            "justification": (
                "At most one sampled-unlabeled accepted proposal per fixed "
                "fit-validation video."
            ),
        },
    }
    all_fast_fail_pass = all(bool(row["passed"]) for row in fast_fail.values())

    reach_all = reachability["all_fit"]["counts"]
    excluded_total = int(reach_all["default_excluded_edges"])
    if excluded_total != 719:
        raise MotionGatedShellError("excluded correct-edge total drifted from 719")
    headroom_recall = headroom["recall_side_ceiling"]
    current_pool_target = int(headroom_recall["target_swaps_in_current_pool"])
    excluded_target = int(headroom_recall["target_swaps_excluded_from_current_pool"])
    if current_pool_target + excluded_target != TARGET_IDEAL_SWAPS:
        raise MotionGatedShellError("headroom target-swap accounting drifted")
    projected_recovered = int(
        math.floor(excluded_target * recall_retained / excluded_total + 0.5)
    )
    projected_total = current_pool_target + projected_recovered
    projected_score = adjusted_score_after_perfect_swaps(projected_total)
    projected_margin = projected_score - BANKED_SCORE
    worthwhile_margin = args.minimum_worthwhile_relative_margin * BANKED_SCORE
    projection_pass = projected_margin >= worthwhile_margin
    promote = all_fast_fail_pass and projection_pass
    recommendation = "PROMOTE_TO_CV57_REPLAY" if promote else "PARK"
    failed_names = [
        name for name, value in fast_fail.items() if not bool(value["passed"])
    ]
    if not projection_pass:
        failed_names.append("projected_oracle_margin")
    rationale = (
        "all fit-only fast-fail gates pass and projected recall headroom clears "
        "the preregistered worthwhile-margin gate"
        if promote
        else "failed " + ", ".join(failed_names)
    )

    decision = {
        "candidates_before": int(exact_counts["candidates_before"]),
        "candidates_after_motion_gate": candidates_after,
        "recall_456_retained": recall_retained,
        "fit_precision": float(validation["overall"]["fit_precision"]),
        "per_family_precision": {
            family: float(validation["families"][family]["fit_precision"])
            for family in FAMILIES
        },
        "capped_utility_delta": int(
            validation["overall"]["capped_utility_delta"]
        ),
        "recommendation": recommendation,
        "rationale": rationale,
    }
    result = {
        "schema": SCHEMA,
        "status": recommendation,
        "seed": args.seed,
        "scope": {
            "fit_only": True,
            "train_video_count": 105,
            "validation_video_count": 12,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "execution": {
            "device": "cpu",
            "deterministic": True,
            "model_training": False,
            "learned_ranker_built": False,
            "transformer_built": False,
            "gpu_action": False,
            "kaggle_action": False,
            "submission_action": False,
        },
        "motion_model": {
            "name": "selected-track constant velocity",
            "prediction": (
                "source position minus its adjacent selected-graph predecessor "
                "position, in physical micrometres"
            ),
            "history_absence_policy": "abstain",
            "residual": "L2(candidate displacement - predicted displacement)",
            "tolerance_grid_um": list(TOLERANCE_GRID_UM),
            "selection_partition": "fit-train105 only",
            "selected_tolerance_um": selected_tolerance,
            "train_gate_admissible": train_admissible,
            "justification": (
                "Constant velocity is causal, parameter-free, already encoded by "
                "the selected-graph history features, and adds no learned weights."
            ),
        },
        "shell_gating": {
            "contract": "shared prepare_extended_radius_shell.shell_candidate_indices",
            "candidates_before": int(exact_counts["candidates_before"]),
            "candidates_after_motion_gate": candidates_after,
            "reduction_fraction": 1.0
            - candidates_after / max(int(exact_counts["candidates_before"]), 1),
            "exact_counts": exact_counts,
            "recall_grid": recall_grid,
            "recall_456_retained": recall_retained,
            "recall_456_fraction": recall_retained / 456.0,
        },
        "acceptance_contract": {
            "motion_before_appearance": True,
            "appearance_metric": contract["appearance"]["metric"],
            "appearance_threshold": appearance_threshold,
            "appearance_value": (
                "challenger NCC minus maximum selected-conflict NCC"
            ),
            "patch_shape": list(PATCH_SHAPE),
            "node_disjoint_ordering": (
                "appearance_swap._greedy_nonconflicting: descending margin, "
                "then challenger identity"
            ),
            "proposal_cap": dict(contract["proposal_cap"]),
            "conflict_required": True,
            "legacy_geometry_gate_recorded_but_not_extrapolated": {
                "threshold": float(contract["geometry_gate"]["threshold"]),
                "reason": (
                    "The learned r10 sparse-feature gate has no defined feature "
                    "representation for out-of-pool r10--12 candidates. The "
                    "deterministic motion gate is the shell geometry prefilter; "
                    "fabricating r10 scorer features would violate fail-closed reuse."
                ),
            },
            "unlabeled_semantics": (
                "sampled deployment background is unknown, but is counted as adverse "
                "for conservative fast-fail precision and utility"
            ),
        },
        "training_tolerance_grid": train_grid,
        "selected_training_operating_point": selected_train,
        "validation": validation,
        "fast_fail_gates": fast_fail,
        "projection": {
            "current_pool_target_swaps": current_pool_target,
            "excluded_target_swaps": excluded_target,
            "excluded_correct_edges_all_fit": excluded_total,
            "motion_retained_of_456": recall_retained,
            "projected_recovered_target_swaps": projected_recovered,
            "projected_total_perfect_swaps": projected_total,
            "projected_in_pool_plus_recovered_oracle_score": projected_score,
            "projected_oracle_margin_over_banked": projected_margin,
            "minimum_worthwhile_relative_margin": (
                args.minimum_worthwhile_relative_margin
            ),
            "minimum_worthwhile_absolute_margin": worthwhile_margin,
            "passed": projection_pass,
        },
        "decision": decision,
        "provenance": {
            "shell_prepare": str(args.shell_prepare),
            "shell_prepare_sha256": _sha256(args.shell_prepare),
            "shell_appearance": str(args.shell_appearance),
            "shell_appearance_sha256": _sha256(args.shell_appearance),
            "reachability": str(args.reachability),
            "reachability_sha256": _sha256(args.reachability),
            "headroom": str(args.headroom),
            "headroom_sha256": _sha256(args.headroom),
            "appearance_contract": str(args.contract),
            "appearance_contract_sha256": contract_sha,
            "edge_artifact": str(args.edge_artifact),
            "edge_artifact_sha256": _sha256(args.edge_artifact),
            "edge_record": edge_record,
            "example_files_hash_checked": len(videos),
            "referenced_artifacts_fit_only_guarded": True,
        },
    }
    encoded = json.dumps(
        result, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n"
    _write_immutable(args.output, encoded)
    markdown_path = args.output.with_suffix(".md")
    _write_immutable(markdown_path, _markdown(result))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--shell-prepare", type=Path, default=DEFAULT_SHELL_PREPARE)
    value.add_argument(
        "--shell-appearance", type=Path, default=DEFAULT_SHELL_APPEARANCE
    )
    value.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    value.add_argument("--headroom", type=Path, default=DEFAULT_HEADROOM)
    value.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--seed", type=int, default=1907)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--minimum-family-precision", type=float, default=0.93)
    value.add_argument("--maximum-train-unlabeled", type=int, default=105)
    value.add_argument("--maximum-validation-unlabeled", type=int, default=12)
    value.add_argument(
        "--minimum-worthwhile-relative-margin", type=float, default=0.035
    )
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report["decision"], sort_keys=True))
