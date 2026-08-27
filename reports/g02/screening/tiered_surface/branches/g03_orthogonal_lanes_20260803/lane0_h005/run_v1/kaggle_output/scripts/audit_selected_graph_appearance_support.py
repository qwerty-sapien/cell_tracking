#!/usr/bin/env python3
"""Audit deployment support for the fit-only selected-graph NCC rule.

The existing background sample stores challenger identities and
challenger-minus-selected displacement features.  Those are sufficient to
reconstruct the selected competitor against the raw detection cache without
rerunning the learned edge scorer or enumerating the 4.7M proposal pool.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.relative_geometry_gate import relative_features  # noqa: E402
from audit_selected_graph_appearance import (  # noqa: E402
    AppearanceAuditError,
    _proposal_groups,
    _sha256,
    extract_patch,
    ncc,
    proposal_metrics,
    unit_vector,
)
from p4_edge_attention import DEFAULT_CACHE, DEFAULT_ZIP, _cache_identity  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-appearance-support.v1"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
DISPLACEMENT_DELTA = slice(32, 35)
DEFAULT_LABELED = ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
DEFAULT_BACKGROUND = ROOT / "reports/g02/selected_graph_background_prepare_v1/prepare.json"
DEFAULT_APPEARANCE = ROOT / "reports/g02/selected_graph_appearance_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_appearance_support_v1.json"


class AppearanceSupportError(RuntimeError):
    """A support-audit identity or reconstruction invariant was violated."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AppearanceSupportError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AppearanceSupportError(f"expected JSON object: {path}")
    return value


def _detections(
    cache: bh_model.SparseInferenceCache,
) -> tuple[
    dict[int, tuple[int, np.ndarray]],
    dict[int, tuple[np.ndarray, np.ndarray, cKDTree]],
]:
    by_id: dict[int, tuple[int, np.ndarray]] = {}
    by_frame: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    for frame in cache.detections:
        ids = np.asarray(frame.node_ids, dtype=np.int64)
        coordinates = np.asarray(frame.coords_zyx, dtype=np.float64)
        if coordinates.shape != (len(ids), 3):
            raise AppearanceSupportError("raw detection arrays are malformed")
        for node_id, coordinate in zip(ids, coordinates, strict=True):
            key = int(node_id)
            if key in by_id:
                raise AppearanceSupportError("raw detection IDs repeat")
            by_id[key] = (int(frame.frame_index), coordinate)
        physical = coordinates * SCALE_ZYX[None, :]
        by_frame[int(frame.frame_index)] = (
            ids,
            physical,
            cKDTree(physical),
        )
    return by_id, by_frame


def reconstruct_winner_candidates(
    challenger: Sequence[int],
    displacement_delta_um: Sequence[float],
    detection_by_id: Mapping[int, tuple[int, np.ndarray]],
    detection_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, cKDTree]],
    *,
    tolerance_um: float,
) -> tuple[tuple[int, int], ...]:
    """Recover every geometrically compatible source/target conflict."""

    source_id, target_id = (int(value) for value in challenger)
    if source_id not in detection_by_id or target_id not in detection_by_id:
        raise AppearanceSupportError("challenger endpoint is absent from cache")
    source_t, source_coord = detection_by_id[source_id]
    target_t, target_coord = detection_by_id[target_id]
    if target_t != source_t + 1:
        raise AppearanceSupportError("challenger is not an adjacent-frame edge")
    delta = np.asarray(displacement_delta_um, dtype=np.float64)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise AppearanceSupportError("stored displacement delta is malformed")
    source_physical = source_coord * SCALE_ZYX
    target_physical = target_coord * SCALE_ZYX
    challenger_displacement = target_physical - source_physical
    winner_displacement = challenger_displacement - delta
    candidates: set[tuple[int, int]] = set()

    target_ids, _target_coords, target_tree = detection_by_frame[target_t]
    target_distance, target_index = target_tree.query(
        source_physical + winner_displacement, k=1
    )
    if float(target_distance) <= tolerance_um:
        pair = (source_id, int(target_ids[int(target_index)]))
        if pair != (source_id, target_id):
            candidates.add(pair)

    source_ids, _source_coords, source_tree = detection_by_frame[source_t]
    source_distance, source_index = source_tree.query(
        target_physical - winner_displacement, k=1
    )
    if float(source_distance) <= tolerance_um:
        pair = (int(source_ids[int(source_index)]), target_id)
        if pair != (source_id, target_id):
            candidates.add(pair)

    if not candidates:
        raise AppearanceSupportError(
            f"winner reconstruction found no candidate: challenger={(source_id, target_id)}"
        )
    return tuple(sorted(candidates))


def reconstruct_winner_pair(
    challenger: Sequence[int],
    displacement_delta_um: Sequence[float],
    detection_by_id: Mapping[int, tuple[int, np.ndarray]],
    detection_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, cKDTree]],
    *,
    tolerance_um: float,
) -> tuple[int, int]:
    """Recover a unique conflict, raising when geometry alone is ambiguous."""

    candidates = reconstruct_winner_candidates(
        challenger,
        displacement_delta_um,
        detection_by_id,
        detection_by_frame,
        tolerance_um=tolerance_um,
    )
    if len(candidates) != 1:
        raise AppearanceSupportError(
            f"winner reconstruction is not unique: challenger={tuple(challenger)} "
            f"candidates={list(candidates)}"
        )
    return candidates[0]


def _relative_from_labeled(value: Mapping[str, Any]) -> np.ndarray:
    labels = np.asarray(value["labels"], dtype=np.int8)
    count = len(labels)
    return relative_features(
        np.asarray(value["challenger_features"], dtype=np.float32),
        np.asarray(value["selected_features"], dtype=np.float32),
        np.asarray(value["challenger_geometry_um"], dtype=np.float32),
        np.asarray(value["selected_geometry_um"], dtype=np.float32),
        challenger_logit=np.asarray(value["challenger_logits"], dtype=np.float32),
        winner_logit=np.asarray(value["selected_logits"], dtype=np.float32),
        dummy_logit=0.0,
        candidate_count=np.full(count, 2, dtype=np.float32),
    )


def _features(
    array: zarr.Array,
    needed: set[int],
    detection_by_id: Mapping[int, tuple[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id in sorted(needed):
        by_frame[detection_by_id[node_id][0]].append(node_id)
    result: dict[int, np.ndarray] = {}
    for frame_index in sorted(by_frame):
        frame = np.asarray(array[frame_index])
        for node_id in by_frame[frame_index]:
            result[node_id] = unit_vector(
                extract_patch(frame, detection_by_id[node_id][1])
            )
    return result


def _validate_labeled_reconstruction(
    value: Mapping[str, Any],
    detection_by_id: Mapping[int, tuple[int, np.ndarray]],
    detection_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, cKDTree]],
    *,
    tolerance_um: float,
) -> dict[str, int]:
    challengers = np.asarray(value["challenger_pairs"], dtype=np.int64)
    selected = np.asarray(value["selected_pairs"], dtype=np.int64)
    relative = _relative_from_labeled(value)
    if (
        challengers.shape != selected.shape
        or challengers.shape != (len(relative), 2)
    ):
        raise AppearanceSupportError("labeled reconstruction arrays disagree")
    ambiguous = 0
    for index in range(len(relative)):
        observed = reconstruct_winner_candidates(
            challengers[index],
            relative[index, DISPLACEMENT_DELTA],
            detection_by_id,
            detection_by_frame,
            tolerance_um=tolerance_um,
        )
        expected = tuple(int(value) for value in selected[index])
        if expected not in observed:
            raise AppearanceSupportError(
                f"winner reconstruction omitted truth: {observed} does not contain {expected}"
            )
        ambiguous += len(observed) > 1
    return {"rows": len(relative), "ambiguous_rows": ambiguous}


def _background_proposals(
    value: Mapping[str, Any],
    detection_by_id: Mapping[int, tuple[int, np.ndarray]],
    detection_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray, cKDTree]],
    *,
    tolerance_um: float,
) -> list[dict[str, Any]]:
    relative = np.asarray(value["relative_features"], dtype=np.float32)
    lengths = np.asarray(value["proposal_lengths"], dtype=np.int64)
    challengers = np.asarray(value["challenger_pairs"], dtype=np.int64)
    if (
        relative.ndim != 2
        or relative.shape[1:] != (44,)
        or lengths.shape != (len(challengers),)
        or challengers.shape[1:] != (2,)
        or int(lengths.sum()) != len(relative)
    ):
        raise AppearanceSupportError("background example arrays disagree")
    proposals: list[dict[str, Any]] = []
    cursor = 0
    for challenger, length in zip(challengers, lengths, strict=True):
        winner_options = [
            reconstruct_winner_candidates(
                challenger,
                relative[cursor + offset, DISPLACEMENT_DELTA],
                detection_by_id,
                detection_by_frame,
                tolerance_um=tolerance_um,
            )
            for offset in range(int(length))
        ]
        cursor += int(length)
        proposals.append(
            {
                "challenger": tuple(int(value) for value in challenger),
                "winner_options": winner_options,
            }
        )
    return proposals


def _score_proposals(
    proposals: Sequence[Mapping[str, Any]],
    features: Mapping[int, np.ndarray],
) -> np.ndarray:
    result: list[float] = []
    for proposal in proposals:
        challenger = tuple(int(value) for value in proposal["challenger"])
        challenger_ncc = ncc(features[challenger[0]], features[challenger[1]])
        # Geometry-only compact artifacts can be ambiguous between a source
        # and target conflict.  Taking the best margin within each option set
        # is an upper bound on actual acceptance and is therefore fail-closed
        # for a deployment-density support check.
        row_margins = [
            max(
                challenger_ncc
                - ncc(features[winner[0]], features[winner[1]])
                for winner in options
            )
            for options in proposal["winner_options"]
        ]
        result.append(float(min(row_margins)))
    return np.asarray(result, dtype=np.float64)


def _support_threshold(
    by_video: Sequence[Mapping[str, Any]],
    *,
    lower_bound: float,
) -> float:
    threshold = float(lower_bound)
    for row in by_video:
        confidence = np.sort(
            np.asarray(row["confidence"], dtype=np.float64)
        )[::-1]
        budget = int(row["budget"])
        if len(confidence) > budget:
            threshold = max(
                threshold,
                float(np.nextafter(confidence[budget], float("inf"))),
            )
    if not math.isfinite(threshold):
        raise AppearanceSupportError("support threshold is non-finite")
    return threshold


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise AppearanceSupportError(f"refusing to overwrite output: {args.output}")
    labeled = _load_json(args.labeled)
    background = _load_json(args.background)
    appearance = _load_json(args.appearance)
    labeled_scope = labeled.get("scope")
    background_scope = background.get("scope")
    appearance_scope = appearance.get("scope")
    if (
        labeled.get("schema")
        != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or background.get("schema")
        != "celltrack-alpha.selected-graph-background-prepare.v1"
        or appearance.get("schema")
        != "celltrack-alpha.selected-graph-appearance-audit.v1"
        or not all(
            isinstance(value, Mapping)
            for value in (labeled_scope, background_scope, appearance_scope)
        )
        or labeled_scope.get("train_video_ids")
        != background_scope.get("train_video_ids")
        or labeled_scope.get("train_video_ids")
        != appearance_scope.get("train_video_ids")
        or labeled_scope.get("validation_video_ids")
        != appearance_scope.get("validation_video_ids")
        or any(
            labeled_scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise AppearanceSupportError("support inputs do not share exact fit-only scope")
    train_ids = tuple(str(value) for value in labeled_scope["train_video_ids"])
    validation_ids = tuple(
        str(value) for value in labeled_scope["validation_video_ids"]
    )
    labeled_by_id = {
        str(row["identity"]): ROOT / str(row["path"]) for row in labeled["videos"]
    }
    background_by_id = {
        str(row["identity"]): ROOT / str(row["path"])
        for row in background["videos"]
    }
    graph_by_id = {
        str(row["identity"]): row["graph"] for row in background["videos"]
    }
    if set(background_by_id) != set(train_ids):
        raise AppearanceSupportError("background video coverage is incomplete")
    original_threshold = float(appearance["ncc_only"]["train_threshold"])
    started = time.perf_counter()
    background_videos: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reconstruction_rows = 0
    reconstruction_ambiguous_rows = 0

    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate((*train_ids, *validation_ids), start=1):
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            detection_by_id, detection_by_frame = _detections(cache)
            labeled_value = torch.load(
                labeled_by_id[identity], map_location="cpu", weights_only=False
            )
            reconstruction = _validate_labeled_reconstruction(
                labeled_value,
                detection_by_id,
                detection_by_frame,
                tolerance_um=args.reconstruction_tolerance_um,
            )
            reconstruction_rows += reconstruction["rows"]
            reconstruction_ambiguous_rows += reconstruction["ambiguous_rows"]
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            if identity in train_ids:
                background_value = torch.load(
                    background_by_id[identity],
                    map_location="cpu",
                    weights_only=False,
                )
                proposals = _background_proposals(
                    background_value,
                    detection_by_id,
                    detection_by_frame,
                    tolerance_um=args.reconstruction_tolerance_um,
                )
                needed = {
                    node_id
                    for proposal in proposals
                    for pair in (
                        proposal["challenger"],
                        *(
                            winner
                            for options in proposal["winner_options"]
                            for winner in options
                        ),
                    )
                    for node_id in pair
                }
                feature_by_id = _features(array, needed, detection_by_id)
                confidence = _score_proposals(proposals, feature_by_id)
                selected_raw_edges = int(
                    graph_by_id[identity]["raw_selected_edges"]
                )
                budget = max(
                    args.minimum_video_budget,
                    int(math.ceil(args.max_proposal_rate * selected_raw_edges)),
                )
                background_videos.append(
                    {
                        "identity": identity,
                        "proposals": len(proposals),
                        "selected_raw_edges": selected_raw_edges,
                        "budget": budget,
                        "confidence": confidence.tolist(),
                    }
                )
            else:
                challengers = np.asarray(
                    labeled_value["challenger_pairs"], dtype=np.int64
                )
                winners = np.asarray(
                    labeled_value["selected_pairs"], dtype=np.int64
                )
                needed = set(challengers.ravel().tolist()) | set(
                    winners.ravel().tolist()
                )
                feature_by_id = _features(array, needed, detection_by_id)
                for row, (challenger, winner, label) in enumerate(
                    zip(
                        challengers,
                        winners,
                        np.asarray(labeled_value["labels"], dtype=np.int8),
                        strict=True,
                    )
                ):
                    challenger_pair = tuple(int(value) for value in challenger)
                    winner_pair = tuple(int(value) for value in winner)
                    validation_rows.append(
                        {
                            "identity": identity,
                            "challenger_pair": list(challenger_pair),
                            "label": int(label),
                            "ncc_margin": ncc(
                                feature_by_id[challenger_pair[0]],
                                feature_by_id[challenger_pair[1]],
                            )
                            - ncc(
                                feature_by_id[winner_pair[0]],
                                feature_by_id[winner_pair[1]],
                            ),
                        }
                    )
            print(
                f"appearance support {index:03d}/117 {identity}",
                flush=True,
            )

    support_threshold = _support_threshold(
        background_videos, lower_bound=original_threshold
    )
    validation_scores = np.asarray(
        [float(row["ncc_margin"]) for row in validation_rows], dtype=np.float64
    )
    validation_groups = _proposal_groups(
        [str(row["identity"]) for row in validation_rows],
        validation_rows,
    )
    validation = proposal_metrics(
        validation_scores, validation_groups, threshold=support_threshold
    )
    per_video: list[dict[str, Any]] = []
    all_confidence: list[float] = []
    for row in background_videos:
        confidence = np.asarray(row.pop("confidence"), dtype=np.float64)
        all_confidence.extend(confidence.tolist())
        accepted_original = int(np.sum(confidence >= original_threshold))
        accepted_support = int(np.sum(confidence >= support_threshold))
        budget = int(row["budget"])
        per_video.append(
            {
                **row,
                "accepted_at_train_threshold": accepted_original,
                "accepted_at_support_threshold": accepted_support,
                "within_budget_at_train_threshold": accepted_original <= budget,
                "within_budget_at_support_threshold": accepted_support <= budget,
                "support_budget_ratio": float(accepted_support / budget),
            }
        )
    background_confidence = np.asarray(all_confidence, dtype=np.float64)
    support_passed = (
        all(bool(row["within_budget_at_support_threshold"]) for row in per_video)
        and int(validation["accepted"]) >= args.minimum_validation_accepted
        and float(validation["precision"]) >= args.minimum_precision
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "scope": {
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "reconstruction": {
            "labeled_rows_supported": reconstruction_rows,
            "labeled_rows_ambiguous": reconstruction_ambiguous_rows,
            "tolerance_um": args.reconstruction_tolerance_um,
            "winner_displacement_source": "relative feature columns 32:35",
            "reran_edge_scorer": False,
            "background_acceptance_bound": (
                "upper: use the highest NCC margin across every geometrically "
                "compatible winner for each row"
            ),
        },
        "threshold": {
            "fit_train_labeled": original_threshold,
            "support_adjusted": support_threshold,
            "selection": (
                "smallest threshold no lower than the labeled-train threshold "
                "that keeps every sampled fit-train video within its sparse budget"
            ),
        },
        "background": {
            "proposals": len(background_confidence),
            "accepted_at_train_threshold": int(
                np.sum(background_confidence >= original_threshold)
            ),
            "accepted_at_support_threshold": int(
                np.sum(background_confidence >= support_threshold)
            ),
            "acceptance_rate_at_train_threshold": float(
                np.mean(background_confidence >= original_threshold)
            ),
            "acceptance_rate_at_support_threshold": float(
                np.mean(background_confidence >= support_threshold)
            ),
            "confidence_quantiles": {
                str(quantile): float(
                    np.quantile(background_confidence, quantile)
                )
                for quantile in (0.5, 0.9, 0.95, 0.99, 0.999)
            },
            "per_video": per_video,
        },
        "validation": validation,
        "support_passed": support_passed,
        "contract": {
            "minimum_validation_accepted": args.minimum_validation_accepted,
            "minimum_precision": args.minimum_precision,
            "max_proposal_rate": args.max_proposal_rate,
            "minimum_video_budget": args.minimum_video_budget,
            "full_pool_claim": False,
            "sample_role": "deterministic hard-plus-hash-uniform deployment support proxy",
        },
        "provenance": {
            "labeled": str(args.labeled),
            "labeled_sha256": _sha256(args.labeled),
            "background": str(args.background),
            "background_sha256": _sha256(args.background),
            "appearance": str(args.appearance),
            "appearance_sha256": _sha256(args.appearance),
        },
        "execution": {
            "device": "cpu",
            "kaggle_action": False,
            "gpu_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    value.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    value.add_argument("--appearance", type=Path, default=DEFAULT_APPEARANCE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--reconstruction-tolerance-um", type=float, default=1.0e-3)
    value.add_argument("--minimum-validation-accepted", type=int, default=10)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--max-proposal-rate", type=float, default=0.005)
    value.add_argument("--minimum-video-budget", type=int, default=16)
    return value


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "support_passed": report["support_passed"],
                "threshold": report["threshold"],
                "background": {
                    key: report["background"][key]
                    for key in (
                        "proposals",
                        "acceptance_rate_at_train_threshold",
                        "acceptance_rate_at_support_threshold",
                    )
                },
                "validation": report["validation"],
            },
            sort_keys=True,
        )
    )
