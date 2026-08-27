#!/usr/bin/env python3
"""Screen tracklet-context features on fit-only selected-graph conflicts.

The current appearance rule compares one challenger edge with its incumbent
edge using centroid-aligned 3-D NCC.  This audit adds only local tracklet
context that is already present in the frozen selected graph:

* for a shared-source conflict, compare both targets with the source's
  predecessor and compare their acceleration residuals;
* for a shared-target conflict, compare both sources with the target's
  successor and compare their velocity continuation residuals.

The 105 fit-train videos select a coarse linear combination and threshold.  The
12 fit-validation videos are used only for a predeclared support decision.  No
CV57, audit, test, public, GPU, or submission signal is used.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from audit_selected_graph_appearance import (  # noqa: E402
    _proposal_groups,
    _select_threshold,
    extract_patch,
    ncc,
    proposal_metrics,
    roc_auc,
    unit_vector,
)
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.edge_data import SOURCE_CONFLICT, TARGET_CONFLICT  # noqa: E402
from p4_edge_attention import DEFAULT_CACHE, DEFAULT_ZIP, _cache_identity  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-tracklet-context-audit.v2"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/selected_graph_tracklet_context_audit_v2.json"
)


class TrackletContextError(RuntimeError):
    """A fit-only tracklet-context audit invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackletContextError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TrackletContextError(f"expected JSON object: {path}")
    return value


def context_features(
    *,
    challenger: tuple[int, int],
    winner: tuple[int, int],
    conflict_kind: int,
    selected_in: Mapping[int, int],
    selected_out: Mapping[int, int],
    detections: Mapping[int, tuple[int, np.ndarray]],
    vectors: Mapping[int, np.ndarray],
) -> tuple[float, float, bool]:
    """Return context-NCC and normalized motion margins for one conflict row."""

    source, target = challenger
    winner_source, winner_target = winner
    source_record = detections[source]
    target_record = detections[target]
    source_t = int(source_record[0])
    if int(target_record[0]) != source_t + 1:
        raise TrackletContextError("challenger is not adjacent in time")
    candidate_velocity = (
        np.asarray(target_record[1], dtype=np.float64)
        - np.asarray(source_record[1], dtype=np.float64)
    ) * SCALE_ZYX

    if conflict_kind == SOURCE_CONFLICT:
        if winner_source != source or winner_target == target:
            raise TrackletContextError("source conflict does not share its source")
        predecessor = selected_in.get(source)
        if predecessor is None:
            return 0.0, 0.0, False
        predecessor_record = detections.get(predecessor)
        winner_record = detections[winner_target]
        if (
            predecessor_record is None
            or int(predecessor_record[0]) != source_t - 1
            or int(winner_record[0]) != source_t + 1
        ):
            return 0.0, 0.0, False
        previous_velocity = (
            np.asarray(source_record[1], dtype=np.float64)
            - np.asarray(predecessor_record[1], dtype=np.float64)
        ) * SCALE_ZYX
        winner_velocity = (
            np.asarray(winner_record[1], dtype=np.float64)
            - np.asarray(source_record[1], dtype=np.float64)
        ) * SCALE_ZYX
        context_ncc = ncc(vectors[predecessor], vectors[target]) - ncc(
            vectors[predecessor], vectors[winner_target]
        )
        motion_margin = (
            np.linalg.norm(winner_velocity - previous_velocity)
            - np.linalg.norm(candidate_velocity - previous_velocity)
        ) / 10.0
        return float(context_ncc), float(motion_margin), True

    if conflict_kind == TARGET_CONFLICT:
        if winner_target != target or winner_source == source:
            raise TrackletContextError("target conflict does not share its target")
        successor = selected_out.get(target)
        if successor is None:
            return 0.0, 0.0, False
        successor_record = detections.get(successor)
        winner_record = detections[winner_source]
        if (
            successor_record is None
            or int(successor_record[0]) != source_t + 2
            or int(winner_record[0]) != source_t
        ):
            return 0.0, 0.0, False
        next_velocity = (
            np.asarray(successor_record[1], dtype=np.float64)
            - np.asarray(target_record[1], dtype=np.float64)
        ) * SCALE_ZYX
        winner_velocity = (
            np.asarray(target_record[1], dtype=np.float64)
            - np.asarray(winner_record[1], dtype=np.float64)
        ) * SCALE_ZYX
        context_ncc = ncc(vectors[source], vectors[successor]) - ncc(
            vectors[winner_source], vectors[successor]
        )
        motion_margin = (
            np.linalg.norm(winner_velocity - next_velocity)
            - np.linalg.norm(candidate_velocity - next_velocity)
        ) / 10.0
        return float(context_ncc), float(motion_margin), True

    raise TrackletContextError("unknown selected-graph conflict kind")


def _video_rows(
    array: zarr.Array,
    cache: bh_model.SparseInferenceCache,
    example_path: Path,
    identity: str,
) -> list[dict[str, Any]]:
    value = torch.load(example_path, map_location="cpu", weights_only=False)
    required = {
        "challenger_pairs",
        "selected_pairs",
        "conflict_kind",
        "labels",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise TrackletContextError(f"invalid prepared example: {example_path}")
    challengers = np.asarray(value["challenger_pairs"], dtype=np.int64)
    winners = np.asarray(value["selected_pairs"], dtype=np.int64)
    kinds = np.asarray(value["conflict_kind"], dtype=np.int8)
    labels = np.asarray(value["labels"], dtype=np.int8)
    count = len(labels)
    if (
        challengers.shape != (count, 2)
        or winners.shape != (count, 2)
        or kinds.shape != (count,)
        or np.any(~np.isin(kinds, (SOURCE_CONFLICT, TARGET_CONFLICT)))
        or np.any(~np.isin(labels, (0, 1)))
    ):
        raise TrackletContextError(f"invalid prepared shapes: {example_path}")

    selected = {tuple(int(v) for v in pair) for pair in winners.tolist()}
    selected_in = {target: source for source, target in selected}
    selected_out = {source: target for source, target in selected}
    if len(selected_in) != len(selected) or len(selected_out) != len(selected):
        raise TrackletContextError("prepared selected-edge subset is not one-to-one")

    detections: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            key = int(node_id)
            if key in detections:
                raise TrackletContextError(f"duplicate detection ID in {identity}")
            detections[key] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    needed = {
        int(node_id)
        for pairs in (challengers, winners)
        for node_id in pairs.ravel().tolist()
    }
    for source, target in challengers.tolist():
        predecessor = selected_in.get(int(source))
        successor = selected_out.get(int(target))
        if predecessor is not None:
            needed.add(predecessor)
        if successor is not None:
            needed.add(successor)
    missing = needed - set(detections)
    if missing:
        raise TrackletContextError(
            f"prepared/context nodes are missing in {identity}: {len(missing)}"
        )
    needed_by_frame: dict[int, list[int]] = {}
    for node_id in sorted(needed):
        needed_by_frame.setdefault(detections[node_id][0], []).append(node_id)
    vectors: dict[int, np.ndarray] = {}
    for frame_index in sorted(needed_by_frame):
        frame = np.asarray(array[frame_index])
        for node_id in needed_by_frame[frame_index]:
            vectors[node_id] = unit_vector(
                extract_patch(frame, detections[node_id][1])
            )

    rows: list[dict[str, Any]] = []
    for index in range(count):
        challenger = tuple(int(v) for v in challengers[index])
        winner = tuple(int(v) for v in winners[index])
        pair_margin = ncc(vectors[challenger[0]], vectors[challenger[1]]) - ncc(
            vectors[winner[0]], vectors[winner[1]]
        )
        context_ncc, motion_margin, available = context_features(
            challenger=challenger,
            winner=winner,
            conflict_kind=int(kinds[index]),
            selected_in=selected_in,
            selected_out=selected_out,
            detections=detections,
            vectors=vectors,
        )
        rows.append(
            {
                "identity": identity,
                "challenger_pair": list(challenger),
                "label": int(labels[index]),
                "pair_ncc_margin": pair_margin,
                "context_ncc_margin": context_ncc,
                "motion_margin_10um": motion_margin,
                "context_available": available,
            }
        )
    return rows


def _score(
    rows: Sequence[Mapping[str, Any]],
    *,
    context_weight: float,
    motion_weight: float,
) -> np.ndarray:
    values = np.asarray(
        [
            float(row["pair_ncc_margin"])
            + context_weight * float(row["context_ncc_margin"])
            + motion_weight * float(row["motion_margin_10um"])
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise TrackletContextError("tracklet-context score is non-finite")
    return values


def _cross_validate(
    rows: Sequence[Mapping[str, Any]],
    *,
    context_weight: float,
    motion_weight: float,
    fold_count: int,
    minimum_train_accepted: int,
    minimum_precision: float,
) -> dict[str, Any]:
    """Run deterministic video-grouped out-of-fold threshold evaluation."""

    if fold_count < 2:
        raise TrackletContextError("cross-validation requires at least two folds")
    identities = sorted({str(row["identity"]) for row in rows})
    assignment = {
        identity: int.from_bytes(
            hashlib.sha256(identity.encode("ascii")).digest()[:8], "big"
        )
        % fold_count
        for identity in identities
    }
    if set(assignment.values()) != set(range(fold_count)):
        raise TrackletContextError("deterministic video folds are incomplete")
    folds: list[dict[str, Any]] = []
    total = {
        "accepted": 0,
        "correct_rescues": 0,
        "harmful_swaps": 0,
        "occupied_correct_edges": 0,
        "occupied_harmful_edges": 0,
    }
    for fold in range(fold_count):
        fit_rows = [
            row for row in rows if assignment[str(row["identity"])] != fold
        ]
        holdout_rows = [
            row for row in rows if assignment[str(row["identity"])] == fold
        ]
        fit_groups = _proposal_groups(
            [str(row["identity"]) for row in fit_rows], fit_rows
        )
        holdout_groups = _proposal_groups(
            [str(row["identity"]) for row in holdout_rows], holdout_rows
        )
        fit_score = _score(
            fit_rows,
            context_weight=context_weight,
            motion_weight=motion_weight,
        )
        holdout_score = _score(
            holdout_rows,
            context_weight=context_weight,
            motion_weight=motion_weight,
        )
        fold_minimum = max(
            10,
            math.ceil(
                minimum_train_accepted * len(fit_rows) / max(1, len(rows))
            ),
        )
        threshold, _sweep = _select_threshold(
            fit_score,
            fit_groups,
            minimum_accepted=fold_minimum,
            minimum_precision=minimum_precision,
        )
        if threshold is None:
            raise TrackletContextError("one cross-validation fold has no threshold")
        metrics = proposal_metrics(
            holdout_score, holdout_groups, threshold=threshold
        )
        for key in total:
            total[key] += int(metrics[key])
        folds.append(
            {
                "fold": fold,
                "fit_video_count": sum(
                    assignment[identity] != fold for identity in identities
                ),
                "holdout_video_count": sum(
                    assignment[identity] == fold for identity in identities
                ),
                "fit_rows": len(fit_rows),
                "holdout_rows": len(holdout_rows),
                "threshold": threshold,
                "holdout": metrics,
            }
        )
    accepted = total["accepted"]
    correct = total["correct_rescues"]
    harmful = total["harmful_swaps"]
    return {
        **total,
        "precision": float(correct / accepted) if accepted else 0.0,
        "conservative_utility": correct - 2 * harmful,
        "fold_count": fold_count,
        "assignment": "sha256(identity)[0:8] modulo fold_count",
        "folds": folds,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise TrackletContextError(f"refusing to overwrite output: {args.output}")
    prepare = _load_json(args.prepare)
    scope = prepare.get("scope")
    videos = prepare.get("videos")
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or not isinstance(scope, Mapping)
        or not isinstance(videos, list)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise TrackletContextError("prepare report is not exact fit-only 105/12")
    train_ids = tuple(str(v) for v in scope["train_video_ids"])
    validation_ids = tuple(str(v) for v in scope["validation_video_ids"])
    expected_ids = (*train_ids, *validation_ids)
    by_id = {str(row["identity"]): ROOT / str(row["path"]) for row in videos}
    if set(by_id) != set(expected_ids):
        raise TrackletContextError("prepared video coverage is incomplete")

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    video_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(expected_ids, start=1):
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            part = _video_rows(array, cache, by_id[identity], identity)
            partition = "train" if identity in train_ids else "validation"
            rows[partition].extend(part)
            video_reports.append(
                {
                    "identity": identity,
                    "partition": partition,
                    "rows": len(part),
                    "context_rows": sum(
                        bool(row["context_available"]) for row in part
                    ),
                }
            )
            print(
                f"tracklet context {index:03d}/117 {identity} "
                f"rows={len(part)} context="
                f"{video_reports[-1]['context_rows']}",
                flush=True,
            )

    train_rows = rows["train"]
    validation_rows = rows["validation"]
    train_groups = _proposal_groups(
        [str(row["identity"]) for row in train_rows], train_rows
    )
    validation_groups = _proposal_groups(
        [str(row["identity"]) for row in validation_rows], validation_rows
    )
    grid: list[dict[str, Any]] = []
    for context_weight in args.context_weight:
        for motion_weight in args.motion_weight:
            train_score = _score(
                train_rows,
                context_weight=context_weight,
                motion_weight=motion_weight,
            )
            validation_score = _score(
                validation_rows,
                context_weight=context_weight,
                motion_weight=motion_weight,
            )
            threshold, _sweep = _select_threshold(
                train_score,
                train_groups,
                minimum_accepted=args.minimum_train_accepted,
                minimum_precision=args.minimum_precision,
            )
            train_metrics = (
                proposal_metrics(train_score, train_groups, threshold=threshold)
                if threshold is not None
                else None
            )
            validation_metrics = (
                proposal_metrics(
                    validation_score,
                    validation_groups,
                    threshold=threshold,
                )
                if threshold is not None
                else None
            )
            grid.append(
                {
                    "context_weight": context_weight,
                    "motion_weight": motion_weight,
                    "train_threshold": threshold,
                    "train_row_auc": roc_auc(
                        np.asarray(
                            [int(row["label"]) for row in train_rows],
                            dtype=np.int8,
                        ),
                        train_score,
                    ),
                    "validation_row_auc": roc_auc(
                        np.asarray(
                            [int(row["label"]) for row in validation_rows],
                            dtype=np.int8,
                        ),
                        validation_score,
                    ),
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "cross_validation": _cross_validate(
                        train_rows,
                        context_weight=context_weight,
                        motion_weight=motion_weight,
                        fold_count=args.fold_count,
                        minimum_train_accepted=args.minimum_train_accepted,
                        minimum_precision=args.minimum_precision,
                    ),
                }
            )
    eligible = [
        row
        for row in grid
        if isinstance(row["train"], Mapping)
        and int(row["cross_validation"]["accepted"])
        >= args.minimum_train_accepted
        and float(row["cross_validation"]["precision"])
        >= args.minimum_precision
    ]
    cross_validation_admissible = bool(eligible)
    selected = max(
        eligible if eligible else grid,
        key=lambda row: (
            int(row["cross_validation"]["conservative_utility"]),
            int(row["cross_validation"]["correct_rescues"]),
            -int(row["cross_validation"]["harmful_swaps"]),
            -abs(float(row["context_weight"])),
            -abs(float(row["motion_weight"])),
        ),
    )
    pair_only = next(
        row
        for row in grid
        if math.isclose(float(row["context_weight"]), 0.0)
        and math.isclose(float(row["motion_weight"]), 0.0)
    )
    selected_validation = selected["validation"]
    pair_validation = pair_only["validation"]
    validation_gain = (
        int(selected_validation["conservative_utility"])
        - int(pair_validation["conservative_utility"])
        if isinstance(selected_validation, Mapping)
        and isinstance(pair_validation, Mapping)
        else None
    )
    support_passed = (
        cross_validation_admissible
        and
        (float(selected["context_weight"]) != 0.0
         or float(selected["motion_weight"]) != 0.0)
        and isinstance(selected_validation, Mapping)
        and int(selected_validation["accepted"])
        >= args.minimum_validation_accepted
        and float(selected_validation["precision"]) >= args.minimum_precision
        and validation_gain is not None
        and validation_gain >= args.minimum_validation_utility_gain
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
        "mechanism": {
            "base": "centroid-aligned full-3D NCC challenger margin",
            "source_context": (
                "predecessor-to-target NCC margin plus acceleration-residual margin"
            ),
            "target_context": (
                "source-to-successor NCC margin plus continuation-residual margin"
            ),
            "selected_graph_context": (
                "conservative subset reconstructed from prepared incumbent rows"
            ),
            "global_rescore": False,
            "gnn": False,
        },
        "data": {
            "train_rows": len(train_rows),
            "train_proposals": len(train_groups),
            "train_context_rows": sum(
                bool(row["context_available"]) for row in train_rows
            ),
            "validation_rows": len(validation_rows),
            "validation_proposals": len(validation_groups),
            "validation_context_rows": sum(
                bool(row["context_available"]) for row in validation_rows
            ),
        },
        "selection_contract": {
            "combination_partition": (
                "deterministic five-fold video-grouped CV inside 105 fit-train"
            ),
            "final_threshold_partition": "all 105 fit-train videos",
            "minimum_train_accepted": args.minimum_train_accepted,
            "minimum_validation_accepted": args.minimum_validation_accepted,
            "minimum_precision": args.minimum_precision,
            "minimum_validation_utility_gain": (
                args.minimum_validation_utility_gain
            ),
            "next_gate": (
                "deployment-background support audit before any CV57 replay"
            ),
        },
        "pair_only": pair_only,
        "selected": selected,
        "cross_validation_admissible": cross_validation_admissible,
        "validation_utility_gain_vs_pair_only": validation_gain,
        "support_passed": support_passed,
        "grid": grid,
        "videos": video_reports,
        "provenance": {
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
            "zip": str(args.zip),
            "cache": str(args.cache),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--minimum-train-accepted", type=int, default=50)
    value.add_argument("--minimum-validation-accepted", type=int, default=10)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--minimum-validation-utility-gain", type=int, default=5)
    value.add_argument("--fold-count", type=int, default=5)
    value.add_argument(
        "--context-weight",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0, 2.0),
    )
    value.add_argument(
        "--motion-weight",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 1.0),
    )
    return value


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"],
                "validation_utility_gain_vs_pair_only": report[
                    "validation_utility_gain_vs_pair_only"
                ],
                "support_passed": report["support_passed"],
            },
            sort_keys=True,
        )
    )
