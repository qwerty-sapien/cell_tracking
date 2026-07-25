#!/usr/bin/env python3
"""Audit full-3D NCC on actual selected-graph rescue/control decisions.

This is a fit-only mechanism screen.  It reuses the exact 105/12 prepared
selected-graph comparisons, extracts centroid-aligned patches from the raw
image archive, and asks whether an appearance margin distinguishes a missed
true challenger from a correct selected edge.  It does not emit predictions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from p4_edge_attention import DEFAULT_CACHE, DEFAULT_ZIP, _cache_identity  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-appearance-audit.v1"
PATCH_SHAPE = (16, 32, 32)
DEFAULT_PREPARE = ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_appearance_audit_v1.json"


class AppearanceAuditError(RuntimeError):
    """A fit-only appearance-audit invariant was violated."""


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
        raise AppearanceAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AppearanceAuditError(f"expected JSON object: {path}")
    return value


def extract_patch(
    frame: np.ndarray,
    center_zyx: Sequence[float],
    shape: Sequence[int] = PATCH_SHAPE,
) -> np.ndarray:
    """Extract a fixed 3-D patch with deterministic reflected padding."""

    if frame.ndim != 3 or tuple(shape) != PATCH_SHAPE:
        raise AppearanceAuditError("appearance patches must be 16x32x32 in 3-D")
    center = np.rint(np.asarray(center_zyx, dtype=np.float64)).astype(int)
    shape_array = np.asarray(shape, dtype=int)
    starts = center - shape_array // 2
    ends = starts + shape_array
    slices = tuple(
        slice(max(0, int(start)), min(int(limit), int(end)))
        for start, end, limit in zip(starts, ends, frame.shape, strict=True)
    )
    patch = np.asarray(frame[slices], dtype=np.float32)
    padding = tuple(
        (max(0, -int(start)), max(0, int(end) - int(limit)))
        for start, end, limit in zip(starts, ends, frame.shape, strict=True)
    )
    if any(left or right for left, right in padding):
        mode = "reflect" if all(size > 1 for size in patch.shape) else "edge"
        patch = np.pad(patch, padding, mode=mode)
    if patch.shape != PATCH_SHAPE:
        raise AppearanceAuditError(
            f"patch shape mismatch: {patch.shape} != {PATCH_SHAPE}"
        )
    return patch


def unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - float(np.mean(values))
    norm = float(np.linalg.norm(centered.ravel()))
    if norm <= 1.0e-12:
        return np.zeros(centered.size, dtype=np.float32)
    return (centered / norm).ravel().astype(np.float32, copy=False)


def ncc(source: np.ndarray, target: np.ndarray) -> float:
    if source.shape != target.shape or source.ndim != 1:
        raise AppearanceAuditError("NCC vectors must be aligned one-dimensional arrays")
    return float(np.dot(source, target))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _proposal_groups(
    identities: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair = row["challenger_pair"]
        grouped[(str(identities[index]), int(pair[0]), int(pair[1]))].append(index)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        indices = grouped[key]
        labels = {int(rows[index]["label"]) for index in indices}
        if len(labels) != 1:
            raise AppearanceAuditError("one challenger has inconsistent labels")
        result.append(
            {
                "identity": key[0],
                "challenger_pair": [key[1], key[2]],
                "rows": indices,
                "label": labels.pop(),
            }
        )
    return result


def proposal_metrics(
    scores: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, int | float]:
    accepted = correct = harmful = occupied_correct = occupied_harmful = 0
    for group in groups:
        indices = np.asarray(group["rows"], dtype=np.int64)
        if float(np.min(scores[indices])) < threshold:
            continue
        accepted += 1
        if int(group["label"]) == 1:
            correct += 1
            occupied_correct += len(indices)
        else:
            harmful += 1
            occupied_harmful += len(indices)
    return {
        "accepted": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "occupied_correct_edges": occupied_correct,
        "occupied_harmful_edges": occupied_harmful,
        "precision": float(correct / accepted) if accepted else 0.0,
        "conservative_utility": correct - 2 * harmful,
    }


def _select_threshold(
    scores: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    *,
    minimum_accepted: int,
    minimum_precision: float,
) -> tuple[float | None, list[dict[str, int | float]]]:
    finite = np.asarray(scores[np.isfinite(scores)], dtype=np.float64)
    if len(finite) != len(scores):
        raise AppearanceAuditError("non-finite appearance score")
    confidences = np.asarray(
        [
            float(np.min(scores[np.asarray(group["rows"], dtype=np.int64)]))
            for group in groups
        ],
        dtype=np.float64,
    )
    labels = np.asarray([int(group["label"]) for group in groups], dtype=np.int8)
    occupied = np.asarray([len(group["rows"]) for group in groups], dtype=np.int64)
    if len(confidences) == 0:
        return None, []
    order = np.argsort(-confidences, kind="stable")
    ordered_confidence = confidences[order]
    ordered_labels = labels[order]
    ordered_occupied = occupied[order]
    cumulative_correct = np.cumsum(ordered_labels, dtype=np.int64)
    cumulative_harmful = np.cumsum(1 - ordered_labels, dtype=np.int64)
    cumulative_correct_edges = np.cumsum(
        ordered_occupied * ordered_labels, dtype=np.int64
    )
    cumulative_harmful_edges = np.cumsum(
        ordered_occupied * (1 - ordered_labels), dtype=np.int64
    )
    block_ends = np.flatnonzero(
        np.r_[ordered_confidence[:-1] != ordered_confidence[1:], True]
    )
    sweep: list[dict[str, int | float]] = [
        {
            "threshold": float(
                np.nextafter(float(ordered_confidence[0]), float("inf"))
            ),
            "accepted": 0,
            "correct_rescues": 0,
            "harmful_swaps": 0,
            "occupied_correct_edges": 0,
            "occupied_harmful_edges": 0,
            "precision": 0.0,
            "conservative_utility": 0,
        }
    ]
    for end in block_ends.tolist():
        accepted = end + 1
        correct = int(cumulative_correct[end])
        harmful = int(cumulative_harmful[end])
        sweep.append(
            {
                "threshold": float(ordered_confidence[end]),
                "accepted": accepted,
                "correct_rescues": correct,
                "harmful_swaps": harmful,
                "occupied_correct_edges": int(cumulative_correct_edges[end]),
                "occupied_harmful_edges": int(cumulative_harmful_edges[end]),
                "precision": float(correct / accepted),
                "conservative_utility": correct - 2 * harmful,
            }
        )
    admissible = [
        row
        for row in sweep
        if int(row["accepted"]) >= minimum_accepted
        and float(row["precision"]) >= minimum_precision
    ]
    if not admissible:
        return None, sweep
    selected = max(
        admissible,
        key=lambda row: (
            int(row["conservative_utility"]),
            int(row["correct_rescues"]),
            -int(row["harmful_swaps"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), sweep


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
        "challenger_logits",
        "selected_logits",
        "conflict_kind",
        "labels",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise AppearanceAuditError(f"invalid prepared example: {example_path}")
    challengers = np.asarray(value["challenger_pairs"], dtype=np.int64)
    selected = np.asarray(value["selected_pairs"], dtype=np.int64)
    challenger_logits = np.asarray(value["challenger_logits"], dtype=np.float32)
    selected_logits = np.asarray(value["selected_logits"], dtype=np.float32)
    kinds = np.asarray(value["conflict_kind"], dtype=np.int8)
    labels = np.asarray(value["labels"], dtype=np.int8)
    count = len(labels)
    if (
        challengers.shape != (count, 2)
        or selected.shape != (count, 2)
        or challenger_logits.shape != (count,)
        or selected_logits.shape != (count,)
        or kinds.shape != (count,)
        or np.any(~np.isin(labels, (0, 1)))
    ):
        raise AppearanceAuditError(f"invalid prepared shapes: {example_path}")

    detection: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            key = int(node_id)
            if key in detection:
                raise AppearanceAuditError(f"duplicate detection ID in {identity}")
            detection[key] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    needed = {
        int(node_id)
        for pairs in (challengers, selected)
        for node_id in pairs.ravel().tolist()
    }
    missing = needed - set(detection)
    if missing:
        raise AppearanceAuditError(
            f"prepared pair references missing detections in {identity}: {len(missing)}"
        )
    needed_by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id in sorted(needed):
        needed_by_frame[detection[node_id][0]].append(node_id)
    features: dict[int, np.ndarray] = {}
    for frame_index in sorted(needed_by_frame):
        frame = np.asarray(array[frame_index])
        for node_id in needed_by_frame[frame_index]:
            features[node_id] = unit_vector(
                extract_patch(frame, detection[node_id][1])
            )

    rows: list[dict[str, Any]] = []
    for index in range(count):
        challenger = tuple(int(value) for value in challengers[index])
        winner = tuple(int(value) for value in selected[index])
        for pair in (challenger, winner):
            source_t = detection[pair[0]][0]
            target_t = detection[pair[1]][0]
            if target_t != source_t + 1:
                raise AppearanceAuditError(
                    f"non-adjacent selected-graph pair in {identity}"
                )
        challenger_ncc = ncc(features[challenger[0]], features[challenger[1]])
        selected_ncc = ncc(features[winner[0]], features[winner[1]])
        rows.append(
            {
                "identity": identity,
                "challenger_pair": list(challenger),
                "selected_pair": list(winner),
                "conflict_kind": int(kinds[index]),
                "label": int(labels[index]),
                "challenger_ncc": challenger_ncc,
                "selected_ncc": selected_ncc,
                "ncc_margin": challenger_ncc - selected_ncc,
                "logit_margin": float(challenger_logits[index] - selected_logits[index]),
            }
        )
    return rows


def _partition_summary(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    threshold: float | None,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    result: dict[str, Any] = {
        "rows": len(rows),
        "proposals": len(groups),
        "rescue_rows": int(labels.sum()),
        "control_rows": int(len(labels) - labels.sum()),
        "row_auc": roc_auc(labels, scores),
        "zero_margin_row_accuracy": float(
            np.mean((scores > 0.0).astype(np.int8) == labels)
        ),
        "margin_quantiles": {
            str(quantile): float(np.quantile(scores, quantile))
            for quantile in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
    }
    if threshold is not None:
        result["frozen_threshold_metrics"] = proposal_metrics(
            scores, groups, threshold=threshold
        )
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise AppearanceAuditError(f"refusing to overwrite output: {args.output}")
    prepare = _load_json(args.prepare)
    scope = prepare.get("scope")
    videos = prepare.get("videos")
    if (
        prepare.get("schema") != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or not isinstance(scope, Mapping)
        or not isinstance(videos, list)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise AppearanceAuditError("prepare report is not exact fit-only 105/12")
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    expected_ids = (*train_ids, *validation_ids)
    by_id = {str(row["identity"]): ROOT / str(row["path"]) for row in videos}
    if set(by_id) != set(expected_ids):
        raise AppearanceAuditError("prepared video coverage is incomplete")

    started = time.perf_counter()
    rows_by_partition: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    video_reports: list[dict[str, Any]] = []
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(expected_ids, start=1):
            slug = identity.replace(":", "__", 1)
            cache_path = args.cache / f"{slug}.bhc"
            meta_path = args.cache / f"{slug}.json"
            cache = bh_model.load_sparse_cache(
                cache_path, expected_identity=_cache_identity(meta_path)
            )
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            part = _video_rows(array, cache, by_id[identity], identity)
            partition = "train" if identity in train_ids else "validation"
            rows_by_partition[partition].extend(part)
            video_reports.append(
                {
                    "identity": identity,
                    "partition": partition,
                    "rows": len(part),
                    "rescue_rows": sum(int(row["label"]) for row in part),
                    "control_rows": sum(1 - int(row["label"]) for row in part),
                }
            )
            print(
                f"appearance audit {index:03d}/117 {identity} rows={len(part)}",
                flush=True,
            )

    train_rows = rows_by_partition["train"]
    validation_rows = rows_by_partition["validation"]
    train_identities = [str(row["identity"]) for row in train_rows]
    validation_identities = [str(row["identity"]) for row in validation_rows]
    train_groups = _proposal_groups(train_identities, train_rows)
    validation_groups = _proposal_groups(validation_identities, validation_rows)
    train_ncc = np.asarray(
        [float(row["ncc_margin"]) for row in train_rows], dtype=np.float64
    )
    validation_ncc = np.asarray(
        [float(row["ncc_margin"]) for row in validation_rows], dtype=np.float64
    )
    train_threshold, train_sweep = _select_threshold(
        train_ncc,
        train_groups,
        minimum_accepted=args.minimum_train_accepted,
        minimum_precision=args.minimum_precision,
    )

    alphas = tuple(float(value) for value in args.alpha)
    additive_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        train_score = np.asarray(
            [
                float(row["logit_margin"]) + alpha * float(row["ncc_margin"])
                for row in train_rows
            ],
            dtype=np.float64,
        )
        validation_score = np.asarray(
            [
                float(row["logit_margin"]) + alpha * float(row["ncc_margin"])
                for row in validation_rows
            ],
            dtype=np.float64,
        )
        threshold, _sweep = _select_threshold(
            train_score,
            train_groups,
            minimum_accepted=args.minimum_train_accepted,
            minimum_precision=args.minimum_precision,
        )
        validation_metrics = (
            proposal_metrics(validation_score, validation_groups, threshold=threshold)
            if threshold is not None
            else None
        )
        additive_rows.append(
            {
                "alpha": alpha,
                "train_threshold": threshold,
                "train_row_auc": roc_auc(
                    np.asarray([int(row["label"]) for row in train_rows]),
                    train_score,
                ),
                "validation_row_auc": roc_auc(
                    np.asarray([int(row["label"]) for row in validation_rows]),
                    validation_score,
                ),
                "validation": validation_metrics,
            }
        )
    admissible = [
        row
        for row in additive_rows
        if isinstance(row["validation"], Mapping)
        and int(row["validation"]["accepted"]) >= args.minimum_validation_accepted
        and float(row["validation"]["precision"]) >= args.minimum_precision
    ]
    selected_additive = (
        max(
            admissible,
            key=lambda row: (
                int(row["validation"]["conservative_utility"]),
                int(row["validation"]["correct_rescues"]),
                -int(row["validation"]["harmful_swaps"]),
                -abs(float(row["alpha"])),
            ),
        )
        if admissible
        else None
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
            "appearance": "zero-mean unit-norm NCC over 16x32x32 centroid-aligned raw patches",
            "decision": "challenger NCC minus actual selected-edge NCC",
            "proposal_acceptance": "every occupied-endpoint row clears the frozen margin threshold",
            "global_rescore": False,
            "attention": False,
            "gnn": False,
        },
        "selection_contract": {
            "minimum_train_accepted": args.minimum_train_accepted,
            "minimum_validation_accepted": args.minimum_validation_accepted,
            "minimum_precision": args.minimum_precision,
            "threshold_fit_partition": "105 fit-train videos only",
            "validation_role": "mechanism support decision only",
        },
        "ncc_only": {
            "train_threshold": train_threshold,
            "train": _partition_summary(
                train_rows, train_ncc, train_groups, train_threshold
            ),
            "validation": _partition_summary(
                validation_rows, validation_ncc, validation_groups, train_threshold
            ),
            "train_threshold_sweep": train_sweep,
        },
        "additive_logit_ncc": {
            "grid": additive_rows,
            "selected_validation_admissible": selected_additive,
        },
        "videos": video_reports,
        "provenance": {
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
            "archive": str(args.zip),
            "archive_bytes": args.zip.stat().st_size,
            "cache": str(args.cache),
            "patch_contract_source": "scripts/falsify_e4_e5.py",
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
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--minimum-train-accepted", type=int, default=50)
    value.add_argument("--minimum-validation-accepted", type=int, default=10)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    )
    return value


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "ncc_train_threshold": report["ncc_only"]["train_threshold"],
                "ncc_validation": report["ncc_only"]["validation"].get(
                    "frozen_threshold_metrics"
                ),
                "selected_additive": report["additive_logit_ncc"][
                    "selected_validation_admissible"
                ],
            },
            sort_keys=True,
        )
    )
