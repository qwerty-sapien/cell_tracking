#!/usr/bin/env python3
"""Prepare exact full-context labeled/background selected-graph proposals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from audit_selected_graph_appearance import (  # noqa: E402
    extract_patch,
    ncc,
    unit_vector,
)
from audit_selected_graph_appearance_support import (  # noqa: E402
    DISPLACEMENT_DELTA,
    _detections,
    reconstruct_winner_candidates,
)
from audit_selected_graph_tracklet_context import (  # noqa: E402
    context_features,
)
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.edge_data import (  # noqa: E402
    SOURCE_CONFLICT,
    TARGET_CONFLICT,
)
from celltrack_alpha.relative_geometry_gate import relative_features  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
    _cache_identity,
    _load_base_scorer,
)
from prepare_extended_radius_shell import _baseline, _sha256  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
DEFAULT_LABELED = (
    ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
)
DEFAULT_BACKGROUND = (
    ROOT / "reports/g02/selected_graph_background_prepare_v1/prepare.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1"
)
CONTEXT_FEATURE_NAMES = (
    "pair_ncc_margin",
    "context_ncc_margin",
    "motion_margin_10um",
    "context_available",
)


class TemporalBackgroundPrepareError(RuntimeError):
    """A full-context preparation identity or shape invariant failed."""


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise TemporalBackgroundPrepareError(f"expected JSON object: {path}")
    return value


def conflict_kind(
    challenger: Sequence[int], winner: Sequence[int]
) -> int:
    """Return the exact shared-source or shared-target conflict kind."""

    source, target = (int(value) for value in challenger)
    winner_source, winner_target = (int(value) for value in winner)
    if source == winner_source and target != winner_target:
        return SOURCE_CONFLICT
    if target == winner_target and source != winner_source:
        return TARGET_CONFLICT
    raise TemporalBackgroundPrepareError(
        f"winner is not a one-endpoint conflict: {challenger} vs {winner}"
    )


def _relative_labeled(value: Mapping[str, Any]) -> np.ndarray:
    count = len(np.asarray(value["labels"]))
    result = relative_features(
        np.asarray(value["challenger_features"], dtype=np.float32),
        np.asarray(value["selected_features"], dtype=np.float32),
        np.asarray(value["challenger_geometry_um"], dtype=np.float32),
        np.asarray(value["selected_geometry_um"], dtype=np.float32),
        challenger_logit=np.asarray(
            value["challenger_logits"], dtype=np.float32
        ),
        winner_logit=np.asarray(value["selected_logits"], dtype=np.float32),
        dummy_logit=0.0,
        candidate_count=np.full(count, 2, dtype=np.float32),
    )
    if result.shape != (count, 44):
        raise TemporalBackgroundPrepareError("labeled relative features drifted")
    return result


def _detection_vectors(
    array: zarr.Array,
    needed: set[int],
    detections: Mapping[int, tuple[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id in sorted(needed):
        if node_id not in detections:
            raise TemporalBackgroundPrepareError(
                f"context node is absent from detections: {node_id}"
            )
        by_frame[int(detections[node_id][0])].append(node_id)
    result: dict[int, np.ndarray] = {}
    for frame_index in sorted(by_frame):
        frame = np.asarray(array[frame_index])
        for node_id in by_frame[frame_index]:
            result[node_id] = unit_vector(
                extract_patch(frame, detections[node_id][1])
            )
    return result


def _context_row(
    *,
    challenger: tuple[int, int],
    winner: tuple[int, int],
    selected_in: Mapping[int, int],
    selected_out: Mapping[int, int],
    detections: Mapping[int, tuple[int, np.ndarray]],
    vectors: Mapping[int, np.ndarray],
) -> np.ndarray:
    pair_margin = ncc(vectors[challenger[0]], vectors[challenger[1]]) - ncc(
        vectors[winner[0]], vectors[winner[1]]
    )
    context_ncc, motion_margin, available = context_features(
        challenger=challenger,
        winner=winner,
        conflict_kind=conflict_kind(challenger, winner),
        selected_in=selected_in,
        selected_out=selected_out,
        detections=detections,
        vectors=vectors,
    )
    result = np.asarray(
        (pair_margin, context_ncc, motion_margin, float(available)),
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise TemporalBackgroundPrepareError("context features are non-finite")
    return result


def _needed_nodes(
    challenger: tuple[int, int],
    winners: Sequence[tuple[int, int]],
    selected_in: Mapping[int, int],
    selected_out: Mapping[int, int],
) -> set[int]:
    result = set(challenger)
    for winner in winners:
        result.update(winner)
        kind = conflict_kind(challenger, winner)
        if kind == SOURCE_CONFLICT:
            predecessor = selected_in.get(challenger[0])
            if predecessor is not None:
                result.add(predecessor)
        else:
            successor = selected_out.get(challenger[1])
            if successor is not None:
                result.add(successor)
    return result


def _video(
    *,
    identity: str,
    array: zarr.Array,
    cache: bh_model.SparseInferenceCache,
    scorer: Any,
    labeled_path: Path,
    background_path: Path | None,
    expected_edges: int,
    reconstruction_tolerance_um: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected, _probabilities, _no_link, graph = _baseline(cache, scorer)
    if int(graph["raw_selected_edges"]) != expected_edges:
        raise TemporalBackgroundPrepareError(
            f"selected graph count drifted for {identity}"
        )
    selected_in = {target: source for source, target in selected}
    selected_out = {source: target for source, target in selected}
    detections, detection_by_frame = _detections(cache)

    labeled = torch.load(labeled_path, map_location="cpu", weights_only=False)
    challengers = np.asarray(labeled["challenger_pairs"], dtype=np.int64)
    winners = np.asarray(labeled["selected_pairs"], dtype=np.int64)
    labels = np.asarray(labeled["labels"], dtype=np.int8)
    relative = _relative_labeled(labeled)
    if (
        challengers.shape != winners.shape
        or challengers.shape != (len(labels), 2)
        or np.any(~np.isin(labels, (0, 1)))
    ):
        raise TemporalBackgroundPrepareError("labeled arrays disagree")
    labeled_options: list[list[list[tuple[int, int]]]] = []
    needed: set[int] = set()
    for challenger_row, winner_row in zip(
        challengers, winners, strict=True
    ):
        challenger = tuple(int(value) for value in challenger_row)
        winner = tuple(int(value) for value in winner_row)
        if winner not in selected:
            raise TemporalBackgroundPrepareError(
                f"labeled incumbent is absent from full graph for {identity}"
            )
        labeled_options.append([[winner]])
        needed.update(
            _needed_nodes(challenger, (winner,), selected_in, selected_out)
        )

    background_relative = np.empty((0, 44), dtype=np.float32)
    background_challengers = np.empty((0, 2), dtype=np.int64)
    background_options: list[list[list[tuple[int, int]]]] = []
    if background_path is not None:
        background = torch.load(
            background_path, map_location="cpu", weights_only=False
        )
        background_relative = np.asarray(
            background["relative_features"], dtype=np.float32
        )
        lengths = np.asarray(background["proposal_lengths"], dtype=np.int64)
        background_challengers = np.asarray(
            background["challenger_pairs"], dtype=np.int64
        )
        if (
            background_relative.ndim != 2
            or background_relative.shape[1:] != (44,)
            or lengths.shape != (len(background_challengers),)
            or int(lengths.sum()) != len(background_relative)
        ):
            raise TemporalBackgroundPrepareError("background arrays disagree")
        cursor = 0
        for challenger_row, length in zip(
            background_challengers, lengths, strict=True
        ):
            challenger = tuple(int(value) for value in challenger_row)
            proposal: list[list[tuple[int, int]]] = []
            for offset in range(int(length)):
                options = list(
                    reconstruct_winner_candidates(
                        challenger,
                        background_relative[
                            cursor + offset, DISPLACEMENT_DELTA
                        ],
                        detections,
                        detection_by_frame,
                        tolerance_um=reconstruction_tolerance_um,
                    )
                )
                options = [winner for winner in options if winner in selected]
                if not options:
                    raise TemporalBackgroundPrepareError(
                        f"background reconstruction missed full graph in {identity}"
                    )
                proposal.append(options)
                needed.update(
                    _needed_nodes(
                        challenger, options, selected_in, selected_out
                    )
                )
            background_options.append(proposal)
            cursor += int(length)

    vectors = _detection_vectors(array, needed, detections)
    labeled_context = (
        np.stack(
            [
                _context_row(
                    challenger=tuple(int(value) for value in challenger),
                    winner=tuple(int(value) for value in winner),
                    selected_in=selected_in,
                    selected_out=selected_out,
                    detections=detections,
                    vectors=vectors,
                )
                for challenger, winner in zip(
                    challengers, winners, strict=True
                )
            ]
        )
        if len(labels)
        else np.empty((0, len(CONTEXT_FEATURE_NAMES)), dtype=np.float32)
    )
    labeled_features = np.concatenate(
        (relative, labeled_context), axis=1
    ).astype(np.float32, copy=False)

    background_rows: list[np.ndarray] = []
    background_groups: list[list[list[int]]] = []
    relative_cursor = 0
    for challenger_row, proposal in zip(
        background_challengers, background_options, strict=True
    ):
        challenger = tuple(int(value) for value in challenger_row)
        group: list[list[int]] = []
        for options in proposal:
            option_rows: list[int] = []
            base = background_relative[relative_cursor]
            for winner in options:
                option_rows.append(len(background_rows))
                background_rows.append(
                    np.concatenate(
                        (
                            base,
                            _context_row(
                                challenger=challenger,
                                winner=winner,
                                selected_in=selected_in,
                                selected_out=selected_out,
                                detections=detections,
                                vectors=vectors,
                            ),
                        )
                    )
                )
            group.append(option_rows)
            relative_cursor += 1
        background_groups.append(group)
    background_features = (
        np.stack(background_rows).astype(np.float32, copy=False)
        if background_rows
        else np.empty((0, 48), dtype=np.float32)
    )
    if labeled_features.shape != (len(labels), 48):
        raise TemporalBackgroundPrepareError("labeled feature shape drifted")
    example = {
        "feature_names": (
            *[f"relative_{index:02d}" for index in range(44)],
            *CONTEXT_FEATURE_NAMES,
        ),
        "labeled_features": labeled_features,
        "labeled_labels": labels,
        "labeled_challenger_pairs": challengers,
        "background_features": background_features,
        "background_groups": background_groups,
        "background_challenger_pairs": background_challengers,
    }
    report = {
        "identity": identity,
        "graph": graph,
        "labeled_rows": len(labels),
        "labeled_positive_rows": int(labels.sum()),
        "labeled_context_rows": int(np.sum(labeled_features[:, -1] > 0.5)),
        "background_proposals": len(background_groups),
        "background_rows": len(background_features),
        "background_context_rows": int(
            np.sum(background_features[:, -1] > 0.5)
        ),
    }
    return report, example


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise TemporalBackgroundPrepareError(
            f"refusing to overwrite output: {args.output}"
        )
    labeled = _load_json(args.labeled)
    background = _load_json(args.background)
    labeled_scope = labeled.get("scope")
    background_scope = background.get("scope")
    if (
        labeled.get("schema")
        != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or background.get("schema")
        != "celltrack-alpha.selected-graph-background-prepare.v1"
        or not isinstance(labeled_scope, Mapping)
        or not isinstance(background_scope, Mapping)
        or len(labeled_scope.get("train_video_ids", [])) != 105
        or len(labeled_scope.get("validation_video_ids", [])) != 12
        or background_scope.get("train_video_ids")
        != labeled_scope.get("train_video_ids")
        or any(
            labeled_scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise TemporalBackgroundPrepareError(
            "inputs do not share exact fit-only 105/12 scope"
        )
    train_ids = tuple(str(value) for value in labeled_scope["train_video_ids"])
    validation_ids = tuple(
        str(value) for value in labeled_scope["validation_video_ids"]
    )
    labeled_by_id = {
        str(row["identity"]): row for row in labeled["videos"]
    }
    background_by_id = {
        str(row["identity"]): row for row in background["videos"]
    }
    if set(labeled_by_id) != set((*train_ids, *validation_ids)) or set(
        background_by_id
    ) != set(train_ids):
        raise TemporalBackgroundPrepareError("input video coverage is incomplete")
    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact,
        args.edge_auxiliary,
        args.support_root,
        "cpu",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    examples = args.output / "examples"
    examples.mkdir()
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate((*train_ids, *validation_ids), start=1):
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            labeled_row = labeled_by_id[identity]
            background_row = background_by_id.get(identity)
            report, example = _video(
                identity=identity,
                array=array,
                cache=cache,
                scorer=scorer,
                labeled_path=ROOT / str(labeled_row["path"]),
                background_path=(
                    ROOT / str(background_row["path"])
                    if background_row is not None
                    else None
                ),
                expected_edges=int(labeled_row["graph"]["raw_selected_edges"]),
                reconstruction_tolerance_um=args.reconstruction_tolerance_um,
            )
            destination = examples / f"{slug}.pt"
            torch.save(example, destination)
            report.update(
                {
                    "partition": (
                        "train" if identity in train_ids else "validation"
                    ),
                    "path": str(destination.relative_to(ROOT)),
                    "sha256": _sha256(destination),
                }
            )
            reports.append(report)
            print(
                f"temporal background {index:03d}/117 {identity} "
                f"labeled={report['labeled_rows']} "
                f"background={report['background_proposals']}",
                flush=True,
            )
    result = {
        "schema": SCHEMA,
        "status": "prepared",
        "scope": {
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "mechanism": {
            "selected_graph": "exact full frozen c=0.25 graph",
            "appearance": "full 16x32x32 raw-patch NCC",
            "context": "predecessor/successor NCC and physical motion residual",
            "background": "existing hard+uniform deployment proposal sample",
            "predictions_emitted": False,
            "gnn": False,
        },
        "feature_names": [
            *[f"relative_{index:02d}" for index in range(44)],
            *CONTEXT_FEATURE_NAMES,
        ],
        "totals": {
            key: sum(int(row[key]) for row in reports)
            for key in (
                "labeled_rows",
                "labeled_positive_rows",
                "labeled_context_rows",
                "background_proposals",
                "background_rows",
                "background_context_rows",
            )
        },
        "videos": reports,
        "frozen_pairwise": edge_record,
        "provenance": {
            "labeled": str(args.labeled),
            "labeled_sha256": _sha256(args.labeled),
            "background": str(args.background),
            "background_sha256": _sha256(args.background),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
        },
    }
    run_bh.atomic_create_json(args.output / "prepare.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    value.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--reconstruction-tolerance-um", type=float, default=0.05)
    return value


if __name__ == "__main__":
    report = prepare(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "totals": report["totals"],
                "elapsed_seconds": report["execution"]["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
