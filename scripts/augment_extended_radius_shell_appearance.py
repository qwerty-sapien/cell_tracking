#!/usr/bin/env python3
"""Add exact raw-patch appearance margins to the fit-only r10--12 um shell."""

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
from celltrack_alpha import bh_model  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
    _cache_identity,
    _load_base_scorer,
)
from prepare_extended_radius_shell import (  # noqa: E402
    _baseline,
    _sha256,
    topology_code,
)


SCHEMA = "celltrack-alpha.extended-radius-shell-appearance.v1"
DEFAULT_PREPARE = ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/extended_radius_shell_appearance_v1"
APPEARANCE_FEATURE_NAMES = (
    "pair_ncc",
    "conflict_ncc_available",
    "maximum_conflict_ncc",
    "pair_minus_maximum_conflict_ncc",
    "source_conflict_ncc",
    "target_conflict_ncc",
)


class ShellAppearanceError(RuntimeError):
    """An appearance-augmentation identity or shape contract failed."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShellAppearanceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ShellAppearanceError(f"expected JSON object: {path}")
    return value


def appearance_row_features(
    pair: Sequence[int],
    conflicts: Sequence[Sequence[int]],
    vectors: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Return challenger NCC and exact incumbent-relative NCC features."""

    if len(pair) != 2 or len(conflicts) > 2:
        raise ShellAppearanceError("appearance row has invalid pair/conflict shape")
    source, target = (int(value) for value in pair)
    try:
        pair_ncc = ncc(vectors[source], vectors[target])
    except KeyError as exc:
        raise ShellAppearanceError(f"missing pair vector: {exc}") from exc
    conflict_scores: list[float] = []
    source_conflict = 0.0
    target_conflict = 0.0
    for conflict in conflicts:
        if len(conflict) != 2:
            raise ShellAppearanceError("appearance conflict is not an edge")
        left, right = (int(value) for value in conflict)
        try:
            score = ncc(vectors[left], vectors[right])
        except KeyError as exc:
            raise ShellAppearanceError(f"missing conflict vector: {exc}") from exc
        conflict_scores.append(score)
        if left == source:
            source_conflict = score
        if right == target:
            target_conflict = score
    available = bool(conflict_scores)
    maximum = max(conflict_scores) if available else 0.0
    margin = pair_ncc - maximum if available else pair_ncc
    result = np.asarray(
        (
            pair_ncc,
            float(available),
            maximum,
            margin,
            source_conflict,
            target_conflict,
        ),
        dtype=np.float32,
    )
    if result.shape != (len(APPEARANCE_FEATURE_NAMES),) or not np.isfinite(
        result
    ).all():
        raise ShellAppearanceError("appearance row features are invalid")
    return result


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
                raise ShellAppearanceError(f"duplicate detection node {key}")
            result[key] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    return result


def _appearance_matrix(
    array: zarr.Array,
    detections: Mapping[int, tuple[int, np.ndarray]],
    pairs: np.ndarray,
    conflicts: Sequence[Sequence[tuple[int, int]]],
) -> np.ndarray:
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or len(conflicts) != len(pairs):
        raise ShellAppearanceError("appearance pair/conflict arrays disagree")
    by_time: dict[int, list[int]] = defaultdict(list)
    for row, pair in enumerate(pairs):
        source, target = (int(value) for value in pair)
        if source not in detections or target not in detections:
            raise ShellAppearanceError("shell pair references a missing detection")
        source_time = detections[source][0]
        if detections[target][0] != source_time + 1:
            raise ShellAppearanceError("shell pair is not adjacent")
        for conflict in conflicts[row]:
            left, right = conflict
            if (
                left not in detections
                or right not in detections
                or detections[left][0] != source_time
                or detections[right][0] != source_time + 1
            ):
                raise ShellAppearanceError("selected conflict is not aligned")
        by_time[source_time].append(row)

    result = np.empty(
        (len(pairs), len(APPEARANCE_FEATURE_NAMES)), dtype=np.float32
    )
    for source_time in sorted(by_time):
        rows = by_time[source_time]
        source_nodes: set[int] = set()
        target_nodes: set[int] = set()
        for row in rows:
            pair_edges = [tuple(int(value) for value in pairs[row])]
            pair_edges.extend(conflicts[row])
            source_nodes.update(edge[0] for edge in pair_edges)
            target_nodes.update(edge[1] for edge in pair_edges)
        source_frame = np.asarray(array[source_time])
        target_frame = np.asarray(array[source_time + 1])
        vectors = {
            node_id: unit_vector(
                extract_patch(source_frame, detections[node_id][1])
            )
            for node_id in sorted(source_nodes)
        }
        vectors.update(
            {
                node_id: unit_vector(
                    extract_patch(target_frame, detections[node_id][1])
                )
                for node_id in sorted(target_nodes)
            }
        )
        for row in rows:
            result[row] = appearance_row_features(
                pairs[row], conflicts[row], vectors
            )
    if not np.isfinite(result).all():
        raise ShellAppearanceError("appearance matrix contains non-finite values")
    return result


def augment(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ShellAppearanceError(f"refusing to overwrite output: {args.output}")
    prepare = _load_json(args.prepare)
    scope = prepare.get("scope")
    videos = prepare.get("videos")
    if (
        prepare.get("schema")
        != "celltrack-alpha.extended-radius-shell-prepare.v1"
        or not isinstance(scope, Mapping)
        or not isinstance(videos, list)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise ShellAppearanceError("input is not exact fit-only shell preparation")
    expected_feature_names = tuple(str(value) for value in prepare["feature_names"])
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
        for index, row in enumerate(videos, start=1):
            identity = str(row["identity"])
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            selected, _probabilities, _no_link, graph = _baseline(cache, scorer)
            if int(graph["raw_selected_edges"]) != int(
                row["graph"]["raw_selected_edges"]
            ):
                raise ShellAppearanceError(
                    f"selected graph count drifted for {identity}"
                )
            selected_out = {source: target for source, target in selected}
            selected_in = {target: source for source, target in selected}
            source = ROOT / str(row["path"])
            value = torch.load(source, map_location="cpu", weights_only=False)
            if (
                not isinstance(value, Mapping)
                or tuple(value.get("feature_names", ())) != expected_feature_names
            ):
                raise ShellAppearanceError(f"invalid shell example: {source}")
            pairs = np.asarray(value["pairs"], dtype=np.int64)
            topology = np.asarray(value["topology"], dtype=np.int8)
            conflicts: list[tuple[tuple[int, int], ...]] = []
            for pair, observed_topology in zip(pairs, topology, strict=True):
                computed_topology, row_conflicts = topology_code(
                    int(pair[0]),
                    int(pair[1]),
                    selected_out,
                    selected_in,
                )
                if computed_topology != int(observed_topology):
                    raise ShellAppearanceError(
                        f"topology drifted for {identity}"
                    )
                conflicts.append(row_conflicts)
            detections = _detection_index(cache)
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            appearance = _appearance_matrix(
                array, detections, pairs, conflicts
            )
            features = np.concatenate(
                (
                    np.asarray(value["features"], dtype=np.float32),
                    appearance,
                ),
                axis=1,
            )
            conflict_array = np.full((len(pairs), 2, 2), -1, dtype=np.int64)
            conflict_count = np.zeros(len(pairs), dtype=np.int8)
            for item, row_conflicts in enumerate(conflicts):
                conflict_count[item] = len(row_conflicts)
                for slot, conflict in enumerate(row_conflicts):
                    conflict_array[item, slot] = conflict
            output_value = {
                **dict(value),
                "feature_names": (
                    *expected_feature_names,
                    *APPEARANCE_FEATURE_NAMES,
                ),
                "features": features,
                "conflict_pairs": conflict_array,
                "conflict_count": conflict_count,
            }
            destination = examples / f"{slug}.pt"
            torch.save(output_value, destination)
            reports.append(
                {
                    "identity": identity,
                    "partition": str(row["partition"]),
                    "rows": len(pairs),
                    "path": str(destination.relative_to(ROOT)),
                    "sha256": _sha256(destination),
                    "pair_ncc_quantiles": {
                        str(quantile): float(
                            np.quantile(appearance[:, 0], quantile)
                        )
                        for quantile in (0.01, 0.1, 0.5, 0.9, 0.99)
                    }
                    if len(appearance)
                    else {},
                }
            )
            print(
                f"shell appearance {index:03d}/117 {identity} rows={len(pairs)}",
                flush=True,
            )
    result = {
        "schema": SCHEMA,
        "status": "augmented",
        "scope": dict(scope),
        "mechanism": {
            "appearance": (
                "zero-mean unit-norm NCC over 16x32x32 centroid-aligned raw patches"
            ),
            "incumbents": "exact frozen c=0.25 selected-graph conflicts",
            "predictions_emitted": False,
            "gnn": False,
        },
        "feature_names": [
            *expected_feature_names,
            *APPEARANCE_FEATURE_NAMES,
        ],
        "videos": reports,
        "totals": dict(prepare["totals"]),
        "frozen_pairwise": edge_record,
        "provenance": {
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
            "zip": str(args.zip),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument(
        "--support-root",
        type=Path,
        default=DEFAULT_SUPPORT,
    )
    return value


if __name__ == "__main__":
    report = augment(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "videos": len(report["videos"]),
                "elapsed_seconds": report["execution"]["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
