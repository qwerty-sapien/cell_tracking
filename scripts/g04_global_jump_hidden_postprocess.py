#!/usr/bin/env python3
"""Apply frozen G1 whole-frame registration to a fresh APNCC run.

The runtime discovers the mounted test identities, consumes the APNCC caches
and fragments produced in the same kernel execution, estimates one label-blind
integer phase shift per complete frame transition, and rebuilds only the
association geometry on shifts whose physical magnitude is strictly above
10 um.  Evaluator node coordinates never change.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile


CSV_FIELDS = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
)
SHAPE_TZYX = (100, 64, 256, 256)
STRIDE_ZYX = (1, 4, 4)
SCALE_ZYX_UM = (1.625, 0.40625, 0.40625)
REGISTERED_SCALE_ZYX_UM = (1.625, 1.625, 1.625)
POOL_RADIUS_UM = 10.0
EDGE_RELATIVE = Path("reports/g02/conflict_edge_f0_fit117_v1/edge.json")
EDGE_SHA256 = "5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034"
AUXILIARY_RELATIVE = Path("reports/g02/v45f4.pt")
AUXILIARY_SHA256 = "1cff90475b819b5652310ccb73ba57e1dc20069885962a06a5e10a6060c04473"
SAFE_DATASET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class GlobalJumpError(RuntimeError):
    """A fresh-run, raw-input, scorer, or graph invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_dataset(value: str) -> str:
    if len(value) > 200 or SAFE_DATASET.fullmatch(value) is None:
        raise GlobalJumpError(f"unsafe dataset label: {value!r}")
    return value


def _integer(row: Mapping[str, str], field: str, row_number: int) -> int:
    try:
        value = str(row[field]).strip()
        result = int(value)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise GlobalJumpError(f"row {row_number}: invalid integer {field}") from exc
    if str(result) != value:
        raise GlobalJumpError(f"row {row_number}: noncanonical integer {field}")
    return result


def read_rows(path: Path) -> tuple[dict[str, list[dict[str, str]]], int]:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise GlobalJumpError(f"submission schema drifted: {path}")
        for index, raw in enumerate(reader):
            row_number = index + 2
            if _integer(raw, "id", row_number) != index:
                raise GlobalJumpError("submission ids are not contiguous")
            dataset = _safe_dataset(str(raw.get("dataset", "")))
            row_type = str(raw.get("row_type", ""))
            if row_type not in {"node", "edge"}:
                raise GlobalJumpError(f"row {row_number}: invalid row type")
            for field in CSV_FIELDS:
                if field not in {"dataset", "row_type"}:
                    _integer(raw, field, row_number)
            by_dataset[dataset].append(dict(raw))
    if not by_dataset:
        raise GlobalJumpError("submission has no datasets")
    validate_rows(by_dataset)
    return dict(by_dataset), sum(map(len, by_dataset.values()))


def node_times(rows: Sequence[Mapping[str, str]]) -> dict[int, int]:
    result = {
        int(row["node_id"]): int(row["t"])
        for row in rows if row["row_type"] == "node"
    }
    if not result or len(result) != sum(row["row_type"] == "node" for row in rows):
        raise GlobalJumpError("node inventory is empty or duplicated")
    return result


def edge_pairs(rows: Sequence[Mapping[str, str]]) -> set[tuple[int, int]]:
    result = {
        (int(row["source_id"]), int(row["target_id"]))
        for row in rows if row["row_type"] == "edge"
    }
    if len(result) != sum(row["row_type"] == "edge" for row in rows):
        raise GlobalJumpError("edge inventory contains duplicates")
    return result


def validate_rows(by_dataset: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, int]:
    totals = Counter()
    for dataset, rows in by_dataset.items():
        times = node_times(rows)
        edges = edge_pairs(rows)
        indegree: Counter[int] = Counter()
        outdegree: Counter[int] = Counter()
        for source, target in edges:
            if source not in times or target not in times:
                raise GlobalJumpError(f"{dataset}: edge endpoint is absent")
            if times[target] != times[source] + 1:
                raise GlobalJumpError(f"{dataset}: edge is not adjacent-time")
            indegree[target] += 1
            outdegree[source] += 1
            if indegree[target] > 1 or outdegree[source] > 1:
                raise GlobalJumpError(f"{dataset}: association graph is not one-to-one")
        totals["datasets"] += 1
        totals["nodes"] += len(times)
        totals["edges"] += len(edges)
    return dict(totals)


def rows_with_edges(
    rows: Sequence[Mapping[str, str]], edges: set[tuple[int, int]]
) -> list[dict[str, Any]]:
    nodes = [dict(row) for row in rows if row["row_type"] == "node"]
    if not nodes:
        raise GlobalJumpError("fragment has no nodes")
    dataset = str(nodes[0]["dataset"])
    node_ids = {int(row["node_id"]) for row in nodes}
    if any(source not in node_ids or target not in node_ids for source, target in edges):
        raise GlobalJumpError("replacement edge references an absent retained node")
    edge_rows = [
        {
            "id": -1, "dataset": dataset, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": source, "target_id": target,
        }
        for source, target in sorted(edges)
    ]
    return nodes + edge_rows


def write_rows(by_dataset: Mapping[str, Sequence[Mapping[str, Any]]], path: Path) -> int:
    if path.exists():
        raise GlobalJumpError(f"refusing to overwrite output: {path}")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    row_id = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            for dataset in sorted(by_dataset):
                rows = by_dataset[dataset]
                nodes = sorted(
                    (row for row in rows if row["row_type"] == "node"),
                    key=lambda row: (int(row["t"]), int(row["node_id"])),
                )
                edges = sorted(
                    (row for row in rows if row["row_type"] == "edge"),
                    key=lambda row: (int(row["source_id"]), int(row["target_id"])),
                )
                for row in (*nodes, *edges):
                    value = dict(row)
                    value["id"] = row_id
                    writer.writerow({field: value[field] for field in CSV_FIELDS})
                    row_id += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return row_id


def robust_windowed(frame_zyx: Any) -> Any:
    import numpy as np

    values = np.asarray(frame_zyx, dtype=np.float32)
    if values.shape != (64, 64, 64) or not np.isfinite(values).all():
        raise GlobalJumpError("registration frame must be finite 64x64x64")
    low, high = np.quantile(values, (0.01, 0.99))
    clipped = np.clip(values, low, high) - np.median(np.clip(values, low, high))
    windows = [np.hanning(size).astype(np.float32) for size in clipped.shape]
    window = windows[0][:, None, None] * windows[1][None, :, None]
    window = window * windows[2][None, None, :]
    result = np.ascontiguousarray(clipped * window, dtype=np.float32)
    if float(np.linalg.norm(result.ravel())) <= 1.0e-12:
        raise GlobalJumpError("registration frame has no windowed signal")
    return result


def integer_phase_shift(reference_zyx: Any, moving_zyx: Any) -> tuple[int, int, int]:
    import numpy as np

    reference = robust_windowed(reference_zyx)
    moving = robust_windowed(moving_zyx)
    cross_power = np.fft.rfftn(reference) * np.conjugate(np.fft.rfftn(moving))
    cross_power /= np.maximum(np.abs(cross_power), np.finfo(np.float64).eps)
    correlation = np.fft.irfftn(cross_power, s=reference.shape, axes=(0, 1, 2))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift: list[int] = []
    for index, size in zip(peak, reference.shape, strict=True):
        value = int(index)
        if value > size // 2:
            value -= size
        shift.append(value)
    return tuple(shift)  # type: ignore[return-value]


def registration_plan(array: Any) -> tuple[Any, set[int], dict[str, Any]]:
    import numpy as np

    if tuple(map(int, array.shape)) != SHAPE_TZYX or np.dtype(array.dtype) != np.dtype("uint16"):
        raise GlobalJumpError("raw array shape/dtype drifted")
    cumulative = np.zeros((100, 3), dtype=np.int64)
    previous = np.asarray(array[0, ::1, ::4, ::4])
    flagged: set[int] = set()
    frozen = 0
    shifts: list[dict[str, Any]] = []
    stride = np.asarray(STRIDE_ZYX, dtype=np.int64)
    registered_scale = np.asarray(REGISTERED_SCALE_ZYX_UM, dtype=np.float64)
    for frame in range(99):
        current = np.asarray(array[frame + 1, ::1, ::4, ::4])
        exact_frozen = bool(np.array_equal(previous, current))
        shift = (0, 0, 0) if exact_frozen else integer_phase_shift(previous, current)
        magnitude = float(np.linalg.norm(np.asarray(shift, dtype=np.float64) * registered_scale))
        pool_breaking = magnitude > POOL_RADIUS_UM
        cumulative[frame + 1] = cumulative[frame]
        if pool_breaking:
            cumulative[frame + 1] += np.asarray(shift, dtype=np.int64) * stride
            flagged.add(frame)
        frozen += int(exact_frozen)
        shifts.append({
            "frame": frame,
            "frozen": exact_frozen,
            "shift_registered_zyx": list(shift),
            "shift_magnitude_um": magnitude,
            "pool_breaking": pool_breaking,
        })
        previous = current
    return cumulative, flagged, {
        "transition_count": 99,
        "frozen_transition_count": frozen,
        "pool_breaking_transition_count": len(flagged),
        "pool_breaking_frames": sorted(flagged),
        "transitions": shifts,
    }


class RawArrayProvider:
    def __init__(self, input_root: Path):
        self.input_root = input_root.resolve()
        self._store: Any | None = None

    def inventory(self) -> set[str]:
        if self.input_root.is_file():
            if not zipfile.is_zipfile(self.input_root):
                raise GlobalJumpError("raw input file is not a ZIP archive")
            with zipfile.ZipFile(self.input_root, "r") as archive:
                result = set()
                for name in archive.namelist():
                    parts = name.split("/")
                    if (
                        len(parts) == 4 and parts[0] == "test"
                        and parts[1].endswith(".zarr") and parts[2:] == ["0", "zarr.json"]
                    ):
                        result.add(_safe_dataset(parts[1][:-5]))
                return result
        test_root = self.input_root / "test"
        if not test_root.is_dir():
            raise GlobalJumpError(f"raw input has no test directory: {self.input_root}")
        return {
            _safe_dataset(path.name[:-5])
            for path in test_root.iterdir()
            if path.is_dir() and path.name.endswith(".zarr")
        }

    def __call__(self, dataset: str) -> Any:
        import zarr

        if self.input_root.is_file():
            if self._store is None:
                self._store = zarr.storage.ZipStore(self.input_root, mode="r")
            return zarr.open_array(
                store=self._store, path=f"test/{dataset}.zarr/0", mode="r", zarr_format=3
            )
        return zarr.open_array(
            str(self.input_root / "test" / f"{dataset}.zarr" / "0"),
            mode="r", zarr_format=3,
        )

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


def _cache_identity(meta_path: Path, bh_model: Any) -> Any:
    payload = json.loads(meta_path.read_text(encoding="ascii"))
    value = payload.get("contract", {}).get("model")
    if not isinstance(value, Mapping):
        raise GlobalJumpError(f"cache model identity is absent: {meta_path}")
    return bh_model.CacheIdentity(**dict(value))


def translated_frames(cache: Any, cumulative: Any) -> tuple[Any, ...]:
    import numpy as np

    all_candidate = np.concatenate([
        np.asarray(frame.coords_zyx, dtype=np.int64)
        + cumulative[int(frame.frame_index)][None, :]
        for frame in cache.detections
    ])
    common_offset = np.maximum(-np.min(all_candidate, axis=0), 0).astype(np.int64)
    return tuple(
        replace(
            frame,
            coords_zyx=(
                np.asarray(frame.coords_zyx, dtype=np.int64)
                + cumulative[int(frame.frame_index)][None, :]
                + common_offset[None, :]
            ),
        )
        for frame in cache.detections
    )


def score_pairs(cache: Any, frames: Sequence[Any], scorer: Any) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    by_frame = {int(frame.frame_index): frame for frame in frames}
    probabilities: dict[tuple[int, int], float] = {}
    no_link: dict[int, float] = {}
    for context in cache.contexts:
        source = by_frame[int(context.source_frame)]
        target = by_frame[int(context.target_frame)]
        scored = scorer.score_pair(context, source, target)
        probabilities.update({
            (int(source_id), int(target_id)): float(probability)
            for source_id, target_id, probability in zip(
                scored.source_ids, scored.target_ids, scored.probabilities, strict=True
            )
        })
        no_link.update({
            int(source_id): float(probability)
            for source_id, probability in zip(
                scored.no_link_source_ids, scored.no_link_probabilities, strict=True
            )
        })
    return probabilities, no_link


def registered_local_edges(
    frames: Sequence[Any], probabilities: Mapping[tuple[int, int], float],
    no_link: Mapping[int, float], flagged: set[int], run_bh: Any, bh_graph: Any,
) -> dict[int, set[tuple[int, int]]]:
    by_frame = {int(frame.frame_index): frame for frame in frames}
    result: dict[int, set[tuple[int, int]]] = {}
    for frame in sorted(flagged):
        nodes = run_bh.detections_to_graph_nodes(
            (by_frame[frame], by_frame[frame + 1]), bh_graph.GraphNode
        )
        ids = {int(node.node_id) for node in nodes}
        local_probabilities = {
            pair: value for pair, value in probabilities.items()
            if pair[0] in ids and pair[1] in ids
        }
        local_no_link = {
            int(node_id): no_link[int(node_id)] for node_id in by_frame[frame].node_ids
        }
        edges, _stats = bh_graph.build_adjacent_edges(
            nodes,
            learned_edge_probabilities=local_probabilities,
            source_no_link_probabilities=local_no_link,
            topology_mode="legacy",
        )
        result[frame] = {(int(edge.source_id), int(edge.target_id)) for edge in edges}
    return result


def apply_registered(
    rows: Sequence[Mapping[str, str]], replacements_by_frame: Mapping[int, set[tuple[int, int]]],
    raw_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    edges = edge_pairs(rows)
    times = node_times(rows)
    retained_raw_ids = raw_ids & set(times)
    counts = Counter()
    for frame, replacements in sorted(replacements_by_frame.items()):
        raw_existing = {
            edge for edge in edges
            if edge[0] in retained_raw_ids and edge[1] in retained_raw_ids
            and times[edge[0]] == frame and times[edge[1]] == frame + 1
        }
        edges.difference_update(raw_existing)
        counts["removed_raw_edges"] += len(raw_existing)
        for source, target in sorted(replacements):
            if source not in times or target not in times:
                counts["skipped_pruned_pairs"] += 1
                continue
            conflicts = {
                edge for edge in edges
                if (edge[0] == source or edge[1] == target)
                and times[edge[0]] == frame and times[edge[1]] == frame + 1
            }
            counts["removed_conflicting_synthetic_edges"] += sum(
                edge[0] not in retained_raw_ids or edge[1] not in retained_raw_ids
                for edge in conflicts
            )
            edges.difference_update(conflicts)
            edges.add((source, target))
            counts["added_registered_edges"] += 1
    return rows_with_edges(rows, edges), dict(counts)


def postprocess(
    *, parent: Path, output: Path, input_root: Path, source_root: Path,
    support_root: Path, work_root: Path, receipt: Path | None,
    expected_output_sha256: str | None, deadline_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + float(deadline_seconds)
    if deadline_seconds <= 0:
        raise GlobalJumpError("deadline must be positive")
    parent = parent.resolve()
    output = output.resolve()
    source_root = source_root.resolve()
    support_root = support_root.resolve()
    work_root = work_root.resolve()
    if not parent.is_file() or output.exists() or parent == output:
        raise GlobalJumpError("parent must exist and output must be a distinct absent path")
    if not source_root.is_dir() or not support_root.is_dir():
        raise GlobalJumpError("materialized APNCC source/support is absent")
    for location in (source_root / "src", source_root / "scripts", support_root / "repo" / "src"):
        sys.path.insert(0, str(location))
    import run_bh
    from celltrack_alpha import bh_edge, bh_graph, bh_model

    edge_path = source_root / EDGE_RELATIVE
    auxiliary_path = source_root / AUXILIARY_RELATIVE
    if sha256_file(edge_path) != EDGE_SHA256 or sha256_file(auxiliary_path) != AUXILIARY_SHA256:
        raise GlobalJumpError("frozen edge scorer input identity drifted")
    parent_rows, parent_count = read_rows(parent)
    parent_totals = validate_rows(parent_rows)
    datasets = set(parent_rows)
    raw = RawArrayProvider(input_root)
    fragment_inventory = {
        path.stem.removeprefix("test__") for path in (work_root / "fragments").glob("test__*.csv")
    }
    cache_inventory = {
        path.stem.removeprefix("test__") for path in (work_root / "cache").glob("test__*.bhc")
    }
    if raw.inventory() != datasets or fragment_inventory != datasets or cache_inventory != datasets:
        raise GlobalJumpError(
            "fresh parent/raw/fragment/cache identity inventories do not match exactly"
        )

    model, bundle = bh_model.load_v45_model(support_root, device="cpu")
    scorer, scorer_record = run_bh.load_sparse_edge_scorer(
        edge_path, EDGE_SHA256, auxiliary_path, model, bundle, edge_module=bh_edge
    )
    if scorer is None or scorer_record is None:
        raise GlobalJumpError("frozen edge scorer did not restore")

    outputs: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    totals = Counter()
    try:
        for dataset in sorted(datasets):
            if time.monotonic() >= deadline:
                raise GlobalJumpError("global-jump postprocess deadline expired")
            slug = f"test__{dataset}"
            fragment_path = work_root / "fragments" / f"{slug}.csv"
            fragment_rows_by_dataset, _count = read_rows(fragment_path)
            if set(fragment_rows_by_dataset) != {dataset}:
                raise GlobalJumpError("fragment identity drifted")
            fragment_rows = fragment_rows_by_dataset[dataset]
            if (
                node_times(fragment_rows) != node_times(parent_rows[dataset])
                or edge_pairs(fragment_rows) != edge_pairs(parent_rows[dataset])
            ):
                raise GlobalJumpError("fresh parent and APNCC fragment graph disagree")
            meta_path = work_root / "cache" / f"{slug}.json"
            cache_path = work_root / "cache" / f"{slug}.bhc"
            cache = bh_model.load_sparse_cache(
                cache_path, expected_identity=_cache_identity(meta_path, bh_model)
            )
            cumulative, flagged, registration = registration_plan(raw(dataset))
            if flagged:
                registered = translated_frames(cache, cumulative)
                probabilities, no_link = score_pairs(cache, registered, scorer)
                replacements = registered_local_edges(
                    registered, probabilities, no_link, flagged, run_bh, bh_graph
                )
                raw_ids = {
                    int(node_id) for frame in cache.detections for node_id in frame.node_ids
                }
                transformed, counts = apply_registered(fragment_rows, replacements, raw_ids)
            else:
                transformed = [dict(row) for row in fragment_rows]
                counts = {}
            validate_rows({dataset: transformed})
            outputs[dataset] = transformed
            totals.update(counts)
            totals["pool_breaking_transitions"] += len(flagged)
            details.append({
                "identity": f"test:{dataset}",
                "fragment_sha256": sha256_file(fragment_path),
                "cache_sha256": sha256_file(cache_path),
                "cache_metadata_sha256": sha256_file(meta_path),
                "registration": registration,
                "graph_edit": counts,
            })
    finally:
        raw.close()

    output_count = write_rows(outputs, output)
    reread, reread_count = read_rows(output)
    final_totals = validate_rows(reread)
    if reread_count != output_count or set(reread) != datasets:
        raise GlobalJumpError("written output failed exact semantic round trip")
    output_sha = sha256_file(output)
    if expected_output_sha256 is not None and output_sha != expected_output_sha256:
        raise GlobalJumpError(
            f"public parity SHA mismatch: expected {expected_output_sha256}, got {output_sha}"
        )
    result = {
        "schema": "celltrack-alpha.g04-global-jump-hidden-postprocess.v1",
        "status": "PASS",
        "mechanism": {
            "name": "G1-frame-global-registration",
            "phase_correlation": "integer-3d-hann-q01-q99-median-centered",
            "pool_breaking_threshold_um": POOL_RADIUS_UM,
            "candidate_radius_um": 10.0,
            "candidate_cap": 8,
            "image_and_feature_sampling": "original",
            "candidate_geometry": "registered-once",
            "evaluator_coordinates": "original-unchanged",
        },
        "dataset_discovery": {
            "source": "fresh_parent_csv_plus_mounted_test_zarr_plus_same-run-cache",
            "datasets": sorted(datasets),
            "dataset_count": len(datasets),
            "inventory_exact_match": True,
            "public_identity_pinned": False,
        },
        "parent": {
            "path": str(parent), "sha256": sha256_file(parent),
            "bytes": parent.stat().st_size, "rows": parent_count, **parent_totals,
        },
        "output": {
            "path": str(output), "sha256": output_sha,
            "bytes": output.stat().st_size, "rows": output_count, **final_totals,
        },
        "transform": {
            **dict(totals),
            "labels_read": False,
            "fresh_submission_generation": True,
            "public_only_output_copy": False,
            "observation_distinct_from_parent": output_sha != sha256_file(parent),
        },
        "scorer": scorer_record,
        "per_dataset": details,
        "elapsed_seconds": time.monotonic() - started,
    }
    if receipt is not None:
        atomic_bytes(receipt.resolve(), canonical_bytes(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--deadline-seconds", type=float, default=3600.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = postprocess(
        parent=args.parent, output=args.output, input_root=args.input_root,
        source_root=args.source_root, support_root=args.support_root,
        work_root=args.work_root, receipt=args.receipt,
        expected_output_sha256=args.expected_output_sha256,
        deadline_seconds=args.deadline_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
