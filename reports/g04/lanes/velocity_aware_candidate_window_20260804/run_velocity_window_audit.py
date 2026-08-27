#!/usr/bin/env python3
"""Audit the fixed-r10 VELOCITY-AWARE-CANDIDATE-WINDOW-v1 ceiling.

The post-G1 APNCC fragment bank supplies the candidate topology and selected
prior edges.  It does not import, instantiate, or call the edge scorer or
graph solver.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[4]
for _location in (ROOT / "src", ROOT / "scripts", ROOT / "external/Tracking_Support/repo/src"):
    if str(_location) not in sys.path:
        sys.path.insert(0, str(_location))

from celltrack_alpha.bh_graph import PHYSICAL_SCALE_ZYX, sparse_min_cost_max_cardinality  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from celltrack_alpha.frame_registration import load_registration_plan  # noqa: E402


SCHEMA = "celltrack-alpha.velocity-aware-candidate-window-audit.v1"
RADIUS_UM = 10.0
CAP = 8
PERSISTENCE = 0.30
PUBLISHED_MEDIAN_STEP_UM = 1.86
MATCH_RADIUS_UM = 7.0
EXPECTED_SHELL_EDGES = 366
EXPECTED_REPORTED_G1_CACHE_CANDIDATES = 25_553
EXPECTED_G1_APNCC_FLAGGED_CANDIDATES = 26_497
G1_COUNTS = (34_389, 2_308, 1_879)
CSV_FIELDS = (
    "identity",
    "family",
    "transition_t",
    "source_id",
    "target_id",
    "global_corrected_displacement_z_um",
    "global_corrected_displacement_y_um",
    "global_corrected_displacement_x_um",
    "global_corrected_distance_um",
    "source_previous_available",
    "source_previous_id",
    "prior_velocity_z_um",
    "prior_velocity_y_um",
    "prior_velocity_x_um",
    "prior_velocity_distance_um",
    "prior_velocity_minus_published_median_um",
    "lag1_direction_cosine",
    "constant_velocity_residual_um",
    "persistence_prediction_residual_um",
    "source_local_density_r10",
    "target_local_density_r10",
    "global_g1_candidate",
    "persistence_prediction_distance_um",
    "persistence_prediction_inside_r10",
    "persistence_prediction_rank_inside_r10",
    "velocity_window_candidate_after_cap8",
)


class AuditError(RuntimeError):
    """Raised if a frozen input or audit invariant drifts."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AuditError(f"JSON object required: {path}")
    return value


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="ascii") as handle:
        handle.write(canonical_json(value).decode("ascii"))
        handle.write("\n")


def read_fragment(path: Path) -> tuple[str, dict[int, tuple[np.ndarray, np.ndarray]], list[tuple[int, int]]]:
    """Read original evaluator coordinates and selected G1 predecessor links."""

    nodes: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    edges: list[tuple[int, int]] = []
    dataset: str | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = (
            "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise AuditError(f"fragment schema drifted: {path}")
        for row in reader:
            value = str(row["dataset"])
            if dataset is None:
                dataset = value
            elif dataset != value:
                raise AuditError(f"mixed dataset fragment: {path}")
            if row["row_type"] == "node":
                t = int(row["t"])
                nodes[t].append(
                    (int(row["node_id"]), np.asarray((row["z"], row["y"], row["x"]), dtype=np.float64))
                )
            elif row["row_type"] == "edge":
                edges.append((int(row["source_id"]), int(row["target_id"])))
            else:
                raise AuditError(f"unknown fragment row type: {row['row_type']!r}")
    if dataset is None or set(nodes) != set(range(100)):
        raise AuditError(f"fragment must contain all 100 frames: {path}")
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_ids: set[int] = set()
    for t, values in nodes.items():
        ids = np.asarray([row[0] for row in values], dtype=np.int64)
        coordinates = np.asarray([row[1] for row in values], dtype=np.float64)
        if len(ids) == 0 or len(ids) != len(set(map(int, ids))) or any(int(node) in all_ids for node in ids):
            raise AuditError(f"node identity drift in {path}")
        all_ids.update(map(int, ids))
        result[t] = (ids, coordinates)
    if len(edges) != len(set(edges)):
        raise AuditError(f"duplicate selected edge in {path}")
    if any(source not in all_ids or target not in all_ids for source, target in edges):
        raise AuditError(f"edge references unknown node in {path}")
    return dataset, result, sorted(edges)


def physical_frames(
    original: Mapping[int, tuple[np.ndarray, np.ndarray]], cumulative_raw_zyx: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    scale = np.asarray(PHYSICAL_SCALE_ZYX, dtype=np.float64)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t, (ids, coords) in original.items():
        shifted = (coords + cumulative_raw_zyx[t][None, :]) * scale[None, :]
        if shifted.shape != coords.shape or not np.isfinite(shifted).all():
            raise AuditError("nonfinite registered physical coordinates")
        result[t] = (ids, shifted)
    return result


def local_density(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    tree = cKDTree(points)
    result = np.asarray(tree.query_ball_point(points, r=RADIUS_UM, return_length=True), dtype=np.int64) - 1
    if np.any(result < 0):
        raise AuditError("local density cannot be negative")
    return result


def deterministic_candidates(
    source_ids: np.ndarray,
    centres: np.ndarray,
    target_ids: np.ndarray,
    target_points: np.ndarray,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], int]:
    """Exact r10/cap8 cKDTree gate, ordered by predicted-centre distance."""

    if centres.shape != (len(source_ids), 3):
        raise AuditError("candidate-centre shape drifted")
    tree = cKDTree(target_points) if len(target_points) else None
    pairs: set[tuple[int, int]] = set()
    ranks: dict[tuple[int, int], int] = {}
    peak = 0
    if tree is None:
        return pairs, ranks, peak
    for source_row, centre in enumerate(centres):
        nearby = [int(value) for value in tree.query_ball_point(centre, r=RADIUS_UM)]
        peak = max(peak, len(nearby))
        nearby.sort(
            key=lambda target_row: (
                float(np.linalg.norm(target_points[target_row] - centre)),
                int(target_ids[target_row]),
                target_row,
            )
        )
        source_id = int(source_ids[source_row])
        for rank, target_row in enumerate(nearby, start=1):
            pair = (source_id, int(target_ids[target_row]))
            ranks[pair] = rank
            if rank <= CAP:
                if pair in pairs:
                    raise AuditError("candidate pair repeats")
                pairs.add(pair)
    return pairs, ranks, peak


def gt_matches(
    frames: Mapping[int, tuple[np.ndarray, np.ndarray]], gt: Mapping[str, Any]
) -> tuple[dict[int, int | None], dict[int, set[tuple[int, int]]]]:
    """Reproduce the frozen per-frame 7-um detection-to-GT matching contract."""

    nodes = gt.get("nodes")
    if not isinstance(nodes, Mapping):
        raise AuditError("GT nodes are missing")
    gt_ids = np.asarray(nodes["node_id"], dtype=np.int64)
    gt_times = np.asarray(nodes["t"], dtype=np.int64)
    gt_raw = np.column_stack((nodes["z"], nodes["y"], nodes["x"])).astype(np.float64, copy=False)
    if gt_raw.shape != (len(gt_ids), 3):
        raise AuditError("GT node table shape drifted")
    scale = np.asarray(PHYSICAL_SCALE_ZYX, dtype=np.float64)
    gt_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t in range(100):
        mask = gt_times == t
        gt_by_frame[t] = (gt_ids[mask], gt_raw[mask] * scale[None, :])
    matches: dict[int, int | None] = {}
    for t, (detection_ids, detection_points) in frames.items():
        ids, points = gt_by_frame[t]
        matches.update({int(value): None for value in detection_ids})
        if len(detection_ids) == 0 or len(ids) == 0:
            continue
        tree = cKDTree(points)
        feasible: list[tuple[int, int, float]] = []
        for row, targets in enumerate(tree.query_ball_point(detection_points, r=MATCH_RADIUS_UM)):
            for target_row in sorted(map(int, targets)):
                distance = float(np.linalg.norm(detection_points[row] - points[target_row]))
                if distance <= MATCH_RADIUS_UM:
                    feasible.append((int(detection_ids[row]), int(ids[target_row]), distance))
        if feasible:
            solved = sparse_min_cost_max_cardinality(
                list(map(int, detection_ids)), list(map(int, ids)), feasible
            )
            for row in solved.matches:
                matches[int(row.source_id)] = int(row.target_id)
    gt_time = {int(node): int(t) for node, t in zip(gt_ids, gt_times, strict=True)}
    positive_by_frame: dict[int, set[tuple[int, int]]] = {t: set() for t in range(99)}
    detection_for_gt = {gt_id: det_id for det_id, gt_id in matches.items() if gt_id is not None}
    for source_gt, target_gt in np.asarray(gt["edges"], dtype=np.int64):
        source_gt_int = int(source_gt)
        target_gt_int = int(target_gt)
        t = gt_time.get(source_gt_int)
        if t is None or gt_time.get(target_gt_int) != t + 1:
            continue
        source_det = detection_for_gt.get(source_gt_int)
        target_det = detection_for_gt.get(target_gt_int)
        if source_det is not None and target_det is not None:
            positive_by_frame[t].add((source_det, target_det))
    return matches, positive_by_frame


def _finite_or_blank(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def summary(values: Iterable[float]) -> Mapping[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"n": 0}
    if not np.isfinite(array).all():
        raise AuditError("summary received nonfinite values")
    return {
        "n": int(len(array)),
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def family(identity: str) -> str:
    try:
        return identity.split(":", 1)[1].split("_", 1)[0]
    except IndexError as exc:
        raise AuditError(f"invalid identity {identity!r}") from exc


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output
    audit_path = output / "AUDIT.json"
    csv_path = output / "SHELL_EDGES.csv"
    manifest_path = output / "RUN_MANIFEST.json"
    if args.partial:
        partial_path = output / f"PARTIAL_V3_{args.start:02d}_{args.stop:02d}.json"
        if partial_path.exists():
            raise AuditError(f"refusing to overwrite partial audit {partial_path}")
    elif any(path.exists() for path in (audit_path, csv_path, manifest_path)):
        raise AuditError("refusing to overwrite a completed audit")
    started = time.perf_counter()

    source_report = load_json(args.g1_report)
    all_selected = tuple(str(value) for value in source_report["selection"]["identities"])
    base_report = load_json(args.base_report)
    if all_selected != tuple(str(value) for value in base_report["selection"]["video_ids"]):
        raise AuditError("G1 fragment selection differs from canonical APNCC CV57")
    if len(all_selected) != 57 or len(set(all_selected)) != 57:
        raise AuditError("canonical CV57 selection is malformed")
    if args.start < 0 or args.stop <= args.start or args.stop > len(all_selected):
        raise AuditError("partial selection bounds are invalid")
    selected = all_selected[args.start:args.stop]
    scores = source_report.get("scores", {}).get("registered_g1", {}).get("summary", {})
    observed_control = (
        float(scores.get("adj_edge_jaccard", math.nan)),
        float(scores.get("edge_jaccard", math.nan)),
        float(scores.get("node_recall", math.nan)),
    )
    expected_control = (0.8867644202595942, 0.891461012028204, 0.990377056057473)
    if observed_control != expected_control:
        raise AuditError(f"G1 control score drifted: {observed_control!r}")
    report_count = int(source_report["candidate_reachability_pool_breaking_cv57"]["candidates_after"])
    if report_count != EXPECTED_REPORTED_G1_CACHE_CANDIDATES:
        raise AuditError("G1 source report's flagged candidate count drifted")

    census = load_json(args.census)
    census_rows = {
        f"{row['split']}:{row['dataset']}": row for row in census.get("videos", [])
    }
    if not set(all_selected).issubset(census_rows):
        raise AuditError("G1 census lacks one or more CV57 identities")
    archive_identity = dict(census.get("archive", {}))
    if archive_identity != {
        "bytes": 87393127165,
        "central_directory_sha256": "055b38ffa73594321f857e5bde7120a7b0236fbeee661f18214f4d910c8b00be",
        "path": "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    }:
        raise AuditError("census raw-archive identity drifted")

    totals: Counter[str] = Counter()
    per_family: dict[str, Counter[str]] = {"44b6": Counter(), "6bba": Counter()}
    shell_rows: list[dict[str, str]] = []
    shell_distances: list[float] = []
    shell_prior_speeds: list[float] = []
    shell_prediction_residuals: list[float] = []
    shell_constant_residuals: list[float] = []
    shell_direction_cosines: list[float] = []
    flagged_baseline_candidates = 0
    fallback_equivalence_failures = 0
    peak_g1_neighbors = peak_velocity_neighbors = 0

    store = ZipStore(args.zip, mode="r")
    try:
        for ordinal, identity in enumerate(selected, start=1):
            slug = identity.replace(":", "__", 1)
            fragment = args.fragments / f"{slug}.csv"
            if not fragment.is_file():
                raise AuditError(f"missing G1 fragment {fragment}")
            dataset, original, selected_edges = read_fragment(fragment)
            if identity != f"train:{dataset}":
                raise AuditError(f"fragment identity drifted: {fragment}")
            plan = load_registration_plan(args.census, identity)
            physical = physical_frames(original, np.asarray(plan.cumulative_shift_raw_zyx, dtype=np.float64))
            node_time = {int(node): t for t, (ids, _points) in physical.items() for node in ids}
            parent_by_target: dict[int, int] = {}
            for parent, child in selected_edges:
                if node_time[parent] + 1 != node_time[child]:
                    continue
                if child in parent_by_target:
                    raise AuditError("G1 selected graph has a same-frame duplicate in-edge")
                parent_by_target[child] = parent
            point_by_id = {
                int(node): point for _t, (ids, points) in physical.items() for node, point in zip(ids, points, strict=True)
            }
            gt = read_geff_arrays(store, "train", dataset)
            # Detection-to-GT matching is in raw evaluator coordinates. Only
            # candidate distances use G1-transformed physical coordinates.
            raw_physical = {
                t: (ids, coords * np.asarray(PHYSICAL_SCALE_ZYX, dtype=np.float64)[None, :])
                for t, (ids, coords) in original.items()
            }
            _matches, positives = gt_matches(raw_physical, gt)
            f = family(identity)

            for t in range(99):
                source_ids, source_points = physical[t]
                target_ids, target_points = physical[t + 1]
                baseline_pairs, baseline_ranks, peak_g1 = deterministic_candidates(
                    source_ids, source_points, target_ids, target_points
                )
                peak_g1_neighbors = max(peak_g1_neighbors, peak_g1)
                source_has_prior = np.zeros(len(source_ids), dtype=bool)
                centres = np.array(source_points, dtype=np.float64, copy=True)
                previous: list[np.ndarray | None] = [None] * len(source_ids)
                previous_id: list[int | None] = [None] * len(source_ids)
                for row, source in enumerate(source_ids):
                    parent = parent_by_target.get(int(source))
                    if parent is None or node_time.get(parent) != t - 1:
                        continue
                    velocity = source_points[row] - point_by_id[parent]
                    centres[row] += PERSISTENCE * velocity
                    source_has_prior[row] = True
                    previous[row] = velocity
                    previous_id[row] = parent
                velocity_pairs, velocity_ranks, peak_velocity = deterministic_candidates(
                    source_ids, centres, target_ids, target_points
                )
                peak_velocity_neighbors = max(peak_velocity_neighbors, peak_velocity)
                source_density = local_density(source_points)
                target_density = local_density(target_points)
                target_row = {int(node): row for row, node in enumerate(target_ids)}
                source_row = {int(node): row for row, node in enumerate(source_ids)}
                if not np.array_equal(centres[~source_has_prior], source_points[~source_has_prior]):
                    raise AuditError("fallback centre drifted")
                for source in source_ids[~source_has_prior]:
                    source_int = int(source)
                    baseline_for_source = {pair for pair in baseline_pairs if pair[0] == source_int}
                    velocity_for_source = {pair for pair in velocity_pairs if pair[0] == source_int}
                    if baseline_for_source != velocity_for_source:
                        fallback_equivalence_failures += 1
                positive = positives[t]
                g1_positive = baseline_pairs & positive
                velocity_positive = velocity_pairs & positive
                new_positive = velocity_positive - baseline_pairs
                lost_positive = g1_positive - velocity_pairs
                totals["g1_candidates"] += len(baseline_pairs)
                totals["velocity_candidates"] += len(velocity_pairs)
                totals["g1_positive_candidates"] += len(g1_positive)
                totals["velocity_positive_candidates"] += len(velocity_positive)
                totals["newly_reachable_true_continuations"] += len(new_positive)
                totals["lost_true_continuations"] += len(lost_positive)
                totals["g1_background_candidates"] += len(baseline_pairs - positive)
                totals["velocity_background_candidates"] += len(velocity_pairs - positive)
                totals["sources_with_prior"] += int(np.sum(source_has_prior))
                totals["sources_without_prior"] += int(np.sum(~source_has_prior))
                totals["velocity_candidates_from_prior_sources"] += sum(
                    1 for source, _target in velocity_pairs if source_has_prior[source_row[source]]
                )
                totals["velocity_candidates_from_fallback_sources"] += sum(
                    1 for source, _target in velocity_pairs if not source_has_prior[source_row[source]]
                )
                per_family[f].update({
                    "g1_candidates": len(baseline_pairs),
                    "velocity_candidates": len(velocity_pairs),
                    "g1_positive_candidates": len(g1_positive),
                    "velocity_positive_candidates": len(velocity_positive),
                    "newly_reachable_true_continuations": len(new_positive),
                    "lost_true_continuations": len(lost_positive),
                    "g1_background_candidates": len(baseline_pairs - positive),
                    "velocity_background_candidates": len(velocity_pairs - positive),
                })
                if bool(census_rows[identity]["path"]):
                    video_path = args.census.parent / str(census_rows[identity]["path"])
                    transition = load_json(video_path)["transitions"][t]
                    if bool(transition["pool_breaking"]):
                        flagged_baseline_candidates += len(baseline_pairs)
                for source, target in sorted(positive):
                    source_index = source_row[source]
                    target_index = target_row[target]
                    displacement = target_points[target_index] - source_points[source_index]
                    distance = float(np.linalg.norm(displacement))
                    if not (RADIUS_UM < distance <= 12.0):
                        continue
                    velocity = previous[source_index]
                    if velocity is None:
                        prior_speed: float | None = None
                        direction: float | None = None
                        constant_residual: float | None = None
                    else:
                        prior_speed = float(np.linalg.norm(velocity))
                        direction = _cosine(velocity, displacement)
                        constant_residual = float(np.linalg.norm(displacement - velocity))
                    prediction_distance = float(np.linalg.norm(target_points[target_index] - centres[source_index]))
                    prediction_residual = prediction_distance if velocity is not None else None
                    inside = prediction_distance <= RADIUS_UM
                    rank = velocity_ranks.get((source, target))
                    row = {
                        "identity": identity,
                        "family": f,
                        "transition_t": str(t),
                        "source_id": str(source),
                        "target_id": str(target),
                        "global_corrected_displacement_z_um": _finite_or_blank(float(displacement[0])),
                        "global_corrected_displacement_y_um": _finite_or_blank(float(displacement[1])),
                        "global_corrected_displacement_x_um": _finite_or_blank(float(displacement[2])),
                        "global_corrected_distance_um": _finite_or_blank(distance),
                        "source_previous_available": str(int(velocity is not None)),
                        "source_previous_id": "" if previous_id[source_index] is None else str(previous_id[source_index]),
                        "prior_velocity_z_um": "" if velocity is None else _finite_or_blank(float(velocity[0])),
                        "prior_velocity_y_um": "" if velocity is None else _finite_or_blank(float(velocity[1])),
                        "prior_velocity_x_um": "" if velocity is None else _finite_or_blank(float(velocity[2])),
                        "prior_velocity_distance_um": _finite_or_blank(prior_speed),
                        "prior_velocity_minus_published_median_um": _finite_or_blank(
                            None if prior_speed is None else prior_speed - PUBLISHED_MEDIAN_STEP_UM
                        ),
                        "lag1_direction_cosine": _finite_or_blank(direction),
                        "constant_velocity_residual_um": _finite_or_blank(constant_residual),
                        "persistence_prediction_residual_um": _finite_or_blank(prediction_residual),
                        "source_local_density_r10": str(int(source_density[source_index])),
                        "target_local_density_r10": str(int(target_density[target_index])),
                        "global_g1_candidate": str(int((source, target) in baseline_pairs)),
                        "persistence_prediction_distance_um": _finite_or_blank(prediction_distance),
                        "persistence_prediction_inside_r10": str(int(inside)),
                        "persistence_prediction_rank_inside_r10": "" if rank is None else str(rank),
                        "velocity_window_candidate_after_cap8": str(int((source, target) in velocity_pairs)),
                    }
                    shell_rows.append(row)
                    shell_distances.append(distance)
                    if prior_speed is not None:
                        shell_prior_speeds.append(prior_speed)
                        shell_prediction_residuals.append(prediction_distance)
                        if constant_residual is not None:
                            shell_constant_residuals.append(constant_residual)
                        if direction is not None:
                            shell_direction_cosines.append(direction)
            print(f"audit {args.start + ordinal:02d}/57 {identity}", flush=True)
    finally:
        store.close()

    if args.partial:
        partial = {
            "schema": "celltrack-alpha.velocity-aware-candidate-window-partial.v1",
            "selection": list(selected),
            "selection_bounds": {"start": args.start, "stop": args.stop, "total": len(all_selected)},
            "totals": dict(totals),
            "per_family": {name: dict(values) for name, values in per_family.items()},
            "shell_rows": shell_rows,
            "shell_distances": shell_distances,
            "shell_prior_speeds": shell_prior_speeds,
            "shell_prediction_residuals": shell_prediction_residuals,
            "shell_constant_residuals": shell_constant_residuals,
            "shell_direction_cosines": shell_direction_cosines,
            "flagged_baseline_candidates": flagged_baseline_candidates,
            "fallback_equivalence_failures": fallback_equivalence_failures,
            "peak_g1_neighbors": peak_g1_neighbors,
            "peak_velocity_neighbors": peak_velocity_neighbors,
            "input_hashes": {
                "g1_report": sha256(args.g1_report),
                "census": sha256(args.census),
                "base_report": sha256(args.base_report),
                "archive_identity": archive_identity,
            },
            "execution": {"elapsed_seconds": time.perf_counter() - started, "scorer_or_graph_called": False},
        }
        write_new_json(partial_path, partial)
        return partial

    if flagged_baseline_candidates != EXPECTED_G1_APNCC_FLAGGED_CANDIDATES:
        raise AuditError(
            "G1 APNCC fragment/census candidate replay mismatch: "
            f"{flagged_baseline_candidates} != {EXPECTED_G1_APNCC_FLAGGED_CANDIDATES}"
        )
    if len(shell_rows) != EXPECTED_SHELL_EDGES:
        raise AuditError(f"residual G1 shell count drifted: {len(shell_rows)} != {EXPECTED_SHELL_EDGES}")
    if any(row["global_g1_candidate"] != "0" for row in shell_rows):
        raise AuditError("a 10--12 um shell edge is unexpectedly inside G1 r10/cap8")
    if fallback_equivalence_failures:
        raise AuditError(f"fallback altered {fallback_equivalence_failures} source candidate sets")

    shell_counter = Counter()
    for row in shell_rows:
        shell_counter["total"] += 1
        shell_counter["previous_available"] += int(row["source_previous_available"])
        shell_counter["predicted_inside_r10"] += int(row["persistence_prediction_inside_r10"])
        shell_counter["candidate_after_cap8"] += int(row["velocity_window_candidate_after_cap8"])
        if row["persistence_prediction_inside_r10"] == "1" and row["velocity_window_candidate_after_cap8"] == "0":
            shell_counter["inside_but_cap8_excluded"] += 1
    tp, fp, fn = G1_COUNTS
    newly = int(totals["newly_reachable_true_continuations"])
    raw_ceiling = newly / float(tp + fp + fn)
    burden_ratio = (
        float(totals["velocity_background_candidates"]) / float(totals["g1_background_candidates"])
        if totals["g1_background_candidates"]
        else math.inf
    )
    gate = {
        "newly_reachable_true_continuations": {
            "observed": newly,
            "minimum": 20,
            "passes": newly >= 20,
        },
        "no_new_fp_raw_edge_ceiling": {
            "formula": "N / (34389 + 2308 + 1879)",
            "observed": raw_ceiling,
            "minimum": 0.0005,
            "passes": raw_ceiling >= 0.0005,
        },
        "background_burden_multiplier": {
            "observed": burden_ratio,
            "maximum": 1.05,
            "passes": burden_ratio <= 1.05,
        },
        "shell_retained_after_cap8": {
            "observed": int(shell_counter["candidate_after_cap8"]),
            "minimum": 20,
            "passes": int(shell_counter["candidate_after_cap8"]) >= 20,
        },
    }
    score_authorized = all(bool(row["passes"]) for row in gate.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETED_NO_SCORE" if not score_authorized else "CEILING_PASSES_SCORE_ELIGIBILITY",
        "mechanism": {
            "name": "VELOCITY-AWARE-CANDIDATE-WINDOW-v1",
            "new_named_mechanism": True,
            "not_radius_widening": True,
            "not_dead_shell_rescreen": True,
            "radius_um": RADIUS_UM,
            "neighbor_cap": CAP,
            "published_median_step_um_per_frame": PUBLISHED_MEDIAN_STEP_UM,
            "lag1_directional_persistence": PERSISTENCE,
            "formula": "qhat[i,t+1] = q[i,t] + 0.30 * (q[i,t] - q[parent,t-1])",
            "fallback": "qhat[i,t+1] = q[i,t] when no selected G1 parent at t-1 exists",
            "coordinate_convention": {
                "candidate_geometry": "(original_voxel_coordinate + G1_cumulative_raw_shift) * physical_scale",
                "physical_scale_zyx_um": list(PHYSICAL_SCALE_ZYX),
                "output_and_evaluator_coordinates": "original APNCC fragment coordinates; unchanged",
                "image_and_feature_sampling": "original; untouched",
                "global_transform_applications": 1,
            },
        },
        "control": {
            "name": "global-jump-corrected APNCC",
            "adjusted_edge_jaccard": observed_control[0],
            "raw_edge_jaccard": observed_control[1],
            "node_recall": observed_control[2],
            "edge_tp_fp_fn": {"tp": tp, "fp": fp, "fn": fn},
            "fragment_aggregate_sha256": source_report["scores"]["registered_g1"]["aggregate_sha256"],
        },
        "residual_shell": {
            "definition": "GT-matched adjacent continuation with global-jump-corrected physical distance in (10,12] um",
            "edge_count": len(shell_rows),
            "by_family": dict(Counter(row["family"] for row in shell_rows)),
            "prior_velocity_available": int(shell_counter["previous_available"]),
            "persistence_prediction_inside_r10": int(shell_counter["predicted_inside_r10"]),
            "retained_after_fixed_cap8": int(shell_counter["candidate_after_cap8"]),
            "inside_r10_but_cap8_excluded": int(shell_counter["inside_but_cap8_excluded"]),
            "global_corrected_distance_um": summary(shell_distances),
            "prior_speed_um_per_frame": summary(shell_prior_speeds),
            "lag1_direction_cosine": summary(shell_direction_cosines),
            "constant_velocity_residual_um": summary(shell_constant_residuals),
            "persistence_prediction_residual_um": summary(shell_prediction_residuals),
        },
        "candidate_pool": {
            "global_jump_apncc": {
                "candidate_pairs": int(totals["g1_candidates"]),
                "positive_candidate_pairs": int(totals["g1_positive_candidates"]),
                "nonpositive_candidate_pairs": int(totals["g1_background_candidates"]),
            },
            "velocity_aware_window": {
                "candidate_pairs": int(totals["velocity_candidates"]),
                "positive_candidate_pairs": int(totals["velocity_positive_candidates"]),
                "nonpositive_candidate_pairs": int(totals["velocity_background_candidates"]),
                "candidate_pairs_from_prior_sources": int(totals["velocity_candidates_from_prior_sources"]),
                "candidate_pairs_from_fallback_sources": int(totals["velocity_candidates_from_fallback_sources"]),
            },
            "delta_velocity_minus_g1": {
                "candidate_pairs": int(totals["velocity_candidates"] - totals["g1_candidates"]),
                "positive_candidate_pairs": int(totals["velocity_positive_candidates"] - totals["g1_positive_candidates"]),
                "nonpositive_candidate_pairs": int(totals["velocity_background_candidates"] - totals["g1_background_candidates"]),
                "newly_reachable_true_continuations": newly,
                "lost_true_continuations": int(totals["lost_true_continuations"]),
            },
            "by_family": {name: dict(values) for name, values in per_family.items()},
            "all_source_queries": {
                "with_lag1_parent": int(totals["sources_with_prior"]),
                "fallback_to_g1_center": int(totals["sources_without_prior"]),
                "fallback_candidate_set_mismatches": fallback_equivalence_failures,
            },
            "peak_uncapped_neighbors": {
                "global_jump_apncc": peak_g1_neighbors,
                "velocity_aware_window": peak_velocity_neighbors,
            },
        },
        "pre_scorer_gate": {
            "gate": gate,
            "verdict": "SCORE_AUTHORIZED" if score_authorized else "NO_SCORE",
            "reason": "all pre-registered ceiling gates pass" if score_authorized else "at least one pre-registered ceiling gate fails",
        },
        "validation": {
            "canonical_cv57_identities": len(selected),
            "fragment_count": len(list(args.fragments.glob("train__*.csv"))),
            "g1_pool_breaking_candidate_replay": {
                "observed": flagged_baseline_candidates,
                "expected": EXPECTED_G1_APNCC_FLAGGED_CANDIDATES,
                "passes": True,
            },
            "residual_shell_edge_count": {"observed": len(shell_rows), "expected": EXPECTED_SHELL_EDGES, "passes": True},
            "shell_g1_r10_exclusion": True,
            "fallback_exact_equivalence": True,
            "scorer_or_graph_called": False,
        },
        "provenance": {
            "g1_report": {"path": str(args.g1_report), "sha256": sha256(args.g1_report)},
            "g1_fragments_directory": str(args.fragments),
            "census": {"path": str(args.census), "sha256": sha256(args.census)},
            "base_report": {"path": str(args.base_report), "sha256": sha256(args.base_report)},
            "archive": archive_identity,
            "audit_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "preregistration": {"path": str(output / "PREREGISTRATION.md"), "sha256": sha256(output / "PREREGISTRATION.md")},
        },
        "execution": {
            "single_thread_environment_required": ["OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"],
            "elapsed_seconds": time.perf_counter() - started,
            "scorer_called": False,
            "graph_called": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
        },
    }
    with csv_path.open("x", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(shell_rows)
    write_new_json(audit_path, result)
    manifest = {
        "schema": "celltrack-alpha.velocity-aware-candidate-window-run-manifest.v1",
        "audit_sha256": sha256(audit_path),
        "shell_edges_csv_sha256": sha256(csv_path),
        "preregistration_sha256": sha256(output / "PREREGISTRATION.md"),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv313/bin/python run_velocity_window_audit.py",
    }
    write_new_json(manifest_path, manifest)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    value.add_argument(
        "--fragments", type=Path,
        default=ROOT / "reports/g04/lanes/data_pathology_repair/score/registered_g1/fragments",
    )
    value.add_argument(
        "--g1-report", type=Path,
        default=ROOT / "reports/g04/lanes/data_pathology_repair/score/REPORT.json",
    )
    value.add_argument(
        "--census", type=Path,
        default=ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json",
    )
    value.add_argument(
        "--base-report", type=Path,
        default=ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/report.json",
    )
    value.add_argument(
        "--zip", type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    value.add_argument("--partial", action="store_true", help="write a bounded CPU partial instead of the final audit")
    value.add_argument("--start", type=int, default=0, help="inclusive CV57 ordinal for a partial")
    value.add_argument("--stop", type=int, default=57, help="exclusive CV57 ordinal for a partial")
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    if "status" in report:
        print(json.dumps({"status": report["status"], "gate": report["pre_scorer_gate"]}, sort_keys=True))
    else:
        print(json.dumps({"status": "PARTIAL_COMPLETED", "selection_bounds": report["selection_bounds"]}, sort_keys=True))
