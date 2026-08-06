#!/usr/bin/env python3
"""Measure fixed-r10/cap8 motion recentering on the exact fit117 G1 graph."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "reports/g04/lanes/velocity_aware_candidate_window_20260804"
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    LANE,
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import g04_score_pathology_repairs as repair  # noqa: E402
import run_velocity_window_audit as audit  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402


DEFAULT_COHORT = LANE / "fit117_g1_selected_v1/SEALED_COHORT.json"
DEFAULT_OUTPUT = LANE / "fit117_recenter_v1"
DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
FIELDS = (
    "identity", "family", "transition_t", "source_id", "target_id",
    "source_in_g1_graph", "target_in_g1_graph", "pair_in_g1_graph_candidate_universe",
    "source_previous_available", "source_previous_id",
    "prior_velocity_z_um", "prior_velocity_y_um", "prior_velocity_x_um", "prior_velocity_distance_um",
    "g1_cumulative_shift_previous_z_vox", "g1_cumulative_shift_previous_y_vox", "g1_cumulative_shift_previous_x_vox",
    "g1_cumulative_shift_source_z_vox", "g1_cumulative_shift_source_y_vox", "g1_cumulative_shift_source_x_vox",
    "g1_cumulative_shift_target_z_vox", "g1_cumulative_shift_target_y_vox", "g1_cumulative_shift_target_x_vox",
    "original_shell_distance_um", "global_corrected_distance_um",
    "persistence_prediction_distance_um", "persistence_prediction_inside_r10",
    "persistence_prediction_rank_inside_r10", "velocity_window_candidate_after_cap8",
    "unreachable_reason",
)


class CeilingError(RuntimeError):
    """The exact fit117 cohort or conservation contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise CeilingError(f"refusing to overwrite {path}")
    temporary.write_bytes(canonical_bytes(dict(value)))
    os.replace(temporary, path)


def scalar(value: float | None) -> str:
    return "" if value is None else format(float(value), ".12g")


def vector_fields(prefix: str, value: np.ndarray | None, suffix: str) -> dict[str, str]:
    axes = ("z", "y", "x")
    return {
        f"{prefix}_{axis}_{suffix}": "" if value is None else scalar(float(value[index]))
        for index, axis in enumerate(axes)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CeilingError(f"refusing to overwrite output: {args.output}")
    started = time.perf_counter()
    cohort = repair.load_json(args.cohort)
    selected = tuple(map(str, cohort.get("selection", {}).get("identities", [])))
    if (
        cohort.get("status") != "PASS_SEALED_EXACT_FIT117_G1_SELECTED_GRAPH"
        or len(selected) != 117
        or len(set(selected)) != 117
        or cohort.get("contract", {}).get("cv57_substitution") is not False
        or cohort.get("contract", {}).get("conservation_guard_relaxed") is not False
        or int(cohort.get("contract", {}).get("shell_positive_count", -1)) != 456
        or int(cohort.get("contract", {}).get("expected_post_g1_residual_shell", -1)) != 366
    ):
        raise CeilingError("sealed cohort contract drifted")
    records = {str(row["identity"]): row for row in cohort["videos"]}
    if set(records) != set(selected):
        raise CeilingError("sealed cohort per-video inventory drifted")
    prepare_path = Path(str(cohort["provenance"]["prepare"]["path"]))
    census_path = Path(str(cohort["provenance"]["census"]["path"]))
    if sha256(prepare_path) != cohort["provenance"]["prepare"]["sha256"]:
        raise CeilingError("fit117 prepare drifted after cohort seal")
    if sha256(census_path) != cohort["provenance"]["census"]["sha256"]:
        raise CeilingError("G1 census drifted after cohort seal")
    prepare = repair.load_json(prepare_path)
    prepare_rows = {str(row["identity"]): row for row in prepare["videos"]}
    census = repair.census_by_identity(census_path)

    args.output.mkdir(parents=True)
    totals: Counter[str] = Counter()
    families: dict[str, Counter[str]] = {"44b6": Counter(), "6bba": Counter()}
    residual_rows: list[dict[str, str]] = []
    shell_all = 0
    shell_rescued = 0
    shell_missing_endpoint = Counter()
    store = ZipStore(args.zip, mode="r")
    try:
        for ordinal, identity in enumerate(selected, start=1):
            family = identity.split(":", 1)[1].split("_", 1)[0]
            fragment_path = Path(str(records[identity]["output"]["path"]))
            if sha256(fragment_path) != records[identity]["output"]["sha256"]:
                raise CeilingError(f"sealed G1 fragment drifted: {identity}")
            dataset, original, selected_edges = audit.read_fragment(fragment_path)
            if identity != f"train:{dataset}":
                raise CeilingError(f"fragment identity drifted: {identity}")
            plan = repair.load_registration_plan(census_path, identity)
            cumulative = np.asarray(plan.cumulative_shift_raw_zyx, dtype=np.float64)
            physical = audit.physical_frames(original, cumulative)
            graph_ids = {int(node) for ids, _ in original.values() for node in ids}
            node_time = {int(node): frame for frame, (ids, _) in physical.items() for node in ids}
            point = {
                int(node): coord
                for _frame, (ids, coords) in physical.items()
                for node, coord in zip(ids, coords, strict=True)
            }
            parent: dict[int, int] = {}
            for source, target in selected_edges:
                if node_time.get(source) is not None and node_time.get(target) == node_time[source] + 1:
                    if target in parent:
                        raise CeilingError(f"selected graph has duplicate in-edge: {identity}:{target}")
                    parent[target] = source

            raw_physical = {
                frame: (ids, coords * np.asarray(audit.PHYSICAL_SCALE_ZYX)[None, :])
                for frame, (ids, coords) in original.items()
            }
            gt = read_geff_arrays(store, "train", dataset)
            _matches, positives = audit.gt_matches(raw_physical, gt)
            baseline_by_frame: dict[int, set[tuple[int, int]]] = {}
            velocity_by_frame: dict[int, set[tuple[int, int]]] = {}
            velocity_rank_by_frame: dict[int, dict[tuple[int, int], int]] = {}
            centres_by_frame: dict[int, dict[int, np.ndarray]] = {}
            for frame in range(99):
                source_ids, source_points = physical[frame]
                target_ids, target_points = physical[frame + 1]
                baseline, _baseline_ranks, peak_base = audit.deterministic_candidates(
                    source_ids, source_points, target_ids, target_points
                )
                centres = np.array(source_points, copy=True)
                has_prior = np.zeros(len(source_ids), dtype=bool)
                for row, source_raw in enumerate(source_ids):
                    source = int(source_raw)
                    previous = parent.get(source)
                    if previous is None or node_time.get(previous) != frame - 1:
                        continue
                    centres[row] += audit.PERSISTENCE * (source_points[row] - point[previous])
                    has_prior[row] = True
                velocity, ranks, peak_velocity = audit.deterministic_candidates(
                    source_ids, centres, target_ids, target_points
                )
                source_row = {int(node): row for row, node in enumerate(source_ids)}
                centres_by_frame[frame] = {
                    int(node): centres[row] for row, node in enumerate(source_ids)
                }
                baseline_by_frame[frame] = baseline
                velocity_by_frame[frame] = velocity
                velocity_rank_by_frame[frame] = ranks
                positive = positives[frame]
                g1_positive = baseline & positive
                velocity_positive = velocity & positive
                metrics = {
                    "g1_candidates": len(baseline),
                    "velocity_candidates": len(velocity),
                    "g1_positive_candidates": len(g1_positive),
                    "velocity_positive_candidates": len(velocity_positive),
                    "newly_reachable_true_continuations": len(velocity_positive - baseline),
                    "lost_true_continuations": len(g1_positive - velocity),
                    "g1_background_candidates": len(baseline - positive),
                    "velocity_background_candidates": len(velocity - positive),
                    "sources_with_prior": int(has_prior.sum()),
                    "sources_without_prior": int((~has_prior).sum()),
                }
                totals.update(metrics)
                families[family].update(metrics)
                totals["peak_g1_neighbors"] = max(totals["peak_g1_neighbors"], peak_base)
                totals["peak_velocity_neighbors"] = max(totals["peak_velocity_neighbors"], peak_velocity)
                for source_raw in source_ids[~has_prior]:
                    source = int(source_raw)
                    if {pair for pair in baseline if pair[0] == source} != {
                        pair for pair in velocity if pair[0] == source
                    }:
                        totals["fallback_candidate_set_mismatches"] += 1

            cache = repair.load_cache(Path(str(records[identity]["inputs"]["cache"]["path"])).parent, identity)
            registered = repair.translated_frames(cache, census[identity]["transitions"], plan=plan)
            cache_time: dict[int, int] = {}
            cache_point: dict[int, np.ndarray] = {}
            scale = np.asarray(audit.PHYSICAL_SCALE_ZYX, dtype=np.float64)
            for frame in registered:
                for node, coord in zip(frame.node_ids, frame.coords_zyx, strict=True):
                    cache_time[int(node)] = int(frame.frame_index)
                    cache_point[int(node)] = np.asarray(coord, dtype=np.float64) * scale
            payload = torch.load(Path(str(records[identity]["inputs"]["shell_example"]["path"])), map_location="cpu", weights_only=False)
            pairs = np.asarray(payload["pairs"], dtype=np.int64)
            labels = np.asarray(payload["labels"], dtype=np.int8)
            features = np.asarray(payload["features"], dtype=np.float64)
            for index in np.flatnonzero(labels == 1):
                source, target = map(int, pairs[index])
                frame = cache_time.get(source)
                if frame is None or cache_time.get(target) != frame + 1:
                    raise CeilingError(f"shell pair is absent/nonadjacent in cache: {identity}:{source}->{target}")
                original_distance = float(np.linalg.norm(features[index, 2:5]))
                if not (10.0 < original_distance <= 12.0):
                    raise CeilingError(f"shell pair original distance drifted: {identity}:{source}->{target}")
                corrected_distance = float(np.linalg.norm(cache_point[target] - cache_point[source]))
                shell_all += 1
                if corrected_distance <= audit.RADIUS_UM:
                    shell_rescued += 1
                    continue
                source_present = source in graph_ids
                target_present = target in graph_ids
                if not source_present:
                    shell_missing_endpoint["source_absent"] += 1
                if not target_present:
                    shell_missing_endpoint["target_absent"] += 1
                previous = parent.get(source) if source_present else None
                velocity = (
                    point[source] - point[previous]
                    if previous is not None and node_time.get(previous) == frame - 1
                    else None
                )
                centre = centres_by_frame.get(frame, {}).get(source)
                prediction_distance = (
                    float(np.linalg.norm(point[target] - centre))
                    if centre is not None and target_present
                    else None
                )
                rank = velocity_rank_by_frame.get(frame, {}).get((source, target))
                retained = (source, target) in velocity_by_frame.get(frame, set())
                if not source_present:
                    reason = "source_absent_from_actual_G1_graph"
                elif not target_present:
                    reason = "target_absent_from_actual_G1_graph"
                elif prediction_distance is None or prediction_distance > audit.RADIUS_UM:
                    reason = "outside_recentered_r10"
                elif rank is None or rank > audit.CAP:
                    reason = "outside_recentered_cap8"
                elif retained:
                    reason = "reachable"
                else:
                    raise CeilingError("unclassified residual shell pair")
                row: dict[str, str] = {
                    "identity": identity,
                    "family": family,
                    "transition_t": str(frame),
                    "source_id": str(source),
                    "target_id": str(target),
                    "source_in_g1_graph": str(int(source_present)),
                    "target_in_g1_graph": str(int(target_present)),
                    "pair_in_g1_graph_candidate_universe": str(int(source_present and target_present)),
                    "source_previous_available": str(int(velocity is not None)),
                    "source_previous_id": "" if previous is None else str(previous),
                    **vector_fields("prior_velocity", velocity, "um"),
                    "prior_velocity_distance_um": scalar(None if velocity is None else np.linalg.norm(velocity)),
                    **vector_fields("g1_cumulative_shift_previous", None if frame == 0 else cumulative[frame - 1], "vox"),
                    **vector_fields("g1_cumulative_shift_source", cumulative[frame], "vox"),
                    **vector_fields("g1_cumulative_shift_target", cumulative[frame + 1], "vox"),
                    "original_shell_distance_um": scalar(original_distance),
                    "global_corrected_distance_um": scalar(corrected_distance),
                    "persistence_prediction_distance_um": scalar(prediction_distance),
                    "persistence_prediction_inside_r10": str(int(prediction_distance is not None and prediction_distance <= audit.RADIUS_UM)),
                    "persistence_prediction_rank_inside_r10": "" if rank is None else str(rank),
                    "velocity_window_candidate_after_cap8": str(int(retained)),
                    "unreachable_reason": reason,
                }
                residual_rows.append(row)
            print(f"fit117 ceiling {ordinal:03d}/117 {identity}", flush=True)
    finally:
        store.close()

    if shell_all != 456 or shell_rescued != 90 or len(residual_rows) != 366:
        raise CeilingError(
            f"strict shell conservation failed: {shell_all}-{shell_rescued}={len(residual_rows)}, expected 456-90=366"
        )
    if totals["fallback_candidate_set_mismatches"]:
        raise CeilingError("no-history fallback changed a candidate set")
    retained_shell = sum(int(row["velocity_window_candidate_after_cap8"]) for row in residual_rows)
    reason_counts = Counter(row["unreachable_reason"] for row in residual_rows)
    newly = int(totals["newly_reachable_true_continuations"])
    lost = int(totals["lost_true_continuations"])
    denominator = sum(audit.G1_COUNTS)
    burden = float(totals["velocity_background_candidates"]) / float(totals["g1_background_candidates"])
    gate = {
        "newly_reachable_true_continuations": {"observed": newly, "minimum": 20, "passes": newly >= 20},
        "optimistic_no_new_fp_raw_edge_ceiling": {
            "formula": "new / (34389+2308+1879)", "observed": newly / denominator,
            "minimum": 0.0005, "passes": newly / denominator >= 0.0005,
        },
        "background_burden_multiplier": {"observed": burden, "maximum": 1.05, "passes": burden <= 1.05},
        "residual_shell_retained_after_cap8": {"observed": retained_shell, "minimum": 20, "passes": retained_shell >= 20},
    }
    eligible = all(bool(value["passes"]) for value in gate.values())
    args.output.mkdir(exist_ok=True)
    csv_path = args.output / "SEALED_INPUTS_366.csv"
    with csv_path.open("x", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(residual_rows)
    result = {
        "schema": "celltrack-alpha.velocity-aware-fit117-reachability-ceiling.v1",
        "status": "MATERIAL_SCORE_AUTHORIZED" if eligible else "COMPLETED_NO_SCORE",
        "mechanism": {
            "name": "VELOCITY-AWARE-CANDIDATE-WINDOW-v1",
            "operation": "per-source motion-predicted recentering",
            "formula": "qhat[t+1]=q[t]+0.30*(q[t]-q[selected_parent,t-1])",
            "radius_um": 10.0, "cap": 8, "radius_widened": False,
            "base": "actual 117-video G1-selected graph",
        },
        "strict_conservation": {
            "original_shell_positives": shell_all,
            "rescued_by_g1_inside_r10": shell_rescued,
            "residual_edges": len(residual_rows),
            "equation": "456-90=366",
            "guard_relaxed": False,
            "passes": True,
        },
        "sealed_inputs": {
            "rows": len(residual_rows), "path": str(csv_path), "sha256": sha256(csv_path),
            "carries_selected_predecessor_vector": True,
            "carries_prior_source/target_g1_shifts": True,
            "carries_recentered_cap8_rank": True,
        },
        "residual_shell": {
            "retained_after_recentered_r10_cap8": retained_shell,
            "reachability_gain": retained_shell,
            "by_reason": dict(reason_counts),
            "endpoint_absence_events": dict(shell_missing_endpoint),
        },
        "deployment_population": {
            "videos": 117,
            "g1_candidate_pairs": int(totals["g1_candidates"]),
            "velocity_candidate_pairs": int(totals["velocity_candidates"]),
            "g1_positive_candidate_pairs": int(totals["g1_positive_candidates"]),
            "velocity_positive_candidate_pairs": int(totals["velocity_positive_candidates"]),
            "newly_reachable_true_continuations": newly,
            "lost_true_continuations": lost,
            "net_reachable_true_continuations": newly - lost,
            "g1_background_candidates": int(totals["g1_background_candidates"]),
            "velocity_background_candidates": int(totals["velocity_background_candidates"]),
            "background_burden_multiplier": burden,
            "by_family": {name: dict(values) for name, values in families.items()},
        },
        "pre_scorer_gate": {
            "gate": gate,
            "verdict": "SCORE_AUTHORIZED" if eligible else "NO_SCORE",
            "exact_scorer_called": False,
        },
        "provenance": {
            "cohort": {"path": str(args.cohort), "sha256": sha256(args.cohort)},
            "prepare": {"path": str(prepare_path), "sha256": sha256(prepare_path)},
            "census": {"path": str(census_path), "sha256": sha256(census_path)},
            "archive": {"path": str(args.zip), "bytes": args.zip.stat().st_size},
            "audit_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "scorer_called": False, "graph_solver_called": False,
            "kaggle_action": False, "runpod_action": False,
        },
    }
    result_path = args.output / "REACHABILITY_CEILING.json"
    write_new_json(result_path, result)
    manifest = {
        "schema": "celltrack-alpha.velocity-aware-fit117-reachability-run-manifest.v1",
        "result_sha256": sha256(result_path),
        "sealed_inputs_sha256": sha256(csv_path),
        "cohort_sha256": sha256(args.cohort),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
    }
    write_new_json(args.output / "RUN_MANIFEST.json", manifest)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "gate": result["pre_scorer_gate"]}, sort_keys=True))
