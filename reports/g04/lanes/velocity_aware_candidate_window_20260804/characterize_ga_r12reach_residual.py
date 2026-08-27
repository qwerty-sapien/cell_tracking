#!/usr/bin/env python3
"""Characterize the sealed GA-R12REACH residual after the G1 rescue.

This is deliberately a cohort characterization, not a new graph construction.
It reads the original 117-video fit-only shell feature records and the frozen
G1 correction plan.  It does not instantiate an edge scorer or graph solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
for location in (ROOT / "src", ROOT / "scripts", ROOT / "external/Tracking_Support/repo/src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha import bh_model  # noqa: E402


SCHEMA = "celltrack-alpha.velocity-aware-ga-r12reach-characterization.v1"
PERSISTENCE = 0.30
PUBLISHED_MEDIAN_STEP_UM = 1.86
EXPECTED_POSITIVES = 456
EXPECTED_RESCUED = 90
EXPECTED_RESIDUAL = 366
FEATURE_NAMES = (
    "distance_um", "retained_rank", "displacement_z_um", "displacement_y_um",
    "displacement_x_um", "source_density_10um", "target_density_10um",
    "topology_code", "conflict_count", "source_history_available",
    "source_acceleration_residual_um", "target_future_available",
    "target_continuation_residual_um", "source_no_link_probability",
    "incumbent_probability_available", "maximum_incumbent_probability",
    "minimum_incumbent_distance_um", "incumbent_minus_challenger_distance_um",
)
CSV_FIELDS = (
    "identity", "family", "transition_t", "source_id", "target_id",
    "raw_displacement_z_um", "raw_displacement_y_um", "raw_displacement_x_um",
    "raw_distance_um", "g1_shift_z_um", "g1_shift_y_um", "g1_shift_x_um",
    "g1_corrected_displacement_z_um", "g1_corrected_displacement_y_um",
    "g1_corrected_displacement_x_um", "g1_corrected_distance_um",
    "source_prior_available", "source_acceleration_residual_um",
    "source_prior_speed_um_per_frame", "prior_speed_minus_published_median_um",
    "source_density_r10", "target_density_r10", "original_shell_rank",
    "persistence_prediction_residual_lower_um", "persistence_prediction_residual_upper_um",
    "persistence_target_in_r10", "candidate_after_recenter_cap8", "integrity_note",
)


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AuditError(f"JSON object required: {path}")
    return value


def new_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="ascii") as handle:
        handle.write(canonical_json(value).decode("ascii"))
        handle.write("\n")


def finite(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def summary(values: list[float]) -> Mapping[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise AuditError("nonfinite summary value")
    return {
        "n": int(len(array)), "minimum": float(array.min()), "p25": float(np.quantile(array, .25)),
        "median": float(np.median(array)), "mean": float(array.mean()), "p75": float(np.quantile(array, .75)),
        "p90": float(np.quantile(array, .90)), "maximum": float(array.max()),
    }


def family(identity: str) -> str:
    return identity.split(":", 1)[1].split("_", 1)[0]


def cache_node_frames(cache_root: Path, identity: str) -> tuple[dict[int, int], str]:
    slug = identity.replace(":", "__", 1)
    metadata = load_json(cache_root / f"{slug}.json")
    model = metadata.get("contract", {}).get("model")
    cache_sha = metadata.get("cache_sha256")
    if not isinstance(model, Mapping) or not isinstance(cache_sha, str):
        raise AuditError(f"cache metadata contract drifted for {identity}")
    try:
        cache = bh_model.load_sparse_cache(
            cache_root / f"{slug}.bhc", expected_identity=bh_model.CacheIdentity(**dict(model))
        )
    except (OSError, ValueError, TypeError, bh_model.CacheValidationError) as exc:
        raise AuditError(f"cannot validate fit cache for {identity}: {exc}") from exc
    node_frame = {int(node): int(frame.frame_index) for frame in cache.detections for node in frame.node_ids}
    if len(node_frame) != sum(len(frame.node_ids) for frame in cache.detections):
        raise AuditError(f"duplicate fit-cache node identity: {identity}")
    return node_frame, cache_sha


def main(args: argparse.Namespace) -> None:
    out = args.output
    audit_path, csv_path, manifest_path = out / "AUDIT.json", out / "GA_R12REACH_RESIDUAL_366.csv", out / "RUN_MANIFEST.json"
    if any(path.exists() for path in (audit_path, csv_path, manifest_path)):
        raise AuditError("refusing to overwrite completed characterization")
    prepare = load_json(args.prepare)
    if prepare.get("schema") != "celltrack-alpha.extended-radius-shell-prepare.v1":
        raise AuditError("GA shell artifact schema drifted")
    if int(prepare.get("totals", {}).get("positive_pairs", -1)) != EXPECTED_POSITIVES:
        raise AuditError("GA shell-positive total drifted")
    census = load_json(args.census)
    if census.get("verdict") != "PASS_COMPLETE_CENSUS":
        raise AuditError("G1 census is not complete")
    census_rows = {f"{row['split']}:{row['dataset']}": row for row in census.get("videos", [])}
    identities = tuple(str(value) for value in (*prepare["scope"]["train_video_ids"], *prepare["scope"]["validation_video_ids"]))
    examples = sorted(args.examples.glob("train__*.pt"))
    if len(identities) != 117 or len(examples) != len(identities):
        raise AuditError("sealed GA fit117 identity/example coverage drifted")

    rows: list[dict[str, str]] = []
    all_positive = rescued = 0
    by_family: Counter[str] = Counter()
    corrected_distance: list[float] = []
    acceleration: list[float] = []
    source_density: list[float] = []
    target_density: list[float] = []
    source_with_prior = 0
    fallback_outside = not_identifiable = 0
    cache_hashes: list[Mapping[str, str]] = []
    for identity in identities:
        slug = identity.replace(":", "__", 1)
        path = args.examples / f"{slug}.pt"
        if not path.is_file():
            raise AuditError(f"missing sealed GA example: {path}")
        # This repository-generated payload contains NumPy arrays.  It is read
        # without executing code, and its checksum is recorded below.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise AuditError(f"feature schema drifted: {path}")
        features = np.asarray(payload["features"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int8)
        pairs = np.asarray(payload["pairs"], dtype=np.int64)
        if features.shape != (len(labels), len(FEATURE_NAMES)) or pairs.shape != (len(labels), 2):
            raise AuditError(f"sealed GA example array shape drifted: {path}")
        node_frame, cache_sha = cache_node_frames(args.cache, identity)
        cache_hashes.append({"identity": identity, "cache_sha256": cache_sha, "example_sha256": sha256(path)})
        census_entry = census_rows.get(identity)
        if census_entry is None:
            raise AuditError(f"G1 census lacks GA identity {identity}")
        transitions = load_json(args.census.parent / str(census_entry["path"]))["transitions"]
        by_transition = {int(item["frame"]): item for item in transitions}
        for row_index in np.flatnonzero(labels == 1):
            all_positive += 1
            source, target = map(int, pairs[row_index])
            t = node_frame.get(source)
            if t is None or node_frame.get(target) != t + 1:
                raise AuditError(f"GA pair/cache adjacent identity drifted: {identity}:{source}->{target}")
            transition = by_transition.get(t)
            if transition is None:
                raise AuditError(f"G1 transition missing: {identity} t={t}")
            raw = features[row_index, 2:5]
            shift = np.asarray(transition["shift_um_zyx"], dtype=np.float64) if bool(transition["pool_breaking"]) else np.zeros(3)
            corrected = raw + shift
            corrected_norm = float(np.linalg.norm(corrected))
            if corrected_norm <= 10.0:
                rescued += 1
                continue
            has_prior = bool(int(features[row_index, 9]))
            accel = float(features[row_index, 10]) if has_prior else None
            lower = upper = None
            target_inside = "no_fallback_outside_r10"
            cap8 = "not_assessable"
            note = "no selected predecessor in sealed GA feature record; exact G1 fallback remains outside r10"
            if has_prior:
                source_with_prior += 1
                assert accel is not None
                target_inside = "not_identifiable_from_sealed_scalar_history"
                not_identifiable += 1
                note = (
                    "predecessor ID/vector and previous-transition G1 shift are absent from the sealed GA record; "
                    "raw acceleration scalar cannot identify the recentered distance or cap8 rank"
                )
                acceleration.append(accel)
            else:
                fallback_outside += 1
            rows.append({
                "identity": identity, "family": family(identity), "transition_t": str(t),
                "source_id": str(source), "target_id": str(target),
                "raw_displacement_z_um": finite(float(raw[0])), "raw_displacement_y_um": finite(float(raw[1])),
                "raw_displacement_x_um": finite(float(raw[2])), "raw_distance_um": finite(float(np.linalg.norm(raw))),
                "g1_shift_z_um": finite(float(shift[0])), "g1_shift_y_um": finite(float(shift[1])),
                "g1_shift_x_um": finite(float(shift[2])),
                "g1_corrected_displacement_z_um": finite(float(corrected[0])),
                "g1_corrected_displacement_y_um": finite(float(corrected[1])),
                "g1_corrected_displacement_x_um": finite(float(corrected[2])),
                "g1_corrected_distance_um": finite(corrected_norm),
                "source_prior_available": str(int(has_prior)), "source_acceleration_residual_um": finite(accel),
                "source_prior_speed_um_per_frame": "", "prior_speed_minus_published_median_um": "",
                "source_density_r10": str(int(features[row_index, 5])), "target_density_r10": str(int(features[row_index, 6])),
                "original_shell_rank": str(int(features[row_index, 1])),
                "persistence_prediction_residual_lower_um": finite(lower),
                "persistence_prediction_residual_upper_um": finite(upper),
                "persistence_target_in_r10": target_inside, "candidate_after_recenter_cap8": cap8,
                "integrity_note": note,
            })
            by_family[family(identity)] += 1
            corrected_distance.append(corrected_norm)
            source_density.append(float(features[row_index, 5]))
            target_density.append(float(features[row_index, 6]))
    if all_positive != EXPECTED_POSITIVES or rescued != EXPECTED_RESCUED or len(rows) != EXPECTED_RESIDUAL:
        raise AuditError(f"GA residual arithmetic drifted: positives={all_positive}, rescued={rescued}, residual={len(rows)}")

    cache_manifest_sha = hashlib.sha256(canonical_json(cache_hashes)).hexdigest()
    result: Mapping[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETED_INTEGRITY_BLOCKED_NO_SCORE",
        "cohort": {
            "name": "GA-R12REACH fit-only117 residual", "original_shell_positive_count": all_positive,
            "g1_rescued_inside_r10": rescued, "residual_count": len(rows), "residual_by_family": dict(by_family),
            "definition": "sealed original 10--12 um/cap8 positive shell, excluding the 90 G1-corrected positives at <=10 um",
        },
        "mechanism": {
            "name": "VELOCITY-AWARE-CANDIDATE-WINDOW-v1", "radius_um": 10.0, "neighbor_cap": 8,
            "lag1_directional_persistence": PERSISTENCE, "published_median_step_um_per_frame": PUBLISHED_MEDIAN_STEP_UM,
            "formula": "qhat = q + 0.30 * (q - q_predecessor)",
            "fallback": "qhat=q when no selected predecessor exists",
        },
        "characterization": {
            "g1_corrected_distance_um": summary(corrected_distance),
            "source_density_r10": summary(source_density),
            "target_density_r10": summary(target_density),
            "source_prior_available": source_with_prior,
            "source_acceleration_residual_um": summary(acceleration),
            "persistence_bounds": {
                "fallback_definitely_outside_r10": fallback_outside,
                "not_identifiable_from_sealed_scalar_history": not_identifiable,
                "note": "No bound is valid for prior sources because the scalar raw acceleration omits both predecessor direction and prior G1 transition shift.",
            },
        },
        "cv57_g1_cache_transfer_control": {
            "cohort": "CV57 / 75 pool-breaking transitions / BHC pregraph cache",
            "candidate_pairs_before_g1": 27276, "candidate_pairs_after_g1": 25553,
            "reachable_matched_gt_continuations_before_g1": 209,
            "reachable_matched_gt_continuations_after_g1": 259,
            "matched_gt_continuations": 311, "nonpositive_candidate_pairs_after_g1": 25294,
            "parent_score_vector": {
                "adjusted_edge_jaccard": 0.8867644202595942,
                "raw_edge_jaccard": 0.891461012028204,
                "node_recall": 0.990377056057473,
                "edge_tp_fp_fn": [34389, 2308, 1879],
            },
            "source": "reports/g04/lanes/data_pathology_repair/score/REPORT.json",
            "note": "Separate from GA-R12REACH fit-only117; valid as a G1 reachability/background control only.",
        },
        "integrity_blocker": {
            "blocked": True,
            "reason": "GA-R12REACH exposes scalar raw acceleration residuals but not selected predecessor IDs/vectors, prior-transition G1 shifts, or recentered target ranks. CV57 G1/APNCC fragment identities are a distinct cohort and cannot supply those endpoints.",
            "consequence": "No exact recentered reachability/background delta, no cap8 determination, no graph/scorer execution, and no score claim.",
        },
        "validation": {"positive_count": all_positive, "rescued_count": rescued, "residual_count": len(rows), "passes": True},
        "provenance": {
            "shell_prepare": {"path": str(args.prepare), "sha256": sha256(args.prepare)},
            "examples_cache_manifest_sha256": cache_manifest_sha,
            "census": {"path": str(args.census), "sha256": sha256(args.census)},
            "g1_report": {"path": str(args.g1_report), "sha256": sha256(args.g1_report)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "execution": {"scorer_called": False, "graph_called": False, "kaggle_action": False, "runpod_action": False, "submission_action": False},
    }
    with csv_path.open("x", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    new_json(audit_path, result)
    new_json(manifest_path, {
        "schema": "celltrack-alpha.velocity-aware-ga-r12reach-manifest.v1",
        "audit_sha256": sha256(audit_path), "csv_sha256": sha256(csv_path),
        "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv313/bin/python characterize_ga_r12reach_residual.py",
    })
    print(json.dumps({"status": result["status"], "residual": len(rows)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    value.add_argument("--output", type=Path, default=here)
    value.add_argument("--prepare", type=Path, default=ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json")
    value.add_argument("--examples", type=Path, default=ROOT / "reports/g02/extended_radius_shell_prepare_v1/examples")
    value.add_argument("--cache", type=Path, default=ROOT / "reports/g02/fitcache_117/bh9out/cache")
    value.add_argument("--census", type=Path, default=ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json")
    value.add_argument("--g1-report", type=Path, default=ROOT / "reports/g04/lanes/data_pathology_repair/score/REPORT.json")
    return value


if __name__ == "__main__":
    main(parser().parse_args())
