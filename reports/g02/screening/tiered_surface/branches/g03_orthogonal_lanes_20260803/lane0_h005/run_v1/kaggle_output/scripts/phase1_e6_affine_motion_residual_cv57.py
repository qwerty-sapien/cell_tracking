#!/usr/bin/env python3
"""Measure leakage-safe global-affine residuals on exact-baseline CV57 FNs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import diagnose_phase1_distance_motion_gate as raw_gate  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.phase1-e6-affine-motion-residual-raw-cv57.v2"
GATE_UM = 10.0
MATCH_RADIUS_UM = 7.0
EXPECTED_BASELINE = {
    "video_count": 57,
    "adj_edge_jaccard": 0.8711885823490648,
    "edge_jaccard": 0.8761665470208184,
    "edge_tp": 34174,
    "edge_fp": 2736,
    "edge_fn": 2094,
    "division_tp": 0.0,
    "division_fp": 0.0,
    "division_fn": 42.0,
}

DEFAULT_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_FRAGMENTS = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
DEFAULT_CACHE = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
DEFAULT_SCORES = (
    ROOT
    / "_ARCHIVE/conflict_residual/reports/g02/"
    "run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json"
)
DEFAULT_PARITY = ROOT / "reports/g02/scorer_parity_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/phase1_e6_affine_motion_residual_raw_cv57_v2.json"
)


class E6Error(RuntimeError):
    """The exact leakage-safe E6 contract was not satisfied."""


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise E6Error(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E6Error(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: Sequence[float]) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _fit_affine(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Fit target=(source-center)@coef with a free intercept (12 DOF)."""

    left = np.asarray(source, dtype=np.float64).reshape((-1, 3))
    right = np.asarray(target, dtype=np.float64).reshape((-1, 3))
    if left.shape != right.shape or left.shape[0] < 4:
        raise E6Error("12-DOF affine fit requires at least four 3-D correspondences")
    center = np.mean(left, axis=0)
    design = np.column_stack((left - center, np.ones(left.shape[0])))
    coefficient, _, rank, singular = np.linalg.lstsq(design, right, rcond=None)
    if int(rank) != 4:
        raise E6Error(f"12-DOF affine design is rank deficient: rank={rank}")
    condition = float(singular[0] / singular[-1])
    if not math.isfinite(condition):
        raise E6Error("12-DOF affine design has nonfinite condition number")
    return center, coefficient, int(rank), condition


def _predict_affine(
    point: np.ndarray, center: np.ndarray, coefficient: np.ndarray
) -> np.ndarray:
    row = np.append(np.asarray(point, dtype=np.float64) - center, 1.0)
    return np.asarray(row @ coefficient, dtype=np.float64)


def _raw_detection_maps(
    detections: Sequence[Any],
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    coord_by_id: dict[int, np.ndarray] = {}
    time_by_id: dict[int, int] = {}
    for frame in detections:
        for row, node_id_value in enumerate(frame.node_ids):
            node_id = int(node_id_value)
            if node_id in coord_by_id:
                raise E6Error(f"duplicate raw FrameDet node ID {node_id}")
            coord_by_id[node_id] = np.asarray(
                frame.coords_zyx[row], dtype=np.float64
            )
            time_by_id[node_id] = int(frame.frame_index)
    return coord_by_id, time_by_id


def _fit_rows_by_pair(
    graph: Any,
    detections: Sequence[Any],
    scale: np.ndarray,
) -> dict[int, list[dict[str, Any]]]:
    coord_by_id, time_by_id = _raw_detection_maps(detections)
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph.edges:
        source_id = int(edge.source_id)
        target_id = int(edge.target_id)
        if source_id not in coord_by_id or target_id not in coord_by_id:
            continue
        source_t = time_by_id[source_id]
        if time_by_id[target_id] != source_t + 1:
            continue
        rows[source_t].append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "source": coord_by_id[source_id] * scale,
                "target": coord_by_id[target_id] * scale,
            }
        )
    return rows


def _baseline_contract(payload: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    summary = payload.get("summary")
    samples = payload.get("samples")
    selection = payload.get("selection")
    execution = payload.get("execution")
    if not all(
        isinstance(value, Mapping) for value in (summary, selection, execution)
    ) or not isinstance(samples, list):
        raise E6Error("baseline score report has the wrong schema")
    identities = tuple(str(value) for value in selection["selected_video_ids"])
    observed = {
        "video_count": len(identities),
        "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
        "edge_jaccard": float(summary["edge_jaccard"]),
        "edge_tp": int(sum(float(row["edge_tp"]) for row in samples)),
        "edge_fp": int(sum(float(row["edge_fp"]) for row in samples)),
        "edge_fn": int(sum(float(row["edge_fn"]) for row in samples)),
        "division_tp": float(summary["division_tp"]),
        "division_fp": float(summary["division_fp"]),
        "division_fn": float(summary["division_fn"]),
    }
    if observed != EXPECTED_BASELINE:
        raise E6Error(f"exact CV57 baseline changed: {observed}")
    if execution.get("division_enabled") is not False:
        raise E6Error("baseline score report is not divisions-off")
    return identities, observed


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.output.exists():
        raise E6Error(f"refusing to overwrite E6 output: {args.output}")
    baseline = _load(args.baseline_scores, "exact baseline scores")
    score_identities, baseline_observed = _baseline_contract(baseline)
    split = _load(args.split, "frozen split")
    scorer = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    archive_ids = scoreall.archive_train_video_ids(args.archive)
    selected = score_bh_cv57.exact_selection(
        split, split_path=args.split, archive_video_ids=archive_ids
    )
    if tuple(selected) != score_identities:
        raise E6Error("baseline score identity order differs from exact CV57")
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(args.fragments)), selected
    )
    fragment_by_identity = dict(manifest.ordered)
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if set(cache_paths) != set(selected) or set(meta_paths) != set(selected):
        raise E6Error("raw cache/meta files do not exactly cover CV57")
    sample_by_identity = {str(row["identity"]): row for row in baseline["samples"]}
    if set(sample_by_identity) != set(selected):
        raise E6Error("baseline samples do not exactly cover CV57")

    cohort: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    matched_tp_total = 0
    matched_fn_total = 0
    with ZipStore(args.archive, mode="r") as store:
        for index, identity in enumerate(selected, start=1):
            _, dataset, _ = scoreall.parse_identity(identity)
            fragment = fragment_by_identity[identity]
            expected_fragment = sample_by_identity[identity].get("source", {})
            if _sha256(fragment) != expected_fragment.get("sha256"):
                raise E6Error(f"exact baseline fragment hash changed for {identity}")
            graph = scoreall._fragment_graph(fragment, identity)
            detections = raw_gate.load_detections(
                cache_paths[identity], meta_paths[identity]
            )
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale_zyx = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            scale = np.asarray(scale_zyx, dtype=np.float64)
            video_row, raw_events, accounting = raw_gate.analyse_video(
                identity=identity,
                graph=graph,
                gt=gt,
                scale_zyx=scale_zyx,
                detections=detections,
                expected_score=sample_by_identity[identity],
            )
            coord_by_id, _ = _raw_detection_maps(detections)
            fit_by_frame = _fit_rows_by_pair(graph, detections, scale)
            video_tp = int(accounting["edge_tp"])
            video_fn = int(accounting["edge_fn"])
            video_cohort = 0
            above_gate_events = [
                event
                for event in raw_events
                if bool(event["is_fn"]) and bool(event["above_10um_fn"])
            ]
            for event in above_gate_events:
                pred_source = int(event["pred_source_id"])
                pred_target = int(event["pred_target_id"])
                source_physical = coord_by_id[pred_source] * scale
                target_physical = coord_by_id[pred_target] * scale
                direct_distance = float(event["raw_distance_um"])
                observed_direct = float(
                    np.linalg.norm(target_physical - source_physical)
                )
                if not math.isclose(
                    direct_distance, observed_direct, rel_tol=0.0, abs_tol=1.0e-12
                ) or direct_distance <= GATE_UM + 1.0e-12:
                    raise E6Error(f"raw >10um event contract changed for {identity}")
                frame = int(event["source_frame"])
                full_rows = fit_by_frame.get(frame, [])
                endpoint_ids = {pred_source, pred_target}
                loo_rows = [
                    row
                    for row in full_rows
                    if row["source_id"] not in endpoint_ids
                    and row["target_id"] not in endpoint_ids
                ]
                excluded = len(full_rows) - len(loo_rows)
                if len(loo_rows) < 4:
                    raise E6Error(
                        f"insufficient leakage-safe affine rows for {identity} t={frame}: "
                        f"full={len(full_rows)} loo={len(loo_rows)}"
                    )
                loo_source = np.asarray(
                    [row["source"] for row in loo_rows], dtype=np.float64
                )
                loo_target = np.asarray(
                    [row["target"] for row in loo_rows], dtype=np.float64
                )
                center, coefficient, rank, condition = _fit_affine(
                    loo_source, loo_target
                )
                affine_prediction = _predict_affine(
                    source_physical, center, coefficient
                )
                residual = float(
                    np.linalg.norm(target_physical - affine_prediction)
                )

                full_source = np.asarray(
                    [row["source"] for row in full_rows], dtype=np.float64
                )
                full_target = np.asarray(
                    [row["target"] for row in full_rows], dtype=np.float64
                )
                full_center, full_coefficient, full_rank, full_condition = _fit_affine(
                    full_source, full_target
                )
                full_residual = float(
                    np.linalg.norm(
                        target_physical
                        - _predict_affine(
                            source_physical, full_center, full_coefficient
                        )
                    )
                )
                cohort.append(
                    {
                        "identity": identity,
                        "source_frame": frame,
                        "gt_source_id": int(event["gt_source_id"]),
                        "gt_target_id": int(event["gt_target_id"]),
                        "pred_source_id": pred_source,
                        "pred_target_id": pred_target,
                        "direct_distance_um": direct_distance,
                        "affine_loo_residual_um": residual,
                        "affine_full_residual_um_sensitivity_only": full_residual,
                        "full_minus_loo_residual_um": full_residual - residual,
                        "below_10um_after_affine_loo": residual <= GATE_UM + 1.0e-12,
                        "fit_correspondences_full": len(full_rows),
                        "fit_correspondences_loo": len(loo_rows),
                        "fit_correspondences_excluded_as_incident": excluded,
                        "fit_rank_loo": rank,
                        "fit_condition_loo": condition,
                        "fit_rank_full": full_rank,
                        "fit_condition_full": full_condition,
                    }
                )
                video_cohort += 1
            expected_row = sample_by_identity[identity]
            expected_tp = int(float(expected_row["edge_tp"]))
            expected_fn = int(float(expected_row["edge_fn"]))
            if (video_tp, video_fn) != (expected_tp, expected_fn):
                raise E6Error(
                    f"official edge accounting mismatch for {identity}: "
                    f"observed={(video_tp, video_fn)} expected={(expected_tp, expected_fn)}"
                )
            matched_tp_total += video_tp
            matched_fn_total += video_fn
            per_video.append(
                {
                    "identity": identity,
                    "edge_tp": video_tp,
                    "edge_fn": video_fn,
                    "above_10um_fn_cohort": video_cohort,
                    "eligible_fn_edges": int(video_row["eligible_fn_edges"]),
                    "in_gate_fn_edges": int(video_row["in_gate_fn_edges"]),
                    "ineligible_gt_edges": int(video_row["ineligible_gt_edges"]),
                    "ineligible_reasons": video_row["ineligible_reasons"],
                    "predicted_adjacent_fit_edges": int(
                        sum(len(rows) for rows in fit_by_frame.values())
                    ),
                }
            )
            print(
                f"E6 audit {index:02d}/{len(selected)} {identity} cohort={video_cohort}",
                flush=True,
            )

    if (matched_tp_total, matched_fn_total) != (
        EXPECTED_BASELINE["edge_tp"],
        EXPECTED_BASELINE["edge_fn"],
    ):
        raise E6Error(
            "aggregate official TP/FN accounting did not reproduce exact CV57"
        )
    if not cohort:
        raise E6Error("new exact-baseline >10um FN cohort is empty")
    in_gate_count = int(sum(row["in_gate_fn_edges"] for row in per_video))
    all_ineligible_reasons = {
        reason: sum(
            int(row["ineligible_reasons"].get(reason, 0)) for row in per_video
        )
        for reason in {
            key for row in per_video for key in row["ineligible_reasons"]
        }
    }
    all_ineligible_count = int(sum(all_ineligible_reasons.values()))
    excluded_fn_count = matched_fn_total - in_gate_count - len(cohort)
    official_unmatched_fn = int(
        all_ineligible_reasons.get("official_unmatched_endpoint", 0)
    )
    excluded_fn_reasons = {
        "official_unmatched_endpoint": official_unmatched_fn,
        "synthetic_or_pruned_raw_endpoint": (
            excluded_fn_count - official_unmatched_fn
        ),
    }
    selected_tp_raw_ineligible = all_ineligible_count - excluded_fn_count
    expected_raw_reconciliation = {
        "above_10um_fn": 317,
        "in_gate_fn": 1232,
        "excluded_fn": 545,
        "excluded_reasons": {
            "official_unmatched_endpoint": 464,
            "synthetic_or_pruned_raw_endpoint": 81,
        },
        "selected_tp_raw_ineligible": 258,
        "all_raw_ineligible_gt_edges": 803,
    }
    observed_raw_reconciliation = {
        "above_10um_fn": len(cohort),
        "in_gate_fn": in_gate_count,
        "excluded_fn": excluded_fn_count,
        "excluded_reasons": excluded_fn_reasons,
        "selected_tp_raw_ineligible": selected_tp_raw_ineligible,
        "all_raw_ineligible_gt_edges": all_ineligible_count,
    }
    if observed_raw_reconciliation != expected_raw_reconciliation:
        raise E6Error(
            "raw FrameDet cohort differs from independent E1 reconciliation: "
            f"{observed_raw_reconciliation}"
        )
    residuals = [float(row["affine_loo_residual_um"]) for row in cohort]
    full_residuals = [
        float(row["affine_full_residual_um_sensitivity_only"]) for row in cohort
    ]
    direct = [float(row["direct_distance_um"]) for row in cohort]
    recovered = sum(value <= GATE_UM + 1.0e-12 for value in residuals)
    fraction = recovered / len(cohort)
    recommend_affine = fraction >= 0.5
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "supersedes": {
            "path": str(
                ROOT / "reports/g02/phase1_e6_affine_motion_residual_cv57_v1.json"
            ),
            "sha256": _sha256(
                ROOT / "reports/g02/phase1_e6_affine_motion_residual_cv57_v1.json"
            ),
            "status": "retracted",
            "reason": (
                "v1 measured final postprocessed CSV coordinates and therefore did not "
                "represent the actual raw FrameDet gate cohort"
            ),
        },
        "selection": {"video_count": len(selected), "video_ids": list(selected)},
        "baseline": baseline_observed,
        "cohort": {
            "derivation": (
                "Recomputed from the exact 57-video divisions-off baseline. FN membership "
                "uses patched official 7um matching on baseline fragments; endpoint "
                "coordinates and the >10um test use original raw cached FrameDet nodes "
                "before smoothing, synthesis, or image-support projection."
            ),
            "historical_counts_imported": False,
            "final_postprocessed_coordinate_cohort_used": False,
            "exact_baseline_edge_fn": matched_fn_total,
            "above_10um_fn_count": len(cohort),
            "in_gate_eligible_fn_count": in_gate_count,
            "excluded_fn_count": excluded_fn_count,
            "excluded_fn_reasons": dict(sorted(excluded_fn_reasons.items())),
            "selected_tp_raw_ineligible_count": selected_tp_raw_ineligible,
            "all_raw_ineligible_gt_edge_count": all_ineligible_count,
            "all_raw_ineligible_gt_edge_reasons": dict(
                sorted(all_ineligible_reasons.items())
            ),
            "exclusion_reconciliation_method": (
                "Official-unmatched endpoints are necessarily FNs. The remaining "
                "excluded FNs are the raw-absent remainder after subtracting eligible "
                "in-gate and above-gate FNs from the exact baseline FN total; the "
                "remaining raw-ineligible GT edges are selected official TPs."
            ),
            "independent_e1_reconciliation": {
                "expected": expected_raw_reconciliation,
                "observed": observed_raw_reconciliation,
                "exact": True,
            },
            "direct_distance_um": _quantiles(direct),
        },
        "fit": {
            "model": "per-video/per-frame-pair global 3D affine (12 DOF)",
            "fit_source": (
                "All direct t->t+1 edges selected by the exact baseline tracker whose "
                "endpoints both exist in raw FrameDet, using those raw coordinates; no "
                "GT correspondence or image label is used to fit the transform."
            ),
            "leakage_control": (
                "For each queried FN, refit after removing every selected-edge "
                "correspondence incident to either matched query endpoint. The target "
                "FN edge is absent by definition."
            ),
            "estimator": "centered ordinary least squares; full-rank 4x3 design",
            "affine_loo_residual_um": _quantiles(residuals),
            "affine_full_residual_um_sensitivity_only": _quantiles(full_residuals),
            "full_minus_loo_residual_um": _quantiles(
                [float(row["full_minus_loo_residual_um"]) for row in cohort]
            ),
            "fit_correspondences_loo": _quantiles(
                [float(row["fit_correspondences_loo"]) for row in cohort]
            ),
            "fit_condition_loo": _quantiles(
                [float(row["fit_condition_loo"]) for row in cohort]
            ),
        },
        "decision": {
            "affine_implementation_gate": {
                "rule": "PASS iff >=50% of the exact >10um FN cohort falls within 10um",
                "status": "PASS" if recommend_affine else "FAIL",
                "below_10um_count": recovered,
                "cohort_count": len(cohort),
                "below_10um_fraction": fraction,
                "recommendation": (
                    "implement global affine motion before learned flow"
                    if recommend_affine
                    else "do not implement affine motion"
                ),
            },
            "motion_axis_bounded_rule": {
                "rule": "PASS iff fewer than 50% fall within 10um after affine warping",
                "status": "PASS" if not recommend_affine else "FAIL",
                "interpretation": (
                    "tail is not predominantly coherent tissue motion; motion axis is bounded near zero"
                    if not recommend_affine
                    else "coherent affine tissue motion explains at least half of the tail"
                ),
            },
        },
        "summary": {
            "baseline_adj_edge_jaccard": baseline_observed[
                "adj_edge_jaccard"
            ],
            "baseline_edge_jaccard": baseline_observed["edge_jaccard"],
            "baseline_edge_tp": baseline_observed["edge_tp"],
            "baseline_edge_fp": baseline_observed["edge_fp"],
            "baseline_edge_fn": baseline_observed["edge_fn"],
            "raw_above_10um_fn_count": len(cohort),
            "affine_below_10um_count": recovered,
            "affine_below_10um_fraction": fraction,
            "affine_implementation_gate": "PASS" if recommend_affine else "FAIL",
            "motion_axis_bounded_rule": "FAIL" if recommend_affine else "PASS",
        },
        "execution": {
            "device": "cpu",
            "detector_forward_calls": 0,
            "training": False,
            "division_enabled": False,
            "geometry_changed": False,
            "production_eligible": False,
            "submit_eligible": False,
            "internet_access": False,
            "optional_deformation_field_run": False,
            "optional_deformation_field_reason": (
                "Skipped: no dependency/provenance expansion was needed to answer the affine gate."
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": {
            "baseline_scores": {
                "path": str(args.baseline_scores),
                "sha256": _sha256(args.baseline_scores),
            },
            "baseline_fragments": {
                "path": str(args.fragments),
                "count": len(fragment_by_identity),
            },
            "raw_caches": {"path": str(args.cache), "count": len(cache_paths)},
            "raw_cohort_contract": {
                "path": str(Path(raw_gate.__file__)),
                "sha256": _sha256(Path(raw_gate.__file__)),
            },
            "split": {"path": str(args.split), "sha256": _sha256(args.split)},
            "archive": {
                "path": str(args.archive),
                "sha256": baseline.get("frozen_scorer", {})
                .get("archive", {})
                .get("sha256"),
                "bytes_rehashed": False,
            },
            "scorer_parity": scorer,
            "script": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
        },
        "per_video": per_video,
        "events": cohort,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    value.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    value.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--baseline-scores", type=Path, default=DEFAULT_SCORES)
    value.add_argument("--scorer-parity", type=Path, default=DEFAULT_PARITY)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main() -> None:
    result = run(parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "cohort": result["cohort"],
                "decision": result["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
