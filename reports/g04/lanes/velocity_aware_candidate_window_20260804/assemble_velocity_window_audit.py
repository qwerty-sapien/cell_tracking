#!/usr/bin/env python3
"""Merge bounded VELOCITY-AWARE-CANDIDATE-WINDOW-v1 CPU audit partials."""

from __future__ import annotations

from collections import Counter
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_velocity_window_audit as audit  # noqa: E402


def load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise audit.AuditError(f"partial is not an object: {path}")
    return value


def main(args: argparse.Namespace) -> None:
    output = args.output
    final_paths = (output / "AUDIT.json", output / "SHELL_EDGES.csv", output / "RUN_MANIFEST.json")
    if any(path.exists() for path in final_paths):
        raise audit.AuditError("refusing to overwrite a completed audit")
    partial_paths = sorted(output.glob("PARTIAL_V3_*.json"))
    if not partial_paths:
        raise audit.AuditError("no partial audit files found")
    partials = [load(path) for path in partial_paths]
    if any(value.get("schema") != "celltrack-alpha.velocity-aware-candidate-window-partial.v1" for value in partials):
        raise audit.AuditError("partial schema drifted")

    g1 = audit.load_json(args.g1_report)
    selected = tuple(str(value) for value in g1["selection"]["identities"])
    base = audit.load_json(args.base_report)
    if selected != tuple(str(value) for value in base["selection"]["video_ids"]):
        raise audit.AuditError("G1 selection differs from canonical CV57")
    seen: list[str] = []
    census = audit.load_json(args.census)
    archive_identity = dict(census.get("archive", {}))
    if archive_identity != {
        "bytes": 87393127165,
        "central_directory_sha256": "055b38ffa73594321f857e5bde7120a7b0236fbeee661f18214f4d910c8b00be",
        "path": "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    }:
        raise audit.AuditError("census raw-archive identity drifted")
    expected_hashes = {
        "g1_report": audit.sha256(args.g1_report),
        "census": audit.sha256(args.census),
        "base_report": audit.sha256(args.base_report),
        "archive_identity": archive_identity,
    }
    totals: Counter[str] = Counter()
    families: dict[str, Counter[str]] = {"44b6": Counter(), "6bba": Counter()}
    shell_rows: list[dict[str, str]] = []
    distances: list[float] = []
    prior_speeds: list[float] = []
    prediction_residuals: list[float] = []
    constant_residuals: list[float] = []
    directions: list[float] = []
    flagged = fallback_failures = peak_g1 = peak_velocity = 0
    execution_seconds = 0.0
    for partial in partials:
        bounds = partial.get("selection_bounds", {})
        batch = tuple(str(value) for value in partial.get("selection", []))
        start, stop = int(bounds.get("start", -1)), int(bounds.get("stop", -1))
        if batch != selected[start:stop] or stop <= start:
            raise audit.AuditError("partial selection ordering/bounds drifted")
        if dict(partial.get("input_hashes", {})) != expected_hashes:
            raise audit.AuditError("partial input hashes differ from current frozen inputs")
        seen.extend(batch)
        totals.update({str(key): int(value) for key, value in partial["totals"].items()})
        for name in families:
            families[name].update({str(key): int(value) for key, value in partial["per_family"][name].items()})
        shell_rows.extend(dict(row) for row in partial["shell_rows"])
        distances.extend(map(float, partial["shell_distances"]))
        prior_speeds.extend(map(float, partial["shell_prior_speeds"]))
        prediction_residuals.extend(map(float, partial["shell_prediction_residuals"]))
        constant_residuals.extend(map(float, partial["shell_constant_residuals"]))
        directions.extend(map(float, partial["shell_direction_cosines"]))
        flagged += int(partial["flagged_baseline_candidates"])
        fallback_failures += int(partial["fallback_equivalence_failures"])
        peak_g1 = max(peak_g1, int(partial["peak_g1_neighbors"]))
        peak_velocity = max(peak_velocity, int(partial["peak_velocity_neighbors"]))
        execution_seconds += float(partial["execution"]["elapsed_seconds"])
    if tuple(seen) != selected:
        raise audit.AuditError("partials do not cover the exact CV57 selection once in order")
    if flagged != audit.EXPECTED_G1_APNCC_FLAGGED_CANDIDATES:
        raise audit.AuditError(f"G1 APNCC candidate replay mismatch: {flagged}")
    if len(shell_rows) != audit.EXPECTED_SHELL_EDGES:
        raise audit.AuditError(f"shell count mismatch: {len(shell_rows)}")
    if any(row["global_g1_candidate"] != "0" for row in shell_rows):
        raise audit.AuditError("shell row unexpectedly belongs to the G1 r10/cap8 pool")
    if fallback_failures:
        raise audit.AuditError(f"fallback candidate-set mismatch: {fallback_failures}")

    shell = Counter()
    for row in shell_rows:
        shell["total"] += 1
        shell["previous_available"] += int(row["source_previous_available"])
        shell["prediction_inside"] += int(row["persistence_prediction_inside_r10"])
        shell["retained"] += int(row["velocity_window_candidate_after_cap8"])
        shell["cap8_excluded"] += int(
            row["persistence_prediction_inside_r10"] == "1" and row["velocity_window_candidate_after_cap8"] == "0"
        )
    tp, fp, fn = audit.G1_COUNTS
    newly = int(totals["newly_reachable_true_continuations"])
    raw_ceiling = newly / float(tp + fp + fn)
    burden = float(totals["velocity_background_candidates"]) / float(totals["g1_background_candidates"])
    gate = {
        "newly_reachable_true_continuations": {"observed": newly, "minimum": 20, "passes": newly >= 20},
        "no_new_fp_raw_edge_ceiling": {
            "formula": "N / (34389 + 2308 + 1879)", "observed": raw_ceiling, "minimum": 0.0005,
            "passes": raw_ceiling >= 0.0005,
        },
        "background_burden_multiplier": {"observed": burden, "maximum": 1.05, "passes": burden <= 1.05},
        "shell_retained_after_cap8": {"observed": int(shell["retained"]), "minimum": 20, "passes": int(shell["retained"]) >= 20},
    }
    eligible = all(bool(row["passes"]) for row in gate.values())
    summary = g1["scores"]["registered_g1"]["summary"]
    control = (float(summary["adj_edge_jaccard"]), float(summary["edge_jaccard"]), float(summary["node_recall"]))
    if control != (0.8867644202595942, 0.891461012028204, 0.990377056057473):
        raise audit.AuditError("G1 control vector drifted")
    result: dict[str, Any] = {
        "schema": audit.SCHEMA,
        "status": "CEILING_PASSES_SCORE_ELIGIBILITY" if eligible else "COMPLETED_NO_SCORE",
        "mechanism": {
            "name": "VELOCITY-AWARE-CANDIDATE-WINDOW-v1", "new_named_mechanism": True,
            "not_radius_widening": True, "not_dead_shell_rescreen": True,
            "radius_um": audit.RADIUS_UM, "neighbor_cap": audit.CAP,
            "published_median_step_um_per_frame": audit.PUBLISHED_MEDIAN_STEP_UM,
            "lag1_directional_persistence": audit.PERSISTENCE,
            "formula": "qhat[i,t+1] = q[i,t] + 0.30 * (q[i,t] - q[parent,t-1])",
            "fallback": "qhat[i,t+1] = q[i,t] when no selected G1 parent at t-1 exists",
            "coordinate_convention": {
                "candidate_geometry": "(original_voxel_coordinate + G1_cumulative_raw_shift) * physical_scale",
                "physical_scale_zyx_um": list(audit.PHYSICAL_SCALE_ZYX),
                "output_and_evaluator_coordinates": "original APNCC fragment coordinates; unchanged",
                "image_and_feature_sampling": "original; untouched", "global_transform_applications": 1,
            },
        },
        "control": {
            "name": "global-jump-corrected APNCC", "adjusted_edge_jaccard": control[0],
            "raw_edge_jaccard": control[1], "node_recall": control[2],
            "edge_tp_fp_fn": {"tp": tp, "fp": fp, "fn": fn},
            "fragment_aggregate_sha256": g1["scores"]["registered_g1"]["aggregate_sha256"],
        },
        "residual_shell": {
            "definition": "GT-matched adjacent continuation with global-jump-corrected physical distance in (10,12] um",
            "edge_count": len(shell_rows), "by_family": dict(Counter(row["family"] for row in shell_rows)),
            "prior_velocity_available": int(shell["previous_available"]),
            "persistence_prediction_inside_r10": int(shell["prediction_inside"]),
            "retained_after_fixed_cap8": int(shell["retained"]),
            "inside_r10_but_cap8_excluded": int(shell["cap8_excluded"]),
            "global_corrected_distance_um": audit.summary(distances),
            "prior_speed_um_per_frame": audit.summary(prior_speeds),
            "lag1_direction_cosine": audit.summary(directions),
            "constant_velocity_residual_um": audit.summary(constant_residuals),
            "persistence_prediction_residual_um": audit.summary(prediction_residuals),
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
            "by_family": {name: dict(values) for name, values in families.items()},
            "all_source_queries": {
                "with_lag1_parent": int(totals["sources_with_prior"]),
                "fallback_to_g1_center": int(totals["sources_without_prior"]),
                "fallback_candidate_set_mismatches": fallback_failures,
            },
            "peak_uncapped_neighbors": {"global_jump_apncc": peak_g1, "velocity_aware_window": peak_velocity},
        },
        "pre_scorer_gate": {
            "gate": gate, "verdict": "SCORE_AUTHORIZED" if eligible else "NO_SCORE",
            "reason": "all pre-registered ceiling gates pass" if eligible else "at least one pre-registered ceiling gate fails",
        },
        "validation": {
            "canonical_cv57_identities": len(selected), "fragment_count": len(list(args.fragments.glob("train__*.csv"))),
            "partial_files": [{"path": str(path), "sha256": audit.sha256(path)} for path in partial_paths],
            "g1_pool_breaking_candidate_replay": {"observed": flagged, "expected": audit.EXPECTED_G1_APNCC_FLAGGED_CANDIDATES, "passes": True},
            "residual_shell_edge_count": {"observed": len(shell_rows), "expected": audit.EXPECTED_SHELL_EDGES, "passes": True},
            "shell_g1_r10_exclusion": True, "fallback_exact_equivalence": True, "scorer_or_graph_called": False,
        },
        "provenance": {
            "g1_report": {"path": str(args.g1_report), "sha256": expected_hashes["g1_report"]},
            "g1_fragments_directory": str(args.fragments),
            "census": {"path": str(args.census), "sha256": expected_hashes["census"]},
            "base_report": {"path": str(args.base_report), "sha256": expected_hashes["base_report"]},
            "archive": archive_identity,
            "audit_script": {"path": str(HERE / "run_velocity_window_audit.py"), "sha256": audit.sha256(HERE / "run_velocity_window_audit.py")},
            "assembler_script": {"path": str(Path(__file__).resolve()), "sha256": audit.sha256(Path(__file__).resolve())},
            "preregistration": {"path": str(output / "PREREGISTRATION.md"), "sha256": audit.sha256(output / "PREREGISTRATION.md")},
        },
        "execution": {
            "single_thread_environment_required": ["OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"],
            "partial_elapsed_seconds_sum": execution_seconds, "scorer_called": False, "graph_called": False,
            "kaggle_action": False, "runpod_action": False, "submission_action": False,
        },
    }
    with final_paths[1].open("x", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(shell_rows)
    audit.write_new_json(final_paths[0], result)
    manifest = {
        "schema": "celltrack-alpha.velocity-aware-candidate-window-run-manifest.v1",
        "audit_sha256": audit.sha256(final_paths[0]), "shell_edges_csv_sha256": audit.sha256(final_paths[1]),
        "preregistration_sha256": audit.sha256(output / "PREREGISTRATION.md"),
        "audit_script_sha256": audit.sha256(HERE / "run_velocity_window_audit.py"),
        "assembler_script_sha256": audit.sha256(Path(__file__).resolve()),
        "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv313/bin/python assemble_velocity_window_audit.py",
    }
    audit.write_new_json(final_paths[2], manifest)
    print(json.dumps({"status": result["status"], "gate": result["pre_scorer_gate"]}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=HERE)
    value.add_argument("--fragments", type=Path, default=HERE.parent / "data_pathology_repair/score/registered_g1/fragments")
    value.add_argument("--g1-report", type=Path, default=HERE.parent / "data_pathology_repair/score/REPORT.json")
    value.add_argument("--census", type=Path, default=HERE.parent / "data_pathology_repair/CENSUS.json")
    value.add_argument("--base-report", type=Path, default=audit.ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/report.json")
    value.add_argument("--zip", type=Path, default=audit.ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip")
    return value


if __name__ == "__main__":
    main(parser().parse_args())
