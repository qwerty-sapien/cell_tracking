#!/usr/bin/env python3
"""Independent, fail-closed audit of the G-02 subsample-fidelity evidence.

This program reads banked JSON receipts only. It never reads candidate fragment CSVs,
never invokes the graph scorer, and never changes the candidate universe to make
incompatible results line up.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SHA256 = {
    "reports/g02/screening/obs_diff_selected_graph_appearance_direct_cv57_v1.json":
        "83d32d49f869fa85fd85be9adca12bded29f33cb377392a140d9500ee3305ca6",
    "reports/g02/selected_graph_appearance_direct_cv57_v1/report.json":
        "8da96a3da9e66869192f446fc6d59fedd6016bdefe4fb272b136df68258a0df4",
    "reports/g02/screening/top50_parallel_dags/branches/c1_ta_hoct_lite/"
    "outputs/c1_n3_44b6_graph.json":
        "6a202b209c9c1b7851ca1e43e4c67e5dc3772e78426fdc267f1c01cfe5953a10",
    "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json":
        "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d",
    "reports/g02/screening/tiered_surface/branches/sa_delete/graph_result.json":
        "28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a",
    "reports/g02/screening/tiered_surface/branches/sa_swap/"
    "result_44b6_postrepair_v1/report.json":
        "62953d270ae383c9bacdea9ab33a3f6980358a31a329b219447460abd5604864",
    "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gate/RESULT_44B6.json":
        "fb9202586242f506a7cd8437d780bc93ba1fcfe5de957e22de8d6a2e73f94834",
    "reports/g02/screening/B2_decomposition.md":
        "2ad2809246b5508f4682800437a434215599d49e1f45a833f0b14b78fdbd441e",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py":
        "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py":
        "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
}

KS = (3, 5, 8, 12)
SCHEMES = ("simple_random", "residual_top10", "family_residual")
DRAW_COUNT = 10_000
SCORE_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(root: Path, relative_path: str) -> dict[str, Any]:
    with (root / relative_path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def family_of(identity: str) -> str:
    dataset = identity.split(":", 1)[-1]
    return dataset.split("_", 1)[0]


def sufficient_stats(row: dict[str, Any]) -> dict[str, float]:
    edge_union = float(row["edge_tp"] + row["edge_fp"] + row["edge_fn"])
    return {
        "adjusted_edge_numerator": edge_union * float(row["adj_edge_jaccard"]),
        "adjusted_edge_denominator": edge_union,
        "division_tp": float(row["division_tp"]),
        "division_fp": float(row["division_fp"]),
        "division_fn": float(row["division_fn"]),
        "residual_mass": float(row["edge_fp"] + row["edge_fn"]),
    }


def score_from_rows(
    rows: Iterable[dict[str, Any]],
    design_weights: Iterable[float] | None = None,
) -> float:
    rows_list = list(rows)
    weights = (
        [1.0] * len(rows_list)
        if design_weights is None
        else [float(value) for value in design_weights]
    )
    if len(weights) != len(rows_list) or not rows_list:
        raise ValueError("rows and design weights must have equal non-zero length")

    stats = [sufficient_stats(row) for row in rows_list]
    adj_num = sum(
        weight * item["adjusted_edge_numerator"]
        for weight, item in zip(weights, stats)
    )
    adj_den = sum(
        weight * item["adjusted_edge_denominator"]
        for weight, item in zip(weights, stats)
    )
    if adj_den <= 0:
        raise ValueError("adjusted-edge denominator is not positive")
    adjusted_edge = adj_num / adj_den

    div_tp = sum(weight * item["division_tp"] for weight, item in zip(weights, stats))
    div_fp = sum(weight * item["division_fp"] for weight, item in zip(weights, stats))
    div_fn = sum(weight * item["division_fn"] for weight, item in zip(weights, stats))
    div_den = div_tp + div_fp + div_fn
    return adjusted_edge if div_den == 0 else adjusted_edge + 0.1 * div_tp / div_den


def candidate_record(
    *,
    candidate_id: str,
    source_path: str,
    reference_id: str,
    reference_score: float,
    candidate_score: float,
    identities: list[str],
    baseline_rows: list[dict[str, Any]] | None,
    candidate_rows: list[dict[str, Any]] | None,
    aggregate_only_reason: str | None = None,
) -> dict[str, Any]:
    sorted_ids = sorted(identities)
    duplicate_ids = sorted(
        identity for identity in set(identities) if identities.count(identity) > 1
    )
    full_delta = float(candidate_score - reference_score)
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_path": source_path,
        "source_sha256": EXPECTED_SHA256[source_path],
        "reference_id": reference_id,
        "reference_score": float(reference_score),
        "candidate_score": float(candidate_score),
        "banked_full_delta": full_delta,
        "video_count": len(identities),
        "unique_video_count": len(set(identities)),
        "duplicate_video_ids": duplicate_ids,
        "video_ids": sorted_ids,
        "video_universe_sha256": canonical_sha256(sorted_ids),
        "families": dict(sorted(
            {
                family: sum(family_of(identity) == family for identity in sorted_ids)
                for family in {family_of(identity) for identity in sorted_ids}
            }.items()
        )),
        "per_video_sufficient_statistics_banked": (
            baseline_rows is not None and candidate_rows is not None
        ),
        "per_video_stat_row_count": (
            len(candidate_rows) if candidate_rows is not None else 0
        ),
        "reconstruction": None,
        "eligible_on_own_banked_universe": False,
        "ineligibility_reasons": [],
    }

    if duplicate_ids:
        record["ineligibility_reasons"].append("duplicate_video_ids")
    if baseline_rows is None or candidate_rows is None:
        record["ineligibility_reasons"].append(
            aggregate_only_reason or "per_video_sufficient_statistics_absent"
        )
        return record
    if len(baseline_rows) != len(identities) or len(candidate_rows) != len(identities):
        record["ineligibility_reasons"].append("per_video_row_count_mismatch")
        return record

    reconstructed_reference = score_from_rows(baseline_rows)
    reconstructed_candidate = score_from_rows(candidate_rows)
    reconstructed_delta = reconstructed_candidate - reconstructed_reference
    errors = {
        "reference_score_abs_error": abs(reconstructed_reference - reference_score),
        "candidate_score_abs_error": abs(reconstructed_candidate - candidate_score),
        "delta_abs_error": abs(reconstructed_delta - full_delta),
    }
    record["reconstruction"] = {
        "reference_score": reconstructed_reference,
        "candidate_score": reconstructed_candidate,
        "delta": reconstructed_delta,
        **errors,
        "tolerance": SCORE_TOLERANCE,
        "passes": max(errors.values()) <= SCORE_TOLERANCE,
    }
    if not record["reconstruction"]["passes"]:
        record["ineligibility_reasons"].append("full_score_reconstruction_failed")
    record["eligible_on_own_banked_universe"] = not record["ineligibility_reasons"]
    return record


def audit_sources(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ap_path = (
        "reports/g02/screening/"
        "obs_diff_selected_graph_appearance_direct_cv57_v1.json"
    )
    ap = load_json(root, ap_path)
    ap_a = ap["frozen_score"]["a"]
    ap_b = ap["frozen_score"]["b"]
    ap_ids = [row["identity"] for row in ap_a["samples"]]

    c1_path = (
        "reports/g02/screening/top50_parallel_dags/branches/"
        "c1_ta_hoct_lite/outputs/c1_n3_44b6_graph.json"
    )
    c1 = load_json(root, c1_path)
    c1_ids = [row["identity"] for row in c1["graph"]["videos"]]

    h1_path = "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json"
    h1 = load_json(root, h1_path)
    h1_ids = [row["identity"] for row in h1["graph"]["videos"]]

    delete_path = (
        "reports/g02/screening/tiered_surface/branches/"
        "sa_delete/graph_result.json"
    )
    delete = load_json(root, delete_path)
    delete_family = delete["family_results"]["44b6"]
    delete_video_rows = delete_family["videos"]
    delete_ids = [row["identity"] for row in delete_video_rows]

    swap_path = (
        "reports/g02/screening/tiered_surface/branches/sa_swap/"
        "result_44b6_postrepair_v1/report.json"
    )
    swap = load_json(root, swap_path)
    swap_ids = [row["identity"] for row in swap["graph"]["videos"]]

    node_path = (
        "reports/g02/screening/tiered_surface/branches/"
        "sa_nodetopo_gate/RESULT_44B6.json"
    )
    node = load_json(root, node_path)
    node_ids = [row["identity"] for row in node["per_video"]]

    candidates = [
        candidate_record(
            candidate_id="APNCC",
            source_path=ap_path,
            reference_id="CV57_C0P25_BASELINE",
            reference_score=ap_a["summary"]["score"],
            candidate_score=ap_b["summary"]["score"],
            identities=ap_ids,
            baseline_rows=ap_a["samples"],
            candidate_rows=ap_b["samples"],
        ),
        candidate_record(
            candidate_id="C1",
            source_path=c1_path,
            reference_id="APNCC_44B6_CONTROL",
            reference_score=c1["graph"]["apncc"]["summary"]["score"],
            candidate_score=c1["graph"]["c1"]["summary"]["score"],
            identities=c1_ids,
            baseline_rows=None,
            candidate_rows=None,
            aggregate_only_reason=(
                "aggregate_scores_and_fragment_provenance_only;"
                "per_video_official_scorer_rows_absent"
            ),
        ),
        candidate_record(
            candidate_id="H1-ZWRFEAT",
            source_path=h1_path,
            reference_id="APNCC_44B6_CONTROL",
            reference_score=h1["graph"]["control"]["summary"]["score"],
            candidate_score=h1["graph"]["candidate"]["summary"]["score"],
            identities=h1_ids,
            baseline_rows=None,
            candidate_rows=None,
            aggregate_only_reason=(
                "aggregate_scores_and_fragment_provenance_only;"
                "per_video_official_scorer_rows_absent"
            ),
        ),
        candidate_record(
            candidate_id="SA-DELETE",
            source_path=delete_path,
            reference_id="CG0_44B6_FOLD0_FIT_CONTROL",
            reference_score=delete_family["control"]["score"],
            candidate_score=delete_family["candidate"]["score"],
            identities=delete_ids,
            baseline_rows=[row["baseline_metrics"] for row in delete_video_rows],
            candidate_rows=[row["candidate_metrics"] for row in delete_video_rows],
        ),
        candidate_record(
            candidate_id="SA-SWAP",
            source_path=swap_path,
            reference_id="APNCC_44B6_CONTROL",
            reference_score=swap["graph"]["control"]["summary"]["score"],
            candidate_score=swap["graph"]["alpha"]["summary"]["score"],
            identities=swap_ids,
            baseline_rows=None,
            candidate_rows=None,
            aggregate_only_reason=(
                "aggregate_scores_and_fragment_provenance_only;"
                "per_video_official_scorer_rows_absent"
            ),
        ),
        candidate_record(
            candidate_id="NODETOPO",
            source_path=node_path,
            reference_id="CG0_44B6_CONTROL",
            reference_score=node["baseline"]["score"],
            candidate_score=node["candidate"]["score"],
            identities=node_ids,
            baseline_rows=[row["baseline_metrics"] for row in node["per_video"]],
            candidate_rows=[row["candidate_metrics"] for row in node["per_video"]],
        ),
    ]

    by_id = {record["candidate_id"]: record for record in candidates}
    sets = {
        candidate_id: set(record["video_ids"])
        for candidate_id, record in by_id.items()
    }
    pairwise = {}
    ids = [record["candidate_id"] for record in candidates]
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            pairwise[f"{left}__{right}"] = {
                "intersection_count": len(sets[left] & sets[right]),
                "same_universe": sets[left] == sets[right],
                "left_only_count": len(sets[left] - sets[right]),
                "right_only_count": len(sets[right] - sets[left]),
            }

    delete_extra = sorted(sets["SA-DELETE"] - sets["H1-ZWRFEAT"])
    shared_36 = (
        sets["C1"] == sets["H1-ZWRFEAT"] == sets["SA-SWAP"] == sets["NODETOPO"]
    )
    universe_audit = {
        "all_six_same_universe": len(
            {record["video_universe_sha256"] for record in candidates}
        ) == 1,
        "apncc_zero_overlap_with_all_44b6_gate_universes": all(
            len(sets["APNCC"] & sets[candidate]) == 0
            for candidate in ("C1", "H1-ZWRFEAT", "SA-DELETE", "SA-SWAP", "NODETOPO")
        ),
        "c1_h1_swap_nodetopo_same_36_ids": shared_36,
        "sa_delete_is_exact_36_plus_five": (
            sets["H1-ZWRFEAT"].issubset(sets["SA-DELETE"])
            and len(delete_extra) == 5
        ),
        "sa_delete_extra_ids": delete_extra,
        "pairwise": pairwise,
        "reference_ids": {
            record["candidate_id"]: record["reference_id"] for record in candidates
        },
        "common_eligible_candidate_count": 0,
        "common_eligible_candidate_ids": [],
        "reason": (
            "No >=4-candidate set has both one unchanged banked full-result universe "
            "and banked per-video sufficient statistics. Intersecting or padding the "
            "universes would change at least one cited full-delta estimand."
        ),
    }
    return candidates, universe_audit


def validate_input_hashes(root: Path) -> list[dict[str, Any]]:
    receipts = []
    for relative_path, expected in EXPECTED_SHA256.items():
        path = root / relative_path
        observed = sha256_file(path)
        receipts.append({
            "path": relative_path,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passes": observed == expected,
        })
    return receipts


def fidelity_cells(blockers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "k": k,
            "scheme": scheme,
            "draws_requested": DRAW_COUNT,
            "draws_executed": 0,
            "status": "NOT_ESTIMABLE",
            "median_spearman_rho": None,
            "pooled_sign_agreement": None,
            "maximum_candidate_95pct_width": None,
            "usable": False,
            "blockers": blockers,
        }
        for k in KS
        for scheme in SCHEMES
    ]


def write_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    columns = [
        "k", "scheme", "draws_requested", "draws_executed", "status",
        "median_spearman_rho", "pooled_sign_agreement",
        "maximum_candidate_95pct_width", "usable", "blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cell in cells:
            row = dict(cell)
            row["blockers"] = " | ".join(cell["blockers"])
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hash_receipts = validate_input_hashes(root)
    if not all(receipt["passes"] for receipt in hash_receipts):
        raise SystemExit("frozen input hash mismatch")

    candidates, universe_audit = audit_sources(root)
    blockers = [
        "three_named_candidates_lack_banked_per_video_sufficient_statistics",
        "banked_candidate_full_delta_universes_are_not_common",
        "candidate_reference_graphs_are_not_common",
        "eligible_common_candidate_count_below_four",
        "residual_top10_reference_is_cv57_and_has_zero_id_overlap_with_44b6_gate_universe",
    ]
    cells = fidelity_cells(blockers)

    manifest = {
        "schema": "g02.subsample-fidelity.independent-source-manifest.v1",
        "status": "UNSAFE_INCOMPLETE_EVIDENCE",
        "read_policy": {
            "banked_json_receipts_only": True,
            "candidate_fragment_csvs_read": False,
            "graph_rescoring_performed": False,
            "new_family_or_cv57_opened": False,
        },
        "official_score_sufficient_stat_estimator": {
            "per_video_fields": [
                "edge_tp", "edge_fp", "edge_fn", "adj_edge_jaccard",
                "division_tp", "division_fp", "division_fn",
            ],
            "adjusted_edge_numerator_i": (
                "(edge_tp_i + edge_fp_i + edge_fn_i) * adj_edge_jaccard_i"
            ),
            "adjusted_edge_denominator_i": "edge_tp_i + edge_fp_i + edge_fn_i",
            "design_weighted_score": (
                "sum(d_i*A_i)/sum(d_i*W_i) + "
                "0.1*sum(d_i*division_tp_i)/"
                "sum(d_i*(division_tp_i+division_fp_i+division_fn_i)); "
                "drop division term when its denominator is zero"
            ),
            "per_video_delta_averaging_forbidden": True,
            "source": (
                "external/kaggle-cell-tracking-competition/src/"
                "tracking_cellmot/metrics.py::summarise"
            ),
        },
        "frozen_input_hash_receipts": hash_receipts,
        "candidates": candidates,
        "universe_audit": universe_audit,
        "b2_reference_audit": {
            "path": "reports/g02/screening/B2_decomposition.md",
            "sha256": EXPECTED_SHA256["reports/g02/screening/B2_decomposition.md"],
            "scope": "c=0.25 CV57 baseline, 57 videos, two families",
            "top10_concentration_reported": 0.52,
            "transfer_to_fit_train105_44b6_gate_universe_supported": False,
        },
        "silent_candidate_drops": [],
        "all_six_candidates_accounted_for": True,
        "primary_verdict": "UNSAFE_INCOMPLETE_EVIDENCE",
        "contract_scientifically_justified": False,
    }
    reproduction = {
        "schema": "g02.subsample-fidelity.independent-reproduction.v1",
        "status": "NOT_ESTIMABLE",
        "preregistered_draw_count": DRAW_COUNT,
        "draws_executed": 0,
        "candidate_count_required": 4,
        "eligible_common_candidate_count": 0,
        "cells": cells,
        "acceptance_rule_evaluated": False,
        "acceptance_rule_result": "NO_CELL_CAN_BE_CLASSIFIED_USABLE",
        "subsample_contract_emitted": False,
        "why_no_draws": (
            "The preregistered rank/sign metrics require a common like-for-like "
            "candidate set. Drawing after intersecting, imputing, rescoring fragments, "
            "or silently dropping named candidates would change the frozen estimand."
        ),
        "diagnostic_note": (
            "APNCC, SA-DELETE, and NODETOPO individually reconstruct their own banked "
            "full scores exactly, but their universes are 57, 41, and 36 videos and "
            "their references differ. Candidatewise independent draws cannot be paired "
            "into the preregistered across-candidate Spearman statistic without an "
            "unregistered candidate-order/seed rule."
        ),
    }

    with (out_dir / "source_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (out_dir / "validation_reproduction.json").open("w", encoding="utf-8") as stream:
        json.dump(reproduction, stream, indent=2, sort_keys=True)
        stream.write("\n")
    write_csv(out_dir / "fidelity_table.csv", cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
