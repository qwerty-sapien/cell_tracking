#!/usr/bin/env python3
"""Audit residual headroom after the frozen appearance direct-swap candidate.

This is a read-only CV57 diagnostic.  It reconciles the exact before/after
scorer rows, quantifies where residual edge errors remain, and computes how
many idealized event corrections would be required to reach a requested local
score.  It must not be used to select a per-video production rule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_bh  # noqa: E402


SCHEMA = "celltrack-alpha.appearance-direct-swap-residual-audit.v1"
EXPECTED_BASELINE = (0.8731495935289003, 34205, 2685, 2063)
EXPECTED_CANDIDATE = (0.8815220699597315, 34320, 2459, 1948)
DEFAULT_RUN = (
    ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/report.json"
)
DEFAULT_OBSERVATION = (
    ROOT
    / "reports/g02/screening/"
    "obs_diff_selected_graph_appearance_direct_cv57_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/selected_graph_appearance_direct_residual_v1.json"
)


class ResidualAuditError(RuntimeError):
    """An appearance residual-audit identity or arithmetic invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResidualAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ResidualAuditError(f"expected JSON object: {path}")
    return value


def minimum_corrections(
    *,
    edge_tp: int,
    edge_fp: int,
    edge_fn: int,
    node_adjustment: float,
    target_score: float,
) -> dict[str, int | float]:
    """Return idealized correction counts for three transparent mechanisms."""

    if (
        min(edge_tp, edge_fp, edge_fn) < 0
        or not 0.0 < node_adjustment <= 1.0
        or not 0.0 < target_score <= 1.0
    ):
        raise ResidualAuditError("invalid target-headroom inputs")
    weight = edge_tp + edge_fp + edge_fn
    target_edge = target_score / node_adjustment
    if not 0.0 < target_edge < 1.0:
        raise ResidualAuditError("target edge Jaccard is outside (0, 1)")

    fn_to_tp = max(0, math.ceil(target_edge * weight - edge_tp))
    fp_prunes = max(0, math.ceil(weight - edge_tp / target_edge))
    perfect_swaps = max(
        0,
        math.ceil((target_edge * weight - edge_tp) / (1.0 + target_edge)),
    )
    if perfect_swaps > min(edge_fp, edge_fn):
        raise ResidualAuditError("target cannot be reached by available perfect swaps")
    return {
        "target_adjusted_score": target_score,
        "fixed_node_adjustment": node_adjustment,
        "required_edge_jaccard": target_edge,
        "fn_to_tp_only": fn_to_tp,
        "fp_prune_only": fp_prunes,
        "perfect_wrong_to_right_swaps": perfect_swaps,
    }


def summarize_videos(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    run_videos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile per-video scorer rows and summarize residual concentration."""

    by_a = {str(row["identity"]): row for row in baseline}
    by_b = {str(row["identity"]): row for row in candidate}
    by_run = {str(row["identity"]): row for row in run_videos}
    if (
        len(by_a) != len(baseline)
        or len(by_b) != len(candidate)
        or len(by_run) != len(run_videos)
        or set(by_a) != set(by_b)
        or set(by_a) != set(by_run)
        or len(by_a) != 57
    ):
        raise ResidualAuditError("per-video identities do not reconcile at 57/57")

    rows: list[dict[str, Any]] = []
    for identity in sorted(by_a):
        a = by_a[identity]
        b = by_b[identity]
        proposal_count = int(by_run[identity]["proposal_count"])
        delta_tp = int(b["edge_tp"]) - int(a["edge_tp"])
        delta_fp = int(b["edge_fp"]) - int(a["edge_fp"])
        delta_fn = int(b["edge_fn"]) - int(a["edge_fn"])
        utility = delta_tp - delta_fp - delta_fn
        residual = int(b["edge_fp"]) + int(b["edge_fn"])
        rows.append(
            {
                "identity": identity,
                "family": identity.split(":", 1)[1].split("_", 1)[0],
                "proposal_count": proposal_count,
                "delta_edge_tp": delta_tp,
                "delta_edge_fp": delta_fp,
                "delta_edge_fn": delta_fn,
                "event_utility": utility,
                "candidate_edge_fp": int(b["edge_fp"]),
                "candidate_edge_fn": int(b["edge_fn"]),
                "candidate_residual": residual,
                "candidate_adjusted_edge_jaccard": float(
                    b["adj_edge_jaccard"]
                ),
            }
        )

    total_residual = sum(int(row["candidate_residual"]) for row in rows)
    total_proposals = sum(int(row["proposal_count"]) for row in rows)
    total_utility = sum(int(row["event_utility"]) for row in rows)
    residual_order = sorted(
        rows,
        key=lambda row: (
            -int(row["candidate_residual"]),
            str(row["identity"]),
        ),
    )
    concentration = {
        str(k): {
            "residual": sum(
                int(row["candidate_residual"]) for row in residual_order[:k]
            ),
            "share": (
                sum(
                    int(row["candidate_residual"])
                    for row in residual_order[:k]
                )
                / total_residual
            ),
        }
        for k in (1, 3, 5, 10, 20)
    }
    family: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "videos": 0,
            "proposals": 0,
            "event_utility": 0,
            "candidate_residual": 0,
        }
    )
    for row in rows:
        bucket = family[str(row["family"])]
        bucket["videos"] = int(bucket["videos"]) + 1
        bucket["proposals"] = int(bucket["proposals"]) + int(
            row["proposal_count"]
        )
        bucket["event_utility"] = int(bucket["event_utility"]) + int(
            row["event_utility"]
        )
        bucket["candidate_residual"] = int(
            bucket["candidate_residual"]
        ) + int(row["candidate_residual"])
    for bucket in family.values():
        bucket["residual_share"] = (
            int(bucket["candidate_residual"]) / total_residual
        )

    return {
        "video_count": len(rows),
        "helpful_videos": sum(int(row["event_utility"]) > 0 for row in rows),
        "neutral_videos": sum(int(row["event_utility"]) == 0 for row in rows),
        "harmful_videos": sum(int(row["event_utility"]) < 0 for row in rows),
        "proposal_count": total_proposals,
        "event_utility": total_utility,
        "event_utility_per_100_proposals": (
            100.0 * total_utility / total_proposals
        ),
        "candidate_residual": total_residual,
        "residual_concentration": concentration,
        "families": dict(sorted(family.items())),
        "largest_residual_videos": residual_order[:10],
        "harmful_video_rows": sorted(
            (row for row in rows if int(row["event_utility"]) < 0),
            key=lambda row: (
                int(row["event_utility"]),
                str(row["identity"]),
            ),
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run = _load(args.run_report)
    observation = _load(args.observation_report)
    if (
        run.get("schema") != "celltrack-alpha.selected-graph-direct-swap.v1"
        or observation.get("schema")
        != "celltrack-alpha.observation-distance.v1"
        or run.get("selection", {}).get("video_count") != 57
    ):
        raise ResidualAuditError("input schemas or CV57 coverage differ")

    baseline = run.get("baseline")
    candidate = run.get("candidate")
    frozen = observation.get("frozen_score")
    if (
        not isinstance(baseline, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(frozen, Mapping)
        or not isinstance(frozen.get("a"), Mapping)
        or not isinstance(frozen.get("b"), Mapping)
    ):
        raise ResidualAuditError("input score sections are missing")
    base_counts = frozen["a"]["edge_counts"]
    candidate_counts = frozen["b"]["edge_counts"]
    observed_base = (
        float(baseline["summary"]["score"]),
        int(base_counts["edge_tp"]),
        int(base_counts["edge_fp"]),
        int(base_counts["edge_fn"]),
    )
    observed_candidate = (
        float(candidate["summary"]["score"]),
        int(candidate_counts["edge_tp"]),
        int(candidate_counts["edge_fp"]),
        int(candidate_counts["edge_fn"]),
    )
    if observed_base != EXPECTED_BASELINE or observed_candidate != EXPECTED_CANDIDATE:
        raise ResidualAuditError("frozen baseline/candidate identity drifted")
    edge_jaccard = float(candidate["summary"]["edge_jaccard"])
    adjusted = float(candidate["summary"]["adj_edge_jaccard"])
    node_adjustment = adjusted / edge_jaccard
    per_video = summarize_videos(
        frozen["a"]["samples"],
        frozen["b"]["samples"],
        candidate["videos"],
    )
    if (
        per_video["event_utility"]
        != (
            (EXPECTED_CANDIDATE[1] - EXPECTED_BASELINE[1])
            - (EXPECTED_CANDIDATE[2] - EXPECTED_BASELINE[2])
            - (EXPECTED_CANDIDATE[3] - EXPECTED_BASELINE[3])
        )
        or per_video["candidate_residual"]
        != EXPECTED_CANDIDATE[2] + EXPECTED_CANDIDATE[3]
    ):
        raise ResidualAuditError("per-video totals do not reconcile")

    result = {
        "schema": SCHEMA,
        "status": "completed_read_only_diagnostic",
        "scope": {
            "video_count": 57,
            "cv57_labels_used_for_selection": False,
            "role": (
                "residual diagnosis and minimum-headroom sizing only; "
                "fit-only data must select every production rule"
            ),
        },
        "baseline": {
            "score": observed_base[0],
            "edge_tp": observed_base[1],
            "edge_fp": observed_base[2],
            "edge_fn": observed_base[3],
        },
        "candidate": {
            "score": observed_candidate[0],
            "edge_jaccard": edge_jaccard,
            "node_adjustment": node_adjustment,
            "edge_tp": observed_candidate[1],
            "edge_fp": observed_candidate[2],
            "edge_fn": observed_candidate[3],
        },
        "target_headroom": minimum_corrections(
            edge_tp=observed_candidate[1],
            edge_fp=observed_candidate[2],
            edge_fn=observed_candidate[3],
            node_adjustment=node_adjustment,
            target_score=args.target_score,
        ),
        "per_video": per_video,
        "next_mechanism_constraint": {
            "minimum_fit_only_proxy_delta": 0.005,
            "must_address": (
                "occupied-endpoint association conflicts beyond the current "
                "centroid-NCC direct-swap support"
            ),
            "preferred_observables": [
                "tracklet predecessor/successor motion consistency",
                "multi-frame appearance consistency",
                "extended-radius candidate reachability",
            ],
            "forbidden_selection_signal": (
                "CV57 video identity, residual rank, scorer label, or per-video router"
            ),
        },
        "provenance": {
            "run_report": str(args.run_report),
            "run_report_sha256": _sha256(args.run_report),
            "observation_report": str(args.observation_report),
            "observation_report_sha256": _sha256(args.observation_report),
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-report", type=Path, default=DEFAULT_RUN)
    value.add_argument(
        "--observation-report", type=Path, default=DEFAULT_OBSERVATION
    )
    value.add_argument("--target-score", type=float, default=0.9)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "target_headroom": report["target_headroom"],
                "per_video": {
                    key: report["per_video"][key]
                    for key in (
                        "helpful_videos",
                        "neutral_videos",
                        "harmful_videos",
                        "event_utility",
                        "candidate_residual",
                    )
                },
            },
            sort_keys=True,
        )
    )
