#!/usr/bin/env python3
"""Deterministic H0 audit of banked proposal proxies versus graph deltas.

This script deliberately contains a small, provenance-explicit evidence table.
It does not open prediction fragments, model artifacts, validation12, or CV57.
Every numeric input below is transcribed from an already-banked report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


SEED = 20260725
BOOTSTRAP_REPLICATES = 20_000
RHO_THRESHOLD = 0.5
PROXY_ORDER = ("precision", "utility_gain", "auc", "support")


PRIMARY_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "GA-RGSWAP-20260724",
        "outcome_id": "GA-RGSWAP-20260724/top10",
        "mechanism_cluster": "relative_geometry_swap",
        "mechanism": "competitor-relative no-link/edge promotion gate",
        "unit": "candidate",
        "graph_scope": "registered CV57 top-10 screen (banked report value only)",
        "graph_adjusted_delta": -0.001633610,
        "graph_raw_delta": -0.0017211703958691649,
        "graph_raw_derivation": {
            "control_tp_fp_fn": [5122, 1125, 725],
            "candidate_tp_fp_fn": [5110, 1125, 737],
            "formula": "TP / (TP + FP + FN), candidate minus control",
        },
        "proxy_screen_predicted_positive": True,
        "proxies": {
            "precision": 3776 / 3777,
            "support": 3777,
        },
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/screening/relative_geometry_swap_gate_v1.md",
                "field": "Registered top-10 result / adjusted edge Jaccard / delta",
                "value_text": "-0.001633610",
            },
            "graph_raw_delta": {
                "path": "reports/g02/screening/relative_geometry_swap_gate_v1.md",
                "field": "Registered top-10 result / edge TP, FP, FN; exact scorer raw-edge-Jaccard derivation",
                "value_text": "5110/(5110+1125+737) - 5122/(5122+1125+725)",
            },
            "precision": {
                "path": "reports/g02/screening/relative_geometry_swap_gate_v1.md",
                "field": "fit-domain validation result; derived exactly as 3,776 correct / (3,776 correct + 1 harmful)",
                "value_text": "3776/3777",
            },
            "support": {
                "path": "reports/g02/screening/relative_geometry_swap_gate_v1.md",
                "field": "fit-domain validation accepted decisions; 3,776 correct + 1 harmful",
                "value_text": "3777",
            },
        },
    },
    {
        "candidate_id": "GA-APNCC-DIRECT-20260724",
        "outcome_id": "GA-APNCC-DIRECT-20260724/full-cv57",
        "mechanism_cluster": "apncc",
        "mechanism": "appearance-aware direct swap",
        "unit": "candidate",
        "graph_scope": "full CV57 confirmation (banked report value only)",
        "graph_adjusted_delta": 0.008372476430831166,
        "graph_raw_delta": 0.008093907370849918,
        "graph_raw_derivation": {
            "control_tp_fp_fn": [34205, 2685, 2063],
            "candidate_tp_fp_fn": [34320, 2459, 1948],
            "formula": "TP / (TP + FP + FN), candidate minus control",
        },
        "proxy_screen_predicted_positive": True,
        "proxies": {
            "precision": 1.0,
            "support": 38,
        },
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/screening/selected_graph_appearance_direct_v1.md",
                "field": "Frozen comparison / change",
                "value_text": "+0.008372476430831166",
            },
            "graph_raw_delta": {
                "path": "reports/g02/screening/selected_graph_appearance_direct_v1.md",
                "field": "Frozen comparison / edge TP, FP, FN; exact scorer raw-edge-Jaccard derivation",
                "value_text": "34320/(34320+2459+1948) - 34205/(34205+2685+2063)",
            },
            "precision": {
                "path": "reports/g02/screening/selected_graph_appearance_direct_v1.md",
                "field": "geometry + NCC + node-disjoint cap on validation / precision",
                "value_text": "1.000000",
            },
            "support": {
                "path": "reports/g02/screening/selected_graph_appearance_direct_v1.md",
                "field": "geometry + NCC + node-disjoint cap on validation / accepted",
                "value_text": "38",
            },
        },
    },
    {
        "candidate_id": "GA-SGBG-20260724",
        "outcome_id": "GA-SGBG-20260724/top10",
        "mechanism_cluster": "selected_graph_background",
        "mechanism": "deployment-background-regularized selected-graph gate",
        "unit": "candidate",
        "graph_scope": "registered CV57 top-10 screen (banked report value only)",
        "graph_adjusted_delta": -0.000861814,
        "graph_raw_delta": -0.0008877679604212263,
        "graph_raw_derivation": {
            "control_tp_fp_fn": [5117, 1133, 730],
            "candidate_tp_fp_fn": [5113, 1136, 734],
            "formula": "TP / (TP + FP + FN), candidate minus control",
        },
        "proxy_screen_predicted_positive": True,
        "proxies": {
            "precision": 1.0,
            "support": 10,
        },
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/screening/selected_graph_background_gate_v1.md",
                "field": "Exact top-10 arm / Delta",
                "value_text": "-0.000861814",
            },
            "graph_raw_delta": {
                "path": "reports/g02/screening/selected_graph_background_gate_v1.md",
                "field": "Exact top-10 arm / TP, FP, FN; exact scorer raw-edge-Jaccard derivation",
                "value_text": "5113/(5113+1136+734) - 5117/(5117+1133+730)",
            },
            "precision": {
                "path": "reports/g02/screening/selected_graph_background_gate_v1.md",
                "field": "labeled validation: 10 accepted, 10 correct, 0 harmful",
                "value_text": "10/10",
            },
            "support": {
                "path": "reports/g02/screening/selected_graph_background_gate_v1.md",
                "field": "labeled validation accepted",
                "value_text": "10",
            },
        },
    },
    {
        "candidate_id": "CTRL-OOF-GRAPH-20260724",
        "outcome_id": "CTRL-OOF-GRAPH-20260724/held-fit69",
        "mechanism_cluster": "temporal_6bba_ranker",
        "mechanism": "6bba temporal family ranker",
        "unit": "candidate",
        "graph_scope": "69-video five-fold held-fit reconstructed graph",
        "graph_adjusted_delta": -0.000612091023178704,
        "graph_raw_delta": -0.0006408361424446913,
        "proxy_screen_predicted_positive": True,
        "proxies": {
            "precision": 0.9648497198166073,
            "utility_gain": 313,
            "support": 1963,
        },
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/screening/temporal_rank_oof_graph_gate_v1.md",
                "field": "temporal minus APNCC / fit69 adjusted edge Jaccard",
                "value_text": "-0.0006120910",
            },
            "graph_raw_delta": {
                "path": "reports/g02/screening/temporal_rank_oof_graph_gate_v1.md",
                "field": "temporal minus APNCC / raw edge Jaccard",
                "value_text": "-0.0006408361",
            },
            "precision": {
                "path": "reports/g02/screening/temporal_rank_oof_graph_gate_v1.md",
                "field": "proposal-only prerequisite / precision",
                "value_text": "0.9648497",
            },
            "utility_gain": {
                "path": "reports/g02/screening/selected_graph_temporal_family_rank_gate_v1.md",
                "field": "6bba OOF temporal utility 1,756 minus pair-NCC control utility 1,443",
                "value_text": "+313",
            },
            "support": {
                "path": "reports/g02/screening/temporal_rank_oof_graph_gate_v1.md",
                "field": "proposal-only prerequisite / accepted rescues (1,894 correct + 69 harmful)",
                "value_text": "1963",
            },
        },
    },
)


REPEATED_OUTCOMES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "GA-APNCC-GLOBAL-20260724",
        "outcome_id": "GA-APNCC-GLOBAL-20260724/top10",
        "mechanism_cluster": "apncc",
        "mechanism": "appearance-aware global score integration",
        "unit": "repeated_outcome",
        "graph_scope": "registered CV57 top-10 screen (banked report value only)",
        "graph_adjusted_delta": 0.006597192755873338,
        "graph_raw_delta": 0.006712388370419764,
        "proxy_screen_predicted_positive": True,
        "proxies": {"precision": 1.0, "support": 38},
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/selected_graph_appearance_screen10_v1/report.json",
                "field": "decision.delta_adjusted_edge_jaccard",
                "value_text": "+0.006597192755873338",
            },
            "graph_raw_delta": {
                "path": "reports/g02/selected_graph_appearance_screen10_v1/report.json",
                "field": "candidate.summary.edge_jaccard - baseline.summary.edge_jaccard",
                "value_text": "0.7398069442443452 - 0.7330945558739255",
            },
            "proxy": {
                "path": "reports/g02/screening/selected_graph_appearance_direct_v1.md",
                "field": "shared banked geometry + NCC + node-disjoint validation screen",
                "value_text": "38/38, support 38",
            },
        },
    },
    {
        "candidate_id": "GA-APNCC-GLOBAL-20260724",
        "outcome_id": "GA-APNCC-GLOBAL-20260724/full-cv57",
        "mechanism_cluster": "apncc",
        "mechanism": "appearance-aware global score integration",
        "unit": "repeated_outcome",
        "graph_scope": "full CV57 confirmation (banked report value only)",
        "graph_adjusted_delta": 0.00425128475063874,
        "graph_raw_delta": 0.004238882455891324,
        "proxy_screen_predicted_positive": True,
        "proxies": {"precision": 1.0, "support": 38},
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/selected_graph_appearance_cv57_v1/report.json",
                "field": "decision.delta_adjusted_edge_jaccard",
                "value_text": "+0.00425128475063874",
            },
            "graph_raw_delta": {
                "path": "reports/g02/selected_graph_appearance_cv57_v1/report.json",
                "field": "candidate.summary.edge_jaccard - baseline.summary.edge_jaccard",
                "value_text": "0.8823483990528158 - 0.8781095165969245",
            },
            "proxy": {
                "path": "reports/g02/screening/selected_graph_appearance_direct_v1.md",
                "field": "shared banked geometry + NCC + node-disjoint validation screen",
                "value_text": "38/38, support 38",
            },
        },
    },
    {
        "candidate_id": "GA-APNCC-DIRECT-20260724",
        "outcome_id": "GA-APNCC-DIRECT-20260724/top10",
        "mechanism_cluster": "apncc",
        "mechanism": "appearance-aware direct swap",
        "unit": "repeated_outcome",
        "graph_scope": "registered CV57 top-10 screen (banked report value only)",
        "graph_adjusted_delta": 0.014685680601916551,
        "graph_raw_delta": 0.014619190678549443,
        "proxy_screen_predicted_positive": True,
        "proxies": {"precision": 1.0, "support": 38},
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/selected_graph_appearance_direct_screen10_v3/report.json",
                "field": "decision.delta_adjusted_edge_jaccard",
                "value_text": "+0.014685680601916551",
            },
            "graph_raw_delta": {
                "path": "reports/g02/selected_graph_appearance_direct_screen10_v3/report.json",
                "field": "candidate.summary.edge_jaccard - baseline.summary.edge_jaccard",
                "value_text": "0.7477137465524749 - 0.7330945558739255",
            },
            "proxy": {
                "path": "reports/g02/screening/selected_graph_appearance_direct_v1.md",
                "field": "shared banked geometry + NCC + node-disjoint validation screen",
                "value_text": "38/38, support 38",
            },
        },
    },
    {
        "candidate_id": "GA-TEMPHYBRID-20260724",
        "outcome_id": "GA-TEMPHYBRID-20260724/full-cv57",
        "mechanism_cluster": "temporal_6bba_ranker",
        "mechanism": "44b6 APNCC + 6bba temporal family ranker",
        "unit": "repeated_outcome",
        "graph_scope": "full CV57 transfer outcome (banked report value only)",
        "graph_adjusted_delta": -0.0014080676020108,
        "graph_raw_delta": -0.0014977843174481809,
        "graph_raw_derivation": {
            "control_tp_fp_fn": [34320, 2459, 1948],
            "candidate_tp_fp_fn": [34308, 2511, 1960],
            "formula": "TP / (TP + FP + FN), candidate minus control",
        },
        "proxy_screen_predicted_positive": True,
        "proxies": {
            "precision": 0.9648497198166073,
            "utility_gain": 313,
            "support": 1963,
        },
        "sources": {
            "graph_adjusted_delta": {
                "path": "reports/g02/screening/selected_graph_temporal_6bba_hybrid_cv57_v1.md",
                "field": "hybrid minus banked / CV57",
                "value_text": "-0.0014080676020108",
            },
            "graph_raw_delta": {
                "path": "reports/g02/screening/selected_graph_temporal_6bba_hybrid_cv57_v1.md",
                "field": "candidate/control TP, FP, FN; exact scorer raw-edge-Jaccard derivation",
                "value_text": "34308/(34308+2511+1960) - 34320/(34320+2459+1948)",
            },
            "proxy": {
                "path": "reports/g02/screening/selected_graph_temporal_family_rank_gate_v1.md",
                "field": "shared 6bba temporal OOF proxy",
                "value_text": "precision 0.9648497; utility gain +313; support 1963",
            },
        },
    },
)


TEMPORAL_FOLD_SENSITIVITY: tuple[dict[str, Any], ...] = (
    {
        "fold": 0,
        "graph_adjusted_delta": -0.00039635532614179514,
        "graph_raw_delta": -0.0004367024204227077,
        "precision": 0.991130820399113,
        "utility_gain": 124,
        "support": 451,
        "proxy_screen_predicted_positive": True,
    },
    {
        "fold": 1,
        "graph_adjusted_delta": -0.00022204305592343765,
        "graph_raw_delta": -0.00026145723522574826,
        "precision": 0.9772727272727273,
        "utility_gain": 92,
        "support": 484,
        "proxy_screen_predicted_positive": True,
    },
    {
        "fold": 2,
        "graph_adjusted_delta": 0.0000590946252868596,
        "graph_raw_delta": 0.000016618575904514543,
        "precision": 0.9481981981981982,
        "utility_gain": 52,
        "support": 444,
        "proxy_screen_predicted_positive": False,
    },
    {
        "fold": 3,
        "graph_adjusted_delta": -0.0012879840952172428,
        "graph_raw_delta": -0.0013025526340039173,
        "precision": 0.9633699633699634,
        "utility_gain": 31,
        "support": 273,
        "proxy_screen_predicted_positive": True,
    },
    {
        "fold": 4,
        "graph_adjusted_delta": -0.001850614843304177,
        "graph_raw_delta": -0.0018542433995581975,
        "precision": 0.932475884244373,
        "utility_gain": 14,
        "support": 311,
        "proxy_screen_predicted_positive": False,
    },
)


EXCLUSIONS: tuple[dict[str, str], ...] = (
    {
        "ids": "GA-POINTCTX-PILOT-20260724; GA-POINTCTX-OOF-20260724; GA-POINTCTX-THRESH-20260724",
        "reason": "Proposal metrics exist, but exact graph scoring was not entered.",
        "graph_delta_status": "absent, not zero",
        "sources": (
            "reports/g02/screening/point_context_full_pool_oof_gate_v1.md; "
            "reports/g02/screening/point_context_oof_threshold_headroom_v1.md; LEDGER.md"
        ),
    },
    {
        "ids": "GA-CROPLINK-PILOT-20260724; GA-CROPLINK-6BBA-OOF-20260724",
        "reason": "Precision/utility/support exist, but no exact graph output was produced.",
        "graph_delta_status": "absent, not zero",
        "sources": "reports/g02/screening/crop_linker_6bba_oof_v1.md; LEDGER.md",
    },
    {
        "ids": "GA-SCENCAL-20260724",
        "reason": "Proposal utility/precision exist; the report explicitly says no predictions were emitted and no graph replay occurred.",
        "graph_delta_status": "absent, not zero",
        "sources": "reports/g02/screening/selected_graph_scenario_calibration_v1.md; LEDGER.md",
    },
    {
        "ids": "GA-TRACKASTRA-QUAL-20260724",
        "reason": "ROC AUC 0.918491 (gain +0.077265) is banked, but the row is ranking-only with no graph or prediction CSV.",
        "graph_delta_status": "absent, not zero",
        "sources": "reports/g02/screening/trackastra_pretrained_qualification_v1.md; LEDGER.md",
    },
    {
        "ids": "GA-SGCONFLICT-20260724",
        "reason": "Proposal precision/support exist, but the support gate failed before score.",
        "graph_delta_status": "absent, not zero",
        "sources": "reports/g02/screening/selected_graph_conflict_gate_v1.md; LEDGER.md",
    },
    {
        "ids": "GA-SP-20260724; GA-B2-20260724; GD-T985-20260724",
        "reason": "Graph deltas exist, but no comparable proposal precision/utility-gain/AUC/support proxy is banked for the scored identity.",
        "graph_delta_status": "measured; proxy side absent",
        "sources": (
            "reports/g02/screening/association_cost_screen_v1.md; "
            "reports/g02/screening/node_threshold_screen_v1.md"
        ),
    },
    {
        "ids": "GA-APNCC-POLICY-20260724",
        "reason": (
            "Adjusted/raw graph deltas are measured, but the nine added proposals are "
            "an observation-difference count, not labeled proposal support, precision, "
            "utility gain, or AUC; treating nine as support would mix definitions."
        ),
        "graph_delta_status": "measured; comparable proxy side absent",
        "sources": "reports/g02/screening/appearance_direct_policy_fit_v1.md; LEDGER.md",
    },
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = _mean(x), _mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            sx = (x[left] > x[right]) - (x[left] < x[right])
            sy = (y[left] > y[right]) - (y[left] < y[right])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                ties_x += 1
            elif sy == 0:
                ties_y += 1
            elif sx == sy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def linear_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, float] | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = _mean(x), _mean(y)
    denominator = sum((value - mx) ** 2 for value in x)
    if denominator == 0:
        return None
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / denominator
    intercept = my - slope * mx
    fitted = [intercept + slope * value for value in x]
    total = sum((value - my) ** 2 for value in y)
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(y, fitted))
    r_squared = 1 - residual / total if total else 1.0
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    statistic,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any] | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rng = random.Random(seed)
    sampled_statistics: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(len(x)) for _ in range(len(x))]
        estimate = statistic([x[index] for index in indices], [y[index] for index in indices])
        if estimate is not None and math.isfinite(estimate):
            sampled_statistics.append(estimate)
    if not sampled_statistics:
        return None
    return {
        "lower": percentile(sampled_statistics, 0.025),
        "upper": percentile(sampled_statistics, 0.975),
        "valid_replicates": len(sampled_statistics),
        "requested_replicates": replicates,
        "method": "paired nonparametric percentile bootstrap",
    }


def sign_agreement(predicted_positive: Sequence[bool], graph_delta: Sequence[float]) -> float:
    if len(predicted_positive) != len(graph_delta) or not graph_delta:
        raise ValueError("sign-agreement inputs must be nonempty and paired")
    agreements = [
        predicted == (delta > 0)
        for predicted, delta in zip(predicted_positive, graph_delta)
    ]
    return sum(agreements) / len(agreements)


def bootstrap_sign_ci(
    predicted_positive: Sequence[bool],
    graph_delta: Sequence[float],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(len(graph_delta)) for _ in range(len(graph_delta))]
        values.append(
            sign_agreement(
                [predicted_positive[index] for index in indices],
                [graph_delta[index] for index in indices],
            )
        )
    return {
        "lower": percentile(values, 0.025),
        "upper": percentile(values, 0.975),
        "valid_replicates": len(values),
        "requested_replicates": replicates,
        "method": "paired nonparametric percentile bootstrap",
    }


def bootstrap_cluster_ci(
    rows: Sequence[dict[str, Any]],
    proxy: str,
    outcome_key: str,
    statistic,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any] | None:
    clusters = sorted({str(row["mechanism_cluster"]) for row in rows})
    if len(clusters) < 2:
        return None
    by_cluster = {
        cluster: [
            row
            for row in rows
            if str(row["mechanism_cluster"]) == cluster
            and proxy in row["proxies"]
            and row.get(outcome_key) is not None
        ]
        for cluster in clusters
    }
    eligible_clusters = [cluster for cluster in clusters if by_cluster[cluster]]
    if len(eligible_clusters) < 2:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_clusters = [
            eligible_clusters[rng.randrange(len(eligible_clusters))]
            for _ in eligible_clusters
        ]
        sampled_rows = [
            row for cluster in sampled_clusters for row in by_cluster[cluster]
        ]
        estimate = statistic(
            [float(row["proxies"][proxy]) for row in sampled_rows],
            [float(row[outcome_key]) for row in sampled_rows],
        )
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None
    return {
        "lower": percentile(estimates, 0.025),
        "upper": percentile(estimates, 0.975),
        "valid_replicates": len(estimates),
        "requested_replicates": replicates,
        "method": "mechanism-cluster nonparametric percentile bootstrap",
    }


def bootstrap_cluster_sign_ci(
    rows: Sequence[dict[str, Any]],
    proxy: str,
    outcome_key: str,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any] | None:
    clusters = sorted(
        {
            str(row["mechanism_cluster"])
            for row in rows
            if proxy in row["proxies"] and row.get(outcome_key) is not None
        }
    )
    if len(clusters) < 2:
        return None
    by_cluster = {
        cluster: [
            row
            for row in rows
            if str(row["mechanism_cluster"]) == cluster
            and proxy in row["proxies"]
            and row.get(outcome_key) is not None
        ]
        for cluster in clusters
    }
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_clusters = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sampled_rows = [
            row for cluster in sampled_clusters for row in by_cluster[cluster]
        ]
        estimates.append(
            sign_agreement(
                [bool(row["proxy_screen_predicted_positive"]) for row in sampled_rows],
                [float(row[outcome_key]) for row in sampled_rows],
            )
        )
    return {
        "lower": percentile(estimates, 0.025),
        "upper": percentile(estimates, 0.975),
        "valid_replicates": len(estimates),
        "requested_replicates": replicates,
        "method": "mechanism-cluster nonparametric percentile bootstrap",
    }


def analyze_clustered_rows(
    rows: Iterable[dict[str, Any]],
    *,
    outcome_key: str,
) -> dict[str, Any]:
    materialized = list(rows)
    result: dict[str, Any] = {}
    for proxy_index, proxy in enumerate(PROXY_ORDER):
        paired = [
            row
            for row in materialized
            if proxy in row["proxies"] and row.get(outcome_key) is not None
        ]
        clusters = sorted({str(row["mechanism_cluster"]) for row in paired})
        x = [float(row["proxies"][proxy]) for row in paired]
        y = [float(row[outcome_key]) for row in paired]
        flags = [bool(row["proxy_screen_predicted_positive"]) for row in paired]
        entry: dict[str, Any] = {
            "row_n": len(paired),
            "cluster_n": len(clusters),
            "clusters": clusters,
            "outcomes": [str(row["outcome_id"]) for row in paired],
            "eligible_for_preregistered_decision": False,
        }
        if len(paired) < 3 or len(clusters) < 3:
            entry.update(
                {
                    "status": "INSUFFICIENT_INDEPENDENT_CLUSTERS",
                    "minimum_clusters": 3,
                    "pearson": None,
                    "spearman_rho": None,
                    "kendall_tau_b": None,
                    "linear_fit": None,
                    "sign_agreement": None,
                }
            )
            result[proxy] = entry
            continue
        base_seed = SEED + 300_000 + proxy_index * 10_000
        entry.update(
            {
                "status": "SENSITIVITY_ONLY",
                "pearson": {
                    "estimate": pearson(x, y),
                    "cluster_ci95": bootstrap_cluster_ci(
                        paired, proxy, outcome_key, pearson, seed=base_seed + 1
                    ),
                },
                "spearman_rho": {
                    "estimate": spearman(x, y),
                    "cluster_ci95": bootstrap_cluster_ci(
                        paired, proxy, outcome_key, spearman, seed=base_seed + 2
                    ),
                },
                "kendall_tau_b": {
                    "estimate": kendall_tau_b(x, y),
                    "cluster_ci95": bootstrap_cluster_ci(
                        paired, proxy, outcome_key, kendall_tau_b, seed=base_seed + 3
                    ),
                },
                "linear_fit": linear_fit(x, y),
                "sign_agreement": {
                    "estimate": sign_agreement(flags, y),
                    "cluster_ci95": bootstrap_cluster_sign_ci(
                        paired, proxy, outcome_key, seed=base_seed + 4
                    ),
                },
            }
        )
        result[proxy] = entry
    return result


def analyze_rows(
    rows: Iterable[dict[str, Any]],
    *,
    outcome_key: str = "graph_adjusted_delta",
    sensitivity: bool = False,
) -> dict[str, Any]:
    materialized = list(rows)
    result: dict[str, Any] = {}
    for proxy_index, proxy in enumerate(PROXY_ORDER):
        paired: list[tuple[float, float, bool, str]] = []
        for row in materialized:
            proxies = row.get("proxies", row)
            if proxy not in proxies or row.get(outcome_key) is None:
                continue
            identity = (
                f"fold-{row['fold']}" if sensitivity else str(row["candidate_id"])
            )
            paired.append(
                (
                    float(proxies[proxy]),
                    float(row[outcome_key]),
                    bool(row["proxy_screen_predicted_positive"]),
                    identity,
                )
            )
        x = [item[0] for item in paired]
        y = [item[1] for item in paired]
        flags = [item[2] for item in paired]
        entry: dict[str, Any] = {
            "n": len(paired),
            "units": [item[3] for item in paired],
            "eligible_for_preregistered_decision": not sensitivity and len(paired) >= 3,
        }
        if len(paired) < 3:
            entry.update(
                {
                    "status": "INSUFFICIENT_N",
                    "minimum_n": 3,
                    "pearson": None,
                    "spearman_rho": None,
                    "kendall_tau_b": None,
                    "linear_fit": None,
                    "sign_agreement": None,
                }
            )
            result[proxy] = entry
            continue
        base_seed = SEED + proxy_index * 10_000 + (100_000 if sensitivity else 0)
        p = pearson(x, y)
        s = spearman(x, y)
        k = kendall_tau_b(x, y)
        entry.update(
            {
                "status": "ESTIMATED",
                "pearson": {
                    "estimate": p,
                    "ci95": bootstrap_ci(x, y, pearson, seed=base_seed + 1),
                },
                "spearman_rho": {
                    "estimate": s,
                    "ci95": bootstrap_ci(x, y, spearman, seed=base_seed + 2),
                },
                "kendall_tau_b": {
                    "estimate": k,
                    "ci95": bootstrap_ci(x, y, kendall_tau_b, seed=base_seed + 3),
                },
                "linear_fit": linear_fit(x, y),
                "sign_agreement": {
                    "definition": (
                        "banked proposal screen pass predicts graph delta > 0; "
                        "graph delta == 0 would count as non-positive"
                    ),
                    "estimate": sign_agreement(flags, y),
                    "ci95": bootstrap_sign_ci(flags, y, seed=base_seed + 4),
                },
            }
        )
        result[proxy] = entry
    return result


def decide(primary_analysis: dict[str, Any]) -> dict[str, Any]:
    clearing: list[dict[str, Any]] = []
    for proxy in PROXY_ORDER:
        entry = primary_analysis[proxy]
        rho = entry.get("spearman_rho")
        if not entry["eligible_for_preregistered_decision"] or not rho:
            continue
        estimate = rho["estimate"]
        if estimate is not None and abs(estimate) >= RHO_THRESHOLD:
            clearing.append(
                {
                    "proxy": proxy,
                    "spearman_rho": estimate,
                    "ci95": rho["ci95"],
                    "n": entry["n"],
                }
            )
    if clearing:
        return {
            "verdict": "RECOVER_PROXY_FAST_FAIL",
            "formal_text": (
                "At least one independent mechanism/identity-cluster proxy clears the "
                "preregistered absolute Spearman-rho threshold. Recover only the "
                "listed proxy/proxies as fast-fails, with the reported bootstrap error bar."
            ),
            "clearing_proxies": clearing,
        }
    return {
        "verdict": "PROPOSAL_ONLY_SCREENING_BANNED",
        "formal_text": (
            "Proposal-only screening BANNED; route candidates straight to the "
            "44b6-first held-fit graph gate."
        ),
        "clearing_proxies": [],
    }


def build_report() -> dict[str, Any]:
    primary = analyze_rows(PRIMARY_CANDIDATES, outcome_key="graph_adjusted_delta")
    primary_raw = analyze_rows(PRIMARY_CANDIDATES, outcome_key="graph_raw_delta")
    complete_outcomes = list(PRIMARY_CANDIDATES) + list(REPEATED_OUTCOMES)
    complete_adjusted = analyze_clustered_rows(
        complete_outcomes,
        outcome_key="graph_adjusted_delta",
    )
    complete_raw = analyze_clustered_rows(
        complete_outcomes,
        outcome_key="graph_raw_delta",
    )
    sensitivity_rows = [
        {
            **row,
            "proxies": {proxy: row[proxy] for proxy in ("precision", "utility_gain", "support")},
        }
        for row in TEMPORAL_FOLD_SENSITIVITY
    ]
    sensitivity = analyze_rows(
        sensitivity_rows,
        outcome_key="graph_adjusted_delta",
        sensitivity=True,
    )
    sensitivity_raw = analyze_rows(
        sensitivity_rows,
        outcome_key="graph_raw_delta",
        sensitivity=True,
    )
    report = {
        "schema": "h0_proxy_validity_v1",
        "status": "COMPLETE",
        "determinism": {
            "seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": 0.95,
        },
        "scope": {
            "data_source": "banked report values only",
            "models_rerun": 0,
            "training_runs": 0,
            "gpu_hours": 0,
            "kaggle_submissions": 0,
            "validation12_or_cv57_artifact_access": False,
            "primary_unit": "one scored representative per mechanism/identity cluster",
            "rho_for_preregistered_decision": "Spearman rank correlation",
            "minimum_n_for_correlation": 3,
            "rho_threshold": RHO_THRESHOLD,
        },
        "primary_representative_policy": {
            "one_row_per_mechanism_identity_cluster": True,
            "relative_geometry_swap": "only comparable scored outcome",
            "selected_graph_background": "only comparable scored outcome",
            "apncc": (
                "final banked direct-swap full-CV57 realization; the global integration "
                "and both top-10 outcomes are repeated realization/scope checks"
            ),
            "temporal_6bba_ranker": (
                "held-fit69 objective-first graph gate; GA-TEMPHYBRID is a later "
                "transfer outcome using the same model/proxy"
            ),
        },
        "primary_corpus": list(PRIMARY_CANDIDATES),
        "primary_proxy_graph_pair_count": sum(
            len(candidate["proxies"]) for candidate in PRIMARY_CANDIDATES
        ),
        "primary_candidate_count": len(PRIMARY_CANDIDATES),
        "primary_analysis": primary,
        "complete_outcome_inventory": {
            "role": (
                "complete-row, cluster-aware sensitivity; repeated graph outcomes "
                "sharing one proxy/model do not increase independent mechanism n"
            ),
            "row_count": len(complete_outcomes),
            "mechanism_cluster_count": len(
                {row["mechanism_cluster"] for row in complete_outcomes}
            ),
            "rows": complete_outcomes,
            "adjusted_analysis": complete_adjusted,
            "raw_analysis": complete_raw,
        },
        "raw_outcome_analysis": {
            "role": (
                "secondary outcome; adjusted graph delta remains the preregistered "
                "decision outcome"
            ),
            "definition": (
                "scorer raw edge Jaccard = TP/(TP+FP+FN); never an observation-arm "
                "prediction Jaccard"
            ),
            "available_candidate_count": sum(
                candidate["graph_raw_delta"] is not None
                for candidate in PRIMARY_CANDIDATES
            ),
            "unavailable_candidates": [
                candidate["candidate_id"]
                for candidate in PRIMARY_CANDIDATES
                if candidate["graph_raw_delta"] is None
            ],
            "analysis": primary_raw,
        },
        "temporal_fold_sensitivity": {
            "role": (
                "within-one-mechanism sensitivity only; folds are correlated and are "
                "not independent candidate evidence or eligible for the decision"
            ),
            "source_paths": [
                "reports/g02/temporal_rank_oof_graph_gate_v1/report.json",
                "reports/g02/selected_graph_temporal_family_rank_gate_v1/report.json",
            ],
            "source_fields": {
                "graph": "graph.per_fold[].decision.adjusted_edge_jaccard_delta",
                "graph_raw": "graph.per_fold[].decision.raw_edge_jaccard_delta",
                "precision": "selected.families.6bba.folds[].held.precision",
                "utility_gain": (
                    "selected.families.6bba.folds[].held.conservative_utility - "
                    "control.families.6bba.folds[].held.conservative_utility"
                ),
                "support": "selected.families.6bba.folds[].held.accepted_labeled",
            },
            "rows": sensitivity_rows,
            "analysis": sensitivity,
            "raw_analysis": sensitivity_raw,
        },
        "exclusions": list(EXCLUSIONS),
        "inventory_rescan": {
            "scope": "banked screening Markdown/JSON summaries only",
            "comparable_additions_found": [
                "GA-APNCC-GLOBAL-20260724/top10",
                "GA-APNCC-GLOBAL-20260724/full-cv57",
                "GA-APNCC-DIRECT-20260724/top10",
                "GA-TEMPHYBRID-20260724/full-cv57",
            ],
            "other_comparable_pairs_found": 0,
            "note": (
                "Other proposal-report examples lack a measured graph outcome; other "
                "graph outcomes lack a comparable proposal precision, utility gain, "
                "AUC, or support value for the scored identity."
            ),
        },
        "decision": decide(primary),
        "limitations": [
            "Only four mechanism/identity clusters have comparable precision/support and graph evidence.",
            "Utility gain has only one independent candidate and AUC has none with a graph delta, so neither can be correlation-tested.",
            "The primary bootstrap is very small-sample and tie-heavy; its intervals are descriptive error bars, not asymptotic guarantees.",
            "The complete-row inventory contains eight graph outcomes but only four mechanism/identity clusters; naïve complete-row ranks overweight APNCC and temporal repeats.",
            "The temporal fold sensitivity can expose within-mechanism behavior but cannot establish cross-candidate proxy validity.",
            "Raw scorer edge Jaccard is explicit for CTRL-OOF-GRAPH and exactly derivable from banked scorer TP/FP/FN for the other three primary candidates.",
        ],
    }
    return report


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _metric_summary(entry: dict[str, Any]) -> str:
    if entry["status"] != "ESTIMATED":
        return f"n={entry['n']}; insufficient (minimum 3)"
    rho = entry["spearman_rho"]
    ci = rho["ci95"]
    return (
        f"n={entry['n']}; ρ={_format_number(rho['estimate'])}; "
        f"95% bootstrap CI [{_format_number(ci['lower'])}, "
        f"{_format_number(ci['upper'])}]"
    )


def _cluster_metric_summary(entry: dict[str, Any], key: str) -> str:
    if entry["status"] != "SENSITIVITY_ONLY":
        return (
            f"rows={entry['row_n']}, clusters={entry['cluster_n']}; "
            "insufficient independent clusters"
        )
    metric = entry[key]
    ci = metric["cluster_ci95"]
    return (
        f"rows={entry['row_n']}, clusters={entry['cluster_n']}; "
        f"{_format_number(metric['estimate'])} "
        f"[{_format_number(ci['lower'])}, {_format_number(ci['upper'])}]"
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# H0 proxy-validity audit — v1",
        "",
        f"**Verdict: {report['decision']['formal_text']}**",
        "",
        "This deterministic audit used only numbers already banked in reports. It ran "
        "no model, training, scorer replay, GPU job, submission, or prediction-artifact "
        "read. The preregistered `rho` is **Spearman rank correlation** on one scored "
        "representative per mechanism/identity cluster; the recovery threshold is "
        "`|rho| >= 0.5`.",
        "",
        "## Primary independent-cluster corpus",
        "",
        f"The honest primary corpus has **{report['primary_candidate_count']} mechanism/"
        f"identity clusters and {report['primary_proxy_graph_pair_count']} populated "
        "proxy/adjusted-graph pairs**, not an imputed ~15 candidates. Missing graph "
        "deltas are absent, never encoded as zero.",
        "",
        "| candidate | graph adjusted delta | precision | utility gain | support | graph scope |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for candidate in report["primary_corpus"]:
        proxies = candidate["proxies"]
        lines.append(
            "| {candidate_id} | {graph} | {precision} | {utility} | {support} | {scope} |".format(
                candidate_id=candidate["candidate_id"],
                graph=_format_number(candidate["graph_adjusted_delta"], 9),
                precision=_format_number(proxies.get("precision"), 7),
                utility=_format_number(proxies.get("utility_gain")),
                support=_format_number(proxies.get("support")),
                scope=candidate["graph_scope"],
            )
        )
    lines.extend(
        [
            "",
            "The RGSWAP precision and support are exact arithmetic from the banked "
            "`3,776 correct / 1 harmful` counts. All other displayed primary values are "
            "direct report fields. The primary chooses one prespecified representative "
            "per cluster: the final banked APNCC direct full realization and the "
            "objective-first temporal held-fit69 graph gate; later/top-10 realizations "
            "remain in the complete inventory. Source paths, fields, and this selection "
            "policy are frozen in the companion JSON.",
            "",
            "## Complete measured-outcome inventory",
            "",
            "The complete inventory contains every comparable banked outcome found in "
            "the rescan: **8 graph rows in 4 mechanism/identity clusters**. "
            "`GA-TEMPHYBRID` repeats the temporal proxy/model on CV57. "
            "`GA-APNCC-GLOBAL` and the APNCC top-10/direct outcomes share the same "
            "banked 38/38 proposal screen. These rows are reported, but their repeats "
            "do not inflate independent mechanism n.",
            "",
            "| outcome | cluster | adjusted delta | raw delta | precision | utility gain | support |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["complete_outcome_inventory"]["rows"]:
        proxies = row["proxies"]
        lines.append(
            f"| {row['outcome_id']} | {row['mechanism_cluster']} | "
            f"{_format_number(row['graph_adjusted_delta'], 9)} | "
            f"{_format_number(row['graph_raw_delta'], 9)} | "
            f"{_format_number(proxies.get('precision'), 7)} | "
            f"{_format_number(proxies.get('utility_gain'))} | "
            f"{_format_number(proxies.get('support'))} |"
        )
    lines.extend(
        [
            "",
            "Complete-row point estimates and **mechanism-cluster bootstrap** intervals "
            "are sensitivity statistics only:",
            "",
            "| outcome / proxy | Pearson | Spearman ρ | Kendall τ-b | sign agreement |",
            "|---|---|---|---|---|",
        ]
    )
    complete = report["complete_outcome_inventory"]
    for outcome_label, analysis in (
        ("adjusted", complete["adjusted_analysis"]),
        ("raw", complete["raw_analysis"]),
    ):
        for proxy in ("precision", "support"):
            entry = analysis[proxy]
            sign = entry["sign_agreement"]
            sign_ci = sign["cluster_ci95"]
            sign_text = (
                f"rows={entry['row_n']}, clusters={entry['cluster_n']}; "
                f"{_format_number(sign['estimate'])} "
                f"[{_format_number(sign_ci['lower'])}, "
                f"{_format_number(sign_ci['upper'])}]"
            )
            lines.append(
                f"| {outcome_label} / {proxy} | "
                f"{_cluster_metric_summary(entry, 'pearson')} | "
                f"{_cluster_metric_summary(entry, 'spearman_rho')} | "
                f"{_cluster_metric_summary(entry, 'kendall_tau_b')} | "
                f"{sign_text} |"
            )
    lines.extend(
        [
            "",
            "Naïve complete-row Spearman magnitudes cross 0.5 because four APNCC rows "
            "repeat one positive mechanism and two temporal rows repeat one negative "
            "mechanism. The cluster bootstrap intervals remain broad. The locked "
            "decision therefore uses the one-representative-per-cluster primary corpus "
            "below, not these repeated rows.",
            "",
            "## Independent-cluster results",
            "",
            "| proxy | n | Pearson r (95% bootstrap CI) | Spearman ρ (95% bootstrap CI) | Kendall τ-b (95% bootstrap CI) | sign agreement (95% bootstrap CI) |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for proxy in PROXY_ORDER:
        entry = report["primary_analysis"][proxy]
        if entry["status"] != "ESTIMATED":
            lines.append(f"| {proxy} | {entry['n']} | insufficient | insufficient | insufficient | insufficient |")
            continue
        cells: list[str] = []
        for key in ("pearson", "spearman_rho", "kendall_tau_b"):
            metric = entry[key]
            ci = metric["ci95"]
            cells.append(
                f"{_format_number(metric['estimate'])} "
                f"[{_format_number(ci['lower'])}, {_format_number(ci['upper'])}]"
            )
        sign = entry["sign_agreement"]
        sign_ci = sign["ci95"]
        sign_text = (
            f"{_format_number(sign['estimate'])} "
            f"[{_format_number(sign_ci['lower'])}, {_format_number(sign_ci['upper'])}]"
        )
        lines.append(
            f"| {proxy} | {entry['n']} | {cells[0]} | {cells[1]} | {cells[2]} | {sign_text} |"
        )
    lines.extend(
        [
            "",
            "Sign agreement asks whether a banked positive proposal-screen outcome "
            "correctly predicted `graph delta > 0`. Only one of the four cluster-level "
            "proposal passes had a positive graph delta. No primary proxy reaches "
            "`|Spearman rho| >= 0.5`; utility gain and AUC are untestable at the "
            "independent-cluster level because their paired n is below 3.",
            "",
            "## Raw scorer-edge-Jaccard outcome",
            "",
            "Adjusted graph delta above is the preregistered decision outcome. This "
            "secondary analysis uses scorer raw edge Jaccard only: `TP/(TP+FP+FN)`. "
            "For RGSWAP, APNCC-direct, and SGBG it is derived exactly from the banked "
            "scorer confusion tuples; CTRL-OOF-GRAPH reports it explicitly. "
            "Prediction-arm observation Jaccards are not used.",
            "",
            "| candidate | raw edge-Jaccard delta | provenance |",
            "|---|---:|---|",
        ]
    )
    for candidate in report["primary_corpus"]:
        provenance = (
            "explicit report field"
            if "graph_raw_derivation" not in candidate
            else "exactly derived from banked scorer TP/FP/FN"
        )
        lines.append(
            f"| {candidate['candidate_id']} | "
            f"{_format_number(candidate['graph_raw_delta'], 9)} | {provenance} |"
        )
    lines.extend(
        [
            "",
            "| proxy | n | Pearson r (95% bootstrap CI) | Spearman ρ (95% bootstrap CI) | Kendall τ-b (95% bootstrap CI) | sign agreement (95% bootstrap CI) |",
            "|---|---:|---|---|---|---|",
        ]
    )
    raw_analysis = report["raw_outcome_analysis"]["analysis"]
    for proxy in PROXY_ORDER:
        entry = raw_analysis[proxy]
        if entry["status"] != "ESTIMATED":
            lines.append(
                f"| {proxy} | {entry['n']} | insufficient | insufficient | "
                "insufficient | insufficient |"
            )
            continue
        cells = []
        for key in ("pearson", "spearman_rho", "kendall_tau_b"):
            metric = entry[key]
            ci = metric["ci95"]
            cells.append(
                f"{_format_number(metric['estimate'])} "
                f"[{_format_number(ci['lower'])}, {_format_number(ci['upper'])}]"
            )
        sign = entry["sign_agreement"]
        sign_ci = sign["ci95"]
        sign_text = (
            f"{_format_number(sign['estimate'])} "
            f"[{_format_number(sign_ci['lower'])}, {_format_number(sign_ci['upper'])}]"
        )
        lines.append(
            f"| {proxy} | {entry['n']} | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {sign_text} |"
        )
    lines.extend(
        [
            "",
            "The raw outcome has the same sign for all four candidates. No raw-outcome "
            "proxy reaches the preregistered `|Spearman rho| >= 0.5` threshold; the "
            "formal adjusted-outcome verdict therefore does not change.",
            "",
            "## Temporal-fold sensitivity (not independent candidate evidence)",
            "",
            "The temporal graph report is the only dense paired source. Its five folds "
            "supply 15 repeated proxy/graph pairs (precision, utility gain, and support "
            "for each fold). They are correlated slices of one mechanism and therefore "
            "cannot promote a campaign-wide fast-fail.",
            "",
            "| proxy | sensitivity result |",
            "|---|---|",
        ]
    )
    sensitivity = report["temporal_fold_sensitivity"]["analysis"]
    for proxy in ("precision", "utility_gain", "support"):
        lines.append(
            f"| {proxy} | adjusted: {_metric_summary(sensitivity[proxy])}; "
            f"raw: {_metric_summary(report['temporal_fold_sensitivity']['raw_analysis'][proxy])} |"
        )
    lines.extend(
        [
            "",
            "Utility gain and support each have point `rho = 0.6` inside this one "
            "temporal mechanism, but the bootstrap intervals are wide and the result "
            "does not replicate across independent candidates. The measured "
            "`GA-TEMPHYBRID` adjusted/raw deltas (`-0.0014080676 / -0.0014977843`) "
            "are included in the complete inventory under the same temporal cluster, "
            "not counted as another independent candidate.",
            "",
            "## Exclusions and missingness",
            "",
            "| ledger id(s) | graph-delta status | reason |",
            "|---|---|---|",
        ]
    )
    for exclusion in report["exclusions"]:
        lines.append(
            f"| {exclusion['ids']} | {exclusion['graph_delta_status']} | {exclusion['reason']} |"
        )
    lines.extend(
        [
            "",
            "In particular, Trackastra AUC, POINTCTX precision, CROPLINK precision/"
            "utility/support, and SCENCAL utility cannot be paired with graph delta "
            "because those graph evaluations do not exist. They were not treated as "
            "failures or zeros.",
            "",
            "## Formal decision",
            "",
            f"**{report['decision']['formal_text']}**",
            "",
            "No proposal proxy is recovered as a legitimate fast-fail. H1 and later "
            "candidates must use the locked objective-first order: data/leakage checks, "
            "then the 44b6-first held-fit graph gate.",
            "",
            "## Reproducibility",
            "",
            f"- Seed: `{report['determinism']['seed']}`.",
            f"- Paired percentile bootstrap replicates: `{report['determinism']['bootstrap_replicates']}`.",
            "- Correlations: Pearson, tie-aware Spearman, and Kendall tau-b.",
            "- Primary unit: one scored representative per mechanism/identity cluster.",
            "- Temporal folds: sensitivity only; excluded from the formal decision.",
            "- GPU-hours / submissions / training runs: `0 / 0 / 0`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/g02/screening/h0_proxy_validity_v1.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/g02/screening/h0_proxy_validity_v1.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
