#!/usr/bin/env python3
"""Deterministic ROI audit for the G-02 alpha-funnel compute decision.

This module deliberately separates observed effects, algebraic ceilings, and
scenario assumptions.  It does not turn heterogeneous prior experiments into
an empirical success probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


REALIZATION_PRECEDENT = 0.2449


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjusted_edge_jaccard(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fp + fn)


def swap_delta(tp: int, fp: int, fn: int, beneficial: int, harmful: int = 0) -> float:
    """Score delta when a swap replaces one false edge with one true edge.

    A beneficial swap increments TP and removes one FP/FN pair.  A harmful swap
    does the inverse.  This is an accounting bound, not a reachability claim.
    """

    baseline = adjusted_edge_jaccard(tp, fp, fn)
    candidate = adjusted_edge_jaccard(
        tp + beneficial - harmful,
        fp - beneficial + harmful,
        fn - beneficial + harmful,
    )
    return candidate - baseline


def minimum_beneficial_swaps(
    tp: int, fp: int, fn: int, minimum_delta: float, harmful: int = 0
) -> int:
    for beneficial in range(harmful, fn + harmful + 1):
        if swap_delta(tp, fp, fn, beneficial, harmful) >= minimum_delta:
            return beneficial
    raise ValueError("minimum delta is not reachable within the false-negative count")


def _resolve_receipt(evidence_path: Path, raw_path: str) -> Path:
    campaign_root = evidence_path.parent.parent
    repo_root = campaign_root.parent.parent
    candidates = (
        evidence_path.parent / raw_path,
        campaign_root / raw_path,
        repo_root / raw_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(raw_path)


def verify_bound_sources(evidence_path: Path, evidence: dict[str, Any]) -> list[dict[str, str]]:
    bound: list[dict[str, str]] = []
    source_records = list(evidence["claude_scope_audit"]["sources"])
    source_records.append(evidence["score_evidence"]["source"])
    for precedent in evidence["relevant_learned_graph_precedents"]:
        source_records.append(precedent["source"])
    substrate = evidence.get("substrate_and_reuse_audit", {})
    for key in ("prior_feature_bank", "peft_capacity_probe"):
        if isinstance(substrate.get(key), dict) and isinstance(substrate[key].get("source"), dict):
            source_records.append(substrate[key]["source"])

    for record in source_records:
        resolved = _resolve_receipt(evidence_path, record["path"])
        observed = sha256_file(resolved)
        if observed != record["sha256"]:
            raise ValueError(
                f"source hash mismatch for {record['path']}: "
                f"expected {record['sha256']}, observed {observed}"
            )
        bound.append({"path": record["path"], "sha256": observed})
    return bound


def compute_analysis(evidence: dict[str, Any], source_verification: list[dict[str, str]]) -> dict[str, Any]:
    score = evidence["score_evidence"]
    baseline = score["baseline"]
    tp, fp, fn = baseline["tp"], baseline["fp"], baseline["fn"]
    minimum_delta = score["minimum_effect"]
    swaps_available = score["perfect_repair_classes"]["swap"]

    required_by_harm = {
        str(harmful): minimum_beneficial_swaps(tp, fp, fn, minimum_delta, harmful)
        for harmful in (0, 1, 2, 5, 10, 20)
    }
    precision_by_harm = {
        harmful: required_by_harm[harmful] / (required_by_harm[harmful] + int(harmful))
        for harmful in required_by_harm
    }
    required_net = minimum_beneficial_swaps(tp, fp, fn, minimum_delta)

    precedents = evidence["relevant_learned_graph_precedents"]
    precedent_deltas = [float(row["adjusted_delta"]) for row in precedents]
    full_gate_successes = sum(bool(row["cleared_full_gate"]) for row in precedents)
    observed_worst = min(precedent_deltas)
    observed_median = statistics.median(precedent_deltas)

    swap_ceiling = 0.05459519603569707
    conditional_best = swap_ceiling * REALIZATION_PRECEDENT
    probabilities = evidence["scenario_probability_assumptions"]
    sensitivity: dict[str, dict[str, float]] = {}
    for name in ("worst", "base", "best_before_target_learnability_pilot"):
        probability = float(probabilities[name])
        expected_score_delta = probability * conditional_best + (1 - probability) * observed_median
        sensitivity[name] = {
            "assumed_probability_of_delta_gt_0_005": probability,
            "conditional_success_delta": conditional_best,
            "conditional_failure_delta": observed_median,
            "expected_score_delta": expected_score_delta,
        }

    break_even_positive = (0.0 - observed_median) / (conditional_best - observed_median)
    break_even_gate = (minimum_delta - observed_median) / (conditional_best - observed_median)

    prices = evidence["provider_prices_usd_per_gpu_hour"]
    substrate = evidence.get("substrate_and_reuse_audit", {})
    cost_table: dict[str, dict[str, float]] = {}
    for hours in (3, 6, 8):
        cost_table[str(hours)] = {
            "rtx_a6000": hours * float(prices["rtx_a6000"]),
            "a100_80gb": hours * float(prices["a100_80gb"]),
        }

    return {
        "schema": "alpha-funnel.f2b-roi-analysis.v1",
        "study_id": evidence["study_id"],
        "source_verification": source_verification,
        "claude_execution_audit": {
            "requested_movies": evidence["claude_scope_audit"]["requested_population_movies"],
            "launched_movies": evidence["claude_scope_audit"]["launched_population_movies"],
            "requested_artifact_count": len(evidence["claude_scope_audit"]["requested_artifacts"]),
            "executed_payload": evidence["claude_scope_audit"]["launched_exporter_payload"],
            "still_requires_model_forward": evidence["claude_scope_audit"]["launched_receipt_says_still_requires_model_forward"],
            "scientific_outcome": evidence["claude_scope_audit"]["scientific_outcome"],
            "candidate_scores_opened": evidence["claude_scope_audit"]["candidate_scores_opened"],
        },
        "effect_arithmetic": {
            "baseline_adjusted_edge_jaccard": adjusted_edge_jaccard(tp, fp, fn),
            "minimum_effect": minimum_delta,
            "minimum_net_beneficial_swaps": required_net,
            "available_perfect_swap_counterfactuals": swaps_available,
            "fraction_of_perfect_swap_count_needed": required_net / swaps_available,
            "beneficial_swaps_needed_by_harmful_swaps": required_by_harm,
            "required_precision_by_harmful_swaps": precision_by_harm,
            "interpretation": "Arithmetic plausibility only; candidate-graph support and learnability remain unmeasured.",
        },
        "prior_outcomes": {
            "heterogeneous_trials": len(precedents),
            "full_gate_successes": full_gate_successes,
            "observed_worst_delta": observed_worst,
            "observed_median_delta": observed_median,
            "exchangeable_probability_estimate_allowed": False,
        },
        "substrate_partition": substrate,
        "scenario_sensitivity": {
            "status": probabilities["status"],
            "realized_fraction_precedent": REALIZATION_PRECEDENT,
            "perfect_swap_ceiling": swap_ceiling,
            "conditional_best_delta": conditional_best,
            "scenarios": sensitivity,
            "probability_needed_for_positive_expected_delta": break_even_positive,
            "probability_needed_for_expected_delta_at_least_0_005": break_even_gate,
            "dollar_ev_available": False,
            "dollar_ev_limitation": "No USD value per score point or leaderboard utility curve has been provided; score-EV and compute cost must remain separate.",
        },
        "thunder_cost_sensitivity_usd": cost_table,
        "locked_f6_gate_audit": {
            "config_space_exhausted": {
                "pass": False,
                "f3_coverage": "0/48",
                "coverage_fraction": 0.0,
                "required_fraction": 0.8,
                "top3_plateau_available": False,
            },
            "residual_is_real": {
                "pass": True,
                "named_term": "swap_target_already_parented",
                "perfect_repair_ceiling": swap_ceiling,
                "warning": "This is a GT-dependent ceiling, not realizable model value.",
            },
            "residual_is_learnable": {
                "pass": False,
                "target_family_pilot_completed": False,
            },
            "economics": {
                "pass": False,
                "reason": "Only scenario assumptions exist; no target-calibrated P(delta > 0.005).",
            },
        },
        "decision": {
            "paid_thunder_gpu_hours_now": 0,
            "verdict": "PARK_PAID_TRAINING",
            "reason": "Three of four locked F6 requirements are not proven.",
            "if_later_promoted": {
                "preferred_value_configuration": "1x RTX A6000 48GB, 8 vCPU, 64GB RAM, 100GB disk",
                "hard_wall_clock_cap_hours": 8,
                "maximum_compute_cost_usd": cost_table["8"]["rtx_a6000"],
                "plan_amendment_required": "F7 currently specifies A100; use A6000 only after explicit amendment and an exact runtime/memory pilot.",
                "a100_fallback_condition": "Use 1x A100 80GB only if the exact A6000 pilot projects beyond 8 hours or fails the frozen memory/runtime gate.",
                "a100_maximum_compute_cost_usd": cost_table["8"]["a100_80gb"],
            },
        },
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    effect = analysis["effect_arithmetic"]
    prior = analysis["prior_outcomes"]
    scenarios = analysis["scenario_sensitivity"]
    decision = analysis["decision"]
    gates = analysis["locked_f6_gate_audit"]
    substrate = analysis["substrate_partition"]
    lines = [
        "# F2B ROI audit",
        "",
        f"**Decision: `{decision['verdict']}`; paid Thunder allocation now = `0 GPU-h`.**",
        "",
        "## What Claude actually ran",
        "",
        "The approved request described a six-movie D/E export, but the launch receipt bound a 57-movie fused pre/post-ILP cache. The receipt itself says detection-threshold and dual-seed candidates still require a model forward. The run is scientifically `INVALID` and opened zero candidate scores.",
        "",
        "## Effect-size arithmetic",
        "",
        f"The +0.005 gate needs at least **{effect['minimum_net_beneficial_swaps']} net correct reparent swaps**, {effect['fraction_of_perfect_swap_count_needed']:.1%} of the 167-swap perfect counterfactual count. This establishes plausibility, not reachability.",
        "",
        "| Harmful swaps | Beneficial swaps needed | Required precision |",
        "|---:|---:|---:|",
    ]
    for harmful, beneficial in effect["beneficial_swaps_needed_by_harmful_swaps"].items():
        lines.append(
            f"| {harmful} | {beneficial} | {effect['required_precision_by_harmful_swaps'][harmful]:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Evidence and scenario bounds",
            "",
            f"The four heterogeneous learned/association precedents produced **{prior['full_gate_successes']}/{prior['heterogeneous_trials']} full-gate successes**. Their median adjusted delta was {prior['observed_median_delta']:+.6f}; the worst was {prior['observed_worst_delta']:+.6f}. They are not exchangeable trials, so no empirical success rate is claimed.",
            "",
            "| Scenario | Assumed P(delta > +0.005) | Score-EV |",
            "|---|---:|---:|",
        ]
    )
    for name, row in scenarios["scenarios"].items():
        lines.append(
            f"| {name} | {row['assumed_probability_of_delta_gt_0_005']:.1%} | {row['expected_score_delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Those probabilities are sensitivity assumptions—not posteriors. Under the conditional +{scenarios['conditional_best_delta']:.6f} success case and the observed-median failure case, P(success) would need to be {scenarios['probability_needed_for_expected_delta_at_least_0_005']:.1%} for expected score gain itself to reach +0.005.",
            "",
            "USD EV cannot be computed without a declared dollar value or utility curve per score point; reporting compute cost beside score-EV avoids inventing that conversion.",
            "",
            "## Substrate partition",
            "",
            f"The existing fused cache can unlock **{substrate.get('cache_supported_frozen_candidates', {}).get('count', 0)}** frozen A/B/C/G/H candidates for local replay. It cannot evaluate the **{substrate.get('model_forward_required_frozen_candidates', {}).get('count', 0)}** D/E candidates because it omits pre-threshold detection and per-seed blend evidence.",
            "",
            f"The prior 41-movie feature bank overlaps **{substrate.get('prior_feature_bank', {}).get('current_six_movie_overlap', 0)}/6** of the current F2 screen, so reusing it would not be the required target-subsample learnability pilot. Its PEFT probe selected AUC {substrate.get('peft_capacity_probe', {}).get('selected_oof_auc', float('nan')):.6f} and retained only {substrate.get('peft_capacity_probe', {}).get('selected_beneficial_at_budget', 0)}/{substrate.get('peft_capacity_probe', {}).get('total_beneficial', 0)} beneficial actions at the graph budget.",
            "",
            "## Locked F6 gate audit",
            "",
            f"- Configuration exhaustion: **FAIL** ({gates['config_space_exhausted']['f3_coverage']} evaluated at F3; 80% required).",
            "- Named residual: **provisionally PASS** (swap perfect-repair ceiling +0.054595, explicitly not realizable value).",
            "- Target-family learnability: **FAIL** (pilot not run).",
            "- Economics: **FAIL** (no target-calibrated probability).",
            "",
            "## Compute recommendation if the gates later pass",
            "",
            "Start with one RTX A6000 48GB at the exact-runtime pilot stage. At the current observed $0.35/GPU-h, an eight-hour hard cap is $2.80. The locked F7 document currently says A100, so this requires an explicit amendment. Keep one A100 80GB ($1.09/GPU-h; $8.72 maximum for eight hours) only as the fallback if the A6000 exact pilot exceeds the wall-time or memory gate.",
            "",
            "No paid instance should be provisioned until the D/E cache experiment and the target-family learnability pilot replace these assumptions with measurements.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_path = args.evidence.resolve()
    evidence = json.loads(evidence_path.read_text())
    verified = verify_bound_sources(evidence_path, evidence)
    analysis = compute_analysis(evidence, verified)
    args.json_out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
