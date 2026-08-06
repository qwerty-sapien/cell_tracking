#!/usr/bin/env python3
"""Build the frozen DAG-2 handoff and diagnostic-only division verdict."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_ROOT = Path("/Users/agent/Desktop/cell_tracking")
DAG2_DIR = SHARED_ROOT / "reports/g02/screening/selector_requirement_44b6/run_v6"
DAG2_SURFACE = DAG2_DIR / "requirement_surface.json"
DAG2_EXPECTED = "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b"
GT = "GT-DEPENDENT"
NONGT = "NON-GT-IDENTIFIABLE"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> object:
    return json.loads(path.read_text())


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    assert sha(DAG2_SURFACE) == DAG2_EXPECTED
    source = load(DAG2_SURFACE)
    results = load(HERE / "N0_N3_RESULTS.json")
    sensitivity = load(HERE / "FP_SENSITIVITY.json")

    current = results["n2_realizable_range"]["a"]["pooled"]["delta"]
    stronger = results["n2_realizable_range"]["a_plus_b"]["pooled"]["delta"]
    ceiling = results["n0_ceiling"]["coalition"]["pooled"]["delta"]
    curve_a = sensitivity["curves"]["a"]["requirements"]
    curve_ab = sensitivity["curves"]["ab"]["requirements"]

    assert current["combined_score_delta"] < 0.005
    assert curve_a["target_combined_delta_plus_0p005"]["status"] == (
        "INFEASIBLE_AT_PERFECT_SELECTION"
    )
    assert stronger["combined_score_delta"] == ceiling["combined_score_delta"]

    handoff = copy.deepcopy(source)
    assert handoff["schema"] == (
        "celltrack-alpha.selector-requirement-44b6.v1.requirement-surface"
    )
    assert set(handoff["requirements"]) == {
        "delete",
        "occupied_swap",
        "restore_topology",
    }

    # DAG-2's copied records were computed with oracle beneficial-action labels.
    # The class-root label is inherited by every numeric leaf without altering
    # any of the frozen numeric values or thresholds.
    for class_name in ("delete", "occupied_swap", "restore_topology"):
        handoff["requirements"][class_name]["information_label"] = GT
        handoff["requirements"][class_name]["source_surface_sha256"] = DAG2_EXPECTED

    handoff["requirements"]["division"] = {
        "breakeven_0": {
            "mean_requirement": None,
            "oracle_best_support": 1,
            "oracle_ceiling": current["combined_score_delta"],
            "robust_p05_requirement": None,
            "status": "NO_MEASURED_SELECTOR_SURFACE_CURRENT_POOL",
            "threshold": 0.0,
        },
        "frozen_gate_verdict": {
            "frozen_auc_floor": 0.65,
            "frozen_retention_floor": 0.15,
            "reason": (
                "No division selector, empirical AUC grid, or retention grid was "
                "measured. The current-pool perfect-selection payoff is below +0.005."
            ),
            "required_mean_retention": None,
            "required_tested_auc": None,
            "verdict": "NOT_EVALUABLE_NO_DIVISION_SELECTOR",
        },
        "target_plus_0_005": {
            "mean_requirement": None,
            "oracle_best_support": 1,
            "oracle_ceiling": current["combined_score_delta"],
            "robust_p05_requirement": None,
            "status": "INFEASIBLE_AT_PERFECT_SELECTION_CURRENT_POOL",
            "threshold": 0.005,
        },
        "representation_conditioned_diagnostic": {
            "event_count": 38,
            "false_fork_boundary": {
                "maximum_false_forks": curve_ab[
                    "target_combined_delta_plus_0p005"
                ]["maximum_false_forks"],
                "minimum_precision": curve_ab[
                    "target_combined_delta_plus_0p005"
                ]["minimum_precision"],
                "model": "EDGE_NEUTRAL_UPPER_BOUND",
            },
            "perfect_selection_payoff": stronger["combined_score_delta"],
            "selector_achievability": "NOT_ESTABLISHED",
            "status": "STRONGER_REPRESENTATION_NOT_BUILT_OR_MEASURED",
        },
        "current_pool_diagnostic": {
            "event_count": 1,
            "perfect_selection_payoff": current["combined_score_delta"],
            "target_status": curve_a["target_combined_delta_plus_0p005"]["status"],
        },
        "information_label": NONGT,
        "source_n0_n3_sha256": sha(HERE / "N0_N3_RESULTS.json"),
        "source_fp_sensitivity_sha256": sha(HERE / "FP_SENSITIVITY.json"),
    }
    dump(HERE / "N4_REQUIREMENT_HANDOFF.json", handoff)

    verdict = {
        "schema": "celltrack-alpha.division-value-n5-verdict.v1",
        "status": "COMPLETE_DIAGNOSTIC_HALT",
        "information_label": NONGT,
        "current_pool": {
            "event_count": 1,
            "combined_score_delta": current["combined_score_delta"],
            "target_plus_0_005_achievable_at_perfect_selection": False,
            "interpretation": (
                "Near-zero current-pool reachability: division is presently as "
                "unreachable as node restoration for an actionable +0.005 gain."
            ),
            "information_label": NONGT,
        },
        "stronger_representation": {
            "event_count": 38,
            "combined_score_delta_upper_bound": stronger["combined_score_delta"],
            "minimum_edge_neutral_precision_for_plus_0_005": curve_ab[
                "target_combined_delta_plus_0p005"
            ]["minimum_precision"],
            "representation_built": False,
            "selector_measured": False,
            "selector_achievability": "NOT_ESTABLISHED",
            "information_label": NONGT,
        },
        "gt_ceiling": {
            "event_count": 38,
            "combined_score_delta": ceiling["combined_score_delta"],
            "division_tp": ceiling["division_tp"],
            "division_fp": ceiling["division_fp"],
            "division_fn": ceiling["division_fn"],
            "division_jaccard": ceiling["division_jaccard"],
            "interpretation": "Oracle ceiling only; not evidence of reachable gain.",
            "information_label": GT,
        },
        "decision": {
            "materially_large_realizable_subset": False,
            "required_selector_precision_achievable_per_dag2": False,
            "amendment_condition_satisfied": False,
            "amendment_proposal_emitted": False,
            "division_fitting": "LOCKED",
            "tier_moves": 0,
            "plan_amendments": 0,
            "dead_actions": 0,
        },
        "reason": (
            "Tier (a) contributes only +0.00240649 at perfect selection, below "
            "the +0.005 target. Tier (b) contains material headroom but has no "
            "built representation or measured selector, so achievability is not "
            "established by DAG 2."
        ),
    }
    dump(HERE / "N5_VERDICT.json", verdict)

    state_fragment = {
        "schema": "g02.live-state-fragment.v1",
        "status": "ROOT_MERGED",
        "information_label": NONGT,
        "frontier": (
            "DIVISION VALUE DIAGNOSTIC TERMINAL / FITTING LOCKED. Exact GT-dependent "
            "38-event ceiling is +0.092067842 combined (38/0/4, Jaccard 0.904762), "
            "but current-pool tier (a) contains 1/38 event and yields only +0.002406490 "
            "even at perfect selection. The remaining 37/38 require a stronger "
            "representation that was not built; its +0.092067842 value is a "
            "representation-conditioned upper bound, not a realized selector gain. "
            "DAG-2 division requirement is NOT_EVALUABLE_NO_DIVISION_SELECTOR. No "
            "amendment proposal, tier move, fitting unlock, DEAD, GPU, Kaggle, budget "
            "debit, RunPod, submission, N1/fold/P0, or plan action."
        ),
        "log": (
            "2026-07-29 — Division-value DAG terminal: GT ceiling +0.092067842; "
            "current-pool perfect-selection +0.002406490 (<+0.005); stronger-representation "
            "selector achievability unestablished; fitting remains locked."
        ),
        "numeric_framing": {
            "gt_ceiling": {
                "combined_score_delta": ceiling["combined_score_delta"],
                "information_label": GT,
            },
            "current_pool": {
                "combined_score_delta": current["combined_score_delta"],
                "information_label": NONGT,
            },
        },
    }
    dump(HERE / "state_fragment.json", state_fragment)

    ledger = f"""| DIVISION-VALUE-REALIZABILITY-20260729 | exact value of 38 frozen Wave-3 division repair bundles | DIAGNOSTIC / GT-DEPENDENCY / REQUIREMENT HANDOFF | frozen N1 compiler/scorer; report GT ceiling separately from deployable information tiers; +0.005 actionability target; no fitting | **DIAGNOSTIC HALT / FITTING LOCKED:** GT ceiling `{ceiling["combined_score_delta"]:+.9f}` [GT-DEPENDENT], TP/FP/FN `{ceiling["division_tp"]}/{ceiling["division_fp"]}/{ceiling["division_fn"]}`, Jaccard `{ceiling["division_jaccard"]:.6f}`; current-pool tier (a) 1/38 and `{current["combined_score_delta"]:+.9f}` [NON-GT-IDENTIFIABLE], below +0.005 even at perfect selection; tier (b) 37/38 requires an unbuilt stronger representation | N0–N3 result `{sha(HERE / "N0_N3_RESULTS.json")[:12]}…`; dependency `{sha(HERE / "GT_DEPENDENCY_TABLE.json")[:12]}…`; FP curve `{sha(HERE / "FP_SENSITIVITY.json")[:12]}…`; DAG-2 source `{DAG2_EXPECTED[:12]}…` | current-pool precision requirement infeasible; representation-conditioned edge-neutral bound reaches +0.005 at precision `{curve_ab["target_combined_delta_plus_0p005"]["minimum_precision"]:.6f}` [NON-GT-IDENTIFIABLE], but no selector/AUC/retention measurement exists | 0 | 0 | 0 / $0 | NO AMENDMENT PROPOSAL / NO TIER MOVE | Oracle headroom is not reachable evidence. Division fitting remains locked pending a separately authorized stronger representation and measured selector surface. | SA-DIVISION-VALUE root coordinator |
"""
    (HERE / "PROPOSED_LEDGER_ROW.md").write_text(ledger)

    report = f"""# Division value: ceiling versus realizable share

## Verdict

**Diagnostic halt; division fitting stays locked.** The exact oracle bundle ceiling is
`{ceiling["combined_score_delta"]:+.12f}` **[GT-DEPENDENT]**, but the subset inferable from
the current candidate pool and features is only `{current["combined_score_delta"]:+.12f}`
**[NON-GT-IDENTIFIABLE]** at perfect selection. That is below the `+0.005` actionability
target. The remaining 37 events require a stronger representation that was not built or
measured, so DAG 2 cannot establish selector achievability.

## Ceiling and dependency decomposition

| quantity | events | combined delta | information label |
|---|---:|---:|---|
| Exact repair-bundle ceiling | 38 | `{ceiling["combined_score_delta"]:+.12f}` | GT-DEPENDENT |
| (a) current pool + current features | 1 | `{current["combined_score_delta"]:+.12f}` | NON-GT-IDENTIFIABLE |
| (b) stronger representation required | 37 | `{results["coalitions_for_exact_shapley"]["b"]["pooled"]["delta"]["combined_score_delta"]:+.12f}` tier-only | NON-GT-IDENTIFIABLE |
| (c) irreducible GT identity | 0 | `+0.000000000000` | GT-DEPENDENT |
| (a)+(b) representation-conditioned upper bound | 38 | `{stronger["combined_score_delta"]:+.12f}` | NON-GT-IDENTIFIABLE |

The GT ceiling has adjusted/raw/node deltas
`{ceiling["adjusted_edge_jaccard_delta"]:+.12f}` /
`{ceiling["raw_edge_jaccard_delta"]:+.12f}` /
`{ceiling["node_recall_delta"]:+.12f}` **[GT-DEPENDENT]**, with division TP/FP/FN
`{ceiling["division_tp"]}/{ceiling["division_fp"]}/{ceiling["division_fn"]}` and Jaccard
`{ceiling["division_jaccard"]:.12f}` **[GT-DEPENDENT]**.

## False-fork sensitivity and DAG-2 handoff

For tier (a), even precision 1 yields only
`{curve_a["target_combined_delta_plus_0p005"]["perfect_precision_payoff"]:+.12f}`
**[NON-GT-IDENTIFIABLE]**, so +0.005 is infeasible. For (a)+(b), the edge-neutral
upper bound stays at or above +0.005 through
`{curve_ab["target_combined_delta_plus_0p005"]["maximum_false_forks"]}` false forks,
equivalent to precision
`{curve_ab["target_combined_delta_plus_0p005"]["minimum_precision"]:.12f}`
**[NON-GT-IDENTIFIABLE]**. This is not an achievability claim: there is no division
selector, empirical AUC grid, or retention measurement.

`N4_REQUIREMENT_HANDOFF.json` preserves the frozen DAG-2 schema and all three source
classes byte-value-exact, adds `division` as the fourth action class, and leaves the
frozen AUC/retention floors unchanged. Its division verdict is
`NOT_EVALUABLE_NO_DIVISION_SELECTOR`.

## Governance

No amendment proposal was emitted because the conjunctive N5 condition failed: the
current realizable subset is not materially large, and the stronger-representation
selector precision is not established as achievable. There are no tier moves, plan
changes, DEAD actions, fitting, GPU/Kaggle/RunPod/submission actions, or budget debit.
"""
    (HERE / "FINAL_REPORT.md").write_text(report)

    barrier = {
        "schema": "g02.division-value.root-barrier.v1",
        "date": "2026-07-29",
        "status": "COMPLETE",
        "information_label": NONGT,
        "division_artifacts": {
            name: sha(HERE / name)
            for name in (
                "N0_N3_RESULTS.json",
                "GT_DEPENDENCY_TABLE.json",
                "FP_SENSITIVITY.json",
                "N4_REQUIREMENT_HANDOFF.json",
                "N5_VERDICT.json",
                "state_fragment.json",
                "PROPOSED_LEDGER_ROW.md",
                "FINAL_REPORT.md",
            )
        },
        "concurrent_barrier_pins": {
            "dag1_dinov2_terminal_invalid_resource_sha256": (
                "9b3f4b455ef6b85f5f1d304a5e6c05742b8cd07540508607407f65ddd0d47ef4"
            ),
            "dag1_cellect_terminal_invalid_sha256": (
                "40aa6dca1f6bb8ab98d9f330c4c22445ec343ce846cfc482636487039aa5c162"
            ),
            "dag2_requirement_surface_sha256": DAG2_EXPECTED,
            "dag2_barrier_sha256": (
                "d07ef7ef2c9be381e22c18c76ed683844bf018ed194e2c393dae15895d70bd54"
            ),
            "dag3_root_commit": "ad35a0e9fd9b60bd1ccfd007e39855c7b4e06dab",
            "dag3_barrier_sha256": (
                "40dca12b4cfcc3e98b8d35508d6fc1f521f5e50e0e035336bc7bfd4cb0914d7a"
            ),
        },
        "governance": {
            "tier_moves": 0,
            "plan_amendments": 0,
            "dead_actions": 0,
            "gpu_actions": 0,
            "kaggle_actions": 0,
            "runpod_actions": 0,
            "budget_debit": 0,
            "division_fitting_locked": True,
            "live_state_reconciled": True,
            "ledger_reconciled": True,
            "dag3_commit_merged": True,
        },
    }
    dump(HERE / "BARRIER_REPORT.json", barrier)

    print(
        json.dumps(
            {
                name: sha(HERE / name)
                for name in (
                    "N4_REQUIREMENT_HANDOFF.json",
                    "N5_VERDICT.json",
                    "BARRIER_REPORT.json",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
