#!/usr/bin/env python3
"""Measure whether a selector over fixed exact-GAP2 actions can clear the gate.

This is the mandatory marginal-value check before extracting a new appearance
observable.  It labels the already-fixed GAP2 action vocabulary with the frozen
official matcher on held_fit105, authenticates additivity against the full run,
exact-scores label-informed action policies, and computes two deliberately
optimistic envelopes that charge neither false edges nor synthetic nodes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from optimize_gap2_action_selector import (  # noqa: E402
    build_selected_graph,
    extract_actions,
    family_summaries,
    label_actions,
)
from score_gap2_same_run_ablation import (  # noqa: E402
    graph_metric_row,
    load_selected_graphs,
    load_stats,
)
from celltrack_alpha.official_eval import official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.metrics import EvaluationResult, per_sample_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    branch = (
        ROOT
        / "reports/g02/screening/tiered_surface/branches/"
        "public_temporalunet_baseline_v1/gap2_probe"
    )
    parser.add_argument(
        "--population-manifest",
        type=Path,
        default=branch.parent / "POPULATIONS.json",
    )
    parser.add_argument("--submission", type=Path, default=branch / "output_v1/submission.csv")
    parser.add_argument("--run-stats", type=Path, default=branch / "output_v1/run_stats.csv")
    parser.add_argument("--full-score", type=Path, default=branch / "HELD_FIT105_SCORE.json")
    parser.add_argument(
        "--control-score",
        type=Path,
        default=branch / "GAP2_SAME_RUN_ABLATION_HELD_FIT105.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def delta(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(control[key])
        for key in ("score", "adj_edge_jaccard", "edge_jaccard", "node_recall")
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.population_manifest.read_text(encoding="utf-8"))
    identities = [str(value) for value in manifest["populations"]["held_fit105"]]
    datasets = [identity.split(":", 1)[1] for identity in identities]
    graphs = load_selected_graphs(args.submission, set(datasets))
    stats = load_stats(args.run_stats)
    full = json.loads(args.full_score.read_text(encoding="utf-8"))
    control = json.loads(args.control_score.read_text(encoding="utf-8"))
    full_rows = {str(row["identity"]): row for row in full["samples"]}
    control_rows = {str(row["identity"]): row for row in control["samples"]}

    import zarr
    from zarr.storage import ZipStore

    labelled_by_identity = {}
    gt_by_identity = {}
    scale_by_identity = {}
    additivity = []
    store = ZipStore(args.archive, mode="r")
    try:
        for ordinal, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = parse_scale(dict(group.attrs))
            gt = read_geff_arrays(store, "train", dataset)
            actions = label_actions(
                graphs[dataset], extract_actions(graphs[dataset], stats[dataset]), gt, scale
            )
            labelled_by_identity[identity] = actions
            gt_by_identity[identity] = gt
            scale_by_identity[identity] = scale
            matched = sum(action.matched_edges for action in actions)
            valid = sum(action.valid_edges for action in actions)
            observed = full_rows[identity]
            base = control_rows[identity]
            additivity.append(
                {
                    "identity": identity,
                    "actions": len(actions),
                    "labelled_matched_edges": matched,
                    "labelled_valid_edges": valid,
                    "observed_edge_tp_delta": int(observed["edge_tp"] - base["edge_tp"]),
                    "observed_edge_fp_delta": int(observed["edge_fp"] - base["edge_fp"]),
                    "observed_edge_fn_delta": int(observed["edge_fn"] - base["edge_fn"]),
                    "tp_additivity_error": int(observed["edge_tp"] - base["edge_tp"] - matched),
                    "fp_additivity_error": int(
                        observed["edge_fp"] - base["edge_fp"] - (valid - matched)
                    ),
                    "node_additivity_error": int(
                        observed["num_pred_nodes"] - base["num_pred_nodes"] - 2 * len(actions)
                    ),
                }
            )
            print(f"oracle labels {ordinal}/{len(identities)} {identity} actions={len(actions)}", flush=True)

        policies = {
            "all_valid_edges_matched": lambda action: (
                action.valid_edges > 0 and action.matched_edges == action.valid_edges
            ),
            "strict_three_edge_path_matched": lambda action: (
                action.valid_edges == 3 and action.matched_edges == 3
            ),
            "positive_edge_balance": lambda action: (
                action.matched_edges > action.valid_edges - action.matched_edges
            ),
        }
        exact_policy_results = {}
        for policy_name, predicate in policies.items():
            selected = {
                action.key
                for identity in identities
                for action in labelled_by_identity[identity]
                if predicate(action)
            }
            rows = []
            structural = Counter()
            for ordinal, identity in enumerate(identities, start=1):
                dataset = identity.split(":", 1)[1]
                chosen = {
                    action.key
                    for action in labelled_by_identity[identity]
                    if action.key in selected
                }
                candidate = build_selected_graph(
                    graphs[dataset], stats[dataset], labelled_by_identity[identity], chosen
                )
                rows.append(
                    {
                        "identity": identity,
                        **graph_metric_row(
                            candidate, gt_by_identity[identity], scale_by_identity[identity]
                        ),
                    }
                )
                structural["actions"] += len(chosen)
                structural["nodes"] += 2 * len(chosen)
                structural["incident_edges"] += sum(
                    len(action.incident_edges)
                    for action in labelled_by_identity[identity]
                    if action.key in chosen
                )
                print(
                    f"oracle exact {policy_name} {ordinal}/{len(identities)} "
                    f"{identity} selected={len(chosen)}",
                    flush=True,
                )
            summary = official_summary(rows)
            families = family_summaries(rows)
            control_families = family_summaries(list(control_rows.values()))
            exact_policy_results[policy_name] = {
                "structural": dict(structural),
                "summary": summary,
                "family_summary": families,
                "family_delta": {
                    family: delta(families[family], control_families[family])
                    for family in families
                },
            }
    finally:
        store.close()

    # Deliberately optimistic envelope: credit every edge that matched in the full
    # GAP2 graph, charge zero added FP, zero synthetic nodes, and carry the full-run
    # node recall.  This cannot be a deployable graph; it asks whether the observed
    # fixed action vocabulary contains enough correctly placed edge mass even before
    # costs and imperfect selection are paid.
    matched_edge_envelope_rows = []
    for identity in identities:
        base = control_rows[identity]
        matched = sum(action.matched_edges for action in labelled_by_identity[identity])
        er = EvaluationResult(
            edge_tp=int(base["edge_tp"]) + matched,
            edge_fp=int(base["edge_fp"]),
            edge_fn=max(0, int(base["edge_fn"]) - matched),
            division_tp=int(base["division_tp"]),
            division_fp=int(base["division_fp"]),
            division_fn=int(base["division_fn"]),
            num_pred_nodes=int(base["num_pred_nodes"]),
        )
        row = per_sample_metrics(
            er,
            float(base["estimated_number_of_nodes"]),
            float(full_rows[identity]["node_recall"]),
        )
        row["division_jaccard"] = float(base["division_jaccard"])
        matched_edge_envelope_rows.append({"identity": identity, **row})
    matched_edge_envelope_summary = official_summary(matched_edge_envelope_rows)
    matched_edge_envelope_family = family_summaries(matched_edge_envelope_rows)
    control_family = family_summaries(list(control_rows.values()))
    matched_edge_envelope_delta = {
        family: delta(matched_edge_envelope_family[family], control_family[family])
        for family in matched_edge_envelope_family
    }

    # Still more permissive stress test: pretend every score-valid incident edge in
    # the fixed vocabulary is a TP, remove its FN counterpart, charge no FP/nodes,
    # and set node recall to 1.0.  This is not a realizable graph and is not claimed
    # as a formal mathematical upper bound because matching interactions may alter
    # validity.  It is a deliberately favorable envelope for the stop-loss decision.
    all_valid_envelope_rows = []
    for identity in identities:
        base = control_rows[identity]
        valid = sum(action.valid_edges for action in labelled_by_identity[identity])
        er = EvaluationResult(
            edge_tp=int(base["edge_tp"]) + valid,
            edge_fp=int(base["edge_fp"]),
            edge_fn=max(0, int(base["edge_fn"]) - valid),
            division_tp=int(base["division_tp"]),
            division_fp=int(base["division_fp"]),
            division_fn=int(base["division_fn"]),
            num_pred_nodes=int(base["num_pred_nodes"]),
        )
        row = per_sample_metrics(
            er,
            float(base["estimated_number_of_nodes"]),
            1.0,
        )
        row["division_jaccard"] = float(base["division_jaccard"])
        all_valid_envelope_rows.append({"identity": identity, **row})
    all_valid_envelope_summary = official_summary(all_valid_envelope_rows)
    all_valid_envelope_family = family_summaries(all_valid_envelope_rows)
    all_valid_envelope_delta = {
        family: delta(all_valid_envelope_family[family], control_family[family])
        for family in all_valid_envelope_family
    }
    permissive_minimum_family = min(
        value["adj_edge_jaccard"] for value in all_valid_envelope_delta.values()
    )
    exact_policy_minimum_family = {
        name: min(
            family["adj_edge_jaccard"]
            for family in result["family_delta"].values()
        )
        for name, result in exact_policy_results.items()
    }
    best_exact_policy = max(
        exact_policy_minimum_family,
        key=exact_policy_minimum_family.__getitem__,
    )
    realistic_ceiling = exact_policy_minimum_family[best_exact_policy]
    required_delta = 0.003
    verdict = (
        "GO_APPEARANCE_SCREEN"
        if permissive_minimum_family >= required_delta
        else "HARD_STOP_FIXED_ACTION_CLASS"
    )
    payload = {
        "schema": "celltrack-alpha.gap2-fixed-action-oracle-ceiling.v2",
        "status": "COMPLETE",
        "candidate": "GAP-ID-TRANSPORT-v1",
        "population": "held_fit105",
        "action_vocabulary": "surviving exact public_v32_gap2 two-node bridges",
        "counts": {
            "actions": sum(len(value) for value in labelled_by_identity.values()),
            "matched_edges_in_full_graph": sum(
                action.matched_edges
                for value in labelled_by_identity.values()
                for action in value
            ),
            "valid_incident_edges_in_full_graph": sum(
                action.valid_edges
                for value in labelled_by_identity.values()
                for action in value
            ),
        },
        "additivity": {
            "aggregate_errors": {
                key: sum(row[key] for row in additivity)
                for key in ("tp_additivity_error", "fp_additivity_error", "node_additivity_error")
            },
            "videos_with_any_error": sum(
                any(row[key] != 0 for key in ("tp_additivity_error", "fp_additivity_error", "node_additivity_error"))
                for row in additivity
            ),
            "samples": additivity,
        },
        "exact_label_oracles": exact_policy_results,
        "observed_matched_edge_zero_cost_envelope": {
            "definition": (
                "credit every full-graph matched GAP2 incident edge; charge zero FP, zero "
                "synthetic nodes, and use full-run node recall"
            ),
            "deployable": False,
            "formal_upper_bound": False,
            "summary": matched_edge_envelope_summary,
            "family_summary": matched_edge_envelope_family,
            "family_delta": matched_edge_envelope_delta,
            "minimum_family_adjusted_delta": min(
                value["adj_edge_jaccard"]
                for value in matched_edge_envelope_delta.values()
            ),
        },
        "permissive_all_valid_zero_cost_envelope": {
            "definition": (
                "credit every score-valid GAP2 incident edge as TP; charge zero FP and "
                "zero synthetic nodes; force node recall to 1.0"
            ),
            "deployable": False,
            "formal_upper_bound": False,
            "summary": all_valid_envelope_summary,
            "family_summary": all_valid_envelope_family,
            "family_delta": all_valid_envelope_delta,
            "minimum_family_adjusted_delta": permissive_minimum_family,
        },
        "gate": {
            "required_minimum_family_adjusted_delta": required_delta,
            "realistic_ceiling_basis": f"exact label-informed policy: {best_exact_policy}",
            "realistic_ceiling_minimum_family_adjusted_delta": realistic_ceiling,
            "permissive_envelope_minimum_family_adjusted_delta": permissive_minimum_family,
            "required_over_realistic_ratio": required_delta / realistic_ceiling,
            "required_over_permissive_ratio": required_delta / permissive_minimum_family,
            "verdict": verdict,
            "appearance_feature_extraction_authorized": verdict == "GO_APPEARANCE_SCREEN",
        },
        "interpretation": (
            "The exact label-informed realistic ceiling and a more favorable nondeployable "
            "envelope are both below gate. Under the standing marginal-value doctrine, the "
            "fixed action vocabulary is therefore a structural STOP: appearance features "
            "cannot justify further compute unless a future mechanism first changes the "
            "action vocabulary. This is an empirical stop-loss conclusion, not a formal "
            "mathematical impossibility claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": payload["counts"], "gate": payload["gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
