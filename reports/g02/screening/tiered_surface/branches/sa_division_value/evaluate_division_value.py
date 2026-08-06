#!/usr/bin/env python3
"""Exact N0--N3 division value and GT-dependency audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE.parents[5]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
    HERE,
):
    sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionCompileError,
    ActionKind,
    GraphAction,
    action_sha256,
    compile_action,
    compile_batch,
)
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from numpy_edge_runtime import FrozenEdgeRuntime, load_cache, sha256  # noqa: E402


LABEL_GT = "GT-DEPENDENT"
LABEL_NONGT = "NON-GT-IDENTIFIABLE"
SPEC = (
    WORKTREE_ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division_cont/"
    "continuity_target_spec.json"
)
PREREG = HERE / "PREREGISTRATION.json"
BASELINE_SCORE = (
    ROOT
    / "_ARCHIVE/conflict_residual/reports/g02/"
    "run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json"
)
FRAGMENTS = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
CACHES = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
FEATURES = ROOT / "kaggle_kernel/bhcv57out_v3/features"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
CHECKPOINT = (
    ROOT
    / "external/Tracking_Support/weights/unet_transformer/split_0/"
    "edge_predictor_best.pth"
)
AUXILIARY = ROOT / "reports/g02/v45f4.pt"
EDGE_HEAD = ROOT / "reports/g02/conflict_edge_f0_fit117_v1/edge.json"

EXECUTION_PINS = {
    "checkpoint": (
        CHECKPOINT,
        "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    ),
    "auxiliary": (
        AUXILIARY,
        "1cff90475b819b5652310ccb73ba57e1dc20069885962a06a5e10a6060c04473",
    ),
    "edge_head": (
        EDGE_HEAD,
        "5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034",
    ),
    "scoreall": (
        ROOT / "scripts/scoreall.py",
        "7b92fed9fe66e9a2faf062528a78b04a65bc1be50f806dff44f51c780da5ac6c",
    ),
}


class AuditError(RuntimeError):
    """The frozen audit contract failed closed."""


def canonical_write(path: Path, value: Any) -> str:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def read_graph(identity: str) -> Any:
    path = FRAGMENTS / f"{identity.replace(':', '__')}.csv"
    graphs = read_submission(path)
    dataset = identity.split(":", 1)[1]
    if set(graphs) != {dataset}:
        raise AuditError(f"fragment identity mismatch: {identity}")
    return graphs[dataset]


def graph_edges(graph: Any) -> set[tuple[int, int]]:
    return {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}


def authenticate() -> dict[str, Any]:
    prereg = load_json(PREREG)
    if prereg["status"] != "FROZEN_BEFORE_CANDIDATE_SCORE_OR_GRAPH_OUTCOMES":
        raise AuditError("preregistration status drifted")
    receipts: dict[str, Any] = {}
    for label, record in prereg["frozen_inputs"].items():
        if label == "cv57_archive":
            if RAW_ARCHIVE.stat().st_size != int(record["bytes"]):
                raise AuditError("raw archive size drifted")
            receipts[label] = {
                "path": str(RAW_ARCHIVE),
                "bytes": RAW_ARCHIVE.stat().st_size,
                "sha256": record["sha256"],
                "hash_source": "banked P0 pin; archive opened read-only",
            }
            continue
        if label == "cv57_fragment_set":
            baseline = load_json(BASELINE_SCORE)
            banked = baseline["input_provenance"]
            if (
                int(banked["file_count"]) != int(record["file_count"])
                or banked["aggregate_sha256"] != record["aggregate_sha256"]
            ):
                raise AuditError("banked fragment-set receipt drifted")
            for row in banked["files"]:
                path = ROOT / row["path"]
                if (
                    path.stat().st_size != int(row["bytes"])
                    or sha256(path) != row["sha256"]
                ):
                    raise AuditError(f"fragment identity drifted: {row['identity']}")
            receipts[label] = {
                "path": str(FRAGMENTS),
                "file_count": int(banked["file_count"]),
                "aggregate_sha256": banked["aggregate_sha256"],
                "all_file_receipts_verified": True,
            }
            continue
        raw_path = record.get("resolved_read_only_path", record.get("path"))
        if raw_path is None:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidates = [WORKTREE_ROOT / candidate, ROOT / candidate]
            candidate = next(
                (
                    path
                    for path in candidates
                    if path.is_file() and sha256(path) == record["sha256"]
                ),
                candidates[0],
            )
        observed = sha256(candidate)
        if observed != record["sha256"]:
            raise AuditError(f"{label} identity drifted: {observed}")
        receipts[label] = {"path": str(candidate), "sha256": observed}
    for label, (path, expected) in EXECUTION_PINS.items():
        observed = sha256(path)
        if observed != expected:
            raise AuditError(f"{label} execution pin drifted: {observed}")
        receipts[label] = {"path": str(path), "sha256": observed}
    return receipts


def make_action(event: Mapping[str, Any]) -> GraphAction:
    repair = event["direct_repair"]
    parent = int(event["parent"]["node_id"])
    children = tuple(sorted(int(row["node_id"]) for row in event["children"]))
    action = GraphAction(
        action_id=(
            f"D1_ORACLE_{str(event['identity']).replace(':', '_')}_"
            f"{int(event['event_index']):03d}"
        ),
        kind=ActionKind.DIVISION_FORK,
        add_edges=tuple(
            (int(source), int(target)) for source, target in repair["required_edges"]
        ),
        replace_outgoing_from=(parent,),
        replace_incoming_to=children,
        expected_conflict_edges=tuple(
            (int(source), int(target))
            for source, target in repair["owned_incident_conflicts"]
        ),
    )
    observed = action_sha256(action)
    if observed != event["wave2_action_sha256"]:
        raise AuditError(
            f"Wave-3 action hash mismatch for {event['event_key']}: {observed}"
        )
    return action


def classify_event(
    event: Mapping[str, Any],
    graph: Any,
    runtime: FrozenEdgeRuntime,
) -> dict[str, Any]:
    identity = str(event["identity"])
    slug = identity.replace(":", "__")
    cache_path = CACHES / f"{slug}.bhc"
    cache = load_cache(cache_path)
    parent = int(event["parent"]["node_id"])
    parent_t = int(event["parent"]["t"])
    children = tuple(sorted(int(row["node_id"]) for row in event["children"]))
    edges = graph_edges(graph)
    nodes = graph.nodes
    score = runtime.score_parent(cache, parent_t, parent)
    probabilities = {
        int(target): float(value) for target, value in score["probabilities"].items()
    }
    ranking = sorted(probabilities, key=lambda target: (-probabilities[target], target))
    top_two = tuple(sorted(ranking[:2])) if len(ranking) >= 2 else tuple()
    exact_edges = {(parent, child) for child in children}
    candidate_or_selected = all(
        edge in edges or edge[1] in probabilities for edge in exact_edges
    )
    second_probability = (
        float(probabilities[ranking[1]]) if len(ranking) >= 2 else None
    )
    third_probability = (
        float(probabilities[ranking[2]]) if len(ranking) >= 3 else None
    )
    no_link = float(score["no_link"])
    tolerance = 1e-12
    tie = (
        second_probability is not None
        and (
            abs(second_probability - no_link) <= tolerance
            or (
                third_probability is not None
                and abs(second_probability - third_probability) <= tolerance
            )
        )
    )
    outgoing_continuation = {
        child: any(
            source == child
            and source in nodes
            and target in nodes
            and int(nodes[target].t) > int(nodes[source].t)
            for source, target in edges
        )
        for child in children
    }
    tier_a_checks = {
        "exact_edges_candidate_or_selected": candidate_or_selected,
        "exact_children_equal_deterministic_top_two": top_two == children,
        "second_probability_gt_no_link": (
            second_probability is not None and second_probability > no_link
        ),
        "both_children_selected_outgoing_continuation": all(
            outgoing_continuation.values()
        ),
        "no_decision_boundary_tie": not tie,
    }
    tier_a = all(tier_a_checks.values())
    tier_b_checks = {
        "parent_present": parent in nodes,
        "both_children_present": all(child in nodes for child in children),
        "children_distinct": len(set(children)) == 2,
        "both_edges_forward_time": (
            parent in nodes
            and all(
                child in nodes and int(nodes[child].t) > int(nodes[parent].t)
                for child in children
            )
        ),
    }
    tier_b = (not tier_a) and all(tier_b_checks.values())
    tier = "a" if tier_a else ("b" if tier_b else "c")
    probability_sum = no_link + sum(probabilities.values())
    if abs(probability_sum - 1.0) > 1e-10:
        raise AuditError(f"candidate probability normalization failed: {identity}")
    feature = load_json(FEATURES / f"{slug}.json")
    return {
        "event_key": event["event_key"],
        "identity": identity,
        "family": event["family"],
        "parent_id": parent,
        "children": list(children),
        "tier": tier,
        "tier_name": {
            "a": "CURRENT_POOL_CURRENT_FEATURES",
            "b": "STRONGER_REPRESENTATION_REQUIRED",
            "c": "GT_IDENTITY_ONLY",
        }[tier],
        "information_label": LABEL_NONGT if tier in ("a", "b") else LABEL_GT,
        "candidate_audit": {
            "candidate_count_for_parent": int(score["candidate_count"]),
            "candidate_probabilities": {
                str(target): probabilities[target] for target in sorted(probabilities)
            },
            "no_link_probability": no_link,
            "probability_sum_including_no_link": probability_sum,
            "deterministic_ranking": ranking,
            "selected_outgoing_continuation": {
                str(key): value for key, value in outgoing_continuation.items()
            },
            "tier_a_checks": tier_a_checks,
            "tier_b_checks": tier_b_checks,
            "cache_sha256": sha256(cache_path),
            "cache_stored_edge_pairs": int(cache["stored_edge_pairs"]),
            "frozen_video_candidate_count": int(
                feature["edge_association"]["candidate_count"]
            ),
        },
    }


def metric(graph: Any, gt: Mapping[str, Any], scale: tuple[float, float, float]) -> dict[str, Any]:
    nodes, edges = scoreall._prediction_rows(graph.dataset, graph)
    return dict(official_metric_row(nodes, edges, dict(gt), scale))


def summarise(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dict(official_summary([dict(row) for row in rows]))
    for key in (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
    ):
        result[key] = int(sum(int(float(row[key])) for row in rows))
    result["information_label"] = label
    return result


def delta_record(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], label: str
) -> dict[str, Any]:
    result = {
        "adjusted_edge_jaccard_delta": float(candidate["adj_edge_jaccard"])
        - float(baseline["adj_edge_jaccard"]),
        "raw_edge_jaccard_delta": float(candidate["edge_jaccard"])
        - float(baseline["edge_jaccard"]),
        "node_recall_delta": float(candidate["node_recall"])
        - float(baseline["node_recall"]),
        "division_tp": int(candidate["division_tp"]),
        "division_fp": int(candidate["division_fp"]),
        "division_fn": int(candidate["division_fn"]),
        "division_jaccard": float(candidate["division_jaccard"]),
        "combined_score_delta": float(candidate["score"]) - float(baseline["score"]),
        "information_label": label,
    }
    if not all(
        math.isfinite(value)
        for key, value in result.items()
        if key != "information_label"
    ):
        raise AuditError("nonfinite metric delta")
    return result


def coalition_key(tiers: Sequence[str]) -> str:
    return "".join(sorted(tiers)) or "none"


def shapley(values: Mapping[str, float]) -> dict[str, float]:
    tiers = ("a", "b", "c")
    output = {tier: 0.0 for tier in tiers}
    factorial = math.factorial
    for tier in tiers:
        others = [value for value in tiers if value != tier]
        for size in range(len(others) + 1):
            weight = (
                factorial(size)
                * factorial(len(tiers) - size - 1)
                / factorial(len(tiers))
            )
            for subset in itertools.combinations(others, size):
                before = coalition_key(subset)
                after = coalition_key((*subset, tier))
                output[tier] += weight * (values[after] - values[before])
    return output


def score_all(
    events: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_payload = load_json(BASELINE_SCORE)
    baseline_by_identity = {
        str(row["identity"]): dict(row) for row in baseline_payload["samples"]
    }
    identities = sorted(baseline_by_identity)
    actions_by_identity: dict[str, list[tuple[str, GraphAction]]] = defaultdict(list)
    event_by_key = {str(row["event_key"]): row for row in events}
    assignment_by_key = {str(row["event_key"]): row for row in assignments}
    compiler_receipts: list[dict[str, Any]] = []
    graphs: dict[str, Any] = {}
    for identity in identities:
        graphs[identity] = read_graph(identity)
    for event in events:
        action = make_action(event)
        identity = str(event["identity"])
        tier = str(assignment_by_key[str(event["event_key"])]["tier"])
        compiled = compile_action(graphs[identity], action)
        actions_by_identity[identity].append((tier, action))
        compiler_receipts.append(
            {
                "event_key": event["event_key"],
                "identity": identity,
                "tier": tier,
                "action_sha256": compiled.action_sha256,
                "before_graph_sha256": compiled.before_graph_sha256,
                "after_graph_sha256": compiled.after_graph_sha256,
                "removed_edges": [list(edge) for edge in compiled.removed_edges],
            }
        )

    coalitions = [
        tuple(combo)
        for size in range(4)
        for combo in itertools.combinations(("a", "b", "c"), size)
    ]
    rows_by_coalition: dict[str, list[dict[str, Any]]] = {
        coalition_key(value): [] for value in coalitions
    }
    metric_cache: dict[tuple[str, str], dict[str, Any]] = {}
    per_video_receipts: list[dict[str, Any]] = []
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for identity in identities:
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            gt = read_geff_arrays(store, "train", dataset)
            per_video: dict[str, Any] = {
                "identity": identity,
                "coalitions": {},
            }
            for coalition in coalitions:
                key = coalition_key(coalition)
                selected = [
                    action
                    for tier, action in actions_by_identity.get(identity, [])
                    if tier in coalition
                ]
                if not selected:
                    row = dict(baseline_by_identity[identity])
                    graph_hash = None
                    batch_receipt = None
                else:
                    ordered = sorted(selected, key=lambda action: action.action_id)
                    batch_error = None
                    try:
                        frozen_batch = compile_batch(graphs[identity], ordered)
                    except ActionCompileError as exc:
                        batch_error = str(exc)
                        frozen_batch = None
                    current = graphs[identity]
                    sequential_receipts = []
                    for action in ordered:
                        compiled = compile_action(current, action)
                        current = compiled.graph
                        sequential_receipts.append(
                            {
                                "action_id": action.action_id,
                                "action_sha256": compiled.action_sha256,
                                "before_graph_sha256": compiled.before_graph_sha256,
                                "after_graph_sha256": compiled.after_graph_sha256,
                            }
                        )
                    if len(ordered) > 1:
                        reverse = graphs[identity]
                        for action in reversed(ordered):
                            reverse = compile_action(reverse, action).graph
                        from celltrack_alpha.action_compiler import graph_sha256

                        if graph_sha256(reverse) != sequential_receipts[-1]["after_graph_sha256"]:
                            raise AuditError(
                                f"sequential N1 composition is order-dependent: {identity}"
                            )
                    graph_hash = sequential_receipts[-1]["after_graph_sha256"]
                    if (
                        frozen_batch is not None
                        and frozen_batch.after_graph_sha256 != graph_hash
                    ):
                        raise AuditError(f"batch/sequential N1 graph mismatch: {identity}")
                    cache_key = (identity, graph_hash)
                    if cache_key not in metric_cache:
                        metric_cache[cache_key] = metric(current, gt, scale)
                    row = dict(metric_cache[cache_key])
                    row["identity"] = identity
                    batch_receipt = {
                        "action_count": len(selected),
                        "after_graph_sha256": graph_hash,
                        "composition": (
                            "compile_batch"
                            if frozen_batch is not None
                            else "ordered_public_compile_action"
                        ),
                        "compile_batch_error": batch_error,
                        "batch_sha256": (
                            frozen_batch.batch_sha256
                            if frozen_batch is not None
                            else None
                        ),
                        "order_independence_checked": len(ordered) > 1,
                        "sequential_receipts": sequential_receipts,
                    }
                rows_by_coalition[key].append(row)
                per_video["coalitions"][key] = batch_receipt
            per_video_receipts.append(per_video)

    result: dict[str, Any] = {}
    for coalition in coalitions:
        key = coalition_key(coalition)
        label = LABEL_GT if "c" in coalition or set(coalition) == {"a", "b", "c"} else LABEL_NONGT
        pooled_baseline = summarise(
            [baseline_by_identity[identity] for identity in identities], label
        )
        pooled_candidate = summarise(rows_by_coalition[key], label)
        families: dict[str, Any] = {}
        for family in ("44b6", "6bba"):
            selected_ids = [
                identity
                for identity in identities
                if identity.startswith(f"train:{family}_")
            ]
            positions = [identities.index(identity) for identity in selected_ids]
            family_baseline = summarise(
                [baseline_by_identity[identity] for identity in selected_ids], label
            )
            family_candidate = summarise(
                [rows_by_coalition[key][position] for position in positions], label
            )
            families[family] = {
                "baseline": family_baseline,
                "candidate": family_candidate,
                "delta": delta_record(family_baseline, family_candidate, label),
            }
        result[key] = {
            "included_tiers": list(coalition),
            "event_count": sum(
                1
                for row in assignments
                if str(row["tier"]) in coalition
            ),
            "information_label": label,
            "pooled": {
                "baseline": pooled_baseline,
                "candidate": pooled_candidate,
                "delta": delta_record(pooled_baseline, pooled_candidate, label),
            },
            "families": families,
        }
    return result, [
        {"single_action_receipts": compiler_receipts},
        {"batch_receipts": per_video_receipts},
    ]


def fp_curves(coalitions: Mapping[str, Any]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for key in ("a", "ab"):
        record = coalitions[key]["pooled"]["delta"]
        true_forks = int(coalitions[key]["event_count"])
        edge_delta = float(record["adjusted_edge_jaccard_delta"])
        def payoff(false_forks: int) -> float:
            return edge_delta + 0.1 * true_forks / (42 + false_forks)

        zero_fp_payoff = payoff(0)
        if zero_fp_payoff >= 0.005:
            crossing_limit = next(
                (
                    false_forks
                    for false_forks in range(0, 100_001)
                    if payoff(false_forks) < 0.005
                ),
                100_000,
            )
            maximum_fp = max(10, crossing_limit)
        else:
            maximum_fp = max(10, true_forks * 2)
        fp_values = list(range(0, maximum_fp + 1))
        points = []
        for false_forks in fp_values:
            precision = (
                true_forks / (true_forks + false_forks)
                if true_forks + false_forks
                else 1.0
            )
            division_jaccard = true_forks / (42 + false_forks)
            points.append(
                {
                    "true_forks": true_forks,
                    "false_forks": false_forks,
                    "precision": precision,
                    "division_jaccard": division_jaccard,
                    "edge_neutral_combined_score_delta": edge_delta
                    + 0.1 * division_jaccard,
                    "information_label": LABEL_NONGT,
                }
            )
        passing_0p005 = [
            point
            for point in points
            if point["edge_neutral_combined_score_delta"] >= 0.005
        ]
        if zero_fp_payoff < 0.005:
            target_requirement = {
                "status": "INFEASIBLE_AT_PERFECT_SELECTION",
                "minimum_precision": None,
                "maximum_false_forks": None,
                "perfect_precision_payoff": zero_fp_payoff,
                "information_label": LABEL_NONGT,
            }
        else:
            boundary = passing_0p005[-1]
            target_requirement = {
                "status": "EDGE_NEUTRAL_UPPER_BOUND",
                "minimum_precision": boundary["precision"],
                "maximum_false_forks": boundary["false_forks"],
                "perfect_precision_payoff": zero_fp_payoff,
                "information_label": LABEL_NONGT,
            }
        curves[key] = {
            "subset": list(coalitions[key]["included_tiers"]),
            "interpretation": (
                "Edge-neutral false-fork upper bound. Actual false forks may also "
                "reduce adjusted edge Jaccard."
            ),
            "information_label": LABEL_NONGT,
            "requirements": {
                "target_combined_delta_plus_0p005": target_requirement,
                "breakeven": {
                    "status": (
                        "NO_POSITIVE_PRECISION_FLOOR_IN_EDGE_NEUTRAL_BOUND"
                        if edge_delta >= 0.0
                        else "SEE_ENUMERATED_BOUNDARY"
                    ),
                    "asymptotic_combined_delta": edge_delta,
                    "information_label": LABEL_NONGT,
                },
            },
            "points": points,
        }
    return {
        "schema": "celltrack-alpha.division-fp-sensitivity.v1",
        "information_label": LABEL_NONGT,
        "curves": curves,
    }


def run() -> dict[str, str]:
    started = time.perf_counter()
    provenance = authenticate()
    spec = load_json(SPEC)
    events = list(spec["events"])
    if len(events) != 38:
        raise AuditError("Wave-3 event count drifted")
    runtime = FrozenEdgeRuntime(CHECKPOINT, AUXILIARY, EDGE_HEAD)
    graphs = {identity: read_graph(identity) for identity in sorted({row["identity"] for row in events})}
    assignments = [
        classify_event(event, graphs[str(event["identity"])], runtime)
        for event in events
    ]
    tier_counts = {
        tier: sum(row["tier"] == tier for row in assignments)
        for tier in ("a", "b", "c")
    }
    coalitions, compiler_receipts = score_all(events, assignments)
    value_map = {
        key: float(record["pooled"]["delta"]["combined_score_delta"])
        for key, record in coalitions.items()
    }
    shapley_values = shapley(value_map)
    ceiling = float(coalitions["abc"]["pooled"]["delta"]["combined_score_delta"])
    if abs(sum(shapley_values.values()) - ceiling) > 1e-12:
        raise AuditError("Shapley allocation does not sum to ceiling")
    dependency = {
        "schema": "celltrack-alpha.division-gt-dependency.v1",
        "mandatory_framing": (
            "The N0 oracle ceiling uses GT endpoint identity. It is not evidence of "
            "reachable gain and cannot unlock division fitting."
        ),
        "tiers": [
            {
                "tier": tier,
                "name": {
                    "a": "CURRENT_POOL_CURRENT_FEATURES",
                    "b": "STRONGER_REPRESENTATION_REQUIRED",
                    "c": "GT_IDENTITY_ONLY",
                }[tier],
                "event_count": tier_counts[tier],
                "event_share": tier_counts[tier] / 38.0,
                "tier_only_combined_score_delta": value_map[tier],
                "shapley_combined_score_share": shapley_values[tier],
                "information_label": LABEL_NONGT if tier in ("a", "b") else LABEL_GT,
            }
            for tier in ("a", "b", "c")
        ],
        "events": assignments,
    }
    results = {
        "schema": "celltrack-alpha.division-value-realizability-results.v1",
        "status": "N0_N3_COMPLETE",
        "mandatory_framing": (
            "GT-dependent ceiling and non-GT-identifiable diagnostic subsets are "
            "separate. No emitted subset is a trained realizer."
        ),
        "n0_ceiling": {
            "coalition": coalitions["abc"],
            "information_label": LABEL_GT,
            "n1_composition": (
                "Every frozen Wave-3 action compiled individually with its pinned "
                "hash. Frozen compile_batch is structurally inapplicable when an "
                "action both owns and re-adds an already-correct daughter edge; "
                "affected per-video graphs therefore use ordered public "
                "compile_action composition with reverse-order hash equality."
            ),
        },
        "n1_gt_dependency": {
            "tier_counts": {
                tier: {
                    "count": tier_counts[tier],
                    "information_label": (
                        LABEL_NONGT if tier in ("a", "b") else LABEL_GT
                    ),
                }
                for tier in ("a", "b", "c")
            },
            "shapley": {
                tier: {
                    "combined_score_share": shapley_values[tier],
                    "information_label": (
                        LABEL_NONGT if tier in ("a", "b") else LABEL_GT
                    ),
                }
                for tier in ("a", "b", "c")
            },
        },
        "n2_realizable_range": {
            "a": coalitions["a"],
            "a_plus_b": coalitions["ab"],
            "caveat": (
                "Information-requirement diagnostic only; neither subset is a "
                "trained selector or deployment measurement."
            ),
            "information_label": LABEL_NONGT,
        },
        "coalitions_for_exact_shapley": coalitions,
        "execution_time": {
            "seconds": time.perf_counter() - started,
            "measurement_type": "operational_not_scientific_outcome",
            "information_label": LABEL_NONGT,
        },
    }
    curves = fp_curves(coalitions)
    provenance_record = {
        "schema": "celltrack-alpha.division-value-provenance.v1",
        "cpu_only": True,
        "gpu_used": False,
        "kaggle_used": False,
        "budget_debit": False,
        "fitting": False,
        "bundle_derivation": False,
        "inputs": provenance,
        "numpy_runtime": {
            "reason": (
                "Frozen caches contain detector features; the hash-pinned released "
                "linear heads are evaluated with NumPy only."
            ),
            "normalization_checks_passed": len(assignments),
            "implementation_sha256": sha256(HERE / "numpy_edge_runtime.py"),
        },
        "compiler_receipts": compiler_receipts,
        "n1_composition_deviation": {
            "preregistered_intent": "compile each action alone and each per-video batch",
            "observed_frozen_compiler_behavior": (
                "compile_batch rejects an action when the same already-correct edge "
                "is in add_edges and expected_conflict_edges; compile_action accepts "
                "the exact pinned action and returns the intended graph."
            ),
            "resolution": (
                "No N1 or bundle bytes changed. Actions are compiled individually, "
                "then disjoint actions are composed in action_id order through the "
                "same public compile_action API. Multi-action videos must match the "
                "reverse-order final graph hash."
            ),
            "outcome_independent": True,
            "fitting_or_tuning": False,
        },
    }
    hashes = {
        "N0_N3_RESULTS.json": canonical_write(HERE / "N0_N3_RESULTS.json", results),
        "GT_DEPENDENCY_TABLE.json": canonical_write(
            HERE / "GT_DEPENDENCY_TABLE.json", dependency
        ),
        "FP_SENSITIVITY.json": canonical_write(
            HERE / "FP_SENSITIVITY.json", curves
        ),
        "RUN_PROVENANCE.json": canonical_write(
            HERE / "RUN_PROVENANCE.json", provenance_record
        ),
    }
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), sort_keys=True))
