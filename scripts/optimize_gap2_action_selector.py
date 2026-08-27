#!/usr/bin/env python3
"""Fit and exactly test a high-abstention selector for exact GAP2 actions.

Hyperparameters are optimized only on the frozen held_fit105 population.  The
selector uses graph geometry and topology available at inference time; GT is
used only to construct a fit-side proxy objective and to run the frozen exact
score.  A frozen parameter receipt can subsequently be replayed on CV57.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from score_gap2_same_run_ablation import (  # noqa: E402
    ablate_gap2,
    graph_metric_row,
    infer_gap2_allocated_block,
    load_selected_graphs,
    load_stats,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    validate_graphs,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    official_summary,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    EvaluationResult,
    _evaluate_matched_graph,
    evaluate,
    per_sample_metrics,
)
import tracksdata as td  # noqa: E402


FEATURE_NAMES = (
    "total_distance_um",
    "max_step_um",
    "step_cv",
    "acceleration_um",
    "upstream_velocity_residual_um",
    "downstream_velocity_residual_um",
    "minimum_endpoint_track_length",
    "maximum_local_density_8um",
)
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


@dataclass(frozen=True)
class Action:
    identity: str
    first_id: int
    second_id: int
    incoming_source: int
    outgoing_target: int
    incident_edges: tuple[tuple[int, int], ...]
    features: tuple[float, ...]
    matched_edges: int = 0
    valid_edges: int = 0
    matched_nodes: int = 0

    @property
    def key(self) -> str:
        return f"{self.identity}:{self.first_id}:{self.second_id}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", choices=("held_fit105", "cv57"), required=True)
    parser.add_argument("--mode", choices=("tune", "replay"), required=True)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument(
        "--submission",
        type=Path,
        default=branch / "output_v1/submission.csv",
    )
    parser.add_argument(
        "--run-stats",
        type=Path,
        default=branch / "output_v1/run_stats.csv",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument("--full-score", type=Path, required=True)
    parser.add_argument("--control-score", type=Path, required=True)
    return parser.parse_args()


def path_length(
    start: int,
    adjacency: dict[int, list[int]],
    reverse: dict[int, list[int]],
    *,
    forward: bool,
    limit: int = 64,
) -> int:
    length = 1
    current = start
    for _ in range(limit - 1):
        choices = adjacency[current] if forward else reverse[current]
        if len(choices) != 1:
            break
        nxt = choices[0]
        reciprocal = reverse[nxt] if forward else adjacency[nxt]
        if len(reciprocal) != 1:
            break
        length += 1
        current = nxt
    return length


def physical(node: Any) -> np.ndarray:
    return np.asarray((node.z, node.y, node.x), dtype=np.float64) * SCALE_ZYX


def extract_actions(graph: SubmissionGraph, stats: dict[str, int]) -> list[Action]:
    start, end, surviving = infer_gap2_allocated_block(graph, stats)
    if start is None or end is None:
        return []
    edge_pairs = {(edge.source_id, edge.target_id) for edge in graph.edges}
    adjacency: dict[int, list[int]] = defaultdict(list)
    reverse: dict[int, list[int]] = defaultdict(list)
    for source, target in edge_pairs:
        adjacency[source].append(target)
        reverse[target].append(source)
    for values in adjacency.values():
        values.sort()
    for values in reverse.values():
        values.sort()

    # Density is measured in the completed graph because it is an inference-time
    # feature.  The queried node itself is subtracted from the count.
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node in graph.nodes.values():
        by_frame[node.t].append(node.node_id)
    density: dict[int, int] = {}
    for frame, ids in by_frame.items():
        if not any(node_id in surviving for node_id in ids):
            continue
        points = np.stack([physical(graph.nodes[node_id]) for node_id in ids])
        tree = cKDTree(points)
        for index, node_id in enumerate(ids):
            if node_id in surviving:
                density[node_id] = max(0, len(tree.query_ball_point(points[index], r=8.0)) - 1)

    actions: list[Action] = []
    for first in range(start, end + 1, 2):
        second = first + 1
        if first not in graph.nodes and second not in graph.nodes:
            continue
        if first not in graph.nodes or second not in graph.nodes or (first, second) not in edge_pairs:
            raise ValueError(f"{graph.dataset}: malformed surviving GAP2 pair {first},{second}")
        first_node, second_node = graph.nodes[first], graph.nodes[second]
        incoming = [
            source
            for source in reverse[first]
            if source not in surviving and graph.nodes[source].t == first_node.t - 1
        ]
        outgoing = [
            target
            for target in adjacency[second]
            if target not in surviving and graph.nodes[target].t == second_node.t + 1
        ]
        if len(incoming) != 1 or not outgoing:
            raise ValueError(
                f"{graph.dataset}: ambiguous GAP2 endpoints for {first},{second}: "
                f"incoming={incoming}, outgoing={outgoing}"
            )
        source = incoming[0]
        # The original GAP2 continuation is the straightest next-frame target;
        # any later safe-division daughter is retained as a causal incident edge
        # but is not used to define the geometric bridge endpoint.
        p1, p2 = physical(first_node), physical(second_node)
        extrapolated = p2 + (p2 - p1)
        target = min(
            outgoing,
            key=lambda node_id: (
                float(np.linalg.norm(physical(graph.nodes[node_id]) - extrapolated)),
                node_id,
            ),
        )
        p0, p3 = physical(graph.nodes[source]), physical(graph.nodes[target])
        vectors = (p1 - p0, p2 - p1, p3 - p2)
        steps = np.asarray([np.linalg.norm(value) for value in vectors], dtype=np.float64)
        acceleration = float(np.linalg.norm(vectors[1] - vectors[0]) + np.linalg.norm(vectors[2] - vectors[1]))

        upstream_residual = 10.0
        predecessor = [
            node_id
            for node_id in reverse[source]
            if graph.nodes[node_id].t == graph.nodes[source].t - 1
        ]
        if len(predecessor) == 1:
            upstream_velocity = p0 - physical(graph.nodes[predecessor[0]])
            upstream_residual = float(np.linalg.norm(vectors[0] - upstream_velocity))
        downstream_residual = 10.0
        successor = [
            node_id
            for node_id in adjacency[target]
            if graph.nodes[node_id].t == graph.nodes[target].t + 1
        ]
        if len(successor) == 1:
            downstream_velocity = physical(graph.nodes[successor[0]]) - p3
            downstream_residual = float(np.linalg.norm(vectors[2] - downstream_velocity))

        upstream_length = path_length(source, adjacency, reverse, forward=False)
        downstream_length = path_length(target, adjacency, reverse, forward=True)
        incident = tuple(
            sorted(
                (edge.source_id, edge.target_id)
                for edge in graph.edges
                if edge.source_id in (first, second) or edge.target_id in (first, second)
            )
        )
        features = (
            float(np.linalg.norm(p3 - p0)),
            float(np.max(steps)),
            float(np.std(steps) / max(1e-9, np.mean(steps))),
            acceleration,
            upstream_residual,
            downstream_residual,
            float(min(upstream_length, downstream_length)),
            float(max(density.get(first, 0), density.get(second, 0))),
        )
        if not all(math.isfinite(value) for value in features):
            raise ValueError(f"{graph.dataset}: non-finite features for {first},{second}")
        actions.append(
            Action(
                identity=f"train:{graph.dataset}",
                first_id=first,
                second_id=second,
                incoming_source=source,
                outgoing_target=target,
                incident_edges=incident,
                features=features,
            )
        )
    return actions


def label_actions(
    graph: SubmissionGraph,
    actions: list[Action],
    gt: dict[str, Any],
    scale: tuple[float, float, float],
) -> list[Action]:
    nodes = [
        {
            "dataset": graph.dataset,
            "node_id": node.node_id,
            "t": node.t,
            "z": node.z,
            "y": node.y,
            "x": node.x,
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": edge.source_id, "target_id": edge.target_id}
        for edge in graph.edges
    ]
    ordered_original = [
        int(row["node_id"])
        for row in sorted(nodes, key=lambda row: (int(row["t"]), int(row["node_id"])))
    ]
    original_to_graph = {node_id: index for index, node_id in enumerate(ordered_original)}
    graph_to_original = {index: node_id for node_id, index in original_to_graph.items()}
    prediction = prediction_to_graph(nodes, edges)
    truth = geff_arrays_to_graph(gt)
    evaluate(prediction, truth, scale=scale)
    edge_table = _evaluate_matched_graph(prediction, truth)
    edge_state: dict[tuple[int, int], tuple[bool, bool]] = {}
    for row in edge_table.iter_rows(named=True):
        source = graph_to_original[int(row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE])]
        target = graph_to_original[int(row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET])]
        edge_state[(source, target)] = (
            bool(row[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK]),
            bool(row["pred_valid"]),
        )
    node_table = prediction.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    matched_node = {
        graph_to_original[int(row[td.DEFAULT_ATTR_KEYS.NODE_ID])]: (
            row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID] is not None
            and int(row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]) != -1
        )
        for row in node_table.iter_rows(named=True)
    }
    labelled = []
    for action in actions:
        states = [edge_state.get(pair, (False, False)) for pair in action.incident_edges]
        labelled.append(
            Action(
                **{
                    **asdict(action),
                    "matched_edges": sum(matched and valid for matched, valid in states),
                    "valid_edges": sum(valid for _matched, valid in states),
                    "matched_nodes": int(matched_node.get(action.first_id, False))
                    + int(matched_node.get(action.second_id, False)),
                }
            )
        )
    return labelled


def score_vector(features: np.ndarray, params: dict[str, Any], center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (features - center[None, :]) / scale[None, :]
    weights = np.asarray([float(params[f"weight_{name}"]) for name in FEATURE_NAMES])
    return standardized @ weights


def select_actions(
    actions: list[Action],
    params: dict[str, Any],
    center: np.ndarray,
    scale: np.ndarray,
) -> set[str]:
    if not actions:
        return set()
    matrix = np.asarray([action.features for action in actions], dtype=np.float64)
    scores = score_vector(matrix, params, center, scale)
    threshold = float(params["score_threshold"])
    return {action.key for action, score in zip(actions, scores, strict=True) if score >= threshold}


def estimated_rows(
    identities: Iterable[str],
    actions_by_identity: dict[str, list[Action]],
    selected: set[str],
    control_rows: dict[str, dict[str, Any]],
    gt_node_counts: dict[str, int],
) -> list[dict[str, Any]]:
    result = []
    for identity in identities:
        control = control_rows[identity]
        chosen = [action for action in actions_by_identity[identity] if action.key in selected]
        matched_edges = sum(action.matched_edges for action in chosen)
        valid_edges = sum(action.valid_edges for action in chosen)
        matched_nodes = sum(action.matched_nodes for action in chosen)
        er = EvaluationResult(
            edge_tp=int(control["edge_tp"]) + matched_edges,
            edge_fp=int(control["edge_fp"]) + valid_edges - matched_edges,
            edge_fn=max(0, int(control["edge_fn"]) - matched_edges),
            division_tp=int(control["division_tp"]),
            division_fp=int(control["division_fp"]),
            division_fn=int(control["division_fn"]),
            num_pred_nodes=int(control["num_pred_nodes"]) + 2 * len(chosen),
        )
        recall = min(
            1.0,
            float(control["node_recall"])
            + matched_nodes / max(1, gt_node_counts[identity]),
        )
        row = per_sample_metrics(er, float(control["estimated_number_of_nodes"]), recall)
        row["division_jaccard"] = float(control["division_jaccard"])
        result.append({"identity": identity, **row})
    return result


def family_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        family: official_summary(
            [row for row in rows if row["identity"].split(":", 1)[1].startswith(family + "_")]
        )
        for family in ("44b6", "6bba")
    }


def deltas(candidate: dict[str, dict[str, Any]], control: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ("score", "adj_edge_jaccard", "edge_jaccard", "node_recall")
    return {
        family: {
            key: float(candidate[family][key]) - float(control[family][key])
            for key in keys
        }
        for family in candidate
    }


def build_selected_graph(
    full: SubmissionGraph,
    stats: dict[str, int],
    actions: list[Action],
    selected: set[str],
) -> SubmissionGraph:
    control, _ = ablate_gap2(full, stats)
    node_ids: set[int] = set()
    edge_pairs: set[tuple[int, int]] = set()
    for action in actions:
        if action.key in selected:
            node_ids.update((action.first_id, action.second_id))
            edge_pairs.update(action.incident_edges)
    graph = SubmissionGraph(
        full.dataset,
        {**control.nodes, **{node_id: full.nodes[node_id] for node_id in node_ids}},
        list(control.edges)
        + [SubmissionEdge(source, target) for source, target in sorted(edge_pairs)],
    )
    validate_graphs({full.dataset: graph}, max_outdegree=None)
    return graph


def main() -> int:
    args = parse_args()
    if args.mode == "tune" and args.population != "held_fit105":
        raise ValueError("Optuna tuning is restricted to held_fit105")
    if args.mode == "replay" and args.params is None:
        raise ValueError("--params is required in replay mode")

    manifest = json.loads(args.population_manifest.read_text(encoding="utf-8"))
    identities = [str(value) for value in manifest["populations"][args.population]]
    datasets = [identity.split(":", 1)[1] for identity in identities]
    full_score = json.loads(args.full_score.read_text(encoding="utf-8"))
    control_score = json.loads(args.control_score.read_text(encoding="utf-8"))
    if full_score.get("population") != args.population or control_score.get("population") != args.population:
        raise ValueError("score population mismatch")
    control_rows = {str(row["identity"]): row for row in control_score["samples"]}
    graphs = load_selected_graphs(args.submission, set(datasets))
    stats = load_stats(args.run_stats)

    import zarr
    from zarr.storage import ZipStore

    actions_by_identity: dict[str, list[Action]] = {}
    gt_by_identity: dict[str, dict[str, Any]] = {}
    scales: dict[str, tuple[float, float, float]] = {}
    gt_node_counts: dict[str, int] = {}
    store = ZipStore(args.archive, mode="r")
    try:
        for ordinal, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = parse_scale(dict(group.attrs))
            gt = read_geff_arrays(store, "train", dataset)
            actions = extract_actions(graphs[dataset], stats[dataset])
            actions = label_actions(graphs[dataset], actions, gt, scale)
            actions_by_identity[identity] = actions
            gt_by_identity[identity] = gt
            scales[identity] = scale
            gt_node_counts[identity] = len(gt["nodes"]["node_id"])
            print(f"features {ordinal}/{len(identities)} {identity} actions={len(actions)}", flush=True)

        actions = [action for identity in identities for action in actions_by_identity[identity]]
        matrix = np.asarray([action.features for action in actions], dtype=np.float64)
        center = np.median(matrix, axis=0)
        feature_scale = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
        feature_scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
        control_family = family_summaries(list(control_rows.values()))

        if args.mode == "tune":
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            best_details: dict[int, dict[str, Any]] = {}

            def objective(trial: Any) -> float:
                params = {
                    f"weight_{name}": trial.suggest_float(f"weight_{name}", -3.0, 3.0)
                    for name in FEATURE_NAMES
                }
                quantile = trial.suggest_float("selection_quantile", 0.90, 0.9995)
                raw_scores = score_vector(matrix, params, center, feature_scale)
                params["score_threshold"] = float(np.quantile(raw_scores, quantile))
                selected = select_actions(actions, params, center, feature_scale)
                if not selected:
                    return -1.0
                rows = estimated_rows(
                    identities,
                    actions_by_identity,
                    selected,
                    control_rows,
                    gt_node_counts,
                )
                summaries = family_summaries(rows)
                delta = deltas(summaries, control_family)
                min_adjusted = min(delta[family]["adj_edge_jaccard"] for family in delta)
                min_raw = min(delta[family]["edge_jaccard"] for family in delta)
                min_node = min(delta[family]["node_recall"] for family in delta)
                support = len(selected)
                # Constrained maximization encoded lexicographically: any raw
                # or node regression is dominated; then maximize the weakest
                # family adjusted gain, with a tiny overall/support tie-break.
                penalty = 0.0
                if min_raw < 0:
                    penalty += 10.0 * abs(min_raw)
                if min_node < -1e-12:
                    penalty += 10.0 * abs(min_node)
                value = min_adjusted - penalty + 1e-3 * (
                    sum(delta[family]["adj_edge_jaccard"] for family in delta) / 2.0
                )
                best_details[trial.number] = {
                    "params": params,
                    "selection_quantile": quantile,
                    "selected_actions": support,
                    "estimated_family_summary": summaries,
                    "estimated_family_delta": delta,
                    "objective": value,
                }
                return value

            sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
            tuned = best_details[study.best_trial.number]
            params = tuned["params"]
            optimization = {
                "library": "optuna",
                "version": optuna.__version__,
                "sampler": "TPESampler(multivariate=True)",
                "seed": args.seed,
                "trials": args.trials,
                "best_trial": study.best_trial.number,
                **tuned,
            }
        else:
            frozen = json.loads(args.params.read_text(encoding="utf-8"))
            params = dict(frozen["selector"]["params"])
            expected_center = np.asarray(frozen["selector"]["feature_center"], dtype=np.float64)
            expected_scale = np.asarray(frozen["selector"]["feature_scale"], dtype=np.float64)
            # Replay uses the fit-side normalization, never the validation population.
            center, feature_scale = expected_center, expected_scale
            optimization = {
                "library": "none",
                "mode": "frozen_replay",
                "source": str(args.params),
                "source_sha256": sha256_file(args.params),
            }

        selected = select_actions(actions, params, center, feature_scale)
        exact_rows = []
        structural = Counter()
        for ordinal, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            chosen = {action.key for action in actions_by_identity[identity]} & selected
            candidate = build_selected_graph(
                graphs[dataset], stats[dataset], actions_by_identity[identity], chosen
            )
            exact_rows.append(
                {
                    "identity": identity,
                    **graph_metric_row(candidate, gt_by_identity[identity], scales[identity]),
                }
            )
            structural["selected_actions"] += len(chosen)
            structural["selected_nodes"] += 2 * len(chosen)
            structural["selected_incident_edges"] += sum(
                len(action.incident_edges)
                for action in actions_by_identity[identity]
                if action.key in chosen
            )
            print(f"exact {ordinal}/{len(identities)} {identity} selected={len(chosen)}", flush=True)
    finally:
        store.close()

    exact_summary = official_summary(exact_rows)
    exact_family = family_summaries(exact_rows)
    exact_delta = deltas(exact_family, control_family)
    min_adjusted = min(value["adj_edge_jaccard"] for value in exact_delta.values())
    min_raw = min(value["edge_jaccard"] for value in exact_delta.values())
    min_node = min(value["node_recall"] for value in exact_delta.values())
    gate = (
        "GO"
        if min_adjusted >= 0.003 and min_raw >= 0.0 and min_node >= -1e-12 and selected
        else ("PARK" if min_adjusted > 0.0 and min_raw >= 0.0 and min_node >= -1e-12 else "DEAD")
    )
    payload = {
        "schema": "celltrack-alpha.gap2-high-abstention-selector.v1",
        "status": "COMPLETE",
        "population": args.population,
        "mode": args.mode,
        "selector": {
            "id": "NB-GAPSEL-v1",
            "feature_names": list(FEATURE_NAMES),
            "feature_center": center.tolist(),
            "feature_scale": feature_scale.tolist(),
            "params": params,
            "inference_inputs": "completed graph geometry/topology only; no labels or pixels",
        },
        "optimization": optimization,
        "population_actions": len(actions),
        "selected_action_keys": sorted(selected),
        "structural": dict(structural),
        "same_run_control_summary": official_summary(list(control_rows.values())),
        "exact_selected_summary": exact_summary,
        "exact_family_summary": exact_family,
        "exact_family_delta_vs_same_run_control": exact_delta,
        "gate": {
            "verdict": gate,
            "minimum_family_adjusted_delta": min_adjusted,
            "minimum_family_raw_delta": min_raw,
            "minimum_family_node_recall_delta": min_node,
            "minimum_adjusted_delta_required": 0.003,
        },
        "provenance": {
            "submission": {"path": str(args.submission), "sha256": sha256_file(args.submission)},
            "run_stats": {"path": str(args.run_stats), "sha256": sha256_file(args.run_stats)},
            "full_score": {"path": str(args.full_score), "sha256": sha256_file(args.full_score)},
            "control_score": {"path": str(args.control_score), "sha256": sha256_file(args.control_score)},
        },
        "samples": exact_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": payload["gate"], "structural": payload["structural"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
