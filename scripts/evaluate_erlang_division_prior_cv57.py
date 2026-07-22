#!/usr/bin/env python3
"""Fit and falsify an Erlang division-opening prior on frozen fit117/CV57."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import gamma
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
from celltrack_alpha import bh_graph  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    read_submission,
)
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.erlang-division-prior-cv57.v1"
EXPECTED_SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
EXPECTED_BASELINE = {
    "adj_edge_jaccard": 0.8711885823490648,
    "edge_jaccard": 0.8761665470208184,
    "division_tp": 0.0,
    "division_fp": 0.0,
    "division_fn": 42.0,
    "score": 0.8711885823490648,
}
SHAPES = (1, 2, 3, 4, 5)
DECISION_THRESHOLD = 0.002

DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_FRAGMENTS = (
    ROOT
    / "reports/g02/learned_bonus_sweep_holdout57_workers/bonus_1/"
    "learned_bonus_1__raw_distance_weight_0p05/fragments"
)
DEFAULT_PARITY = ROOT / "reports/g02/scorer_parity_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/erlang_division_prior_cv57_v1.json"


class ErlangExperimentError(RuntimeError):
    """The frozen offline experiment contract was not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ErlangExperimentError(f"JSON root is not an object: {path}")
    return value


def extract_root_to_division_intervals(
    gt: Mapping[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Measure division-frame minus earliest annotated ancestor frame."""

    nodes = gt["nodes"]
    times = {
        int(node_id): int(frame)
        for node_id, frame in zip(nodes["node_id"], nodes["t"], strict=True)
    }
    incoming: dict[int, int] = {}
    outgoing: dict[int, list[int]] = defaultdict(list)
    for raw_source, raw_target in gt["edges"]:
        source, target = int(raw_source), int(raw_target)
        if target in incoming:
            raise ErlangExperimentError("fit GT contains a merge")
        incoming[target] = source
        outgoing[source].append(target)

    intervals: list[int] = []
    records: list[dict[str, Any]] = []
    for division_id in sorted(node_id for node_id, children in outgoing.items() if len(children) == 2):
        current_id = division_id
        directly_observed_birth = False
        while current_id in incoming:
            predecessor_id = incoming[current_id]
            if len(outgoing[predecessor_id]) == 2:
                directly_observed_birth = True
                break
            current_id = predecessor_id
        birth_frame = times[current_id]
        division_frame = times[division_id]
        elapsed = division_frame - birth_frame
        if elapsed < 0:
            raise ErlangExperimentError("fit GT lineage runs backwards in time")
        intervals.append(elapsed)
        records.append(
            {
                "division_node_id": division_id,
                "birth_node_id": current_id,
                "birth_frame": birth_frame,
                "division_frame": division_frame,
                "elapsed_frames": elapsed,
                "birth_is_annotated_daughter": directly_observed_birth,
            }
        )
    return intervals, records


def fixed_shape_log_likelihood(values: np.ndarray, shape: int, rate: float) -> float:
    return float(
        np.sum(
            shape * math.log(rate)
            + (shape - 1) * np.log(values)
            - rate * values
            - math.lgamma(shape)
        )
    )


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
    }


def fit_erlang(
    intervals: Sequence[int],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    positive = np.asarray([value for value in intervals if value > 0], dtype=np.float64)
    if len(positive) < 2:
        raise ErlangExperimentError("fewer than two positive fit intervals")
    mean = float(np.mean(positive))
    continuous_shape, _location, continuous_scale = gamma.fit(positive, floc=0.0)

    rng = np.random.default_rng(240315011)
    sampled_means = np.empty(bootstrap_replicates, dtype=np.float64)
    sampled_shapes = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        sample = rng.choice(positive, size=len(positive), replace=True)
        sampled_means[index] = float(np.mean(sample))
        sampled_shapes[index] = float(gamma.fit(sample, floc=0.0)[0])

    settings: list[dict[str, Any]] = []
    for shape in SHAPES:
        rate = shape / mean
        bootstrap_rates = shape / sampled_means
        settings.append(
            {
                "shape": shape,
                "rate_per_frame": rate,
                "mean_frames": shape / rate,
                "log_likelihood": fixed_shape_log_likelihood(positive, shape, rate),
                "rate_bootstrap_95pct": [
                    float(np.percentile(bootstrap_rates, 2.5)),
                    float(np.percentile(bootstrap_rates, 97.5)),
                ],
            }
        )
    ranked = sorted(settings, key=lambda row: (-row["log_likelihood"], row["shape"]))
    return {
        "candidate_division_events": len(intervals),
        "usable_positive_intervals": int(len(positive)),
        "excluded_zero_length_roots": int(sum(value == 0 for value in intervals)),
        "interval_summary_frames": percentile_summary(positive),
        "interval_mean_bootstrap_95pct": [
            float(np.percentile(sampled_means, 2.5)),
            float(np.percentile(sampled_means, 97.5)),
        ],
        "continuous_gamma_diagnostic": {
            "shape": float(continuous_shape),
            "rate_per_frame": float(1.0 / continuous_scale),
            "shape_bootstrap_95pct": [
                float(np.percentile(sampled_shapes, 2.5)),
                float(np.percentile(sampled_shapes, 97.5)),
            ],
            "interpretation": "Diagnostic only; production sweep uses integer Erlang shapes.",
        },
        "fixed_shape_fits": settings,
        "maximum_likelihood_integer_shape": int(ranked[0]["shape"]),
        "bootstrap_replicates": bootstrap_replicates,
    }


def graph_inputs(
    graph: SubmissionGraph,
) -> tuple[list[bh_graph.GraphNode], list[bh_graph.GraphEdge]]:
    nodes = [
        bh_graph.GraphNode(node.node_id, node.t, node.z, node.y, node.x)
        for node in graph.nodes.values()
    ]
    by_id = {node.node_id: node for node in nodes}
    edges: list[bh_graph.GraphEdge] = []
    for edge in graph.edges:
        distance = bh_graph.physical_distance(by_id[edge.source_id], by_id[edge.target_id])
        edges.append(
            bh_graph.GraphEdge(
                edge.source_id,
                edge.target_id,
                distance,
                distance,
                "adjacent",
            )
        )
    return nodes, edges


def add_prior_divisions(
    graph: SubmissionGraph,
    prior: bh_graph.ErlangDivisionPrior,
) -> tuple[SubmissionGraph, dict[str, int | float | bool], list[tuple[int, int]]]:
    nodes, edges = graph_inputs(graph)
    result_edges, stats = bh_graph.add_safe_divisions(
        nodes,
        edges,
        lifetime_prior=prior,
    )
    original = {(edge.source_id, edge.target_id) for edge in graph.edges}
    added = sorted(
        (edge.source_id, edge.target_id)
        for edge in result_edges
        if (edge.source_id, edge.target_id) not in original
    )
    result = graph.copy()
    result.edges.extend(SubmissionEdge(source, target) for source, target in added)
    return result, stats, added


def prediction_rows(graph: SubmissionGraph) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
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
    return nodes, edges


def aggregate_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    names = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    return {name: int(sum(float(row[name]) for row in rows)) for name in names}


def validate_baseline(summary: Mapping[str, Any], counts: Mapping[str, int]) -> None:
    observed = {name: float(summary[name]) for name in EXPECTED_BASELINE}
    if observed != EXPECTED_BASELINE:
        raise ErlangExperimentError(f"exact divisions-off baseline changed: {observed}")
    if {name: counts[name] for name in ("edge_tp", "edge_fp", "edge_fn")} != {
        "edge_tp": 34174,
        "edge_fp": 2736,
        "edge_fn": 2094,
    }:
        raise ErlangExperimentError("exact baseline edge totals changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.output.exists():
        raise ErlangExperimentError(f"refusing to overwrite versioned report: {args.output}")
    if tuple(args.shape) != SHAPES:
        raise ErlangExperimentError(f"shape sweep must be exactly {SHAPES}")
    if sha256_file(args.split) != EXPECTED_SPLIT_SHA256:
        raise ErlangExperimentError("frozen grouped split changed")

    split = load_json(args.split)
    fit_ids = tuple(split["folds"]["0"]["fit_video_ids"])
    if len(fit_ids) != 117:
        raise ErlangExperimentError("fold-0 fit scope is not 117 videos")
    scorer = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    archive_ids = scoreall.archive_train_video_ids(args.zip)
    selected = score_bh_cv57.exact_selection(
        split,
        split_path=args.split,
        archive_video_ids=archive_ids,
    )
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(args.baseline_fragments)),
        selected,
    )
    baseline_paths = dict(manifest.ordered)

    interval_records: list[dict[str, Any]] = []
    interval_values: list[int] = []
    fit_divisions_by_family: Counter[str] = Counter()
    fit_divisions_by_video: dict[str, int] = {}
    baseline_rows: list[dict[str, Any]] = []
    rows_by_shape: dict[int, list[dict[str, Any]]] = {shape: [] for shape in SHAPES}
    audit_by_shape: dict[int, list[dict[str, Any]]] = {shape: [] for shape in SHAPES}

    with ZipStore(args.zip, mode="r") as store:
        for identity in fit_ids:
            dataset = identity.split(":", 1)[1]
            intervals, records = extract_root_to_division_intervals(
                read_geff_arrays(store, "train", dataset)
            )
            interval_values.extend(intervals)
            fit_divisions_by_video[identity] = len(intervals)
            fit_divisions_by_family[dataset[:4]] += len(intervals)
            interval_records.extend({"identity": identity, **record} for record in records)

        fit = fit_erlang(interval_values, bootstrap_replicates=args.bootstrap)
        rates = {
            int(row["shape"]): float(row["rate_per_frame"])
            for row in fit["fixed_shape_fits"]
        }

        for index, identity in enumerate(selected, start=1):
            dataset = identity.split(":", 1)[1]
            graphs = read_submission(baseline_paths[identity])
            if set(graphs) != {dataset}:
                raise ErlangExperimentError(f"baseline fragment identity mismatch: {identity}")
            baseline = graphs[dataset]
            baseline_outdegree = Counter(edge.source_id for edge in baseline.edges)
            if max(baseline_outdegree.values(), default=0) > 1:
                raise ErlangExperimentError(f"baseline is not divisions-off: {identity}")

            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            gt = read_geff_arrays(store, "train", dataset)
            base_nodes, base_edges = prediction_rows(baseline)
            baseline_rows.append(
                {"identity": identity, **official_metric_row(base_nodes, base_edges, gt, scale)}
            )

            for shape in SHAPES:
                graph, stats, added = add_prior_divisions(
                    baseline,
                    bh_graph.ErlangDivisionPrior(shape=shape, rate=rates[shape]),
                )
                nodes, edges = prediction_rows(graph)
                rows_by_shape[shape].append(
                    {"identity": identity, **official_metric_row(nodes, edges, gt, scale)}
                )
                audit_by_shape[shape].append(
                    {
                        "identity": identity,
                        "division_edges_added": len(added),
                        "added_edges": [[source, target] for source, target in added],
                        "solver_stats": stats,
                    }
                )
            print(f"Erlang CV57 {index:02d}/{len(selected)} {identity}", flush=True)

    baseline_summary = official_summary(baseline_rows)
    baseline_counts = aggregate_counts(baseline_rows)
    validate_baseline(baseline_summary, baseline_counts)

    settings: list[dict[str, Any]] = []
    for shape in SHAPES:
        summary = official_summary(rows_by_shape[shape])
        counts = aggregate_counts(rows_by_shape[shape])
        gain = float(summary["division_jaccard"]) - float(baseline_summary["division_jaccard"])
        weighted_gain = 0.1 * gain
        settings.append(
            {
                "shape": shape,
                "rate_per_frame": rates[shape],
                "summary": summary,
                "counts": counts,
                "division_jaccard_gain": gain,
                "weighted_division_gain": weighted_gain,
                "adjusted_edge_delta": float(summary["adj_edge_jaccard"])
                - float(baseline_summary["adj_edge_jaccard"]),
                "combined_score_delta": float(summary["score"])
                - float(baseline_summary["score"]),
                "division_edges_added": sum(
                    int(row["division_edges_added"]) for row in audit_by_shape[shape]
                ),
                "per_video_audit": audit_by_shape[shape],
            }
        )
    ranked = sorted(
        settings,
        key=lambda row: (
            -float(row["weighted_division_gain"]),
            -int(row["counts"]["division_tp"]),
            int(row["counts"]["division_fp"]),
            int(row["shape"]),
        ),
    )
    best = ranked[0]
    genuine_tp = int(best["counts"]["division_tp"]) > 0
    passes_gain = float(best["weighted_division_gain"]) >= DECISION_THRESHOLD
    production_enabled = genuine_tp and passes_gain

    report = {
        "schema": SCHEMA,
        "status": "completed",
        "fit": {
            **fit,
            "scope": {
                "fold": 0,
                "fit_video_count": len(fit_ids),
                "heldout_or_test_used": False,
                "division_events_by_family": dict(sorted(fit_divisions_by_family.items())),
                "division_bearing_video_count": sum(value > 0 for value in fit_divisions_by_video.values()),
            },
            "measurement_contract": {
                "birth_definition": "earliest annotated ancestor frame for the dividing GT cell",
                "directly_observed_daughter_birth_events": sum(
                    bool(row["birth_is_annotated_daughter"]) for row in interval_records
                ),
                "root_started_events": sum(
                    not bool(row["birth_is_annotated_daughter"]) for row in interval_records
                ),
                "uncertainty_warning": (
                    "All fit divisions are root-started annotation spans, not multigeneration "
                    "daughter-to-division lifetimes. Three zero-frame roots were excluded. "
                    "The small sample and annotation truncation make the biological prior uncertain."
                ),
            },
            "interval_records": interval_records,
        },
        "baseline": {"summary": baseline_summary, "counts": baseline_counts},
        "settings": settings,
        "decision": {
            "rule": "reject if best division_jaccard gain * 0.1 is below 0.002",
            "threshold": DECISION_THRESHOLD,
            "best_shape": int(best["shape"]),
            "best_weighted_division_gain": float(best["weighted_division_gain"]),
            "best_division_tp": int(best["counts"]["division_tp"]),
            "best_division_fp": int(best["counts"]["division_fp"]),
            "gain_rule_passed": passes_gain,
            "genuine_patched_metric_tp": genuine_tp,
            "status": "ACCEPT" if production_enabled else "REJECT",
            "production_divisions_enabled": production_enabled,
            "interpretation": (
                "Enable only after both the weighted-gain threshold and genuine patched-metric TP condition pass."
                if production_enabled
                else "Keep production divisions OFF."
            ),
        },
        "execution": {
            "cpu_only": True,
            "gpu_used": False,
            "training": False,
            "detector_inference": False,
            "graph_postprocess_only": True,
            "patched_official_metric": True,
            "kaggle_action": False,
            "production_config_changed": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": {
            "archive": {"path": str(args.zip), "sha256": sha256_file(args.zip)},
            "split": {"path": str(args.split), "sha256": sha256_file(args.split)},
            "baseline_fragments": str(args.baseline_fragments),
            "scorer": scorer,
            "graph_source": {
                "path": str(ROOT / "src/celltrack_alpha/bh_graph.py"),
                "sha256": sha256_file(ROOT / "src/celltrack_alpha/bh_graph.py"),
            },
            "script_source": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    value.add_argument("--baseline-fragments", type=Path, default=DEFAULT_FRAGMENTS)
    value.add_argument("--scorer-parity", type=Path, default=DEFAULT_PARITY)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--shape", type=int, nargs="+", default=list(SHAPES))
    value.add_argument("--bootstrap", type=int, default=2000)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.bootstrap <= 0:
        raise ErlangExperimentError("bootstrap replicate count must be positive")
    result = run(args)
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
