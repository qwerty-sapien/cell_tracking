#!/usr/bin/env python3
"""Exact, exclusive CV57 residual taxonomy for the frozen APNCC control.

The official evaluator supplies the scored edge FP/FN events.  Every scored
edge FN is assigned exactly once, in this priority-free partition:

* detection: at least one GT endpoint was not distance-matched;
* fragmentation: both endpoints were matched but their prediction nodes are
  in different weak components of the submitted graph;
* ordinary association: both endpoints were matched and remain in the same
  weak prediction component, yet the GT edge was not recovered.

Scored edge FPs are ordinary association errors.  Division TP/FP/FN events
are a fourth, entirely separate category.  Thus the three edge buckets are
disjoint and exhaustive for official scored edge errors; divisions are never
used to classify an edge event.

The score counterfactuals are score-accounting counterfactuals, not a claim
that independently editing graph rows can achieve each repair without side
effects.  Detection additionally restores the evaluator's node-count term to
its per-video estimated node count.  The full counterfactual therefore has
perfect edge/division events and a neutral node-count factor (the theoretical
fully-fixed ceiling of 1.1 on this division-containing CV57 split).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import tracksdata as td
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[4]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external" / "kaggle-cell-tracking-competition" / "src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import score_bh_cv57  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    official_summary,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.division_metrics import evaluate_divisions  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    EvaluationResult,
    _evaluate,
    _evaluate_matched_graph,
    node_recall,
    per_sample_metrics,
)


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_CONTROL = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/control"
)
DEFAULT_POPULATIONS = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/POPULATIONS.json"
)
DEFAULT_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_PARITY = ROOT / "reports/g02/scorer_parity_v2.json"

CATEGORY_ORDER = ("detection", "ordinary_association", "track_fragmentation", "division")
EDGE_KEYS = ("edge_tp", "edge_fp", "edge_fn")
DIVISION_KEYS = ("division_tp", "division_fp", "division_fn")


class TaxonomyError(RuntimeError):
    """The reproducibility or exclusivity contract did not hold."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaxonomyError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TaxonomyError(f"{label} root must be an object: {path}")
    return value


def weak_components(node_ids: Iterable[int], edges: Iterable[tuple[int, int]]) -> dict[int, int]:
    """Return deterministic weak-component roots for one submitted graph."""

    parent = {int(value): int(value) for value in node_ids}

    def find(value: int) -> int:
        root = value
        while root != parent[root]:
            root = parent[root]
        while value != root:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    for raw_left, raw_right in edges:
        left = find(int(raw_left))
        right = find(int(raw_right))
        if left == right:
            continue
        if right < left:
            left, right = right, left
        parent[right] = left
    return {node_id: find(node_id) for node_id in parent}


def _graph_node_match_map(prediction: td.graph.BaseGraph) -> tuple[dict[int, int], dict[int, int]]:
    """Return pred→GT and GT→pred mappings from the evaluator-mutated graph."""

    rows = prediction.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    ).iter_rows()
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for raw_pred, raw_gt in rows:
        if raw_gt is None or int(raw_gt) == -1:
            continue
        pred_id = int(raw_pred)
        gt_id = int(raw_gt)
        if gt_id in gt_to_pred:
            raise TaxonomyError(f"official matcher assigned GT node {gt_id} twice")
        pred_to_gt[pred_id] = gt_id
        gt_to_pred[gt_id] = pred_id
    return pred_to_gt, gt_to_pred


def _graph_edges(graph: td.graph.BaseGraph) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in graph.edge_attrs()
        .select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET)
        .iter_rows()
    }


def _safe_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TaxonomyError(f"non-finite metric value: {value!r}")
    return result


def score_from_counts(row: Mapping[str, Any]) -> dict[str, float]:
    """Run the exact official score arithmetic from event counts and node count."""

    result = EvaluationResult(
        edge_tp=int(row["edge_tp"]),
        edge_fp=int(row["edge_fp"]),
        edge_fn=int(row["edge_fn"]),
        division_tp=int(row["division_tp"]),
        division_fp=int(row["division_fp"]),
        division_fn=int(row["division_fn"]),
        num_pred_nodes=int(row["num_pred_nodes"]),
    )
    metrics = per_sample_metrics(
        result,
        float(row["estimated_number_of_nodes"]),
        float(row["node_recall"]),
    )
    denominator = metrics["division_tp"] + metrics["division_fp"] + metrics["division_fn"]
    metrics["division_jaccard"] = metrics["division_tp"] / denominator if denominator else float("nan")
    metrics["estimated_number_of_nodes"] = float(row["estimated_number_of_nodes"])
    return {key: float(value) for key, value in metrics.items()}


def local_visualizer_score(row: Mapping[str, Any]) -> float:
    """Match build_track_visualizer.py's per-video score display semantics."""

    adjusted = _safe_float(row["adj_edge_jaccard"])
    denominator = sum(int(row[key]) for key in DIVISION_KEYS)
    if denominator == 0:
        return adjusted
    division = float(row["division_jaccard"])
    if not math.isfinite(division):
        division = 0.0
    return adjusted + 0.1 * division


def classify_video(
    *,
    identity: str,
    graph: Any,
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    """Replay official matching and return an exclusive event taxonomy."""

    dataset = identity.split(":", 1)[1]
    nodes = [
        {
            "dataset": dataset,
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": float(node.z),
            "y": float(node.y),
            "x": float(node.x),
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": int(edge.source_id), "target_id": int(edge.target_id)}
        for edge in graph.edges
    ]
    prediction = prediction_to_graph(nodes, edges)
    truth = geff_arrays_to_graph(gt)
    if prediction.num_nodes() == 0 or prediction.num_edges() == 0:
        raise TaxonomyError(f"{identity} unexpectedly has no scored prediction topology")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _evaluate(prediction, truth, "jaccard", scale_zyx, 7.0)
    pred_to_gt, gt_to_pred = _graph_node_match_map(prediction)
    edge_table = _evaluate_matched_graph(prediction, truth)

    source_key = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
    target_key = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
    matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK
    official_tp_edges: set[tuple[int, int]] = set()
    official_fp = 0
    for record in edge_table.select(source_key, target_key, matched_key, "pred_valid").iter_rows(named=True):
        source = int(record[source_key])
        target = int(record[target_key])
        matched = bool(record[matched_key])
        valid = bool(record["pred_valid"])
        if matched:
            if source not in pred_to_gt or target not in pred_to_gt:
                raise TaxonomyError(f"{identity} has a matched edge with an unmatched endpoint")
            mapped = (pred_to_gt[source], pred_to_gt[target])
            if mapped in official_tp_edges:
                raise TaxonomyError(f"{identity} has duplicate scored TP GT edge {mapped}")
            official_tp_edges.add(mapped)
        elif valid:
            official_fp += 1

    gt_edges = _graph_edges(truth)
    missing_gt_edges = gt_edges - official_tp_edges
    prediction_components = weak_components(prediction.node_ids(), _graph_edges(prediction))
    detection_fn = 0
    fragmentation_fn = 0
    ordinary_fn = 0
    for source, target in missing_gt_edges:
        pred_source = gt_to_pred.get(source)
        pred_target = gt_to_pred.get(target)
        if pred_source is None or pred_target is None:
            detection_fn += 1
        elif prediction_components[pred_source] != prediction_components[pred_target]:
            fragmentation_fn += 1
        else:
            ordinary_fn += 1
    if detection_fn + fragmentation_fn + ordinary_fn != len(missing_gt_edges):
        raise TaxonomyError(f"{identity} edge-FN partition is not exhaustive")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        divisions = evaluate_divisions(prediction, truth, scale=scale_zyx, max_distance=7.0)
    evaluated = {
        "edge_tp": len(official_tp_edges),
        "edge_fp": official_fp,
        "edge_fn": len(missing_gt_edges),
        "division_tp": int(divisions.tp),
        "division_fp": int(divisions.fp),
        "division_fn": int(divisions.fn),
        "num_pred_nodes": int(prediction.num_nodes()),
        "node_recall": float(node_recall(prediction, truth)),
        "estimated_number_of_nodes": float(gt["estimated_number_of_nodes"]),
    }
    metrics = score_from_counts(evaluated)
    categories = {
        "detection": {
            "score_event_count": detection_fn,
            "edge_fn": detection_fn,
            "unmatched_gt_nodes": int(truth.num_nodes() - len(gt_to_pred)),
            "matched_gt_nodes": int(len(gt_to_pred)),
            "node_count_delta_from_estimate": (
                int(prediction.num_nodes()) - float(gt["estimated_number_of_nodes"])
            ),
        },
        "ordinary_association": {
            "score_event_count": official_fp + ordinary_fn,
            "edge_fp": official_fp,
            "edge_fn": ordinary_fn,
        },
        "track_fragmentation": {
            "score_event_count": fragmentation_fn,
            "edge_fn": fragmentation_fn,
            "definition": "both GT endpoints matched; their submitted prediction nodes are in different weak components",
        },
        "division": {
            "score_event_count": int(divisions.fp) + int(divisions.fn),
            "division_tp": int(divisions.tp),
            "division_fp": int(divisions.fp),
            "division_fn": int(divisions.fn),
        },
    }
    if categories["ordinary_association"]["edge_fp"] != evaluated["edge_fp"]:
        raise TaxonomyError(f"{identity} FP partition disagrees with official scorer")
    return {
        "identity": identity,
        "family": dataset.split("_", 1)[0],
        "metrics": {**evaluated, **metrics},
        "categories": categories,
        "partition_validation": {
            "official_edge_fn": evaluated["edge_fn"],
            "detection_edge_fn": detection_fn,
            "fragmentation_edge_fn": fragmentation_fn,
            "ordinary_association_edge_fn": ordinary_fn,
            "sum_edge_fn": detection_fn + fragmentation_fn + ordinary_fn,
            "official_edge_fp": evaluated["edge_fp"],
            "ordinary_association_edge_fp": official_fp,
            "official_edge_residual": evaluated["edge_fp"] + evaluated["edge_fn"],
            "exclusive_edge_residual": detection_fn + fragmentation_fn + ordinary_fn + official_fp,
        },
    }


def transformed_metric_row(row: Mapping[str, Any], enabled: frozenset[str]) -> dict[str, Any]:
    """Apply one or more disjoint score-accounting repairs to one video."""

    source = row["metrics"]
    categories = row["categories"]
    values = {
        key: int(source[key]) for key in (*EDGE_KEYS, *DIVISION_KEYS, "num_pred_nodes")
    }
    values["estimated_number_of_nodes"] = float(source["estimated_number_of_nodes"])
    values["node_recall"] = float(source["node_recall"])

    if "detection" in enabled:
        fixed = int(categories["detection"]["edge_fn"])
        values["edge_tp"] += fixed
        values["edge_fn"] -= fixed
        # A fully corrected detection output has the evaluator's neutral
        # node-count factor.  Sparse unmatched predictions remain unknown in
        # the visualizer; this is only a score-ceiling accounting operation.
        values["num_pred_nodes"] = int(round(values["estimated_number_of_nodes"]))
        values["node_recall"] = 1.0
    if "ordinary_association" in enabled:
        fixed_fn = int(categories["ordinary_association"]["edge_fn"])
        fixed_fp = int(categories["ordinary_association"]["edge_fp"])
        values["edge_tp"] += fixed_fn
        values["edge_fn"] -= fixed_fn
        values["edge_fp"] -= fixed_fp
    if "track_fragmentation" in enabled:
        fixed = int(categories["track_fragmentation"]["edge_fn"])
        values["edge_tp"] += fixed
        values["edge_fn"] -= fixed
    if "division" in enabled:
        values["division_tp"] += int(categories["division"]["division_fn"])
        values["division_fn"] = 0
        values["division_fp"] = 0
    if any(values[key] < 0 for key in (*EDGE_KEYS, *DIVISION_KEYS)):
        raise TaxonomyError(f"negative count after score counterfactual for {row['identity']}")
    return {**values, **score_from_counts(values)}


def summarise_counterfactual(rows: Sequence[Mapping[str, Any]], enabled: frozenset[str]) -> dict[str, Any]:
    scored = [transformed_metric_row(row, enabled) for row in rows]
    return official_summary(scored)


def local_counterfactual_score(row: Mapping[str, Any], enabled: frozenset[str]) -> float:
    return local_visualizer_score(transformed_metric_row(row, enabled))


def shapley_score_attribution(
    score_for_enabled: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Exact four-player Shapley attribution for the nonlinear score formula."""

    cache: dict[frozenset[str], float] = {}

    def score(enabled: frozenset[str]) -> float:
        if enabled not in cache:
            cache[enabled] = float(score_for_enabled(enabled))
        return cache[enabled]

    values = {category: 0.0 for category in CATEGORY_ORDER}
    for order in itertools.permutations(CATEGORY_ORDER):
        active: frozenset[str] = frozenset()
        before = score(active)
        for category in order:
            active = active | {category}
            after = score(active)
            values[category] += after - before
            before = after
    divisor = math.factorial(len(CATEGORY_ORDER))
    values = {category: value / divisor for category, value in values.items()}
    return values, {"|".join(sorted(key)): value for key, value in sorted(cache.items(), key=lambda item: (len(item[0]), sorted(item[0])))}


def _compare_authoritative_rows(
    observed: Sequence[Mapping[str, Any]], reference_path: Path
) -> dict[str, Any]:
    reference = read_object(reference_path, "authoritative APNCC scorer control")
    samples = reference.get("samples")
    if not isinstance(samples, list) or len(samples) != 57:
        raise TaxonomyError("authoritative APNCC control lacks exactly 57 samples")
    expected = {str(row["identity"]): row for row in samples if isinstance(row, dict)}
    if len(expected) != 57 or set(expected) != {str(row["identity"]) for row in observed}:
        raise TaxonomyError("authoritative APNCC rows do not cover the replayed CV57 identities")
    metric_keys = (*EDGE_KEYS, *DIVISION_KEYS, "num_pred_nodes")
    field_mismatches: list[dict[str, Any]] = []
    for row in observed:
        identity = str(row["identity"])
        for key in metric_keys:
            actual = int(row["metrics"][key])
            wanted = int(float(expected[identity][key]))
            if actual != wanted:
                field_mismatches.append({"identity": identity, "field": key, "actual": actual, "expected": wanted})
        for key in ("node_recall", "edge_jaccard", "adj_edge_jaccard"):
            actual = float(row["metrics"][key])
            wanted = float(expected[identity][key])
            if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-15):
                field_mismatches.append({"identity": identity, "field": key, "actual": actual, "expected": wanted})
    observed_summary = official_summary([row["metrics"] for row in observed])
    reference_summary = reference.get("summary", {})
    for key in ("edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score", "node_recall"):
        if not math.isclose(float(observed_summary[key]), float(reference_summary[key]), rel_tol=0.0, abs_tol=1e-15):
            field_mismatches.append({"identity": "__summary__", "field": key, "actual": observed_summary[key], "expected": reference_summary[key]})
    if field_mismatches:
        raise TaxonomyError(f"current scorer replay differs from APNCC control: {field_mismatches[:3]}")
    return {
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "per_video_metric_fields_checked": list(metric_keys) + ["node_recall", "edge_jaccard", "adj_edge_jaccard"],
        "summary_fields_checked": ["edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score", "node_recall"],
        "mismatch_count": 0,
    }


def build_indexed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -(int(row["metrics"]["edge_fp"]) + int(row["metrics"]["edge_fn"])),
            str(row["identity"]),
        ),
    )
    total_residual = sum(int(row["metrics"]["edge_fp"]) + int(row["metrics"]["edge_fn"]) for row in ordered)
    cumulative = 0
    indexed: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        residual = int(row["metrics"]["edge_fp"]) + int(row["metrics"]["edge_fn"])
        cumulative += residual
        baseline = local_visualizer_score(row["metrics"])
        full = local_counterfactual_score(row, frozenset(CATEGORY_ORDER))
        shapley, _ = shapley_score_attribution(
            lambda enabled, selected=row: local_counterfactual_score(selected, enabled)
        )
        categories: dict[str, Any] = {}
        for category in CATEGORY_ORDER:
            one_way = local_counterfactual_score(row, frozenset({category}))
            categories[category] = {
                **dict(row["categories"][category]),
                "score_if_only_fixed": one_way,
                "standalone_score_delta": one_way - baseline,
                "shapley_score_delta": shapley[category],
            }
        indexed.append(
            {
                "rank": rank,
                "identity": row["identity"],
                "dataset": str(row["identity"]).split(":", 1)[1],
                "family": row["family"],
                "residual": residual,
                "error_contribution": residual / total_residual if total_residual else 0.0,
                "cumulative_error_contribution": cumulative / total_residual if total_residual else 0.0,
                "baseline_visualizer_score": baseline,
                "fully_fixed_theoretical_score_ceiling": full,
                "fully_fixed_score_delta": full - baseline,
                "categories": categories,
                "official_metrics": dict(row["metrics"]),
                "partition_validation": dict(row["partition_validation"]),
            }
        )
    return indexed


def make_markdown(payload: Mapping[str, Any], *, baseline_label: str) -> str:
    global_result = payload["global"]
    baseline = global_result["baseline_summary"]
    ceiling = global_result["fully_fixed_theoretical_ceiling"]
    categories = global_result["categories"]
    rows = payload["per_video_index"]
    summary_lines = []
    for category in CATEGORY_ORDER:
        record = categories[category]
        summary_lines.append(
            f"| {category} | {record['score_event_count']:,} | {record['shapley_score_delta']:+.12f} | "
            f"{record['score_if_only_fixed']:.12f} |"
        )
    top = rows[:10]
    video_lines = []
    for row in top:
        count = row["categories"]
        video_lines.append(
            f"| {row['rank']} | {row['identity']} | {row['residual']} | "
            f"{count['detection']['edge_fn']} | {count['ordinary_association']['edge_fp'] + count['ordinary_association']['edge_fn']} | "
            f"{count['track_fragmentation']['edge_fn']} | {count['division']['score_event_count']} |"
        )
    validation = payload["validation"]
    commands = "\n".join(f"    {command}" for command in payload["commands"])
    return f"""# CV57 residual error taxonomy (exclusive, {baseline_label})

The current scorer replay exactly reproduces the frozen APNCC CV57 score:
`{baseline['score']:.15f}`.  The four edge/division categories are disjoint:
each scorer-valid edge FP is ordinary association; each official edge FN is
exactly one of detection, matched-lineage fragmentation, or ordinary
association; division FP/FN events are separate.

The full score-accounting repair ceiling is `{ceiling['score']:.15f}`
(`{ceiling['score'] - baseline['score']:+.15f}` from baseline).  This is the
theoretical 1.1 ceiling for a split that contains division events: perfect
edge score 1.0 plus perfect division Jaccard weighted by 0.1.  It is not a
claim that the independent graph edits are jointly realizable without side
effects.

Dominant category by exact nonlinear-score Shapley attribution:
**{global_result['dominant_category']['name']}**
(`{global_result['dominant_category']['shapley_score_delta']:+.12f}`).

| Category | Exclusive scorer events | Exact Shapley score impact | Score if only fixed |
| --- | ---: | ---: | ---: |
{chr(10).join(summary_lines)}

`score_if_only_fixed` is deliberately shown alongside the additive Shapley
attribution because the official Jaccard and node adjustment are nonlinear.
The four Shapley values sum exactly (within floating-point tolerance) to the
full-ceiling delta.

## Worst videos in visualizer-compatible order

Sorted by `edge_fp + edge_fn` descending, then identity ascending; `rank`,
`residual`, and contribution fields have the same ordering contract as
`scripts/build_track_visualizer.py`.

| Rank | Video | Edge residual | Detection FN | Ordinary edge events | Fragmentation FN | Division errors |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(video_lines)}

The full 57-video stable index and all one-way/Shapley counterfactuals are in
`CV57_RESIDUAL_ERROR_TAXONOMY.json`.

## Inputs and validation

- Current scorer parity: `{payload['scorer']['parity_report']['path']}`
  (`{payload['scorer']['parity_report']['sha256']}`).
- Frozen `{payload['scope']['baseline']}` control: `{payload['inputs']['control_score']['path']}`
  (`{payload['inputs']['control_score']['sha256']}`).
- Submission: `{payload['inputs']['submission']['path']}`
  (`{payload['inputs']['submission']['sha256']}`).
- Raw archive: `{payload['inputs']['archive']['path']}`
  (`{payload['inputs']['archive']['sha256']}`).
- Replay validation: 57/57 per-video official count/metric rows and all five
  aggregate metric fields match the frozen control; edge partition and Shapley
  conservation checks passed (`{validation['status']}`).

## Reproduction command

```sh
{commands}
```
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise TaxonomyError(f"{name}=1 is required for this replay")
    scorer = score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    populations = read_object(args.populations, "public TemporalUNet populations")
    identities = [str(value) for value in populations.get("populations", {}).get("cv57", [])]
    if len(identities) != 57 or len(set(identities)) != 57:
        raise TaxonomyError("population manifest does not contain exactly 57 unique CV57 identities")
    control = read_object(args.control_score, "scorer control")
    submission_path = Path(str(control.get("submission", {}).get("path", "")))
    if not submission_path.is_absolute():
        submission_path = ROOT / submission_path
    if not submission_path.is_file():
        raise TaxonomyError(f"control submission is unavailable: {submission_path}")
    graph_policy = str(control.get("outdegree_diagnostics", {}).get("policy", ""))
    competition_faithful = graph_policy == "competition_faithful_no_outdegree_cap"
    graphs = read_submission(
        submission_path,
        max_outdegree=None if competition_faithful else 2,
    )
    expected_datasets = {identity.split(":", 1)[1] for identity in identities}
    # The control is the frozen 162-video union CSV.  This taxonomy must use
    # exactly its CV57 subset, while rejecting a malformed control that lacks
    # any selected identity.
    if not expected_datasets.issubset(graphs):
        raise TaxonomyError("APNCC union submission does not cover every CV57 dataset")

    rows: list[dict[str, Any]] = []
    store = ZipStore(args.archive, mode="r")
    try:
        for index, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            rows.append(
                classify_video(
                    identity=identity,
                    graph=graphs[dataset],
                    gt=gt,
                    scale_zyx=tuple(float(value) for value in parse_scale(dict(group.attrs))),
                )
            )
            print(f"[{index:02d}/57] {identity}", file=sys.stderr, flush=True)
    finally:
        store.close()

    reference_validation = _compare_authoritative_rows(rows, args.control_score)
    for row in rows:
        partition = row["partition_validation"]
        if partition["sum_edge_fn"] != partition["official_edge_fn"]:
            raise TaxonomyError(f"{row['identity']} has non-conserving FN partition")
        if partition["exclusive_edge_residual"] != partition["official_edge_residual"]:
            raise TaxonomyError(f"{row['identity']} has non-conserving edge residual partition")

    global_baseline = official_summary([row["metrics"] for row in rows])
    global_scores = {
        frozenset(enabled): summarise_counterfactual(rows, frozenset(enabled))
        for size in range(len(CATEGORY_ORDER) + 1)
        for enabled in itertools.combinations(CATEGORY_ORDER, size)
    }
    global_shapley, global_cache = shapley_score_attribution(
        lambda enabled: float(global_scores[enabled]["score"])
    )
    full_enabled = frozenset(CATEGORY_ORDER)
    global_ceiling = global_scores[full_enabled]
    total_delta = float(global_ceiling["score"]) - float(global_baseline["score"])
    if not math.isclose(sum(global_shapley.values()), total_delta, rel_tol=0.0, abs_tol=1e-13):
        raise TaxonomyError("global Shapley score impacts do not sum to the full-ceiling delta")

    totals: dict[str, dict[str, int | float]] = {
        "detection": {
            "edge_fn": sum(int(row["categories"]["detection"]["edge_fn"]) for row in rows),
            "unmatched_gt_nodes": sum(int(row["categories"]["detection"]["unmatched_gt_nodes"]) for row in rows),
        },
        "ordinary_association": {
            "edge_fp": sum(int(row["categories"]["ordinary_association"]["edge_fp"]) for row in rows),
            "edge_fn": sum(int(row["categories"]["ordinary_association"]["edge_fn"]) for row in rows),
        },
        "track_fragmentation": {
            "edge_fn": sum(int(row["categories"]["track_fragmentation"]["edge_fn"]) for row in rows),
        },
        "division": {
            key: sum(int(row["categories"]["division"][key]) for row in rows)
            for key in ("division_tp", "division_fp", "division_fn")
        },
    }
    category_results: dict[str, Any] = {}
    for category in CATEGORY_ORDER:
        counts = dict(totals[category])
        if category == "detection":
            score_events = int(counts["edge_fn"])
        elif category == "ordinary_association":
            score_events = int(counts["edge_fp"]) + int(counts["edge_fn"])
        elif category == "track_fragmentation":
            score_events = int(counts["edge_fn"])
        else:
            score_events = int(counts["division_fp"]) + int(counts["division_fn"])
        one_way = global_scores[frozenset({category})]
        category_results[category] = {
            **counts,
            "score_event_count": score_events,
            "score_if_only_fixed": float(one_way["score"]),
            "standalone_score_delta": float(one_way["score"]) - float(global_baseline["score"]),
            "shapley_score_delta": global_shapley[category],
        }
    edge_partition = {
        "official_edge_fp": int(sum(int(row["metrics"]["edge_fp"]) for row in rows)),
        "official_edge_fn": int(sum(int(row["metrics"]["edge_fn"]) for row in rows)),
        "ordinary_association_edge_fp": int(totals["ordinary_association"]["edge_fp"]),
        "detection_edge_fn": int(totals["detection"]["edge_fn"]),
        "ordinary_association_edge_fn": int(totals["ordinary_association"]["edge_fn"]),
        "track_fragmentation_edge_fn": int(totals["track_fragmentation"]["edge_fn"]),
    }
    if edge_partition["official_edge_fp"] != edge_partition["ordinary_association_edge_fp"]:
        raise TaxonomyError("global FP partition does not reproduce official FP")
    if edge_partition["official_edge_fn"] != sum(
        edge_partition[key]
        for key in ("detection_edge_fn", "ordinary_association_edge_fn", "track_fragmentation_edge_fn")
    ):
        raise TaxonomyError("global FN partition does not reproduce official FN")

    indexed = build_indexed_rows(rows)
    top_identity = [row["identity"] for row in indexed[:10]]
    expected_top = sorted(
        rows,
        key=lambda row: (
            -(int(row["metrics"]["edge_fp"]) + int(row["metrics"]["edge_fn"])),
            str(row["identity"]),
        ),
    )[:10]
    if top_identity != [row["identity"] for row in expected_top]:
        raise TaxonomyError("visualizer-compatible stable index order drifted")
    dominant = max(CATEGORY_ORDER, key=lambda name: (global_shapley[name], -CATEGORY_ORDER.index(name)))

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    payload: dict[str, Any] = {
        "schema": "celltrack-alpha.cv57-exclusive-residual-error-taxonomy.v1",
        "status": "PASS",
        "scope": {
            "baseline": args.baseline_label,
            "video_count": 57,
            "single_thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "no_submission_or_remote_compute": True,
        },
        "definitions": {
            "detection": "official edge FN with at least one GT endpoint not distance-matched; node-count adjustment is assigned here",
            "ordinary_association": "all scorer-valid edge FP plus official edge FN whose matched endpoints remain in the same weak submitted-prediction component",
            "track_fragmentation": "official edge FN whose two GT endpoints are distance-matched but their submitted prediction nodes are in different weak components",
            "division": "official division FP/FN events, kept separate from every edge event",
            "exclusivity": "every scorer-valid edge FP is ordinary association; every official edge FN is exactly one of detection, ordinary association, or track fragmentation",
        },
        "inputs": {
            "control_score": {"path": str(args.control_score), "sha256": sha256_file(args.control_score)},
            "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
            "population_manifest": {"path": str(args.populations), "sha256": sha256_file(args.populations)},
            "archive": {"path": str(args.archive), "sha256": sha256_file(args.archive)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "git_head": git_head,
        },
        "scorer": scorer,
        "global": {
            "baseline_summary": global_baseline,
            "exclusive_edge_partition": edge_partition,
            "categories": category_results,
            "fully_fixed_theoretical_ceiling": global_ceiling,
            "fully_fixed_score_delta": total_delta,
            "dominant_category": {
                "name": dominant,
                "shapley_score_delta": global_shapley[dominant],
                "basis": "largest exact four-player Shapley attribution over the official nonlinear score arithmetic",
            },
            "shapley_score_attribution": global_shapley,
            "score_counterfactuals": {
                key: value for key, value in global_cache.items()
            },
        },
        "per_video_index": indexed,
        "validation": {
            "status": "PASS",
            "submission_graph_policy": (
                "competition_faithful_no_outdegree_cap"
                if competition_faithful
                else "strict_max_outdegree_2"
            ),
            "authoritative_replay": reference_validation,
            "edge_partition": "all 57 videos and global totals reproduce official FP/FN exactly",
            "division_separate": "division FP/FN never assigned to an edge category",
            "shapley_conservation": {
                "sum": sum(global_shapley.values()),
                "full_ceiling_delta": total_delta,
                "absolute_difference": abs(sum(global_shapley.values()) - total_delta),
            },
            "stable_visualizer_index": {
                "sort": "edge_fp + edge_fn descending, identity ascending",
                "top10": top_identity,
                "match": True,
            },
        },
        "commands": [
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python reports/g04/lanes/cv57_residual_error_taxonomy_20260804/build_taxonomy.py",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv313/bin/python -m pytest -q tests/test_public_temporalunet_baseline.py tests/test_track_visualizer.py",
        ],
    }
    return payload


def write_comparison(path: Path, current: Mapping[str, Any], reference_path: Path) -> None:
    """Write a compact, schema-checked APNCC/V32 comparison after V32 scoring."""

    reference = read_object(reference_path, "comparison taxonomy")
    if reference.get("schema") != current.get("schema") or reference.get("status") != "PASS":
        raise TaxonomyError("comparison taxonomy is not a passing compatible receipt")
    rows = []
    for category in CATEGORY_ORDER:
        left = reference["global"]["categories"][category]
        right = current["global"]["categories"][category]
        rows.append(
            f"| {category} | {int(left['score_event_count']):,} | {float(left['shapley_score_delta']):+.12f} | "
            f"{int(right['score_event_count']):,} | {float(right['shapley_score_delta']):+.12f} |"
        )
    reference_name = str(reference.get("scope", {}).get("baseline", "APNCC control"))
    current_name = str(current.get("scope", {}).get("baseline", "V32"))
    text = f"""# APNCC vs V32 CV57 residual taxonomy

| Category | {reference_name} events | {reference_name} Shapley | {current_name} events | {current_name} Shapley |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Dominant APNCC class: **{reference['global']['dominant_category']['name']}**.  Dominant V32 class: **{current['global']['dominant_category']['name']}**.

Both rows are exact exclusive official-scorer replays; score-attribution values are ceilings, not jointly deployable forecasts.
"""
    if path.exists():
        raise TaxonomyError(f"refusing to overwrite comparison: {path}")
    path.write_text(text, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--control-score", type=Path, default=DEFAULT_CONTROL / "APNCC_CV57_SCORE_WITH_FRAGMENTATION.json")
    value.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    value.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    value.add_argument("--scorer-parity", type=Path, default=DEFAULT_PARITY)
    value.add_argument("--output", type=Path, default=OUTPUT_DIR / "CV57_RESIDUAL_ERROR_TAXONOMY.json")
    value.add_argument("--report", type=Path, default=OUTPUT_DIR / "REPORT.md")
    value.add_argument("--baseline-label", default="APNCC frozen CV57 control")
    value.add_argument("--comparison-with", type=Path)
    value.add_argument("--comparison-output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = run(args)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.report.write_text(make_markdown(payload, baseline_label=args.baseline_label), encoding="utf-8")
        if (args.comparison_with is None) != (args.comparison_output is None):
            raise TaxonomyError("comparison input and output must be supplied together")
        if args.comparison_with is not None and args.comparison_output is not None:
            write_comparison(args.comparison_output, payload, args.comparison_with)
    except (TaxonomyError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"build_taxonomy: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "output": str(args.output), "report": str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
