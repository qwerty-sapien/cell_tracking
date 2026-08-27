#!/usr/bin/env python3
"""Classify every GT division AND every predicted fork across a full population.

This extends `diagnose_division_failure.py`'s classify_division taxonomy (imported
unmodified) to run over EVERY video in a population, not just ones with FN>0 in a
prior score artifact. That restriction is exactly why the old script cannot see forks
in zero-GT-division videos: R2's research found 7 of cv57's 11 division FPs live in
2 zero-GT videos holding 216/368 of the total emitted forks (2026-08-07,
dag_prompts/2026-08-07-score-ceiling-research/runs/R2_DIVISION_ALPHA_V2.md).

Nothing here reimplements the scorer. Every classification is read out of the frozen
`division_metrics.score_divisions` / `_pred_division_fork_sets` return values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl
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

from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from diagnose_division_failure import classify_division  # noqa: E402
from tracking_cellmot import division_metrics  # noqa: E402


class InstrumentationError(RuntimeError):
    """The instrumentation cannot preserve the official scorer/data contract."""


def _fork_channel(
    fork_id: int,
    *,
    tp_forks: set[int],
    fp_forks: set[int],
    evaluable_forks: set[int],
    cross_component_forks: set[int],
    malformed_forks: set[int],
) -> str:
    """Return the one channel that explains why the official scorer did (not) count
    `fork_id`, using only the scorer's own already-computed sets -- no new logic."""
    if fork_id in tp_forks:
        return "tp"
    if fork_id in fp_forks:
        if fork_id in malformed_forks:
            return "malformed_fp"
        if fork_id in cross_component_forks:
            return "cross_component_fp"
        if fork_id in evaluable_forks:
            return "evaluable_fp"
        return "considered_fp"
    return "invisible"


def classify_population(
    graphs: Sequence[tuple[str, Any, Any]],
    scale: tuple[float, ...],
    max_distance: float = 7.0,
) -> dict[str, Any]:
    """Classify every GT division and label every predicted fork, for every video.

    Parameters
    ----------
    graphs : sequence of (identity, pred_graph, gt_graph)
        Already-loaded tracksdata graphs -- this function does no file I/O, so it can
        be exercised directly in tests and reused against any loading path.
    """
    per_video: list[dict[str, Any]] = []
    forks: list[dict[str, Any]] = []
    divisions: list[dict[str, Any]] = []

    for identity, pred_graph, gt_graph in graphs:
        official_result = division_metrics.score_divisions(
            pred_graph, gt_graph, scale, max_distance
        )
        matched = division_metrics.match_divisions(pred_graph, gt_graph, scale, max_distance)
        gt_divisions = division_metrics.extract_divisions(gt_graph)
        if set(official_result.scores) != set(matched) or set(matched) != set(gt_divisions):
            raise InstrumentationError(
                f"{identity}: official division decomposition is inconsistent"
            )
        pred_dividing_nodes = {
            int(value)
            for value in pred_graph.node_ids()
            if pred_graph.out_degree(value) >= 2
        }
        evaluable_forks, cross_component_forks, malformed_forks = (
            division_metrics._pred_division_fork_sets(
                pred_graph, gt_graph, scale, max_distance
            )
        )
        node_id_key = division_metrics.td.DEFAULT_ATTR_KEYS.NODE_ID
        node_attrs = pred_graph.node_attrs(attr_keys=[node_id_key, "t"])
        t_by_node = dict(zip(node_attrs[node_id_key], node_attrs["t"]))

        for divider in sorted(gt_divisions):
            record = classify_division(
                matched[divider],
                gt_divisions[divider],
                int(divider),
                pred_dividing_nodes,
                cross_component_forks,
                malformed_forks,
                int(official_result.scores[divider]),
            )
            record["identity"] = identity
            divisions.append(record)

        for fork_id in sorted(pred_dividing_nodes):
            forks.append(
                {
                    "fork_id": int(fork_id),
                    "identity": identity,
                    "t": int(t_by_node[fork_id]),
                    "channel": _fork_channel(
                        fork_id,
                        tp_forks=official_result.tp_forks,
                        fp_forks=official_result.fp_forks,
                        evaluable_forks=evaluable_forks,
                        cross_component_forks=cross_component_forks,
                        malformed_forks=malformed_forks,
                    ),
                    "video_gt_division_count": len(gt_divisions),
                }
            )

        per_video.append(
            {
                "identity": identity,
                "gt_division_count": len(gt_divisions),
                "prediction_dividing_nodes": len(pred_dividing_nodes),
                "official_division_tp": sum(official_result.scores.values()),
                "official_division_fp": len(official_result.fp_forks),
            }
        )

    classes: dict[str, int] = {}
    for division in divisions:
        label = str(division["classification"])
        classes[label] = classes.get(label, 0) + 1

    return {
        "schema": "celltrack-alpha.division-fork-instrumentation.v1",
        "per_video": per_video,
        "forks": forks,
        "divisions": divisions,
        "summary": {
            "gt_divisions": sum(v["gt_division_count"] for v in per_video),
            "prediction_dividing_nodes": sum(v["prediction_dividing_nodes"] for v in per_video),
            "official_division_tp": sum(v["official_division_tp"] for v in per_video),
            "official_division_fp": sum(v["official_division_fp"] for v in per_video),
            "classification_counts": classes,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_population_from_submission(
    submission_path: Path,
    zip_path: Path,
    identities: Iterable[str],
    *,
    max_outdegree: int | None = None,
) -> list[tuple[str, Any, Any]]:
    """Load (identity, pred_graph, gt_graph) triples for `identities` from one
    real competition-format submission.csv plus the GT zip.

    Uses `celltrack_alpha.kaggle_calibration.read_submission` -- the same loader
    `scripts/public_temporalunet_baseline.py:score_submission` uses to produce every
    banked V32 score artifact (`FULL_PUBLIC_V32_*_CV57_SCORE.json`) -- not
    `scoreall.iter_aggregate_graphs`, which expects a different, split-prefixed
    `dataset` column that real competition submission.csv files do not carry.
    `max_outdegree=None` matches the competition-faithful V32 scoring path (no cap).
    """
    identities = list(identities)
    datasets = [identity.split(":", 1)[1] for identity in identities]
    graphs = read_submission(submission_path, max_outdegree=max_outdegree)
    missing = set(datasets) - set(graphs)
    if missing:
        raise InstrumentationError(
            f"{len(missing)} requested datasets absent from {submission_path}: "
            f"{sorted(missing)[:5]}..."
        )
    triples: list[tuple[str, Any, Any]] = []
    with ZipStore(zip_path, mode="r") as store:
        for identity, dataset in zip(identities, datasets):
            graph = graphs[dataset]
            prediction_nodes = [
                {"dataset": dataset, "node_id": node.node_id, "t": node.t,
                 "z": node.z, "y": node.y, "x": node.x}
                for node in graph.nodes.values()
            ]
            prediction_edges = [
                {"source_id": edge.source_id, "target_id": edge.target_id}
                for edge in graph.edges
            ]
            pred_graph = prediction_to_graph(prediction_nodes, prediction_edges)
            gt_arrays = read_geff_arrays(store, "train", dataset)
            gt_graph = geff_arrays_to_graph(gt_arrays)
            triples.append((identity, pred_graph, gt_graph))
            del prediction_nodes, prediction_edges, gt_arrays
    return triples


def propose_additional_daughters(
    pred_graph: Any,
    *,
    fork_id: int,
    scale: tuple[float, ...],
    max_distance: float = 7.0,
) -> set[int]:
    """DIV-WIDESAT (R2 SS3.2): candidate additional daughters for a fork.

    Restricted to nodes in the same next-frame as the fork's existing children,
    within `max_distance` (physical units) of the fork itself -- the scorer's own
    node-matching radius (division_metrics.evaluate_divisions default), so this
    introduces no new free parameter. Already-attached successors are excluded.
    """
    node_id_key = division_metrics.td.DEFAULT_ATTR_KEYS.NODE_ID
    existing = set(pred_graph.successors(fork_id))
    coord_keys = [node_id_key, "t", "z", "y", "x"]
    attrs = pred_graph.node_attrs(attr_keys=coord_keys)
    scale_arr = np.asarray(scale, dtype=np.float64)
    fork_row = attrs.filter(pl.col(node_id_key) == fork_id)
    if fork_row.height == 0:
        raise InstrumentationError(f"fork {fork_id} not present in graph")
    fork_t = int(fork_row["t"][0])
    fork_zyx = np.array(
        [fork_row["z"][0], fork_row["y"][0], fork_row["x"][0]], dtype=np.float64
    )
    target_frame = attrs.filter((pl.col("t") == fork_t + 1) & (pl.col(node_id_key) != fork_id))
    proposals: set[int] = set()
    for row in target_frame.iter_rows(named=True):
        candidate_id = int(row[node_id_key])
        if candidate_id in existing:
            continue
        candidate_zyx = np.array([row["z"], row["y"], row["x"]], dtype=np.float64)
        distance = float(np.linalg.norm((candidate_zyx - fork_zyx) * scale_arr))
        if distance <= max_distance:
            proposals.add(candidate_id)
    return proposals


def _widen_graph(pred_graph: Any, *, width: int, scale: tuple[float, ...], max_distance: float) -> Any:
    """Return a COPY of `pred_graph` with up to `width` daughters per out_degree==2
    fork, splicing in the nearest additional same-frame candidates (DIV-WIDESAT,
    R2 SS3.2). Forks not already at exactly out_degree==2 are left untouched, per spec
    ("every fork already in pred_div_nodes with out_degree == 2")."""
    widened = pred_graph.copy()
    node_id_key = division_metrics.td.DEFAULT_ATTR_KEYS.NODE_ID
    attrs = pred_graph.node_attrs(attr_keys=[node_id_key, "z", "y", "x"])
    coord_by_node = {
        int(row[node_id_key]): np.array([row["z"], row["y"], row["x"]], dtype=np.float64)
        for row in attrs.iter_rows(named=True)
    }
    scale_arr = np.asarray(scale, dtype=np.float64)
    fork_ids = [
        int(value)
        for value in pred_graph.node_ids()
        if pred_graph.out_degree(value) == 2
    ]
    for fork_id in fork_ids:
        extra = width - 2
        if extra <= 0:
            continue
        proposals = propose_additional_daughters(
            pred_graph, fork_id=fork_id, scale=scale, max_distance=max_distance
        )
        if not proposals:
            continue
        fork_zyx = coord_by_node[fork_id]
        ranked = sorted(
            proposals,
            key=lambda candidate: float(
                np.linalg.norm((coord_by_node[candidate] - fork_zyx) * scale_arr)
            ),
        )
        for candidate_id in ranked[: extra if extra < len(ranked) else len(ranked)]:
            widened.add_edge(fork_id, candidate_id, {})
    return widened


def sweep_widesat(
    graphs: Sequence[tuple[str, Any, Any]],
    scale: tuple[float, ...],
    widths: Sequence[int],
    max_distance: float = 7.0,
) -> dict[int, dict[str, Any]]:
    """DIV-WIDESAT sweep (R2 SS3.2): rescore the population's division term at
    each daughter-width cap in `widths`, using only the frozen `score_divisions`.
    `widths` must include 2 (the unmodified baseline) for comparison."""
    results: dict[int, dict[str, Any]] = {}
    for width in widths:
        tp = fp = fn = 0
        for identity, pred_graph, gt_graph in graphs:
            test_graph = pred_graph if width == 2 else _widen_graph(
                pred_graph, width=width, scale=scale, max_distance=max_distance
            )
            scored = division_metrics.score_divisions(test_graph, gt_graph, scale, max_distance)
            tp += sum(scored.scores.values())
            fn += len(scored.scores) - sum(scored.scores.values())
            fp += len(scored.fp_forks)
        denom = tp + fp + fn
        results[width] = {
            "division_tp": tp,
            "division_fp": fp,
            "division_fn": fn,
            "division_jaccard": tp / denom if denom else float("nan"),
        }
    return results


def generate_forks_for_forkless_parents(
    pred_graph: Any,
    gt_graph: Any,
    *,
    scale: tuple[float, ...],
    max_distance: float = 7.0,
) -> Any:
    """DIV-CHANGEPOINT's cheapest-first premise test (R2 SS3.1, kill line #3): for
    every GT division classified `local_parent_fork_missing`, propose a second
    child for each matched parent-side node using the SAME geometric proximity
    test as DIV-WIDESAT (`propose_additional_daughters`) -- a pure-geometry
    generator, not the full change-point statistic R2 proposes. R2's own kill line
    requires checking this floor first: if pure geometry recovers nothing, it
    matches the already-DEAD DIV-REALIZER-A precedent and no temporal-shape
    sophistication is likely to rescue it either.

    Returns a COPY of pred_graph with proposed edges spliced in for every
    forkless matched parent found; does not touch anything else.
    """
    widened = pred_graph.copy()
    pred_dividing_nodes = {
        int(value) for value in pred_graph.node_ids() if pred_graph.out_degree(value) >= 2
    }
    matched = division_metrics.match_divisions(pred_graph, gt_graph, scale, max_distance)
    gt_divisions = division_metrics.extract_divisions(gt_graph)
    node_id_key = division_metrics.td.DEFAULT_ATTR_KEYS.NODE_ID
    attrs = pred_graph.node_attrs(attr_keys=[node_id_key, "z", "y", "x"])
    coord_by_node = {
        int(row[node_id_key]): np.array([row["z"], row["y"], row["x"]], dtype=np.float64)
        for row in attrs.iter_rows(named=True)
    }
    scale_arr = np.asarray(scale, dtype=np.float64)
    for divider in sorted(gt_divisions):
        matched_nodes = division_metrics._matched_division_nodes(
            division_metrics._matched_node_attrs(matched[divider]), gt_divisions[divider], divider
        )
        if matched_nodes is None:
            continue
        parent_ids, _daughter_ids = matched_nodes
        local_nodes = parent_ids | {
            int(successor) for parent_id in parent_ids for successor in pred_graph.successors(parent_id)
        }
        if local_nodes & pred_dividing_nodes:
            continue  # a fork already exists locally; not this candidate's target case
        for parent_id in parent_ids:
            if parent_id not in coord_by_node:
                continue
            proposals = propose_additional_daughters(
                pred_graph, fork_id=parent_id, scale=scale, max_distance=max_distance
            )
            if not proposals:
                continue
            parent_zyx = coord_by_node[parent_id]
            nearest = min(
                proposals,
                key=lambda candidate: float(
                    np.linalg.norm((coord_by_node[candidate] - parent_zyx) * scale_arr)
                ),
            )
            widened.add_edge(parent_id, nearest, {})
    return widened


def load_scale(zip_path: Path, dataset: str) -> tuple[float, ...]:
    with ZipStore(zip_path, mode="r") as store:
        group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
        return tuple(float(value) for value in parse_scale(dict(group.attrs)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True, help="JSON list of identities")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise InstrumentationError(f"refusing to overwrite output: {args.output}")
    identities = json.loads(args.identities.read_text(encoding="utf-8"))
    if not isinstance(identities, list) or not identities:
        raise InstrumentationError("--identities must be a non-empty JSON list")

    triples = load_population_from_submission(args.submission, args.zip, identities)
    # Scale is per-video in principle but constant in practice for this competition;
    # reload it once more here for provenance rather than assuming.
    scale = load_scale(args.zip, identities[0].split(":", 1)[1])

    report = classify_population(triples, scale=scale)
    report["source_submission"] = {
        "path": str(args.submission),
        "sha256": _sha256(args.submission),
    }
    report["source_scorer"] = {
        "division_metrics_sha256": _sha256(
            ROOT / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(report, allow_nan=False, sort_keys=True).encode("ascii") + b"\n"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
