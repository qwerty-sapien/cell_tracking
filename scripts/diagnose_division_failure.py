#!/usr/bin/env python3
"""Classify why official GT divisions are missed by persisted predictions.

For every GT division, this reuses the pinned official distance matching and
separates four progressively stronger conditions: required stages detected,
those stages connected in one prediction component, a prediction fork present
in that component, and official recovery.  The diagnostic never changes or
rescales a prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import polars as pl
import tracksdata as td
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
from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot import division_metrics  # noqa: E402


DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_FRAGMENTS = ROOT / "reports/g02/t04_v45f4_frozen_sparse_fold0_local_20260719/frags"
DEFAULT_SCORES = ROOT / "reports/g02/t04_v45f4_frozen_sparse_fold0_local_20260719/scores.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/division_failure_fold0.json"


class DivisionDiagnosticError(RuntimeError):
    """The diagnostic cannot preserve the official scorer/data contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DivisionDiagnosticError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise DivisionDiagnosticError(f"JSON root is not an object: {path}")
    return value


def _descendants(graph: Any, seed: int) -> set[int]:
    result = {int(seed)}
    stack = [int(seed)]
    while stack:
        for successor in graph.successors(stack.pop()):
            value = int(successor)
            if value not in result:
                result.add(value)
                stack.append(value)
    return result


def stage_coverage(matched_attrs: pl.DataFrame, gt_div: Any, divider: int) -> dict[str, Any]:
    time_counts = gt_div.node_attrs(attr_keys=["t"]).group_by("t").agg(pl.len().alias("n"))
    one_node_times = set(time_counts.filter(pl.col("n") == 1)["t"].to_list())
    children = [int(value) for value in gt_div.successors(divider)]
    lineages = [_descendants(gt_div, child) for child in children]
    if matched_attrs.is_empty():
        matched_times: set[int] = set()
        matched_gt_ids: set[int] = set()
    else:
        matched_times = {int(value) for value in matched_attrs["t"].to_list()}
        matched_gt_ids = {
            int(value)
            for value in matched_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()
            if value is not None and int(value) != -1
        }
    covered = sum(bool(lineage & matched_gt_ids) for lineage in lineages)
    has_one = bool(one_node_times & matched_times)
    return {
        "one_node_stage_matched": has_one,
        "daughter_lineages": len(lineages),
        "daughter_lineages_matched": covered,
        "stage_covered": bool(has_one and covered >= 2),
        "matched_prediction_nodes": int(matched_attrs.height),
    }


def classify_division(
    matched_pred: Any,
    gt_div: Any,
    divider: int,
    pred_dividing_nodes: set[int],
    official_score: int,
) -> dict[str, Any]:
    attrs = division_metrics._matched_node_attrs(matched_pred)
    overall = stage_coverage(attrs, gt_div, divider)
    node_ids = [int(value) for value in attrs["node_id"].to_list()]
    components = division_metrics._weakly_connected_components(matched_pred, node_ids)
    spanning: list[dict[str, Any]] = []
    fork_candidates: set[int] = set()
    for matched_subset, visited in components:
        component_attrs = attrs.filter(pl.col("node_id").is_in(list(matched_subset)))
        coverage = stage_coverage(component_attrs, gt_div, divider)
        if coverage["stage_covered"]:
            forks = {int(value) for value in visited} & pred_dividing_nodes
            fork_candidates |= forks
            spanning.append(
                {
                    **coverage,
                    "visited_prediction_nodes": len(visited),
                    "fork_candidate_ids": sorted(forks),
                }
            )
    if official_score:
        classification = "officially_recovered"
    elif not overall["stage_covered"]:
        classification = "detection_or_localization_stage_missing"
    elif not spanning:
        classification = "association_components_disconnected"
    elif not fork_candidates:
        classification = "connected_but_prediction_fork_missing"
    else:
        classification = "fork_candidate_lost_in_global_pairing"
    return {
        "gt_divider_id": int(divider),
        "official_score": int(official_score),
        "classification": classification,
        "overall_stage_coverage": overall,
        "spanning_component_count": len(spanning),
        "fork_candidate_ids": sorted(fork_candidates),
        "spanning_components": spanning,
    }


def diagnose(
    zip_path: Path,
    split_path: Path,
    fragments: Path,
    scores_path: Path,
) -> dict[str, Any]:
    scores = _load(scores_path)
    scored_archive = scores.get("archive")
    archive_sha256 = (
        scored_archive.get("sha256") if isinstance(scored_archive, Mapping) else None
    )
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
    ):
        raise DivisionDiagnosticError("score artifact does not bind the competition archive")
    samples = scores.get("samples")
    if not isinstance(samples, list):
        raise DivisionDiagnosticError("score artifact has no per-sample rows")
    division_ids = tuple(
        sorted(
            str(row["identity"])
            for row in samples
            if isinstance(row, Mapping) and int(float(row.get("division_fn", 0))) > 0
        )
    )
    split = _load(split_path)
    selection = scoreall.select_video_ids(split, "fold0")
    if not set(division_ids).issubset(selection.selected_video_ids):
        raise DivisionDiagnosticError("division-bearing scores escaped fold-0 heldout")
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(fragments)), selection.selected_video_ids
    )
    paths = dict(manifest.ordered)
    results: list[dict[str, Any]] = []
    with ZipStore(zip_path, mode="r") as store:
        for identity in division_ids:
            _, dataset, _ = scoreall.parse_identity(identity)
            submission = scoreall._fragment_graph(paths[identity], identity)
            prediction_nodes, prediction_edges = scoreall._prediction_rows(dataset, submission)
            pred_graph = prediction_to_graph(prediction_nodes, prediction_edges)
            gt_arrays = read_geff_arrays(store, "train", dataset)
            gt_graph = geff_arrays_to_graph(gt_arrays)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            official_scores = division_metrics.score_divisions(pred_graph, gt_graph, scale, 7.0)
            matched = division_metrics.match_divisions(pred_graph, gt_graph, scale, 7.0)
            gt_divisions = division_metrics.extract_divisions(gt_graph)
            if set(official_scores) != set(matched) or set(matched) != set(gt_divisions):
                raise DivisionDiagnosticError("official division decomposition is inconsistent")
            pred_dividing_nodes = {int(value) for value in pred_graph.dividing_nodes()}
            divisions = [
                classify_division(
                    matched[divider],
                    gt_divisions[divider],
                    int(divider),
                    pred_dividing_nodes,
                    int(official_scores[divider]),
                )
                for divider in sorted(gt_divisions)
            ]
            results.append(
                {
                    "identity": identity,
                    "dataset": dataset,
                    "scale_zyx": list(scale),
                    "prediction_nodes": pred_graph.num_nodes(),
                    "prediction_edges": pred_graph.num_edges(),
                    "prediction_dividing_nodes": len(pred_dividing_nodes),
                    "gt_divisions": len(divisions),
                    "official_division_tp": sum(item["official_score"] for item in divisions),
                    "divisions": divisions,
                }
            )
            del submission, prediction_nodes, prediction_edges, pred_graph, gt_arrays, gt_graph, group
    all_divisions = [division for result in results for division in result["divisions"]]
    classes: dict[str, int] = {}
    for division in all_divisions:
        label = str(division["classification"])
        classes[label] = classes.get(label, 0) + 1
    gt_count = sum(int(result["gt_divisions"]) for result in results)
    if gt_count != 42:
        raise DivisionDiagnosticError(f"expected 42 fold-0 GT divisions, observed {gt_count}")
    return {
        "schema": "celltrack-alpha.division-failure-diagnostic.v1",
        "scope": {
            "subset": "fold0",
            "full_heldout_video_count": len(selection.selected_video_ids),
            "division_bearing_video_count": len(division_ids),
            "division_bearing_video_ids": list(division_ids),
            "fit_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "summary": {
            "gt_divisions": gt_count,
            "prediction_dividing_nodes": sum(
                int(result["prediction_dividing_nodes"]) for result in results
            ),
            "official_division_tp": sum(
                int(result["official_division_tp"]) for result in results
            ),
            "classification_counts": classes,
        },
        "inputs": {
            "zip": {
                "path": str(zip_path),
                "bytes": zip_path.stat().st_size,
                "sha256": archive_sha256,
                "identity_source": "official-score-input-provenance",
            },
            "split": {"path": str(split_path), "sha256": _sha256(split_path)},
            "scores": {"path": str(scores_path), "sha256": _sha256(scores_path)},
            "fragments": str(fragments),
            "official_division_metrics_sha256": _sha256(
                ROOT
                / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py"
            ),
        },
        "samples": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise DivisionDiagnosticError(f"refusing to overwrite output: {args.output}")
    report = diagnose(args.zip, args.split, args.fragments, args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(report, allow_nan=False, sort_keys=True).encode("ascii") + b"\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
