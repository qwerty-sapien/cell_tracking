#!/usr/bin/env python3
"""Build a diagnostic association oracle while freezing predicted nodes exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

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
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    read_submission,
    write_submission,
)
from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


class AssociationOracleError(RuntimeError):
    """The diagnostic could not preserve fixed-node or official matching scope."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AssociationOracleError(f"JSON root is not an object: {path}")
    return value


def _node_rows(graph: SubmissionGraph) -> list[dict[str, int]]:
    return [
        {
            "node_id": node.node_id,
            "t": node.t,
            "z": node.z,
            "y": node.y,
            "x": node.x,
        }
        for node in graph.nodes.values()
    ]


def oracle_graph(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
) -> tuple[SubmissionGraph, dict[str, int]]:
    rows = _node_rows(graph)
    pred = prediction_to_graph(rows, [])
    truth = geff_arrays_to_graph(dict(gt))
    ordered_original_ids = [
        int(row["node_id"]) for row in sorted(rows, key=lambda row: (row["t"], row["node_id"]))
    ]
    pred_internal = [int(value) for value in pred.node_attrs()["node_id"].to_list()]
    if pred_internal != list(range(len(ordered_original_ids))):
        raise AssociationOracleError("prediction graph internal node order is not canonical")

    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options

    matching = DistanceMatching(max_distance=7.0, scale=scale_zyx)
    previous_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred.match(truth, matching=matching)
    finally:
        set_options(show_progress=previous_progress)

    matched = pred.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    gt_to_original: dict[int, int] = {}
    for pred_id, gt_id in matched.iter_rows():
        if gt_id is None or int(gt_id) == -1:
            continue
        internal = int(pred_id)
        if internal < 0 or internal >= len(ordered_original_ids) or int(gt_id) in gt_to_original:
            raise AssociationOracleError("official matching is not one-to-one")
        gt_to_original[int(gt_id)] = ordered_original_ids[internal]

    additions: list[SubmissionEdge] = []
    for source, target in truth.edge_attrs().select("source_id", "target_id").iter_rows():
        pred_source = gt_to_original.get(int(source))
        pred_target = gt_to_original.get(int(target))
        if pred_source is not None and pred_target is not None:
            additions.append(SubmissionEdge(pred_source, pred_target))
    output = SubmissionGraph(graph.dataset, dict(graph.nodes), additions)
    return output, {
        "predicted_nodes": len(graph.nodes),
        "matched_nodes": len(gt_to_original),
        "gt_nodes": truth.num_nodes(),
        "gt_edges": truth.num_edges(),
        "recoverable_gt_edges": len(additions),
    }


def build_oracle(
    zip_path: Path,
    split_path: Path,
    source: Path,
    output: Path,
    *,
    subset: str,
) -> dict[str, Any]:
    if output.exists():
        raise AssociationOracleError(f"refusing to overwrite output: {output}")
    split = _load(split_path)
    selection = scoreall.select_video_ids(split, subset)
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(source)), selection.selected_video_ids
    )
    output.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    with ZipStore(zip_path, mode="r") as store:
        for identity, path in manifest.ordered:
            graphs = read_submission(path)
            if len(graphs) != 1:
                raise AssociationOracleError(f"fragment is not single-dataset: {path}")
            dataset, graph = next(iter(graphs.items()))
            if identity != f"train:{dataset}":
                raise AssociationOracleError("fragment identity and dataset disagree")
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            transformed, row = oracle_graph(graph, gt, scale)
            destination = output / path.name
            write_submission({dataset: transformed}, destination)
            samples.append(
                {
                    "identity": identity,
                    "source_sha256": _sha256(path),
                    "output_sha256": _sha256(destination),
                    **row,
                }
            )
    report = {
        "schema": "celltrack-alpha.fixed-node-association-oracle.v1",
        "purpose": "diagnostic-upper-bound-not-a-candidate",
        "subset": subset,
        "video_count": len(samples),
        "fixed": {
            "prediction_nodes": True,
            "prediction_coordinates": True,
            "node_count_penalty": True,
        },
        "oracle": {
            "official_distance_matching": True,
            "gt_supported_edges_only": True,
            "held_out_labels_used": True,
            "eligible_for_candidate_selection": False,
            "eligible_for_submission": False,
        },
        "source_fragment_set": str(source),
        "inputs": {
            "zip": {"path": str(zip_path), "bytes": zip_path.stat().st_size},
            "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        },
        "totals": {
            key: sum(int(row[key]) for row in samples)
            for key in (
                "predicted_nodes",
                "matched_nodes",
                "gt_nodes",
                "gt_edges",
                "recoverable_gt_edges",
            )
        },
        "samples": samples,
    }
    (output / "report.json").write_bytes(
        json.dumps(report, allow_nan=False, sort_keys=True).encode("ascii") + b"\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=("fold0", "fold1", "fold2", "audit"), default="fold0")
    args = parser.parse_args(argv)
    report = build_oracle(
        args.zip, args.split, args.source, args.output, subset=args.subset
    )
    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
