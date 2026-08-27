#!/usr/bin/env python3
"""Score a submission-style CSV on the locked clips from a CV audit report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_general_route_cv import crop_geff_arrays  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission, sha256_file  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, default=None)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def main() -> int:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    zip_path = args.zip or Path(audit["zip"])
    graphs = read_submission(args.submission)
    audit_rows = {str(row["dataset"]): row for row in audit.get("rows", [])}
    expected = set(audit_rows)
    actual = set(graphs)
    if actual != expected:
        raise ValueError(
            f"submission/audit dataset mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    rows: list[dict[str, Any]] = []
    store = ZipStore(zip_path, mode="r")
    try:
        for dataset in sorted(expected):
            audit_row = audit_rows[dataset]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            gt = read_geff_arrays(store, "train", dataset)
            start = int(audit_row["clip_start"])
            end = int(audit_row["clip_end"])
            gt_clip = crop_geff_arrays(gt, start, end, int(array.shape[0]))
            graph = graphs[dataset]
            nodes = [
                {
                    "dataset": dataset,
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
            metric = official_metric_row(nodes, edges, gt_clip, parse_scale(dict(group.attrs)))
            rows.append(
                {
                    "dataset": dataset,
                    "clip_start": start,
                    "clip_end": end,
                    "num_pred_nodes": len(nodes),
                    "num_pred_edges": len(edges),
                    **metric,
                }
            )
    finally:
        store.close()

    summary = official_summary(rows)
    payload = {
        "schema_version": 1,
        "kind": "locked_cv_submission_score",
        "submission": str(args.submission),
        "submission_sha256": sha256_file(args.submission),
        "audit": str(args.audit),
        "audit_sha256": sha256_file(args.audit),
        "zip": str(zip_path),
        "summary": summary,
        "score": finite(summary.get("score")),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
