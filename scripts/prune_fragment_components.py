#!/usr/bin/env python3
"""Remove topology-only short components from an exact fragment set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    prune_short_components,
    read_submission,
    write_submission,
)


class FragmentPruneError(RuntimeError):
    """Fragment pruning would violate exact coverage or topology."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prune_fragments(
    source: Path,
    output: Path,
    *,
    split_path: Path,
    subset: str,
    max_component_edges: int,
) -> dict[str, Any]:
    if output.exists():
        raise FragmentPruneError(f"refusing to overwrite output: {output}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    selection = scoreall.select_video_ids(split, subset)
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(source)), selection.selected_video_ids
    )
    output.mkdir(parents=True, exist_ok=False)
    stats: list[dict[str, Any]] = []
    for identity, path in manifest.ordered:
        graphs = read_submission(path)
        if len(graphs) != 1:
            raise FragmentPruneError(f"fragment is not single-dataset: {path}")
        dataset, graph = next(iter(graphs.items()))
        transformed, row = prune_short_components(
            graph, max_component_edges=max_component_edges
        )
        destination = output / path.name
        write_submission({dataset: transformed}, destination)
        stats.append(
            {
                "identity": identity,
                "source": str(path),
                "source_sha256": _sha256(path),
                "output": str(destination),
                "output_sha256": _sha256(destination),
                **row,
            }
        )
    report = {
        "schema": "celltrack-alpha.fragment-component-prune.v1",
        "subset": subset,
        "video_count": len(stats),
        "max_component_edges": max_component_edges,
        "policy": (
            "remove-isolated-nodes-only"
            if max_component_edges == 0
            else f"remove-components-with-at-most-{max_component_edges}-edges"
        ),
        "source_fragment_set": str(source),
        "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        "totals": {
            key: sum(int(row[key]) for row in stats)
            for key in (
                "before_nodes",
                "after_nodes",
                "removed_nodes",
                "before_edges",
                "after_edges",
                "removed_edges",
                "removed_components",
            )
        },
        "per_video": stats,
    }
    if max_component_edges == 0 and report["totals"]["removed_edges"] != 0:
        raise FragmentPruneError("isolated-node pruning unexpectedly removed an edge")
    (output / "report.json").write_bytes(
        json.dumps(report, allow_nan=False, sort_keys=True).encode("ascii") + b"\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", type=Path, default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
    )
    parser.add_argument("--subset", choices=("fold0", "fold1", "fold2", "audit"), default="fold0")
    parser.add_argument("--max-component-edges", type=int, default=0)
    args = parser.parse_args(argv)
    report = prune_fragments(
        args.source,
        args.output,
        split_path=args.split,
        subset=args.subset,
        max_component_edges=args.max_component_edges,
    )
    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
