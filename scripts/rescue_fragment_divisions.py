#!/usr/bin/env python3
"""Apply one division-rescue policy derived exclusively from fold-fit GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    read_submission,
    rescue_divisions,
    write_submission,
)


class DivisionRescueError(RuntimeError):
    """A fit-only rescue policy or fragment set is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DivisionRescueError(f"JSON root is not an object: {path}")
    return value


def derive_policy(geometry: Mapping[str, Any]) -> dict[str, int | float | str]:
    scope = geometry.get("scope")
    summary = geometry.get("summary")
    if geometry.get("schema") != "celltrack-alpha.fit-division-geometry.v2":
        raise DivisionRescueError("division geometry must use the v2 schema")
    if not isinstance(scope, Mapping) or not isinstance(summary, Mapping):
        raise DivisionRescueError("division geometry is missing scope or summary")
    if any(bool(scope.get(key)) for key in ("held_out_used", "audit_used", "test_used")):
        raise DivisionRescueError("policy source is not fit-only")

    def statistic(name: str, quantile: str) -> float:
        row = summary.get(name)
        value = row.get(quantile) if isinstance(row, Mapping) else None
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DivisionRescueError(f"missing statistic {name}.{quantile}")
        return float(value)

    # A single predeclared policy: p95 upper envelopes, p05 lower envelope,
    # fit median for ranking, and the observed p95 concurrency ceiling.
    return {
        "selection_rule": "fit-p95-envelope-p05-lower-median-rank",
        "max_parent_child_um": statistic("maximum_parent_child_um", "p95"),
        "min_daughter_separation_um": statistic("sister_um", "p05"),
        "max_daughter_separation_um": statistic("sister_um", "p95"),
        "max_symmetry_delta_um": statistic("symmetry_delta_um", "p95"),
        "max_midpoint_residual_um": statistic("midpoint_residual_um", "p95"),
        "max_forks_per_t": max(
            1, int(math.ceil(statistic("divisions_per_occupied_video_time", "p95")))
        ),
        "target_daughter_separation_um": statistic("sister_um", "p50"),
    }


def rescue_fragments(
    source: Path,
    output: Path,
    *,
    split_path: Path,
    subset: str,
    geometry_path: Path,
) -> dict[str, Any]:
    if output.exists():
        raise DivisionRescueError(f"refusing to overwrite output: {output}")
    geometry = _load(geometry_path)
    policy = derive_policy(geometry)
    split = _load(split_path)
    selection = scoreall.select_video_ids(split, subset)
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(source)), selection.selected_video_ids
    )
    output.mkdir(parents=True, exist_ok=False)
    stats: list[dict[str, Any]] = []
    rescue_args = {key: value for key, value in policy.items() if key != "selection_rule"}
    for identity, path in manifest.ordered:
        graphs = read_submission(path)
        if len(graphs) != 1:
            raise DivisionRescueError(f"fragment is not single-dataset: {path}")
        dataset, graph = next(iter(graphs.items()))
        transformed, row = rescue_divisions(graph, **rescue_args)
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
        "schema": "celltrack-alpha.fit-derived-division-rescue.v1",
        "subset": subset,
        "video_count": len(stats),
        "policy": policy,
        "policy_source": {
            "path": str(geometry_path),
            "sha256": _sha256(geometry_path),
            "fit_only": True,
        },
        "source_fragment_set": str(source),
        "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        "totals": {
            key: sum(int(row[key]) for row in stats)
            for key in ("before_edges", "after_edges", "added_division_edges", "candidate_motifs")
        },
        "per_video": stats,
    }
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
    parser.add_argument(
        "--geometry",
        type=Path,
        default=ROOT / "reports/g02/fit_division_geometry_fold0_v2.json",
    )
    args = parser.parse_args(argv)
    report = rescue_fragments(
        args.source,
        args.output,
        split_path=args.split,
        subset=args.subset,
        geometry_path=args.geometry,
    )
    print(json.dumps({"policy": report["policy"], "totals": report["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
