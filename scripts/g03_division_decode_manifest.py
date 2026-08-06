#!/usr/bin/env python3
"""Build the finite, outcome-blind G-03 division-decode configuration manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "celltrack-alpha.g03-division-decode-config-manifest.v1"
FORK_THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
MAX_PRIMARY_SECONDARY_GAPS = (0.05, 0.10, 0.20, 1.00)
LATE_WINDOWS = (0, 1, 2)
DAUGHTER_PERSISTENCE = (0, 1, 2, 3)
GRANDCHILD_PERSISTENCE = (0, 1, 2)
DUPLICATE_TEMPORAL_WINDOWS = (0, 1, 2)
PER_VIDEO_CAPS: tuple[int | None, ...] = (1, 2, 3, 5, 8, None)


@dataclass(frozen=True)
class DivisionDecodeConfig:
    name: str
    divisions: bool
    fork_threshold: float | None
    max_primary_secondary_gap: float | None
    late_window: int
    daughter_persistence: int
    grandchild_persistence: int
    duplicate_temporal_window: int
    per_video_cap: int | None


def _pct(value: float) -> str:
    return f"{round(value * 100):03d}"


def build_configurations() -> tuple[DivisionDecodeConfig, ...]:
    rows: list[DivisionDecodeConfig] = [
        DivisionDecodeConfig(
            name="DIVISIONS_OFF",
            divisions=False,
            fork_threshold=None,
            max_primary_secondary_gap=None,
            late_window=0,
            daughter_persistence=0,
            grandchild_persistence=0,
            duplicate_temporal_window=0,
            per_video_cap=0,
        )
    ]
    for threshold, gap, late, daughter, grandchild, duplicate, cap in itertools.product(
        FORK_THRESHOLDS,
        MAX_PRIMARY_SECONDARY_GAPS,
        LATE_WINDOWS,
        DAUGHTER_PERSISTENCE,
        GRANDCHILD_PERSISTENCE,
        DUPLICATE_TEMPORAL_WINDOWS,
        PER_VIDEO_CAPS,
    ):
        cap_name = "all" if cap is None else f"{cap:02d}"
        name = (
            f"t{_pct(threshold)}_g{_pct(gap)}_l{late}_"
            f"dp{daughter}_gp{grandchild}_n{duplicate}_c{cap_name}"
        )
        rows.append(
            DivisionDecodeConfig(
                name=name,
                divisions=True,
                fork_threshold=threshold,
                max_primary_secondary_gap=gap,
                late_window=late,
                daughter_persistence=daughter,
                grandchild_persistence=grandchild,
                duplicate_temporal_window=duplicate,
                per_video_cap=cap,
            )
        )
    expected = 18_145
    if len(rows) != expected or len({row.name for row in rows}) != expected:
        raise RuntimeError("division-decode lattice count or identity drift")
    return tuple(rows)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_manifest() -> dict[str, Any]:
    configurations = [asdict(row) for row in build_configurations()]
    axes = {
        "fork_threshold": list(FORK_THRESHOLDS),
        "max_primary_secondary_gap": list(MAX_PRIMARY_SECONDARY_GAPS),
        "late_window": list(LATE_WINDOWS),
        "daughter_persistence": list(DAUGHTER_PERSISTENCE),
        "grandchild_persistence": list(GRANDCHILD_PERSISTENCE),
        "duplicate_temporal_window": list(DUPLICATE_TEMPORAL_WINDOWS),
        "per_video_cap": list(PER_VIDEO_CAPS),
    }
    manifest = {
        "schema": SCHEMA,
        "status": "LOCKED_BEFORE_OUTCOMES",
        "active_configuration_count": len(configurations) - 1,
        "configuration_count_including_null": len(configurations),
        "axes": axes,
        "invariants": {
            "family_shared_configuration": True,
            "exact_equivalence_suppression_before_node_ownership": True,
            "duplicate_survivors_required": 0,
            "adaptive_search_forbidden": True,
            "sealed_confirmation_access_before_nested_oof": False,
        },
        "configurations": configurations,
    }
    manifest["configuration_sha256"] = canonical_sha256(configurations)
    return manifest


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    atomic_write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "count": manifest["configuration_count_including_null"],
                "configuration_sha256": manifest["configuration_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
