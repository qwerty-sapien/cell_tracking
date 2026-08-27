#!/usr/bin/env python3
"""Verify one selector class in an isolated, guarded process."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import independent_verify as review  # noqa: E402
import selector_requirement_class as target  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise review.ReviewError(f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-class",
        required=True,
        choices=("delete", "occupied_swap", "restore_topology"),
    )
    parser.add_argument("--class-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1":
        raise SystemExit("binding local zero-swap guard is required")
    if args.output.exists():
        raise SystemExit("refusing to overwrite class review output")

    target.base._verify_pins()
    reported = load(args.class_result.resolve())
    if (
        reported.get("status") != "COMPLETE"
        or reported.get("action_class") != args.action_class
    ):
        raise review.ReviewError("class result is incomplete or mismatched")
    population, _baseline = target.load_population(args.action_class)
    curve = reported["oracle"]["curve"]
    best = max(
        curve,
        key=lambda row: float(row["graph_delta"]["adjusted_edge_jaccard"]),
    )
    order = target.base.oracle_order(population.delta_s)
    prefix = int(best["candidate_prefix_N"])
    if args.action_class == "occupied_swap":
        selected = review.independent_greedy_swap(population, order, prefix)
        candidate = review.direct_edge_score(target.base, population, selected)
        baseline = target.base.official_summary(list(population.baseline_rows))
    elif args.action_class == "delete":
        selected = np.asarray(order[:prefix], dtype=np.int64)
        candidate = review.direct_edge_score(target.base, population, selected)
        baseline = target.base.official_summary(list(population.baseline_rows))
    else:
        selected = np.asarray(order[:prefix], dtype=np.int64)
        candidate = review.direct_restore_score(target.base, population, selected)
        baseline = target.base.official_summary(list(population.baseline_rows))
    observed = {
        "adjusted_edge_jaccard": float(candidate["adj_edge_jaccard"])
        - float(baseline["adj_edge_jaccard"]),
        "raw_edge_jaccard": float(candidate["edge_jaccard"])
        - float(baseline["edge_jaccard"]),
        "node_recall": float(candidate["node_recall"])
        - float(baseline["node_recall"]),
    }
    for key, value in observed.items():
        if not math.isclose(
            value, float(best["graph_delta"][key]), abs_tol=1e-12
        ):
            raise review.ReviewError(
                f"{args.action_class} {key}: {value} != "
                f"{best['graph_delta'][key]}"
            )
    if len(selected) != int(best["applied_support"]):
        raise review.ReviewError("applied support drifted")
    payload = {
        "schema": "celltrack-alpha.selector-requirement-class-review.v1",
        "status": "PASS",
        "action_class": args.action_class,
        "method": (
            "Fresh isolated population load and direct frozen official scoring "
            "of the oracle-best point; occupied swap uses an independent greedy "
            "ownership implementation."
        ),
        "candidate_prefix_N": prefix,
        "applied_support": len(selected),
        "direct_official_delta": observed,
        "reported_delta": best["graph_delta"],
        "findings": [],
    }
    args.output.mkdir(parents=True)
    review.write(args.output / "CLASS_REVIEW.json", payload)
    print(json.dumps({"status": "PASS", "action_class": args.action_class}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
