#!/usr/bin/env python3
"""Run decoupled H1 zero-wrfeat held-fit graph gates, 44b6 first."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import evaluate_trackastra_linear_oof_graph_gate as engine  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
)


DEFAULT_CONTRACT = (
    ROOT / "reports/g02/screening/h1_zero_wrfeat_graph_gate_contract.json"
)
DEFAULT_FEATURE_ROOT = (
    ROOT / "reports/g02/trackastra_h1_feature_cache"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/screening/h1_zero_wrfeat_graph_gate"
FAMILY_ORDER = ("44b6", "6bba")


class H1ZeroWrfeatGateError(RuntimeError):
    """The fit105-only, 44b6-first H1 gate failed closed."""


def family_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    decision = report.get("decision")
    if not isinstance(decision, Mapping):
        raise H1ZeroWrfeatGateError("family report has no graph decision")
    try:
        result = {
            "adjusted_edge_jaccard_delta": float(
                decision["adjusted_edge_jaccard_delta"]
            ),
            "raw_edge_jaccard_delta": float(
                decision["raw_edge_jaccard_delta"]
            ),
            "node_recall_delta": float(decision["node_recall_delta"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise H1ZeroWrfeatGateError(
            "family report graph metrics are malformed"
        ) from exc
    return result


def classify_family(
    family: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = family_metrics(report)
    adjusted = metrics["adjusted_edge_jaccard_delta"]
    passed = (
        adjusted >= 0.005
        and metrics["raw_edge_jaccard_delta"] >= 0.0
        and metrics["node_recall_delta"] >= 0.0
        and report.get("decision", {}).get("observation_distinct") is True
    )
    if family == "44b6" and adjusted <= 0.0:
        status = "KILL"
        reason = "44b6 adjusted graph delta is at or below zero"
    elif passed:
        status = "PASS"
        reason = "all preregistered family graph guards passed"
    else:
        status = "PARK"
        reason = "family graph result did not clear every promotion guard"
    return {
        "family": family,
        "status": status,
        "reason": reason,
        **metrics,
        "observation_distinct": bool(
            report.get("decision", {}).get("observation_distinct")
        ),
    }


def execute_ordered(
    evaluator: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate 44b6 first and never call 6bba after a 44b6 failure."""
    records = []
    first = classify_family("44b6", evaluator("44b6"))
    records.append(first)
    if first["status"] != "PASS":
        return {
            "status": first["status"],
            "families": records,
            "six_bba_executed": False,
            "cross_validation_authorized": False,
            "next": (
                "H3-MASK-SOURCE eligible as a separate unstarted node"
                if first["status"] == "KILL"
                else "PARK H1; no downstream family or cross-validation run"
            ),
        }
    second = classify_family("6bba", evaluator("6bba"))
    records.append(second)
    if second["status"] != "PASS":
        return {
            "status": "PARK",
            "families": records,
            "six_bba_executed": True,
            "cross_validation_authorized": False,
            "next": "PARK H1; no cross-validation run",
        }
    return {
        "status": "PASS_HELD_FIT_BOTH_FAMILIES",
        "families": records,
        "six_bba_executed": True,
        "cross_validation_authorized": True,
        "next": "full cross-validation is a separate gated node",
    }


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument(
        "--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--raw-archive", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument(
        "--support-root", type=Path, default=DEFAULT_SUPPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise H1ZeroWrfeatGateError(
            "runner must be launched by the zero-swap resource guard"
        )
    for name in (
        "contract",
        "feature_root",
        "cache",
        "raw_archive",
        "edge_artifact",
        "edge_auxiliary",
        "support_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise H1ZeroWrfeatGateError(
            f"refusing to overwrite output: {args.output}"
        )
    contract = engine.load_h1_contract(
        args.contract, args.contract_sha256
    )
    if contract["execution"]["family_order"] != list(FAMILY_ORDER):
        raise H1ZeroWrfeatGateError("family order drifted")
    feature_report = (
        ROOT / contract["candidate"]["feature_cache_report"]
    )
    if not feature_report.is_file():
        raise H1ZeroWrfeatGateError(
            "authenticated zero-wrfeat feature report is not present"
        )
    args.output.mkdir(parents=True, exist_ok=False)

    def evaluate(family: str) -> Mapping[str, Any]:
        family_args = argparse.Namespace(
            contract=args.contract,
            contract_sha256=args.contract_sha256,
            policy=ROOT / "reports/g02/compute_governance_v1.json",
            feature_root=args.feature_root,
            cache=args.cache,
            zip=args.raw_archive,
            edge_artifact=args.edge_artifact,
            edge_auxiliary=args.edge_auxiliary,
            support_root=args.support_root,
            output=args.output / family,
            resume=False,
            h1_family=family,
        )
        return engine.run(family_args)

    result = {
        "schema": "celltrack-alpha.h1-zero-wrfeat-graph-gate.v1",
        "arm": "points-only-zero-wrfeat-12",
        "scope": {
            "fit_train105_only": True,
            "family_order": list(FAMILY_ORDER),
            "forbidden_scope_access": False,
            "proposal_metrics_gate": False,
        },
        **execute_ordered(evaluate),
        "fallback": contract["fallback"],
        "contract": {
            "path": str(args.contract.relative_to(ROOT)),
            "sha256": args.contract_sha256,
        },
    }
    engine.run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def main() -> int:
    report = run(_parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS_HELD_FIT_BOTH_FAMILIES" else 3


if __name__ == "__main__":
    raise SystemExit(main())
