#!/usr/bin/env python3
"""Assemble three complete isolated class measurements into the final surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(
    os.environ.get("CELLTRACK_ROOT", "/Users/agent/Desktop/cell_tracking")
).resolve()
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import selector_requirement as base  # noqa: E402


SCHEMA = "celltrack-alpha.selector-requirement-44b6-aggregate.v1"
CLASSES = ("delete", "occupied_swap", "restore_topology")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_class(path: Path, expected_class: str) -> dict[str, Any]:
    result_path = path.resolve() / "CLASS_RESULT.json"
    value = base.load_json(result_path)
    noise = value.get("noise", {})
    if (
        value.get("schema")
        != "celltrack-alpha.selector-requirement-44b6-class.v1"
        or value.get("status") != "COMPLETE"
        or value.get("action_class") != expected_class
        or value.get("oracle", {}).get("action_class") != expected_class
    ):
        raise base.DiagnosticError(
            f"incomplete or mismatched class result: {expected_class}"
        )
    runs = noise.get("seed_runs", [])
    expected = [
        (float(target), seed)
        for target in base.TARGET_AUCS
        for seed in range(base.SEED_COUNT)
    ]
    pairs = [
        (float(row["target_auc"]), int(row["seed_index"])) for row in runs
    ]
    complete = noise.get("status") == "COMPLETE" and pairs == expected
    population = value.get("population", {})
    degenerate = (
        noise.get("status") == "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE"
        and (
            int(population.get("beneficial", -1)) == 0
            or int(population.get("harmful", -1)) == 0
        )
        and runs == []
        and noise.get("aggregate") is None
        and noise.get("terminal_compiler_replay") == []
    )
    if not (complete or degenerate):
        raise base.DiagnosticError(
            f"class seed/AUC coverage drifted: {expected_class}"
        )
    if value["oracle"]["support_grid"] != base.support_grid(
        int(value["population"]["actions"])
    ):
        raise base.DiagnosticError(
            f"class support grid drifted: {expected_class}"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", type=Path, required=True)
    parser.add_argument("--occupied-swap", type=Path, required=True)
    parser.add_argument("--restore-topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1":
        raise SystemExit("binding local zero-swap guard is required")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite aggregate output")
    started = time.perf_counter()
    paths = {
        "delete": args.delete,
        "occupied_swap": args.occupied_swap,
        "restore_topology": args.restore_topology,
    }
    try:
        base._verify_pins()
        classes = {
            action_class: load_class(paths[action_class], action_class)
            for action_class in CLASSES
        }
        frozen_pins = classes["delete"]["frozen_input_sha256"]
        if any(
            classes[action_class]["frozen_input_sha256"] != frozen_pins
            for action_class in CLASSES
        ):
            raise base.DiagnosticError("class frozen-input receipts differ")

        baselines = {
            action_class: classes[action_class]["baseline"]
            for action_class in CLASSES
        }
        population_summary = {
            action_class: classes[action_class]["population"]
            for action_class in CLASSES
        }
        oracle_results = {
            action_class: {
                key: value
                for key, value in classes[action_class]["oracle"].items()
                if key != "compiler_replay"
            }
            for action_class in CLASSES
        }
        oracle_compiler = {
            action_class: classes[action_class]["oracle"]["compiler_replay"]
            for action_class in CLASSES
        }
        raw_noise = {
            action_class: {
                key: value
                for key, value in classes[action_class]["noise"].items()
                if key not in ("aggregate", "terminal_compiler_replay")
            }
            for action_class in CLASSES
        }
        aggregate_noise = {
            action_class: classes[action_class]["noise"]["aggregate"]
            for action_class in CLASSES
        }
        terminal_compiler = {
            action_class: classes[action_class]["noise"][
                "terminal_compiler_replay"
            ]
            for action_class in CLASSES
        }

        requirements = {}
        for action_class in CLASSES:
            target_req = base.requirement_for(
                oracle_results[action_class]["curve"],
                aggregate_noise[action_class],
                base.GRAPH_TARGET,
            )
            breakeven_req = base.requirement_for(
                oracle_results[action_class]["curve"],
                aggregate_noise[action_class],
                0.0,
            )
            requirements[action_class] = {
                "target_plus_0_005": target_req,
                "breakeven_0": breakeven_req,
                "frozen_gate_verdict": base.threshold_verdict(target_req),
            }
        too_lenient = [
            action_class
            for action_class in CLASSES
            if requirements[action_class]["frozen_gate_verdict"]["verdict"]
            == "TOO_LENIENT"
        ]
        barrier_status = (
            "HUMAN_REVIEW_REQUIRED" if too_lenient else "DIAGNOSTIC_COMPLETE"
        )
        barrier = {
            "schema": f"{base.SCHEMA}.barrier",
            "status": barrier_status,
            "amendment_proposed_for": too_lenient,
            "thresholds_changed": False,
            "tier_moves": 0,
            "promotions": 0,
            "dead_actions": 0,
            "ledger_written": False,
            "plan_changed": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
            "family_6bba_used": False,
            "cv57_used": False,
            "external_action": False,
        }

        output.mkdir(parents=True)
        base.canonical_json(
            output / "dose_response.json",
            {
                "schema": f"{base.SCHEMA}.dose-response",
                "status": "COMPLETE",
                "primary_metric": "adjusted_edge_jaccard_delta",
                "baselines": baselines,
                "classes": oracle_results,
                "compiler_replay": oracle_compiler,
            },
        )
        base.canonical_json(
            output / "noise_surface.json",
            {
                "schema": f"{base.SCHEMA}.noise-surface",
                "status": "COMPLETE",
                "target_aucs": list(base.TARGET_AUCS),
                "seed_count": base.SEED_COUNT,
                "classes": raw_noise,
                "aggregates": aggregate_noise,
                "terminal_compiler_replay": terminal_compiler,
            },
        )
        base.canonical_json(
            output / "requirement_surface.json",
            {
                "schema": f"{base.SCHEMA}.requirement-surface",
                "status": "COMPLETE",
                "grid_only": True,
                "interpolation": False,
                "requirements": requirements,
            },
        )
        base.canonical_json(output / "BARRIER.json", barrier)
        state_fragment = {
            "schema": f"{base.SCHEMA}.state-fragment",
            "status": barrier_status,
            "node": "SELECTOR-REQUIREMENT-44B6-DIAGNOSTIC",
            "diagnostic_only": True,
            "requirements": requirements,
            "tier_moves": [],
            "threshold_changes": [],
            "proposed_amendment": bool(too_lenient),
            "human_review_required": bool(too_lenient),
            "hard_stops_preserved": barrier,
        }
        base.canonical_json(output / "state_fragment.json", state_fragment)

        ledger_lines = [
            "# Proposed ledger row — root review only",
            "",
            "| ID | Status | Evidence | Finding | Consequence |",
            "|---|---|---|---|---|",
            "| `SELECTOR-REQUIREMENT-44B6-DIAGNOSTIC` | "
            f"`{barrier_status}` (diagnostic; no tier move) | "
            "`dose_response.json`, `noise_surface.json`, "
            "`requirement_surface.json`, `BARRIER.json` | "
            + "; ".join(
                f"{name}: {value['frozen_gate_verdict']['verdict']}"
                for name, value in requirements.items()
            )
            + " | Frozen thresholds unchanged; "
            + (
                "amendment proposal awaits human review."
                if too_lenient
                else "no threshold amendment triggered."
            )
            + " |",
            "",
        ]
        (output / "PROPOSED_LEDGER_ROW.md").write_text(
            "\n".join(ledger_lines), encoding="utf-8"
        )
        if too_lenient:
            proposal = [
                "# Proposed separation-gate threshold amendment — human decision required",
                "",
                "This proposal is evidence only. It does not modify the frozen gate.",
                "",
                "The following feasible action classes exceeded at least one current "
                "floor (AUC 0.65, retention 0.15) at the minimum tested selector "
                "setting whose mean exact 44b6 adjusted-edge-Jaccard delta reached "
                "+0.005:",
                "",
            ]
            for action_class in too_lenient:
                req = requirements[action_class]["target_plus_0_005"][
                    "mean_requirement"
                ]
                if req is None:
                    proposal.append(
                        f"- `{action_class}`: no tested AUC through 0.99 reached +0.005."
                    )
                else:
                    proposal.append(
                        f"- `{action_class}`: tested AUC {req['target_auc']:.2f}; "
                        f"mean CG0-budget retention "
                        f"{req['retention']['mean']:.6f}; support prefix "
                        f"{req['candidate_prefix_N']}."
                    )
            proposal.extend(
                [
                    "",
                    "Proposed human decision: retain the current permissive screen "
                    "as an unlock-only diagnostic with an explicit warning that it "
                    "is not graph-sufficient, or raise class-specific floors no "
                    "lower than the measured grid requirements after reviewing the "
                    "complete seed distributions.",
                    "",
                    "**HALT: HUMAN_REVIEW_REQUIRED.**",
                    "",
                ]
            )
            (output / "AMENDMENT_PROPOSAL.md").write_text(
                "\n".join(proposal), encoding="utf-8"
            )
        report = base.markdown_report(
            population_summary, oracle_results, requirements, barrier
        )
        (output / "TERMINAL_REPORT.md").write_text(report, encoding="utf-8")
        class_receipts = {
            action_class: {
                "path": str((paths[action_class].resolve() / "CLASS_RESULT.json")),
                "sha256": sha256(
                    paths[action_class].resolve() / "CLASS_RESULT.json"
                ),
                "elapsed_seconds": classes[action_class]["elapsed_seconds"],
            }
            for action_class in CLASSES
        }
        result = {
            "schema": base.SCHEMA,
            "aggregate_schema": SCHEMA,
            "status": barrier_status,
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_input_sha256": frozen_pins,
            "class_receipts": class_receipts,
            "population": population_summary,
            "requirements": requirements,
            "barrier": barrier,
        }
        base.canonical_json(output / "RESULT.json", result)
    except Exception as exc:
        print(f"aggregate diagnostic failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
