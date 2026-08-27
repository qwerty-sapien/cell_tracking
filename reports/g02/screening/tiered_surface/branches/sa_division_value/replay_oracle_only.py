#!/usr/bin/env python3
"""Fresh oracle-only replay of the 38 frozen division repair bundles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "evaluate_division_value.py"
OUTPUT = HERE / "ORACLE_ONLY_REPLAY.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    specification = importlib.util.spec_from_file_location(
        "frozen_division_value_evaluator", SOURCE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load frozen division-value evaluator")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def receipt_from_banked_execution() -> dict:
    results_path = HERE / "N0_N3_RESULTS.json"
    provenance_path = HERE / "RUN_PROVENANCE.json"
    validation_path = HERE / "N0_N3_VALIDATION.json"
    guard_path = HERE / "resource_guard.json"
    preflight_path = HERE / "idle_preflight.json"
    results = json.loads(results_path.read_text(encoding="ascii"))
    provenance = json.loads(provenance_path.read_text(encoding="ascii"))
    validation = json.loads(validation_path.read_text(encoding="ascii"))
    guard = json.loads(guard_path.read_text(encoding="ascii"))
    preflight = json.loads(preflight_path.read_text(encoding="ascii"))
    expected_hashes = validation["artifact_sha256"]
    for name in (
        "N0_N3_RESULTS.json",
        "RUN_PROVENANCE.json",
        "resource_guard.json",
        "idle_preflight.json",
    ):
        if sha256(HERE / name) != expected_hashes[name]:
            raise RuntimeError(f"banked oracle artifact drift: {name}")
    single = provenance["compiler_receipts"][0]["single_action_receipts"]
    if (
        not validation["passed"]
        or validation["checks"] != 54
        or not guard["passed"]
        or float(guard["swap_growth_mib"]) != 0
        or not preflight["passed"]
        or float(preflight["swap_growth_mib"]) != 0
        or len(single) != 38
        or len({row["event_key"] for row in single}) != 38
        or provenance["fitting"]
        or provenance["bundle_derivation"]
        or provenance["gpu_used"]
        or provenance["kaggle_used"]
        or provenance["budget_debit"]
    ):
        raise RuntimeError("banked oracle execution contract drift")
    coalition = results["n0_ceiling"]["coalition"]
    ceiling = coalition["pooled"]["delta"]
    if (
        ceiling["division_tp"] != 38
        or ceiling["division_fp"] != 0
        or ceiling["division_fn"] != 4
        or not math.isclose(
            ceiling["division_jaccard"], 38 / 42, abs_tol=1e-15
        )
        or not math.isclose(
            ceiling["combined_score_delta"],
            ceiling["adjusted_edge_jaccard_delta"]
            + 0.1 * ceiling["division_jaccard"],
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("banked frozen-N1 oracle result drift")
    return {
        "schema": "celltrack-alpha.division-plan-dossier-oracle-receipt.v1",
        "status": "PASS_BANKED_ORACLE_REVERIFIED_FITTING_LOCKED",
        "observed_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "measurement_origin": {
            "kind": "BANKED_ORACLE_EXECUTION_REVERIFIED",
            "result_path": str(results_path),
            "result_sha256": sha256(results_path),
            "provenance_path": str(provenance_path),
            "provenance_sha256": sha256(provenance_path),
            "resource_guard_path": str(guard_path),
            "resource_guard_sha256": sha256(guard_path),
            "banked_elapsed_seconds": guard["elapsed_seconds"],
            "banked_peak_process_rss_mib": guard["peak_process_rss_mib"],
            "banked_swap_growth_mib": guard["swap_growth_mib"],
        },
        "information_label": "GT-DEPENDENT",
        "input_sha256": {
            "source_evaluator": sha256(SOURCE),
            "continuity_target_spec": sha256(
                HERE.parent / "sa_division_cont/continuity_target_spec.json"
            ),
            "n1_interface": provenance["inputs"]["n1_interface"]["sha256"],
            "n1_implementation": provenance["inputs"]["n1_implementation"][
                "sha256"
            ],
        },
        "authentication_receipt_count": len(provenance["inputs"]),
        "repair_bundle_count": 38,
        "identity_count": len(
            {
                row["identity"]
                for row in json.loads(
                    (HERE / "GT_DEPENDENCY_TABLE.json").read_text(
                        encoding="ascii"
                    )
                )["events"]
            }
        ),
        "frozen_n1_single_action_receipt_count": len(single),
        "frozen_n1_all_action_hashes_match": True,
        "pooled_baseline": coalition["pooled"]["baseline"],
        "pooled_candidate": coalition["pooled"]["candidate"],
        "delta": ceiling,
        "formula_check": {
            "adjusted_edge_delta": ceiling["adjusted_edge_jaccard_delta"],
            "weighted_division_delta": 0.1 * ceiling["division_jaccard"],
            "combined_delta": ceiling["combined_score_delta"],
            "identity": (
                "combined = adjusted_edge_delta + 0.1 * division_jaccard"
            ),
            "passes": True,
        },
        "family_results": coalition["families"],
        "execution": {
            "oracle_only": True,
            "fitting": False,
            "plan_amendment": False,
            "training": False,
            "kaggle": False,
            "gpu": False,
            "runpod": False,
            "submission": False,
            "full_pool_cache": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-banked",
        action="store_true",
        help="emit a fresh receipt for the already executed, hash-pinned oracle",
    )
    args = parser.parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    if args.verify_banked:
        result = receipt_from_banked_execution()
        OUTPUT.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "repair_bundles": result["repair_bundle_count"],
                    "combined_score_delta": result["delta"][
                        "combined_score_delta"
                    ],
                    "division_tp_fp_fn": [
                        result["delta"]["division_tp"],
                        result["delta"]["division_fp"],
                        result["delta"]["division_fn"],
                    ],
                    "output_sha256": sha256(OUTPUT),
                },
                sort_keys=True,
            )
        )
        return 0
    started = time.perf_counter()
    evaluator = load_source()
    provenance = evaluator.authenticate()
    specification = evaluator.load_json(evaluator.SPEC)
    events = list(specification["events"])
    if len(events) != 38:
        raise RuntimeError("frozen repair-bundle count drift")
    runtime = evaluator.FrozenEdgeRuntime(
        evaluator.CHECKPOINT, evaluator.AUXILIARY, evaluator.EDGE_HEAD
    )
    identities = sorted({str(row["identity"]) for row in events})
    graphs = {
        identity: evaluator.read_graph(identity) for identity in identities
    }
    assignments = [
        evaluator.classify_event(
            event, graphs[str(event["identity"])], runtime
        )
        for event in events
    ]
    coalitions, compiler_receipts = evaluator.score_all(events, assignments)
    ceiling = coalitions["abc"]["pooled"]["delta"]
    if (
        ceiling["division_tp"] != 38
        or ceiling["division_fp"] != 0
        or ceiling["division_fn"] != 4
        or not math.isclose(
            ceiling["division_jaccard"], 38 / 42, abs_tol=1e-15
        )
        or not math.isclose(
            ceiling["combined_score_delta"],
            ceiling["adjusted_edge_jaccard_delta"]
            + 0.1 * ceiling["division_jaccard"],
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("fresh frozen-N1 oracle result drift")
    single = compiler_receipts[0]["single_action_receipts"]
    if len(single) != 38 or len({row["event_key"] for row in single}) != 38:
        raise RuntimeError("frozen-N1 compiler receipt coverage drift")
    result = {
        "schema": "celltrack-alpha.division-plan-dossier-oracle-replay.v1",
        "status": "PASS_ORACLE_ONLY_FITTING_LOCKED",
        "observed_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "information_label": "GT-DEPENDENT",
        "input_sha256": {
            "source_evaluator": sha256(SOURCE),
            "continuity_target_spec": sha256(evaluator.SPEC),
            "n1_interface": sha256(
                evaluator.ROOT
                / "reports/g02/screening/tiered_surface/"
                "n1_feasibility_compiler/interface_v1.json"
            ),
            "n1_implementation": sha256(
                evaluator.ROOT / "src/celltrack_alpha/action_compiler.py"
            ),
        },
        "authentication_receipt_count": len(provenance),
        "repair_bundle_count": len(events),
        "identity_count": len(identities),
        "frozen_n1_single_action_receipt_count": len(single),
        "frozen_n1_all_action_hashes_match": True,
        "pooled_baseline": coalitions["abc"]["pooled"]["baseline"],
        "pooled_candidate": coalitions["abc"]["pooled"]["candidate"],
        "delta": ceiling,
        "formula_check": {
            "adjusted_edge_delta": ceiling["adjusted_edge_jaccard_delta"],
            "weighted_division_delta": 0.1 * ceiling["division_jaccard"],
            "combined_delta": ceiling["combined_score_delta"],
            "identity": "combined = adjusted_edge_delta + 0.1 * division_jaccard",
            "passes": True,
        },
        "family_results": coalitions["abc"]["families"],
        "elapsed_seconds": time.perf_counter() - started,
        "execution": {
            "oracle_only": True,
            "fitting": False,
            "plan_amendment": False,
            "training": False,
            "kaggle": False,
            "gpu": False,
            "runpod": False,
            "submission": False,
            "full_pool_cache": False,
        },
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "repair_bundles": result["repair_bundle_count"],
                "combined_score_delta": ceiling["combined_score_delta"],
                "division_tp_fp_fn": [
                    ceiling["division_tp"],
                    ceiling["division_fp"],
                    ceiling["division_fn"],
                ],
                "output_sha256": sha256(OUTPUT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
