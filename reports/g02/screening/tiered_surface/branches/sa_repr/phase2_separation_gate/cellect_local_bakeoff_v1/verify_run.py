#!/usr/bin/env python3
"""Fail closed unless the isolated CELLECT run and frozen gate are complete."""

from __future__ import annotations

import json
from pathlib import Path

import constants


RUN = constants.HERE / "run"
EXPECTED_POPULATION = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)


def main() -> int:
    constants.verify_pins()
    plan = json.loads((RUN / "SHARD_PLAN.json").read_text())
    feature_terminal = json.loads(
        (RUN / "FEATURE_STAGE_TERMINAL.json").read_text()
    )
    if (
        plan.get("status") != "FROZEN_AUTHENTICATED"
        or plan.get("request_file_count") != 77
        or plan.get("request_count") != 1_391_458
        or plan.get("tile_jobs") != 217_595
        or plan.get("population_manifest_sha256") != EXPECTED_POPULATION
        or plan.get("checkpoint_bytes") != 10_197_221
        or plan.get("checkpoint_sha256")
        != constants.PINS[constants.CHECKPOINT]
        or feature_terminal.get("status")
        != "COMPLETE_77_ATOMIC_SHARDS_ZERO_SWAP"
        or feature_terminal.get("request_file_count") != 77
        or feature_terminal.get("request_count") != 1_391_458
        or feature_terminal.get("tile_jobs") != 217_595
        or float(feature_terminal.get("swap_growth_mib", -1)) != 0
    ):
        raise SystemExit("CELLECT feature-stage terminal drift")
    checkpoints = list((RUN / "checkpoints").glob("*/*.json"))
    if len(checkpoints) != 77:
        raise SystemExit(
            f"expected 77 atomic checkpoints, found {len(checkpoints)}"
        )
    guard_root = RUN / "guards"
    for row in plan["requests"]:
        prefix = f"{row['ordinal']:03d}_{row['action_class']}_"
        receipts = list(guard_root.glob(f"{prefix}*.json"))
        passing = []
        for path in receipts:
            value = json.loads(path.read_text())
            if (
                str(value.get("status", "")).startswith("PASS")
                and float(value.get("swap_growth_mib", -1)) == 0
            ):
                passing.append(path)
        if not passing:
            raise SystemExit(
                f"no zero-swap guard evidence for shard {row['ordinal']}"
            )
    for action_class in constants.CLASSES:
        manifest = json.loads(
            (
                RUN
                / "features"
                / action_class
                / "class_manifest.json"
            ).read_text()
        )
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("file_count")
            != constants.EXPECTED_FILES[action_class]
            or manifest.get("action_count")
            != constants.EXPECTED_ACTIONS[action_class]
            or manifest.get("population_manifest_sha256")
            != EXPECTED_POPULATION
            or manifest.get("checkpoint_sha256")
            != constants.PINS[constants.CHECKPOINT]
        ):
            raise SystemExit(f"CELLECT class manifest drift: {action_class}")
    evaluation_guard = json.loads((RUN / "guard_evaluate.json").read_text())
    result_path = RUN / "evaluation" / "SEPARATION_RESULT.json"
    result = json.loads(result_path.read_text())
    if (
        evaluation_guard.get("status") != "PASS"
        or float(evaluation_guard.get("swap_growth_mib", -1)) != 0
        or result.get("status") != "COMPLETE"
        or result.get("amendment_sha256")
        != constants.PINS[constants.AMENDMENT]
        or result.get("sealed_B0_sha256")
        != constants.PINS[constants.SEALED_B0]
        or result.get("population_manifest_sha256") != EXPECTED_POPULATION
        or result.get("execution_order") != ["B0_SEALED", "A1", "A0"]
        or result.get("barrier") != "HOLD_FOR_HUMAN_REVIEW_NO_GRAPH_EXECUTION"
        or result.get("execution", {}).get("kaggle") is not False
        or result.get("execution", {}).get("gpu_quota_hours") != 0
    ):
        raise SystemExit("CELLECT frozen evaluation drift")
    verdicts = {
        action_class: result["results"][action_class]["verdict"]
        for action_class in constants.CLASSES
    }
    if any(
        verdict not in ("PASS_UNLOCK_GRAPH_PREREG_REVIEW", "FAIL_PARK_C")
        for verdict in verdicts.values()
    ):
        raise SystemExit("CELLECT verdict token drift")
    print(
        json.dumps(
            {
                "status": "PASS",
                "atomic_checkpoints": len(checkpoints),
                "verdicts": verdicts,
                "result_sha256": constants.sha256(result_path),
                "kaggle_gpu_hours": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
