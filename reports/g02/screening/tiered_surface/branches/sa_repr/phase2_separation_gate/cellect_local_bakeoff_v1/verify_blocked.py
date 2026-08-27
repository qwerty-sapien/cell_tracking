#!/usr/bin/env python3
"""Verify the CELLECT local resource block from raw receipts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
REPORT = HERE / "BLOCKED_REPORT.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    report = load(REPORT)
    plan = load(RUN / "SHARD_PLAN.json")
    require(
        report["status"] == "BLOCKED_ZERO_SWAP_BEFORE_SHARD_0"
        and report["scientific_verdict"] is None
        and report["frozen_gate_ran"] is False,
        "resource-block status drift",
    )
    require(
        sha256(RUN / "SHARD_PLAN.json")
        == report["frozen_identities"]["shard_plan_sha256"],
        "shard-plan hash drift",
    )
    require(
        (
            len(plan["requests"]),
            plan["request_count"],
            sum(row["tile_jobs"] for row in plan["requests"]),
        )
        == (
            report["request_shards_total"],
            report["request_actions_total"],
            report["tile_jobs_total"],
        ),
        "request-plan counts drift",
    )
    require(
        plan["checkpoint_bytes"] == report["checkpoint"]["bytes"]
        and plan["checkpoint_sha256"] == report["checkpoint"]["sha256"],
        "checkpoint identity drift",
    )
    require(len(report["attempts"]) == 5, "attempt count drift")
    for attempt in report["attempts"]:
        ordinal = attempt["attempt"]
        preflight = RUN / f"preflight_features_attempt{ordinal}.json"
        guard = (
            RUN
            / "guards"
            / f"000_delete_train__44b6_0c582fdc_attempt{ordinal}.json"
        )
        require(
            sha256(preflight) == attempt["preflight_sha256"]
            and sha256(guard) == attempt["guard_sha256"],
            f"attempt receipt hash drift: {ordinal}",
        )
        preflight_data = load(preflight)
        guard_data = load(guard)
        require(
            preflight_data["status"] == "PASS_ZERO_SWAP_GROWTH"
            and preflight_data["swap_growth_mib"] == 0.0,
            f"preflight contract drift: {ordinal}",
        )
        require(
            guard_data["status"] == "FAIL_SWAP_GROWTH"
            and guard_data["forced_stop_on_swap_growth"] is True
            and guard_data["ordinal"] == 0
            and guard_data["returncode"] != 0
            and guard_data["swap_growth_mib"] > 0,
            f"guard contract drift: {ordinal}",
        )
        require(
            math.isclose(
                guard_data["swap_growth_mib"],
                attempt["shard_swap_growth_mib"],
                abs_tol=0.0,
            )
            and math.isclose(
                guard_data["elapsed_seconds"],
                attempt["elapsed_seconds"],
                abs_tol=0.0,
            ),
            f"attempt measurement drift: {ordinal}",
        )
    feature_files = (
        [path for path in (RUN / "features").rglob("*") if path.is_file()]
        if (RUN / "features").exists()
        else []
    )
    checkpoint_files = (
        [
            path
            for path in (RUN / "checkpoints").rglob("*")
            if path.is_file()
        ]
        if (RUN / "checkpoints").exists()
        else []
    )
    require(
        len(feature_files) == report["feature_outputs"] == 0
        and len(checkpoint_files) == report["atomic_checkpoints"] == 0,
        "partial output exists despite pre-shard block",
    )
    require(report["execution"]["local_only"], "local-only label missing")
    require(
        report["execution"]["zero_swap_guard_weakened"] is False,
        "zero-swap guard weakening recorded",
    )
    for forbidden in (
        "kaggle",
        "gpu_quota_debit",
        "runpod",
        "submission",
        "full_pool_cache",
        "training",
        "graph",
        "six_bba",
        "cv57",
        "plan_amendment",
        "dead_promotion",
    ):
        require(
            report["execution"][forbidden] is False,
            f"forbidden action recorded: {forbidden}",
        )
    print(
        json.dumps(
            {
                "status": "PASS_BLOCK_AUTHENTICATED_NO_VERDICT",
                "attempts": len(report["attempts"]),
                "feature_outputs": len(feature_files),
                "atomic_checkpoints": len(checkpoint_files),
                "scientific_verdict": None,
                "kaggle": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
