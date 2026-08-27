#!/usr/bin/env python3
"""Validate the complete isolated G-02 Phase-1 worktree wave fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


MANDATORY_AGENTS = {
    "SA-DELETE",
    "SA-SWAP",
    "SA-REPR",
    "SA-NODETOPO",
    "SA-DIVISION",
}
TERMINAL_STATUSES = {
    "TERMINAL_PENDING_ROOT_VALIDATION",
    "TERMINAL_AMENDED_PENDING_ROOT_VALIDATION",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def fail(message: str) -> None:
    raise SystemExit(message)


def validate_frozen_inputs(manifest: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, record in manifest["frozen_inputs"].items():
        path = Path(record["path"])
        if not path.is_file():
            fail(f"missing frozen input {name}: {path}")
        digest = sha256(path)
        if digest != record["sha256"]:
            fail(f"frozen input drift {name}: {digest}")
        observed[name] = digest
    return observed


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    validator_path = args.validator.resolve()
    output_path = args.output.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    agents = {record["id"]: record for record in manifest["agents"]}
    if set(agents) != MANDATORY_AGENTS:
        fail(f"mandatory lane mismatch: {sorted(agents)}")
    if manifest["wave_policy"]["lead"] != "SA-DELETE":
        fail("SA-DELETE is not the declared lead")
    if agents["SA-DELETE"]["status"] not in TERMINAL_STATUSES:
        fail("SA-DELETE is not terminal at the barrier")

    frozen = validate_frozen_inputs(manifest)
    lane_results: dict[str, Any] = {}
    forbidden_attestations: dict[str, Any] = {}
    for agent_id in sorted(MANDATORY_AGENTS):
        record = agents[agent_id]
        if record["status"] not in TERMINAL_STATUSES:
            fail(f"{agent_id} is not terminal: {record['status']}")
        worktree = Path(record["worktree"]).resolve()
        clean = run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=worktree,
        )
        if clean.returncode:
            fail(f"{agent_id} git status failed: {clean.stderr}")
        if clean.stdout.strip():
            fail(f"{agent_id} worktree is not clean:\n{clean.stdout}")

        command = [
            sys.executable,
            str(validator_path),
            "--worktree",
            str(worktree),
            "--base",
            manifest["git_base"],
            "--branch",
            record["branch"],
            "--allow-prefix",
            record["write_scope"],
        ]
        checked = run(command)
        if checked.returncode:
            fail(
                f"{agent_id} barrier validation failed:\n"
                f"{checked.stdout}{checked.stderr}"
            )
        lane_result = json.loads(checked.stdout)
        if lane_result["head_commit"] != record["head_commit"]:
            fail(f"{agent_id} HEAD drifted from the terminal manifest")
        lane_results[agent_id] = lane_result
        fragment_path = worktree / record["write_scope"] / "state_fragment.json"
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        forbidden = fragment.get("forbidden_scope", {})
        true_external_fields = sorted(
            key
            for key, value in forbidden.items()
            if value is True
            and any(
                token in key
                for token in (
                    "gpu",
                    "kaggle",
                    "cv57",
                    "submission",
                    "runpod",
                    "external",
                )
            )
        )
        if true_external_fields:
            fail(f"{agent_id} reports forbidden actions: {true_external_fields}")
        forbidden_attestations[agent_id] = {
            "cv57_used": forbidden.get("cv57_used"),
            "submission_action": forbidden.get("submission_action"),
            "runpod_action": forbidden.get("runpod_action"),
            "true_external_action_fields": true_external_fields,
        }

    budget_pin = manifest["kaggle_budget_at_launch"]
    budget_path = Path(budget_pin["path"])
    if not budget_path.is_file() or sha256(budget_path) != budget_pin["sha256"]:
        fail("Kaggle budget drifted during a no-GPU Phase-1 wave")
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    if (
        budget.get("active_authorizations") != []
        or float(budget.get("used_kernel_h")) != float(budget_pin["used_kernel_h"])
        or float(budget.get("remaining_kernel_h"))
        != float(budget_pin["remaining_kernel_h"])
    ):
        fail("Kaggle budget arithmetic or authorization state drifted")

    result = {
        "schema": "celltrack-alpha.phase1-root-barrier-validation.v1",
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_snapshot": manifest,
        "git_base": manifest["git_base"],
        "lead": manifest["wave_policy"]["lead"],
        "frozen_inputs": frozen,
        "lanes": lane_results,
        "forbidden_scope_attestations": forbidden_attestations,
        "mandatory_lane_count": len(lane_results),
        "kaggle_budget": {
            "path": str(budget_path),
            "sha256": sha256(budget_path),
            "used_kernel_h": budget["used_kernel_h"],
            "remaining_kernel_h": budget["remaining_kernel_h"],
            "active_authorizations": budget["active_authorizations"],
        },
        "gpu_authorizations": len(budget["active_authorizations"]),
        "kaggle_budget_debit_hours": 0.0,
        "cv57_used": False,
        "submission_action": False,
        "runpod_action": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
