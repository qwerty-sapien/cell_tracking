#!/usr/bin/env python3
"""Run terminal validation under a fresh fail-closed zero-swap guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--branch-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--guard-out", type=Path, required=True)
    parser.add_argument("--validation-out", type=Path, required=True)
    parser.add_argument("--path-audit-out", type=Path, required=True)
    parser.add_argument("--barrier-out", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.guard_out, args.validation_out, args.path_audit_out, args.barrier_out)):
        raise SystemExit("refusing to overwrite validation outputs")

    preflight = json.loads(args.preflight.read_text(encoding="ascii"))
    if preflight.get("status") != "PASS_ZERO_SWAP_GROWTH":
        raise SystemExit("fresh preflight did not pass")
    if time.time() - args.preflight.stat().st_mtime > 300:
        raise SystemExit("preflight is stale")

    root = args.root.resolve()
    branch = args.branch_dir.resolve()
    sys.path.insert(0, str(root / "scripts"))
    import run_local_training_guard as resources  # type: ignore

    commands = [
        [
            sys.executable,
            str(branch / "validate_artifacts.py"),
            "--repo",
            str(Path("/private/tmp/g02_phase1_20260728/sa_nodetopo")),
            "--branch-dir",
            str(branch),
            "--validation-out",
            str(args.validation_out),
            "--path-audit-out",
            str(args.path_audit_out),
        ],
        [
            sys.executable,
            str(root / "reports/g02/screening/tiered_surface/validate_phase1_barrier.py"),
            "--worktree",
            "/private/tmp/g02_phase1_20260728/sa_nodetopo",
            "--base",
            "2c74eed32c6f080faff045ae22847b98bbde8db0",
            "--branch",
            "codex/sa-nodetopo",
            "--allow-prefix",
            "reports/g02/screening/tiered_surface/branches/sa_nodetopo/",
        ],
    ]
    initial = resources.macos_swap_used_bytes()
    peak = initial
    started = time.time()
    status = "PASS"
    receipts = []
    for index, command in enumerate(commands):
        process = subprocess.Popen(
            command,
            cwd="/private/tmp/g02_phase1_20260728/sa_nodetopo",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            current = resources.macos_swap_used_bytes()
            peak = max(peak, current)
            if current > initial:
                process.terminate()
                status = "FAIL_SWAP_GROWTH"
                break
            time.sleep(0.25)
        output, _ = process.communicate(timeout=10)
        receipts.append({"command": command, "returncode": process.returncode, "output": output})
        if status == "PASS" and process.returncode != 0:
            status = "FAIL_COMMAND"
        if status != "PASS":
            break
        if index == 1:
            args.barrier_out.write_text(
                json.dumps(json.loads(output), indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
    ending = resources.macos_swap_used_bytes()
    peak = max(peak, ending)
    if peak > initial:
        status = "FAIL_SWAP_GROWTH"
    guard = {
        "schema": "celltrack-alpha.sa-nodetopo-validation-guard.v1",
        "status": status,
        "preflight_path": str(args.preflight.resolve()),
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial / 1024**2,
        "peak_swap_mib": peak / 1024**2,
        "ending_swap_mib": ending / 1024**2,
        "swap_growth_mib": (peak - initial) / 1024**2,
        "receipts": receipts,
        "external_action": False,
        "gpu_action": False,
    }
    args.guard_out.write_text(
        json.dumps(guard, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    for receipt in receipts:
        print(receipt["output"], end="")
    print(json.dumps({"status": status, "swap_growth_mib": guard["swap_growth_mib"]}))
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
