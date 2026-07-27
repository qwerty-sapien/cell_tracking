#!/usr/bin/env python3
"""Run the focused audit suite under a fail-closed zero-swap guard."""

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
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    branch = args.branch_dir.resolve()
    preflight = json.loads(args.preflight.read_text(encoding="ascii"))
    if preflight.get("status") != "PASS_ZERO_SWAP_GROWTH":
        raise SystemExit("fresh 45-second preflight did not pass")
    if time.time() - args.preflight.stat().st_mtime > 300:
        raise SystemExit("preflight is stale (>300 seconds)")
    if args.guard_out.exists() or args.audit_out.exists():
        raise SystemExit("refusing to overwrite audit outputs")

    sys.path.insert(0, str(root / "scripts"))
    import run_local_training_guard as resources  # type: ignore

    commands = [
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            str(branch / "test_audit_candidate_universe.py"),
        ],
        [
            sys.executable,
            str(branch / "audit_candidate_universe.py"),
            "--root",
            str(root),
            "--out",
            str(args.audit_out),
        ],
    ]
    initial = resources.macos_swap_used_bytes()
    peak = initial
    minimum_free = resources.macos_memory_free_percent()
    started = time.time()
    receipts: list[dict[str, object]] = []
    status = "PASS"
    for command in commands:
        process = subprocess.Popen(
            command,
            cwd=str(branch),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output_parts: list[str] = []
        while process.poll() is None:
            current = resources.macos_swap_used_bytes()
            peak = max(peak, current)
            minimum_free = min(minimum_free, resources.macos_memory_free_percent())
            if current > initial:
                process.terminate()
                status = "FAIL_SWAP_GROWTH"
                break
            time.sleep(0.25)
        output, _ = process.communicate(timeout=10)
        output_parts.append(output or "")
        returncode = process.returncode
        receipts.append(
            {
                "command": command,
                "returncode": returncode,
                "output": "".join(output_parts),
            }
        )
        if status != "PASS" or returncode != 0:
            status = status if status != "PASS" else "FAIL_COMMAND"
            break

    ending = resources.macos_swap_used_bytes()
    peak = max(peak, ending)
    if peak > initial:
        status = "FAIL_SWAP_GROWTH"
    result = {
        "schema": "celltrack-alpha.sa-nodetopo-local-guard.v1",
        "status": status,
        "preflight_path": str(args.preflight.resolve()),
        "preflight_sha256": __import__("hashlib").sha256(
            args.preflight.read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial / 1024**2,
        "peak_swap_mib": peak / 1024**2,
        "ending_swap_mib": ending / 1024**2,
        "swap_growth_mib": (peak - initial) / 1024**2,
        "minimum_system_memory_free_percent": minimum_free,
        "receipts": receipts,
        "external_action": False,
        "gpu_action": False,
    }
    args.guard_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    for receipt in receipts:
        print(receipt["output"], end="")
    print(json.dumps({"status": status, "swap_growth_mib": result["swap_growth_mib"]}))
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
