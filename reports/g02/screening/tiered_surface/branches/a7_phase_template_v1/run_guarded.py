#!/usr/bin/env python3
"""Strict zero-swap and runtime guard for the local A7 CPU screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = Path(__file__).resolve().parents[6]
PYTHON = Path("/Users/agent/Desktop/cell_tracking/.venv/bin/python")
RUNNER = HERE / "run_a7_phase_template.py"
RECOVERY_RUNNER = HERE / "recover_zero_edit_result.py"


class GuardError(RuntimeError):
    pass


def vm_sample() -> tuple[int, int]:
    text = subprocess.run(
        ["vm_stat"], check=True, capture_output=True, text=True
    ).stdout
    page_match = re.search(r"page size of (\d+) bytes", text)
    swap_match = re.search(r"^Swapouts:\s+(\d+)\.$", text, re.MULTILINE)
    if page_match is None or swap_match is None:
        raise GuardError("cannot parse vm_stat")
    return int(swap_match.group(1)), int(page_match.group(1))


def write_receipt(path: Path, value: dict) -> None:
    if path.exists():
        raise GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("features", "evaluate", "all", "recover"), default="all"
    )
    parser.add_argument("--preflight-seconds", type=int, default=45)
    parser.add_argument("--hard-seconds", type=int, default=7200)
    args = parser.parse_args()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if WORKTREE_ROOT not in output.parents or WORKTREE_ROOT not in receipt.parents:
        raise GuardError("guard output must remain inside the lane worktree")
    if args.preflight_seconds < 45 or args.hard_seconds > 7200:
        raise GuardError("guard bounds cannot weaken the frozen contract")

    idle_start, page_size = vm_sample()
    idle_deadline = time.monotonic() + args.preflight_seconds
    while time.monotonic() < idle_deadline:
        time.sleep(min(1.0, max(0.0, idle_deadline - time.monotonic())))
        current, observed_page_size = vm_sample()
        if observed_page_size != page_size or current != idle_start:
            write_receipt(
                receipt,
                {
                    "schema": "celltrack-alpha.a7-local-resource-guard.v1",
                    "status": "PREFLIGHT_FAIL_SWAP_GROWTH",
                    "preflight_seconds": args.preflight_seconds,
                    "swapout_pages_start": idle_start,
                    "swapout_pages_end": current,
                    "page_size": page_size,
                    "child_started": False,
                },
            )
            return 2

    command = [
        str(PYTHON),
        str(RECOVERY_RUNNER if args.stage == "recover" else RUNNER),
        "--output",
        str(output),
        "--stage",
        args.stage,
    ]
    started = time.monotonic()
    process = subprocess.Popen(command)
    terminal_reason = "CHILD_EXIT"
    swap_end = idle_start
    while process.poll() is None:
        swap_end, observed_page_size = vm_sample()
        if observed_page_size != page_size or swap_end != idle_start:
            terminal_reason = "KILL_SWAP_GROWTH"
            process.terminate()
            break
        if time.monotonic() - started > args.hard_seconds:
            terminal_reason = "KILL_RUNTIME"
            process.terminate()
            break
        time.sleep(2.0)
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    returncode = process.wait()
    elapsed = time.monotonic() - started
    swap_end, _ = vm_sample()
    status = (
        "PASS"
        if returncode == 0 and terminal_reason == "CHILD_EXIT" and swap_end == idle_start
        else terminal_reason
    )
    write_receipt(
        receipt,
        {
            "schema": "celltrack-alpha.a7-local-resource-guard.v1",
            "status": status,
            "stage": args.stage,
            "command": command,
            "preflight_seconds": args.preflight_seconds,
            "hard_seconds": args.hard_seconds,
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "peak_rss_bytes": None,
            "swapout_pages_start": idle_start,
            "swapout_pages_end": swap_end,
            "swap_growth_bytes": (swap_end - idle_start) * page_size,
            "page_size": page_size,
        },
    )
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
