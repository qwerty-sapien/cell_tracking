#!/usr/bin/env python3
"""Strict host-wide zero-swap guard for the frozen R0 B4 CPU screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE
while WORKTREE_ROOT.parent != WORKTREE_ROOT and not (WORKTREE_ROOT / "src").is_dir():
    WORKTREE_ROOT = WORKTREE_ROOT.parent
PYTHON = Path("/Users/agent/Desktop/cell_tracking/.venv313/bin/python")
RUNNER = HERE / "evaluate_r0_shell_nolink.py"


class GuardError(RuntimeError):
    """The resource-guard contract was violated."""


def vm_sample() -> tuple[int, int, int]:
    text = subprocess.run(
        ["vm_stat"], check=True, capture_output=True, text=True
    ).stdout
    page_match = re.search(r"page size of (\d+) bytes", text)
    swap_match = re.search(r"^Swapouts:\s+(\d+)\.$", text, re.MULTILINE)
    free_match = re.search(r"^Pages free:\s+(\d+)\.$", text, re.MULTILINE)
    speculative_match = re.search(
        r"^Pages speculative:\s+(\d+)\.$", text, re.MULTILINE
    )
    if None in (page_match, swap_match, free_match, speculative_match):
        raise GuardError("cannot parse vm_stat")
    assert page_match is not None
    assert swap_match is not None
    assert free_match is not None
    assert speculative_match is not None
    return (
        int(swap_match.group(1)),
        int(page_match.group(1)),
        int(free_match.group(1)) + int(speculative_match.group(1)),
    )


def write_receipt(path: Path, value: dict) -> None:
    if path.exists():
        raise GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-seconds", type=int, default=45)
    parser.add_argument("--hard-seconds", type=int, default=7200)
    args = parser.parse_args()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if WORKTREE_ROOT not in output.parents or WORKTREE_ROOT not in receipt.parents:
        raise GuardError("guard artifacts must remain inside the lane worktree")
    if args.preflight_seconds < 45 or args.hard_seconds > 7200:
        raise GuardError("guard bounds cannot weaken the frozen contract")

    idle_start, page_size, free_start = vm_sample()
    minimum_free = free_start
    idle_deadline = time.monotonic() + args.preflight_seconds
    current = idle_start
    while time.monotonic() < idle_deadline:
        time.sleep(min(1.0, max(0.0, idle_deadline - time.monotonic())))
        current, observed_page_size, free_pages = vm_sample()
        minimum_free = min(minimum_free, free_pages)
        if observed_page_size != page_size or current != idle_start:
            write_receipt(
                receipt,
                {
                    "schema": "g02.r0-shell.local-resource-guard.v1",
                    "status": "PREFLIGHT_FAIL_SWAP_GROWTH",
                    "preflight_seconds": args.preflight_seconds,
                    "swapout_pages_start": idle_start,
                    "swapout_pages_end": current,
                    "swap_growth_bytes": (current - idle_start) * page_size,
                    "page_size": page_size,
                    "free_pages_start": free_start,
                    "minimum_free_pages": minimum_free,
                    "child_started": False,
                },
            )
            return 2

    if args.preflight_only:
        write_receipt(
            receipt,
            {
                "schema": "g02.r0-shell.local-resource-guard.v1",
                "status": "PASS_PREFLIGHT_ZERO_SWAP_GROWTH",
                "preflight_seconds": args.preflight_seconds,
                "swapout_pages_start": idle_start,
                "swapout_pages_end": current,
                "swap_growth_bytes": 0,
                "page_size": page_size,
                "free_pages_start": free_start,
                "minimum_free_pages": minimum_free,
                "child_started": False,
            },
        )
        return 0

    command = [str(PYTHON), str(RUNNER), "--output", str(output)]
    started = time.monotonic()
    process = subprocess.Popen(command)
    terminal_reason = "CHILD_EXIT"
    swap_end = idle_start
    while process.poll() is None:
        swap_end, observed_page_size, free_pages = vm_sample()
        minimum_free = min(minimum_free, free_pages)
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
    swap_end, _page_size, free_end = vm_sample()
    minimum_free = min(minimum_free, free_end)
    status = (
        "PASS"
        if returncode == 0 and terminal_reason == "CHILD_EXIT" and swap_end == idle_start
        else terminal_reason
    )
    write_receipt(
        receipt,
        {
            "schema": "g02.r0-shell.local-resource-guard.v1",
            "status": status,
            "command": command,
            "preflight_seconds": args.preflight_seconds,
            "hard_seconds": args.hard_seconds,
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "swapout_pages_start": idle_start,
            "swapout_pages_end": swap_end,
            "swap_growth_bytes": (swap_end - idle_start) * page_size,
            "page_size": page_size,
            "free_pages_start": free_start,
            "minimum_free_pages": minimum_free,
        },
    )
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
