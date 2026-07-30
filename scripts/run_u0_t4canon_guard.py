#!/usr/bin/env python3
"""Run one U0 replay while enforcing zero swap growth and recording resources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_local_training_guard as resources


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--poll-seconds", type=float, default=0.25)
    value.add_argument("command", nargs=argparse.REMAINDER)
    return value


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> int:
    args = parser().parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.poll_seconds <= 0 or args.report.exists():
        raise SystemExit("invalid U0 guarded replay arguments")
    initial_swap = resources.macos_swap_used_bytes()
    peak_swap = initial_swap
    minimum_free = resources.macos_memory_free_percent()
    started = time.time()
    process = subprocess.Popen(command, start_new_session=True)
    peak_rss = 0
    stop_reason = "child_failed"
    try:
        while True:
            return_code = process.poll()
            peak_rss = max(
                peak_rss, resources.process_tree_rss_bytes(process.pid)
            )
            current_swap = resources.macos_swap_used_bytes()
            peak_swap = max(peak_swap, current_swap)
            minimum_free = min(
                minimum_free, resources.macos_memory_free_percent()
            )
            if current_swap > initial_swap:
                stop_reason = "swap_growth"
                _stop(process)
                break
            if return_code is not None:
                stop_reason = "completed" if return_code == 0 else "child_failed"
                break
            time.sleep(args.poll_seconds)
    finally:
        _stop(process)
    ending_swap = resources.macos_swap_used_bytes()
    result = {
        "schema": "celltrack-alpha.u0-t4canon-guard.v1",
        "status": (
            "PASS"
            if stop_reason == "completed" and peak_swap == initial_swap
            else "FAIL_CLOSED"
        ),
        "stop_reason": stop_reason,
        "return_code": process.returncode,
        "elapsed_seconds": time.time() - started,
        "peak_process_tree_rss_gib": peak_rss / 1024**3,
        "initial_swap_mib": initial_swap / 1024**2,
        "peak_swap_mib": peak_swap / 1024**2,
        "ending_swap_mib": ending_swap / 1024**2,
        "swap_growth_mib": (peak_swap - initial_swap) / 1024**2,
        "minimum_system_memory_free_percent": minimum_free,
        "command": command,
        "external_action": False,
        "gpu_action": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
