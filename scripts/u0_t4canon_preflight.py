#!/usr/bin/env python3
"""Record the U0 clean-idle preflight and require zero swap growth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import run_local_training_guard as resources


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--seconds", type=float, default=45.0)
    value.add_argument("--poll-seconds", type=float, default=1.0)
    value.add_argument("--report", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.seconds < 45.0 or args.poll_seconds <= 0:
        raise SystemExit("U0 preflight requires at least 45 seconds")
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite: {args.report}")
    initial_swap = resources.macos_swap_used_bytes()
    peak_swap = initial_swap
    minimum_free = resources.macos_memory_free_percent()
    started = time.time()
    while time.time() - started < args.seconds:
        current = resources.macos_swap_used_bytes()
        peak_swap = max(peak_swap, current)
        minimum_free = min(minimum_free, resources.macos_memory_free_percent())
        if current > initial_swap:
            break
        time.sleep(args.poll_seconds)
    ending_swap = resources.macos_swap_used_bytes()
    elapsed = time.time() - started
    result = {
        "schema": "celltrack-alpha.u0-t4canon-preflight.v1",
        "status": (
            "PASS_ZERO_SWAP_GROWTH"
            if peak_swap == initial_swap and elapsed >= 45.0
            else "FAIL_SWAP_GROWTH"
        ),
        "required_seconds": 45.0,
        "elapsed_seconds": elapsed,
        "initial_swap_mib": initial_swap / 1024**2,
        "peak_swap_mib": peak_swap / 1024**2,
        "ending_swap_mib": ending_swap / 1024**2,
        "swap_growth_mib": (peak_swap - initial_swap) / 1024**2,
        "minimum_system_memory_free_percent": minimum_free,
        "external_action": False,
        "gpu_action": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS_ZERO_SWAP_GROWTH" else 3


if __name__ == "__main__":
    raise SystemExit(main())
