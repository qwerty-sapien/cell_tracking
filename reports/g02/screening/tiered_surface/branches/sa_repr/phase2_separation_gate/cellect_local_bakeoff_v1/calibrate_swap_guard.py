#!/usr/bin/env python3
"""Measure host idle swapout drift and emit a proposal-only CELLECT guard contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "SWAP_GUARD_CALIBRATION.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def swapout_bytes() -> int:
    report = subprocess.check_output(["vm_stat"], text=True)
    page = re.search(r"page size of ([0-9]+) bytes", report)
    swap = re.search(r"^Swapouts:\s+([0-9]+)\.", report, re.MULTILINE)
    if not page or not swap:
        raise RuntimeError("cannot parse vm_stat swapouts")
    return int(page.group(1)) * int(swap.group(1))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def measure_window(seconds: float, interval_seconds: float) -> dict[str, object]:
    started = time.monotonic()
    start = swapout_bytes()
    samples = [start]
    while time.monotonic() - started < seconds:
        time.sleep(min(interval_seconds, max(0.0, seconds - (time.monotonic() - started))))
        samples.append(swapout_bytes())
    end = samples[-1]
    return {
        "started_utc": now(),
        "elapsed_seconds": time.monotonic() - started,
        "sample_interval_seconds": interval_seconds,
        "sample_count": len(samples),
        "swap_start_mib": start / 2**20,
        "swap_peak_mib": max(samples) / 2**20,
        "swap_end_mib": end / 2**20,
        "idle_growth_mib": (max(samples) - start) / 2**20,
        "idle_end_growth_mib": (end - start) / 2**20,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.seconds <= 0 or args.windows not in (1, 3) or args.interval_seconds <= 0:
        raise SystemExit("requires one or three positive-duration windows and positive sampling interval")
    windows = [measure_window(args.seconds, args.interval_seconds) for _ in range(args.windows)]
    maximum = max(float(window["idle_growth_mib"]) for window in windows)
    threshold = max(256.0, 5.0 * maximum)
    result = {
        "schema": "celltrack-alpha.cellect-swap-guard-calibration.v1",
        "status": "MEASURED_PROPOSAL_ONLY" if args.windows == 3 else "MEASURED_IDLE_WINDOW",
        "measured_utc": now(),
        "windows": windows,
        "max_idle_growth_mib": maximum,
        "proposed_guard": {
            "swap_growth_threshold_mib": threshold,
            "formula": "max(256 MiB, 5 * measured max 60-second idle growth)",
            "monitor_interval_seconds": 0.25,
            "sustained_growth_seconds": 15.0,
            "immediate_abort_multiple": 1.5,
            "hard_abort": "abort if growth remains at or above the threshold for 15 continuous seconds; also abort immediately at 1.5x threshold",
            "continuous_monitoring": True,
        },
        "historical_reassessment": {
            "below_proposed_threshold_noise_candidates_mib": [3.5, 12.875, 30.625],
            "genuine_growth_candidate_mib": 488.6875,
            "conclusion": "proposal only; no CELLECT rerun, tier move, or feasibility verdict",
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
