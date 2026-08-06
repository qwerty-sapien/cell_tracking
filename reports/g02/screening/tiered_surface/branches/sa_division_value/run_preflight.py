#!/usr/bin/env python3
"""Forty-five second idle/resource preflight for the CPU-only audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def swap_mib() -> float:
    text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    if match is None:
        raise RuntimeError(f"cannot parse swap usage: {text!r}")
    value = float(match.group(1))
    return value * 1024.0 if match.group(2) == "G" else value


def load_average() -> tuple[float, float, float]:
    return tuple(float(value) for value in os.getloadavg())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before_swap = swap_mib()
    before_load = load_average()
    samples = []
    started = time.time()
    for _ in range(args.seconds):
        samples.append({"time": time.time(), "swap_mib": swap_mib(), "load": load_average()})
        time.sleep(1)
    after_swap = swap_mib()
    after_load = load_average()
    result = {
        "schema": "celltrack-alpha.idle-preflight.v1",
        "requested_seconds": args.seconds,
        "elapsed_seconds": time.time() - started,
        "swap_before_mib": before_swap,
        "swap_after_mib": after_swap,
        "swap_growth_mib": after_swap - before_swap,
        "load_before": before_load,
        "load_after": after_load,
        "sample_count": len(samples),
        "passed": after_swap - before_swap <= 0.0,
        "cpu_only": True,
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    if not result["passed"]:
        raise SystemExit("preflight swap-growth gate failed")


if __name__ == "__main__":
    main()
