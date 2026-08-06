#!/usr/bin/env python3
"""Forty-five-second zero-swap idle preflight for Arm C."""

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


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before_swap = swap_mib()
    before_load = tuple(float(value) for value in os.getloadavg())
    started = time.time()
    samples = []
    for _ in range(args.seconds):
        samples.append(
            {
                "time": time.time(),
                "swap_mib": swap_mib(),
                "load": tuple(float(value) for value in os.getloadavg()),
            }
        )
        time.sleep(1)
    after_swap = swap_mib()
    result = {
        "schema": "celltrack-alpha.idle-preflight.v1",
        "arm": "C",
        "requested_seconds": args.seconds,
        "elapsed_seconds": time.time() - started,
        "swap_before_mib": before_swap,
        "swap_after_mib": after_swap,
        "swap_growth_mib": after_swap - before_swap,
        "load_before": before_load,
        "load_after": tuple(float(value) for value in os.getloadavg()),
        "sample_count": len(samples),
        "passed": after_swap - before_swap <= 0.0,
        "cpu_only": True,
    }
    atomic_json(args.output, result)
    if not result["passed"]:
        raise SystemExit("preflight swap-growth gate failed")


if __name__ == "__main__":
    main()
