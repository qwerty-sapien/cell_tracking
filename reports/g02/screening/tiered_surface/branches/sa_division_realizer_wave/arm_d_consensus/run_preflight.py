#!/usr/bin/env python3
"""Forty-five-second zero-swap idle preflight for Arm D."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def swap_mib() -> float:
    text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    if match is None:
        raise RuntimeError(f"cannot parse swap usage: {text!r}")
    value = float(match.group(1))
    return value * 1024.0 if match.group(2) == "G" else value


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="ascii", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="D")
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial_swap = swap_mib()
    initial_load = os.getloadavg()
    started = time.time()
    peak_swap = initial_swap
    for _ in range(args.seconds):
        peak_swap = max(peak_swap, swap_mib())
        time.sleep(1)
    final_swap = swap_mib()
    payload = {
        "schema": "celltrack-alpha.idle-preflight.v1",
        "arm": args.arm,
        "requested_seconds": args.seconds,
        "elapsed_seconds": time.time() - started,
        "swap_before_mib": initial_swap,
        "swap_after_mib": final_swap,
        "swap_peak_mib": peak_swap,
        "swap_growth_mib": final_swap - initial_swap,
        "load_before": initial_load,
        "load_after": os.getloadavg(),
        "passed": peak_swap <= initial_swap and final_swap <= initial_swap,
        "cpu_only": True,
    }
    atomic_json(args.output, payload)
    if not payload["passed"]:
        raise SystemExit("preflight swap-growth gate failed")


if __name__ == "__main__":
    main()
