#!/usr/bin/env python3
"""45-second zero-swap preflight and guarded command runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import resource
import subprocess
import time


def swap_used_mib() -> float:
    text = subprocess.check_output(["vm_stat"], text=True)
    page = re.search(r"page size of ([0-9]+) bytes", text)
    swap = re.search(r"^Swapouts:\s+([0-9]+)\.", text, re.MULTILINE)
    if not page or not swap:
        raise RuntimeError("cannot parse vm_stat")
    return int(page.group(1)) * int(swap.group(1)) / 2**20


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def preflight(output: Path, seconds: float) -> int:
    started = time.perf_counter()
    initial = swap_used_mib()
    peak = initial
    while time.perf_counter() - started < seconds:
        peak = max(peak, swap_used_mib())
        time.sleep(0.25)
    ending = swap_used_mib()
    growth = max(peak, ending) - initial
    result = {
        "schema": "g02.sa-reverse.idle-preflight.v1",
        "status": "PASS" if growth <= 0 else "FAIL_SWAP_GROWTH",
        "required_seconds": seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "swap_start_mib": initial,
        "swap_peak_mib": peak,
        "swap_end_mib": ending,
        "swap_growth_mib": growth,
        "gpu": False,
    }
    write_once(output, result)
    return 0 if growth <= 0 else 2


def guarded(output: Path, command: list[str]) -> int:
    initial = swap_used_mib()
    peak = initial
    started = time.perf_counter()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    forced = False
    while process.poll() is None:
        peak = max(peak, swap_used_mib())
        if peak > initial:
            forced = True
            process.terminate()
            break
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    ending = swap_used_mib()
    growth = max(peak, ending) - initial
    status = "PASS" if process.returncode == 0 and growth <= 0 and not forced else "FAIL"
    result = {
        "schema": "g02.sa-reverse.resource-guard.v1",
        "status": status,
        "command": command,
        "returncode": process.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_child_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30,
        "swap_start_mib": initial,
        "swap_peak_mib": peak,
        "swap_end_mib": ending,
        "swap_growth_mib": growth,
        "forced_stop_on_swap_growth": forced,
        "stdout": stdout,
        "stderr": stderr,
        "gpu": False,
    }
    write_once(output, result)
    return 0 if status == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    idle = sub.add_parser("preflight")
    idle.add_argument("--output", type=Path, required=True)
    idle.add_argument("--seconds", type=float, default=45.0)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "preflight":
        return preflight(args.output, args.seconds)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("missing guarded command")
    return guarded(args.output, command)


if __name__ == "__main__":
    raise SystemExit(main())
