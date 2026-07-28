#!/usr/bin/env python3
"""Zero-swap 45-second preflight and process-tree resource guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


def swap_used_mib() -> float:
    text = subprocess.check_output(["vm_stat"], text=True)
    page_match = re.search(r"page size of ([0-9]+) bytes", text)
    swap_match = re.search(r"^Swapouts:\s+([0-9]+)\.", text, re.MULTILINE)
    if not page_match or not swap_match:
        raise RuntimeError(f"cannot parse vm_stat swapouts: {text}")
    pages = int(swap_match.group(1))
    page_bytes = int(page_match.group(1))
    return pages * page_bytes / 2**20


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def preflight(output: Path, seconds: float) -> int:
    start = time.perf_counter()
    swap_start = swap_used_mib()
    swap_peak = swap_start
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    while time.perf_counter() - start < seconds:
        swap_peak = max(swap_peak, swap_used_mib())
        rss_peak = max(rss_peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        time.sleep(0.25)
    swap_end = swap_used_mib()
    growth = max(swap_peak, swap_end) - swap_start
    result = {
        "schema": "g02.sa-repr-multi.idle-preflight.v1",
        "status": "PASS" if growth <= 0.0 else "FAIL_SWAP_GROWTH",
        "required_seconds": seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "swap_start_mib": swap_start,
        "swap_peak_mib": swap_peak,
        "swap_end_mib": swap_end,
        "swap_growth_mib": growth,
        "swap_metric": "cumulative vm_stat Swapouts pages converted to MiB",
        "peak_guard_rss_gib": rss_peak / 2**30,
        "gpu": False,
    }
    write_once(output, result)
    return 0 if result["status"] == "PASS" else 2


def guarded(output: Path, command: list[str]) -> int:
    swap_start = swap_used_mib()
    swap_peak = swap_start
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    forced_stop = False
    while process.poll() is None:
        swap_peak = max(swap_peak, swap_used_mib())
        if swap_peak > swap_start:
            forced_stop = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    swap_end = swap_used_mib()
    growth = max(swap_peak, swap_end) - swap_start
    returncode = process.returncode
    peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    status = "PASS" if returncode == 0 and growth <= 0.0 and not forced_stop else "FAIL"
    result = {
        "schema": "g02.sa-repr-multi.resource-guard.v1",
        "status": status,
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_tree_rss_gib": peak_rss / 2**30,
        "swap_start_mib": swap_start,
        "swap_peak_mib": swap_peak,
        "swap_end_mib": swap_end,
        "swap_growth_mib": growth,
        "swap_metric": "cumulative vm_stat Swapouts pages converted to MiB",
        "forced_stop_on_swap_growth": forced_stop,
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
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing guarded command")
    return guarded(args.output, command)


if __name__ == "__main__":
    raise SystemExit(main())
