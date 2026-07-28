#!/usr/bin/env python3
"""Real 45-second idle preflight and zero-swap guarded command runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import time


def swap_used_mib() -> float:
    text = subprocess.check_output(["vm_stat"], text=True)
    page = re.search(r"page size of ([0-9]+) bytes", text)
    swapouts = re.search(r"^Swapouts:\s+([0-9]+)\.", text, re.MULTILINE)
    if page is None or swapouts is None:
        raise RuntimeError("cannot parse cumulative swapouts from vm_stat")
    return int(page.group(1)) * int(swapouts.group(1)) / 2**20


def write_new(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def preflight(output: Path, seconds: float) -> int:
    started = time.perf_counter()
    initial = swap_used_mib()
    peak = initial
    while time.perf_counter() - started < seconds:
        peak = max(peak, swap_used_mib())
        time.sleep(0.25)
    ending = swap_used_mib()
    peak = max(peak, ending)
    growth = peak - initial
    payload: dict[str, object] = {
        "schema": "celltrack-alpha.sa-division-deep-preflight.v1",
        "status": "PASS_ZERO_SWAP_GROWTH" if growth <= 0.0 else "FAIL_SWAP_GROWTH",
        "required_seconds": seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "initial_swap_mib": initial,
        "peak_swap_mib": peak,
        "ending_swap_mib": ending,
        "swap_growth_mib": growth,
        "swap_metric": "cumulative vm_stat Swapouts pages converted to MiB",
        "gpu_action": False,
        "external_action": False,
    }
    write_new(output, payload)
    return 0 if payload["status"] == "PASS_ZERO_SWAP_GROWTH" else 2


def run_guarded(preflight_path: Path, output: Path, command: list[str]) -> int:
    preflight_payload = json.loads(preflight_path.read_text(encoding="ascii"))
    if preflight_payload.get("status") != "PASS_ZERO_SWAP_GROWTH":
        raise RuntimeError("preflight did not pass")
    if float(preflight_payload.get("elapsed_seconds", 0.0)) < 45.0:
        raise RuntimeError("preflight shorter than 45 seconds")
    if time.time() - preflight_path.stat().st_mtime > 300:
        raise RuntimeError("preflight is stale")

    initial = swap_used_mib()
    peak = initial
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stopped = False
    while process.poll() is None:
        peak = max(peak, swap_used_mib())
        if peak > initial:
            stopped = True
            process.terminate()
            break
        time.sleep(0.1)
    stdout, _ = process.communicate(timeout=10)
    ending = swap_used_mib()
    peak = max(peak, ending)
    growth = peak - initial
    status = (
        "PASS"
        if process.returncode == 0 and growth <= 0.0 and not stopped
        else "FAIL"
    )
    payload = {
        "schema": "celltrack-alpha.sa-division-deep-resource-guard.v1",
        "status": status,
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        "command": command,
        "returncode": process.returncode,
        "stdout": stdout,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_tree_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        / 2**30,
        "initial_swap_mib": initial,
        "peak_swap_mib": peak,
        "ending_swap_mib": ending,
        "swap_growth_mib": growth,
        "forced_stop_on_swap_growth": stopped,
        "gpu_action": False,
        "external_action": False,
    }
    write_new(output, payload)
    print(stdout, end="")
    return 0 if status == "PASS" else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--seconds", type=float, default=45.0)
    run = subparsers.add_parser("run")
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "preflight":
        return preflight(args.output, args.seconds)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RuntimeError("missing guarded command")
    return run_guarded(args.preflight, args.output, command)


if __name__ == "__main__":
    raise SystemExit(main())
