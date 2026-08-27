#!/usr/bin/env python3
"""Fresh idle preflight and fail-closed zero-swap subprocess guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time


class GuardError(RuntimeError):
    pass


def swap_used_bytes() -> int:
    text = subprocess.check_output(["vm_stat"], text=True)
    page = re.search(r"page size of ([0-9]+) bytes", text)
    swap = re.search(r"^Swapouts:\s+([0-9]+)\.", text, re.MULTILINE)
    if not page or not swap:
        raise GuardError("cannot parse vm_stat swapouts")
    return int(page.group(1)) * int(swap.group(1))


def write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def preflight(output: Path, seconds: float) -> int:
    if seconds < 45:
        raise GuardError("preflight must be at least 45 seconds")
    started = time.time()
    initial = swap_used_bytes()
    peak = initial
    while time.time() - started < seconds:
        peak = max(peak, swap_used_bytes())
        time.sleep(0.25)
    ending = swap_used_bytes()
    growth = max(peak, ending) - initial
    payload = {
        "schema": "celltrack-alpha.dinov2-local-idle-preflight.v2",
        "status": "PASS_ZERO_SWAP_GROWTH" if growth == 0 else "FAIL_SWAP_GROWTH",
        "observed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": time.time() - started,
        "swap_start_mib": initial / 2**20,
        "swap_peak_mib": peak / 2**20,
        "swap_end_mib": ending / 2**20,
        "swap_growth_mib": growth / 2**20,
        "external_action": False,
        "kaggle_action": False,
        "runpod_action": False,
    }
    write_once(output, payload)
    return 0 if growth == 0 else 3


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_guarded(preflight_path: Path, output: Path, command: list[str]) -> int:
    if output.exists():
        raise GuardError(f"refusing to overwrite {output}")
    fresh = json.loads(preflight_path.read_text(encoding="ascii"))
    if (
        fresh.get("status") != "PASS_ZERO_SWAP_GROWTH"
        or float(fresh.get("swap_growth_mib", -1)) != 0
        or float(fresh.get("elapsed_seconds", 0)) < 45
        or time.time() - preflight_path.stat().st_mtime > 180
    ):
        raise GuardError("fresh zero-swap preflight absent")
    initial = swap_used_bytes()
    peak = initial
    started = time.time()
    environment = dict(os.environ)
    environment["CELLTRACK_DINO_GUARD_ACTIVE"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    status = "PASS"
    while process.poll() is None:
        try:
            line = process.stdout.readline()
        except BlockingIOError:
            line = ""
        if line:
            lines.append(line)
            print(line, end="", flush=True)
        current = swap_used_bytes()
        peak = max(peak, current)
        if current > initial:
            status = "FAIL_SWAP_GROWTH"
            terminate(process)
            break
        time.sleep(0.25)
    remainder = process.stdout.read() or ""
    if remainder:
        lines.append(remainder)
        print(remainder, end="", flush=True)
    returncode = process.wait()
    ending = swap_used_bytes()
    peak = max(peak, ending)
    if peak > initial:
        status = "FAIL_SWAP_GROWTH"
    elif status == "PASS" and returncode != 0:
        status = "FAIL_COMMAND"
    payload = {
        "schema": "celltrack-alpha.dinov2-local-resource-guard.v2",
        "status": status,
        "command": command,
        "returncode": returncode,
        "captured_output": "".join(lines),
        "elapsed_seconds": time.time() - started,
        "swap_start_mib": initial / 2**20,
        "swap_peak_mib": peak / 2**20,
        "swap_end_mib": ending / 2**20,
        "swap_growth_mib": (peak - initial) / 2**20,
        "peak_child_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30,
        "forced_stop_on_swap_growth": status == "FAIL_SWAP_GROWTH",
        "preflight_path": str(preflight_path.resolve()),
        "external_action": False,
        "gpu_action": False,
        "kaggle_action": False,
        "runpod_action": False,
        "submission_action": False,
    }
    write_once(output, payload)
    return 0 if status == "PASS" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    idle = sub.add_parser("preflight")
    idle.add_argument("--output", type=Path, required=True)
    idle.add_argument("--seconds", type=float, default=45)
    guarded = sub.add_parser("run")
    guarded.add_argument("--preflight", type=Path, required=True)
    guarded.add_argument("--output", type=Path, required=True)
    guarded.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "preflight":
        return preflight(args.output, args.seconds)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise GuardError("missing command")
    return run_guarded(args.preflight, args.output, command)


if __name__ == "__main__":
    raise SystemExit(main())
