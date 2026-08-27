#!/usr/bin/env python3
"""Fresh idle preflight and continuous zero-swap guard for local DINOv2 R5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time


class GuardError(RuntimeError):
    pass


def swap_used_mib() -> float:
    text = subprocess.check_output(["vm_stat"], text=True)
    page = re.search(r"page size of ([0-9]+) bytes", text)
    swap = re.search(r"^Swapouts:\s+([0-9]+)\.", text, re.MULTILINE)
    if not page or not swap:
        raise GuardError("cannot parse vm_stat swapouts")
    return int(page.group(1)) * int(swap.group(1)) / 2**20


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def preflight(output: Path, seconds: float) -> int:
    if seconds < 45:
        raise GuardError("idle preflight must be at least 45 seconds")
    started = time.perf_counter()
    initial = swap_used_mib()
    peak = initial
    while time.perf_counter() - started < seconds:
        peak = max(peak, swap_used_mib())
        time.sleep(0.25)
    ending = swap_used_mib()
    growth = max(peak, ending) - initial
    result = {
        "schema": "g02.sa-dinov2-r5.idle-preflight.v1",
        "status": "PASS" if growth <= 0 else "FAIL_SWAP_GROWTH",
        "observed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "required_seconds": seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "swap_start_mib": initial,
        "swap_peak_mib": peak,
        "swap_end_mib": ending,
        "swap_growth_mib": growth,
        "local_accelerator": "MPS",
        "kaggle": False,
        "runpod": False,
    }
    write_once(output, result)
    return 0 if result["status"] == "PASS" else 2


def guarded(
    output: Path,
    preflight_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    quarantine: Path,
    command: list[str],
) -> int:
    preflight_result = json.loads(preflight_path.read_text())
    if (
        preflight_result.get("status") != "PASS"
        or float(preflight_result.get("elapsed_seconds", 0)) < 45
        or float(preflight_result.get("swap_growth_mib", 1)) != 0
    ):
        raise GuardError("fresh idle preflight did not pass")
    for path in (output, stdout_path, stderr_path):
        if path.exists():
            raise GuardError(f"refusing to overwrite {path}")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    initial = swap_used_mib()
    peak = initial
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        forced = False
        while process.poll() is None:
            peak = max(peak, swap_used_mib())
            if peak > initial:
                forced = True
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(0.25)
        returncode = process.wait()
    ending = swap_used_mib()
    growth = max(peak, ending) - initial
    status = "PASS" if returncode == 0 and growth <= 0 and not forced else "FAIL"
    quarantined_path = None
    if status != "PASS" and quarantine.exists():
        quarantined_path = quarantine.with_name(
            quarantine.name
            + ".quarantine."
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        if quarantined_path.exists():
            raise GuardError(f"quarantine target exists: {quarantined_path}")
        shutil.move(str(quarantine), str(quarantined_path))
    result = {
        "schema": "g02.sa-dinov2-r5.resource-guard.v1",
        "status": status,
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_child_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        / 2**30,
        "swap_start_mib": initial,
        "swap_peak_mib": peak,
        "swap_end_mib": ending,
        "swap_growth_mib": growth,
        "forced_stop_on_swap_growth": forced,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "quarantined_partial_output": str(quarantined_path)
        if quarantined_path is not None
        else None,
        "local_accelerator": "MPS",
        "kaggle": False,
        "runpod": False,
    }
    write_once(output, result)
    return 0 if status == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    idle = subparsers.add_parser("preflight")
    idle.add_argument("--output", type=Path, required=True)
    idle.add_argument("--seconds", type=float, default=45.0)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--stdout", type=Path, required=True)
    run.add_argument("--stderr", type=Path, required=True)
    run.add_argument("--quarantine", type=Path, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    combined = subparsers.add_parser("preflight-run")
    combined.add_argument("--preflight-output", type=Path, required=True)
    combined.add_argument("--seconds", type=float, default=45.0)
    combined.add_argument("--output", type=Path, required=True)
    combined.add_argument("--stdout", type=Path, required=True)
    combined.add_argument("--stderr", type=Path, required=True)
    combined.add_argument("--quarantine", type=Path, required=True)
    combined.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "preflight":
        return preflight(args.output, args.seconds)
    if args.mode == "preflight-run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise GuardError("missing guarded command")
        preflight_status = preflight(args.preflight_output, args.seconds)
        if preflight_status != 0:
            return preflight_status
        return guarded(
            args.output,
            args.preflight_output,
            args.stdout,
            args.stderr,
            args.quarantine,
            command,
        )
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise GuardError("missing guarded command")
    return guarded(
        args.output,
        args.preflight,
        args.stdout,
        args.stderr,
        args.quarantine,
        command,
    )


if __name__ == "__main__":
    raise SystemExit(main())
