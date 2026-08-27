#!/usr/bin/env python3
"""Branch-local 45-second idle preflight and zero-swap execution guard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Sequence


SWAP_RE = re.compile(r"\bused\s*=\s*([0-9.]+)([KMG])\b", re.IGNORECASE)
FREE_RE = re.compile(r"System-wide memory free percentage:\s*([0-9.]+)%")
MIB = 1024**2
GIB = 1024**3


class GuardError(RuntimeError):
    """Resource state cannot be proved safe."""


def swap_used() -> int:
    result = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = SWAP_RE.search(result.stdout)
    if match is None:
        raise GuardError("cannot parse vm.swapusage")
    multiplier = {"K": 1024, "M": MIB, "G": GIB}[match.group(2).upper()]
    return int(float(match.group(1)) * multiplier)


def free_percent() -> float:
    result = subprocess.run(
        ["memory_pressure", "-Q"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = FREE_RE.search(result.stdout)
    if match is None:
        raise GuardError("cannot parse memory_pressure")
    return float(match.group(1))


def process_rss(root_pid: int) -> int:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, rss_kib = (int(value) for value in fields)
        except ValueError:
            continue
        rows[pid] = (ppid, rss_kib)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _rss) in rows.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rows[pid][1] * 1024 for pid in descendants if pid in rows)


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise GuardError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def preflight(seconds: float, poll_seconds: float, report: Path) -> int:
    if seconds < 45.0 or poll_seconds <= 0:
        raise GuardError("preflight requires at least 45 seconds")
    initial = swap_used()
    peak = initial
    minimum_free = free_percent()
    started = time.time()
    while time.time() - started < seconds:
        current = swap_used()
        peak = max(peak, current)
        minimum_free = min(minimum_free, free_percent())
        if current > initial:
            break
        time.sleep(poll_seconds)
    ending = swap_used()
    elapsed = time.time() - started
    status = (
        "PASS_ZERO_SWAP_GROWTH"
        if peak == initial and elapsed >= 45.0
        else "FAIL_CLOSED"
    )
    payload = {
        "schema": "celltrack-alpha.sa-delete-preflight.v1",
        "status": status,
        "required_seconds": 45.0,
        "elapsed_seconds": elapsed,
        "initial_swap_mib": initial / MIB,
        "peak_swap_mib": peak / MIB,
        "ending_swap_mib": ending / MIB,
        "swap_growth_mib": (peak - initial) / MIB,
        "minimum_system_memory_free_percent": minimum_free,
        "external_action": False,
        "gpu_action": False,
    }
    write_once(report, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if status.startswith("PASS") else 3


def stop(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def guarded_run(
    command: Sequence[str],
    report: Path,
    *,
    poll_seconds: float,
    maximum_rss_gib: float,
    maximum_seconds: float,
) -> int:
    if not command or poll_seconds <= 0 or maximum_rss_gib <= 0 or maximum_seconds <= 0:
        raise GuardError("invalid guarded-run arguments")
    initial = swap_used()
    peak = initial
    minimum_free = free_percent()
    process = subprocess.Popen(list(command), start_new_session=True)
    started = time.time()
    peak_rss = 0
    stop_reason = "child_failed"
    try:
        while True:
            return_code = process.poll()
            current = swap_used()
            peak = max(peak, current)
            peak_rss = max(peak_rss, process_rss(process.pid))
            minimum_free = min(minimum_free, free_percent())
            if current > initial:
                stop_reason = "swap_growth"
                stop(process)
                break
            if peak_rss > maximum_rss_gib * GIB:
                stop_reason = "rss_limit"
                stop(process)
                break
            if time.time() - started > maximum_seconds:
                stop_reason = "wall_limit"
                stop(process)
                break
            if return_code is not None:
                stop_reason = "completed" if return_code == 0 else "child_failed"
                break
            time.sleep(poll_seconds)
    finally:
        stop(process)
    ending = swap_used()
    status = (
        "PASS"
        if stop_reason == "completed" and peak == initial
        else "FAIL_CLOSED"
    )
    payload = {
        "schema": "celltrack-alpha.sa-delete-run-guard.v1",
        "status": status,
        "stop_reason": stop_reason,
        "return_code": process.returncode,
        "elapsed_seconds": time.time() - started,
        "peak_process_tree_rss_gib": peak_rss / GIB,
        "maximum_rss_gib": maximum_rss_gib,
        "maximum_seconds": maximum_seconds,
        "initial_swap_mib": initial / MIB,
        "peak_swap_mib": peak / MIB,
        "ending_swap_mib": ending / MIB,
        "swap_growth_mib": (peak - initial) / MIB,
        "minimum_system_memory_free_percent": minimum_free,
        "command": list(command),
        "external_action": False,
        "gpu_action": False,
    }
    write_once(report, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if status == "PASS" else 3


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="mode", required=True)
    before = subparsers.add_parser("preflight")
    before.add_argument("--seconds", type=float, default=45.0)
    before.add_argument("--poll-seconds", type=float, default=1.0)
    before.add_argument("--report", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--poll-seconds", type=float, default=0.25)
    run.add_argument("--maximum-rss-gib", type=float, default=4.0)
    run.add_argument("--maximum-seconds", type=float, default=10800.0)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "preflight":
        return preflight(args.seconds, args.poll_seconds, args.report)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return guarded_run(
        command,
        args.report,
        poll_seconds=args.poll_seconds,
        maximum_rss_gib=args.maximum_rss_gib,
        maximum_seconds=args.maximum_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
