#!/usr/bin/env python3
"""Run one literature-approved M5 training stage within memory, swap, and time gates."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "celltrack_alpha"))

from model_candidate_gate import (  # noqa: E402
    CandidateGateError,
    load_json_object,
    validate_candidate,
)


DEFAULT_POLICY = ROOT / "reports" / "g02" / "compute_governance_v1.json"
DEFAULT_LOCK = ROOT / "reports" / "g02" / ".local_training_lane.lock"
GIB = 1024**3
MIB = 1024**2
SWAP_USED_RE = re.compile(r"\bused\s*=\s*([0-9.]+)([KMG])\b", re.IGNORECASE)
MEMORY_FREE_RE = re.compile(
    r"System-wide memory free percentage:\s*([0-9.]+)%",
    re.IGNORECASE,
)


class LocalTrainingGuardError(RuntimeError):
    """Raised when the local resource guard cannot safely launch."""


def parse_swap_used_bytes(text: str) -> int:
    match = SWAP_USED_RE.search(text)
    if match is None:
        raise LocalTrainingGuardError(f"cannot parse vm.swapusage: {text!r}")
    value = float(match.group(1))
    multiplier = {"K": 1024, "M": MIB, "G": GIB}[match.group(2).upper()]
    return int(value * multiplier)


def macos_swap_used_bytes() -> int:
    result = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_swap_used_bytes(result.stdout)


def macos_physical_memory_bytes() -> int:
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise LocalTrainingGuardError(
            f"cannot parse hw.memsize: {result.stdout!r}"
        ) from exc


def parse_memory_free_percent(text: str) -> float:
    match = MEMORY_FREE_RE.search(text)
    if match is None:
        raise LocalTrainingGuardError(f"cannot parse memory_pressure -Q: {text!r}")
    return float(match.group(1))


def macos_memory_free_percent() -> float:
    result = subprocess.run(
        ["memory_pressure", "-Q"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_memory_free_percent(result.stdout)


def process_tree_rss_bytes(root_pid: int) -> int:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[int, tuple[int, int]] = {}
    for raw_line in result.stdout.splitlines():
        fields = raw_line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        rows[pid] = (ppid, rss_kib)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in rows.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rows[pid][1] * 1024 for pid in descendants if pid in rows)


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a training command is required after --")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def run_guarded(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        raise LocalTrainingGuardError("the local M5 guard may run only on macOS")

    policy = load_json_object(args.policy)
    gate = validate_candidate(load_json_object(args.candidate), policy)
    memory_cap_bytes = int(float(gate["maximum_training_memory_gib"]) * GIB)
    reserve_bytes = int(float(gate["background_reserve_gib"]) * GIB)
    minimum_free_percent = float(gate["minimum_system_memory_free_percent"])
    wall_seconds = float(gate["max_wall_minutes"]) * 60.0
    physical_memory = macos_physical_memory_bytes()
    if physical_memory < memory_cap_bytes + reserve_bytes:
        raise LocalTrainingGuardError(
            "physical memory is smaller than the training cap plus required reserve"
        )
    initial_free_percent = macos_memory_free_percent()
    if initial_free_percent < minimum_free_percent:
        raise LocalTrainingGuardError(
            f"system memory free percentage {initial_free_percent:g}% is below "
            f"the required {minimum_free_percent:g}% background reserve"
        )

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LocalTrainingGuardError(
                "another guarded local training stage already owns the local lane"
            ) from exc

        initial_swap = macos_swap_used_bytes()
        environment = os.environ.copy()
        environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
        environment["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.9"
        environment["CELLTRACK_LOCAL_MEMORY_LIMIT_GIB"] = str(
            gate["maximum_training_memory_gib"]
        )
        environment["CELLTRACK_HOLDOUT_ACCESS"] = "0"
        started_at = time.time()
        process = subprocess.Popen(
            args.command,
            env=environment,
            start_new_session=True,
        )
        peak_rss = 0
        peak_swap = initial_swap
        minimum_observed_free_percent = initial_free_percent
        stop_reason = "completed"
        return_code: int | None = None
        try:
            while True:
                return_code = process.poll()
                rss = process_tree_rss_bytes(process.pid)
                swap = macos_swap_used_bytes()
                free_percent = macos_memory_free_percent()
                peak_rss = max(peak_rss, rss)
                peak_swap = max(peak_swap, swap)
                minimum_observed_free_percent = min(
                    minimum_observed_free_percent, free_percent
                )
                elapsed = time.time() - started_at
                if rss > memory_cap_bytes:
                    stop_reason = "memory_cap_exceeded"
                    terminate_process_group(process)
                    return_code = process.returncode
                    break
                if swap > initial_swap:
                    stop_reason = "swap_growth_detected"
                    terminate_process_group(process)
                    return_code = process.returncode
                    break
                if free_percent < minimum_free_percent:
                    stop_reason = "background_reserve_breached"
                    terminate_process_group(process)
                    return_code = process.returncode
                    break
                if elapsed >= wall_seconds:
                    stop_reason = "wall_time_reached"
                    terminate_process_group(process)
                    return_code = process.returncode
                    break
                if return_code is not None:
                    stop_reason = "completed" if return_code == 0 else "child_failed"
                    break
                time.sleep(args.poll_seconds)
        except BaseException:
            stop_reason = "guard_interrupted"
            terminate_process_group(process)
            raise
        finally:
            ended_at = time.time()
            report = {
                "schema": "celltrack-alpha.local-training-guard.v1",
                "candidate_gate": gate,
                "command": list(args.command),
                "started_unix": started_at,
                "ended_unix": ended_at,
                "elapsed_seconds": ended_at - started_at,
                "physical_memory_gib": physical_memory / GIB,
                "memory_cap_gib": memory_cap_bytes / GIB,
                "peak_process_tree_rss_gib": peak_rss / GIB,
                "initial_swap_mib": initial_swap / MIB,
                "peak_swap_mib": peak_swap / MIB,
                "swap_growth_mib": (peak_swap - initial_swap) / MIB,
                "initial_system_memory_free_percent": initial_free_percent,
                "minimum_system_memory_free_percent": minimum_observed_free_percent,
                "required_system_memory_free_percent": minimum_free_percent,
                "stop_reason": stop_reason,
                "return_code": return_code,
                "mps_high_watermark_ratio": 1.0,
                "mps_low_watermark_ratio": 0.9,
            }
            write_report(args.report, report)
    if stop_reason in {
        "memory_cap_exceeded",
        "swap_growth_detected",
        "background_reserve_breached",
        "guard_interrupted",
    }:
        return 3
    if stop_reason == "wall_time_reached":
        return 124
    return int(return_code or 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_guarded(args)
    except (CandidateGateError, LocalTrainingGuardError, OSError, subprocess.SubprocessError) as exc:
        print(f"local training guard refused launch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
