#!/usr/bin/env python3
"""Run the oracle under the 45-second idle, swap, RSS, and six-hour guards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
sys.path.insert(0, str(ROOT / "scripts"))
import run_local_training_guard as resources  # noqa: E402


PREFLIGHT_SECONDS = 45.0
POLL_SECONDS = 0.25
SUSTAINED_THRESHOLD = 256 * 1024**2
IMMEDIATE_THRESHOLD = 384 * 1024**2
SUSTAINED_SECONDS = 15.0
WALL_SECONDS = 6 * 60 * 60


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    os.replace(temporary, path)


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def preflight() -> dict[str, object]:
    started = time.time()
    initial = resources.macos_swap_used_bytes()
    peak = initial
    above_since: float | None = None
    status = "PASS"
    minimum_free = resources.macos_memory_free_percent()
    while time.time() - started < PREFLIGHT_SECONDS:
        current = resources.macos_swap_used_bytes()
        peak = max(peak, current)
        growth = max(0, current - initial)
        minimum_free = min(minimum_free, resources.macos_memory_free_percent())
        if growth >= IMMEDIATE_THRESHOLD:
            status = "ESCALATE_SWAPOUT_IMMEDIATE"
            break
        if growth >= SUSTAINED_THRESHOLD:
            above_since = above_since or time.time()
            if time.time() - above_since >= SUSTAINED_SECONDS:
                status = "ESCALATE_SWAPOUT_SUSTAINED"
                break
        else:
            above_since = None
        time.sleep(POLL_SECONDS)
    ending = resources.macos_swap_used_bytes()
    peak = max(peak, ending)
    return {
        "status": status,
        "required_seconds": PREFLIGHT_SECONDS,
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial / 1024**2,
        "peak_swap_mib": peak / 1024**2,
        "ending_swap_mib": ending / 1024**2,
        "swap_growth_mib": max(0, peak - initial) / 1024**2,
        "minimum_system_memory_free_percent": minimum_free,
    }


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    guard_path = RUNS / "N0_RUN_GUARD.json"
    if guard_path.exists():
        raise SystemExit("refusing to overwrite run guard")
    idle = preflight()
    atomic_json(RUNS / "N0_IDLE_PREFLIGHT.json", idle)
    if idle["status"] != "PASS":
        atomic_json(
            guard_path,
            {
                "schema": "celltrack-alpha.global-reassign-oracle-guard.v1",
                "status": idle["status"],
                "preflight_elapsed_seconds": idle["elapsed_seconds"],
                "elapsed_seconds": idle["elapsed_seconds"],
                "peak_process_tree_rss_gib": 0.0,
                "swap_growth_mib": idle["swap_growth_mib"],
                "local_cpu_only": True,
            },
        )
        return 3

    environment = dict(os.environ)
    environment["GLOBAL_REASSIGN_GUARD_ACTIVE"] = "1"
    environment["CELLTRACK_HOLDOUT_ACCESS"] = "0"
    command = [sys.executable, str(HERE / "global_reassign_oracle.py")]
    initial_swap = resources.macos_swap_used_bytes()
    peak_swap = initial_swap
    peak_rss = 0
    above_since: float | None = None
    started = time.time()
    status = "PASS"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    captured: list[bytes] = []
    while process.poll() is None:
        try:
            chunk = os.read(process.stdout.fileno(), 65536)
        except BlockingIOError:
            chunk = b""
        if chunk:
            captured.append(chunk)
            print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
        current_swap = resources.macos_swap_used_bytes()
        peak_swap = max(peak_swap, current_swap)
        peak_rss = max(peak_rss, resources.process_tree_rss_bytes(process.pid))
        growth = max(0, current_swap - initial_swap)
        if growth >= IMMEDIATE_THRESHOLD:
            status = "ESCALATE_SWAPOUT_IMMEDIATE"
            stop(process)
            break
        if growth >= SUSTAINED_THRESHOLD:
            above_since = above_since or time.time()
            if time.time() - above_since >= SUSTAINED_SECONDS:
                status = "ESCALATE_SWAPOUT_SUSTAINED"
                stop(process)
                break
        else:
            above_since = None
        if time.time() - started >= WALL_SECONDS:
            status = "ESCALATE_WALL_CLOCK_CAP"
            stop(process)
            break
        time.sleep(POLL_SECONDS)
    remainder, _ = process.communicate(timeout=30)
    if remainder:
        captured.append(remainder)
        print(remainder.decode("utf-8", errors="replace"), end="")
    ending_swap = resources.macos_swap_used_bytes()
    peak_swap = max(peak_swap, ending_swap)
    if status == "PASS" and process.returncode != 0:
        status = "INVALID_HARNESS_COMMAND"
    guard = {
        "schema": "celltrack-alpha.global-reassign-oracle-guard.v1",
        "status": status,
        "command": command,
        "returncode": process.returncode,
        "captured_output": b"".join(captured).decode("utf-8", errors="replace"),
        "preflight_elapsed_seconds": idle["elapsed_seconds"],
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial_swap / 1024**2,
        "peak_swap_mib": peak_swap / 1024**2,
        "ending_swap_mib": ending_swap / 1024**2,
        "swap_growth_mib": max(0, peak_swap - initial_swap) / 1024**2,
        "peak_process_tree_rss_gib": peak_rss / 1024**3,
        "wall_clock_limit_seconds": WALL_SECONDS,
        "sustained_swap_threshold_mib": 256,
        "sustained_seconds": SUSTAINED_SECONDS,
        "immediate_swap_threshold_mib": 384,
        "local_cpu_only": True,
        "external_action": False,
        "gpu_action": False,
        "kaggle_action": False,
        "runpod_action": False,
        "submission_action": False,
    }
    atomic_json(guard_path, guard)
    if (RUNS / "N0_GLOBAL_REASSIGN_ORACLE.json").is_file():
        subprocess.run(
            [sys.executable, str(HERE / "global_reassign_oracle.py"), "--finalize-guard"],
            cwd=ROOT,
            check=True,
        )
    print(
        json.dumps(
            {
                "status": status,
                "elapsed_seconds": guard["elapsed_seconds"],
                "peak_process_tree_rss_gib": guard["peak_process_tree_rss_gib"],
                "swap_growth_mib": guard["swap_growth_mib"],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
