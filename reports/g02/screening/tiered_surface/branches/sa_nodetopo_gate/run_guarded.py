#!/usr/bin/env python3
"""Run the 44b6 graph gate under continuous zero-swap and wall-clock guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_local_training_guard as resources  # type: ignore  # noqa: E402


def terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.guard_report.exists():
        raise SystemExit("refusing to overwrite guarded outputs")
    preflight = json.loads(args.preflight.read_text(encoding="ascii"))
    if (
        preflight.get("status") != "PASS_ZERO_SWAP_GROWTH"
        or float(preflight.get("swap_growth_mib", -1)) != 0.0
        or float(preflight.get("elapsed_seconds", 0)) < 45.0
    ):
        raise SystemExit("binding preflight did not pass")
    if time.time() - args.preflight.stat().st_mtime > 180:
        raise SystemExit("preflight is stale")
    command = [
        str(ROOT / ".venv313/bin/python"),
        str(HERE / "evaluate_graph_gate.py"),
        "--output",
        str(args.output.resolve()),
    ]
    env = dict(os.environ)
    env["CELLTRACK_GUARD_ACTIVE"] = "1"
    env["CELLTRACK_HOLDOUT_ACCESS"] = "0"
    env["CELLTRACK_CV57_ACCESS"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    initial_swap = resources.macos_swap_used_bytes()
    peak_swap = initial_swap
    peak_rss = 0
    minimum_free = resources.macos_memory_free_percent()
    started = time.time()
    status = "PASS"
    process = subprocess.Popen(
        command,
        cwd=str(HERE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    captured: list[str] = []
    assert process.stdout is not None
    while process.poll() is None:
        readable, _, _ = select.select([process.stdout], [], [], 0.25)
        if readable:
            line = process.stdout.readline()
            if line:
                captured.append(line)
                print(line, end="", flush=True)
        current_swap = resources.macos_swap_used_bytes()
        peak_swap = max(peak_swap, current_swap)
        peak_rss = max(peak_rss, resources.process_tree_rss_bytes(process.pid))
        minimum_free = min(minimum_free, resources.macos_memory_free_percent())
        if current_swap > initial_swap:
            status = "FAIL_SWAP_GROWTH"
            terminate(process)
            break
        if time.time() - started > 7200:
            status = "FAIL_WALL_CLOCK"
            terminate(process)
            break
    remainder = process.stdout.read()
    if remainder:
        captured.append(remainder)
        print(remainder, end="")
    process.wait()
    ending_swap = resources.macos_swap_used_bytes()
    peak_swap = max(peak_swap, ending_swap)
    if peak_swap > initial_swap:
        status = "FAIL_SWAP_GROWTH"
    elif status == "PASS" and process.returncode != 0:
        status = "FAIL_COMMAND"
    report = {
        "schema": "celltrack-alpha.sa-nodetopo-gate-guard.v1",
        "status": status,
        "command": command,
        "returncode": process.returncode,
        "captured_output": "".join(captured),
        "preflight_path": str(args.preflight.resolve()),
        "preflight_sha256": hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial_swap / 1024**2,
        "peak_swap_mib": peak_swap / 1024**2,
        "ending_swap_mib": ending_swap / 1024**2,
        "swap_growth_mib": (peak_swap - initial_swap) / 1024**2,
        "peak_process_tree_rss_gib": peak_rss / 1024**3,
        "minimum_system_memory_free_percent": minimum_free,
        "wall_clock_limit_seconds": 7200,
        "external_action": False,
        "gpu_action": False,
        "kaggle_action": False,
        "budget_debit": False,
        "runpod_action": False,
        "submission_action": False,
    }
    args.guard_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": status, "swap_growth_mib": report["swap_growth_mib"]}))
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
