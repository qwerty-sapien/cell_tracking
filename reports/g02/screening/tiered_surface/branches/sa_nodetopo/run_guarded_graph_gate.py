#!/usr/bin/env python3
"""Run the NODETOPO 44b6 evaluator under the binding local resource guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


MAX_SECONDS = 2 * 60 * 60


def _terminate(process: subprocess.Popen[str]) -> None:
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--branch-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--guard-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.guard_out.exists() or args.output.exists():
        raise SystemExit("refusing to overwrite guarded outputs")
    preflight = json.loads(args.preflight.read_text(encoding="ascii"))
    if (
        preflight.get("status") != "PASS_ZERO_SWAP_GROWTH"
        or preflight.get("swap_growth_mib") != 0.0
        or float(preflight.get("elapsed_seconds", 0.0)) < 45.0
        or time.time() - args.preflight.stat().st_mtime > 180
    ):
        raise SystemExit("fresh 45-second zero-swap preflight did not pass")

    root = args.root.resolve()
    branch = args.branch_dir.resolve()
    sys.path.insert(0, str(root / "scripts"))
    import run_local_training_guard as resources  # type: ignore

    command = [
        sys.executable,
        str(branch / "evaluate_nodetopo_44b6_graph_gate.py"),
        "--output",
        str(args.output.resolve()),
    ]
    env = dict(os.environ)
    env["CELLTRACK_GUARD_ACTIVE"] = "1"
    env["CELLTRACK_HOLDOUT_ACCESS"] = "0"
    initial_swap = resources.macos_swap_used_bytes()
    peak_swap = initial_swap
    peak_rss = 0
    minimum_free = resources.macos_memory_free_percent()
    started = time.time()
    status = "PASS"
    process = subprocess.Popen(
        command,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    while process.poll() is None:
        current_swap = resources.macos_swap_used_bytes()
        peak_swap = max(peak_swap, current_swap)
        peak_rss = max(
            peak_rss, resources.process_tree_rss_bytes(process.pid)
        )
        minimum_free = min(
            minimum_free, resources.macos_memory_free_percent()
        )
        if current_swap > initial_swap:
            status = "FAIL_SWAP_GROWTH"
            _terminate(process)
            break
        if time.time() - started >= MAX_SECONDS:
            status = "FAIL_WALL_CLOCK"
            _terminate(process)
            break
        time.sleep(1.0)
    output, _ = process.communicate(timeout=30)
    ending_swap = resources.macos_swap_used_bytes()
    peak_swap = max(peak_swap, ending_swap)
    if peak_swap > initial_swap:
        status = "FAIL_SWAP_GROWTH"
    elif status == "PASS" and process.returncode != 0:
        status = "FAIL_COMMAND"
    guard = {
        "schema": "celltrack-alpha.sa-nodetopo-44b6-guard.v1",
        "status": status,
        "command": command,
        "returncode": process.returncode,
        "captured_output": output,
        "preflight_path": str(args.preflight.resolve()),
        "preflight_sha256": hashlib.sha256(
            args.preflight.read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial_swap / 1024**2,
        "peak_swap_mib": peak_swap / 1024**2,
        "ending_swap_mib": ending_swap / 1024**2,
        "swap_growth_mib": (peak_swap - initial_swap) / 1024**2,
        "peak_process_tree_rss_gib": peak_rss / 1024**3,
        "minimum_system_memory_free_percent": minimum_free,
        "wall_clock_limit_seconds": MAX_SECONDS,
        "external_action": False,
        "gpu_action": False,
        "kaggle_action": False,
        "runpod_action": False,
        "submission_action": False,
    }
    args.guard_out.write_text(
        json.dumps(guard, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(output, end="")
    print(
        json.dumps(
            {
                "status": status,
                "swap_growth_mib": guard["swap_growth_mib"],
                "peak_process_tree_rss_gib": guard[
                    "peak_process_tree_rss_gib"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
