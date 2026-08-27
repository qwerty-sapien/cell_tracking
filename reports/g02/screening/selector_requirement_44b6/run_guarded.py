#!/usr/bin/env python3
"""Run the selector diagnostic under the binding zero-swap CPU guard."""

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


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
MAX_SECONDS = 5 * 60 * 60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminate(process: subprocess.Popen[bytes]) -> None:
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--guard-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.guard_output.exists() or args.output.exists():
        raise SystemExit("refusing to overwrite guarded outputs")
    preflight = json.loads(args.preflight.read_text(encoding="ascii"))
    if (
        preflight.get("status") != "PASS_ZERO_SWAP_GROWTH"
        or preflight.get("swap_growth_mib") != 0.0
        or float(preflight.get("elapsed_seconds", 0.0)) < 45.0
        or time.time() - args.preflight.stat().st_mtime > 180
    ):
        raise SystemExit("fresh 45-second zero-swap preflight did not pass")
    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise SystemExit(f"pre-outcome source identity drifted: {relative}")

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_local_training_guard as resources  # type: ignore

    command = [
        sys.executable,
        str(HERE / "selector_requirement.py"),
        "--output",
        str(args.output.resolve()),
    ]
    environment = dict(os.environ)
    environment["CELLTRACK_GUARD_ACTIVE"] = "1"
    environment["CELLTRACK_HOLDOUT_ACCESS"] = "0"
    initial_swap = resources.macos_swap_used_bytes()
    peak_swap = initial_swap
    peak_rss = 0
    minimum_free = resources.macos_memory_free_percent()
    started = time.time()
    status = "PASS"
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if process.stdout is not None:
        os.set_blocking(process.stdout.fileno(), False)
    chunks: list[bytes] = []
    while process.poll() is None:
        if process.stdout is not None:
            try:
                available = os.read(process.stdout.fileno(), 65536)
            except BlockingIOError:
                available = b""
            if available:
                chunks.append(available)
                print(available.decode("utf-8", errors="replace"), end="", flush=True)
        current_swap = resources.macos_swap_used_bytes()
        peak_swap = max(peak_swap, current_swap)
        peak_rss = max(peak_rss, resources.process_tree_rss_bytes(process.pid))
        minimum_free = min(minimum_free, resources.macos_memory_free_percent())
        if current_swap > initial_swap:
            status = "FAIL_SWAP_GROWTH"
            terminate(process)
            break
        if time.time() - started >= MAX_SECONDS:
            status = "FAIL_WALL_CLOCK"
            terminate(process)
            break
        time.sleep(1.0)
    remainder, _ = process.communicate(timeout=30)
    if remainder:
        chunks.append(remainder)
        print(remainder.decode("utf-8", errors="replace"), end="")
    ending_swap = resources.macos_swap_used_bytes()
    peak_swap = max(peak_swap, ending_swap)
    if peak_swap > initial_swap:
        status = "FAIL_SWAP_GROWTH"
    elif status == "PASS" and process.returncode != 0:
        status = "FAIL_COMMAND"
    payload = {
        "schema": "celltrack-alpha.selector-requirement-guard.v1",
        "status": status,
        "command": command,
        "returncode": process.returncode,
        "captured_output": b"".join(chunks).decode("utf-8", errors="replace"),
        "preflight_path": str(args.preflight.resolve()),
        "preflight_sha256": sha256(args.preflight),
        "execution_manifest_path": str(args.manifest.resolve()),
        "execution_manifest_sha256": sha256(args.manifest),
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
    args.guard_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "status": status,
                "swap_growth_mib": payload["swap_growth_mib"],
                "peak_process_tree_rss_gib": payload[
                    "peak_process_tree_rss_gib"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
