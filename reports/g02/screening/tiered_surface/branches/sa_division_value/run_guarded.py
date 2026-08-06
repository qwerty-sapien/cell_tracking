#!/usr/bin/env python3
"""Run the exact audit with wall/RSS/swap fail-closed monitoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def swap_mib() -> float:
    text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    if match is None:
        raise RuntimeError(f"cannot parse swap usage: {text!r}")
    value = float(match.group(1))
    return value * 1024.0 if match.group(2) == "G" else value


def rss_mib(pid: int) -> float:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )
    text = result.stdout.strip()
    return float(text) / 1024.0 if text else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-minutes", type=float, default=90.0)
    parser.add_argument("--rss-gib", type=float, default=8.0)
    args = parser.parse_args()
    started = time.time()
    initial_swap = swap_mib()
    process = subprocess.Popen(
        [str(args.python), str(args.script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_rss = 0.0
    reason = None
    while process.poll() is None:
        peak_rss = max(peak_rss, rss_mib(process.pid))
        swap_growth = swap_mib() - initial_swap
        if swap_growth > 0.0:
            reason = f"swap grew by {swap_growth:.3f} MiB"
        elif peak_rss > args.rss_gib * 1024.0:
            reason = f"RSS exceeded {args.rss_gib:.3f} GiB"
        elif time.time() - started > args.wall_minutes * 60.0:
            reason = f"wall time exceeded {args.wall_minutes:.3f} minutes"
        if reason is not None:
            process.terminate()
            break
        time.sleep(1)
    stdout, stderr = process.communicate()
    final_swap = swap_mib()
    result = {
        "schema": "celltrack-alpha.guarded-execution.v1",
        "command": [str(args.python), str(args.script)],
        "elapsed_seconds": time.time() - started,
        "exit_code": process.returncode,
        "peak_process_rss_mib": peak_rss,
        "swap_before_mib": initial_swap,
        "swap_after_mib": final_swap,
        "swap_growth_mib": final_swap - initial_swap,
        "wall_limit_minutes": args.wall_minutes,
        "rss_limit_gib": args.rss_gib,
        "termination_reason": reason,
        "stdout": stdout,
        "stderr": stderr,
        "passed": process.returncode == 0 and reason is None and final_swap <= initial_swap,
        "cpu_only": True,
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    if not result["passed"]:
        sys.stderr.write(stderr)
        raise SystemExit(reason or f"audit exited {process.returncode}")
    sys.stdout.write(stdout)


if __name__ == "__main__":
    main()
