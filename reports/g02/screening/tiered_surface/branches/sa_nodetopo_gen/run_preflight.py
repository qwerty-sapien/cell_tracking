#!/usr/bin/env python3
"""Run the binding 45-second idle/swap preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path("/Users/agent/Desktop/cell_tracking")
sys.path.insert(0, str(ROOT / "scripts"))
import run_local_training_guard as resources  # type: ignore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite preflight output")
    started = time.time()
    initial_swap = resources.macos_swap_used_bytes()
    peak_swap = initial_swap
    minimum_free = resources.macos_memory_free_percent()
    while time.time() - started < 45.0:
        time.sleep(0.5)
        peak_swap = max(peak_swap, resources.macos_swap_used_bytes())
        minimum_free = min(minimum_free, resources.macos_memory_free_percent())
    ending_swap = resources.macos_swap_used_bytes()
    peak_swap = max(peak_swap, ending_swap)
    growth = peak_swap - initial_swap
    result = {
        "schema": "celltrack-alpha.blockpeak-idle-preflight.v1",
        "status": "PASS_ZERO_SWAP_GROWTH" if growth == 0 else "FAIL_SWAP_GROWTH",
        "required_seconds": 45.0,
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial_swap / 1024**2,
        "peak_swap_mib": peak_swap / 1024**2,
        "ending_swap_mib": ending_swap / 1024**2,
        "swap_growth_mib": growth / 1024**2,
        "minimum_system_memory_free_percent": minimum_free,
        "external_action": False,
        "gpu_action": False,
        "kaggle_action": False,
        "runpod_action": False,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if growth == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
