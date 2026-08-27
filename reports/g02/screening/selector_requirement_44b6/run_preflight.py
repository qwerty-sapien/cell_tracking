#!/usr/bin/env python3
"""Run the mandatory 45-second idle zero-swap preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path("/Users/agent/Desktop/cell_tracking")
sys.path.insert(0, str(ROOT / "scripts"))
import run_local_training_guard as resources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite preflight")
    started = time.time()
    initial = resources.macos_swap_used_bytes()
    peak = initial
    minimum_free = resources.macos_memory_free_percent()
    while time.time() - started < 45.0:
        peak = max(peak, resources.macos_swap_used_bytes())
        minimum_free = min(minimum_free, resources.macos_memory_free_percent())
        time.sleep(1.0)
    ending = resources.macos_swap_used_bytes()
    peak = max(peak, ending)
    growth = peak - initial
    payload = {
        "schema": "celltrack-alpha.selector-requirement-preflight.v1",
        "status": "PASS_ZERO_SWAP_GROWTH" if growth == 0 else "FAIL_SWAP_GROWTH",
        "required_seconds": 45.0,
        "elapsed_seconds": time.time() - started,
        "initial_swap_mib": initial / 1024**2,
        "peak_swap_mib": peak / 1024**2,
        "ending_swap_mib": ending / 1024**2,
        "swap_growth_mib": growth / 1024**2,
        "minimum_system_memory_free_percent": minimum_free,
        "external_action": False,
        "gpu_action": False,
        "kaggle_action": False,
        "runpod_action": False,
        "submission_action": False,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if growth == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
