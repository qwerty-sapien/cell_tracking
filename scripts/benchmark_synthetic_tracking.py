#!/usr/bin/env python3
"""Run the deterministic fixed-detection 3D+t association benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.synthetic import assert_synthetic_benchmark, run_synthetic_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output. The benchmark always prints the same payload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_synthetic_benchmark(args.repetitions)
    assert_synthetic_benchmark(result)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
