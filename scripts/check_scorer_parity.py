#!/usr/bin/env python3
"""Run deterministic scorer-parity fixtures for G-01."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.scorer_parity import write_scorer_parity_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/general_model_scorer_parity.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_scorer_parity_report(args.output)
    print(json.dumps({"overall_pass": report["overall_pass"], "output": str(args.output)}, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
