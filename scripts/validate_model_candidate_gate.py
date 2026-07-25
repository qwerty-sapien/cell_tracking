#!/usr/bin/env python3
"""Validate a literature-first lightweight model candidate before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "celltrack_alpha"))

from model_candidate_gate import (  # noqa: E402
    CandidateGateError,
    load_json_object,
    validate_candidate,
)


DEFAULT_POLICY = ROOT / "reports" / "g02" / "compute_governance_v1.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = validate_candidate(
            load_json_object(args.candidate),
            load_json_object(args.policy),
        )
    except CandidateGateError as exc:
        print(json.dumps({"decision": "FAIL", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
