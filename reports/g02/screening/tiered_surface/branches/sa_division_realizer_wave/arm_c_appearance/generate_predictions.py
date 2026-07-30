#!/usr/bin/env python3
"""Fail-closed Arm-C prediction boundary; this CLI accepts no target path."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_AUDIT = OUTPUT_DIR / "CACHE_AUDIT.json"
PREDICTIONS = OUTPUT_DIR / "PREDICTIONS.json"
PASS = "PASS_ADEQUATE_FROZEN_INTENSITY_CACHE"


def prediction_gate() -> str:
    if not CACHE_AUDIT.is_file():
        return "FAIL_CLOSED_CACHE_AUDIT_MISSING"
    value = json.loads(CACHE_AUDIT.read_text(encoding="ascii"))
    verdict = str(value.get("verdict", ""))
    if verdict != PASS:
        return "FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE"
    return "READY_FOR_IMPLEMENTED_PREDICTION_RULE"


def main() -> None:
    verdict = prediction_gate()
    if verdict != "READY_FOR_IMPLEMENTED_PREDICTION_RULE":
        if PREDICTIONS.exists():
            raise RuntimeError("stale PREDICTIONS.json exists at a closed gate")
        raise SystemExit(verdict)
    raise SystemExit(
        "FAIL_CLOSED_UNIMPLEMENTED: an adequate cache was not preregistered"
    )


if __name__ == "__main__":
    main()
