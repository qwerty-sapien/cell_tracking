#!/usr/bin/env python3
"""Finalize an unevaluated, fail-closed Arm-C result without target access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path(__file__).resolve().parent
PREREG = OUTPUT_DIR / "PREREGISTRATION.json"
CACHE_AUDIT = OUTPUT_DIR / "CACHE_AUDIT.json"
PREFLIGHT = OUTPUT_DIR / "PREFLIGHT.json"
GUARD = OUTPUT_DIR / "GUARD.json"
PREDICTIONS = OUTPUT_DIR / "PREDICTIONS.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON root: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def main() -> None:
    prereg = load(PREREG)
    audit = load(CACHE_AUDIT)
    preflight = load(PREFLIGHT)
    guard = load(GUARD)
    if audit.get("verdict") != "FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE":
        raise RuntimeError("unexpected cache-audit verdict")
    if not preflight.get("passed") or not guard.get("passed"):
        raise RuntimeError("resource guard did not pass")
    if PREDICTIONS.exists():
        raise RuntimeError("PREDICTIONS.json must not exist at a closed cache gate")
    result = {
        "schema": "celltrack-alpha.division-realizer-arm-result.v1",
        "arm": "C",
        "name": prereg["name"],
        "status": "FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE",
        "scientific_verdict": None,
        "evaluated": False,
        "reason": (
            "No pre-existing frozen cache both covers all 21 held-44b6 "
            "identities and exposes physical intensity redistribution summaries."
        ),
        "frozen_receipts": {
            "preregistration_path": "PREREGISTRATION.json",
            "preregistration_sha256": sha256(PREREG),
            "cache_audit_path": "CACHE_AUDIT.json",
            "cache_audit_sha256": sha256(CACHE_AUDIT),
            "preflight_path": "PREFLIGHT.json",
            "preflight_sha256": sha256(PREFLIGHT),
            "guard_path": "GUARD.json",
            "guard_sha256": sha256(GUARD),
        },
        "universe": {
            "family": "44b6",
            "fragments": 21,
            "selected_forks": 0,
            "prediction_file_exists": False,
            "six_bba_read": False,
        },
        "resource_guard": {
            "preflight_seconds": preflight["elapsed_seconds"],
            "preflight_swap_growth_mib": preflight["swap_growth_mib"],
            "audit_elapsed_seconds": guard["elapsed_seconds"],
            "audit_peak_process_rss_mib": guard["peak_process_rss_mib"],
            "audit_swap_growth_mib": guard["swap_growth_mib"],
            "cpu_only": True,
        },
        "duplicate_survivors": 0,
        "metrics": {
            "division_tp": None,
            "division_fp": None,
            "division_fn": None,
            "division_jaccard": None,
            "adjusted_edge_jaccard_delta": None,
            "raw_edge_jaccard_delta": None,
            "node_recall_delta": None,
            "combined_score_delta": None,
        },
        "pass": False,
        "pass_rule": (
            "combined delta > 0, division Jaccard >= 0.30, nonnegative "
            "adjusted/raw/node deltas, and zero duplicates"
        ),
        "fp_sensitivity": {
            "25_tp_0_fp": 0.05952380952380952,
            "25_tp_5_fp": 0.05319148936170213,
            "25_tp_10_fp": 0.04807692307692308,
        },
        "information_firewall": audit["prediction_stage"],
        "forbidden_actions": {
            "raw_archive": False,
            "raw_images": False,
            "cache_created_or_extended": False,
            "target": False,
            "oracle": False,
            "repair_bundles": False,
            "n1": False,
            "official_scorer": False,
            "six_bba": False,
            "cv57": False,
            "kaggle": False,
            "gpu": False,
            "runpod": False,
            "submission": False,
            "dead_promotion": False,
        },
    }
    atomic_json(OUTPUT_DIR / "RESULT.json", result)
    print(result["status"])


if __name__ == "__main__":
    main()
