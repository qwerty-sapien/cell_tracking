#!/usr/bin/env python3
"""Serialization-only recovery for the canonical h000 score receipt.

The first score attempt completed computation but the strict JSON writer
rejected per-video IEEE NaN division diagnostics.  This wrapper reuses the
unchanged canonical scorer implementation and monkey-patches only its atomic
serializer so non-finite floats are represented by explicit JSON strings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts import g03_phase2_h000_canonical as canonical
from scripts import g03_r1_zero_shot as r1
from scripts.g03_phase2_adapter import sha256


RECOVERY_SCHEMA = "celltrack-alpha.g03-h000-score-serialization-recovery.v1"


class RecoveryError(RuntimeError):
    """The h000 serialization recovery contract failed closed."""


def sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    return value


def load_recovery(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise RecoveryError("recovery contract SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError("cannot load recovery contract") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != RECOVERY_SCHEMA
        or value.get("status") != "SEALED_AFTER_SERIALIZATION_FAILURE"
        or value.get("scoring_algorithm_changed") is not False
        or value.get("graph_or_scorer_changed") is not False
        or value.get("allowed_transformation")
        != "replace non-finite float diagnostics with explicit JSON strings"
    ):
        raise RecoveryError("recovery contract is not eligible")
    if sha256(Path(__file__).resolve()) != value["implementation_sha256"]:
        raise RecoveryError("recovery implementation SHA-256 mismatch")
    if sha256(Path(canonical.__file__).resolve()) != value[
        "canonical_implementation_sha256"
    ]:
        raise RecoveryError("canonical implementation drifted")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--recovery-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recovery = load_recovery(args.recovery, args.recovery_sha256)
    if args.output.exists():
        raise RecoveryError("score output already exists; repeat forbidden")
    amendment = recovery["amendment"]
    seal = recovery["canonical_seal"]
    score_args = SimpleNamespace(
        amendment=Path(amendment["path"]),
        amendment_sha256=amendment["sha256"],
        seal=Path(seal["path"]),
        seal_sha256=seal["sha256"],
        output=args.output,
    )
    original_atomic_json = r1.atomic_json

    def sanitized_atomic_json(path: Path, value: Any) -> None:
        original_atomic_json(path, sanitize(value))

    r1.atomic_json = sanitized_atomic_json
    try:
        result = canonical.score(score_args)
    finally:
        r1.atomic_json = original_atomic_json
    clean = sanitize(result)
    clean["serialization_recovery"] = {
        "contract": {"path": str(args.recovery), "sha256": args.recovery_sha256},
        "reason": recovery["reason"],
        "scoring_algorithm_changed": False,
        "graph_or_scorer_changed": False,
        "transformation": recovery["allowed_transformation"],
    }
    # The canonical writer has already published the sanitized payload. Add the
    # recovery provenance only through a separate immutable receipt so the
    # scorer-produced artifact itself remains exactly the wrapper output.
    receipt_path = args.output.with_name("SCORE_SERIALIZATION_RECOVERY.json")
    original_atomic_json(
        receipt_path,
        {
            "schema": RECOVERY_SCHEMA,
            "status": "COMPLETE_DETERMINISTIC_SERIALIZATION_RECOVERY",
            "contract": {
                "path": str(args.recovery),
                "sha256": args.recovery_sha256,
            },
            "score_output": {
                "path": str(args.output),
                "sha256": sha256(args.output),
            },
            "scoring_algorithm_changed": False,
            "graph_or_scorer_changed": False,
            "transformation": recovery["allowed_transformation"],
        },
    )
    print(json.dumps(clean, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
