#!/usr/bin/env python3
"""Apply the pre-score exact-Decimal decision amendment to an R1 score receipt."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import tempfile


CONTRACT_SHA256 = "aefe438e6d8966b25f3665a1f009d108ae15ef0737581c1aa07a57f027050204"
RAW_SCHEMA = "celltrack-alpha.g03-r1-score.v1"
AMENDMENT_SCHEMA = "celltrack-alpha.g03-r1-decision-reference-amendment.v1"
FINAL_SCHEMA = "celltrack-alpha.g03-r1-zero-shot-result.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"not a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-score", type=Path, required=True)
    parser.add_argument("--raw-score-sha256", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    if sha256(args.raw_score) != args.raw_score_sha256:
        raise RuntimeError("raw score hash mismatch")
    if sha256(args.amendment) != args.amendment_sha256:
        raise RuntimeError("decision amendment hash mismatch")
    raw = load(args.raw_score)
    amendment = load(args.amendment)
    if (
        raw.get("schema") != RAW_SCHEMA
        or raw.get("status") != "COMPLETE_ONE_SHOT_ZERO_SHOT_MEASUREMENT"
        or raw.get("contract", {}).get("sha256") != CONTRACT_SHA256
        or raw.get("provenance", {}).get("sealed_before_score") is not True
    ):
        raise RuntimeError("raw score is not finalization-eligible")
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or amendment.get("status") != "SEALED_BEFORE_HELD_FIT_SCORE_READ"
        or amendment.get("amends_contract_sha256") != CONTRACT_SHA256
        or amendment.get("scientific_configuration_changed") is not False
        or amendment.get("integrity", {}).get("held_fit_score_read") is not False
    ):
        raise RuntimeError("decision amendment is not eligible")
    reference = amendment["authoritative_decision_reference"]
    best_text = reference["best_known_score_decimal"]
    floor_text = reference["neighborhood_floor_decimal"]
    best = Decimal(best_text)
    floor = Decimal(floor_text)
    observed = Decimal(str(raw["candidate"]["score"]))
    branch = "A" if observed >= floor or observed > best else "B"
    raw["schema"] = FINAL_SCHEMA
    raw["status"] = "COMPLETE_EXACT_DECIMAL_CLASSIFICATION"
    raw["decision_reference"] = {
        "best_known_score": float(best),
        "best_known_score_decimal": best_text,
        "neighborhood_floor": float(floor),
        "neighborhood_floor_decimal": floor_text,
        "apncc_44b6_reference": raw["decision_reference"][
            "apncc_44b6_reference"
        ],
    }
    raw["candidate"]["branch"] = branch
    raw["decision"]["neighborhood_floor"] = float(floor)
    raw["decision"]["neighborhood_floor_decimal"] = floor_text
    raw["decision"]["best_known_score_decimal"] = best_text
    raw["decision"]["delta_vs_best_known_score"] = float(observed - best)
    raw["decision"]["delta_vs_neighborhood_floor"] = float(observed - floor)
    raw["decision"]["branch"] = branch
    raw["decision"]["branch_rule"] = reference["branch_rule"]
    raw["provenance"]["decision_amendment_sha256"] = args.amendment_sha256
    raw["provenance"]["raw_score_sha256"] = args.raw_score_sha256
    raw["provenance"]["exact_decimal_classification"] = True
    payload = (
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, args.output)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "score": float(observed),
                "branch": branch,
                "neighborhood_floor_decimal": floor_text,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
