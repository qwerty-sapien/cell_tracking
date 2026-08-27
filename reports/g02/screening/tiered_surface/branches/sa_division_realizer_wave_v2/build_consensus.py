#!/usr/bin/env python3
"""Build the sealed, target-blind exact three-way division consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


class ConsensusError(RuntimeError):
    """Fail-closed consensus error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.with_suffix(".receipt.json").exists():
        raise ConsensusError("refusing to overwrite sealed consensus")
    freeze = json.loads(args.implementation_freeze.read_text(encoding="utf-8"))
    if (
        freeze["programs"].get(Path(__file__).name)
        != sha256_file(Path(__file__).resolve())
    ):
        raise ConsensusError("consensus program does not match implementation freeze")
    prereg_sha = sha256_file(args.preregistration)
    if freeze["preregistrations"].get("D") != prereg_sha:
        raise ConsensusError("consensus preregistration drift")
    components: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for arm, path in zip(("A", "B", "C"), args.predictions, strict=True):
        receipt_path = path.with_suffix(".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "FROZEN_BEFORE_TARGET_EVALUATION"
            or receipt.get("arm") != arm
            or receipt.get("prediction_sha256") != sha256_file(path)
        ):
            raise ConsensusError(f"invalid sealed component: {arm}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "SEALED_BEFORE_EVALUATION"
            or payload.get("arm") != arm
            or payload["selection"].get("duplicate_survivors") != 0
        ):
            raise ConsensusError(f"invalid component payload: {arm}")
        components[arm] = {
            row["event_equivalence_key"]: row
            for row in payload["selection"]["predictions"]
        }
        receipts[arm] = {
            "prediction_sha256": sha256_file(path),
            "receipt_sha256": sha256_file(receipt_path),
            "count": len(components[arm]),
        }
    intersection = sorted(
        set(components["A"]) & set(components["B"]) & set(components["C"])
    )
    selected: list[dict[str, Any]] = []
    identities: set[str] = set()
    for key in intersection:
        row = dict(components["B"][key])
        row["consensus"] = {
            "rule": "exact_3_of_3",
            "component_candidate_keys": {
                arm: components[arm][key]["candidate_key"] for arm in ("A", "B", "C")
            },
            "selected_action_source": "B_late_persistence",
        }
        if row["identity"] in identities:
            raise ConsensusError("per-video consensus cap violated")
        identities.add(row["identity"])
        selected.append(row)
    keys = [row["event_equivalence_key"] for row in selected]
    if len(keys) != len(set(keys)):
        raise ConsensusError("FAIL_CLOSED_DUPLICATE_FORK")
    payload = {
        "schema": "celltrack-alpha.division-realizer-predictions.v2",
        "status": "SEALED_BEFORE_EVALUATION",
        "arm": "D",
        "rule": "EXACT_3_OF_3_TARGET_BLIND_CONSENSUS",
        "preregistration": {
            "path": str(args.preregistration.resolve()),
            "sha256": prereg_sha,
        },
        "components": receipts,
        "scope": {
            "video_count": 57,
            "label_or_bundle_read": False,
            "official_scorer_read": False,
            "component_predictions_presealed": True,
        },
        "selection": {
            "count": len(selected),
            "per_video_cap": 1,
            "duplicate_survivors": 0,
            "predictions": selected,
        },
        "cache_policy": {"full_pool_cache": False},
    }
    atomic_json(args.output, payload)
    receipt = {
        "schema": "celltrack-alpha.sealed-prediction-receipt.v1",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "arm": "D",
        "path": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "prediction_sha256": sha256_file(args.output),
        "atomic": True,
        "duplicate_survivors": 0,
    }
    atomic_json(args.output.with_suffix(".receipt.json"), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs=3, type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--implementation-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
