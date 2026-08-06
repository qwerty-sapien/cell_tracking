#!/usr/bin/env python3
"""Build the preregistered Arm-D unanimous consensus without target access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def predictions(payload: dict[str, object], arm: str) -> list[dict[str, object]]:
    if arm == "A":
        value = payload.get("predictions")
    elif arm == "B":
        selection = payload.get("selection")
        value = selection.get("predictions") if isinstance(selection, dict) else None
    else:
        value = payload.get("predictions")
    if not isinstance(value, list):
        raise ValueError(f"Arm {arm} has no predictions list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Arm {arm} predictions contain a non-object")
    return value


def edge_set(item: dict[str, object]) -> tuple[tuple[int, int], ...]:
    action = item.get("action")
    source = action if isinstance(action, dict) else item
    raw_edges = source.get("add_edges")
    if not isinstance(raw_edges, list):
        raise ValueError("prediction lacks add_edges")
    parsed = []
    for edge in raw_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("invalid add edge")
        parsed.append((int(edge[0]), int(edge[1])))
    return tuple(sorted(parsed))


def vote_key(item: dict[str, object]) -> tuple[str, str, tuple[tuple[int, int], ...]]:
    return (
        str(item["identity"]),
        str(item["candidate_key"]),
        edge_set(item),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-a", type=Path, required=True)
    parser.add_argument("--arm-b", type=Path, required=True)
    parser.add_argument("--arm-c", type=Path, required=True)
    parser.add_argument("--arm-c-result", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    preregistration = read_json(args.preregistration)
    if not isinstance(preregistration, dict):
        raise ValueError("invalid preregistration")
    if preregistration.get("required_agreement") == "3_OF_3":
        raise ValueError("unexpected preregistration shape")
    consensus_rule = preregistration.get("consensus_rule")
    if not isinstance(consensus_rule, dict):
        raise ValueError("missing consensus rule")
    if consensus_rule.get("required_agreement") != "3_OF_3":
        raise ValueError("consensus rule is not 3-of-3")
    if consensus_rule.get("missing_arm_vote") != "ABSTAIN":
        raise ValueError("missing-arm rule is not abstention")

    inputs: dict[str, dict[str, object]] = {}
    vote_sets: dict[str, dict[tuple[str, str, tuple[tuple[int, int], ...]], dict[str, object]]] = {}
    for arm, path in (("A", args.arm_a), ("B", args.arm_b)):
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Arm {arm} prediction payload is invalid")
        arm_predictions = predictions(payload, arm)
        keyed = {vote_key(item): item for item in arm_predictions}
        if len(keyed) != len(arm_predictions):
            raise ValueError(f"Arm {arm} contains duplicate consensus vote keys")
        vote_sets[arm] = keyed
        inputs[arm] = {
            "path": str(path),
            "sha256": sha256(path),
            "prediction_count": len(arm_predictions),
            "vote": "PRESENT",
        }

    if args.arm_c.is_file():
        payload = read_json(args.arm_c)
        if not isinstance(payload, dict):
            raise ValueError("Arm C prediction payload is invalid")
        arm_predictions = predictions(payload, "C")
        vote_sets["C"] = {vote_key(item): item for item in arm_predictions}
        inputs["C"] = {
            "path": str(args.arm_c),
            "sha256": sha256(args.arm_c),
            "prediction_count": len(arm_predictions),
            "vote": "PRESENT",
        }
    else:
        result = read_json(args.arm_c_result)
        if not isinstance(result, dict):
            raise ValueError("Arm C terminal result is invalid")
        if result.get("status") != "FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE":
            raise ValueError("missing Arm C prediction lacks the frozen fail-closed receipt")
        inputs["C"] = {
            "path": str(args.arm_c_result),
            "sha256": sha256(args.arm_c_result),
            "prediction_count": None,
            "vote": "MISSING_ABSTAIN",
            "status": result["status"],
        }

    if "C" not in vote_sets:
        unanimous_keys: set[tuple[str, str, tuple[tuple[int, int], ...]]] = set()
    else:
        unanimous_keys = set(vote_sets["A"]) & set(vote_sets["B"]) & set(vote_sets["C"])

    selected = [vote_sets["A"][key] for key in sorted(unanimous_keys)]
    occupied: set[tuple[str, int]] = set()
    for item in selected:
        identity = str(item["identity"])
        for edge in edge_set(item):
            for node in edge:
                owner = (identity, node)
                if owner in occupied:
                    raise ValueError("duplicate occupied node survived consensus")
                occupied.add(owner)

    output = {
        "schema": "celltrack-alpha.division-realizer-predictions.v1",
        "arm": "D",
        "name": "HIGH_PRECISION_UNANIMOUS_CONSENSUS",
        "status": "SEALED_BEFORE_TARGET_EVALUATION",
        "scope": {"family": "44b6", "six_bba_read": False},
        "inputs": inputs,
        "consensus": {
            "required_agreement": "3_OF_3",
            "missing_arm_vote": "ABSTAIN",
            "unanimous_vote_count": len(selected),
            "duplicate_survivors": 0,
        },
        "predictions": selected,
        "information_firewall": {
            "target_read": False,
            "oracle_read": False,
            "repair_bundle_read": False,
            "official_scorer_read": False,
            "outcome_read": False,
        },
    }
    atomic_json(args.output, output)
    receipt = {
        "schema": "celltrack-alpha.division-realizer-prediction-receipt.v1",
        "arm": "D",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "prediction_path": str(args.output),
        "prediction_sha256": sha256(args.output),
        "prediction_bytes": args.output.stat().st_size,
        "preregistration_sha256": sha256(args.preregistration),
        "duplicate_survivors": 0,
        "inputs": inputs,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
