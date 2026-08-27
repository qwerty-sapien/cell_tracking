#!/usr/bin/env python3
"""Adapt the sealed clean-B actions and build a target-blind clean consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_ACTIONS_SHA256 = "635511d9d8221a3bb4b197ab2460b8c82ad80b25b8875675385dfea7401c8ec6"
EXPECTED_RUN_RECEIPT_SHA256 = "ce12552a8dfb9e4c195451b3b33690dbe9540df960d46d34a94e35ee5d695329"
EXPECTED_FIREWALL_SHA256 = "c2d228e4ec9f6e056f262d75f26110caae61702eb1cafb8242bae929f997b034"


class PreparationError(RuntimeError):
    """Fail-closed sealed-input adaptation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(identity: str, parent: int, branches: Sequence[int]) -> str:
    payload = json.dumps(
        [identity, int(parent), sorted(int(value) for value in branches)],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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


def validate_source(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = {
        "actions": sha256_file(args.actions),
        "run_receipt": sha256_file(args.run_receipt),
        "firewall": sha256_file(args.firewall),
    }
    expected = {
        "actions": EXPECTED_ACTIONS_SHA256,
        "run_receipt": EXPECTED_RUN_RECEIPT_SHA256,
        "firewall": EXPECTED_FIREWALL_SHA256,
    }
    if observed != expected:
        raise PreparationError(f"clean-B sealed source drift: {observed}")
    run_receipt = json.loads(args.run_receipt.read_text(encoding="utf-8"))
    firewall = json.loads(args.firewall.read_text(encoding="utf-8"))
    if (
        run_receipt["counts"]["input_videos"] != 57
        or run_receipt["counts"]["output_videos"] != 57
        or run_receipt["counts"]["maximum_selected_forks_per_video"] != 1
        or run_receipt["counts"]["duplicate_action_ids"] != 0
        or firewall["counts"]["forbidden_reads"] != 0
        or firewall["counts"]["target_or_label_reads"] != 0
        or firewall["counts"]["scorer_or_evaluator_runs"] != 0
    ):
        raise PreparationError("clean-B firewall or structural receipt failed")
    return run_receipt, firewall


def adapt_b(args: argparse.Namespace) -> dict[str, Any]:
    run_receipt, firewall = validate_source(args)
    actions = json.loads(args.actions.read_text(encoding="utf-8"))
    if actions.get("video_count") != 57 or actions.get("selected_fork_count") != 48:
        raise PreparationError("clean-B action inventory drift")
    selected: list[dict[str, Any]] = []
    identities: set[str] = set()
    for video in actions["videos"]:
        action = video.get("action")
        if action is None:
            continue
        dataset = str(video["dataset"])
        identity = f"train:{dataset}"
        family = dataset.split("_", 1)[0]
        event_nodes = action["event_nodes"]
        parent = int(event_nodes["p"])
        branches = sorted((int(event_nodes["a"]), int(event_nodes["b"])))
        key = stable_key(identity, parent, branches)
        if identity in identities:
            raise PreparationError("clean-B per-video cap violated")
        identities.add(identity)
        selected.append(
            {
                "candidate_key": key,
                "event_equivalence_key": key,
                "identity": identity,
                "dataset": dataset,
                "family": family,
                "observed_parent_node": parent,
                "observed_branch_nodes": branches,
                "features": {
                    "target_blind_cost": float(event_nodes["cost"]),
                    "axis_scales": [float(value) for value in video["axis_scales"]],
                    "edge_length_upper_tukey_fence": float(
                        video["edge_length_upper_tukey_fence"]
                    ),
                },
                "action": {
                    "kind": "division_fork",
                    "add_edges": [
                        [int(source), int(child)]
                        for source, child in action["add_edges"]
                    ],
                    "replace_outgoing_from": [
                        int(value) for value in action["replace_outgoing_from"]
                    ],
                    "replace_incoming_to": [
                        int(value) for value in action["replace_incoming_to"]
                    ],
                    "expected_conflict_edges": [
                        [int(source), int(child)]
                        for source, child in action["expected_conflict_edges"]
                    ],
                },
                "clean_source_action_id": action["action_id"],
            }
        )
    keys = [row["candidate_key"] for row in selected]
    if len(selected) != 48 or len(keys) != len(set(keys)):
        raise PreparationError("clean-B adapted duplicate/count failure")
    payload = {
        "schema": "celltrack-alpha.division-realizer-predictions.v2",
        "status": "SEALED_BEFORE_EVALUATION",
        "arm": "B",
        "scope": {
            "families": {"44b6": 21, "6bba": 36},
            "video_count": 57,
            "label_or_bundle_read": False,
            "official_scorer_read": False,
            "other_arm_read": False,
            "fresh_context_isolated_design": True,
        },
        "source_receipts": {
            "actions_sha256": EXPECTED_ACTIONS_SHA256,
            "run_receipt_sha256": EXPECTED_RUN_RECEIPT_SHA256,
            "firewall_sha256": EXPECTED_FIREWALL_SHA256,
            "selected_forks": run_receipt["counts"]["selected_forks"],
            "forbidden_reads": firewall["counts"]["forbidden_reads"],
        },
        "selection": {
            "count": len(selected),
            "by_family": {
                family: sum(row["family"] == family for row in selected)
                for family in ("44b6", "6bba")
            },
            "per_video_cap": 1,
            "duplicate_survivors": 0,
            "predictions": selected,
        },
        "cache_policy": {"full_pool_cache": False},
    }
    if args.b_output.exists():
        raise PreparationError("refusing to overwrite clean-B adapted output")
    atomic_json(args.b_output, payload)
    receipt = {
        "schema": "celltrack-alpha.sealed-prediction-receipt.v1",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "arm": "B",
        "path": str(args.b_output.resolve()),
        "bytes": args.b_output.stat().st_size,
        "prediction_sha256": sha256_file(args.b_output),
        "atomic": True,
        "duplicate_survivors": 0,
        "adaptation_only": True,
    }
    atomic_json(args.b_output.with_suffix(".receipt.json"), receipt)
    return receipt


def load_component(path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = path.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "FROZEN_BEFORE_TARGET_EVALUATION"
        or receipt.get("arm") != arm
        or receipt.get("prediction_sha256") != sha256_file(path)
        or payload.get("status") != "SEALED_BEFORE_EVALUATION"
        or payload["selection"].get("duplicate_survivors") != 0
    ):
        raise PreparationError(f"invalid sealed component: {arm}")
    return payload, receipt


def build_consensus(args: argparse.Namespace, b_receipt: Mapping[str, Any]) -> dict[str, Any]:
    prereg_sha = sha256_file(args.d_preregistration)
    components: dict[str, dict[str, dict[str, Any]]] = {}
    component_receipts: dict[str, dict[str, Any]] = {}
    for arm, path in zip(("A", "B", "C"), (args.a, args.b_output, args.c), strict=True):
        payload, receipt = load_component(path, arm)
        rows = {
            row["event_equivalence_key"]: row
            for row in payload["selection"]["predictions"]
        }
        if len(rows) != payload["selection"]["count"]:
            raise PreparationError(f"duplicate equivalence key: {arm}")
        components[arm] = rows
        component_receipts[arm] = {
            "prediction_sha256": sha256_file(path),
            "receipt_sha256": sha256_file(path.with_suffix(".receipt.json")),
            "count": len(rows),
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
            "selected_action_source": "clean_B_canonical_early_equivalence",
        }
        if row["identity"] in identities:
            raise PreparationError("clean consensus per-video cap violated")
        identities.add(row["identity"])
        selected.append(row)
    if args.d_output.exists():
        raise PreparationError("refusing to overwrite clean consensus")
    payload = {
        "schema": "celltrack-alpha.division-realizer-predictions.v2",
        "status": "SEALED_BEFORE_EVALUATION",
        "arm": "D",
        "rule": "EXACT_3_OF_3_TARGET_BLIND_CONSENSUS",
        "preregistration": {
            "path": str(args.d_preregistration.resolve()),
            "sha256": prereg_sha,
        },
        "components": component_receipts,
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
    atomic_json(args.d_output, payload)
    receipt = {
        "schema": "celltrack-alpha.sealed-prediction-receipt.v1",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "arm": "D",
        "path": str(args.d_output.resolve()),
        "bytes": args.d_output.stat().st_size,
        "prediction_sha256": sha256_file(args.d_output),
        "atomic": True,
        "duplicate_survivors": 0,
        "clean_b_prediction_sha256": b_receipt["prediction_sha256"],
    }
    atomic_json(args.d_output.with_suffix(".receipt.json"), receipt)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    b_receipt = adapt_b(args)
    d_receipt = build_consensus(args, b_receipt)
    return {"clean_b": b_receipt, "clean_consensus": d_receipt}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--firewall", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--d-preregistration", type=Path, required=True)
    parser.add_argument("--b-output", type=Path, required=True)
    parser.add_argument("--d-output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
