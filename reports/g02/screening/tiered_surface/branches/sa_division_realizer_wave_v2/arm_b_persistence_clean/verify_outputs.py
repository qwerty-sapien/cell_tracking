#!/usr/bin/env python3
"""Independent target-free audit of the published arm artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from predict import Candidate, FailClosed, parse_graph_bytes, verify_only_parent_changed


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fail(message: str) -> None:
    raise FailClosed(message)


def main() -> int:
    arm_dir = Path(__file__).resolve().parent
    repo_root = arm_dir.parents[6]
    fragment_dir = repo_root / "kaggle_kernel/bhcv57out_v3/fragments"
    prediction_dir = arm_dir / "predictions"
    receipt_bytes = (arm_dir / "RUN_RECEIPT.json").read_bytes()
    action_bytes = (arm_dir / "ACTIONS.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    actions = json.loads(action_bytes)

    if receipt["status"] != "pass":
        fail("run receipt is not passing")
    if digest(action_bytes) != receipt["hashes"]["actions_sha256"]:
        fail("ACTIONS.json hash mismatch")
    input_rows = {Path(row["path"]).name: row for row in receipt["inputs"]}
    output_rows = {Path(row["path"]).name: row for row in receipt["outputs"]}
    action_rows = {f"train__{row['dataset']}.csv": row for row in actions["videos"]}
    prediction_names = {path.name for path in prediction_dir.glob("train__*.csv")}
    expected_names = set(input_rows)
    if not (
        len(input_rows)
        == len(output_rows)
        == len(action_rows)
        == len(prediction_names)
        == 57
    ):
        fail("manifest cardinality mismatch")
    if output_rows.keys() != expected_names or action_rows.keys() != expected_names:
        fail("receipt/action names disagree")
    if prediction_names != expected_names:
        fail("published prediction names disagree")

    selected_count = 0
    output_hashes: dict[str, str] = {}
    action_ids: set[str] = set()
    for name in sorted(expected_names):
        input_bytes = (fragment_dir / name).read_bytes()
        output_bytes = (prediction_dir / name).read_bytes()
        if digest(input_bytes) != input_rows[name]["sha256"]:
            fail(f"{name}: input hash mismatch")
        if digest(output_bytes) != output_rows[name]["sha256"]:
            fail(f"{name}: output hash mismatch")
        before = parse_graph_bytes(input_bytes, name)
        after = parse_graph_bytes(output_bytes, name)
        action = action_rows[name]["action"]
        selected = None
        if action is not None:
            selected_count += 1
            action_id = action["action_id"]
            if action_id in action_ids:
                fail(f"{name}: duplicate action ID")
            action_ids.add(action_id)
            event = action["event_nodes"]
            selected = Candidate(
                p=int(event["p"]),
                a=int(event["a"]),
                b=int(event["b"]),
                ga=int(event["ga"]),
                gb=int(event["gb"]),
                cost=float(event["cost"]),
            )
            if action["add_edges"] != [
                [selected.p, selected.a],
                [selected.p, selected.b],
            ]:
                fail(f"{name}: noncanonical action edge list")
            if action["expected_conflict_edges"] != [[selected.p, selected.a]]:
                fail(f"{name}: incorrect incident-conflict proof")
        verify_only_parent_changed(before, after, selected)
        if int(output_rows[name]["selected_fork_count"]) != int(selected is not None):
            fail(f"{name}: selected count mismatch")
        output_hashes[name] = digest(output_bytes)

    if selected_count != actions["selected_fork_count"]:
        fail("selected fork total disagrees with ACTIONS.json")
    if selected_count != receipt["counts"]["selected_forks"]:
        fail("selected fork total disagrees with RUN_RECEIPT.json")
    manifest_sha = digest(
        json.dumps(
            output_hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "videos": 57,
                "selected_forks": selected_count,
                "maximum_selected_forks_per_video": 1,
                "duplicates": 0,
                "actions_sha256": digest(action_bytes),
                "run_receipt_sha256": digest(receipt_bytes),
                "prediction_manifest_sha256": manifest_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FailClosed, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
