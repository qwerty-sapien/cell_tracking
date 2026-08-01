#!/usr/bin/env python3
"""Authenticate the terminal R0 result without reopening raw image or GT arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_r0_shell_nolink as r0  # noqa: E402


APNCC_ABSOLUTE = {
    "adjusted_edge_jaccard": 0.8665345260361912,
    "raw_edge_jaccard": 0.862596057467424,
    "node_recall": 0.9926180002844743,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise r0.R0Error(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=HERE / "run_v2/RESULT.json")
    parser.add_argument(
        "--selected",
        type=Path,
        default=HERE / "run_v2/SELECTED_ACTIONS_TRUTH_BLIND.json",
    )
    parser.add_argument(
        "--manifest", type=Path, default=HERE / "truth_blind_shards_v1/MANIFEST.json"
    )
    parser.add_argument("--guard", type=Path, default=HERE / "GUARD_2.json")
    parser.add_argument(
        "--prepare",
        type=Path,
        default=r0.INPUT_ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json",
    )
    parser.add_argument(
        "--appearance",
        type=Path,
        default=r0.INPUT_ROOT / "reports/g02/extended_radius_shell_appearance_v1/report.json",
    )
    parser.add_argument("--output", type=Path, default=HERE / "TERMINAL_VALIDATION_V2.json")
    args = parser.parse_args()

    result = r0.load_json(args.result)
    selected = r0.load_json(args.selected)
    manifest = r0.load_json(args.manifest)
    guard = r0.load_json(args.guard)
    prepare = r0.load_json(args.prepare)
    require(result.get("schema") == r0.SCHEMA, "result schema drifted")
    require(selected.get("mechanism") == r0.MECHANISM, "selected mechanism drifted")
    require(
        r0.sha256_file(args.selected)
        == result.get("selection", {}).get("selected_actions_sha256"),
        "selected artifact hash drifted",
    )
    require(
        manifest.get("status") == "COMPLETE_TRUTH_BLIND_POPULATION"
        and int(manifest.get("completed_video_count", -1)) == 36
        and int(manifest.get("expected_video_count", -1)) == 36,
        "truth-blind manifest is incomplete",
    )
    require(
        all(manifest.get(key) is False for key in (
            "selection_emitted",
            "ground_truth_opened",
            "precision_emitted",
            "graph_outcome_emitted",
        )),
        "partial manifest exposed a terminal field",
    )
    require(
        guard.get("status") == "PASS"
        and int(guard.get("swap_growth_bytes", -1)) == 0
        and int(guard.get("returncode", -1)) == 0,
        "terminal resource guard did not pass",
    )

    training, labels, topology, identities, _training_manifest = r0._load_training(
        args.appearance, r0.INPUT_ROOT
    )
    models, calibrations, _fold_reports = r0.fit_nested_models(
        training, labels, topology, identities
    )
    prepared_by_identity = {
        str(row["identity"]): row
        for row in prepare.get("videos", [])
        if row.get("partition") == "train"
        and str(row.get("identity", "")).startswith("train:44b6_")
    }
    authenticated_candidates: list[dict[str, Any]] = []
    authenticated_rows = 0
    seen_identities: set[str] = set()
    for entry in manifest.get("shards", []):
        identity = str(entry["identity"])
        require(identity not in seen_identities, "manifest identity repeats")
        seen_identities.add(identity)
        prepared = prepared_by_identity[identity]
        path = Path(str(entry["path"]))
        require(r0.sha256_file(path) == str(entry["sha256"]), "shard hash drifted")
        fold = r0.outer_fold(identity)
        fingerprint = r0.model_fingerprint(models[fold], calibrations[fold])
        candidates, _video_report = r0._load_truth_blind_shard(
            path,
            identity=identity,
            expected_rows=int(prepared["counts"]["topology_0"]),
            expected_full_shell_rows=int(prepared["counts"]["shell_pairs"]),
            expected_model_fingerprint=fingerprint,
        )
        require(fingerprint == str(entry["model_fingerprint"]), "model fingerprint drifted")
        authenticated_candidates.extend(candidates)
        authenticated_rows += len(candidates)
    require(seen_identities == set(prepared_by_identity), "manifest identity set drifted")
    require(authenticated_rows == 38081, "eligible population count drifted")

    replay = r0.greedy_node_disjoint(authenticated_candidates)
    sealed_actions = selected.get("actions")
    require(replay == sealed_actions, "sealed B4 selection does not replay exactly")
    require(len(replay) == r0.BUDGET, "sealed budget drifted")
    require(not r0._contains_truth_or_terminal_field(selected), "truth leaked into selected artifact")
    occupied: set[tuple[str, int]] = set()
    for row in replay:
        endpoints = {
            (str(row["identity"]), int(row["source_id"])),
            (str(row["identity"]), int(row["target_id"])),
        }
        require(not occupied.intersection(endpoints), "sealed actions are not node-disjoint")
        occupied.update(endpoints)

    labeled_actions = result.get("selection", {}).get("actions_with_truth")
    require(isinstance(labeled_actions, list) and len(labeled_actions) == 4, "labeled B4 drifted")
    require(
        [(row["identity"], row["source_id"], row["target_id"]) for row in labeled_actions]
        == [(row["identity"], row["source_id"], row["target_id"]) for row in replay],
        "post-seal labels do not align with sealed B4",
    )
    require(
        all(row.get("correct_nondivision_continuation") is False for row in labeled_actions),
        "terminal precision result drifted",
    )
    require(
        result.get("precision_gate")
        == {
            "correct": 0,
            "denominator": 4,
            "precision": 0.0,
            "required": 1.0,
            "status": "KILL_EARLY",
        },
        "precision gate drifted",
    )
    require(
        result.get("graph", {}).get("delta_adjusted_raw_node") == [0.0, 0.0, 0.0]
        and result.get("graph", {}).get("opened") is False
        and result.get("graph", {}).get("measured") is False,
        "graph closure drifted",
    )
    require(
        result.get("graph", {}).get("absolute_candidate_adjusted_raw_node")
        == [
            APNCC_ABSOLUTE["adjusted_edge_jaccard"],
            APNCC_ABSOLUTE["raw_edge_jaccard"],
            APNCC_ABSOLUTE["node_recall"],
        ]
        and result.get("graph", {}).get("absolute_control_adjusted_raw_node")
        == result.get("graph", {}).get("absolute_candidate_adjusted_raw_node")
        and result.get("graph", {}).get("absolute_vector_source", {}).get(
            "artifact_sha256"
        )
        == "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d",
        "absolute no-op vector pin drifted",
    )
    require(
        result.get("selection", {}).get("sealed_edit_set_size") == 4
        and result.get("selection", {}).get("applied_edit_set_size") == 0,
        "edit-set sizes drifted",
    )
    require(
        result.get("properties")
        and all(result["properties"].get(key) is True for key in (
            "appearance_grounded", "narrow", "node_disjoint", "recall_preserving"
        )),
        "four-property audit drifted",
    )

    validation = {
        "schema": "g02.r0-shell-freefree.terminal-validation.v1",
        "status": "PASS",
        "checks_passed": 25,
        "authenticated_video_shards": 36,
        "authenticated_eligible_rows": authenticated_rows,
        "sealed_edit_set_size": 4,
        "applied_edit_set_size": 0,
        "precision_at_b4": 0.0,
        "verdict": "KILL_EARLY",
        "exact_delta_adjusted_raw_node": [0.0, 0.0, 0.0],
        "exact_absolute_candidate_adjusted_raw_node": [
            APNCC_ABSOLUTE["adjusted_edge_jaccard"],
            APNCC_ABSOLUTE["raw_edge_jaccard"],
            APNCC_ABSOLUTE["node_recall"],
        ],
        "four_properties": {
            "appearance_grounded": True,
            "narrow": True,
            "node_disjoint": True,
            "recall_preserving": True,
        },
        "guard": {
            "status": "PASS",
            "swap_growth_bytes": 0,
            "minimum_free_pages": int(guard["minimum_free_pages"]),
        },
        "result_sha256": r0.sha256_file(args.result),
        "selected_actions_sha256": r0.sha256_file(args.selected),
        "manifest_sha256": r0.sha256_file(args.manifest),
        "guard_sha256": r0.sha256_file(args.guard),
    }
    r0._atomic_create_json(args.output, validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
