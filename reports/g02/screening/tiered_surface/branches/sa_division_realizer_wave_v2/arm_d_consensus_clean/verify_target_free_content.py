#!/usr/bin/env python3
"""Verify clean B/D content without reading targets, labels, scorers, or outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class AuditError(RuntimeError):
    """Fail-closed target-free audit error."""


HERE = Path(__file__).resolve().parent
WAVE = HERE.parent
B_DIR = WAVE / "arm_b_persistence_clean"
A_DIR = WAVE / "arm_a_geometry_motion"
C_DIR = WAVE / "arm_c_appearance"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
VIDEO_NAME = re.compile(r"^train__(44b6|6bba)_([0-9a-f]{8})\.csv$")


PATHS = {
    "shared_contract": WAVE / "SHARED_CONTRACT.json",
    "implementation_freeze_v1": WAVE / "IMPLEMENTATION_FREEZE.json",
    "implementation_freeze_v2": WAVE / "IMPLEMENTATION_FREEZE_V2.json",
    "frozen_builder": WAVE / "build_consensus.py",
    "original_d_preregistration": WAVE / "arm_d_consensus" / "PREREGISTRATION.json",
    "a_preregistration": A_DIR / "PREREGISTRATION.json",
    "a_predictions": A_DIR / "PREDICTIONS.json",
    "a_prediction_receipt": A_DIR / "PREDICTIONS.receipt.json",
    "c_preregistration": C_DIR / "PREREGISTRATION.json",
    "c_predictions": C_DIR / "PREDICTIONS.json",
    "c_prediction_receipt": C_DIR / "PREDICTIONS.receipt.json",
    "b_preregistration": B_DIR / "PREREGISTRATION.json",
    "b_implementation_freeze": B_DIR / "IMPLEMENTATION_FREEZE.json",
    "b_execution_guard": B_DIR / "EXECUTION_GUARD_AMENDMENT.json",
    "b_run_receipt": B_DIR / "RUN_RECEIPT.json",
    "b_actions": B_DIR / "ACTIONS.json",
    "b_structural_verification": B_DIR / "STRUCTURAL_VERIFICATION.json",
    "b_predictions": B_DIR / "PREDICTIONS.json",
    "b_prediction_receipt": B_DIR / "PREDICTIONS.receipt.json",
    "clean_d_preregistration": HERE / "PREREGISTRATION.json",
    "clean_d_predictions": HERE / "PREDICTIONS.json",
    "clean_d_prediction_receipt": HERE / "PREDICTIONS.receipt.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name}: expected JSON object")
    return value


def event_key(identity: str, p: int, a: int, b: int) -> str:
    require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in (p, a, b)),
        "event key requires integer nodes",
    )
    material = json.dumps(
        [identity, p, sorted((a, b))],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(material).hexdigest()


def canonical_manifest_sha(hashes: dict[str, str]) -> str:
    payload = json.dumps(
        hashes, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_receipt(
    arm: str, predictions_path: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = load_json(predictions_path)
    receipt = load_json(receipt_path)
    require(
        predictions.get("schema") == "celltrack-alpha.division-realizer-predictions.v2"
        and predictions.get("status") == "SEALED_BEFORE_EVALUATION"
        and predictions.get("arm") == arm
        and predictions.get("selection", {}).get("duplicate_survivors") == 0,
        f"{arm}: invalid sealed prediction payload",
    )
    require(
        receipt.get("schema") == "celltrack-alpha.sealed-prediction-receipt.v1"
        and receipt.get("status") == "FROZEN_BEFORE_TARGET_EVALUATION"
        and receipt.get("arm") == arm
        and receipt.get("prediction_sha256") == sha256_file(predictions_path)
        and receipt.get("bytes") == predictions_path.stat().st_size
        and receipt.get("atomic") is True
        and receipt.get("duplicate_survivors") == 0,
        f"{arm}: invalid sealed prediction receipt",
    )
    return predictions, receipt


def validate_b_adaptation(
    b_predictions: dict[str, Any],
    actions: dict[str, Any],
    run_receipt: dict[str, Any],
    structural: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    require(
        actions.get("schema") == "cell-tracking.target-blind-persistence-actions.v1"
        and actions.get("video_count") == 57
        and actions.get("selected_fork_count") == 48,
        "clean B ACTIONS contract drift",
    )
    require(
        run_receipt.get("schema") == "cell-tracking.clean-arm-run-receipt.v1"
        and run_receipt.get("status") == "pass"
        and run_receipt.get("counts", {}).get("input_videos") == 57
        and run_receipt.get("counts", {}).get("output_videos") == 57
        and run_receipt.get("counts", {}).get("selected_forks") == 48,
        "clean B run receipt drift",
    )
    require(
        run_receipt.get("hashes", {}).get("actions_sha256") == hashes["b_actions"],
        "clean B ACTIONS hash is not authenticated by run receipt",
    )
    require(
        structural.get("schema")
        == "cell-tracking.target-free-structural-verification.v1"
        and structural.get("status") == "pass"
        and structural.get("videos") == 57
        and structural.get("selected_forks") == 48
        and structural.get("duplicates") == 0
        and structural.get("actions_sha256") == hashes["b_actions"]
        and structural.get("run_receipt_sha256") == hashes["b_run_receipt"]
        and structural.get("scorer_or_evaluator_used") is False
        and structural.get("scientific_outcome_assessed") is False,
        "clean B structural-verification chain drift",
    )

    input_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    family_inputs = {"44b6": 0, "6bba": 0}
    for label, rows, destination in (
        ("input", run_receipt.get("inputs"), input_hashes),
        ("output", run_receipt.get("outputs"), output_hashes),
    ):
        require(isinstance(rows, list) and len(rows) == 57, f"{label} manifest drift")
        for row in rows:
            require(isinstance(row, dict), f"{label} manifest non-object")
            name = Path(row.get("path", "")).name
            digest = row.get("sha256")
            match = VIDEO_NAME.fullmatch(name)
            require(match is not None, f"{label} manifest invalid video name")
            require(
                isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
                f"{label} manifest invalid SHA-256",
            )
            require(name not in destination, f"{label} manifest duplicate video")
            destination[name] = digest
            if label == "input":
                family_inputs[match.group(1)] += 1
    require(input_hashes.keys() == output_hashes.keys(), "B input/output manifest mismatch")
    require(family_inputs == {"44b6": 21, "6bba": 36}, "B family scope drift")
    require(
        structural.get("prediction_manifest_sha256")
        == canonical_manifest_sha(output_hashes),
        "B prediction manifest is not authenticated by structural receipt",
    )

    action_rows = {row["dataset"]: row for row in actions.get("videos", [])}
    require(len(action_rows) == 57, "B ACTIONS contains duplicate datasets")
    require(
        {f"train__{dataset}.csv" for dataset in action_rows} == set(input_hashes),
        "B ACTIONS and authenticated input manifest disagree",
    )
    predictions = b_predictions.get("selection", {}).get("predictions")
    require(isinstance(predictions, list) and len(predictions) == 48,
            "B adapted selection count drift")
    identities: set[str] = set()
    candidate_keys: set[str] = set()
    equivalence_keys: set[str] = set()
    by_family = {"44b6": 0, "6bba": 0}
    operational_fields = (
        "kind",
        "add_edges",
        "expected_conflict_edges",
        "replace_incoming_to",
        "replace_outgoing_from",
    )
    for row in predictions:
        require(isinstance(row, dict), "B adapted prediction is not an object")
        dataset = row.get("dataset")
        require(isinstance(dataset, str) and dataset in action_rows,
                "B adapted prediction has unknown dataset")
        source_video = action_rows[dataset]
        source_action = source_video.get("action")
        require(source_action is not None, f"{dataset}: adapted a source abstention")
        event = source_action.get("event_nodes", {})
        p, a, b = event.get("p"), event.get("a"), event.get("b")
        identity = f"train:{dataset}"
        key = event_key(identity, p, a, b)
        require(row.get("identity") == identity, f"{dataset}: identity drift")
        require(row.get("family") == dataset.split("_", 1)[0], f"{dataset}: family drift")
        require(row.get("observed_parent_node") == p, f"{dataset}: parent drift")
        require(row.get("observed_branch_nodes") == sorted((a, b)),
                f"{dataset}: branch drift")
        require(row.get("candidate_key") == key, f"{dataset}: candidate-key drift")
        require(row.get("event_equivalence_key") == key,
                f"{dataset}: equivalence-key drift")
        require(
            row.get("clean_source_action_id") == source_action.get("action_id"),
            f"{dataset}: source action identity drift",
        )
        adapted_action = row.get("action")
        require(isinstance(adapted_action, dict), f"{dataset}: missing adapted action")
        require(
            all(adapted_action.get(field) == source_action.get(field)
                for field in operational_fields),
            f"{dataset}: operational action payload drift",
        )
        features = row.get("features", {})
        require(
            features.get("axis_scales") == source_video.get("axis_scales")
            and features.get("edge_length_upper_tukey_fence")
            == source_video.get("edge_length_upper_tukey_fence")
            and features.get("target_blind_cost") == event.get("cost"),
            f"{dataset}: target-blind diagnostic drift",
        )
        require(identity not in identities, f"{dataset}: per-video cap violated")
        require(key not in candidate_keys and key not in equivalence_keys,
                f"{dataset}: duplicate event key")
        identities.add(identity)
        candidate_keys.add(key)
        equivalence_keys.add(key)
        by_family[row["family"]] += 1
    require(by_family == {"44b6": 19, "6bba": 29}, "B selected family counts drift")
    require(
        b_predictions.get("selection", {}).get("count") == 48
        and b_predictions.get("selection", {}).get("by_family") == by_family
        and b_predictions.get("selection", {}).get("per_video_cap") == 1
        and b_predictions.get("selection", {}).get("duplicate_survivors") == 0,
        "B adapted selection summary drift",
    )
    source_receipts = b_predictions.get("source_receipts", {})
    require(
        source_receipts.get("actions_sha256") == hashes["b_actions"]
        and source_receipts.get("run_receipt_sha256") == hashes["b_run_receipt"]
        and source_receipts.get("selected_forks") == 48
        and source_receipts.get("forbidden_reads") == 0,
        "B adapted source-receipt chain drift",
    )
    return {
        "status": "pass",
        "videos": 57,
        "selected": 48,
        "by_family": by_family,
        "per_video_cap": 1,
        "duplicates": 0,
        "authenticated_upstream_inputs": 57,
        "authentication_mode": (
            "hash_chain_to_clean_RUN_RECEIPT_and_target_free_STRUCTURAL_VERIFICATION;"
            "large_CSVs_not_reopened"
        ),
        "authenticated_input_manifest_sha256": canonical_manifest_sha(input_hashes),
        "authenticated_prediction_manifest_sha256": canonical_manifest_sha(output_hashes),
        "operational_action_fields_equal_source": True,
    }


def audit() -> dict[str, Any]:
    hashes = {name: sha256_file(path) for name, path in PATHS.items()}
    shared = load_json(PATHS["shared_contract"])
    freeze_v1 = load_json(PATHS["implementation_freeze_v1"])
    freeze_v2 = load_json(PATHS["implementation_freeze_v2"])
    original_d_prereg = load_json(PATHS["original_d_preregistration"])
    clean_d_prereg = load_json(PATHS["clean_d_preregistration"])
    actions = load_json(PATHS["b_actions"])
    run_receipt = load_json(PATHS["b_run_receipt"])
    structural = load_json(PATHS["b_structural_verification"])
    a_predictions, _ = validate_receipt(
        "A", PATHS["a_predictions"], PATHS["a_prediction_receipt"]
    )
    b_predictions, _ = validate_receipt(
        "B", PATHS["b_predictions"], PATHS["b_prediction_receipt"]
    )
    c_predictions, _ = validate_receipt(
        "C", PATHS["c_predictions"], PATHS["c_prediction_receipt"]
    )
    d_predictions, _ = validate_receipt(
        "D", PATHS["clean_d_predictions"], PATHS["clean_d_prediction_receipt"]
    )
    require(
        shared.get("schema") == "celltrack-alpha.division-realizer-wave-contract.v2"
        and shared.get("status") == "FROZEN_BEFORE_OUTCOMES"
        and shared.get("binding_scope", {}).get("fragment_counts")
        == {"44b6": 21, "6bba": 36, "total": 57},
        "shared frozen contract drift",
    )
    b_audit = validate_b_adaptation(
        b_predictions, actions, run_receipt, structural, hashes
    )

    components = {"A": a_predictions, "B": b_predictions, "C": c_predictions}
    component_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, payload in components.items():
        rows = payload.get("selection", {}).get("predictions")
        require(isinstance(rows, list), f"{arm}: invalid selection list")
        mapping = {row["event_equivalence_key"]: row for row in rows}
        require(len(mapping) == len(rows), f"{arm}: duplicate equivalence key")
        component_maps[arm] = mapping
        recorded = d_predictions.get("components", {}).get(arm, {})
        receipt_key = {"A": "a_prediction_receipt", "B": "b_prediction_receipt",
                       "C": "c_prediction_receipt"}[arm]
        prediction_key = {"A": "a_predictions", "B": "b_predictions",
                          "C": "c_predictions"}[arm]
        require(
            recorded.get("count") == len(rows)
            and recorded.get("prediction_sha256") == hashes[prediction_key]
            and recorded.get("receipt_sha256") == hashes[receipt_key],
            f"D: {arm} component seal drift",
        )
    intersection = sorted(
        set(component_maps["A"]) & set(component_maps["B"]) & set(component_maps["C"])
    )
    expected_rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for key in intersection:
        row = dict(component_maps["B"][key])
        row["consensus"] = {
            "rule": "exact_3_of_3",
            "component_candidate_keys": {
                arm: component_maps[arm][key]["candidate_key"] for arm in ("A", "B", "C")
            },
            "selected_action_source": "B_late_persistence",
        }
        require(row["identity"] not in identities, "D: per-video cap violated")
        identities.add(row["identity"])
        expected_rows.append(row)
    require(
        d_predictions.get("selection", {}).get("predictions") == expected_rows
        and d_predictions.get("selection", {}).get("count") == len(expected_rows)
        and d_predictions.get("selection", {}).get("per_video_cap") == 1
        and d_predictions.get("selection", {}).get("duplicate_survivors") == 0,
        "D content differs from frozen exact-3-of-3 builder semantics",
    )
    require(
        clean_d_prereg.get("schema") == "celltrack-alpha.division-arm-preregistration.v2"
        and clean_d_prereg.get("status")
        == "PREREGISTERED_BEFORE_CLEAN_CONSENSUS_OUTCOME"
        and clean_d_prereg.get("design", {}).get("rule")
        == "exact 3-of-3 intersection of sealed A, clean B, and C predictions",
        "clean D preregistration rule drift",
    )
    require(
        d_predictions.get("preregistration", {}).get("sha256")
        == hashes["clean_d_preregistration"],
        "D payload does not pin clean D preregistration",
    )

    frozen_program_v1 = freeze_v1.get("programs", {}).get("build_consensus.py")
    frozen_program_v2 = freeze_v2.get("programs", {}).get("build_consensus.py")
    frozen_d_v1 = freeze_v1.get("preregistrations", {}).get("D")
    frozen_d_v2 = freeze_v2.get("preregistrations", {}).get("D")
    require(
        frozen_program_v1 == frozen_program_v2 == hashes["frozen_builder"],
        "published builder does not match both implementation freezes",
    )
    require(
        frozen_d_v1 == frozen_d_v2 == hashes["original_d_preregistration"],
        "published freezes do not consistently pin original D preregistration",
    )
    require(
        hashes["clean_d_preregistration"] != frozen_d_v1,
        "expected clean D freeze-provenance mismatch disappeared",
    )
    require(original_d_prereg.get("arm") == "D", "original D preregistration drift")
    return {
        "schema": "celltrack-alpha.target-free-clean-seal-audit.v1",
        "status": "PASS_TARGET_FREE_CONTENT__FAIL_CLOSED_FROZEN_BUILD_PROVENANCE",
        "binding_status": "INVALID_NONBINDING",
        "content_validation": {
            "status": "pass",
            "clean_b": b_audit,
            "clean_d": {
                "status": "pass_content_only",
                "rule": "exact_3_of_3_target_blind_intersection",
                "component_counts": {
                    arm: len(component_maps[arm]) for arm in ("A", "B", "C")
                },
                "intersection_count": len(intersection),
                "selected_count": len(expected_rows),
                "per_video_cap": 1,
                "duplicates": 0,
                "component_seals_match": True,
                "payload_matches_frozen_builder_semantics": True,
            },
        },
        "frozen_build_provenance": {
            "status": "fail_closed",
            "reason": (
                "Both published implementation freezes pin the original D "
                "preregistration, not the clean D preregistration embedded in the "
                "clean consensus. No published pre-outcome clean implementation "
                "freeze exists, so the frozen builder rejects an exact replay with "
                "'consensus preregistration drift'."
            ),
            "frozen_builder_sha256": hashes["frozen_builder"],
            "implementation_freeze_v1_sha256": hashes["implementation_freeze_v1"],
            "implementation_freeze_v2_sha256": hashes["implementation_freeze_v2"],
            "frozen_original_d_preregistration_sha256": frozen_d_v1,
            "actual_original_d_preregistration_sha256": hashes[
                "original_d_preregistration"
            ],
            "clean_d_preregistration_sha256": hashes["clean_d_preregistration"],
            "hashes_match": False,
            "retroactive_freeze_created": False,
            "binding_consensus_allowed": False,
        },
        "artifact_hashes": {
            "clean_b_predictions_sha256": hashes["b_predictions"],
            "clean_b_prediction_receipt_sha256": hashes["b_prediction_receipt"],
            "clean_d_predictions_sha256": hashes["clean_d_predictions"],
            "clean_d_prediction_receipt_sha256": hashes[
                "clean_d_prediction_receipt"
            ],
            "a_predictions_sha256": hashes["a_predictions"],
            "c_predictions_sha256": hashes["c_predictions"],
            "actions_sha256": hashes["b_actions"],
            "run_receipt_sha256": hashes["b_run_receipt"],
            "structural_verification_sha256": hashes[
                "b_structural_verification"
            ],
        },
        "firewall_attestation": {
            "status": "pass",
            "read_scope": [str(path.resolve()) for path in PATHS.values()],
            "forbidden_reads": 0,
            "division_target_or_bundle_read": False,
            "label_or_ground_truth_read": False,
            "scorer_code_or_output_read": False,
            "evaluator_code_or_output_read": False,
            "result_or_terminal_report_read": False,
            "campaign_outcome_report_read": False,
            "clean_terminal_read": False,
            "gpu_used": False,
            "network_used": False,
            "external_action": False,
            "submission_made": False,
            "peft_launched": False,
            "live_state_mutated": False,
            "ledger_read_or_mutated": False,
        },
    }


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise AuditError("refusing to overwrite target-free final receipt")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    value = audit()
    value["verified_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    value["verification_program_sha256"] = sha256_file(Path(__file__).resolve())
    if args.output is not None:
        atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
