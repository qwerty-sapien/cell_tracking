#!/usr/bin/env python3
"""Adapt sealed target-blind persistence actions to the frozen wave interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence


class SealError(RuntimeError):
    """Fail-closed sealing error."""


HEX64 = re.compile(r"^[0-9a-f]{64}$")
VIDEO_NAME = re.compile(r"^train__(44b6|6bba)_([0-9a-f]{8})\.csv$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SealError(f"{path.name}: expected a JSON object")
    return value


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SealError(message)


def exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SealError(f"{label}: expected integer")
    return value


def event_key(identity: str, p: int, a: int, b: int) -> str:
    """Frozen A/C event-equivalence encoding, using only identity+p+a+b."""
    branches = sorted((exact_int(a, "a"), exact_int(b, "b")))
    material = json.dumps(
        [identity, exact_int(p, "p"), branches],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(material).hexdigest()


def manifest_from_rows(
    rows: Any, expected_total: int, label: str
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, int]]:
    require(isinstance(rows, list), f"{label}: expected list")
    require(len(rows) == expected_total, f"{label}: expected {expected_total} rows")
    hashes: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    prefix_counts = {"44b6": 0, "6bba": 0}
    for row in rows:
        require(isinstance(row, dict), f"{label}: non-object row")
        path = row.get("path")
        digest = row.get("sha256")
        require(isinstance(path, str), f"{label}: invalid path")
        require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
                f"{label}: invalid SHA-256")
        name = Path(path).name
        match = VIDEO_NAME.fullmatch(name)
        require(match is not None, f"{label}: invalid video name {name!r}")
        require(name not in hashes, f"{label}: duplicate video {name}")
        hashes[name] = digest
        records[name] = row
        prefix_counts[match.group(1)] += 1
    require(prefix_counts == {"44b6": 21, "6bba": 36},
            f"{label}: family counts drift")
    return hashes, records, prefix_counts


def canonical_manifest_sha(hashes: dict[str, str]) -> str:
    payload = json.dumps(
        hashes, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_actions(
    actions: dict[str, Any], expected_names: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    require(
        actions.get("schema") == "cell-tracking.target-blind-persistence-actions.v1",
        "ACTIONS.json schema drift",
    )
    require(actions.get("video_count") == 57, "ACTIONS.json video count drift")
    videos = actions.get("videos")
    require(isinstance(videos, list) and len(videos) == 57,
            "ACTIONS.json requires exactly 57 video rows")

    selected: list[dict[str, Any]] = []
    datasets: set[str] = set()
    action_ids: set[str] = set()
    candidate_keys: set[str] = set()
    event_keys: set[str] = set()
    by_family = {"44b6": 0, "6bba": 0}

    for video in videos:
        require(isinstance(video, dict), "ACTIONS.json contains a non-object video")
        dataset = video.get("dataset")
        require(isinstance(dataset, str), "ACTIONS.json contains invalid dataset")
        require(dataset not in datasets, f"duplicate dataset {dataset}")
        datasets.add(dataset)
        name = f"train__{dataset}.csv"
        require(name in expected_names, f"dataset absent from authenticated manifest: {dataset}")
        family = dataset.split("_", 1)[0]
        require(family in by_family, f"invalid family for {dataset}")
        action = video.get("action")
        selected_count = video.get("selected_fork_count")
        require(selected_count in (0, 1), f"{dataset}: invalid selected count")
        require((action is not None) == (selected_count == 1),
                f"{dataset}: action/count disagreement")
        if action is None:
            continue
        require(isinstance(action, dict), f"{dataset}: invalid action")
        require(action.get("kind") == "division_fork", f"{dataset}: wrong action kind")
        action_id = action.get("action_id")
        require(isinstance(action_id, str) and action_id not in action_ids,
                f"{dataset}: duplicate or invalid action ID")
        action_ids.add(action_id)
        event = action.get("event_nodes")
        require(isinstance(event, dict), f"{dataset}: missing event nodes")
        p = exact_int(event.get("p"), f"{dataset}.p")
        a = exact_int(event.get("a"), f"{dataset}.a")
        b = exact_int(event.get("b"), f"{dataset}.b")
        ga = exact_int(event.get("ga"), f"{dataset}.ga")
        gb = exact_int(event.get("gb"), f"{dataset}.gb")
        require(len({p, a, b, ga, gb}) == 5, f"{dataset}: repeated event node")
        require(action.get("add_edges") == [[p, a], [p, b]],
                f"{dataset}: action edges changed")
        require(action.get("delete_edges") == [], f"{dataset}: delete edges changed")
        require(action.get("expected_conflict_edges") == [[p, a]],
                f"{dataset}: incident-conflict proof changed")
        require(action.get("replace_incoming_to") == [a, b],
                f"{dataset}: incoming boundary changed")
        require(action.get("replace_outgoing_from") == [p],
                f"{dataset}: outgoing boundary changed")
        require(
            action.get("representation")
            == "canonical_early_of_one_frame_late_equivalence_class",
            f"{dataset}: equivalence representation changed",
        )

        identity = f"train:{dataset}"
        key = event_key(identity, p, a, b)
        require(key not in candidate_keys, f"{dataset}: duplicate candidate key")
        require(key not in event_keys, f"{dataset}: duplicate equivalence key")
        candidate_keys.add(key)
        event_keys.add(key)
        selected.append(
            {
                "identity": identity,
                "dataset": dataset,
                "family": family,
                "candidate_key": key,
                "event_equivalence_key": key,
                "observed_parent_node": p,
                "observed_branch_nodes": sorted((a, b)),
                "action": action,
            }
        )
        by_family[family] += 1

    require(
        {f"train__{dataset}.csv" for dataset in datasets} == expected_names,
        "ACTIONS.json and authenticated input manifest disagree",
    )
    require(len(selected) == actions.get("selected_fork_count") == 48,
            "selected fork total drift")
    selected.sort(key=lambda row: (row["identity"], row["event_equivalence_key"]))
    return selected, by_family


def seal(
    *,
    actions_path: Path,
    run_receipt_path: Path,
    structural_verification_path: Path,
    preregistration_path: Path,
    implementation_freeze_path: Path,
    execution_guard_amendment_path: Path,
    predictor_path: Path,
    shared_contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipt_path = output_path.with_suffix(".receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise SealError("refusing to overwrite a sealed prediction or receipt")

    paths = {
        "actions": actions_path,
        "run_receipt": run_receipt_path,
        "structural_verification": structural_verification_path,
        "preregistration": preregistration_path,
        "implementation_freeze": implementation_freeze_path,
        "execution_guard_amendment": execution_guard_amendment_path,
        "predictor": predictor_path,
        "shared_contract": shared_contract_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    actions = load_json(actions_path)
    run_receipt = load_json(run_receipt_path)
    structural = load_json(structural_verification_path)
    preregistration = load_json(preregistration_path)
    freeze = load_json(implementation_freeze_path)
    guard = load_json(execution_guard_amendment_path)
    shared = load_json(shared_contract_path)

    require(
        shared.get("schema") == "celltrack-alpha.division-realizer-wave-contract.v2"
        and shared.get("status") == "FROZEN_BEFORE_OUTCOMES",
        "shared contract is not frozen v2",
    )
    require(
        shared.get("binding_scope", {}).get("fragment_counts")
        == {"44b6": 21, "6bba": 36, "total": 57},
        "shared fragment scope drift",
    )
    require(
        preregistration.get("schema")
        == "cell-tracking.clean-target-blind-arm-preregistration.v1"
        and preregistration.get("arm") == "arm_b_persistence_clean"
        and preregistration.get("status_at_creation")
        == "locked_before_prediction_generation",
        "clean Arm B preregistration drift",
    )
    require(
        guard.get("schema") == "cell-tracking.execution-guard-amendment.v1"
        and guard.get("status") == "immutable_locked_before_any_fragment_prediction_outcome",
        "execution guard is not locked",
    )
    require(
        freeze.get("schema") == "cell-tracking.clean-arm-implementation-freeze.v1",
        "clean Arm B implementation freeze drift",
    )
    frozen_hashes = freeze.get("hashes", {})
    require(frozen_hashes.get("PREREGISTRATION.json") == hashes["preregistration"],
            "preregistration does not match clean implementation freeze")
    require(
        frozen_hashes.get("EXECUTION_GUARD_AMENDMENT.json")
        == hashes["execution_guard_amendment"],
        "execution guard does not match clean implementation freeze",
    )
    require(frozen_hashes.get("predict.py") == hashes["predictor"],
            "predictor does not match clean implementation freeze")

    require(
        run_receipt.get("schema") == "cell-tracking.clean-arm-run-receipt.v1"
        and run_receipt.get("status") == "pass",
        "clean Arm B run receipt is not passing",
    )
    run_hashes = run_receipt.get("hashes", {})
    require(run_hashes.get("actions_sha256") == hashes["actions"],
            "ACTIONS.json does not match run receipt")
    require(run_hashes.get("preregistration_sha256") == hashes["preregistration"],
            "preregistration does not match run receipt")
    require(run_hashes.get("predictor_sha256") == hashes["predictor"],
            "predictor does not match run receipt")
    require(
        run_hashes.get("execution_guard_amendment_sha256")
        == hashes["execution_guard_amendment"],
        "execution guard does not match run receipt",
    )
    counts = run_receipt.get("counts", {})
    require(
        counts.get("input_videos") == counts.get("output_videos") == 57
        and counts.get("selected_forks") == 48
        and counts.get("maximum_selected_forks_per_video") == 1
        and counts.get("duplicate_action_ids") == 0
        and counts.get("duplicate_input_paths") == 0
        and counts.get("duplicate_output_paths") == 0
        and counts.get("duplicate_output_nodes_or_edges") == 0,
        "run receipt counts are not clean",
    )
    runtime = run_receipt.get("runtime", {})
    require(
        runtime.get("cpu_only") is True
        and runtime.get("network_used") is False
        and runtime.get("scorer_or_evaluator_used") is False
        and runtime.get("zero_swapout_guard") == "pass",
        "run receipt resource/firewall guard is not passing",
    )

    input_hashes, input_rows, input_prefixes = manifest_from_rows(
        run_receipt.get("inputs"), 57, "input manifest"
    )
    output_hashes, output_rows, output_prefixes = manifest_from_rows(
        run_receipt.get("outputs"), 57, "output manifest"
    )
    require(input_hashes.keys() == output_hashes.keys(),
            "input and output video manifests disagree")
    selected, by_family = validate_actions(actions, set(input_hashes))
    for name, row in output_rows.items():
        dataset = name.removeprefix("train__").removesuffix(".csv")
        action_video = next(v for v in actions["videos"] if v["dataset"] == dataset)
        require(
            row.get("selected_fork_count") == action_video["selected_fork_count"],
            f"{dataset}: output/action selection count mismatch",
        )

    require(
        structural.get("schema")
        == "cell-tracking.target-free-structural-verification.v1"
        and structural.get("status") == "pass"
        and structural.get("videos") == 57
        and structural.get("selected_forks") == 48
        and structural.get("maximum_selected_forks_per_video") == 1
        and structural.get("duplicates") == 0
        and structural.get("scorer_or_evaluator_used") is False
        and structural.get("scientific_outcome_assessed") is False,
        "target-free structural verification is not clean",
    )
    require(structural.get("actions_sha256") == hashes["actions"],
            "structural verification ACTIONS hash drift")
    require(structural.get("run_receipt_sha256") == hashes["run_receipt"],
            "structural verification run-receipt hash drift")
    output_manifest_sha = canonical_manifest_sha(output_hashes)
    require(
        structural.get("prediction_manifest_sha256") == output_manifest_sha,
        "structural verification prediction-manifest hash drift",
    )
    input_manifest_sha = canonical_manifest_sha(input_hashes)

    payload = {
        "schema": "celltrack-alpha.division-realizer-predictions.v2",
        "status": "SEALED_BEFORE_EVALUATION",
        "arm": "B",
        "rule": "TARGET_BLIND_PERSISTENCE_CANONICAL_EARLY_FORK",
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": hashes["preregistration"],
        },
        "implementation_freeze_sha256": hashes["implementation_freeze"],
        "source_authentication": {
            "mode": (
                "hash_chain_to_preexisting_target_free_structural_verification;"
                "large_fragment_and_prediction_CSVs_not_reopened"
            ),
            "actions_sha256": hashes["actions"],
            "run_receipt_sha256": hashes["run_receipt"],
            "structural_verification_sha256": hashes["structural_verification"],
            "authenticated_input_count": len(input_rows),
            "authenticated_input_manifest_sha256": input_manifest_sha,
            "authenticated_prediction_count": len(output_rows),
            "authenticated_prediction_manifest_sha256": output_manifest_sha,
        },
        "scope": {
            "video_count": 57,
            "families": input_prefixes,
            "label_or_bundle_read": False,
            "official_scorer_read": False,
            "evaluator_read": False,
            "scientific_choices_changed": False,
            "all_upstream_inputs_authenticated": True,
        },
        "selection": {
            "count": len(selected),
            "by_family": by_family,
            "per_video_cap": 1,
            "duplicate_survivors": 0,
            "predictions": selected,
        },
        "resource_guard": {
            "cpu_only": True,
            "network_used": False,
            "gpu_used": False,
            "submission_made": False,
            "scorer_or_evaluator_used": False,
        },
        "cache_policy": {"full_pool_cache": False},
        "diagnostics": {
            "input_prefix_counts": input_prefixes,
            "output_prefix_counts": output_prefixes,
            "event_key_encoding": (
                "sha256(ascii(compact_json([identity,p,[min(a,b),max(a,b)]])))"
            ),
        },
    }
    atomic_json(output_path, payload)
    receipt = {
        "schema": "celltrack-alpha.sealed-prediction-receipt.v1",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "arm": "B",
        "path": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "prediction_sha256": sha256_file(output_path),
        "atomic": True,
        "duplicate_survivors": 0,
        "selected_count": len(selected),
        "sealing_program_sha256": sha256_file(Path(__file__).resolve()),
        "source_authentication": payload["source_authentication"],
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--structural-verification", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--implementation-freeze", type=Path, required=True)
    parser.add_argument("--execution-guard-amendment", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--shared-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            seal(
                actions_path=args.actions,
                run_receipt_path=args.run_receipt,
                structural_verification_path=args.structural_verification,
                preregistration_path=args.preregistration,
                implementation_freeze_path=args.implementation_freeze,
                execution_guard_amendment_path=args.execution_guard_amendment,
                predictor_path=args.predictor,
                shared_contract_path=args.shared_contract,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SealError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
