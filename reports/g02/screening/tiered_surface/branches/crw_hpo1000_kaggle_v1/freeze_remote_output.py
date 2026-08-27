#!/usr/bin/env python3
"""Fail-closed freeze of the exact CRW-HPO1000 Version-1 remote output."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ARTIFACT_ROOT = HERE / "remote_output_v1" / "crw_hpo1000_v1"
FREEZE = HERE / "REMOTE_OUTPUT_FREEZE.json"
EXPECTED = (
    "DISPERSION_SUMMARY.json",
    "RUN_LOG.jsonl",
    "SEMANTIC_RECEIPT.json",
    "TRIAL_RESULTS.jsonl",
    "WINNER.json",
)
BINDINGS = {
    "manifest_sha256": "3fce7917c2e4eb713b94cb5b936599806901813039c7e58214bc64b065c416a9",
    "carrier_sha256": "c46afff71557c657d97814baa3ba452101502103f56ade550909bbd158b0b02b",
    "parameter_bank_sha256": "21ff9a481d0dc01f9bd6063a7595ea3e56a0ba112b11ffaeb0d5a7dbf49b6997",
    "preregistration_sha256": "5ed8a44693ab9921a5bab4240a5aa570a8e9dc31e92ccaf18884df5ab7c2dd55",
}


class FreezeError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise FreezeError(f"JSON root is not an object: {path.name}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FreezeError(f"JSONL row is not an object: {path.name}:{line_number}")
            rows.append(value)
    return rows


def validate() -> dict[str, Any]:
    if FREEZE.exists():
        raise FreezeError("remote output freeze already exists")
    if not ARTIFACT_ROOT.is_dir() or ARTIFACT_ROOT.is_symlink():
        raise FreezeError(f"exact artifact root is absent or unsafe: {ARTIFACT_ROOT}")
    children = tuple(sorted(path.name for path in ARTIFACT_ROOT.iterdir()))
    if children != EXPECTED:
        raise FreezeError(f"remote output inventory drifted: {children}")
    paths = {name: ARTIFACT_ROOT / name for name in EXPECTED}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise FreezeError("remote output contains a missing, non-file, or symlinked artifact")

    semantic = read_json(paths["SEMANTIC_RECEIPT.json"])
    winner = read_json(paths["WINNER.json"])
    dispersion = read_json(paths["DISPERSION_SUMMARY.json"])
    trials = read_jsonl(paths["TRIAL_RESULTS.jsonl"])
    run_log = read_jsonl(paths["RUN_LOG.jsonl"])
    if (
        semantic.get("schema") != "celltrack-alpha.crw-hpo1000-semantic-receipt.v1"
        or semantic.get("status") != "COMPLETE"
        or semantic.get("completed_workload_units") != 1000
        or semantic.get("expected_workload_units") != 1000
        or semantic.get("full_parameter_bank_size") != 1000
        or semantic.get("unique_tuple_count") != 1000
        or semantic.get("bindings") != BINDINGS
        or semantic.get("scientific_boundaries") != {
            "competition_path_visits": 0,
            "cv57_accessed": False,
            "objective_is_official_score": False,
            "official_scorer_accessed": False,
            "six_bba_accessed": False,
            "submission_artifact_written": False,
        }
    ):
        raise FreezeError("semantic completion or scientific boundary drifted")
    expected_trial_digest = hashlib.sha256(canonical(list(range(1000)))).hexdigest()
    if semantic.get("evaluated_trial_ids_sha256") != expected_trial_digest:
        raise FreezeError("evaluated trial identity digest drifted")
    if len(trials) != 1000:
        raise FreezeError("trial result count is not 1000")
    trial_ids = [row.get("trial_id") for row in trials]
    tuple_ids = [row.get("tuple_id") for row in trials]
    if set(trial_ids) != set(range(1000)) or len(set(tuple_ids)) != 1000:
        raise FreezeError("trial or tuple identities are incomplete or duplicated")
    for row in trials:
        for key in ("objective_net", "positive_videos", "worst_eight_video_net_sum", "selected_transactions"):
            value = row.get(key)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise FreezeError(f"trial metric is invalid: {key}")
    expected_winner = max(
        trials,
        key=lambda row: (
            row["objective_net"],
            row["positive_videos"],
            row["worst_eight_video_net_sum"],
            -row["selected_transactions"],
            -row["trial_id"],
        ),
    )
    selected_ids = winner.get("selected_candidate_ids")
    if (
        winner.get("schema") != "celltrack-alpha.crw-hpo1000-winner.v1"
        or winner.get("status") != "FROZEN_FOR_ONE_LOCAL_GRAPH_SCORE"
        or winner.get("winner") != expected_winner
        or winner.get("graph_score_calls_authorized") != 1
        or winner.get("cv57_access") is not False
        or winner.get("six_bba_access") is not False
        or winner.get("competition_submission") is not False
        or not isinstance(selected_ids, list)
        or len(selected_ids) != expected_winner["selected_transactions"]
        or len(set(selected_ids)) != len(selected_ids)
        or any(not isinstance(value, str) or len(value) != 64 for value in selected_ids)
        or winner.get("selected_candidate_ids_sha256") != hashlib.sha256(canonical(selected_ids)).hexdigest()
    ):
        raise FreezeError("winner identity or one-score boundary drifted")
    if (
        semantic.get("winner_trial_id") != expected_winner["trial_id"]
        or semantic.get("winner_tuple_id") != expected_winner["tuple_id"]
        or semantic.get("winner_objective_net") != expected_winner["objective_net"]
        or semantic.get("winner_selected_transactions") != expected_winner["selected_transactions"]
        or semantic.get("dispersion") != dispersion
    ):
        raise FreezeError("semantic receipt does not bind the exact winner and dispersion")
    if not run_log or any(row.get("event") == "_run_failure" for row in run_log):
        raise FreezeError("run log is empty or contains a failure")
    terminal = run_log[-1]
    if (
        terminal.get("schema") != "celltrack-alpha.run-log.v1"
        or terminal.get("stage") != "complete"
        or terminal.get("event") != "stage_end"
        or terminal.get("status") != "COMPLETE"
        or terminal.get("completed_workload_units") != 1000
    ):
        raise FreezeError("run log lacks the exact terminal marker")
    return {
        "schema": "celltrack-alpha.crw-hpo1000-remote-output-freeze.v1",
        "status": "FROZEN_BEFORE_LOCAL_GRAPH_SCORE",
        "kernel_id": "onebitbrain/g02-crw-hpo1000-v1",
        "kernel_version": 1,
        "dataset_id": "onebitbrain/g02-crw-hpo1000-carrier-v1",
        "dataset_version": 1,
        "artifact_root": str(ARTIFACT_ROOT.relative_to(ROOT)),
        "files": {
            name: {"bytes": paths[name].stat().st_size, "sha256": sha256(paths[name])}
            for name in EXPECTED
        },
        "semantic": {
            "completed_workload_units": 1000,
            "unique_tuple_count": 1000,
            "winner_trial_id": expected_winner["trial_id"],
            "winner_tuple_id": expected_winner["tuple_id"],
            "selected_transactions": len(selected_ids),
            "selected_candidate_ids_sha256": winner["selected_candidate_ids_sha256"],
        },
        "bindings": {
            "freeze_script_sha256": sha256(Path(__file__).resolve()),
            "kernel_code_sha256": "eb936adf6590aa8385c7378de7b54ad7d2537df455bb6ad45f7b2840748c97c5",
            "kernel_metadata_sha256": "8239c480a30b36f36c11109011a4d330887a6e7c365e5ccb0dccb28b7aa1f213",
            "certification_sha256": "469e03de1214a9de2b78af6eae8bf89697590929b9cfdc2d4a35c17151aaba45",
            "dataset_roundtrip_sha256": "872132b68e4bf59cc4f724f09488a43916cba29f5b04445b23ac51091311ddfe",
        },
        "scientific_boundaries": {
            "official_graph_score_calls_consumed": 0,
            "official_graph_score_calls_remaining": 1,
            "cv57_accessed": False,
            "six_bba_accessed": False,
            "competition_submission": False,
            "post_outcome_retuning": False,
        },
    }


def main() -> int:
    payload = validate()
    with FREEZE.open("xb") as handle:
        handle.write(canonical(payload))
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": payload["status"], "freeze": str(FREEZE.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
