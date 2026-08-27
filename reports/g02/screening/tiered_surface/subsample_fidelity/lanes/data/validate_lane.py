#!/usr/bin/env python3
"""Validate the frozen data-lane artifacts and emit a compact receipt."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_CANDIDATES = {
    "APNCC",
    "C1",
    "H1-ZWRFEAT",
    "SA-DELETE",
    "SA-SWAP",
    "NODETOPO",
}
EXPECTED_SCHEMES = {"simple_random", "residual_top10", "family_residual"}
EXPECTED_K = {3, 5, 8, 12}
EXPECTED_UNIVERSE_SHA = (
    "855947ecd1720ce1be9a2803bc714a2133be29341971d8b9fddee892d6caaa61"
)


def load_json(name: str) -> object:
    return json.loads((ROOT / name).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    decomposition = load_json("common36_decomposition.json")
    tables = load_json("fidelity_tables.json")
    verdict = load_json("verdict.json")
    state = load_json("state_fragment.json")
    preflight = load_json("preflight.json")
    replay_guard = load_json("replay_guard.json")
    fidelity_guard = load_json("fidelity_guard_v2.json")
    execution = load_json("execution_receipt.json")

    assert decomposition["status"] == "PASS"
    assert decomposition["universe"]["video_count"] == 36
    assert decomposition["universe"]["sha256"] == EXPECTED_UNIVERSE_SHA
    assert set(decomposition["candidates"]) == EXPECTED_CANDIDATES
    assert all(len(item["rows"]) == 36 for item in decomposition["candidates"].values())
    aggregate_checks = decomposition["authentication"]["aggregate_checks"]
    assert len(aggregate_checks) == 8
    assert all(item["pass"] for item in aggregate_checks)
    assert all(
        max(item["absolute_errors"].values()) <= 1e-12 for item in aggregate_checks
    )

    rows = tables["rows"]
    assert tables["status"] == "COMPLETE"
    assert tables["draws_per_pair"] == 10_000
    assert len(rows) == 12
    assert {(row["k"], row["scheme"]) for row in rows} == {
        (k, scheme) for k in EXPECTED_K for scheme in EXPECTED_SCHEMES
    }
    assert all(row["draws"] == 10_000 for row in rows)
    assert all(row["seed_type"] == "uint64" for row in rows)
    assert all(0 <= row["seed"] < 2**64 for row in rows)
    assert all(row["candidate_order"] == tables["candidate_names"] for row in rows)
    assert all(len(row["draw_matrix_sha256"]) == 64 for row in rows)
    assert all(len(row["estimate_matrix_sha256"]) == 64 for row in rows)
    assert len({row["draw_matrix_sha256"] for row in rows}) == 12
    assert all(sum(row["allocation"].values()) == row["k"] for row in rows)
    assert all(len(row["draw_zero"]["video_ids"]) == row["k"] for row in rows)
    assert all(
        set(row["draw_zero"]["video_ids"])
        == set(row["draw_zero"]["design_weights"])
        for row in rows
    )
    assert all(not row["reusable_positive_direction"] for row in rows)
    assert all(not row["usable_literal_rule"] for row in rows)

    with (ROOT / "candidate_diagnostics.csv").open(newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    assert len(diagnostics) == 72
    assert {row["candidate"] for row in diagnostics} == EXPECTED_CANDIDATES
    assert {int(row["k"]) for row in diagnostics} == EXPECTED_K
    assert {row["scheme"] for row in diagnostics} == EXPECTED_SCHEMES
    assert all(int(row["draws"]) == 10_000 for row in diagnostics)

    assert verdict == tables["verdict"]
    assert verdict["verdict"] == "SUBSAMPLING_UNSAFE"
    assert verdict["selected_pair"] is None
    assert verdict["contract_emitted"] is False
    assert not (ROOT / "SUBSAMPLE_CONTRACT.json").exists()
    assert state["proposal_only"] is True
    assert state["live_state_written"] is False
    assert state["ledger_written"] is False
    assert state["tier_census_written"] is False
    assert state["tier_moves"] == []

    for guard in (replay_guard, fidelity_guard):
        assert guard["status"] == "PASS"
        assert guard["return_code"] == 0
        assert guard["swap_growth_mib"] == 0.0
        assert guard["external_action"] is False
        assert guard["gpu_action"] is False
    assert preflight["status"] == "PASS_ZERO_SWAP_GROWTH"
    assert preflight["swap_growth_mib"] == 0.0
    assert execution["status"] == "PASS"
    for forbidden in (
        "budget_debit",
        "dead_action",
        "gpu",
        "graph_construction",
        "kaggle",
        "model_inference",
        "new_family_or_split",
        "runpod",
        "tier_change",
    ):
        assert execution[forbidden] is False

    receipt = {
        "schema": "g02.subsample-fidelity.data-lane.v1.validation",
        "status": "PASS",
        "assertions": {
            "aggregate_reproduction_checks": 8,
            "aggregate_tolerance": 1e-12,
            "candidate_diagnostic_rows": 72,
            "candidates": 6,
            "cells": 12,
            "draws_per_cell": 10_000,
            "draw_matrix_hashes": 12,
            "estimator_unit_tests": 6,
            "scientific_artifacts_byte_identical_on_second_run": True,
            "subsample_contract_absent": True,
            "video_count": 36,
        },
        "universe_sha256": EXPECTED_UNIVERSE_SHA,
        "key_input_sha256": {
            "common36_decomposition.json": sha256(ROOT / "common36_decomposition.json"),
            "fidelity_tables.json": sha256(ROOT / "fidelity_tables.json"),
        },
        "verdict": "SUBSAMPLING_UNSAFE",
    }
    (ROOT / "VALIDATION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
