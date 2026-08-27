#!/usr/bin/env python3
"""Fail-closed validation and path audit for the PARK-B sweep."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess


OUT = pathlib.Path(__file__).resolve().parent
WORKTREE = OUT.parents[5]
ALLOWED_PREFIX = "reports/g02/screening/tiered_surface/branches/sa_parkb/"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    results_path = OUT / "SUPPORT_FALSIFIERS.json"
    fragment_path = OUT / "state_fragment.json"
    results = json.loads(results_path.read_text())
    fragment = json.loads(fragment_path.read_text())
    failures: list[str] = []
    if results.get("status") != "VERDICT_BARRIER":
        failures.append("support sweep is not at VERDICT_BARRIER")
    if results.get("park_b_nodes_tested") != 22:
        failures.append("support sweep did not cover 22 PARK_B nodes")
    if results.get("identity_checks_passed") != 22:
        failures.append("not all evidence identities passed")
    expected_moves = {
        ("G2", "PARK_A"),
        ("GAP-F1", "PARK_C"),
        ("DET-F0", "PARK_C"),
    }
    observed_moves = {
        (move["id"], move["to"]) for move in results.get("proposed_moves", [])
    }
    if observed_moves != expected_moves:
        failures.append(f"unexpected tier moves: {sorted(observed_moves)}")
    if results.get("unchanged_park_b") != 19:
        failures.append("unchanged PARK_B count is not 19")
    if results.get("memo_quarantine", {}).get("numbers_used_for_tiering") != 0:
        failures.append("memo number was used for tiering")
    if any(results.get("family_execution", {}).values()):
        failures.append("a held family was opened")
    if any(results.get("external_actions", {}).values()):
        failures.append("an external action occurred")
    if len(fragment.get("nodes", [])) != 22:
        failures.append("state fragment does not contain 22 nodes")
    result_hash = sha256(results_path)
    for node in fragment.get("nodes", []):
        if node.get("evidence_sha256") != result_hash:
            failures.append(f"fragment evidence hash mismatch for {node.get('id')}")
        if node.get("evidence_artifact") != (
            "reports/g02/screening/tiered_surface/branches/sa_parkb/"
            "SUPPORT_FALSIFIERS.json"
        ):
            failures.append(f"fragment evidence path mismatch for {node.get('id')}")

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORKTREE,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    changed_paths = [line[3:] for line in status if line]
    out_of_scope = [
        path for path in changed_paths if not path.startswith(ALLOWED_PREFIX)
    ]
    if out_of_scope:
        failures.append("out-of-scope paths: " + ", ".join(out_of_scope))
    if failures:
        raise SystemExit("PARK-B sweep validation failed:\n- " + "\n- ".join(failures))

    validation = {
        "schema": "celltrack-alpha.parkb-support-sweep-validation.v1",
        "status": "PASS",
        "nodes_tested": 22,
        "moves": [
            {"id": node, "to": tier} for node, tier in sorted(expected_moves)
        ],
        "unchanged_park_b": 19,
        "support_falsifiers_sha256": result_hash,
        "state_fragment_sha256": sha256(fragment_path),
        "memo_numbers_used_for_tiering": 0,
        "family_execution": False,
        "external_actions": False,
    }
    (OUT / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    path_audit = {
        "schema": "celltrack-alpha.worktree-path-allowlist.v1",
        "status": "PASS",
        "allowed_prefix": ALLOWED_PREFIX,
        "changed_paths": changed_paths,
        "out_of_scope_paths": [],
    }
    (OUT / "PATH_ALLOWLIST_AUDIT.json").write_text(
        json.dumps(path_audit, indent=2, sort_keys=True) + "\n"
    )
    (OUT / "TERMINAL_REPORT.md").write_text(
        "# SA-PARKB-SWEEP terminal report\n\n"
        "Verdict: **VERDICT_BARRIER**.\n\n"
        "All 22 PARK-B nodes passed banked evidence identity checks. The frozen "
        "support rules propose `G2` B→A, `GAP-F1` B→C, and `DET-F0` B→C; "
        "19 nodes remain PARK-B. These are proposals only for root adjudication. "
        "No graph family, GPU, CV57, submission, RunPod, DEAD, plan amendment, "
        "or root single-writer file opened. C2 and Branch-B memo numbers remained "
        "quarantined.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
