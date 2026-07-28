#!/usr/bin/env python3
"""Fail-closed validation, citation audit, and path audit for PARK-B Wave-2."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from typing import Any


OUT = pathlib.Path(__file__).resolve().parent
WORKTREE = OUT.parents[5]
ROOT_COMMIT = "5a805c2f7240e30c572644d3d5f0e733add04769"
ALLOWED_PREFIX = "reports/g02/screening/tiered_surface/branches/sa_parkb/"
FORBIDDEN_PRIOR_SIBLING_CITATION = (
    "reports/g02/screening/tiered_surface/branches/sa_nodetopo/"
    "PROVENANCE_AUDIT.json"
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, Any]:
    value = json.loads((OUT / name).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {name}")
    return value


def committed_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{ROOT_COMMIT}:{path}"],
        cwd=WORKTREE,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def main() -> int:
    prereg = load("PREREGISTRATION_WAVE2.json")
    inventory = load("PARKB_INVENTORY_WAVE2.json")
    support = load("SUPPORT_FALSIFIERS_WAVE2.json")
    fragment = load("state_fragment.json")
    preflight = load("IDLE_PREFLIGHT_WAVE2.json")
    resource_receipt = load("RESOURCE_RECEIPT_WAVE2.json")
    failures: list[str] = []

    if prereg.get("status") != "FROZEN_BEFORE_WAVE2_SUPPORT_OUTCOMES":
        failures.append("preregistration is not frozen before outcomes")
    if inventory.get("current_park_b_count") != 22:
        failures.append("inventory does not cover all 22 current PARK_B nodes")
    if inventory.get("cpu_testable_count") != 14:
        failures.append("CPU-testable inventory count is not 14")
    if inventory.get("skipped_weight_or_gpu_gated_representation_count") != 8:
        failures.append("skipped representation inventory count is not 8")
    inventory_ids = [row["id"] for row in inventory.get("nodes", [])]
    if len(inventory_ids) != 22 or len(set(inventory_ids)) != 22:
        failures.append("inventory node IDs are incomplete or duplicated")

    cpu_results = support.get("cpu_results", [])
    skipped = support.get("skipped_representation", [])
    if len(cpu_results) != 14:
        failures.append("support audit does not contain 14 CPU results")
    if len(skipped) != 8:
        failures.append("support audit does not contain 8 skipped representation nodes")
    result_ids = [row["id"] for row in cpu_results + skipped]
    if set(result_ids) != set(inventory_ids):
        failures.append("support result IDs differ from the complete inventory")
    if support.get("proposed_moves") != []:
        failures.append("support audit proposed a tier move")
    if support.get("unchanged_park_b") != 22:
        failures.append("support audit does not retain all 22 nodes at PARK_B")
    if any(row.get("tier_move_proposed") for row in cpu_results + skipped):
        failures.append("a node-level tier move was proposed")
    if any(row.get("proposed_tier") != "PARK_B" for row in cpu_results + skipped):
        failures.append("a node-level proposed tier differs from PARK_B")

    citations = [
        item
        for row in cpu_results + skipped
        for item in row.get("citations", [])
    ]
    for item in citations:
        path = item.get("path")
        expected = item.get("sha256")
        if item.get("root_commit") != ROOT_COMMIT:
            failures.append(f"citation does not pin root commit: {path}")
            continue
        if path == FORBIDDEN_PRIOR_SIBLING_CITATION:
            failures.append("prior unmerged NODETOPO citation reappeared")
        try:
            actual = hashlib.sha256(committed_bytes(path)).hexdigest()
        except subprocess.CalledProcessError:
            failures.append(f"citation is absent from committed root: {path}")
            continue
        if actual != expected:
            failures.append(f"citation hash mismatch: {path}: {actual} != {expected}")
    if not citations:
        failures.append("no node-specific path-and-SHA citations were emitted")
    policy = support.get("transition_evidence_policy", {})
    if policy.get("uncited_moves") != 0:
        failures.append("uncited tier moves are nonzero")
    if policy.get("prior_quarantined_fragment_used_for_transition"):
        failures.append("prior quarantined fragment was used for a transition")
    if policy.get("sibling_live_or_unmerged_output_used"):
        failures.append("sibling live or unmerged output was used")
    register_audit = support.get("committed_register_packaging_audit", {})
    if any(
        item.get("eligible_as_transition_evidence")
        for item in register_audit.values()
    ):
        failures.append("an absent clean-checkout branch register was treated as eligible")

    if support.get("memo_quarantine", {}).get("numbers_used_for_tiering") != 0:
        failures.append("a quarantined memo number was used for tiering")
    if any(support.get("family_execution", {}).values()):
        failures.append("a graph family was opened")
    if any(support.get("external_actions", {}).values()):
        failures.append("an external action occurred")
    if any(support.get("forbidden_actions", {}).values()):
        failures.append("a forbidden action occurred")

    if preflight.get("status") != "PASS_ZERO_SWAP_GROWTH":
        failures.append("idle preflight did not pass")
    if float(preflight.get("elapsed_seconds", 0.0)) < 45.0:
        failures.append("idle preflight was shorter than 45 seconds")
    if float(preflight.get("swap_growth_mib", -1.0)) != 0.0:
        failures.append("idle preflight observed swap growth")
    if support.get("compute", {}).get("audit_swap_growth_mib") != 0.0:
        failures.append("support audit observed swap growth")
    if resource_receipt.get("status") != "PASS_CPU_CACHE_ONLY_ZERO_SWAP":
        failures.append("resource receipt did not pass")
    for key in (
        "graph_runs",
        "gpu_jobs",
        "kaggle_launches",
        "budget_debits",
        "cellect_actions",
        "runpod_jobs",
        "submissions",
    ):
        if resource_receipt.get(key) != 0:
            failures.append(f"resource receipt records nonzero {key}")

    fragment_nodes = fragment.get("nodes", [])
    if len(fragment_nodes) != 22:
        failures.append("state fragment does not contain 22 nodes")
    if fragment.get("tier_move_proposed") is not False:
        failures.append("state fragment proposes a tier move")
    if fragment.get("proposed_moves") != []:
        failures.append("state fragment proposed-move list is nonempty")
    if any(node.get("tier") != "PARK_B" for node in fragment_nodes):
        failures.append("state fragment changes a PARK_B tier")
    support_hash = sha256(OUT / "SUPPORT_FALSIFIERS_WAVE2.json")
    for node in fragment_nodes:
        if node.get("evidence_sha256") != support_hash:
            failures.append(f"fragment evidence hash mismatch for {node.get('id')}")
        if node.get("evidence_artifact") != (
            ALLOWED_PREFIX + "SUPPORT_FALSIFIERS_WAVE2.json"
        ):
            failures.append(f"fragment evidence path mismatch for {node.get('id')}")

    terminal_text = (
        "# SA-PARKB-SWEEP CPU Wave-2 terminal report\n\n"
        "Verdict: **VERDICT_BARRIER / NO TIER MOVES**.\n\n"
        "The frozen root census at commit "
        f"`{ROOT_COMMIT}` contains 22 PARK-B nodes. This lane applied the "
        "preregistered cheapest identity/support falsifier to all 14 CPU-testable "
        "nodes and explicitly skipped 8 weight-, fitting-, GPU-, or CELLECT-gated "
        "representation nodes. Every node remains PARK-B; proposed B→A moves: 0; "
        "proposed B→C moves: 0.\n\n"
        "The prior quarantine is repaired: every Wave-2 citation resolves inside the "
        f"committed root tree, and `{FORBIDDEN_PRIOR_SIBLING_CITATION}` is not cited. "
        "The three branch registers remain absent from the clean root commit, so they "
        "were ineligible as transition evidence. The root barrier's committed rejection "
        "of the earlier GAP-F1, DET-F0, and G2 moves was preserved.\n\n"
        f"Support evidence: `{ALLOWED_PREFIX}SUPPORT_FALSIFIERS_WAVE2.json` "
        f"(SHA-256 `{support_hash}`).\n\n"
        f"Resource receipt: `{ALLOWED_PREFIX}RESOURCE_RECEIPT_WAVE2.json` "
        f"(SHA-256 `{sha256(OUT / 'RESOURCE_RECEIPT_WAVE2.json')}`). The real idle "
        f"preflight ran {float(preflight['elapsed_seconds']):.3f} seconds with "
        "0 MiB swap growth; the audit also had 0 MiB swap growth. No graph, 44b6, "
        "6bba, CV57, GPU, Kaggle, budget debit, CELLECT action, RunPod, submission, "
        "DEAD action, or root single-writer mutation occurred. C2/K-best and Branch-B "
        "memo numbers remained quarantined.\n"
    )
    (OUT / "TERMINAL_REPORT_WAVE2.md").write_text(terminal_text)

    validation = {
        "schema": "celltrack-alpha.parkb-support-sweep-validation.v2",
        "status": "PASS" if not failures else "FAIL",
        "root_commit": ROOT_COMMIT,
        "park_b_nodes": 22,
        "cpu_testable": 14,
        "skipped_weight_or_gpu_gated_representation": 8,
        "proposed_moves": 0,
        "unchanged_park_b": 22,
        "citations_checked": len(citations),
        "all_citations_resolve_in_committed_root": not any(
            "citation" in failure for failure in failures
        ),
        "prior_unmerged_nodetopo_citation_absent": not any(
            "NODETOPO citation" in failure for failure in failures
        ),
        "uncited_tier_moves": 0,
        "memo_numbers_used_for_tiering": 0,
        "preflight_elapsed_seconds": preflight.get("elapsed_seconds"),
        "preflight_swap_growth_mib": preflight.get("swap_growth_mib"),
        "audit_swap_growth_mib": support.get("compute", {}).get(
            "audit_swap_growth_mib"
        ),
        "family_execution": False,
        "external_actions": False,
        "artifact_sha256": {
            "preregistration": sha256(OUT / "PREREGISTRATION_WAVE2.json"),
            "inventory": sha256(OUT / "PARKB_INVENTORY_WAVE2.json"),
            "support": support_hash,
            "state_fragment": sha256(OUT / "state_fragment.json"),
            "preflight": sha256(OUT / "IDLE_PREFLIGHT_WAVE2.json"),
            "resource_receipt": sha256(OUT / "RESOURCE_RECEIPT_WAVE2.json"),
            "proposed_ledger_rows": sha256(OUT / "PROPOSED_LEDGER_ROWS_WAVE2.md"),
            "terminal_report": sha256(OUT / "TERMINAL_REPORT_WAVE2.md"),
        },
        "failures": failures,
    }
    (OUT / "VALIDATION_WAVE2.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORKTREE,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    changed_paths = [line[3:] for line in status if line]
    own_path = ALLOWED_PREFIX + "PATH_ALLOWLIST_AUDIT_WAVE2.json"
    if own_path not in changed_paths:
        changed_paths.append(own_path)
    changed_paths = sorted(changed_paths)
    out_of_scope = [
        path for path in changed_paths if not path.startswith(ALLOWED_PREFIX)
    ]
    path_audit = {
        "schema": "celltrack-alpha.worktree-path-allowlist.v2",
        "status": "PASS" if not out_of_scope else "FAIL",
        "base_commit": ROOT_COMMIT,
        "allowed_prefix": ALLOWED_PREFIX,
        "changed_paths": changed_paths,
        "out_of_scope_paths": out_of_scope,
    }
    (OUT / "PATH_ALLOWLIST_AUDIT_WAVE2.json").write_text(
        json.dumps(path_audit, indent=2, sort_keys=True) + "\n"
    )
    if out_of_scope:
        failures.append("out-of-scope paths: " + ", ".join(out_of_scope))
    if failures:
        raise SystemExit("PARK-B Wave-2 validation failed:\n- " + "\n- ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
