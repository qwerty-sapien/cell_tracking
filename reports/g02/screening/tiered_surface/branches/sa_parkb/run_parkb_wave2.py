#!/usr/bin/env python3
"""Deterministic CPU/cache-only PARK-B support and identity audit."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import re
import resource
import subprocess
import time
from typing import Any


FROZEN_ROOT = pathlib.Path("/Users/agent/Desktop/cell_tracking")
OUT = pathlib.Path(__file__).resolve().parent
WORKTREE = OUT.parents[5]
ROOT_COMMIT = "5a805c2f7240e30c572644d3d5f0e733add04769"
RESULT_REL = (
    "reports/g02/screening/tiered_surface/branches/sa_parkb/"
    "SUPPORT_FALSIFIERS_WAVE2.json"
)
EXPECTED_SHARED = {
    "reports/g02/screening/tiered_surface/ALIGNMENT.md":
        "eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37",
    "reports/g02/screening/tiered_surface/SCOPE_MANIFEST.md":
        "066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49",
    "reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md":
        "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116",
    "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json":
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    "reports/g02/grouped_3fold_audit20_v1.json":
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json":
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
}
COMMITTED_EVIDENCE = {
    "tier_census": {
        "path": "reports/g02/screening/tiered_surface/tier_census.json",
        "sha256": "83a00a886116912c0aa15e1e0073d45f89cbf0da4b4ab1c3e465c09206643003",
    },
    "barrier_decisions": {
        "path": "reports/g02/screening/tiered_surface/wave2_barrier_decisions.json",
        "sha256": "9fed2429b16d1d286dc722c78e74842ecae0b65577608b100250d0dc4c008f8d",
    },
    "division_refinement": {
        "path": (
            "reports/g02/screening/tiered_surface/branches/sa_division/"
            "oracle_refinement_results.json"
        ),
        "sha256": "aded494fd19aca50a9123dfdbcccfd37448709330f8449e76984359833c44c22",
    },
    "nodetopo_terminal": {
        "path": (
            "reports/g02/screening/tiered_surface/branches/sa_nodetopo/"
            "RAW_UNIVERSE_TERMINAL_v2.json"
        ),
        "sha256": "54e6babd814e1c84935710db47f758edf8b2886e32f6e8f2565d31f42478855d",
    },
    "representation_audit": {
        "path": (
            "reports/g02/screening/tiered_surface/branches/sa_repr_multi/"
            "SUPPLIER_PROVENANCE_AUDIT.json"
        ),
        "sha256": "64492bcc0278f6d609cf4b2a72a12109b380842aed963847aaa830991962b1e7",
    },
}
CPU_TESTABLE = [
    "AE-P5",
    "AE-P6",
    "AE-P7",
    "DIV-F2",
    "GAP-F1",
    "NODERESTORE-O0",
    "DET-F0",
    "DET-F2",
    "DET-F3",
    "R2",
    "R3",
    "R8",
    "G2",
    "G6",
]
SKIPPED_REPRESENTATION = [
    "AE-P8",
    "AE-P9",
    "R4",
    "R5",
    "R6",
    "R7",
    "R9",
    "R10",
]
REGISTER_PATHS = [
    "reports/g02/screening/tiered_surface/branches/association_expansion.md",
    "reports/g02/screening/tiered_surface/branches/event_node_actions.md",
    "reports/g02/screening/tiered_surface/branches/representation_and_governance.md",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_bytes(data: bytes) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def committed_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{ROOT_COMMIT}:{path}"],
        cwd=WORKTREE,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def committed_exists(path: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{ROOT_COMMIT}:{path}"],
            cwd=WORKTREE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def citation(name: str) -> dict[str, str]:
    item = COMMITTED_EVIDENCE[name]
    return {
        "root_commit": ROOT_COMMIT,
        "path": item["path"],
        "sha256": item["sha256"],
    }


def swap_used_mib() -> float:
    if platform.system() == "Darwin":
        output = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        match = re.search(r"used = ([0-9.]+)([MG])", output)
        if match is None:
            raise RuntimeError(f"cannot parse vm.swapusage: {output!r}")
        value = float(match.group(1))
        return value * 1024.0 if match.group(2) == "G" else value
    meminfo = pathlib.Path("/proc/meminfo").read_text()
    total = float(re.search(r"SwapTotal:\s+(\d+)", meminfo).group(1))
    free = float(re.search(r"SwapFree:\s+(\d+)", meminfo).group(1))
    return (total - free) / 1024.0


def memory_free_percent() -> float | None:
    if platform.system() == "Darwin":
        output = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        page_size_match = re.search(r"page size of (\d+) bytes", output)
        if page_size_match is None:
            return None
        values = {
            key: int(value)
            for key, value in re.findall(r"^([^:]+):\s+(\d+)\.", output, re.MULTILINE)
        }
        memory_bytes = int(
            subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
        )
        total_pages = memory_bytes // int(page_size_match.group(1))
        available_pages = (
            values.get("Pages free", 0)
            + values.get("Pages inactive", 0)
            + values.get("Pages speculative", 0)
            + values.get("Pages purgeable", 0)
        )
        return 100.0 * available_pages / total_pages if total_pages else None
    meminfo = pathlib.Path("/proc/meminfo").read_text()
    total = float(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1))
    available = float(re.search(r"MemAvailable:\s+(\d+)", meminfo).group(1))
    return 100.0 * available / total


def run_idle_preflight() -> dict[str, Any]:
    started_wall = time.time()
    started = time.monotonic()
    initial_swap = swap_used_mib()
    peak_swap = initial_swap
    min_free = memory_free_percent()
    samples = 1
    while time.monotonic() - started < 45.0:
        time.sleep(1.0)
        current_swap = swap_used_mib()
        peak_swap = max(peak_swap, current_swap)
        current_free = memory_free_percent()
        if current_free is not None:
            min_free = current_free if min_free is None else min(min_free, current_free)
        samples += 1
    elapsed = time.monotonic() - started
    final_swap = swap_used_mib()
    peak_swap = max(peak_swap, final_swap)
    growth = peak_swap - initial_swap
    receipt = {
        "schema": "celltrack-alpha.local-idle-preflight.v2",
        "status": "PASS_ZERO_SWAP_GROWTH" if growth == 0.0 else "FAIL_SWAP_GROWTH",
        "started_epoch_seconds": started_wall,
        "elapsed_seconds": elapsed,
        "required_idle_seconds": 45.0,
        "samples": samples,
        "initial_swap_mib": initial_swap,
        "peak_swap_mib": peak_swap,
        "final_swap_mib": final_swap,
        "swap_growth_mib": growth,
        "minimum_memory_free_percent": min_free,
        "load_average_before_audit": list(os.getloadavg()),
    }
    path = OUT / "IDLE_PREFLIGHT_WAVE2.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if growth != 0.0 or elapsed < 45.0:
        raise SystemExit("idle preflight failed")
    return receipt


def validate_inputs() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for relative, expected in EXPECTED_SHARED.items():
        path = FROZEN_ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"frozen input mismatch: {relative}: {actual} != {expected}")
        observed[relative] = {
            "sha256": actual,
            "read_only_root": str(FROZEN_ROOT),
            "pass": True,
        }
    for name, evidence in COMMITTED_EVIDENCE.items():
        data = committed_bytes(evidence["path"])
        actual = sha256_bytes(data)
        if actual != evidence["sha256"]:
            raise SystemExit(
                f"committed evidence mismatch: {name}: {actual} != {evidence['sha256']}"
            )
        observed[f"{ROOT_COMMIT}:{evidence['path']}"] = {
            "sha256": actual,
            "committed": True,
            "pass": True,
        }
    return observed


def node_result(
    node_id: str,
    falsifier: str,
    outcome: str,
    citations: list[dict[str, str]],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": node_id,
        "classification": "CPU_TESTABLE_IDENTITY_OR_SUPPORT",
        "falsifier": falsifier,
        "outcome": outcome,
        "from_tier": "PARK_B",
        "proposed_tier": "PARK_B",
        "tier_move_proposed": False,
        "citations": citations,
        "memo_numbers_used": False,
    }
    if details:
        result["details"] = details
    return result


def run_support_audit(preflight: dict[str, Any]) -> None:
    started = time.monotonic()
    initial_swap = swap_used_mib()
    input_receipt = validate_inputs()
    census = load_json_bytes(committed_bytes(COMMITTED_EVIDENCE["tier_census"]["path"]))
    barrier = load_json_bytes(
        committed_bytes(COMMITTED_EVIDENCE["barrier_decisions"]["path"])
    )
    division = load_json_bytes(
        committed_bytes(COMMITTED_EVIDENCE["division_refinement"]["path"])
    )
    nodetopo = load_json_bytes(
        committed_bytes(COMMITTED_EVIDENCE["nodetopo_terminal"]["path"])
    )
    representation = load_json_bytes(
        committed_bytes(COMMITTED_EVIDENCE["representation_audit"]["path"])
    )
    park_b = [node for node in census["nodes"] if node["tier"] == "PARK_B"]
    ids = [node["id"] for node in park_b]
    if len(ids) != 22 or set(ids) != set(CPU_TESTABLE + SKIPPED_REPRESENTATION):
        raise SystemExit(f"current PARK_B universe mismatch: {ids}")

    register_availability = {
        path: {
            "present_in_root_commit": committed_exists(path),
            "eligible_as_transition_evidence": committed_exists(path),
        }
        for path in REGISTER_PATHS
    }
    if any(item["present_in_root_commit"] for item in register_availability.values()):
        raise SystemExit("expected clean-checkout branch-register packaging gap changed")

    barrier_by_node = {
        row["node"]: row for row in barrier.get("proposal_decisions", [])
    }
    if set(barrier_by_node) != {"GAP-F1", "DET-F0", "G2"}:
        raise SystemExit("unexpected root barrier proposal-decision set")
    supplier_by_node = {
        row["node"]: row for row in representation.get("suppliers", [])
    }

    results = [
        node_result(
            "AE-P5",
            "Find a committed node-specific C0 edge/no-link assignment-cost and MAP identity contract.",
            "PARK_MISSING_COMMITTED_NODE_CONTRACT",
            [citation("tier_census")],
            {"committed_association_register_present": False},
        ),
        node_result(
            "AE-P6",
            "Find a committed immutable forward/reverse endpoint-return contract and disabled-cycle control.",
            "PARK_MISSING_COMMITTED_NODE_CONTRACT",
            [citation("tier_census")],
            {"committed_association_register_present": False},
        ),
        node_result(
            "AE-P7",
            "Find a committed phase/template tracklet identity join frozen without held labels.",
            "PARK_MISSING_COMMITTED_NODE_CONTRACT",
            [citation("tier_census")],
            {"committed_association_register_present": False},
        ),
        node_result(
            "DIV-F2",
            "Check committed both-family association-continuity qualification and authentic fork-input readiness.",
            "PARK_DEPENDENCY",
            [citation("tier_census"), citation("division_refinement")],
            {
                "qualifying_association_arm_count": division["dependency"][
                    "qualifying_arm_count"
                ],
                "fitting_gate_open": division["dependency"]["fitting_gate_open"],
                "tier_move_in_evidence": division["conclusion"]["tier_move_proposed"],
            },
        ),
        node_result(
            "GAP-F1",
            "Require a committed named bridge-realizer result; reject P0 class headroom as a substitute.",
            "PARK_PRIOR_MOVE_REJECTED_NO_REALIZER_RESULT",
            [citation("tier_census"), citation("barrier_decisions")],
            {"root_barrier_reason": barrier_by_node["GAP-F1"]["reason"]},
        ),
        node_result(
            "NODERESTORE-O0",
            "Require a committed complete authentic restoration universe under the zero-swap guard.",
            "PARK_RESOURCE_GUARD_NO_SCIENTIFIC_RESULT",
            [citation("tier_census"), citation("nodetopo_terminal")],
            {
                "terminal_verdict": nodetopo["verdict"],
                "completed_video_count": nodetopo["completed_video_count"],
                "raw_universe_complete": nodetopo["raw_universe_complete"],
                "scientific_result": nodetopo["scientific_result"],
            },
        ),
        node_result(
            "DET-F0",
            "Require a committed named detector/restoration result, not a free-node P0 class value.",
            "PARK_PRIOR_MOVE_REJECTED_NO_AUTHENTIC_UNIVERSE",
            [
                citation("tier_census"),
                citation("barrier_decisions"),
                citation("nodetopo_terminal"),
            ],
            {"root_barrier_reason": barrier_by_node["DET-F0"]["reason"]},
        ),
        node_result(
            "DET-F2",
            "Require a committed complete authentic proposal universe before learned detection expansion.",
            "PARK_AUTHENTIC_UNIVERSE_INCOMPLETE",
            [citation("tier_census"), citation("nodetopo_terminal")],
            {
                "completed_video_count": nodetopo["completed_video_count"],
                "raw_universe_complete": nodetopo["raw_universe_complete"],
            },
        ),
        node_result(
            "DET-F3",
            "Require committed authentic proposals plus a topology-complete node-action join.",
            "PARK_AUTHENTIC_UNIVERSE_INCOMPLETE",
            [citation("tier_census"), citation("nodetopo_terminal")],
            {
                "completed_video_count": nodetopo["completed_video_count"],
                "raw_universe_complete": nodetopo["raw_universe_complete"],
            },
        ),
        node_result(
            "R2",
            "Authenticate complete real-mask/wrfeat provenance, coverage, and identity join.",
            "PARK_AUTHENTIC_INPUT_NOT_READY",
            [citation("tier_census"), citation("representation_audit")],
            {
                "authentication": supplier_by_node["R2"]["authentication"],
                "reason": supplier_by_node["R2"]["reason"],
            },
        ),
        node_result(
            "R3",
            "Require committed R2 authentication before opening real-wrfeat support.",
            "PARK_DEPENDENCY_R2",
            [citation("tier_census"), citation("representation_audit")],
            {
                "authentication": supplier_by_node["R3"]["authentication"],
                "reason": supplier_by_node["R3"]["reason"],
            },
        ),
        node_result(
            "R8",
            "Require a committed immutable reverse supplier and endpoint-return contract; do not run reverse inference here.",
            "PARK_REVERSE_SUPPLIER_NOT_READY",
            [citation("tier_census"), citation("representation_audit")],
            {
                "authentication": supplier_by_node["R8"]["authentication"],
                "reason": supplier_by_node["R8"]["reason"],
                "reverse_inference_run": False,
            },
        ),
        node_result(
            "G2",
            "Require a committed common both-family graph pass plus positive exact regret versus no-op.",
            "PARK_PRIOR_MOVE_REJECTED_NO_GRAPH_OR_REGRET_RESULT",
            [citation("tier_census"), citation("barrier_decisions")],
            {"root_barrier_reason": barrier_by_node["G2"]["reason"]},
        ),
        node_result(
            "G6",
            "Verify the owner RunPod prohibition remains closed without execution.",
            "PARK_OWNER_FORBIDS_RUNPOD",
            [citation("tier_census")],
            {"runpod_action": False, "scientific_transition": False},
        ),
    ]
    if [row["id"] for row in results] != CPU_TESTABLE:
        raise SystemExit("CPU-testable result ordering drifted from preregistration")

    skipped = []
    skip_reasons = {
        "AE-P8": "Dense microscopy embedding node requires authenticated encoder weights and representation compute.",
        "AE-P9": "PEFT adapter node requires an authenticated frozen backbone and fitting.",
        "R4": supplier_by_node["R4"]["reason"],
        "R5": supplier_by_node["R5"]["reason"],
        "R6": supplier_by_node["R6"]["reason"],
        "R7": "CELLECT remains on its separately governed packaging/authorization track; this CPU lane cannot touch it.",
        "R9": "Group-robust tracklet representation requires new frozen event evidence and fitting.",
        "R10": supplier_by_node["R10"]["reason"],
    }
    for node_id in SKIPPED_REPRESENTATION:
        citations = [citation("tier_census")]
        if node_id.startswith("R"):
            citations.append(citation("representation_audit"))
        skipped.append(
            {
                "id": node_id,
                "classification": "SKIP_WEIGHT_OR_GPU_GATED_REPRESENTATION",
                "reason": skip_reasons[node_id],
                "from_tier": "PARK_B",
                "proposed_tier": "PARK_B",
                "tier_move_proposed": False,
                "citations": citations,
                "gpu_requested": False,
                "cellect_touched": False,
            }
        )

    all_records = {row["id"]: row for row in results + skipped}
    inventory_nodes = []
    for node in park_b:
        record = all_records[node["id"]]
        inventory_nodes.append(
            {
                "id": node["id"],
                "branch": node["branch"],
                "classification": record["classification"],
                "from_tier": "PARK_B",
                "proposed_tier": "PARK_B",
                "tier_move_proposed": False,
                "census_evidence_artifact": node["evidence_artifact"],
                "census_evidence_sha256": node["evidence_sha256"],
                "wave2_citations": record["citations"],
            }
        )
    inventory = {
        "schema": "celltrack-alpha.parkb-inventory.v2",
        "status": "COMPLETE",
        "root_commit": ROOT_COMMIT,
        "root_census": citation("tier_census"),
        "current_park_b_count": len(inventory_nodes),
        "cpu_testable_count": len(results),
        "skipped_weight_or_gpu_gated_representation_count": len(skipped),
        "nodes": inventory_nodes,
    }
    inventory_path = OUT / "PARKB_INVENTORY_WAVE2.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    final_swap = swap_used_mib()
    audit_swap_growth = final_swap - initial_swap
    support = {
        "schema": "celltrack-alpha.parkb-support-sweep.v2",
        "status": "VERDICT_BARRIER",
        "owner_agent": "SA-PARKB-SWEEP",
        "root_commit": ROOT_COMMIT,
        "input_authentication": input_receipt,
        "committed_register_packaging_audit": register_availability,
        "inventory": {
            "park_b_total": 22,
            "cpu_testable": 14,
            "skipped_weight_or_gpu_gated_representation": 8,
        },
        "cpu_results": results,
        "skipped_representation": skipped,
        "proposed_moves": [],
        "unchanged_park_b": 22,
        "transition_evidence_policy": {
            "path_and_sha_required": True,
            "committed_root_required": True,
            "uncited_moves": 0,
            "prior_quarantined_fragment_used_for_transition": False,
            "sibling_live_or_unmerged_output_used": False,
        },
        "memo_quarantine": {
            "C2_or_K_best_numbers": True,
            "branch_b_66_of_223": True,
            "numbers_used_for_tiering": 0,
        },
        "family_execution": {"44b6": False, "6bba": False, "CV57": False},
        "compute": {
            "preflight_status": preflight["status"],
            "preflight_elapsed_seconds": preflight["elapsed_seconds"],
            "preflight_swap_growth_mib": preflight["swap_growth_mib"],
            "audit_elapsed_seconds": time.monotonic() - started,
            "audit_initial_swap_mib": initial_swap,
            "audit_final_swap_mib": final_swap,
            "audit_swap_growth_mib": audit_swap_growth,
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 * 1024),
        },
        "external_actions": {
            "gpu": False,
            "kaggle": False,
            "budget_debit": False,
            "cellect": False,
            "submission": False,
            "runpod": False,
        },
        "forbidden_actions": {
            "graph_run": False,
            "cv57": False,
            "dead_promotion": False,
            "root_single_writer_mutation": False,
        },
    }
    if audit_swap_growth != 0.0:
        raise SystemExit(f"swap grew during support audit: {audit_swap_growth} MiB")
    results_path = OUT / "SUPPORT_FALSIFIERS_WAVE2.json"
    results_path.write_text(json.dumps(support, indent=2, sort_keys=True) + "\n")
    results_sha = sha256(results_path)

    fragment_nodes = []
    for node in park_b:
        fragment_node = dict(node)
        fragment_node["owner_agent"] = "SA-PARKB-SWEEP"
        fragment_node["tier"] = "PARK_B"
        fragment_node["evidence_artifact"] = RESULT_REL
        fragment_node["evidence_sha256"] = results_sha
        fragment_node["evidence_basis"] = all_records[node["id"]]["outcome"] if (
            "outcome" in all_records[node["id"]]
        ) else all_records[node["id"]]["classification"]
        fragment_node["last_change_utc"] = "2026-07-28T09:57:54Z"
        fragment_nodes.append(fragment_node)
    fragment = {
        "schema": "celltrack-alpha.phase1-state-fragment.v1",
        "schema_version": 1,
        "updated_utc": "2026-07-28T09:57:54Z",
        "owner_agent": "SA-PARKB-SWEEP",
        "branch": "codex/sa-parkb-deep-wave2",
        "root_commit": ROOT_COMMIT,
        "verdict": "VERDICT_BARRIER",
        "tier_move_proposed": False,
        "proposed_moves": [],
        "nodes": fragment_nodes,
        "inventory": {"total": 22, "cpu_testable": 14, "skipped": 8},
        "family_execution": {"44b6_ran": False, "6bba_ran": False, "cv57_ran": False},
        "memo_quarantine": True,
        "uncited_tier_moves": [],
        "forbidden_scope": {
            "census_write": False,
            "live_state_write": False,
            "ledger_write": False,
            "budget_write": False,
            "gpu_action": False,
            "cellect_action": False,
            "submission_action": False,
            "runpod_action": False,
            "dead_promotion": False,
            "plan_amendment": False,
        },
    }
    (OUT / "state_fragment.json").write_text(
        json.dumps(fragment, indent=2, sort_keys=True) + "\n"
    )

    receipt = {
        "schema": "celltrack-alpha.parkb-resource-receipt.v2",
        "status": "PASS_CPU_CACHE_ONLY_ZERO_SWAP",
        "idle_preflight": {
            "path": (
                "reports/g02/screening/tiered_surface/branches/sa_parkb/"
                "IDLE_PREFLIGHT_WAVE2.json"
            ),
            "sha256": sha256(OUT / "IDLE_PREFLIGHT_WAVE2.json"),
            "elapsed_seconds": preflight["elapsed_seconds"],
            "swap_growth_mib": preflight["swap_growth_mib"],
        },
        "support_audit": {
            "path": RESULT_REL,
            "sha256": results_sha,
            "elapsed_seconds": support["compute"]["audit_elapsed_seconds"],
            "swap_growth_mib": audit_swap_growth,
            "max_rss_mib": support["compute"]["max_rss_mib"],
        },
        "compute": "CPU_CACHE_ONLY",
        "graph_runs": 0,
        "gpu_jobs": 0,
        "kaggle_launches": 0,
        "budget_debits": 0,
        "cellect_actions": 0,
        "runpod_jobs": 0,
        "submissions": 0,
    }
    (OUT / "RESOURCE_RECEIPT_WAVE2.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (OUT / "PROPOSED_LEDGER_ROWS_WAVE2.md").write_text(
        "# SA-PARKB-SWEEP Wave-2 proposed ledger rows\n\n"
        "| lane | inventory | CPU falsifiers | skipped representation | proposed moves | verdict | evidence |\n"
        "|---|---:|---:|---:|---:|---|---|\n"
        f"| SA-PARKB-SWEEP | 22 | 14 | 8 | 0 | VERDICT_BARRIER; all remain PARK-B | "
        f"`{RESULT_REL}` (`{results_sha}`) |\n\n"
        "Root adjudication: append only if the barrier accepts this fragment. No tier move, "
        "graph gate, DEAD action, budget debit, CELLECT action, or external action is proposed.\n"
    )


def main() -> int:
    prereg = OUT / "PREREGISTRATION_WAVE2.json"
    if not prereg.is_file():
        raise SystemExit("Wave-2 preregistration must exist before preflight")
    prereg_value = json.loads(prereg.read_text())
    if prereg_value.get("status") != "FROZEN_BEFORE_WAVE2_SUPPORT_OUTCOMES":
        raise SystemExit("Wave-2 preregistration is not frozen")
    preflight = run_idle_preflight()
    run_support_audit(preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
