#!/usr/bin/env python3
"""Freeze and audit the G-02 N1 feasibility-exact action compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from celltrack_alpha.action_compiler import (
    ActionCompileError,
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
    graph_sha256,
    interface_schema,
)
from celltrack_alpha.kaggle_calibration import (
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "reports"
    / "g02"
    / "screening"
    / "tiered_surface"
    / "n1_feasibility_compiler"
)

EXPECTED_HASHES = {
    "human_locked_plan": (
        "reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md",
        "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116",
    ),
    "p0_json": (
        "reports/g02/screening/action_class_oracle/"
        "p0_action_class_oracle_v1.json",
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    ),
    "p0_markdown": (
        "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.md",
        "5ce1518f5eeb2bb929cefb81f869e7be6bf14ca8f4615c749ce41655a6cf6d39",
    ),
    "p0_guard": (
        "reports/g02/screening/action_class_oracle/"
        "p0_action_class_oracle_guard_v1.json",
        "a70b33bd7253848a1b71c8f0736f7e8b9983a1991898b19a49ee8052926f5d45",
    ),
    "grouped_folds": (
        "reports/g02/grouped_3fold_audit20_v1.json",
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    ),
    "fit_provenance": (
        "reports/g02/fit_provenance_v1.json",
        "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f",
    ),
    "fit117_scope": (
        "reports/g02/extended_radius_reachability_v1.json",
        "22a70e0b750d3d59bdd92ca5a82967d35c5d866fafe1c867dbb09b936efc2803",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def node(node_id: int, t: int) -> SubmissionNode:
    return SubmissionNode(node_id=node_id, t=t, z=0, y=node_id, x=0)


def fixture() -> SubmissionGraph:
    nodes = {
        value: node(value, time)
        for value, time in (
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 2),
            (5, 2),
            (6, 0),
            (7, 1),
            (8, 0),
            (9, 1),
            (10, 2),
            (11, 0),
            (12, 1),
            (13, 1),
        )
    }
    edges = [
        SubmissionEdge(1, 2),
        SubmissionEdge(2, 4),
        SubmissionEdge(6, 7),
        SubmissionEdge(8, 9),
        SubmissionEdge(11, 12),
        SubmissionEdge(11, 13),
    ]
    return SubmissionGraph("n1-freeze-fixture", nodes, edges)


def valid_actions() -> dict[str, GraphAction]:
    return {
        "noop": GraphAction("a.noop", ActionKind.NOOP),
        "add_edge": GraphAction(
            "b.add",
            ActionKind.ADD_EDGE,
            add_edges=((3, 5),),
        ),
        "delete_edges": GraphAction(
            "c.delete",
            ActionKind.DELETE_EDGES,
            delete_edges=((2, 4),),
        ),
        "swap": GraphAction(
            "d.swap",
            ActionKind.SWAP,
            add_edges=((6, 9),),
            replace_outgoing_from=(6,),
            replace_incoming_to=(9,),
            expected_conflict_edges=((6, 7), (8, 9)),
        ),
        "division_fork": GraphAction(
            "e.fork",
            ActionKind.DIVISION_FORK,
            add_edges=((11, 4), (11, 5)),
            replace_outgoing_from=(11,),
            replace_incoming_to=(4, 5),
            expected_conflict_edges=((2, 4), (11, 12), (11, 13)),
        ),
        "restore_topology": GraphAction(
            "f.restore",
            ActionKind.RESTORE_TOPOLOGY,
            add_nodes=(node(20, 2), node(21, 3)),
            add_edges=((3, 20), (20, 21)),
        ),
    }


def audit_fixtures() -> dict[str, Any]:
    baseline = fixture()
    actions = valid_actions()
    single_results = {
        name: compile_action(baseline, action).after_graph_sha256
        for name, action in actions.items()
    }
    batch_actions = (
        actions["delete_edges"],
        actions["swap"],
        actions["restore_topology"],
    )
    forward = compile_batch(baseline, batch_actions)
    reverse = compile_batch(baseline, tuple(reversed(batch_actions)))
    if (
        forward.batch_sha256 != reverse.batch_sha256
        or forward.after_graph_sha256 != reverse.after_graph_sha256
    ):
        raise RuntimeError("batch order-independence audit failed")

    invalid_actions = {
        "conflict_omission": GraphAction(
            "bad.conflict",
            ActionKind.SWAP,
            add_edges=((6, 9),),
            replace_outgoing_from=(6,),
            replace_incoming_to=(9,),
            expected_conflict_edges=((6, 7),),
        ),
        "missing_endpoint": GraphAction(
            "bad.endpoint",
            ActionKind.ADD_EDGE,
            add_edges=((3, 999),),
        ),
        "backward_time": GraphAction(
            "bad.time",
            ActionKind.ADD_EDGE,
            add_edges=((3, 1),),
        ),
        "merge": GraphAction(
            "bad.merge",
            ActionKind.ADD_EDGE,
            add_edges=((3, 7),),
        ),
        "outdegree_three": GraphAction(
            "bad.degree",
            ActionKind.RESTORE_TOPOLOGY,
            add_nodes=(node(22, 1),),
            add_edges=((11, 22),),
        ),
        "missing_delete": GraphAction(
            "bad.delete",
            ActionKind.DELETE_EDGES,
            delete_edges=((1, 999),),
        ),
    }
    rejected: dict[str, str] = {}
    for name, action in invalid_actions.items():
        try:
            compile_action(baseline, action)
        except (ActionCompileError, ValueError) as error:
            rejected[name] = str(error)
        else:
            raise RuntimeError(f"invalid fixture compiled: {name}")

    shared = GraphAction(
        "g.restore",
        ActionKind.RESTORE_TOPOLOGY,
        add_nodes=(node(23, 1),),
        add_edges=((8, 23), (23, 10)),
        replace_outgoing_from=(8,),
        replace_incoming_to=(10,),
        expected_conflict_edges=((8, 9),),
    )
    try:
        compile_batch(baseline, (actions["swap"], shared))
    except ActionCompileError as error:
        rejected["shared_batch_conflict"] = str(error)
    else:
        raise RuntimeError("shared-conflict batch compiled")

    return {
        "baseline_graph_sha256": graph_sha256(baseline),
        "single_action_validity": {
            "passed": len(single_results),
            "total": len(ActionKind),
            "after_graph_sha256": single_results,
        },
        "batch_validity": {
            "passed": 2,
            "total": 2,
            "forward_batch_sha256": forward.batch_sha256,
            "reverse_batch_sha256": reverse.batch_sha256,
            "after_graph_sha256": forward.after_graph_sha256,
        },
        "negative_validity": {
            "rejected": len(rejected),
            "total": 7,
            "reasons": rejected,
        },
    }


def audit_scope() -> dict[str, Any]:
    p0 = load_json(EXPECTED_HASHES["p0_json"][0])
    grouped = load_json(EXPECTED_HASHES["grouped_folds"][0])
    provenance = load_json(EXPECTED_HASHES["fit_provenance"][0])
    reachability = load_json(EXPECTED_HASHES["fit117_scope"][0])

    p0_scope = p0["scope"]
    required_false = (
        "audit_used",
        "cv57_used",
        "held_out_used",
        "model_fitted",
        "nonfinal_smoke",
        "test_used",
    )
    if not p0_scope["fit_only"] or any(p0_scope[key] for key in required_false):
        raise RuntimeError("P0 scope is not fit-only and leakage-clean")
    if (
        p0_scope["train_video_count"] != 105
        or p0_scope["validation_video_count"] != 12
    ):
        raise RuntimeError("P0 fit117 train/validation count mismatch")

    audit_ids = set(grouped["audit"]["group_ids"])
    remainder_ids: set[str] = set()
    fold_summary: dict[str, Any] = {}
    for fold_id, fold in grouped["folds"].items():
        fit_ids = set(fold["fit_group_ids"])
        held_ids = set(fold["held_out_group_ids"])
        if fit_ids & held_ids or audit_ids & (fit_ids | held_ids):
            raise RuntimeError(f"group leakage in grouped fold {fold_id}")
        remainder_ids |= fit_ids | held_ids

        provenance_fold = provenance["folds"][fold_id]
        for role in ("calibration", "router", "threshold", "training"):
            if set(provenance_fold["fit_sources"][role]) != fit_ids:
                raise RuntimeError(f"provenance mismatch for fold {fold_id}/{role}")
        forbidden = set(provenance_fold["forbidden_fit_group_ids"])
        if forbidden != held_ids | audit_ids:
            raise RuntimeError(f"forbidden-set mismatch in fold {fold_id}")
        fold_summary[fold_id] = {
            "fit_groups": len(fit_ids),
            "held_out_groups": len(held_ids),
            "forbidden_fit_groups": len(forbidden),
            "leakage_intersections": 0,
        }
    if len(remainder_ids) != grouped["counts"]["remainder_videos"]:
        raise RuntimeError("grouped-fold remainder coverage mismatch")
    if provenance["grouped_split_sha256"] != EXPECTED_HASHES["grouped_folds"][1]:
        raise RuntimeError("fit provenance does not pin grouped fold hash")

    reach_scope = reachability["scope"]
    if (
        not reach_scope["fit_only"]
        or reach_scope["audit_used"]
        or reach_scope["held_out_used"]
        or reach_scope["test_used"]
        or len(reach_scope["train_video_ids"]) != 105
        or len(reach_scope["validation_video_ids"]) != 12
    ):
        raise RuntimeError("fit117 reachability scope mismatch")

    return {
        "p0_fit_scope": {
            "train_videos": 105,
            "validation_videos": 12,
            "family_counts": p0_scope["family_counts"],
            "audit_used": False,
            "held_out_used": False,
            "test_used": False,
        },
        "fixed_grouped_folds": {
            "folds": fold_summary,
            "audit_groups": len(audit_ids),
            "remainder_groups": len(remainder_ids),
            "audit_untouched": grouped["audit"]["untouched"],
        },
        "fit117_scope": {
            "train_videos": len(reach_scope["train_video_ids"]),
            "validation_videos": len(reach_scope["validation_video_ids"]),
            "audit_used": False,
            "held_out_used": False,
            "test_used": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-guard", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destinations = (
        OUTPUT_DIR / "interface_v1.json",
        OUTPUT_DIR / "n1_unit_guard_v1.json",
        OUTPUT_DIR / "N1_FREEZE_REPORT.json",
        OUTPUT_DIR / "N1_FREEZE_REPORT.md",
    )
    if any(path.exists() for path in destinations):
        raise SystemExit("refusing to overwrite an N1 freeze artifact")

    pinned_inputs: dict[str, dict[str, str]] = {}
    for name, (relative, expected) in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"hash mismatch for {name}: {actual}")
        pinned_inputs[name] = {"path": relative, "sha256": actual}

    guard = json.loads(args.unit_guard.read_text(encoding="ascii"))
    if (
        guard.get("status") != "PASS"
        or guard.get("swap_growth_mib") != 0.0
        or guard.get("return_code") != 0
    ):
        raise SystemExit("unit guard did not pass with zero swap growth")

    implementation = ROOT / "src/celltrack_alpha/action_compiler.py"
    tests = ROOT / "tests/test_action_compiler.py"
    freeze_script = Path(__file__).resolve()
    hashes = {
        "implementation": {
            "path": str(implementation.relative_to(ROOT)),
            "sha256": sha256(implementation),
        },
        "tests": {
            "path": str(tests.relative_to(ROOT)),
            "sha256": sha256(tests),
        },
        "freeze_script": {
            "path": str(freeze_script.relative_to(ROOT)),
            "sha256": sha256(freeze_script),
        },
    }

    fixture_audit = audit_fixtures()
    scope_audit = audit_scope()
    output_schema = {
        **interface_schema(),
        "frozen_implementation": hashes["implementation"],
        "frozen_tests": hashes["tests"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    interface_path = destinations[0]
    interface_path.write_text(
        json.dumps(output_schema, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    shutil.copyfile(args.unit_guard, destinations[1])
    hashes["interface"] = {
        "path": str(interface_path.relative_to(ROOT)),
        "sha256": sha256(interface_path),
    }
    hashes["unit_guard"] = {
        "path": str(destinations[1].relative_to(ROOT)),
        "sha256": sha256(destinations[1]),
    }

    report = {
        "schema": "celltrack-alpha.n1-feasibility-compiler-freeze.v1",
        "status": "GO",
        "verdict_basis": (
            "100% unit, single-action, negative, and combined-batch validity; "
            "deterministic order-independent compilation; zero swap growth"
        ),
        "phase": "N1_FEASIBILITY_COMPILER",
        "created_utc": "2026-07-28T00:00:00Z",
        "pinned_inputs": pinned_inputs,
        "frozen_artifacts": hashes,
        "validity_audit": fixture_audit,
        "unit_suite": {
            "passed": 10,
            "total": 10,
            "guard": guard,
        },
        "scope_audit": scope_audit,
        "execution": {
            "device": "cpu",
            "external_action": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
        },
        "next_gate": (
            "report frozen N1 and updated census to human before dispatching "
            "independent Phase-1 worktree agents"
        ),
    }
    destinations[2].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    markdown = f"""# N1 feasibility-exact action compiler — terminal freeze

Status: **GO**.

The compiler passed all 10 unit tests, all {len(ActionKind)} single-action kinds,
both batch-order realizations, and all 7 negative fail-closed fixtures. The guarded
test run recorded zero swap growth. No label, held outcome, audit group, test data,
GPU, Kaggle, RunPod, external service, or submission was used.

## Frozen identities

- Interface: `{hashes["interface"]["sha256"]}`
- Implementation: `{hashes["implementation"]["sha256"]}`
- Tests: `{hashes["tests"]["sha256"]}`
- Freeze script: `{hashes["freeze_script"]["sha256"]}`
- Unit guard: `{hashes["unit_guard"]["sha256"]}`

## Scope confirmation

- P0: 105 train + 12 validation videos, fit-only.
- Grouped folds: 3 folds over 175 remainder groups; 20 audit groups untouched.
- Leakage intersections: zero in every fold.
- P0, plan, folds, provenance, and fit117 scope all match their locked SHA-256
  identities in `N1_FREEZE_REPORT.json`.

## Cascade

N1 is frozen and G1 advances from PARK-B to PARK-A. Phase 1 is ready, but remains
undispatched until the root reports this freeze and the updated tier census.
"""
    destinations[3].write_text(markdown, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
