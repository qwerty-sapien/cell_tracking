#!/usr/bin/env python3
"""Dependency-safe SA-DIVISION audit.

This module performs no fitting and no held-family scoring.  It authenticates the
frozen dependency evidence, replays synthetic N1 fork/no-op structure, and audits
the already-banked D1 official-fixture classifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
from typing import Any


ROOT = Path("/Users/agent/Desktop/cell_tracking")
ROOT_SRC = ROOT / "src"
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionCompileError,
    ActionKind,
    GraphAction,
    compile_action,
    graph_sha256,
    interface_schema,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)


PINNED = {
    "n1_interface": (
        ROOT
        / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    ),
    "n1_implementation": (
        ROOT / "src/celltrack_alpha/action_compiler.py",
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    ),
    "grouped_folds": (
        ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    ),
    "p0": (
        ROOT
        / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    ),
    "fit_provenance": (
        ROOT / "reports/g02/fit_provenance_v1.json",
        "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f",
    ),
    "d1_preregistration": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N0_preregistration.json",
        "34b2aa020099e14ad7a73820d97e9baefadec386e0e2d9acbd7ab5c1cb07620f",
    ),
    "d1_arm_manifest": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N1_arm_manifest.json",
        "327af0d29dbde19172a61bf396b951d150040bb357c3fbba90e79249035e4776",
    ),
    "d1_dry_validation": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N1_dry_validation.json",
        "f6a96d868e6ad2a21aea123157bc9bddc169e8453c117e829495d933514deaf9",
    ),
    "d1_dependency": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N2_association_dependency.md",
        "9ead4f6dff91c5eadf651ee711a3d73bea31b41f0528d7393a116f0a79418113",
    ),
    "h1_terminal_graph": (
        ROOT / "reports/g02/screening/ga_h1_zero_wrfeat_graph/report.json",
        "94a3cd006cd40aeb45f1773f6e198767133c3724eed720f573b1e3acd6851c9a",
    ),
    "c1_terminal_graph": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/branches/c1_ta_hoct_lite/"
        "outputs/c1_n3_44b6_graph.json",
        "6a202b209c9c1b7851ca1e43e4c67e5dc3772e78426fdc267f1c01cfe5953a10",
    ),
    "u0_terminal": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "u0_t4canon/terminal_report.md",
        "c29d52abd4b48c349818b0e761d8fd2b338da3e23b49164d31c86e11486f0151",
    ),
    "c2_terminal": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "c2_kbest_stability/terminal_report.md",
        "259edb515125bd5616c3df8fcc07c4aa93a35afed895fbb61151612378a0b0a6",
    ),
}

EXPECTED_BASELINE_HASH = "fe992a9ed9abd31f615b8832a90d5cb4a64869d7cfe3d0eac53b317cc4e7a235"
EXPECTED_NOOP_HASH = EXPECTED_BASELINE_HASH
EXPECTED_FORK_HASH = "a55dabfad74baca7b765959a317542aad8958d9a645052a496d20fb6f2305857"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def node(node_id: int, t: int) -> SubmissionNode:
    return SubmissionNode(node_id=node_id, t=t, z=0, y=node_id, x=0)


def n1_fixture() -> SubmissionGraph:
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


def fork_action() -> GraphAction:
    return GraphAction(
        "e.fork",
        ActionKind.DIVISION_FORK,
        add_edges=((11, 4), (11, 5)),
        replace_outgoing_from=(11,),
        replace_incoming_to=(4, 5),
        expected_conflict_edges=((2, 4), (11, 12), (11, 13)),
    )


def dependency_inventory() -> list[dict[str, Any]]:
    h1 = load_json(PINNED["h1_terminal_graph"][0])
    h1_44b6 = h1["families"][0]
    if h1_44b6["family"] != "44b6" or h1["six_bba_executed"]:
        raise AssertionError("H1 terminal family execution drift")

    c1 = load_json(PINNED["c1_terminal_graph"][0])
    if c1["scope"]["family"] != "44b6" or c1["scope"]["cv57_access"]:
        raise AssertionError("C1 terminal family execution drift")
    c1_decision = c1["decision"]

    return [
        {
            "arm": "GA-H1-ZWRFEAT-GRAPH/points-only-zero-wrfeat-12",
            "artifact": str(PINNED["h1_terminal_graph"][0].relative_to(ROOT)),
            "sha256": PINNED["h1_terminal_graph"][1],
            "44b6": {
                "adjusted": h1_44b6["adjusted_edge_jaccard_delta"],
                "raw": h1_44b6["raw_edge_jaccard_delta"],
                "node": h1_44b6["node_recall_delta"],
            },
            "6bba": {"adjusted": None, "raw": None, "node": None},
            "eligible": False,
            "reason": "44b6 regressed; 6bba not run",
        },
        {
            "arm": "C1-TA-HOCT-LITE",
            "artifact": str(PINNED["c1_terminal_graph"][0].relative_to(ROOT)),
            "sha256": PINNED["c1_terminal_graph"][1],
            "44b6": {
                "adjusted": c1_decision["adjusted_edge_jaccard_delta_vs_apncc"],
                "raw": c1_decision["raw_edge_jaccard_delta_vs_apncc"],
                "node": c1_decision["node_recall_delta_vs_apncc"],
            },
            "6bba": {"adjusted": None, "raw": None, "node": None},
            "eligible": False,
            "reason": "44b6 regressed; 6bba not run",
        },
        {
            "arm": "U0-T4CANON",
            "artifact": str(PINNED["u0_terminal"][0].relative_to(ROOT)),
            "sha256": PINNED["u0_terminal"][1],
            "44b6": {"adjusted": None, "raw": None, "node": None},
            "6bba": {"adjusted": None, "raw": None, "node": None},
            "eligible": False,
            "reason": "governance-only; no association graph vector",
        },
        {
            "arm": "C2-KBEST-STABILITY",
            "artifact": str(PINNED["c2_terminal"][0].relative_to(ROOT)),
            "sha256": PINNED["c2_terminal"][1],
            "44b6": {"adjusted": None, "raw": None, "node": None},
            "6bba": {"adjusted": None, "raw": None, "node": None},
            "eligible": False,
            "reason": "dependency PARK; not scored",
        },
    ]


def run_audit() -> dict[str, Any]:
    authenticated: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in PINNED.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(f"pin mismatch for {name}: {observed}")
        authenticated[name] = {
            "path": str(path),
            "sha256": observed,
            "match": True,
        }

    interface = load_json(PINNED["n1_interface"][0])
    runtime_interface = interface_schema()
    if any(interface.get(key) != value for key, value in runtime_interface.items()):
        raise AssertionError("N1 runtime interface fields do not match frozen interface")
    if interface.get("frozen_implementation", {}).get("sha256") != PINNED[
        "n1_implementation"
    ][1]:
        raise AssertionError("N1 frozen interface implementation pin drift")

    baseline = n1_fixture()
    baseline_hash = graph_sha256(baseline)
    noop = compile_action(
        baseline, GraphAction("a.noop", ActionKind.NOOP)
    )
    fork = compile_action(baseline, fork_action())
    if baseline_hash != EXPECTED_BASELINE_HASH:
        raise AssertionError("N1 baseline fixture hash drift")
    if noop.after_graph_sha256 != EXPECTED_NOOP_HASH:
        raise AssertionError("N1 no-op changed graph")
    if fork.after_graph_sha256 != EXPECTED_FORK_HASH:
        raise AssertionError("N1 division-fork hash drift")
    if fork.removed_edges != ((2, 4), (11, 12), (11, 13)):
        raise AssertionError("N1 division fork did not own complete conflicts")

    d1 = load_json(PINNED["d1_dry_validation"][0])
    fixtures = d1["official_fixtures"]
    if len(fixtures) != 14 or any(
        row["observed_tp_fp_fn"] != row["expected_tp_fp_fn"] for row in fixtures
    ):
        raise AssertionError("official division fixture classifications drifted")
    expected_named = {
        "division_node_stage": [1, 2, 0],
        "duplicated_division": [1, 1, 0],
        "late_division": [1, 0, 0],
        "successive_div": [1, 1, 1],
    }
    observed_named = {
        row["name"]: row["observed_tp_fp_fn"] for row in fixtures
    }
    if any(observed_named.get(name) != value for name, value in expected_named.items()):
        raise AssertionError("direct/late/persistent/duplicate fixture semantics drifted")
    if (
        d1["counterfactual_count"] != 51
        or len(d1["counterfactuals"]) != 51
        or not all(
            row["dag_and_time_forward"]
            and row["indegree_le_1"]
            and row["outdegree_le_2"]
            for row in d1["counterfactuals"]
        )
        or d1["duplicate_survivors"] != 0
        or not d1["repeated_output_byte_identical"]
        or d1["held_graph_scoring"]
        or d1["cv57_access"]
        or d1["gpu_action"]
    ):
        raise AssertionError("D1 frozen dry-validation invariants drifted")

    arms = dependency_inventory()
    qualifying = [row for row in arms if row["eligible"]]
    if qualifying:
        raise AssertionError("unexpected qualifying dependency; report root before fitting")

    return {
        "schema": "celltrack-alpha.sa-division-fixture-audit.v1",
        "status": "PASS_PARK_DEPENDENCY",
        "dependency": {
            "gate": {
                "one_arm": True,
                "adjusted_min_each_family": 0.005,
                "raw_min_each_family": 0.0,
                "node_min_each_family": 0.0,
            },
            "candidate_arms": arms,
            "qualifying_arm_count": 0,
            "verdict": "PARK_DEPENDENCY",
            "held_family_execution": False,
        },
        "n1": {
            "baseline_graph_sha256": baseline_hash,
            "noop_after_graph_sha256": noop.after_graph_sha256,
            "noop_unchanged": True,
            "fork_after_graph_sha256": fork.after_graph_sha256,
            "fork_removed_edges": [list(edge) for edge in fork.removed_edges],
            "fork_added_edges": [[11, 4], [11, 5]],
            "atomic_complete_conflict_ownership": True,
        },
        "scorer_semantics": {
            "official_fixtures_passed": 14,
            "official_fixtures_total": 14,
            "named_semantics": expected_named,
            "counterfactuals_passed": 51,
            "counterfactuals_total": 51,
            "duplicate_survivors": 0,
            "repeated_output_byte_identical": True,
            "repeat_sha256": d1["repeat_sha256"],
            "d1_preserved_no_rerun": True,
        },
        "authenticated_inputs": authenticated,
        "execution": {
            "model_fitted": False,
            "family_44b6_ran": False,
            "family_6bba_ran": False,
            "cv57_used": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
            "external_action": False,
        },
        "memo_quarantine": True,
    }


class FixtureAuditTests(unittest.TestCase):
    def test_pinned_inputs_and_dependency_park(self) -> None:
        result = run_audit()
        self.assertEqual(result["dependency"]["qualifying_arm_count"], 0)
        self.assertEqual(result["dependency"]["verdict"], "PARK_DEPENDENCY")

    def test_noop_is_exact_identity(self) -> None:
        baseline = n1_fixture()
        result = compile_action(
            baseline, GraphAction("fixture.noop", ActionKind.NOOP)
        )
        self.assertEqual(result.before_graph_sha256, result.after_graph_sha256)
        self.assertEqual(result.after_graph_sha256, EXPECTED_NOOP_HASH)

    def test_atomic_binary_fork_matches_frozen_hash(self) -> None:
        result = compile_action(n1_fixture(), fork_action())
        self.assertEqual(result.after_graph_sha256, EXPECTED_FORK_HASH)
        self.assertEqual(
            result.removed_edges, ((2, 4), (11, 12), (11, 13))
        )

    def test_fork_omitted_conflict_fails_closed(self) -> None:
        invalid = GraphAction(
            "bad.omission",
            ActionKind.DIVISION_FORK,
            add_edges=((11, 4), (11, 5)),
            replace_outgoing_from=(11,),
            replace_incoming_to=(4, 5),
            expected_conflict_edges=((2, 4), (11, 12)),
        )
        with self.assertRaisesRegex(
            ActionCompileError, "incident-conflict proof mismatch"
        ):
            compile_action(n1_fixture(), invalid)

    def test_nonbinary_fork_fails_closed(self) -> None:
        invalid = GraphAction(
            "bad.nonbinary",
            ActionKind.DIVISION_FORK,
            add_edges=((11, 4),),
            replace_outgoing_from=(11,),
            replace_incoming_to=(4,),
            expected_conflict_edges=((2, 4), (11, 12), (11, 13)),
        )
        with self.assertRaisesRegex(
            ActionCompileError, "one parent and two daughters"
        ):
            compile_action(n1_fixture(), invalid)

    def test_missing_endpoint_fails_closed(self) -> None:
        invalid = GraphAction(
            "bad.endpoint",
            ActionKind.DIVISION_FORK,
            add_edges=((11, 4), (11, 999)),
            replace_outgoing_from=(11,),
            replace_incoming_to=(4, 999),
            expected_conflict_edges=((2, 4), (11, 12), (11, 13)),
        )
        with self.assertRaisesRegex(
            ActionCompileError, "replacement boundary references"
        ):
            compile_action(n1_fixture(), invalid)

    def test_official_semantics_are_immutable(self) -> None:
        result = run_audit()["scorer_semantics"]
        self.assertEqual(result["official_fixtures_passed"], 14)
        self.assertEqual(result["counterfactuals_passed"], 51)
        self.assertEqual(result["duplicate_survivors"], 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FixtureAuditTests)
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        return 1
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    result = run_audit()
    result["tests"] = {
        "passed": test_result.testsRun,
        "total": test_result.testsRun,
        "failures": len(test_result.failures),
        "errors": len(test_result.errors),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": result["status"], "tests": result["tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
