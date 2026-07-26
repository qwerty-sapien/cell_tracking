from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_h1_zero_wrfeat_graph_gate as gate  # noqa: E402
import evaluate_trackastra_linear_oof_graph_gate as engine  # noqa: E402


def _report(adjusted: float, raw: float = 0.0, recall: float = 0.0):
    return {
        "decision": {
            "adjusted_edge_jaccard_delta": adjusted,
            "raw_edge_jaccard_delta": raw,
            "node_recall_delta": recall,
            "observation_distinct": True,
        }
    }


class H1ZeroWrfeatGraphGateTests(unittest.TestCase):
    def test_contract_is_locked_fit105_only(self) -> None:
        path = gate.DEFAULT_CONTRACT
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        contract = engine.load_h1_contract(path, digest)
        self.assertEqual(
            contract["execution"]["family_order"], ["44b6", "6bba"]
        )
        self.assertFalse(contract["candidate"]["proposal_metrics_gate"])
        self.assertEqual(
            contract["fallback"]["status"], "CONDITIONAL_NOT_STARTED"
        )

    def test_44b6_nonpositive_kills_without_6bba(self) -> None:
        calls = []

        def evaluate(family: str):
            calls.append(family)
            return _report(0.0)

        result = gate.execute_ordered(evaluate)
        self.assertEqual(calls, ["44b6"])
        self.assertEqual(result["status"], "KILL")
        self.assertFalse(result["six_bba_executed"])
        self.assertFalse(result["cross_validation_authorized"])

    def test_44b6_subthreshold_parks_without_6bba(self) -> None:
        calls = []

        def evaluate(family: str):
            calls.append(family)
            return _report(0.0049)

        result = gate.execute_ordered(evaluate)
        self.assertEqual(calls, ["44b6"])
        self.assertEqual(result["status"], "PARK")

    def test_raw_or_recall_regression_blocks_downstream(self) -> None:
        for report in (_report(0.006, raw=-1e-9), _report(0.006, recall=-1e-9)):
            calls = []

            def evaluate(family: str):
                calls.append(family)
                return report

            result = gate.execute_ordered(evaluate)
            self.assertEqual(calls, ["44b6"])
            self.assertEqual(result["status"], "PARK")

    def test_6bba_runs_only_after_complete_44b6_pass(self) -> None:
        calls = []

        def evaluate(family: str):
            calls.append(family)
            return _report(0.005)

        result = gate.execute_ordered(evaluate)
        self.assertEqual(calls, ["44b6", "6bba"])
        self.assertEqual(
            result["status"], "PASS_HELD_FIT_BOTH_FAMILIES"
        )
        self.assertTrue(result["cross_validation_authorized"])

    def test_new_runner_has_no_forbidden_scope_arguments(self) -> None:
        source = (
            SCRIPTS / "evaluate_h1_zero_wrfeat_graph_gate.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--split", source)
        self.assertNotIn("--scorer-parity", source)
        self.assertNotIn("score_bh_cv57", source)


if __name__ == "__main__":
    unittest.main()
