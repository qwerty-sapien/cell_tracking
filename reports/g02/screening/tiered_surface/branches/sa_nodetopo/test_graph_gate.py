from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gate = _module("nodetopo_gate", "evaluate_nodetopo_44b6_graph_gate.py")
guard = _module("nodetopo_guard", "run_guarded_graph_gate.py")


class NodeTopoGraphGateTest(unittest.TestCase):
    def test_frozen_population_authenticates(self) -> None:
        observed = gate._verify_pins()
        self.assertEqual(len(gate._load_actions()), 50)
        self.assertEqual(observed[str(gate.ACTIONS.relative_to(gate.ROOT))], gate.PINS[gate.ACTIONS])

    def test_decision_requires_every_preregistered_line(self) -> None:
        baseline = {
            "adj_edge_jaccard": 0.8,
            "edge_jaccard": 0.81,
            "node_recall": 0.9,
        }
        passing = {
            "adj_edge_jaccard": 0.805,
            "edge_jaccard": 0.81,
            "node_recall": 0.9,
        }
        self.assertEqual(gate._decision(baseline, passing)["verdict"], "GO_44B6")
        for key, delta in (
            ("adj_edge_jaccard", 0.004999),
            ("edge_jaccard", -0.000001),
            ("node_recall", -0.000001),
        ):
            candidate = dict(passing)
            candidate[key] = baseline[key] + delta
            self.assertEqual(
                gate._decision(baseline, candidate)["verdict"],
                "KILL_44B6_STOP",
            )

    def test_guard_is_local_and_bounded(self) -> None:
        self.assertEqual(guard.MAX_SECONDS, 2 * 60 * 60)
        self.assertLess(guard.MAX_SECONDS, 5 * 60 * 60)


if __name__ == "__main__":
    unittest.main()
