from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


freeze = load("freeze_remote_output", HERE / "freeze_remote_output.py")
score = load("score_winner_once", HERE / "score_winner_once.py")


class WinnerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._artifact_root = freeze.ARTIFACT_ROOT
        self._freeze = freeze.FREEZE
        self._root = freeze.ROOT

    def tearDown(self) -> None:
        freeze.ARTIFACT_ROOT = self._artifact_root
        freeze.FREEZE = self._freeze
        freeze.ROOT = self._root

    def make_output(self, root: Path) -> None:
        root.mkdir(parents=True)
        trials = [
            {
                "trial_id": index,
                "tuple_id": f"tuple-{index:04d}",
                "objective_net": index,
                "positive_videos": 1,
                "worst_eight_video_net_sum": 0,
                "selected_transactions": 20,
            }
            for index in range(1000)
        ]
        selected = [f"{index:064x}" for index in range(20)]
        winner_row = trials[-1]
        dispersion = {"search_sensitive": True, "low_dispersion": False}
        semantic = {
            "schema": "celltrack-alpha.crw-hpo1000-semantic-receipt.v1",
            "status": "COMPLETE",
            "completed_workload_units": 1000,
            "expected_workload_units": 1000,
            "full_parameter_bank_size": 1000,
            "unique_tuple_count": 1000,
            "evaluated_trial_ids_sha256": hashlib.sha256(freeze.canonical(list(range(1000)))).hexdigest(),
            "winner_trial_id": winner_row["trial_id"],
            "winner_tuple_id": winner_row["tuple_id"],
            "winner_objective_net": winner_row["objective_net"],
            "winner_selected_transactions": 20,
            "dispersion": dispersion,
            "bindings": freeze.BINDINGS,
            "scientific_boundaries": {
                "competition_path_visits": 0,
                "cv57_accessed": False,
                "objective_is_official_score": False,
                "official_scorer_accessed": False,
                "six_bba_accessed": False,
                "submission_artifact_written": False,
            },
        }
        winner = {
            "schema": "celltrack-alpha.crw-hpo1000-winner.v1",
            "status": "FROZEN_FOR_ONE_LOCAL_GRAPH_SCORE",
            "winner": winner_row,
            "selected_candidate_ids": selected,
            "selected_candidate_ids_sha256": hashlib.sha256(freeze.canonical(selected)).hexdigest(),
            "graph_score_calls_authorized": 1,
            "cv57_access": False,
            "six_bba_access": False,
            "competition_submission": False,
        }
        terminal = {
            "schema": "celltrack-alpha.run-log.v1",
            "stage": "complete",
            "event": "stage_end",
            "status": "COMPLETE",
            "completed_workload_units": 1000,
        }
        (root / "TRIAL_RESULTS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trials), encoding="ascii")
        (root / "RUN_LOG.jsonl").write_text(json.dumps(terminal) + "\n", encoding="ascii")
        for name, value in (
            ("SEMANTIC_RECEIPT.json", semantic),
            ("WINNER.json", winner),
            ("DISPERSION_SUMMARY.json", dispersion),
        ):
            (root / name).write_bytes(freeze.canonical(value))

    def test_remote_output_happy_path_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "artifacts"
            self.make_output(root)
            freeze.ARTIFACT_ROOT = root
            freeze.FREEZE = base / "freeze.json"
            freeze.ROOT = base
            payload = freeze.validate()
            self.assertEqual(payload["semantic"]["unique_tuple_count"], 1000)
            semantic = json.loads((root / "SEMANTIC_RECEIPT.json").read_text())
            semantic["completed_workload_units"] = 999
            (root / "SEMANTIC_RECEIPT.json").write_bytes(freeze.canonical(semantic))
            with self.assertRaises(freeze.FreezeError):
                freeze.validate()

    def test_frozen_verdict_truth_table(self) -> None:
        gate = {
            "held_fit_exact_graph_adjusted_delta_min": 0.005,
            "held_fit_exact_graph_raw_delta_min": 0.0,
            "held_fit_exact_graph_node_delta_min": 0.0,
            "selected_transactions_min": 20,
        }
        self.assertEqual(score.verdict({"adjusted": 0.006, "raw": 0.001, "node": 0.0}, 20, 1, gate), ("Y", "GO"))
        self.assertEqual(score.verdict({"adjusted": 0.004, "raw": 0.001, "node": 0.0}, 20, 1, gate), ("N", "PARK"))
        self.assertEqual(score.verdict({"adjusted": -0.001, "raw": 0.001, "node": 0.0}, 20, 1, gate), ("N", "DEAD"))


if __name__ == "__main__":
    unittest.main()
