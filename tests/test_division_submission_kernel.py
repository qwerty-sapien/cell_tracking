from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.kaggle_calibration import SubmissionEdge, SubmissionGraph, SubmissionNode, read_submission, write_submission


def load_builder():
    path = ROOT / "scripts" / "build_division_submission_kernel.py"
    spec = importlib.util.spec_from_file_location("build_division_submission_kernel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def fixture_graph() -> SubmissionGraph:
    nodes = {
        node_id: SubmissionNode(node_id=node_id, t=time, z=0, y=node_id, x=0)
        for node_id, time in ((10, 0), (11, 1), (12, 2), (13, 2), (14, 3), (15, 3))
    }
    return SubmissionGraph(
        "fixture",
        nodes,
        [
            SubmissionEdge(10, 11),
            SubmissionEdge(12, 14),
            SubmissionEdge(13, 15),
        ],
    )


def gate() -> dict[str, object]:
    return {
        "family": "44b6",
        "source_artifact_sha256": "a" * 64,
        "combined_score_delta": 0.001,
        "division_jaccard": 0.30,
        "adjusted_edge_jaccard_delta": 0.0,
        "raw_edge_jaccard_delta": 0.0,
        "node_recall_delta": 0.0,
        "duplicate_survivors": 0,
    }


class DivisionSubmissionKernelTests(unittest.TestCase):
    def run_emitter(self, package: Path, base: Path, candidate: Path, output: Path, *, allow_empty: bool) -> dict:
        report = output.with_suffix(".receipt.json")
        command = [
            sys.executable,
            str(package / "division_submission_kernel.py"),
            "--base-submission",
            str(base),
            "--candidate-receipt",
            str(candidate),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
        if allow_empty:
            command.append("--allow-empty")
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(report.read_text(encoding="utf-8"))

    def test_package_copies_the_frozen_d1_decoder_and_n1_compiler(self) -> None:
        builder = load_builder()
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            manifest = builder.build_package(package, Path(temporary) / "manifest.json")
            files = {row["path"]: row["sha256"] for row in manifest["identity"]["files"]}
            self.assertEqual(
                files["frozen_d1_decoder.py"],
                "e780a398a6586e0c6acd7fb89d54626e732b254992252db91e20f5385a4bbcf6",
            )
            self.assertEqual(
                files["celltrack_alpha/action_compiler.py"],
                "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
            )
            self.assertEqual(
                sha256_file(package / "frozen_d1_evidence.json"),
                "f6a96d868e6ad2a21aea123157bc9bddc169e8453c117e829495d933514deaf9",
            )

    def test_emits_one_scorer_valid_fork_and_suppresses_duplicate_equivalence(self) -> None:
        builder = load_builder()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            builder.build_package(package, root / "manifest.json")
            base = root / "base.csv"
            write_submission({"fixture": fixture_graph()}, base)
            candidate = root / "candidates.json"
            candidate.write_text(
                json.dumps(
                    {
                        "schema": "celltrack-alpha.d1-division-candidates.v1",
                        "base_submission_sha256": sha256_file(base),
                        "gate": gate(),
                        "candidates": [
                            {
                                "dataset": "fixture",
                                "candidate_id": "winner",
                                "parent_id": 11,
                                "child_a_id": 12,
                                "child_b_id": 13,
                                "grandchild_a_id": 14,
                                "grandchild_b_id": 15,
                                "score": 0.9,
                                "equivalence_key": "event-1",
                            },
                            {
                                "dataset": "fixture",
                                "candidate_id": "duplicate",
                                "parent_id": 11,
                                "child_a_id": 12,
                                "child_b_id": 13,
                                "grandchild_a_id": 14,
                                "grandchild_b_id": 15,
                                "score": 0.1,
                                "equivalence_key": "event-1",
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            output = root / "submission.csv"
            receipt = self.run_emitter(package, base, candidate, output, allow_empty=False)
            graph = read_submission(output)["fixture"]
            self.assertEqual({(edge.source_id, edge.target_id) for edge in graph.edges}, {(10, 11), (11, 12), (11, 13), (12, 14), (13, 15)})
            self.assertEqual(receipt["accepted_forks"], 1)
            self.assertEqual(receipt["duplicate_survivors"], 0)
            self.assertEqual(receipt["datasets"]["fixture"]["suppression_reasons"], {"duplicate_equivalence_class": 1})
            self.assertTrue(receipt["output_schema"]["divisions_enabled"])

    def test_empty_divisions_on_round_trip_preserves_canonical_graph_and_score(self) -> None:
        builder = load_builder()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            builder.build_package(package, root / "manifest.json")
            base = root / "base.csv"
            write_submission({"fixture": fixture_graph()}, base)
            candidate = root / "empty.json"
            candidate.write_text(
                json.dumps(
                    {
                        "schema": "celltrack-alpha.d1-division-candidates.v1",
                        "base_submission_sha256": sha256_file(base),
                        "candidates": [],
                    }
                ),
                encoding="utf-8",
            )
            receipt = self.run_emitter(package, base, candidate, root / "submission.csv", allow_empty=True)
            self.assertEqual(receipt["verdict"], "PASS_EMPTY_DIVISIONS_ON")
            self.assertEqual(receipt["score_change"], 0.0)
            self.assertEqual(receipt["base_graph_sha256"], receipt["output_graph_sha256"])
            self.assertFalse(receipt["official_scorer_rerun"])


if __name__ == "__main__":
    unittest.main()
