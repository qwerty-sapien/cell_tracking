from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_submissions",
    ROOT / "scripts/compare_submissions.py",
)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)

from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SUBMISSION_COLUMNS,
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    write_submission,
)


def node(node_id: int, t: int, z: int, y: int, x: int) -> SubmissionNode:
    return SubmissionNode(node_id=node_id, t=t, z=z, y=y, x=x)


def graph(
    dataset: str,
    nodes: list[SubmissionNode],
    edges: list[tuple[int, int]],
) -> SubmissionGraph:
    return SubmissionGraph(
        dataset=dataset,
        nodes={value.node_id: value for value in nodes},
        edges=[SubmissionEdge(*edge) for edge in edges],
    )


class CompareSubmissionsTest(unittest.TestCase):
    def test_score_artifact_fails_cleanly_for_missing_gt_zip(self) -> None:
        artifact = compare.PredictionArtifact(
            label="empty",
            source="test",
            files=(),
            manifest_sha256="0" * 64,
            canonical_sha256="0" * 64,
            graphs={},
        )
        missing = ROOT / "does-not-exist.zip"
        with self.assertRaisesRegex(
            FileNotFoundError, "ground-truth zip does not exist"
        ):
            compare.score_artifact(
                artifact,
                gt_zip=missing,
                scorer_parity=ROOT / "reports/g02/scorer_parity_v1.json",
            )

    def test_json_safe_preserves_strict_output_and_marks_nonfinite_as_null(self) -> None:
        value = compare._json_safe(
            {"ok": 1.0, "nan": float("nan"), "nested": [float("inf")]}
        )
        self.assertEqual(value, {"ok": 1.0, "nan": None, "nested": [None]})
        self.assertEqual(
            json.loads(json.dumps(value, allow_nan=False)),
            value,
        )

    def test_canonical_hash_ignores_csv_row_order_and_presentation_id(self) -> None:
        source = graph(
            "44b6_case",
            [node(1, 0, 0, 0, 0), node(2, 1, 0, 1, 0)],
            [(1, 2)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            write_submission({source.dataset: source}, first)
            with first.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with second.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
                writer.writeheader()
                for index, row in enumerate(reversed(rows), start=100):
                    row["id"] = str(index)
                    writer.writerow(row)
            a = compare.load_prediction(first, label="a")
            b = compare.load_prediction(second, label="b")
        self.assertNotEqual(a.files[0]["sha256"], b.files[0]["sha256"])
        self.assertEqual(a.canonical_sha256, b.canonical_sha256)

    def test_renumbered_identical_graph_is_observationally_identical(self) -> None:
        a = graph(
            "44b6_case",
            [
                node(1, 0, 0, 0, 0),
                node(2, 1, 0, 2, 0),
                node(3, 2, 0, 1, 0),
                node(4, 2, 0, 3, 0),
            ],
            [(1, 2), (2, 3), (2, 4)],
        )
        b = graph(
            "44b6_case",
            [
                node(10, 0, 0, 0, 0),
                node(20, 1, 0, 2, 0),
                node(30, 2, 0, 1, 0),
                node(40, 2, 0, 3, 0),
            ],
            [(10, 20), (20, 30), (20, 40)],
        )
        row = compare.compare_graphs(a, b, max_distance_um=7.0)
        aggregate = compare.aggregate_rows([row])
        classification = compare.classify_distance(aggregate)
        self.assertFalse(row["canonical_equal"])
        self.assertEqual(row["nodes"]["exact_coordinate_matches_after_id"], 4)
        self.assertEqual(row["edges_raw"]["jaccard"], 1.0)
        self.assertEqual(row["forks"]["agreement"], 1.0)
        self.assertEqual(classification["label"], "IDENTICAL")

    def test_matching_is_time_aware_and_uses_physical_distance(self) -> None:
        a = graph(
            "6bba_case",
            [node(1, 0, 0, 0, 0), node(2, 1, 0, 0, 0)],
            [(1, 2)],
        )
        b = graph(
            "6bba_case",
            [
                node(10, 1, 0, 0, 0),
                node(20, 0, 5, 0, 0),  # 8.125 um in z: outside official gate.
            ],
            [(20, 10)],
        )
        mapping, details = compare.match_nodes(a, b, max_distance_um=7.0)
        self.assertEqual(mapping, {10: 2})
        self.assertEqual(details["shared"], 1)
        self.assertEqual(details["a_only"], 1)
        self.assertEqual(details["b_only"], 1)

    def test_nonzero_edge_and_fork_changes_are_distinct_components(self) -> None:
        nodes = [
            node(1, 0, 0, 0, 0),
            node(2, 1, 0, 1, 0),
            node(3, 2, 0, 0, 0),
            node(4, 2, 0, 2, 0),
            node(5, 3, 0, 0, 1),
            node(6, 3, 0, 2, 1),
        ]
        a = graph(
            "44b6_case",
            nodes,
            [(1, 2), (2, 3), (2, 4), (3, 5), (4, 6)],
        )
        b = graph(
            "44b6_case",
            nodes,
            [(1, 2), (2, 3), (3, 5), (4, 6)],
        )
        row = compare.compare_graphs(a, b, max_distance_um=7.0)
        aggregate = compare.aggregate_rows([row])
        classification = compare.classify_distance(aggregate)
        self.assertEqual(row["nodes"]["jaccard"], 1.0)
        self.assertEqual(row["edges_raw"]["a_only"], 1)
        self.assertEqual(row["forks"]["a_only"], 1)
        self.assertEqual(classification["label"], "DISTINCT")
        self.assertIn(
            "raw_edge_disagreement_fraction",
            classification["exceeded_components"],
        )
        self.assertIn(
            "fork_disagreement_fraction",
            classification["exceeded_components"],
        )

    def test_family_aggregation_remains_separate(self) -> None:
        identical = graph(
            "44b6_a",
            [node(1, 0, 0, 0, 0), node(2, 1, 0, 1, 0)],
            [(1, 2)],
        )
        changed_a = graph(
            "6bba_b",
            [node(1, 0, 0, 0, 0), node(2, 1, 0, 1, 0)],
            [(1, 2)],
        )
        changed_b = graph(
            "6bba_b",
            [node(1, 0, 0, 0, 0), node(2, 1, 0, 1, 0)],
            [],
        )
        rows = [
            compare.compare_graphs(identical, identical, max_distance_um=7.0),
            compare.compare_graphs(changed_a, changed_b, max_distance_um=7.0),
        ]
        by_family = {
            family: compare.aggregate_rows(
                [row for row in rows if row["family"] == family]
            )
            for family in ("44b6", "6bba")
        }
        self.assertEqual(by_family["44b6"]["edges_raw"]["jaccard"], 1.0)
        self.assertEqual(by_family["6bba"]["edges_raw"]["jaccard"], 0.0)

    def test_dataset_subset_is_explicit_and_shared(self) -> None:
        args = compare.parse_args(
            [
                "--a",
                "a",
                "--b",
                "b",
                "--output",
                "out.json",
                "--dataset",
                "train:44b6_case",
            ]
        )
        self.assertEqual(args.dataset, ["train:44b6_case"])


if __name__ == "__main__":
    unittest.main()
