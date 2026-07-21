from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cmpgraph  # noqa: E402


def node(dataset: str, node_id: int, t: int, z: int, y: int, x: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "row_type": "node",
        "node_id": node_id,
        "t": t,
        "z": z,
        "y": y,
        "x": x,
        "source_id": -1,
        "target_id": -1,
    }


def edge(dataset: str, source_id: int, target_id: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "row_type": "edge",
        "node_id": -1,
        "t": -1,
        "z": -1,
        "y": -1,
        "x": -1,
        "source_id": source_id,
        "target_id": target_id,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cmpgraph.SUBMISSION_COLUMNS)
        writer.writeheader()
        for row_id, row in enumerate(rows):
            writer.writerow({"id": row_id, **row})


class CompareGraphTests(unittest.TestCase):
    def test_exact_node_and_edge_agreement(self) -> None:
        rows = [
            node("sample", 1, 0, 2, 3, 4),
            node("sample", 2, 1, 2, 4, 4),
            edge("sample", 1, 2),
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = root / "left.csv"
            right = root / "right.csv"
            write_csv(left, rows)
            write_csv(right, rows)
            report = cmpgraph.build_report(str(left), str(right), strict=True)

        self.assertEqual(report["global"]["nodes"]["intersection"], 2)
        self.assertEqual(report["global"]["nodes"]["left_only"], 0)
        self.assertEqual(report["global"]["edges"]["intersection"], 1)
        self.assertEqual(report["global"]["edges"]["left_overlap_ratio"], 1.0)
        self.assertEqual(
            report["global"]["single_child_sources"]["intersection"], 1
        )
        self.assertEqual(report["provenance"]["left"]["file_count"], 1)
        self.assertEqual(len(report["provenance"]["left"]["files"][0]["sha256"]), 64)

    def test_physical_cutoff_and_different_node_ids(self) -> None:
        left_rows = [
            node("sample", 1, 0, 0, 0, 0),
            node("sample", 2, 0, 0, 100, 0),
        ]
        right_rows = [
            node("sample", 101, 0, 0, 0, 17),  # 6.90625 um
            node("sample", 102, 0, 0, 100, 18),  # 7.3125 um
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = root / "left.csv"
            right = root / "right.csv"
            write_csv(left, left_rows)
            write_csv(right, right_rows)
            report = cmpgraph.build_report(str(left), str(right))

        nodes = report["datasets"]["sample"]["nodes"]
        self.assertEqual(nodes["intersection"], 1)
        self.assertEqual(nodes["left_only"], 1)
        self.assertEqual(nodes["right_only"], 1)
        self.assertAlmostEqual(
            report["datasets"]["sample"]["matching_distance"]["max_um"],
            6.90625,
        )

    def test_division_and_single_child_motifs(self) -> None:
        left_rows = [
            node("sample", 1, 0, 0, 0, 0),
            node("sample", 2, 1, 0, 1, 0),
            node("sample", 3, 1, 0, 0, 1),
            node("sample", 4, 2, 0, 2, 0),
            edge("sample", 1, 2),
            edge("sample", 1, 3),
            edge("sample", 2, 4),
        ]
        right_rows = [
            node("sample", 10, 0, 0, 0, 0),
            node("sample", 20, 1, 0, 1, 0),
            node("sample", 30, 1, 0, 0, 1),
            node("sample", 40, 2, 0, 2, 0),
            edge("sample", 10, 20),
            edge("sample", 10, 30),
            edge("sample", 20, 40),
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = root / "left.csv"
            right = root / "right.csv"
            write_csv(left, left_rows)
            write_csv(right, right_rows)
            report = cmpgraph.build_report(str(left), str(right), strict=True)

        dataset = report["datasets"]["sample"]
        self.assertEqual(dataset["edges"]["intersection"], 3)
        self.assertEqual(dataset["division_sources"]["intersection"], 1)
        self.assertEqual(
            dataset["division_sources"]["source_status_intersection"], 1
        )
        self.assertEqual(dataset["single_child_sources"]["intersection"], 1)

    def test_dense_sparse_component_ties_are_stable(self) -> None:
        left = [cmpgraph.Node(index, 0, 0, 0, 0) for index in range(1, 7)]
        right = [cmpgraph.Node(index, 0, 0, 0, 0) for index in range(101, 107)]

        first, distances, stats = cmpgraph.match_frame(left, right)
        second, _, _ = cmpgraph.match_frame(reversed(left), reversed(right))

        self.assertEqual(first, second)
        self.assertEqual(first, dict(zip(range(1, 7), range(101, 107), strict=True)))
        self.assertEqual(distances, [0.0] * 6)
        self.assertEqual(stats.candidate_pairs, 36)
        self.assertEqual(stats.peak_component_candidates, 36)
        self.assertEqual(stats.peak_assignment_dimension, 12)

    def test_malformed_duplicate_and_dangling_inputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            malformed = root / "bad.csv"
            malformed.write_text("dataset,row_type\nsample,node\n", encoding="utf-8")
            with self.assertRaisesRegex(cmpgraph.ComparisonError, "unexpected header"):
                cmpgraph.load_graphs([malformed], strict=False)

            duplicate = root / "duplicate.csv"
            write_csv(
                duplicate,
                [node("sample", 1, 0, 0, 0, 0), node("sample", 1, 1, 0, 0, 0)],
            )
            with self.assertRaisesRegex(cmpgraph.ComparisonError, "duplicate node ID"):
                cmpgraph.load_graphs([duplicate], strict=False)

            dangling = root / "dangling.csv"
            write_csv(
                dangling,
                [node("sample", 1, 0, 0, 0, 0), edge("sample", 1, 9)],
            )
            with self.assertRaisesRegex(cmpgraph.ComparisonError, "dangling edge"):
                cmpgraph.load_graphs([dangling], strict=False)

    def test_strict_rejects_backward_and_nonconsecutive_edges(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name, rows in (
                (
                    "back.csv",
                    [
                        node("sample", 1, 1, 0, 0, 0),
                        node("sample", 2, 0, 0, 0, 0),
                        edge("sample", 1, 2),
                    ],
                ),
                (
                    "skip.csv",
                    [
                        node("sample", 1, 0, 0, 0, 0),
                        node("sample", 2, 2, 0, 0, 0),
                        edge("sample", 1, 2),
                    ],
                ),
            ):
                path = root / name
                write_csv(path, rows)
                cmpgraph.load_graphs([path], strict=False)
                with self.assertRaisesRegex(
                    cmpgraph.ComparisonError, "backward or nonconsecutive"
                ):
                    cmpgraph.load_graphs([path], strict=True)

    def test_fragment_directory_and_glob_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left_dir = root / "left"
            right_dir = root / "right"
            write_csv(left_dir / "a.csv", [node("sample", 1, 0, 0, 0, 0)])
            write_csv(
                left_dir / "b.csv",
                [node("sample", 2, 1, 0, 1, 0), edge("sample", 1, 2)],
            )
            write_csv(right_dir / "a.csv", [node("sample", 11, 0, 0, 0, 0)])
            write_csv(
                right_dir / "b.csv",
                [node("sample", 12, 1, 0, 1, 0), edge("sample", 11, 12)],
            )

            report = cmpgraph.build_report(
                str(left_dir), str(right_dir / "*.csv"), strict=True
            )
            output = root / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts" / "cmpgraph.py"),
                    str(left_dir),
                    str(right_dir),
                    "--strict",
                    "--out",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            on_disk = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["provenance"]["left"]["file_count"], 2)
        self.assertEqual(report["global"]["edges"]["intersection"], 1)
        self.assertEqual(on_disk["global"]["nodes"]["intersection"], 2)

    def test_missing_dataset_overlap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = root / "left.csv"
            right = root / "right.csv"
            write_csv(left, [node("left_ds", 1, 0, 0, 0, 0)])
            write_csv(right, [node("right_ds", 1, 0, 0, 0, 0)])
            with self.assertRaisesRegex(cmpgraph.ComparisonError, "no overlapping"):
                cmpgraph.build_report(str(left), str(right))

    def test_dataset_prefix_normalization_build_report_and_cli(self) -> None:
        left_rows = [
            node("test:sample", 1, 0, 0, 0, 0),
            node("test:sample", 2, 1, 0, 1, 0),
            edge("test:sample", 1, 2),
        ]
        right_rows = [
            node("sample", 11, 0, 0, 0, 0),
            node("sample", 12, 1, 0, 1, 0),
            edge("sample", 11, 12),
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = root / "left.csv"
            right = root / "right.csv"
            write_csv(left, left_rows)
            write_csv(right, right_rows)

            with self.assertRaisesRegex(cmpgraph.ComparisonError, "no overlapping"):
                cmpgraph.build_report(str(left), str(right))
            report = cmpgraph.build_report(
                str(left), str(right), left_strip_prefix="test:"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts" / "cmpgraph.py"),
                    str(left),
                    str(right),
                    "--left-strip-prefix",
                    "test:",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(set(report["datasets"]), {"sample"})
        self.assertEqual(report["global"]["nodes"]["intersection"], 2)
        self.assertEqual(report["global"]["edges"]["intersection"], 1)
        self.assertEqual(report["parameters"]["left_strip_prefix"], "test:")
        self.assertIsNone(report["parameters"]["right_strip_prefix"])
        self.assertEqual(report["provenance"]["left"]["strip_prefix"], "test:")
        self.assertIsNone(report["provenance"]["right"]["strip_prefix"])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        cli_report = json.loads(completed.stdout)
        self.assertEqual(cli_report["global"]["nodes"]["intersection"], 2)
        self.assertEqual(cli_report["parameters"]["left_strip_prefix"], "test:")

    def test_dataset_prefix_normalization_collisions_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            collision = root / "collision.csv"
            right = root / "right.csv"
            write_csv(
                collision,
                [
                    node("test:sample", 1, 0, 0, 0, 0),
                    node("sample", 1, 0, 0, 1, 0),
                ],
            )
            write_csv(right, [node("sample", 10, 0, 0, 0, 0)])
            with self.assertRaisesRegex(
                cmpgraph.ComparisonError, "prefix normalization collision"
            ):
                cmpgraph.build_report(
                    str(collision), str(right), left_strip_prefix="test:"
                )

            empty = root / "empty.csv"
            write_csv(empty, [node("test:", 1, 0, 0, 0, 0)])
            with self.assertRaisesRegex(
                cmpgraph.ComparisonError, "empty dataset ID"
            ):
                cmpgraph.build_report(
                    str(empty), str(right), left_strip_prefix="test:"
                )


if __name__ == "__main__":
    unittest.main()
