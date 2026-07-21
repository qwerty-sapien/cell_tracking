from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import ZipStore

from celltrack_alpha.bh_graph import GraphEdge, GraphNode, GraphResult
from scripts import v45par


class HashValidationTests(unittest.TestCase):
    def test_packet_hash_constants_are_pinned(self) -> None:
        self.assertEqual(
            v45par.PINNED_SHA256[v45par.PREDICTION_ZIP],
            "75304b22389a81f28e9616ff7c63d6d200ef50f7d4348ee68818d4fdcbc97994",
        )
        self.assertEqual(
            v45par.PINNED_SHA256[v45par.EXPECTED_SUBMISSION],
            "6f8d81f6e5d9eaa066e47f0cdd2d22df48eaa74853d2940d0bc957f3ef3a19e1",
        )
        self.assertEqual(
            v45par.PINNED_SHA256[v45par.EXPECTED_RUN],
            "2c2c5e6b345a24426ba28c4ab68c544048ab2a07aa503b0319ac6481d267f090",
        )

    def test_validate_hashes_accepts_exact_bytes_and_rejects_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tiny.bin"
            payload = b"pinned tiny fixture\n"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(v45par.validate_hashes({path: digest})[str(path)], digest)
            path.write_bytes(payload + b"changed")
            with self.assertRaisesRegex(v45par.V45ParityError, "SHA256 mismatch"):
                v45par.validate_hashes({path: digest})

    def test_validate_hashes_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "missing"
            with self.assertRaisesRegex(v45par.V45ParityError, "missing pinned input"):
                v45par.validate_hashes({path: "0" * 64})


class TinyZarrFixtureTests(unittest.TestCase):
    def _write_geff(self, path: Path, *, target_t: int = 1) -> None:
        with ZipStore(path, mode="w") as store:
            group = zarr.create_group(
                store=store,
                path=f"{v45par.PREDICTION_ROOT}/tiny.geff",
                overwrite=True,
            )
            group.create_array("nodes/ids", data=np.array([3, 7], dtype=np.uint64))
            group.create_array(
                "nodes/props/t/values", data=np.array([0, target_t], dtype=np.int32)
            )
            group.create_array(
                "nodes/props/z/values", data=np.array([1.25, 1.5], dtype=np.float64)
            )
            group.create_array(
                "nodes/props/y/values", data=np.array([2.0, 3.0], dtype=np.float64)
            )
            group.create_array(
                "nodes/props/x/values", data=np.array([4.0, 5.0], dtype=np.float64)
            )
            group.create_array(
                "edges/ids", data=np.array([[3, 7]], dtype=np.uint64)
            )
            group.create_array(
                "edges/props/edge_prob/values", data=np.array([0.875], dtype=np.float64)
            )

    def test_load_raw_geff_reads_float_nodes_and_adjacent_probability(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "predictions.zip"
            self._write_geff(path)
            with ZipStore(path, mode="r") as store:
                graph = v45par.load_raw_geff(store, "tiny")
            self.assertEqual([node.node_id for node in graph.nodes], [3, 7])
            self.assertEqual(graph.nodes[0].coord_zyx, (1.25, 2.0, 4.0))
            self.assertEqual(graph.learned_probabilities, {(3, 7): 0.875})
            self.assertEqual(graph.raw_edge_count, 1)
            self.assertEqual(graph.adjacent_frame_pairs, 1)

    def test_load_raw_geff_rejects_nonadjacent_edge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "predictions.zip"
            self._write_geff(path, target_t=2)
            with ZipStore(path, mode="r") as store:
                with self.assertRaisesRegex(v45par.V45ParityError, "not adjacent"):
                    v45par.load_raw_geff(store, "tiny")

    def test_lazy_frame_cache_reads_only_requested_frames(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "images.zip"
            data = np.arange(3 * 2 * 3 * 4, dtype=np.uint16).reshape((3, 2, 3, 4))
            with ZipStore(path, mode="w") as store:
                group = zarr.create_group(store=store, path="test/tiny.zarr", overwrite=True)
                group.create_array("0", data=data, chunks=(1, 2, 3, 4))
            with ZipStore(path, mode="r") as store:
                array = zarr.open_array(store=store, path="test/tiny.zarr/0", mode="r")
                cache = v45par.LazyFrameCache(array)
                np.testing.assert_array_equal(cache(1), data[1])
                np.testing.assert_array_equal(cache(1), data[1])
                np.testing.assert_array_equal(cache(2), data[2])
                self.assertEqual(cache.read_count, 2)
                self.assertEqual(cache.requested_frames, {1, 2})
                self.assertEqual(cache.peak_cached_bytes, data[0].nbytes)
                cache.release()


class TinyCsvFixtureTests(unittest.TestCase):
    def _submission_row(
        self,
        row_id: int,
        dataset: str,
        row_type: str,
        values: tuple[int, ...],
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "id": row_id,
            "dataset": dataset,
            "row_type": row_type,
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": -1,
            "target_id": -1,
        }
        if row_type == "node":
            for name, value in zip(("node_id", "t", "z", "y", "x"), values, strict=True):
                row[name] = value
        else:
            row["source_id"], row["target_id"] = values
        return row

    def test_load_expected_submission_uses_exact_public_tuples(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "submission.csv"
            rows: list[dict[str, object]] = []
            row_id = 0
            for dataset in v45par.DATASETS:
                rows.append(self._submission_row(row_id, dataset, "node", (1, 0, 2, 3, 4)))
                row_id += 1
                rows.append(self._submission_row(row_id, dataset, "node", (2, 1, 3, 4, 5)))
                row_id += 1
                rows.append(self._submission_row(row_id, dataset, "edge", (1, 2)))
                row_id += 1
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=v45par.SUBMISSION_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            graph = v45par.load_expected_submission(path, v45par.DATASETS[0])
            self.assertEqual(graph.nodes, {(1, 0, 2, 3, 4), (2, 1, 3, 4, 5)})
            self.assertEqual(graph.edges, {(1, 2)})

    def test_load_expected_run_requires_all_pinned_columns_and_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=v45par.RUN_COLUMNS)
                writer.writeheader()
                for dataset in v45par.DATASETS:
                    row = {name: "0" for name in v45par.RUN_COLUMNS}
                    row["dataset"] = dataset
                    row["experiment_tag"] = "tiny"
                    writer.writerow(row)
            rows = v45par.load_expected_run(path)
            self.assertEqual(tuple(sorted(rows)), v45par.DATASETS)
            self.assertEqual(rows[v45par.DATASETS[0]]["experiment_tag"], "tiny")


class ComparisonTests(unittest.TestCase):
    def test_public_rounding_occurs_only_at_comparison(self) -> None:
        node = GraphNode(9, 2, 1.5, 2.49, -0.6)
        self.assertEqual(node.coord_zyx, (1.5, 2.49, -0.6))
        self.assertEqual(v45par.public_node_tuple(node), (9, 2, 2, 2, 0))

    def test_compare_graph_reports_complete_exact_diffs(self) -> None:
        nodes = (
            GraphNode(1, 0, 0.0, 0.0, 0.0),
            GraphNode(2, 1, 1.0, 1.0, 1.0),
        )
        edges = (GraphEdge(1, 2, 1.0, 1.0),)
        result = GraphResult(nodes, edges, {}, {})
        expected = v45par.ExpectedGraph(
            frozenset({(1, 0, 0, 0, 0), (2, 1, 1, 1, 2)}),
            frozenset({(2, 1)}),
        )
        node_diff, edge_diff = v45par.compare_graph(result, expected)
        self.assertEqual(node_diff["missing"], [[2, 1, 1, 1, 2]])
        self.assertEqual(node_diff["extra"], [[2, 1, 1, 1, 1]])
        self.assertEqual(edge_diff["missing"], [[2, 1]])
        self.assertEqual(edge_diff["extra"], [[1, 2]])

    def test_canonical_report_is_ascii_and_key_sorted(self) -> None:
        payload = v45par.canonical_json({"z": "\N{GREEK SMALL LETTER ALPHA}", "a": 1})
        payload.decode("ascii")
        self.assertEqual(json.loads(payload), {"a": 1, "z": "\N{GREEK SMALL LETTER ALPHA}"})
        self.assertLess(payload.find(b'"a"'), payload.find(b'"z"'))

    def test_solver_modes_have_distinct_reports_and_sparse_default(self) -> None:
        self.assertEqual(v45par.parse_args(["--all"]).solver_mode, "sparse")
        diagnostic = v45par.parse_args(
            ["--dataset", v45par.DATASETS[0], "--solver-mode", "dense_diagnostic"]
        )
        self.assertEqual(diagnostic.solver_mode, "dense_diagnostic")
        self.assertNotEqual(
            v45par.REPORT_PATHS["sparse"],
            v45par.REPORT_PATHS["dense_diagnostic"],
        )
        self.assertLessEqual(len(v45par.REPORT_PATHS["dense_diagnostic"].name), 15)


if __name__ == "__main__":
    unittest.main()
