from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_cellect_fit117.py"
SPEC = importlib.util.spec_from_file_location("train_cellect_fit117", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Array:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.shape = value.shape

    def __getitem__(self, item):
        return self.value[item]


class TrainCellectTests(unittest.TestCase):
    def test_official_match_restores_unsorted_nonconsecutive_original_ids(self) -> None:
        graph = MODULE.SubmissionGraph(
            "tiny",
            {
                900: SimpleNamespace(node_id=900, t=1, z=2, y=3, x=4),
                100: SimpleNamespace(node_id=100, t=0, z=1, y=2, x=3),
            },
            [SimpleNamespace(source_id=100, target_id=900)],
        )
        gt = {
            "nodes": {
                "node_id": np.asarray([42, 7]),
                "t": np.asarray([1, 0]),
                "z": np.asarray([2.0, 1.0]),
                "y": np.asarray([3.0, 2.0]),
                "x": np.asarray([4.0, 3.0]),
            },
            "edges": np.asarray([[7, 42]], dtype=np.int64),
        }

        observed = MODULE.official_gt_to_pred(graph, gt, (1.0, 1.0, 1.0))

        self.assertEqual(observed, {7: 100, 42: 900})

    def test_training_gate_rejects_failed_e4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": MODULE.EXPECTED_GATE_SCHEMA,
                        "status": "complete",
                        "e4": {"decision": "FAIL"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MODULE.CellectContractError, "training is forbidden"):
                MODULE.validate_gate(path)

    def test_crop_with_pad_preserves_shape_and_coordinates(self) -> None:
        array = _Array(np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6))
        observed = MODULE.crop_with_pad(array, 0, (-1, 2, 3), (4, 4, 4))
        self.assertEqual(observed.shape, (3, 4, 4, 4))
        np.testing.assert_array_equal(observed[:, 0], 0)
        np.testing.assert_array_equal(observed[:, 1:, :3, :3], array.value[:, :3, 2:5, 3:6])

    def test_build_crop_uses_raw_mapped_edges_and_topk_candidates(self) -> None:
        array = _Array(np.ones((3, 8, 16, 16), dtype=np.uint16))
        labels = MODULE.VideoLabels(
            identity="train:tiny",
            coords_by_t={
                0: np.asarray([[4, 8, 8], [4, 9, 9]]),
                1: np.asarray([[4, 8, 9], [4, 10, 10], [4, 2, 2]]),
            },
            ids_by_t={0: np.asarray([1, 2]), 1: np.asarray([3, 4, 5])},
            edges_by_t={0: ((1, 3), (2, 4))},
            counts={},
            provenance={},
        )
        config = MODULE.CellectAdapterConfig(crop_zyx=(8, 16, 16), candidate_top_k=2)
        example = MODULE.build_crop_example(array, labels, 0, 1, (1.0, 1.0, 1.0), config)
        self.assertEqual(example.pair_source_yxz.shape, (2, 16, 16, 8))
        self.assertEqual(example.positive_mask.sum(), 2)
        self.assertTrue(example.positive_mask[0, 0])
        self.assertTrue(np.all(example.candidate_mask.sum(axis=1) >= 2))
        self.assertEqual(int(MODULE.valid_triplet_rows(example).sum()), 2)

    def test_raw_labels_use_framedet_ids_coordinates_and_guaranteed_negatives(self) -> None:
        gt = {
            "nodes": {
                "node_id": np.asarray([10, 11, 12, 13, 14, 15, 16]),
                "t": np.asarray([0, 1, 0, 1, 1, 1, 0]),
            },
            "edges": np.asarray([[10, 11], [12, 13], [12, 14], [16, 15]]),
        }
        detections = [
            SimpleNamespace(
                frame_index=0,
                node_ids=np.asarray([100, 200, 300]),
                coords_zyx=np.asarray([[1.25, 2.5, 3.75], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            ),
            SimpleNamespace(
                frame_index=1,
                node_ids=np.asarray([101, 201, 202, 999]),
                coords_zyx=np.asarray(
                    [[1.5, 2.75, 4.0], [4.25, 5.25, 6.25], [4.5, 5.5, 6.5], [9, 9, 9]]
                ),
            ),
        ]
        labels = MODULE.raw_labels_from_match(
            gt,
            detections,
            {10: 100, 11: 101, 12: 200, 13: 201, 14: 202, 16: 300, 15: 777},
            identity="train:tiny",
            total_frames=3,
        )

        np.testing.assert_array_equal(labels.ids_by_t[0], [100, 200, 300])
        np.testing.assert_allclose(labels.coords_by_t[0][0], [1.25, 2.5, 3.75])
        self.assertEqual(labels.edges_by_t[0], ((100, 101), (200, 201), (200, 202)))
        self.assertEqual(labels.counts["gt_adjacent_edges"], 4)
        self.assertEqual(labels.counts["raw_positive_edges"], 3)
        self.assertEqual(labels.counts["synthetic_or_pruned_raw_endpoint_edges"], 1)
        self.assertEqual(labels.counts["raw_positive_sources"], 2)
        self.assertEqual(labels.counts["raw_division_sources"], 1)
        self.assertEqual(labels.counts["same_frame_guaranteed_negative_pairs"], 9)
        self.assertEqual(labels.counts["temporal_guaranteed_negative_pairs"], 5)
        self.assertEqual(labels.counts["training_positive_edges"], 3)
        self.assertEqual(labels.counts["training_sample_frames"], 1)

    def test_single_candidate_crop_is_not_optimizer_eligible(self) -> None:
        example = MODULE.CropExample(
            pair_source_yxz=np.ones((2, 16, 16, 8), dtype=np.float32),
            pair_target_yxz=np.ones((2, 16, 16, 8), dtype=np.float32),
            source_points_zyx=np.asarray([[4, 8, 8]]),
            target_points_zyx=np.asarray([[4, 8, 9]]),
            positive_mask=np.asarray([[True]]),
            candidate_mask=np.asarray([[True]]),
            physical_distance_sq=np.asarray([[1.0]]),
        )
        np.testing.assert_array_equal(MODULE.valid_triplet_rows(example), [False])


if __name__ == "__main__":
    unittest.main()
