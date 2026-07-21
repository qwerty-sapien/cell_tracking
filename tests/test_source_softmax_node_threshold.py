from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from analyze_source_softmax_node_threshold import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    _filtered_video,
    _summarize,
)
from celltrack_alpha.bh_model import (  # noqa: E402
    CacheIdentity,
    FrameDet,
    PairContext,
    SparseInferenceCache,
    V45_CONFIG,
    v45_config_sha,
)


class SourceSoftmaxNodeThresholdTests(unittest.TestCase):
    def test_frozen_holdout_execution_hash_binds_effective_threshold_to_0970(self) -> None:
        report_path = (
            ROOT
            / "reports/g02/source_softmax_edge_f0_fit117_holdout_dry_v1/report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        declared_execution = report["execution"]["video_config_sha256"]

        self.assertEqual(V45_CONFIG.detection_threshold, 0.970)
        self.assertEqual(DEFAULT_THRESHOLDS[:4], (0.970, 0.975, 0.980, 0.985))
        self.assertEqual(
            v45_config_sha(pair_batch_size=1, view_batch_size=1, d4_views=(0,)),
            declared_execution,
        )

    def test_threshold_replay_filters_detections_and_pair_features_together(self) -> None:
        first = FrameDet(
            frame_index=0,
            node_ids=np.asarray([10, 11], dtype=np.int64),
            coords_zyx=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            scores=np.asarray([0.984, 0.997], dtype=np.float32),
            grid_coords_zyx=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            grid_shape_zyx=(8, 8, 8),
        )
        second = FrameDet(
            frame_index=1,
            node_ids=np.asarray([20, 21], dtype=np.int64),
            coords_zyx=np.asarray([[1, 2, 4], [4, 5, 7]], dtype=np.int64),
            scores=np.asarray([0.999, 0.980], dtype=np.float32),
            grid_coords_zyx=np.asarray([[1, 2, 4], [4, 5, 7]], dtype=np.int64),
            grid_shape_zyx=(8, 8, 8),
        )
        context = PairContext(
            source_frame=0,
            target_frame=1,
            source_ids=np.asarray([10, 11], dtype=np.int64),
            target_ids=np.asarray([20, 21], dtype=np.int64),
            source_feat=np.arange(64, dtype=np.float32).reshape(2, 32),
            target_feat=np.arange(64, 128, dtype=np.float32).reshape(2, 32),
            grid_shape_zyx=(8, 8, 8),
        )
        cache = SparseInferenceCache(
            identity=CacheIdentity(
                config_sha256="0" * 64,
                source_sha256="1" * 64,
                weight_sha256="2" * 64,
                pair_batch_size=1,
                view_batch_size=1,
            ),
            detections=(first, second),
            contexts=(context,),
            edges=(),
        )

        filtered = _filtered_video(cache, 0.985)

        self.assertEqual(filtered.detections[0].node_ids.tolist(), [11])
        self.assertEqual(filtered.detections[1].node_ids.tolist(), [20])
        self.assertEqual(filtered.contexts[0].source_ids.tolist(), [11])
        self.assertEqual(filtered.contexts[0].target_ids.tolist(), [20])
        np.testing.assert_array_equal(
            filtered.contexts[0].source_feat,
            context.source_feat[[1]],
        )
        np.testing.assert_array_equal(
            filtered.contexts[0].target_feat,
            context.target_feat[[0]],
        )

    def test_guard_rejects_higher_adjusted_score_with_unsafe_recall_loss(self) -> None:
        def row(threshold: float, adjusted: float, recall: float) -> dict:
            return {
                "threshold": threshold,
                "edge_tp": 80.0,
                "edge_fp": 10.0,
                "edge_fn": 10.0,
                "edge_jaccard": 0.8,
                "adj_edge_jaccard": adjusted,
                "node_recall": recall,
                "estimated_number_of_nodes": 100.0,
                "num_pred_nodes": 100.0,
            }

        samples = [
            {
                "identity": "train:44b6_a",
                "scenario": "44b6",
                "thresholds": [row(0.970, 0.80, 0.99), row(0.985, 0.81, 0.99)],
            },
            {
                "identity": "train:6bba_a",
                "scenario": "6bba",
                "thresholds": [row(0.970, 0.80, 0.99), row(0.985, 0.90, 0.98)],
            },
        ]

        _aggregate, policy = _summarize(
            samples,
            (0.970, 0.985),
            recall_tolerance=0.002,
        )

        self.assertEqual(policy["44b6"]["selected_threshold"], 0.985)
        self.assertEqual(policy["6bba"]["selected_threshold"], 0.970)
        self.assertEqual(policy["all"]["selected_threshold"], 0.970)
        self.assertEqual(policy["all"]["unguarded_optimum_threshold"], 0.985)


if __name__ == "__main__":
    unittest.main()
