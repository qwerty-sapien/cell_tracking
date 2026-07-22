from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.cellect_adapter import (  # noqa: E402
    CellectAdapterConfig,
    CellectContractError,
    CellectEmbeddingAdapter,
    batch_hard_temporal_triplet,
    cellect_log_preprocess,
    rank_accuracy,
    sample_embeddings_zyx,
    topk_candidate_mask,
    validate_upstream_tree,
)


class _Backbone(nn.Module):
    def forward(self, value: torch.Tensor):
        shape = (value.shape[0], 64, *value.shape[2:])
        feature = torch.arange(64, dtype=value.dtype, device=value.device).view(1, 64, 1, 1, 1)
        feature = feature.expand(shape)
        return value[:, :1], value[:, :1], feature, value[:, :1], value[:, :1]


class CellectAdapterTests(unittest.TestCase):
    def test_division_output_cannot_be_enabled(self) -> None:
        with self.assertRaisesRegex(CellectContractError, "division output remains disabled"):
            CellectAdapterConfig(division_output_enabled=True)

    def test_upstream_contract_hashes_are_pinned(self) -> None:
        observed = validate_upstream_tree(ROOT / "external/CELLECT")
        self.assertEqual(
            observed["upstream_commit"],
            "3586070926f7f1fd5d8df37456861d22bdc63236",
        )
        self.assertEqual(
            observed["weight_sha256"],
            "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5",
        )
        self.assertEqual(
            observed["model_source_sha256"],
            "3b44f7c64c4b948ab7cc14bb32d5a86f61d3f4cff0ee3425a5ec32cc6c17d3ba",
        )

    def test_preprocess_matches_published_transform_without_mutation(self) -> None:
        source = torch.tensor([[[[[0.0, 2.0]]], [[[4.0, 8.0]]]]])
        before = source.clone()
        observed = cellect_log_preprocess(source)
        expected = torch.log1p(torch.tensor([[[[[1902.0, 1902.0]]], [[[1904.0, 1908.0]]]]]))
        torch.testing.assert_close(observed, expected)
        torch.testing.assert_close(source, before)

    def test_embedding_adapter_normalizes_and_samples_zyx(self) -> None:
        adapter = CellectEmbeddingAdapter(_Backbone())
        feature = adapter(torch.ones((1, 2, 16, 16, 3), dtype=torch.float32))
        sampled = sample_embeddings_zyx(feature, torch.tensor([[2, 4, 5], [0, 1, 1]]))
        self.assertEqual(tuple(sampled.shape), (2, 64))
        torch.testing.assert_close(torch.linalg.vector_norm(sampled, dim=1), torch.ones(2))

    def test_topk_gate_uses_physical_distance_and_stable_index_ties(self) -> None:
        source = np.asarray([[0, 0, 0]], dtype=np.float64)
        target = np.asarray([[0, 3, 0], [1, 0, 0], [0, 0, 3]], dtype=np.float64)
        observed = topk_candidate_mask(source, target, (4.0, 1.0, 1.0), k=2)
        np.testing.assert_array_equal(observed, [[True, False, True]])

    def test_hard_triplet_supports_multiple_division_positives(self) -> None:
        source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
        positive = torch.tensor([[True, True, False], [False, False, True]])
        candidates = torch.ones_like(positive)
        loss, counts = batch_hard_temporal_triplet(
            source,
            target,
            positive,
            candidate_mask=candidates,
            margin=0.3,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(counts, {"anchors": 2, "valid_anchors": 2})

    def test_rank_accuracy_only_counts_positive_rows(self) -> None:
        distances = np.asarray([[0.2, 0.1], [0.1, 0.2], [0.5, 0.4]])
        positive = np.asarray([[False, True], [False, True], [False, False]])
        candidates = np.ones_like(positive)
        self.assertEqual(
            rank_accuracy(distances, positive, candidates),
            {"rows": 2, "correct": 1, "top1_accuracy": 0.5},
        )


if __name__ == "__main__":
    unittest.main()
