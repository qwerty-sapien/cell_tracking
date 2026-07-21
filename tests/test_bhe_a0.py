from __future__ import annotations

from unittest import mock
import unittest

import numpy as np
import torch
import torch.nn as nn

import celltrack_alpha.bh_edge as bhe
from celltrack_alpha.bh_edge import (
    A0CheckpointScorer,
    A0ScorerConfig,
    CandidateStats,
    SparseCandidateCOO,
    SparseEdgeConfig,
    SparseEdgeError,
    SparseEdgeScorer,
    gate_sparse_candidates,
)
from celltrack_alpha.bh_model import FrameDet, PairContext, positional_features


class _RecordingBlock(nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.03125, dtype=torch.float32))
        self.fail = fail
        self.calls: list[tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]] = []

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("unsupported attention component")
        if kv_mask is None or kv_mask.dtype != torch.bool:
            raise AssertionError("attention did not receive a boolean KV mask")
        self.calls.append(
            (
                kv_mask.detach().cpu().clone(),
                tuple(q.shape),
                tuple(kv.shape),
            )
        )
        weights = kv_mask.to(dtype=kv.dtype).unsqueeze(-1)
        mean = (kv * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return q + self.gain * mean


class _RecordingPairMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(259, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
        )
        self.row_counts: list[int] = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != 259:
            raise AssertionError("A0 pair MLP must receive candidate rows, never a pair matrix")
        self.row_counts.append(int(values.shape[0]))
        return self.layers(values)


class _A0Transformer(nn.Module):
    def __init__(self, seed: int = 17, *, fail_blocks: bool = False) -> None:
        super().__init__()
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.proj = nn.Linear(64, 128)
        self.norm_in = nn.LayerNorm(128)
        self.blocks = nn.ModuleList(
            [_RecordingBlock(fail=fail_blocks) for _ in range(4)]
        )
        self.norm_out = nn.LayerNorm(128)
        self.pair_mlp = _RecordingPairMLP()
        torch.random.set_rng_state(state)


def _frame(
    frame: int,
    ids: list[int] | np.ndarray,
    coords: list[tuple[int, int, int]] | np.ndarray,
    *,
    grid_shape: tuple[int, int, int] = (64, 64, 64),
) -> FrameDet:
    ids_array = np.asarray(ids, dtype=np.int64)
    coords_array = np.asarray(coords, dtype=np.int64).reshape(len(ids_array), 3)
    return FrameDet(
        frame_index=frame,
        node_ids=ids_array,
        coords_zyx=coords_array,
        scores=np.full(len(ids_array), 0.95, dtype=np.float32),
        grid_coords_zyx=coords_array,
        grid_shape_zyx=grid_shape,
    )


def _context(source: FrameDet, target: FrameDet) -> PairContext:
    source_order = np.arange(len(source.node_ids) - 1, -1, -1)
    target_order = np.arange(len(target.node_ids) - 1, -1, -1)
    source_features = (
        np.arange(len(source.node_ids) * 32, dtype=np.float32).reshape(-1, 32)
        / max(len(source.node_ids) * 32, 1)
    )
    target_features = (
        np.arange(len(target.node_ids) * 32, dtype=np.float32).reshape(-1, 32)
        / max(len(target.node_ids) * 32, 1)
    )
    return PairContext(
        source_frame=source.frame_index,
        target_frame=target.frame_index,
        source_ids=source.node_ids[source_order],
        target_ids=target.node_ids[target_order],
        source_feat=source_features[source_order],
        target_feat=target_features[target_order],
        grid_shape_zyx=source.grid_shape_zyx,
    )


def _checkpoint_inputs(
    context: PairContext,
    frame: FrameDet,
    *,
    source: bool,
) -> torch.Tensor:
    context_ids = context.source_ids if source else context.target_ids
    context_features = context.source_feat if source else context.target_feat
    rows = {int(node_id): index for index, node_id in enumerate(context_ids)}
    features = context_features[[rows[int(node_id)] for node_id in frame.node_ids]]
    time = np.zeros(len(frame.node_ids), dtype=np.int64) if source else np.ones(
        len(frame.node_ids), dtype=np.int64
    )
    pos = positional_features(
        np.column_stack((time, frame.grid_coords_zyx)),
        (2,) + context.grid_shape_zyx,
    )
    return torch.from_numpy(np.ascontiguousarray(np.concatenate((features, pos), axis=1)))


def _dense_reference(
    transformer: _A0Transformer,
    context: PairContext,
    source: FrameDet,
    target: FrameDet,
    candidates: SparseCandidateCOO,
) -> np.ndarray:
    source_input = _checkpoint_inputs(context, source, source=True).unsqueeze(0)
    target_input = _checkpoint_inputs(context, target, source=False).unsqueeze(0)
    source_mask = torch.ones((1, len(source.node_ids)), dtype=torch.bool)
    target_mask = torch.ones((1, len(target.node_ids)), dtype=torch.bool)
    with torch.inference_mode():
        q = transformer.norm_in(transformer.proj(source_input))
        k = transformer.norm_in(transformer.proj(target_input))
        for block in transformer.blocks:
            q = block(q, k, kv_mask=target_mask)
            k = block(k, q, kv_mask=source_mask)
        q = transformer.norm_out(q)[0]
        k = transformer.norm_out(k)[0]
        source_grid, target_grid = torch.meshgrid(
            torch.arange(len(source.node_ids)),
            torch.arange(len(target.node_ids)),
            indexing="ij",
        )
        relative = (
            torch.from_numpy(source.coords_zyx.astype(np.float32))[:, None, :]
            - torch.from_numpy(target.coords_zyx.astype(np.float32))[None, :, :]
        ) / 100.0
        pair_features = torch.cat(
            (
                q[source_grid.reshape(-1)],
                k[target_grid.reshape(-1)],
                relative.reshape(-1, 3),
            ),
            dim=1,
        )
        logits = transformer.pair_mlp(pair_features).reshape(
            len(source.node_ids), len(target.node_ids)
        )
        probabilities = torch.softmax(logits, dim=0)
    return probabilities[
        torch.from_numpy(np.array(candidates.source_index, copy=True)),
        torch.from_numpy(np.array(candidates.target_index, copy=True)),
    ].numpy()


def _permuted(candidates: SparseCandidateCOO, order: np.ndarray) -> SparseCandidateCOO:
    return SparseCandidateCOO(
        source_index=candidates.source_index[order],
        target_index=candidates.target_index[order],
        source_ids=candidates.source_ids[order],
        target_ids=candidates.target_ids[order],
        displacement_um=candidates.displacement_um[order],
        distance_um=candidates.distance_um[order],
        source_density=candidates.source_density[order],
        target_density=candidates.target_density[order],
        stats=candidates.stats,
    )


class A0CheckpointScorerTests(unittest.TestCase):
    def test_tiny_dense_reference_equivalence_and_target_normalization(self) -> None:
        source = _frame(0, [9, 1], [(0, 10, 10), (0, 12, 10)])
        target = _frame(1, [8, 2], [(0, 11, 10), (0, 10, 12)])
        context = _context(source, target)
        candidates = gate_sparse_candidates(
            source,
            target,
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2),
        )
        self.assertEqual(len(candidates), 4)
        transformer = _A0Transformer(seed=29)
        scorer = A0CheckpointScorer(
            transformer, config=A0ScorerConfig(neighbor_cap=2)
        )
        transformer.pair_mlp.row_counts.clear()
        actual = scorer.score_pair(
            context, source, target, candidates=candidates
        )
        expected = _dense_reference(transformer, context, source, target, candidates)
        np.testing.assert_allclose(actual.probabilities, expected, rtol=2e-6, atol=2e-7)
        for target_id in candidates.target_ids:
            observed = actual.probabilities[candidates.target_ids == target_id].sum()
            self.assertAlmostEqual(float(observed), 1.0, places=12)
        self.assertFalse(hasattr(actual, "no_link_probabilities"))
        self.assertIs(scorer.proj, transformer.proj)
        self.assertIs(scorer.blocks, transformer.blocks)
        self.assertEqual(scorer.trainable_parameter_count, 0)
        self.assertEqual(actual.stats.attention_component_count, 1)
        self.assertLessEqual(max(transformer.pair_mlp.row_counts), 256)

    def test_similar_component_batching_uses_correct_padding_masks(self) -> None:
        source = _frame(
            0,
            [1, 2, 3, 10, 11, 12, 13],
            [
                (0, 10, 10),
                (0, 11, 10),
                (0, 12, 10),
                (0, 40, 40),
                (0, 41, 40),
                (0, 42, 40),
                (0, 43, 40),
            ],
        )
        target = _frame(1, [100, 200], [(0, 11, 10), (0, 41, 40)])
        candidates = gate_sparse_candidates(
            source,
            target,
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2),
        )
        transformer = _A0Transformer()
        scorer = A0CheckpointScorer(
            transformer, config=A0ScorerConfig(neighbor_cap=2)
        )
        transformer.pair_mlp.row_counts.clear()
        output = scorer.score_pair(
            _context(source, target), source, target, candidates=candidates
        )
        self.assertEqual(output.stats.component_count, 2)
        self.assertEqual(output.stats.batch_count, 1)
        self.assertEqual(output.stats.max_batch_components, 2)
        self.assertEqual(output.stats.max_padded_source_nodes, 4)
        self.assertEqual(output.stats.max_padded_target_nodes, 1)
        self.assertEqual(output.stats.peak_padding_nodes, 1)
        self.assertEqual(output.stats.peak_attention_cells, 8)
        self.assertEqual(transformer.pair_mlp.row_counts, [len(candidates)])
        first_block = transformer.blocks[0]
        self.assertEqual(len(first_block.calls), 2)
        target_mask, q_shape, kv_shape = first_block.calls[0]
        source_mask, reverse_q_shape, reverse_kv_shape = first_block.calls[1]
        self.assertEqual(q_shape, (2, 4, 128))
        self.assertEqual(kv_shape, (2, 1, 128))
        self.assertTrue(torch.equal(target_mask, torch.ones((2, 1), dtype=torch.bool)))
        self.assertEqual(reverse_q_shape, (2, 1, 128))
        self.assertEqual(reverse_kv_shape, (2, 4, 128))
        self.assertTrue(
            torch.equal(
                source_mask,
                torch.tensor([[True, True, True, False], [True, True, True, True]]),
            )
        )

    def test_component_and_candidate_order_are_deterministic_and_sparse(self) -> None:
        source = _frame(
            0,
            [1, 2, 3, 10, 11, 12, 13],
            [
                (0, 10, 10),
                (0, 11, 10),
                (0, 12, 10),
                (0, 40, 40),
                (0, 41, 40),
                (0, 42, 40),
                (0, 43, 40),
            ],
        )
        target = _frame(1, [100, 200], [(0, 11, 10), (0, 41, 40)])
        context = _context(source, target)
        candidates = gate_sparse_candidates(
            source,
            target,
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2),
        )
        permuted = _permuted(candidates, np.arange(len(candidates) - 1, -1, -1))
        transformer = _A0Transformer(seed=41)
        scorer = A0CheckpointScorer(
            transformer, config=A0ScorerConfig(neighbor_cap=2)
        )
        first = scorer.score_pair(context, source, target, candidates=candidates)
        second = scorer.score_pair(context, source, target, candidates=permuted)
        transformer.train()
        third = scorer.score_pair(context, source, target, candidates=candidates)
        self.assertEqual(first.as_graph_mapping(), second.as_graph_mapping())
        self.assertEqual(first.as_graph_mapping(), third.as_graph_mapping())
        self.assertEqual(first.stats, second.stats)
        self.assertEqual(first.stats, third.stats)
        self.assertEqual(set(first), set(zip(candidates.source_ids, candidates.target_ids)))
        self.assertNotIn((1, 200), first)
        with self.assertRaises(KeyError):
            _ = first[(1, 200)]

    def test_oversized_and_unsupported_components_use_projection_fallback(self) -> None:
        source_coords = [
            (0, 20 + row, 20 + column)
            for row in range(8)
            for column in range(8)
        ]
        source = _frame(0, list(range(1, 65)), source_coords)
        target = _frame(1, [1000], [(0, 23, 23)])
        candidates = gate_sparse_candidates(source, target)
        self.assertEqual(len(candidates), 64)
        transformer = _A0Transformer(seed=53)
        scorer = A0CheckpointScorer(transformer)
        transformer.pair_mlp.row_counts.clear()
        output = scorer.score_pair(
            _context(source, target), source, target, candidates=candidates
        )
        self.assertEqual(output.stats.max_component_nodes, 65)
        self.assertEqual(output.stats.node_budget_fallback_component_count, 1)
        self.assertEqual(output.stats.fallback_component_fraction, 1.0)
        self.assertEqual(output.stats.fallback_candidate_fraction, 1.0)
        self.assertEqual(output.stats.attention_component_count, 0)
        self.assertEqual(sum(len(block.calls) for block in transformer.blocks), 0)
        self.assertLessEqual(max(transformer.pair_mlp.row_counts), 256)
        _, a0_logits, _ = scorer.candidate_logits(
            _context(source, target), source, target, candidates=candidates
        )
        _, projection_logits = SparseEdgeScorer(transformer).candidate_logits(
            _context(source, target), source, target, candidates=candidates
        )
        torch.testing.assert_close(a0_logits, projection_logits, rtol=0.0, atol=0.0)

        small_source = _frame(0, [1, 2], [(0, 10, 10), (0, 11, 10)])
        small_target = _frame(1, [3], [(0, 10, 11)])
        small_candidates = gate_sparse_candidates(small_source, small_target)
        unsupported = _A0Transformer(fail_blocks=True)
        unsupported_scorer = A0CheckpointScorer(unsupported)
        unsupported_output = unsupported_scorer.score_pair(
            _context(small_source, small_target),
            small_source,
            small_target,
            candidates=small_candidates,
        )
        self.assertEqual(
            unsupported_output.stats.unsupported_fallback_component_count, 1
        )
        self.assertEqual(unsupported_output.stats.fallback_component_count, 1)
        self.assertTrue(np.isfinite(unsupported_output.probabilities).all())

        with self.assertRaisesRegex(SparseEdgeError, r"\[2, 64\]"):
            A0ScorerConfig(max_component_nodes=65)
        with self.assertRaisesRegex(SparseEdgeError, r"\[1, 65536\]"):
            A0ScorerConfig(attention_cell_budget=65_537)
        with self.assertRaisesRegex(SparseEdgeError, r"\[1, 256\]"):
            A0ScorerConfig(pair_chunk_size=257)

    def test_adversarial_sparse_frame_shapes_never_allocate_full_pair_matrix(self) -> None:
        source_count = 4_096
        target_count = 8_192
        source_coords = np.column_stack(
            (
                np.zeros(source_count, dtype=np.int64),
                np.zeros(source_count, dtype=np.int64),
                np.arange(source_count, dtype=np.int64),
            )
        )
        target_coords = np.column_stack(
            (
                np.zeros(target_count, dtype=np.int64),
                np.zeros(target_count, dtype=np.int64),
                np.arange(target_count, dtype=np.int64),
            )
        )
        grid_shape = (1, 1, target_count)
        source = _frame(
            0,
            np.arange(source_count, dtype=np.int64),
            source_coords,
            grid_shape=grid_shape,
        )
        target = _frame(
            1,
            np.arange(100_000, 100_000 + target_count, dtype=np.int64),
            target_coords,
            grid_shape=grid_shape,
        )
        source_index = 123
        target_index = 123
        candidates = SparseCandidateCOO(
            source_index=np.asarray([source_index], dtype=np.int64),
            target_index=np.asarray([target_index], dtype=np.int64),
            source_ids=np.asarray([source.node_ids[source_index]], dtype=np.int64),
            target_ids=np.asarray([target.node_ids[target_index]], dtype=np.int64),
            displacement_um=np.zeros((1, 3), dtype=np.float64),
            distance_um=np.zeros(1, dtype=np.float64),
            source_density=np.zeros(1, dtype=np.float64),
            target_density=np.zeros(1, dtype=np.float64),
            stats=CandidateStats(
                source_count=source_count,
                target_count=target_count,
                candidate_count=1,
                radius_um=10.0,
                neighbor_cap=8,
                peak_radius_neighbors=1,
                peak_retained_neighbors=1,
            ),
        )
        transformer = _A0Transformer(seed=67)
        scorer = A0CheckpointScorer(transformer)
        original_np_empty = np.empty
        original_torch_empty = torch.empty

        def guarded_np_empty(shape: object, *args: object, **kwargs: object) -> np.ndarray:
            if tuple(np.atleast_1d(shape).tolist()) in {
                (source_count, target_count),
                (target_count, source_count),
            }:
                raise AssertionError("full-frame NumPy pair matrix requested")
            return original_np_empty(shape, *args, **kwargs)

        def guarded_torch_empty(
            shape: object, *args: object, **kwargs: object
        ) -> torch.Tensor:
            if tuple(np.atleast_1d(shape).tolist()) in {
                (source_count, target_count),
                (target_count, source_count),
            }:
                raise AssertionError("full-frame Torch pair matrix requested")
            return original_torch_empty(shape, *args, **kwargs)

        with mock.patch.object(bhe.np, "empty", side_effect=guarded_np_empty), mock.patch.object(
            bhe.torch, "empty", side_effect=guarded_torch_empty
        ):
            output = scorer.score_pair(
                _context(source, target), source, target, candidates=candidates
            )
        self.assertEqual(output.probabilities.tolist(), [1.0])
        self.assertEqual(output.stats.max_component_nodes, 2)
        self.assertEqual(output.stats.max_padded_source_nodes, 1)
        self.assertEqual(output.stats.max_padded_target_nodes, 1)
        for block in transformer.blocks:
            for _mask, q_shape, kv_shape in block.calls:
                self.assertEqual(q_shape[0], 1)
                self.assertEqual(q_shape[1], 1)
                self.assertEqual(kv_shape[1], 1)


if __name__ == "__main__":
    unittest.main()
