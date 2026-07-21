from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

import celltrack_alpha.bh_edge as bhe
from celltrack_alpha.bh_edge import (
    ExposureMetadata,
    SparseEdgeConfig,
    SparseEdgeError,
    SparseEdgeScorer,
    UNKNOWN_LABEL,
    gate_sparse_candidates,
    label_sparse_training_pairs,
    learned_edge_probability_map,
    load_edge_artifact,
    save_edge_artifact,
)
from celltrack_alpha.bh_graph import GraphNode, calibrate_v45_graph
from celltrack_alpha.bh_model import DEFAULT_WEIGHT_SHA256, FrameDet, PairContext
from celltrack_alpha.v45_head import (
    AUXILIARY_CHANNELS,
    FEATURE_CHANNELS,
    V45AuxCheckpointIdentity,
    V45AuxReadout,
    auxiliary_state_sha256,
)


class _ForbiddenBlock(nn.Module):
    def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("cross-attention block was called")


class _PairMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(259, 32),
            nn.GELU(),
            nn.Dropout(0.75),
            nn.Linear(32, 1),
        )
        self.row_counts: list[int] = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != 259:
            raise AssertionError("pair MLP received a non-COO feature object")
        self.row_counts.append(int(values.shape[0]))
        return self.layers(values)


class _CheckpointTransformer(nn.Module):
    def __init__(self, seed: int = 7) -> None:
        super().__init__()
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.proj = nn.Linear(64, 128)
        self.norm_in = nn.LayerNorm(128)
        self.blocks = nn.ModuleList([_ForbiddenBlock(), _ForbiddenBlock()])
        self.norm_out = nn.LayerNorm(128)
        self.pair_mlp = _PairMLP()
        torch.random.set_rng_state(state)


def _frame(
    frame: int,
    ids: list[int],
    coords: list[tuple[int, int, int]],
    scores: list[float] | None = None,
) -> FrameDet:
    coordinate_array = np.asarray(coords, dtype=np.int64).reshape(len(ids), 3)
    return FrameDet(
        frame_index=frame,
        node_ids=np.asarray(ids, dtype=np.int64),
        coords_zyx=coordinate_array,
        scores=np.asarray(scores if scores is not None else [0.98] * len(ids), dtype=np.float32),
        grid_coords_zyx=coordinate_array,
        grid_shape_zyx=(64, 64, 64),
    )


def _context(source: FrameDet, target: FrameDet) -> PairContext:
    # Reverse context order to prove features are joined by ID rather than row.
    source_order = np.arange(len(source.node_ids) - 1, -1, -1)
    target_order = np.arange(len(target.node_ids) - 1, -1, -1)
    source_features = np.arange(len(source.node_ids) * 32, dtype=np.float32).reshape(-1, 32) / 100.0
    target_features = np.arange(len(target.node_ids) * 32, dtype=np.float32).reshape(-1, 32) / 80.0
    return PairContext(
        source_frame=source.frame_index,
        target_frame=target.frame_index,
        source_ids=source.node_ids[source_order],
        target_ids=target.node_ids[target_order],
        source_feat=source_features[source_order],
        target_feat=target_features[target_order],
        grid_shape_zyx=source.grid_shape_zyx,
    )


def _fixture() -> tuple[FrameDet, FrameDet, PairContext]:
    source = _frame(0, [1, 2], [(0, 10, 10), (0, 30, 30)], [0.98, 0.99])
    target = _frame(
        1,
        [30, 20, 40, 50, 60],
        [(0, 10, 9), (0, 10, 11), (0, 9, 10), (0, 30, 31), (0, 31, 30)],
        [0.91, 0.92, 0.93, 0.94, 0.95],
    )
    return source, target, _context(source, target)


def _aux_readout(
    *,
    checkpoint_sha256: str = "9" * 64,
    backward_x_um: float = -0.40625,
) -> V45AuxReadout:
    state = {
        "weight": torch.zeros((AUXILIARY_CHANNELS, FEATURE_CHANNELS, 1, 1, 1), dtype=torch.float32),
        "bias": torch.tensor([0.0, 0.0, 0.0, backward_x_um, 0.0], dtype=torch.float32),
    }
    identity = V45AuxCheckpointIdentity(
        manifest_sha256="a" * 64,
        artifact_config_sha256="b" * 64,
        source_sha256="c" * 64,
        weight_sha256=DEFAULT_WEIGHT_SHA256,
        runtime_config_sha256="d" * 64,
        target_contract_sha256="e" * 64,
        core_state_sha256="f" * 64,
        auxiliary_state_sha256=auxiliary_state_sha256(state),
        training_identity_sha256="0" * 64,
        completed_steps=1,
        division_enabled=False,
    )
    return V45AuxReadout(identity, state, checkpoint_sha256=checkpoint_sha256)


class CandidateGateTests(unittest.TestCase):
    def test_physical_gate_ties_and_cardinality_cap(self) -> None:
        source, target, _ = _fixture()
        config = SparseEdgeConfig(radius_um=4.0, neighbor_cap=2)
        first = gate_sparse_candidates(source, target, config=config)
        second = gate_sparse_candidates(source, target, config=config)
        np.testing.assert_array_equal(first.source_ids, second.source_ids)
        np.testing.assert_array_equal(first.target_ids, second.target_ids)
        np.testing.assert_array_equal(first.distance_um, second.distance_um)
        self.assertEqual(first.target_ids[first.source_ids == 1].tolist(), [20, 30])
        self.assertLessEqual(len(first), len(source.node_ids) * config.neighbor_cap)
        self.assertLessEqual(first.stats.peak_retained_neighbors, config.neighbor_cap)

        many_source = _frame(
            2,
            list(range(100, 105)),
            [(0, 20 + index, 20) for index in range(5)],
        )
        many_target = _frame(
            3,
            list(range(200, 230)),
            [(0, 15 + index // 6, 15 + index % 6) for index in range(30)],
        )
        dense_cluster = gate_sparse_candidates(many_source, many_target, config=config)
        self.assertLessEqual(len(dense_cluster), 2 * len(many_source.node_ids))
        counts = np.bincount(dense_cluster.source_index, minlength=len(many_source.node_ids))
        self.assertTrue(np.all(counts <= 2))

    def test_anisotropic_physical_radius_and_config_bounds(self) -> None:
        source = _frame(0, [1], [(0, 10, 10)])
        target = _frame(1, [2, 3], [(3, 10, 10), (0, 19, 10)])
        candidates = gate_sparse_candidates(
            source, target, config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2)
        )
        # z=3 is 4.875 um and y=9 is 3.65625 um at the locked physical scale.
        self.assertEqual(candidates.target_ids.tolist(), [3])
        with self.assertRaisesRegex(SparseEdgeError, r"\[4, 10\]"):
            SparseEdgeConfig(radius_um=3.99)
        with self.assertRaisesRegex(SparseEdgeError, "2, 4, or 8"):
            SparseEdgeConfig(neighbor_cap=3)


class SparseScorerTests(unittest.TestCase):
    def test_blocks_are_bypassed_pair_mlp_is_candidate_chunked_and_dummy_normalizes(self) -> None:
        source, target, context = _fixture()
        transformer = _CheckpointTransformer()
        scorer = SparseEdgeScorer(
            transformer,
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2, pair_chunk_size=1),
        )
        self.assertNotIn("blocks", dict(scorer.named_modules()))
        transformer.pair_mlp.row_counts.clear()
        output = scorer.score_pair(context, source, target)
        self.assertTrue(transformer.pair_mlp.row_counts)
        self.assertLessEqual(max(transformer.pair_mlp.row_counts), 1)
        self.assertLessEqual(len(output), len(source.node_ids) * 2)
        for source_id, no_link in zip(
            output.no_link_source_ids, output.no_link_probabilities, strict=True
        ):
            total = no_link + sum(
                probability
                for (candidate_source, _), probability in output.items()
                if candidate_source == int(source_id)
            )
            self.assertAlmostEqual(float(total), 1.0, places=12)
        self.assertEqual(scorer.trainable_parameter_count, 578)
        self.assertLessEqual(scorer.trainable_parameter_count, 50_000)
        self.assertTrue(all(
            not parameter.requires_grad
            for module in scorer._frozen_modules()
            for parameter in module.parameters()
        ))
        self.assertFalse(scorer.config.division_enabled)
        _, training_logits = scorer.candidate_logits(context, source, target)
        training_logits.sum().backward()
        self.assertIsNotNone(scorer.calibrator.output.bias.grad)
        self.assertTrue(all(
            parameter.grad is None
            for module in scorer._frozen_modules()
            for parameter in module.parameters()
        ))

    def test_chunk_parity_identity_start_and_optional_velocity(self) -> None:
        source, target, context = _fixture()
        transformer_a = _CheckpointTransformer(seed=19)
        transformer_b = _CheckpointTransformer(seed=19)
        small = SparseEdgeScorer(
            transformer_a,
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2, pair_chunk_size=1),
        )
        large = SparseEdgeScorer(
            transformer_b,
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2, pair_chunk_size=64),
        )
        candidates = gate_sparse_candidates(source, target, config=small.config)
        _, small_logits = small.candidate_logits(context, source, target, candidates=candidates)
        # Candidate config includes chunk size, so regenerate under the parity scorer.
        large_candidates = gate_sparse_candidates(source, target, config=large.config)
        _, large_logits = large.candidate_logits(context, source, target, candidates=large_candidates)
        torch.testing.assert_close(small_logits, large_logits, rtol=2e-6, atol=2e-7)
        with_velocity = small.score_pair(
            context,
            source,
            target,
            candidates=candidates,
            previous_velocity_um={1: (0.0, 0.0, 0.4), 2: None},
        )
        without_velocity = small.score_pair(context, source, target, candidates=candidates)
        # Zero-initialized residual makes optional features an exact identity start.
        np.testing.assert_array_equal(with_velocity.probabilities, without_velocity.probabilities)
        self.assertFalse(any("dataset" in name for name in small.architecture["residual_features"]))

    def test_candidate_features_are_frozen_and_logits_are_identity_start(self) -> None:
        source, target, context = _fixture()
        scorer = SparseEdgeScorer(_CheckpointTransformer(), config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2))
        candidates, checkpoint_logits, features = scorer.candidate_features(context, source, target)
        _, logits = scorer.candidate_logits(context, source, target, candidates=candidates)
        torch.testing.assert_close(logits, checkpoint_logits, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(features.shape), (len(candidates), 16))
        self.assertFalse(checkpoint_logits.requires_grad)
        self.assertFalse(features.requires_grad)

    def test_auxiliary_motion_is_target_owned_sparse_and_calibrator_only(self) -> None:
        source, target, context = _fixture()
        scorer = SparseEdgeScorer(
            _CheckpointTransformer(),
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2),
            auxiliary=_aux_readout(),
        )
        candidates = gate_sparse_candidates(source, target, config=scorer.config)
        zeros = torch.zeros(len(candidates), dtype=torch.float32)
        features = scorer._residual_features(zeros, context, source, target, candidates, None)
        observed = {
            (int(source_id), int(target_id)): float(value)
            for source_id, target_id, value in zip(
                candidates.source_ids,
                candidates.target_ids,
                features[:, -1].detach().cpu().numpy(),
                strict=True,
            )
        }
        self.assertAlmostEqual(observed[(1, 20)], 0.0, places=12)
        self.assertAlmostEqual(observed[(1, 30)], 0.5 * (0.8125**2), places=12)
        self.assertEqual(scorer.architecture["auxiliary"]["mode"], "target-point-backward-motion-nll")
        self.assertEqual(scorer.trainable_parameter_count, 578)
        before, after = scorer.candidate_logits(context, source, target, candidates=candidates)
        self.assertIs(before, candidates)
        self.assertTrue(torch.isfinite(after).all().item())
        self.assertFalse(any(parameter.requires_grad for parameter in scorer.auxiliary.parameters()))

    def test_no_full_matrix_api_or_attention_registration(self) -> None:
        source_text = inspect.getsource(bhe)
        for forbidden in (
            "torch.cdist(",
            "np.meshgrid(",
            "MultiheadAttention(",
            "pairwise_distances(",
            "linear_sum_assignment(",
        ):
            self.assertNotIn(forbidden, source_text)
        source, target, context = _fixture()
        scorer = SparseEdgeScorer(
            _CheckpointTransformer(),
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2, pair_chunk_size=2),
        )
        candidates, logits = scorer.candidate_logits(context, source, target)
        self.assertEqual(tuple(logits.shape), (len(candidates),))
        self.assertEqual(candidates.displacement_um.shape, (len(candidates), 3))
        self.assertTrue(all(array.ndim == 1 for array in (
            candidates.source_index,
            candidates.target_index,
            candidates.source_ids,
            candidates.target_ids,
            candidates.distance_um,
        )))

    def test_output_mapping_is_directly_consumable_by_graph_calibration(self) -> None:
        source, target, context = _fixture()
        scorer = SparseEdgeScorer(
            _CheckpointTransformer(),
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2),
        )
        pair = scorer.score_pair(context, source, target)
        merged = learned_edge_probability_map([pair])
        self.assertEqual(merged, pair.as_graph_mapping())
        nodes = [
            GraphNode(
                int(node_id),
                frame.frame_index,
                float(coord[0]),
                float(coord[1]),
                float(coord[2]),
                float(score),
            )
            for frame in (source, target)
            for node_id, coord, score in zip(
                frame.node_ids, frame.coords_zyx, frame.scores, strict=True
            )
        ]
        result = calibrate_v45_graph(nodes, learned_edge_probabilities=pair)
        self.assertTrue(result.stats["learned_probabilities_present"])
        self.assertEqual(result.stats["learned_probability_count"], len(pair))


class SparseLabelTests(unittest.TestCase):
    def test_partial_or_unmatched_endpoints_are_unknown_not_negative(self) -> None:
        source, target, _ = _fixture()
        candidates = gate_sparse_candidates(
            source, target, config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2)
        )
        labels = label_sparse_training_pairs(
            candidates,
            detection_to_gt={1: 101, 2: None, 20: 201, 30: 202, 50: 250},
            gt_edges=[(101, 201)],
        )
        observed = {
            (int(source_id), int(target_id)): int(label)
            for source_id, target_id, label in zip(
                labels.source_ids, labels.target_ids, labels.labels, strict=True
            )
        }
        self.assertEqual(observed[(1, 20)], 1)
        self.assertEqual(observed[(1, 30)], 0)
        self.assertEqual(observed[(2, 50)], UNKNOWN_LABEL)
        self.assertTrue(np.all(labels.known == (labels.labels != UNKNOWN_LABEL)))
        self.assertEqual(int((labels.labels == 0).sum()), 1)


class EdgeArtifactTests(unittest.TestCase):
    def test_deterministic_finite_json_roundtrip_and_hashes(self) -> None:
        source, target, context = _fixture()
        transformer = _CheckpointTransformer(seed=23)
        scorer = SparseEdgeScorer(
            transformer,
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2, pair_chunk_size=2),
        )
        equivalent = SparseEdgeScorer(
            _CheckpointTransformer(seed=23),
            config=SparseEdgeConfig(radius_um=4.0, neighbor_cap=2, pair_chunk_size=2),
        )
        with torch.no_grad():
            scorer.calibrator.output.weight.fill_(0.01)
            scorer.calibrator.output.bias.fill_(-0.02)
            equivalent.calibrator.output.weight.fill_(0.01)
            equivalent.calibrator.output.bias.fill_(-0.02)
        exposure = ExposureMetadata(
            fold=0,
            source_groups=("fit-a", "fit-b"),
            source_video_ids=("video-01", "video-02"),
            split_manifest_sha256="1" * 64,
            provenance_manifest_sha256="2" * 64,
        )
        before = scorer.score_pair(context, source, target)
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            identity_a = save_edge_artifact(first, scorer, exposure)
            identity_b = save_edge_artifact(second, equivalent, exposure)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(identity_a, identity_b)
            self.assertEqual(
                hashlib.sha256(scorer.architecture_sha256.encode("ascii")).digest_size,
                32,
            )
            restored, restored_identity = load_edge_artifact(first, transformer)
            after = restored.score_pair(context, source, target)
            self.assertEqual(restored_identity, identity_a)
            np.testing.assert_array_equal(before.probabilities, after.probabilities)
            self.assertTrue(all(
                torch.isfinite(parameter).all().item()
                for parameter in restored.calibrator.parameters()
            ))
            self.assertLessEqual(restored.trainable_parameter_count, 50_000)
            self.assertNotIn(b"pickle", first.read_bytes().lower())

    def test_nonfinite_residual_cannot_be_serialized(self) -> None:
        scorer = SparseEdgeScorer(_CheckpointTransformer())
        with torch.no_grad():
            scorer.calibrator.output.bias.fill_(float("nan"))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(bhe.EdgeArtifactError, "not finite"):
                save_edge_artifact(
                    Path(temporary) / "bad.json",
                    scorer,
                    ExposureMetadata(fold=-1),
                )

    def test_auxiliary_artifact_requires_the_exact_head_identity(self) -> None:
        source, target, context = _fixture()
        transformer = _CheckpointTransformer(seed=67)
        auxiliary = _aux_readout()
        scorer = SparseEdgeScorer(transformer, auxiliary=auxiliary)
        with torch.no_grad():
            scorer.calibrator.output.weight.fill_(0.01)
        expected = scorer.score_pair(context, source, target)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "edge.json"
            identity = save_edge_artifact(artifact, scorer, ExposureMetadata(fold=0))
            restored, actual = load_edge_artifact(artifact, transformer, auxiliary=auxiliary)
            self.assertEqual(actual, identity)
            np.testing.assert_array_equal(
                restored.score_pair(context, source, target).probabilities,
                expected.probabilities,
            )
            with self.assertRaisesRegex(bhe.EdgeArtifactError, "requires a validated"):
                load_edge_artifact(artifact, transformer)
            with self.assertRaisesRegex(bhe.EdgeArtifactError, "checkpoint identity mismatch"):
                load_edge_artifact(
                    artifact,
                    transformer,
                    auxiliary=_aux_readout(checkpoint_sha256="8" * 64),
                )
            artifact.write_bytes(
                artifact.read_bytes().replace(
                    b"celltrack-alpha.bh-sparse-edge-artifact.v2",
                    b"celltrack-alpha.bh-sparse-edge-artifact.v1",
                )
            )
            with self.assertRaises(bhe.EdgeArtifactError):
                load_edge_artifact(artifact, transformer, auxiliary=auxiliary)


if __name__ == "__main__":
    unittest.main()
