"""Focused contract tests for the fit-only conflict-contrastive edge residual."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.fit_conflict_residual import (
    ConflictFitError,
    conflict_loss,
    select_parent_sources,
    select_registry_sources,
)
from celltrack_alpha.bh_edge import (
    ExposureMetadata,
    SparseEdgeConfig,
    SparseEdgeScorer,
    gate_sparse_candidates,
    load_edge_artifact,
    save_edge_artifact,
)
from celltrack_alpha.bh_graph import GraphNode, calibrate_v45_graph
from celltrack_alpha.bh_model import FrameDet, PairContext, VideoInference
from celltrack_alpha.edge_data import (
    ConflictPairExamples,
    SOURCE_CONFLICT,
    TARGET_CONFLICT,
    conflict_examples_from_video,
)


class _PairMLP(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.Linear(259, 16), nn.GELU(), nn.Linear(16, 1))


class _Transformer(nn.Module):
    def __init__(self, seed: int = 71) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.proj = nn.Linear(64, 128)
            self.norm_in = nn.LayerNorm(128)
            self.norm_out = nn.LayerNorm(128)
            self.pair_mlp = _PairMLP()


def _frame(frame: int, ids: list[int], coords: list[tuple[int, int, int]]) -> FrameDet:
    points = np.asarray(coords, dtype=np.int64)
    return FrameDet(
        frame_index=frame,
        node_ids=np.asarray(ids, dtype=np.int64),
        coords_zyx=points,
        scores=np.full(len(ids), 0.95, dtype=np.float32),
        grid_coords_zyx=points,
        grid_shape_zyx=(8, 64, 64),
    )


def _context(source: FrameDet, target: FrameDet) -> PairContext:
    source_features = np.arange(len(source.node_ids) * 32, dtype=np.float32).reshape(-1, 32) / 37.0
    target_features = np.arange(len(target.node_ids) * 32, dtype=np.float32).reshape(-1, 32) / 41.0
    return PairContext(
        source_frame=source.frame_index,
        target_frame=target.frame_index,
        source_ids=source.node_ids,
        target_ids=target.node_ids,
        source_feat=source_features,
        target_feat=target_features,
        grid_shape_zyx=source.grid_shape_zyx,
    )


def _video() -> VideoInference:
    source = _frame(0, [1, 2], [(0, 10, 10), (0, 10, 14)])
    target = _frame(1, [3, 4], [(0, 10, 11), (0, 10, 13)])
    return VideoInference((source, target), (_context(source, target),), d4_views=(0,))


class ConflictConstructionTests(unittest.TestCase):
    def test_true_misses_pair_only_with_actual_source_and_target_conflicts(self) -> None:
        scorer = SparseEdgeScorer(
            _Transformer(), config=SparseEdgeConfig(radius_um=10.0, neighbor_cap=8)
        )
        video = _video()
        selected = {(1, 3), (2, 4)}
        mapping = {1: 101, 2: 102, 3: 103, 4: 104}
        truth = {(101, 104), (102, 103)}
        examples = conflict_examples_from_video(scorer, video, mapping, truth, selected)
        observed = {
            (tuple(true), tuple(competitor), int(kind))
            for true, competitor, kind in zip(
                examples.true_pairs,
                examples.competitor_pairs,
                examples.conflict_kind,
                strict=True,
            )
        }
        self.assertEqual(
            observed,
            {
                ((1, 4), (1, 3), SOURCE_CONFLICT),
                ((1, 4), (2, 4), TARGET_CONFLICT),
                ((2, 3), (2, 4), SOURCE_CONFLICT),
                ((2, 3), (1, 3), TARGET_CONFLICT),
            },
        )
        self.assertEqual(examples.reachable_miss_count, 2)
        self.assertEqual(examples.source_conflict_count, 2)
        self.assertEqual(examples.target_conflict_count, 2)
        self.assertEqual(examples.candidate_count, 4)
        self.assertFalse(scorer.config.division_enabled)

    def test_division_truth_is_not_recast_as_a_negative_continuation(self) -> None:
        scorer = SparseEdgeScorer(_Transformer(), config=SparseEdgeConfig())
        examples = conflict_examples_from_video(
            scorer,
            _video(),
            {1: 101, 2: 102, 3: 103, 4: 104},
            {(101, 103), (101, 104)},
            {(1, 3), (2, 4)},
        )
        self.assertEqual(len(examples), 0)
        self.assertGreater(examples.skipped_division_truth, 0)


class ResidualSemanticsTests(unittest.TestCase):
    @staticmethod
    def _examples(true_base: float, competitor_base: float) -> ConflictPairExamples:
        return ConflictPairExamples(
            true_features=np.zeros((1, 16), dtype=np.float32),
            competitor_features=np.zeros((1, 16), dtype=np.float32),
            true_checkpoint_logits=np.asarray([true_base], dtype=np.float32),
            competitor_checkpoint_logits=np.asarray([competitor_base], dtype=np.float32),
            conflict_kind=np.asarray([SOURCE_CONFLICT], dtype=np.int8),
            true_pairs=np.asarray([[1, 4]], dtype=np.int64),
            competitor_pairs=np.asarray([[1, 3]], dtype=np.int64),
            candidate_count=2,
            reachable_miss_count=1,
            source_conflict_count=1,
            target_conflict_count=0,
            skipped_division_truth=0,
            skipped_without_candidate_conflict=0,
            candidate_topology_sha256="a" * 64,
        )

    def test_pairwise_loss_rewards_true_edge_ranking(self) -> None:
        scorer = SparseEdgeScorer(_Transformer())
        device = scorer.calibrator.output.weight.device
        better = self._examples(2.0, 0.0).as_tensors(device)
        worse = self._examples(0.0, 2.0).as_tensors(device)
        rows = torch.tensor([0], device=device)
        self.assertLess(
            float(conflict_loss(scorer, better, rows).detach()),
            float(conflict_loss(scorer, worse, rows).detach()),
        )

    def test_residual_update_cannot_change_candidate_topology(self) -> None:
        scorer = SparseEdgeScorer(_Transformer(), config=SparseEdgeConfig())
        video = _video()
        source, target = video.detections
        before = gate_sparse_candidates(source, target, config=scorer.config)
        examples = conflict_examples_from_video(
            scorer,
            video,
            {1: 101, 2: 102, 3: 103, 4: 104},
            {(101, 104), (102, 103)},
            {(1, 3), (2, 4)},
        )
        optimizer = torch.optim.SGD(scorer.calibrator.parameters(), lr=0.1)
        tensors = examples.as_tensors(scorer.calibrator.output.weight.device)
        rows = torch.arange(len(examples))
        optimizer.zero_grad(set_to_none=True)
        conflict_loss(scorer, tensors, rows).backward()
        optimizer.step()
        after = gate_sparse_candidates(source, target, config=scorer.config)
        np.testing.assert_array_equal(before.source_ids, after.source_ids)
        np.testing.assert_array_equal(before.target_ids, after.target_ids)
        np.testing.assert_array_equal(before.distance_um, after.distance_um)
        self.assertEqual(len(after), 4)

    def test_checkpoint_restore_preserves_conflict_residual_logits(self) -> None:
        transformer = _Transformer(seed=83)
        scorer = SparseEdgeScorer(transformer)
        video = _video()
        source, target = video.detections
        context = video.contexts[0]
        with torch.no_grad():
            scorer.calibrator.output.weight.fill_(0.02)
            scorer.calibrator.output.bias.fill_(-0.01)
        before = scorer.candidate_logits(context, source, target)[1].detach().clone()
        exposure = ExposureMetadata(
            fold=0,
            source_groups=("fit-a",),
            source_video_ids=("train:fit-a",),
            split_manifest_sha256="1" * 64,
            provenance_manifest_sha256="2" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = pathlib.Path(directory) / "edge.json"
            identity = save_edge_artifact(artifact, scorer, exposure)
            restored, restored_identity = load_edge_artifact(artifact, transformer)
            after = restored.candidate_logits(context, source, target)[1]
        self.assertEqual(restored_identity, identity)
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


class IsolationAndGraphPolicyTests(unittest.TestCase):
    def test_explicit_sources_must_be_parent_fit_sources(self) -> None:
        parent = (
            {"group_id": "fit-a", "stem": "a", "video_id": "train:a"},
            {"group_id": "fit-b", "stem": "b", "video_id": "train:b"},
            {"group_id": "fit-c", "stem": "c", "video_id": "train:c"},
        )
        self.assertEqual(
            [row["video_id"] for row in select_parent_sources(parent, ("a", "train:c"))],
            ["train:a", "train:c"],
        )
        with self.assertRaisesRegex(ConflictFitError, "not declared"):
            select_parent_sources(parent, ("a", "held"))

    def test_explicit_registry_sources_reject_test_or_unknown_videos(self) -> None:
        registry = {
            "records": [
                {"split": "train", "dataset": "a", "video_id": "train:a", "group_id": "fit-a"},
                {"split": "train", "dataset": "b", "video_id": "train:b", "group_id": "fit-b"},
                {"split": "test", "dataset": "x", "video_id": "test:x", "group_id": "test-x"},
            ]
        }
        self.assertEqual(
            [row["video_id"] for row in select_registry_sources(registry, ("a", "train:b"))],
            ["train:a", "train:b"],
        )
        with self.assertRaisesRegex(ConflictFitError, "not a declared competition-train"):
            select_registry_sources(registry, ("a", "test:x"))

    def test_division_disabled_replay_keeps_locked_sparse_solver(self) -> None:
        nodes = [
            GraphNode(1, 0, 0.0, 10.0, 10.0),
            GraphNode(2, 1, 0.0, 10.0, 11.0),
            GraphNode(3, 1, 0.0, 10.0, 12.0),
            GraphNode(4, 2, 0.0, 10.0, 13.0),
        ]
        result = calibrate_v45_graph(nodes, division_enabled=False)
        self.assertFalse(result.provenance["division_enabled"])
        self.assertFalse(result.provenance["full_frame_dense_matrix"])
        self.assertEqual(result.provenance["topology_mode"], "legacy")
        self.assertEqual(result.stats["division_added"], 0)
        self.assertEqual(result.stats["division_global_cap"], 0)
        outdegree: dict[int, int] = {}
        for edge in result.edges:
            outdegree[edge.source_id] = outdegree.get(edge.source_id, 0) + 1
        self.assertLessEqual(max(outdegree.values(), default=0), 1)


if __name__ == "__main__":
    unittest.main()
