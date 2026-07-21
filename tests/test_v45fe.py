from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("v45feval", ROOT / "scripts/v45feval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V45FastEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    @staticmethod
    def _split() -> dict:
        return {
            "artifact": "grouped_3fold_audit20_v1",
            "dataset_registry_sha256": "r" * 64,
            "audit": {"video_ids": ["train:audit"], "group_ids": ["audit"], "untouched": True},
            "folds": {
                str(index): {
                    "held_out_video_ids": [f"train:held{index}"],
                    "held_out_group_ids": [f"held{index}"],
                    "fit_group_ids": [f"fit{index}"],
                }
                for index in range(3)
            },
        }

    @staticmethod
    def _records() -> tuple:
        return tuple(
            SimpleNamespace(video_id=f"train:held{index}", dataset=f"held{index}", split="train", group_id=f"held{index}")
            for index in range(3)
        ) + (SimpleNamespace(video_id="train:audit", dataset="audit", split="train", group_id="audit"),)

    def test_selects_only_heldout_complete_videos(self) -> None:
        scope = self.mod.select_heldout(
            self._records(), self._split(), fold=0, registry_sha256="r" * 64, split_sha256="s" * 64
        )
        self.assertEqual([row.video_id for row in scope.records], ["train:held0"])
        self.assertEqual(scope.audit_ids, ("train:audit",))

    def test_rejects_audit_and_foreign_requested_datasets(self) -> None:
        with self.assertRaisesRegex(self.mod.V45FastEval, "held-out"):
            self.mod.select_heldout(
                self._records(), self._split(), fold=0, registry_sha256="r" * 64, split_sha256="s" * 64,
                requested=("audit",),
            )
        with self.assertRaisesRegex(self.mod.V45FastEval, "held-out"):
            self.mod.select_heldout(
                self._records(), self._split(), fold=0, registry_sha256="r" * 64, split_sha256="s" * 64,
                requested=("held1",),
            )

    def test_rejects_manifest_group_overlap(self) -> None:
        split = self._split()
        split["folds"]["0"]["fit_group_ids"] = ["held0"]
        with self.assertRaisesRegex(self.mod.V45FastEval, "group overlap"):
            self.mod.select_heldout(
                self._records(), split, fold=0, registry_sha256="r" * 64, split_sha256="s" * 64
            )

    def test_training_identity_is_bound_to_checkpoint_identity(self) -> None:
        train_id = {
            "fold": 0,
            "target_contract_sha256": "c" * 64,
            "selected_sources": [{"group_id": "fit", "stem": "source", "video_id": "train:source"}],
        }
        ident = SimpleNamespace(
            steps=4,
            division_enabled=False,
            target_contract_sha256="c" * 64,
            train_sha256=hashlib.sha256(self.mod._canonical(train_id)).hexdigest(),
        )
        result = self.mod.verify_training(
            {"status": "completed", "checkpoint_sha256": "a" * 64, "completed_steps": 4, "identity": train_id},
            "a" * 64,
            ident,
            fold=0,
        )
        self.assertEqual(result[0]["video_id"], "train:source")
        ident.train_sha256 = "b" * 64
        with self.assertRaisesRegex(self.mod.V45FastEval, "identity hash"):
            self.mod.verify_training(
                {"status": "completed", "checkpoint_sha256": "a" * 64, "completed_steps": 4, "identity": train_id},
                "a" * 64,
                ident,
                fold=0,
            )

    def test_no_base_v45_cache_call_is_present(self) -> None:
        source = (ROOT / "scripts/v45feval.py").read_text(encoding="ascii")
        self.assertNotIn("load_inference_cache(", source)
        self.assertNotIn("save_inference_cache(", source)
        self.assertIn("disabled-distinct-fast-execution-identity", source)

    def test_argument_rejects_limit_and_explicit_dataset_mix(self) -> None:
        with self.assertRaisesRegex(self.mod.V45FastEval, "mutually exclusive"):
            self.mod._args(["--input", "in.zip", "--out-dir", "out", "--device", "cpu", "--limit", "1", "--datasets", "x"])

    def test_association_mode_defaults_to_fallback_and_can_be_opted_in(self) -> None:
        fallback = self.mod._args(["--input", "in.zip", "--out-dir", "out", "--device", "cpu"])
        self.assertEqual(fallback.assoc, "fallback")
        with self.assertRaisesRegex(self.mod.V45FastEval, "requires a pinned edge artifact"):
            self.mod._args(["--input", "in.zip", "--out-dir", "out", "--device", "cpu", "--assoc", "sparse"])
        sparse = self.mod._args([
            "--input", "in.zip", "--out-dir", "out", "--device", "cpu", "--assoc", "sparse",
            "--edge-artifact", "edge.json", "--edge-artifact-sha256", "a" * 64,
        ])
        self.assertEqual(sparse.assoc, "sparse")
        self.assertEqual(sparse.division_mode, "artifact")
        inherited = self.mod._args([
            "--input", "in.zip", "--out-dir", "out", "--device", "cpu", "--assoc", "sparse",
            "--edge-artifact", "edge.json", "--edge-artifact-sha256", "a" * 64,
            "--division-mode", "inherited",
        ])
        self.assertEqual(inherited.division_mode, "inherited")
        a0 = self.mod._args(["--input", "in.zip", "--out-dir", "out", "--device", "cpu", "--assoc", "a0"])
        self.assertEqual(a0.assoc, "a0")
        frozen = self.mod._args(["--input", "in.zip", "--out-dir", "out", "--device", "cpu", "--assoc", "frozen"])
        self.assertEqual(frozen.assoc, "frozen")
        with self.assertRaisesRegex(self.mod.V45FastEval, "require sparse"):
            self.mod._args([
                "--input", "in.zip", "--out-dir", "out", "--device", "cpu",
                "--edge-artifact", "edge.json", "--edge-artifact-sha256", "a" * 64,
            ])
        with self.assertRaisesRegex(self.mod.V45FastEval, "fit-only sparse"):
            self.mod._args([
                "--input", "in.zip", "--out-dir", "out", "--device", "cpu",
                "--assoc", "frozen", "--topology", "scored_partial",
            ])
        with self.assertRaisesRegex(self.mod.V45FastEval, "require sparse"):
            self.mod._args([
                "--input", "in.zip", "--out-dir", "out", "--device", "cpu", "--assoc", "a0",
                "--edge-artifact", "edge.json", "--edge-artifact-sha256", "a" * 64,
            ])

    def test_edge_scope_requires_exact_parent_fit_exposure(self) -> None:
        exposure = self.mod.ExposureMetadata(
            fold=0,
            source_groups=("fit0",),
            source_video_ids=("train:source",),
            split_manifest_sha256="a" * 64,
            provenance_manifest_sha256="b" * 64,
        )
        identity = self.mod.EdgeArtifactIdentity(
            architecture_sha256="a" * 64,
            config_sha256="b" * 64,
            pretrained_weight_sha256="c" * 64,
            frozen_subset_sha256="d" * 64,
            residual_weight_sha256="e" * 64,
            auxiliary_checkpoint_sha256=None,
            auxiliary_state_sha256=None,
            exposure=exposure,
        )
        scope = self.mod.EvalScope(
            fold=0,
            records=(),
            audit_ids=("train:audit",),
            held_out_ids=("train:held",),
            group_by_id={"train:source": "fit0", "train:audit": "audit", "train:held": "held"},
            split_sha256="a" * 64,
            registry_sha256="c" * 64,
        )
        split = {
            "folds": {"0": {"fit_group_ids": ["fit0"]}},
            "audit": {"group_ids": ["audit"]},
        }
        sources = ({"video_id": "train:source", "group_id": "fit0"},)
        self.mod.verify_edge_artifact_scope(identity, sources, scope, split, provenance_sha256="b" * 64)
        bad = self.mod.EdgeArtifactIdentity(
            architecture_sha256="a" * 64,
            config_sha256="b" * 64,
            pretrained_weight_sha256="c" * 64,
            frozen_subset_sha256="d" * 64,
            residual_weight_sha256="e" * 64,
            auxiliary_checkpoint_sha256=None,
            auxiliary_state_sha256=None,
            exposure=self.mod.ExposureMetadata(
                fold=0,
                source_groups=("audit",),
                source_video_ids=("train:audit",),
                split_manifest_sha256="a" * 64,
                provenance_manifest_sha256="b" * 64,
            ),
        )
        with self.assertRaisesRegex(self.mod.V45FastEval, "differs|fit-only"):
            self.mod.verify_edge_artifact_scope(bad, sources, scope, split, provenance_sha256="b" * 64)

    def test_edge_scope_preserves_parent_video_group_pairs_before_sorting(self) -> None:
        exposure = self.mod.ExposureMetadata(
            fold=0,
            source_groups=("group-a", "group-z"),
            source_video_ids=("train:a", "train:b"),
            split_manifest_sha256="a" * 64,
            provenance_manifest_sha256="b" * 64,
        )
        identity = self.mod.EdgeArtifactIdentity(
            architecture_sha256="a" * 64,
            config_sha256="b" * 64,
            pretrained_weight_sha256="c" * 64,
            frozen_subset_sha256="d" * 64,
            residual_weight_sha256="e" * 64,
            auxiliary_checkpoint_sha256=None,
            auxiliary_state_sha256=None,
            exposure=exposure,
        )
        scope = self.mod.EvalScope(
            fold=0,
            records=(),
            audit_ids=("train:audit",),
            held_out_ids=("train:held",),
            group_by_id={
                "train:a": "group-z",
                "train:b": "group-a",
                "train:audit": "audit",
                "train:held": "held",
            },
            split_sha256="a" * 64,
            registry_sha256="c" * 64,
        )
        split = {
            "folds": {"0": {"fit_group_ids": ["group-a", "group-z"]}},
            "audit": {"group_ids": ["audit"]},
        }
        sources = (
            {"video_id": "train:a", "group_id": "group-z"},
            {"video_id": "train:b", "group_id": "group-a"},
        )
        self.mod.verify_edge_artifact_scope(identity, sources, scope, split, provenance_sha256="b" * 64)

    def test_edge_scope_allows_a_distinct_registry_bound_fit_subset(self) -> None:
        exposure = self.mod.ExposureMetadata(
            fold=0,
            source_groups=("residual-fit",),
            source_video_ids=("train:residual",),
            split_manifest_sha256="a" * 64,
            provenance_manifest_sha256="b" * 64,
        )
        identity = self.mod.EdgeArtifactIdentity(
            architecture_sha256="a" * 64,
            config_sha256="b" * 64,
            pretrained_weight_sha256="c" * 64,
            frozen_subset_sha256="d" * 64,
            residual_weight_sha256="e" * 64,
            auxiliary_checkpoint_sha256=None,
            auxiliary_state_sha256=None,
            exposure=exposure,
        )
        scope = self.mod.EvalScope(
            fold=0,
            records=(),
            audit_ids=("train:audit",),
            held_out_ids=("train:held",),
            group_by_id={
                "train:parent": "parent-fit",
                "train:residual": "residual-fit",
                "train:audit": "audit",
                "train:held": "held",
            },
            split_sha256="a" * 64,
            registry_sha256="c" * 64,
        )
        split = {
            "folds": {"0": {"fit_group_ids": ["parent-fit", "residual-fit"]}},
            "audit": {"group_ids": ["audit"]},
        }
        sources = ({"video_id": "train:parent", "group_id": "parent-fit"},)
        self.mod.verify_edge_artifact_scope(
            identity, sources, scope, split, provenance_sha256="b" * 64
        )

    def test_parent_manifest_binding_rejects_mismatch(self) -> None:
        metrics = {"identity": {"registry_sha256": "r" * 64, "split_sha256": "s" * 64, "provenance_sha256": "p" * 64}}
        self.assertEqual(
            self.mod.verify_parent_manifests(metrics, registry_sha256="r" * 64, split_sha256="s" * 64, provenance_sha256="p" * 64),
            metrics["identity"],
        )
        with self.assertRaisesRegex(self.mod.V45FastEval, "does not bind"):
            self.mod.verify_parent_manifests(metrics, registry_sha256="r" * 64, split_sha256="q" * 64, provenance_sha256="p" * 64)

    def test_sparse_association_uses_context_frame_ids_not_detection_position(self) -> None:
        scorer = object.__new__(self.mod.SparseEdgeScorer)
        object.__setattr__(scorer, "config", SimpleNamespace(neighbor_cap=8))
        video = SimpleNamespace(
            detections=(
                SimpleNamespace(frame_index=0),
                SimpleNamespace(frame_index=1),
                SimpleNamespace(frame_index=2),
            ),
            contexts=(SimpleNamespace(source_frame=0, target_frame=1),),
        )
        pair = SimpleNamespace(
            stats=SimpleNamespace(candidate_count=1, peak_pair_chunk=1),
            no_link_source_ids=(1,),
            no_link_probabilities=(0.5,),
        )
        observed = []

        def score(context, source, target):
            observed.append((context.source_frame, source.frame_index, target.frame_index))
            return pair

        with (
            patch.object(self.mod.SparseEdgeScorer, "score_pair", side_effect=score),
            patch.object(self.mod, "learned_edge_probability_map", return_value={(1, 2): 0.5}),
        ):
            probabilities, no_link, stats = self.mod.sparse_probs(video, scorer)
        self.assertEqual(observed, [(0, 0, 1)])
        self.assertEqual(probabilities, {(1, 2): 0.5})
        self.assertEqual(no_link, {1: 0.5})
        self.assertEqual(stats["candidate_count"], 1)

    def test_fit_artifact_is_frozen_before_evaluation(self) -> None:
        import torch

        class StubScorer(self.mod.SparseEdgeScorer):
            def __init__(self) -> None:
                torch.nn.Module.__init__(self)
                self.fitted = torch.nn.Parameter(torch.ones(7))

            def train(self, mode: bool = True):
                torch.nn.Module.train(self, mode)
                return self

        scorer = StubScorer()
        self.assertEqual(scorer.trainable_parameter_count, 7)
        fitted = self.mod.freeze_fitted_scorer_for_evaluation(scorer)
        self.assertEqual(fitted, 7)
        self.assertEqual(scorer.trainable_parameter_count, 0)
        self.assertFalse(scorer.training)

    def test_a0_association_gates_original_candidates_with_production_kdtree_bounds(self) -> None:
        class Candidates:
            def __len__(self) -> int:
                return 1

        class Pair:
            source_ids = (1,)
            target_ids = (2,)
            probabilities = (0.75,)
            stats = object()

            def __len__(self) -> int:
                return 1

        scorer = object.__new__(self.mod.A0CheckpointScorer)
        object.__setattr__(scorer, "config", SimpleNamespace(neighbor_cap=8))
        video = SimpleNamespace(
            detections=(SimpleNamespace(frame_index=0), SimpleNamespace(frame_index=1)),
            contexts=(SimpleNamespace(source_frame=0, target_frame=1),),
        )
        stats = {
            "candidate_count": 1,
            "component_count": 1,
            "attention_component_count": 1,
            "fallback_component_count": 0,
            "node_budget_fallback_component_count": 0,
            "attention_budget_fallback_component_count": 0,
            "unsupported_fallback_component_count": 0,
            "attention_candidate_count": 1,
            "fallback_candidate_count": 0,
            "peak_attention_cells": 1,
            "peak_pair_chunk": 1,
        }
        observed = []

        def gate(source, target, *, config):
            observed.append((source.frame_index, target.frame_index, config))
            return Candidates()

        with (
            patch.object(self.mod, "gate_sparse_candidates", side_effect=gate),
            patch.object(self.mod, "A0PairProbabilities", Pair),
            patch.object(self.mod.A0CheckpointScorer, "score_pair", return_value=Pair()),
            patch.object(self.mod, "asdict", return_value=stats),
        ):
            probabilities, audit = self.mod.a0_probs(video, scorer)
        self.assertEqual(probabilities, {(1, 2): 0.75})
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][2].radius_um, self.mod.run_bh.A0_GATE_RADIUS_UM)
        self.assertEqual(observed[0][2].neighbor_cap, self.mod.run_bh.A0_GATE_NEIGHBOR_CAP)
        self.assertEqual(audit["candidate_gate"]["algorithm"], "physical-cKDTree")
        self.assertEqual(audit["attention_component_count"], 1)


if __name__ == "__main__":
    unittest.main()
