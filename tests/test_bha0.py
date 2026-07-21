from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from celltrack_alpha import bh_edge, bh_model
from tests import test_bh as base


BH = base.BH


def _wide_record() -> object:
    return BH.RegistryRecord("test", "wide", "test:wide", "1" * 64, (3, 1, 1, 500))


def _wide_inference() -> object:
    frames = []
    x_values = np.arange(16, dtype=np.int64) * 30
    coords = np.column_stack(
        (np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64), x_values)
    )
    for frame_index in range(3):
        frames.append(
            bh_model.FrameDet(
                frame_index=frame_index,
                node_ids=np.arange(frame_index * 16, (frame_index + 1) * 16, dtype=np.int64),
                coords_zyx=coords,
                scores=np.full(16, 0.99, dtype=np.float32),
                grid_coords_zyx=coords,
                grid_shape_zyx=(1, 1, 500),
            )
        )
    contexts = tuple(
        bh_model.PairContext(
            source_frame=source.frame_index,
            target_frame=target.frame_index,
            source_ids=source.node_ids,
            target_ids=target.node_ids,
            source_feat=np.zeros((16, 32), dtype=np.float32),
            target_feat=np.zeros((16, 32), dtype=np.float32),
            grid_shape_zyx=(1, 1, 500),
        )
        for source, target in zip(frames, frames[1:])
    )
    return bh_model.VideoInference(tuple(frames), contexts)


def _a0_stats(candidates: object) -> object:
    count = len(candidates)
    components = count
    return bh_edge.A0AssociationStats(
        source_count=candidates.stats.source_count,
        target_count=candidates.stats.target_count,
        candidate_count=count,
        component_count=components,
        attention_component_count=components,
        fallback_component_count=0,
        node_budget_fallback_component_count=0,
        attention_budget_fallback_component_count=0,
        unsupported_fallback_component_count=0,
        attention_candidate_count=count,
        fallback_candidate_count=0,
        attention_component_fraction=1.0 if components else 0.0,
        fallback_component_fraction=0.0,
        attention_candidate_fraction=1.0 if count else 0.0,
        fallback_candidate_fraction=0.0,
        max_component_source_nodes=1 if count else 0,
        max_component_target_nodes=1 if count else 0,
        max_component_nodes=2 if count else 0,
        max_component_candidates=1 if count else 0,
        max_component_attention_cells=1 if count else 0,
        batch_count=1 if count else 0,
        max_batch_components=components,
        max_padded_source_nodes=1 if count else 0,
        max_padded_target_nodes=1 if count else 0,
        peak_padding_nodes=0,
        peak_attention_cells=components,
        neighbor_cap=8,
        max_component_nodes_budget=64,
        attention_cell_budget=65_536,
        pair_chunk_size=256,
        peak_pair_chunk=count,
    )


class FakeA0:
    def __init__(self, weight_sha256: str) -> None:
        self.pretrained_weight_sha256 = weight_sha256
        self.architecture = {
            "full_frame_matrix": False,
            "pair_scope": "original-candidate-coo-only",
        }
        self.architecture_sha256 = "d" * 64
        self.frozen_subset_sha256 = "e" * 64
        self.config = bh_edge.A0ScorerConfig()
        self.calls: list[tuple[int, int]] = []

    def score_pair(self, context, source, target, *, candidates):
        self.calls.append(context.frame_pair)
        if candidates.stats.radius_um != 10.0 or candidates.stats.neighbor_cap != 8:
            raise AssertionError("runner did not use the production physical gate")
        if len(candidates) >= len(source.node_ids) * len(target.node_ids):
            raise AssertionError("runner supplied a full-frame NxN candidate set")
        return bh_edge.A0PairProbabilities(
            source_frame=source.frame_index,
            target_frame=target.frame_index,
            source_ids=candidates.source_ids,
            target_ids=candidates.target_ids,
            probabilities=np.ones(len(candidates), dtype=np.float64),
            stats=_a0_stats(candidates),
        )


class A0RunnerTests(unittest.TestCase):
    def test_opt_in_scores_every_cached_context_and_replay_never_infers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            base._make_zarr(input_root / "test/wide.zarr", shape=(3, 1, 1, 500))
            job = BH.discover_jobs(input_root, [_wide_record()])[0]
            identity = bh_model.CacheIdentity("a" * 64, "b" * 64, "c" * 64)
            scorer = FakeA0(identity.weight_sha256)
            infer_calls = 0

            def infer(_model, _frames, **_kwargs):
                nonlocal infer_calls
                infer_calls += 1
                return _wide_inference()

            first_seen: dict[str, object] = {}
            first = BH.process_video(
                job,
                model=object(),
                cache_identity=identity,
                output_dir=root / "out1",
                cache_dir=root / "cache",
                graph_variant=BH.A0_GRAPH_VARIANT,
                expected_frames=3,
                infer_video_fn=infer,
                graph_module=base._fake_graph_module(first_seen),
                a0_scorer=scorer,
                edge_module=bh_edge,
            )
            self.assertEqual(infer_calls, 1)
            self.assertEqual(scorer.calls, [(0, 1), (1, 2)])
            self.assertEqual(len(first_seen["learned"]), 32)
            self.assertFalse(first["cache_reused"])
            self.assertEqual(first["a0_audit"]["pair_count"], 2)
            self.assertEqual(first["a0_audit"]["gate"]["candidate_count"], 32)
            self.assertFalse(first["a0_audit"]["gate"]["full_frame_dense_matrix"])

            feature = BH.strict_json_load(Path(first["feature_summary"]))
            status = BH.strict_json_load(root / "out1/status" / f"{job.slug}.json")
            self.assertEqual(feature["a0_audit"], first["a0_audit"])
            self.assertEqual(status["a0_audit"], first["a0_audit"])
            self.assertEqual(
                status["run_contract"]["graph_config"]["audit_schema"],
                BH.A0_AUDIT_SCHEMA,
            )

            def inference_tripwire(*_args, **_kwargs):
                raise AssertionError("cache replay attempted model inference")

            replay_seen: dict[str, object] = {}
            second = BH.process_video(
                job,
                model=object(),
                cache_identity=identity,
                output_dir=root / "out2",
                cache_dir=root / "cache",
                graph_variant=BH.A0_GRAPH_VARIANT,
                expected_frames=3,
                infer_video_fn=inference_tripwire,
                graph_module=base._fake_graph_module(replay_seen),
                a0_scorer=scorer,
                edge_module=bh_edge,
            )
            self.assertTrue(second["cache_reused"])
            self.assertEqual(infer_calls, 1)
            self.assertEqual(scorer.calls, [(0, 1), (1, 2), (0, 1), (1, 2)])
            self.assertEqual(first_seen["learned"], replay_seen["learned"])

    def test_context_identity_probability_and_dense_provenance_fail_closed(self) -> None:
        inference = _wide_inference()
        identity = bh_model.CacheIdentity("a" * 64, "b" * 64, "c" * 64)
        cache = SimpleNamespace(
            detections=tuple(inference.detections),
            contexts=tuple(inference.contexts[:-1]),
        )
        with self.assertRaisesRegex(BH.RunnerError, "context gap"):
            BH.score_a0_cache(cache, FakeA0(identity.weight_sha256), identity, edge_module=bh_edge)

        complete = SimpleNamespace(
            detections=tuple(inference.detections),
            contexts=tuple(inference.contexts),
        )
        with self.assertRaisesRegex(BH.RunnerError, "checkpoint identity"):
            BH.score_a0_cache(complete, FakeA0("f" * 64), identity, edge_module=bh_edge)

        dense = FakeA0(identity.weight_sha256)
        dense.architecture["full_frame_matrix"] = True
        with self.assertRaisesRegex(BH.RunnerError, "sparse provenance"):
            BH.score_a0_cache(complete, dense, identity, edge_module=bh_edge)

        class BadProbability(FakeA0):
            value = np.nan

            def score_pair(self, context, source, target, *, candidates):
                result = super().score_pair(
                    context, source, target, candidates=candidates
                )
                object.__setattr__(
                    result,
                    "probabilities",
                    np.full(len(candidates), self.value, dtype=np.float64),
                )
                return result

        with self.assertRaisesRegex(BH.RunnerError, "nonfinite"):
            BH.score_a0_cache(
                complete,
                BadProbability(identity.weight_sha256),
                identity,
                edge_module=bh_edge,
            )

        outside = BadProbability(identity.weight_sha256)
        outside.value = 1.01
        with self.assertRaisesRegex(BH.RunnerError, r"outside \[0, 1\]"):
            BH.score_a0_cache(complete, outside, identity, edge_module=bh_edge)

        def frame(frame_index, ids, xs):
            coords = np.column_stack(
                (
                    np.zeros(len(ids), dtype=np.int64),
                    np.zeros(len(ids), dtype=np.int64),
                    np.asarray(xs, dtype=np.int64),
                )
            )
            return bh_model.FrameDet(
                frame_index=frame_index,
                node_ids=np.asarray(ids, dtype=np.int64),
                coords_zyx=coords,
                scores=np.full(len(ids), 0.99, dtype=np.float32),
                grid_coords_zyx=coords,
                grid_shape_zyx=(1, 1, 100),
            )

        duplicate_frames = (
            frame(0, [1], [0]),
            frame(1, [2, 1], [0, 30]),
            frame(2, [2], [30]),
        )
        duplicate_contexts = tuple(
            bh_model.PairContext(
                source_frame=source.frame_index,
                target_frame=target.frame_index,
                source_ids=source.node_ids,
                target_ids=target.node_ids,
                source_feat=np.zeros((len(source.node_ids), 32), dtype=np.float32),
                target_feat=np.zeros((len(target.node_ids), 32), dtype=np.float32),
                grid_shape_zyx=(1, 1, 100),
            )
            for source, target in zip(duplicate_frames, duplicate_frames[1:])
        )
        with self.assertRaisesRegex(BH.RunnerError, "duplicate A0 probability pair"):
            BH.score_a0_cache(
                SimpleNamespace(
                    detections=duplicate_frames,
                    contexts=duplicate_contexts,
                ),
                FakeA0(identity.weight_sha256),
                identity,
                edge_module=bh_edge,
            )

    def test_worker_constructs_exactly_one_a0_and_zero_for_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = SimpleNamespace(
                config_sha256="a" * 64,
                source_sha256="b" * 64,
                weight_sha256="c" * 64,
                pair_batch_size=1,
                view_batch_size=1,
            )
            bundle = SimpleNamespace(hashes=SimpleNamespace(weight_sha256="c" * 64))
            constructions = 0

            class Factory:
                @classmethod
                def from_loaded_model(cls, _model, _bundle):
                    nonlocal constructions
                    constructions += 1
                    return FakeA0("c" * 64)

            edge_module = SimpleNamespace(A0CheckpointScorer=Factory)

            class CacheIdentity:
                @classmethod
                def from_artifact(cls, *_args, **_kwargs):
                    return identity

            model_module = SimpleNamespace(
                load_v45_model=lambda *_args, **_kwargs: (object(), bundle),
                CacheIdentity=CacheIdentity,
            )
            route_module = SimpleNamespace()
            edge_imports = 0

            def import_module(name):
                nonlocal edge_imports
                if name == "celltrack_alpha.bh_model":
                    return model_module
                if name == "celltrack_alpha.bh_route":
                    return route_module
                if name == "celltrack_alpha.bh_edge":
                    edge_imports += 1
                    return edge_module
                raise AssertionError(f"unexpected import {name}")

            def run_worker(variant: str, name: str) -> dict[str, object]:
                aggregate = root / f"{name}.json"
                payload = {
                    "rank": 0,
                    "device": "cuda:0",
                    "jobs": [],
                    "output_dir": str(root / "out"),
                    "cache_dir": str(root / "cache"),
                    "support_root": str(root / "support"),
                    "graph_variant": variant,
                    "pair_batch_size": 1,
                    "view_batch_size": 1,
                    "accelerated_gate": None,
                    "router_json": None,
                    "router_sha256": None,
                    "cache_identity": BH._identity_dict(identity),
                    "deadline_epoch": 10**20,
                    "aggregate_status": str(aggregate),
                }
                job_path = root / f"{name}.job.json"
                job_path.write_text(json.dumps(payload), encoding="utf-8")
                with mock.patch.object(BH.importlib, "import_module", side_effect=import_module):
                    BH._worker_entry(str(job_path))
                return BH.strict_json_load(aggregate)

            a0 = run_worker(BH.A0_GRAPH_VARIANT, "a0")
            self.assertEqual(constructions, 1)
            self.assertEqual(edge_imports, 1)
            self.assertEqual(a0["a0_audit"]["worker"]["scorer_instances"], 1)

            motion = run_worker(BH.LOCKED_GRAPH_VARIANT, "motion")
            self.assertEqual(constructions, 1)
            self.assertEqual(edge_imports, 1)
            self.assertNotIn("a0_audit", motion)

    def test_motion_contract_and_outputs_remain_a0_free(self) -> None:
        parser = BH.build_parser()
        self.assertEqual(parser.parse_args(["--preflight-only"]).graph_variant, BH.LOCKED_GRAPH_VARIANT)
        contract = BH.GRAPH_VARIANTS[BH.LOCKED_GRAPH_VARIANT]
        self.assertEqual(
            contract,
            {
                "learned_edge_probabilities": None,
                "image_refinement": "lazy-p20-intensity-centroid",
                "config": "V45_GRAPH_CONFIG",
                "locked": True,
            },
        )
        self.assertIsNone(
            BH.build_a0_scorer(
                BH.LOCKED_GRAPH_VARIANT,
                object(),
                object(),
                object(),
                edge_module=SimpleNamespace(
                    A0CheckpointScorer=SimpleNamespace(
                        from_loaded_model=lambda *_args: (_ for _ in ()).throw(
                            AssertionError("motion constructed A0")
                        )
                    )
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
