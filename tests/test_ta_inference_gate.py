"""Tests for the bounded TemporalAdapterModel held-out evidence gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external/kaggle-cell-tracking-competition/src"))

from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.ta_inference import (  # noqa: E402
    GateError,
    attest_gate_pair,
    attest_hybrid_checkpoint,
    iter_sliding_frame_outputs,
    run_deterministic_smoke,
    select_held_out_video,
    sparse_alpha_association,
    write_fragment,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class TemporalAdapterInferenceGateTest(unittest.TestCase):
    def _gate_fixture(self, root: Path) -> dict[str, Path]:
        split = {
            "artifact": "fixture-split",
            "audit": {"video_ids": ["train:audit"], "video_count": 1},
            "folds": {
                "0": {
                    "fit_group_ids": ["g-fit", "g-pre"],
                    "fit_video_ids": ["train:fit", "train:pre"],
                    "held_out_video_ids": ["train:held"],
                    "held_out_video_count": 1,
                },
                "1": {"fit_group_ids": [], "fit_video_ids": [], "held_out_video_ids": [], "held_out_video_count": 0},
                "2": {"fit_group_ids": [], "fit_video_ids": [], "held_out_video_ids": [], "held_out_video_count": 0},
            },
        }
        registry = {
            "artifact": "fixture-registry",
            "records": [
                {"split": "train", "dataset": "fit", "video_id": "train:fit", "group_id": "g-fit",
                 "exact_identity_sha256": "1" * 64, "compact_metadata": {"image_shape": [3, 4, 8, 8]}},
                {"split": "train", "dataset": "pre", "video_id": "train:pre", "group_id": "g-pre",
                 "exact_identity_sha256": "2" * 64, "compact_metadata": {"image_shape": [3, 4, 8, 8]}},
                {"split": "train", "dataset": "held", "video_id": "train:held", "group_id": "g-held",
                 "exact_identity_sha256": "3" * 64, "compact_metadata": {"image_shape": [3, 4, 8, 8]}},
                {"split": "train", "dataset": "audit", "video_id": "train:audit", "group_id": "g-audit",
                 "exact_identity_sha256": "4" * 64, "compact_metadata": {"image_shape": [3, 4, 8, 8]}},
            ],
        }
        provenance = {
            "artifact": "fixture-provenance",
            "folds": {"0": {"fit_sources": {"calibration": ["g-fit", "g-pre"]}}},
        }
        paths = {name: root / f"{name}.json" for name in ("split", "registry", "provenance")}
        _write_json(paths["split"], split)
        _write_json(paths["registry"], registry)
        _write_json(paths["provenance"], provenance)

        split_hash = _sha256(paths["split"])
        registry_hash = _sha256(paths["registry"])
        provenance_hash = _sha256(paths["provenance"])
        common = {
            "fold": 0,
            "scenario": "universal",
            "split_sha256": split_hash,
            "registry_sha256": registry_hash,
            "provenance_sha256": provenance_hash,
            "selected_sources": [{"group_id": "g-fit", "stem": "fit", "video_id": "train:fit"}],
            "seed": 17,
            "fraction": 0.1,
            "lr": 1e-5,
            "unfreeze": 0,
            "detector_policy": "frozen_aux",
            "center_weight": 0.0,
            "teacher_weight": 0.0,
        }
        pretrain_checkpoint = root / "pretrain.pt"
        pretrain_checkpoint.write_bytes(b"fold-safe temporal pretrain")
        pretrain_hash = _sha256(pretrain_checkpoint)
        pretrain_metrics = root / "pretrain.json"
        _write_json(
            pretrain_metrics,
            {
                "checkpoint": {"path": "checkpoint.pt", "sha256": pretrain_hash},
                "identity": {
                    "fold": 0,
                    "objective": "alpha-sparse-migration-ranking.v1",
                    "split_sha256": split_hash,
                    "registry_sha256": registry_hash,
                    "provenance_sha256": provenance_hash,
                    "selected_sources": [
                        {"group_id": "g-pre", "stem": "pre", "video_id": "train:pre"}
                    ],
                    "input_policy": {
                        "competition_train_images_only": True,
                        "alpha_image_derived_pseudo_links_used": True,
                        "labels_used": False,
                        "tracks_used": False,
                        "held_out_used": False,
                        "audit_used": False,
                        "test_used": False,
                    },
                },
            },
        )

        for label, temporal_hash in (("baseline", None), ("candidate", pretrain_hash)):
            checkpoint = root / f"{label}.pt"
            checkpoint.write_bytes(f"{label} train output".encode("ascii"))
            metrics = root / f"{label}.json"
            identity = {**common, "temporal_pretrain_sha256": temporal_hash}
            _write_json(
                metrics,
                {
                    "identity": identity,
                    "checkpoint_sha256": _sha256(checkpoint),
                    "completed_steps": 16,
                    "completed_windows": 16,
                    "status": "completed",
                    "execution": {"epochs": 1, "max_steps": 16},
                    "source_groups": ["g-fit"],
                    "source_video_ids": ["train:fit"],
                    "source_records": identity["selected_sources"],
                },
            )
            paths[f"{label}_checkpoint"] = checkpoint
            paths[f"{label}_metrics"] = metrics
        paths["pretrain_checkpoint"] = pretrain_checkpoint
        paths["pretrain_metrics"] = pretrain_metrics
        return paths

    def test_attested_pair_is_equal_budget_fit_only_and_pretrain_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._gate_fixture(Path(temporary))
            gate = attest_gate_pair(
                baseline_checkpoint=paths["baseline_checkpoint"],
                baseline_metrics=paths["baseline_metrics"],
                candidate_checkpoint=paths["candidate_checkpoint"],
                candidate_metrics=paths["candidate_metrics"],
                pretrain_checkpoint=paths["pretrain_checkpoint"],
                pretrain_metrics=paths["pretrain_metrics"],
                split_manifest=paths["split"],
                provenance_manifest=paths["provenance"],
                registry_manifest=paths["registry"],
                fold=0,
                held_out_dataset="held",
            )
            self.assertEqual(gate.held_out_video_id, "train:held")
            self.assertEqual(gate.baseline.completed_steps, gate.candidate.completed_steps)
            self.assertIsNone(gate.baseline.temporal_pretrain_sha256)
            self.assertEqual(gate.candidate.temporal_pretrain_sha256, _sha256(paths["pretrain_checkpoint"]))
            self.assertTrue(gate.audit_disjoint)
            self.assertTrue(gate.test_used is False)

    def test_attestation_rejects_cross_fold_source_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._gate_fixture(Path(temporary))
            metrics = json.loads(paths["candidate_metrics"].read_text(encoding="utf-8"))
            metrics["identity"]["selected_sources"][0]["video_id"] = "train:held"
            metrics["source_video_ids"] = ["train:held"]
            _write_json(paths["candidate_metrics"], metrics)
            with self.assertRaisesRegex(GateError, "fit source"):
                attest_gate_pair(
                    baseline_checkpoint=paths["baseline_checkpoint"],
                    baseline_metrics=paths["baseline_metrics"],
                    candidate_checkpoint=paths["candidate_checkpoint"],
                    candidate_metrics=paths["candidate_metrics"],
                    pretrain_checkpoint=paths["pretrain_checkpoint"],
                    pretrain_metrics=paths["pretrain_metrics"],
                    split_manifest=paths["split"],
                    provenance_manifest=paths["provenance"],
                    registry_manifest=paths["registry"],
                    fold=0,
                    held_out_dataset="held",
                )

    def test_hybrid_attestation_is_migration_bound_and_has_no_target_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._gate_fixture(Path(temporary))
            gate = attest_hybrid_checkpoint(
                candidate_checkpoint=paths["candidate_checkpoint"],
                candidate_metrics=paths["candidate_metrics"],
                pretrain_checkpoint=paths["pretrain_checkpoint"],
                pretrain_metrics=paths["pretrain_metrics"],
                split_manifest=paths["split"],
                provenance_manifest=paths["provenance"],
                registry_manifest=paths["registry"],
                fold=0,
                held_out_dataset="held",
            )
            payload = gate.json()
            self.assertEqual(gate.held_out_video_id, "train:held")
            self.assertEqual(gate.candidate.temporal_pretrain_sha256, gate.pretrain.checkpoint_sha256)
            self.assertNotIn("train:held", gate.candidate.source_video_ids)
            self.assertNotIn("train:held", gate.pretrain.source_video_ids)
            self.assertEqual(payload["reranker_policy"]["learned_or_calibrated_rerank_weights"], 0)
            self.assertFalse(payload["reranker_policy"]["held_out_target_calibration_used"])

    def test_hybrid_attestation_rejects_heldout_candidate_source_before_image_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._gate_fixture(Path(temporary))
            metrics = json.loads(paths["candidate_metrics"].read_text(encoding="utf-8"))
            metrics["identity"]["selected_sources"][0] = {
                "group_id": "g-held",
                "stem": "held",
                "video_id": "train:held",
            }
            metrics["source_groups"] = ["g-held"]
            metrics["source_video_ids"] = ["train:held"]
            _write_json(paths["candidate_metrics"], metrics)
            with self.assertRaisesRegex(GateError, "fit source"):
                attest_hybrid_checkpoint(
                    candidate_checkpoint=paths["candidate_checkpoint"],
                    candidate_metrics=paths["candidate_metrics"],
                    pretrain_checkpoint=paths["pretrain_checkpoint"],
                    pretrain_metrics=paths["pretrain_metrics"],
                    split_manifest=paths["split"],
                    provenance_manifest=paths["provenance"],
                    registry_manifest=paths["registry"],
                    fold=0,
                    held_out_dataset="held",
                )

    def test_select_heldout_rejects_audit_and_test(self) -> None:
        split = {
            "audit": {"video_ids": ["train:audit"]},
            "folds": {"0": {"held_out_video_ids": ["train:held"]}},
        }
        self.assertEqual(select_held_out_video(split, 0, "held"), "train:held")
        with self.assertRaisesRegex(GateError, "held-out"):
            select_held_out_video(split, 0, "audit")
        with self.assertRaisesRegex(GateError, "train"):
            select_held_out_video(split, 0, "test:held")

    def test_sliding_outputs_average_centers_but_keep_child_backward_owner(self) -> None:
        first = np.zeros((2, 6, 1, 1, 1), dtype=np.float32)
        second = np.zeros_like(first)
        first[0, 0] = 0.0
        first[1, 0] = 2.0
        first[1, 1:5] = 10.0
        second[0, 0] = 4.0
        second[0, 1:5] = 99.0
        second[1, 0] = 6.0
        second[1, 1:5] = 20.0
        outputs = list(iter_sliding_frame_outputs([first, second]))
        self.assertEqual([float(item.center_logit[0, 0, 0]) for item in outputs], [0.0, 3.0, 6.0])
        self.assertFalse(outputs[0].association_valid)
        self.assertTrue(outputs[1].association_valid)
        np.testing.assert_array_equal(outputs[1].association[:, 0, 0, 0], np.full(4, 10.0))
        np.testing.assert_array_equal(outputs[2].association[:, 0, 0, 0], np.full(4, 20.0))

    def test_sparse_alpha_association_preserves_bounded_geometric_fallback(self) -> None:
        raw = [
            {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0, "score": 0.9,
             "backward_um": [0.0, 0.0, 0.0], "log_variance": 0.0},
            {"node_id": 2, "t": 0, "z": 0, "y": 0, "x": 5, "score": 0.8,
             "backward_um": [0.0, 0.0, 0.0], "log_variance": 0.0},
            {"node_id": 3, "t": 1, "z": 0, "y": 0, "x": 5, "score": 0.95,
             "backward_um": [0.0, 0.0, -5.0], "log_variance": 0.0},
            {"node_id": 4, "t": 1, "z": 0, "y": 0, "x": 5, "score": 0.7,
             "backward_um": [0.0, 0.0, 100.0], "log_variance": 0.0},
        ]
        _nodes, edges, report = sparse_alpha_association(
            "held", raw, (1.0, 1.0, 1.0), edge_radius_um=6.0, neighbor_cap=1, edge_threshold=0.2
        )
        self.assertIn({"source_id": 1, "target_id": 3}, edges)
        self.assertIn({"source_id": 2, "target_id": 4}, edges)
        self.assertGreater(report["fallback_candidate_edges"], 0)
        self.assertLessEqual(report["peak_candidates_per_target"], 1)
        self.assertFalse(report["dense_nxn"])

    def test_fragment_roundtrip_matches_frozen_scorer_input_schema(self) -> None:
        nodes = [
            {"dataset": "held", "node_id": 1, "t": 0, "z": 0, "y": 1, "x": 2},
            {"dataset": "held", "node_id": 2, "t": 1, "z": 0, "y": 1, "x": 3},
        ]
        edges = [{"source_id": 1, "target_id": 2}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train__held.csv"
            digest = write_fragment(path, "held", nodes, edges)
            self.assertEqual(digest, _sha256(path))
            graph = read_submission(path)["held"]
            self.assertEqual(len(graph.nodes), 2)
            self.assertEqual([(edge.source_id, edge.target_id) for edge in graph.edges], [(1, 2)])

    def test_tiny_smoke_is_deterministic_and_uses_official_parity(self) -> None:
        first = run_deterministic_smoke()
        second = run_deterministic_smoke()
        self.assertEqual(first, second)
        self.assertTrue(first["official_parity_pass"])
        self.assertTrue(first["association"]["fallback_preserved"])
        self.assertEqual(first["scope"], "tiny_synthetic_no_competition_data")


if __name__ == "__main__":
    unittest.main()
