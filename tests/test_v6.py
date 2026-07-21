"""Fold isolation and sparse edge-only coverage for local V6 scoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("score_v6", ROOT / "scripts/score_v6.py")
sv = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sv
SPEC.loader.exec_module(sv)


class V6ScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = sv._exact_json(sv.DEFAULT_MET, sv.MET_SHA256)
        cls.split = sv._exact_json(sv.DEFAULT_SPLIT, sv.SPLIT_SHA256)

    def test_01_cv_scope_accepts_only_fold_heldout(self) -> None:
        fold, datasets, proof = sv.fold_scope(self.split, self.metrics, "cv", ("6bba_0c7fa718",))
        self.assertEqual((fold, datasets), (0, ("6bba_0c7fa718",)))
        self.assertTrue(proof["checkpoint_sources_in_fit"])
        self.assertTrue(proof["audit_20_disjoint"])

    def test_02_audit_20_rejects_before_archive_read(self) -> None:
        with self.assertRaises(sv.V6Error):
            sv.fold_scope(self.split, self.metrics, "cv", ("6bba_3c5691b6",))

    def test_03_fit_and_cv_cannot_cross(self) -> None:
        with self.assertRaises(sv.V6Error):
            sv.fold_scope(self.split, self.metrics, "fit", ("6bba_0c7fa718",))
        fold, datasets, _ = sv.fold_scope(self.split, self.metrics, "fit", ("6bba_d3da753b",))
        self.assertEqual((fold, datasets), (0, ("6bba_d3da753b",)))

    def test_04_sparse_motion_link_is_edge_only(self) -> None:
        raw = [
            {"node_id": 1, "t": 0, "z": 2, "y": 8, "x": 8, "score": 0.9,
             "backward_um": [0.0, 0.0, 0.0], "log_variance": 0.0},
            {"node_id": 2, "t": 1, "z": 2, "y": 8, "x": 12, "score": 0.8,
             "backward_um": [0.0, 0.0, -1.625], "log_variance": 0.0},
        ]
        nodes, edges, graph = sv._link_graph(
            "6bba_0c7fa718", raw, (1.625, 0.40625, 0.40625), 8.0, 4, 0.2
        )
        self.assertEqual(len(nodes), 2)
        self.assertEqual(edges, [{"source_id": 1, "target_id": 2}])
        self.assertEqual(graph["num_divisions"], 0)
        self.assertFalse(graph["dense_nxn"])
        self.assertLessEqual(graph["candidate_edges"], 4)

    def test_05_output_basename_limit(self) -> None:
        with self.assertRaises(sv.V6Error):
            sv._parse_args(["--dataset", "6bba_0c7fa718", "--output", "reports/g02/too-long-score.json"])

    def test_06_nonfinite_metrics_serialize_as_null(self) -> None:
        payload = sv._canonical_bytes({"division_jaccard": float("nan")})
        self.assertEqual(payload, b'{"division_jaccard":null}\n')

    def test_07_frozen_rung_is_fold0_cv_only(self) -> None:
        fold, datasets, proof = sv.fold_scope(self.split, self.metrics, "cv", sv.F0_RUNG_2)
        self.assertEqual((fold, datasets), (0, sv.F0_RUNG_2))
        self.assertEqual(proof["requested_video_ids"], [f"train:{value}" for value in sv.F0_RUNG_2])
        with self.assertRaises(sv.V6Error):
            sv._parse_args(["--rung", "f0-2", "--dataset", sv.F0_RUNG_2[0]])

    def test_08_parent_zero_motion_uses_same_sparse_linker(self) -> None:
        raw = [
            {"node_id": 1, "t": 0, "z": 2, "y": 8, "x": 8, "score": 0.9,
             "backward_um": [0.0, 0.0, 0.0], "log_variance": 0.0},
            {"node_id": 2, "t": 1, "z": 2, "y": 8, "x": 12, "score": 0.8,
             "backward_um": [0.0, 0.0, 0.0], "log_variance": 0.0},
        ]
        _, edges, graph = sv._link_graph(
            "6bba_0c7fa718", raw, (1.625, 0.40625, 0.40625), 8.0, 4, 0.2, "parent"
        )
        self.assertEqual(edges, [{"source_id": 1, "target_id": 2}])
        self.assertFalse(graph["dense_nxn"])
        self.assertEqual(graph["num_divisions"], 0)

    def test_09_public_calibration_threshold_is_allowed(self) -> None:
        args = sv._parse_args(["--dataset", "6bba_0c7fa718", "--det-threshold", "0.97"])
        self.assertEqual(args.det_threshold, 0.97)
        with self.assertRaises(sv.V6Error):
            sv._parse_args(["--dataset", "6bba_0c7fa718", "--det-threshold", "1.0"])

    def test_10_v12_artifact_is_exact_and_fold_safe(self) -> None:
        artifact = sv.checkpoint_artifact(ROOT / "reports/g02/t04v12.json")
        self.assertEqual(artifact["model_id"], "t04v12")
        self.assertEqual(artifact["checkpoint_sha256"], "4bda79c0a6f4e9f3a571ac6a4954b8654161c64ac8977b72c28c70d36c7869be")
        self.assertEqual(artifact["checkpoint_bytes"], 6_373_731)
        fold, datasets, proof = sv.fold_scope(self.split, artifact["metrics"], "cv", ("6bba_0c7fa718",))
        self.assertEqual((fold, datasets), (0, ("6bba_0c7fa718",)))
        self.assertTrue(proof["checkpoint_sources_in_fit"])

    def test_11_artifact_mode_rejects_checkpoint_override(self) -> None:
        args = sv._parse_args([
            "--artifact", "reports/g02/t04v12.json",
            "--checkpoint", "reports/g02/t04v12.pt",
            "--dataset", "6bba_0c7fa718",
        ])
        with self.assertRaises(sv.V6Error):
            sv.selected_artifact(args)


if __name__ == "__main__":
    unittest.main()
