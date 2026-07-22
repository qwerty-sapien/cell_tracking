from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_bh.py"
SPEC = importlib.util.spec_from_file_location("run_bh", SOURCE)
assert SPEC is not None and SPEC.loader is not None
BH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BH
SPEC.loader.exec_module(BH)


def _attrs(high: float = 100.0) -> dict[str, object]:
    return {
        "image_statistics": {"quantiles": {"0.001": 0.0, "0.999": high}},
        "multiscales": [
            {
                "axes": [{"name": name} for name in ("t", "z", "y", "x")],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}
                        ],
                    }
                ],
            }
        ],
    }


def _make_zarr(path: Path, shape: tuple[int, int, int, int] = (3, 2, 4, 4)) -> np.ndarray:
    import zarr

    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.create_array("0", data=data, chunks=(1, *shape[1:]))
    group.attrs.update(_attrs(float(data.max())))
    return data


def _make_zip(path: Path, shape: tuple[int, int, int, int] = (3, 2, 4, 4)) -> np.ndarray:
    import zarr

    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    store = zarr.storage.ZipStore(path, mode="w")
    try:
        group = zarr.open_group(store=store, path="test/tiny.zarr", mode="w", zarr_format=3)
        group.create_array("0", data=data, chunks=(1, *shape[1:]))
        group.attrs.update(_attrs(float(data.max())))
    finally:
        store.close()
    return data


def _record(dataset: str = "tiny", shape: tuple[int, int, int, int] = (3, 2, 4, 4)) -> object:
    return BH.RegistryRecord("test", dataset, f"test:{dataset}", "1" * 64, shape)


def _write_registry(path: Path, records: list[object]) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "split": record.split,
                "dataset": record.dataset,
                "video_id": record.video_id,
                "exact_identity_sha256": record.exact_identity_sha256,
                "compact_metadata": {"image_shape": list(record.image_shape)},
            }
        )
    path.write_text(
        json.dumps(
            {
                "artifact": "dataset_registry_v1",
                "counts": {"videos": len(rows)},
                "records": rows,
            }
        ),
        encoding="utf-8",
    )


def _fake_inference():
    from celltrack_alpha import bh_model

    frames = []
    for frame_index in range(3):
        ids = np.asarray([frame_index * 2, frame_index * 2 + 1], dtype=np.int64)
        coords = np.asarray([[0, 1, 1], [1, 2, 2]], dtype=np.int64)
        frames.append(
            bh_model.FrameDet(
                frame_index=frame_index,
                node_ids=ids,
                coords_zyx=coords,
                scores=np.asarray([0.98, 0.99], dtype=np.float32),
                grid_coords_zyx=coords,
                grid_shape_zyx=(2, 4, 4),
            )
        )
    contexts = []
    for left, right in zip(frames, frames[1:]):
        contexts.append(
            bh_model.PairContext(
                source_frame=left.frame_index,
                target_frame=right.frame_index,
                source_ids=left.node_ids,
                target_ids=right.node_ids,
                source_feat=np.zeros((2, 32), dtype=np.float32),
                target_feat=np.zeros((2, 32), dtype=np.float32),
                grid_shape_zyx=(2, 4, 4),
            )
        )
    return bh_model.VideoInference(tuple(frames), tuple(contexts))


def _fake_graph_module(observed: dict[str, object] | None = None):
    from celltrack_alpha import bh_graph

    def calibrate(nodes, learned_edge_probabilities=None, image_refine_callback=None):
        if observed is not None:
            observed["learned"] = learned_edge_probabilities
            observed["callback"] = image_refine_callback
        by_frame = {}
        for node in nodes:
            by_frame.setdefault(node.t, []).append(node)
        edges = []
        for frame in (0, 1):
            for left, right in zip(sorted(by_frame[frame], key=lambda item: item.node_id),
                                   sorted(by_frame[frame + 1], key=lambda item: item.node_id)):
                edges.append(bh_graph.GraphEdge(left.node_id, right.node_id, 1.0, 1.0, "adjacent"))
        return bh_graph.GraphResult(
            tuple(nodes),
            tuple(edges),
            {"input_node_count": len(nodes)},
            {"full_frame_dense_matrix": False, "production_eligible": True},
        )

    return SimpleNamespace(
        GraphNode=bh_graph.GraphNode,
        V45_GRAPH_CONFIG=bh_graph.V45_GRAPH_CONFIG,
        intensity_centroid_refine=lambda image, coord: coord,
        calibrate_v45_graph=calibrate,
    )


class SimulatedKernelCrash(RuntimeError):
    pass


class DiscoveryTests(unittest.TestCase):
    def test_registry_directory_discovery_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record()
            image = root / "input/test/tiny.zarr"
            _make_zarr(image)
            registry = root / "registry.json"
            _write_registry(registry, [record])
            records = BH.load_registry(registry)
            selected = BH.select_records(records, split="test", requested=["tiny"])
            jobs = BH.discover_jobs(root / "input", selected)
            self.assertEqual([job.identity for job in jobs], ["test:tiny"])
            with self.assertRaisesRegex(BH.DiscoveryError, "unique"):
                BH.select_records(records, split="test", requested=["tiny", "tiny"])

    def test_zip_discovery_and_open_use_zipstore_without_label_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "competition.zip"
            expected = _make_zip(archive)
            with zipfile.ZipFile(archive, mode="a") as handle:
                handle.writestr("test/tiny.geff/nodes/ids/c/0", b"must-not-read")
                handle.writestr("annotations.json", b"must-not-read")
            job = BH.discover_jobs(archive, [_record()])[0]
            original = zipfile.ZipFile.open
            opened: list[str] = []

            def audited(instance, name, *args, **kwargs):
                text = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
                if ".geff/" in text or "annotation" in text.casefold():
                    raise AssertionError(f"label access attempted: {text}")
                opened.append(text)
                return original(instance, name, *args, **kwargs)

            with mock.patch.object(zipfile.ZipFile, "open", audited):
                handle = BH.open_video(job, expected_frames=3)
                try:
                    np.testing.assert_array_equal(np.asarray(handle.array[1]), expected[1])
                finally:
                    handle.close()
            self.assertTrue(opened)
            self.assertFalse(any("geff" in name or "annotation" in name for name in opened))

    def test_duplicate_zip_image_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "competition.zip"
            _make_zip(archive)
            with zipfile.ZipFile(archive, mode="a") as handle:
                metadata = handle.read("test/tiny.zarr/0/zarr.json")
                handle.writestr("test/tiny.zarr/0/zarr.json", metadata)
            with self.assertRaisesRegex(BH.DiscoveryError, "exactly once"):
                BH.discover_jobs(archive, [_record()])

    def test_submit_directory_discovers_hidden_test_ids_from_bounded_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_zarr(root / "test/hidden_b.zarr", shape=(3, 2, 4, 4))
            _make_zarr(root / "test/hidden_a.zarr", shape=(3, 2, 4, 4))
            (root / "test/hidden_a.geff").mkdir()
            jobs = BH.discover_submit_jobs(root)
            self.assertEqual([job.identity for job in jobs], ["test:hidden_a", "test:hidden_b"])
            self.assertTrue(all(job.record.image_shape == (3, 2, 4, 4) for job in jobs))
            self.assertTrue(
                all(job.record.identity_provenance == BH.RUNTIME_METADATA_IDENTITY for job in jobs)
            )
            self.assertTrue(all(len(job.record.exact_identity_sha256) == 64 for job in jobs))
            model_identity = SimpleNamespace(
                config_sha256="a" * 64,
                source_sha256="b" * 64,
                weight_sha256="c" * 64,
                pair_batch_size=1,
                view_batch_size=1,
            )
            contract = BH.inference_contract(jobs[0], model_identity, frame_count=3)
            self.assertNotIn("data_exact_identity_sha256", contract)
            self.assertEqual(
                contract["data_runtime_metadata_identity_sha256"],
                jobs[0].record.exact_identity_sha256,
            )
            self.assertEqual(contract["data_identity_provenance"], BH.RUNTIME_METADATA_IDENTITY)

    def test_submit_hidden_shape_is_discovered_but_frame_contract_remains_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hidden_shape = (7, 3, 5, 9)
            _make_zarr(root / "test/hidden_shape.zarr", shape=hidden_shape)
            job = BH.discover_submit_jobs(root)[0]
            self.assertEqual(job.record.image_shape, hidden_shape)
            self.assertNotEqual(job.record.image_shape, (100, 64, 256, 256))

            handle = BH.open_video(job, expected_frames=hidden_shape[0])
            handle.close()
            with self.assertRaisesRegex(BH.RunnerError, "exactly 100 frames, got 7"):
                BH.open_video(job)

    def test_submit_hidden_group_metadata_is_hashed_and_locked_scale_checked(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "test/hidden_metadata.zarr"
            _make_zarr(image)
            original = BH.discover_submit_jobs(root)[0]

            group = zarr.open_group(str(image), mode="r+")
            compatible_attrs = _attrs(123.0)
            compatible_attrs["hidden_only_acquisition"] = {
                "instrument": "not-present-in-public-four",
            }
            group.attrs.clear()
            group.attrs.update(compatible_attrs)
            compatible = BH.discover_submit_jobs(root)[0]
            self.assertNotEqual(
                compatible.record.exact_identity_sha256,
                original.record.exact_identity_sha256,
            )
            handle = BH.open_video(compatible, expected_frames=3)
            handle.close()

            incompatible_attrs = _attrs(123.0)
            incompatible_attrs["multiscales"][0]["datasets"][0][
                "coordinateTransformations"
            ][0]["scale"] = [1.0, 2.0, 0.40625, 0.40625]
            group.attrs.clear()
            group.attrs.update(incompatible_attrs)
            incompatible = BH.discover_submit_jobs(root)[0]
            self.assertNotEqual(
                incompatible.record.exact_identity_sha256,
                compatible.record.exact_identity_sha256,
            )
            with self.assertRaisesRegex(BH.RunnerError, "incompatible with locked V45 graph"):
                BH.open_video(incompatible, expected_frames=3)

    def test_submit_zip_discovers_unregistered_test_id_without_label_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "competition.zip"
            _make_zip(archive)
            with zipfile.ZipFile(archive, mode="a") as handle:
                handle.writestr("test/tiny.geff/nodes/ids/c/0", b"must-not-read")
            original = zipfile.ZipFile.read
            opened: list[str] = []

            def audited(instance, name, *args, **kwargs):
                text = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
                if ".geff/" in text:
                    raise AssertionError(f"label access attempted: {text}")
                opened.append(text)
                return original(instance, name, *args, **kwargs)

            with mock.patch.object(zipfile.ZipFile, "read", audited):
                jobs = BH.discover_submit_jobs(archive)
            self.assertEqual([job.identity for job in jobs], ["test:tiny"])
            self.assertEqual(
                opened,
                ["test/tiny.zarr/zarr.json", "test/tiny.zarr/0/zarr.json"],
            )

    def test_submit_cli_rejects_frozen_dataset_selection(self) -> None:
        args = BH.build_parser().parse_args(
            [
                "--input", "/tmp/competition", "--mode", "submit", "--split", "test",
                "--datasets", "test:public",
            ]
        )
        with self.assertRaisesRegex(BH.RunnerError, "complete mounted test split"):
            BH.validate_args(args)


class ModelGraphRouteTests(unittest.TestCase):
    def test_fake_model_path_uses_infer_video_cache_and_motion_graph_only(self) -> None:
        from celltrack_alpha import bh_model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            _make_zarr(input_root / "test/tiny.zarr")
            job = BH.discover_jobs(input_root, [_record()])[0]
            identity = bh_model.CacheIdentity("a" * 64, "b" * 64, "c" * 64)
            observed: dict[str, object] = {"infer_calls": 0}

            class FakeModel:
                def forward(self, *args, **kwargs):
                    raise AssertionError("diagnostic dense transformer was called")

            def fake_infer(model, frames, **kwargs):
                observed["infer_calls"] = int(observed["infer_calls"]) + 1
                self.assertEqual(len(frames), 3)
                self.assertEqual(kwargs["batch_size"], 1)
                self.assertEqual(kwargs["view_batch_size"], 1)
                return _fake_inference()

            first = BH.process_video(
                job,
                model=FakeModel(),
                cache_identity=identity,
                output_dir=root / "out1",
                cache_dir=root / "cache",
                expected_frames=3,
                infer_video_fn=fake_infer,
                graph_module=_fake_graph_module(observed),
            )
            self.assertEqual(observed["infer_calls"], 1)
            self.assertIsNone(observed["learned"])
            self.assertEqual(first["route"]["route"], "motion")
            self.assertFalse(first["cache_reused"])
            self.assertEqual(first["image_support_projection"]["projected_nodes"], 0)
            self.assertTrue(first["image_support_projection"]["topology_preserved"])
            feature = json.loads(Path(first["feature_summary"]).read_text(encoding="utf-8"))
            self.assertIn("features", feature["pregraph"])
            self.assertEqual(feature["image_support_projection"], first["image_support_projection"])

            def dense_guard(*args, **kwargs):
                raise AssertionError("cache replay called model inference")

            second = BH.process_video(
                job,
                model=FakeModel(),
                cache_identity=identity,
                output_dir=root / "out2",
                cache_dir=root / "cache",
                expected_frames=3,
                infer_video_fn=dense_guard,
                graph_module=_fake_graph_module(),
            )
            self.assertTrue(second["cache_reused"])
            self.assertEqual(observed["infer_calls"], 1)
            source = SOURCE.read_text(encoding="utf-8")
            self.assertNotIn("predict_sparse_edges(", source)
            self.assertNotIn("diagnostic_allow_dense=True", source)

    def test_view_batch_mode_is_reported_gated_and_cache_distinct(self) -> None:
        from celltrack_alpha import bh_model

        job = BH.VideoJob(_record(), "directory", "/input", "/input/test/tiny.zarr")
        exact_identity = bh_model.CacheIdentity("a" * 64, "b" * 64, "c" * 64, 1, 1)
        fast_identity = bh_model.CacheIdentity("d" * 64, "b" * 64, "c" * 64, 1, 8)
        exact = BH.inference_contract(job, exact_identity, pair_batch_size=1, view_batch_size=1, frame_count=3)
        fast = BH.inference_contract(job, fast_identity, pair_batch_size=1, view_batch_size=8, frame_count=3)
        self.assertNotEqual(exact["sha256"], fast["sha256"])
        self.assertTrue(exact["inference"]["exact_code_path_control"])
        self.assertEqual(exact["inference"]["target_t4_numeric_parity"], "unverified")
        self.assertFalse(fast["inference"]["exact_code_path_control"])
        self.assertEqual(fast["inference"]["target_t4_numeric_parity"], "not-claimed")
        with self.assertRaisesRegex(BH.RunnerError, "score gate"):
            BH.view_batch_report(pair_batch_size=1, view_batch_size=8, accelerated_gate=None)
        gate = {"passed": True, "pair_batch_size": 1, "view_batch_size": 8, "sha256": "e" * 64}
        report = BH.view_batch_report(pair_batch_size=1, view_batch_size=8, accelerated_gate=gate)
        self.assertEqual(report["mode"], "accelerated-experimental")
        self.assertEqual(report["parity_claim"], "none")
        self.assertTrue(report["real_t4_ab_required"])

        one_view_identity = bh_model.CacheIdentity("f" * 64, "b" * 64, "c" * 64, 1, 1)
        one_view = BH.inference_contract(
            job,
            one_view_identity,
            pair_batch_size=1,
            view_batch_size=1,
            frame_count=3,
            d4_views=(0,),
        )
        self.assertEqual(one_view["inference"]["d4_views"], [0])
        self.assertEqual(one_view["inference"]["d4_mode"], "test-only-one-view")
        self.assertNotEqual(one_view["sha256"], exact["sha256"])

    def test_edge_artifact_cli_pin_and_one_view_gate_fail_closed(self) -> None:
        artifact = (
            ROOT
            / "_ARCHIVE/conflict_residual/reports/g02/"
            "conflict_edge_f0_fit117_v1/edge.json"
        )
        expected_sha256 = "5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034"
        record = BH.pinned_edge_artifact_record(artifact, expected_sha256)
        self.assertEqual(record["sha256"], expected_sha256)
        self.assertEqual(record["fitted_parameter_count"], 578)
        self.assertFalse(record["division_enabled"])
        with self.assertRaisesRegex(BH.RunnerError, "does not match"):
            BH.pinned_edge_artifact_record(artifact, "0" * 64)

        args = BH.build_parser().parse_args(
            [
                "--input", "/tmp/competition.zip",
                "--mode", "cv", "--split", "train", "--test-only-one-view",
                "--edge-artifact", str(artifact),
                "--edge-artifact-sha256", expected_sha256,
            ]
        )
        BH.validate_args(args)
        args.mode = "submit"
        with self.assertRaisesRegex(BH.RunnerError, "CV/train"):
            BH.validate_args(args)

    def test_canonical_detection_conversion_preserves_float_until_csv(self) -> None:
        from celltrack_alpha import bh_graph

        inference = _fake_inference()
        graph_nodes = BH.detections_to_graph_nodes(inference.detections, bh_graph.GraphNode)
        self.assertIsInstance(graph_nodes[0].z, float)
        shifted = bh_graph.GraphNode(99, 0, 0.5, 1.5, 2.5, 1.0)
        result = bh_graph.GraphResult((shifted,), (), {}, {"full_frame_dense_matrix": False})
        rows = BH.graph_rows("tiny", result, (3, 2, 4, 4))
        self.assertEqual((rows[0]["z"], rows[0]["y"], rows[0]["x"]), (0, 2, 2))
        self.assertEqual((shifted.z, shifted.y, shifted.x), (0.5, 1.5, 2.5))

    def test_feature_summary_is_finite_identity_free_and_versioned(self) -> None:
        features = BH.build_route_summary(
            _fake_inference().detections,
            frame_shape_zyx=(2, 4, 4),
            scale_zyx=(1.625, 0.40625, 0.40625),
            q_low=0.0,
            q_high=100.0,
            frames=[np.ones((2, 4, 4), dtype=np.uint16) for _ in range(3)],
        )
        self.assertEqual(features["schema"], BH.FEATURE_SCHEMA)
        self.assertNotIn("dataset", json.dumps(features))
        self.assertTrue(all(np.isfinite(value) for value in features["features"].values()))
        self.assertEqual(features["summary"]["detections_per_frame"], [2, 2, 2])


class ImageSupportProjectionTests(unittest.TestCase):
    SHAPE = (3, 4, 5, 6)

    @staticmethod
    def _result(nodes, edges=()):
        from celltrack_alpha import bh_graph

        return bh_graph.GraphResult(
            tuple(nodes), tuple(edges), {"sentinel": 7},
            {"full_frame_dense_matrix": False, "sentinel": "kept"},
        )

    def test_projects_each_lower_and_upper_boundary(self) -> None:
        from celltrack_alpha import bh_graph

        maxima = (3.0, 4.0, 5.0)
        for axis in range(3):
            for side, raw_value, expected in (
                ("lower", -0.25, 0.0),
                ("upper", maxima[axis] + 0.25, maxima[axis]),
            ):
                with self.subTest(axis=axis, side=side):
                    coords = [1.0, 2.0, 3.0]
                    coords[axis] = raw_value
                    original = bh_graph.GraphNode(11, 1, *coords, score=0.73, synthetic=True)
                    projected = BH.project_graph_result_to_image_support(
                        "boundary", self._result((original,)), self.SHAPE
                    )
                    self.assertEqual(projected.nodes[0].coord_zyx[axis], expected)
                    self.assertEqual(projected.nodes[0].node_id, original.node_id)
                    self.assertEqual(projected.nodes[0].t, original.t)
                    self.assertEqual(projected.nodes[0].score, original.score)
                    self.assertEqual(projected.nodes[0].synthetic, original.synthetic)
                    self.assertEqual(projected.stats["image_support_projected_nodes"], 1)
                    self.assertGreater(projected.stats["image_support_max_correction_um"], 0.0)
                    self.assertEqual(projected.provenance["sentinel"], "kept")

    def test_rounding_only_overshoot_is_corrected_before_public_rows(self) -> None:
        from celltrack_alpha import bh_graph

        node = bh_graph.GraphNode(2, 0, 1.0, 2.0, -0.5000001)
        self.assertEqual(BH.public_round_coordinate(node.x), -1)
        projected = BH.project_graph_result_to_image_support(
            "rounding", self._result((node,)), self.SHAPE
        )
        rows = BH.graph_rows("rounding", projected, self.SHAPE)
        self.assertEqual(rows[0]["x"], 0)
        self.assertEqual(
            projected.provenance["image_support_projection"]["boundary"],
            "post-smoothing/public-output",
        )

    def test_exact_physical_limit_is_accepted(self) -> None:
        from celltrack_alpha import bh_graph

        raw_z = -BH.IMAGE_SUPPORT_MAX_CORRECTION_UM / BH.IMAGE_SUPPORT_SCALE_ZYX[0]
        node = bh_graph.GraphNode(3, 0, raw_z, 1.0, 1.0)
        projected = BH.project_graph_result_to_image_support(
            "exact-limit", self._result((node,)), self.SHAPE
        )
        self.assertEqual(projected.nodes[0].z, 0.0)
        self.assertAlmostEqual(
            projected.stats["image_support_max_correction_um"],
            BH.IMAGE_SUPPORT_MAX_CORRECTION_UM,
        )

    def test_over_limit_rejects_with_complete_node_context(self) -> None:
        from celltrack_alpha import bh_graph

        correction = (
            BH.IMAGE_SUPPORT_MAX_CORRECTION_UM
            + BH.IMAGE_SUPPORT_EPSILON_UM
            + 1.0e-9
        )
        node = bh_graph.GraphNode(
            404, 0, -correction / BH.IMAGE_SUPPORT_SCALE_ZYX[0], 1.0, 1.0
        )
        with self.assertRaises(BH.OutputValidationError) as caught:
            BH.project_graph_result_to_image_support(
                "train:adversarial", self._result((node,)), self.SHAPE
            )
        message = str(caught.exception)
        for token in (
            "train:adversarial", "node_id=404", "raw_coords_zyx=",
            "rounded_coords_zyx=", "image_shape_tzyx=(3, 4, 5, 6)",
            "correction_um=", "refine_max_shift_um",
        ):
            self.assertIn(token, message)

    def test_nonfinite_coordinate_is_rejected_not_clamped(self) -> None:
        from celltrack_alpha import bh_graph

        node = bh_graph.GraphNode(405, 0, float("nan"), 1.0, 1.0)
        with self.assertRaises(BH.OutputValidationError) as caught:
            BH.project_graph_result_to_image_support(
                "train:nonfinite", self._result((node,)), self.SHAPE
            )
        message = str(caught.exception)
        self.assertIn("non-finite", message)
        self.assertIn("train:nonfinite", message)
        self.assertIn("rounded_coords_zyx=(None, 1, 1)", message)
        self.assertIn("correction_um=None", message)

    def test_extreme_finite_coordinate_rejects_without_numeric_overflow(self) -> None:
        from celltrack_alpha import bh_graph

        node = bh_graph.GraphNode(406, 0, 1.0e308, 1.0, 1.0)
        with self.assertRaises(BH.OutputValidationError) as caught:
            BH.project_graph_result_to_image_support(
                "train:extreme", self._result((node,)), self.SHAPE
            )
        message = str(caught.exception)
        self.assertIn("train:extreme", message)
        self.assertIn("node_id=406", message)
        self.assertIn("correction_um=", message)

    def test_topology_ids_times_scores_and_edges_are_preserved(self) -> None:
        from celltrack_alpha import bh_graph

        first = bh_graph.GraphNode(10, 0, 1.0, 1.0, -0.25, score=0.8)
        second = bh_graph.GraphNode(20, 1, 1.0, 1.0, 1.0, score=0.9)
        edge = bh_graph.GraphEdge(10, 20, 1.0, 2.0, "adjacent", 0.7)
        original = self._result((first, second), (edge,))
        projected = BH.project_graph_result_to_image_support(
            "topology", original, self.SHAPE
        )
        self.assertIs(projected.edges, original.edges)
        self.assertEqual(
            [(node.node_id, node.t, node.score, node.synthetic) for node in projected.nodes],
            [(node.node_id, node.t, node.score, node.synthetic) for node in original.nodes],
        )
        summary = projected.provenance["image_support_projection"]
        self.assertEqual((summary["node_count"], summary["edge_count"]), (2, 1))
        self.assertTrue(summary["node_count_preserved"])
        self.assertTrue(summary["topology_preserved"])

    def test_interior_graph_is_a_coordinate_noop(self) -> None:
        from celltrack_alpha import bh_graph

        nodes = (
            bh_graph.GraphNode(1, 0, 1.0, 1.0, 1.0, score=0.5),
            bh_graph.GraphNode(2, 1, 2.0, 2.0, 2.0, score=0.6),
        )
        projected = BH.project_graph_result_to_image_support(
            "interior", self._result(nodes), self.SHAPE
        )
        self.assertEqual(projected.nodes, nodes)
        self.assertIs(projected.nodes[0], nodes[0])
        self.assertEqual(projected.stats["image_support_projected_nodes"], 0)
        self.assertEqual(projected.stats["image_support_max_correction_um"], 0.0)

    def test_smoothing_induced_overshoot_is_bounded_and_projected(self) -> None:
        from celltrack_alpha import bh_graph

        xs = (0.0, 0.0, 10.0, 10.0, 10.0)
        nodes = tuple(
            bh_graph.GraphNode(index, index, 0.0, 0.0, x)
            for index, x in enumerate(xs)
        )
        edges = tuple(
            bh_graph.GraphEdge(index, index + 1, 1.0, 1.0)
            for index in range(len(nodes) - 1)
        )
        smoothed_nodes, smoothed_edges, _stats = bh_graph.line_fit_smooth(nodes, edges)
        self.assertLess(smoothed_nodes[0].x, 0.0)
        smoothed = self._result(smoothed_nodes, smoothed_edges)
        projected = BH.project_graph_result_to_image_support(
            "smoothed", smoothed, (5, 1, 1, 11)
        )
        self.assertEqual(projected.nodes[0].x, 0.0)
        self.assertEqual(projected.edges, smoothed_edges)
        self.assertGreater(projected.stats["image_support_projected_nodes"], 0)

    def test_official_scorer_ingestion_accepts_corrected_rows(self) -> None:
        from celltrack_alpha import bh_graph
        from celltrack_alpha.kaggle_calibration import read_submission

        nodes = (
            bh_graph.GraphNode(1, 0, 0.0, 0.0, -0.5000001),
            bh_graph.GraphNode(2, 1, 0.0, 0.0, 1.0),
        )
        edge = bh_graph.GraphEdge(1, 2, 1.0, 1.0)
        projected = BH.project_graph_result_to_image_support(
            "tiny", self._result(nodes, (edge,)), self.SHAPE
        )
        with tempfile.TemporaryDirectory() as temporary:
            fragment = Path(temporary) / "train__tiny.csv"
            BH.write_rows_atomic(fragment, BH.graph_rows("tiny", projected, self.SHAPE))
            official_graph = read_submission(fragment)["tiny"]
        self.assertEqual(set(official_graph.nodes), {1, 2})
        self.assertEqual(
            [(item.source_id, item.target_id) for item in official_graph.edges],
            [(1, 2)],
        )
        self.assertEqual(official_graph.nodes[1].x, 0)


class WorkerFailurePropagationTests(unittest.TestCase):
    def test_parent_surfaces_bounded_worker_identity_results_and_traceback(self) -> None:
        class FakeProcess:
            def __init__(self, exitcode):
                self.exitcode = exitcode

            def is_alive(self):
                return self.exitcode is None

            def terminate(self):
                self.exitcode = -15

            def join(self, timeout=None):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = root / "worker0.json"
            aggregate.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "rank": 0,
                        "device": "cuda:0",
                        "identity": "train:bounded",
                        "error": "OutputValidationError: bounded failure",
                        "traceback": "Traceback: public boundary",
                        "results": [
                            {
                                "identity": "train:complete",
                                "nodes": 8,
                                "edges": 7,
                                "fragment": "/private/path/fragment.csv",
                            }
                        ],
                        "jobs": ["secret-payload-must-not-surface"],
                    }
                ),
                encoding="utf-8",
            )
            processes = [FakeProcess(1), FakeProcess(None)]
            with self.assertRaises(BH.RunnerError) as caught:
                BH.wait_spawned(processes, 10**20, (aggregate, root / "worker1.json"))
        message = str(caught.exception)
        for token in (
            "train:bounded", "train:complete", "cuda:0", "bounded failure",
            "Traceback: public boundary", "results", "observed_exitcodes",
        ):
            self.assertIn(token, message)
        self.assertNotIn("secret-payload-must-not-surface", message)
        self.assertNotIn("/private/path/fragment.csv", message)


class ResumeDeadlineMergeTests(unittest.TestCase):
    def _fragment_status(self, root: Path, job: object, z: int = 0):
        from celltrack_alpha import bh_graph

        graph = bh_graph.GraphResult(
            (bh_graph.GraphNode(0, 0, z, 0, 0),),
            (),
            {},
            {"full_frame_dense_matrix": False, "production_eligible": True},
        )
        rows = BH.graph_rows(job.record.dataset, graph, job.record.image_shape)
        fragment = root / f"{job.slug}.csv"
        BH.write_rows_atomic(fragment, rows)
        status = {
            "schema": BH.STATUS_SCHEMA,
            "ok": True,
            "identity": job.identity,
            "fragment_sha256": BH.sha256_file(fragment),
            "result": {"fragment": str(fragment)},
        }
        return status

    def test_deterministic_two_shard_merge_and_contiguous_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record("b"), _record("a")]
            jobs = [BH.VideoJob(record, "directory", str(root), str(root / record.dataset)) for record in records]
            shards = BH.shard_jobs(jobs, 2)
            self.assertEqual(sorted(job.identity for shard in shards for job in shard), ["test:a", "test:b"])
            statuses = [self._fragment_status(root, job) for job in jobs]
            first = root / "first.csv"
            second = root / "second.csv"
            BH.merge_fragments(first, jobs, statuses)
            BH.merge_fragments(second, list(reversed(jobs)), list(reversed(statuses)))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            rows = BH.read_rows(first)
            self.assertEqual([row["id"] for row in rows], ["0", "1"])
            self.assertEqual([row["dataset"] for row in rows], ["a", "b"])

    def test_resume_rejects_nonmatching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragment = root / "tiny.csv"
            fragment.write_text(",".join(BH.CSV_FIELDS) + "\n", encoding="utf-8")
            status = root / "tiny.json"
            status.write_text(
                json.dumps(
                    {
                        "schema": BH.STATUS_SCHEMA,
                        "ok": True,
                        "run_contract": {"sha256": "wrong"},
                        "fragment_sha256": BH.sha256_file(fragment),
                        "result": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BH.ResumeHashError, "run hash mismatch"):
                BH._resume_result(status, fragment, {"sha256": "expected"})

    def test_deadline_never_publishes_partial_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "submission.csv"
            row = {
                "id": 0,
                "dataset": "tiny",
                "row_type": "node",
                "node_id": 0,
                "t": 0,
                "z": 0,
                "y": 0,
                "x": 0,
                "source_id": -1,
                "target_id": -1,
            }
            with self.assertRaisesRegex(BH.DeadlineError, "deadline"):
                BH.write_rows_atomic(output, [row], deadline_epoch=0.0)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob("*.tmp")), [])


class CrashRecoveryTests(unittest.TestCase):
    def _case(self, root: Path):
        from celltrack_alpha import bh_model

        input_root = root / "input"
        _make_zarr(input_root / "test/tiny.zarr")
        job = BH.discover_jobs(input_root, [_record()])[0]
        identity = bh_model.CacheIdentity("a" * 64, "b" * 64, "c" * 64)
        observed = {"infer_calls": 0, "graph_calls": 0}

        class FakeModel:
            pass

        def fake_infer(model, frames, **kwargs):
            observed["infer_calls"] += 1
            return _fake_inference()

        graph_module = _fake_graph_module()
        calibrate = graph_module.calibrate_v45_graph

        def counted_calibrate(*args, **kwargs):
            observed["graph_calls"] += 1
            return calibrate(*args, **kwargs)

        graph_module.calibrate_v45_graph = counted_calibrate
        kwargs = {
            "job": job,
            "model": FakeModel(),
            "cache_identity": identity,
            "output_dir": root / "out",
            "cache_dir": root / "cache",
            "expected_frames": 3,
            "infer_video_fn": fake_infer,
            "graph_module": graph_module,
        }
        paths = {
            "cache": root / "cache" / f"{job.slug}.bhc",
            "cache_meta": root / "cache" / f"{job.slug}.json",
            "cache_intent": root / "cache" / f"{job.slug}.intent.json",
            "feature": root / "out/features" / f"{job.slug}.json",
            "fragment": root / "out/fragments" / f"{job.slug}.csv",
            "status": root / "out/status" / f"{job.slug}.json",
            "transaction": root / "out/transactions" / f"{job.slug}.json",
        }
        return job, identity, observed, kwargs, paths

    def _crash_patch(self, cut: str, paths: dict[str, Path]):
        if cut == "cache":
            original = BH._save_inference_cache_bytes

            def crash_after_cache(*args, **kwargs):
                original(*args, **kwargs)
                raise SimulatedKernelCrash("after cache publication")

            return mock.patch.object(BH, "_save_inference_cache_bytes", crash_after_cache)
        if cut == "fragment":
            original = BH.write_rows_atomic

            def crash_after_fragment(path, *args, **kwargs):
                result = original(path, *args, **kwargs)
                if Path(path) == paths["fragment"]:
                    raise SimulatedKernelCrash("after fragment publication")
                return result

            return mock.patch.object(BH, "write_rows_atomic", crash_after_fragment)

        original = BH.atomic_create_json
        target = paths[cut]

        def crash_after_json(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if Path(path) == target:
                raise SimulatedKernelCrash(f"after {cut} publication")
            return result

        return mock.patch.object(BH, "atomic_create_json", crash_after_json)

    @staticmethod
    def _snapshot(paths: dict[str, Path]) -> dict[str, bytes | None]:
        return {name: path.read_bytes() if path.exists() else None for name, path in paths.items()}

    def test_same_contract_recovers_after_every_publication_cut(self) -> None:
        expected_counts = {
            "cache": (2, 1),
            "cache_meta": (1, 1),
            "feature": (1, 2),
            "fragment": (1, 2),
            "status": (1, 1),
        }
        for cut in expected_counts:
            with self.subTest(cut=cut), tempfile.TemporaryDirectory() as temporary:
                _job, _identity, observed, kwargs, paths = self._case(Path(temporary))
                with self._crash_patch(cut, paths):
                    with self.assertRaisesRegex(SimulatedKernelCrash, "publication"):
                        BH.process_video(**kwargs)

                result = BH.process_video(**kwargs)
                self.assertEqual(
                    (observed["infer_calls"], observed["graph_calls"]),
                    expected_counts[cut],
                )
                self.assertEqual(result["resumed"], cut == "status")
                self.assertTrue(all(paths[name].is_file() for name in paths))
                status = BH.strict_json_load(paths["status"])
                feature = BH.strict_json_load(paths["feature"])
                self.assertEqual(status["fragment_sha256"], BH.sha256_file(paths["fragment"]))
                self.assertEqual(status["feature_sha256"], BH.sha256_file(paths["feature"]))
                self.assertEqual(feature["run_contract"], status["run_contract"])

    def test_different_contract_preserves_and_rejects_crash_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _job, _identity, _observed, kwargs, paths = self._case(Path(temporary))
            with self._crash_patch("fragment", paths):
                with self.assertRaises(SimulatedKernelCrash):
                    BH.process_video(**kwargs)
            before = self._snapshot(paths)
            with self.assertRaisesRegex(BH.ResumeHashError, "contract mismatch"):
                BH.process_video(**{**kwargs, "router_sha256": "d" * 64})
            self.assertEqual(self._snapshot(paths), before)

    def test_unowned_one_sided_cache_is_never_removed(self) -> None:
        from celltrack_alpha import bh_model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            job = BH.VideoJob(_record(), "directory", str(root), str(root / "tiny.zarr"))
            identity = bh_model.CacheIdentity("a" * 64, "b" * 64, "c" * 64)
            sparse = bh_model.SparseInferenceCache(
                identity,
                tuple(_fake_inference().detections),
                tuple(_fake_inference().contexts),
                (),
            )
            cache_path, meta_path = BH._cache_paths(cache_dir, job)
            bh_model.save_sparse_cache(cache_path, sparse)
            contract = BH.inference_contract(job, identity, frame_count=3)
            before = cache_path.read_bytes()
            with self.assertRaisesRegex(BH.ResumeHashError, "unowned orphaned"):
                BH.load_inference_cache(cache_dir, job, contract, identity, bh_model)
            self.assertEqual(cache_path.read_bytes(), before)
            self.assertFalse(meta_path.exists())
            self.assertFalse(BH._cache_intent_path(cache_dir, job).exists())

    def test_completed_same_contract_resume_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _job, _identity, _observed, kwargs, paths = self._case(Path(temporary))
            BH.process_video(**kwargs)
            before = self._snapshot(paths)

            def inference_guard(*args, **kwargs):
                raise AssertionError("completed output attempted inference")

            resumed = BH.process_video(**{**kwargs, "infer_video_fn": inference_guard})
            self.assertTrue(resumed["resumed"])
            self.assertEqual(self._snapshot(paths), before)


class ManifestAndInvariantTests(unittest.TestCase):
    def test_fixed_rungs_are_nested_own_fold_and_audit_disjoint(self) -> None:
        first = BH.generate_cv_rungs()
        second = BH.generate_cv_rungs()
        self.assertEqual(first, second)
        self.assertEqual([len(first["rungs"][str(size)]) for size in (10, 30, 100)], [10, 30, 100])
        ids10 = [row["video_id"] for row in first["rungs"]["10"]]
        ids30 = [row["video_id"] for row in first["rungs"]["30"]]
        ids100 = [row["video_id"] for row in first["rungs"]["100"]]
        self.assertEqual(ids10, ids30[:10])
        self.assertEqual(ids30, ids100[:30])
        split = json.loads(BH.DEFAULT_SPLITS.read_text(encoding="utf-8"))
        audit = set(split["audit"]["video_ids"])
        self.assertFalse(set(ids100) & audit)
        for row in first["rungs"]["100"]:
            held = split["folds"][str(row["held_out_fold"])]["held_out_video_ids"]
            self.assertIn(row["video_id"], held)
            self.assertEqual(row["checkpoint_exposure"], "unknown")

    def test_csv_invariants_cover_refs_adjacency_bounds_and_all_datasets(self) -> None:
        record = _record()
        rows = [
            {"id": 0, "dataset": "tiny", "row_type": "node", "node_id": 1, "t": 0,
             "z": 0, "y": 0, "x": 0, "source_id": -1, "target_id": -1},
            {"id": 1, "dataset": "tiny", "row_type": "node", "node_id": 2, "t": 1,
             "z": 0, "y": 1, "x": 1, "source_id": -1, "target_id": -1},
            {"id": 2, "dataset": "tiny", "row_type": "edge", "node_id": -1, "t": -1,
             "z": -1, "y": -1, "x": -1, "source_id": 1, "target_id": 2},
        ]
        report = BH.validate_submission_rows(rows, [record])
        self.assertEqual(report, {"rows": 3, "nodes": 2, "edges": 1, "datasets": 1})
        bad = [dict(row) for row in rows]
        bad[1]["t"] = 2
        with self.assertRaisesRegex(BH.OutputValidationError, "frame-adjacent"):
            BH.validate_submission_rows(bad, [record])
        with self.assertRaisesRegex(BH.OutputValidationError, "cover every"):
            BH.validate_submission_rows([], [record])

    def test_quantiles_source_budget_and_preflight_report(self) -> None:
        self.assertEqual(BH.read_quantile_attrs(_attrs(99.0)), (0.0, 99.0))
        with self.assertRaisesRegex(BH.RunnerError, "q0.001"):
            BH.read_quantile_attrs({})
        size = BH.source_size_report()
        self.assertTrue(size["under_budget"])
        self.assertLess(size["source_bytes"], size["budget_bytes"])
        report = BH.preflight(require_cuda=False)
        self.assertEqual(set(report["versions"]), {"python", "torch", "numpy", "scipy", "zarr"})
        self.assertEqual(report["checkpoint"]["exposure"], "unknown")
        self.assertIn("--no-index", report["offline_wheel_install_command"])
        self.assertEqual(report["runtime_installation"], "forbidden")


if __name__ == "__main__":
    unittest.main()
