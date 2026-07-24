from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from celltrack_alpha.bh_model import (
    ArtifactValidationError,
    CacheIdentity,
    CacheValidationError,
    D4_VIEWS,
    FrameDet,
    FrameFeatures,
    ModelValidationError,
    PairContext,
    PairEdges,
    SourceSoftmaxStats,
    SparseEdgeCOO,
    SparseInferenceCache,
    V45_CONFIG,
    _detect_batch,
    _encode_d4,
    chunked_source_softmax_coo,
    d4_invert_and_average,
    d4_transform,
    deduplicate_frame_detections,
    infer_video,
    load_sparse_cache,
    load_sparse_cache_node_times,
    load_v45_model,
    predict_sparse_edges,
    quantile_normalize,
    save_sparse_cache,
    validate_artifact_bundle,
    validate_sparse_edges,
    v45_config_sha,
)


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "external" / "Tracking_Support"


def _dense_source_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=0, keepdims=True)
    exponentials = np.exp(logits - maximum)
    return exponentials / exponentials.sum(axis=0, keepdims=True)


def _frame(frame_index: int, node_ids: list[int], coords: list[tuple[int, int, int]]) -> FrameFeatures:
    count = len(node_ids)
    coords_array = np.asarray(coords, dtype=np.int64).reshape(count, 3)
    return FrameFeatures(
        frame_index=frame_index,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        coords_zyx=coords_array,
        scores=np.linspace(0.97, 0.99, count, dtype=np.float32),
        embeddings=np.arange(count * 32, dtype=np.float32).reshape(count, 32),
        grid_coords_zyx=coords_array,
        grid_shape_zyx=(16, 16, 16),
    )


def _det(frame_index: int, node_ids: list[int], coords: list[tuple[int, int, int]]) -> FrameDet:
    count = len(node_ids)
    coords_array = np.asarray(coords, dtype=np.int64).reshape(count, 3)
    return FrameDet(
        frame_index=frame_index,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        coords_zyx=coords_array,
        scores=np.linspace(0.97, 0.99, count, dtype=np.float32),
        grid_coords_zyx=coords_array,
        grid_shape_zyx=(16, 16, 16),
    )


def _context(source: FrameDet, target: FrameDet, marker: float) -> PairContext:
    return PairContext(
        source_frame=source.frame_index,
        target_frame=target.frame_index,
        source_ids=source.node_ids,
        target_ids=target.node_ids,
        source_feat=np.full((len(source.node_ids), 32), marker, dtype=np.float32),
        target_feat=np.full((len(target.node_ids), 32), marker + 0.5, dtype=np.float32),
        grid_shape_zyx=source.grid_shape_zyx,
    )


class SourceSoftmaxTests(unittest.TestCase):
    def _assert_sparse_parity(self, logits: np.ndarray, floor: float, chunk: int) -> None:
        expected = _dense_source_softmax(logits.astype(np.float64))
        actual = chunked_source_softmax_coo(
            logits,
            source_chunk_size=chunk,
            floor=floor,
        )
        mask = expected > floor
        expected_source, expected_target = np.nonzero(mask)
        np.testing.assert_array_equal(actual.source, expected_source)
        np.testing.assert_array_equal(actual.target, expected_target)
        np.testing.assert_allclose(
            actual.probabilities,
            expected[expected_source, expected_target],
            rtol=1e-12,
            atol=0.0,
        )
        self.assertEqual(actual.stats.candidate_count, int(mask.sum()))
        self.assertLessEqual(actual.stats.peak_source_chunk, chunk)

    def test_chunked_parity_with_dense_fixture(self) -> None:
        logits = np.asarray(
            [
                [0.2, -1.0, 3.0, 0.0],
                [1.5, 0.3, -2.0, 0.1],
                [-0.4, 2.0, 1.0, -0.2],
                [0.8, -0.2, 0.5, 4.0],
                [-1.2, 0.7, 2.2, 1.0],
            ],
            dtype=np.float64,
        )
        self._assert_sparse_parity(logits, floor=0.03, chunk=2)

    def test_chunked_parity_with_extreme_logits(self) -> None:
        import torch

        logits = np.asarray(
            [
                [1000.0, -1000.0, 700.0],
                [999.0, -999.0, -700.0],
                [-1000.0, 1000.0, 699.0],
                [0.0, 999.0, 698.0],
            ],
            dtype=np.float64,
        )
        self._assert_sparse_parity(logits, floor=1e-12, chunk=1)
        tensor = torch.from_numpy(logits)
        expected = torch.softmax(tensor, dim=0).numpy()
        actual = chunked_source_softmax_coo(tensor, source_chunk_size=1, floor=1e-12)
        expected_source, expected_target = np.nonzero(expected > 1e-12)
        np.testing.assert_array_equal(actual.source, expected_source)
        np.testing.assert_array_equal(actual.target, expected_target)
        np.testing.assert_allclose(
            actual.probabilities,
            expected[expected_source, expected_target],
            rtol=1e-12,
            atol=0.0,
        )

    def test_floor_bounds_cardinality_per_target(self) -> None:
        generator = np.random.default_rng(451)
        logits = generator.normal(size=(257, 11)).astype(np.float64)
        floor = 0.05
        result = chunked_source_softmax_coo(logits, source_chunk_size=17, floor=floor)
        counts = np.bincount(result.target, minlength=logits.shape[1])
        self.assertTrue(np.all(counts <= math_ceil_inverse(floor)))
        self.assertEqual(result.stats.peak_source_chunk, 17)
        self.assertLessEqual(result.stats.peak_chunk_candidate_count, 17 * logits.shape[1])

    def test_sparse_result_has_no_dense_output(self) -> None:
        logits = np.arange(35, dtype=np.float64).reshape(7, 5)
        result = chunked_source_softmax_coo(logits, source_chunk_size=3, floor=0.01)
        self.assertEqual(result.source.ndim, 1)
        self.assertEqual(result.target.ndim, 1)
        self.assertEqual(result.probabilities.ndim, 1)
        self.assertFalse(hasattr(result, "logits"))
        self.assertFalse(hasattr(result, "probability_matrix"))


def math_ceil_inverse(value: float) -> int:
    return int(np.ceil(1.0 / value))


class D4AndNormalizationTests(unittest.TestCase):
    def test_d4_inversion_and_average(self) -> None:
        import torch

        original = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
        views = [d4_transform(original, view) for view in range(8)]
        public_views = [
            original,
            np.flip(original, axis=-1),
            np.flip(original, axis=-2),
            np.flip(original, axis=(-2, -1)),
            np.rot90(original, 1, axes=(-2, -1)),
            np.rot90(original, 3, axes=(-2, -1)),
            np.swapaxes(original, -1, -2),
            np.swapaxes(np.rot90(original, 1, axes=(-2, -1)), -1, -2),
        ]
        for actual, expected in zip(views, public_views):
            np.testing.assert_array_equal(actual, expected)
        averaged = d4_invert_and_average(views)
        np.testing.assert_array_equal(averaged, original)
        tensor = torch.from_numpy(original.copy())
        tensor_views = [d4_transform(tensor, view) for view in range(8)]
        torch.testing.assert_close(d4_invert_and_average(tensor_views), tensor, rtol=0.0, atol=0.0)

    def test_quantile_normalization_clamps_lower_only(self) -> None:
        values = np.asarray([-2.0, 1.0, 2.0, 20.0], dtype=np.float32)
        actual = quantile_normalize(values, 1.0, 2.0)
        np.testing.assert_allclose(actual, [0.0, 0.0, 0.999999, 18.99998], rtol=1e-6)
        self.assertGreater(float(actual[-1]), 1.0)
        with self.assertRaisesRegex(ModelValidationError, "nonfinite"):
            quantile_normalize(np.asarray([0.0, np.nan], dtype=np.float32), 0.0, 1.0)

    def test_v45_config_is_frozen(self) -> None:
        self.assertEqual(V45_CONFIG.unet_layers, (32, 64, 128))
        self.assertEqual(V45_CONFIG.spatial_stride_zyx, (1, 4, 4))
        self.assertEqual(V45_CONFIG.detection_threshold, 0.970)
        self.assertEqual(V45_CONFIG.detection_tta, "D4-8-view")
        self.assertFalse(V45_CONFIG.edge_feature_tta)
        self.assertEqual(V45_CONFIG.pool_kernel_downsampled_zyx, (3, 3, 3))
        self.assertEqual(V45_CONFIG.pair_batch_size, 1)
        self.assertEqual(V45_CONFIG.view_batch_size, 1)
        with self.assertRaisesRegex(ValueError, "locked"):
            replace(V45_CONFIG, detection_threshold=0.5)
        with self.assertRaisesRegex(ValueError, "locked"):
            replace(V45_CONFIG, view_batch_size=8)

    def test_d4_subset_identity_is_explicit_and_requires_view_zero(self) -> None:
        exact = v45_config_sha()
        self.assertEqual(exact, v45_config_sha(d4_views=D4_VIEWS))
        self.assertEqual(exact, v45_config_sha(d4_views=(7, 6, 5, 4, 3, 2, 1, 0)))
        self.assertNotEqual(exact, v45_config_sha(d4_views=(0,)))
        self.assertNotEqual(
            v45_config_sha(d4_views=(0, 4)),
            v45_config_sha(d4_views=(0, 1, 2, 3)),
        )
        with self.assertRaisesRegex(ValueError, "include unaugmented view 0"):
            v45_config_sha(d4_views=(1,))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            v45_config_sha(d4_views=(0, 0))


class TemporalApiTests(unittest.TestCase):
    def _assert_video_inference_equal(self, expected: object, actual: object) -> None:
        """Require downstream records to be identical across execution batching."""

        self.assertEqual(expected.d4_views, actual.d4_views)
        self.assertEqual(len(expected.detections), len(actual.detections))
        self.assertEqual(len(expected.contexts), len(actual.contexts))
        for expected_det, actual_det in zip(expected.detections, actual.detections):
            self.assertEqual(expected_det.frame_index, actual_det.frame_index)
            self.assertEqual(expected_det.grid_shape_zyx, actual_det.grid_shape_zyx)
            np.testing.assert_array_equal(expected_det.node_ids, actual_det.node_ids)
            np.testing.assert_array_equal(expected_det.coords_zyx, actual_det.coords_zyx)
            np.testing.assert_array_equal(expected_det.grid_coords_zyx, actual_det.grid_coords_zyx)
            np.testing.assert_array_equal(expected_det.scores, actual_det.scores)
        for expected_context, actual_context in zip(expected.contexts, actual.contexts):
            self.assertEqual(expected_context.frame_pair, actual_context.frame_pair)
            self.assertEqual(expected_context.grid_shape_zyx, actual_context.grid_shape_zyx)
            np.testing.assert_array_equal(expected_context.source_ids, actual_context.source_ids)
            np.testing.assert_array_equal(expected_context.target_ids, actual_context.target_ids)
            np.testing.assert_array_equal(expected_context.source_feat, actual_context.source_feat)
            np.testing.assert_array_equal(expected_context.target_feat, actual_context.target_feat)

    def test_d4_batched_layout_and_ordered_parity(self) -> None:
        import torch

        class EchoTemporal(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.calls: list[torch.Tensor] = []

            def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
                self.calls.append(images.detach().cpu().clone())
                batch, _window, z_size, y_size, x_size = images.shape
                features = torch.zeros(
                    (batch, 2, 32, z_size, y_size, x_size),
                    dtype=images.dtype,
                    device=images.device,
                )
                # The returned view-zero feature volume must not be confused
                # with either another D4 view or the detector-logit average.
                features[:, :, 0] = images + 1000.0
                return features, [
                    images[:, offset : offset + 1].clone()
                    for offset in range(V45_CONFIG.window_size)
                ]

        model_input = torch.arange(2 * 2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 2, 3, 5, 7)
        sequential_model = EchoTemporal().eval()
        expected_features, expected_logits = _encode_d4(
            torch,
            sequential_model,
            model_input,
            view_batch_size=1,
        )
        for view_batch_size in range(2, 9):
            batched_model = EchoTemporal().eval()
            actual_features, actual_logits = _encode_d4(
                torch,
                batched_model,
                model_input,
                view_batch_size=view_batch_size,
            )
            torch.testing.assert_close(actual_features, expected_features, rtol=0.0, atol=0.0)
            for actual, expected in zip(actual_logits, expected_logits):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

            # Rectangular inputs require two shape-compatible batches.  Check
            # every view/pair slice so view 7 cannot disappear as a duplicate.
            seen: list[torch.Tensor] = []
            for call in batched_model.calls:
                self.assertTrue(call.is_contiguous())
                self.assertEqual(call.shape[0] % model_input.shape[0], 0)
                for offset in range(0, call.shape[0], model_input.shape[0]):
                    seen.append(call[offset : offset + model_input.shape[0]])
            self.assertEqual(len(seen), len(D4_VIEWS))
            # The rectangular layout groups non-swapped views first.  View 7
            # therefore precedes the swapped-Y/X group even though summation
            # is restored to public D4 order by ``pending``.
            batch_layout = (0, 1, 2, 3, 7, 4, 5, 6)
            for view, actual_view in zip(batch_layout, seen):
                torch.testing.assert_close(actual_view, d4_transform(model_input, view), rtol=0.0, atol=0.0)

    def test_exact_view_order_first_sight_and_explicit_acceleration(self) -> None:
        import torch

        class FakeTemporal(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.encode_shapes: list[tuple[int, ...]] = []
                self.encode_inputs: list[np.ndarray] = []

            def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
                self.encode_shapes.append(tuple(images.shape))
                self.encode_inputs.append(images.detach().cpu().numpy().copy())
                batch, _window, z_size, y_size, x_size = images.shape
                features = torch.zeros(
                    (batch, 2, 32, z_size, y_size, x_size),
                    dtype=images.dtype,
                    device=images.device,
                )
                logits = [
                    torch.full(
                        (batch, 1, z_size, y_size, x_size),
                        -20.0,
                        dtype=images.dtype,
                        device=images.device,
                    )
                    for _ in range(2)
                ]
                for sample in range(batch):
                    pair_zero = float(images[sample, 0].mean()) < 0.08
                    if pair_zero:
                        source_mask = images[sample, 0] == images[sample, 0].max()
                        target_mask = (images[sample, 1] > 0.4) & (images[sample, 1] < 0.75)
                        source_marker, target_marker = 101.0, 102.0
                    else:
                        source_mask = images[sample, 0] > 0.9
                        target_mask = images[sample, 1] == images[sample, 1].max()
                        source_marker, target_marker = 201.0, 202.0
                    logits[0][sample, 0][source_mask] = 20.0
                    logits[1][sample, 0][target_mask] = 20.0
                    features[sample, 0, 0].fill_(source_marker)
                    features[sample, 1, 0].fill_(target_marker)
                return features, logits

        shape = (3, 20, 28)
        frames = [np.full(shape, value, dtype=np.float32) for value in (1.0, 2.0, 3.0)]
        first_peak = (1, 4, 4)
        first_sight = (1, 4, 8)
        later_only = (1, 12, 20)
        last_peak = (1, 8, 16)
        frames[0][first_peak] = 5.0
        frames[1][first_sight] = 10.0
        frames[1][later_only] = 20.0
        frames[2][last_peak] = 15.0

        model = FakeTemporal().eval()
        result = infer_video(
            model,
            frames,
            q_low=0.0,
            q_high=20.0,
            node_id_start=50,
        )

        self.assertEqual([item.frame_index for item in result.detections], [0, 1, 2])
        np.testing.assert_array_equal(result.detections[1].coords_zyx, [first_sight])
        self.assertNotIn(later_only, [tuple(row) for row in result.detections[1].coords_zyx])
        np.testing.assert_array_equal(result.contexts[0].target_ids, result.detections[1].node_ids)
        np.testing.assert_array_equal(result.contexts[1].source_ids, result.detections[1].node_ids)
        np.testing.assert_array_equal(result.contexts[1].source_feat[:, 0], [201.0])
        self.assertFalse(hasattr(result.contexts[1], "logits"))
        self.assertFalse(hasattr(result.contexts[1], "probability_matrix"))
        self.assertEqual(result.pair_batch_size, 1)
        self.assertEqual(result.view_batch_size, 1)

        per_pair_shapes = [
            (1, 2, 3, 5, 7),
            (1, 2, 3, 5, 7),
            (1, 2, 3, 5, 7),
            (1, 2, 3, 5, 7),
            (1, 2, 3, 7, 5),
            (1, 2, 3, 7, 5),
            (1, 2, 3, 7, 5),
            (1, 2, 3, 5, 7),
        ]
        self.assertEqual(model.encode_shapes, per_pair_shapes * 2)
        normalized = [
            quantile_normalize(frame[:, ::4, ::4], 0.0, 20.0)
            for frame in frames
        ]
        expected_inputs = [
            d4_transform(np.stack((normalized[pair], normalized[pair + 1])), view)
            for pair in range(2)
            for view in range(8)
        ]
        for actual, expected in zip(model.encode_inputs, expected_inputs):
            np.testing.assert_array_equal(actual[0], expected)

        accelerated_model = FakeTemporal().eval()
        accelerated = infer_video(
            accelerated_model,
            frames,
            q_low=0.0,
            q_high=20.0,
            node_id_start=50,
            batch_size=2,
            view_batch_size=8,
        )
        self.assertEqual(accelerated_model.encode_shapes[0], (10, 2, 3, 5, 7))
        self.assertEqual(accelerated_model.encode_shapes[1], (6, 2, 3, 7, 5))
        self.assertEqual(accelerated.pair_batch_size, 2)
        self.assertEqual(accelerated.view_batch_size, 8)
        self.assertNotEqual(result.config_sha256, accelerated.config_sha256)
        self._assert_video_inference_equal(result, accelerated)

        # Isolate execution dimensions.  A parity failure here must identify
        # whether view batching or pair batching caused the downstream change.
        view_batched = infer_video(
            FakeTemporal().eval(),
            frames,
            q_low=0.0,
            q_high=20.0,
            node_id_start=50,
            view_batch_size=8,
        )
        pair_batched = infer_video(
            FakeTemporal().eval(),
            frames,
            q_low=0.0,
            q_high=20.0,
            node_id_start=50,
            batch_size=2,
        )
        self._assert_video_inference_equal(result, view_batched)
        self._assert_video_inference_equal(result, pair_batched)

    def test_video_requires_three_frames(self) -> None:
        with self.assertRaisesRegex(ModelValidationError, "at least 3"):
            infer_video(object(), [np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], q_low=0, q_high=1)

    def test_reduced_d4_subset_is_canonical_and_uses_only_selected_views(self) -> None:
        import torch

        class FakeTemporal(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.encode_shapes: list[tuple[int, ...]] = []

            def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
                self.encode_shapes.append(tuple(images.shape))
                batch, _window, z_size, y_size, x_size = images.shape
                features = torch.zeros(
                    (batch, 2, 32, z_size, y_size, x_size),
                    dtype=images.dtype,
                    device=images.device,
                )
                logits = [
                    torch.full(
                        (batch, 1, z_size, y_size, x_size),
                        -20.0,
                        dtype=images.dtype,
                        device=images.device,
                    )
                    for _ in range(2)
                ]
                return features, logits

        frames = [np.full((3, 20, 28), value, dtype=np.float32) for value in (1.0, 2.0, 3.0)]
        model = FakeTemporal().eval()
        result = infer_video(
            model,
            frames,
            q_low=0.0,
            q_high=3.0,
            d4_views=(4, 0),
        )
        self.assertEqual(result.d4_views, (0, 4))
        self.assertNotEqual(result.config_sha256, v45_config_sha())
        self.assertEqual(
            model.encode_shapes,
            [(1, 2, 3, 5, 7), (1, 2, 3, 7, 5)] * 2,
        )
        with self.assertRaisesRegex(ModelValidationError, "include unaugmented view 0"):
            infer_video(
                FakeTemporal().eval(),
                frames,
                q_low=0.0,
                q_high=3.0,
                d4_views=(1,),
            )

    def test_dense_checkpoint_scorer_fails_closed(self) -> None:
        source = _frame(0, [1], [(1, 1, 1)])
        target = _frame(1, [2], [(1, 1, 2)])
        with self.assertRaisesRegex(ModelValidationError, "diagnostic-only"):
            predict_sparse_edges(object(), source, target)


class ArtifactTests(unittest.TestCase):
    def test_local_artifact_validates_and_wrong_hash_fails(self) -> None:
        bundle = validate_artifact_bundle(SUPPORT)
        self.assertEqual(
            bundle.hashes.weight_sha256,
            "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
        )
        with self.assertRaisesRegex(ArtifactValidationError, "weight_sha256 mismatch"):
            validate_artifact_bundle(SUPPORT, expected_weight_sha256="0" * 64)
        exact = CacheIdentity.from_artifact(bundle)
        accelerated = CacheIdentity.from_artifact(bundle, view_batch_size=8)
        pair_batched = CacheIdentity.from_artifact(bundle, pair_batch_size=2)
        self.assertEqual((exact.pair_batch_size, exact.view_batch_size), (1, 1))
        self.assertEqual(accelerated.view_batch_size, 8)
        self.assertNotEqual(exact.config_sha256, accelerated.config_sha256)
        self.assertNotEqual(exact.config_sha256, pair_batched.config_sha256)
        self.assertNotEqual(exact, accelerated)

    def test_checkpoint_load_is_strict_and_architecture_exact(self) -> None:
        model, bundle = load_v45_model(SUPPORT)
        self.assertFalse(model.training)
        self.assertEqual(model.unet_out_channels, 32)
        self.assertEqual(bundle.hashes.weight_sha256, hashlib.sha256(bundle.weight_path.read_bytes()).hexdigest())
        self.assertEqual(model.detect_head.weight.shape, (1, 32, 1, 1, 1))
        with self.assertRaisesRegex(ModelValidationError, "diagnostic-only"):
            model()
        with self.assertRaisesRegex(ModelValidationError, "diagnostic-only"):
            model.transformer()

    def test_locked_checkpoint_view_batch_keeps_logits_peaks_and_video_records(self) -> None:
        import torch

        model, _bundle = load_v45_model(SUPPORT)
        model_input = torch.linspace(-1.0, 1.0, steps=2 * 2 * 8 * 8 * 8).reshape(2, 2, 8, 8, 8)
        state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with torch.inference_mode():
            sequential_features, sequential_logits = _encode_d4(
                torch,
                model,
                model_input,
                view_batch_size=1,
            )
            batched_features, batched_logits = _encode_d4(
                torch,
                model,
                model_input,
                view_batch_size=8,
            )
        torch.testing.assert_close(batched_features, sequential_features, rtol=1e-6, atol=1e-6)
        for expected_logits, actual_logits in zip(sequential_logits, batched_logits):
            torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-6, atol=1e-6)
            expected_peaks = _detect_batch(torch, expected_logits)
            actual_peaks = _detect_batch(torch, actual_logits)
            for (expected_coords, expected_scores), (actual_coords, actual_scores) in zip(
                expected_peaks, actual_peaks
            ):
                torch.testing.assert_close(actual_coords, expected_coords, rtol=0.0, atol=0.0)
                torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-6, atol=1e-6)
        for name, previous in state_before.items():
            torch.testing.assert_close(model.state_dict()[name], previous, rtol=0.0, atol=0.0)

        frame = np.linspace(0.0, 1.0, num=8 * 32 * 32, dtype=np.float32).reshape(8, 32, 32)
        frames = [frame, np.flip(frame, axis=1).copy(), np.flip(frame, axis=2).copy()]
        sequential_video = infer_video(model, frames, q_low=0.0, q_high=1.0, view_batch_size=1)
        batched_video = infer_video(model, frames, q_low=0.0, q_high=1.0, view_batch_size=8)
        TemporalApiTests()._assert_video_inference_equal(sequential_video, batched_video)


class CacheAndGraphValidationTests(unittest.TestCase):
    def _cache(self) -> SparseInferenceCache:
        first = _det(4, [10, 11], [(1, 2, 3), (4, 5, 6)])
        middle = _det(5, [20, 21], [(1, 2, 4), (4, 5, 7)])
        last = _det(6, [30, 31], [(1, 2, 5), (4, 5, 8)])
        contexts = (_context(first, middle, 4.0), _context(middle, last, 5.0))
        stats_a = SourceSoftmaxStats(2, 2, 1, 1, 2, 1, 0.01)
        stats_b = SourceSoftmaxStats(2, 2, 2, 2, 2, 2, 0.02)
        edges = (
            PairEdges(
                4,
                5,
                SparseEdgeCOO(
                    np.asarray([10, 11], dtype=np.int64),
                    np.asarray([20, 21], dtype=np.int64),
                    np.asarray([0.75, 0.80], dtype=np.float64),
                    stats_a,
                ),
            ),
            PairEdges(
                5,
                6,
                SparseEdgeCOO(
                    np.asarray([20, 21], dtype=np.int64),
                    np.asarray([30, 31], dtype=np.int64),
                    np.asarray([0.70, 0.85], dtype=np.float64),
                    stats_b,
                ),
            ),
        )
        identity = CacheIdentity("1" * 64, "2" * 64, "3" * 64)
        return SparseInferenceCache(identity, (first, middle, last), contexts, edges)

    def test_deterministic_roundtrip_and_tamper_failure(self) -> None:
        cache = self._cache()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.bhm"
            second = root / "second.bhm"
            third = root / "third.bhm"
            save_sparse_cache(first, cache)
            save_sparse_cache(second, cache)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded = load_sparse_cache(first, expected_identity=cache.identity)
            self.assertEqual(
                load_sparse_cache_node_times(
                    first, expected_identity=cache.identity
                ),
                {10: 4, 11: 4, 20: 5, 21: 5, 30: 6, 31: 6},
            )
            save_sparse_cache(third, loaded)
            self.assertEqual(first.read_bytes(), third.read_bytes())
            self.assertFalse(hasattr(loaded.detections[0], "embeddings"))
            np.testing.assert_array_equal(loaded.contexts[1].source_feat, cache.contexts[1].source_feat)
            np.testing.assert_allclose(
                loaded.edges[1].edges.probabilities,
                cache.edges[1].edges.probabilities,
            )
            self.assertIsNot(loaded.edges[0].edges.stats, loaded.edges[1].edges.stats)
            self.assertNotEqual(
                loaded.edges[0].edges.stats.configured_source_chunk,
                loaded.edges[1].edges.stats.configured_source_chunk,
            )

            tampered = bytearray(first.read_bytes())
            tampered[len(tampered) // 2] ^= 1
            first.write_bytes(tampered)
            with self.assertRaisesRegex(CacheValidationError, "digest mismatch"):
                load_sparse_cache(first, expected_identity=cache.identity)
            with self.assertRaisesRegex(CacheValidationError, "digest mismatch"):
                load_sparse_cache_node_times(
                    first, expected_identity=cache.identity
                )

    def test_cache_hash_mismatch_fails_closed(self) -> None:
        cache = self._cache()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.bhm"
            save_sparse_cache(path, cache)
            wrong = CacheIdentity("f" * 64, cache.identity.source_sha256, cache.identity.weight_sha256)
            with self.assertRaisesRegex(CacheValidationError, "identity mismatch"):
                load_sparse_cache(path, expected_identity=wrong)
            with self.assertRaisesRegex(CacheValidationError, "identity mismatch"):
                load_sparse_cache_node_times(
                    path, expected_identity=wrong
                )
            accelerated = CacheIdentity(
                cache.identity.config_sha256,
                cache.identity.source_sha256,
                cache.identity.weight_sha256,
                pair_batch_size=1,
                view_batch_size=8,
            )
            with self.assertRaisesRegex(CacheValidationError, "batching identity mismatch"):
                load_sparse_cache(path, expected_identity=accelerated)

    def test_cache_rejects_global_or_misscoped_stats(self) -> None:
        cache = self._cache()
        with self.assertRaisesRegex(CacheValidationError, "pair-scoped"):
            SparseInferenceCache(
                cache.identity,
                cache.detections,
                cache.contexts,
                cache.edges[0].edges,  # type: ignore[arg-type]
            )
        bad_stats = SourceSoftmaxStats(1, 2, 1, 1, 1, 1, 0.01)
        bad_pair = PairEdges(
            4,
            5,
            SparseEdgeCOO(
                np.asarray([10], dtype=np.int64),
                np.asarray([20], dtype=np.int64),
                np.asarray([0.9], dtype=np.float64),
                bad_stats,
            ),
        )
        with self.assertRaisesRegex(CacheValidationError, "stats disagree"):
            SparseInferenceCache(
                cache.identity,
                cache.detections,
                cache.contexts,
                (bad_pair,),
            )

    def test_adjacency_duplicate_id_and_duplicate_edge_validation(self) -> None:
        source = _frame(1, [1], [(1, 1, 1)])
        duplicate_target = _frame(2, [1], [(2, 2, 2)])
        stats = SourceSoftmaxStats(1, 1, 1, 1, 1, 1, 0.01)
        edge = SparseEdgeCOO(np.asarray([1]), np.asarray([1]), np.asarray([0.9]), stats)
        with self.assertRaisesRegex(ModelValidationError, "duplicate node ID"):
            validate_sparse_edges((source, duplicate_target), edge)

        skipped_target = _frame(3, [2], [(2, 2, 2)])
        skipped_edge = SparseEdgeCOO(np.asarray([1]), np.asarray([2]), np.asarray([0.9]), stats)
        with self.assertRaisesRegex(ModelValidationError, "non-adjacent"):
            validate_sparse_edges((source, skipped_target), skipped_edge)

        adjacent = _frame(2, [2], [(2, 2, 2)])
        duplicate_stats = SourceSoftmaxStats(1, 1, 1, 1, 2, 2, 0.01)
        duplicate_edges = SparseEdgeCOO(
            np.asarray([1, 1]),
            np.asarray([2, 2]),
            np.asarray([0.9, 0.8]),
            duplicate_stats,
        )
        with self.assertRaisesRegex(ModelValidationError, "duplicate edge"):
            validate_sparse_edges((source, adjacent), duplicate_edges)

        backward = SparseEdgeCOO(np.asarray([2]), np.asarray([1]), np.asarray([0.9]), stats)
        with self.assertRaisesRegex(ModelValidationError, "backward"):
            validate_sparse_edges((source, adjacent), backward)

    def test_detection_deduplication_is_deterministic(self) -> None:
        coords = np.asarray([(1, 2, 3), (0, 0, 0), (1, 2, 3)], dtype=np.int64)
        scores = np.asarray([0.97, 0.98, 0.99], dtype=np.float32)
        embeddings = np.zeros((3, 32), dtype=np.float32)
        embeddings[2, 0] = 45.0
        result = deduplicate_frame_detections(
            7,
            coords,
            scores,
            embeddings,
            grid_coords_zyx=coords,
            grid_shape_zyx=(8, 8, 8),
            node_id_start=100,
        )
        np.testing.assert_array_equal(result.coords_zyx, [(0, 0, 0), (1, 2, 3)])
        np.testing.assert_array_equal(result.node_ids, [100, 101])
        self.assertEqual(float(result.embeddings[1, 0]), 45.0)


if __name__ == "__main__":
    unittest.main()
