"""Invariants for the compact one-pass V45 D4 student."""

from __future__ import annotations

import inspect
import hashlib
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.bh_model import ArtifactBundle, ArtifactHashes, D4_VIEWS, _encode_d4, infer_video
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256
from celltrack_alpha.v45_fast import (
    FrozenV45Fast,
    V45FastError,
    fast_readout,
    infer_fast,
    infer_fast_streaming,
    load_fast,
    save_fast,
)


class _FakeV45(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = nn.Conv3d(1, 32, kernel_size=1)
        self.detect_head = nn.Conv3d(32, 1, kernel_size=1)
        self.calls = 0
        self.forward_calls = 0

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.calls += 1
        batch, steps, z, y, x = images.shape
        flat = images.reshape(batch * steps, 1, z, y, x)
        features = self.unet(flat).reshape(batch, steps, 32, z, y, x)
        return features, [self.detect_head(features[:, index]) for index in range(steps)]

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.forward_calls += 1
        raise AssertionError("dense V45 forward is forbidden")


class V45FastTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(45)
        self.core = _FakeV45()
        self.model = FrozenV45Fast(self.core)
        self.images = torch.rand(2, 2, 1, 3, 4, 5)

    @staticmethod
    def _bundle() -> ArtifactBundle:
        return ArtifactBundle(
            support_root=ROOT,
            manifest_path=ROOT / "manifest.json",
            config_path=ROOT / "config.json",
            weight_path=ROOT / "weight.pth",
            hashes=ArtifactHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64),
        )

    def test_identity_start_is_exact_raw_v45_and_accounted(self) -> None:
        with torch.no_grad():
            _features, logits = self.core.encode(self.images[:, :, 0])
            expected = torch.stack(tuple(logits), dim=1)
            output = self.model(self.images)
        torch.testing.assert_close(output[:, :, :1], expected, rtol=0.0, atol=0.0)
        self.assertEqual(float(output[:, :, 1:].abs().max()), 0.0)
        self.assertEqual(self.model.accounting()["trainable"], 1_254)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.core.parameters()))
        self.assertEqual(self.core.forward_calls, 0)

    def test_teacher_is_exact_sequential_d4_mean_and_core_is_frozen(self) -> None:
        self.core.calls = 0
        with torch.no_grad():
            _features, expected = _encode_d4(
                torch,
                self.core,
                self.images[:, :, 0],
                view_batch_size=1,
                d4_views=D4_VIEWS,
            )
            expected_tensor = torch.stack(tuple(expected), dim=1)
        self.core.calls = 0
        observed = self.model.teacher(self.images)
        torch.testing.assert_close(observed, expected_tensor, rtol=0.0, atol=0.0)
        self.assertEqual(self.core.calls, 8)
        self.assertEqual(self.core.forward_calls, 0)
        self.assertFalse(observed.requires_grad)

    def test_update_mutates_only_head_and_division_stays_disabled(self) -> None:
        before = {name: value.detach().clone() for name, value in self.core.state_dict().items()}
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=1e-2,
        )
        teacher = self.model.teacher(self.images)
        optimizer.zero_grad(set_to_none=True)
        output = self.model(self.images)
        loss = (output[:, :, :1] - teacher).square().mean() + output[:, :, 1:5].square().mean()
        loss.backward()
        optimizer.step()
        self.model.enforce()
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in self.core.state_dict().items()))
        self.assertEqual(float(self.model.aux_head.weight[4].detach().abs().max()), 0.0)
        self.assertEqual(float(self.model.aux_head.bias[4].detach().abs()), 0.0)
        self.assertTrue(any(
            parameter.grad is not None
            for name, parameter in self.model.named_parameters()
            if not name.startswith("core.")
        ))

    def test_no_dense_or_transformer_path(self) -> None:
        source = inspect.getsource(sys.modules["celltrack_alpha.v45_fast"])
        for forbidden in ("torch.cdist(", "MultiheadAttention(", "model.forward("):
            self.assertNotIn(forbidden, source)

    def test_one_pass_encode_returns_raw_features_and_forward_output(self) -> None:
        self.core.calls = 0
        features, encoded = self.model.encode_fast(self.images)
        self.assertEqual(self.core.calls, 1)
        self.assertEqual(tuple(features.shape), (2, 2, 32, 3, 4, 5))
        self.assertFalse(features.requires_grad)
        self.assertEqual(tuple(encoded.shape), (2, 2, 6, 3, 4, 5))
        self.core.calls = 0
        forwarded = self.model(self.images)
        self.assertEqual(self.core.calls, 1)
        torch.testing.assert_close(forwarded, encoded, rtol=0.0, atol=0.0)

    def test_fast_video_identity_matches_raw_view_and_has_distinct_execution_id(self) -> None:
        with torch.no_grad():
            self.core.detect_head.weight.zero_()
            self.core.detect_head.bias.fill_(20.0)
        self.model.eval()
        frames = [torch.rand(3, 8, 8) for _ in range(3)]
        dependencies = (torch, nn, object, object)
        with (
            patch("celltrack_alpha.v45_fast._require_torch_and_models", return_value=dependencies),
            patch("celltrack_alpha.bh_model._require_torch_and_models", return_value=dependencies),
        ):
            self.core.calls = 0
            fast = infer_fast(self.model, frames, q_low=0.0, q_high=1.0, batch_size=2)
            self.assertEqual(self.core.calls, 1)
            self.core.calls = 0
            raw = infer_video(
                self.core,
                frames,
                q_low=0.0,
                q_high=1.0,
                batch_size=2,
                view_batch_size=1,
                d4_views=(0,),
            )
        self.assertEqual(self.core.calls, 1)
        self.assertEqual(fast.video.d4_views, (0,))
        self.assertNotEqual(fast.execution_sha256, fast.video.config_sha256)
        self.assertEqual(len(fast.video.detections), len(raw.detections))
        self.assertEqual(len(fast.video.contexts), len(raw.contexts))
        for observed, expected in zip(fast.video.detections, raw.detections, strict=True):
            self.assertTrue((observed.node_ids == expected.node_ids).all())
            self.assertTrue((observed.coords_zyx == expected.coords_zyx).all())
            self.assertTrue((observed.grid_coords_zyx == expected.grid_coords_zyx).all())
            self.assertTrue((observed.scores == expected.scores).all())
        for observed, expected in zip(fast.video.contexts, raw.contexts, strict=True):
            self.assertTrue((observed.source_ids == expected.source_ids).all())
            self.assertTrue((observed.target_ids == expected.target_ids).all())
            self.assertTrue((observed.source_feat == expected.source_feat).all())
            self.assertTrue((observed.target_feat == expected.target_feat).all())

    def test_fast_inference_is_one_pass_not_tta(self) -> None:
        source = inspect.getsource(sys.modules["celltrack_alpha.v45_fast"].infer_fast)
        for forbidden in ("_encode_d4(", "D4_VIEWS", "torch.cdist(", "MultiheadAttention("):
            self.assertNotIn(forbidden, source)

    def test_streaming_fast_inference_matches_batch_one_topology_and_features(self) -> None:
        with torch.no_grad():
            self.core.detect_head.weight.zero_()
            self.core.detect_head.bias.fill_(20.0)
        self.model.eval()
        frames = [torch.rand(3, 8, 8) for _ in range(4)]
        dependencies = (torch, nn, object, object)
        with patch(
            "celltrack_alpha.v45_fast._require_torch_and_models",
            return_value=dependencies,
        ):
            materialized = infer_fast(
                self.model, frames, q_low=0.0, q_high=1.0, batch_size=1
            )
            streamed = infer_fast_streaming(
                self.model,
                lambda frame_index: frames[frame_index],
                len(frames),
                q_low=0.0,
                q_high=1.0,
            )
        self.assertEqual(materialized.execution_sha256, streamed.execution_sha256)
        for observed, expected in zip(
            streamed.video.detections, materialized.video.detections, strict=True
        ):
            np.testing.assert_array_equal(observed.node_ids, expected.node_ids)
            np.testing.assert_array_equal(observed.coords_zyx, expected.coords_zyx)
            np.testing.assert_array_equal(observed.scores, expected.scores)
        for observed, expected in zip(
            streamed.video.contexts, materialized.video.contexts, strict=True
        ):
            np.testing.assert_array_equal(observed.source_ids, expected.source_ids)
            np.testing.assert_array_equal(observed.target_ids, expected.target_ids)
            np.testing.assert_array_equal(observed.source_feat, expected.source_feat)
            np.testing.assert_array_equal(observed.target_feat, expected.target_feat)

    def test_compact_checkpoint_binds_core_target_and_disabled_division(self) -> None:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=1e-2,
        )
        optimizer.zero_grad(set_to_none=True)
        self.model(self.images)[:, :, :1].square().mean().backward()
        optimizer.step()
        self.model.enforce()
        bundle = self._bundle()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = pathlib.Path(temporary) / "v45f.pt"
            checkpoint_sha256, expected = save_fast(
                checkpoint,
                self.model,
                bundle,
                target_sha256=TARGET_CONTRACT_SHA256,
                steps=1,
                train_id={"fold": 0, "teacher": "d4-8"},
            )
            self.assertEqual(checkpoint_sha256, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertLess(checkpoint.stat().st_size, 20_000)
            core = _FakeV45()
            core.load_state_dict(self.core.state_dict())
            restored = FrozenV45Fast(core)
            actual = load_fast(
                checkpoint,
                restored,
                bundle,
                expected_sha256=checkpoint_sha256,
                target_sha256=TARGET_CONTRACT_SHA256,
            )
            self.assertEqual(actual, expected)
            for name, value in self.model.fast_state().items():
                torch.testing.assert_close(restored.fast_state()[name], value, rtol=0.0, atol=0.0)
            with self.assertRaises(V45FastError):
                load_fast(
                    checkpoint,
                    restored,
                    bundle,
                    expected_sha256=checkpoint_sha256,
                    target_sha256="e" * 64,
                )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["state"]["aux_head.weight"][4, 0, 0, 0, 0] = 1.0
            torch.save(payload, checkpoint)
            tampered_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with self.assertRaisesRegex(V45FastError, "division"):
                load_fast(
                    checkpoint,
                    restored,
                    bundle,
                    expected_sha256=tampered_sha256,
                    target_sha256=TARGET_CONTRACT_SHA256,
                )

    def test_fast_readout_binds_exact_loaded_auxiliary_state(self) -> None:
        bundle = self._bundle()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = pathlib.Path(temporary) / "v45f.pt"
            checkpoint_sha256, ident = save_fast(
                checkpoint,
                self.model,
                bundle,
                target_sha256=TARGET_CONTRACT_SHA256,
                steps=1,
                train_id={"fold": 0, "teacher": "d4-8"},
            )
            readout = fast_readout(self.model, bundle, ident, checkpoint_sha256=checkpoint_sha256)
            self.assertEqual(readout.checkpoint_sha256, checkpoint_sha256)
            self.assertEqual(readout.identity.training_identity_sha256, ident.train_sha256)
            self.assertEqual(readout.identity.completed_steps, ident.steps)
            features = torch.rand(4, 32)
            expected = features @ self.model.aux_head.weight.detach().reshape(5, 32).T
            expected = expected + self.model.aux_head.bias.detach()
            torch.testing.assert_close(readout(features), expected, rtol=0.0, atol=0.0)
            with torch.no_grad():
                self.model.aux_head.bias[0].add_(1.0)
            with self.assertRaisesRegex(V45FastError, "state binding"):
                fast_readout(self.model, bundle, ident, checkpoint_sha256=checkpoint_sha256)


if __name__ == "__main__":
    unittest.main()
