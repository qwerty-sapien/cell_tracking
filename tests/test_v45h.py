"""Focused invariants for the frozen-core V45 auxiliary head."""

from __future__ import annotations

import pathlib
import sys
import hashlib
import tempfile
import unittest

import torch
import torch.nn as nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.bh_model import ArtifactBundle, ArtifactHashes
from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256
from celltrack_alpha.v45_head import (
    AUXILIARY_CHANNELS,
    FEATURE_CHANNELS,
    FrozenV45Head,
    V45AuxiliaryCheckpointError,
    V45HeadError,
    load_v45_auxiliary_checkpoint,
    load_v45_auxiliary_readout,
    save_v45_auxiliary_checkpoint,
    v45_state_sha256,
)


class _FakeV45(nn.Module):
    """A narrow encode-only double; its forward route is intentionally unusable."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = nn.Conv3d(1, FEATURE_CHANNELS, kernel_size=1)
        self.detect_head = nn.Conv3d(FEATURE_CHANNELS, 1, kernel_size=1)
        self.transformer = nn.Linear(3, 3)
        self.forward_calls = 0

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch, frames, z, y, x = images.shape
        flat = images.reshape(batch * frames, 1, z, y, x)
        features = self.unet(flat).reshape(batch, frames, FEATURE_CHANNELS, z, y, x)
        logits = [self.detect_head(features[:, index]) for index in range(frames)]
        return features, logits

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.forward_calls += 1
        raise AssertionError("dense V45 forward must never be called by the auxiliary head")


class FrozenV45HeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(45)
        self.core = _FakeV45()
        self.model = FrozenV45Head(self.core)
        self.images = torch.rand(2, 2, 1, 3, 4, 5)

    @staticmethod
    def _bundle(*, weight_sha256: str = "d" * 64) -> ArtifactBundle:
        return ArtifactBundle(
            support_root=ROOT,
            manifest_path=ROOT / "manifest.json",
            config_path=ROOT / "config.json",
            weight_path=ROOT / "weight.pth",
            hashes=ArtifactHashes("a" * 64, "b" * 64, "c" * 64, weight_sha256),
        )

    def test_zero_initialization_preserves_centres_exactly(self) -> None:
        with torch.no_grad():
            _, expected = self.core.encode(self.images[:, :, 0])
            output = self.model(self.images)
        self.assertEqual(tuple(output.shape), (2, 2, 6, 3, 4, 5))
        self.assertTrue(torch.equal(output[:, :, :1], torch.stack(expected, dim=1)))
        self.assertEqual(float(output[:, :, 1:].abs().max()), 0.0)
        self.assertEqual(self.core.forward_calls, 0)

    def test_only_pointwise_head_is_trainable(self) -> None:
        accounting = self.model.parameter_accounting()
        self.assertEqual(accounting["aux_head_total"], FEATURE_CHANNELS * AUXILIARY_CHANNELS + AUXILIARY_CHANNELS)
        self.assertEqual(accounting["trainable"], 165)
        self.assertEqual(accounting["core_trainable"], 0)
        self.assertEqual(
            [name for name, parameter in self.model.named_parameters() if parameter.requires_grad],
            ["aux_head.weight", "aux_head.bias"],
        )

    def test_optimizer_step_cannot_mutate_core_or_centres(self) -> None:
        before_core = {name: value.detach().clone() for name, value in self.core.state_dict().items()}
        before_head = {name: value.detach().clone() for name, value in self.model.aux_head.state_dict().items()}
        before_centres = self.model(self.images)[:, :, :1].detach().clone()
        optimizer = torch.optim.AdamW((parameter for parameter in self.model.parameters() if parameter.requires_grad), lr=1e-2)
        optimizer.zero_grad(set_to_none=True)
        output = self.model(self.images)
        loss = (output[:, :, 1:] - 1.0).square().mean()
        loss.backward()
        optimizer.step()
        after = self.model(self.images)
        self.assertTrue(all(torch.equal(value, before_core[name]) for name, value in self.core.state_dict().items()))
        self.assertTrue(any(not torch.equal(value, before_head[name]) for name, value in self.model.aux_head.state_dict().items()))
        self.assertTrue(torch.equal(after[:, :, :1], before_centres))
        self.assertEqual(self.core.forward_calls, 0)

    def test_train_mode_keeps_core_in_eval(self) -> None:
        self.model.train()
        self.assertTrue(self.model.training)
        self.assertFalse(self.core.training)
        self.assertTrue(self.model.aux_head.training)

    def test_rejects_invalid_inputs_and_specialist_routing(self) -> None:
        with self.assertRaises(V45HeadError):
            self.model(torch.rand(1, 3, 1, 2, 2, 2))
        with self.assertRaises(V45HeadError):
            self.model(self.images, scenario="dense_crowded")

    def test_sparse_point_readout_equals_dense_auxiliary_gather(self) -> None:
        with torch.no_grad():
            self.model.aux_head.weight[:4].normal_()
            self.model.aux_head.bias[:4].normal_()
            self.model.aux_head.weight[4].zero_()
            self.model.aux_head.bias[4].zero_()
        volume = torch.rand(2, FEATURE_CHANNELS, 3, 4, 5)
        batch = torch.tensor([1, 0, 1], dtype=torch.int64)
        z = torch.tensor([2, 0, 1], dtype=torch.int64)
        y = torch.tensor([3, 2, 1], dtype=torch.int64)
        x = torch.tensor([4, 0, 2], dtype=torch.int64)
        expected = self.model.aux_head(volume)[batch, :, z, y, x]
        actual = self.model.auxiliary_at_points(volume[batch, :, z, y, x])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_auxiliary_checkpoint_is_compact_hash_bound_and_division_safe(self) -> None:
        with torch.no_grad():
            self.model.aux_head.weight[:4].normal_()
            self.model.aux_head.bias[:4].normal_()
            self.model.aux_head.weight[4].zero_()
            self.model.aux_head.bias[4].zero_()
        bundle = self._bundle()
        training_identity = {"fold": 0, "selected_sources": ["train:fixture"], "steps": 3}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = pathlib.Path(temporary) / "v45h.pt"
            checkpoint_sha256, expected = save_v45_auxiliary_checkpoint(
                checkpoint,
                self.model,
                bundle,
                target_contract_sha256=TARGET_CONTRACT_SHA256,
                completed_steps=3,
                training_identity=training_identity,
            )
            self.assertEqual(checkpoint_sha256, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(checkpoint.stat().st_size < 10_000, True)
            restored_core = _FakeV45()
            restored_core.load_state_dict(self.core.state_dict())
            restored = FrozenV45Head(restored_core)
            actual = load_v45_auxiliary_checkpoint(
                checkpoint,
                restored,
                bundle,
                expected_checkpoint_sha256=checkpoint_sha256,
                target_contract_sha256=TARGET_CONTRACT_SHA256,
            )
            self.assertEqual(actual, expected)
            for name, value in self.model.auxiliary_state_dict().items():
                torch.testing.assert_close(restored.auxiliary_state_dict()[name], value, rtol=0.0, atol=0.0)
            with self.assertRaises(V45AuxiliaryCheckpointError):
                load_v45_auxiliary_checkpoint(
                    checkpoint,
                    restored,
                    self._bundle(weight_sha256="e" * 64),
                    expected_checkpoint_sha256=checkpoint_sha256,
                    target_contract_sha256=TARGET_CONTRACT_SHA256,
                )
            with self.assertRaises(V45AuxiliaryCheckpointError):
                load_v45_auxiliary_checkpoint(
                    checkpoint,
                    restored,
                    bundle,
                    expected_checkpoint_sha256=checkpoint_sha256,
                    target_contract_sha256="f" * 64,
                )

            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["auxiliary_state"]["weight"][4, 0, 0, 0, 0] = 1.0
            torch.save(payload, checkpoint)
            tampered_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with self.assertRaises(V45AuxiliaryCheckpointError):
                load_v45_auxiliary_checkpoint(
                    checkpoint,
                    restored,
                    bundle,
                    expected_checkpoint_sha256=tampered_sha256,
                    target_contract_sha256=TARGET_CONTRACT_SHA256,
                )

    def test_sparse_readout_restores_without_a_second_v45_core(self) -> None:
        with torch.no_grad():
            self.model.aux_head.weight[:4].normal_()
            self.model.aux_head.bias[:4].normal_()
            self.model.aux_head.weight[4].zero_()
            self.model.aux_head.bias[4].zero_()
        bundle = self._bundle()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = pathlib.Path(temporary) / "v45h.pt"
            checkpoint_sha256, _identity = save_v45_auxiliary_checkpoint(
                checkpoint,
                self.model,
                bundle,
                target_contract_sha256=TARGET_CONTRACT_SHA256,
                completed_steps=1,
                training_identity={"fold": 0, "steps": 1},
            )
            readout = load_v45_auxiliary_readout(
                checkpoint,
                bundle,
                expected_checkpoint_sha256=checkpoint_sha256,
                target_contract_sha256=TARGET_CONTRACT_SHA256,
                core_state_sha256=v45_state_sha256(self.core),
            )
            features = torch.rand(7, FEATURE_CHANNELS)
            torch.testing.assert_close(
                readout(features),
                self.model.auxiliary_at_points(features),
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertFalse(any(parameter.requires_grad for parameter in readout.parameters()))
            with self.assertRaises(V45AuxiliaryCheckpointError):
                load_v45_auxiliary_readout(
                    checkpoint,
                    bundle,
                    expected_checkpoint_sha256=checkpoint_sha256,
                    target_contract_sha256=TARGET_CONTRACT_SHA256,
                    core_state_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
