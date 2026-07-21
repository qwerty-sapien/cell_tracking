"""Unit coverage for the released-core adapter and sparse device contract."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "external/" + "ka" + "ggle-cell-tracking-competition/src")

import torch
from tracking_cellmot.models.temporal_unet import TemporalUNet3D

from celltrack_alpha import t_adapter as ta


WEIGHTS = pathlib.Path("/private/tmp/ckpt-v1/edge_predictor_best.pth")
CONFIG = pathlib.Path("/private/tmp/ckpt-v1/config.json")
V6 = ROOT / "reports/g02/t03-v6.pt"


def rows() -> tuple[list[dict], list[dict]]:
    return ([{"id": 1, "batch": 0, "time": 0, "z": 1, "y": 1, "x": 1},
             {"id": 2, "batch": 0, "time": 1, "z": 1, "y": 2, "x": 1}],
            [{"source_id": 1, "target_id": 2}])


class AdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output = torch.zeros(1, 2, 6, 4, 4, 4)
        self.nodes, self.edges = rows()

    def test_01_exact_config(self) -> None:
        self.assertEqual(ta.verify_config(CONFIG), ta.EXPECTED_CONFIG)

    def test_02_exact_files(self) -> None:
        self.assertIsNone(ta.verify_released_files(WEIGHTS, CONFIG))

    def test_03_strict_restore_identity(self) -> None:
        backbone = TemporalUNet3D(out_channels=32, layers=(32, 64, 128))
        detector, core = ta.load_released_core(backbone, WEIGHTS, CONFIG)
        self.assertEqual(detector.out_channels, 1)
        self.assertEqual((core.tensor_count, core.tensor_numel, core.schema_sha256), (136, 2077996, ta.STATE_SCHEMA_SHA256))

    def test_04_target_follows_output_device(self) -> None:
        target = ta.build_sparse_targets(self.output, self.nodes, self.edges)
        self.assertEqual(target.device, self.output.device)
        self.assertEqual(target.center_supervision_mask.device, self.output.device)
        self.assertEqual(target.center_supervision_mask.dtype, torch.bool)
        self.assertEqual(target.displacement.device, self.output.device)

    def test_05_shape_only_requires_device(self) -> None:
        with self.assertRaises(ta.TargetError):
            ta.build_sparse_targets(self.output.shape, self.nodes, self.edges)

    def test_06_shape_only_explicit_device(self) -> None:
        target = ta.build_sparse_targets(self.output.shape, self.nodes, self.edges, device="cpu")
        self.assertEqual(target.device.type, "cpu")

    def test_07_explicit_wrong_device_rejects(self) -> None:
        with self.assertRaises(ta.TargetError):
            ta.build_sparse_targets(self.output, self.nodes, self.edges, device="meta")

    def test_08_collision_rejects(self) -> None:
        bad = self.nodes + [{"id": 3, "batch": 0, "time": 0, "z": 1, "y": 1, "x": 1}]
        with self.assertRaises(ta.TargetError):
            ta.build_sparse_targets(self.output, bad, self.edges)

    def test_09_multi_parent_rejects(self) -> None:
        bad = self.nodes + [{"id": 3, "batch": 0, "time": 0, "z": 2, "y": 1, "x": 1}]
        with self.assertRaises(ta.TargetError):
            ta.build_sparse_targets(self.output, bad, self.edges + [{"source_id": 3, "target_id": 2}])

    def test_10_nonconsecutive_edge_rejects(self) -> None:
        bad = self.nodes + [{"id": 3, "batch": 0, "time": 0, "z": 2, "y": 1, "x": 1}]
        with self.assertRaises(ta.TargetError):
            ta.build_sparse_targets(self.output, bad, [{"source_id": 1, "target_id": 3}])

    def test_11_nonfinite_coordinate_rejects(self) -> None:
        bad = [dict(self.nodes[0], z=float("nan")), self.nodes[1]]
        with self.assertRaises(ta.TargetError):
            ta.build_sparse_targets(self.output, bad, self.edges)

    def test_12_division_is_disabled_by_default(self) -> None:
        target = ta.build_sparse_targets(self.output, self.nodes, self.edges)
        losses = ta.multitask_loss(self.output, target)
        self.assertEqual(float(losses["division"]), 0.0)

    def test_13_loss_is_finite(self) -> None:
        target = ta.build_sparse_targets(self.output, self.nodes, self.edges)
        losses = ta.multitask_loss(self.output, target)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_13b_center_loss_masks_unlabeled_voxels_and_is_report_only(self) -> None:
        target = ta.build_sparse_targets(self.output, self.nodes, self.edges)
        self.assertEqual(int(target.center_supervision_mask.sum()), len(self.nodes))
        annotated = (0, 0, 0, 1, 1, 1)
        unannotated = (0, 0, 0, 0, 0, 0)
        baseline = torch.zeros_like(self.output, requires_grad=True)
        changed = torch.zeros_like(self.output, requires_grad=True)
        with torch.no_grad():
            changed[unannotated] = 9.0
        baseline_losses = ta.multitask_loss(baseline, target)
        changed_losses = ta.multitask_loss(changed, target)
        self.assertTrue(torch.equal(baseline_losses["center"], changed_losses["center"]))
        self.assertTrue(torch.equal(baseline_losses["total"], changed_losses["total"]))
        changed_losses["total"].backward()
        self.assertEqual(float(changed.grad[:, :, :1].abs().sum()), 0.0)
        annotated_change = torch.zeros_like(self.output, requires_grad=True)
        with torch.no_grad():
            annotated_change[annotated] = 9.0
        self.assertNotEqual(
            float(ta.multitask_loss(annotated_change, target)["center"].detach()),
            float(baseline_losses["center"].detach()),
        )

    def test_13c_empty_center_mask_is_documented_zero(self) -> None:
        target = ta.build_sparse_targets(self.output, [], [])
        losses = ta.multitask_loss(self.output, target)
        self.assertEqual(float(losses["center"]), 0.0)
        self.assertEqual(ta.CENTER_SUPERVISION, "annotated-only")
        self.assertEqual(ta.CENTER_LOSS, "report-only-excluded")

    def test_13e_fractional_child_backward_target_uses_physical_units(self) -> None:
        nodes = [
            {"id": 1, "batch": 0, "time": 0, "z": 1.25, "y": 1.5, "x": 1.75},
            {"id": 2, "batch": 0, "time": 1, "z": 2.0, "y": 3.0, "x": 2.6},
        ]
        target = ta.build_sparse_targets(
            self.output,
            nodes,
            [{"source_id": 1, "target_id": 2}],
            voxel_size_um=(2.0, 0.5, 0.25),
        )
        child = (0, 1, 0, 2, 3, 3)
        self.assertTrue(bool(target.displacement_mask[child]))
        torch.testing.assert_close(
            target.displacement[0, 1, :, 2, 3, 3],
            torch.tensor([-1.5, -0.75, -0.2125]),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(ta.TARGET_CONTRACT["displacement"]["owner"], "child-at-t")
        self.assertEqual(len(ta.TARGET_CONTRACT_SHA256), 64)

    def test_13f_division_owner_and_isotropic_log_variance_contract(self) -> None:
        nodes = [
            {"id": 1, "batch": 0, "time": 0, "z": 1, "y": 1, "x": 1},
            {"id": 2, "batch": 0, "time": 0, "z": 2, "y": 2, "x": 2},
            {"id": 3, "batch": 0, "time": 1, "z": 1, "y": 2, "x": 1},
            {"id": 4, "batch": 0, "time": 1, "z": 2, "y": 3, "x": 2},
            {"id": 5, "batch": 0, "time": 1, "z": 2, "y": 2, "x": 3},
        ]
        target = ta.build_sparse_targets(
            self.output,
            nodes,
            [{"source_id": 1, "target_id": 3}, {"source_id": 2, "target_id": 4}, {"source_id": 2, "target_id": 5}],
        )
        self.assertTrue(bool(target.division_mask[0, 0, 0, 1, 1, 1]))
        self.assertEqual(float(target.division[0, 0, 0, 1, 1, 1]), 0.0)
        self.assertTrue(bool(target.division_mask[0, 0, 0, 2, 2, 2]))
        self.assertEqual(float(target.division[0, 0, 0, 2, 2, 2]), 1.0)
        self.assertFalse(bool(target.division_mask[0, 1, 0, 1, 2, 1]))

        variance_target = ta.build_sparse_targets(self.output, self.nodes, self.edges)
        output = torch.zeros_like(self.output, requires_grad=True)
        with torch.no_grad():
            output[0, 1, 4, 1, 2, 1] = float(torch.log(torch.tensor(2.0)))
        losses = ta.multitask_loss(output, variance_target)
        squared = float(variance_target.displacement[0, 1, :, 1, 2, 1].square().sum())
        expected_nll = 0.5 * (0.5 * squared + 3.0 * float(torch.log(torch.tensor(2.0))))
        self.assertGreater(float(losses["log_variance"].detach()), 0.0)
        self.assertAlmostEqual(float(losses["log_variance"].detach()), expected_nll, places=6)

    def test_14_output_schema_and_specialists(self) -> None:
        backbone = TemporalUNet3D(out_channels=32, layers=(32, 64, 128))
        detector, _ = ta.load_released_core(backbone, WEIGHTS, CONFIG)
        model = ta.TemporalAdapterModel(backbone, detector).eval()
        images = torch.zeros(1, 2, 1, 8, 8, 8)
        with torch.no_grad():
            universal = model(images)
            specialist = model(images, scenario="sparse_clean")
        self.assertEqual(tuple(universal.shape), (1, 2, 6, 8, 8, 8))
        self.assertTrue(torch.equal(universal, specialist))

    def test_15_accounting_and_transformer_exclusion(self) -> None:
        backbone = TemporalUNet3D(out_channels=32, layers=(32, 64, 128))
        detector, _ = ta.load_released_core(backbone, WEIGHTS, CONFIG)
        model = ta.TemporalAdapterModel(backbone, detector)
        self.assertEqual(model.parameter_accounting()["detect_head_trainable"], 0)
        self.assertEqual(model.parameter_accounting()["trainable"], 29627)
        self.assertEqual(model.set_unfreeze_depth(2)["trainable"], 141403)
        self.assertFalse(any("transformer" in name for name, _ in model.named_parameters()))

    def test_15b_released_detector_stays_unchanged_during_auxiliary_step(self) -> None:
        backbone = TemporalUNet3D(out_channels=32, layers=(32, 64, 128))
        detector, _ = ta.load_released_core(backbone, WEIGHTS, CONFIG)
        model = ta.TemporalAdapterModel(backbone, detector)
        detector_before = {name: value.detach().clone() for name, value in model.detect_head.state_dict().items()}
        backbone_before = {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}
        aux_before = {name: value.detach().clone() for name, value in model.aux_head.state_dict().items()}
        output = model(torch.zeros(1, 2, 1, 8, 8, 8))
        target = ta.build_sparse_targets(output, self.nodes, self.edges)
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        ta.multitask_loss(output, target)["total"].backward()
        optimizer.step()
        self.assertTrue(all(torch.equal(value, detector_before[name]) for name, value in model.detect_head.state_dict().items()))
        self.assertTrue(all(torch.equal(value, backbone_before[name]) for name, value in model.backbone.state_dict().items()))
        self.assertTrue(any(not torch.equal(value, aux_before[name]) for name, value in model.aux_head.state_dict().items()))

    def test_15c_teacher_detector_policy_is_explicit_and_backbone_stays_frozen(self) -> None:
        backbone = TemporalUNet3D(out_channels=32, layers=(32, 64, 128))
        detector, _ = ta.load_released_core(backbone, WEIGHTS, CONFIG)
        model = ta.TemporalAdapterModel(backbone, detector)
        accounting = model.set_training_policy(0, detector_policy="released_teacher_distill")
        self.assertEqual(model.detector_policy, "released_teacher_distill")
        self.assertEqual(accounting["detect_head_trainable"], 33)
        self.assertEqual(accounting["backbone_trainable"], 0)
        self.assertEqual(accounting["trainable"], 29660)
        with self.assertRaises(ValueError):
            model.set_training_policy(1, detector_policy="released_teacher_distill")

    def test_15d_teacher_center_loss_anchors_unlabeled_locations_without_negative_labels(self) -> None:
        target = ta.build_sparse_targets(self.output, self.nodes, self.edges)
        output = torch.zeros_like(self.output, requires_grad=True)
        teacher = torch.zeros_like(output[:, :, :1])
        unlabeled = (0, 0, 0, 0, 0, 0)
        with torch.no_grad():
            output[unlabeled] = 2.0
        losses = ta.multitask_loss(
            output,
            target,
            center_weight=0.1,
            teacher_logits=teacher,
            teacher_weight=1.0,
        )
        self.assertGreater(float(losses["teacher"].detach()), 0.0)
        losses["total"].backward()
        self.assertNotEqual(float(output.grad[unlabeled]), 0.0)
        with self.assertRaises(ta.TargetError):
            ta.multitask_loss(output.detach(), target, teacher_weight=1.0)

    def test_16_v6_trained_restore(self) -> None:
        backbone = TemporalUNet3D(out_channels=32, layers=(32, 64, 128))
        model, restored = ta.load_trained_model(backbone, V6, expected_fold=0)
        self.assertEqual((restored.fold, restored.scenario, restored.completed_steps), (0, "universal", 1000))
        self.assertEqual((restored.model_tensor_count, restored.model_numel), (98, 1527270))
        self.assertEqual(len(restored.source_video_ids), 12)
        self.assertFalse(model.training)
        self.assertFalse(any("transformer" in name for name, _ in model.named_parameters()))

    def test_17_v6_wrong_fold_rejects(self) -> None:
        backbone = TemporalUNet3D(out_channels=32, layers=(32, 64, 128))
        with self.assertRaises(ta.CheckpointError):
            ta.load_trained_model(backbone, V6, expected_fold=1)


if __name__ == "__main__":
    unittest.main()
