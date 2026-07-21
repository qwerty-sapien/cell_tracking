import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKER = ROOT / "scripts" / "pack_t05.py"
PYTHON = ROOT / "kaggle_kernel" / "t05.py"
NOTEBOOK = ROOT / "kaggle_kernel" / "t05.ipynb"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class T05PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packer = load("t05_packer_test", PACKER)

    def test_01_deterministic_package_identity(self):
        self.packer.check()
        source = PYTHON.read_bytes()
        self.assertLess(len(source), 1_000_000)
        self.assertEqual(hashlib.sha256(source).hexdigest(), json.loads((ROOT / "kaggle_kernel" / "t05.meta.json").read_text())["source"]["sha256"])
        package = json.loads((ROOT / "kaggle_kernel" / "t05.pack.json").read_text())
        for name, identity in package["package_files"].items():
            data = (ROOT / name).read_bytes()
            self.assertEqual(identity, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        notebook = json.loads(NOTEBOOK.read_text())
        self.assertEqual("".join(notebook["cells"][0]["source"]).encode(), source)
        self.assertEqual(notebook["metadata"]["t05_script_sha256"], hashlib.sha256(source).hexdigest())

    def test_02_fixed_policy_and_dry_contract(self):
        module = load("t05_policy_test", PYTHON)
        args = module._args([])
        self.assertTrue(args.armed)
        self.assertEqual((args.detector_policy, args.center_weight, args.teacher_weight), ("released_teacher_distill", 0.1, 1.0))
        self.assertFalse(module._args(["--dry-run"]).armed)
        candidate = module.trial_record()["candidate"]
        self.assertEqual(candidate["detector_policy"], args.detector_policy)
        self.assertFalse(candidate["released_detector_frozen"])
        self.assertNotIn("sparse_negative_weight", PYTHON.read_text())

    def test_03_teacher_loss_has_gradient_and_never_labels_unannotated_voxels_negative(self):
        module = load("t05_loss_test", PYTHON)
        output = torch.zeros((1, 2, 6, 2, 2, 2), requires_grad=True)
        nodes = [{"id": 1, "batch": 0, "time": 0, "z": 0, "y": 0, "x": 0}, {"id": 2, "batch": 0, "time": 1, "z": 0, "y": 0, "x": 1}]
        targets = module.build_sparse_targets(output, nodes, [{"source_id": 1, "target_id": 2}], (1.0, 1.0, 1.0))
        self.assertEqual(int(targets.center_supervision_mask.sum()), 2)
        with torch.no_grad():
            output[:, :, :1, 1, 1, 1] = 3.0
        teacher = torch.zeros_like(output[:, :, :1])
        losses = module.multitask_loss(output, targets, teacher_logits=teacher)
        self.assertGreater(float(losses["teacher"].detach()), 0.0)
        losses["total"].backward()
        self.assertNotEqual(float(output.grad[:, :, :1, 1, 1, 1].abs().sum()), 0.0)

    def test_04_generated_model_exercises_teacher_student_path(self):
        module = load("t05_model_test", PYTHON)

        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Conv3d(1, 32, 1)

            def forward(self, images):
                batch, steps, _, z, y, x = images.shape
                flat = self.proj(images.reshape(batch * steps, 1, z, y, x))
                return flat.reshape(batch, steps, 32, z, y, x)

        model = module.TemporalAdapterModel(TinyBackbone(), torch.nn.Conv3d(32, 1, 1))
        self.assertEqual(model.parameter_accounting()["backbone_trainable"], 0)
        self.assertGreater(model.parameter_accounting()["detect_head_trainable"], 0)
        teacher = module._released_teacher(model, torch.device("cpu"))
        batch = {
            "frames": torch.ones((1, 2, 1, 2, 2, 2)),
            "nodes": [{"id": 1, "batch": 0, "time": 0, "z": 0, "y": 0, "x": 0}, {"id": 2, "batch": 0, "time": 1, "z": 0, "y": 0, "x": 1}],
            "edges": [{"source_id": 1, "target_id": 2}],
            "voxel_size_um": (1.0, 1.0, 1.0),
        }
        losses = module._batch_loss(model, batch, torch.device("cpu"), teacher)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(model.detect_head.weight.grad)
        self.assertIsNone(model.backbone.proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
