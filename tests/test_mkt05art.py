import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "scripts" / "mkt05art.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class T05ArtifactTests(unittest.TestCase):
    def test_rejects_noncanonical_or_incomplete_remote_outputs_before_copy(self):
        module = load("mkt05art_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint.pt").write_bytes(b"not-a-model")
            (root / "metrics.json").write_text("{}\n")
            with self.assertRaises(ValueError):
                module.build(root)

    def test_descriptor_schema_is_compatible_with_generic_scorer(self):
        module = load("mkt05art_schema")
        expected = module.expected_identity()
        self.assertEqual(expected["detector_policy"], "released_teacher_distill")
        self.assertEqual(expected["center_weight"], 0.1)
        self.assertEqual(expected["teacher_weight"], 1.0)
        self.assertEqual(expected["fold"], 0)
        trial = json.loads((ROOT / "reports" / "g02" / "t05-trial.json").read_text())
        self.assertEqual(expected["weights_sha256"], trial["immutable_inputs"]["checkpoint_sha256"])

    def test_missing_remote_metrics_can_be_reconstructed_from_weights_only_state(self):
        module = load("mkt05art_rebuild")
        expected = module.expected_identity()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            state = {
                "identity": {**expected, "selected_sources": [{"group_id": "g", "video_id": "train:44b6_aabbccdd", "stem": "44b6_aabbccdd"}]},
                "completed_steps": 1,
                "elapsed_seconds": 1.0,
                "loss_components": {"teacher": 0.0},
                "source_groups": ["g"],
                "source_video_ids": ["train:44b6_aabbccdd"],
                "source_records": [{"group_id": "g", "video_id": "train:44b6_aabbccdd", "stem": "44b6_aabbccdd"}],
                "execution": {"epochs": 1, "max_steps": 1},
                "status": "completed",
                "model": {"unused": torch.tensor([1.0])},
            }
            torch.save(state, checkpoint)
            metrics = module.reconstructed_metrics(checkpoint)
            checkpoint_sha = module.sha256(checkpoint)
        self.assertEqual(metrics["metrics_provenance"]["kind"], "checkpoint-state-reconstruction-v1")
        self.assertEqual(metrics["checkpoint_sha256"], checkpoint_sha)


if __name__ == "__main__":
    unittest.main()
