from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


training = load_script("train_organizer_f0", "scripts/train_organizer_f0.py")
builder = load_script("build_organizer_f0_kaggle", "scripts/build_organizer_f0_kaggle.py")
benchmark = load_script(
    "benchmark_organizer_f0_mps",
    "scripts/benchmark_organizer_f0_mps.py",
)


class OrganizerF0TrainingTests(unittest.TestCase):
    def test_frozen_scope_is_exact_fit117_and_holdout57(self) -> None:
        scope = training.frozen_scope()
        self.assertEqual(len(scope["fit117"]), 117)
        self.assertEqual(len(scope["parent_holdout58"]), 58)
        self.assertEqual(len(scope["holdout57"]), 57)
        self.assertEqual(len(scope["audit20"]), 20)
        self.assertFalse(set(scope["fit117"]) & set(scope["parent_holdout58"]))
        self.assertFalse(set(scope["fit117"]) & set(scope["audit20"]))
        self.assertNotIn(training.CV57_EXCLUDED_IDENTITY, scope["holdout57"])

    def test_monitor_is_deterministic_and_fit_only(self) -> None:
        fit = training.frozen_scope()["fit117"]
        first = training.monitor_identities(fit)
        second = training.monitor_identities(tuple(reversed(fit)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertTrue(set(first).issubset(fit))

    def test_official_training_model_and_augmentations_are_hash_locked(self) -> None:
        self.assertEqual(training.verify_official_sources(), training.EXPECTED_SOURCE_HASHES)

    def test_resume_rejects_foreign_402_epoch_checkpoint(self) -> None:
        foreign = torch.load(
            ROOT / "external/Tracking_Support/weights/unet_transformer/split_0/checkpoint_last.pth",
            map_location="cpu",
            weights_only=True,
        )
        with self.assertRaisesRegex(training.TrainingBoundaryError, "wrong schema"):
            training.validate_resume(foreign, {})

    def test_resumable_checkpoint_is_weights_only_loadable(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        generator = torch.Generator().manual_seed(17)
        best = training._normalised_state_dict(model)
        payload = training._checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            best_score=0.5,
            best_epoch=1,
            history=[{"epoch": 1, "monitor_score": 0.5}],
            loader_generator=generator,
            provenance={"schema": training.SCHEMA},
            best_model_state_dict=best,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            torch.save(payload, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(restored["epoch"], 1)
        self.assertEqual(set(restored["best_model_state_dict"]), set(best))
        self.assertIn("mps_rng_state", restored)

    def test_armed_device_selection_never_falls_back_to_cpu(self) -> None:
        with (
            mock.patch.object(training.torch.cuda, "is_available", return_value=False),
            mock.patch.object(training.torch.backends.mps, "is_available", return_value=False),
        ):
            with self.assertRaisesRegex(training.TrainingBoundaryError, "CUDA or MPS"):
                training.resolve_device("auto")

    def test_mps_memory_limit_cannot_be_applied_to_another_device(self) -> None:
        with self.assertRaisesRegex(training.TrainingBoundaryError, "requires the MPS"):
            training.configure_accelerator_memory(
                torch.device("cpu"),
                mps_memory_limit_gib=6.0,
            )

    def test_mps_preflight_batch_sizes_are_strict(self) -> None:
        self.assertEqual(benchmark.parse_batch_sizes("1,2,8"), (1, 2, 8))
        with self.assertRaisesRegex(Exception, "positive"):
            benchmark.parse_batch_sizes("1,0")

    def test_convergence_requires_minimum_and_patience(self) -> None:
        history = [
            {
                "epoch": epoch,
                "monitor_score": 0.9 if epoch == 100 else (0.899 if epoch == 140 else None),
                "best_epoch": 100,
            }
            for epoch in range(1, 141)
        ]
        converged, reason = training.convergence_status(
            history,
            epoch=140,
            min_epochs=100,
            patience_epochs=40,
            min_delta=1e-4,
        )
        self.assertTrue(converged)
        self.assertIn("plateau", reason)

    def test_builder_marks_stage_private_fit_only_and_chains_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage0 = builder.build(0, root)
            stage1 = builder.build(1, root)
            self.assertTrue(stage0["private"])
            self.assertTrue(stage0["fit_only_training"])
            self.assertFalse(stage0["holdout_evaluation"])
            self.assertIsNone(stage0["previous_kernel_ref"])
            self.assertEqual(stage1["previous_kernel_ref"], "onebitbrain/org-f0-00")
            launch0 = (root / "stage00" / "launch.py").read_text(encoding="ascii")
            self.assertIn("SOURCE_PAYLOAD", launch0)
            self.assertIn("base64.b64decode", launch0)
            self.assertNotIn('with_name("source")', launch0)
            self.assertIn('"--batch-size", "8"', launch0)
            self.assertIn('"PYTORCH_ALLOC_CONF"', launch0)
            self.assertIn('"--max-hours", "7.5"', launch0)
            self.assertIn('"--epoch-safety-minutes", "100"', launch0)
            launch = (root / "stage01" / "launch.py").read_text(encoding="ascii")
            self.assertIn("--require-resume", launch)
            self.assertIn("expected exactly one validated prior-stage checkpoint", launch)


if __name__ == "__main__":
    unittest.main()
