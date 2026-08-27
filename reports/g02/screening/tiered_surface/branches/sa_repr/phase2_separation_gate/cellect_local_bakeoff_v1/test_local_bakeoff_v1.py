from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import constants
import run_features


class IsolatedCellectBakeoffTests(unittest.TestCase):
    def test_frozen_inputs_checkpoint_and_population_are_unchanged(self):
        observed = constants.verify_pins()
        self.assertEqual(
            observed[str(constants.AMENDMENT)],
            "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
        )
        self.assertEqual(
            observed[str(constants.SEALED_B0)],
            "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
        )
        self.assertEqual(
            observed[str(constants.POPULATION_MANIFEST)],
            "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
        )
        self.assertEqual(constants.CHECKPOINT.stat().st_size, 10_197_221)
        self.assertEqual(
            observed[str(constants.CHECKPOINT)],
            "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5",
        )

    def test_plan_is_exactly_the_77_declared_shards_only(self):
        plan = run_features.build_plan()
        self.assertEqual(plan["request_file_count"], 77)
        self.assertEqual(plan["request_count"], 1_391_458)
        self.assertEqual(plan["tile_jobs"], 217_595)
        self.assertFalse(plan["full_pool_cache"])
        self.assertTrue(
            all(
                str(constants.REQUEST_ROOT) in row["request_path"]
                for row in plan["requests"]
            )
        )

    def test_atomic_output_and_no_full_pool_cache(self):
        worker = (constants.HERE / "shard_worker.py").read_text()
        supervisor = (constants.HERE / "run_features.py").read_text()
        self.assertIn("os.replace(temporary, path)", worker)
        self.assertIn('"full_pool_cache": False', worker)
        self.assertIn('"full_pool_cache": False', supervisor)

    def test_one_request_child_and_global_swap_baseline(self):
        source = (constants.HERE / "run_features.py").read_text()
        self.assertIn('"--request-index"', source)
        self.assertIn("current > global_swap_baseline", source)
        self.assertIn("process.wait()", source)
        self.assertNotIn("ProcessPoolExecutor", source)
        self.assertNotIn("multiprocessing.Pool", source)

    def test_local_adaptation_has_no_cuda_or_external_path(self):
        source = (constants.HERE / "shard_worker.py").read_text()
        self.assertNotIn('.to("cuda")', source)
        self.assertNotIn("/kaggle/", source)
        self.assertIn(".to(device)", source)
        self.assertIn("ZipStore(constants.RAW_ARCHIVE", source)
        self.assertIn("torch.set_num_threads(args.threads)", source)
        self.assertIn("torch.set_num_interop_threads(1)", source)

    def test_feature_checkpoint_validation_is_shape_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.npz"
            bad = Path(directory) / "bad.npz"
            np.savez(
                good,
                a1=np.zeros((3, 8), np.float32),
                a0=np.zeros((3, 8), np.float32),
            )
            np.savez(
                bad,
                a1=np.zeros((3, 7), np.float32),
                a0=np.zeros((3, 8), np.float32),
            )
            run_features.validate_feature(good, 3)
            with self.assertRaises(run_features.FeatureStageError):
                run_features.validate_feature(bad, 3)

    def test_checkpoint_without_output_fails_closed(self):
        row = {
            "ordinal": 0,
            "action_class": "delete",
            "identity": "train:test",
            "action_count": 1,
            "request_sha256": "0" * 64,
            "output_name": "train__test.npz",
            "tile_jobs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints/delete/train__test.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}")
            with (
                mock.patch.object(run_features, "FEATURES", root / "features"),
                mock.patch.object(
                    run_features, "CHECKPOINTS", root / "checkpoints"
                ),
            ):
                with self.assertRaises(run_features.FeatureStageError):
                    run_features.validate_or_recover(row)

    def test_stage_two_is_the_unchanged_frozen_evaluator(self):
        self.assertEqual(
            constants.sha256(constants.EVALUATOR),
            "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca",
        )
        aggregate = (constants.HERE / "aggregate_evaluate.py").read_text()
        self.assertIn('["B0_SEALED", "A1", "A0"]', aggregate)
        self.assertIn("evaluator.evaluate_class(", aggregate)


if __name__ == "__main__":
    unittest.main()
