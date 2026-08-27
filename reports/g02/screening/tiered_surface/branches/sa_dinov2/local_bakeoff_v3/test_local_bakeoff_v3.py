from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import constants
import run_features


class IsolatedLocalBakeoffTests(unittest.TestCase):
    def test_frozen_inputs_and_population_are_unchanged(self):
        self.assertEqual(
            constants.PINS[constants.AMENDMENT],
            "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
        )
        self.assertEqual(
            constants.PINS[constants.SEALED_B0],
            "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
        )
        self.assertEqual(
            constants.PINS[constants.POPULATION_MANIFEST],
            "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
        )
        manifest = json.loads(constants.POPULATION_MANIFEST.read_text())
        self.assertEqual(
            sum(len(manifest["classes"][key]["videos"]) for key in constants.CLASSES),
            77,
        )

    def test_atomic_output_uses_replace_and_no_full_pool_cache(self):
        worker = (constants.HERE / "shard_worker.py").read_text()
        supervisor = (constants.HERE / "run_features.py").read_text()
        self.assertIn("os.replace(temporary, path)", worker)
        self.assertIn('"full_pool_cache": False', worker)
        self.assertIn('"full_pool_cache": False', supervisor)

    def test_supervisor_invokes_exactly_one_request_index_per_child(self):
        source = (constants.HERE / "run_features.py").read_text()
        self.assertIn('"--request-index"', source)
        self.assertIn("process.wait()", source)
        self.assertNotIn("ProcessPoolExecutor", source)
        self.assertNotIn("multiprocessing.Pool", source)

    def test_feature_checkpoint_validation_is_shape_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.npz"
            bad = Path(directory) / "bad.npz"
            np.savez(good, a1=np.zeros((3, 8), np.float32), a0=np.zeros((3, 8), np.float32))
            np.savez(bad, a1=np.zeros((3, 7), np.float32), a0=np.zeros((3, 8), np.float32))
            run_features.validate_feature(good, 3)
            with self.assertRaises(run_features.FeatureStageError):
                run_features.validate_feature(bad, 3)

    def test_each_shard_guard_compares_to_global_swap_baseline(self):
        source = (constants.HERE / "run_features.py").read_text()
        self.assertIn("current > global_swap_baseline", source)
        self.assertIn("rerun with a new 45-second preflight to resume", source)

    def test_stage_two_is_the_unchanged_frozen_evaluator_path(self):
        self.assertEqual(
            constants.sha256(constants.EVALUATOR),
            "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca",
        )
        self.assertTrue(constants.V2_AGGREGATOR.is_file())
        self.assertTrue(constants.V2_RESOURCE_GUARD.is_file())


if __name__ == "__main__":
    unittest.main()
