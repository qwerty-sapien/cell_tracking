from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import kaggle_prelaunch_guard as guard


DATASET = "onebitbrain/g02-dinov2-r5-cpu-input-v1"
MOUNT = "/kaggle/input/datasets/onebitbrain/g02-dinov2-r5-cpu-input-v1"
MANIFEST_SHA = "a" * 64
CHECKPOINT_SHA = "b" * 64
POPULATION_SHA = "c" * 64


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class PrelaunchGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.kernel = self.root / "kernel"
        self.kernel.mkdir()
        self.registry = {
            "policy": {
                "private_dataset_mount_template": "/kaggle/input/datasets/{owner}/{slug}"
            },
            "dataset_ownership": {DATASET: "DINOV2_R5_CPU_ONLY"},
        }
        self.metadata = {
            "id": "onebitbrain/test",
            "title": "Test",
            "code_file": "script.py",
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [DATASET],
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def valid_source(self) -> str:
        return (
            "from pathlib import Path\n"
            f"EXPECTED_MANIFEST_SHA256 = '{MANIFEST_SHA}'\n"
            f"EXPECTED_CHECKPOINT_SHA256 = '{CHECKPOINT_SHA}'\n"
            f"EXPECTED_POPULATION_SHA256 = '{POPULATION_SHA}'\n"
            "EXPECTED_REQUEST_COUNT = 1391458\n"
            "EXPECTED_REQUEST_FILE_COUNT = 77\n"
            "EXPECTED_WHEEL_COUNT = 3\n"
            "RUN_LOG_PATH = 'RUN_LOG.jsonl'\n"
            "def inventory_input(): pass\n"
            "def log_lookup(): pass\n"
            "def stage_end(): pass\n"
            "def preprocess_counts(): pass\n"
            "def _run_failure(): pass\n"
            "def run_step0():\n"
            f"    return Path('{MOUNT}')\n"
            "STEP0 = run_step0()\n"
            "import numpy\n"
        )

    def contract(self, source: str, attested: bool = True) -> dict:
        code = self.kernel / "script.py"
        code.write_text(source, encoding="utf-8")
        write_json(self.kernel / "kernel-metadata.json", self.metadata)
        payload = {
            "kernel_id": "onebitbrain/test",
            "remote_preflight_kernel_id": "onebitbrain/test",
            "remote_preflight_mode": "inline_step0_before_heavy_work",
            "lane_owner": "DINOV2_R5_CPU_ONLY",
            "dataset_sources": [DATASET],
            "exclusive_dataset_source": DATASET,
            "dataset_version": 1,
            "expected_mount_root": MOUNT,
            "input_manifest_sha256": MANIFEST_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "population_manifest_sha256": POPULATION_SHA,
            "request_count": 1391458,
            "request_file_count": 77,
            "population_file_count": 77,
            "wheel_count": 3,
            "step0_limit_seconds": 5.0,
            "run_logging": {
                "schema": "celltrack-alpha.run-log.v1",
                "stdout_jsonl": True,
                "persisted_with_artifacts": True,
                "download_on_success": True,
                "inventory_before_resolution": True,
                "failure_traceback_with_state": True,
            },
            "code_sha256": guard.sha256(code),
            "kernel_properties": {
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": False,
            },
        }
        if attested:
            payload["remote_preflight_attestation"] = {
                "status": "PASS",
                "dataset_source": DATASET,
                "dataset_version": 1,
                "dataset_status": "ready",
                "input_manifest_sha256": MANIFEST_SHA,
                "checkpoint_sha256": CHECKPOINT_SHA,
                "population_manifest_sha256": POPULATION_SHA,
                "request_count": 1391458,
                "request_file_count": 77,
                "population_file_count": 77,
                "wheel_count": 3,
                "enable_gpu": False,
                "heavy_work_started": False,
                "step0_elapsed_seconds": 0.1,
            }
        return payload

    def run_guard(self, source: str, phase: str = "static", attested: bool = True):
        return guard.check_kernel(
            self.kernel, self.contract(source, attested), self.registry, phase
        )

    def test_valid_static_and_launch_contracts_pass(self) -> None:
        source = self.valid_source()
        static = self.run_guard(source)
        inline = self.run_guard(source, "inline")
        launch = self.run_guard(source, "launch")
        self.assertEqual(static["status"], "PASS")
        self.assertFalse(static["eligible"])
        self.assertEqual(inline["status"], "PASS")
        self.assertTrue(inline["eligible"])
        self.assertEqual(launch["status"], "PASS")
        self.assertTrue(launch["eligible"])
        self.assertEqual(
            launch["metadata_sha256"],
            guard.sha256(self.kernel / "kernel-metadata.json"),
        )
        self.assertEqual(launch["kernel_dir"], str(self.kernel.resolve()))
        self.assertEqual(len(launch["contract_sha256"]), 64)
        self.assertEqual(len(launch["registry_sha256"]), 64)

    def test_launch_attestation_respects_the_gpu_contract(self) -> None:
        source = self.valid_source()
        self.metadata["enable_gpu"] = True
        contract = self.contract(source)
        contract["kernel_properties"]["enable_gpu"] = True
        contract["remote_preflight_attestation"]["enable_gpu"] = True
        result = guard.check_kernel(self.kernel, contract, self.registry, "launch")
        self.assertEqual(result["status"], "PASS")

    def test_recursive_discovery_is_rejected(self) -> None:
        source = self.valid_source().replace(
            f"Path('{MOUNT}')", "Path('/kaggle/input').glob('**/manifest.json')"
        )
        result = self.run_guard(source)
        self.assertIn("forbidden broad input discovery", " ".join(result["failures"]))

    def test_wrong_mount_root_is_rejected(self) -> None:
        source = self.valid_source().replace(MOUNT, "/kaggle/input/wrong")
        result = self.run_guard(source)
        self.assertIn(
            "exact owner-qualified mount root", " ".join(result["failures"])
        )

    def test_heavy_import_before_step0_is_rejected(self) -> None:
        source = "import zarr\n" + self.valid_source()
        result = self.run_guard(source)
        self.assertIn("heavy import before STEP0", " ".join(result["failures"]))

    def test_sibling_import_is_rejected(self) -> None:
        (self.kernel / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        source = "import helper\n" + self.valid_source()
        result = self.run_guard(source)
        self.assertIn("unpackaged sibling", " ".join(result["failures"]))

    def test_dataset_source_registration_drift_is_rejected(self) -> None:
        source = self.valid_source()
        self.metadata["dataset_sources"] = ["onebitbrain/wrong"]
        result = self.run_guard(source)
        self.assertIn("dataset_sources drift", result["failures"])

    def test_declared_kernel_source_drift_is_rejected(self) -> None:
        source = self.valid_source()
        self.metadata["kernel_sources"] = ["onebitbrain/source-a"]
        contract = self.contract(source)
        contract["kernel_sources"] = ["onebitbrain/source-b"]
        result = guard.check_kernel(self.kernel, contract, self.registry, "static")
        self.assertIn("kernel_sources drift", result["failures"])

    def test_cross_lane_dataset_ownership_is_rejected(self) -> None:
        source = self.valid_source()
        self.registry["dataset_ownership"][DATASET] = "CELLECT_R7_ONLY"
        result = self.run_guard(source)
        self.assertIn("not exclusively registered", " ".join(result["failures"]))

    def test_launch_requires_remote_preflight(self) -> None:
        source = self.valid_source()
        result = self.run_guard(source, "launch", attested=False)
        self.assertIn("remote preflight PASS", " ".join(result["failures"]))

    def test_slow_remote_preflight_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["remote_preflight_attestation"]["step0_elapsed_seconds"] = 5.0
        result = guard.check_kernel(self.kernel, contract, self.registry, "launch")
        self.assertIn("exceeded step-0 limit", " ".join(result["failures"]))

    def test_warning_only_step0_allows_elapsed_time_but_not_competition_access(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["step0_elapsed_is_hard_failure"] = False
        contract["step0_limit_seconds"] = 60.0
        contract["step0_competition_path_visits"] = 0
        contract["remote_preflight_attestation"]["step0_elapsed_seconds"] = 61.0
        contract["remote_preflight_attestation"]["competition_path_visits"] = 0
        self.assertEqual(guard.check_kernel(self.kernel, contract, self.registry, "launch")["status"], "PASS")
        contract["remote_preflight_attestation"]["competition_path_visits"] = 1
        result = guard.check_kernel(self.kernel, contract, self.registry, "launch")
        self.assertIn("competition-path visit drift", " ".join(result["failures"]))

    def test_zero_request_count_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["request_count"] = 0
        contract["remote_preflight_attestation"]["request_count"] = 0
        result = guard.check_kernel(self.kernel, contract, self.registry, "launch")
        self.assertIn("request_count must be positive", result["failures"])

    def test_missing_checkpoint_pin_is_rejected(self) -> None:
        source = self.valid_source().replace(CHECKPOINT_SHA, "d" * 64)
        result = self.run_guard(source)
        self.assertIn("pinned checkpoint SHA absent from code", result["failures"])

    def test_attestation_cannot_start_heavy_work(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["remote_preflight_attestation"]["heavy_work_started"] = True
        result = guard.check_kernel(self.kernel, contract, self.registry, "launch")
        self.assertIn(
            "remote preflight attestation drift: heavy_work_started",
            result["failures"],
        )

    def test_cross_kernel_preflight_attestation_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["remote_preflight_kernel_id"] = "onebitbrain/different-kernel"
        result = guard.check_kernel(self.kernel, contract, self.registry, "launch")
        self.assertIn(
            "remote preflight is not bound to the launch kernel ID",
            result["failures"],
        )

    def test_title_slug_drift_is_rejected(self) -> None:
        source = self.valid_source()
        self.metadata["title"] = "Different Kernel"
        result = self.run_guard(source)
        self.assertIn(
            "kernel title does not resolve to the declared ID",
            result["failures"],
        )


if __name__ == "__main__":
    unittest.main()
