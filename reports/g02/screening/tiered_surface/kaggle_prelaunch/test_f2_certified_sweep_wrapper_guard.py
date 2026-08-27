from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import build_f2_certified_sweep_contract as contract_builder
import kaggle_prelaunch_guard as guard


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class F2CertifiedSweepWrapperGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.kernel = self.root / "kernel"
        self.child_kernel = self.root / "child_kernel"
        self.evidence = self.root / "evidence"
        self.kernel.mkdir()
        self.child_kernel.mkdir()
        self.evidence.mkdir()
        self.expected = guard.F2_SWEEP_KERNELS["D"]
        self.candidates = [dict(item) for item in guard.F2_SWEEP_CANDIDATES["D"]]
        self._write_lmt()
        self._write_child()
        self._write_outer()
        self._write_profile_and_certificate()
        self.contract = self._contract()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_lmt(self) -> None:
        policy = {
            "schema": "cell-tracking.autonomy-limits.v1",
            "kaggle": {"autonomous_projection_strictly_below_hours": 8.0},
            "local_resource_safety": {"maximum_combined_rss_mib": 8192.0},
        }
        (self.root / "LMT.md").write_text(
            "<!-- LMT_POLICY_JSON_BEGIN -->\n```json\n"
            + json.dumps(policy, sort_keys=True)
            + "\n```\n<!-- LMT_POLICY_JSON_END -->\n",
            encoding="utf-8",
        )

    def _child_source(self) -> str:
        selected = [item.split(":", 1)[1] for item in guard.F2_SWEEP_IDENTITIES]
        return (
            "from pathlib import Path\n"
            f"SELECTED_DATASETS = {selected!r}\n"
            "SELECTED_TRAIN = Path('/tmp/selected-train')\n"
            "RUN_LOG_PATH = 'RUN_LOG.jsonl'\n"
            "def inventory_input(): pass\n"
            "def log_lookup(): pass\n"
            "def stage_end(): pass\n"
            "def preprocess_counts(): pass\n"
            "def _run_failure(): pass\n"
            "def _v32_ship_step0(): return {'status': 'PASS'}\n"
            "SHIP_STEP0 = _v32_ship_step0()\n"
            "def run_step0(): return {'status': 'PASS'}\n"
            "STEP0 = run_step0()\n"
            "def _v32_materialize_selected_train(): pass\n"
            "_v32_materialize_selected_train()\n"
            "TEST_DIR = SELECTED_TRAIN\n"
            "print('V32_HEAVY_WORK_BEGIN')\n"
            "print('V32_FINAL_GRAPH_ROWS_PRESERVED')\n"
            "print('V32_CACHE_NO_SUBMISSION_CONFIRMED')\n"
        )

    def _write_child(self) -> None:
        self.child_code = self.child_kernel / "child.py"
        self.child_code.write_text(self._child_source(), encoding="utf-8")
        self.child_code.rename(self.child_kernel / "v32_child.py")
        self.child_code = self.child_kernel / "v32_child.py"
        write_json(
            self.child_kernel / "kernel-metadata.json",
            {
                "id": guard.F2_CHILD_VISUAL_KERNEL_ID,
                "title": "f2-v32-child-visual-v1",
                "code_file": "v32_child.py",
            },
        )
        self.asset_hashes = {}
        for relative, content in (
            (guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT, b"primary-checkpoint"),
            (guard.F2_CHILD_VISUAL_SECONDARY_CHECKPOINT, b"secondary-checkpoint"),
            (guard.F2_CHILD_VISUAL_PRIMARY_MANIFEST, b"primary-manifest"),
            (guard.F2_CHILD_VISUAL_SECONDARY_MANIFEST, b"secondary-manifest"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.asset_hashes[relative] = guard.sha256(path)
        input_records = []
        raw_prefix = Path(guard.F2_CHILD_VISUAL_RAW_ROOT) / "train"
        raw_paths = []
        for identity in guard.F2_CHILD_VISUAL_IDENTITIES:
            dataset = identity.split(":", 1)[1]
            group = raw_prefix / f"{dataset}.zarr"
            raw_paths.extend((group / "zarr.json", group / "0" / "zarr.json"))
            raw_paths.extend(
                group / "0" / "c" / str(frame) / "0" / "0" / "0"
                for frame in range(guard.F2_CHILD_VISUAL_FRAMES_PER_IDENTITY)
            )
        for relative in raw_paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(str(relative).encode())
        for relative in [*self.asset_hashes, *(str(path) for path in raw_paths)]:
            path = self.root / relative
            input_records.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": guard.sha256(path),
                }
            )
        input_records.sort(key=lambda item: item["path"])
        self.input_manifest = self.evidence / "input.json"
        write_json(
            self.input_manifest,
            {
                "schema": guard.VISUAL_INPUT_MANIFEST_SCHEMA,
                "status": "PASS_PREPARED",
                "embedded_child_sha256": guard.sha256(self.child_code),
                "preregistered_visual_identities": list(
                    guard.F2_CHILD_VISUAL_IDENTITIES
                ),
                "frames_per_identity": guard.F2_CHILD_VISUAL_FRAMES_PER_IDENTITY,
                "label_free": True,
                "labels_or_ground_truth_permitted": False,
                "scorer_permitted": False,
                "competition_test_namespace_permitted": False,
                "primary_checkpoint": {
                    "path": guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT,
                    "sha256": self.asset_hashes[
                        guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT
                    ],
                },
                "secondary_checkpoint": {
                    "path": guard.F2_CHILD_VISUAL_SECONDARY_CHECKPOINT,
                    "sha256": self.asset_hashes[
                        guard.F2_CHILD_VISUAL_SECONDARY_CHECKPOINT
                    ],
                },
                "files": input_records,
                "inventory_sha256": guard._canonical_sha(input_records),
            },
        )
        trusted = guard.TRUSTED_LOCAL_RUNNERS[guard.F2_CHILD_VISUAL_RUNNER_ID]
        self.runner_path = self.root / trusted["path"]
        self.runner_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(guard.PROJECT_ROOT / trusted["path"], self.runner_path)
        self.driver_path = self.evidence / "driver.py"
        self.driver_path.write_text("# exact fixture driver\n", encoding="utf-8")
        self.environment_path = self.evidence / "environment.json"
        write_json(
            self.environment_path,
            {
                "schema": "celltrack-alpha.f2-child-local-environment.v1",
                "status": "PASS_PREPARED",
                "execution_profile": guard.F2_CHILD_VISUAL_EXECUTION_PROFILE,
                "platform": "linux/amd64",
                "python_version_prefix": "3.12",
                "docker_image_id": guard.F2_CHILD_VISUAL_DOCKER_IMAGE_ID,
                "cpu_only": True,
                "network": "none",
                "docker_memory_mib": guard.F2_CHILD_VISUAL_DOCKER_MEMORY_MIB,
                "maximum_combined_rss_mib": guard.F2_CHILD_VISUAL_MAXIMUM_COMBINED_RSS_MIB,
                "preflight_mode": "stratified_both",
                "frames_per_identity": guard.F2_CHILD_VISUAL_FRAMES_PER_IDENTITY,
                "identities": list(guard.F2_CHILD_VISUAL_IDENTITIES),
                "ffmpeg": {
                    "path": "/fixture/ffmpeg",
                    "sha256": guard.F2_CHILD_VISUAL_FFMPEG_SHA256,
                },
                "docker": {"path": "/fixture/docker", "sha256": "d" * 64},
                "container_driver_path": str(self.driver_path.relative_to(self.root)),
                "container_driver_sha256": guard.sha256(self.driver_path),
                "lmt_sha256": guard.sha256(self.root / "LMT.md"),
            },
        )
        now = datetime.now(timezone.utc).isoformat()
        self.memory_path = self.evidence / "memory.json"
        write_json(
            self.memory_path,
            {
                "schema": "celltrack-alpha.local-memory-headroom.v1",
                "status": "PASS",
                "maximum_combined_rss_mib": guard.F2_CHILD_VISUAL_MAXIMUM_COMBINED_RSS_MIB,
                "authorized_runner_sha256": trusted["sha256"],
                "active_conflicting_jobs": [],
                "available_for_new_run_mib": guard.F2_CHILD_VISUAL_DOCKER_MEMORY_MIB,
                "issued_at": now,
            },
        )
        self.visual_receipt = self.evidence / "LOCAL_VISUAL_PREFLIGHT.json"
        self.execution_receipt = self.evidence / "LOCAL_EXECUTION.json"
        write_json(
            self.visual_receipt,
            {
                "schema": guard.VISUAL_RECEIPT_SCHEMA,
                "status": "PASS",
                "model_execution": True,
            },
        )
        invoked_argv = [
            trusted["path"],
            "--project-root",
            str(self.root.resolve()),
            "run",
            "--memory-headroom-receipt",
            str(self.memory_path.relative_to(self.root)),
        ]
        write_json(
            self.execution_receipt,
            {
                "schema": "celltrack-alpha.local-model-execution.v1",
                "status": "PASS",
                "execution_origin": "local",
                "exit_code": 0,
                "kernel_id": guard.F2_CHILD_VISUAL_KERNEL_ID,
                "model_id": guard.F2_CHILD_VISUAL_MODEL_ID,
                "local_run_id": guard.F2_CHILD_VISUAL_LOCAL_RUN_ID,
                "candidate_code_sha256": guard.sha256(self.child_code),
                "checkpoint_path": guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT,
                "checkpoint_sha256": self.asset_hashes[
                    guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT
                ],
                "input_manifest_path": str(self.input_manifest.relative_to(self.root)),
                "input_manifest_sha256": guard.sha256(self.input_manifest),
                "execution_environment_path": str(
                    self.environment_path.relative_to(self.root)
                ),
                "execution_environment_sha256": guard.sha256(self.environment_path),
                "preregistered_visual_identities": list(
                    guard.F2_CHILD_VISUAL_IDENTITIES
                ),
                "runner_id": guard.F2_CHILD_VISUAL_RUNNER_ID,
                "runner_path": trusted["path"],
                "runner_sha256": trusted["sha256"],
                "invoked_argv": invoked_argv,
                "invoked_argv_sha256": guard._canonical_sha(invoked_argv),
                "started_at": now,
                "ended_at": now,
                "execution_profile": guard.F2_CHILD_VISUAL_EXECUTION_PROFILE,
                "docker_image_id": guard.F2_CHILD_VISUAL_DOCKER_IMAGE_ID,
                "cpu_preflight": True,
                "frames_per_identity": guard.F2_CHILD_VISUAL_FRAMES_PER_IDENTITY,
                "completed_frames_by_identity": {
                    identity: guard.F2_CHILD_VISUAL_FRAMES_PER_IDENTITY
                    for identity in guard.F2_CHILD_VISUAL_IDENTITIES
                },
                "peak_rss_bytes": 1024,
                "labels_or_ground_truth_accessed": False,
                "scorer_accessed": False,
                "competition_test_namespace_accessed": False,
                "submission_absent": True,
                "memory_headroom_receipt_sha256": guard.sha256(self.memory_path),
            },
        )
        self.child_contract_path = self.evidence / "child-visual-contract.json"
        self._reseal_child_contract()

    def _reseal_child_contract(self) -> None:
        write_json(
            self.child_contract_path,
            {
                "schema": guard.F2_CHILD_VISUAL_CONTRACT_SCHEMA,
                "execution_profile": guard.F2_CHILD_VISUAL_EXECUTION_PROFILE,
                "kernel_id": guard.F2_CHILD_VISUAL_KERNEL_ID,
                "model_execution": True,
                "code_sha256": guard.sha256(self.child_code),
                "checkpoint_path": guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT,
                "checkpoint_sha256": self.asset_hashes[
                    guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT
                ],
                "input_manifest_path": str(self.input_manifest.relative_to(self.root)),
                "input_manifest_sha256": guard.sha256(self.input_manifest),
                "execution_environment_path": str(
                    self.environment_path.relative_to(self.root)
                ),
                "execution_environment_sha256": guard.sha256(self.environment_path),
                "preregistered_visual_identities": list(
                    guard.F2_CHILD_VISUAL_IDENTITIES
                ),
                "local_execution_command_sha256": json.loads(
                    self.execution_receipt.read_text(encoding="utf-8")
                )["invoked_argv_sha256"],
                "local_visual_preflight_receipt": {
                    "path": str(self.visual_receipt.relative_to(self.root)),
                    "sha256": guard.sha256(self.visual_receipt),
                },
                "local_execution_receipt": {
                    "path": str(self.execution_receipt.relative_to(self.root)),
                    "sha256": guard.sha256(self.execution_receipt),
                },
            },
        )

    def _outer_source(
        self, *, child_source: str | None = None, candidates: list | None = None
    ) -> str:
        child_source = (
            self.child_code.read_text(encoding="utf-8")
            if child_source is None
            else child_source
        )
        candidates = self.candidates if candidates is None else candidates
        return f"""import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
FAMILY = "D"
CHILD_SOURCE = {child_source!r}
CANDIDATES = {candidates!r}
MOVED_OUTPUTS = ("v32_preilp_cache_v1", "V32_FINAL_GRAPH_ROWS.csv", "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json", "PUBLIC_TEMPORALUNET_STEP0.json", "V32_SHIP_PREFLIGHT_COMPLETION.json", "RUN_LOG.jsonl", "run_stats.csv")
WORKING = Path("/kaggle/working")
CHILD_PATH = WORKING / "child.py"
probe = os.environ.get("V32_RUNTIME_SAFETY_PROBE", "")
preflight = os.environ.get("KAGGLE_PREFLIGHT_MODE", "")
candidate = CANDIDATES[0]
environment = os.environ.copy()
environment.update(candidate["environment"])
environment["F2_SWEEP_CANDIDATE_ID"] = candidate["id"]
if (WORKING / "submission.csv").exists():
    raise RuntimeError("submission present")
completion = {{"submission_absent": not (WORKING / "submission.csv").exists()}}
print("F2_DIRECT_SWEEP_COMPLETE", flush=True)
"""

    def _write_outer(
        self, *, child_source: str | None = None, candidates: list | None = None
    ) -> None:
        self.code = self.kernel / self.expected["code_file"]
        self.code.write_text(
            self._outer_source(child_source=child_source, candidates=candidates),
            encoding="utf-8",
        )
        self.metadata = {
            "code_file": self.expected["code_file"],
            "competition_sources": ["biohub-cell-tracking-during-development"],
            "dataset_sources": [
                "pilkwang/biohub-tracking-support-pack-50ep-v1",
                "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            ],
            "enable_gpu": True,
            "enable_internet": False,
            "enable_tpu": False,
            "id": self.expected["kernel_id"],
            "is_private": True,
            "kernel_type": "script",
            "language": "python",
            "machine_shape": "NvidiaTeslaT4",
            "model_sources": [],
            "title": self.expected["kernel_id"].split("/", 1)[1],
        }
        write_json(self.kernel / "kernel-metadata.json", self.metadata)
        tree = ast.parse(self.code.read_text(encoding="utf-8"))
        embedded = guard._literal_assignment(tree, "CHILD_SOURCE", "child")
        observed_candidates = guard._literal_assignment(
            tree, "CANDIDATES", "candidates"
        )
        freeze = {
            "schema": "celltrack-alpha.f2-direct-de-sweep-source-freeze.v1",
            "status": "PASS_BUILD_ONLY",
            "family": "D",
            "kernel_id": self.expected["kernel_id"],
            "candidate_ids": [item["id"] for item in observed_candidates],
            "frozen_experimental_candidate_ids": [
                item["id"]
                for item in observed_candidates
                if item["id"] != "CONTROL_V32"
            ],
            "candidates": observed_candidates,
            "identities": list(guard.F2_SWEEP_IDENTITIES),
            "same_run_control_id": "CONTROL_V32",
            "evaluate_in_kernel": False,
            "label_free_kernel": True,
            "submission_written": False,
            "child_source_sha256": hashlib.sha256(embedded.encode()).hexdigest(),
            "files": {
                self.expected["code_file"]: guard.sha256(self.code),
                "kernel-metadata.json": guard.sha256(
                    self.kernel / "kernel-metadata.json"
                ),
            },
        }
        write_json(self.kernel / "F2_SWEEP_SOURCE_FREEZE.json", freeze)

    def _profile(self) -> dict:
        first = self.candidates[0]["id"]
        scenarios = []
        for role in ("tiny", "stratified", "stress", "expansive"):
            scenarios.append(
                {
                    "role": role,
                    "env": {
                        "KAGGLE_PREFLIGHT_MODE": role,
                        "KAGGLE_PREFLIGHT_UNITS": "2",
                    },
                    "required_outputs": [
                        {
                            "path": f"{self.expected['output_root']}/{first}/V32_FINAL_GRAPH_ROWS.csv",
                            "kind": "csv",
                        }
                    ],
                    "completion": {
                        "path": "F2_SWEEP_COMPLETION.json",
                        "status_pointer": "/status",
                        "success_values": ["PASS"],
                        "units_pointer": "/completed_units",
                    },
                }
            )
        return {
            "schema": "ship-kaggle-kernels.profile.v1",
            "kernel": {
                "directory": "../kernel",
                "entrypoint": self.expected["code_file"],
                "artifact_sha256": guard.sha256(self.code),
                "metadata_file": "kernel-metadata.json",
                "metadata_sha256": guard.sha256(self.kernel / "kernel-metadata.json"),
                "package_inventory_sha256": guard._ship_package_inventory_sha256(
                    self.kernel
                ),
                "expected_metadata": self.metadata,
                "self_contained": True,
                "upload_transport": "kaggle-cli-save-kernel.v1",
                "execution_control_env": [
                    "KAGGLE_PREFLIGHT_MODE",
                    "KAGGLE_PREFLIGHT_UNITS",
                ],
                "step0": {
                    "assignment_marker": "SHIP_STEP0 = _v32_ship_step0()",
                    "heavy_work_marker": "V32_HEAVY_WORK_BEGIN",
                },
            },
            "scenarios": scenarios,
        }

    def _write_profile_and_certificate(self, *, status: str = "ELIGIBLE") -> None:
        self.profile_path = self.evidence / "profile.json"
        write_json(self.profile_path, self._profile())
        eligible = status == "ELIGIBLE"
        self.certificate_path = self.evidence / "certificate.json"
        write_json(
            self.certificate_path,
            {
                "schema": "ship-kaggle-kernels.certification.v1",
                "status": status,
                "eligible": eligible,
                "autonomous_launch_eligible": eligible,
                "kernel_id": self.expected["kernel_id"],
                "limits_file": str((self.root / "LMT.md").resolve()),
                "limits_sha256": guard.sha256(self.root / "LMT.md"),
                "artifact_sha256": guard.sha256(self.code),
                "metadata_sha256": guard.sha256(self.kernel / "kernel-metadata.json"),
                "package_inventory_sha256": guard._ship_package_inventory_sha256(
                    self.kernel
                ),
                "profile_path": str(self.profile_path.resolve()),
                "profile_sha256": guard.sha256(self.profile_path),
                "quick_debug": {
                    "auto_triggered": True,
                    "name": "focused_pre_push_release_debug",
                    "purpose": "exact script-only fixture",
                    "status": "PASS" if eligible else "FAIL",
                    "transport": "kaggle-cli-save-kernel.v1",
                },
                "static": {"status": "PASS"},
                "repeatability": {"status": "PASS"},
                "runtime_projection": {
                    "status": "PASS",
                    "projected_remote_upper_seconds": 3600.0,
                    "projected_peak_rss_bytes": 1024.0,
                    "physical_memory_limit_bytes": 2048.0,
                },
            },
        )

    def _contract(self) -> dict:
        package = guard._package_inventory(self.kernel, self.root)
        tree = ast.parse(self.code.read_text(encoding="utf-8"))
        child = guard._literal_assignment(tree, "CHILD_SOURCE", "child")
        return {
            "schema": guard.F2_CERTIFIED_SWEEP_CONTRACT_SCHEMA,
            "execution_profile": guard.CERTIFIED_SWEEP_WRAPPER_EXECUTION_PROFILE,
            "family": "D",
            "kernel_id": self.expected["kernel_id"],
            "remote_preflight_kernel_id": self.expected["kernel_id"],
            "lane_owner": self.expected["lane_owner"],
            "model_execution": True,
            "code_sha256": guard.sha256(self.code),
            "metadata_sha256": guard.sha256(self.kernel / "kernel-metadata.json"),
            "package_tree_sha256": package["tree_sha256"],
            "package_inventory_sha256": guard._ship_package_inventory_sha256(
                self.kernel
            ),
            "embedded_child_sha256": hashlib.sha256(child.encode()).hexdigest(),
            "source_freeze": {
                "path": str(
                    (self.kernel / "F2_SWEEP_SOURCE_FREEZE.json").relative_to(self.root)
                ),
                "sha256": guard.sha256(self.kernel / "F2_SWEEP_SOURCE_FREEZE.json"),
            },
            "frozen_candidates_sha256": guard._canonical_sha(self.candidates),
            "preregistered_visual_identities": list(guard.F2_SWEEP_IDENTITIES),
            "same_run_control_id": "CONTROL_V32",
            "submission_output_permitted": False,
            "final_graph_rows_required": True,
            "arbitrary_candidate_or_config_surface_permitted": False,
            "ship_kaggle_certificate": {
                "path": str(self.certificate_path.relative_to(self.root)),
                "sha256": guard.sha256(self.certificate_path),
            },
            "ship_kaggle_profile": {
                "path": str(self.profile_path.relative_to(self.root)),
                "sha256": guard.sha256(self.profile_path),
            },
            "child_visual_evidence": {
                "kernel_dir": str(self.child_kernel.relative_to(self.root)),
                "contract": {
                    "path": str(self.child_contract_path.relative_to(self.root)),
                    "sha256": guard.sha256(self.child_contract_path),
                },
            },
        }

    def _reseal_outer(self) -> None:
        self._write_profile_and_certificate()
        self.contract = self._contract()

    def _run(self) -> dict:
        visual = {
            "status": "PASS",
            "package_tree_sha256": guard._package_inventory(
                self.child_kernel, self.root
            )["tree_sha256"],
            "video_count": 2,
            "identities": list(guard.F2_CHILD_VISUAL_IDENTITIES),
            "artifacts": [{"role": "video", "path": "fixture.mp4", "sha256": "a" * 64}],
        }
        with (
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_PRIMARY_CHECKPOINT_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT],
            ),
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_SECONDARY_CHECKPOINT_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_SECONDARY_CHECKPOINT],
            ),
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_PRIMARY_MANIFEST_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_PRIMARY_MANIFEST],
            ),
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_SECONDARY_MANIFEST_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_SECONDARY_MANIFEST],
            ),
            mock.patch.object(
                guard, "check_local_visual_preflight", return_value=(visual, [])
            ),
        ):
            return guard.check_kernel(
                self.kernel,
                self.contract,
                {},
                "inline",
                expected_initial_kernel_version=1,
                project_root=self.root,
            )

    def test_exact_certified_wrapper_passes(self) -> None:
        result = self._run()
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertTrue(result["eligible"])
        self.assertEqual(result["local_visual_preflight"]["video_count"], 2)

    def test_child_drift_is_rejected_even_with_resealed_outer_certificate(self) -> None:
        changed_child = (
            self.child_code.read_text(encoding="utf-8") + "\nCHILD_DRIFT = True\n"
        )
        self._write_outer(child_source=changed_child)
        self._reseal_outer()
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("visual-executed child bytes", " ".join(result["failures"]))

    def test_candidate_drift_is_rejected_even_with_resealed_outer_certificate(
        self,
    ) -> None:
        changed = [dict(item) for item in self.candidates]
        changed[0] = {
            "id": "D_det0.90",
            "family": "D",
            "delta": {"DET_THRESHOLD": "0.90"},
            "environment": {"F2_DET_THRESHOLD": "0.90"},
        }
        self._write_outer(candidates=changed)
        self._reseal_outer()
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("central frozen family", " ".join(result["failures"]))

    def test_certificate_status_drift_is_rejected(self) -> None:
        self._write_profile_and_certificate(status="REJECTED")
        self.contract = self._contract()
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("certificate drift: status", " ".join(result["failures"]))

    def test_visual_receipt_byte_drift_is_rejected(self) -> None:
        self.visual_receipt.write_text('{"status":"DRIFT"}\n', encoding="utf-8")
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("local visual receipt hash drift", " ".join(result["failures"]))

    def test_pending_visual_receipt_is_rejected_even_when_resealed(self) -> None:
        write_json(
            self.visual_receipt,
            {
                "schema": guard.VISUAL_RECEIPT_SCHEMA,
                "status": "PENDING_REVIEW",
                "model_execution": True,
            },
        )
        self._reseal_child_contract()
        self.contract = self._contract()
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("visual receipt drift: status", " ".join(result["failures"]))

    def test_trusted_runner_byte_drift_is_rejected(self) -> None:
        self.runner_path.write_text(
            self.runner_path.read_text(encoding="utf-8") + "\n# drift\n",
            encoding="utf-8",
        )
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("trusted runner byte drift", " ".join(result["failures"]))

    def test_input_identity_drift_is_rejected_even_when_resealed(self) -> None:
        payload = json.loads(self.input_manifest.read_text(encoding="utf-8"))
        payload["preregistered_visual_identities"] = ["train:wrong"]
        write_json(self.input_manifest, payload)
        self._reseal_child_contract()
        self.contract = self._contract()
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("input manifest drift", " ".join(result["failures"]))

    def test_execution_environment_drift_is_rejected_even_when_resealed(self) -> None:
        payload = json.loads(self.environment_path.read_text(encoding="utf-8"))
        payload["docker_image_id"] = "sha256:" + "0" * 64
        write_json(self.environment_path, payload)
        self._reseal_child_contract()
        self.contract = self._contract()
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("execution environment drift", " ".join(result["failures"]))

    def test_builder_writes_nothing_without_an_eligible_certificate(self) -> None:
        self._write_profile_and_certificate(status="REJECTED")
        registry = self.root / "registry.json"
        write_json(registry, {})
        output = self.root / "SHOULD_NOT_EXIST.json"
        argv = [
            "build_f2_certified_sweep_contract.py",
            "--family",
            "D",
            "--kernel-dir",
            str(self.kernel),
            "--certificate",
            str(self.certificate_path),
            "--profile",
            str(self.profile_path),
            "--child-kernel-dir",
            str(self.child_kernel),
            "--child-visual-contract",
            str(self.child_contract_path),
            "--registry",
            str(registry),
            "--project-root",
            str(self.root),
            "--output",
            str(output),
        ]
        visual = {
            "status": "PASS",
            "package_tree_sha256": guard._package_inventory(
                self.child_kernel, self.root
            )["tree_sha256"],
            "video_count": 2,
            "identities": list(guard.F2_CHILD_VISUAL_IDENTITIES),
            "artifacts": [],
        }
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                guard, "check_local_visual_preflight", return_value=(visual, [])
            ),
        ):
            self.assertEqual(contract_builder.main(), 2)
        self.assertFalse(output.exists())

    def test_builder_writes_nothing_before_exact_visual_pass(self) -> None:
        write_json(
            self.visual_receipt,
            {
                "schema": guard.VISUAL_RECEIPT_SCHEMA,
                "status": "PENDING_REVIEW",
                "model_execution": True,
            },
        )
        self._reseal_child_contract()
        registry = self.root / "registry.json"
        write_json(registry, {})
        output = self.root / "SHOULD_NOT_EXIST_PENDING_VISUAL.json"
        argv = [
            "build_f2_certified_sweep_contract.py",
            "--family",
            "D",
            "--kernel-dir",
            str(self.kernel),
            "--certificate",
            str(self.certificate_path),
            "--profile",
            str(self.profile_path),
            "--child-kernel-dir",
            str(self.child_kernel),
            "--child-visual-contract",
            str(self.child_contract_path),
            "--registry",
            str(registry),
            "--project-root",
            str(self.root),
            "--output",
            str(output),
        ]
        visual = {
            "status": "PASS",
            "package_tree_sha256": guard._package_inventory(
                self.child_kernel, self.root
            )["tree_sha256"],
            "video_count": 2,
            "identities": list(guard.F2_CHILD_VISUAL_IDENTITIES),
            "artifacts": [],
        }
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_PRIMARY_CHECKPOINT_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_PRIMARY_CHECKPOINT],
            ),
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_SECONDARY_CHECKPOINT_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_SECONDARY_CHECKPOINT],
            ),
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_PRIMARY_MANIFEST_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_PRIMARY_MANIFEST],
            ),
            mock.patch.object(
                guard,
                "F2_CHILD_VISUAL_SECONDARY_MANIFEST_SHA256",
                self.asset_hashes[guard.F2_CHILD_VISUAL_SECONDARY_MANIFEST],
            ),
            mock.patch.object(
                guard, "check_local_visual_preflight", return_value=(visual, [])
            ),
        ):
            self.assertEqual(contract_builder.main(), 2)
        self.assertFalse(output.exists())

    def test_test_namespace_visit_before_firewall_is_rejected(self) -> None:
        child = self.child_code.read_text(encoding="utf-8").replace(
            "TEST_DIR = SELECTED_TRAIN\n",
            "print(TEST_DIR)\nTEST_DIR = SELECTED_TRAIN\n",
        )
        self._write_outer(child_source=child)
        self._reseal_outer()
        result = self._run()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("before selected-train firewall", " ".join(result["failures"]))


class F2DV6NativeProductionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bindings = guard.F2_D_V6_NATIVE_BINDINGS
        self._copy_exact_repository_files()
        self.kernel = (self.root / self.bindings["code_relative"]).parent
        self.code = self.root / self.bindings["code_relative"]
        self.metadata = self.kernel / "kernel-metadata.json"
        self.profile = self.root / self.bindings["profile_relative"]
        self.evidence = self.root / "downloaded_native_visual"
        self.evidence.mkdir()
        self._write_remote_evidence()
        self._write_local_decode_and_review()
        self._seal_production()
        self._write_download_zip()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _copy(self, relative: str) -> Path:
        source = guard.PROJECT_ROOT / relative
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    def _copy_exact_repository_files(self) -> None:
        kernel_relative = Path(self.bindings["code_relative"]).parent
        for name in (
            "f2_direct_d_sweep_kernel.py",
            "kernel-metadata.json",
            "F2_SWEEP_SOURCE_FREEZE.json",
        ):
            self._copy((kernel_relative / name).as_posix())
        for relative in (
            self.bindings["profile_relative"],
            guard.F2_D_V6_NATIVE_REQUIREMENTS["path"],
            guard.F2_D_V6_NATIVE_SEALER["path"],
            guard.TRUSTED_REMOTE_VISUAL_RUNNERS[
                "f2-d-v6-native-t4-visual-v1"
            ]["path"],
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v6_native_interactive_cert/interactive_native_certify.py"
            ),
            (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_d_v6_native_interactive_cert/INTERACTIVE_CERTIFICATION_AUDIT.json"
            ),
            "LMT.md",
        ):
            self._copy(relative)

    def _ref(self, path: Path) -> dict[str, str]:
        return {
            "path": path.resolve().relative_to(self.root.resolve()).as_posix(),
            "sha256": guard.sha256(path),
        }

    @staticmethod
    def _native_certifier_bindings(bindings: dict) -> dict:
        return {
            "certifier_sha256": (
                "49d8ea390e67c0aa8a084f119e143ce878197119de81c4e72202ffdbb83d955f"
            ),
            "kernel_artifact_sha256": bindings["code_sha256"],
            "kernel_id": bindings["kernel_id"],
            "kernel_metadata_sha256": bindings["metadata_sha256"],
            "kernel_package_inventory_sha256": bindings[
                "package_inventory_sha256"
            ],
            "lmt_sha256": bindings["lmt_sha256"],
            "profile_sha256": bindings["profile_sha256"],
            "python_version_prefix": "3.12",
        }

    def _write_remote_evidence(self) -> None:
        self.host = {
            "platform": "Linux-fixture",
            "python": "3.12.11",
            "physical_gpus": ["Tesla T4, 7.5", "Tesla T4, 7.5"],
            "application_cuda_visible_devices": "0",
        }
        self.certificate = (
            self.evidence / "F2_DIRECT_D_SWEEP_V6_NATIVE_CERTIFICATION.json"
        )
        write_json(
            self.certificate,
            {
                "schema": "ship-kaggle-kernels.certification.v1",
                "status": "ELIGIBLE",
                "eligible": True,
                "autonomous_launch_eligible": True,
                "kernel_id": self.bindings["kernel_id"],
                "profile_sha256": self.bindings["profile_sha256"],
                "limits_sha256": self.bindings["lmt_sha256"],
                "artifact_sha256": self.bindings["code_sha256"],
                "metadata_sha256": self.bindings["metadata_sha256"],
                "package_inventory_sha256": self.bindings[
                    "package_inventory_sha256"
                ],
                "limits_file": "/workspace/LMT.md",
                "profile_path": "/workspace/" + self.bindings["profile_relative"],
            },
        )
        self.native_result = self.evidence / "NATIVE_INTERACTIVE_RESULT.json"
        write_json(
            self.native_result,
            {
                "schema": "celltrack-alpha.f2-d-v6-native-interactive-result.v1",
                "status": "PASS_EXACT_ELIGIBLE",
                "no_push_performed": True,
                "exact_bindings": self._native_certifier_bindings(self.bindings),
                "runtime_bundle_sha256": self.bindings["native_bundle_sha256"],
                "host": self.host,
                "transport_execution_envelope": guard.F2_D_V6_NATIVE_TRANSPORT_ENVELOPE,
                "exit_code": 0,
                "binding_failures": [],
                "certification_receipt": {
                    "path": "/workspace/output/" + self.certificate.name,
                    "sha256": guard.sha256(self.certificate),
                },
            },
        )
        child_outputs = self.evidence / "child_outputs"
        child_outputs.mkdir()
        self.graph = child_outputs / "V32_FINAL_GRAPH_ROWS.csv"
        self.graph.write_text("frame,node_id,parent_id\n0,1,\n", encoding="utf-8")
        self.completion = child_outputs / "V32_SHIP_PREFLIGHT_COMPLETION.json"
        write_json(self.completion, {"schema": "fixture", "status": "PASS"})
        self.step0 = child_outputs / "PUBLIC_TEMPORALUNET_STEP0.json"
        write_json(self.step0, {"schema": "fixture", "status": "PASS"})
        videos = self.evidence / "videos"
        videos.mkdir()
        self.video_by_identity: dict[str, Path] = {}
        execution_videos = []
        for index, identity in enumerate(guard.F2_D_V6_NATIVE_VISUAL_IDENTITIES):
            video = videos / f"identity_{index}.mp4"
            video.write_bytes(b"fixture-mp4-" + identity.encode("utf-8"))
            self.video_by_identity[identity] = video
            execution_videos.append(
                {
                    "identity": identity,
                    "video_path": video.relative_to(self.evidence).as_posix(),
                    "video_sha256": guard.sha256(video),
                    "remote_full_decode": {
                        "decoder": "ffmpeg:h264",
                        "decoded_frames": 6,
                        "eof_reached": True,
                    },
                }
            )
        self.execution = self.evidence / "NATIVE_T4_VISUAL_EXECUTION.json"
        write_json(
            self.execution,
            {
                "schema": "celltrack-alpha.f2-d-v6-native-t4-visual-execution.v1",
                "status": "PASS_PENDING_DOWNLOADED_REVIEW",
                "execution_profile": guard.F2_D_V6_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
                "model_execution": True,
                "bindings": self.bindings,
                "native_certificate_path": self.certificate.name,
                "native_certificate_sha256": guard.sha256(self.certificate),
                "native_result_path": self.native_result.name,
                "native_result_sha256": guard.sha256(self.native_result),
                "host": self.host,
                "transport_execution_envelope": guard.F2_D_V6_NATIVE_TRANSPORT_ENVELOPE,
                "application_cuda_visible_devices": "0",
                "started_at": "2026-08-22T01:00:00+00:00",
                "ended_at": "2026-08-22T01:10:00+00:00",
                "exit_code": 0,
                "completed_frames_by_identity": {
                    identity: 6
                    for identity in guard.F2_D_V6_NATIVE_VISUAL_IDENTITIES
                },
                "final_graph_rows_path": self.graph.relative_to(
                    self.evidence
                ).as_posix(),
                "final_graph_rows_sha256": guard.sha256(self.graph),
                "child_completion_sha256": guard.sha256(self.completion),
                "child_step0_sha256": guard.sha256(self.step0),
                "videos": execution_videos,
                "labels_or_ground_truth_accessed": False,
                "scorer_accessed": False,
                "competition_test_namespace_accessed": False,
                "submission_absent": True,
            },
        )

    def _write_local_decode_and_review(self) -> None:
        execution = json.loads(self.execution.read_text(encoding="utf-8"))
        self.decode = self.root / "DOWNLOADED_FULL_DECODE.json"
        write_json(
            self.decode,
            {
                "schema": "celltrack-alpha.f2-d-v6-downloaded-mp4-full-decode.v1",
                "status": "PASS",
                "frames_per_identity": 6,
                "all_eof_reached": True,
                "videos": [
                    {
                        "identity": row["identity"],
                        "downloaded_video_path": self.video_by_identity[
                            row["identity"]
                        ].relative_to(self.root).as_posix(),
                        "video_sha256": row["video_sha256"],
                        "remote_full_decode": row["remote_full_decode"],
                        "local_full_decode": {
                            "decoder": "ffmpeg:h264",
                            "decoded_frames": 6,
                            "eof_reached": True,
                        },
                    }
                    for row in execution["videos"]
                ],
            },
        )
        self.review = self.root / "HUMAN_VISUAL_REVIEW.json"
        write_json(
            self.review,
            {
                "schema": "celltrack-alpha.f2-d-v6-native-visual-human-review.v1",
                "status": "PASS",
                "reviews": [
                    {
                        "identity": identity,
                        "video_sha256": guard.sha256(video),
                        "visual_analysis_performed": True,
                        "metrics_only": False,
                        "verdict": "PASS",
                        "reviewer": "fixture-human",
                        "promotion_rationale": "No promotion-blocking visual failure.",
                        "observations": ["Six frames reviewed through EOF."],
                        "failure_taxonomy": {
                            name: "none observed"
                            for name in guard.VISUAL_FAILURE_TAXONOMY
                        },
                        "reviewed_at": "2026-08-22T01:20:00+00:00",
                    }
                    for identity, video in self.video_by_identity.items()
                ],
            },
        )

    def _seal_production(self) -> None:
        runner = guard.TRUSTED_REMOTE_VISUAL_RUNNERS[
            "f2-d-v6-native-t4-visual-v1"
        ]
        reviews = json.loads(self.review.read_text(encoding="utf-8"))["reviews"]
        self.production = self.root / "F2_D_V6_PRODUCTION_CONTRACT.json"
        write_json(
            self.production,
            {
                "schema": guard.F2_D_V6_NATIVE_PRODUCTION_CONTRACT_SCHEMA,
                "status": "PASS",
                "execution_profile": guard.F2_D_V6_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
                "kernel_id": self.bindings["kernel_id"],
                "model_execution": True,
                "exact_bindings": self.bindings,
                "trusted_runner": {
                    "id": "f2-d-v6-native-t4-visual-v1",
                    "path": runner["path"],
                    "sha256": runner["sha256"],
                },
                "native_ship_certificate": self._ref(self.certificate),
                "native_t4_visual_execution": self._ref(self.execution),
                "downloaded_full_decode": self._ref(self.decode),
                "human_visual_review": self._ref(self.review),
                "reviewed_video_sha256_by_identity": {
                    identity: guard.sha256(video)
                    for identity, video in self.video_by_identity.items()
                },
                "human_reviews_sha256": guard._canonical_sha(reviews),
                "science_and_ilp_bytes_unchanged": True,
                "final_graph_rows_required": True,
                "submission_output_permitted": False,
                "labels_ground_truth_scorer_test_training_tuning_permitted": False,
            },
        )

    def _write_download_zip(self, *, extra_member: bool = False) -> None:
        members = [
            self.execution,
            self.certificate,
            self.native_result,
            self.graph,
            self.completion,
            self.step0,
            *self.video_by_identity.values(),
        ]
        records = []
        for path in members:
            records.append(
                {
                    "path": path.relative_to(self.evidence).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": guard.sha256(path),
                }
            )
        manifest = {
            "schema": "celltrack-alpha.f2-d-v6-native-visual-download.v1",
            "files": records,
        }
        self.archive = self.root / "f2_d_v6_native_visual.zip"
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "DOWNLOAD_MANIFEST.json", json.dumps(manifest, sort_keys=True)
            )
            for path in members:
                archive.write(path, path.relative_to(self.evidence).as_posix())
            if extra_member:
                archive.writestr("unmanifested.txt", b"drift")

    def _validate(self) -> tuple[dict, list[dict]]:
        return guard.check_f2_d_v6_native_production_contract(
            production_contract_reference=self._ref(self.production),
            downloaded_zip_reference=self._ref(self.archive),
            kernel_dir=self.kernel,
            code_path=self.code,
            metadata_path=self.metadata,
            embedded_child_sha256=self.bindings["embedded_child_sha256"],
            project_root=self.root,
        )

    def test_exact_native_t4_downloaded_human_pass_is_accepted(self) -> None:
        summary, artifacts = self._validate()
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["video_count"], 2)
        self.assertEqual(len(artifacts), 6)

    def test_native_v6_wrapper_dispatches_to_the_central_validator(self) -> None:
        contract = contract_builder.build_contract(
            family="D",
            kernel_dir=self.kernel,
            certificate=self.certificate,
            profile=self.profile,
            child_kernel_dir=None,
            child_visual_contract=None,
            project_root=self.root,
            native_t4_visual_contract=self.production,
            downloaded_visual_zip=self.archive,
        )
        metadata = guard.load_json(self.metadata)
        source = self.code.read_text(encoding="utf-8")
        ship = {"required": True, "status": "PASS", "eligible": True}
        with mock.patch.object(
            guard,
            "_validate_f2_sweep_ship_certificate",
            return_value=(ship, []),
        ):
            result = guard.check_certified_sweep_wrapper(
                kernel_dir=self.kernel,
                metadata_path=self.metadata,
                metadata=metadata,
                code_path=self.code,
                source=source,
                tree=ast.parse(source),
                contract=contract,
                registry={},
                phase="inline",
                expected_initial_kernel_version=1,
                project_root=self.root,
                contract_path=None,
                registry_path=None,
            )
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertEqual(
            result["local_visual_preflight"]["execution_profile"],
            guard.F2_D_V6_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
        )

    def test_native_certificate_drift_is_rejected_after_resealing(self) -> None:
        certificate = json.loads(self.certificate.read_text(encoding="utf-8"))
        certificate["status"] = "REJECTED"
        write_json(self.certificate, certificate)
        result = json.loads(self.native_result.read_text(encoding="utf-8"))
        result["certification_receipt"]["sha256"] = guard.sha256(self.certificate)
        write_json(self.native_result, result)
        execution = json.loads(self.execution.read_text(encoding="utf-8"))
        execution["native_certificate_sha256"] = guard.sha256(self.certificate)
        execution["native_result_sha256"] = guard.sha256(self.native_result)
        write_json(self.execution, execution)
        self._seal_production()
        self._write_download_zip()
        with self.assertRaisesRegex(guard.GuardError, "certificate drift: status"):
            self._validate()

    def test_download_zip_member_drift_is_rejected(self) -> None:
        self._write_download_zip(extra_member=True)
        with self.assertRaisesRegex(guard.GuardError, "manifest/member set drift"):
            self._validate()

    def test_local_decode_eof_drift_is_rejected_after_resealing(self) -> None:
        decode = json.loads(self.decode.read_text(encoding="utf-8"))
        decode["videos"][0]["local_full_decode"]["eof_reached"] = False
        write_json(self.decode, decode)
        self._seal_production()
        with self.assertRaisesRegex(guard.GuardError, "local full-decode drift"):
            self._validate()

    def test_human_review_hash_drift_is_rejected_after_resealing(self) -> None:
        review = json.loads(self.review.read_text(encoding="utf-8"))
        review["reviews"][0]["video_sha256"] = "0" * 64
        write_json(self.review, review)
        self._seal_production()
        with self.assertRaisesRegex(guard.GuardError, "hash-bound PASS"):
            self._validate()

    def test_transport_envelope_drift_is_rejected_after_resealing(self) -> None:
        execution = json.loads(self.execution.read_text(encoding="utf-8"))
        execution["application_cuda_visible_devices"] = "0,1"
        write_json(self.execution, execution)
        self._seal_production()
        self._write_download_zip()
        with self.assertRaisesRegex(guard.GuardError, "execution drift"):
            self._validate()

    def test_trusted_remote_runner_byte_drift_is_rejected(self) -> None:
        runner = self.root / guard.TRUSTED_REMOTE_VISUAL_RUNNERS[
            "f2-d-v6-native-t4-visual-v1"
        ]["path"]
        runner.write_text(
            runner.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(guard.GuardError, "exact file hash drift"):
            self._validate()

    def test_builder_writes_nothing_for_incomplete_native_route(self) -> None:
        registry = self.root / "registry.json"
        write_json(registry, {})
        output = self.root / "MUST_NOT_EXIST.json"
        argv = [
            "build_f2_certified_sweep_contract.py",
            "--family",
            "D",
            "--kernel-dir",
            str(self.kernel),
            "--certificate",
            str(self.certificate),
            "--profile",
            str(self.profile),
            "--native-t4-visual-contract",
            str(self.production),
            "--registry",
            str(registry),
            "--project-root",
            str(self.root),
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(contract_builder.main(), 2)
        self.assertFalse(output.exists())


class F2DV7TerminalRejectedContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bindings = guard.F2_D_V7_NATIVE_BINDINGS
        kernel_relative = Path(self.bindings["code_relative"]).parent
        for relative in (
            kernel_relative / "f2_direct_d_sweep_kernel.py",
            kernel_relative / "kernel-metadata.json",
            kernel_relative / "F2_SWEEP_SOURCE_FREEZE.json",
            Path(self.bindings["profile_relative"]),
            Path("LMT.md"),
            Path(
                guard.F2_D_NATIVE_VISUAL_ROUTES["v7"][
                    "terminal_rejected_certificate"
                ]["path"]
            ),
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(guard.PROJECT_ROOT / relative, target)
        self.kernel = self.root / kernel_relative
        self.code = self.root / self.bindings["code_relative"]
        self.metadata = self.kernel / "kernel-metadata.json"
        self.profile = self.root / self.bindings["profile_relative"]
        self.certificate = self.root / "candidate_certificate.json"
        self.production = self.root / "candidate_production.json"
        self.archive = self.root / "candidate_visual.zip"
        write_json(self.certificate, {"status": "ELIGIBLE"})
        write_json(self.production, {"status": "PASS"})
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("placeholder", b"blocked")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _ref(self, path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": guard.sha256(path),
        }

    def test_v7_validator_is_hard_blocked_by_exact_terminal_rejection(self) -> None:
        with self.assertRaisesRegex(
            guard.GuardError, "terminal REJECTED; production route is disabled"
        ):
            guard.check_f2_d_v7_native_production_contract(
                production_contract_reference=self._ref(self.production),
                downloaded_zip_reference=self._ref(self.archive),
                kernel_dir=self.kernel,
                code_path=self.code,
                metadata_path=self.metadata,
                embedded_child_sha256=self.bindings["embedded_child_sha256"],
                project_root=self.root,
            )

    def test_v7_terminal_certificate_drift_fails_closed(self) -> None:
        terminal = self.root / guard.F2_D_NATIVE_VISUAL_ROUTES["v7"][
            "terminal_rejected_certificate"
        ]["path"]
        terminal.write_text(terminal.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(guard.GuardError, "certificate binding drift"):
            guard.check_f2_d_v7_native_production_contract(
                production_contract_reference=self._ref(self.production),
                downloaded_zip_reference=self._ref(self.archive),
                kernel_dir=self.kernel,
                code_path=self.code,
                metadata_path=self.metadata,
                embedded_child_sha256=self.bindings["embedded_child_sha256"],
                project_root=self.root,
            )

    def test_v7_builder_selects_additive_profile_but_writes_no_contract(self) -> None:
        contract = contract_builder.build_contract(
            family="D",
            kernel_dir=self.kernel,
            certificate=self.certificate,
            profile=self.profile,
            child_kernel_dir=None,
            child_visual_contract=None,
            project_root=self.root,
            native_t4_visual_contract=self.production,
            downloaded_visual_zip=self.archive,
        )
        self.assertEqual(
            contract["visual_evidence_profile"],
            guard.F2_D_V7_NATIVE_T4_VISUAL_EXECUTION_PROFILE,
        )
        registry = self.root / "registry.json"
        output = self.root / "MUST_NOT_EXIST_D_V7.json"
        write_json(registry, {})
        argv = [
            "build_f2_certified_sweep_contract.py",
            "--family",
            "D",
            "--kernel-dir",
            str(self.kernel),
            "--certificate",
            str(self.certificate),
            "--profile",
            str(self.profile),
            "--native-t4-visual-contract",
            str(self.production),
            "--downloaded-visual-zip",
            str(self.archive),
            "--registry",
            str(registry),
            "--project-root",
            str(self.root),
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(contract_builder.main(), 2)
        self.assertFalse(output.exists())


class F2DV8TerminalRejectedContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bindings = guard.F2_D_V8_NATIVE_BINDINGS
        kernel_relative = Path(self.bindings["code_relative"]).parent
        for relative in (
            kernel_relative / "f2_direct_d_sweep_kernel.py",
            kernel_relative / "kernel-metadata.json",
            kernel_relative / "F2_SWEEP_SOURCE_FREEZE.json",
            Path(self.bindings["profile_relative"]),
            Path("LMT.md"),
            Path(
                guard.F2_D_NATIVE_VISUAL_ROUTES["v8"][
                    "terminal_rejected_certificate"
                ]["path"]
            ),
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(guard.PROJECT_ROOT / relative, target)
        self.kernel = self.root / kernel_relative
        self.code = self.root / self.bindings["code_relative"]
        self.metadata = self.kernel / "kernel-metadata.json"
        self.profile = self.root / self.bindings["profile_relative"]
        self.certificate = self.root / "candidate_certificate.json"
        self.production = self.root / "candidate_production.json"
        self.archive = self.root / "candidate_visual.zip"
        write_json(self.certificate, {"status": "ELIGIBLE"})
        write_json(self.production, {"status": "PASS"})
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("placeholder", b"blocked")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _ref(self, path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": guard.sha256(path),
        }

    def test_v8_route_is_exactly_bound_to_terminal_rejection(self) -> None:
        route = guard.F2_D_NATIVE_VISUAL_ROUTES["v8"]
        self.assertFalse(route["production_enabled"])
        self.assertEqual(
            route["terminal_rejected_certificate"]["sha256"],
            "7b48ca347954487d2e6991e3bf5872d86f951e58773d61fb1b4990ee7fd1496a",
        )
        self.assertEqual(
            guard.TRUSTED_REMOTE_VISUAL_RUNNERS[route["runner_id"]]["sha256"],
            "6c93241708307743738ed908c0cbe5df10fa479b9778b8d1f505f217fd62a69e",
        )
        self.assertEqual(
            self.bindings["native_bundle_sha256"],
            "9340def365eb4a03851560c0c1436f7ad57d1e29870e1ed1d909287066f8d7b3",
        )
        self.assertEqual(
            self.bindings["native_transport_sha256"],
            "55cf867b3e96751aeb3bbf069c7d4a8149c7ec3e42f84a38852f031184dbad3a",
        )

    def test_v8_validator_fails_before_accepting_candidate_evidence(self) -> None:
        with self.assertRaisesRegex(
            guard.GuardError,
            "terminal REJECTED; production route is disabled",
        ):
            guard.check_f2_d_v8_native_production_contract(
                production_contract_reference=self._ref(self.production),
                downloaded_zip_reference=self._ref(self.archive),
                kernel_dir=self.kernel,
                code_path=self.code,
                metadata_path=self.metadata,
                embedded_child_sha256=self.bindings["embedded_child_sha256"],
                project_root=self.root,
            )

    def test_v8_terminal_certificate_drift_fails_closed(self) -> None:
        terminal = self.root / guard.F2_D_NATIVE_VISUAL_ROUTES["v8"][
            "terminal_rejected_certificate"
        ]["path"]
        terminal.write_text(terminal.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(guard.GuardError, "certificate binding drift"):
            guard.check_f2_d_v8_native_production_contract(
                production_contract_reference=self._ref(self.production),
                downloaded_zip_reference=self._ref(self.archive),
                kernel_dir=self.kernel,
                code_path=self.code,
                metadata_path=self.metadata,
                embedded_child_sha256=self.bindings["embedded_child_sha256"],
                project_root=self.root,
            )

    def test_v8_builder_hard_fails_and_writes_no_contract(self) -> None:
        with self.assertRaisesRegex(
            guard.GuardError, "terminal REJECTED; production route is disabled"
        ):
            contract_builder.build_contract(
                family="D",
                kernel_dir=self.kernel,
                certificate=self.certificate,
                profile=self.profile,
                child_kernel_dir=None,
                child_visual_contract=None,
                project_root=self.root,
                native_t4_visual_contract=self.production,
                downloaded_visual_zip=self.archive,
            )
        registry = self.root / "registry.json"
        output = self.root / "MUST_NOT_EXIST_D_V8.json"
        write_json(registry, {})
        argv = [
            "build_f2_certified_sweep_contract.py",
            "--family",
            "D",
            "--kernel-dir",
            str(self.kernel),
            "--certificate",
            str(self.certificate),
            "--profile",
            str(self.profile),
            "--native-t4-visual-contract",
            str(self.production),
            "--downloaded-visual-zip",
            str(self.archive),
            "--registry",
            str(registry),
            "--project-root",
            str(self.root),
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(contract_builder.main(), 2)
        self.assertFalse(output.exists())


class F2DV9TerminalRejectedContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bindings = guard.F2_D_V9_NATIVE_BINDINGS
        kernel_relative = Path(self.bindings["code_relative"]).parent
        for relative in (
            kernel_relative / "f2_direct_d_sweep_kernel.py",
            kernel_relative / "kernel-metadata.json",
            kernel_relative / "F2_SWEEP_SOURCE_FREEZE.json",
            Path(self.bindings["profile_relative"]),
            Path("LMT.md"),
            Path(
                guard.F2_D_NATIVE_VISUAL_ROUTES["v9"][
                    "terminal_rejected_certificate"
                ]["path"]
            ),
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(guard.PROJECT_ROOT / relative, target)
        self.kernel = self.root / kernel_relative
        self.code = self.root / self.bindings["code_relative"]
        self.metadata = self.kernel / "kernel-metadata.json"
        self.profile = self.root / self.bindings["profile_relative"]
        self.certificate = self.root / "candidate_certificate.json"
        self.production = self.root / "candidate_production.json"
        self.archive = self.root / "candidate_visual.zip"
        write_json(self.certificate, {"status": "ELIGIBLE"})
        write_json(self.production, {"status": "PASS"})
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("placeholder", b"blocked")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _ref(self, path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": guard.sha256(path),
        }

    def test_v9_route_is_exactly_bound_to_terminal_rejection(self) -> None:
        route = guard.F2_D_NATIVE_VISUAL_ROUTES["v9"]
        self.assertFalse(route["production_enabled"])
        self.assertEqual(
            route["terminal_rejected_certificate"]["sha256"],
            "5370c55e18d0ec25ec673258e44a9b8ef40e15ff75123c53ecd10362bcfc174c",
        )
        self.assertEqual(
            guard.TRUSTED_REMOTE_VISUAL_RUNNERS[route["runner_id"]]["sha256"],
            "38cf673fa739717ceef61c49d10db66167cad41b2fa360913f0c11ea3e45a143",
        )
        self.assertEqual(
            self.bindings["native_bundle_sha256"],
            "c48b52c5ba3af07e98f7827c0af5c865155d6cd3b9dd46804459737e325602e9",
        )
        self.assertEqual(
            self.bindings["full_support_install_plan_sha256"],
            "44d759547070cca311a89b3676e3a949045f6b242443f2263d39b8593c3553d1",
        )

    def test_v9_validator_fails_before_accepting_candidate_evidence(self) -> None:
        with self.assertRaisesRegex(
            guard.GuardError,
            "terminal REJECTED; production route is disabled",
        ):
            guard.check_f2_d_v9_native_production_contract(
                production_contract_reference=self._ref(self.production),
                downloaded_zip_reference=self._ref(self.archive),
                kernel_dir=self.kernel,
                code_path=self.code,
                metadata_path=self.metadata,
                embedded_child_sha256=self.bindings["embedded_child_sha256"],
                project_root=self.root,
            )

    def test_v9_terminal_certificate_drift_fails_closed(self) -> None:
        terminal = self.root / guard.F2_D_NATIVE_VISUAL_ROUTES["v9"][
            "terminal_rejected_certificate"
        ]["path"]
        terminal.write_text(terminal.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(guard.GuardError, "certificate binding drift"):
            guard.check_f2_d_v9_native_production_contract(
                production_contract_reference=self._ref(self.production),
                downloaded_zip_reference=self._ref(self.archive),
                kernel_dir=self.kernel,
                code_path=self.code,
                metadata_path=self.metadata,
                embedded_child_sha256=self.bindings["embedded_child_sha256"],
                project_root=self.root,
            )

    def test_v9_builder_hard_fails_and_writes_no_contract(self) -> None:
        with self.assertRaisesRegex(
            guard.GuardError,
            "terminal REJECTED; production route is disabled",
        ):
            contract_builder.build_contract(
                family="D",
                kernel_dir=self.kernel,
                certificate=self.certificate,
                profile=self.profile,
                child_kernel_dir=None,
                child_visual_contract=None,
                project_root=self.root,
                native_t4_visual_contract=self.production,
                downloaded_visual_zip=self.archive,
            )
        registry = self.root / "registry.json"
        output = self.root / "MUST_NOT_EXIST_D_V9.json"
        write_json(registry, {})
        argv = [
            "build_f2_certified_sweep_contract.py",
            "--family",
            "D",
            "--kernel-dir",
            str(self.kernel),
            "--certificate",
            str(self.certificate),
            "--profile",
            str(self.profile),
            "--native-t4-visual-contract",
            str(self.production),
            "--downloaded-visual-zip",
            str(self.archive),
            "--registry",
            str(registry),
            "--project-root",
            str(self.root),
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(contract_builder.main(), 2)
        self.assertFalse(output.exists())


class F2DV10PendingNativeCertificateContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bindings = guard.F2_D_V10_NATIVE_BINDINGS
        kernel_relative = Path(self.bindings["code_relative"]).parent
        for relative in (
            kernel_relative / "f2_direct_d_sweep_kernel.py",
            kernel_relative / "kernel-metadata.json",
            kernel_relative / "F2_SWEEP_SOURCE_FREEZE.json",
            Path(self.bindings["profile_relative"]),
            Path("LMT.md"),
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(guard.PROJECT_ROOT / relative, target)
        self.kernel = self.root / kernel_relative
        self.code = self.root / self.bindings["code_relative"]
        self.metadata = self.kernel / "kernel-metadata.json"
        self.profile = self.root / self.bindings["profile_relative"]
        self.certificate = self.root / "candidate_certificate.json"
        self.production = self.root / "candidate_production.json"
        self.archive = self.root / "candidate_visual.zip"
        write_json(self.certificate, {"status": "ELIGIBLE"})
        write_json(self.production, {"status": "PASS"})
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("placeholder", b"blocked")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _ref(self, path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": guard.sha256(path),
        }

    def test_v10_route_is_exactly_bound_but_pending_certificate(self) -> None:
        route = guard.F2_D_NATIVE_VISUAL_ROUTES["v10"]
        self.assertFalse(route["production_enabled"])
        self.assertEqual(
            route["block_reason"], "pending_exact_native_eligible_certificate"
        )
        self.assertEqual(
            guard.TRUSTED_REMOTE_VISUAL_RUNNERS[route["runner_id"]]["sha256"],
            "482e29123979cbcc18ff52cbf2ef3533c63e887516bd1a692becda34e03f6312",
        )
        self.assertEqual(
            self.bindings["certifier_sha256"],
            "5529b376752a68a80f827018b3796807a206ed722afaca5c400a56e118ccffc0",
        )
        self.assertEqual(
            self.bindings["runtime_projection_patch_sha256"],
            "3fef87f23dc4a31a5275b3a928d2818cf26e1997cc37c0d1226be7a1d63c2226",
        )

    def test_v10_validator_fails_before_accepting_candidate_evidence(self) -> None:
        with self.assertRaisesRegex(
            guard.GuardError,
            "disabled pending a downloaded exact ELIGIBLE native certificate",
        ):
            guard.check_f2_d_v10_native_production_contract(
                production_contract_reference=self._ref(self.production),
                downloaded_zip_reference=self._ref(self.archive),
                kernel_dir=self.kernel,
                code_path=self.code,
                metadata_path=self.metadata,
                embedded_child_sha256=self.bindings["embedded_child_sha256"],
                project_root=self.root,
            )

    def test_v10_builder_hard_fails_and_writes_no_contract(self) -> None:
        with self.assertRaisesRegex(
            guard.GuardError,
            "disabled pending a downloaded exact ELIGIBLE native certificate",
        ):
            contract_builder.build_contract(
                family="D",
                kernel_dir=self.kernel,
                certificate=self.certificate,
                profile=self.profile,
                child_kernel_dir=None,
                child_visual_contract=None,
                project_root=self.root,
                native_t4_visual_contract=self.production,
                downloaded_visual_zip=self.archive,
            )
        registry = self.root / "registry.json"
        output = self.root / "MUST_NOT_EXIST_D_V10.json"
        write_json(registry, {})
        argv = [
            "build_f2_certified_sweep_contract.py",
            "--family",
            "D",
            "--kernel-dir",
            str(self.kernel),
            "--certificate",
            str(self.certificate),
            "--profile",
            str(self.profile),
            "--native-t4-visual-contract",
            str(self.production),
            "--downloaded-visual-zip",
            str(self.archive),
            "--registry",
            str(registry),
            "--project-root",
            str(self.root),
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(contract_builder.main(), 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
