from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import kaggle_prelaunch_guard as guard


DATASET = "onebitbrain/g02-dinov2-r5-cpu-input-v1"
MOUNT = "/kaggle/input/datasets/onebitbrain/g02-dinov2-r5-cpu-input-v1"
MANIFEST_BYTES = (
    b'{"preregistered_visual_identities":["train:44b6_fixture"],'
    b'"schema":"celltrack-alpha.local-visual-input-manifest.v1"}\n'
)
CHECKPOINT_BYTES = b"fixture-checkpoint-bytes\n"
ENVIRONMENT_BYTES = b'{"python":"fixture","origin":"local"}\n'
MANIFEST_SHA = hashlib.sha256(MANIFEST_BYTES).hexdigest()
CHECKPOINT_SHA = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
ENVIRONMENT_SHA = hashlib.sha256(ENVIRONMENT_BYTES).hexdigest()
POPULATION_SHA = "c" * 64


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class PrelaunchGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.kernel = self.root / "kernel"
        self.kernel.mkdir()
        self.checkpoint = self.root / "assets/checkpoint.pt"
        self.input_manifest = self.root / "assets/input-manifest.json"
        self.environment = self.root / "assets/environment.json"
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(CHECKPOINT_BYTES)
        self.input_manifest.write_bytes(MANIFEST_BYTES)
        self.environment.write_bytes(ENVIRONMENT_BYTES)
        self.runner = self.root / "scripts/trusted_local_runner.py"
        self.runner.parent.mkdir()
        self.runner.write_text("#!/usr/bin/env python3\nprint('trusted fixture runner')\n")
        self.runner_stdout = self.root / "local_visual/runner.stdout"
        self.runner_stderr = self.root / "local_visual/runner.stderr"
        self.runner_stdout.parent.mkdir()
        self.runner_stdout.write_text("fixture runner completed\n")
        self.runner_stderr.write_text("no errors\n")
        self.runner_id = "unit-test-trusted-runner"
        self.runner_path = str(self.runner.relative_to(self.root))
        self.runner_sha256 = guard.sha256(self.runner)
        self.runner_argv = [self.runner_path, "--contract", "unit-test"]
        self.runner_command_sha256 = guard._canonical_sha(self.runner_argv)
        self.trusted_runner_patch = mock.patch.dict(
            guard.TRUSTED_LOCAL_RUNNERS,
            {
                self.runner_id: {
                    "path": self.runner_path,
                    "sha256": self.runner_sha256,
                }
            },
            clear=True,
        )
        self.trusted_runner_patch.start()
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
        self.trusted_runner_patch.stop()
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

    def make_video(self, path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("ffmpeg is required for media-integrity tests")
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=2:d=1",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )

    def visual_reference(self, code_sha256: str) -> tuple[dict, dict]:
        directory = self.root / "local_visual"
        directory.mkdir(exist_ok=True)
        prediction = directory / "prediction.csv"
        prediction.write_text("source_id,target_id\n1,2\n", encoding="utf-8")
        video = directory / "train__44b6_fixture.mp4"
        self.make_video(video)
        package = guard._package_inventory(self.kernel, self.root)
        candidate_path = str((self.kernel / "script.py").relative_to(self.root))
        checkpoint_path = str(self.checkpoint.relative_to(self.root))
        input_path = str(self.input_manifest.relative_to(self.root))
        environment_path = str(self.environment.relative_to(self.root))
        prediction_path = str(prediction.relative_to(self.root))
        execution = directory / "LOCAL_EXECUTION.json"
        write_json(
            execution,
            {
                "schema": "celltrack-alpha.local-model-execution.v1",
                "status": "PASS",
                "execution_origin": "local",
                "exit_code": 0,
                "kernel_id": "onebitbrain/test",
                "model_id": "fixture-tracker",
                "local_run_id": "fixture-local-run-1",
                "package_path": package["path"],
                "package_tree_sha256": package["tree_sha256"],
                "candidate_code_path": candidate_path,
                "candidate_code_sha256": code_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": CHECKPOINT_SHA,
                "input_manifest_path": input_path,
                "input_manifest_sha256": MANIFEST_SHA,
                "execution_environment_path": environment_path,
                "execution_environment_sha256": ENVIRONMENT_SHA,
                "preregistered_visual_identities": ["train:44b6_fixture"],
                "runner_id": self.runner_id,
                "runner_path": self.runner_path,
                "runner_sha256": self.runner_sha256,
                "invoked_argv": self.runner_argv,
                "invoked_argv_sha256": self.runner_command_sha256,
                "started_at": "2026-08-14T00:00:00Z",
                "ended_at": "2026-08-14T00:01:00Z",
                "prediction_artifacts": [
                    {
                        "identity": "train:44b6_fixture",
                        "path": prediction_path,
                        "sha256": guard.sha256(prediction),
                    }
                ],
                "process_output_receipt": {
                    "schema": "celltrack-alpha.local-process-output.v1",
                    "exit_code": 0,
                    "started_at": "2026-08-14T00:00:00Z",
                    "ended_at": "2026-08-14T00:01:00Z",
                    "stdout_path": str(self.runner_stdout.relative_to(self.root)),
                    "stdout_sha256": guard.sha256(self.runner_stdout),
                    "stderr_path": str(self.runner_stderr.relative_to(self.root)),
                    "stderr_sha256": guard.sha256(self.runner_stderr),
                    "prediction_artifacts": [
                        {
                            "identity": "train:44b6_fixture",
                            "path": prediction_path,
                            "sha256": guard.sha256(prediction),
                        }
                    ],
                },
            },
        )
        execution_sha = guard.sha256(execution)
        sidecar = directory / "train__44b6_fixture.json"
        write_json(
            sidecar,
            {
                "schema": guard.VISUAL_SIDECAR_SCHEMA,
                "identity": "train:44b6_fixture",
                "kernel_id": "onebitbrain/test",
                "model_id": "fixture-tracker",
                "local_run_id": "fixture-local-run-1",
                "package_tree_sha256": package["tree_sha256"],
                "candidate_code_path": candidate_path,
                "candidate_code_sha256": code_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": CHECKPOINT_SHA,
                "input_manifest_path": input_path,
                "input_manifest_sha256": MANIFEST_SHA,
                "execution_environment_path": environment_path,
                "execution_environment_sha256": ENVIRONMENT_SHA,
                "preregistered_visual_identities": ["train:44b6_fixture"],
                "prediction_artifact_path": prediction_path,
                "prediction_artifact_sha256": guard.sha256(prediction),
                "video_path": str(video.relative_to(self.root)),
                "video_sha256": guard.sha256(video),
                "execution_receipt_sha256": execution_sha,
                "rendered_at": "2026-08-14T00:02:00Z",
            },
        )
        receipt = directory / "LOCAL_VISUAL_PREFLIGHT.json"
        write_json(
            receipt,
            {
                "schema": guard.VISUAL_RECEIPT_SCHEMA,
                "status": "PASS",
                "model_execution": True,
                "kernel_id": "onebitbrain/test",
                "model_id": "fixture-tracker",
                "local_run_id": "fixture-local-run-1",
                "package_tree_sha256": package["tree_sha256"],
                "candidate_code_path": candidate_path,
                "candidate_code_sha256": code_sha256,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": CHECKPOINT_SHA,
                "input_manifest_path": input_path,
                "input_manifest_sha256": MANIFEST_SHA,
                "execution_environment_path": environment_path,
                "execution_environment_sha256": ENVIRONMENT_SHA,
                "preregistered_visual_identities": ["train:44b6_fixture"],
                "execution_receipt_sha256": execution_sha,
                "sealed_at": "2026-08-14T00:04:00Z",
                "videos": [
                    {
                        "identity": "train:44b6_fixture",
                        "prediction_artifact_path": str(prediction.relative_to(self.root)),
                        "prediction_artifact_sha256": guard.sha256(prediction),
                        "video_path": str(video.relative_to(self.root)),
                        "video_sha256": guard.sha256(video),
                        "sidecar_path": str(sidecar.relative_to(self.root)),
                        "sidecar_sha256": guard.sha256(sidecar),
                    }
                ],
                "reviews": [
                    {
                        "identity": "train:44b6_fixture",
                        "video_sha256": guard.sha256(video),
                        "visual_analysis_performed": True,
                        "metrics_only": False,
                        "verdict": "PASS",
                        "reviewer": "unit-test-reviewer",
                        "reviewed_at": "2026-08-14T00:03:00Z",
                        "promotion_rationale": "No catastrophic behavior in the bound fixture clip.",
                        "observations": [
                            "The exact candidate prediction remains visually traceable."
                        ],
                        "failure_taxonomy": {
                            key: "not observed in the fixture clip"
                            for key in sorted(guard.VISUAL_FAILURE_TAXONOMY)
                        },
                    }
                ],
            },
        )
        return (
            {"path": str(receipt.relative_to(self.root)), "sha256": guard.sha256(receipt)},
            {"path": str(execution.relative_to(self.root)), "sha256": execution_sha},
        )

    def contract(self, source: str, attested: bool = True) -> dict:
        code = self.kernel / "script.py"
        code.write_text(source, encoding="utf-8")
        write_json(self.kernel / "kernel-metadata.json", self.metadata)
        visual, execution = self.visual_reference(guard.sha256(code))
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
            "input_manifest_path": str(self.input_manifest.relative_to(self.root)),
            "checkpoint_sha256": CHECKPOINT_SHA,
            "checkpoint_path": str(self.checkpoint.relative_to(self.root)),
            "execution_environment_path": str(self.environment.relative_to(self.root)),
            "execution_environment_sha256": ENVIRONMENT_SHA,
            "preregistered_visual_identities": ["train:44b6_fixture"],
            "local_execution_command_sha256": self.runner_command_sha256,
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
            "model_execution": True,
            "local_visual_preflight_receipt": visual,
            "local_execution_receipt": execution,
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
            self.kernel,
            self.contract(source, attested),
            self.registry,
            phase,
            project_root=self.root,
        )

    def check_contract(self, contract: dict, phase: str):
        return guard.check_kernel(
            self.kernel,
            contract,
            self.registry,
            phase,
            project_root=self.root,
        )

    def reseal_video_bindings(self, contract: dict) -> None:
        receipt_path = self.root / contract["local_visual_preflight_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        video_record = receipt["videos"][0]
        video_path = self.root / video_record["video_path"]
        video_sha = guard.sha256(video_path)
        video_record["video_sha256"] = video_sha
        receipt["reviews"][0]["video_sha256"] = video_sha
        sidecar_path = self.root / video_record["sidecar_path"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["video_sha256"] = video_sha
        write_json(sidecar_path, sidecar)
        video_record["sidecar_sha256"] = guard.sha256(sidecar_path)
        write_json(receipt_path, receipt)
        contract["local_visual_preflight_receipt"]["sha256"] = guard.sha256(receipt_path)

    def reseal_execution_binding(self, contract: dict) -> None:
        execution_path = self.root / contract["local_execution_receipt"]["path"]
        execution_sha = guard.sha256(execution_path)
        contract["local_execution_receipt"]["sha256"] = execution_sha
        receipt_path = self.root / contract["local_visual_preflight_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["execution_receipt_sha256"] = execution_sha
        sidecar_path = self.root / receipt["videos"][0]["sidecar_path"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["execution_receipt_sha256"] = execution_sha
        write_json(sidecar_path, sidecar)
        receipt["videos"][0]["sidecar_sha256"] = guard.sha256(sidecar_path)
        write_json(receipt_path, receipt)
        contract["local_visual_preflight_receipt"]["sha256"] = guard.sha256(receipt_path)

    def reseal_identity(self, contract: dict, identity: str) -> None:
        execution_path = self.root / contract["local_execution_receipt"]["path"]
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        execution["prediction_artifacts"][0]["identity"] = identity
        execution["process_output_receipt"]["prediction_artifacts"][0][
            "identity"
        ] = identity
        write_json(execution_path, execution)
        self.reseal_execution_binding(contract)

        receipt_path = self.root / contract["local_visual_preflight_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["videos"][0]["identity"] = identity
        receipt["reviews"][0]["identity"] = identity
        sidecar_path = self.root / receipt["videos"][0]["sidecar_path"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["identity"] = identity
        write_json(sidecar_path, sidecar)
        receipt["videos"][0]["sidecar_sha256"] = guard.sha256(sidecar_path)
        write_json(receipt_path, receipt)
        contract["local_visual_preflight_receipt"]["sha256"] = guard.sha256(receipt_path)

    def _write_remote_lmt(self) -> Path:
        path = self.root / "LMT.md"
        policy = {
            "schema": "cell-tracking.autonomy-limits.v1",
            "local_resource_safety": {"maximum_combined_rss_mib": 8192.0},
        }
        path.write_text(
            "# fixture LMT\n<!-- LMT_POLICY_JSON_BEGIN -->\n```json\n"
            + json.dumps(policy)
            + "\n```\n<!-- LMT_POLICY_JSON_END -->\n",
            encoding="utf-8",
        )
        return path

    def remote_visual_contract(
        self,
        *,
        route: dict | None = None,
    ) -> tuple[dict, dict[str, dict[str, str]]]:
        """Build a complete pending-remote-review exception fixture."""

        route = route or guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE
        is_d2k3 = route["kernel_id"] == guard.REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE[
            "kernel_id"
        ]
        identities = ["train:44b6_sparse", "train:44b6_crowded"]
        manifest = {
            "schema": guard.VISUAL_INPUT_MANIFEST_SCHEMA,
            "preregistered_visual_identities": identities,
        }
        self.input_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        manifest_sha = guard.sha256(self.input_manifest)
        mount = route["mount_root"]
        unattended_defaults = (
            "    mode = os.environ.get('D2K2_MODE', 'release')\n"
            "    units = int(os.environ.get('D2K2_WORK_UNITS', '32'))\n"
            "    if mode == 'release' and units != 32:\n"
            "        raise RuntimeError('release needs all units')\n"
            if is_d2k3
            else ""
        )
        source = (
            ("import os\n" if is_d2k3 else "")
            + "from pathlib import Path\n"
            + f"DECLARED_REMOTE_ROOT = Path('{mount}')\n"
            + "def discover_competition_root():\n"
            + "    return DECLARED_REMOTE_ROOT\n"
            + "def step0():\n"
            + unattended_defaults
            + "    source = discover_competition_root()\n"
            + "    print('D2K2_STEP0_ASSIGNMENT_COMPLETE')\n"
            + "    return source\n"
            + "def preflight(source):\n"
            + "    return source\n"
            + "def release(source):\n"
            + "    print('D2K2_HEAVY_WORK_BEGIN')\n"
            + "    print('D2K2_VISUAL_RELEASE_COMPLETE')\n"
            + "    return ('RESOURCE_CAP_ABORT.json', 'RESOURCE_MONITOR_ABORT.json', 'D2K2_CUDA_EXIT_RECEIPT')\n"
            + "def main():\n"
            + "    source = step0()\n"
            + "    if False:\n"
            + "        preflight(source)\n"
            + "    else:\n"
            + "        release(source)\n"
            + "if __name__ == '__main__':\n"
            + "    main()\n"
        )
        contract = self.contract(source)
        script = self.kernel / "script.py"
        code = self.kernel / "kernel.py"
        script.rename(code)
        self.metadata = dict(route["metadata"])
        write_json(self.kernel / "kernel-metadata.json", self.metadata)
        contract.update(
            {
                "kernel_id": route["kernel_id"],
                "remote_preflight_kernel_id": route["kernel_id"],
                "remote_preflight_mode": (
                    guard.D2K3_INLINE_STEP0_MODE
                    if is_d2k3
                    else "d2k2_step0_before_native_release.v1"
                ),
                "dataset_sources": [],
                "competition_sources": [
                    route["competition_source"]
                ],
                "expected_mount_root": route["mount_root"],
                "remote_visual_input_route": {
                    "mode": route["mode"],
                    "competition_source": route["competition_source"],
                    "mount_root": route["mount_root"],
                },
                "input_manifest_sha256": manifest_sha,
                "preregistered_visual_identities": identities,
                "execution_profile": guard.REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
                "kernel_properties": route["contract_kernel_properties"],
                "code_sha256": guard.sha256(code),
            }
        )
        contract.pop("exclusive_dataset_source")
        contract.pop("remote_preflight_attestation")
        contract.pop("local_visual_preflight_receipt")
        contract.pop("local_execution_receipt")

        assets = self.root / "assets/remote_visual"
        assets.mkdir(parents=True)
        package_contract = self.kernel / "D2K2_CONTRACT.json"
        write_json(
            package_contract,
            {
                "schema": guard.D2K2_PACKAGE_CONTRACT_SCHEMA,
                "candidate_id": guard.D2K2_CANDIDATE_ID,
                "competition_input": {
                    "source": route["competition_source"],
                    "arrays": {
                        identity.split(":", 1)[1]: {}
                        for identity in guard.D2K2_FIXED_IDENTITIES
                    },
                },
                "learning_tuning_network": False,
                "local_simulator_egress_forbidden": True,
                "resource_policy": guard.D2K2_RESOURCE_POLICY.copy(),
            },
        )
        package_manifest = self.kernel / "PACKAGE_MANIFEST.json"
        write_json(
            package_manifest,
            {
                "schema": guard.D2K2_PACKAGE_MANIFEST_SCHEMA,
                "candidate_id": guard.D2K2_CANDIDATE_ID,
                "entrypoint_unsealed_but_certifier_bound": "kernel.py",
                "metadata_unsealed_but_certifier_bound": "kernel-metadata.json",
                "files": [
                    {
                        "path": package_contract.name,
                        "bytes": package_contract.stat().st_size,
                        "sha256": guard.sha256(package_contract),
                    }
                ],
            },
        )
        contract["d2k2_package_contract"] = {
            "path": str(package_contract.relative_to(self.root)),
            "sha256": guard.sha256(package_contract),
        }
        contract["d2k2_package_manifest"] = {
            "path": str(package_manifest.relative_to(self.root)),
            "sha256": guard.sha256(package_manifest),
        }
        lmt_path = self._write_remote_lmt()
        lmt_ref = {"path": "LMT.md", "sha256": guard.sha256(lmt_path)}
        resource = assets / "RESOURCE_STOP.json"
        write_json(
            resource,
            {
                "schema": "celltrack-alpha.local-model-execution-abort.v1",
                "status": "ABORT",
                "failure_type": "ResourceStop",
                "pass_receipt_absent": True,
                "owned_process_group_terminated": True,
                "resource": {"peak_combined_rss_mib": 8192.0},
            },
        )
        resource_ref = {
            "path": str(resource.relative_to(self.root)),
            "sha256": guard.sha256(resource),
        }
        post_review = {
            "remote_mp4_retrieval_required": True,
            "local_visual_review_required": True,
            "score_or_promotion_permitted_before_review": False,
            "remote_review_receipt_schema": guard.REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA,
            "required_remote_mp4_identities": identities,
        }
        gpu_contract = assets / "GPU_VISUAL_ONLY_CONTRACT.json"
        remote_memory = 29_000_000_000
        write_json(
            gpu_contract,
            {
                "schema": guard.REMOTE_VISUAL_ONLY_CONTRACT_SCHEMA,
                "status": "FROZEN",
                "candidate_id": "CELLECT-NATIVE-3HEAD-X3RD-v1",
                "kernel_id": contract["kernel_id"],
                "execution_profile": guard.REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
                "model_execution": True,
                "device": "gpu",
                "fixed_preregistered_visual_identities": identities,
                "remote_memory_limit_bytes": remote_memory,
                "prohibitions": {
                    key: False
                    for key in (
                        "labels", "targets", "scorer", "full38", "training",
                        "tuning", "calibration", "hpo", "submission",
                    )
                },
                "post_remote_review": post_review,
            },
        )
        gpu_ref = {
            "path": str(gpu_contract.relative_to(self.root)),
            "sha256": guard.sha256(gpu_contract),
        }
        profile = assets / "preflight-profile.json"
        write_json(profile, {"schema": "fixture-profile", "kernel": contract["kernel_id"]})
        certificate = assets / "CERTIFICATION.json"
        write_json(
            certificate,
            {
                "schema": "ship-kaggle-kernels.certification.v1",
                "status": "ELIGIBLE",
                "eligible": True,
                "autonomous_launch_eligible": True,
                "kernel_id": contract["kernel_id"],
                "limits_file": str(lmt_path.resolve()),
                "limits_sha256": guard.sha256(lmt_path),
                "profile_path": str(profile.resolve()),
                "profile_sha256": guard.sha256(profile),
                "receipt_content_sha256": "d" * 64,
                "artifact_sha256": guard.sha256(code),
                "metadata_sha256": guard.sha256(self.kernel / "kernel-metadata.json"),
                "package_inventory_sha256": guard._ship_package_inventory_sha256(self.kernel),
                "runtime_projection": {
                    "status": "PASS",
                    "physical_memory_limit_bytes": remote_memory,
                    "projected_peak_rss_bytes": remote_memory - 1,
                },
            },
        )
        certificate_ref = {
            "path": str(certificate.relative_to(self.root)),
            "sha256": guard.sha256(certificate),
        }
        remote = {
            "schema": guard.REMOTE_VISUAL_ONLY_PRELAUNCH_SCHEMA,
            "candidate_id": "CELLECT-NATIVE-3HEAD-X3RD-v1",
            "kernel_id": contract["kernel_id"],
            "execution_profile": guard.REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
            "model_execution": True,
            "candidate_code_path": str(code.relative_to(self.root)),
            "candidate_code_sha256": guard.sha256(code),
            "kernel_package_tree_sha256": guard._package_inventory(self.kernel, self.root)["tree_sha256"],
            "fixed_preregistered_visual_identities": identities,
            "resource_stop_receipt": resource_ref,
            "gpu_visual_only_contract": gpu_ref,
            "ship_kaggle_certificate": certificate_ref,
            "post_remote_review": post_review,
        }
        contract["remote_visual_only"] = remote
        exception_id = (
            "cellect-d2k3-remote-visual-only-v1"
            if is_d2k3
            else "cellect-d2k2-remote-visual-only-v1"
        )
        record = self.root / "division/d2k2_kaggle_visual/REMOTE_VISUAL_ONLY_EXCEPTION.json"
        record.parent.mkdir(parents=True)
        write_json(
            record,
            {
                "schema": guard.REMOTE_VISUAL_ONLY_EXCEPTION_SCHEMA,
                "status": "APPROVED",
                "exception_id": exception_id,
                "candidate_id": remote["candidate_id"],
                "kernel_id": contract["kernel_id"],
                "execution_profile": guard.REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
                "fixed_preregistered_visual_identities": identities,
                "resource_stop_receipt": resource_ref,
                "gpu_visual_only_contract": gpu_ref,
                "ship_kaggle_certificate": certificate_ref,
                "lmt": lmt_ref,
                "post_remote_review": post_review,
            },
        )
        contract["remote_visual_only_exception"] = {
            "exception_id": exception_id,
            "record_path": str(record.relative_to(self.root)),
            "record_sha256": guard.sha256(record),
        }
        allowlist = {
            exception_id: {
                "record_path": contract["remote_visual_only_exception"]["record_path"],
                "record_sha256": contract["remote_visual_only_exception"]["record_sha256"],
                "kernel_id": contract["kernel_id"],
                "execution_profile": contract["execution_profile"],
            }
        }
        return contract, allowlist

    def reseal_remote_exception(self, contract: dict, allowlist: dict[str, dict[str, str]]) -> None:
        """Re-bind a mutated fixture evidence leaf through the central record."""

        remote = contract["remote_visual_only"]
        record_path = self.root / contract["remote_visual_only_exception"]["record_path"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for key in ("resource_stop_receipt", "gpu_visual_only_contract", "ship_kaggle_certificate"):
            reference = remote[key]
            path = self.root / reference["path"]
            reference["sha256"] = guard.sha256(path)
            record[key] = reference.copy()
        record["fixed_preregistered_visual_identities"] = remote[
            "fixed_preregistered_visual_identities"
        ]
        record["post_remote_review"] = remote["post_remote_review"]
        write_json(record_path, record)
        digest = guard.sha256(record_path)
        contract["remote_visual_only_exception"]["record_sha256"] = digest
        allowlist[contract["remote_visual_only_exception"]["exception_id"]]["record_sha256"] = digest

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

    def test_malformed_generic_contract_fails_with_a_receipt_not_keyerror(self) -> None:
        source = self.valid_source()
        code = self.kernel / "script.py"
        code.write_text(source, encoding="utf-8")
        write_json(self.kernel / "kernel-metadata.json", self.metadata)
        result = guard.check_kernel(
            self.kernel,
            {"kernel_id": "onebitbrain/test"},
            self.registry,
            "inline",
            project_root=self.root,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["eligible"])
        self.assertIn("generic contract missing required fields", result["failures"][0])

    def test_launch_attestation_respects_the_gpu_contract(self) -> None:
        source = self.valid_source()
        self.metadata["enable_gpu"] = True
        contract = self.contract(source)
        contract["kernel_properties"]["enable_gpu"] = True
        contract["remote_preflight_attestation"]["enable_gpu"] = True
        result = self.check_contract(contract, "launch")
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
        result = self.check_contract(contract, "static")
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
        result = self.check_contract(contract, "launch")
        self.assertIn("exceeded step-0 limit", " ".join(result["failures"]))

    def test_warning_only_step0_allows_elapsed_time_but_not_competition_access(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["step0_elapsed_is_hard_failure"] = False
        contract["step0_limit_seconds"] = 60.0
        contract["step0_competition_path_visits"] = 0
        contract["remote_preflight_attestation"]["step0_elapsed_seconds"] = 61.0
        contract["remote_preflight_attestation"]["competition_path_visits"] = 0
        self.assertEqual(self.check_contract(contract, "launch")["status"], "PASS")
        contract["remote_preflight_attestation"]["competition_path_visits"] = 1
        result = self.check_contract(contract, "launch")
        self.assertIn("competition-path visit drift", " ".join(result["failures"]))

    def test_zero_request_count_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["request_count"] = 0
        contract["remote_preflight_attestation"]["request_count"] = 0
        result = self.check_contract(contract, "launch")
        self.assertIn("request_count must be positive", result["failures"])

    def test_zero_wheel_base_image_contract_passes_and_requires_explicit_mode(self) -> None:
        source = self.valid_source().replace("EXPECTED_WHEEL_COUNT = 3", "EXPECTED_WHEEL_COUNT = 0")
        contract = self.contract(source)
        contract["wheel_count"] = 0
        contract["remote_preflight_attestation"]["wheel_count"] = 0
        contract["offline_dependency_mode"] = "kaggle_base_image_only"
        self.assertEqual(
            self.check_contract(contract, "launch")["status"],
            "PASS",
        )
        contract.pop("offline_dependency_mode")
        result = self.check_contract(contract, "launch")
        self.assertIn("zero-wheel packages", " ".join(result["failures"]))

    def test_missing_checkpoint_pin_is_rejected(self) -> None:
        source = self.valid_source().replace(CHECKPOINT_SHA, "d" * 64)
        result = self.run_guard(source)
        self.assertIn("pinned checkpoint SHA absent from code", result["failures"])

    def test_attestation_cannot_start_heavy_work(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["remote_preflight_attestation"]["heavy_work_started"] = True
        result = self.check_contract(contract, "launch")
        self.assertIn(
            "remote preflight attestation drift: heavy_work_started",
            result["failures"],
        )

    def test_cross_kernel_preflight_attestation_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["remote_preflight_kernel_id"] = "onebitbrain/different-kernel"
        result = self.check_contract(contract, "launch")
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

    def test_model_kernel_without_visual_receipt_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract.pop("local_visual_preflight_receipt")
        result = self.check_contract(contract, "static")
        self.assertIn("visual preflight receipt", " ".join(result["failures"]))

    def test_changed_candidate_cannot_reuse_a_baseline_receipt(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        receipt_path = self.root / contract["local_visual_preflight_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["candidate_code_sha256"] = "e" * 64
        write_json(receipt_path, receipt)
        contract["local_visual_preflight_receipt"]["sha256"] = guard.sha256(receipt_path)
        result = self.check_contract(contract, "static")
        self.assertIn("candidate_code_sha256", " ".join(result["failures"]))

    def test_video_tampering_after_review_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        (self.root / "local_visual/train__44b6_fixture.mp4").write_bytes(
            b"changed-after-review"
        )
        result = self.check_contract(contract, "static")
        self.assertIn("review video 0 hash drift", " ".join(result["failures"]))

    def test_renamed_non_mp4_is_rejected_even_when_hash_bound(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        video_path = self.root / "local_visual/train__44b6_fixture.mp4"
        video_path.write_bytes(b"not actually a video")
        self.reseal_video_bindings(contract)
        result = self.check_contract(contract, "static")
        self.assertIn("failed ffprobe", " ".join(result["failures"]))

    def test_box_like_24_byte_fake_mp4_is_rejected(self) -> None:
        contract = self.contract(self.valid_source())
        video_path = self.root / "local_visual/train__44b6_fixture.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00moovmdat")
        self.reseal_video_bindings(contract)
        result = self.check_contract(contract, "static")
        self.assertIn("ffprobe", " ".join(result["failures"]))

    def test_truncated_mp4_is_rejected_after_hashes_are_resealed(self) -> None:
        contract = self.contract(self.valid_source())
        video_path = self.root / "local_visual/train__44b6_fixture.mp4"
        body = video_path.read_bytes()
        video_path.write_bytes(body[: max(32, len(body) // 2)])
        self.reseal_video_bindings(contract)
        result = self.check_contract(contract, "static")
        self.assertRegex(
            " ".join(result["failures"]),
            r"ffprobe|decode|zero decodable|no video stream",
        )

    def test_audio_only_mp4_is_rejected(self) -> None:
        contract = self.contract(self.valid_source())
        video_path = self.root / "local_visual/train__44b6_fixture.mp4"
        subprocess.run(
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:duration=0.2",
                "-c:a",
                "aac",
                str(video_path),
            ],
            check=True,
        )
        self.reseal_video_bindings(contract)
        result = self.check_contract(contract, "static")
        self.assertIn("no video stream", " ".join(result["failures"]))

    def test_zero_frame_media_is_rejected(self) -> None:
        contract = self.contract(self.valid_source())
        video_path = self.root / "local_visual/train__44b6_fixture.mp4"
        video_path.write_bytes(
            b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
            b"\x00\x00\x00\x08free\x00\x00\x00\x08mdat"
        )
        self.reseal_video_bindings(contract)
        result = self.check_contract(contract, "static")
        self.assertRegex(" ".join(result["failures"]), r"no video stream|zero decodable|ffprobe")

    def test_checkpoint_and_input_hashes_must_be_full_and_match_files(self) -> None:
        for field, value in (("checkpoint_sha256", None), ("input_manifest_sha256", "abc")):
            with self.subTest(field=field):
                contract = self.contract(self.valid_source())
                contract[field] = value
                result = self.check_contract(contract, "static")
                self.assertIn("must be a lowercase SHA-256", " ".join(result["failures"]))
        contract = self.contract(self.valid_source())
        self.checkpoint.write_bytes(b"tampered checkpoint")
        result = self.check_contract(contract, "static")
        self.assertIn("checkpoint hash drift", " ".join(result["failures"]))

    def test_remote_execution_receipt_cannot_substitute_for_local_run(self) -> None:
        contract = self.contract(self.valid_source())
        execution_path = self.root / contract["local_execution_receipt"]["path"]
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        execution["execution_origin"] = "remote"
        write_json(execution_path, execution)
        self.reseal_execution_binding(contract)
        result = self.check_contract(contract, "static")
        self.assertIn("execution_origin", " ".join(result["failures"]))

    def test_self_authored_execution_receipt_and_synthetic_clip_are_untrusted(self) -> None:
        contract = self.contract(self.valid_source())
        with mock.patch.dict(guard.TRUSTED_LOCAL_RUNNERS, {}, clear=True):
            result = self.check_contract(contract, "static")
        self.assertIn("runner is not centrally trusted", " ".join(result["failures"]))

    def test_trusted_runner_hash_drift_is_rejected(self) -> None:
        contract = self.contract(self.valid_source())
        execution_path = self.root / contract["local_execution_receipt"]["path"]
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        execution["runner_sha256"] = "0" * 64
        write_json(execution_path, execution)
        self.reseal_execution_binding(contract)
        result = self.check_contract(contract, "static")
        self.assertIn("trusted local runner path/hash drift", " ".join(result["failures"]))

    def test_empty_trusted_runner_stderr_is_exactly_hash_bound(self) -> None:
        contract = self.contract(self.valid_source())
        self.runner_stderr.write_bytes(b"")
        execution_path = self.root / contract["local_execution_receipt"]["path"]
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        execution["process_output_receipt"]["stderr_sha256"] = guard.sha256(
            self.runner_stderr
        )
        write_json(execution_path, execution)
        self.reseal_execution_binding(contract)
        result = self.check_contract(contract, "static")
        self.assertEqual(result["status"], "PASS")

    def test_preregistered_identity_set_is_required_and_manifest_bound(self) -> None:
        contract = self.contract(self.valid_source())
        contract.pop("preregistered_visual_identities")
        result = self.check_contract(contract, "static")
        self.assertIn("must be a nonempty identity list", " ".join(result["failures"]))

        contract = self.contract(self.valid_source())
        self.input_manifest.write_text(
            json.dumps({"schema": guard.VISUAL_INPUT_MANIFEST_SCHEMA}) + "\n"
        )
        contract["input_manifest_sha256"] = guard.sha256(self.input_manifest)
        result = self.check_contract(contract, "static")
        self.assertIn(
            "bound input manifest preregistered_visual_identities",
            " ".join(result["failures"]),
        )

    def test_resealed_malformed_identities_are_rejected(self) -> None:
        for identity in (
            "train:",
            ":x",
            "train:../x",
            "train:x/y",
            "train:x\\y",
            "train:\x01x",
        ):
            with self.subTest(identity=repr(identity)):
                contract = self.contract(self.valid_source())
                self.reseal_identity(contract, identity)
                result = self.check_contract(contract, "static")
                self.assertIn("must be split:dataset", " ".join(result["failures"]))

    def test_resealed_valid_but_inconsistent_identity_is_rejected(self) -> None:
        contract = self.contract(self.valid_source())
        self.reseal_identity(contract, "train:6bba_other")
        result = self.check_contract(contract, "static")
        joined = " ".join(result["failures"])
        self.assertRegex(joined, r"filename is inconsistent|differs from preregistration")
    def test_missing_per_video_review_coverage_is_rejected(self) -> None:
        contract = self.contract(self.valid_source())
        receipt_path = self.root / contract["local_visual_preflight_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["reviews"] = []
        write_json(receipt_path, receipt)
        contract["local_visual_preflight_receipt"]["sha256"] = guard.sha256(receipt_path)
        result = self.check_contract(contract, "static")
        self.assertIn("per-video review", " ".join(result["failures"]))

    def test_timestamp_ordering_is_enforced(self) -> None:
        contract = self.contract(self.valid_source())
        receipt_path = self.root / contract["local_visual_preflight_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["reviews"][0]["reviewed_at"] = "2026-08-13T23:00:00Z"
        write_json(receipt_path, receipt)
        contract["local_visual_preflight_receipt"]["sha256"] = guard.sha256(receipt_path)
        result = self.check_contract(contract, "static")
        self.assertIn("predates video rendering", " ".join(result["failures"]))

    def test_specific_non_model_environment_probe_exemption_passes(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["model_execution"] = False
        contract["execution_profile"] = "environment_probe"
        contract["non_model_execution_allowlist_id"] = "unit-test-environment-probe"
        contract["non_model_execution_justification"] = (
            "This kernel only inventories the mounted environment and emits no tracking output."
        )
        contract.pop("local_visual_preflight_receipt")
        contract.pop("local_execution_receipt")
        with mock.patch.dict(
            guard.NON_MODEL_EXECUTION_ALLOWLIST,
            {
                "unit-test-environment-probe": (
                    contract["kernel_id"],
                    contract["execution_profile"],
                    contract["code_sha256"],
                )
            },
            clear=True,
        ):
            result = self.check_contract(contract, "static")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["local_visual_preflight"]["status"], "NOT_APPLICABLE"
        )

    def test_unapproved_non_model_exemption_is_rejected(self) -> None:
        source = self.valid_source()
        contract = self.contract(source)
        contract["model_execution"] = False
        contract["execution_profile"] = "environment_probe"
        contract["non_model_execution_allowlist_id"] = "fabricated"
        contract["non_model_execution_justification"] = (
            "This text attempts to waive visual review without an approved probe profile."
        )
        contract.pop("local_visual_preflight_receipt")
        contract.pop("local_execution_receipt")
        result = self.check_contract(contract, "static")
        self.assertIn("centrally allowlisted", " ".join(result["failures"]))

    def test_allowlisted_non_model_source_with_model_marker_is_rejected(self) -> None:
        source = self.valid_source() + "\ndef run_inference(): pass\n"
        contract = self.contract(source)
        contract.update(
            {
                "model_execution": False,
                "execution_profile": "environment_probe",
                "non_model_execution_allowlist_id": "unit-test-environment-probe",
                "non_model_execution_justification": (
                    "This purported probe attempts to execute a tracking inference path."
                ),
            }
        )
        contract.pop("local_visual_preflight_receipt")
        contract.pop("local_execution_receipt")
        with mock.patch.dict(
            guard.NON_MODEL_EXECUTION_ALLOWLIST,
            {
                "unit-test-environment-probe": (
                    contract["kernel_id"],
                    contract["execution_profile"],
                    contract["code_sha256"],
                )
            },
            clear=True,
        ):
            result = self.check_contract(contract, "static")
        self.assertIn("model-execution markers", " ".join(result["failures"]))

    def test_centrally_pinned_remote_visual_only_exception_passes_pending_review(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            static = self.check_contract(contract, "static")
            inline = self.check_contract(contract, "inline")
            launch = self.check_contract(contract, "launch")
        for result in (static, inline, launch):
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                result["local_visual_preflight"]["status"],
                guard.REMOTE_VISUAL_PENDING_STATUS,
            )
            self.assertEqual(result["local_visual_preflight"]["video_count"], 0)
            self.assertEqual(
                result["local_visual_preflight"]["post_remote_review"]["status"],
                "PENDING_REMOTE_RETRIEVAL_AND_LOCAL_REVIEW",
            )
        self.assertFalse(static["eligible"])
        self.assertTrue(inline["eligible"])
        self.assertTrue(launch["eligible"])

    def test_d2k3_repair_lane_requires_its_own_central_record_and_defaults(self) -> None:
        contract, allowlist = self.remote_visual_contract(
            route=guard.REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE
        )
        self.assertEqual(contract["kernel_id"], "onebitbrain/g02-cellect-visual-only-v3")
        self.assertEqual(contract["remote_preflight_mode"], guard.D2K3_INLINE_STEP0_MODE)
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            inline = self.check_contract(contract, "inline")
            launch = self.check_contract(contract, "launch")
        for result in (inline, launch):
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["eligible"])
            self.assertEqual(
                result["local_visual_preflight"]["status"],
                guard.REMOTE_VISUAL_PENDING_STATUS,
            )

    def test_d2k3_repair_lane_cannot_self_authorize_before_root_record_exists(self) -> None:
        contract, _ = self.remote_visual_contract(
            route=guard.REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE
        )
        result = self.check_contract(contract, "launch")
        self.assertIn("central exact allowlist", " ".join(result["failures"]))

    def test_d2k3_repair_lane_rejects_nondefault_inline_mode(self) -> None:
        contract, allowlist = self.remote_visual_contract(
            route=guard.REMOTE_VISUAL_ONLY_D2K3_COMPETITION_ROUTE
        )
        contract["remote_preflight_mode"] = "d2k2_step0_before_native_release.v1"
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("explicit inline step-0 release mode", " ".join(result["failures"]))

    def test_remote_exception_accepts_only_the_pinned_direct_competition_mount(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        self.assertEqual(contract["dataset_sources"], [])
        self.assertEqual(
            contract["competition_sources"],
            [guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["competition_source"]],
        )
        self.assertEqual(
            contract["expected_mount_root"],
            guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["mount_root"],
        )
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["local_visual_preflight"]["status"],
            guard.REMOTE_VISUAL_PENDING_STATUS,
        )

    def test_remote_exception_rejects_competition_mount_drift(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        wrong_mount = "/kaggle/input/biohub-cell-tracking-during-development-drift"
        contract["remote_visual_input_route"]["mount_root"] = wrong_mount
        contract["expected_mount_root"] = wrong_mount
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        failures = " ".join(result["failures"])
        self.assertIn("competition route is not the centrally pinned D2K2 route", failures)
        self.assertIn("competition mount root drift", failures)

    def test_remote_exception_rejects_private_dataset_source_or_competition_drift(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        contract["dataset_sources"] = [DATASET]
        contract["competition_sources"] = ["biohub-cell-tracking-during-development-drift"]
        contract["exclusive_dataset_source"] = DATASET
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        failures = " ".join(result["failures"])
        self.assertIn("must declare dataset_sources=[]", failures)
        self.assertIn("competition source drift", failures)
        self.assertIn("must not declare a private dataset source", failures)

    def test_remote_exception_rejects_pinned_metadata_drift(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        self.metadata["competition_sources"] = ["wrong-competition"]
        write_json(self.kernel / "kernel-metadata.json", self.metadata)
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("remote visual-only competition metadata drift", result["failures"])

    def test_contract_supplied_remote_exception_cannot_self_attest(self) -> None:
        contract, _ = self.remote_visual_contract()
        result = self.check_contract(contract, "launch")
        self.assertIn("central exact allowlist", " ".join(result["failures"]))

    def test_remote_exception_rejects_a_synthetic_preflight_pass(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        contract["remote_preflight_attestation"] = {"status": "PASS"}
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("must not use a synthetic remote preflight attestation", " ".join(result["failures"]))

    def test_remote_exception_requires_resource_stop_at_current_lmt_ceiling(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        path = self.root / contract["remote_visual_only"]["resource_stop_receipt"]["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["resource"]["peak_combined_rss_mib"] = 8191.0
        write_json(path, payload)
        self.reseal_remote_exception(contract, allowlist)
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("does not prove the current LMT RSS ceiling", " ".join(result["failures"]))

    def test_remote_exception_rejects_lmt_drift_after_record_freeze(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        self._write_remote_lmt().write_text(
            "# changed LMT\n<!-- LMT_POLICY_JSON_BEGIN -->\n```json\n"
            + json.dumps(
                {
                    "schema": "cell-tracking.autonomy-limits.v1",
                    "local_resource_safety": {"maximum_combined_rss_mib": 9000.0},
                }
            )
            + "\n```\n<!-- LMT_POLICY_JSON_END -->\n",
            encoding="utf-8",
        )
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("remote visual-only current LMT hash drift", " ".join(result["failures"]))

    def test_remote_exception_rejects_noneligible_ship_certificate(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        path = self.root / contract["remote_visual_only"]["ship_kaggle_certificate"]["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({"status": "REJECTED", "eligible": False, "autonomous_launch_eligible": False})
        write_json(path, payload)
        self.reseal_remote_exception(contract, allowlist)
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("ship-kaggle certificate drift: status", " ".join(result["failures"]))

    def test_remote_exception_rejects_certificate_for_different_package_bytes(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        path = self.root / contract["remote_visual_only"]["ship_kaggle_certificate"]["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["package_inventory_sha256"] = "0" * 64
        write_json(path, payload)
        self.reseal_remote_exception(contract, allowlist)
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn(
            "ship-kaggle certificate drift: package_inventory_sha256",
            " ".join(result["failures"]),
        )

    def test_remote_exception_rejects_score_or_training_capability(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        path = self.root / contract["remote_visual_only"]["gpu_visual_only_contract"]["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["prohibitions"]["training"] = True
        write_json(path, payload)
        self.reseal_remote_exception(contract, allowlist)
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("does not prohibit labels/score/training", " ".join(result["failures"]))

    def test_remote_exception_requires_exactly_two_fixed_identities(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        third = "train:44b6_third"
        contract["remote_visual_only"]["fixed_preregistered_visual_identities"].append(third)
        record_path = self.root / contract["remote_visual_only_exception"]["record_path"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["fixed_preregistered_visual_identities"].append(third)
        write_json(record_path, record)
        digest = guard.sha256(record_path)
        contract["remote_visual_only_exception"]["record_sha256"] = digest
        allowlist[contract["remote_visual_only_exception"]["exception_id"]]["record_sha256"] = digest
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("requires exactly two fixed identities", " ".join(result["failures"]))

    def test_remote_exception_cannot_remove_the_remote_review_lock(self) -> None:
        contract, allowlist = self.remote_visual_contract()
        path = self.root / contract["remote_visual_only"]["gpu_visual_only_contract"]["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["post_remote_review"]["score_or_promotion_permitted_before_review"] = True
        write_json(path, payload)
        contract["remote_visual_only"]["post_remote_review"] = payload["post_remote_review"]
        self.reseal_remote_exception(contract, allowlist)
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            result = self.check_contract(contract, "launch")
        self.assertIn("post-review lock drift", " ".join(result["failures"]))


class SuccessorRemoteVisualGuardTest(unittest.TestCase):
    """The fresh CELLECT and Linajea profiles stay distinct and fail closed."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.kernel = self.root / "kernel"
        self.kernel.mkdir()
        policy = {
            "schema": "cell-tracking.autonomy-limits.v1",
            "local_resource_safety": {"maximum_combined_rss_mib": 8192.0},
        }
        (self.root / "LMT.md").write_text(
            "# fixture\n<!-- LMT_POLICY_JSON_BEGIN -->\n```json\n"
            + json.dumps(policy)
            + "\n```\n<!-- LMT_POLICY_JSON_END -->\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_roundtrip(
        self,
        *,
        role: str,
        dataset_id: str,
        version: int,
        mount: str,
        manifest_sha: str,
    ) -> dict:
        path = self.root / "roundtrips" / f"{role}.json"
        path.parent.mkdir(exist_ok=True)
        write_json(
            path,
            {
                "schema": guard.PRIVATE_CARRIER_ROUNDTRIP_SCHEMA,
                "status": "PASS",
                "dataset_id": dataset_id,
                "dataset_version": version,
                "expected_mount_root": mount,
                "carrier_manifest_sha256": manifest_sha,
                "comparison": {
                    "same_declared_paths": True,
                    "same_declared_bytes": True,
                    "same_declared_sha256": True,
                    "raw_or_score_payload_present": False,
                },
            },
        )
        return {"path": str(path.relative_to(self.root)), "sha256": guard.sha256(path)}

    def make_surface(self, profile: str) -> tuple[dict, dict, str]:
        linajea = profile == guard.LINAJEA_LEGACY_GPU_EXECUTION_PROFILE
        p100 = profile == guard.CELLECT_P100_EXECUTION_PROFILE
        roles = (
            guard.LINAJEA_TWO_CARRIER_ROLES
            if linajea
            else guard.CELLECT_P100_CARRIER_ROLES
            if p100
            else guard.CELLECT_TWO_CARRIER_ROLES
        )
        kernel_id = (
            guard.LINAJEA_LEGACY_GPU_KERNEL_ID
            if linajea
            else guard.CELLECT_P100_KERNEL_ID
            if p100
            else guard.CELLECT_TWO_CARRIER_KERNEL_ID
        )
        mode = (
            guard.LINAJEA_LEGACY_GPU_STEP0_MODE
            if linajea
            else guard.CELLECT_P100_STEP0_MODE
            if p100
            else guard.CELLECT_TWO_CARRIER_STEP0_MODE
        )
        dataset_ids = (
            ["onebitbrain/linajea-runtime", "onebitbrain/shared-deps"]
            if linajea
            else [
                "onebitbrain/cellect-assets",
                "onebitbrain/shared-deps",
                "onebitbrain/p100-runtime",
            ]
            if p100
            else ["onebitbrain/cellect-assets", "onebitbrain/shared-deps"]
        )
        mounts = (
            [f"/kaggle/input/datasets/{dataset_id}" for dataset_id in dataset_ids]
            if p100
            else ["/kaggle/input/runtime-or-assets", "/kaggle/input/shared-deps"]
        )
        manifests = [chr(ord("a") + index) * 64 for index in range(len(roles))]
        carriers = []
        for index, role in enumerate(roles):
            version = 2 if index == 0 else 1
            carriers.append(
                {
                    "role": role,
                    "dataset_id": dataset_ids[index],
                    "dataset_version": version,
                    "expected_mount_root": mounts[index],
                    "certifier_mount_name": "carrier_" + str(index),
                    "provides_release_imports": index == 1 or (p100 and index == 2),
                    "manifest_sha256": manifests[index],
                    "dataset_roundtrip_receipt": self.make_roundtrip(
                        role=role,
                        dataset_id=dataset_ids[index],
                        version=version,
                        mount=mounts[index],
                        manifest_sha=manifests[index],
                    ),
                }
            )
        if linajea:
            markers = (
                "verify_opaque_payload_transport\n"
                "LINAJEA_V3_STEP0_ASSIGNMENT_COMPLETE\n"
                "LINAJEA_V3_RELEASE_DEPENDENCY_PROBE_PASS\n"
                "LINAJEA_V3_HEAVY_WORK_BEGIN\nLINAJEA_V3_VISUAL_RELEASE_COMPLETE\n"
                "side_1 side_2 dual_consensus _tracking.mp4\n"
            )
            probe_marker = "LINAJEA_V3_RELEASE_DEPENDENCY_PROBE_PASS"
        elif p100:
            markers = (
                "verify_asset_carrier verify_dependency_carrier "
                "verify_torch_dependency_carrier\n"
                "D2K5_STEP0_ASSIGNMENT_COMPLETE\n"
                "D2K5_V4_SM60_RELEASE_IMPORT_PROBE_COMPLETE\n"
                "D2K5_HEAVY_WORK_BEGIN D2K5_VISUAL_RELEASE_COMPLETE\n"
                "D2K5_V4_NATIVE_SM60_TORCH_BOUND PREDICTIONS.json .mp4\n"
            )
            probe_marker = "D2K5_V4_SM60_RELEASE_IMPORT_PROBE_COMPLETE"
        else:
            markers = (
                "verify_asset_carrier verify_dependency_carrier\n"
                "D2K5_STEP0_ASSIGNMENT_COMPLETE\n"
                "D2K5_V2_RELEASE_IMPORT_PROBE_COMPLETE\n"
                "D2K5_HEAVY_WORK_BEGIN D2K5_VISUAL_RELEASE_COMPLETE\n"
                "PREDICTIONS.json .mp4\n"
            )
            probe_marker = "D2K5_V2_RELEASE_IMPORT_PROBE_COMPLETE"
        competition_aliases = (
            ["/kaggle/input/competitions/biohub-cell-tracking-during-development"]
            if p100
            else [
                "/kaggle/input/biohub-cell-tracking-during-development",
                "/kaggle/input/competitions/biohub-cell-tracking-during-development",
            ]
        )
        source = (
            "# " + markers.replace("\n", "\n# ")
            + "\n"
            + "\n".join(f"# {alias}" for alias in competition_aliases)
            + "\n"
            + "\n".join(f"# {mount} {digest}" for mount, digest in zip(mounts, manifests))
            + "\n"
        )
        code = self.kernel / "kernel.py"
        code.write_text(source, encoding="utf-8")
        metadata = {
            "code_file": "kernel.py",
            "competition_sources": ["biohub-cell-tracking-during-development"],
            "dataset_sources": dataset_ids,
            "enable_gpu": True,
            "enable_internet": False,
            "id": kernel_id,
            "is_private": True,
            "kernel_sources": [],
            "kernel_type": "script",
            "language": "python",
            "title": kernel_id.split("/", 1)[1],
        }
        write_json(self.kernel / "kernel-metadata.json", metadata)
        contract = {
            "kernel_id": kernel_id,
            "remote_preflight_kernel_id": kernel_id,
            "lane_owner": "UNIT_TEST_SUCCESSOR",
            "model_execution": True,
            "execution_profile": profile,
            "remote_preflight_mode": mode,
            "dataset_sources": dataset_ids,
            "competition_sources": ["biohub-cell-tracking-during-development"],
            "kernel_properties": {"enable_gpu": True, "enable_internet": False},
            "expected_mount_root": "/kaggle/input/biohub-cell-tracking-during-development",
            "remote_visual_input_route": {
                "mode": mode,
                "competition_source": "biohub-cell-tracking-during-development",
                "competition_mount_aliases": competition_aliases,
            },
            "private_carriers": carriers,
            "release_dependency_probe_marker": probe_marker,
            "preregistered_visual_identities": list(guard.D2K2_FIXED_IDENTITIES),
            "code_sha256": guard.sha256(code),
            "remote_visual_only_exception": {
                "exception_id": "not-production-allowlisted",
                "record_path": "missing.json",
                "record_sha256": "0" * 64,
            },
        }
        return contract, metadata, source

    def make_enhanced_certificate(self, contract: dict, *, marker: str) -> dict:
        carriers = contract["private_carriers"]
        p100 = contract["execution_profile"] == guard.CELLECT_P100_EXECUTION_PROFILE
        profile = self.root / "profile.json"
        roles = ("tiny", "stratified", "stress", "expansive")
        carrier_imports = ["cv2", "zarr"] + (["torch"] if p100 else [])
        required_imports = ["numpy", "pandas", "tifffile", "tqdm"] if p100 else []
        mounts = []
        if p100:
            for carrier in carriers:
                provider_layout = {
                    "source_type": "dataset",
                    "source_id": carrier["dataset_id"],
                    "roundtrip_receipt": carrier["dataset_roundtrip_receipt"],
                }
                if "mount_layout_correction" in carrier:
                    provider_layout["mount_layout_correction"] = carrier[
                        "mount_layout_correction"
                    ]
                mounts.append(
                    {
                        "name": carrier["certifier_mount_name"],
                        "remote_root": carrier["expected_mount_root"],
                        "provider_layout": provider_layout,
                    }
                )
            competition_source = contract["competition_sources"][0]
            mounts.append(
                {
                    "name": "competition",
                    "remote_root": f"/kaggle/input/competitions/{competition_source}",
                    "provider_layout": {
                        "source_type": "competition",
                        "source_id": competition_source,
                    },
                }
            )
        write_json(
            profile,
            {
                **({"mounts": mounts} if p100 else {}),
                **(
                    {
                        "kernel": {
                            "expected_metadata": json.loads(
                                (self.kernel / "kernel-metadata.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                        }
                    }
                    if p100
                    else {}
                ),
                "environment": {
                    **({"python_version_prefix": "3.12"} if p100 else {}),
                    "required_imports": required_imports,
                    "carrier_provided_imports": carrier_imports,
                    "declared_release_imports": required_imports + carrier_imports,
                    "release_dependency_probe": {
                        "stdout_marker": marker,
                        "carrier_mounts": [
                            item["certifier_mount_name"]
                            for item in carriers
                            if item["provides_release_imports"]
                        ],
                    },
                },
                "scenarios": [
                    {"role": role, "stdout_markers": [marker]} for role in roles
                ],
            },
        )
        runs = [
            {
                "status": "PASS",
                "role": role,
                "stdout_tail": marker,
                "upload_materialization": {"transport": "kaggle-cli-save-kernel.v1"},
            }
            for role in roles
            for _ in range(2)
        ]
        cert = self.root / "certificate.json"
        write_json(
            cert,
            {
                "schema": "ship-kaggle-kernels.certification.v1",
                "status": "ELIGIBLE",
                "eligible": True,
                "autonomous_launch_eligible": True,
                "kernel_id": contract["kernel_id"],
                "limits_file": str((self.root / "LMT.md").resolve()),
                "limits_sha256": guard.sha256(self.root / "LMT.md"),
                "artifact_sha256": guard.sha256(self.kernel / "kernel.py"),
                "metadata_sha256": guard.sha256(self.kernel / "kernel-metadata.json"),
                "package_inventory_sha256": guard._ship_package_inventory_sha256(self.kernel),
                "profile_path": str(profile.resolve()),
                "profile_sha256": guard.sha256(profile),
                "receipt_content_sha256": "d" * 64,
                "quick_debug": {
                    "auto_triggered": True,
                    "name": "focused_pre_push_release_debug",
                    "purpose": (
                        "exercise the exact Kaggle CLI script/notebook materialization: only the "
                        "declared entrypoint is present beside the fresh mounts, so undeclared local "
                        "package siblings must fail before a remote version is spent"
                    ),
                    "status": "PASS",
                    "transport": "kaggle-cli-save-kernel.v1",
                },
                "static": {"status": "PASS", "failures": [], "warnings": []},
                "repeatability": {"status": "PASS"},
                "runtime_projection": {
                    "status": "PASS",
                    "physical_memory_limit_bytes": 29_000_000_000,
                    "projected_peak_rss_bytes": 28_000_000_000,
                },
                "simulation_runs": runs,
                "adversarial_runs": [
                    {"fault": fault, "status": "PASS"}
                    for fault in (
                        "missing_mount", "missing_required_file",
                        "corrupt_required_file", "duplicate_basename",
                    )
                ],
            },
        )
        return {"path": str(cert.relative_to(self.root)), "sha256": guard.sha256(cert)}

    def make_successor_exception(
        self,
        profile: str,
    ) -> tuple[dict, dict[str, dict[str, str]]]:
        contract, _, source = self.make_surface(profile)
        linajea = profile == guard.LINAJEA_LEGACY_GPU_EXECUTION_PROFILE
        p100 = profile == guard.CELLECT_P100_EXECUTION_PROFILE
        if linajea:
            transport = self.root / "evidence/runtime-transport.json"
            transport.parent.mkdir(exist_ok=True)
            payload_sha = "c" * 64
            write_json(
                transport,
                {
                    "schema": "celltrack-alpha.linajea-zf-tf1-gpu-runtime-dataset.v2",
                    "status": "BUILT_LOCAL_NO_GPU_FORWARD",
                    "contains_competition_raw": False,
                    "transport": "opaque_gzip_tar_payload_r3",
                    "transport_revision": "R3",
                    "dataset_files_contract": [
                        "CARRIER_DATASET_MANIFEST.json",
                        "linajea_zf_tf115_gpu_v3_runtime.payload",
                    ],
                    "payload": {
                        "path": "linajea_zf_tf115_gpu_v3_runtime.payload",
                        "encoding": "gzip+tar",
                        "archive_name_autodetection_prevented": True,
                        "bytes": 123,
                        "sha256": payload_sha,
                    },
                },
            )
            transport_sha = guard.sha256(transport)
            first = contract["private_carriers"][0]
            old_manifest = first["manifest_sha256"]
            first["manifest_sha256"] = transport_sha
            roundtrip_path = self.root / first["dataset_roundtrip_receipt"]["path"]
            roundtrip = json.loads(roundtrip_path.read_text(encoding="utf-8"))
            roundtrip["carrier_manifest_sha256"] = transport_sha
            write_json(roundtrip_path, roundtrip)
            first["dataset_roundtrip_receipt"]["sha256"] = guard.sha256(roundtrip_path)
            code = self.kernel / "kernel.py"
            code.write_text(source.replace(old_manifest, transport_sha) + f"# {payload_sha}\n", encoding="utf-8")
            contract["code_sha256"] = guard.sha256(code)

        certificate_ref = self.make_enhanced_certificate(
            contract, marker=contract["release_dependency_probe_marker"]
        )
        approval_path = self.root / "evidence/PRIVATE_DATASET_EXTERNAL_ACTION_APPROVAL.json"
        approval_path.parent.mkdir(exist_ok=True)
        newly_created_roles = (
            {"p100_torch_runtime"} if p100 else {"cp312_release_dependencies"}
        )
        if linajea:
            newly_created_roles.add("tf1_runtime_opaque")
        write_json(
            approval_path,
            {
                "schema": guard.PRIVATE_DATASET_EXTERNAL_ACTION_APPROVAL_SCHEMA,
                "status": "APPROVED_AND_EXECUTED",
                "approved_targets": [
                    {
                        "dataset_id": carrier["dataset_id"],
                        "dataset_version": carrier["dataset_version"],
                        "roundtrip_receipt_path": carrier["dataset_roundtrip_receipt"]["path"],
                        "roundtrip_receipt_sha256": carrier["dataset_roundtrip_receipt"]["sha256"],
                    }
                    for carrier in contract["private_carriers"]
                    if carrier["role"] in newly_created_roles
                ],
                "scope_exclusions": [
                    "competition submission",
                    "scoring",
                    "training or fine-tuning",
                    "raw competition data egress",
                ],
            },
        )
        approval_ref = {
            "path": str(approval_path.relative_to(self.root)),
            "sha256": guard.sha256(approval_path),
        }
        identities = list(guard.D2K2_FIXED_IDENTITIES)
        if linajea:
            post_review = {
                "remote_mp4_retrieval_required": True,
                "local_visual_review_required": True,
                "score_or_promotion_permitted_before_review": False,
                "remote_review_receipt_schema": guard.REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA,
                "required_remote_mp4_artifacts": [
                    {
                        "identity": identity,
                        "view": view,
                        "relative_path": identity.split(":", 1)[1] + "/" + view + ".mp4",
                    }
                    for identity in identities
                    for view in guard.LINAJEA_REQUIRED_VIDEO_VIEWS
                ],
            }
        else:
            post_review = {
                "remote_mp4_retrieval_required": True,
                "local_visual_review_required": True,
                "score_or_promotion_permitted_before_review": False,
                "remote_review_receipt_schema": guard.REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA,
                "required_remote_mp4_identities": identities,
            }
            if p100:
                post_review["full_decode_required"] = True
        evidence = self.root / "evidence"
        evidence.mkdir(exist_ok=True)
        gpu_path = evidence / "GPU_VISUAL_ONLY_CONTRACT.json"
        write_json(
            gpu_path,
            {
                "schema": guard.REMOTE_VISUAL_ONLY_CONTRACT_SCHEMA,
                "status": "FROZEN",
                "candidate_id": (
                    "LINAJEA-ZF-7F-BACKVEC-DUALCONS-v3"
                    if linajea else "CELLECT-NATIVE-3HEAD-X3RD-v2"
                ),
                "kernel_id": contract["kernel_id"],
                "execution_profile": profile,
                "model_execution": True,
                "device": "gpu",
                "fixed_preregistered_visual_identities": identities,
                "remote_memory_limit_bytes": 29_000_000_000,
                "prohibitions": {
                    key: False
                    for key in (
                        "labels", "targets", "scorer", "full38", "training",
                        "tuning", "calibration", "hpo", "submission",
                    )
                },
                "post_remote_review": post_review,
            },
        )
        gpu_ref = {"path": str(gpu_path.relative_to(self.root)), "sha256": guard.sha256(gpu_path)}
        remote = {
            "schema": guard.REMOTE_VISUAL_ONLY_PRELAUNCH_SCHEMA,
            "candidate_id": json.loads(gpu_path.read_text(encoding="utf-8"))["candidate_id"],
            "kernel_id": contract["kernel_id"],
            "execution_profile": profile,
            "model_execution": True,
            "candidate_code_path": "kernel/kernel.py",
            "candidate_code_sha256": contract["code_sha256"],
            "kernel_package_tree_sha256": guard._package_inventory(self.kernel, self.root)["tree_sha256"],
            "fixed_preregistered_visual_identities": identities,
            "private_carriers": contract["private_carriers"],
            "private_dataset_external_action_approval": approval_ref,
            "gpu_visual_only_contract": gpu_ref,
            "ship_kaggle_certificate": certificate_ref,
            "post_remote_review": post_review,
        }
        if p100:
            dockerfile = evidence / "Dockerfile.certifier"
            dockerfile.write_text("FROM python:3.12\n", encoding="utf-8")
            certificate_path = self.root / certificate_ref["path"]
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
            interpreter_path = evidence / "LOCAL_CP312_CERTIFIER_EXECUTION_RECEIPT.json"
            write_json(
                interpreter_path,
                {
                    "schema": "celltrack-alpha.local-certifier-interpreter-execution.v1",
                    "status": "PASS",
                    "container": {
                        "image": "unit-cp312:local",
                        "image_id": "sha256:" + "e" * 64,
                        "platform_requested": "linux/amd64",
                        "dockerfile_path": str(dockerfile.relative_to(self.root)),
                        "dockerfile_sha256": guard.sha256(dockerfile),
                    },
                    "interpreter": {
                        "executable": "/usr/local/bin/python",
                        "version": "3.12.14",
                        "implementation": "CPython",
                        "platform": "Linux-unit-x86_64",
                        "machine": "x86_64",
                    },
                    "certification": {
                        "profile_path": str(
                            Path(certificate["profile_path"])
                            .resolve()
                            .relative_to(self.root.resolve())
                        ),
                        "profile_sha256": certificate["profile_sha256"],
                        "receipt_path": certificate_ref["path"],
                        "receipt_sha256": certificate_ref["sha256"],
                        "receipt_content_sha256": certificate[
                            "receipt_content_sha256"
                        ],
                        "static_status": "PASS",
                        "static_failures": [],
                        "simulation_runs_passed": 8,
                        "adversarial_faults_passed": 4,
                    },
                },
            )
            remote["certifier_interpreter_receipt"] = {
                "path": str(interpreter_path.relative_to(self.root)),
                "sha256": guard.sha256(interpreter_path),
            }
        if linajea:
            cpu_path = evidence / "CPU_LAYOUT_FAILURE.json"
            write_json(
                cpu_path,
                {
                    "schema": "celltrack-alpha.linajea-zf-cpu-v2-terminal.v1",
                    "status": "PARK_CPU_LAYOUT_INCOMPATIBLE",
                    "builder_exit_code": 1,
                    "terminal_signature": (
                        "Conv3DBackpropInputOpV2 only supports NDHWC on the CPU "
                        "at frozen channels-first decoder"
                    ),
                    "interpretation": "Platform evidence only; not model-quality evidence.",
                },
            )
            remote["cpu_layout_failure_receipt"] = {
                "path": str(cpu_path.relative_to(self.root)),
                "sha256": guard.sha256(cpu_path),
            }
            transport = self.root / "evidence/runtime-transport.json"
            remote["runtime_transport_manifest"] = {
                "path": str(transport.relative_to(self.root)),
                "sha256": guard.sha256(transport),
            }
        else:
            resource_path = evidence / "RESOURCE_STOP.json"
            write_json(
                resource_path,
                {
                    "schema": "celltrack-alpha.local-model-execution-abort.v1",
                    "status": "ABORT",
                    "failure_type": "ResourceStop",
                    "pass_receipt_absent": True,
                    "owned_process_group_terminated": True,
                    "resource": {"peak_combined_rss_mib": 8882.0},
                },
            )
            remote["resource_stop_receipt"] = {
                "path": str(resource_path.relative_to(self.root)),
                "sha256": guard.sha256(resource_path),
            }
        contract["remote_visual_only"] = remote
        exception_id = "unit-linajea-successor" if linajea else "unit-cellect-successor"
        record_path = evidence / "REMOTE_VISUAL_ONLY_EXCEPTION.json"
        record = {
            "schema": guard.REMOTE_VISUAL_ONLY_EXCEPTION_SCHEMA,
            "status": "APPROVED",
            "exception_id": exception_id,
            "candidate_id": remote["candidate_id"],
            "kernel_id": contract["kernel_id"],
            "execution_profile": profile,
            "fixed_preregistered_visual_identities": identities,
            "private_carriers": contract["private_carriers"],
            "private_dataset_external_action_approval": approval_ref,
            "gpu_visual_only_contract": gpu_ref,
            "ship_kaggle_certificate": certificate_ref,
            "post_remote_review": post_review,
            "lmt": {"path": "LMT.md", "sha256": guard.sha256(self.root / "LMT.md")},
        }
        for key in (
            "certifier_interpreter_receipt",
            "cpu_layout_failure_receipt",
            "runtime_transport_manifest",
            "resource_stop_receipt",
        ):
            if key in remote:
                record[key] = remote[key]
        write_json(record_path, record)
        record_sha = guard.sha256(record_path)
        contract["remote_visual_only_exception"] = {
            "exception_id": exception_id,
            "record_path": str(record_path.relative_to(self.root)),
            "record_sha256": record_sha,
        }
        allowlist = {
            exception_id: {
                "record_path": contract["remote_visual_only_exception"]["record_path"],
                "record_sha256": record_sha,
                "kernel_id": contract["kernel_id"],
                "execution_profile": profile,
            }
        }
        return contract, allowlist

    def test_cellect_two_carrier_surface_passes(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.CELLECT_TWO_CARRIER_EXECUTION_PROFILE
        )
        line, failures = guard.check_successor_remote_visual_surface(
            contract=contract,
            metadata=metadata,
            source=source,
            project_root=self.root,
        )
        self.assertIsNotNone(line)
        self.assertEqual(failures, [])

    def test_cellect_two_carrier_surface_rejects_missing_dependency_carrier(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.CELLECT_TWO_CARRIER_EXECUTION_PROFILE
        )
        contract["private_carriers"].pop()
        _, failures = guard.check_successor_remote_visual_surface(
            contract=contract, metadata=metadata, source=source, project_root=self.root
        )
        self.assertIn("exactly two private carriers", " ".join(failures))

    def test_cellect_p100_surface_requires_three_canonical_provider_roots(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.CELLECT_P100_EXECUTION_PROFILE
        )
        line, failures = guard.check_successor_remote_visual_surface(
            contract=contract,
            metadata=metadata,
            source=source,
            project_root=self.root,
        )
        self.assertIsNotNone(line)
        self.assertEqual(failures, [])

        contract["private_carriers"][2]["expected_mount_root"] = (
            "/kaggle/input/p100-runtime"
        )
        _, failures = guard.check_successor_remote_visual_surface(
            contract=contract,
            metadata=metadata,
            source=source,
            project_root=self.root,
        )
        self.assertIn("canonical owner-qualified Dataset root", " ".join(failures))

    def test_cellect_p100_surface_rejects_legacy_competition_alias(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.CELLECT_P100_EXECUTION_PROFILE
        )
        source += "\n# /kaggle/input/biohub-cell-tracking-during-development\n"
        _, failures = guard.check_successor_remote_visual_surface(
            contract=contract,
            metadata=metadata,
            source=source,
            project_root=self.root,
        )
        self.assertIn("legacy unqualified competition root", " ".join(failures))

    def test_private_carriers_must_name_a_release_import_provider(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.CELLECT_TWO_CARRIER_EXECUTION_PROFILE
        )
        for carrier in contract["private_carriers"]:
            carrier["provides_release_imports"] = False
        _, failures = guard.check_successor_remote_visual_surface(
            contract=contract, metadata=metadata, source=source, project_root=self.root
        )
        self.assertIn("no release-import provider", " ".join(failures))

    def test_linajea_surface_requires_runtime_dataset_version_two(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.LINAJEA_LEGACY_GPU_EXECUTION_PROFILE
        )
        line, failures = guard.check_successor_remote_visual_surface(
            contract=contract, metadata=metadata, source=source, project_root=self.root
        )
        self.assertIsNotNone(line)
        self.assertEqual(failures, [])
        contract["private_carriers"][0]["dataset_version"] = 1
        _, failures = guard.check_successor_remote_visual_surface(
            contract=contract, metadata=metadata, source=source, project_root=self.root
        )
        self.assertIn("Dataset Version 2", " ".join(failures))

    def test_linajea_v7_surface_requires_canonical_provider_roots_and_hardening(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.LINAJEA_LEGACY_GPU_EXECUTION_PROFILE
        )
        contract["kernel_id"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID
        contract["remote_preflight_kernel_id"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID
        metadata["id"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID
        metadata["title"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID.split("/", 1)[1]
        write_json(self.kernel / "kernel-metadata.json", metadata)
        contract["remote_visual_input_route"]["competition_mount_aliases"] = [
            "/kaggle/input/competitions/biohub-cell-tracking-during-development"
        ]
        source = source.replace(
            "# /kaggle/input/biohub-cell-tracking-during-development\n", ""
        )
        source += (
            '\n# "nvidia-smi" "-i" validated_inherited_library_paths\n'
            "# LINAJEA_V3_CV2_CODEC_PROBE_PASS cv2.VideoWriter "
            "cv2.VideoCapture full_decode_mp4\n"
            '# "full_decode_verified": True\n'
        )
        for carrier in contract["private_carriers"]:
            old_mount = carrier["expected_mount_root"]
            new_mount = f"/kaggle/input/datasets/{carrier['dataset_id']}"
            carrier["expected_mount_root"] = new_mount
            source = source.replace(old_mount, new_mount)
            roundtrip_path = self.root / carrier["dataset_roundtrip_receipt"]["path"]
            roundtrip = json.loads(roundtrip_path.read_text(encoding="utf-8"))
            roundtrip["expected_mount_root"] = new_mount
            write_json(roundtrip_path, roundtrip)
            carrier["dataset_roundtrip_receipt"]["sha256"] = guard.sha256(
                roundtrip_path
            )
        line, failures = guard.check_successor_remote_visual_surface(
            contract=contract,
            metadata=metadata,
            source=source,
            project_root=self.root,
        )
        self.assertIsNotNone(line)
        self.assertEqual(failures, [])

        contract["private_carriers"][0]["expected_mount_root"] = (
            "/kaggle/input/linajea-runtime"
        )
        _, failures = guard.check_successor_remote_visual_surface(
            contract=contract,
            metadata=metadata,
            source=source,
            project_root=self.root,
        )
        self.assertIn("canonical owner-qualified Dataset root", " ".join(failures))

    def test_linajea_v7_surface_rejects_unproved_imageio_codec_path(self) -> None:
        contract, metadata, source = self.make_surface(
            guard.LINAJEA_LEGACY_GPU_EXECUTION_PROFILE
        )
        contract["kernel_id"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID
        contract["remote_preflight_kernel_id"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID
        metadata["id"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID
        metadata["title"] = guard.LINAJEA_PROVIDER_ROOT_KERNEL_ID.split("/", 1)[1]
        contract["remote_visual_input_route"]["competition_mount_aliases"] = [
            "/kaggle/input/competitions/biohub-cell-tracking-during-development"
        ]
        source = source.replace(
            "# /kaggle/input/biohub-cell-tracking-during-development\n", ""
        )
        source += (
            '\n# "nvidia-smi" "-i" validated_inherited_library_paths\n'
            "# LINAJEA_V3_CV2_CODEC_PROBE_PASS cv2.VideoWriter "
            "cv2.VideoCapture full_decode_mp4\n"
            '# "full_decode_verified": True imageio libx264\n'
        )
        for carrier in contract["private_carriers"]:
            old_mount = carrier["expected_mount_root"]
            new_mount = f"/kaggle/input/datasets/{carrier['dataset_id']}"
            carrier["expected_mount_root"] = new_mount
            source = source.replace(old_mount, new_mount)
            receipt_path = self.root / carrier["dataset_roundtrip_receipt"]["path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["expected_mount_root"] = new_mount
            write_json(receipt_path, receipt)
            carrier["dataset_roundtrip_receipt"]["sha256"] = guard.sha256(
                receipt_path
            )
        _, failures = guard.check_successor_remote_visual_surface(
            contract=contract,
            metadata=metadata,
            source=source,
            project_root=self.root,
        )
        self.assertIn("unproved imageio/libx264", " ".join(failures))

    def test_linajea_review_lock_is_exactly_six_mp4s(self) -> None:
        identities = list(guard.D2K2_FIXED_IDENTITIES)
        value = {
            "remote_mp4_retrieval_required": True,
            "local_visual_review_required": True,
            "score_or_promotion_permitted_before_review": False,
            "remote_review_receipt_schema": guard.REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA,
            "required_remote_mp4_artifacts": [
                {
                    "identity": identity,
                    "view": view,
                    "relative_path": identity.split(":", 1)[1] + "/" + view + ".mp4",
                }
                for identity in identities
                for view in guard.LINAJEA_REQUIRED_VIDEO_VIEWS
            ],
        }
        result = guard._linajea_remote_review_lock(value, identities=identities)
        self.assertEqual(len(result["required_remote_mp4_artifacts"]), 6)
        value["required_remote_mp4_artifacts"].pop()
        with self.assertRaisesRegex(guard.GuardError, "six-MP4"):
            guard._linajea_remote_review_lock(value, identities=identities)

    def test_enhanced_certificate_binds_release_import_probe_to_all_runs(self) -> None:
        contract, _, _ = self.make_surface(guard.CELLECT_TWO_CARRIER_EXECUTION_PROFILE)
        reference = self.make_enhanced_certificate(
            contract, marker=contract["release_dependency_probe_marker"]
        )
        artifacts, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertEqual(failures, [])
        self.assertEqual({item["role"] for item in artifacts}, {
            "enhanced_ship_kaggle_certificate", "enhanced_ship_kaggle_profile"
        })

        cert_path = self.root / reference["path"]
        certificate = json.loads(cert_path.read_text(encoding="utf-8"))
        certificate["simulation_runs"][0]["stdout_tail"] = "probe skipped"
        write_json(cert_path, certificate)
        reference["sha256"] = guard.sha256(cert_path)
        _, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertIn("missing from simulation run 0", " ".join(failures))

    def test_p100_enhanced_certificate_binds_all_metadata_provider_roots(self) -> None:
        contract, _, _ = self.make_surface(guard.CELLECT_P100_EXECUTION_PROFILE)
        reference = self.make_enhanced_certificate(
            contract, marker=contract["release_dependency_probe_marker"]
        )
        artifacts, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertEqual(failures, [])
        self.assertEqual(
            {item["role"] for item in artifacts},
            {"enhanced_ship_kaggle_certificate", "enhanced_ship_kaggle_profile"},
        )

        certificate_path = self.root / reference["path"]
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        profile_path = Path(certificate["profile_path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["mounts"][2]["provider_layout"]["source_id"] = (
            "onebitbrain/wrong-runtime"
        )
        write_json(profile_path, profile)
        certificate["profile_sha256"] = guard.sha256(profile_path)
        write_json(certificate_path, certificate)
        reference["sha256"] = guard.sha256(certificate_path)
        _, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertIn("provider provenance drift", " ".join(failures))

    def test_p100_enhanced_certificate_rejects_non_kaggle_python_abi(self) -> None:
        contract, _, _ = self.make_surface(guard.CELLECT_P100_EXECUTION_PROFILE)
        reference = self.make_enhanced_certificate(
            contract, marker=contract["release_dependency_probe_marker"]
        )
        certificate_path = self.root / reference["path"]
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        profile_path = Path(certificate["profile_path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["environment"]["python_version_prefix"] = "3.14"
        write_json(profile_path, profile)
        certificate["profile_sha256"] = guard.sha256(profile_path)
        write_json(certificate_path, certificate)
        reference["sha256"] = guard.sha256(certificate_path)
        _, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertIn("Kaggle CPython 3.12", " ".join(failures))

    def test_p100_enhanced_certificate_requires_exact_metadata(self) -> None:
        contract, _, _ = self.make_surface(guard.CELLECT_P100_EXECUTION_PROFILE)
        reference = self.make_enhanced_certificate(
            contract, marker=contract["release_dependency_probe_marker"]
        )
        certificate_path = self.root / reference["path"]
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        profile_path = Path(certificate["profile_path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["kernel"]["expected_metadata"].pop("is_private")
        write_json(profile_path, profile)
        certificate["profile_sha256"] = guard.sha256(profile_path)
        write_json(certificate_path, certificate)
        reference["sha256"] = guard.sha256(certificate_path)
        _, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertIn("exact kernel metadata", " ".join(failures))

    def test_p100_enhanced_certificate_requires_native_import_closure(self) -> None:
        contract, _, _ = self.make_surface(guard.CELLECT_P100_EXECUTION_PROFILE)
        reference = self.make_enhanced_certificate(
            contract, marker=contract["release_dependency_probe_marker"]
        )
        certificate_path = self.root / reference["path"]
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        profile_path = Path(certificate["profile_path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["environment"]["required_imports"].remove("pandas")
        profile["environment"]["declared_release_imports"].remove("pandas")
        write_json(profile_path, profile)
        certificate["profile_sha256"] = guard.sha256(profile_path)
        write_json(certificate_path, certificate)
        reference["sha256"] = guard.sha256(certificate_path)
        _, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertIn("native base dependency closure", " ".join(failures))

    def test_p100_enhanced_certificate_rejects_legacy_competition_mount(self) -> None:
        contract, _, _ = self.make_surface(guard.CELLECT_P100_EXECUTION_PROFILE)
        reference = self.make_enhanced_certificate(
            contract, marker=contract["release_dependency_probe_marker"]
        )
        certificate_path = self.root / reference["path"]
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        profile_path = Path(certificate["profile_path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["mounts"][-1]["remote_root"] = (
            "/kaggle/input/biohub-cell-tracking-during-development"
        )
        write_json(profile_path, profile)
        certificate["profile_sha256"] = guard.sha256(profile_path)
        write_json(certificate_path, certificate)
        reference["sha256"] = guard.sha256(certificate_path)
        _, failures = guard._validate_enhanced_ship_certificate(
            reference,
            contract=contract,
            kernel_dir=self.kernel,
            code_sha256=contract["code_sha256"],
            carriers=contract["private_carriers"],
            project_root=self.root,
        )
        self.assertIn("competition provider root/provenance drift", " ".join(failures))

    def test_external_action_approval_must_bind_the_exact_roundtrip(self) -> None:
        contract, _ = self.make_successor_exception(
            guard.CELLECT_TWO_CARRIER_EXECUTION_PROFILE
        )
        reference = contract["remote_visual_only"][
            "private_dataset_external_action_approval"
        ]
        approval_path = self.root / reference["path"]
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        approval["approved_targets"][0]["roundtrip_receipt_sha256"] = "f" * 64
        write_json(approval_path, approval)
        resealed_reference = {
            "path": reference["path"],
            "sha256": guard.sha256(approval_path),
        }
        with self.assertRaisesRegex(guard.GuardError, "does not bind"):
            guard._validate_private_dataset_external_action_approval(
                resealed_reference,
                carriers=contract["private_carriers"],
                linajea=False,
                project_root=self.root,
            )

    def test_successor_contract_cannot_self_allowlist_through_check_kernel(self) -> None:
        contract, _, _ = self.make_surface(guard.CELLECT_TWO_CARRIER_EXECUTION_PROFILE)
        result = guard.check_kernel(
            self.kernel,
            contract,
            {},
            "launch",
            project_root=self.root,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("central exact allowlist", " ".join(result["failures"]))

    def test_cellect_full_successor_exception_requires_resource_stop_and_enhanced_cert(self) -> None:
        contract, allowlist = self.make_successor_exception(
            guard.CELLECT_TWO_CARRIER_EXECUTION_PROFILE
        )
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            summary, failures = guard.check_successor_remote_visual_exception(
                contract=contract,
                code_sha256=contract["code_sha256"],
                code_path=self.kernel / "kernel.py",
                kernel_dir=self.kernel,
                project_root=self.root,
            )
            integrated = guard.check_kernel(
                self.kernel, contract, {}, "launch", project_root=self.root
            )
        self.assertEqual(failures, [])
        self.assertEqual(summary["status"], guard.REMOTE_VISUAL_PENDING_STATUS)
        self.assertIn("local_resource_stop_receipt", {item["role"] for item in summary["artifacts"]})
        self.assertEqual(integrated["status"], "PASS", integrated["failures"])

    def test_cellect_p100_full_successor_exception_passes_canonical_provider_gate(self) -> None:
        contract, allowlist = self.make_successor_exception(
            guard.CELLECT_P100_EXECUTION_PROFILE
        )
        with mock.patch.dict(
            guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST,
            allowlist,
            clear=True,
        ):
            summary, failures = guard.check_successor_remote_visual_exception(
                contract=contract,
                code_sha256=contract["code_sha256"],
                code_path=self.kernel / "kernel.py",
                kernel_dir=self.kernel,
                project_root=self.root,
            )
            integrated = guard.check_kernel(
                self.kernel, contract, {}, "launch", project_root=self.root
            )
        self.assertEqual(failures, [])
        self.assertEqual(summary["status"], guard.REMOTE_VISUAL_PENDING_STATUS)
        self.assertEqual(integrated["status"], "PASS", integrated["failures"])

    def test_cellect_p100_exception_requires_full_mp4_decode_lock(self) -> None:
        identities = list(guard.D2K2_FIXED_IDENTITIES)
        with self.assertRaisesRegex(guard.GuardError, "full_decode_required"):
            guard._remote_review_lock(
                {
                    "remote_mp4_retrieval_required": True,
                    "local_visual_review_required": True,
                    "score_or_promotion_permitted_before_review": False,
                    "remote_review_receipt_schema": (
                        guard.REMOTE_VISUAL_REVIEW_RECEIPT_SCHEMA
                    ),
                    "required_remote_mp4_identities": identities,
                },
                identities=identities,
                label="CELLECT P100 successor post-review lock",
                require_full_decode=True,
            )

    def test_linajea_full_successor_exception_binds_cpu_layout_opaque_v2_and_six_mp4s(self) -> None:
        contract, allowlist = self.make_successor_exception(
            guard.LINAJEA_LEGACY_GPU_EXECUTION_PROFILE
        )
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            summary, failures = guard.check_successor_remote_visual_exception(
                contract=contract,
                code_sha256=contract["code_sha256"],
                code_path=self.kernel / "kernel.py",
                kernel_dir=self.kernel,
                project_root=self.root,
            )
            integrated = guard.check_kernel(
                self.kernel, contract, {}, "launch", project_root=self.root
            )
        self.assertEqual(failures, [])
        self.assertEqual(summary["status"], guard.REMOTE_VISUAL_PENDING_STATUS)
        self.assertEqual(len(summary["post_remote_review"]["required_remote_mp4_artifacts"]), 6)
        roles = {item["role"] for item in summary["artifacts"]}
        self.assertIn("cpu_layout_failure_receipt", roles)
        self.assertIn("opaque_runtime_transport_manifest", roles)
        self.assertEqual(integrated["status"], "PASS", integrated["failures"])


class ProductionTrustedRunnerRegistryTest(unittest.TestCase):
    EXPECTED = {
        "cellect-native-local-runner-v1": {
            "path": (
                "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/runner/"
                "implementation/cellect_native_local_runner.py"
            ),
            "sha256": "fa483b8db9ff2ea587e9e474a0c7e2444ce69dcb371dac74e5799707f64985e5",
        },
        "ot2-v3-local-runner-v1": {
            "path": (
                "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
                "ot2_v3_runtime/implementation/ot2_v3_local_runner.py"
            ),
            "sha256": "f961d266bdc94e2aa8a1c73c6b42a740d72947bfd8242e26f6de0db2327b3f36",
        },
        "f2-v32-child-local-runner-v1": {
            "path": (
                "dag_prompts/2026-08-20-alpha-funnel-v1/runs/"
                "f2_v32_child_local_visual_v1/runner/f2_v32_child_local_runner.py"
            ),
            "sha256": "ee222bf8a1db6ce933811bf8b8579d94dea7972889f8f8824904b540ce357373",
        },
    }

    def test_exact_independently_reviewed_membership(self) -> None:
        self.assertEqual(guard.TRUSTED_LOCAL_RUNNERS, self.EXPECTED)
        self.assertEqual(len(guard.TRUSTED_LOCAL_RUNNERS), 3)
        self.assertNotIn("v32-prethreshold-local-runner-v1", guard.TRUSTED_LOCAL_RUNNERS)

    def test_registered_physical_runner_hashes_are_exact(self) -> None:
        for entry in guard.TRUSTED_LOCAL_RUNNERS.values():
            path = guard.PROJECT_ROOT / entry["path"]
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(guard.sha256(path), entry["sha256"])

    def test_production_remote_visual_exception_allowlist_is_exactly_reviewed_amendments(self) -> None:
        self.assertEqual(
            guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST,
            {
                "cellect-d2k2-remote-visual-only-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k2_kaggle_visual/REMOTE_VISUAL_ONLY_EXCEPTION.json"
                    ),
                    "record_sha256": (
                        "c227291a53ede50742058adbdae26f70b3d568128a275b09a5468e49f7e8722b"
                    ),
                    "kernel_id": "onebitbrain/g02-cellect-visual-only-v2",
                    "execution_profile": "remote_visual_only_resource_stop.v1",
                },
                "cellect-d2k3-remote-visual-only-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k3_kaggle_visual/REMOTE_VISUAL_ONLY_EXCEPTION.json"
                    ),
                    "record_sha256": (
                        "84f7e4ba2f3710c43e20162c98ae6195b143cdfee2261fc19cff0e280db9765d"
                    ),
                    "kernel_id": "onebitbrain/g02-cellect-visual-only-v3",
                    "execution_profile": "remote_visual_only_resource_stop.v1",
                },
                "cellect-d2k5-remote-visual-only-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k5_gpu_visual/REMOTE_VISUAL_ONLY_EXCEPTION.json"
                    ),
                    "record_sha256": (
                        "eb0ddf68b89a0bdabf37cdf22cd663b313e1b17cc2a9d8cb6b4a11d9db0eb4f2"
                    ),
                    "kernel_id": "onebitbrain/g02-cellect-native-gpu-visual-v1",
                    "execution_profile": "remote_visual_only_resource_stop.v1",
                },
                "cellect-d2k5-v2-two-carrier-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k5_gpu_visual/v2_successor/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "02053872a9ea66ed695031368d245cf3f2d778daee45a06a36ff35ba7b19addf"
                    ),
                    "kernel_id": "onebitbrain/g02-cellect-native-gpu-visual-v2",
                    "execution_profile": (
                        "cellect_remote_visual_two_carrier_release_probe.v1"
                    ),
                },
                "cellect-d2k5-v3-mount-repair-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k5_gpu_visual/v3_mount_root/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "4efce8b0b65973f40b114a9c6d0dff85bad57b4957d7a5c420ee1f88fd45c240"
                    ),
                    "kernel_id": "onebitbrain/g02-cellect-native-gpu-visual-v3",
                    "execution_profile": (
                        "cellect_remote_visual_two_carrier_release_probe.v1"
                    ),
                },
                "cellect-d2k5-v4-p100-three-carrier-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k5_gpu_visual/v4_p100_sm60/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "40b0b26d37e0527842ed9b67081c8f6d86975c22e79243f3078d060c59b2927c"
                    ),
                    "kernel_id": "onebitbrain/g02-cellect-native-gpu-visual-v4-p100",
                    "execution_profile": (
                        "cellect_remote_visual_three_carrier_p100_release_probe.v1"
                    ),
                },
                "cellect-d2k5-v5-p100-monitor-reap-three-carrier-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k5_gpu_visual/v5_p100_monitor_reap/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "cc2814faf6372ff4903aa375f8fb01ead264e924331d1f9eb2e5008e62cd585e"
                    ),
                    "kernel_id": (
                        "onebitbrain/g02-cellect-native-gpu-visual-v5-p100-reap"
                    ),
                    "execution_profile": (
                        "cellect_remote_visual_three_carrier_p100_release_probe.v1"
                    ),
                },
                "cellect-d2k5-v6-p100-typed-terminal-reap-three-carrier-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "division/d2k5_gpu_visual/v6_p100_typed_terminal_reap/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "41f842e80da54089cd8e6e46b7ee9be453fafaca972fadf3a9a7118f386d6763"
                    ),
                    "kernel_id": (
                        "onebitbrain/g02-cellect-native-gpu-visual-v6-p100-typed-reap"
                    ),
                    "execution_profile": (
                        "cellect_remote_visual_three_carrier_p100_release_probe.v1"
                    ),
                },
                "linajea-v4-r3-legacy-gpu-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "exploration/linajea_zf_gpu_visual_v2/gpu_v3/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_V4_R3_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "d74e99281370c1a78ef98786ba061e9b3730be0fd3df1adf4d027d465b032c1b"
                    ),
                    "kernel_id": "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v3",
                    "execution_profile": "linajea_legacy_tf1_gpu_visual.v1",
                },
                "linajea-v4-mount-repair-legacy-gpu-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "exploration/linajea_zf_gpu_visual_v2/gpu_v3/"
                        "mount_root_repair_v4/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "edb72263947eae56ca9b61ac7534a85b8bc9b78944cebc8b17812efce81b7b95"
                    ),
                    "kernel_id": "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v4",
                    "execution_profile": "linajea_legacy_tf1_gpu_visual.v1",
                },
                "linajea-v7-runtime-output-hardened-legacy-gpu-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "exploration/linajea_zf_gpu_visual_v2/gpu_v3/"
                        "runtime_output_hardening_v7/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "54b0e37c52a26782523ebe581c7834a88bd2aaf74610d64372d06dd6b1480bba"
                    ),
                    "kernel_id": "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v7",
                    "execution_profile": "linajea_legacy_tf1_gpu_visual.v1",
                },
                "linajea-v8-host-driver-path-legacy-gpu-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "exploration/linajea_zf_gpu_visual_v2/gpu_v3/"
                        "host_driver_path_v8/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "5debbdeedf0e0357fb1f9d70cf4aa1f44bd26673ea41a50e441b265693258b11"
                    ),
                    "kernel_id": "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v8",
                    "execution_profile": "linajea_legacy_tf1_gpu_visual.v1",
                },
                "linajea-v9-device-routing-legacy-gpu-remote-visual-v1": {
                    "record_path": (
                        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
                        "exploration/linajea_zf_gpu_visual_v2/gpu_v3/"
                        "device_routing_v9/"
                        "REMOTE_VISUAL_ONLY_EXCEPTION_ROOT_APPROVED.json"
                    ),
                    "record_sha256": (
                        "cdc1ab56cf32edf1abb546b1a2e4eea60efed87702110da5f639ff12e5bd022c"
                    ),
                    "kernel_id": "onebitbrain/g02-linajea-zf-tf1-gpu-visual-v9",
                    "execution_profile": "linajea_legacy_tf1_gpu_visual.v1",
                },
            },
        )


class ProductionSuccessorRemoteVisualGuardTest(unittest.TestCase):
    """The exact resealed successor contracts pass both required guard phases."""

    CASES = (
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v2_successor/kernel_package",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v2_successor/prelaunch-contract.json",
            "a84efb8ac51bb930b0c9ddbeaf5bc1349eb6648f4adebdfd1a6e51ba513a5f6b",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v3_mount_root/kernel_package",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v3_mount_root/prelaunch-contract.json",
            "646b3ef85af254df05c1ba10aca2a49d8e82f1acecc6472683623a059850097f",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v4_p100_sm60/kernel_package",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v4_p100_sm60/prelaunch-contract.json",
            "dab67f27a99e3e34e0599a0c9def186760e403f9e6da8bd6afc498444b114e5e",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v5_p100_monitor_reap/kernel_package",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v5_p100_monitor_reap/prelaunch-contract.json",
            "6e1f8c662713b4eee32502952de084e1da6d0dbb06d916a5b27a894f131213aa",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v6_p100_typed_terminal_reap/kernel_package",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v6_p100_typed_terminal_reap/prelaunch-contract.json",
            "93b84b6ccc2fe90de107287d896380fbb77f130220d10f922d90b8489a0c0fc6",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/model_kernel",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/prelaunch-contract-v4-r3.json",
            "09f8c241f9fe6f79a45e7e48865a994777736cd73ad23ab2d4fab08ab1487615",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/mount_root_repair_v4/model_kernel",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/mount_root_repair_v4/"
            "prelaunch-contract.json",
            "5dfabb7a67182d305d227056ff94efe1f24199a6f6c8f2e3d723824894e2178a",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/runtime_output_hardening_v7/model_kernel",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/runtime_output_hardening_v7/"
            "prelaunch-contract.json",
            "9f8c195397acaddeebdc3e38eedcb87d24ee2c59de172e2ff4cf59ba21a4d3c9",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/host_driver_path_v8/model_kernel",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/host_driver_path_v8/"
            "prelaunch-contract.json",
            "cd1852fb8bd243dcbaaeb8c171a0856ad6218f744089420d2a3e3efcd14c84a1",
        ),
        (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/device_routing_v9/model_kernel",
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/device_routing_v9/"
            "prelaunch-contract.json",
            "90619f2f6caa1ef49aeabb70237d9e8931120e891686fa5af876cd99d827ebea",
        ),
    )

    def test_exact_contract_hashes_and_inline_launch_guards(self) -> None:
        root = guard.PROJECT_ROOT
        registry_path = root / (
            "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
            "KAGGLE_FAILURE_REGISTRY.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for package_relative, contract_relative, expected_contract_sha in self.CASES:
            with self.subTest(contract=contract_relative):
                package = root / package_relative
                contract_path = root / contract_relative
                self.assertEqual(guard.sha256(contract_path), expected_contract_sha)
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                for phase in ("inline", "launch"):
                    result = guard.check_kernel(
                        package,
                        contract,
                        registry,
                        phase,
                        expected_initial_kernel_version=1,
                        project_root=root,
                        contract_path=contract_path,
                        registry_path=registry_path,
                    )
                    self.assertEqual(result["status"], "PASS", result["failures"])
                    self.assertEqual(
                        result["local_visual_preflight"]["status"],
                        guard.REMOTE_VISUAL_PENDING_STATUS,
                    )

    def test_cellect_v3_mount_repair_fails_without_hash_bound_correction(self) -> None:
        root = guard.PROJECT_ROOT
        package = root / (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v3_mount_root/kernel_package"
        )
        contract_path = root / (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/"
            "d2k5_gpu_visual/v3_mount_root/prelaunch-contract.json"
        )
        registry_path = root / (
            "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
            "KAGGLE_FAILURE_REGISTRY.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        contract["private_carriers"][1].pop("mount_layout_correction")
        contract["remote_visual_only"]["private_carriers"][1].pop(
            "mount_layout_correction"
        )
        result = guard.check_kernel(
            package,
            contract,
            registry,
            "launch",
            expected_initial_kernel_version=1,
            project_root=root,
            contract_path=contract_path,
            registry_path=registry_path,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("mount-layout correction", " ".join(result["failures"]))

    def test_linajea_v4_mount_repair_requires_full_decode_lock(self) -> None:
        root = guard.PROJECT_ROOT
        package = root / (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/mount_root_repair_v4/model_kernel"
        )
        contract_path = root / (
            "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/"
            "linajea_zf_gpu_visual_v2/gpu_v3/mount_root_repair_v4/"
            "prelaunch-contract.json"
        )
        registry_path = root / (
            "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
            "KAGGLE_FAILURE_REGISTRY.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        contract["remote_visual_only"]["post_remote_review"].pop(
            "full_decode_required"
        )
        result = guard.check_kernel(
            package,
            contract,
            registry,
            "launch",
            expected_initial_kernel_version=1,
            project_root=root,
            contract_path=contract_path,
            registry_path=registry_path,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("post-review lock drift", " ".join(result["failures"]))


class ProductionD2K2RemoteVisualGuardTest(unittest.TestCase):
    """The only live-package proof for the bespoke D2K2 carrier replacement."""

    BASE = (
        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
        "division/d2k2_kaggle_visual"
    )

    def test_current_d2k2_package_passes_only_with_a_central_exception_pin(self) -> None:
        root = guard.PROJECT_ROOT
        visual_root = root / self.BASE
        package = visual_root / "kernel_package"
        record_path = visual_root / "REMOTE_VISUAL_ONLY_EXCEPTION.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        code = package / "kernel.py"
        metadata = package / "kernel-metadata.json"
        identities = list(guard.D2K2_FIXED_IDENTITIES)
        exception_id = record["exception_id"]
        record_sha = guard.sha256(record_path)
        contract = {
            "kernel_id": guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["kernel_id"],
            "remote_preflight_kernel_id": guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE[
                "kernel_id"
            ],
            "remote_preflight_mode": "d2k2_step0_before_native_release.v1",
            "lane_owner": "CELLECT_X1_REMOTE_VISUAL_ONLY",
            "model_execution": True,
            "execution_profile": guard.REMOTE_VISUAL_ONLY_EXECUTION_PROFILE,
            "dataset_sources": [],
            "competition_sources": [
                guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["competition_source"]
            ],
            "expected_mount_root": guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE[
                "mount_root"
            ],
            "remote_visual_input_route": {
                "mode": guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["mode"],
                "competition_source": guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE[
                    "competition_source"
                ],
                "mount_root": guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["mount_root"],
            },
            "kernel_properties": guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE[
                "contract_kernel_properties"
            ],
            "preregistered_visual_identities": identities,
            "code_sha256": guard.sha256(code),
            "d2k2_package_contract": {
                "path": str((package / "D2K2_CONTRACT.json").relative_to(root)),
                "sha256": guard.sha256(package / "D2K2_CONTRACT.json"),
            },
            "d2k2_package_manifest": {
                "path": str((package / "PACKAGE_MANIFEST.json").relative_to(root)),
                "sha256": guard.sha256(package / "PACKAGE_MANIFEST.json"),
            },
            "remote_visual_only": {
                "schema": guard.REMOTE_VISUAL_ONLY_PRELAUNCH_SCHEMA,
                "candidate_id": record["candidate_id"],
                "kernel_id": record["kernel_id"],
                "execution_profile": record["execution_profile"],
                "model_execution": True,
                "candidate_code_path": str(code.relative_to(root)),
                "candidate_code_sha256": guard.sha256(code),
                "kernel_package_tree_sha256": guard._package_inventory(package, root)[
                    "tree_sha256"
                ],
                "fixed_preregistered_visual_identities": identities,
                "resource_stop_receipt": record["resource_stop_receipt"],
                "gpu_visual_only_contract": record["gpu_visual_only_contract"],
                "ship_kaggle_certificate": record["ship_kaggle_certificate"],
                "post_remote_review": record["post_remote_review"],
            },
            "remote_visual_only_exception": {
                "exception_id": exception_id,
                "record_path": str(record_path.relative_to(root)),
                "record_sha256": record_sha,
            },
        }
        allowlist = {
            exception_id: {
                "record_path": contract["remote_visual_only_exception"]["record_path"],
                "record_sha256": record_sha,
                "kernel_id": contract["kernel_id"],
                "execution_profile": contract["execution_profile"],
            }
        }
        self.assertEqual(
            json.loads(metadata.read_text(encoding="utf-8")),
            guard.REMOTE_VISUAL_ONLY_COMPETITION_ROUTE["metadata"],
        )
        with mock.patch.dict(guard.REMOTE_VISUAL_ONLY_EXCEPTION_ALLOWLIST, allowlist, clear=True):
            inline = guard.check_kernel(
                package,
                contract,
                {},
                "inline",
                expected_initial_kernel_version=1,
                project_root=root,
            )
            launch = guard.check_kernel(
                package,
                contract,
                {},
                "launch",
                expected_initial_kernel_version=1,
                project_root=root,
            )
        for result in (inline, launch):
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                result["local_visual_preflight"]["status"],
                guard.REMOTE_VISUAL_PENDING_STATUS,
            )
            self.assertEqual(result["local_visual_preflight"]["video_count"], 0)


class ProductionD2K3RemoteVisualGuardTest(unittest.TestCase):
    """D2K3 is a new package/record, not a retagged D2K2 retry."""

    BASE = (
        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
        "division/d2k3_kaggle_visual"
    )

    def test_current_d2k3_package_and_root_record_pass_the_bound_repair_lane(self) -> None:
        root = guard.PROJECT_ROOT
        visual_root = root / self.BASE
        package = visual_root / "kernel_package"
        contract = json.loads(
            (visual_root / "prelaunch-contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["kernel_id"], "onebitbrain/g02-cellect-visual-only-v3")
        self.assertEqual(contract["remote_preflight_mode"], guard.D2K3_INLINE_STEP0_MODE)
        self.assertEqual(
            contract["remote_visual_only_exception"]["record_sha256"],
            "84f7e4ba2f3710c43e20162c98ae6195b143cdfee2261fc19cff0e280db9765d",
        )
        self.assertEqual(
            guard.sha256(package / "kernel.py"),
            "9f378f3dce6f1447b1f7a9604a451b3983f1930ed0d5bcc58f19dedd945430d8",
        )
        for phase in ("inline", "launch"):
            result = guard.check_kernel(
                package,
                contract,
                {},
                phase,
                expected_initial_kernel_version=1,
                project_root=root,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["eligible"])
            self.assertEqual(
                result["local_visual_preflight"]["status"],
                guard.REMOTE_VISUAL_PENDING_STATUS,
            )
            self.assertEqual(result["local_visual_preflight"]["video_count"], 0)


class ProductionD2K5RemoteVisualGuardTest(unittest.TestCase):
    """D2K5 is script-only and consumes the byte-round-tripped Dataset V2 carrier."""

    BASE = (
        "dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/"
        "division/d2k5_gpu_visual"
    )

    def test_current_d2k5_package_and_root_record_pass_carrier_bound_lane(self) -> None:
        root = guard.PROJECT_ROOT
        visual_root = root / self.BASE
        package = visual_root / "kernel_package"
        contract = json.loads(
            (visual_root / "prelaunch-contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["remote_preflight_mode"], guard.D2K5_INLINE_STEP0_MODE
        )
        self.assertEqual(
            contract["asset_manifest_sha256"], guard.D2K5_ASSET_MANIFEST_SHA256
        )
        self.assertEqual(
            guard.sha256(package / "kernel.py"),
            "358bb34c2a7bf292d1b17c9b19df33b4fbb24a3a189095edef126c39e182ecbb",
        )
        for phase in ("inline", "launch"):
            result = guard.check_kernel(
                package,
                contract,
                {},
                phase,
                expected_initial_kernel_version=1,
                project_root=root,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["eligible"])
            self.assertEqual(
                result["local_visual_preflight"]["status"],
                guard.REMOTE_VISUAL_PENDING_STATUS,
            )
            self.assertEqual(result["local_visual_preflight"]["video_count"], 0)


class M7R6NativeT4CertificationGuardTest(unittest.TestCase):
    BASE = Path(
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r6_native_t4_whole_pipeline_cert"
    )

    def setUp(self) -> None:
        self.root = guard.PROJECT_ROOT
        self.base = self.root / self.BASE
        self.package = self.base / "package_v1"
        self.contract = json.loads(
            (self.base / "M7R6_CENTRAL_GUARD_CONTRACT.json").read_text(encoding="utf-8")
        )
        self.registry = json.loads(
            (self.base / "M7R6_GUARD_REGISTRY.json").read_text(encoding="utf-8")
        )

    def check(self, contract: dict | None = None) -> dict:
        return guard.check_kernel(
            self.package,
            self.contract if contract is None else contract,
            self.registry,
            "static",
            project_root=self.root,
        )

    def test_exact_m7r6_certification_package_passes_static_only(self) -> None:
        result = self.check()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["eligible"])
        self.assertTrue(result["model_execution"])
        self.assertEqual(result["local_visual_preflight"]["status"], "REMOTE_CERTIFICATION_PENDING")
        self.assertFalse(result["local_visual_preflight"]["production_enabled"])

    def test_m7r6_child_binding_drift_fails(self) -> None:
        contract = dict(self.contract)
        contract["authenticated_child_sha256"] = "0" * 64
        result = self.check(contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("M7R6 central contract drift: authenticated_child_sha256", result["failures"])

    def test_m7r6_runtime_bound_drift_fails(self) -> None:
        contract = dict(self.contract)
        contract["whole_run_hard_bound_seconds"] = 28_800
        result = self.check(contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("M7R6 central contract drift: whole_run_hard_bound_seconds", result["failures"])

    def test_m7r6_arbitrary_contract_surface_fails(self) -> None:
        contract = dict(self.contract)
        contract["candidate_override"] = "anything"
        result = self.check(contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("M7R6 central contract has an arbitrary surface", result["failures"])

    def test_m7r6_launch_phase_remains_coordinator_blocked(self) -> None:
        result = guard.check_kernel(
            self.package, self.contract, self.registry, "launch", project_root=self.root
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("current-LMT/quota/fresh-slug/serial-lock" in value for value in result["failures"]))


class M7R7NativeRuntimeDiagnosticGuardTest(unittest.TestCase):
    BASE = Path(
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r7_native_runtime_tuple_diagnostic"
    )

    def setUp(self) -> None:
        self.root = guard.PROJECT_ROOT
        self.base = self.root / self.BASE
        self.package = self.base / "package_v1"
        self.contract = json.loads(
            (self.base / "M7R7_CENTRAL_GUARD_CONTRACT.json").read_text(
                encoding="utf-8"
            )
        )
        self.registry = json.loads(
            (self.base / "M7R7_GUARD_REGISTRY.json").read_text(encoding="utf-8")
        )

    def check(
        self,
        contract: dict | None = None,
        package: Path | None = None,
        phase: str = "inline",
    ) -> dict:
        return guard.check_kernel(
            self.package if package is None else package,
            self.contract if contract is None else contract,
            self.registry,
            phase,
            expected_initial_kernel_version=1,
            project_root=self.root,
            contract_path=self.base / "M7R7_CENTRAL_GUARD_CONTRACT.json",
            registry_path=self.base / "M7R7_GUARD_REGISTRY.json",
        )

    def test_exact_m7r7_runtime_diagnostic_passes_inline(self) -> None:
        result = self.check()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["eligible"])
        self.assertFalse(result["model_execution"])
        self.assertEqual(result["local_visual_preflight"]["status"], "NOT_APPLICABLE")
        self.assertFalse(result["local_visual_preflight"]["production_enabled"])

    def test_m7r7_contract_and_arbitrary_surface_drift_fail(self) -> None:
        contract = dict(self.contract)
        contract["whole_run_hard_bound_seconds"] = 600
        contract["candidate_override"] = "anything"
        result = self.check(contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R7 central contract drift: whole_run_hard_bound_seconds",
            result["failures"],
        )
        self.assertIn("M7R7 central contract has an arbitrary surface", result["failures"])

    def test_m7r7_dependency_sentinel_cannot_become_executable(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.base) as temporary:
            package = Path(temporary) / "package"
            shutil.copytree(self.package, package)
            code = package / "kernel.py"
            source = code.read_text(encoding="utf-8")
            source = source.replace(
                "STEP0 = run_step0()",
                "STEP0 = run_step0()\nprint(UNUSED_DEPENDENCY_SENTINEL_ROOT)",
            )
            code.write_text(source, encoding="utf-8")
            result = self.check(package=package)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R7 dependency-only sentinel became executable", result["failures"]
        )

    def test_m7r7_ship_certificate_drift_fails(self) -> None:
        with mock.patch.dict(
            guard.M7R7_NATIVE_RUNTIME_DIAGNOSTIC_ROUTE,
            {"ship_certificate_sha256": "0" * 64},
        ):
            result = self.check()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("ship certificate" in failure.lower() for failure in result["failures"])
        )

    def test_m7r7_launch_phase_requires_separate_dynamic_coordinator(self) -> None:
        result = self.check(phase="launch")
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "current-LMT/quota/fresh-slug/serial-lock" in failure
                for failure in result["failures"]
            )
        )


class M7R8NativeT4CertificationGuardTest(unittest.TestCase):
    BASE = Path(
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r8_native_t4_whole_pipeline_cert"
    )

    def setUp(self) -> None:
        self.root = guard.PROJECT_ROOT
        self.base = self.root / self.BASE
        self.package = self.base / "package_v1"
        self.contract = json.loads(
            (self.base / "M7R8_CENTRAL_GUARD_CONTRACT.json").read_text(
                encoding="utf-8"
            )
        )
        self.registry = json.loads(
            (self.base / "M7R8_GUARD_REGISTRY.json").read_text(encoding="utf-8")
        )

    def check(self, contract: dict | None = None, phase: str = "inline") -> dict:
        return guard.check_kernel(
            self.package,
            self.contract if contract is None else contract,
            self.registry,
            phase,
            expected_initial_kernel_version=1,
            project_root=self.root,
            contract_path=self.base / "M7R8_CENTRAL_GUARD_CONTRACT.json",
            registry_path=self.base / "M7R8_GUARD_REGISTRY.json",
        )

    def test_exact_m7r8_tuple_only_package_passes_inline(self) -> None:
        result = self.check()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["eligible"])
        self.assertTrue(result["model_execution"])
        self.assertEqual(
            result["local_visual_preflight"]["status"],
            "REMOTE_CERTIFICATION_PENDING",
        )
        self.assertFalse(result["local_visual_preflight"]["production_enabled"])

    def test_m7r8_runtime_tuple_and_arbitrary_surface_drift_fail(self) -> None:
        contract = dict(self.contract)
        contract["m7r7_runtime_tuple_sha256"] = "0" * 64
        contract["runtime_override"] = "anything"
        result = self.check(contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R8 central contract drift: m7r7_runtime_tuple_sha256",
            result["failures"],
        )
        self.assertIn("M7R8 central contract has an arbitrary surface", result["failures"])

    def test_m7r8_parent_normalization_is_exact(self) -> None:
        source = (self.package / "kernel.py").read_text(encoding="utf-8")
        rewrites = (
            ('"python": "3.12"', '"python": "3.12.13"'),
            ('"torch": "2.7.1+cu126"', '"torch": "2.10.0+cu128"'),
            ('"torch_cuda": "12.6"', '"torch_cuda": "12.8"'),
            ("sys.version_info[:2] != (3, 12)", "sys.version_info[:3] != (3, 12, 13)"),
            ('torch.__version__ != "2.7.1+cu126"', 'torch.__version__ != "2.10.0+cu128"'),
            ('torch.version.cuda != "12.6"', 'torch.version.cuda != "12.8"'),
        )
        normalized = source
        for old, new in reversed(rewrites):
            self.assertEqual(normalized.count(new), 1)
            normalized = normalized.replace(new, old, 1)
        parent = (
            self.base.parent
            / "m7r6_native_t4_whole_pipeline_cert/package_v1/kernel.py"
        )
        self.assertEqual(normalized, parent.read_text(encoding="utf-8"))

    def test_m7r8_tuple_evidence_drift_fails(self) -> None:
        with mock.patch.dict(
            guard.M7R8_NATIVE_T4_CERT_ROUTE,
            {"m7r7_runtime_seal_sha256": "0" * 64},
        ):
            result = self.check()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("M7R7 runtime seal" in failure for failure in result["failures"])
        )

    def test_m7r8_launch_phase_requires_dynamic_coordinator(self) -> None:
        result = self.check(phase="launch")
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "current-LMT/quota/fresh-slug/serial-lock" in failure
                for failure in result["failures"]
            )
        )


class M7R13ProviderCertificationGuardTest(unittest.TestCase):
    BASE = Path(
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r13_full21_provider_cert"
    )

    def setUp(self) -> None:
        self.root = guard.PROJECT_ROOT
        self.base = self.root / self.BASE
        self.package = self.base / "package_v1"
        self.contract = json.loads(
            (self.base / "M7R13_CENTRAL_GUARD_CONTRACT.json").read_text(
                encoding="utf-8"
            )
        )
        self.registry = json.loads(
            (self.base / "M7R13_GUARD_REGISTRY.json").read_text(encoding="utf-8")
        )

    def check(
        self,
        contract: dict | None = None,
        package: Path | None = None,
        phase: str = "inline",
    ) -> dict:
        return guard.check_kernel(
            self.package if package is None else package,
            self.contract if contract is None else contract,
            self.registry,
            phase,
            expected_initial_kernel_version=1,
            project_root=self.root,
            contract_path=self.base / "M7R13_CENTRAL_GUARD_CONTRACT.json",
            registry_path=self.base / "M7R13_GUARD_REGISTRY.json",
        )

    def test_exact_m7r13_provider_package_passes_inline(self) -> None:
        result = self.check()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["eligible"])
        self.assertTrue(result["model_execution"])
        self.assertEqual(
            result["local_visual_preflight"]["status"],
            "REMOTE_CERTIFICATION_PENDING",
        )
        self.assertEqual(
            result["local_visual_preflight"]["full21_path_identity_count"], 21
        )
        self.assertFalse(result["local_visual_preflight"]["production_enabled"])

    def test_m7r13_contract_and_tuple_drift_fail(self) -> None:
        contract = dict(self.contract)
        contract["m7r7_runtime_tuple_sha256"] = "0" * 64
        contract["arbitrary_surface"] = True
        result = self.check(contract=contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R13 central contract drift: m7r7_runtime_tuple_sha256",
            result["failures"],
        )
        self.assertIn(
            "M7R13 central contract has an arbitrary surface", result["failures"]
        )

    def test_m7r13_child_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.base) as directory:
            package = Path(directory) / "package"
            shutil.copytree(self.package, package)
            code = package / "kernel.py"
            source = code.read_text(encoding="utf-8")
            source = source.replace(
                guard.M7R13_PROVIDER_CERT_ROUTE["provider_prefix_sha256"],
                "0" * 64,
                1,
            )
            code.write_text(source, encoding="utf-8")
            result = self.check(package=package)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("kernel hash drift" in failure for failure in result["failures"])
        )

    def test_m7r13_evidence_and_launch_phase_are_fail_closed(self) -> None:
        with mock.patch.dict(
            guard.M7R13_PROVIDER_CERT_ROUTE,
            {"m7r12_canary_sha256": "0" * 64},
        ):
            result = self.check()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("M7R12 canary" in failure for failure in result["failures"])
        )
        launch = self.check(phase="launch")
        self.assertEqual(launch["status"], "FAIL")
        self.assertTrue(
            any(
                "current-LMT/quota/fresh-slug/serial-lock" in failure
                for failure in launch["failures"]
            )
        )


class M7R14RawMetadataTransportGuardTest(unittest.TestCase):
    BASE = Path(
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r14_full21_raw_metadata_transport"
    )

    def setUp(self) -> None:
        self.root = guard.PROJECT_ROOT
        self.base = self.root / self.BASE
        self.package = self.base / "package_v1"
        self.contract_path = self.base / "M7R14_CENTRAL_GUARD_CONTRACT.json"
        self.registry_path = self.base / "M7R14_GUARD_REGISTRY.json"
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))

    def check(
        self,
        contract: dict | None = None,
        registry: dict | None = None,
        package: Path | None = None,
        phase: str = "inline",
    ) -> dict:
        return guard.check_kernel(
            self.package if package is None else package,
            self.contract if contract is None else contract,
            self.registry if registry is None else registry,
            phase,
            expected_initial_kernel_version=1,
            project_root=self.root,
            contract_path=self.contract_path,
            registry_path=self.registry_path,
        )

    def test_exact_m7r14_transport_package_passes_static_and_inline(self) -> None:
        static = self.check(phase="static")
        inline = self.check()
        self.assertEqual(static["status"], "PASS")
        self.assertFalse(static["eligible"])
        self.assertEqual(inline["status"], "PASS")
        self.assertTrue(inline["eligible"])
        self.assertTrue(inline["model_execution"])
        visual = inline["local_visual_preflight"]
        self.assertEqual(visual["status"], "REMOTE_CERTIFICATION_PENDING")
        self.assertEqual(visual["full21_path_identity_count"], 21)
        self.assertEqual(visual["raw_metadata_record_count"], 42)
        self.assertEqual(visual["coordination_status"], "BLOCKED_ACTIVE_M10R5")
        self.assertFalse(visual["launch_permitted"])

    def test_m7r14_contract_and_registry_drift_fail(self) -> None:
        contract = dict(self.contract)
        contract["raw_metadata_manifest_sha256"] = "0" * 64
        contract["arbitrary_surface"] = True
        result = self.check(contract=contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R14 central contract drift: raw_metadata_manifest_sha256",
            result["failures"],
        )
        self.assertIn(
            "M7R14 central contract has an arbitrary surface", result["failures"]
        )

        registry = json.loads(json.dumps(self.registry))
        registry["kernel_ownership"].pop(self.contract["kernel_id"])
        result = self.check(registry=registry)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R14 kernel ownership is not exclusively registered",
            result["failures"],
        )

    def test_m7r14_child_or_outer_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.base) as directory:
            package = Path(directory) / "package"
            shutil.copytree(self.package, package)
            code = package / "kernel.py"
            source = code.read_text(encoding="utf-8")
            source = source.replace("WHOLE_RUN_HARD_BOUND_SECONDS = 5400", "WHOLE_RUN_HARD_BOUND_SECONDS = 5401", 1)
            code.write_text(source, encoding="utf-8")
            result = self.check(package=package)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("kernel hash drift" in failure for failure in result["failures"])
        )

    def test_m7r14_launch_phase_is_hard_blocked_by_active_m10r5(self) -> None:
        result = self.check(phase="launch")
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["eligible"])
        self.assertEqual(
            result["failures"],
            ["M7R14 launch blocked while M10R5 has the active remote reservation"],
        )

    def test_m7r14_released_coordination_profile_passes_launch(self) -> None:
        contract_path = self.base / "M7R14_LAUNCH_CENTRAL_GUARD_CONTRACT.json"
        registry_path = (
            self.root
            / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
            "KAGGLE_FAILURE_REGISTRY.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        result = guard.check_kernel(
            self.package,
            contract,
            registry,
            "launch",
            expected_initial_kernel_version=1,
            project_root=self.root,
            contract_path=contract_path,
            registry_path=registry_path,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["eligible"])
        self.assertEqual(result["failures"], [])
        visual = result["local_visual_preflight"]
        self.assertEqual(
            visual["coordination_status"], "PASS_M10R5_TERMINAL_RELEASED"
        )
        self.assertTrue(visual["launch_permitted"])

    def test_m7r14_released_profile_rejects_terminal_evidence_drift(self) -> None:
        contract_path = self.base / "M7R14_LAUNCH_CENTRAL_GUARD_CONTRACT.json"
        registry_path = (
            self.root
            / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
            "KAGGLE_FAILURE_REGISTRY.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        with mock.patch.dict(
            guard.M7R14_RAW_METADATA_LAUNCH_ROUTE,
            {"m10r5_terminal_seal_sha256": "0" * 64},
        ):
            result = guard.check_kernel(
                self.package,
                contract,
                registry,
                "launch",
                expected_initial_kernel_version=1,
                project_root=self.root,
                contract_path=contract_path,
                registry_path=registry_path,
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("M10R5 terminal release seal" in value for value in result["failures"])
        )


class M7R15ActionSupplyCensusGuardTest(unittest.TestCase):
    BASE = Path(
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m7r15_full21_action_supply_census"
    )

    def setUp(self) -> None:
        self.root = guard.PROJECT_ROOT
        self.base = self.root / self.BASE
        self.package = self.base / "package_v1"
        self.contract_path = self.base / "M7R15_CENTRAL_GUARD_CONTRACT.json"
        self.registry_path = (
            self.root
            / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
            "KAGGLE_FAILURE_REGISTRY.json"
        )
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))

    def check(
        self,
        contract: dict | None = None,
        registry: dict | None = None,
        package: Path | None = None,
        phase: str = "inline",
    ) -> dict:
        return guard.check_kernel(
            self.package if package is None else package,
            self.contract if contract is None else contract,
            self.registry if registry is None else registry,
            phase,
            expected_initial_kernel_version=1,
            project_root=self.root,
            contract_path=self.contract_path,
            registry_path=self.registry_path,
        )

    def test_exact_m7r15_package_passes_static_inline_and_launch(self) -> None:
        static = self.check(phase="static")
        inline = self.check()
        launch = self.check(phase="launch")
        self.assertEqual(static["status"], "PASS")
        self.assertFalse(static["eligible"])
        self.assertEqual(inline["status"], "PASS")
        self.assertTrue(inline["eligible"])
        self.assertEqual(launch["status"], "PASS")
        self.assertTrue(launch["eligible"])
        self.assertTrue(launch["model_execution"])
        visual = launch["local_visual_preflight"]
        self.assertEqual(visual["status"], "REMOTE_CERTIFICATION_PENDING")
        self.assertTrue(visual["required"])
        self.assertEqual(visual["full_selected_fit_identity_count"], 21)
        self.assertEqual(visual["threshold_looks"], 1)
        self.assertTrue(visual["full_census_pending"])

    def test_m7r15_contract_and_registry_drift_fail(self) -> None:
        contract = dict(self.contract)
        contract["minimum_legal_alternatives"] = 15
        contract["arbitrary_surface"] = True
        result = self.check(contract=contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R15 central contract drift: minimum_legal_alternatives",
            result["failures"],
        )
        self.assertIn(
            "M7R15 central contract has an arbitrary surface", result["failures"]
        )

        registry = json.loads(json.dumps(self.registry))
        registry["kernel_ownership"].pop(self.contract["kernel_id"])
        result = self.check(registry=registry)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "M7R15 kernel ownership is not exclusively registered",
            result["failures"],
        )

    def test_m7r15_outer_or_child_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.base) as directory:
            package = Path(directory) / "package"
            shutil.copytree(self.package, package)
            code = package / "kernel.py"
            source = code.read_text(encoding="utf-8")
            source = source.replace(
                "MINIMUM_LEGAL_ALTERNATIVES = 16",
                "MINIMUM_LEGAL_ALTERNATIVES = 15",
                1,
            )
            code.write_text(source, encoding="utf-8")
            result = self.check(package=package)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("kernel hash drift" in failure for failure in result["failures"])
        )

    def test_m7r15_parent_terminal_drift_fails_closed(self) -> None:
        with mock.patch.dict(
            guard.M7R15_ACTION_SUPPLY_ROUTE,
            {"m7r14_terminal_sha256": "0" * 64},
        ):
            result = self.check()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("M7R14 terminal seal" in failure for failure in result["failures"])
        )


class M10R3ProviderCertificationGuardTest(unittest.TestCase):
    BASE = Path(
        "dag_prompts/2026-08-23-model-clade-renewal-v1/"
        "m10r3_self_orchestrating_provider_cert"
    )

    def setUp(self) -> None:
        self.root = guard.PROJECT_ROOT
        self.base = self.root / self.BASE
        self.package = self.base / "package_v1"
        self.contract_path = self.base / "M10R3_CENTRAL_GUARD_CONTRACT.json"
        self.registry_path = (
            self.root
            / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
            "KAGGLE_FAILURE_REGISTRY.json"
        )
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))

    def check(self, contract: dict | None = None, phase: str = "inline") -> dict:
        return guard.check_kernel(
            self.package,
            self.contract if contract is None else contract,
            self.registry,
            phase,
            expected_initial_kernel_version=1,
            project_root=self.root,
            contract_path=self.contract_path,
            registry_path=self.registry_path,
        )

    def test_exact_m10r3_no_input_provider_cert_passes_inline(self) -> None:
        result = self.check()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["eligible"])
        self.assertTrue(result["model_execution"])
        visual = result["local_visual_preflight"]
        self.assertEqual(visual["status"], "REMOTE_CERTIFICATION_PENDING")
        self.assertEqual(visual["identities"], [])
        self.assertFalse(visual["target_input"])
        self.assertFalse(visual["production_enabled"])

    def test_m10r3_target_input_and_arbitrary_surface_drift_fail(self) -> None:
        contract = dict(self.contract)
        contract["target_input"] = True
        contract["runtime_override"] = 1
        result = self.check(contract)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("M10R3 central contract drift: target_input", result["failures"])
        self.assertIn("M10R3 central contract has an arbitrary surface", result["failures"])

    def test_m10r3_embedded_child_binding_drift_fails(self) -> None:
        with mock.patch.dict(
            guard.M10R3_PROVIDER_CERT_ROUTE,
            {"m10r2_child_sha256": "0" * 64},
        ):
            result = self.check()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("M10R2 exact child" in value for value in result["failures"]))

    def test_m10r3_launch_phase_requires_dynamic_coordinator(self) -> None:
        result = self.check(phase="launch")
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "current-LMT/quota/fresh-slug/serial-lock" in failure
                for failure in result["failures"]
            )
        )


if __name__ == "__main__":
    unittest.main()
