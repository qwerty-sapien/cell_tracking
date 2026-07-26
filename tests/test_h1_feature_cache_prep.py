from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_h1_feature_cache as prep  # noqa: E402


def _load_builder():
    path = SCRIPTS / "build_trackastra_full_pool_feature_cache.py"
    spec = importlib.util.spec_from_file_location("h1_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(rows: int = 8) -> dict[str, np.ndarray]:
    features = np.linspace(
        -0.25, 0.75, rows * 49, dtype=np.float32
    ).reshape(rows, 49)
    return {
        "row_features": features,
        "challengers": np.arange(rows * 2, dtype=np.int64).reshape(rows, 2),
        "conflicts": np.arange(rows * 2, dtype=np.int64).reshape(rows, 2),
        "proposal_offsets": np.arange(rows + 1, dtype=np.int64),
        "transition_offsets": np.arange(rows + 1, dtype=np.int64),
        "transition_target_t": np.arange(rows, dtype=np.int32),
        "geometry_confidences": np.linspace(
            0.1, 0.9, rows, dtype=np.float32
        ),
        "geometry_probabilities": np.linspace(
            0.2, 0.8, rows, dtype=np.float32
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_assignment(path: Path, name: str):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"missing literal assignment {name} in {path}")


class H1FeatureCachePrepTests(unittest.TestCase):
    def test_authenticated_manifest_is_exact_fit_train105(self) -> None:
        manifest = prep.load_json(prep.DEFAULT_MANIFEST)
        train = tuple(str(value) for value in manifest["identities"])
        self.assertEqual(len(train), 105)
        self.assertEqual(len(set(train)), 105)
        self.assertEqual(
            sum(":44b6_" in identity for identity in train), 36
        )
        self.assertEqual(
            sum(":6bba_" in identity for identity in train), 69
        )
        self.assertTrue(manifest["scope"]["fit_train105_only"])
        self.assertFalse(manifest["scope"]["validation12_access"])
        self.assertFalse(manifest["scope"]["cv57_access"])

    def test_benign_float_drift_requires_human_approval(self) -> None:
        m5 = _payload()
        t4_first = {key: value.copy() for key, value in m5.items()}
        feature = t4_first["row_features"]
        for row in range(feature.shape[0]):
            direction = np.float32(np.inf if row % 2 == 0 else -np.inf)
            feature[row] = np.nextafter(feature[row], direction)
        t4_second = {
            key: value.copy() for key, value in t4_first.items()
        }
        result = prep.compare_option_c_payloads(
            m5=m5, t4_first=t4_first, t4_second=t4_second
        )
        self.assertEqual(
            result["verdict"],
            "BENIGN_FP_PROPOSAL_REQUIRES_HUMAN_APPROVAL",
        )
        self.assertTrue(
            all(
                row["classification"] == "BENIGN_FP"
                for row in result["m5_t4"]["columns"]
            )
        )
        self.assertFalse(
            result["m5_t4"]["tolerance_contract_finalized"]
        )

    def test_geometry_mismatch_is_structural_even_if_float(self) -> None:
        m5 = _payload()
        t4_first = {key: value.copy() for key, value in m5.items()}
        t4_second = {key: value.copy() for key, value in m5.items()}
        t4_first["geometry_probabilities"][0] += np.float32(1.0e-7)
        t4_second["geometry_probabilities"][0] += np.float32(1.0e-7)
        result = prep.compare_option_c_payloads(
            m5=m5, t4_first=t4_first, t4_second=t4_second
        )
        self.assertEqual(
            result["verdict"],
            "STOP_EXACT_GEOMETRY_OR_IDENTITY_DRIFT",
        )

    def test_t4_repeatability_failure_stops(self) -> None:
        m5 = _payload()
        t4_first = {key: value.copy() for key, value in m5.items()}
        t4_second = {key: value.copy() for key, value in m5.items()}
        t4_second["row_features"][0, 0] += np.float32(1.0e-4)
        result = prep.compare_option_c_payloads(
            m5=m5, t4_first=t4_first, t4_second=t4_second
        )
        self.assertEqual(result["verdict"], "STOP_T4_NONDETERMINISTIC")

    def test_missing_or_nonnumeric_exact_field_fails_closed(self) -> None:
        m5 = _payload()
        t4_first = {key: value.copy() for key, value in m5.items()}
        t4_second = {key: value.copy() for key, value in m5.items()}
        del t4_second["conflicts"]
        with self.assertRaisesRegex(
            prep.H1PrepError, "required exact field is missing: conflicts"
        ):
            prep.compare_option_c_payloads(
                m5=m5, t4_first=t4_first, t4_second=t4_second
            )
        t4_second["conflicts"] = np.asarray(
            [["bad", "dtype"]] * 8, dtype="<U5"
        )
        with self.assertRaisesRegex(
            prep.H1PrepError, "unauthorized dtype: conflicts"
        ):
            prep.compare_option_c_payloads(
                m5=m5, t4_first=t4_first, t4_second=t4_second
            )

    def test_option_c_contract_loads_and_legacy_contract_remains_valid(
        self,
    ) -> None:
        builder = _load_builder()
        for relative in (
            "reports/g02/trackastra_full_pool_feature_cache_contract_v2.json",
            "kaggle_kernel/trackastra-h1-feature-cache-prep/option_c_contract.json",
        ):
            path = ROOT / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            contract = builder.load_contract(path, digest)
            families, counts = builder.contract_families(contract)
            if "option_c" in relative:
                self.assertEqual(families, ("44b6", "6bba"))
                self.assertEqual(counts, {"44b6": 36, "6bba": 69})
                self.assertEqual(
                    contract["revision"]["decision"],
                    "DECOUPLE_H1_FEATURE_ARMS_ZERO_WRFEAT_FIRST",
                )
                self.assertFalse(
                    contract["revision"][
                        "real_mask_pre_smoke_prerequisite"
                    ]
                )
                self.assertEqual(
                    contract["features"]["h3_fallback"]["status"],
                    "CONDITIONAL_NOT_STARTED",
                )
            else:
                self.assertEqual(families, ("6bba",))
                self.assertEqual(counts, {"6bba": 69})

    def test_builder_t4_repeatability_checks_every_array(self) -> None:
        builder = _load_builder()
        contract_path = (
            ROOT
            / "kaggle_kernel/trackastra-h1-feature-cache-prep/option_c_contract.json"
        )
        digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        contract = builder.load_contract(contract_path, digest)
        first = _payload()
        second = {key: value.copy() for key, value in first.items()}
        first.update(
            {
                "identity": "train:6bba_edf14583",
                "feature_names": list(prep.FEATURE_NAMES),
                "selected_raw_edges": 4,
                "pool_counts": {"full_pool_proposals": 8},
                "inference": {"inference_calls": 2},
            }
        )
        second.update(
            {
                "identity": "train:6bba_edf14583",
                "feature_names": list(prep.FEATURE_NAMES),
                "selected_raw_edges": 4,
                "pool_counts": {"full_pool_proposals": 8},
                "inference": {"inference_calls": 2},
            }
        )
        builder.assert_t4_repeatability(first, second, contract)
        second["conflicts"][0, 0] += 1
        with self.assertRaisesRegex(
            builder.TrackastraFeatureCacheError,
            "T4 repeatability failed for conflicts",
        ):
            builder.assert_t4_repeatability(first, second, contract)

    def test_option_c_requires_cuda_but_legacy_v2_stays_cpu(self) -> None:
        builder = _load_builder()
        option_path = (
            ROOT
            / "kaggle_kernel/trackastra-h1-feature-cache-prep/option_c_contract.json"
        )
        option = builder.load_contract(
            option_path, _sha256(option_path)
        )
        legacy_path = (
            ROOT
            / "reports/g02/trackastra_full_pool_feature_cache_contract_v2.json"
        )
        legacy = builder.load_contract(
            legacy_path, _sha256(legacy_path)
        )
        self.assertEqual(builder.execution_device(option), "cuda:0")
        self.assertEqual(builder.execution_device(legacy), "cpu")

        cpu_module = builder.torch.nn.Linear(2, 1)
        attestation = builder.attest_module_device(
            cpu_module, "cpu", "synthetic"
        )
        self.assertTrue(attestation["pass"])
        with self.assertRaisesRegex(
            builder.TrackastraFeatureCacheError,
            "device placement drifted",
        ):
            builder.attest_module_device(
                cpu_module, "cuda:0", "synthetic"
            )

    def test_trackastra_inputs_fail_if_option_c_model_stays_cpu(self) -> None:
        builder = _load_builder()
        cpu_model = builder.torch.nn.Linear(4, 4)
        with self.assertRaisesRegex(
            RuntimeError, "not on the required inference device"
        ):
            builder._association(
                cpu_model,
                (1,),
                {1: (0, np.zeros(3, dtype=np.float32))},
                expected_device="cuda:0",
            )

    def test_stage_builders_pass_contract_device_to_inference(self) -> None:
        expectations = {
            "scripts/build_trackastra_apncc_base_cache.py": {
                "load_v45_model": "device",
                "attest_module_device": None,
            },
            "scripts/enrich_trackastra_apncc_cache.py": {
                "load_mmap_meta": "device",
                "trackastra_margins_from_flattened": "expected_device",
                "attest_module_device": None,
            },
        }
        for relative, required in expectations.items():
            tree = ast.parse(
                (ROOT / relative).read_text(encoding="utf-8")
            )
            calls = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    continue
                calls.append((name, node))
            for name, keyword in required.items():
                matching = [node for call_name, node in calls if call_name == name]
                self.assertTrue(
                    matching, msg=f"{relative} does not call {name}"
                )
                if keyword is not None:
                    self.assertTrue(
                        any(
                            any(
                                item.arg == keyword
                                and isinstance(item.value, ast.Name)
                                and item.value.id == "device"
                                for item in node.keywords
                            )
                            for node in matching
                        ),
                        msg=(
                            f"{relative} does not pass {keyword}=device "
                            f"to {name}"
                        ),
                    )

    def test_option_c_environment_and_runtime_gate_are_fail_closed(
        self,
    ) -> None:
        contract = json.loads(
            (
                ROOT
                / "kaggle_kernel/trackastra-h1-feature-cache-prep/option_c_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["environment"],
            {
                "banked_t4_evidence": [
                    {
                        "path": "kaggle_kernel/bhp0-c0-out-v1/run.json",
                        "sha256": "bd459feecf9a05545ced763b0c9a364a885cb375fdbdea08fedf58d1a83e5a9d",
                    },
                    {
                        "path": "reports/g02/fitcache_remote/fc00_v3/bh9.json",
                        "sha256": "25cb4247d337941275f4447d4ff5d26f260aac52028702a8e90cc51d17d94e97",
                    },
                ],
                "cuda_runtime": "12.8.1",
                "cudnn": 91002,
                "device": "Tesla T4",
                "packages": {
                    "PyYAML": "6.0.3",
                    "donfig": "0.8.1.post1",
                    "numcodecs": "0.15.1",
                    "numpy": "2.0.2",
                    "scipy": "1.16.3",
                    "torch": "2.10.0+cu128",
                    "zarr": "3.2.1",
                },
                "python": "3.12.13",
                "torch_cuda": "12.8",
            },
        )
        self.assertFalse(
            contract["runtime_projection"]["full_run_authorized"]
        )
        self.assertEqual(
            contract["runtime_projection"]["status"],
            "BLOCKED_NO_BANKED_OPTION_C_CUDA_TIMING",
        )

    def test_hash_pins_match_actual_files_kernel_and_report(self) -> None:
        package = (
            ROOT / "kaggle_kernel/trackastra-h1-feature-cache-prep"
        )
        kernel_path = package / "trackastra_h1_feature_cache.py"
        dependency_hash = _literal_assignment(
            kernel_path, "SOURCE_DEPENDENCY_MANIFEST_SHA256"
        )
        dependency_path = package / "source_dependency_manifest.json"
        self.assertEqual(_sha256(dependency_path), dependency_hash)
        dependencies = json.loads(
            dependency_path.read_text(encoding="utf-8")
        )
        contract = json.loads(
            (package / "option_c_contract.json").read_text(
                encoding="utf-8"
            )
        )
        trackastra_records = {}
        for record in dependencies["files"]:
            if record["origin"] == "workspace":
                path = ROOT / record["path"]
            elif record["origin"] == "support_dataset":
                path = ROOT / record["local_evidence_path"]
            else:
                trackastra_records[record["path"]] = record["sha256"]
                continue
            self.assertEqual(
                _sha256(path),
                record["sha256"],
                msg=f"stale dependency pin for {record['path']}",
            )
        self.assertEqual(
            trackastra_records, contract["source_files"]
        )

        environment_pins = _literal_assignment(
            kernel_path, "ENVIRONMENT_PINS"
        )
        contract_environment = dict(contract["environment"])
        evidence = contract_environment.pop("banked_t4_evidence")
        self.assertEqual(environment_pins, contract_environment)
        for record in evidence:
            self.assertEqual(
                _sha256(ROOT / record["path"]),
                record["sha256"],
                msg=f"stale T4 environment evidence {record['path']}",
            )

        assets = json.loads(
            (package / "assets_dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for record in assets["files"]:
            relative = record["path"].removeprefix("source_overlay/")
            self.assertEqual(
                _sha256(ROOT / relative),
                record["sha256"],
                msg=f"stale assets manifest pin for {relative}",
            )

        # The banked H1-PREP v1 report is immutable historical evidence. The
        # active zero-wrfeat revision is pinned by the package manifests above.


if __name__ == "__main__":
    unittest.main()
