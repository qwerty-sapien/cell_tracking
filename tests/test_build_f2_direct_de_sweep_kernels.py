from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SWEEP = load_module(
    "build_f2_direct_de_sweep_kernels",
    "scripts/build_f2_direct_de_sweep_kernels.py",
)
PROFILE = load_module(
    "build_f2_sweep_preflight_profile",
    "scripts/build_f2_sweep_preflight_profile.py",
)


class F2DirectSweepBuilderTests(unittest.TestCase):
    def test_v9_rejection_binding_is_projection_only_and_exact(self) -> None:
        binding = SWEEP.v9_rejection_binding()
        self.assertEqual(binding["sha256"], SWEEP.EXPECTED_V9_D_REJECTION_SHA256)
        self.assertEqual(binding["static_status"], "PASS")
        self.assertEqual(binding["repeatability_status"], "PASS")
        self.assertEqual(binding["quick_debug_status"], "PASS")
        self.assertEqual(binding["successful_simulations"], 8)
        self.assertEqual(binding["projection_status"], "FAIL")

    def test_frozen_candidate_families_are_exact(self) -> None:
        self.assertEqual(
            [row["id"] for row in SWEEP.frozen_candidates("D")],
            [
                "D_det0.9375",
                "D_det0.95",
                "D_det0.96",
                "D_det0.975",
                "D_det0.98",
                "D_det0.99",
            ],
        )
        self.assertEqual(len(SWEEP.frozen_candidates("E")), 8)
        runtime = SWEEP.runtime_candidates(SWEEP.frozen_candidates("D"))
        self.assertEqual(runtime[-1], {
            "id": "CONTROL_V32",
            "family": "CONTROL",
            "delta": {},
            "environment": {},
        })

    def test_parameterized_child_preserves_defaults_and_opens_exact_overrides(self) -> None:
        builder = SWEEP._load_v32_builder()
        population = json.loads(builder.POPULATION_DEFAULT.read_text(encoding="utf-8"))
        source, freeze = builder.build_instrumented_source(
            population_manifest=population,
            role="f2_six",
        )
        parameterized = SWEEP.parameterize_child_source(source)
        self.assertEqual(tuple(freeze["identities"]), builder.F2_SIX_IDENTITIES)
        for variable in SWEEP.FAMILY_ENV.values():
            self.assertIn(variable, parameterized)
        self.assertIn("graph.rx_graph.weighted_edge_list()", parameterized)
        self.assertIn("def _v32_rustworkx_nodes(graph):", parameterized)
        self.assertIn('"node_id": int(node_id)', parameterized)
        self.assertNotIn("graph.node_ids()", parameterized)
        self.assertNotIn("node_indices()", parameterized)
        self.assertNotIn(
            'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"', parameterized
        )
        compile(parameterized, "parameterized_child.py", "exec")

    def test_package_is_no_submission_bounded_and_hash_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernel"
            freeze = SWEEP.build_package(
                family="D",
                kernel_id="onebitbrain/f2-v32-direct-d-sweep-test",
                output_dir=output,
            )
            source = (output / "f2_direct_d_sweep_kernel.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("MAX_SECONDS = 7 * 60 * 60", source)
            self.assertIn("F2_DIRECT_SWEEP_COMPLETE", source)
            self.assertIn('"V32_FINAL_GRAPH_ROWS.csv"', source)
            self.assertIn('if (WORKING / "submission.csv").exists():', source)
            self.assertIn("def relay_successful_child_streams", source)
            self.assertIn("sys.stdout.buffer.write(stdout_payload)", source)
            self.assertIn("F2_CHILD_STDERR_RETAINED", source)
            self.assertIn(
                "relay_successful_child_streams(stdout_path, stderr_path)", source
            )
            self.assertIn("graph.rx_graph.weighted_edge_list()", source)
            self.assertIn("def _v32_rustworkx_nodes(graph):", source)
            self.assertIn('"node_id": int(node_id)', source)
            self.assertNotIn("graph.node_ids()", source)
            self.assertNotIn("node_indices()", source)
            self.assertNotIn('.rglob("*"', source)
            self.assertNotIn('COMP_DIR / "test"', source)
            self.assertNotIn(
                "/competitions/biohub-cell-tracking-during-development/test",
                source,
            )
            self.assertFalse(freeze["submission_written"])
            self.assertEqual(
                freeze["parent_v8_native_rejection"]["sha256"],
                SWEEP.EXPECTED_V8_D_REJECTION_SHA256,
            )
            self.assertEqual(
                freeze["parent_v8_native_rejection"]["status"], "REJECTED"
            )
            self.assertEqual(len(freeze["identities"]), 6)
            self.assertEqual(len(freeze["candidate_ids"]), 7)
            self.assertEqual(freeze["same_run_control_id"], "CONTROL_V32")

    def test_profile_workload_units_match_all_candidate_frame_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family, candidates, expected in (
                ("D", 7, [3, 6, 12, 108]),
                ("E", 9, [3, 6, 12, 108]),
            ):
                package = root / f"kernel_{family.lower()}"
                SWEEP.build_package(
                    family=family,
                    kernel_id=f"onebitbrain/f2-{family.lower()}-test",
                    output_dir=package,
                )
                profile_path = root / f"profile_{family.lower()}.json"
                profile = PROFILE.build_profile(
                    package_dir=package,
                    profile_path=profile_path,
                    family=family,
                )
                self.assertEqual(
                    [row["workload_units"] for row in profile["scenarios"]],
                    expected,
                )
                self.assertEqual(
                    [
                        row["env"]["KAGGLE_PREFLIGHT_UNITS"]
                        for row in profile["scenarios"]
                    ],
                    ["3", "3", "12", "54"],
                )
                self.assertEqual(len(set(expected)), 4)
                self.assertEqual(
                    profile["runtime"]["total_workload_units"],
                    candidates * 6 * 100,
                )
                self.assertGreaterEqual(
                    max(expected) / profile["runtime"]["total_workload_units"],
                    profile["runtime"]["min_largest_sample_fraction"],
                )
                for scenario in profile["scenarios"]:
                    self.assertIn(
                        "V32_STANDALONE_DISPLAY_SINK",
                        scenario["stdout_markers"],
                    )

    def test_v10_source_bound_sequential_envelope_and_profile_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family, candidate_count, expected_units in (
                ("D", 7, [3, 6, 100, 1400]),
                ("E", 9, [3, 6, 100, 1800]),
            ):
                package = root / f"kernel_{family.lower()}_v10"
                freeze = SWEEP.build_package(
                    family=family,
                    kernel_id=f"onebitbrain/f2-{family.lower()}-v10-test",
                    output_dir=package,
                    parent_rejection_version="v9",
                    bounded_sequential_envelope=True,
                )
                source = (
                    package / f"f2_direct_{family.lower()}_sweep_kernel.py"
                ).read_text(encoding="utf-8")
                self.assertIn("BOUNDED_SEQUENTIAL_ENVELOPE = True", source)
                self.assertIn('"KAGGLE_PREFLIGHT_CANDIDATES"', source)
                self.assertIn('environment["CUDA_VISIBLE_DEVICES"] = "0"', source)
                outer = ast.parse(source)
                candidate_loops = [
                    node
                    for node in ast.walk(outer)
                    if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "candidate"
                ]
                self.assertEqual(len(candidate_loops), 1)
                self.assertEqual(
                    freeze["parent_v9_native_rejection"]["sha256"],
                    SWEEP.EXPECTED_V9_D_REJECTION_SHA256,
                )
                proof = freeze["runtime_envelope_proof"]
                self.assertEqual(proof["status"], "PASS_SOURCE_BOUND")
                self.assertEqual(proof["candidate_process_concurrency"], 1)
                self.assertEqual(proof["visible_cuda_devices"], ["0"])
                self.assertEqual(proof["production_frame_ceiling"], 100)
                self.assertFalse(proof["science_or_ilp_modified"])

                profile_path = root / f"profile_{family.lower()}_v10.json"
                profile = PROFILE.build_profile(
                    package_dir=package,
                    profile_path=profile_path,
                    family=family,
                    bounded_sequential_v10=True,
                )
                self.assertEqual(
                    [row["workload_units"] for row in profile["scenarios"]],
                    expected_units,
                )
                self.assertEqual(
                    [row["repeats"] for row in profile["scenarios"]],
                    [3, 2, 2, 2],
                )
                self.assertEqual(
                    [
                        row["env"]["KAGGLE_PREFLIGHT_CANDIDATES"]
                        for row in profile["scenarios"]
                    ],
                    ["1", "1", "1", str(candidate_count)],
                )
                model = profile["runtime"]["projection_model"]
                self.assertEqual(model["mode"], "bounded_sequential_replay.v1")
                self.assertEqual(
                    model["startup"]["parent_receipt"]["sha256"],
                    SWEEP.EXPECTED_V9_D_REJECTION_SHA256,
                )
                self.assertEqual(
                    model["expansive_replay"]["full_workload_multiplier"], 3
                )
                self.assertEqual(
                    model["bounded_memory"]["source_proof"],
                    freeze["runtime_envelope_proof"],
                )


if __name__ == "__main__":
    unittest.main()
