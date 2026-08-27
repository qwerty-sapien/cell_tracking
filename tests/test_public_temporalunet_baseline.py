from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_temporalunet_baseline",
    ROOT / "scripts/public_temporalunet_baseline.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def population_payload() -> dict:
    held = [f"train:held_{index:03d}" for index in range(105)]
    cv57 = [f"train:cv_{index:03d}" for index in range(57)]
    payload = {
        "schema": MODULE.POPULATION_SCHEMA,
        "populations": {"held_fit105": held, "cv57": cv57},
        "union": held + cv57,
        "counts": {"held_fit105": 105, "cv57": 57, "union": 162},
        "substrate_only": True,
        "weight_refinement_minimum_local_score": 0.916,
    }
    payload["manifest_sha256"] = MODULE.canonical_sha256(payload)
    return payload


class PublicTemporalUNetBaselineTests(unittest.TestCase):
    def test_hidden_package_is_gate_bound_and_discovers_test_dynamically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_path = root / "score.json"
            taxonomy_path = root / "taxonomy.json"
            terminal_path = root / "terminal.json"
            submission_sha = "a" * 64
            score = {
                "schema": MODULE.SCORE_SCHEMA,
                "status": "PASS",
                "variant": "public_v32",
                "population": "cv57",
                "score": 0.9143584730207287,
                "weight_refinement": False,
                "submission": {"sha256": submission_sha},
                "outdegree_diagnostics": {
                    "policy": "competition_faithful_no_outdegree_cap",
                    "degree_three_parent_count": 2,
                },
            }
            score_path.write_text(json.dumps(score), encoding="utf-8")
            score_sha = MODULE.sha256_file(score_path)
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "global": {"baseline_summary": {"score": score["score"]}},
                        "validation": {
                            "status": "PASS",
                            "submission_graph_policy": "competition_faithful_no_outdegree_cap",
                            "authoritative_replay": {
                                "mismatch_count": 0,
                                "reference_sha256": score_sha,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            terminal_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "kernel": {"version": 1, "terminal_status": "COMPLETE"},
                        "validation": {
                            "submission_sha256": submission_sha,
                            "graph_mutated": False,
                            "graph_policy": "competition_faithful_no_outdegree_cap",
                        },
                    }
                ),
                encoding="utf-8",
            )
            gate = MODULE.validate_hidden_submission_gate(
                score_path=score_path,
                taxonomy_path=taxonomy_path,
                terminal_auth_path=terminal_path,
            )
            source = MODULE.build_hidden_script(expected_test_count=4, local_gate=gate)
            self.assertTrue(gate["eligible"])
            self.assertEqual(gate["local_score"], score["score"])
            # The hidden rerun contract: the test root is resolved against BOTH
            # Kaggle mount conventions and the inventory cardinality is never pinned.
            self.assertIn("COMPETITION_TEST_CANDIDATES = [", source)
            self.assertIn("COMPETITION_TEST = COMPETITION_TEST_CANDIDATES[0]", source)
            self.assertIn(
                "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
                source,
            )
            self.assertIn(
                "/kaggle/input/biohub-cell-tracking-during-development/test", source
            )
            # No fixed-cardinality gate may reappear.  (The separate step0-vs-prediction
            # "drifted between step 0 and prediction" consistency check is retained and
            # is deliberately NOT matched here.)
            self.assertNotIn("EXPECTED_TEST_COUNT", source)
            self.assertNotIn("STEP0 hidden inventory drift", source)
            # The env override must carry the RESOLVED root, never a hard-coded one.
            self.assertIn(
                'os.environ["BIOHUB_TEST_DIR"] = STEP0["competition_test_root"]', source
            )
            self.assertIn("DYNAMIC_TEST_DISCOVERY = True", source)
            self.assertIn('PUBLIC_OUTPUT_COPIED = False', source)
            self.assertIn('RUN_LOG_PATH = Path("/kaggle/working/RUN_LOG.jsonl")', source)
            self.assertNotIn("44b6_", source)
            self.assertNotIn("6bba_", source)
            self.assertNotIn("SELECTED_TRAIN", source)

    def _execute_hidden_step0(self, root: Path, *, n_datasets: int, layout: str) -> dict:
        """Actually EXECUTE the generated STEP0 against a synthetic Kaggle mount.

        A compile()/string check cannot catch unresolved names, dead fallback
        branches or invalid paths -- the two defects this contract exists to
        prevent were both compile-clean.  So this runs the real generated code.
        """
        import contextlib
        import hashlib
        import io

        primary = root / "input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
        secondary = root / "input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1"
        digests = {}
        for pack, tag in ((primary, "primary"), (secondary, "secondary")):
            (pack / "weights/unet_transformer/split_0").mkdir(parents=True, exist_ok=True)
            manifest = pack / "ARTIFACT_MANIFEST.json"
            weight = pack / "weights/unet_transformer/split_0/edge_predictor_best.pth"
            manifest.write_bytes(f'{{"{tag}": true}}'.encode())
            weight.write_bytes(f"{tag}-weights".encode())
            digests[f"{tag}_manifest"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            digests[f"{tag}_weight"] = hashlib.sha256(weight.read_bytes()).hexdigest()

        competition = MODULE.COMPETITION
        candidates = [
            root / f"input/competitions/{competition}/test",
            root / f"input/{competition}/test",
        ]
        chosen = candidates[0] if layout == "competitions" else candidates[1]
        chosen.mkdir(parents=True, exist_ok=True)
        for index in range(n_datasets):
            dataset = chosen / f"hidden_{index:04d}.zarr"
            dataset.mkdir(parents=True, exist_ok=True)
            (dataset / "0").mkdir(exist_ok=True)

        gate = {
            "gate_sha256": "a" * 64,
            "local_score": 0.9143584730207287,
            "eligible": True,
        }
        script = MODULE.build_hidden_script(expected_test_count=4, local_gate=gate)
        marker = "STEP0 = run_step0()"
        self.assertIn(marker, script)
        namespace: dict = {"__name__": "hidden_step0_under_test"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(script[: script.index(marker)], "kernel.py", "exec"), namespace)

        working = root / "working"
        working.mkdir(parents=True, exist_ok=True)
        namespace["RUN_LOG_PATH"] = working / "RUN_LOG.jsonl"
        namespace["PRIMARY_ROOT"] = primary
        namespace["SECONDARY_ROOT"] = secondary
        namespace["PRIMARY_MANIFEST_SHA256"] = digests["primary_manifest"]
        namespace["PRIMARY_WEIGHT_SHA256"] = digests["primary_weight"]
        namespace["SECONDARY_MANIFEST_SHA256"] = digests["secondary_manifest"]
        namespace["SECONDARY_WEIGHT_SHA256"] = digests["secondary_weight"]
        namespace["COMPETITION_TEST_CANDIDATES"] = candidates
        namespace["COMPETITION_TEST"] = candidates[0]
        atomic = namespace["_atomic_json"]
        namespace["_atomic_json"] = lambda path, value: atomic(working / Path(path).name, value)

        with contextlib.redirect_stdout(io.StringIO()):
            return namespace["run_step0"]()

    def test_hidden_step0_accepts_any_positive_inventory_and_both_mount_roots(self) -> None:
        # Regression: the hidden rerun failed because STEP0 pinned the inventory
        # to exactly four datasets AND hard-coded a single mount root.
        for count in (1, 3, 4, 5, 17):
            with self.subTest(hidden_dataset_count=count):
                with tempfile.TemporaryDirectory() as directory:
                    receipt = self._execute_hidden_step0(
                        Path(directory), n_datasets=count, layout="competitions"
                    )
                self.assertEqual(receipt["status"], "PASS")
                self.assertEqual(len(receipt["selected_datasets"]), count)
                self.assertEqual(receipt["request_count"], count)

        for layout in ("competitions", "flat"):
            with self.subTest(mount_layout=layout):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    receipt = self._execute_hidden_step0(root, n_datasets=4, layout=layout)
                expected = (
                    f"input/competitions/{MODULE.COMPETITION}/test"
                    if layout == "competitions"
                    else f"input/{MODULE.COMPETITION}/test"
                )
                # STEP0 must resolve the root that exists, and the env override the
                # notebook body consumes must carry that RESOLVED root.
                self.assertTrue(receipt["competition_test_root"].endswith(expected))
                self.assertEqual(len(receipt["selected_datasets"]), 4)

    def test_hidden_step0_still_fails_closed_on_empty_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                self._execute_hidden_step0(
                    Path(directory), n_datasets=0, layout="competitions"
                )
        self.assertIn("empty", str(caught.exception))

    def test_validate_hidden_output_requires_fresh_dynamic_logged_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_dir = root / "package"
            output_dir = root / "output"
            package_dir.mkdir()
            output_dir.mkdir()
            gate = {
                "schema": MODULE.HIDDEN_GATE_SCHEMA,
                "status": "PASS",
                "eligible": True,
                "local_score": 0.9143584730207287,
                "public_output_copy_allowed": False,
                "public_output_sha256": "f" * 64,
                "g1_assumed_additive": False,
            }
            gate["gate_sha256"] = MODULE.canonical_sha256(gate)
            for name, content in (
                ("public_temporalunet_hidden_kernel.py", "pass\n"),
                ("kernel-metadata.json", "{}\n"),
                ("prelaunch-contract.json", "{}\n"),
            ):
                (package_dir / name).write_text(content, encoding="utf-8")
            package = {
                "schema": MODULE.HIDDEN_PACKAGE_SCHEMA,
                "variant": "public_v32",
                "kernel_id": "onebitbrain/test-hidden",
                "expected_initial_kernel_version": 1,
                "expected_test_count": 1,
                "hidden_test_compatible": True,
                "dynamic_test_discovery": True,
                "public_identity_pinned": False,
                "fresh_submission_generation": True,
                "public_output_copied": False,
                "local_submission_gate": gate,
                "files": {
                    name: MODULE.sha256_file(package_dir / name)
                    for name in (
                        "public_temporalunet_hidden_kernel.py",
                        "kernel-metadata.json",
                        "prelaunch-contract.json",
                    )
                },
                "weight_refinement": False,
                "g1_applied": False,
            }
            (package_dir / "PACKAGE_MANIFEST.json").write_text(
                json.dumps(package), encoding="utf-8"
            )
            submission = output_dir / "submission.csv"
            with submission.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MODULE.CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "id": 0,
                        "dataset": "hidden_a",
                        "row_type": "node",
                        "node_id": 1,
                        "t": 0,
                        "z": 0,
                        "y": 0,
                        "x": 0,
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
            run_stats = output_dir / "run_stats.csv"
            run_stats.write_text("dataset\nhidden_a\n", encoding="utf-8")
            run_log = output_dir / "RUN_LOG.jsonl"
            run_log.write_text(
                "\n".join(
                    json.dumps({"event": event})
                    for event in (
                        "config",
                        "input_inventory",
                        "lookup",
                        "stage_start",
                        "stage_end",
                        "preprocess",
                        "terminal",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            step0 = {
                "schema": "celltrack-alpha.public-temporalunet-hidden-step0.v1",
                "status": "PASS",
                "variant": "public_v32",
                "request_count": 1,
                "population_manifest_sha256": gate["gate_sha256"],
                "checkpoint_sha256": MODULE.PRIMARY_WEIGHT_SHA256,
                "secondary_checkpoint_sha256": MODULE.SECONDARY_WEIGHT_SHA256,
                "selected_datasets": ["hidden_a"],
                "hidden_test_compatible": True,
                "dynamic_test_discovery": True,
                "public_identity_pinned": False,
                "public_output_copied": False,
            }
            run = {
                "schema": "celltrack-alpha.public-temporalunet-hidden-run.v1",
                "status": "PASS",
                "variant": "public_v32",
                "selected_datasets": ["hidden_a"],
                "request_count": 1,
                "population_manifest_sha256": gate["gate_sha256"],
                "local_submission_gate_sha256": gate["gate_sha256"],
                "local_cv57_score": gate["local_score"],
                "primary_weight_sha256": MODULE.PRIMARY_WEIGHT_SHA256,
                "secondary_weight_sha256": MODULE.SECONDARY_WEIGHT_SHA256,
                "public_notebook_sha256": MODULE.PUBLIC_NOTEBOOK_SHA256,
                "public_script_version_id": 337501010,
                "submission": {"sha256": MODULE.sha256_file(submission), "rows": 1},
                "run_stats": {"sha256": MODULE.sha256_file(run_stats)},
                "run_log": {"sha256": MODULE.sha256_file(run_log)},
                "runtime_predict_script_sha256": "a" * 64,
                "nodes": 1,
                "edges": 0,
                "hidden_test_compatible": True,
                "dynamic_test_discovery": True,
                "public_identity_pinned": False,
                "fresh_submission_generation": True,
                "public_output_copied": False,
                "graph_policy": "competition_faithful_no_outdegree_cap",
                "weight_refinement": False,
                "substrate_only": True,
                "g1_applied": False,
            }
            (output_dir / "PUBLIC_TEMPORALUNET_STEP0.json").write_text(
                json.dumps(step0), encoding="utf-8"
            )
            (output_dir / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json").write_text(
                json.dumps(run), encoding="utf-8"
            )
            result = MODULE.validate_hidden_output(
                type(
                    "Args",
                    (),
                    {
                        "package_dir": package_dir,
                        "output_dir": output_dir,
                        "output": root / "validation.json",
                    },
                )()
            )
            self.assertTrue(result["eligible_for_submission_predicate"])
            self.assertTrue(result["dataset_discovery"]["inventory_exact_match"])
            self.assertFalse(result["generation"]["public_output_copied"])

    def test_outdegree_diagnostics_retains_and_logs_degree_three_parent(self) -> None:
        diagnostics = MODULE.outdegree_diagnostics(
            {"toy": ((7, 8), (7, 9), (7, 10), (1, 2))}
        )
        self.assertEqual(diagnostics["policy"], "competition_faithful_no_outdegree_cap")
        self.assertEqual(diagnostics["degree_three_parent_count"], 1)
        self.assertEqual(diagnostics["outdegree_gt_two_parent_count"], 1)
        self.assertEqual(
            diagnostics["parents_with_outdegree_gt_two"],
            [{"dataset": "toy", "parent_id": 7, "outdegree": 3, "daughters": [8, 9, 10]}],
        )

    def test_cap2_parent_log_proves_exact_parents_and_top_two_ranking(self) -> None:
        records = []
        for parent in MODULE.CAP2_EXPECTED_PARENTS:
            daughters = list(parent["daughters"])
            ranking = [
                {
                    "target_id": daughters[0],
                    "edge_prob": 0.5,
                    "distance_um": 10.0,
                    "retained": True,
                },
                {
                    "target_id": daughters[1],
                    "edge_prob": None,
                    "distance_um": 1.0,
                    "retained": True,
                },
                {
                    "target_id": daughters[2],
                    "edge_prob": None,
                    "distance_um": 2.0,
                    "retained": False,
                },
            ]
            records.append(
                {
                    "event": "experimental_cap2_parent",
                    "dataset": parent["dataset"],
                    "parent_id": parent["parent_id"],
                    "outdegree_before": parent["outdegree"],
                    "daughters_before": daughters,
                    "ranking": ranking,
                    "retained_daughters": daughters[:2],
                    "dropped_daughters": daughters[2:],
                }
            )
        self.assertEqual(len(MODULE.validate_cap2_parent_log_records(records)), 2)
        with self.assertRaisesRegex(MODULE.BaselineError, "affected-parent log differs"):
            MODULE.validate_cap2_parent_log_records(records[:1])
        malformed = json.loads(json.dumps(records))
        malformed[0]["ranking"][0]["retained"] = False
        with self.assertRaisesRegex(MODULE.BaselineError, "top-two retention"):
            MODULE.validate_cap2_parent_log_records(malformed)

    def test_validate_shard_authenticates_receipts_and_strict_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_dir = root / "package"
            output_dir = root / "remote"
            package_dir.mkdir()
            output_dir.mkdir()
            for name, content in (
                ("public_temporalunet_kernel.py", "pass\n"),
                ("kernel-metadata.json", "{}\n"),
                ("prelaunch-contract.json", "{}\n"),
            ):
                (package_dir / name).write_text(content, encoding="utf-8")
            shard_sha = "1" * 64
            global_sha = "2" * 64
            package = {
                "schema": "celltrack-alpha.public-temporalunet-package.v1",
                "variant": "public_v32",
                "kernel_id": "onebitbrain/test-shard",
                "weight_refinement": False,
                "shard": {
                    "identities": ["train:toy"],
                    "manifest_sha256": shard_sha,
                    "global_population_manifest_sha256": global_sha,
                    "shard_index": 0,
                    "shard_count": 1,
                },
                "files": {
                    name: MODULE.sha256_file(package_dir / name)
                    for name in (
                        "public_temporalunet_kernel.py",
                        "kernel-metadata.json",
                        "prelaunch-contract.json",
                    )
                },
            }
            (package_dir / "PACKAGE_MANIFEST.json").write_text(
                json.dumps(package), encoding="utf-8"
            )
            submission = output_dir / "submission.csv"
            with submission.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MODULE.CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "id": 0,
                        "dataset": "toy",
                        "row_type": "node",
                        "node_id": 0,
                        "t": 0,
                        "z": 0,
                        "y": 0,
                        "x": 0,
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
            run_stats = output_dir / "run_stats.csv"
            run_stats.write_text("dataset\ntoy\n", encoding="utf-8")
            run_log = output_dir / "RUN_LOG.jsonl"
            run_log.write_text(
                "\n".join(
                    json.dumps({"event": event})
                    for event in ("config", "input_inventory", "lookup", "stage_end", "preprocess", "terminal")
                )
                + "\n",
                encoding="utf-8",
            )
            step0 = {
                "status": "PASS",
                "variant": "public_v32",
                "population_manifest_sha256": shard_sha,
                "global_population_manifest_sha256": global_sha,
                "checkpoint_sha256": MODULE.PRIMARY_WEIGHT_SHA256,
                "secondary_checkpoint_sha256": MODULE.SECONDARY_WEIGHT_SHA256,
                "weight_refinement_allowed": False,
                "request_count": 1,
            }
            run = {
                "status": "PASS",
                "variant": "public_v32",
                "population_manifest_sha256": shard_sha,
                "global_population_manifest_sha256": global_sha,
                "primary_weight_sha256": MODULE.PRIMARY_WEIGHT_SHA256,
                "secondary_weight_sha256": MODULE.SECONDARY_WEIGHT_SHA256,
                "public_notebook_sha256": MODULE.PUBLIC_NOTEBOOK_SHA256,
                "public_script_version_id": 337501010,
                "weight_refinement": False,
                "substrate_only": True,
                "weight_refinement_minimum_local_score": 0.916,
                "selected_datasets": ["toy"],
                "submission": {"sha256": MODULE.sha256_file(submission), "rows": 1},
                "run_stats": {"sha256": MODULE.sha256_file(run_stats)},
                "run_log": {"sha256": MODULE.sha256_file(run_log)},
                "runtime_predict_script_sha256": "a" * 64,
                "nodes": 1,
                "edges": 0,
            }
            (output_dir / "PUBLIC_TEMPORALUNET_STEP0.json").write_text(
                json.dumps(step0), encoding="utf-8"
            )
            (output_dir / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json").write_text(
                json.dumps(run), encoding="utf-8"
            )
            result = MODULE.validate_shard_output(
                type(
                    "Args",
                    (),
                    {
                        "package_dir": package_dir,
                        "output_dir": output_dir,
                        "output": root / "validation.json",
                    },
                )()
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["graphs"]["nodes"], 1)
            self.assertEqual(result["graphs"]["edges"], 0)

    def test_merge_streams_disjoint_receipted_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = root / "populations.json"
            payload = population_payload()
            population.write_text(json.dumps(payload), encoding="utf-8")
            datasets = [identity.split(":", 1)[1] for identity in payload["union"]]
            shard_dirs = []
            for shard_index, selected in enumerate((datasets[::2], datasets[1::2])):
                shard_dir = root / f"shard_{shard_index}"
                shard_dir.mkdir()
                shard_dirs.append(shard_dir)
                submission = shard_dir / "submission.csv"
                with submission.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=MODULE.CSV_COLUMNS)
                    writer.writeheader()
                    for row_id, dataset in enumerate(selected):
                        writer.writerow(
                            {
                                "id": row_id,
                                "dataset": dataset,
                                "row_type": "node",
                                "node_id": 0,
                                "t": 0,
                                "z": 0,
                                "y": 0,
                                "x": 0,
                                "source_id": -1,
                                "target_id": -1,
                            }
                        )
                receipt = {
                    "schema": MODULE.RUN_SCHEMA,
                    "status": "PASS",
                    "variant": "public_v32",
                    "global_population_manifest_sha256": payload["manifest_sha256"],
                    "weight_refinement": False,
                    "selected_datasets": selected,
                    "submission": {
                        "sha256": MODULE.sha256_file(submission),
                        "rows": len(selected),
                    },
                    "nodes": len(selected),
                    "edges": 0,
                }
                (shard_dir / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json").write_text(
                    json.dumps(receipt), encoding="utf-8"
                )
            output = root / "merged.csv"
            merge_receipt = root / "merge.json"
            args = type(
                "Args",
                (),
                {
                    "population_manifest": population,
                    "variant": "public_v32",
                    "shard_dir": shard_dirs,
                    "output": output,
                    "receipt": merge_receipt,
                },
            )()
            result = MODULE.merge_shards(args)
            self.assertEqual(result["submission"]["rows"], 162)
            self.assertEqual(result["nodes"], 162)
            self.assertEqual(result["edges"], 0)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["id"]) for row in rows], list(range(162)))

    def test_fragmentation_is_separate_from_fully_unmatched_detection(self) -> None:
        gt = {
            "nodes": {
                "node_id": [10, 11, 12, 20],
                "t": [0, 1, 2, 0],
                "z": [1, 1, 1, 9],
                "y": [2, 2, 2, 9],
                "x": [3, 3, 3, 9],
            },
            "edges": [(10, 11), (11, 12)],
        }
        nodes = [
            {"node_id": 0, "t": 0, "z": 1, "y": 2, "x": 3},
            {"node_id": 1, "t": 1, "z": 1, "y": 2, "x": 3},
            {"node_id": 2, "t": 2, "z": 1, "y": 2, "x": 3},
        ]
        observed = MODULE.fragmentation_diagnostic(
            nodes,
            [{"source_id": 0, "target_id": 1}],
            gt,
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(observed["gt_lineage_components"], 2)
        self.assertEqual(observed["matched_gt_lineage_components"], 1)
        self.assertEqual(observed["fully_unmatched_gt_lineage_components"], 1)
        self.assertEqual(observed["fragmented_matched_gt_lineage_components"], 1)
        self.assertEqual(observed["excess_prediction_segments"], 1)

    def test_division_row_summary_uses_global_jaccard_counts(self) -> None:
        observed = MODULE.summarize_division_rows(
            [
                {
                    "division_tp": 2,
                    "division_fp": 3,
                    "division_fn": 1,
                    "predicted_dividing_nodes": 7,
                },
                {
                    "division_tp": 1,
                    "division_fp": 0,
                    "division_fn": 2,
                    "predicted_dividing_nodes": 4,
                },
            ]
        )
        self.assertEqual(observed["division_tp"], 3)
        self.assertEqual(observed["division_fp"], 3)
        self.assertEqual(observed["division_fn"], 3)
        self.assertEqual(observed["predicted_dividing_nodes"], 11)
        self.assertEqual(observed["gt_division_count"], 6)
        self.assertEqual(observed["division_jaccard"], 1 / 3)

    def test_immutable_v32_hash_and_three_modes_compile(self) -> None:
        self.assertEqual(MODULE.sha256_file(MODULE.PUBLIC_NOTEBOOK), MODULE.PUBLIC_NOTEBOOK_SHA256)
        expected = {
            "single_seed": (
                "os.environ['BIOHUB_SECONDARY_WEIGHTS'] = ''",
                "os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT'] = '0.0'",
            ),
            "detection_blend": (
                "os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.0'",
                "os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT'] = '0.5'",
            ),
            "public_v32": (
                "os.environ['BIOHUB_SECONDARY_EDGE_WEIGHT'] = '0.15'",
                "os.environ['BIOHUB_SECONDARY_LINK_MODE'] = 'low_margin_consensus'",
            ),
        }
        for variant, tokens in expected.items():
            with self.subTest(variant=variant):
                script = MODULE.build_script(
                    variant=variant,
                    identities=("train:sample",),
                    population_sha256="1" * 64,
                    global_population_sha256="2" * 64,
                )
                compile(script, f"{variant}.py", "exec")
                for token in tokens:
                    self.assertIn(token, script)
                self.assertIn("STEP0 = run_step0()", script)
                self.assertIn("WEIGHT_REFINEMENT_ALLOWED = False", script)
                self.assertIn("WEIGHT_REFINEMENT_MINIMUM_LOCAL_SCORE = 0.916", script)
                self.assertIn("RUN_LOG.jsonl", script)
                self.assertIn("inventory_input", script)
                self.assertIn("_run_failure", script)
                self.assertNotIn("signal.alarm(", script)
                self.assertNotIn("_hard_deadline", script)
                self.assertIn(
                    "if secondary_model is not None and secondary_edge_weight > 0.0:",
                    script,
                )
                self.assertIn(
                    "for _scratch_path in (REPO_DIR, SECONDARY_WEIGHTS_ROOT, SELECTED_TRAIN):",
                    script,
                )
                self.assertLess(
                    script.index("_atomic_json(WORKING_DIR / \"PUBLIC_TEMPORALUNET_RUN_RECEIPT.json\""),
                    script.index("for _scratch_path in (REPO_DIR, SECONDARY_WEIGHTS_ROOT, SELECTED_TRAIN):"),
                )
                self.assertNotIn('.rglob(', script)
                self.assertNotIn("optimizer.step(", script)
                self.assertNotIn("loss.backward(", script)

    def test_cap2_is_separate_postlink_experiment_with_structured_parent_logging(self) -> None:
        baseline = MODULE.build_script(
            variant="public_v32",
            identities=("train:sample",),
            population_sha256="1" * 64,
            global_population_sha256="2" * 64,
        )
        capped = MODULE.build_script(
            variant=MODULE.CAP2_VARIANT,
            identities=("train:sample",),
            population_sha256="1" * 64,
            global_population_sha256="2" * 64,
        )
        compile(capped, "public_v32_cap2.py", "exec")
        self.assertNotIn("experimental_cap2_parent", baseline)
        self.assertIn("experimental_cap2_parent", capped)
        self.assertIn("PUBLIC_TEMPORALUNET_CAP_OUTDEGREE_TWO = True", capped)
        self.assertIn("ranked = sorted(source_edges, key=edge_sort_key, reverse=True)", capped)
        self.assertIn("retained = ranked[:2]", capped)
        self.assertIn("edges = add_safe_divisions_postlink(", capped)
        self.assertLess(
            capped.index("edges = add_safe_divisions_postlink("),
            capped.index('"experimental_cap2_parent"'),
        )
        pristine_safe_division = MODULE.source_cells(
            MODULE.load_object(MODULE.PUBLIC_NOTEBOOK, "public notebook")
        )[5].split("def add_safe_divisions_postlink(", 1)[1].split(
            "def filter_short_track_components(", 1
        )[0]
        capped_safe_division = capped.split("def add_safe_divisions_postlink(", 1)[1].split(
            "def filter_short_track_components(", 1
        )[0]
        self.assertEqual(capped_safe_division, pristine_safe_division)
        cap_block_start = capped.index(
            "    if PUBLIC_TEMPORALUNET_CAP_OUTDEGREE_TWO and edges:\n"
        )
        cap_block_end = capped.index(
            "    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:\n",
            cap_block_start,
        )
        normalized = capped[:cap_block_start] + capped[cap_block_end:]
        normalized = normalized.replace(MODULE.CAP2_VARIANT, "public_v32")
        normalized = normalized.replace(
            "PUBLIC_TEMPORALUNET_CAP_OUTDEGREE_TWO = True",
            "PUBLIC_TEMPORALUNET_CAP_OUTDEGREE_TWO = False",
        )
        normalized = normalized.replace(
            "os.environ['BIOHUB_EXPERIMENTAL_CAP_OUTDEGREE_TWO'] = '1'\n",
            "",
        )
        self.assertEqual(normalized, baseline)

    def test_gap2_variant_flips_only_the_recovery_flag_and_is_otherwise_pristine(self) -> None:
        """R1's Candidate-3-adjacent GPU test: BIOHUB_OUTPUT_GAP2_RECOVERY is an
        existing, already-built, currently-disabled mechanism in the pristine
        public notebook (cell 1). The gap2 variant must flip exactly that one
        line to "1" and change nothing else -- no new code, unlike cap2."""
        baseline = MODULE.build_script(
            variant="public_v32",
            identities=("train:sample",),
            population_sha256="1" * 64,
            global_population_sha256="2" * 64,
        )
        gap2 = MODULE.build_script(
            variant=MODULE.GAP2_VARIANT,
            identities=("train:sample",),
            population_sha256="1" * 64,
            global_population_sha256="2" * 64,
        )
        compile(gap2, "public_v32_gap2.py", "exec")
        self.assertIn('os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "0"', baseline)
        self.assertIn('os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"', gap2)
        self.assertNotIn('os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "0"', gap2)
        normalized = gap2.replace(
            'os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"',
            'os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "0"',
        )
        normalized = normalized.replace(MODULE.GAP2_VARIANT, "public_v32")
        self.assertEqual(normalized, baseline)

    def test_gap2_scoring_preserves_the_competition_faithful_graph_contract(self) -> None:
        self.assertTrue(MODULE.competition_faithful_graph_variant("public_v32"))
        self.assertTrue(MODULE.competition_faithful_graph_variant(MODULE.GAP2_VARIANT))
        self.assertFalse(MODULE.competition_faithful_graph_variant(MODULE.CAP2_VARIANT))

    def test_population_manifest_requires_exact_disjoint_105_and_57(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "populations.json"
            path.write_text(json.dumps(population_payload()), encoding="utf-8")
            observed = MODULE.validate_population_manifest(path)
            self.assertEqual(observed["counts"]["union"], 162)

            bad = population_payload()
            bad["populations"]["cv57"][0] = bad["populations"]["held_fit105"][0]
            bad["union"] = bad["populations"]["held_fit105"] + bad["populations"]["cv57"]
            bad["manifest_sha256"] = MODULE.canonical_sha256(
                {key: value for key, value in bad.items() if key != "manifest_sha256"}
            )
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.BaselineError, "overlaps"):
                MODULE.validate_population_manifest(path)

    def test_build_package_binds_exact_artifacts_and_no_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            population = root / "populations.json"
            population.write_text(json.dumps(population_payload()), encoding="utf-8")
            package_dir = root / "package"
            args = type(
                "Args",
                (),
                {
                    "variant": "detection_blend",
                    "population_manifest": population,
                    "shard_count": 162,
                    "shard_index": 0,
                    "kernel_id": "onebitbrain/test-public-temporalunet",
                    "kernel_dir": package_dir,
                },
            )()
            result = MODULE.build_package(args)
            source = (package_dir / "public_temporalunet_kernel.py").read_text(encoding="utf-8")
            contract = json.loads((package_dir / "prelaunch-contract.json").read_text())
            self.assertFalse(result["weight_refinement"])
            self.assertEqual(contract["checkpoint_sha256"], MODULE.PRIMARY_WEIGHT_SHA256)
            self.assertEqual(contract["secondary_checkpoint_sha256"], MODULE.SECONDARY_WEIGHT_SHA256)
            self.assertEqual(contract["request_count"], 1)
            self.assertFalse(contract["runtime_governance"]["per_run_wall_clock_kill"])
            self.assertEqual(contract["runtime_governance"]["projection_gate_h"], 8.0)
            self.assertTrue(contract["run_logging"]["download_on_success"])
            self.assertIn(MODULE.PRIMARY_MANIFEST_SHA256, source)
            self.assertIn(MODULE.SECONDARY_MANIFEST_SHA256, source)
            self.assertNotIn("timeout=max(1, int(_hard_deadline - time.time()))", source)


if __name__ == "__main__":
    unittest.main()
