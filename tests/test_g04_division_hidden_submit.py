from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import g04_build_division_hidden_submit as BUILDER
from scripts import g04_division_hidden_postprocess as POST


ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "reports/g04/lanes/division_permissive/promoted_unfiltered_geometry_cap5"
    / "public_parent/submission.csv"
)
CACHE = (
    ROOT
    / "reports/g04/lanes/division_permissive/promoted_unfiltered_geometry_cap5"
    / "test_appearance_cache"
)
CANDIDATE = (
    ROOT
    / "reports/g04/lanes/division_permissive/promoted_unfiltered_geometry_cap5"
    / "submission.csv"
)


class HiddenDivisionPostprocessTests(unittest.TestCase):
    def test_higher_outdegree_requires_explicit_external_policy(self) -> None:
        graph = {
            "dataset": "public_v32_fixture",
            "nodes": {
                1: {"node_id": 1, "t": 0, "z": 1, "y": 10, "x": 10},
                2: {"node_id": 2, "t": 1, "z": 1, "y": 11, "x": 10},
                3: {"node_id": 3, "t": 1, "z": 1, "y": 12, "x": 10},
                4: {"node_id": 4, "t": 1, "z": 1, "y": 13, "x": 10},
            },
            "edges": {(1, 2), (1, 3), (1, 4)},
        }
        with self.assertRaisesRegex(POST.PostprocessError, "outdegree > 2"):
            POST.validate_graphs({graph["dataset"]: graph})
        totals = POST.validate_graphs({graph["dataset"]: graph}, max_outdegree=None)
        self.assertEqual(totals["edges"], 3)

    def test_hidden_dataset_label_is_not_pinned(self) -> None:
        graph = {
            "dataset": "never_seen_hidden_shape_7",
            "nodes": {
                0: {"node_id": 0, "t": 0, "z": 1, "y": 10, "x": 10},
                1: {"node_id": 1, "t": 1, "z": 1, "y": 11, "x": 10},
                2: {"node_id": 2, "t": 2, "z": 1, "y": 12, "x": 10},
                3: {"node_id": 3, "t": 2, "z": 1, "y": 12, "x": 10},
            },
            "edges": {(0, 1), (1, 2)},
        }
        candidates, detail = POST.appearance_prefilter(graph)
        self.assertEqual(detail["cached_candidates"], 1)
        self.assertEqual(candidates[0]["identity"], "test:never_seen_hidden_shape_7")
        enriched = dict(candidates[0])
        enriched["intensity_summaries"] = {
            role: {"mass_above_background": mass, "xy_roundness": 0.5}
            for role, mass in (("parent", 10.0), ("daughter_1", 5.0), ("daughter_2", 5.0))
        }
        transformed, diagnostic = POST.build_disjoint_fork_graph(graph, [enriched])
        self.assertEqual(diagnostic["selected_action_count"], 1)
        self.assertEqual(transformed["edges"], {(0, 1), (1, 2), (1, 3)})
        self.assertEqual(POST.validate_graphs({graph["dataset"]: transformed})["dividing_nodes"], 1)

    def test_patch_summary_is_finite(self) -> None:
        array = np.arange(3 * 5 * 25 * 25, dtype=np.float32).reshape(3, 5, 25, 25)
        summary = POST.patch_summary(
            array, {"node_id": 1, "t": 1, "z": 2, "y": 12, "x": 12}
        )
        self.assertEqual(summary["voxel_count"], 5 * 25 * 25)
        self.assertGreater(summary["mass_above_background"], 0.0)
        self.assertGreaterEqual(summary["xy_roundness"], 0.0)

    def test_public_parent_to_promoted_candidate_byte_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            output = temporary / "submission.csv"
            receipt = temporary / "receipt.json"
            result = POST.postprocess_submission(
                PARENT,
                output,
                summary_cache=CACHE,
                receipt_path=receipt,
                expected_output_sha256=BUILDER.PROMOTED_PUBLIC_SHA256,
                deadline_seconds=120,
            )
            self.assertEqual(output.read_bytes(), CANDIDATE.read_bytes())
            self.assertEqual(result["output"]["rows"], 263_297)
            self.assertEqual(result["output"]["dividing_nodes"], 15)
            self.assertEqual(result["transform"]["selected_actions"], 15)
            self.assertFalse(result["dataset_discovery"]["public_identity_pinned"])
            self.assertEqual(json.loads(receipt.read_text())["status"], "PASS")

    def test_provider_inventory_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            cache = temporary / "cache"
            cache.mkdir()
            with self.assertRaisesRegex(POST.PostprocessError, "inventory mismatch"):
                POST.postprocess_submission(
                    PARENT,
                    temporary / "submission.csv",
                    summary_cache=cache,
                    deadline_seconds=20,
                )


class HiddenDivisionPackageTests(unittest.TestCase):
    def test_build_embeds_exact_scored_base_and_frozen_postprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            output = Path(temporary_text) / BUILDER.KERNEL_SLUG
            report = BUILDER.build_package(output)
            identity = report["identity"]
            self.assertEqual(identity["base"]["package_identity_sha256"], BUILDER.BASE_PACKAGE_IDENTITY)
            self.assertEqual(identity["base"]["submission_ref"], 54_942_864)
            self.assertEqual(identity["kernel_ref"], BUILDER.KERNEL_REF)
            self.assertLess(identity["package_bytes"], BUILDER.SOURCE_BUDGET_BYTES)
            self.assertFalse(identity["postprocess"]["policy"]["public_identity_pinned"])
            self.assertEqual(
                identity["postprocess"]["source_sha256"],
                BUILDER.sha256_file(Path(POST.__file__)),
            )
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["competition_sources"], [BUILDER.COMPETITION])
            self.assertEqual(metadata["dataset_sources"], [BUILDER.SUPPORT])
            self.assertTrue(metadata["enable_gpu"])
            self.assertFalse(metadata["enable_internet"])
            source = (output / "main.py").read_text(encoding="ascii")
            compile(source, str(output / "main.py"), "exec")
            self.assertIn("fresh APNCC output", source)
            self.assertIn("inventory_exact_match", source)


if __name__ == "__main__":
    unittest.main()
