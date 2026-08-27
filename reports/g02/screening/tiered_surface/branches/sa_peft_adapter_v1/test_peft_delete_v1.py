from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
KERNEL = ROOT / "kaggle_kernel/g02-peft-delete-v1/g02_peft_delete_v1.py"
SPEC = importlib.util.spec_from_file_location("g02_peft_delete_v1", KERNEL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PEFTDeleteTests(unittest.TestCase):
    def test_preregistered_parameter_fraction(self) -> None:
        expected = (20 * 64 + 64) + (64 * 20 + 20) + (20 + 1)
        self.assertEqual(expected, MODULE.EXPECTED_TRAINABLE_PARAMETERS)
        self.assertLess(
            expected / MODULE.BASE_PARAMETER_COUNT,
            MODULE.TRAINABLE_FRACTION_MAX,
        )

    def test_fixed_sample_digest_and_counts(self) -> None:
        root = (
            ROOT
            / "reports/g02/screening/tiered_surface/branches/sa_repr/"
            "phase2_separation_gate/run_v1/population/delete"
        )
        records = []
        offset = 0
        for path in sorted(root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                y = np.asarray(data["oracle_label"], dtype=np.uint8)
                records.append(
                    {
                        "identity": path.stem.replace("__", ":", 1),
                        "offset": offset,
                        "y": y,
                        "source": np.asarray(data["source_id"]),
                        "target": np.asarray(data["target_id"]),
                        "source_t": np.asarray(data["source_t"]),
                        "target_t": np.asarray(data["target_t"]),
                    }
                )
                offset += len(y)
        indices, digest = MODULE.fixed_negative_sample(records)
        self.assertEqual(len(indices), MODULE.SAMPLE_TOTAL)
        self.assertEqual(digest, MODULE.SAMPLE_SHA256)
        labels = np.concatenate([record["y"] for record in records])
        sampled = labels[np.asarray(indices, dtype=np.int64)]
        self.assertEqual(int(sampled.sum()), MODULE.EXPECTED_BENEFICIAL)
        self.assertEqual(int((sampled == 0).sum()), MODULE.SAMPLE_NEGATIVES)

    def test_graph_metric_noop_is_exact(self) -> None:
        labels_delta = np.asarray([0.01, -0.01, 0.0], dtype=np.float64)
        scores = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
        video = np.zeros(3, dtype=np.int16)
        baseline = {
            "edge_tp": 10.0,
            "edge_fp": 2.0,
            "edge_fn": 3.0,
            "edge_jaccard": 10.0 / 15.0,
            "adj_edge_jaccard": 0.6,
            "node_recall": 0.9,
        }
        result = MODULE.graph_metrics(
            labels_delta=labels_delta,
            scores=scores,
            video=video,
            budgets=np.asarray([2]),
            identities=("train:44b6_test",),
            baselines={"train:44b6_test": baseline},
            include_mask=np.ones(3, dtype=bool),
            multiplier=0.0,
        )
        self.assertEqual(
            result["delta"],
            {
                "adjusted_edge_jaccard": 0.0,
                "raw_edge_jaccard": 0.0,
                "node_recall": 0.0,
            },
        )

    def test_kernel_is_single_file_and_has_no_tree_walk(self) -> None:
        metadata = json.loads(
            (
                ROOT / "kaggle_kernel/g02-peft-delete-v1/kernel-metadata.json"
            ).read_text()
        )
        self.assertEqual(metadata["code_file"], KERNEL.name)
        siblings = list(KERNEL.parent.glob("*.py"))
        self.assertEqual(siblings, [KERNEL])
        source = KERNEL.read_text()
        self.assertNotIn("os.walk(", source)
        self.assertNotIn(".rglob(", source)
        self.assertEqual(metadata["competition_sources"], [])

    def test_manifest_hash_constants_are_filled(self) -> None:
        self.assertRegex(MODULE.ASSET_MANIFEST_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(MODULE.AMENDMENT_SHA256, r"^[0-9a-f]{64}$")

    def test_wheels_have_canonical_names_and_manifest_matching_bytes(self) -> None:
        candidates = sorted(
            Path("/private/tmp").glob(
                "g02-peft-delete-assets-v1-*/asset_manifest.json"
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not candidates:
            self.skipTest("PEFT launch asset bundle has not been built")
        manifest = json.loads(candidates[-1].read_text())
        wheel_rows = [
            row
            for row in manifest["members"]
            if row["path"].startswith("wheels/")
        ]
        self.assertEqual(len(wheel_rows), MODULE.EXPECTED_WHEEL_COUNT)
        archive = candidates[-1].with_name("peft_bundle.tar")
        import tarfile

        with tarfile.open(archive, mode="r:") as handle:
            for row in wheel_rows:
                name = Path(row["path"]).name
                self.assertTrue(name.endswith(".whl"))
                self.assertGreaterEqual(len(name[:-4].split("-")), 5)
                member = handle.extractfile(f"peft_bundle/{row['path']}")
                self.assertIsNotNone(member)
                observed = hashlib.sha256(member.read()).hexdigest()
                self.assertEqual(observed, row["sha256"])


if __name__ == "__main__":
    unittest.main()
