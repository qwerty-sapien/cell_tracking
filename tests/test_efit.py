"""Fit-only scope rules for compact sparse edge calibration."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.edgefit import EdgeFitError, build_scope, load_edge_registry, verify_parent_manifest_binding


class EdgeFitScopeTests(unittest.TestCase):
    @staticmethod
    def _split() -> dict:
        return {
            "artifact": "grouped_3fold_audit20_v1",
            "dataset_registry_sha256": "r" * 64,
            "audit": {"group_ids": ["audit"]},
            "folds": {
                str(index): {
                    "fit_group_ids": ["fit-a", "fit-b", "fit-c", "fit-d"],
                    "held_out_group_ids": ["held"],
                }
                for index in range(3)
            },
        }

    @staticmethod
    def _registry() -> dict:
        return {
            "artifact": "dataset_registry_v1",
            "records": [
                {"video_id": f"train:{group}", "group_id": group}
                for group in ("fit-a", "fit-b", "fit-c", "fit-d", "held", "audit")
            ],
        }

    @staticmethod
    def _projection() -> dict:
        records = []
        for group in ("fit-a", "fit-b", "fit-c", "fit-d"):
            records.append({
                "split": "train",
                "dataset": group,
                "video_id": f"train:{group}",
                "exact_identity_sha256": "a" * 64,
                "compact_metadata": {"image_shape": [2, 3, 4, 5]},
                "group_id": group,
            })
        return {
            "artifact": "celltrack-alpha.edgefit-registry-projection.v1",
            "source_registry_sha256": "a" * 64,
            "parent_projection_sha256": "b" * 64,
            "records": records,
        }

    @staticmethod
    def _sources() -> tuple[dict[str, str], ...]:
        return tuple(
            {"group_id": group, "stem": group, "video_id": f"train:{group}"}
            for group in ("fit-a", "fit-b", "fit-c", "fit-d")
        )

    def test_fit_scope_is_deterministic_and_group_disjoint(self) -> None:
        records = tuple(SimpleNamespace(video_id=f"train:{group}") for group in ("fit-a", "fit-b", "fit-c", "fit-d", "held", "audit"))
        first = build_scope(records, self._registry(), self._split(), self._sources(), fold=0, split_sha256="s" * 64, registry_sha256="r" * 64, provenance_sha256="p" * 64, val_groups=1)
        second = build_scope(records, self._registry(), self._split(), self._sources(), fold=0, split_sha256="s" * 64, registry_sha256="r" * 64, provenance_sha256="p" * 64, val_groups=1)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_groups) & set(first.val_groups))
        self.assertEqual(set(first.train_groups) | set(first.val_groups), {"fit-a", "fit-b", "fit-c", "fit-d"})

    def test_rejects_parent_source_outside_fit_groups(self) -> None:
        bad = (*self._sources()[:3], {"group_id": "held", "stem": "held", "video_id": "train:held"})
        records = tuple(SimpleNamespace(video_id=f"train:{group}") for group in ("fit-a", "fit-b", "fit-c", "fit-d", "held", "audit"))
        with self.assertRaisesRegex(EdgeFitError, "fit-only"):
            build_scope(records, self._registry(), self._split(), bad, fold=0, split_sha256="s" * 64, registry_sha256="r" * 64, provenance_sha256="p" * 64, val_groups=1)

    def test_projection_binds_canonical_registry_and_contains_only_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "reg.json"
            path.write_text(json.dumps(self._projection(), sort_keys=True, separators=(",", ":")), encoding="ascii")
            registry, records, registry_sha = load_edge_registry(path)
        split = self._split()
        split["dataset_registry_sha256"] = registry_sha
        scope = build_scope(records, registry, split, self._sources(), fold=0, split_sha256="s" * 64, registry_sha256=registry_sha, provenance_sha256="p" * 64, val_groups=1)
        self.assertEqual(registry_sha, "a" * 64)
        self.assertEqual(len(scope.records), 4)

    def test_projection_rejects_extra_runtime_record(self) -> None:
        projection = self._projection()
        projection["records"].append({
            "split": "train", "dataset": "held", "video_id": "train:held",
            "exact_identity_sha256": "b" * 64, "compact_metadata": {"image_shape": [2, 3, 4, 5]}, "group_id": "held",
        })
        records = tuple(SimpleNamespace(video_id=row["video_id"]) for row in projection["records"])
        with self.assertRaisesRegex(EdgeFitError, "exactly the parent sources"):
            build_scope(records, projection, self._split(), self._sources(), fold=0, split_sha256="s" * 64, registry_sha256="r" * 64, provenance_sha256="p" * 64, val_groups=1)

    def test_parent_manifest_binding_rejects_mismatch(self) -> None:
        metrics = {"identity": {"registry_sha256": "a" * 64, "split_sha256": "b" * 64, "provenance_sha256": "c" * 64}}
        self.assertEqual(
            verify_parent_manifest_binding(metrics, registry_sha256="a" * 64, split_sha256="b" * 64, provenance_sha256="c" * 64),
            metrics["identity"],
        )
        with self.assertRaisesRegex(EdgeFitError, "does not bind"):
            verify_parent_manifest_binding(metrics, registry_sha256="d" * 64, split_sha256="b" * 64, provenance_sha256="c" * 64)


if __name__ == "__main__":
    unittest.main()
