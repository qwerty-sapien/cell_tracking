from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.localization import (  # noqa: E402
    UnsafeCalibrationError,
    load_offset_calibration,
    offset_for_dataset,
    sha256_file,
)
from celltrack_alpha.routing import default_route_configs  # noqa: E402


class LocalizationProvenanceTests(unittest.TestCase):
    def _write_split_manifest(self, root: Path) -> Path:
        path = root / "split.json"
        payload = {
            "folds": {
                "0": {
                    "calibration_train": ["44b6_train", "6bba_train"],
                    "validation": ["44b6_val", "6bba_val"],
                }
            },
            "excluded_from_general_score": {
                "duplicate_public_ids": ["44b6_dup"]
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_default_route_configs_do_not_include_prefix_offsets(self) -> None:
        for config in default_route_configs().values():
            self.assertEqual(config.prefix_output_offsets_zyx, ())

    def test_legacy_offset_calibration_without_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = self._write_split_manifest(root)
            legacy = root / "legacy.json"
            legacy.write_text(
                json.dumps({"prefix_offsets": {"44b6": {"offset_zyx": [0, -16, 8]}}}),
                encoding="utf-8",
            )
            with self.assertRaises(UnsafeCalibrationError):
                load_offset_calibration(legacy, split_manifest_path=split, expected_fold_id="0")

    def test_fold_safe_calibration_only_applies_to_validation_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = self._write_split_manifest(root)
            payload = {
                "dataset_offsets": {"44b6_val": {"offset_zyx": [0, -3, 2]}},
                "prefix_offsets": {"6bba": {"offset_zyx": [1, 0, -1]}},
                "provenance": {
                    "status": "fold_safe",
                    "fold_id": "0",
                    "split_manifest_sha256": sha256_file(split),
                    "source_datasets": ["44b6_train", "6bba_train"],
                    "calibration_train": ["44b6_train", "6bba_train"],
                    "validation_datasets": ["44b6_val", "6bba_val"],
                },
            }
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_offset_calibration(
                calibration_path,
                split_manifest_path=split,
                expected_fold_id="0",
            )
            self.assertEqual(offset_for_dataset("44b6_val", calibration), (0, -3, 2))
            self.assertEqual(offset_for_dataset("6bba_val", calibration), (1, 0, -1))
            with self.assertRaises(UnsafeCalibrationError):
                offset_for_dataset("44b6_train", calibration)
            with self.assertRaises(UnsafeCalibrationError):
                offset_for_dataset("44b6_other", calibration)

    def test_duplicate_public_source_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = self._write_split_manifest(root)
            payload = {
                "prefix_offsets": {"44b6": {"offset_zyx": [0, -3, 2]}},
                "provenance": {
                    "status": "fold_safe",
                    "fold_id": "0",
                    "source_datasets": ["44b6_dup"],
                    "calibration_train": ["44b6_train", "6bba_train", "44b6_dup"],
                    "validation_datasets": ["44b6_val"],
                },
            }
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(UnsafeCalibrationError):
                load_offset_calibration(calibration_path, split_manifest_path=split)


if __name__ == "__main__":
    unittest.main()
