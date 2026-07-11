from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.pipeline import _validate_array_window, predict_dataset_window  # noqa: E402
from celltrack_alpha.proposals import _validate_volume  # noqa: E402


class PipelineValidationTests(unittest.TestCase):
    def test_valid_tzyx_window(self) -> None:
        array = np.zeros((3, 2, 4, 5), dtype=np.uint16)
        self.assertEqual(_validate_array_window("sample", array, 1, 3), (1, 3))
        self.assertEqual(_validate_array_window("sample", array, 0, None), (0, 3))

    def test_fractional_or_boolean_window_values_are_rejected(self) -> None:
        array = np.zeros((3, 2, 4, 5), dtype=np.uint16)
        for start, end in ((0, 1.5), (False, 2), (0, True)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    _validate_array_window("sample", array, start, end)

    def test_window_boundaries_are_rejected(self) -> None:
        array = np.zeros((3, 2, 4, 5), dtype=np.uint16)
        for start, end in ((-1, 2), (0, 0), (2, 2), (2, 1), (0, 4), (3, None)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    _validate_array_window("sample", array, start, end)

    def test_invalid_array_contract_is_rejected(self) -> None:
        cases = [
            np.zeros((2, 3, 4), dtype=np.uint16),
            np.zeros((1, 0, 3, 4), dtype=np.uint16),
            np.zeros((1, 2, 3, 4), dtype=object),
            np.zeros((1, 2, 3, 4), dtype=np.complex64),
        ]
        for array in cases:
            with self.subTest(shape=array.shape, dtype=array.dtype):
                with self.assertRaises(ValueError):
                    _validate_array_window("sample", array, 0, None)

    def test_scalar_and_malformed_output_offsets_raise_clear_value_error(self) -> None:
        array = np.zeros((1, 2, 4, 5), dtype=np.uint16)
        for value in (1, "1,2,3", (1, 2), (1, 2, False), (1.0, 2, 3)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "exactly three integers"):
                    predict_dataset_window(
                        "sample",
                        array,
                        None,
                        output_offset_override=value,  # type: ignore[arg-type]
                    )

    def test_volume_validation_rejects_nonfinite_and_nonreal_data(self) -> None:
        valid = np.zeros((2, 3, 4), dtype=np.float32)
        self.assertIs(_validate_volume(valid), valid)
        invalid = valid.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            _validate_volume(invalid)
        for value in (np.inf, -np.inf):
            invalid = valid.copy()
            invalid[0, 0, 0] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    _validate_volume(invalid)
        with self.assertRaisesRegex(ValueError, "real numeric"):
            _validate_volume(np.zeros((2, 3, 4), dtype=np.complex64))


if __name__ == "__main__":
    unittest.main()
