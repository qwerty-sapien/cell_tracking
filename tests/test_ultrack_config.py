from __future__ import annotations

import math
import re
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_ultrack.core import (  # noqa: E402
    UltrackConfig,
    _as_config,
    _with_scale,
    resolve_ultrack_areas,
    validate_ultrack_config,
)


class UltrackConfigTests(unittest.TestCase):
    def test_defaults_mapping_and_scale_replacement_validate(self) -> None:
        default = UltrackConfig()
        mapped = _as_config({"foreground_quantile": 0.99, "window_size": None})
        scaled = _with_scale(default, None)

        validate_ultrack_config(default)
        self.assertEqual(mapped.foreground_quantile, 0.99)
        self.assertIsNone(mapped.window_size)
        self.assertEqual(scaled.scale_zyx, (1.625, 0.40625, 0.40625))
        self.assertEqual(resolve_ultrack_areas(replace(default, min_area=10, max_area=20)), (10, 20))

    def test_every_public_field_rejects_a_malformed_value(self) -> None:
        invalid_by_field: dict[str, object] = {
            "scale_zyx": (1.0, 1.0),
            "sigmas_um": (),
            "dog_ratio": 1.0,
            "q_low": -1.0,
            "q_high": 2.0,
            "foreground_quantile": 2.0,
            "foreground_threshold": 2.0,
            "local_contrast_sigma_um": 0.0,
            "seed_h": 0.0,
            "seed_dilation_um": -1.0,
            "contour_preset": "unsupported",
            "min_frontier": 2.0,
            "min_cell_radius_um": 0.0,
            "max_cell_radius_um": 0.0,
            "min_area": 0,
            "max_area": 0,
            "max_link_um": 0.0,
            "max_neighbors": 0,
            "solver_name": "NOT_A_SOLVER",
            "window_size": 0,
            "overlap_size": -1,
            "solution_gap": -1.0,
            "time_limit": 0,
            "n_workers": 0,
            "n_threads": 0,
            "working_dir": "",
            "database_file_name": "",
            "max_segments_per_time": 0,
        }
        field_names = {item.name for item in fields(UltrackConfig)}
        self.assertEqual(set(invalid_by_field), field_names)

        config = UltrackConfig()
        for name, invalid in invalid_by_field.items():
            with self.subTest(name=name, invalid=invalid):
                with self.assertRaisesRegex(ValueError, re.escape(name)):
                    replace(config, **{name: invalid})

    def test_cross_field_invariants_fail_at_construction(self) -> None:
        cases = [
            ({"q_low": 0.8, "q_high": 0.8}, "q_low"),
            (
                {"min_cell_radius_um": 5.0, "max_cell_radius_um": 4.0},
                "min_cell_radius_um",
            ),
            ({"window_size": 12, "overlap_size": 12}, "overlap_size"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    replace(UltrackConfig(), **overrides)

    def test_nonfinite_values_and_boolean_integers_are_rejected(self) -> None:
        for name, invalid in (
            ("solution_gap", math.nan),
            ("scale_zyx", (1.0, math.inf, 1.0)),
            ("max_neighbors", True),
            ("n_threads", False),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    replace(UltrackConfig(), **{name: invalid})

    def test_nonincreasing_explicit_areas_keep_resolver_normalization(self) -> None:
        config = UltrackConfig(min_area=20, max_area=20)

        self.assertEqual(resolve_ultrack_areas(config), (20, 21))

    def test_runtime_entry_rechecks_low_level_mutation(self) -> None:
        config = UltrackConfig()
        object.__setattr__(config, "max_segments_per_time", 0)

        with self.assertRaisesRegex(ValueError, "max_segments_per_time"):
            resolve_ultrack_areas(config)


if __name__ == "__main__":
    unittest.main()
