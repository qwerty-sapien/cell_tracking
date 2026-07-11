from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_baseline.rule_based import DEFAULT_SCALE_ZYX, parse_scale  # noqa: E402


def ngff_attrs(axes: list[Any], scale: list[Any]) -> dict[str, Any]:
    normalized_axes: list[Any] = []
    for axis in axes:
        if not isinstance(axis, str):
            normalized_axes.append(axis)
        elif axis.strip().casefold() in {"z", "y", "x"}:
            normalized_axes.append(
                {"name": axis, "type": "space", "unit": "micrometer"}
            )
        elif axis.strip().casefold() == "t":
            normalized_axes.append({"name": axis, "type": "time", "unit": "second"})
        else:
            normalized_axes.append(axis)
    return {
        "multiscales": [
            {
                "axes": normalized_axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": scale},
                        ],
                    }
                ],
            }
        ]
    }


class ParseScaleTests(unittest.TestCase):
    def test_valid_tzyx_metadata(self) -> None:
        attrs = ngff_attrs(
            [
                {"name": "t", "type": "time"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            [2.0, 1.5, 0.4, 0.3],
        )

        self.assertEqual(parse_scale(attrs), (1.5, 0.4, 0.3))

    def test_noncanonical_axis_order_is_rejected_by_raw_array_api(self) -> None:
        attrs = ngff_attrs(
            [
                {"name": " Y ", "type": "SPACE", "unit": "micrometer"},
                {"name": "T", "type": "time"},
                {"name": "x", "type": "space", "unit": "micrometer"},
                {"name": "Z", "type": "space", "unit": "micrometer"},
            ],
            [0.5, 2.0, 0.25, 3.0],
        )

        with self.assertRaisesRegex(ValueError, "canonical T/Z/Y/X order"):
            parse_scale(attrs)

    def test_spatial_units_are_normalized_to_micrometers(self) -> None:
        cases = {
            "micrometer": ((1.5, 0.4, 0.3), (1.5, 0.4, 0.3)),
            "nanometer": ((1500.0, 400.0, 300.0), (1.5, 0.4, 0.3)),
            "millimeter": ((0.0015, 0.0004, 0.0003), (1.5, 0.4, 0.3)),
        }
        for unit, (scale, expected) in cases.items():
            axes = [
                {"name": "t", "type": "time", "unit": "second"},
                *[
                    {"name": name, "type": "space", "unit": unit}
                    for name in ("z", "y", "x")
                ],
            ]
            with self.subTest(unit=unit):
                self.assertEqual(parse_scale(ngff_attrs(axes, [1.0, *scale])), expected)

    def test_missing_mixed_and_unsupported_spatial_units_raise(self) -> None:
        base_axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        cases = []
        missing = [dict(axis) for axis in base_axes]
        missing[1].pop("unit")
        cases.append((missing, "must declare physical units"))
        mixed = [dict(axis) for axis in base_axes]
        mixed[1]["unit"] = "nanometer"
        cases.append((mixed, "one consistent physical unit"))
        unsupported = [dict(axis) for axis in base_axes]
        for axis in unsupported[1:]:
            axis["unit"] = "pixel"
        cases.append((unsupported, "unsupported spatial unit"))
        for axes, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_scale(ngff_attrs(axes, [1.0, 1.5, 0.4, 0.3]))

    def test_dataset_and_multiscale_scale_transforms_are_composed(self) -> None:
        attrs = ngff_attrs(["t", "z", "y", "x"], [1.0, 2.0, 3.0, 4.0])
        attrs["multiscales"][0]["coordinateTransformations"] = [
            {"type": "scale", "scale": [1.0, 2.0, 2.0, 2.0]}
        ]

        self.assertEqual(parse_scale(attrs), (4.0, 6.0, 8.0))

    def test_scale_transform_is_selected_by_type(self) -> None:
        attrs = ngff_attrs(["t", "z", "y", "x"], [1.0, 1.5, 0.4, 0.3])
        attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"].insert(
            0,
            {"type": "translation", "translation": [0.0, 10.0, 20.0, 30.0]},
        )

        self.assertEqual(parse_scale(attrs), (1.5, 0.4, 0.3))

    def test_missing_physical_metadata_uses_historical_default(self) -> None:
        for attrs in (None, {}, {"image_statistics": {"quantiles": {"0.5": 10}}}):
            with self.subTest(attrs=attrs):
                self.assertEqual(parse_scale(attrs), DEFAULT_SCALE_ZYX)

    def test_present_but_malformed_metadata_raises(self) -> None:
        malformed_cases = {
            "multiscales is not an array": {"multiscales": "invalid"},
            "axes absent": {"multiscales": [{"datasets": []}]},
            "datasets absent": {
                "multiscales": [{"axes": ["t", "z", "y", "x"]}]
            },
            "scale length mismatch": ngff_attrs(
                ["t", "z", "y", "x"], [1.0, 1.5, 0.4]
            ),
            "no scale transform": {
                "multiscales": [
                    {
                        "axes": ["t", "z", "y", "x"],
                        "datasets": [
                            {
                                "coordinateTransformations": [
                                    {
                                        "type": "translation",
                                        "translation": [0, 0, 0, 0],
                                    }
                                ]
                            }
                        ],
                    }
                ]
            },
            "ambiguous multiscales": {
                "multiscales": [
                    ngff_attrs(["t", "z", "y", "x"], [1, 1, 1, 1])[
                        "multiscales"
                    ][0],
                    ngff_attrs(["t", "z", "y", "x"], [1, 1, 1, 1])[
                        "multiscales"
                    ][0],
                ]
            },
        }

        for label, attrs in malformed_cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, "Invalid OME-NGFF physical metadata"
                ):
                    parse_scale(attrs)

    def test_nonfinite_spatial_scales_raise(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            attrs = ngff_attrs(["t", "z", "y", "x"], [1.0, value, 0.4, 0.4])
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    parse_scale(attrs)

    def test_nonpositive_spatial_scales_raise(self) -> None:
        for value in (0.0, -1.0):
            attrs = ngff_attrs(["t", "z", "y", "x"], [1.0, 1.5, value, 0.4])
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    parse_scale(attrs)

    def test_duplicate_spatial_axis_raises(self) -> None:
        attrs = ngff_attrs(
            ["t", "z", "Z", "y", "x"],
            [1.0, 1.5, 1.5, 0.4, 0.4],
        )

        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_scale(attrs)

    def test_missing_spatial_axis_raises(self) -> None:
        attrs = ngff_attrs(["t", "z", "y"], [1.0, 1.5, 0.4])

        with self.assertRaisesRegex(ValueError, r"missing: \['x'\]"):
            parse_scale(attrs)

    def test_actual_archive_shaped_metadata(self) -> None:
        archive_metadata = {
            "attributes": {
                "multiscales": [
                    {
                        "version": "0.5",
                        "axes": [
                            {"name": "T", "type": "time", "unit": "second"},
                            {
                                "name": "Z",
                                "type": "space",
                                "unit": "micrometer",
                            },
                            {
                                "name": "Y",
                                "type": "space",
                                "unit": "micrometer",
                            },
                            {
                                "name": "X",
                                "type": "space",
                                "unit": "micrometer",
                            },
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [
                                            1.0,
                                            1.625,
                                            0.40625,
                                            0.40625,
                                        ],
                                    }
                                ],
                            }
                        ],
                        "name": "0",
                    }
                ],
                "image_statistics": {"quantiles": {"0.01": 38.0, "0.999": 2145.0}},
            },
            "zarr_format": 3,
            "consolidated_metadata": None,
            "node_type": "group",
        }

        self.assertEqual(
            parse_scale(archive_metadata["attributes"]),
            (1.625, 0.40625, 0.40625),
        )


if __name__ == "__main__":
    unittest.main()
