from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from analyze_candgen_ceiling import (  # noqa: E402
    CandgenCeilingError,
    assert_safe_fit_only_path,
    background_budget,
    legacy_score_after_perfect_swaps,
)


class CandgenCeilingTests(unittest.TestCase):
    def test_legacy_controls_reproduce_published_values(self) -> None:
        self.assertAlmostEqual(
            legacy_score_after_perfect_swaps(290),
            0.8956779478999035,
            places=12,
        )
        self.assertAlmostEqual(
            legacy_score_after_perfect_swaps(332),
            0.8977458371966905,
            places=12,
        )

    def test_background_budget_is_the_largest_passing_integer(self) -> None:
        result = background_budget(
            edge_tp=100,
            edge_fn=5,
            node_adjustment=0.98,
            raw_background=10000,
            goal=0.92,
        )
        maximum = int(result["maximum_background_false_positive_edges"])
        adjusted = 0.98 * 100 / (100 + 5 + maximum)
        next_adjusted = 0.98 * 100 / (100 + 5 + maximum + 1)
        self.assertGreaterEqual(adjusted, 0.92)
        self.assertLess(next_adjusted, 0.92)

    def test_forbidden_paths_fail_before_read(self) -> None:
        for path in (
            Path("reports/g02/cv57/report.json"),
            Path("reports/g02/holdout/scores.json"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(CandgenCeilingError, "forbidden"):
                    assert_safe_fit_only_path(path)


if __name__ == "__main__":
    unittest.main()
