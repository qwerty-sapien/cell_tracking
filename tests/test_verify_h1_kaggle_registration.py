from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_h1_kaggle_registration as verify  # noqa: E402


class H1KaggleRegistrationTests(unittest.TestCase):
    def test_exact_remote_sources_pass(self) -> None:
        datasets = sorted(verify.EXPECTED_DATASET_SOURCES)
        report = verify.registration_report(
            {"dataset_sources": datasets},
            {
                "dataset_sources": datasets,
                "competition_sources": sorted(
                    verify.EXPECTED_COMPETITION_SOURCES
                ),
            },
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["timing_smoke_authorized"])

    def test_missing_fit105_source_stops(self) -> None:
        datasets = set(verify.EXPECTED_DATASET_SOURCES)
        datasets.remove(
            "onebitbrain/trackastra-h1-fit105-cache-v1"
        )
        report = verify.registration_report(
            {
                "dataset_sources": sorted(
                    verify.EXPECTED_DATASET_SOURCES
                )
            },
            {
                "dataset_sources": sorted(datasets),
                "competition_sources": sorted(
                    verify.EXPECTED_COMPETITION_SOURCES
                ),
            },
        )
        self.assertEqual(
            report["status"], "STOP_SOURCE_REGISTRATION_DRIFT"
        )
        self.assertEqual(
            report["missing_from_remote"],
            ["onebitbrain/trackastra-h1-fit105-cache-v1"],
        )
        self.assertFalse(report["timing_smoke_authorized"])

    def test_unexpected_or_competition_drift_stops(self) -> None:
        datasets = sorted(
            {
                *verify.EXPECTED_DATASET_SOURCES,
                "onebitbrain/unexpected",
            }
        )
        report = verify.registration_report(
            {
                "dataset_sources": sorted(
                    verify.EXPECTED_DATASET_SOURCES
                )
            },
            {
                "dataset_sources": datasets,
                "competition_sources": [],
            },
        )
        self.assertEqual(
            report["status"], "STOP_SOURCE_REGISTRATION_DRIFT"
        )


if __name__ == "__main__":
    unittest.main()
