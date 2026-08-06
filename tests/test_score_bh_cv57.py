from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pkgbh  # noqa: E402
import score_bh_cv57  # noqa: E402


class CV57ScoreTests(unittest.TestCase):
    def test_kaggle_calibration_ledger_has_three_provenanced_pairs(self) -> None:
        report = json.loads(
            (ROOT / "reports/g02/scorer_parity_v1.json").read_text(encoding="utf-8")
        )
        ledger = report["calibration_pair_ledger"]
        self.assertEqual(ledger["status"], "observed_kaggle_pairs")
        self.assertEqual(ledger["pair_count"], 3)
        self.assertEqual(
            {pair["submission_ref"] for pair in ledger["pairs"]},
            {54910497, 54921344, 54923074},
        )
        self.assertEqual(
            {
                pair["configuration"]["motion_error_weight"]: pair["site_score"]
                for pair in ledger["pairs"]
            },
            {0.0: 0.860, 0.25: 0.877, 2.0: 0.878},
        )
        for pair in ledger["pairs"]:
            self.assertFalse(pair["configuration"]["division_enabled"])
            self.assertEqual(pair["configuration"]["motion_direction"], "backward")
            self.assertEqual(pair["submission_status"], "COMPLETE")
            self.assertIsNone(pair["error_description"])
            self.assertTrue(pair["kernel_output"]["bhdiag_ok"])
            self.assertTrue(pair["kernel_output"]["input_contract_ok"])
            self.assertFalse(any(pair["kernel_output"]["fail_closed"].values()))
            self.assertAlmostEqual(
                pair["site_score"] - pair["local_score"],
                pair["residual_site_minus_local"],
                places=15,
            )
        calibration = ledger["calibration"]
        self.assertEqual(calibration["fit_status"], "provisional_coarse_forecast_only")
        self.assertAlmostEqual(
            calibration["affine_ols"]["slope"],
            0.7193024334975513,
            places=15,
        )
        self.assertTrue(calibration["rank_transfer"]["near_tie_sign_reversal"])
        self.assertEqual(
            calibration["operational_policy"]["heuristic_local_delta_resolution_floor"],
            0.004,
        )
        self.assertEqual(calibration["operational_policy"]["p0_verdict"], "FAIL/HALT")
        self.assertEqual(calibration["operational_policy"]["p1_status"], "locked")

    def test_exact_selection_is_fold0_minus_only_terminal_exclusion(self) -> None:
        split_path = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        registry = json.loads(
            (ROOT / "reports/g02/dataset_registry_v1.json").read_text(encoding="utf-8")
        )
        archive_ids = tuple(
            row["video_id"] for row in registry["records"] if row["split"] == "train"
        )
        selected = score_bh_cv57.exact_selection(
            split,
            split_path=split_path,
            archive_video_ids=archive_ids,
        )
        self.assertEqual(len(selected), 57)
        self.assertEqual(selected, pkgbh.cv57_video_ids(split_path))
        self.assertNotIn(pkgbh.CV57_EXCLUDED_IDENTITY, selected)

    def test_retired_baseline_is_explicitly_not_a_tta_comparator(self) -> None:
        baseline = score_bh_cv57.RETIRED_BASELINE
        self.assertEqual(baseline["d4_views"], [0])
        self.assertTrue(baseline["effective_division_enabled"])
        self.assertNotEqual(
            baseline["edge_artifact_sha256"],
            pkgbh.CV57_EDGE_ARTIFACT_SHA256,
        )

    def test_postpatch_scorer_is_attested_by_green_parity_ledger(self) -> None:
        evidence = score_bh_cv57.validate_revalidated_scorer(
            score_bh_cv57.DEFAULT_SCORER_PARITY
        )
        self.assertEqual(evidence["fixture_count"], 9)
        self.assertEqual(
            evidence["official_checkout"],
            "075fc5f5a52d11077f9dc2b074644618f26939e2",
        )
        self.assertEqual(len(evidence["source_hashes"]), 3)


if __name__ == "__main__":
    unittest.main()
