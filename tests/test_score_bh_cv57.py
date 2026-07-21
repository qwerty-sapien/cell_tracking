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


if __name__ == "__main__":
    unittest.main()
