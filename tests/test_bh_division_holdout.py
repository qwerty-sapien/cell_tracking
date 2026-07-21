from __future__ import annotations

from pathlib import Path
import sys
import unittest

import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.division_metrics import score_divisions  # noqa: E402


ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
FRAGMENT = (
    ROOT
    / "reports/g02/source_softmax_edge_f0_fit117_holdout_parts_v1"
    / "train__44b6_267148e4/frags/train__44b6_267148e4.csv"
)


class FrozenHoldoutDivisionTests(unittest.TestCase):
    @unittest.skipUnless(
        ARCHIVE.exists() and FRAGMENT.exists(),
        "requires the frozen local holdout archive and scored prediction fragment",
    )
    @unittest.expectedFailure
    def test_frozen_holdout_divider_3_is_recovered_by_the_emitted_fork(self) -> None:
        """Regression for the known division in frozen holdout 44b6_267148e4."""

        dataset = "44b6_267148e4"
        submission = scoreall._fragment_graph(FRAGMENT, f"train:{dataset}")
        prediction_nodes, prediction_edges = scoreall._prediction_rows(dataset, submission)
        pred_graph = prediction_to_graph(prediction_nodes, prediction_edges)

        with ZipStore(ARCHIVE, mode="r") as store:
            gt_graph = geff_arrays_to_graph(read_geff_arrays(store, "train", dataset))
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))

        self.assertIn(3, gt_graph.dividing_nodes())
        self.assertEqual(score_divisions(pred_graph, gt_graph, scale, 7.0)[3], 1)
