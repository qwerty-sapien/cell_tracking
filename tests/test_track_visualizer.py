from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import build_track_visualizer as visualizer  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)
from celltrack_alpha.official_eval import official_metric_row  # noqa: E402


METRIC_KEYS = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
)


def score_sample(
    identity: str,
    *,
    edge_fp: int,
    edge_fn: int,
    edge_tp: int = 10,
    division_tp: int = 0,
    division_fp: int = 0,
    division_fn: int = 0,
    adjusted: float = 0.8,
    division_jaccard: float | None = None,
) -> dict[str, object]:
    return {
        "identity": identity,
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "adj_edge_jaccard": adjusted,
        "edge_jaccard": adjusted,
        "division_jaccard": division_jaccard,
        "node_recall": 0.9,
        "scale_zyx": [1.0, 1.0, 1.0],
    }


def score_payload(samples: list[dict[str, object]]) -> dict[str, object]:
    return {"frozen_score": {"b": {"samples": samples}}}


def sparse_gt_fixture() -> dict[str, object]:
    return {
        "nodes": {
            "node_id": np.asarray([10, 11, 12, 13], dtype=np.int64),
            "t": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "z": np.asarray([0, 0, 0, 0], dtype=np.float64),
            "y": np.asarray([10, 11, 12, 13], dtype=np.float64),
            "x": np.asarray([10, 11, 12, 13], dtype=np.float64),
        },
        "edges": np.asarray([[10, 11], [11, 12], [12, 13]], dtype=np.int64),
        "estimated_number_of_nodes": 100.0,
    }


def sparse_prediction_fixture() -> SubmissionGraph:
    # Nodes 1/2/3 localize GT 10/11/12. Nodes 4/5 are deliberately far from
    # every annotation, so their edge is unknown rather than a false positive.
    nodes = {
        1: SubmissionNode(1, 0, 0, 10, 10),
        2: SubmissionNode(2, 1, 0, 11, 11),
        3: SubmissionNode(3, 2, 0, 12, 12),
        4: SubmissionNode(4, 0, 0, 100, 100),
        5: SubmissionNode(5, 1, 0, 101, 101),
    }
    return SubmissionGraph(
        "synthetic_sparse",
        nodes,
        [SubmissionEdge(1, 2), SubmissionEdge(4, 5)],
    )


def official_row(graph: SubmissionGraph, gt: dict[str, object]) -> dict[str, float]:
    nodes, edges = visualizer.graph_rows(graph)
    return official_metric_row(nodes, edges, gt, (1.0, 1.0, 1.0))


def empty_overlay() -> visualizer.OverlayData:
    return visualizer.OverlayData(
        pred_coords={},
        gt_coords={},
        track_ids={},
        matched_pred=set(),
        all_pred_edges=[],
        tp_edges=set(),
        fp_edges=set(),
        fn_edges=set(),
        fragmentation_edges=set(),
        missed_divisions=set(),
        false_divisions=set(),
        division_tp_forks=set(),
    )


class ResidualRankingTests(unittest.TestCase):
    def test_rows_are_ranked_by_residual_then_identity_and_scored(self) -> None:
        rows = visualizer.rank_video_rows(
            score_payload(
                [
                    score_sample("test:b", edge_fp=3, edge_fn=2, adjusted=0.8),
                    score_sample("test:c", edge_fp=1, edge_fn=0, adjusted=0.9),
                    score_sample(
                        "test:a",
                        edge_fp=2,
                        edge_fn=3,
                        division_tp=1,
                        adjusted=0.7,
                        division_jaccard=0.5,
                    ),
                ]
            )
        )

        self.assertEqual([row["identity"] for row in rows], ["test:a", "test:b", "test:c"])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
        self.assertEqual([row["residual"] for row in rows], [5, 5, 1])
        self.assertAlmostEqual(float(rows[0]["score"]), 0.75)
        self.assertAlmostEqual(float(rows[1]["score"]), 0.8)
        self.assertAlmostEqual(sum(float(row["error_contribution"]) for row in rows), 1.0)

    def test_exact_curated_selection_deduplicates_and_keeps_all_tags(self) -> None:
        specs = [
            ("a", .95, 0, 0, None),
            ("b", .90, 2, 0, .90),
            ("c", .85, 2, 1, .80),
            ("d", .80, 2, 2, .70),
            ("e", .70, 0, 10, 0.0),
            ("f", .60, 0, 9, 0.0),
            ("g", .50, 0, 8, 0.0),
            ("h", .40, 1, 6, .20),
            ("i", .30, 0, 0, None),
            ("j", .20, 0, 0, None),
            ("k", .10, 0, 0, None),
            ("l", .05, 0, 0, None),
        ]
        samples = [
            score_sample(
                f"test:{name}",
                edge_fp=index + 1,
                edge_fn=0,
                adjusted=adjusted,
                division_tp=division_tp,
                division_fn=division_fn,
                division_jaccard=division_jaccard,
            )
            for index, (name, adjusted, division_tp, division_fn, division_jaccard) in enumerate(specs)
        ]
        rows = visualizer.normalize_score_rows(score_payload(samples))
        categories = visualizer.selection_categories(rows)
        self.assertEqual(categories["edge_top3"], ["test:a", "test:b", "test:c"])
        self.assertEqual(categories["edge_bottom3"], ["test:l", "test:k", "test:j"])
        self.assertEqual(categories["division_top3"], ["test:b", "test:c", "test:d"])
        self.assertEqual(categories["division_missed_opportunity"], ["test:e", "test:f", "test:g"])
        selected = visualizer.select_review_rows(rows, "default")
        self.assertEqual(
            [row["identity"] for row in selected],
            ["test:a", "test:b", "test:c", "test:l", "test:k", "test:j", "test:d", "test:e", "test:f", "test:g"],
        )
        b = next(row for row in selected if row["identity"] == "test:b")
        self.assertEqual(b["categories"], ["edge_top3", "division_top3"])
        all_rows = visualizer.select_review_rows(rows, "all")
        self.assertEqual([row["identity"] for row in all_rows[:3]], ["test:l", "test:k", "test:j"])

        parser_args = visualizer.build_parser().parse_args(["build"])
        self.assertEqual(parser_args.scope, "default")
        self.assertEqual(parser_args.clip_seconds, 15.0)
        self.assertIsNone(parser_args.fps)
        self.assertTrue(parser_args.hud)
        self.assertEqual(parser_args.hud_corner, "tr")
        self.assertEqual(visualizer.parse_metrics(parser_args.metrics), visualizer.DEFAULT_METRICS)

    def test_ranking_and_selection_reject_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(visualizer.VisualizerError, "no per-video samples"):
            visualizer.rank_video_rows(score_payload([]))
        zero_rows = [score_sample("test:zero", edge_fp=0, edge_fn=0)]
        with self.assertRaisesRegex(visualizer.VisualizerError, "sum must be positive"):
            visualizer.rank_video_rows(score_payload(zero_rows))

        duplicate = score_payload([
            score_sample("test:only", edge_fp=1, edge_fn=0),
            score_sample("test:only", edge_fp=2, edge_fn=0),
        ])
        with self.assertRaisesRegex(visualizer.VisualizerError, "duplicate score identity"):
            visualizer.normalize_score_rows(duplicate)
        with self.assertRaisesRegex(visualizer.VisualizerError, "unsupported scope"):
            visualizer.select_review_rows([], "legacy")


class Prompt3FlagTests(unittest.TestCase):
    def test_unavailable_checked_clear_and_flagged_are_distinct(self) -> None:
        identities = ["test:a", "test:b", "test:c"]
        unavailable = visualizer.load_frame_flags(None, identities)
        self.assertEqual({value["status"] for value in unavailable.values()}, {"unavailable"})
        self.assertTrue(all(value["events"] == [] for value in unavailable.values()))

        payload = {
            "schema": "prompt3.synthetic.v1",
            "checked_identities": ["test:b", "test:c"],
            "events": [
                {"identity": "test:c", "event_type": "global_jump", "target_frame": 8},
                {"identity": "test:c", "event_type": "frozen", "frame_index": 4},
            ],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "flags.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            flags = visualizer.load_frame_flags(path, identities)

        self.assertEqual(flags["test:a"]["status"], "unavailable")
        self.assertEqual(flags["test:b"]["status"], "checked-clear")
        self.assertEqual(flags["test:c"]["status"], "flagged")
        self.assertEqual(
            [(event["event_type"], event["source_frame"], event["target_frame"]) for event in flags["test:c"]["events"]],
            [("frozen", 3, 4), ("global_jump", 7, 8)],
        )
        self.assertEqual(flags["test:c"]["events"][0]["detector_schema"], payload["schema"])

    def test_invalid_prompt3_events_fail_closed(self) -> None:
        invalid_events = (
            ({"identity": "test:a", "event_type": "camera_pan", "target_frame": 2}, "unsupported"),
            ({"identity": "test:a", "event_type": "frozen", "source_frame": 2, "target_frame": 2}, "boundary"),
            ({"identity": "test:a", "event_type": "global_jump"}, "requires target_frame"),
        )
        with tempfile.TemporaryDirectory() as raw_tmp:
            for index, (event, message) in enumerate(invalid_events):
                with self.subTest(event=event):
                    path = Path(raw_tmp) / f"invalid-{index}.json"
                    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
                    with self.assertRaisesRegex(visualizer.VisualizerError, message):
                        visualizer.load_frame_flags(path, ["test:a"])

            missing = Path(raw_tmp) / "does-not-exist.json"
            with self.assertRaisesRegex(visualizer.VisualizerError, "does not exist"):
                visualizer.load_frame_flags(missing, ["test:a"])

    def test_prompt3_event_must_fit_the_rendered_frame_range(self) -> None:
        in_range = {
            "status": "flagged",
            "events": [
                {
                    "event_type": "global_jump",
                    "source_frame": 6,
                    "target_frame": 7,
                }
            ],
        }
        visualizer.validate_flag_bounds(in_range, frames=8, identity="test:a")

        out_of_range = {
            "status": "flagged",
            "events": [
                {
                    "event_type": "frozen",
                    "source_frame": 7,
                    "target_frame": 8,
                }
            ],
        }
        with self.assertRaisesRegex(
            visualizer.VisualizerError,
            r"outside test:a frame range 0\.\.7",
        ):
            visualizer.validate_flag_bounds(out_of_range, frames=8, identity="test:a")

    def test_data_pathology_census_maps_source_transition_to_target_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "videos").mkdir()
            detail = {
                "schema": "celltrack-alpha.g04-data-pathology-video.v1",
                "transitions": [
                    {"frame": 3, "frozen": True, "pool_breaking": True, "shift_magnitude_um": 12.0},
                    {"frame": 4, "frozen": False, "pool_breaking": False},
                ],
            }
            (root / "videos/test__a.json").write_text(json.dumps(detail), encoding="utf-8")
            census = {
                "schema": "celltrack-alpha.g04-data-pathology-census.v1",
                "videos": [{"split": "test", "dataset": "a", "path": "videos/test__a.json"}],
            }
            path = root / "CENSUS.json"
            path.write_text(json.dumps(census), encoding="utf-8")
            flags = visualizer.load_frame_flags(path, ["test:a", "test:unchecked"])
        self.assertEqual(flags["test:a"]["status"], "flagged")
        self.assertEqual(
            [(event["event_type"], event["source_frame"], event["target_frame"]) for event in flags["test:a"]["events"]],
            [("frozen", 3, 4), ("global_jump", 3, 4)],
        )
        self.assertEqual(flags["test:unchecked"]["status"], "unavailable")


class PlaybackAndHudTests(unittest.TestCase):
    def test_default_budget_is_strict_and_all_scope_is_uncapped(self) -> None:
        visualizer.enforce_bundle_budget("default", visualizer.DEFAULT_BUDGET_BYTES - 1)
        with self.assertRaisesRegex(visualizer.VisualizerError, "exceeding"):
            visualizer.enforce_bundle_budget("default", visualizer.DEFAULT_BUDGET_BYTES)
        visualizer.enforce_bundle_budget("all", visualizer.DEFAULT_BUDGET_BYTES * 10)

    def test_clip_seconds_and_fps_override_are_symmetric(self) -> None:
        self.assertAlmostEqual(visualizer.effective_fps(100, clip_seconds=15, fps=None), 100 / 15)
        self.assertEqual(visualizer.effective_fps(100, clip_seconds=100, fps=None), 1.0)
        self.assertEqual(visualizer.effective_fps(100, clip_seconds=180, fps=12.5), 12.5)
        with self.assertRaisesRegex(visualizer.VisualizerError, "slowest target"):
            visualizer.effective_fps(100, clip_seconds=101, fps=None)
        with self.assertRaisesRegex(visualizer.VisualizerError, "positive"):
            visualizer.effective_fps(100, clip_seconds=15, fps=0)

    def test_metric_registry_and_corner_map_fail_closed(self) -> None:
        metrics = visualizer.parse_metrics("frame_index,edge_fp_fn_cumulative,video_id")
        self.assertEqual(metrics, ("frame_index", "edge_fp_fn_cumulative", "video_id"))
        self.assertEqual(
            visualizer.parse_metric_corners("frame_index=tl,video_id:bl", metrics),
            {"frame_index": "tl", "video_id": "bl"},
        )
        with self.assertRaisesRegex(visualizer.VisualizerError, "unknown HUD metrics"):
            visualizer.parse_metrics("made_up")
        with self.assertRaisesRegex(visualizer.VisualizerError, "not selected"):
            visualizer.parse_metric_corners("jump_flag=br", metrics)

    def test_hud_settings_are_render_visible_and_hash_bound(self) -> None:
        row = visualizer.normalize_score_rows(
            score_payload([score_sample("test:hud", edge_fp=1, edge_fn=2, division_fn=3, division_jaccard=0.0)])
        )[0]
        overlay = empty_overlay()
        mip = np.zeros((256, 256), dtype=np.uint16)
        hud = {
            "enabled": True,
            "corner": "tr",
            "metrics": ["frame_index", "edge_fp_fn_cumulative"],
            "metric_corners": {"edge_fp_fn_cumulative": "bl"},
        }
        frame = visualizer.render_frame(
            mip,
            frame_index=2,
            total_frames=5,
            overlay=overlay,
            state=visualizer.build_render_state(overlay),
            score_row=row,
            identity="test:hud",
            intensity_limits=(0.0, 1.0),
            estimated_nodes=100,
            gt_node_count=2,
            flag_info={"status": "checked-clear", "events": []},
            hud_config=hud,
            cumulative_errors={"fp": 1, "fn": 2},
        )
        pixels = np.asarray(frame)
        self.assertTrue(np.any(pixels[:80, 120:250] != 0))
        self.assertTrue(np.any(pixels[180:250, :160] != 0))
        with tempfile.TemporaryDirectory() as raw_tmp:
            fragment = Path(raw_tmp) / "fragment.csv"
            fragment.write_text("x", encoding="utf-8")
            receipt = {"observed_sha256": "raw", "bytes": 1, "mtime_ns": 2}
            first = visualizer.render_request_contract(
                row=row, fragment=fragment, raw_receipt=receipt,
                flag_info={"status": "checked-clear", "events": []}, fps=100 / 15,
                renderer_sha256="renderer", clip_seconds=15, hud_config=hud,
            )
            second = visualizer.render_request_contract(
                row=row, fragment=fragment, raw_receipt=receipt,
                flag_info={"status": "checked-clear", "events": []}, fps=100 / 15,
                renderer_sha256="renderer", clip_seconds=15,
                hud_config={**hud, "corner": "tl"},
            )
        self.assertNotEqual(first, second)


class SparseGroundTruthOverlayTests(unittest.TestCase):
    def test_exact_edge_classes_keep_unknown_edges_out_of_fp(self) -> None:
        gt = sparse_gt_fixture()
        graph = sparse_prediction_fixture()
        row = official_row(graph, gt)
        overlay = visualizer.analyse_graph(graph, gt, (1.0, 1.0, 1.0), row)

        self.assertEqual({key: int(row[key]) for key in METRIC_KEYS}, {
            "edge_tp": 1,
            "edge_fp": 0,
            "edge_fn": 2,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
        })
        self.assertEqual(overlay.counts(), {
            "edge_tp": 1,
            "edge_fp": 0,
            "edge_fn": 2,
            "fragmentation_breaks": 1,
            "plain_fn_edges": 1,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
        })
        self.assertEqual(len(overlay.all_pred_edges), 2)
        self.assertEqual(len(overlay.tp_edges), 1)
        self.assertEqual(overlay.fp_edges, set())
        self.assertEqual(len(overlay.fragmentation_edges), 1)
        self.assertEqual(len(overlay.fn_edges), 1)
        self.assertEqual(len(overlay.matched_pred), 3)
        self.assertEqual(len(overlay.pred_coords) - len(overlay.matched_pred), 2)

        fragment_source, fragment_target = next(iter(overlay.fragmentation_edges))
        self.assertEqual(overlay.gt_coords[fragment_source][0], 1)
        self.assertEqual(overlay.gt_coords[fragment_target][0], 2)
        plain_source, plain_target = next(iter(overlay.fn_edges))
        self.assertEqual(overlay.gt_coords[plain_source][0], 2)
        self.assertEqual(overlay.gt_coords[plain_target][0], 3)

        by_frame = visualizer.frame_error_counts(overlay, frames=4)
        self.assertEqual(by_frame[2]["fragmentation"], 1)
        self.assertEqual(by_frame[3]["fn_edge"], 1)

    def test_score_localization_mismatch_fails_closed(self) -> None:
        gt = sparse_gt_fixture()
        graph = sparse_prediction_fixture()
        row = official_row(graph, gt)
        row["edge_fp"] = 99.0
        with self.assertRaisesRegex(visualizer.VisualizerError, "localization mismatch"):
            visualizer.analyse_graph(graph, gt, (1.0, 1.0, 1.0), row)

    def test_valid_zero_edge_prediction_localizes_fragmentation(self) -> None:
        gt = {
            "nodes": {
                "node_id": np.asarray([10, 11], dtype=np.int64),
                "t": np.asarray([0, 1], dtype=np.int64),
                "z": np.asarray([0, 0], dtype=np.float64),
                "y": np.asarray([10, 11], dtype=np.float64),
                "x": np.asarray([10, 11], dtype=np.float64),
            },
            "edges": np.asarray([[10, 11]], dtype=np.int64),
            "estimated_number_of_nodes": 2.0,
        }
        graph = SubmissionGraph(
            "zero_edges",
            {
                1: SubmissionNode(1, 0, 0, 10, 10),
                2: SubmissionNode(2, 1, 0, 11, 11),
            },
            [],
        )
        # The official edge counts for a prediction with no edges are exact and
        # do not require the full metric wrapper's separate node-recall path.
        row = {
            "edge_tp": 0.0,
            "edge_fp": 0.0,
            "edge_fn": 1.0,
            "division_tp": 0.0,
            "division_fp": 0.0,
            "division_fn": 0.0,
        }
        self.assertEqual(
            {key: int(row[key]) for key in METRIC_KEYS},
            {
                "edge_tp": 0,
                "edge_fp": 0,
                "edge_fn": 1,
                "division_tp": 0,
                "division_fp": 0,
                "division_fn": 0,
            },
        )

        overlay = visualizer.analyse_graph(graph, gt, (1.0, 1.0, 1.0), row)

        self.assertEqual(overlay.all_pred_edges, [])
        self.assertEqual(overlay.tp_edges, set())
        self.assertEqual(overlay.fp_edges, set())
        self.assertEqual(overlay.fn_edges, set())
        self.assertEqual(len(overlay.fragmentation_edges), 1)
        self.assertEqual(len(overlay.matched_pred), 2)
        self.assertEqual(overlay.counts()["edge_fn"], 1)


class ProvenanceAndReuseTests(unittest.TestCase):
    def test_prefixed_aggregate_is_materialized_as_renderable_selected_fragment(self) -> None:
        graph = SubmissionGraph(
            "train:aggregate",
            {1: SubmissionNode(1, 0, 0, 1, 1)},
            [],
        )
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            aggregate = root / "aggregate.csv"
            visualizer.write_submission({"train:aggregate": graph}, aggregate)
            row = {"identity": "train:aggregate"}
            resolved = visualizer.resolve_prediction_fragments([row], aggregate, root / "bundle")
            materialized = visualizer.read_submission(resolved["train:aggregate"])
        self.assertEqual(set(materialized), {"aggregate"})
        self.assertEqual(materialized["aggregate"].dataset, "aggregate")

    def test_sidecar_reuse_requires_current_contract_input_and_video_hash(self) -> None:
        row = {
            **score_sample("test:cached", edge_fp=1, edge_fn=0),
            "rank": 1,
            "score": 0.8,
            "residual": 1,
            "error_contribution": 1.0,
        }
        raw_receipt = {
            "observed_sha256": "raw-sha",
            "bytes": 123,
            "mtime_ns": 456,
        }
        flags = {"status": "unavailable", "events": [], "source": None}
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            fragment = root / "test__cached.csv"
            fragment.write_bytes(b"fragment-v1")
            video = root / "videos" / "test__cached.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video-v1")
            contract = visualizer.render_request_contract(
                row=row,
                fragment=fragment,
                raw_receipt=raw_receipt,
                flag_info=flags,
                fps=7,
                renderer_sha256="renderer-v1",
            )
            sidecar = {
                "identity": "test:cached",
                "frames": 5,
                "video": "videos/test__cached.mp4",
                "video_bytes": video.stat().st_size,
                "video_sha256": visualizer.sha256_file(video),
                "render_contract": contract,
            }

            self.assertTrue(visualizer.sidecar_is_current(sidecar, contract, root, flags))

            changed_contract = {**contract, "fps": 8}
            self.assertFalse(
                visualizer.sidecar_is_current(sidecar, changed_contract, root, flags)
            )

            fragment.write_bytes(b"fragment-v2")
            changed_input_contract = visualizer.render_request_contract(
                row=row,
                fragment=fragment,
                raw_receipt=raw_receipt,
                flag_info=flags,
                fps=7,
                renderer_sha256="renderer-v1",
            )
            self.assertNotEqual(
                changed_input_contract["prediction_fragment_sha256"],
                contract["prediction_fragment_sha256"],
            )
            self.assertFalse(
                visualizer.sidecar_is_current(sidecar, changed_input_contract, root, flags)
            )

            video.write_bytes(b"video-v2")  # Same byte count, different content.
            self.assertFalse(visualizer.sidecar_is_current(sidecar, contract, root, flags))

    def test_validation_rejects_same_size_video_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            bundle = Path(raw_tmp)
            (bundle / "metadata").mkdir()
            (bundle / "videos").mkdir()
            raw_zip = bundle / "raw.zip"
            raw_zip.write_bytes(b"raw-source")
            raw_sha = "synthetic-raw-sha"
            receipt = {
                "schema": "celltrack-alpha.raw-zip-integrity.v1",
                "path": str(raw_zip),
                "bytes": raw_zip.stat().st_size,
                "mtime_ns": raw_zip.stat().st_mtime_ns,
                "observed_sha256": raw_sha,
            }
            receipt_path = bundle / "raw_zip_integrity.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            fragment = bundle / "test__one.csv"
            fragment.write_bytes(b"prediction-fragment")
            video = bundle / "videos" / "test__one.mp4"
            video.write_bytes(b"video-v1")
            original_video_sha = visualizer.sha256_file(video)
            sidecar = {
                "schema": "celltrack-alpha.track-visualizer-video.v1",
                "identity": "test:one",
                "frames": 2,
                "fps": 2.0,
                "encoded_duration_seconds": 1.0,
                "video": "videos/test__one.mp4",
                "video_bytes": video.stat().st_size,
                "video_sha256": original_video_sha,
                "prediction_fragment": str(fragment),
                "prediction_fragment_sha256": visualizer.sha256_file(fragment),
                "render_contract": {
                    "version": visualizer.RENDER_CONTRACT_VERSION,
                    "raw_zip_sha256": raw_sha,
                },
                "prompt3": {"status": "unavailable", "events": []},
                "overlay_counts": {
                    "edge_tp": 1,
                    "edge_fp": 0,
                    "edge_fn": 0,
                    "division_tp": 0,
                    "division_fp": 0,
                    "division_fn": 0,
                },
            }
            metadata_path = bundle / "metadata" / "test__one.json"
            metadata_path.write_text(json.dumps(sidecar), encoding="utf-8")
            manifest = {
                "scope": "all",
                "indexed_video_count": 1,
                "input_sha256": "0" * 64,
                "size": {"measured_bundle_bytes": 0},
                "sources": {
                    "raw_zip": {
                        "observed_sha256": raw_sha,
                        "integrity_receipt": receipt_path.name,
                        "integrity_receipt_sha256": visualizer.sha256_file(receipt_path),
                    }
                },
                "videos": [
                    {
                        "identity": "test:one",
                        "rank": 1,
                        "selection_order": 1,
                        "residual": 1,
                        "rendered": True,
                        "metadata": "metadata/test__one.json",
                        "video": "videos/test__one.mp4",
                        "prompt3_status": "unavailable",
                        "edge_tp": 1,
                        "edge_fp": 0,
                        "edge_fn": 0,
                        "division_tp": 0,
                        "division_fp": 0,
                        "division_fn": 0,
                    }
                ],
            }
            (bundle / "index.html").write_text("<html>test:one</html>", encoding="utf-8")
            for _ in range(3):
                (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                manifest["size"]["measured_bundle_bytes"] = visualizer.bundle_bytes(bundle)
            (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            video.write_bytes(b"video-v2")  # Same byte count, different SHA-256.
            self.assertEqual(video.stat().st_size, sidecar["video_bytes"])
            with (
                mock.patch.object(visualizer, "REGISTERED_RAW_SHA256", raw_sha),
                mock.patch.object(
                    visualizer,
                    "REGISTERED_RAW_BYTES",
                    raw_zip.stat().st_size,
                ),
                mock.patch.object(visualizer, "EXPECTED_VIDEO_COUNT", 1),
                mock.patch.object(visualizer, "EXPECTED_TOP10_SHARE", 1.0),
                mock.patch.object(visualizer, "ffprobe_video") as probe,
                self.assertRaisesRegex(visualizer.VisualizerError, "video SHA-256 drifted"),
            ):
                visualizer.validate_bundle(bundle, min_rendered=1, ffprobe="unused")
            probe.assert_not_called()


class RenderingAndHtmlTests(unittest.TestCase):
    def test_native_256_xy_mip_renders_with_persistent_side_panel(self) -> None:
        volume = np.zeros((3, 256, 256), dtype=np.uint16)
        volume[1, 50, 20] = 100
        mip = np.max(volume, axis=0)
        self.assertEqual(mip.shape, (256, 256))

        overlay = empty_overlay()
        row = {
            "rank": 1,
            "score": 0.75,
            "residual": 4,
            "edge_tp": 3,
            "edge_fp": 1,
            "edge_fn": 3,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 1,
            "adj_edge_jaccard": 0.5,
            "edge_jaccard": 0.5,
            "division_jaccard": 0.0,
            "gt_division_count": 1,
            "node_recall": 0.9,
            "family": "native",
        }
        frame = visualizer.render_frame(
            mip,
            frame_index=1,
            total_frames=3,
            overlay=overlay,
            state=visualizer.build_render_state(overlay),
            score_row=row,
            identity="test:native",
            intensity_limits=(0.0, 100.0),
            estimated_nodes=100.0,
            gt_node_count=3,
            flag_info={
                "status": "flagged",
                "events": [{"event_type": "global_jump", "target_frame": 1}],
            },
            hud_config={"enabled": False, "corner": "tr", "metrics": [], "metric_corners": {}},
        )

        self.assertEqual(frame.mode, "RGB")
        self.assertEqual(frame.size, (256 + visualizer.PANEL_WIDTH, 256))
        pixels = np.asarray(frame)
        self.assertTupleEqual(tuple(pixels[50, 20]), (255, 255, 255))
        self.assertTupleEqual(tuple(pixels[100, 0]), visualizer.COLORS["global_jump"])
        self.assertTupleEqual(tuple(pixels[255, -1]), (14, 17, 24))

    def test_html_contains_sparse_gt_metrics_scrubbing_and_timelines(self) -> None:
        row = {
            **score_sample(
                "test:worst",
                edge_tp=12,
                edge_fp=3,
                edge_fn=4,
                division_tp=1,
                division_fp=2,
                division_fn=3,
            ),
            "rank": 1,
            "dataset": "worst",
            "family": "worst",
            "categories": ["edge_bottom3"],
            "residual": 7,
            "score": 0.5,
            "error_contribution": 1.0,
        }
        sidecar = {
            "video": "videos/test__worst.mp4",
            "frames": 5,
            "fps": 7,
            "prompt3": {
                "status": "flagged",
                "events": [
                    {"event_type": "frozen", "target_frame": 2},
                    {"event_type": "global_jump", "target_frame": 3},
                ],
            },
            "frame_error_counts": [
                {"fp_edge": 0, "fn_edge": 0},
                {"fp_edge": 1, "fragmentation": 1},
                {},
                {},
                {},
            ],
            "gt_annotated_nodes": 3,
            "estimated_total_nodes": 100,
            "annotation_fraction": 0.03,
            "tracklet_count": 4,
        }
        html = visualizer.build_html(
            [row],
            {"test:worst": sidecar},
            {"generated_at": "2026-08-03T00:00:00+00:00"},
        )

        required = (
            "UNANNOTATED / UNKNOWN",
            "FP edge",
            "FN edge",
            "fragmentation break",
            "missed division",
            "false division",
            "Edge TP / FP / FN",
            "Division TP / FP / FN",
            "Frozen-frame and global-jump boundaries",
            "Per-frame error activity",
            "Prompt 3: flagged",
            "<video controls",
            'data-frame="2"',
            'data-frame="3"',
            "frozen before frame 2",
            "global jump before frame 3",
            "Math.round(video.currentTime*fps)",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, html)


if __name__ == "__main__":
    unittest.main()
