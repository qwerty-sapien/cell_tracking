#!/usr/bin/env python3
"""Build a headless, sparse-GT-aware post-scoring visual-review bundle.

``build --scope default`` renders the deterministic four-category review hook.
``build --scope all`` is the explicit, uncapped full-catalog review mode.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "kaggle-cell-tracking-competition" / "src"))

from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionGraph,
    read_submission,
    write_submission,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.division_metrics import score_divisions  # noqa: E402
from tracking_cellmot.metrics import _evaluate, _evaluate_matched_graph  # noqa: E402
import tracksdata as td  # noqa: E402


DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_PREDICTIONS = ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments"
DEFAULT_SCORES = ROOT / "reports/g02/screening/obs_diff_selected_graph_appearance_direct_cv57_v1.json"
DEFAULT_RESIDUAL = ROOT / "reports/g02/selected_graph_appearance_direct_residual_v1.json"
DEFAULT_LATEST_BUNDLE = ROOT / "reports/g04/post_scoring_visual_review/latest"
DEFAULT_ALL_BUNDLE = ROOT / "reports/g04/post_scoring_visual_review/all"
DEFAULT_FRAME_FLAGS = ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json"
RAW_INTEGRITY_BASIS = ROOT / "reports/g02/trackastra_6bba_raw_directory_manifest_v1.json"
REGISTERED_RAW_SHA256 = "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
REGISTERED_RAW_BYTES = 87_393_127_165
EXPECTED_VIDEO_COUNT = 57
EXPECTED_TOP10_SHARE = 0.5232584524619923
DEFAULT_CLIP_SECONDS = 15.0
DEFAULT_METRICS = (
    "edge_adjusted_jaccard",
    "division_jaccard",
    "gt_division_count",
)
DEFAULT_BUDGET_BYTES = 1_000_000_000
TAIL_FRAMES = 6
PANEL_WIDTH = 320
RENDER_CONTRACT_VERSION = 3

COLORS = {
    "gt": (0, 225, 255),
    "annotated": (120, 245, 255),
    "unknown": (165, 170, 185),
    "fp_edge": (255, 70, 75),
    "fn_edge": (255, 165, 35),
    "fragmentation": (255, 40, 210),
    "missed_division": (255, 230, 45),
    "false_division": (170, 105, 255),
    "frozen": (70, 165, 255),
    "global_jump": (255, 190, 40),
}

HUD_CORNERS = ("tl", "tr", "bl", "br")
HUD_REGISTRY: dict[str, dict[str, Any]] = {
    "edge_adjusted_jaccard": {"label": "Edge adj J", "color": (75, 235, 255)},
    "edge_raw_jaccard": {"label": "Edge raw J", "color": (90, 160, 255)},
    "edge_tp_fp_fn": {"label": "Edge TP/FP/FN", "color": (255, 175, 70)},
    "division_jaccard": {"label": "Division J", "color": (245, 105, 255)},
    "division_tp_fp_fn": {"label": "Div TP/FP/FN", "color": (185, 125, 255)},
    "gt_division_count": {"label": "GT divisions", "color": (255, 235, 80)},
    "node_recall": {"label": "Node recall", "color": (95, 245, 145)},
    "combined_score": {"label": "Combined", "color": (255, 245, 190)},
    "frame_index": {"label": "Frame", "color": (245, 248, 255)},
    "frozen_flag": {"label": "Frozen", "color": COLORS["frozen"]},
    "jump_flag": {"label": "Jump", "color": COLORS["global_jump"]},
    "video_id": {"label": "Video", "color": (90, 225, 255)},
    "family": {"label": "Family", "color": (125, 245, 170)},
    "edge_fp_fn_cumulative": {"label": "Edge FP/FN cum", "color": (255, 125, 105)},
}

SELECTION_LABELS = {
    "edge_top3": "Top-3 adjusted edge Jaccard",
    "edge_bottom3": "Bottom-3 adjusted edge Jaccard",
    "division_top3": "Top-3 division Jaccard (GT divisions > 0)",
    "division_missed_opportunity": "Missed division opportunity",
}


class VisualizerError(RuntimeError):
    """Raised for an actionable visualizer input or output failure."""


@dataclass
class OverlayData:
    pred_coords: dict[int, tuple[int, float, float, float]]
    gt_coords: dict[int, tuple[int, float, float, float]]
    track_ids: dict[int, int]
    matched_pred: set[int]
    all_pred_edges: list[tuple[int, int]]
    tp_edges: set[tuple[int, int]]
    fp_edges: set[tuple[int, int]]
    fn_edges: set[tuple[int, int]]
    fragmentation_edges: set[tuple[int, int]]
    missed_divisions: set[int]
    false_divisions: set[int]
    division_tp_forks: set[int]

    def counts(self) -> dict[str, int]:
        return {
            "edge_tp": len(self.tp_edges),
            "edge_fp": len(self.fp_edges),
            "edge_fn": len(self.fn_edges) + len(self.fragmentation_edges),
            "fragmentation_breaks": len(self.fragmentation_edges),
            "plain_fn_edges": len(self.fn_edges),
            "division_tp": len(self.division_tp_forks),
            "division_fp": len(self.false_divisions),
            "division_fn": len(self.missed_divisions),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def canonical_identity(value: str) -> str:
    value = str(value).strip()
    if ":" in value:
        split, dataset = value.split(":", 1)
        if split and dataset:
            return f"{split}:{dataset}"
    if "__" in value:
        split, dataset = value.split("__", 1)
        if split and dataset:
            return f"{split}:{dataset}"
    raise VisualizerError(f"invalid video identity {value!r}; expected split:dataset")


def identity_slug(identity: str) -> str:
    return canonical_identity(identity).replace(":", "__", 1)


def per_video_score(row: Mapping[str, Any]) -> float:
    adjusted = float(row["adj_edge_jaccard"])
    denom = sum(int(float(row[key])) for key in ("division_tp", "division_fp", "division_fn"))
    if denom == 0:
        return adjusted
    division = row.get("division_jaccard")
    division_value = 0.0 if division is None else float(division)
    return adjusted + 0.1 * division_value


def _finite_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def normalize_score_rows(score_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize current scoreall and historical observation-difference rows."""

    samples: Any = score_payload.get("samples")
    if not isinstance(samples, list):
        samples = score_payload.get("rows")
    if not isinstance(samples, list):
        frozen = score_payload.get("frozen_score", {})
        if isinstance(frozen, Mapping):
            branch = frozen.get("b", {})
            samples = branch.get("samples") if isinstance(branch, Mapping) else None
    if not isinstance(samples, list) or not samples:
        raise VisualizerError("score artifact has no per-video samples")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in samples:
        if not isinstance(raw, Mapping):
            raise VisualizerError("score sample is not an object")
        identity_value = raw.get("identity")
        if identity_value is None and raw.get("dataset") is not None:
            identity_value = f"train:{raw['dataset']}"
        identity = canonical_identity(str(identity_value))
        if identity in seen:
            raise VisualizerError(f"duplicate score identity {identity}")
        seen.add(identity)
        split, dataset = identity.split(":", 1)
        try:
            counts = {
                key: int(float(raw.get(key, 0)))
                for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
            }
        except (TypeError, ValueError) as exc:
            raise VisualizerError(f"{identity} has invalid scorer counts") from exc
        if any(value < 0 for value in counts.values()):
            raise VisualizerError(f"{identity} has negative scorer counts")
        adjusted = _finite_float(raw.get("adj_edge_jaccard"))
        if adjusted is None:
            raise VisualizerError(f"{identity} lacks finite adj_edge_jaccard")
        raw_edge = _finite_float(raw.get("edge_jaccard"))
        if raw_edge is None:
            denom = counts["edge_tp"] + counts["edge_fp"] + counts["edge_fn"]
            raw_edge = counts["edge_tp"] / denom if denom else 1.0
        division = _finite_float(raw.get("division_jaccard"))
        gt_divisions = counts["division_tp"] + counts["division_fn"]
        normalized = {
            **dict(raw),
            **counts,
            "identity": identity,
            "split": split,
            "dataset": dataset,
            "family": str(raw.get("family") or dataset.split("_", 1)[0]),
            "adj_edge_jaccard": adjusted,
            "edge_jaccard": raw_edge,
            "division_jaccard": division,
            "gt_division_count": gt_divisions,
            "node_recall": _finite_float(raw.get("node_recall")),
            "residual": counts["edge_fp"] + counts["edge_fn"],
        }
        normalized["score"] = per_video_score(normalized)
        rows.append(normalized)

    ranked = sorted(rows, key=lambda row: (-int(row["residual"]), str(row["identity"])))
    total = sum(int(row["residual"]) for row in ranked)
    cumulative = 0
    for rank, row in enumerate(ranked, start=1):
        cumulative += int(row["residual"])
        row["rank"] = rank
        row["error_contribution"] = int(row["residual"]) / total if total else 0.0
        row["cumulative_error_contribution"] = cumulative / total if total else 0.0
    return ranked


def rank_video_rows(score_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = normalize_score_rows(score_payload)
    total = sum(int(row["residual"]) for row in rows)
    if total <= 0:
        raise VisualizerError("per-video residual sum must be positive")
    return rows


def selection_categories(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return exact, deterministic membership for the four review categories."""

    finite_edge = [row for row in rows if _finite_float(row.get("adj_edge_jaccard")) is not None]
    division_eligible = [
        row
        for row in rows
        if int(row.get("gt_division_count", 0)) > 0
        and _finite_float(row.get("division_jaccard")) is not None
    ]
    missed = [
        row
        for row in division_eligible
        if int(row.get("division_tp", 0)) == 0
        or abs(float(row["division_jaccard"])) <= 1e-12
    ]
    return {
        "edge_top3": [
            str(row["identity"])
            for row in sorted(
                finite_edge,
                key=lambda row: (-float(row["adj_edge_jaccard"]), str(row["identity"])),
            )[:3]
        ],
        "edge_bottom3": [
            str(row["identity"])
            for row in sorted(
                finite_edge,
                key=lambda row: (float(row["adj_edge_jaccard"]), str(row["identity"])),
            )[:3]
        ],
        "division_top3": [
            str(row["identity"])
            for row in sorted(
                division_eligible,
                key=lambda row: (
                    -float(row["division_jaccard"]),
                    -int(row["gt_division_count"]),
                    str(row["identity"]),
                ),
            )[:3]
        ],
        "division_missed_opportunity": [
            str(row["identity"])
            for row in sorted(
                missed,
                key=lambda row: (
                    -int(row["gt_division_count"]),
                    float(row["division_jaccard"]),
                    str(row["identity"]),
                ),
            )[:3]
        ],
    }


def select_review_rows(rows: Sequence[Mapping[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope not in {"default", "all"}:
        raise VisualizerError(f"unsupported scope {scope!r}")
    categories = selection_categories(rows)
    tags_by_identity: dict[str, list[str]] = defaultdict(list)
    for category, identities in categories.items():
        for identity in identities:
            tags_by_identity[identity].append(category)
    by_identity = {str(row["identity"]): dict(row) for row in rows}
    if scope == "default":
        ordered: list[str] = []
        for category in SELECTION_LABELS:
            for identity in categories[category]:
                if identity not in ordered:
                    ordered.append(identity)
    else:
        ordered = [
            str(row["identity"])
            for row in sorted(
                rows,
                key=lambda row: (
                    float(row["adj_edge_jaccard"]),
                    -int(row["residual"]),
                    str(row["identity"]),
                ),
            )
        ]
    selected: list[dict[str, Any]] = []
    for order, identity in enumerate(ordered, start=1):
        row = by_identity[identity]
        row["selection_order"] = order
        row["categories"] = list(tags_by_identity.get(identity, []))
        selected.append(row)
    return selected


def verify_residual_contract(rows: Sequence[Mapping[str, Any]], residual_payload: Mapping[str, Any]) -> None:
    if len(rows) != EXPECTED_VIDEO_COUNT:
        raise VisualizerError(f"expected {EXPECTED_VIDEO_COUNT} CV57 videos, found {len(rows)}")
    expected = [canonical_identity(row["identity"]) for row in residual_payload["per_video"]["largest_residual_videos"]]
    observed = [str(row["identity"]) for row in rows[: len(expected)]]
    if observed != expected:
        raise VisualizerError(f"registered residual top-10 mismatch: observed={observed} expected={expected}")
    share = sum(int(row["residual"]) for row in rows[:10]) / sum(int(row["residual"]) for row in rows)
    registered = float(residual_payload["per_video"]["residual_concentration"]["10"]["share"])
    if not math.isclose(share, registered, rel_tol=0.0, abs_tol=1e-15):
        raise VisualizerError(f"top-10 residual share mismatch: {share} != {registered}")
    if not math.isclose(share, EXPECTED_TOP10_SHARE, rel_tol=0.0, abs_tol=1e-15):
        raise VisualizerError(f"unexpected registered top-10 share {share}")


def load_frame_flags(path: Path | None, identities: Iterable[str]) -> dict[str, dict[str, Any]]:
    canonical = [canonical_identity(identity) for identity in identities]
    if path is None:
        return {identity: {"status": "unavailable", "events": [], "source": None} for identity in canonical}
    if not path.exists():
        raise VisualizerError(f"Prompt 3 flag manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    events: list[Mapping[str, Any]] = []
    checked: set[str] = set()
    if payload.get("schema") == "celltrack-alpha.g04-data-pathology-census.v1":
        videos = payload.get("videos")
        if not isinstance(videos, list):
            raise VisualizerError("Prompt 3 census lacks videos")
        for summary in videos:
            if not isinstance(summary, Mapping):
                raise VisualizerError("Prompt 3 census video entry is not an object")
            identity = canonical_identity(f"{summary.get('split')}:{summary.get('dataset')}")
            checked.add(identity)
            detail_value = summary.get("path")
            if not isinstance(detail_value, str):
                raise VisualizerError(f"Prompt 3 census entry {identity} lacks a detail path")
            detail_path = path.parent / detail_value
            if not detail_path.exists():
                raise VisualizerError(f"Prompt 3 detail does not exist: {detail_path}")
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
            transitions = detail.get("transitions")
            if not isinstance(transitions, list):
                raise VisualizerError(f"Prompt 3 detail {identity} lacks transitions")
            for transition in transitions:
                source = int(transition["frame"])
                diagnostics = {
                    key: transition.get(key)
                    for key in (
                        "shift_magnitude_um",
                        "shift_um_zyx",
                        "pre_ncc",
                        "post_ncc",
                        "ncc_gain",
                    )
                    if key in transition
                }
                if bool(transition.get("frozen")):
                    events.append(
                        {
                            "identity": identity,
                            "event_type": "frozen",
                            "source_frame": source,
                            "target_frame": source + 1,
                            "diagnostics": diagnostics,
                            "detector_schema": detail.get("schema", payload.get("schema")),
                        }
                    )
                if bool(transition.get("pool_breaking")):
                    events.append(
                        {
                            "identity": identity,
                            "event_type": "global_jump",
                            "source_frame": source,
                            "target_frame": source + 1,
                            "diagnostics": diagnostics,
                            "detector_schema": detail.get("schema", payload.get("schema")),
                        }
                    )
    if isinstance(payload.get("events"), list):
        events.extend(payload["events"])
    if isinstance(payload.get("checked_identities"), list):
        checked |= {canonical_identity(value) for value in payload["checked_identities"]}
    if payload.get("schema") != "celltrack-alpha.g04-data-pathology-census.v1" and isinstance(payload.get("videos"), list):
        for video in payload["videos"]:
            identity = canonical_identity(video["identity"])
            if bool(video.get("checked", True)):
                checked.add(identity)
            for event in video.get("events", []):
                events.append({**event, "identity": identity})
    out = {
        identity: {
            "status": "checked-clear" if identity in checked else "unavailable",
            "events": [],
            "source": str(path),
        }
        for identity in canonical
    }
    for raw in events:
        identity = canonical_identity(raw["identity"])
        if identity not in out:
            continue
        event_type = str(raw.get("event_type", ""))
        if event_type not in {"frozen", "global_jump"}:
            raise VisualizerError(f"unsupported Prompt 3 event_type {event_type!r}")
        if "target_frame" in raw:
            target = int(raw["target_frame"])
            source = int(raw.get("source_frame", target - 1))
        elif "frame_index" in raw:
            target = int(raw["frame_index"])
            source = int(raw.get("source_frame", target - 1))
        else:
            raise VisualizerError("Prompt 3 event requires target_frame or frame_index")
        if source < 0 or target < 0 or target <= source:
            raise VisualizerError(f"invalid Prompt 3 frame boundary {source}->{target}")
        event = {
            "identity": identity,
            "event_type": event_type,
            "source_frame": source,
            "target_frame": target,
            "severity": raw.get("severity"),
            "diagnostics": raw.get("diagnostics", {}),
            "detector_schema": raw.get("detector_schema", payload.get("schema")),
        }
        out[identity]["events"].append(event)
        out[identity]["status"] = "flagged"
        checked.add(identity)
    for value in out.values():
        value["events"].sort(key=lambda event: (event["target_frame"], event["event_type"]))
    return out


def validate_flag_bounds(flag_info: Mapping[str, Any], frames: int, identity: str) -> None:
    if frames <= 0:
        raise VisualizerError(f"{identity} has no frames")
    for event in flag_info.get("events", []):
        source = int(event["source_frame"])
        target = int(event["target_frame"])
        if source >= frames or target >= frames:
            raise VisualizerError(
                f"Prompt 3 event {source}->{target} is outside {identity} frame range 0..{frames - 1}"
            )


def graph_rows(graph: SubmissionGraph) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    nodes = [
        {
            "dataset": graph.dataset,
            "node_id": node.node_id,
            "t": node.t,
            "z": node.z,
            "y": node.y,
            "x": node.x,
        }
        for node in graph.nodes.values()
    ]
    edges = [{"source_id": edge.source_id, "target_id": edge.target_id} for edge in graph.edges]
    return nodes, edges


def _coords(graph: td.graph.BaseGraph) -> dict[int, tuple[int, float, float, float]]:
    attrs = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    return {
        int(node_id): (int(t), float(z), float(y), float(x))
        for node_id, t, z, y, x in attrs.iter_rows()
    }


def analyse_graph(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale_zyx: tuple[float, float, float],
    score_row: Mapping[str, Any],
) -> OverlayData:
    node_rows, edge_rows = graph_rows(graph)
    prediction = prediction_to_graph(node_rows, edge_rows)
    truth = geff_arrays_to_graph(dict(gt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _evaluate(prediction, truth, "jaccard", scale_zyx, 7.0)
    if prediction.num_nodes() > 0 and td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID not in prediction.node_attr_keys():
        # The official edge scorer intentionally returns before matching when a
        # prediction has zero edges.  The visualizer still needs node matches to
        # localize its FN edges and distinguish fragmentation from missing ends.
        from tracksdata.metrics import DistanceMatching
        from tracksdata.options import get_options, set_options

        previous_progress = get_options().show_progress
        set_options(show_progress=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction.match(truth, matching=DistanceMatching(max_distance=7.0, scale=scale_zyx))
        finally:
            set_options(show_progress=previous_progress)

    source_key = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
    target_key = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
    matched_key = td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK
    all_pred_edges = [
        (int(row[source_key]), int(row[target_key]))
        for row in prediction.edge_attrs(attr_keys=[source_key, target_key]).iter_rows(named=True)
    ]
    gt_edges = {
        (int(row[source_key]), int(row[target_key]))
        for row in truth.edge_attrs(attr_keys=[source_key, target_key]).iter_rows(named=True)
    }
    tp_edges: set[tuple[int, int]] = set()
    fp_edges: set[tuple[int, int]] = set()
    if all_pred_edges and gt_edges:
        edge_df = _evaluate_matched_graph(prediction, truth)
        for row in edge_df.iter_rows(named=True):
            pair = (int(row[source_key]), int(row[target_key]))
            if bool(row[matched_key]):
                tp_edges.add(pair)
            elif bool(row["pred_valid"]):
                fp_edges.add(pair)
    matched_attrs = prediction.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    pred_to_gt = {
        int(pred): int(target)
        for pred, target in matched_attrs.iter_rows()
        if target is not None and int(target) != -1
    }
    gt_to_pred = {target: pred for pred, target in pred_to_gt.items()}
    matched_gt_edges = {(pred_to_gt[src], pred_to_gt[tgt]) for src, tgt in tp_edges}
    all_fn = gt_edges - matched_gt_edges
    fragmentation = {
        pair for pair in all_fn if pair[0] in gt_to_pred and pair[1] in gt_to_pred
    }
    fn_edges = all_fn - fragmentation

    prediction.assign_tracklet_ids("track_id")
    track_ids = {
        int(node_id): int(track_id)
        for node_id, track_id in prediction.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "track_id"]
        ).iter_rows()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        divisions = score_divisions(prediction, truth, scale_zyx, 7.0)
    missed = {int(gt_id) for gt_id, recovered in divisions.scores.items() if not recovered}
    overlay = OverlayData(
        pred_coords=_coords(prediction),
        gt_coords=_coords(truth),
        track_ids=track_ids,
        matched_pred=set(pred_to_gt),
        all_pred_edges=all_pred_edges,
        tp_edges=tp_edges,
        fp_edges=fp_edges,
        fn_edges=fn_edges,
        fragmentation_edges=fragmentation,
        missed_divisions=missed,
        false_divisions={int(node_id) for node_id in divisions.fp_forks},
        division_tp_forks={int(node_id) for node_id in divisions.tp_forks},
    )
    expected = {
        key: int(float(score_row[key]))
        for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    }
    observed = {key: overlay.counts()[key] for key in expected}
    if observed != expected:
        raise VisualizerError(
            f"official event localization mismatch for {graph.dataset}: observed={observed} expected={expected}"
        )
    return overlay


def normalize_mip(mip: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = (float(value) for value in np.percentile(mip, [1.0, 99.8]))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((mip.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(np.rint(scaled * 255.0), dtype=np.uint8)


def track_color(track_id: int, strength: float = 1.0) -> tuple[int, int, int]:
    hue = (int(track_id) * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.66, 1.0)
    base = np.asarray([red, green, blue]) * 255.0
    dim = 48.0 + strength * (base - 48.0)
    return tuple(int(np.clip(value, 0, 255)) for value in dim)


def draw_dashed(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int = 2,
    dash: float = 5.0,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 1e-9:
        draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), outline=fill, width=width)
        return
    unit_x, unit_y = (x1 - x0) / length, (y1 - y0) / length
    cursor = 0.0
    while cursor < length:
        stop = min(length, cursor + dash)
        draw.line(
            (x0 + unit_x * cursor, y0 + unit_y * cursor, x0 + unit_x * stop, y0 + unit_y * stop),
            fill=fill,
            width=width,
        )
        cursor += dash * 2.0


def draw_star(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    color: tuple[int, int, int],
    radius: float = 8.0,
) -> None:
    cx, cy = center
    points: list[tuple[float, float]] = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        length = radius if index % 2 == 0 else radius * 0.42
        points.append((cx + math.cos(angle) * length, cy + math.sin(angle) * length))
    draw.polygon(points, outline=(20, 20, 20), fill=color)


def _safe_font(size: int = 11) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _safe_bold_font(size: int = 12) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/local/share/fonts/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return _safe_font(size)


def parse_metrics(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    metrics = tuple(str(metric).strip() for metric in raw if str(metric).strip())
    if not metrics:
        raise VisualizerError("--metrics must select at least one registry entry")
    unknown = [metric for metric in metrics if metric not in HUD_REGISTRY]
    if unknown:
        raise VisualizerError(f"unknown HUD metrics: {unknown}; available={sorted(HUD_REGISTRY)}")
    if len(set(metrics)) != len(metrics):
        raise VisualizerError("--metrics contains duplicate entries")
    return metrics


def parse_metric_corners(value: str | None, metrics: Sequence[str]) -> dict[str, str]:
    if value is None or not value.strip():
        return {}
    out: dict[str, str] = {}
    for item in value.split(","):
        separator = "=" if "=" in item else ":" if ":" in item else None
        if separator is None:
            raise VisualizerError("--metric-corner entries must be METRIC=CORNER")
        metric, corner = (part.strip() for part in item.split(separator, 1))
        if metric not in HUD_REGISTRY:
            raise VisualizerError(f"unknown HUD metric in --metric-corner: {metric!r}")
        if metric not in metrics:
            raise VisualizerError(f"--metric-corner metric {metric!r} is not selected by --metrics")
        if corner not in HUD_CORNERS:
            raise VisualizerError(f"invalid HUD corner {corner!r}; choose from {HUD_CORNERS}")
        if metric in out:
            raise VisualizerError(f"duplicate --metric-corner entry for {metric}")
        out[metric] = corner
    return out


def effective_fps(frames: int, *, clip_seconds: float, fps: float | None) -> float:
    if frames <= 0:
        raise VisualizerError("video has no frames")
    if fps is not None:
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise VisualizerError("--fps must be a finite positive number")
        return float(fps)
    if not math.isfinite(float(clip_seconds)) or float(clip_seconds) <= 0:
        raise VisualizerError("--clip-seconds must be a finite positive number")
    if float(clip_seconds) > frames:
        raise VisualizerError(
            f"--clip-seconds={clip_seconds:g} exceeds {frames} frames; native ~1 fps is the slowest target"
        )
    return frames / float(clip_seconds)


def hud_metric_text(
    metric: str,
    *,
    row: Mapping[str, Any],
    identity: str,
    frame_index: int,
    total_frames: int,
    active_flags: Sequence[Mapping[str, Any]],
    cumulative_errors: Mapping[str, int],
) -> str:
    label = str(HUD_REGISTRY[metric]["label"])
    if metric == "edge_adjusted_jaccard":
        value = f"{float(row['adj_edge_jaccard']):.4f}"
    elif metric == "edge_raw_jaccard":
        value = f"{float(row['edge_jaccard']):.4f}"
    elif metric == "edge_tp_fp_fn":
        value = f"{int(row['edge_tp'])}/{int(row['edge_fp'])}/{int(row['edge_fn'])}"
    elif metric == "division_jaccard":
        division = _finite_float(row.get("division_jaccard"))
        value = "N/A" if division is None else f"{division:.4f}"
    elif metric == "division_tp_fp_fn":
        value = f"{int(row['division_tp'])}/{int(row['division_fp'])}/{int(row['division_fn'])}"
    elif metric == "gt_division_count":
        value = str(int(row["gt_division_count"]))
    elif metric == "node_recall":
        recall = _finite_float(row.get("node_recall"))
        value = "N/A" if recall is None else f"{recall:.4f}"
    elif metric == "combined_score":
        value = f"{float(row['score']):.4f}"
    elif metric == "frame_index":
        value = f"{frame_index:03d}/{total_frames - 1:03d}"
    elif metric == "frozen_flag":
        value = "YES" if any(event["event_type"] == "frozen" for event in active_flags) else "—"
    elif metric == "jump_flag":
        value = "YES" if any(event["event_type"] == "global_jump" for event in active_flags) else "—"
    elif metric == "video_id":
        value = identity
    elif metric == "family":
        value = str(row["family"])
    elif metric == "edge_fp_fn_cumulative":
        value = f"{int(cumulative_errors['fp'])}/{int(cumulative_errors['fn'])}"
    else:  # pragma: no cover - guarded by parse_metrics
        raise VisualizerError(f"unhandled HUD metric {metric}")
    return f"{label}: {value}"


def draw_hud(
    image: Image.Image,
    *,
    row: Mapping[str, Any],
    identity: str,
    frame_index: int,
    total_frames: int,
    active_flags: Sequence[Mapping[str, Any]],
    cumulative_errors: Mapping[str, int],
    hud_config: Mapping[str, Any],
    content_width: int,
) -> None:
    if not bool(hud_config.get("enabled", True)):
        return
    metrics = parse_metrics(hud_config.get("metrics", DEFAULT_METRICS))
    default_corner = str(hud_config.get("corner", "tr"))
    if default_corner not in HUD_CORNERS:
        raise VisualizerError(f"invalid HUD corner {default_corner!r}")
    overrides = dict(hud_config.get("metric_corners", {}))
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for metric in metrics:
        corner = str(overrides.get(metric, default_corner))
        if corner not in HUD_CORNERS:
            raise VisualizerError(f"invalid HUD corner {corner!r} for {metric}")
        grouped[corner].append(
            (
                metric,
                hud_metric_text(
                    metric,
                    row=row,
                    identity=identity,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    active_flags=active_flags,
                    cumulative_errors=cumulative_errors,
                ),
            )
        )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _safe_bold_font(11)
    padding = 7
    gap = 2
    margin = 7
    for corner in HUD_CORNERS:
        lines = grouped.get(corner, [])
        if not lines:
            continue
        boxes = [draw.textbbox((0, 0), text, font=font, stroke_width=2) for _, text in lines]
        line_height = max(box[3] - box[1] for box in boxes) + gap
        panel_width = min(content_width - 2 * margin, max(box[2] - box[0] for box in boxes) + 2 * padding)
        panel_height = len(lines) * line_height - gap + 2 * padding
        x0 = margin if corner.endswith("l") else content_width - margin - panel_width
        y0 = margin if corner.startswith("t") else image.height - margin - panel_height
        x1, y1 = x0 + panel_width, y0 + panel_height
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(4, 7, 12, 174), outline=(255, 255, 255, 72))
        y = y0 + padding
        for metric, text_value in lines:
            draw.text(
                (x0 + padding, y),
                text_value,
                font=font,
                fill=(*HUD_REGISTRY[metric]["color"], 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 235),
            )
            y += line_height
    image.paste(overlay, (0, 0), overlay)


def frame_error_counts(overlay: OverlayData, frames: int) -> list[dict[str, int]]:
    counts = [
        {"fp_edge": 0, "fn_edge": 0, "fragmentation": 0, "missed_division": 0, "false_division": 0}
        for _ in range(frames)
    ]
    for key, edges, coords in (
        ("fp_edge", overlay.fp_edges, overlay.pred_coords),
        ("fn_edge", overlay.fn_edges, overlay.gt_coords),
        ("fragmentation", overlay.fragmentation_edges, overlay.gt_coords),
    ):
        for _, target in edges:
            frame = int(coords[target][0])
            if 0 <= frame < frames:
                counts[frame][key] += 1
    for key, nodes, coords in (
        ("missed_division", overlay.missed_divisions, overlay.gt_coords),
        ("false_division", overlay.false_divisions, overlay.pred_coords),
    ):
        for node_id in nodes:
            frame = int(coords[node_id][0])
            if 0 <= frame < frames:
                counts[frame][key] += 1
    return counts


def build_render_state(overlay: OverlayData) -> dict[str, Any]:
    pred_nodes_by_t: dict[int, list[int]] = defaultdict(list)
    gt_nodes_by_t: dict[int, list[int]] = defaultdict(list)
    pred_edges_by_t: dict[int, list[tuple[int, int]]] = defaultdict(list)
    fp_by_t: dict[int, list[tuple[int, int]]] = defaultdict(list)
    fn_by_t: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frag_by_t: dict[int, list[tuple[int, int]]] = defaultdict(list)
    missed_by_t: dict[int, list[int]] = defaultdict(list)
    false_div_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, coords in overlay.pred_coords.items():
        pred_nodes_by_t[int(coords[0])].append(node_id)
    for node_id, coords in overlay.gt_coords.items():
        gt_nodes_by_t[int(coords[0])].append(node_id)
    for pair in overlay.all_pred_edges:
        pred_edges_by_t[int(overlay.pred_coords[pair[1]][0])].append(pair)
    for pair in overlay.fp_edges:
        fp_by_t[int(overlay.pred_coords[pair[1]][0])].append(pair)
    for pair in overlay.fn_edges:
        fn_by_t[int(overlay.gt_coords[pair[1]][0])].append(pair)
    for pair in overlay.fragmentation_edges:
        frag_by_t[int(overlay.gt_coords[pair[1]][0])].append(pair)
    for node_id in overlay.missed_divisions:
        missed_by_t[int(overlay.gt_coords[node_id][0])].append(node_id)
    for node_id in overlay.false_divisions:
        false_div_by_t[int(overlay.pred_coords[node_id][0])].append(node_id)
    return {
        "pred_nodes_by_t": pred_nodes_by_t,
        "gt_nodes_by_t": gt_nodes_by_t,
        "pred_edges_by_t": pred_edges_by_t,
        "fp_by_t": fp_by_t,
        "fn_by_t": fn_by_t,
        "frag_by_t": frag_by_t,
        "missed_by_t": missed_by_t,
        "false_div_by_t": false_div_by_t,
    }


def render_frame(
    mip: np.ndarray,
    *,
    frame_index: int,
    total_frames: int,
    overlay: OverlayData,
    state: Mapping[str, Any],
    score_row: Mapping[str, Any],
    identity: str,
    intensity_limits: tuple[float, float],
    estimated_nodes: float,
    gt_node_count: int,
    flag_info: Mapping[str, Any],
    hud_config: Mapping[str, Any] | None = None,
    cumulative_errors: Mapping[str, int] | None = None,
) -> Image.Image:
    base = normalize_mip(mip, *intensity_limits)
    height, width = base.shape
    image = Image.new("RGB", (width + PANEL_WIDTH, height), (14, 17, 24))
    image.paste(Image.fromarray(np.repeat(base[:, :, None], 3, axis=2), mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(image)

    for target_t in range(max(0, frame_index - TAIL_FRAMES + 1), frame_index + 1):
        age = frame_index - target_t
        strength = max(0.28, 1.0 - age / max(1, TAIL_FRAMES))
        for source, target in state["pred_edges_by_t"].get(target_t, []):
            _, _, sy, sx = overlay.pred_coords[source]
            _, _, ty, tx = overlay.pred_coords[target]
            color = track_color(overlay.track_ids.get(target, target), strength)
            draw.line((sx, sy, tx, ty), fill=color, width=2 if age == 0 else 1)

    for source, target in state["fp_by_t"].get(frame_index, []):
        _, _, sy, sx = overlay.pred_coords[source]
        _, _, ty, tx = overlay.pred_coords[target]
        draw.line((sx, sy, tx, ty), fill=COLORS["fp_edge"], width=4)
    for source, target in state["fn_by_t"].get(frame_index, []):
        _, _, sy, sx = overlay.gt_coords[source]
        _, _, ty, tx = overlay.gt_coords[target]
        draw_dashed(draw, (sx, sy), (tx, ty), fill=COLORS["fn_edge"], width=3)
    for source, target in state["frag_by_t"].get(frame_index, []):
        _, _, sy, sx = overlay.gt_coords[source]
        _, _, ty, tx = overlay.gt_coords[target]
        draw_dashed(draw, (sx, sy), (tx, ty), fill=COLORS["fragmentation"], width=3, dash=3)

    for node_id in state["pred_nodes_by_t"].get(frame_index, []):
        _, _, y, x = overlay.pred_coords[node_id]
        color = track_color(overlay.track_ids.get(node_id, node_id))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=(15, 15, 18))
        ring = COLORS["annotated"] if node_id in overlay.matched_pred else COLORS["unknown"]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=ring, width=1)
    for node_id in state["gt_nodes_by_t"].get(frame_index, []):
        _, _, y, x = overlay.gt_coords[node_id]
        color = COLORS["gt"]
        draw.line((x - 4, y, x + 4, y), fill=color, width=1)
        draw.line((x, y - 4, x, y + 4), fill=color, width=1)
    for node_id in state["missed_by_t"].get(frame_index, []):
        _, _, y, x = overlay.gt_coords[node_id]
        draw_star(draw, (x, y), COLORS["missed_division"])
    for node_id in state["false_div_by_t"].get(frame_index, []):
        _, _, y, x = overlay.pred_coords[node_id]
        color = COLORS["false_division"]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
        draw.line((x - 6, y - 6, x + 6, y + 6), fill=color, width=2)
        draw.line((x - 6, y + 6, x + 6, y - 6), fill=color, width=2)

    active_flags = [event for event in flag_info.get("events", []) if int(event["target_frame"]) == frame_index]
    if active_flags:
        draw.rectangle((0, 0, width - 1, height - 1), outline=COLORS[active_flags[0]["event_type"]], width=4)
        label = " | ".join(event["event_type"].replace("_", " ").upper() for event in active_flags)
        box = (4, 4, min(width - 4, 18 + 7 * len(label)), 21)
        draw.rectangle(box, fill=(15, 15, 20), outline=COLORS[active_flags[0]["event_type"]])
        draw.text((8, 7), label, fill=(255, 255, 255), font=_safe_font(10))

    draw_hud(
        image,
        row=score_row,
        identity=identity,
        frame_index=frame_index,
        total_frames=total_frames,
        active_flags=active_flags,
        cumulative_errors=cumulative_errors or {"fp": 0, "fn": 0},
        hud_config=hud_config
        or {"enabled": True, "corner": "tr", "metrics": DEFAULT_METRICS, "metric_corners": {}},
        content_width=width,
    )

    draw = ImageDraw.Draw(image)
    x0 = width
    draw.rectangle((x0, 0, x0 + PANEL_WIDTH, height), fill=(14, 17, 24))
    small = _safe_font(10)
    y = 8
    annotation_fraction = gt_node_count / estimated_nodes if estimated_nodes > 0 else float("nan")
    draw.rectangle((x0 + 8, y + 1, x0 + PANEL_WIDTH - 8, y + 49), fill=(36, 43, 53), outline=(100, 112, 130))
    draw.text((x0 + 14, y + 7), "SPARSE GT: UNKNOWN IS NOT FALSE POSITIVE", fill=(255, 236, 145), font=small)
    draw.text(
        (x0 + 14, y + 22),
        f"{gt_node_count:,} annotated / est {int(estimated_nodes):,} ({annotation_fraction:.1%})",
        fill=(220, 226, 235),
        font=small,
    )
    draw.text((x0 + 14, y + 35), "Gray rings: unannotated / not scored", fill=COLORS["unknown"], font=small)
    y += 58
    legends = [
        ("fp_edge", "FP edge (scorer-valid)"),
        ("fn_edge", "FN edge (endpoint missing)"),
        ("fragmentation", "Track break / fragmentation"),
        ("missed_division", "Missed division"),
        ("false_division", "False division"),
        ("gt", "GT annotation (+)"),
    ]
    for color_key, label in legends:
        draw.rectangle((x0 + 10, y + 2, x0 + 20, y + 12), fill=COLORS[color_key])
        draw.text((x0 + 27, y), label, fill=(220, 226, 235), font=small)
        y += 14
    y += 2
    flag_status = str(flag_info.get("status", "unavailable"))
    flag_color = (255, 215, 115) if flag_status == "unavailable" else (175, 235, 190)
    draw.text((x0 + 10, y), f"Prompt 3 flags: {flag_status}", fill=flag_color, font=small)
    draw.text((x0 + 10, y + 14), f"Tail: previous {TAIL_FRAMES} frames • XY full MIP", fill=(155, 165, 180), font=small)
    return image


def intensity_limits(attrs: Mapping[str, Any]) -> tuple[float, float]:
    quantiles = attrs.get("image_statistics", {}).get("quantiles", {})
    try:
        return float(quantiles["0.01"]), float(quantiles["0.999"])
    except (KeyError, TypeError, ValueError):
        return float("nan"), float("nan")


def encode_mp4(
    output: Path,
    frames: Iterable[Image.Image],
    *,
    width: int,
    height: int,
    fps: float,
    ffmpeg: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    gop = max(1, int(round(fps)))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.size != (width, height):
                raise VisualizerError(f"frame size changed from {(width, height)} to {frame.size}")
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise VisualizerError(f"ffmpeg failed with code {return_code}: {stderr[-2000:]}")
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise VisualizerError("ffmpeg produced no MP4 bytes")
        os.replace(tmp, output)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        tmp.unlink(missing_ok=True)


def fragment_path(prediction_dir: Path, identity: str) -> Path:
    split, dataset = canonical_identity(identity).split(":", 1)
    return prediction_dir / f"{split}__{dataset}.csv"


def verify_raw_integrity(zip_path: Path, bundle: Path) -> dict[str, Any]:
    basis = json.loads(RAW_INTEGRITY_BASIS.read_text(encoding="utf-8"))
    source = basis.get("source_archive", {})
    if int(source.get("bytes", -1)) != REGISTERED_RAW_BYTES or source.get("sha256") != REGISTERED_RAW_SHA256:
        raise VisualizerError(f"raw integrity basis disagrees with the registered archive: {RAW_INTEGRITY_BASIS}")
    if Path(str(source.get("path", ""))).resolve() != zip_path.resolve():
        raise VisualizerError(f"raw ZIP path is not the archive authenticated by {RAW_INTEGRITY_BASIS}")
    stat = zip_path.stat()
    if stat.st_size != REGISTERED_RAW_BYTES:
        raise VisualizerError(f"raw ZIP size drift: {stat.st_size} != registered {REGISTERED_RAW_BYTES} bytes")
    receipt_path = bundle / "raw_zip_integrity.json"
    basis_sha256 = sha256_file(RAW_INTEGRITY_BASIS)
    reusable_receipts = (
        receipt_path,
        ROOT / "reports/g04/track_visualizer_v1/raw_zip_integrity.json",
        DEFAULT_LATEST_BUNDLE / "raw_zip_integrity.json",
    )
    for candidate in reusable_receipts:
        if not candidate.exists():
            continue
        cached = json.loads(candidate.read_text(encoding="utf-8"))
        if (
            cached.get("schema") == "celltrack-alpha.raw-zip-integrity.v1"
            and cached.get("path") == str(zip_path.resolve())
            and int(cached.get("bytes", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
            and cached.get("observed_sha256") == REGISTERED_RAW_SHA256
            and cached.get("basis_sha256") == basis_sha256
        ):
            if candidate != receipt_path:
                atomic_write_text(receipt_path, json.dumps(cached, indent=2, sort_keys=True) + "\n")
            return cached
    observed = sha256_file(zip_path)
    if observed != REGISTERED_RAW_SHA256:
        raise VisualizerError(f"raw ZIP SHA-256 drift: {observed} != registered {REGISTERED_RAW_SHA256}")
    receipt = {
        "schema": "celltrack-alpha.raw-zip-integrity.v1",
        "path": str(zip_path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "observed_sha256": observed,
        "registered_sha256": REGISTERED_RAW_SHA256,
        "basis": str(RAW_INTEGRITY_BASIS),
        "basis_sha256": basis_sha256,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def render_request_contract(
    *,
    row: Mapping[str, Any],
    fragment: Path,
    raw_receipt: Mapping[str, Any],
    flag_info: Mapping[str, Any],
    fps: float,
    renderer_sha256: str,
    clip_seconds: float | None = None,
    hud_config: Mapping[str, Any] | None = None,
    scope: str = "default",
) -> dict[str, Any]:
    score_fields = {
        key: row.get(key)
        for key in (
            "identity",
            "rank",
            "score",
            "residual",
            "error_contribution",
            "categories",
            "family",
            "scale_zyx",
            "adj_edge_jaccard",
            "edge_jaccard",
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_jaccard",
            "division_tp",
            "division_fp",
            "division_fn",
            "gt_division_count",
            "node_recall",
        )
    }
    return {
        "version": RENDER_CONTRACT_VERSION,
        "renderer_sha256": renderer_sha256,
        "prediction_fragment_sha256": sha256_file(fragment),
        "score_signature_sha256": sha256_json(score_fields),
        "prompt3_signature_sha256": sha256_json(flag_info),
        "raw_zip_sha256": raw_receipt["observed_sha256"],
        "raw_zip_bytes": int(raw_receipt["bytes"]),
        "raw_zip_mtime_ns": int(raw_receipt["mtime_ns"]),
        "fps": float(fps),
        "clip_seconds": float(clip_seconds) if clip_seconds is not None else None,
        "scope": scope,
        "hud": dict(hud_config or {}),
        "tail_frames": TAIL_FRAMES,
        "panel_width": PANEL_WIDTH,
    }


def sidecar_is_current(
    sidecar: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    bundle: Path,
    flag_info: Mapping[str, Any],
) -> bool:
    if sidecar.get("render_contract") != expected_contract:
        return False
    validate_flag_bounds(flag_info, int(sidecar.get("frames", 0)), str(sidecar.get("identity", "video")))
    video_value = sidecar.get("video")
    if not isinstance(video_value, str):
        return False
    video_path = bundle / video_value
    if not video_path.exists() or video_path.stat().st_size != int(sidecar.get("video_bytes", -1)):
        return False
    return sha256_file(video_path) == sidecar.get("video_sha256")


def render_video_task(task: Mapping[str, Any]) -> dict[str, Any]:
    identity = canonical_identity(task["row"]["identity"])
    row = dict(task["row"])
    split, dataset = identity.split(":", 1)
    zip_path = Path(task["zip_path"])
    bundle = Path(task["bundle"])
    flag_info = dict(task["flag_info"])
    fragment = Path(task["fragment_path"])
    graph_map = read_submission(fragment)
    if set(graph_map) != {dataset}:
        raise VisualizerError(f"fragment dataset mismatch in {fragment}: {sorted(graph_map)}")
    graph = graph_map[dataset]
    video_path = bundle / "videos" / f"{identity_slug(identity)}.mp4"
    sidecar_path = bundle / "metadata" / f"{identity_slug(identity)}.json"
    with ZipStore(zip_path, mode="r") as store:
        group = zarr.open_group(store=store, path=f"{split}/{dataset}.zarr", mode="r")
        array = zarr.open_array(store=store, path=f"{split}/{dataset}.zarr/0", mode="r")
        gt = read_geff_arrays(store, split, dataset)
        scale = parse_scale(dict(group.attrs))
        if row.get("scale_zyx") is not None:
            expected_scale = tuple(float(value) for value in row["scale_zyx"])
            if not np.allclose(scale, expected_scale, rtol=0.0, atol=1e-12):
                raise VisualizerError(f"voxel scale mismatch for {identity}: {scale} != {expected_scale}")
        overlay = analyse_graph(graph, gt, scale, row)
        state = build_render_state(overlay)
        limits = intensity_limits(dict(group.attrs))
        frames = int(array.shape[0])
        image_height = int(array.shape[2])
        image_width = int(array.shape[3])
        validate_flag_bounds(flag_info, frames, identity)
        if image_width != 256 or image_height != 256:
            raise VisualizerError(f"expected native 256x256 XY field for {identity}, got {image_width}x{image_height}")
        estimated = float(gt["estimated_number_of_nodes"])
        gt_count = len(gt["nodes"]["node_id"])
        errors_by_frame = frame_error_counts(overlay, frames)
        cumulative_by_frame: list[dict[str, int]] = []
        cumulative_fp = 0
        cumulative_fn = 0
        for counts in errors_by_frame:
            cumulative_fp += int(counts["fp_edge"])
            cumulative_fn += int(counts["fn_edge"]) + int(counts["fragmentation"])
            cumulative_by_frame.append({"fp": cumulative_fp, "fn": cumulative_fn})

        def rendered_frames() -> Iterable[Image.Image]:
            for frame_index in range(frames):
                volume = np.asarray(array[frame_index])
                mip = np.max(volume, axis=0)
                yield render_frame(
                    mip,
                    frame_index=frame_index,
                    total_frames=frames,
                    overlay=overlay,
                    state=state,
                    score_row=row,
                    identity=identity,
                    intensity_limits=limits,
                    estimated_nodes=estimated,
                    gt_node_count=gt_count,
                    flag_info=flag_info,
                    hud_config=task["hud_config"],
                    cumulative_errors=cumulative_by_frame[frame_index],
                )

        encode_mp4(
            video_path,
            rendered_frames(),
            width=image_width + PANEL_WIDTH,
            height=image_height,
            fps=float(task["fps"]),
            ffmpeg=str(task["ffmpeg"]),
        )
    sidecar = {
        "schema": "celltrack-alpha.track-visualizer-video.v1",
        "identity": identity,
        "rank": int(row["rank"]),
        "dataset": dataset,
        "split": split,
        "frames": frames,
        "fps": float(task["fps"]),
        "target_clip_seconds": task.get("clip_seconds"),
        "encoded_duration_seconds": frames / float(task["fps"]),
        "native_xy": [image_height, image_width],
        "video_canvas": [image_height, image_width + PANEL_WIDTH],
        "video": str(video_path.relative_to(bundle)),
        "video_bytes": video_path.stat().st_size,
        "video_sha256": sha256_file(video_path),
        "prediction_fragment": str(fragment),
        "prediction_fragment_sha256": sha256_file(fragment),
        "score": float(row["score"]),
        "residual": int(row["residual"]),
        "error_contribution": float(row["error_contribution"]),
        "metrics": {
            key: int(float(row[key]))
            for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
        },
        "overlay_counts": overlay.counts(),
        "tracklet_count": len(set(overlay.track_ids.values())),
        "predicted_nodes": len(overlay.pred_coords),
        "annotated_predicted_nodes": len(overlay.matched_pred),
        "gt_annotated_nodes": gt_count,
        "estimated_total_nodes": estimated,
        "annotation_fraction": gt_count / estimated if estimated > 0 else None,
        "frame_error_counts": errors_by_frame,
        "prompt3": flag_info,
        "render_contract": dict(task["render_contract"]),
        "rendered_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(sidecar_path, json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar


def _metric_cells(row: Mapping[str, Any]) -> str:
    return (
        f"<td>{float(row['score']):.6f}</td>"
        f"<td>{int(float(row['edge_tp']))} / {int(float(row['edge_fp']))} / {int(float(row['edge_fn']))}</td>"
        f"<td>{int(float(row['division_tp']))} / {int(float(row['division_fp']))} / {int(float(row['division_fn']))}</td>"
    )


def timeline_html(sidecar: Mapping[str, Any]) -> str:
    frames = int(sidecar["frames"])
    prompt3 = sidecar.get("prompt3", {})
    markers: list[str] = []
    for event in prompt3.get("events", []):
        frame = int(event["target_frame"])
        left = 100.0 * frame / max(1, frames - 1)
        event_type = html.escape(str(event["event_type"]))
        label = f"{event_type.replace('_', ' ')} before frame {frame}"
        markers.append(
            f'<button class="flag-marker {event_type}" style="left:{left:.4f}%" '
            f'data-frame="{frame}" aria-label="{html.escape(label)}" title="{html.escape(label)}"></button>'
        )
    error_markers: list[str] = []
    for frame, counts in enumerate(sidecar.get("frame_error_counts", [])):
        total = sum(int(value) for value in counts.values())
        if total <= 0:
            continue
        left = 100.0 * frame / max(1, frames - 1)
        height = min(100, 18 + 16 * math.log2(total + 1))
        title = ", ".join(f"{key.replace('_', ' ')} {value}" for key, value in counts.items() if value)
        error_markers.append(
            f'<span class="error-marker" style="left:{left:.4f}%;height:{height:.1f}%" title="frame {frame}: {html.escape(title)}"></span>'
        )
    status = html.escape(str(prompt3.get("status", "unavailable")))
    return (
        '<div class="timeline-block">'
        '<div class="timeline-label"><span>Error activity</span><span>Prompt 3: '
        + status
        + '</span></div>'
        '<div class="error-timeline" aria-label="Per-frame error activity">'
        + "".join(error_markers)
        + '<span class="playhead" aria-hidden="true"></span></div>'
        '<div class="flag-timeline" aria-label="Frozen-frame and global-jump boundaries">'
        + "".join(markers)
        + '</div><div class="frame-readout" aria-live="polite">Frame 0</div></div>'
    )


def build_html(rows: Sequence[Mapping[str, Any]], rendered: Mapping[str, Mapping[str, Any]], manifest: Mapping[str, Any]) -> str:
    rendered_count = len(rendered)
    total_residual = sum(int(row["residual"]) for row in rows)
    scope = html.escape(str(manifest.get("scope", "default")))
    size_info = manifest.get("size", {})
    measured_bytes = int(size_info.get("measured_bundle_bytes", 0))
    hud_info = manifest.get("hud", {})
    playback = manifest.get("playback", {})
    table_rows: list[str] = []
    cards: list[str] = []
    for row in rows:
        identity = str(row["identity"])
        escaped = html.escape(identity)
        sidecar = rendered.get(identity)
        state = '<span class="ready">ready</span>' if sidecar else '<span class="pending">not rendered yet</span>'
        link = f'<a href="#video-{identity_slug(identity)}">{escaped}</a>' if sidecar else escaped
        categories = list(row.get("categories", []))
        tag_html = "".join(
            f'<span class="tag">{html.escape(SELECTION_LABELS.get(tag, tag))}</span>' for tag in categories
        ) or '<span class="tag neutral">Full catalog</span>'
        table_rows.append(
            "<tr>"
            f"<td>{int(row.get('selection_order', row['rank']))}</td><td>{link}<br><small>{html.escape(str(row['family']))}</small><div class=\"tags\">{tag_html}</div></td>"
            f"<td>{int(row['residual'])}<br><small>{100 * float(row['error_contribution']):.2f}%</small></td>"
            + _metric_cells(row)
            + f"<td>{state}</td></tr>"
        )
        if not sidecar:
            continue
        video = html.escape(str(sidecar["video"]))
        prompt_status = html.escape(str(sidecar.get("prompt3", {}).get("status", "unavailable")))
        cards.append(
            f'<article class="video-card" id="video-{identity_slug(identity)}">'
            f'<header><div><span class="rank">#{int(row.get("selection_order", row["rank"])):02d}</span><h2>{escaped}</h2></div>'
            f'<p>{int(row["residual"])} residual errors · {100 * float(row["error_contribution"]):.2f}% contribution · Prompt 3 {prompt_status}</p></header>'
            f'<div class="tags">{tag_html}</div>'
            f'<video controls playsinline preload="metadata" aria-label="Track review for {escaped}" data-frames="{int(sidecar["frames"])}" data-fps="{float(sidecar["fps"]):.12g}">'
            f'<source src="{video}" type="video/mp4">Your browser does not support MP4 video.</video>'
            + timeline_html(sidecar)
            + '<dl class="metrics">'
            f'<div><dt>Score</dt><dd>{float(row["score"]):.6f}</dd></div>'
            f'<div><dt>Edge TP / FP / FN</dt><dd>{int(float(row["edge_tp"]))} / {int(float(row["edge_fp"]))} / {int(float(row["edge_fn"]))}</dd></div>'
            f'<div><dt>Division TP / FP / FN</dt><dd>{int(float(row["division_tp"]))} / {int(float(row["division_fp"]))} / {int(float(row["division_fn"]))}</dd></div>'
            f'<div><dt>Annotated nuclei</dt><dd>{int(sidecar["gt_annotated_nodes"]):,} / est {int(float(sidecar["estimated_total_nodes"])):,} ({100 * float(sidecar["annotation_fraction"]):.2f}%)</dd></div>'
            f'<div><dt>Tracklets</dt><dd>{int(sidecar["tracklet_count"]):,}</dd></div>'
            '</dl></article>'
        )
    generated = html.escape(str(manifest["generated_at"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>Post-scoring track review</title>
<style>
:root{{--bg:#0b0e14;--panel:#141923;--line:#303849;--text:#edf2f8;--muted:#aab4c2;--accent:#57d6ff;--warn:#ffe38a;--fp:#ff464b;--fn:#ffa523;--frag:#ff28d2;--miss:#ffe62d;--false:#aa69ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}} a{{color:var(--accent)}} :focus-visible{{outline:3px solid var(--accent);outline-offset:3px}}
main{{width:min(1500px,96vw);margin:auto;padding:28px 0 64px}} .hero{{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end}} h1{{font-size:clamp(1.8rem,4vw,3.5rem);line-height:1;margin:.2em 0}} .kicker{{color:var(--accent);letter-spacing:.14em;text-transform:uppercase;font-weight:700}}
.stats{{display:grid;grid-template-columns:repeat(3,minmax(110px,1fr));gap:8px}} .stats div,.metrics div{{background:var(--panel);border:1px solid var(--line);padding:10px 12px;border-radius:8px}} .stats strong{{display:block;font-size:1.4rem}} .warning{{margin:24px 0;padding:14px 18px;border:1px solid #7b6937;background:#2a2415;color:#fff3bd;border-radius:8px}}
.legend{{display:flex;flex-wrap:wrap;gap:9px 16px;margin:16px 0 24px;color:var(--muted)}} .swatch{{width:11px;height:11px;display:inline-block;margin-right:5px;border-radius:2px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}} table{{border-collapse:collapse;width:100%;min-width:850px}} th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid #272e3c}} th{{position:sticky;top:0;background:#1d2330;z-index:1}} small{{color:var(--muted)}} .ready{{color:#8df0ad}} .pending{{color:#e6c06d}}
.tags{{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}} .tag{{display:inline-block;padding:2px 7px;border:1px solid #376379;background:#172b38;color:#b9efff;border-radius:999px;font-size:.7rem}} .tag.neutral{{border-color:#535b6a;background:#252b36;color:#d3dae4}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:22px;margin-top:28px}} .video-card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;scroll-margin-top:12px}} .video-card header{{display:flex;justify-content:space-between;gap:16px;align-items:baseline}} .video-card header>div{{display:flex;gap:9px;align-items:center}} h2{{font-size:1.05rem;margin:0}} .video-card header p{{color:var(--muted);margin:0;font-size:.84rem}} .rank{{background:#233a4a;color:#9fe8ff;border-radius:999px;padding:3px 7px;font-weight:700}}
video{{display:block;width:100%;height:auto;background:#000;border-radius:7px;margin:12px 0 8px}} .timeline-label{{display:flex;justify-content:space-between;color:var(--muted);font-size:.75rem}} .error-timeline,.flag-timeline{{position:relative;height:18px;background:#202635;border:1px solid #394154;border-radius:4px;margin-top:4px}} .flag-timeline{{height:11px}} .error-marker{{position:absolute;bottom:0;width:2px;background:var(--frag);opacity:.85}} .playhead{{position:absolute;top:0;bottom:0;width:2px;background:white;left:0;box-shadow:0 0 4px #fff}} .flag-marker{{position:absolute;top:-3px;width:9px;height:15px;border:1px solid white;transform:translateX(-50%);padding:0;cursor:pointer}} .flag-marker.frozen{{background:#46a5ff}} .flag-marker.global_jump{{background:#ffbe28}} .frame-readout{{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.75rem;text-align:right;margin-top:2px}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(110px,1fr));gap:7px;margin:10px 0 0}} .metrics dt{{font-size:.72rem;color:var(--muted)}} .metrics dd{{margin:2px 0 0;font-weight:650;font-variant-numeric:tabular-nums}}
footer{{color:var(--muted);margin-top:30px}} @media(max-width:760px){{main{{width:94vw;padding-top:18px}} .hero{{grid-template-columns:1fr}} .stats{{grid-template-columns:repeat(3,1fr)}} .video-card header{{display:block}} .metrics{{grid-template-columns:repeat(2,1fr)}}}} @media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important;animation:none!important;transition:none!important}}}}
</style>
</head>
<body><main>
<section class="hero"><div><div class="kicker">Human inspection · {scope} scope</div><h1>Track formation over time</h1><p>The default review is the deduplicated four-category scoring hook; full-catalog review is manual. XY is a native-resolution max-intensity projection and the right panel carries persistent sparse-GT semantics.</p></div>
<div class="stats"><div><strong>{rendered_count}</strong><span>of {len(rows)} rendered</span></div><div><strong>{measured_bytes / 1_000_000:.1f} MB</strong><span>bundle size</span></div><div><strong>{total_residual:,}</strong><span>selected FP + FN</span></div></div></section>
<p><strong>Playback:</strong> target {html.escape(str(playback.get('clip_seconds', '—')))} s; FPS override {html.escape('none' if playback.get('fps_override') is None else str(playback.get('fps_override')))}. <strong>HUD:</strong> {html.escape(', '.join(str(value) for value in hud_info.get('metrics', [])) if hud_info.get('enabled', True) else 'disabled')}.</p>
<aside class="warning"><strong>Sparse annotation warning.</strong> Only GEFF-marked nuclei and scorer-valid edge neighborhoods are annotated. Gray-ringed predictions and neutral track edges are <strong>UNANNOTATED / UNKNOWN</strong>, not false positives. Prompt 3 is unavailable unless a detector manifest is explicitly shown as checked or flagged.</aside>
<div class="legend" aria-label="Overlay legend"><span><i class="swatch" style="background:var(--fp)"></i>FP edge</span><span><i class="swatch" style="background:var(--fn)"></i>FN edge</span><span><i class="swatch" style="background:var(--frag)"></i>fragmentation break</span><span><i class="swatch" style="background:var(--miss)"></i>missed division</span><span><i class="swatch" style="background:var(--false)"></i>false division</span><span><i class="swatch" style="background:#00e1ff"></i>GT annotation</span></div>
<section aria-labelledby="ranked"><h2 id="ranked">Review order and category membership</h2><div class="table-wrap"><table><thead><tr><th>Order</th><th>Video / categories</th><th>Residual / contribution</th><th>Score</th><th>Edge TP / FP / FN</th><th>Division TP / FP / FN</th><th>Animation</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>
<section class="gallery" aria-label="Rendered videos">{''.join(cards)}</section>
<footer>Generated {generated}. Source provenance and exact hashes are in <a href="manifest.json">manifest.json</a>.</footer>
</main>
<script>
document.querySelectorAll('.video-card').forEach(card=>{{const video=card.querySelector('video');const playhead=card.querySelector('.playhead');const readout=card.querySelector('.frame-readout');const frames=Number(video.dataset.frames);const fps=Number(video.dataset.fps);function sync(){{const duration=video.duration||frames/fps;const ratio=Math.max(0,Math.min(1,video.currentTime/duration));playhead.style.left=(ratio*100)+'%';readout.textContent='Frame '+Math.min(frames-1,Math.round(video.currentTime*fps));}} video.addEventListener('timeupdate',sync);video.addEventListener('loadedmetadata',sync);card.querySelectorAll('.flag-marker').forEach(marker=>marker.addEventListener('click',()=>{{video.currentTime=Number(marker.dataset.frame)/fps;video.focus();sync();}}));}});
</script></body></html>"""


def load_sidecars(bundle: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((bundle / "metadata").glob("*.json")) if (bundle / "metadata").exists() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == "celltrack-alpha.track-visualizer-video.v1":
            out[canonical_identity(payload["identity"])] = payload
    return out


def video_frame_count(zip_path: Path, identity: str) -> int:
    split, dataset = canonical_identity(identity).split(":", 1)
    with ZipStore(zip_path, mode="r") as store:
        array = zarr.open_array(store=store, path=f"{split}/{dataset}.zarr/0", mode="r")
        return int(array.shape[0])


def resolve_prediction_fragments(
    rows: Sequence[Mapping[str, Any]], predictions: Path, bundle: Path
) -> dict[str, Path]:
    """Resolve exact source fragments or cache selected graphs from an aggregate CSV."""

    aggregate_graphs: dict[str, SubmissionGraph] | None = None
    aggregate_hash: str | None = None
    if predictions.is_file():
        aggregate_hash = sha256_file(predictions)
        aggregate_graphs = read_submission(predictions)
    elif not predictions.is_dir():
        raise VisualizerError(f"prediction input is neither a CSV nor directory: {predictions}")

    resolved: dict[str, Path] = {}
    for row in rows:
        identity = str(row["identity"])
        _, dataset = identity.split(":", 1)
        source = row.get("source")
        source_path: Path | None = None
        if isinstance(source, Mapping) and source.get("mode") == "fragment" and source.get("path"):
            candidate = Path(str(source["path"]))
            source_path = candidate if candidate.is_absolute() else ROOT / candidate
            if not source_path.exists():
                raise VisualizerError(f"scored prediction fragment no longer exists: {source_path}")
        if source_path is not None:
            resolved[identity] = source_path
            continue
        if aggregate_graphs is None:
            candidate = fragment_path(predictions, identity)
            if not candidate.exists():
                raise VisualizerError(f"prediction fragment does not exist: {candidate}")
            resolved[identity] = candidate
            continue
        graph = aggregate_graphs.get(dataset) or aggregate_graphs.get(identity)
        if graph is None:
            raise VisualizerError(f"aggregate prediction lacks selected dataset {dataset}")
        cache_dir = bundle / "input_fragments" / str(aggregate_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)
        candidate = cache_dir / f"{identity_slug(identity)}.csv"
        if not candidate.exists():
            materialized = graph.copy()
            materialized.dataset = dataset
            write_submission({dataset: materialized}, candidate)
        resolved[identity] = candidate
    return resolved


def fragment_set_hash(fragments_by_identity: Mapping[str, Path]) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for identity in sorted(fragments_by_identity):
        path = fragments_by_identity[identity]
        value = sha256_file(path)
        rows.append({"identity": identity, "path": str(path), "bytes": path.stat().st_size, "sha256": value})
        digest.update(f"{identity}\0{value}\n".encode("utf-8"))
    return digest.hexdigest(), rows


def bundle_bytes(bundle: Path) -> int:
    return sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())


def enforce_bundle_budget(scope: str, measured_bytes: int) -> None:
    if scope == "default" and measured_bytes >= DEFAULT_BUDGET_BYTES:
        raise VisualizerError(
            f"default review bundle is {measured_bytes} bytes, exceeding the <{DEFAULT_BUDGET_BYTES} byte budget"
        )


def garbage_collect_bundle(bundle: Path, identities: Sequence[str]) -> None:
    keep = {identity_slug(identity) for identity in identities}
    for directory, suffix in ((bundle / "videos", ".mp4"), (bundle / "metadata", ".json")):
        if not directory.exists():
            continue
        for path in directory.glob(f"*{suffix}"):
            if path.stem not in keep:
                path.unlink()


def build_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    rendered: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    flags: Mapping[str, Mapping[str, Any]],
    raw_receipt: Mapping[str, Any],
    fragments_by_identity: Mapping[str, Path],
    hud_config: Mapping[str, Any],
    catalog_video_count: int,
    rendered_this_run: int,
    cache_hits: int,
    category_members: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    identities = [str(row["identity"]) for row in rows]
    set_hash, fragments = fragment_set_hash(fragments_by_identity)
    raw_stat = args.zip.stat()
    score_sha256 = sha256_file(args.scores)
    prompt3_sha256 = sha256_file(args.frame_flags) if args.frame_flags else None
    input_sha256 = sha256_json(
        {
            "scope": args.scope,
            "event_kind": args.event_kind,
            "raw_zip_sha256": raw_receipt["observed_sha256"],
            "scores_sha256": score_sha256,
            "fragment_set_sha256": set_hash,
            "prompt3_sha256": prompt3_sha256,
            "clip_seconds": args.clip_seconds,
            "fps_override": args.fps,
            "hud": hud_config,
        }
    )
    videos: list[dict[str, Any]] = []
    for row in rows:
        identity = str(row["identity"])
        sidecar = rendered.get(identity)
        videos.append(
            {
                "identity": identity,
                "rank": int(row["rank"]),
                "selection_order": int(row["selection_order"]),
                "categories": list(row.get("categories", [])),
                "dataset": str(row["dataset"]),
                "family": str(row["family"]),
                "score": float(row["score"]),
                "edge_adjusted_jaccard": float(row["adj_edge_jaccard"]),
                "edge_raw_jaccard": float(row["edge_jaccard"]),
                "division_jaccard": row.get("division_jaccard"),
                "gt_division_count": int(row["gt_division_count"]),
                "node_recall": row.get("node_recall"),
                "residual": int(row["residual"]),
                "error_contribution": float(row["error_contribution"]),
                "edge_tp": int(float(row["edge_tp"])),
                "edge_fp": int(float(row["edge_fp"])),
                "edge_fn": int(float(row["edge_fn"])),
                "division_tp": int(float(row["division_tp"])),
                "division_fp": int(float(row["division_fp"])),
                "division_fn": int(float(row["division_fn"])),
                "rendered": sidecar is not None,
                "video": sidecar.get("video") if sidecar else None,
                "metadata": f"metadata/{identity_slug(identity)}.json" if sidecar else None,
                "prompt3_status": flags[identity]["status"],
                "frames": int(sidecar["frames"]) if sidecar else None,
                "effective_fps": float(sidecar["fps"]) if sidecar else None,
                "encoded_duration_seconds": float(sidecar["encoded_duration_seconds"]) if sidecar else None,
            }
        )
    return {
        "schema": "celltrack-alpha.post-scoring-visual-review.v2",
        "status": "complete" if len(rendered) == len(rows) else "incomplete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_sha256,
        "scope": args.scope,
        "event_kind": args.event_kind,
        "catalog_video_count": catalog_video_count,
        "rendered_video_count": len(rendered),
        "indexed_video_count": len(rows),
        "rendered_this_run": rendered_this_run,
        "cache_hits": cache_hits,
        "playback": {
            "clip_seconds": args.clip_seconds,
            "fps_override": args.fps,
            "rule": "fps override when provided, otherwise num_frames / clip_seconds",
        },
        "hud": dict(hud_config),
        "tail_frames": TAIL_FRAMES,
        "selection": {
            "order": "category precedence for default; adjusted edge Jaccard ascending for all",
            "category_labels": SELECTION_LABELS,
            "category_members": {key: list(value) for key, value in category_members.items()},
        },
        "size": {
            "measured_bundle_bytes": 0,
            "default_budget_bytes": DEFAULT_BUDGET_BYTES if args.scope == "default" else None,
            "budget_enforced": args.scope == "default",
            "budget_pass": None,
        },
        "sources": {
            "raw_zip": {
                "path": str(args.zip),
                "bytes": raw_stat.st_size,
                "registered_sha256": REGISTERED_RAW_SHA256,
                "registered_bytes": REGISTERED_RAW_BYTES,
                "observed_sha256": raw_receipt["observed_sha256"],
                "observed_mtime_ns": int(raw_receipt["mtime_ns"]),
                "integrity_receipt": "raw_zip_integrity.json",
                "integrity_receipt_sha256": sha256_file(args.bundle / "raw_zip_integrity.json"),
                "integrity_basis": str(RAW_INTEGRITY_BASIS),
                "integrity_basis_sha256": raw_receipt["basis_sha256"],
                "verified_at": raw_receipt["verified_at"],
            },
            "predictions": {
                "path": str(args.predictions),
                "fragment_count": len(fragments),
                "fragment_set_sha256": set_hash,
                "fragments": fragments,
            },
            "scores": {"path": str(args.scores), "sha256": score_sha256},
            "prompt3": {
                "path": str(args.frame_flags) if args.frame_flags else None,
                "status": "provided" if args.frame_flags else "unavailable",
                "sha256": prompt3_sha256,
            },
        },
        "sparse_gt_semantics": {
            "annotated": "GEFF node or prediction matched within 7 micrometers at the same frame",
            "false_positive_edge": "official pred_valid and not matched edge",
            "unannotated": "unknown / not scored; never inferred false positive from absence of GT",
            "fragmentation": "official FN edge whose two GT endpoints are both localized in the prediction",
        },
        "videos": videos,
    }


def build_review(args: argparse.Namespace) -> dict[str, Any]:
    args.scope = str(getattr(args, "scope", "default"))
    if args.scope not in {"default", "all"}:
        raise VisualizerError(f"unsupported scope {args.scope!r}")
    args.clip_seconds = float(getattr(args, "clip_seconds", DEFAULT_CLIP_SECONDS))
    args.fps = getattr(args, "fps", None)
    args.hud = bool(getattr(args, "hud", True))
    args.hud_corner = str(getattr(args, "hud_corner", "tr"))
    if args.hud_corner not in HUD_CORNERS:
        raise VisualizerError(f"invalid --hud-corner {args.hud_corner!r}")
    metrics = parse_metrics(getattr(args, "metrics", ",".join(DEFAULT_METRICS)))
    metric_corners = parse_metric_corners(getattr(args, "metric_corner", None), metrics)
    hud_config = {
        "enabled": args.hud,
        "corner": args.hud_corner,
        "metrics": list(metrics),
        "metric_corners": metric_corners,
    }
    if not math.isfinite(args.clip_seconds) or args.clip_seconds <= 0:
        raise VisualizerError("--clip-seconds must be a finite positive number")
    if args.fps is not None and (not math.isfinite(float(args.fps)) or float(args.fps) <= 0):
        raise VisualizerError("--fps must be a finite positive number")
    if int(args.workers) <= 0:
        raise VisualizerError("--workers must be positive")
    if getattr(args, "bundle", None) is None:
        args.bundle = DEFAULT_LATEST_BUNDLE if args.scope == "default" else DEFAULT_ALL_BUNDLE
    args.bundle = Path(args.bundle)
    for path, label in (
        (args.zip, "raw ZIP"),
        (args.predictions, "prediction input"),
        (args.scores, "score artifact"),
    ):
        if not path.exists():
            raise VisualizerError(f"{label} does not exist: {path}")
    ffmpeg = shutil.which(args.ffmpeg) if os.sep not in args.ffmpeg else args.ffmpeg
    if not ffmpeg or not Path(ffmpeg).exists():
        raise VisualizerError(f"ffmpeg executable not found: {args.ffmpeg}")
    score_payload = json.loads(args.scores.read_text(encoding="utf-8"))
    rows = normalize_score_rows(score_payload)
    category_members = selection_categories(rows)
    selected = select_review_rows(rows, args.scope)
    if not selected:
        raise VisualizerError("review selection is empty")
    flags = load_frame_flags(args.frame_flags, [row["identity"] for row in selected])
    args.bundle.mkdir(parents=True, exist_ok=True)
    (args.bundle / "videos").mkdir(exist_ok=True)
    (args.bundle / "metadata").mkdir(exist_ok=True)
    garbage_collect_bundle(args.bundle, [str(row["identity"]) for row in selected])
    raw_receipt = verify_raw_integrity(args.zip, args.bundle)
    renderer_sha256 = sha256_file(Path(__file__))
    fragments_by_identity = resolve_prediction_fragments(selected, args.predictions, args.bundle)
    existing = load_sidecars(args.bundle)
    tasks: list[dict[str, Any]] = []
    cache_hits = 0
    for row in selected:
        identity = str(row["identity"])
        fragment = fragments_by_identity[identity]
        frames = video_frame_count(args.zip, identity)
        fps = effective_fps(frames, clip_seconds=args.clip_seconds, fps=args.fps)
        render_contract = render_request_contract(
            row=row,
            fragment=fragment,
            raw_receipt=raw_receipt,
            flag_info=flags[identity],
            fps=fps,
            renderer_sha256=renderer_sha256,
            clip_seconds=args.clip_seconds,
            hud_config=hud_config,
            scope=args.scope,
        )
        if bool(getattr(args, "cache", True)) and identity in existing:
            if sidecar_is_current(existing[identity], render_contract, args.bundle, flags[identity]):
                cache_hits += 1
                continue
        tasks.append(
            {
                "row": row,
                "zip_path": str(args.zip),
                "fragment_path": str(fragment),
                "bundle": str(args.bundle),
                "flag_info": flags[identity],
                "fps": fps,
                "clip_seconds": args.clip_seconds,
                "hud_config": hud_config,
                "ffmpeg": str(ffmpeg),
                "render_contract": render_contract,
            }
        )
    errors: list[str] = []
    if args.workers == 1:
        for task in tasks:
            try:
                sidecar = render_video_task(task)
                print(json.dumps({"rendered": sidecar["identity"], "rank": sidecar["rank"], "bytes": sidecar["video_bytes"]}), flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['row']['identity']}: {type(exc).__name__}: {exc}")
                break
    elif tasks:
        def consume(executor: ProcessPoolExecutor | ThreadPoolExecutor) -> None:
            futures = {executor.submit(render_video_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    sidecar = future.result()
                    print(json.dumps({"rendered": sidecar["identity"], "rank": sidecar["rank"], "bytes": sidecar["video_bytes"]}), flush=True)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{task['row']['identity']}: {type(exc).__name__}: {exc}")

        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                consume(executor)
        except PermissionError as exc:
            # Some managed macOS sandboxes deny the SC_SEM_NSEMS_MAX probe used
            # by ProcessPoolExecutor before a child starts.  Independent ZIP
            # handles and ffmpeg processes are still safe under bounded threads.
            print(json.dumps({"parallel_fallback": "threads", "reason": str(exc)}), flush=True)
            td.options.set_options(show_progress=False)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                consume(executor)
    if errors:
        raise VisualizerError("render failures:\n" + "\n".join(errors))
    all_sidecars = load_sidecars(args.bundle)
    rendered = {
        str(row["identity"]): all_sidecars[str(row["identity"])]
        for row in selected
        if str(row["identity"]) in all_sidecars
    }
    manifest = build_manifest(
        rows=selected,
        rendered=rendered,
        args=args,
        flags=flags,
        raw_receipt=raw_receipt,
        fragments_by_identity=fragments_by_identity,
        hud_config=hud_config,
        catalog_video_count=len(rows),
        rendered_this_run=len(tasks),
        cache_hits=cache_hits,
        category_members=category_members,
    )
    for _ in range(3):
        atomic_write_text(args.bundle / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        atomic_write_text(args.bundle / "index.html", build_html(selected, rendered, manifest))
        measured = bundle_bytes(args.bundle)
        manifest["size"]["measured_bundle_bytes"] = measured
        manifest["size"]["budget_pass"] = measured < DEFAULT_BUDGET_BYTES if args.scope == "default" else None
    atomic_write_text(args.bundle / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_write_text(args.bundle / "index.html", build_html(selected, rendered, manifest))
    final_bytes = bundle_bytes(args.bundle)
    if final_bytes != int(manifest["size"]["measured_bundle_bytes"]):
        manifest["size"]["measured_bundle_bytes"] = final_bytes
        manifest["size"]["budget_pass"] = final_bytes < DEFAULT_BUDGET_BYTES if args.scope == "default" else None
        atomic_write_text(args.bundle / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        atomic_write_text(args.bundle / "index.html", build_html(selected, rendered, manifest))
        final_bytes = bundle_bytes(args.bundle)
    enforce_bundle_budget(args.scope, final_bytes)
    return {
        "bundle": str(args.bundle),
        "index": str(args.bundle / "index.html"),
        "rendered": len(rendered),
        "indexed": len(selected),
        "catalog": len(rows),
        "rendered_this_run": len(tasks),
        "cache_hits": cache_hits,
        "scope": args.scope,
        "bytes": final_bytes,
        "status": manifest["status"],
    }


def command_build(args: argparse.Namespace) -> int:
    print(json.dumps(build_review(args), indent=2, sort_keys=True))
    return 0


def ffprobe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise VisualizerError(f"ffprobe failed for {path}: {completed.stderr[-1000:]}")
    payload = json.loads(completed.stdout)
    if not payload.get("streams"):
        raise VisualizerError(f"MP4 has no video stream: {path}")
    return payload["streams"][0]


def validate_bundle(bundle: Path, *, min_rendered: int, ffprobe: str) -> dict[str, Any]:
    manifest_path = bundle / "manifest.json"
    index_path = bundle / "index.html"
    if not manifest_path.exists() or not index_path.exists():
        raise VisualizerError(f"bundle lacks manifest.json or index.html: {bundle}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_source = manifest.get("sources", {}).get("raw_zip", {})
    receipt_value = raw_source.get("integrity_receipt")
    if not isinstance(receipt_value, str):
        raise VisualizerError("manifest lacks the observed raw ZIP integrity receipt")
    receipt_path = bundle / receipt_value
    if not receipt_path.exists() or sha256_file(receipt_path) != raw_source.get("integrity_receipt_sha256"):
        raise VisualizerError("raw ZIP integrity receipt is missing or has drifted")
    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        raw_source.get("observed_sha256") != REGISTERED_RAW_SHA256
        or raw_receipt.get("observed_sha256") != REGISTERED_RAW_SHA256
        or int(raw_receipt.get("bytes", -1)) != REGISTERED_RAW_BYTES
    ):
        raise VisualizerError("raw ZIP observed provenance does not match the registered archive")
    raw_path = Path(str(raw_receipt.get("path", "")))
    if (
        not raw_path.exists()
        or raw_path.stat().st_size != int(raw_receipt["bytes"])
        or raw_path.stat().st_mtime_ns != int(raw_receipt["mtime_ns"])
    ):
        raise VisualizerError("raw ZIP stat identity changed after SHA-256 verification")
    videos = manifest.get("videos", [])
    scope = manifest.get("scope")
    if scope not in {"default", "all"}:
        raise VisualizerError(f"manifest has invalid scope {scope!r}")
    if len(videos) != int(manifest.get("indexed_video_count", -1)) or not videos:
        raise VisualizerError("manifest video count is empty or inconsistent")
    input_sha256 = manifest.get("input_sha256")
    if not isinstance(input_sha256, str) or not len(input_sha256) == 64:
        raise VisualizerError("manifest lacks a content input SHA-256")
    actual_bytes = bundle_bytes(bundle)
    if int(manifest.get("size", {}).get("measured_bundle_bytes", -1)) != actual_bytes:
        raise VisualizerError("manifest bundle byte report does not match the filesystem")
    orders = [int(row["selection_order"]) for row in videos]
    if orders != list(range(1, len(videos) + 1)):
        raise VisualizerError("manifest review order is not contiguous")
    if scope == "default":
        if len(videos) > 12:
            raise VisualizerError("default scope contains more than four top/bottom-three categories")
        members = manifest.get("selection", {}).get("category_members", {})
        expected_order: list[str] = []
        for category in SELECTION_LABELS:
            category_ids = members.get(category)
            if not isinstance(category_ids, list) or len(category_ids) > 3:
                raise VisualizerError(f"default category {category} is malformed")
            for identity in category_ids:
                if identity not in expected_order:
                    expected_order.append(identity)
        if [row["identity"] for row in videos] != expected_order:
            raise VisualizerError("default manifest is not the exact deduplicated category order")
        enforce_bundle_budget(scope, actual_bytes)
        if manifest.get("size", {}).get("budget_pass") is not True:
            raise VisualizerError("default bundle does not report a passing budget")
    rendered = [row for row in videos if row.get("rendered")]
    if len(rendered) < min_rendered:
        raise VisualizerError(f"bundle has {len(rendered)} rendered videos; need at least {min_rendered}")
    if any(not row.get("rendered") for row in videos[:min_rendered]):
        raise VisualizerError("front of the review order has a missing animation")
    index_text = index_path.read_text(encoding="utf-8")
    previous_position = -1
    probes: dict[str, Any] = {}
    for row in videos:
        identity = canonical_identity(row["identity"])
        position = index_text.find(identity)
        if position < 0 or position <= previous_position:
            raise VisualizerError(f"index order/content mismatch at {identity}")
        previous_position = position
        if not row.get("rendered"):
            continue
        metadata_path = bundle / str(row["metadata"])
        video_path = bundle / str(row["video"])
        if not metadata_path.exists() or not video_path.exists() or video_path.stat().st_size <= 0:
            raise VisualizerError(f"rendered artifact missing for {identity}")
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        if canonical_identity(sidecar.get("identity", "")) != identity:
            raise VisualizerError(f"{identity} sidecar identity mismatch")
        if int(sidecar.get("video_bytes", -1)) != video_path.stat().st_size:
            raise VisualizerError(f"{identity} video byte count drifted")
        if sha256_file(video_path) != sidecar.get("video_sha256"):
            raise VisualizerError(f"{identity} video SHA-256 drifted")
        contract = sidecar.get("render_contract", {})
        if int(contract.get("version", -1)) != RENDER_CONTRACT_VERSION:
            raise VisualizerError(f"{identity} lacks the current render contract")
        if contract.get("raw_zip_sha256") != raw_receipt.get("observed_sha256"):
            raise VisualizerError(f"{identity} render contract uses a different raw ZIP")
        fragment_path_value = sidecar.get("prediction_fragment")
        if not isinstance(fragment_path_value, str):
            raise VisualizerError(f"{identity} sidecar lacks prediction provenance")
        prediction_fragment = Path(fragment_path_value)
        if not prediction_fragment.exists() or sha256_file(prediction_fragment) != sidecar.get(
            "prediction_fragment_sha256"
        ):
            raise VisualizerError(f"{identity} prediction fragment SHA-256 drifted")
        validate_flag_bounds(sidecar.get("prompt3", {}), int(sidecar.get("frames", 0)), identity)
        if sidecar.get("prompt3", {}).get("status") != row.get("prompt3_status"):
            raise VisualizerError(f"{identity} Prompt 3 manifest/sidecar state mismatch")
        counts = sidecar["overlay_counts"]
        for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn"):
            if int(counts[key]) != int(row[key]):
                raise VisualizerError(f"{identity} {key} overlay/index mismatch")
        stream = ffprobe_video(video_path, ffprobe)
        if int(stream["width"]) != 256 + PANEL_WIDTH or int(stream["height"]) != 256:
            raise VisualizerError(f"{identity} video dimensions are not native-XY-plus-panel")
        if stream.get("codec_name") != "h264":
            raise VisualizerError(f"{identity} codec is {stream.get('codec_name')}, expected h264")
        fps = float(sidecar.get("fps", 0.0))
        frames = int(sidecar.get("frames", 0))
        expected_duration = frames / fps
        observed_duration = _finite_float(stream.get("duration"))
        if observed_duration is not None and abs(observed_duration - expected_duration) > max(0.15, 1.0 / fps):
            raise VisualizerError(
                f"{identity} duration {observed_duration} disagrees with {frames}/{fps}={expected_duration}"
            )
        probes[identity] = stream
    required_phrases = [
        "UNANNOTATED / UNKNOWN",
        "FP edge",
        "FN edge",
        "fragmentation break",
        "missed division",
        "false division",
        "Prompt 3",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in index_text]
    if missing:
        raise VisualizerError(f"index lacks required sparse/error language: {missing}")
    return {
        "status": "PASS",
        "bundle": str(bundle),
        "indexed": len(videos),
        "rendered": len(rendered),
        "scope": scope,
        "bytes": bundle_bytes(bundle),
        "probed_videos": len(probes),
    }


def command_validate(args: argparse.Namespace) -> int:
    ffprobe = shutil.which(args.ffprobe) if os.sep not in args.ffprobe else args.ffprobe
    if not ffprobe or not Path(ffprobe).exists():
        raise VisualizerError(f"ffprobe executable not found: {args.ffprobe}")
    result = validate_bundle(args.bundle, min_rendered=args.min_rendered, ffprobe=str(ffprobe))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="render videos and write/update the sorted HTML index")
    build.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    build.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    build.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    build.add_argument("--bundle", type=Path, default=None, help="defaults to the scope's stable bundle path")
    build.add_argument("--frame-flags", type=Path, default=DEFAULT_FRAME_FLAGS, help="Prompt 3 census or normalized JSON")
    build.add_argument("--scope", choices=("default", "all"), default="default")
    build.add_argument(
        "--event-kind",
        choices=("submission_candidate", "training_checkpoint", "snapshot", "manual"),
        default="manual",
    )
    build.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    build.add_argument("--fps", type=float, default=None, help="exact encoded FPS override")
    build.add_argument("--hud", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--hud-corner", choices=HUD_CORNERS, default="tr")
    build.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="comma-separated HUD registry entries")
    build.add_argument("--metric-corner", default=None, help="comma-separated METRIC=CORNER overrides")
    build.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    build.add_argument("--ffmpeg", default="ffmpeg")
    build.set_defaults(func=command_build)
    validate = subparsers.add_parser("validate", help="validate ranking, metadata, MP4s, and sparse-GT language")
    validate.add_argument("--bundle", type=Path, default=DEFAULT_LATEST_BUNDLE)
    validate.add_argument("--min-rendered", type=int, default=1)
    validate.add_argument("--ffprobe", default="ffprobe")
    validate.set_defaults(func=command_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VisualizerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
