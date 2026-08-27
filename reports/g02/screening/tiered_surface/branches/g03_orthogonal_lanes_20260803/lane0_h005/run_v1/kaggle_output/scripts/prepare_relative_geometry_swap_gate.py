#!/usr/bin/env python3
"""Prepare fit-only edge-vs-edge and edge-vs-no-link swap-gate records."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.edge_data import (  # noqa: E402
    label_sparse_training_pairs,
    match_detections_to_gt,
)
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SPLIT_REPORT,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
    SCALE_ZYX,
    _cache_identity,
    _load_base_scorer,
    _load_json,
    _sha256,
)


SCHEMA = "celltrack-alpha.relative-geometry-swap-prepare.v1"
DEFAULT_OUTPUT = ROOT / "reports/g02/relative_geometry_swap_prepare_v1"


class SwapGatePrepareError(RuntimeError):
    """The swap-gate preparation contract was violated."""


def _records(
    scorer: Any,
    video: Any,
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    gt_out: dict[int, int] = {}
    for source, _target in gt_edges:
        gt_out[int(source)] = gt_out.get(int(source), 0) + 1
    records: list[dict[str, Any]] = []
    counts = {
        "frame_pairs": len(video.contexts),
        "retained_frame_pairs": 0,
        "candidate_edges": 0,
        "known_sources": 0,
        "edge_target_sources": 0,
        "terminal_no_link_sources": 0,
        "unreachable_no_link_sources": 0,
        "division_sources": 0,
        "unknown_sources": 0,
    }
    for context, source, target in zip(
        video.contexts, video.detections[:-1], video.detections[1:], strict=True
    ):
        candidates, checkpoint, features = scorer.candidate_features(context, source, target)
        with torch.no_grad():
            base = checkpoint + scorer.calibrator(features)
        labels = label_sparse_training_pairs(candidates, detection_to_gt, gt_edges).labels
        rows_by_source: list[list[int]] = [[] for _ in range(len(source.node_ids))]
        for row, source_index in enumerate(candidates.source_index):
            rows_by_source[int(source_index)].append(row)
        retained_sources: list[int] = []
        target_rows: list[int] = []
        target_kinds: list[int] = []
        for source_index, rows in enumerate(rows_by_source):
            if not rows:
                continue
            local = np.asarray(rows, dtype=np.int64)
            observed = labels[local]
            if np.any(observed < 0):
                counts["unknown_sources"] += 1
                continue
            positives = np.flatnonzero(observed == 1)
            if len(positives) > 1:
                counts["division_sources"] += 1
                continue
            counts["known_sources"] += 1
            retained_sources.append(source_index)
            if len(positives) == 1:
                target_rows.append(int(local[int(positives[0])]))
                target_kinds.append(0)
                counts["edge_target_sources"] += 1
                continue
            source_id = int(source.node_ids[source_index])
            source_gt = detection_to_gt.get(source_id)
            if source_gt is None:
                raise SwapGatePrepareError("known source unexpectedly lacks a GT match")
            if gt_out.get(int(source_gt), 0) == 0:
                target_kinds.append(1)
                counts["terminal_no_link_sources"] += 1
            else:
                target_kinds.append(2)
                counts["unreachable_no_link_sources"] += 1
            target_rows.append(-1)
        counts["candidate_edges"] += len(candidates)
        if not retained_sources:
            continue
        source_geometry = np.asarray(source.coords_zyx, dtype=np.float64) * SCALE_ZYX
        target_geometry = np.asarray(target.coords_zyx, dtype=np.float64) * SCALE_ZYX
        geometry = np.column_stack(
            (
                source_geometry[candidates.source_index],
                target_geometry[candidates.target_index],
            )
        ).astype(np.float32, copy=False)
        records.append(
            {
                "source_frame": int(source.frame_index),
                "target_frame": int(target.frame_index),
                "source_ids": np.asarray(candidates.source_ids, dtype=np.int64),
                "target_ids": np.asarray(candidates.target_ids, dtype=np.int64),
                "source_index": np.asarray(candidates.source_index, dtype=np.int64),
                "target_index": np.asarray(candidates.target_index, dtype=np.int64),
                "features": features.detach().cpu().numpy().astype(np.float32, copy=False),
                "base_logits": base.detach().cpu().numpy().astype(np.float32, copy=False),
                "geometry_um": geometry,
                "supervised_sources": np.asarray(retained_sources, dtype=np.int64),
                "target_rows": np.asarray(target_rows, dtype=np.int64),
                # 0=edge, 1=true terminal, 2=GT continuation outside cap-8 candidates.
                "target_kinds": np.asarray(target_kinds, dtype=np.int8),
            }
        )
        counts["retained_frame_pairs"] += 1
    return records, counts


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SwapGatePrepareError(f"refusing to overwrite output: {args.output}")
    split_report = _load_json(args.split_report)
    scope = split_report.get("scope")
    if (
        not isinstance(scope, Mapping)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or scope.get("held_out_used") is not False
        or scope.get("audit_used") is not False
        or scope.get("test_used") is not False
    ):
        raise SwapGatePrepareError("parent report is not the exact fit-only 105/12 scope")
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact, args.edge_auxiliary, args.support_root, "cpu"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()
    video_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate((*train_ids, *validation_ids), start=1):
            dataset = identity.split(":", 1)[1]
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            video = bh_model.VideoInference(
                cache.detections,
                cache.contexts,
                pair_batch_size=cache.identity.pair_batch_size,
                view_batch_size=cache.identity.view_batch_size,
            )
            gt = read_geff_arrays(store, "train", dataset)
            matches, gt_edges = match_detections_to_gt(video.detections, gt)
            records, counts = _records(scorer, video, matches, gt_edges)
            destination = frames / f"{slug}.pt"
            torch.save(records, destination)
            video_reports.append(
                {
                    "identity": identity,
                    "partition": "train" if identity in train_ids else "validation",
                    "path": str(destination.relative_to(ROOT)),
                    "sha256": _sha256(destination),
                    **counts,
                }
            )
            print(
                f"swap prepare {index:03d}/117 {identity} "
                f"known={counts['known_sources']} edge={counts['edge_target_sources']} "
                f"no_link={counts['terminal_no_link_sources'] + counts['unreachable_no_link_sources']}",
                flush=True,
            )
    totals = {
        name: sum(int(row[name]) for row in video_reports)
        for name in (
            "frame_pairs",
            "retained_frame_pairs",
            "candidate_edges",
            "known_sources",
            "edge_target_sources",
            "terminal_no_link_sources",
            "unreachable_no_link_sources",
            "division_sources",
            "unknown_sources",
        )
    }
    result = {
        "schema": SCHEMA,
        "status": "prepared",
        "scope": {
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "mechanism": {
            "edge_controls": "fully known candidates with one GT-positive edge",
            "no_link_controls": "fully known candidates with no reachable GT-positive edge",
            "unreachable_policy": "no-link is the operational target when GT continuation is outside cap-8",
            "division_policy": "exclude multi-positive sources",
            "unknown_policy": "exclude any source with an unmatched candidate endpoint",
            "candidate_gate": "frozen physical-cKDTree-10um-cap8",
        },
        "frozen_pairwise": edge_record,
        "frozen_dummy_logit": float(
            scorer.config.dummy_logit
            + scorer.calibrator.dummy_residual.detach().cpu().item()
        ),
        "videos": video_reports,
        "totals": totals,
        "inputs": {
            "split_report": str(args.split_report),
            "split_report_sha256": _sha256(args.split_report),
            "edge_artifact": str(args.edge_artifact),
            "edge_artifact_sha256": _sha256(args.edge_artifact),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_bh.atomic_create_json(args.output / "prepare.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--split-report", type=Path, default=DEFAULT_SPLIT_REPORT)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    result = prepare(parser().parse_args(argv))
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
