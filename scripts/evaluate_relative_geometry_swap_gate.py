#!/usr/bin/env python3
"""Evaluate the frozen relative-geometry swap gate on the registered CV57 slice."""

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
import zarr
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
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.relative_geometry_gate import (  # noqa: E402
    RelativeGeometryGate,
    choose_swap,
    relative_features,
)
from p4_edge_attention_cv57 import _normalize, _solve  # noqa: E402


SCHEMA = "celltrack-alpha.relative-geometry-swap-screen.v1"
REGISTERED_TOP10 = (
    "train:44b6_267148e4",
    "train:44b6_71a4179f",
    "train:44b6_87bba6c4",
    "train:44b6_cf2536e8",
    "train:44b6_d5e7d891",
    "train:44b6_e57ff5c6",
    "train:6bba_3abfe10a",
    "train:6bba_786893ac",
    "train:6bba_ebdf3b34",
    "train:6bba_f17befbc",
)
DEFAULT_GATE = ROOT / "reports/g02/relative_geometry_swap_gate_v2/gate.pt"
DEFAULT_GATE_REPORT = ROOT / "reports/g02/relative_geometry_swap_gate_v2/report.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/relative_geometry_swap_screen10_v1"


class SwapGateEvalError(RuntimeError):
    """The frozen affected-slice evaluation contract was violated."""


def _load_gate(path: Path) -> tuple[RelativeGeometryGate, float, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "celltrack-alpha.relative-geometry-swap-gate.v1"
    ):
        raise SwapGateEvalError("swap-gate artifact schema is invalid")
    model = RelativeGeometryGate(hidden=int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, float(payload["threshold"]), float(payload["dummy_logit"])


def _gate_cache(
    cache: Any,
    scorer: Any,
    gate: RelativeGeometryGate,
    *,
    threshold: float,
    dummy_logit: float,
    promotion_margin: float,
) -> tuple[dict[tuple[int, int], float], dict[int, float], dict[str, int]]:
    probabilities: dict[tuple[int, int], float] = {}
    no_link: dict[int, float] = {}
    counts = {
        "sources": 0,
        "edge_winners": 0,
        "dummy_winners": 0,
        "accepted_promotions": 0,
        "edge_to_edge_promotions": 0,
        "dummy_to_edge_promotions": 0,
    }
    by_frame = {int(frame.frame_index): frame for frame in cache.detections}
    for context in cache.contexts:
        source = by_frame[int(context.source_frame)]
        target = by_frame[int(context.target_frame)]
        candidates, checkpoint, features = scorer.candidate_features(context, source, target)
        with torch.no_grad():
            logits = (checkpoint + scorer.calibrator(features)).detach().cpu().numpy()
        source_geometry = np.asarray(source.coords_zyx, dtype=np.float64) * np.asarray(
            (1.625, 0.40625, 0.40625), dtype=np.float64
        )
        target_geometry = np.asarray(target.coords_zyx, dtype=np.float64) * np.asarray(
            (1.625, 0.40625, 0.40625), dtype=np.float64
        )
        geometry = np.column_stack(
            (
                source_geometry[candidates.source_index],
                target_geometry[candidates.target_index],
            )
        ).astype(np.float32, copy=False)
        promoted = np.asarray(logits, dtype=np.float32).copy()
        for source_index, source_id in enumerate(source.node_ids):
            rows = np.flatnonzero(candidates.source_index == source_index)
            counts["sources"] += 1
            if not len(rows):
                counts["dummy_winners"] += 1
                continue
            best_edge = int(rows[int(np.argmax(logits[rows]))])
            winner = -1 if dummy_logit >= float(logits[best_edge]) else best_edge
            if winner < 0:
                counts["dummy_winners"] += 1
                challengers = rows
                winner_features = np.zeros((len(rows), features.shape[1]), dtype=np.float32)
                source_position = geometry[rows, :3]
                winner_geometry = np.column_stack((source_position, source_position)).astype(
                    np.float32, copy=False
                )
                winner_logits = np.full(len(rows), dummy_logit, dtype=np.float32)
                winner_dummy = np.ones(len(rows), dtype=np.float32)
            else:
                counts["edge_winners"] += 1
                challengers = rows[rows != winner]
                winner_features = np.repeat(
                    features[winner].detach().cpu().numpy()[None],
                    len(challengers),
                    axis=0,
                )
                winner_geometry = np.repeat(
                    geometry[winner][None], len(challengers), axis=0
                )
                winner_logits = np.full(
                    len(challengers), logits[winner], dtype=np.float32
                )
                winner_dummy = np.zeros(len(challengers), dtype=np.float32)
            if not len(challengers):
                continue
            relative = relative_features(
                features[challengers].detach().cpu().numpy(),
                winner_features,
                geometry[challengers],
                winner_geometry,
                challenger_logit=logits[challengers],
                winner_logit=winner_logits,
                dummy_logit=dummy_logit,
                candidate_count=np.full(
                    len(challengers), len(rows) + 1, dtype=np.float32
                ),
                winner_is_dummy=winner_dummy,
            )
            with torch.no_grad():
                gate_probabilities = torch.sigmoid(
                    gate(torch.from_numpy(relative))
                ).numpy()
            decision = choose_swap(
                challengers, gate_probabilities, threshold=threshold
            )
            if decision.challenger_row is None:
                continue
            counts["accepted_promotions"] += 1
            if winner < 0:
                counts["dummy_to_edge_promotions"] += 1
            else:
                counts["edge_to_edge_promotions"] += 1
            promoted[decision.challenger_row] = (
                max(dummy_logit, float(logits[best_edge])) + promotion_margin
            )
        local_probabilities, local_no_link = _normalize(
            candidates.source_ids,
            candidates.target_ids,
            candidates.source_index,
            source.node_ids,
            promoted,
            dummy_logit,
        )
        if set(probabilities).intersection(local_probabilities) or set(no_link).intersection(
            local_no_link
        ):
            raise SwapGateEvalError("candidate identities repeat across frame pairs")
        probabilities.update(local_probabilities)
        no_link.update(local_no_link)
    return probabilities, no_link, counts


def _edge_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        name: int(sum(float(row[name]) for row in rows))
        for name in ("edge_tp", "edge_fp", "edge_fn")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SwapGateEvalError(f"refusing to overwrite output: {args.output}")
    if args.promotion_margin <= 0.0:
        raise SwapGateEvalError("promotion margin must be positive")
    gate_report = json.loads(args.gate_report.read_text(encoding="ascii"))
    if (
        gate_report.get("scope", {}).get("held_out_used") is not False
        or gate_report.get("scope", {}).get("audit_used") is not False
        or gate_report.get("scope", {}).get("test_used") is not False
    ):
        raise SwapGateEvalError("gate training was not fit-only")
    split = sweep._load(args.split, "frozen split")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    cv57 = score_bh_cv57.exact_selection(
        split,
        split_path=args.split,
        archive_video_ids=scoreall.archive_train_video_ids(args.zip),
    )
    if not set(REGISTERED_TOP10).issubset(cv57):
        raise SwapGateEvalError("registered affected slice escaped CV57")
    selected = tuple(identity for identity in cv57 if identity in set(REGISTERED_TOP10))
    if set(selected) != set(REGISTERED_TOP10):
        raise SwapGateEvalError("registered affected slice identity mismatch")
    baseline_names = {
        identity.replace(":", "__", 1) + ".csv" for identity in selected
    }
    baseline_candidates = [
        path
        for path in scoreall.resolve_fragment_paths(str(args.baseline_fragments))
        if path.name in baseline_names
    ]
    baseline_manifest = scoreall.validate_fragment_set(baseline_candidates, selected)
    cache_paths = {
        path.stem.replace("__", ":", 1): path for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(selected).issubset(cache_paths) or not set(selected).issubset(meta_paths):
        raise SwapGateEvalError("cached CV57 coverage is incomplete")
    base_model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sweep._sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise SwapGateEvalError("frozen sparse edge scorer did not restore")
    gate, threshold, dummy_logit = _load_gate(args.gate)
    args.output.mkdir(parents=True)
    fragment_dir = args.output / "fragments"
    fragment_dir.mkdir()
    video_reports: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(selected, start=1):
            dataset = identity.split(":", 1)[1]
            meta = sweep._load(meta_paths[identity], f"cache metadata for {identity}")
            cache = bh_model.load_sparse_cache(
                cache_paths[identity], expected_identity=sweep._cache_identity(meta)
            )
            probabilities, no_link, gate_counts = _gate_cache(
                cache,
                scorer,
                gate,
                threshold=threshold,
                dummy_logit=dummy_logit,
                promotion_margin=args.promotion_margin,
            )
            graph_nodes = run_bh.detections_to_graph_nodes(
                cache.detections, bh_graph.GraphNode
            )
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            image = group["0"]
            shape = tuple(int(value) for value in image.shape)
            solved = _solve(
                identity=identity,
                dataset=dataset,
                shape=shape,
                graph_nodes=graph_nodes,
                probabilities=probabilities,
                no_link=no_link,
                refiner=sweep.MemoizedImageRefiner(image),
                fragment_dir=fragment_dir,
            )
            video_reports.append({"identity": identity, **gate_counts, **solved})
            for name, value in gate_counts.items():
                totals[name] = totals.get(name, 0) + int(value)
            print(
                f"swap screen {index:02d}/10 {identity} "
                f"promotions={gate_counts['accepted_promotions']}",
                flush=True,
            )
        candidate_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(fragment_dir)), selected
        )
        baseline_rows, baseline_provenance, baseline_coverage = scoreall.score_fragment_input(
            store, baseline_manifest
        )
        candidate_rows, candidate_provenance, candidate_coverage = scoreall.score_fragment_input(
            store, candidate_manifest
        )
    baseline_by_id = {row["identity"]: row for row in baseline_rows}
    candidate_by_id = {row["identity"]: row for row in candidate_rows}
    ordered_baseline = [baseline_by_id[identity] for identity in selected]
    ordered_candidate = [candidate_by_id[identity] for identity in selected]
    baseline_summary = scoreall.official_summary(ordered_baseline)
    candidate_summary = scoreall.official_summary(ordered_candidate)
    delta = float(candidate_summary["adj_edge_jaccard"]) - float(
        baseline_summary["adj_edge_jaccard"]
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {
            "role": "preregistered-worst-residual-top10",
            "video_count": len(selected),
            "video_ids": list(selected),
        },
        "configuration": {
            "threshold": threshold,
            "promotion_margin": args.promotion_margin,
            "motion_direction": "backward",
            "motion_error_weight": 0.25,
            "division_enabled": False,
            "gnn": False,
        },
        "baseline": {
            "summary": baseline_summary,
            "edge_counts": _edge_counts(ordered_baseline),
            "coverage": baseline_coverage,
            "input_provenance": baseline_provenance,
        },
        "candidate": {
            "summary": candidate_summary,
            "edge_counts": _edge_counts(ordered_candidate),
            "coverage": candidate_coverage,
            "input_provenance": candidate_provenance,
            "gate_counts": totals,
            "videos": video_reports,
        },
        "decision": {
            "delta_adjusted_edge_jaccard": delta,
            "proxy_gate": 0.005,
            "status": "CONFIRM" if delta >= 0.005 else "PARK",
            "full_cv57_authorized": delta >= 0.005,
        },
        "provenance": {
            "gate": str(args.gate),
            "gate_sha256": sweep._sha256(args.gate),
            "gate_report": str(args.gate_report),
            "gate_report_sha256": sweep._sha256(args.gate_report),
            "edge_artifact": edge_record,
            "scorer_parity": str(args.scorer_parity),
        },
        "execution": {
            "device": "cpu",
            "kaggle_action": False,
            "gpu_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=sweep.DEFAULT_CACHE)
    value.add_argument("--baseline-fragments", type=Path, default=sweep.DEFAULT_BASELINE)
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY)
    value.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    value.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    value.add_argument("--promotion-margin", type=float, default=0.01)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({"status": report["status"], "decision": report["decision"]}))
