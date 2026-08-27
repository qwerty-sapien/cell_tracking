#!/usr/bin/env python3
"""Prepare and fit the P4 one-layer edge-self-attention prototype.

Both arms consume the frozen fitted pairwise logits and the same label-safe,
cap-8 sparse source groups.  The only architectural difference is one masked
edge-self-attention layer in the attention arm.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import random
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
from celltrack_alpha import bh_edge, bh_model  # noqa: E402
from celltrack_alpha.edge_attention import (  # noqa: E402
    EdgeResidualModel,
    attention_neighbours,
    supervised_source_loss,
)
from celltrack_alpha.edge_data import (  # noqa: E402
    label_sparse_training_pairs,
    match_detections_to_gt,
)
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.p4-edge-attention-prototype.v1"
DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
DEFAULT_SPLIT_REPORT = (
    ROOT / "_ARCHIVE/conflict_residual/reports/g02/conflict_edge_f0_fit117_v1/report.json"
)
DEFAULT_EDGE = (
    ROOT / "_ARCHIVE/conflict_residual/reports/g02/conflict_edge_f0_fit117_v1/edge.json"
)
DEFAULT_AUXILIARY = ROOT / "_ARCHIVE/compact_head/reports/g02/v45f4.pt"
DEFAULT_SUPPORT = ROOT / "external/Tracking_Support"
DEFAULT_OUTPUT = ROOT / "reports/g02/p4_edge_attention_prototype_v1"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


class P4Error(RuntimeError):
    """The P4 prototype contract was not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P4Error(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise P4Error(f"JSON must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    run_bh.atomic_create_json(path, value)


def _load_base_scorer(
    edge_artifact: Path, auxiliary: Path, support_root: Path, device: str
) -> tuple[Any, Any, Any]:
    model, bundle = bh_model.load_v45_model(support_root, device=device)
    scorer, record = run_bh.load_sparse_edge_scorer(
        edge_artifact,
        _sha256(edge_artifact),
        auxiliary,
        model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or record is None or scorer.config.neighbor_cap != 8:
        raise P4Error("frozen cap-8 pairwise scorer did not restore")
    return scorer, model, record


def _cache_identity(meta_path: Path) -> bh_model.CacheIdentity:
    value = _load_json(meta_path).get("contract", {}).get("model")
    if not isinstance(value, Mapping):
        raise P4Error(f"cache metadata has no model identity: {meta_path}")
    return bh_model.CacheIdentity(**dict(value))


def _frame_records(
    scorer: Any,
    video: bh_model.VideoInference,
    detection_to_gt: Mapping[int, int | None],
    gt_edges: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    candidate_count = supervised_count = skipped_ambiguous = skipped_unknown = 0
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
        supervised: list[int] = []
        positives: list[int] = []
        for source_index, rows in enumerate(rows_by_source):
            # P4 is a competition mechanism, so single-candidate sources carry
            # no useful self-attention comparison and are excluded from loss.
            if len(rows) < 2:
                continue
            local = np.asarray(rows, dtype=np.int64)
            observed = labels[local]
            positive = np.flatnonzero(observed == 1)
            if len(positive) != 1:
                skipped_ambiguous += 1
                continue
            if np.any(observed < 0):
                skipped_unknown += 1
                continue
            supervised.append(source_index)
            positives.append(int(local[int(positive[0])]))
        candidate_count += len(candidates)
        supervised_count += len(supervised)
        if not supervised:
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
                "supervised_sources": np.asarray(supervised, dtype=np.int64),
                "positive_rows": np.asarray(positives, dtype=np.int64),
            }
        )
    return records, {
        "frame_pairs": len(video.contexts),
        "supervised_frame_pairs": len(records),
        "candidate_edges": candidate_count,
        "supervised_sources": supervised_count,
        "skipped_ambiguous_sources": skipped_ambiguous,
        "skipped_unknown_sources": skipped_unknown,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise P4Error(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    frames_root = args.output / "frames"
    frames_root.mkdir()
    split_report = _load_json(args.split_report)
    scope = split_report.get("scope")
    if not isinstance(scope, Mapping):
        raise P4Error("parent fit report has no scope")
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    if len(train_ids) != 105 or len(validation_ids) != 12:
        raise P4Error("P4 must reuse the exact 105/12 fit-only split")
    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact, args.edge_auxiliary, args.support_root, "cpu"
    )
    started = time.perf_counter()
    video_reports: list[dict[str, Any]] = []
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate((*train_ids, *validation_ids), start=1):
            dataset = identity.split(":", 1)[1]
            slug = identity.replace(":", "__", 1)
            cache_path = args.cache / f"{slug}.bhc"
            meta_path = args.cache / f"{slug}.json"
            cache = bh_model.load_sparse_cache(
                cache_path, expected_identity=_cache_identity(meta_path)
            )
            video = bh_model.VideoInference(
                cache.detections,
                cache.contexts,
                pair_batch_size=cache.identity.pair_batch_size,
                view_batch_size=cache.identity.view_batch_size,
            )
            gt = read_geff_arrays(store, "train", dataset)
            matches, gt_edges = match_detections_to_gt(video.detections, gt)
            records, counts = _frame_records(scorer, video, matches, gt_edges)
            destination = frames_root / f"{slug}.pt"
            torch.save(records, destination)
            video_reports.append(
                {
                    "identity": identity,
                    "partition": "train" if identity in train_ids else "validation",
                    "path": str(destination),
                    "sha256": _sha256(destination),
                    **counts,
                }
            )
            print(
                f"P4 prepare {index:03d}/117 {identity} "
                f"frames={counts['supervised_frame_pairs']} sources={counts['supervised_sources']}",
                flush=True,
            )
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
            "candidate_gate": "physical-cKDTree-10um-cap8",
            "supervision": "GT-matched one-positive known-label sources with 2-8 candidates",
            "attention_locality": "shared endpoint or physical edge-midpoint distance <=15um",
            "complexity_bound": "O(E^2)=O(|V|^2*k^2), k<=8",
            "gnn": False,
        },
        "frozen_pairwise": edge_record,
        "frozen_dummy_logit": float(
            scorer.config.dummy_logit
            + scorer.calibrator.dummy_residual.detach().cpu().item()
        ),
        "videos": video_reports,
        "totals": {
            key: sum(int(row[key]) for row in video_reports)
            for key in (
                "frame_pairs",
                "supervised_frame_pairs",
                "candidate_edges",
                "supervised_sources",
                "skipped_ambiguous_sources",
                "skipped_unknown_sources",
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output / "prepare.json", result)
    return result


def _record_tensors(record: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    neighbours, neighbour_mask = attention_neighbours(
        record["source_index"], record["target_index"], record["geometry_um"]
    )
    return {
        "features": torch.as_tensor(record["features"], device=device),
        "base_logits": torch.as_tensor(record["base_logits"], device=device),
        "geometry_um": torch.as_tensor(record["geometry_um"], device=device),
        "source_index": torch.as_tensor(record["source_index"], device=device),
        "supervised_sources": torch.as_tensor(record["supervised_sources"], device=device),
        "positive_rows": torch.as_tensor(record["positive_rows"], device=device),
        "neighbour_index": torch.as_tensor(neighbours, device=device),
        "neighbour_mask": torch.as_tensor(neighbour_mask, device=device),
    }


def _forward(model: EdgeResidualModel, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return tensors["base_logits"] + model(
        tensors["features"],
        tensors["geometry_um"],
        tensors["neighbour_index"],
        tensors["neighbour_mask"],
    )


def _metrics(
    model: EdgeResidualModel,
    paths: Sequence[Path],
    device: torch.device,
    dummy_logit: float,
) -> dict[str, float | int]:
    model.eval()
    losses: list[float] = []
    correct = sources = frames = edges = attention_cells = 0
    with torch.no_grad():
        for path in paths:
            records = torch.load(path, map_location="cpu", weights_only=False)
            for record in records:
                tensors = _record_tensors(record, device)
                logits = _forward(model, tensors)
                loss = supervised_source_loss(
                    logits,
                    tensors["source_index"],
                    tensors["supervised_sources"],
                    tensors["positive_rows"],
                    dummy_logit=dummy_logit,
                )
                losses.append(float(loss.item()))
                for source, positive in zip(
                    tensors["supervised_sources"], tensors["positive_rows"], strict=True
                ):
                    rows = torch.nonzero(
                        tensors["source_index"] == source, as_tuple=False
                    ).flatten()
                    correct += int(rows[torch.argmax(logits[rows])].item() == int(positive.item()))
                    sources += 1
                frames += 1
                edges += int(logits.numel())
                attention_cells += int(tensors["neighbour_mask"].sum().item())
    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "top1_accuracy": correct / max(sources, 1),
        "correct_sources": correct,
        "supervised_sources": sources,
        "frame_pairs": frames,
        "candidate_edges": edges,
        "attention_cells": attention_cells,
    }


def fit(args: argparse.Namespace) -> dict[str, Any]:
    prepare_report = _load_json(args.output / "prepare.json")
    scope = prepare_report["scope"]
    by_id = {row["identity"]: Path(row["path"]) for row in prepare_report["videos"]}
    train_paths = [by_id[value] for value in scope["train_video_ids"]]
    validation_paths = [by_id[value] for value in scope["validation_video_ids"]]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise P4Error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairwise = EdgeResidualModel(attention=False, hidden=args.hidden, heads=args.heads).to(device)
    torch.manual_seed(args.seed)
    attention = EdgeResidualModel(attention=True, hidden=args.hidden, heads=args.heads).to(device)
    attention.input.load_state_dict(copy.deepcopy(pairwise.input.state_dict()))
    attention.input_norm.load_state_dict(copy.deepcopy(pairwise.input_norm.state_dict()))
    attention.output.load_state_dict(copy.deepcopy(pairwise.output.state_dict()))
    models = {"no_attention": pairwise, "attention": attention}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }
    dummy_logit = float(prepare_report["frozen_dummy_logit"])
    initial = {
        name: _metrics(model, validation_paths, device, dummy_logit)
        for name, model in models.items()
    }
    best = {
        name: {"loss": float(initial[name]["loss"]), "state": copy.deepcopy(model.state_dict())}
        for name, model in models.items()
    }
    history: list[dict[str, Any]] = []
    steps = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        order = list(train_paths)
        random.Random(args.seed + epoch).shuffle(order)
        train_loss = {name: [] for name in models}
        stop = False
        for path in order:
            records = torch.load(path, map_location="cpu", weights_only=False)
            random.Random(args.seed + epoch + int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16)).shuffle(records)
            for record in records:
                tensors = _record_tensors(record, device)
                for name, model in models.items():
                    model.train()
                    optimizers[name].zero_grad(set_to_none=True)
                    logits = _forward(model, tensors)
                    loss = supervised_source_loss(
                        logits,
                        tensors["source_index"],
                        tensors["supervised_sources"],
                        tensors["positive_rows"],
                        dummy_logit=dummy_logit,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizers[name].step()
                    train_loss[name].append(float(loss.item()))
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    stop = True
                    break
            if stop:
                break
        validation = {
            name: _metrics(model, validation_paths, device, dummy_logit)
            for name, model in models.items()
        }
        for name, model in models.items():
            if float(validation[name]["loss"]) < float(best[name]["loss"]):
                best[name] = {
                    "loss": float(validation[name]["loss"]),
                    "state": copy.deepcopy(model.state_dict()),
                }
        history.append(
            {
                "epoch": epoch + 1,
                "steps": steps,
                "train_loss": {
                    name: float(np.mean(values)) for name, values in train_loss.items()
                },
                "validation": validation,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if stop:
            break
    for name, model in models.items():
        model.load_state_dict(best[name]["state"], strict=True)
        model.eval()
    final = {
        name: _metrics(model, validation_paths, device, dummy_logit)
        for name, model in models.items()
    }
    artifact_path = args.output / "models.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "hidden": args.hidden,
            "heads": args.heads,
            "dummy_logit": dummy_logit,
            "no_attention": models["no_attention"].state_dict(),
            "attention": models["attention"].state_dict(),
        },
        artifact_path,
    )
    result = {
        "schema": SCHEMA,
        "status": "fitted",
        "device": str(device),
        "same_sparse_supervision": True,
        "only_architectural_difference": "one geometry-masked edge-self-attention layer",
        "parameters": {name: model.parameter_count for name, model in models.items()},
        "optimization": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "steps": steps,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "initial_validation": initial,
        "final_validation": final,
        "history": history,
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output / "fit.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("prepare", "fit"))
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--split-report", type=Path, default=DEFAULT_SPLIT_REPORT)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--epochs", type=int, default=2)
    value.add_argument("--max-steps", type=int, default=0)
    value.add_argument("--hidden", type=int, default=32)
    value.add_argument("--heads", type=int, default=4)
    value.add_argument("--lr", type=float, default=1.0e-3)
    value.add_argument("--weight-decay", type=float, default=1.0e-4)
    value.add_argument("--seed", type=int, default=57)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = prepare(args) if args.command == "prepare" else fit(args)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
