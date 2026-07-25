#!/usr/bin/env python3
"""Prepare deterministic unlabeled background for the selected-graph gate.

Known labeled rescue/control challengers are excluded.  Each fit-train video
contributes a bounded mixture of:

* high-confidence proposals under the v1 gate (hard deployment background);
* hash-uniform proposals (coverage background).

The examples are unlabeled, not asserted GT negatives.  Their only permitted
use is as a low-weight abstention prior.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from evaluate_selected_graph_conflict_gate import (  # noqa: E402
    _load_gate,
    _proposal_feature_groups,
)
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    _cache_identity,
    _load_base_scorer,
)
from prepare_selected_graph_conflict_gate import _selected_edges  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-background-prepare.v1"
DEFAULT_PREPARE = ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
DEFAULT_GATE = ROOT / "reports/g02/selected_graph_conflict_gate_v1/gate.pt"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_background_prepare_v1"


class BackgroundPrepareError(RuntimeError):
    """The background preparation contract was violated."""


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise BackgroundPrepareError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_rank(seed: int, identity: str, pair: Sequence[int]) -> int:
    payload = f"{seed}|{identity}|{int(pair[0])}|{int(pair[1])}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _known_challengers(path: Path) -> set[tuple[int, int]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or "challenger_pairs" not in value:
        raise BackgroundPrepareError(f"invalid labeled example artifact: {path}")
    pairs = np.asarray(value["challenger_pairs"], dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise BackgroundPrepareError(f"invalid labeled challenger pairs: {path}")
    return {(int(source), int(target)) for source, target in pairs.tolist()}


def _select_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    identity: str,
    known: set[tuple[int, int]],
    gate: torch.nn.Module,
    seed: int,
    sample_rate: float,
    minimum: int,
    maximum: int,
    hard_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    for raw in groups:
        pair = tuple(int(value) for value in raw["challenger"])
        if pair in known:
            continue
        features = np.asarray(raw["relative_features"], dtype=np.float32)
        with torch.no_grad():
            probabilities = torch.sigmoid(gate(torch.from_numpy(features))).numpy()
        candidates.append(
            {
                "challenger": pair,
                "features": features,
                "confidence": float(np.min(probabilities)),
                "hash_rank": _hash_rank(seed, identity, pair),
            }
        )
    target = min(
        len(candidates),
        maximum,
        max(minimum, int(math.ceil(sample_rate * len(candidates)))),
    )
    hard_count = min(target, int(round(target * hard_fraction)))
    hard = sorted(
        candidates,
        key=lambda row: (
            -float(row["confidence"]),
            int(row["hash_rank"]),
        ),
    )[:hard_count]
    hard_pairs = {row["challenger"] for row in hard}
    uniform = sorted(
        (row for row in candidates if row["challenger"] not in hard_pairs),
        key=lambda row: int(row["hash_rank"]),
    )[: target - hard_count]
    selected = sorted(
        (*hard, *uniform),
        key=lambda row: (
            int(row["challenger"][0]),
            int(row["challenger"][1]),
        ),
    )
    return selected, {
        "unlabeled_proposals": len(candidates),
        "selected_proposals": len(selected),
        "selected_hard": len(hard),
        "selected_uniform": len(uniform),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise BackgroundPrepareError(f"refusing to overwrite output: {args.output}")
    if (
        not 0.0 < args.sample_rate <= 1.0
        or not 0.0 <= args.hard_fraction <= 1.0
        or args.minimum_per_video < 1
        or args.maximum_per_video < args.minimum_per_video
    ):
        raise BackgroundPrepareError("background sampling parameters are invalid")
    parent = _load(args.prepare)
    scope = parent.get("scope")
    videos = parent.get("videos")
    if (
        parent.get("schema") != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or not isinstance(scope, Mapping)
        or not isinstance(videos, list)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or scope.get("held_out_used") is not False
        or scope.get("audit_used") is not False
        or scope.get("test_used") is not False
    ):
        raise BackgroundPrepareError("parent preparation is not exact fit-only 105/12")
    by_id = {str(row["identity"]): ROOT / str(row["path"]) for row in videos}
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    if not set(train_ids).issubset(by_id):
        raise BackgroundPrepareError("labeled fit-train coverage is incomplete")

    scorer, _model, edge_record = _load_base_scorer(
        args.edge_artifact, args.edge_auxiliary, args.support_root, "cpu"
    )
    gate, _threshold = _load_gate(args.gate)
    args.output.mkdir(parents=True, exist_ok=False)
    examples = args.output / "examples"
    examples.mkdir()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, identity in enumerate(train_ids, start=1):
        slug = identity.replace(":", "__", 1)
        cache_path = args.cache / f"{slug}.bhc"
        cache = bh_model.load_sparse_cache(
            cache_path,
            expected_identity=_cache_identity(args.cache / f"{slug}.json"),
        )
        selected_edges, graph = _selected_edges(cache, scorer)
        groups, proposal_counts = _proposal_feature_groups(
            cache, scorer, selected_edges
        )
        selected_groups, selection_counts = _select_groups(
            groups,
            identity=identity,
            known=_known_challengers(by_id[identity]),
            gate=gate,
            seed=args.seed,
            sample_rate=args.sample_rate,
            minimum=args.minimum_per_video,
            maximum=args.maximum_per_video,
            hard_fraction=args.hard_fraction,
        )
        feature_parts = [row["features"] for row in selected_groups]
        features = (
            np.concatenate(feature_parts, axis=0)
            if feature_parts
            else np.empty((0, 44), dtype=np.float32)
        )
        lengths = np.asarray(
            [len(row["features"]) for row in selected_groups], dtype=np.int8
        )
        destination = examples / f"{slug}.pt"
        torch.save(
            {
                "relative_features": features,
                "proposal_lengths": lengths,
                "challenger_pairs": np.asarray(
                    [row["challenger"] for row in selected_groups], dtype=np.int64
                ).reshape(len(selected_groups), 2),
                "v1_confidence": np.asarray(
                    [row["confidence"] for row in selected_groups], dtype=np.float32
                ),
            },
            destination,
        )
        rows.append(
            {
                "identity": identity,
                "path": str(destination.relative_to(ROOT)),
                "sha256": _sha256(destination),
                **proposal_counts,
                **selection_counts,
                "selected_rows": len(features),
                "v1_confidence_quantiles": {
                    str(quantile): (
                        float(
                            np.quantile(
                                [row["confidence"] for row in selected_groups],
                                quantile,
                            )
                        )
                        if selected_groups
                        else None
                    )
                    for quantile in (0.5, 0.9, 0.95, 0.99)
                },
                "graph": graph,
            }
        )
        print(
            f"background prepare {index:03d}/105 {identity} "
            f"pool={selection_counts['unlabeled_proposals']} "
            f"sample={selection_counts['selected_proposals']}",
            flush=True,
        )

    totals = {
        name: sum(int(row[name]) for row in rows)
        for name in (
            "candidate_pairs",
            "compared_rows",
            "raw_proposals",
            "unlabeled_proposals",
            "selected_proposals",
            "selected_hard",
            "selected_uniform",
            "selected_rows",
        )
    }
    result = {
        "schema": SCHEMA,
        "status": "prepared",
        "scope": {
            "train_video_ids": list(train_ids),
            "fit_train_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "sampling": {
            "seed": args.seed,
            "sample_rate": args.sample_rate,
            "minimum_per_video": args.minimum_per_video,
            "maximum_per_video": args.maximum_per_video,
            "hard_fraction": args.hard_fraction,
            "hard_source": "highest v1 proposal confidence",
            "coverage_source": "lowest sha256 rank per identity/challenger",
            "known_labeled_challengers_excluded": True,
            "label_contract": "unlabeled deployment background; low-weight abstention only",
        },
        "videos": rows,
        "totals": totals,
        "provenance": {
            "parent_prepare": str(args.prepare),
            "parent_prepare_sha256": _sha256(args.prepare),
            "v1_gate": str(args.gate),
            "v1_gate_sha256": _sha256(args.gate),
            "edge_artifact": edge_record,
        },
        "execution": {
            "device": "cpu",
            "kaggle_action": False,
            "gpu_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "prepare.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--sample-rate", type=float, default=0.01)
    value.add_argument("--minimum-per-video", type=int, default=64)
    value.add_argument("--maximum-per-video", type=int, default=256)
    value.add_argument("--hard-fraction", type=float, default=0.5)
    value.add_argument("--seed", type=int, default=131)
    return value


if __name__ == "__main__":
    report = prepare(parser().parse_args())
    print(json.dumps({"status": report["status"], "totals": report["totals"]}))
