#!/usr/bin/env python3
"""Audit fit-only proposal windows for the compact point-context pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
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
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
from fit_selected_graph_temporal_background_gate import (  # noqa: E402
    proposal_confidence,
)


SCHEMA = "celltrack-alpha.compact-point-context-data-audit.v2"
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
CANDIDATE_SHA256 = (
    "1f79c6d6e690942ee370a822cdc866457571e5a81c3cdb56e3c5f6a22688dbfa"
)
PREPARE_SHA256 = (
    "c41bc216d4dfd642126d476a66a3fd9659f54a91dc49463f6b468a301a971d89"
)


class PointContextAuditError(RuntimeError):
    """The compact point-context data boundary or shape contract failed."""


def _group_labeled(
    pairs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[list[list[int]]]]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[(int(pair[0]), int(pair[1]))].append(index)
    unique_pairs = []
    proposal_labels = []
    groups = []
    for pair, indices in grouped.items():
        observed = {int(labels[index]) for index in indices}
        if len(observed) != 1:
            raise PointContextAuditError("labeled proposal has mixed labels")
        unique_pairs.append(pair)
        proposal_labels.append(observed.pop())
        groups.append([[index] for index in indices])
    return (
        np.asarray(unique_pairs, dtype=np.int64).reshape(-1, 2),
        np.asarray(proposal_labels, dtype=np.int8),
        groups,
    )


def _node_times(cache: bh_model.SparseVideoCache) -> dict[int, int]:
    result: dict[int, int] = {}
    for frame in cache.detections:
        for node_id in frame.node_ids:
            key = int(node_id)
            if key in result:
                raise PointContextAuditError("cache node ID is duplicated")
            result[key] = int(frame.frame_index)
    return result


def _window_stats(
    pairs: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    times: Mapping[int, int],
    *,
    token_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    windows: dict[int, list[tuple[float, int, int, int]]] = defaultdict(list)
    targets: dict[int, list[int]] = defaultdict(list)
    for index, (pair, label, score) in enumerate(
        zip(pairs, labels, scores, strict=True)
    ):
        source, target = (int(pair[0]), int(pair[1]))
        if source not in times or target not in times:
            raise PointContextAuditError("proposal references an absent node")
        source_t, target_t = times[source], times[target]
        if target_t != source_t + 1:
            raise PointContextAuditError("proposal is not an adjacent edge")
        windows[target_t].append((float(score), int(label), index, target))
        targets[target].append(int(label))

    multiple_positive_targets = sum(
        sum(label == 1 for label in group) > 1 for group in targets.values()
    )
    result = []
    totals = {
        "windows": 0,
        "tokens": 0,
        "positive_tokens": 0,
        "positive_tokens_retained": 0,
        "windows_over_cap": 0,
        "multiple_positive_targets": multiple_positive_targets,
        "parental_groups": len(targets),
        "no_parent_groups": sum(
            not any(label == 1 for label in group)
            for group in targets.values()
        ),
        "maximum_parent_candidates": max(
            (len(group) for group in targets.values()), default=0
        ),
    }
    for target_t, tokens in sorted(windows.items()):
        # Background (-1) wins exact ties, matching the conservative rank audit.
        selected = sorted(
            tokens,
            key=lambda row: (-row[0], row[1] != -1, row[2]),
        )[:token_cap]
        positive = sum(row[1] == 1 for row in tokens)
        retained = sum(row[1] == 1 for row in selected)
        row = {
            "target_t": target_t,
            "tokens": len(tokens),
            "positive_tokens": positive,
            "positive_tokens_retained": retained,
            "over_cap": len(tokens) > token_cap,
        }
        result.append(row)
        totals["windows"] += 1
        totals["tokens"] += len(tokens)
        totals["positive_tokens"] += positive
        totals["positive_tokens_retained"] += retained
        totals["windows_over_cap"] += int(row["over_cap"])
    return result, totals


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise PointContextAuditError(
            f"refusing to overwrite output: {args.output}"
        )
    if sweep._sha256(args.candidate) != args.candidate_sha256:
        raise PointContextAuditError("candidate manifest SHA-256 mismatch")
    if sweep._sha256(args.prepare) != args.prepare_sha256:
        raise PointContextAuditError("prepared-data SHA-256 mismatch")
    candidate = load_json_object(args.candidate)
    policy = load_json_object(args.policy)
    gate = validate_candidate(candidate, policy)
    prepare = json.loads(args.prepare.read_text(encoding="ascii"))
    scope = prepare.get("scope")
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
        or not isinstance(scope, Mapping)
        or scope.get("audit_used") is not False
        or scope.get("held_out_used") is not False
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
    ):
        raise PointContextAuditError("prepared fit-only scope is inadmissible")
    selected_ids = tuple(
        str(value)
        for key in ("train_video_ids", "validation_video_ids")
        for value in scope[key]
    )
    if len(set(selected_ids)) != 117:
        raise PointContextAuditError("fit-only identities are not unique")

    split = sweep._load(args.split, "frozen split")
    cv57 = set(
        score_bh_cv57.exact_selection(
            split,
            split_path=args.split,
            archive_video_ids=scoreall.archive_train_video_ids(args.zip),
        )
    )
    overlap = sorted(set(selected_ids).intersection(cv57))
    if overlap:
        raise PointContextAuditError(f"fit117 overlaps CV57: {overlap}")

    by_id = {str(row["identity"]): row for row in prepare["videos"]}
    if set(by_id) != set(selected_ids):
        raise PointContextAuditError("prepared video coverage is incomplete")
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(selected_ids).issubset(cache_paths) or not set(
        selected_ids
    ).issubset(meta_paths):
        raise PointContextAuditError("fit117 cache coverage is incomplete")

    totals: dict[str, int] = defaultdict(int)
    family_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    videos = []
    started = time.perf_counter()
    for index, identity in enumerate(selected_ids, start=1):
        row = by_id[identity]
        payload = torch.load(
            ROOT / str(row["path"]), map_location="cpu", weights_only=False
        )
        labeled_x = np.asarray(payload["labeled_features"], dtype=np.float32)
        labeled_y = np.asarray(payload["labeled_labels"], dtype=np.int8)
        labeled_pairs_raw = np.asarray(
            payload["labeled_challenger_pairs"], dtype=np.int64
        )
        background_x = np.asarray(
            payload["background_features"], dtype=np.float32
        )
        background_pairs = np.asarray(
            payload["background_challenger_pairs"], dtype=np.int64
        )
        background_groups = payload["background_groups"]
        if (
            labeled_x.shape != (len(labeled_y), 48)
            or labeled_pairs_raw.shape != (len(labeled_y), 2)
            or background_x.ndim != 2
            or background_x.shape[1:] != (48,)
            or background_pairs.shape != (len(background_groups), 2)
            or not np.isfinite(labeled_x).all()
            or not np.isfinite(background_x).all()
        ):
            raise PointContextAuditError(
                f"prepared arrays are malformed for {identity}"
            )
        labeled_pairs, proposal_labels, labeled_groups = _group_labeled(
            labeled_pairs_raw, labeled_y
        )
        labeled_scores = proposal_confidence(
            labeled_x[:, 44], labeled_groups
        )
        background_scores = proposal_confidence(
            background_x[:, 44], background_groups
        )
        labeled_set = {
            (int(pair[0]), int(pair[1])) for pair in labeled_pairs
        }
        keep_background = np.asarray(
            [
                (int(pair[0]), int(pair[1])) not in labeled_set
                for pair in background_pairs
            ],
            dtype=bool,
        )
        pairs = np.concatenate(
            (labeled_pairs, background_pairs[keep_background])
        )
        labels = np.concatenate(
            (
                proposal_labels,
                np.full(int(keep_background.sum()), -1, dtype=np.int8),
            )
        )
        scores = np.concatenate(
            (labeled_scores, background_scores[keep_background])
        )
        meta = sweep._load(
            meta_paths[identity], f"cache metadata for {identity}"
        )
        cache = bh_model.load_sparse_cache(
            cache_paths[identity],
            expected_identity=sweep._cache_identity(meta),
        )
        windows, counts = _window_stats(
            pairs,
            labels,
            scores,
            _node_times(cache),
            token_cap=args.token_cap,
        )
        partition = str(row["partition"])
        family = identity.split(":", 1)[1].split("_", 1)[0]
        for key, value in counts.items():
            if key == "maximum_parent_candidates":
                totals[key] = max(totals[key], int(value))
                family_totals[family][key] = max(
                    family_totals[family][key], int(value)
                )
            else:
                totals[key] += int(value)
                family_totals[family][key] += int(value)
        videos.append(
            {
                "identity": identity,
                "partition": partition,
                "family": family,
                "labeled_proposals": len(labeled_pairs),
                "background_proposals": int(keep_background.sum()),
                "duplicate_background_proposals_removed": int(
                    len(keep_background) - keep_background.sum()
                ),
                "window_counts": counts,
                "maximum_window_tokens": max(
                    (window["tokens"] for window in windows), default=0
                ),
            }
        )
        print(
            f"point-context audit {index:03d}/{len(selected_ids)} {identity} "
            f"tokens={counts['tokens']} windows={counts['windows']}",
            flush=True,
        )

    positive_recall = totals["positive_tokens_retained"] / max(
        totals["positive_tokens"], 1
    )
    max_window_tokens = max(
        (int(row["maximum_window_tokens"]) for row in videos), default=0
    )
    # Conservative fp32 attention-score storage for two layers and four heads.
    attention_score_mib = (
        2 * 4 * args.token_cap * args.token_cap * 4 / (1024 * 1024)
    )
    passed = (
        totals["multiple_positive_targets"] == 0
        and positive_recall >= args.minimum_positive_retention
        and max_window_tokens <= args.token_cap
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "candidate_gate": gate,
        "scope": {
            "fit_only": True,
            "train_videos": 105,
            "validation_videos": 12,
            "audit_used": False,
            "cv57_overlap": overlap,
        },
        "token_contract": {
            "unit": "one node-disjoint challenger proposal",
            "window": (
                "one adjacent transition; each 48-feature token already includes "
                "predecessor/successor temporal context"
            ),
            "feature_count": 48,
            "token_cap": args.token_cap,
            "ranking_for_cap_audit": "pair-NCC proposal confidence",
            "minimum_positive_retention": args.minimum_positive_retention,
        },
        "totals": {
            **dict(totals),
            "positive_retention": positive_recall,
            "maximum_window_tokens": max_window_tokens,
        },
        "families": {
            family: dict(counts)
            for family, counts in sorted(family_totals.items())
        },
        "memory_projection": {
            "two_layer_four_head_fp32_attention_scores_mib": attention_score_mib,
            "candidate_projected_peak_memory_gib": gate[
                "projected_peak_memory_gib"
            ],
            "policy_maximum_training_memory_gib": gate[
                "maximum_training_memory_gib"
            ],
        },
        "videos": videos,
        "provenance": {
            "candidate": str(args.candidate),
            "candidate_sha256": args.candidate_sha256,
            "policy": str(args.policy),
            "policy_sha256": sweep._sha256(args.policy),
            "prepare": str(args.prepare),
            "prepare_sha256": args.prepare_sha256,
            "cache": str(args.cache),
            "split": str(args.split),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "training_started": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--candidate",
        type=Path,
        default=(
            ROOT
            / "reports/g02/model_candidate_compact_point_context_transformer_v1.json"
        ),
    )
    value.add_argument("--candidate-sha256", default=CANDIDATE_SHA256)
    value.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "reports/g02/compute_governance_v1.json",
    )
    value.add_argument(
        "--prepare",
        type=Path,
        default=(
            ROOT
            / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
        ),
    )
    value.add_argument("--prepare-sha256", default=PREPARE_SHA256)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/compact_point_context_data_audit_v2.json",
    )
    value.add_argument("--token-cap", type=int, default=512)
    value.add_argument(
        "--minimum-positive-retention", type=float, default=0.999
    )
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "totals": report["totals"],
                "memory_projection": report["memory_projection"],
            },
            sort_keys=True,
        )
    )
