#!/usr/bin/env python3
"""Audit leakage-safe labels, crops, and calibration split for the 3-D linker."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
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
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import compare_submissions as compare  # noqa: E402
import run_bh  # noqa: E402
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import appearance_swap, bh_model  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionGraph,
    SubmissionNode,
)
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
SCHEMA = "celltrack-alpha.crop-linker-data-audit.v1"
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
CANDIDATE_SHA256 = (
    "59478e378bc67d0036e3dbd5e0bd215d987afded43f793b87994f379d940ae80"
)
PREPARE_SHA256 = (
    "c41bc216d4dfd642126d476a66a3fd9659f54a91dc49463f6b468a301a971d89"
)


class CropLinkerAuditError(RuntimeError):
    """The crop-linker fit-only label or image contract failed."""


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def calibration_ids(
    train_ids: list[str],
    *,
    count_44b6: int = 4,
    count_6bba: int = 8,
) -> tuple[str, ...]:
    result = []
    for family, count in (("44b6", count_44b6), ("6bba", count_6bba)):
        eligible = sorted(
            (
                identity
                for identity in train_ids
                if identity.split(":", 1)[1].startswith(family + "_")
            ),
            key=lambda identity: (_stable_key(identity), identity),
        )
        if len(eligible) < count:
            raise CropLinkerAuditError(
                f"insufficient {family} videos for calibration"
            )
        result.extend(eligible[:count])
    return tuple(sorted(result))


def exact_pair_label(
    pair: tuple[int, int],
    mapping_pred_to_gt: Mapping[int, int],
    gt_edges: set[tuple[int, int]],
) -> str:
    source, target = pair
    if source not in mapping_pred_to_gt or target not in mapping_pred_to_gt:
        return "uncertain_masked"
    mapped = (
        int(mapping_pred_to_gt[source]),
        int(mapping_pred_to_gt[target]),
    )
    return "positive" if mapped in gt_edges else "guaranteed_negative"


def _prediction_graph(
    dataset: str, cache: bh_model.SparseVideoCache
) -> tuple[SubmissionGraph, dict[int, tuple[int, np.ndarray]]]:
    nodes = {}
    positions = {}
    for frame in cache.detections:
        for node_id, coord in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            key = int(node_id)
            z, y, x = (int(value) for value in coord)
            nodes[key] = SubmissionNode(
                node_id=key,
                t=int(frame.frame_index),
                z=z,
                y=y,
                x=x,
            )
            positions[key] = (
                int(frame.frame_index),
                np.asarray((z, y, x), dtype=np.float32),
            )
    return SubmissionGraph(dataset=dataset, nodes=nodes, edges=[]), positions


def _gt_graph(
    dataset: str, gt: Mapping[str, Any]
) -> tuple[SubmissionGraph, set[tuple[int, int]]]:
    raw = gt["nodes"]
    nodes = {
        int(node_id): SubmissionNode(
            node_id=int(node_id),
            t=int(t),
            z=int(z),
            y=int(y),
            x=int(x),
        )
        for node_id, t, z, y, x in zip(
            raw["node_id"],
            raw["t"],
            raw["z"],
            raw["y"],
            raw["x"],
            strict=True,
        )
    }
    edges = {
        (int(source), int(target)) for source, target in gt["edges"]
    }
    return SubmissionGraph(dataset=dataset, nodes=nodes, edges=[]), edges


def _unique_prepared_pairs(
    payload: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], str], dict[str, int]]:
    labeled_pairs = np.asarray(
        payload["labeled_challenger_pairs"], dtype=np.int64
    ).reshape(-1, 2)
    labeled_labels = np.asarray(payload["labeled_labels"], dtype=np.int8)
    if len(labeled_pairs) != len(labeled_labels):
        raise CropLinkerAuditError("labeled pair arrays disagree")
    result: dict[tuple[int, int], str] = {}
    for pair, label in zip(labeled_pairs, labeled_labels, strict=True):
        key = (int(pair[0]), int(pair[1]))
        role = "prepared_positive" if int(label) == 1 else "prepared_harmful"
        previous = result.get(key)
        if previous is not None and previous != role:
            raise CropLinkerAuditError("prepared pair has mixed labels")
        result[key] = role
    labeled_count = len(result)
    background_pairs = np.asarray(
        payload["background_challenger_pairs"], dtype=np.int64
    ).reshape(-1, 2)
    duplicate_background = 0
    for pair in background_pairs:
        key = (int(pair[0]), int(pair[1]))
        if key in result:
            duplicate_background += 1
            continue
        result[key] = "prepared_background"
    return result, {
        "unique_labeled_pairs": labeled_count,
        "unique_background_pairs": len(result) - labeled_count,
        "duplicate_background_pairs_removed": duplicate_background,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CropLinkerAuditError(
            f"refusing to overwrite output: {args.output}"
        )
    if sweep._sha256(args.candidate) != args.candidate_sha256:
        raise CropLinkerAuditError("candidate manifest SHA-256 mismatch")
    if sweep._sha256(args.prepare) != args.prepare_sha256:
        raise CropLinkerAuditError("prepared-data SHA-256 mismatch")
    gate = validate_candidate(
        load_json_object(args.candidate), load_json_object(args.policy)
    )
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
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
        raise CropLinkerAuditError("prepared fit-only scope is inadmissible")
    train_ids = [str(value) for value in scope["train_video_ids"]]
    validation_ids = [str(value) for value in scope["validation_video_ids"]]
    calibration = set(calibration_ids(train_ids))
    train_fit = set(train_ids) - calibration
    validation = set(validation_ids)
    if train_fit & calibration or train_fit & validation or calibration & validation:
        raise CropLinkerAuditError("fit/calibration/validation split overlaps")

    split = sweep._load(args.split, "frozen split")
    cv57 = set(
        score_bh_cv57.exact_selection(
            split,
            split_path=args.split,
            archive_video_ids=scoreall.archive_train_video_ids(args.zip),
        )
    )
    overlap = sorted((train_fit | calibration | validation).intersection(cv57))
    if overlap:
        raise CropLinkerAuditError(f"fit117 overlaps CV57: {overlap}")

    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    by_id = {str(row["identity"]): row for row in prepare["videos"]}
    all_ids = train_fit | calibration | validation
    if (
        set(by_id) != all_ids
        or not all_ids.issubset(cache_paths)
        or not all_ids.issubset(meta_paths)
    ):
        raise CropLinkerAuditError("fit117 artifact coverage is incomplete")

    totals: dict[str, int] = defaultdict(int)
    partitions: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    prepared_crosswalk: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    samples = []
    videos = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(sorted(all_ids), start=1):
            dataset = identity.split(":", 1)[1]
            role = (
                "fit_train"
                if identity in train_fit
                else "calibration"
                if identity in calibration
                else "validation"
            )
            family = dataset.split("_", 1)[0]
            raw = by_id[identity]
            payload = torch.load(
                ROOT / str(raw["path"]),
                map_location="cpu",
                weights_only=False,
            )
            pairs, pair_counts = _unique_prepared_pairs(payload)
            meta = sweep._load(
                meta_paths[identity], f"cache metadata for {identity}"
            )
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=sweep._cache_identity(meta),
            )
            pred_graph, positions = _prediction_graph(dataset, cache)
            gt = scoreall.read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in scoreall.parse_scale(dict(group.attrs)))
            if not np.allclose(scale, compare.SCALE_ZYX, atol=0.0, rtol=0.0):
                raise CropLinkerAuditError(
                    f"physical scale drifted for {identity}: {scale}"
                )
            truth_graph, gt_edges = _gt_graph(dataset, gt)
            mapping, match_details = compare.match_nodes(
                truth_graph, pred_graph, max_distance_um=7.0
            )
            local: dict[str, int] = defaultdict(int)
            target_candidates: dict[int, int] = defaultdict(int)
            for pair, prepared_role in sorted(pairs.items()):
                label = exact_pair_label(pair, mapping, gt_edges)
                local[label] += 1
                totals[label] += 1
                partitions[role][family][label] += 1
                prepared_crosswalk[prepared_role][label] += 1
                target_candidates[pair[1]] += 1
                if len(samples) < args.patch_samples * 3:
                    source = positions.get(pair[0])
                    target = positions.get(pair[1])
                    if source is not None and target is not None:
                        samples.append(
                            {
                                "key": _stable_key(
                                    f"{identity}:{pair[0]}:{pair[1]}"
                                ),
                                "identity": identity,
                                "dataset": dataset,
                                "label": label,
                                "source_t": source[0],
                                "source_coord": source[1],
                                "target_t": target[0],
                                "target_coord": target[1],
                            }
                        )
            videos.append(
                {
                    "identity": identity,
                    "partition": role,
                    "family": family,
                    "prepared_pairs": pair_counts,
                    "labels": dict(local),
                    "matched_detection_nodes": len(mapping),
                    "prediction_nodes": len(pred_graph.nodes),
                    "ground_truth_nodes": len(truth_graph.nodes),
                    "maximum_candidates_per_target": max(
                        target_candidates.values(), default=0
                    ),
                    "match_details": match_details,
                }
            )
            print(
                f"crop audit {index:03d}/{len(all_ids)} {identity} "
                f"pos={local['positive']} neg={local['guaranteed_negative']} "
                f"masked={local['uncertain_masked']}",
                flush=True,
            )

        selected_samples = sorted(
            samples, key=lambda row: (row["key"], row["identity"])
        )[: args.patch_samples]
        crop_started = time.perf_counter()
        crop_rows = []
        for sample in selected_samples:
            group = zarr.open_group(
                store=store,
                path=f"train/{sample['dataset']}.zarr",
                mode="r",
            )
            image = group["0"]
            source = appearance_swap.extract_patch(
                np.asarray(image[sample["source_t"]]),
                sample["source_coord"],
            )
            target = appearance_swap.extract_patch(
                np.asarray(image[sample["target_t"]]),
                sample["target_coord"],
            )
            crop_rows.append(
                {
                    "identity": sample["identity"],
                    "label": sample["label"],
                    "source_finite": bool(np.isfinite(source).all()),
                    "target_finite": bool(np.isfinite(target).all()),
                    "source_std": float(np.std(source)),
                    "target_std": float(np.std(target)),
                    "shape": list(source.shape),
                }
            )
        crop_elapsed = time.perf_counter() - crop_started

    expected_shape = list(appearance_swap.PATCH_SHAPE)
    crop_passed = (
        len(crop_rows) == args.patch_samples
        and all(
            row["source_finite"]
            and row["target_finite"]
            and row["source_std"] > 0.0
            and row["target_std"] > 0.0
            and row["shape"] == expected_shape
            for row in crop_rows
        )
    )
    minimums_passed = True
    for role in ("fit_train", "calibration", "validation"):
        for family in ("44b6", "6bba"):
            counts = partitions[role][family]
            required_positive = 20 if role == "fit_train" else 2
            required_negative = 10 if role == "fit_train" else 1
            if (
                counts["positive"] < required_positive
                or counts["guaranteed_negative"] < required_negative
            ):
                minimums_passed = False
    passed = crop_passed and minimums_passed
    bytes_per_pair_fp32 = (
        2 * int(np.prod(appearance_swap.PATCH_SHAPE)) * 4
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "candidate_gate": gate,
        "scope": {
            "fit_only": True,
            "fit_train_video_ids": sorted(train_fit),
            "calibration_video_ids": sorted(calibration),
            "validation_video_ids": sorted(validation),
            "audit_used": False,
            "cv57_overlap": overlap,
        },
        "label_contract": {
            "positive": (
                "both detections matched by the official-style 7um physical "
                "assignment and the mapped pair is a GT edge"
            ),
            "guaranteed_negative": (
                "both detections matched but the mapped pair is not a GT edge"
            ),
            "uncertain_masked": (
                "one or both detections unmatched; never train as negative"
            ),
            "calibration_selection": (
                "four 44b6 and eight 6bba train105 identities selected by "
                "stable SHA-256 ordering before model fitting"
            ),
        },
        "totals": dict(totals),
        "partitions": {
            role: {
                family: dict(counts)
                for family, counts in sorted(families.items())
            }
            for role, families in sorted(partitions.items())
        },
        "prepared_label_crosswalk": {
            role: dict(counts)
            for role, counts in sorted(prepared_crosswalk.items())
        },
        "crop_contract": {
            "shape_zyx": expected_shape,
            "normalization": "per-crop zero-mean/unit-norm at model input",
            "sample_count": len(crop_rows),
            "sample_passed": crop_passed,
            "sample_elapsed_seconds": crop_elapsed,
            "sample_pairs_per_second": (
                len(crop_rows) / crop_elapsed if crop_elapsed > 0 else None
            ),
            "fp32_bytes_per_pair": bytes_per_pair_fp32,
            "fp32_mib_per_batch64": bytes_per_pair_fp32 * 64 / (1024 * 1024),
            "rows": crop_rows,
        },
        "minimum_support_passed": minimums_passed,
        "videos": videos,
        "provenance": {
            "candidate": str(args.candidate),
            "candidate_sha256": args.candidate_sha256,
            "policy": str(args.policy),
            "policy_sha256": sweep._sha256(args.policy),
            "prepare": str(args.prepare),
            "prepare_sha256": args.prepare_sha256,
            "cache": str(args.cache),
            "zip": str(args.zip),
            "zip_sha256": sweep._sha256(args.zip),
            "scorer_parity": str(args.scorer_parity),
            "scorer_parity_sha256": sweep._sha256(args.scorer_parity),
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
            ROOT / "reports/g02/model_candidate_calibrated_3d_crop_linker_v1.json"
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
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    value.add_argument(
        "--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY
    )
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_data_audit_v1.json",
    )
    value.add_argument("--patch-samples", type=int, default=64)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "totals": report["totals"],
                "partitions": report["partitions"],
                "crop_contract": {
                    key: report["crop_contract"][key]
                    for key in (
                        "sample_count",
                        "sample_passed",
                        "sample_pairs_per_second",
                        "fp32_mib_per_batch64",
                    )
                },
            },
            sort_keys=True,
        )
    )
