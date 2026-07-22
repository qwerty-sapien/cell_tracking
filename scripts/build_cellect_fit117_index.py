#!/usr/bin/env python3
"""Build the fit-only temporal index for the GPL CELLECT experiment.

Only Zarr/GEFF metadata and sparse labels are read. Image pixels are not read,
no model is trained, and CV57 identities are used only as an exclusion check.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.cellect_adapter import CellectContractError, sha256_file  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.cellect-fit117-index.v1"
EXPECTED_ARCHIVE_BYTES = 87_393_127_165
EXPECTED_FIT_VIDEOS = 117
EXCLUDED_HOLDOUT_IDENTITY = "train:6bba_f4ae811c"
SPLIT_SEED = "cellect-fit117-internal-validation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=ROOT / "reports/g02/fitcache_117/manifest.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/cellect_fit117_index_v1.json",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CellectContractError(f"JSON root is not an object: {path}")
    return value


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def holdout_ids(split: dict[str, Any]) -> list[str]:
    fold = split.get("folds", {}).get("0")
    if not isinstance(fold, dict):
        raise CellectContractError("fold 0 is missing from split")
    held = [str(value) for value in fold.get("held_out_video_ids", ())]
    if len(held) != 58 or EXCLUDED_HOLDOUT_IDENTITY not in held:
        raise CellectContractError("frozen 58-video fold-0 holdout changed")
    result = [value for value in held if value != EXCLUDED_HOLDOUT_IDENTITY]
    if len(result) != 57:
        raise CellectContractError("exact holdout is not 57 videos")
    return result


def stratified_video_split(
    identities: Iterable[str], validation_fraction: float
) -> tuple[list[str], list[str]]:
    """Split whole videos, stratified by the dataset prefix."""

    if not 0.0 < validation_fraction < 0.5:
        raise CellectContractError("validation fraction must lie in (0, 0.5)")
    strata: dict[str, list[str]] = defaultdict(list)
    for identity in identities:
        split_name, dataset = str(identity).split(":", 1)
        if split_name != "train":
            raise CellectContractError(f"non-train identity in fit117: {identity}")
        strata[dataset.split("_", 1)[0]].append(str(identity))
    train: list[str] = []
    validation: list[str] = []
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda value: (
                hashlib.sha256(f"{SPLIT_SEED}:{value}".encode("ascii")).hexdigest(),
                value,
            ),
        )
        validation_count = max(1, int(round(len(ordered) * validation_fraction)))
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
        if not train or not validation:
            raise CellectContractError(f"empty internal split for stratum {key}")
    return sorted(train), sorted(validation)


def video_record(store: ZipStore, identity: str) -> dict[str, Any]:
    split_name, dataset = identity.split(":", 1)
    image = zarr.open_array(store=store, path=f"{split_name}/{dataset}.zarr/0", mode="r")
    if len(image.shape) != 4 or image.shape[0] < 3:
        raise CellectContractError(f"CELLECT requires a (T,Z,Y,X) video with T>=3: {identity}")
    gt = read_geff_arrays(store, split_name, dataset)
    scale = parse_scale(gt["attrs"])
    nodes = gt["nodes"]
    ids = [int(value) for value in nodes["node_id"]]
    node_t = {int(node_id): int(t) for node_id, t in zip(ids, nodes["t"], strict=True)}
    outgoing: Counter[int] = Counter()
    adjacent_edges = 0
    training_edges = 0
    sample_frames: set[int] = set()
    for source, target in np.asarray(gt["edges"], dtype=np.int64):
        source_id, target_id = int(source), int(target)
        if source_id not in node_t or target_id not in node_t:
            raise CellectContractError(f"GEFF edge references a missing node: {identity}")
        source_t, target_t = node_t[source_id], node_t[target_id]
        if target_t != source_t + 1:
            continue
        adjacent_edges += 1
        outgoing[source_id] += 1
        # Upstream CELLECT compares embeddings from (t,t+1) and (t+1,t+2).
        if source_t + 2 < int(image.shape[0]):
            training_edges += 1
            sample_frames.add(source_t)
    if not adjacent_edges or not sample_frames:
        raise CellectContractError(f"no usable adjacent temporal labels: {identity}")
    division_sources = sum(count > 1 for count in outgoing.values())
    return {
        "identity": identity,
        "image_shape_tzyx": [int(value) for value in image.shape],
        "dtype": str(image.dtype),
        "scale_zyx_um": [float(value) for value in scale],
        "node_count": len(ids),
        "adjacent_edge_count": adjacent_edges,
        "training_edge_count": training_edges,
        "division_source_count": division_sources,
        "sample_frames": sorted(sample_frames),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.archive, args.fit_manifest, args.split):
        if not path.is_file():
            raise CellectContractError(f"missing CELLECT index input: {path}")
    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise CellectContractError(
            f"raw archive byte count changed: {args.archive.stat().st_size} != {EXPECTED_ARCHIVE_BYTES}"
        )
    if args.output.exists():
        raise CellectContractError(f"refusing to overwrite index output: {args.output}")

    fit = load_json(args.fit_manifest)
    identities = [str(value) for value in fit.get("video_ids", ())]
    if len(identities) != EXPECTED_FIT_VIDEOS or len(set(identities)) != EXPECTED_FIT_VIDEOS:
        raise CellectContractError("fit manifest is not the exact 117-video corpus")
    expected_boundary = {
        "audit_used": False,
        "competition_submission_performed": False,
        "held_out_used": False,
        "test_used": False,
        "train_fit_only": True,
    }
    if fit.get("data_boundary") != expected_boundary:
        raise CellectContractError("fit117 manifest no longer proves a fit-only boundary")
    held = holdout_ids(load_json(args.split))
    if set(identities) & set(held):
        raise CellectContractError("fit117 overlaps exact CV57")
    train_ids, validation_ids = stratified_video_split(identities, args.validation_fraction)
    if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != set(identities):
        raise CellectContractError("internal video split is not a disjoint exact partition")

    records: list[dict[str, Any]] = []
    with ZipStore(args.archive, mode="r") as store:
        for identity in sorted(identities):
            records.append(video_record(store, identity))
    record_by_id = {row["identity"]: row for row in records}
    totals = {
        "node_count": sum(int(row["node_count"]) for row in records),
        "adjacent_edge_count": sum(int(row["adjacent_edge_count"]) for row in records),
        "training_edge_count": sum(int(row["training_edge_count"]) for row in records),
        "division_source_count": sum(int(row["division_source_count"]) for row in records),
        "frame_pair_sample_count": sum(len(row["sample_frames"]) for row in records),
    }
    if any(value <= 0 for value in totals.values()):
        raise CellectContractError(f"fit117 CELLECT index has empty totals: {totals}")
    payload = {
        "schema": SCHEMA,
        "status": "validated",
        "data_boundary": {
            "train_fit_only": True,
            "held_out_used_for_exclusion_only": True,
            "held_out_labels_or_pixels_read": False,
            "test_used": False,
            "kaggle_action_performed": False,
        },
        "inputs": {
            "archive": {"path": str(args.archive), "bytes": args.archive.stat().st_size},
            "fit_manifest": {
                "path": str(args.fit_manifest),
                "sha256": sha256_file(args.fit_manifest),
            },
            "split": {"path": str(args.split), "sha256": sha256_file(args.split)},
        },
        "internal_split": {
            "method": "whole-video SHA-256 order stratified by dataset prefix",
            "seed": SPLIT_SEED,
            "validation_fraction_requested": float(args.validation_fraction),
            "train_video_ids": train_ids,
            "validation_video_ids": validation_ids,
            "train_video_ids_sha256": canonical_sha(train_ids),
            "validation_video_ids_sha256": canonical_sha(validation_ids),
        },
        "totals": totals,
        "videos": record_by_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(json.dumps({
        "schema": payload["schema"],
        "status": payload["status"],
        "train_videos": len(payload["internal_split"]["train_video_ids"]),
        "validation_videos": len(payload["internal_split"]["validation_video_ids"]),
        "totals": payload["totals"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
