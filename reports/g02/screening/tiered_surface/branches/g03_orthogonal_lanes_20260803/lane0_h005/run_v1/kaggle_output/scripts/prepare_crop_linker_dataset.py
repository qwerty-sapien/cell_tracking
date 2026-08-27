#!/usr/bin/env python3
"""Materialize the hash-pinned fit-only crop-linker dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
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
from audit_compact_point_context_data import _group_labeled  # noqa: E402
from audit_crop_linker_data import (  # noqa: E402
    CANDIDATE_SHA256,
    DEFAULT_CACHE,
    PREPARE_SHA256,
    _gt_graph,
    _prediction_graph,
    exact_pair_label,
)
from celltrack_alpha import appearance_swap, bh_model  # noqa: E402
from celltrack_alpha.crop_linker import normalize_crops  # noqa: E402
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
from fit_compact_point_context_pilot import proposal_features  # noqa: E402


SCHEMA = "celltrack-alpha.crop-linker-dataset.v1"
CONTRACT_SHA256 = (
    "0e9c7a9d3dbcd49823ff198bfa77737ec9d07707513872813639fde24d8f6e6d"
)
AUDIT_SHA256 = (
    "b994c48b6696546c36b6c351c61d4829f1a55d897b301d2ed657b2727880f4e7"
)


class CropDatasetError(RuntimeError):
    """The cached crop dataset violated its frozen label/split contract."""


def _pair_table(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labeled_x = np.asarray(payload["labeled_features"], dtype=np.float32)
    labeled_y = np.asarray(payload["labeled_labels"], dtype=np.int8)
    labeled_pairs_raw = np.asarray(
        payload["labeled_challenger_pairs"], dtype=np.int64
    ).reshape(-1, 2)
    labeled_pairs, proposal_labels, labeled_groups = _group_labeled(
        labeled_pairs_raw, labeled_y
    )
    labeled_features = proposal_features(labeled_x, labeled_groups)
    labeled_roles = np.where(
        proposal_labels == 1, 1, 0
    ).astype(np.int8)

    background_x = np.asarray(
        payload["background_features"], dtype=np.float32
    )
    background_pairs = np.asarray(
        payload["background_challenger_pairs"], dtype=np.int64
    ).reshape(-1, 2)
    background_groups = payload["background_groups"]
    background_features = proposal_features(
        background_x, background_groups
    )
    labeled_set = {
        (int(pair[0]), int(pair[1])) for pair in labeled_pairs
    }
    keep = np.asarray(
        [
            (int(pair[0]), int(pair[1])) not in labeled_set
            for pair in background_pairs
        ],
        dtype=bool,
    )
    pairs = np.concatenate((labeled_pairs, background_pairs[keep]))
    features = np.concatenate(
        (labeled_features, background_features[keep])
    )
    # 1=prepared positive, 0=prepared harmful, -1=prepared background.
    roles = np.concatenate(
        (
            labeled_roles,
            np.full(int(keep.sum()), -1, dtype=np.int8),
        )
    )
    if (
        pairs.shape != (len(features), 2)
        or roles.shape != (len(features),)
        or features.shape[1:] != (48,)
    ):
        raise CropDatasetError("prepared pair table arrays disagree")
    return pairs, features, roles


def _normalized_patch(
    frame: np.ndarray, coord: np.ndarray
) -> np.ndarray:
    patch = appearance_swap.extract_patch(frame, coord)
    tensor = torch.from_numpy(
        np.asarray(patch, dtype=np.float32)[None, None]
    )
    normalized = normalize_crops(tensor)[0, 0].numpy()
    if (
        normalized.shape != appearance_swap.PATCH_SHAPE
        or not np.isfinite(normalized).all()
    ):
        raise CropDatasetError("normalized crop is malformed")
    return normalized.astype(np.float16)


def _sha_size(path: Path) -> tuple[str, int]:
    return sweep._sha256(path), path.stat().st_size


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CropDatasetError(f"refusing to overwrite output: {args.output}")
    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") == "1":
        raise CropDatasetError("holdout access is forbidden")
    for path, expected, name in (
        (args.candidate, args.candidate_sha256, "candidate"),
        (args.prepare, args.prepare_sha256, "prepare"),
        (args.audit, args.audit_sha256, "audit"),
        (args.contract, args.contract_sha256, "contract"),
    ):
        if sweep._sha256(path) != expected:
            raise CropDatasetError(f"{name} SHA-256 mismatch")
    gate = validate_candidate(
        load_json_object(args.candidate), load_json_object(args.policy)
    )
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    audit = json.loads(args.audit.read_text(encoding="ascii"))
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    prepare = json.loads(args.prepare.read_text(encoding="ascii"))
    if (
        audit.get("schema") != "celltrack-alpha.crop-linker-data-audit.v1"
        or audit.get("status") != "PASS"
        or audit.get("scope", {}).get("cv57_overlap") != []
        or contract.get("schema")
        != "celltrack-alpha.crop-linker-pilot-contract.v1"
        or contract.get("data_audit", {}).get("sha256")
        != args.audit_sha256
        or prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
    ):
        raise CropDatasetError("input audit/contract/prepare is inadmissible")
    partition_by_id = {
        identity: role
        for role, key in (
            ("fit_train", "fit_train_video_ids"),
            ("calibration", "calibration_video_ids"),
            ("validation", "validation_video_ids"),
        )
        for identity in audit["scope"][key]
    }
    if len(partition_by_id) != 117:
        raise CropDatasetError("audit split does not cover fit117")
    by_id = {str(row["identity"]): row for row in prepare["videos"]}
    if set(by_id) != set(partition_by_id):
        raise CropDatasetError("prepared identities drifted from audit split")

    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if not set(by_id).issubset(cache_paths) or not set(by_id).issubset(
        meta_paths
    ):
        raise CropDatasetError("fit117 cache coverage is incomplete")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=args.output.name + ".tmp.",
            dir=args.output.parent,
        )
    )
    example_dir = temporary / "examples"
    example_dir.mkdir()
    totals: dict[str, int] = defaultdict(int)
    files = []
    videos = []
    started = time.perf_counter()
    try:
        with ZipStore(args.zip, mode="r") as store:
            for index, identity in enumerate(
                sorted(partition_by_id), start=1
            ):
                dataset = identity.split(":", 1)[1]
                raw = by_id[identity]
                payload = torch.load(
                    ROOT / str(raw["path"]),
                    map_location="cpu",
                    weights_only=False,
                )
                pairs, context, prepared_roles = _pair_table(payload)
                meta = sweep._load(
                    meta_paths[identity],
                    f"cache metadata for {identity}",
                )
                cache = bh_model.load_sparse_cache(
                    cache_paths[identity],
                    expected_identity=sweep._cache_identity(meta),
                )
                pred_graph, positions = _prediction_graph(dataset, cache)
                gt = scoreall.read_geff_arrays(store, "train", dataset)
                truth_graph, gt_edges = _gt_graph(dataset, gt)
                mapping, _match = compare.match_nodes(
                    truth_graph, pred_graph, max_distance_um=7.0
                )
                labels = np.full(len(pairs), -1, dtype=np.int8)
                for row_index, pair in enumerate(pairs):
                    label = exact_pair_label(
                        (int(pair[0]), int(pair[1])),
                        mapping,
                        gt_edges,
                    )
                    if label == "positive":
                        labels[row_index] = 1
                    elif label == "guaranteed_negative":
                        labels[row_index] = 0
                keep = labels >= 0
                kept_pairs = pairs[keep]
                kept_context = context[keep]
                kept_labels = labels[keep]
                kept_roles = prepared_roles[keep]
                needed: dict[int, set[int]] = defaultdict(set)
                for source_id, target_id in kept_pairs:
                    source_id, target_id = int(source_id), int(target_id)
                    for node_id in (source_id, target_id):
                        if node_id not in positions:
                            raise CropDatasetError(
                                f"proposal node is absent for {identity}"
                            )
                        needed[positions[node_id][0]].add(node_id)
                group = zarr.open_group(
                    store=store,
                    path=f"train/{dataset}.zarr",
                    mode="r",
                )
                image = group["0"]
                crops: dict[int, np.ndarray] = {}
                for frame_index in sorted(needed):
                    frame = np.asarray(image[frame_index])
                    for node_id in sorted(needed[frame_index]):
                        crops[node_id] = _normalized_patch(
                            frame, positions[node_id][1]
                        )
                source_crops = np.stack(
                    [crops[int(pair[0])] for pair in kept_pairs]
                ) if len(kept_pairs) else np.empty(
                    (0, *appearance_swap.PATCH_SHAPE), dtype=np.float16
                )
                target_crops = np.stack(
                    [crops[int(pair[1])] for pair in kept_pairs]
                ) if len(kept_pairs) else np.empty(
                    (0, *appearance_swap.PATCH_SHAPE), dtype=np.float16
                )
                artifact = example_dir / (
                    identity.replace(":", "__", 1) + ".pt"
                )
                torch.save(
                    {
                        "schema": SCHEMA,
                        "identity": identity,
                        "partition": partition_by_id[identity],
                        "family": dataset.split("_", 1)[0],
                        "pairs": kept_pairs,
                        "source_crops": source_crops,
                        "target_crops": target_crops,
                        "context": kept_context.astype(
                            np.float32, copy=False
                        ),
                        "labels": kept_labels,
                        "prepared_roles": kept_roles,
                        "masked_count": int(np.sum(~keep)),
                    },
                    artifact,
                )
                file_hash, file_bytes = _sha_size(artifact)
                positive = int(np.sum(kept_labels == 1))
                negative = int(np.sum(kept_labels == 0))
                masked = int(np.sum(~keep))
                totals["positive"] += positive
                totals["guaranteed_negative"] += negative
                totals["uncertain_masked"] += masked
                totals["saved_examples"] += len(kept_labels)
                totals["bytes"] += file_bytes
                files.append(
                    {
                        "identity": identity,
                        "path": str(
                            args.output
                            / "examples"
                            / artifact.name
                        ),
                        "sha256": file_hash,
                        "bytes": file_bytes,
                    }
                )
                videos.append(
                    {
                        "identity": identity,
                        "partition": partition_by_id[identity],
                        "positive": positive,
                        "guaranteed_negative": negative,
                        "uncertain_masked": masked,
                    }
                )
                print(
                    f"crop dataset {index:03d}/{len(partition_by_id)} "
                    f"{identity} saved={len(kept_labels)} masked={masked}",
                    flush=True,
                )

        expected_totals = audit["totals"]
        if any(
            int(totals[key]) != int(expected_totals[key])
            for key in (
                "positive",
                "guaranteed_negative",
                "uncertain_masked",
            )
        ):
            raise CropDatasetError(
                "materialized labels do not reconcile with the audit"
            )
        result = {
            "schema": SCHEMA,
            "status": "PASS",
            "scope": {
                "fit_only": True,
                "partitions": {
                    role: sorted(
                        identity
                        for identity, observed in partition_by_id.items()
                        if observed == role
                    )
                    for role in ("fit_train", "calibration", "validation")
                },
                "holdout_access": False,
                "audit_access": False,
            },
            "label_contract": contract["labels"],
            "crop_contract": contract["crop"],
            "totals": dict(totals),
            "files": files,
            "videos": videos,
            "provenance": {
                "candidate": str(args.candidate),
                "candidate_sha256": args.candidate_sha256,
                "candidate_gate": gate,
                "contract": str(args.contract),
                "contract_sha256": args.contract_sha256,
                "audit": str(args.audit),
                "audit_sha256": args.audit_sha256,
                "prepare": str(args.prepare),
                "prepare_sha256": args.prepare_sha256,
                "zip": str(args.zip),
                "zip_sha256": sweep._sha256(args.zip),
                "scorer_parity": str(args.scorer_parity),
                "scorer_parity_sha256": sweep._sha256(
                    args.scorer_parity
                ),
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
        run_bh.atomic_create_json(temporary / "report.json", result)
        os.replace(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--candidate",
        type=Path,
        default=(
            ROOT
            / "reports/g02/model_candidate_calibrated_3d_crop_linker_v1.json"
        ),
    )
    value.add_argument("--candidate-sha256", default=CANDIDATE_SHA256)
    value.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "reports/g02/compute_governance_v1.json",
    )
    value.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_pilot_contract_v1.json",
    )
    value.add_argument("--contract-sha256", default=CONTRACT_SHA256)
    value.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_data_audit_v1.json",
    )
    value.add_argument("--audit-sha256", default=AUDIT_SHA256)
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
    value.add_argument(
        "--scorer-parity", type=Path, default=sweep.DEFAULT_PARITY
    )
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_dataset_v1",
    )
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "totals": report["totals"],
                "execution": report["execution"],
            },
            sort_keys=True,
        )
    )
