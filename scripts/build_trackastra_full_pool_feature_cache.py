#!/usr/bin/env python3
"""Build a resumable label-free Trackastra+APNCC full-pool feature cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
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
from celltrack_alpha import (  # noqa: E402
    appearance_swap,
    bh_edge,
    bh_model,
    point_context_swap,
)
from evaluate_trackastra_zero_wrfeat_signal import (  # noqa: E402
    _association,
    select_window_nodes,
    sha256,
)
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
)
from prepare_extended_radius_shell import _baseline  # noqa: E402
from trackastra_checkpoint_loader import load_mmap_meta  # noqa: E402


SCHEMA = "celltrack-alpha.trackastra-full-pool-feature-cache.v2"
VIDEO_SCHEMA = "celltrack-alpha.trackastra-full-pool-video-features.v2"
CHECKPOINT_SCHEMA = (
    "celltrack-alpha.trackastra-full-pool-feature-checkpoint.v2"
)
CONTRACT_SCHEMA = (
    "celltrack-alpha.trackastra-full-pool-feature-cache-contract.v2"
)
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/trackastra_full_pool_feature_cache_contract_v2.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "e060b284da623a373a83c8dd690985503af945399518f4d77eb517fddb8128ac"
)


class TrackastraFeatureCacheError(RuntimeError):
    """The sealed label-free feature cache failed closed."""


def _cache_identity(meta: Mapping[str, Any]) -> bh_model.CacheIdentity:
    try:
        model = meta["contract"]["model"]
        return bh_model.CacheIdentity(
            str(model["config_sha256"]),
            str(model["source_sha256"]),
            str(model["weight_sha256"]),
            int(model["pair_batch_size"]),
            int(model["view_batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackastraFeatureCacheError(
            "cache metadata lacks a valid model identity"
        ) from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackastraFeatureCacheError(
            f"cannot read {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise TrackastraFeatureCacheError(f"{label} must be a JSON object")
    return value


def _require(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise TrackastraFeatureCacheError(f"{label} identity mismatch: {path}")


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(path, expected_sha256, "contract")
    value = _load_json(path, "contract")
    scope = value.get("scope")
    features = value.get("features")
    checkpoint_loader = value.get("checkpoint_loader")
    parity_reference = (
        checkpoint_loader.get("parity_reference")
        if isinstance(checkpoint_loader, Mapping)
        else None
    )
    stages = value.get("stages")
    execution = value.get("execution")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or not all(
            isinstance(section, Mapping)
            for section in (
                scope,
                features,
                checkpoint_loader,
                parity_reference,
                stages,
                execution,
            )
        )
    ):
        raise TrackastraFeatureCacheError("contract is malformed")
    if (
        scope.get("fit_train105_only") is not True
        or scope.get("label_access") is not False
        or scope.get("training") is not False
        or any(
            scope.get(key) is not False
            for key in (
                "validation12_access",
                "cv57_access",
                "promotion_authorized",
                "kaggle_action",
                "gpu_action",
                "runpod_action",
            )
        )
        or int(features.get("base_feature_count", 0)) != 48
        or int(features.get("trackastra_feature_count", 0)) != 1
        or int(features.get("output_feature_count", 0)) != 49
        or int(features.get("max_tokens", 0)) != 1024
        or checkpoint_loader.get("kind")
        != (
            "stream-authenticated mmap weights plus meta-device "
            "construction and assign-in-place state restoration"
        )
        or checkpoint_loader.get("tensor_values_changed") is not False
        or checkpoint_loader.get("checkpoint_member_must_be_preextracted")
        is not True
        or parity_reference.get("identity") != "train:6bba_edf14583"
        or not isinstance(parity_reference.get("path"), str)
        or not isinstance(parity_reference.get("sha256"), str)
        or execution.get("atomic_video_publication") is not True
        or execution.get("resume_requires_exact_checkpoint") is not True
        or execution.get("two_stage_required") is not True
        or execution.get("model_stack_coexistence") is not False
        or not isinstance(stages.get("apncc_base"), Mapping)
        or not isinstance(stages.get("trackastra_enrichment"), Mapping)
    ):
        raise TrackastraFeatureCacheError("contract is not fail-closed")
    return value


def flatten_windows(
    windows: Sequence[point_context_swap.RuntimeProposalWindow],
    trackastra_margins: Sequence[Sequence[np.ndarray]],
) -> dict[str, Any]:
    """Flatten lossless proposal/conflict identities and 49-D row features."""

    if len(windows) != len(trackastra_margins):
        raise TrackastraFeatureCacheError(
            "window and Trackastra score coverage disagree"
        )
    challengers: list[tuple[int, int]] = []
    conflicts: list[tuple[int, int]] = []
    row_features: list[np.ndarray] = []
    proposal_offsets = [0]
    transition_offsets = [0]
    transition_target_t = []
    geometry_confidences: list[float] = []
    geometry_probabilities: list[float] = []
    for window, window_margins in zip(
        windows, trackastra_margins, strict=True
    ):
        window.validate()
        if len(window_margins) != len(window.challengers):
            raise TrackastraFeatureCacheError(
                "proposal score count differs from runtime window"
            )
        transition_target_t.append(int(window.target_t))
        for index, challenger in enumerate(window.challengers):
            base = np.asarray(window.row_features[index], dtype=np.float32)
            margins = np.asarray(window_margins[index], dtype=np.float32)
            probabilities = np.asarray(
                window.geometry_probabilities[index], dtype=np.float32
            )
            if (
                base.shape != (len(window.conflicts[index]), 48)
                or margins.shape != (len(window.conflicts[index]),)
                or probabilities.shape != (len(window.conflicts[index]),)
                or not np.isfinite(margins).all()
            ):
                raise TrackastraFeatureCacheError(
                    "row feature and conflict coverage disagree"
                )
            challengers.append(
                (int(challenger[0]), int(challenger[1]))
            )
            geometry_confidences.append(
                float(window.geometry_confidences[index])
            )
            for row, conflict in enumerate(window.conflicts[index]):
                conflicts.append((int(conflict[0]), int(conflict[1])))
                row_features.append(
                    np.concatenate(
                        (base[row], margins[row : row + 1])
                    ).astype(np.float32, copy=False)
                )
                geometry_probabilities.append(float(probabilities[row]))
            proposal_offsets.append(len(conflicts))
        transition_offsets.append(len(challengers))
    feature_array = (
        np.stack(row_features).astype(np.float32, copy=False)
        if row_features
        else np.empty((0, 49), dtype=np.float32)
    )
    result = {
        "challengers": np.asarray(challengers, dtype=np.int64).reshape(-1, 2),
        "conflicts": np.asarray(conflicts, dtype=np.int64).reshape(-1, 2),
        "row_features": feature_array,
        "proposal_offsets": np.asarray(proposal_offsets, dtype=np.int64),
        "transition_offsets": np.asarray(
            transition_offsets, dtype=np.int64
        ),
        "transition_target_t": np.asarray(
            transition_target_t, dtype=np.int64
        ),
        "geometry_confidences": np.asarray(
            geometry_confidences, dtype=np.float32
        ),
        "geometry_probabilities": np.asarray(
            geometry_probabilities, dtype=np.float32
        ),
    }
    proposal_count = len(challengers)
    row_count = len(conflicts)
    if (
        result["challengers"].shape != (proposal_count, 2)
        or result["conflicts"].shape != (row_count, 2)
        or feature_array.shape != (row_count, 49)
        or result["proposal_offsets"].shape != (proposal_count + 1,)
        or int(result["proposal_offsets"][-1]) != row_count
        or result["transition_offsets"].shape != (len(windows) + 1,)
        or int(result["transition_offsets"][-1]) != proposal_count
        or result["transition_target_t"].shape != (len(windows),)
        or result["geometry_confidences"].shape != (proposal_count,)
        or result["geometry_probabilities"].shape != (row_count,)
        or not np.isfinite(feature_array).all()
    ):
        raise TrackastraFeatureCacheError(
            "flattened feature cache is malformed"
        )
    return result


def flatten_base_windows(
    windows: Sequence[point_context_swap.RuntimeProposalWindow],
) -> dict[str, Any]:
    """Flatten the lossless 48-D APNCC stage without Trackastra."""

    zero_margins = tuple(
        tuple(
            np.zeros(len(conflicts), dtype=np.float32)
            for conflicts in window.conflicts
        )
        for window in windows
    )
    result = flatten_windows(windows, zero_margins)
    result["row_features"] = np.asarray(
        result["row_features"][:, :48], dtype=np.float32
    )
    if result["row_features"].shape != (len(result["conflicts"]), 48):
        raise TrackastraFeatureCacheError(
            "base feature cache is not exactly 48-D"
        )
    return result


def trackastra_margins_from_flattened(
    *,
    model: torch.nn.Module,
    detections: Sequence[bh_model.FrameDet],
    base: Mapping[str, Any],
    max_tokens: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Enrich a flattened APNCC cache without restoring V45 or raw images."""

    challengers = np.asarray(base.get("challengers"), dtype=np.int64)
    conflicts = np.asarray(base.get("conflicts"), dtype=np.int64)
    proposal_offsets = np.asarray(
        base.get("proposal_offsets"), dtype=np.int64
    )
    transition_offsets = np.asarray(
        base.get("transition_offsets"), dtype=np.int64
    )
    transition_target_t = np.asarray(
        base.get("transition_target_t"), dtype=np.int64
    )
    if (
        challengers.ndim != 2
        or challengers.shape[1:] != (2,)
        or conflicts.ndim != 2
        or conflicts.shape[1:] != (2,)
        or proposal_offsets.shape != (len(challengers) + 1,)
        or transition_offsets.shape != (len(transition_target_t) + 1,)
        or int(proposal_offsets[-1]) != len(conflicts)
        or int(transition_offsets[-1]) != len(challengers)
    ):
        raise TrackastraFeatureCacheError(
            "flattened APNCC identities are malformed"
        )
    by_node: dict[int, tuple[int, np.ndarray]] = {}
    frame_nodes: dict[int, tuple[int, ...]] = {}
    for frame in detections:
        ids = tuple(int(value) for value in frame.node_ids)
        frame_nodes[int(frame.frame_index)] = ids
        for node, coordinate in zip(
            ids, np.asarray(frame.coords_zyx), strict=True
        ):
            if node in by_node:
                raise TrackastraFeatureCacheError(
                    "cache repeats a detection identity"
                )
            by_node[node] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float32),
            )

    margins = np.empty(len(conflicts), dtype=np.float32)
    cropped = peak_tokens = 0
    for transition_index, target_t in enumerate(transition_target_t):
        proposal_start = int(transition_offsets[transition_index])
        proposal_stop = int(transition_offsets[transition_index + 1])
        required: set[int] = set()
        for proposal_index in range(proposal_start, proposal_stop):
            required.update(map(int, challengers[proposal_index]))
            row_start = int(proposal_offsets[proposal_index])
            row_stop = int(proposal_offsets[proposal_index + 1])
            required.update(map(int, conflicts[row_start:row_stop].ravel()))
        nodes, was_cropped = select_window_nodes(
            by_node=by_node,
            frame_nodes=frame_nodes,
            source_frame=int(target_t) - 1,
            required=required,
            window_frames=4,
            max_tokens=max_tokens,
        )
        if not required <= set(nodes):
            raise TrackastraFeatureCacheError(
                "Trackastra packing lost a proposal endpoint"
            )
        association, index_by_node = _association(model, nodes, by_node)
        peak_tokens = max(peak_tokens, len(nodes))
        cropped += int(was_cropped)
        for proposal_index in range(proposal_start, proposal_stop):
            challenger = challengers[proposal_index]
            challenger_score = association[
                index_by_node[int(challenger[0])],
                index_by_node[int(challenger[1])],
            ]
            row_start = int(proposal_offsets[proposal_index])
            row_stop = int(proposal_offsets[proposal_index + 1])
            for row_index in range(row_start, row_stop):
                conflict = conflicts[row_index]
                margins[row_index] = (
                    challenger_score
                    - association[
                        index_by_node[int(conflict[0])],
                        index_by_node[int(conflict[1])],
                    ]
                )
    if not np.isfinite(margins).all():
        raise TrackastraFeatureCacheError(
            "Trackastra enrichment produced non-finite margins"
        )
    return margins, {
        "inference_calls": len(transition_target_t),
        "cropped_calls": cropped,
        "peak_tokens": peak_tokens,
    }


def enrich_base_payload(
    base: Mapping[str, Any], margins: np.ndarray
) -> dict[str, Any]:
    """Append one aligned Trackastra margin to every 48-D conflict row."""

    features = np.asarray(base.get("row_features"), dtype=np.float32)
    margins = np.asarray(margins, dtype=np.float32)
    if (
        features.ndim != 2
        or features.shape[1] != 48
        or margins.shape != (features.shape[0],)
        or not np.isfinite(features).all()
        or not np.isfinite(margins).all()
    ):
        raise TrackastraFeatureCacheError(
            "base features and Trackastra margins do not align"
        )
    return {
        key: np.asarray(base[key]).copy()
        for key in (
            "challengers",
            "conflicts",
            "proposal_offsets",
            "transition_offsets",
            "transition_target_t",
            "geometry_confidences",
            "geometry_probabilities",
        )
    } | {
        "row_features": np.concatenate(
            (features, margins[:, None]), axis=1
        ).astype(np.float32, copy=False)
    }


def _trackastra_margins(
    *,
    model: torch.nn.Module,
    windows: Sequence[point_context_swap.RuntimeProposalWindow],
    cache: bh_model.SparseInferenceCache,
    max_tokens: int,
) -> tuple[list[list[np.ndarray]], dict[str, int]]:
    by_node: dict[int, tuple[int, np.ndarray]] = {}
    frame_nodes: dict[int, tuple[int, ...]] = {}
    for frame in cache.detections:
        ids = tuple(int(value) for value in frame.node_ids)
        frame_nodes[int(frame.frame_index)] = ids
        for node, coordinate in zip(
            ids, np.asarray(frame.coords_zyx), strict=True
        ):
            if node in by_node:
                raise TrackastraFeatureCacheError(
                    "cache repeats a detection identity"
                )
            by_node[node] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float32),
            )
    result: list[list[np.ndarray]] = []
    cropped = peak_tokens = 0
    for window in windows:
        required = {
            node
            for challenger, conflicts in zip(
                window.challengers, window.conflicts, strict=True
            )
            for pair in (challenger, *conflicts)
            for node in pair
        }
        nodes, was_cropped = select_window_nodes(
            by_node=by_node,
            frame_nodes=frame_nodes,
            source_frame=int(window.target_t) - 1,
            required=required,
            window_frames=4,
            max_tokens=max_tokens,
        )
        if not required <= set(nodes):
            raise TrackastraFeatureCacheError(
                "Trackastra packing lost a proposal endpoint"
            )
        association, index_by_node = _association(model, nodes, by_node)
        peak_tokens = max(peak_tokens, len(nodes))
        cropped += int(was_cropped)
        window_result = []
        for challenger, conflicts in zip(
            window.challengers, window.conflicts, strict=True
        ):
            challenger_score = association[
                index_by_node[int(challenger[0])],
                index_by_node[int(challenger[1])],
            ]
            window_result.append(
                np.asarray(
                    [
                        challenger_score
                        - association[
                            index_by_node[int(conflict[0])],
                            index_by_node[int(conflict[1])],
                        ]
                        for conflict in conflicts
                    ],
                    dtype=np.float32,
                )
            )
        result.append(window_result)
    return result, {
        "inference_calls": len(windows),
        "cropped_calls": cropped,
        "peak_tokens": peak_tokens,
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)


def assert_v1_parity(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Require exact v2 equivalence to the authenticated conventional output."""

    reference = contract["checkpoint_loader"].get("parity_reference")
    if not isinstance(reference, Mapping):
        raise TrackastraFeatureCacheError(
            "checkpoint-loader parity reference is missing"
        )
    path = ROOT / str(reference.get("path"))
    _require(path, str(reference.get("sha256")), "v1 parity reference")
    try:
        expected = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrackastraFeatureCacheError(
            "cannot restore v1 parity reference"
        ) from exc
    if (
        not isinstance(expected, Mapping)
        or payload.get("identity") != reference.get("identity")
        or expected.get("identity") != reference.get("identity")
        or payload.get("feature_names") != expected.get("feature_names")
        or payload.get("selected_raw_edges")
        != expected.get("selected_raw_edges")
        or payload.get("pool_counts") != expected.get("pool_counts")
        or payload.get("inference") != expected.get("inference")
    ):
        raise TrackastraFeatureCacheError(
            "v2 scalar/identity parity with v1 failed"
        )
    for key in (
        "challengers",
        "conflicts",
        "row_features",
        "proposal_offsets",
        "transition_offsets",
        "transition_target_t",
        "geometry_confidences",
        "geometry_probabilities",
    ):
        if not np.array_equal(
            np.asarray(payload.get(key)), np.asarray(expected.get(key))
        ):
            raise TrackastraFeatureCacheError(
                f"v2 numerical parity with v1 failed for {key}"
            )


def _checkpoint(
    *,
    path: Path,
    feature_path: Path,
    identity: str,
    contract_sha256: str,
    implementation_sha256: str,
    expected_proposals: int,
) -> dict[str, Any] | None:
    if not path.exists() and not feature_path.exists():
        return None
    if not path.is_file() or not feature_path.is_file():
        raise TrackastraFeatureCacheError(
            f"incomplete feature checkpoint for {identity}"
        )
    value = _load_json(path, f"{identity} feature checkpoint")
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("identity") != identity
        or value.get("contract_sha256") != contract_sha256
        or value.get("implementation_sha256") != implementation_sha256
        or int(value.get("proposal_count", -1)) != expected_proposals
        or value.get("feature_sha256") != sha256(feature_path)
    ):
        raise TrackastraFeatureCacheError(
            f"feature checkpoint identity drifted for {identity}"
        )
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = load_contract(args.contract, args.contract_sha256)
    if contract["execution"].get("two_stage_required") is True:
        raise TrackastraFeatureCacheError(
            "co-resident V45+Trackastra builder is forbidden by v2; run "
            "build_trackastra_apncc_base_cache.py, then "
            "enrich_trackastra_apncc_cache.py in a new process"
        )
    implementation_sha256 = sha256(Path(__file__))
    inputs = contract["inputs"]
    for label in (
        "appearance_contract",
        "fit_prepare",
        "full_pool_feasibility",
        "asset_audit",
    ):
        record = inputs[label]
        _require(
            ROOT / str(record["path"]), str(record["sha256"]), label
        )
    raw_archive = inputs.get("raw_archive")
    if not isinstance(raw_archive, Mapping):
        raise TrackastraFeatureCacheError(
            "contract raw archive identity is missing"
        )
    expected_archive = (ROOT / str(raw_archive["path"])).resolve()
    if (
        args.zip != expected_archive
        or not args.zip.is_file()
        or args.zip.stat().st_size != int(raw_archive["bytes"])
    ):
        raise TrackastraFeatureCacheError(
            f"raw archive identity mismatch: {args.zip}"
        )
    prepare = _load_json(
        ROOT / str(inputs["fit_prepare"]["path"]), "fit prepare"
    )
    feasibility = _load_json(
        ROOT / str(inputs["full_pool_feasibility"]["path"]),
        "full-pool feasibility",
    )
    scope = prepare.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise TrackastraFeatureCacheError("fit prepare scope drifted")
    eligible = tuple(
        str(identity)
        for identity in scope["train_video_ids"]
        if str(identity).startswith("train:6bba_")
    )
    if len(eligible) != 69:
        raise TrackastraFeatureCacheError("eligible video count drifted")
    parity_identity = contract["checkpoint_loader"].get(
        "parity_reference", {}
    ).get("identity")
    if not parity_identity or eligible[0] != parity_identity:
        raise TrackastraFeatureCacheError(
            "v1 parity video is not the first eligible identity"
        )
    prepare_rows = {
        str(row["identity"]): row for row in prepare["videos"]
    }
    feasibility_rows = {
        str(row["identity"]): row for row in feasibility["videos"]
    }
    if set(eligible) != set(feasibility_rows):
        raise TrackastraFeatureCacheError(
            "feasibility video coverage drifted"
        )
    appearance_contract, geometry_gate = appearance_swap.load_contract(
        ROOT / str(inputs["appearance_contract"]["path"]),
        expected_sha256=str(inputs["appearance_contract"]["sha256"]),
        root=ROOT,
    )
    model = load_mmap_meta(
        source_root=args.source_root,
        checkpoint_zip=args.checkpoint_zip,
        checkpoint_member=args.checkpoint_member,
        contract=contract,
    )
    base_model, bundle = bh_model.load_v45_model(
        args.support_root, device="cpu"
    )
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise TrackastraFeatureCacheError(
            "frozen edge scorer did not restore"
        )

    args.output.mkdir(parents=True, exist_ok=args.resume)
    videos_root = args.output / "videos"
    videos_root.mkdir(exist_ok=args.resume)
    files = []
    resumed = 0
    with ZipStore(args.zip, mode="r") as store:
        for ordinal, identity in enumerate(eligible, start=1):
            slug = identity.replace(":", "__", 1)
            feature_path = videos_root / f"{slug}.pt"
            checkpoint_path = videos_root / f"{slug}.json"
            expected_proposals = int(
                feasibility_rows[identity]["geometry_proposals"]
            )
            existing = (
                _checkpoint(
                    path=checkpoint_path,
                    feature_path=feature_path,
                    identity=identity,
                    contract_sha256=args.contract_sha256,
                    implementation_sha256=implementation_sha256,
                    expected_proposals=expected_proposals,
                )
                if args.resume
                else None
            )
            if existing is not None:
                files.append(existing)
                resumed += 1
                print(
                    f"[{ordinal}/69] {identity}: authenticated resume",
                    flush=True,
                )
                continue
            if feature_path.exists() or checkpoint_path.exists():
                raise TrackastraFeatureCacheError(
                    f"refusing to overwrite {identity} feature checkpoint"
                )
            meta = _load_json(
                args.cache / f"{slug}.json", f"{identity} cache metadata"
            )
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(meta),
            )
            selected, _probabilities, _no_link, graph = _baseline(
                cache, scorer
            )
            if int(graph["raw_selected_edges"]) != int(
                prepare_rows[identity]["graph"]["raw_selected_edges"]
            ):
                raise TrackastraFeatureCacheError(
                    f"{identity} selected graph identity drifted"
                )
            dataset = identity.split(":", 1)[1]
            image = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )["0"]
            windows, pool_counts = (
                point_context_swap.build_full_pool_windows(
                    cache,
                    scorer,
                    selected,
                    geometry_gate,
                    image,
                    identity=identity,
                    family=1,
                    geometry_threshold=float(
                        appearance_contract["geometry_gate"]["threshold"]
                    ),
                )
            )
            if (
                int(pool_counts["full_pool_proposals"])
                != expected_proposals
                or len(windows)
                != int(feasibility_rows[identity]["nonempty_transitions"])
            ):
                raise TrackastraFeatureCacheError(
                    f"{identity} full-pool counts drifted"
                )
            margins, inference = _trackastra_margins(
                model=model,
                windows=windows,
                cache=cache,
                max_tokens=int(contract["features"]["max_tokens"]),
            )
            flattened = flatten_windows(windows, margins)
            payload = {
                "schema": VIDEO_SCHEMA,
                "identity": identity,
                "contract_sha256": args.contract_sha256,
                "implementation_sha256": implementation_sha256,
                "feature_names": [
                    *[f"apncc_context_{index:02d}" for index in range(48)],
                    "trackastra_margin",
                ],
                "selected_raw_edges": len(selected),
                "pool_counts": pool_counts,
                "inference": inference,
                **flattened,
            }
            if any("label" in key.lower() for key in payload):
                raise TrackastraFeatureCacheError(
                    "label-like key escaped into feature payload"
                )
            if identity == parity_identity:
                assert_v1_parity(payload, contract)
            _atomic_torch_save(feature_path, payload)
            record = {
                "schema": CHECKPOINT_SCHEMA,
                "identity": identity,
                "contract_sha256": args.contract_sha256,
                "implementation_sha256": implementation_sha256,
                "feature_path": str(feature_path.relative_to(ROOT)),
                "feature_sha256": sha256(feature_path),
                "bytes": feature_path.stat().st_size,
                "proposal_count": len(flattened["challengers"]),
                "row_count": len(flattened["conflicts"]),
                "transition_count": len(windows),
                **inference,
            }
            _atomic_json(checkpoint_path, record)
            files.append(record)
            print(
                f"[{ordinal}/69] {identity}: "
                f"{record['proposal_count']} proposals / "
                f"{record['transition_count']} calls",
                flush=True,
            )
            del cache, windows, margins, flattened, payload
            gc.collect()
    if len(files) != 69:
        raise TrackastraFeatureCacheError(
            "feature cache video coverage is incomplete"
        )
    return {
        "schema": SCHEMA,
        "verdict": "PASS_LABEL_FREE_FEATURE_CACHE",
        "contract": {
            "path": str(args.contract.relative_to(ROOT)),
            "sha256": args.contract_sha256,
        },
        "implementation_sha256": implementation_sha256,
        "scope": contract["scope"],
        "feature_contract": contract["features"],
        "checkpoint_loader": contract["checkpoint_loader"],
        "aggregate": {
            "video_count": len(files),
            "proposal_count": sum(
                int(row["proposal_count"]) for row in files
            ),
            "row_count": sum(int(row["row_count"]) for row in files),
            "inference_calls": sum(
                int(row["inference_calls"]) for row in files
            ),
            "cropped_calls": sum(
                int(row["cropped_calls"]) for row in files
            ),
            "peak_tokens": max(int(row["peak_tokens"]) for row in files),
            "bytes": sum(int(row["bytes"]) for row in files),
            "resumed_video_count": resumed,
        },
        "execution": {
            "device": "cpu",
            "training": False,
            "label_access": False,
            "elapsed_seconds": time.perf_counter() - started,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
        "files": files,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-zip", type=Path, required=True)
    parser.add_argument("--checkpoint-member", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument(
        "--support-root", type=Path, default=DEFAULT_SUPPORT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "contract",
        "source_root",
        "checkpoint_zip",
        "checkpoint_member",
        "cache",
        "zip",
        "edge_artifact",
        "edge_auxiliary",
        "support_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    report_path = args.output / "report.json"
    if report_path.exists():
        raise TrackastraFeatureCacheError(
            f"refusing to overwrite completed report: {report_path}"
        )
    report = build(args)
    _atomic_json(report_path, report)
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
