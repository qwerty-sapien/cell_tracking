#!/usr/bin/env python3
"""Materialize authentic time-reversed Trackastra evidence on frozen 44b6 rows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
import torch


MODEL_SHA = "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
ARCHIVE_SHA = "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e"
CONTRACT_SHA = "4da06d487ba1787f91467a01862e4f7471daa16244674f5171960774da06334b"
ENDPOINT_CONTRACT_SHA = "4eaf9d8f13eb0121d804be4252e6b53667ae65f7955c3eadf858d257804fd294"
FORWARD_REPORT_SHA = "d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf"
SEED = 20260725


class ReverseSupplierError(RuntimeError):
    """Authentic reverse evidence failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReverseSupplierError(f"JSON object required: {path}")
    return value


def safe_load_forward(path: Path) -> dict[str, Any]:
    """Load hash-pinned NumPy/Tensor containers without enabling arbitrary pickle."""

    allowed = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
        np.dtypes.Int32DType,
        np.dtypes.Float32DType,
        np.dtypes.BoolDType,
        np.dtypes.StrDType,
        np.dtypes.ObjectDType,
    ]
    with torch.serialization.safe_globals(allowed):
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ReverseSupplierError("forward payload is not a dictionary")
    return value


def cache_identity(meta: Mapping[str, Any], bh_model: Any) -> Any:
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
        raise ReverseSupplierError("cache metadata lacks model identity") from exc


def select_window_nodes(
    *,
    by_node: Mapping[int, tuple[int, np.ndarray]],
    frame_nodes: Mapping[int, Sequence[int]],
    source_frame: int,
    required: set[int],
    window_frames: int = 4,
    max_tokens: int = 1024,
) -> tuple[tuple[int, ...], bool]:
    """Exact frozen forward packing rule."""

    all_frames = sorted(frame_nodes)
    if not all_frames:
        raise ReverseSupplierError("cache has no detection frames")
    min_frame, max_frame = all_frames[0], all_frames[-1]
    start = min(max(source_frame - 1, min_frame), max(min_frame, max_frame - 3))
    allowed_frames = tuple(
        frame
        for frame in range(start, start + window_frames)
        if frame in frame_nodes
    )
    candidates = {
        int(node) for frame in allowed_frames for node in frame_nodes[frame]
    }
    if not required <= candidates:
        raise ReverseSupplierError("proposal endpoint is outside its frozen window")
    cropped = len(candidates) > max_tokens
    if cropped:
        if len(required) > max_tokens:
            raise ReverseSupplierError("required endpoints exceed token cap")
        centroid = np.mean(
            [np.asarray(by_node[node][1], dtype=np.float64) for node in required],
            axis=0,
        )
        optional = sorted(
            candidates - required,
            key=lambda node: (
                float(
                    np.sum(
                        (np.asarray(by_node[node][1], dtype=np.float64) - centroid)
                        ** 2
                    )
                ),
                int(by_node[node][0]),
                int(node),
            ),
        )
        selected = set(required)
        selected.update(optional[: max_tokens - len(required)])
    else:
        selected = candidates
    ordered = tuple(
        sorted(selected, key=lambda node: (int(by_node[node][0]), int(node)))
    )
    return ordered, cropped


def reverse_association(
    model: torch.nn.Module,
    nodes: Sequence[int],
    by_node: Mapping[int, tuple[int, np.ndarray]],
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    original = np.stack(
        [
            np.concatenate(
                (
                    np.asarray((by_node[node][0],), dtype=np.float32),
                    np.asarray(by_node[node][1], dtype=np.float32),
                )
            )
            for node in nodes
        ]
    )
    reversed_coords = original.copy()
    reversed_coords[:, 0] = (
        float(original[:, 0].min())
        + float(original[:, 0].max())
        - original[:, 0]
    )
    timepoints = torch.from_numpy(reversed_coords[:, 0].astype(np.int64))
    coords = torch.from_numpy(reversed_coords).unsqueeze(0)
    features = torch.zeros((1, len(nodes), 12), dtype=torch.float32)
    with torch.inference_mode():
        logits = model(coords, features=features)
        association = model.normalize_output(
            logits, timepoints.unsqueeze(0), coords
        )[0]
    value = association.detach().cpu().numpy().astype(np.float64)
    if value.shape != (len(nodes), len(nodes)) or not np.isfinite(value).all():
        raise ReverseSupplierError("reverse association matrix is malformed")
    return (
        value,
        {int(node): index for index, node in enumerate(nodes)},
        original[:, 0].astype(np.int64),
    )


def returned_score(
    association: np.ndarray, index_by_node: Mapping[int, int], pair: Sequence[int]
) -> float:
    """Return score for original u->v from the genuine reverse v->u entry."""

    source, target = map(int, pair)
    return float(association[index_by_node[target], index_by_node[source]])


def returned_null(
    association: np.ndarray,
    index_by_node: Mapping[int, int],
    original_times: np.ndarray,
    *,
    source: int,
    target_t: int,
) -> float:
    source_index = index_by_node[int(source)]
    target_indices = np.flatnonzero(original_times == int(target_t))
    raw = 1.0 - float(association[target_indices, source_index].sum())
    if not -1e-6 <= raw <= 1.0 + 1e-6:
        raise ReverseSupplierError(f"returned no-link mass outside guard: {raw}")
    return float(np.clip(raw, 0.0, 1.0))


def validate_forward_payload(base: Mapping[str, Any], identity: str) -> dict[str, np.ndarray]:
    arrays = {
        name: np.asarray(base.get(name))
        for name in (
            "challengers",
            "conflicts",
            "proposal_offsets",
            "transition_offsets",
            "transition_target_t",
            "row_features",
        )
    }
    challengers = np.asarray(arrays["challengers"], dtype=np.int64)
    conflicts = np.asarray(arrays["conflicts"], dtype=np.int64)
    proposal_offsets = np.asarray(arrays["proposal_offsets"], dtype=np.int64)
    transition_offsets = np.asarray(arrays["transition_offsets"], dtype=np.int64)
    targets = np.asarray(arrays["transition_target_t"], dtype=np.int64)
    row_features = np.asarray(arrays["row_features"], dtype=np.float32)
    if (
        base.get("identity") != identity
        or challengers.shape != (len(proposal_offsets) - 1, 2)
        or conflicts.shape != (int(proposal_offsets[-1]), 2)
        or transition_offsets.shape != (len(targets) + 1,)
        or int(transition_offsets[-1]) != len(challengers)
        or row_features.shape != (len(conflicts), 49)
        or not np.isfinite(row_features).all()
        or np.any(np.diff(proposal_offsets) < 0)
        or np.any(np.diff(transition_offsets) < 0)
    ):
        raise ReverseSupplierError(f"malformed frozen forward payload: {identity}")
    return {
        "challengers": challengers,
        "conflicts": conflicts,
        "proposal_offsets": proposal_offsets,
        "transition_offsets": transition_offsets,
        "transition_target_t": targets,
        "forward_margin": np.asarray(row_features[:, -1], dtype=np.float32),
    }


def materialize_video(
    *,
    model: torch.nn.Module,
    detections: Sequence[Any],
    base: Mapping[str, Any],
    identity: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    data = validate_forward_payload(base, identity)
    challengers = data["challengers"]
    conflicts = data["conflicts"]
    proposal_offsets = data["proposal_offsets"]
    transition_offsets = data["transition_offsets"]
    targets = data["transition_target_t"]
    row_count = len(conflicts)

    by_node: dict[int, tuple[int, np.ndarray]] = {}
    frame_nodes: dict[int, tuple[int, ...]] = {}
    for frame in detections:
        ids = tuple(int(value) for value in frame.node_ids)
        frame_nodes[int(frame.frame_index)] = ids
        for node, coordinate in zip(ids, np.asarray(frame.coords_zyx), strict=True):
            if node in by_node:
                raise ReverseSupplierError(f"{identity} repeats node {node}")
            by_node[node] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float32),
            )

    challenger_score = np.empty(row_count, dtype=np.float32)
    conflict_score = np.empty(row_count, dtype=np.float32)
    challenger_null = np.empty(row_count, dtype=np.float32)
    conflict_null = np.empty(row_count, dtype=np.float32)
    peak_tokens = cropped = 0
    for transition_index, target_t in enumerate(targets):
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
        )
        if not required <= set(nodes):
            raise ReverseSupplierError("packing lost a required endpoint")
        association, index_by_node, original_times = reverse_association(
            model, nodes, by_node
        )
        cropped += int(was_cropped)
        peak_tokens = max(peak_tokens, len(nodes))
        for proposal_index in range(proposal_start, proposal_stop):
            challenger = challengers[proposal_index]
            if (
                by_node[int(challenger[0])][0] != int(target_t) - 1
                or by_node[int(challenger[1])][0] != int(target_t)
            ):
                raise ReverseSupplierError("challenger is not adjacent at target")
            c_score = returned_score(association, index_by_node, challenger)
            c_null = returned_null(
                association,
                index_by_node,
                original_times,
                source=int(challenger[0]),
                target_t=int(target_t),
            )
            row_start = int(proposal_offsets[proposal_index])
            row_stop = int(proposal_offsets[proposal_index + 1])
            for row_index in range(row_start, row_stop):
                conflict = conflicts[row_index]
                if (
                    by_node[int(conflict[0])][0] != int(target_t) - 1
                    or by_node[int(conflict[1])][0] != int(target_t)
                ):
                    raise ReverseSupplierError("conflict is not adjacent at target")
                challenger_score[row_index] = c_score
                challenger_null[row_index] = c_null
                conflict_score[row_index] = returned_score(
                    association, index_by_node, conflict
                )
                conflict_null[row_index] = returned_null(
                    association,
                    index_by_node,
                    original_times,
                    source=int(conflict[0]),
                    target_t=int(target_t),
                )

    reverse_margin = np.asarray(challenger_score - conflict_score, dtype=np.float32)
    forward_margin = data["forward_margin"]
    output = {
        "challengers": challengers,
        "conflicts": conflicts,
        "proposal_offsets": proposal_offsets,
        "transition_offsets": transition_offsets,
        "transition_target_t": targets,
        "forward_margin": forward_margin,
        "reverse_challenger_score": challenger_score,
        "reverse_conflict_score": conflict_score,
        "reverse_margin": reverse_margin,
        "reverse_challenger_null": challenger_null,
        "reverse_conflict_null": conflict_null,
        "cycle_min_margin": np.asarray(
            np.minimum(forward_margin, reverse_margin), dtype=np.float32
        ),
        "cycle_abs_disagreement": np.asarray(
            np.abs(forward_margin - reverse_margin), dtype=np.float32
        ),
        "cycle_sign_agreement": np.asarray(
            (forward_margin >= 0) == (reverse_margin >= 0), dtype=np.uint8
        ),
    }
    if any(
        value.shape[0] != row_count
        for key, value in output.items()
        if key
        not in {
            "challengers",
            "proposal_offsets",
            "transition_offsets",
            "transition_target_t",
        }
    ) or any(
        not np.isfinite(value).all()
        for value in output.values()
        if np.issubdtype(value.dtype, np.floating)
    ):
        raise ReverseSupplierError("row coverage or finiteness failure")
    return output, {
        "inference_calls": len(targets),
        "cropped_calls": cropped,
        "peak_tokens": peak_tokens,
        "proposal_count": len(challengers),
        "row_count": row_count,
    }


def packing_stats(
    *,
    detections: Sequence[Any],
    base: Mapping[str, Any],
    identity: str,
) -> dict[str, int]:
    """Recompute non-model packing counters for authenticated resume files."""

    data = validate_forward_payload(base, identity)
    by_node: dict[int, tuple[int, np.ndarray]] = {}
    frame_nodes: dict[int, tuple[int, ...]] = {}
    for frame in detections:
        ids = tuple(int(value) for value in frame.node_ids)
        frame_nodes[int(frame.frame_index)] = ids
        for node, coordinate in zip(ids, np.asarray(frame.coords_zyx), strict=True):
            if node in by_node:
                raise ReverseSupplierError(f"{identity} repeats node {node}")
            by_node[node] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float32),
            )
    cropped = peak_tokens = 0
    for transition_index, target_t in enumerate(data["transition_target_t"]):
        proposal_start = int(data["transition_offsets"][transition_index])
        proposal_stop = int(data["transition_offsets"][transition_index + 1])
        required: set[int] = set()
        for proposal_index in range(proposal_start, proposal_stop):
            required.update(map(int, data["challengers"][proposal_index]))
            start = int(data["proposal_offsets"][proposal_index])
            stop = int(data["proposal_offsets"][proposal_index + 1])
            required.update(map(int, data["conflicts"][start:stop].ravel()))
        nodes, was_cropped = select_window_nodes(
            by_node=by_node,
            frame_nodes=frame_nodes,
            source_frame=int(target_t) - 1,
            required=required,
        )
        cropped += int(was_cropped)
        peak_tokens = max(peak_tokens, len(nodes))
    return {
        "inference_calls": len(data["transition_target_t"]),
        "cropped_calls": cropped,
        "peak_tokens": peak_tokens,
        "proposal_count": len(data["challengers"]),
        "row_count": len(data["conflicts"]),
    }


def authenticate_resume_npz(
    path: Path, base: Mapping[str, Any], identity: str
) -> dict[str, np.ndarray]:
    data = validate_forward_payload(base, identity)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    expected_keys = {
        "challengers",
        "conflicts",
        "proposal_offsets",
        "transition_offsets",
        "transition_target_t",
        "forward_margin",
        "reverse_challenger_score",
        "reverse_conflict_score",
        "reverse_margin",
        "reverse_challenger_null",
        "reverse_conflict_null",
        "cycle_min_margin",
        "cycle_abs_disagreement",
        "cycle_sign_agreement",
    }
    if set(arrays) != expected_keys:
        raise ReverseSupplierError(f"resume NPZ schema mismatch: {identity}")
    for key in (
        "challengers",
        "conflicts",
        "proposal_offsets",
        "transition_offsets",
        "transition_target_t",
        "forward_margin",
    ):
        if not np.array_equal(arrays[key], data[key]):
            raise ReverseSupplierError(f"resume NPZ join drift: {identity} {key}")
    row_count = len(data["conflicts"])
    if any(
        arrays[key].shape != (row_count,)
        for key in expected_keys
        - {
            "challengers",
            "conflicts",
            "proposal_offsets",
            "transition_offsets",
            "transition_target_t",
        }
    ) or any(
        not np.isfinite(value).all()
        for value in arrays.values()
        if np.issubdtype(value.dtype, np.floating)
    ):
        raise ReverseSupplierError(f"resume NPZ coverage failure: {identity}")
    return arrays


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write compressed NPZ bytes with fixed members, order, metadata and timestamp."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--checkpoint-member", type=Path, required=True)
    parser.add_argument("--option-contract", type=Path, required=True)
    parser.add_argument("--endpoint-contract", type=Path, required=True)
    parser.add_argument("--forward-report", type=Path, required=True)
    parser.add_argument("--forward-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prior-guard", type=Path)
    parser.add_argument("--prior-guard-sha")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.resume and (
        args.prior_guard is None
        or args.prior_guard_sha is None
        or sha256(args.prior_guard) != args.prior_guard_sha
    ):
        raise SystemExit("resume requires exact prior resource-guard receipt")
    pins = {
        args.checkpoint_archive: ARCHIVE_SHA,
        args.checkpoint_member: MODEL_SHA,
        args.option_contract: CONTRACT_SHA,
        args.endpoint_contract: ENDPOINT_CONTRACT_SHA,
        args.forward_report: FORWARD_REPORT_SHA,
    }
    for path, expected in pins.items():
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"pinned input mismatch: {path}")

    scripts = args.source_bundle / "scripts"
    source = args.source_bundle / "src"
    for location in (scripts, source):
        sys.path.insert(0, str(location))
    from celltrack_alpha import bh_model
    from trackastra_checkpoint_loader import load_mmap_meta

    contract = load_json(args.option_contract)
    report = load_json(args.forward_report)
    eligible = sorted(
        (
            row
            for row in report["files"]
            if str(row["identity"]).startswith("train:44b6_")
        ),
        key=lambda row: str(row["identity"]),
    )
    if len(eligible) != 36:
        raise SystemExit("frozen forward report does not contain 36 44b6 videos")

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    model = load_mmap_meta(
        source_root=args.source_root,
        checkpoint_zip=args.checkpoint_archive,
        checkpoint_member=args.checkpoint_member,
        contract=contract,
    )
    args.output.mkdir(parents=True, exist_ok=args.resume)
    videos_root = args.output / "videos"
    videos_root.mkdir(exist_ok=args.resume)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    witness_arrays: dict[str, np.ndarray] | None = None
    repeat: dict[str, Any] | None = None
    if args.resume:
        guard = load_json(args.prior_guard)
        first_identity = str(eligible[0]["identity"])
        first_stats = f"[1/36] {first_identity} "
        if (
            guard.get("status") != "FAIL"
            or guard.get("forced_stop_on_swap_growth") is not True
            or float(guard.get("swap_growth_mib", 0.0)) <= 0.0
            or first_stats not in str(guard.get("stdout", ""))
        ):
            raise SystemExit("prior guard cannot attest the completed repeat witness")
        repeat = {
            "identity": first_identity,
            "arrays": "all arrays compared by np.array_equal before first progress line",
            "all_array_equal": True,
            "inference_calls": int(eligible[0]["transition_count"]),
            "receipt": {
                "path": str(args.prior_guard),
                "sha256": args.prior_guard_sha,
                "ordering_contract": "first progress line is emitted only after repeat passes",
            },
        }
    for ordinal, row in enumerate(eligible, 1):
        identity = str(row["identity"])
        slug = identity.replace(":", "__", 1)
        forward_path = args.forward_root / f"{slug}.pt"
        if sha256(forward_path) != row["feature_sha256"]:
            raise SystemExit(f"forward hash mismatch: {identity}")
        base = safe_load_forward(forward_path)
        meta_path = args.cache_root / f"{slug}.json"
        cache_path = args.cache_root / f"{slug}.bhc"
        meta = load_json(meta_path)
        cache = bh_model.load_sparse_cache(
            cache_path, expected_identity=cache_identity(meta, bh_model)
        )
        destination = videos_root / f"{slug}.npz"
        resumed = destination.is_file()
        if resumed:
            if not args.resume:
                raise SystemExit(f"refusing existing output: {destination}")
            arrays = authenticate_resume_npz(destination, base, identity)
            stats = packing_stats(
                detections=cache.detections, base=base, identity=identity
            )
        else:
            arrays, stats = materialize_video(
                model=model,
                detections=cache.detections,
                base=base,
                identity=identity,
            )
            write_deterministic_npz(destination, arrays)
        record = {
            "identity": identity,
            "path": f"videos/{destination.name}",
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "forward_path": str(forward_path),
            "forward_sha256": row["feature_sha256"],
            "cache_path": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "cache_meta_path": str(meta_path),
            "cache_meta_sha256": sha256(meta_path),
            "authenticated_resume": resumed,
            **stats,
        }
        rows.append(record)
        if repeat is None and witness_arrays is None and stats["inference_calls"] > 0:
            witness_arrays = {key: value.copy() for key, value in arrays.items()}
            repeated, repeated_stats = materialize_video(
                model=model,
                detections=cache.detections,
                base=base,
                identity=identity,
            )
            equal = {
                key: bool(np.array_equal(value, repeated[key]))
                for key, value in witness_arrays.items()
            }
            repeat = {
                "identity": identity,
                "arrays": equal,
                "all_array_equal": all(equal.values()),
                "inference_calls": repeated_stats["inference_calls"],
            }
            if not repeat["all_array_equal"]:
                raise SystemExit("repeat witness is not byte-identical")
        print(
            f"[{ordinal}/36] {identity} {'resumed' if resumed else 'materialized'} "
            f"calls={stats['inference_calls']} "
            f"rows={stats['row_count']}",
            flush=True,
        )

    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "inference_calls",
            "cropped_calls",
            "proposal_count",
            "row_count",
        )
    }
    totals["video_count"] = len(rows)
    totals["peak_tokens"] = max(int(row["peak_tokens"]) for row in rows)
    if (
        totals["video_count"],
        totals["inference_calls"],
        totals["proposal_count"],
        totals["row_count"],
    ) != (36, 3539, 84883, 121534):
        raise SystemExit(f"frozen population totals drifted: {totals}")
    manifest = {
        "schema": "g02.trackastra-authentic-reverse-cycle-manifest.v1",
        "status": "COMPLETE",
        "classification": "NON-GT-IDENTIFIABLE",
        "supplier": "TRACKASTRA-AUTHENTIC-REVERSE-44B6-V1",
        "endpoint_return": "A_reverse[index(v),index(u)] from a new reversed-time forward pass",
        "prohibited_shortcuts_used": False,
        "pins": {
            "checkpoint_member_sha256": MODEL_SHA,
            "checkpoint_archive_sha256": ARCHIVE_SHA,
            "option_contract_sha256": CONTRACT_SHA,
            "endpoint_contract_sha256": ENDPOINT_CONTRACT_SHA,
            "forward_report_sha256": FORWARD_REPORT_SHA,
        },
        "determinism": repeat,
        "totals": totals,
        "files": rows,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "cpu_only": True,
            "gpu": False,
            "kaggle_kernel": False,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "labels_read": False,
            "training": False,
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "totals": totals}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
