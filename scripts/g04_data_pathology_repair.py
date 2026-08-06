#!/usr/bin/env python3
"""Audit and repair acquisition pathologies before sparse candidate generation.

This is the frame-registration node preregistered in
``reports/g04/lanes/data_pathology_repair/PREREGISTRATION.md``.  Phase
correlation is used once per complete frame transition; it is never emitted as
a candidate feature.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any
import zipfile

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_OUTPUT = ROOT / "reports/g04/lanes/data_pathology_repair"
SCHEMA = "celltrack-alpha.g04-data-pathology-census.v1"
VIDEO_SCHEMA = "celltrack-alpha.g04-data-pathology-video.v1"
SHAPE_TZYX = (100, 64, 256, 256)
STRIDE_ZYX = (1, 4, 4)
SCALE_ZYX_UM = (1.625, 0.40625, 0.40625)
REGISTERED_SCALE_ZYX_UM = tuple(
    scale * stride for scale, stride in zip(SCALE_ZYX_UM, STRIDE_ZYX, strict=True)
)
POOL_RADIUS_UM = 10.0
CHUNK_RE = re.compile(
    r"^(train|test)/([^/]+)\.zarr/0/c/(\d+)/0/0/0$"
)


class PathologyError(RuntimeError):
    """An immutable data or execution contract was violated."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise PathologyError(f"refusing stale temporary output: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def evaluator_stride(frame_zyx: np.ndarray) -> np.ndarray:
    """Return the organizer-exact spatial stride view, never a block mean."""

    frame = np.asarray(frame_zyx)
    if frame.shape != SHAPE_TZYX[1:] or frame.dtype != np.uint16:
        raise PathologyError(
            f"raw frame contract drifted: shape={frame.shape}, dtype={frame.dtype}"
        )
    return np.asarray(frame[::1, ::4, ::4])


def robust_windowed(frame_zyx: np.ndarray) -> np.ndarray:
    """Robustly centre and Hann-window one strided 3-D volume."""

    values = np.asarray(frame_zyx, dtype=np.float32)
    if values.shape != (64, 64, 64) or not np.isfinite(values).all():
        raise PathologyError("registration frame must be finite 64x64x64")
    low, high = np.quantile(values, (0.01, 0.99))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise PathologyError("registration quantiles are nonfinite")
    clipped = np.clip(values, low, high)
    clipped = clipped - np.median(clipped)
    windows = [np.hanning(size).astype(np.float32) for size in clipped.shape]
    window = windows[0][:, None, None] * windows[1][None, :, None]
    window = window * windows[2][None, None, :]
    result = clipped * window
    if float(np.linalg.norm(result.ravel())) <= 1.0e-12:
        raise PathologyError("registration frame has no windowed signal")
    return np.ascontiguousarray(result, dtype=np.float32)


def integer_phase_shift(
    reference_zyx: np.ndarray, moving_zyx: np.ndarray
) -> tuple[int, int, int]:
    """Return the integer shift that moves ``moving`` onto ``reference``."""

    reference = robust_windowed(reference_zyx)
    moving = robust_windowed(moving_zyx)
    reference_fft = np.fft.rfftn(reference)
    moving_fft = np.fft.rfftn(moving)
    cross_power = reference_fft * np.conjugate(moving_fft)
    amplitude = np.abs(cross_power)
    cross_power /= np.maximum(amplitude, np.finfo(np.float64).eps)
    correlation = np.fft.irfftn(cross_power, s=reference.shape, axes=(0, 1, 2))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift = []
    for index, size in zip(peak, reference.shape, strict=True):
        value = int(index)
        if value > size // 2:
            value -= size
        shift.append(value)
    return tuple(shift)  # type: ignore[return-value]


def overlap_slices(
    shape: Sequence[int], shift_zyx: Sequence[int]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Slices comparing reference with moving translated by ``shift_zyx``."""

    if len(shape) != 3 or len(shift_zyx) != 3:
        raise PathologyError("overlap requires three spatial axes")
    reference: list[slice] = []
    moving: list[slice] = []
    for size_raw, shift_raw in zip(shape, shift_zyx, strict=True):
        size = int(size_raw)
        shift = int(shift_raw)
        if size <= 0 or abs(shift) >= size:
            raise PathologyError("registration shift leaves no overlap")
        if shift >= 0:
            reference.append(slice(shift, size))
            moving.append(slice(0, size - shift))
        else:
            reference.append(slice(0, size + shift))
            moving.append(slice(-shift, size))
    return tuple(reference), tuple(moving)


def overlap_ncc(
    reference_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    shift_zyx: Sequence[int],
) -> float:
    """Zero-mean NCC on the non-wrapping overlap after an integer shift."""

    reference = np.asarray(reference_zyx, dtype=np.float64)
    moving = np.asarray(moving_zyx, dtype=np.float64)
    if reference.shape != moving.shape or reference.ndim != 3:
        raise PathologyError("NCC volumes must have equal 3-D shapes")
    reference_slice, moving_slice = overlap_slices(reference.shape, shift_zyx)
    left = reference[reference_slice].ravel()
    right = moving[moving_slice].ravel()
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def shift_magnitude_um(shift_zyx: Sequence[int]) -> float:
    shift = np.asarray(shift_zyx, dtype=np.float64)
    scale = np.asarray(REGISTERED_SCALE_ZYX_UM, dtype=np.float64)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise PathologyError("registration shift is malformed")
    return float(np.linalg.norm(shift * scale))


def _members(
    archive: Path,
) -> tuple[
    dict[tuple[str, str], dict[int, zipfile.ZipInfo]],
    str,
]:
    videos: dict[tuple[str, str], dict[int, zipfile.ZipInfo]] = defaultdict(dict)
    central_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            match = CHUNK_RE.match(info.filename)
            if match is None:
                continue
            key = (match.group(1), match.group(2))
            frame = int(match.group(3))
            if frame in videos[key]:
                raise PathologyError(f"duplicate raw chunk: {key} frame {frame}")
            videos[key][frame] = info
            central_rows.append(
                {
                    "compress_size": info.compress_size,
                    "crc32": info.CRC,
                    "file_size": info.file_size,
                    "filename": info.filename,
                }
            )
    if not videos or any(set(frames) != set(range(100)) for frames in videos.values()):
        raise PathologyError("raw archive does not contain complete 100-frame videos")
    fingerprint = hashlib.sha256(canonical_json(sorted(central_rows, key=lambda row: row["filename"]))).hexdigest()
    return dict(videos), fingerprint


def exact_frozen_transitions(
    archive: Path,
    videos: Mapping[tuple[str, str], Mapping[int, zipfile.ZipInfo]],
) -> tuple[dict[tuple[str, str], set[int]], dict[str, int]]:
    """Confirm candidate equalities by exact encoded-chunk byte comparison."""

    frozen: dict[tuple[str, str], set[int]] = defaultdict(set)
    crc_candidates = confirmed = rejected = 0
    with zipfile.ZipFile(archive) as handle:
        for key in sorted(videos):
            frames = videos[key]
            for frame in range(99):
                left = frames[frame]
                right = frames[frame + 1]
                if left.CRC != right.CRC or left.file_size != right.file_size:
                    continue
                crc_candidates += 1
                if handle.read(left) == handle.read(right):
                    frozen[key].add(frame)
                    confirmed += 1
                else:
                    rejected += 1
    return dict(frozen), {
        "crc_and_size_candidates": crc_candidates,
        "encoded_byte_equal_confirmed": confirmed,
        "encoded_byte_unequal_rejected": rejected,
    }


def _run_lengths(transitions: Iterable[int]) -> list[int]:
    values = sorted(set(int(value) for value in transitions))
    if not values:
        return []
    lengths: list[int] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        lengths.append(previous - start + 1)
        start = previous = value
    lengths.append(previous - start + 1)
    return lengths


def analyze_video(
    array: Any,
    *,
    split: str,
    dataset: str,
    frozen: set[int],
) -> dict[str, Any]:
    if tuple(int(value) for value in array.shape) != SHAPE_TZYX or np.dtype(array.dtype) != np.dtype("uint16"):
        raise PathologyError(f"{split}:{dataset}: raw array shape/dtype drifted")
    transitions: list[dict[str, Any]] = []
    previous = np.asarray(array[0, ::1, ::4, ::4])
    if previous.shape != (64, 64, 64) or previous.dtype != np.uint16:
        raise PathologyError(f"{split}:{dataset}: stride read drifted")
    for frame in range(99):
        current = np.asarray(array[frame + 1, ::1, ::4, ::4])
        if frame in frozen:
            shift = (0, 0, 0)
            pre_ncc = post_ncc = 1.0
        else:
            shift = integer_phase_shift(previous, current)
            pre_ncc = overlap_ncc(previous, current, (0, 0, 0))
            post_ncc = overlap_ncc(previous, current, shift)
        magnitude = shift_magnitude_um(shift)
        transitions.append(
            {
                "frame": frame,
                "frozen": frame in frozen,
                "ncc_gain": post_ncc - pre_ncc,
                "post_ncc": post_ncc,
                "pre_ncc": pre_ncc,
                "pool_breaking": magnitude > POOL_RADIUS_UM,
                "shift_magnitude_um": magnitude,
                "shift_registered_zyx": list(shift),
                "shift_um_zyx": [
                    float(value * scale)
                    for value, scale in zip(shift, REGISTERED_SCALE_ZYX_UM, strict=True)
                ],
            }
        )
        previous = current
    return {
        "schema": VIDEO_SCHEMA,
        "dataset": dataset,
        "family": dataset.split("_", 1)[0],
        "split": split,
        "shape_tzyx": list(SHAPE_TZYX),
        "scale_zyx_um": list(SCALE_ZYX_UM),
        "stride_zyx": list(STRIDE_ZYX),
        "registered_scale_zyx_um": list(REGISTERED_SCALE_ZYX_UM),
        "frozen_transitions": sorted(frozen),
        "frozen_run_transition_lengths": _run_lengths(frozen),
        "transitions": transitions,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise PathologyError("cannot summarize empty/nonfinite distribution")
    points = (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    return {
        f"p{int(round(point * 100)):02d}": float(np.quantile(array, point))
        for point in points
    }


def _magnitude_histogram(values: Sequence[float]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    edges = (0.0, 1.625, 3.25, 5.0, 7.5, 10.0, 12.0, 15.0, 20.0, math.inf)
    result: dict[str, int] = {}
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        label = f"({left:g},{right:g}]" if math.isfinite(right) else f"({left:g},inf)"
        if left == 0.0:
            result["[0,1.625]"] = int(np.sum((array >= left) & (array <= right)))
        else:
            result[label] = int(np.sum((array > left) & (array <= right)))
    return result


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    archive: Path,
    central_directory_sha256: str,
    frozen_audit: Mapping[str, int],
    elapsed_seconds: float,
) -> dict[str, Any]:
    transitions = [transition for row in rows for transition in row["transitions"]]
    moving = [row for row in transitions if not row["frozen"]]
    magnitudes = [float(row["shift_magnitude_um"]) for row in moving]
    gains = [float(row["ncc_gain"]) for row in moving]
    by_scope: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        for family in ("44b6", "6bba"):
            selected_videos = [
                row for row in rows if row["split"] == split and row["family"] == family
            ]
            if not selected_videos:
                continue
            selected_transitions = [
                transition for row in selected_videos for transition in row["transitions"]
            ]
            by_scope[f"{split}:{family}"] = {
                "videos": len(selected_videos),
                "transitions": len(selected_transitions),
                "frozen_transitions": sum(bool(row["frozen"]) for row in selected_transitions),
                "pool_breaking_transitions": sum(bool(row["pool_breaking"]) for row in selected_transitions),
                "shift_magnitude_um_quantiles_nonfrozen": _quantiles(
                    [
                        float(row["shift_magnitude_um"])
                        for row in selected_transitions
                        if not row["frozen"]
                    ]
                ),
            }
    all_run_lengths = [
        length for row in rows for length in row["frozen_run_transition_lengths"]
    ]
    result = {
        "schema": SCHEMA,
        "verdict": "PASS_COMPLETE_CENSUS",
        "archive": {
            "path": str(archive.relative_to(ROOT)),
            "bytes": archive.stat().st_size,
            "central_directory_sha256": central_directory_sha256,
        },
        "contract": {
            "shape_tzyx": list(SHAPE_TZYX),
            "scale_zyx_um": list(SCALE_ZYX_UM),
            "stride_zyx": list(STRIDE_ZYX),
            "stride_operation": "volume[:, ::4, ::4]",
            "registered_scale_zyx_um": list(REGISTERED_SCALE_ZYX_UM),
            "registration": "integer-3d-phase-correlation-hann-q01-q99-median-centered",
            "pool_breaking_threshold_um": POOL_RADIUS_UM,
        },
        "counts": {
            "videos": len(rows),
            "transitions": len(transitions),
            "frozen_transitions": sum(bool(row["frozen"]) for row in transitions),
            "frozen_videos": sum(bool(row["frozen_transitions"]) for row in rows),
            "moving_transitions": len(moving),
            "pool_breaking_transitions": sum(bool(row["pool_breaking"]) for row in moving),
            "pool_breaking_videos": sum(
                any(transition["pool_breaking"] for transition in row["transitions"])
                for row in rows
            ),
        },
        "frozen_confirmation": dict(frozen_audit),
        "frozen_run_transition_length_histogram": dict(
            sorted(Counter(all_run_lengths).items())
        ),
        "shift_magnitude_um_quantiles_nonfrozen": _quantiles(magnitudes),
        "shift_magnitude_um_histogram_nonfrozen": _magnitude_histogram(magnitudes),
        "registration_ncc_gain_quantiles_nonfrozen": _quantiles(gains),
        "by_scope": by_scope,
        "videos": [
            {
                "dataset": row["dataset"],
                "family": row["family"],
                "frozen_transitions": len(row["frozen_transitions"]),
                "max_shift_magnitude_um": max(
                    float(value["shift_magnitude_um"]) for value in row["transitions"]
                ),
                "path": f"videos/{row['split']}__{row['dataset']}.json",
                "pool_breaking_transitions": sum(
                    bool(value["pool_breaking"]) for value in row["transitions"]
                ),
                "split": row["split"],
            }
            for row in rows
        ],
        "elapsed_seconds": elapsed_seconds,
    }
    return result


def run_census(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if not args.archive.is_file():
        raise PathologyError(f"raw archive missing: {args.archive}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "videos").mkdir(exist_ok=True)
    videos, central_sha = _members(args.archive)
    frozen, frozen_audit = exact_frozen_transitions(args.archive, videos)
    rows: list[dict[str, Any]] = []
    store: ZipStore | None = None
    try:
        for ordinal, (split, dataset) in enumerate(sorted(videos), start=1):
            output_path = args.output / "videos" / f"{split}__{dataset}.json"
            if output_path.is_file():
                row = json.loads(output_path.read_text(encoding="ascii"))
                if (
                    row.get("schema") != VIDEO_SCHEMA
                    or row.get("split") != split
                    or row.get("dataset") != dataset
                    or row.get("frozen_transitions") != sorted(frozen.get((split, dataset), set()))
                ):
                    raise PathologyError(f"checkpoint drifted: {output_path}")
                rows.append(row)
                print(f"[{ordinal:03d}/{len(videos):03d}] {split}:{dataset} resume", flush=True)
                continue
            if store is None:
                store = ZipStore(args.archive, mode="r")
            group = zarr.open_group(
                store=store,
                path=f"{split}/{dataset}.zarr",
                mode="r",
            )
            array = group["0"]
            row = analyze_video(
                array,
                split=split,
                dataset=dataset,
                frozen=frozen.get((split, dataset), set()),
            )
            atomic_json(output_path, row)
            rows.append(row)
            print(
                f"[{ordinal:03d}/{len(videos):03d}] {split}:{dataset} "
                f"frozen={len(row['frozen_transitions'])} "
                f"pool_breaking={sum(bool(value['pool_breaking']) for value in row['transitions'])}",
                flush=True,
            )
    finally:
        if store is not None:
            store.close()
    result = summarize(
        rows,
        archive=args.archive,
        central_directory_sha256=central_sha,
        frozen_audit=frozen_audit,
        elapsed_seconds=time.perf_counter() - started,
    )
    atomic_json(args.output / "CENSUS.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--command", choices=("census",), default="census")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "census":
            result = run_census(args)
        else:  # pragma: no cover - argparse owns the choice set.
            raise PathologyError(f"unsupported command: {args.command}")
    except (OSError, ValueError, zipfile.BadZipFile, PathologyError) as exc:
        print(f"G04 data-pathology repair failed closed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
