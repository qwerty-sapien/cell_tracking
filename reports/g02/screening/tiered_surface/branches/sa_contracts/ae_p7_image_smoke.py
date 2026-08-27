#!/usr/bin/env python3
"""Run the frozen, no-score AE-P7 raw-image interface smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def center_crop(array: Any, frame: int, center: tuple[int, int, int], size: tuple[int, int, int]) -> np.ndarray:
    """Read one bounded TZYX crop, replicating only at the image border."""
    half = tuple(value // 2 for value in size)
    starts = tuple(center[index] - half[index] for index in range(3))
    ends = tuple(starts[index] + size[index] for index in range(3))
    source_starts = tuple(max(0, value) for value in starts)
    source_ends = tuple(min(int(array.shape[index + 1]), ends[index]) for index in range(3))
    crop = np.asarray(array[(frame, *[slice(source_starts[index], source_ends[index]) for index in range(3)])])
    padding = tuple(
        (source_starts[index] - starts[index], ends[index] - source_ends[index])
        for index in range(3)
    )
    return np.pad(crop, padding, mode="edge")


def phase_features(source: np.ndarray, destination: np.ndarray) -> dict[str, Any]:
    """Deterministic phase-correlation primitive from the frozen P7 contract."""
    if source.shape != destination.shape or source.ndim != 3:
        raise RuntimeError("matched 3-D source and destination crops required")
    source64 = np.asarray(source, dtype=np.float64)
    destination64 = np.asarray(destination, dtype=np.float64)
    product = np.fft.fftn(source64) * np.conj(np.fft.fftn(destination64))
    normalized = product / np.maximum(np.abs(product), 1e-12)
    correlation = np.fft.ifftn(normalized).real
    peak_flat = int(np.argmax(correlation))
    peak_index = tuple(int(value) for value in np.unravel_index(peak_flat, correlation.shape))
    peak = float(correlation[peak_index])
    mask = np.ones(correlation.shape, dtype=bool)
    radii = tuple(min(1, value) for value in peak_index)
    mask[
        tuple(
            slice(max(0, peak_index[index] - radii[index]), min(correlation.shape[index], peak_index[index] + radii[index] + 1))
            for index in range(3)
        )
    ] = False
    second = float(np.max(correlation[mask])) if np.any(mask) else 0.0
    shift = tuple(
        value if value <= correlation.shape[index] // 2 else value - correlation.shape[index]
        for index, value in enumerate(peak_index)
    )
    ratio = float(peak / max(abs(second), 1e-12))
    values = np.asarray((peak, second, ratio, *shift), dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite phase feature")
    return {"peak": peak, "second_peak": second, "peak_ratio": ratio, "shift_zyx_voxels": list(shift)}


def normalized(crop: np.ndarray, quantiles: dict[str, Any]) -> np.ndarray:
    low = float(quantiles["0.01"])
    high = float(quantiles["0.99"])
    if not high > low:
        raise RuntimeError("invalid image quantiles")
    value = np.clip((np.asarray(crop, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
    if not np.isfinite(value).all():
        raise RuntimeError("non-finite normalized crop")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=HERE / "AE_P7_FIT105_SMOKE_CONTRACT.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    contract = load(args.contract)
    if contract["status"] != "FROZEN_BEFORE_EXECUTION" or contract["node"] != "AE-P7":
        raise RuntimeError("not the frozen AE-P7 smoke contract")
    source_contract = HERE / contract["pins"]["ae_p7_contract"]["path"].rsplit("/", 1)[-1]
    if sha256(source_contract) != contract["pins"]["ae_p7_contract"]["sha256"]:
        raise RuntimeError("AE-P7 node-contract pin drift")
    archive = Path(contract["pins"]["raw_archive"]["path"])
    if archive.stat().st_size != int(contract["pins"]["raw_archive"]["bytes"]):
        raise RuntimeError("raw archive size pin drift")

    read = contract["read_contract"]
    expected_shape = tuple(int(value) for value in read["expected_shape_tzyx"])
    expected_chunks = tuple(int(value) for value in read["expected_chunks_tzyx"])
    center = tuple(int(value) for value in read["center_zyx"])
    size = tuple(int(value) for value in read["crop_size_zyx"])
    extracted_root = Path(read["extracted_root"])
    rows = []
    with ZipStore(archive, mode="r") as store:
        for identity in contract["samples"]:
            path = f"train/{identity}.zarr"
            archive_group = zarr.open_group(store=store, path=path, mode="r")
            archive_array = archive_group["0"]
            extracted_group = zarr.open_group(extracted_root / f"{identity}.zarr", mode="r")
            extracted_array = extracted_group["0"]
            if tuple(archive_array.shape) != expected_shape or tuple(archive_array.chunks) != expected_chunks:
                raise RuntimeError(f"archive geometry drift: {identity}")
            if tuple(extracted_array.shape) != expected_shape or tuple(extracted_array.chunks) != expected_chunks:
                raise RuntimeError(f"extracted geometry drift: {identity}")
            quantiles = dict(archive_group.attrs["image_statistics"]["quantiles"])
            archive_crops = [center_crop(archive_array, int(frame), center, size) for frame in read["frames"]]
            extracted_crops = [center_crop(extracted_array, int(frame), center, size) for frame in read["frames"]]
            if any(not np.array_equal(left, right) for left, right in zip(archive_crops, extracted_crops, strict=True)):
                raise RuntimeError(f"archive/extracted crop mismatch: {identity}")
            source = normalized(archive_crops[0], quantiles)
            destination = normalized(archive_crops[1], quantiles)
            first = phase_features(source, destination)
            second = phase_features(source, destination)
            if first != second:
                raise RuntimeError(f"non-deterministic phase features: {identity}")
            rows.append(
                {
                    "identity": identity,
                    "crop_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in archive_crops],
                    "phase_features": first,
                    "archive_extracted_equal": True,
                }
            )

    result = {
        "schema": "celltrack-alpha.ae-p7-fit105-smoke-result.v1",
        "status": "PASS_INTERFACE_ONLY",
        "contract_sha256": sha256(args.contract),
        "sample_count": len(rows),
        "samples": rows,
        "scope": contract["scope"],
        "conclusion": "The authenticated archive supports bounded read-only raw-image access and deterministic finite phase features without full image-tree extraction. No P7 selector, fit, score, graph, or tier result exists.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sample_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
