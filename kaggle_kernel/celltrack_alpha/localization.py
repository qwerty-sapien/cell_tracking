"""Fold-safe localization and output-offset calibration helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsafeCalibrationError(ValueError):
    """Raised when an offset/localizer calibration lacks fold-safe provenance."""


@dataclass(frozen=True)
class OffsetCalibration:
    """Validated offset calibration payload.

    Offsets are post-detection voxel corrections in z/y/x order. They are allowed
    only when a calibration file declares fold-safe provenance against the locked
    G-01 split manifest.
    """

    path: Path
    payload: dict[str, Any]
    fold_id: str
    calibration_train: frozenset[str]
    validation_datasets: frozenset[str]
    excluded_duplicate_public_ids: frozenset[str]
    split_manifest_sha256: str | None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_dataset_set(value: Any, name: str) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise UnsafeCalibrationError(f"{name} must be a list of dataset strings")
    return frozenset(value)


def _offset_tuple(value: Any, *, context: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise UnsafeCalibrationError(f"{context} offset_zyx must contain exactly three integers")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise UnsafeCalibrationError(f"{context} offset_zyx must contain exactly three integers")
        out.append(int(item))
    return tuple(out)  # type: ignore[return-value]


def _load_split_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise UnsafeCalibrationError(f"split manifest is required for offset calibration: {path}")
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def validate_offset_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    split_manifest_path: Path,
    expected_fold_id: str | None = None,
) -> OffsetCalibration:
    """Validate that a calibration file can be used for held-out scoring."""

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise UnsafeCalibrationError("offset calibration must include a provenance object")
    if provenance.get("status") != "fold_safe":
        raise UnsafeCalibrationError("offset calibration provenance.status must be 'fold_safe'")
    fold_id = str(provenance.get("fold_id", ""))
    if not fold_id:
        raise UnsafeCalibrationError("offset calibration provenance.fold_id is required")
    if expected_fold_id is not None and fold_id != str(expected_fold_id):
        raise UnsafeCalibrationError(
            f"offset calibration fold_id={fold_id!r} does not match expected {expected_fold_id!r}"
        )

    split_manifest, split_hash = _load_split_manifest(split_manifest_path)
    folds = split_manifest.get("folds", {})
    fold = folds.get(fold_id)
    if not isinstance(fold, dict):
        raise UnsafeCalibrationError(f"fold {fold_id!r} is not present in split manifest")

    calibration_train = _as_dataset_set(provenance.get("calibration_train"), "provenance.calibration_train")
    validation_datasets = _as_dataset_set(provenance.get("validation_datasets"), "provenance.validation_datasets")
    expected_train = frozenset(str(v) for v in fold.get("calibration_train", []))
    expected_validation = frozenset(str(v) for v in fold.get("validation", []))
    if calibration_train - expected_train:
        raise UnsafeCalibrationError("calibration_train contains datasets outside the locked fold training set")
    if validation_datasets - expected_validation:
        raise UnsafeCalibrationError("validation_datasets contains datasets outside the locked fold validation set")
    if calibration_train & validation_datasets:
        raise UnsafeCalibrationError("calibration_train overlaps validation_datasets")

    excluded = frozenset(
        str(v)
        for v in split_manifest.get("excluded_from_general_score", {}).get(
            "duplicate_public_ids", []
        )
    )
    source_datasets = _as_dataset_set(provenance.get("source_datasets"), "provenance.source_datasets")
    if source_datasets - calibration_train:
        raise UnsafeCalibrationError("source_datasets must be a subset of calibration_train")
    if source_datasets & excluded:
        raise UnsafeCalibrationError("source_datasets include duplicate public IDs excluded from general score")

    recorded_split_hash = provenance.get("split_manifest_sha256")
    if recorded_split_hash is not None and str(recorded_split_hash) != split_hash:
        raise UnsafeCalibrationError("calibration split_manifest_sha256 does not match current split manifest")

    for section_name in ("dataset_offsets", "prefix_offsets"):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise UnsafeCalibrationError(f"{section_name} must be an object")
        for key, row in section.items():
            if not isinstance(row, dict):
                raise UnsafeCalibrationError(f"{section_name}.{key} must be an object")
            _offset_tuple(row.get("offset_zyx"), context=f"{section_name}.{key}")

    return OffsetCalibration(
        path=path,
        payload=payload,
        fold_id=fold_id,
        calibration_train=calibration_train,
        validation_datasets=validation_datasets,
        excluded_duplicate_public_ids=excluded,
        split_manifest_sha256=split_hash,
    )


def load_offset_calibration(
    path: Path | None,
    *,
    split_manifest_path: Path = Path("reports/general_model_split_manifest.json"),
    expected_fold_id: str | None = None,
) -> OffsetCalibration | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnsafeCalibrationError("offset calibration root must be an object")
    return validate_offset_payload(
        payload,
        path=path,
        split_manifest_path=split_manifest_path,
        expected_fold_id=expected_fold_id,
    )


def offset_for_dataset(dataset: str, calibration: OffsetCalibration | None) -> tuple[int, int, int] | None:
    if calibration is None:
        return None
    if dataset in calibration.calibration_train:
        raise UnsafeCalibrationError(
            f"refusing to apply held-out scoring offset to calibration dataset {dataset!r}"
        )
    if dataset not in calibration.validation_datasets:
        raise UnsafeCalibrationError(
            f"dataset {dataset!r} is not in the calibration validation set for fold {calibration.fold_id}"
        )
    payload = calibration.payload
    dataset_offsets = payload.get("dataset_offsets", {})
    if dataset in dataset_offsets:
        return _offset_tuple(dataset_offsets[dataset]["offset_zyx"], context=f"dataset_offsets.{dataset}")
    prefix_offsets = payload.get("prefix_offsets", {})
    for prefix, row in prefix_offsets.items():
        if dataset.startswith(str(prefix)):
            return _offset_tuple(row["offset_zyx"], context=f"prefix_offsets.{prefix}")
    return None
