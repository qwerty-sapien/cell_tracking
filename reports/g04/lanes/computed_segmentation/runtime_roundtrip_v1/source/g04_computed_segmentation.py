#!/usr/bin/env python3
"""Computed 3-D segmentations and faithful Trackastra ``wrfeat`` export.

This module deliberately treats every label image it creates or consumes as a
``COMPUTED_SEGMENTATION``.  It never represents those labels as H3, authentic,
or organizer-provided masks.

The checkpoint-facing feature matrix has the exact 12-channel Trackastra
``wrfeat`` order.  Richer morphology requested for division work is emitted in
a separate matrix because overloading the checkpoint channels would be
semantically wrong.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Protocol

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABEL = "COMPUTED_SEGMENTATIONS"
SCHEMA = "celltrack-alpha.g04-computed-segmentation-features.v1"
SIDECAR_SCHEMA = "celltrack-alpha.g04-computed-segmentation-sidecar.v1"
PREFLIGHT_SCHEMA = "celltrack-alpha.g04-computed-segmentation-preflight.v1"
MANIFEST_SCHEMA = "celltrack-alpha.g04-computed-segmentation-wrfeat-manifest.v1"

TRACKASTRA_NATIVE_FEATURE_NAMES = (
    "equivalent_diameter_area",
    "intensity_mean",
    "inertia_tensor-0-0",
    "inertia_tensor-0-1",
    "inertia_tensor-0-2",
    "inertia_tensor-1-0",
    "inertia_tensor-1-1",
    "inertia_tensor-1-2",
    "inertia_tensor-2-0",
    "inertia_tensor-2-1",
    "inertia_tensor-2-2",
    "border_dist",
)

DESCRIPTIVE_FEATURE_NAMES = (
    "volume_voxels",
    "axis_length_major_physical",
    "axis_length_intermediate_physical",
    "axis_length_minor_physical",
    "elongation_major_over_minor",
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "solidity_3d_convex_hull",
    "equivalent_diameter_area",
    "border_dist",
)


class ComputedSegmentationError(RuntimeError):
    """The computed-segmentation pipeline failed closed."""


class Segmenter(Protocol):
    backend_name: str
    provenance: Mapping[str, Any]

    def segment(self, image_zyx: np.ndarray) -> np.ndarray:
        """Return a uint32 Z/Y/X label image (zero is background)."""


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".json"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise ComputedSegmentationError(f"{label} is absent: {path}")
    observed = sha256(path)
    if observed != expected_sha256.lower():
        raise ComputedSegmentationError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return observed


def _zarr_array(zarr_path: Path):
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised by preflight
        raise ComputedSegmentationError("zarr is not installed") from exc
    if not zarr_path.exists():
        raise ComputedSegmentationError(f"raw Zarr is absent: {zarr_path}")
    value = zarr.open(str(zarr_path), mode="r")
    if hasattr(value, "shape"):
        array = value
        attrs: Mapping[str, Any] = getattr(value, "attrs", {})
    elif "0" in value:
        array = value["0"]
        attrs = getattr(value, "attrs", {})
    else:
        raise ComputedSegmentationError("Zarr has no root array or level '0'")
    if len(array.shape) != 4:
        raise ComputedSegmentationError(
            f"expected T/Z/Y/X raw data, got shape {array.shape}"
        )
    return array, attrs


def spatial_scale_zyx(attrs: Mapping[str, Any]) -> tuple[float, float, float]:
    """Read OME-Zarr scale; fall back to unit spacing with explicit provenance."""
    try:
        multiscales = attrs["multiscales"]
        transform = multiscales[0]["datasets"][0]["coordinateTransformations"]
        scale = next(row["scale"] for row in transform if row["type"] == "scale")
        value = tuple(float(item) for item in scale[-3:])
    except (KeyError, IndexError, StopIteration, TypeError, ValueError):
        value = (1.0, 1.0, 1.0)
    if len(value) != 3 or not all(item > 0 and math.isfinite(item) for item in value):
        raise ComputedSegmentationError(f"invalid spatial scale: {value}")
    return value


def load_detection_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            node_ids = np.asarray(data["node_ids"], dtype=np.int64)
            timepoints = np.asarray(data["timepoints"], dtype=np.int32)
            coords = np.asarray(data["coords_zyx"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise ComputedSegmentationError(f"invalid detection NPZ: {path}") from exc
    return validate_detections(node_ids, timepoints, coords)


def load_bh_detections(
    cache_root: Path, identity: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the campaign's authenticated sparse point detections."""
    for location in (
        ROOT / "src",
        ROOT / "external/Tracking_Support/repo/src",
    ):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    try:
        from celltrack_alpha import bh_model

        slug = identity.replace(":", "__", 1)
        meta_path = cache_root / f"{slug}.json"
        meta = json.loads(meta_path.read_text(encoding="ascii"))
        model = meta["contract"]["model"]
        cache_identity = bh_model.CacheIdentity(
            str(model["config_sha256"]),
            str(model["source_sha256"]),
            str(model["weight_sha256"]),
            int(model["pair_batch_size"]),
            int(model["view_batch_size"]),
        )
        frames = bh_model.load_sparse_cache_detections(
            cache_root / f"{slug}.bhc", expected_identity=cache_identity
        )
    except Exception as exc:
        raise ComputedSegmentationError(
            f"cannot load authenticated detections for {identity}"
        ) from exc
    node_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    coord_chunks: list[np.ndarray] = []
    for frame in frames:
        ids = np.asarray(frame.node_ids, dtype=np.int64)
        node_chunks.append(ids)
        time_chunks.append(np.full(len(ids), int(frame.frame_index), np.int32))
        coord_chunks.append(np.asarray(frame.coords_zyx, dtype=np.float32))
    return validate_detections(
        np.concatenate(node_chunks),
        np.concatenate(time_chunks),
        np.concatenate(coord_chunks),
    )


def validate_detections(
    node_ids: np.ndarray, timepoints: np.ndarray, coords_zyx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_ids = np.asarray(node_ids, dtype=np.int64)
    timepoints = np.asarray(timepoints, dtype=np.int32)
    coords_zyx = np.asarray(coords_zyx, dtype=np.float32)
    if (
        node_ids.ndim != 1
        or timepoints.shape != node_ids.shape
        or coords_zyx.shape != (len(node_ids), 3)
        or len(node_ids) == 0
        or len(np.unique(node_ids)) != len(node_ids)
        or np.any(timepoints < 0)
        or not np.isfinite(coords_zyx).all()
    ):
        raise ComputedSegmentationError("detection arrays are malformed")
    order = np.lexsort((node_ids, timepoints))
    return node_ids[order], timepoints[order], coords_zyx[order]


class Cellpose3D:
    """Offline-only Cellpose 3.1.1.2 adapter with authenticated weight bytes."""

    backend_name = "cellpose-3d-cyto3"

    def __init__(
        self,
        *,
        weight: Path,
        weight_sha256: str,
        license_file: Path,
        license_sha256: str,
        source_url: str,
        diameter_xy: float,
        anisotropy: float,
        cellprob_threshold: float = 0.0,
        flow3d_smooth: float = 0.0,
        min_size: int = 15,
        device: str = "cpu",
        seed: int = 20260803,
    ) -> None:
        verify_file(weight, weight_sha256, "Cellpose weight")
        verify_file(license_file, license_sha256, "Cellpose license")
        if not source_url.startswith("https://"):
            raise ComputedSegmentationError("Cellpose source URL must be HTTPS")
        if device not in {"cpu", "cuda"}:
            raise ComputedSegmentationError(
                "the deterministic G-04 adapter permits only cpu or cuda"
            )
        try:
            installed = importlib.metadata.version("cellpose")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ComputedSegmentationError(
                "cellpose==3.1.1.2 is absent; install it in an isolated Python "
                "3.10/3.11 environment before running"
            ) from exc
        if installed != "3.1.1.2":
            raise ComputedSegmentationError(
                f"cellpose version drift: expected 3.1.1.2, got {installed}"
            )
        import torch
        from cellpose import models

        numpy_version = importlib.metadata.version("numpy")
        declared_requirements = importlib.metadata.requires("cellpose") or []
        numpy_requirements = [
            item for item in declared_requirements if item.lower().startswith("numpy")
        ]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        use_cuda = device == "cuda"
        if use_cuda:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if use_cuda and not torch.cuda.is_available():
            raise ComputedSegmentationError("cuda was requested but is unavailable")
        if use_cuda:
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch_device = torch.device(device)
        self._model = models.CellposeModel(
            gpu=use_cuda,
            pretrained_model=str(weight),
            diam_mean=30.0,
            nchan=2,
            device=torch_device,
        )
        self._diameter_xy = float(diameter_xy)
        self._anisotropy = float(anisotropy)
        self._cellprob_threshold = float(cellprob_threshold)
        self._flow3d_smooth = float(flow3d_smooth)
        self._min_size = int(min_size)
        self.provenance = {
            "backend": self.backend_name,
            "package": {
                "name": "cellpose",
                "version": installed,
                "declared_numpy_requirements": numpy_requirements,
                "observed_numpy_version": numpy_version,
                "declared_bound_exception": (
                    "carrier uses NumPy 2.4.6 above Cellpose 3.1.1.2's declared "
                    "numpy<2.1 bound; this is reported, never hidden"
                    if numpy_version == "2.4.6"
                    else None
                ),
            },
            "weight": {"path": str(weight), "sha256": weight_sha256.lower()},
            "license": {
                "spdx": "BSD-3-Clause",
                "path": str(license_file),
                "sha256": license_sha256.lower(),
            },
            "source_url": source_url,
            "runtime": {
                "do_3D": True,
                "channels": [0, 0],
                "diameter_xy": self._diameter_xy,
                "anisotropy": self._anisotropy,
                "cellprob_threshold": self._cellprob_threshold,
                "flow3D_smooth": self._flow3d_smooth,
                "min_size": self._min_size,
                "device": device,
                "seed": seed,
            },
        }

    def segment(self, image_zyx: np.ndarray) -> np.ndarray:
        image = np.asarray(image_zyx)
        if image.ndim != 3 or not np.isfinite(image).all():
            raise ComputedSegmentationError("Cellpose input must be finite Z/Y/X")
        masks, _flows, _styles = self._model.eval(
            image,
            channels=[0, 0],
            channel_axis=None,
            z_axis=0,
            normalize={
                "normalize": True,
                "percentile": [1.0, 99.0],
                "norm3D": True,
            },
            diameter=self._diameter_xy,
            do_3D=True,
            anisotropy=self._anisotropy,
            cellprob_threshold=self._cellprob_threshold,
            flow3D_smooth=self._flow3d_smooth,
            min_size=self._min_size,
        )
        value = np.asarray(masks)
        if value.shape != image.shape or value.ndim != 3 or np.any(value < 0):
            raise ComputedSegmentationError("Cellpose returned malformed labels")
        return value.astype(np.uint32, copy=False)


def _trackastra_normalize_frame(intensity: np.ndarray) -> np.ndarray:
    """Mirror Trackastra's per-frame float32 p1/p99.8 normalization."""
    value = np.asarray(intensity, dtype=np.float32)
    low, high = np.percentile(value, (1.0, 99.8)).astype(np.float32)
    return (value - low) / (high - low + np.float32(1e-8))


def _label_properties(
    labels: np.ndarray,
    raw_intensity: np.ndarray,
    native_intensity: np.ndarray,
    scale_zyx: Sequence[float],
) -> dict[int, dict[str, Any]]:
    try:
        from skimage.measure import regionprops
    except ImportError as exc:
        raise ComputedSegmentationError("scikit-image is not installed") from exc
    labels = np.asarray(labels)
    raw_intensity = np.asarray(raw_intensity)
    native_intensity = np.asarray(native_intensity)
    if (
        labels.shape != raw_intensity.shape
        or labels.shape != native_intensity.shape
        or labels.ndim != 3
    ):
        raise ComputedSegmentationError("labels/intensity must have equal Z/Y/X shape")
    scale = np.asarray(scale_zyx, dtype=np.float64)
    result: dict[int, dict[str, Any]] = {}
    raw_by_label = {
        int(region.label): region
        for region in regionprops(labels, intensity_image=raw_intensity)
    }
    for region in regionprops(labels, intensity_image=native_intensity):
        coords = np.asarray(region.coords, dtype=np.float64)
        physical = coords * scale[None, :]
        covariance = (
            np.cov(physical, rowvar=False, bias=True)
            if len(physical) > 1
            else np.zeros((3, 3), dtype=np.float64)
        )
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        # A uniform solid ellipsoid has coordinate variance a^2 / 5.
        axes = 2.0 * np.sqrt(5.0 * eigenvalues)
        minor = max(float(axes[2]), np.finfo(np.float64).eps)
        raw_region = raw_by_label[int(region.label)]
        values = raw_intensity[tuple(np.asarray(region.coords).T)].astype(np.float64)
        try:
            solidity = float(region.solidity)
        except (AttributeError, NotImplementedError, ValueError) as exc:
            raise ComputedSegmentationError(
                "3-D convex-hull solidity is unavailable for a computed region"
            ) from exc
        inertia = np.asarray(region.inertia_tensor, dtype=np.float64)
        if inertia.shape != (3, 3):
            raise ComputedSegmentationError("3-D inertia tensor is malformed")
        edge_distance = np.minimum.reduce(
            (
                coords[:, 1],
                labels.shape[1] - 1 - coords[:, 1],
                coords[:, 2],
                labels.shape[2] - 1 - coords[:, 2],
            )
        )
        border_dist = float(np.max(1.0 - np.minimum(edge_distance / 5.0, 1.0)))
        native = np.concatenate(
            (
                np.asarray(
                    [region.equivalent_diameter_area, region.intensity_mean],
                    dtype=np.float64,
                ),
                inertia.reshape(-1),
                np.asarray([border_dist], dtype=np.float64),
            )
        )
        descriptive = np.asarray(
            (
                region.area,
                axes[0],
                axes[1],
                axes[2],
                axes[0] / minor,
                raw_region.intensity_mean,
                np.std(values),
                np.min(values),
                np.max(values),
                solidity,
                region.equivalent_diameter_area,
                border_dist,
            ),
            dtype=np.float64,
        )
        if (
            native.shape != (12,)
            or descriptive.shape != (12,)
            or not np.isfinite(native).all()
            or not np.isfinite(descriptive).all()
        ):
            raise ComputedSegmentationError("computed region features are non-finite")
        result[int(region.label)] = {
            "centroid": np.asarray(region.centroid, dtype=np.float64),
            "native": native.astype(np.float32),
            "descriptive": descriptive.astype(np.float32),
        }
    return result


def map_detections_to_regions(
    labels: np.ndarray,
    intensity: np.ndarray,
    coords_zyx: np.ndarray,
    *,
    scale_zyx: Sequence[float],
    max_nearest_distance_physical: float,
) -> dict[str, np.ndarray]:
    """Map points deterministically: containing label, then nearest centroid."""
    labels = np.asarray(labels)
    coords = np.asarray(coords_zyx, dtype=np.float64)
    properties = _label_properties(
        labels,
        intensity,
        _trackastra_normalize_frame(intensity),
        scale_zyx,
    )
    n = len(coords)
    native = np.full((n, 12), np.nan, dtype=np.float32)
    descriptive = np.full((n, 12), np.nan, dtype=np.float32)
    matched = np.zeros(n, dtype=np.bool_)
    mask_labels = np.zeros(n, dtype=np.uint32)
    match_mode = np.full(n, "unmatched", dtype="U16")
    distances = np.full(n, np.inf, dtype=np.float32)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    rounded = np.rint(coords).astype(np.int64)
    in_bounds = np.all((rounded >= 0) & (rounded < np.asarray(labels.shape)), axis=1)
    labels_sorted = np.asarray(sorted(properties), dtype=np.int64)
    centroids = (
        np.stack([properties[int(label)]["centroid"] for label in labels_sorted])
        if len(labels_sorted)
        else np.empty((0, 3), dtype=np.float64)
    )
    for index in range(n):
        chosen = 0
        mode = "unmatched"
        distance = math.inf
        if in_bounds[index]:
            chosen = int(labels[tuple(rounded[index])])
            if chosen in properties:
                mode = "contains"
                distance = 0.0
        if chosen == 0 and len(labels_sorted):
            delta = (centroids - coords[index][None, :]) * scale[None, :]
            candidate_distances = np.linalg.norm(delta, axis=1)
            minimum = float(np.min(candidate_distances))
            # labels_sorted makes equal-distance resolution deterministic.
            candidate = int(labels_sorted[int(np.argmin(candidate_distances))])
            if minimum <= max_nearest_distance_physical:
                chosen = candidate
                mode = "nearest_centroid"
                distance = minimum
        if chosen:
            prop = properties[chosen]
            native[index] = prop["native"]
            descriptive[index] = prop["descriptive"]
            matched[index] = True
            mask_labels[index] = chosen
            match_mode[index] = mode
            distances[index] = distance
    return {
        "features": native,
        "descriptive_regionprops": descriptive,
        "matched": matched,
        "mask_labels": mask_labels,
        "match_mode": match_mode,
        "match_distance_physical": distances,
    }


def build_video_features(
    *,
    image: Any,
    labels_by_time: Mapping[int, np.ndarray],
    node_ids: np.ndarray,
    timepoints: np.ndarray,
    coords_zyx: np.ndarray,
    scale_zyx: Sequence[float],
    max_nearest_distance_physical: float,
) -> dict[str, np.ndarray]:
    node_ids, timepoints, coords_zyx = validate_detections(
        node_ids, timepoints, coords_zyx
    )
    features = np.full((len(node_ids), 12), np.nan, np.float32)
    descriptive = np.full((len(node_ids), 12), np.nan, np.float32)
    matched = np.zeros(len(node_ids), np.bool_)
    mask_labels = np.zeros(len(node_ids), np.uint32)
    match_mode = np.full(len(node_ids), "unmatched", dtype="U16")
    distances = np.full(len(node_ids), np.inf, np.float32)
    for t in np.unique(timepoints):
        if int(t) not in labels_by_time:
            continue
        selection = np.flatnonzero(timepoints == t)
        mapped = map_detections_to_regions(
            labels_by_time[int(t)],
            np.asarray(image[int(t)]),
            coords_zyx[selection],
            scale_zyx=scale_zyx,
            max_nearest_distance_physical=max_nearest_distance_physical,
        )
        features[selection] = mapped["features"]
        descriptive[selection] = mapped["descriptive_regionprops"]
        matched[selection] = mapped["matched"]
        mask_labels[selection] = mapped["mask_labels"]
        match_mode[selection] = mapped["match_mode"]
        distances[selection] = mapped["match_distance_physical"]
    return {
        "node_ids": node_ids,
        "timepoints": timepoints,
        "coords_zyx": coords_zyx,
        "features": features,
        "descriptive_regionprops": descriptive,
        "matched": matched,
        "mask_labels": mask_labels,
        "match_mode": match_mode,
        "match_distance_physical": distances,
    }


def validate_checkpoint_payload(payload: Mapping[str, np.ndarray]) -> None:
    node_ids = np.asarray(payload["node_ids"])
    features = np.asarray(payload["features"])
    descriptive = np.asarray(payload["descriptive_regionprops"])
    matched = np.asarray(payload["matched"])
    if (
        features.shape != (len(node_ids), 12)
        or descriptive.shape != (len(node_ids), 12)
        or matched.shape != (len(node_ids),)
        or not np.all(matched)
        or not np.isfinite(features).all()
        or not np.isfinite(descriptive).all()
        or np.any(np.all(features == 0, axis=1))
    ):
        raise ComputedSegmentationError(
            "payload is incomplete; Trackastra requires every detection matched "
            "to finite, nonzero checkpoint-native wrfeat"
        )


def publish_payload(
    output: Path,
    *,
    payload: Mapping[str, np.ndarray],
    identity: str,
    zarr_path: Path,
    scale_zyx: Sequence[float],
    segmenter_provenance: Mapping[str, Any],
    max_nearest_distance_physical: float = 30.0,
    label_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_checkpoint_payload(payload)
    arrays: dict[str, Any] = dict(payload)
    arrays.update(
        {
            "schema": np.asarray(SCHEMA),
            "provenance_label": np.asarray(LABEL),
            "authentic_masks": np.asarray(False),
            "organizer_provided": np.asarray(False),
            "h3_authentic_masks": np.asarray(False),
            "identity": np.asarray(identity),
            "feature_names": np.asarray(TRACKASTRA_NATIVE_FEATURE_NAMES),
            "feature_schema": np.asarray("trackastra-ctc-wrfeat12.v1"),
            "descriptive_feature_names": np.asarray(DESCRIPTIVE_FEATURE_NAMES),
            "scale_zyx": np.asarray(scale_zyx, dtype=np.float64),
        }
    )
    _atomic_npz(output, arrays)
    modes, counts = np.unique(payload["match_mode"], return_counts=True)
    duplicate_mask_assignments = 0
    for t in np.unique(payload["timepoints"]):
        labels_at_t = np.asarray(payload["mask_labels"])[
            np.asarray(payload["timepoints"]) == t
        ]
        duplicate_mask_assignments += int(
            len(labels_at_t) - len(np.unique(labels_at_t))
        )
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS_COMPLETE_COMPUTED_SEGMENTATION_FEATURES",
        "provenance_label": LABEL,
        "explicit_nonclaims": {
            "authentic_masks": False,
            "organizer_provided": False,
            "h3_authentic_masks": False,
        },
        "identity": identity,
        "source_zarr": str(zarr_path),
        "scale_zyx": list(map(float, scale_zyx)),
        "feature_contract": {
            "checkpoint_matrix_key": "features",
            "checkpoint_matrix_alias": "trackastra_native_wrfeat12",
            "checkpoint_feature_names": list(TRACKASTRA_NATIVE_FEATURE_NAMES),
            "checkpoint_feature_schema": "trackastra-ctc-wrfeat12.v1",
            "checkpoint_intensity_normalization": (
                "per-frame float32 p1/p99.8, exactly Trackastra normalize(x)"
            ),
            "descriptive_matrix_key": "descriptive_regionprops",
            "descriptive_feature_names": list(DESCRIPTIVE_FEATURE_NAMES),
            "checkpoint_channels_overloaded": False,
        },
        "mapping": {
            "policy": "containing_label_else_nearest_physical_centroid",
            "max_nearest_distance_physical": float(
                max_nearest_distance_physical
            ),
            "detection_count": int(len(payload["node_ids"])),
            "matched_count": int(np.sum(payload["matched"])),
            "mode_counts": {str(k): int(v) for k, v in zip(modes, counts, strict=True)},
            "duplicate_mask_assignment_count_within_timepoint": (
                duplicate_mask_assignments
            ),
        },
        "segmenter": dict(segmenter_provenance),
        "computed_label_artifact": dict(label_artifact or {}),
        "artifact": {"path": str(output), "sha256": sha256(output)},
    }
    _atomic_json(output.with_suffix(".json"), sidecar)
    return sidecar


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    array, attrs = _zarr_array(args.zarr)
    if args.detections_npz:
        detections = load_detection_npz(args.detections_npz)
        detection_source = str(args.detections_npz)
    else:
        detections = load_bh_detections(args.cache_root, args.identity)
        detection_source = f"{args.cache_root}::{args.identity}"
    weight = args.weight
    license_file = args.license_file
    checks = {
        "zarr_import": importlib.util.find_spec("zarr") is not None,
        "skimage_import": importlib.util.find_spec("skimage") is not None,
        "cellpose_3_1_1_2_import": False,
        "weight_present": bool(weight and weight.is_file()),
        "weight_sha256_matches": False,
        "license_present": bool(license_file and license_file.is_file()),
        "license_sha256_matches": False,
    }
    runtime: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "cellpose_declared_requirements": [],
        "declared_bound_exception": None,
    }
    try:
        checks["cellpose_3_1_1_2_import"] = (
            importlib.metadata.version("cellpose") == "3.1.1.2"
        )
        requirements = importlib.metadata.requires("cellpose") or []
        runtime["cellpose_declared_requirements"] = requirements
        numpy_requirements = [
            item for item in requirements if item.lower().startswith("numpy")
        ]
        if runtime["numpy"] == "2.4.6" and any("<2.1" in item for item in numpy_requirements):
            runtime["declared_bound_exception"] = (
                "NumPy 2.4.6 is above Cellpose's declared numpy<2.1 bound; "
                "readiness here means authenticated import/inference smoke, not "
                "claim of upstream-supported dependency compatibility"
            )
    except importlib.metadata.PackageNotFoundError:
        pass
    if checks["weight_present"] and args.weight_sha256:
        checks["weight_sha256_matches"] = sha256(weight) == args.weight_sha256.lower()
    if checks["license_present"] and args.license_sha256:
        checks["license_sha256_matches"] = (
            sha256(license_file) == args.license_sha256.lower()
        )
    ready = all(checks.values())
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY" if ready else "BLOCKED_PRETRAINED_ASSET_OR_RUNTIME",
        "provenance_label": LABEL,
        "explicit_nonclaims": {
            "authentic_masks": False,
            "organizer_provided": False,
            "h3_authentic_masks": False,
        },
        "identity": args.identity,
        "raw": {
            "path": str(args.zarr),
            "shape_tzyx": list(map(int, array.shape)),
            "dtype": str(array.dtype),
            "scale_zyx": list(spatial_scale_zyx(attrs)),
        },
        "detections": {
            "source": detection_source,
            "count": int(len(detections[0])),
            "timepoint_min": int(np.min(detections[1])),
            "timepoint_max": int(np.max(detections[1])),
        },
        "backend": {
            "name": "cellpose-3d-cyto3",
            "pinned_package_version": "3.1.1.2",
            "source_url": args.source_url,
            "software_license_spdx": "BSD-3-Clause",
            "weight_path": str(weight) if weight else None,
            "weight_sha256_expected": args.weight_sha256,
            "license_path": str(license_file) if license_file else None,
            "license_sha256_expected": args.license_sha256,
        },
        "runtime": runtime,
        "checks": checks,
        "blockers": [key for key, passed in checks.items() if not passed],
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.preflight_report:
        _atomic_json(args.preflight_report, report)
    return report


def _load_segmenter(args: argparse.Namespace, scale: Sequence[float]) -> Cellpose3D:
    return Cellpose3D(
        weight=args.weight,
        weight_sha256=args.weight_sha256,
        license_file=args.license_file,
        license_sha256=args.license_sha256,
        source_url=args.source_url,
        diameter_xy=args.diameter_xy,
        anisotropy=args.anisotropy or scale[0] / scale[1],
        cellprob_threshold=args.cellprob_threshold,
        flow3d_smooth=args.flow3d_smooth,
        min_size=args.min_size,
        device=args.device,
        seed=args.seed,
    )


def pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Run exactly one real Zarr frame; never emit production-ready features."""
    started = time.perf_counter()
    array, attrs = _zarr_array(args.zarr)
    scale = spatial_scale_zyx(attrs)
    t = int(args.pilot_timepoint)
    if t < 0 or t >= int(array.shape[0]):
        raise ComputedSegmentationError(f"pilot timepoint {t} is out of bounds")
    segmenter = _load_segmenter(args, scale)
    frame = np.asarray(array[t])
    labels = segmenter.segment(frame)
    if args.detections_npz:
        node_ids, timepoints, coords = load_detection_npz(args.detections_npz)
    else:
        node_ids, timepoints, coords = load_bh_detections(
            args.cache_root, args.identity
        )
    selection = np.flatnonzero(timepoints == t)
    mapped = map_detections_to_regions(
        labels,
        frame,
        coords[selection],
        scale_zyx=scale,
        max_nearest_distance_physical=args.max_nearest_distance_physical,
    )
    mask_count = int(len(np.unique(labels)) - int(np.any(labels == 0)))
    labels_sha256 = hashlib.sha256(
        labels.astype("<u4", copy=False).tobytes(order="C")
    ).hexdigest()
    if args.pilot_labels:
        _atomic_npz(
            args.pilot_labels,
            {
                "schema": np.asarray(
                    "celltrack-alpha.g04-computed-segmentation-pilot-labels.v1"
                ),
                "coverage": np.asarray("PILOT_PARTIAL_ONE_FRAME"),
                "provenance_label": np.asarray(LABEL),
                "authentic_masks": np.asarray(False),
                "organizer_provided": np.asarray(False),
                "h3_authentic_masks": np.asarray(False),
                "identity": np.asarray(args.identity),
                "timepoint": np.asarray(t, dtype=np.int32),
                "labels_zyx": labels,
                "labels_logical_sha256": np.asarray(labels_sha256),
            },
        )
    report = {
        "schema": "celltrack-alpha.g04-computed-segmentation-pilot.v1",
        "status": (
            "PASS_INTERFACE_NONEMPTY_COMPUTED_SEGMENTATION"
            if mask_count > 0
            else "FAIL_EMPTY_COMPUTED_SEGMENTATION"
        ),
        "coverage": "PILOT_PARTIAL_ONE_FRAME",
        "production_manifest_eligible": False,
        "provenance_label": LABEL,
        "explicit_nonclaims": {
            "authentic_masks": False,
            "organizer_provided": False,
            "h3_authentic_masks": False,
        },
        "identity": args.identity,
        "timepoint": t,
        "source_zarr": str(args.zarr),
        "frame": {
            "shape_zyx": list(map(int, frame.shape)),
            "dtype": str(frame.dtype),
            "raw_logical_sha256": hashlib.sha256(
                np.ascontiguousarray(frame).tobytes()
            ).hexdigest(),
            "scale_zyx": list(scale),
        },
        "computed_segmentation": {
            "mask_count": mask_count,
            "foreground_voxel_count": int(np.count_nonzero(labels)),
            "labels_logical_sha256": labels_sha256,
            "pilot_labels_path": str(args.pilot_labels) if args.pilot_labels else None,
            "pilot_labels_file_sha256": (
                sha256(args.pilot_labels) if args.pilot_labels else None
            ),
        },
        "detection_mapping_diagnostic": {
            "detection_count": int(len(selection)),
            "matched_count": int(np.sum(mapped["matched"])),
            "unmatched_count": int(np.sum(~mapped["matched"])),
            "matched_fraction": float(np.mean(mapped["matched"])) if len(selection) else 0.0,
            "all_matched": bool(np.all(mapped["matched"])) if len(selection) else False,
        },
        "segmenter": dict(segmenter.provenance),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.pilot_report:
        _atomic_json(args.pilot_report, report)
    return report


def _expected_identities(path: Path) -> tuple[str, ...]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComputedSegmentationError("cannot read expected identities") from exc
    rows = (
        value.get("identities", value.get("entries"))
        if isinstance(value, Mapping)
        else value
    )
    if not isinstance(rows, list):
        raise ComputedSegmentationError("expected identities must be a JSON list")
    identities = tuple(
        str(row["identity"] if isinstance(row, Mapping) else row) for row in rows
    )
    if not identities or len(set(identities)) != len(identities):
        raise ComputedSegmentationError("expected identities are empty or repeated")
    return identities


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    expected = _expected_identities(args.expected_identities_file)
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    for path in args.feature_npz:
        sidecar_path = path.with_suffix(".json")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            with np.load(path, allow_pickle=False) as data:
                identity = str(data["identity"].item())
                payload = {
                    key: np.asarray(data[key])
                    for key in (
                        "node_ids",
                        "features",
                        "descriptive_regionprops",
                        "matched",
                    )
                }
                valid_contract = (
                    data["provenance_label"].item() == LABEL
                    and data["authentic_masks"].item() is False
                    and data["organizer_provided"].item() is False
                    and data["feature_schema"].item()
                    == "trackastra-ctc-wrfeat12.v1"
                    and tuple(data["feature_names"].tolist())
                    == TRACKASTRA_NATIVE_FEATURE_NAMES
                )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ComputedSegmentationError(f"invalid feature artifact: {path}") from exc
        validate_checkpoint_payload(payload)
        digest = sha256(path)
        if (
            not valid_contract
            or identity in observed
            or sidecar.get("status")
            != "PASS_COMPLETE_COMPUTED_SEGMENTATION_FEATURES"
            or sidecar.get("artifact", {}).get("sha256") != digest
            or float(sidecar.get("mapping", {}).get(
                "max_nearest_distance_physical", -1.0
            ))
            != 30.0
        ):
            raise ComputedSegmentationError(
                f"feature artifact contract/provenance failed: {path}"
            )
        observed.add(identity)
        rows.append(
            {
                "identity": identity,
                "path": os.path.relpath(path, args.output.parent),
                "sha256": digest,
            }
        )
    if observed != set(expected):
        raise ComputedSegmentationError(
            "manifest coverage mismatch: missing="
            f"{sorted(set(expected) - observed)}, extra={sorted(observed - set(expected))}"
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "PASS_COMPLETE_PRODUCTION_COVERAGE",
        "coverage": "PRODUCTION_COMPLETE_ONLY",
        "partial_or_pilot_allowed": False,
        "provenance_label": LABEL,
        "authentic_masks": False,
        "organizer_provided": False,
        "feature_schema": "trackastra-ctc-wrfeat12.v1",
        "feature_names": list(TRACKASTRA_NATIVE_FEATURE_NAMES),
        "identities": sorted(rows, key=lambda row: row["identity"]),
        "expected_identities": list(expected),
    }
    _atomic_json(args.output, manifest)
    return manifest


def _completed_feature(path: Path) -> bool:
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    label_path = Path(str(sidecar.get("computed_label_artifact", {}).get("path", "")))
    return (
        sidecar.get("status") == "PASS_COMPLETE_COMPUTED_SEGMENTATION_FEATURES"
        and sidecar.get("artifact", {}).get("sha256") == sha256(path)
        and label_path.is_dir()
    )


def batch(args: argparse.Namespace) -> dict[str, Any]:
    """Kaggle-oriented, deadline-admitted, atomic per-video CV57 worker."""
    if not (1 <= args.deadline_seconds <= 39_600):
        raise ComputedSegmentationError("batch deadline must be in [1,39600] seconds")
    if args.seconds_per_frame_upper_bound <= 0:
        raise ComputedSegmentationError(
            "batch requires a positive CUDA-derived frame-time upper bound"
        )
    verify_file(args.population_manifest, args.population_sha256, "population manifest")
    population = json.loads(args.population_manifest.read_text(encoding="ascii"))
    entries = population.get("entries")
    scope = population.get("scope", {})
    if (
        not isinstance(entries, list)
        or len(entries) != 57
        or len({str(row.get("identity")) for row in entries}) != 57
        or scope.get("cv57_only") is not True
        or scope.get("family_counts") != {"44b6": 21, "6bba": 36}
    ):
        raise ComputedSegmentationError("batch population is not exact CV57")
    if not (0 <= args.shard_index < args.shard_count):
        raise ComputedSegmentationError("invalid shard index/count")
    selected = [
        row
        for index, row in enumerate(entries)
        if index % args.shard_count == args.shard_index
    ]
    args.batch_output.mkdir(parents=True, exist_ok=True)
    feature_root = args.batch_output / "features"
    label_root = args.batch_output / "computed_segmentations"
    feature_root.mkdir(exist_ok=True)
    label_root.mkdir(exist_ok=True)
    started = time.monotonic()
    absolute_deadline = started + float(args.deadline_seconds)
    state_path = args.batch_output / "BATCH_STATE.json"
    completed: list[dict[str, Any]] = []
    stopped_before: str | None = None
    for row in selected:
        identity = str(row["identity"])
        slug = identity.replace(":", "__", 1)
        output = feature_root / f"{slug}.npz"
        if _completed_feature(output):
            completed.append(
                {"identity": identity, "feature_path": str(output), "resumed": True}
            )
            continue
        dataset = identity.split(":", 1)[1]
        zarr_path = args.raw_root / "train" / f"{dataset}.zarr"
        image, _attrs = _zarr_array(zarr_path)
        projected_seconds = (
            int(image.shape[0]) * float(args.seconds_per_frame_upper_bound)
            + float(args.per_video_finalize_margin_seconds)
        )
        remaining = absolute_deadline - time.monotonic()
        if remaining < projected_seconds:
            stopped_before = identity
            break
        video_args = argparse.Namespace(**vars(args))
        video_args.zarr = zarr_path
        video_args.identity = identity
        video_args.output = output
        video_args.labels_output = label_root / f"{slug}.zarr"
        video_args.absolute_deadline_monotonic = absolute_deadline
        video_args.frame_deadline_seconds = float(args.seconds_per_frame_upper_bound)
        sidecar = run(video_args)
        completed.append(
            {
                "identity": identity,
                "feature_path": str(output),
                "feature_sha256": sidecar["artifact"]["sha256"],
                "resumed": False,
            }
        )
        _atomic_json(
            state_path,
            {
                "schema": "celltrack-alpha.g04-computed-segmentation-batch-state.v1",
                "status": "RUNNING_ATOMIC_VIDEO_BOUNDARIES",
                "production_manifest_eligible": False,
                "population_sha256": args.population_sha256,
                "shard": {"index": args.shard_index, "count": args.shard_count},
                "completed": completed,
            },
        )
    selected_identities = [str(row["identity"]) for row in selected]
    completed_identities = {str(row["identity"]) for row in completed}
    shard_complete = completed_identities == set(selected_identities)
    report = {
        "schema": "celltrack-alpha.g04-computed-segmentation-cv57-batch.v1",
        "status": (
            "PASS_ATOMIC_SHARD_COMPLETE"
            if shard_complete
            else "STOPPED_BEFORE_UNADMITTABLE_VIDEO"
        ),
        "production_manifest_eligible": False,
        "manifest_rule": (
            "Only the separate exact-57 manifest command may promote after all "
            "shards are downloaded and authenticated"
        ),
        "population": {
            "path": str(args.population_manifest),
            "sha256": args.population_sha256,
            "identity_count": 57,
            "family_counts": {"44b6": 21, "6bba": 36},
        },
        "shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "identities": selected_identities,
            "complete": shard_complete,
        },
        "runtime_guard": {
            "deadline_seconds": args.deadline_seconds,
            "seconds_per_frame_upper_bound": args.seconds_per_frame_upper_bound,
            "per_video_finalize_margin_seconds": args.per_video_finalize_margin_seconds,
            "stopped_before_identity": stopped_before,
        },
        "completed": completed,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(state_path, report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    array, attrs = _zarr_array(args.zarr)
    scale = spatial_scale_zyx(attrs)
    if args.detections_npz:
        node_ids, timepoints, coords = load_detection_npz(args.detections_npz)
    else:
        node_ids, timepoints, coords = load_bh_detections(
            args.cache_root, args.identity
        )
    segmenter = _load_segmenter(args, scale)
    label_output = args.labels_output or args.output.with_name(
        f"{args.output.stem}.computed_segmentations.zarr"
    )
    if label_output.exists():
        raise ComputedSegmentationError(
            f"computed-label destination already exists: {label_output}"
        )
    label_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise ComputedSegmentationError("zarr is not installed") from exc
    payload: dict[str, np.ndarray] = {
        "node_ids": node_ids,
        "timepoints": timepoints,
        "coords_zyx": coords,
        "features": np.full((len(node_ids), 12), np.nan, np.float32),
        "descriptive_regionprops": np.full((len(node_ids), 12), np.nan, np.float32),
        "matched": np.zeros(len(node_ids), np.bool_),
        "mask_labels": np.zeros(len(node_ids), np.uint32),
        "match_mode": np.full(len(node_ids), "unmatched", dtype="U16"),
        "match_distance_physical": np.full(len(node_ids), np.inf, np.float32),
    }
    logical_digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(
        dir=label_output.parent, prefix=f".{label_output.name}."
    ) as temporary_root:
        temporary_path = Path(temporary_root) / "labels.zarr"
        group = zarr.open_group(str(temporary_path), mode="w")
        raw_chunks = getattr(array, "chunks", None)
        spatial_chunks = (
            tuple(int(item) for item in raw_chunks[-3:])
            if raw_chunks and len(raw_chunks) == 4
            else tuple(int(item) for item in array.shape[-3:])
        )
        label_array = group.create_array(
            "labels",
            shape=tuple(int(item) for item in array.shape),
            chunks=(1, *spatial_chunks),
            dtype=np.uint32,
        )
        group.attrs.update(
            {
                "schema": "celltrack-alpha.g04-computed-segmentations-zarr.v1",
                "provenance_label": LABEL,
                "authentic_masks": False,
                "organizer_provided": False,
                "h3_authentic_masks": False,
                "identity": args.identity,
                "source_zarr": str(args.zarr),
                "scale_zyx": list(scale),
                "segmenter": dict(segmenter.provenance),
            }
        )
        for t in range(int(array.shape[0])):
            deadline = getattr(args, "absolute_deadline_monotonic", None)
            frame_bound = float(getattr(args, "frame_deadline_seconds", 0.0))
            if deadline is not None and time.monotonic() + frame_bound >= deadline:
                raise ComputedSegmentationError(
                    "hard deadline guard stopped before starting another frame"
                )
            frame = np.asarray(array[t])
            labels = segmenter.segment(frame)
            label_array[t] = labels
            logical_digest.update(np.asarray(t, dtype="<i8").tobytes())
            logical_digest.update(labels.astype("<u4", copy=False).tobytes(order="C"))
            selection = np.flatnonzero(timepoints == t)
            if len(selection):
                mapped = map_detections_to_regions(
                    labels,
                    frame,
                    coords[selection],
                    scale_zyx=scale,
                    max_nearest_distance_physical=args.max_nearest_distance_physical,
                )
                for key in (
                    "features",
                    "descriptive_regionprops",
                    "matched",
                    "mask_labels",
                    "match_mode",
                    "match_distance_physical",
                ):
                    payload[key][selection] = mapped[key]
        validate_checkpoint_payload(payload)
        os.replace(temporary_path, label_output)
    label_artifact = {
        "path": str(label_output),
        "schema": "celltrack-alpha.g04-computed-segmentations-zarr.v1",
        "logical_sha256_definition": (
            "sha256(concat(little-endian int64 timepoint, little-endian "
            "uint32 C-order label bytes) in ascending processed timepoint order)"
        ),
        "logical_sha256": logical_digest.hexdigest(),
        "processed_timepoints": list(range(int(array.shape[0]))),
    }
    return publish_payload(
        args.output,
        payload=payload,
        identity=args.identity,
        zarr_path=args.zarr,
        scale_zyx=scale,
        segmenter_provenance=segmenter.provenance,
        max_nearest_distance_physical=args.max_nearest_distance_physical,
        label_artifact=label_artifact,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command", choices=("preflight", "pilot", "run", "batch", "manifest")
    )
    value.add_argument("--zarr", type=Path)
    value.add_argument("--identity")
    detections = value.add_mutually_exclusive_group(required=False)
    detections.add_argument("--detections-npz", type=Path)
    detections.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "reports/g02/fitcache_117/bh9out/cache",
    )
    value.add_argument("--output", type=Path)
    value.add_argument("--labels-output", type=Path)
    value.add_argument("--preflight-report", type=Path)
    value.add_argument("--pilot-timepoint", type=int, default=0)
    value.add_argument("--pilot-report", type=Path)
    value.add_argument("--pilot-labels", type=Path)
    value.add_argument("--feature-npz", type=Path, action="append", default=[])
    value.add_argument("--expected-identities-file", type=Path)
    value.add_argument("--population-manifest", type=Path)
    value.add_argument("--population-sha256")
    value.add_argument("--raw-root", type=Path)
    value.add_argument("--batch-output", type=Path)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--shard-count", type=int, default=1)
    value.add_argument("--deadline-seconds", type=int, default=37_800)
    value.add_argument("--seconds-per-frame-upper-bound", type=float, default=0.0)
    value.add_argument("--per-video-finalize-margin-seconds", type=float, default=600.0)
    value.add_argument("--weight", type=Path)
    value.add_argument("--weight-sha256")
    value.add_argument("--license-file", type=Path)
    value.add_argument("--license-sha256")
    value.add_argument(
        "--source-url",
        default="https://github.com/MouseLand/cellpose/tree/v3.1.1.2",
    )
    value.add_argument("--diameter-xy", type=float, default=30.0)
    value.add_argument("--anisotropy", type=float)
    value.add_argument("--cellprob-threshold", type=float, default=0.0)
    value.add_argument("--flow3d-smooth", type=float, default=0.0)
    value.add_argument("--min-size", type=int, default=15)
    value.add_argument("--max-nearest-distance-physical", type=float, default=30.0)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--seed", type=int, default=20260803)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "manifest":
        required = {
            "--output": args.output,
            "--feature-npz": args.feature_npz,
            "--expected-identities-file": args.expected_identities_file,
        }
    elif args.command == "batch":
        required = {
            "--population-manifest": args.population_manifest,
            "--population-sha256": args.population_sha256,
            "--raw-root": args.raw_root,
            "--cache-root": args.cache_root,
            "--batch-output": args.batch_output,
            "--seconds-per-frame-upper-bound": args.seconds_per_frame_upper_bound,
        }
    else:
        required = {"--zarr": args.zarr, "--identity": args.identity}
    if args.command in {"pilot", "run", "batch"}:
        required.update(
            {
                "--weight": args.weight,
                "--weight-sha256": args.weight_sha256,
                "--license-file": args.license_file,
                "--license-sha256": args.license_sha256,
            }
        )
    if args.command == "run":
        required["--output"] = args.output
    missing = [name for name, item in required.items() if not item]
    if missing:
        raise ComputedSegmentationError(
            f"{args.command} missing required inputs: " + ", ".join(missing)
        )
    if args.command == "run":
        report = run(args)
    elif args.command == "pilot":
        report = pilot(args)
    elif args.command == "batch":
        report = batch(args)
    elif args.command == "manifest":
        report = build_manifest(args)
    else:
        report = preflight(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("status") in {
        "READY",
        "PASS_COMPLETE_COMPUTED_SEGMENTATION_FEATURES",
        "PASS_INTERFACE_NONEMPTY_COMPUTED_SEGMENTATION",
        "PASS_COMPLETE_PRODUCTION_COVERAGE",
        "PASS_ATOMIC_SHARD_COMPLETE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
