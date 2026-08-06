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
import gc
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
SCHEMA = "celltrack-alpha.g04-computed-segmentation-features.v2"
SIDECAR_SCHEMA = "celltrack-alpha.g04-computed-segmentation-sidecar.v2"
PREFLIGHT_SCHEMA = "celltrack-alpha.g04-computed-segmentation-preflight.v1"
MANIFEST_SCHEMA = "celltrack-alpha.g04-computed-segmentation-wrfeat-manifest.v1"
CROP_SCHEMA = "celltrack-alpha.g04-crop-computed-segmentation-wrfeat-pilot.v1"
CROP_SHAPE_ZYX = (16, 32, 32)
CROP_PRIMARY_SHAPE_ZYX = (32, 64, 64)
CROP_FALLBACK_SHAPE_ZYX = (48, 96, 96)
CROP_GPU_SCHEMA = "celltrack-alpha.g04-crop-computed-segmentation-wrfeat-pilot.v2"
CROP_GEOMETRY_SCHEMA = "celltrack-alpha.g04-crop-geometry-audit.v1"
CROP_GEOMETRY_MARGIN_UM = 2.0
CROP_SHAPE_LADDER_Z = (16, 24, 32, 48, 64)
CROP_SHAPE_LADDER_YX = (32, 48, 64, 80, 96, 128, 160, 192, 256)

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
    "solidity_voxel_cell_convex_hull_3d",
    "equivalent_diameter_area",
    "border_dist",
)

DESCRIPTIVE_FEATURE_SCHEMA = (
    "celltrack-alpha.g04-computed-morphology12.voxel-cell-solidity.v2"
)
VOXEL_CELL_SOLIDITY_DEFINITION = (
    "foreground voxel-cell volume divided by the physical 3-D convex-hull "
    "volume of every voxel's eight half-voxel-offset corners"
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
        self._device = device
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

    def segment_batch(
        self,
        images_bzyx: np.ndarray,
        *,
        network_batch_size: int = 64,
    ) -> np.ndarray:
        """Run genuine cross-crop 3-D plane batching through frozen Cellpose.

        Cellpose 3.1.1.2's public list/5-D ``eval`` path recursively loops over
        volumes.  Here corresponding YX/ZY/ZX planes from every crop share the
        same ``run_net`` calls, then unchanged 3-D dynamics are applied to each
        split flow field.
        """

        images = np.asarray(images_bzyx)
        if (
            images.ndim != 4
            or len(images) == 0
            or not np.isfinite(images).all()
            or type(network_batch_size) is not int
            or network_batch_size < 1
        ):
            raise ComputedSegmentationError("batched Cellpose input is malformed")
        if self._device != "cuda":
            raise ComputedSegmentationError(
                "genuine cross-detection Cellpose batching is CUDA-only"
            )
        try:
            import torch
            from cellpose import core, transforms
        except ImportError as exc:  # pragma: no cover - dependency preflight
            raise ComputedSegmentationError("Cellpose batch dependencies are absent") from exc
        if not torch.cuda.is_available():
            raise ComputedSegmentationError("CUDA disappeared before batched segmentation")

        prepared: list[np.ndarray] = []
        original_shapes: list[tuple[int, int, int]] = []
        rescale = float(self._model.diam_mean / self._diameter_xy)
        normalize_params = {
            "normalize": True,
            "percentile": [1.0, 99.0],
            "norm3D": True,
        }
        for image in images:
            converted = transforms.convert_image(
                image,
                channels=[0, 0],
                channel_axis=None,
                z_axis=0,
                do_3D=True,
                nchan=self._model.nchan,
            )
            if converted.ndim < 4:
                converted = converted[np.newaxis, ...]
            converted = np.asarray(converted)
            if converted.ndim != 4:
                raise ComputedSegmentationError("Cellpose crop conversion is not Z/Y/X/C")
            original_shapes.append(tuple(int(value) for value in converted.shape[:3]))
            converted = transforms.normalize_img(converted, **normalize_params)
            z_size, y_size, x_size = converted.shape[:3]
            if rescale != 1.0:
                converted = transforms.resize_image(
                    converted, Ly=int(y_size * rescale), Lx=int(x_size * rescale)
                )
            converted = transforms.resize_image(
                converted.transpose(1, 0, 2, 3),
                Ly=int(z_size * self._anisotropy * rescale),
                Lx=int(x_size * rescale),
            ).transpose(1, 0, 2, 3)
            prepared.append(np.ascontiguousarray(converted, dtype=np.float32))
        prepared_shapes = {value.shape for value in prepared}
        if len(prepared_shapes) != 1:
            raise ComputedSegmentationError("batched Cellpose crops changed shape unequally")

        batch = np.stack(prepared, axis=0)
        spatial_shape = batch.shape[1:4]
        flow_batch = np.zeros((len(batch), *spatial_shape, 4), dtype=np.float32)
        permutations = ((0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3))
        inverse = ((0, 1, 2), (1, 0, 2), (1, 2, 0))
        components = ((1, 2), (0, 2), (0, 1))
        for orientation in range(3):
            planes = [value.transpose(permutations[orientation]) for value in prepared]
            plane_count = planes[0].shape[0]
            joined = np.concatenate(planes, axis=0)
            network = (
                self._model.net
                if orientation == 0 or self._model.net_ortho is None
                else self._model.net_ortho
            )
            output, _styles = core.run_net(
                network,
                joined,
                batch_size=network_batch_size,
                augment=False,
                bsize=224,
                tile_overlap=0.1,
                rsz=None,
            )
            if output.shape[0] != len(batch) * plane_count:
                raise ComputedSegmentationError("batched Cellpose plane split drifted")
            for crop_index in range(len(batch)):
                local = output[
                    crop_index * plane_count : (crop_index + 1) * plane_count
                ]
                flow_batch[crop_index, ..., -1] += local[..., -1].transpose(
                    inverse[orientation]
                )
                for component_index in range(2):
                    flow_batch[
                        crop_index, ..., components[orientation][component_index]
                    ] += local[..., component_index].transpose(inverse[orientation])
            del joined, output

        masks: list[np.ndarray] = []
        for crop_index, (flow, original_shape) in enumerate(
            zip(flow_batch, original_shapes, strict=True)
        ):
            z_size, y_size, x_size = original_shape
            if rescale != 1.0:
                flow = transforms.resize_image(flow, Ly=y_size, Lx=x_size)
            if z_size != flow.shape[0]:
                flow = transforms.resize_image(
                    flow.transpose(1, 0, 2, 3), Ly=z_size, Lx=x_size
                ).transpose(1, 0, 2, 3)
            cellprob = flow[..., -1]
            d_p = flow[..., :-1].transpose((3, 0, 1, 2))
            if self._flow3d_smooth > 0:
                from scipy.ndimage import gaussian_filter

                d_p = gaussian_filter(
                    d_p,
                    (0, self._flow3d_smooth, self._flow3d_smooth, self._flow3d_smooth),
                )
            mask = self._model._compute_masks(
                (*original_shape, self._model.nchan),
                d_p,
                cellprob,
                flow_threshold=0.4,
                cellprob_threshold=self._cellprob_threshold,
                interp=True,
                min_size=self._min_size,
                max_size_fraction=0.4,
                niter=200.0 / rescale,
                do_3D=True,
                stitch_threshold=0.0,
            )
            value = np.asarray(mask)
            if value.shape != original_shape or np.any(value < 0):
                raise ComputedSegmentationError(
                    f"batched Cellpose mask {crop_index} is malformed"
                )
            masks.append(value.astype(np.uint32, copy=False))
        torch.cuda.empty_cache()
        gc.collect()
        return np.stack(masks, axis=0)


def _trackastra_normalize_frame(intensity: np.ndarray) -> np.ndarray:
    """Mirror Trackastra's per-frame float32 p1/p99.8 normalization."""
    value = np.asarray(intensity, dtype=np.float32)
    low, high = np.percentile(value, (1.0, 99.8)).astype(np.float32)
    return (value - low) / (high - low + np.float32(1e-8))


def _voxel_cell_convex_hull_solidity(
    coords_zyx: np.ndarray,
    scale_zyx: Sequence[float],
) -> float:
    """Return finite 3-D solidity for the union of closed voxel cells.

    ``skimage.regionprops.solidity`` constructs a convex hull from voxel
    centers.  A legitimate one-slice or one-line 3-D object therefore has a
    zero-volume center hull and scikit-image 0.26 reports ``inf``.  Treating a
    voxel as the unit cell it represents makes every nonempty object
    full-dimensional without inventing a numeric sentinel: the numerator is
    the union volume of its cells and the denominator is their 3-D convex-hull
    volume.  The physical spacing is applied to both terms.
    """
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise ComputedSegmentationError("scipy is not installed") from exc
    coords = np.asarray(coords_zyx, dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    if (
        coords.ndim != 2
        or coords.shape[1:] != (3,)
        or len(coords) == 0
        or scale.shape != (3,)
        or not np.isfinite(coords).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ComputedSegmentationError(
            "voxel-cell solidity requires nonempty finite Z/Y/X coordinates "
            "and positive finite spacing"
        )
    offsets = np.asarray(
        (
            (-0.5, -0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, 0.5, 0.5),
            (0.5, -0.5, -0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, -0.5),
            (0.5, 0.5, 0.5),
        ),
        dtype=np.float64,
    )
    corners = ((coords[:, None, :] + offsets[None, :, :]) * scale).reshape(-1, 3)
    corners = np.unique(corners, axis=0)
    try:
        hull_volume = float(ConvexHull(corners).volume)
    except QhullError as exc:  # pragma: no cover - voxel corners span 3-D
        raise ComputedSegmentationError(
            "voxel-cell convex hull unexpectedly failed for a nonempty region"
        ) from exc
    cell_volume = float(np.prod(scale, dtype=np.float64))
    ratio = float(len(coords) * cell_volume / hull_volume)
    if not math.isfinite(hull_volume) or hull_volume <= 0 or not math.isfinite(ratio):
        raise ComputedSegmentationError(
            "voxel-cell convex-hull solidity is non-finite"
        )
    # The voxel union is a subset of its convex hull.  Permit only negligible
    # Qhull roundoff above one; this is not a missing-value substitution.
    if ratio <= 0 or ratio > 1.0 + 1e-9:
        raise ComputedSegmentationError(
            f"voxel-cell convex-hull solidity is outside (0,1]: {ratio!r}"
        )
    return min(ratio, 1.0)


def _nonfinite_feature_error(
    *,
    label: int,
    voxel_count: int,
    affine_dimension: int,
    centroid: np.ndarray,
    native: np.ndarray,
    descriptive: np.ndarray,
) -> ComputedSegmentationError:
    fields: list[str] = []
    values: list[str] = []
    for prefix, names, vector in (
        ("centroid", ("z", "y", "x"), centroid),
        ("native", TRACKASTRA_NATIVE_FEATURE_NAMES, native),
        ("descriptive", DESCRIPTIVE_FEATURE_NAMES, descriptive),
    ):
        for name, item in zip(names, np.asarray(vector).reshape(-1), strict=True):
            if not np.isfinite(item):
                fields.append(f"{prefix}.{name}")
                values.append(repr(float(item)))
    return ComputedSegmentationError(
        "computed region features are non-finite: "
        f"label={label}; voxel_count={voxel_count}; "
        f"center_affine_dimension={affine_dimension}; "
        f"fields={fields}; values={values}"
    )


def _label_properties(
    labels: np.ndarray,
    raw_intensity: np.ndarray,
    native_intensity: np.ndarray,
    scale_zyx: Sequence[float],
    *,
    coordinate_offset_zyx: Sequence[int] = (0, 0, 0),
    frame_shape_zyx: Sequence[int] | None = None,
    native_only: bool = False,
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
    offset = np.asarray(coordinate_offset_zyx, dtype=np.float64)
    frame_shape = np.asarray(
        labels.shape if frame_shape_zyx is None else frame_shape_zyx,
        dtype=np.int64,
    )
    if (
        offset.shape != (3,)
        or not np.isfinite(offset).all()
        or frame_shape.shape != (3,)
        or np.any(frame_shape <= 0)
    ):
        raise ComputedSegmentationError("invalid region coordinate frame")
    result: dict[int, dict[str, Any]] = {}
    raw_by_label = (
        {
            int(region.label): region
            for region in regionprops(labels, intensity_image=raw_intensity)
        }
        if not native_only
        else {}
    )
    for region in regionprops(labels, intensity_image=native_intensity):
        local_coords = np.asarray(region.coords, dtype=np.float64)
        coords = local_coords + offset[None, :]
        inertia = np.asarray(region.inertia_tensor, dtype=np.float64)
        if inertia.shape != (3, 3):
            raise ComputedSegmentationError("3-D inertia tensor is malformed")
        edge_distance = np.minimum.reduce(
            (
                coords[:, 1],
                frame_shape[1] - 1 - coords[:, 1],
                coords[:, 2],
                frame_shape[2] - 1 - coords[:, 2],
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
        centroid = np.asarray(region.centroid, dtype=np.float64) + offset
        if native_only:
            # The crop lane consumes only Trackastra's checkpoint-native wrfeat.
            # Avoid the physical-axis covariance, raw-intensity summaries, affine
            # rank and (especially) 3-D Qhull solidity used by the separate
            # descriptive audit.  Including them would turn the GPU timing pilot
            # back into a CPU geometry benchmark.
            descriptive = np.full(12, np.nan, dtype=np.float64)
            affine_dimension = -1
            solidity_method = "not_computed_native_only"
        else:
            physical = coords * scale[None, :]
            covariance = (
                np.cov(physical, rowvar=False, bias=True)
                if len(physical) > 1
                else np.zeros((3, 3), dtype=np.float64)
            )
            eigenvalues = np.sort(
                np.maximum(np.linalg.eigvalsh(covariance), 0.0)
            )[::-1]
            # A uniform solid ellipsoid has coordinate variance a^2 / 5.
            axes = 2.0 * np.sqrt(5.0 * eigenvalues)
            minor = max(float(axes[2]), np.finfo(np.float64).eps)
            raw_region = raw_by_label[int(region.label)]
            values = raw_intensity[
                tuple(np.asarray(region.coords).T)
            ].astype(np.float64)
            centered = coords - np.mean(coords, axis=0, keepdims=True)
            affine_dimension = int(np.linalg.matrix_rank(centered))
            solidity = _voxel_cell_convex_hull_solidity(coords, scale)
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
            solidity_method = "voxel_cell_convex_hull_3d"
        if (
            native.shape != (12,)
            or descriptive.shape != (12,)
            or centroid.shape != (3,)
            or not np.isfinite(centroid).all()
            or not np.isfinite(native).all()
            or (not native_only and not np.isfinite(descriptive).all())
        ):
            raise _nonfinite_feature_error(
                label=int(region.label),
                voxel_count=int(region.area),
                affine_dimension=affine_dimension,
                centroid=centroid,
                native=native,
                descriptive=descriptive,
            )
        result[int(region.label)] = {
            "centroid": centroid,
            "native": native.astype(np.float32),
            "descriptive": descriptive.astype(np.float32),
            "center_affine_dimension": affine_dimension,
            "solidity_method": solidity_method,
        }
    return result


def map_detections_to_regions(
    labels: np.ndarray,
    intensity: np.ndarray,
    coords_zyx: np.ndarray,
    *,
    scale_zyx: Sequence[float],
    max_nearest_distance_physical: float,
    coordinate_offset_zyx: Sequence[int] = (0, 0, 0),
    frame_shape_zyx: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Map points deterministically: containing label, then nearest centroid."""
    labels = np.asarray(labels)
    coords = np.asarray(coords_zyx, dtype=np.float64)
    properties = _label_properties(
        labels,
        intensity,
        _trackastra_normalize_frame(intensity),
        scale_zyx,
        coordinate_offset_zyx=coordinate_offset_zyx,
        frame_shape_zyx=frame_shape_zyx,
    )
    n = len(coords)
    native = np.full((n, 12), np.nan, dtype=np.float32)
    descriptive = np.full((n, 12), np.nan, dtype=np.float32)
    affine_dimensions = np.full(n, -1, dtype=np.int8)
    matched = np.zeros(n, dtype=np.bool_)
    mask_labels = np.zeros(n, dtype=np.uint32)
    match_mode = np.full(n, "unmatched", dtype="U16")
    distances = np.full(n, np.inf, dtype=np.float32)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    offset = np.asarray(coordinate_offset_zyx, dtype=np.int64)
    if offset.shape != (3,):
        raise ComputedSegmentationError("invalid detection coordinate offset")
    rounded = np.rint(coords).astype(np.int64) - offset[None, :]
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
            affine_dimensions[index] = prop["center_affine_dimension"]
            matched[index] = True
            mask_labels[index] = chosen
            match_mode[index] = mode
            distances[index] = distance
    return {
        "features": native,
        "descriptive_regionprops": descriptive,
        "computed_object_center_affine_dimension": affine_dimensions,
        "matched": matched,
        "mask_labels": mask_labels,
        "match_mode": match_mode,
        "match_distance_physical": distances,
    }


def centered_crop(
    image_zyx: np.ndarray,
    center_zyx: Sequence[float],
    *,
    shape_zyx: Sequence[int] = CROP_SHAPE_ZYX,
) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    """Extract a fixed zero-padded crop and retain its full-frame origin."""

    image = np.asarray(image_zyx)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    center = np.rint(np.asarray(center_zyx, dtype=np.float64)).astype(np.int64)
    if (
        image.ndim != 3
        or shape.shape != (3,)
        or np.any(shape <= 0)
        or np.any(shape % 2 != 0)
        or center.shape != (3,)
    ):
        raise ComputedSegmentationError("invalid detection-centred crop contract")
    origin = center - shape // 2
    end = origin + shape
    source_start = np.maximum(origin, 0)
    source_end = np.minimum(end, np.asarray(image.shape, dtype=np.int64))
    target_start = source_start - origin
    target_end = target_start + (source_end - source_start)
    crop = np.zeros(tuple(int(value) for value in shape), dtype=image.dtype)
    if np.all(source_end > source_start):
        source_slice = tuple(
            slice(int(left), int(right))
            for left, right in zip(source_start, source_end, strict=True)
        )
        target_slice = tuple(
            slice(int(left), int(right))
            for left, right in zip(target_start, target_end, strict=True)
        )
        crop[target_slice] = image[source_slice]
    local_center = center - origin
    return (
        crop,
        tuple(int(value) for value in origin),
        tuple(int(value) for value in local_center),
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ComputedSegmentationError("cannot summarize an empty/nonfinite distribution")
    return {
        name: float(np.quantile(array, point))
        for name, point in (
            ("p00", 0.0),
            ("p50", 0.5),
            ("p90", 0.9),
            ("p95", 0.95),
            ("p99", 0.99),
            ("p100", 1.0),
        )
    }


def _axis_quantiles(values: np.ndarray) -> dict[str, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ComputedSegmentationError("axis distribution must be N-by-Z/Y/X")
    return {
        axis: _quantiles(array[:, index])
        for index, axis in enumerate(("z", "y", "x"))
    }


def _next_ladder(value: float, ladder: Sequence[int], axis: str) -> int:
    for candidate in ladder:
        if candidate >= value - 1e-12:
            return int(candidate)
    raise ComputedSegmentationError(
        f"required {axis} crop {value:.3f} exceeds the frozen shape ladder"
    )


def crop_geometry_audit(
    *,
    labels_zyx: np.ndarray,
    raw_frame_zyx: np.ndarray,
    coords_zyx: np.ndarray,
    scale_zyx: Sequence[float],
    max_nearest_distance_physical: float = 30.0,
    current_shape_zyx: Sequence[int] = CROP_SHAPE_ZYX,
    margin_um: float = CROP_GEOMETRY_MARGIN_UM,
) -> dict[str, Any]:
    """Measure computed-object extents and choose the preregistered p99 crop."""

    labels = np.asarray(labels_zyx, dtype=np.uint32)
    frame = np.asarray(raw_frame_zyx)
    coords = np.asarray(coords_zyx, dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    current_shape = np.asarray(current_shape_zyx, dtype=np.int64)
    if (
        labels.shape != frame.shape
        or labels.ndim != 3
        or coords.ndim != 2
        or coords.shape[1:] != (3,)
        or scale.shape != (3,)
        or np.any(scale <= 0)
        or current_shape.shape != (3,)
        or np.any(current_shape <= 0)
        or not math.isfinite(margin_um)
        or margin_um <= 0
    ):
        raise ComputedSegmentationError("crop geometry inputs are malformed")
    mapped = map_detections_to_regions(
        labels,
        frame,
        coords,
        scale_zyx=scale,
        max_nearest_distance_physical=max_nearest_distance_physical,
    )
    try:
        from skimage.measure import regionprops
    except ImportError as exc:
        raise ComputedSegmentationError("scikit-image is not installed") from exc
    regions = {int(region.label): region for region in regionprops(labels)}
    if not regions:
        raise ComputedSegmentationError("geometry label volume has no computed objects")
    object_voxel_extents: list[np.ndarray] = []
    object_physical_extents: list[np.ndarray] = []
    object_voxel_counts: list[float] = []
    object_equivalent_diameter_voxels: list[float] = []
    for label in sorted(regions):
        region = regions[label]
        bbox = np.asarray(region.bbox, dtype=np.int64)
        extent = bbox[3:] - bbox[:3]
        object_voxel_extents.append(extent)
        object_physical_extents.append(extent * scale)
        object_voxel_counts.append(float(region.area))
        object_equivalent_diameter_voxels.append(float(region.equivalent_diameter_area))

    margin_voxels = np.ceil(margin_um / scale).astype(np.int64)
    anchored_required_half: list[np.ndarray] = []
    anchor_offsets_physical: list[np.ndarray] = []
    anchored_full_containment: list[bool] = []
    matched_labels: list[int] = []
    for index in np.flatnonzero(mapped["matched"]):
        label = int(mapped["mask_labels"][index])
        region = regions[label]
        bbox = np.asarray(region.bbox, dtype=np.float64)
        low = bbox[:3]
        high = bbox[3:] - 1.0
        coord = coords[int(index)]
        required = np.maximum(coord - low, high - coord) + margin_voxels
        required = np.maximum(required, 0.0)
        anchored_required_half.append(required)
        centroid = np.asarray(region.centroid, dtype=np.float64)
        anchor_offsets_physical.append(np.abs(centroid - coord) * scale)
        anchored_full_containment.append(
            bool(np.all(required <= current_shape.astype(np.float64) / 2.0))
        )
        matched_labels.append(label)
    if not anchored_required_half:
        raise ComputedSegmentationError("geometry audit mapped no detections")
    required_half = np.stack(anchored_required_half)
    offsets = np.stack(anchor_offsets_physical)
    required_full_p99 = 2.0 * np.quantile(required_half, 0.99, axis=0)
    selected_shape = (
        _next_ladder(required_full_p99[0], CROP_SHAPE_LADDER_Z, "Z"),
        _next_ladder(required_full_p99[1], CROP_SHAPE_LADDER_YX, "Y"),
        _next_ladder(required_full_p99[2], CROP_SHAPE_LADDER_YX, "X"),
    )
    offset_p95 = np.quantile(offsets, 0.95, axis=0)
    result = {
        "schema": CROP_GEOMETRY_SCHEMA,
        "provenance_label": LABEL,
        "explicit_nonclaims": {
            "authentic_masks": False,
            "organizer_provided": False,
            "ground_truth_cell_sizes": False,
        },
        "object_population": {
            "unique_computed_object_count": len(regions),
            "voxel_extent_zyx": _axis_quantiles(np.stack(object_voxel_extents)),
            "physical_extent_zyx_um": _axis_quantiles(
                np.stack(object_physical_extents)
            ),
            "voxel_count": _quantiles(np.asarray(object_voxel_counts)),
            "equivalent_diameter_voxels": _quantiles(
                np.asarray(object_equivalent_diameter_voxels)
            ),
        },
        "detection_anchor_population": {
            "detection_count": len(coords),
            "matched_count": int(np.sum(mapped["matched"])),
            "unmatched_count": int(np.sum(~mapped["matched"])),
            "unique_selected_object_count": len(set(matched_labels)),
            "match_mode_counts": {
                str(mode): int(count)
                for mode, count in zip(
                    *np.unique(mapped["match_mode"], return_counts=True), strict=True
                )
            },
            "absolute_detection_to_centroid_offset_zyx_um": _axis_quantiles(offsets),
            "required_half_extent_zyx_voxels_including_margin": _axis_quantiles(
                required_half
            ),
            "current_crop_full_containment_count": int(
                np.sum(anchored_full_containment)
            ),
            "current_crop_full_containment_fraction_of_matched": float(
                np.mean(anchored_full_containment)
            ),
        },
        "selection": {
            "current_shape_zyx": current_shape.tolist(),
            "margin_um": float(margin_um),
            "margin_voxels_zyx": margin_voxels.tolist(),
            "required_full_shape_p99_zyx": required_full_p99.tolist(),
            "selected_shape_zyx": list(selected_shape),
            "bounded_shape_gate": bool(
                np.all(np.asarray(selected_shape) <= np.asarray((48, 128, 128)))
            ),
            "recenter_required_by_prereg": bool(np.any(offset_p95 > margin_um)),
            "selection_rule": (
                "smallest frozen even ladder shape covering per-axis p99 "
                "detection-anchored full object plus 2um margin"
            ),
        },
    }
    return result


def intensity_recentered_anchor(
    frame_zyx: np.ndarray,
    coord_zyx: Sequence[float],
    *,
    scale_zyx: Sequence[float],
    anchor_shape_zyx: Sequence[int] = CROP_SHAPE_ZYX,
    maximum_shift_um: float = 4.0,
) -> tuple[np.ndarray, float]:
    """Move an anchor toward local median-subtracted positive intensity."""

    frame = np.asarray(frame_zyx)
    coord = np.asarray(coord_zyx, dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    if (
        frame.ndim != 3
        or coord.shape != (3,)
        or scale.shape != (3,)
        or np.any(scale <= 0)
        or not math.isfinite(maximum_shift_um)
        or maximum_shift_um <= 0
    ):
        raise ComputedSegmentationError("intensity recenter inputs are malformed")
    crop, origin, _local_center = centered_crop(
        frame, coord, shape_zyx=anchor_shape_zyx
    )
    values = np.asarray(crop, dtype=np.float64)
    weights = np.maximum(values - np.median(values), 0.0)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= np.finfo(np.float64).eps:
        return coord.copy(), 0.0
    grid = np.indices(values.shape, dtype=np.float64).reshape(3, -1).T
    centroid_local = np.sum(grid * weights.reshape(-1, 1), axis=0) / total
    proposed = centroid_local + np.asarray(origin, dtype=np.float64)
    delta = proposed - coord
    physical_norm = float(np.linalg.norm(delta * scale))
    if physical_norm > maximum_shift_um:
        delta *= maximum_shift_um / physical_norm
        physical_norm = maximum_shift_um
    anchor = np.clip(coord + delta, 0.0, np.asarray(frame.shape) - 1.0)
    return anchor, physical_norm


def _segment_detection_crops(
    segmenter: Segmenter,
    crops: np.ndarray,
    *,
    crop_batch_size: int,
    network_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Segment equal-shaped crops with CUDA batching or deterministic fallback."""

    values = np.asarray(crops)
    if (
        values.ndim != 4
        or len(values) == 0
        or type(crop_batch_size) is not int
        or crop_batch_size < 1
        or type(network_batch_size) is not int
        or network_batch_size < 1
    ):
        raise ComputedSegmentationError("crop segmentation batch is malformed")
    labels = np.zeros(values.shape, dtype=np.uint32)
    seconds_per_detection = np.zeros(len(values), dtype=np.float64)
    batch_count = 0
    genuine = bool(
        callable(getattr(segmenter, "segment_batch", None))
        and getattr(segmenter, "_device", None) == "cuda"
    )
    started_total = time.perf_counter()
    if genuine:
        for start in range(0, len(values), crop_batch_size):
            end = min(start + crop_batch_size, len(values))
            started = time.perf_counter()
            local = np.asarray(
                segmenter.segment_batch(
                    values[start:end], network_batch_size=network_batch_size
                ),
                dtype=np.uint32,
            )
            elapsed = time.perf_counter() - started
            if local.shape != values[start:end].shape:
                raise ComputedSegmentationError("batched segmenter changed crop shape")
            labels[start:end] = local
            seconds_per_detection[start:end] = elapsed / (end - start)
            batch_count += 1
    else:
        for index, crop in enumerate(values):
            started = time.perf_counter()
            local = np.asarray(segmenter.segment(crop), dtype=np.uint32)
            seconds_per_detection[index] = time.perf_counter() - started
            if local.shape != crop.shape:
                raise ComputedSegmentationError("crop segmenter changed crop shape")
            labels[index] = local
            batch_count += 1
    return labels, {
        "genuine_cross_detection_gpu_batching": genuine,
        "crop_batch_size": crop_batch_size if genuine else 1,
        "network_plane_batch_size": network_batch_size if genuine else 1,
        "segmentation_batch_count": batch_count,
        "segmentation_seconds": time.perf_counter() - started_total,
    }, seconds_per_detection


def crop_detection_features(
    *,
    frame_zyx: np.ndarray,
    coords_zyx: np.ndarray,
    segmenter: Segmenter,
    scale_zyx: Sequence[float],
    max_nearest_distance_physical: float,
    shape_zyx: Sequence[int] = CROP_SHAPE_ZYX,
    recenter: bool = False,
    maximum_recenter_um: float = 4.0,
    crop_batch_size: int = 32,
    network_batch_size: int = 64,
    native_only: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    """Compute real native wrfeat from independently segmented detection crops."""

    frame = np.asarray(frame_zyx)
    coords = np.asarray(coords_zyx, dtype=np.float64)
    if frame.ndim != 3 or coords.ndim != 2 or coords.shape[1:] != (3,):
        raise ComputedSegmentationError("crop feature inputs are malformed")
    normalized_frame = _trackastra_normalize_frame(frame)
    count = len(coords)
    features = np.full((count, 12), np.nan, np.float32)
    descriptive = np.full((count, 12), np.nan, np.float32)
    matched = np.zeros(count, np.bool_)
    mask_labels = np.zeros(count, np.uint32)
    match_mode = np.full(count, "unmatched", dtype="U16")
    distances = np.full(count, np.inf, np.float32)
    boundary_contact = np.zeros(count, np.bool_)
    foreground_voxels = np.zeros(count, np.int64)
    mask_counts = np.zeros(count, np.int32)
    elapsed_seconds = np.zeros(count, np.float64)
    anchor_coords = np.zeros((count, 3), dtype=np.float64)
    recenter_shift_um = np.zeros(count, dtype=np.float64)
    raw_crops: list[np.ndarray] = []
    native_crops: list[np.ndarray] = []
    origins: list[tuple[int, int, int]] = []
    local_centers: list[tuple[int, int, int]] = []
    for index, coord in enumerate(coords):
        anchor = coord
        if recenter:
            anchor, recenter_shift_um[index] = intensity_recentered_anchor(
                frame,
                coord,
                scale_zyx=scale_zyx,
                maximum_shift_um=maximum_recenter_um,
            )
        anchor_coords[index] = anchor
        raw_crop, origin, local_center = centered_crop(
            frame, anchor, shape_zyx=shape_zyx
        )
        native_crop, normalized_origin, _ = centered_crop(
            normalized_frame, anchor, shape_zyx=shape_zyx
        )
        if normalized_origin != origin:
            raise ComputedSegmentationError("raw/normalized crop origins disagree")
        raw_crops.append(raw_crop)
        native_crops.append(native_crop)
        origins.append(origin)
        local_centers.append(local_center)
    segmentation_started = time.perf_counter()
    label_batch, batch_runtime, segment_seconds = _segment_detection_crops(
        segmenter,
        np.stack(raw_crops),
        crop_batch_size=crop_batch_size,
        network_batch_size=network_batch_size,
    )
    batch_runtime["including_dispatch_seconds"] = (
        time.perf_counter() - segmentation_started
    )
    for index, coord in enumerate(coords):
        started = time.perf_counter()
        raw_crop = raw_crops[index]
        native_crop = native_crops[index]
        origin = origins[index]
        local_center = local_centers[index]
        labels = label_batch[index]
        properties = _label_properties(
            labels,
            raw_crop,
            native_crop,
            scale_zyx,
            coordinate_offset_zyx=origin,
            frame_shape_zyx=frame.shape,
            native_only=native_only,
        )
        local = np.asarray(local_center, dtype=np.int64)
        chosen = int(labels[tuple(local)]) if np.all(local < labels.shape) else 0
        mode = "contains" if chosen in properties else "unmatched"
        distance = 0.0 if mode == "contains" else math.inf
        if chosen not in properties and properties:
            labels_sorted = np.asarray(sorted(properties), dtype=np.int64)
            centroids = np.stack(
                [properties[int(label)]["centroid"] for label in labels_sorted]
            )
            delta = (
                centroids - anchor_coords[index][None, :]
            ) * np.asarray(scale_zyx)[None, :]
            candidate_distances = np.linalg.norm(delta, axis=1)
            candidate_row = int(np.argmin(candidate_distances))
            candidate_distance = float(candidate_distances[candidate_row])
            if candidate_distance <= max_nearest_distance_physical:
                chosen = int(labels_sorted[candidate_row])
                mode = "nearest_centroid"
                distance = candidate_distance
        if chosen in properties:
            prop = properties[chosen]
            features[index] = prop["native"]
            descriptive[index] = prop["descriptive"]
            matched[index] = True
            mask_labels[index] = chosen
            match_mode[index] = mode
            distances[index] = distance
            selected = labels == chosen
            boundary_contact[index] = bool(
                np.any(selected[0])
                or np.any(selected[-1])
                or np.any(selected[:, 0])
                or np.any(selected[:, -1])
                or np.any(selected[:, :, 0])
                or np.any(selected[:, :, -1])
            )
            foreground_voxels[index] = int(np.count_nonzero(selected))
        mask_counts[index] = int(len(properties))
        elapsed_seconds[index] = segment_seconds[index] + (
            time.perf_counter() - started
        )
        if progress:
            print(
                f"crop wrfeat {index + 1:04d}/{count:04d} "
                f"matched={bool(matched[index])} boundary={bool(boundary_contact[index])} "
                f"seconds={elapsed_seconds[index]:.3f}",
                flush=True,
            )
    return {
        "features": features,
        "descriptive_regionprops": descriptive,
        "matched": matched,
        "mask_labels": mask_labels,
        "match_mode": match_mode,
        "match_distance_physical": distances,
        "selected_mask_boundary_contact": boundary_contact,
        "selected_mask_foreground_voxels": foreground_voxels,
        "computed_mask_count": mask_counts,
        "elapsed_seconds_per_detection": elapsed_seconds,
        "anchor_coords_zyx": anchor_coords.astype(np.float32),
        "recenter_shift_um": recenter_shift_um.astype(np.float32),
        "batch_runtime": batch_runtime,
    }


def adaptive_crop_detection_features(
    *,
    frame_zyx: np.ndarray,
    coords_zyx: np.ndarray,
    segmenter: Segmenter,
    scale_zyx: Sequence[float],
    max_nearest_distance_physical: float,
    primary_shape_zyx: Sequence[int] = CROP_PRIMARY_SHAPE_ZYX,
    fallback_shape_zyx: Sequence[int] = CROP_FALLBACK_SHAPE_ZYX,
    maximum_recenter_um: float = 4.0,
    crop_batch_size: int = 32,
    network_batch_size: int = 64,
    native_only: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the frozen primary crop and one boundary/invalid fallback."""

    started = time.perf_counter()
    primary = crop_detection_features(
        frame_zyx=frame_zyx,
        coords_zyx=coords_zyx,
        segmenter=segmenter,
        scale_zyx=scale_zyx,
        max_nearest_distance_physical=max_nearest_distance_physical,
        shape_zyx=primary_shape_zyx,
        recenter=True,
        maximum_recenter_um=maximum_recenter_um,
        crop_batch_size=crop_batch_size,
        network_batch_size=network_batch_size,
        native_only=native_only,
        progress=progress,
    )
    finite = np.isfinite(primary["features"]).all(axis=1)
    nonzero = ~np.all(primary["features"] == 0, axis=1)
    primary_valid = (
        primary["matched"]
        & finite
        & nonzero
        & ~primary["selected_mask_boundary_contact"]
    )
    fallback_indices = np.flatnonzero(~primary_valid)
    final: dict[str, Any] = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in primary.items()
        if key != "batch_runtime"
    }
    attempted = np.zeros(len(primary_valid), dtype=np.bool_)
    recovered = np.zeros(len(primary_valid), dtype=np.bool_)
    fallback_runtime: Mapping[str, Any] | None = None
    if len(fallback_indices):
        attempted[fallback_indices] = True
        fallback = crop_detection_features(
            frame_zyx=frame_zyx,
            coords_zyx=np.asarray(coords_zyx)[fallback_indices],
            segmenter=segmenter,
            scale_zyx=scale_zyx,
            max_nearest_distance_physical=max_nearest_distance_physical,
            shape_zyx=fallback_shape_zyx,
            recenter=True,
            maximum_recenter_um=maximum_recenter_um,
            crop_batch_size=crop_batch_size,
            network_batch_size=network_batch_size,
            native_only=native_only,
            progress=progress,
        )
        fallback_finite = np.isfinite(fallback["features"]).all(axis=1)
        fallback_nonzero = ~np.all(fallback["features"] == 0, axis=1)
        fallback_valid = (
            fallback["matched"]
            & fallback_finite
            & fallback_nonzero
            & ~fallback["selected_mask_boundary_contact"]
        )
        for local_index, global_index in enumerate(fallback_indices):
            if fallback_valid[local_index]:
                recovered[global_index] = True
                for key, value in fallback.items():
                    if isinstance(value, np.ndarray) and value.shape[:1] == (
                        len(fallback_indices),
                    ):
                        final[key][global_index] = value[local_index]
            else:
                final["features"][global_index] = np.nan
                final["descriptive_regionprops"][global_index] = np.nan
                final["matched"][global_index] = False
                final["mask_labels"][global_index] = 0
                final["match_mode"][global_index] = "unmatched"
                final["match_distance_physical"][global_index] = np.inf
                final["selected_mask_boundary_contact"][global_index] = bool(
                    primary["selected_mask_boundary_contact"][global_index]
                    or fallback["selected_mask_boundary_contact"][local_index]
                )
            final["elapsed_seconds_per_detection"][global_index] = (
                primary["elapsed_seconds_per_detection"][global_index]
                + fallback["elapsed_seconds_per_detection"][local_index]
            )
        fallback_runtime = fallback["batch_runtime"]
    final["fallback_attempted"] = attempted
    final["fallback_recovered"] = recovered
    final["batch_runtime"] = {
        "primary": primary["batch_runtime"],
        "fallback": fallback_runtime,
        "primary_shape_zyx": list(map(int, primary_shape_zyx)),
        "fallback_shape_zyx": list(map(int, fallback_shape_zyx)),
        "fallback_attempted_count": int(np.sum(attempted)),
        "fallback_recovered_count": int(np.sum(recovered)),
        "wall_seconds": time.perf_counter() - started,
    }
    return final


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
    affine_dimensions = np.full(len(node_ids), -1, np.int8)
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
        affine_dimensions[selection] = mapped[
            "computed_object_center_affine_dimension"
        ]
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
        "computed_object_center_affine_dimension": affine_dimensions,
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
    affine_dimensions = np.asarray(
        payload.get("computed_object_center_affine_dimension", np.empty(0))
    )
    if (
        features.shape != (len(node_ids), 12)
        or descriptive.shape != (len(node_ids), 12)
        or matched.shape != (len(node_ids),)
        or affine_dimensions.shape != (len(node_ids),)
        or np.any((affine_dimensions < 0) | (affine_dimensions > 3))
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
            "descriptive_feature_schema": np.asarray(DESCRIPTIVE_FEATURE_SCHEMA),
            "solidity_definition": np.asarray(VOXEL_CELL_SOLIDITY_DEFINITION),
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
            "descriptive_feature_schema": DESCRIPTIVE_FEATURE_SCHEMA,
            "center_affine_dimension_key": (
                "computed_object_center_affine_dimension"
            ),
            "solidity": {
                "name": "solidity_voxel_cell_convex_hull_3d",
                "definition": VOXEL_CELL_SOLIDITY_DEFINITION,
                "undefined_sentinel": None,
                "legacy_skimage_center_hull_used": False,
            },
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


def crop_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Run the preregistered all-detection crop wrfeat pilot on one frame."""

    started = time.perf_counter()
    if (
        tuple(args.crop_shape_zyx) != CROP_PRIMARY_SHAPE_ZYX
        or tuple(args.fallback_crop_shape_zyx) != CROP_FALLBACK_SHAPE_ZYX
        or args.maximum_recenter_um != 4.0
        or not 1 <= args.crop_batch_size <= 32
        or args.network_batch_size != 64
    ):
        raise ComputedSegmentationError(
            "crop GPU pilot parameters drifted from preregistration amendment 1"
        )
    array, attrs = _zarr_array(args.zarr)
    scale = spatial_scale_zyx(attrs)
    t = int(args.pilot_timepoint)
    if t < 0 or t >= int(array.shape[0]):
        raise ComputedSegmentationError(f"crop pilot timepoint {t} is out of bounds")
    if args.detections_npz:
        node_ids, timepoints, coords = load_detection_npz(args.detections_npz)
    else:
        node_ids, timepoints, coords = load_bh_detections(
            args.cache_root, args.identity
        )
    selection = np.flatnonzero(timepoints == t)
    if len(selection) == 0:
        raise ComputedSegmentationError("crop pilot frame has no detections")
    frame = np.asarray(array[t])
    init_started = time.perf_counter()
    segmenter = _load_segmenter(args, scale)
    initialization_seconds = time.perf_counter() - init_started
    result = adaptive_crop_detection_features(
        frame_zyx=frame,
        coords_zyx=coords[selection],
        segmenter=segmenter,
        scale_zyx=scale,
        max_nearest_distance_physical=args.max_nearest_distance_physical,
        primary_shape_zyx=args.crop_shape_zyx,
        fallback_shape_zyx=args.fallback_crop_shape_zyx,
        maximum_recenter_um=args.maximum_recenter_um,
        crop_batch_size=args.crop_batch_size,
        network_batch_size=args.network_batch_size,
        progress=True,
    )
    finite = np.isfinite(result["features"]).all(axis=1)
    nonzero = ~np.all(result["features"] == 0, axis=1)
    valid = result["matched"] & finite & nonzero
    crop_seconds = np.asarray(result["elapsed_seconds_per_detection"], dtype=np.float64)
    feature_seconds = float(np.sum(crop_seconds))
    projected_cv57_hours = feature_seconds * 5700.0 / 3600.0
    modes, mode_counts = np.unique(result["match_mode"], return_counts=True)
    report = {
        "schema": CROP_GPU_SCHEMA,
        "status": (
            "PASS_CROP_INTERFACE_COMPLETE"
            if bool(np.all(valid))
            else "FAIL_CROP_INTERFACE_INCOMPLETE"
        ),
        "production_manifest_eligible": False,
        "provenance_label": LABEL,
        "explicit_nonclaims": {
            "authentic_masks": False,
            "organizer_provided": False,
            "h3_authentic_masks": False,
        },
        "identity": args.identity,
        "timepoint": t,
        "crop_contract": {
            "primary_shape_zyx": list(args.crop_shape_zyx),
            "fallback_shape_zyx": list(args.fallback_crop_shape_zyx),
            "padding": "constant-zero",
            "unit_of_work": "batched-independent-detection-crops-per-frame",
            "anchor": (
                "local-median-subtracted-positive-intensity-centroid-clipped-4um"
            ),
            "native_intensity_normalization": "whole-frame-float32-p1-p99p8",
            "border_distance_coordinate_frame": "original-full-frame-xy",
            "mapping_radius_physical": float(args.max_nearest_distance_physical),
        },
        "population": {
            "detection_count": int(len(selection)),
            "matched_count": int(np.sum(result["matched"])),
            "finite_nonzero_count": int(np.sum(valid)),
            "empty_mask_crop_count": int(np.sum(result["computed_mask_count"] == 0)),
            "selected_mask_boundary_contact_count": int(
                np.sum(result["selected_mask_boundary_contact"])
            ),
            "selected_mask_boundary_contact_fraction": float(
                np.mean(result["selected_mask_boundary_contact"])
            ),
            "match_mode_counts": {
                str(mode): int(count)
                for mode, count in zip(modes, mode_counts, strict=True)
            },
            "fallback_attempted_count": int(np.sum(result["fallback_attempted"])),
            "fallback_recovered_count": int(np.sum(result["fallback_recovered"])),
        },
        "runtime": {
            "device": args.device,
            "initialization_seconds": initialization_seconds,
            "feature_seconds_for_frame": feature_seconds,
            "wall_seconds": time.perf_counter() - started,
            "seconds_per_detection_quantiles": {
                "p00": float(np.quantile(crop_seconds, 0.0)),
                "p50": float(np.quantile(crop_seconds, 0.5)),
                "p95": float(np.quantile(crop_seconds, 0.95)),
                "p100": float(np.quantile(crop_seconds, 1.0)),
            },
            "projected_cv57_hours_same_device": projected_cv57_hours,
            "batching": result["batch_runtime"],
            "terminal_whole_frame_reference": {
                "device": "cuda-t4",
                "repeat_seconds": [43.459801244000005, 42.774226285000054],
                "combined_cv57_heldfit105_worst_hours": 195.58589296169947,
                "cross_device_speed_claim": False,
            },
        },
        "segmenter": dict(segmenter.provenance),
    }
    if args.output:
        arrays = {
            key: value
            for key, value in result.items()
            if isinstance(value, np.ndarray)
        }
        arrays.update(
            {
                "schema": np.asarray(CROP_GPU_SCHEMA),
                "provenance_label": np.asarray(LABEL),
                "identity": np.asarray(args.identity),
                "timepoint": np.asarray(t, dtype=np.int32),
                "node_ids": node_ids[selection],
                "coords_zyx": coords[selection],
                "feature_names": np.asarray(TRACKASTRA_NATIVE_FEATURE_NAMES),
            }
        )
        _atomic_npz(args.output, arrays)
        report["artifact"] = {"path": str(args.output), "sha256": sha256(args.output)}
    if args.pilot_report:
        _atomic_json(args.pilot_report, report)
    return report


def crop_geometry(args: argparse.Namespace) -> dict[str, Any]:
    """Audit the fixed design-frame computed labels without segmenting again."""

    array, attrs = _zarr_array(args.zarr)
    scale = spatial_scale_zyx(attrs)
    if args.detections_npz:
        _node_ids, timepoints, coords = load_detection_npz(args.detections_npz)
    else:
        _node_ids, timepoints, coords = load_bh_detections(
            args.cache_root, args.identity
        )
    try:
        with np.load(args.pilot_labels, allow_pickle=False) as archive:
            labels = np.asarray(archive["labels_zyx"], dtype=np.uint32)
            label_identity = str(archive["identity"].item())
            t = int(archive["timepoint"].item())
            provenance = str(archive["provenance_label"].item())
            authentic = bool(archive["authentic_masks"].item())
    except (OSError, KeyError, ValueError) as exc:
        raise ComputedSegmentationError("invalid computed pilot-label artifact") from exc
    if (
        label_identity != args.identity
        or provenance != LABEL
        or authentic
        or t < 0
        or t >= int(array.shape[0])
    ):
        raise ComputedSegmentationError("computed pilot-label provenance drifted")
    selection = np.flatnonzero(timepoints == t)
    if len(selection) == 0:
        raise ComputedSegmentationError("geometry design frame has no detections")
    result = crop_geometry_audit(
        labels_zyx=labels,
        raw_frame_zyx=np.asarray(array[t]),
        coords_zyx=coords[selection],
        scale_zyx=scale,
        max_nearest_distance_physical=args.max_nearest_distance_physical,
    )
    result.update(
        {
            "identity": args.identity,
            "timepoint": t,
            "source_zarr": str(args.zarr),
            "pilot_labels": {
                "path": str(args.pilot_labels),
                "sha256": sha256(args.pilot_labels),
            },
        }
    )
    if args.geometry_report:
        _atomic_json(args.geometry_report, result)
    return result


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
                        "computed_object_center_affine_dimension",
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
                    and data["descriptive_feature_schema"].item()
                    == DESCRIPTIVE_FEATURE_SCHEMA
                    and data["solidity_definition"].item()
                    == VOXEL_CELL_SOLIDITY_DEFINITION
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
        "command",
        choices=(
            "preflight",
            "pilot",
            "crop-geometry",
            "crop-pilot",
            "run",
            "batch",
            "manifest",
        ),
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
    value.add_argument("--geometry-report", type=Path)
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
    value.add_argument(
        "--crop-shape-zyx",
        type=int,
        nargs=3,
        default=CROP_PRIMARY_SHAPE_ZYX,
    )
    value.add_argument(
        "--fallback-crop-shape-zyx",
        type=int,
        nargs=3,
        default=CROP_FALLBACK_SHAPE_ZYX,
    )
    value.add_argument("--maximum-recenter-um", type=float, default=4.0)
    value.add_argument("--crop-batch-size", type=int, default=32)
    value.add_argument("--network-batch-size", type=int, default=64)
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
    elif args.command == "crop-geometry":
        required = {
            "--zarr": args.zarr,
            "--identity": args.identity,
            "--pilot-labels": args.pilot_labels,
            "--geometry-report": args.geometry_report,
        }
    else:
        required = {"--zarr": args.zarr, "--identity": args.identity}
    if args.command in {"pilot", "crop-pilot", "run", "batch"}:
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
    elif args.command == "crop-pilot":
        report = crop_pilot(args)
    elif args.command == "crop-geometry":
        report = crop_geometry(args)
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
        "PASS_CROP_INTERFACE_COMPLETE",
        "PASS_COMPLETE_PRODUCTION_COVERAGE",
        "PASS_ATOMIC_SHARD_COMPLETE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
