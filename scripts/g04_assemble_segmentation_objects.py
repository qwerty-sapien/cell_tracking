#!/usr/bin/env python3
"""Assemble raw-only computed objects into detection-aligned wrfeat locally.

Remote shards contain no learned detections.  This local-only assembler verifies
exact frame coverage and every downloaded artifact, maps the existing learned
detections to deterministic nearest computed centroids at the preregistered
30-physical-unit radius, rejects every missing/non-finite/all-zero row, and
publishes the exact manifest consumed by ``g04_trackastra_computed_wrfeat.py``.
It never reads GEFF, targets, edges, or metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for _location in (ROOT / "src", ROOT / "scripts"):
    if str(_location) not in sys.path:
        sys.path.insert(0, str(_location))

POPULATION_SCHEMA = "celltrack-alpha.g04-segmentation-production-population.v1"
SHARD_RECEIPT_SCHEMA = "celltrack-alpha.g04-computed-object-production-shard.v2"
FRAME_SCHEMA = "celltrack-alpha.g04-computed-object-frame.v2"
FEATURE_MANIFEST_SCHEMA = (
    "celltrack-alpha.g04-computed-segmentation-wrfeat-manifest.v1"
)
FEATURE_SCHEMA = "trackastra-ctc-wrfeat12.v1"
FEATURE_CONTRACT_SHA256 = (
    "910177ae086013bfdff04b1e9528526ff4961f170fbdba49af6f578a8a3436b7"
)
PROVENANCE_LABEL = "COMPUTED_SEGMENTATIONS"
MAPPING_RADIUS_PHYSICAL = 30.0


class AssemblyError(RuntimeError):
    """A shard, coverage, provenance, mapping, or feature invariant failed."""


@dataclass(frozen=True)
class FrameObjects:
    identity: str
    timepoint: int
    computed_label_ids: Any
    centroids_zyx: Any
    native_wrfeat12: Any
    descriptive_regionprops: Any
    center_affine_dimensions: Any
    scale_zyx: Any
    raw_frame_sha256: str
    computed_labels_sha256: str
    computed_object_payload_sha256: str
    path: Path
    sha256: str


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise AssemblyError("numpy is required") from exc
    return np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AssemblyError(f"{label} must be a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scalar_text(value: Any, field: str) -> str:
    np = _numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise AssemblyError(f"{field} must be scalar")
    item = array.reshape(-1)[0]
    return item.decode("ascii") if isinstance(item, bytes) else str(item)


def scalar_bool(value: Any, field: str) -> bool:
    np = _numpy()
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "bui":
        raise AssemblyError(f"{field} must be scalar boolean")
    return bool(array.reshape(-1)[0])


def load_population(path: Path) -> dict[str, dict[str, Any]]:
    value = load_json(path, "production population")
    rows = value.get("identities")
    if (
        value.get("schema") != POPULATION_SCHEMA
        or value.get("status") != "ROOT_FROZEN_EXACT_FRAME_POPULATION"
        or not isinstance(rows, list)
        or not rows
    ):
        raise AssemblyError("production population contract drift")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AssemblyError("population row must be an object")
        identity = str(row.get("identity", ""))
        frames = row.get("frame_indices")
        scale = row.get("scale_zyx")
        if (
            ":" not in identity
            or identity in result
            or not isinstance(frames, list)
            or not frames
            or any(type(item) is not int or item < 0 for item in frames)
            or frames != sorted(set(frames))
            or not isinstance(scale, list)
            or len(scale) != 3
            or any(
                not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) <= 0
                for item in scale
            )
        ):
            raise AssemblyError(f"invalid population row: {identity!r}")
        result[identity] = {
            "frame_indices": tuple(int(item) for item in frames),
            "scale_zyx": tuple(float(item) for item in scale),
        }
    if list(result) != sorted(result):
        raise AssemblyError("population identities must be sorted")
    return result


def _validate_artifact_inventory(root: Path, receipt: Mapping[str, Any]) -> None:
    rows = receipt.get("artifact_files")
    if not isinstance(rows, list) or not rows:
        raise AssemblyError(f"shard artifact inventory is empty: {root}")
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise AssemblyError(f"unsafe shard artifact path: {relative!r}")
        seen.add(relative)
        path = root / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("bytes", -1))
            or sha256(path) != row.get("sha256")
        ):
            raise AssemblyError(f"shard artifact byte mismatch: {path}")


def load_frame(path: Path, expected_identity: str, expected_timepoint: int) -> FrameObjects:
    np = _numpy()
    required = {
        "schema",
        "identity",
        "timepoint",
        "computed_label_ids",
        "centroids_zyx",
        "trackastra_native_wrfeat12",
        "descriptive_regionprops",
        "computed_object_center_affine_dimension",
        "feature_names",
        "descriptive_feature_names",
        "descriptive_feature_schema",
        "solidity_definition",
        "feature_contract_sha256",
        "feature_schema",
        "scale_zyx",
        "raw_frame_sha256",
        "computed_labels_sha256",
        "computed_object_payload_sha256",
        "provenance_label",
        "authentic_masks",
        "organizer_provided",
        "h3_authentic_masks",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise AssemblyError(f"frame artifact missing {sorted(missing)}: {path}")
            schema = scalar_text(archive["schema"], "schema")
            identity = scalar_text(archive["identity"], "identity")
            timepoint_array = np.asarray(archive["timepoint"])
            timepoint = int(timepoint_array.reshape(-1)[0])
            labels = np.asarray(archive["computed_label_ids"], dtype="<u4")
            centroids = np.asarray(archive["centroids_zyx"], dtype="<f4")
            native = np.asarray(archive["trackastra_native_wrfeat12"], dtype="<f4")
            descriptive = np.asarray(
                archive["descriptive_regionprops"], dtype="<f4"
            )
            affine_dimensions = np.asarray(
                archive["computed_object_center_affine_dimension"],
                dtype=np.int8,
            )
            scale = np.asarray(archive["scale_zyx"], dtype="<f8")
            raw_sha = scalar_text(archive["raw_frame_sha256"], "raw_frame_sha256")
            label_sha = scalar_text(
                archive["computed_labels_sha256"], "computed_labels_sha256"
            )
            payload_sha = scalar_text(
                archive["computed_object_payload_sha256"],
                "computed_object_payload_sha256",
            )
            provenance = scalar_text(archive["provenance_label"], "provenance_label")
            feature_schema = scalar_text(archive["feature_schema"], "feature_schema")
            feature_names = tuple(str(item) for item in archive["feature_names"].tolist())
            descriptive_names = tuple(
                str(item) for item in archive["descriptive_feature_names"].tolist()
            )
            descriptive_schema = scalar_text(
                archive["descriptive_feature_schema"],
                "descriptive_feature_schema",
            )
            solidity_definition = scalar_text(
                archive["solidity_definition"], "solidity_definition"
            )
            feature_contract_sha = scalar_text(
                archive["feature_contract_sha256"],
                "feature_contract_sha256",
            )
            explicit_nonclaims = (
                scalar_bool(archive["authentic_masks"], "authentic_masks"),
                scalar_bool(archive["organizer_provided"], "organizer_provided"),
                scalar_bool(archive["h3_authentic_masks"], "h3_authentic_masks"),
            )
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, AssemblyError):
            raise
        raise AssemblyError(f"cannot read computed-object frame: {path}") from exc
    from g04_computed_segmentation import (
        DESCRIPTIVE_FEATURE_NAMES,
        DESCRIPTIVE_FEATURE_SCHEMA,
        TRACKASTRA_NATIVE_FEATURE_NAMES,
        VOXEL_CELL_SOLIDITY_DEFINITION,
    )

    n = len(labels)
    digest = hashlib.sha256()
    for payload in (labels, centroids, native, descriptive, affine_dimensions):
        digest.update(np.ascontiguousarray(payload).tobytes())
    if (
        schema != FRAME_SCHEMA
        or identity != expected_identity
        or timepoint != expected_timepoint
        or provenance != PROVENANCE_LABEL
        or explicit_nonclaims != (False, False, False)
        or feature_schema != FEATURE_SCHEMA
        or feature_names != TRACKASTRA_NATIVE_FEATURE_NAMES
        or descriptive_names != DESCRIPTIVE_FEATURE_NAMES
        or descriptive_schema != DESCRIPTIVE_FEATURE_SCHEMA
        or solidity_definition != VOXEL_CELL_SOLIDITY_DEFINITION
        or feature_contract_sha != FEATURE_CONTRACT_SHA256
        or labels.ndim != 1
        or n == 0
        or labels.tolist() != sorted(set(int(item) for item in labels))
        or centroids.shape != (n, 3)
        or native.shape != (n, 12)
        or descriptive.shape != (n, 12)
        or affine_dimensions.shape != (n,)
        or scale.shape != (3,)
        or not np.isfinite(centroids).all()
        or not np.isfinite(native).all()
        or not np.isfinite(descriptive).all()
        or np.any((affine_dimensions < 0) | (affine_dimensions > 3))
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or np.any(native[:, 0] <= 0)
        or np.any(np.all(native == 0, axis=1))
        or any(len(value) != 64 for value in (raw_sha, label_sha, payload_sha))
        or digest.hexdigest() != payload_sha
    ):
        raise AssemblyError(f"computed-object frame contract drift: {path}")
    return FrameObjects(
        identity=identity,
        timepoint=timepoint,
        computed_label_ids=labels,
        centroids_zyx=centroids,
        native_wrfeat12=native,
        descriptive_regionprops=descriptive,
        center_affine_dimensions=affine_dimensions,
        scale_zyx=scale,
        raw_frame_sha256=raw_sha,
        computed_labels_sha256=label_sha,
        computed_object_payload_sha256=payload_sha,
        path=path,
        sha256=sha256(path),
    )


def load_shard_outputs(roots: Sequence[Path]) -> dict[tuple[str, int], FrameObjects]:
    frames: dict[tuple[str, int], FrameObjects] = {}
    shard_ids: set[str] = set()
    for root in roots:
        receipt_path = root / "G04_PRODUCTION_SHARD_RECEIPT.json"
        receipt = load_json(receipt_path, "production shard receipt")
        if (
            receipt.get("schema") != SHARD_RECEIPT_SCHEMA
            or receipt.get("status") != "PASS_COMPLETE_SHARD"
            or receipt.get("production_manifest_eligible") is not False
            or receipt.get("requires_local_detection_mapping") is not True
            or receipt.get("feature_contract_sha256")
            != FEATURE_CONTRACT_SHA256
            or receipt.get("geff_access") is not False
            or receipt.get("learned_detection_access") is not False
            or receipt.get("gt_access") is not False
            or receipt.get("metric_access") is not False
            or receipt.get("submission") is not False
        ):
            raise AssemblyError(f"shard receipt is not a raw-only complete PASS: {root}")
        shard_id = str(receipt.get("shard_id", ""))
        if not shard_id or shard_id in shard_ids:
            raise AssemblyError(f"duplicate/invalid shard id: {shard_id!r}")
        shard_ids.add(shard_id)
        _validate_artifact_inventory(root, receipt)
        assignments = receipt.get("assignments")
        if not isinstance(assignments, list) or not assignments:
            raise AssemblyError(f"shard has no assignments: {root}")
        normalized = sorted(
            (
                (str(row.get("identity", "")), int(row.get("timepoint", -1)))
                for row in assignments
            )
        )
        if len(normalized) != len(set(normalized)):
            raise AssemblyError(f"shard repeats assignments: {root}")
        for identity, timepoint in normalized:
            key = (identity, timepoint)
            if ":" not in identity or timepoint < 0 or key in frames:
                raise AssemblyError(f"duplicate/invalid cross-shard assignment: {key}")
            slug = identity.replace(":", "__", 1)
            path = root / "frames" / slug / f"t{timepoint:04d}.npz"
            sidecar_path = path.with_suffix(".json")
            sidecar = load_json(sidecar_path, "computed-object frame sidecar")
            frame = load_frame(path, identity, timepoint)
            if (
                sidecar.get("status") != "PASS_COMPLETE_FRAME"
                or sidecar.get("identity") != identity
                or sidecar.get("timepoint") != timepoint
                or sidecar.get("computed_labels_sha256")
                != frame.computed_labels_sha256
                or sidecar.get("computed_object_payload_sha256")
                != frame.computed_object_payload_sha256
                or sidecar.get("raw_frame_sha256") != frame.raw_frame_sha256
                or sidecar.get("feature_contract_sha256")
                != FEATURE_CONTRACT_SHA256
                or sidecar.get("artifact", {}).get("sha256") != frame.sha256
                or sidecar.get("geff_access") is not False
                or sidecar.get("learned_detection_access") is not False
                or sidecar.get("gt_access") is not False
                or sidecar.get("metric_access") is not False
                or sidecar.get("submission") is not False
            ):
                raise AssemblyError(f"frame sidecar drift: {sidecar_path}")
            frames[key] = frame
    return frames


def require_exact_coverage(
    population: Mapping[str, Mapping[str, Any]],
    frames: Mapping[tuple[str, int], FrameObjects],
) -> None:
    expected = {
        (identity, timepoint)
        for identity, row in population.items()
        for timepoint in row["frame_indices"]
    }
    observed = set(frames)
    if observed != expected:
        missing = sorted(expected - observed)[:8]
        extra = sorted(observed - expected)[:8]
        raise AssemblyError(
            f"computed-object frame coverage mismatch; missing={missing}, extra={extra}"
        )
    np = _numpy()
    for (identity, _timepoint), frame in frames.items():
        expected_scale = np.asarray(population[identity]["scale_zyx"], dtype=np.float64)
        if not np.array_equal(frame.scale_zyx, expected_scale):
            raise AssemblyError(f"computed-object spatial scale mismatch: {identity}")


def map_detections_to_objects(
    *,
    node_ids: Any,
    timepoints: Any,
    coords_zyx: Any,
    frames: Mapping[int, FrameObjects],
    radius_physical: float = MAPPING_RADIUS_PHYSICAL,
) -> dict[str, Any]:
    np = _numpy()
    from g04_computed_segmentation import validate_detections

    node_ids, timepoints, coords = validate_detections(
        node_ids, timepoints, coords_zyx
    )
    n = len(node_ids)
    features = np.full((n, 12), np.nan, dtype=np.float32)
    descriptive = np.full((n, 12), np.nan, dtype=np.float32)
    affine_dimensions = np.full(n, -1, dtype=np.int8)
    mask_labels = np.zeros(n, dtype=np.uint32)
    distances = np.full(n, np.inf, dtype=np.float32)
    matched = np.zeros(n, dtype=np.bool_)
    modes = np.full(n, "unmatched", dtype="U24")
    for timepoint in np.unique(timepoints):
        t = int(timepoint)
        if t not in frames:
            raise AssemblyError(f"no computed objects for detection frame {t}")
        frame = frames[t]
        selection = np.flatnonzero(timepoints == t)
        scale = np.asarray(frame.scale_zyx, dtype=np.float64)
        object_coords = np.asarray(frame.centroids_zyx, dtype=np.float64)
        for index in selection:
            delta = (object_coords - coords[index][None, :]) * scale[None, :]
            candidate_distances = np.linalg.norm(delta, axis=1)
            chosen = int(np.argmin(candidate_distances))
            distance = float(candidate_distances[chosen])
            if distance > radius_physical:
                continue
            features[index] = frame.native_wrfeat12[chosen]
            descriptive[index] = frame.descriptive_regionprops[chosen]
            affine_dimensions[index] = frame.center_affine_dimensions[chosen]
            mask_labels[index] = frame.computed_label_ids[chosen]
            distances[index] = distance
            matched[index] = True
            modes[index] = "nearest_centroid"
    if (
        not np.all(matched)
        or not np.isfinite(features).all()
        or not np.isfinite(descriptive).all()
        or np.any((affine_dimensions < 0) | (affine_dimensions > 3))
        or not np.isfinite(distances).all()
        or np.any(mask_labels == 0)
        or np.any(features[:, 0] <= 0)
        or np.any(np.all(features == 0, axis=1))
    ):
        bad = np.flatnonzero(~matched)[:8].tolist()
        raise AssemblyError(
            f"radius-{radius_physical:g} mapping is incomplete/nonfinite; "
            f"unmatched_indices={bad}"
        )
    return {
        "node_ids": node_ids,
        "timepoints": timepoints,
        "coords_zyx": coords,
        "features": features,
        "descriptive_regionprops": descriptive,
        "computed_object_center_affine_dimension": affine_dimensions,
        "matched": matched,
        "mask_labels": mask_labels,
        "match_mode": modes,
        "match_distance_physical": distances,
    }


def publish_identity(
    output: Path,
    *,
    identity: str,
    payload: Mapping[str, Any],
    frames: Mapping[int, FrameObjects],
) -> dict[str, Any]:
    np = _numpy()
    from g04_computed_segmentation import (
        DESCRIPTIVE_FEATURE_NAMES,
        DESCRIPTIVE_FEATURE_SCHEMA,
        TRACKASTRA_NATIVE_FEATURE_NAMES,
        VOXEL_CELL_SOLIDITY_DEFINITION,
    )

    arrays = dict(payload)
    arrays.update(
        {
            "schema": np.asarray(
                "celltrack-alpha.g04-computed-segmentation-features.v2"
            ),
            "identity": np.asarray(identity),
            "provenance_label": np.asarray(PROVENANCE_LABEL),
            "authentic_masks": np.asarray(False),
            "organizer_provided": np.asarray(False),
            "h3_authentic_masks": np.asarray(False),
            "feature_names": np.asarray(TRACKASTRA_NATIVE_FEATURE_NAMES),
            "feature_schema": np.asarray(FEATURE_SCHEMA),
            "descriptive_feature_names": np.asarray(DESCRIPTIVE_FEATURE_NAMES),
            "descriptive_feature_schema": np.asarray(
                DESCRIPTIVE_FEATURE_SCHEMA
            ),
            "solidity_definition": np.asarray(VOXEL_CELL_SOLIDITY_DEFINITION),
            "feature_contract_sha256": np.asarray(FEATURE_CONTRACT_SHA256),
            "scale_zyx": np.asarray(next(iter(frames.values())).scale_zyx),
        }
    )
    atomic_npz(output, arrays)
    artifact_sha = sha256(output)
    sidecar = {
        "schema": "celltrack-alpha.g04-local-object-detection-assembly.v2",
        "status": "PASS_EXACT_COVERAGE_RADIUS30",
        "identity": identity,
        "provenance_label": PROVENANCE_LABEL,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "explicit_nonclaims": {
            "authentic_masks": False,
            "organizer_provided": False,
            "h3_authentic_masks": False,
        },
        "mapping": {
            "policy": "deterministic_nearest_computed_centroid",
            "radius_physical": MAPPING_RADIUS_PHYSICAL,
            "detection_count": int(len(payload["node_ids"])),
            "matched_count": int(np.sum(payload["matched"])),
            "maximum_distance_physical": float(
                np.max(payload["match_distance_physical"])
            ),
        },
        "frame_count": len(frames),
        "source_frame_artifacts": [
            {
                "timepoint": int(timepoint),
                "path": str(frame.path),
                "sha256": frame.sha256,
                "computed_labels_sha256": frame.computed_labels_sha256,
            }
            for timepoint, frame in sorted(frames.items())
        ],
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": artifact_sha,
        },
        "geff_access": False,
        "gt_access": False,
        "metric_access": False,
        "submission": False,
    }
    atomic_json(output.with_suffix(".json"), sidecar)
    return sidecar


def assemble(
    *,
    population_manifest: Path,
    cache_root: Path,
    shard_outputs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    population = load_population(population_manifest)
    frames = load_shard_outputs(shard_outputs)
    require_exact_coverage(population, frames)
    from g04_computed_segmentation import (
        DESCRIPTIVE_FEATURE_SCHEMA,
        TRACKASTRA_NATIVE_FEATURE_NAMES,
        load_bh_detections,
    )

    identity_rows: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    for identity in sorted(population):
        node_ids, timepoints, coords = load_bh_detections(cache_root, identity)
        expected_frames = set(population[identity]["frame_indices"])
        observed_detection_frames = set(int(item) for item in timepoints)
        if not observed_detection_frames <= expected_frames:
            raise AssemblyError(
                f"detection frames escape exact production coverage: {identity}"
            )
        identity_frames = {
            t: frames[(identity, t)] for t in population[identity]["frame_indices"]
        }
        payload = map_detections_to_objects(
            node_ids=node_ids,
            timepoints=timepoints,
            coords_zyx=coords,
            frames=identity_frames,
        )
        relative = Path("features") / f"{identity.replace(':', '__', 1)}.npz"
        output = output_dir / relative
        sidecar = publish_identity(
            output,
            identity=identity,
            payload=payload,
            frames=identity_frames,
        )
        identity_rows.append(
            {
                "identity": identity,
                "path": str(relative),
                "sha256": sidecar["artifact"]["sha256"],
                "detection_count": int(len(node_ids)),
                "frame_count": len(identity_frames),
            }
        )
        sidecar_rows.append(
            {
                "identity": identity,
                "path": str(relative.with_suffix(".json")),
                "sha256": sha256(output.with_suffix(".json")),
            }
        )
    manifest = {
        "schema": FEATURE_MANIFEST_SCHEMA,
        "status": "PASS_COMPLETE_EXACT_POPULATION_RADIUS30",
        "provenance_label": PROVENANCE_LABEL,
        "authentic_masks": False,
        "organizer_provided": False,
        "h3_authentic_masks": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(TRACKASTRA_NATIVE_FEATURE_NAMES),
        "descriptive_feature_schema": DESCRIPTIVE_FEATURE_SCHEMA,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "mapping_policy": {
            "kind": "deterministic_nearest_computed_centroid",
            "radius_physical": MAPPING_RADIUS_PHYSICAL,
        },
        "population_manifest_sha256": sha256(population_manifest),
        "identities": identity_rows,
    }
    manifest_path = output_dir / "wrfeat_manifest.json"
    atomic_json(manifest_path, manifest)
    receipt = {
        "schema": "celltrack-alpha.g04-computed-object-assembly-receipt.v2",
        "status": "PASS_READY_FOR_TRACKASTRA_WRFEAT",
        "identity_count": len(identity_rows),
        "frame_count": len(frames),
        "detection_count": sum(row["detection_count"] for row in identity_rows),
        "mapping_radius_physical": MAPPING_RADIUS_PHYSICAL,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "wrfeat_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "feature_sidecars": sidecar_rows,
        "geff_access": False,
        "gt_access": False,
        "metric_access": False,
        "submission": False,
    }
    atomic_json(output_dir / "ASSEMBLY_RECEIPT.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--population-manifest", required=True, type=Path)
    value.add_argument("--cache-root", required=True, type=Path)
    value.add_argument(
        "--shard-output", required=True, action="append", type=Path
    )
    value.add_argument("--output-dir", required=True, type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise AssemblyError("output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = assemble(
        population_manifest=args.population_manifest,
        cache_root=args.cache_root,
        shard_outputs=args.shard_output,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
