#!/usr/bin/env python3
"""Run authenticated Trackastra with honestly computed 3-D wrfeat channels.

This adapter deliberately separates two claims:

* the masks used to derive features are COMPUTED SEGMENTATIONS; and
* the 12 values injected into the frozen Trackastra checkpoint have the exact
  semantics and order expected by that checkpoint.

It never accepts organizer/authentic-mask labels, unmatched detections, silent
zero filling, or a morphology sidecar substituted for checkpoint-native
wrfeat.  The production ``run`` command materializes divisions-enabled graph
fragments and scores them with the repository's exact scorer.  The
``synthetic-smoke`` command exercises the same contract, feature injection,
and fork-capable greedy decoder without reading campaign targets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Iterable, Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
for _location in (ROOT / "src", ROOT / "scripts"):
    if str(_location) not in sys.path:
        sys.path.insert(0, str(_location))

FEATURE_MANIFEST_SCHEMA = (
    "celltrack-alpha.g04-computed-segmentation-wrfeat-manifest.v1"
)
FEATURE_SCHEMA = "trackastra-ctc-wrfeat12.v1"
PROVENANCE_LABEL = "COMPUTED_SEGMENTATIONS"
FEATURE_NAMES = (
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
MODEL_SHA256 = (
    "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
)
MODEL_ARCHIVE_SHA256 = (
    "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e"
)
SOURCE_LICENSE_SHA256 = (
    "cc9ea92a72a63101ff425cc525faeace28ec9fc9c658a0fc01daf236e06326c9"
)
MODEL_PARAMETER_COUNT = 27_456_880
WINDOW_FRAMES = 4
MAX_TOKENS = 1_024
SCALE_ZYX = (1.625, 0.40625, 0.40625)
DEFAULT_RADIUS_UM = 12.0
DEFAULT_NEIGHBOR_CAP = 16
DEFAULT_GREEDY_THRESHOLD = 0.5
DEFAULT_TRACKASTRA_SOURCE = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
    / "run_v1/kaggle_output/trackastra-source"
)
DEFAULT_REGISTRATION_CENSUS = (
    ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json"
)

from celltrack_alpha.frame_registration import (  # noqa: E402
    CoordinateViews,
    RegistrationPlan,
    coordinate_views,
    load_registration_plan,
)


class ContractError(RuntimeError):
    """A provenance, schema, identity, inference, or score invariant failed."""


@dataclass(frozen=True)
class VideoFeatures:
    identity: str
    node_ids: Any
    timepoints: Any
    coords_zyx: Any
    features: Any
    matched: Any
    path: Path
    sha256: str


@dataclass(frozen=True)
class DetectionTable:
    node_ids: Any
    timepoints: Any
    coords_zyx: Any


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ContractError("numpy is required (use .venv313/bin/python)") from exc
    return np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value


def _np_text(value: Any, field: str) -> str:
    np = _numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ContractError(f"NPZ field {field} must be scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("ascii")
    return str(item)


def _np_bool(value: Any, field: str) -> bool:
    np = _numpy()
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "bui":
        raise ContractError(f"NPZ field {field} must be a scalar boolean")
    return bool(array.reshape(-1)[0])


def _validate_provenance(value: Mapping[str, Any], *, label: str) -> None:
    if value.get("provenance_label") != PROVENANCE_LABEL:
        raise ContractError(
            f"{label} must be labelled {PROVENANCE_LABEL}, not authentic masks"
        )
    if value.get("authentic_masks") is not False:
        raise ContractError(f"{label} must declare authentic_masks=false")
    if value.get("organizer_provided") is not False:
        raise ContractError(f"{label} must declare organizer_provided=false")


def load_feature_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path, "computed-wrfeat manifest")
    if value.get("schema") != FEATURE_MANIFEST_SCHEMA:
        raise ContractError("computed-wrfeat manifest schema mismatch")
    _validate_provenance(value, label="computed-wrfeat manifest")
    if value.get("feature_schema") != FEATURE_SCHEMA:
        raise ContractError("checkpoint-native feature schema mismatch")
    if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
        raise ContractError("checkpoint-native 12-channel order mismatch")
    rows = value.get("identities")
    if not isinstance(rows, list) or not rows:
        raise ContractError("computed-wrfeat manifest has no identities")
    seen: set[str] = set()
    root = path.parent.resolve()
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("computed-wrfeat identity row must be an object")
        identity = row.get("identity")
        relative = row.get("path")
        expected_sha = row.get("sha256")
        if not isinstance(identity, str) or ":" not in identity or identity in seen:
            raise ContractError(f"invalid or repeated feature identity: {identity!r}")
        if not isinstance(relative, str) or not relative.endswith(".npz"):
            raise ContractError(f"invalid feature path for {identity}")
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ContractError(f"feature path escapes manifest root: {identity}")
        if not candidate.is_file() or sha256(candidate) != expected_sha:
            raise ContractError(f"feature byte hash mismatch: {identity}")
        seen.add(identity)
    return value


def feature_rows(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["identity"]): row for row in manifest["identities"]}


def load_video_features(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    identity: str,
) -> VideoFeatures:
    np = _numpy()
    rows = feature_rows(manifest)
    if identity not in rows:
        raise ContractError(f"computed wrfeat missing identity: {identity}")
    row = rows[identity]
    path = (manifest_path.parent / str(row["path"])).resolve()
    required = {
        "node_ids",
        "timepoints",
        "coords_zyx",
        "features",
        "matched",
        "feature_names",
        "feature_schema",
        "provenance_label",
        "authentic_masks",
        "organizer_provided",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ContractError(f"feature artifact missing keys {sorted(missing)}: {identity}")
        node_ids = np.asarray(archive["node_ids"])
        timepoints = np.asarray(archive["timepoints"])
        coords = np.asarray(archive["coords_zyx"], dtype=np.float32)
        features = np.asarray(archive["features"], dtype=np.float32)
        matched = np.asarray(archive["matched"])
        names = tuple(str(item) for item in np.asarray(archive["feature_names"]).tolist())
        np_provenance = {
            "provenance_label": _np_text(archive["provenance_label"], "provenance_label"),
            "authentic_masks": _np_bool(archive["authentic_masks"], "authentic_masks"),
            "organizer_provided": _np_bool(
                archive["organizer_provided"], "organizer_provided"
            ),
        }
        feature_schema = _np_text(archive["feature_schema"], "feature_schema")
    _validate_provenance(np_provenance, label=f"feature artifact {identity}")
    if feature_schema != FEATURE_SCHEMA or names != FEATURE_NAMES:
        raise ContractError(f"feature channel semantics drifted: {identity}")
    if node_ids.ndim != 1 or node_ids.dtype.kind not in "iu":
        raise ContractError(f"node_ids must be a one-dimensional integer array: {identity}")
    if timepoints.shape != node_ids.shape or timepoints.dtype.kind not in "iu":
        raise ContractError(f"timepoints do not align with node_ids: {identity}")
    n = len(node_ids)
    if coords.shape != (n, 3) or features.shape != (n, 12):
        raise ContractError(f"computed feature array shape mismatch: {identity}")
    if matched.shape != (n,) or matched.dtype.kind != "b" or not matched.all():
        raise ContractError(f"unmatched detections may not be zero-filled: {identity}")
    if len(set(int(item) for item in node_ids)) != n:
        raise ContractError(f"repeated node identity in computed features: {identity}")
    if not np.isfinite(coords).all() or not np.isfinite(features).all():
        raise ContractError(f"non-finite computed features: {identity}")
    if n and np.any(np.all(features == 0, axis=1)):
        raise ContractError(f"all-zero wrfeat row is forbidden: {identity}")
    if n and np.any(features[:, 0] <= 0):
        raise ContractError(f"equivalent diameter must be positive: {identity}")
    return VideoFeatures(
        identity=identity,
        node_ids=node_ids.astype(np.int64, copy=False),
        timepoints=timepoints.astype(np.int64, copy=False),
        coords_zyx=coords,
        features=features,
        matched=matched,
        path=path,
        sha256=str(row["sha256"]),
    )


def detections_to_table(detections: Sequence[Any]) -> DetectionTable:
    np = _numpy()
    node_ids: list[Any] = []
    timepoints: list[Any] = []
    coords: list[Any] = []
    for frame in detections:
        ids = np.asarray(frame.node_ids)
        points = np.asarray(frame.coords_zyx, dtype=np.float32)
        if ids.ndim != 1 or points.shape != (len(ids), 3):
            raise ContractError("detection cache frame schema mismatch")
        node_ids.extend(ids.tolist())
        timepoints.extend([int(frame.frame_index)] * len(ids))
        coords.extend(points.tolist())
    return DetectionTable(
        node_ids=np.asarray(node_ids, dtype=np.int64),
        timepoints=np.asarray(timepoints, dtype=np.int64),
        coords_zyx=np.asarray(coords, dtype=np.float32).reshape(-1, 3),
    )


def align_features(
    detections: DetectionTable,
    computed: VideoFeatures,
    *,
    coordinate_atol: float = 1e-3,
) -> Any:
    np = _numpy()
    if len(set(int(item) for item in detections.node_ids)) != len(detections.node_ids):
        raise ContractError("detection cache repeats a node identity")
    det_index = {int(node): index for index, node in enumerate(detections.node_ids)}
    feat_index = {int(node): index for index, node in enumerate(computed.node_ids)}
    if set(det_index) != set(feat_index):
        missing = sorted(set(det_index) - set(feat_index))[:5]
        extra = sorted(set(feat_index) - set(det_index))[:5]
        raise ContractError(f"feature/detection node mismatch; missing={missing}, extra={extra}")
    aligned = np.empty((len(detections.node_ids), 12), dtype=np.float32)
    for node, det_i in det_index.items():
        feat_i = feat_index[node]
        if int(detections.timepoints[det_i]) != int(computed.timepoints[feat_i]):
            raise ContractError(f"feature timepoint mismatch for node {node}")
        if not np.allclose(
            detections.coords_zyx[det_i],
            computed.coords_zyx[feat_i],
            rtol=0.0,
            atol=coordinate_atol,
        ):
            raise ContractError(f"feature coordinate mismatch for node {node}")
        aligned[det_i] = computed.features[feat_i]
    return aligned


def validate_asset_audit(path: Path, checkpoint: Path, archive: Path) -> dict[str, Any]:
    value = _read_json(path, "Trackastra asset audit")
    assets = value.get("assets", {})
    checkpoint_meta = value.get("checkpoint", {})
    interface = value.get("interface", {})
    if (
        value.get("schema") != "celltrack-alpha.trackastra-pretrained-asset-audit.v1"
        or assets.get("checkpoint_member_sha256") != MODEL_SHA256
        or assets.get("checkpoint_archive_sha256") != MODEL_ARCHIVE_SHA256
        or assets.get("source_license") != "BSD-3-Clause"
        or assets.get("source_license_sha256") != SOURCE_LICENSE_SHA256
        or checkpoint_meta.get("architecture", {}).get("feat_dim") != 12
        or checkpoint_meta.get("training_features") != "wrfeat"
        or interface.get("required_feature_count") != 12
        or interface.get("required_feature_family") != "wrfeat-regionprops2"
    ):
        raise ContractError("Trackastra asset audit is not the authenticated wrfeat model")
    if sha256(checkpoint) != MODEL_SHA256 or sha256(archive) != MODEL_ARCHIVE_SHA256:
        raise ContractError("Trackastra checkpoint bytes do not match authenticated audit")
    return value


def configure_imports(trackastra_source: Path) -> None:
    for location in (trackastra_source, ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    # The model imports matplotlib through optional visualization helpers.
    if "matplotlib" not in sys.modules:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")


def load_model(
    *,
    trackastra_source: Path,
    checkpoint: Path,
    checkpoint_archive: Path,
    device: str,
) -> Any:
    configure_imports(trackastra_source)
    try:
        import torch
        import yaml
        from trackastra.model.model import TrackingTransformer
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ContractError(
            "Trackastra runtime dependencies are missing (use .venv313/bin/python)"
        ) from exc
    with zipfile.ZipFile(checkpoint_archive) as bundle:
        config = yaml.safe_load(bundle.read("ctc/config.yaml"))
    if config.get("feat_dim") != 12 or config.get("coord_dim") != 3:
        raise ContractError("checkpoint architecture is not 3-D wrfeat12")
    with torch.device("meta"):
        model = TrackingTransformer(**config)
    state = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True, assign=True)
    model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != MODEL_PARAMETER_COUNT:
        raise ContractError("Trackastra parameter count drifted")
    return model


def cache_identity(meta: Mapping[str, Any]) -> Any:
    from celltrack_alpha import bh_model

    model = meta["contract"]["model"]
    return bh_model.CacheIdentity(
        str(model["config_sha256"]),
        str(model["source_sha256"]),
        str(model["weight_sha256"]),
        int(model["pair_batch_size"]),
        int(model["view_batch_size"]),
    )


def load_detections(cache_root: Path, identity: str) -> Sequence[Any]:
    from celltrack_alpha import bh_model

    slug = identity.replace(":", "__", 1)
    meta = _read_json(cache_root / f"{slug}.json", f"detection metadata {identity}")
    return bh_model.load_sparse_cache_detections(
        cache_root / f"{slug}.bhc", expected_identity=cache_identity(meta)
    )


def _node_maps(
    table: DetectionTable,
    aligned_features: Any,
    *,
    candidate_coords_zyx: Any | None = None,
) -> tuple[dict[int, Any], dict[int, Any]]:
    np = _numpy()
    geometry = (
        np.asarray(table.coords_zyx, dtype=np.float32)
        if candidate_coords_zyx is None
        else np.asarray(candidate_coords_zyx, dtype=np.float32)
    )
    if geometry.shape != np.asarray(table.coords_zyx).shape or not np.isfinite(geometry).all():
        raise ContractError("candidate geometry coordinates are malformed")
    by_node: dict[int, Any] = {}
    by_feature: dict[int, Any] = {}
    for index, node_value in enumerate(table.node_ids):
        node = int(node_value)
        by_node[node] = (int(table.timepoints[index]), geometry[index])
        by_feature[node] = aligned_features[index]
    return by_node, by_feature


def registered_table_views(
    table: DetectionTable,
    plan: RegistrationPlan,
) -> CoordinateViews:
    """Create the mandatory geometry view after original-coordinate feature alignment."""

    try:
        return coordinate_views(plan, table.timepoints, table.coords_zyx)
    except ValueError as exc:
        raise ContractError(f"registration preprocessing failed: {exc}") from exc


def infer_association(
    model: Any,
    ordered_nodes: Sequence[int],
    by_node: Mapping[int, Any],
    by_feature: Mapping[int, Any],
    *,
    device: str,
) -> Any:
    np = _numpy()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ContractError("torch is required for Trackastra inference") from exc
    coordinates = np.stack(
        [
            np.concatenate(
                (
                    np.asarray((by_node[node][0],), dtype=np.float32),
                    np.asarray(by_node[node][1], dtype=np.float32),
                )
            )
            for node in ordered_nodes
        ]
    )
    feature_array = np.stack(
        [np.asarray(by_feature[node], dtype=np.float32) for node in ordered_nodes]
    )
    if feature_array.shape != (len(ordered_nodes), 12):
        raise ContractError("inference feature tensor is not wrfeat12")
    coords = torch.from_numpy(coordinates).unsqueeze(0).to(device)
    timepoints = torch.from_numpy(coordinates[:, 0].astype(np.int64)).unsqueeze(0).to(device)
    features = torch.from_numpy(feature_array).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(coords, features=features)
        association = model.normalize_output(logits, timepoints, coords)[0]
    result = association.detach().to(device="cpu", dtype=torch.float32).numpy()
    if result.shape != (len(ordered_nodes), len(ordered_nodes)):
        raise ContractError("Trackastra association matrix shape drifted")
    return result, {node: index for index, node in enumerate(ordered_nodes)}


def source_chunks(
    source_ids: Any,
    source_coords: Any,
    target_ids: Any,
    target_coords: Any,
    *,
    radius_um: float,
    neighbor_cap: int,
    initial_chunk: int = 128,
) -> Iterable[tuple[list[tuple[int, list[tuple[int, float]]]], set[int]]]:
    np = _numpy()
    scale = np.asarray(SCALE_ZYX, dtype=np.float64)
    source_physical = source_coords * scale[None, :]
    target_physical = target_coords * scale[None, :]
    order = np.lexsort((source_coords[:, 2], source_coords[:, 1], source_coords[:, 0]))
    pending = [order[start : start + initial_chunk] for start in range(0, len(order), initial_chunk)]
    while pending:
        indices = pending.pop(0)
        candidates: list[tuple[int, list[tuple[int, float]]]] = []
        required_targets: set[int] = set()
        for source_index in indices:
            distances = np.linalg.norm(
                target_physical - source_physical[int(source_index)], axis=1
            )
            hits = [int(index) for index in np.flatnonzero(distances <= radius_um + 1e-9)]
            hits.sort(key=lambda index: (float(distances[index]), int(target_ids[index])))
            row = [
                (int(target_ids[index]), float(distances[index]))
                for index in hits[:neighbor_cap]
            ]
            required_targets.update(target for target, _distance in row)
            candidates.append((int(source_ids[int(source_index)]), row))
        required_count = len(indices) + len(required_targets)
        if required_count > MAX_TOKENS:
            if len(indices) == 1:
                raise ContractError("one source neighborhood exceeds 1024 tokens")
            middle = len(indices) // 2
            pending.insert(0, indices[middle:])
            pending.insert(0, indices[:middle])
            continue
        yield candidates, required_targets


def infer_video_bank(
    *,
    table: DetectionTable,
    aligned_features: Any,
    model: Any,
    device: str,
    radius_um: float = DEFAULT_RADIUS_UM,
    neighbor_cap: int = DEFAULT_NEIGHBOR_CAP,
    registered_views: CoordinateViews | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    np = _numpy()
    if registered_views is None:
        candidate_coords = np.asarray(table.coords_zyx, dtype=np.float32)
        registration_receipt = None
    else:
        if (
            not np.array_equal(registered_views.timepoints, table.timepoints)
            or not np.array_equal(registered_views.original_zyx, table.coords_zyx)
        ):
            raise ContractError("registration views do not match original detection table")
        candidate_coords = registered_views.candidate_zyx
        registration_receipt = dict(registered_views.receipt)
    by_node, by_feature = _node_maps(
        table,
        aligned_features,
        candidate_coords_zyx=candidate_coords,
    )
    frame_nodes: dict[int, tuple[Any, Any]] = {}
    for t in sorted(set(int(item) for item in table.timepoints)):
        mask = table.timepoints == t
        frame_nodes[t] = (table.node_ids[mask], candidate_coords[mask])
    if not frame_nodes:
        raise ContractError("detection cache has no frames")
    min_frame, max_frame = min(frame_nodes), max(frame_nodes)
    edges: list[tuple[int, int, float, float]] = []
    calls = 0
    peak_tokens = 0
    for source_t in range(min_frame, max_frame):
        if source_t not in frame_nodes or source_t + 1 not in frame_nodes:
            continue
        source_ids, source_coords = frame_nodes[source_t]
        target_ids, target_coords = frame_nodes[source_t + 1]
        if not len(source_ids) or not len(target_ids):
            continue
        start = min(max(source_t - 1, min_frame), max(min_frame, max_frame - 3))
        context = [
            int(node)
            for frame in range(start, start + WINDOW_FRAMES)
            if frame in frame_nodes
            for node in frame_nodes[frame][0]
        ]
        for candidates, required_targets in source_chunks(
            source_ids,
            source_coords,
            target_ids,
            target_coords,
            radius_um=radius_um,
            neighbor_cap=neighbor_cap,
        ):
            required_sources = {source for source, _targets in candidates}
            required = required_sources | required_targets
            centroid = np.mean([by_node[node][1] for node in required], axis=0)
            optional = sorted(
                set(context) - required,
                key=lambda node: (
                    float(np.sum((by_node[node][1] - centroid) ** 2)),
                    by_node[node][0],
                    node,
                ),
            )
            selected = required | set(optional[: MAX_TOKENS - len(required)])
            ordered = tuple(sorted(selected, key=lambda node: (by_node[node][0], node)))
            association, index = infer_association(
                model, ordered, by_node, by_feature, device=device
            )
            calls += 1
            peak_tokens = max(peak_tokens, len(ordered))
            for source, targets in candidates:
                for target, distance in targets:
                    edges.append(
                        (
                            source,
                            target,
                            float(association[index[source], index[target]]),
                            distance,
                        )
                    )
    nodes = np.column_stack(
        (
            table.node_ids,
            table.timepoints,
            table.coords_zyx,
        )
    ).astype(np.float64, copy=False)
    nodes = nodes[np.lexsort((nodes[:, 0], nodes[:, 1]))]
    edge_array = np.asarray(sorted(edges), dtype=np.float64).reshape(-1, 4)
    return nodes, edge_array, {
        "inference_calls": calls,
        "peak_tokens": peak_tokens,
        "feature_rows": len(table.node_ids),
        "nonzero_feature_rows": int(np.count_nonzero(np.any(aligned_features != 0, axis=1))),
        "registration": registration_receipt,
        "candidate_coordinates": "registered" if registered_views is not None else "original",
        "output_coordinates": "original",
    }


def solve_divisions(edges: Any, *, threshold: float = DEFAULT_GREEDY_THRESHOLD) -> tuple[tuple[tuple[int, int], ...], dict[str, int]]:
    """Trackastra's global greedy decoder with ``allow_divisions=True``."""
    incoming: set[int] = set()
    outgoing: Counter[int] = Counter()
    selected: list[tuple[int, int]] = []
    considered = 0
    for source_f, target_f, weight_f, _distance_f in sorted(
        edges, key=lambda row: (-float(row[2]), int(row[0]), int(row[1]))
    ):
        source, target, weight = int(source_f), int(target_f), float(weight_f)
        if weight < threshold:
            break
        considered += 1
        if target in incoming or outgoing[source] >= 2:
            continue
        incoming.add(target)
        outgoing[source] += 1
        selected.append((source, target))
    return tuple(sorted(selected)), {
        "threshold_considered": considered,
        "selected_edges": len(selected),
        "forks": sum(value == 2 for value in outgoing.values()),
    }


def write_graph_fragment(
    *, identity: str, nodes: Any, selected: Sequence[tuple[int, int]], path: Path
) -> dict[str, Any]:
    from celltrack_alpha.kaggle_calibration import (
        SubmissionEdge,
        SubmissionGraph,
        SubmissionNode,
        validate_graphs,
        write_submission,
    )

    dataset = identity.split(":", 1)[1]
    used = {node for edge in selected for node in edge}
    node_map = {
        int(node): SubmissionNode(
            int(node), int(t), int(round(z)), int(round(y)), int(round(x))
        )
        for node, t, z, y, x in nodes
        if int(node) in used
    }
    graph = SubmissionGraph(
        dataset=dataset,
        nodes=node_map,
        edges=[SubmissionEdge(source, target) for source, target in selected],
    )
    validate_graphs({dataset: graph})
    write_submission({dataset: graph}, path)
    return {"nodes": len(graph.nodes), "edges": len(graph.edges), "sha256": sha256(path)}


def load_population(path: Path, scope: str) -> tuple[str, ...]:
    value = _read_json(path, "population manifest")
    identities = value.get("identities")
    if identities is None and isinstance(value.get("entries"), list):
        identities = [row.get("identity") for row in value["entries"]]
    if not isinstance(identities, list) or not all(isinstance(item, str) for item in identities):
        raise ContractError("population manifest identities are malformed")
    result = tuple(identities)
    if len(set(result)) != len(result):
        raise ContractError("population manifest repeats identities")
    family_counts = Counter(identity.split(":", 1)[1].split("_", 1)[0] for identity in result)
    expected = {
        "held-fit105": (105, Counter({"44b6": 36, "6bba": 69})),
        "cv57": (57, Counter({"44b6": 21, "6bba": 36})),
    }[scope]
    if len(result) != expected[0] or family_counts != expected[1]:
        raise ContractError(
            f"{scope} population mismatch: n={len(result)}, families={dict(family_counts)}"
        )
    return result


def score_fragments(
    *, fragment_root: Path, identities: Sequence[str], archive: Path
) -> dict[str, Any]:
    import scoreall
    from zarr.storage import ZipStore

    paths = scoreall.resolve_fragment_paths(fragment_root)
    manifest = scoreall.validate_fragment_set(paths, tuple(identities))
    with ZipStore(archive, mode="r") as store:
        rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
    summary = scoreall.official_summary(rows)
    safe_summary = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in summary.items()
    }
    return {
        "summary": safe_summary,
        "coverage": coverage,
        "input_provenance": provenance,
        "edge_counts": {
            key: int(sum(int(row[key]) for row in rows))
            for key in ("edge_tp", "edge_fp", "edge_fn")
        },
    }


def validate_command(args: argparse.Namespace) -> int:
    manifest = load_feature_manifest(args.features_manifest)
    identities = (
        load_population(args.population_manifest, args.scope)
        if args.population_manifest is not None
        else tuple(feature_rows(manifest))
    )
    missing = set(identities) - set(feature_rows(manifest))
    if missing:
        raise ContractError(f"computed wrfeat is missing requested identities: {sorted(missing)[:5]}")
    rows = []
    for identity in identities:
        video = load_video_features(args.features_manifest, manifest, identity)
        rows.append(
            {
                "identity": identity,
                "nodes": len(video.node_ids),
                "sha256": video.sha256,
            }
        )
    report = {
        "schema": "celltrack-alpha.g04-computed-wrfeat-validation.v1",
        "status": "PASS",
        "provenance_label": PROVENANCE_LABEL,
        "authentic_masks": False,
        "organizer_provided": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "identity_count": len(rows),
        "node_count": sum(row["nodes"] for row in rows),
        "rows": rows,
    }
    if args.output is not None:
        if args.output.exists():
            raise ContractError(f"refusing to overwrite output: {args.output}")
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_command(args: argparse.Namespace) -> int:
    np = _numpy()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite output directory: {args.output}")
    if not (0.0 < args.threshold <= 1.0):
        raise ContractError("greedy threshold must be in (0, 1]")
    if args.radius_um <= 0.0 or args.neighbor_cap < 1:
        raise ContractError("candidate radius and neighbor cap must be positive")
    manifest = load_feature_manifest(args.features_manifest)
    identities = load_population(args.population_manifest, args.scope)
    missing = set(identities) - set(feature_rows(manifest))
    if missing:
        raise ContractError(f"computed wrfeat is missing requested identities: {sorted(missing)[:5]}")
    validate_asset_audit(args.asset_audit, args.checkpoint, args.checkpoint_archive)
    import torch

    torch.manual_seed(20260803)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260803)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = load_model(
        trackastra_source=args.trackastra_source,
        checkpoint=args.checkpoint,
        checkpoint_archive=args.checkpoint_archive,
        device=args.device,
    )
    bank_root = args.output / "bank"
    fragment_root = args.output / "fragments"
    rows = []
    for ordinal, identity in enumerate(identities, start=1):
        computed = load_video_features(args.features_manifest, manifest, identity)
        detections = detections_to_table(load_detections(args.cache_root, identity))
        aligned = align_features(detections, computed)
        registration_plan = load_registration_plan(args.registration_census, identity)
        views = registered_table_views(detections, registration_plan)
        nodes, edges, runtime = infer_video_bank(
            table=detections,
            aligned_features=aligned,
            model=model,
            device=args.device,
            radius_um=args.radius_um,
            neighbor_cap=args.neighbor_cap,
            registered_views=views,
        )
        selected, decoder = solve_divisions(edges, threshold=args.threshold)
        slug = identity.replace(":", "__", 1)
        bank_path = bank_root / f"{slug}.npz"
        fragment_path = fragment_root / f"{slug}.csv"
        atomic_npz(bank_path, nodes=nodes, edges=edges)
        graph = write_graph_fragment(
            identity=identity, nodes=nodes, selected=selected, path=fragment_path
        )
        row = {
            "identity": identity,
            "feature_sha256": computed.sha256,
            "bank_sha256": sha256(bank_path),
            "candidate_edges": len(edges),
            **runtime,
            **decoder,
            "graph": graph,
        }
        rows.append(row)
        atomic_json(args.output / "progress" / f"{slug}.json", row)
        print(f"g04 wrfeat {ordinal:03d}/{len(identities)} {identity}", flush=True)
    scored = score_fragments(
        fragment_root=fragment_root, identities=identities, archive=args.archive
    )
    report = {
        "schema": "celltrack-alpha.g04-trackastra-computed-wrfeat-score.v1",
        "status": "COMPLETE",
        "scope": args.scope,
        "provenance_label": PROVENANCE_LABEL,
        "authentic_masks": False,
        "organizer_provided": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "checkpoint_sha256": MODEL_SHA256,
        "checkpoint_archive_sha256": MODEL_ARCHIVE_SHA256,
        "inference": {
            "window_frames": WINDOW_FRAMES,
            "max_tokens": MAX_TOKENS,
            "radius_um": args.radius_um,
            "neighbor_cap": args.neighbor_cap,
            "scale_zyx": list(SCALE_ZYX),
            "registration_preprocessing": {
                "standing": True,
                "census": str(args.registration_census),
                "census_sha256": sha256(args.registration_census),
                "candidate_topology_and_geometry": "registered",
                "feature_sampling_and_output": "original",
                "applications_per_video": 1,
            },
        },
        "decoder": {
            "name": "trackastra-global-descending-weight-greedy",
            "threshold": args.threshold,
            "divisions": True,
            "maximum_out_degree": 2,
        },
        "identity_count": len(rows),
        "fork_count": sum(row["forks"] for row in rows),
        "rows": rows,
        "score": scored,
    }
    atomic_json(args.output / "REPORT.json", report)
    print(json.dumps({"status": "COMPLETE", "scope": args.scope, **scored["summary"]}, sort_keys=True))
    return 0


class _SyntheticFrame:
    def __init__(self, frame_index: int, node_ids: Any, coords_zyx: Any):
        self.frame_index = frame_index
        self.node_ids = node_ids
        self.coords_zyx = coords_zyx


class _SyntheticForkModel:
    """Tiny deterministic stand-in that proves the 12-D tensor reaches forward()."""

    def __init__(self):
        import torch

        self.torch = torch
        self.last_features = None

    def __call__(self, coords: Any, *, features: Any) -> Any:
        self.last_features = features.detach().cpu().clone()
        # Positive logits from node 1 to both next-frame nodes; other logits low.
        logits = self.torch.full(
            (coords.shape[0], coords.shape[1], coords.shape[1]),
            -8.0,
            dtype=features.dtype,
            device=features.device,
        )
        logits[:, 0, 1] = features[:, 0, 0] + 2.0
        logits[:, 0, 2] = features[:, 0, 0] + 1.0
        return logits

    def normalize_output(self, logits: Any, timepoints: Any, coords: Any) -> Any:
        return self.torch.sigmoid(logits)


def synthetic_smoke_command(args: argparse.Namespace) -> int:
    np = _numpy()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite output: {args.output}")
    with tempfile.TemporaryDirectory(prefix="g04-wrfeat-smoke-") as temporary:
        root = Path(temporary)
        identity = "synthetic:wrfeat_fork"
        node_ids = np.asarray([1, 2, 3], dtype=np.int64)
        timepoints = np.asarray([0, 1, 1], dtype=np.int64)
        coords = np.asarray([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
        features = np.zeros((3, 12), dtype=np.float32)
        features[:, 0] = [3.0, 2.0, 2.5]
        features[:, 1] = [0.8, 0.4, 0.5]
        artifact = root / "video.npz"
        atomic_npz(
            artifact,
            node_ids=node_ids,
            timepoints=timepoints,
            coords_zyx=coords,
            features=features,
            matched=np.ones(3, dtype=np.bool_),
            feature_names=np.asarray(FEATURE_NAMES),
            feature_schema=np.asarray(FEATURE_SCHEMA),
            provenance_label=np.asarray(PROVENANCE_LABEL),
            authentic_masks=np.asarray(False),
            organizer_provided=np.asarray(False),
        )
        manifest_path = root / "manifest.json"
        atomic_json(
            manifest_path,
            {
                "schema": FEATURE_MANIFEST_SCHEMA,
                "provenance_label": PROVENANCE_LABEL,
                "authentic_masks": False,
                "organizer_provided": False,
                "feature_schema": FEATURE_SCHEMA,
                "feature_names": list(FEATURE_NAMES),
                "identities": [
                    {"identity": identity, "path": "video.npz", "sha256": sha256(artifact)}
                ],
            },
        )
        manifest = load_feature_manifest(manifest_path)
        computed = load_video_features(manifest_path, manifest, identity)
        frames = [
            _SyntheticFrame(0, node_ids[:1], coords[:1]),
            _SyntheticFrame(1, node_ids[1:], coords[1:]),
        ]
        detections = detections_to_table(frames)
        aligned = align_features(detections, computed)
        model = _SyntheticForkModel()
        nodes, edges, runtime = infer_video_bank(
            table=detections,
            aligned_features=aligned,
            model=model,
            device="cpu",
            radius_um=10.0,
            neighbor_cap=8,
        )
        selected, decoder = solve_divisions(edges, threshold=0.5)
        if selected != ((1, 2), (1, 3)) or decoder["forks"] != 1:
            raise ContractError("synthetic divisions-enabled decode did not emit one fork")
        if model.last_features is None or not np.array_equal(
            model.last_features.numpy()[0], aligned
        ):
            raise ContractError("synthetic model did not receive the computed 12-D tensor")
        report = {
            "schema": "celltrack-alpha.g04-trackastra-computed-wrfeat-synthetic-smoke.v1",
            "status": "PASS",
            "provenance_label": PROVENANCE_LABEL,
            "authentic_masks": False,
            "organizer_provided": False,
            "feature_schema": FEATURE_SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "feature_tensor_injected_exactly": True,
            "node_count": len(nodes),
            "candidate_edge_count": len(edges),
            "selected_edges": [list(edge) for edge in selected],
            "decoder": {
                "divisions": True,
                "threshold": 0.5,
                "forks": decoder["forks"],
            },
            "runtime": runtime,
        }
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate computed wrfeat bytes")
    validate.add_argument("--features-manifest", type=Path, required=True)
    validate.add_argument("--population-manifest", type=Path)
    validate.add_argument("--scope", choices=("held-fit105", "cv57"), default="held-fit105")
    validate.add_argument("--output", type=Path)
    validate.set_defaults(function=validate_command)

    smoke = subparsers.add_parser(
        "synthetic-smoke", help="exercise provenance, injection, and fork decoding"
    )
    smoke.add_argument("--output", type=Path, required=True)
    smoke.set_defaults(function=synthetic_smoke_command)

    run = subparsers.add_parser("run", help="infer and exactly score one complete population")
    run.add_argument("--scope", choices=("held-fit105", "cv57"), required=True)
    run.add_argument("--features-manifest", type=Path, required=True)
    run.add_argument("--population-manifest", type=Path, required=True)
    run.add_argument("--cache-root", type=Path, required=True)
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument(
        "--registration-census",
        type=Path,
        default=DEFAULT_REGISTRATION_CENSUS,
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    run.add_argument("--trackastra-source", type=Path, default=DEFAULT_TRACKASTRA_SOURCE)
    run.add_argument(
        "--checkpoint", type=Path, default=ROOT / "assets/trackastra-authentic-v1/model.pt"
    )
    run.add_argument(
        "--checkpoint-archive",
        type=Path,
        default=ROOT / "assets/trackastra-authentic-v1/ctc.zip",
    )
    run.add_argument(
        "--asset-audit",
        type=Path,
        default=ROOT / "reports/g02/trackastra_pretrained_asset_audit_v1.json",
    )
    run.add_argument("--radius-um", type=float, default=DEFAULT_RADIUS_UM)
    run.add_argument("--neighbor-cap", type=int, default=DEFAULT_NEIGHBOR_CAP)
    run.add_argument("--threshold", type=float, default=DEFAULT_GREEDY_THRESHOLD)
    run.set_defaults(function=run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.function(args))
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
