"""Direct two-frame GEFF/Zarr source for compact V45 student training.

The training candidate must not depend on an undeclared ``tracking_cellmot``
runtime package.  This reader validates the labeled directory-Zarr inputs and
loads only the two frames required for each compact-head optimization step.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch


class FastDataError(ValueError):
    """A labeled V45 fit source is malformed or outside its declared root."""


def _int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise FastDataError(f"GEFF {key} must be an integer")
    return int(value)


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise FastDataError(f"GEFF {key} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise FastDataError(f"GEFF {key} must be finite")
    return result


def _quantile(values: Mapping[str, Any], wanted: float) -> float:
    for key, value in values.items():
        try:
            if abs(float(str(key).removeprefix("q")) - wanted) <= 1.0e-9:
                result = float(value)
                if math.isfinite(result):
                    return result
        except (TypeError, ValueError):
            continue
    raise FastDataError(f"image quantile {wanted} is missing or nonfinite")


def _output_point(row: Mapping[str, Any], shape: Sequence[int]) -> tuple[int, int, int]:
    point = tuple(int(round(_float(row, axis))) for axis in ("z", "y", "x"))
    if len(shape) != 3 or any(index < 0 or index >= limit for index, limit in zip(point, shape, strict=True)):
        raise FastDataError("downsampled GEFF point lies outside the image")
    return point


class FastSource:
    """Labeled V45 windows with no frame cache and no implicit dependencies."""

    def __init__(self, data_root: Path, downsample: Sequence[int]) -> None:
        try:
            self.data_root = Path(data_root).resolve(strict=True)
        except OSError as exc:
            raise FastDataError("training data root does not exist") from exc
        if not self.data_root.is_dir():
            raise FastDataError("training data root is not a directory")
        self.downsample = tuple(int(value) for value in downsample)
        if self.downsample != (1, 4, 4):
            raise FastDataError("fast V45 training requires downsample (1, 4, 4)")
        self.window_stats: dict[str, Any] = {
            "considered": 0,
            "usable": 0,
            "collision_windows": 0,
            "skipped_window_ids": [],
        }

    def _contained(self, path: Path, *, directory: bool) -> Path:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise FastDataError(f"missing source path: {path.name}") from exc
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise FastDataError(f"source path escapes training root: {path.name}") from exc
        if directory:
            if not resolved.is_dir():
                raise FastDataError(f"source path is not a directory: {path.name}")
        elif not resolved.is_file():
            raise FastDataError(f"source path is not a regular file: {path.name}")
        return resolved

    def _paths(self, record: Any) -> tuple[Path, Path]:
        stem = getattr(record, "stem", None)
        if not isinstance(stem, str) or not stem:
            raise FastDataError("fit record has no source stem")
        zarr_path = self._contained(self.data_root / f"{stem}.zarr", directory=True)
        geff_path = self._contained(self.data_root / f"{stem}.geff", directory=True)
        required = (
            zarr_path / "zarr.json",
            zarr_path / "0" / "zarr.json",
            geff_path / "zarr.json",
            geff_path / "nodes" / "ids" / "zarr.json",
            geff_path / "edges" / "ids" / "zarr.json",
            *(geff_path / "nodes" / "props" / axis / "values" / "zarr.json" for axis in ("t", "z", "y", "x")),
        )
        for path in required:
            self._contained(path, directory=False)
        return zarr_path, geff_path

    def _metadata(
        self,
        record: Any,
    ) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], tuple[float, float, float], float, float]:
        try:
            import zarr
        except ImportError as exc:
            raise FastDataError("fast V45 training requires a declared Zarr runtime") from exc
        zarr_path, geff_path = self._paths(record)
        try:
            image = zarr.open_group(str(zarr_path), mode="r")
            array = image["0"]
            geff = zarr.open_group(str(geff_path), mode="r")
            image_shape = tuple(int(value) for value in array.shape)
            image_attrs = dict(image.attrs)
            geff_meta = json.loads((geff_path / "zarr.json").read_text(encoding="utf-8"))
            geff_attrs = dict(geff.attrs)
            if len(image_shape) != 4 or min(image_shape) < 1 or geff_meta.get("zarr_format") != 3:
                raise ValueError("invalid image or GEFF Zarr-v3 shape")
            multiscale = image_attrs["multiscales"][0]
            axes = multiscale["axes"]
            if (
                not isinstance(axes, list)
                or any(not isinstance(axis, Mapping) for axis in axes)
                or tuple(str(axis.get("name", "")).lower() for axis in axes) != ("t", "z", "y", "x")
            ):
                raise ValueError("image axes differ from t/z/y/x")
            transform = multiscale["datasets"][0]["coordinateTransformations"][0]
            if transform.get("type") != "scale" or len(transform.get("scale", ())) != 4:
                raise ValueError("image physical scale is invalid")
            scale = tuple(float(value) for value in transform["scale"][1:])
            quantiles = image_attrs["image_statistics"]["quantiles"]
            q_low, q_high = _quantile(quantiles, 0.001), _quantile(quantiles, 0.999)
            graph = geff_attrs.get("geff", geff_meta.get("attributes", {}).get("geff"))
            if not isinstance(graph, Mapping) or graph.get("directed") is not True:
                raise ValueError("GEFF must be directed")
            version = str(graph.get("geff_version", ""))
            graph_axes = graph.get("axes")
            if (
                (version != "1.1" and not version.startswith("1.1."))
                or not isinstance(graph_axes, list)
                or any(not isinstance(axis, Mapping) for axis in graph_axes)
                or tuple(str(axis.get("name", "")).lower() for axis in graph_axes) != ("t", "z", "y", "x")
            ):
                raise ValueError("GEFF version or axes are incompatible")
            graph_scale = tuple(float(axis["scale"]) for axis in graph_axes[1:])
            if any(not math.isfinite(value) or value <= 0.0 for value in (*scale, *graph_scale)):
                raise ValueError("nonpositive physical scale")
            if any(abs(left - right) > 1.0e-9 for left, right in zip(scale, graph_scale, strict=True)):
                raise ValueError("GEFF and image physical scales disagree")
            ids = np.asarray(geff["nodes/ids"])
            values = {axis: np.asarray(geff[f"nodes/props/{axis}/values"]) for axis in ("t", "z", "y", "x")}
            edge_ids = np.asarray(geff["edges/ids"])
        except (FastDataError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FastDataError("direct GEFF/Zarr-v3 metadata decode failed") from exc
        if (
            q_high <= q_low
            or ids.ndim != 1
            or len(ids) == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(values["t"].dtype, np.integer)
            or any(
                values[axis].ndim != 1
                or len(values[axis]) != len(ids)
                or not np.issubdtype(values[axis].dtype, np.number)
                for axis in values
            )
        ):
            raise FastDataError("GEFF node table is invalid")
        nodes: list[dict[str, Any]] = []
        for index, node_id in enumerate(ids.tolist()):
            row = {
                "node_id": int(node_id),
                "t": int(values["t"][index]),
                "z": float(values["z"][index]),
                "y": float(values["y"][index]),
                "x": float(values["x"][index]),
            }
            if row["node_id"] < 0 or row["t"] < 0:
                raise FastDataError("GEFF node identifiers and time must be nonnegative")
            nodes.append(row)
        names = getattr(edge_ids.dtype, "names", None)
        if names:
            if (
                edge_ids.ndim != 1
                or set(names) != {"source_id", "target_id"}
                or any(not np.issubdtype(edge_ids.dtype.fields[name][0], np.integer) for name in names)
            ):
                raise FastDataError("GEFF edge ID schema is invalid")
            edges = [{"source_id": int(row["source_id"]), "target_id": int(row["target_id"])} for row in edge_ids]
        elif edge_ids.ndim == 2 and edge_ids.shape[1:] == (2,) and np.issubdtype(edge_ids.dtype, np.integer):
            edges = [{"source_id": int(row[0]), "target_id": int(row[1])} for row in edge_ids]
        else:
            raise FastDataError("GEFF edge ID schema is invalid")
        indexed = {row["node_id"]: row for row in nodes}
        if len(indexed) != len(nodes):
            raise FastDataError("GEFF contains duplicate node IDs")
        children: set[int] = set()
        for edge in edges:
            source, target = edge["source_id"], edge["target_id"]
            if source not in indexed or target not in indexed or target in children:
                raise FastDataError("GEFF has missing endpoint or multiple parent")
            if indexed[target]["t"] != indexed[source]["t"] + 1:
                raise FastDataError("GEFF edge is not forward consecutive")
            children.add(target)
        voxel_size = tuple(value * downsample for value, downsample in zip(scale, self.downsample, strict=True))
        return array, nodes, edges, voxel_size, q_low, q_high

    def _windows(self, record: Any) -> Iterator[tuple[int, dict[str, Any]]]:
        array, nodes, edges, voxel_size, q_low, q_high = self._metadata(record)
        raw_shape = tuple(int(value) for value in array.shape[1:])
        output_shape = tuple((limit + downsample - 1) // downsample for limit, downsample in zip(raw_shape, self.downsample, strict=True))
        by_id: dict[int, dict[str, Any]] = {}
        by_time: dict[int, list[dict[str, Any]]] = {}
        for raw in nodes:
            node_id, frame = _int(raw, "node_id"), _int(raw, "t")
            if node_id in by_id or frame >= int(array.shape[0]):
                raise FastDataError("GEFF has duplicate node ID or out-of-range time")
            coordinates: dict[str, float] = {}
            for axis, downsample, raw_limit, output_limit in zip(
                ("z", "y", "x"), self.downsample, raw_shape, output_shape, strict=True
            ):
                value = _float(raw, axis)
                if value < 0.0 or value >= raw_limit:
                    raise FastDataError("GEFF coordinate lies outside raw image")
                coordinates[axis] = min(value / downsample, float(output_limit - 1))
            row = {"id": node_id, "batch": 0, "time": frame, **coordinates}
            by_id[node_id] = row
            by_time.setdefault(frame, []).append(row)
        valid_edges: list[tuple[int, int]] = []
        children: set[int] = set()
        for edge in edges:
            source, target = _int(edge, "source_id"), _int(edge, "target_id")
            if source not in by_id or target not in by_id or target in children:
                raise FastDataError("GEFF edge is invalid")
            if by_id[target]["time"] != by_id[source]["time"] + 1:
                raise FastDataError("GEFF edge is not forward consecutive")
            children.add(target)
            valid_edges.append((source, target))
        group_id = getattr(record, "group_id", None)
        video_id = getattr(record, "video_id", None)
        stem = getattr(record, "stem", None)
        if not all(isinstance(value, str) and value for value in (group_id, video_id, stem)):
            raise FastDataError("fit record identity is invalid")
        for start in range(int(array.shape[0]) - 1):
            first, second = by_time.get(start, ()), by_time.get(start + 1, ())
            if not first or not second:
                continue
            self.window_stats["considered"] += 1
            window_nodes = [{**row, "time": row["time"] - start} for row in (*first, *second)]
            points = [(row["time"], *_output_point(row, output_shape)) for row in window_nodes]
            if len(points) != len(set(points)):
                self.window_stats["collision_windows"] += 1
                self.window_stats["skipped_window_ids"].append(f"{group_id}:{start}")
                continue
            self.window_stats["usable"] += 1
            yield start, {
                "array": array,
                "nodes": window_nodes,
                "edges": [
                    {"source_id": source, "target_id": target}
                    for source, target in valid_edges
                    if by_id[source]["time"] == start and by_id[target]["time"] == start + 1
                ],
                "voxel_size_um": voxel_size,
                "q_low": q_low,
                "q_high": q_high,
                "output_shape": output_shape,
                "group_id": group_id,
            }

    def _read(self, metadata: Mapping[str, Any], start: int) -> dict[str, Any]:
        raw = np.asarray(metadata["array"][start : start + 2, ::1, ::4, ::4], dtype=np.float32)
        if raw.shape[0] != 2 or tuple(raw.shape[1:]) != tuple(metadata["output_shape"]) or not np.isfinite(raw).all():
            raise FastDataError("two-frame Zarr slab is absent or nonfinite")
        denominator = float(metadata["q_high"]) - float(metadata["q_low"])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise FastDataError("image quantile range is invalid")
        frames = torch.from_numpy(np.clip((raw - float(metadata["q_low"])) / denominator, 0.0, 1.0)).unsqueeze(0).unsqueeze(2)
        return {
            "frames": frames,
            "nodes": metadata["nodes"],
            "edges": metadata["edges"],
            "voxel_size_um": metadata["voxel_size_um"],
            "window_id": f"{metadata['group_id']}:{start}",
        }

    def batches(
        self,
        records: Sequence[Any],
        *,
        completed_windows: int,
        deadline: float,
        epochs: int = 1,
    ) -> Iterator[dict[str, Any]]:
        if type(completed_windows) is not int or completed_windows < 0 or type(epochs) is not int or epochs < 1:
            raise FastDataError("completed_windows and epochs are invalid")
        position = 0
        usable = False
        for epoch in range(1, epochs + 1):
            for record in records:
                for start, metadata in self._windows(record):
                    usable = True
                    if time.monotonic() >= deadline:
                        return
                    if position >= completed_windows:
                        batch = self._read(metadata, start)
                        batch["epoch"] = epoch
                        yield batch
                    position += 1
        if not usable:
            raise FastDataError("no usable fit-only windows remain")


__all__ = ["FastDataError", "FastSource"]
