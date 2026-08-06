#!/usr/bin/env python3
"""CPU-only reader for the frozen V45 sparse-edge cache and fitted head.

This module implements only the already-frozen deployment-time operations used
by the division information audit: cache decoding, radius/cap candidate gating,
the released projection/pair MLP, the auxiliary motion readout, and the fitted
578-parameter residual.  It deliberately contains no fitting or graph mutation.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import zipfile
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erf


CACHE_MAGIC = b"CTBHMV3\x00"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


class RuntimeError(ValueError):
    """Frozen runtime input failed closed validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unpack_array(record: dict[str, Any], dtype: str) -> np.ndarray:
    wanted = np.dtype(dtype)
    if record.get("dtype") != wanted.str:
        raise RuntimeError("cache array dtype mismatch")
    shape = tuple(int(value) for value in record["shape"])
    raw = base64.b64decode(record["data"].encode("ascii"), validate=True)
    if len(raw) != math.prod(shape) * wanted.itemsize:
        raise RuntimeError("cache array byte count mismatch")
    result = np.frombuffer(raw, dtype=wanted).reshape(shape).copy()
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise RuntimeError("cache contains nonfinite values")
    return result


def load_cache(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if payload[: len(CACHE_MAGIC)] != CACHE_MAGIC:
        raise RuntimeError("cache magic mismatch")
    body_length = int.from_bytes(payload[len(CACHE_MAGIC) : len(CACHE_MAGIC) + 8], "big")
    start = len(CACHE_MAGIC) + 8
    end = start + body_length
    if end + 32 != len(payload) or hashlib.sha256(payload[:end]).digest() != payload[end:]:
        raise RuntimeError("cache digest or length mismatch")
    record = json.loads(payload[start:end].decode("ascii"))
    detections: dict[int, dict[str, Any]] = {}
    for row in record["detections"]:
        frame = int(row["frame_index"])
        detections[frame] = {
            "frame": frame,
            "node_ids": _unpack_array(row["node_ids"], "<i8"),
            "coords": _unpack_array(row["voxel_coords_zyx"], "<i8"),
            "grid_coords": _unpack_array(row["grid_coords_zyx"], "<i8"),
            "scores": _unpack_array(row["scores"], "<f4"),
            "grid_shape": tuple(int(value) for value in row["grid_shape_zyx"]),
        }
    contexts: dict[int, dict[str, Any]] = {}
    for row in record["contexts"]:
        frame = int(row["source_frame"])
        contexts[frame] = {
            "source_frame": frame,
            "target_frame": int(row["target_frame"]),
            "source_ids": _unpack_array(row["source_ids"], "<i8"),
            "target_ids": _unpack_array(row["target_ids"], "<i8"),
            "source_feat": _unpack_array(row["source_features"], "<f4"),
            "target_feat": _unpack_array(row["target_features"], "<f4"),
            "grid_shape": tuple(int(value) for value in row["grid_shape_zyx"]),
        }
    return {
        "detections": detections,
        "contexts": contexts,
        "hashes": record["hashes"],
        "stored_edge_pairs": sum(
            len(_unpack_array(row["source_ids"], "<i8")) for row in record["edges"]
        ),
    }


class _StorageType:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class _TorchlessUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return lambda storage, offset, shape, stride, *unused: {
                "storage": storage,
                "offset": int(offset),
                "shape": tuple(int(value) for value in shape),
                "stride": tuple(int(value) for value in stride),
            }
        if module == "torch" and name.endswith("Storage"):
            mapping = {
                "FloatStorage": "<f4",
                "DoubleStorage": "<f8",
                "HalfStorage": "<f2",
                "LongStorage": "<i8",
                "IntStorage": "<i4",
                "ByteStorage": "u1",
                "BoolStorage": "?",
            }
            if name not in mapping:
                raise pickle.UnpicklingError(f"unsupported storage {name}")
            return _StorageType(mapping[name])
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

    def persistent_load(self, pid: Any) -> dict[str, Any]:
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
            raise pickle.UnpicklingError("unsupported persistent record")
        return {"dtype": pid[1].dtype, "key": str(pid[2]), "size": int(pid[4])}


def load_torch_zip(path: Path) -> Any:
    with zipfile.ZipFile(path) as archive:
        data_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        root = data_name.rsplit("/", 1)[0]
        payload = _TorchlessUnpickler(io.BytesIO(archive.read(data_name))).load()

        def materialize(value: Any) -> Any:
            if isinstance(value, dict) and {"storage", "offset", "shape", "stride"} <= set(value):
                storage = value["storage"]
                raw = archive.read(f"{root}/data/{storage['key']}")
                flat = np.frombuffer(raw, dtype=np.dtype(storage["dtype"]))
                offset = int(value["offset"])
                shape = tuple(value["shape"])
                stride = tuple(value["stride"])
                view = np.lib.stride_tricks.as_strided(
                    flat[offset:],
                    shape=shape,
                    strides=tuple(step * flat.dtype.itemsize for step in stride),
                    writeable=False,
                )
                return np.array(view, copy=True)
            if isinstance(value, dict):
                return type(value)((key, materialize(child)) for key, child in value.items())
            if isinstance(value, list):
                return [materialize(child) for child in value]
            if isinstance(value, tuple):
                return tuple(materialize(child) for child in value)
            return value

        return materialize(payload)


def load_json_head(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="ascii"))
    tensors: dict[str, np.ndarray] = {}
    for record in payload["weights"]["tensors"]:
        raw = base64.b64decode(record["data_base64"].encode("ascii"), validate=True)
        array = np.frombuffer(raw, dtype="<f4").reshape(tuple(record["shape"])).copy()
        tensors[str(record["name"])] = array
    return tensors


def _linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return value @ weight.T + bias


def _layer_norm(
    value: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    value32 = np.asarray(value, dtype=np.float32)
    mean = value32.mean(axis=-1, keepdims=True, dtype=np.float32)
    variance = ((value32 - mean) ** 2).mean(axis=-1, keepdims=True, dtype=np.float32)
    return ((value32 - mean) / np.sqrt(variance + np.float32(epsilon))) * weight + bias


def _gelu(value: np.ndarray) -> np.ndarray:
    return np.asarray(
        0.5 * value * (1.0 + erf(value / np.float32(math.sqrt(2.0)))),
        dtype=np.float32,
    )


def positional_features(coords: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    coords32 = coords.astype(np.float32, copy=False)
    frequencies = 2.0 ** np.arange(4, dtype=np.float32)
    parts = []
    for axis, size in enumerate(shape):
        angles = (coords32[:, axis] / float(size))[:, None] * frequencies[None, :] * np.pi
        parts.extend((np.sin(angles), np.cos(angles)))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _local_counts(points: np.ndarray, radius: float = 10.0) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=np.float64)
    tree = cKDTree(points)
    return np.asarray(
        [
            max(0, int(tree.query_ball_point(point, radius, return_length=True)) - 1)
            for point in points
        ],
        dtype=np.float64,
    )


def parent_candidates(
    source: dict[str, Any],
    target: dict[str, Any],
    parent_id: int,
) -> dict[str, np.ndarray]:
    matches = np.flatnonzero(source["node_ids"] == parent_id)
    if len(matches) != 1:
        return {
            "source_index": np.empty(0, dtype=np.int64),
            "target_index": np.empty(0, dtype=np.int64),
            "target_ids": np.empty(0, dtype=np.int64),
            "displacement": np.empty((0, 3), dtype=np.float64),
            "distance": np.empty(0, dtype=np.float64),
            "source_density": np.empty(0, dtype=np.float64),
            "target_density": np.empty(0, dtype=np.float64),
        }
    source_index = int(matches[0])
    source_points = source["coords"].astype(np.float64) * SCALE_ZYX
    target_points = target["coords"].astype(np.float64) * SCALE_ZYX
    tree = cKDTree(target_points)
    neighbors = [int(value) for value in tree.query_ball_point(source_points[source_index], 10.0)]
    delta = (
        target_points[np.asarray(neighbors, dtype=np.int64)] - source_points[source_index]
        if neighbors
        else np.empty((0, 3), dtype=np.float64)
    )
    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta)) if neighbors else np.empty(0)
    order = sorted(
        range(len(neighbors)),
        key=lambda local: (
            float(distance[local]),
            int(target["node_ids"][neighbors[local]]),
            neighbors[local],
        ),
    )[:8]
    target_index = np.asarray([neighbors[local] for local in order], dtype=np.int64)
    displacement = np.asarray([delta[local] for local in order], dtype=np.float64).reshape(-1, 3)
    distances = np.asarray([distance[local] for local in order], dtype=np.float64)
    source_density_all = _local_counts(source_points)
    target_density_all = _local_counts(target_points)
    return {
        "source_index": np.full(len(order), source_index, dtype=np.int64),
        "target_index": target_index,
        "target_ids": target["node_ids"][target_index],
        "displacement": displacement,
        "distance": distances,
        "source_density": np.full(len(order), source_density_all[source_index], dtype=np.float64),
        "target_density": target_density_all[target_index],
    }


class FrozenEdgeRuntime:
    def __init__(self, checkpoint: Path, auxiliary: Path, head: Path) -> None:
        state = load_torch_zip(checkpoint)
        self.core = {
            name: np.asarray(state[name], dtype=np.float32)
            for name in (
                "transformer.proj.weight",
                "transformer.proj.bias",
                "transformer.norm_in.weight",
                "transformer.norm_in.bias",
                "transformer.norm_out.weight",
                "transformer.norm_out.bias",
                "transformer.pair_mlp.0.weight",
                "transformer.pair_mlp.0.bias",
                "transformer.pair_mlp.3.weight",
                "transformer.pair_mlp.3.bias",
                "transformer.pair_mlp.5.weight",
                "transformer.pair_mlp.5.bias",
            )
        }
        auxiliary_state = load_torch_zip(auxiliary)["state"]
        self.aux_weight = np.asarray(auxiliary_state["aux_head.weight"], dtype=np.float32).reshape(5, 32)
        self.aux_bias = np.asarray(auxiliary_state["aux_head.bias"], dtype=np.float32)
        self.head = load_json_head(head)

    def _hidden(
        self, feat: np.ndarray, grid: np.ndarray, t: int, shape: tuple[int, int, int]
    ) -> np.ndarray:
        coords = np.column_stack((np.full(len(grid), t, dtype=np.int64), grid))
        value = np.concatenate((feat, positional_features(coords, (2,) + shape)), axis=1)
        value = _linear(
            value,
            self.core["transformer.proj.weight"],
            self.core["transformer.proj.bias"],
        )
        value = _layer_norm(
            value,
            self.core["transformer.norm_in.weight"],
            self.core["transformer.norm_in.bias"],
        )
        return _layer_norm(
            value,
            self.core["transformer.norm_out.weight"],
            self.core["transformer.norm_out.bias"],
        )

    def score_parent(
        self,
        cache: dict[str, Any],
        frame: int,
        parent_id: int,
    ) -> dict[str, Any]:
        source = cache["detections"].get(frame)
        target = cache["detections"].get(frame + 1)
        context = cache["contexts"].get(frame)
        if source is None or target is None or context is None:
            return {"probabilities": {}, "no_link": 1.0, "candidate_count": 0}
        candidates = parent_candidates(source, target, parent_id)
        if not len(candidates["target_ids"]):
            return {"probabilities": {}, "no_link": 1.0, "candidate_count": 0}
        source_rows = {int(node): index for index, node in enumerate(context["source_ids"])}
        target_rows = {int(node): index for index, node in enumerate(context["target_ids"])}
        source_order = [source_rows[int(node)] for node in source["node_ids"]]
        target_order = [target_rows[int(node)] for node in target["node_ids"]]
        source_hidden = self._hidden(
            context["source_feat"][source_order], source["grid_coords"], 0, context["grid_shape"]
        )
        target_hidden = self._hidden(
            context["target_feat"][target_order], target["grid_coords"], 1, context["grid_shape"]
        )
        si = candidates["source_index"]
        ti = candidates["target_index"]
        relative = (
            source["coords"][si].astype(np.float32)
            - target["coords"][ti].astype(np.float32)
        ) / np.float32(100.0)
        pair = np.concatenate((source_hidden[si], target_hidden[ti], relative), axis=1)
        checkpoint_logits = _linear(
            pair,
            self.core["transformer.pair_mlp.0.weight"],
            self.core["transformer.pair_mlp.0.bias"],
        )
        checkpoint_logits = _gelu(checkpoint_logits)
        checkpoint_logits = _linear(
            checkpoint_logits,
            self.core["transformer.pair_mlp.3.weight"],
            self.core["transformer.pair_mlp.3.bias"],
        )
        checkpoint_logits = _gelu(checkpoint_logits)
        checkpoint_logits = _linear(
            checkpoint_logits,
            self.core["transformer.pair_mlp.5.weight"],
            self.core["transformer.pair_mlp.5.bias"],
        ).reshape(-1)

        outputs = (
            context["target_feat"][target_order].astype(np.float32) @ self.aux_weight.T
            + self.aux_bias
        )
        predicted_backward = outputs[ti, 1:4].astype(np.float64)
        log_variance = np.clip(outputs[ti, 4].astype(np.float64), -10.0, 10.0)
        motion_residual = -candidates["displacement"] - predicted_backward
        motion_nll = 0.5 * (
            np.exp(-log_variance) * np.sum(motion_residual**2, axis=1)
            + 3.0 * log_variance
        )
        source_index = int(si[0])
        source_conf = np.full(len(si), float(source["scores"][source_index]))
        target_conf = target["scores"][ti].astype(np.float64)
        source_density = np.log1p(candidates["source_density"])
        target_density = np.log1p(candidates["target_density"])
        displacement = candidates["displacement"]
        residual_features = np.column_stack(
            (
                checkpoint_logits.astype(np.float64),
                displacement / 10.0,
                candidates["distance"] / 10.0,
                source_conf,
                target_conf,
                source_density,
                target_density,
                target_density - source_density,
                np.zeros(len(si), dtype=np.float64),
                displacement / 10.0,
                candidates["distance"] / 10.0,
                motion_nll,
            )
        ).astype(np.float32)
        residual = _linear(
            _gelu(
                _linear(
                    residual_features,
                    self.head["input.weight"],
                    self.head["input.bias"],
                )
            ),
            self.head["output.weight"],
            self.head["output.bias"],
        ).reshape(-1)
        logits = checkpoint_logits.astype(np.float32) + residual
        dummy = float(self.head["dummy_residual"])
        maximum = max(dummy, float(np.max(logits)))
        dummy_exp = math.exp(dummy - maximum)
        exponentials = np.exp(logits.astype(np.float64) - maximum)
        denominator = dummy_exp + float(exponentials.sum())
        probabilities = exponentials / denominator
        return {
            "probabilities": {
                int(target_id): float(probability)
                for target_id, probability in zip(
                    candidates["target_ids"], probabilities, strict=True
                )
            },
            "no_link": float(dummy_exp / denominator),
            "candidate_count": len(probabilities),
        }
