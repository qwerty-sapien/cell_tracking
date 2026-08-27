#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build, but never launch, a label-free V32 pre/post-ILP cache kernel.

The accepted public notebook and ``scripts/public_temporalunet_baseline.py``
remain untouched.  This builder first asks the existing immutable builder for
the ordinary V32 source, then inserts one separately labelled exporter patch.
The exporter is deliberately sparse: it records the fused candidate graph and
the same-run selected ILP graph, not dense logits or feature maps.

This file has no Kaggle client and no launch/submission command.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts/public_temporalunet_baseline.py"
POPULATION_DEFAULT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/"
    "public_temporalunet_baseline_v1/POPULATIONS.json"
)
EXPECTED_PUBLIC_NOTEBOOK_SHA256 = (
    "3f995a97a51c9517bdf8d8da2f606ffd184a51c841de87c962d911a336b7847d"
)
EXPECTED_BASE_PREDICTOR_SHA256 = (
    "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
)
EXPECTED_V32_RUNTIME_PREDICTOR_SHA256 = (
    "64734adad551aec6ff7fa805ea24288fef3faab369cbc729f6bbe5fa52529dc1"
)
EXPECTED_BASE_BUILDER_SHA256 = (
    "8f26c2b10a5f8eb5dddd222c832147577fcc7f469a2ebb633f12a5b98bea8271"
)
EXPECTED_POPULATION_FILE_SHA256 = (
    "6c8153d79bcda20e76d74de141928a054b0d123f3e80088a4fbe4c392ea9d5ae"
)
EXPECTED_POPULATION_SELF_SHA256 = (
    "4b18591377fea4faba6e9f7fdcb230acc0b5a43f79dea2b7d532c2b02dc75dc5"
)
EXPECTED_PRIMARY_MANIFEST_SHA256 = (
    "bc20f1f04cfb682af3b27a836ce9a57f44a2fe27508dc59039200b2a23188103"
)
EXPECTED_PRIMARY_WEIGHT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
)
EXPECTED_SECONDARY_MANIFEST_SHA256 = (
    "7f01c0b2f9491606a1b543339e072230e25317d9c70fb9a18aea59e002a995eb"
)
EXPECTED_SECONDARY_WEIGHT_SHA256 = (
    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
)
EXPECTED_CONFIG_SHA256 = (
    "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
)
CACHE_SCHEMA = "celltrack-alpha.v32-preilp-replay-cache.v1"
PARTIAL_SCHEMA = "celltrack-alpha.v32-preilp-export-partial.v1"
SOURCE_FREEZE_SCHEMA = "celltrack-alpha.v32-preilp-source-freeze.v1"
KERNEL_MARKER = "print('Calibrated dual-seed runtime patch applied')"
INSTRUMENTATION_MARKER = "# ---- V32 LABEL-FREE PRE/POST-ILP CACHE EXPORTER ----"
TERMINAL_INVENTORY_MARKER = "# ---- V32 CACHE TERMINAL INVENTORY ----"
RUSTWORKX_NODE_BUILD_ANCHOR = '''    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])'''
RUSTWORKX_NODE_BUILD_REPLACEMENT = '''    node_ids = graph.bulk_add_nodes([
        {
            "node_id": int(node_id),
            "t": int(t), "z": float(z), "y": float(y), "x": float(x),
        }
        for node_id, (t, z, y, x) in enumerate(coords)
    ])
    if node_ids != list(range(len(coords))):
        raise RuntimeError("V32 rustworkx node construction id drift")'''
CANARY_IDENTITY = "train:44b6_cf8fed6b"
VISUAL_CANARY_IDENTITIES = (
    "train:44b6_d754aa59",
    "train:44b6_18ced818",
)
F2_SIX_IDENTITIES = (
    "train:44b6_cf2536e8",
    "train:44b6_3bb3690f",
    "train:44b6_aaf8b0ea",
    "train:44b6_c8e2a523",
    "train:44b6_87bba6c4",
    "train:44b6_996155de",
)
FULL_SUPPORT_WHEELHOUSE = (
    ROOT
    / "dag_prompts/2026-08-16-five-kernel-validation-v1/v32/"
    "provider_primary/wheels"
)
EXPECTED_FULL_SUPPORT_WHEEL_COUNT = 62
EXPECTED_FULL_SUPPORT_INSTALL_PLAN_SHA256 = (
    "44d759547070cca311a89b3676e3a949045f6b242443f2263d39b8593c3553d1"
)


class CacheBuildError(ValueError):
    """The isolated cache package could not be frozen safely."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def full_support_install_plan() -> list[dict[str, Any]]:
    if not FULL_SUPPORT_WHEELHOUSE.is_dir() or FULL_SUPPORT_WHEELHOUSE.is_symlink():
        raise CacheBuildError("exact full-support wheelhouse is absent or symbolic")
    rows = []
    for path in sorted(FULL_SUPPORT_WHEELHOUSE.glob("*.whl")):
        if path.is_symlink() or not path.is_file():
            raise CacheBuildError(f"full-support wheel is not an exact file: {path}")
        rows.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if len(rows) != EXPECTED_FULL_SUPPORT_WHEEL_COUNT:
        raise CacheBuildError("full-support wheel count drift")
    if canonical_sha256(rows) != EXPECTED_FULL_SUPPORT_INSTALL_PLAN_SHA256:
        raise CacheBuildError("full-support install-plan digest drift")
    return rows


def runtime_split_sha256(identities: Sequence[str]) -> str:
    datasets = sorted(identity.split(":", 1)[1] for identity in identities)
    payload = json.dumps(
        [{"split": 0, "train": [], "test": datasets}], indent=2
    ).encode("utf-8")
    return sha256_bytes(payload)


def _load_base_builder():
    specification = importlib.util.spec_from_file_location(
        "_v32_immutable_baseline_builder", BASE_BUILDER
    )
    if specification is None or specification.loader is None:
        raise CacheBuildError(f"cannot import {BASE_BUILDER}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise CacheBuildError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def predictor_export_helpers_source() -> str:
    """Return self-contained code inserted into the reconstructed predictor."""

    return textwrap.dedent(
        r'''
        # ---- V32 LABEL-FREE PRE/POST-ILP CACHE EXPORTER ----
        import fcntl as _v32_fcntl
        import hashlib as _v32_hashlib
        import tempfile as _v32_tempfile

        _V32_CACHE_ROOT = Path(os.environ.get(
            "BIOHUB_V32_CACHE_ROOT", "/kaggle/working/v32_preilp_cache_v1"
        ))
        _V32_CACHE_ENABLED = os.environ.get("BIOHUB_V32_CACHE_EXPORT", "0") == "1"
        _V32_OPEN_AUDIT_PATH = _V32_CACHE_ROOT / "OPEN_AUDIT.jsonl"
        _V32_OPEN_AUDIT_READY = False
        _V32_FORBIDDEN_OPEN_MARKERS = (
            ".geff", "ground_truth", "ground-truth", "/labels/", "/gt/",
            "/scorer/", "b0_outcomes", "/m3/", "m3_", "-m3-",
            "association" + "_ranker", "local_association" + "_ranker",
        )
        _V32_VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
        _V32_NODE_DTYPE = np.dtype([
            ("ordinal", "<i8"), ("node_id", "<i8"), ("t", "<i4"),
            ("z_vox", "<f8"), ("y_vox", "<f8"), ("x_vox", "<f8"),
            ("z_um", "<f8"), ("y_um", "<f8"), ("x_um", "<f8"),
        ])
        _V32_PRE_EDGE_DTYPE = np.dtype([
            ("ordinal", "<i8"), ("source_id", "<i8"), ("target_id", "<i8"),
            ("edge_prob", "<f8"), ("edge_dist_grid", "<f8"),
        ])
        _V32_POST_EDGE_DTYPE = np.dtype([
            ("ordinal", "<i8"), ("source_id", "<i8"), ("target_id", "<i8"),
            ("edge_prob", "<f8"),
        ])
        _V32_FRAME_PAIR_DTYPE = np.dtype([
            ("source_t", "<i4"), ("target_t", "<i4"),
            ("source_count", "<i8"), ("target_count", "<i8"),
            ("candidate_start", "<i8"), ("candidate_count", "<i8"),
            ("skipped_empty", "u1"),
        ])


        def _v32_sha256(path):
            digest = _v32_hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return digest.hexdigest()


        def _v32_atomic_npy(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise RuntimeError(f"V32 cache refuses to overwrite {path}")
            descriptor, temporary = _v32_tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.save(handle, value, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            except Exception:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise
            return {"path": path.name, "bytes": path.stat().st_size,
                    "sha256": _v32_sha256(path), "rows": int(len(value)),
                    "shape": list(value.shape), "dtype": value.dtype.descr}


        def _v32_atomic_json(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise RuntimeError(f"V32 cache refuses to overwrite {path}")
            payload = (json.dumps(value, allow_nan=False, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("ascii")
            descriptor, temporary = _v32_tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            except Exception:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise
            return _v32_hashlib.sha256(payload).hexdigest()


        def _v32_record_input_opens(paths):
            global _V32_OPEN_AUDIT_READY
            _V32_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            with _V32_OPEN_AUDIT_PATH.open("ab") as handle:
                _v32_fcntl.flock(handle.fileno(), _v32_fcntl.LOCK_EX)
                try:
                    for raw_path in paths:
                        path = Path(raw_path)
                        text = str(path)
                        normalized = "/" + text.replace("\\", "/").lower().strip("/")
                        if any(marker in normalized for marker in _V32_FORBIDDEN_OPEN_MARKERS):
                            raise RuntimeError(f"protected V32 input-open path: {text}")
                        event = {
                            "event": "input_open",
                            "path": text,
                            "schema": "celltrack-alpha.v32-label-free-open-audit.v1",
                            "shard": os.environ.get("BIOHUB_GPU_SHARD", "single"),
                        }
                        if path.is_file() and not path.is_symlink():
                            event.update({
                                "kind": "file",
                                "bytes": path.stat().st_size,
                                "sha256": _v32_sha256(path),
                            })
                        elif path.is_dir():
                            metadata = []
                            for name in (".zarray", ".zattrs", "zarr.json"):
                                candidate = path / name
                                if candidate.is_file() and not candidate.is_symlink():
                                    metadata.append({
                                        "name": name,
                                        "bytes": candidate.stat().st_size,
                                        "sha256": _v32_sha256(candidate),
                                    })
                            if not metadata:
                                raise RuntimeError(f"V32 Zarr array metadata is absent: {path}")
                            metadata_payload = (json.dumps(
                                metadata, allow_nan=False, sort_keys=True,
                                separators=(",", ":"),
                            ) + "\n").encode("ascii")
                            event.update({
                                "kind": "zarr_array",
                                "metadata": metadata,
                                "metadata_sha256": _v32_hashlib.sha256(
                                    metadata_payload
                                ).hexdigest(),
                            })
                        else:
                            raise RuntimeError(
                                f"V32 audited input is absent, a symlink, or non-regular: {path}"
                            )
                        payload = (json.dumps(
                            event, allow_nan=False, sort_keys=True,
                            separators=(",", ":"),
                        ) + "\n").encode("ascii")
                        handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    _v32_fcntl.flock(handle.fileno(), _v32_fcntl.LOCK_UN)
            _V32_OPEN_AUDIT_READY = True


        def _v32_source_zarr_path(ds_path):
            base = Path(ds_path)
            if base.suffix in {".zarr", ".geff"}:
                base = base.parent / base.stem
            return base.parent / f"{base.stem}.zarr"


        def _v32_rustworkx_nodes(graph):
            rx_graph = graph.rx_graph
            expected_count = int(graph.num_nodes())
            endpoint_map = {}
            seen_node_ids = set()
            rows = []
            for local_id in range(expected_count):
                try:
                    raw = rx_graph[local_id]
                except (IndexError, KeyError) as exc:
                    raise RuntimeError(
                        "V32 rustworkx local node id drift"
                    ) from exc
                if not isinstance(raw, dict):
                    raise RuntimeError("V32 rustworkx node payload is not a dictionary")
                missing = {"node_id", "t", "z", "y", "x"} - set(raw)
                if missing:
                    raise RuntimeError(
                        f"V32 rustworkx node payload is missing keys: {sorted(missing)}"
                    )
                raw_node_id = raw["node_id"]
                if isinstance(raw_node_id, bool) or not isinstance(
                    raw_node_id, (int, np.integer)
                ):
                    raise RuntimeError("V32 rustworkx node_id is not an integer")
                node_id = int(raw_node_id)
                if node_id in seen_node_ids:
                    raise RuntimeError("V32 exported node ids are not unique")
                endpoint_map[local_id] = node_id
                seen_node_ids.add(node_id)
                rows.append((node_id, raw))
            if len(rows) != expected_count:
                raise RuntimeError("V32 rustworkx node snapshot count drift")
            return endpoint_map, rows


        def _v32_nodes(graph):
            _, rows = _v32_rustworkx_nodes(graph)
            value = np.empty(len(rows), dtype=_V32_NODE_DTYPE)
            for ordinal, (node_id, row) in enumerate(rows):
                z, y, x = float(row["z"]), float(row["y"]), float(row["x"])
                value[ordinal] = (
                    ordinal, node_id, int(row["t"]), z, y, x,
                    z * _V32_VOXEL_SCALE_UM[0], y * _V32_VOXEL_SCALE_UM[1],
                    x * _V32_VOXEL_SCALE_UM[2],
                )
            return value


        def _v32_edges(graph, stage):
            endpoint_map, _ = _v32_rustworkx_nodes(graph)
            rows = []
            for source_local, target_local, raw in graph.rx_graph.weighted_edge_list():
                source_local = int(source_local)
                target_local = int(target_local)
                if source_local not in endpoint_map or target_local not in endpoint_map:
                    raise RuntimeError("V32 rustworkx edge endpoint is absent from node map")
                if not isinstance(raw, dict):
                    raise RuntimeError("V32 rustworkx edge payload is not a dictionary")
                required = {"edge_id", "edge_prob"}
                if stage == "pre_ilp":
                    required.add("edge_dist")
                missing = required - set(raw)
                if missing:
                    raise RuntimeError(
                        f"V32 rustworkx edge payload is missing keys: {sorted(missing)}"
                    )
                source_id = endpoint_map[source_local]
                target_id = endpoint_map[target_local]
                if "source_id" in raw and int(raw["source_id"]) != source_id:
                    raise RuntimeError("V32 rustworkx edge source_id mapping drift")
                if "target_id" in raw and int(raw["target_id"]) != target_id:
                    raise RuntimeError("V32 rustworkx edge target_id mapping drift")
                rows.append((int(raw["edge_id"]), source_id, target_id, raw))
            expected_count = int(graph.num_edges())
            if len(rows) != expected_count:
                raise RuntimeError("V32 rustworkx edge count drift")
            edge_ids = [row[0] for row in rows]
            if len(set(edge_ids)) != len(edge_ids):
                raise RuntimeError("V32 rustworkx edge ids are not unique")
            rows.sort(key=lambda row: row[0])
            dtype = _V32_PRE_EDGE_DTYPE if stage == "pre_ilp" else _V32_POST_EDGE_DTYPE
            value = np.empty(len(rows), dtype=dtype)
            for ordinal, (_, source_id, target_id, row) in enumerate(rows):
                if stage == "pre_ilp":
                    value[ordinal] = (
                        ordinal, source_id, target_id,
                        float(row["edge_prob"]), float(row["edge_dist"]),
                    )
                else:
                    value[ordinal] = (
                        ordinal, source_id, target_id,
                        float(row["edge_prob"]),
                    )
            return value


        def _v32_frame_pairs(nodes, edges, frame_count):
            value = np.empty(max(0, frame_count - 1), dtype=_V32_FRAME_PAIR_DTYPE)
            node_time = {int(row["node_id"]): int(row["t"]) for row in nodes}
            cursor = 0
            for t in range(frame_count - 1):
                source_count = int(np.count_nonzero(nodes["t"] == t))
                target_count = int(np.count_nonzero(nodes["t"] == t + 1))
                start = cursor
                while cursor < len(edges) and node_time[int(edges[cursor]["source_id"])] == t:
                    cursor += 1
                value[t] = (
                    t, t + 1, source_count, target_count, start, cursor - start,
                    int(source_count == 0 or target_count == 0),
                )
            if cursor != len(edges):
                raise RuntimeError("V32 pre-ILP edges are not frame-pair ordered")
            return value


        def _v32_export_graph(dataset, ds_path, weights_path, splits_file, stage, graph):
            if not _V32_CACHE_ENABLED:
                return
            if stage not in {"pre_ilp", "post_ilp"}:
                raise RuntimeError(f"unknown V32 cache stage {stage}")
            source_zarr = _v32_source_zarr_path(ds_path)
            if not source_zarr.is_dir() or not (source_zarr / "0").exists():
                raise RuntimeError(f"V32 source Zarr is absent or malformed: {source_zarr}")
            if stage == "pre_ilp":
                opened = [source_zarr / "0"]
                if not _V32_OPEN_AUDIT_READY:
                    opened.extend([weights_path, weights_path.parent / "config.json", splits_file])
                    secondary_text = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()
                    if secondary_text:
                        secondary_path = Path(secondary_text)
                        opened.extend([secondary_path, secondary_path.parent / "config.json"])
                _v32_record_input_opens(opened)
            directory = _V32_CACHE_ROOT / "datasets" / dataset
            directory.mkdir(parents=True, exist_ok=True)
            nodes = _v32_nodes(graph)
            edges = _v32_edges(graph, stage)
            if stage == "pre_ilp":
                node_name, edge_name = "PRE_ILP_NODES.npy", "PRE_ILP_EDGES.npy"
            else:
                node_name, edge_name = "POST_ILP_NODES.npy", "POST_ILP_EDGES.npy"
            files = {
                "nodes": _v32_atomic_npy(directory / node_name, nodes),
                "edges": _v32_atomic_npy(directory / edge_name, edges),
            }
            frame_count = int(zarr.open_group(str(source_zarr), mode="r")["0"].shape[0])
            if stage == "pre_ilp":
                frame_pairs = _v32_frame_pairs(nodes, edges, frame_count)
                files["frame_pairs"] = _v32_atomic_npy(
                    directory / "FRAME_PAIRS.npy", frame_pairs
                )
            receipt = {
                "schema": "celltrack-alpha.v32-preilp-export-partial.v1",
                "status": "PASS_UNSEALED",
                "dataset": dataset,
                "identity": f"train:{dataset}",
                "stage": stage,
                "frame_count": frame_count,
                "label_free": True,
                "evaluate": False,
                "m3_mounted": False,
                "source_zarr": str(source_zarr),
                "files": files,
            }
            _v32_atomic_json(directory / f"{stage.upper()}_RECEIPT.json", receipt)
        '''
    ).strip() + "\n\n"


def instrument_predictor_source(source: str) -> str:
    """Insert sparse pre/post-ILP exporters into one reconstructed predictor."""

    if INSTRUMENTATION_MARKER in source:
        raise CacheBuildError("predictor is already cache-instrumented")
    source = _replace_once(
        source,
        RUSTWORKX_NODE_BUILD_ANCHOR,
        RUSTWORKX_NODE_BUILD_REPLACEMENT,
        "bind direct rustworkx node ids",
    )
    helper_anchor = "# =============================================================================\n# Prediction loop\n"
    source = _replace_once(
        source,
        helper_anchor,
        predictor_export_helpers_source() + helper_anchor,
        "insert cache exporter helpers",
    )
    pre_anchor = "        graph = build_graph(coords, edges)\n        if cfg.use_ilp"
    pre_replacement = (
        "        graph = build_graph(coords, edges)\n"
        "        _v32_export_graph(name, ds_path, weights_path, splits_file, \"pre_ilp\", graph)\n"
        "        _v32_skip_ilp = (\n"
        "            os.environ.get(\"BIOHUB_V32_CPU_PREFLIGHT_SKIP_ILP\", \"\") == \"1\"\n"
        "            and bool(os.environ.get(\"KAGGLE_PREFLIGHT_MODE\", \"\").strip())\n"
        "            and not torch.cuda.is_available()\n"
        "        )\n"
        "        if cfg.use_ilp and not _v32_skip_ilp"
    )
    source = _replace_once(source, pre_anchor, pre_replacement, "insert pre-ILP export")
    post_anchor = "        save_graph(graph, output_dir / f\"{name}.geff\")"
    post_replacement = (
        "        _v32_export_graph(name, ds_path, weights_path, splits_file, \"post_ilp\", graph)\n"
        + post_anchor
    )
    source = _replace_once(source, post_anchor, post_replacement, "insert post-ILP export")
    compile(source, "v32_cache_predict_unet_transformer.py", "exec")
    return source


def kernel_runtime_patch_source() -> str:
    """Return kernel code that instruments the already reconstructed predictor."""

    helpers = predictor_export_helpers_source()
    post_anchor = '        save_graph(graph, output_dir / f"{name}.geff")'
    post_replacement = (
        '        _v32_export_graph(name, ds_path, weights_path, splits_file, '
        '"post_ilp", graph)\n'
        + post_anchor
    )
    # Reuse the exact same insertion logic as instrument_predictor_source, but
    # perform it inside the Kaggle kernel after all V32 dual-seed patches land.
    snippet = f'''
{INSTRUMENTATION_MARKER}
os.environ["BIOHUB_V32_CACHE_EXPORT"] = "1"
os.environ.setdefault("BIOHUB_V32_CACHE_ROOT", "/kaggle/working/v32_preilp_cache_v1")
_v32_cache_source = _ps.read_text()
_v32_cache_uninstrumented_sha256 = hashlib.sha256(
    _v32_cache_source.encode("utf-8")
).hexdigest()
if _v32_cache_uninstrumented_sha256 != {EXPECTED_V32_RUNTIME_PREDICTOR_SHA256!r}:
    raise RuntimeError(
        "V32 uninstrumented runtime predictor hash drift: "
        f"{{_v32_cache_uninstrumented_sha256}}"
    )
_v32_cache_node_old = {RUSTWORKX_NODE_BUILD_ANCHOR!r}
_v32_cache_node_new = {RUSTWORKX_NODE_BUILD_REPLACEMENT!r}
if _v32_cache_source.count(_v32_cache_node_old) != 1:
    raise RuntimeError("V32 rustworkx node-build anchor cardinality drift")
_v32_cache_source = _v32_cache_source.replace(
    _v32_cache_node_old, _v32_cache_node_new, 1
)
_v32_cache_helper_anchor = {"# =============================================================================\n# Prediction loop\n"!r}
_v32_cache_helpers = {helpers!r}
if _v32_cache_source.count(_v32_cache_helper_anchor) != 1:
    raise RuntimeError("V32 cache helper anchor cardinality drift")
_v32_cache_source = _v32_cache_source.replace(
    _v32_cache_helper_anchor, _v32_cache_helpers + _v32_cache_helper_anchor, 1
)
_v32_cache_pre_old = {"        graph = build_graph(coords, edges)\n        if cfg.use_ilp"!r}
_v32_cache_pre_new = {"        graph = build_graph(coords, edges)\n        _v32_export_graph(name, ds_path, weights_path, splits_file, \"pre_ilp\", graph)\n        _v32_skip_ilp = (\n            os.environ.get(\"BIOHUB_V32_CPU_PREFLIGHT_SKIP_ILP\", \"\") == \"1\"\n            and bool(os.environ.get(\"KAGGLE_PREFLIGHT_MODE\", \"\").strip())\n            and not torch.cuda.is_available()\n        )\n        if cfg.use_ilp and not _v32_skip_ilp"!r}
if _v32_cache_source.count(_v32_cache_pre_old) != 1:
    raise RuntimeError("V32 cache pre-ILP anchor cardinality drift")
_v32_cache_source = _v32_cache_source.replace(
    _v32_cache_pre_old, _v32_cache_pre_new, 1
)
_v32_cache_post_old = {post_anchor!r}
_v32_cache_post_new = {post_replacement!r}
if _v32_cache_source.count(_v32_cache_post_old) != 1:
    raise RuntimeError("V32 cache post-ILP anchor cardinality drift")
_v32_cache_source = _v32_cache_source.replace(
    _v32_cache_post_old, _v32_cache_post_new, 1
)
compile(_v32_cache_source, str(_ps), "exec")
_ps.write_text(_v32_cache_source)
print("Label-free V32 pre/post-ILP cache exporter applied")
'''
    return textwrap.dedent(snippet)


def terminal_cache_inventory_source() -> str:
    """Bind every partial cache byte into the ordinary terminal run receipt."""

    return textwrap.dedent(
        r'''
        # ---- V32 CACHE TERMINAL INVENTORY ----
        _v32_terminal_cache_root = Path(os.environ.get(
            "BIOHUB_V32_CACHE_ROOT", "/kaggle/working/v32_preilp_cache_v1"
        ))
        if not _v32_terminal_cache_root.is_dir():
            raise RuntimeError("V32 cache root is absent at terminal publication")
        _v32_terminal_root_names = {
            path.name for path in _v32_terminal_cache_root.iterdir()
        }
        if _v32_terminal_root_names != {"datasets", "OPEN_AUDIT.jsonl"}:
            raise RuntimeError("V32 terminal cache root file-set drift")
        _v32_terminal_datasets_root = _v32_terminal_cache_root / "datasets"
        _v32_terminal_expected = sorted(str(value) for value in SELECTED_DATASETS)
        _v32_terminal_observed = sorted(
            path.name for path in _v32_terminal_datasets_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        if _v32_terminal_observed != _v32_terminal_expected:
            raise RuntimeError("V32 terminal cache population drift")
        _v32_terminal_audit = _v32_terminal_cache_root / "OPEN_AUDIT.jsonl"
        if not _v32_terminal_audit.is_file():
            raise RuntimeError("V32 terminal open audit is absent")
        _v32_terminal_required = {
            "PRE_ILP_NODES.npy", "PRE_ILP_EDGES.npy", "FRAME_PAIRS.npy",
            "PRE_ILP_RECEIPT.json", "POST_ILP_NODES.npy", "POST_ILP_EDGES.npy",
            "POST_ILP_RECEIPT.json",
        }
        _v32_terminal_files = [{
            "path": "OPEN_AUDIT.jsonl",
            "bytes": _v32_terminal_audit.stat().st_size,
            "sha256": _sha256(_v32_terminal_audit),
        }]
        _v32_terminal_frame_counts = {}
        for _v32_terminal_dataset in _v32_terminal_expected:
            _v32_terminal_directory = _v32_terminal_datasets_root / _v32_terminal_dataset
            _v32_terminal_names = {
                path.name for path in _v32_terminal_directory.iterdir() if path.is_file()
            }
            if _v32_terminal_names != _v32_terminal_required:
                raise RuntimeError(
                    f"V32 terminal cache file-set drift for {_v32_terminal_dataset}"
                )
            for _v32_terminal_path in sorted(_v32_terminal_directory.iterdir()):
                if not _v32_terminal_path.is_file():
                    raise RuntimeError(
                        f"V32 terminal cache contains a non-file: {_v32_terminal_path}"
                    )
                _v32_terminal_files.append({
                    "path": str(_v32_terminal_path.relative_to(_v32_terminal_cache_root)),
                    "bytes": _v32_terminal_path.stat().st_size,
                    "sha256": _sha256(_v32_terminal_path),
                })
            _v32_terminal_pre_receipt = json.loads(
                (_v32_terminal_directory / "PRE_ILP_RECEIPT.json").read_text(
                    encoding="utf-8"
                )
            )
            _v32_terminal_post_receipt = json.loads(
                (_v32_terminal_directory / "POST_ILP_RECEIPT.json").read_text(
                    encoding="utf-8"
                )
            )
            _v32_terminal_pre_frames = int(_v32_terminal_pre_receipt["frame_count"])
            _v32_terminal_post_frames = int(_v32_terminal_post_receipt["frame_count"])
            if _v32_terminal_pre_frames != _v32_terminal_post_frames:
                raise RuntimeError(
                    f"V32 terminal pre/post frame-count drift for {_v32_terminal_dataset}"
                )
            _v32_terminal_frame_counts[_v32_terminal_dataset] = _v32_terminal_pre_frames
        _v32_terminal_payload = (
            json.dumps(_v32_terminal_files, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n"
        ).encode("ascii")
        _receipt["v32_preilp_cache"] = {
            "schema": "celltrack-alpha.v32-preilp-terminal-inventory.v1",
            "status": "PASS_UNSEALED",
            "path": str(_v32_terminal_cache_root),
            "identities": [f"train:{value}" for value in SELECTED_DATASETS],
            "files": _v32_terminal_files,
            "inventory_sha256": hashlib.sha256(_v32_terminal_payload).hexdigest(),
            "scientific_use_open": False,
        }
        if SHIP_PREFLIGHT_MODE:
            if any(
                count != SHIP_PREFLIGHT_UNITS
                for count in _v32_terminal_frame_counts.values()
            ):
                raise RuntimeError(
                    "V32 preflight completed frame counts differ from the requested bound"
                )
            _v32_preflight_completion = {
                "schema": "celltrack-alpha.v32-ship-preflight-completion.v1",
                "status": "PASS",
                "completed_units": sum(_v32_terminal_frame_counts.values()),
                "mode": SHIP_PREFLIGHT_MODE,
                "selected_datasets": list(SELECTED_DATASETS),
                "frames_per_identity": SHIP_PREFLIGHT_UNITS,
                "observed_frames_by_identity": _v32_terminal_frame_counts,
                "execution_device": (
                    "cuda" if _torch.cuda.is_available() else "cpu"
                ),
                "cache_inventory_sha256": _receipt["v32_preilp_cache"]["inventory_sha256"],
            }
            _atomic_json(
                WORKING_DIR / "V32_SHIP_PREFLIGHT_COMPLETION.json",
                _v32_preflight_completion,
            )
            print("V32_SHIP_PREFLIGHT_COMPLETE")
        '''
    ).strip() + "\n"


def preflight_control_source() -> str:
    """Return an outcome-blind, exact workload selector used only by ship tests."""

    return textwrap.dedent(
        r'''
        SHIP_PREFLIGHT_MODE = os.environ.get("KAGGLE_PREFLIGHT_MODE", "").strip()
        SHIP_PREFLIGHT_UNITS_TEXT = os.environ.get("KAGGLE_PREFLIGHT_UNITS", "").strip()
        SHIP_PREFLIGHT_UNITS = 0
        _V32_SHIP_MODES = {
            "tiny_sparse": ["44b6_d754aa59"],
            "stratified_both": ["44b6_d754aa59", "44b6_18ced818"],
            "stress_crowded": ["44b6_18ced818"],
            "expansive_both": ["44b6_d754aa59", "44b6_18ced818"],
        }
        if SHIP_PREFLIGHT_MODE:
            if SHIP_PREFLIGHT_MODE not in _V32_SHIP_MODES:
                raise RuntimeError("unknown KAGGLE_PREFLIGHT_MODE")
            try:
                SHIP_PREFLIGHT_UNITS = int(SHIP_PREFLIGHT_UNITS_TEXT)
            except ValueError as exc:
                raise RuntimeError("KAGGLE_PREFLIGHT_UNITS must be an integer") from exc
            if not 3 <= SHIP_PREFLIGHT_UNITS <= 100:
                raise RuntimeError("KAGGLE_PREFLIGHT_UNITS must be in [3, 100]")
            SELECTED_DATASETS = list(_V32_SHIP_MODES[SHIP_PREFLIGHT_MODE])
            REQUEST_COUNT = len(SELECTED_DATASETS)
            REQUEST_FILE_COUNT = REQUEST_COUNT
        elif SHIP_PREFLIGHT_UNITS_TEXT:
            raise RuntimeError("KAGGLE_PREFLIGHT_UNITS requires KAGGLE_PREFLIGHT_MODE")
        '''
    ).strip() + "\n"


def deferred_step0_source() -> str:
    """Defer raw-data access until after the full release import probe."""

    return textwrap.dedent(
        r'''
        STEP0 = None

        SELECTED_TRAIN = Path("/kaggle/working/public_temporalunet_selected_train")


        def _v32_materialize_selected_train():
            if SELECTED_TRAIN.exists():
                shutil.rmtree(SELECTED_TRAIN)
            SELECTED_TRAIN.mkdir(parents=True, exist_ok=False)
            for dataset in SELECTED_DATASETS:
                source = COMPETITION_TRAIN / f"{dataset}.zarr"
                destination = SELECTED_TRAIN / f"{dataset}.zarr"
                if SHIP_PREFLIGHT_MODE:
                    # The predictor's --slice option slices the video list, not
                    # a video's time axis.  Build a real Zarr-v3 prefix view so
                    # preflight workload units are genuinely frames.
                    (destination / "0" / "c").mkdir(parents=True)
                    shutil.copy2(source / "zarr.json", destination / "zarr.json")
                    array_metadata = json.loads(
                        (source / "0" / "zarr.json").read_text(encoding="utf-8")
                    )
                    source_frames = int(array_metadata["shape"][0])
                    if SHIP_PREFLIGHT_UNITS > source_frames:
                        raise RuntimeError(
                            f"preflight frame request exceeds {dataset}: "
                            f"{SHIP_PREFLIGHT_UNITS} > {source_frames}"
                        )
                    array_metadata["shape"][0] = SHIP_PREFLIGHT_UNITS
                    _atomic_json(destination / "0" / "zarr.json", array_metadata)
                    source_chunks = source / "0" / "c"
                    destination_chunks = destination / "0" / "c"
                    for frame in range(SHIP_PREFLIGHT_UNITS):
                        frame_source = source_chunks / str(frame)
                        if frame_source.exists():
                            shutil.copytree(
                                frame_source, destination_chunks / str(frame)
                            )
                else:
                    destination.symlink_to(source, target_is_directory=True)
            os.environ["BIOHUB_TEST_DIR"] = str(SELECTED_TRAIN)
            preprocess_counts(
                (
                    "selected_train_frame_prefix_copy"
                    if SHIP_PREFLIGHT_MODE
                    else "selected_train_symlink_view"
                ),
                len(SELECTED_DATASETS),
                0,
                created_symlinks=(0 if SHIP_PREFLIGHT_MODE else len(SELECTED_DATASETS)),
                copied_frame_prefixes=(
                    SHIP_PREFLIGHT_UNITS * len(SELECTED_DATASETS)
                    if SHIP_PREFLIGHT_MODE
                    else 0
                ),
                raw_inputs_modified=0,
            )
        '''
    ).strip() + "\n"


def dependency_probe_and_step0_source() -> str:
    """Probe the release import surface before raw-data access or model load."""

    return textwrap.dedent(
        r'''
        def _v32_exact_offline_package_dirs(artifacts):
            candidates = [artifacts / "wheels", artifacts]
            result = []
            for candidate in candidates:
                candidate = candidate.expanduser()
                if candidate not in result and _has_package_file(candidate):
                    result.append(candidate)
            if not result or result[0] != artifacts / "wheels":
                raise RuntimeError("exact primary provider wheelhouse is unavailable")
            return result


        find_offline_package_dirs = _v32_exact_offline_package_dirs
        ensure_dependencies(ARTIFACTS)


        def _v32_release_dependency_probe():
            modules = (
                "numpy", "pandas", "scipy", "torch",
                "tracksdata", "zarr", "pyscipopt", "geff", "geff_spec", "ilpy",
                "polars", "blosc2", "dask", "imagecodecs", "skimage", "pyarrow",
                "rustworkx", "sqlalchemy", "numcodecs", "donfig", "google_crc32c",
                "bidict", "psygnal", "rich", "networkx", "pydantic", "imageio",
                "PIL", "tifffile", "yaml", "tqdm",
            )
            failures = {}
            for name in modules:
                try:
                    importlib.import_module(name)
                except Exception as exc:
                    failures[name] = f"{type(exc).__name__}: {exc}"
            if failures:
                raise ImportError("V32 release import probe failed: " + json.dumps(failures, sort_keys=True))
            print("V32_RELEASE_IMPORT_PROBE_COMPLETE")


        _v32_release_dependency_probe()
        STEP0 = run_step0()
        _v32_materialize_selected_train()
        TEST_DIR = SELECTED_TRAIN
        print("V32_HEAVY_WORK_BEGIN")
        materialize_inference_repo(ARTIFACTS)
        '''
    ).strip() + "\n"


def dependency_restart_source() -> str:
    """Return the one-shot authenticated clean-interpreter restart seam."""

    return textwrap.dedent(
        r'''
        import hmac as _v32_hmac

        _V32_DEPENDENCY_REEXEC_ENV = "BIOHUB_V32_DEPENDENCY_REEXEC_V1"
        _V32_DEPENDENCY_REEXEC_CONTEXT = (
            b"celltrack-alpha.v32-dependency-reexec.v1\0"
            b"numpy==2.4.6\0numba==0.65.1\0llvmlite==0.47.0"
        )


        def _v32_dependency_reexec_token() -> str:
            source_path = Path(__file__).resolve()
            if not source_path.is_file() or source_path.is_symlink():
                raise RuntimeError("V32 dependency re-exec source is not an exact regular file")
            return hashlib.sha256(
                _V32_DEPENDENCY_REEXEC_CONTEXT + b"\0" + source_path.read_bytes()
            ).hexdigest()


        _v32_dependency_reexec_marker = os.environ.get(
            _V32_DEPENDENCY_REEXEC_ENV, ""
        ).strip()
        _V32_DEPENDENCY_REEXEC_AUTHENTICATED = False
        if _v32_dependency_reexec_marker:
            if not _v32_hmac.compare_digest(
                _v32_dependency_reexec_marker, _v32_dependency_reexec_token()
            ):
                raise RuntimeError("V32 dependency re-exec authentication failed")
            _V32_DEPENDENCY_REEXEC_AUTHENTICATED = True
            print("V32_DEPENDENCY_REEXEC_AUTHENTICATED", flush=True)


        def _v32_restart_after_dependency_install() -> None:
            if _V32_DEPENDENCY_REEXEC_AUTHENTICATED:
                raise ImportError(
                    "dependency recovery requested a second interpreter restart"
                )
            source_path = Path(__file__).resolve()
            environment = os.environ.copy()
            environment[_V32_DEPENDENCY_REEXEC_ENV] = (
                _v32_dependency_reexec_token()
            )
            argv = [sys.executable, str(source_path), *sys.argv[1:]]
            print("V32_DEPENDENCY_REEXEC_REQUESTED", flush=True)
            os.execve(sys.executable, argv, environment)
            raise RuntimeError("V32 dependency re-exec unexpectedly returned")
        '''
    ).strip() + "\n"


def standalone_display_source() -> str:
    """Return a deterministic sink for inherited notebook preview calls."""

    return textwrap.dedent(
        r'''
        def display(_value) -> None:
            print("V32_STANDALONE_DISPLAY_SINK", flush=True)
        '''
    ).strip() + "\n"


def full_support_install_source() -> str:
    """Return the exact all-wheel install transaction used on any drift."""

    plan = full_support_install_plan()
    return textwrap.dedent(
        f'''
        _V32_FULL_SUPPORT_INSTALL_PLAN = {plan!r}
        _V32_FULL_SUPPORT_INSTALL_PLAN_SHA256 = (
            {EXPECTED_FULL_SUPPORT_INSTALL_PLAN_SHA256!r}
        )


        def _v32_exact_full_support_wheels(artifacts: Path) -> list[Path]:
            wheelhouse = artifacts / "wheels"
            if not wheelhouse.is_dir() or wheelhouse.is_symlink():
                raise ImportError("exact full-support wheelhouse is absent or symbolic")
            observed = []
            for path in sorted(wheelhouse.glob("*.whl")):
                if path.is_symlink() or not path.is_file():
                    raise ImportError(f"full-support wheel is not exact: {{path}}")
                observed.append({{
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }})
            payload = (json.dumps(
                observed, allow_nan=False, separators=(",", ":"), sort_keys=True
            ) + "\\n").encode("ascii")
            digest = hashlib.sha256(payload).hexdigest()
            if (
                observed != _V32_FULL_SUPPORT_INSTALL_PLAN
                or digest != _V32_FULL_SUPPORT_INSTALL_PLAN_SHA256
            ):
                raise ImportError("exact full-support install-plan drift")
            return [wheelhouse / row["filename"] for row in observed]


        def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
            if _V32_DEPENDENCY_REEXEC_AUTHENTICATED:
                raise ImportError(
                    "dependency drift remained after the exact full-support restart: "
                    + ", ".join(missing)
                )
            wheels = _v32_exact_full_support_wheels(artifacts)
            command = [
                sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", *[str(path) for path in wheels],
            ]
            print(
                "Installing exact full-support plan for dependency drift:", missing,
                flush=True,
            )
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode != 0:
                raise ImportError(
                    "exact full-support offline install failed\\n"
                    + (result.stdout or "")[-2000:]
                    + "\\n"
                    + (result.stderr or "")[-2000:]
                )
            print(
                "V32_FULL_SUPPORT_INSTALL_PLAN_BOUND="
                + _V32_FULL_SUPPORT_INSTALL_PLAN_SHA256,
                flush=True,
            )
            _v32_restart_after_dependency_install()
        '''
    ).strip() + "\n"


def child_diagnostic_probe_source() -> str:
    """Exercise durable bounded child diagnostics without touching campaign data."""

    return textwrap.dedent(
        r'''
        if os.environ.get("V32_RUNTIME_SAFETY_PROBE", "").strip() == "child_diagnostics":
            _v32_diag_root = Path("/kaggle/working/runtime_safety")
            _v32_diag_root.mkdir(parents=True, exist_ok=True)
            _v32_diag_cap = 128
            _v32_diag_child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.stdout.write('V32_DIAGNOSTIC_STDOUT\\n'); "
                        "sys.stderr.write('V32_DIAGNOSTIC_STDERR\\n'); "
                        "raise SystemExit(17)"
                    ),
                ],
                check=False,
                capture_output=True,
            )
            if _v32_diag_child.returncode != 17:
                raise RuntimeError("V32 diagnostic child return-code drift")
            _v32_diag_records = {}
            for _v32_diag_name, _v32_diag_payload in (
                ("stdout", _v32_diag_child.stdout),
                ("stderr", _v32_diag_child.stderr),
            ):
                _v32_diag_captured = _v32_diag_payload[-_v32_diag_cap:]
                _v32_diag_path = _v32_diag_root / f"child.{_v32_diag_name}.tail.log"
                _v32_diag_temp = _v32_diag_path.with_suffix(_v32_diag_path.suffix + ".tmp")
                _v32_diag_temp.write_bytes(_v32_diag_captured)
                _v32_diag_temp.replace(_v32_diag_path)
                _v32_diag_records[_v32_diag_name] = {
                    "path": str(_v32_diag_path.relative_to(Path("/kaggle/working"))),
                    "total_bytes": len(_v32_diag_payload),
                    "full_sha256": hashlib.sha256(_v32_diag_payload).hexdigest(),
                    "captured_bytes": len(_v32_diag_captured),
                    "captured_sha256": hashlib.sha256(_v32_diag_captured).hexdigest(),
                    "truncated": len(_v32_diag_payload) > len(_v32_diag_captured),
                }
            _v32_diag_receipt = {
                "schema": "celltrack-alpha.v32-child-diagnostic-probe.v1",
                "status": "PASS",
                "completed_units": 1,
                "child_returncode": _v32_diag_child.returncode,
                "persisted_before_cleanup": True,
                **_v32_diag_records,
            }
            _v32_diag_receipt_path = _v32_diag_root / "child_diagnostic_receipt.json"
            _v32_diag_receipt_temp = _v32_diag_receipt_path.with_suffix(".json.tmp")
            _v32_diag_receipt_temp.write_text(
                json.dumps(_v32_diag_receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _v32_diag_receipt_temp.replace(_v32_diag_receipt_path)
            print("V32_CHILD_DIAGNOSTIC_PROBE_PASS")
            raise SystemExit(0)
        '''
    ).strip() + "\n"


def exact_inventory_source() -> str:
    """Replace broad tree discovery in the inherited audit logger."""

    return textwrap.dedent(
        r'''
        def inventory_input(label, root):
            root = Path(root)
            if not root.exists():
                _run_log("input_inventory", label=label, root=str(root), status="missing", files=[])
                return []
            files = []
            pending = [root]
            while pending:
                directory = pending.pop()
                for path in sorted(directory.iterdir(), key=lambda value: value.name, reverse=True):
                    if path.is_symlink():
                        raise RuntimeError(f"input inventory refuses symlink: {path}")
                    if path.is_dir():
                        pending.append(path)
                    elif path.is_file():
                        entry = {"path": str(path), "bytes": path.stat().st_size}
                        if HASH_INPUT_CONTENTS:
                            entry["sha256"] = _run_sha256(path)
                        files.append(entry)
            files.sort(key=lambda value: value["path"])
            _run_log("input_inventory", label=label, root=str(root), status="present",
                     content_hashing=HASH_INPUT_CONTENTS, files=files)
            return files
        '''
    ).strip() + "\n"


def ship_step0_source() -> str:
    """Verify exact required mount bytes before the first heavy release import."""

    return textwrap.dedent(
        r'''
        def _v32_ship_step0():
            exact = {
                PRIMARY_ROOT / "ARTIFACT_MANIFEST.json": PRIMARY_MANIFEST_SHA256,
                PRIMARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": PRIMARY_WEIGHT_SHA256,
                SECONDARY_ROOT / "ARTIFACT_MANIFEST.json": SECONDARY_MANIFEST_SHA256,
                SECONDARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": SECONDARY_WEIGHT_SHA256,
            }
            raw_metadata = {
                "44b6_d754aa59": (
                    "9578e220275fb092c7f81f73651d16a15b478ce5fbd91fe50eaf25d18a38417b",
                    "73e3ff105fe0c50983cff97a50bfa95f8c0da16e1cb5adb56968d1cb7e1c41b0",
                ),
                "44b6_18ced818": (
                    "d687f710e8a3ab583ce1cf5ed5ba4553b4c7d15e02ce350e65eaa5c75262df43",
                    "73e3ff105fe0c50983cff97a50bfa95f8c0da16e1cb5adb56968d1cb7e1c41b0",
                ),
            }
            for path, expected in exact.items():
                if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
                    raise RuntimeError(f"V32 ship Step 0 exact-byte failure: {path}")
            for dataset in SELECTED_DATASETS:
                group = COMPETITION_TRAIN / f"{dataset}.zarr"
                if not group.is_dir() or group.is_symlink():
                    raise RuntimeError(f"V32 ship Step 0 raw group failure: {group}")
                expected_metadata = raw_metadata.get(dataset)
                if SHIP_PREFLIGHT_MODE and expected_metadata is None:
                    raise RuntimeError(
                        f"V32 ship preflight has no frozen raw metadata for {dataset}"
                    )
                for index, relative in enumerate(("zarr.json", "0/zarr.json")):
                    path = group / relative
                    if not path.is_file() or path.is_symlink():
                        raise RuntimeError(f"V32 ship Step 0 raw metadata failure: {path}")
                    if (
                        expected_metadata is not None
                        and _sha256(path) != expected_metadata[index]
                    ):
                        raise RuntimeError(
                            f"V32 ship Step 0 raw metadata hash failure: {path}"
                        )
            print("V32_SHIP_STEP0_COMPLETE")
            return {"status": "PASS", "selected_datasets": list(SELECTED_DATASETS)}


        SHIP_STEP0 = _v32_ship_step0()
        '''
    ).strip() + "\n"


def instrument_kernel_source(source: str) -> str:
    """Instrument an ordinary generated V32 kernel under a fresh identity."""

    if INSTRUMENTATION_MARKER in source:
        raise CacheBuildError("kernel is already cache-instrumented")
    forbidden = (
        "local_association_ranker",
        "ASSOCIATION_RANKER",
        "evaluate=True",
        "ground_truth",
        "B0_OUTCOMES",
        '"/gt/',
        '"/m3/',
    )
    hits = [token for token in forbidden if token in source]
    if hits:
        raise CacheBuildError(f"base cache kernel contains a protected/ranker token: {hits}")
    if source.count(KERNEL_MARKER) != 1:
        raise CacheBuildError("generated kernel does not have the exact V32 patch marker")
    result = _replace_once(
        source,
        "PUBLIC_SCRIPT_VERSION_ID = 337501010\n",
        "PUBLIC_SCRIPT_VERSION_ID = 337501010\n" + preflight_control_source(),
        "insert ship preflight workload control",
    )
    result = _replace_once(
        result,
        'TEST_DIR = Path(_test_dir_override) if _test_dir_override else COMP_DIR / "test"\n',
        'TEST_DIR = (\n'
        '    Path(_test_dir_override)\n'
        '    if _test_dir_override\n'
        '    else Path("/kaggle/working/public_temporalunet_selected_train")\n'
        ')\n',
        "remove the pre-rebind competition-test namespace visit",
    )
    result = _replace_once(
        result,
        'PACKAGE_SPECS = {\n',
        'PACKAGE_SPECS = {\n'
        '    "numpy": ("numpy", "numpy==2.4.6"),\n',
        "pin the compatible NumPy ABI before downstream imports",
    )
    result = _replace_once(
        result,
        '    "numpy": ("numpy", "numpy==2.4.6"),\n',
        '    "numpy": ("numpy", "numpy==2.4.6"),\n'
        '    "packaging": ("packaging", "packaging"),\n',
        "close the offline runtime metadata dependency set",
    )
    result = _replace_once(
        result,
        '    "packaging": ("packaging", "packaging"),\n',
        '    "packaging": ("packaging", "packaging"),\n'
        '    "scipy": ("scipy", "scipy"),\n',
        "close the offline tracksdata numeric dependency set",
    )
    result = _replace_once(
        result,
        '    "tracksdata": ("tracksdata", "tracksdata"),\n',
        '    "tracksdata": ("tracksdata", "tracksdata"),\n'
        '    "numba": ("numba", "numba==0.65.1"),\n'
        '    "llvmlite": ("llvmlite", "llvmlite==0.47.0"),\n',
        "pin the compatible tracksdata JIT dependency set",
    )
    result = _replace_once(
        result,
        '    "markdown_it": ("markdown_it", "markdown-it-py"),\n',
        '    "markdown_it": ("markdown_it", "markdown-it-py"),\n'
        '    "mdurl": ("mdurl", "mdurl"),\n',
        "close markdown-it offline dependency set",
    )
    result = _replace_once(
        result,
        'def packages_requiring_refresh() -> list[str]:\n'
        '    refresh: list[str] = []\n',
        'def packages_requiring_refresh() -> list[str]:\n'
        '    refresh: list[str] = []\n'
        '    try:\n'
        '        from importlib import metadata as _distribution_metadata\n'
        '        _v32_exact_distribution_versions = {\n'
        '            "numpy": "2.4.6",\n'
        '            "numba": "0.65.1",\n'
        '            "llvmlite": "0.47.0",\n'
        '        }\n'
        '        for _v32_distribution, _v32_expected_version in (\n'
        '            _v32_exact_distribution_versions.items()\n'
        '        ):\n'
        '            try:\n'
        '                if (\n'
        '                    _distribution_metadata.version(_v32_distribution)\n'
        '                    != _v32_expected_version\n'
        '                ):\n'
        '                    refresh.append(_v32_distribution)\n'
        '            except Exception:\n'
        '                refresh.append(_v32_distribution)\n'
        '    except Exception:\n'
        '        refresh.extend(["numpy", "numba", "llvmlite"])\n',
        "refresh incompatible bootstrap distributions without importing their ABIs",
    )
    result = _replace_once(
        result,
        '    force_reinstall = bool({"polars", "zarr"} & set(missing))\n',
        '    force_reinstall = bool(\n'
        '        {"polars", "zarr", "numpy", "numba", "llvmlite"} & set(missing)\n'
        '    )\n',
        "allow exact compatible bootstrap wheels to replace base ABI drift",
    )
    result = _replace_once(
        result,
        'def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:\n',
        dependency_restart_source()
        + '\n'
        + 'def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:\n',
        "insert one-shot authenticated clean-interpreter dependency restart",
    )
    result = _replace_once(
        result,
        '        if result.returncode == 0:\n'
        '            purge_imported_modules(missing)\n'
        '            print("Offline dependency install succeeded.")\n'
        '            return\n',
        '        if result.returncode == 0:\n'
        '            print("Offline dependency install succeeded; restarting cleanly.")\n'
        '            _v32_restart_after_dependency_install()\n',
        "restart instead of reusing an ABI-contaminated interpreter after offline install",
    )
    result = _replace_once(
        result,
        '        if result.returncode == 0:\n'
        '            purge_imported_modules(missing)\n'
        '            print("PyPI dependency install succeeded.")\n'
        '            return\n',
        '        if result.returncode == 0:\n'
        '            print("PyPI dependency install succeeded; restarting cleanly.")\n'
        '            _v32_restart_after_dependency_install()\n',
        "restart instead of reusing an ABI-contaminated interpreter after online install",
    )
    result = _replace_once(
        result,
        'def ensure_dependencies(artifacts: Path) -> None:\n',
        full_support_install_source()
        + '\n'
        + 'def ensure_dependencies(artifacts: Path) -> None:\n',
        "replace staged missing-only recovery with one exact full-support transaction",
    )
    result = _replace_once(
        result,
        'def link_or_copy_tree(src: Path, dst: Path) -> None:\n',
        'def make_owner_writable_tree(root: Path) -> None:\n'
        '    """Make only a copied working tree writable; never mutate provider inputs."""\n'
        '    pending = [root]\n'
        '    while pending:\n'
        '        path = pending.pop()\n'
        '        if path.is_symlink():\n'
        '            continue\n'
        '        mode = path.stat().st_mode\n'
        '        path.chmod(mode | 0o200 | (0o100 if path.is_dir() else 0))\n'
        '        if path.is_dir():\n'
        '            pending.extend(sorted(path.iterdir(), key=lambda value: value.name))\n'
        '\n'
        '\n'
        'def link_or_copy_tree(src: Path, dst: Path) -> None:\n',
        "add non-root-safe permissions for copied Kaggle working trees",
    )
    result = _replace_once(
        result,
        '    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)\n\n'
        '    weights_src = artifacts / "weights"\n',
        '    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)\n'
        '    make_owner_writable_tree(REPO_DIR)\n\n'
        '    weights_src = artifacts / "weights"\n',
        "repair read-only provider modes only after copying into Kaggle working",
    )
    old_inventory = textwrap.dedent(
        '''
        def inventory_input(label, root):
            root = Path(root)
            if not root.exists():
                _run_log("input_inventory", label=label, root=str(root), status="missing", files=[])
                return []
            files = []
            for directory, _dirs, names in os.walk(root):
                for name in sorted(names):
                    path = Path(directory) / name
                    if path.is_file():
                        entry = {"path": str(path), "bytes": path.stat().st_size}
                        if HASH_INPUT_CONTENTS:
                            entry["sha256"] = _run_sha256(path)
                        files.append(entry)
            _run_log("input_inventory", label=label, root=str(root), status="present",
                     content_hashing=HASH_INPUT_CONTENTS, files=files)
            return files
        '''
    ).strip() + "\n"
    result = _replace_once(
        result,
        old_inventory,
        exact_inventory_source(),
        "replace broad input inventory discovery",
    )
    result = _replace_once(
        result,
        '    inventory_input("competition_train", COMPETITION_TRAIN)\n',
        '    for _v32_selected_dataset in SELECTED_DATASETS:\n'
        '        inventory_input(\n'
        '            f"competition_train_selected:{_v32_selected_dataset}",\n'
        '            COMPETITION_TRAIN / f"{_v32_selected_dataset}.zarr",\n'
        '        )\n',
        "restrict raw inventory to the selected train identities",
    )
    eager_step0 = textwrap.dedent(
        '''
        STEP0 = run_step0()

        SELECTED_TRAIN = Path("/kaggle/working/public_temporalunet_selected_train")
        if SELECTED_TRAIN.exists():
            shutil.rmtree(SELECTED_TRAIN)
        SELECTED_TRAIN.mkdir(parents=True, exist_ok=False)
        for dataset in SELECTED_DATASETS:
            (SELECTED_TRAIN / f"{dataset}.zarr").symlink_to(
                COMPETITION_TRAIN / f"{dataset}.zarr", target_is_directory=True
            )
        os.environ["BIOHUB_TEST_DIR"] = str(SELECTED_TRAIN)
        preprocess_counts("selected_train_symlink_view", len(SELECTED_DATASETS), 0,
                          created_symlinks=len(SELECTED_DATASETS), raw_inputs_modified=0)
        '''
    ).strip() + "\n"
    result = _replace_once(
        result,
        eager_step0,
        deferred_step0_source(),
        "defer competition access until release dependency probe",
    )
    result = _replace_once(
        result,
        "ensure_dependencies(ARTIFACTS)\nmaterialize_inference_repo(ARTIFACTS)\n",
        dependency_probe_and_step0_source(),
        "insert release dependency probe and delayed step0",
    )
    result = _replace_once(
        result,
        "import subprocess\n",
        "import subprocess\n" + child_diagnostic_probe_source(),
        "insert child-diagnostic runtime-safety probe",
    )
    cover_cell = textwrap.dedent(
        '''
        # ---- PUBLIC V32 CODE CELL 0 ----
        from pathlib import Path
        from IPython.display import Image, display

        cover_image_path = Path("/kaggle/input/datasets/pilkwang/pilkwang-public-dataset-for-notebooks-figures/biohub_embryo.png")
        if cover_image_path.exists():
            display(Image(filename=str(cover_image_path)))
        '''
    ).strip() + "\n"
    result = _replace_once(
        result,
        cover_cell,
        "# ---- PUBLIC V32 CODE CELL 0 (non-scientific cover display omitted) ----\n",
        "remove cosmetic notebook-only cover import",
    )
    result = _replace_once(
        result,
        "import pandas as pd\n",
        ship_step0_source() + standalone_display_source() + "import pandas as pd\n",
        "insert exact ship Step 0 before heavy imports",
    )
    result = _replace_once(
        result,
        'if not _torch.cuda.is_available():\n'
        '    raise RuntimeError(\n'
        '        "CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again."\n'
        '    )\n'
        'print("CUDA device:", _torch.cuda.get_device_name(0))\n',
        'if _torch.cuda.is_available():\n'
        '    print("CUDA device:", _torch.cuda.get_device_name(0))\n'
        'elif SHIP_PREFLIGHT_MODE:\n'
        '    os.environ["BIOHUB_V32_CPU_PREFLIGHT_SKIP_ILP"] = "1"\n'
        '    print("V32_CPU_PREFLIGHT_DEVICE")\n'
        '    print("V32_CPU_PREFLIGHT_ILP_SKIPPED")\n'
        'else:\n'
        '    raise RuntimeError(\n'
        '        "CUDA GPU is required for a full V32 cache run; CPU is allowed only for bounded ship preflight."\n'
        '    )\n',
        "allow only bounded ship preflight to exercise the CPU fallback",
    )
    result = _replace_once(
        result,
        '    "cuda_device_name": _torch.cuda.get_device_name(0),\n',
        '    "cuda_device_name": (\n'
        '        _torch.cuda.get_device_name(0)\n'
        '        if _torch.cuda.is_available()\n'
        '        else "CPU_SHIP_PREFLIGHT"\n'
        '    ),\n',
        "make the terminal device receipt safe for bounded CPU preflight",
    )
    result = result.replace(
        KERNEL_MARKER, KERNEL_MARKER + "\n" + kernel_runtime_patch_source(), 1
    )
    terminal_anchor = (
        '_atomic_json(WORKING_DIR / "PUBLIC_TEMPORALUNET_RUN_RECEIPT.json", _receipt)'
    )
    result = _replace_once(
        result,
        terminal_anchor,
        terminal_cache_inventory_source()
        + textwrap.dedent(
            r'''

            # Preserve the exact final postprocessed graph rows under a name
            # that Kaggle cannot interpret as a competition submission.
            _v32_final_graph_rows = WORKING_DIR / "V32_FINAL_GRAPH_ROWS.csv"
            shutil.copy2(SUBMISSION_PATH, _v32_final_graph_rows)
            _receipt["final_graph_rows"] = {
                "path": str(_v32_final_graph_rows),
                "sha256": _sha256(_v32_final_graph_rows),
                "rows": row_id,
                "submission_filename": False,
            }
            print("V32_FINAL_GRAPH_ROWS_PRESERVED")
            '''
        )
        + terminal_anchor
        + textwrap.dedent(
            r'''

            # A cache-only kernel must never auto-create a competition submission.
            if SUBMISSION_PATH.exists():
                SUBMISSION_PATH.unlink()
            if SUBMISSION_PATH.exists():
                raise RuntimeError("V32 cache kernel failed to remove submission.csv")
            print("V32_CACHE_NO_SUBMISSION_CONFIRMED")
            '''
        ),
        "insert terminal cache inventory",
    )
    compile(result, "v32_preilp_cache_kernel.py", "exec")
    return result


def select_identities(
    population_manifest: Mapping[str, Any], *, role: str, canary_identity: str | None = None
) -> tuple[str, ...]:
    populations = population_manifest.get("populations")
    if not isinstance(populations, dict):
        raise CacheBuildError("population manifest has no populations object")
    held = tuple(str(value) for value in populations.get("held_fit105", ()))
    cv57 = tuple(str(value) for value in populations.get("cv57", ()))
    if role == "held_fit44b6":
        selected = tuple(value for value in held if value.startswith("train:44b6_"))
        expected = 36
    elif role == "held_fit6bba":
        selected = tuple(value for value in held if value.startswith("train:6bba_"))
        expected = 69
    elif role == "cv57":
        selected = cv57
        expected = 57
    elif role == "canary":
        if canary_identity != CANARY_IDENTITY or canary_identity not in held:
            raise CacheBuildError(f"canary identity is frozen at {CANARY_IDENTITY}")
        selected = (canary_identity,)
        expected = 1
    elif role == "visual_canary":
        if canary_identity is not None:
            raise CacheBuildError("visual canary has no selectable identity")
        if any(identity not in held for identity in VISUAL_CANARY_IDENTITIES):
            raise CacheBuildError("visual canary identities drifted outside held-fit105")
        selected = VISUAL_CANARY_IDENTITIES
        expected = 2
    elif role == "f2_six":
        if canary_identity is not None:
            raise CacheBuildError("F2 six-movie role has no selectable identity")
        if any(identity not in cv57 for identity in F2_SIX_IDENTITIES):
            raise CacheBuildError("F2 six-movie identities drifted outside CV57")
        selected = F2_SIX_IDENTITIES
        expected = 6
    else:
        raise CacheBuildError(f"unsupported role {role!r}")
    if len(selected) != expected or len(set(selected)) != expected:
        raise CacheBuildError(f"{role} identity count/uniqueness drift")
    return selected


def build_instrumented_source(
    *, population_manifest: Mapping[str, Any], role: str, canary_identity: str | None = None
) -> tuple[str, dict[str, Any]]:
    if sha256_file(BASE_BUILDER) != EXPECTED_BASE_BUILDER_SHA256:
        raise CacheBuildError("immutable public V32 builder hash drift")
    if sha256_file(POPULATION_DEFAULT) != EXPECTED_POPULATION_FILE_SHA256:
        raise CacheBuildError("frozen population file hash drift")
    exact_local_artifacts = {
        ROOT / "external/Tracking_Support/ARTIFACT_MANIFEST.json": EXPECTED_PRIMARY_MANIFEST_SHA256,
        ROOT / "external/Tracking_Support/weights/unet_transformer/split_0/edge_predictor_best.pth": EXPECTED_PRIMARY_WEIGHT_SHA256,
        ROOT / "external/Tracking_Support/weights/unet_transformer/split_0/config.json": EXPECTED_CONFIG_SHA256,
        ROOT / "external/public_temporalunet_baseline/secondary_seed314159/ARTIFACT_MANIFEST.json": EXPECTED_SECONDARY_MANIFEST_SHA256,
        ROOT / "external/public_temporalunet_baseline/secondary_seed314159/weights/unet_transformer/split_0/edge_predictor_best.pth": EXPECTED_SECONDARY_WEIGHT_SHA256,
        ROOT / "external/public_temporalunet_baseline/secondary_seed314159/weights/unet_transformer/split_0/config.json": EXPECTED_CONFIG_SHA256,
    }
    for path, expected_sha256 in exact_local_artifacts.items():
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise CacheBuildError(f"frozen V32 artifact hash drift: {path}")
    base = _load_base_builder()
    if sha256_file(base.PUBLIC_NOTEBOOK) != EXPECTED_PUBLIC_NOTEBOOK_SHA256:
        raise CacheBuildError("public V32 notebook hash drift")
    if sha256_file(base.ROOT / "external/Tracking_Support/repo/scripts/predict_unet_transformer.py") != EXPECTED_BASE_PREDICTOR_SHA256:
        raise CacheBuildError("base predictor hash drift")
    frozen_population = base.validate_population_manifest(POPULATION_DEFAULT)
    if frozen_population.get("manifest_sha256") != EXPECTED_POPULATION_SELF_SHA256:
        raise CacheBuildError("frozen population self-hash drift")
    checked = frozen_population if population_manifest is None else dict(population_manifest)
    if canonical_bytes(checked) != canonical_bytes(frozen_population):
        raise CacheBuildError("population object differs from the frozen population file")
    identities = select_identities(checked, role=role, canary_identity=canary_identity)
    subset = {
        "schema": "celltrack-alpha.v32-preilp-population.v1",
        "role": role,
        "identities": list(identities),
        "parent_population_self_sha256": str(checked["manifest_sha256"]),
    }
    subset_sha = canonical_sha256(subset)
    base_source = base.build_script(
        variant="public_v32",
        identities=identities,
        population_sha256=subset_sha,
        global_population_sha256=str(checked["manifest_sha256"]),
    )
    source = instrument_kernel_source(base_source)
    freeze = {
        "schema": SOURCE_FREEZE_SCHEMA,
        "status": "PASS_BUILD_ONLY",
        "role": role,
        "identities": list(identities),
        "subset_manifest_sha256": subset_sha,
        "population_self_sha256": str(checked["manifest_sha256"]),
        "public_notebook_sha256": EXPECTED_PUBLIC_NOTEBOOK_SHA256,
        "base_builder_sha256": EXPECTED_BASE_BUILDER_SHA256,
        "base_predictor_sha256": EXPECTED_BASE_PREDICTOR_SHA256,
        "expected_uninstrumented_runtime_predictor_sha256": EXPECTED_V32_RUNTIME_PREDICTOR_SHA256,
        "primary_manifest_sha256": EXPECTED_PRIMARY_MANIFEST_SHA256,
        "primary_weight_sha256": EXPECTED_PRIMARY_WEIGHT_SHA256,
        "primary_config_sha256": EXPECTED_CONFIG_SHA256,
        "secondary_manifest_sha256": EXPECTED_SECONDARY_MANIFEST_SHA256,
        "secondary_weight_sha256": EXPECTED_SECONDARY_WEIGHT_SHA256,
        "secondary_config_sha256": EXPECTED_CONFIG_SHA256,
        "runtime_split_sha256": runtime_split_sha256(identities),
        "population_file_sha256": EXPECTED_POPULATION_FILE_SHA256,
        "base_kernel_source_sha256": sha256_bytes(base_source.encode("utf-8")),
        "instrumented_kernel_source_sha256": sha256_bytes(source.encode("utf-8")),
        "cache_schema": CACHE_SCHEMA,
        "full_support_install_plan_sha256": (
            EXPECTED_FULL_SUPPORT_INSTALL_PLAN_SHA256
        ),
        "full_support_wheel_count": EXPECTED_FULL_SUPPORT_WHEEL_COUNT,
        "effective_v32_motion_relink": {
            "tight_um": 6.0,
            "relaxed_um": 10.0,
            "velocity_weight": 0.5,
            "raw_distance_weight": 0.05,
            "learned_bonus": 1.0,
            "max_frame_nodes": 2600,
            "output_edge_max_um": 14.0,
        },
        "label_free": True,
        "evaluate": False,
        "m3_mounted": False,
        "launch_authorized_by_this_manifest": False,
    }
    return source, freeze


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    population = json.loads(args.population_manifest.read_text(encoding="utf-8"))
    source, freeze = build_instrumented_source(
        population_manifest=population,
        role=args.role,
        canary_identity=args.canary_identity,
    )
    if args.output_dir.exists():
        raise CacheBuildError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    code_path = args.output_dir / "v32_preilp_cache_kernel.py"
    code_path.write_text(source, encoding="utf-8")
    metadata = {
        "id": args.kernel_id,
        "title": args.kernel_id.split("/", 1)[-1],
        "code_file": code_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "model_sources": [],
    }
    metadata_path = args.output_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        **freeze,
        "kernel_id": args.kernel_id,
        "files": {
            code_path.name: sha256_file(code_path),
            metadata_path.name: sha256_file(metadata_path),
        },
    }
    freeze_path = args.output_dir / "CACHE_SOURCE_FREEZE.json"
    freeze_path.write_bytes(canonical_bytes(freeze))
    return {**freeze, "source_freeze_sha256": sha256_file(freeze_path)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--population-manifest", type=Path, default=POPULATION_DEFAULT)
    value.add_argument(
        "--role",
        choices=(
            "canary", "visual_canary", "f2_six",
            "held_fit44b6", "held_fit6bba", "cv57",
        ),
        required=True,
    )
    value.add_argument("--canary-identity")
    value.add_argument("--kernel-id", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = build_package(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
