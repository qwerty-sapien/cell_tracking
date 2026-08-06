#!/usr/bin/env python3
"""Verify local CPU Trackastra inference against the frozen fit105 T4 bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import zipfile

import numpy as np
from scipy.spatial import cKDTree
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
BRANCH = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
REMOTE_OUTPUT = BRANCH / "run_v1/kaggle_output"
TRACKASTRA_SOURCE = REMOTE_OUTPUT / "trackastra-source"
SUPPORT_SOURCE = REMOTE_OUTPUT / "src"
ASSET_ROOT = ROOT / "assets/trackastra-authentic-v1"
FIT_CACHE_ROOT = ROOT / "reports/g02/fitcache_117/bh9out/cache"
T4_BANK_ROOT = REMOTE_OUTPUT / "trackastra_native_fit105_bank_v1/videos"
SELECTED_ROOT = BRANCH / "fit105_sweep_scored/variants/r08_c08_a035_off"
MODEL_SHA256 = "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
ARCHIVE_SHA256 = "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e"
MAX_TOKENS = 1_024
WINDOW_FRAMES = 4
INFERENCE_RADIUS_UM = 12.0
INFERENCE_NEIGHBOR_CAP = 16
SELECTED_RADIUS_UM = 8.0
SELECTED_NEIGHBOR_CAP = 8
ASSOCIATION_THRESHOLD = 0.35
INITIAL_SOURCE_CHUNK = 128
SCALE_ZYX = (1.625, 0.40625, 0.40625)


class ParityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def configure_imports() -> None:
    for location in (TRACKASTRA_SOURCE, SUPPORT_SOURCE, ROOT / "src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    # Trackastra imports matplotlib only for unused visualization helpers.
    if "matplotlib" not in sys.modules:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")


def load_model():
    configure_imports()
    from trackastra.model.model import TrackingTransformer

    if sha256(ASSET_ROOT / "model.pt") != MODEL_SHA256:
        raise ParityError("checkpoint SHA mismatch")
    if sha256(ASSET_ROOT / "ctc.zip") != ARCHIVE_SHA256:
        raise ParityError("checkpoint archive SHA mismatch")
    with zipfile.ZipFile(ASSET_ROOT / "ctc.zip") as archive:
        config = yaml.safe_load(archive.read("ctc/config.yaml"))
    with torch.device("meta"):
        model = TrackingTransformer(**config)
    state = torch.load(
        ASSET_ROOT / "model.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True, assign=True)
    model.to(torch.device("cpu")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != 27_456_880:
        raise ParityError("Trackastra parameter count drifted")
    return model


def cache_identity(meta: dict):
    configure_imports()
    from celltrack_alpha import bh_model

    model = meta["contract"]["model"]
    return bh_model.CacheIdentity(
        str(model["config_sha256"]),
        str(model["source_sha256"]),
        str(model["weight_sha256"]),
        int(model["pair_batch_size"]),
        int(model["view_batch_size"]),
    )


def load_detections(identity: str):
    configure_imports()
    from celltrack_alpha import bh_model

    slug = identity.replace(":", "__", 1)
    meta = json.loads((FIT_CACHE_ROOT / f"{slug}.json").read_text(encoding="ascii"))
    return bh_model.load_sparse_cache_detections(
        FIT_CACHE_ROOT / f"{slug}.bhc", expected_identity=cache_identity(meta)
    )


def infer_association(model, ordered_nodes, by_node):
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
    timepoints = torch.from_numpy(coordinates[:, 0].astype(np.int64))
    coords = torch.from_numpy(coordinates).unsqueeze(0)
    features = torch.zeros((1, len(ordered_nodes), 12), dtype=torch.float32)
    with torch.inference_mode():
        logits = model(coords, features=features)
        association = model.normalize_output(
            logits, timepoints.unsqueeze(0), coords
        )[0]
    value = association.detach().to(device="cpu", dtype=torch.float32).numpy()
    if value.shape != (len(ordered_nodes), len(ordered_nodes)):
        raise ParityError("Trackastra association shape drifted")
    return value, {node: index for index, node in enumerate(ordered_nodes)}


def source_chunks(source_ids, source_coords, target_ids, target_coords):
    scale = np.asarray(SCALE_ZYX, dtype=np.float64)
    source_physical = source_coords * scale[None, :]
    target_physical = target_coords * scale[None, :]
    tree = cKDTree(target_physical)
    order = np.lexsort(
        (source_coords[:, 2], source_coords[:, 1], source_coords[:, 0])
    )
    pending = [
        order[start : start + INITIAL_SOURCE_CHUNK]
        for start in range(0, len(order), INITIAL_SOURCE_CHUNK)
    ]
    while pending:
        indices = pending.pop(0)
        candidates = []
        required_targets = set()
        for source_index in indices:
            hits = tree.query_ball_point(
                source_physical[source_index], r=INFERENCE_RADIUS_UM
            )
            hits = sorted(
                hits,
                key=lambda target_index: (
                    float(
                        np.linalg.norm(
                            target_physical[target_index]
                            - source_physical[source_index]
                        )
                    ),
                    int(target_ids[target_index]),
                ),
            )[:INFERENCE_NEIGHBOR_CAP]
            row = []
            for target_index in hits:
                distance = float(
                    np.linalg.norm(
                        target_physical[target_index]
                        - source_physical[source_index]
                    )
                )
                row.append((int(target_ids[target_index]), distance))
                required_targets.add(int(target_ids[target_index]))
            candidates.append((int(source_ids[source_index]), row))
        required_count = len(indices) + len(required_targets)
        if required_count > MAX_TOKENS:
            if len(indices) == 1:
                raise ParityError("one source neighborhood exceeds 1024 tokens")
            middle = len(indices) // 2
            pending.insert(0, indices[middle:])
            pending.insert(0, indices[:middle])
            continue
        yield candidates, required_targets


def run_video(identity: str, model) -> tuple[np.ndarray, np.ndarray, dict]:
    started = time.perf_counter()
    detections = load_detections(identity)
    by_node = {}
    frame_nodes = {}
    for frame in detections:
        ids = np.asarray(frame.node_ids, dtype=np.int64)
        coords = np.asarray(frame.coords_zyx, dtype=np.float32)
        frame_nodes[int(frame.frame_index)] = (ids, coords)
        for node, coord in zip(ids, coords, strict=True):
            by_node[int(node)] = (int(frame.frame_index), coord)
    all_frames = sorted(frame_nodes)
    min_frame, max_frame = all_frames[0], all_frames[-1]
    edge_source = []
    edge_target = []
    edge_weight = []
    edge_distance = []
    calls = peak_tokens = 0
    for source_t in range(min_frame, max_frame):
        if source_t not in frame_nodes or source_t + 1 not in frame_nodes:
            continue
        source_ids, source_coords = frame_nodes[source_t]
        target_ids, target_coords = frame_nodes[source_t + 1]
        if len(source_ids) == 0 or len(target_ids) == 0:
            continue
        start = min(max(source_t - 1, min_frame), max(min_frame, max_frame - 3))
        context_candidates = [
            int(node)
            for frame in range(start, start + WINDOW_FRAMES)
            if frame in frame_nodes
            for node in frame_nodes[frame][0]
        ]
        for candidates, required_targets in source_chunks(
            source_ids, source_coords, target_ids, target_coords
        ):
            required_sources = {source for source, _ in candidates}
            required = required_sources | required_targets
            centroid = np.mean([by_node[node][1] for node in required], axis=0)
            optional = sorted(
                set(context_candidates) - required,
                key=lambda node: (
                    float(np.sum((by_node[node][1] - centroid) ** 2)),
                    by_node[node][0],
                    node,
                ),
            )
            selected = required | set(optional[: MAX_TOKENS - len(required)])
            ordered = tuple(
                sorted(selected, key=lambda node: (by_node[node][0], node))
            )
            association, index = infer_association(model, ordered, by_node)
            calls += 1
            peak_tokens = max(peak_tokens, len(ordered))
            for source, targets in candidates:
                for target, distance in targets:
                    edge_source.append(source)
                    edge_target.append(target)
                    edge_weight.append(
                        float(association[index[source], index[target]])
                    )
                    edge_distance.append(distance)
    nodes = sorted(
        (
            int(node),
            int(t),
            float(coord[0]),
            float(coord[1]),
            float(coord[2]),
        )
        for node, (t, coord) in by_node.items()
    )
    nodes.sort(key=lambda row: (row[1], row[0]))
    edges = sorted(
        zip(edge_source, edge_target, edge_weight, edge_distance, strict=True),
        key=lambda row: (row[0], row[1]),
    )
    return (
        np.asarray(nodes, dtype=np.float64).reshape(-1, 5),
        np.asarray(edges, dtype=np.float64).reshape(-1, 4),
        {
            "identity": identity,
            "inference_calls": calls,
            "peak_tokens": peak_tokens,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def selected_edges(edges: np.ndarray) -> tuple[tuple[int, int], ...]:
    by_source: dict[int, list[tuple[int, float, float]]] = {}
    for source, target, weight, distance in edges:
        if distance <= SELECTED_RADIUS_UM + 1e-9:
            by_source.setdefault(int(source), []).append(
                (int(target), float(weight), float(distance))
            )
    eligible = []
    for source, candidates in by_source.items():
        for target, weight, distance in sorted(
            candidates, key=lambda row: (row[2], row[0])
        )[:SELECTED_NEIGHBOR_CAP]:
            eligible.append((source, target, weight, distance))
    incoming: set[int] = set()
    outgoing: set[int] = set()
    result = []
    for source, target, weight, _distance in sorted(
        eligible, key=lambda row: (-row[2], row[0], row[1])
    ):
        if weight < ASSOCIATION_THRESHOLD:
            break
        if target in incoming or source in outgoing:
            continue
        incoming.add(target)
        outgoing.add(source)
        result.append((source, target))
    return tuple(sorted(result))


def expected_graph_edges(identity: str) -> tuple[tuple[int, int], ...]:
    configure_imports()
    from celltrack_alpha.kaggle_calibration import read_submission

    path = SELECTED_ROOT / f"{identity.replace(':', '__', 1)}.csv"
    graphs = read_submission(path)
    if len(graphs) != 1:
        raise ParityError(f"graph ambiguity: {identity}")
    graph = next(iter(graphs.values()))
    return tuple(sorted((int(edge.source_id), int(edge.target_id)) for edge in graph.edges))


def compare(identity: str, model) -> dict:
    slug = identity.replace(":", "__", 1)
    nodes, edges, runtime = run_video(identity, model)
    with np.load(T4_BANK_ROOT / f"{slug}.npz") as archived:
        expected_nodes = archived["nodes"]
        expected_edges = archived["edges"]
    if not np.array_equal(nodes, expected_nodes):
        raise ParityError(f"node array drift: {identity}")
    if not np.array_equal(edges[:, :2], expected_edges[:, :2]):
        raise ParityError(f"candidate edge identity drift: {identity}")
    if not np.array_equal(edges[:, 3], expected_edges[:, 3]):
        raise ParityError(f"candidate distance drift: {identity}")
    weight_delta = np.abs(edges[:, 2] - expected_edges[:, 2])
    cpu_selected = selected_edges(edges)
    t4_selected = selected_edges(expected_edges)
    frozen_graph = expected_graph_edges(identity)
    if t4_selected != frozen_graph:
        raise ParityError(f"T4 bank does not reproduce frozen selected graph: {identity}")
    return {
        **runtime,
        "candidate_edges": int(len(edges)),
        "max_abs_weight_delta_cpu_vs_t4": float(weight_delta.max(initial=0.0)),
        "mean_abs_weight_delta_cpu_vs_t4": float(weight_delta.mean()),
        "threshold_side_flip_count": int(
            np.count_nonzero(
                (edges[:, 2] >= ASSOCIATION_THRESHOLD)
                != (expected_edges[:, 2] >= ASSOCIATION_THRESHOLD)
            )
        ),
        "cpu_selected_edge_count": len(cpu_selected),
        "t4_selected_edge_count": len(t4_selected),
        "cpu_selected_equals_t4": cpu_selected == t4_selected,
        "t4_selected_equals_frozen_graph": t4_selected == frozen_graph,
        "cpu_selected_symmetric_difference": len(
            set(cpu_selected).symmetric_difference(t4_selected)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity",
        action="append",
        default=[],
        help="Fit105 identity; may be supplied more than once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identities = args.identity or ["train:6bba_718b21f9"]
    if args.output.exists():
        raise ParityError(f"refusing to overwrite output: {args.output}")
    torch.manual_seed(20260731)
    torch.use_deterministic_algorithms(True)
    model = load_model()
    rows = [compare(identity, model) for identity in identities]
    report = {
        "schema": "celltrack-alpha.trackastra-native-local-cpu-parity.v1",
        "status": "PASS" if all(row["cpu_selected_equals_t4"] for row in rows) else "FAIL",
        "scope": "fit105 only; no CV57 cache, GT, graph, or score access",
        "device": "cpu",
        "torch_version": torch.__version__,
        "checkpoint_sha256": MODEL_SHA256,
        "checkpoint_archive_sha256": ARCHIVE_SHA256,
        "selected_configuration": "r08_c08_a035_off",
        "rows": rows,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
