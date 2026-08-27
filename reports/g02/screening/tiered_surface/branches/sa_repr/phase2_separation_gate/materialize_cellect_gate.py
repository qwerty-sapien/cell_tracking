#!/usr/bin/env python3
"""Materialize the frozen fit-only CELLECT separation populations and feature requests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore

ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
for path in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    sys.path.insert(0, str(path))

import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.division_metrics import evaluate_divisions  # noqa: E402

AMENDMENT_SHA = "c3c86ec91307fc19a393f060912b50fe95fe42a0d68102508d154c3fe498458c"
DELETE_SHA = "e2b62eadb4925e3bd7108fe86f9945d2de40465d6a2c7f9e0dd797081cfda94d"
SWAP_SHA = "1441f2e49ada926b4b7b28a63b1a43849823c835c1e3f36a05212193afdef695"
CG0_RATE = 848.0 / 4_022_839.0


class MaterializeError(RuntimeError):
    pass


_DIVISION_PRED_GRAPH: Any = None
_DIVISION_GT_GRAPH: Any = None
_DIVISION_SCALE: tuple[float, float, float] | None = None


def exact_division_after_delete(task: tuple[int, int]) -> tuple[int, float]:
    """Return exact official division Jaccard after deleting one tracksdata edge."""
    index, edge_id = task
    if (
        _DIVISION_PRED_GRAPH is None
        or _DIVISION_GT_GRAPH is None
        or _DIVISION_SCALE is None
    ):
        raise MaterializeError("division worker was not initialized")
    candidate = _DIVISION_PRED_GRAPH.copy()
    candidate.remove_edge(edge_id=edge_id)
    counts = evaluate_divisions(candidate, _DIVISION_GT_GRAPH, scale=_DIVISION_SCALE)
    denominator = counts.tp + counts.fp + counts.fn
    return index, counts.tp / denominator if denominator else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fold(identity: str) -> int:
    return int.from_bytes(hashlib.sha256(identity.encode("ascii")).digest()[:8], "big") % 5


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise MaterializeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise MaterializeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def coords(graph: Any, pairs: np.ndarray) -> dict[str, np.ndarray]:
    count = len(pairs)
    result = {
        "source_id": pairs[:, 0].astype(np.int64, copy=False),
        "target_id": pairs[:, 1].astype(np.int64, copy=False),
        "source_t": np.empty(count, dtype=np.int16),
        "target_t": np.empty(count, dtype=np.int16),
        "source_zyx": np.empty((count, 3), dtype=np.int16),
        "target_zyx": np.empty((count, 3), dtype=np.int16),
    }
    for index, (source, target) in enumerate(pairs):
        a = graph.nodes[int(source)]
        b = graph.nodes[int(target)]
        result["source_t"][index] = int(a.t)
        result["target_t"][index] = int(b.t)
        result["source_zyx"][index] = (int(a.z), int(a.y), int(a.x))
        result["target_zyx"][index] = (int(b.z), int(b.y), int(b.x))
    if np.any(result["target_t"] != result["source_t"] + 1):
        raise MaterializeError("non-adjacent action endpoint")
    return result


def score_from_counts(row: dict[str, Any], tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    raw = tp / denominator if denominator else float("nan")
    adjusted = max(0.0, raw * (1.0 - 0.1 * float(row["total_node_ratio"])))
    div_denom = int(row["division_tp"]) + int(row["division_fp"]) + int(row["division_fn"])
    division = int(row["division_tp"]) / div_denom if div_denom else 0.0
    return adjusted + 0.1 * division


def materialize_delete(sd: Any, ss: Any, output: Path) -> dict[str, Any]:
    p0, _folds, _provenance, _inputs = sd.authenticate_inputs()
    archive = ROOT / p0["inputs"]["archive"]["path"]
    population, _records = sd.load_population(p0, archive=archive)
    rows = []
    with ZipStore(archive, mode="r") as store:
        for identity in sorted(population):
            video = population[identity]
            if video.family != "44b6":
                continue
            dataset = identity.split(":", 1)[1]
            fragment = ROOT / p0["control"]["fragment_root"] / f"{identity.replace(':', '__', 1)}.csv"
            graph = read_submission(fragment)[dataset]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            pred_to_gt, _ = ss.official_node_matching(graph, gt, scale)
            truth, gt_out, gt_in, _ = ss._truth_topology(gt)
            baseline_edges, outgoing, _incoming = ss._edge_sets(graph)
            base = dict(video.baseline)
            base_score = score_from_counts(
                base, int(base["edge_tp"]), int(base["edge_fp"]), int(base["edge_fn"])
            )
            reported_division = float(base.get("division_jaccard") or 0.0)
            reported_score = float(base["adj_edge_jaccard"]) + 0.1 * reported_division
            if not math.isclose(base_score, reported_score, rel_tol=0.0, abs_tol=1e-12):
                raise MaterializeError(
                    f"{identity}: analytic baseline score {base_score:.17g} does not "
                    f"replay reported score {reported_score:.17g}"
                )
            delta = np.zeros(len(video.edges), dtype=np.float64)
            fork_indices = []
            replay_tp = 0
            replay_fp = 0
            for index, raw_pair in enumerate(video.edges):
                pair = (int(raw_pair[0]), int(raw_pair[1]))
                is_true, is_valid = ss._edge_class(pair, pred_to_gt, truth, gt_out, gt_in)
                if is_true:
                    replay_tp += 1
                    after = score_from_counts(
                        base,
                        int(base["edge_tp"]) - 1,
                        int(base["edge_fp"]),
                        int(base["edge_fn"]) + 1,
                    )
                    delta[index] = after - base_score
                elif is_valid:
                    replay_fp += 1
                    after = score_from_counts(
                        base,
                        int(base["edge_tp"]),
                        int(base["edge_fp"]) - 1,
                        int(base["edge_fn"]),
                    )
                    delta[index] = after - base_score
                if len(outgoing.get(pair[0], ())) == 2:
                    fork_indices.append(index)
            expected_tp = int(base["edge_tp"])
            expected_fp = int(base["edge_fp"])
            if (replay_tp, replay_fp) != (expected_tp, expected_fp):
                raise MaterializeError(
                    f"{identity}: edge-class replay {(replay_tp, replay_fp)} does not "
                    f"match frozen baseline {(expected_tp, expected_fp)}"
                )
            # Exact official division replay for fork-breaking deletions. Edge and
            # node terms are already exact above, so recomputing them per fork is
            # redundant and prohibitively slow.
            if fork_indices:
                nodes, edge_rows = scoreall._prediction_rows(dataset, graph)
                pred_graph = prediction_to_graph(nodes, edge_rows)
                gt_graph = geff_arrays_to_graph(gt)
                ordered_original_ids = [
                    int(row["node_id"])
                    for row in sorted(
                        nodes, key=lambda row: (int(row["t"]), int(row["node_id"]))
                    )
                ]
                graph_ids = [int(value) for value in pred_graph.node_ids()]
                if len(ordered_original_ids) != len(graph_ids):
                    raise MaterializeError(f"{identity}: tracksdata node-map mismatch")
                node_map = dict(zip(ordered_original_ids, graph_ids, strict=True))
                edge_id_map = {
                    (int(row["source_id"]), int(row["target_id"])): int(row["edge_id"])
                    for row in pred_graph.edge_attrs().iter_rows(named=True)
                }
                tasks = []
                for index in fork_indices:
                    source = node_map[int(video.edges[index, 0])]
                    target = node_map[int(video.edges[index, 1])]
                    try:
                        edge_id = edge_id_map[(source, target)]
                    except KeyError as error:
                        raise MaterializeError(
                            f"{identity}: tracksdata edge-map mismatch {(source, target)}"
                        ) from error
                    tasks.append((index, edge_id))
                global _DIVISION_PRED_GRAPH, _DIVISION_GT_GRAPH, _DIVISION_SCALE
                _DIVISION_PRED_GRAPH = pred_graph
                _DIVISION_GT_GRAPH = gt_graph
                _DIVISION_SCALE = scale
                exact_divisions = [exact_division_after_delete(task) for task in tasks]
                for index, exact_division in exact_divisions:
                    delta[index] += 0.1 * (exact_division - reported_division)
                if identity == "train:44b6_0c582fdc":
                    parity_index = fork_indices[0]
                    kept = set(baseline_edges)
                    kept.remove(
                        (
                            int(video.edges[parity_index, 0]),
                            int(video.edges[parity_index, 1]),
                        )
                    )
                    changed = sd.p0_engine._copy_with(graph, edges=kept)
                    exact = sd.p0_engine._metric(changed, gt, scale)
                    exact_division = float(exact["division_jaccard"])
                    if not math.isfinite(exact_division):
                        exact_division = 0.0
                    exact_score = float(exact["adj_edge_jaccard"]) + 0.1 * exact_division
                    replay_score = base_score + float(delta[parity_index])
                    if not math.isclose(
                        replay_score, exact_score, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise MaterializeError(
                            f"{identity}: fork-local replay {replay_score:.17g} does "
                            f"not match full scorer {exact_score:.17g}"
                        )
                    print(
                        f"division parity {identity} index={parity_index} PASS",
                        flush=True,
                    )
            labels = (delta > 0.0).astype(np.uint8)
            pairs = np.asarray(video.edges, dtype=np.int64)
            endpoint = coords(graph, pairs)
            stem = identity.replace(":", "__", 1)
            population_path = output / "population" / "delete" / f"{stem}.npz"
            request_path = output / "feature_requests" / "delete" / f"{stem}.npz"
            budget = int(math.floor(CG0_RATE * int((labels == 0).sum())))
            save_npz(
                population_path,
                **endpoint,
                b0=np.asarray(video.features, dtype=np.float32),
                oracle_delta_S=delta,
                oracle_label=labels,
                fold=np.full(len(labels), fold(identity), dtype=np.int8),
                fp_budget=np.asarray([budget], dtype=np.int64),
            )
            save_npz(request_path, **endpoint)
            rows.append(
                {
                    "identity": identity,
                    "count": len(labels),
                    "beneficial": int(labels.sum()),
                    "harmful": int((labels == 0).sum()),
                    "division_changing": len(fork_indices),
                    "fp_budget": budget,
                    "population_path": str(population_path.relative_to(ROOT)),
                    "population_sha256": sha256(population_path),
                    "request_path": str(request_path.relative_to(ROOT)),
                    "request_sha256": sha256(request_path),
                }
            )
            print(f"delete {identity} actions={len(labels)} beneficial={int(labels.sum())}", flush=True)
    return {"class": "delete", "videos": rows}


def materialize_swap(ss: Any, output: Path) -> dict[str, Any]:
    prereg_path = ROOT / "reports/g02/screening/tiered_surface/branches/sa_swap/PREREGISTRATION.json"
    prereg = ss.authenticate(prereg_path, ss.PREREG_SHA256)
    feature_records = ss._feature_records(prereg)
    control_records, control_report = ss._control_records(prereg)
    identities = tuple(str(value) for value in control_report["scope"]["video_ids"])
    archive = ROOT / prereg["frozen_inputs"]["raw_archive"]["path"]
    rows = []
    with ZipStore(archive, mode="r") as store:
        for identity in identities:
            dataset = identity.split(":", 1)[1]
            graph = ss._graph(dataset, Path(control_records[identity]["resolved_path"]))
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, graph)
            baseline = ss.official_metric_row(nodes, edges, gt, scale)
            payload = ss._load_payload(feature_records[identity])
            records, _exclusions, witness = ss.build_population(graph, payload, gt, scale, baseline)
            arrays = ss._record_arrays(records)
            pairs = np.stack((arrays["source_ids"], arrays["target_ids"]), axis=1)
            endpoint = coords(graph, pairs)
            delta = np.asarray(arrays["regrets"], dtype=np.float64) / ss.TARGET_SCALE
            labels = (delta > 0.0).astype(np.uint8)
            stem = identity.replace(":", "__", 1)
            population_path = output / "population" / "occupied_swap" / f"{stem}.npz"
            request_path = output / "feature_requests" / "occupied_swap" / f"{stem}.npz"
            budget = int(math.floor(CG0_RATE * int((labels == 0).sum())))
            save_npz(
                population_path,
                **endpoint,
                b0=np.asarray(arrays["features"], dtype=np.float32),
                oracle_delta_S=delta,
                oracle_label=labels,
                fold=np.full(len(labels), fold(identity), dtype=np.int8),
                fp_budget=np.asarray([budget], dtype=np.int64),
            )
            save_npz(request_path, **endpoint)
            rows.append(
                {
                    "identity": identity,
                    "count": len(labels),
                    "beneficial": int(labels.sum()),
                    "harmful": int((labels == 0).sum()),
                    "fp_budget": budget,
                    "witness_compile_count": witness["witness_compile_count"],
                    "population_path": str(population_path.relative_to(ROOT)),
                    "population_sha256": sha256(population_path),
                    "request_path": str(request_path.relative_to(ROOT)),
                    "request_sha256": sha256(request_path),
                }
            )
            print(f"swap {identity} actions={len(labels)} beneficial={int(labels.sum())}", flush=True)
    return {"class": "occupied_swap", "videos": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if sha256(args.amendment) != AMENDMENT_SHA:
        raise SystemExit("per-class amendment identity mismatch")
    delete_path = ROOT / "reports/g02/screening/tiered_surface/branches/sa_delete/sa_delete.py"
    swap_path = ROOT / "reports/g02/screening/tiered_surface/branches/sa_swap/run_sa_swap.py"
    if sha256(delete_path) != DELETE_SHA or sha256(swap_path) != SWAP_SHA:
        raise SystemExit("Phase-1 implementation identity mismatch")
    sd = load_module("phase1_delete", delete_path)
    ss = load_module("phase1_swap", swap_path)
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    delete = materialize_delete(sd, ss, args.output)
    swap = materialize_swap(ss, args.output)
    manifest = {
        "schema": "celltrack-alpha.cellect-separation-population.v1",
        "status": "COMPLETE",
        "amendment_sha256": AMENDMENT_SHA,
        "cg0_acceptance_rate": CG0_RATE,
        "classes": {"delete": delete, "occupied_swap": swap},
        "totals": {
            name: {
                "actions": sum(row["count"] for row in value["videos"]),
                "beneficial": sum(row["beneficial"] for row in value["videos"]),
                "harmful": sum(row["harmful"] for row in value["videos"]),
                "fp_budget": sum(row["fp_budget"] for row in value["videos"]),
            }
            for name, value in (("delete", delete), ("occupied_swap", swap))
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu": False,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
        },
    }
    atomic_json(args.output / "population_manifest.json", manifest)
    print(json.dumps(manifest["totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
