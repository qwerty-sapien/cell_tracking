#!/usr/bin/env python3
"""Exact-score velocity-recentered candidates on the frozen G1 CV57 base."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "reports/g04/lanes/velocity_aware_candidate_window_20260804"
for location in (
    ROOT / "src", ROOT / "scripts", ROOT / "external/Tracking_Support/repo/src", LANE,
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import g04_score_pathology_repairs as repair  # noqa: E402
import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.frame_registration import load_registration_plan  # noqa: E402


DEFAULT_CEILING = LANE / "fit117_recenter_v1/REACHABILITY_CEILING.json"
DEFAULT_G1_REPORT = ROOT / "reports/g04/lanes/data_pathology_repair/score/REPORT.json"
DEFAULT_G1_FRAGMENTS = ROOT / "reports/g04/lanes/data_pathology_repair/score/registered_g1/fragments"
DEFAULT_CENSUS = ROOT / "reports/g04/lanes/data_pathology_repair/CENSUS.json"
DEFAULT_OUTPUT = LANE / "exact_cv57_v1"
PERSISTENCE = 0.30


class ScoreError(RuntimeError):
    """A materiality, scorer-parity, or exact graph invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise ScoreError(f"refusing to overwrite {path}")
    temporary.write_bytes(canonical_bytes(dict(value)))
    os.replace(temporary, path)


@dataclass(frozen=True)
class CandidateRows:
    source_index: np.ndarray
    target_index: np.ndarray
    source_ids: np.ndarray
    target_ids: np.ndarray
    displacement_um: np.ndarray
    distance_um: np.ndarray
    source_density: np.ndarray
    target_density: np.ndarray
    peak_radius_neighbors: int

    def __len__(self) -> int:
        return len(self.source_index)


def query_candidates(
    source: bh_model.FrameDet,
    target: bh_model.FrameDet,
    query_source_raw: np.ndarray,
    *,
    config: bh_edge.SparseEdgeConfig,
) -> CandidateRows:
    scale = np.asarray(config.scale_zyx, dtype=np.float64)
    query_raw = np.asarray(query_source_raw, dtype=np.float64)
    if query_raw.shape != (len(source.node_ids), 3) or not np.isfinite(query_raw).all():
        raise ScoreError("motion-predicted query coordinates are malformed")
    query_points = query_raw * scale[None, :]
    target_raw = np.asarray(target.coords_zyx, dtype=np.float64)
    target_points = target_raw * scale[None, :]
    source_density = bh_edge._local_neighbor_counts(query_points, config.local_density_radius_um)
    target_density = bh_edge._local_neighbor_counts(target_points, config.local_density_radius_um)
    tree = cKDTree(target_points) if len(target_points) else None
    source_indices: list[int] = []
    target_indices: list[int] = []
    displacements: list[np.ndarray] = []
    distances: list[float] = []
    peak = 0
    if tree is not None:
        for source_row, centre in enumerate(query_points):
            nearby = [int(value) for value in tree.query_ball_point(centre, config.radius_um)]
            peak = max(peak, len(nearby))
            nearby.sort(
                key=lambda target_row: (
                    float(np.linalg.norm(target_points[target_row] - centre)),
                    int(target.node_ids[target_row]),
                    target_row,
                )
            )
            for target_row in nearby[: config.neighbor_cap]:
                displacement = target_points[target_row] - centre
                source_indices.append(source_row)
                target_indices.append(target_row)
                displacements.append(displacement)
                distances.append(float(np.linalg.norm(displacement)))
    source_index = np.asarray(source_indices, dtype=np.int64)
    target_index = np.asarray(target_indices, dtype=np.int64)
    displacement = (
        np.asarray(displacements, dtype=np.float64).reshape(-1, 3)
        if displacements else np.empty((0, 3), dtype=np.float64)
    )
    distance = np.asarray(distances, dtype=np.float64)
    if len(distance) and (np.max(distance) > config.radius_um + 1e-12 or not np.isfinite(distance).all()):
        raise ScoreError("recentered candidates violate fixed r10")
    return CandidateRows(
        source_index=source_index,
        target_index=target_index,
        source_ids=np.asarray(source.node_ids[source_index], dtype=np.int64),
        target_ids=np.asarray(target.node_ids[target_index], dtype=np.int64),
        displacement_um=displacement,
        distance_um=distance,
        source_density=np.asarray(source_density[source_index], dtype=np.float64),
        target_density=np.asarray(target_density[target_index], dtype=np.float64),
        peak_radius_neighbors=peak,
    )


def exact_score_rows(
    scorer: bh_edge.SparseEdgeScorer,
    context: bh_model.PairContext,
    source: bh_model.FrameDet,
    target: bh_model.FrameDet,
    query_source_raw: np.ndarray,
) -> tuple[dict[tuple[int, int], float], dict[int, float], CandidateRows]:
    """Run the frozen scorer exactly, changing only candidate/query coordinates."""

    candidates = query_candidates(source, target, query_source_raw, config=scorer.config)
    source_hidden, target_hidden = scorer._checkpoint_node_features(context, source, target)
    device = scorer._device()
    source_index = torch.from_numpy(np.array(candidates.source_index, copy=True)).to(device=device)
    target_index = torch.from_numpy(np.array(candidates.target_index, copy=True)).to(device=device)
    query_coords = torch.from_numpy(np.ascontiguousarray(query_source_raw, dtype=np.float32)).to(device=device)
    target_coords = torch.from_numpy(np.ascontiguousarray(target.coords_zyx, dtype=np.float32)).to(device=device)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(candidates), scorer.config.pair_chunk_size):
            end = min(start + scorer.config.pair_chunk_size, len(candidates))
            sources = source_index[start:end]
            targets = target_index[start:end]
            relative = (query_coords[sources] - target_coords[targets]) / 100.0
            pair_features = torch.cat((source_hidden[sources], target_hidden[targets], relative), dim=1)
            chunks.append(scorer.pair_mlp(pair_features).squeeze(-1).float())
    checkpoint_logits = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32, device=device)
    residual_features = scorer._residual_features(
        checkpoint_logits, context, source, target, candidates, None
    )
    logits_tensor = checkpoint_logits.detach() + scorer.calibrator(residual_features)
    logits = logits_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    probabilities = np.empty(len(candidates), dtype=np.float64)
    no_link = np.ones(len(source.node_ids), dtype=np.float64)
    dummy = float(scorer.config.dummy_logit + scorer.calibrator.dummy_residual.detach().cpu().item())
    rows_by_source: list[list[int]] = [[] for _ in range(len(source.node_ids))]
    for row, source_row in enumerate(candidates.source_index):
        rows_by_source[int(source_row)].append(row)
    for source_row, local_rows in enumerate(rows_by_source):
        if not local_rows:
            continue
        rows = np.asarray(local_rows, dtype=np.int64)
        local_logits = logits[rows]
        maximum = max(dummy, float(np.max(local_logits)))
        dummy_exp = math.exp(dummy - maximum)
        candidate_exp = np.exp(local_logits - maximum)
        denominator = dummy_exp + float(candidate_exp.sum())
        probabilities[rows] = candidate_exp / denominator
        no_link[source_row] = dummy_exp / denominator
    if not np.isfinite(probabilities).all() or not np.isfinite(no_link).all():
        raise ScoreError("recentered scorer returned nonfinite probabilities")
    pair_map = {
        (int(source_id), int(target_id)): float(probability)
        for source_id, target_id, probability in zip(
            candidates.source_ids, candidates.target_ids, probabilities, strict=True
        )
    }
    no_link_map = {
        int(source_id): float(probability)
        for source_id, probability in zip(source.node_ids, no_link, strict=True)
    }
    return pair_map, no_link_map, candidates


def query_raw_by_frame(
    frames: Sequence[bh_model.FrameDet], selected_edges: Sequence[tuple[int, int]]
) -> dict[int, np.ndarray]:
    frame_by_id = {
        int(node): int(frame.frame_index)
        for frame in frames for node in frame.node_ids
    }
    point = {
        int(node): np.asarray(coord, dtype=np.float64)
        for frame in frames for node, coord in zip(frame.node_ids, frame.coords_zyx, strict=True)
    }
    parent: dict[int, int] = {}
    for source, target in selected_edges:
        if source in frame_by_id and target in frame_by_id and frame_by_id[target] == frame_by_id[source] + 1:
            if target in parent:
                raise ScoreError("G1 graph contains a duplicate adjacent in-edge")
            parent[target] = source
    result: dict[int, np.ndarray] = {}
    for frame in frames[:-1]:
        query = np.asarray(frame.coords_zyx, dtype=np.float64).copy()
        for row, source_raw in enumerate(frame.node_ids):
            source = int(source_raw)
            previous = parent.get(source)
            if previous is not None and frame_by_id.get(previous) == frame.frame_index - 1:
                query[row] += PERSISTENCE * (point[source] - point[previous])
        result[int(frame.frame_index)] = query
    return result


def local_edges(
    source: bh_model.FrameDet,
    target: bh_model.FrameDet,
    query_raw: np.ndarray,
    probabilities: Mapping[tuple[int, int], float],
    no_link: Mapping[int, float],
) -> set[tuple[int, int]]:
    nodes: list[bh_graph.GraphNode] = []
    for row, node_raw in enumerate(source.node_ids):
        coord = query_raw[row]
        nodes.append(
            bh_graph.GraphNode(int(node_raw), int(source.frame_index), float(coord[0]), float(coord[1]), float(coord[2]), float(source.scores[row]))
        )
    for row, node_raw in enumerate(target.node_ids):
        coord = np.asarray(target.coords_zyx[row], dtype=np.float64)
        nodes.append(
            bh_graph.GraphNode(int(node_raw), int(target.frame_index), float(coord[0]), float(coord[1]), float(coord[2]), float(target.scores[row]))
        )
    edges, _stats = bh_graph.build_adjacent_edges(
        nodes,
        learned_edge_probabilities=probabilities,
        source_no_link_probabilities=no_link,
        topology_mode="legacy",
    )
    return {(int(edge.source_id), int(edge.target_id)) for edge in edges}


def parity_control(
    scorer: bh_edge.SparseEdgeScorer,
    context: bh_model.PairContext,
    source: bh_model.FrameDet,
    target: bh_model.FrameDet,
) -> dict[str, Any]:
    query = np.asarray(source.coords_zyx, dtype=np.float64)
    pair_map, no_link, candidates = exact_score_rows(scorer, context, source, target, query)
    reference_candidates = bh_edge.gate_sparse_candidates(source, target, config=scorer.config)
    reference = scorer.score_pair(context, source, target, candidates=reference_candidates)
    if (
        not np.array_equal(candidates.source_index, reference_candidates.source_index)
        or not np.array_equal(candidates.target_index, reference_candidates.target_index)
        or not np.array_equal(candidates.source_ids, reference.source_ids)
        or not np.array_equal(candidates.target_ids, reference.target_ids)
    ):
        raise ScoreError("zero-recenter candidate membership differs from stock scorer")
    observed = np.asarray([pair_map[(int(a), int(b))] for a, b in zip(reference.source_ids, reference.target_ids, strict=True)])
    observed_no_link = np.asarray([no_link[int(node)] for node in reference.no_link_source_ids])
    if not np.array_equal(observed, reference.probabilities) or not np.array_equal(observed_no_link, reference.no_link_probabilities):
        raise ScoreError("zero-recenter probabilities are not bit-exact with stock scorer")
    return {"candidate_count": len(candidates), "pair_probabilities_bit_exact": True, "no_link_probabilities_bit_exact": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ScoreError(f"refusing to overwrite score output: {args.output}")
    ceiling = repair.load_json(args.ceiling)
    if (
        ceiling.get("status") != "MATERIAL_SCORE_AUTHORIZED"
        or ceiling.get("pre_scorer_gate", {}).get("verdict") != "SCORE_AUTHORIZED"
        or ceiling.get("pre_scorer_gate", {}).get("exact_scorer_called") is not False
        or int(ceiling.get("strict_conservation", {}).get("residual_edges", -1)) != 366
        or ceiling.get("strict_conservation", {}).get("guard_relaxed") is not False
    ):
        raise ScoreError("material prescorer gate is absent or drifted")
    report = repair.load_json(args.g1_report)
    selected = tuple(map(str, report["selection"]["identities"]))
    control_summary = report["scores"]["registered_g1"]["summary"]
    control_vector = (
        float(control_summary["adj_edge_jaccard"]),
        float(control_summary["edge_jaccard"]),
        float(control_summary["node_recall"]),
    )
    if len(selected) != 57 or control_vector != (0.8867644202595942, 0.891461012028204, 0.990377056057473):
        raise ScoreError("frozen G1 CV57 control drifted")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    fragment_root = args.output / "fragments"
    fragment_root.mkdir()
    base_summary, base_provenance = repair.score_fragment_dir(args.zip, args.g1_fragments, selected)
    if (
        float(base_summary["adj_edge_jaccard"]),
        float(base_summary["edge_jaccard"]),
        float(base_summary["node_recall"]),
    ) != control_vector:
        raise ScoreError("input G1 fragments do not reproduce the frozen control")
    census = repair.census_by_identity(args.census)
    model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact, sha256(args.edge_artifact), args.edge_auxiliary,
        model, bundle, edge_module=bh_edge,
    )
    if scorer is None or edge_record is None or edge_record.get("sha256") != "5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034":
        raise ScoreError("frozen edge scorer failed identity restoration")

    totals: Counter[str] = Counter()
    parity_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    for ordinal, identity in enumerate(selected, start=1):
        source_path = repair.fragment_path(args.g1_fragments, identity)
        base_rows = repair.read_fragment(source_path)
        _dataset, _original, selected_edges = __import__("run_velocity_window_audit").read_fragment(source_path)
        cache = repair.load_cache(args.cache, identity)
        plan = load_registration_plan(args.census, identity)
        registered = repair.translated_frames(cache, census[identity]["transitions"], plan=plan)
        by_frame = {int(frame.frame_index): frame for frame in registered}
        query = query_raw_by_frame(registered, selected_edges)
        replacements: dict[int, set[tuple[int, int]]] = {}
        candidate_count = 0
        peak = 0
        for context in cache.contexts:
            frame = int(context.source_frame)
            source = by_frame[frame]
            target = by_frame[frame + 1]
            if frame == 0:
                parity_rows.append({"identity": identity, "frame": 0, **parity_control(scorer, context, source, target)})
            probabilities, no_link, candidates = exact_score_rows(
                scorer, context, source, target, query[frame]
            )
            replacements[frame] = local_edges(
                source, target, query[frame], probabilities, no_link
            )
            candidate_count += len(candidates)
            peak = max(peak, candidates.peak_radius_neighbors)
        raw_ids = {int(node_id) for frame in cache.detections for node_id in frame.node_ids}
        output_rows, edits = repair.apply_registered(base_rows, replacements, raw_ids)
        output_path = repair.fragment_path(fragment_root, identity)
        repair.write_fragment(output_path, output_rows)
        totals.update(edits)
        totals["candidate_pairs"] += candidate_count
        totals["peak_uncapped_neighbors"] = max(totals["peak_uncapped_neighbors"], peak)
        video_rows.append({
            "identity": identity, "candidate_pairs": candidate_count,
            "peak_uncapped_neighbors": peak, "graph_edit": edits,
            "output_sha256": sha256(output_path),
        })
        print(f"velocity score {ordinal:02d}/57 {identity}", flush=True)

    summary, provenance = repair.score_fragment_dir(args.zip, fragment_root, selected)
    vector = (
        float(summary["adj_edge_jaccard"]),
        float(summary["edge_jaccard"]),
        float(summary["node_recall"]),
    )
    result = {
        "schema": "celltrack-alpha.velocity-aware-candidate-window-exact-cv57.v1",
        "status": "COMPLETED_EXACT_SCORE",
        "mechanism": {
            "name": "VELOCITY-AWARE-CANDIDATE-WINDOW-v1",
            "base": "global-jump-corrected APNCC",
            "radius_um": 10.0, "cap": 8, "radius_widened": False,
            "lag1_persistence": PERSISTENCE,
            "candidate_only_recenter": True,
            "checkpoint_features_and_image_sampling": "unchanged; qhat replaces source coordinate only for candidate geometry and its existing relative/displacement inputs",
            "scorer_weights_changed": False, "graph_matcher": "unchanged legacy",
        },
        "prescorer_authorization": {"path": str(args.ceiling), "sha256": sha256(args.ceiling)},
        "control": {
            "adjusted_edge_jaccard": control_vector[0], "raw_edge_jaccard": control_vector[1],
            "node_recall": control_vector[2], "aggregate_sha256": base_provenance["aggregate_sha256"],
        },
        "score": {
            "adjusted_edge_jaccard": vector[0], "raw_edge_jaccard": vector[1],
            "node_recall": vector[2],
            "delta_adjusted": vector[0] - control_vector[0],
            "delta_raw": vector[1] - control_vector[1],
            "delta_node": vector[2] - control_vector[2],
            "summary": summary, **provenance,
        },
        "parity_control": {
            "scope": "frame 0 (no possible predecessor) for all 57 CV videos",
            "rows": parity_rows,
            "all_pair_probabilities_bit_exact": all(row["pair_probabilities_bit_exact"] for row in parity_rows),
            "all_no_link_probabilities_bit_exact": all(row["no_link_probabilities_bit_exact"] for row in parity_rows),
        },
        "totals": dict(totals),
        "videos": video_rows,
        "provenance": {
            "g1_report": {"path": str(args.g1_report), "sha256": sha256(args.g1_report)},
            "g1_fragments": str(args.g1_fragments),
            "census": {"path": str(args.census), "sha256": sha256(args.census)},
            "edge_scorer": edge_record,
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started, "device": "cpu",
            "kaggle_action": False, "runpod_action": False,
        },
    }
    write_new(args.output / "EXACT_SCORE.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    value.add_argument("--g1-report", type=Path, default=DEFAULT_G1_REPORT)
    value.add_argument("--g1-fragments", type=Path, default=DEFAULT_G1_FRAGMENTS)
    value.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    value.add_argument("--cache", type=Path, default=sweep.DEFAULT_CACHE)
    value.add_argument("--edge-artifact", type=Path, default=sweep.DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=sweep.DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=sweep.DEFAULT_SUPPORT)
    value.add_argument("--zip", type=Path, default=sweep.DEFAULT_ZIP)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "score": result["score"]}, sort_keys=True))
