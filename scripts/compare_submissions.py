#!/usr/bin/env python3
"""Compare two cell-tracking predictions at observation level.

The comparison is deliberately separate from local-to-Kaggle score transfer:

* raw and canonical identities are recorded before matching;
* node correspondence is deterministic, time-aware, one-to-one, and physical;
* prediction-to-prediction edge overlap is called edge Jaccard;
* node, edge, and structural-fork distances remain a vector;
* optional GT scoring routes through the attested frozen scorer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SCALE_ZYX,
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    read_submission,
)


DEFAULT_MAX_DISTANCE_UM = 7.0
ZERO_NULL_ENVELOPE = {
    "node_unmatched_fraction": 0.0,
    "node_displacement_max_um": 0.0,
    "raw_edge_disagreement_fraction": 0.0,
    "eligible_edge_disagreement_fraction": 0.0,
    "fork_disagreement_fraction": 0.0,
}


@dataclass(frozen=True)
class PredictionArtifact:
    label: str
    source: str
    files: tuple[dict[str, Any], ...]
    manifest_sha256: str
    canonical_sha256: str
    graphs: Mapping[str, SubmissionGraph]

    def identity_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "file_count": len(self.files),
            "files": list(self.files),
            "manifest_sha256": self.manifest_sha256,
            "canonical_sha256": self.canonical_sha256,
            "dataset_count": len(self.graphs),
            "node_count": sum(len(graph.nodes) for graph in self.graphs.values()),
            "edge_count": sum(len(graph.edges) for graph in self.graphs.values()),
        }


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_safe(value: Any) -> Any:
    """Convert scorer diagnostics to strict JSON without hiding non-finite values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _graph_semantics(graph: SubmissionGraph) -> dict[str, Any]:
    return {
        "dataset": graph.dataset,
        "nodes": [
            [node.node_id, node.t, node.z, node.y, node.x]
            for node in sorted(
                graph.nodes.values(),
                key=lambda node: (node.t, node.node_id, node.z, node.y, node.x),
            )
        ],
        "edges": sorted(
            [[edge.source_id, edge.target_id] for edge in graph.edges],
        ),
    }


def canonical_prediction_sha256(graphs: Mapping[str, SubmissionGraph]) -> str:
    digest = hashlib.sha256()
    for dataset in sorted(graphs):
        digest.update(_canonical_json(_graph_semantics(graphs[dataset])))
        digest.update(b"\n")
    return digest.hexdigest()


def _merge_graphs(
    destination: dict[str, SubmissionGraph],
    incoming: Mapping[str, SubmissionGraph],
    *,
    path: Path,
) -> None:
    overlap = sorted(set(destination).intersection(incoming))
    if overlap:
        raise ValueError(f"prediction input repeats datasets {overlap} in {path}")
    destination.update(incoming)


def load_prediction(path: Path, *, label: str) -> PredictionArtifact:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        paths = [resolved]
    elif resolved.is_dir():
        paths = sorted(
            candidate.resolve()
            for candidate in resolved.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == ".csv"
        )
        if not paths:
            raise ValueError(f"prediction directory contains no CSV files: {resolved}")
    else:
        raise ValueError(f"prediction input does not exist: {resolved}")

    records: list[dict[str, Any]] = []
    graphs: dict[str, SubmissionGraph] = {}
    for candidate in paths:
        sha256, size = _sha256_file(candidate)
        loaded = read_submission(candidate)
        _merge_graphs(graphs, loaded, path=candidate)
        records.append(
            {
                "path": str(candidate),
                "sha256": sha256,
                "bytes": size,
                "datasets": sorted(loaded),
            }
        )
    manifest_sha256 = hashlib.sha256(
        _canonical_json(
            [
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in records
            ]
        )
    ).hexdigest()
    return PredictionArtifact(
        label=label,
        source=str(resolved),
        files=tuple(records),
        manifest_sha256=manifest_sha256,
        canonical_sha256=canonical_prediction_sha256(graphs),
        graphs=graphs,
    )


def _node_coord(node: SubmissionNode) -> tuple[int, int, int, int]:
    return node.t, node.z, node.y, node.x


def _distance_um(a: SubmissionNode, b: SubmissionNode) -> float:
    return math.sqrt(
        sum(
            ((float(av) - float(bv)) * scale) ** 2
            for av, bv, scale in zip(
                a.coord_zyx,
                b.coord_zyx,
                SCALE_ZYX,
                strict=True,
            )
        )
    )


def _official_style_spatial_matches(
    a_nodes: Sequence[SubmissionNode],
    b_nodes: Sequence[SubmissionNode],
    *,
    max_distance_um: float,
) -> list[tuple[int, int, float]]:
    """Mirror tracksdata DistanceMatching for one already time-aligned frame."""

    if not a_nodes or not b_nodes:
        return []
    ordered_a = sorted(a_nodes, key=lambda node: node.node_id)
    ordered_b = sorted(b_nodes, key=lambda node: node.node_id)
    a_coords = np.asarray([node.coord_zyx for node in ordered_a], dtype=np.float64)
    b_coords = np.asarray([node.coord_zyx for node in ordered_b], dtype=np.float64)
    scale = np.asarray(SCALE_ZYX, dtype=np.float64)
    distances = cdist(b_coords * scale, a_coords * scale)
    b_indices, a_indices = np.where(distances <= max_distance_um)
    if len(a_indices) == 0:
        return []

    weights = sp.csr_array(
        (
            1.0 / (1.0 + distances[b_indices, a_indices]),
            (a_indices, b_indices),
        ),
        shape=(len(ordered_a), len(ordered_b)),
        dtype=np.float32,
    )
    try:
        rows, columns = sp.csgraph.min_weight_full_bipartite_matching(
            weights,
            maximize=True,
        )
    except ValueError:
        fill_value = -1.0
        dense = np.full(weights.shape, fill_value, dtype=np.float32)
        coo = weights.tocoo()
        dense[coo.row, coo.col] = coo.data
        rows, columns = linear_sum_assignment(dense, maximize=True)
        keep = ~np.isclose(dense[rows, columns], fill_value)
        rows = rows[keep]
        columns = columns[keep]

    result = [
        (
            ordered_a[int(row)].node_id,
            ordered_b[int(column)].node_id,
            float(distances[int(column), int(row)]),
        )
        for row, column in zip(rows, columns, strict=True)
        if float(distances[int(column), int(row)]) <= max_distance_um
    ]
    return sorted(result)


def match_nodes(
    graph_a: SubmissionGraph,
    graph_b: SubmissionGraph,
    *,
    max_distance_um: float = DEFAULT_MAX_DISTANCE_UM,
) -> tuple[dict[int, int], dict[str, Any]]:
    if max_distance_um <= 0:
        raise ValueError("max_distance_um must be positive")

    mapping_b_to_a: dict[int, int] = {}
    used_a: set[int] = set()
    used_b: set[int] = set()
    exact_id_coordinate = 0
    exact_coordinate = 0
    spatial_distances: list[float] = []

    for node_id in sorted(set(graph_a.nodes).intersection(graph_b.nodes)):
        node_a = graph_a.nodes[node_id]
        node_b = graph_b.nodes[node_id]
        if _node_coord(node_a) == _node_coord(node_b):
            mapping_b_to_a[node_id] = node_id
            used_a.add(node_id)
            used_b.add(node_id)
            exact_id_coordinate += 1

    a_by_coord: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    b_by_coord: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for node in graph_a.nodes.values():
        if node.node_id not in used_a:
            a_by_coord[_node_coord(node)].append(node.node_id)
    for node in graph_b.nodes.values():
        if node.node_id not in used_b:
            b_by_coord[_node_coord(node)].append(node.node_id)
    for coord in sorted(set(a_by_coord).intersection(b_by_coord)):
        for a_id, b_id in zip(
            sorted(a_by_coord[coord]),
            sorted(b_by_coord[coord]),
            strict=False,
        ):
            mapping_b_to_a[b_id] = a_id
            used_a.add(a_id)
            used_b.add(b_id)
            exact_coordinate += 1

    a_by_time: dict[int, list[SubmissionNode]] = defaultdict(list)
    b_by_time: dict[int, list[SubmissionNode]] = defaultdict(list)
    for node in graph_a.nodes.values():
        if node.node_id not in used_a:
            a_by_time[node.t].append(node)
    for node in graph_b.nodes.values():
        if node.node_id not in used_b:
            b_by_time[node.t].append(node)
    for timepoint in sorted(set(a_by_time).intersection(b_by_time)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
            matches = _official_style_spatial_matches(
                a_by_time[timepoint],
                b_by_time[timepoint],
                max_distance_um=max_distance_um,
            )
        for a_id, b_id, distance in matches:
            if a_id in used_a or b_id in used_b:
                raise AssertionError("spatial matcher returned a duplicate node")
            mapping_b_to_a[b_id] = a_id
            used_a.add(a_id)
            used_b.add(b_id)
            spatial_distances.append(distance)

    shared = len(mapping_b_to_a)
    a_only = len(graph_a.nodes) - shared
    b_only = len(graph_b.nodes) - shared
    union = shared + a_only + b_only
    details = {
        "a_nodes": len(graph_a.nodes),
        "b_nodes": len(graph_b.nodes),
        "shared": shared,
        "a_only": a_only,
        "b_only": b_only,
        "jaccard": shared / union if union else 1.0,
        "exact_id_coordinate_matches": exact_id_coordinate,
        "exact_coordinate_matches_after_id": exact_coordinate,
        "spatial_matches": len(spatial_distances),
        "spatial_distance_mean_um": (
            sum(spatial_distances) / len(spatial_distances)
            if spatial_distances
            else 0.0
        ),
        "spatial_distance_max_um": max(spatial_distances, default=0.0),
        "max_distance_um": max_distance_um,
    }
    return mapping_b_to_a, details


def _edge_set(
    graph: SubmissionGraph,
    *,
    consecutive_only: bool,
) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for edge in graph.edges:
        source = graph.nodes[edge.source_id]
        target = graph.nodes[edge.target_id]
        if consecutive_only and target.t != source.t + 1:
            continue
        result.add((edge.source_id, edge.target_id))
    return result


def compare_edges(
    graph_a: SubmissionGraph,
    graph_b: SubmissionGraph,
    mapping_b_to_a: Mapping[int, int],
    *,
    consecutive_only: bool,
) -> dict[str, Any]:
    edges_a = _edge_set(graph_a, consecutive_only=consecutive_only)
    edges_b = _edge_set(graph_b, consecutive_only=consecutive_only)
    mapped_b = {
        (mapping_b_to_a[source], mapping_b_to_a[target])
        for source, target in edges_b
        if source in mapping_b_to_a and target in mapping_b_to_a
    }
    shared = len(edges_a.intersection(mapped_b))
    a_only = len(edges_a) - shared
    b_only = len(edges_b) - shared
    union = shared + a_only + b_only
    return {
        "a_edges": len(edges_a),
        "b_edges": len(edges_b),
        "shared": shared,
        "a_only": a_only,
        "b_only": b_only,
        "jaccard": shared / union if union else 1.0,
        "b_edges_with_both_nodes_matched": len(mapped_b),
        "consecutive_only": consecutive_only,
    }


def _fork_event_signatures(
    graph: SubmissionGraph,
    *,
    node_map: Mapping[int, int] | None = None,
) -> set[tuple[Any, ...]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.source_id].append(edge.target_id)
        incoming[edge.target_id].append(edge.source_id)

    result: set[tuple[Any, ...]] = set()
    for divider in sorted(outgoing):
        children = sorted(set(outgoing[divider]))
        if len(children) != 2:
            continue
        involved = {
            divider,
            *incoming.get(divider, ()),
            *children,
            *(grandchild for child in children for grandchild in outgoing.get(child, ())),
        }
        if node_map is not None and not involved.issubset(node_map):
            continue

        def mapped(node_id: int) -> int:
            return node_map[node_id] if node_map is not None else node_id

        child_branches = tuple(
            sorted(
                (
                    mapped(child),
                    tuple(sorted(mapped(node) for node in outgoing.get(child, ()))),
                )
                for child in children
            )
        )
        result.add(
            (
                mapped(divider),
                tuple(sorted(mapped(node) for node in incoming.get(divider, ()))),
                child_branches,
            )
        )
    return result


def compare_forks(
    graph_a: SubmissionGraph,
    graph_b: SubmissionGraph,
    mapping_b_to_a: Mapping[int, int],
) -> dict[str, Any]:
    forks_a = _fork_event_signatures(graph_a)
    all_forks_b = _fork_event_signatures(graph_b)
    mapped_forks_b = _fork_event_signatures(graph_b, node_map=mapping_b_to_a)
    shared = len(forks_a.intersection(mapped_forks_b))
    a_only = len(forks_a) - shared
    b_only = len(all_forks_b) - shared
    union = shared + a_only + b_only
    return {
        "a_forks": len(forks_a),
        "b_forks": len(all_forks_b),
        "shared": shared,
        "a_only": a_only,
        "b_only": b_only,
        "agreement": shared / union if union else 1.0,
        "b_forks_with_full_local_topology_matched": len(mapped_forks_b),
        "semantics": "divider + predecessor + two child branches + grandchildren",
    }


def _empty_graph(dataset: str) -> SubmissionGraph:
    return SubmissionGraph(dataset=dataset, nodes={}, edges=[])


def compare_graphs(
    graph_a: SubmissionGraph,
    graph_b: SubmissionGraph,
    *,
    max_distance_um: float,
) -> dict[str, Any]:
    mapping, nodes = match_nodes(
        graph_a,
        graph_b,
        max_distance_um=max_distance_um,
    )
    return {
        "dataset": graph_a.dataset,
        "family": graph_a.dataset.split("_", 1)[0],
        "canonical_equal": _graph_semantics(graph_a) == _graph_semantics(graph_b),
        "nodes": nodes,
        "edges_raw": compare_edges(
            graph_a,
            graph_b,
            mapping,
            consecutive_only=False,
        ),
        "edges_scorer_eligible": compare_edges(
            graph_a,
            graph_b,
            mapping,
            consecutive_only=True,
        ),
        "forks": compare_forks(graph_a, graph_b, mapping),
    }


def _sum_component(
    rows: Iterable[Mapping[str, Any]],
    *,
    key: str,
    a_total_key: str,
    b_total_key: str,
    ratio_name: str,
) -> dict[str, Any]:
    components = [row[key] for row in rows]
    a_total = sum(int(component[a_total_key]) for component in components)
    b_total = sum(int(component[b_total_key]) for component in components)
    shared = sum(int(component["shared"]) for component in components)
    a_only = sum(int(component["a_only"]) for component in components)
    b_only = sum(int(component["b_only"]) for component in components)
    union = shared + a_only + b_only
    return {
        a_total_key: a_total,
        b_total_key: b_total,
        "shared": shared,
        "a_only": a_only,
        "b_only": b_only,
        ratio_name: shared / union if union else 1.0,
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nodes = _sum_component(
        rows,
        key="nodes",
        a_total_key="a_nodes",
        b_total_key="b_nodes",
        ratio_name="jaccard",
    )
    nodes.update(
        {
            "exact_id_coordinate_matches": sum(
                int(row["nodes"]["exact_id_coordinate_matches"]) for row in rows
            ),
            "exact_coordinate_matches_after_id": sum(
                int(row["nodes"]["exact_coordinate_matches_after_id"]) for row in rows
            ),
            "spatial_matches": sum(int(row["nodes"]["spatial_matches"]) for row in rows),
            "spatial_distance_max_um": max(
                (float(row["nodes"]["spatial_distance_max_um"]) for row in rows),
                default=0.0,
            ),
        }
    )
    return {
        "dataset_count": len(rows),
        "canonical_equal_datasets": sum(bool(row["canonical_equal"]) for row in rows),
        "nodes": nodes,
        "edges_raw": _sum_component(
            rows,
            key="edges_raw",
            a_total_key="a_edges",
            b_total_key="b_edges",
            ratio_name="jaccard",
        ),
        "edges_scorer_eligible": _sum_component(
            rows,
            key="edges_scorer_eligible",
            a_total_key="a_edges",
            b_total_key="b_edges",
            ratio_name="jaccard",
        ),
        "forks": _sum_component(
            rows,
            key="forks",
            a_total_key="a_forks",
            b_total_key="b_forks",
            ratio_name="agreement",
        ),
    }


def _fractions(aggregate: Mapping[str, Any]) -> dict[str, float]:
    nodes = aggregate["nodes"]
    raw = aggregate["edges_raw"]
    eligible = aggregate["edges_scorer_eligible"]
    forks = aggregate["forks"]
    return {
        "node_unmatched_fraction": (
            (nodes["a_only"] + nodes["b_only"])
            / max(1, nodes["a_nodes"] + nodes["b_nodes"])
        ),
        "node_displacement_max_um": float(nodes["spatial_distance_max_um"]),
        "raw_edge_disagreement_fraction": (
            (raw["a_only"] + raw["b_only"])
            / max(1, raw["a_edges"] + raw["b_edges"])
        ),
        "eligible_edge_disagreement_fraction": (
            (eligible["a_only"] + eligible["b_only"])
            / max(1, eligible["a_edges"] + eligible["b_edges"])
        ),
        "fork_disagreement_fraction": (
            (forks["a_only"] + forks["b_only"])
            / max(1, forks["a_forks"] + forks["b_forks"])
        ),
    }


def classify_distance(
    aggregate: Mapping[str, Any],
    *,
    null_envelope: Mapping[str, float] = ZERO_NULL_ENVELOPE,
) -> dict[str, Any]:
    observed = _fractions(aggregate)
    exact_coordinate_matches = (
        aggregate["nodes"]["exact_id_coordinate_matches"]
        + aggregate["nodes"]["exact_coordinate_matches_after_id"]
    )
    zero_distance = (
        aggregate["nodes"]["a_only"] == 0
        and aggregate["nodes"]["b_only"] == 0
        and exact_coordinate_matches == aggregate["nodes"]["shared"]
        and aggregate["edges_raw"]["a_only"] == 0
        and aggregate["edges_raw"]["b_only"] == 0
        and aggregate["forks"]["a_only"] == 0
        and aggregate["forks"]["b_only"] == 0
    )
    within_envelope = all(
        observed[key] <= float(null_envelope[key])
        for key in ZERO_NULL_ENVELOPE
    )
    if zero_distance:
        label = "IDENTICAL"
    elif within_envelope:
        label = "LOW-INFORMATION"
    else:
        label = "DISTINCT"
    exceeded = {
        key: {
            "observed": observed[key],
            "limit": float(null_envelope[key]),
        }
        for key in ZERO_NULL_ENVELOPE
        if observed[key] > float(null_envelope[key])
    }
    return {
        "label": label,
        "zero_distance": zero_distance,
        "observed": observed,
        "null_envelope": {
            key: float(null_envelope[key]) for key in ZERO_NULL_ENVELOPE
        },
        "exceeded_components": exceeded,
    }


def _load_null_envelope(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(ZERO_NULL_ENVELOPE)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("null envelope must be a JSON object")
    missing = sorted(set(ZERO_NULL_ENVELOPE) - set(value))
    extra = sorted(set(value) - set(ZERO_NULL_ENVELOPE))
    if missing or extra:
        raise ValueError(f"null envelope keys mismatch: missing={missing}, extra={extra}")
    result = {key: float(value[key]) for key in ZERO_NULL_ENVELOPE}
    if any(not math.isfinite(number) or number < 0 for number in result.values()):
        raise ValueError("null envelope values must be finite and non-negative")
    return result


def score_artifact(
    artifact: PredictionArtifact,
    *,
    gt_zip: Path,
    scorer_parity: Path,
) -> dict[str, Any]:
    if not gt_zip.is_file():
        raise FileNotFoundError(f"ground-truth zip does not exist: {gt_zip}")

    from zarr.storage import ZipStore

    import scoreall
    from score_bh_cv57 import validate_revalidated_scorer

    scorer_identity = validate_revalidated_scorer(scorer_parity)
    store = ZipStore(gt_zip, mode="r")
    rows: list[dict[str, Any]] = []
    try:
        for dataset in sorted(artifact.graphs):
            rows.append(
                scoreall.score_graph(
                    store,
                    f"train:{dataset}",
                    artifact.graphs[dataset],
                    source={
                        "mode": "observation_compare",
                        "artifact": artifact.canonical_sha256,
                    },
                )
            )
    finally:
        store.close()
    summary = scoreall.official_summary(rows)
    edge_counts = {
        key: int(sum(float(row[key]) for row in rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    edge_counts["W"] = sum(edge_counts.values())
    return {
        "scorer": scorer_identity,
        "summary": summary,
        "edge_counts": edge_counts,
        "samples": rows,
    }


def _parse_counts(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("expected edge counts must be TP,FP,FN")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def assert_score(
    result: Mapping[str, Any],
    *,
    expected_score: float | None,
    expected_counts: tuple[int, int, int] | None,
    label: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    if expected_score is not None:
        observed = float(result["summary"]["score"])
        passed = math.isclose(observed, expected_score, rel_tol=0.0, abs_tol=tolerance)
        checks["score"] = {
            "observed": observed,
            "expected": expected_score,
            "tolerance": tolerance,
            "pass": passed,
        }
        if not passed:
            raise AssertionError(f"{label} score mismatch: {observed} != {expected_score}")
    if expected_counts is not None:
        observed_counts = tuple(
            int(result["edge_counts"][key]) for key in ("edge_tp", "edge_fp", "edge_fn")
        )
        passed = observed_counts == expected_counts
        checks["edge_counts"] = {
            "observed": list(observed_counts),
            "expected": list(expected_counts),
            "pass": passed,
        }
        if not passed:
            raise AssertionError(
                f"{label} edge-count mismatch: {observed_counts} != {expected_counts}"
            )
    return checks


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    artifact_a = load_prediction(args.a, label=args.label_a)
    artifact_b = load_prediction(args.b, label=args.label_b)
    if args.dataset:
        datasets = sorted(
            value.split(":", 1)[1] if value.startswith("train:") else value
            for value in args.dataset
        )
        if len(set(datasets)) != len(datasets):
            raise ValueError("--dataset values must be unique")
        missing_a = sorted(set(datasets) - set(artifact_a.graphs))
        missing_b = sorted(set(datasets) - set(artifact_b.graphs))
        if missing_a or missing_b:
            raise ValueError(
                f"selected datasets must exist in both artifacts; "
                f"A missing={missing_a}, B missing={missing_b}"
            )
        if args.gt_zip is not None:
            raise ValueError("--dataset subset scoring with --gt-zip is not supported")
    else:
        datasets = sorted(set(artifact_a.graphs).union(artifact_b.graphs))
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        graph_a = artifact_a.graphs.get(dataset, _empty_graph(dataset))
        graph_b = artifact_b.graphs.get(dataset, _empty_graph(dataset))
        rows.append(
            compare_graphs(
                graph_a,
                graph_b,
                max_distance_um=args.max_distance_um,
            )
        )
    aggregate = aggregate_rows(rows)
    by_family: dict[str, Any] = {}
    families = sorted({row["family"] for row in rows})
    for family in families:
        by_family[family] = aggregate_rows(
            [row for row in rows if row["family"] == family]
        )
    envelope = _load_null_envelope(args.null_envelope)
    report: dict[str, Any] = {
        "schema": "celltrack-alpha.observation-distance.v1",
        "artifacts": {
            "a": artifact_a.identity_dict(),
            "b": artifact_b.identity_dict(),
        },
        "parameters": {
            "scale_zyx_um": list(SCALE_ZYX),
            "max_distance_um": args.max_distance_um,
            "matching": "exact ID+coordinate, exact coordinate, then official-style optimal physical matching per dataset/time",
            "edge_overlap_name": "edge_jaccard",
            "division_comparison": "structural fork/lineage agreement; GT division metrics only in optional frozen score",
            "selected_datasets": datasets,
        },
        "aggregate": aggregate,
        "by_family": by_family,
        "by_dataset": rows,
        "observation_class": classify_distance(
            aggregate,
            null_envelope=envelope,
        ),
    }
    if args.gt_zip is not None:
        score_a = score_artifact(
            artifact_a,
            gt_zip=args.gt_zip,
            scorer_parity=args.scorer_parity,
        )
        score_b = score_artifact(
            artifact_b,
            gt_zip=args.gt_zip,
            scorer_parity=args.scorer_parity,
        )
        report["frozen_score"] = {"a": score_a, "b": score_b}
        report["reconciliation"] = {
            "a": assert_score(
                score_a,
                expected_score=args.expect_a_score,
                expected_counts=_parse_counts(args.expect_a_edge_counts),
                label=args.label_a,
            ),
            "b": assert_score(
                score_b,
                expected_score=args.expect_b_score,
                expected_counts=_parse_counts(args.expect_b_edge_counts),
                label=args.label_b,
            ),
        }
    elif any(
        value is not None
        for value in (
            args.expect_a_score,
            args.expect_b_score,
            args.expect_a_edge_counts,
            args.expect_b_edge_counts,
        )
    ):
        raise ValueError("score/count expectations require --gt-zip")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-distance-um", type=float, default=DEFAULT_MAX_DISTANCE_UM)
    parser.add_argument("--null-envelope", type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Restrict comparison to a dataset present in both inputs; repeat as needed.",
    )
    parser.add_argument("--gt-zip", type=Path)
    parser.add_argument(
        "--scorer-parity",
        type=Path,
        default=ROOT / "reports/g02/scorer_parity_v2.json",
    )
    parser.add_argument("--expect-a-score", type=float)
    parser.add_argument("--expect-b-score", type=float)
    parser.add_argument("--expect-a-edge-counts")
    parser.add_argument("--expect-b-edge-counts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classification": report["observation_class"]["label"],
                "node_jaccard": report["aggregate"]["nodes"]["jaccard"],
                "raw_edge_jaccard": report["aggregate"]["edges_raw"]["jaccard"],
                "fork_agreement": report["aggregate"]["forks"]["agreement"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
