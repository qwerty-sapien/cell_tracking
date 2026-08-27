#!/usr/bin/env python3
"""Dry validation for the frozen D1 motif/suppression contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import polars as pl
import tracksdata as td


ROOT = Path(__file__).resolve().parents[7]
SCORER_SRC = ROOT / "external/kaggle-cell-tracking-competition/src"
sys.path.insert(0, str(SCORER_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d1_decoder import (  # noqa: E402
    ForkCandidate,
    duplicate_survivors,
    emit_forks,
    suppress_candidates,
    validate_graph,
)
from tracking_cellmot.division_metrics import evaluate_divisions  # noqa: E402


EXPECTED = {
    "3_division": (0, 1, 0),
    "complex": (1, 0, 0),
    "complex2": (1, 0, 0),
    "division": (1, 1, 1),
    "division_at_end": (1, 0, 0),
    "division_node_stage": (1, 2, 0),
    "division_test": (1, 0, 0),
    "duplicated_division": (1, 1, 0),
    "edges_wrong_division_ok": (1, 0, 1),
    "late_division": (1, 0, 0),
    "merge_delay": (1, 0, 0),
    "node_conflict": (1, 0, 0),
    "simple": (0, 0, 0),
    "successive_div": (1, 1, 1),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _graph(spec: dict[str, Any]) -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    ids: dict[int, int] = {}
    for node in spec["nodes"]:
        ids[int(node["uid"])] = graph.add_node(
            {"t": int(node["t"]), "z": 0.0, "y": float(node["y"]), "x": 0.0}
        )
    for edge in spec["edges"]:
        graph.add_edge(ids[int(edge["source"])], ids[int(edge["target"])], {})
    return graph


def validate_official_fixtures() -> list[dict[str, Any]]:
    fixtures = ROOT / "external/kaggle-cell-tracking-competition/tests/assets/sandbox_examples"
    results: list[dict[str, Any]] = []
    for name, expected in EXPECTED.items():
        path = fixtures / f"{name}.json"
        case = json.loads(path.read_text(encoding="utf-8"))
        counts = evaluate_divisions(
            _graph(case["pred"]),
            _graph(case["gt"]),
            max_distance=float(case["max_distance"]),
        )
        observed = (int(counts.tp), int(counts.fp), int(counts.fn))
        if observed != expected:
            raise AssertionError(f"{name}: expected {expected}, observed {observed}")
        results.append(
            {
                "name": name,
                "expected_tp_fp_fn": list(expected),
                "observed_tp_fp_fn": list(observed),
                "fixture_sha256": _sha256(path),
            }
        )
    return results


def counterfactual(seed: int) -> dict[str, Any]:
    offset = seed * 100
    parent = offset + 1
    children = (offset + 2, offset + 3)
    grandchildren = (offset + 4, offset + 5)
    nodes_t = {
        parent: 1,
        children[0]: 2,
        children[1]: 2,
        grandchildren[0]: 3,
        grandchildren[1]: 3,
        offset + 20: 1,
        offset + 21: 2,
        offset + 22: 2,
    }
    base_edges = (
        (parent, children[0]),
        (children[0], grandchildren[0]),
        (children[1], grandchildren[1]),
    )
    winner = ForkCandidate(
        parent,
        children[0],
        children[1],
        grandchildren[0],
        grandchildren[1],
        0.90 + seed / 10000.0,
        f"event-{seed}",
    )
    duplicate = ForkCandidate(
        offset + 20,
        offset + 21,
        offset + 22,
        None,
        None,
        0.20,
        f"event-{seed}",
    )
    selected, suppressed = suppress_candidates(
        (duplicate, winner),
        require_persistence=True,
    )
    edges = emit_forks(base_edges, selected)
    validate_graph(nodes_t, edges)
    if selected != (winner,):
        raise AssertionError("persistent winner was not selected")
    if duplicate_survivors(selected) != 0:
        raise AssertionError("equivalent duplicate survived")
    if len(suppressed) != 1:
        raise AssertionError("expected one suppressed alternative")
    return {
        "seed": seed,
        "selected": [candidate.equivalence_key for candidate in selected],
        "suppressed": [
            {
                "key": record.candidate.equivalence_key,
                "reason": record.reason,
                "winner": record.winner_key,
            }
            for record in suppressed
        ],
        "edges": [list(edge) for edge in edges],
        "indegree_le_1": True,
        "outdegree_le_2": True,
        "dag_and_time_forward": True,
    }


def main() -> int:
    official = validate_official_fixtures()
    first = [counterfactual(seed) for seed in range(51)]
    second = [counterfactual(seed) for seed in range(51)]
    first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode("ascii")
    second_bytes = json.dumps(second, sort_keys=True, separators=(",", ":")).encode("ascii")
    if first_bytes != second_bytes:
        raise AssertionError("dry repeats are not byte-identical")
    report = {
        "schema": "celltrack-alpha.d1-dry-validation.v1",
        "official_fixture_count": len(official),
        "official_fixtures": official,
        "counterfactual_count": len(first),
        "counterfactuals": first,
        "duplicate_survivors": 0,
        "repeat_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "repeated_output_byte_identical": True,
        "held_graph_scoring": False,
        "cv57_access": False,
        "gpu_action": False,
        "verdict": "GO",
    }
    target = Path(__file__).resolve().parent / "N1_dry_validation.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "official_fixture_count",
        "counterfactual_count",
        "duplicate_survivors",
        "repeat_sha256",
        "repeated_output_byte_identical",
        "verdict",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
