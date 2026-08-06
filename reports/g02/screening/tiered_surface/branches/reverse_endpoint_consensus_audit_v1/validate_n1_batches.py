#!/usr/bin/env python3
"""Validate exact N1 node-disjointness without opening a graph score."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SELECTION_SHA256 = "1ef8f3ed56f04567bd23ab87ad79582a0e39b859c01890b0fe3ce562cfd3e6c8"
APNCC_REPORT_SHA256 = "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d"
N1_SHA256 = "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5"


class N1ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_json(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected:
        raise N1ValidationError(f"JSON identity mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise N1ValidationError(f"JSON object required: {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise N1ValidationError(f"refusing to overwrite: {args.output}")
    selection = checked_json(args.selection, SELECTION_SHA256)
    report = checked_json(args.apncc_report, APNCC_REPORT_SHA256)
    n1_path = args.workspace_root / "src/celltrack_alpha/action_compiler.py"
    if sha256(n1_path) != N1_SHA256:
        raise N1ValidationError("N1 implementation identity mismatch")
    sys.path.insert(0, str(args.workspace_root / "src"))
    from celltrack_alpha.action_compiler import (
        ActionKind,
        GraphAction,
        compile_batch,
        graph_sha256,
    )
    from celltrack_alpha.kaggle_calibration import read_submission

    control_rows = {
        str(row["identity"]): row
        for row in report["graph"]["control"]["input_provenance"]["files"]
    }
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selection["selected"]:
        grouped[str(row["identity"])].append(row)
    batch_rows = []
    total_actions = total_removed = 0
    for identity in sorted(grouped):
        control = control_rows[identity]
        path = Path(str(control["path"]))
        if sha256(path) != str(control["sha256"]):
            raise N1ValidationError(f"APNCC fragment identity mismatch: {identity}")
        dataset = identity.split(":", 1)[1]
        graphs = read_submission(path)
        graph = graphs.get(dataset)
        if graph is None or len(graphs) != 1:
            raise N1ValidationError(f"APNCC graph mismatch: {identity}")
        owned: set[int] = set()
        actions = []
        expected_removed: set[tuple[int, int]] = set()
        for row in grouped[identity]:
            touched = set(map(int, row["owned_nodes"]))
            if touched & owned:
                raise N1ValidationError(f"node ownership overlap: {identity}")
            owned.update(touched)
            source = int(row["source_id"])
            target = int(row["target_id"])
            removed = tuple(
                sorted(tuple(map(int, edge)) for edge in row["removed_edges"])
            )
            expected_removed.update(removed)
            actions.append(
                GraphAction(
                    action_id=f"literal-consensus:{source}:{target}",
                    kind=ActionKind.SWAP,
                    add_edges=((source, target),),
                    replace_outgoing_from=(source,),
                    replace_incoming_to=(target,),
                    expected_conflict_edges=removed,
                )
            )
        compiled = compile_batch(graph, tuple(actions))
        if (
            set(compiled.removed_edges) != expected_removed
            or set(compiled.graph.nodes) != set(graph.nodes)
            or any(
                (action.add_edges[0][0], action.add_edges[0][1])
                not in {
                    (int(edge.source_id), int(edge.target_id))
                    for edge in compiled.graph.edges
                }
                for action in actions
            )
        ):
            raise N1ValidationError(f"compiled batch semantics drifted: {identity}")
        total_actions += len(actions)
        total_removed += len(compiled.removed_edges)
        batch_rows.append(
            {
                "identity": identity,
                "actions": len(actions),
                "removed_edges": len(compiled.removed_edges),
                "owned_nodes": len(owned),
                "before_graph_sha256": graph_sha256(graph),
                "after_graph_sha256": graph_sha256(compiled.graph),
                "observation_distinct": graph_sha256(graph) != graph_sha256(compiled.graph),
                "batch_sha256": compiled.batch_sha256,
            }
        )
    if total_actions != 53:
        raise N1ValidationError(f"selected action count drifted: {total_actions}")
    return {
        "schema": "g02.reverse-endpoint-consensus.n1-validation.v1",
        "status": "PASS",
        "selection_sha256": SELECTION_SHA256,
        "n1_implementation_sha256": N1_SHA256,
        "actions": total_actions,
        "videos": len(batch_rows),
        "removed_edges": total_removed,
        "all_node_disjoint": True,
        "all_n1_batches_accepted": True,
        "all_edited_graphs_observation_distinct": all(
            row["observation_distinct"] for row in batch_rows
        ),
        "graph_scored": False,
        "batches": batch_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--apncc-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "actions", "videos", "removed_edges")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
