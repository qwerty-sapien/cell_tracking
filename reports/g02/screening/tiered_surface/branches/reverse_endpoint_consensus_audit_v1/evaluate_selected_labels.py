#!/usr/bin/env python3
"""Open only the hash-sealed literal-consensus labels and apply the frozen gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SELECTION_SHA256 = "1ef8f3ed56f04567bd23ab87ad79582a0e39b859c01890b0fe3ce562cfd3e6c8"
FREEZE_SHA256 = "a9e7451f6d789e910401a80d898d8eff5fe8c4169eaade7d02ac4adef33aec2f"
POPULATION_MANIFEST_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"


class LabelGateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_json(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected:
        raise LabelGateError(f"JSON identity mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LabelGateError(f"JSON object required: {path}")
    return value


def decide(labels: np.ndarray, deltas: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    deltas = np.asarray(deltas, dtype=np.float64)
    if labels.shape != deltas.shape or labels.ndim != 1 or not np.isfinite(deltas).all():
        raise LabelGateError("selected label arrays are malformed")
    count = len(labels)
    beneficial = int(labels.sum())
    harmful = count - beneficial
    precision = beneficial / count if count else 0.0
    total_delta = float(deltas.sum())
    checks = {
        "edit_count_4_to_64": 4 <= count <= 64,
        "beneficial_at_least_4": beneficial >= 4,
        "harmful_zero": harmful == 0,
        "precision_one": precision == 1.0,
        "summed_oracle_delta_positive": total_delta > 0.0,
    }
    opened = all(checks.values())
    return {
        "edit_count": count,
        "beneficial": beneficial,
        "harmful": harmful,
        "precision": precision,
        "sum_oracle_delta_S": total_delta,
        "min_oracle_delta_S": float(deltas.min()) if count else None,
        "max_oracle_delta_S": float(deltas.max()) if count else None,
        "checks": checks,
        "graph_gate_open": opened,
        "verdict": "OPEN_EXACT_44B6_GRAPH" if opened else "CLOSE_GRAPH_EXACT_NOOP",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise LabelGateError(f"refusing to overwrite: {args.output}")
    selection = checked_json(args.selection, SELECTION_SHA256)
    freeze = checked_json(args.selection_freeze, FREEZE_SHA256)
    population = checked_json(args.population_manifest, POPULATION_MANIFEST_SHA256)
    if (
        selection.get("status") != "SEALED_LABEL_BLIND_SELECTION"
        or selection.get("scope", {}).get("labels_read") is not False
        or freeze.get("status") != "FROZEN_BEFORE_SELECTED_LABEL_READ"
        or freeze.get("selection", {}).get("sha256") != SELECTION_SHA256
        or population.get("status") != "COMPLETE"
    ):
        raise LabelGateError("selection was not frozen label-blind")
    video_rows = {
        str(row["identity"]): row
        for row in population["classes"]["occupied_swap"]["videos"]
    }
    indices: defaultdict[str, list[int]] = defaultdict(list)
    for row in selection["selected"]:
        indices[str(row["identity"])].append(int(row["index"]))
    label_values: list[int] = []
    delta_values: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    for identity in sorted(indices):
        record = video_rows.get(identity)
        if record is None:
            raise LabelGateError(f"population identity missing: {identity}")
        path = args.workspace_root / str(record["population_path"])
        if not path.is_file() or sha256(path) != str(record["population_sha256"]):
            raise LabelGateError(f"population shard identity mismatch: {identity}")
        with np.load(path, allow_pickle=False) as loaded:
            labels = np.asarray(loaded["oracle_label"], dtype=np.uint8)
            deltas = np.asarray(loaded["oracle_delta_S"], dtype=np.float64)
        if labels.shape != (int(record["count"]),) or deltas.shape != labels.shape:
            raise LabelGateError(f"population label alignment drifted: {identity}")
        for index in indices[identity]:
            label = int(labels[index])
            delta = float(deltas[index])
            label_values.append(label)
            delta_values.append(delta)
            selected_rows.append(
                {
                    "identity": identity,
                    "index": index,
                    "oracle_label": label,
                    "oracle_delta_S": delta,
                }
            )
    gate = decide(np.asarray(label_values), np.asarray(delta_values))
    return {
        "schema": "g02.reverse-endpoint-consensus.selected-label-gate.v1",
        "status": "COMPLETE",
        "selection_sha256": SELECTION_SHA256,
        "selection_freeze_sha256": FREEZE_SHA256,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "scope": {
            "selected_labels_read": len(label_values),
            "unselected_labels_read": 0,
            "ground_truth_read": False,
            "graph_scored": False,
        },
        "gate": gate,
        "selected_outcomes": selected_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
