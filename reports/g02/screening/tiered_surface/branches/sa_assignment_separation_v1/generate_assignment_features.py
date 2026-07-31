#!/usr/bin/env python3
"""Generate frozen C0/C2 assignment-evidence features for the 44b6 gate.

This program deliberately never reads ``oracle_label`` or ``oracle_delta_S``.
It reuses the hash-pinned C0 action reconstruction and exact finite component
enumeration, attaches binary assignment events to both frozen action classes,
and writes the eight-column A1 plus the frozen within-video/frame A0 control.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np


CONTRACT_SCHEMA = "celltrack-alpha.assignment-evidence-separation-contract.v1"
MANIFEST_SCHEMA = "celltrack-alpha.assignment-evidence-feature-manifest.v1"
FEATURE_NAMES = (
    "event_in_c0_map",
    "force_in_regret",
    "forbid_regret",
    "signed_min_marginal_c_in_minus_c_out",
    "kbest_membership_fraction",
    "kbest_rank_weighted_membership",
    "log1p_component_action_count",
    "log1p_local_competitor_count",
)
A0_SEED = "ASSIGNMENT-EVIDENCE-A0-V1"
K_BEST = 8
EXPECTED_COUNTS = {"delete": 1_345_497, "occupied_swap": 45_961}


class FeatureError(RuntimeError):
    """Raised when a frozen identity or feature invariant is violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_c0_module(path: Path, expected_sha256: str) -> Any:
    if sha256(path) != expected_sha256:
        raise FeatureError("C0 implementation SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("frozen_ae_p5_assignment", path)
    if spec is None or spec.loader is None:
        raise FeatureError("cannot load frozen C0 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def event_features(
    component: Sequence[Any],
    ranked: Sequence[tuple[Any, ...]],
    assignment_key: Callable[[Sequence[Any]], tuple[float, tuple[str, ...]]],
    event: Callable[[tuple[Any, ...]], bool],
    local_competitor_count: int,
) -> np.ndarray:
    """Return the frozen eight diagnostics for one binary edit event."""
    if not ranked:
        raise FeatureError("assignment component has no feasible state")
    costs = np.asarray([assignment_key(state)[0] for state in ranked], dtype=np.float64)
    if not np.isfinite(costs).all():
        raise FeatureError("nonfinite C0 assignment cost")
    membership = np.asarray([event(state) for state in ranked], dtype=bool)
    best_cost = float(costs[0])
    magnitude = 1.0 + math.fsum(abs(float(action.cost)) for action in component)

    if membership.any():
        c_include = float(costs[membership].min())
    else:
        c_include = best_cost + magnitude
    if (~membership).any():
        c_exclude = float(costs[~membership].min())
    else:
        # The explicit empty assignment makes this unreachable for the frozen
        # edit events, but retaining the symmetric rule makes it fail-finite.
        c_exclude = best_cost + magnitude

    k_eff = min(K_BEST, len(ranked))
    top = membership[:k_eff].astype(np.float64)
    weights = np.arange(k_eff, 0, -1, dtype=np.float64)
    values = np.asarray(
        [
            float(membership[0]),
            c_include - best_cost,
            c_exclude - best_cost,
            c_include - c_exclude,
            float(top.mean()),
            float(np.dot(top, weights) / weights.sum()),
            math.log1p(len(component)),
            math.log1p(local_competitor_count),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise FeatureError("nonfinite assignment feature")
    return values.astype(np.float32)


def isolated_delete_feature() -> np.ndarray:
    """Feature for a baseline edge no frozen candidate assignment can remove."""
    return np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def a0_permutation(
    a1: np.ndarray,
    action_class: str,
    identity: str,
    source_t: np.ndarray,
    source_id: np.ndarray,
    target_id: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Deterministically derange A1 within video/source-frame strata."""
    a0 = np.empty_like(a1)
    moved = 0
    singleton = 0
    for frame in sorted(set(int(value) for value in source_t.tolist())):
        indices = np.flatnonzero(source_t == frame)
        if len(indices) == 1:
            a0[indices] = a1[indices]
            singleton += 1
            continue
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: hashlib.sha256(
                (
                    f"{A0_SEED}|{action_class}|{identity}|{frame}|"
                    f"{int(source_id[index])}|{int(target_id[index])}|{index}"
                ).encode("utf-8")
            ).digest(),
        )
        donors = ordered[1:] + ordered[:1]
        a0[np.asarray(ordered)] = a1[np.asarray(donors)]
        moved += len(ordered)
    if not np.isfinite(a0).all():
        raise FeatureError("nonfinite A0 feature")
    return a0, {"moved_rows": moved, "singleton_rows": singleton}


def write_feature_file(
    output: Path,
    a1: np.ndarray,
    a0: np.ndarray,
) -> None:
    if output.exists():
        with np.load(output, allow_pickle=False) as prior:
            if (
                set(prior.files) != {"a1", "a0"}
                or not np.array_equal(prior["a1"], a1)
                or not np.array_equal(prior["a0"], a0)
            ):
                raise FeatureError(f"existing feature file differs: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, a1=a1, a0=a0)
    # np.savez appends .npz when the supplied temporary suffix is not .npz.
    actual = temporary if temporary.exists() else Path(f"{temporary}.npz")
    actual.replace(output)


def generate_video(
    *,
    c0: Any,
    identity: str,
    delete_shard: Path | None,
    swap_shard: Path | None,
    control_root: Path,
    feature_root: Path,
) -> dict[str, Any]:
    swap_by_row: dict[int, np.ndarray] = {}
    delete_by_edge: dict[tuple[int, int], np.ndarray] = {}
    component_count = 0
    max_component_size = 0
    max_assignment_count = 0

    if swap_shard is not None:
        control_path = control_root / f"{swap_shard.stem}.csv"
        if not control_path.is_file():
            raise FeatureError(f"missing pinned APNCC control {control_path}")
        graphs = c0.read_submission(control_path)
        if len(graphs) != 1:
            raise FeatureError(f"control is not exactly one graph: {control_path}")
        baseline = next(iter(graphs.values()))
        actions = c0.load_actions(swap_shard, baseline)
        if any(action.video != identity for action in actions):
            raise FeatureError("C0 video identity drift")

        for component in c0.components(actions):
            component_count += 1
            max_component_size = max(max_component_size, len(component))
            ranked = sorted(c0.feasible_assignments(component), key=c0._assignment_key)
            max_assignment_count = max(max_assignment_count, len(ranked))
            slot_sets = {action: set(action.owned) for action in component}
            for action in component:
                competitors = sum(
                    1
                    for other in component
                    if other is not action and slot_sets[action].intersection(slot_sets[other])
                )
                swap_by_row[action.row] = event_features(
                    component,
                    ranked,
                    c0._assignment_key,
                    lambda state, selected=action: selected in state,
                    competitors,
                )

            incumbent_edges = sorted(
                {
                    edge
                    for action in component
                    for edge in action.incumbent_edges
                }
            )
            for edge in incumbent_edges:
                removers = tuple(
                    action for action in component if edge in action.incumbent_edges
                )
                delete_by_edge[edge] = event_features(
                    component,
                    ranked,
                    c0._assignment_key,
                    lambda state, choices=removers: any(action in state for action in choices),
                    len(removers),
                )

    rows: dict[str, Any] = {}
    for action_class, shard in (
        ("delete", delete_shard),
        ("occupied_swap", swap_shard),
    ):
        if shard is None:
            continue
        with np.load(shard, allow_pickle=False) as data:
            required = ("source_id", "target_id", "source_t")
            if any(field not in data.files for field in required):
                raise FeatureError(f"{shard.name} lacks a frozen identity field")
            source_id = np.asarray(data["source_id"], dtype=np.int64)
            target_id = np.asarray(data["target_id"], dtype=np.int64)
            source_t = np.asarray(data["source_t"], dtype=np.int64)
        count = len(source_id)
        a1 = np.empty((count, len(FEATURE_NAMES)), dtype=np.float32)
        if action_class == "occupied_swap":
            if set(swap_by_row) != set(range(count)):
                raise FeatureError(f"{identity} swap row coverage mismatch")
            for index in range(count):
                a1[index] = swap_by_row[index]
        else:
            isolated = isolated_delete_feature()
            for index, pair in enumerate(zip(source_id, target_id, strict=True)):
                a1[index] = delete_by_edge.get(
                    (int(pair[0]), int(pair[1])), isolated
                )
        a0, permutation = a0_permutation(
            a1, action_class, identity, source_t, source_id, target_id
        )
        output = feature_root / action_class / shard.name
        write_feature_file(output, a1, a0)
        rows[action_class] = {
            "path": str(output),
            "sha256": sha256(output),
            "actions": count,
            "nonzero_a1_rows": int(np.any(a1 != 0.0, axis=1).sum()),
            "a0": permutation,
        }

    return {
        "identity": identity,
        "components": component_count,
        "max_component_actions": max_component_size,
        "max_feasible_assignments": max_assignment_count,
        "classes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--c0-implementation", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.manifest.exists():
        raise SystemExit(f"refusing to overwrite {args.manifest}")
    if sha256(args.contract) != args.contract_sha256:
        raise SystemExit("assignment-evidence contract SHA-256 mismatch")
    contract = json.loads(args.contract.read_text())
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "FROZEN_BEFORE_SUPPLIER_FEATURES_OR_SCORES"
    ):
        raise SystemExit("assignment-evidence contract is not frozen")
    if contract["feasibility_and_sampling"]["decision"] != "FULL_POPULATION_NO_SAMPLING":
        raise SystemExit("contract does not authorize full-population generation")

    started = time.perf_counter()
    c0 = load_c0_module(
        args.c0_implementation,
        contract["pins"]["c0_implementation"]["sha256"],
    )
    shards: dict[str, dict[str, Path]] = {}
    for action_class in ("delete", "occupied_swap"):
        for path in sorted((args.population_root / action_class).glob("*.npz")):
            identity = path.stem.replace("__", ":", 1)
            if action_class in shards.setdefault(identity, {}):
                raise FeatureError("duplicate population shard")
            shards[identity][action_class] = path
    if not shards:
        raise FeatureError("empty frozen population")

    videos = []
    totals = {"delete": 0, "occupied_swap": 0}
    for identity in sorted(shards):
        video = generate_video(
            c0=c0,
            identity=identity,
            delete_shard=shards[identity].get("delete"),
            swap_shard=shards[identity].get("occupied_swap"),
            control_root=args.control_root,
            feature_root=args.feature_root,
        )
        videos.append(video)
        for action_class, row in video["classes"].items():
            totals[action_class] += int(row["actions"])
    if totals != EXPECTED_COUNTS:
        raise FeatureError(f"population coverage mismatch: {totals}")

    file_rows = [
        {"action_class": action_class, "identity": video["identity"], **row}
        for video in videos
        for action_class, row in video["classes"].items()
    ]
    aggregate = hashlib.sha256()
    for row in sorted(file_rows, key=lambda item: (item["action_class"], item["identity"])):
        aggregate.update(
            f"{row['action_class']}|{row['identity']}|{row['sha256']}\n".encode("utf-8")
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "COMPLETE_FULL_POPULATION",
        "supplier": contract["supplier"],
        "amendment_sha256": contract["pins"]["frozen_preregistration"]["sha256"],
        "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
        "c0_implementation": {
            "path": str(args.c0_implementation),
            "sha256": sha256(args.c0_implementation),
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_width": len(FEATURE_NAMES),
        "feature_dtype": "float32",
        "population_coverage": 1.0,
        "action_counts": totals,
        "feature_files": sorted(
            file_rows, key=lambda item: (item["action_class"], item["identity"])
        ),
        "feature_file_aggregate_sha256": aggregate.hexdigest(),
        "component_census": {
            "components": sum(video["components"] for video in videos),
            "maximum_component_actions": max(
                video["max_component_actions"] for video in videos
            ),
            "maximum_feasible_assignments": max(
                video["max_feasible_assignments"] for video in videos
            ),
        },
        "a0": {
            "seed": A0_SEED,
            "rule": contract["alignment_control"]["rule"],
            "moved_rows": sum(
                row["a0"]["moved_rows"] for row in file_rows
            ),
            "singleton_rows": sum(
                row["a0"]["singleton_rows"] for row in file_rows
            ),
        },
        "integrity": {
            "oracle_label_read": False,
            "oracle_delta_read": False,
            "nonfinite_features": 0,
            "duplicate_or_missing_rows": 0,
            "sampled": False,
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_only": True,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "gpu": False,
            "submission": False,
        },
        "barrier": contract["barrier"]["halt"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(f".{args.manifest.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "action_counts": totals,
                "component_census": manifest["component_census"],
                "elapsed_seconds": manifest["execution"]["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
