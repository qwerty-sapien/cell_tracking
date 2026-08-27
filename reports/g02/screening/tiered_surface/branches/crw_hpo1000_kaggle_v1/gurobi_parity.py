#!/usr/bin/env python3
"""Compare compact exact CRW component selection with lexicographically exact Gurobi."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ALPHA = ROOT / "dag_prompts/2026-08-10-action-vocabulary-alpha"
CURRENT_LMT = ROOT / "dag_prompts/2026-08-12-lmt-resume"
for location in (ALPHA, CURRENT_LMT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import a1r2_crw_supplier as a1r2  # noqa: E402
import a1r4_lmt_runtime as lmt_runtime  # noqa: E402
import current_lmt  # noqa: E402


CARRIER = HERE / "input_dataset/crw_hpo1000_carrier.npz"
MANIFEST = HERE / "input_dataset/CARRIER_MANIFEST.json"
PREREG = HERE / "PREREGISTRATION.json"
OUTPUT = HERE / "GUROBI_PARITY_RECEIPT.json"


class ParityError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feasible_masks(slot_masks: np.ndarray) -> list[int]:
    n = len(slot_masks)
    result: list[int] = []
    for canonical_mask in range((1 << n) - 1, -1, -1):
        occupied = 0
        valid = True
        for position in range(n):
            bit = 1 << (n - 1 - position)
            if canonical_mask & bit:
                slots = int(slot_masks[position])
                if occupied & slots:
                    valid = False
                    break
                occupied |= slots
        if valid:
            result.append(canonical_mask)
    return result


def exact_select(slot_masks: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    best_value = -1
    best_mask = -1
    for mask in feasible_masks(slot_masks):
        value = 0
        valid = True
        for position, weight in enumerate(weights):
            bit = 1 << (len(weights) - 1 - position)
            if mask & bit:
                if int(weight) <= 0:
                    valid = False
                    break
                value += int(weight)
        if valid and (value > best_value or (value == best_value and mask > best_mask)):
            best_value, best_mask = value, mask
    return best_value, best_mask


def gurobi_select(env: Any, slot_masks: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("crw_hpo1000_parity", env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = 20260812
    variables = [model.addVar(vtype=GRB.BINARY, name=f"x_{index}") for index in range(len(weights))]
    for slot in range(32):
        owners = [variables[index] for index, mask in enumerate(slot_masks) if int(mask) & (1 << slot)]
        if owners:
            model.addConstr(gp.quicksum(owners) <= 1)
    primary = gp.quicksum(int(weights[index]) * variables[index] for index in range(len(weights)))
    for index, weight in enumerate(weights):
        if int(weight) <= 0:
            model.addConstr(variables[index] == 0)
    model.setObjective(primary, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ParityError(f"Gurobi primary status {model.Status}")
    optimum = int(round(model.ObjVal))
    model.addConstr(primary == optimum)
    mask = 0
    for position, variable in enumerate(variables):
        model.setObjective(variable, GRB.MAXIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise ParityError(f"Gurobi lexicographic status {model.Status}")
        bit_value = int(variable.X > 0.5)
        model.addConstr(variable == bit_value)
        if bit_value:
            mask |= 1 << (len(weights) - 1 - position)
    model.dispose()
    return optimum, mask


def main() -> int:
    if OUTPUT.exists():
        raise ParityError("refusing to overwrite immutable parity receipt")
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    if sha256(CARRIER) != manifest["carrier_npz"]["sha256"]:
        raise ParityError("carrier hash drifted")
    prereg = json.loads(PREREG.read_text(encoding="ascii"))
    with np.load(CARRIER, allow_pickle=False) as payload:
        ptr = np.asarray(payload["component_ptr"])
        indices = np.asarray(payload["component_candidate_index"])
        slots = np.asarray(payload["component_slot_mask"])
        candidate_ids = np.asarray(payload["candidate_id"])
    by_size: defaultdict[int, list[int]] = defaultdict(list)
    for component in range(len(ptr) - 1):
        by_size[int(ptr[component + 1] - ptr[component])].append(component)
    selected_components: list[int] = []
    for size in sorted(by_size):
        group = by_size[size]
        selected_components.extend(sorted({group[0], group[len(group) // 2], group[-1]}))

    telemetry = a1r2.Telemetry(cpu_limit_seconds=None, rss_limit_gib=None)
    monitor = a1r2.LMTResourceMonitor(a1r2.load_lmt_policy(ROOT / "LMT.md"), telemetry)
    workspace = current_lmt.WorkspaceGrowthGuard(ROOT)
    original_rss = lmt_runtime.install()
    started = time.monotonic()
    cases: list[dict[str, Any]] = []
    try:
        monitor.calibrate()
        monitor.start_background()
        workspace.start()
        import gurobipy as gp

        if ".".join(map(str, gp.gurobi.version())) != "13.0.2":
            raise ParityError("Gurobi version drifted")
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        try:
            for component in selected_components:
                start, stop = int(ptr[component]), int(ptr[component + 1])
                component_indices = indices[start:stop]
                component_slots = slots[start:stop]
                n = len(component_indices)
                digest = hashlib.sha256(b"".join(candidate_ids[component_indices].tolist())).digest()
                base = np.fromiter((digest[index % len(digest)] for index in range(n)), dtype=np.int64)
                vectors = {
                    "empty": np.zeros(n, dtype=np.int64),
                    "single": np.asarray([1000 if index == int(base.argmax()) else 0 for index in range(n)], dtype=np.int64),
                    "ascending": np.arange(1, n + 1, dtype=np.int64) * 1000,
                    "descending": np.arange(n, 0, -1, dtype=np.int64) * 1000,
                    "ties": np.full(n, 1000, dtype=np.int64),
                    "hashed": (base + 1) * 1000,
                }
                for vector_name, weights in vectors.items():
                    exact = exact_select(component_slots, weights)
                    gurobi = gurobi_select(env, component_slots, weights)
                    if exact != gurobi:
                        raise ParityError(
                            f"component {component} size {n} vector {vector_name}: {exact} != {gurobi}"
                        )
                    cases.append(
                        {
                            "component": component,
                            "size": n,
                            "vector": vector_name,
                            "primary_objective": exact[0],
                            "canonical_mask": exact[1],
                        }
                    )
                    monitor.check()
                    workspace.check()
        finally:
            env.dispose()
        receipt = {
            "schema": "celltrack-alpha.crw-hpo1000-gurobi-parity.v1",
            "status": "PASS",
            "preregistration_sha256": sha256(PREREG),
            "carrier_manifest_sha256": sha256(MANIFEST),
            "carrier_npz_sha256": sha256(CARRIER),
            "gurobi_version": "13.0.2",
            "component_sizes": sorted(by_size),
            "components_sampled": len(selected_components),
            "probability_weight_vectors_per_component": 6,
            "cases": len(cases),
            "case_digest": hashlib.sha256(canonical(cases)).hexdigest(),
            "exact_primary_and_canonical_bit_vector_matches": len(cases),
            "mismatches": 0,
            "elapsed_seconds": time.monotonic() - started,
            "resources": {"rss": monitor.snapshot(), "workspace": workspace.snapshot()},
            "preregistered_requirement": prereg["selection"]["gurobi_parity"],
        }
        OUTPUT.write_bytes(canonical(receipt))
        print(json.dumps({"status": "PASS", "cases": len(cases), "output": str(OUTPUT.relative_to(ROOT))}))
        return 0
    finally:
        monitor.stop()
        workspace.stop()
        lmt_runtime.restore(original_rss)


if __name__ == "__main__":
    raise SystemExit(main())
