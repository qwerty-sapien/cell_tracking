#!/usr/bin/env python3
"""Solve the preregistered three-node GPU launch portfolio with Gurobi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gurobipy as gp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-selector", type=Path, required=True)
    parser.add_argument("--kbest-contract", type=Path, required=True)
    parser.add_argument("--centerline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gpu-hours", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gap = json.loads(args.gap_selector.read_text(encoding="utf-8"))
    kbest = json.loads(args.kbest_contract.read_text(encoding="utf-8"))
    centerline = json.loads(args.centerline.read_text(encoding="utf-8"))
    nodes = [
        {
            "id": "NB-GAPSEL-v1",
            "projected_gpu_hours": 0.35,
            "eligible": gap["gate"]["verdict"] == "GO",
            "cpu_verdict": gap["gate"]["verdict"],
            "utility": max(0.0, float(gap["gate"]["minimum_family_adjusted_delta"])),
            "reason": "fit105 exact minimum-family adjusted delta must be >=0.003",
        },
        {
            "id": "K8-MINMARG-GRB-v1",
            "projected_gpu_hours": 0.35,
            "eligible": bool(kbest["activation"]["active"]),
            "cpu_verdict": kbest["activation"]["verdict"],
            "utility": 0.0,
            "reason": "owner-locked exact C0 cost/no-link contract and MAP control are required",
        },
        {
            "id": "TEMP-CENTERLINE-v1",
            "projected_gpu_hours": 0.55,
            "eligible": centerline["gate"]["verdict"] == "GO" and centerline["launch_eligible"],
            "cpu_verdict": centerline["gate"]["verdict"],
            "utility": max(0.0, float(centerline["gate"]["observed_oof_auc"]) - 0.5),
            "reason": "grouped OOF AUC >=0.70 on a fit-only exact-action bank is required",
        },
    ]
    model = gp.Model("g02_three_node_gpu_portfolio")
    model.Params.OutputFlag = 0
    launch = model.addVars(len(nodes), vtype=gp.GRB.BINARY, name="launch")
    model.addConstr(
        gp.quicksum(nodes[index]["projected_gpu_hours"] * launch[index] for index in range(len(nodes)))
        <= args.max_gpu_hours,
        name="aggregate_gpu_hours",
    )
    for index, node in enumerate(nodes):
        model.addConstr(launch[index] <= int(node["eligible"]), name=f"eligible_{index}")
        model.addConstr(
            node["projected_gpu_hours"] * launch[index] <= 7.999999,
            name=f"under_autoapproval_limit_{index}",
        )
    # One primary objective exists (frozen adjusted score); GPU-hours are a
    # deterministic secondary tie-break.  A Pareto search would manufacture a
    # choice where governance already defines hard feasibility constraints.
    model.setObjective(
        gp.quicksum(
            (nodes[index]["utility"] * 1_000_000.0 - nodes[index]["projected_gpu_hours"] * 1e-3)
            * launch[index]
            for index in range(len(nodes))
        ),
        gp.GRB.MAXIMIZE,
    )
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi portfolio status {model.Status} is not OPTIMAL")
    selected = []
    for index, node in enumerate(nodes):
        node["launch"] = bool(round(launch[index].X))
        if node["launch"]:
            selected.append(node["id"])
    payload = {
        "schema": "celltrack-alpha.g02-three-node-gpu-portfolio.v1",
        "status": "OPTIMAL",
        "solver": {
            "name": "gurobi",
            "version": ".".join(str(value) for value in gp.gurobi.version()),
            "objective": "lexicographic proxy: maximize eligible frozen-score utility, then minimize GPU-hours",
            "pareto_role": "diagnostic only; not used for selection",
        },
        "budget": {
            "maximum_total_gpu_hours": args.max_gpu_hours,
            "selected_projected_gpu_hours": sum(
                node["projected_gpu_hours"] for node in nodes if node["launch"]
            ),
            "per_run_explicit_approval_threshold_hours": 8.0,
        },
        "nodes": nodes,
        "selected": selected,
        "cmux_action": "SEND_ELIGIBLE_KERNELS_ONLY" if selected else "NO_SEND_NO_ELIGIBLE_NODE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
