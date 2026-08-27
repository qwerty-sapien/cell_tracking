#!/usr/bin/env python3
"""Audit threshold-only headroom in the parked point-context OOF models."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
import evaluate_point_context_oof_graph_gate as point_gate  # noqa: E402
from fit_compact_point_context_pilot import score_windows  # noqa: E402
from fit_selected_graph_temporal_rank_gate import (  # noqa: E402
    THRESHOLD_GRID_SIZE,
    _metrics,
    _select,
)


SCHEMA = "celltrack-alpha.point-context-oof-threshold-headroom.v1"
CONTRACT_SCHEMA = (
    "celltrack-alpha.point-context-oof-threshold-headroom-contract.v1"
)
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/point_context_oof_threshold_headroom_contract_v1.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "fc71f52e0f87286629863bd054f7c840af3a6c56b4eec6e86c360b79ec8308d8"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/point_context_oof_threshold_headroom_v1/report.json"
)


class ThresholdHeadroomError(RuntimeError):
    """The fit-only threshold-headroom diagnostic failed closed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ThresholdHeadroomError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ThresholdHeadroomError(f"{label} must be a JSON object")
    return value


def _root_file(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThresholdHeadroomError(f"{label} path is missing")
    path = ROOT / value
    if not path.is_file():
        raise ThresholdHeadroomError(f"{label} is missing: {path}")
    return path


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load the immutable diagnostic contract and enforce its closed scope."""

    if len(expected_sha256) != 64 or sweep._sha256(path) != expected_sha256:
        raise ThresholdHeadroomError("diagnostic contract SHA-256 mismatch")
    value = _load_json(path, "diagnostic contract")
    source = value.get("source")
    scope = value.get("scope")
    diagnostic = value.get("diagnostic")
    gate = value.get("plausibility_gate")
    decision = value.get("decision")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or not all(
            isinstance(section, Mapping)
            for section in (source, scope, diagnostic, gate, decision)
        )
    ):
        raise ThresholdHeadroomError("diagnostic contract is malformed")
    floors = diagnostic.get("calibration_precision_floor_grid")
    expected_floors = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]
    if (
        scope.get("family") != "6bba"
        or int(scope.get("fit_train_video_count", 0)) != 69
        or int(scope.get("outer_fold_count", 0)) != 5
        or scope.get("fit_only") is not True
        or any(
            scope.get(name) is not False
            for name in (
                "validation12_access",
                "cv57_access",
                "kaggle_action",
                "gpu_action",
                "runpod_action",
            )
        )
        or floors != expected_floors
        or int(diagnostic.get("minimum_calibration_accepted", 0)) != 25
        or int(diagnostic.get("oracle_minimum_held_accepted", 0)) != 25
        or float(diagnostic.get("oracle_minimum_held_precision", 0.0))
        != 0.95
        or int(diagnostic.get("threshold_grid_size", 0))
        != THRESHOLD_GRID_SIZE
        or diagnostic.get("reproduce_banked_fold_metrics_before_analysis")
        is not True
        or decision.get("promotion_authorized") is not False
        or decision.get("graph_replay_authorized") is not False
        or decision.get("cv57_authorized") is not False
    ):
        raise ThresholdHeadroomError("diagnostic contract drifted")
    return value


def _metric_subset(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "accepted_labeled",
            "correct_rescues",
            "harmful_swaps",
            "precision",
            "conservative_utility",
            "background_accepted",
            "total_selected",
        )
    }


def _threshold_record(
    threshold: float, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "threshold": float(threshold) if math.isfinite(threshold) else None,
        "metrics": _metric_subset(metrics),
    }


def calibration_floor_passes(
    aggregate: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> bool:
    """Apply the preregistered diagnostic plausibility rule."""

    return (
        float(aggregate["precision"])
        >= float(gate["minimum_aggregate_held_precision"])
        and int(aggregate["accepted_labeled"])
        >= int(gate["minimum_aggregate_held_accepted"])
        and int(aggregate["conservative_utility"])
        >= int(gate["minimum_conservative_utility"])
        and all(
            float(fold["precision"])
            >= float(gate["minimum_each_fold_held_precision"])
            for fold in folds
        )
    )


def _validate_source(
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
    list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]],
    Path,
]:
    source = contract["source"]
    gate_contract_path = _root_file(source["gate_contract"], "gate contract")
    gate_report_path = _root_file(
        str(source["gate_output"]) + "/report.json", "gate report"
    )
    prepare_path = _root_file(source["prepare_report"], "prepare report")
    for path, expected, label in (
        (
            gate_contract_path,
            source["gate_contract_sha256"],
            "gate contract",
        ),
        (gate_report_path, source["gate_report_sha256"], "gate report"),
        (prepare_path, source["prepare_report_sha256"], "prepare report"),
    ):
        if sweep._sha256(path) != expected:
            raise ThresholdHeadroomError(f"{label} SHA-256 mismatch")
    gate_contract = point_gate.load_contract(
        gate_contract_path, source["gate_contract_sha256"]
    )
    gate_report = _load_json(gate_report_path, "gate report")
    prepare = _load_json(prepare_path, "prepare report")
    if (
        gate_report.get("schema") != point_gate.SCHEMA
        or gate_report.get("status") != "PARK"
        or gate_report.get("decision", {}).get("reason")
        != "held proposal precision/support prerequisite failed"
        or gate_report.get("scope", {}).get("validation12_access") is not False
        or gate_report.get("scope", {}).get("cv57_access") is not False
        or prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
    ):
        raise ThresholdHeadroomError("source evidence is inadmissible")
    scope = prepare.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or scope.get("audit_used") is not False
        or scope.get("held_out_used") is not False
        or scope.get("test_used") is not False
        or len(scope.get("train_video_ids", ())) != 105
        or len(scope.get("validation_video_ids", ())) != 12
    ):
        raise ThresholdHeadroomError("prepared scope is not sealed fit-only")
    validation = {str(value) for value in scope["validation_video_ids"]}
    eligible = tuple(
        str(value)
        for value in scope["train_video_ids"]
        if str(value).startswith("train:6bba_")
    )
    if (
        len(eligible) != int(contract["scope"]["fit_train_video_count"])
        or set(eligible).intersection(validation)
    ):
        raise ThresholdHeadroomError("eligible fit scope drifted")
    partitions = point_gate.nested_partitions(
        eligible, int(contract["scope"]["outer_fold_count"])
    )
    return (
        gate_contract,
        prepare,
        eligible,
        partitions,
        gate_report_path.parent,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Replay held OOF scores and measure threshold-only diagnostic headroom."""

    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise ThresholdHeadroomError(
            "diagnostic must run inside the local training guard"
        )
    if args.output.exists():
        raise ThresholdHeadroomError(f"refusing to overwrite: {args.output}")
    contract = load_contract(args.contract, args.contract_sha256)
    gate_contract, prepare, eligible, partitions, gate_output = (
        _validate_source(contract)
    )
    model_dir = gate_output / "fold_models"
    point_gate.validate_fold_directory(model_dir, len(partitions))
    code_identity = point_gate.fold_code_identity()
    architecture = gate_contract["candidate"]["architecture"]
    optimization = gate_contract["candidate"]["optimization"]
    floors = [
        float(value)
        for value in contract["diagnostic"][
            "calibration_precision_floor_grid"
        ]
    ]
    minimum = int(
        contract["diagnostic"]["minimum_calibration_accepted"]
    )
    floor_records: dict[float, list[dict[str, Any]]] = {
        floor: [] for floor in floors
    }
    oracle_records = []
    banked_records = []
    started = time.perf_counter()
    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(partitions):
        restored = point_gate.load_fold_checkpoint(
            model_dir=model_dir,
            held_fold=held_fold,
            fit_ids=fit_ids,
            calibration_ids=calibration_ids,
            held_ids=held_ids,
            architecture=architecture,
            seed=int(optimization["seed_base"]) + held_fold,
            contract_sha256=contract["source"]["gate_contract_sha256"],
            code_identity=code_identity,
            fold_count=len(partitions),
        )
        if restored is None:
            raise ThresholdHeadroomError(
                f"fold {held_fold} checkpoint is missing"
            )
        model, mean, scale, banked_threshold, fold_record = restored
        calibration_windows, calibration_selected = (
            point_gate._load_partition_windows(
                prepare,
                cache_root=args.cache,
                identities=calibration_ids,
            )
        )
        calibration_scored = score_windows(
            model,
            calibration_windows,
            selected_edges=calibration_selected,
            mean=mean,
            scale=scale,
            device=torch.device("cpu"),
            batch_size=int(optimization["batch_size_windows"]),
        )
        del calibration_windows, calibration_selected
        gc.collect()
        held_windows, held_selected = point_gate._load_partition_windows(
            prepare, cache_root=args.cache, identities=held_ids
        )
        held_scored = score_windows(
            model,
            held_windows,
            selected_edges=held_selected,
            mean=mean,
            scale=scale,
            device=torch.device("cpu"),
            batch_size=int(optimization["batch_size_windows"]),
        )
        del held_windows, held_selected
        gc.collect()
        banked_metrics = _metrics(held_scored, banked_threshold)
        if _metric_subset(banked_metrics) != dict(fold_record["held"]):
            raise ThresholdHeadroomError(
                f"fold {held_fold} banked held metrics did not reproduce"
            )
        banked_records.append(
            {
                "held_fold": held_fold,
                **_threshold_record(banked_threshold, banked_metrics),
            }
        )
        for floor in floors:
            threshold, calibration_metrics = _select(
                calibration_scored,
                minimum_accepted=minimum,
                minimum_precision=floor,
            )
            held_metrics = _metrics(held_scored, threshold)
            floor_records[floor].append(
                {
                    "held_fold": held_fold,
                    "calibration_precision_floor": floor,
                    "calibration": _metric_subset(calibration_metrics),
                    **_threshold_record(threshold, held_metrics),
                }
            )
        oracle_threshold, oracle_metrics = _select(
            held_scored,
            minimum_accepted=int(
                contract["diagnostic"]["oracle_minimum_held_accepted"]
            ),
            minimum_precision=float(
                contract["diagnostic"]["oracle_minimum_held_precision"]
            ),
        )
        oracle_records.append(
            {
                "held_fold": held_fold,
                **_threshold_record(oracle_threshold, oracle_metrics),
            }
        )
        del calibration_scored, held_scored, model
        gc.collect()
        print(
            f"threshold headroom fold {held_fold + 1:02d}/"
            f"{len(partitions)} reproduced",
            flush=True,
        )

    floor_summaries = []
    qualifying = []
    for floor in floors:
        held_metrics = [
            dict(record["metrics"]) for record in floor_records[floor]
        ]
        aggregate = point_gate.aggregate_metrics(held_metrics)
        passes = calibration_floor_passes(
            aggregate, held_metrics, contract["plausibility_gate"]
        )
        record = {
            "calibration_precision_floor": floor,
            "aggregate_held": aggregate,
            "folds": floor_records[floor],
            "plausibility_gate_passed": passes,
        }
        floor_summaries.append(record)
        if passes:
            qualifying.append(record)
    qualifying.sort(
        key=lambda value: (
            int(value["aggregate_held"]["conservative_utility"]),
            int(value["aggregate_held"]["correct_rescues"]),
            float(value["calibration_precision_floor"]),
        ),
        reverse=True,
    )
    oracle_aggregate = point_gate.aggregate_metrics(
        [dict(record["metrics"]) for record in oracle_records]
    )
    verdict = (
        contract["decision"]["pass_label"]
        if qualifying
        else contract["decision"]["fail_label"]
    )
    result = {
        "schema": SCHEMA,
        "status": verdict,
        "scope": {
            "family": "6bba",
            "source_partition": "fit-train105",
            "video_count": len(eligible),
            "validation12_access": False,
            "cv57_access": False,
            "promotion_authorized": False,
        },
        "banked_reproduction": {
            "folds": banked_records,
            "aggregate": point_gate.aggregate_metrics(
                [dict(record["metrics"]) for record in banked_records]
            ),
        },
        "shared_calibration_floor_grid": floor_summaries,
        "qualifying_shared_floors": [
            {
                "calibration_precision_floor": value[
                    "calibration_precision_floor"
                ],
                "aggregate_held": value["aggregate_held"],
            }
            for value in qualifying
        ],
        "oracle_upper_bound": {
            "non_authoritative": True,
            "folds": oracle_records,
            "aggregate": oracle_aggregate,
        },
        "decision": {
            "status": verdict,
            "promotion_authorized": False,
            "graph_replay_authorized": False,
            "cv57_authorized": False,
            "next_action": (
                "preregister a new cross-fitted calibration candidate"
                if qualifying
                else "close threshold-only repair and select a new representation"
            ),
        },
        "provenance": {
            "contract": str(args.contract),
            "contract_sha256": args.contract_sha256,
            "gate_report_sha256": contract["source"][
                "gate_report_sha256"
            ],
            "fold_code_identity": code_identity,
        },
        "execution": {
            "device": "cpu",
            "guard_required": True,
            "elapsed_seconds": time.perf_counter() - started,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument(
        "--cache", type=Path, default=point_gate.DEFAULT_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except ThresholdHeadroomError as exc:
        print(f"point-context threshold headroom failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "qualifying_shared_floors": result[
                    "qualifying_shared_floors"
                ],
                "oracle_aggregate": result["oracle_upper_bound"][
                    "aggregate"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
