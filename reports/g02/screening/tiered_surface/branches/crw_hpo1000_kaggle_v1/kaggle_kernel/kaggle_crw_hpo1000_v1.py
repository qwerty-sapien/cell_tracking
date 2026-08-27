#!/usr/bin/env python3
"""Evaluate the frozen 1,000-tuple CRW selector bank; emit no competition artifact."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any


BOOT_STARTED = time.monotonic()
DATASET_ROOT = Path("/kaggle/input/datasets/onebitbrain/g02-crw-hpo1000-carrier-v1")
WORK_ROOT = Path("/kaggle/working/crw_hpo1000_v1")
WORK_ROOT.mkdir(parents=True, exist_ok=False)
RUN_LOG = WORK_ROOT / "RUN_LOG.jsonl"

MANIFEST_SHA256 = "3fce7917c2e4eb713b94cb5b936599806901813039c7e58214bc64b065c416a9"
CARRIER_SHA256 = "c46afff71557c657d97814baa3ba452101502103f56ade550909bbd158b0b02b"
PARAMETER_BANK_SHA256 = "21ff9a481d0dc01f9bd6063a7595ea3e56a0ba112b11ffaeb0d5a7dbf49b6997"
PREREGISTRATION_SHA256 = "5ed8a44693ab9921a5bab4240a5aa570a8e9dc31e92ccaf18884df5ab7c2dd55"
EXPECTED_CANDIDATES = 92_905
EXPECTED_COMPONENTS = 40_894
EXPECTED_PARAMETERS = 1_000
EXPECTED_REQUEST_COUNT = 1_000
EXPECTED_REQUEST_FILE_COUNT = 4
EXPECTED_POPULATION_FILE_COUNT = 4
EXPECTED_WHEEL_COUNT = 0
QUANTIZATION = 1_000_000


class KernelError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log_lookup(stage: str, event: str, **fields: Any) -> None:
    record = {
        "schema": "celltrack-alpha.run-log.v1",
        "elapsed_seconds": time.monotonic() - BOOT_STARTED,
        "stage": stage,
        "event": event,
        **fields,
    }
    with RUN_LOG.open("ab") as handle:
        handle.write(canonical(record))
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(record, sort_keys=True), flush=True)


def fail(stage: str, error: BaseException, **state: Any) -> None:
    log_lookup(
        stage,
        "_run_failure",
        error_type=type(error).__name__,
        error=str(error),
        traceback=traceback.format_exc(),
        state=state,
    )


def unique_basename(name: str) -> Path:
    if Path(name).name != name:
        raise KernelError("declared asset is not a flat basename")
    path = DATASET_ROOT / name
    if not path.is_file():
        raise KernelError(f"declared asset is absent: {name}")
    return path


def run_step0() -> dict[str, Any]:
    try:
        log_lookup(
            "step0",
            "inventory_input",
            root=str(DATASET_ROOT),
            exists=DATASET_ROOT.is_dir(),
        )
        manifest_path = unique_basename("CARRIER_MANIFEST.json")
        if sha256(manifest_path) != MANIFEST_SHA256:
            raise KernelError("carrier manifest hash drifted")
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        if (
            manifest.get("schema") != "celltrack-alpha.crw-hpo1000-private-carrier.v1"
            or manifest.get("status") != "IMMUTABLE_UPLOAD_READY"
            or manifest.get("dataset_id") != "onebitbrain/g02-crw-hpo1000-carrier-v1"
            or manifest.get("dataset_version") != 1
            or manifest.get("population", {}).get("candidates") != EXPECTED_CANDIDATES
            or manifest.get("population", {}).get("components") != EXPECTED_COMPONENTS
            or manifest.get("population", {}).get("folds") != 5
            or not all(manifest.get("forbidden_payload_absent", {}).values())
        ):
            raise KernelError("carrier manifest semantic drift")
        records = {
            "carrier": ("crw_hpo1000_carrier.npz", CARRIER_SHA256),
            "parameters": ("PARAMETER_BANK.json", PARAMETER_BANK_SHA256),
            "preregistration": ("PREREGISTRATION.json", PREREGISTRATION_SHA256),
        }
        paths: dict[str, str] = {}
        for label, (name, expected) in records.items():
            path = unique_basename(name)
            if sha256(path) != expected:
                raise KernelError(f"{label} hash drifted")
            paths[label] = str(path)
        parameter_bank = json.loads(Path(paths["parameters"]).read_text(encoding="ascii"))
        if (
            parameter_bank.get("schema") != "celltrack-alpha.crw-hpo1000-parameter-bank.v1"
            or parameter_bank.get("status") != "FROZEN_BEFORE_CARRIER_SUPERVISION"
            or parameter_bank.get("total_unique_combinations") != EXPECTED_PARAMETERS
            or len(parameter_bank.get("trials", [])) != EXPECTED_PARAMETERS
            or len({row.get("tuple_id") for row in parameter_bank.get("trials", [])}) != EXPECTED_PARAMETERS
        ):
            raise KernelError("parameter-bank semantic drift")
        result = {
            "schema": "celltrack-alpha.crw-hpo1000-step0.v1",
            "status": "PASS",
            "dataset_source": "onebitbrain/g02-crw-hpo1000-carrier-v1",
            "dataset_version": 1,
            "manifest_sha256": MANIFEST_SHA256,
            "carrier_sha256": CARRIER_SHA256,
            "parameter_bank_sha256": PARAMETER_BANK_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "candidates": EXPECTED_CANDIDATES,
            "components": EXPECTED_COMPONENTS,
            "parameters": EXPECTED_PARAMETERS,
            "request_count": EXPECTED_REQUEST_COUNT,
            "request_file_count": EXPECTED_REQUEST_FILE_COUNT,
            "population_file_count": EXPECTED_POPULATION_FILE_COUNT,
            "wheel_count": EXPECTED_WHEEL_COUNT,
            "enable_gpu": False,
            "competition_path_visits": 0,
            "heavy_work_started": False,
            "paths": paths,
        }
        log_lookup("step0", "stage_end", **result)
        return result
    except BaseException as error:
        fail("step0", error, heavy_work_started=False)
        raise


STEP0 = run_step0()


import numpy as np  # noqa: E402


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(canonical(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_carrier() -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(unique_basename("CARRIER_MANIFEST.json").read_text(encoding="ascii"))
    parameters = json.loads(Path(STEP0["paths"]["parameters"]).read_text(encoding="ascii"))
    with np.load(Path(STEP0["paths"]["carrier"]), allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if set(arrays) != set(manifest["arrays"]):
        raise KernelError("carrier array names drifted")
    for name, array in arrays.items():
        record = manifest["arrays"][name]
        if (
            str(array.dtype) != record["dtype"]
            or list(array.shape) != record["shape"]
            or array_sha256(array) != record["sha256"]
        ):
            raise KernelError(f"carrier array drift: {name}")
    if (
        arrays["features"].shape != (EXPECTED_CANDIDATES, 18)
        or len(arrays["component_ptr"]) != EXPECTED_COMPONENTS + 1
        or int(arrays["component_ptr"][-1]) != EXPECTED_CANDIDATES
    ):
        raise KernelError("carrier shape drift")
    return arrays, manifest, parameters


def fit_specs(features: np.ndarray, target: np.ndarray, fold: np.ndarray) -> list[dict[str, np.ndarray]]:
    specs: list[dict[str, np.ndarray]] = []
    for held in range(5):
        train = fold != held
        valid = fold == held
        if not train.any() or not valid.any() or len(set(target[train].tolist())) != 2:
            raise KernelError(f"fold {held} lacks train/validation class support")
        train_x = features[train]
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale == 0.0] = 1.0
        z = (train_x - mean) / scale
        design = np.column_stack((np.ones(len(z), dtype=np.float64), z))
        positive = target[train] > 0
        negative = ~positive
        n = float(len(design))
        valid_z = (features[valid] - mean) / scale
        specs.append(
            {
                "g_positive": design[positive].T @ design[positive] / n,
                "g_negative": design[negative].T @ design[negative] / n,
                "s_positive": design[positive].sum(axis=0) / n,
                "s_negative": design[negative].sum(axis=0) / n,
                "valid_design": np.column_stack((np.ones(int(valid.sum()), dtype=np.float64), valid_z)),
                "valid_indices": np.flatnonzero(valid),
            }
        )
    return specs


def predict_oof(specs: list[dict[str, np.ndarray]], count: int, params: dict[str, float]) -> np.ndarray:
    probabilities = np.full(count, np.nan, dtype=np.float64)
    ridge = float(params["ridge_l2"])
    positive_weight = float(params["positive_weight"])
    temperature = float(params["temperature"])
    penalty = np.eye(19, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for spec in specs:
        gram = spec["g_negative"] + positive_weight * spec["g_positive"] + penalty
        rhs = positive_weight * spec["s_positive"] - spec["s_negative"]
        coefficients = np.linalg.solve(gram, rhs)
        raw = spec["valid_design"] @ coefficients
        clipped = np.clip(raw / temperature, -40.0, 40.0)
        probabilities[spec["valid_indices"]] = 1.0 / (1.0 + np.exp(-clipped))
    if not np.isfinite(probabilities).all():
        raise KernelError("OOF prediction coverage is incomplete")
    return probabilities


def feasible_matrix(slot_masks: np.ndarray) -> np.ndarray:
    n = len(slot_masks)
    rows: list[list[int]] = []
    for canonical_mask in range((1 << n) - 1, -1, -1):
        occupied = 0
        row = [0] * n
        valid = True
        for position in range(n):
            bit = 1 << (n - 1 - position)
            if canonical_mask & bit:
                slots = int(slot_masks[position])
                if occupied & slots:
                    valid = False
                    break
                occupied |= slots
                row[position] = 1
        if valid:
            rows.append(row)
    return np.asarray(rows, dtype=np.int8)


def build_solver(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    ptr = arrays["component_ptr"]
    candidate_index = arrays["component_candidate_index"]
    slot_mask = arrays["component_slot_mask"]
    pair_rows: list[tuple[int, int]] = []
    complex_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for component in range(len(ptr) - 1):
        start, stop = int(ptr[component]), int(ptr[component + 1])
        indices = candidate_index[start:stop]
        slots = slot_mask[start:stop]
        if len(indices) == 2:
            if not int(slots[0]) & int(slots[1]):
                raise KernelError("size-two conflict component is disjoint")
            pair_rows.append((int(indices[0]), int(indices[1])))
        else:
            complex_rows.append((np.asarray(indices, dtype=np.int64), feasible_matrix(slots)))
    if len(pair_rows) != 36_523 or len(complex_rows) != EXPECTED_COMPONENTS - 36_523:
        raise KernelError("solver component histogram drift")
    return {"pairs": np.asarray(pair_rows, dtype=np.int64), "complex": complex_rows}


def exact_select(probabilities: np.ndarray, threshold: float, solver: dict[str, Any]) -> np.ndarray:
    weights = np.rint((probabilities - threshold) * QUANTIZATION).astype(np.int64)
    chosen: list[int] = []
    pairs = solver["pairs"]
    pair_weights = weights[pairs]
    choose_left = (pair_weights[:, 0] > 0) & (
        (pair_weights[:, 1] <= 0) | (pair_weights[:, 0] >= pair_weights[:, 1])
    )
    choose_right = (pair_weights[:, 1] > 0) & ~choose_left
    chosen.extend(pairs[choose_left, 0].tolist())
    chosen.extend(pairs[choose_right, 1].tolist())
    for indices, feasible in solver["complex"]:
        local = weights[indices]
        ineligible = local <= 0
        valid = np.ones(len(feasible), dtype=bool)
        if ineligible.any():
            valid = feasible[:, ineligible].sum(axis=1) == 0
        scores = feasible @ local
        best_value = scores[valid].max()
        # Feasible rows are in descending canonical-bit order, so first max is the tie-break.
        row = int(np.flatnonzero(valid & (scores == best_value))[0])
        chosen.extend(indices[feasible[row] > 0].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def trial_metrics(
    trial: dict[str, Any], arrays: dict[str, np.ndarray], specs: list[dict[str, np.ndarray]], solver: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    params = {name: float(value) for name, value in trial["params"].items()}
    probabilities = predict_oof(specs, len(arrays["features"]), params)
    selected = exact_select(probabilities, params["decision_probability"], solver)
    totals = arrays["attribution"][selected].sum(axis=0, dtype=np.int64) if len(selected) else np.zeros(5, dtype=np.int64)
    video_net = np.bincount(
        arrays["identity_code"][selected].astype(np.int64),
        weights=arrays["attribution"][selected, 0].astype(np.float64),
        minlength=36,
    ).astype(np.int64)
    result = {
        "trial_id": int(trial["trial_id"]),
        "tuple_id": str(trial["tuple_id"]),
        "stratum": str(trial["stratum"]),
        "params": params,
        "objective_net": int(totals[0]),
        "attribution": {
            "net": int(totals[0]),
            "tp_added": int(totals[1]),
            "fp_added": int(totals[2]),
            "tp_removed": int(totals[3]),
            "fp_removed": int(totals[4]),
        },
        "selected_transactions": int(len(selected)),
        "positive_videos": int((video_net > 0).sum()),
        "worst_eight_video_net_sum": int(np.sort(video_net)[:8].sum()),
    }
    return result, selected


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(np.float64)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.sqrt((left_rank @ left_rank) * (right_rank @ right_rank)))
    return 0.0 if denominator == 0.0 else float((left_rank @ right_rank) / denominator)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    objective = np.asarray([row["objective_net"] for row in results], dtype=np.float64)
    quantiles = np.quantile(objective, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    median = float(quantiles[3])
    normalizer = max(1.0, abs(median))
    normalized_iqr = float((quantiles[4] - quantiles[2]) / normalizer)
    top_decile_lift = float((np.quantile(objective, 0.9) - median) / normalizer)
    by_stratum: dict[str, Any] = {}
    for stratum in sorted({row["stratum"] for row in results}):
        values = np.asarray([row["objective_net"] for row in results if row["stratum"] == stratum], dtype=np.float64)
        by_stratum[stratum] = {
            "count": len(values),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        }
    correlations = {}
    for name in ("ridge_l2", "positive_weight", "temperature", "decision_probability"):
        values = np.asarray([float(row["params"][name]) for row in results], dtype=np.float64)
        correlations[name] = rank_correlation(values, objective)
    return {
        "objective_quantiles": {
            name: float(value)
            for name, value in zip(("min", "p05", "p25", "median", "p75", "p95", "max"), quantiles, strict=True)
        },
        "normalized_iqr": normalized_iqr,
        "normalized_top_decile_lift": top_decile_lift,
        "search_sensitive": normalized_iqr >= 0.25 or top_decile_lift >= 0.25,
        "low_dispersion": normalized_iqr < 0.10 and top_decile_lift < 0.10,
        "by_stratum": by_stratum,
        "parameter_objective_rank_correlations": correlations,
        "selected_support": {
            "minimum": min(row["selected_transactions"] for row in results),
            "median": float(np.median([row["selected_transactions"] for row in results])),
            "maximum": max(row["selected_transactions"] for row in results),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


def run() -> None:
    requested = int(os.environ.get("CRW_HPO_TRIALS", str(EXPECTED_PARAMETERS)))
    expected = int(os.environ.get("CRW_EXPECTED_TRIALS", str(requested)))
    if requested != expected or requested <= 0 or requested > EXPECTED_PARAMETERS:
        raise KernelError("trial execution-control values are invalid")
    log_lookup("hpo", "heavy_work_start", requested_trials=requested)
    arrays, manifest, parameters = load_carrier()
    log_lookup(
        "hpo",
        "preprocess_counts",
        candidates=len(arrays["features"]),
        components=len(arrays["component_ptr"]) - 1,
        parameter_bank_size=len(parameters["trials"]),
    )
    specs = fit_specs(arrays["features"], arrays["positive_target"], arrays["fold"])
    solver = build_solver(arrays)
    if requested == EXPECTED_PARAMETERS:
        trial_ids = np.arange(EXPECTED_PARAMETERS, dtype=np.int64)
    else:
        trial_ids = np.rint(np.linspace(0, EXPECTED_PARAMETERS - 1, requested)).astype(np.int64)
        if len(set(trial_ids.tolist())) != requested:
            raise KernelError("scenario trial sampling is not unique")
    results: list[dict[str, Any]] = []
    selected_by_trial: dict[int, np.ndarray] = {}
    for ordinal, trial_id in enumerate(trial_ids.tolist(), start=1):
        result, selected = trial_metrics(parameters["trials"][trial_id], arrays, specs, solver)
        results.append(result)
        selected_by_trial[trial_id] = selected
        if ordinal == 1 or ordinal == requested or ordinal % 25 == 0:
            log_lookup("hpo", "trial_progress", completed=ordinal, total=requested, trial_id=trial_id)
    winner = max(
        results,
        key=lambda row: (
            row["objective_net"],
            row["positive_videos"],
            row["worst_eight_video_net_sum"],
            -row["selected_transactions"],
            -row["trial_id"],
        ),
    )
    selected = selected_by_trial[int(winner["trial_id"])]
    selected_ids = [arrays["candidate_id"][index].decode("ascii") for index in selected]
    summary = summarize(results)
    trial_results_path = WORK_ROOT / "TRIAL_RESULTS.jsonl"
    with trial_results_path.open("wb") as handle:
        for result in results:
            handle.write(canonical(result))
    winner_payload = {
        "schema": "celltrack-alpha.crw-hpo1000-winner.v1",
        "status": "FROZEN_FOR_ONE_LOCAL_GRAPH_SCORE",
        "winner": winner,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_ids_sha256": hashlib.sha256(canonical(selected_ids)).hexdigest(),
        "graph_score_calls_authorized": 1,
        "cv57_access": False,
        "six_bba_access": False,
        "competition_submission": False,
    }
    write_json(WORK_ROOT / "WINNER.json", winner_payload)
    write_json(WORK_ROOT / "DISPERSION_SUMMARY.json", summary)
    receipt = {
        "schema": "celltrack-alpha.crw-hpo1000-semantic-receipt.v1",
        "status": "COMPLETE",
        "completed_workload_units": len(results),
        "expected_workload_units": requested,
        "full_parameter_bank_size": EXPECTED_PARAMETERS,
        "evaluated_trial_ids_sha256": hashlib.sha256(canonical(trial_ids.tolist())).hexdigest(),
        "unique_tuple_count": len({row["tuple_id"] for row in results}),
        "winner_trial_id": winner["trial_id"],
        "winner_tuple_id": winner["tuple_id"],
        "winner_objective_net": winner["objective_net"],
        "winner_selected_transactions": winner["selected_transactions"],
        "dispersion": summary,
        "solver": {
            "method": "EXACT_FEASIBLE_SUBSET_COMPONENT_DECOMPOSITION",
            "pair_components": len(solver["pairs"]),
            "complex_components": len(solver["complex"]),
            "components": EXPECTED_COMPONENTS,
            "maximum_variables": manifest["population"]["maximum_variables"],
            "maximum_slots": manifest["population"]["maximum_slots"],
            "never_truncated": True,
            "gurobi_runtime_dependency": False,
        },
        "bindings": {
            "manifest_sha256": MANIFEST_SHA256,
            "carrier_sha256": CARRIER_SHA256,
            "parameter_bank_sha256": PARAMETER_BANK_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
        "scientific_boundaries": {
            "objective_is_official_score": False,
            "official_scorer_accessed": False,
            "cv57_accessed": False,
            "six_bba_accessed": False,
            "competition_path_visits": 0,
            "submission_artifact_written": False,
        },
    }
    write_json(WORK_ROOT / "SEMANTIC_RECEIPT.json", receipt)
    log_lookup("complete", "stage_end", status="COMPLETE", completed_workload_units=len(results))


try:
    run()
except BaseException as error:
    fail("runtime", error, step0_status=STEP0.get("status"))
    raise
