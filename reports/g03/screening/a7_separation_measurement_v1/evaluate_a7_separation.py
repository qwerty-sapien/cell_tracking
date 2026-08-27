#!/usr/bin/env python3
"""Run the preregistered A7 B0 -> A1 -> A0 separation measurement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
PREREG = HERE / "PREREGISTRATION.json"
A0_SEED = "A7-PHASE-SEPARATION-A0-V1"
EXPECTED_ACTIONS = 45_961
EXPECTED_POSITIVES = 134
EXPECTED_VIDEOS = 36


class A7SeparationError(RuntimeError):
    """The frozen A7 separation contract could not be preserved."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise A7SeparationError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise A7SeparationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def authenticate_prereg() -> dict[str, Any]:
    prereg = load_json(PREREG)
    if prereg.get("status") != "FROZEN_BEFORE_A7_SEPARATION_SCORE":
        raise A7SeparationError("A7 separation preregistration is not frozen")
    for name, pin in prereg["frozen_inputs"].items():
        path = ROOT / str(pin["path"])
        observed = sha256(path)
        if observed != str(pin["sha256"]):
            raise A7SeparationError(
                f"{name} SHA-256 drift: expected {pin['sha256']}, got {observed}"
            )
    return prereg


def load_evaluator(prereg: dict[str, Any]) -> Any:
    path = ROOT / prereg["frozen_inputs"]["frozen_separation_evaluator"]["path"]
    spec = importlib.util.spec_from_file_location("frozen_a7_separation_evaluator", path)
    if spec is None or spec.loader is None:
        raise A7SeparationError("cannot load frozen separation evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def population_root(prereg: dict[str, Any]) -> Path:
    manifest_path = ROOT / prereg["frozen_inputs"]["population_manifest"]["path"]
    return manifest_path.parent


def a0_permutation(
    supplier: np.ndarray,
    *,
    identity: str,
    source_t: np.ndarray,
    source_id: np.ndarray,
    target_id: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Deterministically derange the supplier vector within video/frame."""

    a1 = np.asarray(supplier, dtype=np.float64)
    if a1.ndim != 2 or a1.shape[1] != 1 or not np.isfinite(a1).all():
        raise A7SeparationError("finite one-column A7 supplier required")
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
                    f"{A0_SEED}|occupied_swap|{identity}|{frame}|"
                    f"{int(source_id[index])}|{int(target_id[index])}|{index}"
                ).encode("utf-8")
            ).digest(),
        )
        donors = ordered[1:] + ordered[:1]
        ordered_array = np.asarray(ordered, dtype=np.int64)
        donor_array = np.asarray(donors, dtype=np.int64)
        a0[ordered_array] = a1[donor_array]
        if not np.array_equal(
            np.sort(a0[ordered_array, 0]), np.sort(a1[ordered_array, 0])
        ):
            raise A7SeparationError("A0 stratum is not an exact permutation")
        moved += len(ordered)
    if not np.isfinite(a0).all():
        raise A7SeparationError("nonfinite A0 supplier")
    return a0, {"moved_rows": moved, "singleton_rows": singleton}


def feature_records(prereg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population_manifest = load_json(
        ROOT / prereg["frozen_inputs"]["population_manifest"]["path"]
    )
    feature_manifest = load_json(
        ROOT / prereg["frozen_inputs"]["a7_feature_manifest"]["path"]
    )
    population_rows = population_manifest["classes"]["occupied_swap"]["videos"]
    feature_rows = feature_manifest["shards"]
    if len(population_rows) != EXPECTED_VIDEOS or len(feature_rows) != EXPECTED_VIDEOS:
        raise A7SeparationError("A7 video coverage drift")
    feature_by_identity = {str(row["identity"]): row for row in feature_rows}
    if len(feature_by_identity) != EXPECTED_VIDEOS:
        raise A7SeparationError("duplicate A7 feature identity")
    records = []
    for row in population_rows:
        identity = str(row["identity"])
        feature = feature_by_identity.get(identity)
        if feature is None:
            raise A7SeparationError(f"missing A7 feature shard: {identity}")
        if int(row["count"]) != int(feature["actions"]):
            raise A7SeparationError(f"population/feature count mismatch: {identity}")
        population_path = ROOT / str(row["population_path"])
        feature_path = ROOT / str(feature["path"])
        if sha256(population_path) != str(row["population_sha256"]):
            raise A7SeparationError(f"population shard drift: {identity}")
        if sha256(feature_path) != str(feature["sha256"]):
            raise A7SeparationError(f"feature shard drift: {identity}")
        records.append(
            {
                "identity": identity,
                "population_path": population_path,
                "feature_path": feature_path,
                "count": int(row["count"]),
            }
        )
    if sum(row["count"] for row in records) != EXPECTED_ACTIONS:
        raise A7SeparationError("A7 action cardinality drift")
    return records, feature_manifest


def load_supplier(
    prereg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records, feature_manifest = feature_records(prereg)
    by_stem: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    global_indices = []
    moved_rows = 0
    singleton_rows = 0
    fold_counts = {str(index): 0 for index in range(5)}
    for row in records:
        with np.load(row["population_path"], allow_pickle=False) as population:
            source_id = np.asarray(population["source_id"], dtype=np.int64)
            target_id = np.asarray(population["target_id"], dtype=np.int64)
            source_t = np.asarray(population["source_t"], dtype=np.int64)
            folds = np.asarray(population["fold"], dtype=np.int64)
        with np.load(row["feature_path"], allow_pickle=False) as feature:
            supplier = np.asarray(feature["action_margin"], dtype=np.float64)[:, None]
            global_index = np.asarray(feature["global_index"], dtype=np.int64)
        count = row["count"]
        if any(len(value) != count for value in (source_id, target_id, source_t, folds, supplier, global_index)):
            raise A7SeparationError(f"A7 row alignment drift: {row['identity']}")
        if not set(int(value) for value in np.unique(folds)).issubset(set(range(5))):
            raise A7SeparationError("fold value outside frozen five-fold range")
        a0, stats = a0_permutation(
            supplier,
            identity=row["identity"],
            source_t=source_t,
            source_id=source_id,
            target_id=target_id,
        )
        stem = row["population_path"].stem
        if stem in by_stem:
            raise A7SeparationError(f"duplicate population stem: {stem}")
        by_stem[stem] = (supplier, a0)
        global_indices.append(global_index)
        moved_rows += stats["moved_rows"]
        singleton_rows += stats["singleton_rows"]
        for fold in range(5):
            fold_counts[str(fold)] += int((folds == fold).sum())
    if not np.array_equal(
        np.sort(np.concatenate(global_indices)), np.arange(EXPECTED_ACTIONS, dtype=np.int64)
    ):
        raise A7SeparationError("A7 global-index coverage drift")

    population_paths = sorted(
        population_root(prereg).joinpath("population", "occupied_swap").glob("*.npz")
    )
    if len(population_paths) != EXPECTED_VIDEOS:
        raise A7SeparationError("frozen evaluator population coverage drift")
    supplier = np.concatenate([by_stem[path.stem][0] for path in population_paths], axis=0)
    a0 = np.concatenate([by_stem[path.stem][1] for path in population_paths], axis=0)
    if supplier.shape != (EXPECTED_ACTIONS, 1) or a0.shape != supplier.shape:
        raise A7SeparationError("assembled A7 supplier shape drift")
    receipt = {
        "actions": EXPECTED_ACTIONS,
        "videos": EXPECTED_VIDEOS,
        "feature_manifest_aggregate_sha256": feature_manifest["aggregate_sha256"],
        "supplier_columns": ["action_margin"],
        "a0_seed": A0_SEED,
        "a0_moved_rows": moved_rows,
        "a0_singleton_rows": singleton_rows,
        "fold_action_counts": fold_counts,
        "oracle_arrays_read": False,
    }
    return supplier, a0, receipt


def preflight(output: Path) -> None:
    started = time.perf_counter()
    prereg = authenticate_prereg()
    supplier, a0, receipt = load_supplier(prereg)
    if np.array_equal(supplier, a0):
        raise A7SeparationError("A0 is globally identical to A1")
    write_json(
        output,
        {
            "schema": "celltrack-alpha.g03-a7-separation-pre-score-preflight.v1",
            "status": "PASS_TARGET_BLIND_READY_FOR_B0",
            "preregistration": {"path": str(PREREG.relative_to(ROOT)), "sha256": sha256(PREREG)},
            "receipt": receipt,
            "elapsed_seconds": time.perf_counter() - started,
            "separation_score_opened": False,
            "graph_opened": False,
        },
    )


def run_b0(output: Path) -> None:
    started = time.perf_counter()
    prereg = authenticate_prereg()
    evaluator = load_evaluator(prereg)
    data = evaluator.load_class(population_root(prereg), None, "occupied_swap")
    observed = evaluator.evaluate_b0_class(data, "occupied_swap")
    if observed["actions"] != EXPECTED_ACTIONS or observed["beneficial"] != EXPECTED_POSITIVES:
        raise A7SeparationError("B0 population/positive identity drift")
    inherited_path = ROOT / prereg["frozen_inputs"]["inherited_b0_seal"]["path"]
    inherited = load_json(inherited_path)["results"]["occupied_swap"]
    if observed != inherited:
        raise A7SeparationError("fresh B0 differs from inherited pre-A7 B0 seal")
    write_json(
        output,
        {
            "schema": "celltrack-alpha.g03-a7-separation-b0-sealed.v1",
            "status": "SEALED_BEFORE_A1_A0",
            "preregistration_sha256": sha256(PREREG),
            "evaluator_sha256": prereg["frozen_inputs"]["frozen_separation_evaluator"]["sha256"],
            "inherited_b0_sha256": prereg["frozen_inputs"]["inherited_b0_seal"]["sha256"],
            "exact_match_to_inherited_b0": True,
            "execution_order": ["B0_SEALED"],
            "result": observed,
            "elapsed_seconds": time.perf_counter() - started,
            "a1_or_a0_opened": False,
        },
    )


def run_score(output: Path, b0_path: Path, authorization_path: Path) -> None:
    started = time.perf_counter()
    prereg = authenticate_prereg()
    authorization = load_json(authorization_path)
    if authorization.get("status") != "AUTHORIZED_AFTER_B0_SEAL":
        raise A7SeparationError("A1/A0 score authorization is not valid")
    if authorization.get("b0_sha256") != sha256(b0_path):
        raise A7SeparationError("B0 authorization SHA-256 mismatch")
    b0 = load_json(b0_path)
    if b0.get("status") != "SEALED_BEFORE_A1_A0" or not b0.get("exact_match_to_inherited_b0"):
        raise A7SeparationError("B0 is not an authenticated seal")

    evaluator = load_evaluator(prereg)
    data = evaluator.load_class(population_root(prereg), None, "occupied_swap")
    supplier, supplier_a0, feature_receipt = load_supplier(prereg)
    if len(data["y"]) != len(supplier):
        raise A7SeparationError("A7 supplier/population order drift")
    arms = {
        "A1": np.concatenate((data["b0"], supplier), axis=1),
        "A0": np.concatenate((data["b0"], supplier_a0), axis=1),
    }
    b0_folds = {int(row["held_fold"]): row for row in b0["result"]["folds"]}
    requirements = prereg["binding_swap_requirements"]
    folds = []
    for held_fold in range(5):
        held = data["fold"] == held_fold
        sealed = b0_folds[held_fold]
        counts = (
            int(held.sum()),
            int(data["y"][held].sum()),
            int((data["y"][held] == 0).sum()),
        )
        if counts != (sealed["held_actions"], sealed["beneficial"], sealed["harmful"]):
            raise A7SeparationError(f"held-fold identity mismatch: {held_fold}")
        metrics = {
            "B0": sealed["B0"],
            "A1": evaluator.arm_metrics(data, arms["A1"], held_fold),
            "A0": evaluator.arm_metrics(data, arms["A0"], held_fold),
        }
        a1 = metrics["A1"]
        b0_metrics = metrics["B0"]
        a0_metrics = metrics["A0"]
        lifts = {
            "A1_minus_B0_auc": a1["auc"] - b0_metrics["auc"],
            "A1_minus_A0_auc": a1["auc"] - a0_metrics["auc"],
            "A1_minus_B0_retention": a1["retention"] - b0_metrics["retention"],
            "A1_minus_A0_retention": a1["retention"] - a0_metrics["retention"],
        }
        checks = {
            "auc_absolute_floor": a1["auc"] >= requirements["auc_absolute_floor"],
            "retention_absolute_floor": a1["retention"] >= requirements["retention_absolute_floor"],
            "auc_lift_over_B0": lifts["A1_minus_B0_auc"] >= requirements["auc_lift_over_B0"],
            "auc_lift_over_A0": lifts["A1_minus_A0_auc"] >= requirements["auc_lift_over_A0"],
            "retention_lift_over_B0": lifts["A1_minus_B0_retention"] >= requirements["retention_lift_over_B0"],
            "retention_lift_over_A0": lifts["A1_minus_A0_retention"] >= requirements["retention_lift_over_A0"],
            "per_video_fp_budget": all(row["budget_pass"] for row in a1["videos"]),
        }
        folds.append(
            {
                "held_fold": held_fold,
                "held_actions": counts[0],
                "beneficial": counts[1],
                "harmful": counts[2],
                "arms": metrics,
                "lifts": lifts,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    passed = all(row["pass"] for row in folds)
    means = {
        arm: {
            metric: float(np.mean([row["arms"][arm][metric] for row in folds]))
            for metric in ("auc", "retention")
        }
        for arm in ("B0", "A1", "A0")
    }
    write_json(
        output,
        {
            "schema": "celltrack-alpha.g03-a7-separation-result.v1",
            "status": "COMPLETE",
            "verdict": "PASS_OPEN_CONDITIONAL_REALIZATION" if passed else "FAIL_PARK_A7_ON_EVIDENCE",
            "pass": passed,
            "action_class": "occupied_swap",
            "actions": EXPECTED_ACTIONS,
            "beneficial": EXPECTED_POSITIVES,
            "execution_order": ["B0_SEALED", "A1", "A0"],
            "preregistration_sha256": sha256(PREREG),
            "b0_seal": {
                "path": str(b0_path.resolve().relative_to(ROOT)),
                "sha256": sha256(b0_path),
            },
            "score_authorization": {
                "path": str(authorization_path.resolve().relative_to(ROOT)),
                "sha256": sha256(authorization_path),
            },
            "binding_requirements": requirements,
            "feature_receipt": feature_receipt,
            "folds": folds,
            "unweighted_fold_means_for_reporting_only": means,
            "conditional_realization_open": passed,
            "graph_opened": False,
            "cv57_opened": False,
            "gpu_hours": 0.0,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "b0", "score"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b0", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight(args.output)
    elif args.stage == "b0":
        run_b0(args.output)
    else:
        if args.b0 is None or args.authorization is None:
            raise SystemExit("score stage requires --b0 and --authorization")
        run_score(args.output, args.b0, args.authorization)
    print(json.dumps({"stage": args.stage, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
