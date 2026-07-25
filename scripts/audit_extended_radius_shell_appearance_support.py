#!/usr/bin/env python3
"""Screen exact appearance rules for sparse r10--12 um shell abstention."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402


SCHEMA = "celltrack-alpha.extended-radius-shell-appearance-support.v1"
DEFAULT_INPUT = ROOT / "reports/g02/extended_radius_shell_appearance_v1/report.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/extended_radius_shell_appearance_support_v1.json"
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TOPOLOGY_POLICIES = ("all", "free", "conflict")


class ShellSupportError(RuntimeError):
    """The shell-support screen contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise ShellSupportError(f"expected JSON object: {path}")
    return value


def _metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    eligible: np.ndarray,
) -> dict[str, int | float]:
    accepted = eligible & (scores >= threshold)
    positives = int(np.sum(accepted & (labels == 1)))
    known_negatives = int(np.sum(accepted & (labels == 0)))
    unknown_background = int(np.sum(accepted & (labels == -1)))
    total = positives + known_negatives + unknown_background
    return {
        "accepted": total,
        "correct_links": positives,
        "known_negative_links": known_negatives,
        "sampled_unknown_background": unknown_background,
        "sample_precision": float(positives / max(total, 1)),
        "definite_precision": float(
            positives / max(positives + known_negatives, 1)
        ),
        "conservative_utility": (
            positives - 2 * known_negatives - 2 * unknown_background
        ),
    }


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    *,
    minimum_correct: int,
    minimum_precision: float,
    maximum_unknown: int,
) -> dict[str, Any]:
    """Choose an admissible threshold or the best ≥minimum-correct diagnostic."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    eligible = np.asarray(eligible, dtype=bool)
    if (
        scores.shape != labels.shape
        or eligible.shape != labels.shape
        or not np.isfinite(scores).all()
        or np.any(~np.isin(labels, (-1, 0, 1)))
    ):
        raise ShellSupportError("threshold arrays are invalid")
    positive_scores = np.unique(scores[eligible & (labels == 1)])
    if not len(positive_scores):
        raise ShellSupportError("eligible policy has no positive links")
    rows = []
    for threshold in positive_scores:
        metrics = _metrics(scores, labels, float(threshold), eligible)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
                "admissible": (
                    int(metrics["correct_links"]) >= minimum_correct
                    and float(metrics["sample_precision"]) >= minimum_precision
                    and int(metrics["sampled_unknown_background"])
                    <= maximum_unknown
                ),
            }
        )
    admissible = [row for row in rows if row["admissible"]]
    if admissible:
        selected = max(
            admissible,
            key=lambda row: (
                int(row["conservative_utility"]),
                int(row["correct_links"]),
                -int(row["sampled_unknown_background"]),
                -int(row["known_negative_links"]),
                float(row["threshold"]),
            ),
        )
        return {"admissible": True, "selected": selected}
    supported = [
        row for row in rows if int(row["correct_links"]) >= minimum_correct
    ]
    if not supported:
        supported = rows
    diagnostic = max(
        supported,
        key=lambda row: (
            float(row["sample_precision"]),
            int(row["conservative_utility"]),
            int(row["correct_links"]),
            -int(row["sampled_unknown_background"]),
            -int(row["known_negative_links"]),
            float(row["threshold"]),
        ),
    )
    return {"admissible": False, "selected": diagnostic}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    binary = labels == 1
    positives = int(binary.sum())
    negatives = len(labels) - positives
    ranks = rankdata(scores, method="average")
    return float(
        (float(ranks[binary].sum()) - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _partition(
    report: Mapping[str, Any], partition: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    topology = []
    for row in report["videos"]:
        if row["partition"] != partition:
            continue
        value = torch.load(
            ROOT / str(row["path"]), map_location="cpu", weights_only=False
        )
        features.append(np.asarray(value["features"], dtype=np.float32))
        labels.append(np.asarray(value["labels"], dtype=np.int8))
        topology.append(np.asarray(value["topology"], dtype=np.int8))
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(topology),
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ShellSupportError(f"refusing to overwrite output: {args.output}")
    report = _load_json(args.input)
    scope = report.get("scope")
    if (
        report.get("schema")
        != "celltrack-alpha.extended-radius-shell-appearance.v1"
        or not isinstance(scope, Mapping)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise ShellSupportError("appearance input is not exact fit-only 105/12")
    started = time.perf_counter()
    train_x, train_y, train_topology = _partition(report, "train")
    validation_x, validation_y, validation_topology = _partition(
        report, "validation"
    )
    feature_names = tuple(str(value) for value in report["feature_names"])
    pair_index = feature_names.index("pair_ncc")
    margin_index = feature_names.index("pair_minus_maximum_conflict_ncc")
    grid: list[dict[str, Any]] = []
    for weight in WEIGHTS:
        train_score = (
            train_x[:, pair_index] + weight * train_x[:, margin_index]
        )
        validation_score = (
            validation_x[:, pair_index]
            + weight * validation_x[:, margin_index]
        )
        for topology_policy in TOPOLOGY_POLICIES:
            if topology_policy == "all":
                train_eligible = np.ones(len(train_y), dtype=bool)
                validation_eligible = np.ones(len(validation_y), dtype=bool)
            elif topology_policy == "free":
                train_eligible = train_topology == 0
                validation_eligible = validation_topology == 0
            else:
                train_eligible = train_topology > 0
                validation_eligible = validation_topology > 0
            selection = select_threshold(
                train_score,
                train_y,
                train_eligible,
                minimum_correct=args.minimum_train_correct,
                minimum_precision=args.minimum_precision,
                maximum_unknown=args.maximum_train_unknown,
            )
            threshold = float(selection["selected"]["threshold"])
            grid.append(
                {
                    "margin_weight": weight,
                    "topology_policy": topology_policy,
                    "train": selection["selected"],
                    "train_admissible": selection["admissible"],
                    "validation": _metrics(
                        validation_score,
                        validation_y,
                        threshold,
                        validation_eligible,
                    ),
                }
            )
    admissible = [row for row in grid if row["train_admissible"]]
    selected = (
        max(
            admissible,
            key=lambda row: (
                int(row["train"]["conservative_utility"]),
                int(row["train"]["correct_links"]),
                -int(row["train"]["sampled_unknown_background"]),
            ),
        )
        if admissible
        else max(
            grid,
            key=lambda row: (
                float(row["train"]["sample_precision"]),
                int(row["train"]["conservative_utility"]),
                int(row["train"]["correct_links"]),
            ),
        )
    )
    validation_passed = (
        bool(selected["train_admissible"])
        and int(selected["validation"]["correct_links"])
        >= args.minimum_validation_correct
        and float(selected["validation"]["sample_precision"])
        >= args.minimum_precision
        and int(selected["validation"]["sampled_unknown_background"])
        <= args.maximum_validation_unknown
    )
    result = {
        "schema": SCHEMA,
        "status": "BANK" if validation_passed else "PARK",
        "scope": dict(scope),
        "selection_contract": {
            "selection_partition": "105 fit-train videos only",
            "validation_partition": "fixed untouched 12 fit-validation videos",
            "minimum_train_correct": args.minimum_train_correct,
            "minimum_validation_correct": args.minimum_validation_correct,
            "minimum_precision": args.minimum_precision,
            "maximum_train_sampled_unknown": args.maximum_train_unknown,
            "maximum_validation_sampled_unknown": (
                args.maximum_validation_unknown
            ),
            "grid": {
                "margin_weights": list(WEIGHTS),
                "topology_policies": list(TOPOLOGY_POLICIES),
            },
        },
        "data": {
            "train_rows": len(train_y),
            "train_positive": int(np.sum(train_y == 1)),
            "validation_rows": len(validation_y),
            "validation_positive": int(np.sum(validation_y == 1)),
        },
        "row_auc": {
            "train_pair_ncc": _auc(train_y, train_x[:, pair_index]),
            "train_ncc_margin": _auc(train_y, train_x[:, margin_index]),
            "validation_pair_ncc": _auc(
                validation_y, validation_x[:, pair_index]
            ),
            "validation_ncc_margin": _auc(
                validation_y, validation_x[:, margin_index]
            ),
        },
        "selected": selected,
        "validation_passed": validation_passed,
        "grid": grid,
        "provenance": {
            "input": str(args.input),
            "input_sha256": _sha256(args.input),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--minimum-train-correct", type=int, default=25)
    value.add_argument("--minimum-validation-correct", type=int, default=5)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--maximum-train-unknown", type=int, default=0)
    value.add_argument("--maximum-validation-unknown", type=int, default=0)
    return value


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"],
                "validation_passed": report["validation_passed"],
            },
            sort_keys=True,
        )
    )
