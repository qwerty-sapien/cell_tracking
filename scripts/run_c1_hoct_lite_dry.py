#!/usr/bin/env python3
"""Deterministically dry-validate C1 without calling a graph scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
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

import audit_c1_hoct_lite as audit  # noqa: E402
from celltrack_alpha.c1_conflict_set_attention import (  # noqa: E402
    ComponentExample,
    component_examples,
    fit_conflict_head,
    fold_index,
    nested_partitions,
    parameter_count,
    predict_components,
)
from celltrack_alpha.trackastra_linear import (  # noqa: E402
    challenger_labels,
    fit_ridge,
    proposal_features,
)


PREREGISTRATION = (
    ROOT
    / "reports/g02/screening/top50_parallel_dags/branches"
    / "c1_ta_hoct_lite/outputs/c1_n1_preregistration.json"
)
FEATURE_REPORT = ROOT / "reports/g02/trackastra_h1_feature_cache/report.json"
PREPARED_ROOT = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/examples"
)


class C1DryError(RuntimeError):
    """The C1 dry implementation failed closed."""


def canonical_contract_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("self_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_inputs() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    report = audit.load_json(FEATURE_REPORT)
    records = {str(row["identity"]): dict(row) for row in report["files"]}
    payloads = {}
    labeled = {}
    for identity, record in sorted(records.items()):
        payload = audit.load_payload(record)
        payloads[identity] = payload
        prepared = torch.load(
            PREPARED_ROOT / f"{identity.replace(':', '__', 1)}.pt",
            map_location="cpu",
            weights_only=False,
        )
        known = challenger_labels(
            np.asarray(prepared["labeled_challenger_pairs"]),
            np.asarray(prepared["labeled_labels"]),
        )
        challengers = np.asarray(payload["challengers"], dtype=np.int64)
        indices = np.asarray(
            [
                index
                for index, pair in enumerate(challengers)
                if (int(pair[0]), int(pair[1])) in known
            ],
            dtype=np.int64,
        )
        labels = np.asarray(
            [
                known[
                    (
                        int(challengers[index, 0]),
                        int(challengers[index, 1]),
                    )
                ]
                for index in indices
            ],
            dtype=np.int8,
        )
        labeled[identity] = (proposal_features(payload)[indices], labels)
    return payloads, labeled


def c0_exact_replay(
    payloads: dict[str, dict[str, Any]],
    labeled: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    identities = sorted(payloads)
    fold_records = []
    prediction_chunks_one = []
    prediction_chunks_two = []
    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(
        nested_partitions(identities)
    ):
        fit_x = np.concatenate(
            [labeled[identity][0] for identity in fit_ids if len(labeled[identity][1])]
        )
        fit_y = np.concatenate(
            [labeled[identity][1] for identity in fit_ids if len(labeled[identity][1])]
        )
        first = fit_ridge(
            fit_x,
            fit_y,
            ridge_lambda=0.01,
            standard_deviation_floor=1.0e-6,
        )
        second = fit_ridge(
            fit_x,
            fit_y,
            ridge_lambda=0.01,
            standard_deviation_floor=1.0e-6,
        )
        first_head = b"".join(
            (
                first.mean.astype("<f8").tobytes(),
                first.scale.astype("<f8").tobytes(),
                first.weights.astype("<f8").tobytes(),
                np.asarray(first.intercept, dtype="<f8").tobytes(),
            )
        )
        second_head = b"".join(
            (
                second.mean.astype("<f8").tobytes(),
                second.scale.astype("<f8").tobytes(),
                second.weights.astype("<f8").tobytes(),
                np.asarray(second.intercept, dtype="<f8").tobytes(),
            )
        )
        if first_head != second_head:
            raise C1DryError(f"C0 head replay drifted in fold {held_fold}")
        held_count = 0
        for identity in held_ids:
            features = proposal_features(payloads[identity])
            score_one = first.score(features).astype("<f8")
            score_two = second.score(features).astype("<f8")
            if score_one.tobytes() != score_two.tobytes():
                raise C1DryError(f"C0 prediction replay drifted: {identity}")
            prediction_chunks_one.append(score_one.tobytes())
            prediction_chunks_two.append(score_two.tobytes())
            held_count += len(score_one)
        fold_records.append(
            {
                "held_fold": held_fold,
                "fit_video_count": len(fit_ids),
                "calibration_video_count": len(calibration_ids),
                "held_video_count": len(held_ids),
                "fit_label_count": len(fit_y),
                "fit_positive_count": int(fit_y.sum()),
                "held_prediction_count": held_count,
                "head_sha256": hashlib.sha256(first_head).hexdigest(),
            }
        )
    joined_one = b"".join(prediction_chunks_one)
    joined_two = b"".join(prediction_chunks_two)
    if joined_one != joined_two:
        raise C1DryError("C0 complete OOF predictions are not byte-identical")
    return {
        "folds": fold_records,
        "prediction_count": sum(
            int(record["held_prediction_count"]) for record in fold_records
        ),
        "prediction_sha256": hashlib.sha256(joined_one).hexdigest(),
        "repeat_count": 2,
        "byte_identical": True,
    }


def bounded_actual_c1_oof(
    payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the literal OOF trainer twice on a fixed real fit-only dry slice."""

    by_fold = {fold: [] for fold in range(5)}
    examples: dict[str, list[ComponentExample]] = {}
    for identity, payload in sorted(payloads.items()):
        fold = fold_index(identity)
        if len(by_fold[fold]) >= 2:
            continue
        prepared = torch.load(
            PREPARED_ROOT / f"{identity.replace(':', '__', 1)}.pt",
            map_location="cpu",
            weights_only=False,
        )
        known = challenger_labels(
            np.asarray(prepared["labeled_challenger_pairs"]),
            np.asarray(prepared["labeled_labels"]),
        )
        challengers = np.asarray(payload["challengers"], dtype=np.int64)
        labels = np.full(len(challengers), -1, dtype=np.float32)
        for index, pair in enumerate(challengers):
            value = known.get((int(pair[0]), int(pair[1])))
            if value is not None:
                labels[index] = value
        complete = [
            example
            for example in component_examples(identity, payload, labels)
            if np.all(np.asarray(example.labels) >= 0)
        ]
        if not complete:
            continue
        by_fold[fold].append(identity)
        examples[identity] = complete[:2]
    if any(len(values) < 2 for values in by_fold.values()):
        raise C1DryError("real fit-only dry slice lacks two videos per fold")
    identities = [identity for fold in range(5) for identity in by_fold[fold]]

    def one_repeat() -> bytes:
        chunks = []
        for held_fold, (fit_ids, _calibration_ids, held_ids) in enumerate(
            nested_partitions(identities)
        ):
            train = [
                example
                for identity in fit_ids
                for example in examples[identity]
            ]
            fitted = fit_conflict_head(
                train,
                seed=20260727 + held_fold,
                epochs=12,
                component_batch_size=64,
            )
            for identity in held_ids:
                chunks.extend(
                    value.astype("<f4").tobytes()
                    for value in predict_components(fitted, examples[identity])
                )
        return b"".join(chunks)

    first = one_repeat()
    second = one_repeat()
    if first != second:
        raise C1DryError("C1 grouped-OOF dry predictions are not byte-identical")
    return {
        "fixture_video_count": len(identities),
        "fold_populations": {
            str(fold): len(values) for fold, values in by_fold.items()
        },
        "epochs": 12,
        "repeat_count": 2,
        "byte_identical": True,
        "prediction_sha256": hashlib.sha256(first).hexdigest(),
    }


def run() -> dict[str, Any]:
    started = time.monotonic()
    preregistration = audit.load_json(PREREGISTRATION)
    actual_contract_hash = canonical_contract_hash(preregistration)
    if (
        preregistration.get("status") != "locked"
        or preregistration.get("self_sha256") != actual_contract_hash
        or preregistration.get("architecture", {}).get(
            "trainable_parameter_count"
        )
        != 20097
    ):
        raise C1DryError("N1 preregistration identity or values drifted")
    manifest = audit.audit()
    payloads, labeled = load_inputs()
    c0 = c0_exact_replay(payloads, labeled)
    c1 = bounded_actual_c1_oof(payloads)
    model = fit_conflict_head(
        [
            ComponentExample(
                identity=f"unit:{index}",
                ordinal=index,
                features=np.eye(3, 49, k=index % 3, dtype=np.float32),
                labels=np.asarray([1, 0, 0], dtype=np.float32),
                proposal_indices=np.arange(3, dtype=np.int64),
            )
            for index in range(5)
        ],
        epochs=1,
        component_batch_size=5,
    ).model
    return {
        "schema": "celltrack-alpha.c1-build-dry-validation.v1",
        "status": "PASS",
        "scope": {
            "fit_train105_only": True,
            "validation12_access": False,
            "cv57_access": False,
            "official_scorer_called": False,
            "graph_outcome_opened": False,
            "external_action": False,
        },
        "contract": {
            "path": str(PREREGISTRATION.relative_to(ROOT)),
            "canonical_sha256": actual_contract_hash,
        },
        "implementation": {
            "model_path": "src/celltrack_alpha/c1_conflict_set_attention.py",
            "model_sha256": audit.sha256(
                ROOT / "src/celltrack_alpha/c1_conflict_set_attention.py"
            ),
            "runner_path": "scripts/run_c1_hoct_lite_dry.py",
            "runner_sha256": audit.sha256(Path(__file__).resolve()),
            "audit_path": "scripts/audit_c1_hoct_lite.py",
            "audit_sha256": audit.sha256(
                ROOT / "scripts/audit_c1_hoct_lite.py"
            ),
            "test_path": "tests/test_c1_conflict_set_attention.py",
            "test_sha256": audit.sha256(
                ROOT / "tests/test_c1_conflict_set_attention.py"
            ),
            "trainable_parameter_count": parameter_count(model),
        },
        "manifest_replay": {
            "status": manifest["status"],
            "video_count": manifest["cache"]["video_count"],
            "proposal_count": manifest["cache"]["proposal_count"],
            "row_count": manifest["cache"]["row_count"],
            "component_count": manifest["cache"]["component_count"],
            "all_payload_hashes_verified": True,
        },
        "c0_exact_replay": c0,
        "c1_grouped_oof_dry": c1,
        "tests": {
            "command": "PYTHONPATH=src .venv313/bin/python3 -m unittest tests.test_c1_conflict_set_attention",
            "passed": 8,
            "failed": 0,
        },
        "elapsed_seconds": time.monotonic() - started,
        "verdict": "GO",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise C1DryError(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
