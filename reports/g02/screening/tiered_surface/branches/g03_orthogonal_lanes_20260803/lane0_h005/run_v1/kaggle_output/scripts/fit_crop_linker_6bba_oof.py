#!/usr/bin/env python3
"""Run the guarded nested grouped-OOF 6bba crop-linker confirmation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import fit_crop_linker_pilot as pilot  # noqa: E402
import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha.crop_linker import (  # noqa: E402
    SiameseCropLinker,
    calibration_metrics,
    fit_temperature,
)
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
from fit_selected_graph_temporal_background_gate import _fold  # noqa: E402
from fit_selected_graph_temporal_rank_gate import _metrics, _select  # noqa: E402


SCHEMA = "celltrack-alpha.crop-linker-6bba-nested-oof.v1"
CANDIDATE_SHA256 = (
    "e1598688690a84c09b62014daf56a79c2b8d8b60185ea7ec51c77011f39d6141"
)
CONTRACT_SHA256 = (
    "24316b2ccd58f875e329e6b3792ff40ea13d6079334027d00b9599a1b4a28930"
)
DATASET_SHA256 = pilot.DATASET_REPORT_SHA256
PREPARE_SHA256 = pilot.PREPARE_SHA256
PILOT_REPORT_SHA256 = (
    "082e682b74a78ae74005ac0d7b86ee7d76b446bae909b3cdb9a8bd93a29bf333"
)
FAMILY_AUDIT_SHA256 = (
    "9eb568aff7f1d9e043030fb67ce964d0ef98d3f85d451ed20f9dfc9bb8257dd3"
)
FROZEN_RUNTIME = {
    "fold_count": 5,
    "steps_per_fold": 500,
    "batch_size": 64,
    "evaluation_batch_size": 128,
    "embedding_width": 32,
    "hidden": 64,
    "learning_rate": 3.0e-4,
    "weight_decay": 1.0e-4,
    "gradient_norm_clip": 5.0,
    "seed_base": 260724,
    "minimum_calibration_accepted": 20,
    "minimum_calibration_precision": 0.95,
}


class CropOofError(RuntimeError):
    """The nested OOF scope, artifact, or runtime contract was violated."""


@dataclass(frozen=True)
class FoldRoles:
    outer: int
    calibration: int
    fit_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    held_ids: tuple[str, ...]

    def validate(self, eligible: Sequence[str]) -> None:
        fit = set(self.fit_ids)
        calibration = set(self.calibration_ids)
        held = set(self.held_ids)
        if (
            self.calibration != (self.outer + 1) % 5
            or fit & calibration
            or fit & held
            or calibration & held
            or fit | calibration | held != set(eligible)
            or not fit
            or not calibration
            or not held
        ):
            raise CropOofError("nested fold roles overlap or omit identities")


def eligible_identities(dataset: Mapping[str, Any]) -> tuple[str, ...]:
    partitions = dataset.get("scope", {}).get("partitions", {})
    source = [
        str(identity)
        for partition in ("fit_train", "calibration")
        for identity in partitions.get(partition, ())
    ]
    result = tuple(
        sorted(
            identity
            for identity in source
            if identity.split(":", 1)[1].startswith("6bba_")
        )
    )
    validation = set(partitions.get("validation", ()))
    if (
        len(result) != 69
        or len(result) != len(set(result))
        or set(result) & validation
        or any(not identity.startswith("train:6bba_") for identity in result)
    ):
        raise CropOofError("eligible 6bba scope drifted")
    return result


def fold_roles(
    eligible: Sequence[str], outer: int, count: int = 5
) -> FoldRoles:
    if count != 5 or not 0 <= outer < count:
        raise CropOofError("only the frozen five-fold design is admissible")
    calibration = (outer + 1) % count
    result = FoldRoles(
        outer=outer,
        calibration=calibration,
        fit_ids=tuple(
            identity
            for identity in eligible
            if _fold(identity, count) not in (outer, calibration)
        ),
        calibration_ids=tuple(
            identity
            for identity in eligible
            if _fold(identity, count) == calibration
        ),
        held_ids=tuple(
            identity
            for identity in eligible
            if _fold(identity, count) == outer
        ),
    )
    result.validate(eligible)
    return result


def load_eligible_examples(
    dataset: Mapping[str, Any],
    identities: Sequence[str],
) -> pilot.Examples:
    file_by_id = {
        str(row["identity"]): row for row in dataset.get("files", ())
    }
    sources = []
    targets = []
    contexts = []
    labels = []
    video_indices = []
    families = []
    for video_index, identity in enumerate(identities):
        record = file_by_id.get(identity)
        if record is None:
            raise CropOofError(f"dataset file missing for {identity}")
        path = Path(str(record["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sweep._sha256(path) != str(record["sha256"])
        ):
            raise CropOofError(f"dataset file identity mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema")
            != "celltrack-alpha.crop-linker-dataset.v1"
            or payload.get("identity") != identity
            or payload.get("family") != "6bba"
            or payload.get("partition") == "validation"
        ):
            raise CropOofError(f"ineligible dataset payload: {identity}")
        source = np.asarray(payload["source_crops"], dtype=np.float16)
        target = np.asarray(payload["target_crops"], dtype=np.float16)
        context = np.asarray(payload["context"], dtype=np.float32)
        label = np.asarray(payload["labels"], dtype=np.int8)
        count = len(label)
        sources.append(source)
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        video_indices.append(
            np.full(count, video_index, dtype=np.int32)
        )
        families.append("6bba")
    result = pilot.Examples(
        source=np.concatenate(sources),
        target=np.concatenate(targets),
        context=np.concatenate(contexts),
        labels=np.concatenate(labels),
        video_index=np.concatenate(video_indices),
        identities=tuple(identities),
        families=tuple(families),
    )
    result.validate()
    return result


def subset_examples(
    examples: pilot.Examples, identities: Sequence[str]
) -> pilot.Examples:
    index_by_id = {
        identity: index
        for index, identity in enumerate(examples.identities)
    }
    if len(identities) != len(set(identities)) or any(
        identity not in index_by_id for identity in identities
    ):
        raise CropOofError("subset identities are missing or duplicated")
    masks = [
        examples.video_index == index_by_id[identity]
        for identity in identities
    ]
    keep = np.logical_or.reduce(masks)
    old_to_new = {
        index_by_id[identity]: new
        for new, identity in enumerate(identities)
    }
    video_index = np.asarray(
        [old_to_new[int(index)] for index in examples.video_index[keep]],
        dtype=np.int32,
    )
    result = pilot.Examples(
        source=examples.source[keep],
        target=examples.target[keep],
        context=examples.context[keep],
        labels=examples.labels[keep],
        video_index=video_index,
        identities=tuple(identities),
        families=tuple("6bba" for _ in identities),
    )
    result.validate()
    return result


def sum_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row["correct_rescues"]) for row in rows)
    harmful = sum(int(row["harmful_swaps"]) for row in rows)
    accepted = correct + harmful
    return {
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "precision": correct / max(accepted, 1),
        "conservative_utility": correct - 2 * harmful,
        "folds": len(rows),
    }


def validate_contract(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    dataset: Mapping[str, Any],
    eligible: Sequence[str],
) -> None:
    observed_runtime = {
        key: getattr(args, key) for key in FROZEN_RUNTIME
    }
    scope = contract.get("scope", {})
    nested = contract.get("nested_oof", {})
    model = contract.get("model", {})
    optimization = contract.get("optimization", {})
    calibration = contract.get("calibration", {})
    promotion = contract.get("promotion", {})
    constraints = contract.get("constraints", {})
    roles = [
        fold_roles(eligible, outer, args.fold_count)
        for outer in range(args.fold_count)
    ]
    observed_counts = [len(role.held_ids) for role in roles]
    held = [
        identity for role in roles for identity in role.held_ids
    ]
    if (
        observed_runtime != FROZEN_RUNTIME
        or contract.get("schema")
        != "celltrack-alpha.crop-linker-6bba-oof-contract.v1"
        or scope.get("family") != "6bba"
        or scope.get("eligible_videos") != len(eligible)
        or scope.get("excluded_source_partition") != "validation"
        or scope.get("prior_validation_reuse") is not False
        or nested.get("outer_folds") != args.fold_count
        or nested.get("outer_fold_video_counts") != observed_counts
        or sorted(held) != sorted(eligible)
        or len(held) != len(set(held))
        or model.get("embedding_width") != args.embedding_width
        or model.get("hidden_width") != args.hidden
        or model.get("context_features") != 48
        or optimization.get("steps_per_outer_fold")
        != args.steps_per_fold
        or optimization.get("batch_size") != args.batch_size
        or optimization.get("learning_rate") != args.learning_rate
        or optimization.get("weight_decay") != args.weight_decay
        or optimization.get("gradient_norm_clip")
        != args.gradient_norm_clip
        or optimization.get("seed_base") != args.seed_base
        or calibration.get("minimum_accepted")
        != args.minimum_calibration_accepted
        or calibration.get("minimum_precision")
        != args.minimum_calibration_precision
        or promotion.get("minimum_oof_precision") != 0.95
        or promotion.get("minimum_outer_fold_precision") != 0.93
        or promotion.get("minimum_oof_utility_gain_over_control") != 50
        or promotion.get("minimum_outer_fold_utility_gain") != 0
        or constraints.get("validation12_access") is not False
        or constraints.get("holdout_access") is not False
        or constraints.get("audit_access") is not False
        or constraints.get("kaggle_action") is not False
        or constraints.get("gpu_action") is not False
        or constraints.get("runpod_action") is not False
        or constraints.get("swap_growth_allowed") is not False
        or dataset.get("status") != "PASS"
    ):
        raise CropOofError("OOF contract, scope, or runtime drifted")


def train_fold(
    fit: pilot.Examples,
    calibration: pilot.Examples,
    held: pilot.Examples,
    *,
    selected_edges: Mapping[str, int],
    outer: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = args.seed_base + outer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    context_mean = fit.context.mean(axis=0).astype(np.float32)
    context_scale = fit.context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1.0e-5] = 1.0
    model = SiameseCropLinker(
        context_features=48,
        embedding_width=args.embedding_width,
        hidden=args.hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    positives = np.flatnonzero(fit.labels == 1)
    negatives = np.flatnonzero(fit.labels == 0)
    if (
        not len(positives)
        or not len(negatives)
        or not np.any(calibration.labels == 1)
        or not np.any(calibration.labels == 0)
        or not np.any(held.labels == 1)
        or not np.any(held.labels == 0)
    ):
        raise CropOofError(f"outer fold {outer} lacks both classes")
    generator = np.random.default_rng(seed)
    half = args.batch_size // 2
    losses = []
    for step in range(1, args.steps_per_fold + 1):
        chosen = np.concatenate(
            (
                generator.choice(positives, size=half, replace=True),
                generator.choice(
                    negatives,
                    size=args.batch_size - half,
                    replace=True,
                ),
            )
        )
        generator.shuffle(chosen)
        source = torch.from_numpy(
            fit.source[chosen].astype(np.float32)
        )[:, None].to(device)
        target = torch.from_numpy(
            fit.target[chosen].astype(np.float32)
        )[:, None].to(device)
        context = torch.from_numpy(
            (fit.context[chosen] - context_mean) / context_scale
        ).to(device)
        labels = torch.from_numpy(
            fit.labels[chosen].astype(np.float32)
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(source, target, context)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_norm_clip
        )
        optimizer.step()
        losses.append(float(loss.item()))
        if step % 100 == 0 or step == args.steps_per_fold:
            print(
                f"6bba crop OOF fold={outer} "
                f"step={step}/{args.steps_per_fold} "
                f"loss={np.mean(losses[-100:]):.6f}",
                flush=True,
            )

    calibration_logits = pilot.model_logits(
        model,
        calibration,
        context_mean=context_mean,
        context_scale=context_scale,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    calibration_labels = torch.from_numpy(
        calibration.labels.astype(np.int64)
    )
    temperature = fit_temperature(
        calibration_logits, calibration_labels
    )
    raw_calibration = calibration_metrics(
        calibration_logits, calibration_labels
    )
    calibrated_logits = temperature.apply(calibration_logits)
    calibrated_calibration = calibration_metrics(
        calibrated_logits, calibration_labels
    )
    calibration_improved = (
        calibrated_calibration["negative_log_likelihood"]
        < raw_calibration["negative_log_likelihood"]
        and calibrated_calibration["brier_score"]
        < raw_calibration["brier_score"]
    )
    held_logits = pilot.model_logits(
        model,
        held,
        context_mean=context_mean,
        context_scale=context_scale,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    candidate_calibration = pilot.score_records(
        calibration,
        torch.sigmoid(calibrated_logits).numpy(),
        selected_edges=selected_edges,
    )
    candidate_held = pilot.score_records(
        held,
        torch.sigmoid(temperature.apply(held_logits)).numpy(),
        selected_edges=selected_edges,
    )
    candidate_threshold, candidate_calibration_metrics = _select(
        candidate_calibration,
        minimum_accepted=args.minimum_calibration_accepted,
        minimum_precision=args.minimum_calibration_precision,
    )
    candidate_held_metrics = _metrics(
        candidate_held, candidate_threshold
    )
    control_calibration = pilot.score_records(
        calibration,
        calibration.context[:, 44].astype(np.float64),
        selected_edges=selected_edges,
    )
    control_held = pilot.score_records(
        held,
        held.context[:, 44].astype(np.float64),
        selected_edges=selected_edges,
    )
    control_threshold, control_calibration_metrics = _select(
        control_calibration,
        minimum_accepted=args.minimum_calibration_accepted,
        minimum_precision=args.minimum_calibration_precision,
    )
    control_held_metrics = _metrics(control_held, control_threshold)
    utility_gain = (
        int(candidate_held_metrics["conservative_utility"])
        - int(control_held_metrics["conservative_utility"])
    )
    fold_report = {
        "outer_fold": outer,
        "seed": seed,
        "fit_video_count": len(fit.identities),
        "calibration_video_count": len(calibration.identities),
        "held_video_count": len(held.identities),
        "fit_example_count": len(fit.labels),
        "calibration_example_count": len(calibration.labels),
        "held_example_count": len(held.labels),
        "held_identities": list(held.identities),
        "optimization": {
            "steps": args.steps_per_fold,
            "final_loss": losses[-1],
            "mean_last_100_loss": float(np.mean(losses[-100:])),
        },
        "calibration": {
            "temperature": temperature.temperature,
            "raw": raw_calibration,
            "calibrated": calibrated_calibration,
            "improved_nll_and_brier": calibration_improved,
        },
        "candidate": {
            "threshold": (
                float(candidate_threshold)
                if np.isfinite(candidate_threshold)
                else None
            ),
            "threshold_admissible": bool(
                np.isfinite(candidate_threshold)
            ),
            "calibration": candidate_calibration_metrics,
            "held": candidate_held_metrics,
        },
        "control": {
            "threshold": (
                float(control_threshold)
                if np.isfinite(control_threshold)
                else None
            ),
            "threshold_admissible": bool(
                np.isfinite(control_threshold)
            ),
            "calibration": control_calibration_metrics,
            "held": control_held_metrics,
        },
        "utility_gain": utility_gain,
    }
    state = {
        "outer_fold": outer,
        "state_dict": model.state_dict(),
        "context_mean": context_mean,
        "context_scale": context_scale,
        "temperature": temperature.temperature,
        "threshold": candidate_threshold,
        "seed": seed,
    }
    return fold_report, state


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CropOofError(f"refusing to overwrite output: {args.output}")
    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise CropOofError("OOF must run inside the local training guard")
    for path, expected, name in (
        (args.candidate, CANDIDATE_SHA256, "candidate"),
        (args.contract, CONTRACT_SHA256, "contract"),
        (args.dataset_report, DATASET_SHA256, "dataset"),
        (args.prepare, PREPARE_SHA256, "prepare"),
        (args.pilot_report, PILOT_REPORT_SHA256, "pilot report"),
        (args.family_audit, FAMILY_AUDIT_SHA256, "family audit"),
    ):
        if sweep._sha256(path) != expected:
            raise CropOofError(f"{name} SHA-256 mismatch")
    candidate_gate = validate_candidate(
        load_json_object(args.candidate),
        load_json_object(args.policy),
    )
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    dataset = json.loads(
        args.dataset_report.read_text(encoding="ascii")
    )
    prepare = json.loads(args.prepare.read_text(encoding="ascii"))
    eligible = eligible_identities(dataset)
    validate_contract(args, contract, dataset, eligible)
    prepared_train = set(
        prepare.get("scope", {}).get("train_video_ids", ())
    )
    prepared_validation = set(
        prepare.get("scope", {}).get("validation_video_ids", ())
    )
    if (
        set(eligible) - prepared_train
        or set(eligible) & prepared_validation
        or prepare.get("scope", {}).get("fit_only") is not True
    ):
        raise CropOofError("prepare split disagrees with OOF scope")
    selected_edges = {
        str(row["identity"]): int(row["graph"]["raw_selected_edges"])
        for row in prepare["videos"]
        if str(row["identity"]) in set(eligible)
    }
    if set(selected_edges) != set(eligible):
        raise CropOofError("selected-edge budgets are incomplete")

    started = time.perf_counter()
    all_examples = load_eligible_examples(dataset, eligible)
    folds = []
    states = []
    for outer in range(args.fold_count):
        roles = fold_roles(eligible, outer, args.fold_count)
        fit = subset_examples(all_examples, roles.fit_ids)
        calibration = subset_examples(
            all_examples, roles.calibration_ids
        )
        held = subset_examples(all_examples, roles.held_ids)
        fold_report, state = train_fold(
            fit,
            calibration,
            held,
            selected_edges=selected_edges,
            outer=outer,
            args=args,
        )
        folds.append(fold_report)
        states.append(state)

    candidate_oof = sum_metrics(
        [fold["candidate"]["held"] for fold in folds]
    )
    control_oof = sum_metrics(
        [fold["control"]["held"] for fold in folds]
    )
    utility_gain = (
        int(candidate_oof["conservative_utility"])
        - int(control_oof["conservative_utility"])
    )
    promotion = contract["promotion"]
    passed = (
        float(candidate_oof["precision"])
        >= float(promotion["minimum_oof_precision"])
        and all(
            float(fold["candidate"]["held"]["precision"])
            >= float(promotion["minimum_outer_fold_precision"])
            for fold in folds
        )
        and utility_gain
        >= int(promotion["minimum_oof_utility_gain_over_control"])
        and all(
            int(fold["utility_gain"])
            >= int(promotion["minimum_outer_fold_utility_gain"])
            for fold in folds
        )
        and int(candidate_oof["accepted_labeled"])
        >= int(control_oof["accepted_labeled"])
        and all(
            fold["calibration"]["improved_nll_and_brier"]
            for fold in folds
        )
        and all(
            fold["candidate"]["threshold_admissible"]
            and fold["control"]["threshold_admissible"]
            for fold in folds
        )
    )

    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "oof_models.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "candidate_sha256": CANDIDATE_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "dataset_sha256": DATASET_SHA256,
            "fold_states": states,
        },
        artifact,
    )
    result = {
        "schema": SCHEMA,
        "status": "PROMOTE-CV57-PREREGISTRATION" if passed else "PARK",
        "scope": {
            "family": "6bba",
            "eligible_videos": len(eligible),
            "eligible_examples": len(all_examples.labels),
            "outer_predictions_per_video": 1,
            "validation12_access": False,
            "cv57_access": False,
            "audit_access": False,
        },
        "candidate_gate": candidate_gate,
        "runtime": dict(FROZEN_RUNTIME),
        "folds": folds,
        "oof": {
            "candidate": candidate_oof,
            "control": control_oof,
            "utility_gain": utility_gain,
        },
        "decision": {
            "passed": passed,
            "criteria": promotion,
            "next_step": (
                promotion["next_step_on_pass"]
                if passed
                else promotion["next_step_on_failure"]
            ),
        },
        "artifact": {
            "path": str(artifact),
            "sha256": sweep._sha256(artifact),
        },
        "provenance": {
            "candidate": str(args.candidate),
            "candidate_sha256": CANDIDATE_SHA256,
            "contract": str(args.contract),
            "contract_sha256": CONTRACT_SHA256,
            "dataset_report": str(args.dataset_report),
            "dataset_report_sha256": DATASET_SHA256,
            "prepare": str(args.prepare),
            "prepare_sha256": PREPARE_SHA256,
            "pilot_report": str(args.pilot_report),
            "pilot_report_sha256": PILOT_REPORT_SHA256,
            "family_audit": str(args.family_audit),
            "family_audit_sha256": FAMILY_AUDIT_SHA256,
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "guard_required": True,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--candidate",
        type=Path,
        default=(
            ROOT
            / "reports/g02/model_candidate_6bba_crop_linker_oof_v1.json"
        ),
    )
    value.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_6bba_oof_contract_v1.json",
    )
    value.add_argument(
        "--dataset-report",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_dataset_v1/report.json",
    )
    value.add_argument(
        "--prepare",
        type=Path,
        default=(
            ROOT
            / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
        ),
    )
    value.add_argument(
        "--pilot-report",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_pilot_v1/report.json",
    )
    value.add_argument(
        "--family-audit",
        type=Path,
        default=(
            ROOT
            / "reports/g02/crop_linker_family_threshold_audit_v1.json"
        ),
    )
    value.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "reports/g02/compute_governance_v1.json",
    )
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_6bba_oof_v1",
    )
    value.add_argument("--fold-count", type=int, default=5)
    value.add_argument("--steps-per-fold", type=int, default=500)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--evaluation-batch-size", type=int, default=128)
    value.add_argument("--embedding-width", type=int, default=32)
    value.add_argument("--hidden", type=int, default=64)
    value.add_argument("--learning-rate", type=float, default=3.0e-4)
    value.add_argument("--weight-decay", type=float, default=1.0e-4)
    value.add_argument("--gradient-norm-clip", type=float, default=5.0)
    value.add_argument("--seed-base", type=int, default=260724)
    value.add_argument(
        "--minimum-calibration-accepted", type=int, default=20
    )
    value.add_argument(
        "--minimum-calibration-precision", type=float, default=0.95
    )
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {"status": report["status"], "oof": report["oof"]},
            sort_keys=True,
        )
    )
