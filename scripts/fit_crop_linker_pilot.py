#!/usr/bin/env python3
"""Run the guarded fit-only calibrated Siamese 3-D crop-linker pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

import run_bh  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from audit_crop_linker_data import CANDIDATE_SHA256, PREPARE_SHA256  # noqa: E402
from celltrack_alpha.crop_linker import (  # noqa: E402
    SiameseCropLinker,
    calibration_metrics,
    fit_temperature,
)
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
from fit_selected_graph_temporal_rank_gate import _metrics, _select  # noqa: E402
from prepare_crop_linker_dataset import CONTRACT_SHA256  # noqa: E402


SCHEMA = "celltrack-alpha.crop-linker-pilot.v1"
DATASET_REPORT_SHA256 = (
    "2a57d973341b308170ba654a9ae0e8d65aff9a98e1939a652c2c519ecb1b1d4c"
)
FROZEN_RUNTIME = {
    "seed": 240725,
    "embedding_width": 32,
    "hidden": 64,
    "learning_rate": 3.0e-4,
    "batch_size": 64,
    "evaluation_batch_size": 128,
    "max_steps": 500,
    "minimum_calibration_accepted": 20,
    "pilot_precision_floor": 0.93,
    "full_precision_floor": 0.95,
}


class CropPilotError(RuntimeError):
    """The guarded crop-linker pilot violated its frozen contract."""


@dataclass(frozen=True)
class Examples:
    source: np.ndarray
    target: np.ndarray
    context: np.ndarray
    labels: np.ndarray
    video_index: np.ndarray
    identities: tuple[str, ...]
    families: tuple[str, ...]

    def validate(self) -> None:
        count = len(self.labels)
        if (
            self.source.shape != (count, 16, 32, 32)
            or self.target.shape != self.source.shape
            or self.context.shape != (count, 48)
            or self.video_index.shape != (count,)
            or np.any(~np.isin(self.labels, (0, 1)))
            or not np.isfinite(self.source).all()
            or not np.isfinite(self.target).all()
            or not np.isfinite(self.context).all()
            or len(self.identities) != len(self.families)
        ):
            raise CropPilotError("cached crop example arrays disagree")


def validate_pilot_contract(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> None:
    """Fail closed if a runtime knob, split, or cached-label count drifts."""
    observed_runtime = {
        key: getattr(args, key) for key in FROZEN_RUNTIME
    }
    if observed_runtime != FROZEN_RUNTIME:
        raise CropPilotError("runtime knobs drifted from the frozen pilot")
    split = contract.get("split", {})
    model = contract.get("model", {})
    pilot = contract.get("pilot", {})
    promotion = contract.get("promotion", {})
    constraints = contract.get("constraints", {})
    partitions = dataset.get("scope", {}).get("partitions", {})
    expected_partition_counts = {
        "fit_train": int(split.get("fit_train_videos", -1)),
        "calibration": int(split.get("calibration_videos", -1)),
        "validation": int(split.get("validation_videos", -1)),
    }
    observed_partition_counts = {
        name: len(partitions.get(name, ()))
        for name in expected_partition_counts
    }
    partition_ids = [
        identity
        for name in expected_partition_counts
        for identity in partitions.get(name, ())
    ]
    totals = dataset.get("totals", {})
    labels = contract.get("labels", {})
    if (
        model.get("embedding_width") != args.embedding_width
        or pilot.get("optimizer_steps") != args.max_steps
        or pilot.get("batch_size") != args.batch_size
        or pilot.get("validation_access_before_step_500") is not False
        or observed_partition_counts != expected_partition_counts
        or len(partition_ids) != len(set(partition_ids))
        or split.get("cv57_overlap") != 0
        or totals.get("positive") != labels.get("positive_count")
        or totals.get("guaranteed_negative")
        != labels.get("guaranteed_negative_count")
        or totals.get("uncertain_masked")
        != labels.get("uncertain_masked_count")
        or promotion.get("pilot_precision_floor_overall")
        != args.pilot_precision_floor
        or promotion.get("pilot_precision_floor_per_family")
        != args.pilot_precision_floor
        or promotion.get("full_precision_floor_overall")
        != args.full_precision_floor
        or promotion.get("minimum_validation_utility_gain_over_pair_ncc")
        != 5
        or promotion.get("minimum_family_utility_gain") != 0
        or constraints.get("holdout_access") is not False
        or constraints.get("audit_access") is not False
        or constraints.get("gpu_action") is not False
        or constraints.get("kaggle_action") is not False
        or constraints.get("runpod_action") is not False
        or constraints.get("swap_growth_allowed") is not False
    ):
        raise CropPilotError("pilot contract or cached dataset drifted")


def load_examples(
    report: Mapping[str, Any],
    *,
    partition: str,
    verify_hashes: bool,
) -> Examples:
    expected_ids = tuple(report["scope"]["partitions"][partition])
    file_by_id = {
        str(row["identity"]): row for row in report["files"]
    }
    source_parts = []
    target_parts = []
    context_parts = []
    label_parts = []
    video_parts = []
    identities = []
    families = []
    for video_index, identity in enumerate(expected_ids):
        record = file_by_id.get(identity)
        if record is None:
            raise CropPilotError(f"dataset file missing for {identity}")
        path = ROOT / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or verify_hashes
            and sweep._sha256(path) != str(record["sha256"])
        ):
            raise CropPilotError(f"dataset file identity mismatch: {path}")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if (
            value.get("schema") != "celltrack-alpha.crop-linker-dataset.v1"
            or value.get("identity") != identity
            or value.get("partition") != partition
        ):
            raise CropPilotError(f"dataset payload drifted for {identity}")
        source = np.asarray(value["source_crops"], dtype=np.float16)
        target = np.asarray(value["target_crops"], dtype=np.float16)
        context = np.asarray(value["context"], dtype=np.float32)
        labels = np.asarray(value["labels"], dtype=np.int8)
        count = len(labels)
        source_parts.append(source)
        target_parts.append(target)
        context_parts.append(context)
        label_parts.append(labels)
        video_parts.append(
            np.full(count, video_index, dtype=np.int32)
        )
        identities.append(identity)
        families.append(str(value["family"]))
    result = Examples(
        source=np.concatenate(source_parts),
        target=np.concatenate(target_parts),
        context=np.concatenate(context_parts),
        labels=np.concatenate(label_parts),
        video_index=np.concatenate(video_parts),
        identities=tuple(identities),
        families=tuple(families),
    )
    result.validate()
    return result


def model_logits(
    model: SiameseCropLinker,
    examples: Examples,
    *,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(examples.labels), batch_size):
            stop = min(start + batch_size, len(examples.labels))
            source = torch.from_numpy(
                examples.source[start:stop].astype(np.float32)
            )[:, None].to(device)
            target = torch.from_numpy(
                examples.target[start:stop].astype(np.float32)
            )[:, None].to(device)
            context = torch.from_numpy(
                (
                    examples.context[start:stop] - context_mean
                )
                / context_scale
            ).to(device)
            result.append(model(source, target, context).cpu())
    return torch.cat(result)


def score_records(
    examples: Examples,
    scores: np.ndarray,
    *,
    selected_edges: Mapping[str, int],
) -> list[dict[str, Any]]:
    if scores.shape != examples.labels.shape:
        raise CropPilotError("score/example arrays disagree")
    result = []
    for video_index, identity in enumerate(examples.identities):
        mask = examples.video_index == video_index
        result.append(
            {
                "identity": identity,
                "selected_edges": int(selected_edges[identity]),
                "labeled_confidence": scores[mask],
                "labels": examples.labels[mask],
                "background_confidence": np.empty(
                    0, dtype=np.float64
                ),
            }
        )
    return result


def family_metrics(
    scored: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, dict[str, Any]]:
    return {
        family: _metrics(
            [
                row
                for row in scored
                if str(row["identity"]).split(":", 1)[1].startswith(
                    family + "_"
                )
            ],
            threshold,
        )
        for family in ("44b6", "6bba")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise CropPilotError(f"refusing to overwrite output: {args.output}")
    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise CropPilotError("pilot must run inside the local training guard")
    for path, expected, name in (
        (args.candidate, args.candidate_sha256, "candidate"),
        (args.contract, args.contract_sha256, "contract"),
        (args.dataset_report, args.dataset_report_sha256, "dataset"),
        (args.prepare, args.prepare_sha256, "prepare"),
    ):
        if sweep._sha256(path) != expected:
            raise CropPilotError(f"{name} SHA-256 mismatch")
    candidate_gate = validate_candidate(
        load_json_object(args.candidate), load_json_object(args.policy)
    )
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    dataset = json.loads(
        args.dataset_report.read_text(encoding="ascii")
    )
    prepare = json.loads(args.prepare.read_text(encoding="ascii"))
    if (
        contract.get("schema")
        != "celltrack-alpha.crop-linker-pilot-contract.v1"
        or dataset.get("schema")
        != "celltrack-alpha.crop-linker-dataset.v1"
        or dataset.get("status") != "PASS"
        or dataset.get("provenance", {}).get("contract_sha256")
        != args.contract_sha256
        or prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
    ):
        raise CropPilotError("pilot inputs are inadmissible")
    validate_pilot_contract(args, contract, dataset)
    selected_edges = {
        str(row["identity"]): int(row["graph"]["raw_selected_edges"])
        for row in prepare["videos"]
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    started = time.perf_counter()
    fit = load_examples(
        dataset, partition="fit_train", verify_hashes=True
    )
    context_mean = fit.context.mean(axis=0).astype(np.float32)
    context_scale = fit.context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1.0e-5] = 1.0

    model = SiameseCropLinker(
        context_features=48,
        embedding_width=args.embedding_width,
        hidden=args.hidden,
    ).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    positives = np.flatnonzero(fit.labels == 1)
    negatives = np.flatnonzero(fit.labels == 0)
    if len(positives) == 0 or len(negatives) == 0:
        raise CropPilotError("fit partition lacks both classes")
    generator = np.random.default_rng(args.seed)
    half = args.batch_size // 2
    losses = []
    for step in range(1, args.max_steps + 1):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if step % 50 == 0 or step == args.max_steps:
            print(
                f"crop-linker pilot step={step}/{args.max_steps} "
                f"loss={np.mean(losses[-50:]):.6f}",
                flush=True,
            )

    # The contract forbids even loading calibration/validation examples before
    # the final fit step. Calibration is loaded first; validation remains sealed
    # until model fitting and temperature fitting are both complete.
    calibration = load_examples(
        dataset, partition="calibration", verify_hashes=True
    )
    calibration_logits = model_logits(
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

    validation = load_examples(
        dataset, partition="validation", verify_hashes=True
    )
    validation_logits = model_logits(
        model,
        validation,
        context_mean=context_mean,
        context_scale=context_scale,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    calibration_probabilities = torch.sigmoid(
        calibrated_logits
    ).numpy()
    validation_probabilities = torch.sigmoid(
        temperature.apply(validation_logits)
    ).numpy()
    candidate_calibration_scored = score_records(
        calibration,
        calibration_probabilities,
        selected_edges=selected_edges,
    )
    candidate_validation_scored = score_records(
        validation,
        validation_probabilities,
        selected_edges=selected_edges,
    )
    candidate_threshold, candidate_calibration_metrics = _select(
        candidate_calibration_scored,
        minimum_accepted=args.minimum_calibration_accepted,
        minimum_precision=args.full_precision_floor,
    )
    candidate_validation_metrics = _metrics(
        candidate_validation_scored, candidate_threshold
    )
    candidate_families = family_metrics(
        candidate_validation_scored, candidate_threshold
    )

    control_calibration_scored = score_records(
        calibration,
        calibration.context[:, 44].astype(np.float64),
        selected_edges=selected_edges,
    )
    control_validation_scored = score_records(
        validation,
        validation.context[:, 44].astype(np.float64),
        selected_edges=selected_edges,
    )
    control_threshold, control_calibration_metrics = _select(
        control_calibration_scored,
        minimum_accepted=args.minimum_calibration_accepted,
        minimum_precision=args.full_precision_floor,
    )
    control_validation_metrics = _metrics(
        control_validation_scored, control_threshold
    )
    control_families = family_metrics(
        control_validation_scored, control_threshold
    )
    validation_gain = (
        int(candidate_validation_metrics["conservative_utility"])
        - int(control_validation_metrics["conservative_utility"])
    )
    family_gains = {
        family: (
            int(candidate_families[family]["conservative_utility"])
            - int(control_families[family]["conservative_utility"])
        )
        for family in ("44b6", "6bba")
    }
    pilot_passed = (
        calibration_improved
        and np.isfinite(candidate_threshold)
        and validation_gain
        >= int(
            contract["promotion"][
                "minimum_validation_utility_gain_over_pair_ncc"
            ]
        )
        and float(candidate_validation_metrics["precision"])
        >= args.pilot_precision_floor
        and all(
            float(candidate_families[family]["precision"])
            >= args.pilot_precision_floor
            for family in ("44b6", "6bba")
        )
        and all(gain >= 0 for gain in family_gains.values())
    )

    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "pilot.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "state_dict": model.state_dict(),
            "context_mean": context_mean,
            "context_scale": context_scale,
            "temperature": temperature.temperature,
            "threshold": candidate_threshold,
            "embedding_width": args.embedding_width,
            "hidden": args.hidden,
            "candidate_sha256": args.candidate_sha256,
            "contract_sha256": args.contract_sha256,
            "dataset_report_sha256": args.dataset_report_sha256,
        },
        artifact,
    )
    result = {
        "schema": SCHEMA,
        "status": "CONTINUE-OOF" if pilot_passed else "PARK",
        "scope": {
            "fit_only": True,
            "fit_train_videos": len(fit.identities),
            "calibration_videos": len(calibration.identities),
            "validation_videos": len(validation.identities),
            "holdout_access": False,
            "audit_access": False,
        },
        "architecture": {
            "shared_siamese_encoder": True,
            "channels": [8, 16, 32],
            "embedding_width": args.embedding_width,
            "hidden": args.hidden,
            "context_features": 48,
            "parameter_count": parameter_count,
        },
        "optimization": {
            "seed": args.seed,
            "steps": args.max_steps,
            "batch_size": args.batch_size,
            "balanced_classes": True,
            "learning_rate": args.learning_rate,
            "final_loss": losses[-1],
            "mean_last_50_loss": float(np.mean(losses[-50:])),
        },
        "calibration": {
            "temperature": temperature.temperature,
            "raw": raw_calibration,
            "calibrated": calibrated_calibration,
            "improved_nll_and_brier": calibration_improved,
        },
        "control": {
            "threshold": control_threshold,
            "calibration": control_calibration_metrics,
            "validation": control_validation_metrics,
            "validation_families": control_families,
        },
        "candidate": {
            "threshold": candidate_threshold,
            "calibration": candidate_calibration_metrics,
            "validation": candidate_validation_metrics,
            "validation_families": candidate_families,
        },
        "decision": {
            "pilot_precision_floor": args.pilot_precision_floor,
            "full_precision_floor": args.full_precision_floor,
            "minimum_validation_utility_gain": int(
                contract["promotion"][
                    "minimum_validation_utility_gain_over_pair_ncc"
                ]
            ),
            "validation_utility_gain": validation_gain,
            "validation_family_utility_gains": family_gains,
            "pilot_passed": pilot_passed,
            "next_step": (
                "grouped-five-fold-OOF"
                if pilot_passed
                else "PARK-no-OOF-or-CV57"
            ),
        },
        "artifact": {
            "path": str(artifact),
            "sha256": sweep._sha256(artifact),
        },
        "provenance": {
            "candidate": str(args.candidate),
            "candidate_sha256": args.candidate_sha256,
            "candidate_gate": candidate_gate,
            "contract": str(args.contract),
            "contract_sha256": args.contract_sha256,
            "dataset_report": str(args.dataset_report),
            "dataset_report_sha256": args.dataset_report_sha256,
            "prepare": str(args.prepare),
            "prepare_sha256": args.prepare_sha256,
        },
        "execution": {
            "device": str(device),
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
            / "reports/g02/model_candidate_calibrated_3d_crop_linker_v1.json"
        ),
    )
    value.add_argument("--candidate-sha256", default=CANDIDATE_SHA256)
    value.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "reports/g02/compute_governance_v1.json",
    )
    value.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_pilot_contract_v1.json",
    )
    value.add_argument("--contract-sha256", default=CONTRACT_SHA256)
    value.add_argument(
        "--dataset-report",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_dataset_v1/report.json",
    )
    value.add_argument(
        "--dataset-report-sha256", default=DATASET_REPORT_SHA256
    )
    value.add_argument(
        "--prepare",
        type=Path,
        default=(
            ROOT
            / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
        ),
    )
    value.add_argument("--prepare-sha256", default=PREPARE_SHA256)
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/crop_linker_pilot_v1",
    )
    value.add_argument("--seed", type=int, default=240725)
    value.add_argument("--embedding-width", type=int, default=32)
    value.add_argument("--hidden", type=int, default=64)
    value.add_argument("--learning-rate", type=float, default=3.0e-4)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--evaluation-batch-size", type=int, default=128)
    value.add_argument("--max-steps", type=int, default=500)
    value.add_argument(
        "--minimum-calibration-accepted", type=int, default=20
    )
    value.add_argument("--pilot-precision-floor", type=float, default=0.93)
    value.add_argument("--full-precision-floor", type=float, default=0.95)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {"status": report["status"], "decision": report["decision"]},
            sort_keys=True,
        )
    )
