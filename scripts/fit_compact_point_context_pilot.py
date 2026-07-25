#!/usr/bin/env python3
"""Run the guarded fit-only compact point-context transformer fast-fail."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import random
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
from audit_compact_point_context_data import (  # noqa: E402
    CANDIDATE_SHA256,
    DEFAULT_CACHE,
    PREPARE_SHA256,
    _group_labeled,
)
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
from celltrack_alpha.point_context import (  # noqa: E402
    PointContextTransformer,
    ProposalWindow,
    collate_windows,
    parental_probabilities,
    parental_softmax_loss,
)
from fit_selected_graph_temporal_background_gate import (  # noqa: E402
    _video_data,
)
from fit_selected_graph_temporal_rank_gate import (  # noqa: E402
    _feature_scores,
    _metrics,
    _select,
)


SCHEMA = "celltrack-alpha.compact-point-context-pilot.v1"
DATA_AUDIT_SHA256 = (
    "83ddb9e5784c3a9365b6d52d63234564c69e3948ea35928dc6c459adc03212aa"
)
FROZEN_RUNTIME = {
    "seed": 240724,
    "width": 96,
    "heads": 4,
    "layers": 2,
    "dropout": 0.1,
    "learning_rate": 3.0e-4,
    "background_weight": 0.05,
    "batch_size": 64,
    "max_steps": 500,
    "minimum_train_accepted": 50,
    "minimum_train_precision": 0.95,
    "minimum_pilot_precision": 0.93,
}


class PointContextPilotError(RuntimeError):
    """The guarded fit-only pilot contract failed."""


def proposal_features(
    rows: np.ndarray, groups: Sequence[Sequence[Sequence[int]]]
) -> np.ndarray:
    """Select the worst best-option conflict row for each proposal."""
    result = []
    for group in groups:
        if not group or any(not options for options in group):
            raise PointContextPilotError("proposal group is empty")
        best_rows = [
            max(
                (int(index) for index in options),
                key=lambda index: (float(rows[index, 44]), -index),
            )
            for options in group
        ]
        worst = min(
            best_rows, key=lambda index: (float(rows[index, 44]), index)
        )
        result.append(rows[worst])
    if not result:
        return np.empty((0, rows.shape[1]), dtype=np.float32)
    value = np.asarray(result, dtype=np.float32)
    if not np.isfinite(value).all():
        raise PointContextPilotError("proposal features are non-finite")
    return value


def build_windows(
    prepare: Mapping[str, Any],
    *,
    cache_root: Path,
    allowed_identities: set[str] | None = None,
) -> tuple[list[ProposalWindow], dict[str, int], list[dict[str, Any]]]:
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in cache_root.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in cache_root.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    windows: list[ProposalWindow] = []
    selected_edges: dict[str, int] = {}
    loaded_videos = []
    for raw in prepare["videos"]:
        identity = str(raw["identity"])
        if (
            allowed_identities is not None
            and identity not in allowed_identities
        ):
            continue
        if identity not in cache_paths or identity not in meta_paths:
            raise PointContextPilotError(f"cache is missing {identity}")
        payload = torch.load(
            ROOT / str(raw["path"]), map_location="cpu", weights_only=False
        )
        labeled_x = np.asarray(payload["labeled_features"], dtype=np.float32)
        labeled_y = np.asarray(payload["labeled_labels"], dtype=np.int8)
        labeled_pairs_raw = np.asarray(
            payload["labeled_challenger_pairs"], dtype=np.int64
        )
        background_x = np.asarray(
            payload["background_features"], dtype=np.float32
        )
        background_pairs = np.asarray(
            payload["background_challenger_pairs"], dtype=np.int64
        )
        background_groups = payload["background_groups"]
        labeled_pairs, proposal_labels, labeled_groups = _group_labeled(
            labeled_pairs_raw, labeled_y
        )
        labeled_features = proposal_features(labeled_x, labeled_groups)
        background_features = proposal_features(
            background_x, background_groups
        )
        labeled_set = {
            (int(pair[0]), int(pair[1])) for pair in labeled_pairs
        }
        keep_background = np.asarray(
            [
                (int(pair[0]), int(pair[1])) not in labeled_set
                for pair in background_pairs
            ],
            dtype=bool,
        )
        pairs = np.concatenate(
            (labeled_pairs, background_pairs[keep_background])
        )
        features = np.concatenate(
            (labeled_features, background_features[keep_background])
        )
        labels = np.concatenate(
            (
                proposal_labels,
                np.full(int(keep_background.sum()), -1, dtype=np.int8),
            )
        )
        meta = sweep._load(
            meta_paths[identity], f"cache metadata for {identity}"
        )
        times = bh_model.load_sparse_cache_node_times(
            cache_paths[identity],
            expected_identity=sweep._cache_identity(meta),
        )
        by_time: dict[int, list[int]] = defaultdict(list)
        for row_index, pair in enumerate(pairs):
            source, target = int(pair[0]), int(pair[1])
            if source not in times or target not in times:
                raise PointContextPilotError("proposal node is absent")
            if times[target] != times[source] + 1:
                raise PointContextPilotError("proposal is not adjacent")
            by_time[times[target]].append(row_index)
        family = 0 if identity.split(":", 1)[1].startswith("44b6_") else 1
        for target_t, indices in sorted(by_time.items()):
            order = np.asarray(
                sorted(
                    indices,
                    key=lambda index: (
                        int(pairs[index, 1]),
                        int(pairs[index, 0]),
                    ),
                ),
                dtype=np.int64,
            )
            window = ProposalWindow(
                identity=identity,
                family=family,
                target_t=target_t,
                features=features[order],
                labels=labels[order],
                target_ids=pairs[order, 1].astype(np.int64, copy=False),
            )
            window.validate()
            windows.append(window)
        selected_edges[identity] = int(raw["graph"]["raw_selected_edges"])
        loaded_videos.append(_video_data(raw))
    return windows, selected_edges, loaded_videos


def validate_runtime_contract(args: argparse.Namespace) -> None:
    observed = {
        key: getattr(args, key) for key in FROZEN_RUNTIME
    }
    if (
        observed != FROZEN_RUNTIME
        or args.candidate_sha256 != CANDIDATE_SHA256
        or args.prepare_sha256 != PREPARE_SHA256
        or args.data_audit_sha256 != DATA_AUDIT_SHA256
        or args.cache.resolve() != DEFAULT_CACHE.resolve()
        or args.output.resolve()
        != (
            ROOT / "reports/g02/compact_point_context_pilot_v1"
        ).resolve()
    ):
        raise PointContextPilotError(
            "runtime knobs or artifact identities drifted"
        )


def score_windows(
    model: PointContextTransformer,
    windows: Sequence[ProposalWindow],
    *,
    selected_edges: Mapping[str, int],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    by_video: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"labeled": [], "labels": [], "background": []}
    )
    model.eval()
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        batch = collate_windows(
            batch_windows, mean=mean, scale=scale, device=device
        )
        edge_logits, hidden = model(
            batch["features"], batch["families"], batch["padding_mask"]
        )
        probabilities = parental_probabilities(
            model,
            edge_logits,
            hidden,
            batch["target_ids"],
            batch["padding_mask"],
        )
        for window, scores in zip(
            batch_windows, probabilities, strict=True
        ):
            record = by_video[window.identity]
            labeled = window.labels != -1
            record["labeled"].extend(float(value) for value in scores[labeled])
            record["labels"].extend(
                int(value) for value in window.labels[labeled]
            )
            record["background"].extend(
                float(value) for value in scores[~labeled]
            )
    return [
        {
            "identity": identity,
            "selected_edges": int(selected_edges[identity]),
            "labeled_confidence": np.asarray(
                record["labeled"], dtype=np.float64
            ),
            "labels": np.asarray(record["labels"], dtype=np.int8),
            "background_confidence": np.asarray(
                record["background"], dtype=np.float64
            ),
        }
        for identity, record in sorted(by_video.items())
    ]


def _family_metrics(
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
        raise PointContextPilotError(
            f"refusing to overwrite output: {args.output}"
        )
    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise PointContextPilotError(
            "pilot must run inside the local training guard"
        )
    validate_runtime_contract(args)
    if sweep._sha256(args.candidate) != args.candidate_sha256:
        raise PointContextPilotError("candidate manifest SHA-256 mismatch")
    if sweep._sha256(args.prepare) != args.prepare_sha256:
        raise PointContextPilotError("prepared-data SHA-256 mismatch")
    if sweep._sha256(args.data_audit) != args.data_audit_sha256:
        raise PointContextPilotError("dry-data audit SHA-256 mismatch")
    data_audit = json.loads(args.data_audit.read_text(encoding="ascii"))
    if (
        data_audit.get("schema")
        != "celltrack-alpha.compact-point-context-data-audit.v2"
        or data_audit.get("status") != "PASS"
        or data_audit.get("scope", {}).get("cv57_overlap") != []
    ):
        raise PointContextPilotError("dry-data audit is inadmissible")
    candidate = load_json_object(args.candidate)
    policy = load_json_object(args.policy)
    candidate_gate = validate_candidate(candidate, policy)
    prepare = json.loads(args.prepare.read_text(encoding="ascii"))
    scope = prepare.get("scope")
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
        or not isinstance(scope, Mapping)
        or scope.get("audit_used") is not False
        or scope.get("held_out_used") is not False
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
    ):
        raise PointContextPilotError("prepared fit-only scope is inadmissible")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    started = time.perf_counter()
    train_ids = set(str(value) for value in scope["train_video_ids"])
    validation_ids = set(
        str(value) for value in scope["validation_video_ids"]
    )
    if train_ids & validation_ids:
        raise PointContextPilotError("train and validation identities overlap")
    train_windows, train_selected_edges, train_video_data = build_windows(
        prepare,
        cache_root=args.cache,
        allowed_identities=train_ids,
    )
    if not train_windows:
        raise PointContextPilotError("train windows are empty")

    all_train_features = np.concatenate(
        [window.features for window in train_windows]
    ).astype(np.float32, copy=False)
    mean = all_train_features.mean(axis=0)
    scale = all_train_features.std(axis=0)
    scale[scale < 1.0e-5] = 1.0
    model = PointContextTransformer(
        feature_count=48,
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    losses = []
    group_totals: dict[str, int] = defaultdict(int)
    while step < args.max_steps:
        order = torch.randperm(len(train_windows), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            chosen = [
                train_windows[index]
                for index in order[start : start + args.batch_size]
            ]
            batch = collate_windows(
                chosen, mean=mean, scale=scale, device=device
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            edge_logits, hidden = model(
                batch["features"],
                batch["families"],
                batch["padding_mask"],
            )
            loss, groups = parental_softmax_loss(
                model,
                edge_logits,
                hidden,
                batch["labels"],
                batch["target_ids"],
                batch["padding_mask"],
                background_weight=args.background_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            for key, value in groups.items():
                group_totals[key] += int(value)
            step += 1
            if step % 50 == 0 or step == args.max_steps:
                print(
                    f"point-context pilot step={step}/{args.max_steps} "
                    f"loss={np.mean(losses[-50:]):.6f}",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    train_scored = score_windows(
        model,
        train_windows,
        selected_edges=train_selected_edges,
        mean=mean,
        scale=scale,
        device=device,
        batch_size=args.batch_size,
    )
    # Validation payloads and caches stay physically unopened until all 500
    # fitting steps and train scoring are complete.
    (
        validation_windows,
        validation_selected_edges,
        validation_video_data,
    ) = build_windows(
        prepare,
        cache_root=args.cache,
        allowed_identities=validation_ids,
    )
    if not validation_windows:
        raise PointContextPilotError("validation windows are empty")
    validation_scored = score_windows(
        model,
        validation_windows,
        selected_edges=validation_selected_edges,
        mean=mean,
        scale=scale,
        device=device,
        batch_size=args.batch_size,
    )
    threshold, train_metrics = _select(
        train_scored,
        minimum_accepted=args.minimum_train_accepted,
        minimum_precision=args.minimum_train_precision,
    )
    validation_metrics = _metrics(validation_scored, threshold)
    validation_families = _family_metrics(validation_scored, threshold)

    control_threshold, control_train = _select(
        _feature_scores(train_video_data, 44),
        minimum_accepted=args.minimum_train_accepted,
        minimum_precision=args.minimum_train_precision,
    )
    control_validation_scored = _feature_scores(validation_video_data, 44)
    control_validation = _metrics(
        control_validation_scored, control_threshold
    )
    control_families = _family_metrics(
        control_validation_scored, control_threshold
    )
    validation_gain = (
        int(validation_metrics["conservative_utility"])
        - int(control_validation["conservative_utility"])
    )
    family_gains = {
        family: (
            int(validation_families[family]["conservative_utility"])
            - int(control_families[family]["conservative_utility"])
        )
        for family in ("44b6", "6bba")
    }
    pilot_passed = (
        np.isfinite(threshold)
        and validation_gain > 0
        and float(validation_metrics["precision"])
        >= args.minimum_pilot_precision
        and all(
            float(validation_families[family]["precision"])
            >= args.minimum_pilot_precision
            for family in ("44b6", "6bba")
        )
        and all(gain >= 0 for gain in family_gains.values())
    )

    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "pilot.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "feature_count": 48,
            "width": args.width,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "mean": mean,
            "scale": scale,
            "threshold": threshold,
            "state_dict": model.state_dict(),
            "family_mapping": {"44b6": 0, "6bba": 1},
            "candidate_sha256": args.candidate_sha256,
            "prepare_sha256": args.prepare_sha256,
        },
        artifact,
    )
    result = {
        "schema": SCHEMA,
        "status": "CONTINUE-OOF" if pilot_passed else "PARK",
        "scope": {
            "fit_only": True,
            "train_videos": 105,
            "validation_videos": 12,
            "holdout_access": False,
            "audit_access": False,
        },
        "architecture": {
            "token": (
                "worst best-option row per challenger; 48 full-context features"
            ),
            "window": "one adjacent transition with temporal context features",
            "parental_softmax": True,
            "explicit_no_parent": True,
            "width": args.width,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "parameter_count": parameter_count,
        },
        "optimization": {
            "seed": args.seed,
            "steps": step,
            "batch_size_windows": args.batch_size,
            "learning_rate": args.learning_rate,
            "background_group_weight": args.background_weight,
            "final_loss": losses[-1],
            "mean_last_50_loss": float(np.mean(losses[-50:])),
            "parental_groups_seen": dict(group_totals),
        },
        "control": {
            "train_threshold": control_threshold,
            "train": control_train,
            "validation": control_validation,
            "validation_families": control_families,
        },
        "candidate": {
            "train_threshold": threshold,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_families": validation_families,
        },
        "decision": {
            "pilot_precision_floor": args.minimum_pilot_precision,
            "full_promotion_precision_floor": 0.95,
            "full_promotion_validation_utility_gain": 5,
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
            "policy": str(args.policy),
            "policy_sha256": sweep._sha256(args.policy),
            "prepare": str(args.prepare),
            "prepare_sha256": args.prepare_sha256,
            "data_audit": str(args.data_audit),
            "data_audit_sha256": args.data_audit_sha256,
            "cache": str(args.cache),
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
            / "reports/g02/model_candidate_compact_point_context_transformer_v1.json"
        ),
    )
    value.add_argument("--candidate-sha256", default=CANDIDATE_SHA256)
    value.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "reports/g02/compute_governance_v1.json",
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
        "--data-audit",
        type=Path,
        default=ROOT / "reports/g02/compact_point_context_data_audit_v2.json",
    )
    value.add_argument("--data-audit-sha256", default=DATA_AUDIT_SHA256)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/compact_point_context_pilot_v1",
    )
    value.add_argument("--seed", type=int, default=240724)
    value.add_argument("--width", type=int, default=96)
    value.add_argument("--heads", type=int, default=4)
    value.add_argument("--layers", type=int, default=2)
    value.add_argument("--dropout", type=float, default=0.1)
    value.add_argument("--learning-rate", type=float, default=3.0e-4)
    value.add_argument("--background-weight", type=float, default=0.05)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--max-steps", type=int, default=500)
    value.add_argument("--minimum-train-accepted", type=int, default=50)
    value.add_argument("--minimum-train-precision", type=float, default=0.95)
    value.add_argument("--minimum-pilot-precision", type=float, default=0.93)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(
        json.dumps(
            {"status": report["status"], "decision": report["decision"]},
            sort_keys=True,
        )
    )
