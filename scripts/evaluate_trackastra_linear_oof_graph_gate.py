#!/usr/bin/env python3
"""Run the nested frozen-Trackastra linear-head held-fit graph gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep  # noqa: E402
from celltrack_alpha import appearance_swap, bh_edge, bh_graph, bh_model  # noqa: E402
from celltrack_alpha.model_candidate_gate import (  # noqa: E402
    load_json_object,
    validate_candidate,
)
from celltrack_alpha.trackastra_linear import (  # noqa: E402
    LinearHead,
    aggregate_precision,
    challenger_labels,
    fit_ridge,
    labeled_subset,
    precision_metrics,
    proposal_features,
    propose_swaps,
    select_threshold,
)
from evaluate_point_context_oof_graph_gate import nested_partitions  # noqa: E402
from evaluate_selected_graph_conflict_gate import (  # noqa: E402
    _edge_counts,
    _realization,
)
from evaluate_selected_graph_direct_swap import _solve_direct  # noqa: E402
from evaluate_temporal_rank_oof_graph_gate import (  # noqa: E402
    _cap,
    graph_gate_decision,
    validate_fit_scope,
)
from fit_selected_graph_temporal_background_gate import _fold  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
)
from prepare_extended_radius_shell import _baseline  # noqa: E402


SCHEMA = "celltrack-alpha.trackastra-linear-oof-graph-gate.v1"
CONTRACT_SCHEMA = (
    "celltrack-alpha.trackastra-linear-oof-graph-gate-contract.v1"
)
H1_CONTRACT_SCHEMA = (
    "celltrack-alpha.h1-zero-wrfeat-graph-gate-contract.v1"
)
FOLD_SCHEMA = "celltrack-alpha.trackastra-linear-oof-fold.v1"
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/trackastra_linear_oof_graph_gate_contract_v1.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "893504032ef208182ed9c27b35f4fbfebdbe08d6577278698a3efde7ec734da6"
)
DEFAULT_POLICY = ROOT / "reports/g02/compute_governance_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/trackastra_linear_oof_graph_gate_v1"
DEFAULT_FEATURE_ROOT = (
    ROOT / "reports/g02/trackastra_full_pool_feature_cache_v1"
)
CODE_PATHS = (
    "scripts/evaluate_trackastra_linear_oof_graph_gate.py",
    "src/celltrack_alpha/trackastra_linear.py",
    "scripts/evaluate_point_context_oof_graph_gate.py",
    "scripts/evaluate_temporal_rank_oof_graph_gate.py",
    "scripts/evaluate_selected_graph_direct_swap.py",
    "src/celltrack_alpha/appearance_swap.py",
    "src/celltrack_alpha/bh_graph.py",
)


class TrackastraOOFGraphGateError(RuntimeError):
    """The pre-registered Trackastra OOF graph screen failed closed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackastraOOFGraphGateError(
            f"cannot read {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise TrackastraOOFGraphGateError(f"{label} must be an object")
    return value


def _require(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sweep._sha256(path) != expected:
        raise TrackastraOOFGraphGateError(
            f"{label} identity mismatch: {path}"
        )


def load_contract(path: Path, expected: str) -> dict[str, Any]:
    _require(path, expected, "contract")
    value = _load_json(path, "contract")
    scope = value.get("scope")
    candidate = value.get("candidate")
    head = candidate.get("linear_head") if isinstance(candidate, Mapping) else None
    threshold = (
        candidate.get("threshold_selection")
        if isinstance(candidate, Mapping)
        else None
    )
    prerequisite = (
        candidate.get("proposal_prerequisite")
        if isinstance(candidate, Mapping)
        else None
    )
    graph = value.get("graph_gate")
    execution = value.get("execution")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or not all(
            isinstance(section, Mapping)
            for section in (
                scope,
                candidate,
                head,
                threshold,
                prerequisite,
                graph,
                execution,
            )
        )
        or scope.get("family") != "6bba"
        or scope.get("source_partition") != "fit-train105"
        or int(scope.get("eligible_video_count", 0)) != 69
        or int(scope.get("outer_fold_count", 0)) != 5
        or scope.get("outer_fold_assignment")
        != "sha256(video identity)[0:8] modulo 5"
        or scope.get("calibration_fold")
        != "(outer_held_fold + 1) modulo 5"
        or any(
            scope.get(key) is not False
            for key in (
                "validation12_access",
                "cv57_access",
                "promotion_authorized",
                "kaggle_action",
                "gpu_action",
                "runpod_action",
            )
        )
        or head.get("kind")
        != "standardized ridge least-squares binary ranker with an unpenalized intercept"
        or int(head.get("feature_count", 0)) != 49
        or float(head.get("ridge_lambda", 0.0)) != 0.01
        or float(head.get("standard_deviation_floor", 0.0)) != 1.0e-6
        or head.get("adaptive_hyperparameter_search") is not False
        or int(threshold.get("minimum_calibration_accepted", 0)) != 100
        or float(threshold.get("minimum_calibration_precision", 0.0))
        != 0.95
        or int(prerequisite.get("minimum_held_accepted_total", 0))
        != 1000
        or float(prerequisite.get("minimum_held_precision_total", 0.0))
        != 0.95
        or float(
            prerequisite.get("minimum_held_precision_every_fold", 0.0)
        )
        != 0.93
        or int(
            prerequisite.get("minimum_held_accepted_every_fold", 0)
        )
        != 100
        or float(graph.get("minimum_adjusted_edge_jaccard_delta", 0.0))
        != 0.005
        or graph.get("require_nonnegative_raw_edge_jaccard_delta")
        is not True
        or float(graph.get("minimum_node_recall_delta", -1.0)) != 0.0
        or graph.get("require_observation_distinct") is not True
        or execution.get("device") != "cpu"
        or execution.get("guard_required") is not True
        or execution.get("swap_growth_allowed") is not False
        or execution.get("graph_resume") is not False
    ):
        raise TrackastraOOFGraphGateError(
            "contract values are not the locked design"
        )
    return value


def load_h1_contract(path: Path, expected: str) -> dict[str, Any]:
    """Load the decoupled fit105-only H1 contract without split metadata."""
    _require(path, expected, "H1 contract")
    value = _load_json(path, "H1 contract")
    scope = value.get("scope")
    candidate = value.get("candidate")
    graph = value.get("graph_gate")
    execution = value.get("execution")
    fallback = value.get("fallback")
    if (
        value.get("schema") != H1_CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or not all(
            isinstance(section, Mapping)
            for section in (
                scope,
                candidate,
                graph,
                execution,
                fallback,
            )
        )
        or scope.get("families") != ["44b6", "6bba"]
        or scope.get("family_counts") != {"44b6": 36, "6bba": 69}
        or scope.get("source_partition") != "fit-train105"
        or int(scope.get("outer_fold_count", 0)) != 5
        or any(
            scope.get(key) is not False
            for key in (
                "validation_access",
                "cross_validation_access",
                "holdout_access",
                "promotion_authorized",
                "kaggle_action",
                "gpu_action",
                "runpod_action",
            )
        )
        or candidate.get("arm") != "points-only-zero-wrfeat-12"
        or candidate.get("proposal_metrics_gate") is not False
        or float(graph.get("minimum_adjusted_edge_jaccard_delta", 0.0))
        != 0.005
        or graph.get("require_nonnegative_raw_edge_jaccard_delta")
        is not True
        or float(graph.get("minimum_node_recall_delta", -1.0)) != 0.0
        or graph.get("kill_44b6_if_adjusted_delta_at_or_below")
        != 0.0
        or execution.get("family_order") != ["44b6", "6bba"]
        or execution.get("guard_required") is not True
        or fallback.get("node") != "H3-MASK-SOURCE"
        or fallback.get("pseudo_masks_forbidden") is not True
        or fallback.get("status") != "CONDITIONAL_NOT_STARTED"
    ):
        raise TrackastraOOFGraphGateError(
            "H1 contract values are not the locked zero-wrfeat design"
        )
    return value


def _code_identity() -> dict[str, str]:
    result = {}
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise TrackastraOOFGraphGateError(
                f"implementation file is missing: {relative}"
            )
        result[relative] = sweep._sha256(path)
    return result


def _feature_records(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    feature_root: Path,
) -> dict[str, dict[str, Any]]:
    requirements = contract["candidate"]["feature_cache_requirements"]
    aggregate = report.get("aggregate")
    files = report.get("files")
    if (
        report.get("schema") != requirements["schema"]
        or report.get("verdict") != requirements["verdict"]
        or report.get("contract", {}).get("sha256")
        != contract["candidate"]["feature_cache_contract_sha256"]
        or not isinstance(aggregate, Mapping)
        or int(aggregate.get("video_count", 0))
        != int(requirements["video_count"])
        or (
            "proposal_count" in requirements
            and int(aggregate.get("proposal_count", 0))
            != int(requirements["proposal_count"])
        )
        or (
            "inference_calls" in requirements
            and int(aggregate.get("inference_calls", 0))
            != int(requirements["inference_calls"])
        )
        or not isinstance(files, list)
        or len(files) != int(requirements["video_count"])
    ):
        raise TrackastraOOFGraphGateError(
            "feature-cache report does not meet its locked contract"
        )
    result = {}
    for raw in files:
        if not isinstance(raw, Mapping):
            raise TrackastraOOFGraphGateError(
                "feature-cache file record is malformed"
            )
        identity = str(raw["identity"])
        path = ROOT / str(raw["feature_path"])
        if (
            identity in result
            or feature_root not in path.resolve().parents
        ):
            raise TrackastraOOFGraphGateError(
                "feature-cache file escaped its output identity"
            )
        _require(path, str(raw["feature_sha256"]), identity)
        result[identity] = dict(raw)
    return result


def _load_feature(
    record: Mapping[str, Any],
    *,
    cache_contract_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    path = ROOT / str(record["feature_path"])
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrackastraOOFGraphGateError(
            f"cannot load feature cache: {path}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema")
        not in (
            "celltrack-alpha.trackastra-full-pool-video-features.v1",
            "celltrack-alpha.trackastra-full-pool-video-features.v2",
        )
        or value.get("identity") != record["identity"]
        or value.get("contract_sha256") != cache_contract_sha256
        or value.get("implementation_sha256") != implementation_sha256
        or len(value.get("feature_names", [])) != 49
        or len(np.asarray(value.get("challengers")))
        != int(record["proposal_count"])
    ):
        raise TrackastraOOFGraphGateError(
            f"feature payload identity drifted: {record['identity']}"
        )
    return value


def _video_labeled(
    identity: str,
    prepare_row: Mapping[str, Any],
    feature_record: Mapping[str, Any],
    *,
    cache_contract_sha256: str,
    implementation_sha256: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    prepared_path = ROOT / str(prepare_row["path"])
    try:
        prepared = torch.load(
            prepared_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise TrackastraOOFGraphGateError(
            f"cannot load prepared labels for {identity}"
        ) from exc
    known = challenger_labels(
        np.asarray(prepared["labeled_challenger_pairs"]),
        np.asarray(prepared["labeled_labels"]),
    )
    payload = _load_feature(
        feature_record,
        cache_contract_sha256=cache_contract_sha256,
        implementation_sha256=implementation_sha256,
    )
    features, labels, _indices = labeled_subset(payload, known)
    return features, labels, len(known)


def _partition_arrays(
    data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    identities: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not identities or any(identity not in data for identity in identities):
        raise TrackastraOOFGraphGateError(
            "nested partition feature coverage is incomplete"
        )
    return (
        np.concatenate([data[identity][0] for identity in identities]),
        np.concatenate([data[identity][1] for identity in identities]),
    )


def _fold_paths(root: Path, held_fold: int) -> tuple[Path, Path]:
    directory = root / f"trackastra_6bba_fold{held_fold}"
    return directory / "head.npz", directory / "checkpoint.json"


def _atomic_fold(
    *,
    root: Path,
    held_fold: int,
    head: LinearHead,
    threshold: float,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    held_ids: Sequence[str],
    calibration_metrics: Mapping[str, Any],
    held_metrics: Mapping[str, Any],
    contract_sha256: str,
    code_identity: Mapping[str, str],
) -> dict[str, Any]:
    artifact, checkpoint = _fold_paths(root, held_fold)
    if artifact.parent.exists():
        raise TrackastraOOFGraphGateError(
            f"fold checkpoint already exists: {held_fold}"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact.parent.name}.", suffix=".stage", dir=root
        )
    )
    try:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            mean=head.mean,
            scale=head.scale,
            weights=head.weights,
            intercept=np.asarray(head.intercept),
            ridge_lambda=np.asarray(head.ridge_lambda),
            threshold=np.asarray(threshold),
        )
        run_bh.atomic_create_bytes(stage / artifact.name, buffer.getvalue())
        record = {
            "schema": FOLD_SCHEMA,
            "held_fold": held_fold,
            "calibration_fold": (held_fold + 1) % 5,
            "fit_ids": list(fit_ids),
            "calibration_ids": list(calibration_ids),
            "held_ids": list(held_ids),
            "threshold": float(threshold),
            "calibration": dict(calibration_metrics),
            "held": dict(held_metrics),
            "contract_sha256": contract_sha256,
            "code_identity": dict(code_identity),
            "artifact_sha256": sweep._sha256(stage / artifact.name),
        }
        run_bh.atomic_create_json(stage / checkpoint.name, record)
        os.rename(stage, artifact.parent)
    finally:
        if stage.exists():
            for path in stage.iterdir():
                path.unlink()
            stage.rmdir()
    return record


def _restore_fold(
    *,
    root: Path,
    held_fold: int,
    fit_ids: Sequence[str],
    calibration_ids: Sequence[str],
    held_ids: Sequence[str],
    contract_sha256: str,
    code_identity: Mapping[str, str],
) -> tuple[LinearHead, float, dict[str, Any]] | None:
    artifact, checkpoint = _fold_paths(root, held_fold)
    if not artifact.exists() and not checkpoint.exists():
        return None
    if not artifact.is_file() or not checkpoint.is_file():
        raise TrackastraOOFGraphGateError(
            f"partial fold checkpoint: {held_fold}"
        )
    record = _load_json(checkpoint, f"fold {held_fold}")
    if (
        record.get("schema") != FOLD_SCHEMA
        or record.get("held_fold") != held_fold
        or record.get("calibration_fold") != (held_fold + 1) % 5
        or record.get("fit_ids") != list(fit_ids)
        or record.get("calibration_ids") != list(calibration_ids)
        or record.get("held_ids") != list(held_ids)
        or record.get("contract_sha256") != contract_sha256
        or record.get("code_identity") != dict(code_identity)
        or record.get("artifact_sha256") != sweep._sha256(artifact)
    ):
        raise TrackastraOOFGraphGateError(
            f"fold checkpoint identity drifted: {held_fold}"
        )
    with np.load(artifact, allow_pickle=False) as value:
        head = LinearHead(
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            weights=np.asarray(value["weights"], dtype=np.float64),
            intercept=float(value["intercept"]),
            ridge_lambda=float(value["ridge_lambda"]),
        )
        threshold = float(value["threshold"])
    head.validate()
    if threshold != float(record["threshold"]):
        raise TrackastraOOFGraphGateError(
            f"fold threshold identity drifted: {held_fold}"
        )
    return head, threshold, record


def _write_park(
    *,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    held: Mapping[str, Any],
    reason: str,
    candidate_gate: Mapping[str, Any],
    code_identity: Mapping[str, str],
    started: float,
) -> dict[str, Any]:
    family = str(getattr(args, "h1_family", "6bba"))
    h1_mode = contract.get("schema") == H1_CONTRACT_SCHEMA
    result = {
        "schema": SCHEMA,
        "status": "PARK",
        "scope": {
            "family": family,
            "source_partition": "fit-train105",
            "video_count": int(
                contract.get("scope", {})
                .get("family_counts", {})
                .get(family, 69)
            ),
            "validation12_access": False,
            "cv57_access": False,
            "promotion_authorized": False,
        },
        "proposal_prerequisite": {
            "held_oof": dict(held),
            "contract": contract["candidate"].get(
                "proposal_prerequisite",
                {"gating": False, "report_only": True},
            ),
            "passed": False,
        },
        "folds": list(folds),
        "graph": None,
        "decision": {
            "status": "PARK",
            "reason": reason,
            "promotion_authorized": False,
            "proposal_metrics_gate": not h1_mode,
        },
        "provenance": {
            "contract": str(args.contract),
            "contract_sha256": args.contract_sha256,
            "candidate_gate": dict(candidate_gate),
            "code_identity": dict(code_identity),
        },
        "execution": {
            "device": "cpu",
            "guard_required": True,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0":
        raise TrackastraOOFGraphGateError(
            "run must be launched by the local resource guard"
        )
    if args.output.exists() and not args.resume:
        raise TrackastraOOFGraphGateError(
            f"refusing to overwrite output: {args.output}"
        )
    h1_mode = getattr(args, "h1_family", None) is not None
    contract = (
        load_h1_contract(args.contract, args.contract_sha256)
        if h1_mode
        else load_contract(args.contract, args.contract_sha256)
    )
    candidate = contract["candidate"]
    required_inputs = [
        (
            candidate["feature_cache_contract"],
            candidate["feature_cache_contract_sha256"],
            "feature-cache contract",
        ),
        (
            contract["control"]["runtime_contract"],
            contract["control"]["runtime_contract_sha256"],
            "APNCC runtime contract",
        ),
    ]
    if h1_mode:
        required_inputs.append(
            (
                contract["data"]["fit_manifest"],
                contract["data"]["fit_manifest_sha256"],
                "fit105 identity manifest",
            )
        )
    else:
        required_inputs.extend(
            [
                (
                    candidate["manifest"],
                    candidate["manifest_sha256"],
                    "candidate manifest",
                ),
                (
                    contract["data"]["prepare_report"],
                    contract["data"]["prepare_report_sha256"],
                    "prepare report",
                ),
            ]
        )
    for relative, expected, label in required_inputs:
        _require(ROOT / relative, expected, label)
    if (
        args.zip != (ROOT / contract["data"]["raw_archive"]).resolve()
        or args.zip.stat().st_size
        != int(contract["data"]["raw_archive_bytes"])
    ):
        raise TrackastraOOFGraphGateError("raw archive identity drifted")
    candidate_gate = (
        {
            "status": "NOT_A_GATE",
            "reason": "H0 bans proposal-only promotion gates",
        }
        if h1_mode
        else validate_candidate(
            load_json_object(ROOT / candidate["manifest"]),
            load_json_object(args.policy),
        )
    )
    feature_report = _load_json(
        ROOT / candidate["feature_cache_report"], "feature-cache report"
    )
    feature_records = _feature_records(
        feature_report, contract, args.feature_root
    )
    implementation_sha256 = str(feature_report["implementation_sha256"])
    if h1_mode:
        family = str(args.h1_family)
        if family not in ("44b6", "6bba"):
            raise TrackastraOOFGraphGateError("unknown H1 family")
        fit_manifest = _load_json(
            ROOT / contract["data"]["fit_manifest"],
            "fit105 identity manifest",
        )
        identities = tuple(str(value) for value in fit_manifest["identities"])
        expected_count = int(contract["scope"]["family_counts"][family])
        eligible = tuple(
            identity
            for identity in identities
            if identity.startswith(f"train:{family}_")
        )
        if (
            fit_manifest.get("scope", {}).get("fit_train105_only") is not True
            or len(identities) != 105
            or len(set(identities)) != 105
            or len(eligible) != expected_count
            or set(eligible) != {
                identity
                for identity in feature_records
                if identity.startswith(f"train:{family}_")
            }
            or set(feature_records) != set(identities)
        ):
            raise TrackastraOOFGraphGateError(
                "authenticated fit105 family coverage drifted"
            )
        examples_root = ROOT / contract["data"]["prepared_examples_root"]
        prepare_rows = {
            identity: {
                "identity": identity,
                "path": str(
                    (
                        examples_root
                        / f"{identity.replace(':', '__', 1)}.pt"
                    ).relative_to(ROOT)
                ),
            }
            for identity in eligible
        }
        if any(
            not (ROOT / str(row["path"])).is_file()
            for row in prepare_rows.values()
        ):
            raise TrackastraOOFGraphGateError(
                "fit-only prepared label coverage is incomplete"
            )
    else:
        prepare = _load_json(
            ROOT / contract["data"]["prepare_report"], "prepare report"
        )
        split = sweep._load(args.split, "frozen split")
        score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
        cv57 = set(
            score_bh_cv57.exact_selection(
                split,
                split_path=args.split,
                archive_video_ids=scoreall.archive_train_video_ids(args.zip),
            )
        )
        eligible = validate_fit_scope(
            prepare["scope"], prepare["scope"], cv57
        )
        if (
            len(eligible) != 69
            or any(
                not identity.startswith("train:6bba_")
                for identity in eligible
            )
            or set(eligible) != set(feature_records)
        ):
            raise TrackastraOOFGraphGateError(
                "eligible fit-only feature coverage drifted"
            )
        prepare_rows = {
            str(row["identity"]): row
            for row in prepare["videos"]
            if str(row["identity"]) in set(eligible)
        }
    partitions = nested_partitions(eligible, 5)
    labeled_data = {}
    label_coverage = {}
    for identity in eligible:
        features, labels, known_count = _video_labeled(
            identity,
            prepare_rows[identity],
            feature_records[identity],
            cache_contract_sha256=candidate[
                "feature_cache_contract_sha256"
            ],
            implementation_sha256=implementation_sha256,
        )
        labeled_data[identity] = (features, labels)
        label_coverage[identity] = {
            "prepared_labeled_proposals": known_count,
            "full_pool_labeled_proposals": len(labels),
            "full_pool_labeled_positive": int(labels.sum()),
        }

    if args.resume:
        if (
            not args.output.is_dir()
            or (args.output / "report.json").exists()
            or not (args.output / "fold_models").is_dir()
            or any((args.output / "control_fragments").iterdir())
            or any((args.output / "candidate_fragments").iterdir())
        ):
            raise TrackastraOOFGraphGateError(
                "resume output contains partial graph or completed state"
            )
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "fold_models").mkdir()
        (args.output / "control_fragments").mkdir()
        (args.output / "candidate_fragments").mkdir()
    started = time.perf_counter()
    code_identity = _code_identity()
    heads: dict[int, tuple[LinearHead, float]] = {}
    fold_records = []
    fold_metrics = []
    head_contract = candidate["linear_head"]
    threshold_contract = candidate["threshold_selection"]
    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(partitions):
        restored = _restore_fold(
            root=args.output / "fold_models",
            held_fold=held_fold,
            fit_ids=fit_ids,
            calibration_ids=calibration_ids,
            held_ids=held_ids,
            contract_sha256=args.contract_sha256,
            code_identity=code_identity,
        )
        if restored is not None:
            head, threshold, record = restored
            heads[held_fold] = (head, threshold)
            fold_records.append(record)
            fold_metrics.append(record["held"])
            print(f"Trackastra fold {held_fold + 1}/5 resumed", flush=True)
            continue
        fit_x, fit_y = _partition_arrays(labeled_data, fit_ids)
        calibration_x, calibration_y = _partition_arrays(
            labeled_data, calibration_ids
        )
        held_x, held_y = _partition_arrays(labeled_data, held_ids)
        head = fit_ridge(
            fit_x,
            fit_y,
            ridge_lambda=float(head_contract["ridge_lambda"]),
            standard_deviation_floor=float(
                head_contract["standard_deviation_floor"]
            ),
        )
        try:
            threshold, calibration_metrics = select_threshold(
                head.score(calibration_x),
                calibration_y,
                minimum_accepted=int(
                    threshold_contract["minimum_calibration_accepted"]
                ),
                minimum_precision=float(
                    threshold_contract["minimum_calibration_precision"]
                ),
            )
        except ValueError:
            return _write_park(
                args=args,
                contract=contract,
                folds=fold_records,
                held=aggregate_precision(fold_metrics),
                reason=f"fold {held_fold} has no admissible calibration threshold",
                candidate_gate=candidate_gate,
                code_identity=code_identity,
                started=started,
            )
        held_metrics = precision_metrics(
            head.score(held_x), held_y, threshold
        )
        record = _atomic_fold(
            root=args.output / "fold_models",
            held_fold=held_fold,
            head=head,
            threshold=threshold,
            fit_ids=fit_ids,
            calibration_ids=calibration_ids,
            held_ids=held_ids,
            calibration_metrics=calibration_metrics,
            held_metrics=held_metrics,
            contract_sha256=args.contract_sha256,
            code_identity=code_identity,
        )
        heads[held_fold] = (head, threshold)
        fold_records.append(record)
        fold_metrics.append(held_metrics)
        print(
            f"Trackastra fold {held_fold + 1}/5 "
            f"threshold={threshold:.8f} "
            f"held={held_metrics['accepted_labeled']} "
            f"precision={held_metrics['precision']:.6f}",
            flush=True,
        )

    held_oof = aggregate_precision(fold_metrics)
    prerequisite = candidate.get(
        "proposal_prerequisite",
        {"gating": False, "report_only": True},
    )
    proposal_passed = h1_mode or (
        held_oof["accepted_labeled"]
        >= int(prerequisite["minimum_held_accepted_total"])
        and held_oof["precision"]
        >= float(prerequisite["minimum_held_precision_total"])
        and all(
            int(value["accepted_labeled"])
            >= int(prerequisite["minimum_held_accepted_every_fold"])
            and float(value["precision"])
            >= float(prerequisite["minimum_held_precision_every_fold"])
            for value in fold_metrics
        )
    )
    if not proposal_passed:
        return _write_park(
            args=args,
            contract=contract,
            folds=fold_records,
            held=held_oof,
            reason="nested held proposal precision/support prerequisite failed",
            candidate_gate=candidate_gate,
            code_identity=code_identity,
            started=started,
        )

    runtime_path = ROOT / contract["control"]["runtime_contract"]
    apncc_contract, geometry_gate = appearance_swap.load_contract(
        runtime_path,
        expected_sha256=contract["control"]["runtime_contract_sha256"],
        root=ROOT,
    )
    base_model, bundle = bh_model.load_v45_model(
        args.support_root, device="cpu"
    )
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sweep._sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise TrackastraOOFGraphGateError("edge scorer did not restore")
    cache_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.bhc")
    }
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    control_dir = args.output / "control_fragments"
    candidate_dir = args.output / "candidate_fragments"
    graph_records = []
    identical = 0
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate(eligible, start=1):
            held_fold = _fold(identity, 5)
            head, threshold = heads[held_fold]
            meta = sweep._load(
                meta_paths[identity], f"cache metadata for {identity}"
            )
            cache = bh_model.load_sparse_cache(
                cache_paths[identity],
                expected_identity=sweep._cache_identity(meta),
            )
            selected, _unused, _no_link, graph = _baseline(cache, scorer)
            expected_selected = (
                None
                if h1_mode
                else int(
                    prepare_rows[identity]["graph"][
                        "raw_selected_edges"
                    ]
                )
            )
            if (
                expected_selected is not None
                and (
                    len(selected) != expected_selected
                    or int(graph["raw_selected_edges"])
                    != expected_selected
                )
            ):
                raise TrackastraOOFGraphGateError(
                    f"selected graph identity drifted for {identity}"
                )
            probabilities, no_link, association = run_bh.score_sparse_edge_cache(
                cache, scorer, edge_module=bh_edge
            )
            graph_nodes = run_bh.detections_to_graph_nodes(
                cache.detections, bh_graph.GraphNode
            )
            adjacent_edges, adjacent_stats = bh_graph.build_adjacent_edges(
                graph_nodes,
                learned_edge_probabilities=probabilities,
                source_no_link_probabilities=no_link,
                topology_mode="legacy",
            )
            dataset = identity.split(":", 1)[1]
            image = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )["0"]
            refiner = sweep.MemoizedImageRefiner(image)
            gap_nodes, gap_edges, gap_stats = bh_graph.close_one_frame_gaps(
                graph_nodes,
                adjacent_edges,
                image_refine_callback=refiner,
            )
            raw_ids = {int(node.node_id) for node in graph_nodes}
            _baseline_nodes, baseline_edges, _baseline_post = (
                bh_graph.postprocess_graph(gap_nodes, gap_edges)
            )
            reconstructed = {
                (int(edge.source_id), int(edge.target_id))
                for edge in baseline_edges
                if int(edge.source_id) in raw_ids
                and int(edge.target_id) in raw_ids
            }
            if reconstructed != selected:
                raise TrackastraOOFGraphGateError(
                    f"selected graph reconstruction drifted for {identity}"
                )
            cap = _cap(apncc_contract, len(selected))
            control_proposals, control_counts = appearance_swap.propose_swaps(
                cache,
                scorer,
                selected,
                geometry_gate,
                image,
                geometry_threshold=float(
                    apncc_contract["geometry_gate"]["threshold"]
                ),
                appearance_threshold=float(
                    apncc_contract["appearance"]["threshold"]
                ),
                maximum=cap,
            )
            feature_payload = _load_feature(
                feature_records[identity],
                cache_contract_sha256=candidate[
                    "feature_cache_contract_sha256"
                ],
                implementation_sha256=implementation_sha256,
            )
            candidate_scores = head.score(
                proposal_features(feature_payload)
            )
            candidate_proposals, candidate_counts = propose_swaps(
                feature_payload,
                candidate_scores,
                threshold=threshold,
                maximum=cap,
            )
            shape = tuple(int(value) for value in image.shape)
            control_solved = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=shape,
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=control_proposals,
                probabilities=probabilities,
                fragment_dir=control_dir,
            )
            candidate_solved = _solve_direct(
                identity=identity,
                dataset=dataset,
                shape=shape,
                graph_nodes=graph_nodes,
                gap_nodes=gap_nodes,
                gap_edges=gap_edges,
                proposals=candidate_proposals,
                probabilities=probabilities,
                fragment_dir=candidate_dir,
            )
            control_realization = _realization(
                Path(control_solved["fragment"]), control_proposals
            )
            candidate_realization = _realization(
                Path(candidate_solved["fragment"]), candidate_proposals
            )
            if (
                control_realization["retained_conflicts"] != 0
                or candidate_realization["retained_conflicts"] != 0
            ):
                raise TrackastraOOFGraphGateError(
                    f"swap realization failed for {identity}"
                )
            control_hash = sweep._sha256(
                Path(control_solved["fragment"])
            )
            candidate_hash = sweep._sha256(
                Path(candidate_solved["fragment"])
            )
            equal = control_hash == candidate_hash
            identical += int(equal)
            graph_records.append(
                {
                    "identity": identity,
                    "held_fold": held_fold,
                    "fit_contains_identity": False,
                    "calibration_contains_identity": False,
                    "association": association,
                    "adjacent_stats": adjacent_stats,
                    "gap_stats": gap_stats,
                    "proposal_cap": cap,
                    "control_proposals": len(control_proposals),
                    "candidate_proposals": len(candidate_proposals),
                    "control_proposal_counts": control_counts,
                    "candidate_score_counts": candidate_counts,
                    "control_realization": control_realization,
                    "candidate_realization": candidate_realization,
                    "control_fragment_sha256": control_hash,
                    "candidate_fragment_sha256": candidate_hash,
                    "fragments_exactly_equal": equal,
                }
            )
            print(
                f"Trackastra graph {index:02d}/{len(eligible)} {identity} "
                f"fold={held_fold} apncc={len(control_proposals)} "
                f"candidate={len(candidate_proposals)} equal={equal}",
                flush=True,
            )
        control_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(control_dir)), eligible
        )
        candidate_manifest = scoreall.validate_fragment_set(
            scoreall.resolve_fragment_paths(str(candidate_dir)), eligible
        )
        control_rows, control_provenance, control_coverage = (
            scoreall.score_fragment_input(store, control_manifest)
        )
        candidate_rows, candidate_provenance, candidate_coverage = (
            scoreall.score_fragment_input(store, candidate_manifest)
        )
    control_by_id = {row["identity"]: row for row in control_rows}
    candidate_by_id = {row["identity"]: row for row in candidate_rows}
    ordered_control = [control_by_id[value] for value in eligible]
    ordered_candidate = [candidate_by_id[value] for value in eligible]
    control_summary = scoreall.official_summary(ordered_control)
    candidate_summary = scoreall.official_summary(ordered_candidate)
    decision = graph_gate_decision(
        control_summary, candidate_summary, contract["graph_gate"]
    )
    observation_distinct = identical < len(eligible)
    graph_passed = (
        decision["counterfactual_gate_status"] == "PASS-GRAPH"
        and observation_distinct
    )
    decision.update(
        {
            "status": "PASS-GRAPH" if graph_passed else "PARK",
            "proposal_prerequisite_passed": (
                None if h1_mode else True
            ),
            "proposal_metrics_gate": not h1_mode,
            "observation_distinct": observation_distinct,
            "promotion_authorized": False,
            "next_action": (
                "return-to-H1-44b6-first-coordinator"
                if h1_mode
                else (
                    "validation12-only-confirmation"
                    if graph_passed
                    else "PARK-no-validation12-or-CV57"
                )
            ),
        }
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS-GRAPH" if graph_passed else "PARK",
        "scope": {
            "family": str(getattr(args, "h1_family", "6bba")),
            "source_partition": "fit-train105",
            "video_count": len(eligible),
            "video_ids": list(eligible),
            "held_predictions_per_video": 1,
            "validation12_access": False,
            "cv57_access": False,
            "promotion_authorized": False,
        },
        "proposal_prerequisite": {
            "held_oof": held_oof,
            "folds": fold_metrics,
            "contract": prerequisite,
            "passed": None if h1_mode else True,
            "gate_used": not h1_mode,
        },
        "label_coverage": label_coverage,
        "folds": fold_records,
        "graph": {
            "control": {
                "summary": control_summary,
                "edge_counts": _edge_counts(ordered_control),
                "coverage": control_coverage,
                "input_provenance": control_provenance,
            },
            "candidate": {
                "summary": candidate_summary,
                "edge_counts": _edge_counts(ordered_candidate),
                "coverage": candidate_coverage,
                "input_provenance": candidate_provenance,
            },
            "observation_identity": {
                "exactly_equal_fragment_count": identical,
                "observation_distinct_fragment_count": len(eligible) - identical,
                "all_observations_identical": not observation_distinct,
            },
            "videos": graph_records,
        },
        "decision": decision,
        "provenance": {
            "contract": str(args.contract),
            "contract_sha256": args.contract_sha256,
            "candidate_gate": candidate_gate,
            "feature_cache_report": str(
                ROOT / candidate["feature_cache_report"]
            ),
            "feature_cache_implementation_sha256": implementation_sha256,
            "code_identity": code_identity,
        },
        "execution": {
            "device": "cpu",
            "guard_required": True,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split", type=Path, default=sweep.DEFAULT_SPLIT)
    parser.add_argument(
        "--scorer-parity",
        type=Path,
        default=sweep.DEFAULT_PARITY,
    )
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument(
        "--support-root", type=Path, default=DEFAULT_SUPPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "contract",
        "policy",
        "feature_root",
        "cache",
        "zip",
        "split",
        "scorer_parity",
        "edge_artifact",
        "edge_auxiliary",
        "support_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "proposal_prerequisite": report["proposal_prerequisite"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
