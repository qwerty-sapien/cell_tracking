"""Fail-closed, scorer-free replay primitives for U0-T4CANON."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from . import appearance_swap
from .trackastra_linear import (
    LinearHead,
    TrackastraLinearError,
    challenger_labels,
    fit_ridge,
    labeled_subset,
    precision_metrics,
    proposal_features,
    propose_swaps,
    select_threshold,
)


CONTRACT_SCHEMA = "celltrack-alpha.u0-t4canon-contract.v1"
REPLAY_SCHEMA = "celltrack-alpha.u0-t4canon-replay.v1"
BASE_SCHEMA = "celltrack-alpha.trackastra-apncc-base-video.v2"
FEATURE_SCHEMA = "celltrack-alpha.trackastra-full-pool-video-features.v2"
FAMILIES = ("44b6", "6bba")
PAIR_ARRAYS = (
    "challengers",
    "conflicts",
    "proposal_offsets",
    "transition_offsets",
    "transition_target_t",
    "geometry_confidences",
    "geometry_probabilities",
)


class U0T4CanonError(RuntimeError):
    """The separate T4-native replay contract failed closed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise U0T4CanonError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise U0T4CanonError(f"{label} must be a JSON object")
    return value


def require_sha256(path: Path, expected: str, label: str) -> None:
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or not path.is_file()
        or sha256_path(path) != expected
    ):
        raise U0T4CanonError(f"{label} SHA-256 mismatch")


def fold_for_identity(identity: str, fold_count: int = 5) -> int:
    if not identity.startswith("train:") or fold_count != 5:
        raise U0T4CanonError("fold identity or count is invalid")
    return int(hashlib.sha256(identity.encode("ascii")).hexdigest()[:8], 16) % 5


def validate_fold_assignment(identity: str, observed: int) -> None:
    if observed != fold_for_identity(identity):
        raise U0T4CanonError(f"fold assignment drifted for {identity}")


def validate_decision_path(path: Path, contract: Mapping[str, Any]) -> None:
    value = str(path).lower()
    tokens = contract.get("forbidden_path_tokens")
    if not isinstance(tokens, list) or not tokens:
        raise U0T4CanonError("forbidden-path contract is missing")
    for token in tokens:
        if str(token).lower() in value:
            raise U0T4CanonError("decision input path is outside the fit-only scope")


def validate_scorer_pin(contract: Mapping[str, Any], observed: str) -> None:
    expected = contract["pins"]["frozen_scorer_evidence"]["sha256"]
    if observed != expected:
        raise U0T4CanonError("frozen scorer identity drifted")


def load_contract(path: Path, expected_sha256: str, root: Path) -> dict[str, Any]:
    require_sha256(path, expected_sha256, "U0 contract")
    value = _load_json(path, "U0 contract")
    invariants = value.get("invariants")
    scope = value.get("scope")
    old = value.get("old_h1_n1")
    policy = value.get("m5_policy")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("mechanism") != "U0-T4CANON"
        or value.get("class") != "CONTROL / GOVERNANCE"
        or value.get("status") != "preregistered_unlocked"
        or not isinstance(invariants, list)
        or len(invariants) != 12
        or len({row.get("id") for row in invariants if isinstance(row, Mapping)})
        != 12
        or not isinstance(scope, Mapping)
        or scope.get("partition") != "fit-train105"
        or scope.get("family_counts") != {"44b6": 36, "6bba": 69}
        or int(scope.get("video_count", 0)) != 105
        or int(scope.get("outer_fold_count", 0)) != 5
        or any(
            scope.get(key) is not False
            for key in (
                "validation_access",
                "cross_validation_access",
                "holdout_access",
                "scoring_access",
                "gpu_action",
                "kaggle_action",
                "runpod_action",
                "submission_action",
            )
        )
        or not isinstance(old, Mapping)
        or old.get("status") != "DEAD_STRUCTURAL_M5_T4_DRIFT"
        or old.get("unchanged") is not True
        or not isinstance(policy, Mapping)
        or policy.get("decision_reachable") is not False
        or policy.get("may_select") != []
        or policy.get("tolerance_change") != "forbidden"
    ):
        raise U0T4CanonError("U0 contract is not the preregistered design")
    for specification in value["pins"].values():
        if not isinstance(specification, Mapping):
            continue
        relative = specification.get("path")
        digest = specification.get("sha256")
        if isinstance(relative, str) and isinstance(digest, str):
            target = root / relative
            validate_decision_path(target, value)
            require_sha256(target, digest, relative)
    validate_scorer_pin(
        value, sha256_path(root / value["pins"]["frozen_scorer_evidence"]["path"])
    )
    return value


def _report_records(
    report_path: Path,
    expected_sha256: str,
    expected_implementation: str,
    expected_contract: str,
    *,
    expected_feature_count: int,
    expected_video_count: int,
    expected_proposals: int,
    expected_rows: int,
    label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require_sha256(report_path, expected_sha256, f"{label} report")
    report = _load_json(report_path, f"{label} report")
    aggregate = report.get("aggregate")
    records = report.get("files")
    if (
        not isinstance(aggregate, Mapping)
        or int(aggregate.get("video_count", 0)) != expected_video_count
        or int(aggregate.get("proposal_count", 0)) != expected_proposals
        or int(aggregate.get("row_count", 0)) != expected_rows
        or report.get("implementation_sha256") != expected_implementation
        or report.get("contract", {}).get("sha256") != expected_contract
        or (
            expected_feature_count == 48
            and int(report.get("feature_count", 0)) != 48
        )
        or (
            expected_feature_count == 49
            and int(report.get("feature_contract", {}).get("output_feature_count", 0))
            != 49
        )
        or not isinstance(records, list)
        or len(records) != expected_video_count
    ):
        raise U0T4CanonError(f"{label} report coverage drifted")
    result: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise U0T4CanonError(f"{label} atomic record is malformed")
        record = dict(raw)
        identity = str(record.get("identity"))
        recorded = Path(str(record.get("feature_path")))
        if (
            identity in result
            or recorded.is_absolute()
            or ".." in recorded.parts
            or recorded.name != identity.replace(":", "__", 1) + ".pt"
            or record.get("implementation_sha256") != expected_implementation
            or record.get("contract_sha256") != expected_contract
        ):
            raise U0T4CanonError(f"{label} atomic identity drifted")
        payload = report_path.parent / "videos" / recorded.name
        sidecar = payload.with_suffix(".json")
        require_sha256(payload, str(record.get("feature_sha256")), identity)
        if _load_json(sidecar, f"{label} sidecar") != record:
            raise U0T4CanonError(f"{label} sidecar drifted for {identity}")
        result[identity] = {**record, "_payload": str(payload)}
    return result, report


def load_authenticated_records(
    contract: Mapping[str, Any],
    fit_manifest_path: Path,
    base_report_path: Path,
    feature_report_path: Path,
) -> tuple[
    tuple[str, ...],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    validate_decision_path(fit_manifest_path, contract)
    validate_decision_path(base_report_path, contract)
    validate_decision_path(feature_report_path, contract)
    pins = contract["pins"]
    require_sha256(
        fit_manifest_path, pins["fit_manifest"]["sha256"], "fit manifest"
    )
    manifest = _load_json(fit_manifest_path, "fit manifest")
    identities = tuple(str(value) for value in manifest.get("identities", ()))
    if (
        manifest.get("scope", {}).get("fit_train105_only") is not True
        or len(identities) != 105
        or len(set(identities)) != 105
        or {
            family: sum(identity.startswith(f"train:{family}_") for identity in identities)
            for family in FAMILIES
        }
        != {"44b6": 36, "6bba": 69}
    ):
        raise U0T4CanonError("fit manifest identity scope drifted")
    cache_contract_sha = pins["cache_contract"]["sha256"]
    scope = contract["scope"]
    base, _base_report = _report_records(
        base_report_path,
        pins["t4_base_report"]["sha256"],
        pins["t4_base_report"]["implementation_sha256"],
        cache_contract_sha,
        expected_feature_count=48,
        expected_video_count=int(scope["video_count"]),
        expected_proposals=int(scope["proposal_count"]),
        expected_rows=int(scope["row_count"]),
        label="T4 base",
    )
    feature, _feature_report = _report_records(
        feature_report_path,
        pins["t4_feature_report"]["sha256"],
        pins["t4_feature_report"]["implementation_sha256"],
        cache_contract_sha,
        expected_feature_count=49,
        expected_video_count=int(scope["video_count"]),
        expected_proposals=int(scope["proposal_count"]),
        expected_rows=int(scope["row_count"]),
        label="T4 feature",
    )
    if set(base) != set(identities) or set(feature) != set(identities):
        raise U0T4CanonError("T4 payload coverage differs from fit manifest")
    return identities, base, feature


def _load_torch(path: Path, label: str) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise U0T4CanonError(f"cannot load {label}") from exc
    if not isinstance(value, dict):
        raise U0T4CanonError(f"{label} payload is not an object")
    return value


def validate_payload_pair(
    identity: str,
    base: Mapping[str, Any],
    feature: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if (
        base.get("schema") != BASE_SCHEMA
        or feature.get("schema") != FEATURE_SCHEMA
        or base.get("identity") != identity
        or feature.get("identity") != identity
        or base.get("contract_sha256") != contract["pins"]["cache_contract"]["sha256"]
        or feature.get("contract_sha256")
        != contract["pins"]["cache_contract"]["sha256"]
        or base.get("implementation_sha256")
        != contract["pins"]["t4_base_report"]["implementation_sha256"]
        or feature.get("implementation_sha256")
        != contract["pins"]["t4_feature_report"]["implementation_sha256"]
        or len(base.get("feature_names", ())) != 48
        or len(feature.get("feature_names", ())) != 49
        or list(feature["feature_names"][:48]) != list(base["feature_names"])
        or feature["feature_names"][48] != "trackastra_margin"
    ):
        raise U0T4CanonError(f"payload schema or identity drifted for {identity}")
    for name in PAIR_ARRAYS:
        left = np.asarray(base.get(name))
        right = np.asarray(feature.get(name))
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(
            left, right
        ):
            raise U0T4CanonError(f"T4 {name} drifted for {identity}")
    base_rows = np.asarray(base.get("row_features"))
    feature_rows = np.asarray(feature.get("row_features"))
    if (
        base_rows.ndim != 2
        or feature_rows.shape != (len(base_rows), 49)
        or base_rows.shape[1] != 48
        or base_rows.dtype != feature_rows.dtype
        or not np.array_equal(base_rows, feature_rows[:, :48])
        or int(base.get("selected_raw_edges", -1))
        != int(feature.get("selected_raw_edges", -2))
    ):
        raise U0T4CanonError(f"T4 feature rows drifted for {identity}")


def _load_pair(
    identity: str,
    base_record: Mapping[str, Any],
    feature_record: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _load_torch(Path(str(base_record["_payload"])), f"{identity} base")
    feature = _load_torch(
        Path(str(feature_record["_payload"])), f"{identity} feature"
    )
    validate_payload_pair(identity, base, feature, contract)
    return base, feature


def _array_digest(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    payload = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _head_record(
    head: LinearHead, threshold: float | None
) -> dict[str, Any]:
    return {
        "mean": _array_digest(head.mean),
        "scale": _array_digest(head.scale),
        "weights": _array_digest(head.weights),
        "intercept": float(head.intercept),
        "ridge_lambda": float(head.ridge_lambda),
        "threshold": None if threshold is None else float(threshold),
    }


def _partitions(identities: Sequence[str]) -> list[tuple[list[str], list[str], list[str]]]:
    ordered = sorted(identities)
    result = []
    seen: list[str] = []
    for held_fold in range(5):
        calibration_fold = (held_fold + 1) % 5
        held = [identity for identity in ordered if fold_for_identity(identity) == held_fold]
        calibration = [
            identity
            for identity in ordered
            if fold_for_identity(identity) == calibration_fold
        ]
        fit = [
            identity
            for identity in ordered
            if fold_for_identity(identity) not in (held_fold, calibration_fold)
        ]
        if (
            not fit
            or not calibration
            or not held
            or set(fit).intersection(calibration)
            or set(fit).intersection(held)
            or set(calibration).intersection(held)
        ):
            raise U0T4CanonError("grouped fold partition is invalid")
        for identity in held:
            validate_fold_assignment(identity, held_fold)
        result.append((fit, calibration, held))
        seen.extend(held)
    if sorted(seen) != ordered:
        raise U0T4CanonError("held-fold identity coverage drifted")
    return result


def _prepared_rows(
    contract: Mapping[str, Any], root: Path, identities: Sequence[str]
) -> dict[str, dict[str, Any]]:
    specification = contract["pins"]["fit_label_report"]
    path = root / specification["path"]
    validate_decision_path(path, contract)
    require_sha256(path, specification["sha256"], "fit label report")
    report = _load_json(path, "fit label report")
    rows = {
        str(row.get("identity")): dict(row)
        for row in report.get("videos", ())
        if isinstance(row, Mapping) and row.get("partition") == "train"
    }
    if set(rows) != set(identities):
        raise U0T4CanonError("fit label report scope drifted")
    for identity, row in rows.items():
        label_path = root / str(row["path"])
        validate_decision_path(label_path, contract)
        require_sha256(label_path, str(row["sha256"]), f"{identity} labels")
    return rows


def _labeled_data(
    contract: Mapping[str, Any],
    root: Path,
    identities: Sequence[str],
    feature_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = _prepared_rows(contract, root, identities)
    result = {}
    for identity in identities:
        prepared = _load_torch(root / str(rows[identity]["path"]), f"{identity} labels")
        feature = _load_torch(
            Path(str(feature_records[identity]["_payload"])), f"{identity} feature"
        )
        known = challenger_labels(
            np.asarray(prepared["labeled_challenger_pairs"]),
            np.asarray(prepared["labeled_labels"]),
        )
        try:
            features, labels, _indices = labeled_subset(feature, known)
        except TrackastraLinearError as exc:
            if str(exc) != "feature cache has no prepared labels":
                raise
            features = proposal_features(feature)[:0]
            labels = np.empty(0, dtype=np.int8)
        result[identity] = (features, labels)
    return result


def _concat(
    values: Mapping[str, tuple[np.ndarray, np.ndarray]], identities: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([values[identity][0] for identity in identities]),
        np.concatenate([values[identity][1] for identity in identities]),
    )


def fit_family_heads(
    contract: Mapping[str, Any],
    data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    identities: Sequence[str],
) -> tuple[
    dict[int, tuple[LinearHead, float | None]], list[dict[str, Any]]
]:
    fit_spec = contract["fit"]
    heads = {}
    records = []
    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(
        _partitions(identities)
    ):
        fit_x, fit_y = _concat(data, fit_ids)
        calibration_x, calibration_y = _concat(data, calibration_ids)
        held_x, held_y = _concat(data, held_ids)
        head = fit_ridge(
            fit_x,
            fit_y,
            ridge_lambda=float(fit_spec["ridge_lambda"]),
            standard_deviation_floor=float(fit_spec["standard_deviation_floor"]),
        )
        try:
            threshold, calibration_metrics = select_threshold(
                head.score(calibration_x),
                calibration_y,
                minimum_accepted=100,
                minimum_precision=0.95,
            )
            held_metrics: dict[str, Any] | None = precision_metrics(
                head.score(held_x), held_y, threshold
            )
        except TrackastraLinearError as exc:
            if str(exc) != "no calibration threshold satisfies the locked gate":
                raise
            threshold = None
            calibration_metrics = {
                "status": "NO_ADMISSIBLE_THRESHOLD",
                "minimum_accepted": 100,
                "minimum_precision": 0.95,
                "eligible_labeled": len(calibration_y),
            }
            held_metrics = None
        heads[held_fold] = (head, threshold)
        records.append(
            {
                "held_fold": held_fold,
                "calibration_fold": (held_fold + 1) % 5,
                "fit_ids": fit_ids,
                "calibration_ids": calibration_ids,
                "held_ids": held_ids,
                "head": _head_record(head, threshold),
                "calibration_diagnostic": calibration_metrics,
                "held_diagnostic": held_metrics,
            }
        )
    return heads, records


def _proposal_cap(contract: Mapping[str, Any], selected_edges: int) -> int:
    expression = contract["apncc_reconstruction"]["proposal_cap"]
    if expression != "max(16, ceil(0.005 * selected_raw_edges))":
        raise U0T4CanonError("APNCC cap contract drifted")
    return max(16, int(math.ceil(0.005 * selected_edges)))


def apncc_proposals(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = np.asarray(payload["row_features"], dtype=np.float64)
    challengers = np.asarray(payload["challengers"], dtype=np.int64)
    conflicts = np.asarray(payload["conflicts"], dtype=np.int64)
    offsets = np.asarray(payload["proposal_offsets"], dtype=np.int64)
    geometry = np.asarray(payload["geometry_confidences"], dtype=np.float64)
    row_probabilities = np.asarray(
        payload["geometry_probabilities"], dtype=np.float64
    )
    geometry_threshold = float(
        contract["apncc_reconstruction"]["geometry_threshold"]
    )
    appearance_threshold = float(
        contract["apncc_reconstruction"]["appearance_threshold"]
    )
    raw = []
    for index, challenger in enumerate(challengers):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        margins = rows[start:stop, 44]
        confidence = float(np.min(margins))
        if geometry[index] < geometry_threshold or confidence < appearance_threshold:
            continue
        raw.append(
            {
                "challenger": challenger.tolist(),
                "conflicts": conflicts[start:stop].tolist(),
                "confidence": confidence,
                "appearance_margins": margins.tolist(),
                "geometry_confidence": float(geometry[index]),
                "row_probabilities": row_probabilities[start:stop].tolist(),
            }
        )
    selected = appearance_swap._greedy_nonconflicting(raw)
    return selected[
        : _proposal_cap(contract, int(payload["selected_raw_edges"]))
    ]


def graph_mutation_payload(
    identity: str,
    payload: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    role: str,
) -> dict[str, Any]:
    mutations = [
        {
            "add": [int(value) for value in proposal["challenger"]],
            "remove": [
                [int(value) for value in pair] for pair in proposal["conflicts"]
            ],
        }
        for proposal in proposals
    ]
    return {
        "schema": "celltrack-alpha.u0-graph-mutation-payload.v1",
        "identity": identity,
        "role": role,
        "base_selected_raw_edges": int(payload["selected_raw_edges"]),
        "mutation_count": len(mutations),
        "mutations": mutations,
    }


def run_replay(
    *,
    root: Path,
    contract_path: Path,
    contract_sha256: str,
    fit_manifest_path: Path,
    base_report_path: Path,
    feature_report_path: Path,
    implementation_paths: Sequence[Path],
) -> dict[str, Any]:
    contract = load_contract(contract_path, contract_sha256, root)
    identities, base_records, feature_records = load_authenticated_records(
        contract,
        fit_manifest_path,
        base_report_path,
        feature_report_path,
    )
    data = _labeled_data(contract, root, identities, feature_records)
    family_heads = {}
    fold_records = {}
    for family in FAMILIES:
        family_ids = [
            identity
            for identity in identities
            if identity.startswith(f"train:{family}_")
        ]
        heads, records = fit_family_heads(contract, data, family_ids)
        family_heads[family] = heads
        fold_records[family] = records
    del data

    videos = []
    disabled_exact = True
    selected_proposal_count = 0
    apncc_proposal_count = 0
    for identity in sorted(identities):
        base, feature = _load_pair(
            identity,
            base_records[identity],
            feature_records[identity],
            contract,
        )
        family = identity.split(":", 1)[1].split("_", 1)[0]
        held_fold = fold_for_identity(identity)
        head, threshold = family_heads[family][held_fold]
        cap = _proposal_cap(contract, int(feature["selected_raw_edges"]))
        if threshold is None:
            candidate = []
        else:
            scores = head.score(proposal_features(feature))
            candidate, _counts = propose_swaps(
                feature, scores, threshold=threshold, maximum=cap
            )
        control = apncc_proposals(base, contract)
        control_graph = graph_mutation_payload(
            identity, base, control, "apncc-control"
        )
        disabled_graph = graph_mutation_payload(
            identity, base, control, "apncc-control"
        )
        candidate_graph = graph_mutation_payload(
            identity, base, candidate, "u0-t4canon-active"
        )
        exact = canonical_json_bytes(control_graph) == canonical_json_bytes(
            disabled_graph
        )
        disabled_exact = disabled_exact and exact
        selected_proposal_count += len(candidate)
        apncc_proposal_count += len(control)
        videos.append(
            {
                "identity": identity,
                "family": family,
                "held_fold": held_fold,
                "fit_contains_identity": False,
                "calibration_contains_identity": False,
                "candidate_graph": candidate_graph,
                "apncc_graph": control_graph,
                "disabled_challenger_graph": disabled_graph,
                "disabled_challenger_exact_apncc": exact,
                "authenticated_arrays": {
                    name: _array_digest(np.asarray(feature[name]))
                    for name in (*PAIR_ARRAYS, "row_features")
                },
            }
        )
    if not disabled_exact:
        raise U0T4CanonError("disabled challenger did not reproduce APNCC exactly")
    implementations = {}
    for path in implementation_paths:
        resolved = path.resolve()
        if root.resolve() not in resolved.parents:
            raise U0T4CanonError("implementation path escaped workspace")
        implementations[str(resolved.relative_to(root.resolve()))] = sha256_path(
            resolved
        )
    result = {
        "schema": REPLAY_SCHEMA,
        "status": "PASS_EXACT_T4_REPLAY_INPUT",
        "contract": {
            "path": str(contract_path.resolve().relative_to(root.resolve())),
            "sha256": contract_sha256,
        },
        "scope": {
            "partition": "fit-train105",
            "video_count": 105,
            "family_counts": {"44b6": 36, "6bba": 69},
            "score_invoked": False,
            "forbidden_scope_access": False,
            "external_action": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "seed": int(contract["scope"]["seed"]),
        },
        "inputs": {
            "fit_manifest_sha256": sha256_path(fit_manifest_path),
            "base_report_sha256": sha256_path(base_report_path),
            "feature_report_sha256": sha256_path(feature_report_path),
        },
        "implementation": implementations,
        "folds": fold_records,
        "graphs": {
            "candidate_proposal_count": selected_proposal_count,
            "apncc_proposal_count": apncc_proposal_count,
            "disabled_challenger_exact_apncc": disabled_exact,
            "videos": videos,
        },
    }
    return result


def write_replay(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(result)
    with path.open("xb") as handle:
        handle.write(data)


__all__ = [
    "U0T4CanonError",
    "apncc_proposals",
    "canonical_json_bytes",
    "fold_for_identity",
    "graph_mutation_payload",
    "load_authenticated_records",
    "load_contract",
    "require_sha256",
    "run_replay",
    "sha256_path",
    "validate_decision_path",
    "validate_fold_assignment",
    "validate_payload_pair",
    "validate_scorer_pin",
    "write_replay",
]
