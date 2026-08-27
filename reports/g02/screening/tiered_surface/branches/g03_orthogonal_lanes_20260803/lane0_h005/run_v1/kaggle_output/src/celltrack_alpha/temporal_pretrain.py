"""Fold-safe alpha-guided sparse adaptation of released temporal blocks.

The objective consumes persisted alpha detections and selected graph edges.  It
never discovers pairs during training: each positive migration and its two
same-coordinate stale negatives are gathered directly from a declared sparse
link.  The only trainable parameters are the released ``temporal_blocks``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .t_adapter import CONFIG_SHA256, EXPECTED_CONFIG, WEIGHTS_SHA256


PRETRAIN_SCHEMA = "celltrack-alpha.temporal-pseudolink-pretrain.v2"
PRETRAIN_OBJECTIVE = "alpha-sparse-migration-ranking.v1"
PRETRAIN_INPUT_POLICY = {
    "competition_train_images_only": True,
    "alpha_image_derived_pseudo_links_used": True,
    "labels_used": False,
    "tracks_used": False,
    "test_used": False,
    "held_out_used": False,
    "audit_used": False,
}
PRETRAIN_PAIRING_POLICY = {
    "spatial_pair_matrix": False,
    "object_pair_matrix": False,
    "pair_discovery_during_training": False,
    "spatial_correspondence": "persisted-alpha-selected-migration-links",
    "negative_correspondence": "linked-coordinate-stale-voxel-only",
    "temporal_window": 2,
    "maximum_links_per_window": "identity-bound",
}

ALPHA_CACHE_META_SCHEMA = "celltrack-alpha.bh-cache-meta.v1"
ALPHA_CACHE_INTENT_SCHEMA = "celltrack-alpha.bh-cache-intent.v1"
ALPHA_STATUS_SCHEMA = "celltrack-alpha.bh-video-status.v1"
ALPHA_TRANSACTION_SCHEMA = "celltrack-alpha.bh-video-transaction.v1"
ALPHA_FEATURE_SCHEMA = "celltrack-alpha.bh-video-features.v1"
ALPHA_RUNNER_SCHEMA = "celltrack-alpha.bh-runner.v1"
ALPHA_CSV_COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)


class TemporalPretrainError(ValueError):
    """Raised when a pseudo-link, artifact, objective, or checkpoint is invalid."""


@dataclass(frozen=True)
class TemporalPretrainLoad:
    """Fold and source identity of a restored temporal-pretraining artifact."""

    checkpoint_sha256: str
    fold: int
    completed_steps: int
    source_groups: tuple[str, ...]
    source_video_ids: tuple[str, ...]
    backbone_tensor_count: int
    backbone_numel: int


@dataclass(frozen=True)
class PseudoLink:
    """One candidate-backed adjacent migration selected by alpha."""

    source_id: int
    target_id: int
    source_zyx: tuple[int, int, int]
    target_zyx: tuple[int, int, int]
    confidence: float
    distance_um: float


@dataclass(frozen=True)
class DivisionMotif:
    """One candidate-backed parent with exactly two adjacent alpha children."""

    parent_id: int
    child_ids: tuple[int, int]
    parent_zyx: tuple[int, int, int]
    child_zyx: tuple[tuple[int, int, int], tuple[int, int, int]]
    confidence: float


@dataclass(frozen=True)
class PseudoLinkWindow:
    """Bounded sparse supervision for one adjacent two-frame image window."""

    start_frame: int
    links: tuple[PseudoLink, ...]
    divisions: tuple[DivisionMotif, ...]


@dataclass(frozen=True)
class AlphaPseudoLinkArtifact:
    """Validated fit-source alpha artifact and its bounded sparse windows."""

    group_id: str
    video_id: str
    stem: str
    exact_identity_sha256: str
    cache_sha256: str
    fragment_sha256: str
    status_sha256: str
    feature_sha256: str
    transaction_sha256: str
    run_contract_sha256: str
    alpha_weight_sha256: str
    windows: tuple[PseudoLinkWindow, ...]
    summary: Mapping[str, int]

    def provenance(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "video_id": self.video_id,
            "stem": self.stem,
            "exact_identity_sha256": self.exact_identity_sha256,
            "cache_sha256": self.cache_sha256,
            "fragment_sha256": self.fragment_sha256,
            "status_sha256": self.status_sha256,
            "feature_sha256": self.feature_sha256,
            "transaction_sha256": self.transaction_sha256,
            "run_contract_sha256": self.run_contract_sha256,
            "alpha_weight_sha256": self.alpha_weight_sha256,
            "summary": dict(self.summary),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite_tensor(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return _finite_tensor(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TemporalPretrainError(f"duplicate JSON key in alpha artifact: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                TemporalPretrainError(f"nonfinite JSON token in alpha artifact: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TemporalPretrainError(f"invalid alpha JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise TemporalPretrainError(f"alpha JSON artifact is not an object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _artifact_path(root: Path, relative: str) -> Path:
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise TemporalPretrainError(f"missing or escaping alpha artifact: {relative}") from exc
    if not path.is_file():
        raise TemporalPretrainError(f"alpha artifact is not a file: {relative}")
    return path


def _artifact_reference_matches(value: Any, expected_parent: str, expected_name: str) -> bool:
    if not isinstance(value, str):
        return False
    reference = Path(value)
    return reference.name == expected_name and reference.parent.name == expected_parent


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TemporalPretrainError(f"alpha {label} record is missing or malformed")
    return value


def temporal_parameter_names(backbone: nn.Module) -> tuple[str, ...]:
    """Return the only released-backbone parameters this stage may update."""

    blocks = getattr(backbone, "temporal_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) < 2:
        raise TemporalPretrainError("released backbone does not expose temporal blocks")
    names = tuple(name for name, _ in backbone.named_parameters() if name.startswith("temporal_blocks."))
    if not names:
        raise TemporalPretrainError("released backbone has no temporal parameters")
    return names


def set_temporal_only_training(backbone: nn.Module) -> dict[str, int]:
    """Freeze the spatial released core and enable only temporal-attention weights."""

    allowed = set(temporal_parameter_names(backbone))
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad = name in allowed
    trainable_names = {name for name, parameter in backbone.named_parameters() if parameter.requires_grad}
    if trainable_names != allowed:
        raise TemporalPretrainError("temporal-only parameter policy drift")
    total = sum(parameter.numel() for parameter in backbone.parameters())
    trainable = sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
    if trainable <= 0 or trainable >= total:
        raise TemporalPretrainError("temporal-only parameter accounting is invalid")
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


class ReleasedTemporalStudent(nn.Module):
    """Released-initialized backbone with a cheap temporal-parameter trust region."""

    def __init__(self, released_backbone: nn.Module) -> None:
        super().__init__()
        self.student = released_backbone
        set_temporal_only_training(self.student)
        self._released_names: list[str] = []
        for index, (name, parameter) in enumerate(
            (item for item in self.student.named_parameters() if item[1].requires_grad)
        ):
            self._released_names.append(name)
            self.register_buffer(f"_released_temporal_{index}", parameter.detach().clone(), persistent=False)
        self.train(True)

    def train(self, mode: bool = True) -> "ReleasedTemporalStudent":
        # Temporal blocks use LayerNorm/MHA.  BatchNorm running state belongs to
        # the released spatial core and is therefore held fixed.
        super().train(mode)
        self.student.train(mode)
        for module in self.student.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 6
            or frames.shape[1:3] != (2, 1)
            or not _finite_tensor(frames)
        ):
            raise TemporalPretrainError("pretraining frames must be finite (B,2,1,Z,Y,X) tensors")
        features = self.student(frames)
        if features.ndim != 6 or features.shape[:2] != frames.shape[:2] or features.shape[3:] != frames.shape[3:]:
            raise TemporalPretrainError("released student feature schema mismatch")
        return features

    def release_anchor_loss(self) -> torch.Tensor:
        parameters = dict(self.student.named_parameters())
        squared = None
        count = 0
        for index, name in enumerate(self._released_names):
            parameter = parameters[name]
            released = getattr(self, f"_released_temporal_{index}")
            term = (parameter - released).square().sum()
            squared = term if squared is None else squared + term
            count += parameter.numel()
        if squared is None or count <= 0:
            raise TemporalPretrainError("released temporal anchor has no parameters")
        return squared / count


def _validate_sparse_coordinates(
    features: torch.Tensor,
    batch_indices: torch.Tensor,
    coords: torch.Tensor,
    label: str,
) -> None:
    if batch_indices.dtype != torch.long or coords.dtype != torch.long:
        raise TemporalPretrainError(f"{label} indices must use torch.long")
    if batch_indices.device != features.device or coords.device != features.device:
        raise TemporalPretrainError(f"{label} indices must share the feature device")
    if batch_indices.ndim != 1 or coords.shape != (len(batch_indices), 3):
        raise TemporalPretrainError(f"{label} coordinates must have shape (K,3)")
    if len(batch_indices) == 0:
        raise TemporalPretrainError(f"{label} coordinates are empty")
    spatial = torch.as_tensor(features.shape[3:], device=features.device, dtype=torch.long)
    if bool((batch_indices < 0).any()) or bool((batch_indices >= features.shape[0]).any()):
        raise TemporalPretrainError(f"{label} batch index is out of bounds")
    if bool((coords < 0).any()) or bool((coords >= spatial[None]).any()):
        raise TemporalPretrainError(f"{label} coordinate is out of bounds")


def _gather(features: torch.Tensor, batch_indices: torch.Tensor, frame: int, coords: torch.Tensor) -> torch.Tensor:
    z, y, x = coords.unbind(dim=1)
    return features[batch_indices, frame, :, z, y, x]


def sparse_migration_loss(
    model: ReleasedTemporalStudent,
    features: torch.Tensor,
    *,
    link_batch_indices: torch.Tensor,
    source_zyx: torch.Tensor,
    target_zyx: torch.Tensor,
    link_weights: torch.Tensor,
    margin: float,
    release_anchor_weight: float,
    division_batch_indices: torch.Tensor | None = None,
    division_zyx: torch.Tensor | None = None,
    division_weights: torch.Tensor | None = None,
    division_weight: float = 0.0,
    sibling_cosine_ceiling: float = 0.9,
) -> dict[str, torch.Tensor]:
    """Rank declared migrations above stale same-coordinate alternatives.

    Work beyond the dense backbone forward is ``O((K+D)*C)`` for ``K`` links,
    ``D`` optional division motifs, and feature width ``C``.  No candidates are
    enumerated and no ``K x K`` or spatial pair tensor is constructed.
    """

    if features.ndim != 6 or features.shape[1] != 2 or not _finite_tensor(features):
        raise TemporalPretrainError("migration features must be finite (B,2,C,Z,Y,X)")
    _validate_sparse_coordinates(features, link_batch_indices, source_zyx, "source")
    _validate_sparse_coordinates(features, link_batch_indices, target_zyx, "target")
    if bool(torch.all(source_zyx == target_zyx, dim=1).any()):
        raise TemporalPretrainError("migration links must not be same-coordinate pairs")
    if (
        link_weights.shape != (len(link_batch_indices),)
        or link_weights.device != features.device
        or not link_weights.is_floating_point()
        or not _finite_tensor(link_weights)
        or bool((link_weights <= 0).any())
    ):
        raise TemporalPretrainError("migration link weights must be positive finite values")
    if not math.isfinite(margin) or not 0.0 < margin <= 2.0:
        raise TemporalPretrainError("migration margin must be finite and in (0,2]")
    if not math.isfinite(release_anchor_weight) or release_anchor_weight < 0:
        raise TemporalPretrainError("release anchor weight must be finite and nonnegative")
    if not math.isfinite(division_weight) or division_weight < 0:
        raise TemporalPretrainError("division weight must be finite and nonnegative")
    if not math.isfinite(sibling_cosine_ceiling) or not -1.0 <= sibling_cosine_ceiling <= 1.0:
        raise TemporalPretrainError("sibling cosine ceiling must be finite and in [-1,1]")

    source = F.normalize(_gather(features, link_batch_indices, 0, source_zyx), dim=1)
    target = F.normalize(_gather(features, link_batch_indices, 1, target_zyx), dim=1)
    stale_target = F.normalize(_gather(features, link_batch_indices, 1, source_zyx), dim=1)
    stale_source = F.normalize(_gather(features, link_batch_indices, 0, target_zyx), dim=1)
    positive_similarity = (source * target).sum(dim=1)
    forward_stale = (source * stale_target).sum(dim=1)
    reverse_stale = (target * stale_source).sum(dim=1)
    terms = 0.5 * (
        F.relu(margin + forward_stale - positive_similarity)
        + F.relu(margin + reverse_stale - positive_similarity)
    )
    migration = (terms * link_weights).sum() / link_weights.sum()
    positive = (positive_similarity * link_weights).sum() / link_weights.sum()
    stale = (0.5 * (forward_stale + reverse_stale) * link_weights).sum() / link_weights.sum()

    division = features.sum() * 0.0
    division_count = 0
    division_values = (division_batch_indices, division_zyx, division_weights)
    if division_weight > 0:
        if any(value is None for value in division_values):
            raise TemporalPretrainError("enabled division objective requires division tensors")
        assert division_batch_indices is not None and division_zyx is not None and division_weights is not None
        if division_zyx.ndim != 3 or division_zyx.shape[1:] != (3, 3):
            raise TemporalPretrainError("division coordinates must have shape (D,3,3)")
        if division_zyx.dtype != torch.long or division_zyx.device != features.device:
            raise TemporalPretrainError("division coordinates must be same-device torch.long")
        if division_batch_indices.shape != (len(division_zyx),):
            raise TemporalPretrainError("division batch indices have incompatible shape")
        if (
            division_weights.shape != (len(division_zyx),)
            or division_weights.device != features.device
            or not division_weights.is_floating_point()
            or not _finite_tensor(division_weights)
            or (len(division_weights) > 0 and bool((division_weights <= 0).any()))
        ):
            raise TemporalPretrainError("division weights must be positive finite values")
        if len(division_zyx) > 0:
            for index, label in enumerate(("division parent", "division child one", "division child two")):
                _validate_sparse_coordinates(features, division_batch_indices, division_zyx[:, index], label)
            parent = F.normalize(_gather(features, division_batch_indices, 0, division_zyx[:, 0]), dim=1)
            child_one = F.normalize(_gather(features, division_batch_indices, 1, division_zyx[:, 1]), dim=1)
            child_two = F.normalize(_gather(features, division_batch_indices, 1, division_zyx[:, 2]), dim=1)
            centroid = F.normalize(child_one + child_two, dim=1)
            parent_centroid = 1.0 - (parent * centroid).sum(dim=1)
            sibling_separation = F.relu((child_one * child_two).sum(dim=1) - sibling_cosine_ceiling)
            division_terms = parent_centroid + sibling_separation
            division = (division_terms * division_weights).sum() / division_weights.sum()
            division_count = len(division_zyx)

    release_anchor = model.release_anchor_loss()
    total = migration + division_weight * division + release_anchor_weight * release_anchor
    values = (total, migration, division, release_anchor, positive, stale)
    if not all(_finite_tensor(value) for value in values):
        raise TemporalPretrainError("sparse migration objective is nonfinite")
    return {
        "total": total,
        "migration": migration,
        "division": division,
        "release_anchor": release_anchor,
        "positive_similarity": positive,
        "stale_similarity": stale,
        "link_count": torch.as_tensor(float(len(link_batch_indices)), device=features.device),
        "division_count": torch.as_tensor(float(division_count), device=features.device),
    }


def _validate_alpha_contract(
    contract: Mapping[str, Any],
    *,
    expected_exact_identity_sha256: str,
) -> Mapping[str, Any]:
    if contract.get("schema") != ALPHA_CACHE_META_SCHEMA:
        raise TemporalPretrainError("alpha inference contract schema mismatch")
    unsigned = {key: value for key, value in contract.items() if key != "sha256"}
    if contract.get("sha256") != _canonical_sha256(unsigned):
        raise TemporalPretrainError("alpha inference contract digest mismatch")
    if contract.get("data_exact_identity_sha256") != expected_exact_identity_sha256:
        raise TemporalPretrainError("alpha inference data identity mismatch")
    inference = _required_mapping(contract.get("inference"), "inference")
    if (
        inference.get("api") != "bh_model.infer_video"
        or inference.get("learned_dense_transformer") is not False
        or not isinstance(inference.get("frame_count"), int)
        or inference.get("frame_count", 0) < 2
    ):
        raise TemporalPretrainError("alpha inference is not the sparse adjacent-frame path")
    model = _required_mapping(contract.get("model"), "model")
    for key in ("config_sha256", "source_sha256", "weight_sha256"):
        if not _is_sha256(model.get(key)):
            raise TemporalPretrainError("alpha model identity is incomplete")
    return model


def load_alpha_pseudo_link_artifact(
    artifact_root: str | Path,
    *,
    group_id: str,
    video_id: str,
    stem: str,
    expected_exact_identity_sha256: str,
    min_migration_um: float,
    min_link_confidence: float,
    max_links_per_window: int,
    max_divisions_per_window: int,
) -> AlphaPseudoLinkArtifact:
    """Validate one persisted alpha bundle and project it to bounded links."""

    if video_id != f"train:{stem}" or not group_id.startswith("g02-"):
        raise TemporalPretrainError("alpha pseudo-link source is not a canonical train source")
    if not _is_sha256(expected_exact_identity_sha256):
        raise TemporalPretrainError("registry exact image identity is invalid")
    if not math.isfinite(min_migration_um) or min_migration_um <= 0:
        raise TemporalPretrainError("minimum migration must be positive and finite")
    if not math.isfinite(min_link_confidence) or not 0.0 < min_link_confidence <= 1.0:
        raise TemporalPretrainError("minimum link confidence must be in (0,1]")
    if max_links_per_window < 1 or max_divisions_per_window < 1:
        raise TemporalPretrainError("pseudo-link and division caps must be positive")
    try:
        root = Path(artifact_root).resolve(strict=True)
    except OSError as exc:
        raise TemporalPretrainError("alpha artifact root does not exist") from exc
    if not root.is_dir():
        raise TemporalPretrainError("alpha artifact root is not a directory")
    slug = f"train__{stem}"
    paths = {
        "cache": _artifact_path(root, f"cache/{slug}.bhc"),
        "meta": _artifact_path(root, f"cache/{slug}.json"),
        "intent": _artifact_path(root, f"cache/{slug}.intent.json"),
        "fragment": _artifact_path(root, f"fragments/{slug}.csv"),
        "feature": _artifact_path(root, f"features/{slug}.json"),
        "status": _artifact_path(root, f"status/{slug}.json"),
        "transaction": _artifact_path(root, f"transactions/{slug}.json"),
    }
    meta, intent = _read_json(paths["meta"]), _read_json(paths["intent"])
    feature, status = _read_json(paths["feature"]), _read_json(paths["status"])
    transaction = _read_json(paths["transaction"])
    contract = _required_mapping(meta.get("contract"), "cache contract")
    model_identity = _validate_alpha_contract(
        contract, expected_exact_identity_sha256=expected_exact_identity_sha256
    )
    if (
        meta.get("schema") != ALPHA_CACHE_META_SCHEMA
        or meta.get("cache_sha256") != _file_sha256(paths["cache"])
        or intent.get("schema") != ALPHA_CACHE_INTENT_SCHEMA
        or intent.get("identity") != video_id
        or intent.get("contract") != contract
    ):
        raise TemporalPretrainError("alpha cache metadata/intent mismatch")
    intent_artifacts = _required_mapping(intent.get("artifacts"), "cache artifact paths")
    if not (
        _artifact_reference_matches(intent_artifacts.get("cache"), "cache", f"{slug}.bhc")
        and _artifact_reference_matches(intent_artifacts.get("meta"), "cache", f"{slug}.json")
    ):
        raise TemporalPretrainError("alpha cache intent paths mismatch")

    run_contract = _required_mapping(status.get("run_contract"), "run contract")
    if (
        run_contract.get("schema") != ALPHA_RUNNER_SCHEMA
        or run_contract.get("sha256")
        != _canonical_sha256({key: value for key, value in run_contract.items() if key != "sha256"})
        or run_contract.get("inference_sha256") != contract.get("sha256")
        or run_contract.get("graph_variant") != "a0"
    ):
        raise TemporalPretrainError("alpha run contract is not a valid A0 sparse run")
    graph_config = _required_mapping(run_contract.get("graph_config"), "graph config")
    candidate_gate = _required_mapping(graph_config.get("candidate_gate"), "candidate gate")
    scale_raw = candidate_gate.get("scale_zyx")
    if (
        graph_config.get("locked") is not True
        or graph_config.get("full_frame_dense_matrix") is not False
        or candidate_gate.get("algorithm") != "physical-cKDTree"
        or candidate_gate.get("full_frame_dense_matrix", False) is not False
        or not isinstance(scale_raw, list)
        or len(scale_raw) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0 for value in scale_raw)
    ):
        raise TemporalPretrainError("alpha graph contract violates the sparse physical gate")
    scale_zyx = tuple(float(value) for value in scale_raw)

    for payload, schema, label in (
        (status, ALPHA_STATUS_SCHEMA, "status"),
        (feature, ALPHA_FEATURE_SCHEMA, "feature"),
        (transaction, ALPHA_TRANSACTION_SCHEMA, "transaction"),
    ):
        if payload.get("schema") != schema or payload.get("identity") != video_id or payload.get("run_contract") != run_contract:
            raise TemporalPretrainError(f"alpha {label} provenance mismatch")
    if status.get("ok") is not True:
        raise TemporalPretrainError("alpha status is not complete")
    if transaction.get("inference_contract") != contract or feature.get("graph_variant") != "a0":
        raise TemporalPretrainError("alpha transaction/inference provenance mismatch")
    status_result = _required_mapping(status.get("result"), "status result")
    if (
        status_result.get("identity") != video_id
        or status_result.get("dataset") != stem
        or status_result.get("graph_variant") != "a0"
        or status_result.get("cache_contract_sha256") != contract.get("sha256")
        or status_result.get("run_contract_sha256") != run_contract.get("sha256")
        or not _artifact_reference_matches(status_result.get("fragment"), "fragments", f"{slug}.csv")
        or not _artifact_reference_matches(status_result.get("feature_summary"), "features", f"{slug}.json")
    ):
        raise TemporalPretrainError("alpha status result provenance mismatch")
    fragment_sha256 = _file_sha256(paths["fragment"])
    feature_sha256 = _file_sha256(paths["feature"])
    if status.get("fragment_sha256") != fragment_sha256 or status.get("feature_sha256") != feature_sha256:
        raise TemporalPretrainError("alpha fragment/feature digest mismatch")
    transaction_artifacts = _required_mapping(transaction.get("artifacts"), "transaction artifacts")
    if not (
        _artifact_reference_matches(transaction_artifacts.get("fragment"), "fragments", f"{slug}.csv")
        and _artifact_reference_matches(transaction_artifacts.get("feature"), "features", f"{slug}.json")
        and _artifact_reference_matches(transaction_artifacts.get("status"), "status", f"{slug}.json")
    ):
        raise TemporalPretrainError("alpha transaction paths mismatch")
    audit = _required_mapping(status.get("a0_audit"), "A0 audit")
    audit_cache = _required_mapping(audit.get("cache"), "A0 cache audit")
    audit_graph = _required_mapping(audit.get("graph"), "A0 graph audit")
    audit_gate = _required_mapping(audit.get("gate"), "A0 gate audit")
    audit_probabilities = _required_mapping(audit.get("probabilities"), "A0 probability audit")
    if (
        feature.get("a0_audit") != audit
        or audit_cache.get("complete_adjacent_coverage") is not True
        or audit_graph.get("full_frame_dense_matrix") is not False
        or audit_graph.get("variant") != "a0"
        or audit_gate.get("full_frame_dense_matrix") is not False
        or audit_probabilities.get("finite_unit_interval") is not True
        or audit_probabilities.get("scope") != "original-candidate-coo-only"
    ):
        raise TemporalPretrainError("alpha A0 audit does not prove sparse finite inference")

    from .bh_model import CacheIdentity, load_sparse_cache

    try:
        cache_identity = CacheIdentity(**dict(model_identity))
        cache = load_sparse_cache(paths["cache"], expected_identity=cache_identity)
    except Exception as exc:
        raise TemporalPretrainError("alpha sparse cache validation failed") from exc
    frame_count = int(_required_mapping(contract.get("inference"), "inference")["frame_count"])
    if len(cache.detections) != frame_count or len(cache.contexts) != frame_count - 1 or len(cache.edges) != 0:
        raise TemporalPretrainError("alpha cache coverage or no-dense-edge contract mismatch")

    candidates: dict[int, tuple[int, tuple[int, int, int], tuple[int, int, int], float]] = {}
    for frame in cache.detections:
        for index, raw_id in enumerate(frame.node_ids):
            node_id = int(raw_id)
            if node_id in candidates:
                raise TemporalPretrainError("alpha cache contains duplicate candidate IDs")
            candidates[node_id] = (
                frame.frame_index,
                tuple(int(value) for value in frame.grid_coords_zyx[index]),
                tuple(int(value) for value in frame.coords_zyx[index]),
                float(frame.scores[index]),
            )

    fragment_nodes: dict[int, int] = {}
    fragment_edges: list[tuple[int, int]] = []
    row_ids: set[int] = set()
    try:
        with paths["fragment"].open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ALPHA_CSV_COLUMNS:
                raise TemporalPretrainError("alpha fragment column schema mismatch")
            for row in reader:
                row_id = int(row["id"])
                if row_id in row_ids or row["dataset"] != stem:
                    raise TemporalPretrainError("alpha fragment row identity mismatch")
                row_ids.add(row_id)
                if row["row_type"] == "node":
                    node_id, frame_index = int(row["node_id"]), int(row["t"])
                    if node_id < 0 or frame_index < 0 or node_id in fragment_nodes:
                        raise TemporalPretrainError("alpha fragment node is invalid or duplicate")
                    fragment_nodes[node_id] = frame_index
                elif row["row_type"] == "edge":
                    fragment_edges.append((int(row["source_id"]), int(row["target_id"])))
                else:
                    raise TemporalPretrainError("alpha fragment row type is invalid")
    except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as exc:
        if isinstance(exc, TemporalPretrainError):
            raise
        raise TemporalPretrainError("alpha graph fragment is malformed") from exc
    if not fragment_nodes or not fragment_edges:
        raise TemporalPretrainError("alpha graph fragment has no nodes or edges")
    if (
        status_result.get("frames") != frame_count
        or status_result.get("nodes") != len(fragment_nodes)
        or status_result.get("edges") != len(fragment_edges)
        or status_result.get("rows") != len(row_ids)
    ):
        raise TemporalPretrainError("alpha status result counts disagree with the fragment")
    seen_edges: set[tuple[int, int]] = set()
    incoming: dict[int, int] = {}
    outgoing: dict[int, list[int]] = {}
    for edge in fragment_edges:
        source_id, target_id = edge
        if (
            edge in seen_edges
            or source_id not in fragment_nodes
            or target_id not in fragment_nodes
            or fragment_nodes[target_id] != fragment_nodes[source_id] + 1
        ):
            raise TemporalPretrainError("alpha fragment edge topology is invalid")
        seen_edges.add(edge)
        incoming[target_id] = incoming.get(target_id, 0) + 1
        outgoing.setdefault(source_id, []).append(target_id)
    if any(value > 1 for value in incoming.values()) or any(len(value) > 2 for value in outgoing.values()):
        raise TemporalPretrainError("alpha fragment contains a merge or out-degree above two")

    links_by_frame: dict[int, list[PseudoLink]] = {}
    candidate_backed = 0
    migrating = 0
    for source_id, target_id in fragment_edges:
        if source_id not in candidates or target_id not in candidates:
            continue
        candidate_backed += 1
        source_frame, source_grid, source_voxel, source_score = candidates[source_id]
        target_frame, target_grid, target_voxel, target_score = candidates[target_id]
        if target_frame != source_frame + 1 or fragment_nodes[source_id] != source_frame or fragment_nodes[target_id] != target_frame:
            raise TemporalPretrainError("alpha cache and fragment time identity disagree")
        distance_um = math.sqrt(sum(
            ((float(target_voxel[index]) - float(source_voxel[index])) * scale_zyx[index]) ** 2
            for index in range(3)
        ))
        confidence = min(source_score, target_score)
        if source_grid == target_grid or distance_um <= min_migration_um or confidence < min_link_confidence:
            continue
        migrating += 1
        links_by_frame.setdefault(source_frame, []).append(PseudoLink(
            source_id,
            target_id,
            source_grid,
            target_grid,
            confidence,
            distance_um,
        ))

    divisions_by_frame: dict[int, list[DivisionMotif]] = {}
    candidate_backed_divisions = 0
    for parent_id, child_ids_raw in outgoing.items():
        if len(child_ids_raw) != 2 or parent_id not in candidates or any(child not in candidates for child in child_ids_raw):
            continue
        child_ids = tuple(sorted(child_ids_raw))
        parent_frame, parent_grid, _parent_voxel, parent_score = candidates[parent_id]
        child_records = (candidates[child_ids[0]], candidates[child_ids[1]])
        if any(record[0] != parent_frame + 1 for record in child_records):
            raise TemporalPretrainError("alpha division motif is not adjacent")
        child_grids = (child_records[0][1], child_records[1][1])
        if child_grids[0] == child_grids[1]:
            raise TemporalPretrainError("alpha division children share a grid coordinate")
        confidence = min(parent_score, child_records[0][3], child_records[1][3])
        if confidence < min_link_confidence:
            continue
        candidate_backed_divisions += 1
        divisions_by_frame.setdefault(parent_frame, []).append(DivisionMotif(
            parent_id,
            child_ids,
            parent_grid,
            child_grids,
            confidence,
        ))

    windows: list[PseudoLinkWindow] = []
    retained_links = 0
    retained_divisions = 0
    for start in sorted(set(links_by_frame) | set(divisions_by_frame)):
        links = tuple(sorted(
            links_by_frame.get(start, ()),
            key=lambda item: (-item.confidence, item.source_id, item.target_id),
        )[:max_links_per_window])
        divisions = tuple(sorted(
            divisions_by_frame.get(start, ()),
            key=lambda item: (-item.confidence, item.parent_id, item.child_ids),
        )[:max_divisions_per_window])
        if links:
            windows.append(PseudoLinkWindow(start, links, divisions))
            retained_links += len(links)
            retained_divisions += len(divisions)
    if not windows or retained_links == 0:
        raise TemporalPretrainError("alpha artifact has no retained migrating pseudo-links")
    summary = {
        "frame_count": frame_count,
        "candidate_count": len(candidates),
        "selected_edge_count": len(fragment_edges),
        "candidate_backed_edge_count": candidate_backed,
        "eligible_migration_count": migrating,
        "retained_migration_count": retained_links,
        "candidate_backed_division_count": candidate_backed_divisions,
        "retained_division_count": retained_divisions,
        "retained_window_count": len(windows),
        "maximum_links_per_window": max_links_per_window,
        "maximum_divisions_per_window": max_divisions_per_window,
    }
    return AlphaPseudoLinkArtifact(
        group_id=group_id,
        video_id=video_id,
        stem=stem,
        exact_identity_sha256=expected_exact_identity_sha256,
        cache_sha256=_file_sha256(paths["cache"]),
        fragment_sha256=fragment_sha256,
        status_sha256=_file_sha256(paths["status"]),
        feature_sha256=feature_sha256,
        transaction_sha256=_file_sha256(paths["transaction"]),
        run_contract_sha256=str(run_contract["sha256"]),
        alpha_weight_sha256=str(model_identity["weight_sha256"]),
        windows=tuple(windows),
        summary=summary,
    )


def _source_identity(
    identity: Mapping[str, Any],
    allowed_fit_groups: set[str] | None,
    allowed_fit_video_ids: set[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if identity.get("input_policy") != PRETRAIN_INPUT_POLICY or identity.get("pairing_policy") != PRETRAIN_PAIRING_POLICY:
        raise TemporalPretrainError("pretraining input/pairing policy mismatch")
    records, artifacts = identity.get("selected_sources"), identity.get("alpha_artifacts")
    if not isinstance(records, list) or not records or not isinstance(artifacts, list) or len(artifacts) != len(records):
        raise TemporalPretrainError("pretraining checkpoint has no one-to-one source/artifact provenance")
    groups: list[str] = []
    videos: list[str] = []
    artifact_scope: set[tuple[str, str]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise TemporalPretrainError("pretraining alpha artifact provenance is malformed")
        group, video = artifact.get("group_id"), artifact.get("video_id")
        hashes = (
            artifact.get("exact_identity_sha256"),
            artifact.get("cache_sha256"),
            artifact.get("fragment_sha256"),
            artifact.get("status_sha256"),
            artifact.get("feature_sha256"),
            artifact.get("transaction_sha256"),
            artifact.get("run_contract_sha256"),
            artifact.get("alpha_weight_sha256"),
        )
        if not isinstance(group, str) or not isinstance(video, str) or not all(_is_sha256(value) for value in hashes):
            raise TemporalPretrainError("pretraining alpha artifact identity is incomplete")
        artifact_scope.add((group, video))
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"group_id", "video_id", "stem"}:
            raise TemporalPretrainError("pretraining selected-source schema mismatch")
        group, video, stem = record["group_id"], record["video_id"], record["stem"]
        if (
            not isinstance(group, str)
            or not isinstance(video, str)
            or not isinstance(stem, str)
            or video != f"train:{stem}"
            or (group, video) not in artifact_scope
            or (allowed_fit_groups is not None and group not in allowed_fit_groups)
            or (allowed_fit_video_ids is not None and video not in allowed_fit_video_ids)
        ):
            raise TemporalPretrainError("pretraining source is not fit-only competition train imagery")
        groups.append(group)
        videos.append(video)
    if len(groups) != len(set(groups)) or len(videos) != len(set(videos)) or len(artifact_scope) != len(records):
        raise TemporalPretrainError("pretraining source/artifact provenance is duplicate")
    return tuple(groups), tuple(videos)


def load_temporal_pretrain_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_fold: int,
    expected_manifest_hashes: Mapping[str, str],
    allowed_fit_groups: set[str] | None = None,
    allowed_fit_video_ids: set[str] | None = None,
    expected_sha256: str | None = None,
) -> TemporalPretrainLoad:
    """Strictly restore a fold-bound alpha-pseudolink backbone for fine-tuning."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise TemporalPretrainError(f"missing temporal pretraining checkpoint: {path}")
    observed_sha256 = _file_sha256(path)
    if expected_sha256 is not None and not hmac.compare_digest(observed_sha256, expected_sha256):
        raise TemporalPretrainError("temporal pretraining checkpoint hash mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TemporalPretrainError("weights-only temporal checkpoint load failed") from exc
    required = {"schema", "identity", "backbone", "optimizer", "progress", "metrics"}
    if not isinstance(payload, Mapping) or set(payload) != required or payload.get("schema") != PRETRAIN_SCHEMA:
        raise TemporalPretrainError("temporal pretraining checkpoint schema mismatch")
    if not _finite_tree(payload):
        raise TemporalPretrainError("temporal pretraining checkpoint contains nonfinite state")
    identity, state, progress = payload.get("identity"), payload.get("backbone"), payload.get("progress")
    if not isinstance(identity, Mapping) or not isinstance(state, Mapping) or not isinstance(progress, Mapping):
        raise TemporalPretrainError("temporal pretraining checkpoint content is malformed")
    if identity.get("objective") != PRETRAIN_OBJECTIVE or identity.get("fold") != expected_fold:
        raise TemporalPretrainError("temporal pretraining objective/fold mismatch")
    immutable = {
        "split_sha256": identity.get("split_sha256"),
        "provenance_sha256": identity.get("provenance_sha256"),
        "registry_sha256": identity.get("registry_sha256"),
    }
    if immutable != dict(expected_manifest_hashes):
        raise TemporalPretrainError("temporal pretraining split binding mismatch")
    if identity.get("released_weights_sha256") != WEIGHTS_SHA256 or identity.get("released_config_sha256") != CONFIG_SHA256:
        raise TemporalPretrainError("temporal pretraining released-parent binding mismatch")
    if identity.get("released_config") != EXPECTED_CONFIG or identity.get("trainable_scope") != "temporal_blocks_only":
        raise TemporalPretrainError("temporal pretraining released configuration/scope mismatch")
    if (allowed_fit_groups is None) != (allowed_fit_video_ids is None):
        raise TemporalPretrainError("fit group and video scopes must be supplied together")
    groups, videos = _source_identity(identity, allowed_fit_groups, allowed_fit_video_ids)
    steps = progress.get("completed_steps")
    windows = progress.get("completed_windows")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or windows != steps:
        raise TemporalPretrainError("temporal pretraining checkpoint has invalid progress")
    expected_names = tuple(backbone.state_dict())
    if tuple(state) != expected_names:
        raise TemporalPretrainError("temporal pretraining backbone tensor schema mismatch")
    released_state = backbone.state_dict()
    allowed_changes = set(temporal_parameter_names(backbone))
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or not _finite_tensor(value):
            raise TemporalPretrainError("temporal pretraining backbone contains an invalid tensor")
        if name not in allowed_changes and not torch.equal(value.detach().cpu(), released_state[name].detach().cpu()):
            raise TemporalPretrainError("temporal pretraining checkpoint changed the frozen released spatial core")
    try:
        backbone.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise TemporalPretrainError("strict temporal-pretrained backbone restore failed") from exc
    return TemporalPretrainLoad(
        checkpoint_sha256=observed_sha256,
        fold=expected_fold,
        completed_steps=steps,
        source_groups=groups,
        source_video_ids=videos,
        backbone_tensor_count=len(state),
        backbone_numel=sum(value.numel() for value in state.values()),
    )
