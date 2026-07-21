"""Bounded Mean-Teacher pseudo labels for the sparse G-02 tracker.

The module deliberately operates on point and COO-like edge records.  It never
constructs an object-by-object or frame-by-frame dense association matrix.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SSL_SCHEMA = "celltrack-alpha.pseudo-ssl.v1"
TASKS = ("detection", "continuation", "division")
PERMITTED_PARAMETER_PREFIXES = (
    "detect_head.",
    "aux_head.",
    "specialists.",
    "backbone.head.",
    "backbone.decoder_blocks.",
    "backbone.temporal_blocks.",
    "student.temporal_blocks.",
    "temporal_blocks.",
)


class PseudoSSLError(ValueError):
    """Raised when SSL scope, geometry, or sparse-label invariants fail."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_tensor(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(
        torch.isfinite(value).all().item()
    )


def _load_json(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    if not path.is_file():
        raise PseudoSSLError(f"missing provenance manifest: {path}")
    observed = _sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise PseudoSSLError(f"immutable provenance manifest mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PseudoSSLError(f"invalid provenance manifest: {path}") from exc
    if not isinstance(value, dict):
        raise PseudoSSLError(f"provenance manifest is not an object: {path}")
    return value


@dataclass(frozen=True)
class FitOnlyScope:
    """Immutable fold membership used before opening any pseudo-label source."""

    fold: int
    split_sha256: str
    provenance_sha256: str
    registry_sha256: str
    fit_group_ids: frozenset[str]
    fit_video_ids: frozenset[str]
    forbidden_group_ids: frozenset[str]
    forbidden_video_ids: frozenset[str]
    exact_image_identity: Mapping[str, str]
    video_to_group: Mapping[str, str]

    @classmethod
    def from_files(
        cls,
        fold: int,
        split_path: str | Path,
        provenance_path: str | Path,
        registry_path: str | Path,
        *,
        expected_hashes: Mapping[str, str] | None = None,
    ) -> "FitOnlyScope":
        if fold not in (0, 1, 2):
            raise PseudoSSLError("fold must be 0, 1, or 2")
        paths = {
            "split_sha256": Path(split_path),
            "provenance_sha256": Path(provenance_path),
            "registry_sha256": Path(registry_path),
        }
        expected = dict(expected_hashes or {})
        payloads = {
            key: _load_json(path, expected.get(key)) for key, path in paths.items()
        }
        hashes = {key: _sha256_file(path) for key, path in paths.items()}
        split = payloads["split_sha256"]
        provenance = payloads["provenance_sha256"]
        registry = payloads["registry_sha256"]
        try:
            split_fold = split["folds"][str(fold)]
            provenance_fold = provenance["folds"][str(fold)]
            fit_groups = frozenset(split_fold["fit_group_ids"])
            fit_videos = frozenset(split_fold["fit_video_ids"])
            held_groups = frozenset(split_fold["held_out_group_ids"])
            held_videos = frozenset(split_fold["held_out_video_ids"])
            audit_groups = frozenset(split["audit"]["group_ids"])
            audit_videos = frozenset(split["audit"]["video_ids"])
            forbidden_groups = frozenset(provenance_fold["forbidden_fit_group_ids"])
            fit_sources = provenance_fold["fit_sources"]
            records = registry["records"]
        except (KeyError, TypeError) as exc:
            raise PseudoSSLError("fit-only manifest schema is incomplete") from exc
        if (
            not fit_groups
            or not fit_videos
            or fit_groups & forbidden_groups
            or fit_groups & held_groups
            or fit_groups & audit_groups
            or not held_groups | audit_groups <= forbidden_groups
            or not isinstance(fit_sources, Mapping)
            or any(frozenset(fit_sources.get(role, ())) != fit_groups for role in ("training", "calibration", "router", "threshold"))
        ):
            raise PseudoSSLError("fit-only group isolation invariant failed")
        exact: dict[str, str] = {}
        video_to_group: dict[str, str] = {}
        if not isinstance(records, list):
            raise PseudoSSLError("registry records are not a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise PseudoSSLError("registry record is malformed")
            video_id = record.get("video_id")
            group_id = record.get("group_id")
            image_hash = record.get("exact_identity_sha256")
            if not isinstance(video_id, str) or not isinstance(group_id, str) or not _is_sha256(image_hash):
                raise PseudoSSLError("registry record lacks a canonical image identity")
            if video_id in exact:
                raise PseudoSSLError("registry contains duplicate video identities")
            exact[video_id] = str(image_hash)
            video_to_group[video_id] = group_id
        if (
            set(fit_videos) - set(exact)
            or {video_to_group[video] for video in fit_videos} != set(fit_groups)
            or fit_videos & (held_videos | audit_videos)
        ):
            raise PseudoSSLError("registry video mapping disagrees with fold membership")
        return cls(
            fold=fold,
            split_sha256=hashes["split_sha256"],
            provenance_sha256=hashes["provenance_sha256"],
            registry_sha256=hashes["registry_sha256"],
            fit_group_ids=fit_groups,
            fit_video_ids=fit_videos,
            forbidden_group_ids=forbidden_groups,
            forbidden_video_ids=frozenset(held_videos | audit_videos),
            exact_image_identity=exact,
            video_to_group=video_to_group,
        )

    def validate(self, group_id: str, video_id: str, exact_identity_sha256: str) -> None:
        if (
            not video_id.startswith("train:")
            or group_id not in self.fit_group_ids
            or video_id not in self.fit_video_ids
            or group_id in self.forbidden_group_ids
            or video_id in self.forbidden_video_ids
            or self.video_to_group.get(video_id) != group_id
            or self.exact_image_identity.get(video_id) != exact_identity_sha256
        ):
            raise PseudoSSLError("pseudo-label source is not an exact fit-only train video")


@dataclass(frozen=True)
class BatchProvenance:
    group_id: str
    video_id: str
    exact_identity_sha256: str
    teacher_sha256: str
    source_role: str = "fit-unlabeled-images"

    def validate(self, scope: FitOnlyScope) -> None:
        if self.source_role != "fit-unlabeled-images" or not _is_sha256(self.teacher_sha256):
            raise PseudoSSLError("pseudo-label provenance role or teacher identity is invalid")
        scope.validate(self.group_id, self.video_id, self.exact_identity_sha256)


def permitted_trainable_names(model: nn.Module) -> tuple[str, ...]:
    """Validate and return the exact compact parameter partition eligible for EMA."""

    names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if not names:
        raise PseudoSSLError("Mean Teacher has no trainable adapter/temporal parameters")
    disallowed = [
        name for name in names if not any(name.startswith(prefix) for prefix in PERMITTED_PARAMETER_PREFIXES)
    ]
    if disallowed:
        raise PseudoSSLError(f"disallowed Mean-Teacher trainable parameter: {disallowed[0]}")
    return names


class EMATeacherState:
    """EMA shadows for the permitted trainable partition only.

    Frozen checkpoint parameters and buffers are intentionally absent.  Teacher
    inference temporarily swaps the shadows into the student module, avoiding a
    second copy of the frozen released core.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise PseudoSSLError("EMA decay must be finite and in [0,1)")
        self.decay = float(decay)
        self.names = permitted_trainable_names(model)
        parameters = dict(model.named_parameters())
        self.shadow = {name: parameters[name].detach().clone() for name in self.names}
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if permitted_trainable_names(model) != self.names:
            raise PseudoSSLError("EMA trainable parameter scope drift")
        parameters = dict(model.named_parameters())
        for name in self.names:
            value = parameters[name].detach()
            shadow = self.shadow[name]
            if value.shape != shadow.shape or value.dtype != shadow.dtype or value.device != shadow.device:
                raise PseudoSSLError("EMA parameter tensor identity drift")
            shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
        self.steps += 1

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Run a sequential teacher forward with only EMA parameters swapped."""

        if permitted_trainable_names(model) != self.names:
            raise PseudoSSLError("EMA teacher scope drift")
        parameters = dict(model.named_parameters())
        student = {name: parameters[name].detach().clone() for name in self.names}
        training = model.training
        try:
            with torch.no_grad():
                for name in self.names:
                    parameters[name].copy_(self.shadow[name])
            model.eval()
            with torch.no_grad():
                yield
        finally:
            with torch.no_grad():
                for name in self.names:
                    parameters[name].copy_(student[name])
            model.train(training)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SSL_SCHEMA,
            "decay": self.decay,
            "steps": self.steps,
            "names": list(self.names),
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, model: nn.Module, state: Mapping[str, Any]) -> None:
        if (
            set(state) != {"schema", "decay", "steps", "names", "shadow"}
            or state.get("schema") != SSL_SCHEMA
            or tuple(state.get("names", ())) != self.names
            or state.get("decay") != self.decay
            or isinstance(state.get("steps"), bool)
            or not isinstance(state.get("steps"), int)
            or int(state["steps"]) < 0
            or not isinstance(state.get("shadow"), Mapping)
        ):
            raise PseudoSSLError("EMA checkpoint schema or parameter scope mismatch")
        parameters = dict(model.named_parameters())
        loaded: dict[str, torch.Tensor] = {}
        for name in self.names:
            value = state["shadow"].get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameters[name].shape
                or value.dtype != parameters[name].dtype
                or not _finite_tensor(value)
            ):
                raise PseudoSSLError("EMA checkpoint contains an invalid shadow tensor")
            loaded[name] = value.detach().to(parameters[name].device).clone()
        if set(state["shadow"]) != set(self.names):
            raise PseudoSSLError("EMA checkpoint contains frozen or unknown parameters")
        self.shadow = loaded
        self.steps = int(state["steps"])


@dataclass(frozen=True)
class CurriculumSpec:
    confidence_start: float
    confidence_floor: float
    uncertainty_start: float
    uncertainty_ceiling: float
    coverage_start: float
    coverage_end: float

    def __post_init__(self) -> None:
        values = (
            self.confidence_start,
            self.confidence_floor,
            self.uncertainty_start,
            self.uncertainty_ceiling,
            self.coverage_start,
            self.coverage_end,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise PseudoSSLError("curriculum values must be finite probabilities")
        if (
            self.confidence_start < self.confidence_floor
            or self.uncertainty_start > self.uncertainty_ceiling
            or self.coverage_start > self.coverage_end
        ):
            raise PseudoSSLError("curriculum must relax monotonically")


DEFAULT_CURRICULA = {
    "detection": CurriculumSpec(0.98, 0.75, 0.08, 0.35, 0.02, 0.35),
    "continuation": CurriculumSpec(0.99, 0.80, 0.05, 0.25, 0.01, 0.25),
    "division": CurriculumSpec(0.995, 0.95, 0.03, 0.10, 0.002, 0.05),
}


@dataclass(frozen=True)
class SelectionAudit:
    task: str
    total: int
    accepted: int
    rejected: Mapping[str, int]
    target_coverage: float
    realized_coverage: float
    confidence_threshold: float
    uncertainty_threshold: float

    def __post_init__(self) -> None:
        if self.task not in TASKS or self.total < 0 or self.accepted < 0:
            raise PseudoSSLError("invalid pseudo-label accounting")
        if self.accepted + sum(self.rejected.values()) != self.total:
            raise PseudoSSLError("pseudo-label accounting does not reconcile")


@dataclass(frozen=True)
class SelectionResult:
    mask: torch.Tensor
    audit: SelectionAudit


class AdaptiveThresholdCurriculum:
    """Task-specific confidence/uncertainty gates with bounded coverage ramp."""

    def __init__(self, curricula: Mapping[str, CurriculumSpec] | None = None, ramp_steps: int = 2_000) -> None:
        self.curricula = dict(curricula or DEFAULT_CURRICULA)
        if set(self.curricula) != set(TASKS) or ramp_steps < 1:
            raise PseudoSSLError("curriculum must define all tasks and a positive ramp")
        self.ramp_steps = int(ramp_steps)

    def select(
        self,
        task: str,
        confidence: torch.Tensor,
        uncertainty: torch.Tensor,
        *,
        step: int,
        division_gate_open: bool = False,
        preeligible: torch.Tensor | None = None,
        preeligible_reason: str = "dependency_rejected",
    ) -> SelectionResult:
        if task not in self.curricula or step < 0:
            raise PseudoSSLError("unknown curriculum task or negative step")
        if (
            confidence.ndim != 1
            or uncertainty.shape != confidence.shape
            or confidence.device != uncertainty.device
            or not confidence.is_floating_point()
            or not uncertainty.is_floating_point()
            or not _finite_tensor(confidence)
            or not _finite_tensor(uncertainty)
            or bool(((confidence < 0) | (confidence > 1)).any())
            or bool(((uncertainty < 0) | (uncertainty > 1)).any())
        ):
            raise PseudoSSLError("confidence and uncertainty must be finite [0,1] vectors")
        total = len(confidence)
        allowed = torch.ones(total, dtype=torch.bool, device=confidence.device)
        if preeligible is not None:
            if preeligible.shape != confidence.shape or preeligible.dtype is not torch.bool or preeligible.device != confidence.device:
                raise PseudoSSLError("preeligible mask must match the candidate vector")
            allowed &= preeligible
        progress = min(1.0, step / self.ramp_steps)
        spec = self.curricula[task]
        conf_min = spec.confidence_start + progress * (spec.confidence_floor - spec.confidence_start)
        unc_max = spec.uncertainty_start + progress * (spec.uncertainty_ceiling - spec.uncertainty_start)
        target = spec.coverage_start + progress * (spec.coverage_end - spec.coverage_start)
        if task == "division" and not division_gate_open:
            mask = torch.zeros(total, dtype=torch.bool, device=confidence.device)
            return SelectionResult(mask, SelectionAudit(task, total, 0, {"gate_closed": total}, target, 0.0, conf_min, unc_max))
        below = allowed & (confidence < conf_min)
        above = allowed & ~below & (uncertainty > unc_max)
        eligible = allowed & ~below & ~above
        budget = min(int(eligible.sum().item()), int(math.ceil(total * target)))
        accepted_indices: list[int] = []
        if budget > 0:
            indices = torch.nonzero(eligible, as_tuple=False).flatten().detach().cpu().tolist()
            indices.sort(key=lambda index: (-float(confidence[index]), float(uncertainty[index]), index))
            accepted_indices = indices[:budget]
        mask = torch.zeros(total, dtype=torch.bool, device=confidence.device)
        if accepted_indices:
            mask[torch.as_tensor(accepted_indices, device=mask.device)] = True
        reasons: dict[str, int] = {}
        dependency = int((~allowed).sum().item())
        if dependency:
            reasons[preeligible_reason] = dependency
        for label, count in (
            ("below_confidence", int(below.sum().item())),
            ("above_uncertainty", int(above.sum().item())),
            ("curriculum_budget", int(eligible.sum().item()) - len(accepted_indices)),
        ):
            if count:
                reasons[label] = count
        realized = len(accepted_indices) / total if total else 0.0
        adaptive_conf = min((float(confidence[index]) for index in accepted_indices), default=conf_min)
        adaptive_unc = max((float(uncertainty[index]) for index in accepted_indices), default=unc_max)
        return SelectionResult(
            mask,
            SelectionAudit(task, total, len(accepted_indices), reasons, target, realized, adaptive_conf, adaptive_unc),
        )


@dataclass
class AcceptanceLedger:
    """Per-video, per-task acceptance/rejection totals."""

    videos: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def add(self, video_id: str, audit: SelectionAudit) -> None:
        if not video_id.startswith("train:"):
            raise PseudoSSLError("acceptance ledger only accepts canonical train video IDs")
        task = self.videos.setdefault(video_id, {}).setdefault(
            audit.task,
            {
                "total": 0,
                "accepted": 0,
                "rejected": {},
                "last_target_coverage": 0.0,
                "last_confidence_threshold": 0.0,
                "last_uncertainty_threshold": 0.0,
            },
        )
        task["total"] += audit.total
        task["accepted"] += audit.accepted
        for reason, count in audit.rejected.items():
            task["rejected"][reason] = task["rejected"].get(reason, 0) + count
        task["last_target_coverage"] = audit.target_coverage
        task["last_confidence_threshold"] = audit.confidence_threshold
        task["last_uncertainty_threshold"] = audit.uncertainty_threshold
        task["realized_coverage"] = task["accepted"] / task["total"] if task["total"] else 0.0
        if task["accepted"] + sum(task["rejected"].values()) != task["total"]:
            raise PseudoSSLError("cumulative pseudo-label accounting does not reconcile")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SSL_SCHEMA,
            "videos": self.videos,
            "dense_nxn_association": False,
        }


@dataclass(frozen=True)
class PseudoNode:
    node_id: int
    batch: int
    time: int
    zyx: tuple[int, int, int]
    probability: float
    uncertainty: float
    division_probability: float


@dataclass(frozen=True)
class PseudoEdge:
    source_id: int
    target_id: int
    probability: float
    uncertainty: float
    residual_um: float


@dataclass(frozen=True)
class PseudoDivision:
    parent_id: int
    child_ids: tuple[int, int]
    probability: float
    uncertainty: float


@dataclass(frozen=True)
class SparsePseudoGraph:
    video_id: str
    shape_zyx: tuple[int, int, int]
    voxel_size_um: tuple[float, float, float]
    nodes: tuple[PseudoNode, ...]
    edges: tuple[PseudoEdge, ...]
    divisions: tuple[PseudoDivision, ...]
    neighbor_cap: int
    edge_radius_um: float
    full_frame_dense_matrix: bool = False


@dataclass(frozen=True)
class SelectedPseudoGraph:
    graph: SparsePseudoGraph
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    division_mask: torch.Tensor
    audits: Mapping[str, SelectionAudit]


@dataclass(frozen=True)
class FlipTransform:
    """Axis-preserving weak-to-strong spatial transform for anisotropic data."""

    shape_zyx: tuple[int, int, int]
    flips_zyx: tuple[bool, bool, bool] = (False, False, False)

    def __post_init__(self) -> None:
        if len(self.shape_zyx) != 3 or any(value < 1 for value in self.shape_zyx):
            raise PseudoSSLError("transform shape must be three positive dimensions")
        if len(self.flips_zyx) != 3 or any(type(value) is not bool for value in self.flips_zyx):
            raise PseudoSSLError("transform flips must be three booleans")

    def image(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3 or tuple(value.shape[-3:]) != self.shape_zyx:
            raise PseudoSSLError("image shape disagrees with weak-to-strong transform")
        dimensions = [value.ndim - 3 + axis for axis, flip in enumerate(self.flips_zyx) if flip]
        return torch.flip(value, dimensions) if dimensions else value

    def point(self, zyx: Sequence[int]) -> tuple[int, int, int]:
        if len(zyx) != 3:
            raise PseudoSSLError("point must use z/y/x order")
        point = tuple(int(value) for value in zyx)
        if any(value < 0 or value >= limit for value, limit in zip(point, self.shape_zyx, strict=True)):
            raise PseudoSSLError("point is outside the transform volume")
        return tuple(
            limit - 1 - value if flip else value
            for value, limit, flip in zip(point, self.shape_zyx, self.flips_zyx, strict=True)
        )

    def physical_vector(self, vector_zyx_um: Sequence[float]) -> tuple[float, float, float]:
        if len(vector_zyx_um) != 3 or any(not math.isfinite(float(value)) for value in vector_zyx_um):
            raise PseudoSSLError("physical vector must contain three finite values")
        return tuple(
            -float(value) if flip else float(value)
            for value, flip in zip(vector_zyx_um, self.flips_zyx, strict=True)
        )


def validate_physical_graph(graph: SparsePseudoGraph) -> None:
    """Check one video's physical coordinate and bounded sparse graph contract."""

    if not graph.video_id.startswith("train:") or graph.full_frame_dense_matrix is not False:
        raise PseudoSSLError("pseudo graph must be a train-video sparse graph")
    if (
        len(graph.shape_zyx) != 3
        or len(graph.voxel_size_um) != 3
        or any(value < 1 for value in graph.shape_zyx)
        or any(not math.isfinite(value) or value <= 0 for value in graph.voxel_size_um)
        or graph.neighbor_cap not in (2, 4, 8)
        or not math.isfinite(graph.edge_radius_um)
        or not 4.0 <= graph.edge_radius_um <= 10.0
    ):
        raise PseudoSSLError("invalid per-video physical sparse-gate configuration")
    nodes: dict[int, PseudoNode] = {}
    for node in graph.nodes:
        if (
            node.node_id in nodes
            or node.batch < 0
            or node.time < 0
            or len(node.zyx) != 3
            or any(value < 0 or value >= limit for value, limit in zip(node.zyx, graph.shape_zyx, strict=True))
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (node.probability, node.uncertainty, node.division_probability))
        ):
            raise PseudoSSLError("invalid pseudo detection coordinate or probability")
        nodes[node.node_id] = node
    seen_edges: set[tuple[int, int]] = set()
    incoming_candidates: dict[int, int] = {}
    for edge in graph.edges:
        pair = (edge.source_id, edge.target_id)
        if (
            pair in seen_edges
            or edge.source_id not in nodes
            or edge.target_id not in nodes
            or nodes[edge.target_id].time != nodes[edge.source_id].time + 1
            or nodes[edge.target_id].batch != nodes[edge.source_id].batch
            or not all(math.isfinite(value) for value in (edge.probability, edge.uncertainty, edge.residual_um))
            or not 0.0 <= edge.probability <= 1.0
            or not 0.0 <= edge.uncertainty <= 1.0
            or not 0.0 <= edge.residual_um <= graph.edge_radius_um + 1e-6
        ):
            raise PseudoSSLError("invalid physical continuation candidate")
        seen_edges.add(pair)
        incoming_candidates[edge.target_id] = incoming_candidates.get(edge.target_id, 0) + 1
    if any(count > graph.neighbor_cap for count in incoming_candidates.values()):
        raise PseudoSSLError("sparse continuation neighbor cap exceeded")
    for motif in graph.divisions:
        if (
            motif.parent_id not in nodes
            or len(set(motif.child_ids)) != 2
            or any(child not in nodes for child in motif.child_ids)
            or any(nodes[child].time != nodes[motif.parent_id].time + 1 for child in motif.child_ids)
            or any((motif.parent_id, child) not in seen_edges for child in motif.child_ids)
            or not math.isfinite(motif.probability)
            or not math.isfinite(motif.uncertainty)
            or not 0.0 <= motif.probability <= 1.0
            or not 0.0 <= motif.uncertainty <= 1.0
        ):
            raise PseudoSSLError("invalid sparse division motif")


def extract_sparse_pseudo_graph(
    teacher_output: torch.Tensor,
    *,
    video_id: str,
    voxel_size_um: Sequence[float],
    candidate_floor: float = 0.20,
    max_points_per_frame: int = 512,
    edge_radius_um: float = 8.0,
    neighbor_cap: int = 4,
) -> SparsePseudoGraph:
    """Extract two-frame detections and physical cKDTree continuation COO rows."""

    if (
        teacher_output.ndim != 6
        or teacher_output.shape[:3] != (1, 2, 6)
        or not _finite_tensor(teacher_output)
        or not 0.0 <= candidate_floor <= 1.0
        or max_points_per_frame < 1
    ):
        raise PseudoSSLError("teacher output must be finite (1,2,6,Z,Y,X)")
    shape = tuple(int(value) for value in teacher_output.shape[-3:])
    scale = tuple(float(value) for value in voxel_size_um)
    if len(scale) != 3 or any(not math.isfinite(value) or value <= 0 for value in scale):
        raise PseudoSSLError("voxel scale must be three positive finite micron values")
    center = torch.sigmoid(teacher_output[0, :, 0]).detach()
    detection_uncertainty = (4.0 * center * (1.0 - center)).detach()
    motion_uncertainty = torch.sigmoid(teacher_output[0, :, 4]).detach()
    division = torch.sigmoid(teacher_output[0, :, 5]).detach()
    motion = teacher_output[0, :, 1:4].detach()
    pooled = F.max_pool3d(center[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    center_cpu = center.cpu()
    detection_uncertainty_cpu = detection_uncertainty.cpu()
    motion_uncertainty_cpu = motion_uncertainty.cpu()
    division_cpu = division.cpu()
    motion_cpu = motion.cpu()
    nodes: list[PseudoNode] = []
    by_time: dict[int, list[PseudoNode]] = {0: [], 1: []}
    node_motion: dict[int, tuple[float, float, float]] = {}
    node_motion_uncertainty: dict[int, float] = {}
    next_id = 0
    for time_index in (0, 1):
        coords = torch.nonzero((center[time_index] >= candidate_floor) & (center[time_index] == pooled[time_index]), as_tuple=False)
        if len(coords) > max_points_per_frame:
            values = center[time_index][tuple(coords.unbind(dim=1))]
            order = torch.argsort(values, descending=True, stable=True)[:max_points_per_frame]
            coords = coords[order]
        for raw in coords.detach().cpu().tolist():
            zyx = tuple(int(value) for value in raw)
            node = PseudoNode(
                node_id=next_id,
                batch=0,
                time=time_index,
                zyx=zyx,
                probability=float(center_cpu[time_index, *zyx]),
                uncertainty=float(detection_uncertainty_cpu[time_index, *zyx]),
                division_probability=float(division_cpu[time_index, *zyx]),
            )
            nodes.append(node)
            by_time[time_index].append(node)
            node_motion[next_id] = tuple(float(value) for value in motion_cpu[time_index, :, *zyx])
            node_motion_uncertainty[next_id] = float(motion_uncertainty_cpu[time_index, *zyx])
            next_id += 1
    edges: list[PseudoEdge] = []
    if by_time[0] and by_time[1]:
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise PseudoSSLError("scipy cKDTree is required for sparse physical candidates") from exc
        source_um = [tuple(coordinate * unit for coordinate, unit in zip(node.zyx, scale, strict=True)) for node in by_time[0]]
        tree = cKDTree(source_um)
        for child in by_time[1]:
            child_um = tuple(coordinate * unit for coordinate, unit in zip(child.zyx, scale, strict=True))
            predicted_parent_um = tuple(value + delta for value, delta in zip(child_um, node_motion[child.node_id], strict=True))
            distances, indices = tree.query(
                predicted_parent_um,
                k=min(neighbor_cap, len(source_um)),
                distance_upper_bound=edge_radius_um,
            )
            distance_values = [float(distances)] if not hasattr(distances, "__len__") else [float(value) for value in distances]
            index_values = [int(indices)] if not hasattr(indices, "__len__") else [int(value) for value in indices]
            for residual, source_index in zip(distance_values, index_values, strict=True):
                if not math.isfinite(residual) or source_index >= len(by_time[0]):
                    continue
                source = by_time[0][source_index]
                probability = math.sqrt(source.probability * child.probability) * math.exp(
                    -0.5 * (residual / edge_radius_um) ** 2
                )
                edge_uncertainty = max(
                    child.uncertainty,
                    node_motion_uncertainty[child.node_id],
                    min(1.0, residual / edge_radius_um),
                )
                edges.append(PseudoEdge(source.node_id, child.node_id, probability, edge_uncertainty, residual))
    edges.sort(key=lambda edge: (edge.target_id, -edge.probability, edge.source_id))
    by_source: dict[int, list[PseudoEdge]] = {}
    for edge in edges:
        by_source.setdefault(edge.source_id, []).append(edge)
    node_index = {node.node_id: node for node in nodes}
    divisions: list[PseudoDivision] = []
    for source_id, outgoing in sorted(by_source.items()):
        distinct = sorted(outgoing, key=lambda edge: (-edge.probability, edge.target_id))
        if len(distinct) < 2:
            continue
        first, second = distinct[:2]
        parent = node_index[source_id]
        probability = min(parent.division_probability, first.probability, second.probability)
        divisions.append(
            PseudoDivision(
                source_id,
                tuple(sorted((first.target_id, second.target_id))),
                probability,
                max(parent.uncertainty, first.uncertainty, second.uncertainty),
            )
        )
    graph = SparsePseudoGraph(
        video_id=video_id,
        shape_zyx=shape,
        voxel_size_um=scale,
        nodes=tuple(nodes),
        edges=tuple(edges),
        divisions=tuple(divisions),
        neighbor_cap=neighbor_cap,
        edge_radius_um=float(edge_radius_um),
    )
    validate_physical_graph(graph)
    return graph


def select_pseudo_graph(
    graph: SparsePseudoGraph,
    curriculum: AdaptiveThresholdCurriculum,
    *,
    step: int,
    division_gate_open: bool = False,
    ledger: AcceptanceLedger | None = None,
) -> SelectedPseudoGraph:
    """Apply task curricula with endpoint/motif dependency accounting."""

    validate_physical_graph(graph)
    device = torch.device("cpu")
    node_conf = torch.tensor([node.probability for node in graph.nodes], dtype=torch.float32, device=device)
    node_unc = torch.tensor([node.uncertainty for node in graph.nodes], dtype=torch.float32, device=device)
    detection = curriculum.select("detection", node_conf, node_unc, step=step)
    node_positions = {node.node_id: index for index, node in enumerate(graph.nodes)}
    edge_dependency = torch.tensor(
        [
            bool(detection.mask[node_positions[edge.source_id]])
            and bool(detection.mask[node_positions[edge.target_id]])
            for edge in graph.edges
        ],
        dtype=torch.bool,
    )
    edge_conf = torch.tensor([edge.probability for edge in graph.edges], dtype=torch.float32)
    edge_unc = torch.tensor([edge.uncertainty for edge in graph.edges], dtype=torch.float32)
    continuation = curriculum.select(
        "continuation",
        edge_conf,
        edge_unc,
        step=step,
        preeligible=edge_dependency,
        preeligible_reason="endpoint_rejected",
    )
    accepted_edges = {
        (edge.source_id, edge.target_id)
        for edge, accepted in zip(graph.edges, continuation.mask.tolist(), strict=True)
        if accepted
    }
    division_dependency = torch.tensor(
        [
            bool(detection.mask[node_positions[motif.parent_id]])
            and all(bool(detection.mask[node_positions[child]]) for child in motif.child_ids)
            and all((motif.parent_id, child) in accepted_edges for child in motif.child_ids)
            for motif in graph.divisions
        ],
        dtype=torch.bool,
    )
    division_conf = torch.tensor([motif.probability for motif in graph.divisions], dtype=torch.float32)
    division_unc = torch.tensor([motif.uncertainty for motif in graph.divisions], dtype=torch.float32)
    division = curriculum.select(
        "division",
        division_conf,
        division_unc,
        step=step,
        division_gate_open=division_gate_open,
        preeligible=division_dependency,
        preeligible_reason="motif_dependency_rejected",
    )
    audits = {
        "detection": detection.audit,
        "continuation": continuation.audit,
        "division": division.audit,
    }
    if ledger is not None:
        for audit in audits.values():
            ledger.add(graph.video_id, audit)
    return SelectedPseudoGraph(graph, detection.mask, continuation.mask, division.mask, audits)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum()
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).eps)


def sparse_pseudo_losses(
    student_output: torch.Tensor,
    selected: SelectedPseudoGraph,
    transform: FlipTransform,
    *,
    detection_weight: float = 1.0,
    continuation_weight: float = 1.0,
    motion_weight: float = 0.25,
    division_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Weak-teacher/strong-student point and candidate-only edge losses."""

    graph = selected.graph
    validate_physical_graph(graph)
    if (
        student_output.ndim != 6
        or student_output.shape[2] != 6
        or tuple(student_output.shape[-3:]) != graph.shape_zyx
        or transform.shape_zyx != graph.shape_zyx
        or not _finite_tensor(student_output)
    ):
        raise PseudoSSLError("student output is not a finite transform-compatible six-channel tensor")
    weights = (detection_weight, continuation_weight, motion_weight, division_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise PseudoSSLError("pseudo loss weights must be finite and nonnegative")
    nodes = {node.node_id: node for node in graph.nodes}
    zero = student_output.sum() * 0.0

    detection_terms: list[torch.Tensor] = []
    detection_weights: list[float] = []
    for node, accepted in zip(graph.nodes, selected.node_mask.tolist(), strict=True):
        if not accepted:
            continue
        zyx = transform.point(node.zyx)
        logit = student_output[node.batch, node.time, 0, *zyx]
        target = logit.new_tensor(node.probability)
        detection_terms.append(F.binary_cross_entropy_with_logits(logit, target, reduction="none"))
        detection_weights.append(node.probability * (1.0 - node.uncertainty))
    detection = (
        _weighted_mean(torch.stack(detection_terms), student_output.new_tensor(detection_weights))
        if detection_terms
        else zero
    )

    continuation_terms: list[torch.Tensor] = []
    motion_terms: list[torch.Tensor] = []
    edge_weights: list[float] = []
    for edge, accepted in zip(graph.edges, selected.edge_mask.tolist(), strict=True):
        if not accepted:
            continue
        source, target = nodes[edge.source_id], nodes[edge.target_id]
        target_zyx = transform.point(target.zyx)
        source_um = tuple(value * unit for value, unit in zip(source.zyx, graph.voxel_size_um, strict=True))
        target_um = tuple(value * unit for value, unit in zip(target.zyx, graph.voxel_size_um, strict=True))
        backward_um = transform.physical_vector(
            tuple(parent - child for parent, child in zip(source_um, target_um, strict=True))
        )
        prediction = student_output[target.batch, target.time, 1:4, *target_zyx]
        target_vector = prediction.new_tensor(backward_um)
        residual = torch.linalg.vector_norm(prediction - target_vector)
        edge_logit = 4.0 * (1.0 - residual / graph.edge_radius_um)
        continuation_terms.append(
            F.binary_cross_entropy_with_logits(edge_logit, edge_logit.new_tensor(edge.probability), reduction="none")
        )
        motion_terms.append(F.smooth_l1_loss(prediction, target_vector, reduction="mean"))
        edge_weights.append(edge.probability * (1.0 - edge.uncertainty))
    if continuation_terms:
        edge_weight_tensor = student_output.new_tensor(edge_weights)
        continuation = _weighted_mean(torch.stack(continuation_terms), edge_weight_tensor)
        motion = _weighted_mean(torch.stack(motion_terms), edge_weight_tensor)
    else:
        continuation = motion = zero

    division_terms: list[torch.Tensor] = []
    division_weights: list[float] = []
    for motif, accepted in zip(graph.divisions, selected.division_mask.tolist(), strict=True):
        if not accepted:
            continue
        parent = nodes[motif.parent_id]
        parent_zyx = transform.point(parent.zyx)
        logit = student_output[parent.batch, parent.time, 5, *parent_zyx]
        division_terms.append(
            F.binary_cross_entropy_with_logits(logit, logit.new_tensor(motif.probability), reduction="none")
        )
        division_weights.append(motif.probability * (1.0 - motif.uncertainty))
    division = (
        _weighted_mean(torch.stack(division_terms), student_output.new_tensor(division_weights))
        if division_terms
        else zero
    )
    total = (
        detection_weight * detection
        + continuation_weight * continuation
        + motion_weight * motion
        + division_weight * division
    )
    if not all(_finite_tensor(value) for value in (total, detection, continuation, motion, division)):
        raise PseudoSSLError("pseudo-label loss is nonfinite")
    return {
        "total": total,
        "detection": detection,
        "continuation": continuation,
        "motion": motion,
        "division": division,
        "accepted_detections": student_output.new_tensor(float(selected.node_mask.sum())),
        "accepted_edges": student_output.new_tensor(float(selected.edge_mask.sum())),
        "accepted_divisions": student_output.new_tensor(float(selected.division_mask.sum())),
    }


def sparse_contract(graph: SparsePseudoGraph) -> dict[str, Any]:
    """Serializable attestation used by experiment manifests."""

    validate_physical_graph(graph)
    return {
        "schema": SSL_SCHEMA,
        "video_id": graph.video_id,
        "scale_zyx_um": list(graph.voxel_size_um),
        "candidate_nodes": len(graph.nodes),
        "candidate_edges": len(graph.edges),
        "candidate_divisions": len(graph.divisions),
        "neighbor_cap": graph.neighbor_cap,
        "edge_radius_um": graph.edge_radius_um,
        "candidate_representation": "COO-records",
        "full_frame_dense_matrix": False,
        "dense_nxn_association": False,
    }
