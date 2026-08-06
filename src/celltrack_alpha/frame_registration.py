"""Standing frame-registration preprocessing for candidate construction.

The evaluator-coordinate view is immutable.  Only the candidate-geometry view
receives cumulative whole-frame shifts, and callers must build every edge or
division candidate from that second view.  Image/feature sampling continues to
use the original coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "celltrack-alpha.frame-registration-plan.v1"
VIEWS_SCHEMA = "celltrack-alpha.registered-coordinate-views.v1"
RECEIPT_SCHEMA = "celltrack-alpha.frame-registration-receipt.v1"
REGISTERED_STRIDE_ZYX = (1, 4, 4)
POOL_BREAKING_THRESHOLD_UM = 10.0


class FrameRegistrationError(ValueError):
    """A registration input or transform-once invariant was violated."""


def _readonly(value: Any, *, dtype: Any, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise FrameRegistrationError(
            f"{label} must have shape (N,{','.join(map(str, shape_tail))})"
        )
    if not np.isfinite(array).all():
        raise FrameRegistrationError(f"{label} contains nonfinite values")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RegistrationPlan:
    """One cumulative target-aligning transform per frame."""

    identity: str
    cumulative_shift_raw_zyx: np.ndarray
    pool_breaking_frames: tuple[int, ...]
    census_sha256: str
    threshold_um: float = POOL_BREAKING_THRESHOLD_UM
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise FrameRegistrationError("registration plan schema drifted")
        if not isinstance(self.identity, str) or ":" not in self.identity:
            raise FrameRegistrationError("registration identity must be split:dataset")
        shifts = _readonly(
            self.cumulative_shift_raw_zyx,
            dtype=np.int64,
            shape_tail=(3,),
            label="cumulative shifts",
        )
        if len(shifts) < 2 or np.any(shifts[0] != 0):
            raise FrameRegistrationError("registration plan must start at exact zero")
        frames = tuple(int(value) for value in self.pool_breaking_frames)
        if frames != tuple(sorted(set(frames))) or any(
            frame < 0 or frame >= len(shifts) - 1 for frame in frames
        ):
            raise FrameRegistrationError("pool-breaking frames are invalid")
        if not math.isfinite(self.threshold_um) or self.threshold_um != 10.0:
            raise FrameRegistrationError("standing threshold must remain exactly 10 um")
        if (
            not isinstance(self.census_sha256, str)
            or len(self.census_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.census_sha256)
        ):
            raise FrameRegistrationError("census SHA-256 is malformed")
        object.__setattr__(self, "cumulative_shift_raw_zyx", shifts)
        object.__setattr__(self, "pool_breaking_frames", frames)

    @property
    def frame_count(self) -> int:
        return len(self.cumulative_shift_raw_zyx)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "identity": self.identity,
            "census_sha256": self.census_sha256,
            "frame_count": self.frame_count,
            "pool_breaking_threshold_um": self.threshold_um,
            "pool_breaking_transition_count": len(self.pool_breaking_frames),
            "pool_breaking_frames": list(self.pool_breaking_frames),
            "registered_stride_zyx": list(REGISTERED_STRIDE_ZYX),
            "coordinate_contract": {
                "image_and_feature_sampling": "original",
                "candidate_topology_and_geometry": "registered",
                "evaluator_output": "original",
                "applications_per_video": 1,
            },
        }


@dataclass(frozen=True, slots=True)
class CoordinateViews:
    """Original and registered coordinates with transform-once provenance."""

    timepoints: np.ndarray
    original_zyx: np.ndarray
    candidate_zyx: np.ndarray
    common_offset_raw_zyx: np.ndarray
    receipt: Mapping[str, Any]
    schema: str = VIEWS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VIEWS_SCHEMA:
            raise FrameRegistrationError("coordinate-view schema drifted")
        timepoints = np.asarray(self.timepoints, dtype=np.int64)
        if timepoints.ndim != 1 or np.any(timepoints < 0):
            raise FrameRegistrationError("timepoints must be nonnegative and one-dimensional")
        original = _readonly(
            self.original_zyx, dtype=np.float32, shape_tail=(3,), label="original coordinates"
        )
        candidate = _readonly(
            self.candidate_zyx, dtype=np.float32, shape_tail=(3,), label="candidate coordinates"
        )
        offset = np.asarray(self.common_offset_raw_zyx, dtype=np.float32)
        if (
            len(timepoints) != len(original)
            or len(original) != len(candidate)
            or offset.shape != (3,)
            or not np.isfinite(offset).all()
        ):
            raise FrameRegistrationError("coordinate-view arrays disagree")
        timepoints = np.ascontiguousarray(timepoints)
        offset = np.ascontiguousarray(offset)
        timepoints.setflags(write=False)
        offset.setflags(write=False)
        object.__setattr__(self, "timepoints", timepoints)
        object.__setattr__(self, "original_zyx", original)
        object.__setattr__(self, "candidate_zyx", candidate)
        object.__setattr__(self, "common_offset_raw_zyx", offset)


def build_registration_plan(
    *,
    identity: str,
    transitions: Sequence[Mapping[str, Any]],
    census_sha256: str,
    frame_count: int = 100,
) -> RegistrationPlan:
    """Validate census rows and compose only the frozen >10-um jump steps."""

    if type(frame_count) is not int or frame_count < 2:
        raise FrameRegistrationError("frame_count must be an integer >=2")
    by_frame: dict[int, Mapping[str, Any]] = {}
    for row in transitions:
        frame = int(row.get("frame", -1))
        if frame in by_frame:
            raise FrameRegistrationError("duplicate registration transition")
        by_frame[frame] = row
    if set(by_frame) != set(range(frame_count - 1)):
        raise FrameRegistrationError("registration transition coverage is incomplete")
    cumulative = np.zeros((frame_count, 3), dtype=np.int64)
    flagged: list[int] = []
    stride = np.asarray(REGISTERED_STRIDE_ZYX, dtype=np.int64)
    for frame in range(frame_count - 1):
        row = by_frame[frame]
        magnitude = float(row.get("shift_magnitude_um", math.nan))
        declared = row.get("pool_breaking")
        if not math.isfinite(magnitude) or type(declared) is not bool:
            raise FrameRegistrationError("registration magnitude/flag is malformed")
        expected = magnitude > POOL_BREAKING_THRESHOLD_UM
        if declared is not expected:
            raise FrameRegistrationError("pool-breaking flag disagrees with exact >10-um rule")
        shift = np.asarray(row.get("shift_registered_zyx"), dtype=np.int64)
        if shift.shape != (3,):
            raise FrameRegistrationError("registered shift must contain Z/Y/X")
        step = shift * stride if declared else np.zeros(3, dtype=np.int64)
        cumulative[frame + 1] = cumulative[frame] + step
        if declared:
            flagged.append(frame)
    return RegistrationPlan(
        identity=identity,
        cumulative_shift_raw_zyx=cumulative,
        pool_breaking_frames=tuple(flagged),
        census_sha256=census_sha256,
    )


def load_registration_plan(census: Path, identity: str) -> RegistrationPlan:
    """Load one video plan from the canonical pathology-census namespace."""

    if not census.is_file():
        raise FrameRegistrationError(f"census is absent: {census}")
    try:
        split, dataset = identity.split(":", 1)
    except ValueError as exc:
        raise FrameRegistrationError("identity must be split:dataset") from exc
    if split not in {"train", "test"} or not dataset:
        raise FrameRegistrationError("identity split/dataset is invalid")
    video = census.parent / "videos" / f"{split}__{dataset}.json"
    try:
        row = json.loads(video.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameRegistrationError(f"cannot read video census: {video}") from exc
    if row.get("split") != split or row.get("dataset") != dataset:
        raise FrameRegistrationError("video census identity drifted")
    transitions = row.get("transitions")
    if not isinstance(transitions, list):
        raise FrameRegistrationError("video census transitions are absent")
    return build_registration_plan(
        identity=identity,
        transitions=transitions,
        census_sha256=sha256_file(census),
        frame_count=100,
    )


def coordinate_views(
    plan: RegistrationPlan,
    timepoints: Any,
    original_zyx: Any,
    *,
    nonnegative_candidate_storage: bool = False,
) -> CoordinateViews:
    """Apply a plan exactly once and retain the untouched evaluator view."""

    if isinstance(original_zyx, CoordinateViews):
        raise FrameRegistrationError("registration cannot be applied twice")
    times = np.asarray(timepoints, dtype=np.int64)
    original = np.asarray(original_zyx, dtype=np.float32)
    if (
        times.ndim != 1
        or original.shape != (len(times), 3)
        or np.any(times < 0)
        or np.any(times >= plan.frame_count)
        or not np.isfinite(original).all()
    ):
        raise FrameRegistrationError("coordinate inputs are malformed")
    candidate = original.astype(np.float64) + plan.cumulative_shift_raw_zyx[times]
    offset = np.zeros(3, dtype=np.float64)
    if nonnegative_candidate_storage and len(candidate):
        offset = np.maximum(-np.min(candidate, axis=0), 0.0)
        candidate += offset[None, :]
    return CoordinateViews(
        timepoints=times,
        original_zyx=original,
        candidate_zyx=candidate.astype(np.float32),
        common_offset_raw_zyx=offset.astype(np.float32),
        receipt=plan.receipt(),
    )

