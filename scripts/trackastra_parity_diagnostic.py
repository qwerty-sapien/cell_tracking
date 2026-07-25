#!/usr/bin/env python3
"""Launch and analyze the one-video Trackastra T4 parity diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARITY_IDENTITY = "train:6bba_edf14583"
REFERENCE_PATH = (
    ROOT
    / "reports/g02/trackastra_full_pool_feature_cache_v1/videos"
    / "train__6bba_edf14583.pt"
)
REFERENCE_SHA256 = (
    "fed771aaf78df907460e8833a66d698067766e942258c088e31219e998115016"
)
PRODUCTION_KERNEL = (
    ROOT / "kaggle_kernel/trackastra-6bba-feat/trackastra_feat.py"
)
PRODUCTION_KERNEL_SHA256 = (
    "caf3e02e20f45dc688c16e3018269cae11bffee3a92c2930f675d1dca35613f8"
)
KERNEL_DIR = ROOT / "kaggle_kernel/trackastra-parity-diagnostic"
KERNEL_ID = "onebitbrain/trackastra-6bba-parity-diag"
T4_MANIFEST_NAME = "trackastra_parity_t4_manifest.json"
T4_RUN1_NAME = "trackastra_parity_t4_run1_row_features.npy"
T4_RUN2_NAME = "trackastra_parity_t4_run2_row_features.npy"
DEFAULT_OUTPUT = ROOT / "reports/g02/trackastra_parity_diagnostic_t4_v1"
DEFAULT_JSON_REPORT = (
    ROOT
    / "reports/g02/screening/trackastra_6bba_parity_diagnostic_v1.json"
)
DEFAULT_MD_REPORT = (
    ROOT
    / "reports/g02/screening/trackastra_6bba_parity_diagnostic_v1.md"
)
FEATURE_NAMES = tuple(
    [f"apncc_context_{index:02d}" for index in range(48)]
    + ["trackastra_margin"]
)
FORBIDDEN_SCOPE = re.compile(
    r"(^|[^a-z0-9])(44b6|holdout|validation12|cv57)([^a-z0-9]|$)",
    re.IGNORECASE,
)
FLOAT_DTYPES = frozenset(
    (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))
)
PROBABILITY_DERIVED_COLUMNS = frozenset((5, 6, 21, 22, 47, 48))
INTENSITY_REDUCTION_COLUMNS = frozenset((44, 45))


class DiagnosticError(RuntimeError):
    """A diagnostic input, scope, output, or comparison invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_authorized_scope(identity: str, *paths_or_labels: object) -> None:
    """Fail closed unless every requested input is the one authorized 6bba video."""

    if identity != PARITY_IDENTITY:
        raise DiagnosticError(
            f"single-video scope is restricted to {PARITY_IDENTITY}; got {identity}"
        )
    for value in paths_or_labels:
        text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
        match = FORBIDDEN_SCOPE.search(text)
        if match is not None:
            raise DiagnosticError(
                f"forbidden diagnostic scope token: {match.group(2)} in {text}"
            )


def _authenticated_pickle_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"required M5 v1 reference is missing: {path}")
    actual = sha256_file(path)
    if actual != REFERENCE_SHA256:
        raise DiagnosticError(
            "required M5 v1 reference SHA-256 mismatch: "
            f"{path} expected={REFERENCE_SHA256} actual={actual}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                name for name in archive.namelist() if name.endswith("/data.pkl")
            ]
            if len(members) != 1:
                raise DiagnosticError(
                    f"reference must contain exactly one data.pkl: {path}"
                )
            value = pickle.loads(archive.read(members[0]))
    except DiagnosticError:
        raise
    except Exception as exc:
        raise DiagnosticError(f"cannot restore authenticated reference: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError("authenticated reference payload is not a mapping")
    return value


def load_reference_row_features(
    path: Path = REFERENCE_PATH,
) -> tuple[np.ndarray, tuple[str, ...]]:
    payload = _authenticated_pickle_payload(path)
    features = np.asarray(payload.get("row_features"))
    names = tuple(str(name) for name in payload.get("feature_names", ()))
    if (
        payload.get("identity") != PARITY_IDENTITY
        or features.shape != (5409, 49)
        or features.dtype != np.dtype("float32")
        or names != FEATURE_NAMES
    ):
        raise DiagnosticError("authenticated M5 v1 row_features identity drifted")
    return features, names


def semantic_flags(
    index: int, m5: np.ndarray, t4: np.ndarray
) -> dict[str, bool]:
    scale = float(
        max(
            np.max(np.abs(np.asarray(m5, dtype=np.float64))),
            np.max(np.abs(np.asarray(t4, dtype=np.float64))),
        )
    )
    return {
        "near_zero_magnitude": scale <= 1.0e-4,
        "probability_derived": index in PROBABILITY_DERIVED_COLUMNS,
        "intensity_reduction": index in INTENSITY_REDUCTION_COLUMNS,
    }


def compute_columnwise_stats(
    m5: np.ndarray, t4: np.ndarray
) -> dict[str, np.ndarray | bool]:
    """Compute the exact requested absolute-difference probe columnwise."""

    reference = np.asarray(m5)
    candidate = np.asarray(t4)
    if (
        reference.ndim != 2
        or candidate.shape != reference.shape
        or reference.shape[1] == 0
    ):
        raise DiagnosticError(
            f"row_features shapes differ: M5={reference.shape} T4={candidate.shape}"
        )
    abs_diff = np.abs(reference - candidate)
    m5_float = reference.astype(np.float64, copy=False)
    scale = np.max(np.abs(m5_float), axis=0)
    eps = np.finfo(np.float64).eps
    if reference.dtype in FLOAT_DTYPES:
        eps = float(np.finfo(reference.dtype).eps)
    denominator_floor = np.maximum(scale * eps, 1.0e-12)
    denominator = np.maximum(np.abs(m5_float), denominator_floor[None, :])
    max_rel = np.max(abs_diff / denominator, axis=0)
    return {
        "scale": scale,
        "max": abs_diff.max(axis=0),
        "mean": abs_diff.mean(axis=0),
        "p99": np.quantile(abs_diff, 0.99, axis=0),
        "max_rel": max_rel,
        "max_rel_denominator_floor": denominator_floor,
        "allclose": bool(
            np.allclose(
                reference,
                candidate,
                rtol=1.0e-5,
                atol=1.0e-7,
                equal_nan=True,
            )
        ),
    }


def classify_column(
    m5: np.ndarray,
    t4: np.ndarray,
    *,
    flags: dict[str, bool] | None = None,
) -> tuple[str, str]:
    """Classify a single aligned column as benign FP drift or structural."""

    reference = np.asarray(m5)
    candidate = np.asarray(t4)
    if reference.shape != candidate.shape or reference.ndim != 1:
        return "STRUCTURAL", "shape mismatch"
    if reference.dtype != candidate.dtype:
        return "STRUCTURAL", "dtype mismatch"
    if reference.dtype not in FLOAT_DTYPES:
        return (
            ("BENIGN_FP", "exact non-float column")
            if np.array_equal(reference, candidate)
            else ("STRUCTURAL", "non-float values differ")
        )
    if np.isnan(reference).any() or np.isnan(candidate).any():
        return "STRUCTURAL", "NaN present"
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return "STRUCTURAL", "non-finite value present"
    if np.array_equal(reference, candidate):
        return "BENIGN_FP", "byte-identical"

    ref64 = reference.astype(np.float64, copy=False)
    t464 = candidate.astype(np.float64, copy=False)
    signed = t464 - ref64
    abs_diff = np.abs(signed)
    scale = float(np.max(np.abs(ref64)))
    eps = float(np.finfo(reference.dtype).eps)
    fp_envelope = max(1.0e-7, 128.0 * eps * max(scale, 1.0e-3))
    median = float(np.median(signed))
    spread = float(np.max(np.abs(signed - median)))
    same_sign_fraction = float(
        max(np.mean(signed > 0.0), np.mean(signed < 0.0))
    )
    offset_floor = max(1.0e-7, 16.0 * eps * max(scale, 1.0e-3))
    if (
        abs(median) > offset_floor
        and same_sign_fraction >= 0.99
        and spread <= max(offset_floor, abs(median) * 1.0e-3)
    ):
        return "STRUCTURAL", "whole-column offset"
    if np.allclose(
        reference,
        candidate,
        rtol=1.0e-5,
        atol=1.0e-7,
        equal_nan=True,
    ):
        return "BENIGN_FP", "passes initial FP probe"
    effective_flags = flags or {}
    relaxed_envelope = (
        max(5.0e-7, 512.0 * eps * max(scale, 1.0e-3))
        if any(effective_flags.values())
        else fp_envelope
    )
    if float(abs_diff.max()) <= relaxed_envelope:
        return "BENIGN_FP", "bounded by float reduction envelope"
    relative_scale = float(abs_diff.max()) / max(scale, 1.0e-12)
    if relative_scale <= 1.0e-5:
        return "BENIGN_FP", "small scale-relative drift"
    return "STRUCTURAL", "drift exceeds float reduction envelope"


def _proposed_tolerance(
    *,
    classification: str,
    maximum: float,
    scale: float,
    flags: dict[str, bool],
) -> tuple[float, float]:
    if classification == "STRUCTURAL":
        return 0.0, 0.0
    if maximum == 0.0:
        return 0.0, 0.0
    relative = maximum / max(scale, 1.0e-12)
    if any(flags.values()):
        return max(1.0e-7, 2.0 * maximum), 1.0e-5
    return 1.0e-7, max(1.0e-5, 2.0 * relative)


def build_column_records(
    m5: np.ndarray, t4: np.ndarray, names: Iterable[str]
) -> tuple[list[dict[str, Any]], bool]:
    reference = np.asarray(m5)
    candidate = np.asarray(t4)
    feature_names = tuple(names)
    if (
        reference.shape != candidate.shape
        or reference.ndim != 2
        or reference.shape[1] != len(feature_names)
    ):
        raise DiagnosticError("column record inputs are not aligned")
    stats = compute_columnwise_stats(reference, candidate)
    records: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        flags = semantic_flags(index, reference[:, index], candidate[:, index])
        classification, reason = classify_column(
            reference[:, index], candidate[:, index], flags=flags
        )
        maximum = float(np.asarray(stats["max"])[index])
        scale = float(np.asarray(stats["scale"])[index])
        proposed_atol, proposed_rtol = _proposed_tolerance(
            classification=classification,
            maximum=maximum,
            scale=scale,
            flags=flags,
        )
        dtype = (
            reference.dtype.name
            if reference.dtype == candidate.dtype
            else f"{reference.dtype.name}!={candidate.dtype.name}"
        )
        records.append(
            {
                "index": index,
                "name": name,
                "dtype": dtype,
                "scale": scale,
                "max": maximum,
                "mean": float(np.asarray(stats["mean"])[index]),
                "p99": float(np.asarray(stats["p99"])[index]),
                "max_rel": float(np.asarray(stats["max_rel"])[index]),
                "max_rel_denominator_floor": float(
                    np.asarray(stats["max_rel_denominator_floor"])[index]
                ),
                "initial_allclose": bool(
                    np.allclose(
                        reference[:, index],
                        candidate[:, index],
                        rtol=1.0e-5,
                        atol=1.0e-7,
                        equal_nan=True,
                    )
                ),
                **flags,
                "needs_column_specific_tolerance": bool(
                    maximum > 0.0 and any(flags.values())
                ),
                "classification": classification,
                "classification_reason": reason,
                "proposed_atol": proposed_atol,
                "proposed_rtol": proposed_rtol,
            }
        )
    return records, bool(stats["allclose"])


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def _markdown_report(report: dict[str, Any]) -> str:
    decision = report["DECISION"]
    lines = [
        "# Trackastra 6bba one-video parity diagnostic v1",
        "",
        f"Status: **{report['status']}**",
        "",
        "This is a diagnostic recommendation only. It does not modify the "
        "authenticated M5 v1 reference and does not enact a final tolerance "
        "contract.",
        "",
        "## Decision",
        "",
        f"- T4↔T4 exact repeatability: `{decision['t4_t4_repeatable']}`",
        f"- Any structural column: `{decision['any_structural']}`",
        f"- Recommendation: **{decision['recommendation']}**",
        f"- Rationale: {decision['rationale']}",
        "",
        "## Runtime and retrieval",
        "",
        f"- Identity: `{report['scope']['identity']}` (one video only)",
        f"- Kernel: `{report['kernel']['id']}`; private, internet disabled, "
        "`NvidiaTeslaT4`",
        f"- Kernel elapsed: `{report['runtime']['elapsed_seconds']:.6f}` s "
        f"(`{report['runtime']['t4_hours']:.8f}` T4-h)",
        f"- Libraries: NumPy `{report['runtime']['numpy']}`, PyTorch "
        f"`{report['runtime']['torch']}`, CUDA `{report['runtime']['cuda']}`, "
        f"cuDNN `{report['runtime']['cudnn']}`",
        f"- Output listing verified: `{report['retrieval']['listed']}`; "
        f"download verified: `{report['retrieval']['downloaded']}`",
        "",
        "The initial cross-platform probe is exactly "
        "`np.allclose(rtol=1e-5, atol=1e-7, equal_nan=True)`. Proposed "
        "per-column tolerances below are evidence-derived recommendations and "
        "require explicit human approval before becoming a contract.",
        "",
        "## Columnwise M5↔T4 drift",
        "",
        "| # | Column | dtype | scale | max | mean | p99 | max_rel | flags | "
        "class | proposed atol | proposed rtol |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for column in decision["columns"]:
        flags = ",".join(
            label
            for key, label in (
                ("near_zero_magnitude", "near-zero"),
                ("probability_derived", "probability"),
                ("intensity_reduction", "intensity-reduction"),
            )
            if column[key]
        ) or "—"
        lines.append(
            "| {index} | `{name}` | `{dtype}` | {scale:.6g} | {max:.6g} | "
            "{mean:.6g} | {p99:.6g} | {max_rel:.6g} | {flags} | "
            "**{classification}** | {proposed_atol:.6g} | "
            "{proposed_rtol:.6g} |".format(flags=flags, **column)
        )
    lines.extend(
        [
            "",
            "## Contract recommendation",
            "",
            "Retain exact `array_equal` checks for integer identities, offsets, "
            "transition targets, and geometry/structural arrays. Retain exact "
            "T4↔T4 repeatability for the complete 49-D float matrix. If and "
            "only if every cross-platform column is `BENIGN_FP`, use the "
            "reported per-column M5↔T4 tolerances as a proposed review input, "
            "not as an enacted contract.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_downloaded(
    output_dir: Path,
    *,
    json_report: Path = DEFAULT_JSON_REPORT,
    md_report: Path = DEFAULT_MD_REPORT,
    listed_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assert_authorized_scope(PARITY_IDENTITY, output_dir)
    manifest_path = output_dir / T4_MANIFEST_NAME
    run1_path = output_dir / T4_RUN1_NAME
    run2_path = output_dir / T4_RUN2_NAME
    for path in (manifest_path, run1_path, run2_path):
        if not path.is_file():
            raise DiagnosticError(f"required downloaded T4 artifact is missing: {path}")
    manifest = _load_json(manifest_path, "T4 diagnostic manifest")
    if (
        manifest.get("schema")
        != "celltrack-alpha.trackastra-parity-t4-diagnostic.v1"
        or manifest.get("status") != "PASS_OUTPUT_PERSISTED"
        or manifest.get("scope", {}).get("identity") != PARITY_IDENTITY
        or manifest.get("scope", {}).get("video_count") != 1
        or manifest.get("scope", {}).get("training") is not False
        or manifest.get("scope", {}).get("competition_submission") is not False
        or manifest.get("scope", {}).get("runpod") is not False
    ):
        raise DiagnosticError("T4 diagnostic manifest scope or status drifted")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or "T4" not in str(runtime.get("cuda_device_0", ""))
        or float(runtime.get("elapsed_seconds", math.inf)) >= 360.0
        or float(runtime.get("t4_hours", math.inf)) >= 0.1
    ):
        raise DiagnosticError("T4 diagnostic runtime attestation failed")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DiagnosticError("T4 diagnostic file manifest is missing")
    for path in (run1_path, run2_path):
        record = files.get(path.name)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise DiagnosticError(f"downloaded T4 artifact identity mismatch: {path}")

    m5, names = load_reference_row_features()
    t4_run1 = np.load(run1_path, allow_pickle=False)
    t4_run2 = np.load(run2_path, allow_pickle=False)
    if (
        t4_run1.shape != m5.shape
        or t4_run2.shape != m5.shape
        or tuple(manifest.get("feature_names", ())) != names
    ):
        raise DiagnosticError("T4 row_features shape or feature names drifted")
    t4_repeatable = bool(np.array_equal(t4_run1, t4_run2))
    mismatch_columns = [
        names[index]
        for index in range(len(names))
        if not np.array_equal(t4_run1[:, index], t4_run2[:, index])
    ]
    if (
        bool(manifest.get("t4_t4_repeatable")) != t4_repeatable
        or list(manifest.get("t4_t4_mismatch_columns", ())) != mismatch_columns
    ):
        raise DiagnosticError("kernel/local T4 repeatability result disagrees")
    columns, initial_allclose = build_column_records(m5, t4_run1, names)
    any_structural = (not t4_repeatable) or any(
        column["classification"] == "STRUCTURAL" for column in columns
    )
    recommendation = (
        "REAL_BUG_STOP" if any_structural else "ADOPT_TOLERANCE_CONTRACT"
    )
    if not t4_repeatable:
        rationale = (
            "The same pinned one-video extraction was not byte-repeatable within "
            "one T4 kernel session; stop for nondeterministic operations in "
            + ", ".join(mismatch_columns)
            + "."
        )
    elif any_structural:
        structural = [
            column["name"]
            for column in columns
            if column["classification"] == "STRUCTURAL"
        ]
        rationale = (
            "T4 repeatability passed, but M5-to-T4 drift is structural in "
            + ", ".join(structural)
            + "; do not relax parity."
        )
    else:
        rationale = (
            "T4 repeatability passed exactly and every M5-to-T4 difference is "
            "consistent with bounded floating-point drift. Recommend a "
            "human-reviewed T4-canonical tolerance contract; do not enact it "
            "in this diagnostic."
        )
    listing_names = {
        str(row.get("name") or row.get("fileName") or row.get("ref", ""))
        .replace("\\", "/")
        .rsplit("/", 1)[-1]
        for row in (listed_files or [])
    }
    required_names = {T4_MANIFEST_NAME, T4_RUN1_NAME, T4_RUN2_NAME}
    listed = required_names <= listing_names if listed_files is not None else False
    report = {
        "schema": "celltrack-alpha.trackastra-6bba-parity-diagnostic.v1",
        "status": "PASS_DIAGNOSTIC_COMPLETE",
        "diagnostic_only": True,
        "contract_change_enacted": False,
        "human_approval_required_for_final_contract": True,
        "scope": {
            "identity": PARITY_IDENTITY,
            "video_count": 1,
            "family": "6bba",
            "training": False,
            "competition_submission": False,
            "runpod": False,
            "forbidden_44b6_holdout_cv57": True,
        },
        "reference": {
            "path": str(REFERENCE_PATH.relative_to(ROOT)),
            "sha256": REFERENCE_SHA256,
        },
        "kernel": {
            "id": KERNEL_ID,
            "source_path": str(
                (KERNEL_DIR / "trackastra_parity_kernel.py").relative_to(ROOT)
            ),
            "source_sha256": sha256_file(
                KERNEL_DIR / "trackastra_parity_kernel.py"
            ),
            "production_source_sha256": PRODUCTION_KERNEL_SHA256,
            "private": True,
            "internet": False,
            "machine_shape": "NvidiaTeslaT4",
        },
        "runtime": runtime,
        "retrieval": {
            "listed": listed,
            "downloaded": True,
            "listed_files": listed_files or [],
            "output_dir": str(output_dir.relative_to(ROOT)),
        },
        "initial_probe": {
            "rtol": 1.0e-5,
            "atol": 1.0e-7,
            "equal_nan": True,
            "allclose": initial_allclose,
            "final_contract": False,
        },
        "statistics_definition": {
            "abs_diff": "np.abs(m5 - t4)",
            "max": "abs_diff.max(axis=0)",
            "mean": "abs_diff.mean(axis=0)",
            "p99": "np.quantile(abs_diff, 0.99, axis=0)",
            "scale": "np.max(np.abs(m5), axis=0)",
            "max_rel": (
                "max(abs_diff / max(abs(m5), max(scale*eps(dtype), 1e-12)))"
            ),
            "probability_derived_columns": sorted(
                PROBABILITY_DERIVED_COLUMNS
            ),
            "intensity_reduction_columns": sorted(
                INTENSITY_REDUCTION_COLUMNS
            ),
        },
        "t4_t4_mismatch_columns": mismatch_columns,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (manifest_path, run1_path, run2_path)
        },
        "DECISION": {
            "t4_t4_repeatable": t4_repeatable,
            "columns": columns,
            "any_structural": any_structural,
            "recommendation": recommendation,
            "rationale": rationale,
        },
    }
    _write_json(json_report, report)
    md_report.parent.mkdir(parents=True, exist_ok=True)
    md_report.write_text(_markdown_report(report), encoding="utf-8")
    return report


def preflight() -> dict[str, Any]:
    assert_authorized_scope(PARITY_IDENTITY, KERNEL_DIR, REFERENCE_PATH)
    load_reference_row_features()
    if sha256_file(PRODUCTION_KERNEL) != PRODUCTION_KERNEL_SHA256:
        raise DiagnosticError(
            f"pinned production kernel source drifted: {PRODUCTION_KERNEL}"
        )
    metadata = _load_json(KERNEL_DIR / "kernel-metadata.json", "kernel metadata")
    if (
        metadata.get("id") != KERNEL_ID
        or metadata.get("code_file") != "trackastra_parity_kernel.py"
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or metadata.get("competition_sources")
        != ["biohub-cell-tracking-during-development"]
        or metadata.get("dataset_sources")
        != [
            "onebitbrain/trackastra-6bba-assets-v1",
            "onebitbrain/trackastra-6bba-cache-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ]
    ):
        raise DiagnosticError("minimal kernel metadata scope drifted")
    source = KERNEL_DIR / "trackastra_parity_kernel.py"
    if not source.is_file():
        raise DiagnosticError(f"minimal kernel source is missing: {source}")
    return {
        "identity": PARITY_IDENTITY,
        "reference_sha256": REFERENCE_SHA256,
        "production_kernel_sha256": PRODUCTION_KERNEL_SHA256,
        "diagnostic_kernel_sha256": sha256_file(source),
        "kernel_id": KERNEL_ID,
        "t4_runtime_limit_seconds": 335,
        "maximum_t4_hours": 335.0 / 3600.0,
    }


def _kaggle_binary(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(value) if (value := shutil.which("kaggle")) else None,
        Path("/Library/Frameworks/Python.framework/Versions/3.13/bin/kaggle"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise DiagnosticError(
        "required Kaggle CLI is missing; expected `kaggle` or "
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/kaggle"
    )


def _kaggle_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if environment.get("KAGGLE_API_TOKEN"):
        return environment
    credentials_path = Path.home() / ".kaggle/credentials.json"
    if not credentials_path.is_file():
        raise DiagnosticError(
            "required Kaggle authentication is missing: "
            f"{credentials_path} or KAGGLE_API_TOKEN"
        )
    try:
        credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
        token = credentials["access_token"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(
            f"cannot read authenticated Kaggle credential: {credentials_path}"
        ) from exc
    if not isinstance(token, str) or len(token) < 32:
        raise DiagnosticError(
            f"authenticated Kaggle access token is invalid: {credentials_path}"
        )
    environment["KAGGLE_API_TOKEN"] = token
    return environment


def _run(
    command: list[str],
    *,
    timeout: float | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    output = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    if completed.returncode != 0:
        raise DiagnosticError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def _status(
    kaggle: Path, environment: dict[str, str]
) -> tuple[str, str]:
    output = _run(
        [str(kaggle), "kernels", "status", KERNEL_ID],
        timeout=60,
        environment=environment,
    )
    upper = output.upper()
    for status in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED"):
        if status in upper:
            return status, output
    return "UNKNOWN", output


def _parse_file_listing(value: str) -> list[dict[str, Any]]:
    lines = [line for line in value.splitlines() if line.strip()]
    header = next(
        (index for index, line in enumerate(lines) if line.startswith("name,")),
        None,
    )
    if header is None:
        return []
    try:
        return [
            dict(row)
            for row in csv.DictReader(io.StringIO("\n".join(lines[header:])))
            if row.get("name")
            and not str(row["name"]).startswith(("Warning:", "Next Page Token"))
        ]
    except csv.Error as exc:
        raise DiagnosticError("cannot parse Kaggle output-file listing") from exc


def _list_kaggle_files(
    kaggle: Path, environment: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    for _page in range(10):
        command = [
            str(kaggle),
            "kernels",
            "files",
            "-v",
            "--page-size",
            "200",
        ]
        if page_token is not None:
            command.extend(("--page-token", page_token))
        command.append(KERNEL_ID)
        raw = _run(
            command,
            timeout=120,
            environment=environment,
        )
        rows.extend(_parse_file_listing(raw))
        match = re.search(r"Next Page Token = (\S+)", raw)
        if match is None:
            return rows
        page_token = match.group(1)
        if page_token in seen_tokens:
            raise DiagnosticError("Kaggle output pagination repeated a token")
        seen_tokens.add(page_token)
    raise DiagnosticError("Kaggle output listing exceeded ten pages")


def retrieve(
    *,
    kaggle: Path,
    environment: dict[str, str],
    output_dir: Path,
    json_report: Path,
    md_report: Path,
) -> dict[str, Any]:
    status, raw_status = _status(kaggle, environment)
    if status != "COMPLETE":
        raise DiagnosticError(
            f"Kaggle diagnostic is not complete: {status}\n{raw_status}"
        )
    listing = _list_kaggle_files(kaggle, environment)
    required = {T4_MANIFEST_NAME, T4_RUN1_NAME, T4_RUN2_NAME}
    listing_names = {
        str(row.get("name") or row.get("fileName") or row.get("ref", ""))
        .replace("\\", "/")
        .rsplit("/", 1)[-1]
        for row in listing
    }
    if not required <= listing_names:
        raise DiagnosticError(
            f"Kaggle outputs are not listed/downloadable: "
            f"missing={sorted(required-listing_names)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(kaggle),
            "kernels",
            "output",
            "-p",
            str(output_dir),
            "-o",
            "--file-pattern",
            (
                "^trackastra_parity_t4_"
                "(manifest\\.json|run[12]_row_features\\.npy)$"
            ),
            KERNEL_ID,
        ],
        timeout=300,
        environment=environment,
    )
    return compare_downloaded(
        output_dir,
        json_report=json_report,
        md_report=md_report,
        listed_files=listing,
    )


def launch(
    *,
    kaggle_path: Path | None,
    output_dir: Path,
    json_report: Path,
    md_report: Path,
    poll_seconds: float,
    wait_minutes: float,
) -> dict[str, Any]:
    preflight()
    existing_required = [
        output_dir / name
        for name in (T4_MANIFEST_NAME, T4_RUN1_NAME, T4_RUN2_NAME)
        if (output_dir / name).exists()
    ]
    if existing_required:
        raise DiagnosticError(
            "refusing to overwrite existing diagnostic outputs: "
            + ", ".join(str(path) for path in existing_required)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle = _kaggle_binary(kaggle_path)
    environment = _kaggle_environment()
    push_output = _run(
        [
            str(kaggle),
            "kernels",
            "push",
            "-p",
            str(KERNEL_DIR),
            "-t",
            "335",
        ],
        timeout=180,
        environment=environment,
    )
    deadline = time.monotonic() + wait_minutes * 60.0
    snapshots: list[dict[str, str]] = []
    while True:
        status, raw = _status(kaggle, environment)
        snapshots.append({"status": status, "raw": raw})
        if status == "COMPLETE":
            break
        if status in {"ERROR", "CANCELLED"}:
            logs = _run(
                [str(kaggle), "kernels", "logs", KERNEL_ID],
                timeout=120,
                environment=environment,
            )
            raise DiagnosticError(
                f"Kaggle diagnostic ended {status}; logs follow:\n{logs}"
            )
        if time.monotonic() >= deadline:
            raise DiagnosticError(
                f"Kaggle diagnostic did not finish within {wait_minutes} minutes"
            )
        time.sleep(poll_seconds)
    report = retrieve(
        kaggle=kaggle,
        environment=environment,
        output_dir=output_dir,
        json_report=json_report,
        md_report=md_report,
    )
    report["launch"] = {
        "push_output": push_output,
        "status_snapshots": snapshots,
    }
    _write_json(json_report, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    compare.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    compare.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("--kaggle", type=Path)
    retrieve_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    retrieve_parser.add_argument(
        "--json-report", type=Path, default=DEFAULT_JSON_REPORT
    )
    retrieve_parser.add_argument(
        "--md-report", type=Path, default=DEFAULT_MD_REPORT
    )
    run = subparsers.add_parser("launch")
    run.add_argument("--kaggle", type=Path)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    run.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    run.add_argument("--poll-seconds", type=float, default=5.0)
    run.add_argument("--wait-minutes", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "preflight":
            result = preflight()
        elif args.command == "compare":
            result = compare_downloaded(
                args.output_dir.resolve(),
                json_report=args.json_report.resolve(),
                md_report=args.md_report.resolve(),
            )
        elif args.command == "retrieve":
            preflight()
            result = retrieve(
                kaggle=_kaggle_binary(
                    args.kaggle.resolve() if args.kaggle else None
                ),
                environment=_kaggle_environment(),
                output_dir=args.output_dir.resolve(),
                json_report=args.json_report.resolve(),
                md_report=args.md_report.resolve(),
            )
        else:
            result = launch(
                kaggle_path=args.kaggle.resolve() if args.kaggle else None,
                output_dir=args.output_dir.resolve(),
                json_report=args.json_report.resolve(),
                md_report=args.md_report.resolve(),
                poll_seconds=args.poll_seconds,
                wait_minutes=args.wait_minutes,
            )
    except DiagnosticError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
