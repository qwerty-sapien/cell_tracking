#!/usr/bin/env python3
"""Build the deterministic private Kaggle package for the V45 sparse runner."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "bh45pkg"
CV57_OUTPUT = ROOT / "kaggle_kernel" / "bhcv57pkg"
DEFAULT_REGISTRY = ROOT / "reports" / "g02" / "dataset_registry_v1.json"
DEFAULT_SUPPORT = ROOT / "external" / "Tracking_Support"
DEFAULT_SPLIT = ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json"

SOURCE_LIMIT = 1_000_000
SUBMISSION_DEADLINE_MINUTES = 360
KERNEL_SLUG = "bhv45-submit"
KERNEL_REF = f"onebitbrain/{KERNEL_SLUG}"
CV57_KERNEL_SLUG = "bhv45-cv57"
APPEARANCE_SUBMIT_KERNEL_SLUG = "bhap-submit"
APPEARANCE_CV57_KERNEL_SLUG = "bhap-cv57"
P0_CALIBRATIONS = {
    "p0-c0": {
        "kernel_slug": "bhp0-c0",
        "motion_direction": "backward",
        "motion_error_weight": 0.0,
        "learned_bonus": 1.0,
        "local_cv57_score": 0.8483945155545333,
    },
    "p0-c2": {
        "kernel_slug": "bhp0-c2",
        "motion_direction": "backward",
        "motion_error_weight": 2.0,
        "learned_bonus": 1.0,
        "local_cv57_score": 0.8721499474601118,
    },
    "p0-lb4": {
        "kernel_slug": "bhp0-lb4",
        "motion_direction": "forward",
        "motion_error_weight": 1.0,
        "learned_bonus": 4.0,
        "local_cv57_score": 0.8720910462365732,
    },
    "p0-lb8": {
        "kernel_slug": "bhp0-lb8",
        "motion_direction": "forward",
        "motion_error_weight": 1.0,
        "learned_bonus": 8.0,
        "local_cv57_score": 0.8670102112352311,
    },
    "p0-lb16": {
        "kernel_slug": "bhp0-lb16",
        "motion_direction": "forward",
        "motion_error_weight": 1.0,
        "learned_bonus": 16.0,
        "local_cv57_score": 0.8479018413756242,
    },
}
KAGGLE_ACCELERATOR_ID = "NvidiaTeslaT4"
RELEASED_GRAPH_VARIANT = "v45-motion-lazyrefine-locked"
SUPPORT_DATASET_SLUG = "biohub-tracking-support-pack-50ep-v1"
SUPPORT_REF = f"pilkwang/{SUPPORT_DATASET_SLUG}"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"
SUPPORT_MOUNT_CANDIDATES = (
    f"/kaggle/input/datasets/pilkwang/{SUPPORT_DATASET_SLUG}",
    f"/kaggle/input/{SUPPORT_DATASET_SLUG}",
)

ENTRYPOINT_NAME = "bh45.py"
ARCHIVE_NAME = "bh45src.zip"
ATTACHMENT_NAME = "bh45att.json"
MANIFEST_NAME = "bh45man.json"
METADATA_NAME = "kernel-metadata.json"  # Kaggle requires this fixed basename.

EXPECTED_ARTIFACT_HASHES = {
    "manifest_sha256": "bc20f1f04cfb682af3b27a836ce9a57f44a2fe27508dc59039200b2a23188103",
    "config_sha256": "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a",
    "source_sha256": "8c4621658956f724b161236b451d7d0ed30759aa3f3a5ea28021b4eeb1171e9b",
    "weight_sha256": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
}

SUPPORT_SOURCE_PATHS = (
    "repo/src/biohub_tracking/models/temporal_unet.py",
    "repo/src/biohub_tracking/models/simple_node_transformer.py",
    "repo/scripts/train_unet_transformer.py",
)
SUPPORT_ARTIFACT_PATHS = (
    "ARTIFACT_MANIFEST.json",
    "weights/unet_transformer/split_0/config.json",
    "weights/unet_transformer/split_0/edge_predictor_best.pth",
    *SUPPORT_SOURCE_PATHS,
)
SOURCE_PATHS = (
    "scripts/run_bh.py",
    "src/celltrack_alpha/bh_model.py",
    "src/celltrack_alpha/bh_graph.py",
    "src/celltrack_alpha/bh_edge.py",
    "src/celltrack_alpha/bh_route.py",
)
EDGE_SOURCE_PATHS = (
    "src/celltrack_alpha/v45_fast.py",
    "src/celltrack_alpha/v45_head.py",
    "src/celltrack_alpha/t_adapter.py",
)
CV57_EDGE_ARTIFACT = "reports/g02/conflict_edge_f0_fit117_v1/edge.json"
CV57_EDGE_ARTIFACT_SHA256 = "5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034"
CV57_AUXILIARY_CHECKPOINT = "reports/g02/v45f4.pt"
CV57_AUXILIARY_CHECKPOINT_SHA256 = "1cff90475b819b5652310ccb73ba57e1dc20069885962a06a5e10a6060c04473"
CV57_EXCLUDED_IDENTITY = "train:6bba_f4ae811c"
CV57_SELECTION_SHA256 = "58d090bc83702065b3bc8bc035c5d8e8911519c71a8d3a58f995e5abaf3099f1"
APPEARANCE_CONTRACT = "reports/g02/appearance_direct_swap_contract_v1.json"
APPEARANCE_CONTRACT_SHA256 = "cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7"
APPEARANCE_GATE = "reports/g02/selected_graph_background_sparse_w050_v1/gate.pt"
APPEARANCE_GATE_SHA256 = "04a4cec214b5848363af08ab9a092f5ac8cc077b83f946ea81fc986d36e91ed1"

DEPENDENCIES = (
    ("torch", "torch", "2.2", "3", ">=2.2,<3"),
    ("numpy", "numpy", "1.26", "3", ">=1.26,<3"),
    ("scipy", "scipy", "1.11", "2", ">=1.11,<2"),
    ("zarr", "zarr", "3.0.10", "4", ">=3.0.10,<4"),
    ("numcodecs", "numcodecs", "0.14", "0.16", ">=0.14,<0.16"),
    ("donfig", "donfig", "0.8", "1", ">=0.8,<1"),
    ("google-crc32c", "google_crc32c", "1.5", "2", ">=1.5,<2"),
    ("packaging", "packaging", "22", "27", ">=22,<27"),
    ("typing-extensions", "typing_extensions", "4.13", "5", ">=4.13,<5"),
    ("Deprecated", "deprecated", "1.2", "2", ">=1.2,<2"),
    ("wrapt", "wrapt", "1.16", "3", ">=1.16,<3"),
)

_SAFE_SLUG = re.compile(r"[a-z0-9-]+\Z")


class PackageError(RuntimeError):
    """The private package cannot be built without violating its contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise PackageError(f"required package input is missing: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_slug(slug: str, label: str) -> None:
    if len(slug) > 15 or _SAFE_SLUG.fullmatch(slug) is None:
        raise PackageError(f"{label} must be a lowercase Kaggle slug of at most 15 characters: {slug!r}")


def compact_registry(
    path: Path,
    *,
    expected_test_count: int = 4,
    split: str = "test",
    selected_video_ids: Sequence[str] | None = None,
) -> bytes:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read source registry {path}: {exc}") from exc
    if not isinstance(source, Mapping) or source.get("artifact") != "dataset_registry_v1":
        raise PackageError("source registry is not dataset_registry_v1")
    rows = source.get("records")
    if not isinstance(rows, list):
        raise PackageError("source registry records must be a list")
    if split not in {"train", "test"}:
        raise PackageError(f"compact registry split is invalid: {split!r}")
    selected = tuple(selected_video_ids) if selected_video_ids is not None else None
    if selected is not None and (
        not selected
        or len(set(selected)) != len(selected)
        or any(not isinstance(item, str) or not item.startswith(f"{split}:") for item in selected)
    ):
        raise PackageError("compact registry selection must be nonempty, unique, and match its split")
    selected_set = set(selected) if selected is not None else None
    records = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("split") != split:
            continue
        try:
            shape = [int(item) for item in row["compact_metadata"]["image_shape"]]
            record = {
                "split": split,
                "dataset": str(row["dataset"]),
                "video_id": str(row["video_id"]),
                "exact_identity_sha256": str(row["exact_identity_sha256"]),
                "compact_metadata": {"image_shape": shape},
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise PackageError("a test registry record lacks a required runner identity field") from exc
        if record["video_id"] != f"{split}:{record['dataset']}" or len(shape) != 4 or any(item <= 0 for item in shape):
            raise PackageError(f"invalid compact {split} registry identity: {record['video_id']!r}")
        digest = record["exact_identity_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PackageError(f"invalid exact identity SHA-256 for {record['video_id']}")
        if selected_set is None or record["video_id"] in selected_set:
            records.append(record)
    records.sort(key=lambda item: item["video_id"])
    expected_count = len(selected) if selected is not None else expected_test_count
    observed_ids = [item["video_id"] for item in records]
    if (
        len(records) != expected_count
        or len(set(observed_ids)) != len(records)
        or (selected is not None and set(observed_ids) != selected_set)
    ):
        raise PackageError(
            f"compact registry requires {expected_count} unique {split} records, found {len(records)}"
        )
    payload = {
        "artifact": "dataset_registry_v1",
        "schema_version": 1,
        "source_registry_sha256": sha256_file(path),
        "counts": {f"{split}_videos": len(records), "videos": len(records)},
        "records": records,
    }
    return canonical_json(payload) + b"\n"


def cv57_video_ids(split_path: Path = DEFAULT_SPLIT) -> tuple[str, ...]:
    """Return the frozen terminal 57-video fold-0 holdout selection."""

    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        fold = payload["folds"]["0"]
        held_out = fold["held_out_video_ids"]
        audit_ids = payload["audit"]["video_ids"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read fold-0 split manifest {split_path}: {exc}") from exc
    if (
        not isinstance(held_out, list)
        or len(held_out) != 58
        or len(set(held_out)) != 58
        or any(not isinstance(item, str) or not item.startswith("train:") for item in held_out)
        or CV57_EXCLUDED_IDENTITY not in held_out
        or set(held_out) & set(audit_ids)
    ):
        raise PackageError("fold-0 holdout does not match the frozen 58-video pre-exclusion contract")
    selected = tuple(sorted(item for item in held_out if item != CV57_EXCLUDED_IDENTITY))
    observed_sha256 = sha256_bytes(canonical_json(list(selected)))
    if len(selected) != 57 or observed_sha256 != CV57_SELECTION_SHA256:
        raise PackageError(
            f"terminal fold-0 selection identity mismatch: expected {CV57_SELECTION_SHA256}, got {observed_sha256}"
        )
    return selected


def source_tree_sha256(support_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SUPPORT_SOURCE_PATHS:
        path = support_root / relative
        if not path.is_file():
            raise PackageError(f"authoritative support source is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def deterministic_zip(members: Mapping[str, bytes]) -> bytes:
    from io import BytesIO

    target = BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            parts = Path(name).parts
            if Path(name).is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
                raise PackageError(f"unsafe source archive member: {name!r}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return target.getvalue()


def dependency_install_specs(statuses: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return only packages whose import/version probe was not compatible."""

    result = []
    for distribution, _module, _minimum, _maximum, specifier in DEPENDENCIES:
        status = statuses.get(distribution)
        if not isinstance(status, Mapping) or status.get("compatible") is not True:
            result.append(f"{distribution}{specifier}")
    return result


def pip_command(python: str, wheel_dir: str, specs: Sequence[str]) -> list[str]:
    return [
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--find-links",
        wheel_dir,
        *specs,
    ]


def submit_run_spec() -> dict[str, Any]:
    return {
        "profile": "submit",
        "mode": "submit",
        "split": "test",
        "datasets": None,
        "output_file": "submission.csv",
        "graph_variant": RELEASED_GRAPH_VARIANT,
        "topology_mode": "legacy",
        "detection_tta": "D4-8-view",
        "division_enabled": False,
        "edge_artifact": edge_artifact_run_spec(),
    }


def p0_calibration_run_spec(experiment_id: str) -> dict[str, Any]:
    if experiment_id not in P0_CALIBRATIONS:
        raise PackageError(f"unsupported P0 calibration: {experiment_id!r}")
    calibration = P0_CALIBRATIONS[experiment_id]
    run = submit_run_spec()
    run["profile"] = "p0-public-calibration"
    run["p0_calibration"] = {
        "schema": "celltrack-alpha.p0-public-calibration.v1",
        "experiment_id": experiment_id,
        "purpose": "public-metric-parity-calibration",
        "motion_direction": calibration["motion_direction"],
        "motion_error_weight": calibration["motion_error_weight"],
        "learned_bonus": calibration["learned_bonus"],
        "local_cv57_score": calibration["local_cv57_score"],
        "baseline_submission_id": 54910497,
        "baseline_motion_error_weight": 0.25,
        "production_eligible": False,
        "submit_eligible": True,
    }
    return run


def edge_artifact_run_spec() -> dict[str, str]:
    return {
        "path": CV57_EDGE_ARTIFACT,
        "sha256": CV57_EDGE_ARTIFACT_SHA256,
        "auxiliary_checkpoint_path": CV57_AUXILIARY_CHECKPOINT,
        "auxiliary_checkpoint_sha256": CV57_AUXILIARY_CHECKPOINT_SHA256,
    }


def cv57_run_spec(split_path: Path = DEFAULT_SPLIT) -> dict[str, Any]:
    selected = cv57_video_ids(split_path)
    return {
        "profile": "fold0-holdout-57",
        "mode": "cv",
        "split": "train",
        "datasets": list(selected),
        "selection_sha256": CV57_SELECTION_SHA256,
        "excluded_identity": CV57_EXCLUDED_IDENTITY,
        "output_file": "cv57.csv",
        "graph_variant": RELEASED_GRAPH_VARIANT,
        "topology_mode": "legacy",
        "detection_tta": "D4-8-view",
        "division_enabled": False,
        "edge_artifact": edge_artifact_run_spec(),
    }


def appearance_run_spec(
    mode: str,
    split_path: Path = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Return the sole hash-pinned appearance probe contract."""

    if mode == "submit":
        run = submit_run_spec()
        run["profile"] = "appearance-direct-swap-submit"
    elif mode == "cv57":
        run = cv57_run_spec(split_path)
        run["profile"] = "appearance-direct-swap-cv57"
    else:
        raise PackageError(f"invalid appearance run mode: {mode!r}")
    run["appearance_swap"] = {
        "contract_path": APPEARANCE_CONTRACT,
        "contract_sha256": APPEARANCE_CONTRACT_SHA256,
        "geometry_gate_path": APPEARANCE_GATE,
        "geometry_gate_sha256": APPEARANCE_GATE_SHA256,
        "local_score": 0.8815220699597315,
        "edge_tp": 34320,
        "edge_fp": 2459,
        "edge_fn": 1948,
        "production_eligible": False,
        "submit_eligible": True,
    }
    return run


def validate_run_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    run = json.loads(canonical_json(value).decode("ascii"))
    mode = run.get("mode")
    split = run.get("split")
    datasets = run.get("datasets")
    edge = run.get("edge_artifact")
    calibration = run.get("p0_calibration")
    appearance = run.get("appearance_swap")
    if mode not in {"cv", "submit"} or split not in {"train", "test"}:
        raise PackageError("run specification has an invalid mode or split")
    if (
        run.get("graph_variant") != RELEASED_GRAPH_VARIANT
        or run.get("topology_mode") != "legacy"
        or run.get("detection_tta") != "D4-8-view"
    ):
        raise PackageError("run specification must use the released graph variant, legacy topology, and D4-8 detection")
    if run.get("division_enabled") is not False:
        raise PackageError("run specification must keep divisions disabled")
    output_file = run.get("output_file")
    if not isinstance(output_file, str) or Path(output_file).name != output_file or not output_file.endswith(".csv"):
        raise PackageError("run specification output must be one CSV basename")
    if mode == "submit":
        if split != "test" or datasets is not None:
            raise PackageError("submit run specification must target the complete test split")
    else:
        if (
            split != "train"
            or not isinstance(datasets, list)
            or not datasets
            or len(set(datasets)) != len(datasets)
            or any(not isinstance(item, str) or not item.startswith("train:") for item in datasets)
        ):
            raise PackageError("CV run specification has an invalid train selection")
    if not isinstance(edge, Mapping) or dict(edge) != edge_artifact_run_spec():
        raise PackageError("run specification must use the pinned conflict edge artifact")
    if calibration is not None:
        experiment_id = calibration.get("experiment_id") if isinstance(calibration, Mapping) else None
        expected = (
            p0_calibration_run_spec(experiment_id)["p0_calibration"]
            if experiment_id in P0_CALIBRATIONS
            else None
        )
        if (
            mode != "submit"
            or run.get("profile") != "p0-public-calibration"
            or not isinstance(calibration, Mapping)
            or dict(calibration) != expected
        ):
            raise PackageError("run specification has an invalid P0 calibration contract")
    elif run.get("profile") == "p0-public-calibration":
        raise PackageError("P0 calibration profile is missing its calibration contract")
    if appearance is not None:
        expected = appearance_run_spec(
            "submit" if mode == "submit" else "cv57"
        )["appearance_swap"]
        expected_profile = (
            "appearance-direct-swap-submit"
            if mode == "submit"
            else "appearance-direct-swap-cv57"
        )
        if (
            calibration is not None
            or run.get("profile") != expected_profile
            or not isinstance(appearance, Mapping)
            or dict(appearance) != expected
        ):
            raise PackageError(
                "run specification has an invalid appearance-swap contract"
            )
    elif str(run.get("profile", "")).startswith("appearance-direct-swap"):
        raise PackageError(
            "appearance-swap profile is missing its contract"
        )
    for key in ("path", "auxiliary_checkpoint_path"):
        path = edge.get(key)
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise PackageError(f"run specification has an unsafe {key}")
    for key in ("sha256", "auxiliary_checkpoint_sha256"):
        digest = edge.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PackageError(f"run specification has an invalid {key}")
    return run


def runner_command(
    python: str,
    source_root: str,
    competition_root: str,
    support_root: str,
    output_root: str,
    *,
    view_batch_size: int = 1,
    score_gate: str | None = None,
    run_spec: Mapping[str, Any] | None = None,
) -> list[str]:
    if type(view_batch_size) is not int or not 1 <= view_batch_size <= 8:
        raise PackageError("view_batch_size must be an integer in [1, 8]")
    if view_batch_size == 1 and score_gate is not None:
        raise PackageError("sequential view_batch_size=1 cannot use an accelerated score gate")
    if view_batch_size > 1 and not score_gate:
        raise PackageError("view_batch_size 2..8 requires an explicit score gate")
    source = Path(source_root)
    run = validate_run_spec(run_spec or submit_run_spec())
    command = [
        python,
        str(source / "scripts" / "run_bh.py"),
        "--input",
        competition_root,
        "--registry",
        str(source / "bhreg.json"),
        "--support-root",
        support_root,
        "--out-dir",
        output_root,
        "--output-file",
        run["output_file"],
        "--mode",
        run["mode"],
        "--split",
        run["split"],
        "--devices",
        "cuda:0,cuda:1",
        "--pair-batch-size",
        "1",
        "--view-batch-size",
        str(view_batch_size),
        "--deadline-min",
        str(SUBMISSION_DEADLINE_MINUTES),
        "--source-budget-bytes",
        "999999",
        "--graph-variant",
        run["graph_variant"],
    ]
    if run["datasets"] is not None:
        command.extend(("--datasets", ",".join(run["datasets"])))
    edge = run["edge_artifact"]
    if edge is not None:
        command.extend(
            (
                "--edge-artifact",
                str(source / edge["path"]),
                "--edge-artifact-sha256",
                edge["sha256"],
            )
        )
    calibration = run.get("p0_calibration")
    if calibration is not None:
        command.extend(
            (
                "--p0-experiment-id",
                calibration["experiment_id"],
                "--p0-motion-direction",
                calibration["motion_direction"],
                "--p0-motion-error-weight",
                str(calibration["motion_error_weight"]),
                "--p0-learned-bonus",
                str(calibration["learned_bonus"]),
            )
        )
    appearance = run.get("appearance_swap")
    if appearance is not None:
        command.extend(
            (
                "--appearance-swap-contract",
                str(source / appearance["contract_path"]),
                "--appearance-swap-contract-sha256",
                appearance["contract_sha256"],
            )
        )
    if score_gate is not None:
        command.extend(("--accelerated-score-gate", score_gate))
    return command


ENTRYPOINT_SOURCE = r'''#!/usr/bin/env python3
"""Offline, fail-closed launcher for the private V45 sparse Kaggle kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
MANIFEST = HERE / "bh45man.json"
ATTACHMENT = HERE / "bh45att.json"
SOURCE_ARCHIVE = HERE / "bh45src.zip"
SOURCE_ROOT = WORK / "bh45src"
DIAGNOSTIC = WORK / "bhdiag.json"
DEADLINE_SECONDS = 360 * 60
KSHAPE = "NvidiaTeslaT4"
RELEASED_GRAPH_VARIANT = "v45-motion-lazyrefine-locked"
GPU_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_VERSION", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_REQUIRE_CUDA",
)
GPU_DIAG = None
P0_CALIBRATIONS = {
    "p0-c0": ("backward", 0.0, 1.0, 0.8483945155545333),
    "p0-c2": ("backward", 2.0, 1.0, 0.8721499474601118),
    "p0-lb4": ("forward", 1.0, 4.0, 0.8720910462365732),
    "p0-lb8": ("forward", 1.0, 8.0, 0.8670102112352311),
    "p0-lb16": ("forward", 1.0, 16.0, 0.8479018413756242),
}
DEPENDENCIES = (
    ("torch", "torch", "2.2", "3", ">=2.2,<3"),
    ("numpy", "numpy", "1.26", "3", ">=1.26,<3"),
    ("scipy", "scipy", "1.11", "2", ">=1.11,<2"),
    ("zarr", "zarr", "3.0.10", "4", ">=3.0.10,<4"),
    ("numcodecs", "numcodecs", "0.14", "0.16", ">=0.14,<0.16"),
    ("donfig", "donfig", "0.8", "1", ">=0.8,<1"),
    ("google-crc32c", "google_crc32c", "1.5", "2", ">=1.5,<2"),
    ("packaging", "packaging", "22", "27", ">=22,<27"),
    ("typing-extensions", "typing_extensions", "4.13", "5", ">=4.13,<5"),
    ("Deprecated", "deprecated", "1.2", "2", ">=1.2,<2"),
    ("wrapt", "wrapt", "1.16", "3", ">=1.16,<3"),
)


class LaunchError(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key %r" % key)
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding="ascii"), object_pairs_hook=reject)
    except Exception as exc:
        raise LaunchError("invalid package JSON %s: %s" % (path, exc)) from exc


def _release(value):
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value))
    if match is None:
        return None
    return tuple(int(item) for item in match.group(1).split("."))


def _compare(left, right):
    width = max(len(left), len(right))
    return (left + (0,) * (width - len(left))) < (right + (0,) * (width - len(right)))


def _compatible(version, minimum, maximum):
    observed = _release(version)
    low = _release(minimum)
    high = _release(maximum)
    return observed is not None and low is not None and high is not None and not _compare(observed, low) and _compare(observed, high)


def _verify_package():
    payload = _load(MANIFEST)
    identity = payload.get("identity")
    if not isinstance(identity, dict) or payload.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest():
        raise LaunchError("package manifest identity SHA-256 mismatch")
    files = identity.get("files")
    if not isinstance(files, list):
        raise LaunchError("package manifest file list is missing")
    observed_bytes = MANIFEST.stat().st_size
    for item in files:
        path = HERE / str(item.get("path", ""))
        if not path.is_file():
            raise LaunchError("kernel package member is missing: %s" % path)
        actual_hash = _sha_file(path)
        actual_bytes = path.stat().st_size
        if actual_hash != item.get("sha256") or actual_bytes != item.get("bytes"):
            raise LaunchError(
                "kernel member identity mismatch for %s: expected bytes=%s sha256=%s, got bytes=%s sha256=%s"
                % (path.name, item.get("bytes"), item.get("sha256"), actual_bytes, actual_hash)
            )
        observed_bytes += actual_bytes
    if observed_bytes != identity.get("package_bytes") or observed_bytes >= identity.get("budget_bytes", 0):
        raise LaunchError("aggregate kernel source budget mismatch: bytes=%s manifest=%s" % (observed_bytes, identity.get("package_bytes")))
    return payload


def _probe_dependencies():
    # Probe in a disposable interpreter so an incompatible module is never
    # retained in this launcher's sys.modules across an offline pip repair.
    probe = r"""import importlib
import importlib.metadata
import json
import sys
dependencies = json.loads(sys.stdin.read())
result = {}
for distribution, module, _minimum, _maximum, specifier in dependencies:
    record = {"distribution": distribution, "import": module, "specifier": specifier}
    try:
        version = importlib.metadata.version(distribution)
        imported = importlib.import_module(module)
        record.update({"version": version, "import_file": str(getattr(imported, "__file__", None)), "error": None})
    except BaseException as exc:
        record.update({"version": None, "import_file": None, "error": "%s: %s" % (type(exc).__name__, exc)})
    result[distribution] = record
print("BHDEP=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        input=json.dumps(DEPENDENCIES),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    marker = "BHDEP="
    line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)), None)
    if completed.returncode != 0 or line is None:
        raise LaunchError(
            "dependency import/version probe failed before image reads: returncode=%s stderr=%s"
            % (completed.returncode, completed.stderr[-4000:])
        )
    try:
        statuses = json.loads(line[len(marker):])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LaunchError("dependency probe returned malformed diagnostics") from exc
    for distribution, _module, minimum, maximum, _specifier in DEPENDENCIES:
        record = statuses[distribution]
        record["compatible"] = record["error"] is None and _compatible(record["version"], minimum, maximum)
    return statuses


def _source_tree_sha(root, relatives):
    digest = hashlib.sha256()
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _validate_attachment(diagnostics):
    payload = _load(ATTACHMENT)
    identity = payload.get("identity")
    if not isinstance(identity, dict) or payload.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest():
        raise LaunchError("attachment manifest identity SHA-256 mismatch")
    candidates = identity.get("mount_candidates")
    if not isinstance(candidates, list) or not candidates or any(not isinstance(item, str) for item in candidates):
        raise LaunchError("support attachment mount candidate list is missing")
    required_files = identity.get("required_files", [])
    source_identity = identity.get("source_tree", {})
    expected_wheels = identity.get("wheelhouse", {}).get("files", [])
    expected_names = sorted(str(item.get("path")) for item in expected_wheels)
    candidate_results = []
    for mount in candidates:
        support = Path(mount)
        artifact_results = []
        for item in required_files:
            path = support / str(item.get("path", ""))
            actual_bytes = path.stat().st_size if path.is_file() else None
            actual_hash = _sha_file(path) if path.is_file() else None
            artifact_results.append({
                "path": str(path), "expected_bytes": item.get("bytes"), "actual_bytes": actual_bytes,
                "expected_sha256": item.get("sha256"), "actual_sha256": actual_hash,
                "ok": actual_bytes == item.get("bytes") and actual_hash == item.get("sha256"),
            })
        actual_source = _source_tree_sha(support, source_identity.get("paths", []))
        wheel_dir = support / str(identity.get("wheelhouse", {}).get("path", "wheels"))
        actual_names = sorted(path.name for path in wheel_dir.glob("*.whl")) if wheel_dir.is_dir() else []
        wheel_errors = []
        if actual_names != expected_names:
            wheel_errors.append({"expected_names": expected_names, "actual_names": actual_names})
        if not any(not item["ok"] for item in artifact_results) and actual_source == source_identity.get("sha256"):
            for item in expected_wheels:
                path = wheel_dir / str(item.get("path", ""))
                actual_bytes = path.stat().st_size if path.is_file() else None
                actual_hash = _sha_file(path) if path.is_file() else None
                if actual_bytes != item.get("bytes") or actual_hash != item.get("sha256"):
                    wheel_errors.append({
                        "path": str(path), "expected_bytes": item.get("bytes"), "actual_bytes": actual_bytes,
                        "expected_sha256": item.get("sha256"), "actual_sha256": actual_hash,
                    })
        result = {
            "mount": mount,
            "support_files": artifact_results,
            "source_tree": {"expected_sha256": source_identity.get("sha256"), "actual_sha256": actual_source},
            "wheelhouse": {
                "path": str(wheel_dir), "expected_count": len(expected_wheels),
                "actual_count": len(actual_names), "errors": wheel_errors,
            },
        }
        result["ok"] = (
            not any(not item["ok"] for item in artifact_results)
            and actual_source == source_identity.get("sha256")
            and not wheel_errors
        )
        candidate_results.append(result)
        if result["ok"]:
            diagnostics["support_candidates"] = candidate_results
            diagnostics["support_selected"] = mount
            return support, wheel_dir
    diagnostics["support_candidates"] = candidate_results
    raise LaunchError(
        "public support dataset identity mismatch at every known mount before image reads; inspect bhdiag.json for expected/actual checkpoint, source, and wheel SHA-256 values"
    )


def _install_missing(statuses, wheel_dir, diagnostics):
    specs = []
    for distribution, _module, _minimum, _maximum, specifier in DEPENDENCIES:
        if statuses[distribution]["compatible"] is not True:
            specs.append(distribution + specifier)
    compatible_core = [name for name in ("torch", "numpy", "scipy") if statuses[name]["compatible"]]
    if any(spec.split("<", 1)[0].split(">", 1)[0] in compatible_core for spec in specs):
        raise LaunchError("internal dependency plan would reinstall a compatible torch/numpy/scipy")
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", str(wheel_dir), *specs]
    diagnostics["dependency_plan"] = {"install_specs": specs, "command": command if specs else None}
    if not specs:
        return statuses
    environment = dict(os.environ)
    environment.update({"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    result = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    diagnostics["pip"] = {"returncode": result.returncode, "output_tail": result.stdout[-12000:]}
    if result.returncode != 0:
        raise LaunchError(
            "offline dependency repair failed; no network fallback is permitted. Missing/incompatible=%s; wheel input=%s; inspect bhdiag.json"
            % (specs, wheel_dir)
        )
    repaired = _probe_dependencies()
    diagnostics["dependencies_after"] = repaired
    failed = {name: value for name, value in repaired.items() if value["compatible"] is not True}
    if failed:
        raise LaunchError("offline dependency repair remained incompatible: %s" % sorted(failed))
    return repaired


def _extract_source(package_manifest):
    if SOURCE_ROOT.exists():
        raise LaunchError("refusing to overwrite existing source extraction directory: %s" % SOURCE_ROOT)
    archive_identity = package_manifest["identity"].get("archive", {})
    expected = {item["path"]: item for item in archive_identity.get("members", [])}
    with zipfile.ZipFile(SOURCE_ARCHIVE, "r") as archive:
        names = sorted(item.filename for item in archive.infolist() if not item.is_dir())
        if names != sorted(expected):
            raise LaunchError("source archive member set mismatch")
        for name in names:
            parts = Path(name).parts
            if Path(name).is_absolute() or not parts or any(part in ("", ".", "..") for part in parts):
                raise LaunchError("unsafe source archive member: %r" % name)
            value = archive.read(name)
            item = expected[name]
            if len(value) != item["bytes"] or hashlib.sha256(value).hexdigest() != item["sha256"]:
                raise LaunchError("source archive member SHA-256 mismatch: %s" % name)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(SOURCE_ARCHIVE, "r") as archive:
        archive.extractall(SOURCE_ROOT)


def _diagnostic_call(function):
    try:
        return function()
    except BaseException as exc:
        return {"error_type": type(exc).__name__, "error_message": str(exc)}


def _diagnostic_command(command):
    try:
        result = subprocess.run(
            command, check=False, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10,
        )
        return {
            "command": command, "returncode": int(result.returncode),
            "stdout": result.stdout[-8000:], "stderr": result.stderr[-4000:],
        }
    except BaseException as exc:
        return {"command": command, "error_type": type(exc).__name__, "error_message": str(exc)}


def gpu():
    """Attest the exact T4x2 runtime and retain classified failure evidence."""

    global GPU_DIAG
    torch = importlib.import_module("torch")
    count = int(torch.cuda.device_count())
    devices = []
    for index in range(count):
        name = _diagnostic_call(lambda i=index: str(torch.cuda.get_device_name(i)))
        capability = _diagnostic_call(lambda i=index: list(torch.cuda.get_device_capability(i)))
        properties = _diagnostic_call(lambda i=index: torch.cuda.get_device_properties(i))
        row = {"index": index, "name": name, "capability": capability}
        if isinstance(properties, dict):
            row["properties_error"] = properties
        else:
            row.update({
                "total_memory_bytes": int(properties.total_memory),
                "multi_processor_count": int(properties.multi_processor_count),
            })
        devices.append(row)
    names = [row["name"] if isinstance(row["name"], str) else "<name-error>" for row in devices]
    name_mismatch = any("T4" not in name.upper() for name in names)
    mismatches = []
    if count != 2:
        mismatches.append("device_count")
    if name_mismatch:
        mismatches.append("device_name")
    if count == 0:
        classification = "no_cuda_devices"
    elif name_mismatch:
        classification = "unexpected_accelerator_family"
    elif count != 2:
        classification = "honored_t4_wrong_device_count"
    else:
        classification = "exact_t4x2"
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    GPU_DIAG = {
        "requested": {
            "kaggle_machine_shape": KSHAPE,
            "logical_accelerator": "T4x2",
            "expected_device_count": 2,
            "expected_name_contains": "T4",
            "devices": ["cuda:0", "cuda:1"],
        },
        "observed": {
            "classification": classification,
            "mismatches": mismatches,
            "device_count": count,
            "devices": devices,
            "torch_version": str(getattr(torch, "__version__", None)),
            "torch_cuda_version": str(getattr(getattr(torch, "version", None), "cuda", None)),
            "torch_cuda_available": _diagnostic_call(lambda: bool(torch.cuda.is_available())),
            "torch_compiled_arch_list": _diagnostic_call(lambda: list(torch.cuda.get_arch_list())),
            "cudnn_version": _diagnostic_call(lambda: cudnn.version()) if cudnn is not None else None,
        },
        "environment": {key: os.environ.get(key) for key in GPU_ENV_KEYS},
        "nvidia_smi": {
            "inventory": _diagnostic_command(["nvidia-smi", "-L"]),
            "query": _diagnostic_command([
                "nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]),
        },
    }
    if mismatches:
        raise LaunchError(
            "exact T4x2 required; requested_shape=%s classification=%s mismatches=%r count=%s names=%r"
            % (KSHAPE, classification, mismatches, count, names)
        )
    return names


def _competition_root():
    candidates = (
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
        Path("/kaggle/input/biohub-cell-tracking-during-development"),
    )
    for path in candidates:
        if (path / "test").is_dir():
            return path
    raise LaunchError("competition test input is missing from the two known Kaggle mount paths")


def _runner_command(competition, support, package_manifest, view_batch_size, score_gate):
    if not 1 <= view_batch_size <= 8:
        raise LaunchError("--view-batch-size must be in [1, 8]")
    if view_batch_size == 1 and score_gate is not None:
        raise LaunchError("sequential view_batch_size=1 cannot use --score-gate")
    if view_batch_size > 1 and score_gate is None:
        raise LaunchError("view_batch_size 2..8 requires an explicit --score-gate")
    run = package_manifest.get("identity", {}).get("runtime", {}).get("run")
    if not isinstance(run, dict):
        raise LaunchError("package run specification is missing")
    if run.get("mode") not in ("cv", "submit") or run.get("split") not in ("train", "test"):
        raise LaunchError("package run specification mode/split is invalid")
    if (
        run.get("graph_variant") != RELEASED_GRAPH_VARIANT
        or run.get("topology_mode") != "legacy"
        or run.get("detection_tta") != "D4-8-view"
    ):
        raise LaunchError("package run specification is not released-graph legacy-topology D4-8")
    if run.get("division_enabled") is not False:
        raise LaunchError("package run specification attempted to enable divisions")
    calibration = run.get("p0_calibration")
    if calibration is not None:
        experiment_id = calibration.get("experiment_id") if isinstance(calibration, dict) else None
        expected = P0_CALIBRATIONS.get(experiment_id)
        if (
            run.get("profile") != "p0-public-calibration"
            or expected is None
            or set(calibration) != {
                "schema", "experiment_id", "purpose", "motion_direction",
                "motion_error_weight", "learned_bonus",
                "local_cv57_score", "baseline_submission_id",
                "baseline_motion_error_weight", "production_eligible", "submit_eligible",
            }
            or calibration.get("schema") != "celltrack-alpha.p0-public-calibration.v1"
            or calibration.get("purpose") != "public-metric-parity-calibration"
            or calibration.get("motion_direction") != expected[0]
            or calibration.get("motion_error_weight") != expected[1]
            or calibration.get("learned_bonus") != expected[2]
            or calibration.get("local_cv57_score") != expected[3]
            or calibration.get("baseline_submission_id") != 54910497
            or calibration.get("baseline_motion_error_weight") != 0.25
            or calibration.get("production_eligible") is not False
            or calibration.get("submit_eligible") is not True
        ):
            raise LaunchError("package P0 calibration contract is invalid")
    appearance = run.get("appearance_swap")
    if appearance is not None:
        expected_appearance = {
            "contract_path": "reports/g02/appearance_direct_swap_contract_v1.json",
            "contract_sha256": "cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7",
            "geometry_gate_path": "reports/g02/selected_graph_background_sparse_w050_v1/gate.pt",
            "geometry_gate_sha256": "04a4cec214b5848363af08ab9a092f5ac8cc077b83f946ea81fc986d36e91ed1",
            "local_score": 0.8815220699597315,
            "edge_tp": 34320,
            "edge_fp": 2459,
            "edge_fn": 1948,
            "production_eligible": False,
            "submit_eligible": True,
        }
        expected_profile = (
            "appearance-direct-swap-submit"
            if run.get("mode") == "submit"
            else "appearance-direct-swap-cv57"
        )
        if (
            calibration is not None
            or run.get("profile") != expected_profile
            or appearance != expected_appearance
            or view_batch_size != 1
            or score_gate is not None
        ):
            raise LaunchError("package appearance-swap contract is invalid")
    datasets = run.get("datasets")
    if datasets is not None and (not isinstance(datasets, list) or not datasets):
        raise LaunchError("package run specification datasets are invalid")
    command = [
        sys.executable, str(SOURCE_ROOT / "scripts" / "run_bh.py"),
        "--input", str(competition), "--registry", str(SOURCE_ROOT / "bhreg.json"),
        "--support-root", str(support), "--out-dir", str(WORK), "--output-file", str(run.get("output_file")),
        "--mode", str(run["mode"]), "--split", str(run["split"]),
        "--devices", "cuda:0,cuda:1", "--pair-batch-size", "1",
        "--view-batch-size", str(view_batch_size), "--deadline-min", "360",
        "--source-budget-bytes", "999999", "--graph-variant", str(run["graph_variant"]),
    ]
    if datasets is not None:
        command.extend(("--datasets", ",".join(datasets)))
    edge = run.get("edge_artifact")
    if edge is not None:
        if not isinstance(edge, dict):
            raise LaunchError("package edge artifact specification is invalid")
        command.extend((
            "--edge-artifact", str(SOURCE_ROOT / str(edge.get("path"))),
            "--edge-artifact-sha256", str(edge.get("sha256")),
        ))
    if calibration is not None:
        command.extend((
            "--p0-experiment-id", str(calibration["experiment_id"]),
            "--p0-motion-direction", str(calibration["motion_direction"]),
            "--p0-motion-error-weight", str(calibration["motion_error_weight"]),
            "--p0-learned-bonus", str(calibration["learned_bonus"]),
        ))
    if appearance is not None:
        command.extend((
            "--appearance-swap-contract",
            str(SOURCE_ROOT / str(appearance["contract_path"])),
            "--appearance-swap-contract-sha256",
            str(appearance["contract_sha256"]),
        ))
    if score_gate is not None:
        command.extend(("--accelerated-score-gate", str(score_gate)))
    return command


def _run_with_deadline(command, environment, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LaunchError("360-minute global deadline expired before V45 launch")
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise LaunchError("360-minute global deadline reached; V45 process group terminated") from exc
    if returncode != 0:
        raise LaunchError("scripts/run_bh.py failed with exit code %s" % returncode)


def _write_diagnostic(value):
    WORK.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_bytes(_canonical(value) + b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-batch-size", type=int, default=1)
    parser.add_argument("--score-gate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    diagnostics = {"schema": "bhv45-kaggle-diagnostic.v1", "network": "forbidden", "ok": False}
    try:
        package_manifest = _verify_package()
        diagnostics["dependencies_before"] = _probe_dependencies()
        support, wheel_dir = _validate_attachment(diagnostics)
        diagnostics["dependencies_final"] = _install_missing(diagnostics["dependencies_before"], wheel_dir, diagnostics)
        diagnostics["gpu_names"] = gpu()
        diagnostics["gpu_attestation"] = GPU_DIAG
        _extract_source(package_manifest)
        competition = _competition_root()
        environment = dict(os.environ)
        python_paths = [str(SOURCE_ROOT / "src"), str(support / "repo" / "src")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update({
            "PYTHONPATH": os.pathsep.join(python_paths), "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "CUDA_VISIBLE_DEVICES": "0,1",
        })
        command = _runner_command(competition, support, package_manifest, args.view_batch_size, args.score_gate)
        diagnostics["runner"] = {
            "command": command, "device_count": 2, "devices": ["cuda:0", "cuda:1"],
            "view_batch_size": args.view_batch_size, "score_gate": str(args.score_gate) if args.score_gate else None,
            "deadline_seconds": DEADLINE_SECONDS,
        }
        _write_diagnostic(diagnostics)
        _run_with_deadline(command, environment, deadline)
        input_contract = WORK / "input_contract.json"
        if input_contract.is_file():
            diagnostics["input_contract"] = _load(input_contract)
        diagnostics.update({"ok": True, "elapsed_seconds": time.monotonic() - started})
        _write_diagnostic(diagnostics)
        return 0
    except Exception as exc:
        diagnostics["gpu_attestation"] = GPU_DIAG
        input_contract = WORK / "input_contract.json"
        if input_contract.is_file():
            diagnostics["input_contract"] = _diagnostic_call(lambda: _load(input_contract))
        diagnostics["error"] = "%s: %s" % (type(exc).__name__, exc)
        diagnostics["elapsed_seconds"] = time.monotonic() - started
        _write_diagnostic(diagnostics)
        print(diagnostics["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def monolithic_entrypoint(payloads: Mapping[str, bytes]) -> bytes:
    """Embed every attested runtime file in the sole Kaggle ``code_file``.

    ``kaggle kernels push`` uploads the metadata-selected code file but does not
    preserve arbitrary sibling files.  The small wrapper materializes the
    already-attested logical package under ``/kaggle/working`` and executes its
    launcher from there.  This mirrors the proven fit-cache packaging model.
    """

    expected = {ENTRYPOINT_NAME, ARCHIVE_NAME, ATTACHMENT_NAME, MANIFEST_NAME, METADATA_NAME}
    if set(payloads) != expected:
        raise PackageError(
            f"monolithic payload coverage mismatch: expected={sorted(expected)} got={sorted(payloads)}"
        )
    encoded = {
        name: base64.b64encode(payloads[name]).decode("ascii")
        for name in sorted(payloads)
    }
    payload_json = json.dumps(encoded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    source = f'''#!/usr/bin/env python3
"""Materialize and execute the attested private BH CV package."""

import base64
from pathlib import Path

_ROOT = Path("/kaggle/working/.bh45pkg")
_PAYLOADS = {payload_json}

if _ROOT.exists():
    raise RuntimeError("refusing to overwrite existing embedded package root: %s" % _ROOT)
_ROOT.mkdir(parents=True, exist_ok=False)
for _name, _encoded in _PAYLOADS.items():
    if _name != Path(_name).name:
        raise RuntimeError("unsafe embedded package basename: %r" % _name)
    (_ROOT / _name).write_bytes(base64.b64decode(_encoded, validate=True))

_ENTRY = _ROOT / {ENTRYPOINT_NAME!r}
_GLOBALS = {{
    "__name__": "__main__",
    "__file__": str(_ENTRY),
    "__package__": None,
    "__cached__": None,
}}
exec(compile(_ENTRY.read_bytes(), str(_ENTRY), "exec"), _GLOBALS)
'''
    return source.encode("ascii")


def _attachment_payload(
    support_root: Path,
    expected_hashes: Mapping[str, str],
    *,
    support_mounts: Sequence[str],
) -> bytes:
    required = [file_record(support_root / relative, relative_to=support_root) for relative in SUPPORT_ARTIFACT_PATHS]
    expected_by_path = {
        "ARTIFACT_MANIFEST.json": expected_hashes["manifest_sha256"],
        "weights/unet_transformer/split_0/config.json": expected_hashes["config_sha256"],
        "weights/unet_transformer/split_0/edge_predictor_best.pth": expected_hashes["weight_sha256"],
    }
    mismatches = [
        f"{item['path']}: expected {expected_by_path[item['path']]}, got {item['sha256']}"
        for item in required
        if item["path"] in expected_by_path and item["sha256"] != expected_by_path[item["path"]]
    ]
    actual_source_sha = source_tree_sha256(support_root)
    if actual_source_sha != expected_hashes["source_sha256"]:
        mismatches.append(
            f"support source tree: expected {expected_hashes['source_sha256']}, got {actual_source_sha}"
        )
    if mismatches:
        raise PackageError("support/checkpoint SHA-256 validation failed: " + "; ".join(mismatches))

    wheel_dir = support_root / "wheels"
    wheels = [file_record(path, relative_to=wheel_dir) for path in sorted(wheel_dir.glob("*.whl"))]
    if not wheels:
        raise PackageError(f"public support wheel input is empty: {wheel_dir}")
    wheel_bytes = sum(item["bytes"] for item in wheels)
    identity = {
        "attachment_only": True,
        "external_dataset": True,
        "publish": False,
        "dataset_ref": SUPPORT_REF,
        "dataset_slug": SUPPORT_DATASET_SLUG,
        "mount_candidates": list(support_mounts),
        "local_reuse_path": support_root.as_posix(),
        "checkpoint": {
            "path": "weights/unet_transformer/split_0/edge_predictor_best.pth",
            "bytes": next(item["bytes"] for item in required if item["path"].endswith("edge_predictor_best.pth")),
            "sha256": expected_hashes["weight_sha256"],
        },
        "required_files": required,
        "source_tree": {"paths": list(SUPPORT_SOURCE_PATHS), "sha256": actual_source_sha},
        "wheelhouse": {
            "path": "wheels",
            "included_in_kernel_source": False,
            "files": wheels,
            "file_count": len(wheels),
            "bytes": wheel_bytes,
            "manifest_sha256": sha256_bytes(canonical_json(wheels)),
        },
    }
    payload = {
        "schema": "bhv45-kaggle-attachment.v1",
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical_json(identity)),
    }
    return canonical_json(payload) + b"\n"


def _archive_members(
    repo_root: Path,
    registry: Path,
    *,
    expected_test_count: int,
    run_spec: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    run = validate_run_spec(run_spec or submit_run_spec())
    source_paths = SOURCE_PATHS + (EDGE_SOURCE_PATHS if run["edge_artifact"] is not None else ())
    if run.get("appearance_swap") is not None:
        source_paths += (
            "src/celltrack_alpha/relative_geometry_gate.py",
            "src/celltrack_alpha/appearance_swap.py",
        )
    missing = [path for path in source_paths if not (repo_root / path).is_file()]
    if missing:
        raise PackageError(f"runner source files are missing: {missing}")
    members = {path: (repo_root / path).read_bytes() for path in source_paths}
    members["src/celltrack_alpha/__init__.py"] = b'"""Private V45 sparse runner modules."""\n'
    members["bhreg.json"] = compact_registry(
        registry,
        expected_test_count=expected_test_count,
        split=run["split"],
        selected_video_ids=run["datasets"],
    )
    edge = run["edge_artifact"]
    if edge is not None:
        for path_key, sha_key in (
            ("path", "sha256"),
            ("auxiliary_checkpoint_path", "auxiliary_checkpoint_sha256"),
        ):
            relative = edge[path_key]
            source = repo_root / relative
            if not source.is_file() or sha256_file(source) != edge[sha_key]:
                raise PackageError(f"pinned CV edge input identity mismatch: {relative}")
            members[relative] = source.read_bytes()
    appearance = run.get("appearance_swap")
    if appearance is not None:
        for path_key, sha_key in (
            ("contract_path", "contract_sha256"),
            ("geometry_gate_path", "geometry_gate_sha256"),
        ):
            relative = appearance[path_key]
            source = repo_root / relative
            if not source.is_file() or sha256_file(source) != appearance[sha_key]:
                raise PackageError(
                    f"pinned appearance input identity mismatch: {relative}"
                )
            members[relative] = source.read_bytes()
    return members


def _archive_identity(members: Mapping[str, bytes], archive: bytes) -> dict[str, Any]:
    return {
        "path": ARCHIVE_NAME,
        "bytes": len(archive),
        "sha256": sha256_bytes(archive),
        "uncompressed_bytes": sum(len(value) for value in members.values()),
        "members": [
            {"path": name, "bytes": len(members[name]), "sha256": sha256_bytes(members[name])}
            for name in sorted(members)
        ],
    }


def _metadata_payload(kernel_slug: str = KERNEL_SLUG) -> bytes:
    value = {
        "id": f"onebitbrain/{kernel_slug}",
        "title": kernel_slug,
        "code_file": ENTRYPOINT_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": KAGGLE_ACCELERATOR_ID,
        "competition_sources": [COMPETITION_SLUG],
        "dataset_sources": [SUPPORT_REF],
    }
    return json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _manifest_payload(
    files: Mapping[str, bytes],
    archive_identity: Mapping[str, Any],
    *,
    kernel_slug: str = KERNEL_SLUG,
    run_spec: Mapping[str, Any] | None = None,
) -> bytes:
    run = validate_run_spec(run_spec or submit_run_spec())
    file_items = [
        {"path": name, "bytes": len(value), "sha256": sha256_bytes(value)}
        for name, value in sorted(files.items())
    ]
    package_bytes = 0
    while True:
        identity = {
            "private": True,
            "network": "forbidden",
            "publish_actions": False,
            "budget_bytes": SOURCE_LIMIT,
            "package_bytes": package_bytes,
            "package_file_count": len(files) + 1,
            "strictly_under_budget": True,
            "kernel_slug": kernel_slug,
            "external_support": {
                "dataset_ref": SUPPORT_REF,
                "dataset_slug": SUPPORT_DATASET_SLUG,
                "created_by_kernel_owner": False,
                "long_slug_exception": "pre-existing public Kaggle input",
            },
            "fixed_long_basename_exception": METADATA_NAME,
            "files": file_items,
            "archive": dict(archive_identity),
            "runtime": {
                "runner": "scripts/run_bh.py",
                "deadline_minutes": SUBMISSION_DEADLINE_MINUTES,
                "devices": ["cuda:0", "cuda:1"],
                "gpu_count": 2,
                "machine_shape": KAGGLE_ACCELERATOR_ID,
                "gpu_attestation_classifications": [
                    "no_cuda_devices",
                    "unexpected_accelerator_family",
                    "honored_t4_wrong_device_count",
                    "exact_t4x2",
                ],
                "pair_batch_size": 1,
                "default_view_batch_size": 1,
                "accelerated_view_batch_sizes": [2, 3, 4, 5, 6, 7, 8],
                "accelerated_score_gate_required": True,
                "dependency_install": "python -m pip install --no-index --no-deps --find-links",
                "install_only_missing_or_incompatible": True,
                "compatible_core_reinstall": False,
                "run": run,
            },
        }
        payload = {
            "schema": "bhv45-kaggle-package.v1",
            "identity": identity,
            "identity_sha256": sha256_bytes(canonical_json(identity)),
        }
        encoded = canonical_json(payload) + b"\n"
        observed = sum(len(value) for value in files.values()) + len(encoded)
        if observed == package_bytes:
            return encoded
        package_bytes = observed


def build_package(
    repo_root: Path = ROOT,
    output_dir: Path | None = None,
    *,
    registry: Path | None = None,
    support_root: Path | None = None,
    expected_artifact_hashes: Mapping[str, str] = EXPECTED_ARTIFACT_HASHES,
    expected_test_count: int = 4,
    support_mounts: Sequence[str] = SUPPORT_MOUNT_CANDIDATES,
    profile: str = "submit",
    split_manifest: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    registry = (registry or (repo_root / "reports/g02/dataset_registry_v1.json")).resolve()
    support_root = (support_root or (repo_root / "external/Tracking_Support")).resolve()
    split_manifest = (split_manifest or (repo_root / "reports/g02/grouped_3fold_audit20_v1.json")).resolve()
    if profile == "submit":
        kernel_slug = KERNEL_SLUG
        run_spec = submit_run_spec()
        default_output = repo_root / "kaggle_kernel/bh45pkg"
    elif profile == "cv57":
        kernel_slug = CV57_KERNEL_SLUG
        run_spec = cv57_run_spec(split_manifest)
        default_output = repo_root / "kaggle_kernel/bhcv57pkg"
    elif profile in P0_CALIBRATIONS:
        kernel_slug = P0_CALIBRATIONS[profile]["kernel_slug"]
        run_spec = p0_calibration_run_spec(profile)
        default_output = repo_root / "kaggle_kernel" / kernel_slug
    elif profile == "appearance-submit":
        kernel_slug = APPEARANCE_SUBMIT_KERNEL_SLUG
        run_spec = appearance_run_spec("submit", split_manifest)
        default_output = repo_root / "kaggle_kernel" / kernel_slug
    elif profile == "appearance-cv57":
        kernel_slug = APPEARANCE_CV57_KERNEL_SLUG
        run_spec = appearance_run_spec("cv57", split_manifest)
        default_output = repo_root / "kaggle_kernel" / kernel_slug
    else:
        raise PackageError(f"unsupported package profile: {profile!r}")
    output_dir = (output_dir or default_output).resolve()

    validate_slug(kernel_slug, "kernel slug")
    for name in (ENTRYPOINT_NAME, ARCHIVE_NAME, ATTACHMENT_NAME, MANIFEST_NAME, output_dir.name):
        if len(name) > 15:
            raise PackageError(f"created basename exceeds 15 characters: {name}")
    if (
        not support_mounts
        or len(set(support_mounts)) != len(support_mounts)
        or any(not Path(item).is_absolute() or Path(item).name != SUPPORT_DATASET_SLUG for item in support_mounts)
    ):
        raise PackageError("support mount candidates must be unique absolute paths ending in the public dataset slug")
    if set(expected_artifact_hashes) != set(EXPECTED_ARTIFACT_HASHES):
        raise PackageError("expected artifact hash mapping has missing or unexpected fields")

    members = _archive_members(
        repo_root,
        registry,
        expected_test_count=expected_test_count,
        run_spec=run_spec,
    )
    archive = deterministic_zip(members)
    archive_identity = _archive_identity(members, archive)
    entrypoint_template = ENTRYPOINT_SOURCE.encode("ascii")
    attachment = _attachment_payload(support_root, expected_artifact_hashes, support_mounts=support_mounts)
    metadata = _metadata_payload(kernel_slug)
    logical_files = {
        ENTRYPOINT_NAME: entrypoint_template,
        ARCHIVE_NAME: archive,
        ATTACHMENT_NAME: attachment,
        METADATA_NAME: metadata,
    }
    manifest = _manifest_payload(logical_files, archive_identity, kernel_slug=kernel_slug, run_spec=run_spec)
    logical_files[MANIFEST_NAME] = manifest
    files = dict(logical_files)
    files[ENTRYPOINT_NAME] = monolithic_entrypoint(logical_files)
    package_bytes = sum(len(value) for value in files.values())
    if package_bytes >= SOURCE_LIMIT:
        raise PackageError(f"aggregate submitted kernel source must be <{SOURCE_LIMIT} bytes, got {package_bytes}")
    if any(name.endswith(".whl") for name in files) or any(name.endswith(".whl") for name in members):
        raise PackageError("offline wheels must remain an attached private input, not kernel source")

    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in files)
    if unexpected:
        raise PackageError(f"refusing to mix private package with unexpected files: {unexpected}")
    for name, value in files.items():
        temporary = output_dir / f".{name}.tmp"
        temporary.write_bytes(value)
        temporary.replace(output_dir / name)

    records = [file_record(output_dir / name, relative_to=output_dir) for name in sorted(files)]
    report = {
        "schema": "bhv45-package-build.v1",
        "profile": profile,
        "kernel_ref": f"onebitbrain/{kernel_slug}",
        "output_dir": str(output_dir),
        "aggregate_bytes": package_bytes,
        "submitted_code_bytes": len(files[ENTRYPOINT_NAME]),
        "logical_package_bytes": sum(len(value) for value in logical_files.values()),
        "budget_bytes": SOURCE_LIMIT,
        "strictly_under_budget": package_bytes < SOURCE_LIMIT,
        "file_count": len(records),
        "files": records,
        "archive": archive_identity,
        "attachment_manifest_sha256": sha256_bytes(attachment),
        "package_manifest_sha256": sha256_bytes(manifest),
        "network_or_kaggle_actions": False,
        "run": run_spec,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--profile",
        choices=(
            "submit",
            "cv57",
            *P0_CALIBRATIONS,
            "appearance-submit",
            "appearance-cv57",
        ),
        default="submit",
    )
    args = parser.parse_args(argv)
    report = build_package(
        ROOT,
        args.output,
        registry=args.registry,
        support_root=args.support_root,
        split_manifest=args.split_manifest,
        profile=args.profile,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
