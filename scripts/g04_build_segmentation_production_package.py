#!/usr/bin/env python3
"""Build, but never launch, a timing-bound G-04 segmentation production shard.

The root coordinator supplies an arbitrary shard manifest.  This builder never
chooses or freezes a shard count.  It derives the per-frame hard bound only from
an authenticated terminal timing result and its supervisor receipt, verifies
the assigned organizer raw bytes against the local competition archive, and
renders a private/no-network Kaggle package that mounts only the competition
source and the already-published Cellpose runtime dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "kaggle_kernel/g04-computed-segmentation-production-template/"
    "g04_segmentation_production_template.py"
)
RUNTIME_CARRIER = (
    ROOT / "kaggle_kernel/g04-computed-segmentation-runtime-cp312-v1"
)
RUNTIME_DATASET_ID = "onebitbrain/g04-computed-segmentation-runtime-cp312-v1"
RUNTIME_MOUNT_ROOT = (
    "/kaggle/input/datasets/onebitbrain/"
    "g04-computed-segmentation-runtime-cp312-v1"
)
RUNTIME_MANIFEST_SHA256 = (
    "0ccf485df9f9962b897037b9ea2a218992b220d587193954c115cab93f7e01e0"
)
RUNTIME_DATASET_METADATA_SHA256 = (
    "b900364d4b01522a4282bc68ff59f50c1c0f4c877620558c005b7d51fe91d0d6"
)
MODEL_SHA256 = (
    "2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c"
)
COMPETITION_SOURCE = "biohub-cell-tracking-during-development"
SHARD_SCHEMA = "celltrack-alpha.g04-segmentation-production-shard.v1"
TIMING_RESULT_SCHEMA = "celltrack-alpha.g04-computed-segmentation-cuda-timing.v2"
TIMING_SUPERVISOR_SCHEMA = (
    "celltrack-alpha.g04-computed-segmentation-timing-supervisor.v2"
)
FEATURE_CONTRACT_SHA256 = (
    "910177ae086013bfdff04b1e9528526ff4961f170fbdba49af6f578a8a3436b7"
)
PRODUCTION_CONTRACT_SCHEMA = (
    "celltrack-alpha.g04-segmentation-production-runtime-contract.v2"
)
SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SAFE_DATASET = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class BuildError(RuntimeError):
    """A timing, shard, raw-byte, or package invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"{label} must be a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_runtime_carrier() -> dict[str, Any]:
    manifest_path = RUNTIME_CARRIER / "RUNTIME_CARRIER_MANIFEST.json"
    metadata_path = RUNTIME_CARRIER / "dataset-metadata.json"
    if sha256(manifest_path) != RUNTIME_MANIFEST_SHA256:
        raise BuildError("runtime manifest bytes drifted")
    if sha256(metadata_path) != RUNTIME_DATASET_METADATA_SHA256:
        raise BuildError("runtime dataset metadata bytes drifted")
    manifest = load_json(manifest_path, "runtime manifest")
    rows = manifest.get("files")
    if (
        manifest.get("status") != "READY_AUTHENTICATED_OFFLINE_CP312"
        or manifest.get("file_count") != 29
        or not isinstance(rows, list)
        or len(rows) != 29
        or manifest.get("install_contract", {}).get("wheel_count") != 24
        or manifest.get("base_runtime", {}).get("abi") != "cp312"
    ):
        raise BuildError("runtime carrier contract drifted")
    return manifest


def validate_terminal_timing(
    result_path: Path, supervisor_path: Path
) -> dict[str, Any]:
    result = load_json(result_path, "terminal timing result")
    supervisor = load_json(supervisor_path, "terminal timing supervisor")
    result_sha = sha256(result_path)
    if (
        result.get("schema") != TIMING_RESULT_SCHEMA
        or result.get("status") != "PASS_CUDA_TIMING_GATE"
        or result.get("timing_only") is not True
        or result.get("production_manifest_eligible") is not False
        or result.get("gt_access") is not False
        or result.get("gt_node_access") is not False
        or result.get("gt_edge_access") is not False
        or result.get("detection_access") is not False
        or result.get("held_fit_metric_access") is not False
        or result.get("cv57_metric_access") is not False
        or result.get("submission") is not False
    ):
        raise BuildError("timing result is not a terminal raw-only PASS")
    if (
        supervisor.get("schema") != TIMING_SUPERVISOR_SCHEMA
        or supervisor.get("status") != "PASS_ATOMIC_TIMING_RECEIPT"
        or supervisor.get("output_sha256") != result_sha
        or supervisor.get("production_manifest_eligible") is not False
        or supervisor.get("submission") is not False
        or supervisor.get("feature_contract_sha256")
        != FEATURE_CONTRACT_SHA256
    ):
        raise BuildError("timing supervisor is absent, divergent, or ineligible")
    repetitions = result.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != 2:
        raise BuildError("terminal timing must contain exactly two repeats")
    elapsed: list[float] = []
    label_hashes: set[str] = set()
    region_hashes: set[str] = set()
    for index, row in enumerate(repetitions):
        if row.get("repeat") != index:
            raise BuildError("timing repeat order drifted")
        seconds = float(row.get("elapsed_seconds", 0.0))
        if not math.isfinite(seconds) or seconds <= 0:
            raise BuildError("timing repeat duration is invalid")
        elapsed.append(seconds)
        label_hashes.add(str(row.get("labels_sha256", "")))
        region_hashes.add(str(row.get("regionprops_payload_sha256", "")))
        if int(row.get("mask_count", 0)) <= 0 or row.get("mask_count") != row.get(
            "region_count"
        ):
            raise BuildError("timing repeat has no complete computed objects")
        if int(row.get("legacy_skimage_solidity_nonfinite_count", 0)) <= 0:
            raise BuildError("timing repeat did not reproduce the v1 solidity failure")
        histogram = row.get("center_affine_dimension_histogram")
        subdimensional = row.get("subdimensional_objects")
        if (
            not isinstance(histogram, Mapping)
            or set(histogram) != {"0", "1", "2", "3"}
            or sum(int(value) for value in histogram.values())
            != int(row["region_count"])
            or not isinstance(subdimensional, list)
        ):
            raise BuildError("timing dimensionality diagnostics are malformed")
    if (
        len(label_hashes) != 1
        or len(region_hashes) != 1
        or any(len(value) != 64 for value in label_hashes | region_hashes)
    ):
        raise BuildError("timing repeat payloads are not byte-stable")
    derived = int(math.ceil(max(elapsed) * 1.25))
    computed_contract = result.get("computed_object_contract", {})
    projections = result.get("runtime_projection", {})
    cv57 = projections.get("cv57", {})
    held_fit105 = projections.get("held_fit105", {})
    quota = projections.get("quota_feasibility", {})
    steady = elapsed[-1]
    worst = max(elapsed)
    if (
        computed_contract.get("feature_contract_sha256")
        != FEATURE_CONTRACT_SHA256
        or computed_contract.get("undefined_solidity_sentinel") is not None
        or computed_contract.get("legacy_skimage_center_hull_used_as_feature")
        is not False
        or projections.get("steady_definition")
        != "second_of_two_post_initialization_repeats"
        or projections.get("worst_definition")
        != "max_of_two_post_initialization_repeats"
        or cv57.get("frame_count") != 5_700
        or held_fit105.get("frame_count") != 10_500
        or float(cv57.get("steady_seconds_per_frame", 0.0)) != steady
        or float(cv57.get("worst_seconds_per_frame", 0.0)) != worst
        or float(held_fit105.get("steady_seconds_per_frame", 0.0)) != steady
        or float(held_fit105.get("worst_seconds_per_frame", 0.0)) != worst
        or quota.get(
            "fresh_authoritative_reconciliation_required_before_production"
        )
        is not True
    ):
        raise BuildError("v2 feature/projection/quota contract drifted")
    admission = result.get("production_admission", {})
    if (
        admission.get("rule") != "ceil(max(repeat_seconds) * 1.25)"
        or admission.get("seconds_per_frame_upper_bound") != derived
        or supervisor.get("seconds_per_frame_upper_bound") != derived
    ):
        raise BuildError("timing-derived hard frame bound drifted")
    return {
        "result_sha256": result_sha,
        "supervisor_sha256": sha256(supervisor_path),
        "hard_frame_bound_seconds": derived,
        "repeat_elapsed_seconds": elapsed,
        "labels_sha256": next(iter(label_hashes)),
        "regionprops_payload_sha256": next(iter(region_hashes)),
        "timing_identity": result.get("input", {}).get("identity"),
        "timing_timepoint": result.get("input", {}).get("timepoint"),
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "runtime_projection": projections,
    }


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden in value or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def validate_shard_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SHARD_SCHEMA:
        raise BuildError("shard manifest schema mismatch")
    if value.get("status") != "ROOT_SUPPLIED_UNFROZEN_SHARD":
        raise BuildError("shard manifest status mismatch")
    if _contains_key(value, "shard_count"):
        raise BuildError("shard count must remain unfrozen and absent")
    shard_id = str(value.get("shard_id", ""))
    split = str(value.get("split", ""))
    if not SAFE_SLUG.fullmatch(shard_id) or split not in {"train", "test"}:
        raise BuildError("unsafe shard id or split")
    videos = value.get("videos")
    if not isinstance(videos, list) or not videos:
        raise BuildError("shard manifest has no videos")
    identities: set[str] = set()
    assignments: set[tuple[str, int]] = set()
    raw_paths: set[str] = set()
    normalized_videos: list[dict[str, Any]] = []
    for video in videos:
        if not isinstance(video, Mapping):
            raise BuildError("shard video row must be an object")
        identity = str(video.get("identity", ""))
        if ":" not in identity:
            raise BuildError(f"invalid identity: {identity!r}")
        identity_split, dataset = identity.split(":", 1)
        if identity_split != split or not SAFE_DATASET.fullmatch(dataset):
            raise BuildError(f"identity/split drift: {identity}")
        if identity in identities:
            raise BuildError(f"duplicate identity: {identity}")
        identities.add(identity)
        frames = video.get("frame_indices")
        if (
            not isinstance(frames, list)
            or not frames
            or any(type(item) is not int or item < 0 for item in frames)
            or frames != sorted(set(frames))
        ):
            raise BuildError(f"invalid frame assignment: {identity}")
        scale = video.get("scale_zyx")
        if (
            not isinstance(scale, list)
            or len(scale) != 3
            or any(
                not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) <= 0
                for item in scale
            )
        ):
            raise BuildError(f"invalid spatial scale: {identity}")
        expected_paths = {
            f"{dataset}.zarr/zarr.json",
            f"{dataset}.zarr/0/zarr.json",
            *(f"{dataset}.zarr/0/c/{frame}/0/0/0" for frame in frames),
        }
        rows = video.get("raw_files")
        if not isinstance(rows, list) or len(rows) != len(expected_paths):
            raise BuildError(f"raw file inventory size drift: {identity}")
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise BuildError(f"raw file row is malformed: {identity}")
            relative = str(row.get("path", ""))
            path = Path(relative)
            digest = str(row.get("sha256", ""))
            if (
                not relative
                or path.is_absolute()
                or ".." in path.parts
                or relative in indexed
                or relative in raw_paths
                or type(row.get("bytes")) is not int
                or int(row["bytes"]) <= 0
                or len(digest) != 64
            ):
                raise BuildError(f"unsafe raw file row: {identity}::{relative}")
            indexed[relative] = {
                "path": relative,
                "bytes": int(row["bytes"]),
                "sha256": digest,
            }
            raw_paths.add(relative)
        if set(indexed) != expected_paths:
            raise BuildError(f"raw file set is not exact: {identity}")
        for frame in frames:
            pair = (identity, int(frame))
            if pair in assignments:
                raise BuildError(f"duplicate frame assignment: {pair}")
            assignments.add(pair)
        normalized_videos.append(
            {
                "identity": identity,
                "frame_indices": list(frames),
                "scale_zyx": [float(item) for item in scale],
                "raw_files": [indexed[path] for path in sorted(indexed)],
            }
        )
    return {
        "schema": SHARD_SCHEMA,
        "status": "ROOT_SUPPLIED_UNFROZEN_SHARD",
        "shard_id": shard_id,
        "split": split,
        "videos": sorted(normalized_videos, key=lambda row: row["identity"]),
    }


def validate_competition_archive(
    archive_path: Path, shard: Mapping[str, Any]
) -> str:
    if not archive_path.is_file():
        raise BuildError(f"competition archive is absent: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for video in shard["videos"]:
                for row in video["raw_files"]:
                    member = f"{shard['split']}/{row['path']}"
                    try:
                        info = archive.getinfo(member)
                    except KeyError as exc:
                        raise BuildError(f"competition member is absent: {member}") from exc
                    if info.file_size != row["bytes"]:
                        raise BuildError(f"competition member byte size drift: {member}")
                    digest = hashlib.sha256()
                    with archive.open(info) as handle:
                        for block in iter(lambda: handle.read(1 << 20), b""):
                            digest.update(block)
                    if digest.hexdigest() != row["sha256"]:
                        raise BuildError(f"competition member SHA drift: {member}")
    except zipfile.BadZipFile as exc:
        raise BuildError("competition archive is invalid") from exc
    return sha256(archive_path)


def render_template(
    template: str, shard: Mapping[str, Any], contract: Mapping[str, Any]
) -> str:
    replacements = {
        "__RUNTIME_MANIFEST_SHA256__": RUNTIME_MANIFEST_SHA256,
        "__SHARD_MANIFEST_SHA256__": canonical_sha256(shard),
        "__PRODUCTION_CONTRACT_SHA256__": canonical_sha256(contract),
        "__SHARD_MANIFEST_LITERAL__": repr(dict(shard)),
        "__PRODUCTION_CONTRACT_LITERAL__": repr(dict(contract)),
    }
    rendered = template
    for token, replacement in replacements.items():
        if rendered.count(token) != 1:
            raise BuildError(f"template token count drift: {token}")
        rendered = rendered.replace(token, replacement)
    if "__" + "SHARD_MANIFEST" in rendered or "__PRODUCTION_CONTRACT" in rendered:
        raise BuildError("unresolved production template token")
    return rendered


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    validate_runtime_carrier()
    timing = validate_terminal_timing(args.timing_result, args.timing_supervisor)
    shard = validate_shard_manifest(load_json(args.shard_manifest, "shard manifest"))
    competition_archive_sha = validate_competition_archive(
        args.competition_archive, shard
    )
    if not SAFE_SLUG.fullmatch(args.kernel_id.split("/", 1)[-1]):
        raise BuildError("kernel id slug is unsafe")
    if "/" not in args.kernel_id or len(args.kernel_id.split("/", 1)[1]) > 50:
        raise BuildError("kernel id must be owner/slug with a <=50-character slug")
    if args.output.exists() and any(args.output.iterdir()):
        raise BuildError("output directory must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)

    frame_count = sum(len(row["frame_indices"]) for row in shard["videos"])
    raw_file_count = sum(len(row["raw_files"]) for row in shard["videos"])
    contract = {
        "schema": PRODUCTION_CONTRACT_SCHEMA,
        "status": "TIMING_BOUND_TEMPLATE_NOT_LAUNCH_AUTHORIZED",
        "timing": timing,
        "hard_frame_bound_seconds": timing["hard_frame_bound_seconds"],
        "frame_finalize_margin_seconds": int(args.frame_finalize_margin_seconds),
        "process_hard_kill_seconds": int(args.process_hard_kill_seconds),
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "runtime_remote_file_count": 30,
        "wheel_count": 24,
        "model_sha256": MODEL_SHA256,
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "competition_source": COMPETITION_SOURCE,
        "shard_manifest_sha256": canonical_sha256(shard),
        "assignment_count": frame_count,
        "raw_file_count": raw_file_count,
        "negative_scope": {
            "geff_access": False,
            "learned_detection_access": False,
            "gt_access": False,
            "metric_access": False,
            "submission": False,
        },
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    code_name = "g04_segmentation_production.py"
    code_path = args.output / code_name
    code_path.write_text(render_template(template, shard, contract), encoding="utf-8")
    atomic_json(args.output / "SHARD_MANIFEST.json", shard)
    atomic_json(args.output / "PRODUCTION_RUNTIME_CONTRACT.json", contract)
    metadata = {
        "id": args.kernel_id,
        "title": args.kernel_id.split("/", 1)[1],
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [COMPETITION_SOURCE],
        "dataset_sources": [RUNTIME_DATASET_ID],
        "model_sources": [],
    }
    atomic_json(args.output / "kernel-metadata.json", metadata)
    code_sha = sha256(code_path)
    metadata_sha = sha256(args.output / "kernel-metadata.json")
    prelaunch = {
        "schema": "celltrack-alpha.g04-segmentation-production-prelaunch.v2",
        "status": "DRAFT_REQUIRES_ROOT_GUARD_AND_SERIAL_LAUNCH_CONTROLS",
        "kernel_id": args.kernel_id,
        "remote_preflight_kernel_id": args.kernel_id,
        "remote_preflight_mode": "inline_step0_before_heavy_work",
        "lane_owner": "G04_COMPUTED_SEGMENTATION_PRODUCTION",
        "dataset_sources": [RUNTIME_DATASET_ID],
        "exclusive_dataset_source": RUNTIME_DATASET_ID,
        "dataset_version": 1,
        "expected_mount_root": RUNTIME_MOUNT_ROOT,
        "input_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "checkpoint_sha256": MODEL_SHA256,
        "population_manifest_sha256": canonical_sha256(shard),
        "request_count": frame_count,
        "request_file_count": raw_file_count,
        "population_file_count": raw_file_count,
        "wheel_count": 24,
        "step0_limit_seconds": 600.0,
        "step0_elapsed_is_hard_failure": False,
        "step0_competition_path_visits": 1,
        "code_sha256": code_sha,
        "kernel_metadata_sha256": metadata_sha,
        "kernel_properties": {
            "enable_gpu": True,
            "enable_internet": False,
            "enable_tpu": False,
            "machine_shape": "NvidiaTeslaT4",
        },
        "launch_authorized": False,
        "hard_frame_bound_seconds": timing["hard_frame_bound_seconds"],
        "timing_result_sha256": timing["result_sha256"],
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "runtime_projection": timing["runtime_projection"],
        "shard_count_frozen": False,
        "negative_scope": contract["negative_scope"],
    }
    atomic_json(args.output / "prelaunch-contract.json", prelaunch)
    receipt = {
        "schema": "celltrack-alpha.g04-segmentation-production-package-build.v2",
        "status": "PASS_BUILT_NOT_LAUNCH_AUTHORIZED",
        "kernel_id": args.kernel_id,
        "shard_id": shard["shard_id"],
        "shard_count_frozen": False,
        "frame_assignment_count": frame_count,
        "raw_file_count": raw_file_count,
        "competition_archive_sha256": competition_archive_sha,
        "timing_result_sha256": timing["result_sha256"],
        "timing_supervisor_sha256": timing["supervisor_sha256"],
        "hard_frame_bound_seconds": timing["hard_frame_bound_seconds"],
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
        "external_actions": {
            "dataset_upload": False,
            "kernel_launch": False,
            "submission": False,
        },
    }
    atomic_json(args.output / "BUILD_RECEIPT.json", receipt)
    launch_files = (
        code_name,
        "kernel-metadata.json",
        "SHARD_MANIFEST.json",
        "PRODUCTION_RUNTIME_CONTRACT.json",
        "prelaunch-contract.json",
        "BUILD_RECEIPT.json",
    )
    package_manifest = {
        "schema": "celltrack-alpha.g04-segmentation-production-package.v2",
        "status": "BUILT_NOT_LAUNCH_AUTHORIZED",
        "kernel_id": args.kernel_id,
        "files": [
            {
                "path": name,
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256(args.output / name),
            }
            for name in launch_files
        ],
        "file_count_excluding_manifest": len(launch_files),
    }
    atomic_json(args.output / "PACKAGE_MANIFEST.json", package_manifest)
    return {
        **receipt,
        "package_manifest_sha256": sha256(args.output / "PACKAGE_MANIFEST.json"),
        "code_sha256": code_sha,
        "kernel_metadata_sha256": metadata_sha,
        "prelaunch_contract_sha256": sha256(
            args.output / "prelaunch-contract.json"
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--timing-result", required=True, type=Path)
    value.add_argument("--timing-supervisor", required=True, type=Path)
    value.add_argument("--shard-manifest", required=True, type=Path)
    value.add_argument("--competition-archive", required=True, type=Path)
    value.add_argument("--kernel-id", required=True)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--process-hard-kill-seconds", type=int, default=39_600)
    value.add_argument("--frame-finalize-margin-seconds", type=int, default=120)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.process_hard_kill_seconds <= 0
        or args.frame_finalize_margin_seconds < 0
    ):
        raise BuildError("runtime limits must be nonnegative")
    result = build_package(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
