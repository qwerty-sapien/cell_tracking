#!/usr/bin/env python3
"""Human-launched frozen H1 cache build; never training or submission."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile


WORKING = Path("/kaggle/working")
CONTRACT_SHA256 = (
    "a7e866c34c15f46671ae3c30aeb4cf1107e5dd55acd63458a22890b69ff29404"
)
SEED = 20260725
ENVIRONMENT_PINS = {
    "python": "3.12.13",
    "packages": {
        "PyYAML": "6.0.3",
        "donfig": "0.8.1.post1",
        "numcodecs": "0.15.1",
        "numpy": "2.0.2",
        "scipy": "1.16.3",
        "torch": "2.10.0+cu128",
        "zarr": "3.2.1",
    },
    "cuda_runtime": "12.8.1",
    "torch_cuda": "12.8",
    "cudnn": 91002,
    "device": "Tesla T4",
}
RAW_MANIFEST_SHA256 = (
    "665c505b4e38a03f5a501527fde58a064d3ce8fac413898656a4f3f3e6cf781e"
)
SOURCE_DEPENDENCY_MANIFEST_SHA256 = (
    "999db8a2a475415abc9ff5dde1d18d366c9caf40a311ddfb09af4c008df51326"
)


class KernelError(RuntimeError):
    """A T4, input, fit-scope, parity, or output invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_root(owner: str, slug: str) -> Path:
    candidates = (
        Path("/kaggle/input/datasets") / owner / slug,
        Path("/kaggle/input") / slug,
    )
    matches = [candidate for candidate in candidates if candidate.is_dir()]
    if len(matches) != 1:
        raise KernelError(f"missing or ambiguous dataset {owner}/{slug}")
    return matches[0]


def competition_root(slug: str) -> Path:
    candidates = (
        Path("/kaggle/input/competitions") / slug,
        Path("/kaggle/input") / slug,
    )
    matches = [candidate for candidate in candidates if candidate.is_dir()]
    if len(matches) != 1:
        raise KernelError(f"missing or ambiguous competition {slug}")
    return matches[0]


def require(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise KernelError(f"pinned file identity mismatch: {path}")


def run(command: list[str], env: dict[str, str]) -> float:
    print(json.dumps({"command": command}), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, env=env)
    return time.perf_counter() - started


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise KernelError(
                    f"unsafe archive member: {member.filename}"
                )
        archive.extractall(destination)


def configure_runtime(support: Path) -> dict[str, Any]:
    wheel_root = support / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(wheel_root),
            "zarr==3.2.1",
            "numcodecs==0.15.1",
            "donfig==0.8.1.post1",
        ],
        check=True,
        timeout=90,
    )
    import numpy as np
    import scipy
    import torch
    import zarr

    package_versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in ENVIRONMENT_PINS["packages"]
    }
    if (
        platform.python_version() != ENVIRONMENT_PINS["python"]
        or package_versions != ENVIRONMENT_PINS["packages"]
        or np.__version__ != ENVIRONMENT_PINS["packages"]["numpy"]
        or scipy.__version__ != ENVIRONMENT_PINS["packages"]["scipy"]
        or torch.__version__ != ENVIRONMENT_PINS["packages"]["torch"]
        or zarr.__version__ != ENVIRONMENT_PINS["packages"]["zarr"]
        or os.environ.get("CUDA_VERSION")
        != ENVIRONMENT_PINS["cuda_runtime"]
        or str(torch.version.cuda) != ENVIRONMENT_PINS["torch_cuda"]
        or int(torch.backends.cudnn.version()) != ENVIRONMENT_PINS["cudnn"]
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
        or torch.cuda.get_device_name(0) != ENVIRONMENT_PINS["device"]
    ):
        raise KernelError("exact banked T4 canonical environment is absent")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "python": platform.python_version(),
        "packages": package_versions,
        "cuda_runtime": os.environ["CUDA_VERSION"],
        "torch_cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()),
        "device_count": torch.cuda.device_count(),
        "device_0": torch.cuda.get_device_name(0),
        "seed": SEED,
        "deterministic_algorithms": True,
        "tf32": False,
    }


def main() -> int:
    started = time.perf_counter()
    legacy_assets = dataset_root(
        "onebitbrain", "trackastra-6bba-assets-v1"
    )
    option_assets = dataset_root(
        "onebitbrain", "trackastra-h1-option-c-assets-v1"
    )
    cache_dataset = dataset_root(
        "onebitbrain", "trackastra-h1-fit105-cache-v1"
    )
    support = dataset_root(
        "pilkwang", "biohub-tracking-support-pack-50ep-v1"
    )
    raw = competition_root("biohub-cell-tracking-during-development")

    raw_manifest = (
        option_assets
        / "source_overlay/kaggle_kernel/trackastra-h1-feature-cache-prep"
        / "raw_fit105_directory_manifest.json"
    )
    if not raw_manifest.is_file():
        raise KernelError(
            "authenticated 44b6+6bba fit-train105 raw manifest is absent"
        )
    require(raw_manifest, RAW_MANIFEST_SHA256)
    source = legacy_assets / "source_bundle"
    if not source.is_dir():
        raise KernelError("banked source bundle is absent")
    shutil.copytree(source, WORKING, dirs_exist_ok=True)
    require(
        legacy_assets / "trackastra_source.zip.bin",
        "c3e0a8f8cf642586944c0da3e56b8224f9a8d00e2af308b8de97bc9e31168770",
    )
    require(
        legacy_assets / "ctc.zip.bin",
        "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e",
    )
    require(
        legacy_assets / "model.pt",
        "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f",
    )
    safe_extract(
        legacy_assets / "trackastra_source.zip.bin",
        WORKING / "trackastra-source",
    )
    overlay = option_assets / "source_overlay"
    dependency_manifest_path = (
        overlay
        / "kaggle_kernel/trackastra-h1-feature-cache-prep"
        / "source_dependency_manifest.json"
    )
    require(
        dependency_manifest_path,
        SOURCE_DEPENDENCY_MANIFEST_SHA256,
    )
    dependency_manifest = json.loads(
        dependency_manifest_path.read_text(encoding="ascii")
    )
    if (
        dependency_manifest.get("schema")
        != "celltrack-alpha.trackastra-h1-source-dependencies.v1"
    ):
        raise KernelError("source dependency manifest schema drifted")
    for record in dependency_manifest.get("files", ()):
        relative = str(record.get("path", ""))
        expected = str(record.get("sha256", ""))
        origin = record.get("origin")
        if origin == "workspace":
            incoming = overlay / relative
            require(incoming, expected)
            destination = WORKING / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(incoming, destination)
            require(destination, expected)
        elif origin == "support_dataset":
            require(support / relative, expected)
        elif origin != "trackastra_archive":
            raise KernelError(
                f"unknown source dependency origin: {origin}"
            )
    contract_source = (
        overlay
        / "kaggle_kernel/trackastra-h1-feature-cache-prep"
        / "option_c_contract.json"
    )
    require(contract_source, CONTRACT_SHA256)
    contract = (
        WORKING
        / "kaggle_kernel/trackastra-h1-feature-cache-prep"
        / "option_c_contract.json"
    )
    contract.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(contract_source, contract)
    contract_payload = json.loads(contract.read_text(encoding="ascii"))
    if (
        contract_payload.get("environment") is None
        or contract_payload.get("status") != "locked-zero-wrfeat-primary"
        or contract_payload.get("revision", {}).get("decision")
        != "DECOUPLE_H1_FEATURE_ARMS_ZERO_WRFEAT_FIRST"
        or contract_payload.get("revision", {}).get(
            "real_mask_pre_smoke_prerequisite"
        )
        is not False
        or contract_payload.get("features", {})
        .get("h3_fallback", {})
        .get("status")
        != "CONDITIONAL_NOT_STARTED"
    ):
        raise KernelError("zero-wrfeat primary contract is absent or drifted")
    contract_environment = dict(contract_payload["environment"])
    contract_environment.pop("banked_t4_evidence", None)
    if contract_environment != ENVIRONMENT_PINS:
        raise KernelError("kernel and contract environment pins disagree")

    runtime = configure_runtime(support)
    projection = contract_payload.get("runtime_projection", {})
    timing_smoke_required = (
        projection.get("full_run_authorized") is not True
        or projection.get("status") != "PASS_BANKED_OPTION_C_CUDA_TIMING"
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(SEED),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CELLTRACK_HOLDOUT_ACCESS": "0",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(WORKING / "src"),
                    str(WORKING / "scripts"),
                    str(support / "repo/src"),
                )
            ),
        }
    )
    cache_root = (
        cache_dataset / "reports/g02/fitcache_117/bh9out/cache"
    )
    if timing_smoke_required:
        timings = []
        for identity in (
            "train:44b6_cf8fed6b",
            "train:6bba_edf14583",
        ):
            slug = identity.replace(":", "__", 1)
            base_smoke = (
                WORKING / f"reports/g02/h1_timing_base_{slug}"
            )
            output_smoke = (
                WORKING / f"reports/g02/h1_timing_enriched_{slug}"
            )
            stage_a_seconds = run(
                [
                    sys.executable,
                    str(
                        WORKING
                        / "scripts/build_trackastra_apncc_base_cache.py"
                    ),
                    "--contract",
                    str(contract),
                    "--contract-sha256",
                    CONTRACT_SHA256,
                    "--identity",
                    identity,
                    "--cache",
                    str(cache_root),
                    "--raw-root",
                    str(raw),
                    "--raw-manifest",
                    str(raw_manifest),
                    "--raw-manifest-sha256",
                    RAW_MANIFEST_SHA256,
                    "--edge-artifact",
                    str(legacy_assets / "edge.json"),
                    "--edge-auxiliary",
                    str(legacy_assets / "v45f4.pt"),
                    "--support-root",
                    str(support),
                    "--output",
                    str(base_smoke),
                ],
                env,
            )
            stage_b_seconds = run(
                [
                    sys.executable,
                    str(
                        WORKING
                        / "scripts/enrich_trackastra_apncc_cache.py"
                    ),
                    "--contract",
                    str(contract),
                    "--contract-sha256",
                    CONTRACT_SHA256,
                    "--identity",
                    identity,
                    "--source-root",
                    str(WORKING / "trackastra-source"),
                    "--checkpoint-zip",
                    str(legacy_assets / "ctc.zip.bin"),
                    "--checkpoint-member",
                    str(legacy_assets / "model.pt"),
                    "--cache",
                    str(cache_root),
                    "--base",
                    str(base_smoke),
                    "--output",
                    str(output_smoke),
                ],
                env,
            )
            base_report = json.loads(
                (base_smoke / "report.json").read_text(encoding="ascii")
            )
            enriched_report = json.loads(
                (output_smoke / "report.json").read_text(
                    encoding="ascii"
                )
            )
            if (
                base_report.get("execution", {}).get("device") != "cuda:0"
                or enriched_report.get("execution", {}).get("device")
                != "cuda:0"
                or base_report.get("execution", {})
                .get("device_attestation", {})
                .get("v45", {})
                .get("pass")
                is not True
                or enriched_report.get("execution", {})
                .get("device_attestation", {})
                .get("trackastra", {})
                .get("pass")
                is not True
                or enriched_report.get("execution", {})
                .get("device_attestation", {})
                .get("coordinates_and_features_follow_trackastra_device")
                is not True
                or {
                    str(row.get("identity"))
                    for row in enriched_report.get("option_c_parity", ())
                    if isinstance(row, dict)
                    and row.get("t4_array_equal") is True
                }
                != {identity}
            ):
                raise KernelError(
                    f"cuda:0 timing smoke device attestation failed: {identity}"
                )
            timings.append(
                {
                    "identity": identity,
                    "family": identity.split(":", 1)[1].split("_", 1)[0],
                    "stage_a_seconds": stage_a_seconds,
                    "stage_b_seconds_including_repeatability": stage_b_seconds,
                }
            )
        by_family = {row["family"]: row for row in timings}
        projected_seconds = sum(
            count
            * (
                float(by_family[family]["stage_a_seconds"])
                + float(
                    by_family[family][
                        "stage_b_seconds_including_repeatability"
                    ]
                )
            )
            for family, count in {"44b6": 36, "6bba": 69}.items()
        )
        projected_hours_with_margin = projected_seconds * 1.35 / 3600.0
        smoke = {
            "schema": "celltrack-alpha.trackastra-h1-cuda-timing-smoke.v1",
            "verdict": (
                "PASS_TIMING_ONLY_NO_FULL_CACHE"
                if projected_hours_with_margin <= 7.0
                else "STOP_PROJECTED_RUNTIME_EXCEEDS_7_HOURS"
            ),
            "runtime": runtime,
            "timings": timings,
            "projection_method": (
                "Conservative family-wise one-video timings; Stage B retains "
                "the witness repeatability rerun; multiply by family counts "
                "and add 35% margin."
            ),
            "projected_hours_with_margin": projected_hours_with_margin,
            "fits_inside_7_hours": projected_hours_with_margin <= 7.0,
            "full_run_executed": False,
            "human_contract_update_required": True,
        }
        (WORKING / "trackastra_h1_timing_smoke.json").write_text(
            json.dumps(smoke, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        print(json.dumps(smoke, sort_keys=True), flush=True)
        return 0 if smoke["fits_inside_7_hours"] else 3
    base = WORKING / "reports/g02/trackastra_h1_apncc_base"
    output = WORKING / "reports/g02/trackastra_h1_feature_cache"
    stage_a = run(
        [
            sys.executable,
            str(WORKING / "scripts/build_trackastra_apncc_base_cache.py"),
            "--contract",
            str(contract),
            "--contract-sha256",
            CONTRACT_SHA256,
            "--cache",
            str(cache_root),
            "--raw-root",
            str(raw),
            "--raw-manifest",
            str(raw_manifest),
            "--raw-manifest-sha256",
            RAW_MANIFEST_SHA256,
            "--edge-artifact",
            str(legacy_assets / "edge.json"),
            "--edge-auxiliary",
            str(legacy_assets / "v45f4.pt"),
            "--support-root",
            str(support),
            "--output",
            str(base),
        ],
        env,
    )
    stage_b = run(
        [
            sys.executable,
            str(WORKING / "scripts/enrich_trackastra_apncc_cache.py"),
            "--contract",
            str(contract),
            "--contract-sha256",
            CONTRACT_SHA256,
            "--source-root",
            str(WORKING / "trackastra-source"),
            "--checkpoint-zip",
            str(legacy_assets / "ctc.zip.bin"),
            "--checkpoint-member",
            str(legacy_assets / "model.pt"),
            "--cache",
            str(cache_root),
            "--base",
            str(base),
            "--output",
            str(output),
        ],
        env,
    )
    report = json.loads((output / "report.json").read_text(encoding="ascii"))
    base_report = json.loads(
        (base / "report.json").read_text(encoding="ascii")
    )
    parity_rows = report.get("option_c_parity")
    parity_identities = (
        {
            str(row.get("identity"))
            for row in parity_rows
            if isinstance(row, dict)
            and row.get("t4_array_equal") is True
        }
        if isinstance(parity_rows, list)
        else set()
    )
    if (
        report.get("aggregate", {}).get("video_count") != 105
        or base_report.get("execution", {}).get("device") != "cuda:0"
        or report.get("execution", {}).get("device") != "cuda:0"
        or base_report.get("execution", {})
        .get("device_attestation", {})
        .get("v45", {})
        .get("pass")
        is not True
        or base_report.get("execution", {})
        .get("device_attestation", {})
        .get("edge_scorer", {})
        .get("pass")
        is not True
        or report.get("execution", {})
        .get("device_attestation", {})
        .get("trackastra", {})
        .get("pass")
        is not True
        or report.get("execution", {})
        .get("device_attestation", {})
        .get("coordinates_and_features_follow_trackastra_device")
        is not True
        or parity_identities
        != {"train:44b6_cf8fed6b", "train:6bba_edf14583"}
    ):
        raise KernelError("final two-family coverage or T4 repeatability failed")
    final = {
        "schema": "celltrack-alpha.trackastra-h1-kaggle-run.v1",
        "verdict": "CACHE_GENERATED_QUARANTINED_PENDING_M5_T4_HUMAN_APPROVAL",
        "runtime": runtime,
        "scope": {
            "fit_train105_only": True,
            "family_counts": {"44b6": 36, "6bba": 69},
            "validation12_access": False,
            "cv57_access": False,
            "training": False,
            "competition_submission": False,
        },
        "stage_a_seconds": stage_a,
        "stage_b_seconds": stage_b,
        "elapsed_seconds": time.perf_counter() - started,
        "contract_sha256": CONTRACT_SHA256,
        "aggregate": report["aggregate"],
        "option_c_parity": report["option_c_parity"],
        "tolerance_contract_finalized": False,
        "human_approval_required": True,
    }
    (WORKING / "trackastra_h1_run_manifest.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
