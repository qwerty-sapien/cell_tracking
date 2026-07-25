#!/usr/bin/env python3
"""Materialize the frozen two-stage 6bba Trackastra feature cache on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile


WORKING = Path("/kaggle/working")
EXPECTED_MANIFEST_SHA256 = (
    "53789b467b529ef9d10be1bc9cf5d8391a1ec65733810f72de46f59a20cbb0c4"
)
EXPECTED_RAW_MANIFEST_SHA256 = (
    "1a87473654dcfcc970daa06db4f8b2012af7a831720055980720f0449541c5ff"
)
CONTRACT_SHA256 = (
    "e060b284da623a373a83c8dd690985503af945399518f4d77eb517fddb8128ac"
)
PARITY_IDENTITY = "train:6bba_edf14583"
PARITY_REFERENCE_SHA256 = (
    "fed771aaf78df907460e8833a66d698067766e942258c088e31219e998115016"
)
EXPECTED_PACKAGES = {
    "trackastra_source.zip.bin": (
        88108,
        "c3e0a8f8cf642586944c0da3e56b8224f9a8d00e2af308b8de97bc9e31168770",
    ),
    "ctc.zip.bin": (
        101957234,
        "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e",
    ),
    "model.pt": (
        109906058,
        "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f",
    ),
    "edge.json": (
        14106,
        "5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034",
    ),
    "v45f4.pt": (
        8661,
        "1cff90475b819b5652310ccb73ba57e1dc20069885962a06a5e10a6060c04473",
    ),
}


class KernelError(RuntimeError):
    """A pinned input, scope, parity, or output invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or sha256(path) != expected_sha256
    ):
        raise KernelError(f"pinned input identity mismatch: {path}")


def verify_records(root: Path, records: object, label: str) -> None:
    if not isinstance(records, list) or not records:
        raise KernelError(f"{label} manifest records are absent")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise KernelError(f"{label} manifest record drifted")
        relative = str(record.get("path", ""))
        if relative in seen or relative.startswith("/") or ".." in Path(relative).parts:
            raise KernelError(f"{label} manifest path drifted: {relative}")
        seen.add(relative)
        require(
            root / relative,
            int(record.get("bytes", -1)),
            str(record.get("sha256", "")),
        )


def dataset_root(owner: str, slug: str) -> Path:
    candidates = (
        Path("/kaggle/input/datasets") / owner / slug,
        Path("/kaggle/input") / slug,
    )
    matches = [candidate for candidate in candidates if candidate.is_dir()]
    if len(matches) != 1:
        raise KernelError(
            f"expected exactly one mounted dataset root for {owner}/{slug}"
        )
    return matches[0]


def competition_root(slug: str) -> Path:
    candidates = (
        Path("/kaggle/input/competitions") / slug,
        Path("/kaggle/input") / slug,
    )
    matches = [candidate for candidate in candidates if candidate.is_dir()]
    if len(matches) != 1:
        raise KernelError(
            f"expected exactly one mounted competition root for {slug}"
        )
    return matches[0]


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if not members:
            raise KernelError(f"empty archive: {archive_path}")
        for member in members:
            target = (destination / member.filename).resolve()
            if (
                target != destination_resolved
                and destination_resolved not in target.parents
            ):
                raise KernelError(
                    f"unsafe archive member in {archive_path}: {member.filename}"
                )
        archive.extractall(destination)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise KernelError(f"expected JSON object: {path}")
    return value


def ensure_runtime(support_root: Path) -> dict[str, str]:
    wheel_root = support_root / "wheels"
    try:
        import zarr

        if zarr.__version__ != "3.2.1":
            raise ImportError(f"unexpected zarr {zarr.__version__}")
    except ImportError:
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
        )
        import zarr

    import numpy
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
        or "T4" not in torch.cuda.get_device_name(0)
    ):
        raise KernelError("authorized Kaggle T4 runtime is not available")
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "zarr": zarr.__version__,
        "cuda_device_count": str(torch.cuda.device_count()),
        "cuda_device_0": torch.cuda.get_device_name(0),
    }


def run_stage(command: list[str], env: dict[str, str]) -> float:
    print(json.dumps({"stage_command": command}), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, env=env)
    return time.perf_counter() - started


def verify_report(report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = load_json(report_path)
    aggregate = report.get("aggregate")
    files = report.get("files")
    if (
        report.get("verdict") != "PASS_LABEL_FREE_FEATURE_CACHE"
        or report.get("contract", {}).get("sha256") != CONTRACT_SHA256
        or report.get("scope", {}).get("family") != "6bba"
        or report.get("scope", {}).get("training") is not False
        or not isinstance(aggregate, dict)
        or aggregate.get("video_count") != 69
        or aggregate.get("proposal_count") != 110471
        or aggregate.get("inference_calls") != 5824
        or not isinstance(files, list)
        or len(files) != 69
    ):
        raise KernelError("final report coverage or scope drifted")
    identities = [str(row.get("identity")) for row in files]
    if (
        len(set(identities)) != 69
        or identities[0] != PARITY_IDENTITY
        or any(not identity.startswith("train:6bba_") for identity in identities)
    ):
        raise KernelError("final output identities drifted")
    verified: list[dict[str, Any]] = []
    for row in files:
        feature_path = WORKING / str(row["feature_path"])
        actual = sha256(feature_path)
        if actual != row.get("feature_sha256"):
            raise KernelError(f"feature checkpoint SHA drifted: {feature_path}")
        checkpoint = load_json(feature_path.with_suffix(".json"))
        if (
            checkpoint.get("identity") != row.get("identity")
            or checkpoint.get("contract_sha256") != CONTRACT_SHA256
            or checkpoint.get("feature_sha256") != actual
        ):
            raise KernelError(
                f"checkpoint identity drifted: {row.get('identity')}"
            )
        verified.append(
            {
                "identity": row["identity"],
                "feature_sha256": actual,
                "base_feature_sha256": row["base_feature_sha256"],
                "proposal_count": row["proposal_count"],
                "row_count": row["row_count"],
            }
        )
    return report, verified


def main() -> int:
    started = time.perf_counter()
    assets = dataset_root("onebitbrain", "trackastra-6bba-assets-v1")
    cache = dataset_root("onebitbrain", "trackastra-6bba-cache-v1")
    raw = competition_root("biohub-cell-tracking-during-development")
    support = dataset_root(
        "pilkwang", "biohub-tracking-support-pack-50ep-v1"
    )

    manifest_path = assets / "input_manifest.json"
    require(
        manifest_path,
        manifest_path.stat().st_size,
        EXPECTED_MANIFEST_SHA256,
    )
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema")
        != "celltrack-alpha.trackastra-6bba-kaggle-input-manifest.v1"
        or manifest.get("scope", {}).get("video_count") != 69
        or manifest.get("scope", {}).get("parity_reference")
        != PARITY_IDENTITY
        or manifest.get("contract", {}).get("sha256") != CONTRACT_SHA256
    ):
        raise KernelError("input manifest scope drifted")

    for name in EXPECTED_PACKAGES:
        expected_bytes, expected_sha = EXPECTED_PACKAGES[name]
        require(assets / name, expected_bytes, expected_sha)
    raw_manifest_path = (
        assets / "trackastra_6bba_raw_directory_manifest_v1.json"
    )
    require(
        raw_manifest_path,
        raw_manifest_path.stat().st_size,
        EXPECTED_RAW_MANIFEST_SHA256,
    )
    raw_manifest = load_json(raw_manifest_path)
    if (
        raw_manifest.get("schema")
        != "celltrack-alpha.trackastra-6bba-raw-directory-manifest.v1"
        or raw_manifest.get("scope", {}).get("video_count") != 69
        or raw_manifest.get("scope", {}).get("parity_reference")
        != PARITY_IDENTITY
    ):
        raise KernelError("raw directory manifest scope drifted")
    if manifest.get("raw_directory_manifest", {}).get("sha256") != (
        EXPECTED_RAW_MANIFEST_SHA256
    ):
        raise KernelError("input/raw manifest linkage drifted")
    verify_records(
        assets / "source_bundle",
        manifest.get("source_inputs"),
        "source bundle",
    )
    verify_records(cache, manifest.get("cache_inputs"), "cache")
    shutil.copytree(
        assets / "source_bundle", WORKING, dirs_exist_ok=True
    )
    safe_extract(
        assets / "trackastra_source.zip.bin",
        WORKING / "trackastra-source",
    )
    runtime = ensure_runtime(support)
    support_manifest = support / "ARTIFACT_MANIFEST.json"
    support_manifest_sha256 = sha256(support_manifest)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(WORKING / "src"),
            str(WORKING / "scripts"),
            str(support / "repo/src"),
            env.get("PYTHONPATH", ""),
        )
    )
    contract = (
        WORKING
        / "reports/g02/trackastra_full_pool_feature_cache_contract_v2.json"
    )
    cache_root = cache / "reports/g02/fitcache_117/bh9out/cache"
    base_root = WORKING / "reports/g02/trackastra_apncc_base_cache_v2"
    output_root = (
        WORKING / "reports/g02/trackastra_full_pool_feature_cache_v2"
    )

    stage_a_seconds = run_stage(
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
            str(raw_manifest_path),
            "--raw-manifest-sha256",
            EXPECTED_RAW_MANIFEST_SHA256,
            "--edge-artifact",
            str(assets / "edge.json"),
            "--edge-auxiliary",
            str(assets / "v45f4.pt"),
            "--support-root",
            str(support),
            "--output",
            str(base_root),
        ],
        env,
    )
    base_report = load_json(base_root / "report.json")
    if (
        base_report.get("verdict") != "PASS_APNCC_BASE_CACHE"
        or base_report.get("aggregate", {}).get("video_count") != 69
        or base_report.get("feature_count") != 48
        or base_report.get("model_stack", {}).get("trackastra_loaded")
        is not False
    ):
        raise KernelError("Stage A did not publish a complete 48-D cache")

    stage_b_seconds = run_stage(
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
            str(assets / "ctc.zip.bin"),
            "--checkpoint-member",
            str(assets / "model.pt"),
            "--cache",
            str(cache_root),
            "--base",
            str(base_root),
            "--output",
            str(output_root),
        ],
        env,
    )
    report, files = verify_report(output_root / "report.json")
    parity = files[0]
    final = {
        "schema": "celltrack-alpha.trackastra-6bba-kaggle-run.v1",
        "verdict": "PASS_69_OF_69_EXACT_V1_NUMERICAL_PARITY",
        "scope": {
            "family": "6bba",
            "video_count": len(files),
            "training": False,
            "competition_submission": False,
        },
        "contract_sha256": CONTRACT_SHA256,
        "input_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "support_manifest_sha256": support_manifest_sha256,
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - started,
        "stage_a_seconds": stage_a_seconds,
        "stage_b_seconds": stage_b_seconds,
        "parity": {
            "identity": parity["identity"],
            "exact_numerical_parity": True,
            "reference_file_sha256": PARITY_REFERENCE_SHA256,
            "output_feature_sha256": parity["feature_sha256"],
        },
        "aggregate": report["aggregate"],
        "files": files,
    }
    final_path = WORKING / "trackastra_6bba_feature_run_manifest.json"
    final_path.write_text(
        json.dumps(final, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print("FINAL_MANIFEST " + json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
