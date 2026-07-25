#!/usr/bin/env python3
"""Persist two one-video Trackastra extractions for T4 parity diagnosis."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
import zipfile


WORKING = Path("/kaggle/working")
PARITY_IDENTITY = "train:6bba_edf14583"
PARITY_SLUG = "train__6bba_edf14583"
KERNEL_LIMIT_SECONDS = 335.0
EXPECTED_MANIFEST_SHA256 = (
    "53789b467b529ef9d10be1bc9cf5d8391a1ec65733810f72de46f59a20cbb0c4"
)
EXPECTED_RAW_MANIFEST_SHA256 = (
    "1a87473654dcfcc970daa06db4f8b2012af7a831720055980720f0449541c5ff"
)
CONTRACT_SHA256 = (
    "e060b284da623a373a83c8dd690985503af945399518f4d77eb517fddb8128ac"
)
PRODUCTION_KERNEL_SHA256 = (
    "caf3e02e20f45dc688c16e3018269cae11bffee3a92c2930f675d1dca35613f8"
)
PARITY_REFERENCE_SHA256 = (
    "fed771aaf78df907460e8833a66d698067766e942258c088e31219e998115016"
)
OUTPUT_MANIFEST = WORKING / "trackastra_parity_t4_manifest.json"
RUN_PATHS = (
    WORKING / "trackastra_parity_t4_run1_row_features.npy",
    WORKING / "trackastra_parity_t4_run2_row_features.npy",
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
    """A pinned input, scope, runtime, or persisted-output invariant failed."""


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise KernelError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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
    destination.mkdir(parents=True, exist_ok=False)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if not members:
            raise KernelError(f"empty archive: {archive_path}")
        for member in members:
            target = (destination / member.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise KernelError(
                    f"unsafe archive member in {archive_path}: {member.filename}"
                )
        archive.extractall(destination)


def verify_records(
    root: Path,
    records: object,
    label: str,
    *,
    selected_prefix: str | None = None,
) -> None:
    if not isinstance(records, list) or not records:
        raise KernelError(f"{label} manifest records are absent")
    selected = []
    for record in records:
        if not isinstance(record, dict):
            raise KernelError(f"{label} manifest record drifted")
        relative = str(record.get("path", ""))
        if selected_prefix is None or Path(relative).name.startswith(selected_prefix):
            selected.append(record)
    if not selected:
        raise KernelError(f"{label} has no authorized one-video records")
    seen: set[str] = set()
    for record in selected:
        relative = str(record.get("path", ""))
        if relative in seen or relative.startswith("/") or ".." in Path(relative).parts:
            raise KernelError(f"{label} manifest path drifted: {relative}")
        seen.add(relative)
        require(
            root / relative,
            int(record.get("bytes", -1)),
            str(record.get("sha256", "")),
        )


def ensure_runtime(support_root: Path) -> dict[str, Any]:
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
            timeout=45,
        )
        import zarr

    import numpy as np
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
        or "T4" not in torch.cuda.get_device_name(0)
    ):
        raise KernelError("authorized single Kaggle T4 runtime is not available")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "zarr": zarr.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_0": torch.cuda.get_device_name(0),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "seed": 0,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def reset_seeds() -> None:
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def run_stage_a(
    *,
    ordinal: int,
    env: dict[str, str],
    contract: Path,
    cache_root: Path,
    raw: Path,
    raw_manifest: Path,
    assets: Path,
    support: Path,
    started: float,
) -> tuple[Path, float]:
    elapsed = time.perf_counter() - started
    remaining = KERNEL_LIMIT_SECONDS - elapsed
    if remaining <= 90.0:
        raise KernelError("insufficient T4 budget before repeated Stage A")
    output = WORKING / f"parity_run_{ordinal}/base"
    command = [
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
        EXPECTED_RAW_MANIFEST_SHA256,
        "--edge-artifact",
        str(assets / "edge.json"),
        "--edge-auxiliary",
        str(assets / "v45f4.pt"),
        "--support-root",
        str(support),
        "--identity",
        PARITY_IDENTITY,
        "--output",
        str(output),
    ]
    print(json.dumps({"stage_a_run": ordinal, "command": command}), flush=True)
    stage_started = time.perf_counter()
    subprocess.run(
        command,
        check=True,
        env=env,
        timeout=min(150.0, max(1.0, remaining - 120.0)),
    )
    return output, time.perf_counter() - stage_started


def extract_stage_b(
    *,
    ordinal: int,
    base_root: Path,
    contract_path: Path,
    source_root: Path,
    checkpoint_zip: Path,
    checkpoint_member: Path,
    cache_root: Path,
    model: Any,
    started: float,
) -> tuple[Any, dict[str, int], dict[str, Any], float]:
    import numpy as np

    from build_trackastra_apncc_base_cache import _cache_identity
    from build_trackastra_full_pool_feature_cache import (
        enrich_base_payload,
        load_contract,
        trackastra_margins_from_flattened,
    )
    from celltrack_alpha import bh_model
    from enrich_trackastra_apncc_cache import (
        _load_base_video,
        _validate_base_report,
    )

    elapsed = time.perf_counter() - started
    if elapsed >= KERNEL_LIMIT_SECONDS - 80.0:
        raise KernelError("insufficient T4 budget before repeated Stage B")
    reset_seeds()
    contract = load_contract(contract_path, CONTRACT_SHA256)
    _base_report, base_rows = _validate_base_report(
        base_root / "report.json",
        contract_sha256=CONTRACT_SHA256,
        expected_identities=(PARITY_IDENTITY,),
    )
    record = base_rows[PARITY_IDENTITY]
    base = _load_base_video(
        record,
        identity=PARITY_IDENTITY,
        contract_sha256=CONTRACT_SHA256,
    )
    meta = load_json(cache_root / f"{PARITY_SLUG}.json")
    detections = bh_model.load_sparse_cache_detections(
        cache_root / f"{PARITY_SLUG}.bhc",
        expected_identity=_cache_identity(meta),
    )
    stage_started = time.perf_counter()
    margins, inference = trackastra_margins_from_flattened(
        model=model,
        detections=detections,
        base=base,
        max_tokens=int(contract["features"]["max_tokens"]),
    )
    flattened = enrich_base_payload(base, margins)
    row_features = np.asarray(flattened["row_features"], dtype=np.float32)
    structural = {
        key: np.asarray(flattened[key]).copy()
        for key in (
            "challengers",
            "conflicts",
            "proposal_offsets",
            "transition_offsets",
            "transition_target_t",
            "geometry_confidences",
            "geometry_probabilities",
        )
    }
    seconds = time.perf_counter() - stage_started
    print(
        json.dumps(
            {
                "stage_b_run": ordinal,
                "seconds": seconds,
                "shape": list(row_features.shape),
                **inference,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del base, detections, margins, flattened
    gc.collect()
    return row_features, inference, structural, seconds


def file_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def diagnostic() -> dict[str, Any]:
    import numpy as np

    started = time.perf_counter()
    if PARITY_IDENTITY != "train:6bba_edf14583":
        raise KernelError("parity identity constant drifted")
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
    scope = manifest.get("scope")
    if (
        manifest.get("schema")
        != "celltrack-alpha.trackastra-6bba-kaggle-input-manifest.v1"
        or not isinstance(scope, dict)
        or scope.get("parity_reference") != PARITY_IDENTITY
        or scope.get("family") != "6bba"
        or scope.get("training") is not False
        or scope.get("label_access") is not False
        or scope.get("competition_submission") is not False
        or scope.get("validation12_access") is not False
        or scope.get("cv57_access") is not False
        or manifest.get("contract", {}).get("sha256") != CONTRACT_SHA256
    ):
        raise KernelError("input manifest scope drifted")
    for name, (expected_bytes, expected_sha) in EXPECTED_PACKAGES.items():
        require(assets / name, expected_bytes, expected_sha)
    raw_manifest = (
        assets / "trackastra_6bba_raw_directory_manifest_v1.json"
    )
    require(
        raw_manifest,
        raw_manifest.stat().st_size,
        EXPECTED_RAW_MANIFEST_SHA256,
    )
    verify_records(
        assets / "source_bundle",
        manifest.get("source_inputs"),
        "source bundle",
    )
    verify_records(
        cache,
        manifest.get("cache_inputs"),
        "cache",
        selected_prefix=PARITY_SLUG,
    )
    shutil.copytree(assets / "source_bundle", WORKING, dirs_exist_ok=True)
    safe_extract(
        assets / "trackastra_source.zip.bin",
        WORKING / "trackastra-source",
    )
    runtime = ensure_runtime(support)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(WORKING / "src"),
                    str(WORKING / "scripts"),
                    str(support / "repo/src"),
                    env.get("PYTHONPATH", ""),
                )
            ),
        }
    )
    contract = (
        WORKING
        / "reports/g02/trackastra_full_pool_feature_cache_contract_v2.json"
    )
    cache_root = cache / "reports/g02/fitcache_117/bh9out/cache"
    reset_seeds()
    base_root, stage_a_seconds_value = run_stage_a(
        ordinal=1,
        env=env,
        contract=contract,
        cache_root=cache_root,
        raw=raw,
        raw_manifest=raw_manifest,
        assets=assets,
        support=support,
        started=started,
    )
    stage_a_seconds = [stage_a_seconds_value]

    sys.path[:0] = [
        str(WORKING / "src"),
        str(WORKING / "scripts"),
        str(support / "repo/src"),
    ]
    from build_trackastra_full_pool_feature_cache import load_contract
    from trackastra_checkpoint_loader import load_mmap_meta

    contract_value = load_contract(contract, CONTRACT_SHA256)
    model = load_mmap_meta(
        source_root=WORKING / "trackastra-source",
        checkpoint_zip=assets / "ctc.zip.bin",
        checkpoint_member=assets / "model.pt",
        contract=contract_value,
    )
    runtime["trackastra_parameter_device"] = str(
        next(model.parameters()).device
    )
    rows = []
    inferences = []
    structural_runs = []
    stage_b_seconds = []
    for ordinal in (1, 2):
        row_features, inference, structural, seconds = extract_stage_b(
            ordinal=ordinal,
            base_root=base_root,
            contract_path=contract,
            source_root=WORKING / "trackastra-source",
            checkpoint_zip=assets / "ctc.zip.bin",
            checkpoint_member=assets / "model.pt",
            cache_root=cache_root,
            model=model,
            started=started,
        )
        if row_features.shape != (5409, 49) or row_features.dtype != np.float32:
            raise KernelError(
                f"Stage-B row_features identity drifted: {row_features.shape} "
                f"{row_features.dtype}"
            )
        np.save(RUN_PATHS[ordinal - 1], row_features, allow_pickle=False)
        rows.append(row_features)
        inferences.append(inference)
        structural_runs.append(structural)
        stage_b_seconds.append(seconds)
    mismatch_columns = [
        (
            f"apncc_context_{index:02d}"
            if index < 48
            else "trackastra_margin"
        )
        for index in range(49)
        if not np.array_equal(rows[0][:, index], rows[1][:, index])
    ]
    structural_equal = {
        key: bool(
            np.array_equal(structural_runs[0][key], structural_runs[1][key])
        )
        for key in structural_runs[0]
    }
    elapsed = time.perf_counter() - started
    if elapsed >= KERNEL_LIMIT_SECONDS:
        raise KernelError(
            f"diagnostic exceeded <0.1 T4-h cap: {elapsed:.6f} seconds"
        )
    runtime.update(
        {
            "elapsed_seconds": elapsed,
            "t4_hours": elapsed / 3600.0,
            "stage_a_seconds": stage_a_seconds,
            "stage_b_seconds": stage_b_seconds,
        }
    )
    return {
        "schema": "celltrack-alpha.trackastra-parity-t4-diagnostic.v1",
        "status": "PASS_OUTPUT_PERSISTED",
        "scope": {
            "identity": PARITY_IDENTITY,
            "video_count": 1,
            "family": "6bba",
            "training": False,
            "competition_submission": False,
            "runpod": False,
            "holdout_access": False,
            "cv57_access": False,
        },
        "pins": {
            "input_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "raw_manifest_sha256": EXPECTED_RAW_MANIFEST_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "production_kernel_sha256": PRODUCTION_KERNEL_SHA256,
            "m5_v1_reference_sha256": PARITY_REFERENCE_SHA256,
        },
        "runtime": runtime,
        "feature_names": [
            *[f"apncc_context_{index:02d}" for index in range(48)],
            "trackastra_margin",
        ],
        "inference": inferences,
        "t4_t4_repeatable": bool(np.array_equal(rows[0], rows[1])),
        "t4_t4_mismatch_columns": mismatch_columns,
        "structural_arrays_equal": structural_equal,
        "files": {
            path.name: file_record(path)
            for path in RUN_PATHS
        },
    }


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.perf_counter()
    try:
        result = diagnostic()
    except Exception as exc:
        failure = {
            "schema": "celltrack-alpha.trackastra-parity-t4-diagnostic.v1",
            "status": "STOP_FAIL_CLOSED",
            "scope": {
                "identity": PARITY_IDENTITY,
                "video_count": 1,
                "training": False,
                "competition_submission": False,
                "runpod": False,
            },
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
            "persisted_partial_files": {
                path.name: file_record(path)
                for path in RUN_PATHS
                if path.is_file()
            },
        }
        atomic_json(OUTPUT_MANIFEST, failure)
        print("STOP_FAIL_CLOSED " + json.dumps(failure, sort_keys=True), flush=True)
        return 0
    atomic_json(OUTPUT_MANIFEST, result)
    print("FINAL_MANIFEST " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
