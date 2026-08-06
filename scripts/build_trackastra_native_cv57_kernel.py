#!/usr/bin/env python3
"""Build the one-shot, fit-selected Trackastra CV57 bank kernel.

The GPU inference surface is byte-for-byte derived from the already completed
fit105 kernel.  This builder changes only the authenticated input population,
output name, and provenance fields; graph selection remains entirely determined
by the completed fit105 receipt and is applied locally after the bank returns.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRANCH = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
FIT_SOURCE = ROOT / "kaggle_kernel/g02-trackastra-native-fit-v1/g02_trackastra_native_fit_v1.py"
CACHE_ROOT = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
FIT_RESULT = BRANCH / "fit105_sweep_scored/FIT105_RESULTS.json"
OUTPUT = ROOT / "kaggle_kernel/g02-trackastra-native-cv57-v1"
SEAL = BRANCH / "CV57_FIT_SELECTION_SEAL.json"
BUILD_RECEIPT = BRANCH / "CV57_KERNEL_BUILD_RECEIPT.json"
SOURCE_AUTH = BRANCH / "CV57_SOURCE_AUTHENTICATION.json"
KERNEL_ID = "onebitbrain/g02-trackastra-native-cv57-v1"
SOURCE_KERNEL_ID = "onebitbrain/bhv45-cv57"
SOURCE_KERNEL_VERSION = 3
ASSET_DATASET = "onebitbrain/trackastra-6bba-assets-v1"
ASSET_ROOT = "/kaggle/input/datasets/onebitbrain/trackastra-6bba-assets-v1"
REMOTE_CACHE_ROOT = "/kaggle/input/bhv45-cv57/cache"
CHECKPOINT_SHA256 = "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def cache_manifest(root: Path) -> dict[str, Any]:
    entries = []
    bhc_paths = sorted(root.glob("train__*.bhc"))
    if len(bhc_paths) != 57:
        raise BuildError(f"expected 57 CV57 caches, found {len(bhc_paths)}")
    for bhc in bhc_paths:
        slug = bhc.stem
        meta_path = root / f"{slug}.json"
        intent_path = root / f"{slug}.intent.json"
        if not meta_path.is_file() or not intent_path.is_file():
            raise BuildError(f"incomplete cache triplet for {slug}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        identity = slug.replace("__", ":", 1)
        bhc_hash = sha256(bhc)
        if (
            intent.get("identity") != identity
            or meta.get("cache_sha256") != bhc_hash
            or intent.get("contract") != meta.get("contract")
        ):
            raise BuildError(f"cache provenance drift for {identity}")
        entries.append(
            {
                "identity": identity,
                "slug": slug,
                "bhc": {
                    "bytes": bhc.stat().st_size,
                    "sha256": bhc_hash,
                },
                "meta": {
                    "bytes": meta_path.stat().st_size,
                    "sha256": sha256(meta_path),
                },
                "cache_contract_sha256": canonical_sha(meta["contract"]),
            }
        )
    family_counts = {
        family: sum(row["identity"].startswith(f"train:{family}_") for row in entries)
        for family in ("44b6", "6bba")
    }
    if family_counts != {"44b6": 21, "6bba": 36}:
        raise BuildError(f"CV57 family counts drifted: {family_counts}")
    value = {
        "schema": "celltrack-alpha.trackastra-native-cv57-cache-manifest.v1",
        "source_kernel_id": SOURCE_KERNEL_ID,
        "source_kernel_version": SOURCE_KERNEL_VERSION,
        "scope": {
            "cv57_only": True,
            "video_count": 57,
            "family_counts": family_counts,
            "fit105_access": False,
            "gt_access": False,
        },
        "entries": entries,
    }
    value["manifest_sha256"] = canonical_sha(value)
    return value


def replace_function(source: str, name: str, next_name: str, replacement: str) -> str:
    pattern = rf"def {name}\(.*?(?=\ndef {next_name}\()"
    updated, count = re.subn(pattern, replacement.rstrip() + "\n\n", source, flags=re.S)
    if count != 1:
        raise BuildError(f"could not replace function {name}: {count}")
    return updated


def derived_runtime(
    fit_source: str,
    manifest: dict[str, Any],
    fit_result_sha: str,
    selected: dict[str, Any],
) -> str:
    identities = [row["identity"] for row in manifest["entries"]]
    manifest_sha = manifest["manifest_sha256"]
    source = fit_source
    substitutions = {
        '"""Bank Trackastra-native adjacent-edge weights on frozen fit105 detections."""':
            '"""Bank Trackastra-native adjacent-edge weights for the sealed CV57 measurement."""',
        'CACHE_ROOT = Path(\n    "/kaggle/input/datasets/onebitbrain/trackastra-h1-fit105-cache-v1"\n)':
            f'CACHE_ROOT = Path("{REMOTE_CACHE_ROOT}")',
        'INPUT_MANIFEST_SHA256 = (\n    "ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28"\n)':
            f'INPUT_MANIFEST_SHA256 = "{manifest_sha}"',
        'POPULATION_MANIFEST_SHA256 = INPUT_MANIFEST_SHA256':
            'POPULATION_MANIFEST_SHA256 = INPUT_MANIFEST_SHA256',
        'REQUEST_COUNT = 105': 'REQUEST_COUNT = 57',
        'REQUEST_FILE_COUNT = 210': 'REQUEST_FILE_COUNT = 114',
        'OUTPUT = WORKING / "trackastra_native_fit105_bank_v1"':
            'OUTPUT = WORKING / "trackastra_native_cv57_bank_v1"',
        '"schema": "celltrack-alpha.trackastra-native-fit105-bank-run.v1"':
            '"schema": "celltrack-alpha.trackastra-native-cv57-bank-run.v1"',
        '"fit_train105_only": True,\n            "family_counts": {"44b6": 36, "6bba": 69},\n            "validation12_access": False,\n            "cv57_access": False,':
            '"fit105_access": False,\n            "family_counts": {"44b6": 21, "6bba": 36},\n            "validation12_access": False,\n            "cv57_access": True,',
        'expected = {"44b6": 36, "6bba": 69}':
            'expected = {"44b6": 21, "6bba": 36}',
    }
    for old, new in substitutions.items():
        if source.count(old) != 1:
            raise BuildError(f"fit-kernel derivation anchor drifted: {old[:72]!r}")
        source = source.replace(old, new)

    literal_manifest = repr(manifest)
    literal_identities = repr(identities)
    selected_config = selected["configuration"]
    provenance_constants = f'''\nCV57_CACHE_MANIFEST = {literal_manifest}\nCV57_IDENTITIES = {literal_identities}\nFIT105_RESULTS_SHA256 = "{fit_result_sha}"\nSELECTED_CONFIGURATION_NAME = {selected["name"]!r}\nSELECTED_CONFIGURATION = {selected_config!r}\n'''
    anchor = 'SCALE_ZYX = (1.625, 0.40625, 0.40625)\n'
    if source.count(anchor) != 1:
        raise BuildError("constant insertion anchor drifted")
    source = source.replace(anchor, anchor + provenance_constants)

    step0 = '''def run_step0() -> dict:
    started = time.perf_counter()
    report = {
        "schema": "celltrack-alpha.trackastra-native-cv57-step0.v1",
        "status": "FAIL",
        "heavy_work_started": False,
        "competition_path_visits": 0,
        "dataset_source": "onebitbrain/trackastra-6bba-assets-v1",
        "kernel_source": "onebitbrain/bhv45-cv57",
        "kernel_source_version": 3,
        "dataset_version": 1,
        "dataset_status": "ready",
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "request_count": REQUEST_COUNT,
        "request_file_count": REQUEST_FILE_COUNT,
        "population_file_count": REQUEST_COUNT,
        "wheel_count": WHEEL_COUNT,
        "enable_gpu": True,
        "fit105_results_sha256": FIT105_RESULTS_SHA256,
        "selected_configuration_name": SELECTED_CONFIGURATION_NAME,
    }
    try:
        if not ASSET_ROOT.is_dir() or not CACHE_ROOT.is_dir():
            raise RuntimeError("exact asset or kernel-source mount is absent")
        require(ASSET_ROOT / "model.pt", CHECKPOINT_SHA256)
        require(ASSET_ROOT / "ctc.zip.bin", CHECKPOINT_ARCHIVE_SHA256)
        require(ASSET_ROOT / "trackastra_source.zip.bin", SOURCE_ARCHIVE_SHA256)
        manifest_payload = dict(CV57_CACHE_MANIFEST)
        claimed_manifest_sha = manifest_payload.pop("manifest_sha256", None)
        computed_manifest_sha = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if claimed_manifest_sha != INPUT_MANIFEST_SHA256 or computed_manifest_sha != INPUT_MANIFEST_SHA256:
            raise RuntimeError("embedded CV57 manifest SHA drifted")
        entries = CV57_CACHE_MANIFEST.get("entries", [])
        if len(entries) != REQUEST_COUNT or tuple(row["identity"] for row in entries) != tuple(CV57_IDENTITIES):
            raise RuntimeError("CV57 identity population drifted")
        for row in entries:
            slug = row["slug"]
            bhc = CACHE_ROOT / f"{slug}.bhc"
            meta = CACHE_ROOT / f"{slug}.json"
            if (
                not bhc.is_file()
                or bhc.stat().st_size != row["bhc"]["bytes"]
                or sha256(bhc) != row["bhc"]["sha256"]
                or not meta.is_file()
                or meta.stat().st_size != row["meta"]["bytes"]
                or sha256(meta) != row["meta"]["sha256"]
            ):
                raise RuntimeError(f"CV57 cache hash mismatch: {slug}")
        report["status"] = "PASS"
        report["identity_count"] = len(entries)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["step0_elapsed_seconds"] = time.perf_counter() - started
    WORKING.mkdir(parents=True, exist_ok=True)
    (WORKING / "step0.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\\\\n", encoding="ascii"
    )
    print(json.dumps({"STEP0": report}), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError(report.get("error", "step 0 failed"))
    return report


STEP0 = run_step0()
'''
    source = replace_function(source, "run_step0", "safe_extract", step0)
    if source.count("STEP0 = run_step0()") != 1:
        raise BuildError("inline step-0 boundary drifted")

    configure = '''def configure() -> tuple[Path, list[str]]:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    VIDEO_ROOT.mkdir()
    source_root = WORKING / "trackastra-source"
    safe_extract(ASSET_ROOT / "trackastra_source.zip.bin", source_root)
    bundle = ASSET_ROOT / "source_bundle"
    if not bundle.is_dir():
        raise RuntimeError("authenticated source_bundle is absent")
    shutil.copytree(bundle, WORKING, dirs_exist_ok=True)
    for location in (
        source_root,
        WORKING / "src",
        WORKING / "scripts",
        WORKING / "external/Tracking_Support/repo/src",
    ):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    return source_root, list(CV57_IDENTITIES)
'''
    source = replace_function(source, "configure", "load_model", configure)

    source = source.replace(
        'root = CACHE_ROOT / "reports/g02/fitcache_117/bh9out/cache"',
        'root = CACHE_ROOT',
    )
    if source.count('root = CACHE_ROOT') != 1:
        raise BuildError("cache loader derivation drifted")

    report_anchor = (
        '"heavy_work_started": True,\n'
        '        "step0": STEP0,\n'
        '        "checkpoint_sha256": CHECKPOINT_SHA256,\n'
    )
    if source.count(report_anchor) != 1:
        raise BuildError("report provenance anchor drifted")
    source = source.replace(
        report_anchor,
        report_anchor
        + '        "fit105_results_sha256": FIT105_RESULTS_SHA256,\n'
        + '        "selected_configuration_name": SELECTED_CONFIGURATION_NAME,\n'
        + '        "selected_configuration": SELECTED_CONFIGURATION,\n'
        + '        "source_kernel_id": "onebitbrain/bhv45-cv57",\n'
        + '        "source_kernel_version": 3,\n',
    )
    ast.parse(source)
    if "fit_train105_only" in source.split("def run_step0", 1)[1].split("STEP0 =", 1)[0]:
        raise BuildError("fit105 step-0 scope leaked into CV57 runtime")
    return source


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.seal.exists() or args.build_receipt.exists():
        raise BuildError("refusing to overwrite an existing package or seal")
    fit_result = json.loads(args.fit_result.read_text(encoding="utf-8"))
    source_auth = json.loads(args.source_auth.read_text(encoding="utf-8"))
    if fit_result.get("status") != "COMPLETE_EXACT_108_VARIANT_SWEEP":
        raise BuildError("fit105 108-way sweep is not complete")
    if len(fit_result.get("ranking", [])) != 108:
        raise BuildError("fit105 ranking is incomplete")
    selected = fit_result["selected"]
    if fit_result["ranking"][0]["name"] != selected["name"]:
        raise BuildError("selected configuration does not match rank 1")
    manifest = cache_manifest(args.cache_root)
    if (
        source_auth.get("status") != "PASS_REMOTE_VERSION3_INVENTORY_AND_LOCAL_BYTES"
        or source_auth.get("source_kernel", {}).get("id") != SOURCE_KERNEL_ID
        or source_auth.get("source_kernel", {}).get("version") != SOURCE_KERNEL_VERSION
        or source_auth.get("local_download", {}).get("cache_manifest_sha256")
        != manifest["manifest_sha256"]
    ):
        raise BuildError("authenticated remote CV57 source does not match local cache bytes")
    fit_result_sha = sha256(args.fit_result)
    runtime = derived_runtime(
        args.fit_source.read_text(encoding="utf-8"), manifest, fit_result_sha, selected
    )
    args.output.mkdir(parents=True)
    code = args.output / "script.py"
    atomic_text(code, runtime)
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_ID.split("/", 1)[1],
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": [ASSET_DATASET],
        "kernel_sources": [SOURCE_KERNEL_ID],
        "model_sources": [],
    }
    atomic_json(args.output / "kernel-metadata.json", metadata)
    contract = {
        "schema": "celltrack-alpha.trackastra-native-cv57-prelaunch.v1",
        "kernel_id": KERNEL_ID,
        "lane_owner": "TRACKASTRA_NATIVE_BASE_WAVE_V1",
        "dataset_sources": [ASSET_DATASET],
        "kernel_sources": [SOURCE_KERNEL_ID],
        "exclusive_dataset_source": ASSET_DATASET,
        "expected_mount_root": ASSET_ROOT,
        "dataset_version": 1,
        "kernel_properties": {"enable_gpu": True, "enable_internet": False},
        "code_sha256": sha256(code),
        "input_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "population_manifest_sha256": manifest["manifest_sha256"],
        "request_count": 57,
        "request_file_count": 114,
        "population_file_count": 57,
        "wheel_count": 3,
        "remote_preflight_kernel_id": KERNEL_ID,
        "remote_preflight_mode": "inline_step0_before_heavy_work",
        "step0_limit_seconds": 60.0,
        "step0_elapsed_is_hard_failure": False,
        "step0_competition_path_visits": 0,
        "authorization": {
            "source": "OWNER_SCOPE_AMENDMENT_20260731.json",
            "action": "exactly one fit-selected native+hybrid CV57 measurement",
            "hard_kill_seconds": 18000,
            "public_submission": False,
        },
    }
    atomic_json(args.output / "prelaunch-contract.json", contract)
    atomic_json(args.output / "cv57-cache-manifest.json", manifest)
    seal = {
        "schema": "celltrack-alpha.trackastra-native-cv57-fit-selection-seal.v1",
        "status": "IRREVOCABLY_SELECTED_BEFORE_CV57",
        "fit105_results": str(args.fit_result),
        "fit105_results_sha256": fit_result_sha,
        "selected_name": selected["name"],
        "selected_configuration": selected["configuration"],
        "selected_fit105_summary": selected["summary"],
        "selected_fit105_rank": 1,
        "fit105_candidate_count": 108,
        "hybrid_fit105_summary": fit_result["hybrid"]["summary"],
        "selection_rule": fit_result["selection_rule"],
        "cv57_source_authentication": str(args.source_auth),
        "cv57_source_authentication_sha256": sha256(args.source_auth),
        "cv57_access_during_selection": False,
        "cv57_candidate_measurements_before_seal": 0,
        "public_submission": False,
    }
    atomic_json(args.seal, seal)
    receipt = {
        "schema": "celltrack-alpha.trackastra-native-cv57-kernel-build.v1",
        "status": "READY_FOR_GUARD_NOT_PUSHED",
        "kernel_id": KERNEL_ID,
        "code_sha256": sha256(code),
        "metadata_sha256": sha256(args.output / "kernel-metadata.json"),
        "contract_sha256": sha256(args.output / "prelaunch-contract.json"),
        "cache_manifest_sha256": manifest["manifest_sha256"],
        "selection_seal_sha256": sha256(args.seal),
        "fit105_results_sha256": fit_result_sha,
        "source_fit_kernel_sha256": sha256(args.fit_source),
        "cv57_source_authentication_sha256": sha256(args.source_auth),
        "selected_name": selected["name"],
        "cv57_video_count": 57,
        "external_action": False,
    }
    atomic_json(args.build_receipt, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fit-source", type=Path, default=FIT_SOURCE)
    value.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    value.add_argument("--fit-result", type=Path, default=FIT_RESULT)
    value.add_argument("--output", type=Path, default=OUTPUT)
    value.add_argument("--seal", type=Path, default=SEAL)
    value.add_argument("--build-receipt", type=Path, default=BUILD_RECEIPT)
    value.add_argument("--source-auth", type=Path, default=SOURCE_AUTH)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
