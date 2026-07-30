#!/usr/bin/env python3
"""Static and attested prelaunch guard for G-02 Kaggle kernels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any


HEAVY_IMPORTS = {
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "tracksdata",
    "zarr",
}
FORBIDDEN_DISCOVERY_SNIPPETS = (
    '.glob("**/',
    ".glob('**/",
    ".rglob(",
)


class GuardError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GuardError(f"{path}: expected JSON object")
    return payload


def title_slug(title: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def imported_roots(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module:
        return {node.module.split(".", 1)[0]}
    return set()


def step0_line(tree: ast.Module) -> int:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "STEP0" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Call):
            name = value.func.id if isinstance(value.func, ast.Name) else ""
            if name in {"run_step0", "run_cellect_step0"}:
                return node.lineno
    raise GuardError("top-level STEP0 assignment calling run_step0 is missing")


def check_kernel(
    kernel_dir: Path,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    failures: list[str] = []
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise GuardError("kernel-metadata.json missing")
    metadata = load_json(metadata_path)
    code_path = kernel_dir / str(metadata.get("code_file", ""))
    if not code_path.is_file():
        raise GuardError("declared code_file missing")
    source = code_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(code_path))

    expected_kernel_id = contract["kernel_id"]
    if metadata.get("id") != expected_kernel_id:
        failures.append("kernel ID drift")
    expected_slug = expected_kernel_id.split("/", 1)[1]
    if title_slug(str(metadata.get("title", ""))) != expected_slug:
        failures.append("kernel title does not resolve to the declared ID")
    expected_sources = contract["dataset_sources"]
    if metadata.get("dataset_sources") != expected_sources:
        failures.append("dataset_sources drift")
    for key, expected in contract["kernel_properties"].items():
        if metadata.get(key) != expected:
            failures.append(f"kernel property drift: {key}")

    dataset_source = contract["exclusive_dataset_source"]
    owner = registry.get("dataset_ownership", {}).get(dataset_source)
    if owner != contract["lane_owner"]:
        failures.append("dataset source is not exclusively registered to this lane")

    expected_root = contract["expected_mount_root"]
    template = registry["policy"]["private_dataset_mount_template"]
    dataset_owner, dataset_slug = dataset_source.split("/", 1)
    rendered = template.format(owner=dataset_owner, slug=dataset_slug)
    if expected_root != rendered:
        failures.append("contract mount root does not match the registry template")
    if expected_root not in source:
        failures.append("code does not contain the exact owner-qualified mount root")

    for snippet in FORBIDDEN_DISCOVERY_SNIPPETS:
        if snippet in source:
            failures.append(f"forbidden broad input discovery: {snippet}")

    sibling_modules = {
        path.stem for path in kernel_dir.glob("*.py") if path.resolve() != code_path.resolve()
    }
    imported = set()
    for node in ast.walk(tree):
        imported.update(imported_roots(node))
    overlap = sorted(imported & sibling_modules)
    if overlap:
        failures.append(f"code_file imports unpackaged sibling modules: {overlap}")

    try:
        first_step0_line = step0_line(tree)
    except GuardError as exc:
        failures.append(str(exc))
        first_step0_line = 10**9
    for node in tree.body:
        if getattr(node, "lineno", 10**9) >= first_step0_line:
            break
        heavy = imported_roots(node) & HEAVY_IMPORTS
        if heavy:
            failures.append(f"heavy import before STEP0: {sorted(heavy)}")

    code_sha = sha256(code_path)
    if code_sha != contract["code_sha256"]:
        failures.append("code SHA drift")
    manifest_sha = contract["input_manifest_sha256"]
    if manifest_sha not in source:
        failures.append("pinned input-manifest SHA absent from code")
    checkpoint_sha = contract["checkpoint_sha256"]
    if checkpoint_sha not in source:
        failures.append("pinned checkpoint SHA absent from code")
    population_sha = contract["population_manifest_sha256"]
    if population_sha not in source:
        failures.append("pinned population-manifest SHA absent from code")
    request_count = int(contract["request_count"])
    request_file_count = int(contract["request_file_count"])
    wheel_count = int(contract["wheel_count"])
    if request_count <= 0:
        failures.append("request_count must be positive")
    if request_file_count <= 0:
        failures.append("request_file_count must be positive")
    if wheel_count <= 0:
        failures.append("wheel_count must be positive")
    if str(request_count) not in source.replace("_", ""):
        failures.append("positive exact request count absent from code")
    if str(request_file_count) not in source.replace("_", ""):
        failures.append("positive exact request-file count absent from code")
    if str(wheel_count) not in source.replace("_", ""):
        failures.append("positive exact wheel count absent from code")

    attestation = contract.get("remote_preflight_attestation")
    if phase == "launch":
        if contract.get("remote_preflight_kernel_id") != expected_kernel_id:
            failures.append("remote preflight is not bound to the launch kernel ID")
        if not isinstance(attestation, dict) or attestation.get("status") != "PASS":
            failures.append("remote preflight PASS attestation missing")
        else:
            exact = {
                "dataset_source": dataset_source,
                "dataset_version": contract["dataset_version"],
                "dataset_status": "ready",
                "input_manifest_sha256": manifest_sha,
                "checkpoint_sha256": checkpoint_sha,
                "population_manifest_sha256": population_sha,
                "request_count": request_count,
                "request_file_count": request_file_count,
                "population_file_count": int(contract["population_file_count"]),
                "wheel_count": wheel_count,
                "enable_gpu": False,
                "heavy_work_started": False,
            }
            for key, expected in exact.items():
                if attestation.get(key) != expected:
                    failures.append(f"remote preflight attestation drift: {key}")
            if float(attestation.get("step0_elapsed_seconds", 999.0)) >= float(
                contract["step0_limit_seconds"]
            ):
                failures.append("remote preflight exceeded step-0 limit")

    return {
        "schema": "celltrack-alpha.kaggle-prelaunch-guard-result.v1",
        "status": "PASS" if not failures else "FAIL",
        "phase": phase,
        "lane_owner": contract["lane_owner"],
        "kernel_id": metadata.get("id"),
        "remote_preflight_kernel_id": contract.get("remote_preflight_kernel_id"),
        "code_file": str(code_path),
        "code_sha256": code_sha,
        "dataset_sources": metadata.get("dataset_sources"),
        "exclusive_dataset_source": dataset_source,
        "expected_mount_root": expected_root,
        "step0_line": None if first_step0_line == 10**9 else first_step0_line,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--phase", choices=("static", "launch"), default="static")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = check_kernel(
        args.kernel_dir,
        load_json(args.contract),
        load_json(args.registry),
        args.phase,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
