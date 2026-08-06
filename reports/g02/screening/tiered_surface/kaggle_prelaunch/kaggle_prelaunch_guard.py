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
PRISTINE_PUBLIC_V32_NOTEBOOK_SHA256 = (
    "3f995a97a51c9517bdf8d8da2f606ffd184a51c841de87c962d911a336b7847d"
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


def _result(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    code_path: Path,
    metadata: dict[str, Any],
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    failures: list[str],
    step0: int | None,
    expected_initial_kernel_version: int | None,
) -> dict[str, Any]:
    """Return the common, CMUX-compatible predicate payload."""

    passed = not failures
    result = {
        "schema": "celltrack-alpha.kaggle-prelaunch-guard-result.v1",
        "status": "PASS" if passed else "FAIL",
        "eligible": passed and phase in {"inline", "launch"},
        "phase": phase,
        "lane_owner": contract["lane_owner"],
        "kernel_id": metadata.get("id"),
        "remote_preflight_kernel_id": contract.get("remote_preflight_kernel_id"),
        "kernel_dir": str(kernel_dir.resolve()),
        "metadata_sha256": sha256(metadata_path),
        "code_file": str(code_path.resolve()),
        "code_sha256": sha256(code_path),
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "registry_sha256": hashlib.sha256(
            json.dumps(registry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "dataset_sources": metadata.get("dataset_sources"),
        "kernel_sources": metadata.get("kernel_sources", []),
        "exclusive_dataset_source": contract.get("exclusive_dataset_source"),
        "expected_mount_root": contract.get("expected_mount_root"),
        "step0_line": step0,
        "failures": failures,
    }
    if expected_initial_kernel_version is not None:
        result["expected_initial_kernel_version"] = expected_initial_kernel_version
    return result


def check_exact_pristine_control(
    *,
    kernel_dir: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    code_path: Path,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None,
) -> dict[str, Any]:
    """Validate a source-byte-exact public-notebook control.

    This is deliberately narrower than the ordinary script guard.  Its only
    admissible source is the pinned public notebook, so adding instrumentation
    to satisfy the normal logging contract would invalidate the control.
    """

    failures: list[str] = []
    expected_kernel_id = contract.get("kernel_id")
    if not isinstance(expected_kernel_id, str) or metadata.get("id") != expected_kernel_id:
        failures.append("kernel ID drift")
    if not isinstance(expected_kernel_id, str) or title_slug(str(metadata.get("title", ""))) != expected_kernel_id.split("/", 1)[-1]:
        failures.append("kernel title does not resolve to the declared ID")
    if metadata.get("kernel_type") != "notebook":
        failures.append("pristine control must use a Kaggle notebook")
    if code_path.suffix != ".ipynb":
        failures.append("pristine control code file must remain .ipynb")
    if contract.get("pristine_notebook_sha256") != PRISTINE_PUBLIC_V32_NOTEBOOK_SHA256:
        failures.append("pristine control contract does not pin the known public V32 notebook")
    if sha256(code_path) != PRISTINE_PUBLIC_V32_NOTEBOOK_SHA256:
        failures.append("pristine control notebook bytes differ from the pinned public V32 source")
    if contract.get("code_sha256") != sha256(code_path):
        failures.append("pristine control code SHA drift")
    if metadata.get("dataset_sources") != contract.get("dataset_sources"):
        failures.append("dataset_sources drift")
    for key, expected in contract.get("kernel_properties", {}).items():
        if metadata.get(key) != expected:
            failures.append(f"kernel property drift: {key}")
    if expected_initial_kernel_version not in {None, 1}:
        failures.append("expected initial kernel version must be exactly 1")
    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None,
        expected_initial_kernel_version=expected_initial_kernel_version,
    )


def check_kernel(
    kernel_dir: Path,
    contract: dict[str, Any],
    registry: dict[str, Any],
    phase: str,
    expected_initial_kernel_version: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    metadata_path = kernel_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise GuardError("kernel-metadata.json missing")
    metadata = load_json(metadata_path)
    code_path = kernel_dir / str(metadata.get("code_file", ""))
    if not code_path.is_file():
        raise GuardError("declared code_file missing")
    if contract.get("execution_profile") == "exact_pristine_control":
        return check_exact_pristine_control(
            kernel_dir=kernel_dir,
            metadata_path=metadata_path,
            metadata=metadata,
            code_path=code_path,
            contract=contract,
            registry=registry,
            phase=phase,
            expected_initial_kernel_version=expected_initial_kernel_version,
        )
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
    expected_kernel_sources = contract.get("kernel_sources")
    if (
        expected_kernel_sources is not None
        and metadata.get("kernel_sources", []) != expected_kernel_sources
    ):
        failures.append("kernel_sources drift")
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

    allow_pristine_discovery = contract.get("allow_pristine_broad_discovery") is True
    if allow_pristine_discovery and contract.get("execution_profile") not in {
        "bisection_step0_only",
        "bisection_telemetry_no_input_hash",
    }:
        failures.append("only the registered Step-0 bisection stage may retain pristine broad discovery")
    if not allow_pristine_discovery:
        for snippet in FORBIDDEN_DISCOVERY_SNIPPETS:
            if snippet in source:
                failures.append(f"forbidden broad input discovery: {snippet}")

    logging_contract = contract.get("run_logging")
    required_logging = {
        "schema": "celltrack-alpha.run-log.v1",
        "stdout_jsonl": True,
        "persisted_with_artifacts": True,
        "download_on_success": True,
        "inventory_before_resolution": True,
        "failure_traceback_with_state": True,
    }
    step0_only = contract.get("execution_profile") == "bisection_step0_only"
    if step0_only:
        if logging_contract != {"mode": "intentionally_absent_for_first_bisection_stage"}:
            failures.append("Step-0-only bisection logging exception is malformed")
    elif not isinstance(logging_contract, dict) or any(
        logging_contract.get(key) != expected
        for key, expected in required_logging.items()
    ):
        failures.append("mandatory comprehensive run-logging contract is absent or malformed")
    required_logging_markers = (
        "RUN_LOG.jsonl",
        "inventory_input",
        "log_lookup",
        "stage_end",
        "preprocess_counts",
        "_run_failure",
    )
    missing_logging_markers = [marker for marker in required_logging_markers if marker not in source]
    if missing_logging_markers and not step0_only:
        failures.append(
            "code lacks mandatory comprehensive run-logging markers: "
            + ", ".join(missing_logging_markers)
        )

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
    if phase == "inline":
        if contract.get("remote_preflight_mode") != "inline_step0_before_heavy_work":
            failures.append("inline launch mode is not declared")
        # This is the only coherent one-version mode: the actual remote input
        # assertion is the first executable kernel statement, before imports
        # or the banked runner.  A remote result cannot exist before this same
        # one-shot version is pushed.
        if "STEP0 = run_step0()" not in source:
            failures.append("inline step-0 execution boundary is absent")
        if contract.get("step0_competition_path_visits") == 0:
            pre_step0_source = "\n".join(source.splitlines()[:first_step0_line - 1])
            if "/kaggle/input/competitions/" in pre_step0_source or "/kaggle/input/biohub-cell-tracking" in pre_step0_source:
                failures.append("competition path is referenced before inline step 0")
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
                "enable_gpu": contract["kernel_properties"].get("enable_gpu"),
                "heavy_work_started": False,
            }
            for key, expected in exact.items():
                if attestation.get(key) != expected:
                    failures.append(f"remote preflight attestation drift: {key}")
            if (
                contract.get("step0_elapsed_is_hard_failure", True)
                and float(attestation.get("step0_elapsed_seconds", 999.0)) >= float(
                    contract["step0_limit_seconds"]
                )
            ):
                failures.append("remote preflight exceeded step-0 limit")
            if contract.get("step0_competition_path_visits") == 0 and attestation.get("competition_path_visits") != 0:
                failures.append("remote preflight competition-path visit drift")

    if expected_initial_kernel_version not in {None, 1}:
        failures.append("expected initial kernel version must be exactly 1")

    return _result(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        code_path=code_path,
        metadata=metadata,
        contract=contract,
        registry=registry,
        phase=phase,
        failures=failures,
        step0=None if first_step0_line == 10**9 else first_step0_line,
        expected_initial_kernel_version=expected_initial_kernel_version,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--phase", choices=("static", "inline", "launch"), default="static")
    parser.add_argument("--expected-initial-kernel-version", type=int, choices=(1,))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = check_kernel(
        args.kernel_dir,
        load_json(args.contract),
        load_json(args.registry),
        args.phase,
        args.expected_initial_kernel_version,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
