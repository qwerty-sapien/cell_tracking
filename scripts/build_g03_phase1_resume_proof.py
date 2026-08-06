#!/usr/bin/env python3
"""Materialize standalone Kaggle scripts for the G-03 Phase 1 resume proof.

Kaggle's ``code_file`` contract executes one declared file.  The first G-03 A
package assumed sibling project modules would be importable and therefore died
before Torch or CUDA initialisation.  This builder deliberately renders all
runtime implementation and its role contract into one Python source file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_SOURCES = ("g03_checkpoint_resume.py", "g03_resume_torch.py")
CONFIG = {
    "seed": 20260731,
    "input_width": 256,
    "hidden_width": 512,
    "batch_size": 768,
    "learning_rate": 0.0003,
    "segment_minimum_seconds": 3600.0,
    "checkpoint_interval_seconds": 600.0,
    "amp_enabled": False,
    "dataloader_workers": 0,
    "device": "cuda",
}
HANDOFF_ID = "g03-phase1-a-to-b-v2"
HANDOFF_DATASET = "onebitbrain/g03-phase1-handoff-v2"
FIT_CACHE_DATASET = "onebitbrain/trackastra-h1-fit105-cache-v1"
SINGLE_FILE_NAME = "script.py"


class BuildError(RuntimeError):
    """The package cannot be materialized faithfully."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def source_hashes() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in COMPONENT_SOURCES:
        path = ROOT / "scripts" / name
        if not path.is_file():
            raise BuildError(f"source is absent: {path}")
        values[name] = sha256(path)
    return values


def role_contract(role: str, hashes: dict[str, str]) -> dict[str, Any]:
    kernel_id = f"onebitbrain/g03-phase1-resume-{role.lower()}-v2"
    return {
        "schema": "celltrack-alpha.g03-phase1-contract.v2",
        "role": role,
        "kernel_id": kernel_id,
        "producer_kernel": "onebitbrain/g03-phase1-resume-a-v2",
        "handoff_id": HANDOFF_ID,
        "source_hashes": hashes,
        "config": CONFIG,
        "scientific_selection_authorized": False,
    }


def metadata(role: str) -> dict[str, Any]:
    dataset_sources = []
    if role == "B":
        dataset_sources = [HANDOFF_DATASET, FIT_CACHE_DATASET]
    return {
        "id": f"onebitbrain/g03-phase1-resume-{role.lower()}-v2",
        "title": f"g03-phase1-resume-{role.lower()}-v2",
        "code_file": SINGLE_FILE_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": dataset_sources,
    }


def _remove_module_imports(source: str, module_name: str) -> str:
    """Remove future and sibling imports while retaining the exact implementation."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    removals: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {
            "__future__",
            module_name,
        }:
            removals.append((node.lineno - 1, node.end_lineno))
    for start, end in reversed(removals):
        del lines[start:end]
    return "".join(lines)


def _component_source(name: str) -> str:
    path = ROOT / "scripts" / name
    source = path.read_text(encoding="utf-8")
    if name == "g03_resume_torch.py":
        source = _remove_module_imports(source, "g03_checkpoint_resume")
    else:
        source = _remove_module_imports(source, "")
    return source.rstrip() + "\n"


def _single_file_source(role: str, hashes: dict[str, str]) -> str:
    contract = role_contract(role, hashes)
    contract_bytes = canonical(contract)
    preamble = """#!/usr/bin/env python3
# GENERATED FILE — all G-03 Phase 1 runtime dependencies are embedded below.
# Kaggle must execute this file alone; no sibling project Python files are required.
from __future__ import annotations

"""
    epilogue = f'''\n# Immutable self-contained transport boundary (rendered by the package builder).
_G03_INLINE_ROLE = {role!r}
_G03_INLINE_CONTRACT = json.loads({contract_bytes.decode("ascii")!r})
_G03_INLINE_CONTRACT_SHA256 = sha256_bytes(canonical_json_bytes(_G03_INLINE_CONTRACT))
_G03_INLINE_COMPONENT_HASHES = json.loads({canonical(hashes).decode("ascii")!r})
HANDOFF_ID = _G03_INLINE_CONTRACT["handoff_id"]
HANDOFF_DATASET_ROOT = Path("/kaggle/input/datasets/onebitbrain/g03-phase1-handoff-v2")
CONFIG = _G03_INLINE_CONTRACT["config"]

def _source_hashes(root: Path) -> dict[str, str]:
    del root
    return dict(_G03_INLINE_COMPONENT_HASHES)

def _load_contract(role: str) -> tuple[dict[str, Any], str, dict[str, str]]:
    observed = _source_hashes(Path("."))
    contract = dict(_G03_INLINE_CONTRACT)
    if (
        role != _G03_INLINE_ROLE
        or contract.get("schema") != "celltrack-alpha.g03-phase1-contract.v2"
        or contract.get("role") != role
        or contract.get("source_hashes") != observed
        or contract.get("config") != CONFIG
        or contract.get("handoff_id") != HANDOFF_ID
    ):
        raise TorchResumeProofError("embedded Phase 1 contract/source identity drifted")
    return contract, _G03_INLINE_CONTRACT_SHA256, observed

def _g03_single_file_selftest() -> None:
    contract, contract_sha, observed = _load_contract(_G03_INLINE_ROLE)
    if contract_sha != sha256_bytes(canonical_json_bytes(contract)):
        raise TorchResumeProofError("embedded contract hash self-test failed")
    if observed != _G03_INLINE_COMPONENT_HASHES:
        raise TorchResumeProofError("embedded component-hash self-test failed")
    print(json.dumps({{
        "status": "PASS_G03_SINGLE_FILE_SELFTEST",
        "role": _G03_INLINE_ROLE,
        "contract_sha256": contract_sha,
        "component_hashes": observed,
        "imports_project_modules": False,
    }}, sort_keys=True), flush=True)

if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--self-test"]:
        _g03_single_file_selftest()
    elif _G03_INLINE_ROLE == "A":
        run_a()
    elif _G03_INLINE_ROLE == "B":
        run_b()
    else:
        raise TorchResumeProofError("embedded role is invalid")
'''
    source = preamble + _component_source("g03_checkpoint_resume.py") + "\n" + _component_source("g03_resume_torch.py") + epilogue
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise BuildError(f"generated standalone script does not parse for {role}") from exc
    local_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            local_imports.extend(alias.name for alias in node.names if alias.name.startswith("g03_"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("g03_"):
            local_imports.append(node.module or "")
    if local_imports:
        raise BuildError(f"generated script retains project-local imports: {sorted(local_imports)}")
    return source


def _assert_isolated_selftest(script: Path, role: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        isolated = Path(directory)
        executable = isolated / SINGLE_FILE_NAME
        shutil.copy2(script, executable)
        process = subprocess.run(
            [sys.executable, executable.name, "--self-test"],
            cwd=isolated,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0:
            raise BuildError(
                f"isolated {role} self-test failed: rc={process.returncode}; stderr={process.stderr.strip()}"
            )
        try:
            receipt = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise BuildError(f"isolated {role} self-test produced invalid JSON") from exc
    if receipt.get("status") != "PASS_G03_SINGLE_FILE_SELFTEST" or receipt.get("role") != role:
        raise BuildError(f"isolated {role} self-test receipt drifted")
    return receipt


def materialize_role(root: Path, role: str, hashes: dict[str, str]) -> dict[str, Any]:
    directory = root / role.lower()
    directory.mkdir(parents=False)
    script = directory / SINGLE_FILE_NAME
    script.write_text(_single_file_source(role, hashes), encoding="utf-8")
    (directory / "kernel-metadata.json").write_bytes(canonical(metadata(role)))
    selftest = _assert_isolated_selftest(script, role)
    observed = {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
    return {"path": str(directory), "sha256": observed, "selftest": selftest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new empty package directory")
    parser.add_argument(
        "--self-test-single-file",
        action="store_true",
        help="materialize and execute isolated A/B single-file self-tests",
    )
    args = parser.parse_args(argv)
    if args.self_test_single_file:
        if args.output is not None:
            raise BuildError("--output is incompatible with --self-test-single-file")
        with tempfile.TemporaryDirectory() as directory:
            receipt = build(Path(directory) / "packages")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise BuildError("--output is required unless --self-test-single-file is used")
    receipt = build(args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise BuildError(f"refusing to replace existing package output: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise BuildError(f"package parent is not a directory: {output.parent}")
    hashes = source_hashes()
    output.mkdir(parents=True)
    receipt = {
        "schema": "celltrack-alpha.g03-phase1-single-file-package-build.v2",
        "source_hashes": hashes,
        "source_bundle_sha256": hashlib.sha256(canonical(hashes)).hexdigest(),
        "roles": [materialize_role(output, role, hashes) for role in ("A", "B")],
    }
    (output / "BUILD_RECEIPT.json").write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
