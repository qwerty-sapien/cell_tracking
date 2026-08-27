#!/usr/bin/env python3
"""Build the deterministic one-cell private Kaggle notebook for bh9gb."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts import pkgbh9
except ModuleNotFoundError:  # Direct ``python scripts/mknb9.py`` execution.
    import pkgbh9  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "kaggle_kernel" / "bh9pkg"
DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "bh9.ipynb"
PACKAGE_FILES = (
    pkgbh9.ENTRYPOINT_NAME,
    pkgbh9.ARCHIVE_NAME,
    pkgbh9.ATTACHMENT_NAME,
    pkgbh9.MANIFEST_NAME,
    pkgbh9.METADATA_NAME,
)
RUNTIME_FILES = (
    pkgbh9.ATTACHMENT_NAME,
    pkgbh9.ARCHIVE_NAME,
)
SOURCE_LIMIT = pkgbh9.SOURCE_LIMIT


class NotebookError(RuntimeError):
    """The notebook cannot preserve the package's hash/runtime contract."""


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise NotebookError(f"invalid package JSON {path}: {exc}") from exc


def _embedded_files(source: bytes) -> dict[str, bytes]:
    try:
        tree = ast.parse(source, filename=pkgbh9.ENTRYPOINT_NAME)
        encoded: str | None = None
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "EMBED" for target in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "json"
                and node.value.func.attr == "loads"
                and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant)
                and isinstance(node.value.args[0].value, str)
            ):
                encoded = node.value.args[0].value
                break
        if encoded is None:
            raise ValueError("EMBED assignment is missing")
        envelope = json.loads(encoded)
    except (SyntaxError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise NotebookError(f"invalid self-contained entrypoint envelope: {exc}") from exc
    expected = {
        pkgbh9.MANIFEST_NAME, pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME,
    }
    if not isinstance(envelope, dict) or set(envelope) != expected:
        raise NotebookError("self-contained entrypoint member set is invalid")
    decoded: dict[str, bytes] = {}
    for name in sorted(expected):
        row = envelope[name]
        if not isinstance(row, dict) or set(row) != {"b85", "bytes", "sha256"}:
            raise NotebookError(f"invalid self-contained entrypoint record: {name}")
        try:
            raw = base64.b85decode(row["b85"].encode("ascii"))
        except (AttributeError, UnicodeError, ValueError) as exc:
            raise NotebookError(f"invalid self-contained entrypoint encoding: {name}") from exc
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise NotebookError(f"self-contained entrypoint hash mismatch: {name}")
        decoded[name] = raw
    return decoded


def attest_package(package_dir: Path) -> dict[str, dict[str, Any]]:
    package_dir = package_dir.resolve()
    observed = sorted(path.name for path in package_dir.iterdir() if path.is_file())
    if observed != sorted(PACKAGE_FILES):
        raise NotebookError(
            f"package member set mismatch: expected={sorted(PACKAGE_FILES)}, observed={observed}"
        )
    manifest = _load(package_dir / pkgbh9.MANIFEST_NAME)
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if (
        not isinstance(identity, dict)
        or manifest.get("schema") != "bh9-kaggle-package.v1"
        or manifest.get("identity_sha256")
        != pkgbh9.sha256_bytes(pkgbh9.canonical_json(identity))
    ):
        raise NotebookError("package manifest identity hash is invalid")
    if (
        identity.get("private") is not True
        or identity.get("internet") is not False
        or identity.get("trials") != 1
        or identity.get("accelerator") != "T4x2"
        or identity.get("deadline_seconds") != 3000
        or identity.get("runtime_attestation_level") != pkgbh9.RUNTIME_ATTESTATION_LEVEL
        or identity.get("self_contained_code_file") is not True
        or identity.get("offline_dependency_repair")
        != "attested_exact_wheel_paths_no_index_no_deps"
        or identity.get("embedded_files")
        != sorted((pkgbh9.MANIFEST_NAME, pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME))
        or identity.get("hermetic_child_import_layout") != list(pkgbh9.CHILD_IMPORT_LAYOUT)
        or identity.get("preserved_environment_keys")
        != list(pkgbh9.PRESERVED_ENVIRONMENT_KEYS)
        or identity.get("child_cwd") != "materialized_source_root"
        or identity.get("python_safe_path") is not True
        or identity.get("inherits_pythonpath") is not False
    ):
        raise NotebookError("package manifest does not describe the private offline one-trial T4x2 run")
    rows = identity.get("files")
    if (
        not isinstance(rows, list)
        or {row.get("path") for row in rows} != set(RUNTIME_FILES)
        or len(rows) != len(RUNTIME_FILES)
    ):
        raise NotebookError("package manifest file coverage is invalid")
    by_name = {row["path"]: row for row in rows}
    total = (package_dir / pkgbh9.MANIFEST_NAME).stat().st_size
    entrypoint = (package_dir / pkgbh9.ENTRYPOINT_NAME).read_bytes()
    embedded = _embedded_files(entrypoint)
    for name in PACKAGE_FILES:
        path = package_dir / name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if name in RUNTIME_FILES:
            row = by_name[name]
            if row.get("bytes") != len(raw) or row.get("sha256") != digest:
                raise NotebookError(f"package member hash mismatch: {name}")
            total += len(raw)
    for name, raw in embedded.items():
        if raw != (package_dir / name).read_bytes():
            raise NotebookError(f"entrypoint embedded sibling differs from package: {name}")
    if (
        total != identity.get("package_bytes")
        or total >= SOURCE_LIMIT
        or len(entrypoint) >= SOURCE_LIMIT
    ):
        raise NotebookError("package aggregate byte identity is invalid")
    metadata = _load(package_dir / pkgbh9.METADATA_NAME)
    if (
        metadata.get("is_private") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("enable_gpu") is not True
        or metadata.get("competition_sources") != [pkgbh9.COMPETITION_SLUG]
        or metadata.get("dataset_sources") != [pkgbh9.SUPPORT_REF]
    ):
        raise NotebookError("Kaggle metadata does not attest the private offline inputs")
    return {
        pkgbh9.ENTRYPOINT_NAME: {
            "bytes": len(entrypoint),
            "sha256": hashlib.sha256(entrypoint).hexdigest(),
            "base64": base64.b64encode(entrypoint).decode("ascii"),
        }
    }


def notebook_payload(package_dir: Path) -> dict[str, Any]:
    envelope = attest_package(package_dir)
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    source = f'''import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path("/kaggle/working").resolve(strict=True)
CODE = WORK / "bh9.py"
ENVELOPE = json.loads({json.dumps(encoded)})
if set(ENVELOPE) != {{"bh9.py"}} or CODE.exists() or CODE.parent.resolve(strict=True) != WORK:
    raise RuntimeError("one-trial standalone code path is invalid")
row = ENVELOPE["bh9.py"]
raw = base64.b64decode(row["base64"], validate=True)
if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
    raise RuntimeError("embedded code_file hash mismatch")
with CODE.open("xb") as handle:
    handle.write(raw)
keep = {json.dumps(list(pkgbh9.PRESERVED_ENVIRONMENT_KEYS))}
environment = {{key: os.environ[key] for key in keep if key in os.environ}}
environment.update({{
    "HOME": str(WORK),
    "TMPDIR": str(WORK),
    "PIP_CONFIG_FILE": os.devnull,
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONSAFEPATH": "1",
}})
python = Path(sys.executable).resolve(strict=True)
prefixes = {{Path(sys.prefix).resolve(strict=True), Path(sys.base_prefix).resolve(strict=True)}}
if not python.is_file() or not any(python == root or root in python.parents for root in prefixes):
    raise RuntimeError("Python executable escaped its runtime prefix")
result = subprocess.run(
    [str(python), "-P", str(CODE)], check=False, cwd=str(WORK), env=environment
)
if result.returncode != 0:
    raise RuntimeError("bh9gb failed closed; inspect /kaggle/working/bh9fail.json")
print("bh9gb completed; evidence=/kaggle/working/bh9.json run=/kaggle/working/bh9out/run.json fragments=/kaggle/working/bh9out/fragments")
'''
    return {
        "cells": [{
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {
                "is_private": True, "enable_internet": False,
                "accelerator": "T4x2", "trials": 1,
                "competition": pkgbh9.COMPETITION_SLUG,
                "dataset": pkgbh9.SUPPORT_REF,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebook(
    package_dir: Path = DEFAULT_PACKAGE,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if not output.name or len(output.name) >= 15:
        raise NotebookError(f"notebook basename must be under 15 characters: {output.name!r}")
    payload = notebook_payload(package_dir)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    if len(raw) >= SOURCE_LIMIT:
        raise NotebookError(f"notebook source must be <{SOURCE_LIMIT} bytes, got {len(raw)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(raw)
    temporary.replace(output)
    return {
        "path": str(output), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "strictly_under_budget": True, "cell_count": 1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(build_notebook(args.package, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
