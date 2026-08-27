#!/usr/bin/env python3
"""Build the deterministic private Kaggle T4x2 package for bh9gb."""

from __future__ import annotations

import argparse
import ast
import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import bh9gb  # noqa: E402
import pkgbh  # noqa: E402


DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "bh9pkg"
DEFAULT_ARCHIVE = bh9gb.ARCHIVE
DEFAULT_REGISTRY = bh9gb.REGISTRY
DEFAULT_SPLIT = bh9gb.SPLIT
DEFAULT_SUPPORT = bh9gb.SUPPORT

SOURCE_LIMIT = 1_000_000
RUNTIME_ATTESTATION_LEVEL = 6
KERNEL_SLUG = "bh9gb-t4x2"
KERNEL_REF = f"onebitbrain/{KERNEL_SLUG}"
KAGGLE_ACCELERATOR_ID = "NvidiaTeslaT4"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"
SUPPORT_DATASET_SLUG = "biohub-tracking-support-pack-50ep-v1"
SUPPORT_REF = f"pilkwang/{SUPPORT_DATASET_SLUG}"
COMPETITION_MOUNTS = (
    f"/kaggle/input/competitions/{COMPETITION_SLUG}",
    f"/kaggle/input/{COMPETITION_SLUG}",
)
SUPPORT_MOUNTS = (
    f"/kaggle/input/datasets/pilkwang/{SUPPORT_DATASET_SLUG}",
    f"/kaggle/input/{SUPPORT_DATASET_SLUG}",
)

ENTRYPOINT_NAME = "bh9.py"
ARCHIVE_NAME = "bh9src.zip"
ATTACHMENT_NAME = "bh9att.json"
MANIFEST_NAME = "bh9man.json"
METADATA_NAME = "kernel-metadata.json"  # Kaggle-required inherited basename.
FROZEN_NAME = "bh9m.json"
REGISTRY_NAME = "bh9reg.json"
SPLIT_NAME = "bh9sp.json"

CHILD_IMPORT_LAYOUT = (
    "embedded:src",
    "embedded:external/kaggle-cell-tracking-competition/src",
    "support:repo/src",
    "embedded:scripts",
)
PRESERVED_ENVIRONMENT_KEYS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
)

SUPPORT_SOURCE_PATHS = (
    "repo/src/biohub_tracking/__init__.py",
    "repo/src/biohub_tracking/models/__init__.py",
    *pkgbh.SUPPORT_SOURCE_PATHS,
)
SUPPORT_ARTIFACT_PATHS = (
    *pkgbh.SUPPORT_ARTIFACT_PATHS[:3],
    *SUPPORT_SOURCE_PATHS,
)
EXPECTED_ARTIFACT_HASHES = {
    **pkgbh.EXPECTED_ARTIFACT_HASHES,
    "source_sha256": "4abfa269dea9fcb2afd5a7d4cfc59ddbf323488f36a98e5631afc2868489a590",
}
SOURCE_ENTRY_PATHS = (
    "scripts/bh9gb.py",
    "scripts/scoreall.py",
)
LOCAL_MODULE_ROOTS = (
    "scripts",
    "src",
    "external/kaggle-cell-tracking-competition/src",
)
LOCAL_MODULE_PREFIXES = ("celltrack_alpha", "celltrack_baseline", "tracking_cellmot")

# Every import needed by the runner/scorer is probed before image discovery.
# Offline pip receives only entries that are absent or outside these bounds.
DEPENDENCIES = (
    ("torch", "torch", "2.2", "3", ">=2.2,<3"),
    ("numpy", "numpy", "2", "3", ">=2,<3"),
    ("scipy", "scipy", "1.11", "2", ">=1.11,<2"),
    ("zarr", "zarr", "3.0.10", "4", ">=3.0.10,<4"),
    ("numcodecs", "numcodecs", "0.14", "0.16", ">=0.14,<0.16"),
    ("Deprecated", "deprecated", "1", "2", ">=1,<2"),
    ("wrapt", "wrapt", "1.10", "3", ">=1.10,<3"),
    ("donfig", "donfig", "0.8", "1", ">=0.8,<1"),
    ("google-crc32c", "google_crc32c", "1.5", "2", ">=1.5,<2"),
    ("packaging", "packaging", "22", "27", ">=22,<27"),
    ("typing-extensions", "typing_extensions", "4.13", "5", ">=4.13,<5"),
    ("polars", "polars", "1.36", "2", ">=1.36,<2"),
    ("tracksdata", "tracksdata", "0.1", "1", ">=0.1,<1"),
    ("bidict", "bidict", "0.23", "1", ">=0.23,<1"),
    ("blosc2", "blosc2", "3", "5", ">=3,<5"),
    ("dask", "dask", "2024.10", "2027", ">=2024.10,<2027"),
    ("geff", "geff", "1.1", "2", ">=1.1,<2"),
    ("ilpy", "ilpy", "0.5", "1", ">=0.5,<1"),
    ("imagecodecs", "imagecodecs", "2025.3", "2027", ">=2025.3,<2027"),
    ("numba", "numba", "0.60", "1", ">=0.60,<1"),
    ("psygnal", "psygnal", "0.14", "1", ">=0.14,<1"),
    ("pyarrow", "pyarrow", "14", "25", ">=14,<25"),
    ("rich", "rich", "13", "16", ">=13,<16"),
    ("rustworkx", "rustworkx", "0.17", "1", ">=0.17,<1"),
    ("scikit-image", "skimage", "0.24", "1", ">=0.24,<1"),
    ("SQLAlchemy", "sqlalchemy", "2", "3", ">=2,<3"),
    ("tqdm", "tqdm", "4.60", "5", ">=4.60,<5"),
    ("ndindex", "ndindex", "1", "2", ">=1,<2"),
    ("msgpack", "msgpack", "1", "2", ">=1,<2"),
    ("numexpr", "numexpr", "2.14", "3", ">=2.14,<3"),
    ("requests", "requests", "2", "3", ">=2,<3"),
    ("threadpoolctl", "threadpoolctl", "3", "4", ">=3,<4"),
    ("click", "click", "8.1", "9", ">=8.1,<9"),
    ("cloudpickle", "cloudpickle", "3", "4", ">=3,<4"),
    ("fsspec", "fsspec", "2021.9", "2027", ">=2021.9,<2027"),
    ("partd", "partd", "1.4", "2", ">=1.4,<2"),
    ("PyYAML", "yaml", "5.4", "7", ">=5.4,<7"),
    ("toolz", "toolz", "0.12", "2", ">=0.12,<2"),
    ("locket", "locket", "1", "2", ">=1,<2"),
    ("geff-spec", "geff_spec", "1.1", "1.2", ">=1.1,<1.2"),
    ("networkx", "networkx", "3.2", "4", ">=3.2,<4"),
    ("pydantic", "pydantic", "2.11", "3", ">=2.11,<3"),
    ("typer", "typer", "0.14", "1", ">=0.14,<1"),
    ("annotated-types", "annotated_types", "0.6", "1", ">=0.6,<1"),
    ("pydantic-core", "pydantic_core", "2.30", "3", ">=2.30,<3"),
    ("typing-inspection", "typing_inspection", "0.4", "1", ">=0.4,<1"),
    ("polars-runtime-32", "_polars_runtime_32", "1.36", "2", ">=1.36,<2"),
    ("llvmlite", "llvmlite", "0.43", "1", ">=0.43,<1"),
    ("markdown-it-py", "markdown_it", "2.2", "5", ">=2.2,<5"),
    ("Pygments", "pygments", "2.13", "3", ">=2.13,<3"),
    ("mdurl", "mdurl", "0.1", "1", ">=0.1,<1"),
    ("pillow", "PIL", "10", "13", ">=10,<13"),
    ("imageio", "imageio", "2.33", "3", ">=2.33,<3"),
    ("tifffile", "tifffile", "2022.8", "2027", ">=2022.8,<2027"),
    ("lazy-loader", "lazy_loader", "0.4", "1", ">=0.4,<1"),
    ("greenlet", "greenlet", "1", "4", ">=1,<4"),
    ("charset-normalizer", "charset_normalizer", "2", "4", ">=2,<4"),
    ("idna", "idna", "2.5", "4", ">=2.5,<4"),
    ("urllib3", "urllib3", "1.26", "3", ">=1.26,<3"),
    ("certifi", "certifi", "2023.5", "2027", ">=2023.5,<2027"),
    ("annotated-doc", "annotated_doc", "0.0.2", "1", ">=0.0.2,<1"),
    ("shellingham", "shellingham", "1.3", "2", ">=1.3,<2"),
    ("PySCIPOpt", "pyscipopt", "6", "7", ">=6,<7"),
)

# Kaggle's CUDA image supplies torch. Every other required distribution must
# have exactly one compatible, hash-attested repair wheel in the support pack.
BASE_ENVIRONMENT_DISTRIBUTIONS = ("torch",)

_SAFE_SLUG = re.compile(r"[a-z0-9-]+\Z")
_SECRET_NAME = re.compile(
    r"(?:^|/)(?:\.env(?:\..*)?|credentials?(?:\..*)?|kaggle\.json|"
    r"id_rsa(?:\..*)?|[^/]*(?:secret|token)[^/]*|[^/]+\.(?:key|pem))$",
    re.IGNORECASE,
)


class PackageError(RuntimeError):
    """The bh9gb package cannot be built without violating its contract."""


def _module_name(relative: str) -> str:
    path = Path(relative)
    for root in LOCAL_MODULE_ROOTS:
        try:
            local = path.relative_to(root)
        except ValueError:
            continue
        if local.suffix != ".py" or local.name == "__init__.py":
            return ""
        return ".".join(local.with_suffix("").parts)
    return ""


def _local_module_path(repo_root: Path, module: str) -> str | None:
    if not module or any(not part or not part.isidentifier() for part in module.split(".")):
        return None
    suffix = Path(*module.split(".")).with_suffix(".py")
    matches = [
        (Path(root) / suffix).as_posix()
        for root in LOCAL_MODULE_ROOTS
        if (repo_root / root / suffix).is_file()
    ]
    if len(matches) > 1:
        raise PackageError(f"ambiguous local module {module!r}: {matches}")
    return matches[0] if matches else None


def _is_required_local_module(module: str, *, relative: bool) -> bool:
    return relative or module == "run_bh" or module.startswith(
        tuple(f"{prefix}." for prefix in LOCAL_MODULE_PREFIXES)
    )


def _imported_modules(source: bytes, relative: str) -> set[tuple[str, bool, bool]]:
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as error:
        raise PackageError(f"cannot parse packaged Python source {relative}: {error}") from error
    current = _module_name(relative)
    package = current.rpartition(".")[0]
    imported: set[tuple[str, bool, bool]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update((alias.name, False, True) for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom):
            is_relative = node.level > 0
            if is_relative:
                parts = package.split(".") if package else []
                keep = len(parts) - (node.level - 1)
                if keep < 0:
                    raise PackageError(f"relative import escapes package in {relative}")
                base_parts = parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base and node.module:
                imported.add((base, is_relative, True))
            for alias in node.names:
                if alias.name != "*" and base:
                    imported.add((f"{base}.{alias.name}", is_relative, node.module is None))
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            imported.add((node.args[0].value, False, True))
    return imported


def derive_source_paths(
    repo_root: Path,
    entry_paths: Sequence[str] = SOURCE_ENTRY_PATHS,
) -> tuple[str, ...]:
    """Return the recursive local import closure for the runtime entrypoints."""

    repo_root = repo_root.resolve()
    pending = list(entry_paths)
    discovered: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in discovered:
            continue
        path = repo_root / relative
        if not path.is_file():
            raise PackageError(f"packaged source file is missing: {relative}")
        discovered.add(relative)
        for module, is_relative, required in sorted(
            _imported_modules(path.read_bytes(), relative)
        ):
            dependency = _local_module_path(repo_root, module)
            if dependency is not None:
                if dependency not in discovered:
                    pending.append(dependency)
            elif required and _is_required_local_module(module, relative=is_relative):
                raise PackageError(
                    f"packaged source {relative} imports missing local module {module!r}"
                )
    return tuple(sorted(discovered))


SOURCE_PATHS = derive_source_paths(ROOT)


canonical_json = pkgbh.canonical_json
sha256_bytes = pkgbh.sha256_bytes
sha256_file = pkgbh.sha256_file
deterministic_zip = pkgbh.deterministic_zip


def support_source_tree_sha256(support_root: Path) -> str:
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


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise PackageError(f"required package input is missing: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalized_distribution_name(value: str) -> str:
    """Return the comparison form used for wheel/distribution names."""

    return re.sub(r"[-_.]+", "-", value).lower()


def wheel_filename_identity(name: str) -> tuple[str, str]:
    """Read normalized distribution and exact version from a safe wheel basename."""

    if name != Path(name).name or not name.endswith(".whl"):
        raise PackageError(f"wheel manifest path is not a safe wheel basename: {name!r}")
    parts = name[:-4].split("-")
    if len(parts) not in (5, 6) or not parts[0] or not parts[1]:
        raise PackageError(f"wheel basename does not have a deterministic PEP 427 identity: {name!r}")
    return normalized_distribution_name(parts[0]), parts[1]


def _version_release(value: str) -> tuple[int, ...] | None:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value))
    return tuple(map(int, match.group(1).split("."))) if match else None


def version_compatible(version: str, minimum: str, maximum: str) -> bool:
    """Apply the bh9 release bounds while explicitly accepting pre/local builds."""

    observed = _version_release(version)
    lower = _version_release(minimum)
    upper = _version_release(maximum)
    if observed is None or lower is None or upper is None:
        return False
    width = max(len(observed), len(lower), len(upper))
    padded = lambda item: item + (0,) * (width - len(item))
    return padded(lower) <= padded(observed) < padded(upper)


def dependency_wheel_plan(wheels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Audit all required repair wheels and return the deterministic exact plan."""

    by_distribution: dict[str, Mapping[str, Any]] = {}
    versions: dict[str, str] = {}
    for row in wheels:
        if not isinstance(row, Mapping):
            raise PackageError("wheel manifest contains a non-object record")
        name = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(size, int) or isinstance(size, bool) or size <= 0
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise PackageError(f"wheel manifest record is incomplete: {row!r}")
        distribution, version = wheel_filename_identity(name)
        if distribution in by_distribution:
            raise PackageError(f"wheel manifest has duplicate distribution: {distribution}")
        by_distribution[distribution] = row
        versions[distribution] = version

    base = {normalized_distribution_name(item) for item in BASE_ENVIRONMENT_DISTRIBUTIONS}
    expected = {
        normalized_distribution_name(distribution)
        for distribution, *_rest in DEPENDENCIES
        if normalized_distribution_name(distribution) not in base
    }
    if set(by_distribution) != expected:
        missing = sorted(expected - set(by_distribution))
        unexpected = sorted(set(by_distribution) - expected)
        raise PackageError(
            f"wheel manifest does not exactly cover repairable dependencies: "
            f"missing={missing}, unexpected={unexpected}"
        )

    plan = []
    for distribution, module, minimum, maximum, policy in DEPENDENCIES:
        normalized = normalized_distribution_name(distribution)
        if normalized in base:
            continue
        row = by_distribution[normalized]
        version = versions[normalized]
        if not version_compatible(version, minimum, maximum):
            raise PackageError(
                f"attested wheel version violates compatibility policy: "
                f"{distribution}=={version} not in {policy}"
            )
        plan.append({
            "distribution": distribution,
            "module": module,
            "version": version,
            "minimum": minimum,
            "maximum": maximum,
            "policy": policy,
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        })
    return plan


def exact_wheel_command(python: str, wheel_paths: Sequence[str]) -> list[str]:
    paths = [str(Path(value)) for value in wheel_paths]
    if not paths or any(not Path(value).is_absolute() or not value.endswith(".whl") for value in paths):
        raise PackageError("pip repair requires one or more exact absolute wheel paths")
    return [
        python, "-m", "pip", "install", "--no-index", "--no-deps",
        "--force-reinstall", *paths,
    ]


def compact_registry(path: Path, *, expected_train_count: int = bh9gb.TRAIN_COUNT) -> bytes:
    payload = bh9gb.strict_json(path)
    if not isinstance(payload, Mapping) or payload.get("artifact") != "dataset_registry_v1":
        raise PackageError("source registry is not dataset_registry_v1")
    source = payload.get("source")
    rows = payload.get("records")
    if not isinstance(source, Mapping) or not isinstance(rows, list):
        raise PackageError("source registry lacks records or ZIP identity")
    records = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("split") != "train":
            continue
        try:
            dataset = str(row["dataset"])
            identity = str(row["video_id"])
            digest = str(row["exact_identity_sha256"])
            shape = [int(item) for item in row["compact_metadata"]["image_shape"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise PackageError("train registry record lacks a required runner identity field") from exc
        if identity != f"train:{dataset}" or len(shape) != 4 or any(item <= 0 for item in shape):
            raise PackageError(f"invalid compact train identity {identity!r}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PackageError(f"invalid exact identity SHA-256 for {identity}")
        records.append({
            "split": "train", "dataset": dataset, "video_id": identity,
            "exact_identity_sha256": digest, "compact_metadata": {"image_shape": shape},
        })
    records.sort(key=lambda item: item["video_id"])
    if len(records) != expected_train_count or len({row["video_id"] for row in records}) != len(records):
        raise PackageError(
            f"compact registry requires {expected_train_count} unique train records, found {len(records)}"
        )
    compact = {
        "artifact": "dataset_registry_v1",
        "schema_version": 1,
        "source_registry_sha256": sha256_file(path),
        "source": {
            "zip_central_directory_sha256": source.get("zip_central_directory_sha256"),
            "zip_entry_count": source.get("zip_entry_count"),
        },
        "counts": {"train_videos": len(records), "videos": len(records)},
        "records": records,
    }
    return canonical_json(compact) + b"\n"


def compact_split(path: Path, registry_bytes: bytes, repo_root: Path, *, origin_registry_sha: str) -> bytes:
    payload = bh9gb.strict_json(path)
    if not isinstance(payload, Mapping) or payload.get("dataset_registry_sha256") != origin_registry_sha:
        raise PackageError("source split does not attest the source registry")
    expected = payload.get("source_hashes")
    if not isinstance(expected, Mapping):
        raise PackageError("source split lacks frozen scorer hashes")
    source_hashes = {}
    for relative in bh9gb.SCORER_FILES:
        actual = sha256_file(repo_root / relative)
        if expected.get(relative) != actual:
            raise PackageError(f"frozen scorer source mismatch: {relative}")
        source_hashes[relative] = actual
    compact = {
        "artifact": "bh9_split_attestation",
        "dataset_registry_sha256": sha256_bytes(registry_bytes),
        "source_split_sha256": sha256_file(path),
        "source_hashes": source_hashes,
    }
    return canonical_json(compact) + b"\n"


def _origin_inventory(archive: Path, registry: Path, identities: Sequence[str]) -> dict[str, Any]:
    if not archive.is_file():
        raise PackageError(f"local competition ZIP is missing: {archive}")
    inventory = bh9gb.image_inventory(archive, identities)
    payload = bh9gb.strict_json(registry)
    source = payload.get("source") if isinstance(payload, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("zip_central_directory_sha256") != inventory["central_sha256"]
        or source.get("zip_entry_count") != inventory["entry_count"]
    ):
        raise PackageError("local ZIP central-directory identity differs from the registry")
    return inventory


def _frozen_payload(
    inventory: Mapping[str, Any],
    selection: Mapping[str, Any],
    registry_bytes: bytes,
    split_bytes: bytes,
    *,
    origin_registry_sha: str,
    origin_split_sha: str,
    train_count: int,
) -> bytes:
    identity = {
        "seed": bh9gb.SEED,
        "target_compressed_bytes": bh9gb.TARGET_BYTES,
        "train_count": train_count,
        "archive_central_sha256": inventory["central_sha256"],
        "archive_entry_count": inventory["entry_count"],
        "registry_sha256": sha256_bytes(registry_bytes),
        "split_sha256": sha256_bytes(split_bytes),
        "origin_registry_sha256": origin_registry_sha,
        "origin_split_sha256": origin_split_sha,
        "compressed_bytes": dict(inventory["compressed_bytes"]),
        "selection": dict(selection),
    }
    payload = {
        "schema": "celltrack-alpha.bh9m.v1",
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical_json(identity)),
    }
    return canonical_json(payload) + b"\n"


def _archive_members(
    repo_root: Path,
    archive: Path,
    registry: Path,
    split: Path,
    *,
    expected_train_count: int,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    source_paths = derive_source_paths(repo_root)
    registry_bytes = compact_registry(registry, expected_train_count=expected_train_count)
    split_bytes = compact_split(
        split, registry_bytes, repo_root, origin_registry_sha=sha256_file(registry)
    )
    registry_doc = json.loads(registry_bytes)
    identities = tuple(row["video_id"] for row in registry_doc["records"])
    inventory = _origin_inventory(archive, registry, identities)
    selection = bh9gb.select_bytes(identities, inventory["compressed_bytes"])
    if expected_train_count == bh9gb.TRAIN_COUNT and (
        selection["selected_count"] != 22
        or selection["selected_compressed_bytes"] != 9_337_408_312
        or selection["selection_sha256"]
        != "43687468e9f51d9d8e5795eb91bc081a645a3384fd599ebb6f56cb7129ba67f1"
        or inventory["central_sha256"]
        != "5edffe5596e0918f92336ffb269b17df4f2e5f41b55a67bfa01e0253d58cdc77"
    ):
        raise PackageError("local inputs do not reproduce the frozen bh9gb selection")
    frozen = _frozen_payload(
        inventory,
        selection,
        registry_bytes,
        split_bytes,
        origin_registry_sha=sha256_file(registry),
        origin_split_sha=sha256_file(split),
        train_count=expected_train_count,
    )
    members = {relative: (repo_root / relative).read_bytes() for relative in source_paths}
    members.update({
        "src/celltrack_alpha/__init__.py": b'"""bh9gb runtime modules."""\n',
        "src/celltrack_baseline/__init__.py": b'"""bh9gb scorer helpers."""\n',
        "external/kaggle-cell-tracking-competition/src/tracking_cellmot/__init__.py":
            b'"""tracking-cellmot scorer."""\n',
        REGISTRY_NAME: registry_bytes,
        SPLIT_NAME: split_bytes,
        FROZEN_NAME: frozen,
    })
    unsafe = sorted(
        name for name in members
        if Path(name).is_absolute() or ".." in Path(name).parts or _SECRET_NAME.search(name)
    )
    if unsafe:
        raise PackageError(f"unsafe or secret-like source archive member names: {unsafe}")
    return members, {"inventory": inventory, "selection": selection}


ENTRYPOINT_SOURCE = r'''#!/usr/bin/env python3
"""Offline, fail-closed bh9gb Kaggle T4x2 launcher."""
from __future__ import annotations
import base64, hashlib, importlib, importlib.metadata, json, os, re, signal, subprocess, sys, time, traceback, zipfile
from pathlib import Path

WORK=Path("/kaggle/working")
INPUT=Path("/kaggle/input")
PKG=WORK/"bh9pkg"; MAN=PKG/"bh9man.json"; ATT=PKG/"bh9att.json"; SRC=PKG/"bh9src.zip"
ROOT=WORK/"bh9src"; OK=WORK/"bh9.json"; FAIL=WORK/"bh9fail.json"; OUT=WORK/"bh9out"
DEADLINE=3000
LEVEL=6
EMBED=json.loads("__BH9_EMBED__")
CM=("/kaggle/input/competitions/biohub-cell-tracking-during-development","/kaggle/input/biohub-cell-tracking-during-development")
SM=("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1","/kaggle/input/biohub-tracking-support-pack-50ep-v1")
DEPS=json.loads("__BH9_DEPS__")
KSHAPE="__BH9_SHAPE__"
BASE={"torch"}
LAYOUT=("embedded:src","embedded:external/kaggle-cell-tracking-competition/src","support:repo/src","embedded:scripts")
KEEP=("CUDA_HOME","CUDA_PATH","LD_LIBRARY_PATH","LIBRARY_PATH","NVIDIA_DRIVER_CAPABILITIES","NVIDIA_VISIBLE_DEVICES","PATH")
REQ=(
 ("celltrack_alpha","embedded_src"),("celltrack_alpha.bh_model","embedded_src"),("celltrack_alpha.bh_graph","embedded_src"),("celltrack_alpha.bh_edge","embedded_src"),("celltrack_alpha.v45_head","embedded_src"),("celltrack_alpha.bh_route","embedded_src"),("celltrack_alpha.kaggle_calibration","embedded_src"),("celltrack_alpha.official_eval","embedded_src"),
 ("celltrack_baseline.rule_based","embedded_src"),("tracking_cellmot","official_src"),("tracking_cellmot.metrics","official_src"),("tracking_cellmot.division_metrics","official_src"),
 ("biohub_tracking","support_src"),("biohub_tracking.models.temporal_unet","support_src"),("biohub_tracking.models.simple_node_transformer","support_src"),
 ("run_bh","scripts"),("scoreall","scripts"),("bh9gb","scripts"),
)
GPU_ENV_KEYS=("CUDA_VISIBLE_DEVICES","CUDA_VERSION","NVIDIA_VISIBLE_DEVICES","NVIDIA_DRIVER_CAPABILITIES","NVIDIA_REQUIRE_CUDA")
GPU_DIAG=None

class LaunchError(RuntimeError): pass
def canon(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as h:
  for b in iter(lambda:h.read(1048576),b""): d.update(b)
 return d.hexdigest()
def load(path):
 def pairs(items):
  out={}
  for k,v in items:
   if k in out: raise ValueError("duplicate key "+k)
   out[k]=v
  return out
 return json.loads(Path(path).read_text(encoding="ascii"),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def ver(v):
 m=re.match(r"\s*(\d+(?:\.\d+)*)",str(v)); return tuple(map(int,m.group(1).split("."))) if m else None
def lt(a,b):
 n=max(len(a),len(b)); return a+(0,)*(n-len(a)) < b+(0,)*(n-len(b))
def compat(v,lo,hi):
 v,lo,hi=ver(v),ver(lo),ver(hi); return v is not None and lo is not None and hi is not None and not lt(v,lo) and lt(v,hi)
def norm(v): return re.sub(r"[-_.]+","-",str(v)).lower()
def wheel_id(name):
 if not isinstance(name,str) or name!=Path(name).name or not name.endswith(".whl"): raise LaunchError("unsafe wheel manifest basename: "+repr(name))
 parts=name[:-4].split("-")
 if len(parts) not in (5,6) or not parts[0] or not parts[1]: raise LaunchError("invalid deterministic wheel basename: "+name)
 return norm(parts[0]),parts[1]
def wheel_plan(rows):
 if not isinstance(rows,list): raise LaunchError("wheel manifest rows are missing")
 by={}; versions={}
 for row in rows:
  if not isinstance(row,dict) or type(row.get("bytes")) is not int or row["bytes"]<=0 or not isinstance(row.get("sha256"),str) or re.fullmatch(r"[0-9a-f]{64}",row["sha256"]) is None: raise LaunchError("invalid wheel manifest record")
  dist,version=wheel_id(row.get("path"))
  if dist in by: raise LaunchError("duplicate attested wheel distribution: "+dist)
  by[dist]=row; versions[dist]=version
 expected={norm(d) for d,m,lo,hi,s in DEPS if norm(d) not in BASE}
 if set(by)!=expected: raise LaunchError("attested wheel dependency coverage mismatch: missing=%r unexpected=%r"%(sorted(expected-set(by)),sorted(set(by)-expected)))
 out=[]
 for dist,mod,lo,hi,policy in DEPS:
  key=norm(dist)
  if key in BASE: continue
  row=by[key]; version=versions[key]
  if not compat(version,lo,hi): raise LaunchError("attested exact wheel violates compatibility policy: %s==%s not in %s"%(dist,version,policy))
  out.append({"distribution":dist,"module":mod,"version":version,"minimum":lo,"maximum":hi,"policy":policy,"path":row["path"],"bytes":row["bytes"],"sha256":row["sha256"]})
 return out
def remaining(end,stage):
 n=end-time.monotonic()
 if n<=0: raise LaunchError("50-minute deadline reached before "+stage)
 return n
def under(path,root): return path==root or root in path.parents
def safe_dir(path,root,label):
 try: root=Path(root).resolve(strict=True); path=Path(path).resolve(strict=True)
 except OSError as e: raise LaunchError(label+" path is missing or unresolved") from e
 if not root.is_dir() or not path.is_dir() or not under(path,root): raise LaunchError(label+" path is not a directory under its allowed root: "+str(path))
 return path
def safe_file(path,root,label):
 try: root=Path(root).resolve(strict=True); path=Path(path).resolve(strict=True)
 except OSError as e: raise LaunchError(label+" path is missing or unresolved") from e
 if not root.is_dir() or not path.is_file() or not under(path,root): raise LaunchError(label+" path is not a file under its allowed root: "+str(path))
 return path
def python_exe():
 try: path=Path(sys.executable).resolve(strict=True); roots={Path(sys.prefix).resolve(strict=True),Path(sys.base_prefix).resolve(strict=True)}
 except OSError as e: raise LaunchError("Python executable path is missing or unresolved") from e
 if not path.is_file() or not any(under(path,root) for root in roots): raise LaunchError("Python executable escaped its runtime prefix: "+str(path))
 return path
def base_env():
 env={k:os.environ[k] for k in KEEP if k in os.environ}
 env.update({"HOME":str(WORK),"TMPDIR":str(WORK),"PIP_CONFIG_FILE":os.devnull,"PIP_NO_INDEX":"1","PIP_DISABLE_PIP_VERSION_CHECK":"1","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","PYTHONSAFEPATH":"1"})
 return env
def child_paths(support):
 embedded=safe_dir(ROOT,WORK,"materialized embedded source root")
 support=safe_dir(support,support,"attested support root")
 paths={
  "embedded_src":safe_dir(embedded/"src",embedded,"materialized embedded source root/src"),
  "official_src":safe_dir(embedded/"external/kaggle-cell-tracking-competition/src",embedded,"embedded official scorer source root"),
  "support_src":safe_dir(support/"repo/src",support,"attested Tracking_Support repo/src"),
  "scripts":safe_dir(embedded/"scripts",embedded,"materialized embedded scripts root"),
 }
 if tuple(("embedded:src","embedded:external/kaggle-cell-tracking-competition/src","support:repo/src","embedded:scripts"))!=LAYOUT: raise LaunchError("child import layout drift")
 return paths
def child_env(paths):
 if set(paths)!={"embedded_src","official_src","support_src","scripts"}: raise LaunchError("child import path key drift")
 env=base_env(); env.update({"PYTHONPATH":os.pathsep.join(str(paths[x]) for x in ("embedded_src","official_src","support_src","scripts")),"CUDA_VISIBLE_DEVICES":"0,1","BH9_TRIAL":"1"})
 return env
def probe_required(paths,env,end):
 code="""import importlib,json,sys\nfrom pathlib import Path\ndef under(p,r): return p==r or r in p.parents\nrows=json.loads(sys.stdin.read()); out={}\nfor name,key,root in rows:\n m=importlib.import_module(name); allowed=Path(root).resolve(strict=True); origin=getattr(m,'__file__',None)\n locations=[Path(origin).resolve(strict=True)] if origin else [Path(x).resolve(strict=True) for x in getattr(m,'__path__',())]\n if not locations or any(not under(x,allowed) for x in locations): raise RuntimeError('required import escaped attested root: '+name+' '+repr([str(x) for x in locations]))\n out[name]=[str(x) for x in locations]\nprint(json.dumps(out,sort_keys=True,separators=(',',':')))\n"""
 rows=[[name,key,str(paths[key])] for name,key in REQ]
 run=subprocess.run([str(python_exe()),"-P","-c",code],input=json.dumps(rows),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=str(ROOT),env=env,timeout=remaining(end,"required source import probe"))
 if run.returncode!=0: raise LaunchError("required source import failed before data reads: "+run.stderr[-2000:])
 try: out=json.loads(run.stdout)
 except Exception as e: raise LaunchError("required source import probe returned invalid JSON") from e
 if set(out)!={name for name,key in REQ}: raise LaunchError("required source import probe coverage mismatch")
 return out
def materialize():
 expected={"bh9man.json","bh9att.json","bh9src.zip"}
 if set(EMBED)!=expected: raise LaunchError("embedded runtime member set mismatch")
 decoded={}
 for name in sorted(expected):
  row=EMBED[name]
  if not isinstance(row,dict) or set(row)!={"b85","bytes","sha256"}: raise LaunchError("invalid embedded record: "+name)
  raw=base64.b85decode(row["b85"].encode("ascii"))
  if len(raw)!=row["bytes"] or hashlib.sha256(raw).hexdigest()!=row["sha256"]: raise LaunchError("embedded runtime hash mismatch: "+name)
  decoded[name]=raw
 if PKG.exists(): raise LaunchError("one-trial runtime package path already exists")
 PKG.mkdir(parents=False)
 for name in sorted(decoded):
  with (PKG/name).open("xb") as h: h.write(decoded[name])
  if sha(PKG/name)!=EMBED[name]["sha256"]: raise LaunchError("materialized runtime hash mismatch: "+name)
def verify_pkg():
 p=load(MAN); i=p.get("identity")
 if not isinstance(i,dict) or p.get("identity_sha256")!=hashlib.sha256(canon(i)).hexdigest(): raise LaunchError("package identity hash mismatch")
 if i.get("private") is not True or i.get("internet") is not False or i.get("trials")!=1 or i.get("deadline_seconds")!=3000 or i.get("accelerator")!="T4x2" or i.get("devices")!=["cuda:0","cuda:1"] or i.get("graph_variant")!="a0" or i.get("selected_count")!=22 or i.get("adjacent_pair_count")!=2178 or i.get("runtime_attestation_level")!=LEVEL or i.get("self_contained_code_file") is not True or i.get("embedded_files")!=sorted(EMBED) or i.get("offline_dependency_repair")!="attested_exact_wheel_paths_no_index_no_deps" or i.get("hermetic_child_import_layout")!=list(LAYOUT) or i.get("preserved_environment_keys")!=list(KEEP) or i.get("child_cwd")!="materialized_source_root" or i.get("python_safe_path") is not True or i.get("inherits_pythonpath") is not False: raise LaunchError("package runtime policy mismatch")
 rows=i.get("files")
 if not isinstance(rows,list) or {x.get("path") for x in rows}!={"bh9att.json","bh9src.zip"} or len(rows)!=2: raise LaunchError("runtime manifest file coverage mismatch")
 total=MAN.stat().st_size
 for x in rows:
  pth=PKG/str(x.get("path",""))
  if not pth.is_file() or pth.stat().st_size!=x.get("bytes") or sha(pth)!=x.get("sha256"): raise LaunchError("package member mismatch: "+str(pth))
  total+=pth.stat().st_size
 if total!=i.get("package_bytes") or total>=i.get("budget_bytes",0) or Path(__file__).stat().st_size>=i.get("budget_bytes",0): raise LaunchError("package byte budget mismatch")
 return i
def one_mount(candidates,label):
 found=[Path(x) for x in candidates if Path(x).is_dir()]
 try: allowed=INPUT.resolve(strict=True); resolved=[x.resolve(strict=True) for x in found]
 except OSError as e: raise LaunchError("Kaggle input or "+label+" mount is unresolved") from e
 if any(not under(x,allowed) for x in resolved): raise LaunchError(label+" mount escaped /kaggle/input")
 unique={str(x):x for x in resolved}
 if len(unique)!=1: raise LaunchError("expected one exact "+label+" mount; found="+repr([str(x) for x in found]))
 return next(iter(unique.values()))
def verify_att():
 p=load(ATT); i=p.get("identity")
 if not isinstance(i,dict) or p.get("identity_sha256")!=hashlib.sha256(canon(i)).hexdigest(): raise LaunchError("attachment identity hash mismatch")
 if i.get("mount_candidates")!=list(SM) or i.get("dependencies")!=DEPS or i.get("base_environment_dependencies")!=sorted(BASE): raise LaunchError("attachment policy mismatch")
 root=one_mount(SM,"support")
 for x in i.get("required_files",[]):
  pth=root/str(x.get("path",""))
  if not pth.is_file() or pth.stat().st_size!=x.get("bytes") or sha(pth)!=x.get("sha256"): raise LaunchError("support/checkpoint/source mismatch: "+str(pth))
 wheels=root/str(i.get("wheelhouse",{}).get("path","wheels")); rows=i.get("wheelhouse",{}).get("files",[]); plan=wheel_plan(rows)
 if i.get("dependency_wheels")!=plan or i.get("dependency_wheels_sha256")!=hashlib.sha256(canon(plan)).hexdigest(): raise LaunchError("derived exact-wheel plan attestation mismatch")
 expected_names=sorted(x["path"] for x in rows); actual_names=sorted(x.name for x in wheels.glob("*.whl")) if wheels.is_dir() else []
 if actual_names!=expected_names: raise LaunchError("wheel directory does not exactly match its attested manifest")
 for x in rows:
  pth=wheels/str(x.get("path",""))
  if not pth.is_file() or pth.stat().st_size!=x.get("bytes") or sha(pth)!=x.get("sha256"): raise LaunchError("dependency wheel mismatch: "+str(pth))
 if hashlib.sha256(canon(rows)).hexdigest()!=i.get("wheelhouse",{}).get("manifest_sha256"): raise LaunchError("wheel manifest hash mismatch")
 return root,wheels,plan
def probe(end):
 code="""import importlib,importlib.metadata,json,sys
deps=json.loads(sys.stdin.read()); out={}
for dist,mod,lo,hi,spec in deps:
 try:
  value=importlib.metadata.version(dist); imported=importlib.import_module(mod)
  out[dist]={\"version\":value,\"module_file\":str(getattr(imported,\"__file__\",None)),\"error\":None,\"specifier\":spec}
 except BaseException as e: out[dist]={\"version\":None,\"module_file\":None,\"error\":type(e).__name__+\": \"+str(e),\"specifier\":spec}
print(json.dumps(out,sort_keys=True,separators=(\",\",\":\")))
"""
 run=subprocess.run([str(python_exe()),"-P","-c",code],input=json.dumps(DEPS),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=str(WORK),env=base_env(),timeout=remaining(end,"dependency probe"))
 if run.returncode!=0: raise LaunchError("dependency probe failed: "+run.stderr[-2000:])
 try: out=json.loads(run.stdout)
 except Exception as e: raise LaunchError("dependency probe returned invalid JSON") from e
 for dist,mod,lo,hi,spec in DEPS: out[dist]["compatible"]=out[dist]["error"] is None and compat(out[dist]["version"],lo,hi)
 return out
def verify_selected(wheels,selected):
 root=wheels.resolve(); out=[]
 for row in selected:
  name=row["path"]; pth=(root/name).resolve()
  if pth.parent!=root or pth.name!=name or not pth.is_file() or pth.stat().st_size!=row["bytes"] or sha(pth)!=row["sha256"]: raise LaunchError("selected exact wheel failed immediate basename/size/SHA verification: "+str(pth))
  out.append(dict(row,absolute_path=str(pth)))
 return out
def repair(wheels,plan,end):
 before=probe(end); by={norm(x["distribution"]):x for x in plan}; selected=[]
 for dist,mod,lo,hi,policy in DEPS:
  if before[dist]["compatible"]: continue
  key=norm(dist)
  if key in BASE: raise LaunchError("incompatible base dependency has no attested repair wheel: "+dist)
  selected.append(by[key])
 exact=verify_selected(wheels,selected) if selected else []
 cmd=[]
 if exact:
  cmd=[str(python_exe()),"-m","pip","install","--no-index","--no-deps","--force-reinstall",*[x["absolute_path"] for x in exact]]
  subprocess.run(cmd,check=True,cwd=str(WORK),env=base_env(),timeout=remaining(end,"offline exact-wheel dependency repair"))
 after=probe(end); bad={k:v for k,v in after.items() if not v["compatible"]}
 if bad: raise LaunchError("dependencies remain incompatible after exact-wheel repair: "+repr(bad))
 mismatch={x["distribution"]:{"expected":x["version"],"observed":after[x["distribution"]]["version"]} for x in exact if after[x["distribution"]]["version"]!=x["version"]}
 if mismatch: raise LaunchError("installed versions differ from attested exact wheels: "+repr(mismatch))
 return {"before":before,"attested_wheel_audit":plan,"install_plan":exact,"pip_command":cmd,"after":after,"smoke_checked_modules":[m for d,m,lo,hi,s in DEPS]}
def extract():
 identity=load(MAN)["identity"]["archive"]; expected={x["path"]:x for x in identity["members"]}
 if ROOT.exists(): raise LaunchError("one-trial source path already exists")
 with zipfile.ZipFile(SRC) as z:
  names=[x.filename for x in z.infolist() if not x.is_dir()]
  if sorted(names)!=sorted(expected) or len(names)!=len(set(names)): raise LaunchError("source archive member set mismatch")
  for name in sorted(names):
   if Path(name).is_absolute() or ".." in Path(name).parts: raise LaunchError("unsafe source member")
   raw=z.read(name); row=expected[name]
   if len(raw)!=row["bytes"] or hashlib.sha256(raw).hexdigest()!=row["sha256"]: raise LaunchError("source member hash mismatch: "+name)
   dst=ROOT/name; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(raw)
def diagnostic_call(fn):
 try: return fn()
 except BaseException as exc: return {"error_type":type(exc).__name__,"error_message":str(exc)}
def diagnostic_command(command):
 try:
  run=subprocess.run(command,check=False,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
  return {"command":command,"returncode":int(run.returncode),"stdout":run.stdout[-8000:],"stderr":run.stderr[-4000:]}
 except BaseException as exc:
  return {"command":command,"error_type":type(exc).__name__,"error_message":str(exc)}
def gpu():
 global GPU_DIAG
 torch=importlib.import_module("torch")
 count=int(torch.cuda.device_count()); devices=[]
 for index in range(count):
  name=diagnostic_call(lambda i=index:str(torch.cuda.get_device_name(i)))
  capability=diagnostic_call(lambda i=index:list(torch.cuda.get_device_capability(i)))
  properties=diagnostic_call(lambda i=index:torch.cuda.get_device_properties(i))
  row={"index":index,"name":name,"capability":capability}
  if isinstance(properties,dict): row["properties_error"]=properties
  else:
   row.update({"total_memory_bytes":int(properties.total_memory),"multi_processor_count":int(properties.multi_processor_count)})
  devices.append(row)
 names=[row["name"] if isinstance(row["name"],str) else "<name-error>" for row in devices]
 name_mismatch=any("T4" not in name.upper() for name in names)
 mismatches=[]
 if count!=2: mismatches.append("device_count")
 if name_mismatch: mismatches.append("device_name")
 if count==0: classification="no_cuda_devices"
 elif name_mismatch: classification="unexpected_accelerator_family"
 elif count!=2: classification="honored_t4_wrong_device_count"
 else: classification="exact_t4x2"
 cudnn=getattr(getattr(torch,"backends",None),"cudnn",None)
 GPU_DIAG={
  "requested":{"kaggle_machine_shape":KSHAPE,"logical_accelerator":"T4x2","expected_device_count":2,"expected_name_contains":"T4","devices":["cuda:0","cuda:1"]},
  "observed":{"classification":classification,"mismatches":mismatches,"device_count":count,"devices":devices,"torch_version":str(getattr(torch,"__version__",None)),"torch_cuda_version":str(getattr(getattr(torch,"version",None),"cuda",None)),"torch_cuda_available":diagnostic_call(lambda:bool(torch.cuda.is_available())),"torch_compiled_arch_list":diagnostic_call(lambda:list(torch.cuda.get_arch_list())),"cudnn_version":diagnostic_call(lambda:cudnn.version()) if cudnn is not None else None},
  "environment":{key:os.environ.get(key) for key in GPU_ENV_KEYS},
  "nvidia_smi":{"inventory":diagnostic_command(["nvidia-smi","-L"]),"query":diagnostic_command(["nvidia-smi","--query-gpu=index,name,compute_cap,memory.total,driver_version","--format=csv,noheader,nounits"])},
 }
 if mismatches: raise LaunchError("exact T4x2 required; requested_shape=%s classification=%s mismatches=%r count=%s names=%r"%(KSHAPE,classification,mismatches,count,names))
 return names
def verify_ok(policy):
 p=load(OK); gates=p.get("gates"); rt=p.get("runtime_attestation"); score=p.get("official_score",{}).get("summary",{}).get("score"); selection=p.get("selection",{}); runner=p.get("runner",{}); audit=runner.get("a0_audit",{}) if isinstance(runner,dict) else {}; validation=runner.get("a0_validation",{}) if isinstance(runner,dict) else {}
 a0=("graph_variant_a0","exactly_two_a0_scorer_instances","exact_22_video_a0_audit","exact_2178_adjacent_pairs","all_cache_contexts_covered","finite_unique_original_coo_probabilities","full_frame_dense_matrix_false","balanced_attention_fallback_accounting","checkpoint_weight_identity")
 required=("dual_gpu","no_dense_nxn","complete_score_coverage","graph_validity","deadline_met","score_strictly_above_0_90",*a0)
 if p.get("passed") is not True or not isinstance(gates,dict) or any(gates.get(x) is not True for x in required): raise LaunchError("success evidence gates are incomplete")
 if not isinstance(rt,dict) or rt.get("level")!=LEVEL or rt.get("passed") is not True or rt.get("maximum_seconds")!=DEADLINE: raise LaunchError("level-6 runtime attestation is invalid")
 if not isinstance(score,(int,float)) or not float(score)>float(policy["minimum_score"]): raise LaunchError("official score is not strictly above 0.90")
 if selection.get("selected_count")!=22 or selection.get("selected_compressed_bytes")!=9337408312: raise LaunchError("success evidence selection differs from fixed 22-video 9GB manifest")
 if not isinstance(runner,dict) or runner.get("graph_variant")!="a0" or not isinstance(validation,dict) or any(validation.get(x) is not True for x in a0): raise LaunchError("success evidence A0 validation is incomplete")
 if not isinstance(audit,dict) or audit.get("scope")!="run" or audit.get("video_count")!=22 or audit.get("frame_count")!=2200 or audit.get("pair_count")!=2178 or audit.get("workers")!={"worker_count":2,"scorer_instances":2}: raise LaunchError("success evidence A0 run audit coverage is invalid")
 if audit.get("gate",{}).get("full_frame_dense_matrix") is not False or audit.get("probabilities",{}).get("finite_unit_interval") is not True or audit.get("probabilities",{}).get("candidate_count")!=audit.get("probabilities",{}).get("unique_pair_count"): raise LaunchError("success evidence A0 sparse probability audit is invalid")
 identity=runner.get("checkpoint_weight_identity",{})
 if not isinstance(identity,dict) or identity.get("matched") is not True or len({identity.get("preflight_weight_sha256"),identity.get("frozen_cache_weight_sha256"),identity.get("a0_audit_weight_sha256")})!=1: raise LaunchError("success evidence checkpoint weight identity is invalid")
 return p
def fail(exc,stage,started):
 if FAIL.exists(): return
 data={"schema":"celltrack-alpha.bh9-launch.v1","status":"failed_closed","passed":False,"stage":stage,"elapsed_seconds":time.monotonic()-started,"error":{"type":type(exc).__name__,"message":str(exc)},"diagnostics":{"requested_machine_shape":KSHAPE,"gpu_attestation":GPU_DIAG,"runtime":{"python_version":sys.version,"python_executable":sys.executable,"cwd":str(Path.cwd())}},"traceback":traceback.format_exc()[-16000:],"success_evidence":str(OK)}
 try:
  with FAIL.open("x",encoding="ascii") as h: h.write(canon(data).decode("ascii")+"\n")
 except FileExistsError: pass
def main():
 started=time.monotonic(); end=started+DEADLINE; stage="embedded_materialization"
 def alarm(_s,_f): raise LaunchError("hard 50-minute notebook deadline reached")
 signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,DEADLINE)
 try:
  if OK.exists() or FAIL.exists(): raise LaunchError("one-trial contract refuses existing bh9 evidence")
  materialize(); stage="package_attestation"; policy=verify_pkg(); stage="attachment_attestation"; support,wheels,plan=verify_att()
  stage="dependency_attestation"; deps=repair(wheels,plan,end)
  stage="source_attestation"; extract(); paths=child_paths(support); env=child_env(paths)
  stage="required_import_attestation"; imports=probe_required(paths,env,end)
  stage="gpu_attestation"; names=gpu()
  stage="competition_discovery"; competition=one_mount(CM,"competition")
  stage="bh9gb"
  script=safe_file(ROOT/"scripts/bh9gb.py",ROOT,"bh9gb child script")
  for pth,label in ((ROOT/"bh9reg.json","registry"),(ROOT/"bh9sp.json","split"),(ROOT/"bh9m.json","frozen manifest")): safe_file(pth,ROOT,label)
  safe_dir(competition,competition,"competition mount"); safe_dir(support,support,"support mount"); safe_dir(OUT.parent,WORK,"child output parent")
  cmd=[str(python_exe()),"-P",str(script),"--archive",str(competition),"--registry",str(ROOT/"bh9reg.json"),"--split",str(ROOT/"bh9sp.json"),"--manifest",str(ROOT/"bh9m.json"),"--support",str(support),"--out-dir",str(OUT),"--evidence",str(OK),"--failure",str(FAIL)]
  run=subprocess.run(cmd,check=False,cwd=str(ROOT),env=env,timeout=remaining(end,"bh9gb completion"))
  if run.returncode!=0: raise LaunchError("bh9gb exited %s"%run.returncode)
  if not OK.is_file() or FAIL.exists() or not (OUT/"run.json").is_file() or not (OUT/"fragments").is_dir(): raise LaunchError("bh9gb evidence set is incomplete")
  stage="level_6_attestation"; evidence=verify_ok(policy)
  print(json.dumps({"evidence":str(OK),"run":str(OUT/"run.json"),"fragments":str(OUT/"fragments"),"gpus":names,"gpu_attestation":GPU_DIAG,"dependencies":deps,"required_imports":imports,"runtime_attestation":evidence["runtime_attestation"]},sort_keys=True))
  return 0
 except BaseException as exc:
  fail(exc,stage,started); traceback.print_exc(file=sys.stderr); return 2
 finally: signal.setitimer(signal.ITIMER_REAL,0)
if __name__=="__main__": raise SystemExit(main())
'''


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
        f"{row['path']}: expected {expected_by_path[row['path']]}, got {row['sha256']}"
        for row in required
        if row["path"] in expected_by_path and row["sha256"] != expected_by_path[row["path"]]
    ]
    source_sha = support_source_tree_sha256(support_root)
    if source_sha != expected_hashes["source_sha256"]:
        mismatches.append(f"support source tree: expected {expected_hashes['source_sha256']}, got {source_sha}")
    if mismatches:
        raise PackageError("support/checkpoint SHA-256 validation failed: " + "; ".join(mismatches))
    wheel_dir = support_root / "wheels"
    wheels = [file_record(path, relative_to=wheel_dir) for path in sorted(wheel_dir.glob("*.whl"))]
    if not wheels:
        raise PackageError(f"offline support wheelhouse is empty: {wheel_dir}")
    dependency_wheels = dependency_wheel_plan(wheels)
    identity = {
        "dataset_ref": SUPPORT_REF,
        "dataset_slug": SUPPORT_DATASET_SLUG,
        "mount_candidates": list(support_mounts),
        "required_files": required,
        "source_tree": {"paths": list(SUPPORT_SOURCE_PATHS), "sha256": source_sha},
        "checkpoint": next(row for row in required if row["path"].endswith("edge_predictor_best.pth")),
        "dependencies": [list(row) for row in DEPENDENCIES],
        "base_environment_dependencies": sorted(BASE_ENVIRONMENT_DISTRIBUTIONS),
        "dependency_wheels": dependency_wheels,
        "dependency_wheels_sha256": sha256_bytes(canonical_json(dependency_wheels)),
        "wheelhouse": {
            "path": "wheels", "files": wheels, "file_count": len(wheels),
            "bytes": sum(row["bytes"] for row in wheels),
            "manifest_sha256": sha256_bytes(canonical_json(wheels)),
            "included_in_kernel_source": False,
        },
    }
    payload = {
        "schema": "bh9-kaggle-attachment.v1", "identity": identity,
        "identity_sha256": sha256_bytes(canonical_json(identity)),
    }
    return canonical_json(payload) + b"\n"


def _archive_identity(members: Mapping[str, bytes], archive: bytes) -> dict[str, Any]:
    return {
        "path": ARCHIVE_NAME, "bytes": len(archive), "sha256": sha256_bytes(archive),
        "uncompressed_bytes": sum(len(value) for value in members.values()),
        "members": [
            {"path": name, "bytes": len(members[name]), "sha256": sha256_bytes(members[name])}
            for name in sorted(members)
        ],
    }


def _metadata_payload() -> bytes:
    value = {
        "id": KERNEL_REF, "title": KERNEL_SLUG, "code_file": ENTRYPOINT_NAME,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "machine_shape": KAGGLE_ACCELERATOR_ID,
        "competition_sources": [COMPETITION_SLUG], "dataset_sources": [SUPPORT_REF],
    }
    return json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _manifest_payload(files: Mapping[str, bytes], archive_identity: Mapping[str, Any]) -> bytes:
    records = [
        {"path": name, "bytes": len(value), "sha256": sha256_bytes(value)}
        for name, value in sorted(files.items())
    ]
    package_bytes = 0
    while True:
        identity = {
            "private": True, "internet": False, "trials": 1,
            "deadline_seconds": bh9gb.DEADLINE_SECONDS, "accelerator": "T4x2",
            "devices": list(bh9gb.DEVICES), "minimum_score": bh9gb.MIN_SCORE,
            "runtime_attestation_level": RUNTIME_ATTESTATION_LEVEL,
            "self_contained_code_file": True,
            "embedded_files": sorted((MANIFEST_NAME, ATTACHMENT_NAME, ARCHIVE_NAME)),
            "seed": bh9gb.SEED, "selected_count": bh9gb.FROZEN_VIDEO_COUNT,
            "graph_variant": bh9gb.run_bh.A0_GRAPH_VARIANT,
            "adjacent_pair_count": bh9gb.FROZEN_ADJACENT_PAIR_COUNT,
            "selected_compressed_bytes": 9_337_408_312, "sparse_graph": True,
            "package_bytes": package_bytes, "budget_bytes": SOURCE_LIMIT,
            "files": records, "archive": dict(archive_identity),
            "competition_mounts": list(COMPETITION_MOUNTS),
            "support_mounts": list(SUPPORT_MOUNTS),
            "offline_dependency_repair": "attested_exact_wheel_paths_no_index_no_deps",
            "hermetic_child_import_layout": list(CHILD_IMPORT_LAYOUT),
            "preserved_environment_keys": list(PRESERVED_ENVIRONMENT_KEYS),
            "child_cwd": "materialized_source_root",
            "python_safe_path": True,
            "inherits_pythonpath": False,
        }
        payload = {
            "schema": "bh9-kaggle-package.v1", "identity": identity,
            "identity_sha256": sha256_bytes(canonical_json(identity)),
        }
        encoded = canonical_json(payload) + b"\n"
        observed = sum(len(value) for value in files.values()) + len(encoded)
        if observed == package_bytes:
            return encoded
        package_bytes = observed


def _entrypoint_payload(files: Mapping[str, bytes]) -> bytes:
    expected = {MANIFEST_NAME, ATTACHMENT_NAME, ARCHIVE_NAME}
    if set(files) != expected:
        raise PackageError(
            f"entrypoint embedding requires {sorted(expected)}, got {sorted(files)}"
        )
    envelope = {
        name: {
            "b85": base64.b85encode(files[name]).decode("ascii"),
            "bytes": len(files[name]),
            "sha256": sha256_bytes(files[name]),
        }
        for name in sorted(files)
    }
    encoded = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    marker = '"__BH9_EMBED__"'
    dependency_marker = '"__BH9_DEPS__"'
    accelerator_marker = '"__BH9_SHAPE__"'
    if (
        ENTRYPOINT_SOURCE.count(marker) != 1
        or ENTRYPOINT_SOURCE.count(dependency_marker) != 1
        or ENTRYPOINT_SOURCE.count(accelerator_marker) != 1
    ):
        raise PackageError("entrypoint embedding marker is missing or ambiguous")
    encoded_dependencies = canonical_json([list(row) for row in DEPENDENCIES]).decode("ascii")
    source = (
        ENTRYPOINT_SOURCE.replace(marker, json.dumps(encoded))
        .replace(dependency_marker, json.dumps(encoded_dependencies))
        .replace(accelerator_marker, json.dumps(KAGGLE_ACCELERATOR_ID))
        .encode("ascii")
    )
    if len(source) >= SOURCE_LIMIT:
        raise PackageError(
            f"self-contained bh9.py must be <{SOURCE_LIMIT} bytes, got {len(source)}"
        )
    compile(source, ENTRYPOINT_NAME, "exec")
    return source


def _validate_mounts(values: Sequence[str], slug: str, label: str) -> None:
    if (
        not values or len(set(values)) != len(values)
        or any(not Path(value).is_absolute() or Path(value).name != slug for value in values)
    ):
        raise PackageError(f"{label} mount candidates must be unique exact absolute slug paths")


def build_package(
    repo_root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    archive: Path | None = None,
    registry: Path | None = None,
    split: Path | None = None,
    support_root: Path | None = None,
    expected_artifact_hashes: Mapping[str, str] = EXPECTED_ARTIFACT_HASHES,
    expected_train_count: int = bh9gb.TRAIN_COUNT,
    competition_mounts: Sequence[str] = COMPETITION_MOUNTS,
    support_mounts: Sequence[str] = SUPPORT_MOUNTS,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(); output_dir = output_dir.resolve()
    archive = (archive or DEFAULT_ARCHIVE).resolve()
    registry = (registry or (repo_root / "reports/g02/dataset_registry_v1.json")).resolve()
    split = (split or (repo_root / "reports/g02/grouped_3fold_audit20_v1.json")).resolve()
    support_root = (support_root or (repo_root / "external/Tracking_Support")).resolve()
    if _SAFE_SLUG.fullmatch(KERNEL_SLUG) is None or len(KERNEL_SLUG) >= 15:
        raise PackageError("kernel slug must be lowercase and under 15 characters")
    for name in (ENTRYPOINT_NAME, ARCHIVE_NAME, ATTACHMENT_NAME, MANIFEST_NAME, output_dir.name):
        if not name or len(name) >= 15:
            raise PackageError(f"created basename must be under 15 characters: {name!r}")
    _validate_mounts(competition_mounts, COMPETITION_SLUG, "competition")
    _validate_mounts(support_mounts, SUPPORT_DATASET_SLUG, "support")
    if set(expected_artifact_hashes) != set(EXPECTED_ARTIFACT_HASHES):
        raise PackageError("expected support hash mapping has missing or unexpected fields")
    members, frozen = _archive_members(
        repo_root, archive, registry, split, expected_train_count=expected_train_count
    )
    source_archive = deterministic_zip(members)
    archive_identity = _archive_identity(members, source_archive)
    attachment = _attachment_payload(
        support_root, expected_artifact_hashes, support_mounts=support_mounts
    )
    runtime_files = {
        ARCHIVE_NAME: source_archive,
        ATTACHMENT_NAME: attachment,
    }
    manifest = _manifest_payload(runtime_files, archive_identity)
    files = {
        ENTRYPOINT_NAME: _entrypoint_payload({
            MANIFEST_NAME: manifest,
            ATTACHMENT_NAME: attachment,
            ARCHIVE_NAME: source_archive,
        }),
        **runtime_files,
        METADATA_NAME: _metadata_payload(),
        MANIFEST_NAME: manifest,
    }
    aggregate = sum(len(value) for value in files.values())
    if aggregate >= SOURCE_LIMIT:
        raise PackageError(f"aggregate Kaggle kernel source must be <{SOURCE_LIMIT} bytes, got {aggregate}")
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in files)
    if unexpected:
        raise PackageError(f"refusing to mix bh9 package with unexpected files: {unexpected}")
    for index, (name, value) in enumerate(files.items()):
        temporary = output_dir / f"tmp{index}"
        temporary.write_bytes(value)
        temporary.replace(output_dir / name)
    records = [file_record(output_dir / name, relative_to=output_dir) for name in sorted(files)]
    return {
        "schema": "bh9-package-build.v1", "output_dir": str(output_dir),
        "aggregate_bytes": aggregate, "budget_bytes": SOURCE_LIMIT,
        "strictly_under_budget": True, "file_count": len(records), "files": records,
        "package_sha256": sha256_bytes(canonical_json(records)),
        "package_manifest_sha256": sha256_file(output_dir / MANIFEST_NAME),
        "source_archive_sha256": sha256_file(output_dir / ARCHIVE_NAME),
        "selection": frozen["selection"], "network_or_kaggle_actions": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    args = parser.parse_args(argv)
    report = build_package(
        ROOT, args.output, archive=args.archive, registry=args.registry,
        split=args.split, support_root=args.support_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
