#!/usr/bin/env python3
"""Audit/package H1 Trackastra inputs and classify Option-C device parity."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREPARE = (
    ROOT
    / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
CACHE_ROOT = ROOT / "reports/g02/fitcache_117/bh9out/cache"
PACKAGE_ROOT = ROOT / "kaggle_kernel/trackastra-h1-feature-cache-prep"
DEFAULT_MANIFEST = PACKAGE_ROOT / "cache_dataset_manifest.json"
DEFAULT_DATASET_METADATA = PACKAGE_ROOT / "cache_dataset_metadata.json"
DEFAULT_RAW_MANIFEST = PACKAGE_ROOT / "raw_fit105_directory_manifest.json"
ASSETS_MANIFEST = PACKAGE_ROOT / "assets_dataset_manifest.json"
ASSETS_DATASET_METADATA = PACKAGE_ROOT / "assets_dataset_metadata.json"
SOURCE_DEPENDENCY_MANIFEST = (
    PACKAGE_ROOT / "source_dependency_manifest.json"
)
RAW_ARCHIVE = (
    ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
)
LEGACY_6BBA_RAW_MANIFEST = (
    ROOT / "reports/g02/trackastra_6bba_raw_directory_manifest_v1.json"
)
FAMILIES = ("44b6", "6bba")
FAMILY_COUNTS = {"44b6": 36, "6bba": 69}
FEATURE_NAMES = tuple(
    [f"apncc_context_{index:02d}" for index in range(48)]
    + ["trackastra_margin"]
)
EXACT_FIELDS = (
    "challengers",
    "conflicts",
    "proposal_offsets",
    "transition_offsets",
    "transition_target_t",
    "geometry_confidences",
    "geometry_probabilities",
)
DEFAULT_ATOL = 1.0e-7
DEFAULT_RTOL = 1.0e-5


class H1PrepError(RuntimeError):
    """A fit-scope, package, parity, or approval invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H1PrepError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise H1PrepError(f"expected JSON object: {path}")
    return value


def fit_identities(
    prepare: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scope = prepare.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise H1PrepError("fit prepare does not fail closed")
    train = tuple(str(value) for value in scope.get("train_video_ids", ()))
    validation = tuple(
        str(value) for value in scope.get("validation_video_ids", ())
    )
    if len(train) != 105 or len(set(train)) != 105:
        raise H1PrepError("fit-train105 identity count drifted")
    counts = Counter(
        identity.split(":", 1)[1].split("_", 1)[0] for identity in train
    )
    if dict(counts) != FAMILY_COUNTS:
        raise H1PrepError(f"fit family counts drifted: {dict(counts)}")
    if set(train) & set(validation):
        raise H1PrepError("validation12 intersects fit-train105")
    if any(
        not any(identity.startswith(f"train:{family}_") for family in FAMILIES)
        for identity in (*train, *validation)
    ):
        raise H1PrepError("unexpected family or split identity")
    return train, validation


def cache_records(
    identities: Sequence[str], cache_root: Path = CACHE_ROOT
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for identity in identities:
        slug = identity.replace(":", "__", 1)
        for suffix in (".bhc", ".json"):
            path = cache_root / f"{slug}{suffix}"
            if not path.is_file():
                raise H1PrepError(f"required fit cache is missing: {path}")
            records.append(
                {
                    "identity": identity,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def build_cache_manifest(
    *,
    prepare_path: Path = PREPARE,
    cache_root: Path = CACHE_ROOT,
) -> dict[str, Any]:
    prepare = load_json(prepare_path)
    train, validation = fit_identities(prepare)
    records = cache_records(train, cache_root)
    return {
        "schema": "celltrack-alpha.trackastra-h1-fit105-cache-dataset.v1",
        "status": "LOCAL_SOURCE_MANIFEST_VERIFIED_REMOTE_DATASET_NOT_CREATED",
        "scope": {
            "fit_train105_only": True,
            "families": list(FAMILIES),
            "family_counts": FAMILY_COUNTS,
            "video_count": len(train),
            "validation12_access": False,
            "cv57_access": False,
            "training": False,
        },
        "prepare": {
            "path": str(prepare_path.relative_to(ROOT)),
            "sha256": sha256_file(prepare_path),
        },
        "identities": list(train),
        "excluded_validation12_sha256": hashlib.sha256(
            ("\n".join(sorted(validation)) + "\n").encode("ascii")
        ).hexdigest(),
        "files": records,
        "aggregate": {
            "file_count": len(records),
            "bytes": sum(int(row["bytes"]) for row in records),
        },
        "remote_dataset": {
            "slug": "onebitbrain/trackastra-h1-fit105-cache-v1",
            "created": False,
            "version": None,
        },
    }


def build_raw_manifest(
    *,
    archive_path: Path = RAW_ARCHIVE,
    legacy_6bba_path: Path = LEGACY_6BBA_RAW_MANIFEST,
) -> dict[str, Any]:
    train, validation = fit_identities(load_json(PREPARE))
    legacy = load_json(legacy_6bba_path)
    records = [
        dict(row)
        for row in legacy.get("files", ())
        if isinstance(row, Mapping)
        and any(
            str(row.get("path", "")).startswith(
                f"train/{identity.split(':', 1)[1]}.zarr/"
            )
            for identity in train
            if ":6bba_" in identity
        )
    ]
    six_prefixes = {
        f"train/{identity.split(':', 1)[1]}.zarr/"
        for identity in train
        if ":6bba_" in identity
    }
    if (
        not records
        or {
            str(row["path"]).split(".zarr/", 1)[0] + ".zarr/"
            for row in records
        }
        != six_prefixes
    ):
        raise H1PrepError("banked 6bba raw manifest coverage drifted")
    prefixes_44b6 = tuple(
        f"train/{identity.split(':', 1)[1]}.zarr/"
        for identity in train
        if ":44b6_" in identity
    )
    if not archive_path.is_file():
        raise H1PrepError(f"raw archive is missing: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        selected = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.startswith(prefixes_44b6)
        ]
        observed_prefixes = {
            member.filename.split(".zarr/", 1)[0] + ".zarr/"
            for member in selected
        }
        if observed_prefixes != set(prefixes_44b6):
            raise H1PrepError("44b6 raw archive coverage is incomplete")
        for ordinal, member in enumerate(selected, start=1):
            digest = hashlib.sha256()
            with archive.open(member) as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            records.append(
                {
                    "bytes": member.file_size,
                    "path": member.filename,
                    "sha256": digest.hexdigest(),
                }
            )
            if ordinal % 250 == 0:
                print(
                    f"hashed {ordinal}/{len(selected)} 44b6 raw files",
                    flush=True,
                )
    records.sort(key=lambda row: str(row["path"]))
    validation_prefixes = tuple(
        f"train/{identity.split(':', 1)[1]}.zarr/"
        for identity in validation
    )
    if any(
        str(row["path"]).startswith(validation_prefixes)
        for row in records
    ):
        raise H1PrepError("raw manifest includes validation12")
    return {
        "schema": "celltrack-alpha.trackastra-h1-raw-directory-manifest.v1",
        "scope": {
            "fit_train105_only": True,
            "families": list(FAMILIES),
            "family_counts": FAMILY_COUNTS,
            "video_count": 105,
            "identities": list(train),
            "validation12_access": False,
            "cv57_access": False,
        },
        "source_archive": {
            "path": str(archive_path.relative_to(ROOT)),
            "bytes": archive_path.stat().st_size,
            "sha256": (
                "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
            ),
        },
        "files": records,
        "aggregate": {
            "file_count": len(records),
            "bytes": sum(int(row["bytes"]) for row in records),
        },
    }


def _same_structure(
    reference: np.ndarray, candidate: np.ndarray
) -> str | None:
    if reference.shape != candidate.shape:
        return "shape mismatch"
    if reference.dtype != candidate.dtype:
        return "dtype mismatch"
    if reference.dtype.kind != "f":
        return "non-float dtype"
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return "non-finite value"
    return None


def classify_float_column(
    m5: np.ndarray,
    t4: np.ndarray,
    *,
    default_atol: float = DEFAULT_ATOL,
    default_rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    reference = np.asarray(m5)
    candidate = np.asarray(t4)
    structural = _same_structure(reference, candidate)
    if structural is not None or reference.ndim != 1:
        return {
            "classification": "STRUCTURAL_BUG",
            "reason": structural or "column is not one-dimensional",
            "proposed_atol": 0.0,
            "proposed_rtol": 0.0,
        }
    if reference.dtype.kind != "f":
        return {
            "classification": "STRUCTURAL_BUG",
            "reason": "tolerance is forbidden for non-float column",
            "proposed_atol": 0.0,
            "proposed_rtol": 0.0,
        }
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    maximum = float(np.max(np.abs(difference), initial=0.0))
    scale = float(
        max(
            np.max(np.abs(reference.astype(np.float64)), initial=0.0),
            np.max(np.abs(candidate.astype(np.float64)), initial=0.0),
        )
    )
    if maximum == 0.0:
        classification, reason = "BENIGN_FP", "exact"
        proposed_atol, proposed_rtol = 0.0, 0.0
    else:
        median = float(np.median(difference))
        centered = float(
            np.max(np.abs(difference - median), initial=0.0)
        )
        same_sign = float(
            max(np.mean(difference > 0.0), np.mean(difference < 0.0))
        )
        if (
            abs(median) > default_atol
            and same_sign >= 0.99
            and centered <= max(default_atol, abs(median) * 1.0e-3)
        ):
            classification, reason = (
                "STRUCTURAL_BUG",
                "coherent whole-column offset",
            )
            proposed_atol, proposed_rtol = 0.0, 0.0
        elif np.allclose(
            reference,
            candidate,
            atol=default_atol,
            rtol=default_rtol,
            equal_nan=False,
        ):
            classification, reason = (
                "BENIGN_FP",
                "within preregistered float probe",
            )
            proposed_atol = max(default_atol, 2.0 * maximum)
            proposed_rtol = max(
                default_rtol, 2.0 * maximum / max(scale, 1.0e-12)
            )
        else:
            classification, reason = (
                "STRUCTURAL_BUG",
                "outside preregistered float probe",
            )
            proposed_atol, proposed_rtol = 0.0, 0.0
    return {
        "classification": classification,
        "reason": reason,
        "maximum_absolute_difference": maximum,
        "scale": scale,
        "proposed_atol": proposed_atol,
        "proposed_rtol": proposed_rtol,
        "final_contract_status": "REQUIRES_HUMAN_APPROVAL",
    }


def compare_option_c_payloads(
    *,
    m5: Mapping[str, Any],
    t4_first: Mapping[str, Any],
    t4_second: Mapping[str, Any],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    names = tuple(str(value) for value in feature_names)
    if names != FEATURE_NAMES:
        raise H1PrepError("49-D feature identity drifted")
    exact_records = []
    t4_repeatability = True
    cross_device_exact = True
    for key in EXACT_FIELDS:
        if any(
            key not in payload
            for payload in (m5, t4_first, t4_second)
        ):
            raise H1PrepError(f"required exact field is missing: {key}")
        m5_value = np.asarray(m5.get(key))
        t4_a = np.asarray(t4_first.get(key))
        t4_b = np.asarray(t4_second.get(key))
        expected_kinds = (
            "f" if key.startswith("geometry_") else "iu"
        )
        if any(
            value.dtype.kind not in expected_kinds
            for value in (m5_value, t4_a, t4_b)
        ):
            raise H1PrepError(
                f"exact field has unauthorized dtype: {key}"
            )
        repeated = np.array_equal(t4_a, t4_b)
        aligned = np.array_equal(m5_value, t4_a)
        t4_repeatability &= repeated
        cross_device_exact &= aligned
        exact_records.append(
            {
                "field": key,
                "t4_array_equal": repeated,
                "m5_t4_exact": aligned,
            }
        )
    if any(
        "row_features" not in payload
        for payload in (m5, t4_first, t4_second)
    ):
        raise H1PrepError("required row_features field is missing")
    m5_features = np.asarray(m5["row_features"])
    t4_a_features = np.asarray(t4_first["row_features"])
    t4_b_features = np.asarray(t4_second["row_features"])
    if (
        m5_features.ndim != 2
        or m5_features.shape[1] != len(names)
        or t4_a_features.shape != m5_features.shape
        or t4_b_features.shape != m5_features.shape
        or any(
            value.dtype.kind != "f"
            for value in (
                m5_features,
                t4_a_features,
                t4_b_features,
            )
        )
    ):
        raise H1PrepError("row_features are not aligned 49-D arrays")
    feature_repeatability = np.array_equal(
        t4_a_features, t4_b_features
    )
    t4_repeatability &= feature_repeatability
    columns = []
    for index, name in enumerate(names):
        result = classify_float_column(
            m5_features[:, index], t4_a_features[:, index]
        )
        columns.append({"index": index, "name": name, **result})
    structural = [
        row["name"]
        for row in columns
        if row["classification"] == "STRUCTURAL_BUG"
    ]
    if not t4_repeatability:
        verdict = "STOP_T4_NONDETERMINISTIC"
    elif not cross_device_exact:
        verdict = "STOP_EXACT_GEOMETRY_OR_IDENTITY_DRIFT"
    elif structural:
        verdict = "STOP_STRUCTURAL_FLOAT_DRIFT"
    else:
        verdict = "BENIGN_FP_PROPOSAL_REQUIRES_HUMAN_APPROVAL"
    return {
        "schema": "celltrack-alpha.trackastra-option-c-comparison.v1",
        "verdict": verdict,
        "t4_repeatability": {
            "all_arrays_array_equal": t4_repeatability,
            "row_features_array_equal": feature_repeatability,
        },
        "m5_t4": {
            "exact_fields_pass": cross_device_exact,
            "exact_fields": exact_records,
            "columns": columns,
            "structural_columns": structural,
            "tolerance_contract_finalized": False,
            "human_approval_required": True,
        },
    }


def stage_cache_dataset(
    manifest: Mapping[str, Any],
    destination: Path,
) -> None:
    if destination.exists():
        raise H1PrepError(f"refusing to overwrite staging directory: {destination}")
    destination.mkdir(parents=True)
    for record in manifest["files"]:
        source = ROOT / str(record["path"])
        target = destination / str(record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    (destination / "cache_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    shutil.copy2(
        DEFAULT_DATASET_METADATA, destination / "dataset-metadata.json"
    )


def stage_assets_dataset(destination: Path) -> None:
    manifest = load_json(ASSETS_MANIFEST)
    if (
        manifest.get("status")
        != "LOCAL_SOURCE_MANIFEST_COMPLETE_REMOTE_DATASET_NOT_CREATED"
    ):
        raise H1PrepError("Option-C assets manifest is incomplete")
    if destination.exists():
        raise H1PrepError(f"refusing to overwrite staging directory: {destination}")
    destination.mkdir(parents=True)
    package_prefix = (
        "source_overlay/kaggle_kernel/"
        "trackastra-h1-feature-cache-prep/"
    )
    for record in manifest["files"]:
        relative = str(record["path"])
        if relative.startswith("source_overlay/scripts/"):
            source = ROOT / relative.removeprefix("source_overlay/")
        elif relative.startswith(package_prefix):
            source = PACKAGE_ROOT / relative.removeprefix(package_prefix)
        else:
            raise H1PrepError(f"unexpected assets path: {relative}")
        if (
            not source.is_file()
            or sha256_file(source) != str(record["sha256"])
        ):
            raise H1PrepError(f"assets source identity drifted: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dependencies = load_json(SOURCE_DEPENDENCY_MANIFEST)
    if (
        dependencies.get("schema")
        != "celltrack-alpha.trackastra-h1-source-dependencies.v1"
    ):
        raise H1PrepError("source dependency manifest schema drifted")
    for record in dependencies.get("files", ()):
        if record.get("origin") != "workspace":
            continue
        relative = str(record.get("path", ""))
        source = ROOT / relative
        if (
            not source.is_file()
            or sha256_file(source) != str(record.get("sha256", ""))
        ):
            raise H1PrepError(
                f"source dependency identity drifted: {source}"
            )
        target = destination / "source_overlay" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(ASSETS_MANIFEST, destination / "assets_dataset_manifest.json")
    shutil.copy2(
        ASSETS_DATASET_METADATA, destination / "dataset-metadata.json"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    manifest.add_argument("--stage-dir", type=Path)
    stage_cache = subparsers.add_parser("stage-cache")
    stage_cache.add_argument("--stage-dir", type=Path, required=True)
    raw = subparsers.add_parser("raw-manifest")
    raw.add_argument("--output", type=Path, default=DEFAULT_RAW_MANIFEST)
    assets = subparsers.add_parser("stage-assets")
    assets.add_argument("--stage-dir", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--m5", type=Path, required=True)
    compare.add_argument("--t4-first", type=Path, required=True)
    compare.add_argument("--t4-second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise H1PrepError(f"comparison artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "manifest":
        result = build_cache_manifest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        if args.stage_dir is not None:
            stage_cache_dataset(result, args.stage_dir)
    elif args.command == "stage-cache":
        manifest = load_json(DEFAULT_MANIFEST)
        if (
            manifest.get("schema")
            != "celltrack-alpha.trackastra-h1-fit105-cache-dataset.v1"
            or manifest.get("scope", {}).get("fit_train105_only")
            is not True
            or manifest.get("scope", {}).get("video_count") != 105
            or manifest.get("scope", {}).get("family_counts")
            != FAMILY_COUNTS
        ):
            raise H1PrepError(
                "authenticated fit105 cache manifest drifted"
            )
        stage_cache_dataset(manifest, args.stage_dir)
        result = {"verdict": "FIT105_CACHE_DATASET_STAGED"}
    elif args.command == "raw-manifest":
        result = build_raw_manifest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    elif args.command == "stage-assets":
        stage_assets_dataset(args.stage_dir)
        result = {"verdict": "ASSETS_DATASET_STAGED"}
    else:
        result = compare_option_c_payloads(
            m5=_load_npz(args.m5),
            t4_first=_load_npz(args.t4_first),
            t4_second=_load_npz(args.t4_second),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        if result["verdict"].startswith("STOP_"):
            return 3
    print(json.dumps(result.get("aggregate", result.get("verdict")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
