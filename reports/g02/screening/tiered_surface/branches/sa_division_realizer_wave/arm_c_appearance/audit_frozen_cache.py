#!/usr/bin/env python3
"""Authenticate existing Arm-C caches without reading targets or raw images."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


MAIN_ROOT = Path("/Users/agent/Desktop/cell_tracking")
OUTPUT_DIR = Path(__file__).resolve().parent
SHARED_CONTRACT = (
    MAIN_ROOT
    / "reports/g02/screening/tiered_surface/branches/"
    "sa_division_realizer_wave/SHARED_CONTRACT.json"
)
SHARED_CONTRACT_SHA256 = (
    "08f3e875e77a5c7f93c56ce779df93f4cbbed1bb8a9a7f183369452c9e29e647"
)
FRAGMENT_ROOT = MAIN_ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
SPARSE_CACHE_ROOT = MAIN_ROOT / "kaggle_kernel/bhcv57out_v3/cache"
H1_MANIFEST = (
    MAIN_ROOT / "reports/g02/screening/h1_existing_cache_feature_report.json"
)
H1_MANIFEST_SHA256 = (
    "d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf"
)
PARITY3_ROOT = (
    MAIN_ROOT
    / "_ARCHIVE/conflict_residual/reports/g02/"
    "run_bh_conflict_edge_f0_parity3_v1"
)
SPARSE_RECORD_SET_SHA256 = (
    "9357342bed7ba13584bc71f71e715759775466d1f1c478c5415c3833079728a2"
)
EXPECTED_44B6 = 21
EXPECTED_6BBA_PRESENT_BUT_FORBIDDEN = 36
EXPECTED_H1_44B6 = 36
EXPECTED_H1_HELD_OVERLAP = 0
EXPECTED_PARITY3_HELD_OVERLAP = 3

INTENSITY_OBSERVABLE_NAMES = frozenset(
    {
        "raw_sum",
        "excess_sum",
        "mean_intensity",
        "intensity_variance",
        "relative_intensity_variance",
        "intensity_patch",
        "voxel_patch",
        "physical_intensity_mass",
    }
)


class CacheAuditError(RuntimeError):
    """An immutable input or cache-audit invariant changed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CacheAuditError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def dataclass_field_names(value: Any) -> list[str]:
    if not is_dataclass(value):
        raise CacheAuditError(f"expected dataclass payload, got {type(value)!r}")
    return [field.name for field in fields(value)]


def held_identity(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("train__44b6_"):
        raise CacheAuditError(f"non-44b6 path escaped the family filter: {path}")
    return "train:" + stem.removeprefix("train__")


def inspect_sparse_caches(held: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sys.path.insert(0, str(MAIN_ROOT / "src"))
    from celltrack_alpha import bh_model  # noqa: PLC0415

    records: list[dict[str, Any]] = []
    observed_payload_fields: set[str] = set()
    observed_detection_fields: set[str] = set()
    observed_context_fields: set[str] = set()
    intensity_fields: set[str] = set()
    for identity in sorted(held):
        stem = identity.replace(":", "__", 1)
        metadata_path = SPARSE_CACHE_ROOT / f"{stem}.json"
        cache_path = SPARSE_CACHE_ROOT / f"{stem}.bhc"
        metadata = load_json(metadata_path)
        observed_cache_sha = sha256(cache_path)
        if observed_cache_sha != metadata.get("cache_sha256"):
            raise CacheAuditError(f"sparse-cache SHA mismatch: {cache_path}")
        try:
            model = metadata["contract"]["model"]
            expected_identity = bh_model.CacheIdentity(
                config_sha256=str(model["config_sha256"]),
                source_sha256=str(model["source_sha256"]),
                weight_sha256=str(model["weight_sha256"]),
                pair_batch_size=int(model["pair_batch_size"]),
                view_batch_size=int(model["view_batch_size"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheAuditError(f"invalid cache identity: {metadata_path}") from exc
        payload = bh_model.load_sparse_cache(
            cache_path, expected_identity=expected_identity
        )
        payload_fields = dataclass_field_names(payload)
        detection_fields = (
            dataclass_field_names(payload.detections[0])
            if payload.detections
            else []
        )
        context_fields = (
            dataclass_field_names(payload.contexts[0])
            if payload.contexts
            else []
        )
        observed_payload_fields.update(payload_fields)
        observed_detection_fields.update(detection_fields)
        observed_context_fields.update(context_fields)
        all_fields = set(payload_fields) | set(detection_fields) | set(context_fields)
        intensity_fields.update(all_fields & INTENSITY_OBSERVABLE_NAMES)
        records.append(
            {
                "identity": identity,
                "cache_path": str(cache_path.relative_to(MAIN_ROOT)),
                "cache_sha256": observed_cache_sha,
                "cache_bytes": cache_path.stat().st_size,
                "metadata_path": str(metadata_path.relative_to(MAIN_ROOT)),
                "metadata_sha256": sha256(metadata_path),
            }
        )
        del payload
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    record_set_sha = hashlib.sha256(canonical).hexdigest()
    if record_set_sha != SPARSE_RECORD_SET_SHA256:
        raise CacheAuditError(
            f"sparse record-set SHA drifted: {record_set_sha}"
        )
    schema = {
        "payload_fields": sorted(observed_payload_fields),
        "detection_fields": sorted(observed_detection_fields),
        "context_fields": sorted(observed_context_fields),
        "physical_intensity_fields": sorted(intensity_fields),
        "latent_feature_note": (
            "source_feat and target_feat are learned 32-D association latents; "
            "the preregistration forbids substituting them for additive physical "
            "intensity mass or voxel summaries."
        ),
    }
    return records, schema


def main() -> None:
    if sha256(SHARED_CONTRACT) != SHARED_CONTRACT_SHA256:
        raise CacheAuditError("shared-contract SHA mismatch")
    all_fragments = sorted(FRAGMENT_ROOT.glob("train__*.csv"))
    paths_44b6 = [
        path for path in all_fragments if path.name.startswith("train__44b6_")
    ]
    paths_6bba = [
        path for path in all_fragments if path.name.startswith("train__6bba_")
    ]
    if (
        len(paths_44b6) != EXPECTED_44B6
        or len(paths_6bba) != EXPECTED_6BBA_PRESENT_BUT_FORBIDDEN
        or len(all_fragments) != EXPECTED_44B6 + EXPECTED_6BBA_PRESENT_BUT_FORBIDDEN
    ):
        raise CacheAuditError("frozen 21/36 family inventory drifted")
    held = {held_identity(path) for path in paths_44b6}
    if len(held) != EXPECTED_44B6:
        raise CacheAuditError("held 44b6 identity collision")
    selected_paths = [str(path.relative_to(MAIN_ROOT)) for path in paths_44b6]
    if any("6bba" in path for path in selected_paths):
        raise CacheAuditError("6bba path entered the selected audit")

    sparse_records, sparse_schema = inspect_sparse_caches(held)

    if sha256(H1_MANIFEST) != H1_MANIFEST_SHA256:
        raise CacheAuditError("H1 feature-cache manifest SHA mismatch")
    h1 = load_json(H1_MANIFEST)
    h1_44b6 = {
        str(row["identity"])
        for row in h1.get("files", [])
        if ":44b6_" in str(row.get("identity", ""))
    }
    h1_overlap = sorted(held & h1_44b6)
    if (
        len(h1_44b6) != EXPECTED_H1_44B6
        or len(h1_overlap) != EXPECTED_H1_HELD_OVERLAP
    ):
        raise CacheAuditError("H1 fit105/held identity boundary drifted")

    parity_paths = sorted((PARITY3_ROOT / "cache").glob("train__44b6_*.bhc"))
    parity_identities = {
        "train:" + path.stem.removeprefix("train__") for path in parity_paths
    }
    parity_overlap = sorted(held & parity_identities)
    if len(parity_overlap) != EXPECTED_PARITY3_HELD_OVERLAP:
        raise CacheAuditError("archived parity3 coverage drifted")

    physical_intensity_adequate = bool(
        sparse_schema["physical_intensity_fields"]
    )
    verdict = (
        "PASS_ADEQUATE_FROZEN_INTENSITY_CACHE"
        if physical_intensity_adequate
        else "FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE"
    )
    result = {
        "schema": "celltrack-alpha.division-arm-c-cache-audit.v1",
        "arm": "C",
        "verdict": verdict,
        "scope": {
            "selected_family": "44b6",
            "selected_fragment_count": len(paths_44b6),
            "selected_paths": selected_paths,
            "forbidden_6bba_paths_read": 0,
            "6bba_paths_present_but_not_read": len(paths_6bba),
            "inventory_correction": "21 44b6 plus 36 6bba",
        },
        "candidates": {
            "h1_full_pool_feature_cache": {
                "manifest_path": str(H1_MANIFEST.relative_to(MAIN_ROOT)),
                "manifest_sha256": sha256(H1_MANIFEST),
                "declared_44b6": len(h1_44b6),
                "held_44b6_overlap": len(h1_overlap),
                "overlap_identities": h1_overlap,
                "adequate": False,
                "reason": "zero held-44b6 identity coverage",
            },
            "held_sparse_v45_cache": {
                "root": str(SPARSE_CACHE_ROOT.relative_to(MAIN_ROOT)),
                "record_count": len(sparse_records),
                "record_set_sha256": SPARSE_RECORD_SET_SHA256,
                "bytes": sum(record["cache_bytes"] for record in sparse_records),
                "schema": sparse_schema,
                "adequate": physical_intensity_adequate,
                "reason": (
                    "no physical intensity mass, moment, or voxel-patch fields"
                    if not physical_intensity_adequate
                    else None
                ),
                "records": sparse_records,
            },
            "archived_parity3_cache": {
                "root": str(PARITY3_ROOT.relative_to(MAIN_ROOT)),
                "held_44b6_overlap": len(parity_overlap),
                "overlap_identities": parity_overlap,
                "adequate": False,
                "reason": "partial 3/21 coverage",
            },
        },
        "prediction_stage": {
            "started": False,
            "predictions_written": False,
            "target_read": False,
            "oracle_read": False,
            "repair_bundle_read": False,
            "official_scorer_read": False,
            "raw_archive_read": False,
            "raw_image_read": False,
            "cache_created_or_extended": False,
        },
        "stop_rule": (
            "No existing cache simultaneously covers all 21 held-44b6 identities "
            "and exposes physical intensity redistribution summaries. Stop before "
            "prediction and target evaluation."
        ),
    }
    atomic_json(OUTPUT_DIR / "CACHE_AUDIT.json", result)
    print(verdict)


if __name__ == "__main__":
    main()
