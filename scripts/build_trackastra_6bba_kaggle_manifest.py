#!/usr/bin/env python3
"""Build the pinned input manifest for the 6bba Trackastra Kaggle kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "celltrack-alpha.trackastra-6bba-kaggle-input-manifest.v1"
RAW_DIRECTORY_SCHEMA = (
    "celltrack-alpha.trackastra-6bba-raw-directory-manifest.v1"
)
PARITY_IDENTITY = "train:6bba_edf14583"
CONTRACT = ROOT / "reports/g02/trackastra_full_pool_feature_cache_contract_v2.json"
PREPARE = (
    ROOT
    / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/trackastra_6bba_feature_kernel_manifest_v1.json"
)
DEFAULT_RAW_DIRECTORY_OUTPUT = (
    ROOT / "reports/g02/trackastra_6bba_raw_directory_manifest_v1.json"
)


class ManifestError(RuntimeError):
    """The kernel input package is incomplete or drifted."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, *, relative_to: Path | None = ROOT) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"missing input: {path}")
    label = (
        str(path.relative_to(relative_to))
        if relative_to is not None and path.is_relative_to(relative_to)
        else str(path)
    )
    return {
        "path": label,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ManifestError(f"expected JSON object: {path}")
    return value


def source_records(archive: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as handle:
        names = sorted(
            name for name in handle.namelist() if not name.endswith("/")
        )
    if len(names) != len(set(names)):
        raise ManifestError("source bundle repeats an archive member")
    for name in names:
        source = ROOT / name
        rows.append(record(source))
    return rows


def cache_records(eligible: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in eligible:
        slug = identity.replace(":", "__", 1)
        for suffix in (".bhc", ".json"):
            rows.append(record(CACHE / f"{slug}{suffix}"))
    return rows


def eligible_identities() -> tuple[str, ...]:
    prepare = load_json(PREPARE)
    eligible = tuple(
        str(identity)
        for identity in prepare["scope"]["train_video_ids"]
        if str(identity).startswith("train:6bba_")
    )
    if len(eligible) != 69 or eligible[0] != PARITY_IDENTITY:
        raise ManifestError("6bba scope or parity order drifted")
    return eligible


def raw_directory_manifest(
    archive: Path,
    eligible: tuple[str, ...],
    *,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    prefixes = {
        identity: f"train/{identity.split(':', 1)[1]}.zarr/"
        for identity in eligible
    }
    records: list[dict[str, Any]] = []
    covered: set[str] = set()
    with zipfile.ZipFile(archive) as handle:
        infos = sorted(
            (
                info
                for info in handle.infolist()
                if not info.is_dir()
                and any(
                    info.filename.startswith(prefix)
                    for prefix in prefixes.values()
                )
            ),
            key=lambda info: info.filename,
        )
        for ordinal, info in enumerate(infos, start=1):
            digest = hashlib.sha256()
            with handle.open(info) as member:
                for block in iter(lambda: member.read(1 << 20), b""):
                    digest.update(block)
            records.append(
                {
                    "path": info.filename,
                    "bytes": info.file_size,
                    "sha256": digest.hexdigest(),
                }
            )
            for identity, prefix in prefixes.items():
                if info.filename.startswith(prefix):
                    covered.add(identity)
                    break
            if ordinal % 256 == 0:
                print(
                    f"hashed {ordinal}/{len(infos)} raw 6bba files",
                    flush=True,
                )
    if (
        covered != set(eligible)
        or len(records) != 7038
        or len({str(record["path"]) for record in records}) != len(records)
    ):
        raise ManifestError("raw directory coverage drifted")
    source_archive = (
        {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
        }
        if archive_sha256 is not None
        else record(archive, relative_to=None)
    )
    if (
        len(str(source_archive["sha256"])) != 64
        or source_archive["bytes"] != 87393127165
    ):
        raise ManifestError("raw source archive identity drifted")
    return {
        "schema": RAW_DIRECTORY_SCHEMA,
        "scope": {
            "family": "6bba",
            "video_count": 69,
            "identities": list(eligible),
            "parity_reference": PARITY_IDENTITY,
            "training": False,
            "label_access": False,
            "test_access": False,
        },
        "source_archive": source_archive,
        "files": records,
    }


def build(
    args: argparse.Namespace, eligible: tuple[str, ...]
) -> dict[str, Any]:

    assets = args.assets_dir.resolve()
    cache_archive = args.cache_archive.resolve()
    raw_archive = args.raw_archive.resolve()
    source_bundle = assets / "source_bundle.zip"
    packaged = {
        name: record(assets / name, relative_to=assets)
        for name in (
            "source_bundle.zip",
            "trackastra_source.zip",
            "ctc.zip",
            "model.pt",
            "edge.json",
            "v45f4.pt",
        )
    }
    packaged["cache_6bba.zip"] = record(
        cache_archive, relative_to=cache_archive.parent
    )
    if args.raw_archive_sha256:
        if (
            len(args.raw_archive_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in args.raw_archive_sha256
            )
            or not raw_archive.is_file()
        ):
            raise ManifestError("precomputed raw archive identity is invalid")
        packaged["raw_archive"] = {
            "path": str(raw_archive),
            "bytes": raw_archive.stat().st_size,
            "sha256": args.raw_archive_sha256,
        }
    else:
        packaged["raw_archive"] = record(raw_archive, relative_to=None)

    contract = record(CONTRACT)
    if (
        contract["sha256"]
        != "e060b284da623a373a83c8dd690985503af945399518f4d77eb517fddb8128ac"
    ):
        raise ManifestError("two-stage contract SHA-256 drifted")
    if packaged["raw_archive"]["bytes"] != 87393127165:
        raise ManifestError("raw archive byte count drifted")
    raw_directory = record(args.raw_directory_output.resolve())

    return {
        "schema": SCHEMA,
        "scope": {
            "family": "6bba",
            "video_count": len(eligible),
            "identities": list(eligible),
            "parity_reference": PARITY_IDENTITY,
            "training": False,
            "label_access": False,
            "validation12_access": False,
            "cv57_access": False,
            "competition_submission": False,
        },
        "contract": contract,
        "source_inputs": source_records(source_bundle),
        "cache_inputs": cache_records(eligible),
        "raw_directory_manifest": raw_directory,
        "packaged_inputs": packaged,
        "datasets": {
            "assets": {
                "slug": "onebitbrain/trackastra-6bba-assets-v1",
                "version": None,
            },
            "cache": {
                "slug": "onebitbrain/trackastra-6bba-cache-v1",
                "version": None,
            },
            "raw": {
                "slug": "biohub-cell-tracking-during-development",
                "source_type": "competition",
                "version": None,
            },
            "support": {
                "slug": "pilkwang/biohub-tracking-support-pack-50ep-v1",
                "version": None,
            },
        },
        "local_parity_invocations": [
            [
                "/usr/local/bin/python3.13",
                "scripts/build_trackastra_apncc_base_cache.py",
                "--identity",
                PARITY_IDENTITY,
                "--raw-root",
                "/private/tmp/trackastra-6bba-parity-raw",
                "--raw-manifest",
                str(args.raw_directory_output.resolve()),
                "--raw-manifest-sha256",
                raw_directory["sha256"],
                "--output",
                "reports/g02/trackastra_kaggle_parity_base_v1",
            ],
            [
                "/usr/local/bin/python3.13",
                "scripts/enrich_trackastra_apncc_cache.py",
                "--identity",
                PARITY_IDENTITY,
                "--source-root",
                "/private/tmp/trackastra-src.sLG6lP/"
                "trackastra-fb265888c8fce2ac16a362ebf182ccd5ea48e936",
                "--checkpoint-zip",
                "/private/tmp/trackastra_ctc_v0.3.0.zip",
                "--checkpoint-member",
                "/private/tmp/trackastra-mmap-v1/model.pt",
                "--base",
                "reports/g02/trackastra_kaggle_parity_base_v1",
                "--output",
                "reports/g02/trackastra_kaggle_parity_enriched_v1",
            ],
        ],
        "kaggle_stage_invocations": [
            [
                "python",
                "/kaggle/working/scripts/build_trackastra_apncc_base_cache.py",
                "--contract",
                "/kaggle/working/reports/g02/"
                "trackastra_full_pool_feature_cache_contract_v2.json",
                "--contract-sha256",
                contract["sha256"],
                "--cache",
                "/kaggle/input/datasets/onebitbrain/"
                "trackastra-6bba-cache-v1/reports/g02/"
                "fitcache_117/bh9out/cache",
                "--raw-root",
                "/kaggle/input/competitions/"
                "biohub-cell-tracking-during-development",
                "--raw-manifest",
                "/kaggle/input/datasets/onebitbrain/"
                "trackastra-6bba-assets-v1/"
                "trackastra_6bba_raw_directory_manifest_v1.json",
                "--raw-manifest-sha256",
                raw_directory["sha256"],
                "--edge-artifact",
                "/kaggle/input/datasets/onebitbrain/"
                "trackastra-6bba-assets-v1/edge.json",
                "--edge-auxiliary",
                "/kaggle/input/datasets/onebitbrain/"
                "trackastra-6bba-assets-v1/v45f4.pt",
                "--support-root",
                "/kaggle/input/datasets/pilkwang/"
                "biohub-tracking-support-pack-50ep-v1",
                "--output",
                "/kaggle/working/reports/g02/"
                "trackastra_apncc_base_cache_v2",
            ],
            [
                "python",
                "/kaggle/working/scripts/enrich_trackastra_apncc_cache.py",
                "--contract",
                "/kaggle/working/reports/g02/"
                "trackastra_full_pool_feature_cache_contract_v2.json",
                "--contract-sha256",
                contract["sha256"],
                "--source-root",
                "/kaggle/working/trackastra-source",
                "--checkpoint-zip",
                "/kaggle/input/datasets/onebitbrain/"
                "trackastra-6bba-assets-v1/ctc.zip.bin",
                "--checkpoint-member",
                "/kaggle/input/datasets/onebitbrain/"
                "trackastra-6bba-assets-v1/model.pt",
                "--cache",
                "/kaggle/input/datasets/onebitbrain/"
                "trackastra-6bba-cache-v1/reports/g02/"
                "fitcache_117/bh9out/cache",
                "--base",
                "/kaggle/working/reports/g02/"
                "trackastra_apncc_base_cache_v2",
                "--output",
                "/kaggle/working/reports/g02/"
                "trackastra_full_pool_feature_cache_v2",
            ],
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--cache-archive", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument(
        "--raw-directory-output",
        type=Path,
        default=DEFAULT_RAW_DIRECTORY_OUTPUT,
    )
    parser.add_argument(
        "--raw-archive-sha256",
        help="precomputed SHA-256 from an immediately preceding full-file hash",
    )
    parser.add_argument(
        "--reuse-raw-directory-manifest",
        action="store_true",
        help=(
            "reuse an existing authenticated raw-directory manifest after "
            "validating its full 69-video scope and source archive identity"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eligible = eligible_identities()
    if args.reuse_raw_directory_manifest:
        raw_payload = load_json(args.raw_directory_output)
        if (
            raw_payload.get("schema") != RAW_DIRECTORY_SCHEMA
            or raw_payload.get("scope", {}).get("identities")
            != list(eligible)
            or len(raw_payload.get("files", ())) != 7038
            or raw_payload.get("source_archive", {}).get("sha256")
            != args.raw_archive_sha256
            or raw_payload.get("source_archive", {}).get("bytes")
            != args.raw_archive.resolve().stat().st_size
        ):
            raise ManifestError(
                "reused raw directory manifest identity drifted"
            )
    else:
        raw_payload = raw_directory_manifest(
            args.raw_archive.resolve(),
            eligible,
            archive_sha256=args.raw_archive_sha256,
        )
        atomic_json(args.raw_directory_output, raw_payload)
    payload = build(args, eligible)
    atomic_json(args.output, payload)
    print(json.dumps(payload["packaged_inputs"], sort_keys=True))
    return 0


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
