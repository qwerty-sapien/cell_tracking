#!/usr/bin/env python3
"""Build the V45/raw-only 48-D APNCC half of the Trackastra cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time

import torch
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from build_trackastra_full_pool_feature_cache import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_CONTRACT_SHA256,
    TrackastraFeatureCacheError,
    _atomic_json,
    _atomic_torch_save,
    _checkpoint,
    _load_json,
    _require,
    flatten_base_windows,
    load_contract,
)
from celltrack_alpha import appearance_swap, bh_edge, bh_model, point_context_swap  # noqa: E402
from evaluate_trackastra_zero_wrfeat_signal import sha256  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
)
from prepare_extended_radius_shell import _baseline  # noqa: E402


SCHEMA = "celltrack-alpha.trackastra-apncc-base-cache.v2"
VIDEO_SCHEMA = "celltrack-alpha.trackastra-apncc-base-video.v2"
CHECKPOINT_SCHEMA = "celltrack-alpha.trackastra-apncc-base-checkpoint.v2"
DEFAULT_OUTPUT = ROOT / "reports/g02/trackastra_apncc_base_cache_v2"


def _cache_identity(meta: Mapping[str, object]) -> bh_model.CacheIdentity:
    try:
        model = meta["contract"]["model"]
        assert isinstance(model, Mapping)
        return bh_model.CacheIdentity(
            str(model["config_sha256"]),
            str(model["source_sha256"]),
            str(model["weight_sha256"]),
            int(model["pair_batch_size"]),
            int(model["view_batch_size"]),
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise TrackastraFeatureCacheError(
            "cache metadata lacks a valid model identity"
        ) from exc


def _validate_raw_archive(
    path: Path,
    *,
    expected_path: Path,
    expected_bytes: int,
    external_sha256: str | None,
) -> None:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise TrackastraFeatureCacheError(
            f"raw archive identity mismatch: {path}"
        )
    if path == expected_path:
        return
    if (
        not external_sha256
        or len(external_sha256) != 64
        or sha256(path) != external_sha256
    ):
        raise TrackastraFeatureCacheError(
            f"external raw archive identity mismatch: {path}"
        )


def _validate_raw_directory(
    root: Path,
    manifest_path: Path,
    *,
    manifest_sha256: str,
    identities: tuple[str, ...],
) -> None:
    if (
        not root.is_dir()
        or not manifest_path.is_file()
        or len(manifest_sha256) != 64
        or sha256(manifest_path) != manifest_sha256
    ):
        raise TrackastraFeatureCacheError(
            "raw directory manifest identity mismatch"
        )
    manifest = _load_json(manifest_path, "raw directory manifest")
    scope = manifest.get("scope")
    records = manifest.get("files")
    if (
        manifest.get("schema")
        != "celltrack-alpha.trackastra-6bba-raw-directory-manifest.v1"
        or not isinstance(scope, Mapping)
        or scope.get("family") != "6bba"
        or scope.get("video_count") != 69
        or not isinstance(records, list)
    ):
        raise TrackastraFeatureCacheError(
            "raw directory manifest scope drifted"
        )
    authorized = tuple(str(value) for value in scope.get("identities", ()))
    if (
        len(authorized) != 69
        or len(set(authorized)) != 69
        or any(
            identity not in authorized
            or not identity.startswith("train:6bba_")
            for identity in identities
        )
    ):
        raise TrackastraFeatureCacheError(
            "raw directory identity scope drifted"
        )
    expected: dict[str, tuple[int, str]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TrackastraFeatureCacheError(
                "raw directory manifest record drifted"
            )
        relative = str(record.get("path", ""))
        expected[relative] = (
            int(record.get("bytes", -1)),
            str(record.get("sha256", "")),
        )
    if len(expected) != len(records):
        raise TrackastraFeatureCacheError(
            "raw directory manifest repeats a path"
        )
    for identity in identities:
        dataset = identity.split(":", 1)[1]
        prefix = f"train/{dataset}.zarr/"
        selected = {
            relative: record
            for relative, record in expected.items()
            if relative.startswith(prefix)
        }
        store_root = root / "train" / f"{dataset}.zarr"
        actual = {
            str(path.relative_to(root))
            for path in store_root.rglob("*")
            if path.is_file()
        }
        if not selected or actual != set(selected):
            raise TrackastraFeatureCacheError(
                f"raw directory coverage drifted for {identity}"
            )
        for relative, (expected_bytes, expected_sha256) in selected.items():
            path = root / relative
            if (
                path.stat().st_size != expected_bytes
                or len(expected_sha256) != 64
                or sha256(path) != expected_sha256
            ):
                raise TrackastraFeatureCacheError(
                    f"raw directory input drifted: {relative}"
                )


def _eligible(
    contract: Mapping[str, object],
    identity: str | None = None,
) -> tuple[tuple[str, ...], dict[str, object], dict[str, object]]:
    inputs = contract["inputs"]
    assert isinstance(inputs, Mapping)
    prepare = _load_json(
        ROOT / str(inputs["fit_prepare"]["path"]), "fit prepare"
    )
    feasibility = _load_json(
        ROOT / str(inputs["full_pool_feasibility"]["path"]),
        "full-pool feasibility",
    )
    scope = prepare.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("fit_only") is not True
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise TrackastraFeatureCacheError("fit prepare scope drifted")
    full_eligible = tuple(
        str(identity)
        for identity in scope["train_video_ids"]
        if str(identity).startswith("train:6bba_")
    )
    if len(full_eligible) != 69:
        raise TrackastraFeatureCacheError("eligible video count drifted")
    if identity is not None:
        parity_identity = str(
            contract["checkpoint_loader"]["parity_reference"]["identity"]
        )
        if identity != parity_identity or identity not in full_eligible:
            raise TrackastraFeatureCacheError(
                "single-video scope is restricted to the parity reference"
            )
        eligible = (identity,)
    else:
        eligible = full_eligible
    prepare_rows = {
        str(row["identity"]): row for row in prepare["videos"]
    }
    feasibility_rows = {
        str(row["identity"]): row for row in feasibility["videos"]
    }
    if set(full_eligible) != set(feasibility_rows):
        raise TrackastraFeatureCacheError(
            "feasibility video coverage drifted"
        )
    return eligible, prepare_rows, feasibility_rows


def _base_checkpoint(
    *,
    path: Path,
    feature_path: Path,
    identity: str,
    contract_sha256: str,
    implementation_sha256: str,
    expected_proposals: int,
) -> dict[str, object] | None:
    if not path.exists() and not feature_path.exists():
        return None
    if not path.is_file() or not feature_path.is_file():
        raise TrackastraFeatureCacheError(
            f"incomplete base checkpoint for {identity}"
        )
    value = _load_json(path, f"{identity} base checkpoint")
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("identity") != identity
        or value.get("contract_sha256") != contract_sha256
        or value.get("implementation_sha256") != implementation_sha256
        or int(value.get("proposal_count", -1)) != expected_proposals
        or value.get("feature_sha256") != sha256(feature_path)
    ):
        raise TrackastraFeatureCacheError(
            f"base checkpoint identity drifted for {identity}"
        )
    return value


def build(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    contract = load_contract(args.contract, args.contract_sha256)
    if contract["execution"].get("model_stack_coexistence") is not False:
        raise TrackastraFeatureCacheError(
            "two-stage model isolation contract drifted"
        )
    implementation_sha256 = sha256(Path(__file__))
    inputs = contract["inputs"]
    for label in (
        "appearance_contract",
        "fit_prepare",
        "full_pool_feasibility",
        "asset_audit",
    ):
        record = inputs[label]
        _require(ROOT / str(record["path"]), str(record["sha256"]), label)
    eligible, prepare_rows, feasibility_rows = _eligible(
        contract, getattr(args, "identity", None)
    )
    raw_root = getattr(args, "raw_root", None)
    if raw_root is None:
        raw_archive = inputs["raw_archive"]
        expected_archive = (ROOT / str(raw_archive["path"])).resolve()
        _validate_raw_archive(
            args.zip,
            expected_path=expected_archive,
            expected_bytes=int(raw_archive["bytes"]),
            external_sha256=getattr(args, "raw_archive_sha256", None),
        )
    else:
        raw_manifest = getattr(args, "raw_manifest", None)
        raw_manifest_sha256 = getattr(args, "raw_manifest_sha256", None)
        if raw_manifest is None or raw_manifest_sha256 is None:
            raise TrackastraFeatureCacheError(
                "directory-backed raw input requires its pinned manifest"
            )
        _validate_raw_directory(
            raw_root,
            raw_manifest,
            manifest_sha256=raw_manifest_sha256,
            identities=eligible,
        )
    appearance_contract, geometry_gate = appearance_swap.load_contract(
        ROOT / str(inputs["appearance_contract"]["path"]),
        expected_sha256=str(inputs["appearance_contract"]["sha256"]),
        root=ROOT,
    )
    base_model, bundle = bh_model.load_v45_model(
        args.support_root, device="cpu"
    )
    scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        sha256(args.edge_artifact),
        args.edge_auxiliary,
        base_model,
        bundle,
        edge_module=bh_edge,
    )
    if scorer is None or edge_record is None:
        raise TrackastraFeatureCacheError(
            "frozen edge scorer did not restore"
        )

    args.output.mkdir(parents=True, exist_ok=args.resume)
    videos_root = args.output / "videos"
    videos_root.mkdir(exist_ok=args.resume)
    files: list[dict[str, object]] = []
    resumed = 0
    store_context = (
        nullcontext(None)
        if raw_root is not None
        else ZipStore(args.zip, mode="r")
    )
    with store_context as store:
        for ordinal, identity in enumerate(eligible, start=1):
            slug = identity.replace(":", "__", 1)
            feature_path = videos_root / f"{slug}.pt"
            checkpoint_path = videos_root / f"{slug}.json"
            expected_proposals = int(
                feasibility_rows[identity]["geometry_proposals"]
            )
            existing = (
                _base_checkpoint(
                    path=checkpoint_path,
                    feature_path=feature_path,
                    identity=identity,
                    contract_sha256=args.contract_sha256,
                    implementation_sha256=implementation_sha256,
                    expected_proposals=expected_proposals,
                )
                if args.resume
                else None
            )
            if existing is not None:
                files.append(existing)
                resumed += 1
                print(
                    f"[{ordinal}/{len(eligible)}] {identity}: base resume",
                    flush=True,
                )
                continue
            if feature_path.exists() or checkpoint_path.exists():
                raise TrackastraFeatureCacheError(
                    f"refusing to overwrite base checkpoint for {identity}"
                )
            meta = _load_json(
                args.cache / f"{slug}.json", f"{identity} cache metadata"
            )
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(meta),
            )
            selected, _probabilities, _no_link, graph = _baseline(
                cache, scorer
            )
            if int(graph["raw_selected_edges"]) != int(
                prepare_rows[identity]["graph"]["raw_selected_edges"]
            ):
                raise TrackastraFeatureCacheError(
                    f"{identity} selected graph identity drifted"
                )
            dataset = identity.split(":", 1)[1]
            if raw_root is None:
                image = zarr.open_group(
                    store=store, path=f"train/{dataset}.zarr", mode="r"
                )["0"]
            else:
                image = zarr.open_group(
                    store=str(raw_root / "train" / f"{dataset}.zarr"),
                    mode="r",
                )["0"]
            windows, pool_counts = point_context_swap.build_full_pool_windows(
                cache,
                scorer,
                selected,
                geometry_gate,
                image,
                identity=identity,
                family=1,
                geometry_threshold=float(
                    appearance_contract["geometry_gate"]["threshold"]
                ),
            )
            if (
                int(pool_counts["full_pool_proposals"])
                != expected_proposals
                or len(windows)
                != int(feasibility_rows[identity]["nonempty_transitions"])
            ):
                raise TrackastraFeatureCacheError(
                    f"{identity} full-pool counts drifted"
                )
            flattened = flatten_base_windows(windows)
            payload = {
                "schema": VIDEO_SCHEMA,
                "identity": identity,
                "contract_sha256": args.contract_sha256,
                "implementation_sha256": implementation_sha256,
                "feature_names": [
                    f"apncc_context_{index:02d}" for index in range(48)
                ],
                "selected_raw_edges": len(selected),
                "pool_counts": pool_counts,
                **flattened,
            }
            _atomic_torch_save(feature_path, payload)
            record = {
                "schema": CHECKPOINT_SCHEMA,
                "identity": identity,
                "contract_sha256": args.contract_sha256,
                "implementation_sha256": implementation_sha256,
                "feature_path": str(feature_path.relative_to(ROOT)),
                "feature_sha256": sha256(feature_path),
                "bytes": feature_path.stat().st_size,
                "proposal_count": len(flattened["challengers"]),
                "row_count": len(flattened["conflicts"]),
                "transition_count": len(windows),
            }
            _atomic_json(checkpoint_path, record)
            files.append(record)
            print(
                f"[{ordinal}/{len(eligible)}] {identity}: "
                f"{record['proposal_count']} proposals / "
                f"{record['transition_count']} calls",
                flush=True,
            )
            del cache, windows, flattened, payload, image
            gc.collect()
    aggregate = {
        "video_count": len(files),
        "proposal_count": sum(int(row["proposal_count"]) for row in files),
        "row_count": sum(int(row["row_count"]) for row in files),
        "transition_count": sum(
            int(row["transition_count"]) for row in files
        ),
        "bytes": sum(int(row["bytes"]) for row in files),
        "resumed_video_count": resumed,
    }
    if (
        aggregate["video_count"] != len(eligible)
        or aggregate["proposal_count"]
        != sum(
            int(feasibility_rows[identity]["geometry_proposals"])
            for identity in eligible
        )
        or aggregate["transition_count"]
        != sum(
            int(feasibility_rows[identity]["nonempty_transitions"])
            for identity in eligible
        )
    ):
        raise TrackastraFeatureCacheError(
            "base cache aggregate coverage drifted"
        )
    return {
        "schema": SCHEMA,
        "verdict": "PASS_APNCC_BASE_CACHE",
        "contract": {
            "path": str(args.contract.relative_to(ROOT)),
            "sha256": args.contract_sha256,
        },
        "implementation_sha256": implementation_sha256,
        "scope": contract["scope"],
        "feature_count": 48,
        "model_stack": {
            "v45_loaded": True,
            "trackastra_loaded": False,
            "raw_image_access": True,
        },
        "aggregate": aggregate,
        "execution": {
            "device": "cpu",
            "training": False,
            "label_access": False,
            "elapsed_seconds": time.perf_counter() - started,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
        "files": files,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument(
        "--raw-root",
        type=Path,
        help=(
            "directory-mounted competition root; only contract-listed "
            "train:6bba_ Zarr stores are permitted"
        ),
    )
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        help="manifest containing per-file SHA-256 records for --raw-root",
    )
    parser.add_argument(
        "--raw-manifest-sha256",
        help="required SHA-256 of --raw-manifest",
    )
    parser.add_argument(
        "--raw-archive-sha256",
        help=(
            "required SHA-256 when --zip is an externally mounted copy "
            "rather than the contract-relative local path"
        ),
    )
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument(
        "--support-root", type=Path, default=DEFAULT_SUPPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--identity",
        help=(
            "run only the contract parity reference; any other identity "
            "fails closed"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "contract",
        "cache",
        "zip",
        "edge_artifact",
        "edge_auxiliary",
        "support_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("raw_root", "raw_manifest"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    report_path = args.output / "report.json"
    if report_path.exists():
        raise TrackastraFeatureCacheError(
            f"refusing to overwrite completed report: {report_path}"
        )
    report = build(args)
    _atomic_json(report_path, report)
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
