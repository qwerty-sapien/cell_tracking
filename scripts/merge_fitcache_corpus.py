#!/usr/bin/env python3
"""Validate and assemble the exact fold-0 fit-cache corpus.

The merged corpus contains only the 117 fold-0 fit videos declared by the
frozen fit-cache plan.  Source artifacts are hard-linked after their hashes,
cross-file attestations, shard evidence, and Kaggle GPU allocation have all
been validated.  No held-out, audit, test, or submission artifact is copied.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "kaggle_kernel/fitcache/plan.json"
EXISTING = ROOT / "kaggle_kernel/bh9o7/bh9out"
REMOTE = (
    ROOT / "reports/g02/fitcache_remote/fc00_v3",
    ROOT / "reports/g02/fitcache_remote/fc01_v3",
)
OUTPUT = ROOT / "reports/g02/fitcache_117"

BOUNDARY = {
    "train_fit_only": True,
    "held_out_used": False,
    "audit_used": False,
    "test_used": False,
    "competition_submission_performed": False,
}
ARTIFACT_SUFFIXES = {
    "cache": (".bhc", ".json", ".intent.json"),
    "features": (".json",),
    "fragments": (".csv",),
    "status": (".json",),
    "transactions": (".json",),
}


class FitCacheMergeError(RuntimeError):
    """A fit-cache source or merged-corpus invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitCacheMergeError(message)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FitCacheMergeError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FitCacheMergeError(f"JSON root is not an object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FitCacheMergeError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _slug(identity: str) -> str:
    return identity.replace(":", "__", 1)


def _artifact_paths(root: Path, identity: str) -> dict[str, Path]:
    slug = _slug(identity)
    return {
        "cache_bhc": root / "cache" / f"{slug}.bhc",
        "cache_meta": root / "cache" / f"{slug}.json",
        "cache_intent": root / "cache" / f"{slug}.intent.json",
        "features": root / "features" / f"{slug}.json",
        "fragments": root / "fragments" / f"{slug}.csv",
        "status": root / "status" / f"{slug}.json",
        "transactions": root / "transactions" / f"{slug}.json",
    }


def _artifact_relative_paths(identity: str) -> dict[str, Path]:
    slug = _slug(identity)
    return {
        "cache_bhc": Path("cache") / f"{slug}.bhc",
        "cache_meta": Path("cache") / f"{slug}.json",
        "cache_intent": Path("cache") / f"{slug}.intent.json",
        "features": Path("features") / f"{slug}.json",
        "fragments": Path("fragments") / f"{slug}.csv",
        "status": Path("status") / f"{slug}.json",
        "transactions": Path("transactions") / f"{slug}.json",
    }


def _validate_video(root: Path, identity: str) -> dict[str, Any]:
    paths = _artifact_paths(root, identity)
    for label, path in paths.items():
        _require(path.is_file() and path.stat().st_size > 0, f"missing {label}: {path}")

    meta = _load(paths["cache_meta"])
    intent = _load(paths["cache_intent"])
    features = _load(paths["features"])
    status = _load(paths["status"])
    transaction = _load(paths["transactions"])

    _require(meta.get("schema") == "celltrack-alpha.bh-cache-meta.v1", f"bad cache schema: {identity}")
    _require(intent.get("schema") == "celltrack-alpha.bh-cache-intent.v1", f"bad intent schema: {identity}")
    _require(features.get("schema") == "celltrack-alpha.bh-video-features.v1", f"bad feature schema: {identity}")
    _require(status.get("schema") == "celltrack-alpha.bh-video-status.v1", f"bad status schema: {identity}")
    _require(transaction.get("schema") == "celltrack-alpha.bh-video-transaction.v1", f"bad transaction schema: {identity}")

    for label, value in (
        ("intent", intent.get("identity")),
        ("features", features.get("identity")),
        ("status", status.get("identity")),
        ("status result", status.get("result", {}).get("identity")),
        ("transaction", transaction.get("identity")),
    ):
        _require(value == identity, f"{label} identity mismatch: {identity}")
    _require(status.get("ok") is True, f"video status is not successful: {identity}")
    _require(features.get("graph_variant") == "a0", f"feature graph variant is not a0: {identity}")
    _require(status.get("result", {}).get("graph_variant") == "a0", f"status graph variant is not a0: {identity}")

    contract = meta.get("contract")
    _require(isinstance(contract, Mapping), f"cache contract is missing: {identity}")
    _require(intent.get("contract") == contract, f"intent/cache contract mismatch: {identity}")
    _require(transaction.get("inference_contract") == contract, f"transaction/cache contract mismatch: {identity}")
    _require(
        transaction.get("run_contract") == status.get("run_contract"),
        f"transaction/status run contract mismatch: {identity}",
    )
    gate = status.get("run_contract", {}).get("graph_config", {}).get("candidate_gate")
    _require(
        gate
        == {
            "algorithm": "physical-cKDTree",
            "neighbor_cap": 8,
            "radius_um": 10.0,
            "scale_zyx": [1.625, 0.40625, 0.40625],
        },
        f"candidate gate mismatch: {identity}",
    )

    hashes = {label: _sha(path) for label, path in paths.items()}
    _require(meta.get("cache_sha256") == hashes["cache_bhc"], f"cache hash mismatch: {identity}")
    _require(status.get("feature_sha256") == hashes["features"], f"feature hash mismatch: {identity}")
    _require(status.get("fragment_sha256") == hashes["fragments"], f"fragment hash mismatch: {identity}")

    expected_basenames = {
        "cache": paths["cache_bhc"].name,
        "meta": paths["cache_meta"].name,
        "feature": paths["features"].name,
        "fragment": paths["fragments"].name,
        "status": paths["status"].name,
    }
    intent_artifacts = intent.get("artifacts", {})
    transaction_artifacts = transaction.get("artifacts", {})
    for label in ("cache", "meta"):
        _require(
            Path(str(intent_artifacts.get(label, ""))).name == expected_basenames[label],
            f"intent {label} path mismatch: {identity}",
        )
    for label in ("feature", "fragment", "status"):
        _require(
            Path(str(transaction_artifacts.get(label, ""))).name == expected_basenames[label],
            f"transaction {label} path mismatch: {identity}",
        )

    model = contract.get("model", {})
    _require(
        model.get("weight_sha256") == "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
        f"released detector weight mismatch: {identity}",
    )
    return {
        "source_root": str(root.relative_to(ROOT)),
        "model": dict(model),
        "files": {
            label: {"bytes": path.stat().st_size, "sha256": hashes[label]}
            for label, path in paths.items()
        },
    }


def _validate_remote(root: Path, shard: Mapping[str, Any]) -> dict[str, Any]:
    identities = shard.get("video_ids")
    _require(isinstance(identities, list) and identities, f"invalid shard identities: {root}")
    expected_sha = _canonical_sha(identities)
    _require(expected_sha == shard.get("selection_sha256"), f"plan shard hash mismatch: {root}")

    evidence = _load(root / "bh9.json")
    _require(evidence.get("schema") == "celltrack-alpha.fit-cache-run.v1", f"bad shard evidence schema: {root}")
    _require(evidence.get("status") == "completed", f"shard did not complete: {root}")
    _require(evidence.get("purpose") == "fold0-fit-supervised-scaling", f"bad shard purpose: {root}")
    _require(evidence.get("fold") == 0, f"bad shard fold: {root}")
    _require(evidence.get("selection_sha256") == expected_sha, f"shard selection hash mismatch: {root}")
    _require(evidence.get("selected_video_ids") == identities, f"shard identity selection mismatch: {root}")
    _require(evidence.get("selected_video_count") == len(identities), f"shard count mismatch: {root}")
    _require(evidence.get("data_boundary") == BOUNDARY, f"shard data boundary mismatch: {root}")

    expected_counts = {
        folder: len(identities) * len(suffixes)
        for folder, suffixes in ARTIFACT_SUFFIXES.items()
    }
    _require(evidence.get("artifact_file_counts") == expected_counts, f"shard artifact attestation mismatch: {root}")
    for folder, expected in expected_counts.items():
        actual = sum(1 for path in (root / "bh9out" / folder).iterdir() if path.is_file())
        _require(actual == expected, f"shard {folder} file count mismatch: {root}")

    gpu = evidence.get("gpu_attestation", {})
    requested = gpu.get("requested", {})
    observed = gpu.get("observed", {})
    _require(
        requested
        == {
            "devices": ["cuda:0", "cuda:1"],
            "expected_device_count": 2,
            "expected_name_contains": "T4",
            "kaggle_machine_shape": "NvidiaTeslaT4",
            "logical_accelerator": "T4x2",
        },
        f"requested GPU shape mismatch: {root}",
    )
    _require(evidence.get("gpus") == ["Tesla T4", "Tesla T4"], f"GPU names mismatch: {root}")
    _require(observed.get("classification") == "exact_t4x2", f"GPU classification mismatch: {root}")
    _require(observed.get("device_count") == 2, f"GPU count mismatch: {root}")
    _require(observed.get("torch_cuda_available") is True, f"Torch CUDA unavailable: {root}")
    _require(observed.get("mismatches") == [], f"GPU mismatches were reported: {root}")
    devices = observed.get("devices")
    _require(isinstance(devices, list) and len(devices) == 2, f"GPU inventory missing: {root}")
    for index, device in enumerate(devices):
        _require(device.get("index") == index, f"GPU index mismatch: {root}")
        _require("T4" in str(device.get("name")), f"unexpected GPU family: {root}")
        _require(device.get("capability") == [7, 5], f"GPU capability mismatch: {root}")
    smi = gpu.get("nvidia_smi", {})
    for command in ("inventory", "query"):
        _require(smi.get(command, {}).get("returncode") == 0, f"nvidia-smi {command} failed: {root}")
        _require(smi.get(command, {}).get("stdout"), f"nvidia-smi {command} was empty: {root}")

    run_path = root / "bh9out/run.json"
    _require(_sha(run_path) == evidence.get("run_sha256"), f"run report hash mismatch: {root}")
    run = _load(run_path)
    _require(run.get("schema") == "celltrack-alpha.bh-runner.v1", f"bad run schema: {root}")
    _require(run.get("ok") is True and run.get("mode") == "cv", f"run was not successful CV: {root}")
    _require(run.get("graph_variant") == "a0", f"run graph variant mismatch: {root}")
    _require(run.get("devices") == ["cuda:0", "cuda:1"], f"run device list mismatch: {root}")
    _require(run.get("model_loads") == 2, f"run model load count mismatch: {root}")
    _require(run.get("requested_identities") == sorted(identities), f"run identities mismatch: {root}")
    flattened = [item for worker in run.get("whole_video_shards", []) for item in worker]
    _require(sorted(flattened) == sorted(identities), f"worker shard coverage mismatch: {root}")
    return {
        "root": str(root.relative_to(ROOT)),
        "selection_sha256": expected_sha,
        "selected_video_count": len(identities),
        "run_sha256": evidence["run_sha256"],
        "gpu_classification": observed["classification"],
        "gpu_names": evidence["gpus"],
        "torch_version": observed.get("torch_version"),
        "torch_cuda_version": observed.get("torch_cuda_version"),
    }


def _validate_plan(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    plan = document.get("plan")
    packages = document.get("packages")
    _require(isinstance(plan, Mapping), "fit-cache plan is missing")
    _require(isinstance(packages, list) and len(packages) == 2, "fit-cache packages are missing")
    shards = plan.get("shards")
    _require(isinstance(shards, list) and len(shards) == 2, "fit-cache shards are missing")
    _require(plan.get("schema") == "celltrack-alpha.fit-cache-plan.v1", "fit-cache plan schema mismatch")
    _require(plan.get("fold") == 0 and plan.get("fit_video_count") == 117, "fit-cache plan scope mismatch")
    _require(plan.get("already_complete_video_count") == 14, "existing fit-cache count mismatch")
    _require(plan.get("missing_video_count") == 103, "remote fit-cache count mismatch")
    _require([row.get("video_count") for row in shards] == [52, 51], "remote shard sizes mismatch")
    _require(plan.get("data_boundary") == BOUNDARY, "fit-cache plan data boundary mismatch")

    fit_ids = plan.get("fit_video_ids")
    existing_ids = plan.get("already_complete_video_ids")
    missing_ids = plan.get("missing_video_ids")
    _require(
        all(isinstance(value, list) for value in (fit_ids, existing_ids, missing_ids)),
        "fit-cache plan identity lists are malformed",
    )
    shard_ids = [identity for shard in shards for identity in shard.get("video_ids", [])]
    _require(len(fit_ids) == len(set(fit_ids)) == 117, "fit-cache plan identities are not unique")
    _require(len(existing_ids) == len(set(existing_ids)) == 14, "existing identities are not unique")
    _require(len(shard_ids) == len(set(shard_ids)) == 103, "remote identities are not unique")
    _require(set(existing_ids).isdisjoint(shard_ids), "existing and remote selections overlap")
    _require(set(existing_ids) | set(shard_ids) == set(fit_ids), "source selections do not cover fit corpus")
    _require(set(missing_ids) == set(shard_ids), "missing selection differs from shard coverage")
    return plan, tuple(shards)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)


def merge(output: Path) -> Mapping[str, Any]:
    _require(not output.exists(), f"refusing to overwrite merged corpus: {output}")
    document = _load(PLAN)
    plan, shards = _validate_plan(document)
    remote_reports = tuple(_validate_remote(root, shard) for root, shard in zip(REMOTE, shards, strict=True))

    source_by_id = {identity: EXISTING for identity in plan["already_complete_video_ids"]}
    for root, shard in zip(REMOTE, shards, strict=True):
        source_by_id.update({identity: root / "bh9out" for identity in shard["video_ids"]})
    _require(set(source_by_id) == set(plan["fit_video_ids"]), "fit-cache source map is incomplete")

    videos: dict[str, Any] = {}
    model_identities: set[str] = set()
    for identity in plan["fit_video_ids"]:
        videos[identity] = _validate_video(source_by_id[identity], identity)
        model_identities.add(_canonical_sha(videos[identity]["model"]))
    _require(len(model_identities) == 1, "fit-cache sources do not share one released model identity")

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    _require(not temporary.exists(), f"temporary merge path already exists: {temporary}")
    try:
        merged_root = temporary / "bh9out"
        for folder in ARTIFACT_SUFFIXES:
            (merged_root / folder).mkdir(parents=True, exist_ok=False)
        for identity in plan["fit_video_ids"]:
            source_paths = _artifact_paths(source_by_id[identity], identity)
            relative_paths = _artifact_relative_paths(identity)
            for label, source in source_paths.items():
                destination = merged_root / relative_paths[label]
                os.link(source, destination)
                _require(
                    source.stat().st_ino == destination.stat().st_ino,
                    f"merged artifact is not a hard link: {destination}",
                )

        expected_counts = {
            folder: len(plan["fit_video_ids"]) * len(suffixes)
            for folder, suffixes in ARTIFACT_SUFFIXES.items()
        }
        actual_counts = {
            folder: sum(1 for path in (merged_root / folder).iterdir() if path.is_file())
            for folder in ARTIFACT_SUFFIXES
        }
        _require(actual_counts == expected_counts, "merged artifact counts are not exact")
        merged_ids = {
            path.name.removesuffix(".bhc").replace("__", ":", 1)
            for path in (merged_root / "cache").glob("*.bhc")
        }
        _require(merged_ids == set(plan["fit_video_ids"]), "merged cache identities are not exact")

        manifest_core = {
            "schema": "celltrack-alpha.fit-cache-corpus.v1",
            "status": "validated",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fold": 0,
            "video_count": len(plan["fit_video_ids"]),
            "video_ids": list(plan["fit_video_ids"]),
            "artifact_file_counts": actual_counts,
            "merge_method": "hard-link",
            "data_boundary": BOUNDARY,
            "plan": {
                "path": str(PLAN.relative_to(ROOT)),
                "sha256": _sha(PLAN),
                "plan_sha256": plan.get("plan_sha256"),
                "existing_video_count": len(plan["already_complete_video_ids"]),
                "remote_video_count": len(plan["missing_video_ids"]),
            },
            "remote_runs": list(remote_reports),
            "model_identity_sha256": next(iter(model_identities)),
            "videos": videos,
        }
        manifest = {**manifest_core, "manifest_sha256": _canonical_sha(manifest_core)}
        _write_json(temporary / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    report = merge(args.output.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "video_count": report["video_count"],
                "artifact_file_counts": report["artifact_file_counts"],
                "manifest_sha256": report["manifest_sha256"],
                "output": str(args.output.resolve()),
                "gpu_classifications": [
                    row["gpu_classification"] for row in report["remote_runs"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FitCacheMergeError as exc:
        print(f"fit-cache merge failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
