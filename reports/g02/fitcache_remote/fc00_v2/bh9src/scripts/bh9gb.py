#!/usr/bin/env python3
"""Leakage-safe, deterministic 9 GB V45 train-video benchmark."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_bh  # noqa: E402


ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
SUPPORT = ROOT / "external/Tracking_Support"
OUT_DIR = ROOT / "reports/g02/bh9out"
EVIDENCE = ROOT / "reports/g02/bh9.json"
FAILURE = ROOT / "reports/g02/bh9fail.json"
FROZEN = ROOT / "reports/g02/bh9m.json"

SEED = 20_260_718
TARGET_BYTES = 9_000_000_000
TRAIN_COUNT = 199
DEADLINE_SECONDS = 50 * 60
MIN_SCORE = 0.90
RUNTIME_ATTESTATION_LEVEL = 6
DEVICES = ("cuda:0", "cuda:1")
FROZEN_VIDEO_COUNT = 22
FROZEN_FRAMES_PER_VIDEO = run_bh.FRAME_COUNT
FROZEN_FRAME_COUNT = FROZEN_VIDEO_COUNT * FROZEN_FRAMES_PER_VIDEO
FROZEN_ADJACENT_PAIR_COUNT = FROZEN_VIDEO_COUNT * (FROZEN_FRAMES_PER_VIDEO - 1)
SUBMISSION = "sub.csv"
SCORER_FILES = (
    "src/celltrack_alpha/official_eval.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py",
    "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py",
)
IDENTITY_RE = re.compile(r"^train:([A-Za-z0-9][A-Za-z0-9_.-]*)$")
A0_EVIDENCE_GATES = (
    "graph_variant_a0",
    "exactly_two_a0_scorer_instances",
    "exact_22_video_a0_audit",
    "exact_2178_adjacent_pairs",
    "all_cache_contexts_covered",
    "finite_unique_original_coo_probabilities",
    "full_frame_dense_matrix_false",
    "balanced_attention_fallback_accounting",
    "checkpoint_weight_identity",
)


class Bh9Error(RuntimeError):
    """The benchmark cannot produce acceptance evidence safely."""


class Bh9Deadline(Bh9Error):
    """The global 50-minute wall-clock deadline expired."""


class Bh9Rejected(Bh9Error):
    """A complete score did not pass the fixed acceptance gate."""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def _pairs_no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_dupes,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {item}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Bh9Error(f"invalid JSON in {path}: {exc}") from exc


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + suffix
    ).encode("ascii")


def canonical_sha(value: Any, *, newline: bool = False) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=newline)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Bh9Error(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def short_created_name(path: Path, label: str) -> None:
    if not path.name or len(path.name) >= 15:
        raise Bh9Error(f"{label} basename must be under 15 characters: {path.name!r}")


class HardLimit(AbstractContextManager["HardLimit"]):
    """Enforce the one global deadline across selection, inference, and scoring."""

    def __init__(self, seconds: float = DEADLINE_SECONDS) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("deadline seconds must be positive and finite")
        self.seconds = float(seconds)
        self.deadline = time.monotonic() + self.seconds
        self._old_handler: Any = None
        self._old_timer = (0.0, 0.0)

    def __enter__(self) -> "HardLimit":
        if threading.current_thread() is not threading.main_thread():
            raise Bh9Error("hard deadline requires the main thread")
        if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
            raise Bh9Error("hard wall-clock deadline is unsupported on this platform")
        self._old_handler = signal.getsignal(signal.SIGALRM)
        self._old_timer = signal.getitimer(signal.ITIMER_REAL)
        if self._old_timer[0] > 0:
            raise Bh9Error("refusing to replace an existing process alarm")

        def expired(_signum: int, _frame: Any) -> None:
            raise Bh9Deadline("hard 50-minute deadline reached")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._old_handler)
        return False

    def remaining(self, stage: str) -> float:
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise Bh9Deadline(f"hard 50-minute deadline reached before {stage}")
        return value


def load_train_registry(path: Path) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    payload = strict_json(path)
    if not isinstance(payload, Mapping) or payload.get("artifact") != "dataset_registry_v1":
        raise Bh9Error("registry is not dataset_registry_v1")
    records = payload.get("records")
    if not isinstance(records, list):
        raise Bh9Error("registry records must be a list")
    identities: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise Bh9Error("registry contains a non-object record")
        if record.get("split") != "train":
            continue
        identity = record.get("video_id")
        dataset = record.get("dataset")
        if not isinstance(identity, str) or not isinstance(dataset, str):
            raise Bh9Error("train registry record lacks an identity")
        match = IDENTITY_RE.fullmatch(identity)
        if match is None or match.group(1) != dataset:
            raise Bh9Error(f"malformed train registry identity {identity!r}")
        identities.append(identity)
    counts = payload.get("counts")
    declared = counts.get("train_videos") if isinstance(counts, Mapping) else None
    if declared != TRAIN_COUNT or len(identities) != TRAIN_COUNT:
        raise Bh9Error(
            f"registry must contain exactly {TRAIN_COUNT} train videos; "
            f"declared={declared!r}, observed={len(identities)}"
        )
    if len(set(identities)) != len(identities):
        raise Bh9Error("registry contains duplicate train identities")
    return tuple(sorted(identities)), payload


def _central_record(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "filename": info.filename,
        "crc32": f"{info.CRC:08x}",
        "compressed_size": int(info.compress_size),
        "uncompressed_size": int(info.file_size),
    }


def image_inventory(
    archive: Path,
    identities: Sequence[str],
) -> dict[str, Any]:
    """Read central-directory metadata only; GEFF members never affect image bytes."""

    wanted = set(identities)
    if not wanted or len(wanted) != len(identities):
        raise Bh9Error("image inventory identities must be nonempty and unique")
    sizes = {identity: 0 for identity in wanted}
    member_counts = {identity: 0 for identity in wanted}
    root_counts = {identity: 0 for identity in wanted}
    member_names: dict[str, set[str]] = {identity: set() for identity in wanted}
    archive_images: set[str] = set()
    central: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive, mode="r") as handle:
            infos = handle.infolist()
            central = [_central_record(info) for info in sorted(infos, key=lambda item: item.filename)]
            for info in infos:
                parts = info.filename.split("/")
                if len(parts) < 3 or parts[0] != "train" or not parts[1].endswith(".zarr"):
                    continue
                dataset = parts[1][:-5]
                identity = f"train:{dataset}"
                archive_images.add(identity)
                if identity not in wanted:
                    continue
                relative = "/".join(parts[2:])
                if relative in member_names[identity]:
                    raise Bh9Error(f"duplicate image ZIP member for {identity}: {relative}")
                member_names[identity].add(relative)
                if relative == "0/zarr.json":
                    root_counts[identity] += 1
                if not info.is_dir():
                    if info.compress_size < 0:
                        raise Bh9Error(f"negative compressed member size for {identity}")
                    sizes[identity] += int(info.compress_size)
                    member_counts[identity] += 1
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise Bh9Error(f"cannot inspect competition ZIP {archive}: {exc}") from exc

    missing = sorted(wanted - archive_images)
    extra = sorted(archive_images - wanted)
    bad_roots = {key: value for key, value in root_counts.items() if value != 1}
    empty = sorted(key for key, value in member_counts.items() if value <= 0 or sizes[key] <= 0)
    if missing or extra or bad_roots or empty:
        raise Bh9Error(
            "registry/archive image identity mismatch: "
            f"missing={missing}, extra={extra}, roots={bad_roots}, empty={empty}"
        )
    return {
        "compressed_bytes": {key: sizes[key] for key in sorted(sizes)},
        "member_counts": {key: member_counts[key] for key in sorted(member_counts)},
        "train_image_bytes": sum(sizes.values()),
        "central_sha256": canonical_sha(central, newline=True),
        "entry_count": len(central),
    }


def directory_image_inventory(root: Path, identities: Sequence[str]) -> dict[str, Any]:
    """Validate only extracted train image identities; do not inspect any labels."""

    wanted = set(identities)
    if not wanted or len(wanted) != len(identities):
        raise Bh9Error("directory inventory identities must be nonempty and unique")
    train = root / "train"
    if not train.is_dir():
        raise Bh9Error(f"competition directory has no train directory: {train}")
    try:
        observed = {
            f"train:{path.name[:-5]}"
            for path in train.iterdir()
            if path.is_dir() and path.name.endswith(".zarr")
        }
    except OSError as exc:
        raise Bh9Error(f"cannot enumerate competition train images: {exc}") from exc
    missing = sorted(wanted - observed)
    extra = sorted(observed - wanted)
    if missing or extra or len(observed) != TRAIN_COUNT:
        raise Bh9Error(
            "registry/directory image identity mismatch: "
            f"expected={TRAIN_COUNT}, observed={len(observed)}, missing={missing}, extra={extra}"
        )
    return {
        "input_kind": "directory",
        "train_image_count": len(observed),
        "train_image_identities_sha256": canonical_sha(sorted(observed), newline=True),
        "labels_inspected": False,
    }


def seeded_order(identities: Sequence[str], seed: int = SEED) -> tuple[str, ...]:
    if type(seed) is not int or seed < 0:
        raise Bh9Error("frozen seed must be a nonnegative integer")
    unique = tuple(identities)
    if not unique or len(set(unique)) != len(unique):
        raise Bh9Error("seeded identities must be nonempty and unique")

    def key(identity: str) -> tuple[bytes, str]:
        if IDENTITY_RE.fullmatch(identity) is None:
            raise Bh9Error(f"invalid train identity {identity!r}")
        digest = hashlib.sha256(f"{seed}\0{identity}".encode("ascii")).digest()
        return digest, identity

    return tuple(sorted(unique, key=key))


def select_bytes(
    identities: Sequence[str],
    sizes: Mapping[str, int],
    *,
    target: int = TARGET_BYTES,
    seed: int = SEED,
) -> dict[str, Any]:
    if type(target) is not int or target <= 0:
        raise Bh9Error("byte target must be a positive integer")
    if set(sizes) != set(identities):
        raise Bh9Error("compressed-byte inventory does not exactly cover the registry")
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise Bh9Error("every registered video must have positive exact compressed bytes")
    total = sum(sizes.values())
    if total < target:
        raise Bh9Error(f"train image bytes {total} are below the fixed target {target}")
    ordered = seeded_order(identities, seed)
    rows: list[dict[str, Any]] = []
    cumulative = 0
    for identity in ordered:
        video_bytes = sizes[identity]
        cumulative += video_bytes
        rows.append(
            {
                "identity": identity,
                "compressed_bytes": video_bytes,
                "cumulative_bytes": cumulative,
            }
        )
        if cumulative >= target:
            break
    if not rows or cumulative < target or cumulative - rows[-1]["compressed_bytes"] >= target:
        raise Bh9Error("selection did not stop at the first whole-video target crossing")
    core = {
        "seed": seed,
        "ordering": "sha256(seed\\0identity),identity",
        "target_compressed_bytes": target,
        "compressed_byte_definition": "sum(ZipInfo.compress_size) for train/<dataset>.zarr members",
        "registry_train_count": len(identities),
        "train_image_compressed_bytes": total,
        "selected_count": len(rows),
        "selected_compressed_bytes": cumulative,
        "count_fraction": len(rows) / len(identities),
        "byte_fraction": cumulative / total,
        "selected": rows,
        "selected_video_ids": [row["identity"] for row in rows],
    }
    return {**core, "selection_sha256": canonical_sha(core)}


def build_frozen_manifest(
    inventory: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    registry_sha256: str,
    split_sha256: str,
    origin_registry_sha256: str | None = None,
    origin_split_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze ZIP-only byte evidence for use with Kaggle's extracted directory."""

    sizes = inventory.get("compressed_bytes")
    if not isinstance(sizes, Mapping):
        raise Bh9Error("frozen manifest requires the exact ZIP compressed-byte inventory")
    recomputed = select_bytes(tuple(sorted(str(key) for key in sizes)), sizes)
    if dict(selection) != recomputed:
        raise Bh9Error("frozen manifest selection differs from the byte-only selection")
    identity = {
        "seed": SEED,
        "target_compressed_bytes": TARGET_BYTES,
        "train_count": TRAIN_COUNT,
        "archive_central_sha256": str(inventory.get("central_sha256", "")),
        "archive_entry_count": int(inventory.get("entry_count", 0)),
        "registry_sha256": registry_sha256,
        "split_sha256": split_sha256,
        "origin_registry_sha256": origin_registry_sha256 or registry_sha256,
        "origin_split_sha256": origin_split_sha256 or split_sha256,
        "compressed_bytes": {str(key): int(sizes[key]) for key in sorted(sizes)},
        "selection": recomputed,
    }
    if (
        len(identity["archive_central_sha256"]) != 64
        or identity["archive_entry_count"] <= 0
        or len(identity["registry_sha256"]) != 64
        or len(identity["split_sha256"]) != 64
    ):
        raise Bh9Error("frozen manifest lacks exact source identities")
    return {
        "schema": "celltrack-alpha.bh9m.v1",
        "identity": identity,
        "identity_sha256": canonical_sha(identity),
    }


def load_frozen_manifest(
    path: Path,
    *,
    registry_sha256: str,
    split_sha256: str,
    identities: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = strict_json(path)
    identity = payload.get("identity") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "celltrack-alpha.bh9m.v1"
        or not isinstance(identity, Mapping)
        or payload.get("identity_sha256") != canonical_sha(identity)
    ):
        raise Bh9Error("immutable directory manifest identity is invalid")
    if (
        identity.get("seed") != SEED
        or identity.get("target_compressed_bytes") != TARGET_BYTES
        or identity.get("train_count") != TRAIN_COUNT
        or identity.get("registry_sha256") != registry_sha256
        or identity.get("split_sha256") != split_sha256
    ):
        raise Bh9Error("immutable directory manifest does not match the frozen benchmark inputs")
    sizes = identity.get("compressed_bytes")
    if not isinstance(sizes, Mapping) or set(sizes) != set(identities):
        raise Bh9Error("immutable directory manifest does not exactly cover registry identities")
    recomputed = select_bytes(identities, sizes)
    if identity.get("selection") != recomputed:
        raise Bh9Error("immutable directory manifest selection was altered")
    central_sha = identity.get("archive_central_sha256")
    entry_count = identity.get("archive_entry_count")
    if (
        not isinstance(central_sha, str)
        or len(central_sha) != 64
        or type(entry_count) is not int
        or entry_count <= 0
    ):
        raise Bh9Error("immutable directory manifest lacks the ZIP central-directory identity")
    return dict(identity), recomputed


def validate_labels(archive: Path, selected: Sequence[str]) -> dict[str, Any]:
    """After selection is frozen, require exactly one GEFF root per selected video."""

    wanted = {f"train/{identity.split(':', 1)[1]}.geff/zarr.json": identity for identity in selected}
    counts = {identity: 0 for identity in selected}
    try:
        with zipfile.ZipFile(archive, mode="r") as handle:
            for info in handle.infolist():
                identity = wanted.get(info.filename)
                if identity is not None:
                    counts[identity] += 1
    except (OSError, zipfile.BadZipFile) as exc:
        raise Bh9Error(f"cannot validate selected GEFF roots: {exc}") from exc
    invalid = {key: value for key, value in counts.items() if value != 1}
    if invalid:
        raise Bh9Error(f"selected GEFF roots must occur exactly once: {invalid}")
    return {
        "validated_after_selection": True,
        "selected_image_root_count": len(counts),
        "selected_geff_root_count": len(counts),
        "unselected_geff_opened": False,
    }


def validate_selected_paths(source: Path, selected: Sequence[str]) -> dict[str, Any]:
    """Validate image and GEFF roots only after the immutable selection exists."""

    if source.is_file():
        return validate_labels(source, selected)
    if not source.is_dir():
        raise Bh9Error(f"competition input does not exist: {source}")
    invalid: dict[str, list[str]] = {}
    for identity in selected:
        dataset = identity.split(":", 1)[1]
        expected = (
            source / "train" / f"{dataset}.zarr" / "0" / "zarr.json",
            source / "train" / f"{dataset}.geff" / "zarr.json",
        )
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            invalid[identity] = missing
    if invalid:
        raise Bh9Error(f"selected directory image/GEFF roots are missing: {invalid}")
    return {
        "validated_after_selection": True,
        "selected_image_root_count": len(selected),
        "selected_geff_root_count": len(selected),
        "unselected_geff_opened": False,
        "directory_structured_paths": True,
    }


def attest_inputs(
    archive: Path,
    registry: Path,
    split: Path,
    manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    for path, label in ((registry, "registry"), (split, "split")):
        if not path.is_file():
            raise Bh9Error(f"missing {label}: {path}")
    if not archive.exists():
        raise Bh9Error(f"missing competition input: {archive}")
    identities, registry_doc = load_train_registry(registry)
    registry_sha = sha256_file(registry)
    split_sha = sha256_file(split)
    frozen_identity: dict[str, Any] | None = None
    if archive.is_file():
        if manifest is not None:
            raise Bh9Error("ZIP mode does not accept a directory frozen manifest")
        inventory = image_inventory(archive, identities)
    elif archive.is_dir():
        if manifest is None or not manifest.is_file():
            raise Bh9Error("directory mode requires an immutable frozen manifest")
        frozen_identity, frozen_selection = load_frozen_manifest(
            manifest,
            registry_sha256=registry_sha,
            split_sha256=split_sha,
            identities=identities,
        )
        directory = directory_image_inventory(archive, identities)
        sizes = dict(frozen_identity["compressed_bytes"])
        inventory = {
            **directory,
            "compressed_bytes": sizes,
            "member_counts": None,
            "train_image_bytes": sum(sizes.values()),
            "central_sha256": frozen_identity["archive_central_sha256"],
            "entry_count": frozen_identity["archive_entry_count"],
            "frozen_selection_sha256": frozen_selection["selection_sha256"],
        }
    else:
        raise Bh9Error(f"competition input is neither a ZIP nor directory: {archive}")
    source = registry_doc.get("source")
    if not isinstance(source, Mapping):
        raise Bh9Error("registry has no source identity")
    if (
        source.get("zip_central_directory_sha256") != inventory["central_sha256"]
        or source.get("zip_entry_count") != inventory["entry_count"]
    ):
        raise Bh9Error("competition ZIP central-directory identity differs from the registry")

    split_doc = strict_json(split)
    if not isinstance(split_doc, Mapping):
        raise Bh9Error("split manifest must be an object")
    if split_doc.get("dataset_registry_sha256") != registry_sha:
        raise Bh9Error("split manifest references another dataset registry")
    expected_sources = split_doc.get("source_hashes")
    if not isinstance(expected_sources, Mapping):
        raise Bh9Error("split manifest does not freeze official scorer sources")
    scorer_hashes: dict[str, str] = {}
    for relative in SCORER_FILES:
        actual = sha256_file(ROOT / relative)
        if expected_sources.get(relative) != actual:
            raise Bh9Error(f"frozen official scorer source mismatch: {relative}")
        scorer_hashes[relative] = actual
    hashes = {
        "archive_central_sha256": inventory["central_sha256"],
        "registry_sha256": registry_sha,
        "split_sha256": split_sha,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "runner_sha256": sha256_file(ROOT / "scripts/run_bh.py"),
        "scoreall_sha256": sha256_file(ROOT / "scripts/scoreall.py"),
        "official_scorer_sources": scorer_hashes,
    }
    if manifest is not None:
        hashes["frozen_manifest_sha256"] = sha256_file(manifest)
        hashes["origin_registry_sha256"] = frozen_identity["origin_registry_sha256"]
        hashes["origin_split_sha256"] = frozen_identity["origin_split_sha256"]
    return inventory, hashes, identities


def runner_args(args: argparse.Namespace, selected: Sequence[str], remaining: float) -> argparse.Namespace:
    minutes = min(DEADLINE_SECONDS, remaining) / 60.0
    values = [
        "--input", str(args.archive),
        "--registry", str(args.registry),
        "--support-root", str(args.support),
        "--out-dir", str(args.out_dir),
        "--output-file", SUBMISSION,
        "--mode", "cv",
        "--split", "train",
        "--datasets", ",".join(selected),
        "--devices", ",".join(DEVICES),
        "--pair-batch-size", "1",
        "--view-batch-size", "1",
        "--deadline-min", f"{minutes:.9f}",
        "--graph-variant", run_bh.A0_GRAPH_VARIANT,
    ]
    parsed = run_bh.build_parser().parse_args(values)
    run_bh.validate_args(parsed)
    return parsed


def _a0_count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Bh9Error(f"A0 {label} must be a nonnegative integer")
    return value


def _a0_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Bh9Error(f"A0 {label} must be a lowercase SHA-256")
    return value


def _validate_a0_association(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise Bh9Error(f"A0 {label} association audit is missing")
    counts = {
        name: _a0_count(value.get(name), f"{label} association {name}")
        for name in (
            "candidate_count",
            "component_count",
            "attention_component_count",
            "fallback_component_count",
            "node_budget_fallback_component_count",
            "attention_budget_fallback_component_count",
            "unsupported_fallback_component_count",
            "attention_candidate_count",
            "fallback_candidate_count",
        )
    }
    if (
        counts["attention_component_count"] + counts["fallback_component_count"]
        != counts["component_count"]
        or counts["node_budget_fallback_component_count"]
        + counts["attention_budget_fallback_component_count"]
        + counts["unsupported_fallback_component_count"]
        != counts["fallback_component_count"]
        or counts["attention_candidate_count"] + counts["fallback_candidate_count"]
        != counts["candidate_count"]
    ):
        raise Bh9Error(f"A0 {label} attention/fallback accounting does not balance")
    denominators = {
        "attention_component_fraction": max(counts["component_count"], 1),
        "fallback_component_fraction": max(counts["component_count"], 1),
        "attention_candidate_fraction": max(counts["candidate_count"], 1),
        "fallback_candidate_fraction": max(counts["candidate_count"], 1),
    }
    numerators = {
        "attention_component_fraction": counts["attention_component_count"],
        "fallback_component_fraction": counts["fallback_component_count"],
        "attention_candidate_fraction": counts["attention_candidate_count"],
        "fallback_candidate_fraction": counts["fallback_candidate_count"],
    }
    for name, denominator in denominators.items():
        try:
            observed = float(value.get(name))
        except (TypeError, ValueError, OverflowError) as exc:
            raise Bh9Error(f"A0 {label} {name} is invalid") from exc
        expected = numerators[name] / denominator
        if not math.isfinite(observed) or not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise Bh9Error(f"A0 {label} {name} disagrees with its counts")


def _validate_a0_audit(
    value: Any,
    *,
    scope: str,
    label: str,
    video_count: int,
    frame_count: int,
    pair_count: int,
    weight_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != run_bh.A0_AUDIT_SCHEMA:
        raise Bh9Error(f"A0 {label} audit schema is missing or invalid")
    if (
        value.get("scope") != scope
        or _a0_count(value.get("video_count"), f"{label} video_count") != video_count
        or _a0_count(value.get("frame_count"), f"{label} frame_count") != frame_count
        or _a0_count(value.get("pair_count"), f"{label} pair_count") != pair_count
    ):
        raise Bh9Error(f"A0 {label} audit coverage is incomplete")
    checkpoint = value.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or _a0_sha(checkpoint.get("weight_sha256"), f"{label} checkpoint weight")
        != weight_sha256
        or checkpoint.get("factory") != "A0CheckpointScorer.from_loaded_model"
        or checkpoint.get("full_frame_dense_matrix") is not False
    ):
        raise Bh9Error(f"A0 {label} checkpoint identity does not match preflight")
    _a0_sha(checkpoint.get("architecture_sha256"), f"{label} checkpoint architecture")
    _a0_sha(checkpoint.get("frozen_subset_sha256"), f"{label} frozen checkpoint subset")

    gate = value.get("gate")
    if not isinstance(gate, Mapping):
        raise Bh9Error(f"A0 {label} gate audit is missing")
    candidate_count = _a0_count(gate.get("candidate_count"), f"{label} candidate_count")
    if (
        gate.get("algorithm") != "physical-cKDTree"
        or float(gate.get("radius_um", float("nan"))) != run_bh.A0_GATE_RADIUS_UM
        or gate.get("neighbor_cap") != run_bh.A0_GATE_NEIGHBOR_CAP
        or gate.get("scale_zyx") != list(run_bh.IMAGE_SUPPORT_SCALE_ZYX)
        or gate.get("full_frame_dense_matrix") is not False
    ):
        raise Bh9Error(f"A0 {label} sparse gate provenance is invalid")

    probabilities = value.get("probabilities")
    association = value.get("association")
    _validate_a0_association(association, label)
    association_count = _a0_count(
        association.get("candidate_count"),  # type: ignore[union-attr]
        f"{label} association candidate_count",
    )
    if (
        not isinstance(probabilities, Mapping)
        or probabilities.get("scope") != "original-candidate-coo-only"
        or probabilities.get("finite_unit_interval") is not True
        or _a0_count(
            probabilities.get("candidate_count"), f"{label} probability candidate_count"
        )
        != candidate_count
        or _a0_count(
            probabilities.get("unique_pair_count"), f"{label} unique probability count"
        )
        != candidate_count
        or association_count != candidate_count
    ):
        raise Bh9Error(f"A0 {label} original-COO probability audit is incomplete")

    if scope == "video":
        cache = value.get("cache")
        graph = value.get("graph")
        if (
            not isinstance(cache, Mapping)
            or cache.get("record_types") != ["FrameDet", "PairContext"]
            or _a0_count(cache.get("detection_count"), f"{label} cache detections")
            != frame_count
            or _a0_count(cache.get("context_count"), f"{label} cache contexts")
            != pair_count
            or cache.get("complete_adjacent_coverage") is not True
            or type(cache.get("reused")) is not bool
            or not isinstance(graph, Mapping)
            or graph.get("variant") != run_bh.A0_GRAPH_VARIANT
            or _a0_count(
                graph.get("learned_probability_count"), f"{label} learned probabilities"
            )
            != candidate_count
            or graph.get("full_frame_dense_matrix") is not False
        ):
            raise Bh9Error(f"A0 {label} cache/graph audit is incomplete")
    return value


def validate_runner(
    report: Mapping[str, Any],
    selected: Sequence[str],
    out_dir: Path,
) -> dict[str, Any]:
    expected = set(selected)
    if len(selected) != FROZEN_VIDEO_COUNT or len(expected) != FROZEN_VIDEO_COUNT:
        raise Bh9Error(f"A0 benchmark requires exactly {FROZEN_VIDEO_COUNT} unique videos")
    shards = report.get("whole_video_shards")
    flat = [str(item) for shard in shards for item in shard] if isinstance(shards, list) else []
    view = report.get("view_execution")
    preflight = report.get("preflight")
    checkpoint = preflight.get("checkpoint") if isinstance(preflight, Mapping) else None
    preflight_hashes = checkpoint.get("hashes") if isinstance(checkpoint, Mapping) else None
    preflight_weight = (
        preflight_hashes.get("weight_sha256") if isinstance(preflight_hashes, Mapping) else None
    )
    try:
        preflight_cuda_count = int(preflight.get("cuda_count", 0))  # type: ignore[union-attr]
    except (TypeError, ValueError, OverflowError):
        preflight_cuda_count = 0
    if (
        report.get("ok") is not True
        or report.get("mode") != "cv"
        or report.get("devices") != list(DEVICES)
        or report.get("model_loads") != 2
        or report.get("graph_variant") != run_bh.A0_GRAPH_VARIANT
        or report.get("requested_identities") != sorted(selected)
        or not isinstance(shards, list)
        or len(shards) != 2
        or any(not isinstance(shard, list) or not shard for shard in shards)
        or len(flat) != len(expected)
        or len(set(flat)) != len(flat)
        or set(flat) != expected
        or not isinstance(view, Mapping)
        or view.get("mode") != "exact-sequential"
        or view.get("pair_batch_size") != 1
        or view.get("view_batch_size") != 1
        or view.get("exact_code_path_control") is not True
        or not isinstance(preflight, Mapping)
        or preflight.get("ok") is not True
        or preflight_cuda_count < 2
    ):
        raise Bh9Error("A0 runner report violates dual-GPU/sparse/coverage contract")
    preflight_weight = _a0_sha(preflight_weight, "preflight checkpoint weight")

    workers = report.get("workers")
    if (
        not isinstance(workers, list)
        or len(workers) != 2
        or any(not isinstance(item, Mapping) or item.get("ok") is not True for item in workers)
        or {item.get("device") for item in workers} != set(DEVICES)
        or {item.get("rank") for item in workers} != {0, 1}
    ):
        raise Bh9Error("A0 runner did not complete exactly two successful workers")
    if any(
        not isinstance(worker.get("results"), list)
        or any(not isinstance(item, Mapping) for item in worker["results"])
        for worker in workers
    ):
        raise Bh9Error("A0 worker result coverage is malformed")
    results = [item for worker in workers for item in worker["results"]]
    if {item.get("identity") for item in results} != expected or len(results) != len(expected):
        raise Bh9Error("A0 per-video results do not exactly cover the frozen selection")
    per_video: list[dict[str, Any]] = []
    video_audits: list[Mapping[str, Any]] = []
    for item in sorted(results, key=lambda value: str(value["identity"])):
        identity = str(item["identity"])
        projection = item.get("image_support_projection")
        try:
            projection_limit = float(projection.get("limit_um"))  # type: ignore[union-attr]
            projection_epsilon = float(projection.get("epsilon_um"))  # type: ignore[union-attr]
            projection_count = int(projection.get("projected_nodes"))  # type: ignore[union-attr]
            projection_maximum = float(projection.get("max_correction_um"))  # type: ignore[union-attr]
            projection_nodes = int(projection.get("node_count"))  # type: ignore[union-attr]
            projection_edges = int(projection.get("edge_count"))  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise Bh9Error(f"invalid image-support projection evidence for {identity}") from exc
        if (
            not isinstance(projection, Mapping)
            or projection.get("boundary") != "post-smoothing/public-output"
            or projection.get("scale_zyx") != [1.625, 0.40625, 0.40625]
            or not all(math.isfinite(value) for value in (
                projection_limit, projection_epsilon, projection_maximum
            ))
            or projection_limit != 3.2
            or projection_epsilon < 0.0
            or projection_count < 0
            or projection_count > int(item["nodes"])
            or projection_maximum < 0.0
            or projection_maximum > projection_limit + projection_epsilon
            or projection_nodes != int(item["nodes"])
            or projection_edges != int(item["edges"])
            or projection.get("node_count_preserved") is not True
            or projection.get("topology_preserved") is not True
        ):
            raise Bh9Error(f"invalid image-support projection evidence for {identity}")
        slug = identity.replace(":", "__")
        status_path = out_dir / "status" / f"{slug}.json"
        tx_path = out_dir / "transactions" / f"{slug}.json"
        status = run_bh.strict_json_load(status_path)
        if not isinstance(status, Mapping) or status.get("ok") is not True or status.get("identity") != identity:
            raise Bh9Error(f"invalid transactional status for {identity}")
        run_contract = status.get("run_contract")
        graph_config = run_contract.get("graph_config") if isinstance(run_contract, Mapping) else None
        if (
            not isinstance(run_contract, Mapping)
            or run_contract.get("graph_variant") != run_bh.A0_GRAPH_VARIANT
            or graph_config != run_bh.GRAPH_VARIANTS[run_bh.A0_GRAPH_VARIANT]
            or item.get("graph_variant") != run_bh.A0_GRAPH_VARIANT
            or item.get("run_contract_sha256") != run_contract.get("sha256")
            or run_contract.get("sha256")
            != run_bh.canonical_sha256(
                {key: value for key, value in run_contract.items() if key != "sha256"}
            )
        ):
            raise Bh9Error(f"non-A0, non-sparse, or unlocked graph contract for {identity}")
        video_audit = _validate_a0_audit(
            item.get("a0_audit"),
            scope="video",
            label=identity,
            video_count=1,
            frame_count=FROZEN_FRAMES_PER_VIDEO,
            pair_count=FROZEN_FRAMES_PER_VIDEO - 1,
            weight_sha256=preflight_weight,
        )
        cache_audit = video_audit["cache"]
        if cache_audit.get("reused") is not item.get("cache_reused"):
            raise Bh9Error(f"A0 cache reuse audit disagrees for {identity}")
        fragment = Path(str(item.get("fragment", "")))
        feature = Path(str(item.get("feature_summary", "")))
        cache_path = out_dir / "cache" / f"{slug}.bhc"
        cache_meta_path = out_dir / "cache" / f"{slug}.json"
        for path, label in (
            (fragment, "fragment"),
            (feature, "feature"),
            (tx_path, "transaction"),
            (cache_path, "frozen cache"),
            (cache_meta_path, "frozen cache metadata"),
        ):
            if not path.is_file():
                raise Bh9Error(f"missing {label} output for {identity}: {path}")
        transaction = run_bh.strict_json_load(tx_path)
        artifacts = transaction.get("artifacts") if isinstance(transaction, Mapping) else None
        if (
            not isinstance(transaction, Mapping)
            or transaction.get("schema") != run_bh.TRANSACTION_SCHEMA
            or transaction.get("identity") != identity
            or transaction.get("run_contract") != run_contract
            or not isinstance(artifacts, Mapping)
            or artifacts.get("fragment") != str(fragment)
            or artifacts.get("feature") != str(feature)
            or artifacts.get("status") != str(status_path)
        ):
            raise Bh9Error(f"invalid transaction contract for {identity}")
        feature_payload = run_bh.strict_json_load(feature)
        if (
            not isinstance(feature_payload, Mapping)
            or feature_payload.get("identity") != identity
            or feature_payload.get("graph_variant") != run_bh.A0_GRAPH_VARIANT
            or feature_payload.get("run_contract") != run_contract
            or feature_payload.get("a0_audit") != video_audit
            or status.get("a0_audit") != video_audit
        ):
            raise Bh9Error(f"incomplete A0 video audit publication for {identity}")
        cache_meta = run_bh.strict_json_load(cache_meta_path)
        cache_contract = cache_meta.get("contract") if isinstance(cache_meta, Mapping) else None
        cache_model = cache_contract.get("model") if isinstance(cache_contract, Mapping) else None
        if (
            not isinstance(cache_meta, Mapping)
            or cache_meta.get("schema") != run_bh.CACHE_META_SCHEMA
            or cache_meta.get("cache_path") != str(cache_path)
            or cache_meta.get("cache_sha256") != sha256_file(cache_path)
            or not isinstance(cache_contract, Mapping)
            or cache_contract.get("sha256") != item.get("cache_contract_sha256")
            or cache_contract.get("sha256") != run_contract.get("inference_sha256")
            or cache_contract.get("sha256")
            != run_bh.canonical_sha256(
                {key: value for key, value in cache_contract.items() if key != "sha256"}
            )
            or not isinstance(cache_model, Mapping)
            or cache_model.get("weight_sha256") != preflight_weight
            or video_audit["checkpoint"].get("weight_sha256")
            != cache_model.get("weight_sha256")
        ):
            raise Bh9Error(f"A0 frozen cache/preflight checkpoint identity mismatch for {identity}")
        fragment_sha = sha256_file(fragment)
        feature_sha = sha256_file(feature)
        if status.get("fragment_sha256") != fragment_sha or status.get("feature_sha256") != feature_sha:
            raise Bh9Error(f"transactional output hash mismatch for {identity}")
        per_video.append(
            {
                "identity": identity,
                "seconds": float(item["seconds"]),
                "nodes": int(item["nodes"]),
                "edges": int(item["edges"]),
                "rows": int(item["rows"]),
                "resumed": bool(item["resumed"]),
                "image_support_projection": dict(projection),
                "fragment": {
                    "path": str(fragment),
                    "bytes": fragment.stat().st_size,
                    "sha256": fragment_sha,
                },
                "feature_sha256": feature_sha,
                "transaction_sha256": sha256_file(tx_path),
                "cache": {
                    "path": str(cache_path),
                    "sha256": sha256_file(cache_path),
                    "metadata_path": str(cache_meta_path),
                    "metadata_sha256": sha256_file(cache_meta_path),
                    "checkpoint_weight_sha256": cache_model["weight_sha256"],
                },
                "run_contract_sha256": str(item["run_contract_sha256"]),
                "a0_audit": dict(video_audit),
            }
        )
        video_audits.append(video_audit)

    worker_audits: list[dict[str, Any]] = []
    scorer_instances = 0
    for worker in sorted(workers, key=lambda value: int(value["rank"])):
        worker_results = worker["results"]
        rank = int(worker["rank"])
        device = str(worker["device"])
        audit = _validate_a0_audit(
            worker.get("a0_audit"),
            scope="worker",
            label=f"worker {rank}",
            video_count=len(worker_results),
            frame_count=len(worker_results) * FROZEN_FRAMES_PER_VIDEO,
            pair_count=len(worker_results) * (FROZEN_FRAMES_PER_VIDEO - 1),
            weight_sha256=preflight_weight,
        )
        worker_record = audit.get("worker")
        if (
            not isinstance(worker_record, Mapping)
            or worker_record.get("rank") != rank
            or worker_record.get("device") != device
            or worker_record.get("scorer_instances") != 1
        ):
            raise Bh9Error(f"A0 worker {rank} scorer lifecycle audit is invalid")
        scorer_instances += int(worker_record["scorer_instances"])
        child_audits = [
            item["a0_audit"]
            for item in worker_results
        ]
        try:
            expected_worker_audit = run_bh.aggregate_a0_audits(child_audits, scope="worker")
        except Exception as exc:
            raise Bh9Error(f"A0 worker {rank} video audits cannot be aggregated") from exc
        expected_worker_audit["worker"] = dict(worker_record)
        if audit != expected_worker_audit:
            raise Bh9Error(f"A0 worker {rank} audit differs from its video audits")
        worker_audits.append(dict(audit))

    run_audit = _validate_a0_audit(
        report.get("a0_audit"),
        scope="run",
        label="run",
        video_count=FROZEN_VIDEO_COUNT,
        frame_count=FROZEN_FRAME_COUNT,
        pair_count=FROZEN_ADJACENT_PAIR_COUNT,
        weight_sha256=preflight_weight,
    )
    run_workers = run_audit.get("workers")
    if (
        not isinstance(run_workers, Mapping)
        or run_workers.get("worker_count") != 2
        or run_workers.get("scorer_instances") != 2
        or scorer_instances != 2
    ):
        raise Bh9Error("A0 run must attest exactly two scorer instances total")
    try:
        expected_run_audit = run_bh.aggregate_a0_audits(video_audits, scope="run")
    except Exception as exc:
        raise Bh9Error("A0 run video audits cannot be aggregated") from exc
    expected_run_audit["workers"] = dict(run_workers)
    if run_audit != expected_run_audit:
        raise Bh9Error("A0 run audit differs from the complete video audit set")

    submission = report.get("submission")
    submission_path = out_dir / SUBMISSION
    if (
        not isinstance(submission, Mapping)
        or submission.get("datasets") != len(expected)
        or not submission_path.is_file()
        or submission.get("sha256") != sha256_file(submission_path)
    ):
        raise Bh9Error("merged submission coverage/hash is invalid")
    run_path = out_dir / "run.json"
    if not run_path.is_file():
        raise Bh9Error("runner did not atomically publish run.json")
    if run_bh.strict_json_load(run_path) != report:
        raise Bh9Error("atomic run.json content differs from the returned runner report")
    return {
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "devices": list(DEVICES),
        "model_loads": 2,
        "whole_video_shards": shards,
        "graph_variant": report["graph_variant"],
        "a0_audit": dict(run_audit),
        "worker_audits": worker_audits,
        "a0_validation": {name: True for name in A0_EVIDENCE_GATES},
        "checkpoint_weight_identity": {
            "preflight_weight_sha256": preflight_weight,
            "frozen_cache_weight_sha256": preflight_weight,
            "a0_audit_weight_sha256": run_audit["checkpoint"]["weight_sha256"],
            "matched": True,
        },
        "no_dense_nxn": True,
        "dense_guard": "run_bh rejects full_frame_dense_matrix unless false before publication",
        "transactional_per_video": True,
        "run_json": {"path": str(run_path), "sha256": sha256_file(run_path)},
        "submission": dict(submission),
        "per_video": per_video,
    }


def score_selected(
    archive: Path,
    fragments: Path,
    selected: Sequence[str],
) -> dict[str, Any]:
    try:
        import scoreall
        from zarr.storage import LocalStore, ZipStore
    except (ImportError, ModuleNotFoundError) as exc:
        raise Bh9Error(f"official scorer dependency is unavailable: {exc}") from exc
    paths = sorted(path for path in fragments.iterdir() if path.is_file() and path.suffix == ".csv")
    try:
        manifest = scoreall.validate_fragment_set(paths, selected)
        if archive.is_file():
            store = ZipStore(archive, mode="r")
            store_kind = "ZipStore"
        elif archive.is_dir():
            store = LocalStore(archive, read_only=True)
            store_kind = "LocalStore(read_only=True)"
        else:
            raise Bh9Error(f"official scorer input disappeared: {archive}")
        try:
            rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        by_identity = {row["identity"]: row for row in rows}
        if set(by_identity) != set(selected) or len(rows) != len(selected):
            raise Bh9Error("official scorer returned partial or duplicate identity coverage")
        rows = [by_identity[identity] for identity in selected]
        summary = scoreall.official_summary(rows)
    except Bh9Error:
        raise
    except Exception as exc:
        raise Bh9Error(f"official scorer failed closed: {type(exc).__name__}: {exc}") from exc
    score = summary.get("score")
    if (
        summary.get("n") != len(selected)
        or summary.get("n_adj") != len(selected)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise Bh9Error("official scorer produced incomplete or non-finite aggregate coverage")
    for row in rows:
        if (
            row.get("identity") not in set(selected)
            or row.get("full_video") is not True
            or row.get("cropped") is not False
            or int(row.get("ground_truth_nodes", 0)) <= 0
            or not math.isfinite(float(row.get("adj_edge_jaccard", float("nan"))))
        ):
            raise Bh9Error(f"invalid per-video official score row: {row.get('identity')}")
    return {
        "summary": summary,
        "samples": rows,
        "input_provenance": provenance,
        "coverage": coverage,
        "only_selected_geff_labels": True,
        "full_video": True,
        "competition_parity": True,
        "zarr_store": store_kind,
        "read_only_store": True,
        "nonfinite_metric_encoding": "null",
    }


def build_evidence(
    args: argparse.Namespace,
    selection: Mapping[str, Any],
    hashes: Mapping[str, Any],
    label_check: Mapping[str, Any],
    runner: Mapping[str, Any],
    score: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    aggregate = score["summary"]
    score_passed = float(aggregate["score"]) > MIN_SCORE
    runtime_passed = elapsed <= DEADLINE_SECONDS
    a0_validation = runner.get("a0_validation")
    a0_audit = runner.get("a0_audit")
    audit_workers = a0_audit.get("workers") if isinstance(a0_audit, Mapping) else None
    checkpoint_identity = runner.get("checkpoint_weight_identity")
    per_video = runner.get("per_video")
    worker_audits = runner.get("worker_audits")
    a0_passed = (
        isinstance(a0_validation, Mapping)
        and all(a0_validation.get(name) is True for name in A0_EVIDENCE_GATES)
        and runner.get("graph_variant") == run_bh.A0_GRAPH_VARIANT
        and isinstance(a0_audit, Mapping)
        and a0_audit.get("scope") == "run"
        and a0_audit.get("video_count") == FROZEN_VIDEO_COUNT
        and a0_audit.get("frame_count") == FROZEN_FRAME_COUNT
        and a0_audit.get("pair_count") == FROZEN_ADJACENT_PAIR_COUNT
        and audit_workers == {"worker_count": 2, "scorer_instances": 2}
        and isinstance(per_video, list)
        and len(per_video) == FROZEN_VIDEO_COUNT
        and isinstance(worker_audits, list)
        and len(worker_audits) == 2
        and isinstance(checkpoint_identity, Mapping)
        and checkpoint_identity.get("matched") is True
    )
    passed = score_passed and runtime_passed and a0_passed
    evidence = {
        "schema": "celltrack-alpha.bh9gb.v1",
        "kind": "leakage_safe_9gb_v45_benchmark",
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "minimum_score": MIN_SCORE,
        "deadline_seconds": DEADLINE_SECONDS,
        "elapsed_seconds": elapsed,
        "runtime_attestation": {
            "level": RUNTIME_ATTESTATION_LEVEL,
            "scope": "fixed_22_video_9gb_two_t4_end_to_end_official_score",
            "maximum_seconds": DEADLINE_SECONDS,
            "elapsed_seconds": elapsed,
            "passed": runtime_passed,
        },
        "archive": {
            "path": str(args.archive.resolve()),
            "bytes": args.archive.stat().st_size if args.archive.is_file() else None,
            "input_kind": "zip" if args.archive.is_file() else "directory",
            "central_directory_sha256": hashes["archive_central_sha256"],
        },
        "hashes": dict(hashes),
        "selection": dict(selection),
        "leakage_contract": {
            "selection_allowed_inputs": [
                "registry train video identity",
                "image Zarr ZIP member path",
                "image Zarr ZipInfo.compress_size",
                "frozen seed",
            ],
            "selection_forbidden_inputs": ["GEFF labels", "GEFF metadata", "scores"],
            "labels_opened_during_selection": False,
            **dict(label_check),
        },
        "runner": dict(runner),
        "official_score": dict(score),
        "gates": {
            "exact_registry_train_count": True,
            "target_bytes_reached": True,
            "whole_video_selection": True,
            "dual_gpu": True,
            "no_dense_nxn": True,
            "transactional_per_video": True,
            "complete_score_coverage": True,
            "graph_validity": True,
            **{
                name: bool(a0_validation.get(name) is True)
                if isinstance(a0_validation, Mapping)
                else False
                for name in A0_EVIDENCE_GATES
            },
            "deadline_met": runtime_passed,
            "score_strictly_above_0_90": score_passed,
        },
    }
    return json_safe(evidence)


def execute(
    args: argparse.Namespace,
    limit: HardLimit,
    *,
    runner_fn: Callable[[argparse.Namespace], Mapping[str, Any]] = run_bh.run_pipeline,
    scorer_fn: Callable[[Path, Path, Sequence[str]], Mapping[str, Any]] = score_selected,
    state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = {} if state is None else state
    started = time.monotonic()
    state["stage"] = "input_attestation"
    inventory, hashes, identities = attest_inputs(
        args.archive, args.registry, args.split, getattr(args, "manifest", None)
    )
    state["hashes"] = hashes
    state["archive"] = {
        "path": str(args.archive.resolve()),
        "bytes": args.archive.stat().st_size if args.archive.is_file() else None,
        "input_kind": "zip" if args.archive.is_file() else "directory",
        "central_directory_sha256": hashes["archive_central_sha256"],
    }
    state["stage"] = "selection"
    selection = select_bytes(identities, inventory["compressed_bytes"])
    frozen_selection_sha = inventory.get("frozen_selection_sha256")
    if frozen_selection_sha is not None and frozen_selection_sha != selection["selection_sha256"]:
        raise Bh9Error("directory selection differs from the immutable frozen manifest")
    state["selection"] = selection
    if args.select_only:
        return selection, {"hashes": hashes}
    selected = selection["selected_video_ids"]
    state["stage"] = "label_root_validation"
    label_check = validate_selected_paths(args.archive, selected)
    state["label_validation"] = label_check
    state["stage"] = "v45_runner"
    parsed = runner_args(args, selected, limit.remaining("V45 runner"))
    runner_raw = runner_fn(parsed)
    runner = validate_runner(runner_raw, selected, args.out_dir.resolve())
    state["runner"] = runner
    state["stage"] = "official_scorer"
    limit.remaining("official scoring")
    score = scorer_fn(args.archive.resolve(), args.out_dir.resolve() / "fragments", selected)
    state["official_score"] = score
    elapsed = time.monotonic() - started
    state["stage"] = "evidence_publication"
    limit.remaining("evidence publication")
    evidence = build_evidence(args, selection, hashes, label_check, runner, score, elapsed)
    if not evidence["passed"]:
        raise Bh9Rejected(
            f"official aggregate score {score['summary']['score']} is not above {MIN_SCORE}",
            evidence,
        )
    return evidence, {"hashes": hashes, "selection": selection}


def failure_evidence(
    args: argparse.Namespace,
    exc: BaseException,
    elapsed: float,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    progress = dict(state or {})
    return json_safe({
        "schema": "celltrack-alpha.bh9gb.v1",
        "kind": "leakage_safe_9gb_v45_benchmark",
        "status": "failed_closed",
        "passed": False,
        "deadline_seconds": DEADLINE_SECONDS,
        "elapsed_seconds": elapsed,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "success_evidence": str(args.evidence.resolve()),
        "stage": progress.pop("stage", "startup"),
        **progress,
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--failure", type=Path, default=FAILURE)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--select-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.out_dir, "output directory"),
        (args.evidence, "evidence"),
        (args.failure, "failure evidence"),
    ):
        short_created_name(path, label)
    if args.evidence.resolve() == args.failure.resolve():
        raise Bh9Error("success and failure evidence paths must differ")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    state: dict[str, Any] = {}
    try:
        validated = False
        validate_args(args)
        validated = True
        if not args.select_only and (args.evidence.exists() or args.failure.exists()):
            raise Bh9Error("refusing to overwrite immutable benchmark evidence")
        with HardLimit() as limit:
            result, _state = execute(args, limit, state=state)
            if not args.select_only:
                run_bh.atomic_create_json(args.evidence, result, time.time() + limit.remaining("evidence write"))
                limit.remaining("post-evidence fsync")
        if args.select_only:
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        print(
            json.dumps(
                {
                    "evidence": str(args.evidence),
                    "sha256": sha256_file(args.evidence),
                    "selected_count": result["selection"]["selected_count"],
                    "selected_compressed_bytes": result["selection"]["selected_compressed_bytes"],
                    "score": result["official_score"]["summary"]["score"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Bh9Rejected as exc:
        try:
            run_bh.atomic_create_json(args.failure, json_safe(exc.evidence))
            failure_path = str(args.failure)
        except Exception as write_exc:
            failure_path = f"unwritten ({type(write_exc).__name__}: {write_exc})"
        print(f"bh9gb: {exc}; rejection evidence={failure_path}", file=sys.stderr)
        return 3
    except (Exception, KeyboardInterrupt) as exc:
        failure_path = "unwritten"
        if validated and not args.select_only and not args.failure.exists():
            try:
                run_bh.atomic_create_json(
                    args.failure,
                    failure_evidence(args, exc, time.monotonic() - started, state),
                )
                failure_path = str(args.failure)
            except Exception as write_exc:
                failure_path = f"unwritten ({type(write_exc).__name__}: {write_exc})"
        print(
            f"bh9gb: failed closed: {type(exc).__name__}: {exc}; evidence={failure_path}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
