#!/usr/bin/env python3
"""Build a private frozen-held-out V45 D4-view diagnostic kernel.

This package produces local-scoring CSVs only.  It cannot submit to the
competition, and it only evaluates two train videos that are held out by fold
zero.  The purpose is to measure whether fewer detector TTA views retain
enough graph quality to justify an inference-speed routing policy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import pkgbh
except ModuleNotFoundError:
    import pkgbh  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "vs45pkg"
DEFAULT_NBOUT = ROOT / "kaggle_kernel" / "vs45nb"
DEFAULT_REGISTRY = ROOT / "reports" / "g02" / "dataset_registry_v1.json"
DEFAULT_SUPPORT = ROOT / "external" / "Tracking_Support"

ENTRYPOINT = "vs45.py"
MANIFEST = "vs45man.json"
CONFIG = "vs45cfg.json"
NOTEBOOK = "vs45.ipynb"
KERNEL_SLUG = "bh-v45d4nb"
KERNEL_REF = f"onebitbrain/{KERNEL_SLUG}"
DEADLINE_SECONDS = 55 * 60
BATCH_PROFILE_DEADLINE_SECONDS = 15 * 60
TARGETS = ("44b6_24264f12", "6bba_0c7fa718")
VIEW_SETS = (("v1", (0,)), ("v2", (0, 4)), ("v4", (0, 1, 4, 5)), ("v8", tuple(range(8))))
BATCH_PROFILE_VIEWS = (("v8b1", tuple(range(8)), 1), ("v8b8", tuple(range(8)), 8))
BATCH_PROFILE_SLUG = "bh-v45b8"


class PackageError(RuntimeError):
    """The diagnostic package cannot be built with its declared controls."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _kernel_ref(kernel_slug: str) -> str:
    if not isinstance(kernel_slug, str) or not kernel_slug or len(kernel_slug) > 15:
        raise PackageError("kernel slug must be a nonempty basename of at most 15 characters")
    return f"onebitbrain/{kernel_slug}"


def _metadata(kernel_slug: str = KERNEL_SLUG) -> bytes:
    payload = {
        "id": _kernel_ref(kernel_slug),
        "title": kernel_slug,
        "code_file": ENTRYPOINT,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [pkgbh.COMPETITION_SLUG],
        "dataset_sources": [pkgbh.SUPPORT_REF],
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _load_registry(registry: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        rows = payload["records"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read dataset registry: {exc}") from exc
    if not isinstance(rows, list):
        raise PackageError("dataset registry records must be a list")
    selected = []
    for target in TARGETS:
        matches = [row for row in rows if isinstance(row, Mapping) and row.get("split") == "train" and row.get("dataset") == target]
        if len(matches) != 1:
            raise PackageError(f"expected one held-out train registry record for {target!r}, found {len(matches)}")
        row = matches[0]
        metadata = row.get("compact_metadata")
        if not isinstance(metadata, Mapping):
            raise PackageError(f"held-out record {target!r} lacks compact metadata")
        shape = metadata.get("image_shape")
        if shape != [100, 64, 256, 256]:
            raise PackageError(f"held-out record {target!r} has an unexpected image shape: {shape!r}")
        if row.get("video_id") != f"train:{target}" or not isinstance(row.get("exact_identity_sha256"), str):
            raise PackageError(f"held-out record {target!r} has invalid identity fields")
        selected.append(
            {
                "split": "train",
                "dataset": target,
                "video_id": row["video_id"],
                "exact_identity_sha256": row["exact_identity_sha256"],
                "image_shape": shape,
                "group_id": row.get("group_id"),
            }
        )
    return selected


def _normalize_view_sets(
    view_sets: Sequence[Sequence[Any]] | None,
) -> tuple[tuple[str, tuple[int, ...], int, bool], ...]:
    """Validate named D4 view sets, preserving whether a batch is explicit."""

    raw_sets = VIEW_SETS if view_sets is None else view_sets
    if isinstance(raw_sets, (str, bytes)) or not isinstance(raw_sets, Sequence) or not raw_sets:
        raise PackageError("view sets must be a nonempty sequence")
    normalized: list[tuple[str, tuple[int, ...], int, bool]] = []
    names: set[str] = set()
    for item in raw_sets:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) not in (2, 3):
            raise PackageError("each view set must be (name, views) or (name, views, batch size)")
        name, views = item[0], item[1]
        explicit_batch = len(item) == 3
        view_batch_size = item[2] if explicit_batch else 1
        if not isinstance(name, str) or not name or len(name) > 15 or name in names:
            raise PackageError("view-set names must be unique nonempty basenames of at most 15 characters")
        if isinstance(views, (str, bytes)) or not isinstance(views, Sequence):
            raise PackageError("view-set views must be a sequence")
        canonical = tuple(sorted(views))
        if not canonical or canonical[0] != 0 or len(set(canonical)) != len(canonical):
            raise PackageError("view sets must include unique unaugmented view 0")
        if any(type(view) is not int or not 0 <= view < 8 for view in canonical):
            raise PackageError("view-set IDs must be integers in [0, 7]")
        if type(view_batch_size) is not int or not 1 <= view_batch_size <= len(canonical):
            raise PackageError("view batch size must be in [1, selected view count]")
        names.add(name)
        normalized.append((name, canonical, view_batch_size, explicit_batch))
    return tuple(normalized)


def _view_set_payload(view_sets: Sequence[tuple[str, tuple[int, ...], int, bool]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for name, views, view_batch_size, explicit_batch in view_sets:
        item: dict[str, Any] = {"name": name, "views": list(views)}
        if explicit_batch:
            item["view_batch_size"] = view_batch_size
        payload.append(item)
    return payload


def _config(
    registry: Path,
    *,
    view_sets: Sequence[Sequence[Any]] | None = None,
    deadline_seconds: int = DEADLINE_SECONDS,
) -> bytes:
    normalized_views = _normalize_view_sets(view_sets)
    if type(deadline_seconds) is not int or not 1 <= deadline_seconds <= DEADLINE_SECONDS:
        raise PackageError("diagnostic deadline must be a positive integer no greater than the historical cap")
    payload = {
        "schema": "bhv45-d4-view-diagnostic.v1",
        "submission_generation": False,
        "competition_submission": False,
        "network": "forbidden",
        "fold": 0,
        "target_role": "frozen-held-out",
        "targets": _load_registry(registry),
        "view_sets": _view_set_payload(normalized_views),
        "graph_variant": "a0",
        "device": "cuda:0",
        "deadline_seconds": deadline_seconds,
    }
    return _canonical(payload) + b"\n"


def _manifest(
    files: Mapping[str, bytes],
    *,
    config_sha256: str,
    kernel_slug: str,
    view_sets: Sequence[tuple[str, tuple[int, ...], int, bool]],
    deadline_seconds: int,
) -> bytes:
    file_records = [{"path": name, "bytes": len(value), "sha256": _sha(value)} for name, value in sorted(files.items())]
    package_bytes = 0
    while True:
        identity = {
            "private": True,
            "network": "forbidden",
            "publish_actions": False,
            "submission_generation": False,
            "competition_submission": False,
            "budget_bytes": pkgbh.SOURCE_LIMIT,
            "package_bytes": package_bytes,
            "package_file_count": len(files) + 1,
            "strictly_under_budget": True,
            "kernel_slug": kernel_slug,
            "diagnostic": {
                "kind": "frozen-held-out-v45-d4-view-ablation",
                "targets": list(TARGETS),
                "view_sets": _view_set_payload(view_sets),
                "graph_variant": "a0",
                "device": "cuda:0",
                "deadline_seconds": deadline_seconds,
                "config_sha256": config_sha256,
            },
            "files": file_records,
        }
        payload = {
            "schema": "bhv45-d4-view-package.v1",
            "identity": identity,
            "identity_sha256": _sha(_canonical(identity)),
        }
        encoded = _canonical(payload) + b"\n"
        observed = sum(len(value) for value in files.values()) + len(encoded)
        if observed == package_bytes:
            return encoded
        package_bytes = observed


ENTRYPOINT_SOURCE = r'''#!/usr/bin/env python3
"""Run frozen-held-out V45 D4 subsets and write local-scoring CSVs only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import bh45


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
MANIFEST = HERE / "vs45man.json"
CONFIG = HERE / "vs45cfg.json"
OUTPUT = WORK / "vs45out"


class ViewError(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key %r" % key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="ascii"), object_pairs_hook=reject)


def _verify_package():
    payload = _load(MANIFEST)
    identity = payload.get("identity")
    if not isinstance(identity, dict) or payload.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest():
        raise ViewError("view diagnostic package identity mismatch")
    observed = MANIFEST.stat().st_size
    for item in identity.get("files", []):
        path = HERE / str(item.get("path", ""))
        if not path.is_file() or path.stat().st_size != item.get("bytes") or _sha_file(path) != item.get("sha256"):
            raise ViewError("view diagnostic member identity mismatch: %s" % path.name)
        observed += path.stat().st_size
    if observed != identity.get("package_bytes") or observed >= identity.get("budget_bytes", 0):
        raise ViewError("view diagnostic package source budget mismatch")
    return payload


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload) + b"\n")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _job(bh_run, competition, item):
    record = bh_run.RegistryRecord(
        item["split"], item["dataset"], item["video_id"], item["exact_identity_sha256"], tuple(item["image_shape"])
    )
    image = competition / "train" / (record.dataset + ".zarr")
    if not image.is_dir():
        raise ViewError("held-out train image is unavailable: %s" % image)
    return bh_run.VideoJob(record, "directory", str(competition), str(image))


def _predict_one(bh_model, bh_graph, bh_run, model, bundle, scorer, job, views, view_batch_size):
    handle = bh_run.open_video(job)
    try:
        q_low, q_high = bh_run.read_quantile_attrs(handle.attrs)
        frames = bh_run.read_all_frames(handle.array, 100, None)
        identity = bh_model.CacheIdentity.from_artifact(bundle, d4_views=views)
        started = time.perf_counter()
        inference = bh_model.infer_video(
            model, frames, q_low=q_low, q_high=q_high, start_frame_index=0,
            node_id_start=0, batch_size=1, view_batch_size=view_batch_size, d4_views=views,
        )
        if inference.d4_views != tuple(views):
            raise ViewError("inference D4 provenance differs from requested views")
        cache = SimpleNamespace(detections=inference.detections, contexts=inference.contexts)
        probabilities, audit = bh_run.score_a0_cache(cache, scorer, identity)
        nodes = bh_run.detections_to_graph_nodes(inference.detections, bh_graph.GraphNode)
        refiner = bh_run.LazyImageRefiner(handle.array, bh_graph.intensity_centroid_refine, deadline_epoch=None)
        graph = bh_graph.calibrate_v45_graph(
            nodes, learned_edge_probabilities=probabilities, image_refine_callback=refiner,
        )
        graph = bh_run.project_graph_result_to_image_support(
            job.record.dataset, graph, job.record.image_shape,
            scale_zyx=bh_graph.V45_GRAPH_CONFIG.scale_zyx,
            max_correction_um=bh_graph.V45_GRAPH_CONFIG.refine_max_shift_um,
        )
        rows = bh_run.graph_rows(job.record.dataset, graph, job.record.image_shape)
        return rows, {
            "seconds": time.perf_counter() - started,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "d4_views": list(inference.d4_views), "view_batch_size": view_batch_size,
            "config_sha256": identity.config_sha256,
            "a0": audit,
            "lazy_refinement": refiner.report(),
        }
    finally:
        handle.close()


def main():
    started = time.monotonic()
    diag = {"schema": "bhv45-d4-view-run.v1", "ok": False, "submission_generation": False, "network": "forbidden"}
    try:
        package = _verify_package()
        config = _load(CONFIG)
        if package["identity"]["diagnostic"].get("config_sha256") != _sha_file(CONFIG):
            raise ViewError("view configuration identity differs from signed package")
        base = bh45._verify_package()
        diag["package_identity_sha256"] = package["identity_sha256"]
        diag["base_package_identity_sha256"] = base["identity_sha256"]
        diag["dependencies_before"] = bh45._probe_dependencies()
        support, wheels = bh45._validate_attachment(diag)
        diag["dependencies_final"] = bh45._install_missing(diag["dependencies_before"], wheels, diag)
        bh45.SOURCE_ROOT = WORK / "vs45src"
        bh45._extract_source(base)
        competition = bh45._competition_root()
        if not (competition / "train").is_dir():
            raise ViewError("competition input does not expose train images")
        sys.path.insert(0, str(bh45.SOURCE_ROOT / "src"))
        sys.path.insert(0, str(bh45.SOURCE_ROOT / "scripts"))
        sys.path.insert(0, str(support / "repo" / "src"))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(bh45.SOURCE_ROOT / "src"), str(support / "repo" / "src")])
        import run_bh as bh_run
        from celltrack_alpha import bh_edge, bh_graph, bh_model
        import torch
        if not torch.cuda.is_available():
            raise ViewError("view diagnostic requires CUDA")
        model, bundle = bh_model.load_v45_model(support, device="cuda:0")
        full_identity = bh_model.CacheIdentity.from_artifact(bundle)
        scorer = bh_run.build_a0_scorer(bh_run.A0_GRAPH_VARIANT, model, bundle, full_identity, edge_module=bh_edge)
        targets = [_job(bh_run, competition, item) for item in config["targets"]]
        view_sets = [
            (
                str(item["name"]),
                tuple(int(value) for value in item["views"]),
                int(item.get("view_batch_size", 1)),
            )
            for item in config["view_sets"]
        ]
        diag.update({"checkpoint_sha256": bundle.hashes.weight_sha256, "targets": [job.identity for job in targets], "runs": {}})
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, views, view_batch_size in view_sets:
            if time.monotonic() - started >= float(config["deadline_seconds"]):
                raise ViewError("global diagnostic deadline reached before %s" % name)
            rows = []
            details = {}
            run_started = time.perf_counter()
            for job in targets:
                part, report = _predict_one(
                    bh_model, bh_graph, bh_run, model, bundle, scorer, job, views, view_batch_size
                )
                rows.extend(part)
                details[job.identity] = report
            for index, row in enumerate(rows):
                row["id"] = index
            bh_run.validate_submission_rows(rows, [job.record for job in targets])
            output = OUTPUT / (name + ".csv")
            _write_csv(output, rows, bh_run.CSV_FIELDS)
            diag["runs"][name] = {
                "views": list(views), "view_batch_size": view_batch_size,
                "seconds": time.perf_counter() - run_started,
                "csv": output.name, "csv_sha256": _sha_file(output), "rows": len(rows), "videos": details,
            }
            _write_json(OUTPUT / "vs45run.json", diag)
        diag["elapsed_seconds"] = time.monotonic() - started
        diag["ok"] = True
        _write_json(OUTPUT / "vs45run.json", diag)
        return 0
    except Exception as exc:
        diag["error"] = "%s: %s" % (type(exc).__name__, exc)
        diag["elapsed_seconds"] = time.monotonic() - started
        _write_json(OUTPUT / "vs45run.json", diag)
        print(diag["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write(output: Path, files: Mapping[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in files)
    if unexpected:
        raise PackageError(f"refusing to mix view diagnostic package with unexpected files: {unexpected}")
    for name, value in files.items():
        temporary = output / f".{name}.tmp"
        temporary.write_bytes(value)
        temporary.replace(output / name)


def build_package(
    repo_root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    registry: Path = DEFAULT_REGISTRY,
    support_root: Path = DEFAULT_SUPPORT,
    view_sets: Sequence[Sequence[Any]] | None = None,
    kernel_slug: str = KERNEL_SLUG,
    deadline_seconds: int = DEADLINE_SECONDS,
) -> dict[str, Any]:
    """Build the bounded private diagnostic package without Kaggle side effects."""

    kernel_ref = _kernel_ref(kernel_slug)
    for name in (ENTRYPOINT, MANIFEST, CONFIG, output_dir.name):
        if len(name) > 15:
            raise PackageError(f"created basename exceeds 15 characters: {name}")
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    registry = registry.resolve()
    support_root = support_root.resolve()
    # ``pkgbh`` owns the base package and intentionally rejects unknown files.
    # Remove only this builder's prior generated members before recreating it.
    for name in (ENTRYPOINT, CONFIG, MANIFEST):
        prior = output_dir / name
        if prior.is_file():
            prior.unlink()
    pkgbh.build_package(repo_root, output_dir, registry=registry, support_root=support_root)
    base_manifest = json.loads((output_dir / pkgbh.MANIFEST_NAME).read_text(encoding="ascii"))
    archive_identity = base_manifest["identity"]["archive"]
    base_files = {
        pkgbh.ENTRYPOINT_NAME: (output_dir / pkgbh.ENTRYPOINT_NAME).read_bytes(),
        pkgbh.ARCHIVE_NAME: (output_dir / pkgbh.ARCHIVE_NAME).read_bytes(),
        pkgbh.ATTACHMENT_NAME: (output_dir / pkgbh.ATTACHMENT_NAME).read_bytes(),
        pkgbh.METADATA_NAME: _metadata(kernel_slug),
    }
    base_files[pkgbh.MANIFEST_NAME] = pkgbh._manifest_payload(base_files, archive_identity)
    normalized_views = _normalize_view_sets(view_sets)
    config = _config(registry, view_sets=view_sets, deadline_seconds=deadline_seconds)
    files = {ENTRYPOINT: ENTRYPOINT_SOURCE.encode("ascii"), CONFIG: config, **base_files}
    files[MANIFEST] = _manifest(
        files,
        config_sha256=_sha(config),
        kernel_slug=kernel_slug,
        view_sets=normalized_views,
        deadline_seconds=deadline_seconds,
    )
    aggregate = sum(len(value) for value in files.values())
    if aggregate >= pkgbh.SOURCE_LIMIT:
        raise PackageError(f"aggregate submitted kernel source must be <{pkgbh.SOURCE_LIMIT} bytes, got {aggregate}")
    _write(output_dir, files)
    return {
        "schema": "bhv45-d4-view-build.v1",
        "output_dir": str(output_dir),
        "aggregate_bytes": aggregate,
        "strictly_under_budget": True,
        "kernel": kernel_ref,
        "view_sets": _view_set_payload(normalized_views),
        "deadline_seconds": deadline_seconds,
        "config_sha256": _sha(config),
        "network_or_kaggle_actions": False,
        "submission_generation": False,
    }


def build_notebook(
    package_dir: Path,
    output_dir: Path = DEFAULT_NBOUT,
    *,
    kernel_slug: str = KERNEL_SLUG,
) -> dict[str, Any]:
    """Build a self-extracting Kaggle notebook for a verified package.

    Kaggle script uploads execute only ``code_file`` and do not retain sibling
    package members.  The notebook therefore embeds every signed member and
    materializes the package under ``/kaggle/working`` before invoking it.
    """

    package_dir = package_dir.resolve()
    output_dir = output_dir.resolve()
    names = ("bh45.py", "bh45att.json", "bh45man.json", "bh45src.zip", "kernel-metadata.json", ENTRYPOINT, CONFIG, MANIFEST)
    raw_files: dict[str, str] = {}
    for name in names:
        path = package_dir / name
        if not path.is_file():
            raise PackageError(f"notebook package member is missing: {path}")
        raw_files[name] = base64.b64encode(path.read_bytes()).decode("ascii")
    encoded = json.dumps(raw_files, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    source = f'''import base64
import json
from pathlib import Path
import subprocess
import sys

root = Path("/kaggle/working/vs45pkg")
files = json.loads({json.dumps(encoded)})
root.mkdir(parents=True, exist_ok=True)
for name, value in files.items():
    (root / name).write_bytes(base64.b64decode(value, validate=True))
result = subprocess.run([sys.executable, str(root / "{ENTRYPOINT}")], check=False)
print("vs45 inner return code:", result.returncode)
if result.returncode:
    print("vs45 preserved failure diagnostics at /kaggle/working/vs45out/vs45run.json")
'''
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_bytes = _canonical(notebook) + b"\n"
    if len(notebook_bytes) >= pkgbh.SOURCE_LIMIT:
        raise PackageError(f"notebook source must be <{pkgbh.SOURCE_LIMIT} bytes, got {len(notebook_bytes)}")
    metadata = {
        "id": _kernel_ref(kernel_slug),
        "title": kernel_slug,
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [pkgbh.COMPETITION_SLUG],
        "dataset_sources": [pkgbh.SUPPORT_REF],
    }
    metadata_bytes = json.dumps(metadata, indent=2, sort_keys=True).encode("ascii") + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {NOTEBOOK: notebook_bytes, pkgbh.METADATA_NAME: metadata_bytes}
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in files)
    if unexpected:
        raise PackageError(f"refusing to mix notebook package with unexpected files: {unexpected}")
    for name, value in files.items():
        temporary = output_dir / f".{name}.tmp"
        temporary.write_bytes(value)
        temporary.replace(output_dir / name)
    return {
        "schema": "bhv45-d4-view-notebook.v1",
        "output_dir": str(output_dir),
        "notebook_bytes": len(notebook_bytes),
        "strictly_under_budget": True,
        "kernel": _kernel_ref(kernel_slug),
        "submission_generation": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--notebook-output", type=Path)
    parser.add_argument(
        "--batch-profile",
        action="store_true",
        help="build only serial/batched eight-view V45 runtime variants; requires explicit output paths",
    )
    args = parser.parse_args(argv)
    if args.batch_profile and args.output == DEFAULT_OUTPUT:
        raise PackageError("--batch-profile requires an explicit --output to preserve the historical diagnostic")
    view_sets = BATCH_PROFILE_VIEWS if args.batch_profile else None
    kernel_slug = BATCH_PROFILE_SLUG if args.batch_profile else KERNEL_SLUG
    deadline_seconds = BATCH_PROFILE_DEADLINE_SECONDS if args.batch_profile else DEADLINE_SECONDS
    report = build_package(
        ROOT,
        args.output,
        registry=args.registry,
        support_root=args.support_root,
        view_sets=view_sets,
        kernel_slug=kernel_slug,
        deadline_seconds=deadline_seconds,
    )
    if args.notebook_output is not None:
        report["notebook"] = build_notebook(args.output, args.notebook_output, kernel_slug=kernel_slug)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
