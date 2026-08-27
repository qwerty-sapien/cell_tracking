#!/usr/bin/env python3
"""Build a private, offline Kaggle T4 package for V45 plus edge calibration."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts import build_v45f as parent
from scripts import pkgbh


OUTPUT = ROOT / "kaggle_kernel" / "e4pkg"
REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
PROVENANCE = ROOT / "reports/g02/fit_provenance_v1.json"
SUPPORT = ROOT / "external/Tracking_Support"

ENTRY = "e4fit.py"
BOOT = "e4bh.py"
RUN = "e4run.py"
ATTACHMENT = "e4att.json"
ARCHIVE = "e4src.zip"
MANIFEST = "e4man.json"
CONFIG = "e4cfg.json"
METADATA = "kernel-metadata.json"
RUNTIME_ROOT = "e4run"
KERNEL = "bh-e4f0"
SOURCE_LIMIT = 1_000_000

FOLD, FRACTION, LR, SEED = 0, 0.1, 1.0e-4, 45
MAX_STEPS, MAX_HOURS, MAX_CHARGE = 1_000, 0.75, 0.30
PARENT_SOURCE_COUNT = 12

SOURCE_PATHS = (
    "scripts/edgefit.py",
    "scripts/run_bh.py",
    "scripts/train_ta.py",
    "scripts/train_v45f.py",
    "src/celltrack_alpha/bh_edge.py",
    "src/celltrack_alpha/bh_graph.py",
    "src/celltrack_alpha/bh_model.py",
    "src/celltrack_alpha/bh_route.py",
    "src/celltrack_alpha/edge_data.py",
    "src/celltrack_alpha/fast_data.py",
    "src/celltrack_alpha/g02_integrity.py",
    "src/celltrack_alpha/t_adapter.py",
    "src/celltrack_alpha/v45_fast.py",
    "src/celltrack_alpha/v45_head.py",
    "src/celltrack_baseline/__init__.py",
    "src/celltrack_baseline/rule_based.py",
)


class PackageError(RuntimeError):
    """The two-stage private package cannot be built safely."""


def _canon(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact(path: Path, expected: str) -> Mapping[str, Any]:
    if _sha_file(path) != expected:
        raise PackageError(f"immutable input hash mismatch: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"immutable input is invalid JSON: {path.name}") from exc
    if not isinstance(value, Mapping):
        raise PackageError(f"immutable input is not an object: {path.name}")
    return value


def _records(registry: Path, split: Mapping[str, Any], provenance: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    from scripts.train_ta import _fold_data, resolve_fit_records, select_fit_groups

    registry_data = _load_exact(registry, "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158")
    selected = select_fit_groups(FOLD, split, provenance, FRACTION, SEED, ())
    _fit, forbidden = _fold_data(FOLD, split, provenance)
    records = resolve_fit_records(selected, registry_data, forbidden)
    result = tuple(record.json() for record in records)
    if len(result) != PARENT_SOURCE_COUNT or tuple(row["group_id"] for row in result) != selected:
        raise PackageError("selected parent sources are not the frozen fold-0 fit groups")
    return result


def _parent_projection(registry: Path, split: Mapping[str, Any], provenance: Mapping[str, Any]) -> bytes:
    value = {
        "schema": "celltrack-alpha.v45-fast-projection.v1",
        "source_registry_sha256": _sha_file(registry),
        "split_sha256": _sha_file(SPLIT),
        "provenance_sha256": _sha_file(PROVENANCE),
        "fold": FOLD,
        "fraction": FRACTION,
        "seed": SEED,
        "requested_groups": [],
        "selected_sources": list(_records(registry, split, provenance)),
    }
    return _canon(value) + b"\n"


def _edge_registry(registry: Path, parent_projection: bytes) -> bytes:
    try:
        source = json.loads(registry.read_text(encoding="ascii"))
        parent = json.loads(parent_projection.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError("cannot build compact edgefit registry") from exc
    rows = source.get("records") if isinstance(source, Mapping) else None
    selected = parent.get("selected_sources") if isinstance(parent, Mapping) else None
    if not isinstance(rows, list) or not isinstance(selected, list) or not selected:
        raise PackageError("edgefit registry source identities are invalid")
    selected_ids = {row.get("video_id") for row in selected if isinstance(row, Mapping)}
    if len(selected_ids) != PARENT_SOURCE_COUNT or len(selected_ids) != len(selected) or any(not isinstance(video_id, str) for video_id in selected_ids):
        raise PackageError("edgefit registry parent identities are invalid")
    by_id = {row.get("video_id"): row for row in rows if isinstance(row, Mapping) and isinstance(row.get("video_id"), str)}
    if set(by_id) & selected_ids != selected_ids:
        raise PackageError("edgefit registry parent identity is absent from the frozen registry")
    compact: list[dict[str, Any]] = []
    for video_id in sorted(selected_ids):
        row = by_id[video_id]
        metadata = row.get("compact_metadata")
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("image_shape"), list):
            raise PackageError("edgefit registry parent lacks image shape")
        compact.append({
            "split": row.get("split"),
            "dataset": row.get("dataset"),
            "video_id": row.get("video_id"),
            "exact_identity_sha256": row.get("exact_identity_sha256"),
            "compact_metadata": {"image_shape": list(metadata["image_shape"])},
            "group_id": row.get("group_id"),
        })
    if len(compact) != PARENT_SOURCE_COUNT:
        raise PackageError("edgefit registry does not contain the exact parent source count")
    value = {
        "artifact": "celltrack-alpha.edgefit-registry-projection.v1",
        "source_registry_sha256": _sha_file(registry),
        "parent_projection_sha256": _sha(parent_projection),
        "records": compact,
    }
    return _canon(value) + b"\n"


def _archive(parent_projection: bytes, edge_registry: bytes) -> tuple[bytes, dict[str, Any]]:
    missing = [name for name in SOURCE_PATHS if not (ROOT / name).is_file()]
    if missing:
        raise PackageError(f"required source files are missing: {missing}")
    members = {name: (ROOT / name).read_bytes() for name in SOURCE_PATHS}
    members["src/celltrack_alpha/__init__.py"] = parent.TRAIN_INIT
    members["split.json"] = SPLIT.read_bytes()
    members["prov.json"] = PROVENANCE.read_bytes()
    members["reg.json"] = edge_registry
    members["fitrec.json"] = parent_projection
    archive = pkgbh.deterministic_zip(members)
    identity = {
        "path": ARCHIVE,
        "bytes": len(archive),
        "sha256": _sha(archive),
        "uncompressed_bytes": sum(len(value) for value in members.values()),
        "members": [
            {"path": name, "bytes": len(members[name]), "sha256": _sha(members[name])}
            for name in sorted(members)
        ],
    }
    return archive, identity


def _config(parent_projection: bytes, edge_registry: bytes) -> bytes:
    value = {
        "schema": "edgefit-private-v1",
        "private": True,
        "network": "forbidden",
        "scored_submission": False,
        "fold": FOLD,
        "fraction": FRACTION,
        "lr": LR,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "maximum_hours": MAX_HOURS,
        "maximum_charge": MAX_CHARGE,
        "accelerator": "NvidiaTeslaT4",
        "parent_head_parameters": 1_254,
        "edge_parameters": 578,
        "edge_val_groups": 3,
        "edge_max_steps": 1_200,
        "parent_projection_sha256": _sha(parent_projection),
        "edge_registry_sha256": _sha(edge_registry),
        "v45_weight_sha256": pkgbh.EXPECTED_ARTIFACT_HASHES["weight_sha256"],
    }
    return _canon(value) + b"\n"


def _run_source() -> bytes:
    return r'''#!/usr/bin/env python3
"""Run a bounded private parent fit followed by sparse edge calibration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import e4bh as bh45


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
PARENT = WORK / "e4par"
EDGE = WORK / "e4out"
CONFIG = HERE / "e4cfg.json"


class E4LaunchError(RuntimeError):
    pass


def _canon(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _load(path):
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise E4LaunchError("invalid private config") from exc
    if not isinstance(value, dict) or _canon(value) != Path(path).read_bytes().rstrip(b"\n"):
        raise E4LaunchError("private config is not canonical")
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    diagnostic = {"schema": "edgefit-private-run.v1", "network": "forbidden", "ok": False}
    try:
        bh45.MANIFEST = HERE / "e4man.json"
        bh45.ATTACHMENT = HERE / "e4att.json"
        bh45.SOURCE_ARCHIVE = HERE / "e4src.zip"
        bh45.SOURCE_ROOT = WORK / "e4src"
        bh45.DIAGNOSTIC = WORK / "e4diag.json"
        package = bh45._verify_package()
        config = _load(CONFIG)
        if config.get("accelerator") != "NvidiaTeslaT4":
            raise E4LaunchError("private edge fit requires NvidiaTeslaT4")
        diagnostic["dependencies_before"] = bh45._probe_dependencies()
        support, wheels = bh45._validate_attachment(diagnostic)
        diagnostic["dependencies_final"] = bh45._install_missing(diagnostic["dependencies_before"], wheels, diagnostic)
        bh45._extract_source(package)
        competition = bh45._competition_root()
        if not (competition / "train").is_dir():
            raise E4LaunchError("competition training labels are unavailable")
        environment = dict(os.environ)
        python_paths = [str(bh45.SOURCE_ROOT / "src"), str(bh45.SOURCE_ROOT / "scripts"), str(support / "repo" / "src")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update({
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        })
        deadline = started + float(config["maximum_hours"]) * 3600.0 + 300.0
        parent_command = [
            sys.executable, str(bh45.SOURCE_ROOT / "scripts" / "train_v45f.py"),
            "--fold", str(config["fold"]), "--fraction", str(config["fraction"]),
            "--lr", str(config["lr"]), "--seed", str(config["seed"]),
            "--epochs", "1", "--max-steps", str(config["max_steps"]),
            "--maximum-hours", str(config["maximum_hours"]), "--maximum-charge", str(config["maximum_charge"]),
            "--support-root", str(support), "--data-root", str(competition / "train"),
            "--split-manifest", str(bh45.SOURCE_ROOT / "split.json"),
            "--provenance-manifest", str(bh45.SOURCE_ROOT / "prov.json"),
            "--record-projection", str(bh45.SOURCE_ROOT / "fitrec.json"),
            "--output-dir", str(PARENT), "--armed",
        ]
        diagnostic["parent"] = {"command": parent_command, "device": "cuda:0"}
        bh45._write_diagnostic(diagnostic)
        bh45._run_with_deadline(parent_command, environment, deadline)
        checkpoint, metrics = PARENT / "checkpoint.pt", PARENT / "metrics.json"
        if not checkpoint.is_file() or not metrics.is_file():
            raise E4LaunchError("parent fit omitted checkpoint or metrics")
        parent_metrics = _load(metrics)
        if parent_metrics.get("status") != "completed" or parent_metrics.get("completed_steps", 0) < 1:
            raise E4LaunchError("parent fit was not completed")
        edge_command = [
            sys.executable, str(bh45.SOURCE_ROOT / "scripts" / "edgefit.py"),
            "--input", str(competition), "--out-dir", str(EDGE), "--fold", str(config["fold"]),
            "--device", "cuda", "--support-root", str(support),
            "--checkpoint", str(checkpoint), "--checkpoint-sha256", _sha(checkpoint),
            "--metrics", str(metrics), "--metrics-sha256", _sha(metrics),
            "--registry", str(bh45.SOURCE_ROOT / "reg.json"),
            "--split", str(bh45.SOURCE_ROOT / "split.json"),
            "--provenance", str(bh45.SOURCE_ROOT / "prov.json"),
            "--val-groups", str(config["edge_val_groups"]),
            "--max-steps", str(config["edge_max_steps"]), "--seed", str(config["seed"]),
        ]
        diagnostic["edge"] = {"command": edge_command, "device": "cuda:0"}
        bh45._write_diagnostic(diagnostic)
        bh45._run_with_deadline(edge_command, environment, deadline)
        edge_report, edge_artifact = EDGE / "report.json", EDGE / "edge.json"
        if not edge_report.is_file() or not edge_artifact.is_file():
            raise E4LaunchError("edge fit omitted report or edge artifact")
        edge_metrics = _load(edge_report)
        if edge_metrics.get("status") != "completed":
            raise E4LaunchError("edge fit was not completed")
        diagnostic.update({
            "ok": True,
            "elapsed_seconds": time.monotonic() - started,
            "parent_checkpoint_sha256": _sha(checkpoint),
            "edge_artifact_sha256": _sha(edge_artifact),
        })
        bh45._write_diagnostic(diagnostic)
        return 0
    except Exception as exc:
        diagnostic.update({"error": "%s: %s" % (type(exc).__name__, exc), "elapsed_seconds": time.monotonic() - started})
        bh45._write_diagnostic(diagnostic)
        print(diagnostic["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("ascii")


def _metadata() -> bytes:
    value = {
        "id": f"onebitbrain/{KERNEL}",
        "title": KERNEL,
        "code_file": ENTRY,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": [pkgbh.COMPETITION_SLUG],
        "dataset_sources": [pkgbh.SUPPORT_REF],
    }
    return json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _manifest(files: Mapping[str, bytes], archive: Mapping[str, Any], config: bytes) -> bytes:
    rows = [{"path": name, "bytes": len(value), "sha256": _sha(value)} for name, value in sorted(files.items())]
    package_bytes = 0
    while True:
        identity = {
            "private": True,
            "network": "forbidden",
            "publish_actions": False,
            "scored_submission": False,
            "budget_bytes": SOURCE_LIMIT,
            "package_bytes": package_bytes,
            "package_file_count": len(files) + 1,
            "strictly_under_budget": True,
            "kernel_slug": KERNEL,
            "fixed_long_basename_exception": METADATA,
            "external_support": {"dataset_ref": pkgbh.SUPPORT_REF, "created_by_kernel_owner": False},
            "files": rows,
            "archive": dict(archive),
            "fit": {
                "fold": FOLD,
                "maximum_hours": MAX_HOURS,
                "maximum_charge": MAX_CHARGE,
                "parent_head_parameters": 1_254,
                "edge_parameters": 578,
                "config_sha256": _sha(config),
            },
        }
        payload = {"schema": "edgefit-private-package.v1", "identity": identity, "identity_sha256": _sha(_canon(identity))}
        encoded = _canon(payload) + b"\n"
        observed = sum(len(value) for value in files.values()) + len(encoded)
        if observed == package_bytes:
            return encoded
        package_bytes = observed


def build(output: Path = OUTPUT) -> dict[str, Any]:
    created = (output.name, ENTRY, BOOT, RUN, ATTACHMENT, ARCHIVE, MANIFEST, CONFIG, RUNTIME_ROOT)
    if any(len(name) > 15 for name in created):
        raise PackageError("a generated basename exceeds fifteen characters")
    split = _load_exact(SPLIT, "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769")
    provenance = _load_exact(PROVENANCE, "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f")
    parent_projection = _parent_projection(REGISTRY, split, provenance)
    edge_registry = _edge_registry(REGISTRY, parent_projection)
    archive, archive_identity = _archive(parent_projection, edge_registry)
    config = _config(parent_projection, edge_registry)
    attachment = pkgbh._attachment_payload(SUPPORT, pkgbh.EXPECTED_ARTIFACT_HASHES, support_mounts=pkgbh.SUPPORT_MOUNT_CANDIDATES)
    runtime = {
        BOOT: pkgbh.ENTRYPOINT_SOURCE.encode("ascii"),
        RUN: _run_source(),
        ATTACHMENT: attachment,
        ARCHIVE: archive,
        CONFIG: config,
        METADATA: _metadata(),
    }
    manifest = _manifest(runtime, archive_identity, config)
    runtime[MANIFEST] = manifest
    old_run, old_root = parent.RUN, parent.RUNTIME_ROOT
    try:
        parent.RUN, parent.RUNTIME_ROOT = RUN, RUNTIME_ROOT
        bundle = parent._wrapper(pkgbh.deterministic_zip(runtime), tuple(runtime))
    finally:
        parent.RUN, parent.RUNTIME_ROOT = old_run, old_root
    files = {ENTRY: bundle, METADATA: runtime[METADATA]}
    if sum(len(value) for value in files.values()) >= SOURCE_LIMIT:
        raise PackageError("private kernel source exceeds one megabyte")
    if output.exists():
        raise PackageError(f"refusing to replace existing package: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, value in files.items():
            (output / name).write_bytes(value)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {
        "schema": "edgefit-build.v1",
        "output": str(output),
        "aggregate_bytes": sum(len(value) for value in files.values()),
        "bundle_bytes": len(pkgbh.deterministic_zip(runtime)),
        "bundle_sha256": _sha(pkgbh.deterministic_zip(runtime)),
        "selected_sources": json.loads(parent_projection.decode("ascii"))["selected_sources"],
        "network_or_kaggle_actions": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
