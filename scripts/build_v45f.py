#!/usr/bin/env python3
"""Build the private, bounded V45 one-pass student training package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pkgbh


OUTPUT = ROOT / "kaggle_kernel" / "v45fp4"
REGISTRY = ROOT / "reports" / "g02" / "dataset_registry_v1.json"
SPLIT = ROOT / "reports" / "g02" / "grouped_3fold_audit20_v1.json"
PROVENANCE = ROOT / "reports" / "g02" / "fit_provenance_v1.json"
SUPPORT = ROOT / "external" / "Tracking_Support"

ENTRY = "v45f.py"
BOOT = "bh45.py"
RUN = "v45run.py"
ATTACHMENT = "v45fatt.json"
ARCHIVE = "v45fsrc.zip"
MANIFEST = "v45fman.json"
METADATA = "kernel-metadata.json"  # Required by Kaggle.
RUNTIME_ROOT = "v45frun"
KERNEL = "bh-v45f4"
ACCELERATOR = "NvidiaTeslaT4"
SOURCE_LIMIT = 1_000_000
FOLD, FRACTION, LR, SEED = 0, 0.1, 1.0e-4, 45
MAX_STEPS, MAX_HOURS, MAX_CHARGE = 1_000, 0.75, 0.30

SOURCE_PATHS = (
    "scripts/train_ta.py",
    "scripts/train_v45f.py",
    "src/celltrack_alpha/bh_model.py",
    "src/celltrack_alpha/fast_data.py",
    "src/celltrack_alpha/t_adapter.py",
    "src/celltrack_alpha/v45_fast.py",
    "src/celltrack_alpha/v45_head.py",
)

# The production package initializer imports the unrelated rule-based inference
# stack.  This fit-only archive deliberately ships only the modules the compact
# trainer imports, so it needs an otherwise empty package initializer.
TRAIN_INIT = b'"""Fit-only celltrack-alpha package."""\n'


class PackageError(RuntimeError):
    """The bounded private training package cannot be constructed safely."""


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


def _load_exact(path: Path, expected: str) -> dict[str, Any]:
    if _sha_file(path) != expected:
        raise PackageError(f"immutable input hash mismatch: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PackageError(f"immutable input is not valid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise PackageError(f"immutable input is not an object: {path.name}")
    return value


def _records(registry: Path, split: Mapping[str, Any], provenance: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    from scripts.train_ta import _fold_data, resolve_fit_records, select_fit_groups

    registry_data = _load_exact(registry, "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158")
    selected = select_fit_groups(FOLD, split, provenance, FRACTION, SEED, ())
    _fit, forbidden = _fold_data(FOLD, split, provenance)
    records = resolve_fit_records(selected, registry_data, forbidden)
    result = tuple(record.json() for record in records)
    if not result or tuple(row["group_id"] for row in result) != selected:
        raise PackageError("frozen V45 fast selection is not exact")
    return result


def _projection(registry: Path, split: Mapping[str, Any], provenance: Mapping[str, Any]) -> bytes:
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


def _metadata() -> bytes:
    value = {
        "id": f"onebitbrain/{KERNEL}",
        "title": KERNEL,
        "code_file": ENTRY,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [pkgbh.COMPETITION_SLUG],
        "dataset_sources": [pkgbh.SUPPORT_REF],
    }
    return json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _archive(repo: Path, projection: bytes) -> tuple[bytes, dict[str, Any]]:
    missing = [name for name in SOURCE_PATHS if not (repo / name).is_file()]
    if missing:
        raise PackageError(f"training sources are missing: {missing}")
    members = {name: (repo / name).read_bytes() for name in SOURCE_PATHS}
    members["src/celltrack_alpha/__init__.py"] = TRAIN_INIT
    members["split.json"] = SPLIT.read_bytes()
    members["prov.json"] = PROVENANCE.read_bytes()
    members["fitrec.json"] = projection
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


def _entrypoint() -> bytes:
    return r'''#!/usr/bin/env python3
"""Run one hash-bound, private V45 D4-teacher student fit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import bh45


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
OUTPUT = WORK / "v45fout"
CONFIG = HERE / "v45fcfg.json"


class FastLaunchError(RuntimeError):
    pass


def _canon(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _load(path):
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except Exception as exc:
        raise FastLaunchError("invalid fast config") from exc
    if not isinstance(value, dict) or _canon(value) != Path(path).read_bytes().rstrip(b"\n"):
        raise FastLaunchError("fast config is not canonical")
    return value


def main():
    started = time.monotonic()
    diagnostic = {"schema": "v45f-private-run.v1", "network": "forbidden", "ok": False}
    try:
        bh45.MANIFEST = HERE / "v45fman.json"
        bh45.ATTACHMENT = HERE / "v45fatt.json"
        bh45.SOURCE_ARCHIVE = HERE / "v45fsrc.zip"
        bh45.SOURCE_ROOT = WORK / "v45fsrc"
        bh45.DIAGNOSTIC = WORK / "v45fdiag.json"
        package = bh45._verify_package()
        config = _load(CONFIG)
        if config.get("accelerator") != "NvidiaTeslaT4":
            raise FastLaunchError("private fast fit requires the NvidiaTeslaT4 accelerator")
        diagnostic["dependencies_before"] = bh45._probe_dependencies()
        support, wheels = bh45._validate_attachment(diagnostic)
        diagnostic["dependencies_final"] = bh45._install_missing(diagnostic["dependencies_before"], wheels, diagnostic)
        bh45._extract_source(package)
        competition = bh45._competition_root()
        train_root = competition / "train"
        if not train_root.is_dir():
            raise FastLaunchError("private training input lacks a train directory")
        environment = dict(os.environ)
        python_paths = [str(bh45.SOURCE_ROOT / "src"), str(support / "repo" / "src")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update({
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        })
        command = [
            sys.executable, str(bh45.SOURCE_ROOT / "scripts" / "train_v45f.py"),
            "--fold", str(config["fold"]), "--fraction", str(config["fraction"]),
            "--lr", str(config["lr"]), "--seed", str(config["seed"]),
            "--epochs", str(config["epochs"]), "--max-steps", str(config["max_steps"]),
            "--maximum-hours", str(config["maximum_hours"]), "--maximum-charge", str(config["maximum_charge"]),
            "--support-root", str(support), "--data-root", str(train_root),
            "--split-manifest", str(bh45.SOURCE_ROOT / "split.json"),
            "--provenance-manifest", str(bh45.SOURCE_ROOT / "prov.json"),
            "--record-projection", str(bh45.SOURCE_ROOT / "fitrec.json"),
            "--output-dir", str(OUTPUT), "--armed",
        ]
        deadline = started + float(config["maximum_hours"]) * 3600.0 + 300.0
        diagnostic["runner"] = {"command": command, "deadline_seconds": deadline - started, "device": "cuda:0"}
        bh45._write_diagnostic(diagnostic)
        bh45._run_with_deadline(command, environment, deadline)
        metrics = OUTPUT / "metrics.json"
        checkpoint = OUTPUT / "checkpoint.pt"
        if not metrics.is_file() or not checkpoint.is_file():
            raise FastLaunchError("bounded student fit did not emit metrics and checkpoint")
        result = _load(metrics)
        if result.get("completed_steps", 0) < 1 or result.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise FastLaunchError("bounded student artifact identity is invalid")
        diagnostic.update({"ok": True, "elapsed_seconds": time.monotonic() - started, "completed_steps": result["completed_steps"]})
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


def _wrapper(bundle: bytes, members: Sequence[str]) -> bytes:
    """Return the script Kaggle executes when it transfers only ``code_file``.

    Kaggle's script runner copies the selected code file to ``/kaggle/src`` but
    does not promise its source-directory siblings are importable.  Keep the
    private runtime in a deterministic embedded archive instead of relying on
    those siblings being transferred.  The wrapper is deliberately stdlib-only
    because it runs before the offline dependency probe in ``bh45.py``.
    """

    encoded = base64.b85encode(bundle).decode("ascii")
    expected = tuple(sorted(members))
    return f'''#!/usr/bin/env python3
"""Self-extracting private V45 fast-fit launcher for Kaggle script mode."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import os
from pathlib import Path
import runpy
import sys
import zipfile


BUNDLE_SHA256 = {hashlib.sha256(bundle).hexdigest()!r}
BUNDLE_BYTES = {len(bundle)!r}
RUNTIME_MEMBERS = {expected!r}
WORK = Path(os.environ.get("V45F_WRAP_WORK", "/kaggle/working"))
RUNTIME_ROOT = {RUNTIME_ROOT!r}


class WrapperError(RuntimeError):
    pass


def _decode_bundle():
    try:
        value = base64.b85decode({encoded!r}.encode("ascii"))
    except Exception as exc:
        raise WrapperError("embedded runtime archive cannot be decoded") from exc
    if len(value) != BUNDLE_BYTES or hashlib.sha256(value).hexdigest() != BUNDLE_SHA256:
        raise WrapperError("embedded runtime archive identity mismatch")
    return value


def _safe_member(name):
    if (
        not isinstance(name, str)
        or not name
        or name.startswith(("/", "./"))
        or "//" in name
        or "\\\\" in name
        or ":" in name
        or "\\x00" in name
    ):
        return False
    path = Path(name)
    parts = path.parts
    return (
        bool(parts)
        and not path.is_absolute()
        and not any(part in ("", ".", "..") for part in parts)
    )


def extract_runtime():
    root = WORK / RUNTIME_ROOT
    if root.exists():
        raise WrapperError("refusing to overwrite extracted runtime: %s" % root)
    data = _decode_bundle()
    try:
        with zipfile.ZipFile(BytesIO(data), "r") as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in infos]
            if tuple(sorted(names)) != RUNTIME_MEMBERS or len(names) != len(set(names)):
                raise WrapperError("embedded runtime member set mismatch")
            for item in infos:
                if not _safe_member(item.filename):
                    raise WrapperError("unsafe embedded runtime member: %r" % item.filename)
                if item.flag_bits & 0x1 or item.compress_type != zipfile.ZIP_DEFLATED:
                    raise WrapperError("unsupported embedded runtime member encoding")
                archive.read(item)
            root.mkdir(parents=True, exist_ok=False)
            for item in infos:
                target = root / item.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
    except WrapperError:
        raise
    except Exception as exc:
        raise WrapperError("embedded runtime extraction failed") from exc
    return root


def main():
    root = extract_runtime()
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / {RUN!r}), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("ascii")


def _config(projection: bytes) -> bytes:
    value = {
        "schema": "v45f-private-fit.v1",
        "private": True,
        "network": "forbidden",
        "scored_submission": False,
        "fold": FOLD,
        "fraction": FRACTION,
        "lr": LR,
        "seed": SEED,
        "epochs": 1,
        "max_steps": MAX_STEPS,
        "maximum_hours": MAX_HOURS,
        "maximum_charge": MAX_CHARGE,
        "accelerator": ACCELERATOR,
        "teacher": "exact-sequential-d4-8-logit-mean",
        "student_views": [0],
        "head_parameter_count": 1_254,
        "division_emission": False,
        "projection_sha256": _sha(projection),
        "v45_weight_sha256": pkgbh.EXPECTED_ARTIFACT_HASHES["weight_sha256"],
    }
    return _canon(value) + b"\n"


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
                "config_sha256": _sha(config),
                "fold": FOLD,
                "fraction": FRACTION,
                "maximum_hours": MAX_HOURS,
                "maximum_charge": MAX_CHARGE,
                "accelerator": ACCELERATOR,
                "teacher": "exact-sequential-d4-8-logit-mean",
                "student_views": [0],
                "division_emission": False,
            },
        }
        payload = {"schema": "v45f-private-package.v1", "identity": identity, "identity_sha256": _sha(_canon(identity))}
        encoded = _canon(payload) + b"\n"
        observed = sum(len(value) for value in files.values()) + len(encoded)
        if observed == package_bytes:
            return encoded
        package_bytes = observed


def build(output: Path = OUTPUT, *, registry: Path = REGISTRY, support: Path = SUPPORT) -> dict[str, Any]:
    for name in (output.name, ENTRY, BOOT, RUN, ATTACHMENT, ARCHIVE, MANIFEST, RUNTIME_ROOT, "v45fcfg.json"):
        if len(name) > 15:
            raise PackageError(f"created basename exceeds fifteen characters: {name}")
    split = _load_exact(SPLIT, "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769")
    provenance = _load_exact(PROVENANCE, "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f")
    projection = _projection(registry, split, provenance)
    archive, archive_identity = _archive(ROOT, projection)
    config = _config(projection)
    attachment = pkgbh._attachment_payload(support, pkgbh.EXPECTED_ARTIFACT_HASHES, support_mounts=pkgbh.SUPPORT_MOUNT_CANDIDATES)
    metadata = _metadata()
    runtime = {
        BOOT: pkgbh.ENTRYPOINT_SOURCE.encode("ascii"),
        RUN: _entrypoint(),
        ATTACHMENT: attachment,
        ARCHIVE: archive,
        "v45fcfg.json": config,
        METADATA: metadata,
    }
    manifest = _manifest(runtime, archive_identity, config)
    runtime[MANIFEST] = manifest
    bundle = pkgbh.deterministic_zip(runtime)
    files = {ENTRY: _wrapper(bundle, tuple(runtime)), METADATA: metadata}
    if sum(len(value) for value in files.values()) >= SOURCE_LIMIT:
        raise PackageError("private fast package exceeds the one megabyte source limit")
    if output.exists():
        raise PackageError(f"refusing to replace an existing package output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, value in files.items():
            (output / name).write_bytes(value)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {
        "schema": "v45f-build.v1",
        "output": str(output),
        "aggregate_bytes": sum(len(value) for value in files.values()),
        "package_sha256": _sha(manifest),
        "bundle_bytes": len(bundle),
        "bundle_sha256": _sha(bundle),
        "runtime_members": sorted(runtime),
        "projection_sha256": _sha(projection),
        "selected_sources": json.loads(projection.decode("ascii"))["selected_sources"],
        "network_or_kaggle_actions": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--support", type=Path, default=SUPPORT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, registry=args.registry, support=args.support), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
