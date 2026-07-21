#!/usr/bin/env python3
"""Build the deterministic private Kaggle package for the V45 sparse runner."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "bh45pkg"
DEFAULT_REGISTRY = ROOT / "reports" / "g02" / "dataset_registry_v1.json"
DEFAULT_SUPPORT = ROOT / "external" / "Tracking_Support"

SOURCE_LIMIT = 1_000_000
KERNEL_SLUG = "bhv45-private"
KERNEL_REF = f"onebitbrain/{KERNEL_SLUG}"
SUPPORT_DATASET_SLUG = "biohub-tracking-support-pack-50ep-v1"
SUPPORT_REF = f"pilkwang/{SUPPORT_DATASET_SLUG}"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"
SUPPORT_MOUNT_CANDIDATES = (
    f"/kaggle/input/datasets/pilkwang/{SUPPORT_DATASET_SLUG}",
    f"/kaggle/input/{SUPPORT_DATASET_SLUG}",
)

ENTRYPOINT_NAME = "bh45.py"
ARCHIVE_NAME = "bh45src.zip"
ATTACHMENT_NAME = "bh45att.json"
MANIFEST_NAME = "bh45man.json"
METADATA_NAME = "kernel-metadata.json"  # Kaggle requires this fixed basename.

EXPECTED_ARTIFACT_HASHES = {
    "manifest_sha256": "bc20f1f04cfb682af3b27a836ce9a57f44a2fe27508dc59039200b2a23188103",
    "config_sha256": "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a",
    "source_sha256": "8c4621658956f724b161236b451d7d0ed30759aa3f3a5ea28021b4eeb1171e9b",
    "weight_sha256": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
}

SUPPORT_SOURCE_PATHS = (
    "repo/src/biohub_tracking/models/temporal_unet.py",
    "repo/src/biohub_tracking/models/simple_node_transformer.py",
    "repo/scripts/train_unet_transformer.py",
)
SUPPORT_ARTIFACT_PATHS = (
    "ARTIFACT_MANIFEST.json",
    "weights/unet_transformer/split_0/config.json",
    "weights/unet_transformer/split_0/edge_predictor_best.pth",
    *SUPPORT_SOURCE_PATHS,
)
SOURCE_PATHS = (
    "scripts/run_bh.py",
    "src/celltrack_alpha/bh_model.py",
    "src/celltrack_alpha/bh_graph.py",
    "src/celltrack_alpha/bh_edge.py",
    "src/celltrack_alpha/bh_route.py",
)

DEPENDENCIES = (
    ("torch", "torch", "2.2", "3", ">=2.2,<3"),
    ("numpy", "numpy", "1.26", "3", ">=1.26,<3"),
    ("scipy", "scipy", "1.11", "2", ">=1.11,<2"),
    ("zarr", "zarr", "3.0.10", "4", ">=3.0.10,<4"),
    ("numcodecs", "numcodecs", "0.14", "0.16", ">=0.14,<0.16"),
    ("donfig", "donfig", "0.8", "1", ">=0.8,<1"),
    ("google-crc32c", "google_crc32c", "1.5", "2", ">=1.5,<2"),
    ("packaging", "packaging", "22", "27", ">=22,<27"),
    ("typing-extensions", "typing_extensions", "4.13", "5", ">=4.13,<5"),
    ("Deprecated", "deprecated", "1.2", "2", ">=1.2,<2"),
    ("wrapt", "wrapt", "1.16", "3", ">=1.16,<3"),
)

_SAFE_SLUG = re.compile(r"[a-z0-9-]+\Z")


class PackageError(RuntimeError):
    """The private package cannot be built without violating its contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise PackageError(f"required package input is missing: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_slug(slug: str, label: str) -> None:
    if len(slug) > 15 or _SAFE_SLUG.fullmatch(slug) is None:
        raise PackageError(f"{label} must be a lowercase Kaggle slug of at most 15 characters: {slug!r}")


def compact_registry(path: Path, *, expected_test_count: int = 4) -> bytes:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read source registry {path}: {exc}") from exc
    if not isinstance(source, Mapping) or source.get("artifact") != "dataset_registry_v1":
        raise PackageError("source registry is not dataset_registry_v1")
    rows = source.get("records")
    if not isinstance(rows, list):
        raise PackageError("source registry records must be a list")
    records = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("split") != "test":
            continue
        try:
            shape = [int(item) for item in row["compact_metadata"]["image_shape"]]
            record = {
                "split": "test",
                "dataset": str(row["dataset"]),
                "video_id": str(row["video_id"]),
                "exact_identity_sha256": str(row["exact_identity_sha256"]),
                "compact_metadata": {"image_shape": shape},
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise PackageError("a test registry record lacks a required runner identity field") from exc
        if record["video_id"] != f"test:{record['dataset']}" or len(shape) != 4 or any(item <= 0 for item in shape):
            raise PackageError(f"invalid compact test registry identity: {record['video_id']!r}")
        digest = record["exact_identity_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PackageError(f"invalid exact identity SHA-256 for {record['video_id']}")
        records.append(record)
    records.sort(key=lambda item: item["video_id"])
    if len(records) != expected_test_count or len({item["video_id"] for item in records}) != len(records):
        raise PackageError(
            f"compact registry requires {expected_test_count} unique test records, found {len(records)}"
        )
    payload = {
        "artifact": "dataset_registry_v1",
        "schema_version": 1,
        "source_registry_sha256": sha256_file(path),
        "counts": {"test_videos": len(records), "videos": len(records)},
        "records": records,
    }
    return canonical_json(payload) + b"\n"


def source_tree_sha256(support_root: Path) -> str:
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


def deterministic_zip(members: Mapping[str, bytes]) -> bytes:
    from io import BytesIO

    target = BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            parts = Path(name).parts
            if Path(name).is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
                raise PackageError(f"unsafe source archive member: {name!r}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return target.getvalue()


def dependency_install_specs(statuses: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return only packages whose import/version probe was not compatible."""

    result = []
    for distribution, _module, _minimum, _maximum, specifier in DEPENDENCIES:
        status = statuses.get(distribution)
        if not isinstance(status, Mapping) or status.get("compatible") is not True:
            result.append(f"{distribution}{specifier}")
    return result


def pip_command(python: str, wheel_dir: str, specs: Sequence[str]) -> list[str]:
    return [
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--find-links",
        wheel_dir,
        *specs,
    ]


def runner_command(
    python: str,
    source_root: str,
    competition_root: str,
    support_root: str,
    output_root: str,
    *,
    view_batch_size: int = 1,
    score_gate: str | None = None,
) -> list[str]:
    if type(view_batch_size) is not int or not 1 <= view_batch_size <= 8:
        raise PackageError("view_batch_size must be an integer in [1, 8]")
    if view_batch_size == 1 and score_gate is not None:
        raise PackageError("sequential view_batch_size=1 cannot use an accelerated score gate")
    if view_batch_size > 1 and not score_gate:
        raise PackageError("view_batch_size 2..8 requires an explicit score gate")
    source = Path(source_root)
    command = [
        python,
        str(source / "scripts" / "run_bh.py"),
        "--input",
        competition_root,
        "--registry",
        str(source / "bhreg.json"),
        "--support-root",
        support_root,
        "--out-dir",
        output_root,
        "--output-file",
        "submission.csv",
        "--devices",
        "cuda:0,cuda:1",
        "--pair-batch-size",
        "1",
        "--view-batch-size",
        str(view_batch_size),
        "--deadline-min",
        "120",
        "--source-budget-bytes",
        "999999",
    ]
    if score_gate is not None:
        command.extend(("--accelerated-score-gate", score_gate))
    return command


ENTRYPOINT_SOURCE = r'''#!/usr/bin/env python3
"""Offline, fail-closed launcher for the private V45 sparse Kaggle kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
MANIFEST = HERE / "bh45man.json"
ATTACHMENT = HERE / "bh45att.json"
SOURCE_ARCHIVE = HERE / "bh45src.zip"
SOURCE_ROOT = WORK / "bh45src"
DIAGNOSTIC = WORK / "bhdiag.json"
DEADLINE_SECONDS = 120 * 60
DEPENDENCIES = (
    ("torch", "torch", "2.2", "3", ">=2.2,<3"),
    ("numpy", "numpy", "1.26", "3", ">=1.26,<3"),
    ("scipy", "scipy", "1.11", "2", ">=1.11,<2"),
    ("zarr", "zarr", "3.0.10", "4", ">=3.0.10,<4"),
    ("numcodecs", "numcodecs", "0.14", "0.16", ">=0.14,<0.16"),
    ("donfig", "donfig", "0.8", "1", ">=0.8,<1"),
    ("google-crc32c", "google_crc32c", "1.5", "2", ">=1.5,<2"),
    ("packaging", "packaging", "22", "27", ">=22,<27"),
    ("typing-extensions", "typing_extensions", "4.13", "5", ">=4.13,<5"),
    ("Deprecated", "deprecated", "1.2", "2", ">=1.2,<2"),
    ("wrapt", "wrapt", "1.16", "3", ">=1.16,<3"),
)


class LaunchError(RuntimeError):
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
    try:
        return json.loads(Path(path).read_text(encoding="ascii"), object_pairs_hook=reject)
    except Exception as exc:
        raise LaunchError("invalid package JSON %s: %s" % (path, exc)) from exc


def _release(value):
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value))
    if match is None:
        return None
    return tuple(int(item) for item in match.group(1).split("."))


def _compare(left, right):
    width = max(len(left), len(right))
    return (left + (0,) * (width - len(left))) < (right + (0,) * (width - len(right)))


def _compatible(version, minimum, maximum):
    observed = _release(version)
    low = _release(minimum)
    high = _release(maximum)
    return observed is not None and low is not None and high is not None and not _compare(observed, low) and _compare(observed, high)


def _verify_package():
    payload = _load(MANIFEST)
    identity = payload.get("identity")
    if not isinstance(identity, dict) or payload.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest():
        raise LaunchError("package manifest identity SHA-256 mismatch")
    files = identity.get("files")
    if not isinstance(files, list):
        raise LaunchError("package manifest file list is missing")
    observed_bytes = MANIFEST.stat().st_size
    for item in files:
        path = HERE / str(item.get("path", ""))
        if not path.is_file():
            raise LaunchError("kernel package member is missing: %s" % path)
        actual_hash = _sha_file(path)
        actual_bytes = path.stat().st_size
        if actual_hash != item.get("sha256") or actual_bytes != item.get("bytes"):
            raise LaunchError(
                "kernel member identity mismatch for %s: expected bytes=%s sha256=%s, got bytes=%s sha256=%s"
                % (path.name, item.get("bytes"), item.get("sha256"), actual_bytes, actual_hash)
            )
        observed_bytes += actual_bytes
    if observed_bytes != identity.get("package_bytes") or observed_bytes >= identity.get("budget_bytes", 0):
        raise LaunchError("aggregate kernel source budget mismatch: bytes=%s manifest=%s" % (observed_bytes, identity.get("package_bytes")))
    return payload


def _probe_dependencies():
    # Probe in a disposable interpreter so an incompatible module is never
    # retained in this launcher's sys.modules across an offline pip repair.
    probe = r"""import importlib
import importlib.metadata
import json
import sys
dependencies = json.loads(sys.stdin.read())
result = {}
for distribution, module, _minimum, _maximum, specifier in dependencies:
    record = {"distribution": distribution, "import": module, "specifier": specifier}
    try:
        version = importlib.metadata.version(distribution)
        imported = importlib.import_module(module)
        record.update({"version": version, "import_file": str(getattr(imported, "__file__", None)), "error": None})
    except BaseException as exc:
        record.update({"version": None, "import_file": None, "error": "%s: %s" % (type(exc).__name__, exc)})
    result[distribution] = record
print("BHDEP=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        input=json.dumps(DEPENDENCIES),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    marker = "BHDEP="
    line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)), None)
    if completed.returncode != 0 or line is None:
        raise LaunchError(
            "dependency import/version probe failed before image reads: returncode=%s stderr=%s"
            % (completed.returncode, completed.stderr[-4000:])
        )
    try:
        statuses = json.loads(line[len(marker):])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LaunchError("dependency probe returned malformed diagnostics") from exc
    for distribution, _module, minimum, maximum, _specifier in DEPENDENCIES:
        record = statuses[distribution]
        record["compatible"] = record["error"] is None and _compatible(record["version"], minimum, maximum)
    return statuses


def _source_tree_sha(root, relatives):
    digest = hashlib.sha256()
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _validate_attachment(diagnostics):
    payload = _load(ATTACHMENT)
    identity = payload.get("identity")
    if not isinstance(identity, dict) or payload.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest():
        raise LaunchError("attachment manifest identity SHA-256 mismatch")
    candidates = identity.get("mount_candidates")
    if not isinstance(candidates, list) or not candidates or any(not isinstance(item, str) for item in candidates):
        raise LaunchError("support attachment mount candidate list is missing")
    required_files = identity.get("required_files", [])
    source_identity = identity.get("source_tree", {})
    expected_wheels = identity.get("wheelhouse", {}).get("files", [])
    expected_names = sorted(str(item.get("path")) for item in expected_wheels)
    candidate_results = []
    for mount in candidates:
        support = Path(mount)
        artifact_results = []
        for item in required_files:
            path = support / str(item.get("path", ""))
            actual_bytes = path.stat().st_size if path.is_file() else None
            actual_hash = _sha_file(path) if path.is_file() else None
            artifact_results.append({
                "path": str(path), "expected_bytes": item.get("bytes"), "actual_bytes": actual_bytes,
                "expected_sha256": item.get("sha256"), "actual_sha256": actual_hash,
                "ok": actual_bytes == item.get("bytes") and actual_hash == item.get("sha256"),
            })
        actual_source = _source_tree_sha(support, source_identity.get("paths", []))
        wheel_dir = support / str(identity.get("wheelhouse", {}).get("path", "wheels"))
        actual_names = sorted(path.name for path in wheel_dir.glob("*.whl")) if wheel_dir.is_dir() else []
        wheel_errors = []
        if actual_names != expected_names:
            wheel_errors.append({"expected_names": expected_names, "actual_names": actual_names})
        if not any(not item["ok"] for item in artifact_results) and actual_source == source_identity.get("sha256"):
            for item in expected_wheels:
                path = wheel_dir / str(item.get("path", ""))
                actual_bytes = path.stat().st_size if path.is_file() else None
                actual_hash = _sha_file(path) if path.is_file() else None
                if actual_bytes != item.get("bytes") or actual_hash != item.get("sha256"):
                    wheel_errors.append({
                        "path": str(path), "expected_bytes": item.get("bytes"), "actual_bytes": actual_bytes,
                        "expected_sha256": item.get("sha256"), "actual_sha256": actual_hash,
                    })
        result = {
            "mount": mount,
            "support_files": artifact_results,
            "source_tree": {"expected_sha256": source_identity.get("sha256"), "actual_sha256": actual_source},
            "wheelhouse": {
                "path": str(wheel_dir), "expected_count": len(expected_wheels),
                "actual_count": len(actual_names), "errors": wheel_errors,
            },
        }
        result["ok"] = (
            not any(not item["ok"] for item in artifact_results)
            and actual_source == source_identity.get("sha256")
            and not wheel_errors
        )
        candidate_results.append(result)
        if result["ok"]:
            diagnostics["support_candidates"] = candidate_results
            diagnostics["support_selected"] = mount
            return support, wheel_dir
    diagnostics["support_candidates"] = candidate_results
    raise LaunchError(
        "public support dataset identity mismatch at every known mount before image reads; inspect bhdiag.json for expected/actual checkpoint, source, and wheel SHA-256 values"
    )


def _install_missing(statuses, wheel_dir, diagnostics):
    specs = []
    for distribution, _module, _minimum, _maximum, specifier in DEPENDENCIES:
        if statuses[distribution]["compatible"] is not True:
            specs.append(distribution + specifier)
    compatible_core = [name for name in ("torch", "numpy", "scipy") if statuses[name]["compatible"]]
    if any(spec.split("<", 1)[0].split(">", 1)[0] in compatible_core for spec in specs):
        raise LaunchError("internal dependency plan would reinstall a compatible torch/numpy/scipy")
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", str(wheel_dir), *specs]
    diagnostics["dependency_plan"] = {"install_specs": specs, "command": command if specs else None}
    if not specs:
        return statuses
    environment = dict(os.environ)
    environment.update({"PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    result = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    diagnostics["pip"] = {"returncode": result.returncode, "output_tail": result.stdout[-12000:]}
    if result.returncode != 0:
        raise LaunchError(
            "offline dependency repair failed; no network fallback is permitted. Missing/incompatible=%s; wheel input=%s; inspect bhdiag.json"
            % (specs, wheel_dir)
        )
    repaired = _probe_dependencies()
    diagnostics["dependencies_after"] = repaired
    failed = {name: value for name, value in repaired.items() if value["compatible"] is not True}
    if failed:
        raise LaunchError("offline dependency repair remained incompatible: %s" % sorted(failed))
    return repaired


def _extract_source(package_manifest):
    if SOURCE_ROOT.exists():
        raise LaunchError("refusing to overwrite existing source extraction directory: %s" % SOURCE_ROOT)
    archive_identity = package_manifest["identity"].get("archive", {})
    expected = {item["path"]: item for item in archive_identity.get("members", [])}
    with zipfile.ZipFile(SOURCE_ARCHIVE, "r") as archive:
        names = sorted(item.filename for item in archive.infolist() if not item.is_dir())
        if names != sorted(expected):
            raise LaunchError("source archive member set mismatch")
        for name in names:
            parts = Path(name).parts
            if Path(name).is_absolute() or not parts or any(part in ("", ".", "..") for part in parts):
                raise LaunchError("unsafe source archive member: %r" % name)
            value = archive.read(name)
            item = expected[name]
            if len(value) != item["bytes"] or hashlib.sha256(value).hexdigest() != item["sha256"]:
                raise LaunchError("source archive member SHA-256 mismatch: %s" % name)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(SOURCE_ARCHIVE, "r") as archive:
        archive.extractall(SOURCE_ROOT)


def _competition_root():
    candidates = (
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
        Path("/kaggle/input/biohub-cell-tracking-during-development"),
    )
    for path in candidates:
        if (path / "test").is_dir():
            return path
    raise LaunchError("competition test input is missing from the two known Kaggle mount paths")


def _runner_command(competition, support, view_batch_size, score_gate):
    if not 1 <= view_batch_size <= 8:
        raise LaunchError("--view-batch-size must be in [1, 8]")
    if view_batch_size == 1 and score_gate is not None:
        raise LaunchError("sequential view_batch_size=1 cannot use --score-gate")
    if view_batch_size > 1 and score_gate is None:
        raise LaunchError("view_batch_size 2..8 requires an explicit --score-gate")
    command = [
        sys.executable, str(SOURCE_ROOT / "scripts" / "run_bh.py"),
        "--input", str(competition), "--registry", str(SOURCE_ROOT / "bhreg.json"),
        "--support-root", str(support), "--out-dir", str(WORK), "--output-file", "submission.csv",
        "--devices", "cuda:0,cuda:1", "--pair-batch-size", "1",
        "--view-batch-size", str(view_batch_size), "--deadline-min", "120",
        "--source-budget-bytes", "999999",
    ]
    if score_gate is not None:
        command.extend(("--accelerated-score-gate", str(score_gate)))
    return command


def _run_with_deadline(command, environment, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LaunchError("120-minute global deadline expired before V45 launch")
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise LaunchError("120-minute global deadline reached; V45 process group terminated") from exc
    if returncode != 0:
        raise LaunchError("scripts/run_bh.py failed with exit code %s" % returncode)


def _write_diagnostic(value):
    WORK.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC.write_bytes(_canonical(value) + b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-batch-size", type=int, default=1)
    parser.add_argument("--score-gate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    diagnostics = {"schema": "bhv45-kaggle-diagnostic.v1", "network": "forbidden", "ok": False}
    try:
        package_manifest = _verify_package()
        diagnostics["dependencies_before"] = _probe_dependencies()
        support, wheel_dir = _validate_attachment(diagnostics)
        diagnostics["dependencies_final"] = _install_missing(diagnostics["dependencies_before"], wheel_dir, diagnostics)
        _extract_source(package_manifest)
        competition = _competition_root()
        environment = dict(os.environ)
        python_paths = [str(SOURCE_ROOT / "src"), str(support / "repo" / "src")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update({
            "PYTHONPATH": os.pathsep.join(python_paths), "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "CUDA_VISIBLE_DEVICES": "0,1",
        })
        command = _runner_command(competition, support, args.view_batch_size, args.score_gate)
        diagnostics["runner"] = {
            "command": command, "device_count": 2, "devices": ["cuda:0", "cuda:1"],
            "view_batch_size": args.view_batch_size, "score_gate": str(args.score_gate) if args.score_gate else None,
            "deadline_seconds": DEADLINE_SECONDS,
        }
        _write_diagnostic(diagnostics)
        _run_with_deadline(command, environment, deadline)
        diagnostics.update({"ok": True, "elapsed_seconds": time.monotonic() - started})
        _write_diagnostic(diagnostics)
        return 0
    except Exception as exc:
        diagnostics["error"] = "%s: %s" % (type(exc).__name__, exc)
        diagnostics["elapsed_seconds"] = time.monotonic() - started
        _write_diagnostic(diagnostics)
        print(diagnostics["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
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
        f"{item['path']}: expected {expected_by_path[item['path']]}, got {item['sha256']}"
        for item in required
        if item["path"] in expected_by_path and item["sha256"] != expected_by_path[item["path"]]
    ]
    actual_source_sha = source_tree_sha256(support_root)
    if actual_source_sha != expected_hashes["source_sha256"]:
        mismatches.append(
            f"support source tree: expected {expected_hashes['source_sha256']}, got {actual_source_sha}"
        )
    if mismatches:
        raise PackageError("support/checkpoint SHA-256 validation failed: " + "; ".join(mismatches))

    wheel_dir = support_root / "wheels"
    wheels = [file_record(path, relative_to=wheel_dir) for path in sorted(wheel_dir.glob("*.whl"))]
    if not wheels:
        raise PackageError(f"public support wheel input is empty: {wheel_dir}")
    wheel_bytes = sum(item["bytes"] for item in wheels)
    identity = {
        "attachment_only": True,
        "external_dataset": True,
        "publish": False,
        "dataset_ref": SUPPORT_REF,
        "dataset_slug": SUPPORT_DATASET_SLUG,
        "mount_candidates": list(support_mounts),
        "local_reuse_path": support_root.as_posix(),
        "checkpoint": {
            "path": "weights/unet_transformer/split_0/edge_predictor_best.pth",
            "bytes": next(item["bytes"] for item in required if item["path"].endswith("edge_predictor_best.pth")),
            "sha256": expected_hashes["weight_sha256"],
        },
        "required_files": required,
        "source_tree": {"paths": list(SUPPORT_SOURCE_PATHS), "sha256": actual_source_sha},
        "wheelhouse": {
            "path": "wheels",
            "included_in_kernel_source": False,
            "files": wheels,
            "file_count": len(wheels),
            "bytes": wheel_bytes,
            "manifest_sha256": sha256_bytes(canonical_json(wheels)),
        },
    }
    payload = {
        "schema": "bhv45-kaggle-attachment.v1",
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical_json(identity)),
    }
    return canonical_json(payload) + b"\n"


def _archive_members(
    repo_root: Path,
    registry: Path,
    *,
    expected_test_count: int,
) -> dict[str, bytes]:
    missing = [path for path in SOURCE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise PackageError(f"runner source files are missing: {missing}")
    members = {path: (repo_root / path).read_bytes() for path in SOURCE_PATHS}
    members["src/celltrack_alpha/__init__.py"] = b'"""Private V45 sparse runner modules."""\n'
    members["bhreg.json"] = compact_registry(registry, expected_test_count=expected_test_count)
    return members


def _archive_identity(members: Mapping[str, bytes], archive: bytes) -> dict[str, Any]:
    return {
        "path": ARCHIVE_NAME,
        "bytes": len(archive),
        "sha256": sha256_bytes(archive),
        "uncompressed_bytes": sum(len(value) for value in members.values()),
        "members": [
            {"path": name, "bytes": len(members[name]), "sha256": sha256_bytes(members[name])}
            for name in sorted(members)
        ],
    }


def _metadata_payload() -> bytes:
    value = {
        "id": KERNEL_REF,
        "title": KERNEL_SLUG,
        "code_file": ENTRYPOINT_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [COMPETITION_SLUG],
        "dataset_sources": [SUPPORT_REF],
    }
    return json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _manifest_payload(files: Mapping[str, bytes], archive_identity: Mapping[str, Any]) -> bytes:
    file_items = [
        {"path": name, "bytes": len(value), "sha256": sha256_bytes(value)}
        for name, value in sorted(files.items())
    ]
    package_bytes = 0
    while True:
        identity = {
            "private": True,
            "network": "forbidden",
            "publish_actions": False,
            "budget_bytes": SOURCE_LIMIT,
            "package_bytes": package_bytes,
            "package_file_count": len(files) + 1,
            "strictly_under_budget": True,
            "kernel_slug": KERNEL_SLUG,
            "external_support": {
                "dataset_ref": SUPPORT_REF,
                "dataset_slug": SUPPORT_DATASET_SLUG,
                "created_by_kernel_owner": False,
                "long_slug_exception": "pre-existing public Kaggle input",
            },
            "fixed_long_basename_exception": METADATA_NAME,
            "files": file_items,
            "archive": dict(archive_identity),
            "runtime": {
                "runner": "scripts/run_bh.py",
                "deadline_minutes": 120,
                "devices": ["cuda:0", "cuda:1"],
                "gpu_count": 2,
                "pair_batch_size": 1,
                "default_view_batch_size": 1,
                "accelerated_view_batch_sizes": [2, 3, 4, 5, 6, 7, 8],
                "accelerated_score_gate_required": True,
                "dependency_install": "python -m pip install --no-index --no-deps --find-links",
                "install_only_missing_or_incompatible": True,
                "compatible_core_reinstall": False,
            },
        }
        payload = {
            "schema": "bhv45-kaggle-package.v1",
            "identity": identity,
            "identity_sha256": sha256_bytes(canonical_json(identity)),
        }
        encoded = canonical_json(payload) + b"\n"
        observed = sum(len(value) for value in files.values()) + len(encoded)
        if observed == package_bytes:
            return encoded
        package_bytes = observed


def build_package(
    repo_root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    registry: Path | None = None,
    support_root: Path | None = None,
    expected_artifact_hashes: Mapping[str, str] = EXPECTED_ARTIFACT_HASHES,
    expected_test_count: int = 4,
    support_mounts: Sequence[str] = SUPPORT_MOUNT_CANDIDATES,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    registry = (registry or (repo_root / "reports/g02/dataset_registry_v1.json")).resolve()
    support_root = (support_root or (repo_root / "external/Tracking_Support")).resolve()

    validate_slug(KERNEL_SLUG, "kernel slug")
    for name in (ENTRYPOINT_NAME, ARCHIVE_NAME, ATTACHMENT_NAME, MANIFEST_NAME, output_dir.name):
        if len(name) > 15:
            raise PackageError(f"created basename exceeds 15 characters: {name}")
    if (
        not support_mounts
        or len(set(support_mounts)) != len(support_mounts)
        or any(not Path(item).is_absolute() or Path(item).name != SUPPORT_DATASET_SLUG for item in support_mounts)
    ):
        raise PackageError("support mount candidates must be unique absolute paths ending in the public dataset slug")
    if set(expected_artifact_hashes) != set(EXPECTED_ARTIFACT_HASHES):
        raise PackageError("expected artifact hash mapping has missing or unexpected fields")

    members = _archive_members(repo_root, registry, expected_test_count=expected_test_count)
    archive = deterministic_zip(members)
    archive_identity = _archive_identity(members, archive)
    entrypoint = ENTRYPOINT_SOURCE.encode("ascii")
    attachment = _attachment_payload(support_root, expected_artifact_hashes, support_mounts=support_mounts)
    metadata = _metadata_payload()
    files = {
        ENTRYPOINT_NAME: entrypoint,
        ARCHIVE_NAME: archive,
        ATTACHMENT_NAME: attachment,
        METADATA_NAME: metadata,
    }
    manifest = _manifest_payload(files, archive_identity)
    files[MANIFEST_NAME] = manifest
    package_bytes = sum(len(value) for value in files.values())
    if package_bytes >= SOURCE_LIMIT:
        raise PackageError(f"aggregate submitted kernel source must be <{SOURCE_LIMIT} bytes, got {package_bytes}")
    if any(name.endswith(".whl") for name in files) or any(name.endswith(".whl") for name in members):
        raise PackageError("offline wheels must remain an attached private input, not kernel source")

    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in files)
    if unexpected:
        raise PackageError(f"refusing to mix private package with unexpected files: {unexpected}")
    for name, value in files.items():
        temporary = output_dir / f".{name}.tmp"
        temporary.write_bytes(value)
        temporary.replace(output_dir / name)

    records = [file_record(output_dir / name, relative_to=output_dir) for name in sorted(files)]
    report = {
        "schema": "bhv45-package-build.v1",
        "output_dir": str(output_dir),
        "aggregate_bytes": package_bytes,
        "budget_bytes": SOURCE_LIMIT,
        "strictly_under_budget": package_bytes < SOURCE_LIMIT,
        "file_count": len(records),
        "files": records,
        "archive": archive_identity,
        "attachment_manifest_sha256": sha256_bytes(attachment),
        "package_manifest_sha256": sha256_bytes(manifest),
        "network_or_kaggle_actions": False,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    args = parser.parse_args(argv)
    report = build_package(
        ROOT,
        args.output,
        registry=args.registry,
        support_root=args.support_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
