#!/usr/bin/env python3
"""Offline, fail-closed launcher for the private V45 sparse Kaggle kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
DEADLINE_SECONDS = 360 * 60
KSHAPE = "NvidiaTeslaT4"
RELEASED_GRAPH_VARIANT = "v45-motion-lazyrefine-locked"
GPU_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_VERSION", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_REQUIRE_CUDA",
)
GPU_DIAG = None
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


def _diagnostic_call(function):
    try:
        return function()
    except BaseException as exc:
        return {"error_type": type(exc).__name__, "error_message": str(exc)}


def _diagnostic_command(command):
    try:
        result = subprocess.run(
            command, check=False, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10,
        )
        return {
            "command": command, "returncode": int(result.returncode),
            "stdout": result.stdout[-8000:], "stderr": result.stderr[-4000:],
        }
    except BaseException as exc:
        return {"command": command, "error_type": type(exc).__name__, "error_message": str(exc)}


def gpu():
    """Attest the exact T4x2 runtime and retain classified failure evidence."""

    global GPU_DIAG
    torch = importlib.import_module("torch")
    count = int(torch.cuda.device_count())
    devices = []
    for index in range(count):
        name = _diagnostic_call(lambda i=index: str(torch.cuda.get_device_name(i)))
        capability = _diagnostic_call(lambda i=index: list(torch.cuda.get_device_capability(i)))
        properties = _diagnostic_call(lambda i=index: torch.cuda.get_device_properties(i))
        row = {"index": index, "name": name, "capability": capability}
        if isinstance(properties, dict):
            row["properties_error"] = properties
        else:
            row.update({
                "total_memory_bytes": int(properties.total_memory),
                "multi_processor_count": int(properties.multi_processor_count),
            })
        devices.append(row)
    names = [row["name"] if isinstance(row["name"], str) else "<name-error>" for row in devices]
    name_mismatch = any("T4" not in name.upper() for name in names)
    mismatches = []
    if count != 2:
        mismatches.append("device_count")
    if name_mismatch:
        mismatches.append("device_name")
    if count == 0:
        classification = "no_cuda_devices"
    elif name_mismatch:
        classification = "unexpected_accelerator_family"
    elif count != 2:
        classification = "honored_t4_wrong_device_count"
    else:
        classification = "exact_t4x2"
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    GPU_DIAG = {
        "requested": {
            "kaggle_machine_shape": KSHAPE,
            "logical_accelerator": "T4x2",
            "expected_device_count": 2,
            "expected_name_contains": "T4",
            "devices": ["cuda:0", "cuda:1"],
        },
        "observed": {
            "classification": classification,
            "mismatches": mismatches,
            "device_count": count,
            "devices": devices,
            "torch_version": str(getattr(torch, "__version__", None)),
            "torch_cuda_version": str(getattr(getattr(torch, "version", None), "cuda", None)),
            "torch_cuda_available": _diagnostic_call(lambda: bool(torch.cuda.is_available())),
            "torch_compiled_arch_list": _diagnostic_call(lambda: list(torch.cuda.get_arch_list())),
            "cudnn_version": _diagnostic_call(lambda: cudnn.version()) if cudnn is not None else None,
        },
        "environment": {key: os.environ.get(key) for key in GPU_ENV_KEYS},
        "nvidia_smi": {
            "inventory": _diagnostic_command(["nvidia-smi", "-L"]),
            "query": _diagnostic_command([
                "nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]),
        },
    }
    if mismatches:
        raise LaunchError(
            "exact T4x2 required; requested_shape=%s classification=%s mismatches=%r count=%s names=%r"
            % (KSHAPE, classification, mismatches, count, names)
        )
    return names


def _competition_root():
    candidates = (
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
        Path("/kaggle/input/biohub-cell-tracking-during-development"),
    )
    for path in candidates:
        if (path / "test").is_dir():
            return path
    raise LaunchError("competition test input is missing from the two known Kaggle mount paths")


def _runner_command(competition, support, package_manifest, view_batch_size, score_gate):
    if not 1 <= view_batch_size <= 8:
        raise LaunchError("--view-batch-size must be in [1, 8]")
    if view_batch_size == 1 and score_gate is not None:
        raise LaunchError("sequential view_batch_size=1 cannot use --score-gate")
    if view_batch_size > 1 and score_gate is None:
        raise LaunchError("view_batch_size 2..8 requires an explicit --score-gate")
    run = package_manifest.get("identity", {}).get("runtime", {}).get("run")
    if not isinstance(run, dict):
        raise LaunchError("package run specification is missing")
    if run.get("mode") not in ("cv", "submit") or run.get("split") not in ("train", "test"):
        raise LaunchError("package run specification mode/split is invalid")
    if (
        run.get("graph_variant") != RELEASED_GRAPH_VARIANT
        or run.get("topology_mode") != "legacy"
        or run.get("detection_tta") != "D4-8-view"
    ):
        raise LaunchError("package run specification is not released-graph legacy-topology D4-8")
    if run.get("division_enabled") is not False:
        raise LaunchError("package run specification attempted to enable divisions")
    datasets = run.get("datasets")
    if datasets is not None and (not isinstance(datasets, list) or not datasets):
        raise LaunchError("package run specification datasets are invalid")
    command = [
        sys.executable, str(SOURCE_ROOT / "scripts" / "run_bh.py"),
        "--input", str(competition), "--registry", str(SOURCE_ROOT / "bhreg.json"),
        "--support-root", str(support), "--out-dir", str(WORK), "--output-file", str(run.get("output_file")),
        "--mode", str(run["mode"]), "--split", str(run["split"]),
        "--devices", "cuda:0,cuda:1", "--pair-batch-size", "1",
        "--view-batch-size", str(view_batch_size), "--deadline-min", "360",
        "--source-budget-bytes", "999999", "--graph-variant", str(run["graph_variant"]),
    ]
    if datasets is not None:
        command.extend(("--datasets", ",".join(datasets)))
    edge = run.get("edge_artifact")
    if edge is not None:
        if not isinstance(edge, dict):
            raise LaunchError("package edge artifact specification is invalid")
        command.extend((
            "--edge-artifact", str(SOURCE_ROOT / str(edge.get("path"))),
            "--edge-artifact-sha256", str(edge.get("sha256")),
        ))
    if score_gate is not None:
        command.extend(("--accelerated-score-gate", str(score_gate)))
    return command


def _run_with_deadline(command, environment, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LaunchError("360-minute global deadline expired before V45 launch")
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
        raise LaunchError("360-minute global deadline reached; V45 process group terminated") from exc
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
        diagnostics["gpu_names"] = gpu()
        diagnostics["gpu_attestation"] = GPU_DIAG
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
        command = _runner_command(competition, support, package_manifest, args.view_batch_size, args.score_gate)
        diagnostics["runner"] = {
            "command": command, "device_count": 2, "devices": ["cuda:0", "cuda:1"],
            "view_batch_size": args.view_batch_size, "score_gate": str(args.score_gate) if args.score_gate else None,
            "deadline_seconds": DEADLINE_SECONDS,
        }
        _write_diagnostic(diagnostics)
        _run_with_deadline(command, environment, deadline)
        input_contract = WORK / "input_contract.json"
        if input_contract.is_file():
            diagnostics["input_contract"] = _load(input_contract)
        diagnostics.update({"ok": True, "elapsed_seconds": time.monotonic() - started})
        _write_diagnostic(diagnostics)
        return 0
    except Exception as exc:
        diagnostics["gpu_attestation"] = GPU_DIAG
        input_contract = WORK / "input_contract.json"
        if input_contract.is_file():
            diagnostics["input_contract"] = _diagnostic_call(lambda: _load(input_contract))
        diagnostics["error"] = "%s: %s" % (type(exc).__name__, exc)
        diagnostics["elapsed_seconds"] = time.monotonic() - started
        _write_diagnostic(diagnostics)
        print(diagnostics["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
