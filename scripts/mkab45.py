#!/usr/bin/env python3
"""Build a private, hash-pinned V45 GPU batching A/B kernel.

This is deliberately not a submission runner.  It evaluates the exact V45
eight-view detector on two frames from one public test video, comparing the
reference sequential path against two batches of four compatible D4 views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Works for ``python -m scripts.mkab45`` and test imports.
    from scripts import pkgbh
except ModuleNotFoundError:  # Also support the documented direct script entrypoint.
    import pkgbh  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "kaggle_kernel" / "ab45p2"
ENTRYPOINT_NAME = "ab45.py"
MANIFEST_NAME = "ab45man.json"
KERNEL_SLUG = "biohub-v45abnb"
KERNEL_REF = f"onebitbrain/{KERNEL_SLUG}"
BENCHMARK_DATASET = "44b6_0113de3b"
DEADLINE_SECONDS = 15 * 60
SOURCE_LIMIT = pkgbh.SOURCE_LIMIT


class PackageError(RuntimeError):
    """The private numerical A/B package could not be assembled safely."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metadata(kernel_slug: str) -> bytes:
    if len(kernel_slug) > 15 or not kernel_slug.replace("-", "").isalnum() or kernel_slug.lower() != kernel_slug:
        raise PackageError(f"invalid private kernel slug: {kernel_slug!r}")
    payload = {
        "id": f"onebitbrain/{kernel_slug}",
        "title": kernel_slug,
        "code_file": ENTRYPOINT_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": [pkgbh.COMPETITION_SLUG],
        "dataset_sources": [pkgbh.SUPPORT_REF],
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _manifest(files: Mapping[str, bytes], *, kernel_slug: str, mode: str) -> bytes:
    if mode not in {"batch", "compile"}:
        raise PackageError(f"unsupported A/B mode: {mode!r}")
    records = [
        {"path": name, "bytes": len(value), "sha256": _sha(value)}
        for name, value in sorted(files.items())
    ]
    package_bytes = 0
    while True:
        identity = {
            "private": True,
            "network": "forbidden",
            "publish_actions": False,
            "submission_generation": False,
            "budget_bytes": SOURCE_LIMIT,
            "package_bytes": package_bytes,
            "package_file_count": len(files) + 1,
            "strictly_under_budget": True,
            "kernel_slug": kernel_slug,
            "benchmark": {
                "kind": "v45_d4_batch_ab" if mode == "batch" else "v45_compile_ab",
                "mode": mode,
                "dataset": BENCHMARK_DATASET,
                "frames": [0, 1],
                "view_batch_sizes": [1, 4] if mode == "batch" else [1],
                "requires_exact_feature_logit_and_detection_equality": True,
                "deadline_seconds": DEADLINE_SECONDS,
                "full_video_inference": False,
                "submission_csv": False,
            },
            "base_v45_package": {
                "entrypoint": pkgbh.ENTRYPOINT_NAME,
                "archive": pkgbh.ARCHIVE_NAME,
                "attachment": pkgbh.ATTACHMENT_NAME,
                "manifest": pkgbh.MANIFEST_NAME,
            },
            "files": records,
        }
        payload = {
            "schema": "biohub-v45-ab-package.v1",
            "identity": identity,
            "identity_sha256": _sha(_canonical(identity)),
        }
        encoded = _canonical(payload) + b"\n"
        observed = sum(len(value) for value in files.values()) + len(encoded)
        if observed == package_bytes:
            return encoded
        package_bytes = observed


def _require_benchmark_dataset(registry: Path) -> None:
    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        records = payload["records"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read test registry for A/B target: {exc}") from exc
    if not any(
        isinstance(row, Mapping)
        and row.get("split") == "test"
        and row.get("dataset") == BENCHMARK_DATASET
        for row in records
    ):
        raise PackageError(f"A/B target {BENCHMARK_DATASET!r} is absent from the exact test registry")


def _write(output: Path, files: Mapping[str, bytes]) -> None:
    permitted = set(files)
    if output.exists():
        unexpected = sorted(path.name for path in output.iterdir() if path.name not in permitted)
        if unexpected:
            raise PackageError(f"refusing to mix A/B package with unexpected files: {unexpected}")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        temporary = output / f".{name}.tmp"
        temporary.write_bytes(value)
        temporary.replace(output / name)


def build_package(
    repo_root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    registry: Path | None = None,
    support_root: Path | None = None,
    expected_artifact_hashes: Mapping[str, str] = pkgbh.EXPECTED_ARTIFACT_HASHES,
    expected_test_count: int = 4,
    support_mounts: Sequence[str] = pkgbh.SUPPORT_MOUNT_CANDIDATES,
    kernel_slug: str = KERNEL_SLUG,
    mode: str = "batch",
) -> dict[str, Any]:
    """Create the private A/B package without performing any Kaggle action."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    registry = (registry or repo_root / "reports/g02/dataset_registry_v1.json").resolve()
    support_root = (support_root or repo_root / "external/Tracking_Support").resolve()
    if len(kernel_slug) > 15:
        raise PackageError("A/B kernel slug exceeds the 15-character limit")
    if mode not in {"batch", "compile"}:
        raise PackageError("A/B mode must be batch or compile")
    for name in (ENTRYPOINT_NAME, MANIFEST_NAME, output_dir.name):
        if len(name) > 15:
            raise PackageError(f"created basename exceeds 15 characters: {name}")
    _require_benchmark_dataset(registry)

    # First build the existing V45 package so the source archive and support
    # attachment retain their established deterministic construction rules.
    pkgbh.build_package(
        repo_root,
        output_dir,
        registry=registry,
        support_root=support_root,
        expected_artifact_hashes=expected_artifact_hashes,
        expected_test_count=expected_test_count,
        support_mounts=support_mounts,
    )
    base_manifest = json.loads((output_dir / pkgbh.MANIFEST_NAME).read_text(encoding="ascii"))
    archive = base_manifest["identity"]["archive"]
    base_files = {
        name: (output_dir / name).read_bytes()
        for name in (pkgbh.ENTRYPOINT_NAME, pkgbh.ARCHIVE_NAME, pkgbh.ATTACHMENT_NAME)
    }
    base_files[pkgbh.METADATA_NAME] = _metadata(kernel_slug)
    base_files[pkgbh.MANIFEST_NAME] = pkgbh._manifest_payload(base_files, archive)

    entrypoint = ENTRYPOINT_SOURCE.encode("ascii")
    files = {ENTRYPOINT_NAME: entrypoint, **base_files}
    files[MANIFEST_NAME] = _manifest(files, kernel_slug=kernel_slug, mode=mode)
    aggregate_bytes = sum(len(value) for value in files.values())
    if aggregate_bytes >= SOURCE_LIMIT:
        raise PackageError(f"aggregate submitted kernel source must be <{SOURCE_LIMIT} bytes, got {aggregate_bytes}")
    if any(name.endswith(".whl") for name in files):
        raise PackageError("offline wheels must remain an attached Kaggle input")
    _write(output_dir, files)
    return {
        "schema": "biohub-v45-ab-build.v1",
        "output_dir": str(output_dir),
        "aggregate_bytes": aggregate_bytes,
        "budget_bytes": SOURCE_LIMIT,
        "strictly_under_budget": True,
        "file_count": len(files),
        "files": [
            {"path": name, "bytes": len(value), "sha256": _sha(value)}
            for name, value in sorted(files.items())
        ],
        "benchmark_dataset": BENCHMARK_DATASET,
        "kernel": f"onebitbrain/{kernel_slug}",
        "mode": mode,
        "network_or_kaggle_actions": False,
        "submission_generation": False,
    }


ENTRYPOINT_SOURCE = r'''#!/usr/bin/env python3
"""Private, fail-closed numerical A/B for V45 sequential and batched D4 TTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import bh45


HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
MANIFEST = HERE / "ab45man.json"
OUTPUT = WORK / "v45ab.json"
DATASET = "44b6_0113de3b"
DEADLINE_SECONDS = 15 * 60


class ABError(RuntimeError):
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
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key %r" % key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="ascii"), object_pairs_hook=reject)


def _verify_package():
    payload = _load(MANIFEST)
    identity = payload.get("identity")
    if not isinstance(identity, dict) or payload.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest():
        raise ABError("A/B package manifest identity SHA-256 mismatch")
    files = identity.get("files")
    if not isinstance(files, list):
        raise ABError("A/B package manifest has no file list")
    observed = MANIFEST.stat().st_size
    for item in files:
        path = HERE / str(item.get("path", ""))
        if not path.is_file() or path.stat().st_size != item.get("bytes") or _sha_file(path) != item.get("sha256"):
            raise ABError("A/B package member identity mismatch: %s" % path.name)
        observed += path.stat().st_size
    if observed != identity.get("package_bytes") or observed >= identity.get("budget_bytes", 0):
        raise ABError("A/B aggregate source budget mismatch")
    benchmark = identity.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("dataset") != DATASET or benchmark.get("submission_csv") is not False:
        raise ABError("A/B benchmark contract is malformed")
    return payload


def _signature(value):
    material = value.detach().contiguous().cpu().numpy()
    return {
        "shape": list(material.shape),
        "dtype": str(material.dtype),
        "sha256": hashlib.sha256(material.tobytes()).hexdigest(),
    }


def _detection_signature(torch, logits):
    result = []
    for value in logits:
        coords, scores = bh_model._detect_batch(torch, value)[0]
        result.append({"coords": _signature(coords), "scores": _signature(scores)})
    return result


def _sync(torch):
    torch.cuda.synchronize("cuda:0")


def _encode(torch, model, model_input, view_batch_size):
    _sync(torch)
    torch.cuda.empty_cache()
    _sync(torch)
    baseline = int(torch.cuda.memory_allocated("cuda:0"))
    torch.cuda.reset_peak_memory_stats("cuda:0")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    with torch.inference_mode():
        features, logits = bh_model._encode_d4(torch, model, model_input, view_batch_size=view_batch_size)
    end_event.record()
    _sync(torch)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated("cuda:0"))
    return {
        "view_batch_size": view_batch_size,
        "wall_seconds": elapsed,
        "cuda_milliseconds": float(start_event.elapsed_time(end_event)),
        "memory_allocated_before": baseline,
        "memory_peak": peak,
        "memory_peak_delta": peak - baseline,
        "features": features,
        "logits": logits,
        "feature_signature": _signature(features),
        "logit_signatures": [_signature(value) for value in logits],
        "detection_signatures": _detection_signature(torch, logits),
    }


def _compare(torch, reference, candidate):
    max_abs = float(torch.max(torch.abs(reference["features"] - candidate["features"])).item())
    feature_equal = bool(torch.equal(reference["features"], candidate["features"]))
    logits_equal = []
    for left, right in zip(reference["logits"], candidate["logits"], strict=True):
        logits_equal.append(bool(torch.equal(left, right)))
        max_abs = max(max_abs, float(torch.max(torch.abs(left - right)).item()))
    result = {
        "feature_bitwise_equal": feature_equal,
        "logit_bitwise_equal": logits_equal,
        "detection_signatures_equal": reference["detection_signatures"] == candidate["detection_signatures"],
        "max_abs_difference": max_abs,
    }
    result["exactly_equal"] = feature_equal and all(logits_equal) and result["detection_signatures_equal"]
    return result


def _public_result(run):
    return {key: value for key, value in run.items() if key not in {"features", "logits"}}


def _compiled_proxy(torch, model):
    class Encoder(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, imgs):
            return self.base.encode(imgs)

    compiled = torch.compile(
        Encoder(model),
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )

    class Proxy:
        def encode(self, imgs):
            return compiled(imgs)

    return Proxy()


def _write(payload):
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(_canonical(payload) + b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch", "compile"), default="batch")
    args = parser.parse_args(argv)
    started = time.monotonic()
    payload = {
        "schema": "biohub-v45-d4-ab.v1",
        "ok": False,
        "network": "forbidden",
        "submission_generation": False,
        "dataset": DATASET,
        "frames": [0, 1],
        "mode": args.mode,
        "view_batch_sizes": [1, 4] if args.mode == "batch" else [1],
    }
    try:
        package = _verify_package()
        if package["identity"]["benchmark"].get("mode") != args.mode:
            raise ABError("A/B requested mode does not match the signed package contract")
        base = bh45._verify_package()
        payload["package_identity_sha256"] = package["identity_sha256"]
        payload["base_package_identity_sha256"] = base["identity_sha256"]
        payload["dependencies_before"] = bh45._probe_dependencies()
        support, wheels = bh45._validate_attachment(payload)
        payload["dependencies_final"] = bh45._install_missing(payload["dependencies_before"], wheels, payload)
        bh45.SOURCE_ROOT = WORK / "ab45src"
        bh45._extract_source(base)
        competition = bh45._competition_root()
        sys.path.insert(0, str(bh45.SOURCE_ROOT / "src"))
        sys.path.insert(0, str(bh45.SOURCE_ROOT / "scripts"))
        sys.path.insert(0, str(support / "repo" / "src"))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(bh45.SOURCE_ROOT / "src"), str(support / "repo" / "src")])
        import numpy as np
        import torch
        import run_bh as bh_run
        from celltrack_alpha import bh_model as imported_model
        global bh_model
        bh_model = imported_model
        if not torch.cuda.is_available():
            raise ABError("private A/B requires one CUDA device")
        records = bh_run.select_records(bh_run.load_registry(bh45.SOURCE_ROOT / "bhreg.json"), split="test", requested=[DATASET])
        job = bh_run.discover_jobs(competition, records)[0]
        handle = bh_run.open_video(job)
        try:
            q_low, q_high = bh_run.read_quantile_attrs(handle.attrs)
            frames = [np.asarray(handle.array[index]) for index in (0, 1)]
        finally:
            handle.close()
        model, bundle = bh_model.load_v45_model(support, device="cuda:0")
        parameter = next(model.parameters())
        downsampled = np.stack([frame[::1, ::4, ::4] for frame in frames], axis=0)
        normalized = bh_model.quantile_normalize(downsampled, q_low, q_high)
        model_input = torch.from_numpy(np.ascontiguousarray(normalized)).to(device="cuda:0", dtype=parameter.dtype)[None]
        if tuple(model_input.shape) != (1, 2, 64, 64, 64):
            raise ABError("registered two-frame inference grid is not (1,2,64,64,64): %s" % (tuple(model_input.shape),))
        payload["input"] = {"shape": list(model_input.shape), "dtype": str(model_input.dtype), "q_low": q_low, "q_high": q_high}
        payload["checkpoint"] = {"path": str(bundle.weight_path), "sha256": bundle.hashes.weight_sha256}
        # Warm each execution shape once. Timed calls then compare the exact
        # detector ensemble and feature path used by V45 production inference.
        _encode(torch, model, model_input, 1)
        if args.mode == "batch":
            _encode(torch, model, model_input, 4)
            sequential = _encode(torch, model, model_input, 1)
            candidate = _encode(torch, model, model_input, 4)
            candidate_name = "batched"
        else:
            compiled_started = time.perf_counter()
            compiled = _compiled_proxy(torch, model)
            _encode(torch, compiled, model_input, 1)
            payload["compile_warmup_seconds"] = time.perf_counter() - compiled_started
            payload["compile_config"] = {
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": True,
                "dynamic": False,
            }
            sequential = _encode(torch, model, model_input, 1)
            candidate = _encode(torch, compiled, model_input, 1)
            candidate_name = "compiled"
        payload["runs"] = {"sequential": _public_result(sequential), candidate_name: _public_result(candidate)}
        payload["comparison"] = _compare(torch, sequential, candidate)
        payload["speedup_wall"] = sequential["wall_seconds"] / candidate["wall_seconds"]
        payload["speedup_cuda"] = sequential["cuda_milliseconds"] / candidate["cuda_milliseconds"]
        payload["elapsed_seconds"] = time.monotonic() - started
        if payload["elapsed_seconds"] > DEADLINE_SECONDS:
            raise ABError("A/B deadline exceeded")
        if not payload["comparison"]["exactly_equal"]:
            raise ABError("A/B candidate output differs from the sequential V45 reference")
        payload["ok"] = True
        _write(payload)
        return 0
    except Exception as exc:
        payload["error"] = "%s: %s" % (type(exc).__name__, exc)
        payload["elapsed_seconds"] = time.monotonic() - started
        _write(payload)
        print(payload["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=ROOT / "reports/g02/dataset_registry_v1.json")
    parser.add_argument("--support-root", type=Path, default=ROOT / "external/Tracking_Support")
    parser.add_argument("--kernel-slug", default=KERNEL_SLUG)
    parser.add_argument("--mode", choices=("batch", "compile"), default="batch")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_package(
                ROOT,
                args.output,
                registry=args.registry,
                support_root=args.support_root,
                kernel_slug=args.kernel_slug,
                mode=args.mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
