#!/usr/bin/env python3
"""Private, fail-closed numerical A/B for V45 sequential and batched D4 TTA."""

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


def _write(payload):
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(_canonical(payload) + b"\n")


def main():
    started = time.monotonic()
    payload = {
        "schema": "biohub-v45-d4-ab.v1",
        "ok": False,
        "network": "forbidden",
        "submission_generation": False,
        "dataset": DATASET,
        "frames": [0, 1],
        "view_batch_sizes": [1, 4],
    }
    try:
        package = _verify_package()
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
        # Warm each execution shape once.  Timed calls then compare the same
        # detector ensemble and feature path used by V45 production inference.
        _encode(torch, model, model_input, 1)
        _encode(torch, model, model_input, 4)
        sequential = _encode(torch, model, model_input, 1)
        batched = _encode(torch, model, model_input, 4)
        payload["runs"] = {"sequential": _public_result(sequential), "batched": _public_result(batched)}
        payload["comparison"] = _compare(torch, sequential, batched)
        payload["speedup_wall"] = sequential["wall_seconds"] / batched["wall_seconds"]
        payload["speedup_cuda"] = sequential["cuda_milliseconds"] / batched["cuda_milliseconds"]
        payload["elapsed_seconds"] = time.monotonic() - started
        if payload["elapsed_seconds"] > DEADLINE_SECONDS:
            raise ABError("A/B deadline exceeded")
        if not payload["comparison"]["exactly_equal"]:
            raise ABError("batched D4 output differs from the sequential V45 reference")
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
