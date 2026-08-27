#!/usr/bin/env python3
"""Offline synthetic support/determinism audit for the pinned CELLECT supplier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class NetworkForbidden(RuntimeError):
    pass


def _forbid_network(*args: Any, **kwargs: Any) -> Any:
    raise NetworkForbidden("network access is forbidden in the CELLECT support audit")


def git_output(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(tree), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prereg", type=Path, required=True)
    value.add_argument("--prereg-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--provenance-output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.output.exists() or args.provenance_output.exists():
        raise SystemExit("refusing to overwrite CELLECT audit output")
    observed_prereg = sha256(args.prereg)
    if observed_prereg != args.prereg_sha256:
        raise SystemExit(f"CELLECT preregistration drift: {observed_prereg}")
    contract = json.loads(args.prereg.read_text(encoding="utf-8"))
    upstream = contract["upstream"]
    adapter = contract["offline_adapter"]
    for spec, path_key, hash_key in (
        (upstream, "license_path", "license_sha256"),
        (upstream, "model_source_path", "model_source_sha256"),
        (upstream, "weight_path", "weight_sha256"),
        (adapter, "model_path", "model_sha256"),
        (adapter, "adapter_path", "adapter_sha256"),
    ):
        path = Path(spec[path_key])
        digest = sha256(path)
        if digest != spec[hash_key]:
            raise RuntimeError(f"pinned CELLECT artifact drift: {path}: {digest}")

    tree = Path(upstream["local_tree"])
    git_head = git_output(tree, "rev-parse", "HEAD")
    git_status = git_output(tree, "status", "--short")
    if git_head != upstream["git_commit"] or git_status:
        raise RuntimeError("CELLECT upstream tree is not the clean pinned revision")
    tracked = git_output(
        tree,
        "ls-files",
        "--error-unmatch",
        str(Path(upstream["weight_path"]).relative_to(tree)),
    )
    if not tracked:
        raise RuntimeError("CELLECT weight is not tracked by the pinned repository")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = "/private/tmp/sa-repr-torch-offline"
    socket.create_connection = _forbid_network
    urllib.request.urlopen = _forbid_network

    sys.path.insert(0, adapter["source_root"])
    import torch
    from celltrack_alpha.cellect_model import (
        CellectModelError,
        PUBLISHED_CHECKPOINT_PARAMETERS,
        load_published_backbone,
    )

    torch.set_num_threads(1)
    torch.manual_seed(contract["synthetic_witness"]["seed"])
    total = 1 * 2 * 32 * 32 * 8
    source = torch.linspace(1.0, 4096.0, total, dtype=torch.float32).reshape(
        1, 2, 32, 32, 8
    )
    outputs = []
    elapsed = []
    parameter_counts = []
    for _ in range(contract["synthetic_witness"]["repeat_loads"]):
        started = time.monotonic()
        model = load_published_backbone(Path(upstream["weight_path"]), device="cpu")
        model.eval()
        parameter_counts.append(sum(value.numel() for value in model.parameters()))
        with torch.inference_mode():
            embedding = model(source).embedding
        elapsed.append(time.monotonic() - started)
        outputs.append(embedding)
    if parameter_counts != [PUBLISHED_CHECKPOINT_PARAMETERS] * 2:
        raise RuntimeError(f"CELLECT parameter count drift: {parameter_counts}")
    if tuple(outputs[0].shape) != (1, 64, 32, 32, 8):
        raise RuntimeError(f"CELLECT embedding shape drift: {tuple(outputs[0].shape)}")
    if not bool(torch.isfinite(outputs[0]).all().item()):
        raise RuntimeError("CELLECT emitted a nonfinite embedding")
    repeat_equal = bool(torch.equal(outputs[0], outputs[1]))
    if not repeat_equal:
        raise RuntimeError("CELLECT repeated fresh loads are not byte-identical")

    point_ids = [
        ("synthetic", 0, 10, 4, 8, 1),
        ("synthetic", 0, 11, 12, 16, 3),
        ("synthetic", 0, 12, 24, 20, 6),
    ]
    point_vectors = torch.stack(
        [outputs[0][0, :, y, x, z] for _, _, _, y, x, z in point_ids]
    )
    if tuple(point_vectors.shape) != (3, 64):
        raise RuntimeError("CELLECT point sampling support drift")

    nonfinite_failed_closed = False
    invalid = source.clone()
    invalid[0, 0, 0, 0, 0] = float("nan")
    try:
        with torch.inference_mode():
            model(invalid)
    except CellectModelError:
        nonfinite_failed_closed = True
    if not nonfinite_failed_closed:
        raise RuntimeError("CELLECT accepted nonfinite input")

    baseline = torch.tensor([0.25, -0.5, 1.75], dtype=torch.float64)
    no_feature = baseline.clone()
    no_feature_exact = tensor_sha(baseline) == tensor_sha(no_feature)
    if not no_feature_exact:
        raise RuntimeError("CELLECT no-feature control changed baseline")

    provenance = {
        "schema": "g02.sa-repr.cellect-provenance-manifest.v1",
        "supplier_id": contract["supplier_id"],
        "preregistration_sha256": observed_prereg,
        "upstream": {
            "repository": upstream["repository"],
            "git_commit": git_head,
            "tree_clean": True,
            "weight_tracked": True,
            "weight_path": upstream["weight_path"],
            "weight_sha256": upstream["weight_sha256"],
            "model_source_path": upstream["model_source_path"],
            "model_source_sha256": upstream["model_source_sha256"],
            "repository_license": upstream["repository_license"],
            "license_path": upstream["license_path"],
            "license_sha256": upstream["license_sha256"],
            "weight_license_evidence": upstream["weight_license_evidence"],
        },
        "offline_adapter": adapter,
        "network_attempts": 0,
        "external_downloads": 0,
        "gpu_jobs": 0,
        "runpod_jobs": 0,
        "submissions": 0,
    }
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    result = {
        "schema": "g02.sa-repr.cellect-support-audit.v1",
        "supplier_id": contract["supplier_id"],
        "status": "PASS",
        "preregistration_sha256": observed_prereg,
        "provenance_manifest": {
            "path": str(args.provenance_output),
            "sha256": sha256(args.provenance_output),
        },
        "support": {
            "device": "cpu",
            "threads": 1,
            "input_shape": list(source.shape),
            "embedding_shape": list(outputs[0].shape),
            "embedding_dtype": str(outputs[0].dtype),
            "parameter_count": parameter_counts[0],
            "finite": True,
            "point_identity_count": len(point_ids),
            "point_vectors_shape": list(point_vectors.shape),
            "point_vectors_sha256": tensor_sha(point_vectors),
        },
        "determinism": {
            "fresh_load_count": len(outputs),
            "embedding_sha256": [tensor_sha(value) for value in outputs],
            "byte_identical": repeat_equal,
        },
        "runtime": {
            "synthetic_load_plus_forward_seconds": elapsed,
            "full_fit105_projection_hours": None,
            "gpu_request_credible": False,
        },
        "controls": {
            "nonfinite_input_failed_closed": nonfinite_failed_closed,
            "no_feature_exact_baseline": no_feature_exact,
            "baseline_sha256": tensor_sha(baseline),
            "no_feature_sha256": tensor_sha(no_feature),
            "within_frame_embedding_permutation": "frozen in supplier contract; not a performance result",
        },
        "data_access": {
            "synthetic_only": True,
            "fit_images": False,
            "labels": False,
            "held_images": False,
            "cv57": False,
        },
        "feature_cache_produced": False,
        "graph_scoring": False,
        "supplier_contract_eligible": True,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
