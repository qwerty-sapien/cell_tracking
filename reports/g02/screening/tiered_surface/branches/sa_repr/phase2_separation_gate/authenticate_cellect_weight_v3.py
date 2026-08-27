#!/usr/bin/env python3
"""Hash-authenticate and safely load the frozen official CELLECT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
SUPPLIER_CONTRACT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/FROZEN_SUPPLIER_CONTRACT_v2.json"
)


class AuthenticationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    contract = json.loads(SUPPLIER_CONTRACT.read_text())
    checkpoint = contract["checkpoint"]
    loader = contract["safe_loader"]
    weight_path = Path(checkpoint["local_path"])
    loader_path = Path(loader["path"])
    observed_weight = sha256(weight_path)
    observed_loader = sha256(loader_path)
    if observed_weight != checkpoint["sha256"]:
        raise AuthenticationError("checkpoint hash mismatch before load")
    if observed_loader != loader["sha256"]:
        raise AuthenticationError("safe-loader hash mismatch before load")
    if not loader["torch_load_weights_only_required"] or loader["unsafe_pickle_fallback_permitted"]:
        raise AuthenticationError("supplier contract does not fail closed on unsafe loading")
    sys.path.insert(0, str(ROOT / "_LATER/cellect/src"))
    from celltrack_alpha.cellect_model import load_published_backbone

    started = time.perf_counter()
    model = load_published_backbone(weight_path, device="cpu").eval()
    elapsed = time.perf_counter() - started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != contract["representation_and_controls"]["parameter_count"]:
        raise AuthenticationError("loaded parameter count mismatch")
    result = {
        "schema": "celltrack-alpha.cellect-weight-safe-load.v3",
        "status": "PASS_HASHED_BEFORE_TENSOR_ONLY_LOAD",
        "supplier_contract_sha256": sha256(SUPPLIER_CONTRACT),
        "checkpoint_path": str(weight_path),
        "checkpoint_sha256": observed_weight,
        "safe_loader_sha256": observed_loader,
        "torch_weights_only_required": True,
        "unsafe_pickle_fallback": False,
        "strict_state_and_keyset_load_passed": True,
        "parameter_count": parameter_count,
        "device": "cpu",
        "elapsed_seconds": elapsed,
        "training": False,
        "forward_ran": False,
        "network": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
