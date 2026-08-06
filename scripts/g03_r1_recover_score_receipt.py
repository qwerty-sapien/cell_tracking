#!/usr/bin/env python3
"""Recover an R1 receipt after NaN-only JSON serialization failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any


RUNNER_SHA256 = "90b2cce6c7f38d280b6ac460f99712d07590d534e9cab94f39e397b25cfedf47"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        raise RuntimeError("infinite metric cannot be recovered")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--inference-seal", type=Path, required=True)
    parser.add_argument("--inference-seal-sha256", required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--incident-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.runner) != RUNNER_SHA256:
        raise RuntimeError("frozen R1 runner hash drifted")
    if sha256(args.incident) != args.incident_sha256:
        raise RuntimeError("serialization incident hash drifted")
    if args.output.exists():
        raise RuntimeError("raw score output unexpectedly exists")
    spec = importlib.util.spec_from_file_location("g03_r1_frozen_runner", args.runner)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen R1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_atomic_json = module.atomic_json

    def recovery_atomic_json(path: Path, value: dict[str, Any]) -> str:
        recovered = sanitize(value)
        recovered["provenance"]["scorer_receipt_recovery"] = {
            "incident_path": str(args.incident),
            "incident_sha256": args.incident_sha256,
            "sanitization": "non-finite float NaN -> explicit JSON string NaN",
            "scientific_configuration_changed": False,
            "identical_sealed_fragments": True,
        }
        return original_atomic_json(path, recovered)

    module.atomic_json = recovery_atomic_json
    score_args = argparse.Namespace(
        contract=args.contract,
        contract_sha256=args.contract_sha256,
        inference_seal=args.inference_seal,
        inference_seal_sha256=args.inference_seal_sha256,
        output=args.output,
    )
    result = module.score(score_args)
    print(
        module.json.dumps(
            {
                "status": "RECOVERED_IDENTICAL_SCORE_RECEIPT",
                "output": str(args.output),
                "sha256": sha256(args.output),
                "score": float(result["candidate"]["score"]),
                "adj_edge_jaccard": float(
                    result["candidate"]["adj_edge_jaccard"]
                ),
                "edge_jaccard": float(result["candidate"]["edge_jaccard"]),
                "node_recall": float(result["candidate"]["node_recall"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
