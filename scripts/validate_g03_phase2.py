#!/usr/bin/env python3
"""Independent fail-closed validation of the G-03 Phase-2 h000 packet."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.g03_phase2_adapter import validate_h000  # noqa: E402


PACKET = ROOT / "reports/g03/screening/phase2_zero_init_v1"

EXPECTED_HASHES = {
    "scripts/g03_phase2_adapter.py": "ab396f59c117c7b3b80800b811c8680d0e1cbdecb083ec4911c43eeab267cf6a",
    "scripts/g03_phase2_h000.py": "9bd6deebb210a08f4308e78cbb6dba9c87e92c64461d6feb95d4f15efc901b93",
    "scripts/g03_phase2_h000_canonical.py": "78eb9021f4a4763ff1a21258c48b121c96becbe27f53da9ded5e5b3069f561e7",
    "scripts/g03_phase2_h000_score_recovery.py": "bedd1978314e804501799b77d4319252ce68d4d656d33445a377f5d864d537a2",
    "reports/g03/screening/phase2_zero_init_v1/PHASE2_ZERO_INIT_CONTRACT.json": "fb1d1f3331a52e159368de6fec6cec25d193c7b8371e19baad41a3de662435aa",
    "reports/g03/screening/phase2_zero_init_v1/H000_RUNTIME_EQUIVALENCE_AMENDMENT.json": "d1a317f40a2bb9aa767cfd2e93a485cf61ef6d592663eae15f4e06a773c9f021",
    "reports/g03/screening/phase2_zero_init_v1/H000_SCORE_SERIALIZATION_RECOVERY_CONTRACT.json": "ea67744aa4da4df3bc43fb7222eec3b3216137434ec5db6fbf1ff1090df0c4f9",
    "reports/g03/screening/phase2_zero_init_v1/h000/h000.pt": "603e02f8f152db642a177d57763a07bebf84518f868abbbe1e0246187b018f95",
    "reports/g03/screening/phase2_zero_init_v1/h000/PREFLIGHT.json": "9d4b9c04fff3c4ad22a4532a70c813176ce7525a4418fa1f41a07c8cee475889",
    "reports/g03/screening/phase2_zero_init_v1/h000/canonical/CANONICAL_INFERENCE_SEAL.json": "46f1c28fd8b0fabdd50688c6b7eb585f68ccbf0d1e5331c5fedc327ca98c9049",
    "reports/g03/screening/phase2_zero_init_v1/h000/CANONICAL_SCORE.json": "d7189066b4122ff11e8a82fef616e292a9053f947c0a2e7dce9b3923ef7fc1c0",
    "reports/g03/screening/wave_r/R1_ZERO_SHOT_CONTRACT.json": "aefe438e6d8966b25f3665a1f009d108ae15ef0737581c1aa07a57f027050204",
    "scripts/g03_r1_zero_shot.py": "90b2cce6c7f38d280b6ac460f99712d07590d534e9cab94f39e397b25cfedf47",
}


class ValidationError(RuntimeError):
    """An immutable identity or scientific assertion failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {relative}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def exact_float(actual: Any, expected: float, label: str) -> None:
    require(
        isinstance(actual, (int, float))
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15),
        f"{label} drifted: {actual!r} != {expected!r}",
    )


def main() -> int:
    started = time.perf_counter()
    checks: list[str] = []

    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        require(actual == expected, f"SHA-256 mismatch for {relative}: {actual}")
    checks.append(f"{len(EXPECTED_HASHES)} immutable SHA-256 pins")

    checkpoint_validation = validate_h000(PACKET / "h000/h000.pt")
    require(
        checkpoint_validation["status"]
        == "PASS_H000_ZERO_INIT_AND_TRAINABILITY",
        "checkpoint zero-init validation failed",
    )
    projection = checkpoint_validation["projection"]
    require(
        projection["in_features"] == 48
        and projection["out_features"] == 12
        and projection["weight_nonzero"] == 0
        and projection["bias_nonzero"] == 0
        and checkpoint_validation["output_probe_nonzero"] == 0,
        "h000 is not exact-zero Linear(48,12)",
    )
    gradients = checkpoint_validation["gradient_probe"]
    require(
        gradients["hidden_abs_sum"] > 0.0
        and gradients["projection_weight_grad_abs_sum"] > 0.0
        and gradients["projection_bias_grad_abs_sum"] > 0.0,
        "h000 projection trainability proof failed",
    )
    checks.append("checkpoint exact-zero output and nonzero projection gradients")

    seal = load_json(
        "reports/g03/screening/phase2_zero_init_v1/h000/canonical/"
        "CANONICAL_INFERENCE_SEAL.json"
    )
    require(
        seal.get("status") == "PASS_CANONICAL_H000_INTERFACE_AND_GRAPH_PARITY",
        "canonical inference seal is not PASS",
    )
    runtime = seal["canonical_runtime"]
    require(
        runtime["byte_identical_fragment_count"] == 36
        and runtime["fragment_aggregate_sha256"]
        == "5dfe083f13a4ff6f671918fa5ff6b1cee84c24b37ba699663fe9f1115b3d9c0a"
        and runtime["new_gpu_action"] is False
        and runtime["new_gpu_hours"] == 0.0,
        "canonical parity aggregate or resource state drifted",
    )
    videos = seal.get("videos")
    require(isinstance(videos, list) and len(videos) == 36, "seal is not 36 videos")
    identities: set[str] = set()
    for video in videos:
        identity = str(video["identity"])
        require(identity not in identities, f"duplicate parity identity: {identity}")
        identities.add(identity)
        canonical = ROOT / video["canonical_fragment_path"]
        reference = ROOT / video["reference_fragment_path"]
        canonical_hash = sha256(canonical)
        reference_hash = sha256(reference)
        require(
            video["fragment_byte_identical"] is True
            and canonical_hash == video["canonical_fragment_sha256"]
            and reference_hash == video["reference_fragment_sha256"]
            and canonical_hash == reference_hash,
            f"fragment parity failed: {identity}",
        )
    checks.append("36/36 independently rehashed canonical/reference graph pairs")

    score = load_json(
        "reports/g03/screening/phase2_zero_init_v1/h000/CANONICAL_SCORE.json"
    )
    require(
        score.get("status")
        == "COMPLETE_CANONICAL_H000_ZERO_FEATURE_MEASUREMENT",
        "canonical score status drifted",
    )
    exact_float(score["candidate"]["score"], 0.7630876414338513, "h000 score")
    exact_float(score["parity"]["absolute_score_error"], 0.0, "parity error")
    require(
        score["parity"]["pass"] is True
        and score["parity"]["canonical_fragments_byte_identical"] == "36/36"
        and score["decision"]["branch"] == "B"
        and score["decision"]["branch_b_dossier_verdict"] == "FAIL"
        and score["decision"]["funded_t4_hours"] == 0.0
        and score["execution"]["new_gpu_action"] is False
        and score["execution"]["new_gpu_hours"] == 0.0
        and score["execution"]["optimizer_steps"] == 0,
        "score parity, allocation, or zero-resource decision drifted",
    )
    recovery = load_json(
        "reports/g03/screening/phase2_zero_init_v1/h000/"
        "SCORE_SERIALIZATION_RECOVERY.json"
    )
    require(
        recovery["status"] == "COMPLETE_DETERMINISTIC_SERIALIZATION_RECOVERY"
        and recovery["score_output"]["sha256"]
        == EXPECTED_HASHES[
            "reports/g03/screening/phase2_zero_init_v1/h000/CANONICAL_SCORE.json"
        ]
        and recovery["graph_or_scorer_changed"] is False
        and recovery["scoring_algorithm_changed"] is False,
        "serialization-only recovery binding drifted",
    )
    checks.append("canonical score, exact parity, Branch-B, and recovery receipt")

    r1 = load_json("reports/g03/screening/wave_r/R1_ZERO_SHOT_RESULT.json")
    exact_float(r1["candidate"]["score"], 0.7637338804169099, "historical R1")
    controlled_delta = score["candidate"]["score"] - r1["candidate"]["score"]
    exact_float(controlled_delta, -0.0006462389830586, "h000 minus R1")
    checks.append("historical random-adapter R1 identity and controlled delta")

    dossier_path = PACKET / "BRANCH_B_EFFECT_SIZE_DOSSIER.md"
    dossier_text = dossier_path.read_text(encoding="utf-8")
    require("PENDING" not in dossier_text, "dossier retains a pending placeholder")
    for required_text in (
        "+0.0022",
        "+0.0039",
        "single A100",
        "600 epochs",
        "0/30 T4-hours funded",
    ):
        require(required_text in dossier_text, f"dossier omission: {required_text}")
    audit = load_json(
        "reports/g03/screening/phase2_zero_init_v1/BRANCH_B_EFFECT_SIZE_AUDIT.json"
    )
    exact_float(audit["corrected_h000"]["score"], 0.7630876414338513, "audit h000")
    require(
        audit["verdict"] == "FAIL"
        and audit["funded_t4_hours"] == 0.0
        and all(value == "FAIL" for value in audit["gates"].values()),
        "Branch-B audit did not fail closed",
    )
    h005 = load_json(
        "reports/g03/screening/phase2_zero_init_v1/H005_ALLOCATION_DECISION.json"
    )
    require(
        h005["status"] == "NOT_FUNDED_BRANCH_B_DOSSIER_FAIL"
        and h005["branch_b_dossier"]["sha256"] == sha256(PACKET / "BRANCH_B_EFFECT_SIZE_AUDIT.json")
        and h005["h005"]["score"] is None
        and h005["h005"]["slope_from_h000_per_t4_hour"] is None
        and h005["decision"]["funded_t4_hours"] == 0.0
        and h005["decision"]["new_gpu_hours"] == 0.0
        and h005["decision"]["optimizer_steps"] == 0
        and h005["decision"]["h030_authorized"] is False
        and h005["decision"]["preserved_t4_hours"] == 30.0,
        "h005 did not remain unfunded and fail closed",
    )
    checks.append("effect-size dossier FAIL and h005/h030 zero-spend closure")

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "schema": "celltrack-alpha.g03-phase2-root-validation.v1",
                "status": "PASS_G03_PHASE2_H000_PACKET_AND_ALLOCATION_CLOSURE",
                "checks": checks,
                "h000_score": score["candidate"]["score"],
                "controlled_delta_h000_minus_random_r1": controlled_delta,
                "branch": "B",
                "dossier_verdict": "FAIL",
                "h005": "NOT_FUNDED_NOT_MEASURED",
                "new_gpu_hours": 0.0,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
