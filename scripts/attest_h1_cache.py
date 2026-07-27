#!/usr/bin/env python3
"""Authenticate the quarantined H1 fit105 cache and M5↔T4 tolerance gate."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_h1_feature_cache import compare_option_c_payloads  # noqa: E402


EXPECTED_CONTRACT_SHA256 = (
    "4da06d487ba1787f91467a01862e4f7471daa16244674f5171960774da06334b"
)
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch_cuda": "12.8",
    "cuda_runtime": "12.8.1",
    "cudnn": 91002,
    "device_0": "Tesla T4",
    "seed": 20260725,
    "tf32": False,
    "deterministic_algorithms": True,
}
EXPECTED_AGGREGATE = {
    "video_count": 105,
    "proposal_count": 195354,
    "row_count": 280095,
    "inference_calls": 9363,
    "cropped_calls": 4616,
    "peak_tokens": 1024,
    "bytes": 83615281,
    "resumed_video_count": 0,
}


class H1CacheAttestationError(RuntimeError):
    """The downloaded cache failed a preregistered integrity requirement."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise H1CacheAttestationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H1CacheAttestationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise H1CacheAttestationError(f"{label} is not an object: {path}")
    return value


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise H1CacheAttestationError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return observed


def resolve_output_path(output_root: Path, relative: str, label: str) -> Path:
    path = (output_root / relative).resolve()
    if output_root.resolve() not in path.parents:
        raise H1CacheAttestationError(f"{label} escapes output root: {relative}")
    return path


def assert_closed_scope(scope: Mapping[str, Any], label: str) -> None:
    if (
        scope.get("fit_train105_only") is not True
        or scope.get("family_counts") != {"44b6": 36, "6bba": 69}
        or any(
            scope.get(key) is not False
            for key in (
                "validation12_access",
                "cv57_access",
                "training",
            )
        )
    ):
        raise H1CacheAttestationError(f"{label} scope is not fit105-only")
    if scope.get("competition_submission") not in (None, False):
        raise H1CacheAttestationError(f"{label} claims a competition submission")


def verify_atomic_corpus(
    *,
    output_root: Path,
    report: Mapping[str, Any],
    contract_sha256: str,
    label: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    rows = report.get("files")
    if not isinstance(rows, list) or len(rows) != 105:
        raise H1CacheAttestationError(f"{label} does not contain 105 file rows")
    implementation = str(report.get("implementation_sha256", ""))
    if len(implementation) != 64:
        raise H1CacheAttestationError(f"{label} implementation hash is invalid")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise H1CacheAttestationError(f"{label} file row is malformed")
        record = dict(raw)
        identity = str(record.get("identity", ""))
        if (
            identity in result
            or not identity.startswith(("train:44b6_", "train:6bba_"))
            or record.get("contract_sha256") != contract_sha256
            or record.get("implementation_sha256") != implementation
        ):
            raise H1CacheAttestationError(
                f"{label} atomic identity/contract drift: {identity}"
            )
        feature = resolve_output_path(
            output_root, str(record.get("feature_path", "")), identity
        )
        checkpoint = feature.with_suffix(".json")
        require_hash(feature, str(record.get("feature_sha256", "")), identity)
        if feature.stat().st_size != int(record.get("bytes", -1)):
            raise H1CacheAttestationError(f"{identity} byte count drifted")
        atomic = load_json(checkpoint, f"{identity} atomic manifest")
        if atomic != record:
            raise H1CacheAttestationError(
                f"{identity} atomic manifest differs from report"
            )
        result[identity] = record
    families = Counter(identity.split(":", 1)[1].split("_", 1)[0] for identity in result)
    if families != Counter({"44b6": 36, "6bba": 69}):
        raise H1CacheAttestationError(f"{label} family coverage drifted: {families}")
    return result, implementation


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def verify_t4_parity(
    *,
    output_root: Path,
    records: object,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    if not isinstance(records, list) or len(records) != 2:
        raise H1CacheAttestationError("T4 parity record count drifted")
    expected = {"train:44b6_cf8fed6b", "train:6bba_edf14583"}
    payloads: dict[str, dict[str, np.ndarray]] = {}
    evidence = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise H1CacheAttestationError("T4 parity record is malformed")
        identity = str(raw.get("identity", ""))
        if identity not in expected or identity in payloads:
            raise H1CacheAttestationError(f"unexpected T4 witness: {identity}")
        first_path = resolve_output_path(
            output_root, str(raw.get("run1_path", "")), f"{identity} T4 run1"
        )
        second_path = resolve_output_path(
            output_root, str(raw.get("run2_path", "")), f"{identity} T4 run2"
        )
        first_hash = require_hash(
            first_path, str(raw.get("run1_sha256", "")), f"{identity} T4 run1"
        )
        second_hash = require_hash(
            second_path, str(raw.get("run2_sha256", "")), f"{identity} T4 run2"
        )
        first = load_npz(first_path)
        second = load_npz(second_path)
        equal = (
            set(first) == set(second)
            and all(np.array_equal(first[key], second[key]) for key in first)
        )
        if raw.get("t4_array_equal") is not True or not equal:
            raise H1CacheAttestationError(f"T4 nondeterminism: {identity}")
        payloads[identity] = first
        evidence.append(
            {
                "identity": identity,
                "run1_sha256": first_hash,
                "run2_sha256": second_hash,
                "array_equal": True,
            }
        )
    if set(payloads) != expected:
        raise H1CacheAttestationError("T4 witness coverage drifted")
    return payloads, sorted(evidence, key=lambda row: row["identity"])


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise H1CacheAttestationError(f"refusing to overwrite output: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--contract-sha256", default=EXPECTED_CONTRACT_SHA256
    )
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--m5-reference", type=Path, required=True)
    parser.add_argument("--attestation-output", type=Path, required=True)
    parser.add_argument("--tolerance-output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = args.output_root.resolve()
    contract_path = args.contract.resolve()
    run_path = args.run_manifest.resolve()
    feature_path = args.feature_report.resolve()
    base_path = args.base_report.resolve()
    m5_path = args.m5_reference.resolve()
    if args.contract_sha256 != EXPECTED_CONTRACT_SHA256:
        raise H1CacheAttestationError("unexpected H1 feature contract identity")
    require_hash(contract_path, args.contract_sha256, "H1 feature contract")
    contract = load_json(contract_path, "H1 feature contract")
    run_manifest = load_json(run_path, "H1 run manifest")
    feature_report = load_json(feature_path, "H1 feature report")
    base_report = load_json(base_path, "H1 base report")

    if (
        contract.get("schema")
        != "celltrack-alpha.trackastra-h1-feature-cache-contract.option-c.v1"
        or contract.get("revision", {}).get("decision")
        != "DECOUPLE_H1_FEATURE_ARMS_ZERO_WRFEAT_FIRST"
        or run_manifest.get("schema")
        != "celltrack-alpha.trackastra-h1-kaggle-run.v1"
        or run_manifest.get("contract_sha256") != args.contract_sha256
        or run_manifest.get("verdict")
        != "CACHE_GENERATED_QUARANTINED_PENDING_M5_T4_HUMAN_APPROVAL"
        or run_manifest.get("aggregate") != EXPECTED_AGGREGATE
        or run_manifest.get("tolerance_contract_finalized") is not False
        or run_manifest.get("human_approval_required") is not True
    ):
        raise H1CacheAttestationError("run manifest contract/aggregate drifted")
    assert_closed_scope(run_manifest.get("scope", {}), "run manifest")
    runtime = run_manifest.get("runtime")
    if not isinstance(runtime, Mapping) or any(
        runtime.get(key) != value for key, value in EXPECTED_RUNTIME.items()
    ):
        raise H1CacheAttestationError("canonical T4 runtime drifted")

    if (
        feature_report.get("schema")
        != "celltrack-alpha.trackastra-full-pool-feature-cache.v2"
        or feature_report.get("verdict") != "PASS_LABEL_FREE_FEATURE_CACHE"
        or feature_report.get("contract", {}).get("sha256")
        != args.contract_sha256
        or feature_report.get("aggregate") != EXPECTED_AGGREGATE
    ):
        raise H1CacheAttestationError("feature report contract/aggregate drifted")
    assert_closed_scope(feature_report.get("scope", {}), "feature report")
    feature_rows, feature_implementation = verify_atomic_corpus(
        output_root=output_root,
        report=feature_report,
        contract_sha256=args.contract_sha256,
        label="feature corpus",
    )

    base_aggregate = {
        "video_count": 105,
        "proposal_count": 195354,
        "row_count": 280095,
        "transition_count": 9363,
        "bytes": 81894769,
        "resumed_video_count": 0,
    }
    if (
        base_report.get("schema")
        != "celltrack-alpha.trackastra-apncc-base-cache.v2"
        or base_report.get("verdict") != "PASS_APNCC_BASE_CACHE"
        or base_report.get("contract", {}).get("sha256")
        != args.contract_sha256
        or base_report.get("aggregate") != base_aggregate
    ):
        raise H1CacheAttestationError("base report contract/aggregate drifted")
    assert_closed_scope(base_report.get("scope", {}), "base report")
    base_rows, base_implementation = verify_atomic_corpus(
        output_root=output_root,
        report=base_report,
        contract_sha256=args.contract_sha256,
        label="base corpus",
    )
    for identity, record in feature_rows.items():
        if record.get("base_feature_sha256") != base_rows[identity].get(
            "feature_sha256"
        ):
            raise H1CacheAttestationError(
                f"base-to-feature provenance drifted: {identity}"
            )
    base_report_hash = require_hash(
        base_path,
        str(feature_report.get("base_report", {}).get("sha256", "")),
        "feature-pinned base report",
    )

    t4_payloads, t4_evidence = verify_t4_parity(
        output_root=output_root,
        records=run_manifest.get("option_c_parity"),
    )
    if run_manifest.get("option_c_parity") != feature_report.get(
        "option_c_parity"
    ):
        raise H1CacheAttestationError("run/report T4 parity records drifted")

    parity_reference = contract.get("checkpoint_loader", {}).get(
        "parity_reference", {}
    )
    if (
        parity_reference.get("identity") != "train:6bba_edf14583"
        or require_hash(
            m5_path,
            str(parity_reference.get("sha256", "")),
            "banked M5 reference",
        )
        != parity_reference.get("sha256")
    ):
        raise H1CacheAttestationError("M5 reference contract drifted")
    m5_payload = torch.load(m5_path, map_location="cpu", weights_only=False)
    if not isinstance(m5_payload, Mapping):
        raise H1CacheAttestationError("M5 reference is not a mapping")
    comparison = compare_option_c_payloads(
        m5=m5_payload,
        t4_first=t4_payloads["train:6bba_edf14583"],
        t4_second=t4_payloads["train:6bba_edf14583"],
    )
    exact_field_diagnostics = []
    t4_reference = t4_payloads["train:6bba_edf14583"]
    for record in comparison["m5_t4"]["exact_fields"]:
        field = str(record["field"])
        m5_value = np.asarray(m5_payload[field])
        t4_value = np.asarray(t4_reference[field])
        aligned = m5_value.shape == t4_value.shape
        exact_field_diagnostics.append(
            {
                **record,
                "m5_shape": list(m5_value.shape),
                "t4_shape": list(t4_value.shape),
                "m5_dtype": str(m5_value.dtype),
                "t4_dtype": str(t4_value.dtype),
                "mismatch_count": (
                    int(np.count_nonzero(m5_value != t4_value))
                    if aligned
                    else None
                ),
                "maximum_absolute_difference": (
                    float(
                        np.max(
                            np.abs(
                                t4_value.astype(np.float64)
                                - m5_value.astype(np.float64)
                            ),
                            initial=0.0,
                        )
                    )
                    if aligned
                    else None
                ),
            }
        )
    comparison_verdict = str(comparison.get("verdict", ""))
    if comparison_verdict == "STOP_T4_NONDETERMINISTIC":
        node_status = "DEAD_T4_NONDETERMINISTIC"
    elif comparison_verdict.startswith("STOP_"):
        node_status = "DEAD_STRUCTURAL_M5_T4_DRIFT"
    else:
        node_status = "HUMAN_DECISION_TOLERANCE_APPROVAL_REQUIRED"

    tolerance = {
        "schema": "celltrack-alpha.h1-m5-t4-tolerance-proposal.v1",
        "node_id": "H1-ZWRFEAT-CACHE-ATTEST",
        "status": node_status,
        "reference": {
            "identity": "train:6bba_edf14583",
            "m5_path": str(m5_path),
            "m5_sha256": sha256(m5_path),
            "t4_sha256": t4_evidence[1]["run1_sha256"],
        },
        "preregistered_probe": {"atol": 1e-7, "rtol": 1e-5},
        "exact_field_diagnostics": exact_field_diagnostics,
        "comparison": comparison,
        "human_approval_permitted": not comparison_verdict.startswith("STOP_"),
    }
    attestation = {
        "schema": "celltrack-alpha.h1-cache-attestation.v1",
        "node_id": "H1-ZWRFEAT-CACHE-ATTEST",
        "status": node_status,
        "cache_integrity": "PASS",
        "contract": {
            "path": str(contract_path),
            "sha256": args.contract_sha256,
        },
        "artifacts": {
            "run_manifest": {"path": str(run_path), "sha256": sha256(run_path)},
            "feature_report": {
                "path": str(feature_path),
                "sha256": sha256(feature_path),
            },
            "base_report": {"path": str(base_path), "sha256": base_report_hash},
        },
        "coverage": {
            "video_count": 105,
            "family_counts": {"44b6": 36, "6bba": 69},
            "feature_atomic_files_verified": 105,
            "base_atomic_files_verified": 105,
            "proposal_count": EXPECTED_AGGREGATE["proposal_count"],
            "row_count": EXPECTED_AGGREGATE["row_count"],
            "inference_calls": EXPECTED_AGGREGATE["inference_calls"],
        },
        "implementations": {
            "feature": feature_implementation,
            "base": base_implementation,
        },
        "runtime": dict(runtime),
        "elapsed_seconds": run_manifest.get("elapsed_seconds"),
        "stage_a_seconds": run_manifest.get("stage_a_seconds"),
        "stage_b_seconds": run_manifest.get("stage_b_seconds"),
        "t4_repeatability": t4_evidence,
        "m5_t4_verdict": comparison_verdict,
        "graph_metrics": {"44b6": None, "6bba": None},
        "competition_score": None,
        "submission_csv_produced": False,
        "next": (
            "STOP: N2 graph gate and submission packaging remain closed"
            if comparison_verdict.startswith("STOP_")
            else "HUMAN-GATE: explicit tolerance approval before N2"
        ),
    }
    return attestation, tolerance


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    attestation, tolerance = run(args)
    write_json(args.attestation_output, attestation)
    write_json(args.tolerance_output, tolerance)
    print(json.dumps(attestation, sort_keys=True))
    return 3 if attestation["status"].startswith("DEAD_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
