#!/usr/bin/env python3
"""Run decoupled H1 zero-wrfeat held-fit graph gates, 44b6 first."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import evaluate_trackastra_linear_oof_graph_gate as engine  # noqa: E402
from p4_edge_attention import (  # noqa: E402
    DEFAULT_AUXILIARY,
    DEFAULT_CACHE,
    DEFAULT_EDGE,
    DEFAULT_SUPPORT,
    DEFAULT_ZIP,
)


DEFAULT_CONTRACT = (
    ROOT
    / "reports/g02/screening/"
    "h1_zero_wrfeat_graph_gate_contract_parity_repair.json"
)
DEFAULT_FEATURE_ROOT = (
    ROOT / "reports/g02/trackastra_h1_feature_cache"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/screening/ga_h1_zero_wrfeat_graph"
)
FAMILY_ORDER = ("44b6", "6bba")


class H1ZeroWrfeatGateError(RuntimeError):
    """The fit105-only, 44b6-first H1 gate failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H1ZeroWrfeatGateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise H1ZeroWrfeatGateError(f"{label} must be an object")
    return value


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise H1ZeroWrfeatGateError(f"{label} identity mismatch: {path}")


def verify_parity_repair(
    contract: Mapping[str, Any],
    *,
    edge_artifact: Path,
    edge_auxiliary: Path,
) -> dict[str, Any]:
    repair = contract.get("parity_repair")
    pins = contract.get("pinned_inputs")
    if (
        not isinstance(repair, Mapping)
        or not isinstance(pins, Mapping)
        or repair.get("canonical_platform") != "Tesla T4"
        or repair.get("verdict") != "BENIGN_FP"
        or repair.get("m5_is_a_pass_fail_gate") is not False
    ):
        raise H1ZeroWrfeatGateError("parity-repair contract is not locked")
    for label, raw in pins.items():
        if not isinstance(raw, Mapping):
            raise H1ZeroWrfeatGateError(f"malformed pinned input: {label}")
        path = (ROOT / str(raw["path"])).resolve()
        _verify_file(path, str(raw["sha256"]), str(label))
    if edge_artifact != (ROOT / str(pins["edge_scorer"]["path"])).resolve():
        raise H1ZeroWrfeatGateError("edge scorer CLI path is not pinned")
    if edge_auxiliary != (
        ROOT / str(pins["edge_auxiliary"]["path"])
    ).resolve():
        raise H1ZeroWrfeatGateError("edge auxiliary CLI path is not pinned")
    repair_path = (ROOT / str(repair["parity_contract"])).resolve()
    _verify_file(
        repair_path,
        str(repair["parity_contract_sha256"]),
        "parity-repair contract",
    )
    parity = _load_json(repair_path, "parity-repair contract")
    platform = parity.get("platform_contract", {})
    classification = parity.get("classification", {})
    safety = classification.get("safety_catch", {})
    witnesses = parity.get("t4_repeatability_witnesses", [])
    if (
        parity.get("status") != "LOCKED_BENIGN_FP"
        or platform.get("canonical_platform") != "Tesla T4"
        or platform.get("m5_is_a_pass_fail_gate") is not False
        or classification.get("verdict") != "BENIGN_FP"
        or safety.get("structured_drift_detected") is not False
        or safety.get("nan_or_inf_count") != 0
        or safety.get("shape_match") is not True
        or safety.get("dtype_match") is not True
        or len(witnesses) != 2
        or any(
            witness.get("array_equal") is not True
            or witness.get("run1_sha256") != witness.get("run2_sha256")
            for witness in witnesses
        )
    ):
        raise H1ZeroWrfeatGateError(
            "parity-repair safety classification did not pass"
        )
    attestation = _load_json(
        (ROOT / str(pins["cache_attestation"]["path"])).resolve(),
        "cache attestation",
    )
    if (
        attestation.get("cache_integrity") != "PASS"
        or attestation.get("coverage", {}).get("video_count") != 105
        or attestation.get("t4_repeatability") != witnesses
    ):
        raise H1ZeroWrfeatGateError(
            "cache attestation does not support the repair contract"
        )
    return {
        "verdict": "BENIGN_FP",
        "canonical_platform": "Tesla T4",
        "t4_repeatability_witnesses": witnesses,
        "cache_integrity": "PASS",
    }


def verify_fit_inputs(
    contract: Mapping[str, Any],
    *,
    cache_root: Path,
) -> dict[str, int]:
    data = contract["data"]
    manifest_path = (ROOT / str(data["fit_manifest"])).resolve()
    prepare_path = (ROOT / str(data["prepare_report"])).resolve()
    _verify_file(
        manifest_path,
        str(data["fit_manifest_sha256"]),
        "fit105 manifest",
    )
    _verify_file(
        prepare_path,
        str(data["prepare_report_sha256"]),
        "fit-label prepare report",
    )
    manifest = _load_json(manifest_path, "fit105 manifest")
    identities = {str(value) for value in manifest.get("identities", [])}
    if (
        len(identities) != 105
        or any(not value.startswith("train:") for value in identities)
    ):
        raise H1ZeroWrfeatGateError("fit105 identities drifted")
    verified_cache_files = 0
    for raw in manifest.get("files", []):
        if not isinstance(raw, Mapping):
            raise H1ZeroWrfeatGateError("malformed fit-cache manifest row")
        path = (ROOT / str(raw["path"])).resolve()
        if cache_root not in path.parents:
            raise H1ZeroWrfeatGateError("fit-cache file escaped cache root")
        if path.stat().st_size != int(raw["bytes"]):
            raise H1ZeroWrfeatGateError(f"fit-cache size mismatch: {path}")
        _verify_file(path, str(raw["sha256"]), "fit-cache file")
        verified_cache_files += 1
    prepare = _load_json(prepare_path, "fit-label prepare report")
    rows = {
        str(raw["identity"]): raw
        for raw in prepare.get("videos", [])
        if isinstance(raw, Mapping)
        and str(raw.get("identity", "")) in identities
    }
    if set(rows) != identities:
        raise H1ZeroWrfeatGateError("fit-label identity coverage drifted")
    for identity, raw in rows.items():
        path = (ROOT / str(raw["path"])).resolve()
        _verify_file(path, str(raw["sha256"]), f"fit label {identity}")
    return {
        "fit_identities": len(identities),
        "fit_cache_files": verified_cache_files,
        "fit_label_files": len(rows),
    }


def family_metrics(report: Mapping[str, Any]) -> dict[str, float] | None:
    decision = report.get("decision")
    if not isinstance(decision, Mapping):
        raise H1ZeroWrfeatGateError("family report has no graph decision")
    if report.get("graph") is None and decision.get("status") == "PARK":
        return None
    try:
        result = {
            "adjusted_edge_jaccard_delta": float(
                decision["adjusted_edge_jaccard_delta"]
            ),
            "raw_edge_jaccard_delta": float(
                decision["raw_edge_jaccard_delta"]
            ),
            "node_recall_delta": float(decision["node_recall_delta"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise H1ZeroWrfeatGateError(
            "family report graph metrics are malformed"
        ) from exc
    return result


def classify_family(
    family: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    held = report.get("proposal_prerequisite", {}).get("held_oof", {})
    proposal = {
        "proposal_precision": (
            float(held["precision"]) if "precision" in held else None
        ),
        "proposal_accepted_labeled": (
            int(held["accepted_labeled"])
            if "accepted_labeled" in held
            else None
        ),
        "proposal_correct_rescues": (
            int(held["correct_rescues"])
            if "correct_rescues" in held
            else None
        ),
        "proposal_harmful_swaps": (
            int(held["harmful_swaps"])
            if "harmful_swaps" in held
            else None
        ),
        "proposal_metrics_gate": False,
    }
    metrics = family_metrics(report)
    if metrics is None:
        return {
            "family": family,
            "status": "PARK",
            "reason": str(
                report.get("decision", {}).get(
                    "reason", "pre-graph family screen parked"
                )
            ),
            "adjusted_edge_jaccard_delta": None,
            "raw_edge_jaccard_delta": None,
            "node_recall_delta": None,
            "observation_distinct": False,
            **proposal,
        }
    adjusted = metrics["adjusted_edge_jaccard_delta"]
    passed = (
        adjusted >= 0.005
        and metrics["raw_edge_jaccard_delta"] >= 0.0
        and metrics["node_recall_delta"] >= 0.0
        and report.get("decision", {}).get("observation_distinct") is True
    )
    if family == "44b6" and adjusted <= 0.0:
        status = "KILL"
        reason = "44b6 adjusted graph delta is at or below zero"
    elif passed:
        status = "PASS"
        reason = "all preregistered family graph guards passed"
    else:
        status = "PARK"
        reason = "family graph result did not clear every promotion guard"
    return {
        "family": family,
        "status": status,
        "reason": reason,
        **metrics,
        "observation_distinct": bool(
            report.get("decision", {}).get("observation_distinct")
        ),
        **proposal,
    }


def execute_ordered(
    evaluator: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate 44b6 first and never call 6bba after a 44b6 failure."""
    records = []
    first = classify_family("44b6", evaluator("44b6"))
    records.append(first)
    if first["status"] != "PASS":
        return {
            "status": first["status"],
            "families": records,
            "six_bba_executed": False,
            "cross_validation_authorized": False,
            "next": (
                "H3-MASK-SOURCE eligible as a separate unstarted node"
                if first["status"] == "KILL"
                else "PARK H1; no downstream family or cross-validation run"
            ),
        }
    second = classify_family("6bba", evaluator("6bba"))
    records.append(second)
    if second["status"] != "PASS":
        return {
            "status": "PARK",
            "families": records,
            "six_bba_executed": True,
            "cross_validation_authorized": False,
            "next": "PARK H1; no cross-validation run",
        }
    return {
        "status": "PASS_HELD_FIT_BOTH_FAMILIES",
        "families": records,
        "six_bba_executed": True,
        "cross_validation_authorized": True,
        "next": "full cross-validation is a separate gated node",
    }


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument(
        "--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--raw-archive", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    parser.add_argument(
        "--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY
    )
    parser.add_argument(
        "--support-root", type=Path, default=DEFAULT_SUPPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0"
        or os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1"
    ):
        raise H1ZeroWrfeatGateError(
            "runner must be launched by the zero-swap resource guard"
        )
    for name in (
        "contract",
        "feature_root",
        "cache",
        "raw_archive",
        "edge_artifact",
        "edge_auxiliary",
        "support_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise H1ZeroWrfeatGateError(
            f"refusing to overwrite output: {args.output}"
        )
    contract = engine.load_h1_contract(
        args.contract, args.contract_sha256
    )
    if contract["execution"]["family_order"] != list(FAMILY_ORDER):
        raise H1ZeroWrfeatGateError("family order drifted")
    feature_report = (
        ROOT / contract["candidate"]["feature_cache_report"]
    )
    if not feature_report.is_file():
        raise H1ZeroWrfeatGateError(
            "authenticated zero-wrfeat feature report is not present"
        )
    parity_evidence = verify_parity_repair(
        contract,
        edge_artifact=args.edge_artifact,
        edge_auxiliary=args.edge_auxiliary,
    )
    fit_input_evidence = verify_fit_inputs(
        contract,
        cache_root=args.cache,
    )
    args.output.mkdir(parents=True, exist_ok=False)

    def evaluate(family: str) -> Mapping[str, Any]:
        family_args = argparse.Namespace(
            contract=args.contract,
            contract_sha256=args.contract_sha256,
            policy=ROOT / "reports/g02/compute_governance_v1.json",
            feature_root=args.feature_root,
            cache=args.cache,
            zip=args.raw_archive,
            edge_artifact=args.edge_artifact,
            edge_auxiliary=args.edge_auxiliary,
            support_root=args.support_root,
            output=args.output / family,
            resume=False,
            h1_family=family,
        )
        return engine.run(family_args)

    result = {
        "schema": "celltrack-alpha.h1-zero-wrfeat-graph-gate.v1",
        "arm": "points-only-zero-wrfeat-12",
        "scope": {
            "fit_train105_only": True,
            "family_order": list(FAMILY_ORDER),
            "forbidden_scope_access": False,
            "proposal_metrics_gate": False,
        },
        "parity_repair": parity_evidence,
        "input_attestation": fit_input_evidence,
        **execute_ordered(evaluate),
        "fallback": contract["fallback"],
        "contract": {
            "path": str(args.contract.relative_to(ROOT)),
            "sha256": args.contract_sha256,
        },
    }
    engine.run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def main() -> int:
    report = run(_parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS_HELD_FIT_BOTH_FAMILIES" else 3


if __name__ == "__main__":
    raise SystemExit(main())
