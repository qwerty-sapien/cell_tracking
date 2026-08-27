#!/usr/bin/env python3
"""Score the exact released-detector D4 CV57 output with frozen scorer code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import pkgbh  # noqa: E402
import scoreall  # noqa: E402


RETIRED_BASELINE = {
    "score": 0.8495036625176058,
    "status": "retired_no_submission_path",
    "detector": "frozen-v45-one-pass-compact-head",
    "d4_views": [0],
    "edge_artifact_sha256": "8621a5aa2f2a52894a1b9f14d0f442673eac9eeb221427cce3522f01bc4eb9db",
    "division_mode": "inherited",
    "effective_division_enabled": True,
}


class CV57ScoreError(ValueError):
    """The exact CV57 score contract was not satisfied."""


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CV57ScoreError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CV57ScoreError(f"{label} must be a JSON object: {path}")
    return value


def validate_revalidated_scorer(path: Path) -> dict[str, Any]:
    """Require the current scorer to match a green parity ledger.

    The grouped split remains an immutable selection artifact and therefore
    continues to attest the scorer that existed when the split was frozen.
    A post-patch rescore instead takes scorer identity from the separately
    re-frozen parity ledger.
    """

    report = _load_object(path, "scorer parity ledger")
    fixture = report.get("fixture_report")
    expected = report.get("source_hashes")
    if report.get("artifact") != "scorer_parity_v1" or not isinstance(fixture, dict):
        raise CV57ScoreError("scorer parity ledger has the wrong schema")
    if fixture.get("overall_pass") is not True or fixture.get("edge_only_overall_pass") is not True:
        raise CV57ScoreError("scorer parity ledger is not green")
    if len(fixture.get("cases", ())) < 7:
        raise CV57ScoreError("scorer parity ledger has fewer than seven fixtures")
    if not isinstance(expected, dict):
        raise CV57ScoreError("scorer parity ledger lacks source hashes")

    observed: dict[str, str] = {}
    for relative in scoreall.SCORER_SOURCES:
        expected_hash = expected.get(relative)
        if not isinstance(expected_hash, str):
            raise CV57ScoreError(f"scorer parity ledger lacks a hash for {relative}")
        actual_hash = scoreall.sha256_file(ROOT / relative)
        if actual_hash != expected_hash:
            raise CV57ScoreError(
                f"revalidated scorer source hash mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        observed[relative] = actual_hash

    checkout = fixture.get("official_checkout")
    if not isinstance(checkout, str) or len(checkout) != 40:
        raise CV57ScoreError("scorer parity ledger lacks an official checkout SHA")
    return {
        "source_hashes": observed,
        "official_checkout": checkout,
        "fixture_count": len(fixture["cases"]),
        "parity_report": {"path": str(path), "sha256": scoreall.sha256_file(path)},
    }


def exact_selection(
    split: Mapping[str, Any],
    *,
    split_path: Path,
    archive_video_ids: Sequence[str],
) -> tuple[str, ...]:
    fold = scoreall.select_video_ids(
        split,
        "fold0",
        archive_train_video_ids=archive_video_ids,
    )
    if len(fold.selected_video_ids) != 58 or pkgbh.CV57_EXCLUDED_IDENTITY not in fold.selected_video_ids:
        raise CV57ScoreError("frozen fold 0 is not the expected 58-video parent scope")
    selected = tuple(
        identity
        for identity in fold.selected_video_ids
        if identity != pkgbh.CV57_EXCLUDED_IDENTITY
    )
    if selected != pkgbh.cv57_video_ids(split_path):
        raise CV57ScoreError("exact CV57 selection differs from the package selection")
    observed_hash = pkgbh.sha256_bytes(pkgbh.canonical_json(list(selected)))
    if observed_hash != pkgbh.CV57_SELECTION_SHA256:
        raise CV57ScoreError("exact CV57 selection SHA-256 mismatch")
    return selected


def validate_execution_evidence(
    *,
    selected: Sequence[str],
    manifest: scoreall.FragmentManifest,
    features: Path,
    statuses: Path,
    run_path: Path,
    diag_path: Path,
    merged_path: Path,
) -> dict[str, Any]:
    run = _load_object(run_path, "run manifest")
    diag = _load_object(diag_path, "launcher diagnostic")
    if run.get("ok") is not True or run.get("mode") != "cv":
        raise CV57ScoreError("run manifest is not a completed CV run")
    if tuple(run.get("requested_identities", ())) != tuple(selected):
        raise CV57ScoreError("run manifest does not request the exact CV57 selection")
    if run.get("graph_variant") != pkgbh.RELEASED_GRAPH_VARIANT:
        raise CV57ScoreError("run manifest did not use the released graph variant")
    edge = run.get("edge_artifact")
    if not isinstance(edge, dict) or edge.get("sha256") != pkgbh.CV57_EDGE_ARTIFACT_SHA256:
        raise CV57ScoreError("run manifest edge artifact SHA-256 mismatch")
    if edge.get("division_enabled") is not False:
        raise CV57ScoreError("run manifest edge artifact enabled divisions")
    gpu = diag.get("gpu_attestation", {}).get("observed", {})
    if diag.get("ok") is not True or gpu.get("classification") != "exact_t4x2":
        raise CV57ScoreError("launcher did not attest exact_t4x2 completion")

    feature_paths = {path.stem.replace("__", ":", 1): path for path in features.glob("*.json")}
    status_paths = {path.stem.replace("__", ":", 1): path for path in statuses.glob("*.json")}
    if set(feature_paths) != set(selected) or set(status_paths) != set(selected):
        raise CV57ScoreError("feature/status files do not exactly cover CV57")
    fragment_paths = dict(manifest.ordered)
    expected_views = list(range(8))
    worker_seconds: list[float] = []
    merged_rows = 0
    for identity in selected:
        feature = _load_object(feature_paths[identity], f"feature {identity}")
        status = _load_object(status_paths[identity], f"status {identity}")
        if status.get("schema") != "celltrack-alpha.bh-video-status.v1" or status.get("ok") is not True:
            raise CV57ScoreError(f"status is incomplete for {identity}")
        if status.get("identity") != identity or feature.get("identity") != identity:
            raise CV57ScoreError(f"feature/status identity mismatch for {identity}")
        if status.get("fragment_sha256") != scoreall.sha256_file(fragment_paths[identity]):
            raise CV57ScoreError(f"fragment SHA-256 mismatch for {identity}")
        if status.get("feature_sha256") != scoreall.sha256_file(feature_paths[identity]):
            raise CV57ScoreError(f"feature SHA-256 mismatch for {identity}")
        view = feature.get("view_execution", {})
        association = feature.get("edge_association", {})
        feature_edge = feature.get("edge_artifact", {})
        if (
            feature.get("graph_variant") != pkgbh.RELEASED_GRAPH_VARIANT
            or view.get("d4_views") != expected_views
            or view.get("mode") != "exact-sequential"
            or view.get("production_eligible") is not True
            or feature_edge.get("sha256") != pkgbh.CV57_EDGE_ARTIFACT_SHA256
            or feature_edge.get("division_enabled") is not False
            or association.get("topology_mode") != "legacy"
            or association.get("division_enabled") is not False
            or association.get("graph", {}).get("division_enabled") is not False
        ):
            raise CV57ScoreError(f"execution configuration mismatch for {identity}")
        result = status.get("result", {})
        worker_seconds.append(float(result["seconds"]))
        merged_rows += int(result["rows"])

    submission = run.get("submission", {})
    merged_sha256 = scoreall.sha256_file(merged_path)
    if submission.get("sha256") != merged_sha256 or int(submission.get("rows", -1)) != merged_rows:
        raise CV57ScoreError("merged CV57 output does not match run/status evidence")
    return {
        "kernel_ref": "onebitbrain/bhv45-cv57/3",
        "gpu_classification": "exact_t4x2",
        "graph_variant": pkgbh.RELEASED_GRAPH_VARIANT,
        "d4_views": expected_views,
        "view_execution_mode": "exact-sequential",
        "topology_mode": "legacy",
        "division_enabled": False,
        "edge_artifact_sha256": pkgbh.CV57_EDGE_ARTIFACT_SHA256,
        "video_count": len(selected),
        "fragment_count": len(manifest.ordered),
        "feature_count": len(feature_paths),
        "status_count": len(status_paths),
        "runner_elapsed_seconds": float(run["elapsed_seconds"]),
        "launcher_elapsed_seconds": float(diag["elapsed_seconds"]),
        "wall_seconds_per_video": float(run["elapsed_seconds"]) / len(selected),
        "worker_seconds_min": min(worker_seconds),
        "worker_seconds_mean": sum(worker_seconds) / len(worker_seconds),
        "worker_seconds_max": max(worker_seconds),
        "merged_rows": merged_rows,
        "merged_sha256": merged_sha256,
        "run_manifest": {"path": str(run_path), "sha256": scoreall.sha256_file(run_path)},
        "launcher_diagnostic": {"path": str(diag_path), "sha256": scoreall.sha256_file(diag_path)},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    split = _load_object(args.split, "frozen split")
    scorer = validate_revalidated_scorer(args.scorer_parity)
    archive_ids = scoreall.archive_train_video_ids(args.zip)
    selected = exact_selection(split, split_path=args.split, archive_video_ids=archive_ids)
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(args.frags)),
        selected,
    )
    execution = validate_execution_evidence(
        selected=selected,
        manifest=manifest,
        features=args.features,
        statuses=args.statuses,
        run_path=args.run,
        diag_path=args.diag,
        merged_path=args.merged,
    )

    store = ZipStore(args.zip, mode="r")
    try:
        rows, input_provenance, coverage = scoreall.score_fragment_input(store, manifest)
    finally:
        store.close()
    by_identity = {row["identity"]: row for row in rows}
    if set(by_identity) != set(selected):
        raise CV57ScoreError("scored identities do not exactly match CV57")
    rows = [by_identity[identity] for identity in selected]
    summary = scoreall.official_summary(rows)
    result = {
        "schema": "celltrack-alpha.bh-cv57-frozen-score.v1",
        "status": "completed",
        "selection": {
            "parent_scope": "fold0",
            "parent_video_count": 58,
            "selected_video_count": len(selected),
            "selected_video_ids": list(selected),
            "excluded_identity": pkgbh.CV57_EXCLUDED_IDENTITY,
            "selection_sha256": pkgbh.CV57_SELECTION_SHA256,
        },
        "summary": summary,
        "samples": rows,
        "execution": execution,
        "coverage": coverage,
        "input_provenance": input_provenance,
        "frozen_scorer": {
            **scorer,
            "split": {"path": str(args.split), "sha256": scoreall.sha256_file(args.split)},
            "archive": {"path": str(args.zip), "sha256": scoreall.sha256_file(args.zip)},
            "revalidated_after_upstream_patch": True,
        },
        "retired_baseline_comparison": {
            "retired": RETIRED_BASELINE,
            "new_score": summary["score"],
            "numeric_difference": summary["score"] - RETIRED_BASELINE["score"],
            "comparable_as_tta_measurement": False,
            "differing_dimensions": ["detector", "edge_artifact", "division_mode"],
            "interpretation": "The numeric difference is not a TTA delta.",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frags", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--statuses", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--diag", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scorer-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (CV57ScoreError, scoreall.ScoreAllError, OSError, ValueError) as exc:
        print(f"score_bh_cv57: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
