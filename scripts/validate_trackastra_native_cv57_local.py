#!/usr/bin/env python3
"""Independently audit the sealed local Trackastra CV57 one-shot artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402


EXPECTED_SELECTION = "r08_c08_a035_off"
EXPECTED_CONTROL_SCORE = 0.8815220699597315


class AuditError(RuntimeError):
    """An immutable population, receipt, graph, or result invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="ascii"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.population_manifest)
    report = read_json(args.native_report)
    entries = manifest.get("entries", [])
    identities = tuple(str(row["identity"]) for row in entries)
    expected_slugs = tuple(identity.replace(":", "__", 1) for identity in identities)

    require(len(entries) == 57 and len(set(identities)) == 57, "manifest is not 57 unique identities")
    require(manifest.get("scope", {}).get("cv57_only") is True, "manifest is not CV57-only")
    require(report.get("status") == "COMPLETE", "native report is not COMPLETE")
    require(report.get("selected_configuration_name") == EXPECTED_SELECTION, "selection drift")
    require(report.get("scope", {}).get("alternative_variant_access") is False, "alternative variant access")
    require(report.get("scope", {}).get("gt_access") is False, "GT was accessed during inference")
    require(report.get("scope", {}).get("tuning") is False, "CV57 tuning was recorded")
    require(report.get("scope", {}).get("cv57_cache_access_per_identity") == 1, "cache access count drift")
    require(report.get("execution", {}).get("public_submission") is False, "public submission recorded")
    require(report.get("execution", {}).get("runpod_action") is False, "RunPod action recorded")

    fragment_root = args.native_report.parent / "fragments"
    access_root = args.native_report.parent / "access"
    actual_csv = {path.name for path in fragment_root.glob("*.csv")}
    actual_receipts = {path.name for path in fragment_root.glob("*.json")}
    actual_markers = {path.name for path in access_root.glob("*.started.json")}
    require(actual_csv == {f"{slug}.csv" for slug in expected_slugs}, "CSV population mismatch")
    require(actual_receipts == {f"{slug}.json" for slug in expected_slugs}, "receipt population mismatch")
    require(actual_markers == {f"{slug}.started.json" for slug in expected_slugs}, "access-marker population mismatch")

    report_videos = report.get("videos", [])
    require(tuple(row.get("identity") for row in report_videos) == identities, "report identity order mismatch")
    totals: Counter[str] = Counter()
    per_identity: list[dict[str, Any]] = []
    for ordinal, (entry, identity, slug, report_row) in enumerate(
        zip(entries, identities, expected_slugs, report_videos, strict=True), start=1
    ):
        csv_path = fragment_root / f"{slug}.csv"
        receipt = read_json(fragment_root / f"{slug}.json")
        marker = read_json(access_root / f"{slug}.started.json")
        require(marker.get("identity") == identity, f"marker identity mismatch: {identity}")
        require(marker.get("cv57_cache_access_count") == 1, f"marker access count mismatch: {identity}")
        require(marker.get("repeat_forbidden") is True, f"marker repeat flag mismatch: {identity}")
        require(receipt.get("identity") == identity, f"receipt identity mismatch: {identity}")
        require(receipt.get("cv57_cache_access_count") == 1, f"receipt access count mismatch: {identity}")
        require(receipt.get("alternative_variant_access") is False, f"alternative access: {identity}")
        require(receipt.get("gt_access") is False, f"GT access: {identity}")
        require(receipt.get("selected_configuration_name") == EXPECTED_SELECTION, f"selection drift: {identity}")
        require(receipt.get("sha256") == sha256(csv_path), f"CSV hash mismatch: {identity}")
        require(receipt.get("bhc_sha256") == entry["bhc"]["sha256"], f"BHC hash mismatch: {identity}")
        require(receipt.get("meta_sha256") == entry["meta"]["sha256"], f"metadata hash mismatch: {identity}")
        expected_report_row = dict(receipt)
        expected_report_row["population_ordinal"] = ordinal
        require(report_row == expected_report_row, f"report/receipt drift: {identity}")

        graphs = read_submission(csv_path)
        require(len(graphs) == 1, f"ambiguous graph count: {identity}")
        graph = next(iter(graphs.values()))
        out_degree = Counter(edge.source_id for edge in graph.edges)
        forks = sum(1 for degree in out_degree.values() if degree > 1)
        require(len(graph.nodes) == receipt["nodes"], f"node count mismatch: {identity}")
        require(len(graph.edges) == receipt["selected_edges"], f"edge count mismatch: {identity}")
        require(forks == receipt["forks"] == 0, f"fork count mismatch: {identity}")
        totals.update(
            video_count=1,
            node_count=len(graph.nodes),
            selected_edge_count=len(graph.edges),
            candidate_edge_count=int(receipt["candidate_edges"]),
            inference_calls=int(receipt["inference_calls"]),
            fork_count=forks,
        )
        per_identity.append({"identity": identity, "csv_sha256": receipt["sha256"]})

    aggregate = report.get("aggregate", {})
    for key in (
        "video_count",
        "node_count",
        "selected_edge_count",
        "candidate_edge_count",
        "inference_calls",
        "fork_count",
    ):
        require(int(aggregate.get(key, -1)) == totals[key], f"aggregate mismatch: {key}")
    family_counts = Counter(identity.split(":", 1)[1].split("_", 1)[0] for identity in identities)
    require(aggregate.get("family_counts") == dict(family_counts), "family-count mismatch")

    journal_rows = [json.loads(line) for line in args.journal.read_text(encoding="ascii").splitlines() if line]
    require(not [row for row in journal_rows if row.get("event") == "FAILED"], "failure recorded in journal")
    access_events = [row for row in journal_rows if row.get("event") == "ACCESS_STARTED"]
    complete_events = [row for row in journal_rows if row.get("event") == "GRAPH_COMPLETE"]
    require(tuple(row.get("identity") for row in access_events) == identities, "journal access order/count mismatch")
    require(tuple(row.get("identity") for row in complete_events) == identities, "journal completion order/count mismatch")
    require(all(row.get("cv57_cache_access_count") == 1 for row in access_events), "journal repeated access")

    scoring: dict[str, Any] | None = None
    if args.results is not None:
        result = read_json(args.results)
        require(result.get("status") == "COMPLETE_EXACT_ONE_SHOT_CV57", "score result is not complete")
        require(result.get("selection", {}).get("name") == EXPECTED_SELECTION, "scored selection drift")
        require(tuple(result.get("population", {}).get("identities", [])) == identities, "scored population drift")
        require(result.get("population", {}).get("alternative_variant_access") is False, "scored alternative access")
        require(result.get("population", {}).get("tuning") is False, "scored tuning")
        require(result.get("population", {}).get("repeat_cv57") is False, "repeat CV57 recorded")
        require(result.get("execution", {}).get("cv57_measurement_count") == 1, "measurement count drift")
        require(result.get("execution", {}).get("public_submission") is False, "public submission recorded")
        require(result.get("execution", {}).get("runpod_action") is False, "RunPod action recorded")
        require(result.get("native", {}).get("name") == EXPECTED_SELECTION, "native result name drift")
        require(result.get("hybrid", {}).get("name") == "APNCC_NATIVE_FORK_HYBRID", "hybrid name drift")
        control = float(result["apncc_control"]["summary"]["score"])
        hybrid = float(result["hybrid"]["summary"]["score"])
        native = float(result["native"]["summary"]["score"])
        require(control == EXPECTED_CONTROL_SCORE, "control score drift")
        require(hybrid == control, "zero-fork hybrid is not exactly APNCC")
        require(result["hybrid"].get("totals", {}).get("native_forks") == 0, "hybrid native forks nonzero")
        require(result["hybrid"].get("totals", {}).get("inserted_forks") == 0, "hybrid inserted forks nonzero")
        require(bool(result.get("beats_apncc")) == (max(native, hybrid) > control), "beats-control flag drift")
        ranked = result.get("ranking", [])
        require(
            [float(row["summary"]["score"]) for row in ranked]
            == sorted((control, hybrid, native), reverse=True),
            "ranking arithmetic mismatch",
        )
        scoring = {
            "results_sha256": sha256(args.results),
            "control_score": control,
            "native_score": native,
            "hybrid_score": hybrid,
            "beats_apncc": bool(result["beats_apncc"]),
        }

    audit_report = {
        "schema": "celltrack-alpha.trackastra-native-cv57-local-independent-audit.v1",
        "status": "PASS",
        "native_report": str(args.native_report),
        "native_report_sha256": sha256(args.native_report),
        "population_manifest": str(args.population_manifest),
        "population_manifest_sha256": sha256(args.population_manifest),
        "journal": str(args.journal),
        "journal_sha256": sha256(args.journal),
        "selection": EXPECTED_SELECTION,
        "population": {
            "video_count": len(identities),
            "family_counts": dict(family_counts),
            "access_events": len(access_events),
            "completion_events": len(complete_events),
            "per_identity": per_identity,
        },
        "aggregate_recomputed": dict(totals),
        "invariants": {
            "all_csv_hashes_match_receipts": True,
            "all_cache_hashes_match_manifest": True,
            "all_access_counts_exactly_one": True,
            "all_graphs_independently_parsed": True,
            "all_forks_zero": True,
            "alternative_variant_access": False,
            "gt_access_during_inference": False,
            "tuning": False,
            "public_submission": False,
            "runpod_action": False,
        },
        "scoring": scoring,
    }
    if args.output is not None:
        if args.output.exists() and not args.allow_overwrite:
            raise AuditError(f"refusing to overwrite audit: {args.output}")
        atomic_json(args.output, audit_report)
    return audit_report


def parser() -> argparse.ArgumentParser:
    branch = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
    native_root = branch / "cv57_local_native_v1"
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--native-report", type=Path, default=native_root / "report.json")
    value.add_argument("--journal", type=Path, default=native_root / "JOURNAL.jsonl")
    value.add_argument(
        "--population-manifest",
        type=Path,
        default=ROOT / "kaggle_kernel/g02-trackastra-native-cv57-v1/cv57-cache-manifest.json",
    )
    value.add_argument("--results", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--allow-overwrite", action="store_true")
    return value


if __name__ == "__main__":
    result = audit(parser().parse_args())
    print(json.dumps({"status": result["status"], "population": result["population"]["video_count"], "scoring": result["scoring"]}, sort_keys=True))
