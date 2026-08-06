#!/usr/bin/env python3
"""Independently validate the completed fit105 sweep and sealed winner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from evaluate_trackastra_native_base_wave import atomic_json, configurations, sha256


ROOT = Path(__file__).resolve().parents[1]
BRANCH = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
DEFAULT_RESULT = BRANCH / "fit105_sweep_scored/FIT105_RESULTS.json"
DEFAULT_OUTPUT = BRANCH / "fit105_sweep_scored"
DEFAULT_MATERIALIZED = BRANCH / "fit105_sweep_complete"
DEFAULT_PREREG = BRANCH / "PREREGISTRATION.json"
DEFAULT_VALIDATION = BRANCH / "FIT105_SELECTION_VALIDATION.json"


class ValidationError(RuntimeError):
    pass


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    summary = row["summary"]
    return (
        -float(summary["score"]),
        -float(summary["division_jaccard"]),
        -float(summary["adj_edge_jaccard"]),
        -float(summary["edge_jaccard"]),
        -float(summary["node_recall"]),
        str(row["name"]),
    )


def aggregate_digest(videos: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for video in videos:
        digest.update(str(video["identity"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(video["fragment_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.validation.exists():
        raise ValidationError("refusing to overwrite fit105 validation")
    result = json.loads(args.result.read_text(encoding="ascii"))
    prereg = json.loads(args.prereg.read_text(encoding="ascii"))
    expected_configs = {row["name"]: row for row in configurations(prereg)}
    score_paths = sorted((args.output / "scores").glob("*.json"))
    manifest_paths = sorted((args.materialized / "variants").glob("*/manifest.json"))
    if (
        result.get("status") != "COMPLETE_EXACT_108_VARIANT_SWEEP"
        or len(result.get("ranking", [])) != 108
        or len(score_paths) != 108
        or len(manifest_paths) != 108
    ):
        raise ValidationError("fit105 result or receipt population is incomplete")

    result_rows = {row["name"]: row for row in result["ranking"]}
    score_rows = {}
    manifests = {}
    identities = None
    for path in manifest_paths:
        row = json.loads(path.read_text(encoding="ascii"))
        name = path.parent.name
        config = expected_configs.get(name)
        video_ids = tuple(str(video["identity"]) for video in row.get("videos", []))
        if (
            config is None
            or row.get("configuration") != config
            or row.get("video_count") != 105
            or len(video_ids) != 105
            or len(set(video_ids)) != 105
            or row.get("aggregate_graph_sha256") != aggregate_digest(row["videos"])
        ):
            raise ValidationError(f"structural manifest drift: {name}")
        if identities is None:
            identities = video_ids
        elif video_ids != identities:
            raise ValidationError(f"structural identity order drift: {name}")
        manifests[name] = row

    for path in score_paths:
        row = json.loads(path.read_text(encoding="ascii"))
        name = path.stem
        if (
            row.get("name") != name
            or name not in expected_configs
            or row.get("aggregate_graph_sha256") != manifests[name]["aggregate_graph_sha256"]
            or row.get("score_derivation") not in {
                "DIRECT_FROZEN_OFFICIAL_SCORER",
                "BYTE_IDENTICAL_TO_DIRECT_EXACT_SCORER_REPRESENTATIVE",
            }
        ):
            raise ValidationError(f"score receipt drift: {name}")
        reported = result_rows.get(name)
        if (
            reported is None
            or reported.get("configuration") != expected_configs[name]
            or reported.get("summary") != row.get("summary")
            or reported.get("edge_counts") != row.get("edge_counts")
            or reported.get("aggregate_graph_sha256") != row.get("aggregate_graph_sha256")
        ):
            raise ValidationError(f"FIT105_RESULTS score mismatch: {name}")
        score_rows[name] = row

    by_digest: dict[str, list[str]] = {}
    for name, row in score_rows.items():
        by_digest.setdefault(row["aggregate_graph_sha256"], []).append(name)
    direct_count = 0
    alias_count = 0
    for digest, names in by_digest.items():
        representative = sorted(names)[0]
        reference = score_rows[representative]
        for name in names:
            row = score_rows[name]
            if row["summary"] != reference["summary"] or row["edge_counts"] != reference["edge_counts"]:
                raise ValidationError(f"graph-equivalent score mismatch: {digest}")
            expected_derivation = (
                "DIRECT_FROZEN_OFFICIAL_SCORER"
                if name == representative
                else "BYTE_IDENTICAL_TO_DIRECT_EXACT_SCORER_REPRESENTATIVE"
            )
            if row["score_derivation"] != expected_derivation:
                raise ValidationError(f"representative derivation drift: {name}")
            direct_count += name == representative
            alias_count += name != representative

    independent_ranking = sorted(result_rows.values(), key=selection_key)
    if [row["name"] for row in independent_ranking] != [row["name"] for row in result["ranking"]]:
        raise ValidationError("reported ranking differs from independent ranking")
    selected = result["selected"]
    if selected["name"] != independent_ranking[0]["name"] or selected["summary"] != independent_ranking[0]["summary"]:
        raise ValidationError("selected winner differs from the independent argmax")
    hybrid = result.get("hybrid", {})
    if (
        hybrid.get("source_native_configuration") != selected["name"]
        or hybrid.get("coverage", {}).get("expected_fragment_count") != 105
        or hybrid.get("coverage", {}).get("train_fragment_count") != 105
    ):
        raise ValidationError("fit105 hybrid receipt is incomplete")

    validation = {
        "schema": "celltrack-alpha.trackastra-native-fit105-selection-validation.v1",
        "status": "PASS_INDEPENDENT_108_WAY_SELECTION",
        "fit105_results": str(args.result),
        "fit105_results_sha256": sha256(args.result),
        "configuration_count": len(expected_configs),
        "score_receipt_count": len(score_rows),
        "structural_manifest_count": len(manifests),
        "identity_count": len(identities or ()),
        "unique_graph_count": len(by_digest),
        "direct_exact_score_count": direct_count,
        "byte_identical_alias_count": alias_count,
        "selected_name": selected["name"],
        "selected_summary": selected["summary"],
        "hybrid_summary": hybrid["summary"],
        "cv57_access": False,
        "selection_recomputed_independently": True,
        "submission": False,
    }
    atomic_json(args.validation, validation)
    return validation


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--materialized", type=Path, default=DEFAULT_MATERIALIZED)
    value.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    value.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    return value


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
