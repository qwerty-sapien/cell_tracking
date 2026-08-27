#!/usr/bin/env python3
"""Build a fail-closed all-pairs observation matrix for the P0B ladder."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import compare_submissions as compare  # noqa: E402


SCHEMA = "celltrack-alpha.dispersion-ladder-observation-matrix.v1"
DEFAULT_OUTPUT = (
    ROOT
    / "reports/g02/screening/dispersion_ladder_observation_matrix_v1.json"
)
COMPARISON_TOOL_SHA256 = (
    "ade5dd17ebb04117540e9f4d480334fb9a50a25b63f362f023c8b4700c23adae"
)


class DispersionMatrixError(RuntimeError):
    """A frozen ladder identity or streaming observation comparison failed."""


@dataclass(frozen=True)
class Rung:
    rung_id: str
    label: str
    mechanism: str
    local_cv57: float
    fragment_dir: Path
    report: Path
    report_sha256: str
    edge_tp: int
    edge_fp: int
    edge_fn: int
    public_displayed: float | None


RUNGS = (
    Rung(
        "c0",
        "backward motion c=0",
        "graph cost/solver",
        0.8483945155545333,
        ROOT
        / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/"
        "motion_error_weight_0/fragments",
        ROOT
        / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/report.json",
        "409d632879bdbc7220ffdb48b900580a0c4d9c7f5ec64da69883e931859b844f",
        33787,
        3324,
        2481,
        0.860,
    ),
    Rung(
        "c2",
        "backward motion c=2",
        "graph cost/solver",
        0.8721499474601118,
        ROOT
        / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/"
        "motion_error_weight_2/fragments",
        ROOT
        / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/report.json",
        "409d632879bdbc7220ffdb48b900580a0c4d9c7f5ec64da69883e931859b844f",
        34178,
        2699,
        2090,
        0.878,
    ),
    Rung(
        "c0p25",
        "backward motion c=0.25",
        "graph cost/solver",
        0.8731495935289003,
        ROOT
        / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/"
        "motion_error_weight_0p25/fragments",
        ROOT
        / "reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/report.json",
        "409d632879bdbc7220ffdb48b900580a0c4d9c7f5ec64da69883e931859b844f",
        34205,
        2685,
        2063,
        0.877,
    ),
    Rung(
        "p4_no_attention",
        "P4 no-attention control",
        "learned association architecture/control",
        0.8742900459910163,
        ROOT / "reports/g02/p4_edge_attention_cv57_v1/no_attention/fragments",
        ROOT / "reports/g02/p4_edge_attention_cv57_v1/report.json",
        "914989293b9f51ce8c92bf20cac8e8a37f8e115c16664c1231e9c0836d441b20",
        34227,
        2658,
        2041,
        None,
    ),
    Rung(
        "apncc_global",
        "APNCC global re-solve",
        "appearance association postprocessor + global solver",
        0.877400878279539,
        ROOT / "reports/g02/selected_graph_appearance_cv57_v1/fragments",
        ROOT / "reports/g02/selected_graph_appearance_cv57_v1/report.json",
        "cfa24b670b09fd42786d1748457d15d4cbdaa455bcdc51224c230fe42757cbbb",
        34281,
        2584,
        1987,
        None,
    ),
    Rung(
        "apncc_direct",
        "APNCC exact direct swap",
        "appearance association postprocessor + local direct solver",
        0.8815220699597315,
        ROOT
        / "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments",
        ROOT
        / "reports/g02/selected_graph_appearance_direct_cv57_v1/report.json",
        "8da96a3da9e66869192f446fc6d59fedd6016bdefe4fb272b136df68258a0df4",
        34320,
        2459,
        1948,
        None,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fragment_files(path: Path) -> dict[str, Path]:
    """Index a fragment directory without retaining any prediction graph."""

    if not path.is_dir():
        raise DispersionMatrixError(f"fragment directory is missing: {path}")
    result: dict[str, Path] = {}
    for source in sorted(path.glob("*.csv")):
        graphs = compare.read_submission(source)
        if len(graphs) != 1:
            raise DispersionMatrixError(
                f"fragment must contain exactly one dataset: {source}"
            )
        dataset = next(iter(graphs))
        if dataset in result:
            raise DispersionMatrixError(
                f"dataset {dataset} repeats in {path}"
            )
        result[dataset] = source
    if len(result) != 57:
        raise DispersionMatrixError(
            f"expected 57 fragments in {path}, found {len(result)}"
        )
    return result


def canonical_streaming_sha256(files: Mapping[str, Path]) -> str:
    """Match compare_submissions canonical hashing one dataset at a time."""

    digest = hashlib.sha256()
    for dataset, path in sorted(files.items()):
        graph = compare.read_submission(path)[dataset]
        digest.update(
            compare._canonical_json(compare._graph_semantics(graph))
        )
        digest.update(b"\n")
    return digest.hexdigest()


def compare_fragment_dirs(
    left: Path,
    right: Path,
    *,
    max_distance_um: float = 7.0,
) -> dict[str, Any]:
    """Use Phase-0 graph semantics while bounding memory to one video pair."""

    files_left = fragment_files(left)
    files_right = fragment_files(right)
    if set(files_left) != set(files_right):
        raise DispersionMatrixError(
            "pairwise fragment dataset identities disagree"
        )
    rows = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for dataset in sorted(files_left):
        graph_left = compare.read_submission(files_left[dataset])[dataset]
        graph_right = compare.read_submission(files_right[dataset])[dataset]
        row = compare.compare_graphs(
            graph_left,
            graph_right,
            max_distance_um=max_distance_um,
        )
        rows.append(row)
        by_family.setdefault(row["family"], []).append(row)
    aggregate = compare.aggregate_rows(rows)
    return {
        "aggregate": aggregate,
        "by_family": {
            family: compare.aggregate_rows(family_rows)
            for family, family_rows in sorted(by_family.items())
        },
        "observation_class": compare.classify_distance(aggregate),
    }


def validate_rungs(rungs: Sequence[Rung]) -> list[dict[str, Any]]:
    if len(rungs) != 6 or len({rung.rung_id for rung in rungs}) != len(rungs):
        raise DispersionMatrixError("ladder must contain six unique rungs")
    report_cache: dict[Path, dict[str, Any]] = {}
    records = []
    for rung in rungs:
        if sha256_file(rung.report) != rung.report_sha256:
            raise DispersionMatrixError(
                f"report SHA-256 mismatch for {rung.rung_id}"
            )
        report_cache.setdefault(
            rung.report, json.loads(rung.report.read_text(encoding="utf-8"))
        )
        files = fragment_files(rung.fragment_dir)
        records.append(
            {
                "rung_id": rung.rung_id,
                "label": rung.label,
                "mechanism": rung.mechanism,
                "local_cv57": rung.local_cv57,
                "edge_counts": {
                    "edge_tp": rung.edge_tp,
                    "edge_fp": rung.edge_fp,
                    "edge_fn": rung.edge_fn,
                },
                "public_displayed": rung.public_displayed,
                "public_interval": (
                    None
                    if rung.public_displayed is None
                    else [
                        rung.public_displayed - 0.0005,
                        rung.public_displayed + 0.0005,
                    ]
                ),
                "fragment_dir": str(rung.fragment_dir),
                "fragment_count": len(files),
                "canonical_sha256": canonical_streaming_sha256(files),
                "source_report": str(rung.report),
                "source_report_sha256": rung.report_sha256,
            }
        )
    return records


def build_matrix(rungs: Sequence[Rung] = RUNGS) -> dict[str, Any]:
    if sha256_file(ROOT / "scripts/compare_submissions.py") != COMPARISON_TOOL_SHA256:
        raise DispersionMatrixError("comparison tool SHA-256 drifted")
    rung_records = validate_rungs(rungs)
    comparisons = []
    for left, right in itertools.combinations(rungs, 2):
        result = compare_fragment_dirs(left.fragment_dir, right.fragment_dir)
        comparisons.append(
            {
                "a": left.rung_id,
                "b": right.rung_id,
                "local_delta_a_minus_b": left.local_cv57 - right.local_cv57,
                **result,
            }
        )
        print(
            f"{left.rung_id} vs {right.rung_id}: "
            f"{result['observation_class']['label']} "
            f"node={result['aggregate']['nodes']['jaccard']:.9f} "
            f"edge={result['aggregate']['edges_raw']['jaccard']:.9f}",
            flush=True,
        )
    return {
        "schema": SCHEMA,
        "status": "completed",
        "scope": {
            "video_count": 57,
            "rung_count": len(rungs),
            "pair_count": len(comparisons),
            "streaming_max_live_video_pairs": 1,
            "gt_rescoring_performed": False,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
        "method": {
            "comparison_tool": str(ROOT / "scripts/compare_submissions.py"),
            "comparison_tool_sha256": COMPARISON_TOOL_SHA256,
            "matching": (
                "exact ID+coordinate, exact coordinate, then deterministic "
                "official-style physical matching at 7 um"
            ),
            "distance_vector": [
                "node_jaccard",
                "raw_edge_jaccard",
                "scorer_eligible_edge_jaccard",
                "fork_agreement",
            ],
            "null_envelope": compare.ZERO_NULL_ENVELOPE,
        },
        "rungs": rung_records,
        "comparisons": comparisons,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output.exists():
        raise DispersionMatrixError(
            f"refusing to overwrite output: {args.output}"
        )
    result = build_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"output": str(args.output), "pairs": 15}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
