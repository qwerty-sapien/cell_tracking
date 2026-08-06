#!/usr/bin/env python3
"""Dry-run the divisions-on Lane 1 emitter on the banked APNCC 44b6 graph.

This is a packaging rehearsal only.  It reuses the frozen D1 receipt, verifies
the banked APNCC fragment hashes, emits an empty-candidate submission.csv, and
proves that the canonical scorer input graph (and therefore its score) is
unchanged.  It does not rerun the official scorer or access any new data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "kaggle_kernel" / "g02_division_lane1"
APNCC_REPORT = ROOT / "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json"
APNCC_REPORT_SHA256 = "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d"
DEFAULT_REPORT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division"
    / "LANE1_DIVISION_DRY_RUN.json"
)

sys.path.insert(0, str(ROOT / "src"))
from celltrack_alpha.action_compiler import graph_sha256  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission, write_submission  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def aggregate_graph_sha256(graphs: dict[str, Any]) -> str:
    return canonical_sha256(
        [
            {"dataset": dataset, "graph_sha256": graph_sha256(graphs[dataset])}
            for dataset in sorted(graphs)
        ]
    )


def banked_apncc_graph() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(APNCC_REPORT) != APNCC_REPORT_SHA256:
        raise RuntimeError("banked APNCC report SHA-256 drifted")
    report = json.loads(APNCC_REPORT.read_text(encoding="utf-8"))
    control = report.get("graph", {}).get("control", {})
    provenance = control.get("input_provenance", {})
    files = provenance.get("files")
    if (
        provenance.get("file_count") != 36
        or not isinstance(files, list)
        or len(files) != 36
        or control.get("coverage", {}).get("missing_relevant_fragments") != []
        or control.get("coverage", {}).get("duplicate_relevant_fragments") != {}
    ):
        raise RuntimeError("banked APNCC fragment coverage is incomplete")
    for item in files:
        path = Path(str(item["path"]))
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"banked APNCC fragment drifted: {path}")
    return control, files


def append_fragment(source: Path, destination: Path, next_id: int) -> int:
    """Append a schema-valid fragment while re-numbering global submission IDs."""

    with source.open("r", encoding="utf-8", newline="") as input_handle, destination.open(
        "a", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames != [
            "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
        ]:
            raise RuntimeError(f"unexpected output schema in fragment: {source}")
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
        for row in reader:
            row["id"] = str(next_id)
            writer.writerow(row)
            next_id += 1
    return next_id


def load_packaged_emitter(package: Path):
    """Load the built package once; every fragment uses that exact runtime source."""

    package_text = str(package)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    source = package / "division_submission_kernel.py"
    spec = importlib.util.spec_from_file_location("lane1_packaged_emitter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the built Lane 1 emitter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dry_run(package: Path, report_path: Path) -> dict[str, Any]:
    control, files = banked_apncc_graph()
    output_dir = package / "dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "submission.csv"
    emitter = load_packaged_emitter(package)
    with tempfile.TemporaryDirectory(prefix="lane1-apncc-", dir=output_dir) as temporary:
        temporary_root = Path(temporary)
        staged_output = temporary_root / "submission.csv"
        with staged_output.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
                ],
            ).writeheader()
        base_graph_hashes: list[dict[str, str]] = []
        output_graph_hashes: list[dict[str, str]] = []
        fragment_receipts: list[dict[str, Any]] = []
        next_id = 0
        for index, item in enumerate(sorted(files, key=lambda row: str(row["identity"]))):
            base_csv = Path(str(item["path"]))
            base_graphs = read_submission(base_csv)
            if len(base_graphs) != 1:
                raise RuntimeError(f"banked fragment is not single-dataset: {base_csv}")
            dataset, base_graph = next(iter(base_graphs.items()))
            base_graph_hash = graph_sha256(base_graph)
            empty_candidates = temporary_root / f"empty_candidates_{index:02d}.json"
            empty_candidates.write_text(
                json.dumps(
                    {
                        "schema": "celltrack-alpha.d1-division-candidates.v1",
                        "base_submission_sha256": sha256_file(base_csv),
                        "candidates": [],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            fragment_output = temporary_root / f"output_{index:02d}.csv"
            receipt = emitter.emit_submission(
                base_submission=base_csv,
                candidate_receipt=empty_candidates,
                output=fragment_output,
                allow_empty=True,
            )
            if (
                receipt.get("verdict") != "PASS_EMPTY_DIVISIONS_ON"
                or receipt.get("accepted_forks") != 0
                or receipt.get("duplicate_survivors") != 0
                or receipt.get("score_change") != 0.0
                or receipt.get("base_graph_sha256") != receipt.get("output_graph_sha256")
                or receipt.get("official_scorer_rerun") is not False
            ):
                raise RuntimeError("empty division emitter did not preserve a scorer input fragment")
            output_graphs = read_submission(fragment_output)
            output_dataset, output_graph = next(iter(output_graphs.items()))
            if output_dataset != dataset or graph_sha256(output_graph) != base_graph_hash:
                raise RuntimeError("fragment schema round-trip changed the banked APNCC graph")
            base_graph_hashes.append({"dataset": dataset, "graph_sha256": base_graph_hash})
            output_graph_hashes.append({"dataset": dataset, "graph_sha256": graph_sha256(output_graph)})
            next_id = append_fragment(fragment_output, staged_output, next_id)
            fragment_receipts.append(
                {
                    "dataset": dataset,
                    "base_submission_sha256": sha256_file(base_csv),
                    "output_submission_sha256": receipt["output_submission_sha256"],
                    "output_graph_sha256": receipt["output_graph_sha256"],
                }
            )
        base_graph_sha256 = canonical_sha256(sorted(base_graph_hashes, key=lambda row: row["dataset"]))
        output_graph_sha256 = canonical_sha256(sorted(output_graph_hashes, key=lambda row: row["dataset"]))
        if base_graph_sha256 != output_graph_sha256:
            raise RuntimeError("streamed submission.csv changed the banked APNCC scorer input graph")
        staged_output.replace(output_csv)
    result = {
        "schema": "celltrack-alpha.lane1-division-dry-run.v1",
        "purpose": "empty divisions-on output-schema rehearsal only",
        "banked_apncc": {
            "report": str(APNCC_REPORT.relative_to(ROOT)),
            "report_sha256": APNCC_REPORT_SHA256,
            "fragment_count": len(files),
            "frozen_score_summary": control["summary"],
            "canonical_graph_sha256": base_graph_sha256,
        },
        "output": {
            "submission": str(output_csv.relative_to(ROOT)),
            "submission_sha256": sha256_file(output_csv),
            "fragments_validated": len(fragment_receipts),
            "rows": next_id,
            "schema_valid": True,
            "divisions_enabled": True,
            "accepted_forks": 0,
            "duplicate_survivors": 0,
        },
        "score": {
            "official_scorer_rerun": False,
            "score_change": 0.0,
            "basis": "each emitted fragment retained its exact canonical scorer-input graph; the streamed submission.csv is their schema-valid union",
        },
        "frozen_d1": receipt["frozen_contract"],
        "fragments": fragment_receipts,
        "verdict": "PASS_EMPTY_DIVISIONS_ON_SCHEMA_ROUND_TRIP",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = dry_run(args.package.resolve(), args.report.resolve())
    print(json.dumps({key: result[key] for key in ("verdict", "score")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
