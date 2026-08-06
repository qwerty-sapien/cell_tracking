#!/usr/bin/env python3
"""Exactly score the sealed native CV57 graph and preregistered APNCC/fork hybrid."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import evaluate_trackastra_native_base_wave as frozen  # noqa: E402
import scoreall  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    read_submission,
    write_submission,
)


EXPECTED_SELECTION = "r08_c08_a035_off"
EXPECTED_SELECTION_SEAL_SHA256 = "0c1b3ed028bfdef4dd3672cdb70afce63b408747d6c088d398ca86b2a9836bcc"
EXPECTED_FIT_RESULT_SHA256 = "36d6aa953ca32e3231624838f20e84691aaf184e0e19028f6e9d79120a31a3f9"
EXPECTED_FROZEN_EVALUATOR_SHA256 = "458e958fa856c1643577ee41f0eea3a9b45de438064d52a6a87e20906489863e"
EXPECTED_CONTROL_REPORT_SHA256 = "8da96a3da9e66869192f446fc6d59fedd6016bdefe4fb272b136df68258a0df4"
CONTROL_SCORE = 0.8815220699597315


class OneShotError(RuntimeError):
    """A selection, population, graph, scorer, or one-shot invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def graph_signature(graph) -> tuple[Any, ...]:
    nodes = tuple(
        sorted((node_id, node.t, node.z, node.y, node.x) for node_id, node in graph.nodes.items())
    )
    edges = tuple(sorted((edge.source_id, edge.target_id) for edge in graph.edges))
    return graph.dataset, nodes, edges


def verify_control(report_path: Path, fragment_root: Path, identities: tuple[str, ...]) -> dict:
    if sha256(report_path) != EXPECTED_CONTROL_REPORT_SHA256:
        raise OneShotError("APNCC CV57 control report hash mismatch")
    report = json.loads(report_path.read_text(encoding="ascii"))
    control = report.get("candidate", {})
    if (
        float(control.get("summary", {}).get("score", -1)) != CONTROL_SCORE
        or int(control.get("coverage", {}).get("expected_fragment_count", -1)) != 57
    ):
        raise OneShotError("APNCC CV57 control vector mismatch")
    provenance = control.get("input_provenance", {}).get("files", [])
    if tuple(str(row.get("identity")) for row in provenance) != identities:
        raise OneShotError("APNCC CV57 identity order mismatch")
    for row in provenance:
        fragment = fragment_root / f"{row['identity'].replace(':', '__', 1)}.csv"
        if not fragment.is_file() or sha256(fragment) != row.get("sha256"):
            raise OneShotError(f"APNCC CV57 fragment mismatch: {row['identity']}")
    return control


def materialize_hybrid(
    *,
    output: Path,
    identities: tuple[str, ...],
    apncc_root: Path,
    native_root: Path,
) -> tuple[Path, dict[str, int]]:
    destination = output / "hybrid" / "fragments"
    destination.mkdir(parents=True)
    totals: Counter[str] = Counter()
    for identity in identities:
        filename = f"{identity.replace(':', '__', 1)}.csv"
        apncc_graphs = read_submission(apncc_root / filename)
        native_graphs = read_submission(native_root / filename)
        if len(apncc_graphs) != 1 or len(native_graphs) != 1:
            raise OneShotError(f"graph ambiguity for {identity}")
        apncc = next(iter(apncc_graphs.values()))
        native = next(iter(native_graphs.values()))
        hybrid, counts = frozen.hybrid_graph(apncc, native)
        if counts["native_forks"] != 0 or graph_signature(hybrid) != graph_signature(apncc):
            raise OneShotError(f"sealed no-fork hybrid is not APNCC-identical for {identity}")
        write_submission({hybrid.dataset: hybrid}, destination / filename)
        totals.update(counts)
    if totals["native_forks"] != 0 or totals["inserted_forks"] != 0:
        raise OneShotError("sealed no-fork hybrid unexpectedly changed an edge")
    return destination, dict(totals)


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise OneShotError(f"refusing to overwrite output: {args.output}")
    if (
        sha256(args.selection_seal) != EXPECTED_SELECTION_SEAL_SHA256
        or sha256(args.fit_result) != EXPECTED_FIT_RESULT_SHA256
        or sha256(Path(frozen.__file__).resolve()) != EXPECTED_FROZEN_EVALUATOR_SHA256
    ):
        raise OneShotError("sealed fit selection or frozen hybrid implementation hash mismatch")
    seal = json.loads(args.selection_seal.read_text(encoding="ascii"))
    fit = json.loads(args.fit_result.read_text(encoding="ascii"))
    native_report = json.loads(args.native_report.read_text(encoding="ascii"))
    population = json.loads(args.population_manifest.read_text(encoding="ascii"))
    identities = tuple(str(row["identity"]) for row in population.get("entries", []))
    if (
        seal.get("status") != "IRREVOCABLY_SELECTED_BEFORE_CV57"
        or seal.get("cv57_access_before_seal") is not False
        or seal.get("selected", {}).get("name") != EXPECTED_SELECTION
        or fit.get("selected", {}).get("name") != EXPECTED_SELECTION
        or native_report.get("status") != "COMPLETE"
        or native_report.get("selected_configuration_name") != EXPECTED_SELECTION
        or native_report.get("fit_selection_seal_sha256") != EXPECTED_SELECTION_SEAL_SHA256
        or native_report.get("aggregate", {}).get("video_count") != 57
        or population.get("scope", {}).get("cv57_only") is not True
        or len(identities) != 57
    ):
        raise OneShotError("sealed selection, native run, or CV57 population mismatch")
    expected_csv = {f"{identity.replace(':', '__', 1)}.csv" for identity in identities}
    actual_csv = {path.name for path in args.native_fragments.glob("*.csv")}
    if actual_csv != expected_csv:
        raise OneShotError("native CV57 fragment population mismatch")
    control = verify_control(args.control_report, args.apncc_fragments, identities)
    scorer_hashes = scoreall.verify_scorer_sources(
        json.loads(args.split.read_text(encoding="ascii"))
    )
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    native = frozen.score_one(
        (EXPECTED_SELECTION, str(args.native_fragments), identities, str(args.zip))
    )
    hybrid_root, hybrid_totals = materialize_hybrid(
        output=args.output,
        identities=identities,
        apncc_root=args.apncc_fragments,
        native_root=args.native_fragments,
    )
    hybrid = frozen.score_one(
        ("APNCC_NATIVE_FORK_HYBRID", str(hybrid_root), identities, str(args.zip))
    )
    for row in (native, hybrid):
        row["delta_vs_apncc_control"] = float(row["summary"]["score"]) - CONTROL_SCORE
    hybrid["totals"] = hybrid_totals
    ranked = sorted(
        [
            {"name": "APNCC_CONTROL", "summary": control["summary"], "delta_vs_apncc_control": 0.0},
            {"name": hybrid["name"], "summary": hybrid["summary"], "delta_vs_apncc_control": hybrid["delta_vs_apncc_control"]},
            {"name": native["name"], "summary": native["summary"], "delta_vs_apncc_control": native["delta_vs_apncc_control"]},
        ],
        key=lambda row: (-float(row["summary"]["score"]), row["name"]),
    )
    report = {
        "schema": "celltrack-alpha.trackastra-native-cv57-one-shot.v1",
        "status": "COMPLETE_EXACT_ONE_SHOT_CV57",
        "selection": {
            "name": EXPECTED_SELECTION,
            "seal": str(args.selection_seal),
            "seal_sha256": EXPECTED_SELECTION_SEAL_SHA256,
            "fit_result_sha256": EXPECTED_FIT_RESULT_SHA256,
        },
        "population": {
            "video_count": 57,
            "family_counts": population["scope"]["family_counts"],
            "identities": list(identities),
            "alternative_variant_access": False,
            "tuning": False,
            "repeat_cv57": False,
        },
        "apncc_control": {
            "summary": control["summary"],
            "edge_counts": control["edge_counts"],
            "source_report": str(args.control_report),
            "source_report_sha256": EXPECTED_CONTROL_REPORT_SHA256,
        },
        "native": native,
        "hybrid": hybrid,
        "ranking": ranked,
        "beats_apncc": any(
            float(row["summary"]["score"]) > CONTROL_SCORE
            for row in (native, hybrid)
        ),
        "provenance": {
            "native_report": str(args.native_report),
            "native_report_sha256": sha256(args.native_report),
            "native_fragments": str(args.native_fragments),
            "population_manifest": str(args.population_manifest),
            "population_manifest_sha256": sha256(args.population_manifest),
            "frozen_hybrid_evaluator_sha256": EXPECTED_FROZEN_EVALUATOR_SHA256,
            "frozen_scorer_source_hashes": scorer_hashes,
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "cv57_measurement_count": 1,
            "kaggle_inference_action": True,
            "public_submission": False,
            "runpod_action": False,
        },
    }
    atomic_json(args.output / "CV57_ONE_SHOT_RESULTS.json", frozen.safe_number(report))
    return report


def parser() -> argparse.ArgumentParser:
    branch = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--selection-seal", type=Path, default=branch / "FIT_SELECTION_SEAL.json")
    value.add_argument("--fit-result", type=Path, default=branch / "fit105_sweep_scored/FIT105_RESULTS.json")
    value.add_argument("--native-report", type=Path, required=True)
    value.add_argument("--native-fragments", type=Path, required=True)
    value.add_argument("--population-manifest", type=Path, default=ROOT / "kaggle_kernel/g02-trackastra-native-cv57-v1/cv57-cache-manifest.json")
    value.add_argument("--control-report", type=Path, default=ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/report.json")
    value.add_argument("--apncc-fragments", type=Path, default=ROOT / "reports/g02/selected_graph_appearance_direct_cv57_v1/fragments")
    value.add_argument("--split", type=Path, default=branch / "FIT_SCORER_SOURCE_MANIFEST_REPAIR_1.json")
    value.add_argument("--zip", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip")
    value.add_argument("--output", type=Path, default=branch / "cv57_one_shot")
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(
        json.dumps(
            {
                "native": result["native"]["summary"],
                "hybrid": result["hybrid"]["summary"],
                "beats_apncc": result["beats_apncc"],
            },
            sort_keys=True,
        )
    )
