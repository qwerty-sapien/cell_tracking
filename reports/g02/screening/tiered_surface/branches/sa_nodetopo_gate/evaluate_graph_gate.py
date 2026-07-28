#!/usr/bin/env python3
"""Evaluate the exact committed BLOCKPEAK restore population at the 44b6 graph gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import warnings
import zipfile

import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREGISTRATION.json"
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
)
from celltrack_alpha.kaggle_calibration import SubmissionNode, read_submission  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    official_metric_row,
    official_summary,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


class GateError(RuntimeError):
    """Fail-closed contract violation."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"JSON root is not an object: {path}")
    return value


def central_directory_identity(path: Path) -> tuple[str, int]:
    with zipfile.ZipFile(path) as archive:
        rows = [
            {
                "compressed_size": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "filename": info.filename,
                "uncompressed_size": int(info.file_size),
            }
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
        ]
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii") + b"\n"
    return hashlib.sha256(encoded).hexdigest(), len(rows)


def committed_blob_sha256(relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=HERE,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError as exc:
        raise GateError(f"expected root-scoped path: {path}") from exc


def authenticate() -> tuple[
    dict[str, Any],
    tuple[str, ...],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    prereg = load_json(PREREG)
    if prereg.get("outcomes_opened") is not False:
        raise GateError("preregistration must precede outcomes")
    receipts: dict[str, Any] = {}
    for label, pin in prereg["frozen_inputs"].items():
        if label == "raw_archive":
            path = Path(pin["path"])
            if not path.is_file() or path.stat().st_size != int(pin["bytes"]):
                raise GateError("raw archive byte identity mismatch")
            central_sha, count = central_directory_identity(path)
            if (
                central_sha != pin["central_directory_sha256"]
                or count != int(pin["entry_count"])
            ):
                raise GateError("raw archive central directory drifted")
            receipts[label] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "central_directory_sha256": central_sha,
                "entry_count": count,
                "full_sha256": pin["sha256_banked_in_p0"],
                "full_sha256_receipt_source": prereg["frozen_inputs"]["p0"]["path"],
            }
            continue
        path = Path(pin["path"])
        observed = sha256(path)
        expected = pin["sha256"]
        if expected == "PIN_AT_EXECUTION_BEFORE_OUTCOME":
            expected = observed
        if observed != expected:
            raise GateError(f"{label} hash mismatch: {observed} != {expected}")
        receipt: dict[str, Any] = {"path": str(path), "sha256": observed}
        if pin.get("must_be_committed_at_head") is True:
            relative = _relative_to_root(path)
            committed = committed_blob_sha256(relative)
            if committed != observed:
                raise GateError(f"{label} root bytes differ from HEAD blob")
            receipt["committed_blob_sha256"] = committed
            receipt["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
            ).strip()
        receipts[label] = receipt

    identity_source = load_json(Path(prereg["frozen_inputs"]["fit105_identity_source"]["path"]))
    identities = tuple(
        sorted(
            str(row["identity"])
            for row in identity_source["files"]
            if str(row["identity"]).startswith("train:44b6_")
        )
    )
    if len(identities) != 36 or len(set(identities)) != 36:
        raise GateError("fit105 44b6 identity coverage drifted")

    population_path = Path(prereg["frozen_inputs"]["action_population"]["path"])
    with gzip.open(population_path, "rt", encoding="ascii") as handle:
        actions = [json.loads(line) for line in handle if line.strip()]
    action_ids = [str(row["action_id"]) for row in actions]
    action_identities = {str(row["identity"]) for row in actions}
    if (
        len(actions) != 50
        or len(set(action_ids)) != 50
        or len(action_identities) != 24
        or not action_identities <= set(identities)
        or {str(row["action_kind"]) for row in actions} != {"restore_topology"}
    ):
        raise GateError("committed action population contract drifted")

    manifest = load_json(Path(prereg["frozen_inputs"]["fit117_baseline_manifest"]["path"]))
    fragments_root = ROOT / "reports/g02/fitcache_117/bh9out/fragments"
    fragment_receipts = []
    for identity in identities:
        filename = identity.replace(":", "__", 1) + ".csv"
        path = fragments_root / filename
        expected = manifest["videos"][identity]["files"]["fragments"]
        observed = sha256(path)
        if path.stat().st_size != int(expected["bytes"]) or observed != expected["sha256"]:
            raise GateError(f"baseline fragment drifted: {identity}")
        fragment_receipts.append(
            {
                "identity": identity,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    receipts["baseline_fragments"] = fragment_receipts
    stress = load_json(Path(prereg["frozen_inputs"]["stress_curve"]["path"]))
    return prereg, identities, actions, stress, receipts


def graph_action(row: Mapping[str, Any]) -> GraphAction:
    return GraphAction(
        action_id=str(row["action_id"]),
        kind=ActionKind.RESTORE_TOPOLOGY,
        add_nodes=tuple(
            SubmissionNode(
                node_id=int(node["node_id"]),
                t=int(node["t"]),
                z=int(node["z"]),
                y=int(node["y"]),
                x=int(node["x"]),
            )
            for node in row["add_nodes"]
        ),
        add_edges=tuple(
            (int(source), int(target)) for source, target in row["add_edges"]
        ),
        replace_outgoing_from=tuple(int(value) for value in row["replace_outgoing_from"]),
        replace_incoming_to=tuple(int(value) for value in row["replace_incoming_to"]),
        expected_conflict_edges=tuple(
            (int(source), int(target))
            for source, target in row["expected_conflict_edges"]
        ),
    )


def metric(graph: Any, gt: Mapping[str, Any], scale: tuple[float, float, float]) -> dict[str, Any]:
    nodes, edges = scoreall._prediction_rows(graph.dataset, graph)
    return dict(official_metric_row(nodes, edges, dict(gt), scale))


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dict(official_summary([dict(row) for row in rows]))
    for key in (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
    ):
        result[key] = int(sum(int(float(row[key])) for row in rows))
    return result


def gate_decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    adjusted = float(candidate["adj_edge_jaccard"]) - float(
        baseline["adj_edge_jaccard"]
    )
    raw = float(candidate["edge_jaccard"]) - float(baseline["edge_jaccard"])
    node = float(candidate["node_recall"]) - float(baseline["node_recall"])
    finite = all(math.isfinite(value) for value in (adjusted, raw, node))
    checks = {
        "adjusted_ge_0p005": finite
        and adjusted >= float(gate["minimum_adjusted_edge_jaccard_delta"]),
        "raw_ge_0": finite
        and raw >= float(gate["minimum_raw_edge_jaccard_delta"]),
        "node_ge_0": finite and node >= float(gate["minimum_node_recall_delta"]),
    }
    passed = all(checks.values())
    return {
        "verdict": gate["pass_verdict"] if passed else gate["fail_verdict"],
        "passed": passed,
        "adjusted_edge_jaccard_delta": adjusted,
        "raw_edge_jaccard_delta": raw,
        "node_recall_delta": node,
        "checks": checks,
        "second_family_opened": False,
        "cv57_opened": False,
        "pass_unblocks_proposed": (
            ["DET-F0", "DET-F2", "DET-F3", "NODERESTORE-O0"] if passed else []
        ),
        "instantiation_tier_proposal": (
            None
            if passed
            else {
                "id": "BLOCKPEAK-RESTORE-V1",
                "tier": "PARK_C",
                "scope": "this exact 50-action instantiation only",
                "broader_family_tier_changes": [],
            }
        ),
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    prereg, identities, action_rows, stress, receipts = authenticate()
    actions_by_identity: dict[str, list[GraphAction]] = defaultdict(list)
    for row in action_rows:
        actions_by_identity[str(row["identity"])].append(graph_action(row))

    budget_50 = next(
        row for row in stress["false_node_budget_stress_curve"] if int(row["budget"]) == 50
    )
    expected_batches = budget_50["batch_receipts"]
    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    per_video = []
    compiled_single_count = 0
    compiled_batch_count = 0
    archive_path = Path(prereg["frozen_inputs"]["raw_archive"]["path"])
    fragments_root = ROOT / "reports/g02/fitcache_117/bh9out/fragments"
    with ZipStore(archive_path, mode="r") as store:
        for index, identity in enumerate(identities, start=1):
            dataset = identity.split(":", 1)[1]
            fragment_path = fragments_root / (identity.replace(":", "__", 1) + ".csv")
            graphs = read_submission(fragment_path)
            baseline = graphs.get(dataset)
            if baseline is None or len(graphs) != 1:
                raise GateError(f"fragment identity mismatch: {identity}")
            selected = actions_by_identity.get(identity, [])
            single_receipts = []
            for action in selected:
                compiled = compile_action(baseline, action)
                compiled_single_count += 1
                single_receipts.append(
                    {
                        "action_id": action.action_id,
                        "action_sha256": compiled.action_sha256,
                        "before_graph_sha256": compiled.before_graph_sha256,
                        "after_graph_sha256": compiled.after_graph_sha256,
                        "removed_edges": [list(pair) for pair in compiled.removed_edges],
                    }
                )
            batch_receipt = None
            if selected:
                compiled_batch = compile_batch(baseline, selected)
                compiled_batch_count += 1
                expected = expected_batches[identity]
                for key in ("before_graph_sha256", "after_graph_sha256", "batch_sha256"):
                    if getattr(compiled_batch, key) != expected[key]:
                        raise GateError(f"{identity} N1 batch receipt drifted at {key}")
                candidate = compiled_batch.graph
                batch_receipt = {
                    "action_count": len(selected),
                    "batch_sha256": compiled_batch.batch_sha256,
                    "before_graph_sha256": compiled_batch.before_graph_sha256,
                    "after_graph_sha256": compiled_batch.after_graph_sha256,
                    "removed_edges": [list(pair) for pair in compiled_batch.removed_edges],
                }
            else:
                candidate = baseline

            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            gt = read_geff_arrays(store, "train", dataset)
            baseline_metric = metric(baseline, gt, scale)
            candidate_metric = metric(candidate, gt, scale)
            baseline_rows.append(baseline_metric)
            candidate_rows.append(candidate_metric)
            per_video.append(
                {
                    "identity": identity,
                    "action_count": len(selected),
                    "baseline_fragment_sha256": sha256(fragment_path),
                    "single_action_receipts": single_receipts,
                    "batch_receipt": batch_receipt,
                    "baseline_metrics": baseline_metric,
                    "candidate_metrics": candidate_metric,
                }
            )
            print(
                f"[{index:02d}/36] {identity} actions={len(selected)}",
                flush=True,
            )
            del baseline, candidate, gt, group

    if compiled_single_count != 50 or compiled_batch_count != 24:
        raise GateError("N1 replay coverage mismatch")
    baseline_summary = summarise(baseline_rows)
    candidate_summary = summarise(candidate_rows)
    decision = gate_decision(baseline_summary, candidate_summary, prereg["gate"])
    report = {
        "schema": "celltrack-alpha.sa-nodetopo-44b6-graph-gate.v1",
        "status": decision["verdict"],
        "mechanism": prereg["mechanism"],
        "family": "44b6",
        "fit_only": True,
        "population": {
            "actions": 50,
            "affected_videos": 24,
            "scored_videos": 36,
            "action_kind": "restore_topology",
            "single_n1_replays": compiled_single_count,
            "batch_n1_replays": compiled_batch_count,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "decision": decision,
        "per_video": per_video,
        "frozen_input_receipts": receipts,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "local_only": True,
            "stream_one_video_at_a_time": True,
            "gpu_action": False,
            "kaggle_action": False,
            "budget_debit": False,
            "runpod_action": False,
            "submission_action": False,
            "external_action": False,
        },
        "scope": {
            "family_order": ["44b6", "6bba", "CV57"],
            "second_family_opened": False,
            "cv57_used": False,
            "model_fitted": False,
            "threshold_selected": False,
            "dead_proposed": False,
            "plan_amended": False,
            "preexisting_untracked_root_graph_gate_outputs_used": False,
            "sibling_live_outputs_used": False,
            "root_single_writer_files_written": False,
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise GateError(f"refusing to overwrite: {args.output}")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "elapsed_seconds": report["execution"]["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
