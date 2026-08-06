#!/usr/bin/env python3
"""Run the sealed Trackastra-native CV57 graph once from the local frozen cache."""

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

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import verify_trackastra_native_cpu_parity as core  # noqa: E402


BRANCH = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1"
CV57_CACHE_ROOT = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
EXPECTED_SELECTION = "r08_c08_a035_off"
EXPECTED_SELECTION_SEAL_SHA256 = "0c1b3ed028bfdef4dd3672cdb70afce63b408747d6c088d398ca86b2a9836bcc"
EXPECTED_FIT_RESULT_SHA256 = "36d6aa953ca32e3231624838f20e84691aaf184e0e19028f6e9d79120a31a3f9"
EXPECTED_POPULATION_FILE_SHA256 = "c9e8eafb595c424d92422af6b4f110d37c0137a39b4e24bd3e8c465c83888913"
EXPECTED_POPULATION_CONTENT_SHA256 = "9f8b2fab0ab2d66e56f60dd45969bc6c21a46c18f193fa8791743d022f0ee979"
INTERNAL_DEADLINE_SECONDS = 17_100


class OneShotError(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def journal(path: Path, value: dict) -> None:
    rendered = json.dumps(value, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="ascii") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(rendered, flush=True)


def canonical_manifest_sha(payload: dict) -> str:
    value = dict(payload)
    claimed = value.pop("manifest_sha256", None)
    computed = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if claimed != computed:
        raise OneShotError("population manifest canonical SHA mismatch")
    return computed


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(core.sha256(path)))
    return digest.hexdigest()


def verify_preaccess(args: argparse.Namespace) -> tuple[dict, dict, tuple[str, ...]]:
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    runtime_contract = contract.get("runtime", {})
    selection_seal = args.selection_seal
    fit_result = args.fit_result
    population_path = args.population_manifest
    if contract.get("status") != "IRREVOCABLE_LOCAL_ONE_SHOT_PREACCESS":
        raise OneShotError("local one-shot contract is not irrevocably sealed")
    if contract.get("cv57_scientific_measurement_before_contract") is not False:
        raise OneShotError("contract does not attest zero prior scientific CV57 measurement")
    if contract.get("selected_configuration") != EXPECTED_SELECTION:
        raise OneShotError("contract selection drift")
    if core.sha256(Path(__file__).resolve()) != runtime_contract.get("runner_sha256"):
        raise OneShotError("runner SHA drift")
    if core.sha256(Path(core.__file__).resolve()) != runtime_contract.get("inference_core_sha256"):
        raise OneShotError("inference core SHA drift")
    if core.sha256(selection_seal) != EXPECTED_SELECTION_SEAL_SHA256:
        raise OneShotError("fit selection seal SHA mismatch")
    if core.sha256(fit_result) != EXPECTED_FIT_RESULT_SHA256:
        raise OneShotError("fit result SHA mismatch")
    if core.sha256(population_path) != EXPECTED_POPULATION_FILE_SHA256:
        raise OneShotError("CV57 population manifest file SHA mismatch")
    if core.sha256(core.ASSET_ROOT / "model.pt") != core.MODEL_SHA256:
        raise OneShotError("checkpoint SHA mismatch")
    if core.sha256(core.ASSET_ROOT / "ctc.zip") != core.ARCHIVE_SHA256:
        raise OneShotError("checkpoint archive SHA mismatch")
    source_tree_sha = tree_sha256(core.TRACKASTRA_SOURCE)
    support_tree_sha = tree_sha256(core.SUPPORT_SOURCE)
    if source_tree_sha != runtime_contract.get("trackastra_source_tree_sha256"):
        raise OneShotError("Trackastra source tree SHA mismatch")
    if support_tree_sha != runtime_contract.get("support_source_tree_sha256"):
        raise OneShotError("support source tree SHA mismatch")
    for parity in contract.get("fit_cpu_parity_reports", []):
        path = ROOT / str(parity["path"])
        if core.sha256(path) != parity["sha256"]:
            raise OneShotError(f"fit-only CPU parity receipt drift: {path}")
        report = json.loads(path.read_text(encoding="ascii"))
        if report.get("status") != "PASS" or not all(
            row.get("cpu_selected_equals_t4") is True for row in report.get("rows", [])
        ):
            raise OneShotError(f"fit-only CPU graph parity failed: {path}")
    seal = json.loads(selection_seal.read_text(encoding="ascii"))
    fit = json.loads(fit_result.read_text(encoding="ascii"))
    population = json.loads(population_path.read_text(encoding="ascii"))
    if canonical_manifest_sha(population) != EXPECTED_POPULATION_CONTENT_SHA256:
        raise OneShotError("CV57 population content SHA mismatch")
    identities = tuple(str(row["identity"]) for row in population.get("entries", []))
    if (
        seal.get("status") != "IRREVOCABLY_SELECTED_BEFORE_CV57"
        or seal.get("cv57_access_before_seal") is not False
        or seal.get("selected", {}).get("name") != EXPECTED_SELECTION
        or fit.get("selected", {}).get("name") != EXPECTED_SELECTION
        or population.get("scope", {}).get("cv57_only") is not True
        or population.get("scope", {}).get("family_counts") != {"44b6": 21, "6bba": 36}
        or len(identities) != 57
        or len(set(identities)) != 57
    ):
        raise OneShotError("selection or population invariant failed")
    return contract, population, identities


def graph_from_arrays(identity: str, nodes: np.ndarray, edges: np.ndarray):
    core.configure_imports()
    from celltrack_alpha.kaggle_calibration import (
        SubmissionEdge,
        SubmissionGraph,
        SubmissionNode,
        validate_graphs,
    )

    selected = core.selected_edges(edges)
    used = {node for pair in selected for node in pair}
    node_rows = {int(row[0]): row for row in nodes if int(row[0]) in used}
    dataset = identity.split(":", 1)[1]
    graph = SubmissionGraph(
        dataset=dataset,
        nodes={
            node: SubmissionNode(
                node,
                int(row[1]),
                int(round(float(row[2]))),
                int(round(float(row[3]))),
                int(round(float(row[4]))),
            )
            for node, row in node_rows.items()
        },
        edges=[SubmissionEdge(source, target) for source, target in selected],
    )
    validate_graphs({dataset: graph})
    return graph


def prepare_output(output: Path, contract_sha: str) -> tuple[Path, Path, Path]:
    fragment_root = output / "fragments"
    access_root = output / "access"
    state_path = output / "RUN_STATE.json"
    if not output.exists():
        fragment_root.mkdir(parents=True)
        access_root.mkdir()
        atomic_json(
            state_path,
            {
                "schema": "celltrack-alpha.trackastra-native-cv57-local-run-state.v1",
                "status": "IN_PROGRESS",
                "contract_sha256": contract_sha,
                "scientific_measurement_count": 0,
                "public_submission": False,
            },
        )
    else:
        if not state_path.is_file():
            raise OneShotError("existing output lacks run state")
        state = json.loads(state_path.read_text(encoding="ascii"))
        if state.get("status") == "COMPLETE":
            raise OneShotError("refusing to repeat completed CV57 inference")
        if (
            state.get("status") != "IN_PROGRESS"
            or state.get("contract_sha256") != contract_sha
            or state.get("scientific_measurement_count") != 0
        ):
            raise OneShotError("existing partial output state drift")
        if not fragment_root.is_dir() or not access_root.is_dir():
            raise OneShotError("partial output directory structure drift")
    return fragment_root, access_root, state_path


def completed_receipt(
    identity: str, fragment_root: Path, access_root: Path
) -> dict | None:
    slug = identity.replace(":", "__", 1)
    csv_path = fragment_root / f"{slug}.csv"
    receipt_path = fragment_root / f"{slug}.json"
    marker_path = access_root / f"{slug}.started.json"
    present = (csv_path.is_file(), receipt_path.is_file(), marker_path.is_file())
    if present == (False, False, False):
        return None
    if present != (True, True, True):
        raise OneShotError(
            f"identity was accessed but did not complete; repeat forbidden: {identity}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    if (
        receipt.get("identity") != identity
        or receipt.get("sha256") != core.sha256(csv_path)
        or receipt.get("cv57_cache_access_count") != 1
    ):
        raise OneShotError(f"completed fragment receipt drift: {identity}")
    return receipt


def process_identity(
    *,
    identity: str,
    manifest_row: dict,
    model,
    output: Path,
    fragment_root: Path,
    access_root: Path,
    journal_path: Path,
) -> dict:
    core.configure_imports()
    from celltrack_alpha.kaggle_calibration import write_submission

    slug = identity.replace(":", "__", 1)
    bhc_path = CV57_CACHE_ROOT / f"{slug}.bhc"
    meta_path = CV57_CACHE_ROOT / f"{slug}.json"
    marker_path = access_root / f"{slug}.started.json"
    marker = {
        "schema": "celltrack-alpha.trackastra-native-cv57-local-access.v1",
        "identity": identity,
        "started_unix_ns": time.time_ns(),
        "cv57_cache_access_count": 1,
        "repeat_forbidden": True,
    }
    atomic_json(marker_path, marker)
    journal(journal_path, {**marker, "event": "ACCESS_STARTED"})
    if (
        not bhc_path.is_file()
        or bhc_path.stat().st_size != int(manifest_row["bhc"]["bytes"])
        or core.sha256(bhc_path) != manifest_row["bhc"]["sha256"]
        or not meta_path.is_file()
        or meta_path.stat().st_size != int(manifest_row["meta"]["bytes"])
        or core.sha256(meta_path) != manifest_row["meta"]["sha256"]
    ):
        raise OneShotError(f"CV57 cache hash mismatch after access began: {identity}")
    core.FIT_CACHE_ROOT = CV57_CACHE_ROOT
    nodes, edges, runtime = core.run_video(identity, model)
    graph = graph_from_arrays(identity, nodes, edges)
    csv_path = fragment_root / f"{slug}.csv"
    write_submission({graph.dataset: graph}, csv_path)
    receipt = {
        "schema": "celltrack-alpha.trackastra-native-cv57-local-video-graph.v1",
        "identity": identity,
        "device": "cpu",
        "selected_configuration_name": EXPECTED_SELECTION,
        "nodes": len(graph.nodes),
        "selected_edges": len(graph.edges),
        "candidate_edges": int(len(edges)),
        "forks": 0,
        "inference_calls": runtime["inference_calls"],
        "peak_tokens": runtime["peak_tokens"],
        "elapsed_seconds": runtime["elapsed_seconds"],
        "path": str(csv_path.relative_to(output)),
        "sha256": core.sha256(csv_path),
        "bhc_sha256": manifest_row["bhc"]["sha256"],
        "meta_sha256": manifest_row["meta"]["sha256"],
        "cv57_cache_access_count": 1,
        "alternative_variant_access": False,
        "gt_access": False,
    }
    atomic_json(fragment_root / f"{slug}.json", receipt)
    journal(journal_path, {**receipt, "event": "GRAPH_COMPLETE"})
    return receipt


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    contract, population, identities = verify_preaccess(args)
    if args.preflight_only:
        torch.manual_seed(20260731)
        torch.use_deterministic_algorithms(True)
        model = core.load_model()
        report = {
            "status": "PASS_PREACCESS_ONLY",
            "cv57_cache_file_access": False,
            "cv57_gt_access": False,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "identity_count_from_manifest": len(identities),
            "runner_sha256": contract["runtime"]["runner_sha256"],
        }
        print(json.dumps(report, sort_keys=True))
        return report
    contract_sha = core.sha256(args.contract)
    fragment_root, access_root, state_path = prepare_output(args.output, contract_sha)
    journal_path = args.output / "JOURNAL.jsonl"
    existing = {
        identity: completed_receipt(identity, fragment_root, access_root)
        for identity in identities
    }
    # All failure-prone imports, checkpoint checks, and model loading happen before
    # the first CV57 cache marker is written.
    torch.manual_seed(20260731)
    torch.use_deterministic_algorithms(True)
    model = core.load_model()
    by_identity = {str(row["identity"]): row for row in population["entries"]}
    results = []
    for ordinal, identity in enumerate(identities, start=1):
        if time.monotonic() - started >= INTERNAL_DEADLINE_SECONDS - 120:
            raise TimeoutError("internal 17,100-second deadline reached")
        receipt = existing[identity]
        if receipt is None:
            receipt = process_identity(
                identity=identity,
                manifest_row=by_identity[identity],
                model=model,
                output=args.output,
                fragment_root=fragment_root,
                access_root=access_root,
                journal_path=journal_path,
            )
        receipt["population_ordinal"] = ordinal
        results.append(receipt)
    counts = Counter(identity.split(":", 1)[1].split("_", 1)[0] for identity in identities)
    expected_csv = {f"{identity.replace(':', '__', 1)}.csv" for identity in identities}
    actual_csv = {path.name for path in fragment_root.glob("*.csv")}
    if actual_csv != expected_csv:
        raise OneShotError("completed CV57 graph population mismatch")
    report = {
        "schema": "celltrack-alpha.trackastra-native-cv57-local-selected-graph-run.v1",
        "status": "COMPLETE",
        "heavy_work_started": True,
        "execution": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "elapsed_seconds": time.monotonic() - started,
            "local_only": True,
            "kaggle_kernel_action": False,
            "public_submission": False,
            "runpod_action": False,
        },
        "checkpoint_sha256": core.MODEL_SHA256,
        "fit105_results_sha256": EXPECTED_FIT_RESULT_SHA256,
        "fit_selection_seal_sha256": EXPECTED_SELECTION_SEAL_SHA256,
        "selected_configuration_name": EXPECTED_SELECTION,
        "selected_configuration": {
            "name": EXPECTED_SELECTION,
            "solver": "trackastra-global-descending-weight-greedy",
            "radius_um": core.SELECTED_RADIUS_UM,
            "neighbor_cap": core.SELECTED_NEIGHBOR_CAP,
            "association_threshold": core.ASSOCIATION_THRESHOLD,
            "division_threshold": None,
            "divisions": False,
        },
        "scope": {
            "fit105_access": False,
            "validation12_access": False,
            "cv57_access": True,
            "cv57_cache_access_per_identity": 1,
            "gt_access": False,
            "alternative_variant_access": False,
            "tuning": False,
            "submission": False,
        },
        "aggregate": {
            "family_counts": dict(counts),
            "video_count": len(results),
            "node_count": sum(int(row["nodes"]) for row in results),
            "candidate_edge_count": sum(int(row["candidate_edges"]) for row in results),
            "selected_edge_count": sum(int(row["selected_edges"]) for row in results),
            "fork_count": 0,
            "inference_calls": sum(int(row["inference_calls"]) for row in results),
            "peak_tokens": max(int(row["peak_tokens"]) for row in results),
        },
        "videos": results,
        "provenance": {
            "contract": str(args.contract),
            "contract_sha256": contract_sha,
            "population_manifest": str(args.population_manifest),
            "population_manifest_file_sha256": EXPECTED_POPULATION_FILE_SHA256,
            "population_manifest_content_sha256": EXPECTED_POPULATION_CONTENT_SHA256,
            "runner_sha256": contract["runtime"]["runner_sha256"],
            "inference_core_sha256": contract["runtime"]["inference_core_sha256"],
        },
    }
    atomic_json(args.output / "report.json", report)
    atomic_json(
        state_path,
        {
            "schema": "celltrack-alpha.trackastra-native-cv57-local-run-state.v1",
            "status": "COMPLETE",
            "contract_sha256": contract_sha,
            "scientific_measurement_count": 0,
            "native_inference_action_count": 1,
            "cv57_identity_access_count": 57,
            "public_submission": False,
        },
    )
    print(json.dumps({"FINAL": "COMPLETE", "aggregate": report["aggregate"]}, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument(
        "--selection-seal", type=Path, default=BRANCH / "FIT_SELECTION_SEAL.json"
    )
    value.add_argument(
        "--fit-result",
        type=Path,
        default=BRANCH / "fit105_sweep_scored/FIT105_RESULTS.json",
    )
    value.add_argument(
        "--population-manifest",
        type=Path,
        default=ROOT / "kaggle_kernel/g02-trackastra-native-cv57-v1/cv57-cache-manifest.json",
    )
    value.add_argument(
        "--output", type=Path, default=BRANCH / "cv57_local_native_v1"
    )
    value.add_argument("--preflight-only", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
