#!/usr/bin/env python3
"""Build and validate the source-forward V32 submission repair bisection.

The bisection starts with the source-byte-exact public notebook and adds only
the declared V32 transport deltas in a fixed, cumulative order.  It does not
score locally, alter model weights, or submit to Kaggle; CMUX remains the sole
external mutation path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import public_temporalunet_baseline as baseline  # noqa: E402


STAGES = (
    "pristine_control",
    "step0_no_input_hash",
    "telemetry_jsonl_no_input_hash",
    "patch_exact_mount_discovery",
    "patch_secondary_weight_boundary",
    "patch_secondary_weight_message",
    "patch_skip_zero_weight_secondary_inference",
    "input_hashing_last",
)

STAGE_LABELS = {
    "pristine_control": "byte-exact public V32 notebook; no local source delta",
    "step0_no_input_hash": "Step-0 only; no telemetry and no recursive input-content hashing",
    "telemetry_jsonl_no_input_hash": "Step-0 plus telemetry/JSONL; paths and sizes only",
    "patch_exact_mount_discovery": "add replace_once #1: exact support-mount discovery",
    "patch_secondary_weight_boundary": "add replace_once #2: allow zero secondary edge weight",
    "patch_secondary_weight_message": "add replace_once #3: matching validation message",
    "patch_skip_zero_weight_secondary_inference": "add replace_once #4: skip zero-weight secondary inference",
    "input_hashing_last": "turn full recursive input content SHA-256 back on",
}


class BisectionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise BisectionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_index(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError as exc:
        raise BisectionError(f"unknown bisection stage {stage!r}") from exc


def stage_settings(stage: str) -> dict[str, Any]:
    index = stage_index(stage)
    return {
        "stage": stage,
        "stage_index": index,
        "source_byte_exact": stage == "pristine_control",
        "step0_only": stage == "step0_no_input_hash",
        "telemetry_jsonl": index >= stage_index("telemetry_jsonl_no_input_hash"),
        "replacement_count": max(0, min(4, index - stage_index("telemetry_jsonl_no_input_hash"))),
        "hash_input_contents": stage == "input_hashing_last",
        "label": STAGE_LABELS[stage],
    }


def raw_notebook_cells() -> list[str]:
    notebook = baseline.load_object(baseline.PUBLIC_NOTEBOOK, "pinned public V32 notebook")
    return [cell.replace("from __future__ import annotations\n\n", "") for cell in baseline.source_cells(notebook)]


def step0_without_telemetry_source(*, expected_test_count: int, local_gate: Mapping[str, Any]) -> str:
    """A standalone preflight with no JSONL layer and no recursive input hash."""

    binding = baseline.artifact_binding()
    return f'''\
import hashlib
import json
import os
from pathlib import Path
import time

EXPECTED_TEST_COUNT = {expected_test_count}
PRIMARY_ROOT = Path({baseline.PRIMARY_ROOT!r})
SECONDARY_ROOT = Path({baseline.SECONDARY_ROOT!r})
COMPETITION_TEST = Path({baseline.COMPETITION_TEST!r})
PRIMARY_MANIFEST_SHA256 = {baseline.PRIMARY_MANIFEST_SHA256!r}
PRIMARY_WEIGHT_SHA256 = {baseline.PRIMARY_WEIGHT_SHA256!r}
SECONDARY_MANIFEST_SHA256 = {baseline.SECONDARY_MANIFEST_SHA256!r}
SECONDARY_WEIGHT_SHA256 = {baseline.SECONDARY_WEIGHT_SHA256!r}
POPULATION_MANIFEST_SHA256 = {str(local_gate['gate_sha256'])!r}
INPUT_MANIFEST_SHA256 = {binding['binding_sha256']!r}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    temporary.replace(path)


def run_step0():
    started = time.time()
    exact_files = {{
        PRIMARY_ROOT / "ARTIFACT_MANIFEST.json": PRIMARY_MANIFEST_SHA256,
        PRIMARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": PRIMARY_WEIGHT_SHA256,
        SECONDARY_ROOT / "ARTIFACT_MANIFEST.json": SECONDARY_MANIFEST_SHA256,
        SECONDARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": SECONDARY_WEIGHT_SHA256,
    }}
    observed = {{}}
    for path, expected in exact_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"STEP0 missing exact public artifact: {{path}}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"STEP0 hash mismatch for {{path}}: {{actual}} != {{expected}}")
        observed[str(path)] = actual
    if not COMPETITION_TEST.is_dir():
        raise FileNotFoundError(f"STEP0 hidden test root is unavailable: {{COMPETITION_TEST}}")
    selected = sorted(
        path.name[:-5] for path in COMPETITION_TEST.iterdir()
        if path.is_dir() and path.name.endswith(".zarr")
    )
    if len(selected) != EXPECTED_TEST_COUNT or len(set(selected)) != EXPECTED_TEST_COUNT:
        raise RuntimeError(f"STEP0 hidden inventory drift: {{len(selected)}} != {{EXPECTED_TEST_COUNT}}")
    for dataset in selected:
        source = COMPETITION_TEST / f"{{dataset}}.zarr"
        if not (source / "0").exists():
            raise FileNotFoundError(f"STEP0 malformed hidden dataset: {{source}}")
    globals()["SELECTED_DATASETS"] = selected
    _atomic_json(Path("/kaggle/working/PUBLIC_TEMPORALUNET_STEP0.json"), {{
        "schema": "celltrack-alpha.public-temporalunet-hidden-step0.v1",
        "status": "PASS",
        "variant": "public_v32",
        "request_count": len(selected),
        "request_file_count": len(selected),
        "population_file_count": len(selected),
        "wheel_count": 60,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "global_population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "checkpoint_sha256": PRIMARY_WEIGHT_SHA256,
        "secondary_checkpoint_sha256": SECONDARY_WEIGHT_SHA256,
        "dataset_source": {baseline.SECONDARY_SOURCE!r},
        "dataset_version": 1,
        "dataset_status": "ready",
        "enable_gpu": True,
        "heavy_work_started": False,
        "competition_path_visits": len(selected),
        "artifacts": observed,
        "selected_datasets": selected,
        "hidden_test_compatible": True,
        "dynamic_test_discovery": True,
        "public_identity_pinned": False,
        "public_output_copied": False,
        "recursive_input_content_hashing": False,
        "elapsed_seconds": time.time() - started,
    }})
    return selected


STEP0 = run_step0()
SELECTED_DATASETS = STEP0
os.environ["BIOHUB_TEST_DIR"] = str(COMPETITION_TEST)
'''


def build_step0_only_script(*, expected_test_count: int, local_gate: Mapping[str, Any]) -> str:
    cells = raw_notebook_cells()
    cells[3] = cells[3] + "\n" + baseline.variant_override("public_v32")
    pieces = [step0_without_telemetry_source(expected_test_count=expected_test_count, local_gate=local_gate)]
    for index, cell in enumerate(cells):
        pieces.append(f"\n# ---- PUBLIC V32 CODE CELL {index} ----\n{cell}\n")
    script = "".join(pieces)
    compile(script, "v32_step0_no_input_hash.py", "exec")
    return script


def metadata(kernel_id: str, code_file: str, *, notebook: bool) -> dict[str, Any]:
    return {
        "id": kernel_id,
        "title": kernel_id.split("/", 1)[1],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook" if notebook else "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [baseline.COMPETITION],
        "dataset_sources": [baseline.PRIMARY_SOURCE, baseline.SECONDARY_SOURCE],
        "model_sources": [],
    }


def regular_logging_contract() -> dict[str, Any]:
    return {
        "schema": "celltrack-alpha.run-log.v1",
        "stdout_jsonl": True,
        "persisted_with_artifacts": True,
        "download_on_success": True,
        "inventory_before_resolution": True,
        "failure_traceback_with_state": True,
    }


def build_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = stage_settings(args.stage)
    if args.kernel_dir.exists():
        raise BisectionError(f"refusing to overwrite package directory {args.kernel_dir}")
    if not args.kernel_id.startswith("onebitbrain/"):
        raise BisectionError("kernel ID must be owner-qualified")
    args.kernel_dir.mkdir(parents=True)
    if settings["source_byte_exact"]:
        code_path = args.kernel_dir / "public_v32_pristine.ipynb"
        shutil.copyfile(baseline.PUBLIC_NOTEBOOK, code_path)
        package_metadata = metadata(args.kernel_id, code_path.name, notebook=True)
        contract = {
            "schema": "celltrack-alpha.v32-submission-bisect-prelaunch.v1",
            "execution_profile": "exact_pristine_control",
            "kernel_id": args.kernel_id,
            "lane_owner": "PUBLIC_TEMPORALUNET_BASELINE_ADOPTION",
            "dataset_sources": package_metadata["dataset_sources"],
            "exclusive_dataset_source": baseline.SECONDARY_SOURCE,
            "expected_mount_root": baseline.SECONDARY_ROOT,
            "kernel_properties": {
                key: package_metadata[key]
                for key in ("enable_gpu", "enable_internet", "enable_tpu", "machine_shape")
            },
            "pristine_notebook_sha256": baseline.PUBLIC_NOTEBOOK_SHA256,
            "code_sha256": sha256(code_path),
            "bisection": settings,
        }
        local_gate: Mapping[str, Any] = {}
    else:
        local_gate = baseline.validate_hidden_submission_gate(
            score_path=args.score_receipt,
            taxonomy_path=args.taxonomy_receipt,
            terminal_auth_path=args.terminal_authentication,
        )
        if settings["step0_only"]:
            script = build_step0_only_script(
                expected_test_count=args.expected_test_count, local_gate=local_gate
            )
        else:
            script = baseline.build_hidden_script(
                expected_test_count=args.expected_test_count,
                local_gate=local_gate,
                replacement_count=int(settings["replacement_count"]),
                hash_input_contents=bool(settings["hash_input_contents"]),
            )
        code_path = args.kernel_dir / "public_temporalunet_hidden_kernel.py"
        code_path.write_text(script, encoding="utf-8")
        package_metadata = metadata(args.kernel_id, code_path.name, notebook=False)
        contract = {
            "schema": "celltrack-alpha.v32-submission-bisect-prelaunch.v1",
            "execution_profile": (
                "bisection_step0_only"
                if settings["step0_only"]
                else (
                    "bisection_telemetry_no_input_hash"
                    if settings["stage"] == "telemetry_jsonl_no_input_hash"
                    else "standard"
                )
            ),
            "kernel_id": args.kernel_id,
            "lane_owner": "PUBLIC_TEMPORALUNET_BASELINE_ADOPTION",
            "dataset_sources": package_metadata["dataset_sources"],
            "dataset_version": 1,
            "exclusive_dataset_source": baseline.SECONDARY_SOURCE,
            "expected_mount_root": baseline.SECONDARY_ROOT,
            "kernel_properties": {
                key: package_metadata[key]
                for key in ("enable_gpu", "enable_internet", "enable_tpu", "machine_shape")
            },
            "code_sha256": sha256(code_path),
            "input_manifest_sha256": baseline.artifact_binding()["binding_sha256"],
            "checkpoint_sha256": baseline.PRIMARY_WEIGHT_SHA256,
            "secondary_checkpoint_sha256": baseline.SECONDARY_WEIGHT_SHA256,
            "population_manifest_sha256": local_gate["gate_sha256"],
            "population_file_count": args.expected_test_count,
            "request_count": args.expected_test_count,
            "request_file_count": args.expected_test_count,
            "wheel_count": 60,
            "hidden_test_compatible": True,
            "dynamic_test_discovery": True,
            "public_identity_pinned": False,
            "public_output_copied": False,
            "remote_preflight_mode": "inline_step0_before_heavy_work",
            "remote_preflight_kernel_id": args.kernel_id,
            "step0_competition_path_visits": args.expected_test_count,
            "step0_elapsed_is_hard_failure": False,
            "step0_limit_seconds": 900.0,
            "run_logging": (
                {"mode": "intentionally_absent_for_first_bisection_stage"}
                if settings["step0_only"]
                else regular_logging_contract()
            ),
            "runtime_governance": {
                "per_run_wall_clock_kill": False,
                "projection_gate_h": 8.0,
                "weekly_shared_ledger_cap_h": 30.0,
            },
            "local_submission_gate": local_gate,
            "allow_pristine_broad_discovery": bool(settings["replacement_count"] == 0),
            "bisection": settings,
        }
    metadata_path = args.kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(package_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract_path = args.kernel_dir / "prelaunch-contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "celltrack-alpha.v32-submission-bisect-package.v1",
        "kernel_id": args.kernel_id,
        "expected_initial_kernel_version": 1,
        "bisection": settings,
        "public_notebook_sha256": baseline.PUBLIC_NOTEBOOK_SHA256,
        "local_submission_gate": local_gate,
        "files": {
            code_path.name: sha256(code_path),
            metadata_path.name: sha256(metadata_path),
            contract_path.name: sha256(contract_path),
        },
    }
    manifest["package_sha256"] = canonical_sha256(manifest)
    atomic_json(args.kernel_dir / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def csv_structure(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    node_rows = 0
    edge_rows = 0
    datasets: list[str] = []
    dataset_transitions = 0
    last_dataset: str | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != baseline.CSV_COLUMNS:
            raise BisectionError(
                f"{path}: header mismatch: observed={reader.fieldnames!r} expected={baseline.CSV_COLUMNS!r}"
            )
        for expected_id, row in enumerate(reader):
            if set(row) != set(baseline.CSV_COLUMNS):
                raise BisectionError(f"{path}: malformed row keys at row {expected_id}")
            try:
                values = {key: int(str(row[key])) for key in baseline.CSV_COLUMNS if key not in {"dataset", "row_type"}}
            except ValueError as exc:
                raise BisectionError(f"{path}: non-integer numeric field at row {expected_id}") from exc
            if values["id"] != expected_id:
                raise BisectionError(f"{path}: IDs are not zero-based contiguous at row {expected_id}")
            dataset = str(row["dataset"])
            if not dataset:
                raise BisectionError(f"{path}: blank dataset at row {expected_id}")
            row_type = str(row["row_type"])
            if row_type not in {"node", "edge"}:
                raise BisectionError(f"{path}: bad row_type {row_type!r} at row {expected_id}")
            if row_type == "node":
                node_rows += 1
                if values["source_id"] != -1 or values["target_id"] != -1:
                    raise BisectionError(f"{path}: node row has edge IDs at row {expected_id}")
            else:
                edge_rows += 1
                if values["source_id"] < 0 or values["target_id"] < 0:
                    raise BisectionError(f"{path}: edge row has negative endpoint at row {expected_id}")
            if dataset != last_dataset:
                datasets.append(dataset)
                dataset_transitions += 1
                last_dataset = dataset
            digest.update(("\x1f".join(str(row[key]) for key in baseline.CSV_COLUMNS) + "\n").encode())
            rows += 1
    return {
        "path": str(path),
        "sha256": sha256(path),
        "columns": baseline.CSV_COLUMNS,
        "rows": rows,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "dataset_run_order": datasets,
        "dataset_transitions": dataset_transitions,
        "ordered_row_sha256": digest.hexdigest(),
    }


def compare_csv(args: argparse.Namespace) -> dict[str, Any]:
    left = csv_structure(args.pristine_csv)
    right = csv_structure(args.candidate_csv)
    compared = {
        "columns_identical": left["columns"] == right["columns"],
        "row_count_identical": left["rows"] == right["rows"],
        "dataset_run_order_identical": left["dataset_run_order"] == right["dataset_run_order"],
        "node_count_identical": left["node_rows"] == right["node_rows"],
        "edge_count_identical": left["edge_rows"] == right["edge_rows"],
        "full_ordered_content_identical": left["ordered_row_sha256"] == right["ordered_row_sha256"],
    }
    result = {
        "schema": "celltrack-alpha.v32-submission-bisect-csv-comparison.v1",
        "status": "PASS" if all(compared.values()) else "DIFF",
        "pristine": left,
        "candidate": right,
        "comparison": compared,
    }
    atomic_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-stage")
    build.add_argument("--stage", choices=STAGES, required=True)
    build.add_argument("--kernel-id", required=True)
    build.add_argument("--kernel-dir", type=Path, required=True)
    build.add_argument("--expected-test-count", type=int, default=4)
    build.add_argument("--score-receipt", type=Path, required=True)
    build.add_argument("--taxonomy-receipt", type=Path, required=True)
    build.add_argument("--terminal-authentication", type=Path, required=True)
    build.set_defaults(func=build_stage)
    compare = sub.add_parser("compare-csv")
    compare.add_argument("--pristine-csv", type=Path, required=True)
    compare.add_argument("--candidate-csv", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=compare_csv)
    return value


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.func(args)
    except (BisectionError, baseline.BaselineError, OSError, ValueError) as exc:
        print(f"v32_submission_bisect: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
