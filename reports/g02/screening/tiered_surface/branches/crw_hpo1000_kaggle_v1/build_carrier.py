#!/usr/bin/env python3
"""Build the compact labelled CRW-HPO1000 private carrier under current LMT guards."""

from __future__ import annotations

from collections import Counter
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ALPHA = ROOT / "dag_prompts/2026-08-10-action-vocabulary-alpha"
CURRENT_LMT = ROOT / "dag_prompts/2026-08-12-lmt-resume"
for location in (ALPHA, CURRENT_LMT, ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import a1r2_crw_supplier as a1r2  # noqa: E402
import a1r3_crw_supplier as a1r3  # noqa: E402
import a1r4_lmt_runtime as lmt_runtime  # noqa: E402
import current_lmt  # noqa: E402


PREREG = HERE / "PREREGISTRATION.json"
PARAMETERS = HERE / "PARAMETER_BANK.json"
FREEZE = HERE / "CARRIER_FREEZE.json"
OUTPUT = HERE / "input_dataset"
RECEIPT = HERE / "CARRIER_BUILD_RECEIPT.json"
BASE_CONTRACT = ALPHA / "a1_supplier_contract.json"
INVENTORY = ALPHA / "runs/A1R4_COMPONENT_INVENTORY.json"
DATASET_ID = "onebitbrain/g02-crw-hpo1000-carrier-v1"


class CarrierError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(canonical(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def verify_binding(record: Mapping[str, Any], label: str) -> Path:
    path = (ROOT / str(record.get("path", ""))).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise CarrierError(f"missing {label} binding")
    if sha256(path) != str(record.get("sha256", "")):
        raise CarrierError(f"{label} binding drifted")
    return path


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prereg = json.loads(PREREG.read_text(encoding="ascii"))
    parameters = json.loads(PARAMETERS.read_text(encoding="ascii"))
    freeze = json.loads(FREEZE.read_text(encoding="ascii"))
    if prereg.get("status") != "FROZEN_BEFORE_NEW_CARRIER_SUPERVISION_MATERIALIZATION":
        raise CarrierError("preregistration status drifted")
    for label, record in prereg["bindings"].items():
        verify_binding(record, label)
    if (
        parameters.get("status") != "FROZEN_BEFORE_CARRIER_SUPERVISION"
        or parameters.get("total_unique_combinations") != 1000
        or len(parameters.get("trials", [])) != 1000
        or len({row["tuple_id"] for row in parameters["trials"]}) != 1000
    ):
        raise CarrierError("parameter bank semantic drift")
    expected = {
        "preregistration": PREREG,
        "parameter_bank": PARAMETERS,
        "builder": Path(__file__).resolve(),
        "base_supplier_contract": BASE_CONTRACT,
        "component_inventory": INVENTORY,
        "lmt": ROOT / "LMT.md",
        "compact_bank_source": ALPHA / "a1r3_crw_supplier.py",
        "conflict_source": ALPHA / "a1r2_crw_supplier.py",
        "rss_runtime": ALPHA / "a1r4_lmt_runtime.py",
        "current_lmt_runtime": CURRENT_LMT / "current_lmt.py",
    }
    if freeze.get("schema") != "celltrack-alpha.crw-hpo1000-carrier-freeze.v1":
        raise CarrierError("carrier freeze schema drift")
    if freeze.get("status") != "AUTHORIZED_ONCE_BEFORE_LABEL_ACCESS":
        raise CarrierError("carrier freeze is not authorized")
    if set(freeze.get("bindings", {})) != set(expected):
        raise CarrierError("carrier freeze binding names drifted")
    for label, path in expected.items():
        record = freeze["bindings"][label]
        if (ROOT / record["path"]).resolve() != path.resolve() or sha256(path) != record["sha256"]:
            raise CarrierError(f"carrier freeze binding drift: {label}")
    return prereg, parameters, freeze


class CompositeGuard:
    def __init__(self, monitor: a1r2.LMTResourceMonitor, workspace: current_lmt.WorkspaceGrowthGuard) -> None:
        self.active_monitor = monitor
        self.monitor = monitor
        self.workspace = workspace
        self.maximum_combined_rss_gib = current_lmt.current_lmt_values()["maximum_combined_rss_gib"]

    def check(self) -> None:
        self.monitor.check()
        self.workspace.check()
        if self.monitor.combined_rss_gib() >= self.maximum_combined_rss_gib:
            raise CarrierError("current_lmt_maximum_combined_rss_abort")


def build_components(bank: a1r3.CompactCandidateBank, identities: list[str], guard: CompositeGuard) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    component_ptr = [0]
    candidate_index: list[int] = []
    slot_mask: list[int] = []
    histogram: Counter[int] = Counter()
    max_slots = 0
    for identity in identities:
        global_indices = np.asarray(list(bank.video_indices(identity)), dtype=np.int64)
        view = bank.take(global_indices.tolist())
        components = a1r2.conflict_components(view, list(range(len(view))))
        for component in components:
            ordered = sorted(component, key=lambda local: view[local].candidate_id)
            slots = sorted({slot for local in ordered for slot in view[local].slots}, key=repr)
            if len(slots) > 32:
                raise CarrierError("component exceeds uint32 slot-mask carrier")
            bit = {slot: 1 << position for position, slot in enumerate(slots)}
            for local in ordered:
                candidate_index.append(int(global_indices[local]))
                slot_mask.append(sum(bit[slot] for slot in set(view[local].slots)))
            component_ptr.append(len(candidate_index))
            histogram[len(ordered)] += 1
            max_slots = max(max_slots, len(slots))
        guard.check()
    arrays = {
        "component_ptr": np.asarray(component_ptr, dtype=np.int64),
        "component_candidate_index": np.asarray(candidate_index, dtype=np.int32),
        "component_slot_mask": np.asarray(slot_mask, dtype=np.uint32),
    }
    if len(candidate_index) != len(bank) or len(set(candidate_index)) != len(bank):
        raise CarrierError("component coverage is not exact")
    return arrays, {
        "components": len(component_ptr) - 1,
        "component_size_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "maximum_variables": max(histogram),
        "maximum_slots": max_slots,
    }


def build() -> dict[str, Any]:
    prereg, parameters, freeze = verify_freeze()
    if OUTPUT.exists() or RECEIPT.exists():
        raise CarrierError("canonical carrier output namespace is not fresh")
    inventory = json.loads(INVENTORY.read_text(encoding="ascii"))["global_components"]
    base_contract = json.loads(BASE_CONTRACT.read_text(encoding="ascii"))
    staging = HERE / f".input_dataset.{os.getpid()}.tmp"
    if staging.exists():
        raise CarrierError("staging path already exists")
    scratch = Path(tempfile.mkdtemp(prefix="crw-hpo1000-bank-"))
    scratch.rmdir()
    telemetry = a1r2.Telemetry(cpu_limit_seconds=None, rss_limit_gib=None)
    layout = a1r3.LayoutTelemetry(stage="crw_hpo1000_carrier_not_started")
    policy = a1r2.load_lmt_policy(ROOT / "LMT.md")
    monitor = a1r2.LMTResourceMonitor(policy, telemetry)
    workspace = current_lmt.WorkspaceGrowthGuard(ROOT)
    original_rss = lmt_runtime.install()
    bank: a1r3.CompactCandidateBank | None = None
    started = time.monotonic()
    completed = False
    try:
        monitor.calibrate()
        monitor.start_background()
        workspace.start()
        guard = CompositeGuard(monitor, workspace)
        identities, baselines, bank, x, feature_sha, outer_folds, request_rows = a1r3.compact_prepare_bank(
            base_contract, guard, scratch, layout
        )
        if feature_sha != prereg["bindings"]["component_inventory"]["feature_membership_sha256"]:
            raise CarrierError("feature membership digest drifted")
        bank = a1r3.compact_attach_supervision(bank, baselines)
        if np.any(bank.attribution[:, 0] == a1r3.SENTINEL):
            raise CarrierError("supervision coverage is incomplete")
        component_arrays, component_receipt = build_components(bank, identities, guard)
        expected_histogram = {str(key): int(value) for key, value in inventory["component_size_histogram"].items()}
        if component_receipt != {
            "components": int(inventory["components"]),
            "component_size_histogram": expected_histogram,
            "maximum_variables": int(inventory["maximum_variables"]),
            "maximum_slots": int(inventory["maximum_slots"]),
        }:
            raise CarrierError("component inventory drifted")
        identity_code = np.asarray(bank.meta["identity_code"], dtype=np.uint16)
        fold = np.asarray([outer_folds[identities[int(code)]] for code in identity_code], dtype=np.uint8)
        candidate_id = np.asarray(
            [bank.candidate_id_at(index).encode("ascii") for index in range(len(bank))],
            dtype="S64",
        )
        arrays = {
            "features": np.asarray(x, dtype=np.float64),
            "positive_target": np.asarray(bank.attribution[:, 0] > 0, dtype=np.uint8),
            "attribution": np.asarray(bank.attribution, dtype=np.int32),
            "identity_code": identity_code,
            "fold": fold,
            "candidate_id": candidate_id,
            **component_arrays,
        }
        if (
            arrays["features"].shape != (92905, 18)
            or len(arrays["positive_target"]) != 92905
            or len(set(arrays["fold"].tolist())) != 5
            or int(request_rows) != 45961
        ):
            raise CarrierError("carrier population shape drifted")
        staging.mkdir(parents=False, exist_ok=False)
        carrier_path = staging / "crw_hpo1000_carrier.npz"
        np.savez_compressed(carrier_path, **arrays)
        shutil.copy2(PARAMETERS, staging / PARAMETERS.name)
        shutil.copy2(PREREG, staging / PREREG.name)
        array_manifest = {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
            for name, value in arrays.items()
        }
        carrier_manifest = {
            "schema": "celltrack-alpha.crw-hpo1000-private-carrier.v1",
            "status": "IMMUTABLE_UPLOAD_READY",
            "dataset_id": DATASET_ID,
            "dataset_version": 1,
            "preregistration_sha256": sha256(PREREG),
            "parameter_bank_sha256": sha256(PARAMETERS),
            "carrier_npz": {
                "path": carrier_path.name,
                "bytes": carrier_path.stat().st_size,
                "sha256": sha256(carrier_path),
            },
            "arrays": array_manifest,
            "population": {
                "videos": len(identities),
                "requests": int(request_rows),
                "candidates": len(bank),
                "features": arrays["features"].shape[1],
                "positive_targets": int(arrays["positive_target"].sum()),
                "folds": 5,
                **component_receipt,
            },
            "identities": identities,
            "feature_membership_sha256": feature_sha,
            "forbidden_payload_absent": {
                "raw_images": True,
                "raw_truth_graphs": True,
                "official_scorer": True,
                "cv57_or_6bba": True,
                "submission_artifact": True,
                "credentials": True,
            },
        }
        manifest_path = staging / "CARRIER_MANIFEST.json"
        manifest_path.write_bytes(canonical(carrier_manifest))
        dataset_metadata = {
            "id": DATASET_ID,
            "isPrivate": True,
            "licenses": [{"name": "other"}],
            "title": "G02 CRW HPO1000 Private Carrier v1",
        }
        (staging / "dataset-metadata.json").write_bytes(canonical(dataset_metadata))
        guard.check()
        os.replace(staging, OUTPUT)
        completed = True
        return {
            "schema": "celltrack-alpha.crw-hpo1000-carrier-build-receipt.v1",
            "status": "COMPLETE",
            "preregistration_sha256": sha256(PREREG),
            "parameter_bank_sha256": sha256(PARAMETERS),
            "carrier_freeze_sha256": sha256(FREEZE),
            "carrier_manifest_sha256": sha256(OUTPUT / "CARRIER_MANIFEST.json"),
            "carrier_npz_sha256": sha256(OUTPUT / "crw_hpo1000_carrier.npz"),
            "dataset_metadata_sha256": sha256(OUTPUT / "dataset-metadata.json"),
            "elapsed_seconds": time.monotonic() - started,
            "feature_membership_sha256": feature_sha,
            "parameter_count": int(parameters["total_unique_combinations"]),
            "component_inventory": component_receipt,
            "layout": layout.snapshot(),
            "resources": {
                "rss": monitor.snapshot(),
                "workspace": workspace.snapshot(),
            },
            "scientific_boundaries": {
                "official_scorer_accessed": False,
                "cv57_accessed": False,
                "six_bba_accessed": False,
                "external_action": False,
            },
        }
    finally:
        monitor.stop()
        workspace.stop()
        if bank is not None:
            bank.close()
        gc.collect()
        if scratch.exists():
            shutil.rmtree(scratch)
        if not completed and staging.exists():
            shutil.rmtree(staging)
        lmt_runtime.restore(original_rss)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required because carrier construction opens held-fit supervision")
    receipt = build()
    if RECEIPT.exists():
        raise CarrierError("refusing to overwrite immutable receipt")
    RECEIPT.write_bytes(canonical(receipt))
    print(json.dumps({"status": receipt["status"], "receipt": str(RECEIPT.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
