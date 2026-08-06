#!/usr/bin/env python3
"""Fail-closed validator for the frozen G-03 Phase-1 B-v2 transport proof.

This validator is deliberately external to the hash-frozen Kaggle package.  It
binds the exact package bytes to the verified A-v2 atomic tree, an exact
round-tripped carrier dataset, and the campaign failure registry.  Its
``prelaunch`` receipt is compatible with ``scripts/cmux_ops.py kernel_push``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


HANDOFF_SCHEMA = "celltrack-alpha.g03-checkpoint-handoff.v1"
GUARD_SCHEMA = "celltrack-alpha.kaggle-prelaunch-guard-result.v1"
TERMINAL_SCHEMA = "celltrack-alpha.g03-phase1-b-v2-terminal-validation.v1"
HEX64 = re.compile(r"[0-9a-f]{64}")


class ValidationError(RuntimeError):
    """An integrity or contract condition failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _safe_relative(value: Any) -> Path:
    require(isinstance(value, str) and bool(value), "handoff path is malformed")
    path = Path(value)
    require(
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"unsafe handoff path: {value!r}",
    )
    return path


def verify_handoff(root: Path) -> dict[str, Any]:
    """Verify the complete declared atomic handoff tree without importing runtime code."""
    require(root.is_dir(), f"handoff directory is absent: {root}")
    ready = load_object(root / "READY.json", "READY marker")
    inventory = load_object(root / "INVENTORY.json", "handoff inventory")
    manifest = load_object(root / "MANIFEST.json", "handoff manifest")
    require(manifest.get("schema") == HANDOFF_SCHEMA, "handoff manifest schema drifted")
    require(inventory.get("schema") == HANDOFF_SCHEMA, "handoff inventory schema drifted")
    require(ready.get("schema") == HANDOFF_SCHEMA, "handoff READY schema drifted")
    handoff_id = manifest.get("handoff_id")
    require(isinstance(handoff_id, str) and bool(handoff_id), "handoff identity is absent")
    require(
        inventory.get("handoff_id") == handoff_id
        and ready.get("handoff_id") == handoff_id,
        "handoff identity differs across atomic markers",
    )
    records = inventory.get("files")
    require(isinstance(records, list) and bool(records), "handoff inventory is empty")
    normalized: list[dict[str, Any]] = []
    for raw in records:
        require(isinstance(raw, Mapping), "handoff inventory record is malformed")
        relative = _safe_relative(raw.get("path"))
        require(relative.as_posix() not in {"READY.json", "INVENTORY.json"}, "marker declared as payload")
        expected_sha = raw.get("sha256")
        expected_bytes = raw.get("bytes")
        require(
            isinstance(expected_sha, str) and HEX64.fullmatch(expected_sha) is not None,
            f"malformed payload hash: {relative}",
        )
        require(isinstance(expected_bytes, int) and expected_bytes >= 0, f"malformed payload size: {relative}")
        path = root / relative
        require(path.is_file(), f"declared handoff member is absent: {relative}")
        require(path.stat().st_size == expected_bytes, f"handoff member size drifted: {relative}")
        actual_sha = sha256_file(path)
        require(actual_sha == expected_sha, f"handoff member hash drifted: {relative}")
        normalized.append(
            {"path": relative.as_posix(), "bytes": expected_bytes, "sha256": actual_sha}
        )
    require(normalized == records, "handoff inventory ordering or representation drifted")
    expected_tree = sha256_bytes(canonical_json_bytes(normalized))
    require(inventory.get("tree_sha256") == expected_tree, "handoff tree digest drifted")
    require(ready.get("tree_sha256") == expected_tree, "READY tree digest drifted")
    actual_names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"READY.json", "INVENTORY.json"}
    )
    require(actual_names == [row["path"] for row in normalized], "handoff has missing, extra, or reordered members")
    return {
        "handoff_id": handoff_id,
        "tree_sha256": expected_tree,
        "manifest_sha256": sha256_file(root / "MANIFEST.json"),
        "inventory_sha256": sha256_file(root / "INVENTORY.json"),
        "ready_sha256": sha256_file(root / "READY.json"),
        "files": normalized,
        "manifest": manifest,
    }


def verify_dataset_root(root: Path, expected_dataset_id: str) -> dict[str, Any]:
    require(root.is_dir(), f"dataset root is absent: {root}")
    actual_top = sorted(path.name for path in root.iterdir())
    require(
        actual_top in (["handoff"], ["dataset-metadata.json", "handoff"]),
        f"dataset root members drifted: {actual_top}",
    )
    metadata_sha: str | None = None
    if "dataset-metadata.json" in actual_top:
        metadata_path = root / "dataset-metadata.json"
        metadata = load_object(metadata_path, "dataset metadata")
        slug = expected_dataset_id.split("/", 1)[-1]
        require(metadata.get("id") == expected_dataset_id, "dataset metadata ID drifted")
        require(metadata.get("title") == slug, "dataset metadata title drifted")
        require(metadata.get("isPrivate") is True, "handoff dataset must remain private")
        metadata_sha = sha256_file(metadata_path)
    receipt = verify_handoff(root / "handoff")
    receipt["dataset_metadata_sha256"] = metadata_sha
    return receipt


def _isolated_self_test(code: Path, expected_contract_sha: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="g03-b-selftest-") as raw:
        isolated = Path(raw)
        target = isolated / code.name
        shutil.copyfile(code, target)
        process = subprocess.run(
            [sys.executable, str(target), "--self-test"],
            cwd=isolated,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    require(process.returncode == 0, f"isolated B self-test failed: {process.stderr.strip()}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    require(bool(lines), "isolated B self-test emitted no receipt")
    try:
        receipt = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValidationError("isolated B self-test receipt is not JSON") from exc
    require(isinstance(receipt, dict), "isolated B self-test receipt must be an object")
    require(receipt.get("status") == "PASS_G03_SINGLE_FILE_SELFTEST", "B self-test status failed")
    require(receipt.get("role") == "B", "B self-test role drifted")
    require(receipt.get("imports_project_modules") is False, "B package imports project siblings")
    require(receipt.get("contract_sha256") == expected_contract_sha, "embedded B contract hash drifted")
    return receipt


def _verify_a_binding(a_output: Path, a_validation_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = load_object(a_validation_path, "A terminal validation")
    require(validation.get("status") == "PASS_A_HANDOFF_READY", "A terminal validation is not PASS")
    require(validation.get("remote", {}).get("status") == "COMPLETE", "A remote status is not COMPLETE")
    require(validation.get("remote", {}).get("kernel_version") == 1, "A exact kernel version drifted")
    handoff = verify_handoff(a_output / "handoff")
    expected = validation.get("handoff", {})
    for key in ("tree_sha256", "manifest_sha256", "inventory_sha256", "ready_sha256"):
        require(handoff[key] == expected.get(key), f"A terminal {key} binding drifted")
    a_receipt_path = a_output / "A_RECEIPT.json"
    a_receipt = load_object(a_receipt_path, "A receipt")
    require(a_receipt.get("status") == "PASS_A_HANDOFF_READY", "A receipt is not PASS")
    require(a_receipt.get("selection_authorized") is False, "A receipt improperly authorizes selection")
    require(sha256_file(a_receipt_path) == validation.get("receipt", {}).get("sha256"), "A receipt hash drifted")
    require(a_receipt.get("handoff", {}).get("tree_sha256") == handoff["tree_sha256"], "A receipt tree binding drifted")
    return validation, handoff


def validate_prelaunch(args: argparse.Namespace) -> dict[str, Any]:
    package = args.package_dir.resolve()
    contract_path = args.contract.resolve()
    registry_path = args.registry.resolve()
    contract = load_object(contract_path, "transport repair contract")
    require(contract.get("status") == "LOCKED_PRE_GPU_TRANSPORT_REPAIR", "transport contract is not locked")
    b_contract = contract.get("kernels", {}).get("b")
    require(isinstance(b_contract, dict), "transport contract has no B package")
    kernel_id = b_contract.get("id")
    metadata_path = package / "kernel-metadata.json"
    metadata = load_object(metadata_path, "B kernel metadata")
    code_path = package / str(metadata.get("code_file", ""))
    require(code_path.is_file(), "declared B code file is absent")
    require(metadata.get("id") == kernel_id, "B metadata kernel ID drifted")
    require(metadata.get("title") == str(kernel_id).split("/", 1)[-1], "B metadata title drifted")
    require(metadata.get("kernel_type") == "script" and metadata.get("language") == "python", "B runtime type drifted")
    require(metadata.get("enable_gpu") is True and metadata.get("enable_internet") is False, "B GPU/network controls drifted")
    require(metadata.get("machine_shape") == "NvidiaTeslaT4", "B machine shape drifted")
    require(metadata.get("is_private") is True, "B kernel must remain private")
    require(metadata.get("dataset_sources") == b_contract.get("dataset_sources"), "B dataset sources drifted")
    require(sha256_file(code_path) == b_contract.get("code_file_sha256"), "frozen B code hash drifted")
    require(sha256_file(metadata_path) == b_contract.get("metadata_sha256"), "frozen B metadata hash drifted")
    require(registry_path.is_file(), "failure registry is absent")
    validation, a_handoff = _verify_a_binding(args.a_output.resolve(), args.a_validation.resolve())
    dataset = verify_dataset_root(args.dataset_root.resolve(), contract.get("handoff", {}).get("dataset_id"))
    for key in ("handoff_id", "tree_sha256", "manifest_sha256"):
        require(dataset[key] == a_handoff[key], f"carrier dataset {key} differs from A")
    self_test = _isolated_self_test(code_path, b_contract.get("embedded_contract_sha256"))
    return {
        "schema": GUARD_SCHEMA,
        "status": "PASS",
        "eligible": True,
        "phase": "launch",
        "failures": [],
        "kernel_id": kernel_id,
        "kernel_dir": str(package),
        "metadata_sha256": sha256_file(metadata_path),
        "code_file": str(code_path.resolve()),
        "code_sha256": sha256_file(code_path),
        "contract_sha256": sha256_file(contract_path),
        "registry_sha256": sha256_file(registry_path),
        "observed_at": int(time.time()),
        "domain": {
            "a_kernel_version": validation["remote"]["kernel_version"],
            "handoff_id": dataset["handoff_id"],
            "handoff_tree_sha256": dataset["tree_sha256"],
            "handoff_manifest_sha256": dataset["manifest_sha256"],
            "dataset_metadata_sha256": dataset["dataset_metadata_sha256"],
            "isolated_self_test": self_test,
            "expected_output_files": [
                "g03_phase1_b_output/B_RECEIPT.json",
                "g03_phase1_b_output/GRAPH_EMISSION.json",
            ],
            "selection_authorized": False,
        },
    }


def _one_recursive(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    require(len(matches) == 1, f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def validate_terminal(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_object(args.contract.resolve(), "transport repair contract")
    b_contract = contract.get("kernels", {}).get("b", {})
    validation = load_object(args.a_validation.resolve(), "A terminal validation")
    root = args.output_root.resolve()
    receipt_path = _one_recursive(root, "B_RECEIPT.json")
    graph_path = _one_recursive(root, "GRAPH_EMISSION.json")
    require(receipt_path.parent == graph_path.parent, "B terminal artifacts have different roots")
    receipt = load_object(receipt_path, "B receipt")
    graph = load_object(graph_path, "B graph emission")
    require(receipt.get("schema") == "celltrack-alpha.g03-phase1-b-receipt.v1", "B receipt schema drifted")
    require(receipt.get("status") == "PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED", "B terminal status is not PASS")
    require(receipt.get("contract_sha256") == b_contract.get("embedded_contract_sha256"), "B contract binding drifted")
    require(receipt.get("selection_authorized") is False, "B receipt improperly authorizes selection")
    expected_handoff = validation.get("handoff", {})
    require(receipt.get("handoff", {}).get("tree_sha256") == expected_handoff.get("tree_sha256"), "B handoff tree drifted")
    require(receipt.get("handoff", {}).get("manifest_sha256") == expected_handoff.get("manifest_sha256"), "B handoff manifest drifted")
    require(receipt.get("resumed_global_step") == validation.get("receipt", {}).get("global_step"), "B resume step drifted")
    require(
        isinstance(receipt.get("final_state_digest"), str)
        and HEX64.fullmatch(receipt["final_state_digest"]) is not None,
        "B final state digest is malformed",
    )
    require(sha256_file(graph_path) == receipt.get("graph_emission_sha256"), "B graph receipt hash drifted")
    swap = receipt.get("swap_guard", {})
    require(isinstance(swap.get("growth_bytes"), int), "B swap receipt is malformed")
    require(swap["growth_bytes"] < contract.get("limits", {}).get("sustained_swapout_limit_bytes", 0), "B swap hard limit was reached")
    require(graph.get("schema") == "celltrack-alpha.g03-phase1-44b6-structural-graph-delta.v1", "B graph schema drifted")
    require(graph.get("verdict") == "STRUCTURAL_NOOP_PASS", "B graph verdict drifted")
    require(graph.get("official_frozen_graph_objective_evaluated") is False, "B improperly evaluated official objective")
    require(graph.get("selection_authorized") is False, "B graph improperly authorizes selection")
    require(graph.get("checkpoint_state_digest") == receipt.get("final_state_digest"), "B graph checkpoint binding drifted")
    expected_graph = contract.get("held_fit_emission", {})
    scope = graph.get("scope", {})
    require(scope.get("family") == expected_graph.get("family"), "B graph family drifted")
    require(scope.get("identity_count") == expected_graph.get("identity_count"), "B graph identity count drifted")
    require(scope.get("manifest_sha256") == expected_graph.get("manifest_sha256"), "B fit manifest drifted")
    require(graph.get("delta") == expected_graph.get("expected_structural_delta"), "B structural delta drifted")
    records = graph.get("records")
    require(isinstance(records, list) and len(records) == expected_graph.get("identity_count"), "B graph record count drifted")
    identities: list[str] = []
    for row in records:
        require(isinstance(row, dict), "B graph record is malformed")
        identity = row.get("identity")
        require(isinstance(identity, str) and identity.startswith("train:44b6_"), "B graph identity drifted")
        require(row.get("control_edge_sha256") == row.get("candidate_edge_sha256"), "B no-op graph is not byte-identical")
        identities.append(identity)
    require(len(set(identities)) == len(identities), "B graph identities are duplicated")
    return {
        "schema": TERMINAL_SCHEMA,
        "status": receipt["status"],
        "kernel_id": b_contract.get("id"),
        "b_receipt_sha256": sha256_file(receipt_path),
        "graph_emission_sha256": sha256_file(graph_path),
        "handoff_tree_sha256": expected_handoff.get("tree_sha256"),
        "resumed_global_step": receipt.get("resumed_global_step"),
        "final_state_digest": receipt.get("final_state_digest"),
        "swap_growth_bytes": swap.get("growth_bytes"),
        "official_frozen_graph_objective_evaluated": False,
        "selection_authorized": False,
    }


def _write_or_print(value: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(output)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    handoff = sub.add_parser("handoff", help="verify one atomic handoff directory")
    handoff.add_argument("--root", type=Path, required=True)
    handoff.add_argument("--output", type=Path)
    pre = sub.add_parser("prelaunch", help="emit a cmux-compatible B-v2 launch receipt")
    pre.add_argument("--package-dir", type=Path, required=True)
    pre.add_argument("--contract", type=Path, required=True)
    pre.add_argument("--registry", type=Path, required=True)
    pre.add_argument("--a-output", type=Path, required=True)
    pre.add_argument("--a-validation", type=Path, required=True)
    pre.add_argument("--dataset-root", type=Path, required=True)
    pre.add_argument("--output", type=Path)
    terminal = sub.add_parser("terminal", help="validate downloaded B-v2 terminal artifacts")
    terminal.add_argument("--contract", type=Path, required=True)
    terminal.add_argument("--a-validation", type=Path, required=True)
    terminal.add_argument("--output-root", type=Path, required=True)
    terminal.add_argument("--output", type=Path)
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "handoff":
            result = verify_handoff(args.root.resolve())
        elif args.command == "prelaunch":
            result = validate_prelaunch(args)
        else:
            result = validate_terminal(args)
        _write_or_print(result, args.output)
        return 0
    except ValidationError as exc:
        print(f"FAIL_G03_B_VALIDATION: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
