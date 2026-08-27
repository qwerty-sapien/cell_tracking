from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time

import pytest


HERE = Path(__file__).resolve().parent
KERNEL = HERE / "kaggle_cpu_kernel" / "kaggle_selector_cpu.py"
METADATA = HERE / "kaggle_cpu_kernel" / "kernel-metadata.json"
BUILDER = HERE / "build_kaggle_cpu_bundle.py"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("selector_bundle_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_kernel_step0() -> tuple[dict[str, object], float]:
    source = KERNEL.read_text(encoding="utf-8")
    prefix, separator, _ = source.partition("\nSTARTED = time.time()")
    assert separator
    namespace: dict[str, object] = {"__file__": str(KERNEL)}
    started = time.perf_counter()
    exec(compile(prefix, str(KERNEL), "exec"), namespace)
    elapsed = time.perf_counter() - started
    return namespace, elapsed


def expanded_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    input_root = tmp_path / "input"
    dataset = (
        input_root
        / "datasets"
        / "onebitbrain"
        / "g02-selector-requirement-cpu-inputs-v1"
    )
    bundle = dataset / "selector_cpu_bundle"
    bundle.mkdir(parents=True)

    plain_relative = (
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/"
        "generated/action_population.jsonl"
    )
    population_relative = (
        "reports/g02/screening/tiered_surface/branches/sa_repr/"
        "phase2_separation_gate/run_v1/population_manifest.json"
    )
    payloads = {
        plain_relative: b'{"action":"delete"}\n',
        population_relative: b'{"status":"fixture"}\n',
        "external/Tracking_Support/wheels/fixture.whl": b"wheel",
    }
    rows = []
    for relative, payload in payloads.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        row: dict[str, object] = {
            "path": relative,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "transport": "byte_exact_regular_member",
        }
        if relative == plain_relative:
            row["transport"] = "gzip_payload_as_plain_member"
            row["source_path"] = f"{plain_relative}.gz"
        rows.append(row)

    inventory_bytes = (
        json.dumps(
            {
                "schema": "celltrack-alpha.selector-cpu-bundle-inventory.v2",
                "verification": "member_sha256_only",
                "files": rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    inventory = bundle / "selector_cpu_bundle_inventory.json"
    inventory.write_bytes(inventory_bytes)
    pins = {
        "inventory": sha256_bytes(inventory_bytes),
        "population": sha256_bytes(payloads[population_relative]),
        "plain": sha256_bytes(payloads[plain_relative]),
    }
    manifest_bytes = (
        json.dumps(
            {
                "schema": "celltrack-alpha.selector-kaggle-cpu-bundle.v2",
                "status": "FROZEN_BEFORE_KAGGLE_CPU_OUTCOMES",
                "archive": {
                    "path": "selector_cpu_bundle.tar",
                    "bytes": 1,
                    "format": "uncompressed_tar",
                    "compression": "none",
                    "verification": "member_sha256_only",
                    "member_count": len(rows) + 1,
                },
                "inventory_sha256": pins["inventory"],
                "population_manifest_sha256": pins["population"],
                "population_action_count": 1,
                "wheel_count": 1,
                "required_train_video_ids": ["train:vid1"],
                "competition_source": "biohub-cell-tracking-during-development",
                "enable_gpu": False,
                "ram_gib": 16,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    (dataset / "selector_cpu_bundle_manifest.json").write_bytes(manifest_bytes)
    pins["manifest"] = sha256_bytes(manifest_bytes)
    (
        input_root
        / "competitions"
        / "biohub-cell-tracking-during-development"
        / "train"
        / "vid1.zarr"
    ).mkdir(parents=True)
    return input_root, pins, bundle


def test_declared_code_file_is_self_contained_and_resolves_under_five_seconds():
    metadata = json.loads(METADATA.read_text())
    assert metadata["code_file"] == KERNEL.name
    tree = ast.parse(KERNEL.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "kaggle_selector_step0" not in imported
    assert not (KERNEL.parent / "kaggle_selector_step0.py").exists()

    namespace, elapsed = load_kernel_step0()
    assert callable(namespace["run_step0"])
    assert namespace["STEP0_SYMBOL_RECEIPT"]["resolved"] is True
    assert namespace["STEP0_SYMBOL_RECEIPT"]["elapsed_seconds"] < 5.0
    assert elapsed < 5.0


def test_uncompressed_tar_is_deterministic_and_member_addressed(tmp_path: Path):
    builder = load_builder()
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    inventory = b'{"verification":"member_sha256_only"}\n'
    archive_a = tmp_path / "a.tar"
    archive_b = tmp_path / "b.tar"
    entries = [
        (f"nested/{'long-segment-' * 24}first.bin", first),
        ("second.bin", second),
    ]
    builder.write_uncompressed_tar(archive_a, entries, inventory)
    builder.write_uncompressed_tar(archive_b, entries, inventory)

    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert not archive_a.read_bytes().startswith(b"\x1f\x8b")
    with tarfile.open(archive_a, mode="r:") as handle:
        members = handle.getmembers()
        assert [member.name for member in members] == [
            f"nested/{'long-segment-' * 24}first.bin",
            "second.bin",
            "selector_cpu_bundle_inventory.json",
        ]
        assert all(member.mtime == 0 and member.isfile() for member in members)
        assert handle.extractfile(members[0]).read() == b"alpha"


def test_expanded_directory_is_verified_by_member_sha256(tmp_path: Path):
    namespace, _ = load_kernel_step0()
    input_root, pins, _ = expanded_fixture(tmp_path)
    receipt = namespace["run_step0"](
        input_root,
        expected_manifest_sha256=pins["manifest"],
        expected_inventory_sha256=pins["inventory"],
        expected_population_sha256=pins["population"],
        expected_plain_population_sha256=pins["plain"],
    )
    assert receipt["status"] == "PASS"
    assert receipt["archive_format"] == "uncompressed_tar"
    assert receipt["container_roundtrip_verified"] is False
    assert receipt["member_verification"] == "sha256"
    assert receipt["step0_symbol_elapsed_seconds"] < 5.0


def test_genuine_member_hash_mismatch_fails_closed(tmp_path: Path):
    namespace, _ = load_kernel_step0()
    input_root, pins, bundle = expanded_fixture(tmp_path)
    wheel = bundle / "external/Tracking_Support/wheels/fixture.whl"
    wheel.write_bytes(b"WHEEL")
    with pytest.raises(
        namespace["Step0Error"], match="member SHA-256 mismatch"
    ):
        namespace["run_step0"](
            input_root,
            expected_manifest_sha256=pins["manifest"],
            expected_inventory_sha256=pins["inventory"],
            expected_population_sha256=pins["population"],
            expected_plain_population_sha256=pins["plain"],
        )
