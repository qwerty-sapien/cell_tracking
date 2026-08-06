from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "kaggle_kernel/g04-computed-segmentation-runtime-cp312-v1"
KERNEL = ROOT / "kaggle_kernel/g04-computed-segmentation-timing-v1"
COMPETITION_ARCHIVE = (
    ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
)
os.environ["G04_LOCAL_PACKAGE_TEST"] = "1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_manifest_rebuild_and_exact_files() -> None:
    builder = load(
        "g04_runtime_manifest_test",
        ROOT / "scripts/g04_build_segmentation_runtime_manifest.py",
    )
    expected = builder.build(CARRIER)
    observed = json.loads((CARRIER / "RUNTIME_CARRIER_MANIFEST.json").read_text())
    assert observed == expected
    assert observed["status"] == "READY_AUTHENTICATED_OFFLINE_CP312"
    assert observed["file_count"] == 29
    assert observed["install_contract"]["wheel_count"] == 24
    assert observed["base_runtime"]["abi"] == "cp312"
    assert observed["base_runtime"]["numpy"] == "2.0.2"
    assert observed["base_runtime"]["scipy"] == "1.16.3"
    assert observed["base_runtime"]["torch_gpu_required"] == "2.10.0+cu128"
    dataset_metadata = json.loads((CARRIER / "dataset-metadata.json").read_text())
    assert len(dataset_metadata["title"]) <= 50
    assert dataset_metadata["title"] == dataset_metadata["id"].split("/", 1)[1]


def test_runtime_preserves_kaggle_numeric_and_torch_stack() -> None:
    manifest = json.loads((CARRIER / "RUNTIME_CARRIER_MANIFEST.json").read_text())
    wheels = [row for row in manifest["files"] if row["kind"] == "wheel"]
    distributions = {str(row["distribution"]).lower() for row in wheels}
    assert not {"numpy", "scipy", "torch", "torchvision"} & distributions
    cellpose = next(row for row in wheels if row["distribution"] == "cellpose")
    assert cellpose["version"] == "3.1.1.2"
    assert "numpy<2.1,>=1.20.0" in cellpose["requirements"]
    assert manifest["cellpose"]["base_numpy_satisfies_declared_constraint"] is True
    assert "not represented as a separate weight license" in manifest["cellpose"][
        "weight_terms_disclosure"
    ]


def test_timing_step0_module_and_exact_raw_byte_validators() -> None:
    timing_path = KERNEL / "g04_computed_segmentation_timing_v1.py"
    timing = load("g04_timing_package_test", timing_path)
    manifest, wheels = timing.validate_runtime_carrier(CARRIER)
    assert manifest["file_count"] == 29
    assert timing.RUNTIME_REMOTE_FILE_COUNT == 30
    assert len(wheels) == 24
    assert timing.REQUEST_COUNT == 1
    assert timing.REQUEST_FILE_COUNT == 3
    assert timing.POPULATION_FILE_COUNT == 3
    with tempfile.TemporaryDirectory() as temporary:
        train_root = Path(temporary) / "train"
        with zipfile.ZipFile(COMPETITION_ARCHIVE) as archive:
            for row in timing.COMPETITION_INPUT_FILES:
                member = f"train/{row['path']}"
                archive.extract(member, Path(temporary))
        contract = timing.validate_competition_input(train_root)
    assert contract == timing.competition_input_contract()
    assert len(contract["files"]) == 3
    assert (
        hashlib.sha256(
            json.dumps(
                contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        ).hexdigest()
        == timing.COMPETITION_INPUT_CONTRACT_SHA256
    )


def test_step0_top_level_imports_are_standard_library_only() -> None:
    timing_path = KERNEL / "g04_computed_segmentation_timing_v1.py"
    tree = ast.parse(timing_path.read_text())
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imports.add(str(node.module).split(".", 1)[0])
    assert imports <= {
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "signal",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "traceback",
    }
    source = timing_path.read_text()
    assert source.index("def run_step0") < source.index("def install_runtime")
    assert "third_party_imports\": False" in source
    assert "raw_zarr_opened\": False" in source


def test_timing_prelaunch_contract_is_byte_consistent() -> None:
    prelaunch = json.loads((KERNEL / "prelaunch-contract.json").read_text())
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text())
    assert prelaunch["code_sha256"] == sha256(
        KERNEL / "g04_computed_segmentation_timing_v1.py"
    )
    assert prelaunch["kernel_metadata_sha256"] == sha256(
        KERNEL / "kernel-metadata.json"
    )
    assert prelaunch["runtime_dataset"]["manifest_sha256"] == sha256(
        CARRIER / "RUNTIME_CARRIER_MANIFEST.json"
    )
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert metadata["dataset_sources"] == [
        "onebitbrain/g04-computed-segmentation-runtime-cp312-v1"
    ]
    assert prelaunch["timing_gate"]["identity"] == "train:44b6_24264f12"
    assert prelaunch["timing_gate"]["timepoint"] == 0
    assert prelaunch["hard_limits"]["repeat_count"] == 2
    assert prelaunch["hard_limits"]["process_hard_kill_seconds"] == 3600
    assert prelaunch["negative_scope"]["production_manifest_eligible"] is False
    assert prelaunch["negative_scope"]["submission"] is False
    assert prelaunch["negative_scope"]["detection_access"] is False
    assert prelaunch["negative_scope"]["gt_node_access"] is False
    assert prelaunch["negative_scope"]["gt_edge_access"] is False
    assert prelaunch["request_file_count"] == 3
    assert prelaunch["population_file_count"] == 3
    assert prelaunch["population_manifest_sha256"] == prelaunch[
        "competition_input"
    ]["contract_sha256"]
    assert prelaunch["runtime_dataset"][
        "mounted_remote_file_count_including_manifest"
    ] == 30
    for field in (
        "exclusive_dataset_source",
        "lane_owner",
        "expected_mount_root",
        "input_manifest_sha256",
        "checkpoint_sha256",
        "population_manifest_sha256",
        "request_count",
        "request_file_count",
        "population_file_count",
        "wheel_count",
        "remote_preflight_mode",
    ):
        assert field in prelaunch


def test_timing_package_manifest_authenticates_every_launch_file() -> None:
    package = json.loads((KERNEL / "PACKAGE_MANIFEST.json").read_text())
    assert package["status"] == "READY_PRELAUNCH_NO_EXTERNAL_MUTATION"
    assert package["package_file_count_excluding_manifest"] == 3
    assert {row["path"] for row in package["files"]} == {
        "g04_computed_segmentation_timing_v1.py",
        "kernel-metadata.json",
        "prelaunch-contract.json",
    }
    for row in package["files"]:
        path = KERNEL / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]


def test_timing_source_has_deadline_and_atomic_receipt_guards() -> None:
    source = (KERNEL / "g04_computed_segmentation_timing_v1.py").read_text()
    required = (
        "PROCESS_HARD_KILL_SECONDS = 3_600",
        "STEP0 = run_step0()",
        "start_new_session=True",
        "os.killpg(process.pid, signal.SIGKILL)",
        "PASS_CUDA_TIMING_GATE",
        "PASS_ATOMIC_TIMING_RECEIPT",
        "production_manifest_eligible\": False",
        "ceil(max(repeat_seconds) * 1.25)",
        "repeated CUDA labels or per-mask regionprops payloads are not byte-stable",
        "computed mask count and regionprops count disagree",
        "competition_input_contract_sha256",
        "gt_node_access\": False",
        "gt_edge_access\": False",
    )
    for fragment in required:
        assert fragment in source
    assert ".geff" not in source.casefold()
    assert "load_bh_detections" not in source
    assert "cv57-cache" not in source.casefold()


def test_real_shared_guard_and_cmux_predicate_accept_overlay() -> None:
    guard = load(
        "g04_real_kaggle_prelaunch_guard_test",
        ROOT
        / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
        "kaggle_prelaunch_guard.py",
    )
    cmux = load("g04_real_cmux_predicate_test", ROOT / "scripts/cmux_ops.py")
    contract = json.loads((KERNEL / "prelaunch-contract.json").read_text())
    registry = json.loads(
        (
            ROOT
            / "reports/g04/lanes/computed_segmentation/"
            "TIMING_TRANSPORT_REGISTRY.json"
        ).read_text()
    )
    static = guard.check_kernel(KERNEL, contract, registry, "static")
    inline = guard.check_kernel(KERNEL, contract, registry, "inline")
    assert static["status"] == "PASS"
    assert static["failures"] == []
    assert inline["status"] == "PASS"
    assert inline["eligible"] is True
    assert inline["failures"] == []
    with tempfile.TemporaryDirectory() as temporary:
        receipt = Path(temporary) / "inline.json"
        receipt.write_text(json.dumps(inline), encoding="utf-8")
        bound = cmux._validate_kernel_push_predicate(
            receipt,
            kernel_id="onebitbrain/g04-computed-segmentation-timing-v1",
            kernel_dir=KERNEL,
        )
    assert bound["code_sha256"] == inline["code_sha256"]
    assert bound["metadata_sha256"] == inline["metadata_sha256"]
