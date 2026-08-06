from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel/g04-computed-segmentation-timing-v2"
CODE = KERNEL / "g04_computed_segmentation_timing_v2.py"
TRACKASTRA_WRFEAT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1/"
    "run_v1/kaggle_output/trackastra-source/trackastra/data/wrfeat.py"
)
EXPECTED_FEATURE_CONTRACT_SHA256 = (
    "910177ae086013bfdff04b1e9528526ff4961f170fbdba49af6f578a8a3436b7"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_package_has_exact_four_files_and_authenticated_inventory():
    files = sorted(path.name for path in KERNEL.iterdir())
    assert files == [
        "PACKAGE_MANIFEST.json",
        "g04_computed_segmentation_timing_v2.py",
        "kernel-metadata.json",
        "prelaunch-contract.json",
    ]
    package = json.loads((KERNEL / "PACKAGE_MANIFEST.json").read_text())
    assert package["schema"].endswith("timing-package.v2")
    assert package["status"] == "PREPARED_CHANGED_SIGNATURE_V2_NO_LAUNCH"
    assert package["package_file_count_excluding_manifest"] == 3
    assert package["kernel_id"] == "onebitbrain/g04-computed-segmentation-timing-v2"
    assert {row["path"] for row in package["files"]} == set(files) - {
        "PACKAGE_MANIFEST.json"
    }
    for row in package["files"]:
        path = KERNEL / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]


def test_v2_metadata_and_prelaunch_contract_are_byte_consistent():
    metadata_path = KERNEL / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    prelaunch = json.loads((KERNEL / "prelaunch-contract.json").read_text())
    assert metadata["id"] == "onebitbrain/g04-computed-segmentation-timing-v2"
    assert metadata["code_file"] == CODE.name
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert metadata["dataset_sources"] == [
        "onebitbrain/g04-computed-segmentation-runtime-cp312-v1"
    ]
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert prelaunch["schema"].endswith("timing-prelaunch.v2")
    assert prelaunch["code_sha256"] == sha256(CODE)
    assert prelaunch["kernel_metadata_sha256"] == sha256(metadata_path)
    changed = prelaunch["timing_gate"]["changed_signature"]
    assert changed["feature_contract_sha256"] == EXPECTED_FEATURE_CONTRACT_SHA256
    assert changed["undefined_sentinel"] is None
    assert changed["replacement_field"] == "solidity_voxel_cell_convex_hull_3d"
    populations = prelaunch["timing_gate"]["projection_populations"]
    assert populations["cv57_frame_count"] == 5_700
    assert populations["held_fit105_frame_count"] == 10_500
    assert populations[
        "fresh_authoritative_quota_reconciliation_required_before_production"
    ] is True


def test_v2_step0_remains_standard_library_only_and_scope_closed():
    tree = ast.parse(CODE.read_text())
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
    source = CODE.read_text()
    assert "STEP0 = run_step0()" in source
    assert "load_bh_detections" not in source
    assert ".geff" not in source.casefold()
    assert "cv57-cache" not in source.casefold()
    for fragment in (
        "label_properties_v2",
        "solidity_voxel_cell_convex_hull_3d",
        "center_affine_dimension_histogram",
        "legacy_skimage_solidity_nonfinite_objects",
        "subdimensional_objects",
        "second_of_two_post_initialization_repeats",
        "combined_projected_t4_gpu_hours_worst",
        "fresh_authoritative_reconciliation_required_before_production",
        "celltrack-alpha.g04-computed-segmentation-cuda-timing.v2",
    ):
        assert fragment in source


def test_pinned_trackastra_native_wrfeat_has_no_solidity_or_silent_imputation():
    tree = ast.parse(TRACKASTRA_WRFEAT.read_text())
    properties = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_PROPERTIES"
            for target in node.targets
        )
    )
    assert properties["regionprops2"] == (
        "equivalent_diameter_area",
        "intensity_mean",
        "inertia_tensor",
        "border_dist",
    )
    source = TRACKASTRA_WRFEAT.read_text()
    assert "solidity" not in source
    assert "nan_to_num" not in source


def test_production_template_carries_the_same_changed_feature_signature():
    source = (
        ROOT
        / "kaggle_kernel/g04-computed-segmentation-production-template/"
        "g04_segmentation_production_template.py"
    ).read_text()
    assert EXPECTED_FEATURE_CONTRACT_SHA256 in source
    assert "label_properties_v2" in source
    assert "solidity_voxel_cell_convex_hull_3d" in source
    assert "computed_object_center_affine_dimension" in source
    assert "_label_properties(" not in source

