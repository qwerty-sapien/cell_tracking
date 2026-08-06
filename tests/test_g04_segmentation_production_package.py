from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/g04_build_segmentation_production_package.py"
ASSEMBLER_PATH = ROOT / "scripts/g04_assemble_segmentation_objects.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load("g04_production_builder_test", BUILDER_PATH)
assembler = load("g04_production_assembler_test", ASSEMBLER_PATH)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def timing_pair(root: Path, elapsed: tuple[float, float] = (10.1, 12.2)):
    bound = int(math.ceil(max(elapsed) * 1.25))
    initialization = 2.0
    steady = elapsed[-1]
    worst = max(elapsed)

    def projection(frame_count: int):
        steady_hours = (initialization + frame_count * steady) / 3_600.0
        worst_hours = (initialization + frame_count * worst) / 3_600.0
        return {
            "frame_count": frame_count,
            "one_initialization_seconds": initialization,
            "steady_repeat_index": 1,
            "steady_seconds_per_frame": steady,
            "worst_seconds_per_frame": worst,
            "projected_t4_gpu_hours_steady_plus_one_initialization": steady_hours,
            "projected_t4_gpu_hours_worst_plus_one_initialization": worst_hours,
            "fits_single_11h_kernel_at_worst": worst_hours <= 11.0,
            "fits_reference_remaining_quota_at_worst": worst_hours <= 28.79,
        }

    cv57 = projection(5_700)
    held_fit105 = projection(10_500)
    result = {
        "schema": "celltrack-alpha.g04-computed-segmentation-cuda-timing.v2",
        "status": "PASS_CUDA_TIMING_GATE",
        "timing_only": True,
        "production_manifest_eligible": False,
        "input": {"identity": "train:timing", "timepoint": 0},
        "repetitions": [
            {
                "repeat": index,
                "elapsed_seconds": seconds,
                "labels_sha256": "a" * 64,
                "regionprops_payload_sha256": "b" * 64,
                "mask_count": 3,
                "region_count": 3,
                "legacy_skimage_solidity_nonfinite_count": 1,
                "legacy_skimage_solidity_nonfinite_objects": [
                    {
                        "label": 3,
                        "field": "legacy_skimage_solidity_3d_center_hull",
                        "value": "inf",
                        "center_affine_dimension": 2,
                        "voxel_count": 15,
                        "bbox_zyxzyx": [1, 2, 3, 2, 5, 8],
                    }
                ],
                "center_affine_dimension_histogram": {
                    "0": 0,
                    "1": 0,
                    "2": 1,
                    "3": 2,
                },
                "subdimensional_objects": [
                    {
                        "label": 3,
                        "center_affine_dimension": 2,
                        "voxel_count": 15,
                        "bbox_zyxzyx": [1, 2, 3, 2, 5, 8],
                    }
                ],
            }
            for index, seconds in enumerate(elapsed)
        ],
        "computed_object_contract": {
            "feature_contract_sha256": builder.FEATURE_CONTRACT_SHA256,
            "undefined_solidity_sentinel": None,
            "legacy_skimage_center_hull_used_as_feature": False,
        },
        "initialization_seconds": initialization,
        "runtime_projection": {
            "steady_definition": "second_of_two_post_initialization_repeats",
            "worst_definition": "max_of_two_post_initialization_repeats",
            "cv57": cv57,
            "held_fit105": held_fit105,
            "combined_projected_t4_gpu_hours_steady": (
                cv57[
                    "projected_t4_gpu_hours_steady_plus_one_initialization"
                ]
                + held_fit105[
                    "projected_t4_gpu_hours_steady_plus_one_initialization"
                ]
            ),
            "combined_projected_t4_gpu_hours_worst": (
                cv57[
                    "projected_t4_gpu_hours_worst_plus_one_initialization"
                ]
                + held_fit105[
                    "projected_t4_gpu_hours_worst_plus_one_initialization"
                ]
            ),
            "quota_feasibility": {
                "fresh_authoritative_reconciliation_required_before_production": True
            },
        },
        "production_admission": {
            "rule": "ceil(max(repeat_seconds) * 1.25)",
            "seconds_per_frame_upper_bound": bound,
        },
        "gt_access": False,
        "gt_node_access": False,
        "gt_edge_access": False,
        "detection_access": False,
        "held_fit_metric_access": False,
        "cv57_metric_access": False,
        "submission": False,
    }
    result_path = root / "timing.json"
    write_json(result_path, result)
    supervisor = {
        "schema": "celltrack-alpha.g04-computed-segmentation-timing-supervisor.v2",
        "status": "PASS_ATOMIC_TIMING_RECEIPT",
        "output_sha256": sha(result_path),
        "seconds_per_frame_upper_bound": bound,
        "production_manifest_eligible": False,
        "submission": False,
        "feature_contract_sha256": builder.FEATURE_CONTRACT_SHA256,
    }
    supervisor_path = root / "supervisor.json"
    write_json(supervisor_path, supervisor)
    return result_path, supervisor_path, bound


def shard_and_archive(root: Path):
    payloads = {
        "tiny.zarr/zarr.json": b"root-metadata",
        "tiny.zarr/0/zarr.json": b"array-metadata",
        "tiny.zarr/0/c/0/0/0/0": b"frame-zero",
        "tiny.zarr/0/c/2/0/0/0": b"frame-two",
    }
    rows = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(payloads.items())
    ]
    shard = {
        "schema": "celltrack-alpha.g04-segmentation-production-shard.v1",
        "status": "ROOT_SUPPLIED_UNFROZEN_SHARD",
        "shard_id": "shard-alpha",
        "split": "train",
        "videos": [
            {
                "identity": "train:tiny",
                "frame_indices": [0, 2],
                "scale_zyx": [1.625, 0.40625, 0.40625],
                "raw_files": rows,
            }
        ],
    }
    shard_path = root / "shard.json"
    write_json(shard_path, shard)
    archive_path = root / "competition.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path, payload in payloads.items():
            archive.writestr(f"train/{path}", payload)
    return shard_path, archive_path, shard


def test_builder_derives_bound_and_renders_raw_only_unfrozen_package() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        timing, supervisor, bound = timing_pair(root)
        shard, archive, _value = shard_and_archive(root)
        output = root / "package"
        args = builder.parser().parse_args(
            [
                "--timing-result",
                str(timing),
                "--timing-supervisor",
                str(supervisor),
                "--shard-manifest",
                str(shard),
                "--competition-archive",
                str(archive),
                "--kernel-id",
                "onebitbrain/g04-seg-prod-alpha",
                "--output",
                str(output),
            ]
        )
        receipt = builder.build_package(args)
        assert receipt["status"] == "PASS_BUILT_NOT_LAUNCH_AUTHORIZED"
        assert receipt["hard_frame_bound_seconds"] == bound
        contract = json.loads((output / "PRODUCTION_RUNTIME_CONTRACT.json").read_text())
        metadata = json.loads((output / "kernel-metadata.json").read_text())
        prelaunch = json.loads((output / "prelaunch-contract.json").read_text())
        package = json.loads((output / "PACKAGE_MANIFEST.json").read_text())
        source = (output / "g04_segmentation_production.py").read_text()
        assert contract["hard_frame_bound_seconds"] == bound
        assert contract["timing"]["result_sha256"] == sha(timing)
        assert contract["assignment_count"] == 2
        assert contract["raw_file_count"] == 4
        assert metadata["competition_sources"] == [
            "biohub-cell-tracking-during-development"
        ]
        assert metadata["dataset_sources"] == [
            "onebitbrain/g04-computed-segmentation-runtime-cp312-v1"
        ]
        assert prelaunch["launch_authorized"] is False
        assert prelaunch["shard_count_frozen"] is False
        assert prelaunch["request_count"] == 2
        assert prelaunch["request_file_count"] == 4
        assert package["status"] == "BUILT_NOT_LAUNCH_AUTHORIZED"
        assert package["file_count_excluding_manifest"] == 6
        for row in package["files"]:
            path = output / row["path"]
            assert path.stat().st_size == row["bytes"]
            assert sha(path) == row["sha256"]
        assert "STEP0 = run_step0()" in source
        assert "load_bh_detections" not in source
        assert ".geff" not in source.casefold()
        assert "g02-trackastra-native-cv57-cache" not in source
        tree = ast.parse(source)
        step0_line = next(
            node.lineno
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "STEP0" for target in node.targets)
        )
        heavy = {"numpy", "scipy", "torch", "zarr", "skimage", "cellpose"}
        for node in tree.body:
            if node.lineno >= step0_line:
                break
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".", 1)[0] for alias in node.names} & heavy
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in heavy


def test_builder_rejects_nonterminal_timing_and_any_shard_count() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        timing, supervisor, _bound = timing_pair(root)
        value = json.loads(timing.read_text())
        value["status"] = "RUNNING"
        write_json(timing, value)
        with pytest.raises(builder.BuildError, match="terminal raw-only PASS"):
            builder.validate_terminal_timing(timing, supervisor)
        _shard_path, _archive, shard = shard_and_archive(root)
        shard["shard_count"] = 9
        with pytest.raises(builder.BuildError, match="must remain unfrozen"):
            builder.validate_shard_manifest(shard)


def object_payload(
    identity: str,
    timepoint: int,
    centroids: np.ndarray,
    native: np.ndarray,
    descriptive: np.ndarray,
):
    from scripts import g04_computed_segmentation as segmentation

    labels = np.arange(1, len(centroids) + 1, dtype="<u4")
    centroids = np.asarray(centroids, dtype="<f4")
    native = np.asarray(native, dtype="<f4")
    descriptive = np.asarray(descriptive, dtype="<f4")
    affine_dimensions = np.full(len(labels), 3, dtype=np.int8)
    digest = hashlib.sha256()
    for array in (labels, centroids, native, descriptive, affine_dimensions):
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "schema": np.asarray("celltrack-alpha.g04-computed-object-frame.v2"),
        "identity": np.asarray(identity),
        "timepoint": np.asarray(timepoint, dtype=np.int32),
        "computed_label_ids": labels,
        "centroids_zyx": centroids,
        "trackastra_native_wrfeat12": native,
        "descriptive_regionprops": descriptive,
        "computed_object_center_affine_dimension": affine_dimensions,
        "feature_names": np.asarray(segmentation.TRACKASTRA_NATIVE_FEATURE_NAMES),
        "descriptive_feature_names": np.asarray(
            segmentation.DESCRIPTIVE_FEATURE_NAMES
        ),
        "descriptive_feature_schema": np.asarray(
            segmentation.DESCRIPTIVE_FEATURE_SCHEMA
        ),
        "solidity_definition": np.asarray(
            segmentation.VOXEL_CELL_SOLIDITY_DEFINITION
        ),
        "feature_contract_sha256": np.asarray(
            assembler.FEATURE_CONTRACT_SHA256
        ),
        "feature_schema": np.asarray("trackastra-ctc-wrfeat12.v1"),
        "scale_zyx": np.asarray([1.625, 0.40625, 0.40625], dtype="<f8"),
        "raw_frame_sha256": np.asarray("c" * 64),
        "computed_labels_sha256": np.asarray("d" * 64),
        "computed_object_payload_sha256": np.asarray(digest.hexdigest()),
        "provenance_label": np.asarray("COMPUTED_SEGMENTATIONS"),
        "authentic_masks": np.asarray(False),
        "organizer_provided": np.asarray(False),
        "h3_authentic_masks": np.asarray(False),
    }


def write_shard_output(root: Path, identity: str = "train:tiny") -> None:
    artifacts = []
    assignments = []
    native0 = np.ones((2, 12), dtype=np.float32)
    native0[:, 0] = [3.0, 4.0]
    native1 = np.ones((1, 12), dtype=np.float32) * 2
    native1[:, 0] = 5.0
    frames = {
        0: object_payload(
            identity,
            0,
            np.asarray([[2.0, 10.0, 10.0], [4.0, 30.0, 30.0]]),
            native0,
            np.ones((2, 12), dtype=np.float32) * 7,
        ),
        1: object_payload(
            identity,
            1,
            np.asarray([[3.0, 12.0, 12.0]]),
            native1,
            np.ones((1, 12), dtype=np.float32) * 8,
        ),
    }
    for timepoint, arrays in frames.items():
        relative = Path("frames") / "train__tiny" / f"t{timepoint:04d}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        sidecar = {
            "schema": "celltrack-alpha.g04-computed-object-frame-sidecar.v2",
            "status": "PASS_COMPLETE_FRAME",
            "identity": identity,
            "timepoint": timepoint,
            "raw_frame_sha256": "c" * 64,
            "computed_labels_sha256": "d" * 64,
            "computed_object_payload_sha256": str(
                arrays["computed_object_payload_sha256"].item()
            ),
            "feature_contract_sha256": assembler.FEATURE_CONTRACT_SHA256,
            "artifact": {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            },
            "geff_access": False,
            "learned_detection_access": False,
            "gt_access": False,
            "metric_access": False,
            "submission": False,
        }
        sidecar_path = path.with_suffix(".json")
        write_json(sidecar_path, sidecar)
        artifacts.extend(
            [
                sidecar["artifact"],
                {
                    "path": str(relative.with_suffix(".json")),
                    "bytes": sidecar_path.stat().st_size,
                    "sha256": sha(sidecar_path),
                },
            ]
        )
        assignments.append({"identity": identity, "timepoint": timepoint})
    receipt = {
        "schema": "celltrack-alpha.g04-computed-object-production-shard.v2",
        "status": "PASS_COMPLETE_SHARD",
        "shard_id": "synthetic-a",
        "assignments": assignments,
        "artifact_files": sorted(artifacts, key=lambda row: row["path"]),
        "production_manifest_eligible": False,
        "requires_local_detection_mapping": True,
        "feature_contract_sha256": assembler.FEATURE_CONTRACT_SHA256,
        "geff_access": False,
        "learned_detection_access": False,
        "gt_access": False,
        "metric_access": False,
        "submission": False,
    }
    write_json(root / "G04_PRODUCTION_SHARD_RECEIPT.json", receipt)


def test_local_assembly_exact_coverage_radius30_and_trackastra_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        shard_root = root / "shard"
        write_shard_output(shard_root)
        population_path = root / "population.json"
        write_json(
            population_path,
            {
                "schema": "celltrack-alpha.g04-segmentation-production-population.v1",
                "status": "ROOT_FROZEN_EXACT_FRAME_POPULATION",
                "identities": [
                    {
                        "identity": "train:tiny",
                        "frame_indices": [0, 1],
                        "scale_zyx": [1.625, 0.40625, 0.40625],
                    }
                ],
            },
        )
        import g04_computed_segmentation as segmentation

        detections = (
            np.asarray([11, 12, 13], dtype=np.int64),
            np.asarray([0, 0, 1], dtype=np.int32),
            np.asarray(
                [[2.0, 10.2, 10.1], [4.0, 29.8, 30.2], [3.0, 12.1, 12.1]],
                dtype=np.float32,
            ),
        )
        monkeypatch.setattr(
            segmentation,
            "load_bh_detections",
            lambda _cache_root, identity: detections
            if identity == "train:tiny"
            else (_ for _ in ()).throw(AssertionError(identity)),
        )
        output = root / "assembled"
        receipt = assembler.assemble(
            population_manifest=population_path,
            cache_root=root / "local-cache",
            shard_outputs=[shard_root],
            output_dir=output,
        )
        assert receipt["status"] == "PASS_READY_FOR_TRACKASTRA_WRFEAT"
        assert receipt["frame_count"] == 2
        assert receipt["detection_count"] == 3
        assert receipt["mapping_radius_physical"] == 30.0
        trackastra = load(
            "g04_trackastra_manifest_compat_test",
            ROOT / "scripts/g04_trackastra_computed_wrfeat.py",
        )
        manifest_path = output / "wrfeat_manifest.json"
        manifest = trackastra.load_feature_manifest(manifest_path)
        video = trackastra.load_video_features(
            manifest_path, manifest, "train:tiny"
        )
        assert video.features.shape == (3, 12)
        assert np.all(video.matched)
        with np.load(video.path, allow_pickle=False) as archive:
            assert np.isfinite(archive["descriptive_regionprops"]).all()
            assert np.all(archive["match_mode"] == "nearest_centroid")
            assert np.max(archive["match_distance_physical"]) < 1.0


def test_local_assembly_rejects_missing_frame_and_out_of_radius() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        shard_root = root / "shard"
        write_shard_output(shard_root)
        frames = assembler.load_shard_outputs([shard_root])
        population = {
            "train:tiny": {
                "frame_indices": (0, 1, 2),
                "scale_zyx": (1.625, 0.40625, 0.40625),
            }
        }
        with pytest.raises(assembler.AssemblyError, match="coverage mismatch"):
            assembler.require_exact_coverage(population, frames)
        frame0 = frames[("train:tiny", 0)]
        with pytest.raises(assembler.AssemblyError, match="incomplete/nonfinite"):
            assembler.map_detections_to_objects(
                node_ids=np.asarray([1]),
                timepoints=np.asarray([0]),
                coords_zyx=np.asarray([[60.0, 250.0, 250.0]], dtype=np.float32),
                frames={0: frame0},
            )
