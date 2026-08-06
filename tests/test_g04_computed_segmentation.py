from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g04_computed_segmentation",
    ROOT / "scripts/g04_computed_segmentation.py",
)
assert SPEC is not None and SPEC.loader is not None
g04 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = g04
SPEC.loader.exec_module(g04)


def cube_fixture():
    labels = np.zeros((9, 9, 9), dtype=np.uint32)
    labels[2:5, 2:5, 2:5] = 7
    intensity = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
    return labels, intensity


def test_native_schema_is_exact_checkpoint_contract():
    assert g04.TRACKASTRA_NATIVE_FEATURE_NAMES == (
        "equivalent_diameter_area",
        "intensity_mean",
        "inertia_tensor-0-0",
        "inertia_tensor-0-1",
        "inertia_tensor-0-2",
        "inertia_tensor-1-0",
        "inertia_tensor-1-1",
        "inertia_tensor-1-2",
        "inertia_tensor-2-0",
        "inertia_tensor-2-1",
        "inertia_tensor-2-2",
        "border_dist",
    )
    assert len(g04.DESCRIPTIVE_FEATURE_NAMES) == 12
    assert (
        "solidity_voxel_cell_convex_hull_3d"
        in g04.DESCRIPTIVE_FEATURE_NAMES
    )
    assert g04.DESCRIPTIVE_FEATURE_SCHEMA.endswith("voxel-cell-solidity.v2")
    defaults = g04.parser().parse_args(["preflight"])
    assert defaults.max_nearest_distance_physical == 30.0


def test_batch_hard_deadline_cannot_reach_kaggle_cap(tmp_path):
    args = g04.parser().parse_args(
        [
            "batch",
            "--deadline-seconds",
            "39601",
            "--seconds-per-frame-upper-bound",
            "1",
        ]
    )
    with pytest.raises(g04.ComputedSegmentationError, match="deadline"):
        g04.batch(args)


def test_trackastra_normalization_is_per_frame_float32_p1_p998():
    value = np.arange(1000, dtype=np.uint16).reshape(10, 10, 10)
    observed = g04._trackastra_normalize_frame(value)
    low, high = np.percentile(value.astype(np.float32), (1, 99.8)).astype(np.float32)
    expected = (value.astype(np.float32) - low) / (high - low + np.float32(1e-8))
    np.testing.assert_array_equal(observed, expected)


def test_mapping_emits_native_and_separate_rich_morphology():
    labels, intensity = cube_fixture()
    mapped = g04.map_detections_to_regions(
        labels,
        intensity,
        np.asarray([[3.0, 3.0, 3.0]], dtype=np.float32),
        scale_zyx=(2.0, 1.0, 1.0),
        max_nearest_distance_physical=3.0,
    )
    assert mapped["matched"].tolist() == [True]
    assert mapped["mask_labels"].tolist() == [7]
    assert mapped["match_mode"].tolist() == ["contains"]
    assert mapped["features"].shape == (1, 12)
    assert mapped["descriptive_regionprops"].shape == (1, 12)
    assert np.isfinite(mapped["features"]).all()
    assert np.isfinite(mapped["descriptive_regionprops"]).all()

    rich = dict(
        zip(
            g04.DESCRIPTIVE_FEATURE_NAMES,
            mapped["descriptive_regionprops"][0],
            strict=True,
        )
    )
    assert rich["volume_voxels"] == 27
    assert rich["axis_length_major_physical"] > rich["axis_length_minor_physical"]
    assert rich["solidity_voxel_cell_convex_hull_3d"] == pytest.approx(1.0)
    assert rich["intensity_max"] == intensity[4, 4, 4]

    native_frame = g04._trackastra_normalize_frame(intensity)
    expected_native_mean = float(np.mean(native_frame[labels == 7]))
    assert mapped["features"][0, 1] == pytest.approx(expected_native_mean)
    # Cube is more than one pixel from XY frame edge: exact WRFeatures cutoff.
    assert mapped["features"][0, -1] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("coordinates", "expected_dimension"),
    (
        ([(1, 2, 1), (1, 2, 2), (1, 2, 3)], 1),
        (
            [
                (1, 1, 1),
                (1, 1, 2),
                (1, 1, 3),
                (1, 2, 1),
                (1, 2, 2),
                (1, 2, 3),
            ],
            2,
        ),
    ),
)
def test_voxel_cell_solidity_is_finite_for_line_and_planar_objects(
    coordinates, expected_dimension
):
    from skimage.measure import regionprops

    labels = np.zeros((4, 6, 6), dtype=np.uint32)
    for coordinate in coordinates:
        labels[coordinate] = 9
    intensity = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
    legacy = float(regionprops(labels)[0].solidity)
    assert not np.isfinite(legacy)

    properties = g04._label_properties(
        labels,
        intensity,
        g04._trackastra_normalize_frame(intensity),
        (1.625, 0.40625, 0.40625),
    )[9]
    solidity_index = g04.DESCRIPTIVE_FEATURE_NAMES.index(
        "solidity_voxel_cell_convex_hull_3d"
    )
    solidity = float(properties["descriptive"][solidity_index])
    assert 0.0 < solidity <= 1.0
    assert properties["center_affine_dimension"] == expected_dimension
    assert properties["solidity_method"] == "voxel_cell_convex_hull_3d"


def test_nonfinite_diagnostic_names_exact_label_object_and_field():
    labels, intensity = cube_fixture()
    native = g04._trackastra_normalize_frame(intensity)
    native[labels == 7] = np.nan
    with pytest.raises(
        g04.ComputedSegmentationError,
        match=(
            r"label=7; voxel_count=27; center_affine_dimension=3; "
            r"fields=\['native.intensity_mean'\]; values=\['nan'\]"
        ),
    ):
        g04._label_properties(
            labels,
            intensity,
            native,
            (1.0, 1.0, 1.0),
        )


def test_mapping_uses_deterministic_nearest_and_exposes_unmatched():
    labels, intensity = cube_fixture()
    near = g04.map_detections_to_regions(
        labels,
        intensity,
        np.asarray([[3.0, 3.0, 5.0]], dtype=np.float32),
        scale_zyx=(1.0, 1.0, 1.0),
        max_nearest_distance_physical=2.1,
    )
    assert near["matched"].tolist() == [True]
    assert near["match_mode"].tolist() == ["nearest_centroid"]
    assert near["match_distance_physical"][0] == pytest.approx(2.0)

    far = g04.map_detections_to_regions(
        labels,
        intensity,
        np.asarray([[8.0, 8.0, 8.0]], dtype=np.float32),
        scale_zyx=(1.0, 1.0, 1.0),
        max_nearest_distance_physical=2.0,
    )
    assert far["matched"].tolist() == [False]
    assert far["match_mode"].tolist() == ["unmatched"]
    assert np.isnan(far["features"]).all()
    with pytest.raises(g04.ComputedSegmentationError, match="incomplete"):
        g04.validate_checkpoint_payload(
            {
                "node_ids": np.asarray([11]),
                "features": far["features"],
                "descriptive_regionprops": far["descriptive_regionprops"],
                "matched": far["matched"],
            }
        )


def test_centered_crop_is_fixed_shape_and_retains_negative_origin():
    image = np.arange(24 * 40 * 40, dtype=np.uint16).reshape(24, 40, 40)
    crop, origin, center = g04.centered_crop(
        image, (8, 2, 2), shape_zyx=(16, 32, 32)
    )
    assert crop.shape == (16, 32, 32)
    assert origin == (0, -14, -14)
    assert center == (8, 16, 16)
    np.testing.assert_array_equal(crop[:, 14:, 14:], image[:16, :18, :18])
    assert not np.any(crop[:, :14, :])
    assert not np.any(crop[:, :, :14])


def test_crop_wrfeat_uses_global_normalization_and_full_frame_border():
    class FixtureSegmenter:
        backend_name = "fixture"
        provenance = {"backend": "fixture"}

        def segment(self, image_zyx):
            labels = np.zeros_like(image_zyx, dtype=np.uint32)
            center = np.asarray(labels.shape) // 2
            labels[
                center[0] - 1 : center[0] + 2,
                center[1] - 1 : center[1] + 2,
                center[2] - 1 : center[2] + 2,
            ] = 3
            return labels

    frame = np.arange(24 * 40 * 40, dtype=np.uint16).reshape(24, 40, 40)
    coord = np.asarray([[8.0, 2.0, 2.0]], dtype=np.float32)
    observed = g04.crop_detection_features(
        frame_zyx=frame,
        coords_zyx=coord,
        segmenter=FixtureSegmenter(),
        scale_zyx=(1.625, 0.40625, 0.40625),
        max_nearest_distance_physical=30.0,
    )
    assert observed["matched"].tolist() == [True]
    assert observed["match_mode"].tolist() == ["contains"]
    assert observed["selected_mask_boundary_contact"].tolist() == [False]
    normalized = g04._trackastra_normalize_frame(frame)
    expected_mean = float(np.mean(normalized[7:10, 1:4, 1:4]))
    assert observed["features"][0, 1] == pytest.approx(expected_mean)
    # The crop-local object is far from its XY edge, but its original-frame
    # coordinates are one voxel from the border, so native border_dist is 0.8.
    assert observed["features"][0, -1] == pytest.approx(0.8)


def test_intensity_recenter_is_bounded_in_physical_space():
    frame = np.zeros((32, 64, 64), dtype=np.uint16)
    frame[18:21, 42:45, 42:45] = 1000
    coord = np.asarray([16.0, 32.0, 32.0])
    anchor, shift = g04.intensity_recentered_anchor(
        frame,
        coord,
        scale_zyx=(1.625, 0.40625, 0.40625),
        maximum_shift_um=4.0,
    )
    assert shift == pytest.approx(4.0)
    assert np.linalg.norm((anchor - coord) * np.asarray((1.625, 0.40625, 0.40625))) == pytest.approx(4.0)


def test_adaptive_crop_uses_genuine_batch_and_recovers_boundary_rows():
    class BatchFixture:
        backend_name = "batch-fixture"
        provenance = {"backend": "batch-fixture"}
        _device = "cuda"

        def __init__(self):
            self.calls = []

        def segment_batch(self, images, *, network_batch_size):
            self.calls.append((tuple(images.shape), network_batch_size))
            labels = np.zeros_like(images, dtype=np.uint32)
            for index in range(len(labels)):
                center = np.asarray(labels[index].shape) // 2
                if labels.shape[1:] == g04.CROP_PRIMARY_SHAPE_ZYX:
                    labels[index, :, :, :] = 3  # forced primary boundary failure
                else:
                    labels[
                        index,
                        center[0] - 2 : center[0] + 3,
                        center[1] - 3 : center[1] + 4,
                        center[2] - 3 : center[2] + 4,
                    ] = 7
            return labels

        def segment(self, image):  # pragma: no cover - genuine batch must be used
            raise AssertionError("serial segment path was used")

    frame = np.ones((64, 128, 128), dtype=np.uint16)
    coords = np.asarray([[32, 64, 64], [30, 60, 62]], dtype=np.float32)
    segmenter = BatchFixture()
    observed = g04.adaptive_crop_detection_features(
        frame_zyx=frame,
        coords_zyx=coords,
        segmenter=segmenter,
        scale_zyx=(1.625, 0.40625, 0.40625),
        max_nearest_distance_physical=30.0,
        crop_batch_size=2,
        network_batch_size=64,
    )
    assert observed["fallback_attempted"].tolist() == [True, True]
    assert observed["fallback_recovered"].tolist() == [True, True]
    assert observed["matched"].tolist() == [True, True]
    assert observed["selected_mask_boundary_contact"].tolist() == [False, False]
    assert np.isfinite(observed["features"]).all()
    assert segmenter.calls == [
        ((2, *g04.CROP_PRIMARY_SHAPE_ZYX), 64),
        ((2, *g04.CROP_FALLBACK_SHAPE_ZYX), 64),
    ]
    assert observed["batch_runtime"]["primary"][
        "genuine_cross_detection_gpu_batching"
    ] is True


def test_geometry_audit_reports_object_size_and_crop_mismatch():
    labels = np.zeros((32, 80, 80), dtype=np.uint32)
    labels[5:25, 10:55, 15:55] = 4
    frame = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape)
    report = g04.crop_geometry_audit(
        labels_zyx=labels,
        raw_frame_zyx=frame,
        coords_zyx=np.asarray([[15, 32, 35]], dtype=np.float32),
        scale_zyx=(1.625, 0.40625, 0.40625),
    )
    assert report["object_population"]["unique_computed_object_count"] == 1
    assert report["object_population"]["voxel_extent_zyx"]["z"]["p99"] == 20
    assert report["detection_anchor_population"][
        "current_crop_full_containment_fraction_of_matched"
    ] == 0.0


def test_publish_payload_marks_non_authentic_and_is_adapter_compatible(tmp_path):
    labels, intensity = cube_fixture()
    mapped = g04.map_detections_to_regions(
        labels,
        intensity,
        np.asarray([[3.0, 3.0, 3.0]], dtype=np.float32),
        scale_zyx=(1.0, 1.0, 1.0),
        max_nearest_distance_physical=2.0,
    )
    payload = {
        "node_ids": np.asarray([11], dtype=np.int64),
        "timepoints": np.asarray([0], dtype=np.int32),
        "coords_zyx": np.asarray([[3.0, 3.0, 3.0]], dtype=np.float32),
        **mapped,
    }
    output = tmp_path / "features.npz"
    sidecar = g04.publish_payload(
        output,
        payload=payload,
        identity="train:fixture",
        zarr_path=tmp_path / "raw.zarr",
        scale_zyx=(1.0, 1.0, 1.0),
        segmenter_provenance={
            "backend": "fixture-computed-segmenter",
            "weight": {"sha256": "a" * 64},
            "license": {"spdx": "BSD-3-Clause", "sha256": "b" * 64},
        },
    )
    assert sidecar["provenance_label"] == "COMPUTED_SEGMENTATIONS"
    assert sidecar["explicit_nonclaims"] == {
        "authentic_masks": False,
        "organizer_provided": False,
        "h3_authentic_masks": False,
    }
    assert sidecar["feature_contract"]["checkpoint_channels_overloaded"] is False
    assert json.loads(output.with_suffix(".json").read_text()) == sidecar
    with np.load(output, allow_pickle=False) as data:
        assert data["provenance_label"].item() == "COMPUTED_SEGMENTATIONS"
        assert data["authentic_masks"].item() is False
        assert data["organizer_provided"].item() is False
        assert data["feature_schema"].item() == "trackastra-ctc-wrfeat12.v1"
        assert tuple(data["feature_names"].tolist()) == g04.TRACKASTRA_NATIVE_FEATURE_NAMES
        assert data["matched"].tolist() == [True]
        assert data["features"].shape == (1, 12)


def test_manifest_requires_exact_complete_identity_set(tmp_path):
    labels, intensity = cube_fixture()
    mapped = g04.map_detections_to_regions(
        labels,
        intensity,
        np.asarray([[3.0, 3.0, 3.0]], dtype=np.float32),
        scale_zyx=(1.0, 1.0, 1.0),
        max_nearest_distance_physical=2.0,
    )
    payload = {
        "node_ids": np.asarray([11], dtype=np.int64),
        "timepoints": np.asarray([0], dtype=np.int32),
        "coords_zyx": np.asarray([[3.0, 3.0, 3.0]], dtype=np.float32),
        **mapped,
    }
    feature = tmp_path / "features.npz"
    g04.publish_payload(
        feature,
        payload=payload,
        identity="train:fixture",
        zarr_path=tmp_path / "raw.zarr",
        scale_zyx=(1.0, 1.0, 1.0),
        segmenter_provenance={"backend": "fixture"},
    )
    expected = tmp_path / "expected.json"
    g04._atomic_json(expected, {"identities": ["train:fixture"]})
    output = tmp_path / "wrfeat_manifest.json"
    args = g04.parser().parse_args(
        [
            "manifest",
            "--output",
            str(output),
            "--expected-identities-file",
            str(expected),
            "--feature-npz",
            str(feature),
        ]
    )
    manifest = g04.build_manifest(args)
    assert manifest["schema"] == g04.MANIFEST_SCHEMA
    assert manifest["coverage"] == "PRODUCTION_COMPLETE_ONLY"
    assert manifest["partial_or_pilot_allowed"] is False
    assert manifest["identities"][0]["identity"] == "train:fixture"
    assert manifest["identities"][0]["sha256"] == g04.sha256(feature)

    g04._atomic_json(expected, {"identities": ["train:fixture", "train:missing"]})
    with pytest.raises(g04.ComputedSegmentationError, match="coverage mismatch"):
        g04.build_manifest(args)


def test_zarr_and_detection_preflight_reports_precise_asset_blockers(tmp_path):
    import zarr

    raw = tmp_path / "fixture.zarr"
    group = zarr.open_group(str(raw), mode="w")
    group.create_array(
        "0", data=np.zeros((2, 4, 8, 8), dtype=np.uint16), chunks=(1, 4, 8, 8)
    )
    group.attrs["multiscales"] = [
        {
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 2.0, 1.0, 1.0]}
                    ],
                }
            ]
        }
    ]
    detections = tmp_path / "detections.npz"
    np.savez_compressed(
        detections,
        node_ids=np.asarray([1, 2], dtype=np.int64),
        timepoints=np.asarray([0, 1], dtype=np.int32),
        coords_zyx=np.asarray([[1, 2, 3], [1, 2, 4]], dtype=np.float32),
    )
    args = g04.parser().parse_args(
        [
            "preflight",
            "--zarr",
            str(raw),
            "--identity",
            "train:fixture",
            "--detections-npz",
            str(detections),
        ]
    )
    report = g04.preflight(args)
    assert report["status"] == "BLOCKED_PRETRAINED_ASSET_OR_RUNTIME"
    assert report["raw"]["shape_tzyx"] == [2, 4, 8, 8]
    assert report["raw"]["scale_zyx"] == [2.0, 1.0, 1.0]
    assert report["detections"]["count"] == 2
    if importlib.util.find_spec("cellpose") is None:
        assert "cellpose_3_1_1_2_import" in report["blockers"]
    else:
        assert report["checks"]["cellpose_3_1_1_2_import"] is True
    assert "weight_present" in report["blockers"]
    assert "license_present" in report["blockers"]
