from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from celltrack_alpha.frame_registration import RegistrationPlan
from celltrack_alpha.temporalunet_registration import (
    canonical_payload_bytes,
    patch_predict_source,
    registration_payload,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PREDICTOR = (
    ROOT / "external/kaggle-cell-tracking-competition/scripts/predict_unet_transformer.py"
)


def fixture_plan() -> RegistrationPlan:
    return RegistrationPlan(
        identity="train:44b6_fixture",
        cumulative_shift_raw_zyx=np.asarray(
            [[0, 0, 0], [1, -8, 12], [1, -8, 12]], dtype=np.int64
        ),
        pool_breaking_frames=(0,),
        census_sha256="a" * 64,
    )


def test_payload_binds_transform_once_coordinate_contract() -> None:
    plan = fixture_plan()
    payload = registration_payload({plan.identity: plan})
    assert payload["coordinate_contract"] == {
        "image_and_unet_feature_sampling": "original",
        "position_association_candidate_geometry": "registered",
        "evaluator_output": "original",
        "applications_per_video": 1,
    }
    decoded = json.loads(canonical_payload_bytes({plan.identity: plan}))
    assert decoded["videos"]["44b6_fixture"]["cumulative_shift_raw_zyx"][1] == [
        1,
        -8,
        12,
    ]


def test_public_predictor_patch_keeps_feature_lookup_raw_and_geometry_registered() -> None:
    patched = patch_predict_source(PUBLIC_PREDICTOR.read_text(encoding="utf-8"))
    assert (
        "unet_out[:, f_idx], p_coords_src, p_mask_src" in patched
    )
    assert "p_geometry_src, p_geometry_tgt" in patched
    assert "p_coords_src * ds_arr_t" not in patched
    assert "geometry_src_raw[i] - geometry_tgt_raw[j]" in patched
    assert "coords[:, 1:] *= ds_arr" in patched
    assert "--registration-plan-json" in patched
    compile(patched, "<registered-v32-test>", "exec")
