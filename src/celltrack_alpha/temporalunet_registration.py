"""Deterministic runtime patch for a registered TemporalUNet3D replay.

The immutable public-V32 run is never edited in place.  After that base is
authenticated, a distinct replay may patch its prediction source with this
module.  UNet feature lookup stays at original image coordinates; only
position features, association geometry, edge distances, and hence candidate
topology receive the one cumulative frame transform.  Emitted node
coordinates remain original evaluator coordinates.
"""

from __future__ import annotations

import json
from typing import Mapping

import numpy as np

from .frame_registration import RegistrationPlan


class TemporalUNetRegistrationError(ValueError):
    """The public runtime source or registration payload drifted."""


def registration_payload(plans: Mapping[str, RegistrationPlan]) -> dict[str, object]:
    """Render exact per-dataset raw-grid shifts for a distinct registered replay."""

    if not plans:
        raise TemporalUNetRegistrationError("registered replay requires at least one plan")
    videos: dict[str, object] = {}
    for identity, plan in sorted(plans.items()):
        if identity != plan.identity:
            raise TemporalUNetRegistrationError("registration plan identity key drifted")
        _split, dataset = identity.split(":", 1)
        if dataset in videos:
            raise TemporalUNetRegistrationError("registration payload repeats a dataset")
        videos[dataset] = {
            "cumulative_shift_raw_zyx": np.asarray(
                plan.cumulative_shift_raw_zyx, dtype=np.int64
            ).tolist(),
            "receipt": plan.receipt(),
        }
    return {
        "schema": "celltrack-alpha.temporalunet-registered-replay.v1",
        "coordinate_contract": {
            "image_and_unet_feature_sampling": "original",
            "position_association_candidate_geometry": "registered",
            "evaluator_output": "original",
            "applications_per_video": 1,
        },
        "videos": videos,
    }


def canonical_payload_bytes(plans: Mapping[str, RegistrationPlan]) -> bytes:
    return (
        json.dumps(
            registration_payload(plans),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def patch_predict_source(source: str) -> str:
    """Patch one known public predictor source; fail if any seam is ambiguous."""

    replacements = (
        (
            "    downsample: tuple[int, ...] = (1, 4, 4),\n",
            "    downsample: tuple[int, ...] = (1, 4, 4),\n"
            "    registration_shifts_raw_zyx: np.ndarray | None = None,\n"
        ),
        (
            "    ds_arr_t = torch.from_numpy(ds_arr).to(device)   # for predict_edges (original-space coords)\n",
            "    ds_arr_t = torch.from_numpy(ds_arr).to(device)   # feature-grid to raw-grid scale\n"
            "    if registration_shifts_raw_zyx is None:\n"
            "        raise ValueError('standing global-jump preprocessing plan is required')\n"
            "    registration_shifts_raw_zyx = np.asarray(\n"
            "        registration_shifts_raw_zyx, dtype=np.float32\n"
            "    )\n"
            "    if (registration_shifts_raw_zyx.shape != (T, 3)\n"
            "            or not np.isfinite(registration_shifts_raw_zyx).all()):\n"
            "        raise ValueError('registration shift plan shape/value drift')\n",
        ),
        (
            "            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)\n"
            "            # Use window-relative time (f_idx, f_idx+1) normalised by W, not absolute frame index.\n",
            "            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)\n"
            "            geometry_src_raw = (\n"
            "                c_src[:, 1:].astype(np.float32) * ds_arr[None, :]\n"
            "                + registration_shifts_raw_zyx[t_src][None, :]\n"
            "            )\n"
            "            geometry_tgt_raw = (\n"
            "                c_tgt[:, 1:].astype(np.float32) * ds_arr[None, :]\n"
            "                + registration_shifts_raw_zyx[t_tgt][None, :]\n"
            "            )\n"
            "            p_geometry_src = torch.from_numpy(geometry_src_raw).unsqueeze(0).to(device)\n"
            "            p_geometry_tgt = torch.from_numpy(geometry_tgt_raw).unsqueeze(0).to(device)\n"
            "            # Use window-relative time with registered spatial position features.\n",
        ),
        (
            "            c_src_rel = c_src.copy()\n"
            "            c_src_rel[:, 0] = f_idx\n"
            "            c_tgt_rel = c_tgt.copy()\n"
            "            c_tgt_rel[:, 0] = f_idx + 1\n",
            "            c_src_rel = c_src.astype(np.float32, copy=True)\n"
            "            c_src_rel[:, 0] = f_idx\n"
            "            c_src_rel[:, 1:] = geometry_src_raw / ds_arr[None, :]\n"
            "            c_tgt_rel = c_tgt.astype(np.float32, copy=True)\n"
            "            c_tgt_rel[:, 0] = f_idx + 1\n"
            "            c_tgt_rel[:, 1:] = geometry_tgt_raw / ds_arr[None, :]\n",
        ),
        (
            "                    coords_so_far[gi, 1:].astype(np.float32)\n"
            "                    - coords_so_far[gj, 1:].astype(np.float32)\n",
            "                    geometry_src_raw[i] - geometry_tgt_raw[j]\n",
        ),
        (
            "    evaluate: bool = False,\n) -> None:",
            "    evaluate: bool = False,\n"
            "    registration_plans: dict[str, object] | None = None,\n"
            ") -> None:",
        ),
        (
            "        coords, edges = predict_video(\n"
            "                model, ds_path, device,\n"
            "                cfg=cfg,\n",
            "        dataset_key = Path(name).stem\n"
            "        if registration_plans is None or dataset_key not in registration_plans:\n"
            "            raise ValueError(f'missing standing registration plan for {dataset_key}')\n"
            "        plan_row = registration_plans[dataset_key]\n"
            "        coords, edges = predict_video(\n"
            "                model, ds_path, device,\n"
            "                cfg=cfg,\n"
            "                registration_shifts_raw_zyx=np.asarray(\n"
            "                    plan_row['cumulative_shift_raw_zyx'], dtype=np.float32\n"
            "                ),\n",
        ),
        (
            "    parser.add_argument(\"--evaluate\", action=\"store_true\",\n",
            "    parser.add_argument(\"--registration-plan-json\", type=str, required=True,\n"
            "                        help=\"Exact standing frame-registration payload.\")\n"
            "    parser.add_argument(\"--evaluate\", action=\"store_true\",\n",
        ),
        (
            "    args = parser.parse_args()\n\n"
            "    from dataspec import DATASET_PATH\n",
            "    args = parser.parse_args()\n"
            "    registration_payload = json.loads(Path(args.registration_plan_json).read_text())\n"
            "    if (registration_payload.get('schema')\n"
            "            != 'celltrack-alpha.temporalunet-registered-replay.v1'):\n"
            "        raise ValueError('registration payload schema drift')\n"
            "    registration_plans = registration_payload.get('videos')\n"
            "    if not isinstance(registration_plans, dict) or not registration_plans:\n"
            "        raise ValueError('registration payload contains no video plans')\n\n"
            "    from dataspec import DATASET_PATH\n",
        ),
        (
            "            evaluate=args.evaluate,\n",
            "            evaluate=args.evaluate,\n"
            "            registration_plans=registration_plans,\n",
        ),
    )
    patched = source
    for old, new in replacements:
        count = patched.count(old)
        if count != 1:
            raise TemporalUNetRegistrationError(
                f"public predictor registration seam expected once, found {count}"
            )
        patched = patched.replace(old, new, 1)
    geometry_old = (
        "                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n"
    )
    geometry_count = patched.count(geometry_old)
    if geometry_count not in {1, 2}:
        raise TemporalUNetRegistrationError(
            "public predictor must contain one primary and at most one secondary "
            f"association geometry seam, found {geometry_count}"
        )
    patched = patched.replace(
        geometry_old,
        "                p_geometry_src, p_geometry_tgt,\n",
    )
    # Feature indexing must still use original p_coords, while every model
    # association call must now consume the registered geometry tensors.
    if "_index_features(\n                unet_out[:, f_idx], p_coords_src" not in patched:
        raise TemporalUNetRegistrationError("original-coordinate feature lookup seam drifted")
    if "p_coords_src * ds_arr_t" in patched:
        raise TemporalUNetRegistrationError("unregistered association geometry remains")
    compile(patched, "<registered_predict_unet_transformer>", "exec")
    return patched
