from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.build_trackastra_full_pool_feature_cache import (
    DEFAULT_CONTRACT,
    DEFAULT_CONTRACT_SHA256,
    TrackastraFeatureCacheError,
    assert_v1_parity,
    enrich_base_payload,
    flatten_base_windows,
    flatten_windows,
    load_contract,
)
from celltrack_alpha.point_context_swap import RuntimeProposalWindow


def _window() -> RuntimeProposalWindow:
    rows = (
        np.vstack(
            (
                np.arange(48, dtype=np.float32),
                np.arange(48, dtype=np.float32) + 1,
            )
        ),
        np.arange(48, dtype=np.float32)[None],
    )
    return RuntimeProposalWindow(
        identity="train:6bba_fixture",
        family=1,
        target_t=2,
        features=np.stack((rows[0][0], rows[1][0])),
        target_ids=np.asarray((20, 21), dtype=np.int64),
        challengers=((10, 20), (11, 21)),
        conflicts=(((10, 22), (12, 20)), ((11, 23),)),
        row_features=rows,
        geometry_confidences=np.asarray((0.8, 0.9), dtype=np.float32),
        geometry_probabilities=(
            np.asarray((0.8, 0.7), dtype=np.float32),
            np.asarray((0.9,), dtype=np.float32),
        ),
    )


def test_flatten_windows_preserves_lossless_row_identities() -> None:
    result = flatten_windows(
        (_window(),),
        (
            (
                np.asarray((0.25, -0.5), dtype=np.float32),
                np.asarray((0.75,), dtype=np.float32),
            ),
        ),
    )

    assert result["challengers"].tolist() == [[10, 20], [11, 21]]
    assert result["conflicts"].tolist() == [
        [10, 22],
        [12, 20],
        [11, 23],
    ]
    assert result["proposal_offsets"].tolist() == [0, 2, 3]
    assert result["transition_offsets"].tolist() == [0, 2]
    assert result["transition_target_t"].tolist() == [2]
    assert result["row_features"].shape == (3, 49)
    assert result["row_features"][:, -1].tolist() == [0.25, -0.5, 0.75]


def test_two_stage_base_and_enrichment_preserve_row_alignment() -> None:
    base = flatten_base_windows((_window(),))
    assert base["row_features"].shape == (3, 48)
    np.testing.assert_array_equal(
        base["row_features"][0], np.arange(48, dtype=np.float32)
    )

    enriched = enrich_base_payload(
        base, np.asarray((0.25, -0.5, 0.75), dtype=np.float32)
    )
    assert enriched["row_features"].shape == (3, 49)
    np.testing.assert_array_equal(
        enriched["challengers"], base["challengers"]
    )
    np.testing.assert_array_equal(enriched["conflicts"], base["conflicts"])
    assert enriched["row_features"][:, -1].tolist() == [
        0.25,
        -0.5,
        0.75,
    ]

    with pytest.raises(
        TrackastraFeatureCacheError, match="do not align"
    ):
        enrich_base_payload(base, np.asarray((0.25,), dtype=np.float32))


def test_v2_contract_and_exact_v1_parity_reference() -> None:
    contract = load_contract(DEFAULT_CONTRACT, DEFAULT_CONTRACT_SHA256)
    reference = contract["checkpoint_loader"]["parity_reference"]
    payload = torch.load(
        reference["path"], map_location="cpu", weights_only=False
    )
    assert_v1_parity(payload, contract)

    drifted = dict(payload)
    drifted["row_features"] = np.asarray(payload["row_features"]).copy()
    drifted["row_features"][0, -1] += 1.0e-7
    with pytest.raises(
        TrackastraFeatureCacheError,
        match="row_features",
    ):
        assert_v1_parity(drifted, contract)
