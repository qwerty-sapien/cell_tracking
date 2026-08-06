from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.g03_phase2_adapter import (
    Phase2AdapterError,
    Phase2CropAdapter,
    adapter_census,
    deterministic_probe,
    publish_h000,
    sha256,
    validate_h000,
)


def test_projection_and_output_are_bitwise_zero() -> None:
    torch.manual_seed(20260803)
    adapter = Phase2CropAdapter()
    census = adapter_census(adapter)
    assert census["projection"] == {
        "in_features": 48,
        "out_features": 12,
        "bias": True,
        "weight_nonzero": 0,
        "bias_nonzero": 0,
    }
    crops = deterministic_probe(batch_size=3)
    hidden = adapter.hidden(crops)
    output = adapter(crops)
    assert torch.count_nonzero(hidden).item() > 0
    assert torch.equal(output, torch.zeros_like(output))


def test_zero_output_projection_learns_then_unblocks_encoder() -> None:
    torch.manual_seed(20260803)
    adapter = Phase2CropAdapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.05)
    crops = deterministic_probe(batch_size=2)
    target = torch.linspace(-0.5, 0.5, 24).reshape(2, 12)

    first_loss = (adapter(crops) - target).square().mean()
    first_loss.backward()
    assert adapter.projection.weight.grad.abs().sum().item() > 0
    assert adapter.projection.bias.grad.abs().sum().item() > 0
    assert adapter.conv1.weight.grad.abs().sum().item() == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second_loss = (adapter(crops) - target).square().mean()
    second_loss.backward()
    assert adapter.conv1.weight.grad.abs().sum().item() > 0
    assert adapter.conv2.weight.grad.abs().sum().item() > 0


def test_invalid_crop_shape_fails_closed() -> None:
    with pytest.raises(Phase2AdapterError, match="shape"):
        Phase2CropAdapter()(torch.zeros(1, 3, 8, 32, 32))


def test_publish_is_immutable_and_validates(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_bytes(b'{"schema":"test"}\n')
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-bytes")
    archive = tmp_path / "ctc.zip"
    archive.write_bytes(b"archive-bytes")
    checkpoint = tmp_path / "h000.pt"

    receipt = publish_h000(
        output=checkpoint,
        seed=20260803,
        phase2_contract_path=contract,
        phase2_contract_sha256=sha256(contract),
        trackastra_model_path=model,
        trackastra_archive_path=archive,
    )
    assert receipt["sha256"] == sha256(checkpoint)
    assert receipt["adapter_output_probe_nonzero"] == 0
    assert validate_h000(checkpoint)["status"] == (
        "PASS_H000_ZERO_INIT_AND_TRAINABILITY"
    )
    with pytest.raises(Phase2AdapterError, match="overwrite"):
        publish_h000(
            output=checkpoint,
            seed=20260803,
            phase2_contract_path=contract,
            phase2_contract_sha256=sha256(contract),
            trackastra_model_path=model,
            trackastra_archive_path=archive,
        )


def test_validation_detects_referenced_model_drift(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_bytes(b'{"schema":"test"}\n')
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-bytes")
    archive = tmp_path / "ctc.zip"
    archive.write_bytes(b"archive-bytes")
    checkpoint = tmp_path / "h000.pt"
    publish_h000(
        output=checkpoint,
        seed=20260803,
        phase2_contract_path=contract,
        phase2_contract_sha256=sha256(contract),
        trackastra_model_path=model,
        trackastra_archive_path=archive,
    )
    model.write_bytes(b"changed-model-bytes")
    with pytest.raises(Phase2AdapterError, match="model reference drifted"):
        validate_h000(checkpoint)
