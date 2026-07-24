from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tarfile
import zipfile

import pytest
import torch
import yaml

from scripts.audit_trackastra_pretrained_assets import (
    TrackastraAssetAuditError,
    audit,
)


COMMIT = "f" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assets(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "ctc.zip"
    state = {"weight": torch.ones(2, 3)}
    payload = io.BytesIO()
    torch.save(state, payload)
    config = {
        "coord_dim": 3,
        "d_model": 512,
        "feat_dim": 12,
        "feat_embed_per_dim": 8,
        "nhead": 4,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "window": 4,
        "causal_norm": "quiet_softmax",
        "attn_positional_bias": "rope",
    }
    training = {
        "git_commit": COMMIT,
        "features": "wrfeat",
        "max_tokens": 1024,
        "input_train": ["data/example/train/01"],
    }
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("ctc/model.pt", payload.getvalue())
        archive.writestr("ctc/config.yaml", yaml.safe_dump(config))
        archive.writestr("ctc/train_config.yaml", yaml.safe_dump(training))

    source = tmp_path / "source.tar.gz"
    license_payload = b"BSD 3-Clause License\n\nfixture\n"
    info = tarfile.TarInfo(f"trackastra-{COMMIT}/LICENSE")
    info.size = len(license_payload)
    with tarfile.open(source, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(license_payload))
    return checkpoint, source


def _audit(checkpoint: Path, source: Path):
    return audit(
        checkpoint_zip=checkpoint,
        source_tar=source,
        expected_checkpoint_zip_sha256=_sha(checkpoint),
        expected_source_tar_sha256=_sha(source),
        expected_source_commit=COMMIT,
        release_url="https://example.test/ctc.zip",
        source_url="https://example.test/source.tar.gz",
    )


def test_audit_accepts_pinned_assets(tmp_path: Path) -> None:
    checkpoint, source = _assets(tmp_path)
    result = _audit(checkpoint, source)

    assert result["verdict"] == (
        "STATIC-PROVENANCE-PASS_FEATURE-INTERFACE-UNRESOLVED"
    )
    assert result["checkpoint"]["parameter_count"] == 6
    assert result["checkpoint"]["tensor_bytes"] == 24
    assert result["assets"]["source_license"] == "BSD-3-Clause"
    assert result["interface"]["training_authorized"] is False


def test_audit_rejects_hash_drift(tmp_path: Path) -> None:
    checkpoint, source = _assets(tmp_path)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")

    with pytest.raises(TrackastraAssetAuditError, match="SHA-256 mismatch"):
        audit(
            checkpoint_zip=checkpoint,
            source_tar=source,
            expected_checkpoint_zip_sha256="0" * 64,
            expected_source_tar_sha256=_sha(source),
            expected_source_commit=COMMIT,
            release_url="https://example.test/ctc.zip",
            source_url="https://example.test/source.tar.gz",
        )


def test_audit_rejects_commit_mismatch(tmp_path: Path) -> None:
    checkpoint, source = _assets(tmp_path)

    with pytest.raises(
        TrackastraAssetAuditError, match="source commit does not match"
    ):
        audit(
            checkpoint_zip=checkpoint,
            source_tar=source,
            expected_checkpoint_zip_sha256=_sha(checkpoint),
            expected_source_tar_sha256=_sha(source),
            expected_source_commit="e" * 40,
            release_url="https://example.test/ctc.zip",
            source_url="https://example.test/source.tar.gz",
        )
