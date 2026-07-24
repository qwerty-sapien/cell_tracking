"""Low-peak, fail-closed loader for a pinned Trackastra checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile
from typing import Any, Mapping

import torch
import yaml

from evaluate_trackastra_zero_wrfeat_signal import (
    _install_pinned_source,
    sha256,
)


class TrackastraCheckpointLoaderError(RuntimeError):
    """The pinned mmap/meta checkpoint restoration failed closed."""


def load_mmap_meta(
    *,
    source_root: Path,
    checkpoint_zip: Path,
    checkpoint_member: Path,
    contract: Mapping[str, Any],
) -> torch.nn.Module:
    """Restore exact tensors without allocating a duplicate initialized body."""

    asset = contract.get("asset_audit")
    source_files = contract.get("source_files")
    if (
        not isinstance(asset, Mapping)
        or not isinstance(source_files, Mapping)
        or not checkpoint_zip.is_file()
        or not checkpoint_member.is_file()
        or sha256(checkpoint_zip)
        != asset.get("checkpoint_archive_sha256")
        or sha256(checkpoint_member)
        != asset.get("checkpoint_member_sha256")
    ):
        raise TrackastraCheckpointLoaderError(
            "checkpoint archive/member identity mismatch"
        )
    TrackingTransformer = _install_pinned_source(
        source_root, source_files
    )
    try:
        with zipfile.ZipFile(checkpoint_zip) as archive:
            config = yaml.safe_load(archive.read("ctc/config.yaml"))
            digest = hashlib.sha256()
            with archive.open("ctc/model.pt") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    except Exception as exc:
        raise TrackastraCheckpointLoaderError(
            "cannot authenticate checkpoint archive members"
        ) from exc
    if (
        not isinstance(config, dict)
        or digest.hexdigest() != asset.get("checkpoint_member_sha256")
    ):
        raise TrackastraCheckpointLoaderError(
            "checkpoint archive member drifted"
        )
    try:
        with torch.device("meta"):
            model = TrackingTransformer(**config)
        state = torch.load(
            checkpoint_member,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        model.load_state_dict(state, strict=True, assign=True)
    except Exception as exc:
        raise TrackastraCheckpointLoaderError(
            "mmap/meta checkpoint restoration failed"
        ) from exc
    if any(parameter.is_meta for parameter in model.parameters()):
        raise TrackastraCheckpointLoaderError(
            "restoration left meta parameters"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = [
    "TrackastraCheckpointLoaderError",
    "load_mmap_meta",
]
