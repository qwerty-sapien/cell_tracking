from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_division_colab_source_snapshot import (
    REQUIRED_FILES,
    SnapshotError,
    build_snapshot,
    verify_snapshot,
)


def test_source_snapshot_is_immutable_and_tamper_evident(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    for relative in (*REQUIRED_FILES, "training/entrypoint.py"):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"exact:{relative}\n", encoding="ascii")
    output = tmp_path / "snapshot"
    first = build_snapshot(
        output,
        extra_files=["training/entrypoint.py"],
        repository_root=repository,
    )
    second = build_snapshot(
        output,
        extra_files=["training/entrypoint.py"],
        repository_root=repository,
    )
    assert first == second == verify_snapshot(output)
    assert first["tree_sha256"]
    (output / "training/entrypoint.py").write_text("tampered\n", encoding="ascii")
    with pytest.raises(SnapshotError, match="drifted"):
        verify_snapshot(output)
