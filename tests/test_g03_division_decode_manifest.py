from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.g03_division_decode_manifest import (
    SCHEMA,
    build_configurations,
    build_manifest,
    canonical_sha256,
)


def test_lattice_is_complete_unique_and_has_one_null() -> None:
    rows = build_configurations()
    assert len(rows) == 18_145
    assert len({row.name for row in rows}) == len(rows)
    nulls = [row for row in rows if not row.divisions]
    assert len(nulls) == 1
    assert nulls[0].name == "DIVISIONS_OFF"
    assert sum(row.divisions for row in rows) == 18_144


def test_manifest_hash_binds_exact_configuration_order() -> None:
    manifest = build_manifest()
    assert manifest["schema"] == SCHEMA
    assert manifest["status"] == "LOCKED_BEFORE_OUTCOMES"
    assert manifest["configuration_count_including_null"] == 18_145
    assert manifest["configuration_sha256"] == canonical_sha256(
        manifest["configurations"]
    )
    assert manifest["invariants"]["adaptive_search_forbidden"] is True


def test_cli_writes_replay_identical_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/g03_division_decode_manifest.py",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(result.stdout)
    on_disk = json.loads(output.read_text(encoding="ascii"))
    assert receipt["count"] == 18_145
    assert receipt["configuration_sha256"] == on_disk["configuration_sha256"]
    assert on_disk == build_manifest()
