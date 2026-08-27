#!/usr/bin/env python3
"""Hash every committed data-lane artifact except this generated manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file() and path != OUTPUT
    )
    payload = {
        "schema": "g02.subsample-fidelity.data-lane.v1.artifact-manifest",
        "status": "COMPLETE",
        "hash_algorithm": "SHA-256",
        "self_excluded": OUTPUT.name,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
