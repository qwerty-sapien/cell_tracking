#!/usr/bin/env python3
"""Resume exact scoring from an attested complete fit105 materialization.

This is an infrastructure-only adapter: all graph construction, official scoring,
selection, hybrid construction, and report assembly remain in the frozen evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_trackastra_native_base_wave as frozen  # noqa: E402


EXPECTED_FROZEN_SHA256 = "458e958fa856c1643577ee41f0eea3a9b45de438064d52a6a87e20906489863e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = frozen.parser()
    parser.description = __doc__
    parser.add_argument("--materialized", type=Path, required=True)
    args = parser.parse_args()
    frozen_source = Path(frozen.__file__).resolve()
    if sha256(frozen_source) != EXPECTED_FROZEN_SHA256:
        raise SystemExit("frozen evaluator hash mismatch")
    if not 1 <= args.workers <= 12:
        raise SystemExit("--workers must be between 1 and 12")

    materialized = args.materialized.resolve()
    variants = materialized / "variants"
    if not variants.is_dir():
        raise SystemExit("materialized variants directory is missing")

    def reuse_materialization(
        *,
        bank_dir: Path,
        output: Path,
        identities: tuple[str, ...],
        configs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        structural: dict[str, Any] = {}
        expected_names = {row["name"] for row in configs}
        actual_names = {path.parent.name for path in variants.glob("*/manifest.json")}
        if actual_names != expected_names:
            raise frozen.NativeSweepError("materialized configuration set mismatch")
        for config in configs:
            manifest_path = variants / config["name"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            if manifest.get("configuration") != config or manifest.get("video_count") != 105:
                raise frozen.NativeSweepError(
                    f"materialized manifest mismatch for {config['name']}"
                )
            manifest_identities = tuple(str(row.get("identity")) for row in manifest.get("videos", []))
            if manifest_identities != identities:
                raise frozen.NativeSweepError(
                    f"materialized identity order mismatch for {config['name']}"
                )
            for video in manifest["videos"]:
                fragment = variants / config["name"] / f"{video['identity'].replace(':', '__', 1)}.csv"
                if not fragment.is_file() or sha256(fragment) != video.get("fragment_sha256"):
                    raise frozen.NativeSweepError(
                        f"materialized fragment mismatch for {config['name']} {video['identity']}"
                    )
            structural[config["name"]] = manifest

        bank_hashes: dict[str, str] = {}
        for identity in identities:
            _nodes, _edges, bank_hash = frozen.load_bank(bank_dir, identity)
            bank_hashes[identity] = bank_hash

        destination = output / "variants"
        if destination.exists() or destination.is_symlink():
            raise frozen.NativeSweepError("resume variants destination already exists")
        os.symlink(variants, destination, target_is_directory=True)
        print(
            "resume attestation complete: 108 manifests, 11340 fragments, 105 banks",
            flush=True,
        )
        return structural, bank_hashes

    frozen.materialize = reuse_materialization
    report = frozen.run(args)
    print(
        json.dumps(
            {
                "selected": report["selected"]["name"],
                "summary": report["selected"]["summary"],
                "resume_adapter": str(Path(__file__).resolve()),
                "frozen_evaluator_sha256": EXPECTED_FROZEN_SHA256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
