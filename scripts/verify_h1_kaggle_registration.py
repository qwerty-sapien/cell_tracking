#!/usr/bin/env python3
"""Fail closed unless Kaggle registered every H1 kernel data source."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


EXPECTED_DATASET_SOURCES = frozenset(
    {
        "onebitbrain/trackastra-6bba-assets-v1",
        "onebitbrain/trackastra-h1-option-c-assets-v1",
        "onebitbrain/trackastra-h1-fit105-cache-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    }
)
EXPECTED_COMPETITION_SOURCES = frozenset(
    {"biohub-cell-tracking-during-development"}
)


class RegistrationError(RuntimeError):
    """Kaggle did not preserve the locked kernel source registration."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegistrationError(f"cannot read metadata: {path}") from exc
    if not isinstance(value, dict):
        raise RegistrationError(f"metadata is not an object: {path}")
    return value


def registration_report(
    local: Mapping[str, Any],
    remote: Mapping[str, Any],
) -> dict[str, Any]:
    local_datasets = frozenset(
        str(value) for value in local.get("dataset_sources", ())
    )
    remote_datasets = frozenset(
        str(value) for value in remote.get("dataset_sources", ())
    )
    remote_competitions = frozenset(
        str(value) for value in remote.get("competition_sources", ())
    )
    missing_local = sorted(EXPECTED_DATASET_SOURCES - local_datasets)
    missing_remote = sorted(EXPECTED_DATASET_SOURCES - remote_datasets)
    unexpected_remote = sorted(remote_datasets - EXPECTED_DATASET_SOURCES)
    competition_drift = sorted(
        EXPECTED_COMPETITION_SOURCES ^ remote_competitions
    )
    passed = not (
        missing_local
        or missing_remote
        or unexpected_remote
        or competition_drift
    )
    return {
        "schema": "celltrack-alpha.h1-kaggle-registration-check.v1",
        "status": "PASS" if passed else "STOP_SOURCE_REGISTRATION_DRIFT",
        "expected_dataset_sources": sorted(EXPECTED_DATASET_SOURCES),
        "local_dataset_sources": sorted(local_datasets),
        "remote_dataset_sources": sorted(remote_datasets),
        "missing_from_local": missing_local,
        "missing_from_remote": missing_remote,
        "unexpected_remote": unexpected_remote,
        "competition_source_drift": competition_drift,
        "timing_smoke_authorized": passed,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-metadata", type=Path, required=True)
    parser.add_argument("--remote-metadata", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = registration_report(
        load_json(args.local_metadata),
        load_json(args.remote_metadata),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
