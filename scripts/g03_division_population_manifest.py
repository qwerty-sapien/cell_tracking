#!/usr/bin/env python3
"""Freeze the target-blind G-03 R4 development and confirmation populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "celltrack-alpha.g03-r4-division-population.v1"
SEED = "g03-r4-fivefold-fit105-v1"
EXPECTED_SOURCE_SHA256 = {
    "reports/g02/trackastra_h1_feature_cache/report.json":
        "d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf",
    "reports/g02/dataset_registry_v1.json":
        "21f3b3f8bad0324474d4065dda87a89ca2ea6cc770349c426b9399faa6982158",
    "reports/g02/grouped_3fold_audit20_v1.json":
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def build_manifest(root: Path) -> dict[str, Any]:
    for relative, wanted in EXPECTED_SOURCE_SHA256.items():
        observed = sha256(root / relative)
        if observed != wanted:
            raise RuntimeError(f"source identity drift: {relative}: {observed}")
    h1 = load_json(root / "reports/g02/trackastra_h1_feature_cache/report.json")
    registry = load_json(root / "reports/g02/dataset_registry_v1.json")
    legacy_split = load_json(root / "reports/g02/grouped_3fold_audit20_v1.json")
    registry_by_id = {str(row["video_id"]): row for row in registry["records"]}
    fit105 = sorted(str(row["identity"]) for row in h1["files"])
    development = [identity for identity in fit105 if identity.startswith("train:6bba_")]
    held_fit = [identity for identity in fit105 if identity.startswith("train:44b6_")]
    hidden_test = sorted(
        str(row["video_id"]) for row in registry["records"] if row.get("split") == "test"
    )
    contaminated_audit = sorted(
        str(value) for value in legacy_split["audit"]["video_ids"]
    )
    if (len(fit105), len(development), len(held_fit), len(hidden_test), len(contaminated_audit)) != (105, 69, 36, 4, 20):
        raise RuntimeError("architecture population counts drifted")
    roles = [set(development), set(held_fit), set(hidden_test), set(contaminated_audit)]
    if any(roles[left] & roles[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("R4 population roles overlap")
    if any(
        identity not in registry_by_id
        for identity in development + held_fit + hidden_test + contaminated_audit
    ):
        raise RuntimeError("R4 identity absent from frozen dataset registry")

    development_rows = [
        {
            "identity": identity,
            "group_id": str(registry_by_id[identity]["group_id"]),
            "registry_stratum": str(
                registry_by_id[identity]["compact_metadata"]["stratum"]
            ),
        }
        for identity in development
    ]
    by_stratum: dict[str, list[dict[str, str]]] = {}
    for row in development_rows:
        by_stratum.setdefault(row["registry_stratum"], []).append(row)
    folds: dict[str, list[dict[str, str]]] = {str(index): [] for index in range(5)}
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda row: (
                hashlib.sha256(
                    f"{SEED}|{row['group_id']}|{row['identity']}".encode("ascii")
                ).hexdigest(),
                row["identity"],
            ),
        )
        offset = int(hashlib.sha256(f"{SEED}|{stratum}".encode("ascii")).hexdigest(), 16) % 5
        for ordinal, row in enumerate(ordered):
            folds[str((ordinal + offset) % 5)].append(row)
    for rows in folds.values():
        rows.sort(key=lambda row: row["identity"])
    if sorted(row["identity"] for rows in folds.values() for row in rows) != development:
        raise RuntimeError("five-fold population coverage drifted")
    if len({row["group_id"] for row in development_rows}) != len(development_rows):
        raise RuntimeError("development group leakage: one group maps to multiple videos")

    def role_rows(identities: list[str]) -> list[dict[str, str]]:
        return [
            {
                "identity": identity,
                "group_id": str(registry_by_id[identity]["group_id"]),
                "registry_stratum": str(
                    registry_by_id[identity]["compact_metadata"]["stratum"]
                ),
            }
            for identity in identities
        ]

    manifest = {
        "schema": SCHEMA,
        "status": "LOCKED_BEFORE_ANY_G03_DIVISION_CANDIDATE_OUTCOME",
        "seed": SEED,
        "sources": EXPECTED_SOURCE_SHA256,
        "roles": {
            "development_nested_oof": {
                "description": "fit105 6bba-only development; five grouped folds",
                "video_count": 69,
                "target_access": "allowed_only_to_score_presealed_predictions",
                "videos": development_rows,
            },
            "held_fit_one_shot_confirmation": {
                "description": "fit105 44b6 held-fit; one decoder overlay after nested-OOF selection, never selects a setting",
                "video_count": 36,
                "target_access": "one_R4_parent_plus_presealed_decoder_overlay_score_after_config_freeze",
                "videos": role_rows(held_fit),
            },
            "sealed_hidden_test_probe": {
                "description": "four competition hidden-test identities; one observation-distinct autonomous daily probe",
                "video_count": 4,
                "target_access": "exactly_one_submission_after_deployable_setting_and submission_bytes_are_sealed",
                "videos": role_rows(hidden_test),
            },
            "excluded_contaminated_audit20": {
                "description": "legacy audit20 has archived G01 calibration/evaluation exposure and is not confirmatory",
                "video_count": 20,
                "target_access": "forbidden_for_this_R4_node",
                "videos": role_rows(contaminated_audit),
            },
        },
        "outer_folds": folds,
        "constraints": {
            "family_specific_configuration_splicing": False,
            "held_fit_can_select_configuration": False,
            "held_fit_decoder_overlay_evaluations_allowed": 1,
            "confirmation_can_select_configuration": False,
            "division_labels_used_for_split": False,
            "split_stratification_source": "target-blind frozen registry_stratum",
            "all_association_banks_use_one_R1_scientific_configuration": True,
            "development_bank_count_required": 69,
            "held_fit_bank_count_required": 36,
            "hidden_test_bank_count_required": 4,
            "contaminated_audit20_bank_count_required": 0,
        },
    }
    manifest["population_sha256"] = canonical_sha256(
        {
            "seed": SEED,
            "roles": manifest["roles"],
            "outer_folds": folds,
            "constraints": manifest["constraints"],
        }
    )
    return manifest


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_manifest(args.root.resolve())
    atomic_write(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "population_sha256": value["population_sha256"],
                "fold_sizes": {key: len(rows) for key, rows in value["outer_folds"].items()},
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
