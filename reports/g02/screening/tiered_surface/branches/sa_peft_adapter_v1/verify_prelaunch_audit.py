#!/usr/bin/env python3
"""Replay the §6.2 PEFT population/outer-fold prelaunch hard stop."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
AUDIT = HERE / "PRELAUNCH_AUDIT_20260731.json"
PREREG = HERE / "PREREGISTRATION.json"
POPULATION_MANIFEST = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/"
    "phase2_separation_gate/run_v1/population_manifest.json"
)
POPULATION_ROOT = POPULATION_MANIFEST.parent / "population/delete"
SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
CHECKPOINT = Path(
    "/private/tmp/trackastra-owner-roundtrip.20260729/resolved/model.pt"
)


class VerificationError(RuntimeError):
    """The sealed prelaunch finding could not be replayed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise VerificationError(f"{path}: expected a JSON object")
    return value


def video_groups(registry: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in registry["records"]:
        identity = row.get("video_id") or row.get("identity")
        group = row.get("group_id")
        if isinstance(identity, str) and isinstance(group, str):
            result[identity] = group
    for row in registry["groups"]:
        group = row.get("group_id")
        if not isinstance(group, str):
            continue
        for key in ("video_ids", "members", "record_ids"):
            for identity in row.get(key, []):
                if isinstance(identity, str) and identity.startswith(("train:", "test:")):
                    result[identity] = group
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def main() -> int:
    audit = load(AUDIT)
    prereg = load(PREREG)
    manifest = load(POPULATION_MANIFEST)
    split = load(SPLIT)
    registry = load(REGISTRY)

    pins = audit["authenticated_inputs"]
    for key, path in (
        ("preregistration", PREREG),
        ("delete_population_manifest", POPULATION_MANIFEST),
        ("outer_split", SPLIT),
        ("dataset_registry", REGISTRY),
        ("trackastra_checkpoint", CHECKPOINT),
    ):
        require(path.is_file(), f"missing declared input: {path}")
        require(
            sha256(path) == pins[key]["sha256"],
            f"byte-SHA mismatch: {path}",
        )

    require(
        prereg["data_and_selection"]["folds"] == 3
        and prereg["data_and_selection"]["folds_sha256"] == sha256(SPLIT),
        "preregistered outer split is not the pinned grouped three-fold artifact",
    )
    require(
        prereg["kill_line"]["nonfinite_or_missing_fold"] == "FAIL_CLOSED",
        "missing-fold policy was weakened",
    )

    target = manifest["totals"]["delete"]
    require(
        target
        == {
            "actions": 1345497,
            "beneficial": 1118,
            "fp_budget": 261,
            "harmful": 1344379,
        },
        "DELETE target surface drifted",
    )

    identity_to_group = video_groups(registry)
    assignments = split["group_assignments"]
    outer_videos: Counter[str] = Counter()
    outer_actions: Counter[str] = Counter()
    outer_beneficial: Counter[str] = Counter()
    population_identities: set[str] = set()
    for row in manifest["classes"]["delete"]["videos"]:
        identity = row["identity"]
        population_identities.add(identity)
        group = identity_to_group.get(identity)
        require(group is not None, f"no registry group for {identity}")
        assignment = assignments.get(group)
        require(assignment is not None, f"no split assignment for {identity}")
        outer_videos[assignment] += 1
        outer_actions[assignment] += int(row["count"])
        outer_beneficial[assignment] += int(row["beneficial"])

    require(
        outer_videos == Counter({"fold_1": 21, "fold_2": 20}),
        f"unexpected outer video support: {outer_videos}",
    )
    require(
        outer_actions == Counter({"fold_1": 648405, "fold_2": 697092}),
        f"unexpected outer action support: {outer_actions}",
    )
    require(
        outer_beneficial == Counter({"fold_1": 576, "fold_2": 542}),
        f"unexpected outer beneficial support: {outer_beneficial}",
    )
    require(
        not (population_identities & set(split["folds"]["0"]["held_out_video_ids"])),
        "outer fold 0 unexpectedly has population support",
    )

    embedded_videos: Counter[int] = Counter()
    embedded_actions: Counter[int] = Counter()
    embedded_beneficial: Counter[int] = Counter()
    for path in sorted(POPULATION_ROOT.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            fold = np.asarray(payload["fold"])
            labels = np.asarray(payload["oracle_label"])
            require(fold.ndim == 1 and len(fold) == len(labels), f"{path}: bad fold array")
            require(np.all(fold == fold[0]), f"{path}: mixed embedded folds")
            value = int(fold[0])
            embedded_videos[value] += 1
            embedded_actions[value] += len(fold)
            embedded_beneficial[value] += int(labels.sum())

    require(
        embedded_videos == Counter({0: 8, 1: 5, 2: 9, 3: 7, 4: 12}),
        f"embedded video folds drifted: {embedded_videos}",
    )
    require(
        embedded_actions
        == Counter({0: 244465, 1: 173496, 2: 311859, 3: 238772, 4: 376905}),
        f"embedded action folds drifted: {embedded_actions}",
    )
    require(
        embedded_beneficial
        == Counter({0: 306, 1: 99, 2: 167, 3: 195, 4: 351}),
        f"embedded beneficial folds drifted: {embedded_beneficial}",
    )
    require(
        audit["status"] == "PRELAUNCH_HARD_STOP_INVALID_EXPERIMENT"
        and audit["decision"]["launch"] is False
        and audit["decision"]["gpu_debit_h"] == 0.0,
        "audit decision drifted",
    )

    print(
        "PASS: exact DELETE target authentic; grouped outer fold 0 has zero "
        "population support; frozen missing-fold policy requires hard stop"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
