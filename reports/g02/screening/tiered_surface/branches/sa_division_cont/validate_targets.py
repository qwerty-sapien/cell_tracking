#!/usr/bin/env python3
"""Fail-closed validation for the division-continuity target specification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division_deep/"
    "oracle_audit_v2.json"
)
SOURCE_SHA256 = "6d1a2754bbe597d53c29863290cba51225d3138b002b83ac4d4ad9c5d306c119"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    spec = json.loads((HERE / "continuity_target_spec.json").read_text(encoding="ascii"))
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, evidence: object) -> None:
        if not condition:
            raise AssertionError(f"{name}: {evidence}")
        checks.append({"name": name, "passed": True, "evidence": evidence})

    counts = spec["counts"]
    events = spec["events"]
    check("source_pin", sha256(SOURCE) == SOURCE_SHA256, sha256(SOURCE))
    check("event_count", len(events) == counts["fragmented_events"] == 38, len(events))
    check(
        "family_partition",
        counts["family_44b6"] == 9 and counts["family_6bba"] == 29,
        counts,
    )
    check(
        "motif_partition",
        counts["motifs"]
        == {
            "DOUBLE_DAUGHTER_REPARENT_DIRECT": 1,
            "SINGLE_DAUGHTER_REPARENT_DIRECT": 37,
        },
        counts["motifs"],
    )
    check(
        "late_equivalence_partition",
        counts["late_fork_equivalence_available"] == 36
        and counts["direct_only_missing_persistence"] == 2,
        counts,
    )
    event_keys = [row["event_key"] for row in events]
    check("event_key_uniqueness", len(event_keys) == len(set(event_keys)), event_keys)
    action_hashes = [row["wave2_action_sha256"] for row in events]
    check(
        "wave2_action_identity",
        len(action_hashes) == len(set(action_hashes)) == 38,
        len(action_hashes),
    )
    check(
        "direct_edges_binary",
        all(len(row["direct_repair"]["required_edges"]) == 2 for row in events),
        [len(row["direct_repair"]["required_edges"]) for row in events],
    )
    check(
        "direct_add_count",
        sum(len(row["direct_repair"]["net_add_edges"]) for row in events) == 39,
        sum(len(row["direct_repair"]["net_add_edges"]) for row in events),
    )
    check(
        "direct_delete_count",
        sum(len(row["direct_repair"]["net_delete_edges"]) for row in events) == 27,
        sum(len(row["direct_repair"]["net_delete_edges"]) for row in events),
    )
    check(
        "late_connectivity",
        all(
            row["late_fork_equivalence"]["preferred"][
                "connects_all_role_and_persistence_nodes"
            ]
            for row in events
            if row["late_fork_equivalence"]["available"]
        ),
        counts["late_fork_equivalence_available"],
    )
    check(
        "fitting_locked",
        spec["contract"]["fitting_locked"]
        and not spec["contract"]["graph_scoring"]
        and not spec["contract"]["cv57_fitting"]
        and spec["contract"]["association_arm_must_be_label_free"],
        spec["contract"],
    )
    check(
        "no_forbidden_actions",
        not any(
            bool(spec["execution"][key])
            for key in (
                "budget_debit",
                "cellect_touched",
                "dead_action",
                "gpu",
                "kaggle",
                "runpod",
                "submission",
            )
        ),
        spec["execution"],
    )
    output = {
        "schema": "celltrack-alpha.sa-division-continuity-validation.v1",
        "status": "PASS",
        "passed": len(checks),
        "total": len(checks),
        "checks": checks,
        "artifact_hashes": {
            "continuity_target_spec.json": sha256(HERE / "continuity_target_spec.json"),
            "derive_continuity_targets.py": sha256(HERE / "derive_continuity_targets.py"),
            "PREREGISTRATION.md": sha256(HERE / "PREREGISTRATION.md"),
        },
    }
    path = HERE / "validation.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "passed": len(checks), "total": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
