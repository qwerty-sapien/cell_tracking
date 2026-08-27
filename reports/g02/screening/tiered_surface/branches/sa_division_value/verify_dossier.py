#!/usr/bin/env python3
"""Fail-closed verifier for the division plan-amendment dossier."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCREENING = HERE.parents[2]
PLAN = SCREENING / "campaign_narrowing_and_hypothesis_dag_v1.md"
B1 = SCREENING / "B1_division_tp.md"
SPEC = HERE.parent / "sa_division_cont" / "continuity_target_spec.json"
BUNDLES = HERE / "DIVISION_REPAIR_BUNDLES.json"
REPLAY = HERE / "ORACLE_ONLY_REPLAY.json"
DOSSIER = HERE / "PLAN_AMENDMENT_DOSSIER.md"
METHODOLOGY = (
    HERE.parent
    / "sa_repr"
    / "phase2_separation_gate"
    / "OPERATING_POINT_METHODOLOGY.md"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_bundle_rows(specification: dict, bundle_artifact: dict) -> None:
    source_rows = specification["events"]
    rows = bundle_artifact["events"]
    require(len(source_rows) == len(rows) == 38, "repair-bundle count drift")
    require(
        [row["event_key"] for row in rows]
        == [row["event_key"] for row in source_rows],
        "repair-bundle order or identity drift",
    )
    for source, row in zip(source_rows, rows, strict=True):
        direct = source["direct_repair"]
        require(
            row["direct_required_edges"] == direct["required_edges"],
            f"direct required-edge drift: {row['event_key']}",
        )
        require(
            row["direct_net_add_edges"] == direct["net_add_edges"]
            and row["direct_net_delete_edges"] == direct["net_delete_edges"],
            f"direct edit drift: {row['event_key']}",
        )
        require(
            row["motif_class"] == source["motif_class"]
            and row["family"] == source["family"]
            and row["identity"] == source["identity"],
            f"event metadata drift: {row['event_key']}",
        )
        require(
            not row["cross_bundle_occupied_node_overlaps"],
            f"cross-bundle occupied-node overlap: {row['event_key']}",
        )
        source_late = source["late_fork_equivalence"]
        late = row["late_fork_equivalence"]
        require(
            late["available"] == source_late["available"],
            f"late-equivalence availability drift: {row['event_key']}",
        )
        require(
            late["scorer_equivalence_key"]
            == source_late["scorer_equivalence_key"],
            f"scorer-equivalence-key drift: {row['event_key']}",
        )
        if late["available"]:
            preferred = source_late["preferred"]
            require(
                late["continuation_child_id"]
                == preferred["continuation_child_id"]
                and late["linear_required_edges"]
                == preferred["linear_continuity"]["required_edges"]
                and late["late_fork_required_edges"]
                == preferred["late_fork"]["required_edges"],
                f"late-equivalence edge drift: {row['event_key']}",
            )
            require(
                row["duplicate_suppression_risk"].startswith(
                    "MUTUALLY_EXCLUSIVE_DIRECT_LATE"
                ),
                f"late duplicate-risk label drift: {row['event_key']}",
            )
        else:
            require(
                late["reason"] == source_late["reason"]
                and row["duplicate_suppression_risk"].startswith(
                    "DIRECT_ONLY"
                ),
                f"direct-only reason drift: {row['event_key']}",
            )


def main() -> int:
    expected_hashes = {
        PLAN: "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116",
        B1: "c9f2b4d1b8e33f5dae456de9e0ac1649b08f2cda8591a67b9d00d4a355dffbb1",
        SPEC: "0926ab647915ad4e2e790252c040867e0bbc45a396155cecfe7f5813d6efe221",
        REPLAY: "0b3c52995a3e18fc705a09bfa5f7f95de6878412acee97901bd5d720199d73cc",
        BUNDLES: "198aeda1d0c0522bd1aedacdbb48e2b343aca0b25cb41d68dd8cb72cbfc87b14",
    }
    for path, expected in expected_hashes.items():
        require(path.is_file(), f"missing required artifact: {path}")
        require(sha256(path) == expected, f"frozen artifact drift: {path}")

    specification = load(SPEC)
    bundle_artifact = load(BUNDLES)
    replay = load(REPLAY)
    require(
        specification["status"] == "PASS_ORACLE_TARGET_SPEC_FITTING_LOCKED"
        and bundle_artifact["status"]
        == "DIAGNOSTIC_ONLY_FITTING_LOCKED",
        "frozen/locked status drift",
    )
    families = Counter(row["family"] for row in specification["events"])
    motifs = Counter(row["motif_class"] for row in specification["events"])
    late = Counter(
        row["late_fork_equivalence"]["available"]
        for row in specification["events"]
    )
    require(families == {"44b6": 9, "6bba": 29}, "family counts drift")
    require(
        motifs
        == {
            "SINGLE_DAUGHTER_REPARENT_DIRECT": 37,
            "DOUBLE_DAUGHTER_REPARENT_DIRECT": 1,
        },
        "motif counts drift",
    )
    require(late == {True: 36, False: 2}, "late-equivalence counts drift")
    verify_bundle_rows(specification, bundle_artifact)

    require(
        replay["status"] == "PASS_BANKED_ORACLE_REVERIFIED_FITTING_LOCKED"
        and replay["repair_bundle_count"] == 38,
        "oracle-replay status drift",
    )
    delta = replay["delta"]
    require(
        (delta["division_tp"], delta["division_fp"], delta["division_fn"])
        == (38, 0, 4),
        "oracle division confusion counts drift",
    )
    require(
        math.isclose(delta["division_jaccard"], 38 / 42, abs_tol=1e-15)
        and math.isclose(
            delta["combined_score_delta"],
            delta["adjusted_edge_jaccard_delta"]
            + 0.1 * delta["division_jaccard"],
            abs_tol=1e-15,
        )
        and math.isclose(
            delta["combined_score_delta"],
            0.09206784151854919,
            abs_tol=1e-15,
        ),
        "oracle score arithmetic drift",
    )
    require(replay["execution"]["oracle_only"], "oracle-only label missing")
    for forbidden in (
        "fitting",
        "plan_amendment",
        "training",
        "kaggle",
        "gpu",
        "runpod",
        "submission",
        "full_pool_cache",
    ):
        require(
            replay["execution"][forbidden] is False,
            f"forbidden execution flag set: {forbidden}",
        )

    dossier = DOSSIER.read_text(encoding="utf-8")
    for token in (
        "0.881522",
        "0.9255–0.9422",
        "+0.044–+0.061",
        "+0.0286",
        "+0.0476",
        "+0.0595",
        "0.896",
        "38/0/4",
        "+0.092067841519",
        "+0.0532",
        "+0.0481",
        "44b6 first",
        "FITTING REMAINS LOCKED",
        "NO AMENDMENT MADE",
    ):
        require(token in dossier, f"dossier evidence missing: {token}")
    for event in specification["events"]:
        require(
            event["event_key"] in dossier,
            f"dossier event row missing: {event['event_key']}",
        )

    methodology = METHODOLOGY.read_text(encoding="utf-8")
    for token in (
        "10^-4",
        "ROC-AUC is not diagnostic",
        "retention (recall)",
        "precision/recall-at-budget",
        "This note changes no gate",
        "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
        "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    ):
        require(token in methodology, f"methodology evidence missing: {token}")

    print(
        json.dumps(
            {
                "status": "PASS_FITTING_LOCKED",
                "repair_bundles": len(bundle_artifact["events"]),
                "family_counts": dict(families),
                "late_equivalence": {"available": 36, "direct_only": 2},
                "oracle_combined_delta": delta["combined_score_delta"],
                "plan_amendment": False,
                "fitting": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
