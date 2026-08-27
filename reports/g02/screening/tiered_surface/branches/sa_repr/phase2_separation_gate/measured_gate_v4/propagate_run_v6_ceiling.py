"""Propagate the compiler-valid run_v6 occupied-swap ceiling into the tier census.

LIVE_STATE records that SWAP-O0's 44b6 ceiling basis was corrected from the legacy
CG0 fit117 diagnostic `+0.075828` to the compiler-valid run_v6 measurement
`+0.018571397`. Only AE-P1 was actually migrated in the machine-readable census.
AE-P2, AE-P5, AE-P7, and SWAP-O0 itself still carry the superseded value.

This script applies the AE-P1 correction template to those four nodes by targeted
textual replacement, so no unrelated float literal in the file is reformatted.

It changes no tier, no verdict, and no evidence hash other than the census's own.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
CENSUS = ROOT / "reports/g02/screening/tiered_surface/tier_census.json"

BASELINE_SHA = "2d8299ee24b9402b7d57246dbd311c50ec45d695acc0420eb413476367eda22f"

RUN_V6 = {
    "basis": "compiler_valid_frozen_occupied_swap_population_run_v6",
    "control": "exact frozen 44b6 graph control",
    "evidence_artifact": "reports/g02/screening/selector_requirement_44b6/run_v6/requirement_surface.json",
    "evidence_sha256": "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b",
    "family_44b6_dS": 0.01857139664048635,
    "legacy_note": (
        "The prior CG0 fit117 hd44 diagnostic overstates the compiler-valid frozen "
        "occupied-swap ceiling measured by run_v6."
    ),
}

# run_v6 measured 44b6 only. The 6bba ceiling was never re-measured under the
# compiler-valid basis, so each node keeps its own legacy 6bba figure, labelled.
SIX_BBA_BASIS = "legacy_CG0_fit117_not_remeasured_under_run_v6"

TARGETS = ("AE-P2", "AE-P5", "AE-P7", "SWAP-O0")
INDENT = " " * 8


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def find_block(raw: str, node_id: str) -> tuple[int, int, dict]:
    marker = f'"id": "{node_id}"'
    start_id = raw.find(marker)
    if start_id < 0:
        raise SystemExit(f"node {node_id} not found")
    if raw.find(marker, start_id + 1) >= 0:
        raise SystemExit(f"node {node_id} id is not unique")
    start = raw.find('"oracle_headroom": {', start_id)
    if start < 0:
        raise SystemExit(f"node {node_id} has no oracle_headroom")
    end = raw.find("\n      }", start)
    if end < 0:
        raise SystemExit(f"node {node_id} headroom block is unterminated")
    end += len("\n      }")
    block = raw[start:end]
    body = json.loads("{" + block.split("{", 1)[1])
    return start, end, body


def render(existing: dict) -> str:
    ordered = [
        ("basis", RUN_V6["basis"]),
        ("control", RUN_V6["control"]),
        ("evidence_artifact", RUN_V6["evidence_artifact"]),
        ("evidence_sha256", RUN_V6["evidence_sha256"]),
        ("family_44b6_dS", RUN_V6["family_44b6_dS"]),
        ("legacy_cg0_fit117_hd44", existing["family_44b6_dS"]),
        ("legacy_note", RUN_V6["legacy_note"]),
        ("family_6bba_dS", existing["family_6bba_dS"]),
        ("family_6bba_basis", SIX_BBA_BASIS),
    ]
    lines = [f"{INDENT}{json.dumps(k)}: {json.dumps(v)}" for k, v in ordered]
    return '"oracle_headroom": {\n' + ",\n".join(lines) + "\n      }"


def main() -> int:
    raw = CENSUS.read_text()
    if sha256_text(raw) != BASELINE_SHA:
        raise SystemExit("census baseline hash mismatch; refusing to edit")

    changes = []
    for node_id in TARGETS:
        start, end, existing = find_block(raw, node_id)
        if existing.get("basis") == RUN_V6["basis"]:
            print(f"{node_id}: already migrated, skipped")
            continue
        if existing["family_44b6_dS"] != 0.075827918 and not (
            abs(existing["family_44b6_dS"] - 0.07582789132186107) < 1e-15
        ):
            raise SystemExit(
                f"{node_id} carries an unexpected 44b6 basis "
                f"{existing['family_44b6_dS']}; refusing to edit"
            )
        raw = raw[:start] + render(existing) + raw[end:]
        changes.append(
            {
                "node": node_id,
                "from_family_44b6_dS": existing["family_44b6_dS"],
                "to_family_44b6_dS": RUN_V6["family_44b6_dS"],
            }
        )

    parsed = json.loads(raw)  # fail closed on malformed output
    CENSUS.write_text(raw)

    # The census sorts its fallback queue by descending family_44b6_dS, so this
    # correction reorders it. Report the new PARK_A ordering.
    park_a = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("tier") == "PARK_A" and isinstance(node.get("oracle_headroom"), dict):
                park_a.append(
                    (node["id"], node["oracle_headroom"].get("family_44b6_dS"))
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(parsed)
    park_a.sort(key=lambda row: (-(row[1] or 0.0), row[0]))

    print(json.dumps({
        "changes": changes,
        "new_census_sha256": sha256_text(raw),
        "park_a_by_descending_44b6_ceiling": park_a[:10],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
