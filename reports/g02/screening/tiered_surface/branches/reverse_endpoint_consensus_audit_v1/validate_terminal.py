#!/usr/bin/env python3
"""Independent invariant checks for the Lane-2 terminal artifact chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PINS = {
    "PREREGISTRATION.json": "9d0c578b4ef09de25556096c4d31857aa14be6646183755d9954e86311d36b91",
    "SELECTION.json": "1ef8f3ed56f04567bd23ab87ad79582a0e39b859c01890b0fe3ce562cfd3e6c8",
    "SELECTION_FREEZE.json": "a9e7451f6d789e910401a80d898d8eff5fe8c4169eaade7d02ac4adef33aec2f",
    "LABEL_GATE.json": "aad015e283de8a19824e8717641e7f6d80079420e372d5c70f5f63122029cdf3",
    "N1_VALIDATION.json": "7b855abf566e5325225b2c01c89b4246e951ff842ba27cbfe17b515c433710de",
    "RESULT.json": "43a63e12ee0ce298f7703d71d8711b6c96dfe869e323efb59148a3c81a6debe7",
    "TERMINAL_REPORT.md": "b0af1388e8c0598c76b14a31de3f9c8bcbc8136fa87423ad5e48011c1988a7a4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> int:
    checks: dict[str, bool] = {
        f"pin_{name}": sha256(HERE / name) == expected
        for name, expected in PINS.items()
    }
    selection = load("SELECTION.json")
    freeze = load("SELECTION_FREEZE.json")
    labels = load("LABEL_GATE.json")
    n1 = load("N1_VALIDATION.json")
    result = load("RESULT.json")
    checks.update(
        {
            "selection_was_label_blind": selection["scope"]["labels_read"] is False,
            "selection_preceded_labels": freeze["selection"]["sha256"] == PINS["SELECTION.json"],
            "population_exact": selection["scope"]["population"] == 45_961,
            "support_exact": selection["selection_counts"]["selected"] == 53,
            "disabled_reverse_null": selection["selection_counts"]["disabled_reverse_accept_count"] == 0,
            "no_ranking": selection["thresholds"]["ranking"] is False,
            "labels_only_selected": labels["scope"]["selected_labels_read"] == 53
            and labels["scope"]["unselected_labels_read"] == 0,
            "all_selected_neutral": labels["gate"]["beneficial"] == 0
            and labels["gate"]["harmful"] == 53
            and labels["gate"]["min_oracle_delta_S"] == 0.0
            and labels["gate"]["max_oracle_delta_S"] == 0.0,
            "graph_gate_closed": labels["gate"]["graph_gate_open"] is False,
            "n1_exact": n1["status"] == "PASS"
            and n1["actions"] == 53
            and n1["removed_edges"] == 106
            and n1["all_node_disjoint"] is True
            and n1["all_n1_batches_accepted"] is True,
            "graph_not_scored": n1["graph_scored"] is False
            and labels["scope"]["graph_scored"] is False,
            "deployed_noop": result["edit_set"]["deployed_edits"] == 0
            and result["exact_44b6_vector"]["delta_adjusted_raw_node"] == [0.0, 0.0, 0.0],
            "properties_reported": set(result["properties"])
            == {"appearance_grounded", "narrow", "node_disjoint", "recall_preserving"},
            "appearance_caveat_honest": result["properties"]["appearance_grounded"]["satisfied"] is False,
            "downstream_closed": not any(
                result["downstream"][name]
                for name in ("AE-P6_unblocked", "R8_unblocked", "C3_unblocked")
            ),
            "no_forbidden_action": not any(result["forbidden_actions"].values()),
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    receipt = {
        "schema": "g02.reverse-endpoint-consensus-audit.validation.v1",
        "status": "PASS" if not failed else "FAIL",
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "failed": failed,
        "checks": checks,
    }
    output = HERE / "VALIDATION_RECEIPT.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "checks_passed", "checks_total", "failed")}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
