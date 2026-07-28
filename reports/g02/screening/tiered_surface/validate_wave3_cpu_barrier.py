#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SURFACE = ROOT / "reports/g02/screening/tiered_surface"
OUTPUT = SURFACE / "wave3_cpu_barrier_validation.json"

EXPECTED_FILES = {
    "reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md":
        "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116",
    "reports/g02/screening/tiered_surface/ALIGNMENT.md":
        "eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37",
    "reports/g02/screening/tiered_surface/SCOPE_MANIFEST.md":
        "066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49",
    "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json":
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    "reports/g02/grouped_3fold_audit20_v1.json":
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json":
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    "reports/g02/screening/tiered_surface/kaggle_budget.json":
        "2e02bc4f0032b1125f42604bd4021a24fd70bb45b786c82e89ad17290bf31fc6",
    "reports/g02/screening/tiered_surface/tier_census.json":
        "4c7b6735cb2b03abdae3e1780e7b207a6e14ea237e199126ca47e7a9de944aa7",
    "reports/g02/screening/tiered_surface/wave3_cpu_barrier_decisions.json":
        "202cc2a50efbe25628f6fc9880b58bec9ab07412fb4b760789a9f34e1b71bec9",
}

LANES = {
    "codex/sa-nodetopo-gate-wave3": (
        "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gate/",
        "b5dfc1578ba5ef603439a911cd1bd43114497e09",
    ),
    "codex/sa-dinov2-wave3": (
        "reports/g02/screening/tiered_surface/branches/sa_dinov2/",
        "3f0863d3094d7c747890e4f8ea2ae8859c7c2c3a",
    ),
    "codex/sa-reverse-wave3": (
        "reports/g02/screening/tiered_surface/branches/sa_reverse/",
        "077449d0521c800ee04eb2f6f7684dfa93eb42f2",
    ),
    "codex/sa-contracts-wave3": (
        "reports/g02/screening/tiered_surface/branches/sa_contracts/",
        "15939e1",
    ),
    "codex/sa-division-cont-wave3": (
        "reports/g02/screening/tiered_surface/branches/sa_division_cont/",
        "3be58ef",
    ),
}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks: list[dict[str, object]] = []

def check(name: str, passed: bool, detail: object = None) -> None:
    row: dict[str, object] = {"check": name, "pass": bool(passed)}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)

for rel, expected in EXPECTED_FILES.items():
    actual = digest(ROOT / rel)
    check(f"pin:{rel}", actual == expected, actual)

decisions = json.loads((SURFACE / "wave3_cpu_barrier_decisions.json").read_text())
census = json.loads((SURFACE / "tier_census.json").read_text())
live = (ROOT / "reports/g02/screening/LIVE_STATE.md").read_text()
ledger = (ROOT / "reports/g02/screening/LEDGER.md").read_text()

check("five_lanes", len(decisions["lanes"]) == 5)
check("accepted_moves_three", decisions["governance"]["accepted_tier_moves"] == 3)
check("uncited_moves_zero", decisions["governance"]["uncited_tier_moves"] == 0)
check("forbidden_actions_zero", all(decisions["governance"][key] == 0 for key in [
    "dead_promotions", "gpu_actions", "kaggle_actions", "runpod_actions",
    "submission_actions", "plan_amendments", "cellect_actions",
]))
check("budget_debit_zero", decisions["governance"]["budget_debit_hours"] == 0.0)

ids = [node["id"] for node in census["nodes"]]
actual_rollup: dict[str, int] = {}
for node in census["nodes"]:
    actual_rollup[node["tier"]] = actual_rollup.get(node["tier"], 0) + 1
check("node_count_45_unique", len(ids) == 45 and len(set(ids)) == 45)
check("rollup_exact", actual_rollup == {"PARK_A": 15, "PARK_B": 19, "PARK_C": 10, "DEAD": 1}, actual_rollup)
check("stored_rollup_exact", census["rollup"] == {"PARK_A": 15, "PARK_B": 19, "PARK_C": 10, "DEAD": 1}, census["rollup"])

by_id = {node["id"]: node for node in census["nodes"]}
check("ae_p5_park_a", by_id["AE-P5"]["tier"] == "PARK_A")
check("ae_p7_park_a", by_id["AE-P7"]["tier"] == "PARK_A")
check("g2_park_c_exact_result", by_id["G2"]["tier"] == "PARK_C" and by_id["G2"]["evidence_sha256"] == "62953d270ae383c9bacdea9ab33a3f6980358a31a329b219447460abd5604864")
check("nodetopo_parent_preserved", by_id["NODETOPO-O0"]["tier"] == "PARK_A" and by_id["NODETOPO-O0"]["latest_instantiation"]["tier"] == "PARK_C")
check("noderestore_parent_preserved", by_id["NODERESTORE-O0"]["tier"] == "PARK_B")
check("r4_dependency_cleared", by_id["R4"]["tier"] == "PARK_B" and by_id["R4"]["blocked_on"] == [])
check("r5_staged_only", by_id["R5"]["tier"] == "PARK_B" and "BOUNDED_R5_ENDPOINT_FEATURE_MATERIALIZATION_AUTHORIZATION" in by_id["R5"]["blocked_on"])
check("reverse_parked", by_id["AE-P6"]["tier"] == "PARK_B" and by_id["R8"]["tier"] == "PARK_B")
check("division_fitting_locked", by_id["DIV-F1"]["tier"] == "PARK_A" and "ASSOCIATION_CONTINUITY_BOTH_FAMILIES" in by_id["DIV-F1"]["blocked_on"])

for branch, (prefix, tip) in LANES.items():
    subprocess.run(["git", "cat-file", "-e", f"{tip}^{{commit}}"], cwd=ROOT, check=True)
    names = subprocess.check_output(
        ["git", "diff", "--name-only", "a8c65b4", tip],
        cwd=ROOT, text=True,
    ).splitlines()
    check(f"scope:{branch}", bool(names) and all(name.startswith(prefix) for name in names), {"files": len(names), "prefix": prefix})

check("live_census_current", "LIVE 0 / PARK-A 15 / PARK-B 19 / PARK-C 10 / DEAD 1" in live and "4c7b6735" in live)
check("live_log_wave3", "CPU-only Wave-3 **ROOT BARRIER PASS**" in live)
check("ledger_wave3_rows", all(token in ledger for token in [
    "NODETOPO-GATE-W3-20260729",
    "DINOV2-R4-R5-W3-20260729",
    "REVERSE-SUPPLIER-W3-20260729",
    "NODE-CONTRACTS-W3-20260729",
    "DIVISION-CONTINUITY-W3-20260729",
    "CPU-WAVE3-ROOT-BARRIER-20260729",
]))

result = {
    "schema": "celltrack-alpha.wave3-cpu-barrier-validation.v1",
    "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
    "passed": sum(bool(item["pass"]) for item in checks),
    "total": len(checks),
    "checks": checks,
}
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)
