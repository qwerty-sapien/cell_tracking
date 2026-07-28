#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("G02_SOURCE_ROOT", Path(__file__).resolve().parents[6]))
HERE = Path(__file__).resolve().parent

EXPECTED = {
    "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json":
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    "reports/g02/grouped_3fold_audit20_v1.json":
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json":
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    "reports/g02/screening/tiered_surface/branches/sa_swap/result_44b6_postrepair_v1/report.json":
        "62953d270ae383c9bacdea9ab33a3f6980358a31a329b219447460abd5604864",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks = []
for rel, expected in EXPECTED.items():
    actual = digest(ROOT / rel)
    checks.append({"check": f"pin:{rel}", "pass": actual == expected, "actual": actual})

for name in [
    "PREREGISTRATION.json",
    "AE_P5_C0_ASSIGNMENT_CONTRACT.json",
    "AE_P7_PHASE_TEMPLATE_CONTRACT.json",
    "SUPPORT_IDENTITY_RESOLUTION.json",
    "state_fragment.json",
]:
    json.loads((HERE / name).read_text())
    checks.append({"check": f"json:{name}", "pass": True})

resolution = json.loads((HERE / "SUPPORT_IDENTITY_RESOLUTION.json").read_text())
checks.append({"check": "five_nodes_exact", "pass": set(resolution["nodes"]) == {"AE-P5", "AE-P7", "AE-P8", "AE-P9", "G2"}})
checks.append({"check": "p5_positive_banked_class", "pass": resolution["nodes"]["AE-P5"]["banked_action_class"]["family_44b6_score_delta"] > 0})
checks.append({"check": "p7_positive_banked_class", "pass": resolution["nodes"]["AE-P7"]["banked_action_class"]["family_44b6_score_delta"] > 0})
checks.append({"check": "g2_exact_scope", "pass": resolution["nodes"]["G2"]["instantiation"] == "SA-SWAP-W1-EXACT-REGRET"})
checks.append({"check": "g2_graph_fail", "pass": resolution["nodes"]["G2"]["graph_result"]["adjusted_delta"] < 0 and resolution["nodes"]["G2"]["graph_result"]["raw_delta"] < 0})
checks.append({"check": "hard_stops", "pass": all(v is False for v in resolution["governance"].values())})

result = {
    "schema": "celltrack-alpha.sa-contracts-validation.v1",
    "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
    "passed": sum(c["pass"] for c in checks),
    "total": len(checks),
    "checks": checks,
}
(HERE / "VALIDATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)
