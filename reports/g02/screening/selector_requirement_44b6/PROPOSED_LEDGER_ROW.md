# Proposed ledger row — root review only

| ID | Status | Evidence | Finding | Consequence |
|---|---|---|---|---|
| `SELECTOR-REQUIREMENT-44B6-DIAGNOSTIC` | `INVALID_RESOURCE_NO_VERDICT` (diagnostic; no tier move) | `RESOURCE_BARRIER.json`, `guard_attempt1_resource.json`, `guard.json` | Two fresh zero-swap preflights passed, but the original and one streamed-lifetime attempt grew swap by 16.31 MiB and 1,567.69 MiB respectively; neither emitted a result file. | No dose/noise/requirement or gate-sufficiency verdict; frozen AUC 0.65 / retention 0.15 thresholds unchanged; any retry requires human review. |
