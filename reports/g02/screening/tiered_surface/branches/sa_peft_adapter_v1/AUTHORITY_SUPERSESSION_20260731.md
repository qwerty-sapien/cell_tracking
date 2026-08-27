# PEFT terminal-authority supersession

**Current authority:** the 12:27 `TERMINAL_REPORT.md` and its validated run-v1
receipts.  
**Superseded snapshot:** the 12:04
`TERMINAL_AUTHORITY_RECONCILIATION_20260731.json` conclusion that the node was
blocked and no amendment existed.

The earlier reconciliation was correct when written. It ceased to control this
one node when the owner recorded `PREREGISTRATION_AMENDMENT_A.json` at 12:12,
explicitly overriding the terminal campaign closure for `sa_peft_adapter_v1`
only and choosing the five frozen hash folds. This happened before the 12:19
serial push and 12:26 outcome. The run used those five folds: held action counts
`244,465/173,496/311,859/238,772/376,905` and beneficial counts
`306/99/167/195/351`.

Two separate artifacts carry different governance roles and must not be
conflated:

- Amendment A (`4c4890ee…e63e01`) is the owner authority that superseded the
  earlier closeout for this node.
- `RUNTIME_MARGIN_AMENDMENT_20260731.json` (`a5b72dbc…1093a`) is the executed
  five-fold/runtime design. Its remote `peft_assets/amendment.json` copy is
  byte-identical. It changes no scientific kill threshold.

The `recorded_utc` value inside Amendment A has a malformed `Z` suffix: the
artifact says `12:12Z`, while its filesystem mtime and the complete surrounding
chronology establish that 12:12 is the campaign's Asia/Singapore wall time
(`04:12Z`). The subsequent machine receipts are unambiguous: prelaunch quota at
`04:17Z`, push at `04:19Z`, and terminal outcome at `04:26Z`. This timestamp
labeling defect is disclosed; it does not reverse the pre-outcome ordering.

The sole v1 is `FAIL_PARK_C`, passed independent replay 15/15, and created no
submission. Its authoritative quota movement is `5.02/24.98` to `5.06/24.94 h`
used/remaining, a `0.04 T4-h` debit. A fresh read-only quota query at
`2026-07-31T04:55:20Z` still reports `5.06 h` used and `24.94 h` remaining.

Machine-readable seal: `AUTHORITY_SUPERSESSION_20260731.json`.
