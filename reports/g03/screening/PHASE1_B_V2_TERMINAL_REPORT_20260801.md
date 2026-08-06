# G-03 Phase 1 B-v2 terminal report

Date: 2026-08-01  
Verdict: **PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED / PHASE 1 COMPLETE**

## Terminal identity and cost

- Kernel: `onebitbrain/g03-phase1-resume-b-v2`, exact version `1` (`0→1`), pushed exactly once.
- Frozen code SHA-256: `9bf7e81255ef7d4a2aef48d4644f3e252101b67745eb3e37f83c521d4446f6f2`.
- Frozen metadata SHA-256: `9f83bdfa1fc5a7c42daa4efa604f7404b002907cb083691b3c345908b203bed5`.
- Push began `2026-08-01T02:37:37.737217Z`; authenticated terminal `COMPLETE` was observed at `2026-08-01T03:38:40.188402Z`.
- Post-terminal quota: `1.01/28.99 T4-h` used/remaining, refresh `2026-08-08T00:00:00Z`.
- Owner ceiling for B was 8 T4-h where justified. The exact run needed 1.01 T4-h; the unused 6.99 h is preserved because a repeat would be observation-identical.

## Independent terminal validation

The downloaded output passed `scripts/validate_g03_phase1_b_prelaunch.py terminal`:

| invariant | observed |
|---|---|
| status | `PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED` |
| resumed global step | `1,953,711` |
| handoff tree | `73a6cc566ba91894cd1da0ace59581ffa89ba63cd324b829c6e2ed860b5b56ac` |
| final uninterrupted-reference digest | `816acee4efb6d0c757308bf27d9e0b912e30caee7dba0bfbdec3c80d3c80387b` |
| swap growth | `0 bytes` |
| structural graph scope | fit105/44b6, 36 identities |
| structural deltas | adjusted edge Jaccard `0.0`; raw edge Jaccard `0.0`; node recall `0.0` |
| B receipt SHA-256 | `8d6781045b9a6780de2929ecb94479b9ebff07e301a101f20bf21456a8db3e69` |
| graph receipt SHA-256 | `8863ea78a7bdb30e148c2864667b30f56aa61175eafb8bca786b50d55752b3cd` |
| downloaded log SHA-256 | `186f5de6ef2853d3d621cd7c7a9947ca0da3aa64d0856b557b0ae6c8546a64ff` |

Every candidate/control edge hash matched. The output explicitly records
`official_frozen_graph_objective_evaluated:false` and
`selection_authorized:false`. The `0/0/0` values are a structural plumbing
receipt, not scientific performance and not a checkpoint-selection read.

## Decision

Phase 1's cross-kernel checkpoint/resume mechanism is proven for this frozen
protocol. No B-v3, repeat, CV57 measurement, competition submission, or final
slot action is justified. Phase-2 preparation may begin, but scientific model
training remains closed until the raw-crop, adapter, cost, output, and graph
checkpoint contracts are frozen and independently guarded.

## Evidence

- `PHASE1_B_V2_TERMINAL_VALIDATION_20260801.json`
- `PHASE1_HANDOFF_DATASET_RECEIPT_20260801.json`
- `PHASE1_B_V2_REMOTE_ROUNDTRIP_GUARD_20260801.json`
- `phase1_b_v2_ops/kernel_push_2026-08-01T023737737217Z.json`
- `phase1_b_v2_ops/kernel_status_2026-08-01T033838464770Z.json`
- `phase1_b_v2_ops/quota_2026-08-01T033924112772Z.json`

