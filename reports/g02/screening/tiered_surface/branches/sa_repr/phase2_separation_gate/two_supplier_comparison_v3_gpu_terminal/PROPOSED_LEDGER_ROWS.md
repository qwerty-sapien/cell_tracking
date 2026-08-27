# Proposed ledger rows

| Date | Node | Evidence | Result | State effect |
|---|---|---|---|---|
| 2026-07-30 | `DINOV2-R5-SEPARATION-GPU-V1-V2-LAUNCH-RACE` | Reconciliation `ca91b2c0…9ea0`; latest-v2 terminal `06e15b5c…8287`; step0 `4c59710f…557f`; inventory `81438a82…41fc`; log `9f579493…d522` | Concurrent race created byte-identical v1 and v2 under one-run authorization. Latest v2 passed the full 24,968-file / 87.726 GB inventory and all content resolution; pip rejected the content-hash wheel basename before worker/model/features/B0/A1/A0. Full authenticated quota delta 0.74 GPU-h is debited. | Infrastructure INVALID, no verdict; serial authorization violation banked; R5 stays PARK-B; no PARK-C or graph unlock. |
| 2026-07-30 | `DINOV2-R5-CPU-PROMOTED-V2` | Terminal receipt `e9076800…ef90`; remote terminal `0f87b986…2ea3`; step0 `59a27587…8e38e` | Owner-authenticated status is ERROR. Step 0 passed in 0.053483 s for all 77 requests / 1,391,458 actions, then the authenticated projection was 38.662 h and failed closed. No result manifest or scientific outcome. | 0.1163353 CPU-h / 0 GPU-h; R5 remains PARK-B; no PARK-C or graph unlock. |
| 2026-07-30 | `CELLECT-R7-BLEED-STOP-V1` | Terminal `3c7f509d…86f8`; validation `7ad094f6…b03d`; kernel `7637a8e3…d5c9` | Embedded 77 request records and all required content pins; opaque 83-file expanded mount passes with zero external manifests and fails closed for missing/duplicate content. No remote launch or debit. | Local transport PASS only; prior scientific status remains infrastructure INVALID; R7 stays PARK-B; no graph unlock. |
| 2026-07-30 | `TWO-SUPPLIER-SEPARATION-CROSS-READ-V3` | Frozen gate `9fb030bd…3310e`; sealed B0 `a9ce8c0b…b4ff2`; run_v6 requirements `6e62a29c…6c71b`; comparison `TWO_SUPPLIER_COMPARISON.json` | No complete feature/class manifest for either supplier. All four cells are NOT EVALUATED against frozen 0.65/0.15 plus lifts/controls and run_v6 requirements. DELETE remains primary at ~4.41× swap's compiler-valid oracle value. | Zero PASS, FAIL, PARK-C, or graph unlock; halt for human review. |

Budget note: the repo debit subtotal is 3.6435 GPU-h used / 26.3565 h
remaining, while authorization capacity uses the lower owner-authenticated
postrun balance of 25.47 GPU-h. The launch-race row has been applied to the root
ledger; the other rows remain provenance proposals.
