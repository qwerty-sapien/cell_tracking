# DINOv2 R5 Kaggle GPU — infrastructure INVALID, launch race reconciled

The one-run serial authorization was violated by a concurrent push race. A
concurrent actor created authorized version 1 at 17:55:29Z; after the root
pre-push absence check, the root launch call created the byte-identical version
2, whose owner-recorded last-run time is 17:57:50.703Z. Two remote runs were
therefore triggered under a one-run authorization. No retry is authorized.

The exact terminal output available through the owner API is the latest
version-2 run. Its first action completed a full recursive `/kaggle/input`
inventory: 24,968 files, 87,725,717,409 bytes, and per-file SHA-256 in
1,242.975765 seconds. Step 0 then resolved all 77 frozen request files, the
pinned checkpoint, official source tar, three wheels, and the competition root
by content hash. It used no external bundle manifest and reused zero
quarantined MPS rows.

The run failed immediately after step-0 PASS when pip rejected
`wheel-2a3175…f9d.whl` as an invalid wheel filename. The dataset had preserved
wheel bytes but replaced the canonical distribution filename with a
content-hash basename. No dependency installation completed, source was not
extracted, no worker was spawned, no model was loaded, and no feature or class
manifest was written. Sealed B0, A1, and A0 were never opened.

The byte-identical version-1 package cannot reach model or feature work either:
its first pip command names the same invalid wheel. Available owner APIs do not
address private output/status by version, so its individual terminal runtime is
not asserted. The latest status is ERROR and no external authorization remains
active.

This is an infrastructure INVALID, not a separation failure. Delete and
occupied swap are both NOT EVALUATED against the frozen gate and run_v6
requirements. R5 remains PARK-B; there is no PARK-C move or graph
preregistration unlock.

Accounting uses the full owner-authenticated pre/post quota delta, not the
individually timed v2 value: 3.79→4.53 GPU-h used, hence **0.74 GPU-h**
debited for the launch race. The repo subtotal is 3.6435 h used / 26.3565 h
remaining; the capacity gate uses the lower live owner balance,
**25.47 GPU-h remaining**. On the user's supplied 27.4239 h basis, the
postrun balance is 26.6839 h. Reconciliation:
`LAUNCH_RACE_RECONCILIATION.json`.
