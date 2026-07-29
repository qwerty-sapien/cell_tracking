# DINOv2 R5 Kaggle GPU v1 — infrastructure INVALID, no verdict

The single authorized GPU run was pushed exactly once. Its first action
completed a full recursive `/kaggle/input` inventory: 24,968 files,
87,725,717,409 bytes, and per-file SHA-256 in 1,242.975765 seconds. Step 0 then
resolved all 77 frozen request files, the pinned checkpoint, official source
tar, three wheels, and the competition root by content hash. It used no
external bundle manifest and reused zero quarantined MPS rows.

The run failed immediately after step-0 PASS when pip rejected
`wheel-2a3175…f9d.whl` as an invalid wheel filename. The dataset had preserved
wheel bytes but replaced the canonical distribution filename with a
content-hash basename. No dependency installation completed, source was not
extracted, no worker was spawned, no model was loaded, and no feature or class
manifest was written. Sealed B0, A1, and A0 were never opened.

This is an infrastructure INVALID, not a separation failure. Delete and
occupied swap are both NOT EVALUATED against the frozen gate and run_v6
requirements. R5 remains PARK-B; there is no PARK-C move or graph
preregistration unlock. The conservative terminal debit is 0.3989 GPU-h,
leaving 26.6976 h. The one-run authorization is closed and no retry is
authorized.
