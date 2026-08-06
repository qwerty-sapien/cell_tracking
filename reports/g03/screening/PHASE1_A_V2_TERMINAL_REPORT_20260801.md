# G-03 Phase 1 transport repair — A v2 terminal

**Verdict: `PASS_A_HANDOFF_READY`; B v2 is the next permitted action.**

`onebitbrain/g03-phase1-resume-a-v2` version 1 is remotely `COMPLETE`. Its
downloaded `A_RECEIPT.json` reports exactly `3,600.000229 s` of synchronized T4
training, global step `1,953,711`, a complete reference endpoint at step
`3,907,422`, zero swap growth, and `selection_authorized:false`.

Independent local verification of the downloaded atomic handoff passes. The
declared tree contains exactly `MANIFEST.json`, `state/checkpoint.pt`, and
`state/contract.json`; the checkpoint is `3,174,314` bytes with SHA-256
`8a6c28675f929b613cdbaba69618aaf4cbd2db06f0f761e54a5f4a52100e134e`.
The tree SHA-256 is
`73a6cc566ba91894cd1da0ace59581ffa89ba63cd324b829c6e2ed860b5b56ac`,
and the manifest SHA-256 is
`af2c079470935661479efc5e2f5ee2302df512d3a7f57268db0cbcd0ee7dd7ef`.
No undeclared, missing, size-drifted, or hash-drifted handoff member exists.

The run crossed the weekly quota refresh, so its exact old-window debit cannot
be inferred from the post-refresh balance and is not reported as zero. The
current authenticated balance is `0.00/30.00 h` used/remaining with refresh
`2026-08-08T00:00:00`. The observed remote wall time was approximately 3.08 h,
inside the amended 4 T4-h cap.

This is infrastructure evidence only. It did not evaluate the frozen graph
objective, train the G-03 scientific model, access CV57, create a submission,
or use the remaining final slot. Under the locked v2 amendment, the exact
downloaded handoff may now be published as the immutable
`onebitbrain/g03-phase1-handoff-v2` dataset and consumed by the already-frozen B
v2 package. B must still prove bit-identical cross-kernel continuation before
Phase 2 can be considered.

