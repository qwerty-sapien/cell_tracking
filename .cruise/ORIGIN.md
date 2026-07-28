CONTEXT
Three corrections to the CELLECT diagnosis, all provable from banked artifacts — no new API calls:

1. AUTH: The 403 on ListDatasetFiles is NOT rules/entitlement. onebitbrain/g02-cellect-r7-
   published-assets-v1 is the user's OWN private dataset; rules were accepted ~3 weeks ago and
   the CLI has worked all along (see Q-APNCC-PUBLIC submission in LEDGER). It is an agent-shell
   token/scope problem. STANDING RULE: all authenticated CLI work (Kaggle list/push/submit)
   goes through computer-use cmux where the real session lives — not the agent's own token.
   Re-run `kaggle datasets files ...` via cmux.

2. FILENAME BUG (real, fix now): locate_one() uses Path("/kaggle/input").rglob(pattern), which
   treats the argument as a GLOB, not a literal. Fix: resolve the weight by CONTENT — glob "*.pth"
   (and *.bin/*.tar members), select the file whose SHA-256 == f3e8e7303976dc5fb6e52ade12d1ff795
   3d2a28e30c06274b9f309562cd89ce5, fail closed if none. Make it step 0 of the kernel (before dep
   install) so a miss costs ~1s, not the observed ~650s.

3. BYTE MISMATCH (the deeper defect, previously missed): manifest stages 7 files / 19,467,606 B
   with the .pth at 10,197,221 B. Remote observed total is 18,508,409 B; non-checkpoint files are
   9,270,385 B, implying a remote checkpoint of 9,238,024 B — 959,197 B SMALLER than authenticated.
   Kaggle's dataset pipeline is transforming the file, so its SHA can never match. FIX: upload the
   checkpoint inside an UNCOMPRESSED .tar (or as .bin), extract in-kernel, THEN verify SHA-256
   against the pin. Also rename to safe ASCII (cellect_uext_x3rd_149_0_4p6540.pth) inside the tar.
   Verify via cmux that the stored object round-trips to the exact SHA BEFORE any launch.

TASK — do 1-3 above, no kernel launch until cmux confirms a byte-exact round-trip.

IN PARALLEL (CPU, no Kaggle, no budget): run the human-held NODETOPO 44b6 graph gate on its 50
N1-valid restoration actions. Rationale: unlike delete/swap (393 positives in 4.02M — the sparse
needle regime that has killed every selector), NODETOPO is a small dense action set that avoids
that failure mode entirely. Preregistered gate unchanged: adjusted >= +0.005, raw/node >= 0, 44b6
first. Report at the barrier.

HARD STOPS unchanged: submission, DEAD, plan amendment, >=5h GPU, RunPod.
