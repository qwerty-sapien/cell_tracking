# OCC-SLICED-W1-v1 launch receipt

Date: 2026-08-09 SGT (`2026-08-08T20:22:19.714534Z` push start)  
Kernel: `onebitbrain/g02-occ-sliced-w1-v1`  
Remote version: **1**  
Disposition: **VERSION 1 TERMINAL KFG-09 FAILURE / SCIENTIFIC UNMEASURED**

## Scientific boundary

This is a CPU-only fit-side observation supplier over the frozen 45,961
occupied-swap actions.  It writes fourteen sliced-Wasserstein A1 features and a
deterministic within-target-frame A0 control.  It cannot fit a selector, read
oracle labels or scores, run the graph scorer, access CV57, or write a
competition artifact.

The recursive path terminated as
`N0=Y -> N1=Y -> N2=N(NUMERICAL) -> N2B=Y -> N3=Y -> K1=Y`.  N2 failed its
frozen numerical gate because the entropic Sinkhorn pilot's p99 marginal
residual was `0.0027969605 > 0.001`.  The preregistered exact sliced-W1 fallback
then passed coverage, finiteness, byte-determinism, and runtime gates, with a
conservative full-population projection of `0.8223577614 CPU-h`.

## Package and checks

- scientific code SHA-256:
  `c9f06897641f8658e3c5c11908badd78f56745787df6f2694d6fb2be4342887d`;
- metadata SHA-256:
  `b8f26575447e12fff65ea3fb4b6f91d8b896ef86360d06c9487912bbcf3df2f8`;
- package-manifest SHA-256 before push:
  `a911a17a47c4f2468cc8f6600ef20a5303fc97d6b621fa8b986dcff267ba74ee`;
- inline prelaunch receipt SHA-256:
  `f6fcd44db13a659a3cea7321af410ebcb370539296f6e8ba77399f53edc4d541`;
- eleven focused kernel parity/safety checks and seventeen shared prelaunch
  failure-regression tests passed;
- credential, forbidden-output, label, scorer, and broad-discovery scans passed.

## CMUX result

The fresh-slug guard established that the kernel did not exist, and one CMUX
push call returned code zero with no semantic failure and the authenticated
message `Kernel version 1 successfully pushed.`  No retry occurred.

- fresh-slug status receipt SHA-256:
  `88f780475ad800e70de6b56dce8bf7e02b5a324f02adc349df5214fda839e55f`;
- push receipt SHA-256:
  `20906086463d058dad20fba6bca2e8584213067561abf1822481f1e05c5aafaf`;
- pre-push quota receipt SHA-256:
  `591870fd0c192205c0161e41a12af8076b78dcce99e8288eadc3ac71c1f3c6f4`;
- post-push quota receipt SHA-256:
  `ca20ee86b166c16cd4f955ae9807b0108d467367fae6a21bc39973980acc1f12`.

Authoritative quota was unchanged across the push at `6.88 h` used and
`23.12 h` remaining of `30 h`.  Scientific GPU projection is `0.0 h`; the CMUX
wrapper requires a positive administrative reservation, so `0.01 GPU-h`
remains reserved pending a later authorized terminal reconciliation.  The
three-direction portfolio therefore remains `0.0/15.0 GPU-h` scientifically.

Per the owner's instruction, this launch was not polled by Codex.  The owner
later supplied its terminal log: Step 0 inventoried 169 files / 187,738,899
bytes and stopped at `bundle manifest: expected one content match, found 0`
with `heavy_work_started=false`.  Those totals exactly identify the previously
authenticated DINOv2 v8 carrier under the mutable shared dataset slug, the
registered KFG-09 failure family.  No scientific output exists; no kernel
output was downloaded or locally scored, and no competition submission was
made.

The locally validated repair uses a dedicated immutable input slug and fresh
one-version kernel slug.  It is documented in
`V1_FAILURE_AND_REPAIR_PLAN.md` and is paused for explicit authority to upload
the private 9.88-MB input dataset.
