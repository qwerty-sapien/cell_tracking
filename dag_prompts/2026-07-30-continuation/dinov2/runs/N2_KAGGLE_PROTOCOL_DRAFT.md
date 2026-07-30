# N2 — DPRC-O0 fresh-supplier protocol (draft only)

Date: 2026-07-30 (Asia/Singapore)
Status: **preregistered design; no external action is authorized**

## Fresh-supplier Kaggle protocol

1. **Identity.** DPRC-O0 uses an official DINOv2 ViT-S/14-with-registers
   release, full foreground patch-grid cycle features, and a fresh source/input
   manifest. It categorically excludes the R5 non-register checkpoint,
   b938bf1b…0cd9, its source archive, 77 request files, package, inputs, all
   feature rows, and quarantined partial outputs.
2. **Step 0, before pip/NumPy/Torch/Zarr/model/data decode.** Resolve one exact
   owner-approved manifest path only; verify canonical source-tree and weight
   hashes, license receipt, wheel filenames and hashes, explicit raw-data roots,
   fresh request/materialization manifest, positive counts, and absence of every
   R5 identifier. Missing, duplicate, unexpected, or R5-derived content aborts
   within five seconds. No recursive input scan is permitted.
3. **Bounded scientific scope.** First collect only a runtime calibration on two
   predeclared non-score videos; its feature output is discarded and may not be
   inspected for association quality. It may continue only when its conservative
   extrapolation, including token reduction and scoring, is at most 4.0 GPU-h.
   A successful run then covers the complete frozen 44b6 delete and
   occupied-swap populations only—never a score-bearing video subsample, because
   the campaign's subsample-fidelity barrier prohibits one. It has a 5.0 GPU-h
   hard kill; 6bba, CV57, graph evaluation, and submissions remain closed.
4. **Feature contract.** Each endpoint is reduced immediately from its complete
   patch grid to a fixed 4-by-4 foreground spatial descriptor; pair evaluation
   emits only C(i,j), foreground-cycle coverage, and the conflict-pair margin.
   No learned Delta-DINO, fine-tuning, no-link head, graph-cost tuning, or
   GT-dependent filter is allowed. The frozen fit-only evaluator is the sole
   later consumer, and only under its existing gates.
5. **Resource and retry rules.** Record /proc/meminfo swap at process start and
   throughout; any positive growth above 0 MiB kills the run. A step-0 failure,
   absent channel/input contract, calibration projection above 4.0 GPU-h, wall
   time above 5.0 GPU-h, swap growth, missing manifest, or incomplete class
   manifest terminalizes this exact DPRC route with **no retry**. Partial
   features are quarantined and cannot be scored or combined.

## Why this is a new route

The protocol needs a separately approved release, hashes, input construction, and
manifest. It neither repairs an R5 failure nor makes a modified R5 launch; its
new information is mask-restricted spatial correspondence rather than R5's global
XY endpoint representation. It specifies no actual source download, asset
creation, kernel, or run.
