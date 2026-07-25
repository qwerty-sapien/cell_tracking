# Appearance direct-swap public-probe package

**State: READY-FOR-HUMAN-APPROVAL. No upload, launch, or submission performed.**

## Immutable package

The local package is `kaggle_kernel/bhap-submit`, targeting the private kernel
`onebitbrain/bhap-submit`.

| member | bytes | sha256 |
|---|---:|---|
| `bh45.py` | 274,611 | `eafac752d0b67ec35e18bf6a6f2d03ecc1f9d1d27e0813959df00880cfd77db5` |
| `bh45att.json` | 11,600 | `d22f7478f251c1c7272b587b1c240d2f30b3f2708a160a38b652443babe51ea6` |
| `bh45man.json` | 4,991 | `fd189d6a7956082f55ce1f09c53d51692c0c1d54d94189d52f6a018ca87f8cb5` |
| `bh45src.zip` | 160,912 | `70b5f7bb37499ab82c18986465672bf4291241f492748edbd01989d2c2e7bf8c` |
| `kernel-metadata.json` | 407 | `b371ee295c9b2b27270f7cea04a776a5e1f98bb7408301a430f8b19e22affc38` |

The aggregate submitted package is 452,521 bytes, below the 1,000,000-byte
source cap. The canonical manifest identity is
`779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b`.
Internet access is disabled. The package is private and uses the existing
attached support dataset.

The embedded runtime pins:

- appearance contract
  `cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7`;
- geometry gate
  `04a4cec214b5848363af08ab9a092f5ac8cc077b83f946ea81fc986d36e91ed1`;
- edge artifact
  `5c6377cfdfe8bbc75bcef52c602cf86c402f07451a7751047a0adba87e7e8034`;
- auxiliary checkpoint
  `1cff90475b819b5652310ccb73ba57e1dc20069885962a06a5e10a6060c04473`.

Any missing file, hash mismatch, unknown contract field, changed threshold,
changed local score/tuple, unset cap, or changed graph stage aborts execution.

## Package-path parity

The packaged runtime recomputed every geometry/NCC proposal from cached
detections, scorer features, and raw images. It did not reuse the frozen
proposal list. All 57 proposal sets matched the banked parent exactly.

The runtime replay then reproduced:

| result | value |
|---|---:|
| exact `c=0.25` control | 0.8731495935289003 |
| package-path candidate | **0.8815220699597315** |
| delta | **+0.008372476430831166** |
| candidate TP / FP / FN | **34,320 / 2,459 / 1,948** |
| runtime proposal parity | exact on 57/57 videos |
| candidate fragment-tree sha256 | `ce1483483dd14634d9b40633a5c751f31dc64396a30febcb9425cfec89efee85` |

The new runtime fragment tree is byte-identical to the banked candidate tree.
The runtime-parity report is
`reports/g02/selected_graph_appearance_runtime_parity_cv57_v1/report.json`
(`sha256 a02d3439354d22614a223d5effbd32f9b6f3f794dd0a8db08915b41c874180e8`).

The package builder and runtime fail-closed paths pass focused unit/integration
tests. The actual hidden-test `submission.csv` cannot be generated offline
because the package intentionally requires the mounted Kaggle test set and
exact T4x2 runtime. That launch remains behind the human gate.

## Runtime and quota estimate

The last exact sequential T4x2 production run completed in 322.4 seconds.
The full-CV57 CPU appearance replay took about 1,008 seconds for 57 videos; a
four-test-video proportional allowance is about 71 seconds before two-worker
parallelism. Allowing overhead, preregister **7 minutes wall time** and at most
**0.25 Kaggle GPU-hours** for one T4x2 generation run. RunPod cost is `$0`.

This action does not itself spend a competition submission. If the generated
output passes the package manifest, row-count, exception, and observation
checks, submitting it to the competition consumes exactly **one** of the daily
five screening submissions. The two final-selection slots remain reserved.

## Proposed gated action

After explicit approval:

1. push exactly the local `bhap-submit` package to the private Kaggle kernel;
2. launch one exact sequential T4x2 run, capped at 7 minutes expected /
   0.25 GPU-hours budgeted;
3. inspect `run.json`, `bhdiag.json`, row counts, hashes, and exceptions;
4. present the generated `submission.csv` identity before—or as part of an
   explicit combined authorization for—one competition submission;
5. record the displayed public score as a rounded interval and immediately
   update P0B before selecting any further probe.

No RunPod action is proposed.
