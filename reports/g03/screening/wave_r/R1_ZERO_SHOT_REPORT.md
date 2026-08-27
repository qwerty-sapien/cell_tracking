# G-03 Wave R1 zero-shot measurement

Status: **COMPLETE — Branch B**  
Scientific configurations evaluated: **1**  
Training/GPU/Kaggle/CV57/submission: **0 steps / 0 GPU-h / none / none / none**

## Result

The authentic frozen Trackastra checkpoint plus the deterministic untrained
crop adapter scored **0.7637338804169099** on the exact 36-video 44b6 held-fit
objective. The authoritative allocation-policy floor is the exact decimal
`0.868299238910600`; therefore this candidate is **Branch B**. It is
`-0.1045653584936901` below that floor and `-0.1177881895430901` below the
stated best-known score `0.881522069960`. The best known score is unchanged.

| component | result |
|---|---:|
| combined / adjusted edge Jaccard | 0.7637338804169099 |
| raw edge Jaccard | 0.7584738824299984 |
| node recall | 0.9723341840137287 |
| edge TP / FP / FN | 9264 / 1200 / 1750 |
| division TP / FP / FN | 0 / 0 / 9 |
| division Jaccard | 0.0 |

Reference points are explicit: all policy deltas use best known
`0.881522069960`; APNCC 44b6 `0.866534526036` is descriptive only. The R1
score is `-0.1028006456190901` versus APNCC.

## Frozen estimator and retained bank

The sole adapter has 764 frozen parameters and zero training steps:
`AvgPool3d(4,8,8) -> Conv3d(3,8,k3,p1) -> exact GELU -> global average ->
Linear(8,12) -> LayerNorm(no affine)`, seed `20260801`. Inputs are raw
three-frame `(3,16,32,32)` crops with temporal endpoint clamp, replicate
spatial padding, nearest-integer centers, and per-crop global mean/RMS
normalization followed by clamp `[-5,5]`.

Authentic Trackastra ran with its frozen CTC config (`feat_dim=12`, window 4,
max 1024 tokens). The compiler scored adjacent targets within 12 micrometres,
cap 16; graph decode then used radius 8, cap 8, threshold 0.35, global
descending one-in/one-out greedy, and divisions off.

Inference covered 1,283,326 detections in 11,690 model calls and retained
6,268,942 candidate-edge probabilities. The 36 compressed per-video banks are
47,162,409 bytes total and contain `node_id`, `frame`, `coords_zyx`,
`edge_source`, `edge_target`, `probability`, and `distance_um`. Aggregate bank
SHA is `886fbd3ab5b4a46be448005f4e98153227021c54f4bee7b09afcfff40c9cb4bd`;
aggregate compiled-graph SHA is
`e7a2aaf1bf565acfc2948ca79da17f742efa2fd51be8b7609ff0e3b5e239dbd6`.
R4 can decode divisions from these banks without rerunning Trackastra.

## Integrity account

The contract, implementation, adapter, raw archive, caches, authentic model,
Trackastra source tree, and scorer sources were hash-authenticated before
inference. All 36 graphs and banks were sealed before the first target access.
Every video had all-nonzero `wrfeat`; all 12 channels showed image-conditioned
variance in every video.

The first target-access attempt completed metric calculation but its receipt
writer rejected undefined per-video division Jaccards under `allow_nan=false`;
no candidate value or artifact was exposed. Incident SHA
`79ebb0514988ae1bfe2499d2d25e6c5c9ab07fb431fbb1b715e1616c98e1fd81`
was sealed before one deterministic technical recovery pass. Recovery used the
identical graph and scorer hashes and changed only non-finite per-video
diagnostics to the explicit string `"NaN"`. Terminal accounting is target
access attempts 2, scientific configurations 1, exposed candidate scores 1.

## Principal hashes

- Contract: `aefe438e6d8966b25f3665a1f009d108ae15ef0737581c1aa07a57f027050204`
- Exact decision amendment: `2ed11e6e8ba50db2ad7ccab6343406b06a38aa9b46efab7fc300668627eb738c`
- Frozen implementation: `90b2cce6c7f38d280b6ac460f99712d07590d534e9cab94f39e397b25cfedf47`
- Adapter file/state: `957446610bc9ee681ec5d8769f7d26e81750caadc4428aff4214f37ddafd3921` /
  `5abad1640a1a289f7a71aa618c520476cac4edae02b91806b87fc3212b4b9f8f`
- Preflight: `c7e263592c14eff0270ee8acf35efccf0b2db3dff7b035faf74447da790f9950`
- Inference seal: `c9cf2f7c8a63fecb4d5c822b5f93538f8df609c6577c68245d9f25d4d4c67ea1`
- Final result: `55300582b831454ab6ba2b538cba7271d0c99bb230c2db24a2c049477f514372`
- Validation receipt: `b25665f2724d8437ed862a574f30d59afdc40de21708585f3738e759c04ee16c`
- Retained recovery helper (post-hoc provenance):
  `b1ca85fce69c7e7cd20a8ea773f2b55dba3da0d7673a13cb9a1fba85a19b82c6`
- Retained exact finalizer (post-hoc provenance):
  `0b1afb0ffdb027412898f65bf017bb0d5c8b6090aa45a807dcaa614dd0520654`

The machine-readable result is `R1_ZERO_SHOT_RESULT.json`; the complete path
and hash chain is in `R1_VALIDATION_RECEIPT.json`.

## Reproduction command sequence

```text
python scripts/g03_r1_zero_shot.py make-adapter --output reports/g03/screening/wave_r/R1_artifacts/adapter_seed20260801.pt --seed 20260801
python scripts/g03_r1_zero_shot.py make-member-map --raw-manifest /Users/agent/Desktop/cell_tracking/kaggle_kernel/trackastra-h1-feature-cache-prep/raw_fit105_directory_manifest.json --h1-report /Users/agent/Desktop/cell_tracking/reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json --output reports/g03/screening/wave_r/R1_artifacts/MEMBER_MAP.json
python scripts/g03_r1_zero_shot.py preflight --contract reports/g03/screening/wave_r/R1_ZERO_SHOT_CONTRACT.json --contract-sha256 aefe438e6d8966b25f3665a1f009d108ae15ef0737581c1aa07a57f027050204 --output reports/g03/screening/wave_r/R1_artifacts/PREFLIGHT.json
python scripts/g03_r1_zero_shot.py infer --contract reports/g03/screening/wave_r/R1_ZERO_SHOT_CONTRACT.json --contract-sha256 aefe438e6d8966b25f3665a1f009d108ae15ef0737581c1aa07a57f027050204 --preflight reports/g03/screening/wave_r/R1_artifacts/PREFLIGHT.json --preflight-sha256 c7e263592c14eff0270ee8acf35efccf0b2db3dff7b035faf74447da790f9950 --output reports/g03/screening/wave_r/R1_artifacts/inference
```

The target-access recovery and exact-finalizer helper bytes were preserved in the
isolated worktree and are now promoted with the hashes above. They were **not** bound
by the initial pre-execution contract or root promotion receipt, so this retention does
not retroactively make them pre-pinned. The verified raw/final receipts reproduce the
same candidate metrics and the post-hoc reproducibility amendment records that scope.
The helpers must not be rerun; R1 is terminal.
