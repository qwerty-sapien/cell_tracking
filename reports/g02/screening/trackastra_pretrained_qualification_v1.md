# Frozen Trackastra qualification and zero-wrfeat signal gate

Date: 2026-07-24  
Decision: **BANK qualification; authorize grouped-OOF frozen-head design only**

## Result

The official Trackastra CTC checkpoint is provenance-qualified and locally
feasible. Its frozen zero-wrfeat association margin also passed the
preregistered fit-only ranking-signal gate:

| score | proposals | positives / negatives | ROC AUC | average precision |
|---|---:|---:|---:|---:|
| frozen Trackastra challenger-minus-incumbent | 497 | 411 / 86 | **0.918491** | **0.980347** |
| pair-NCC challenger-minus-incumbent | 497 | 411 / 86 | 0.841227 | 0.965034 |
| delta |  |  | **+0.077265** | +0.015313 |

This was frozen CPU inference over five fit-train `6bba` videos selected before
scoring at graph-density ranks 1, 17, 34, 51 and 67 of the 67 videos with
labeled support. It did not fit a parameter, select a threshold, open
validation12/CV57, construct a graph, or touch Kaggle/RunPod.

The result is mechanism evidence, not a projected competition delta. Per-video
behavior is heterogeneous: Trackastra was weaker than the already-perfect
pair-NCC ranking on two sparse videos, approximately tied on the middle video,
and stronger on the two densest videos. The aggregate gain therefore requires
grouped cross-video confirmation and exact graph scoring; it cannot justify
direct CV57 evaluation.

## Provenance and interface

The fail-closed asset audit reproduced:

- official checkpoint archive SHA-256
  `2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e`;
- checkpoint member SHA-256
  `24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f`;
- pinned training source commit
  `fb265888c8fce2ac16a362ebf182ccd5ea48e936`;
- BSD-3-Clause source license;
- 27,456,880 parameters and 109,827,520 tensor bytes;
- a 3-D, width-512, 6-encoder/6-decoder, four-head, four-frame,
  quiet-parental-softmax configuration with a 1,024-token training cap.

An exact-source CPU feasibility check completed finite forward passes from 128
through 1,024 tokens; the 1,024-token call took about 0.50 seconds with about
0.52 GB maximum RSS. The sealed signal run itself took 190.59 seconds, made
457 inference calls, and respected the 1,024-token cap.

The unresolved interface mismatch remains important. The released checkpoint
was trained with twelve `wrfeat` channels derived from instance masks
(`equivalent_diameter_area`, `intensity_mean`, inertia-tensor components and
border distance). This competition pipeline currently supplies point
detections rather than instance masks. The signal gate passed twelve
zero-filled channels, so it proves useful pretrained spatial/temporal
association behavior under a deterministic missing-feature input; it does not
prove compatibility with the checkpoint's intended morphology features.

No direct evidence of overlap between the competition's zebrafish videos and
the checkpoint's declared CTC/btrack/synthetic training list was found. This is
an evidence statement, not a content-hash proof of non-overlap.

## Next gate

Design one new `6bba` frozen-head identity with:

1. the released body frozen and hash-pinned;
2. Trackastra challenger/incumbent margins generated for the complete deployed
   proposal pool, not only labeled examples;
3. five grouped outer folds with disjoint fit/calibration/held videos;
4. proposal prerequisites of at least 0.95 aggregate precision and 0.93 in
   every fold;
5. exact held-fit graph reconstruction against APNCC, requiring adjusted edge
   Jaccard `>= +0.005`, nonnegative raw edge-Jaccard delta, nonnegative node
   recall delta, and observation-distinct fragments.

Fail or infeasible full-pool coverage parks this released zero-wrfeat route
before validation12/CV57. Passing authorizes the next local confirmation only;
LoRA, Kaggle GPU, RunPod, public submission and final selection remain separate
gates.

## Artifacts

- Asset report:
  `reports/g02/trackastra_pretrained_asset_audit_v1.json`
  (`2f809076…0132`)
- Locked signal contract:
  `reports/g02/trackastra_zero_wrfeat_signal_contract_v1.json`
  (`7404891a…c76d`)
- Signal report:
  `reports/g02/trackastra_zero_wrfeat_signal_v1/report.json`
  (`da3e68a2…33c4`)
- Official code: <https://github.com/weigertlab/trackastra>
- Official model registry:
  <https://github.com/weigertlab/trackastra-models>
