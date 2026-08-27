# Frozen-correction closeout and training decision dossier

Date: 2026-07-31  
Status: **HUMAN DECISION REQUIRED — PLAN REMAINS LOCKED**

## Executive decision

The old `~478 T4-h` figure does **not** establish that every training experiment
is infeasible. It prices 300 epochs of the full organizer TemporalUNet3D, while
the organizer publishes a three-epoch checkpoint and the campaign needs a
useful falsifier, not full convergence. A workload-matched, useful-checkpoint
experiment can plausibly fit in `2–6 T4-h`.

I recommend opening one newly preregistered `RAW3D-NOLINK-MVE` node. I do not
recommend Trackastra fine-tuning: it is now cost-feasible, but it consumes the
same point/morphology information that has saturated. The raw-3D arm is the
only option that tests missing image information. This is not a launch
authorization or a score promise. The human-locked plan remains unchanged
until Option 1 is explicitly chosen.

## PEFT finding banked

The authenticated adapter trajectory was:

| epoch | DELETE AUC |
|---:|---:|
| 1 | 0.448881 |
| 2 | 0.515572 |
| 3 | 0.598619 |
| 4 | 0.622770 |
| 5 | 0.630757 |
| 6 | 0.633477 |
| 7 | 0.634590 |
| 8 | 0.634749 |

Epochs 6–8 gained only `0.001272`, so the adapter converged near AUC `0.635`,
far below the measured `0.85` requirement. Five current approaches occupy the
same weak band: incumbent `0.502`, C1 `0.502034`, PEFT `0.634749`, DINOv2
`0.670412`, and C2 assignment evidence `0.670235`. Across two backbones,
adapter/no-adapter, appearance, conflict-set, and assignment evidence, the
binding limitation is the representational information available for the
current action label—not adapter capacity or the choice between the tested
frozen backbones.

The selection result is more consequential than the late-epoch AUC. Nested
graph-only selection chose epochs `1/2/2/2/2`, yielding full-population AUC
`0.496495`, while rejecting epoch 8's descriptive `0.634749`. Adjusted/raw/node
graph deltas were only `+0.000133827/+0.000133160/0`, versus the binding
`+0.005` adjusted line. At effects of order `1e-4`, graph delta cannot resolve
within-trajectory checkpoint quality.

This extends the H0 proxy prohibition: graph delta remains the terminal
scientific endpoint, but it must not choose an epoch inside a trajectory at
this resolution. A future MVE must freeze its checkpoint horizon before
outcomes and read the held graph once.

Finally, the `4.8 h` PEFT projection resolved to `145.680 s`
(`0.04047 h`), an overestimate of `118.6×`. That is a mandatory recalibration
lesson, not a universal speed multiplier. The organizer's own rate estimate
was approximately right: `478/300 = 1.593 h/epoch`; seven epochs project to
`11.15 h`, close to the observed `~11.5 h`. The stale assumption was that
epoch 300 was required; the
[official organizer repository](https://github.com/royerlab/kaggle-cell-tracking-competition)
releases a short-horizon checkpoint. Future exclusions must use a
workload-matched timing calibration and a useful-checkpoint horizon.

## Feasibility re-derivation

### Trackastra fine-tune

Trackastra is a `27.46M`-parameter association transformer already pretrained
for cell tracking. Its paper learns pairwise associations from complete
spatiotemporal detection context and reports transfer across bacteria, cell
cultures, and fluorescent-particle datasets; none is this zebrafish
competition or its custom scorer. The official repository supports pretrained
models and training. [Trackastra paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf);
[official code](https://github.com/weigertlab/trackastra).

There are two relevant measured anchors, and they answer different questions.
The isolated Trackastra feature work over all 105 fit videos made 9,363 calls,
peaked at 1,024 tokens, and took `609.550925 s = 0.169320 T4-h`. The complete
cache kernel took `2.0348 h`, including the much larger Stage-A/data pipeline.
The earlier `5.538884 h` full-cache projection was therefore only `2.72×`
high—far more relevant to full-model work than PEFT's `118.6×` miss.

Using the conservative **complete-kernel** `2.0348 h` anchor, a `2.5–4.0×`
training/inference factor, two balanced quarter-coverage training passes plus
one full 44b6 evaluation, then a `1.25×` safety factor, gives:

`1.25 × [2 × 0.25 × 2.0348 × (2.5–4.0) + (36/105) × 2.0348]`

`= 4.05–5.96 T4-h`.

An isolated-compute cross-check gives a lower engineering range, not a second
promise. Applying `2.5–4.0×` plus 15% loader/checkpoint overhead to the measured
`0.169320 h` forward work gives `0.487–0.779 h` per fit105-equivalent update
pass. Three to five passes plus `0.25–0.75 h` of setup/graph work gives about
`1.71–4.64 T4-h` for one useful warm-start checkpoint. An honest five-fold,
one-seed selection run is roughly `7–20 h`; two seeds are roughly `14–40 h`.
Thus one checkpoint may fit under 5 h, but a selection-grade experiment does
not.

The official pinned training code defaults to 50,000 samples per epoch and 100
epochs; 50,000 / 9,363 is about 5.34 fit105-equivalent passes. That default is
not a sensible warm-start horizon here, but it prevents calling one timing
epoch “convergence.” Full-backpropagation memory at 1,024 tokens is also
unmeasured; frozen inference fitting on T4 does not prove the backward pass.

The `4.05–5.96 h` figure is therefore the safe end-to-end MVE envelope, not
convergence. Its campaign planning
envelope is local `0.878–0.896`: downside from fine-tuning regression, upside
bounded by the existing edge-only pool ceiling near `0.896`. This range is not
a literature result or confidence interval.

Overfitting risk is high. The held 44b6 surface is small, extremely imbalanced,
and has already supported multiple frozen supplier screens. Worse, Trackastra
adds capacity but no missing raw-volume information. Its MVE would use fixed
balanced quarter-passes spanning the 105 fit videos, a pre-fixed endpoint, and
one full 44b6 graph read. It is costed here as a comparator, not recommended.

### Compact 3D association/detection model

The proposed `RAW3D-NOLINK-MVE` is a warm-started, crop-based 3D model with at
most `5M` trainable parameters and explicit no-link/background examples. It
would consume raw spatiotemporal intensity—the information absent from the
saturated point/morphology suppliers—and fall back to banked APNCC outside
high-confidence edits.

The closest campaign implementation supplies a stronger lower-bound timing
anchor than cross-paper runtime transfer. The banked Siamese 3D crop linker has
30,057 parameters, paired `16×32×32` crops, and 5,893 saved examples (`4,008`
positive and `1,885` guaranteed negative; 590,520,183 bytes). Data preparation
took `331.805 s` on CPU, one 500-step fit took `120.696 s`, and five 500-step
outer fits took `583.554 s`. Linear extrapolation puts 2,000–5,000 steps at
`8–20 min` for one fit or `39–97 min` for five folds on CPU, before graph
evaluation. Compute is not the obstacle for a truly compact model.

That exact implementation is not available for a longer retry: it failed 44b6
validation precision at `0.888889 < 0.93` and 6bba five-fold OOF precision at
`0.923294 < 0.95`, and belongs to the DEAD from-scratch-CROPLINK category.
`RAW3D-NOLINK-MVE` is distinct only if it authenticates a compatible pretrained
3D checkpoint or an explicit domain-adaptation source before training. No such
checkpoint, input-scale match, or license/format receipt is pinned today.

Primary literature makes this scientifically plausible but does not predict
this competition's score:

- ELEPHANT uses sparse 3D detection and optical-flow/link supervision. Its
  authors report that a flow model taking roughly 30 hours from scratch can
  fine-tune from a pretrained model in minutes, on C. elegans and crustacean
  domains—not this dataset. [ELEPHANT](https://elifesciences.org/articles/69380).
- 3DeeCellTracker reports `1–2 h` single-GPU 3D U-Net retraining after one
  annotated volume, across worm, zebrafish-heart, and tumor-spheroid domains.
  That is a feasibility prior, not a portable runtime or score.
  [3DeeCellTracker](https://elifesciences.org/articles/59187).
- OrganoidTracker 2.0 learns 3D crop-based detection, linking, and division
  probabilities from roughly 200,000 detections/links across 1,405 intestinal
  organoid frames. Its domain, labels, and metric differ materially.
  [OrganoidTracker 2.0](https://www.nature.com/articles/s41592-025-02845-6).
  Its official training guide recommends fine-tuning a pretrained model for
  smaller data and reports 11-GB training hardware and a default 50-epoch
  early-stopped schedule.
  [Official training guide](https://jvzonlab.github.io/OrganoidTracker/TRAINING_THE_NETWORK.html).
- The Cell Tracking Challenge review documents the rise of learned methods and
  their training/generalization risks, under CTC rather than this campaign's
  adjusted Jaccard. [CTC review](https://www.nature.com/articles/s41592-023-01879-y).

The organizer's full model rate gives a cautious upper anchor: three epochs at
`1.593 h/epoch` are `4.78 h`. A compact crop model should be cheaper, but no
cross-architecture division by `118.6` is allowed. First spend at most
`0.25 T4-h` on a matching throughput/memory calibration. Only if that supports
a hard useful-checkpoint envelope of `2–6 T4-h` may the fixed run be proposed.
Before even that calibration, a zero-GPU N0 must authenticate the actual
pretrained checkpoint, license, architecture/input compatibility, and proof
that this is not simply the DEAD 30k from-scratch linker with more steps.

Its APNCC-fallback planning envelope is local `0.881522–0.900`. The floor is
the unchanged fallback; the top is a planning bound for a minimal experiment,
not a claim of attainable performance. Risk of 44b6 overfit remains high, so
the MVE needs fit-only grouping, fixed augmentation/sampling, a fixed endpoint,
atomic resume, and exactly one full held-44b6 graph evaluation. Pass remains
adjusted/raw/node `>= +0.005/0/0`; otherwise stop.

## What the public 0.91+ tier does—and does not—tell us

The official competition exposes the custom metric and leaderboard scores, but
not a primary-source mapping from the `0.91+` entries to their methods.
[Official evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation).
It would be speculation to claim that those entries use Trackastra fine-tuning,
compact 3D training, or any named architecture.

Campaign transfer evidence maps public `0.910` to local approximately
`0.9255–0.9422`. Neither MVE planning envelope reaches that band. A scoped
training run would answer whether raw 3D information can produce a material
held-graph gain; it does not have literature-grounded evidence of reaching the
leaderboard tier.

## Option 1 — open one scoped training node

Amend the locked plan to open only `RAW3D-NOLINK-MVE` under a new
preregistration. “Open” first means a zero-GPU provenance/compatibility node,
not permission to fit:

- N0: authenticate a specific warm-start checkpoint, license, input scale, and
  loader; fail closed with zero training if no compatible asset exists;
- then, and only then, warm-started crop-based raw-3D detection/association,
  `<=5M` trainable parameters, explicit no-link/background, no full organizer
  convergence;
- `<=0.25 T4-h` matching calibration first; projected useful checkpoint
  `2–6 T4-h`; hard kill and serial launch controls fixed before launch;
- fit-only grouped data; frozen sampling, schedule, augmentation, and endpoint;
  no training loss, proposal metric, AUC, or graph delta used to choose epoch;
- one full held-44b6 graph evaluation at the pre-fixed checkpoint;
- pass only at adjusted/raw/node deltas `>=+0.005/0/0`; report AUC and
  FP-budget retention descriptively; fail closes the node;
- APNCC fallback everywhere outside accepted high-confidence edits;
- no 6bba, CV57, or competition submission without the existing downstream
  gates.

Evidence for this option: the `478 h` premise does not price a useful
checkpoint; literature supports warm-started compact 3D adaptation; and raw
volume is the one information source not tested by the saturated correction
representations. Evidence against it: high overfit risk, no evidence that the
MVE can reach public `0.91`, and a possible `2–6 T4-h` opportunity cost.

## Option 2 — consolidate at 0.881 and stop

Keep APNCC-DIRECT at local `0.881522` / public `0.881`, close training, and
spend no further campaign compute.

Evidence for this option: five approaches saturate at AUC `0.50–0.67`; the
graph effect is only order `1e-4`; four action/representation families have
failed at the binding operating point; and no honest MVE score envelope reaches
the estimated local requirement for public `0.91`. Evidence against it: it
would rely on a 300-epoch cost argument that no longer excludes a short,
warm-started raw-3D falsifier, leaving the missing-information hypothesis
untested.

## Recommendation and barrier

**Recommend Option 1**, solely as one bounded information-source falsifier,
with N0 asset authentication as a binding zero-GPU precondition.
Trackastra fine-tuning is not recommended. The expected value is scientific:
determine whether raw 3D context breaks the representation ceiling at a cost
that is now plausibly single-digit T4-hours.

**HALT. NO TRAINING, GPU KERNEL, OR PLAN AMENDMENT WAS MADE.** The owner's
standing Kaggle authorization removes repeated access approval only; it does
not choose Option 1 or bypass this locked-plan amendment barrier.
