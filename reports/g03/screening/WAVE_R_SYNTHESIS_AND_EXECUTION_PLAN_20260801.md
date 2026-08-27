# G-03 Wave R — synthesis, allocation, and autonomous execution plan

Date: 2026-08-01  
Status: **R1–R4 synthesized; Branch B allocation decision final; CPU R4 execution active**

## Decision frame and measured result

The campaign best remains `0.881522069960`. The binding allocation floor is the
exact owner formula:

`0.881522069960 * 0.985 = 0.868299238910600`.

R1 measured one pretrained Trackastra plus real-image, untrained 12-D crop-adapter
configuration on the exact 36-video `44b6` held-fit objective. All inference bytes
sealed before target access. Its final combined/adjusted-edge score is
`0.7637338804169099`, so its deltas are `-0.1045653584936901` versus the floor and
`-0.1177881895430901` versus the campaign best. It is **Branch B**.

R1's separate terms are adjusted edge Jaccard `0.7637338804169099`, raw edge
Jaccard `0.7584738824299984`, node recall `0.9723341840137287`, edge
`TP/FP/FN = 9264/1200/1750`, and division `TP/FP/FN/Jaccard = 0/0/9/0`.
It trained for zero steps and consumed `0 T4-h`.

The first frozen-scorer process computed metrics but strict JSON could not serialize a
legitimate per-video division-Jaccard NaN. Before any value was exposed, an incident
was sealed and one identical-input recovery stringified only that diagnostic. Terminal
accounting is two target-access attempts, one scientific configuration, one exposed
candidate score, no post-result tuning, and no repeated inference. The validated R1
result SHA is `55300582b831454ab6ba2b538cba7271d0c99bb230c2db24a2c049477f514372`.

## Candidate classification and GPU allocation

| candidate | G-03 evidence | policy class | 30 T4-h decision |
|---|---|---|---|
| `G03-TA-CROPADAPTER-E2E-V1`, represented by R1 zero-shot real-`wrfeat` measurement | `0.7637338804169099` | Branch B | **INELIGIBLE / 0 h**. R2 fails the required `<=30 T4-h` comparability proof and fails calibrated paths to `0.915` and `0.945`. |
| R4 post-hoc division decoder on the fixed R1 association output | local outcome pending; CPU-only transformation | allocation branch not applicable until it has an absolute score; development is allowed | **0 h**. Execute the fixed CPU sweep in parallel; it cannot acquire the GPU lock. |
| Chen et al. zero-shot SAM2 lineage tracker | related-domain primary result only; no G-03 score or metric conversion | PENDING / unclassified | **0 h**. It is an observation-distinct future measurement node, not evidence for the present Trackastra candidate. |
| G-02 native Trackastra and G-02 A/B/C division realizers | archived G-02 evidence | outside G-03 authority; exact failed mechanisms remain parked | **0 h**. No rescreen under the old mechanism names. |

There is therefore **no eligible candidate** to receive the single 30-hour allocation.
Nominal Wave-R allocation is `0/30 T4-h`; current Wave-R consumption is `0 T4-h`.
The latest authenticated global Kaggle account balance remains `1.01/28.99 T4-h`
used/remaining with refresh at `2026-08-08 00:00 UTC`. The unspent allocation is
preserved; quota expiry is not a reason to launch an ineligible job. RunPod remains
forbidden.

## Explicit reference points for every gate

No delta below has an implicit control.

- Allocation floor: candidate absolute score `>=0.868299238910600`, derived from
  campaign best `0.881522069960`, or an absolute new best.
- If a future candidate becomes eligible for the first 30 training hours, the owner
  continuation gate is `+0.010` versus campaign best `0.881522069960`, hence absolute
  score `>=0.891522069960`.
- The 90-hour continuation gate is `+0.035` versus that same campaign best, hence
  absolute score `>=0.916522069960`. There is no inferred 60-hour gate.
- Branch-B evidence must separately support comparability within `<=30 T4-h` and a
  realistic absolute path through `0.915` and `0.945`; R2 supports neither for the
  current candidate.
- R4 development attribution is always against the exact same-video `DIVISIONS_OFF`
  parent graph identified by SHA; edge and division terms remain separate.
- R4's `44b6` one-shot confirmation has a closed same-population reference set:
  APNCC `0.866534526036` (aggregate graph
  `48b99c5db7024c0998b779d2dfcf30bade9c12d74ec679325cc1974ceb9f7ccb`)
  and R1 `DIVISIONS_OFF` `0.7637338804169099` (parent aggregate
  `e7a2aaf1bf565acfc2948ca79da17f742efa2fd51be8b7609ff0e3b5e239dbd6`).
  The binding same-population absolute hurdle is therefore strictly greater than
  `0.866534526036`; the campaign-best comparison to `0.881522069960` is also printed
  but labeled cross-population, not used as a silent causal control.
- R4 confirmation additionally requires positive division TP, division Jaccard
  `>=0.30`, division FP reported explicitly, zero duplicate survivors, and no adjusted
  edge/raw edge/node regression versus the named same-video parent. Every outer fold
  must contain a GT division and have nonnegative combined attribution versus its named
  parent.
- The banked false-fork prior remains explicit: at `TP=25`, division contribution is
  `+0.059523810`, `+0.053191489`, and `+0.048076923` for `FP=0/5/10`. It is a risk
  curve, not a G-03 achievability claim.

## Autonomous CPU execution order

1. Promote and validate the exact R1 contract, adapter, Trackastra checkpoint/source,
   36 held-fit banks/parents, result, and incident-aware validation receipt. Root
   promotion receipt SHA is the sole authorization for R4 bank extension.
2. Run target-blind inference for exact dev69 `6bba` under the identical R1 scientific
   configuration. Seal all dev69 banks and `DIVISIONS_OFF` parents plus the already
   sealed held36 identities into one local-105 coverage receipt before any development
   scorer access. This is CPU-only and score-blind.
3. Run the fixed 18,144-active plus `DIVISIONS_OFF` lattice on dev69. For each of five
   outer folds, tune only on the other four folds, seal one inner winner before reading
   its outer fold, and record edge TP/FP/FN/Jaccards and division TP/FP/FN/Jaccard
   separately. Exact graph hashes are deduplicated before scorer calls.
4. Select the deployable configuration only by highest outer-winner count and then
   lexicographically smallest configuration name. Metric re-pooling for this final
   selection is forbidden.
5. If all development gates pass, bind the validated R1 score/parent hash, APNCC
   reference, chosen config, and all held parent/overlay hashes; write the one-shot
   marker before the sole held36 target access. Otherwise park this exact decoder node.
6. Only after a passed held36 local gate, infer hidden4, seal the hidden bank and
   byte-exact submission artifact, reconcile serial lock/quota, and make one autonomous
   observation-distinct daily probe. Hidden TP/FP/FN are unavailable and must never be
   invented. A failed local gate means no hidden inference and no submission.

R4 uses the effective population SHA
`e25416da1d240ceabf2b1457ceb6625b361523e7a537ed2f4ac195280b425954`,
decoder SHA `d310394310116ebf257d74159a39be1d77679a9dd7aec14d97f3589e9f7da44a`,
bank-extension SHA `7cc4c7d2d27d5b4ed45b11cbdd0e73d33a07099f9904fa579bdd6012659e0a9c`,
and exact-runner-binding OOF-orchestrator SHA
`dc30414e9805ccec52f6662dbb0fb3e07dc794eeeb393a90d632db4044f5fdcc`,
OOF-runner SHA
`2196a84d1ad4a555552b7aeaa5a8717627a4b4d675dbd580daaa61fffe0df5f0`,
and repaired target-blind input-generator SHA
`c0937c982820c7ec9e3551bc9b5201cd7c8a5bafb96d9f77a810c8f9f0d1d2d9`.
Audit20 is refused. Hidden4 remains unopened. The only halt condition remains the
owner-only final-slot selection after the autonomous evidence chain is exhausted.
