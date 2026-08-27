# C1 N1 — immutable preregistration

Date: 2026-07-27  
Verdict: **GO**  
Canonical contract hash: `d4a3651a2e3d429f432d90d7387478c18545f11612f481b9e43928536e26f5e9`

The complete machine-readable contract is
[`c1_n1_preregistration.json`](c1_n1_preregistration.json). Its `self_sha256` is the
SHA-256 of canonical compact sorted JSON after removing that one field.

## Frozen model

Each proposal is represented by the limiting authenticated 49-D
APNCC/Trackastra conflict row. Proposals within a transition are connected into a
set only when their challenger/displaced-conflict edges share an endpoint. The model
is:

`49 → Linear(64) → one 4-head self-attention block → residual LayerNorm → shared scalar head`.

Exactly one learned 64-D no-link token is prepended per component. The inference
score is `proposal_logit - no_link_logit`. There is no positional embedding, graph
adjacency input, topology propagation, cross-component communication, division head,
or backbone gradient. The exact trainable count is **20,097**.

## Frozen fit protocol

- Five stable video-group folds:
  `int.from_bytes(sha256(identity)[0:8], "big") % 5`.
- Calibration fold is `(held+1) % 5`; no edge-row may cross video groups.
- One seed, `20260727`; CPU deterministic algorithms.
- Fit-only normalization; AdamW, learning rate `3e-4`, weight decay `1e-4`,
  12 fixed epochs, 64 components/batch, gradient clip 1.0.
- Edge BCE is computed only on exact joined labels. Positive weight is the fit-only
  negative/positive ratio clipped to `[1,20]`.
- No-link BCE is computed only for completely labeled components; target is one iff
  the component has no positive proposal. Partial components omit the no-link
  target. Zero-label videos are inference-only, never imputed negative.
- The calibration rule is fixed before outcomes: maximize accepted support among
  finite score cutoffs with precision `>=0.95` and at least 100 accepted, with the
  higher cutoff winning ties. This diagnostic threshold is never a promotion gate.

## Controls and falsification

- Exact APNCC unchanged control.
- `C0-T4TA-LINEAR`: standardized 49-D ridge least squares, lambda `0.01`, identical
  rows/folds/calibration/selection.
- `C1-ABL-NOATTN`: same C1 projection, normalization, no-link token, and output,
  with the attention result set exactly to zero. It diagnoses the proposed
  interaction and cannot select the active arm.
- `epsilon_null=max(1e-12, 2 * maximum exact-rerun adjusted-score difference)`.

N3 is preregistered at adjusted 44b6 delta `>=+0.005`, raw and node-recall deltas
`>=0`, observation-distinct output, and C1-minus-C0
`>=max(0.001, epsilon_null)`. Adjusted 44b6 delta `<=0` is DEAD. N4 repeats the
`+0.005`/raw/node gate unchanged on 6bba.

## Outcome routes and budget

There is one architecture, one seed, and no search. N2 may implement and dry-validate
without calling the official scorer. N3 remains dependency-locked until the owner
locks `U0-T4CANON`. No GPU, Kaggle, RunPod, submission, push, or external mutation is
permitted.

## Proposed LEDGER row

`C1-TA-HOCT-LITE-N1 | PREREGISTRATION | fixed 20097-param one-block head |
d4a3651a…6f5e9; no score opened | 0 external usage | GO | all degrees and graph
routes frozen; sparse-label handling fail-closed`
