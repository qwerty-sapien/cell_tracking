# Reverse N2 — independent-evidence falsifier

**Date:** 2026-07-30
**Input:** `N1_LITERATURE.md`
**Verdict:** **GO** — the independence and directional-signal claims now have an
explicit, no-data kill test and a bounded source-only protocol. No test was run and
no asset was created or acquired.

## Preregistered observable

For an instance pair `u_t → v_t+1`, a new MCC supplier may emit only three
normalized mask-contour statistics, with `P_u` the ordered source contour,
`C_v` the target contour, `F` its forward offset field, `B` its independently
evaluated backward field, `Π_v` nearest-contour re-localization, `d_u` source
equivalent diameter, and `n(p)` the source outward normal:

```
L_land   = median_p distance(F(p), C_v) / d_u
L_return = median_p ||B(Π_v(F(p))) - p|| / d_u
L_normal = median_p [1 - |cos(F(p)-p, n(p))|]
```

For `||F(p)-p|| < 1e-6 d_u`, the corresponding normal term is **one** (bad),
not zero. The later diagnostic scalar, if ever allowed, is
`E_MCC = L_land + L_return + 0.25 L_normal` (lower is better). The three values,
not an association probability, are the retained output. Division, first/last
frame, mask-topology change, empty/border-touching contours, or a missing field
must emit `UNUSABLE`, never a favourable value.

## Static analytic fixture (preregistered; no campaign data)

This fixture is a mathematical specification, not a created data file.

1. Let `P_u` be 64 equally-spaced points on the unit circle. The true target is the
   concentric circle of radius `1.10`; set `F_true(p)=1.10p` and
   `B_true(q)=q/1.10`. Expected values: `L_land=L_return=L_normal=0`, hence
   `E_MCC=0`.
2. The zero-field null uses `F_0(p)=p`, `B_0(q)=q`. It has a landing distance of
   `0.10/2=0.05` (using `d_u=2`), return zero, and `L_normal=1`; hence
   `E_MCC=0.30`. It must be rejected despite perfect round-trip return.
3. The wrong-target null uses the radius-`1.30` target while retaining
   `F_true/B_true`. Its landing term is `0.20/2=0.10`, so it must rank below the
   true target even though its return field is exact.
4. A permutation test replaces `B_true(Π_v(F_true(p)))` with the one-step cyclic
   permutation of source points. Its median normalized return error must be
   strictly positive and `E_MCC > 0`.

**Pass condition:** all four inequalities hold in exact rational/float64 arithmetic:
`E_true + 0.05 <= E_zero`, `E_true + 0.05 <= E_wrong`,
`E_true < E_permuted`, and no nonzero/finite input is classified `UNUSABLE`.

**Kill lines:**

- **DEAD:** a zero field, wrong contour, or permuted back-map passes or ties the
  true continuation; the claimed future evidence then collapses to an uninformative
  cycle/geometry shortcut.
- **DEAD:** any implementation needs Trackastra source/checkpoint/cache, its
  reverse endpoint contract, forward/reverse margin, or a derived row to compute
  `E_MCC`.
- **PARK:** the public source cannot provide a reproducibly licensed, hash-pinned
  code-and-data package, or its required mask/contour semantics cannot be expressed
  without opening campaign data.
- **HUMAN-DECISION:** only a passing static gate plus an approved new-source,
  source-only Kaggle protocol can acquire material or run a model. A passing fixture
  has no graph or supplier-pass implication.

## Independence fixture

The minimal function signature is
`mcc(frame_t, mask_u, frame_t1, mask_v, source_model) -> {L_land,L_return,L_normal}`.
A compliance fixture passes opaque arrays with no graph columns. It fails closed if
the source tree, input manifest, environment, or output schema contains any of:
`trackastra`, `model.pt` with the terminal hash, `ctc.zip.bin`, `A_reverse`,
`forward_margin`, `reverse_margin`, `reverse_*`, `challenger`, `conflict`, or a
reverse-cache path. This verifies information independence, not numerical quality.

There is no existing isolated N2 harness in this DAG. Per the task constraint,
I did not manufacture synthetic files/assets or execute a new model. The four
analytic cases above are the preregistration that such a harness must implement
unchanged after human authorization.

## Objective link

If, and only if, every source/authentication and static test later passes, the
mechanism can propose one frozen scalar selector for the separate 44b6-first graph
contract. It has no authority to edit graph actions or use proposal metrics as a
gate. The eventual success requirement remains the campaign's unchanged graph
criterion (including adjusted delta `>= +0.005` and non-regressing raw/node terms
where applicable); `E_MCC` itself is never a success claim.

## Proposed ledger row — not written

`REVERSE-MCC-N2 | GO_TO_HUMAN_GATE | analytic landing/return/normal fixture
preregistered with zero-flow, wrong-target, and permutation kill lines; all
Trackastra identifiers forbidden | no execution, asset, campaign data, scorer, or
shared-state mutation.`

**Cascade:** N2 GO → N3 human Kaggle gate.
