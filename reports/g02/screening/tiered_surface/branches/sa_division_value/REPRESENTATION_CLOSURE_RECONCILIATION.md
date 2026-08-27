# DAG 4 division-repair realization — representation-closure reconciliation

The CPU realization was already complete before DINOv2 closed the representation
lever. This addendum changes no frozen input, compiler, scorer, action, or metric.
It reconciles the terminal interpretation only.

## N0 — GT-dependent ceiling

All 38 exact frozen repair bundles passed the public frozen-N1 `compile_action`
interface. Ordered composition and reverse-order final-graph hash equality were
used because `compile_batch` rejects an action that owns and re-adds an already
correct daughter edge.

| Quantity | Exact result |
|---|---:|
| Adjusted-edge delta | `+0.0015916510423586994` |
| Raw-edge delta | `+0.0016075248677311915` |
| Node delta | `0` |
| Division TP / FP / FN | `38 / 0 / 4` |
| Division Jaccard | `0.9047619047619048` |
| Combined delta | `+0.09206784151854919` |

This is the **GT-DEPENDENT ceiling**, not a deployable result.

## N1–N2 — information decomposition and honest realizable range

| Tier | Events | Exact combined delta | Current status |
|---|---:|---:|---|
| (a) Existing pool + current features | 1 | `+0.002406489981796045` | available perfect-selection diagnostic **[NON-GT-IDENTIFIABLE]** |
| (b) Stronger representation required | 37 | `+0.08966133384644293` tier-only | **UNAVAILABLE [NON-GT-IDENTIFIABLE]** |
| (c) GT-only identity | 0 | `0` | none |

The honest currently realizable range is therefore the tier-(a) point alone:
`+0.002406489981796045` **[NON-GT-IDENTIFIABLE]** at perfect selection; it is
not a realized selector gain. The former `(a)+(b)` value
`+0.09206784151854919` remains only an **UNAVAILABLE,
representation-conditioned counterfactual [NON-GT-IDENTIFIABLE]**.

## N3 — false-positive sensitivity

The banked diagnostic curve at 25 true-positive divisions is:

| TP | FP | Division-only combined addition |
|---:|---:|---:|
| 25 | 0 | `+0.05952380952380952` |
| 25 | 5 | `+0.05319148936170213` |
| 25 | 10 | `+0.04807692307692308` |

These are **NON-GT-IDENTIFIABLE** sensitivity diagnostics, not realized
predictions. Their direct banked source is
`sa_division_realizer_wave/arm_c_appearance/RESULT.json`,
SHA-256 `4e04cc3a8a182f5113d0a0f1414eed4e45b03d885613ad7d4a2145e79de365d9`.

## N4 — barrier

Tier (a) is below the preregistered `+0.005` materiality line even at perfect
selection. No amendment proposal is triggered. There is no tier move, plan
amendment, graph opening, DEAD action, fitting, GPU action, or submission.
