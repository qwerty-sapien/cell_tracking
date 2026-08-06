# Division value: ceiling versus realizable share (N0–N3)

## Mandatory interpretation

The pooled **+0.0920678415** result is **GT-DEPENDENT**. It restores the exact
GT-identified endpoints in all 38 frozen Wave-3 bundles and is not evidence of a
reachable gain. It cannot justify unlocking division fitting or amending the
locked plan.

The current-pool/current-feature subset contains only **1/38** events. Its exact
replay is **+0.0024064900** combined score and is
**NON-GT-IDENTIFIABLE** because this DAG measures an information-qualified
subset, not a trained selector. The stronger-representation tier contains the
other **37/38** events and contributes **+0.0896613338** alone, but this is also
**NON-GT-IDENTIFIABLE** headroom: no stronger-representation realizer was built
or measured here.

## N0 GT-dependent ceiling

| Scope | Adjusted edge Δ | Raw edge Δ | Node Δ | Division TP/FP/FN | Division Jaccard | Combined Δ | Label |
|---|---:|---:|---:|---:|---:|---:|---|
| 44b6 | +0.0021877277 | +0.0021914829 | 0 | 9/0/2 | 0.8181818182 | +0.0840059095 | GT-DEPENDENT |
| 6bba | +0.0014817514 | +0.0014995325 | 0 | 29/0/2 | 0.9354838710 | +0.0950301385 | GT-DEPENDENT |
| pooled CV57 | +0.0015916510 | +0.0016075249 | 0 | 38/0/4 | 0.9047619048 | +0.0920678415 | GT-DEPENDENT |

## N1 GT-dependency decomposition

| Tier | Meaning | Events | Event share | Tier-only combined Δ | Shapley share | Label |
|---|---|---:|---:|---:|---:|---|
| a | current pool + current features | 1 | 0.0263157895 | +0.0024064900 | +0.0024064988 | NON-GT-IDENTIFIABLE |
| b | stronger representation required | 37 | 0.9736842105 | +0.0896613338 | +0.0896613427 | NON-GT-IDENTIFIABLE |
| c | GT identity only / deployment-infeasible | 0 | 0 | 0 | 0 | GT-DEPENDENT |

The sole tier-a event is
`train:6bba_61ecbe65|gt-divider:207|event:0`. Its exact children are the
deterministic current-probability top two, the second probability
`0.3303201212` exceeds no-link `0.2517110087`, both children have observable
selected outgoing continuation, and there is no boundary tie.

## N2 honest range

- Current pool/current features only (tier a): division **1/0/41**,
  division Jaccard `0.0238095238`, adjusted-edge Δ `+0.0000255376`,
  combined Δ **+0.0024064900** — **NON-GT-IDENTIFIABLE** diagnostic.
- Current tier plus stronger-representation headroom (tiers a+b): division
  **38/0/4**, division Jaccard `0.9047619048`, combined Δ
  **+0.0920678415** — **NON-GT-IDENTIFIABLE** representation-conditioned
  upper bound, not a realizer.

The present reachable-looking share is therefore near zero relative to the
ceiling: one exact event and 2.61% of the pooled combined-score headroom. The
division lever currently exhibits the NODETOPO lesson: almost all banked value
comes from GT endpoint access or an unbuilt representation.

## N3 false-fork sensitivity

`FP_SENSITIVITY.json` enumerates the edge-neutral upper-bound payoff at every
integer FP count. Precision is `TP/(TP+FP)` and the division term is
`0.1 × TP/(42+FP)`; the measured true-bundle adjusted-edge term is held fixed.
Actual false forks can be worse because they may also damage edge Jaccard.

For tier a, even perfect selection yields only **+0.0024064900**, so the
`+0.005` graph threshold is **infeasible at precision 1.0**. For the
representation-conditioned a+b upper bound, the last edge-neutral point above
`+0.005` is **1,072 FP at precision 0.0342342342** (the next point falls below
the threshold). That boundary is not evidence that such precision is achievable:
tier b has no built representation or selector, and the edge-neutral assumption
is optimistic.

## Execution boundary

All 38 frozen actions compiled individually with their pinned Wave-2 hashes.
Frozen `compile_batch` rejects an action that owns and re-adds an already-correct
daughter edge, so disjoint per-video actions were composed through ordered public
`compile_action` calls. Multi-event videos produced identical final graph hashes
in reverse order. No N1, bundle, scorer, fold, P0, or plan bytes changed.

The final CPU-only replay took 187.019 seconds, peaked at 1.138 GiB process RSS,
and had 0 MiB swap growth. No Kaggle, GPU, RunPod, budget debit, fitting, bundle
derivation, submission, tier move, DEAD decision, or plan amendment occurred.
