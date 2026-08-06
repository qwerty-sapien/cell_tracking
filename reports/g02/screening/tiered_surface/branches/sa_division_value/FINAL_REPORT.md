# Division value: ceiling versus realizable share

## Verdict

**Diagnostic halt; division fitting stays locked.** The exact oracle bundle ceiling is
`+0.092067841519` **[GT-DEPENDENT]**, but the subset inferable from
the current candidate pool and features is only `+0.002406489982`
**[NON-GT-IDENTIFIABLE]** at perfect selection. That is below the `+0.005` actionability
target. The remaining 37 events require a stronger representation that was not built or
measured, so DAG 2 cannot establish selector achievability.

## Ceiling and dependency decomposition

| quantity | events | combined delta | information label |
|---|---:|---:|---|
| Exact repair-bundle ceiling | 38 | `+0.092067841519` | GT-DEPENDENT |
| (a) current pool + current features | 1 | `+0.002406489982` | NON-GT-IDENTIFIABLE |
| (b) stronger representation required | 37 | `+0.089661333846` tier-only | NON-GT-IDENTIFIABLE |
| (c) irreducible GT identity | 0 | `+0.000000000000` | GT-DEPENDENT |
| (a)+(b) representation-conditioned upper bound | 38 | `+0.092067841519` | NON-GT-IDENTIFIABLE |

The GT ceiling has adjusted/raw/node deltas
`+0.001591651042` /
`+0.001607524868` /
`+0.000000000000` **[GT-DEPENDENT]**, with division TP/FP/FN
`38/0/4` and Jaccard
`0.904761904762` **[GT-DEPENDENT]**.

## False-fork sensitivity and DAG-2 handoff

For tier (a), even precision 1 yields only
`+0.002406489982`
**[NON-GT-IDENTIFIABLE]**, so +0.005 is infeasible. For (a)+(b), the edge-neutral
upper bound stays at or above +0.005 through
`1072` false forks,
equivalent to precision
`0.034234234234`
**[NON-GT-IDENTIFIABLE]**. This is not an achievability claim: there is no division
selector, empirical AUC grid, or retention measurement.

`N4_REQUIREMENT_HANDOFF.json` preserves the frozen DAG-2 schema and all three source
classes byte-value-exact, adds `division` as the fourth action class, and leaves the
frozen AUC/retention floors unchanged. Its division verdict is
`NOT_EVALUABLE_NO_DIVISION_SELECTOR`.

## Governance

No amendment proposal was emitted because the conjunctive N5 condition failed: the
current realizable subset is not materially large, and the stronger-representation
selector precision is not established as achievable. There are no tier moves, plan
changes, DEAD actions, fitting, GPU/Kaggle/RunPod/submission actions, or budget debit.
