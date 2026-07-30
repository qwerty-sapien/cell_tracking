# Selector requirement diagnostic — terminal resource barrier

**Status: `INVALID_RESOURCE_NO_VERDICT`.**

The requested exact 44b6 selector dose/noise requirement was not measured. Both
scientific attempts passed a fresh 45-second zero-swap preflight, then violated
the binding zero-swap rule during guarded execution:

| Attempt | Execution design | Elapsed | Peak process-tree RSS | Swap growth | Result files |
|---|---|---:|---:|---:|---:|
| 1 | All class banks resident | 722.620 s | 1.904 GiB | +16.31 MiB | 0 |
| 2 | Streamed class lifetimes | 1,034.259 s | 1.453 GiB | +1,567.69 MiB | 0 |

The second design was a documented execution-only repair: populations, target
AUCs, 20 seeds, support grid, corruption law, scorer, compiler, graph target,
and frozen thresholds were unchanged. All 17 focused tests passed before it ran.
The guard terminated both processes, and the empty partial output directories
were quarantined. Console markers and partial in-memory calculations are not
consumed as scientific evidence.

## Requested findings

- Dose-response curves: unavailable (`INVALID_RESOURCE`).
- Noise surface: unavailable (`INVALID_RESOURCE`).
- Minimum requirement for +0.005: no verdict for delete, occupied-swap, or
  restore-topology.
- Minimum requirement for breakeven: no verdict for all three classes.
- Frozen AUC >=0.65 / retention >=0.15 sufficiency: no verdict for all three
  classes.
- N3 threshold-amendment trigger: not evaluated, so no amendment is proposed or
  applied.

## Governance consequence

No tier move, promotion, DEAD action, threshold change, plan change, LEDGER
write, GPU/Kaggle/RunPod/submission action, 6bba access, CV57 access, or external
action occurred. The diagnostic halts. Any new execution design or retry
requires human review.

See `RESOURCE_BARRIER.json` for the machine-readable barrier and
`PROPOSED_LEDGER_ROW.md` for the proposed root-only ledger entry.
