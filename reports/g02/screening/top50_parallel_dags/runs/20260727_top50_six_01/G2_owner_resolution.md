# G2 owner resolution — U0 lock and C1-N3 resume

Date: 2026-07-27  
Verdict: **LOCK / RESUME**

The owner supplied the exact authorized text:

```text
LOCK U0-T4CANON contract
08cd90e5d67f0bf9c7822caa1a057f767b5e7e66e0889fcde04ef317cfc016c1.
RESUME C1 FROM N3 UNDER THE FROZEN LOCAL/CACHE-ONLY DAG.
```

## Consequences

- U0 contract `08cd90e5…016c1` is the locked canonical T4-domain governance contract.
- Old H1-ZWRFEAT N1 remains `DEAD_STRUCTURAL_M5_T4_DRIFT`.
- U0's 5/5 44b6 `NO_ADMISSIBLE_THRESHOLD` result remains negative beta evidence.
- C1 resumes from N3 under preregistration
  `d4a3651a2e3d429f432d90d7387478c18545f11612f481b9e43928536e26f5e9`.
- C1 uses one CPU configuration/seed, 44b6 first, maximum 3 wall-hours and 4 GiB RSS
  per family. Any non-GO halts. Only a complete N3 GO may auto-cascade to N4/6bba.
- CV57, GPU, Kaggle, RunPod, paid compute, acquisition, submission, push, and other
  external mutation remain unauthorized.

