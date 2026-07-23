# Local ceiling inventory v1

Status: **complete diagnostic; no submission**

The repository-wide inventory of completed exact CV57 reports finds:

| candidate | adjusted edge Jaccard | delta vs banked `c=0.25` | role |
|---|---:|---:|---|
| banked backward motion `c=0.25` | 0.873149594 | — | current banked baseline |
| P4 identical no-attention control | **0.874290046** | +0.001140452 | current local high |
| P4 one-layer attention | 0.874006533 | +0.000856939 | DEAD causally; -0.000283513 vs its control |

The no-attention control is observation-distinct from the banked baseline (node
Jaccard 0.991023; edge Jaccard 0.974953) and its frozen score/count tuple
reconciles exactly at `34,227 / 2,658 / 2,041`.

Per-video paired scoring shows:

- no-attention wins 31/57 videos and loses 26/57;
- it improves `44b6` from 0.854173 to 0.857529;
- it improves `6bba` from 0.876647 to 0.877372;
- on the registered worst-residual top-10 slice it gains +0.004760 with
  +10 TP, -32 FP, and -10 FN;
- an inadmissible hindsight oracle choosing the better arm per video reaches only
  **0.876831363**.

The oracle is diagnostic, not a candidate. Its value below 0.88 proves that a
selector, scenario switch, or hedge over only these two predictions cannot break the
target barrier. A new association mechanism must create correct edges absent from
both arms.

Artifact:

`reports/g02/screening/obs_diff_p4_no_attention_vs_c0p25_scored_v1.json`
