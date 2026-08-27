# NODETOPO 44b6 graph-gate terminal report

Verdict: **KILL_44B6_STOP**.

The exact frozen `BLOCKPEAK-RESTORE-V1` population was replayed on all 36
fit-only 44b6 graphs. All 50 `restore_topology` actions and all 24 affected
video batches matched their banked frozen-N1 receipts exactly. The candidate
graphs add 50 nodes and have a net increase of three edges (100 added, 97
removed). Twenty-four of 36 graph fragments are observation-distinct.

| Arm | Adjusted edge Jaccard | Raw edge Jaccard | Node recall | Edge TP / FP / FN |
|---|---:|---:|---:|---:|
| Frozen baseline | 0.859584981641078 | 0.856184798807750 | 0.993925630490104 | 10341 / 1064 / 673 |
| 50-action candidate | 0.859581545490710 | 0.856184798807750 | 0.993925630490104 | 10341 / 1064 / 673 |
| Candidate − baseline | **−0.000003436150368** | **0.000000000000000** | **0.000000000000000** | 0 / 0 / 0 |

The preregistered adjusted floor is `+0.005`; raw and node-recall floors are
zero. Raw and node recall pass at equality, but adjusted fails. The added false
nodes do not change official edge matches and incur a small adjusted-node-count
penalty.

The binding CPU guard passed in 147.872 s (scorer: 143.265 s), with peak
process-tree RSS 1.025 GiB, swap growth 0.0 MiB, and minimum system-free memory
35%. The fresh preflight passed for 45.452 s with 0.0 MiB swap growth.

The run stopped at the 44b6 barrier. No 6bba, CV57, GPU, Kaggle, RunPod,
submission, external action, DEAD promotion, tier move, or plan amendment
occurred.

Authoritative machine result:
`graph_gate_44b6/report.json` (`73c02c27fa6a2fc46a8e7453c40ae630e946ddd2a0b29a679f0dd40ea5ec8a14`).
Binding guard:
`graph_gate_44b6_guard_retry.json` (`af840289fc8a08cede8285e2f61485081920eef4f3ce8f76e4c506b17698141c`).
Fresh preflight:
`graph_gate_44b6_preflight_retry.json` (`87dca9b9a6b7f2f3a2645aaa4dfc0f7db8ef8001deff9156ba2d85681680836d`).

The first guard receipt, `graph_gate_44b6_guard.json`, is an invalid pre-score
attempt: it failed closed on the obsolete grouped-fold scorer-source pin before
creating graphs or metrics. The retry authenticated the banked parity-tested
official scorer sources and the focused terminal suite passed 10/10.
