# SA-DELETE terminal report

Status: **BANKED — PARK-C (44b6 kill)**
Node: `DELETE-O0 / AE-P1`
Decision time: 2026-07-27T18:19:37Z

## Decision

The preregistered conservative retention/no-link beta failed the first-family
held-fit graph gate:

| 44b6 metric | Control | Candidate | Delta | Gate |
|---|---:|---:|---:|---|
| adjusted edge Jaccard | 0.8627823968 | 0.8626552778 | **-0.0001271190** | FAIL (`>=+0.005`) |
| raw edge Jaccard | 0.8584832506 | 0.8583514437 | **-0.0001318070** | FAIL (`>=0`) |
| node recall | 0.9944597103 | 0.9944597103 | **0.0000000000** | PASS (`>=0`) |

The compiler accepted all actions (`invalid_action_count=0`). The candidate
proposed 14,685 deletions; the minimum-bundle rule compiled 14,170 across 36
action videos and selected N1 no-op on five videos. Diagnostic labels identify
14,158 deleted false edges and 12 deleted true edges. Every individual action
compile matched its final one-action batch compile.

Because the 44b6 GO conjunction was false, the exact frozen gate prevented
6bba graph scoring. CV57 was never accessed. The terminal verdict emitted by
the runner is `PARK_C_44B6_KILL`.

## Interpretation

The P0 delete oracle contains real headroom, but this exact 13-parameter,
12-feature retention realization does not capture it at graph level. The
negative adjusted and raw deltas falsify this instantiation; they do not justify
a DEAD promotion of every learned deletion mechanism. Do not rerun this exact
instantiation without a new named mechanism and a newly locked contract.

## Reproducibility and scope

- Result: `graph_result.json`, SHA-256
  `28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a`.
- Frozen population: exact P0 fit117 / fold-0 fit set (41 44b6 + 76 6bba);
  audit overlap zero. The 6bba rows participated only in the preregistered
  fit/calibration population, not the conditional 6bba graph gate.
- Five deterministic nested folds; all five 13-parameter Newton/IRLS fits
  converged in four iterations.
- N1 implementation SHA-256
  `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5`;
  interface SHA-256
  `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865`.
- Replay preflight: 45.52 seconds, zero swap growth, minimum free memory 42%.
- Replay guard: PASS, 283.99 seconds, 1.198 GiB peak process-tree RSS, zero
  swap growth, minimum free memory 34%.
- No GPU, Kaggle, RunPod, submission, public, validation, audit, test, or CV57
  action occurred.

## Serialization repair provenance

The first fully computed 44b6 pass failed only when strict JSON serialization
encountered undefined per-video division Jaccards. No aggregate was printed or
banked and 6bba did not open. Before replay, the implementation was changed
only to encode non-finite diagnostic floats as JSON `null`; model fitting,
thresholds, actions, graph scoring, and gates were unchanged. The repair is
locked in `postoutcome_serialization_repair_v1.json`; the failed guard
`guard_graph.json` is retained as negative evidence.
