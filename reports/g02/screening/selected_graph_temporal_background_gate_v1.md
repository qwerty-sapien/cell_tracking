# Full-context temporal/background selected-graph gate

Decision: **PARK. Do not replay on CV57.**

This CPU-only screen repaired the two limitations of the earlier temporal
proxy: it reconstructed the complete frozen backward-motion `c=0.25`
selected graph for every video, and it trained against the existing
hard+uniform deployment-background sample rather than labeled conflicts alone.

## Exact fit-only preparation

The immutable 105/12 split produced:

| data | rows/proposals |
|---|---:|
| labeled conflict rows | 8,954 |
| labeled positive rows | 5,537 |
| labeled rows with full predecessor/successor context | 8,541 |
| fit-train deployment-background proposals | 19,398 |
| expanded ambiguous background option rows | 29,667 |
| background rows with full context | 27,282 |

Each row combines the existing 44 competitor-relative features with full
`16×32×32` pair-NCC margin, predecessor/successor NCC margin, physical motion
margin, and a context-availability flag. Ambiguous geometry reconstruction is
kept as alternatives and scored conservatively: maximum across compatible
incumbents, then minimum across a proposal's conflicts.

## Frozen selection contract

Four compact linear/MLP configurations were compared with deterministic
five-fold video-grouped OOF training. Every fold selected its threshold from
fit videos only. Admissibility required:

- at least 50 OOF accepted labeled proposals;
- OOF labeled precision at least 0.95;
- every held-out video below `max(16, ceil(0.005 × selected_edges))` sampled
  background acceptances.

The fixed 12-video validation partition was not used for configuration or
threshold selection.

## Result

No configuration was OOF-admissible:

| configuration | OOF correct / harmful | precision | background accepted | all-video support |
|---|---:|---:|---:|---|
| linear, background weight 0.05 | 1,807 / 27 | 0.9853 | 4,038 | FAIL |
| MLP-16, background weight 0.05 | 2,411 / 82 | 0.9671 | 2,732 | FAIL |
| MLP-16, background weight 0.20 | 2,545 / 171 | 0.9370 | 2,198 | FAIL |
| MLP-32, background weight 0.20 | 2,479 / 153 | 0.9419 | 2,029 | FAIL |

The best diagnostic was the linear gate. Refit on all 105 fit-train videos,
its threshold passed in-sample background support and achieved 0.9869 train
precision. Applied once to validation, it accepted 113 correct and one harmful
proposal: 0.9912 precision but utility 111. The existing pair-NCC rule accepts
164 correct and eight harmful proposals on the same validation rows, utility
148. Thus the new gate improves abstention but loses too much recall and does
not meet the preregistered utility floor of 153.

This branch is not eligible for CV57. The result narrows the next model need:
the remaining improvement requires a representation that adds recall while
maintaining pair-NCC's precision, not simply stronger background suppression.

Artifacts:

- `scripts/prepare_selected_graph_temporal_background_gate.py`
- `scripts/fit_selected_graph_temporal_background_gate.py`
- `reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json`
- `reports/g02/selected_graph_temporal_background_gate_v1/report.json`
- focused tests under `tests/`

No held-out/audit/test labels, Kaggle submission, GPU launch, or paid compute
were used.
