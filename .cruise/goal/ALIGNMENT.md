# G-02 Alignment and Coverage

## Stem

After separate Plan approval and Start, deliver the checkpoint-based sparse hybrid cell-lineage tracker: fine-tune the released official `TemporalUNet3D` with compact heads/adapters, physical-coordinate KD-tree-gated sparse association, and constrained global solving, while retaining the rule-based alpha as teacher/fallback; completion requires official grouped local CV, untouched 20-video audit, and a displayed qualifying private Kaggle score each >= `0.92`, with final runtime projected <= six hours. Kaggle effects remain private and within locked spend, concurrency, and submission caps; prohibit dense NxN association, large from-scratch training, publishing, unrelated Kaggle/account actions, and git history or remote effects.

## Requirement-to-evidence trace

| Requirement | Acceptance | Decisions | Producer tasks | Mandatory gates |
| --- | --- | --- | --- | --- |
| R-01 scorer parity | AC-01 | D-02, D-09 | T-02, T-09 | E-01, E-08 |
| R-02 leakage-safe validation | AC-02 | D-02 | T-02, T-03, T-05, T-06, T-10 | E-02, E-09 |
| R-03 compact learned core | AC-02, AC-07 | D-01, D-03, D-08 | T-03, T-06, T-10 | E-03, E-09 |
| R-04 sparse graph | AC-04, AC-05 | D-01, D-08 | T-04, T-06, T-07, T-10 | E-04, E-07, E-09 |
| R-05 routing | AC-03 | D-02, D-03, D-08 | T-05, T-06, T-10 | E-05, E-09 |
| R-06 HPO/runtime/manifest | AC-05, AC-07 | D-04, D-05, D-08 | T-06, T-07, T-10 | E-06, E-07, E-10 |
| R-07 controlled Kaggle evidence | AC-01, AC-06, AC-07 | D-04–D-07, D-09, D-10 | T-08, T-09, T-10 | E-08, E-10 |
| R-08 division/completion integrity | AC-02, AC-04, AC-06 | D-08, D-09 | T-04, T-06, T-10, T-01 | E-04, E-09, E-10 |

## Acceptance-to-gate trace

| Acceptance | Pass condition | Gate and evidence |
| --- | --- | --- |
| AC-01 | Every available frozen local/site pair has absolute residual <= `0.02`. | E-01: scorer semantic fixtures, pair ledger, calibration-vs-holdout role audit. |
| AC-02 | Grouped 3-fold and untouched 20-video audit are each >= `0.92`, with diagnostics. | E-02/E-03/E-09: frozen split manifest, fold reports, audit report. |
| AC-03 | Mean router regret <= `0.01`; worst scenario <= `0.03`; fallback tested. | E-05: OOF route/oracle/fallback report. |
| AC-04 | All graph invariants and full coverage pass. | E-04: graph validator, per-dataset coverage/constraint report. |
| AC-05 | Four-clip <= seven minutes; measured proxy projects <= six hours. | E-07: timestamped stage profile and extrapolation inputs. |
| AC-06 | Private qualifying Kaggle submission complete with displayed score >= `0.92`. | E-08/E-10: artifact-linked Kaggle ID/status/score and cap ledger. |
| AC-07 | Manifest joins hashes, folds, config, package, profile, scores, and spend. | E-06/E-10: reproducibility and accounting audit. |

## DAG and ownership

```text
T-02 ─┬─> T-03 ─> T-04 ─> T-05 ─> T-06 ─> T-07 ─> T-08 ─> T-09 ─> T-10 ─> T-01
      └───────────────────────────────────────────────────────────────────────────^ 
```

`T-02` establishes immutable scorer/split identities. `T-03` produces checkpoint/head outputs; `T-04` consumes them to produce sparse graph evidence; `T-05` consumes out-of-fold predictions; `T-06` owns cached HPO and division promotion; `T-07` makes the promoted inference package/runtime profile; `T-08` is the only ordinary controlled Kaggle mutation node; `T-09` is read/analysis of returned scores and frozen calibration roles; `T-10` owns final validation/package/submission evidence. `T-01` is the sole final read-only adversarial verifier and runs after every producer.

| Mutable resource | Single active owner | Serialization and protection |
| --- | --- | --- |
| Split/scorer manifests under `reports/g02` | T-02 | Immutable versioned input thereafter; later tasks may append references but not edit membership. |
| Model adapters/heads and training scripts | T-03, then T-06 | T-06 changes only promoted candidates; preserve alpha code/fallback. |
| Candidate gating/solver code | T-04, then T-06 | No dense NxN allocation; tests and graph reports required before downstream use. |
| Router code/config | T-05, then T-06 | OOF-only fit; low-confidence universal policy fixed at 0.60 unless a locked decision changes. |
| Cache/HPO ledger and candidate manifests | T-06 | Atomic per-candidate identity; no concurrent writer. |
| Package/runtime profile | T-07, then T-10 | Package hash changes invalidate prior timing evidence. |
| Kaggle notebooks, scored-submission/spend ledger | T-08, then T-10 | Serialized; one scored/day, ten total, two paid concurrent maximum. |
| Site-calibration ledger | T-09 | Read only pre-score definitions; no post-hoc role reassignment. |

## Permission and gate coverage

- No preparation task accesses product paths, network, tests, or Kaggle. Start-time local tests and benchmarks map to T-02–T-07 and T-10 only.
- Network/Kaggle effects map exclusively to T-08/T-10 and D-04–D-07/D-10. A browser is a fallback only after purpose-built authenticated CLI/API access fails.
- No task authorizes public publishing, dataset creation, external GPU, git history mutation, push, credential change, or a cost beyond D-05.
- A task may not bypass an upstream mandatory gate with a later score or package result. T-01 reports BLOCK/NOT_RUN when evidence is absent; it cannot remediate.

## Coverage validation

- T-01 appears exactly once, has no writable repository path or external effect, and is the final node.
- The dependency graph is acyclic and every task has an explicit owner, allowed effect, focused evidence, bounded remediation, and stop rule.
- All eight requirements, seven acceptance criteria, ten decisions, scorer/split/training/graph/router/HPO/runtime/Kaggle/calibration workstreams, and cost/submission limits have a producer and at least one gate.
- The G-01 local result and 403 state are preserved as historical evidence, never promoted to G-02 success evidence.
