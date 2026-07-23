# G-02 Roadmap

## Phase 0 — Controller and authority

The controller freezes artifact identities, permissions, submission/final-selection
counters, compute budgets, and sole-writer ledgers. G-02 is active. Ordinary commits and
pushes to the private repository are permitted; every Kaggle submission and RunPod launch
requires explicit human approval.

## Phase 1 — Trustworthy evaluation foundations and transfer audit

P0A/T-02 locks the official scorer semantics and group/audit split registry before fitting.
Once those pass, local evidence consolidation, mechanism screens, and candidate development
may proceed. P0B separately audits local-to-Kaggle transfer with canonical observation
comparison, score-dispersed/mechanism-diverse probes, and rounded-score intervals. P0B
uncertainty limits public forecasting and final selection; it does not freeze local work.

## Phase 2 — Local mechanism screens and compact learned hybrid

Consolidate existing B1/B2/E1 evidence before rerunning it. T-03 adds the
released-checkpoint shared backbone and output heads. T-04 builds KD-tree candidate
gating, lightweight learned edge scoring, and global constrained lineage solving. T-05
establishes the specialists and OOF router. Association and detection GPU gates are
independent and depend on measured headroom, proxy gain, runtime, and budget—not on the
transfer-model verdict.

## Phase 3 — Promotion and performance

T-06 uses cached 10/30/100 percent successive halving, applies the division activation rule, and promotes only candidates meeting metric/graph/runtime gates. T-07 packages the promoted candidate and measures end-to-end timing.

## Phase 4 — Sequential external measurement and endpoint

T-08 performs only predeclared, ledgered private Kaggle experiments inside
cost/concurrency/submission caps. Calibration probes count against the five-per-day cap,
are approved and measured one at a time, and update the remaining design after each
rounded result. T-09 fits only identifiable interval-aware transfer models without
changing frozen roles. T-10 reruns full local/audit/runtime/package checks and creates or
verifies the qualifying private Kaggle submission while keeping two final selections
uncommitted until the terminal decision.

## Phase 5 — Independent closure check

T-01 reads all artifacts adversarially. It can PASS only when local, audit, package, runtime, graph, cost, and displayed Kaggle site evidence all satisfy the locked endpoint. It does not modify any file or external system.

## Milestones and hard gates

| Milestone | Entry gate | Exit gate |
| --- | --- | --- |
| M-01 foundation | Start approved | Immutable grouped split/audit and exact scorer-semantic fixture manifests pass E-01/E-02. |
| M-01B transfer design | P0A PASS and exact prediction identities | Observation-aware, interval-aware sequential public-probe design is preregistered; unresolved transfer does not block M-02. |
| M-02 edge-only hybrid | M-01 | Edge-only grouped CV >= `0.90`, sparse graph valid, allowing division evaluation. |
| M-03 promoted router | M-02 | Specialist/router regret gates, HPO promotion gates, and grouped CV >= `0.92`. |
| M-04 production readiness | M-03 | Untouched audit >= `0.92`, four-clip/runtime proxy gates, package and manifest pass. |
| M-05 external endpoint | M-04 | Qualifying private Kaggle site score >= `0.92`, within ledgered limits. |
| M-06 final verification | M-05 | T-01 PASS; otherwise Blocked with precise failing gate. |

## Stop boundaries

Do not start a paid or scored external action without a recorded maximum
charge/submission count, exact artifact identity, required local gates, and explicit
human approval. Never exceed five Kaggle submissions per competition day or consume
either final selection during screening. RunPod always requires launch-specific approval.
Do not use the audit to select a candidate. Do not add divisions early. Stop rather than
broaden any permission or lessen the metric, scorer, graph, runtime, or cost threshold.
