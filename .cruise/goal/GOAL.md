# Goal: G-02 — Checkpoint-Based Sparse Hybrid Lineage Tracker

- State: **Approved (inactive)**. This contract is bound to the separately approved G-02 Plan; it does not activate execution.
- Plan approval: `.cruise/runs/20260716T082626Z-goal-g02/plan-approval-02.yaml` (`70769ec01215beecbebdb8e6f64fb029821c69d6c873b2d8d167d41e7b81e08d`), `execution_authorized: false`; required next selection is **Start Cruise**.
- Baseline: `c3b9b703b049fc25f10e2b42bb986bb94f137a72`; dirty-state digest `6c4d0a9af20a760e7ff6dbde5dcdae0c3f0f72bc43c8b2c1336dbcaf5238ca6d`.

## Immutable contract and first reads

Before any native execution action, read in this order: `ORIGIN.yaml`, `ALIGNMENT.md`, `PLAN.md`, and this `GOAL.md`. `SPEC.md` and `DECISIONS.md` are normative detail; do not reinterpret them. Revalidate the raw-prompt hash and all locked artifact identities:

| Artifact | SHA-256 |
| --- | --- |
| `ORIGIN.yaml` | `067328a2d3828b767777a0f6a38960ae1ec4b851487b8f36fc4f71543851554a` |
| `SPEC.md` | `a075723eeefe0b829b914b9e077deb16c14c9e8997180e488f2560b19da81a16` |
| `DECISIONS.md` | `57764184460a95d8544e779eb3e80623b0080b42962aa8e87ddc6a6e0c9d5725` |
| `ALIGNMENT.md` | `fa95d6ddfa3cc787036569e5e10535074f9f1aedcdf5b166fa49ca325e991512` |
| `ROADMAP.md` | `e0991ba058253e29dfa70c067968048b14e8adaa1d6161fed321a51e9e65076d` |
| `PLAN.md` | `ddb7dc34cd0ad747601f92bca01d4a73851f4e22182a0f96272296f4063538bf` |

Preserve these bytes, the G-01 archive, and all accepted pre-existing worktree changes. A mismatch is interruption, not a repair opportunity.

## Objective and stopping condition

After confirmed Start, deliver the approved low-compute tracker: fine-tune the released official `TemporalUNet3D` only with compact heads/adapters; use physical-coordinate KD-tree-gated sparse learned association and constrained global solving; retain the rule-based alpha as teacher, fallback, and regression baseline.

Completion is conjunctive: grouped three-fold official local CV **>= 0.92**, untouched stratified 20-video audit **>= 0.92**, displayed qualifying private Kaggle score **>= 0.92**, scorer parity **<= 0.02** on every available frozen pair, all graph/router/package gates PASS, and measured final-workload projection **<= 6 h** (with public four-clip time **<= 7 min**). Local-only success is not completion. The current historical baseline is **0.8713516075529171, local-only**; it is neither calibration nor Kaggle evidence.

## Coordinator, state, and execution loop

The root coordinator owns activation checks, DAG state, integration, foreign-diff review, and the sole `CLOSEOUT.md` write. It assigns `recommended_level: 4` and uses the Doctor-attested Cruise worker for every substantive planning, execution, and verification task. Workers return immutable attempt reports and never update closeout directly.

For every Ready node, use the fast loop: **orient → smallest micro-plan → preflight ownership/effects → implement → focused deterministic evidence → preserve attempts → integrate → checkpoint**. The micro-plan records inputs, exclusive writes/resources, smallest change, exact focused check, effect authority, and rollback/stop. A node completes only with current output identities and its focused evidence; a failed or stale node never advances on assertions alone.

Maintain durable progress memory under `.cruise/runs/20260716T082626Z-goal-g02/`: node task state and attempts; artifact hashes; experiment and promotion ledgers; spend and scored-submission counters; runtime profiles; frozen calibration/holdout roles; blockers; foreign-diff incidents; and the next Ready node. Keep source, package, model, fold, cache, configuration, score, and spend identities joinable. Preserve every failed attempt and invalidate dependent evidence when an upstream identity changes.

## Scorecard

| Acceptance | Mandatory PASS evidence |
| --- | --- |
| AC-01 | Official semantic fixtures and every frozen local/site residual <= `0.02`. |
| AC-02 | Grouped 3-fold and untouched 20-video audit both >= `0.92`, with per-sample node/edge/division diagnostics. |
| AC-03 | OOF router mean regret <= `0.01`, worst scenario <= `0.03`, and universal fallback test. |
| AC-04 | Valid, complete graphs: unique IDs, forward time, legal degree/divisions, no merges, full coverage. |
| AC-05 | Four clips <= `7 min`; measured 199-video/final-workload proxy <= `6 h` end to end. |
| AC-06 | Artifact-linked private Kaggle submission is complete with displayed score >= `0.92`. |
| AC-07 | Immutable manifest joins source/checkpoint/fold/config/package/runtime/scores and pre/post-trial spend. |

| Gate | Mandatory PASS evidence |
| --- | --- |
| E-01 | Official scorer fixture, frozen calibration roles, and parity residuals. |
| E-02 | Immutable hash-grouped folds, exact audit membership, and fit-source leakage audit. |
| E-03 | Released checkpoint/hash, compact head schema, fold isolation, deterministic restore. |
| E-04 | Physical KD-tree sparse candidates; graph legality and coverage; no dense NxN association. |
| E-05 | OOF specialist/router/oracle evidence and low-confidence universal fallback. |
| E-06 | Cache identity, 10/30/100 promotion ledger, reproducible manifest, pre-launch accounting. |
| E-07 | Current package replay plus four-clip and measured final-workload runtime profile. |
| E-08 | Private artifact-linked Kaggle IDs/status/displayed scores and frozen calibration roles. |
| E-09 | Grouped CV/audit >= `0.92`, diagnostics, and gated division-policy evidence. |
| E-10 | Spend/concurrency/submission ledger passes and final displayed site score >= `0.92`. |

## Checkpoints and ownership

Run `T-02 → T-03 → T-04 → T-05 → T-06 → T-07 → T-08 → T-09 → T-10`, then the final independent, read-only `T-01`. Respect each task's PLAN dependencies and exclusive mutable resource owner; safe disjoint work may overlap only when ownership is unambiguous and the two-paid-trial concurrency ceiling permits it. `T-01` only assesses E-01 through E-10 and returns `PASS`, `BLOCK`, or `NOT_RUN`; it never fixes or weakens evidence.

Hard checkpoint rules include: freeze scorer/splits before fitting; add division only after edge-only grouped CV >= `0.90`, retaining it only if combined score improves and adjusted edge Jaccard declines <= `0.005`; promote only cache-identified 10/30/100-percent candidates that pass metric, graph, and runtime gates; reprofile after package/model/config identity changes; and require all local gates before a scored submission.

## Permissions and effects

Before Start, planning writes only are permitted; product writes, tests, network, Kaggle, paid compute, and every external effect are inactive. After confirmed Start, task-owned writes are limited to `src/celltrack_alpha`, `scripts`, `tests`, `kaggle_kernel`, `reports/g02`, and narrowly necessary configuration/manifests, exactly as each PLAN node further limits them. Preserve alpha and official schema/scorer semantics.

Network and Kaggle actions are only within the approved private competition/notebook/submission scope of T-08/T-10 (browser only after purpose-built authenticated access fails). No public notebook/model/dataset, external paid GPU, unrelated Kaggle or account action, credential change, git history mutation, commit, push, dense NxN association, or large from-scratch training is allowed. Private data flow is training-notebook output directly to private inference; no Kaggle dataset is created.

No more than two paid trials may run concurrently; development is capped at `$8/day` and `$80/month`, preserving `$2/day` and `$20/month` for final validation (platform maxima `$10/day`/`$100/month`); no more than one scored submission per day or ten total. Each paid/scored action needs its frozen artifact, local graph/package/runtime evidence, maximum-charge prediction, cap/concurrency/submission ledger entry, and recorded result.

The locked permissions define scope only. They do **not** grant a runtime sandbox, tool, browser, process, network, paid-compute, or external-effect approval: recheck the current environment and obtain any required time-of-use approval before that action. Never treat a runtime approval as authority to expand this contract.

## Interruptions and closeout

Immediately stop the affected node and preserve evidence for baseline or locked-artifact drift; permission expansion; scorer or split leakage; unavailable mandatory evidence; cap, quota, authentication, or submission barrier; unsafe rollback; or exhausted bounded remediation. Also stop when a locked decision must change, foreign/ambiguous drift appears, a required check is weakened or stale, or the only credible improvement is forbidden. Record the last valid identity, affected requirement/gate/decision, observed evidence, owner, and exact minimum resume prerequisite.

`CLOSEOUT.md` is created only after confirmed native activation (never by this compilation slice). The coordinator records actual attempts, changes, effects, evidence, and activation provenance there. Mark the Goal complete only when every mandatory AC and E gate is PASS and the full conjunctive stopping condition holds. Otherwise mark **Blocked**, preserve all work, and state the exact missing permission, evidence, decision, or remediation needed to resume; never infer an unavailable Kaggle score or lower a gate.
