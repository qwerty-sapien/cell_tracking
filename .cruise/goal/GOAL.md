# Goal: G-02 — Checkpoint-Based Sparse Hybrid Lineage Tracker

- State: **Active**. Start was approved on 2026-07-16 and is recorded in
  `.cruise/runs/sparse-hybrid/goal-appr.yaml` and `activation.yaml`.
- Plan approval: `.cruise/runs/sparse-hybrid/plan-ap2.yaml`; the 2026-07-24
  coordinator amendment adopts the revised screening/calibration DAG and updates D-06/D-10.
- Baseline: `c3b9b703b049fc25f10e2b42bb986bb94f137a72`; dirty-state digest `6c4d0a9af20a760e7ff6dbde5dcdae0c3f0f72bc43c8b2c1336dbcaf5238ca6d`.

## Contract identities and first reads

Before any native execution action, read in this order: `ORIGIN.yaml`, `ALIGNMENT.md`, `PLAN.md`, and this `GOAL.md`. `SPEC.md` and `DECISIONS.md` are normative detail; do not reinterpret them. Revalidate the raw-prompt hash and all locked artifact identities:

The 2026-07-16 approval receipt preserves the pre-amendment hashes. The following are the
current coordinator-amended identities adopted on 2026-07-24:

| Artifact | Current SHA-256 |
| --- | --- |
| `ORIGIN.yaml` | `e793becedadcf7be418c90e354c54ccd5a2c1d091624d5addacf78f2ca70d42c` |
| `SPEC.md` | `4f936c5420b64a3b8c1886c1fcfd544862b76cfec04ca57098c97e207ab4d64b` |
| `DECISIONS.md` | `cf3cad10475efd998f42c7eab2f6e4d2be7a27c78509dba906416bddaf9bda31` |
| `ALIGNMENT.md` | `a725011b533c472a6bc7145418c5148a684fc36da3e4a9afc3a34a68230bd68c` |
| `ROADMAP.md` | `03c49bfa5104963dc3cef3d161ee03ec0d7fee5a52bd79de5e167c9721d5c3ff` |
| `PLAN.md` | `9dc4a60084226c46144cfd90121f5bac8fbcbcd261303584804ab3858e6c8b4e` |

Preserve these bytes, the G-01 archive, and all accepted pre-existing worktree changes.
Unexplained mismatch is an interruption; a later explicit coordinator amendment must
record replacement identities.

## Objective and stopping condition

Deliver the approved low-compute tracker: fine-tune the released official
`TemporalUNet3D` only with compact heads/adapters; use physical-coordinate
KD-tree-gated sparse learned association and constrained global solving; retain the
rule-based alpha as teacher, fallback, and regression baseline. The immediate milestone
is a materially stronger baseline above 0.88; the next is robust performance in the
competitive 0.90s. These milestones guide prioritization without lowering the 0.92
completion threshold.

Completion is conjunctive: grouped three-fold official local CV **>= 0.92**, untouched stratified 20-video audit **>= 0.92**, displayed qualifying private Kaggle score **>= 0.92**, exact official scorer semantic parity, a decision-grade interval-aware local-to-Kaggle transfer certificate, all graph/router/package gates PASS, and measured final-workload projection **<= 6 h** (with public four-clip time **<= 7 min**). Local-only success is not completion. The current historical baseline is **0.8713516075529171, local-only**; it is neither calibration nor Kaggle evidence.

## Coordinator, state, and execution loop

The root coordinator owns activation checks, DAG state, integration, foreign-diff review, and the sole `CLOSEOUT.md` write. It assigns `recommended_level: 4` and uses the Doctor-attested Cruise worker for every substantive planning, execution, and verification task. Workers return immutable attempt reports and never update closeout directly.

For every Ready node, use the fast loop: **orient → smallest micro-plan → preflight ownership/effects → implement → focused deterministic evidence → preserve attempts → integrate → checkpoint**. The micro-plan records inputs, exclusive writes/resources, smallest change, exact focused check, effect authority, and rollback/stop. A node completes only with current output identities and its focused evidence; a failed or stale node never advances on assertions alone.

Maintain durable progress memory under `.cruise/runs/sparse-hybrid/`: node task state
and attempts; artifact hashes; experiment and promotion ledgers; spend and
scored-submission counters; runtime profiles; frozen calibration/holdout roles; blockers;
foreign-diff incidents; and the next Ready node. Keep source, package, model, fold, cache,
configuration, score, and spend identities joinable. Preserve every failed attempt and
invalidate dependent evidence when an upstream identity changes.

## Scorecard

| Acceptance | Mandatory PASS evidence |
| --- | --- |
| AC-01 | Official semantic fixtures and source hashes pass exactly; separately, any transfer model used for public-score decisions meets its preregistered interval-aware out-of-sample bound. |
| AC-02 | Grouped 3-fold and untouched 20-video audit both >= `0.92`, with per-sample node/edge/division diagnostics. |
| AC-03 | OOF router mean regret <= `0.01`, worst scenario <= `0.03`, and universal fallback test. |
| AC-04 | Valid, complete graphs: unique IDs, forward time, legal degree/divisions, no merges, full coverage. |
| AC-05 | Four clips <= `7 min`; measured 199-video/final-workload proxy <= `6 h` end to end. |
| AC-06 | Artifact-linked private Kaggle submission is complete with displayed score >= `0.92`. |
| AC-07 | Immutable manifest joins source/checkpoint/fold/config/package/runtime/scores and pre/post-trial spend. |

| Gate | Mandatory PASS evidence |
| --- | --- |
| E-01 | Official scorer semantic fixtures and frozen scorer identity. Transfer calibration is tracked separately in E-08. |
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

Use the revised controller in `reports/g02/screening/DAG.md`. Scorer semantics and
split integrity gate local development. Transfer calibration runs in parallel and gates
public-score inference, probe interpretation, and final selection. The active learned-model
path remains `T-03 → T-04 → T-05 → T-06 → T-07`; controlled public measurement maps to
T-08/T-09, integration to T-10, and final independent verification to T-01. Respect each
task's ownership and effect bounds.

Hard checkpoint rules include: freeze scorer/splits before fitting; add division only after edge-only grouped CV >= `0.90`, retaining it only if combined score improves and adjusted edge Jaccard declines <= `0.005`; promote only cache-identified 10/30/100-percent candidates that pass metric, graph, and runtime gates; reprofile after package/model/config identity changes; and require all local gates before a scored submission.

## Permissions and effects

Before Start, planning writes only are permitted; product writes, tests, network, Kaggle, paid compute, and every external effect are inactive. After confirmed Start, task-owned writes are limited to `src/celltrack_alpha`, `scripts`, `tests`, `kaggle_kernel`, `reports/g02`, and narrowly necessary configuration/manifests, exactly as each PLAN node further limits them. Preserve alpha and official schema/scorer semantics.

Network and Kaggle actions are only within the approved private
competition/notebook/submission scope of their PLAN nodes (browser only after
purpose-built authenticated access fails). Ordinary commits and pushes to this private
repository are allowed. No public notebook/model/dataset, unrelated Kaggle or account
action, credential change, destructive git-history rewrite, dense NxN association, or
large from-scratch training is allowed. RunPod requires explicit human approval for each
launch after the job's artifact identity, estimated GPU-hours, and maximum dollar cost
are recorded. Private data flow is training-notebook output directly to private inference;
no Kaggle dataset is created.

No more than two paid trials may run concurrently; development is capped at `$8/day` and `$80/month`, preserving `$2/day` and `$20/month` for final validation (platform maxima `$10/day`/`$100/month`). Initial training is capped at 30 cumulative Kaggle T4 GPU-hours per model, and models that cannot become viable inside that ceiling are archived. RunPod has a `$10` total campaign cap and is unavailable before an accepted score reaches `0.89`; any later higher-quality-GPU-dependent optimization requires literature or theoretical support for an expected score improvement greater than `5%` (ordinary hard improvement gate `3.5%`). No more than five Kaggle submissions may be made per competition day, and at most two final submissions may be selected. Each paid/scored action needs its frozen artifact, local graph/package/runtime evidence, maximum-charge prediction, cap/concurrency/submission ledger entry, and recorded result.

New learned candidates follow a literature-first lightweight fast-fail funnel. A
primary-evidence manifest and fit-only mechanism test must pass before training. Keep at
most one guarded M5 training lane available alongside eligible external work, limited to
`18 GiB` training memory with `6 GiB` reserved for background work, zero holdout access,
zero swap growth, and seven hours maximum per unattended stage. Local promotion does not
authorize or expand Kaggle or RunPod use.

The locked permissions define scope only. They do **not** grant a runtime sandbox, tool,
browser, process, network, paid-compute, or external-effect approval: recheck the current
environment and obtain any required time-of-use approval before that action. In particular,
each RunPod launch requires a fresh explicit human approval. Ordinary private-repository
commits and pushes do not require a separate phase approval.

## Interruptions and closeout

Immediately stop the affected node and preserve evidence for baseline or locked-artifact drift; permission expansion; scorer or split leakage; unavailable mandatory evidence; cap, quota, authentication, or submission barrier; unsafe rollback; or exhausted bounded remediation. Also stop when a locked decision must change, foreign/ambiguous drift appears, a required check is weakened or stale, or the only credible improvement is forbidden. Record the last valid identity, affected requirement/gate/decision, observed evidence, owner, and exact minimum resume prerequisite.

`CLOSEOUT.md` is created only after confirmed native activation (never by this compilation slice). The coordinator records actual attempts, changes, effects, evidence, and activation provenance there. Mark the Goal complete only when every mandatory AC and E gate is PASS and the full conjunctive stopping condition holds. Otherwise mark **Blocked**, preserve all work, and state the exact missing permission, evidence, decision, or remediation needed to resume; never infer an unavailable Kaggle score or lower a gate.
