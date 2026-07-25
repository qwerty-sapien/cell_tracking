# G-02 Locked Decisions

- Planning state: Active; amended by explicit coordinator direction on 2026-07-24
- Authority: coordinator-supplied approved direction and locked-decision packet
- Reopening rule: only an explicit new coordinator decision may reopen a locked item. The
  2026-07-24 direction reopened D-06, D-10, and the calibration/DAG policy. Runtime,
  sandbox, Kaggle, and RunPod approvals are still required whenever surfaced.

| ID | Status | Decision |
| --- | --- | --- |
| D-01 | Locked | Checkpoint plus sparse hybrid is the major direction. |
| D-02 | Locked | Grouped three-fold funnel validation plus untouched stratified 20-video audit. |
| D-03 | Locked | Shared backbone plus three lightweight specialist adapters/heads. |
| D-04 | Locked | At most two concurrent paid compute trials across Kaggle and RunPod. |
| D-05 | Locked | Autonomous Kaggle compute inside development/reserve caps. |
| D-06 | Locked | At most five Kaggle submissions/day and two final selections. |
| D-07 | Locked | Private-notebook handoff only; no Kaggle dataset creation. |
| D-08 | Locked | Preserve alpha teacher/fallback; no full NxN association. |
| D-09 | Locked | Both local official evidence and displayed Kaggle site score are authoritative completion evidence. |
| D-10 | Amended | Commits and pushes to this private repository are permitted; no history rewrite or public release. RunPod requires explicit human approval for each launch. |
| D-11 | Locked | Scorer semantic parity gates all local scoring; local-to-Kaggle transfer calibration gates public inference and final selection, not local model development. |
| D-12 | Locked | The campaign objective is a materially stronger baseline: first break 0.88, then pursue robust scores in the competitive 0.90s; the contractual completion target remains 0.92. |
| D-13 | Locked | Initial training is capped at 30 cumulative Kaggle T4 GPU-hours per model. Archive models that cannot become viable inside that ceiling. RunPod is capped at $10 total, is unavailable before an accepted score reaches 0.89, and thereafter requires literature/theory-backed expected score improvement greater than 5%. The hard baseline improvement gate is 3.5%. |
| D-14 | Locked | Prefer literature-validated lightweight fast-fail candidates. Maintain one fit-only M5 training lane in parallel with eligible external work, capped at 18 GiB with at least 6 GiB reserved for background work, no holdout access, no swap growth, and at most seven hours per unattended stage. |

## D-01 — Fine-tuned released checkpoint plus sparse hybrid

Use the released official `TemporalUNet3D` checkpoint as the model core. Add sparse center, backward-displacement, uncertainty, division, and gated-edge outputs; fine-tune compact heads/adapters and permitted backbone depths only. Replace the dense transformer all-pairs association stage with KD-tree-gated sparse learned association and global constrained solving. Large from-scratch training is prohibited.

## D-02 — Leakage-safe validation funnel

Hash-group near duplicates, stratify scenario groups, hold 20 videos untouched, cheaply screen candidates, and promote only qualifying candidates to grouped three-fold CV. Folds, group hashes, audit membership, fit-source partitions, and scenario labels are immutable per manifest version. The audit is never used for threshold/HPO/router calibration.

## D-03 — Three specialists with one backbone

The adapters/heads are exactly `sparse_clean`, `dense_crowded`, and `low_snr_depth`, backed by one shared checkpoint. An out-of-fold scenario router uses image statistics and model uncertainty. Confidence below `0.60` selects the universal adapter, and its effect must be measured.

## D-04 — Paid-concurrency ceiling

No more than two paid compute trials may run concurrently across Kaggle and RunPod.
Training/inference launches must serialize any shared notebook/output/package identity and
ledger update. Free local work is still bounded by the runtime gates; it does not bypass
validation requirements.

## D-05 — Spend accounting and reserve

Development spend is capped at `$8/day` and `$80/month`. `$2/day` and `$20/month` are reserved for final validation, so ordinary development may not consume that reserve. The platform maxima of `$10/day` and `$100/month` are never exceeded. Before launch, record rate, maximum charge, duration, concurrency, projected cumulative daily/monthly spend, and reserve remaining; reject a launch when any projected total exceeds its applicable cap.

RunPod additionally has a hard campaign sub-cap of `$10` total. It is not an initial
training platform and is unavailable until D-13's score and evidence gates pass. A
specifically approved launch may use less than the remaining sub-cap; no approval silently
raises either the RunPod sub-cap or the overall D-05 caps.

## D-06 — Scored-submission ceiling

Use at most five Kaggle submissions on a competition day and select at most two final submissions. A submission counts when Kaggle creates it; failed upload/auth/permission attempts are logged but do not count. Each actual score is attached to exactly one artifact hash and local prediction.

## D-07 — Private Kaggle notebook data flow

Use private training and inference notebooks only. Pass training-notebook outputs directly to a private inference notebook. Do not create or publish a Kaggle dataset, public notebook, public model, or other public artifact. Attach only competition data and the released public checkpoint within the approved scope.

## D-08 — Alpha preservation and sparse association

The current rule-based alpha is retained as a teacher, fallback, and regression baseline. It must remain available to the universal fallback and ablations. No task may depend on a full NxN association matrix; candidate count, KD-tree radius, neighbor cap, and graph memory are first-class runtime evidence.

## D-09 — Completion source of truth

The endpoint requires both local official metric evidence and a qualifying displayed Kaggle site score. The historical `0.8713516075529171` is local rule-based audit evidence only. A local score, calibration fit, CSV validator pass, notebook completion, or external report does not substitute for a Kaggle site score.

## D-10 — Version control and external-compute boundaries

Ordinary commits and pushes to this private repository are permitted, including phase
checkpoints. Do not rewrite shared history, publish a model/dataset/notebook, create
unrelated Kaggle/account effects, or expose credential values. Private Kaggle actions
remain limited to their task scopes. RunPod is permitted only after a specific human
approval that identifies the job, estimated GPU-hours, maximum dollar cost, and artifact
identity; approval for one launch does not authorize another. Browser/Chrome access, if
needed, stays within Kaggle competition/notebook/submission pages and only when the
purpose-built CLI/API lacks authenticated access.

## D-11 — Separate scorer correctness from transfer calibration

Official scorer semantic parity is the prerequisite for local scores, decomposition, error
taxonomy, candidate development, and local GPU gates. The empirical mapping from frozen
local scores to rounded Kaggle scores is a separate transfer-calibration workstream. An
unresolved transfer model blocks fine public-score forecasting, interpretation of public
probes, and final portfolio selection; it does not halt leakage-safe local development.
Every public probe must be artifact-linked, observation-distinct, sequential, and charged
against D-06.

## D-12 — Performance objective

Optimize for a robust, materially stronger tracker rather than calibration alone. The
immediate milestone is to exceed 0.88 with the frozen local scorer and an artifact-linked
Kaggle result. The next milestone is a generalizing score in the 0.90s. The final G-02
completion threshold remains 0.92 locally, on the untouched audit, and on Kaggle; milestones
are progress evidence and do not lower acceptance criteria.

## D-13 — Per-model compute viability and RunPod escalation

Initial training for any one model may consume at most 30 cumulative Kaggle T4 GPU-hours,
across all sessions and quota resets. Before launch, project time to the model's minimum
credible training state. Archive the model when that state cannot be reached inside the
ceiling; local hardware does not bypass this viability gate.

RunPod may be considered only for a higher-quality-GPU-dependent optimization after an
artifact-linked accepted score has reached at least `0.89`. The exact optimization must
have literature validation or a defensible theoretical mechanism supporting an expected
score improvement greater than `5%`; the ordinary hard baseline improvement gate is
`3.5%`. RunPod is never used for speculative initial training, remains capped at `$10`
total, and still requires explicit approval for each exact launch.

## D-14 — Literature-first local fast-fail lane

Prefer lightweight candidates that can be falsified cheaply on the Apple M5 before they
consume external GPU quota. Before implementation or training, each candidate must have a
machine-validated manifest containing at least one primary peer-reviewed source, an
explicit mapping from a measured fit-only failure mode to the proposed mechanism, a
falsification test, an expected improvement meeting the `3.5%` hard gate, and
predeclared promotion and early-stop criteria.

Maintain at most one guarded local training job, and run that lane in parallel with
eligible Kaggle or RunPod work when useful. Local training is fit-only and may never
inspect the holdout. Cap the training process tree at `18 GiB`, reserve at least `6 GiB`
for background work, stop on any swap growth, and end every unattended stage within
seven hours. A failed literature, learning, memory, swap, or time gate is recorded and
the candidate is resized, parked, or archived rather than silently escalated. This lane
does not weaken D-13: Kaggle remains capped at 30 cumulative T4 GPU-hours per model and
RunPod retains every score, evidence, budget, and exact-approval gate.

## Search and promotion policy

The permitted search axes are: learning rate `{1e-5, 3e-5, 1e-4}`; sparse-negative weight `{0.001, 0.01, 0.05}`; detection threshold `0.20–0.70`; NMS radius `2–5` um; edge radius `4–10` um; neighbor cap `{2, 4, 8}`; edge threshold `0.20–0.80`; birth/death cost `0.05–0.50`; gap frames `{1, 2}`; division threshold `0.30–0.80`; and backbone unfreeze depth `{0, 1, 2}`.

All candidates begin at 10 percent cached screening, then 30 percent, then 100 percent grouped validation. A promotion requires current-stage metric improvement or a predeclared diagnostic purpose, graph validity, and predicted/runtime gate pass. Add divisions only after edge-only grouped CV >= `0.90`; retain them only if combined score improves and adjusted edge Jaccard drops by no more than `0.005`.

## Permission matrix

| Capability | Decision | Boundary |
| --- | --- | --- |
| Planning/controller writes | Packet | Root coordinator owns shared goal/DAG/ledger state. |
| Product writes | Packet | `src/celltrack_alpha`, `scripts`, `tests`, `kaggle_kernel`, `reports/g02`, narrowly necessary configuration/manifests. |
| Local tests/benchmarks | Packet | Allowed to produce task evidence under the active Plan. |
| Guarded local M5 training | D-14 | One fit-only job; validated literature manifest; 18 GiB process-tree cap; >=6 GiB reserve; no swap growth; <=7 hours unattended; no holdout access. |
| Network reads | Packet | Official competition/Kaggle, primary literature, official package/model docs, required package metadata. |
| Private Kaggle notebooks and submissions | D-04–D-07 | Competition/notebook/submission scope, caps and runtime approvals apply. |
| Paid compute | D-04–D-05 | Predicted maximum charge within development/reserve/platform caps; two concurrent maximum. |
| Browser/Chrome | D-10 | Kaggle pages only, after CLI/API cannot access authenticated state. |
| Private-repository commit/push | D-10 | Permitted; no destructive history rewrite or public release. |
| RunPod | D-10/D-13 | Only after score >=0.89 and >5% literature/theory-backed expected improvement; $10 total cap; per-launch explicit approval plus pre-launch hours/$/artifact ledger. |
| Public publishing, unrelated external effects, history rewrite | D-10 | Prohibited. |
