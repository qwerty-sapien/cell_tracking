# G-02 Locked Decisions

- Planning state: Draft, pre-Start
- Authority: coordinator-supplied approved direction and locked-decision packet
- Reopening rule: only an explicit new coordinator decision may reopen a locked item. Runtime/sandbox/Kaggle approvals are still required whenever surfaced.

| ID | Status | Decision |
| --- | --- | --- |
| D-01 | Locked | Checkpoint plus sparse hybrid is the major direction. |
| D-02 | Locked | Grouped three-fold funnel validation plus untouched stratified 20-video audit. |
| D-03 | Locked | Shared backbone plus three lightweight specialist adapters/heads. |
| D-04 | Locked | At most two concurrent paid Kaggle trials. |
| D-05 | Locked | Autonomous Kaggle compute inside development/reserve caps. |
| D-06 | Locked | At most one scored submission/day and ten total. |
| D-07 | Locked | Private-notebook handoff only; no Kaggle dataset creation. |
| D-08 | Locked | Preserve alpha teacher/fallback; no full NxN association. |
| D-09 | Locked | Both local official evidence and displayed Kaggle site score are authoritative completion evidence. |
| D-10 | Locked | No public model, unrelated Kaggle action, git history mutation/push, or external paid GPU. |

## D-01 — Fine-tuned released checkpoint plus sparse hybrid

Use the released official `TemporalUNet3D` checkpoint as the model core. Add sparse center, backward-displacement, uncertainty, division, and gated-edge outputs; fine-tune compact heads/adapters and permitted backbone depths only. Replace the dense transformer all-pairs association stage with KD-tree-gated sparse learned association and global constrained solving. Large from-scratch training is prohibited.

## D-02 — Leakage-safe validation funnel

Hash-group near duplicates, stratify scenario groups, hold 20 videos untouched, cheaply screen candidates, and promote only qualifying candidates to grouped three-fold CV. Folds, group hashes, audit membership, fit-source partitions, and scenario labels are immutable per manifest version. The audit is never used for threshold/HPO/router calibration.

## D-03 — Three specialists with one backbone

The adapters/heads are exactly `sparse_clean`, `dense_crowded`, and `low_snr_depth`, backed by one shared checkpoint. An out-of-fold scenario router uses image statistics and model uncertainty. Confidence below `0.60` selects the universal adapter, and its effect must be measured.

## D-04 — Paid-concurrency ceiling

No more than two paid Kaggle trials may run concurrently. Training/inference launches must serialize any shared notebook/output/package identity and ledger update. Free local work is still bounded by the runtime gates; it does not bypass validation requirements.

## D-05 — Spend accounting and reserve

Development spend is capped at `$8/day` and `$80/month`. `$2/day` and `$20/month` are reserved for final validation, so ordinary development may not consume that reserve. The platform maxima of `$10/day` and `$100/month` are never exceeded. Before launch, record rate, maximum charge, duration, concurrency, projected cumulative daily/monthly spend, and reserve remaining; reject a launch when any projected total exceeds its applicable cap.

## D-06 — Scored-submission ceiling

Use at most one scored Kaggle submission on a calendar day and ten total. A submission becomes scored only when Kaggle creates it; failed upload/auth/permission attempts are logged but do not count as scored. Each actual score is attached to exactly one artifact hash and local prediction; blind resubmits are prohibited.

## D-07 — Private Kaggle notebook data flow

Use private training and inference notebooks only. Pass training-notebook outputs directly to a private inference notebook. Do not create or publish a Kaggle dataset, public notebook, public model, or other public artifact. Attach only competition data and the released public checkpoint within the approved scope.

## D-08 — Alpha preservation and sparse association

The current rule-based alpha is retained as a teacher, fallback, and regression baseline. It must remain available to the universal fallback and ablations. No task may depend on a full NxN association matrix; candidate count, KD-tree radius, neighbor cap, and graph memory are first-class runtime evidence.

## D-09 — Completion source of truth

The endpoint requires both local official metric evidence and a qualifying displayed Kaggle site score. The historical `0.8713516075529171` is local rule-based audit evidence only. A local score, calibration fit, CSV validator pass, notebook completion, or external report does not substitute for a Kaggle site score.

## D-10 — External and history boundaries

Do not publish a model, use an external paid GPU provider, create unrelated Kaggle/account effects, mutate git history, commit, or push. Do not expose credential values. Browser/Chrome access, if needed, stays within Kaggle competition/notebook/submission pages and only when the purpose-built CLI/API lacks authenticated access.

## Search and promotion policy

The permitted search axes are: learning rate `{1e-5, 3e-5, 1e-4}`; sparse-negative weight `{0.001, 0.01, 0.05}`; detection threshold `0.20–0.70`; NMS radius `2–5` um; edge radius `4–10` um; neighbor cap `{2, 4, 8}`; edge threshold `0.20–0.80`; birth/death cost `0.05–0.50`; gap frames `{1, 2}`; division threshold `0.30–0.80`; and backbone unfreeze depth `{0, 1, 2}`.

All candidates begin at 10 percent cached screening, then 30 percent, then 100 percent grouped validation. A promotion requires current-stage metric improvement or a predeclared diagnostic purpose, graph validity, and predicted/runtime gate pass. Add divisions only after edge-only grouped CV >= `0.90`; retain them only if combined score improves and adjusted edge Jaccard drops by no more than `0.005`.

## Permission matrix

| Capability | Decision | Boundary |
| --- | --- | --- |
| Planning writes | Packet | Only listed `.cruise/goal` ownership in this preparation slice. |
| Product writes after Start | Packet | `src/celltrack_alpha`, `scripts`, `tests`, `kaggle_kernel`, `reports/g02`, narrowly necessary configuration/manifests. |
| Local tests/benchmarks | Packet | Allowed only after Start and only to produce task evidence. |
| Network reads | Packet | Official competition/Kaggle, primary literature, official package/model docs, required package metadata. |
| Private Kaggle notebooks and submissions | D-04–D-07 | Competition/notebook/submission scope, caps and runtime approvals apply. |
| Paid compute | D-04–D-05 | Predicted maximum charge within development/reserve/platform caps; two concurrent maximum. |
| Browser/Chrome | D-10 | Kaggle pages only, after CLI/API cannot access authenticated state. |
| External GPU, public publishing, git history/remote | D-10 | Prohibited. |
