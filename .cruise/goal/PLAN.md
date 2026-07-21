# G-02 Execution Plan

- State: Draft; execution requires separate Plan approval and Start.
- Global precondition: preserve the archived G-01 suite and unrelated worktree changes. No task may call a historical local audit or a CSV validity result a Kaggle score.
- Global effect boundary after Start: task-owned paths only under `src/celltrack_alpha`, `scripts`, `tests`, `kaggle_kernel`, `reports/g02`, plus narrowly necessary configuration/manifests. No task may write `GOAL.md` or `CLOSEOUT.md` in this preparation contract.

## T-02 — Freeze official scorer parity and leakage-safe splits

- State: Not started
- Owner: evaluation/data-integrity worker
- Dependencies: none
- Satisfies: R-01, R-02, R-06, R-08; AC-01, AC-02, AC-07
- Preserves: D-02, D-08, D-09; historical G-01 evidence and alpha baseline.
- Verification: E-01 scorer semantic/parity gate and E-02 split/audit leakage gate.
- Allowed paths/effects: `src/celltrack_alpha/**`, `scripts/**`, `tests/**`, `reports/g02/**`, and narrowly necessary fold/config manifests only. Local tests/benchmarks are allowed after Start; no network, Kaggle, package install, or product path outside this scope.
- Focused evidence: versioned scorer fixture report; official calibration-pair ledger with local/site fields explicitly null until observed; scale/matching/adjustment/division semantics test report; group-hash manifest; immutable three-fold membership; exact stratified 20-video audit membership; per-fit source provenance; baseline alpha report labeled local-only.
- Remediation bounds: one deterministic split/scorer-manifest correction pass for a discovered collision, leakage, or semantic mismatch; rerun only affected focused tests and regenerate affected reports. Any changed audit membership or scorer semantics invalidates downstream candidates.
- Stop rules: stop Blocked on ambiguous group ownership, leakage that cannot be resolved without discarding the audit, unavailable official metric semantics, or parity error > `0.02` on an available calibration pair after the bounded correction. Do not fit thresholds, adapters, router, or site correction on audit/validation labels.

## T-03 — Fine-tune checkpoint and compact multi-task specialist heads

- State: Not started
- Owner: model-training worker
- Dependencies: T-02
- Satisfies: R-02, R-03, R-06; AC-02, AC-07
- Preserves: D-01, D-02, D-03, D-08, D-10.
- Verification: E-02 fold isolation, E-03 checkpoint/head training gate, and E-06 manifest/accounting gate.
- Allowed paths/effects: `src/celltrack_alpha/**`, `scripts/**`, `tests/**`, `reports/g02/**`, `kaggle_kernel/**` only for a private training-notebook entrypoint, and narrowly necessary configuration/manifests. Private Kaggle training may use only approved competition/checkpoint data, remains private, and obeys D-04/D-05/D-07; no scored submission, public effect, external GPU, or dense NxN association.
- Focused evidence: checkpoint source URL/version/hash and license record; frozen input/fold hashes; model manifest with architecture and trainable parameter count; center/backward-displacement/uncertainty/division target definitions; sparse-negative weighting; per-fold loss and official local diagnostics; output tensor schema; deterministic restore smoke; predicted runtime/spend before each paid launch and actual ledger after it.
- Remediation bounds: at most the supplied `{0,1,2}` unfreeze depths and supplied learning-rate/sparse-negative-weight grid through the T-06 promotion funnel. Repair only loading/shape/provenance defects or lightweight-head loss wiring; do not replace the checkpoint, train a large model from scratch, or introduce a new task family.
- Stop rules: stop Blocked if the released checkpoint cannot be identified/legally used, a launch would breach D-04/D-05, fold leakage appears, checkpoint restore is not deterministic enough for manifesting, or the only credible improvement is forbidden large training. A paid launch without a pre-launch maximum-charge record is prohibited.

## T-04 — Build sparse candidate graphs and constrained global solving

- State: Not started
- Owner: association/solver worker
- Dependencies: T-02, T-03
- Satisfies: R-03, R-04, R-08; AC-02, AC-04, AC-05
- Preserves: D-01, D-02, D-08, D-09.
- Verification: E-03 output-head contract, E-04 graph-invariant gate, and E-07 sparse/runtime gate.
- Allowed paths/effects: `src/celltrack_alpha/**`, `scripts/**`, `tests/**`, `reports/g02/**`, and narrowly necessary configuration/manifests. Local tests/benchmarks only; no network or Kaggle effect.
- Focused evidence: physical KD-tree gating report for edge radii `4–10` um and neighbor caps `2/4/8`; maximal candidate graph cache identity; edge/division feature and scorer manifest; node/edge counts and cache memory; graph-validator report for unique IDs, forward time, one parent, at most two children, no merges, valid divisions, and complete dataset coverage; edge-only grouped CV diagnostic.
- Remediation bounds: tune only detection/NMS/edge radii, neighbor cap, edge threshold, birth/death cost, and gap frames in the supplied ranges; repair solver legality or candidate provenance defects. Candidate graph cache format must retain maximal gated candidates so later thresholds do not rerun inference.
- Stop rules: stop Blocked if an implementation materializes a full NxN association matrix, violates graph legality after one bounded solver/candidate repair, cannot use physical scale, or misses complete coverage. Do not enable division output as a retained feature until the T-06 edge-only gate is met.

## T-05 — Train specialists and out-of-fold router

- State: Not started
- Owner: routing/validation worker
- Dependencies: T-02, T-03, T-04
- Satisfies: R-02, R-05, R-06; AC-02, AC-03, AC-07
- Preserves: D-02, D-03, D-08, D-09.
- Verification: E-02 leakage gate, E-05 router gate, and E-06 manifest gate.
- Allowed paths/effects: `src/celltrack_alpha/**`, `scripts/**`, `tests/**`, `reports/g02/**`, and narrowly necessary configuration/manifests. Local evaluations only; no network, Kaggle, or package effect.
- Focused evidence: specialist assignment rationale and class counts; OOF-only router training rows; image-statistic and uncertainty feature schema; specialist oracle/selected/universal predictions per sample; confidence histogram; low-confidence (< `0.60`) fallback test; mean and per-scenario regret report; source-fold hashes.
- Remediation bounds: correct feature provenance, regularization, and compact router calibration using training folds only; retain exactly the three named specialists and one universal fallback. Re-run the OOF report after any router change.
- Stop rules: stop Blocked if route features leak labels or validation/audit identities, the router cannot produce the fixed universal fallback, regret stays above `0.01` mean or `0.03` worst scenario after one bounded calibration pass, or a proposed remedy changes specialist count/backbone scope.

## T-06 — Cache-aware successive halving and gated division promotion

- State: Not started
- Owner: optimization/integration worker
- Dependencies: T-02, T-03, T-04, T-05
- Satisfies: R-02, R-03, R-04, R-05, R-06, R-08; AC-02, AC-03, AC-04, AC-05, AC-07
- Preserves: D-01–D-05, D-08–D-10.
- Verification: E-02 through E-07 except external E-08/E-10.
- Allowed paths/effects: `src/celltrack_alpha/**`, `scripts/**`, `tests/**`, `reports/g02/**`, `kaggle_kernel/**` only for private non-public candidate integration, and narrowly necessary configuration/manifests. Paid private Kaggle trials obey D-04/D-05/D-07; no scored submission or dataset/publication action.
- Focused evidence: candidate registry with source/config/cache hashes; 10/30/100 percent stage results; graph and runtime gate for every candidate; pre-launch maximum-charge estimates and actual spend; concurrency timestamps; retained-cache hit/miss evidence; grouped fold diagnostics; division activation report showing edge-only score, combined-score delta, and adjusted-edge-Jaccard delta.
- Remediation bounds: evaluate only the locked search space and thresholds; maximum two paid trials concurrently; promote only candidates satisfying metric, graph, and runtime gates. The division branch may be added only after edge-only grouped CV >= `0.90`; retain it only when combined score improves and adjusted edge Jaccard declines <= `0.005`.
- Stop rules: stop Blocked if cache provenance is incomplete, a candidate would exceed a spend/concurrency cap, no candidate survives a stage because graph validity fails, edge-only score cannot reach `0.90` yet division is requested, or a change requires a new model family/dense association. Do not spend the D-05 reserve during development.

## T-07 — Measure runtime and create private production package

- State: Not started
- Owner: runtime/package worker
- Dependencies: T-06
- Satisfies: R-04, R-06, R-07; AC-04, AC-05, AC-07
- Preserves: D-04–D-10 and the alpha fallback.
- Verification: E-04 graph validity, E-06 reproducibility, and E-07 runtime/package gate.
- Allowed paths/effects: `src/celltrack_alpha/**`, `scripts/**`, `tests/**`, `kaggle_kernel/**`, `reports/g02/**`, and narrowly necessary configuration/manifests. Private notebook/package generation is permitted after Start; no Kaggle scored submission, dataset creation, public publication, or external GPU.
- Focused evidence: immutable package manifest and source/checkpoint/config hashes; dependency versions/lock changes; full graph/schema validator report; stage-level time and peak-memory profile for preprocessing, inference, candidate build, solve, and CSV generation; public four-clip elapsed time; measured 199-video/final-workload proxy and documented extrapolation; package restore/replay result.
- Remediation bounds: optimize streaming, batching, cache reuse, NMS/candidate caps, serialization, and package dependency wiring without changing frozen scorer/split semantics or reintroducing NxN association. Reprofile whenever package/model/config identity changes.
- Stop rules: stop Blocked when four-clip elapsed time exceeds seven minutes or the measured proxy projects above six hours after one bounded runtime pass, package identity cannot be reproduced, graph/schema validation fails, or required runtime depends on a forbidden provider/public artifact.

## T-08 — Run controlled private Kaggle experiments

- State: Not started
- Owner: Kaggle experiment worker
- Dependencies: T-07
- Satisfies: R-01, R-06, R-07; AC-01, AC-06, AC-07
- Preserves: D-04–D-10, all local gates, and site-score provenance.
- Verification: E-08 external experiment/score-provenance gate and E-10 spend/submission ledger gate.
- Allowed paths/effects: `kaggle_kernel/**`, `reports/g02/**`, narrowly necessary manifests/configuration, and approved private Kaggle competition/notebook/submission state only. May read official competition/notebook/submission state, update private notebooks, attach competition data/released public checkpoint, run private training/inference, upload qualifying CSVs, and read scores. Browser is limited to Kaggle competition/notebook/submission pages only when authenticated CLI/API access fails.
- Focused evidence: pre-score experiment card with artifact/package/model/fold hashes, local official prediction, graph/schema/runtime pass, daily/monthly spend projection, submission count, and expected maximum charge; Kaggle notebook/version/submit ID/timestamps/status/displayed score; daily and cumulative ledger; calibration/holdout role frozen before score observation.
- Remediation bounds: at most two concurrent paid trials; at most one scored submission per day and ten total. Use 10/30/100 percent local promotion before a site trial. A failed permission/auth/upload attempt may receive one diagnosis/retry only when no scored submission was created and no new boundary is crossed.
- Stop rules: stop Blocked for HTTP 403 or other absent submission permission after the bounded diagnosis, quota/cost/concurrency cap breach, unexpected account/payment/credential prompt, public/dataset action request, or missing local gate. Never declare a site score from a notebook log, local output, or absent submission record.

## T-09 — Calibrate site relationship and select final candidate

- State: Not started
- Owner: score-calibration analyst
- Dependencies: T-02, T-08
- Satisfies: R-01, R-02, R-06, R-07; AC-01, AC-02, AC-06, AC-07
- Preserves: D-02, D-06, D-09 and frozen pre-score roles.
- Verification: E-01 scorer/site-parity gate, E-02 leakage gate, E-08 external-provenance gate.
- Allowed paths/effects: `reports/g02/**`, `scripts/**`, `tests/**`, and narrowly necessary manifests/configuration. Read existing Kaggle score state only; no new Kaggle mutation, model/package change, or audit selection.
- Focused evidence: site/local pair table with artifact IDs and exact score labels; calibration versus untouched holdout designation frozen before return; residuals; any approved local correction fitted only on calibration pairs; re-scored holdout residuals; candidate ranking with grouped-CV/audit/runtime evidence kept separate from site score.
- Remediation bounds: use identity, bias, or affine correction only if predeclared and supported by calibration pairs; do not refit or reassign holdouts after observing them. A correction is advisory for promotion and never substitutes for AC-06.
- Stop rules: stop Blocked if no legitimate Kaggle score exists, any pair lacks an artifact identity, residual exceeds `0.02` on an available holdout and cannot be corrected within the frozen protocol, or a proposed action requires extra submissions beyond D-06. Do not claim local-only success.

## T-10 — Produce final local/audit/package evidence and qualifying submission

- State: Not started
- Owner: release/integration worker
- Dependencies: T-02, T-06, T-07, T-08, T-09
- Satisfies: R-01 through R-08; AC-01 through AC-07
- Preserves: D-01 through D-10, especially alpha fallback and external boundaries.
- Verification: E-01 through E-10 before handoff to T-01.
- Allowed paths/effects: `src/celltrack_alpha/**`, `scripts/**`, `tests/**`, `kaggle_kernel/**`, `reports/g02/**`, narrowly necessary configuration/manifests, and the tightly scoped private Kaggle submission/notebook actions permitted by D-04–D-07. A final scored submission is allowed only when its daily/total/cost ledger and all local gates are already PASS; if T-08 already produced the qualifying exact final artifact, T-10 verifies it rather than resubmitting.
- Focused evidence: final immutable manifest; complete per-sample grouped-fold and untouched-audit node/edge/division diagnostics; fixed scorer and split hashes; router and fallback report; graph/schema/package replay; four-clip and 199-video proxy profile; spend/submission ledger; Kaggle submission ID/status/displayed score tied to final artifact.
- Remediation bounds: one integration repair for a manifest/package/schema/report linkage defect, followed by re-running only invalidated mandatory local gates. Model/HPO changes return to T-06; score-calibration changes return to T-09; no blind final resubmission.
- Stop rules: stop Blocked unless grouped CV >= `0.92`, audit >= `0.92`, graph/package/runtime gates pass, scorer parity meets AC-01, and a qualifying Kaggle submission has displayed score >= `0.92`. Stop on any cap/boundary failure rather than lowering a gate or consuming an unplanned score.

## T-01 — Final read-only adversarial verification

- State: Not started
- Owner: independent verification worker
- Dependencies: T-02, T-03, T-04, T-05, T-06, T-07, T-08, T-09, T-10
- Satisfies: R-01 through R-08; AC-01 through AC-07
- Preserves: all locked decisions, archive integrity, unrelated changes, and final evidence immutability.
- Verification: independently assesses E-01 through E-10.
- Allowed paths/effects: read-only inspection of `.cruise/goal/**`, archived G-01, allowed source/report/package manifests, and already-created Kaggle evidence. No repository write, test execution, network call, Kaggle action, package install, or remediation.
- Focused evidence: origin raw-prompt/hash and archive comparison; decision/permission audit; split/audit immutability; scorer semantics and pair residuals; checkpoint/package/fold/config hashes; graph/runtime/router/HPO records; spend/submission counts; final Kaggle submission ID/status/displayed score; explicit PASS/BLOCK/NOT_RUN for E-01 through E-10.
- Remediation bounds: report one precise owner and the minimum upstream task to revisit for each failed gate. It may request one re-verification after that remediation; it may not edit evidence or relax criteria.
- Stop rules: PASS only when every mandatory gate passes and the displayed site score and both local endpoints are >= `0.92`. Otherwise return BLOCK or NOT_RUN with exact evidence; never infer unavailable evidence or modify any artifact.

## Mandatory gates

| Gate | Threat prevented | Pass condition |
| --- | --- | --- |
| E-01 Official scorer and site parity | Semantic drift or local/site conflation | Official fixture passes and every available frozen pair residual <= `0.02`. |
| E-02 Split/audit integrity | Near-duplicate, fit-source, or audit leakage | Immutable group/fold/audit manifest and provenance checks pass. |
| E-03 Checkpoint/head contract | Wrong or untraceable learned core | Released checkpoint/hash, lightweight head schema, fold isolation, and restore evidence pass. |
| E-04 Sparse graph validity | Dense association, illegal lineage, incomplete coverage | KD-tree/candidate evidence and every graph constraint/coverage check pass. |
| E-05 Router | Specialist selection leakage or excess regret | OOF regret/fallback thresholds pass. |
| E-06 HPO/reproducibility | Unbounded search or irreproducible promotion | Cache identity, 10/30/100 ledger, manifest, and pre-launch accounting pass. |
| E-07 Runtime/package | Hidden runtime/package failure | Four-clip and measured proxy thresholds plus package replay pass. |
| E-08 Kaggle provenance/calibration | Fabricated/misattributed site score | Private artifact-linked submissions/scores and frozen calibration roles pass. |
| E-09 Local quality/division | Early/divergent division or insufficient local quality | Grouped CV/audit >= `0.92`; division policy and diagnostics pass. |
| E-10 External limits and endpoint | Overspend/quota breach or local-only completion | D-04/D-05/D-06 ledger passes and final displayed score >= `0.92`. |
