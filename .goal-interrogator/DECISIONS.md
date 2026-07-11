# Decision Ledger

- Planning state: Draft
- Last reconciled: 2026-07-11
- Next decision ID: D-09

## Status Index

| ID | Short Title | Status | Affected IDs | Supersession |
| --- | --- | --- | --- | --- |
| D-01 | Success Target And CV Protocol | Locked | R-01, R-02, R-11, AC-01, AC-02, AC-10, G-01 | - |
| D-02 | Duplicate Route Boundary | Locked | R-03, R-07, AC-03, G-01 | - |
| D-03 | Offset And Localizer Leakage Policy | Locked | R-04, R-05, AC-03, AC-04, G-01 | - |
| D-04 | Primary Direction And Hybrid Branches | Locked | R-04, R-06, AC-04, AC-05, G-01 | - |
| D-05 | Compute And Environment Boundary | Locked | R-08, AC-07, G-01 | - |
| D-06 | Division Work Gate | Locked | R-06, AC-05, G-01 | - |
| D-07 | Reproducibility Baseline | Proposed | R-10, AC-09, G-01 | - |
| D-08 | Kaggle Scorer Parity | Locked | R-02, R-11, AC-02, AC-10, G-01 | - |

## D-01 - Success Target And CV Protocol

- **Status:** Locked
- **Question:** What evidence defines reaching `0.85` for the non-duplicate/general model?
- **Context and evidence:** The current duplicate route already scores `1.1936`, but the best general fallback evidence is `0.6283` on one full train video. Official metric wrappers exist; Kaggle discussion/leaderboard pages were not fetchable with reliable content. The user also requested local scorer parity with Kaggle scoring for iterative testing.
- **Options considered:**
  - **A - Official local non-duplicate CV is the gate (selected):** Predeclare video/dataset-level held-out splits across `44b6` and `6bba`; use official local metric only; treat public/private Kaggle as separate external evidence.
  - **B - Public leaderboard is the gate:** Faster alignment with Kaggle but risks duplicate shortcut and overfitting to public split.
  - **C - Full train self-score is the gate:** Easier to run but weak for generalization and offset leakage.
- **Decision or temporary assumption:** Official local non-duplicate CV is the gate. The activated work must also update/verify the local scorer so iterative local testing matches the Kaggle competition scoring semantics.
- **Rationale:** Official local CV is the only measurable, leakage-controllable way to improve a non-duplicate model before external submission.
- **Consequences:** Requires split manifest, strict no-label leakage, official metric evidence, scorer parity checks, and potentially slower iterations than public LB probing.
- **Affected IDs and artifacts:** R-01, R-02, R-11, AC-01, AC-02, AC-10, ROADMAP G-01, PLAN VG-01/VG-04/VG-11, GOAL.
- **Approval owner:** User.
- **Approval record:** 2026-07-11 user selected "A" and added "also update the local scorer to match how the kaggle competition scores, ensuring parity and enabling iterative testing w accuracy/alignment".
- **Reopening trigger:** User selects a different score target, Kaggle releases/verifies hidden/public evidence that changes the risk calculus, or the official metric changes.
- **Supersedes:** None.
- **Superseded by:** None

## D-02 - Duplicate Route Boundary

- **Status:** Locked
- **Question:** How should the existing duplicate-GEFF production route be treated during this Goal?
- **Context and evidence:** `reports/duplicate_geff_submission_report.json` proves the current four public/test datasets match train Zarrs and score `1.1936` through GEFF copying. The route detects train/test image identity and emits the paired train GEFF graph as the test prediction. This is useful for the current ZIP but not a general model.
- **Options considered:**
  - **A - Preserve but quarantine:** Keep duplicate route intact for current public ZIP; add guards so it cannot contribute to general-model CV or `>=0.85` evidence.
  - **B - Disable duplicate route during all execution:** Lower leakage risk but removes useful production shortcut and regression target.
  - **C - Remove duplicate route:** Delete or remove the simple duplicate-copy route from the current execution path because it is not a model.
- **Decision or temporary assumption:** Remove or disable the duplicate-copy route for this Goal. If activated, implementation may delete the simple GEFF-copy route from the general model path after confirming no protected non-duplicate behavior depends on it; historical reports/submissions may remain as evidence.
- **Rationale:** The route is effectively a duplicate-data shortcut, not a tracking model, and it creates the highest leakage risk for `>=0.85` claims.
- **Consequences:** Execution must add leak checks, remove/disable duplicate-copy routing from the general path, and ensure no score evidence uses GEFF copying.
- **Affected IDs and artifacts:** R-03, R-07, AC-03, PLAN N1, GOAL protected behavior.
- **Approval owner:** User.
- **Approval record:** 2026-07-11 user asked what the duplicate route does and stated "if it's a simple cp/duplicate, just delete"; repository evidence shows it is a train-GEFF copy route after duplicate Zarr detection.
- **Reopening trigger:** Competition test split changes, duplicate evidence becomes invalid, or user chooses to focus solely on leaderboard exploitation.
- **Supersedes:** None.
- **Superseded by:** None

## D-03 - Offset And Localizer Leakage Policy

- **Status:** Locked
- **Question:** Are offsets/localizer calibration allowed if learned only from training folds?
- **Context and evidence:** A `44b6` prefix offset improved one full-video official score from `0.2107` to `0.6283`, proving localization is important. The same evidence makes hard-coded prefix offsets a leakage risk if reused on held-out/test claims.
- **Options considered:**
  - **A - Allow fold-safe train-calibrated localizers (selected):** Offsets/refiners are permitted only when derived from training folds and replayed by manifest.
  - **B - Ban offsets entirely:** Strongest anti-leakage posture but may reject the highest-leverage current signal.
  - **C - Allow prefix offsets globally:** Fast but high risk of overfit and misleading held-out evidence.
- **Decision or temporary assumption:** Allow fold-safe train-calibrated localizers only.
- **Rationale:** A general localizer/refiner is the best low-compute path, but evidence must prove it is not held-out/test label leakage. This is not an already pretrained model decision; it authorizes calibration/refinement parameters derived from allowed training folds, plus deterministic localizer methods, with provenance.
- **Consequences:** Requires calibration provenance, fold-safe split logic, and static/config leak gates.
- **Affected IDs and artifacts:** R-04, R-05, AC-03, AC-04, PLAN N3/N4, GOAL.
- **Approval owner:** User.
- **Approval record:** 2026-07-11 user selected "A" and asked whether this meant an already calibrated set of parameters from a pretrained model. Clarification recorded: it does not imply pretrained weights; it means fold-safe calibration/refinement from training folds unless a later branch explicitly introduces pretrained assets.
- **Reopening trigger:** Candidate recall remains poor after deterministic/fold-safe localizers, or user forbids all offsets.
- **Supersedes:** None.
- **Superseded by:** None

## D-04 - Primary Direction And Hybrid Branches

- **Status:** Locked
- **Question:** Which direction should G-01 double down on while still hybridizing other methods?
- **Context and evidence:** Current score jump came from localization; official baseline and Linajea support sparse point detection/localization plus graph solving; Ultrack supports multi-hypothesis proposal selection; Trackastra supports learned association from detections.
- **Options considered:**
  - **A - Center/localizer first, Ultrack and learned scoring as grafted optional branches (selected):** Improve node recall before edge/division sophistication.
  - **B - Ultrack-style multi-hypothesis as primary:** Strong literature fit but Python/memory risks are unresolved locally.
  - **C - Learned edge/division scorer as primary:** Useful after candidate recall improves, but cannot recover missing nodes.
- **Decision or temporary assumption:** Center/localizer first, with Ultrack/multi-hypothesis and learned edge/division scoring as optional grafted branches enabled only by ablation.
- **Rationale:** Candidate/localizer quality sets the upper bound; it is the cheapest current lever and can feed the other branches.
- **Consequences:** First implementation nodes focus on localizer/refiner and candidate recall gates before scorer/division complexity.
- **Affected IDs and artifacts:** R-04, R-06, AC-04, AC-05, ROADMAP G-01, PLAN N3-N8.
- **Approval owner:** User.
- **Approval record:** 2026-07-11 user selected "A".
- **Reopening trigger:** Candidate recall gate passes but edge score remains low, or Ultrack/deep branch becomes cheap and demonstrably better.
- **Supersedes:** None.
- **Superseded by:** None

## D-05 - Compute And Environment Boundary

- **Status:** Locked
- **Question:** What compute and environment side effects are allowed in G-01?
- **Context and evidence:** Local Python is `3.14.6`; Ultrack/official training dependencies are not guaranteed. User wants low compute and to avoid large training-from-scratch. Existing docs mention a `$25` paid-pilot cap, but no current approval authorizes paid compute.
- **Options considered:**
  - **A - Local/Kaggle-free main path:** Use deterministic methods, tabular scorer, and current environment for the primary route.
  - **B - Separate Python 3.11/3.12 environment for Ultrack/official baseline:** Independent optional path for compatibility experiments, with explicit environment provenance.
  - **C - Paid GPU pilot up to `$25` immediately:** Faster learned experiments, but premature before localizer evidence.
- **Decision or temporary assumption:** Use A and B as independent paths: the main G-01 route stays local/Kaggle-free and low-compute; a separate Python 3.11/3.12 environment branch may be used for Ultrack/official-baseline compatibility experiments. Paid GPU remains disallowed.
- **Rationale:** Current bottleneck can be tested cheaply in the main path, while a separate environment can de-risk optional Ultrack/official-baseline work without contaminating the current Python 3.14 environment.
- **Consequences:** The optional environment path must have separate provenance and cannot become mandatory score evidence unless ablation and scorer parity gates pass. Package installs/network commands still require execution-time approval under the sandbox/tool policy; no paid compute is authorized.
- **Affected IDs and artifacts:** R-08, AC-07, PLAN side-effect approvals, GOAL.
- **Approval owner:** User.
- **Approval record:** 2026-07-11 user selected "A,B" and asked whether the two can be independent paths.
- **Reopening trigger:** Localizer/scorer ablations plateau below useful range, a dependency branch becomes necessary for the score gate, or paid compute is explicitly requested later.
- **Supersedes:** None.
- **Superseded by:** None

## D-06 - Division Work Gate

- **Status:** Locked
- **Question:** Should division scoring be mandatory in the current slice?
- **Context and evidence:** Final score weights division Jaccard by `0.1`; current `44b6` fallback has no divisions and score loss is dominated by node/edge performance. Wrong divisions can corrupt graph structure.
- **Options considered:**
  - **A - Precision-first division only after edge/localization is strong (selected):** Add conservative division scorer only if it improves final official CV.
  - **B - Require division scorer in G-01 regardless:** May improve final score but risks false forks before edges are stable.
  - **C - Exclude divisions entirely from G-01:** Simplifies work but may cap final score below target if division-heavy folds matter.
- **Decision or temporary assumption:** Division scoring is precision-first and only enabled after edge/localization is strong and ablation improves final official CV.
- **Rationale:** Division gains matter, but false positives can harm both edge graph and division metric.
- **Consequences:** PLAN includes division node after edge scorer, with enablement controlled by ablation.
- **Affected IDs and artifacts:** R-06, AC-05, PLAN N9, GOAL.
- **Approval owner:** User.
- **Approval record:** 2026-07-11 user selected "A" for branch priority, which includes divisions only by ablation after the main localizer path.
- **Reopening trigger:** Held-out division Jaccard or final score shows divisions are necessary before `0.85`.
- **Supersedes:** None.
- **Superseded by:** None

## D-07 - Reproducibility Baseline

- **Status:** Proposed
- **Question:** How should execution handle the unborn Git state and untracked workspace?
- **Context and evidence:** `git status` shows no commits and all files untracked, including `.env`, data, source, reports, scripts, and `gurobi.lic`. Relevant evidence hashes were captured for this draft, but no immutable repo commit exists.
- **Options considered:**
  - **A - Require a per-run manifest of relevant file hashes and exclude secrets (recommended):** Does not force a commit but preserves reproducibility.
  - **B - Require a Git commit before activation:** Stronger identity but may be outside current workflow.
  - **C - Use path names and timestamps only:** Too weak for an optimization/research goal.
- **Decision or temporary assumption:** Recommended A; not locked.
- **Rationale:** The goal can proceed safely without a commit if every evidence-producing file/config/output is hashed and secrets are excluded.
- **Consequences:** PLAN includes baseline and final manifest nodes.
- **Affected IDs and artifacts:** R-10, AC-09, PLAN N0/N10, GOAL.
- **Approval owner:** User.
- **Approval record:** Pending.
- **Reopening trigger:** User wants commit-bound reproducibility or repo becomes committed before activation.
- **Supersedes:** None.
- **Superseded by:** None.

## D-08 - Kaggle Scorer Parity

- **Status:** Locked
- **Question:** How should the local scorer be aligned with Kaggle competition scoring for iterative testing?
- **Context and evidence:** The repo already wraps the official metric through `src/celltrack_alpha/official_eval.py`, but previous reports distinguish approximate metrics from official metrics. The user explicitly requested local scorer parity with Kaggle scoring.
- **Options considered:**
  - **A - Official wrapper parity as mandatory (selected):** Local iterative scoring must call or faithfully wrap the official competition metric, with self-checks and parity tests.
  - **B - Approximate scorer for iteration only:** Faster but risks optimizing the wrong objective.
  - **C - Leaderboard-only validation:** Too slow and leakage-prone.
- **Decision or temporary assumption:** The activated work must update and verify the local scorer/evaluation path so local iterative results match the Kaggle competition scoring semantics, including adjusted edge Jaccard, division Jaccard, sparse matching, node-count penalty, and schema/graph validity diagnostics.
- **Rationale:** The target is graph-level Kaggle score, so local iteration must optimize the same objective and expose score components.
- **Consequences:** Adds R-11, AC-10, PLAN scorer-parity work, and VG-11. Approximate metrics may remain cheap diagnostics but cannot drive completion.
- **Affected IDs and artifacts:** R-02, R-11, AC-02, AC-10, PLAN N2/N10/VG-11, GOAL.
- **Approval owner:** User.
- **Approval record:** 2026-07-11 user: "also update the local scorer to match how the kaggle competition scores, ensuring parity and enabling iterative testing w accuracy/alignment".
- **Reopening trigger:** Official competition metric changes, Kaggle publishes updated scoring code, or local scorer parity tests diverge.
- **Supersedes:** None.
- **Superseded by:** None
