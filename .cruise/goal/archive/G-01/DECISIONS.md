# Decisions

- Planning state: Draft
- Mode/current slice: New / G-01
- Last reconciled: 2026-07-15
- Repository identity: `c3b9b703b049fc25f10e2b42bb986bb94f137a72` / tree `6545134fe081bf5a1e5b8e7b44b07ea8b0b66935`

## Index

| ID | Status | Topic |
| --- | --- | --- |
| D-01 | Locked | External Cruise worker data transfer |
| D-02 | Locked | Kaggle account submission and verification envelope |
| D-03 | Locked | Score endpoint source of truth |
| D-04 | Locked | Bounded Kaggle submission budget |
| D-05 | Locked | Unscored Kaggle kernel remote generation |
| D-06 | Locked | Continued unscored Kaggle kernel repair attempts |
| D-07 | Locked | Kaggle-to-local scorer calibration experiment |

## D-01 - External Cruise Worker Data Transfer

- Status: Locked
- Question: Allow the required one-time Cruise worker retry, knowing it sends the bounded repository-derived packet and returned evidence to the external Codex worker service?
- Evidence: The initial `cruise worker --level 3` launch disconnected before completion. The host retry was rejected until the user explicitly authorized repository-derived packet/evidence transfer.
- Chosen path: Allow the Cruise worker retry and ordinary Cruise worker launches needed for this G-01 workflow.
- Allowed: Send bounded Cruise worker packets and returned evidence derived from this repository to the external Codex worker service for Cruise planning, implementation, and verification.
- Prohibited: Sending secret values, Kaggle tokens, passwords, unrelated personal data, or unbounded repository dumps.
- Owner: Cruise coordinator.
- Duration: Current G-01 Cruise workflow.
- Runtime approvals: Still required when surfaced by sandbox/tooling.
- Evidence: User message on 2026-07-15: "I am explicitly allowing both cruise kaggle submission path inspection..."
- Reopening trigger: Worker packets need credential values, unrelated private data, or broader external disclosure.

## D-02 - Kaggle Account Submission and Verification Envelope

- Status: Locked
- Question: May Cruise use the user's logged-in Kaggle/Chrome state to inspect the Biohub competition page, submit/upload generated `submission.csv`, read web content and scores, and delete/replace generated submissions if needed?
- Evidence: The current local score is not a Kaggle site score. The requested verification requires external Kaggle account access and an account-mutating submission.
- Chosen path: Permit Kaggle account interaction for this competition verification and iteration goal.
- Allowed: Navigate Kaggle in the user's logged-in Chrome session; read competition/submission/leaderboard web content; upload generated `submission.csv`; check returned score/status; delete or replace generated `submission.csv` artifacts and Kaggle submissions when necessary for this G-01 workflow.
- Prohibited: Purchases, profile/account setting changes, credential changes, submissions to unrelated competitions, publication of secrets, or paid compute without a separate explicit approval.
- Owner: Kaggle submission/verification node after Cruise Start.
- Duration: Current G-01 Cruise workflow, ending at closeout or explicit revocation.
- Cost/resource ceiling: No paid compute; low-compute local/Kaggle notebook work only unless separately approved.
- Data sensitivity boundary: Competition submission files and public/competition-visible scores only; do not expose secret token values.
- Runtime approvals: Sandbox/tool/browser approvals still apply.
- Evidence: User message on 2026-07-15 granting Kaggle account permissions.
- Reopening trigger: Need to use Kaggle API credentials directly, upload non-submission data, run paid compute, or interact with another account/resource.

## D-03 - Score Endpoint Source Of Truth

- Status: Locked
- Question: Is the revised `0.92` endpoint a Kaggle site score or a local CV score?
- Evidence: The user asked whether the prior score is "the true score reflected by the site" and asked to revise the endpoint to `0.92`. The worker confirmed the `0.8713516075529171` score is local locked-CV evidence, not a Kaggle score.
- Chosen path: Treat `0.92` as the authoritative Kaggle site score endpoint. Local CV and scorer parity are necessary gates, not final proof.
- Consequences: G-01 cannot complete from local CV alone. If Kaggle site score remains below `0.92` after the approved bounded iteration budget, close as Blocked with score evidence and next required decision.
- Affected artifacts: SPEC R-01/R-04, PLAN E-01/E-05, ALIGNMENT, GOAL.
- Approval owner and record: User origin prompt and 2026-07-15 permission message.
- Reopening trigger: User explicitly changes endpoint definition or Kaggle stops returning usable scores.

## D-04 - Bounded Kaggle Submission Budget

- Status: Locked
- Question: How many Kaggle submissions may this Cruise run consume before a checkpoint?
- Evidence: The user granted broad Kaggle upload/check/delete/replace permission, but the competition may have quotas and the run must avoid uncontrolled external mutation.
- Chosen path: Allow up to 5 valid Kaggle competition submissions in this G-01 calibration phase: one baseline verification submission plus up to four predeclared diagnostic variants. Diagnostic variants do not need a minimum expected leaderboard score, but each must isolate one scorer/model axis and pass schema/graph validation before submission.
- Prohibited: More than 5 valid scored submissions in this phase, paid Kaggle compute, unrelated submissions, blind retries, or changing multiple diagnostic axes in a way that prevents attribution.
- Consequences: Use the baseline and diagnostic scores to calibrate and then hold out part of the submission set to verify the local scorer. After five valid site-scored submissions, further Kaggle submissions require a new explicit checkpoint even if the scorer is calibrated or the site score remains below `0.92`.
- Approval owner and record: User explicitly raised the budget on 2026-07-15 to the baseline plus another 3-4 diagnostic submissions and waived a minimum score threshold for those calibration submissions.
- Reopening trigger: User explicitly raises or lowers the cap, Kaggle quota is lower, or no valid score is produced.

## D-05 - Unscored Kaggle Kernel Remote Generation

- Status: Locked
- Question: May G-01 replace the blocked local T-03 full-generation dependency with an unscored private Kaggle kernel run that generates `submission.csv`, then download and locally validate that CSV before any scored competition submission?
- Evidence: T-03 exhausted two local full-generation attempts without producing a CSV. T-03R passed package/Zarr/tiny-clip smoke checks and classified the likely blocker as local resource/runtime, recommending the remote-generation revision. The user replied `1` to approve `.cruise/runs/g01/proposed_revision_remote_generation.md` with digest `f1578db9327fdb808eb0ca7450f883731b053c3355683311b78bc1c198dd2e41`.
- Chosen path: Approve T-03B/T-03C as an active replacement path for the blocked T-03 dependency.
- Allowed: Push/update private Kaggle kernel `onebitbrain/biohub-celltrack-alpha`; run it on Kaggle free CPU with GPU and internet disabled; read kernel status/logs; download generated `submission.csv` into `reports/kaggle_g01/`; locally validate the downloaded CSV before any scored competition submission.
- Prohibited: Paid compute, GPU, internet-enabled kernel runtime, public publication, unrelated Kaggle account changes, credential changes, submissions to other competitions, printing secret values, or triggering a scored competition submission during the T-03B generation step.
- Budget: One remote generation run plus one packaging/runtime repair rerun if the first run fails before producing a CSV.
- Consequences: D-04 scored submission cap remains unchanged; T-04 may only submit after T-03C validates the downloaded CSV.
- Affected artifacts: `.cruise/runs/g01/proposed_revision_remote_generation.md`, `.cruise/runs/g01/T-03B_remote_kaggle_generation.command_plan.yaml`, `.cruise/runs/g01/T-03C_validate_remote_csv.packet.yaml`, CLOSEOUT.
- Approval owner and record: User reply `1` on 2026-07-15 after the blocked closeout and proposal presentation.
- Reopening trigger: Kaggle asks for payment, GPU, credential/account changes, broader publication, additional remote reruns, or any action outside this competition.

## D-06 - Continued Unscored Kaggle Kernel Repair Attempts

- Status: Locked
- Question: After D-05's single repair rerun failed before producing a CSV, may G-01 continue adjusting the unscored Kaggle submission script and rerunning the private kernel until required runtime/package conditions are fulfilled?
- Evidence: Kernel version 1 failed before CSV with missing `zarr`; version 2 failed before CSV with missing `celltrack_alpha`. The user replied: "1, and continue attempting as you adjust the submission script to ensure everything that is required is fulfilled and ensuring successful submission". A single-file bundled candidate with package identity `339c23bfa63f09f67eb8649fa214dcbad3a1434fa52d48bb55c12f7cbf2d2c6f` passed local isolated import and fallback Zarr chunk-read checks.
- Chosen path: Permit additional unscored private Kaggle kernel repair reruns to obtain a generated `submission.csv`, while preserving the scored submission cap.
- Allowed: Modify only packaging/runtime/submission-script code needed for Kaggle execution; push/run the private CPU/no-internet/no-GPU kernel; read logs/status; download generated artifacts; validate locally.
- Prohibited: Paid compute, GPU, internet-enabled runtime, public kernel publication, credential/account changes, unrelated Kaggle actions, submissions to other competitions, printing secret values, changing the model-scoring endpoint, or creating a scored competition submission before T-03C local CSV validation passes.
- Budget: Up to five additional unscored repair reruns from package identity `339c23bfa63f09f67eb8649fa214dcbad3a1434fa52d48bb55c12f7cbf2d2c6f`, stopping earlier if a CSV is produced and validated, if a new permission boundary appears, or if failures no longer identify a concrete packaging/runtime defect.
- Consequences: D-04 remains the controlling scored-submission budget: at most five valid scored competition submissions in the current calibration phase after local CSV validation.
- Affected artifacts: `scripts/build_kaggle_kernel.py`, `kaggle_kernel/alpha_submission.py`, `reports/kaggle_g01/**`, `.cruise/runs/g01/T-03B_remote_generation_attempt.yaml`, CLOSEOUT.
- Approval owner and record: User reply on 2026-07-15 approving option `1` and continued script adjustment attempts.
- Reopening trigger: More than five additional unscored reruns are needed, paid/GPU/internet runtime is required, or the next change would affect model behavior rather than packaging/runtime execution.

## D-07 - Kaggle-To-Local Scorer Calibration Experiment

- Status: Locked
- Question: How should G-01 use the additional Kaggle submissions to learn the competition scoring response without requiring each diagnostic candidate to improve the leaderboard?
- Evidence: The current `0.8713516075529171` result is local locked-CV evidence only. The user requested comparison against authoritative Kaggle scores, 3-4 additional diagnostic submissions, and local scorer revision until it is within `+-2%` of Kaggle.
- Chosen path: Treat `+-2%` as an absolute score error of `<=0.02`. Submit one baseline followed by four schema-valid, single-axis variants: division-edge ablation, conservative long-edge pruning, deterministic low-confidence/node-density pruning, and one frozen association-pruning holdout. Freeze variant definitions before observing their Kaggle scores.
- Calibration protocol: Use the baseline and first two diagnostic pairs to identify formula/weight/domain-shift corrections. Use the remaining two diagnostic pairs as holdouts. Compare both absolute scores and score deltas from baseline. A calibrated local scorer is accepted only when every frozen holdout has `abs(local_score - kaggle_score) <= 0.02`; report relative error as a secondary diagnostic.
- Allowed: Modify the local scorer and its tests; generate validated CSV variants from the same model output; submit within D-04; record submission IDs, artifact hashes, local predictions, Kaggle scores, and residuals.
- Prohibited: Using hidden-label inference to alter individual test detections, unbounded leaderboard probing, paid compute, large-model training, post-hoc modification of frozen holdout definitions, or declaring parity from calibration points alone.
- Consequences: Once parity passes, local score becomes the primary iteration signal. A later Kaggle verification submission still requires available quota or a new explicit budget decision and remains the source of truth for the `0.92` endpoint.
- Approval owner and record: User message on 2026-07-15 requesting 3-4 additional submissions specifically to internalize Kaggle scoring and calibrate the local scorer.
- Reopening trigger: Kaggle exposes component-level scores that support a better design, fewer than five submissions are available, a diagnostic CSV fails validation, or the holdout residual remains above `0.02` after the bounded experiment.
