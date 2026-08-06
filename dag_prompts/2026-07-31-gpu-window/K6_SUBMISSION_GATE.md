# K6 — submission: a human gate, not a node an agent completes

**No agent submits.** `kaggle_budget.json` records
`public_submission: EXPLICIT_HUMAN_APPROVAL_REQUIRED`, and owner governance keeps public
submission as a separate gate from GPU authorization. Approval for a GPU run is **not**
approval to submit.

## What is already true about the submission path

The transport is **proven working** — this is the one piece of infrastructure that is not a
liability:

- `onebitbrain/bhap-transport-smoke-v2/1` passed step 0 in `2.715378342 s`: all `62` declared
  wheel byte hashes, expected manifest, zero competition-path visits.
- Against the authenticated public-`0.881` T4 archive `031aa5f6…7219`, the `263,282`-row
  submission and all four fragments are **byte-identical**; non-timing `run.json` fields match.
- The earlier `1,096`-row deficit was a `BENIGN_FP` reference-selection error, not drift.
- The Lane-1 division-emission package (identity
  `4673a59079b7b9e72bbe261f6a62e2dcf004854e0d0657838bb56ddd144facf7`) is hash-pinned and
  ready to consume a passing 44b6 candidate receipt: frozen-D1 scorer-valid fork validation,
  one-to-one duplicate suppression, frozen-N1 atomic fork compilation. Its BANKED APNCC dry
  run emitted a divisions-ON, empty `submission.csv` with the exact scorer-input graph preserved.

So if a candidate ever passes, the road to a CSV is short and already tested. **That is not a
reason to drive down it.**

## The bar

| Quantity | Value |
|---|---|
| Current banked public | `0.881` (local `0.881522`) |
| Baseline | public `0.877` (local `0.873150`) |
| Submission-worthiness floor | local `>= 0.89` |
| Accepted public `0.89` needs | local `~0.895–0.900` |
| Top-50 bar | public `0.910` → local `~0.9255–0.9422` |
| Transfer slope | `~0.48–0.66`, soft and rounding-bounded |

A `+0.005` graph-delta PASS does not move `0.881522` to `0.89`. **Do not propose a
submission on a K5 PASS alone.** Two finals remain uncommitted; keep them that way.

## What to bring the owner if K5 passes

A written proposal containing, and nothing more:

1. The K5 adjudication and its independent replay.
2. The realized local score of the candidate end-to-end — measured, not projected — against
   the banked `0.881522`.
3. An explicit statement of whether it clears the local `>=0.89` submission-worthiness floor.
   If it does not, the proposal's own recommendation must be **do not submit**.
4. Observation-distinctness evidence versus the banked candidate. H4's closeout rule: a public
   probe attaches only to a local-`>=0.89` **observation-distinct** candidate.
5. The remaining-submissions count and which finals stay uncommitted.

## If and only if the owner approves in writing

- Run the submission through **computer-use cmux**. Never the agent-shell token.
- Use the existing guarded wrapper `scripts/submit_kaggle_alpha.py`; do not write a new
  submission path. It deliberately separates local readiness failures from Kaggle permission
  failures, and it does not accept rules or reuse browser state — keep both properties.
- Fresh guard receipts apply to a submission push exactly as to any other push.
- Record the attempt, the returned public score, and the pre/post quota in
  `LIVE_STATE.md` and the ledger row.

## Default

**No submission in this window.** The window's deliverable is a scientific verdict on the
plan's last named escalation. That is the honest output, and it is worth having on its own.
