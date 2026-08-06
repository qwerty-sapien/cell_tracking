# C4 — the submission predicate: evaluate it, then act on the result

Submission is **pre-authorized** by `AGENTS.md` under a condition. This node makes that
condition machine-checkable so you neither ask needlessly nor submit on a hunch.

> Kaggle kernel launch, **competition submission**, and monitoring are pre-authorized when an
> **observation-distinct candidate or approved experimental node makes the action
> productive**. Do not ask again for Kaggle access in that case. **Quota expiry alone is not
> productivity.**

Two readings are both wrong: treating this as blanket permission to submit whenever a run
finishes, and treating it as no permission at all. Evaluate the predicate and follow it.

## The predicate

Write `SUBMISSION_PREDICATE_RECEIPT.json` with each clause evaluated from **measured**
artifacts. `eligible` is the AND of all of them.

```
P1  realized_local_score          measured end-to-end on the frozen scorer, NOT projected
P2  P1 >= 0.89                    the submission-worthiness floor
P3  observation_distinct          graph observably distinct from the banked APNCC candidate,
                                  proven by fragment hashes, not by score proximity
P4  scorer_input_graph_valid      compiles through frozen N1; zero duplicate survivors
P5  gate_chain_clean              K5 independent replay PASS + provenance closure at clean
                                  checkout
P6  submissions_remaining > 0     and the finals-uncommitted policy still honored
P7  not_a_quota_expiry_rationale  explicit: the reason is the candidate, not the calendar
```

`cmux_ops.competition_submit` refuses unless this receipt exists and reports `eligible: true`.
There is no override flag. If you find yourself wanting one, the predicate is telling you
something.

## For the PEFT lane specifically, it evaluates FALSE

Do the arithmetic rather than deferring to this section, but here is what it will say:

| Quantity | Value |
|---|---:|
| Banked APNCC candidate | local `0.881522069960` → public `0.881` |
| Baseline | local `0.873150` → public `0.877` |
| Submission-worthiness floor | local `>= 0.89` |
| Local needed for an accepted public `0.89` | `~0.895–0.900` |
| PEFT PASS threshold | adjusted-edge delta `>= +0.005` |

A `+0.005` adjusted-edge graph delta does not move `0.881522` to `0.89`. Even a PASS at the
kill line leaves **P2 false**, so `eligible` is false and **no submission follows from this
node**. That is not caution — it is the number.

The edge-only current-pool projection also caps near `0.896`, below the `0.9255–0.9422` local
range implied by the `0.910` public top-50 bar. Transfer slope is `~0.48–0.66`, soft and
rounding-bounded; do not treat it as precise enough to argue a marginal case over the line.

**Therefore: write the receipt, record `eligible: false`, and stop.** Do not submit. Do not
ask the owner to submit — the predicate they authored already answered.

## If it ever evaluates TRUE

Then submission is authorized and you should proceed without asking:

1. Run through `cmux_ops.competition_submit` with the predicate receipt path. Never a
   free-form command.
2. Use the existing guarded wrapper `scripts/submit_kaggle_alpha.py` for local readiness
   checks — it deliberately separates local failures from Kaggle permission failures, and it
   does not accept rules or reuse browser state. Keep both properties. Do not write a new
   submission path.
3. Fresh guard receipts apply to a submission push exactly as to any other push.
4. Competition: `biohub-cell-tracking-during-development`.
5. The transport path is **proven**: `bhap-transport-smoke-v2/1` passed step 0 in
   `2.715378342 s`, and against the authenticated public-`0.881` archive `031aa5f6…7219` the
   `263,282`-row submission and all four fragments are byte-identical. The earlier
   `1,096`-row deficit was a `BENIGN_FP` reference-selection error. Pick the right baseline
   and prove which one you picked.
6. Record the attempt, the returned public score, and the pre/post quota in
   `LIVE_STATE.md`, the terminal report, and the ledger row.

## Never

- Never submit because quota is expiring. `AGENTS.md` rules this out by name.
- Never submit a candidate that is not observation-distinct from the banked one — a
  near-duplicate burns a submission and buys no information.
- Never accept competition rules or claim they need accepting. Done ~2026-07-08; asserting
  otherwise contradicts `Q-APNCC-PUBLIC` in `LEDGER.md`.
- Never let a credential reach a receipt, report, or commit.
- RunPod remains **forbidden** regardless of any result here.

## Done when

`SUBMISSION_PREDICATE_RECEIPT.json` exists with every clause evaluated from measured
artifacts and an explicit `eligible` boolean, and the action taken matches it. For the PEFT
lane the expected terminal state is `eligible: false`, **no submission**, recorded plainly.
