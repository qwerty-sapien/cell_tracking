# Lane adjudication — marginal-value gate applied to the whole surface

Date: 2026-07-31. Written **before** any compute request in this window, as
`reports/g02/screening/MARGINAL_VALUE_GATE.md` requires.

Every lane below was checked against the gate's four questions: realistic (not best-case)
ceiling; comparison to the `+0.005` actionability floor; whether the limiting resource is
structural rather than something the mechanism can move; and whether a STOP applies to the
mechanism *class* rather than the tested instance.

## Verdicts

| Lane | Git branch(es) | State | Realistic ceiling | Gate verdict |
|---|---|---|---|---|
| **PEFT adapter §6.2** | none yet (`main` worktree, `sa_peft_adapter_v1`) | preregistered, **never attempted**; prelaunch hard-stop on fold envelope; now also closed by the terminal report | unmeasured; delete oracle `+0.081897667` | **STOP** — see "The PEFT reassessment" below; owner override only |
| Division realizer | `codex/div-realizer-{geometry,persistence,appearance,consensus}` | v2 full-38 wave terminal, A/B/C/D all fail | GT-dependent `+0.092067841519`; honest available tier (a) `+0.002406490` | **STOP** — below floor; representation closure is structural |
| DINOv2 R5 | `codex/g02-dinov2-kaggle-gpu` | `FAIL_PARK_C` both classes | delete retention `0.001307` vs floor `0.020394` | **STOP** — class-level operating-point failure |
| DPRC-O0 (DINOv2 + registers) | none yet | owner-authorized, no asset/kernel | same class as R5 | **STOP under the gate**, but carries a standing owner authorization — see below |
| CELLECT R7 | `codex/g02-cellect-bleed-stop` | permanent PARK-C, **no further spend** | none — no scientific verdict ever produced | **STOP** — precommitted hard rule |
| Reverse Trackastra | `codex/sa-reverse-authentic-44b6` | terminal `FAIL_PARK_C`, no-retry | delete retention `0–0.003268`; swap `0` in 5/5 folds | **STOP** — terminal |
| Recall supply | none (merged) | N2 DEAD at `9/290 = 3.10%` | action-vocabulary wall, not generator quality | **STOP** — class-level, per the gate's own worked example |
| NODETOPO | `codex/sa-nodetopo` | `NODETOPO-O0` PARK-A; `restore_topology` oracle ceiling `0.000` | `0` beneficial actions in the exact generated population | **STOP** — class-degenerate; CPU-only anyway |
| C2 assignment separation | none (merged) | `FAIL_PARK_C` both classes | delete AUC `0.670235` vs `0.85` | **STOP** — terminal |
| Assoc. / swap / delete O0 | `codex/sa-division`, others | PARK-A/PARK-C | require a **newly named** mechanism | **STOP** — no new mechanism named |
| APNCC transport | none (merged) | parity **PASS**, transport **VALID** | not a science lane | **N/A** — infrastructure only; consumes no window |

## Branch-state note

The `codex/*` branches are historical worktree snapshots. Their artifacts have already been
copied into `main`'s working tree (206 staged adds at the time of writing), and `main`'s
`LIVE_STATE.md` records every one of their outcomes. **No branch holds unreconciled
science.** "Which branch to spend compute on" is therefore not the real question — the real
question is which *node* is launch-eligible, and the answer is exactly one.

## The four suppliers, and why a fifth is not the answer

`DINOv2`, `CELLECT`, `Trackastra-reverse`, and `C2/K-best` all failed the same way, and the
campaign has already named the reason (`LIVE_STATE.md`,
"EXTREME-PREVALENCE OPERATING-POINT FINDING"): at roughly `10^-4` beneficial-action
prevalence, ROC-AUC is not diagnostic of usable lift, and the CG0 false-positive-budget tail
binds. DINOv2 reached AUC `0.670`/`0.682` and retained `0.001307`/`0` against a floor of
`0.020394`/`0.067910` — short by more than `15×`.

A **frozen generic descriptor has no notion of "beneficial action."** That is the structural
limit, and gate clause 4 makes it a class-level STOP: a fifth frozen representation, however
good its ImageNet-style features, hits the same wall.

The PEFT node is the only proposal that moves the limiting factor rather than re-testing
around it — it fits a small adapter **directly against the graph-delta objective** under
nested OOF. That is the campaign plan's own §6.2 escalation, written in advance, precisely
for "if frozen fails". Its precondition is now satisfied four times over.

**This is not a prediction that it will pass.** The honest prior is low: the adapter must
close a `15×` retention gap on `1,118` beneficial actions out of `1,345,497`. It was
justified because it is the one untested *class*, it was preregistered before the outcomes
were known, and its kill line is the real metric rather than a proxy.

## The PEFT reassessment (2026-07-31, after the terminal report)

The argument above was written before the root coordinator froze
`TERMINAL_CAMPAIGN_REPORT.md`. That report supplies evidence that **undercuts the PEFT case
on the merits**, not merely on governance, and this adjudication now follows it.

Two *independent* information sources — DINOv2 endpoint appearance and C2 assignment
evidence (MAP / min-marginal / K-best, covering all `1,391,458` actions with no sampling) —
converge on DELETE at AUC `0.670412` and `0.670235`. The difference is `0.000176`.

Those sources share almost nothing: one is a frozen visual descriptor, the other is
combinatorial assignment structure. That they land on the same number to four decimals is
the signature of a **label-side ceiling, not a feature-side one**. The campaign reads it as
an empirical Bayes ceiling near `0.670` for the current **pointwise** beneficial-action
label — explicitly a campaign inference from convergent evidence, not a formal bound.

Both suppliers were already fit heads (`B0→A1`) supervised on that same pointwise label. A
LoRA adapter adds representation capacity against the identical target. If `0.670` is a
property of the label rather than the features, additional capacity buys approximately
nothing — and the requirement is `0.85`.

The terminal report's reopening condition names this directly: "another endpoint-appearance
encoder, architectural substitution, upload-path repair, or uncalibrated proxy does not
satisfy the condition." **A frozen-backbone adapter is an architectural substitution.**

**Revised verdict: STOP.** Not because a gate says so, but because the convergence evidence
says the estimand is the bottleneck. Running PEFT now would most likely purchase a fourth
independent measurement of `~0.670` for `5.4–8.6` GPU-h. That is a legitimate thing for the
owner to buy if they want the confirmation on the record — the node is fully specified and
ready — but it should be bought knowingly, not as the default use of expiring quota.

## DPRC-O0: authorized but gate-negative

The owner wrote **"DINOv2: go ahead"**, and
`dag_prompts/2026-07-30-continuation/dinov2/runs/N3_HUMAN_REQUEST.md` records one bounded
route at a `5.0` GPU-h ceiling with a `<=4.0` GPU-h calibrated projection requirement.

That authorization is real and predates this adjudication. But the gate, applied honestly,
returns STOP: registers change DINOv2's attention-artifact behavior, not the prevalence
problem that killed the non-register checkpoint at the same operating point.

**Recommendation: do not spend the window on it.** If the owner wants it run anyway — which
is their call, and the authorization already exists — it belongs *after* K4 completes, using
leftover hours, and the gate's clause-1 note ("state the realistic ceiling in writing before
the request") must be written into the DPRC node report first. See
[`FILL_DPRC_O0.md`](FILL_DPRC_O0.md).

## Honest bottom line on the window

Under the campaign's current disposition: **`0` of `24.98` GPU-h is spendable.** The surface
is closed and nothing pending satisfies its reopening condition.

If the owner overrides, in descending order of defensibility:

| Choice | GPU-h | What it buys |
|---|---:|---|
| Nothing | `0` | The `24.98` h expires; a fresh `30` h opens `2026-08-01T00:00:00` |
| PEFT (`H0` + override) | `5.4–8.6` | A fourth independent measurement of the `~0.670` ceiling, on the record, closing §6.2 by experiment rather than by inference |
| DPRC-O0 | `≤5.0` | A fifth frozen encoder; gate-negative twice over |

The remaining hours cannot be spent without authoring and sealing a new preregistration, and
a contract written in a hurry to consume expiring quota is exactly the failure mode
`MARGINAL_VALUE_GATE.md` and §6.3 were written to prevent.

**Recommendation: let the quota expire and spend the 20 h on
[`R0_REFORMULATION.md`](R0_REFORMULATION.md) instead.** It is the only direction that
satisfies the reopening condition on its merits, it is CPU-and-design work that fits the
window, and it leaves next week's full `30` h available to fund whatever it produces. Trading
expiring GPU hours you cannot legitimately use for a sealed preregistration you can is the
better deal.
