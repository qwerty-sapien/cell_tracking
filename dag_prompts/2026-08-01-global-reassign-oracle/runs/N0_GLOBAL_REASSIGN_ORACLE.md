# GLOBAL-REASSIGN-ORACLE-V1 terminal report

**Terminal status:** `ESCALATE_COMPILER_INEXPRESSIBLE`. **Section 4 verdict:** `CONTINUE_PROPOSAL_ONLY`. Every quantitative scientific or execution value below is [GT-DEPENDENT]. This is a full-fidelity, local-CPU ceiling diagnostic, not a selector or deployable method.

## Section 4 decision arithmetic

Precedent-adjusted gain, reported first as sealed: **+0.01906337599412862 [GT-DEPENDENT]** = 0.2449 [GT-DEPENDENT] × (T2(unbounded) − T0).

Raw T2(unbounded) − T0 increment: **+0.0778414699637755 [GT-DEPENDENT]**. The actionability floor is +0.005 [GT-DEPENDENT]. Because +0.01906337599412862 [GT-DEPENDENT] is not below +0.005 [GT-DEPENDENT], `STOP_CLASS` was not emitted and the scientific verdict is **`CONTINUE_PROPOSAL_ONLY`**.

That verdict authorizes a written proposal only. It authorizes no selector, training, representation, GPU, Kaggle, 6bba, CV57, submission, tier move, DEAD change, or plan amendment. The independent compiler finding remains the node's terminal escalation.

## Final tier table

All deltas are exact versus the applicable frozen baseline. T1 is the sealed integrity-only replay of the distinct run_v6 population and is excluded from T2-minus-T0 arithmetic.

| Tier | Adjusted delta | Raw delta | Node delta | Combined delta | Correct edges installed | FPs removed | Compiled chains | Chain histogram |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T3 | +0.10497364637634421 [GT-DEPENDENT] | +0.10444588914597719 [GT-DEPENDENT] | +0.0 [GT-DEPENDENT] | +0.17164031304301086 [GT-DEPENDENT] | 465 [GT-DEPENDENT] | 958 [GT-DEPENDENT] | 0 [GT-DEPENDENT] | not applicable |
| T0 | +0.01857139664048635 [GT-DEPENDENT] | +0.018473552040216057 [GT-DEPENDENT] | +0.0 [GT-DEPENDENT] | +0.01857139664048635 [GT-DEPENDENT] | 52 [GT-DEPENDENT] | 192 [GT-DEPENDENT] | 29,574 [GT-DEPENDENT] | not applicable |
| T1 | +0.0818976668050011 [GT-DEPENDENT] | +0.08148958008096019 [GT-DEPENDENT] | +0.0 [GT-DEPENDENT] | +0.08965524256257684 [GT-DEPENDENT] | 0 [GT-DEPENDENT] | 1,118 [GT-DEPENDENT] | 2,048 [GT-DEPENDENT] | not applicable |
| T2 L=2 | +0.09641286660426185 [GT-DEPENDENT] | +0.0959260137644794 [GT-DEPENDENT] | +0.0 [GT-DEPENDENT] | +0.1630795332709285 [GT-DEPENDENT] | 465 [GT-DEPENDENT] | 713 [GT-DEPENDENT] | 459 [GT-DEPENDENT] | {1: 453, 2: 6} [GT-DEPENDENT] |
| T2 L=3 | +0.09641286660426185 [GT-DEPENDENT] | +0.0959260137644794 [GT-DEPENDENT] | +0.0 [GT-DEPENDENT] | +0.1630795332709285 [GT-DEPENDENT] | 465 [GT-DEPENDENT] | 713 [GT-DEPENDENT] | 459 [GT-DEPENDENT] | {1: 453, 2: 6} [GT-DEPENDENT] |
| T2 L=4 | +0.09641286660426185 [GT-DEPENDENT] | +0.0959260137644794 [GT-DEPENDENT] | +0.0 [GT-DEPENDENT] | +0.1630795332709285 [GT-DEPENDENT] | 465 [GT-DEPENDENT] | 713 [GT-DEPENDENT] | 459 [GT-DEPENDENT] | {1: 453, 2: 6} [GT-DEPENDENT] |
| T2 unbounded | +0.09641286660426185 [GT-DEPENDENT] | +0.0959260137644794 [GT-DEPENDENT] | +0.0 [GT-DEPENDENT] | +0.1630795332709285 [GT-DEPENDENT] | 465 [GT-DEPENDENT] | 713 [GT-DEPENDENT] | 459 [GT-DEPENDENT] | {1: 453, 2: 6} [GT-DEPENDENT] |

T3 did not trigger its +0.030 [GT-DEPENDENT] early stop. T0 reproduced the sealed +0.018571397 [GT-DEPENDENT] control with absolute error 3.5951365148778613e-10 [GT-DEPENDENT]. T1 reproduced the sealed +0.081897667 [GT-DEPENDENT] control with absolute error 1.949988920335599e-10 [GT-DEPENDENT]. Both are within the unchanged 1e-9 [GT-DEPENDENT] tolerance, so the harness is valid.

## Compiler expressibility: case (b)

The five failures, repeated identically at every T2 depth [GT-DEPENDENT], are **not closure under-merges**. They are a genuine expressive limit of the frozen N1 batch compiler.

The sealed definition requires every displaced baseline-true edge to be re-added as a retained anchor and requires division daughters sharing a source to occupy the same fixed-point closure. The harness does exactly that. For each failure, one correct missing daughter and one correct incumbent daughter share a source; the incumbent is in both `add_edges` and that same chain's `expected_conflict_edges`:

| Video | Chain | Missing correct edge | Retained baseline-true anchor |
|---|---:|---|---|
| `train:44b6_12dfb391` | 15 [GT-DEPENDENT] | (33371, 33915) [GT-DEPENDENT] | (33371, 33890) [GT-DEPENDENT] |
| `train:44b6_e28840c6` | 9 [GT-DEPENDENT] | (21048, 21333) [GT-DEPENDENT] | (21048, 21334) [GT-DEPENDENT] |
| `train:44b6_d754aa59` | 0 [GT-DEPENDENT] | (3729, 3788) [GT-DEPENDENT] | (3729, 3795) [GT-DEPENDENT] |
| `train:44b6_c50204e0` | 1 [GT-DEPENDENT] | (11484, 12028) [GT-DEPENDENT] | (11484, 11985) [GT-DEPENDENT] |
| `train:44b6_5f15d135` | 5 [GT-DEPENDENT] | (12266, 12553) [GT-DEPENDENT] | (12266, 12556) [GT-DEPENDENT] |

Every failing anchor has exactly one add owner and one conflict owner: the same chain. None is owned by a second chain. The decisive control is `train:44b6_d754aa59`, whose entire selected population is one chain [GT-DEPENDENT]. Its individual action compiles, but a frozen `compile_batch` containing only that action fails on (3729, 3795) [GT-DEPENDENT]. A two-chain under-merge is therefore impossible.

The exact frozen-code cause is in [`compile_batch`](../../../src/celltrack_alpha/action_compiler.py): it registers an action's `add_edges`, then iterates the same action's removed conflicts and raises `ActionCompileError: edge ... is added by another action` whenever a removed edge is already registered. [`compile_action`](../../../src/celltrack_alpha/action_compiler.py) accepts the same atomic remove-and-readd transaction and constructs `(baseline − conflicts) ∪ additions`, which preserves the retained anchor correctly. The error text says “another action,” but the owner is the current action.

Disposition: **`ESCALATE_COMPILER_INEXPRESSIBLE`**. No repair 5 receipt was created, the harness SHA remains `d8f9e764de29074b21ad94411fe74b97150b4a5fb9f0b8db7988dc9e3bacba32`, and no T2 rerun is required. No rerun was self-executed. The frozen compiler was not changed, no conflicting chain was dropped, and no preregistered rule was loosened.

## What the oracle changes

### Chain depth is not the unlock

T2 L=2/3/4/unbounded are byte-identical. The histogram is {1: 453, 2: 6} [GT-DEPENDENT]: 453/459 = 98.69281045751634% [GT-DEPENDENT] of closure components are single-edge chains, and 453/465 = 97.41935483870968% [GT-DEPENDENT] of installed correct edges come from those single-edge components. The binding restriction is therefore the both-ends-free/node-disjoint rule on a **single install**, not chain depth.

This directly corrects [`N0B_RECONCILIATION.md`](../../2026-07-30-recall-and-division/recall/runs/N0B_RECONCILIATION.md), which diagnosed the dominant obstacle as a correct fix needing “a chain of other wrong edges.” That explanation is not supported on the measured full 36-video population [GT-DEPENDENT]; almost every beneficial fixed-point component is already length one [GT-DEPENDENT].

### Installing correct edges dominates deleting false positives

T2 and T3 install the same 465 [GT-DEPENDENT] correct edges. They differ only in FP removal: 713 [GT-DEPENDENT] versus 958 [GT-DEPENDENT]. The additional 245 [GT-DEPENDENT] T3 deletions are worth only +0.00856077977208236 adjusted [GT-DEPENDENT] and +0.00851987538149779 raw [GT-DEPENDENT]. T2 reaches 91.84482956666014% [GT-DEPENDENT] of the T3 adjusted ceiling.

Thus roughly 92% [GT-DEPENDENT] of achievable headroom is present with the correct-edge install set, while extra false-positive deletion supplies the small remaining increment. This changes the emphasis of the campaign's DELETE-selector history: the next proposal should target feasible correct-edge installation, not deeper chaining or another delete-only selector. The measured DELETE AUC requirement remains 0.85 [GT-DEPENDENT]; this GT-dependent ceiling does not satisfy or relax it.

## Execution and repair chain

The guarded run passed in 2,200.629405975342 s [GT-DEPENDENT], including a 45.131184816360474 s [GT-DEPENDENT] clean-idle preflight. Peak process-tree RSS was 1.140411376953125 GiB [GT-DEPENDENT]; swap growth was 0.0 MiB [GT-DEPENDENT]. All 36 experimental videos [GT-DEPENDENT] were scored with no subsampling. Guard status was `PASS`.

| Repair | Receipt | Harness old SHA | Harness new SHA | Outcome-free unit suite |
|---|---|---|---|---|
| 1 | `N0_INFRASTRUCTURE_ONLY_EXECUTION_REPAIR_1.json` | `674a44b8…a4db0` | `c3dd315f…4d29f` | 8/8 PASS [GT-DEPENDENT] |
| 2 | `N0_INFRASTRUCTURE_ONLY_EXECUTION_REPAIR_2.json` | `c3dd315f…4d29f` | `b49f8031…a5351` | 12/12 PASS [GT-DEPENDENT] |
| 3 | `N0_INFRASTRUCTURE_ONLY_EXECUTION_REPAIR_3.json` | `b49f8031…a5351` | `3090082c…b28d` | 13/13 PASS [GT-DEPENDENT] |
| 4 | `N0_INFRASTRUCTURE_ONLY_EXECUTION_REPAIR_4.json` | `3090082c…b28d` | `d8f9e764…cba32` | 15/15 PASS in each interpreter [GT-DEPENDENT] |
| 5 | not applicable: case (b), no receipt or mutation | `d8f9e764…cba32` | `d8f9e764…cba32` | no rerun required |

Repairs 1–4 [GT-DEPENDENT] changed no science, population, scope, estimand, tier definition, threshold, tolerance, decision arithmetic, selection rule, scorer, compiler, or preregistration. Repair 5 [GT-DEPENDENT] was not performed because the closure was correct and the frozen compiler is the limiting interface.

No Kaggle, GPU, Kaggle CPU kernel, submission, RunPod, selector, AUC, training, representation, 6bba, CV57, held-fold, tier, DEAD, plan, or LEDGER mutation occurred.
