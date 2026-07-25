# CG2 — Build the selected abstention realizer

> Load `.agents/skills/seeking-alpha/SKILL.md`, `.agents/skills/dag-prompts/SKILL.md`,
> `DAG.md`, and `cg1_abstention_screen_v1`. Runs on a CG1 **plan** survivor.
> Seeking-alpha **Phase 6**: build only the winner, preregistered against the FP budget
> and the held-fit graph objective.

## Scope

Implement the single CG1-selected **abstention/selection** mechanism over the current
r10/cap-8 pool — realize the latent 0.960918 oracle by rejecting background within the
budget while retaining correct edges, with an operating point that **transfers across
videos and holds on 44b6**. This is *not* candidate generation (deprioritized) and *not*
another from-scratch ranker (DEAD). If appearance/Trackastra features help the abstention
score, use them here; the `K_trackastra_6bba_feature_kernel_v1` cache feeds this.

## Preregistration (write before building)

- **Objective:** held-fit graph delta on the frozen scorer vs banked APNCC 0.881522.
- **Success:** ≥ **+0.005** on **both** families, no raw-score or node-recall regression,
  evaluated **44b6 first**.
- **FP budget:** stay within CG0's per-video background-admission budget on the
  *unlabeled deployment* pool, with the operating point set by a **transferable** rule
  (conformal quantile / calibrated statistic), not a per-video oracle threshold.
- **Kill line:** 44b6 held graph delta ≤ 0, or the operating point fails to transfer
  (family collapse), or FP admissions exceed budget → PARK; do not retune on held folds.
- **Generalization:** nested grouped-OOF; per-family floors; a strong pretrained
  abstention feature that still fails 44b6 → domain-shift verdict → escalate to a PEFT
  adapter (frozen backbone, <1 % params), never a full fine-tune (seeking-alpha / §6.2).

## Build → gate order

```
literature/manifest gate → data/leakage gate → abstention over 44b6 → held-fit 44b6 graph
   → 6bba → full CV57 → (HUMAN-GATE: approval) → one observation-distinct public probe
```

Proposal-level precision/recall may be *reported* but never *gate* — only the held-fit
graph delta promotes (`CTRL-OOF-GRAPH`).

## Compute / gates (inherited)

- Fit-only CPU under the zero-swap guard; GPU only behind the P1 gate with a preregistered
  ≥+0.005 proxy and human approval; RunPod only after an accepted 0.89. No GNN.

## Output

`reports/g02/screening/candgen/cg2_abstention_realizer_v1.md` + machine report + LEDGER
row: mechanism identity, per-family held-fit graph deltas, FP realized vs budget,
transfer evidence, and a BANK / PARK / HUMAN-DECISION verdict.

## Verdict & cascade

- **GO (BANK)** — held-fit +0.005 both families, budget met → **HUMAN-GATE**: this is the
  first candidate with a reachable path toward 0.92; halt and request explicit approval
  for one observation-distinct public probe (≥0.89 local). Do **not** auto-submit.
- **PARK** — fails the graph gate or transfer → halt; report; consider `CG_ESCALATE`.
