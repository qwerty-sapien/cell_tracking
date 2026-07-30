# Recall N0 — pool recall ceiling (the alpha node)

> Load `skills/seeking-alpha/SKILL.md`, `DAG.md`, `reports/g02/screening/LIVE_STATE.md`,
> `campaign_narrowing_and_hypothesis_dag_v1.md` (`H2`), `B1_division_tp.md`, the P0
> action-class surface, and the frozen action population `4ab7bb23…76f0f`.
> CPU only. No GPU, Kaggle, asset, or network action.

## Objective / gate

Of the true edges the frozen 44b6 held graph is missing, **how many are absent from the
frozen candidate pool entirely, and what is their zero-FP oracle ceiling?**

This is computable now, for free, and it decides the lane. It also audits the standing
`free-add oracle = 0` result: if most FNs are in fact pool-present, then adds fail for a
*cost* reason (FP admission), not a *supply* reason, and the mechanism class this lane
proposes is misdirected.

## Tasks

1. On the frozen held-fit 44b6 population (36 videos, unchanged folds, unchanged scorer),
   enumerate the held edge FN set. The banked reference counts are TP/FP/FN
   `10,341 / 1,064 / 673`; reproduce them exactly before proceeding and stop if they
   differ.
2. Partition every FN into exactly one of:
   - **(i) pool-present** — a legal `add` candidate for that FN exists in the frozen
     action population, and
   - **(ii) pool-absent** — no legal candidate exists at any rank.
   Report both counts and their sum against `673`. Record the geometric distribution of
   (ii) (displacement histogram), and specifically how much of it falls in the 10–12 µm
   shell that `H2` sized at `456` correct edges.
3. Compute, with the frozen official scorer and no threshold search, the **zero-FP oracle
   ceiling** separately for (i) and (ii): the adjusted edge Jaccard delta from admitting
   exactly that partition's true edges and nothing else. Report raw edge Jaccard and node
   recall deltas alongside.
4. Reconcile (i)'s ceiling with the recorded `free-add = 0` P0 result. State plainly
   whether adds fail on supply, on FP cost, or on both, with the number that shows it.
5. Do not propose a mechanism, threshold, model, or repair. This node measures headroom.

## Output

`runs/N0_POOL_RECALL_CEILING.md` — the two counts, the two ceilings, the displacement
histogram, the shell overlap, the `free-add = 0` reconciliation, exact input hashes, guard
runtime / peak RSS / swap growth — plus a proposed (not written) LEDGER row.

## Verdict (preregistered)

- **GO**: partition (ii) has a zero-FP oracle ceiling `>= +0.030` adjusted edge Jaccard on
  44b6. Supply is then large enough to be worth a mechanism.
- **PARK**: (ii)'s ceiling is `> +0.005` but `< +0.030`. Real but insufficient alone;
  halt and report rather than fund a generator.
- **DEAD**: (ii)'s ceiling is `<= +0.005`, or (ii) is empty. Candidate supply cannot cover
  the gap, the `~0.896` in-pool projection stands as the campaign ceiling, and public
  `0.910` is **not supported by any currently named mechanism** — report that plainly.
- **ESCALATE**: the banked TP/FP/FN cannot be reproduced, or the frozen population cannot
  be joined to the held FN set without a plan change.

## Cascade

- On **GO** → proceed directly to `N1_mechanism_and_crowding.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
