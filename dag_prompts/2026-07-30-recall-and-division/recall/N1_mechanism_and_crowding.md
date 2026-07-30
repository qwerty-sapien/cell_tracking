# Recall N1 — name a label-free supply mechanism and survive the crowding test

> Load `skills/seeking-alpha/SKILL.md` (Phases 3–5 and the crowding test),
> `references/novelty-techniques.md`, `DAG.md`, `runs/N0_POOL_RECALL_CEILING.md`,
> LIVE_STATE's DEAD list, and `campaign_narrowing_and_hypothesis_dag_v1.md` `H2`.
> CPU only. No GPU, Kaggle, asset, or checkpoint download.

## Objective / gate

Is there at least one **label-free candidate-supply mechanism** that could generate N0's
pool-absent true edges, is grounded in primary literature, is not on the DEAD list, is not
a selector in disguise, and has a nameable reason it will not be replicated by every
competent team?

## Tasks

1. Restate N0's pool-absent set as an information requirement: what must be observable,
   without labels, to nominate those specific edges and not the `5.34M` shell background?
2. Generate `3–5` candidates using at least two distinct novelty generators
   (cross-disciplinary transfer, literature-edge mining, first-principles re-derivation,
   constraint inversion, boundary reasoning). Reach wide here; commitment is later.
3. Apply the hard disqualifiers and reject on the spot:
   - anything that re-ranks or re-thresholds the frozen population (that is the measured
     `~0.896` family, not this lane);
   - anything equivalent to a DEAD mechanism — `R12SHELL`, `MGSHELL`, `TEMPBG`,
     `POINTCTX`, `CROPLINK`, `TEMPRANK`, `TEMPHYBRID`, `TRACKCTX`, scalar
     recalibration/thresholding, hand-tuned graph-cost or motion knobs, organizer
     `TemporalUNet3D`;
   - anything requiring GT, oracle bundles, or held-fold data at decision time;
   - anything reinstating frozen-Trackastra zero-wrfeat features as its supplier;
   - anything whose stated envelope exceeds the 25.47 GPU-h launch capacity.
4. For each survivor, run the crowding test explicitly: of a thousand competent teams, how
   many reach this exact move, and **why** will the rest not? Name the reason (hard-won
   operationalization, non-obvious transfer, obscure literature, moved constraint). A
   survivor with no nameable reason is beta — record it as such and do not rank it first.
5. Ground each survivor in `>=2` **primary** sources with resolvable identifiers. Prefer
   old, obscure, or cross-field work. State for each source what it does and does not
   establish about this campaign's conditions — a source showing a mechanism works
   elsewhere is not evidence of transfer here.
6. For the top-ranked survivor, state its **label-free opportunity count**: how many
   candidates it would nominate on a frozen fit-only slice, and the implied FP inflation
   against the current pool size.

## Output

`runs/N1_MECHANISM_AND_CROWDING.md` — the candidate table (mechanism, generator,
disqualifier result, crowding verdict, sources, opportunity count), the ranked survivors
by pertinence × feasibility × defensibility, and a proposed (not written) LEDGER row.

## Verdict (preregistered)

- **GO**: `>=1` survivor is label-free, non-DEAD, non-selector, grounded in `>=2` primary
  sources, has a stated opportunity count, and carries a named defensibility reason.
- **PARK**: survivors exist but all are beta (no nameable reason) or lack an opportunity
  count. Report the beta list; do not fund it.
- **DEAD**: every candidate collapses onto a DEAD mechanism or onto pool selection.
- **ESCALATE**: the only viable mechanism class needs a locked plan decision (for example
  reopening the detection front-end, which is Conditional-PARK) before it can be defined.

## Cascade

- On **GO** → proceed directly to `N2_supply_falsifier.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
