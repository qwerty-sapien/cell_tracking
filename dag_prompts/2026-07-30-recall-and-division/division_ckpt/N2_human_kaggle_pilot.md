# Division-ckpt N2 — bounded T4 pilot: mechanism efficacy only

> Load `DAG.md`, `runs/N0_CHECKPOINT_PROVENANCE.md`,
> `runs/N1_FROZEN_INFERENCE_FALSIFIER.md`, LIVE_STATE, and the Kaggle failure audit
> (17 incidents / 14 root-cause families). **HUMAN-GATE — halts for explicit owner
> approval even on GO.** All authenticated Kaggle CLI work goes through computer-use cmux.

## Objective / gate

Using the frozen external checkpoint **inference-only**, does its `Q` observable separate
**fit-side** division events from matched non-division controls better than a free
geometry-only baseline, inside a bounded T4 allocation?

Because the weight is external and frozen, one embedding pass is legitimate — there is no
per-fold encoder to leak. The leakage surface moves entirely to the *selector*, which this
node does not build. This node measures mechanism efficacy. **It cannot produce a campaign
score, and its result may not be reported as one.**

## Preregistered budget

- Ceiling for this node: **`<=8` T4-h**, inside the owner's 10–16 h weekly envelope and
  well inside the 25.47 GPU-h launch capacity. One run, no parallel bets.
- A measured two-video calibration must project the complete pass at `<=8` T4-h. If it
  projects higher, abort before the full pass; do not borrow from a later node.
- GPU-disabled metadata for any preflight kernel; GPU only for the authorized pass.

## Tasks

1. Prepare the private input asset: pinned weight, pinned source checkout, offline wheels,
   fixture set, fold manifests. Reuse the N1 step-0 preflight verbatim; it must
   authenticate the exact owner-qualified mount path, every manifest by content hash, and
   fold-manifest disjointness inside five seconds before any pip, import, or model load.
   Fail closed on missing, duplicate, or transformed payload.
2. Predeclare the two calibration videos and the full fit-side frame set before launching.
   Calibration outputs are timing-only and are discarded without analysis.
3. Run the single authorized inference-only pass. Emit the append-only `A, S, U, Q` records
   with the order checksum and the schema rejection active. No action ID, pool rank/score,
   continuity result, oracle-bundle ID, GT field, threshold, or graph edit may be written.
4. Build the **free geometry-only control** from data already available: daughter
   displacement symmetry and size/intensity symmetry, with no learned component. This is
   the baseline the deep representation must beat to be worth anything.
5. Measure separation on **fit-side division events only** versus matched controls:
   AUC for `Q`, AUC for the geometry-only control, and the exact Clopper–Pearson 95%
   lower bound for each. Held folds stay closed. The 38 repair bundles may be reported as
   a diagnostic overlay only; they may not set a threshold or select anything.
6. Report the actual GPU debit and update `kaggle_budget.json` inputs for the root
   coordinator. Quarantine any partial shard; no class without a complete manifest may be
   aggregated.

## Verdict (preregistered)

- **GO**: `Q` separation AUC has an exact Clopper–Pearson 95% lower bound `> 0.70`
  **and** exceeds the geometry-only control's *upper* 95% bound, with a complete manifest,
  `<=8` T4-h actual debit, and step 0 passing on the first attempt.
- **PARK**: separation is above chance but does not beat the geometry-only control, or the
  run is bounded-out on capacity. A frozen representation that cannot beat free geometry
  does not justify an amendment request; report and halt.
- **DEAD**: `Q` separation lower bound `<= 0.50`, or the schema/step-0 contract is
  violated. This checkpoint is dead as a division observable.
- **ESCALATE**: the pass cannot be completed within the weekly envelope without changing
  the mechanism.
- **HUMAN-DECISION**: mandatory before launch regardless of the above.

## Output

`runs/N2_PILOT_RESULT.md` — step-0 receipt, calibration projection, actual debit, `Q` and
control AUCs with exact bounds, manifest hashes, and a proposed (not written) LEDGER row.
Also update `reports/g02/screening/LIVE_STATE.md` per the campaign update protocol.

## Cascade

- On **GO** → **halt**. Present the result and the `N3` amendment request to the owner.
  Do not advance to `N3` without explicit written approval of both amendments named there.
- On PARK / DEAD / ESCALATE → halt and report.
