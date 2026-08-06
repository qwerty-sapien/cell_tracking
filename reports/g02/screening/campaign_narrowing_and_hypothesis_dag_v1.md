# Campaign narrowing & hypothesis-driven DAG — v1

> Load `_shared_context.md`, `DAG.md`, `P1_framework_and_rubric.md`, and
> `P0R_recalibration_wide_dispersion.md`. This document consolidates ~40 banked
> bets in `LEDGER.md` into (a) what the evidence rules in/out and (b) a redesigned
> DAG that runs on falsifiable, objective-level hypotheses instead of a falsified
> proposal-level proxy. It does not change scorer/transfer/GPU/submission policy.

## 0. The finding that reorganizes the campaign

**Proposal-level metrics do not predict the frozen-scorer graph objective, and the
campaign has proven it directly.**

- `CTRL-OOF-GRAPH`: proposal OOF precision **0.9648** (passes the 0.95 gate) →
  graph adjusted **−0.000612**, raw **−0.000641**, **+2 TP / +49 FP**.
- `GA-TEMPHYBRID`: strong fit-only 6bba precision/utility → full CV57
  **−0.001408** (−12 TP / +52 FP).
- Every from-scratch ranker screened on proposal precision/"utility gain" that then
  reached a graph score **regressed**. The graph solve plus the 0.5%/min-16 per-video
  acceptance cap washes out proposal gains; any precision shortfall becomes retained FP.

**Consequence:** the two-tier ladder `proposal SCREEN (≥0.95) → graph CONFIRM` is a
false-positive generator, not a fast-fail. The only tests that have ever agreed with
the objective are the objective itself: the held-fit graph gate and the frozen CV57
scorer. The redesigned DAG turns on that axis.

## 1. Ruled OUT (prune these; several PARK→DEAD)

| Mechanism | Decisive evidence | New status |
|---|---|---|
| Detection / segmentation front-end | E1 **ASSOCIATION-BOUND**; fixed-node oracle **0.975**; FN lower bound 61.9% | Deprioritize `GD`; not a lever for an association-bound residual. |
| Division / lineage postprocessing | B1 **0/51** counterfactual restores was measured before the 38 exact frozen-N1 repair bundles and before six association mechanisms plus four representation suppliers failed | **AMENDED 2026-07-31 (§6.4):** fit-only division realizers are unlocked from the association-continuity prerequisite; 44b6-first graph/scorer gates remain binding. |
| Scalar recalibration / threshold / distribution normalization | `SCENCAL` −195 utility; `POINTCTX-THRESH` no transferable floor; `APNCC-POLICY` score-neutral | **DEAD** as a modeling lever. |
| Hand-tuned graph-cost / motion knobs | sweeps +0.00003…+0.002; `GA-SP` **−0.033**; `LC-P4CTRL` two-arm oracle **0.8768** | **DEAD**: existing-output selection cannot cross 0.88. |
| Geometry/NCC/constant-velocity abstention on the extended-radius shell | `R12SHELL` 0.14 prec; `MGSHELL` 0.015 prec; `TEMPBG` over-abstains | **DEAD** without a learned no-link model. |
| GT-conflict-trained selected-graph gates | `SGCONFLICT` over-predicts **2.5–17×** on the unlabeled pool | PARK until explicit unlabeled-abstention regularization. |
| From-scratch learned rankers (point-context, crop-linker, temporal MLP) | 44b6 precision 0.833 / 0.889 / 0.902; 6bba 0.98–1.00 | **DEAD** as a class without pretraining or explicit domain adaptation. |
| Full organizer TemporalUNet3D to convergence | `CTRL-COMPUTE` ~478 T4-h vs 30-h cap; `M1-ORGCKPT` 3.96% precision | **DEAD** under budget as specified. |

## 2. Ruled IN (the frontier)

- **Appearance-based association is the only banked win that transferred.**
  `APNCC-DIRECT` +0.008372 → **0.881522** local; `Q-APNCC-PUBLIC` **0.881** public,
  local−public **−0.000522** (within one display tick). Appearance survives the
  graph solve and the leaderboard.
- **A strong pretrained/frozen model is the only large generalizable signal.**
  `TRACKASTRA-QUAL` ROC AUC **0.9185 vs 0.8412** (+0.077), CPU-feasible full pool
  (1.213 h projected). Pretraining is what sidesteps the from-scratch 44b6 overfit.

## 3. The residual is two coupled sub-problems

`APNCC-RESIDUAL`: 4,407 residual FP+FN; **378 ideal swaps reach local 0.90**; top-10
videos hold **52.3%**. It is association-sized and concentrated. Reaching it requires
solving both:

- **S1 — cross-dataset generalization (44b6).** The universal kill criterion; every
  from-scratch model failed here while acing 6bba. Post-rescoring rewards exactly this
  robustness (champion 0.985→0.913).
- **S2 — unlabeled background abstention.** `R12REACH`: 456 recoverable correct edges
  in the 10–12 µm shell buried in 5.34M background. Recall expansion needs a model
  that learns "no-link" against the deployment population, not GT-observable conflicts.

**The entire surviving question:** can a Trackastra-class frozen appearance+motion
model, with explicit unlabeled-background abstention, clear the per-family (44b6)
floor **at the graph level**?

## 3.1 Public-target arithmetic (planning heuristic — NOT a decision-grade law)

The measured CV57→public transfer (`p0_parity.md`: 4 pairs; OLS slope ≈0.66, dominated
by the distant `c=0` control; the single high-range APNCC point implies ≈0.48; fine
transfer STILL-UNRESOLVED and rounding-bounded) means **local and public 0.89 are
different bars**:

- Banked APNCC: local `0.881522` → public `0.881`.
- Coarse slope ⇒ **public 0.89 requires local ≈0.895–0.900**, i.e. at or above the
  projected in-pool oracle `~0.896` (`achievable_headroom_v1`).

Consequences:

- H4's "submit at local ≥0.89" is a **submission-worthiness floor** (observation-distinct,
  above the |Δlocal|<0.004 abstention zone) — **not** sufficient for an *accepted* (public)
  0.89; local 0.89 maps to public ≈0.887.
- An **accepted 0.89 — hence the RunPod gate — needs local ≈0.895+**, which re-ranking the
  current pool (oracle ~0.896) can at best barely reach. It therefore depends on H1
  extending to **44b6** and on **H2** recall (§3, S1+S2). This is independent
  corroboration of §3 from the transfer side.

Caveat: the slope is soft (one high-range anchor; rounding-bounded). Per §4, one point
cannot establish a law; the first H1 high-range public probe sharpens it. Use this only to
set target bars, never to forecast a specific public score.

## 4. Redesigned DAG — objective-first, hypothesis-driven

### Principles

1. **Node = falsifiable hypothesis about the objective.** State a preregistered
   effect size on the frozen-scorer graph delta / CV57 **and** a per-family floor
   (both 44b6 and 6bba). No proposal-only pass may promote a candidate.
2. **Cheapest *valid* falsification first.** The held-fit graph gate (CPU, cached
   fragments) is the SCREEN, not a late CONFIRM. Run it **44b6-first** so candidates
   die on the hardest family before 6bba/full-CV57 compute is spent.
3. **Proxy-validity is itself gated (H0 below).** No cheap proxy is trusted until it
   is shown to correlate with graph delta on the existing corpus.
4. **Prune on proven absence.** Section 1 DEADs are removed from the frontier; do not
   re-screen them without a new mechanism.
5. **Calibration is settled enough.** Coarse transfer PASSED; fine transfer is
   rounding-unresolvable. Optimize local CV57; submit only observation-distinct
   candidates at local **≥0.89**; hold 2 finals as a generalization hedge.

### Falsification order for any candidate

```
manifest/literature gate → data/leakage gate → 44b6 held-fit GRAPH delta
   → 6bba held-fit GRAPH delta → full CV57 → (approval) → one public probe
```

A candidate that fails the 44b6 graph delta is PARKed/DEADed immediately; it never
consumes 6bba or CV57 compute. Proposal metrics may be *reported* but never *gate*.

### The hypotheses

**H0 — proxy validity (run first, cheap, decisive for methodology).**
Regress the objective on the proxies the campaign already produced. Assemble the
~15 `(proposal metric, held-fit graph delta)` pairs from the ledger; fit graph-delta
on OOF precision, utility gain, AUC, support. Preregistered decision: if no proxy
predicts graph-delta sign at rank-correlation |ρ| ≥ 0.5, **formally ban proposal-only
screening** and route every candidate straight to the graph gate. If one does,
recover it as a legitimate fast-fail with its measured error bar.
_Output: `h0_proxy_validity_v1.md`. No new training._

**H1 — frozen Trackastra + tiny calibrated head clears the graph floor, extended to 44b6 (the live lever).**
Extend the feature cache to **both families** (not the 6bba-only prereg): a 6bba-only
frozen head replays the shape that already regressed (`GA-TEMPHYBRID` −0.0014) and
cannot test the surviving question. Keep the backbone **frozen** (see §6.2); fit only
the nested grouped-OOF calibrated head, then run straight into the held-fit graph gate
**44b6 first**. Preregistered success: held-fit adjusted edge Jaccard **≥ +0.005 on
both families**, no raw-score or node-recall regression. Kill if 44b6 graph delta ≤ 0.
Generate the 6bba+44b6 feature caches on Kaggle GPU (§6.3) to escape the local swap
deadlock. _Highest-value node; it directly tests the surviving question._

**H2 — a learned no-link model makes the recall shell reachable (tests S2).**
Only meaningful if a model with real signal exists (H1's Trackastra features).
Hypothesis: Trackastra-scored abstention on the 10–12 µm shell recovers a net-positive
share of the 456 correct edges at the frozen scorer **without** admitting the 5.34M
background. Preregister on the *unlabeled deployment* pool, not GT conflicts. Kill if
graph delta ≤ 0 or background admission exceeds the per-video cap. _Falsifies whether
the extended-radius headroom is reachable at all._

**H3 — 44b6 failure is domain-shift, not capacity (tests S1, and unblocks H1).**
The Trackastra qualification used a **zero-filled 12-channel wrfeat interface**;
mask-feature compatibility is unresolved. Hypothesis: supplying real mask/appearance
features (or per-video/per-family feature normalization) closes the 6bba↔44b6 graph
gap. Test cheaply as an ablation *inside* H1's 44b6 graph gate: zeroed vs real mask
features. If a strong pretrained model *still* fails 44b6 with real features, then
44b6 needs explicit per-family handling and S1 (final selection) must hedge with a
family-tagged Pick A/Pick B rather than one universal model.

**H4 — calibration closeout (no new submission).**
Record: coarse high-range alignment PASS; fine affine/rank transfer STILL-UNRESOLVED
and rounding-bounded; no calibration-only probe. Next public probe attaches only to a
local-≥0.89 observation-distinct candidate (expected from H1). Keep 2 finals uncommitted.
**Bar note (§3.1):** local ≥0.89 is the submission-worthiness floor, not accepted-0.89;
the public 0.89 / RunPod gate needs local ≈0.895–0.900.

### Gate / governance (unchanged, carried forward)

- Frozen scorer only; commit locally per node; **no Kaggle submission, Kaggle-GPU,
  or RunPod launch without explicit human approval.**
- Zero-swap CPU discipline via two-stage isolation (never co-load V45 + Trackastra);
  meta-device/mmap loaders; 45-second clean idle preflight before any run.
- GPU escalation stays behind measured graph-level headroom + the P1 gate; RunPod
  only after an accepted **0.89**, ≥5% theory-backed expected gain, $10 cap.
- Mechanism ≠ family: a mechanism law needs ≥2 score-separated anchors; one APNCC
  point cannot establish one.

## 5. What to do next (ranked)

1. **H0 proxy-validity audit** — cheap, and it formally settles whether the whole
   screening layer should be deleted. Do this before spending another pilot.
2. **H1 Trackastra graph gate, 44b6-first** — resume the two-stage cache in a stable
   window; this is the one live path to > 0.881 that the evidence still supports.
3. **H3 mask-feature ablation** folded into H1's 44b6 gate — it may be the difference
   between H1 passing and joining the 44b6 graveyard.
4. **H2 background-abstention shell** only after H1 shows real graph-level signal.
5. **Prune** the Section 1 DEADs from the active frontier so they stop reappearing.

## 6. Decisions locked 2026-07-25 (human authority)

### 6.1 DEAD register (decision 1)

Promoted from PARK to **DEAD** — removed from the active frontier; not re-screened
without a new, named mechanism (LEDGER decision columns to be updated by the coordinator):

- Scalar recalibration / threshold / distribution normalization — `SCENCAL`,
  `POINTCTX-THRESH`, `APNCC-POLICY`.
- Hand-tuned graph-cost / motion knobs — historical sweeps, `GA-SP`, `LC-P4CTRL`
  (existing-output selection oracle 0.8768 < 0.88).
- Geometry / NCC / constant-velocity abstention on the extended-radius shell —
  `R12SHELL`, `MGSHELL`, `TEMPBG`.
- From-scratch learned rankers without pretraining or explicit domain adaptation —
  `POINTCTX`, `CROPLINK`, `TEMPRANK` (44b6 floor failed 4×).
- Full organizer `TemporalUNet3D` to convergence — `CTRL-COMPUTE` (~478 T4-h; DEAD
  under the 30-h/model cap).

**Conditional PARK (revisitable, NOT dead):** detection front-end (only if association
is solved and detection becomes the new bottleneck); division postprocessing is
superseded by the explicit fit-only unlock in §6.4; GT-conflict-trained gates (only
with unlabeled-abstention regularization).

### 6.2 Warm-start / convergence (decision 2)

The pretraining you proposed is **already embodied by the frozen Trackastra checkpoint**:
it is a foundation model for cell/nucleus-tracking association, and the competition
data itself is zebrafish embryo development — so the frozen 27.46M weights *are* the
embryo-pretrained initialization. Do not re-pretrain (that is the DEAD ~478 T4-h regime
and will not fit the budget).

The local-minima worry is largely misplaced for this design and secondary elsewhere:

- In H1 the backbone is **frozen** and only a small ridge/linear conflict head is fit.
  That objective is **convex** — a global optimum in closed form, no local minima, and
  warm-starting is moot.
- If the backbone were fine-tuned on 44b6, the dominant hazard is **not** a bad basin.
  A good pretrained init lands in a **wider, flatter** basin than random init — that is
  *why* pretraining works. The proven failure mode is **overfitting tiny 44b6 data**
  (few videos, 27M params → destroys the pretrained generalization that is the whole
  value) plus **negative transfer / catastrophic forgetting**. Every from-scratch model
  hit exactly this (44b6 precision 0.83–0.90).

**Convergence recipe if any training is authorized** (matches the G-02 compact-heads/
adapters mandate): keep the backbone frozen; escalate only to a **parameter-efficient
adapter (LoRA/bottleneck, <1% of params)**, never a full fine-tune; **select on the
44b6 held-fit graph delta** (nested OOF), not training/proposal loss; if unfrozen, use
discriminative/layer-wise LRs (tiny on backbone), warmup + cosine decay, weight decay,
dropout, gradient clipping, short schedules with early stopping, and **fold/seed
ensembling or SWA** to average across basins rather than chase one minimum. Realize the
curriculum as *frozen Trackastra (broad) → tiny 44b6 adapter (specialize)*, not a
re-pretrain.

**Decision:** this GPU window does **inference only** (frozen feature generation), not
training. Test whether the frozen 44b6 features clear the graph floor first (H1+H3);
escalate to a PEFT adapter only if frozen fails, selected on the 44b6 graph gate.

### 6.3 GPU window — use-it-or-lose-it (decision 3)

Free Kaggle quota (~8 h) resets in ~7 h. The blocker was host swap, not GPU; there is
**no preregistered GPU training job ready**, and rushing one would trip the LIT/M5 gate
and the §6.2 overfit risk. The one GPU-appropriate, high-value action that fits:

- **Run the already-implemented, feasibility-passed Trackastra full-pool feature
  generation on Kaggle GPU, for both 6bba and 44b6.** Kaggle's 16 GB RAM + T4×2 removes
  the V45/Trackastra co-residence swap problem entirely. Projected inference 1.213 h
  (6bba) — 44b6 adds proportionally; both fit inside 7 h with packaging margin.
- This directly unblocks H1 and produces exactly the caches the 44b6-extended graph
  gate needs. Frozen inference → not "initial training" → does not consume the 30-h
  training cap.
- **Runs in parallel with H0** (CPU/local proxy-validity audit), as requested.

**Caveat / fallback:** no GPU kernel exists yet; packaging the checkpoint (as a Kaggle
dataset) + scripts + data access is required, and the launch is human-executed. If a
clean packaged kernel cannot be verified before the reset, **reserve the quota** — sunk
free-quota is not a reason to launch a job that risks a mid-run guillotine or an
un-preregistered spend.

### 6.4 Division fitting dependency amendment (decision 4; approved 2026-07-31)

The human owner **unlocks fit-only division realizers from the
association-continuity prerequisite**. This supersedes only the sequencing rule
that required one association-continuity arm to pass both families before division
fitting. It does not promote a division mechanism or weaken any outcome gate.

Evidence basis:

- the frozen-N1 oracle repaired `38/42` division events at TP/FP/FN `38/0/4`,
  division Jaccard `0.904761904762`, and exact combined delta
  `+0.092067841519`;
- the edge-only current-pool projection caps near `0.896`, below the local
  planning range `0.9255–0.9422` at every realization;
- six subsequent association mechanisms realized approximately zero or negative
  usable graph lift; and
- four independent representations failed at the binding roughly `10^-4`
  prevalence operating point.

The authorized first wave consists of four separately preregistered,
CPU-first realizers against the same 38-event target: geometry/motion,
late-persistence, frozen-cache appearance, and their high-precision consensus.
GT repair bundles define outcomes only and are forbidden as input features.
Every arm runs `44b6` first, suppresses to one candidate per scorer-equivalence
event, requires zero duplicate survivors, and emits forks atomically through
frozen N1. PASS remains exact combined delta `>0`, division Jaccard `>=0.30`,
and nonnegative adjusted-edge, raw-edge, and node-recall deltas.

All other authority is unchanged: the frozen scorer and N1 compiler remain
immutable; `6bba` and CV57 stay closed until the 44b6 human barrier; submission,
Kaggle/GPU, RunPod, and any `>=5 h` GPU action retain their human gates; DEAD
authority and the §6.1 register remain unchanged.

### 6.5 Division falsifier population correction (decision 5; approved 2026-07-31)

The human owner corrects the **division falsifier scope only**. The prior
`44b6`-first rule is a cross-family generalization guard designed for
association mechanisms. It is not a valid falsifier for division when only
`9/38` repair-target events are in `44b6` and `29/38` are in `6bba`.

Accordingly, a division realizer is now screened once on the complete frozen
38-event repair-target population across both families. Its binding result is
the all-38 TP/FP/FN and division Jaccard together with the frozen-scorer
adjusted-edge, raw-edge, node-recall, and exact combined deltas over the
authenticated 57-fragment population. `44b6` and `6bba` results are still
reported separately as generalization diagnostics, but neither family alone
is a promotion or falsification gate.

This correction does not change any association arm: association mechanisms
retain their existing `44b6`-first kill line and may not open `6bba` after a
`44b6` failure. It also does not change the division PASS line, frozen N1 or
scorer, one-to-one/zero-duplicate rule, information firewall, or any external
human gate.

The first wave's geometry/motion observation—zero exact target matches among
the nine `44b6` targets—is therefore reclassified from a surface kill to
**INCONCLUSIVE-UNDERPOWERED**. Its measured `0/1/11` official `44b6` division
result and negative edge deltas remain immutable historical facts; only the
scope-level interpretation is corrected. A full-population application of the
unchanged frozen geometry/motion rule is required before the arm can receive a
division-surface verdict.

### 6.6 RAW3D-NOLINK-MVE scoped training decision (decision 6; approved 2026-07-31)

The human owner selects **Option 1** from the locked training-feasibility dossier and
opens the named, observation-distinct `RAW3D-NOLINK-MVE` node. This is a scoped
exception to the campaign closeout, not a reopening of the DEAD learned crop-linker
or any other representation supplier. The node must begin with a zero-weight crude
3-D endpoint-information probe on the complete frozen 44b6 DELETE population before
any weight acquisition, training, or GPU spend.

Before any outcome, the probe freezes its population, five video-group folds, raw
endpoint feature contract, sealed B0→A1→A0 order, metrics, and continuation line.
It reports against DINOv2-XY mean AUC `0.670411743453` and the measured operating
requirement AUC/CG0-budget retention `0.85/0.020393559928`. AUC may decide only
whether this zero-weight information source warrants a provenance stage; it may
never select a learned checkpoint. Any learned stage must select exclusively on
nested-OOF 44b6 held-fit graph delta. The H0 proxy ban explicitly covers
within-trajectory checkpoint selection: training loss, proposal metrics, and AUC
are reporting-only.

The provisional GPU envelope is `2–6` T4-hours. A projection below `5 h` may launch
under the standing authority only with the serial lock, authoritative quota
reconciliation, and a hard `5 h` kill; a projection at or above `5 h` returns to the
human barrier. Passing a learned 44b6 graph gate requires adjusted/raw/node deltas
`>=+0.005/0/0`, then and only then opens `6bba` and CV57. A prediction may be
submitted only if it is observation-distinct and local CV57 is at least `0.895`.
Quota expiry does not widen any scientific scope.

The preregistered zero-weight realization is now terminal. It completed all `41`
videos / `1,345,497` actions / `1,118` beneficial actions on CPU in `254.704 s`.
Its A1 mean/fold-floor AUC was `0.677967209/0.599374874`, full-population retention
was `5/1,118 = 0.004472272`, and A0 mean AUC was `0.646573736`. It therefore failed
the frozen `0.680411743` mean-and-fold-floor continuation line and the measured
`0.85/0.020394` surface. The RAW3D GPU/training branch did not open and consumed
zero Kaggle quota. This terminal result does not change any other human authority
recorded after this scoped decision.
