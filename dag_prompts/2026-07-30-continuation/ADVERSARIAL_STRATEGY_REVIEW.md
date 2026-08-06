# Adversarial strategy review — can the retained frontier credibly reach public 0.910?

**Date:** 2026-07-30  
**Disposition:** recommendation only. This review does not alter `LIVE_STATE.md`, the
ledger, tiers, gates, compute authorizations, or the human-locked campaign plan.

## Bottom line

There is **no evidence-backed four-week path from the present state to a public 0.910**.
The only retained branch that is both genuinely new and plausibly score-bearing inside
the current budget is **DPRC-O0** (DINOv2-register foreground patch correspondence),
but it is an unmeasured selector over the existing association-action population. It
should be retained as the lead *falsification* bet, not represented as a 0.91 plan.

`TAP-SYM-DIV` is the only new proposal with a theoretical effect scale large enough to
matter if it solved divisions, but its reported upside is GT-dependent and its
leakage-safe training plan is infeasible (at least 40 GPU-h, versus 25.47 GPU-h current
launch capacity). MCC reverse, MAIN's duration hazard, and NODETOPO neighbor
conservation are useful research ideas but have no demonstrated campaign input,
selector, or score-bearing route. They should not dilute the one available
score-bearing experiment.

The appropriate claim is therefore: **retain a narrow, conditional DPRC experiment;
retain division only as an off-budget option-value hypothesis; park/prune the other
continuation implementations for this four-week window.** A public 0.91 should be
treated as a discovery target requiring a new source of recall/division information,
not as an extrapolation from a selector passing the current 44b6 gate.

## What the evidence actually says

### Target arithmetic and the non-compounding rule

| Item | Evidence | Adversarial implication |
|---|---|---|
| Banked public result | APNCC is local `0.881522` and public `0.881`. | The observed public gap to `0.910` is `+0.029`, much larger than a routine ranking effect. |
| Planning conversion | `p0_parity.md` maps public `0.910` to local approximately `0.9255–0.9422`; fine transfer is explicitly unresolved. | A candidate needs about `+0.0440–+0.0607` local over APNCC just to enter the planning range. This is not a confidence interval and cannot certify public 0.91. |
| Current-pool ceiling | `LIVE_STATE.md` records an edge-only current-pool projection of about `0.896`. | Even the stated in-pool oracle is short by roughly `0.030–0.046` local. A better selector over the same pool is therefore not, by itself, a credible 0.91 mechanism. |
| Action-class oracles | Delete `+0.081898`, occupied swap `+0.018571`, topology `0`, and the division diagnostic's `+0.092068` arise from different frozen populations/baselines; division targets use GT. | These are **not additive forecast components**. Adding their deltas to APNCC would mix controls, families, action spaces, and—in the division case—oracle labels. |
| Transfer evidence | Four local/public pairs include a near-baseline rank reversal; P0B fine ranking remains unresolved. | A local near-tie cannot be promoted to a public claim. Any 0.91 narrative needs a materially larger, observation-distinct CV57 result and an approved submission. |

The campaign already has the right antidote to wishful arithmetic: full frozen 44b6
first, no video subsampling, then 6bba, then CV57. The subsampling audit found **0/12
usable cells**, so any proposal that relies on a tiny video result as an efficacy
decision is invalid rather than merely noisy.

### Gate severity and capacity

- The viable delete/swap selector must clear the *measured* all-fold bars, not the
  obsolete loose gate: delete AUC/retention `>=0.85/0.020394`; occupied swap
  `>=0.90/0.067910`, together with the B0/A0 lift clauses. The robust-p05 bars are
  still more stringent. Passing a synthetic fixture, a retrieval benchmark, or a
  mean-only AUC does not approach this standard.
- The GPU ledger has `3.6435/30.0` GPU-h debited, but actual launch capacity is the
  lower owner-authenticated **25.47 GPU-h**. No CPU/GPU authorization is active.
  `TAP-SYM-DIV`'s stated lower bound is 40 GPU-h before inference; it is not a
  scheduling problem that can be solved by squeezing the current allocation.
- Reverse Trackastra, the DINO R5 local/CPU/GPU routes, CELLECT's authorized launch,
  exact SWAP-O0, DELETE-O0, the 50-action topology restoration, and fixed-k AE-P5
  are terminal routes. Renaming their reducer, transport, or threshold is crowding,
  not a new mechanism.

## Branch-by-branch assessment

### 1. DINO DPRC-O0 — **retain as the only primary score-bearing falsifier**

**What is evidence.** DPRC is materially distinct from terminal R5: it excludes the
non-register weight and quarantined R5 inputs/rows, requires a separately pinned
register checkpoint, and uses foreground patch-cycle coverage rather than a pooled
endpoint feature. Its synthetic contract correctly catches a global-pooling failure:
the identity grid gives `C=1`, the spatially reversed grid gives `C=0.25`, uniform
tokens tie, empty foreground fails closed, and repeat is exact. That establishes only
that the formula is non-degenerate.

The external literature supports *parts* of the causal story, not its campaign effect.
The register paper reports smoother dense feature maps after adding register tokens
([Darcet et al., 2024](https://arxiv.org/abs/2309.16588)); DINO-Tracker supports
localized DINO features for correspondence, but crucially achieves its result through
per-video test-time training/refinement ([Tumanyan et al., 2024](https://arxiv.org/abs/2403.14548)).
That forbidden refinement is negative evidence for a frozen DPRC descriptor. Cell-DINO
does show that DINO-style self-supervision can encode fluorescent-cell morphology, but
it is domain-trained—not proof that an official natural-image DINO register weight,
an unvalidated channel mapping, and 4-by-4 crops transfer to this embryo problem
([Moutakanni et al., 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013828)).

**Headroom assessment (inference).** Delete has a large *label-oracle* ceiling, so a
strong new representation could be valuable. But DPRC does not create new detections,
edges, masks, or division candidates; it re-ranks existing delete/swap actions. Its
best plausible role is therefore to realize some of the current-pool headroom, whose
documented overall edge-only projection is still about `0.896`. It is credible as a
test of whether there is any remaining representation signal; it is **not credible as
a stand-alone route to 0.910**.

**Crowding and implementation risk.** The new supplier plus spatial observable clears
the formal no-retry distinction from R5. The practical risks are still severe:

- a register checkpoint/source/license and microscopy channel/crop transform are not
  authenticated;
- foreground masks/grid alignment may introduce segmentation-boundary artifacts;
- 4-by-4 reduction can erase the very fine morphology it claims to preserve;
- mutual local cycles can be brittle under normal motion, division, and crowded
  overlap; and
- a pass on the unit fixture says nothing about class/fold AUC, retention, or graph
  benefit.

**Compute feasibility.** The N3 design is bounded at 5 GPU-h with a `<=4` GPU-h
two-video extrapolation, so it fits the *numerical* remaining balance. It still needs
explicit human authorization and must not reuse any R5 artifact. It is the only
candidate here whose proposed campaign experiment can fit the current allocation.

**Recommendation:** **RETAIN, rank 1**, with no public-score forecast. Kill this exact
route immediately if source/weight/provenance or the raw channel/mask contract is
ambiguous, the calibration projects above 4 GPU-h, any R5 fingerprint is found, or
either class lacks a complete manifest. On a complete run, PARK-C it if any supplier ×
class fails the binding all-fold gate; a separation PASS merely earns a new graph
preregistration, not a promotion or a 0.91 claim.

### 2. TAP-SYM-DIV — **retain as off-budget option value; prune from the four-week execution portfolio**

**What is evidence.** The division audit is genuinely important: 38 of 42 CV57
division FNs are present-but-fragmented, and 38 exact repair bundles produce a
combined-score diagnostic ceiling of `+0.092067842` with division TP/FP/FN `38/0/4`.
The proposed raw-pixel channel is also scientifically distinct from the current action
pool. Time-arrow pretraining can capture time-asymmetric processes including division
([Gallusser et al., 2023](https://arxiv.org/abs/2305.05511)); raw daughter symmetry
has been used for unsupervised mitosis detection ([Gilad et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6662301/)).

**Why this is not current score headroom.** Only **1/38** diagnostic events is in the
current-pool tier that a deployable existing selector could reach; even perfect choice
there is `+0.002406490`, below the actionability line. The other 37 require the
unbuilt representation. The `+0.092068` number is an oracle, GT-dependent,
representation-conditioned upper bound—not a predicted gain, calibrated target, or
legal selection rule. Division fitting also remains dependency-locked until a single
association-continuity arm passes both 44b6 and 6bba with the required nonnegative
vectors. That prerequisite is currently zero-for-zero.

**Feasibility.** The N2 contract's leakage-safe form needs five fit-only encoders.
At the published estimate of about 8 GPU-h each this is at least **40 GPU-h**, with an
upstream default of at least 24 GB GPU memory. That violates both the weekly 30-h cap
and the 25.47-h launch capacity before held-side inference, source authentication, or
integration. Training one encoder across held raw frames would be a different,
transductive mechanism and would invalidate the intended outer-fold claim.

**Recommendation:** **RETAIN scientifically, rank 2 only by upside; allocate 0 GPU-h
in this window.** It should be re-opened only with a separately named, source-pinned
pretrained representation whose fold/data rights and resource envelope are proven
before any campaign read. Stop/park on no released checkpoint, unverified 2D-to-volume
contract, any held-frame encoder exposure, `>=24`-GB device mismatch, or a plan still
requiring five 8-hour trainings. Do not spend the remaining quota trying to turn the
40-hour proposal into a smaller unvalidated variant.

### 3. MAIN (AE-P7 duration hazard / AE-P2 adaptive appearance) — **research-only; no credible 0.91 headroom yet**

**What is evidence.** The continuation's N0/N1 establishes that phase duration and
cell-cycle context are biologically meaningful in other settings. It does *not*
establish a usable phase-state observation in this project. N2 is PARK because there
is neither a duration/phase interface nor an isolated fixture, and the existing test
environment would not test that information even if repaired. Fixed-k AE-P5 is more
decisively negative: it applies to zero of 44,161 components and is byte-identical to
APNCC.

**Adversarial reading.** A duration hazard needs a trustworthy phase estimate. Without
one, it is a threshold over unavailable latent labels; with a learned phase estimator,
it becomes a new representation/training project rather than the advertised cheap
hazard. Likewise, component-adaptive `k` may avoid AE-P5's structural inertia, but if
it only reconditions existing appearance/ranks on component size it is dangerously
close to the already DEAD selection/calibration family. It has no demonstrated
opportunity count, action coverage, or all-fold separation.

**Headroom and compute.** Neither candidate has a known action-class ceiling beyond
the small occupied-swap pool (`+0.018571397` oracle on the frozen diagnostic).
Neither adds recall. There is no empirical basis to say that either can supply the
`+0.044–+0.061` local planning gap.

**Recommendation:** **PRUNE AE-P7-DURATION-HAZARD as an execution candidate for this
window; RETAIN AE-P2 only for one no-score structural novelty review.** That review
must prove (before implementation) an available non-label phase/appearance input, a
nonzero action rate distinct from AE-P5, and a causal observable not reconstructible
from an existing scalar/rank. Stop if any condition fails. No GPU, graph, or 44b6 run
should follow merely from a synthetic fixture.

### 4. Reverse MCC contour-return — **park as source feasibility, not a score branch**

**What is evidence.** MCC is genuinely different from authentic Reverse Trackastra:
its landing/round-trip/normal statistics are based on contour correspondence rather
than a reversed association matrix. The static falsifier correctly rejects zero flow,
wrong contours, and a permuted return. The primary paper is directly about
live-cell-contour point tracking and shows why a snap-to-contour step is needed to
avoid a trivial cycle shortcut ([Jang, Lee & Kim, 2023](https://arxiv.org/abs/2303.08364)).

**Why the current path is weak.** The cited method follows contour points, not cell
identity through crowded 3-D lineage events. Its public source has no observed released
checkpoint; source training reportedly takes roughly a day on a TITAN RTX. The proposed
2 GPU-h task is only a public-source compatibility receipt. It cannot read campaign
data, make a feature cache, produce an action, or test score headroom. Moreover,
ordered instance contours/mask semantics for this campaign are not authenticated in
the current record; the campaign's own authentic-mask audit found no complete
mask/wrfeat supplier. A successful pilot would therefore leave the crucial bridge
unproven.

**Recommendation:** **RETAIN as a low-cost provenance question, but PRUNE it from the
four-week score portfolio (rank 4).** Do not spend its 2-GPU-h pilot allocation unless
the owner specifically values the independent source receipt. Terminally PARK this
version on absent rights, no deterministic checkpoint inside the cap, unavailable
ordered contour semantics, or any proposed campaign integration that reuses
Trackastra-derived fields. A source-only success is not a rationale to request a
campaign graph run.

### 5. NODETOPO neighbor conservation — **prune this implementation from the portfolio**

**What is evidence.** The broader topology surface once had P0 oracle value, but the
only complete frozen restoration population has 50 actions and **zero beneficial**
events. Its independently repeated 44b6 graph result is `-0.00000343615 / 0 / 0`.
The proposed neighbor-conservation test has no extractor or fixture; its citations
establish that local spatial relationships can help tracking elsewhere, not that this
candidate population contains the required signal.

**Adversarial reading.** The new proposal no longer restores nodes and has not said
which legal existing action it selects, how its neighborhood remains stable through
division/segmentation change, or what label-free opportunity it exploits. It risks
becoming a softer replay of an exact mechanism with no beneficial actions. Creating a
synthetic signature only proves that a hand-written score can respond to a hand-written
signature; it does not locate score-bearing events.

**Recommendation:** **PRUNE NODETOPO-NEIGHBOR-CONSERVATION for four weeks (rank 5).**
Keep the broader `NODETOPO-O0` catalogued status untouched, but require any future
new mechanism to name (i) a legal non-restoration action class, (ii) an authenticated
neighborhood input, and (iii) a label-free opportunity count before literature or a
fixture earns implementation time. Do not rebuild the raw universe or replay the
50-action gate.

## Ranked four-week portfolio

This is a resource-allocation recommendation, not permission to run any step that is
currently human-gated.

| Rank | Branch and objective | Four-week allocation | Why it earns the slot | Explicit stop criterion | Retain/prune recommendation |
|---:|---|---|---|---|---|
| 1 | `DPRC-O0`: test a new frozen spatial representation on delete and swap | One preparation/authentication review; **at most one** owner-approved `<=5` GPU-h route only if all admission tests pass | Only candidate that is new, directly selector-bearing, bounded inside capacity, and can test both valuable classes | Source/weight/channel/mask ambiguity; `>4` GPU-h projection; R5 contamination; no complete class manifest; or failure of any all-fold class gate | **Retain** as lead falsifier; no 0.91 promise |
| 2 | Division: preserve `TAP-SYM-DIV` option value without launching it | 0 GPU-h; one paper/checkpoint/rights feasibility review only | Potentially large oracle division value, but no selector exists | No independently releasable pretrained route that fits folds/capacity; any need for five 8-h trainings; unverified XY/3-D contract | **Retain off-budget; prune from execution** |
| 3 | MAIN: prove or reject an actually available `AE-P2` observation | One small no-score design/structural gate; no graph compute | Cheapest way to prevent continued investment in a rebranded AE-P5/scalar rule | Input reconstructible from existing ranks; zero/non-distinct action rate; no phase/appearance input; no objective-linked action specification | **Prune P7 now; retain P2 only through this gate** |
| 4 | MCC: determine whether an independent source can even be reproduced | 0 GPU-h by default; optional owner-valued source-only pilot, never campaign integration | Genuine independent observation, but no score path this period | Rights/checkpoint/ordered-contour absence, resource breach, or any Trackastra coupling | **Park/prune as score work** |
| 5 | NODETOPO neighbor conservation | No implementation or compute | Exact topology realization already has no signal | Missing legal action class/opportunity input is already sufficient | **Prune implementation** |

### Sequencing and portfolio guardrails

1. **Do not run parallel GPU bets.** Complete DPRC's provenance/channel/crop admission
   package before any owner decision. This avoids repeating R5's transport race and
   protects the remaining capacity for one interpretable result.
2. **Use the division feasibility review as a kill, not a scavenger hunt.** A new
   released source/weight is a new named mechanism and must restart at N0; it cannot
   be smuggled in as a TAP execution repair.
3. **Do not call a representation pass a campaign win.** A DPRC separation PASS is
   still only permission to propose the 44b6 graph gate. It must then clear 44b6,
   6bba, CV57, and the human submission gate before it can inform public 0.91.
4. **Use a target checkpoint after each real graph result.** If a full CV57 candidate
   cannot plausibly exceed the current-pool `~0.896` projection by adding a new,
   authenticated recall/division channel, stop spending on selector variants. The
   missing `~0.030–0.046` local planning gap is the evidence requirement, not a
   motivational target.

## Portfolio-wide stop conditions

Stop the retained continuation strategy—not just an individual branch—if either of
the following happens:

1. A complete, integrity-valid DPRC run fails either delete or swap against the
   binding measured all-fold requirement. Its result would be the first direct test
   of a new frozen spatial supplier on the surviving action classes; repeating it
   with crop, mask, `k`, or threshold tweaks would be post-outcome selector tuning.
2. No division representation can be specified within the weekly cap using
   fit-only raw data and an independently authenticated source. In that event the
   only theoretical high-upside branch remains unavailable, and 0.91 should be
   declared **not currently supported by the retained frontier** rather than pursued
   through increasingly crowded re-ranking variants.

Conversely, do **not** loosen the stop merely because a branch has attractive
literature, a synthetic fixture, a source-only pilot, an action oracle, or a
proposal-level separation metric. Each has already been shown, in this campaign or
in the cited papers, to be insufficient for the frozen graph objective.

## Evidence sources consulted

- Campaign state, target and action-surface evidence: `reports/g02/screening/LIVE_STATE.md`,
  `campaign_narrowing_and_hypothesis_dag_v1.md`, and `p0_parity.md`.
- Continuation outputs: all five lane result sets under
  `dag_prompts/2026-07-30-continuation/{dinov2,reverse,division,main,nodetopo}/runs/`.
- Retained terminal evidence: DINO, reverse, representation/separation, selector
  requirement, division, NODETOPO, contracts, swap, and delete terminal reports.
- Primary external checks: the DINO registers paper, DINO-Tracker, Cell-DINO,
  time-arrow prediction, symmetry-based mitosis, and MCC contour-tracking papers
  linked inline above. Literature establishes mechanism plausibility only; all
  score-headroom and portfolio rankings labeled as inferences remain contingent on
  the frozen campaign gates.
