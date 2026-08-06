# N1 — Division alpha (the uncrowded lane) — RUN REPORT

Node: `dag_prompts/2026-08-06-alpha-hunt/N1_DIVISION_ALPHA.md`
Date: 2026-08-06. Research only. No code written to `scripts/` or `src/`, nothing launched,
no GPU, no submission.

Scorer sources read in full (frozen, hashes pinned in the evidence file below):

- `external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py`
  (`0635c386…60f4dc9`)
- `external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py`
  (`cfdd596e…67ca7444`)
- `external/kaggle-cell-tracking-competition/metrics.md`

Evidence file:
`reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_S00_NATIVE_DIVISION_SCORE.json`

**Verdict: GO.** Named alpha: **`DIV-EVOP`** — the metric-derived division operating
policy (recall-first at an exact, derived 4% precision bar; FP priced only inside the
sparse-GT footprint; free 3rd fork branch), instantiated on V32-native edge probability.
Two candidates pass both rigor gates; one further candidate is labelled a research bet.

---

## 0. The governing equation (exact, not an approximation)

`score = adj_edge_jaccard + 0.1 · division_jaccard` (`metrics.py:34`, `metrics.py:377`).

In `score_divisions`, `candidates` is keyed by **every** GT dividing node
(`division_metrics.py:365, 384`), and

```python
scores = {div: int(div in pairing) for div in candidates}
...
tp = sum(result.scores.values())
fn = len(result.scores) - tp
```

so `tp + fn ≡ G`, the number of GT divisions — a **constant** we cannot change. Therefore

> **`division_jaccard = tp / (G + fp)`**

Numerical verification against the banked evidence file (both exact):

| population | tp | fp | fn | G = tp+fn | tp/(G+fp) | reported J |
|---|---|---|---|---|---|---|
| cv57 | 1 | 11 | 12 | 13 | 1/24 = 0.0416667 | 0.041666666 ✓ |
| shard-0 (27 vids) | 2 | 52 | 23 | 25 | 2/77 = 0.0259740 | 0.025974025 ✓ |

### 0.1 The exact acceptance threshold — the single most important number in this node

Add `K` forks, of which a fraction `p` are correct. `J' > J` iff

```
(tp + pK)/(G + fp + (1-p)K)  >  tp/(G + fp)
⇔ p·(G + fp)  >  tp·(1 - p)
⇔ p  >  tp / (G + fp + tp)
```

and since `J = tp/(G+fp)`,

> **`p* = J / (1 + J)`**

This is exactly the Jaccard analogue of the known F1 result (optimal F1 threshold = `F*/2`;
Lipton, Elkan & Narayanaswamy, *Thresholding Classifiers to Maximize F1 Score*,
ECML-PKDD 2014). Consistency check: `F = 2J/(1+J)` ⇒ `F/2 = J/(1+J)`. Identical. It is a
fixed point: raising `J` raises the bar, so the sweep must be iterated to convergence.

**At our operating points:**

- cv57: `p* = 0.0416667 / 1.0416667 = 0.0400` → **4.00%**
- shard-0: `p* = 0.025974 / 1.025974 = 0.02532` → **2.53%**

Equivalently: **one division TP is worth `(G+fp)/tp` = 24 division FPs on cv57, 38.5 on
shard-0.** Emitting a candidate fork is score-positive at **4% precision**. Every division
policy in the public tier, and every division policy this campaign has run, has been tuned
for precision in the 50–100% region. That is the whole thesis of this node.

Corollary that must be written down because it contradicts a former internal constraint:
LIVE_STATE records "near-zero-FP division operation" as a former lane constraint. Under
`J = tp/(G+fp)` a near-zero-FP operating point is **provably suboptimal** whenever any
available candidate pool has >4% precision. It should never be re-imposed.

### 0.2 The edge term does not constrain division aggression

Adding a fork adds an *edge*, not a node, so the node-count penalty
(`adjusted = J·(1 − 0.1·(T_pred−T_true)/T_true)`, `metrics.py:447`) is untouched. On the
edge side an added edge is only counted at all if `pred_valid = out_valid | in_valid`
(`metrics.py:193`), i.e. its source matches a GT node with out-degree > 0 or its target
matches a GT node with in-degree > 0. Cost of one edge FP ≈ `J_e / N_e` where
`N_e = edge_tp+edge_fp+edge_fn`. Gain of one division TP = `0.1/(G+fp)` ≈ `0.0042` (cv57).

> Break-even: we can afford **≈ `0.0042 · N_e / J_e` edge FPs per division TP**.

Using the nearest banked CV57 edge counts (`edge TP/FP/FN = 34389/2308/1879`, i.e.
`N_e ≈ 38,576`, `J_e ≈ 0.9`): cost per edge FP `≈ 2.3e-5`, gain per division TP `≈ 4.2e-3`
⇒ **~180 edge FPs are affordable per division TP.** A fork costs at most **one** edge FP
(see §3.4 — branches 3+ are dropped before FP counting), so the edge term is not a
constraint on this lane by a factor of ~10². *(Probe P0 re-pins `N_e`/`J_e` on the live V32
substrate; the conclusion survives any `N_e ≥ 250·J_e`, i.e. `N_e ≥ ~225`.)*

---

## 1. Task 1 — which rule collapses 368 forks to 1 TP

The collapse is **two-stage**, and the banked per-video numbers let us size both stages
exactly rather than guess.

### 1.1 Stage 1 — the evaluability gate removes 96.7% of forks

`fp_forks` is built from exactly three sets (`division_metrics.py:389`):

```python
fp_forks = (considered | evaluable_forks | invalid_forks) - tp_forks
```

`tp + fp = 12` on cv57 ⇒ `|considered ∪ evaluable_forks ∪ invalid_forks| = 12`. We emitted
**368** forks. **356 of 368 forks (96.7%) were never evaluated by the metric at all.**

This is not the rule that costs us TPs — it is the rule that makes forks nearly free
(§2). But it explains why fork *count* and division score are almost decoupled.

### 1.2 Stage 2 — the deciding rule for the 12 FNs is `local_forks`, i.e. **we never place a fork at the anchor**

The candidate set for a GT division is built here (`division_metrics.py:365-381`):

```python
for div_node, matched_pred in matched.items():
    matched_nodes = _matched_division_nodes(_matched_node_attrs(matched_pred), gt_divisions[div_node], div_node)
    if matched_nodes is None:
        candidates[div_node] = set()
        continue

    parent_ids, daughter_ids = matched_nodes
    local_nodes = parent_ids | {
        successor for parent_id in parent_ids for successor in matched_pred.successors(parent_id)
    }
    local_forks = local_nodes & pred_div_nodes
    considered |= local_forks
    candidates[div_node] = {
        pred_div
        for pred_div in local_forks - invalid_forks
        if _is_strongly_connected_division(matched_pred, pred_div, parent_ids, daughter_ids)
    }
```

`local_forks` is the hard bottleneck: a fork can only ever become a TP if it is
**`parent_ids` itself or an immediate successor of `parent_ids`**, where `parent_ids` are
prediction nodes matched (≤7 µm, same-timepoint optimal assignment) to the GT divider or
its immediate predecessor (`division_metrics.py:216-217`):

```python
gt_parent_ids = {divider_id, *gt_div.predecessors(divider_id)}
parent_ids = {pred_id for pred_id, gt_id in node_to_gt.items() if gt_id in gt_parent_ids}
```

Sizing it: `considered` ⊆ (tp ∪ fp) = 12 forks total, and — see §1.3 — **at least 7 of the
11 cv57 FPs are in videos with zero GT divisions**, where the `for div_node …` loop never
executes and `considered` is provably empty. So **at most 5 of our 368 forks were ever
`considered` as candidates for any of the 13 GT divisions.** With 12 FNs, the dominant
failure is *no candidate fork existed at the anchor*, not *a candidate was rejected*.

Cross-check against banked oracle evidence: the GT-dependent frozen-N1 replay achieves
division `38/0/4`, J = `0.904761904762`, combined `+0.092067841519`
(`tiered_surface/branches/sa_division_value/REPRESENTATION_CLOSURE_RECONCILIATION.json`).
An oracle reaching 38/42 proves `_matched_division_nodes` almost never returns `None` —
detection coverage of the division window is **not** the binding constraint. The binding
constraint is that we do not fork at the right node.

**Answer to Task 1:** the killing rule is `local_forks` (the *local parent anchor* clause,
`division_metrics.py:372-376` + `_is_strongly_connected_division:260-261`), operating as a
**recall** constraint. The two-distinct-daughter bipartite test, cross-component evidence,
and unmerged-branch tests are *not* what costs us the 12 FNs — they never get the chance,
because no candidate fork is in the window.

### 1.3 The 11 FPs: a hard derived attribution

Per-video decomposition from the evidence file (`samples[]`, cv57 rows). `G_video = tp+fn`:

| video | forks | G | tp | fp |
|---|---|---|---|---|
| 44b6_341df25f | 20 | 1 | 0 | 0 |
| **44b6_87bba6c4** | **106** | **0** | 0 | **5** |
| **44b6_c96cfa10** | **110** | **0** | 0 | **2** |
| 6bba_07e24132 | 60 | 2 | 0 | 2 |
| 6bba_2819ca14 | 1 | 1 | 0 | 0 |
| 6bba_4ffd3da3 | 49 | 3 | 0 | 1 |
| 6bba_6ca87370 | 11 | 1 | 0 | 0 |
| 6bba_7f87b3d8 | 5 | 3 | 0 | 0 |
| 6bba_b204cac7 | 6 | 2 | 1 | 0 |
| **total** | **368** | **13** | **1** | **11** |

Two videos contain **zero GT divisions** yet carry **216/368 forks (59%) and 7/11 FPs
(64%)**. In a video with no GT divisions, `matched` is empty, so `considered = ∅`. Those 7
FPs are therefore *provably* from `evaluable_forks` or `cross_component_forks` /
`malformed_forks` only:

```python
evaluable_forks = {
    pred_id for pred_id in pred_forks
    if pred_id in pred_to_gt and gt_graph.out_degree(pred_to_gt[pred_id]) >= 1
}
```

i.e. **a fork placed on a node that matches an annotated, non-terminal GT cell**, or a fork
whose two branches land in two different GT weak components (a *lineage merge*, not a
division). Those 7 FPs have **zero possible upside** — no TP is reachable in those videos.

> **Derived operational statement:** 64% of our division FPs are not "we guessed the wrong
> division", they are "we forked on an annotated non-dividing track" or "we merged two
> annotated lineages". Those are structurally different errors and admit a structural,
> target-free filter (candidate `DIV-CONSERVE`, §4.2).

### 1.4 Secondary rules, quoted, for completeness

**Directed local topology + two distinct daughter branches** (`:260-271`):

```python
pred_parent_ids = {pred_div, *pred_graph.predecessors(pred_div)}
if pred_parent_ids.isdisjoint(parent_ids):
    return False

pred_lineages = [{child, *pred_graph.successors(child)} for child in pred_graph.successors(pred_div)]
lineage_edges = {...}
return len(_bipartite_max_matching(list(lineage_edges), lineage_edges)) >= 2
```

Derivable timing window: the anchor is the GT divider (t) or its predecessor (t−1); the
fork is the anchor or an immediate successor. **The fork may therefore sit at t−1, t, or
t+1 and nowhere else** — exactly the ±1 frame the prose claims. GT daughter evidence spans
GT child (t+1) and GT grandchild (t+2); pred evidence spans the fork's child and
grandchild. The slack is a clean ±1 frame, hard-stop at ±2.

**Unmerged branches** (`_branch_component_evidence:424, :430`):

```python
if set(graph.predecessors(child)) != {pred_div}:
    return None, True          # malformed
...
if any(set(graph.predecessors(node)) != {child} for node in grandchildren):
    return None, True          # malformed
```

This is the **only FP channel with no GT term at all**. For a forest-structured submission
(every node ≤ 1 parent) `malformed_forks = ∅`. Any in-degree ≥ 2 node adjacent to a fork is
an unconditional, GT-independent FP — see probe P2.

**Cross-component rejection** (`:477-490`): a fork is invalid if ≥2 of its child branches
carry GT-component evidence from ≥2 distinct components. Evidence needs a matched direct
child, or unanimous matched grandchildren when the child is unmatched.

---

## 2. Task 2 — the metric asymmetry, established from the code

**Claim: a predicted fork is eligible to become an FP only if it is spatially/temporally
adjacent to the sparse GT annotation. Otherwise it is invisible — not TP, not FP, not FN.**

Verified channel by channel against the code, not the prose:

| FP channel | code | requires GT proximity? |
|---|---|---|
| `evaluable_forks` | `:469-472` | **Yes** — fork must match a GT node (≤7 µm, same t, optimal assignment) with `out_degree ≥ 1`. |
| `considered` | `:376` via `parent_ids` at `:216-217` | **Yes** — fork must be at, or an immediate successor of, a pred node matched to a GT divider/grandparent. Note `considered` is only populated when `_matched_division_nodes` ≠ `None`. |
| `cross_component_forks` | `:477-490` | **Yes** — needs ≥2 child branches each with a matched child (or unanimous matched grandchildren). |
| `malformed_forks` | `:424, :430` | **No** — pure prediction-graph topology. Empty for a forest submission. |

Three additional exemptions read straight off the code:

1. **Annotation-terminus exemption.** `gt_graph.out_degree(pred_to_gt[pred_id]) >= 1` — a
   fork matched to the *last* annotated node of a GT track is **not** evaluable. Forks at
   annotation termini are free.
2. **Detection-gap exemption.** If `_matched_division_nodes` returns `None`
   (`:222`: no matched parent, or <2 matched daughter lineages), then
   `candidates[div] = set()` and `considered` gains **nothing**. A GT division we cannot
   see costs one FN and *cannot* generate FPs. **Trying is never worse than not trying.**
3. **Empty-GT-division exemption.** A video with no GT divisions cannot generate
   `considered` FPs at all (only `evaluable`/`cross-component`).

**Empirical confirmation of the 97% figure:** `tp + fp = 12` of `368` forks ⇒
**356/368 = 96.74% of emitted forks were silently ignored.** The stated reading is
confirmed by the code and by the numbers, independently.

### 2.1 What the asymmetry actually implies — and what it does *not*

The naive reading ("forks are free, so spam them") is **wrong**, and the boundary case
proves it. Push `K → all nodes`: every node in the annotated footprint becomes an
`evaluable_fork`, so `fp` grows with the *GT node count* (hundreds per video) while `tp` is
capped at `G` (≈1–3 per video). `J = G/(G + O(10²·V))` → ~0.004. Worse than today. The other
extreme, `K = 0`, gives `J = 0/G = 0`. Also worse than today.

**There is an interior optimum, and the code tells us exactly where it is.** Since forks
outside the annotated footprint are free in *both* terms (§0.2 — an edge whose endpoints
match nothing is not `pred_valid` either), they contribute exactly zero expected value.
Fork emission should be **concentrated, not uniform**, and inside the footprint it should be
**aggressive to a 4% precision bar** — not the ~99% precision bar the public
`add_safe_divisions_postlink` caps (0.76% of frame sources / 0.375% of edges) are tuned for.

> **Answer to Task 2: the metric is decisively recall-first at our operating point, and it
> is spatially targeted, not uniform.** Emit **more** forks, but only where they can be
> evaluated; a candidate needs **4% precision** (cv57) / **2.5%** (shard-0) to be worth
> emitting. Uniform precision control is the wrong instrument entirely.

This is a property of the published metric — the ignore rule is stated in `metrics.md`
line 118 ("unmatched, structurally valid forks with no local GT evidence are ignored") and
is a direct consequence of the organizers' *stated* sparse-annotation design (`metrics.md`
lines 11–13, 34). No fabrication is involved. The prose frames it as a **leniency**; the
code makes it a **targeting instruction**. That gap is the crowding moat (§5).

---

## 3. Task 3 — the achievable ceiling, given the fixed detection front-end

### 3.1 Hard structural ceiling

`tp ≤ D`, where `D` = number of GT divisions whose window satisfies
`_matched_division_nodes ≠ None` — i.e. detection supplies a node ≤7 µm from the GT divider
(or its parent) **and** nodes in ≥2 of the GT daughter lineages. `fp ≥ 0`. So

```
division_jaccard_ceiling = D / G      (achieved only at fp = 0)
score_ceiling            = 0.1 · D / G
```

### 3.2 The banked measurement of `D`

Already measured, GT-dependent, 57 videos, frozen-N1/APNCC substrate: **division
`38/0/4`, `J = 0.904761904762`, combined `+0.092067841519`** (adjusted/raw/node
`+0.001591651042/+0.001607524868/0`). So on that substrate **`D/G = 38/42 = 90.5%`** — the
detection front-end is *not* the binding constraint, and ~0.090 of the 0.1 division term is
structurally reachable.

Caveats that must travel with that number: (i) it is an **oracle** ceiling, not a promise —
the same LIVE_STATE entry says representation closure leaves the honestly-available
diagnostic at only `+0.002406489982`; (ii) it was measured on APNCC, whereas our live
substrate is public-V32; (iii) the banked TP curve is `+0.059523810 / +0.053191489 /
+0.048076923` at `0 / 5 / 10` FP — i.e. **FP tolerance is broad**, exactly as `J = tp/(G+fp)`
predicts.

### 3.3 Realistic ceiling on the live V32 substrate (cv57)

`J = tp/(13 + fp)`, `0.1·J` in score:

| tp | fp = 11 | fp = 4 | fp = 0 |
|---|---|---|---|
| 1 (today) | 0.0417 → **+0.00417** | 0.0588 → +0.00588 | 0.0769 → +0.00769 |
| 3 | 0.125 → +0.0125 | 0.176 → +0.0176 | 0.231 → +0.0231 |
| 6 | 0.250 → +0.0250 | 0.353 → +0.0353 | 0.462 → +0.0462 |
| 12 (≈D) | 0.500 → +0.0500 | 0.706 → +0.0706 | 0.923 → +0.0923 |

**Even the pessimistic lane (tp 1→3, fp unchanged) is 3× the currently-banked
+0.0042.** FP suppression alone (11→4, tp=1) is worth only +0.0017. **Recall dominates FP
suppression by roughly an order of magnitude — but both are positive and they are not in
conflict below the 4% bar.**

### 3.4 A code-only ceiling extension: the third branch is free (`DIV-K3`)

The out-degree cap lives **only in the edge path** (`metrics.py:140-153`) and is applied
**before** the FP tally:

```python
edge_attrs = edge_attrs.with_columns(
    pl.col(EDGE_ID).rank("ordinal").over(EDGE_SOURCE).alias("_out_rank"))
...
edge_attrs = edge_attrs.filter(pl.col("_out_rank") <= 2).drop("_out_rank")
```

`edge_fp` is then computed on the *filtered* frame (`metrics.py:313-316`). The division path
has **no cap**: `pred_div_nodes = {n for n in ... if out_degree(n) >= 2}` (`:354-357`) and
`pred_lineages` iterates over *all* successors (`:264`).

Consequences, all derived from the code:
1. A node's **3rd and later outgoing edges cannot become edge FPs** — they are dropped.
2. The node still counts as a fork for the division metric, and **all** of its branches are
   eligible to satisfy the two-distinct-daughter bipartite matching.
3. The surviving two edges are the two with the **lowest `EDGE_ID`**, i.e. insertion order —
   which we control.
4. Cost: extra branches raise the chance of triggering `cross_component_forks` (`:479`
   iterates all children).

So the marginal edge cost of a fork is capped at **one** edge FP regardless of how many
daughter candidates we attach. **Two gating unknowns before this can be used:** (a) does the
Kaggle-side submission ingestion reject out-degree > 2? (b) our own strict graph validator
currently rejects it (shard-2 was rejected for "two parents have outdegree 3"), so that is
*our* constraint, not the scorer's — a movable constraint, but a deliberate decision, and it
emits a scorer warning. Flagged, not assumed.

---

## 4. Task 4 — candidate mechanisms

DEAD-list checked (`LIVE_STATE.md:1700-1712`). Nothing below re-proposes: deterministic
recall / shell abstention (`R12SHELL`, `MGSHELL`, `TEMPBG`); from-scratch rankers
(`POINTCTX`, `CROPLINK`, `TEMPRANK`, `TEMPHYBRID`, `TRACKCTX`); scalar recalibration /
thresholding of a **proxy** (`SCENCAL`, `POINTCTX-THRESH`, `APNCC-POLICY`); hand-tuned
graph-cost / motion knobs (`GA-SP`, `LC-P4CTRL`); organizer `TemporalUNet3D` training;
`NDR-FBC-local-translation-v1`; frozen Trackastra zero-wrfeat H1; `C1-TA-HOCT-LITE`.
Divisions are explicitly **Conditional-PARK (revisitable)**, so this node is open.

Also checked against the banked division negatives, which are strictly binding:
- geometry/motion event detection: **FAIL** (`0/1/11`, J=0; parent-event head AUROC
  `0.521521`, AP `3.1367e-5` — random, under a contract that **excluded raw appearance**);
- persistence: **FAIL** (`0/49/11`, J=0);
- appearance: **never actually screened** — "failed closed before prediction because no
  existing frozen cache contained adequate physical intensity fields";
- 3-of-3 consensus: abstained everywhere;
- blanket aggression on APNCC (`unfiltered_geometry_cap5`, 239 forks): division `0/11/42`,
  gain was **edge-only** `+0.000113406777`;
- conditional daughter-pair head: **SOLVED** (AUROC `0.897081`, AP `0.368520`).
- Recorded conclusion: *"division event detection has no working mechanism; conditional
  daughter assignment is solved."*

Two facts change the picture and license this node: (i) the substrate is now **public-V32**,
not APNCC — and V32 native branching is **the only target-free mechanism in campaign history
that has produced a division TP** (2/25 shard-0, 1/13 cv57); (ii) among *evaluable* forks
V32's native precision is **1/12 = 8.3%**, which is **above the derived 4% bar**. At the
margin the correct action is therefore to **add**, not subtract.

### 4.1 `DIV-EVOP` — metric-derived EV-optimal division operating policy ★ ALPHA

- **Mechanism.** Stop treating division emission as a precision problem. (a) Build a *pool*
  of candidate `(parent, frame ∈ {t−1,t,t+1}, daughter-pair)` triples from V32's own linker
  output — native forks **plus** the candidates the geometric gate currently suppresses.
  (b) Rank them by the frozen network's own second-child `edge_prob` (an appearance-bearing
  observable, not geometry/motion). (c) Choose the emission count `K` by **directly
  maximising `J = tp/(G+fp)` on the frozen scorer**, iterating the fixed point
  `p* = J/(1+J)` to convergence. (d) Attach the top-3 daughter candidates per accepted fork
  (`DIV-K3`, §3.4) so daughter-selection error costs nothing extra — legitimate because the
  daughter head is the *solved* half.
- **Novelty technique.** First-principles re-derivation (the closed form `J = tp/(G+fp)` and
  `p* = J/(1+J)`) + constraint inversion (the "near-zero-FP division operation" and the
  0.76% public cap are movable, and provably wrong) + boundary reasoning (§2.1 rules out
  both extremes and localises the optimum).
- **Crowding.** ~30–80 / 1000. Ranking by model probability is beta; the alpha is the
  *operating point*. Deriving `p* = J/(1+J) ≈ 4%` requires reading `score_divisions`
  closely enough to see that `tp+fn ≡ G`, and realising the FP channel is gated on sparse-GT
  proximity. `metrics.md` presents the ignore rule as leniency, which actively misdirects.
- **Grounding.** (i) Exact derivation above, numerically verified on two independent
  populations. (ii) Lipton, Elkan & Narayanaswamy, *Thresholding Classifiers to Maximize F1
  Score*, ECML-PKDD 2014 — optimal F-threshold = `F*/2`; the Jaccard form `J/(1+J)` is
  identical under `F = 2J/(1+J)`. (iii) Elkan, *The Foundations of Cost-Sensitive Learning*,
  IJCAI 2001 — asymmetric-cost decision thresholds. (iv) Campaign-internal: the banked TP/FP
  curve `+0.059523810/+0.053191489/+0.048076923` at `0/5/10` FP is exactly the `1/(G+fp)`
  shape and independently confirms broad FP tolerance.
- **Not DEAD-adjacent, and here is why.** It is *not* `SCENCAL`/`POINTCTX-THRESH`: those
  recalibrated a classifier against a **proxy** metric. This sweeps `K` against the **frozen
  official scorer itself**, with a threshold that is *derived, not fitted* (zero free
  parameters). It is not a from-scratch ranker (no training; frozen V32 head). It is not
  geometry/motion (the observable is a learned edge probability).
- **CHEAP LOCAL FALSIFICATION (CPU-only, banked shard-0 outputs, no GPU, no submission).**
  Sweep `K` over the ranked candidate pool; rescore with the frozen scorer on cv57 and on
  all 27 shard-0 videos.
- **PREREGISTERED KILL (written before running).**
  - **KILL** if `max_K division_jaccard(cv57) < 0.0625` (< 1.5× the current 0.041667)
    **AND** `max_K division_jaccard(shard-0 27) < 0.0390` (< 1.5× the current 0.025974).
  - **KILL** if the optimal `K` is `0` on both populations (i.e. the pool has no candidate
    above the 4% bar).
  - **KILL** if the combined-score delta at the cv57-optimal `K` is `≤ 0` when re-measured
    on shard-0 (transfer failure / cv57 overfit).
  - **GO** requires a positive combined delta on **both** populations *and* `tp` strictly
    increasing (an FP-suppression-only result is a different, weaker verdict:
    `PARTIAL — FP-lane only`).

### 4.2 `DIV-CONSERVE` — cell-count conservation as a target-free lineage-merge filter

- **Mechanism.** §1.3 proved ≥7/11 cv57 FPs are `evaluable`/`cross-component` forks in
  videos with **no** GT divisions — i.e. forks that either sit on an annotated non-dividing
  track or bridge two annotated lineages. Both are the same physical error: a **stolen
  daughter**. Target-free signature: a true division increases the local cell count by +1
  with no compensating loss; a stolen-daughter fork is accompanied by a **track termination
  in the same space-time neighbourhood** (the robbed track loses its successor). Reject any
  fork whose second branch is matched by a co-located, same-frame track death within the
  daughter-search radius.
- **Novelty technique.** First-principles (conservation bookkeeping) + cross-disciplinary
  transfer from network-flow multi-target tracking, where a division is a flow *source* and
  a spurious fork adjacent to a flow *sink* is the canonical signature of mis-assignment.
- **Crowding.** ~150–300 / 1000. This is the most "findable" of the candidates — but the
  public tier demonstrably has not applied it (its heuristic tests only three absolute
  distances and never looks at what the adopted orphan was taken *from*).
- **Grounding.** Zhang, Li & Nevatia, *Global data association for multi-object tracking
  using network flows*, CVPR 2008; Berclaz, Fleuret, Türetken & Fua, *Multiple object
  tracking using k-shortest paths optimization*, IEEE TPAMI 33(9), 2011 — both encode
  appearance/disappearance as explicit source/sink flow with conservation at every node.
- **CHEAP LOCAL FALSIFICATION.** On the banked cv57 output, label each of the 368 forks with
  the conservation flag; recompute the frozen scorer with flagged forks removed.
- **PREREGISTERED KILL.**
  - **KILL** if the filter removes the single existing cv57 TP (any TP loss is
    disqualifying, since 1 TP ≈ 24 FP).
  - **KILL** if `fp(cv57)` falls by `< 3` of 11 (i.e. `J < 1/21 = 0.0476`) **and**
    `fp(shard-0)` falls by `< 12` of 52.
  - **GO** requires `Δ combined > 0` on both populations with `Δtp = 0`.

### 4.3 `DIV-TDILATE` — exploit the exact ±1-frame acceptance window

- **Mechanism.** §1.4 derives that the metric accepts a fork at exactly `t−1`, `t`, `t+1`
  and hard-stops at ±2. The organizers say so explicitly because "the exact timepoint at
  which a cell visibly splits is somewhat subjective" (`metrics.md:52-54`). If our timing
  error distribution has mass beyond ±1, we lose TPs for a *timing* reason, not a detection
  or ranking reason. Fix: for each accepted division candidate, emit the fork at the best of
  `{t−1,t,t+1}`, or — where spare detections exist — at all three. Cost: 2 extra
  `considered` FPs; gain if it converts one FN: `(2,13)` vs `(1,11)` → `J = 2/26 = 0.0769`
  vs `0.0417`. **Net positive at a 33% conversion rate**, and it needs no new signal at all.
- **Novelty technique.** Boundary reasoning on the acceptance window + constraint inversion
  (we treat division *timing* as something to get right; the metric says it only has to be
  right to ±1, so trade timing precision for coverage).
- **Crowding.** ~80–150 / 1000. The ±1 window is in the prose; using it as a *dilation
  budget* rather than a *tolerance* is the non-obvious step.
- **Grounding.** Direct derivation from `division_metrics.py:216-217, 260-264` plus the
  organizers' own stated rationale. No external citation needed — it is a property of the
  scorer.
- **CHEAP LOCAL FALSIFICATION.** Instrumented probe P1 (below) measures, for each of the 12
  cv57 FNs, the frame offset between our nearest candidate fork and the GT divider.
- **PREREGISTERED KILL.**
  - **KILL** if `< 2` of the 12 cv57 FNs have a candidate fork at offset exactly ±2 or ±3
    (i.e. there is no timing-limited population to recover).
  - **KILL** if applying the dilation yields `Δ combined ≤ 0` on shard-0.

### 4.4 `DIV-PAIRHEAD-AS-EVENT` — use the solved conditional head as the missing event detector

- **Mechanism.** The campaign has a *random* parent-event head (AUROC `0.521521`) and a
  *working* conditional daughter-pair head (AUROC `0.897081`, AP `0.368520`). The standard
  cascade — detect event, then assign daughters — is broken at stage 1. Constraint
  inversion: **delete stage 1** and use `max over candidate pairs (a,b) of P(pair | parent)`
  as the event score. Under the 4% bar we do not need a good event detector, only one whose
  top-`K` list beats 4%.
- **Novelty technique.** Constraint inversion (a conditional model repurposed as a marginal
  detector).
- **Crowding.** ~100–200 / 1000 in general; ~5/1000 *here*, because no other team has both
  heads measured separately.
- **Grounding.** Campaign-internal measurement of both heads (LIVE_STATE, post-outcome
  split). Statistical caveat stated up front: a conditional model can be perfectly
  informative *given* a division and carry **zero** marginal information about whether one
  occurred — that is precisely the geometry-only story the AUROC-0.52 result tells. If both
  heads share the geometry/lineage/motion feature contract, this candidate is **already
  implied-dead**. That is a cheap, decisive check and must run first.
- **CHEAP LOCAL FALSIFICATION.** (i) Confirm the two heads' feature contracts differ. (ii)
  Compute `max`-pair score per parent on the banked training population; measure AUROC/AP
  against the 42 GT division events; measure precision in the top `2G`.
- **PREREGISTERED KILL.**
  - **KILL immediately** if the daughter head's feature contract is geometry/lineage/motion
    only (subsumed by the falsified AUROC-0.52 result).
  - **KILL** if event AUROC `≤ 0.65`, or if top-`2G` precision `< 4%`.

### 4.5 `DIV-UNION` — replace consensus with union as the combination rule (methodological)

- **Mechanism.** The banked division realizer wave combined arms by a **frozen 3-of-3
  consensus**, which "abstained everywhere". Consensus/intersection maximises precision.
  Under `J = tp/(G+fp)` with `p* ≈ 4%`, the **union** of weak detectors is the score-optimal
  combination whenever each arm's marginal precision exceeds `p*`. This is a binding
  correction to campaign methodology, not a new detector.
- **Crowding.** ~50–120 / 1000. Follows immediately once `p*` is derived; almost nobody
  derives `p*`.
- **Grounding.** Derivation §0.1. Ensemble literature: union/OR-combination is optimal under
  strongly asymmetric FN:FP cost (Elkan 2001, as above).
- **Honest value assessment.** All three realizer arms scored **0 TP** (`0/23/38`,
  `0/48/38`, `0/22/38`); the union of zero-TP sets is still zero TP. So the standalone value
  today is **zero**. Its value is as a **preregistered rule for any future multi-arm
  division wave** — do not re-run a consensus gate.
- **PREREGISTERED KILL.** Not a fundable candidate; record as a standing methodological
  constraint on future division nodes. No spend.

### 4.6 `DIV-APPEARANCE-EVENT` — raw-appearance mitotic-event detection ▲ RESEARCH BET

- **Mechanism.** The one observable class the campaign has **never screened**. The measured
  parent-event head was contractually **geometry/lineage/motion-only, raw appearance
  excluded**; the appearance arm of the realizer wave "failed closed before prediction"
  for cache/interface reasons, not scientific ones. Mitosis has a strong, well-documented
  *appearance* signature — chromatin condensation, cell rounding, and a transient intensity
  rise — which no geometric feature can express. AE-P7's three-sample raw-image crop
  interface is now `PASS_INTERFACE_ONLY`, so the blocking infrastructure exists.
- **Novelty technique.** Literature-edge mining (the mitosis-detection literature is a
  separate, older field from cell *tracking*, and the public tier does not touch it).
- **Crowding.** ~40–100 / 1000 as an idea; near-0 among the public tier, which contains no
  learned division component at all.
- **Grounding (primary sources).**
  - Huh, Ker, Bise, Chen & Kanade, *Automated mitosis detection of stem cell populations in
    phase-contrast microscopy images*, IEEE Trans. Medical Imaging 30(3):586–596, 2011.
  - Liu, Li & Kanade, *Mitosis sequence detection using hidden conditional random fields*,
    IEEE ISBI 2010 / CVPR-era HCRF line — mitosis as a *temporal sequence* of appearance
    states, not an instantaneous test.
  - Amat, Lemon, Mossing, McDole, Wan, Branson, Myers & Keller, *Fast, accurate
    reconstruction of cell lineages from large-scale fluorescence microscopy data*, Nature
    Methods 11:951–958, 2014 (TGMM) — the canonical light-sheet developmental-imaging
    lineage pipeline, exactly this modality; division is detected by Gaussian-mixture
    splitting on the image, then filtered temporally.
  - Bao, Murray, Boyle, Ooi, Sandel & Waterston, PNAS 103(8):2707–2712, 2006 (StarryNite /
    AceTree) — the sister-symmetry and division-timing priors for developmental lineages.
- **Why it is a research bet, not a plan.** It cannot be falsified cheaply on banked
  artifacts alone; it needs the raw-image crop cache, and a positive result would then need
  GPU (human-gated). It is labelled accordingly.
- **Cheapest possible first falsification (still CPU-only, if the crop interface holds).**
  On the 42 banked GT division events, compute *hand-specified, untrained* appearance
  statistics in the parent crop (mean/max intensity, intensity kurtosis, sphericity /
  eigenvalue ratio of the second moment, and their 1-frame temporal derivatives) and measure
  event AUROC against matched non-dividing parents.
- **PREREGISTERED KILL.**
  - **KILL** if untrained appearance statistics give event AUROC `≤ 0.70` on the 42 events
    (that is the "there is no appearance signal to learn" verdict, and it closes the lane
    for good rather than escalating to GPU).
  - **KILL** if the crop interface cannot deliver physical intensity fields for `≥ 90%` of
    the 42 events (repeat of the previous infrastructure failure — record as
    NOT-EVALUATED, not as a scientific FAIL).

### 4.7 Goal inversion — "how would I guarantee failure here, and am I doing any of it?"

Run as a generator (novelty-techniques §7). Five ways to guarantee failure on the division
term; **we are currently doing four of them.**

1. *Tune fork emission for high precision.* — **Doing it.** The public gate is engineered
   for ~99% precision against a 4% bar.
2. *Combine weak division arms by consensus.* — **Did it.** The 3-of-3 gate abstained
   everywhere (§4.5).
3. *Impose a near-zero-FP division constraint.* — **Did it.** Provably suboptimal (§0.1).
4. *Optimise a proxy instead of the frozen scorer.* — the campaign's hardest-won lesson;
   `DIV-EVOP` is defined to sweep the frozen scorer directly.
5. **Measure fork quality on the whole fork population instead of the evaluable
   subpopulation.** — **Doing it, and it is actively steering us wrong.** Our own headline
   diagnostic is "368 forks → 1 TP", which reads as **0.27% precision** and licenses the
   conclusion "our division emission is catastrophically imprecise, suppress it". The
   metric-relevant precision is `tp/(tp+fp) = 1/12 = 8.3%` — **above** the 4.00% bar. The
   correct marginal action is to **add** forks, and the standard framing of our own evidence
   file points the opposite way. This is a self-inflicted failure mode and the single
   cheapest thing to fix in this report.

*(Precision caveat on the 368: `predicted_dividing_nodes` is computed as
`sum(value == 2 for value in outdegree.values())` — out-degree **exactly** 2. Nodes with
out-degree ≥ 3 are forks for the division metric but are excluded from that count. Shard-0/1
strict validation reported zero out-degree violations, so 368 ≈ the true fork count and the
96.7%-ignored figure stands; on shard 2, where two out-degree-3 parents existed, the count
would be a slight undercount.)*

### 4.8 Rejected before promotion (recorded so they are not re-derived)

- **`DIV-ATARGET` (predict the sparse-GT annotation footprint and target it).** Sound
  reading of §2, but **dominated**: forks *outside* the footprint are free on both terms, so
  there is nothing to save by avoiding them, and there is no compute constraint on a
  postlink heuristic. Targeting buys nothing that "be aggressive everywhere, at the 4% bar"
  does not already buy. Rejected on dominance, not on rigor.
- **`DIV-LAPMARGIN` (inverted Lowe ratio / LAP second-best-assignment margin as a division
  signal).** Genuinely attractive as a cross-disciplinary transfer (Lowe's SIFT ratio test
  inverted; Murty 1968 / Cox & Miller, IEEE TAES 31(1):486–489, 1995, on ranked assignments
  for MTT ambiguity), and it is scale-free where the public 4.66/8.5/7.65 µm constants are
  not. **But** it is a derived geometry/motion statistic, and the banked parent-event head
  under a geometry/lineage/motion-only contract scored AUROC `0.521521` — random. Absent
  positive evidence that the assignment margin was outside that feature class, proposing it
  would be a re-screen of a falsified observable. Held as **conditional**: only revivable if
  a check shows the margin statistic was not in the screened feature set.
- **`DIV-PERSIST` (TGMM-style daughter persistence + volume symmetry).** Already run and
  falsified: persistence arm `0/49/11`, J = 0. Do not re-propose.

---

## 5. Task 5 — ranking, and the single alpha

Scored on pertinence × feasibility × defensibility (each 1–5).

| # | candidate | pertinence | feasibility | defensibility | product | status |
|---|---|---|---|---|---|---|
| 1 | **`DIV-EVOP`** (+`DIV-K3`) | 5 | 5 | 4 | **100** | **ALPHA — GO** |
| 2 | `DIV-TDILATE` | 3 | 5 | 4 | 60 | GO (fold into `DIV-EVOP`'s pool) |
| 3 | `DIV-CONSERVE` | 3 | 5 | 3 | 45 | GO (co-deploy, FP lane) |
| 4 | `DIV-APPEARANCE-EVENT` | 5 | 2 | 5 | 50 | **RESEARCH BET** |
| 5 | `DIV-PAIRHEAD-AS-EVENT` | 4 | 4 | 2 | 32 | cheap probe first; likely implied-dead |
| 6 | `DIV-UNION` | 1 | 5 | 3 | 15 | standing methodological rule, no spend |

### The named alpha: `DIV-EVOP`

**What it is.** The division sub-problem is not a detection problem, it is an *operating
point* problem. The exact objective geometry is `J = tp/(G + fp)` with acceptance threshold
`p* = J/(1+J)` — **4.00% on cv57, 2.53% on shard-0** — and an FP channel that prices forks
only inside the sparse-GT footprint (and, for branches 3+, not at all). `DIV-EVOP` replaces
every precision-first gate in our division path with a direct sweep of the frozen scorer at
that derived operating point, using V32's own edge probability as the ranking observable,
the ±1-frame dilation as the candidate pool, and out-degree-3 emission to make daughter
selection free.

**Why it is defensible — the nameable reason.** Three independent facts, each of which
requires reading the *scorer source* rather than the metric prose, and each of which points
the opposite way from what every competitor and every prior campaign node assumed:

1. `tp + fn ≡ G` is a **constant**, so `J = tp/(G+fp)` and a division TP is worth **24 FPs**
   at our operating point. Nobody who reads only `metrics.md` derives this.
2. The FP channel is **gated on proximity to sparse GT** — 96.7% of our forks were never
   evaluated. `metrics.md` line 118 states the ignore rule as a *leniency*; the code makes it
   a *targeting instruction*. That inversion is the moat.
3. The out-degree cap exists **only in the edge path and only after FP counting**, so the
   3rd branch of a fork is free. This is invisible unless you trace the polars pipeline
   in `metrics.py:140-153` against `division_metrics.py:354-357`.

The public tier runs `add_safe_divisions_postlink` at 4.66/8.5/7.65 µm with caps at 0.76%
of frame sources and 0.375% of edges — a rule engineered for ~99% precision. Our own
campaign at one point carried a "near-zero-FP division operation" constraint and combined
weak division arms by **3-of-3 consensus**. Both are precision-maximising designs pointed at
a metric that pays at a 4% precision bar. **That mismatch, not a better model, is the alpha.**

**Cost to falsify: zero GPU, zero submission, CPU-only, on already-banked shard-0 outputs.**

### Rigor-gate summary (required for GO)

| candidate | evidentiality gate | local-falsifiability gate | verdict |
|---|---|---|---|
| `DIV-EVOP` | PASS — exact derivation verified on 2 populations + Lipton et al. 2014, Elkan 2001 | PASS — preregistered `K`-sweep with numeric kill lines | **plan** |
| `DIV-CONSERVE` | PASS — Zhang et al. 2008, Berclaz et al. 2011 | PASS — flag-and-rescore, preregistered | **plan** |
| `DIV-TDILATE` | PASS — first-principles from the scorer | PASS — offset histogram (P1), preregistered | **plan** |
| `DIV-PAIRHEAD-AS-EVENT` | PARTIAL — internal measurement only | PASS | probe |
| `DIV-APPEARANCE-EVENT` | PASS — 4 primary sources | PARTIAL — needs crop cache | **research bet** |
| `DIV-UNION` | PASS — derivation | N/A (no standalone effect) | rule |

**Preregistered GO threshold from the node contract is met:** ≥1 candidate passes both
gates with a named defensibility reason.

---

## 6. Probes to run first (all CPU-only, all on banked artifacts, all preregistered)

Order matters — P1 and P3 decide how `DIV-EVOP` is instantiated.

- **P0 — pin the edge denominator.** Record `N_e = edge_tp+edge_fp+edge_fn` and `J_e` for
  cv57 and shard-0. *Purpose:* confirm §0.2's "edge term does not constrain division
  aggression". *Kill:* if the affordable edge-FP budget per division TP is `< 3`, the whole
  aggression argument weakens and `DIV-EVOP` must be re-derived jointly over both terms.
- **P1 — FN anatomy (the highest-value probe).** For each of the 12 cv57 FNs, record which
  gate fired: (a) `_matched_division_nodes is None` (detection gap — unfixable, front-end
  frozen); (b) `local_forks == ∅` (no fork at the anchor — the recall lane); (c)
  `_is_strongly_connected_division` returned `False` (topology lane); (d) in `invalid_forks`;
  (e) bipartite leftover. Also record, for (b), the frame offset from the GT divider to our
  nearest candidate fork. *Purpose:* splits the 12 FNs into recall / topology / timing /
  unreachable, and directly feeds `DIV-TDILATE`'s kill line. §1.2 predicts (b) dominates.
- **P2 — merge audit.** Count nodes with in-degree ≥ 2 that are children or grandchildren of
  a fork. This is the **only GT-independent FP channel** (`:424, :430`). *Expected 0* for a
  forest submission; any non-zero count is free FP removal at zero risk.
- **P3 — FP anatomy.** Split the 11 cv57 / 52 shard-0 FPs into `considered`,
  `evaluable_forks`, `cross_component_forks`, `malformed_forks`. §1.3 already proves ≥7/11
  are non-`considered`; this quantifies it and sizes `DIV-CONSERVE`.

## 7. Dominant risk — statistical power (stated plainly)

`G = 13` on cv57, `25` on shard-0, `42` on the full 57-video population. **A cv57 division
Jaccard move from 0.0417 to 0.0833 is one single true positive.** Every tuned parameter on
this term is at severe overfitting risk, and the division term is worth at most 0.1 of the
combined score. This is the strongest argument *for* `DIV-EVOP` specifically and *against*
any learned division ranker: `p* = J/(1+J)` is **derived, not fitted** — zero free
parameters — and the only fitted quantity is a single integer `K`, swept against the frozen
scorer on two disjoint populations with a preregistered transfer check. Any candidate that
introduces more than one free parameter on this term should be rejected on power grounds
alone.

## 8. Proposed LEDGER row (NOT written — root coordinator is the sole writer of `LEDGER.md`)

Per `CLAUDE.md`, this node does not write `reports/g02/screening/LEDGER.md`. Row text for the
root coordinator, in the existing column order:

- **id** — `G02-N1-DIVISION-METRIC-OPERATING-POINT-20260806`
- **mechanism** — frozen-scorer source derivation of the division objective geometry, and the
  `DIV-EVOP` EV-optimal fork-emission policy on the public-V32 substrate
- **tier** — RESEARCH / SCORER-SOURCE DERIVATION / GO (no spend)
- **predicted delta** — cv57 division J `0.041667 → ≥0.0625` and shard-0 `0.025974 → ≥0.0390`
  at the swept optimum; combined-score headroom `+0.0042 → up to +0.092` (banked oracle)
- **actual delta** — none measured; derivation + preregistration only
- **local identity** — `division_metrics.py` `0635c386…60f4dc9`; `metrics.py`
  `cfdd596e…67ca7444`; evidence `FULL_PUBLIC_V32_S00_NATIVE_DIVISION_SCORE.json`
- **observation class** — read-only analysis of the frozen scorer and already-banked shard-0
  outputs; no new prediction, no refit, no threshold search executed
- **Kaggle submissions** — 0 · **Kaggle GPU-hours** — 0 · **RunPod** — 0 / $0
- **decision** — `GO / PREREGISTER DIV-EVOP + DIV-CONSERVE + DIV-TDILATE / PROBES P0–P3 CPU-ONLY / NO SPEND`
- **reason** — `division_jaccard = tp/(G+fp)` exactly (`tp+fn ≡ G`), so the fork-acceptance
  threshold is the derived `p* = J/(1+J)` = 4.00% (cv57) / 2.53% (shard-0) and one TP is
  worth 24 FPs. 96.7% of emitted forks are never evaluated because every FP channel except
  `malformed_forks` is gated on proximity to the sparse GT. ≥7 of 11 cv57 FPs are in the two
  videos with zero GT divisions. The public gate, the former "near-zero-FP division
  operation" constraint, and 3-of-3 consensus are all precision-maximising designs aimed at
  a recall-first metric. Zero free parameters beyond a single swept integer `K`.
- **commit** — `dag_prompts/2026-08-06-alpha-hunt/runs/N1_DIVISION_ALPHA.md`

## 9. Cascade

Node contract: **GO (≥1 candidate passes both gates) → proceed to
`N4_KERNEL_PACKAGE_SUBMIT.md` (HUMAN-GATE, halts).** All work in §6 and §4.1–4.3 is
CPU-only on banked artifacts and requires no GPU, no Kaggle, and no submission; the human
gate is reached only when a packaged kernel is proposed.
