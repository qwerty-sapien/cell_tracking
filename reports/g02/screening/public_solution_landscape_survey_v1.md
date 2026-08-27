# Public Solution Landscape Survey (PROMPT B)

**Date:** 2026-08-06 · **Type:** research only, nothing executed, nothing implemented
**Evidence:** 18 public kernels pulled via `kaggle kernels pull`; official scorer at
`external/kaggle-cell-tracking-competition` (checkout `075fc5f`); public leaderboard
snapshot `2026-08-05T19:22:39` (2,019 teams).

---

## 0. Headline

**The 0.95–0.97 tier does not exist.** It was a single metric exploit, the host patched it on
**2026-07-17**, and Kaggle rescored. On the live leaderboard **zero of 2,019 teams score ≥0.95**;
the maximum is **0.948**. Every author of a "0.95–0.97" notebook now sits **below** our own
CV57 0.914358.

The notebook badge scores in the prompt are stale pre-patch artifacts. The real ladder is
0.914 (us, ~rank 96) → 0.921 (best clean public author) → 0.948 (live #1), and that remaining
gap is legitimate modelling.

---

## 1. Notebook survey

All 18 kernels are **inference-only** — zero contain a training loop (`grep -c
'\.backward()|optimizer\.step()'` = 0 across every file). All but one load public checkpoints
from `pilkwang/biohub-tracking-support-pack-50ep-v1` (`TemporalUNet3D` +
`SimpleNodeTransformer`).

### Tier A — hub/fork augmented (the "0.95–0.97" tier)

| Notebook | Badge | Approach | Model family | Weights | Division handling |
|---|---|---|---|---|---|
| `xiaoleilian/biohub-ct-mix-divaug` | 0.969 | 2× UNet3D heatmap **average** (own preproc each), `peak_local_max` + physical NMS, 2-pass µm-gated Hungarian, gap-close, short-track filter, line-fit | custom `UNet3D(base=24)` | own private dataset `biohub-unet3d-weights` | **none in the model** — Hungarian is 1-to-1, so out-degree ≤ 1 and the base graph has *zero* forks. All divisions are fabricated by the final cell. |
| `kaiwalyaatulraut/biohub-solution` | 0.966 | minimal baseline + TTA + hub/fork cell | TemporalUNet3D + SimpleNodeTransformer | public support pack | none real; hack `MAX_COMPONENTS=3000, FORKS=20` |
| `kaiwalyaatulraut/biohub-cell-tracking-solution` | 0.966 | **byte-identical duplicate** of the above (md5 `63f1be70…`) | same | same | same |
| `kaiwalyaatulraut/biohub-competition-solution` | 0.965 | minimal baseline + `yoikoarmor/biohub-last-call-postprocess-v1` | same | public support pack | none real; hack `1400, 5` |
| `boristown/dark-agi-…-solution` | 0.952 | minimal baseline, variant sweep (`BIOHUB_VARIANT` V1…V9, comments record `V4 forks=4 = 0.950`) | same | public support pack | none real; hack, forks tuned as a **hyperparameter** |
| `nikitagajbhiye30/biohub-00` | 0.896 | fork of `kirneo/metric-hack-last-call-update` | same | public support pack | none real; hack `3400, 32` |

Tier A signature: `MAX_COMPONENTS`/`FORKS` present, `add_safe_divisions_postlink` **absent**,
ILP **absent**.

### Tier B — clean lineage (the "0.91" tier)

All descend from `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` and are
parameterised by `BIOHUB_*` env vars. All use the public support-pack checkpoints, ILP linking
(`ILP_DIVISION_WEIGHT=1.0`), and the geometric `add_safe_divisions_postlink`.

| Notebook | Badge | What it adds over the lineage | Division handling |
|---|---|---|---|
| `yusuketogashi/no-hack-…-another-approch-3rd` | 0.914 | bidirectional harmonic edge fusion (0.20); 4-view JS-reliability log-pool edge TTA; 22-feature learned local association ranker (0.85); forward-acceleration lookahead | ILP division weight 1.0 + safe-div postlink (constants below) |
| `nikitagajbhiye30/biohub-11` | 0.913 | DeepCenter center-prior veto (`REQUIRE=1`, gap veto @0.25, epoch 500); secondary detection weight 0.475 | same |
| `praxel/biohub-0-902-motion-division-calibration` | 0.901 | **origin of the safe-division constants** (§4) | same |
| `yaroslavkholmirzayev/…-v4-unet-ilp-reproduction` | 0.897 | edge-policy veto reproduction, `DET_THRESHOLD=0.99` | same |
| `pilkwang/…-learned-graph-w-gap-recovery` | 0.894 | the lineage root, `DET_THRESHOLD=0.985` | same |
| `mige551/biohub-clean-122-det-0-9690` | 0.969¹ | short-track rescue params; header states "No artificial hubs, fake forks, negative-time nodes, out-of-volume nodes, or cross-clip graph edges are created." | same |

¹ The `0.9690` in mige551's title is **`BIOHUB_DET_THRESHOLD=0.9690`, a detection threshold —
not a score.** The prompt's reading of mige551 as a 0.969-tier entry is a title collision; the
notebook's own score axis says "Clean 0.910+". mige551's live LB is 0.915.

---

## 2. Central question: what separates the tiers

**(b) Exploitation of the scorer.** Not ensembling, not modelling. The separation is one
appended cell, and its author labels it himself.

### 2.1 The exploit code

`xiaoleilian/biohub-ct-mix-divaug` cell 4, verbatim:

```python
# ============================================================================
# OPTIONAL FINAL CELL — division-term augmentation (metric hack)
# Append as the LAST cell of the M001 kernel, AFTER submission.csv is written.
# Validated offline on host_metric replica: VAL 0.8388 -> 0.9203 (+0.0815).
# This is a METRIC EXPLOIT (fake out-of-volume / negative-time hub + forks that
# game division_jaccard). Public score; does NOT commit to final-2. Patchable.
# ============================================================================
MAX_COMPONENTS, FORKS = 1400, 5
...
roots = [n for n in nid if n not in inc][:MAX_COMPONENTS]
hub = nxt; nxt += 1
new = [_row(ds, "node", hub, -1000, -10000, -10000, -10000)]
new += [_row(ds, "edge", s=hub, tg=r) for r in roots]
prev = hub
for i in range(FORKS):
    d, c, co = range(nxt, nxt + 3); nxt += 3; tt = -999 + 2 * i
    new += [_row(ds, "node", d, tt, -10000, -10000, -10000),
            _row(ds, "node", c, tt + 1, -10000, -10000, -10000),
            _row(ds, "node", co, tt + 1, -10001, -10000, -10000),
            _row(ds, "edge", s=prev, tg=d), _row(ds, "edge", s=d, tg=c),
            _row(ds, "edge", s=d, tg=co)]
    prev = co
```

It fabricates one **hub** node at `t=-1000`, `(z,y,x)=(-10000,-10000,-10000)` — a coordinate
outside the imaged volume at a timepoint that does not exist — wires it to every real track
root, then chains a few synthetic parent→(child, continuation) **forks** at negative
timepoints. `boristown` tunes `FORKS` as a hyperparameter (`# V4 forks=4 = 0.950`);
`kirneo`/`nikitagajbhiye30` push it to `MAX_COMPONENTS=3400, FORKS=32` with the note
`# 32 Forks 2600 components -> 32 Forks 3400 components`.

### 2.2 Why it worked — the pre-patch division metric

The pre-patch `metrics.md` awarded a division TP on four conditions, the last two being:

> - **Single connected component.** All the matched predicted nodes above lie in one
>   connected component of the predicted graph.
> - **Contains a predicted fork.** The component includes at least one predicted dividing
>   node (a predicted cell with two outgoing edges).

Both are **global graph properties**. Wiring every root to one hub collapses the entire
prediction into a single weakly connected component; adding any fork anywhere puts a fork in
that component. Both conditions then hold for *every* GT division simultaneously, so the
division Jaccard saturates without predicting a single real division. Since
`score = adj_edge_jaccard + 0.1 · division_jaccard`, that is worth up to +0.1 — matching the
author's measured **+0.0815**.

This is exactly the sparse-GT/degenerate-graph class the prompt asked about, and it is why
`xiaoleilian`'s Hungarian tracker — which structurally *cannot* emit a fork — could post a
division score at all.

### 2.3 The patch

`external/kaggle-cell-tracking-competition` commit **`aa65e90`, 2026-07-17 16:47 -0700,
Thibaut Goldsborough** (the host), titled verbatim:

> `updating metric to patch weakly connected component exploit`

It replaces global reachability with strict local directed topology —
`metrics.md` now reads "**Merely sharing a weakly connected component is not sufficient**" —
and adds three structural defenses in `metrics.py`:

```python
# Cap out-degree: a dividing cell has at most two children, so a predicted node
# with more than two outgoing edges is biologically invalid.
edge_attrs = edge_attrs.filter(pl.col("_out_rank") <= 2)
```
kills the 1,400–3,400-way hub, and

```python
.filter(pl.col("_target_t") - pl.col("_source_t") == 1)
```
drops every hub→root edge (source at `t=-1000`, target at `t=0`), plus merged-edge dedup.

Post-patch the fabricated nodes are pure cost: they inflate `num_pred_nodes` in
`adj_edge_jaccard = max(0, J · (1 − 0.1·(N_pred − N_true)/N_true))` while contributing nothing.

### 2.4 Empirical confirmation

Live public leaderboard, 2,019 teams, snapshot 2026-08-05:

| threshold | teams |
|---|---|
| ≥ 0.96 | **0** |
| ≥ 0.95 | **0** |
| ≥ 0.94 | 4 (max 0.948) |
| ≥ 0.914358 (our CV57) | 95 |

And the authors themselves, current live scores:

| Author | Badge on their hacked notebook | **Live LB now** |
|---|---|---|
| xiaoleilian | 0.969 | **0.889** (rank 961) |
| kaiwalyaatulraut | 0.966 / 0.965 | **0.901** (rank 675) |
| boristown | 0.952 | **0.911** (rank 434) |
| kirneo | metric-hack-last-call | **0.908** |
| outwrest | metric hack TTA | **0.881** |
| — clean authors — | | |
| pilkwang | 0.894 | **0.921** (rank 25) |
| yusuketogashi | 0.914 | **0.916** (rank 54) |
| mige551 | "0.9690" (= det threshold) | **0.915** (rank 60) |

Every hack author is now **below** our 0.914; every clean author is at or above it. `yusuketogashi`
states it plainly in a 2026-07-23 notebook: *"Kaggle has addressed the historical metric exploit,
so 0.908 is treated as the legitimate clean baseline."*

---

## 3. Rules adjudication

Rules read live from the competition rules page (not inferred). There is **no clause naming
"metric exploitation"**. The governing provisions are:

- **§3.8.d** — "Competition Sponsor reserves the right to disqualify any Participant … if the
  Competition Sponsor reasonably believes that the Participant has attempted to **undermine the
  legitimate operation of the Competition by cheating, deception, or other unfair playing
  practices or abuses**."
- **§3.4.d** — "Submissions are void if they are in whole or part illegible, incomplete,
  damaged, **altered**, counterfeit, obtained through fraud, or late."
- **§2.8 / §3.9.b** — winners must deliver reproducible code and a methodology description;
  prizes are contingent on a compliance review that may disqualify.

**Verdict: not permitted, and moot.** The hub/fork augmentation emits nodes that correspond to
no cell, at timepoints that do not exist, for the sole purpose of manipulating the scorer. That
is squarely within §3.8.d's discretionary scope, the host demonstrated its position by patching
it, and it is a $60k Research Code Competition where the winner must defend the method on a
recorded call. It also no longer works.

**Not implemented. Not recommended. Surfaced for owner decision only.** Our existing campaign
principle rejecting "metric gaming through node pruning" is consistent with this finding and
should stand.

One related item deserves an explicit flag because it is *not* fabrication and *is* live: the
adjustment factor `(1 − 0.1·(N_pred − N_true)/N_true)` **exceeds 1 when `N_pred < N_true`**, so
under-predicting nodes earns a genuine multiplier bonus. The whole clean tier sits at very high
detection thresholds (0.96875–0.99), which is at least partly this effect. This is threshold
tuning against a published metric, not fabrication — but it is the reason our own
`DET_THRESHOLD=0.96875` is where it is, and it is a real knob rather than a modelling gain.

---

## 4. `praxel/biohub-0-902-motion-division-calibration` (0.901) — deep read

**Finding: we already run this notebook's division calibration verbatim. There is nothing to
port.**

Its entire division mechanism is `add_safe_divisions_postlink` — a purely geometric
orphan-adoption pass with hard caps, no learning:

```python
source_ids   = [n for n in ids_by_t[t]     if len(out_by_source.get(n, [])) == 1]   # exactly one child
candidate_ids= [n for n in child_frame_ids if n not in incoming and n not in used_targets]  # orphans at t+1
...
child_dist  = edge_distance_um(source, existing_child)
if child_dist  > SAFE_DIV_EXISTING_CHILD_MAX_UM: continue   # 7.65 µm
parent_dist = edge_distance_um(source, candidate)
if parent_dist > SAFE_DIV_MAX_UM:                continue   # 4.66 µm
sister_dist = edge_distance_um(existing_child, candidate)
if sister_dist > SAFE_DIV_SISTER_MAX_UM:         continue   # 8.5 µm
score = parent_dist + 0.15 * sister_dist                     # rank, then cap
```
capped at `frame_cap = 0.0076 × |sources|` per frame and `global_cap = 0.00375 × |edges|`.

Its five constants are **bit-identical to our V32 config**:

| knob | praxel | our V32 | lineage default |
|---|---|---|---|
| `SAFE_DIV_MAX_UM` | 4.66 | 4.66 | 4.7 |
| `SAFE_DIV_SISTER_MAX_UM` | 8.5 | 8.5 | 7.2 |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | 7.65 | 7.65 | 7.8 |
| `SAFE_DIV_FRAME_FRAC_CAP` | 0.0076 | 0.0076 | 0.008 |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | 0.00375 | 0.00375 | 0.004 |

We inherited them through the pilkwang lineage. The 0.901→0.914 difference between praxel and us
is elsewhere (dual-seed blend, gap-density adaptivity, ILP appearance/disappearance weights).

**The important negative result:** this heuristic is the *entirety* of public division handling.
**No public notebook contains a learned division classifier.** And on our own numbers it barely
works — `FULL_PUBLIC_V32_S00_NATIVE_DIVISION_SCORE.json`, CV57 population:

```
division_tp = 1, division_fp = 11, division_fn = 12  →  division_jaccard = 0.0417
predicted_dividing_nodes = 368   vs   gt_division_count = 13
```

We emit 368 forks to catch one. The division term contributes `0.1 × 0.0417 = 0.0042` of our
0.914358; a perfect division term is worth 0.1. **≈0.096 of headroom sits in the division term
alone** — consistent with LIVE_STATE's finding that division is the dominant score ceiling, and
it confirms the ceiling is *not* addressable by tuning praxel's constants, because we are
already at them.

---

## 5. Legitimate **and** absent from our pipeline — the real backlog

Verified by grepping our V32 source (`external/public_temporalunet_baseline/pilkwang_two_seed_v32`).
None of these touches the scorer; all are ordinary modelling.

> **Corrected 2026-08-06 by N2 anchor verification**
> (`dag_prompts/2026-08-06-alpha-hunt/runs/N2_CLEAN_PORT_PLAN.md`). Three items below were
> mis-scoped in the original write-up:
> 1. **Items 1 and 2 are ONE atomic patch, not two.** Bidirectional fusion lives inside the
>    closure the 4-view TTA loop calls, and the body hard-raises unless
>    `EDGE_TTA_MODE=js_reliability_log_pool` **and** `VIEWS=4`
>    (`…-3rd.py:1654`). They cannot be ported or ranked separately.
> 2. **Item 4 is NOT free-standing.** The *functions* are ~30 lines of pure geometry, but the
>    author's call site (`:2731`) is nested inside
>    `if LOCAL_ASSOCIATION_RANKER_MODE == 'full_motion_assignment':` (`:2718`) and its bonus
>    rides on a ranker-blended `evidence` term. A standalone port is feasible (~10 lines of
>    glue) but is an **unmeasured variant**, not the author's measured mechanism. The
>    "highest value per unit of effort" call below was premature.
> 3. **Items 5 and 6 are not upside.** The 0.916 author sets
>    `BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE="0"` and `BIOHUB_USE_DEEPCENTER_VETO='0'` — **exactly
>    as we do** (`:97-98`). The "config-disabled relative to a stronger solution" framing came
>    from comparing against `biohub-11` and `mige551`, both of which score *below*
>    yusuketogashi. Adopting their flags means overriding the best clean author's own
>    ablation. N2 drops item 5 and keeps item 6 only because it is free to test.
>
> **Blocker B1 (compile-clean, runtime-fatal).** The `_bi_old` anchor *does* match our
> V32-patched source and *does* pass `compile()`, but `_bi_new` references `_edge_tta_imgs`,
> which our chain never binds: their dual-seed replacement ends `_edge_tta_imgs = imgs`, ours
> ends `del imgs`. Porting items 1+2 means **editing code we already ship**, not adding a
> patch, and `compile()` provably does not catch this class — it would fail as a `NameError`
> after GPU allocation and model load.

**Absent from our code entirely** (all from `yusuketogashi/no-hack-…-3rd`, author live 0.916):

1. **Bidirectional harmonic edge fusion** (`BIDIRECTIONAL_EDGE_WEIGHT=0.20`,
   `harmonic_probability`) — run `predict_edges` forward *and* reverse, rescale the reverse
   logits to the forward distribution, combine as a weighted harmonic mean of softmax
   probabilities. Symmetry consistency on the association head. Cheapest of the four; no new
   weights.
2. **4-view JS-reliability log-pool edge TTA** (`EDGE_TTA_MODE=js_reliability_log_pool`,
   `EDGE_TTA_VIEWS=4`) — pool edge log-probabilities across 4 augmented views weighted by
   Jensen–Shannon reliability. Inference-time only; costs ~4× edge-head compute.
3. **22-feature learned local association ranker** (`pilkwang/biohub-local-association-ranker-unet300-v1`,
   weight 0.85 with 0.15 primary retention) — a learned re-ranker over candidate associations.
   Needs an extra public checkpoint; closest thing to a real learned head in the public tier.
4. **Forward-acceleration lookahead** (`MAX_ACCEL_UM=4.0`, `MAX_BONUS=0.20`) — reward a
   candidate edge when it admits a low-acceleration continuation into `t+2`:
   `support = max(0, 1 − residual/max_accel)`. ~30 lines, pure geometry, no weights, no
   retraining. **Highest value per unit of effort.**

**Present in our code but disabled by config** (flip-a-flag experiments, not ports):

5. **DeepCenter center-prior veto** — `USE_DEEPCENTER_VETO=0` for us; `biohub-11` runs it at
   `REQUIRE=1`, `GAP_VETO=1`, threshold 0.25 with the `unet3d-center-prior-v1` checkpoint.
6. **Adaptive short-track rescue** — `ADAPTIVE_SHORT_TRACK_RESCUE=0` for us; mige551 (live
   0.915) runs it with `MIN_LEN=4`, `MIN_MEAN_EDGE_PROB=0.82`, `MAX_MEAN_EDGE_DIST_UM=3.25`,
   `MAX_NODES_FRAC=0.018`. Directly targets our known fragmentation-FN term.
7. Minor: `SECONDARY_DETECTION_WEIGHT` 0.5 (ours) vs 0.475 (biohub-11, no-hack-3rd).

**Not in the public tier at all — and where the actual 0.096 lives:** a *learned* division
classifier. The public ceiling on divisions is a geometric heuristic we already run at its
tuned constants. Nobody public has solved this. That is both the largest headroom and the least
crowded lane, and it is exactly what the division-temporal-Colab lane in LIVE_STATE is aimed at.

---

## 6. What this changes

- The "we are 0.05 behind the public frontier" framing is **wrong**. We are ~0.034 behind the
  live #1 (0.948) and inside the top 5% of 2,019 teams at rank ~96.
- No 0.95+ target should be set against those notebook badges.
- The clean public frontier that is genuinely ahead of us is **pilkwang at 0.921** (+0.007),
  reachable via items 1–7 above, none of which requires training.
- The prompt's suspicion was correct on the mechanism and understated on the scope: it is not
  *partly* metric exploitation, it is **entirely** metric exploitation, and it is already dead.
