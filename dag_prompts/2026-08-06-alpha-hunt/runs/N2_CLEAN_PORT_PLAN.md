# N2 — Clean public-mechanism port plan (beta lane)

**Node:** N2 of `dag_prompts/2026-08-06-alpha-hunt/DAG.md`
**Date:** 2026-08-06
**Scope:** RESEARCH + FEASIBILITY ONLY. No pipeline modification, no launch, no submission.
**Verdict:** **GO (conditional)** — see §7. All six mechanisms are locatable and all string-patch
anchors were verified against the real support-pack source. Two structural blockers were found
that the node contract's framing did not anticipate (§2.1 blocker B1, §2.4 blocker B2).

---

## 0. Provenance — what was actually read

| Artifact | Path | Notes |
|---|---|---|
| Our V32 notebook (py export) | `…/scratchpad/src/OURS_v32.py` | 132 KB; matches `external/public_temporalunet_baseline/pilkwang_two_seed_v32/biohub-cell-tracking-two-seeds-logit-blend.ipynb` |
| Best clean public | `…/scratchpad/src/yusuketogashi_no-hack-biohub-cell-another-approch-3rd.py` | 195 KB; author live LB 0.916; `EXPERIMENT_TAG = "biohub_162_forward_acceleration_lookahead_target0916_nohack"` |
| DeepCenter reference | `…/scratchpad/src/nikitagajbhiye30_biohub-11.py` | 138 KB |
| Short-track-rescue reference | `…/scratchpad/src/mige551_biohub-clean-122-det-0-9690.py` | 111 KB |
| **Real support-pack source** | `/Users/agent/Desktop/cell_tracking/external/Tracking_Support/repo/scripts/predict_unet_transformer.py` | **677 lines, md5 `a2bbd2b33666aab525a9b4b510c48bd9`** — this is the authoritative patch target |
| Anchor verification harness | `…/scratchpad/verify_anchors.py` | AST-extracts every patch literal, applies in source order, reports match counts + compiles |

### Which support-pack file is authoritative

Three copies of `predict_unet_transformer.py` exist locally. Only one is the patch target:

| Path | Lines | md5 | Verdict |
|---|---|---|---|
| `external/Tracking_Support/repo/scripts/…` | 677 | `a2bbd2b3…` | **AUTHORITATIVE.** Sibling of `ARTIFACT_MANIFEST.json`; this dir *is* the `biohub-tracking-support-pack-*` artifact that gets unpacked to `WORKING_DIR/tracking_repo` at runtime (`REPO_DIR = WORKING_DIR / "tracking_repo"`, `OURS_v32.py:142`). Imports `biohub_tracking.*`. |
| `external/kaggle-cell-tracking-competition/scripts/…` | 677 | `63b550b8…` | Pre-rename dev checkout. Identical except 3 import lines (`tracking_cellmot.*` → `biohub_tracking.*`). Do **not** anchor-test against this. |
| `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/smoke_remote_output_v1/tracking_repo/scripts/…` | 933 | `c8f4552e…` | Already-patched *output* of a prior run. Not a valid anchor target. |

---

## 1. Executive summary

**These six mechanisms are BETA, not alpha.** Every one of them is posted publicly in a Kaggle
notebook that any of the ~2,019 teams can open, read, and copy in an afternoon. The measured
delta they represent is the gap from our **local CV57 0.914358 / verified public 0.881** to the
best clean public author's **live LB 0.916**. They are table stakes — the price of standing on
the current public frontier — and must never be booked as the campaign's differentiator. The
campaign's edge, if it exists, has to come from N1/N3, not here.

**Three findings that change the port plan:**

1. **Mechanisms 1 and 2 are one atomic patch, not two.** The bidirectional harmonic fusion lives
   *inside* the closure `_harmonic_probability_from_unet`, which the 4-view TTA loop calls once
   per view. The patch body hard-raises `RuntimeError` unless
   `BIOHUB_EDGE_TTA_MODE == "js_reliability_log_pool"` **and** `BIOHUB_EDGE_TTA_VIEWS == 4`.
   There is no code path that runs bidirectional fusion alone. Ranking them separately is not
   possible; they are a single 4.0× edge-inference item.

2. **The edge-TTA patch silently compiles but dies at runtime on our V32 source.** The `_bi_old`
   anchor matches our patched source exactly (count = 1) and the result passes `compile()` — but
   `_bi_new` references `_edge_tta_imgs`, a name our patch chain never binds, because our
   dual-seed patch #2 ends with `del imgs` where theirs ends with `_edge_tta_imgs = imgs`.
   **This is the single most dangerous item in this plan:** the notebook's own safety gate is
   `compile()`, and `compile()` passes. Failure is a `NameError` on the first edge pair, i.e.
   after GPU allocation, model load, and detection inference — deep into a paid GPU run.
   See §2.1 / B1.

3. **The forward-acceleration lookahead is not free-standing.** The contract calls it "~30 lines,
   pure geometry, no weights, no retraining. Best value per unit effort." The *functions* are
   exactly that (`yusuketogashi…3rd.py:1476-1516`, 41 lines incl. a self-test). But the author's
   **call site is nested inside `if LOCAL_ASSOCIATION_RANKER_MODE == 'full_motion_assignment':`**,
   inside `assign_pass()` inside `motion_relink_edges()` — a branch that only exists once the
   22-feature ranker (M3) is loaded. As the author ships it, **M4 depends on M3**. A standalone
   M4 port into our cost formula is feasible (~10 lines of glue) but is then an *unmeasured
   variant*, not the 0.916 code path. See §2.4 / B2.

**Also worth stating plainly:** the best clean public author sets
`BIOHUB_USE_DEEPCENTER_VETO = '0'` and `BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE = '0'` — **exactly as
we do** (`yusuketogashi…3rd.py:97-98` vs `OURS_v32.py:99-100`). Mechanisms 5 and 6 are *not* part
of the 0.916 recipe. Whoever framed them as "disabled in our config" relative to a stronger
public solution was comparing against `nikitagajbhiye30/biohub-11` and
`mige551/biohub-clean-122`, which score **below** yusuketogashi. Porting M5/M6 moves us *away*
from the measured 0.916 configuration, not toward it.

---

## 2. Per-mechanism analysis

### 2.1 M1+M2 — Bidirectional harmonic edge fusion + 4-view JS-reliability log-pool edge TTA

**One patch. Cannot be split.**

| Field | Value |
|---|---|
| Source | `yusuketogashi…3rd.py:1626-1808` (`_bidirectional_weight_guard`, `_bi_old`, `_bi_new`) |
| Patch target | `predict_unet_transformer.py`, inside the per-frame-pair edge loop |
| Config | `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT="0.20"`, `BIOHUB_EDGE_TTA_MODE="js_reliability_log_pool"`, `BIOHUB_EDGE_TTA_VIEWS="4"` |
| (a) Inference cost | **~4.0× on the edge/association half.** 4 views; each view = 1 extra `model.encode` (views 2-4) + 2 `predict_edges` (fwd + rev). Detection half unchanged. Net wall-clock on a V32-shaped run: **≈2.4–3.0×** end-to-end (detection TTA already dominates; edge work moves from ~25% to ~55% of the budget). |
| (b) Extra checkpoint | **None.** Reuses the loaded primary `model` only. Zero new datasets. |
| (c) Integration risk | **HIGH** — see B1 below, plus the reverse-direction caveat below the table. |
| (d) Zero-GPU local CV57 validation | **NO for the full mechanism** — it needs `model.encode` and `model.predict_edges` on volumetric input; that is GPU work by construction (our V32 hard-raises without CUDA at `OURS_v32.py:930`). **YES for the numerics**: the JS-reliability + log-opinion-pool block operates purely on a `(4, n_src, n_tgt)` probability tensor and is exactly reproducible on CPU from cached logits. See §5. |
| (e) Ordering | Must be applied **after** the dual-seed `_ensemble_replacements` list — `_bi_old`'s tail is `if secondary_model is not None:`, a string that exists only post-dual-seed. And **requires editing two entries of that list first** (B1). |

**Config note:** `BIOHUB_BIDIRECTIONAL_FUSION_MODE = "harmonic_probability"` is **decorative**. It
is set (`:108`), echoed into a config dict (`:1424`), and written to the audit blob (`:4151`) —
but the fusion code never reads it. The harmonic mean is hardcoded:

```
_harmonic = 1.0 / ((1.0 - _bidirectional_weight) / _forward_prob
                   + _bidirectional_weight / _reverse_prob)
```

Do not build a mode-switch around this key; it buys nothing and invites drift.

**Reverse-direction caveat (verified, and the real risk in this mechanism).** The patch calls
`predict_edges` with src/tgt swapped and then `.transpose(1, 2)`:

- **Shape-wise this is correct.** `predict_edges`
  (`external/Tracking_Support/repo/scripts/train_unet_transformer.py:506-520`) is a thin wrapper
  over `SimpleNodeTransformer.forward`, documented to return `(B, N_t, N_t1)`. Swapping gives
  `(B, N_tgt, N_src)`; the transpose restores `(B, N_src, N_tgt)`. Softmax is taken over `dim=1`
  (i.e. over *sources* for each target) consistently for both directions. Confirmed.
- **Semantically the reverse pass is out-of-distribution.** The transformer is trained on
  forward `(t → t+1)` pairs only. Feeding it `(t+1 → t)` is not a symmetry the model was trained
  to respect — it is a *different, deliberately perturbed* view, which is precisely why it adds
  information and precisely why the author must rescale it (mean/std match onto the forward
  distribution) before mixing. **The rescale ratio is clamped to `[0.5, 2.0]`.** If the reverse
  logits' spread falls outside that band, the clamp binds and the reverse view enters mis-calibrated
  rather than being rejected. **Instrument this:** log the fraction of frame pairs where
  `_reverse_ratio` hits either clamp bound. A high hit rate means the mechanism is silently
  degraded on our seeds and the author's `w=0.20` is not transferable to us.

#### BLOCKER B1 — the `_edge_tta_imgs` lifetime gap (compile-clean, runtime-fatal)

Our dual-seed patch list and theirs are **identical in 4 of 6 entries**. The two that differ are
exactly the two that keep the input tensor alive for edge TTA:

| Entry | Ours (`OURS_v32.py:988` list) | Theirs (`yusuketogashi…3rd.py:1604` list) |
|---|---|---|
| `[2]` — replacement **tail** | `…\n            del secondary_det_logits\n\n        del imgs` | `…\n            del secondary_det_logits\n\n        _edge_tta_imgs = imgs` |
| `[4]` — replacement | `'        del unet_out\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n'` | `'        del unet_out\n        del _edge_tta_imgs\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n'` |

Entries `[1]`, `[3]`, `[5]`, `[6]` are byte-identical between the two notebooks.

**Measured proof (harness output):**

```
bi_old matches OUR patched source: 1
compiles: YES
_edge_tta_imgs assigned anywhere in OUR patched+bi source: False
_edge_tta_imgs USED count: 3
```

So: anchor matches, patch applies, `compile()` says fine, and the run dies with
`NameError: name '_edge_tta_imgs' is not defined` at the first frame pair — on the GPU, after
model load. **Any port of M1+M2 must first rewrite our `_ensemble_replacements[2]` and `[4]`.**
This is a modification of code we already ship, not an additive patch.

**Note on `del imgs`:** the string `"        del imgs"` occurs **5 times** in our patched source.
Do not do a bare `.replace("        del imgs", …)`. The `[2]` anchor is the full multi-line
context string and is unique (count = 1, verified) — use it.

---

### 2.2 M3 — 22-feature learned local association ranker

| Field | Value |
|---|---|
| Source | `yusuketogashi…3rd.py:383-830` (loader, alias table, preflight at `:816-827`), `:2502` (`_ranker_context`), `:2599-2601` (`_ranker_matrix`), `:2690-2717` (scoring), `:2718+` (`full_motion_assignment` branch) |
| Patch target | **Notebook body, not the support pack.** No string patch involved. |
| Config | `BIOHUB_USE_LOCAL_ASSOCIATION_RANKER=1`, `BIOHUB_LOCAL_RANKER_MODE=full_motion_assignment`, `_FULL_WEIGHT=0.85`, `_PRIMARY_RETAIN_WEIGHT=0.15`, `_MARGIN_UM=0.35`, `_MIN_ADVANTAGE=0.15`, `_MAX_BONUS=0.20` |
| (a) Inference cost | **~1.0× GPU.** Ranker is a small MLP / sklearn-style `predict_proba` over an `(n_candidates, 22)` matrix, run once per frame pair on CPU. Adds CPU time in `motion_relink_edges` only. Negligible against volumetric inference. |
| (b) Extra checkpoint | **YES — `pilkwang/biohub-local-association-ranker-unet300-v1`.** Hard requirement. Loader raises if absent (`:773`). Search roots at `:551-556`. Must be attached as a Kaggle input to the submission kernel. |
| (c) Integration risk | **HIGH (largest code delta).** Requires rewriting `motion_relink_edges` from our **119 lines** (`OURS_v32.py:1695-1813`) to their **224 lines** (`yusuketogashi…3rd.py:2603-2826`) — a 157-changed-line diff. Restructures the cost matrix into `baseline_cost` / `primary_matrix` / `ranker_matrix` / `valid_by_row` / `predicted_by_source`, adds `node_time`, adds a `_ranker_context` global, and adds `raw_edges_for_context` to the signature. Our notebook has **none** of `LOCAL_ASSOCIATION_RANKER`, `_ranker_matrix`, `predicted_by_source`, `node_time`. |
| (d) Zero-GPU local CV57 validation | **YES — fully.** This is the standout: the ranker consumes only graph-derived features (degrees, densities, µm distances, `t_norm`, `edge_prob`) that exist in a cached candidate graph. Given cached V32 edge logits + node table, the whole `motion_relink_edges` rewrite is replayable on CPU and scoreable on CV57 with zero GPU. **Highest-confidence local validation of the six.** |
| (e) Ordering | Independent of M1+M2 (different file, different stage). **Must precede M4** as the author ships it (B2). Attach the ranker dataset before any run. |

**Feature contract** (`:400-470`): the 22 canonical names are `edge_prob`, `has_learned_edge`,
`source_in_degree`, `source_out_degree`, `target_in_degree`, `target_out_degree`,
`source_frame_count`, `target_frame_count`, `source_density_7um`, `target_density_7um`,
`candidate_rank_dist`, `candidate_count`, `edge_dz_um`, `edge_dy_um`, `edge_dx_um`,
`edge_dist_um`, `edge_xy_um`, `edge_abs_z_um`, `motion_dist_um`, `motion_gain_um`,
`source_has_prev`, `target_has_next`, `target_best_next_prob`, `t_norm`. (That enumeration is 24
names; the checkpoint's own `feature_names` metadata selects the operative 22 — the loader reads
it from the artifact at `:696-714` and falls back to `_RANKER_FALLBACK_FEATURES` only if absent.
**Do not hardcode a 22-name list; read it from the checkpoint.**) A large alias table maps
legacy public names (`learned_edge_prob`, `src_in_degree`, `raw_distance_um`, …) onto the
canonical set, so the loader tolerates several public artifact vintages.

---

### 2.3 M4 — Forward-acceleration lookahead

| Field | Value |
|---|---|
| Source | `yusuketogashi…3rd.py:1476-1501` (`forward_acceleration_lookahead`), `:1503-1508` (`forward_acceleration_bonus`), `:1510-1524` (semantic self-test), call site `:2731` / bonus `:2742` inside the `:2718` branch |
| Patch target | **Notebook body.** No string patch. |
| Config | `BIOHUB_USE_FORWARD_ACCELERATION_LOOKAHEAD=1`, `BIOHUB_FORWARD_LOOKAHEAD_MAX_ACCEL_UM=4.0`, `BIOHUB_FORWARD_LOOKAHEAD_MAX_BONUS=0.20` |
| (a) Inference cost | **1.0× GPU; CPU O(n_src × n_cand × \|frame_{t+1}\|).** Pure numpy geometry, no model calls. Inner loop scans all nodes at `target_t + 1` per candidate. On dense frames this is the one place it could bite — the author gates it with `next_step_gate_um=MOTION_RELINK_RELAXED_UM`, which prunes early. Measure before assuming free. |
| (b) Extra checkpoint | **None.** No weights, no retraining. |
| (c) Integration risk | **LOW for the functions, MEDIUM for the call site** — see B2. |
| (d) Zero-GPU local CV57 validation | **YES — fully.** Needs only `node_time`, `ids_by_t`, `position_um` and a cost matrix. All CPU. The author even ships a deterministic semantic unit test (`:1510-1524`) we can lift verbatim as a regression check. |
| (e) Ordering | As shipped: **after M3** (B2). As a standalone variant: independent, can go first. |

The mechanism itself is genuinely small and clean:

```python
current_velocity = target_pos - source_pos
for next_id in ids_by_t.get(target_t + 1, []):
    next_velocity = position_um[next_id] - target_pos
    if norm(next_velocity) > next_step_gate_um:  continue
    residuals.append(norm(next_velocity - current_velocity))
return min(residuals), len(residuals)
# bonus = max_bonus * max(0, 1 - residual_um / max_accel_um)
```

i.e. reward a candidate edge when *some* node in the following frame continues it at near-constant
velocity. Bonus is subtracted from the assignment cost.

#### BLOCKER B2 — M4's call site depends on M3

Enclosing-scope trace of the call site (verified line numbers):

```
line 2603   def motion_relink_edges(
line 2648     def assign_pass(
line 2718       if LOCAL_ASSOCIATION_RANKER_MODE == 'full_motion_assignment':
line 2731         residual_um, continuation_count = forward_acceleration_lookahead(…)
line 2742         bonus = forward_acceleration_bonus(…)
```

`LOCAL_ASSOCIATION_RANKER_MODE` is only meaningful once the ranker is loaded;
`motion_relink_edges` raises at entry if it is not (`if LOCAL_ASSOCIATION_RANKER is None: raise
RuntimeError('Local association ranker was not loaded during preflight.')`). So **the contract's
"best value per unit effort, no dependencies" framing is wrong for the shipped code path.**

**Two honest options:**

- **M4-faithful** (ranked below): port M3 first, then M4 lands as ~22 lines inside the branch the
  author measured. This reproduces 0.916's actual code path.
- **M4-standalone** (ranked as a cheap probe): our cost line is
  `cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob`
  (`OURS_v32.py`, inside `motion_relink_edges`). Subtracting the bonus there is ~10 lines of glue
  plus a one-line `node_time = {nid: int(n['t']) for nid, n in nodes_by_id.items()}`. We already
  have `ids_by_t`, `position_um`, `MOTION_RELINK_RELAXED_UM`, `MOTION_RELINK_LEARNED_BONUS`.
  **But be explicit in the ledger: this is an unmeasured variant.** The author's bonus is applied
  against an `evidence` term blending ranker and primary probability; ours would be applied
  against raw primary probability. Same geometry, different cost surface. It may help, it may not,
  and 0.916 is not evidence for it.

---

### 2.4 M5 — DeepCenter center-prior veto (present, config-disabled)

| Field | Value |
|---|---|
| Our setting | `BIOHUB_USE_DEEPCENTER_VETO='0'` (`OURS_v32.py:100`) |
| **yusuketogashi (0.916) setting** | **`'0'` — same as ours** (`:98`). Also `REQUIRE=0`, `GAP_VETO=0`, `SAFE_DIV_VETO=0`, `CHECKPOINT=''`. |
| `nikitagajbhiye30/biohub-11` setting | `REQUIRE=1`, `GAP_VETO=1`, threshold `0.25` |
| (a) Inference cost | **+1 full volumetric 3D U-Net pass** over every frame (center-prior head), plus a per-frame score cache (`_SCORE_CACHE_MAX_FRAMES=8`). Roughly **+30–50%** end-to-end. |
| (b) Extra checkpoint | **YES — `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`**, relative path `weights/full_frame_center/checkpoint_last.pt`. |
| (c) Integration risk | **LOW code risk** (all plumbing already in our notebook, `:261-280` equivalents; it is a pure config flip + dataset attach) **but HIGH evidence risk** — we would be adopting a mechanism the strongest clean public solution deliberately switches off. |
| (d) Zero-GPU local CV57 validation | **NO.** The veto consumes a learned per-voxel center prior; producing it requires a second GPU model pass. Only replayable if we first cache DeepCenter scores for CV57 — which is itself a GPU job. |
| (e) Ordering | Independent. Interacts with M1+M2 only through the edge probabilities it vetoes. |

**Recommendation: do not port.** It costs a checkpoint, ~40% inference, and cannot be validated
without GPU — to adopt a setting the 0.916 author rejected. If someone wants it, it is a config
flip we can make in one line at any future time; there is no porting work to bank.

---

### 2.5 M6 — Adaptive short-track rescue (present, config-disabled)

| Field | Value |
|---|---|
| Our setting | `BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE='0'` (`OURS_v32.py:99`) |
| **yusuketogashi (0.916) setting** | **`'0'` — same as ours** (`:97`) |
| `mige551` setting | `MIN_LEN=4`, `MIN_MEAN_EDGE_PROB=0.82`, `MAX_MEAN_EDGE_DIST_UM=3.25`, `MAX_NODES_FRAC=0.018`, `MAX_NODES_ABS=180`, trigger frac `0.10` |
| (a) Inference cost | **1.0×.** Pure post-processing over the output graph. Free. |
| (b) Extra checkpoint | **None.** |
| (c) Integration risk | **LOWEST of all six** — a config flip plus five numeric constants. All parameters already parsed in our notebook. |
| (d) Zero-GPU local CV57 validation | **YES — fully, and cheaply.** Operates on the final graph after short-track filtering. Replayable on cached CV57 graphs on CPU in minutes. **The cheapest experiment in this document.** |
| (e) Ordering | Last. Strictly post-processing; touches nothing upstream. |

**Recommendation: run it as a free CV57 sweep, expect nothing.** Zero GPU, zero checkpoint, ~5
constants. It interacts with our `BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1` and
`BIOHUB_OUTPUT_MIN_TRACK_LEN=6` — the rescue re-admits components the filter removed, gated on
`removed_frac > 0.10`. Because it is free to test locally, test it; because 0.916 disables it,
do not expect it to survive.

---

## 3. Verified patch anchors

All anchors below were applied programmatically to the pristine 677-line
`external/Tracking_Support/repo/scripts/predict_unet_transformer.py`, in source order, with match
counts recorded. Harness: `…/scratchpad/verify_anchors.py`.

### 3.1 Our V32 chain (baseline — confirms the harness and our own patches are sound)

```
[line 940] _old                      count=1  OK   (8-view planar detection TTA)
[line 988] _ensemble_replacements[1] count=1  OK   | downsample: tuple[int, ...] = (1, 4, 4),
[line 988] _ensemble_replacements[2] count=1  OK   | for f in range(W):        <-- MUST EDIT for M1+M2
[line 988] _ensemble_replacements[3] count=1  OK   | edge_logits_pair = model.predict_edges(
[line 988] _ensemble_replacements[4] count=1  OK   | del unet_out              <-- MUST EDIT for M1+M2
[line 988] _ensemble_replacements[5] count=1  OK   | model, window_size, downsample = load_model(weights_path, devi
[line 988] _ensemble_replacements[6] count=1  OK   | unet_batch_size=unet_batch_size,
-> 7 patches applied; result COMPILES
```

### 3.2 yusuketogashi chain (the target state)

```
[line 1556] _old                      count=1  OK
[line 1604] _ensemble_replacements[1] count=1  OK
[line 1604] _ensemble_replacements[2] count=1  OK
[line 1604] _ensemble_replacements[3] count=1  OK
[line 1604] _ensemble_replacements[4] count=1  OK
[line 1604] _ensemble_replacements[5] count=1  OK
[line 1604] _ensemble_replacements[6] count=1  OK
[line 1633] _bi_old                   count=1  OK   (bidirectional + 4-view edge TTA)
-> 8 patches applied; result COMPILES
```

### 3.3 Anchor status summary

| Anchor | Applies to pristine support pack? | Applies to **our V32-patched** source? | Flag |
|---|---|---|---|
| `_old` (8-view det TTA) | ✅ count=1 | n/a (we already ship it, identical) | — |
| `_ensemble_replacements[1]` | ✅ count=1 | identical to ours | — |
| `_ensemble_replacements[2]` | ✅ count=1 | **differs** — ours ends `del imgs`, theirs `_edge_tta_imgs = imgs` | ⚠️ **B1 — must edit ours** |
| `_ensemble_replacements[3]` | ✅ count=1 | identical to ours | — |
| `_ensemble_replacements[4]` | ✅ count=1 | **differs** — theirs adds `del _edge_tta_imgs` | ⚠️ **B1 — must edit ours** |
| `_ensemble_replacements[5]` | ✅ count=1 | identical to ours | — |
| `_ensemble_replacements[6]` | ✅ count=1 | identical to ours | — |
| `_bi_old` | ✅ count=1 (post-dual-seed) | ✅ **count=1, compiles — but runtime-fatal without B1 fix** | 🔴 **B1 — false-clean anchor** |

**No anchor fails to match.** The risk is the opposite of what the contract anticipated: an anchor
that matches, compiles, and still breaks. Flagging this is the substantive output of Task 1.

**Anchor hygiene rules for whoever executes the port:**
- Anchor against `external/Tracking_Support/repo/scripts/predict_unet_transformer.py`
  (md5 `a2bbd2b3…`), never the `tracking_cellmot` checkout or the `smoke_remote_output_v1` copy.
- Keep the existing `_ensemble_count != 1 → raise` guard pattern. It is the only thing standing
  between us and a silent partial patch.
- `compile()` is **not** a sufficient gate — B1 proves it. Add a post-patch name-binding assertion:
  every `_edge_tta_imgs`-class temporary referenced in a `_new` body must have a matching
  `^\s*<name>\s*=` in the patched source. ~5 lines, catches this whole failure class.

---

## 4. Ranked port order

Ranking is **expected CV57 gain per unit integration risk**, with a strong prior that anything the
0.916 author disabled is not a gain.

| Rank | Mechanism | GPU cost | New dataset | Risk | Zero-GPU CV57? | Rationale |
|---|---|---|---|---|---|---|
| **1** | **M6 — adaptive short-track rescue** | 1.0× | none | lowest | **yes, minutes** | Free to test. Config-only. Pure post-processing. Costs nothing to falsify. Prior: will not survive (0.916 disables it), but the *information* is free. Run it first precisely because it cannot block anything. |
| **2** | **M4-standalone — forward-acceleration lookahead** | 1.0× | none | low | **yes, fully** | ~30 lines of pure numpy + ~10 lines glue. No weights, no checkpoint, no GPU. Ships with a deterministic self-test. **Caveat: unmeasured variant** (B2) — the author's bonus rides on a ranker-blended `evidence` term, ours would ride on raw primary prob. |
| **3** | **M3 — 22-feature local association ranker** | ~1.0× | **yes** (`biohub-local-association-ranker-unet300-v1`) | high (157-line diff) | **yes, fully** | Biggest code delta but **fully CPU-validatable on CV57**. This is the mechanism where local validation is most trustworthy, so the high code risk is bounded by cheap measurement. Unblocks M4-faithful. |
| **4** | **M4-faithful — lookahead inside `full_motion_assignment`** | 1.0× | (M3's) | low, given M3 | **yes** | Once M3 lands this is ~22 lines in the branch the author measured. Reproduces the actual 0.916 path. Compare against M4-standalone's CV57 to learn whether the `evidence` blend matters. |
| **5** | **M1+M2 — bidirectional harmonic fusion + 4-view edge TTA** | **~2.4–3.0× end-to-end** | none | **high (B1)** | **partial** (numerics only) | No new checkpoint, and it is the mechanism most likely to carry real signal (2 directions × 4 views of genuinely new model evidence). But it is the only item that (i) cannot be locally validated end-to-end without GPU, (ii) requires editing code we already ship, and (iii) has a compile-clean/runtime-fatal trap. Land it last, behind everything cheap. |
| **—** | **M5 — DeepCenter center-prior veto** | +30–50% | **yes** | low code / high evidence | **no** | **DO NOT PORT.** 0.916 disables it. Costs a checkpoint and ~40% inference, cannot be validated without GPU, and adopting it means overriding the best clean public author's own ablation. It is a one-line config flip available forever; there is nothing to bank by porting it now. |

**Recommended execution order: M6 → M4-standalone → M3 → M4-faithful → M1+M2. M5 dropped.**

The shape of this ordering: **every item that can be falsified on CPU comes before the item that
cannot.** M1+M2 is the biggest single cost in the plan and the only one that must be paid in GPU
time to learn anything. Ranks 1-4 are cumulatively ~0 GPU cost and answer most of the question.

---

## 5. Zero-GPU local CV57 validation plan, per mechanism

Precondition for all of it: **cache one V32 CV57 forward pass** — per frame pair, the
`edge_logits_pair` tensor; per dataset, the node table (`t`, `position_um`, degrees, densities)
and the candidate-graph edge list. One GPU run, reused by every experiment below. Without this
cache, nothing here is zero-GPU; with it, ranks 1-4 are entirely CPU.

| Mech | Validatable on CPU? | Method | Verdict signal |
|---|---|---|---|
| **M6** | **yes, complete** | Replay the short-track filter + rescue over cached final graphs. Sweep `MIN_LEN`, `MIN_MEAN_EDGE_PROB`, `MAX_MEAN_EDGE_DIST_UM`, `MAX_NODES_FRAC/ABS` from mige551's values. | CV57 delta vs 0.914358. Minutes per point. |
| **M4-standalone** | **yes, complete** | Lift `forward_acceleration_lookahead` / `forward_acceleration_bonus` verbatim. Run the author's semantic self-test (`:1510-1524`) as a regression gate. Recompute our `cost` matrix from the cache with the bonus subtracted; re-run assignment. Sweep `MAX_ACCEL_UM ∈ [2,6]`, `MAX_BONUS ∈ [0.05,0.30]`. | CV57 delta. Also report `forward_lookahead_bonus_edges / candidates` to confirm the mechanism is actually firing and not silently gated to zero by `MOTION_RELINK_RELAXED_UM`. |
| **M3** | **yes, complete** | Attach the ranker artifact locally (CPU inference). Rebuild the 22-feature matrix from the cached node/edge tables. **First run the author's own preflight** (`:816-827`) — it probes the checkpoint, reads `feature_names`, and fails loudly on a feature-contract mismatch. Then replay the `full_motion_assignment` cost surface. | CV57 delta, plus `local_ranker_scored` count and mean ranker probability vs the author's audit fields. If feature names disagree with the checkpoint metadata, **stop** — do not fall back to the hardcoded list. |
| **M4-faithful** | **yes** | Same cache as M3, with the bonus inside the `evidence` branch. | CV57 delta **vs M4-standalone** — this is the experiment that tells us whether B2 mattered. |
| **M1+M2** | **partial only** | Cannot produce reverse-direction or TTA-view logits without GPU. **What can be done on CPU:** (i) unit-test the JS-reliability + log-opinion-pool block on synthetic `(4, n_src, n_tgt)` probability tensors — check that identical views give uniform `_view_weight`, that one outlier view is smoothly downweighted, that `_pooled_prob` sums to 1; (ii) **simulate the bidirectional half at zero extra GPU cost by re-running `predict_edges` with src/tgt swapped on the *cached* features** if the feature cache was retained — this gives the fusion's CV57 delta at `w=0.20` without the 4-view multiplier. | (ii) is the high-value cheap probe: **it isolates M1 from M2 empirically even though they cannot be isolated in code.** If bidirectional fusion alone shows ~0 on CV57, the 4× TTA is very unlikely to be worth 2.4-3.0× wall clock. |
| **M5** | **no** | Would require caching a DeepCenter forward pass — itself a GPU job on a checkpoint we do not have. | — (not porting) |

**Preregister before running:** CV57 GO threshold per mechanism, and the decision rule for a
negative result (drop vs. re-tune). Otherwise the sweeps in this table are an invitation to
overfit 57 videos.

---

## 6. Conflicts with our constraints

### 6.1 Fork outdegree capped at 2 — **NO CONFLICT**

Enforced in the support pack at `predict_unet_transformer.py:83-93`:

```python
max_children_per_node: int | None = None
def __post_init__(self) -> None:
    # When ILP is enabled it handles parent/children constraints itself,
    # so greedy limits are left unconstrained (None).  When ILP is
    # disabled, default to 1/1 to avoid unconstrained edge assignment.
    if not self.use_ilp:
        if self.max_parents_per_node is None:  self.max_parents_per_node = 1
        if self.max_children_per_node is None: self.max_children_per_node = 2
```

We run with ILP on (`--use-ilp`, `OURS_v32.py:1052`), so the cap is the ILP's own division
constraint, not the greedy limiter. None of M1-M6 touches ILP constraint construction:
M1+M2 change only edge *logit values* upstream of candidate generation; M3/M4 change only the
*cost matrix* inside `motion_relink_edges` (post-ILP relink); M6 is pure post-processing.
The `add_safe_divisions_postlink` path (`OURS_v32.py:2247`) and its frac caps
(`SAFE_DIV_FRAME_FRAC_CAP=0.0076`, `GLOBAL_FRAC_CAP=0.00375`) are likewise untouched.

**Watch item, not a conflict:** M1+M2's log-opinion pool sharpens the edge distribution
(`softmax(_pooled_log)` after per-view reweighting). Sharper edge posteriors → more confident
division candidates → the frac caps may start binding where they previously did not. Log
`safe_divisions_added` before/after and confirm the caps are not silently truncating.

### 6.2 Dual-seed logit blending — **NO CONFLICT; ONE DELIBERATE DELTA**

Our dual-seed config vs the 0.916 author's, key by key:

| Key | Ours (`OURS_v32.py`) | yusuketogashi (`:1378-1389`) | Status |
|---|---|---|---|
| `BIOHUB_SECONDARY_EDGE_WEIGHT` | `0.15` | `0.15` | same |
| `BIOHUB_SECONDARY_DETECTION_WEIGHT` | **`0.5`** | **`0.475`** | ⚠️ **only difference** |
| `BIOHUB_SECONDARY_LINK_MODE` | `low_margin_consensus` | `low_margin_consensus` | same |
| `BIOHUB_SECONDARY_MIX_TEMPERATURE` | `1` | `1` | same |
| `BIOHUB_SECONDARY_LOW_MARGIN_MAX` | `0.35` | `0.35` | same |
| `BIOHUB_DUAL_SEED_EDGE_THRESHOLD` | `0.48` | `0.48` | same |

The blending *machinery* is byte-identical (4 of 6 `_ensemble_replacements` entries match
exactly; the other 2 differ only by the `_edge_tta_imgs` lifetime, §2.1/B1). So M1+M2 composes
with our dual-seed path by construction — the author runs both together.

On `0.5` vs `0.475`: a 5% relative difference in how much the secondary seed's *detection* logits
are mixed in. It is a free one-line A/B on CV57 with zero GPU **if** detection logits are cached.
Worth resolving before attributing any M1+M2 delta, so we are not comparing across two changes.
Do not blind-copy `0.475` — our `0.5` may have been tuned on CV57; check the ledger for where it
came from before overwriting it.

### 6.3 Detection threshold 0.96875 — **NO CONFLICT**

`BIOHUB_DET_THRESHOLD = "0.96875"` in **both** our V32 (`:80`) and yusuketogashi (`:78`).
Identical. No mechanism in this plan reads or modifies it. M1+M2 operate strictly on *edge*
logits, downstream of detection; M3/M4/M6 operate on the graph.

### 6.4 `EXPECTED_TEST_COUNT = 4` — **NO CONFLICT IN THE NOTEBOOK; LIVE HAZARD IN THE HARNESS**

Checked directly. **Neither notebook contains a fixed test count.** Both derive the test set
dynamically:

```python
def list_test_stems() -> list[str]:                    # OURS_v32.py:1010, yusuketogashi:1811
    stems = sorted(p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith(".zarr"))
```

and every downstream check is relative (`len(geffs) != len(test_stems)` at `OURS_v32.py:2748`;
`worker_count = min(2, available_gpu_count, len(test_stems))` at `:1180`;
`expected_datasets = set(test_stems)` at `:2846`). All cardinality-safe.

The `EXPECTED_TEST_COUNT = 4` gate lives in **our kernel-packaging harness**, not the notebook:

- `scripts/public_temporalunet_baseline.py:533` (template), used at `:590, 615-617, 655`
- `scripts/v32_submission_bisect.py:115`, used at `:163`

with the failing predicate
`if len(selected) != EXPECTED_TEST_COUNT or len(set(selected)) != EXPECTED_TEST_COUNT: raise`.

**Consequences for this plan:**

1. **No mechanism here introduces a count-derived constant.** M1+M2 iterate frame pairs within a
   dataset; M3/M4 iterate candidates within a frame pair; M6 iterates tracks. None iterates
   datasets or divides by a dataset count. **The port plan is cardinality-neutral.**
2. **But the port plan is also not a fix.** Any kernel built by
   `scripts/public_temporalunet_baseline.py` or `scripts/v32_submission_bisect.py` will still
   fail closed on the hidden rerun regardless of which mechanisms are ported. **That is N0/N4's
   problem and must be resolved before N4 packaging, or every mechanism in this document scores
   zero on the LB.** N0's remediation (demote the count gate to a warning, per
   `runs/N0_SUBMISSION_FORENSICS.md:271`) is a hard prerequisite for banking any of this.
3. **Rule for the porter:** do not add `len(test_stems)`-derived constants, per-dataset
   normalizers, or `!= <literal>` inventory assertions while integrating. Keep every new stat a
   running accumulator (as the author does — `stats['forward_lookahead_candidates'] += …`), never
   a mean over a presumed dataset count.

---

## 7. Verdict

**GO (conditional).** Per the node contract's gate — "a ranked port order with exact patch anchors
and a zero-GPU CV57 validation defined per mechanism; any anchor that will not match our V32
source must be flagged" — all three are delivered: ranked order in §4, anchors verified against
the real support-pack source in §3, per-mechanism zero-GPU plan in §5. No anchor fails to match;
the one anchor hazard (B1) is a *false-clean* match and is flagged explicitly.

**Conditions on proceeding to N4:**

1. **B1 must be fixed before any M1+M2 GPU run.** Rewrite `_ensemble_replacements[2]` and `[4]`,
   and add the name-binding assertion (§3.3) — `compile()` alone will not catch this class.
2. **B2 must be recorded honestly.** M4-standalone is an unmeasured variant; do not attribute the
   author's 0.916 to it in the ledger.
3. **M5 is dropped**, and M6 is expected to fail. Both because the 0.916 author disables them.
4. **N0's `EXPECTED_TEST_COUNT` remediation is a hard prerequisite** for N4. Without it every
   mechanism here scores zero on the hidden rerun.
5. **Preregister CV57 GO thresholds** before the sweeps in §5, or they overfit 57 videos.

**Framing for the ledger, stated plainly as the contract requires:** these six mechanisms are
**BETA**. They are publicly posted in Kaggle notebooks readable by any of the ~2,019 teams. They
are the *measured* route from our 0.914358 toward the best clean public author's ~0.916 — table
stakes, not an edge. Funding them is correct as cheap insurance against being below the public
frontier. Booking them as the campaign's differentiator would be a category error.

---

## 8. Reproduction

```bash
# Anchor verification (re-runnable, no GPU, ~2s)
python3 /private/tmp/claude-501/-Users-agent-Desktop-cell-tracking/\
a7c44d62-1f2f-4555-82d5-7c200244dd3d/scratchpad/verify_anchors.py

# Re-pull notebook sources if the scratchpad is cleared
/Users/agent/Desktop/cell_tracking/.venv313/bin/kaggle kernels pull \
  yusuketogashi/no-hack-biohub-cell-another-approch-3rd -p <dir> -m
/Users/agent/Desktop/cell_tracking/.venv313/bin/kaggle kernels pull \
  nikitagajbhiye30/biohub-11 -p <dir> -m
/Users/agent/Desktop/cell_tracking/.venv313/bin/kaggle kernels pull \
  mige551/biohub-clean-122-det-0-9690 -p <dir> -m
```

Artifacts written by this node (scratchpad, non-authoritative):
`verify_anchors.py`, `patched_ours.py`, `patched_theirs.py`, `patched_ours_plus_bi.py`.

**No files under `scripts/` or `src/` were modified by this node.**

---

## 9. Proposed LEDGER row (for the root coordinator)

Per `CLAUDE.md`, the root coordinator is the sole writer of
`reports/g02/screening/LEDGER.md`. **This node did not write it.** The row below is drafted to the
existing 14-column schema and is ready to paste.

```
| G02-CLEAN-PUBLIC-MECHANISM-PORT-PLAN-20260806 | six legitimate mechanisms from the best clean public notebook (`yusuketogashi/no-hack-biohub-cell-another-approch-3rd`, author live LB `0.916`): bidirectional harmonic edge fusion, 4-view JS-reliability log-pool edge TTA, 22-feature learned local association ranker, forward-acceleration lookahead, DeepCenter center-prior veto, adaptive short-track rescue | BETA / RESEARCH + FEASIBILITY / CPU / NO PIPELINE CHANGE | deliver a ranked port order with patch anchors verified against the real support-pack source and a zero-GPU CV57 validation per mechanism; flag every anchor that will not match our V32 | **GO (CONDITIONAL):** all 8 anchors verified against pristine support pack `predict_unet_transformer.py` (677 lines, md5 `a2bbd2b3…c48bd9`), all `count=1`, patched result compiles. **Two structural blockers found. B1:** the `_bi_old` edge-TTA anchor matches our V32-patched source and passes `compile()` but is runtime-fatal — `_bi_new` references `_edge_tta_imgs`, which our dual-seed `_ensemble_replacements[2]` never binds (ours ends `del imgs`, theirs `_edge_tta_imgs = imgs`); `[4]` also differs. 4 of 6 entries byte-identical. **B2:** forward-acceleration lookahead's call site is nested inside `if LOCAL_ASSOCIATION_RANKER_MODE == 'full_motion_assignment'` — as shipped M4 depends on M3, contradicting the "no dependencies" framing. **Mechanisms 1+2 are one atomic patch** (hard-raises unless `EDGE_TTA_MODE=js_reliability_log_pool` and `VIEWS=4`). **The 0.916 author disables DeepCenter veto and adaptive short-track rescue exactly as we do** | pristine support pack md5 `a2bbd2b33666aab525a9b4b510c48bd9`; AST anchor harness `scratchpad/verify_anchors.py` (re-runnable, ~2 s, no GPU); our V32 chain 7/7 anchors `count=1` + compiles; yusuketogashi chain 8/8 `count=1` + compiles; cross-application ours+`_bi` compiles with `_edge_tta_imgs` unbound | read-only source diff against the authoritative support-pack artifact, not the `tracking_cellmot` checkout or the already-patched `smoke_remote_output_v1` copy; no pipeline modification, no launch, no submission | 0 | 0 | 0 / $0 | GO CONDITIONAL / PORT ORDER M6→M4-STANDALONE→M3→M4-FAITHFUL→M1+M2 / DROP M5 / B1 BLOCKS ANY M1+M2 GPU RUN | These six mechanisms are **BETA**: publicly posted and copyable by any of ~2,019 teams. They are the measured route from local CV57 `0.914358` toward the best clean public author's `~0.916` — table stakes, not an edge; they must not be booked as the campaign's differentiator. Ordering puts every CPU-falsifiable mechanism ahead of the only one that cannot be validated without GPU (M1+M2, ~2.4–3.0x end-to-end). M5 dropped: it costs a checkpoint plus ~40% inference, cannot be validated without GPU, and the 0.916 author disables it. `compile()` is not a sufficient patch gate — B1 proves it; a name-binding assertion is required. Notebooks are cardinality-safe (`list_test_stems()` is dynamic in both); the `EXPECTED_TEST_COUNT = 4` gate lives in our packaging harness (`scripts/public_temporalunet_baseline.py:533`, `scripts/v32_submission_bisect.py:115`) and N0's remediation is a hard prerequisite for N4 or every mechanism here scores zero on the hidden rerun. | `dag_prompts/2026-08-06-alpha-hunt/runs/N2_CLEAN_PORT_PLAN.md` |
```
