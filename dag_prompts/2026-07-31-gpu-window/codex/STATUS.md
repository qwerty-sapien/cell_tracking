# Pack status — read before running any node

**Last reconciled: 2026-07-31 07:38Z. Two lanes closed since this pack was drafted; a third
is RUNNING and Codex holds the serial lock — do not touch Kaggle.**

## Live now

`onebitbrain/g02-trackastra-native-fit-v1` — native-base wave Q1/Q2: does a pretrained
tracker's native graph and forks beat APNCC `0.881522069960`? Fit105 (36 44b6 / 69 6bba),
one CV57 exposure, `17,100 s` internal / `18,000 s` external kill, halt immediately if
strictly above control. Objective is `adjusted_edge_jaccard + 0.1 × division_jaccard`.
Owner granted the `>=5 h` per-launch gate for this window. Status at poll 3: `RUNNING`,
`96.59 s` elapsed. **A supervisor already holds the lock through terminal download.**

## RAW3D-NOLINK-MVE (Option 1) — CLOSED, zero GPU

Plan §6.6 recorded the owner selecting Option 1 and made a zero-weight raw-3D probe the
binding precondition before any weight acquisition, training, or GPU spend. That probe ran
and **failed**:

| Held fold | A1 AUC | CG0-budget retention |
|---:|---:|---:|
| 0 | `0.726453902` | `0.003267974` |
| 1 | `0.775858844` | `0.010101010` |
| 2 | `0.599374874` | `0.011976048` |
| 3 | `0.646261214` | `0.005128205` |
| 4 | `0.641887210` | `0.000000000` |
| **mean** | **`0.677967209`** | **`0.006094647`** |
| **fold floor** | **`0.599374874`** | **`0.000000000`** |

CPU only, `254.704 s`, all 41 videos / `1,345,497` actions / `1,118` beneficial. It beat
DINOv2-XY on mean AUC (`0.677967209` vs `0.670411743`) and `A1−A0` was `+0.031393472` — so
raw 3-D *does* carry genuine endpoint-pair information — but it missed the frozen
`0.680411743` mean-and-fold-floor continuation line and the `0.85/0.020394` surface by a
wide margin. **Verdict: PARK-C raw-3D; Stage 1 closed; no acquisition, training, or GPU.**

Option 1 therefore consumed **zero Kaggle quota** and needs none.

---

*Earlier reconciliation, still accurate:*

## The PEFT lane this pack was first written for is TERMINAL

While `C0`–`C4` were being authored, Codex launched and completed the node. Outcome:

| Field | Value |
|---|---:|
| Verdict | **FAIL / CASE B / PARK-C** |
| Debit | `0.04` T4-h (`24.98 → 24.94` remaining) |
| Step 0 | `1.500973 s`, zero competition-path visits |
| Training | `145.680 s` — versus a `4.8 h` projection, a **`118.6×` miss** |
| Descriptive AUC | `0.634749` at epoch 8 |
| Graph-only nested selection | epochs `1/2/2/2/2`, AUC `0.496495`, retention `0.002683` |
| Adjusted / raw / node | `+0.000133827` / `+0.000133160` / `0` — adjusted misses `+0.005` |
| Local replay | 15/15 |

Amendment A was consumed as intended: five outer folds (`1/2/2/2/2` epochs), sealed
`PREREGISTRATION.json` untouched, one serial version, exact version delta `+1`.

**Do not re-run this node.** Frozen case B is binding: no larger adapter and no full
fine-tune on this backbone. `6bba`, CV57, packaging, and submission did not run and stay closed.

## What the result means

The adapter scored **below** the frozen suppliers it was meant to beat. DINOv2 appearance and
C2 assignment evidence both landed at DELETE AUC `~0.670`; the trained adapter reached
`0.634749` descriptive and `0.496495` under honest graph-only selection — the latter is
essentially chance.

That is the label-side-ceiling hypothesis confirmed by experiment rather than inference:
adding supervised capacity against the **pointwise** beneficial-action label did not help,
and the plan's own §6.2 escalation is now closed empirically. This is a real result. It cost
`0.04` T4-h.

## What this pack is now

`C1`–`C4` were written against the PEFT lane but are **lane-parameterized**: wherever they
name a slug, kernel directory, contract, projection ceiling, or hard kill, substitute the
values from whichever node is currently authorized. The gates, failure families, auth facts,
and receipt requirements are lane-independent and carry over unchanged.

**`C0` has not been built and is still needed.** The PEFT run hit the known owner `403` on
its first dataset status query and correctly did not retry — exactly the endpoint-scope
behavior `C0`/`C1` encode. Building `cmux_ops.py` once, shared, remains the fix for
`KFG-18` (no structural call site) and for the launch-race class.

## The next decision is at a human barrier

`tiered_surface/branches/training_feasibility_v1/DECISION_DOSSIER.md` recommends, and blocks on:

- **Option 1 — open one scoped `RAW3D-NOLINK-MVE` node.** Warm-started crop-based 3D model,
  `<=0.25` T4-h matching calibration first, then a projected `2–6` T4-h useful checkpoint,
  hard kill and serial launch fixed before launch. Recommended as the only new-information
  falsifier.
- **Option 2 — consolidate at `0.881` and stop.**

Trackastra fine-tuning is re-costed at `4.05–5.96` T4-h — now affordable, but **not
recommended**: it consumes the budget without producing new information.

**No agent may choose.** This is a locked-plan amendment barrier. Until the owner picks,
`C1`–`C3` have no authorized target and must not run.

## Budget

`5.06` used / `24.94` remaining of `30.0`, refresh `2026-08-01T00:00:00`. Competition deadline
`2026-09-29T23:59:00` — the season is long; this window is not the last one.
**Quota expiry is not productivity** (`AGENTS.md`). If Option 1 is chosen, it needs `2–6` h,
not `24`.
