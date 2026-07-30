# Recall N2 — cheap local falsifier for candidate supply

**Verdict: DEAD.** The preregistered `NDR-FBC-local-translation-v1` realization
recovered **9 / 290 = 3.103448%** of the frozen slice’s pool-absent true edges.
That is below the registered `< 20%` DEAD boundary. This is a terminal result for
this tested realization; it does **not** advance to `N3_human_kaggle_gate.md`.
A different observable would require a new named mechanism rather than a retry,
parameter adjustment, or reframing of this run.

## Frozen scope, written before generation

The slice manifest was written before raw candidate generation:
[`N2_SUPPLY_SLICE_MANIFEST.json`](N2_SUPPLY_SLICE_MANIFEST.json), content SHA-256
`8ceb16da41d92966e1612439b66a2c3b39084b68d315b38c929d2b9f43fd4e16`
(self-declared canonical manifest SHA-256
`77f633818a8579759e920dacdb957557a572053291152da243123d3a2712a6f8`).

It fixes the lexicographically first twelve 6bba identities from the H1 fit105
source, with every frame `0..99` in each (no video subsampling):

`train:6bba_07477033`, `085bf656`, `09961292`, `0e7c0d07`, `12665c0e`,
`13f19531`, `1d0d8384`, `1ebfb80d`, `1f58c2f6`, `207c6aaf`, `20852818`, and
`268e1230`.

The exact evaluator target was committed in that manifest before the generator
ran: all adjacent-frame truth continuation edges on those videos that were both
absent from selected baseline truth edges and outside the frozen physical
`10 um` / cap-`8` pool. It contains **290** edges (`target` SHA-256
`f54dbdc1c0581ad29c4079cbabab297909f16aed61a10930416b512a342a3b14`), of
which **162** lack a prediction endpoint and correctly remain in the recovery
denominator. The same slice’s frozen candidate pool is **502,336** pairs.

Only fit-only data were opened. No held fold, training, graph run, score/delta,
GPU, Kaggle asset/action, RunPod, submission, remote push, or upload occurred.

## Fixed minimum viable mechanism

`NDR-FBC-local-translation-v1` is the N1 supplier’s smallest CPU-only,
label-blind instantiation:

- for every nonterminal baseline node without a selected outgoing edge, take a
  raw-intensity local `±[8, 32, 32]` voxel `z/y/x` crop at `t` and `t+1`;
- subtract each crop median, clamp negative residuals to zero, and transport the
  source by the two raw intensity centroids’ displacement;
- offer the nearest frozen `t+1` detection in physical space, only if the reverse
  raw transport returns to the original source; and
- discard offers already in the frozen `10 um` / cap-`8` pool.

The window, operation, one-offer policy, nearest-node tie break, and reverse
certificate were fixed in the manifest. The generator did not load the committed
target labels.

## Exact results

| Gate quantity | Result |
|---|---:|
| Pool-absent target edges | **290** |
| Recovered target edges | **9** |
| **Recovery** | **3.103448%** |
| Frozen candidate pool | 502,336 |
| New certified, out-of-pool offers | 1,173 |
| Candidate-set inflation | **1.002335090457×** (increment `0.233509%`) |
| 10–12 um background pairs on this slice | 310,802 |
| 10–12 um target edges / recovered | 13 / 1 |
| 10–12 um target recovery | 7.692308% |

Generation accounting: 7,931 sources reached the reverse-certificate decision;
561 failed that certificate, 6,197 certified offers duplicated the frozen pool,
and 1,173 survived as new candidates. These quantities do not alter the verdict:
the preregistered recovery is `9 / 290`, far below `20%`.

## Required integrity fixtures

All passed (failed closed as required):

| Fixture | Result |
|---|---|
| Permuted/non-adjacent frame map | rejected before raw registration |
| Label-shuffled evaluator target | rejected by canonical target digest |
| Empty foreground | raised `EmptyForegroundError`; emitted no offer |
| Byte repeat | byte-identical candidate receipt |

Both generation passes produced candidate SHA-256
`0eecfb958147d5ede6093017d5f531a3667501d7a8312a0a63943bbf6fd1713f`.
The complete result receipt SHA-256 is
`81a2b6d5e74906ca8a14dfe25f0ece7bad9b521b143ecb8b1173891e392ab757`.

## Local resource guard

- Fresh 45-second idle preflight: **PASS_ZERO_SWAP_GROWTH**, `45.949648 s`,
  `0 MiB` swap growth; receipt SHA-256
  `75ca567a25d7bab5f9fb967f80a41c8f94e359a337737364741afb3e1b4515a4`.
- Guarded N2 run: **PASS**, `315.815286 s`, peak process-tree RSS
  `0.405228 GiB`, initial/peak/ending swap `673.120000 / 673.120000 /
  673.120000 MiB`, hence **0 MiB** growth; receipt SHA-256
  `acffbc9284e7fabd7b1e1994aa652ee9073265d5696add0cd3db3188ffc8c9b4`.

## Registered decision

The N2 threshold is `GO >= 50%` recovery at `<= 3x` inflation, `PARK >= 20%`
under the stated limitations, and `DEAD < 20%` recovery (or an integrity
failure). Although inflation (`1.002335090457×`) and all integrity conditions
were favorable, **3.103448% < 20%**, so the registered verdict is **DEAD**.

## Proposed LEDGER row — do not write

| ID | Mechanism | Gate | Result | Status | Note |
|---|---|---|---|---|---|
| `RECALL-N2-NDR-FBC-20260730` | `NDR-FBC-local-translation-v1`, the N1 raw-intensity forward/backward candidate supplier | recovery `>=50%` at `<=3x`; PARK `>=20%`; all fixtures/repeat/zero-swap required | `9/290 = 3.103448%`; `1.002335090457×`; fixtures closed; repeat exact; `0 MiB` swap | **DEAD** | Tested realization is terminal. Do not tune or re-screen it; N3 was not entered. |
