# Recall N0 — pool recall ceiling

**Verdict: GO.** The preregistered 44b6 pool-absent zero-FP adjusted-edge-Jaccard
ceiling is **`+0.052004594313`**, above the fixed `>= +0.030` threshold. This
is a CPU-only diagnostic; no model, threshold, Kaggle asset/kernel/run, GPU debit,
submission, remote push, or ledger write occurred.

## Frozen scope and integrity

- Scope: all **36** frozen held-fit 44b6 videos; no subsampling.
- Baseline reproduced exactly: edge TP/FP/FN = **10,341 / 1,064 / 673**;
  adjusted edge Jaccard `0.859584981641`, raw edge Jaccard `0.856184798808`,
  node recall `0.993925630490`.
- Frozen population manifest SHA-256: `4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c`.
  The 36 matching occupied-swap shards authenticated to the manifest; their
  combined receipt hash is `31c2cabc990960b9be43d7bd62e9f9f30fec8015adbb9933641314761c717af7`.
- Baseline receipt `RESULT_44B6.json` SHA-256:
  `fb9202586242f506a7cd8437d780bc93ba1fcfe5de957e22de8d6a2e73f94834`;
  authenticated baseline-fragment aggregate receipt:
  `dcaac9cf999a12577e42ef0491afe592df11e89b100624c5a500211267c21530`.
- Raw archive was read at its frozen size `87,393,127,165` bytes, under its
  banked SHA-256 `538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19`.

## FN partition and zero-FP ceilings

Pool-present means the matched prediction-node pair is a member of the pinned
occupied-swap candidate population. Pool-absent means no such legal frozen candidate
exists; a missing matched prediction endpoint is necessarily absent. No positive
free-add action exists in the frozen P0 surface.

| FN partition | Count | Zero-FP adjusted Δ | Zero-FP raw Δ | Node-recall Δ | Oracle TP/FP/FN |
|---|---:|---:|---:|---:|---:|
| Pool-present | 47 | `+0.003922925139` | `+0.003891372744` | `0` | 10,388 / 1,064 / 626 |
| Pool-absent | 626 | **`+0.052004594313`** | `+0.051829773141` | `0` | 10,967 / 1,064 / 47 |
| Total held FNs | 673 | — | — | — | — |

The partition reconciles exactly: `47 + 626 = 673`. Of the 626 pool-absent FNs,
**84** lack at least one matched prediction endpoint.

## Pool-absent geometry

Physical GT-endpoint displacement (µm), using the frozen per-video scale:

| Bin | Count |
|---|---:|
| `[0,7)` | 534 |
| `[7,10)` | 66 |
| `[10,12]` | **18** |
| `(12,15]` | 6 |
| `>15` | 2 |

Range: `0`–`18.691915239282` µm; median `2.187723202898` µm. The requested
10–12 µm shell overlap is **18/626** pool-absent held FNs.

## Free-add reconciliation

P0 recorded `free-add = 0`. This audit shows why that result does not eliminate
recall supply: **626/673 (93.02%)** held FNs are outside the frozen pool entirely,
and its zero-FP ceiling is `+0.052004594313`. The 47 in-pool FNs have only
`+0.003922925139` zero-FP headroom, below `+0.005`; none is a positive free-add
action, so existing-pool recovery is also conflict/cost constrained. Thus the
dominant limitation is supply, with an additional no-free-add cost constraint for
the small present subset.

## Local resource guard

- Fresh idle preflight: **PASS_ZERO_SWAP_GROWTH**, `45.737638 s`, initial/peak/end
  swap `673.120/673.120/673.120 MiB`, growth `0 MiB`; SHA-256
  `461323c11c87f9464680cb6faaf544f7f4ee5a0218a4d9aed2168bc2ca26217d`.
- Guarded audit: **PASS**, `52.741064 s`, peak process-tree RSS `0.601456 GiB`,
  swap growth `0 MiB`; receipt SHA-256
  `cfad74dabb6c3dd88695cb70457f4309ca404bbcb1736d73073d44d01f93eda3`.
- Audit receipt (including expected benign official-matcher warnings) SHA-256:
  `7d812fbbf2884e26e945e2d3044c4b438cd5f68124210cc6e685cb66a24a71ba`.

## Proposed LEDGER row — do not write

| ID | Mechanism | Gate | Result | Status | Note |
|---|---|---|---|---|---|
| `RECALL-N0-POOL-CEILING-20260730` | frozen-pool recall ceiling diagnostic | 44b6 held-fit zero-FP pool-absent oracle `>= +0.030` adjusted edge Jaccard | TP/FP/FN `10,341/1,064/673` reproduced; present/absent `47/626`; pool-absent adjusted/raw/node Δ `+0.052004594313/+0.051829773141/0`; 10–12 µm absent shell `18` | **GO → N1** | Diagnostic only; no mechanism selected and no external action. |
