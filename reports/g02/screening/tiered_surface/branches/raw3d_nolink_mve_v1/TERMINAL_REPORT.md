# RAW3D-NOLINK-MVE Stage 0 — terminal report

## Verdict

**PARK-C raw-3D and consolidate the campaign at the banked 0.881 result.**

Stage 0 completed on CPU in `254.704 s` over all 41 frozen DELETE videos,
`1,345,497` actions, and `1,118` beneficial actions. The crude raw-3D supplier
improved mean AUC over DINOv2-XY, but it failed both clauses of the
pre-outcome advance rule and remains far below the measured AUC/retention
surface. Stage 1 is closed. No acquisition, pretrained asset gate, training,
GPU, graph, 6bba, CV57, or submission is authorized.

## Frozen design and integrity

- Preregistration: `5b76da90…8588dd`, frozen before any raw-3D outcome.
- Sealed B0: `a9ce8c0…b4ff2`, opened first by contract.
- Frozen five-fold class-balanced ridge probe: `548b681…30ca`, `lambda=0.01`.
- Raw archive: banked full SHA `538ed8a4…dfd19`; byte size and canonical
  central-directory SHA `5edffe55…cdc77` passed at Step 0.
- All 41 request and population byte-SHAs passed. Step 0 took `0.171 s`.
- The label-blind extractor streamed one frame/video at a time and used no
  weights, pseudo-masks, or pretrained assets. It emitted the fixed eight
  action summaries after 35 crude 3-D endpoint statistics.
- All 41 temporary feature shards were hashed before labels opened, then
  deleted after evaluation. No voxel, crop, endpoint, or full-pool cache was
  retained.

## Results

| held fold | A1 AUC | A1 CG0-budget retention |
|---:|---:|---:|
| 0 | `0.726453902` | `0.003267974` |
| 1 | `0.775858844` | `0.010101010` |
| 2 | `0.599374874` | `0.011976048` |
| 3 | `0.646261214` | `0.005128205` |
| 4 | `0.641887210` | `0.000000000` |
| mean | **`0.677967209`** | **`0.006094647`** |
| fold floor | **`0.599374874`** | **`0.000000000`** |

The full-population operating point retained `5/1,118 = 0.004472272`
beneficial actions while consuming 261 harmful-action budget units. The A0
permutation-control mean AUC was `0.646573736`; A1−A0 was `+0.031393472`.
That is evidence that raw 3-D contains some genuine endpoint-pair information,
but not enough at the required tail or across held-video folds.

Against references:

- DINOv2-XY mean AUC `0.670411743`: raw-3D improves by `+0.007555465`.
- Frozen material-improvement line `0.680411743`: raw-3D mean misses by
  `0.002444535`.
- The same binding fold-floor line `0.680411743`: raw-3D misses by
  `0.081036869`.
- Measured requirement AUC `0.85`: raw-3D misses by `0.172032791`.
- Measured mean retention `0.020393560`: raw-3D misses by `0.014298913`.

Both preregistered advance checks are false. The strong folds 0/1 do not
rescue the fold-2/3/4 transfer failure, and retention still captures only five
beneficial actions at the exact CG0 budget.

## Decision and barrier

`RAW3D-NOLINK-MVE` moves to PARK-C. The result does not claim raw intensities
contain zero information; it says this crude, zero-weight realization fails the
minimum robust improvement required to justify asset acquisition. Per the
human decision, Stage 1 does not open and the campaign consolidates at
APNCC-DIRECT local `0.881522069960` / public `0.881`.

The existing `bhap-submit` version 1 (`submission ref 54942864`, package
`779aa8d3…606b`) is explicitly selected as `1/2` competition finals. The
deadline is `2026-09-29 23:59 UTC`. This was selection bookkeeping only: no new
submission was made and no slot or GPU quota was consumed.

The frozen result's field `resource.max_rss_kib` is mislabeled on Darwin;
`ru_maxrss=2,894,741,504` means bytes (`2760.641 MiB`), not KiB. The separate
resource note corrects only that non-scientific interpretation; the frozen
result and scientific verdict are unchanged.

**HALT.** Trackastra fine-tuning remains unrecommended and unauthorized; there
is no Stage-1 asset search or Stage-3 training follow-on.
