# CELLECT-R7 authenticated-weight separation v3 — terminal report

Status: **INVALID / infrastructure failure before checkpoint load and supplier features**.

## Frozen chronology

The supplied `U-ext+-x3rd-149.0-4.6540.pth` is byte-identical to the checkpoint tracked at
official CELLECT commit `3586070926f7f1fd5d8df37456861d22bdc63236`:

- checkpoint SHA-256:
  `f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5`;
- Git blob SHA-1: `2523c69af7f29c94f365a7b8af3bcba4b6d9b67a`;
- source: `https://github.com/zzz333za/CELLECT`;
- paper DOI: `10.1038/s41592-025-02886-x`;
- checkpoint distribution license: GPL-2.0-only under the repository's top-level license.

Supplier contract v2 (`8d34445e…d3664`) requires a hash check before load and a
tensor-only `torch.load(..., weights_only=True)` path with no unsafe pickle fallback.
The guarded local authentication passed strict state/keyset loading with 2,541,972
parameters, no forward pass, no training, and zero swap growth.

The per-supplier x action-class preregistration was re-frozen at
`9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e`
before any v3 feature, B0, A1, A0, or verdict output. It records that the exact population
and labels were already banked under the earlier INVALID attempts and were not regenerated
or changed. Both-class reporting remains mandatory; cross-class conjunction remains absent.

Private Kaggle asset version 3 reported all seven pinned uploads successful and reached
`ready` at `2026-07-28T07:13:49.317Z`. The exact both-class B0 was then sealed, before
A1/A0, at
`a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2`.

## Terminal execution

Exactly one private T4 kernel version 3 was launched. Its offline Zarr dependencies
installed successfully. Before checkpoint load, feature computation, or A1/A0, it stopped
with:

`FeatureError: expected one U-ext+-x3rd-149.0-4.6540.pth, found 0`

The terminal log is
`run_v3/kaggle_output/g02-cellect-r7-separation-v1.log`
(`7e9ba2c97c0e84353dc9479eb05d568384ffe0759a2d2207c2d6f25aa6e06177`).
No feature output was emitted. The metered log runtime was 656.776 seconds; the conservative
debit is 0.1825 kernel-hours.

## Verdict and barrier

There is **no scientific PASS or FAIL**. No A1/A0 metric or per-class verdict ran, so the
frozen INVALID semantics prohibit a PARK-C demotion. CELLECT R7 remains PARK-B.

The authorization is exhausted. No additional retry, training, full-pool cache, graph,
6bba, CV57, submission, RunPod action, DEAD promotion, plan amendment, or tier move is
authorized. `uncited_tier_moves` remains zero.
