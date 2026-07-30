# Reverse N2 — Kaggle-first source-authentication protocol (draft only)

**Status:** Draft for N3 owner decision. This document authorizes nothing and has
not created, downloaded, uploaded, launched, submitted, or pushed anything.

## Narrow proposed test

Run **one** source-only MCC compatibility pilot on a Kaggle T4 after owner approval.
It is strictly a reproducibility/authentication test on the authors' public source
data—not campaign video, detection masks, rows, actions, scorer, graph, 44b6,
6bba, CV57, or a submission. It cannot produce a campaign feature cache or an
association outcome.

### Source and rights

- Code: Jang–Lee–Kim's public
  [`JunbongJang/contour-tracking`](https://github.com/JunbongJang/contour-tracking)
  repository; its displayed license is Apache-2.0.
- Data: the authors link their public data from the repository (Zenodo record
  `7754589`). This run must treat the dataset's own record-level access and license
  metadata as unresolved until step 0 authenticates them. Public availability is
  not a substitute for a verified reuse right.
- Model: no released MCC checkpoint was observed in the public repository. The pilot
  may not silently substitute a Trackastra, DINOv2, CELLECT, or generic optical-flow
  weight. A source-trained checkpoint, if the owner explicitly permits it, is a new
  asset and remains source-only/quarantined.

## Mandatory step 0 — before framework import, data decode, training, or GPU use

Pure-stdlib code must, within five seconds:

1. verify a pinned repository commit/tree hash, Apache-2.0 `LICENSE` hash, and
   source manifest;
2. verify the Zenodo record ID, record access state, explicit data-license string,
   file manifest, and each supplied file checksum against an owner-approved
   manifest;
3. reject absent/ambiguous/restricted/non-reusable rights, a changed file, an
   absent ordered-mask input, any Trackastra identifier/output, or any mounted
   campaign artifact; and
4. emit one compact `STEP0.json` and exit without a GPU import on failure.

The owner must stage any then-approved source bundle privately with its manifest;
this agent will not create that bundle. Source authenticity/rights failure consumes
**0 GPU-h** and ends the one permitted attempt.

## Exact slice, ceiling, quarantine, and no-retry rule

- Slice: one named, hash-pinned 200-frame public source validation video at its
  documented resolution, with author-format images and segmentation masks only.
- Ceiling: one T4, `<=2.0` GPU-h wall-clock and `<=12 GiB` RAM; no multi-GPU,
  CPU fallback, public submission, or external storage. The source's reported
  day-long training is not reproduced; a failure to establish a bounded pilot is
  a stop, not permission to enlarge it.
- Stop conditions: step 0 failure; dependency/source mismatch; no deterministic
  checkpoint within the ceiling; any GPU/RAM budget breach; any MCC fixture
  failure; any campaign path or forbidden reverse identifier; or a nonfinite/
  incomplete output.
- Output: source commit/tree and license hashes, rights/record receipt, input
  checksums, environment lock, resource ledger, static-fixture result, one
  source-only contour output manifest, and a quarantined terminal receipt. No
  campaign prediction, graph action, edge row, or score is permitted.
- No retry: any stop is terminal for this exact source/version/slice/configuration.
  A changed source, data release, model, budget, or slice is a new named mechanism
  requiring a fresh N0–N3 sequence and owner decision.

## Later decision boundary

A clean source-only receipt proves only that the independently sourced observable
can be materialized under its rights and resource cap. It does **not** approve a
campaign run. Any request to use campaign masks/videos or a graph gate must return
to a new human-reviewed DAG with a source checksum, new data-rights proof,
predeclared integration, and the frozen objective-level test.
