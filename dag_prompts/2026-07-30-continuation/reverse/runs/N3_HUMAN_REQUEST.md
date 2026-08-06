# Reverse N3 — owner decision request

**Date:** 2026-07-30
**Verdict:** **TERMINAL BEFORE COMPUTE** — the owner approved the named bounded
pilot, but source/rights preflight showed no released deterministic checkpoint
or self-contained <=2-GPU-hour inference witness. See
[`N3_SOURCE_AUDIT_TERMINAL.md`](N3_SOURCE_AUDIT_TERMINAL.md). No Kaggle asset,
kernel, data download, GPU use, campaign data access, or campaign outcome
occurred.

## Recorded decision and result

The owner approved the one bounded source-only Kaggle T4 compatibility pilot.
The required source/rights audit then stopped it before compute: the pinned
repository has no released deterministic checkpoint and requires ordered
contour-label preprocessing plus training. It cannot satisfy the declared
<=2-GPU-hour witness without replacing the mechanism or changing the budget.
Approval never extended to a campaign data read, a feature cache, a graph/scorer
run, a second family, CV57, a tier change, a submission, a GPU retry, or any
downstream request.

## New source, rights, and evidence independence

- **New source:** Jang, Lee & Kim, *Unsupervised Contour Tracking of Live Cells by
  Mechanical and Cycle Consistency Losses* ([primary paper](https://arxiv.org/abs/2303.08364));
  the authors' [`contour-tracking` repository](https://github.com/JunbongJang/contour-tracking)
  declares Apache-2.0. The authors link their source data via Zenodo record
  `7754589`.
- **Rights condition:** code license is visible; data access and record-level reuse
  license must be authenticated at step 0 from the actual approved record. If
  missing, ambiguous, restricted, or incompatible, the run stops before GPU work.
- **Distinct evidence:** for a prospective mask pair only, MCC emits target-contour
  landing residual, forward-then-backward contour return residual, and normal-motion
  compatibility. It does not emit a reverse association endpoint, a margin, or a
  null mass. N1 contains two independent primary sources and the zero-field
  counterevidence; N2 preregisters the anti-trivial fixture.

## Step 0 and proposed bounded slice

Before TensorFlow/PyTorch import, image decode, training, or GPU activation, a
pure-stdlib step 0 must authenticate the repository commit/tree, Apache license,
Zenodo record/access/license, data-file checksum manifest, absence of all campaign
paths, and absence of every forbidden Trackastra identifier. It must complete in
five seconds or fail closed.

The sole permitted slice is one named, hash-pinned, 200-frame **public source**
validation video with its author-format masks. It excludes every campaign artifact.
The hard limit is one Kaggle T4, at most two GPU-hours, at most 12 GiB RAM, and no
multi-GPU/CPU-fallback expansion. The source has no observed published checkpoint;
therefore failure to produce a deterministic source-only checkpoint/output within
this ceiling is terminal, rather than grounds to widen the job.

## Stop conditions, output receipt, and no-retry semantics

Stop immediately on any step-0/authentication/rights mismatch, dependency drift,
fixture failure, missing ordered-mask input, nonfinite/incomplete output, resource
breach, campaign-path access, or forbidden reverse identifier. The receipt may
contain only source/data/rights hashes, environment lock, resource ledger, static
fixture result, source-only contour-output manifest, and terminal status; it is
quarantined from campaign artifacts.

There is **no retry** for the exact source version, data record, slice, or config.
Any changed version, resource ceiling, data, or model is a new named mechanism and
must begin at N0 with a new owner decision.

## Explicit Reverse Trackastra prohibition list

The pilot must not retrieve, mount, load, use, compare, derive from, or reproduce:

- Trackastra `model.pt` (`24c290ce…981f`) or `ctc.zip.bin`
  (`2ede0848…a0e`), its source/interface, zero-wrfeat fields, or forward cache;
- any genuine reversed-time inference, `t_rev` packing, endpoint return
  `A_reverse[v,u]`, reverse/challenger/conflict/null/margin descriptors, or reverse
  cache/row/output; and
- campaign images, masks, Zarrs, action populations, B0/A1/A0, folds, scorer,
  graph compiler, 44b6, 6bba, CV57, Kaggle submission, RunPod, or shared ledger/live
  state.

## Proposed ledger row — not written

`REVERSE-MCC-N3 | HUMAN-DECISION | approve or reject one 2-GPU-hour, public-source
only MCC provenance/compatibility pilot; all terminal Reverse Trackastra assets and
all campaign data/scoring prohibited | no action pending owner.`

**Cascade:** halted at the human gate. No asset, kernel, upload, download, launch,
submission, or push is authorized by this document.
