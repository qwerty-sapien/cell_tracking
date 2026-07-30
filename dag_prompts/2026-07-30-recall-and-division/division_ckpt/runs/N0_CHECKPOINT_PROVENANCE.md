# Division-ckpt N0 — frozen external checkpoint provenance

**Verdict: PARK — missing authenticated provenance and 16-GiB feasibility record.**

The preregistered `GO` threshold is **at least one** released checkpoint with all five
required facts: a source revision, weight bytes plus SHA-256, a research-permitting
license, a documented corpus excluding every competition movie, and a
source-published inference configuration fitting a 16-GiB T4. This audit establishes
**0 / >=1 qualifying checkpoints**. It does not establish that no release exists, so
it does not make the stronger `DEAD` claim. It also does not substitute a training
configuration, a different backbone, or a proxy memory estimate for the missing
inference configuration.

## Scope and immutable stops

- Read: the parent division DAG, prior division N0/N1/N2/N3 records, current live
  state, and `reports/g02/screening/B1_division_tp.md`.
- No campaign frame, fragment, request, row, action population, repair bundle, model,
  scorer, GPU, Kaggle asset/kernel/run, remote push, submission, or RunPod action was
  read, created, or used. GPU debit: **0 h**.
- No lineage from `TAP-SYM-DIV`'s escalated five-fold design, R5, or CELLECT was
  reused: **0** artifacts/fragments/requests/rows/launches.
- `B1` remains `PARK`: **0** association-continuity arms have passed the required
  both-family condition. Division fitting remains locked. These are independent
  downstream locks; this node did not attempt to alter either.

## Candidate-release audit

The table records every candidate/release lead available in the pinned campaign
evidence. A row without every pinned field is deliberately *not* counted as a
candidate for the `GO` numerator.

| Lead examined | Source / weight identity | Pin and provenance result | Rights and 16-GiB result | Disposition |
|---|---|---|---|---|
| Official TAP / `tarrow` source | <https://github.com/weigertlab/tarrow>; primary study: Gallusser, Stieber & Weigert (2023), DOI `10.1007/978-3-031-43993-3_52`, preprint <https://arxiv.org/abs/2305.05511> | Prior N1 records the public source as BSD-3-Clause but records **no authenticated source revision, dependency, or checkpoint**. No released weight URL, byte count, SHA-256, constructor/config pin, or training-corpus exclusion was authenticated in this run. | The historical source record is a code-license assertion, not a weight-license receipt: license-file bytes/SHA-256, weight research/embedding/retention terms are **not established**. The only documented compute fact is the upstream default of `>=24 GiB` for training and about `8 GPU-h` per model; it is not a source-published 16-GiB inference configuration. | Not eligible; cannot count toward GO. |
| Independently released weight from the TAP primary-literature lead | Same primary study above; no separately pinned release artifact is present in the campaign evidence | No independent weight URL, revision, byte size, SHA-256, load constructor, corpus record, or exclusion proof was authenticated. | No weight license/derived-embedding/retention terms or 16-GiB inference configuration were authenticated. | Not eligible; cannot count toward GO. |
| Daughter-symmetry literature lead | Gilad *et al.* (2019), DOI `10.1093/bioinformatics/bty1034`, <https://pmc.ncbi.nlm.nih.gov/articles/PMC6662301/> | This supports the non-learned daughter-symmetry rationale only. No frozen temporal-representation checkpoint, source/weight pin, corpus exclusion, or loader identity was authenticated. | No checkpoint rights or 16-GiB source-published inference configuration was authenticated. | Not an admissible frozen-external checkpoint. |

`R5`, CELLECT, and the prior `TAP-SYM-DIV` training design were not candidates in this
table: admitting any of them would reuse a terminal/existing route or turn this new
frozen-external mechanism into a different mechanism.

## Authentication limitation and exact failed prerequisites

N0 requires authenticated source/weight/license retrieval through computer-use cmux.
The read-only cmux inspection on 2026-07-30 exposed an accessible pane labelled
`Login or Register | Kaggle`, rather than an available authenticated retrieval
session; the inspection itself was unresponsive for the initial capture and was not
retried. No remote command or download was issued. Consequently the following exact
required values remain **unmeasured**, not inferred:

| Required N0 field | Verified count / value |
|---|---:|
| Pinned source revisions | 0 |
| Pinned weight objects with byte count and SHA-256 | 0 |
| License files with SHA-256 for a weight | 0 |
| Documented training corpora proven to exclude all competition movies | 0 |
| Source-published frozen inference configurations proven to fit 16 GiB | 0 |
| Qualifying released checkpoints | 0 |

The existing prior records support only these non-substitutable context facts:
`tarrow` is a public source lead reported as BSD-3-Clause; its upstream default is
`>=24 GiB` **for training**; and its planning estimate is about `8 GPU-h` per model.
None supplies the exact frozen weight, corpus-exclusion proof, or 16-GiB
inference-only configuration required by this node.

## Preregistered verdict application

`GO` is false at **0 / >=1** qualifying checkpoints. `DEAD` is not asserted because
the required owner-session audit could not establish the universal claim that no
released checkpoint exists or that every candidate trained on competition data.
`ESCALATE` is not applicable: no admissible checkpoint and no term requiring a plan or
governance change was established. Under N0's fixed alternatives, the missing
rights/corpus/16-GiB documentation produces **PARK**. The lane halts here; N1's
synthetic fixture, N2's human Kaggle protocol, and N3's efficacy gate were not opened.

## Proposed LEDGER row — not written

`FROZEN-EXTERNAL-DIV-CKPT | N0 PARK | 0/>=1 qualifying checkpoints; no authenticated source revision, weight bytes/SHA-256, weight-license receipt, corpus-exclusion proof, or source-published 16-GiB inference configuration; no external/campaign action | next: owner decides whether to leave parked or authorize a newly scoped authenticated provenance review | no tier move`.

## Owner decision required

The owner must decide whether to leave this frozen-checkpoint pilot parked, or later
authorize a **newly scoped** provenance review when an authenticated cmux retrieval
session is available. Such a review must still meet N0's original five evidence
requirements before N1 can open; it is not authority to train TAP, change a threshold,
read campaign data, use Kaggle, or bypass the separate division-fitting and B1 locks.
