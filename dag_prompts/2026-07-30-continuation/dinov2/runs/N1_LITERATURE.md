# N1 — literature grounding: DPRC-O0

Date: 2026-07-30 (Asia/Singapore)
Verdict: **GO — with the stated non-transfer claims**

## Primary evidence

1. Tumanyan, Singer, Bagon & Dekel, *DINO-Tracker: Taming DINO for
   Self-Supervised Point Tracking in a Single Video*, ECCV 2024,
   [DOI: 10.1007/978-3-031-73347-5_21](https://doi.org/10.1007/978-3-031-73347-5_21)
   ([author paper](https://dino-tracker.github.io/assets/dino_tracker.pdf)).
   This primary tracking study uses localized DINOv2 features as the semantic
   prior for point correspondence, including occlusions. It supports the
   structural part of DPRC: local DINO tokens can carry correspondence
   information rather than only pooled identity.

2. Moutakanni *et al.*, *Cell-DINO: Self-supervised image-based embeddings for
   cell fluorescent microscopy*, PLOS Computational Biology 2025,
   [DOI: 10.1371/journal.pcbi.1013828](https://doi.org/10.1371/journal.pcbi.1013828).
   This primary microscopy study applies the DINOv2 training strategy to
   fluorescent cell images and reports morphology-rich single-cell
   representations. It supports the microscopy-side premise that a DINO-family
   representation can preserve cell morphology relevant to association.

3. Wu *et al.*, *DynaMorph: self-supervised learning of morphodynamic states of
   live cells*, Molecular Biology of the Cell 2022,
   [DOI: 10.1091/mbc.e21-11-0561](https://doi.org/10.1091/mbc.e21-11-0561).
   In live-cell time-lapse microscopy, a self-supervised morphodynamic
   representation kept stable-morphology trajectories localized while exposing
   morphology-changing trajectories. This is the explicit live-microscopy
   analogue for treating local appearance continuity as an association signal.

These are independent primary studies: one establishes DINOv2-localized tracking
as a structural analogue; two establish that self-supervised cell morphology is
meaningful, including in live-cell time lapse. They do **not** establish a G-02
score, or transfer to its imaging modality.

## Assumptions, transfer argument, and negative evidence

The proposed supplier is the official DINOv2 ViT-S/14 register variant, not the
terminal R5 non-register checkpoint. The later preflight must prove its source,
weight hash, license, input channel contract, patch-grid shape, and that its crop
normalization is independently materialized. It cannot borrow any R5 asset or
claim compatibility merely because it shares the DINO family.

The transfer argument is structural, not a performance extrapolation: each G-02
candidate edge compares two segmented, cell-centred observations; a true adjacent
edge should preserve more spatially localized foreground morphology than a
globally similar wrong neighbour. The proposed mutual local match records that
property, and the later frozen evaluator—not its proposal metric—must decide
whether it helps delete or occupied-swap.

Most importantly, DINO-Tracker reports that raw DINO matching lacks temporal
consistency and fine localization in ambiguous regions; its gains require
per-video learned refinement. DPRC expressly does **not** import that training
route. This negative result is why N2 can only validate the contract, not infer
tracking value. Cell-DINO likewise uses microscopy-specific training/data, so it
does not license a claim that an unvalidated channel mapping will transfer.

## Mapping to the frozen action classes

- **Delete:** low `C(i,j)` for the incumbent means it is a candidate for the
  frozen fit-only evaluator to separate from retained links.
- **Occupied swap:** within the same source/target conflict, the signed
  `C(i, alternative) - C(i, incumbent)` can distinguish an alternative that
  preserves local foreground structure. No hand-tuned threshold, graph-cost
  change, or GT-aware conflict rule is proposed.

## Proposed ledger row — not written

`DPRC-O0 | N1 GO | primary DINO-localized-tracking + cell-microscopy +
live-microscopy structural support; raw-match and channel-transfer limitations
preregistered | N2 fixture only`

**Cascade:** GO to N2.
