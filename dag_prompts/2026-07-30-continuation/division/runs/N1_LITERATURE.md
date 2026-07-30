# Division N1 — primary evidence audit

**Verdict: GO.** Two independent primary studies support the narrow causal claim that raw time-resolved image representations contain division-relevant information absent from a proposal/action table. They do **not** establish a gain on the frozen G-02 graph objective.

## Candidate under audit

`TAP-SYM-DIV` uses a frozen, label-free time-arrow dense representation of local raw parent/daughter patches and a parameter-free daughter-reflection relation. It will not consume oracle continuity labels or a current-pool score. Any future decoder or action step remains locked behind a human amendment.

## Independent primary evidence

1. Gallusser, Stieber & Weigert, 2023, **MICCAI**, [Self-supervised Dense Representation Learning for Live-Cell Microscopy with Time Arrow Prediction](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_52) (also [open preprint](https://arxiv.org/abs/2305.05511)). The primary study trains from raw, unlabeled live-cell movies to predict temporal direction, finds dense features that localize time-asymmetric processes including division, and tests those features on mitosis detection/segmentation. A small classifier on fixed representations outperformed its supervised baseline in the low-label mitosis experiment. This directly supports the time-arrow half of the candidate.

2. Gilad *et al.*, 2019, **Bioinformatics**, [Fully unsupervised symmetry-based mitosis detection in time-lapse cell microscopy](https://pmc.ncbi.nlm.nih.gov/articles/PMC6662301/), [DOI](https://doi.org/10.1093/bioinformatics/bty1034). This primary study obtains mitotic candidates from local spatio-temporal patch changes, then combines reflected daughter similarity with mother-to-daughter-pair similarity for mother/two-daughter association. It reports application to 2D and 3D Cell Tracking Challenge fluorescence sequences without labelled training. This independently supports the daughter-symmetry half of the candidate.

The first method is learned self-supervised dense representation learning; the second is an unsupervised raw-patch/symmetry mechanism. Their agreement is evidence for a new observation channel, not for an untested graph improvement.

## Applicability and counterevidence

| Question | Finding | Consequence |
|---|---|---|
| Division evidence | Both sources show raw local temporal appearance has division signal. | Supports evidence-only representation. |
| Lineage association | Gilad *et al.* form mother--daughter-pair triplets; TAP itself stops at representation/detection. | It cannot bypass association-continuity prerequisites. |
| Modality | TAP's released contract is 2D+time. Gilad includes 3D but assumes divisions in XY. | Future work must preregister XY-plane extraction; no 3D portability claim now. |
| Biology/scene | Gilad assumes mostly symmetric divisions and reports clustered/asymmetric limits. | Daughter coherence is evidence, not a universal selection rule. |
| Compute | TAP documents about 8 GPU-hours/model and a default >=24 GB NVIDIA GPU configuration. | N2 must treat this as a binding feasibility fact. |

The public [TAP repository](https://github.com/weigertlab/tarrow) is BSD-3-Clause, but no source revision, dependency, or checkpoint has been authenticated or acquired here. That keeps this literature audit separate from implementation.

## Surviving causal claim and proposed ledger row

If a parent-to-two-daughters transition has an irreversible visual change and two coherent daughter appearances, `TAP-SYM-DIV` can observe information unavailable to the current action-pool selector. It might distinguish a true division-like triplet from a geometrically similar non-division triplet. It does **not** claim to recover the 38 oracle bundles, predict adjusted edge Jaccard, or provide a legal action.

`TAP-SYM-DIV | N1 GO | Gallusser 2023 time-arrow representation + Gilad et al. 2019 mother/two-daughter raw-patch evidence | 2D/3D and symmetry limits retained | no fit/no score/no tier move | next: non-fitting falsifier`.

This row is proposed only; `LEDGER.md` was not edited.

**Cascade:** GO to N2.
