# Recall N1 — mechanism and crowding

**Verdict: GO.** `NDR-FBC-ONE-OFFER` is a label-free, non-DEAD candidate-supply
mechanism with a fixed opportunity count and three primary sources. It is not a
re-ranker, threshold, graph-cost change, or reinstatement of H1’s dead zero-wrfeat
supplier. No model was fitted; no GT was read by the opportunity-count run; no GPU,
Kaggle, asset, checkpoint download, remote push, submission, or ledger write occurred.

## Information requirement from N0

N0 found that **626/673** 44b6 held FNs are absent from the frozen pool, for a zero-FP
adjusted ceiling of `+0.052004594313`; merely selecting old candidates cannot reach
them. A supply mechanism must therefore observe, without GT at decision time, a
cross-frame signal that predicts a target centroid outside the static 10 µm/cap-8
candidate set while limiting offers far below the `5,342,071` shell-background pairs.
The 84 pool-absent FNs with an unmatched prediction endpoint further mean that the
method may not assume a complete selected graph.

## Generated candidates and hard-disqualifier screen

| Candidate | Novelty generator | Disqualifier result | Crowd result |
|---|---|---|---|
| **NDR-FBC-ONE-OFFER** — raw-volume, local nonrigid 3D registration; transport each unlinked source by the forward field, require reverse-cycle agreement, then offer its nearest next-frame detection | Cross-disciplinary transfer (diffeomorphic microscopy registration) + literature-edge mining | **SURVIVES.** Raw intensities/deformations generate a new correspondence before any graph score; no GT, frozen-pool score, threshold, H1 feature, or hand-tuned graph cost is used at decision time. CPU-only N2 falsifier; no immediate GPU envelope. | Exact move estimated **5–20 / 1,000** competent teams. Generic optical flow is beta; the defensible narrowing is *diffeomorphic 3D cell registration → out-of-pool offer only at currently unlinked sources → inverse-field certificate*, rather than a candidate ranker. |
| Entropic point-set transport from centroid clouds | Constraint inversion | **REJECT.** With only centroid geometry/mass costs it is a renamed hand-tuned geometry/motion mechanism (`MGSHELL`/`TEMPBG` family), which is DEAD. | Not ranked. |
| Fit-only self-supervised temporal correspondence embedding | Literature-edge mining | **REJECT.** A learned correspondence score trained from scratch is a from-scratch ranker in the `POINTCTX`/`CROPLINK`/`TEMPRANK` DEAD class. | Not ranked. |
| 12 µm/cap-8 shell expansion | Boundary reasoning | **REJECT.** Exact `R12SHELL` re-screen; DEAD. | Not ranked. |
| New detection/segmentation proposals | Constraint inversion | **REJECT for this lane.** Detection front-end is Conditional-PARK and needs a locked-plan amendment; it cannot be substituted here. | Not ranked. |

## Primary grounding for the survivor

1. Kim, Chen, Spector, Eils & Rohr, *Nonrigid Registration of 2-D and 3-D Dynamic
   Cell Nuclei Images for Improved Classification of Subcellular Particle Motion*,
   IEEE TIP (2011), DOI [`10.1109/TIP.2010.2076377`](https://doi.org/10.1109/TIP.2010.2076377).
   This is primary 3D dynamic-nuclei evidence that an intensity-based nonrigid
   registration can separate deformation from observed particle motion. It does **not**
   establish recall, false-positive control, or transfer to G-02 embryos.
2. Jiang *et al.*, *Fine-grained, nonlinear registration of live cell movies reveals
   spatiotemporal organization of diffuse molecular processes*, PLOS Computational
   Biology (2022), DOI [`10.1371/journal.pcbi.1009667`](https://doi.org/10.1371/journal.pcbi.1009667).
   This is primary live-cell evidence for topology-regularized/diffeomorphic
   registration as a means to retain location through cell deformation. It does **not**
   validate nucleus association, the G-02 scorer, or a background rate.
3. Jeong, Lin, Porikli & Kwak, *Imposing Consistency for Optical Flow Estimation*,
   CVPR (2022), pp. 3181–3191, [official proceedings record](https://openaccess.thecvf.com/content/CVPR2022/html/Jeong_Imposing_Consistency_for_Optical_Flow_Estimation_CVPR_2022_paper.html).
   This primary method paper establishes forward/backward/occlusion consistency as a
   label-free certificate component. It is natural-image optical flow, not 3D microscopy,
   so it supplies no G-02 efficacy claim.

## Fixed label-free opportunity count

The N2 mechanism contract is fixed before N2: for every nonterminal baseline node with
no selected outgoing edge, compute a local raw-volume forward displacement and its
reverse certificate, then make **one** correspondence offer. Existing-pool duplicates
will be discarded; the following count is therefore a hard maximum on unique new
candidates, not an outcome-selected support.

| Item | Value |
|---|---:|
| Frozen label-free fit117 videos / nodes / selected edges | 117 / 2,878,849 / 2,762,399 |
| Eligible unlinked, nonterminal sources (= offers) | **105,992** |
| 44b6 / 6bba offers | 49,553 / 56,439 |
| Current fit117 pool | 7,874,673 |
| Maximum pool multiplier | **1.013459860492×** |
| Maximum increment | `1.3459860492%` of current pool; `1.9840994251%` of the 5,342,071 shell-background reference |

This count used no labels or model fitting. It is the required N1 opportunity count;
N2, not N1, will determine the actual reverse-certified new-candidate count and its
recovery rate on a frozen fit-only slice.

## Opportunity-count integrity and local guard

- H1 fit105 source report SHA-256: `d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf`.
- Fit117 manifest SHA-256: `20ef8056bea8afb60f76cf99357c7b62215b2db6c408d78a1c3ba21c41431a33`;
  authenticated 117-fragment combined receipt:
  `04e2eae699691d47992ba3250c4edef51e063b82e8a9eaea5fac9fc3778f61e5`.
- Fresh preflight: **PASS_ZERO_SWAP_GROWTH**, `45.488562 s`, swap growth `0 MiB`,
  receipt `b3c25985264c318b4206f0b2d9a64ed1e96c154987fae700c491b8a758d5aae2`.
- Guarded count: **PASS**, `29.096102 s`, peak RSS `0.042603 GiB`, swap growth
  `0 MiB`, receipt `fc267f18c13bd0c90690889a1a0d9ba0c8b6ec032fcc04579690456cfbbeb262`.

## Ranked survivor

| Rank | Mechanism | Pertinence | Feasibility | Defensibility |
|---:|---|---|---|---|
| 1 | `NDR-FBC-ONE-OFFER` | High: directly expands the 626-edge absent channel | Medium: CPU-only falsifier first; no asset or GPU authorization | High enough for a falsifier: non-obvious 3D registration + inverse-field certificate, primary-grounded |

## Proposed LEDGER row — do not write

| ID | Mechanism | Gate | Result | Status | Note |
|---|---|---|---|---|---|
| `RECALL-N1-NDR-FBC-20260730` | `NDR-FBC-ONE-OFFER` raw-volume, forward/backward-consistent 3D registration candidate supplier | label-free, non-DEAD, non-selector, >=2 primary sources, named defensibility and stated opportunity count | 3 primary sources; 105,992 fixed fit117 offers; maximum `1.013459860492×` pool | **GO → N2** | N2 must measure recovery and actual union size on a frozen fit-only slice; no efficacy claim yet. |
