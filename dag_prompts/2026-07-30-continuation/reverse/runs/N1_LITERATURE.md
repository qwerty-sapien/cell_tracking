# Reverse N1 — future-evidence literature audit

**Date:** 2026-07-30
**Input:** `N0_SCOPE.md` (MCC contour-return witness)
**Verdict:** **GO** — two independent primary sources support the *specific*
round-trip/re-localization observable. This is not evidence of campaign gain.

## Primary evidence

1. **Jang, Lee & Kim, “Unsupervised Contour Tracking of Live Cells by Mechanical
   and Cycle Consistency Losses,” CVPR 2023** — [primary paper](https://arxiv.org/abs/2303.08364),
   [open full text](https://ar5iv.labs.arxiv.org/html/2303.08364), and
   [authors' Apache-2.0 source](https://github.com/JunbongJang/contour-tracking).
   This is directly on phase-contrast and confocal live-cell videos. Its two-frame
   mechanism predicts contour-point offsets, snaps the forward point to the target
   contour, then uses a backward offset to calculate return consistency; it combines
   that loss with an offset-versus-contour-normal loss. Its explicit anti-shortcut is
   material: without the snap-to-target-contour step, a zero forward/backward field
   would minimize the cycle loss. On the phase-contrast evaluation, their combined
   method reports spatial accuracies `0.729/0.937/0.974` at tolerances
   `.02/.04/.06` and contour accuracies `0.762/0.925/0.971` at `.01/.02/.03`,
   exceeding UFlow, PoST, and the mechanical baseline in that table. It also leads
   each reported confocal threshold, though lower overall. This supports a
   *learned contour landing + return + normal-motion witness*, not generic
   bidirectional graph scoring.

2. **Wang, Jabri & Efros, “Learning Correspondence from the Cycle-Consistency of
   Time,” CVPR 2019** — [primary paper](https://arxiv.org/abs/1903.07593),
   [CVF record](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Correspondence_From_the_Cycle-Consistency_of_Time_CVPR_2019_paper.html),
   and [authors' code/model record](https://github.com/xiaolonw/TimeCycle).
   On unlabeled video it learns a feature map by requiring a tracked patch to return
   to its start through a time cycle, then uses nearest-neighbour correspondence at
   test time. The primary paper reports generalization without fine-tuning to video
   object segmentation, keypoints, and optical flow, outperforming earlier
   self-supervised work and competing with supervised methods. This is a structural
   analogue, not a generic reverse-tracking citation: it supplies the required
   *return-to-origin plus re-localization* principle and independently notes the
   stationary/identity shortcut that N2 must falsify.

The sources are independent (different author groups, codebases, data and domains).
The first establishes the observable on live-cell contours; the second establishes
why a temporal return constraint can provide correspondence evidence without
labels. Neither uses Trackastra, its checkpoint, zero-wrfeat interface, a reversed
time axis, an endpoint matrix, or this campaign's cache.

## Assumptions and counterevidence

- Jang et al. study 2-D phase-contrast/confocal cell contours, with all frames
  segmented and border-touching points excluded; the campaign may violate these
  conditions (3-D geometry, crowded divisions, imperfect masks, occlusion).
- Their reported source training is one day on a TITAN RTX and the public repository
  exposes source/data links but no published release. A checkpoint, exact data
  rights, dependency pinning, and resource feasibility are therefore unresolved;
  this node did not retrieve them.
- The paper states that natural forward and backward cell movement differ. Thus the
  return residual must be a *soft diagnostic* and can never be a hard no-link veto,
  especially around division or topology change.
- Cycle consistency alone is not identity proof. Both the contour paper's snap step
  and TimeCycle's re-localization safeguard are evidence that zero/identity flow is
  a real degeneracy. N2 must kill that null analytically before any model or data
  acquisition.
- Literature does not validate transfer to the frozen graph objective. Per the
  campaign plan, accuracy, contour accuracy, and a feature score are report-only;
  a later allowed candidate would still need the 44b6-first graph delta and all
  unchanged floors.

## Independence statement

The proposed score is computed only from an independently sourced contour model's
forward/backward offset fields and the two candidate masks: target landing,
round-trip return, and normal alignment. It has no access to forward margin,
challenger/conflict identities, `A_reverse[v,u]`, reverse null mass, reverse cache,
or a Trackastra derived descriptor. It would remain defined if the terminal
Trackastra artifacts had never existed.

## Proposed ledger row — not written

`REVERSE-MCC-LIT | GO_TO_STATIC_FALSIFIER | two primary sources: live-cell
mechanical+cycle contour correspondence and independent unlabeled-video
re-localization; known zero-flow/domain-shift counterevidence retained | no model,
asset, data, scorer, or graph action.`

**Cascade:** N1 GO → N2 static falsifier.
