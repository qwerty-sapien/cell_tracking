# Lane 2 trained fork detector — terminal ceiling-first report

Date: 2026-08-03  
Verdict: **FAIL CROSS-FAMILY CEILING / STOP BEFORE FINAL 38 / NO GPU / NO GRAPH**

This lane is not a re-screen of the earlier `0/38` realizer. That result used
hand-crafted geometry, persistence, and decode rules; it never supplied an
authentic pretrained or trained raw-intensity fork input. Here the first model
is genuinely trained on raw `3-D+t` voxel crops.

The single preregistered CPU probe was a class-balanced L2 logistic model
(`C=0.01`, liblinear, seed `20260803`) over flattened normalized
`3x8x16x16` intensity crops. Its first fit used 66 positive and 264 negative
samples from 69 6bba identities. Cross-family validation was present from that
first fit and used 9 positives plus 36 negatives from 36 held-fit 44b6
identities. Fourteen train negatives and sixteen validation negatives required
the preregistered far-from-sparse-GT fallback; this weak-negative content is a
limitation, not hidden authentication.

At threshold `0.5`, validation was TP/FP/FN/TN `5/6/4/30`, precision
`0.454545`, recall `0.555556`, ROC AUC `0.777778`, and average precision
`0.570905`. The required operating class was recall at least `25/38 =
0.657895` with precision at least `25/35 = 0.714286`. The best attainable
precision after reaching the recall requirement was only `0.500000` (TP/FP
`7/7`, recall `0.777778`, threshold `0.2911959`). No tie-safe threshold passed.

The preregistered kill line therefore fired. The valid V2 final population
seal remains unopened by crop extraction or model scoring: `final_38_access`
is false. No compact CNN, final direct score, graph integration, hybrid,
Kaggle kernel, GPU hour, submission, or RunPod action followed. The additive
value schedule (`+0.0595/+0.0532/+0.0481` for TP25 at FP `0/5/10`) remains
descriptive only because this detector did not establish the required input
operating point.

Primary literature gives mechanism evidence but not a transferable target
effect. The Linajea C. elegans extension reports per 1000 GT edges that its
candidate-selection/classifier/structured-SVM system changes FP-division and
FN-division from `0.58/1.2` to `0.053/0.40` on mskcc and from `0.66/1.6` to
`0.20/0.49` on nih-ls; total errors change `24 -> 9.6` and `45 -> 20`.
That is C. elegans, not target zebrafish, and the entire structured interface
differs. The original zebrafish Linajea system is same-domain whole-embryo
lineage reconstruction, but its seven-frame dense movement interface and
400,000-iteration budget do not quantify this fork detector's target effect.

Authoritative result: `run_v1/CROSS_FAMILY_RESULT.json`. Model SHA-256:
`a45a749ebdfab27748b732e34171e1cad8c6568aebd1229db68be39155bc3057`.
Development crop SHA-256:
`df6ac997c0c370f2d090ca5e5d8982ec282ac4741a525d3ea80735a0ded98a5e`.
