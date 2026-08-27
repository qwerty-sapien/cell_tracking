# Division requirement surface — post-outcome measurement

Decision: **park division event detection; bank daughter assignment; redirect
near-term mechanism effort to ordinary association (+0.061628263275 Shapley
ceiling).** This is an audit of sealed outputs, with no refit, threshold search,
graph integration, or V32 read. It is not a claim that division is unreachable.

The exact 80,013,717-action deployment population has 83 annotated divisions
(base rate `1.037322e-6`). A material `+0.005` combined-score contribution
requires division Jaccard at least `0.05`: minimally five true divisions with
at most 17 false actions in the first 22, i.e. **precision@22 >= 22.73%**.
Even under the most favourable placement of those five hits compatible with
that top-22 limit, AP must be at least `0.0087779`, or **8,462x** the deployment
base rate. The observed pair-action AP is `1.9093e-5` (18.41x lift), roughly
460x below that conservative AP-lift floor.

There is no scalar AUROC equivalent: a ranking with five positives in the
first 22 can have AUROC anywhere from about `0.077` to `1.0`, while the required
operating point is TPR `7.69%` at FPR no greater than `2.1246e-7`. This is why
the observed AUROC `0.8893` was not merely below a higher AUROC gate—it was the
wrong gate.

The re-analysis separates the two heads:

| Head | Population | AUROC | AP |
| --- | ---: | ---: | ---: |
| Event: does the parent divide? | 2,858,003 parents / 82 positive | 0.521521 | 0.00003137 |
| Pair: which daughters, conditional on a true dividing parent? | 2,292 pairs / 65 positive | 0.897081 | 0.368520 |

The conditional pair head is a **working banked asset**: daughter assignment is
solved for reachable true dividing parents at AUROC `0.897081` / AP `0.368520`.
The event head is effectively random. Its frozen contract used
geometry/lineage/motion only and excluded raw texture/appearance, so this is a
failure to provide a working division-event mechanism—not evidence against an
appearance-based mitosis detector. A fifth end-to-end attempt under the same
signal contract is therefore not scientifically justified.

The precise unblock is an authorized appearance-capable parent-event detector
that clears the deployment operating requirement (at minimum 5 TP / <=17 FP
in the first 22, precision@22 >=22.73%, AP-lift >=8462x), while reusing the
banked conditional pair head. Until then, near-term work goes to ordinary
association rather than another end-to-end division refit.

Of the 18 unreachable divisions, 11 are absent because the relevant parent or
daughter is not matched into the prediction candidate universe, 4 are outside
nearest-8, and 3 have both daughters offered but no pair record. The frozen
synthetic generator has no r10 cutoff, so **zero can be assigned to r10**.

Authoritative machine receipt: `REQUIREMENT_SURFACE_V1.json`.
