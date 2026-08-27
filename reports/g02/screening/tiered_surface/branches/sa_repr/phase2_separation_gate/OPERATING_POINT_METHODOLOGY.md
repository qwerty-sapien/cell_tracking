# Supplier methodology at extreme prevalence

Status: **BANKED FINDING / FUTURE-GATE CHANGE PROPOSED FOR HUMAN APPROVAL**

At the frozen supplier population's roughly `10^-4` beneficial-action prevalence,
ROC-AUC is not diagnostic of campaign utility. It averages rank ordering across the
whole negative distribution, while the graph can admit only an extremely small
false-positive tail. A representation can therefore show real global rank signal and
still recover essentially no true action at the allowable operating point.

The campaign now observes that separation directly. DINOv2 achieved mean AUC
`0.670/0.682` against an incumbent near `0.502`, yet retained only `0.0013/0.000` of
beneficial actions under the CG0 per-video FP budget. Trackastra had AUC `0.918` but
produced `-0.0111` at its graph read. The banked CG1/PU/CRC supplier results likewise
retain only about `0.4–1.8%`. The repeated failure is at the constrained operating
point, not necessarily in rank signal.

The binding criterion for this campaign is therefore **retention (recall) at the
frozen CG0 false-positive budget**, accompanied by precision/FP count at that same
budget. AUC remains useful as a descriptive secondary statistic and a debugging aid,
but it must not be interpreted as evidence that the extreme tail is usable.

Proposal for human approval: future supplier gates should lead with
precision/recall-at-budget (per class and per video), with AUC reported secondarily,
rather than lead with an AUC floor. Any new gate must freeze the exact deployment
population, per-video FP budget, tie policy, and aggregation before outcomes.

This note changes no gate. Preregistration
`9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e`,
sealed B0 `a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2`,
and every existing verdict remain unchanged.
