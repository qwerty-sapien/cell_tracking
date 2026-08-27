# PEFT escalation node — preregistered, not launched

This node implements the campaign plan's own §6.2 escalation after frozen
representations failed. It is not a new mechanism and it does not authorize a
full fine-tune.

The imaging backbone and every base Trackastra weight remain frozen. Only a
LoRA or bottleneck adapter in the association transformer/head may train, and
the exact trainable/base parameter ratio must be below 1%. Selection is by
nested-OOF `44b6` held-fit frozen-scorer graph delta, never proposal loss. The
outer split is the existing grouped three-fold artifact
`reports/g02/grouped_3fold_audit20_v1.json`
(`ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769`).
The complete optimization recipe, `4.8` T4-hour projection, hard
`4 h 55 min` wall-clock kill, authoritative `25.47`-hour starting balance, and
scientific kill line are frozen in `PREREGISTRATION.json`.

No model fitting, GPU launch, `6bba`, CV57, graph continuation, or submission
has occurred. This artifact opens the node for the campaign barrier only.
