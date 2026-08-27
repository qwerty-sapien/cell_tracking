# §6.2 PEFT runtime-margin amendment — frozen before launch

The authorized realization is a rank-64 residual bottleneck in the association
head over the frozen 20-column action representation: the sealed 12-column B0
plus the authenticated 8-column Trackastra-derived assignment evidence. The
27,456,880-parameter Trackastra backbone and imaging path remain frozen. The
adapter has 2,665 expected trainable parameters (0.009706% of the authenticated
base), so this is PEFT and not a full fine-tune.

Training retains all 1,118 beneficial DELETE actions and a globally fixed
100:1 sample of 111,800 harmful actions, for 112,918 training rows. The seed,
canonical SHA-256 sampling rule, and expected selected-row digest are frozen in
`RUNTIME_MARGIN_AMENDMENT_20260731.json`. This is standard rare-event
subsampling for the measured approximately 8.31e-4 prevalence. It changes
training cost only: every reported AUC, CG0-budget retention value, and graph
delta uses the complete 1,345,497-action background.

The schedule is shortened from 24 to 8 epochs with patience 2. Atomic
checkpointing occurs every 25 optimizer steps. The conservative projection is
2.25 T4-h, including a 3x margin, while the hard kill remains 5 h.

One execution defect was found before outcomes: the preregistered grouped
three-fold artifact intersects this exact 41-video population as 0/21/20 held
videos, so it has an empty outer fold. The amendment uses the already frozen
five-fold assignment embedded in the population shards; this is the same split
used to seal B0 and evaluate every supplier. No threshold or scientific kill
line changes.

Nested selection is graph-only. Inner-OOF held-fit graph delta selects the
epoch and per-video action-rate multiplier; training loss and proposal metrics
are never selectors. The unchanged outer kill is adjusted delta at least
`+0.005`, raw and node deltas nonnegative, and every outer adjusted delta
nonnegative.
