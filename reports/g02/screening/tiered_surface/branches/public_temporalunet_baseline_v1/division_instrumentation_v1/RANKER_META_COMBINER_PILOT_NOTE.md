# Ranker meta-combiner pilot — methodological finding (2026-08-07)

**Result:** AUC ~0.50 for edge_prob alone (0.4973), -edge_dist alone (0.5046), and a
5-fold CV logistic combination (0.4803), on the one real edge_prob-carrying artifact
found on disk (`smoke_remote_output_v1/.../6bba_edf14583.geff`, 47,848 edges, 718
GT-matched).

**Checkpoint authenticity verified, not a toy model:** `primary_weight_sha256` in this
run's receipt (`12f6881ee3620a...`) matches the exact production checkpoint used in
every real cv57/hidden-submission run this session — "smoke" here means "one video,"
not "stub model." Labels verified sane: 718/760 GT edges matched (~94.5% recall),
consistent with a real, reasonably-performing classifier.

**Why the near-chance AUC is not itself the finding:** this `.geff` stores the
**post-ILP solution graph** — the edges already SELECTED as final by the ILP, which
itself optimizes heavily on `edge_prob` as its cost. By construction, edges that
survive selection are the ones the ILP was already confident about; residual
edge_prob variance *within the selected set* need not carry much leftover signal
about which are still wrong, since the wrong survivors are typically wrong for
reasons (occlusion, track crossings) a scalar probability doesn't resolve.

**The actual blocker:** R3's ranker meta-combiner proposal needs the **pre-ILP
candidate pool** (multiple scored candidate edges per node, before selection) to
be a meaningful test — that data does not exist on disk anywhere. Generating it
requires a new targeted inference run (the GPU-bound model forward pass), which is
gated the same way R1's gap-closing validation is: real testing needs compute this
report does not have standing authorization to spend.

**Verdict:** blocked on data availability, not a negative result about the ranker
fusion hypothesis itself. Do not cite the ~0.50 AUC above as evidence against
meta-combining the local-association ranker — it answers a different, easier
question (does post-selection edge_prob separate right/wrong survivors) than the
one that matters (would fusing the ranker's 22 features with edge_prob improve
selection *before* the ILP commits).
