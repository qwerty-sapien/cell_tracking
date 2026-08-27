# N2 — enforce the association prerequisite

> Load `DAG.md`, `runs/<run-id>/N1_decoder_validation.md`, `../../../LIVE_STATE.md`,
> and the authenticated terminal reports for all association arms.

## Objective / gate

Has association continuity improved enough that division decoding is causally
eligible? **GO** requires one named association arm to have adjusted graph delta
`>=+0.005` on **both** 44b6 and 6bba held-fit gates, raw edge and node-recall deltas
`>=0` on both, and a positive increase in distinct connected two-branch opportunities
under the frozen D1 motif grammar.

## Tasks

1. Verify source hashes, split identities, graph/scorer hashes, and terminal verdicts;
   proposal diagnostics cannot satisfy this gate.
2. Recount connected fork opportunities from frozen emitted graphs without GT-based
   thresholding and compare with the unchanged baseline.
3. Record the exact qualifying arm and its minimum-family vector. Do not combine
   partial passes from different arms.

## Output

Write `runs/<run-id>/N2_association_dependency.md` with authenticated inputs, opportunity
counts, verdict, and a proposed root-coordinator ledger row.

## Verdict

- **GO** — one arm satisfies every both-family graph and opportunity condition.
- **PARK** — no completed arm satisfies the full prerequisite; D1 execution remains
  closed but its foundation is retained.
- **DEAD** — the dependency evidence is contaminated, leaked, or cannot be reproduced.

## Cascade

- On **GO** → proceed directly to `N3_44b6_graph_kill.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

