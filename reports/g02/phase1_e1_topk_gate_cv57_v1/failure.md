# Phase 1 E1 top-10 gate — stopped on runtime surprise

The exact divisions-off CV57 replay stopped before official scoring. Seven of 57 fragments completed; video 8 (`train:44b6_71a4179f`) failed before its fragment was written.

The top-10 graph produced node `24242` at pre-projection graph coordinates `(11.53333333333334, 13.600000000000026, -9.06666666666663)`. Projecting it into the `(100, 64, 256, 256)` image would require a `3.683333333333318 um` correction, exceeding the frozen `3.2 um` limit by `0.4833333333333183 um`.

No workaround was attempted. There is no official top-k score, no production-default change, no training, no Kaggle action, and no submission-eligible artifact.

Implementation references:

- Deterministic top-k query: `src/celltrack_alpha/bh_graph.py:917`.
- Experimental gate activation and radius control diagnostic: `src/celltrack_alpha/bh_graph.py:1139` and `src/celltrack_alpha/bh_graph.py:1205`.
- Divisions-off top-k replay call: `scripts/score_phase1_topk_gate_cv57.py:222`.
- Fail-closed image-support projection call: `scripts/score_phase1_topk_gate_cv57.py:240`.
- Frozen correction guard: `scripts/run_bh.py:2118`.
