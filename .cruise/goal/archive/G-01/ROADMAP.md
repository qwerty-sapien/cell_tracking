# Roadmap

- Planning state: Draft
- Mode/current slice: New / G-01
- Last reconciled: 2026-07-15
- Repository identity: `c3b9b703b049fc25f10e2b42bb986bb94f137a72` / tree `6545134fe081bf5a1e5b8e7b44b07ea8b0b66935`
- Origin: `.cruise/goal/ORIGIN.yaml` (`cf050ce8f0385c4df44dab480f1f2cec4018aa4dcab54b4a189e16be56ff6aa3`)

## Current Slice

### G-01 - Kaggle Site Verification And Low-Compute Push To 0.92

- Status: Current
- Outcome: Submit the correct alpha model to Kaggle, capture the true site score, and iterate with low-compute methods until score `>=0.92` or the D-04 submission cap is exhausted.
- Completion proof: AC-01 through AC-06, including site score `>=0.92`.
- Block proof: scored submissions exhausted below `0.92`, Kaggle blocks scoring/submission, or low-compute envelope cannot plausibly improve without new approval.

## Future Slice Candidates

### G-02 - Expanded Modeling Or Compute Decision

- Status: Future only if G-01 blocks below `0.92`.
- Trigger: G-01 closeout shows that low-compute improvements and three site submissions did not reach `0.92`.
- Possible scope: Decide whether to expand submission budget, use GPU/paid compute, train heavier models, or pivot to a different architecture.

### G-03 - Final Packaging And Reproducible Release

- Status: Future only after a satisfactory site score.
- Trigger: G-01 reaches `>=0.92`.
- Possible scope: freeze the winning package, produce a reproducibility bundle, clean generated artifacts, and optionally commit/push if requested.
