# Final-artifact strictness audit — 2026-08-04

## Outcome

The active `scoreall` final-input gateways already reject malformed graph
artifacts before any call to `official_metric_row`, whose internal
`prediction_to_graph` conversion is intentionally permissive for diagnostic
callers.  No production refactor was needed.

| Gateway | Strict boundary | Evidence |
| --- | --- | --- |
| `scoreall --submission` | `iter_aggregate_graphs` creates one `SubmissionGraph` per contiguous dataset and calls `validate_graphs` before yielding it to `score_graph`. | Missing endpoint, duplicate edge, and self edge all reject. |
| `scoreall --frags` | `_fragment_graph` calls `read_submission`, which calls `validate_graphs` before returning the one expected graph. | The same three malformed cases all reject. |
| Current V32 scoring / CV scorer | `public_temporalunet_baseline.py` and `score_cv_submission.py` use the same strict `read_submission`; V32 terminal output additionally uses the strict G04 parser/validator. | Read-only call-site audit. |
| Other `score_graph` callers | `compare_submissions` obtains graphs through `read_submission`; the G03 cache rereads them through that parser; `ablate` validates both source and transformed graphs. | Read-only call-site audit. |

`scripts/scoreall.py` does not directly import or call `prediction_to_graph`.
It converts only the already-validated `SubmissionGraph` to row dictionaries
for `official_metric_row`.  The remaining direct `prediction_to_graph` users
are official-scoring support or explicitly diagnostic/visualization scripts,
not final-output gateways.

## Regression added

`tests/test_scoreall.py::ScoreAllTests::test_final_input_gateways_reject_invalid_edge_contracts`
drives each bad CSV through both public scoreall input loaders:

- edge with a missing endpoint;
- duplicate edge; and
- self edge (rejected as a non-forward edge).

No source file was changed by this strictness lane.  The pre-existing
post-scoring visual-review edits in `scripts/scoreall.py` and its tests were
preserved.

## Verification

All commands used one thread:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

| Command | Result |
| --- | --- |
| `.venv313/bin/python -m pytest -q tests/test_scoreall.py -k final_input_gateways_reject_invalid_edge_contracts` | `1 passed, 16 deselected, 6 subtests passed in 2.09s` |
| `.venv313/bin/python -m pytest -q tests/test_scoreall.py tests/test_kaggle_calibration.py tests/test_g02_integrity.py` | `36 passed, 22 subtests passed in 3.37s` |
| `git diff --check` | PASS (exit 0) |

## Remaining condition

Future final-output code must continue to enter through `read_submission` or
explicitly call `validate_graphs` before conversion to official-scoring rows.
`prediction_to_graph` must remain reserved for its legacy diagnostic/scorer
semantics; making it strict would change those callers' behavior.
