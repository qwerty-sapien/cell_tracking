# Lane 4 pipeline P1 receipt — 2026-08-04

## Scope

- Added a small deterministic, successful `predict_dataset_window` fixture before
  changing the active pipeline seam.  It uses a two-frame `(2, 7, 33, 33)`
  float32 in-memory window, forces `clean_point`, emits 14 nodes and 7 edges,
  and pins the canonical JSON graph SHA-256 to
  `97c218aefe8ff5449f0b1367289e3db9ceca6659e092a5be13664204b98082bf`.
- Extracted only the entrypoint's existing array-window and output-offset
  validation/normalization to `src/celltrack_alpha/pipeline_inputs.py`.
  `pipeline.py` keeps its `_validate_array_window` import alias for private
  compatibility and keeps the unchanged public `predict_dataset_window` API.
- No data reader existed in the active in-memory entrypoint.  Adding one would
  change the API/backing-store contract, so the safe input-loading boundary is
  limited to validation and normalization of the caller-provided array.
- No generated, kernel-hash-coupled, archived, PARK-COST, campaign-state, or
  ledger file was changed.

## Verification

All tests used:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

| Command | Result |
| --- | --- |
| `.venv313/bin/python -m pytest -q tests/test_pipeline_validation.py -k small_end_to_end_fixture` (before seam extraction) | `1 passed, 32 deselected in 0.58s` |
| `.venv313/bin/python -m pytest -q tests/test_pipeline_validation.py` | `33 passed, 20 subtests passed in 4.17s` |
| `.venv313/bin/python -m pytest -q tests/test_routing.py tests/test_alpha_solver.py` | `41 passed, 19 subtests passed in 1.07s` |
| `.venv313/bin/python -m compileall -q src/celltrack_alpha/pipeline.py src/celltrack_alpha/pipeline_inputs.py` | PASS (exit 0) |
| public `from celltrack_alpha import predict_dataset_window` fixture | PASS; same route, node/edge counts, and graph SHA-256 |
| `git diff --check` | PASS (exit 0) |

The initial broader command named a nonexistent `tests/test_proposals.py`; it
ran no tests after the already-successful pipeline module and was replaced by
the discovered active routing/solver targets above.

## Residual P1 blockers

- The audit's repository-wide final-artifact strictness remains: callers that
  create a submission graph must use the strict parser/graph validator rather
  than permissive `prediction_to_graph`.
- The audit's package-tree/contract/registry binding and external-concurrency
  submission-reference attribution remain separate P1 work.  Neither is on the
  active in-memory pipeline execution path changed here.
