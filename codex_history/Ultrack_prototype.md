Implemented the Ultrack prototype branch.

Key files:
- [core.py](/Users/agent/Desktop/cell_tracking/src/celltrack_ultrack/core.py:27): `UltrackConfig`, foreground/contour builder, lazy Ultrack runner, Kaggle node/edge conversion.
- [validate_gurobi.py](/Users/agent/Desktop/cell_tracking/scripts/validate_gurobi.py:34): secret-safe `.env` load, optional installs, tiny Gurobi MILP, `ultrack check_gurobi`, JSON report.
- [run_ultrack_zip_cv.py](/Users/agent/Desktop/cell_tracking/scripts/run_ultrack_zip_cv.py:50): ZIP-streamed train clip CV runner with Gurobi/CBC solver selection, derived cache root, CSV validation, sparse metrics.
- [ultrack_runbook.md](/Users/agent/Desktop/cell_tracking/docs/ultrack_runbook.md:1): runnable validation and CV commands.

Verification:
- `py_compile` passed for all new Python files.
- ZIP IO/input smoke passed on a 3-frame `44b6` clip without extracting raw zarrs.
- `gurobipy` 13.0.2 installed and solved the tiny MILP; report written to `reports/gurobi_validation.json`.
- Full local Ultrack validation is blocked because this venv is Python 3.14 and current `ultrack` releases require Python `<3.13`. The runbook documents using Python 3.11/3.12 or Kaggle for the full check.

I used the current Ultrack docs for the foreground/contours API, `Tracker.to_tracks_layer()`, `ultrack check_gurobi`, and `tracking_config.solver_name`: https://royerlab.github.io/ultrack/ and https://royerlab.github.io/ultrack/configuration.html.
