# Comprehensive run logging contract

Status: **BINDING from 2026-08-04 for every local and Kaggle run.**

Each run writes newline-delimited JSON to both stdout and `RUN_LOG.jsonl` in its
artifact directory. The persisted file is part of the terminal artifact set and is
downloaded from Kaggle with successful outputs.

Required event sequence:

1. `config` first: every threshold, radius, window, scale, convention, immutable
   artifact binding, and projection in force.
2. `input_inventory` before any lookup or processing: recursive file listing for
   every mount/input, with path, byte size, and SHA-256. Missing roots are recorded
   explicitly as `status: "missing"`.
3. `lookup` for every resolution: `sought`, `found`, and `status` (`hit` or `miss`).
4. `stage_end` for every stage: duration, peak RSS, and record counts in/out.
5. `preprocess` / `correction` for every such stage: `examined` and `modified`, even
   when modification is zero.
6. `failure` on every unhandled error: full formatted traceback plus current state
   and config. A bare error string is non-compliant.

The active public-TemporalUNet builder emits this schema inline and its output
validator rejects a terminal artifact lacking the persisted log. Other launchers
must adopt this contract before their next run.
