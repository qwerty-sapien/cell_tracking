# CELLECT local swap-guard reassessment — adopted, no further spend

The historical zero-growth abort was not discriminating: its `+3.5 MiB`
shard-start stop was below the banked idle-control range (`+10.06–54.87 MiB`).
The fresh three-window measurement in `SWAP_GUARD_CALIBRATION.json` observed
`0.0 MiB` peak growth in each full 60-second window; the conservative floor
therefore sets the adopted threshold to `256 MiB`, not zero.

Adopted monitor: sample swapout continuously every 0.25 seconds;
hard-abort if growth stays at or above `256 MiB` for 15 continuous seconds or
reaches `384 MiB` (1.5x) at any sample. This preserves a real-harm stop while
filtering sub-noise drift.

Historical interpretation: `+3.5`, `+12.875`, and `+30.625 MiB` are
sub-threshold noise candidates; `+174.6875 MiB` is below the proposed hard
threshold and warrants logging, not a feasibility verdict; `+488.6875 MiB`
would immediately abort under the adopted guard and is a genuine high-pressure
signal. The earlier `+3.5 MiB` stop is retired as noise-confounded, but CELLECT
remains blocked in practice because `+488.6875 MiB` exceeds the `384 MiB`
immediate-abort line and the separate final infrastructure closeout permanently
forbids further spend. This adoption makes no tier move, scientific verdict, or
rerun authorization. Authority and exact parameters are frozen in
`SWAP_GUARD_ADOPTION.json`.
