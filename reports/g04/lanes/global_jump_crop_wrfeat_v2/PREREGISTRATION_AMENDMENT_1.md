# Amendment 1 — object-size crop and bounded fallback

Date: 2026-08-03  
Timing: frozen after C0 geometry output, before implementing/running C1 or opening
any GPU pilot result.

The literal detection-to-selected-full-frame-mask p99 rule in C0 selected
`(48,128,80)`, but the same audit shows why that is not a faithful cell-size
measurement: 243 detections alias onto only 146 selected computed masks, and the
nearest-centroid offsets extend to 17.42/20.77/12.94 um in Z/Y/X.  Expanding a
per-detection crop to encompass a distant aliased full-frame mask would preserve
the diagnostic error rather than repair it.

The unique computed-object distribution is the binding size evidence instead.
Its p99 Z/Y/X voxel extents are `23.30/39.30/34.90`.  Adding the frozen 2-um
margin on both sides requires `27.30/49.30/44.90` voxels.  The amended primary
crop is therefore the next frozen ladder size with symmetric XY batching:
`(32,64,64)`.

The preregistered recenter is now made exact: extract the original
`(16,32,32)` anchor crop; subtract its local median; clamp negative weights to
zero; take the weighted voxel centroid; and translate the anchor toward that
centroid by at most 4.0 um in physical Euclidean norm.  A zero-weight crop keeps
the original detection.  No label or target is used.

Primary crops are processed first in deterministic GPU batches.  Only a primary
crop that is empty, unmatched, nonfinite, all-zero, or whose selected mask
touches its crop boundary enters one deterministic fallback batch at
`(48,96,96)`.  The fallback replaces the primary row only if it is finite,
nonzero, matched, and not boundary-touching; otherwise the row remains failed.
There is no third crop, shape sweep, or per-family setting.

The C2 populations, coverage thresholds, runtime projection, budget gate, and
prohibition on a full run remain unchanged.  Runtime includes both primary and
fallback work.  The unamended `CROP_GEOMETRY_44b6_40c45f5a_T000.json` remains
preserved as the diagnostic that motivated this correction.
