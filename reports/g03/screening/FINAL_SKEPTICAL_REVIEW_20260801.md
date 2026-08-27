# G-03 B-v2 and weekly-direction skeptical review

Date: 2026-08-01  
Review status: **PASS / no unresolved in-scope finding**

This review was performed after the terminal reports and state updates. It
rechecked the external-action receipts and scientific recommendation from
first principles rather than treating remote `COMPLETE` as success.

## Findings challenged

1. **Could B be a false pass?** The sole push is exact version `0→1` and is
   bound to frozen code `9bf7e812…6f6f2`, metadata `9f83bdfa…203bed5`, and the
   fresh predicate receipt. The downloaded output independently replays the
   terminal validator: exact A tree, resume step, unique 36-member inventory,
   candidate/control edge hashes, no-op deltas, contract hash, and swap limit
   all pass. The frozen runtime emits PASS only after comparing the resumed
   endpoint with A's stored uninterrupted-reference state. No mismatch or
   missing binding was found.
2. **Was compute overused?** Post-terminal authoritative quota is
   `1.01/28.99 h` used/remaining. The run took about 61 minutes and stayed
   within both the earlier four-hour package amendment and the owner's newer
   eight-hour maximum. Another B run would repeat the same frozen observation,
   so preserving 6.99 owner-authorized hours is required.
3. **Was plumbing mistaken for science?** Both B artifacts explicitly set
   `official_frozen_graph_objective_evaluated:false` and
   `selection_authorized:false`. The `0/0/0` structural result is not a score,
   does not satisfy a continuation gate, and cannot support submission.
4. **Did the direction audit omit a parked node?** A machine cross-check found
   all 14 `PARK_A` IDs and all 18 `PARK_B` IDs explicitly named in
   `UNEXPLORED_PARKA_DIRECTION_AUDIT_20260801.md`; none is missing. Newer G-02
   terminal evidence was used where machine rows were stale.
5. **Could a completed kernel be submitted?** The DINOv2 preflight, Trackastra
   cache, A-v2, and B-v2 are input/cache/infrastructure artifacts, not eligible
   competition predictions. The only measured native candidate lost by
   `-0.052328011880`, and its hybrid is exactly APNCC. Submission remains none.
6. **Did Phase-2 preparation leak evaluation data?** Initial preparation prose
   called the whole 105-video universe “training scope.” Review corrected this:
   the exact train/evaluation partition remains an open preregistration item,
   44b6 is evaluation-only, and audit/CV57 remain inaccessible.
7. **Were the owner gates changed correctly?** Fixed graph checkpoints remain
   at 30/60/90 T4-hours. The binding continuation lines are `>=+0.010` at 30
   and `>=+0.035` at 90. A fixed 60-hour graph read is diagnostic only; no
   numerical 60-hour continuation rule was invented.

## Verification replay

- G-03 validator plus CMUX: 41 tests, PASS.
- G-02 prelaunch regression: 17 tests, PASS.
- Failure-registry freshness: PASS, no uncovered slug.
- Terminal validation regeneration: byte-identical to the recorded JSON.
- Python compilation, JSON parsing, and `git diff --check`: PASS.
- PARK-A/PARK-B name census: `14/14` and `18/18`, PASS.

## Independent conclusion

The costly action was productive and terminally complete. No repeat, v3,
submission, CV57 read, second encoder lane, G-02 reopen, or RunPod action is
justified. This week's dominant work should be local/CPU closure of the
Phase-2 crop provenance, crop semantics, adapter/trainability census, declared
output set, and immutable 30/60/90 checkpoint schedule. Only then should one
small preregistered T4 calibration measure VRAM, examples/second, checkpoint
bytes, and 30/90-hour cost projection.

