# G-03 Phase 1 — owner-authorized single-file transport repair

**Status:** locked before replacement GPU outcome.  This amendment is strictly
limited to the Phase 1 infrastructure proof.  It authorizes no Phase 2
association training, graph-based checkpoint selection, CV57, submission, or
final-slot action.

## Reason and authority

`onebitbrain/g03-phase1-resume-a-v1` stopped after 0.70 seconds with
`ModuleNotFoundError: g03_resume_torch`.  The terminal log establishes that it
had not imported Torch, initialized CUDA, read a dataset, trained, created a
checkpoint, or consumed measurable GPU quota.  The failure is therefore a
transport defect, not evidence about resume behavior or association training.

The owner’s 2026-08-01 instruction, “can u do that now, ignoring that
unnecessary restriction”, expressly replaces **only** the previous Phase 1
no-retry restriction.  It permits exactly one new, observation-distinct
transport-repair attempt: A v2 followed, only after A passes all handoff
integrity checks, by B v2.  A v1 remains an immutable terminal record.

## Frozen replacement boundary

- The builder emits `script.py` as the declared Kaggle `code_file` for both
  private T4 kernels.  Each script contains the handoff implementation, Torch
  workflow, immutable role contract, and component hashes.  It may not import
  any sibling project module, including `g03_checkpoint_resume` or
  `g03_resume_torch`.
- The builder executes `script.py --self-test` from a newly created directory
  containing only that script.  A package cannot be launched unless both role
  receipts are `PASS_G03_SINGLE_FILE_SELFTEST`.
- Session A is `onebitbrain/g03-phase1-resume-a-v2`; it trains the exact fixed
  fp32 synthetic 256→512→256 workload for at least 3,600 seconds, checkpointing
  at least every 600 seconds and publishing `g03-phase1-a-to-b-v2`.
- The carrier is a new private immutable dataset
  `onebitbrain/g03-phase1-handoff-v2`.  Its root must contain exactly the
  atomic `handoff/` publication object: payload hashes, manifest, inventory,
  and READY marker.  Any partial object or identity drift fails closed.
- Session B is a new kernel, `onebitbrain/g03-phase1-resume-b-v2`.  It receives
  only the v2 handoff dataset and the pinned
  `onebitbrain/trackastra-h1-fit105-cache-v1` cache.  It validates the entire
  handoff, checkpoint SHA, component hashes, configuration, runtime
  fingerprint, optimizer/scheduler/scaler state, Python/NumPy/Torch CPU/Torch
  CUDA RNG state, and zero-worker sampler cursor before one optimizer step.
- On bit-identical continuation, B emits the same 36-identity fit105/44b6
  structural no-op graph receipt: all deltas exactly 0.0,
  `official_frozen_graph_objective_evaluated:false`, and
  `selection_authorized:false`.  This is plumbing evidence only.

## Pass/fail, controls, and cost

PASS requires the original A→B proof conditions plus successful isolated
single-file self-tests for the exact hash-pinned A/B package files.  All
existing controls remain binding: `cmux_ops.py` serial lock, authoritative
quota before/after receipts, no-running-kernel preflight, exact version/output
accounting, session cap, sustained 256 MiB/15 s swap guard, and byte-verified
handoff carrier.

This amendment budgets at most 4 T4-h inside the original 20 T4-h Phase 1
ceiling: about 3 h for A’s timed segment plus uninterrupted reference and 1 h
for B’s resumed continuation.  A v2 transport failure, state mismatch,
partial handoff, or missing structural receipt is a hard stop.  No A v3/B v3,
model change, or further retry is authorized without a new owner decision.
