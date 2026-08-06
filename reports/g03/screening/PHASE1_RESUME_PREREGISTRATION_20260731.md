# G-03 Phase 1 — cross-session resume infrastructure preregistration

**Status:** locked before GPU outcome.  This is an infrastructure proof, not a
candidate screen.  It authorizes no Phase 2 training, graph-gated checkpoint choice,
CV57, submission, or final-slot action.

## Objective and scope

Prove that a deliberately small CUDA association workload can train continuously for
at least 3,600 wall-clock seconds, checkpoint, terminate, resume from the published
checkpoint in a **new private Kaggle kernel**, and finish bit-identically with an
uninterrupted reference trajectory.  The proof boundary is all of: model parameters,
optimizer, scheduler, disabled AMP scaler state, Python/NumPy/Torch CPU/Torch CUDA RNG,
and the deterministic zero-worker sampler cursor.  The workload is synthetic and
does not read competition data.

The only data read in the second kernel is the pinned fit105 manifest from
`onebitbrain/trackastra-h1-fit105-cache-v1`, SHA-256
`ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28`.
It is used solely to authenticate and enumerate the 36 `train:44b6_*` identities.
No raw cache, labels, graph score, 6bba data, validation12 data, CV57 data, or public
data may be read.

## Frozen workload and endpoints

- Source: `scripts/g03_checkpoint_resume.py`, `scripts/g03_resume_torch.py`,
  `scripts/g03_phase1_a.py`, `scripts/g03_phase1_b.py`, and
  `scripts/build_g03_phase1_resume_proof.py`.  The package builder writes a receipt
  with every source SHA-256.  That receipt, plus this document, is the launch pin;
  changing any source requires a new preregistration.
- Device: one Tesla T4, private script kernels, Internet disabled, no competition
  source.  Kernel A gets no dataset source.  Kernel B gets only the new handoff dataset
  and the pinned fit105 cache dataset.
- Session A (`onebitbrain/g03-phase1-resume-a-v1`) trains a fixed fp32
  256→512→256 MLP for at least 3,600 seconds, writes an atomic local checkpoint every
  600 seconds, then publishes `g03-phase1-a-to-b-v1`.  It independently regenerates a
  same-kernel uninterrupted 2×step reference before publishing the handoff.
- The sole handoff carrier is a new private immutable dataset
  `onebitbrain/g03-phase1-handoff-v1`, created from A's finished output.  Its root must
  contain exactly the atomic `handoff/` publication object: payload hashes, manifest,
  inventory, and READY marker.  Partial objects, SHA drift, undeclared files, and a
  conflicting idempotent restart all fail closed.
- Session B (`onebitbrain/g03-phase1-resume-b-v1`) is pushed only after A is COMPLETE,
  its output is downloaded and independently SHA-verified, and the handoff dataset is
  READY.  B first verifies the package source, manifest, full tree hash, checkpoint
  hash, config, device/runtime fingerprint, and all state before it performs a single
  optimizer step.  It trains exactly the saved number of additional steps and compares
  the complete canonical state digest to A's uninterrupted reference.
- After bit-identical continuation passes, B emits the 36-member **44b6 structural
  graph-delta receipt** from the resumed checkpoint.  Candidate and control edges are
  deliberately byte-identical, so adjusted/raw/node delta must each be exactly `0.0`.
  It is an emission-plumbing proof only: `official_frozen_graph_objective_evaluated`
  must be `false`, no target values may be accessed, and the receipt may not select a
  checkpoint or be treated as a Phase 2 graph result.

## Pass/fail and budget

PASS requires all of the following, without a retry:

1. A reports `training_minutes >= 60`, READY handoff, a same-kernel reference equality,
   and a complete immutable publication tree.
2. B is a genuinely new kernel version, verifies every handoff component before
   restore, and its complete final canonical state digest exactly equals A's reference.
3. B emits one 36-identity, fit105/44b6-only structural receipt with all graph deltas
   exactly zero and `selection_authorized:false`.
4. Guard, serial lock, quota-before/after receipts, output SHA receipts, exact version
   accounting, no swap-category violation (sustained >=256 MiB), and both private
   kernel terminal receipts pass.

Any mismatch, partial handoff, runtime/source/config drift, missing new-kernel boundary,
or inability to produce the declared structural receipt is a **Phase 1 HARD FAIL**.
No second handoff attempt is permitted under this preregistration; G-03 stops for
reassessment as chartered.  Phase 1 is capped at 20 T4-h; expected use is about 4 T4-h
(A ≈3 h for the timed segment plus reference, B ≈1 h for continuation), leaving margin
for platform overhead but not a retry.

## Operational controls

All authenticated Kaggle operations use `scripts/cmux_ops.py` with its serial lock and
pre/post authoritative quota bracket.  Before each mutation, record the current source
bundle SHA, the exact package-directory SHA inventory, frozen metadata, `git rev-parse
HEAD`, no-running-kernel status, and the contract SHA.  Do not trust a local scheduler
or an expired quota window as a reason to launch.  A and B run from this dedicated
worktree/commit only; no uncommitted root-worktree output crosses the boundary.

The result does not amend the selected G-03 architecture or its Phase 2 gates.  The
first scientific checkpoint is still assessed only on the frozen 44b6 held-fit graph
objective at the preregistered fixed training interval.
