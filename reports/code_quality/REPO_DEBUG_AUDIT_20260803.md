# Repository debug, dead-code, and maintainability audit

Date: 2026-08-03  
Method: `cruise:debug`, two adversarial read-only review waves plus local
reproduction and focused repairs.

## Outcome

The recent G04 code-submission failure was not reproduced as a malformed CSV.
The high-confidence, evidence-bounded cause is a hidden-rerun transport mismatch:
the invalid kernel hard-coded four public identities and recorded
`competition_input_visits=0`; ref `55201829` then completed with blank scores and
`total_bytes=0`. This is the same root-cause family as ref `54890576`, whose
frozen public registry failed when Kaggle reran it against a different hidden
population. The hidden traceback for `55201829` is not available, so this is an
inference from exact local/remote artifacts, not a claim of direct traceback
observation.

The active submission path is now globally one-shot, exact-version bound, dynamically
hidden-test compatible, and reference reconciled. The obsolete public-copy
transport, orphaned pseudo-SSL tests, and unused Ultrack prototype are archived,
not left as plausible active fallbacks. A zero-edge official-score crash and an
automatic-routing partial-map failure were also repaired.

Crop-level computed wrfeat is `PARK-COST`, not `PARK-EVIDENCE` or `DEAD`.
The `32x64x64` recentered geometry plus `48x96x96` fallback and `914/916`
coverage proof are banked. No fourth throughput configuration is allowed without
a new named architecture that eliminates redundant tri-orientation work.

## Scope and exclusions

Reviewed surfaces:

- all active Python under `src/`, `scripts/`, and `tests/` (427 files,
  approximately 194k lines at audit time);
- active package, scoring, routing, candidate/graph, G03/G04 orchestration, and
  Kaggle mutation/submission paths;
- test discovery, active runbooks, failure registry, and ownership checks;
- archived implementations only far enough to prove active references were
  removed and provenance retained.

Excluded from code-quality classification: raw/derived data, immutable result
payloads, vendored `external/` sources except where evaluator parity required
comparison, generated kernel payloads except where they were the failure
evidence, and root Markdown basenames beginning `_` (write-only user references
under the selected debug workflow).

## Confirmed defects and repairs

### Submission reliability

1. **Public-only code output could be submitted to a hidden rerun.** The old G04
   transport copied a public CSV and never inspected competition input. It is now
   terminally documented and archived. `KFG-25` records both refs `54890576` and
   `55201829` and forbids retries under a different wrapper.
2. **A downloaded output was not bound to an exact kernel version.** The G04
   terminal guard now requires a successful CMUX `kernel_output` receipt whose
   argv names exact `owner/slug/version` and the exact download directory.
3. **Predicates were reusable.** The shared submission lifecycle writes an
   atomic claim in a canonical cross-wrapper store before the external mutation.
   Changing the receipt directory or copying the predicate cannot bypass it.
   Any attempt, including timeout or unresolved reference, consumes the predicate permanently.
4. **CLI status zero was treated as submission success.** The lifecycle takes a
   pre-submit inventory, submits once, takes a post-submit inventory, and
   succeeds only when exactly one new reference is bound. It classifies blank
   `COMPLETE` separately from scored success.
5. **Mutation serialization was incomplete.** Dataset create/version and direct
   competition submit now acquire the shared serial lock automatically. A bypass
   requires an actual currently-held lock object, not a boolean flag.
6. **Direct-file predicate validation was only `eligible:true`.** It now requires
   schema/status/competition/mode, exact artifact path/hash/bytes, no failures,
   and explicit non-revoked/non-consumed state.
7. **The ownership registry was stale.** Four live slugs found by the structural
   freshness check were registered. The check now returns `PASS`.
8. **The mutation boundary accepted unsafe operational states.** Kernel pushes
   now query the latest slug and refuse failed, timed-out, or unparseable status.
   Non-finite quota, ledger, or reservation values are rejected. Credential
   injection is limited to a resolved trusted Kaggle executable. Direct-file
   predicates receive canonical consumed-on-attempt protection.
9. **Stale lock recovery and reservation release were racy.** The delete-and-
   replace stale-file protocol is gone. A persistent diagnostic file now carries
   an OS advisory lock on an open descriptor, so process death releases ownership
   atomically and no supervisor ever unlinks another owner's lock. Releasing an
   active reservation requires exact parsed terminal status evidence for that
   kernel, or a failed-push receipt whose predicate binds that kernel. Bare
   `ERROR`, substring states such as `INCOMPLETE`, and cross-kernel receipts fail
   closed.

No external Kaggle mutation was made during this repair.

### Scoring and routing correctness

1. **Valid node-only or empty predictions crashed the full score wrapper.** The
   organizer intentionally returns a zero result without adding match attrs in
   this case. `official_metric_row` now applies the organizer CLI's zero-recall
   rule and has node-only and empty-prediction regression tests.
2. **The edge-only wrapper mislabeled detection recall as official node recall
   for zero-edge graphs.** It now uses the same zero-recall evaluator contract.
3. **Automatic routing accepted partial config maps and later raised an opaque
   key error.** Auto mode now requires every route label used by the heuristic,
   checks key/config-label agreement, and tells callers to use `forced_label`
   for a partial map.
4. **Sparse tie noise could override a real cost difference.** Tie perturbation
   is now restricted to components where every feasible edge has exactly the
   same original cost; a mixed-cost `1e-15` regression proves the lower-cost
   matching wins.
5. **Three private helpers were provably unreferenced.** `_detect_logits`,
   `_windows_for`, and `_sqlite_metric_count` were removed; their surrounding
   suites remain green.

### Test and lifecycle hygiene

1. Pytest collection previously failed on three tests whose implementations had
   already been archived. Pseudo-SSL and Ultrack tests now live with their
   implementations under `_ARCHIVE/`; active collection succeeds.
2. The remaining Ultrack runner and active runbook commands were archived, and
   active architecture/TODO documents now point to the archive rather than a
   nonexistent execution path.
3. `pyproject.toml` now supplies canonical active pytest roots and Python paths,
   including the standing prelaunch guard and active global-reassign oracle.
   It intentionally does not pretend to be a dependency lock or installable
   package manifest.

## Dead, disused, and retained classifications

| Surface | Classification | Action / reason |
|---|---|---|
| G04 public-copy transport (`g04_division_code_transport.py`, old submit wrapper, tests) | terminal invalid infrastructure path | archived together; kernel and receipts retained as KFG-25 evidence |
| `celltrack_ultrack`, installer/validator, ZIP CV runner, two tests, runbook | unused unsupported prototype | archived together; not an active fallback |
| pseudo-SSL orphan test | disused implementation family | moved beside already-archived implementation |
| `_detect_logits` in `bh_model.py` | dead private helper | removed; batch detector is the only caller path |
| `_windows_for` in point-context gate | dead private helper | removed; partition loader performs the active coverage check |
| `_sqlite_metric_count` in G04 sweep | dead private helper | removed; active recovery queries the metric count inline |
| old endpoint/residual probe cluster | historical experiment code, not proven dead | retained; archive only after its reports/imports are lifecycle-indexed |
| active V32/G03/G04 kernel builders and frozen package sources | hash/evidence coupled | retained even when large; deleting or broad-splitting them would silently invalidate receipts |

Scientific mechanisms remain governed by `LIVE_STATE.md` statuses. This audit
does not relabel scientific evidence as code deadness.

## Monolith assessment and refactor boundary

Largest active files at audit time include:

| File | Lines | Decision |
|---|---:|---|
| `scripts/run_bh.py` | 3,935 | packaged runtime/evidence coupled; split only behind parity fixtures and a new package identity |
| `src/celltrack_alpha/bh_edge.py` | 2,763 | active sparse-candidate contract; defer mechanical split during V32/G04 integration |
| `src/celltrack_alpha/pipeline.py` | 2,741 | genuine orchestration monolith and missing end-to-end test; highest-priority future seam |
| `src/celltrack_alpha/bh_model.py` | 2,712 after cleanup | active model/cache contract with many frozen hashes; defer broad split |
| `scripts/g04_computed_segmentation.py` | 2,560 | PARK-COST reusable implementation; preserve banked geometry/runtime evidence |
| `src/celltrack_alpha/bh_graph.py` | 2,367 | active graph realization; split only with byte-parity graph fixtures |
| `scripts/g03_division_oof_orchestrator.py` | 2,324 | campaign orchestrator; future split into contract, scheduling, and receipt layers |

The safe monolith refactor completed here was submission lifecycle extraction:
the active G04 wrapper is now 167 lines and delegates mutation/claim/inventory/
reference logic to one tested 356-line shared module. The old duplicate wrapper
was archived. Broad source splits were not performed merely to reduce line
counts: several files are embedded into generated kernels or frozen by SHA, and
the current worktree contains concurrent campaign changes. Those splits require
explicit compatibility seams and byte-parity tests first.

Recommended future seams, in order:

1. extract `pipeline.py` validation/input loading from execution after adding one
   successful end-to-end `predict_dataset_window` fixture;
2. extract `bh_model.py` cache identity/serialization before detector/model
   separation;
3. extract `bh_graph.py` candidate indexing from realization with exact edge-set
   parity;
4. turn G03/G04 orchestrators into thin CLIs over contract/scheduler/receipt
   modules;
5. replace eager `celltrack_alpha.__init__` imports only after public-import and
   kernel-package compatibility tests exist.

## Test-suite triage closeout

The former broad result was capped at 25 failures and therefore was not a
complete inventory. The uncapped run found 43 failures. All are now explicitly
reviewed: four active defects were fixed and 39 terminal/retired exact nodes are
registered as strict XFAIL with per-node lifecycle reasons. No test qualified
for deletion because every retired failure still documents a frozen package,
hash, or unavailable owner-local integration boundary. Full enumeration:
`TEST_SUITE_TRIAGE_20260803.md`.

Active scorer entrypoints now use `scorer_parity_v2.json`, which re-freezes the
zero-edge repair without rewriting immutable parity-v1 evidence. The managed
macOS process-pool tests bypass only the denied semaphore-limit sysconf probe;
they still create real worker processes and verify byte parity and termination.

## Verification

- Active collection: `1237 tests collected`, zero collection errors, including
  the standing prelaunch guard and active oracle.
- Submission/guard/scorer/oracle focused suite: `100 passed, 7 subtests passed`.
- Official scorer parity suite: `9 passed, 7 subtests passed`.
- Routing/config suite: `26 passed, 118 subtests passed`.
- Removed-helper surrounding suites: `36 passed, 3 subtests passed`.
- Canonical uncapped suite: `1198 passed, 39 strict xfailed, 267 subtests
  passed`, zero failures.

## Remaining actionable debt

### Reviewed XFAIL — retired V45 sparse/dense assignment tie semantics

The mixed-cost optimizer defect is fixed and the banked adjacent assignments now
match. The artifact test still finds one sparse/dense one-frame-gap pair
difference (`22521` versus `22519` into target `23208`) at equal cardinality.
Component-local sparse solving cannot generally reproduce LAPJV's full
rectangular filler-dependent tie. Pin the SciPy/runtime and choose a golden
semantic contract before changing the banked assertion; cross-mode exact ties
require a true secondary objective, not general epsilon noise. The exact banked
V45 diagnostic is now a strict lifecycle XFAIL (`R12`); active mixed-cost and
random exhaustive-optimum regressions remain passing.

The pre-existing transformed cost `cost / scale + 2.0` can also collapse
distinctions smaller than one float64 ULP near 2.0. Until the solver has a stated
numerical-resolution contract, “exact” means exact at represented transformed
cost, not exact over arbitrary real-valued inputs.

### P1 — complete kernel-package binding

CMUX revalidates metadata and declared code bytes under the lock, but current
guard schemas do not uniformly provide a complete package manifest or
resolvable contract/registry paths. Extra packaged files can drift after
prelaunch without central detection. Migrate every guard protocol to one common
package-tree/contract/registry binding, then make CMUX require it.

### P1 — submission-reference attribution under external concurrency

The lifecycle polls a bounded inventory and requires exactly one new ref, but an
out-of-band concurrent submission could still be the sole delta and be
misattributed because a stable Kaggle message/file/ref schema has not yet been
authenticated. The serial lock controls repository actors only; reconcile the
row's message/file/time evidence before treating it as scored success.

### Resolved — active source-pin migration after scorer repair

Historical packages that freeze the SHA of `official_eval.py` still fail closed,
as designed, and carry strict per-node XFAIL reasons. Active scoring uses the
new `scorer_parity_v2.json` identity after green official parity tests. No
historical receipt or parity-v1 evidence was edited.

### P1 — final-artifact strictness

`prediction_to_graph` remains permissive for legacy diagnostic callers: missing
endpoints, duplicate edges, and self edges are dropped. `scoreall` reads and
validates `SubmissionGraph` first, but any new final-output gateway must use the
strict parser/graph validator rather than call this helper directly.

### P2 — environment and dependency contract

The repository still has no authoritative dependency lock, supported Python
matrix, or single install command. The working `.venv` and `.venv313` have
different capabilities. Add a lock only after separating campaign/runtime/test
extras; a partial list would provide false reproducibility.

### P2 — lifecycle inventory

Many active and historical scripts lack an explicit lifecycle manifest. Add a
machine-readable index (`ACTIVE`, `PARK-*`, `ARCHIVED`, `GENERATED`) before the
next large deletion wave so historical probes cannot masquerade as supported
entrypoints.

### P2 — test taxonomy refinement

Hash-pinned historical release tests and active unit tests still share one
default collection, but every known retired red is now an exact-node strict
XFAIL and the default target is reviewed-green. A later marker split into active
correctness, release-integrity, artifact-available, and external-runtime groups
would improve runtime and reporting; it is no longer required to distinguish
reviewed failures from an untriaged pile.

## Final gates

- Crop-wrfeat stays `PARK-COST`; no fourth throughput configuration.
- The geometry/coverage proof remains reusable if Trackastra/wrfeat returns.
- No RunPod action is authorized by this audit.
- No public-only code-output transport may be retried.
- A future Kaggle code submission must use the shared exact-version, one-shot
  lifecycle and end in an explicitly reconciled submission state.

The local `_ARCHIVE/` tree is gitignored and source-only; tracked tombstones,
reports, and Git history are the durable retirement record. Archived suites are
not part of active collection and are not promised to replay in place.
