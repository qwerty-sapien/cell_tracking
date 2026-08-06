# C0 — the cmux ops module: bounded command plan, serial lock, receipts

## ✅ BUILT 2026-07-31

- Module: [`scripts/cmux_ops.py`](../../../scripts/cmux_ops.py)
- Tests: [`tests/test_cmux_ops.py`](../../../tests/test_cmux_ops.py) — **23/23 pass**,
  `.venv/bin/python -m unittest tests.test_cmux_ops`
- **Zero external calls made.** The suite drives a fake `kaggle` executable that records
  argv and emits fixtures.

Placed in `scripts/` rather than under `reports/g02/.../kaggle_prelaunch/` because G-02 is
now archived and G-03 needs the same instrument; it sits beside `submit_kaggle_alpha.py` and
`build_kaggle_kernel.py`. It **complements** `kaggle_prelaunch_guard.py` rather than
duplicating it — the guard validates *what you are about to push*, this bounds *how you push
it*. Neither replaces the other, and the runbook's "extend, don't fork" mandate still applies
to the guard.

Verify the bounded surface at any time:

```bash
.venv/bin/python scripts/cmux_ops.py --print-plan
```

What shipped, against the spec below: 9 enumerated ops; argv lists with `shell=False`;
`O_CREAT|O_EXCL` lock released in `finally`; stale-break requiring **both** age and an
in-flight probe, with an unparseable lock treated as fresh (fail safe); credential-scrubbed
receipts that still record the true output SHA-256; `kernel_push` refusing on
`RUNNING`/`QUEUED`; `competition_submit` refusing without `eligible: true` and with no
override flag; `bracketed_launch` aborting if the quota pre-query fails; and no retry
anywhere.

---

*Original specification, retained:*

**No external calls in this node.** You are building the instrument that every later node
uses. Build it with tests before it ever touches the owner's session.

> **Why this is still the open item.** Two independent ad-hoc launch paths have now failed in
> ways this module prevents by construction:
> - the PEFT v1 run hit the known owner `403` on first dataset status (endpoint scope — `C1`
>   encodes verify-by-download-and-hash instead);
> - the native-fit launch stopped before push because **Kaggle's empty-search literal
>   `Not found` was not parsed** by a lane-local launcher, needing a target-aware repair.
>
> Neither was a science problem. Both are the `KFG-18` pattern — the shared guard and a
> shared ops layer exist or are specified, but each lane keeps writing its own launcher and
> rediscovering the same edges. Build this once.

Model it on the existing
`reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_cellect_cmux_roundtrip.py`
— read that first. It already established the `command_plan()` pattern here. Generalize it;
do not fork a second convention.

Target: `reports/g02/screening/tiered_surface/kaggle_prelaunch/cmux_ops.py`, beside the
shared guard, so every lane reuses one module (`KFG-18` exists because lanes wrote their own).

## Required surface

```python
def command_plan(kaggle: str, **kw) -> dict[str, list[str]]:
    """The COMPLETE set of external commands this module may ever execute."""
```

Enumerate exactly these, and nothing else:

| Op | argv | Mutating |
|---|---|---|
| `quota` | `kaggle quota --format json` | no |
| `dataset_status` | `kaggle datasets status <id> --format json` | no |
| `dataset_version` | `kaggle datasets version -p <dir> -m <msg> -r skip` | **yes** |
| `dataset_create` | `kaggle datasets create -p <dir>` | **yes** |
| `dataset_download` | `kaggle datasets download -d <id> -p <dir> --unzip` | no |
| `kernel_push` | `kaggle kernels push -p <dir>` | **yes, spends quota** |
| `kernel_status` | `kaggle kernels status <id>` | no |
| `kernel_output` | `kaggle kernels output <id> -p <dir>` | no |
| `competition_submit` | `kaggle competitions submit -c <comp> -f <csv> -m <msg>` | **yes, irreversible** |

Rules the module enforces in code, not in comments:

1. **Whitelist execution.** A single `run(op, ...)` entry point that refuses any op not in the
   plan. No `shell=True`. No string interpolation into a command. argv lists only.
2. **`competition_submit` is predicate-gated.** It raises unless a caller passes an explicit
   `submission_predicate_receipt` path whose content evaluates `eligible: true` under
   [`C4_SUBMISSION_PREDICATE.md`](C4_SUBMISSION_PREDICATE.md). No default, no override flag.
3. **Serial lock.** Mutating ops acquire an exclusive lockfile
   (`kaggle_prelaunch/.cmux_serial.lock`) via `O_CREAT|O_EXCL`, containing holder PID, node
   name, kernel/dataset ID, and acquisition UTC. Stale-lock policy: a lock older than the
   preregistered hard kill **plus** one hour may be broken **only** after `kernel_status`
   confirms no in-flight run — record the break and its evidence in the receipt. Release in a
   `finally`.
4. **Receipts, always.** Every invocation writes JSON: op name, exact argv, UTC start/end,
   duration, returncode, SHA-256 of stdout and stderr, and scrubbed stdout. **Scrub before
   writing** — drop anything matching key/token/secret/authorization/bearer/cookie patterns
   and any long base64-ish run. A receipt must never carry a credential.
5. **Quota bracketing.** `bracketed_launch(...)` queries quota, runs the mutating op, queries
   quota again, and returns both plus the delta. The delta is the authoritative debit. If
   either query fails, the launch does not proceed.
6. **Idempotence guard on push.** Before `kernel_push`, call `kernel_status`. If a version of
   that kernel is `RUNNING` or `QUEUED`, **refuse**. This is the direct fix for the
   `0.74` GPU-h launch race; make refusing the default path, not a check the caller opts into.
7. **No retry logic.** Ever. A failed op returns its receipt and stops. Retry policy is a
   human/registry decision (`KAGGLE_FAILURE_REGISTRY.json`), not a loop in this module.

## Tests (write first, run locally, no network)

Use a fake `kaggle` executable on `PATH` that echoes fixtures and records argv.

- Unknown op raises; argv is never string-interpolated.
- `competition_submit` raises without an `eligible: true` predicate receipt, and with an
  `eligible: false` one.
- Lock is exclusive: second acquirer fails while the first holds it.
- Lock releases on exception, not just on success.
- Stale-lock break refuses while `kernel_status` reports `RUNNING`.
- `kernel_push` refuses when a version is `RUNNING`/`QUEUED`.
- Receipts contain argv, hashes, and returncode; a fixture containing a fake token is
  **absent** from the written receipt.
- `bracketed_launch` aborts if the pre-query fails, and reports the delta correctly.

## Do not

- Do not put a credential, path to a credential, or environment dump in any receipt.
- Do not add `kaggle competitions leaderboard`, `kernels list`, or any op "for convenience".
  The plan is the security boundary; every addition needs a reason in the node report.
- Do not import this module into a kernel that runs on Kaggle. It is a local/cmux-side tool.

## Done when

`cmux_ops.py` exists beside the shared guard, the fake-`kaggle` suite passes with real output
pasted into the node report, and **zero external calls have been made.** Then `C1` may start.
