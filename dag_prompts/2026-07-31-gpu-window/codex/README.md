# Codex pack — autonomous cmux control

**Audience: Codex, running unattended.** Claude authored this; both agents share
`reports/g02/screening/LIVE_STATE.md`. Read it first, and update it after every state change.

> **⚠ Read [`STATUS.md`](STATUS.md) first.** The §6.2 PEFT lane this pack was originally
> written against **completed and FAILED** on 2026-07-31 (`0.04` T4-h, adjusted
> `+0.000133827` vs `+0.005`, PARK-C). `C1`–`C4` are **lane-parameterized** — substitute the
> slug, kernel dir, contract, projection ceiling, and hard kill of whichever node is
> authorized. `C0` was never built and is still needed. The next target is behind a
> locked-plan amendment barrier (`DECISION_DOSSIER.md`, Option 1 vs Option 2) that **no agent
> may resolve**.

## Your authorization, stated exactly

`AGENTS.md`, standing owner authorization dated 2026-07-31:

> Kaggle kernel launch, competition submission, and monitoring are **pre-authorized** when an
> observation-distinct candidate or approved experimental node makes the action productive.
> **Do not ask again for Kaggle access in that case.** Quota expiry alone is not productivity;
> preserve the locked-plan/preregistration gates, shared serial lock, authoritative quota
> reconciliation, prelaunch integrity checks, hard runtime kill, exact version accounting, and
> terminal monitoring. RunPod still requires explicit human approval.

Plus, recorded 2026-07-31 in `LIVE_STATE.md` and
`sa_peft_adapter_v1/PREREGISTRATION_AMENDMENT_A.json`:

- the owner **overrode** `TERMINAL_CAMPAIGN_REPORT.md` for `sa_peft_adapter_v1` **only**;
- the owner chose **H0 Option A** (five frozen hash folds, `8.0` T4-h approved above the
  `5` h auto-ceiling).

That authorization was exercised: `sa_peft_adapter_v1` ran and is now terminal
(see [`STATUS.md`](STATUS.md)). The standing authorization persists for **the next** approved
node — once the owner resolves the `DECISION_DOSSIER.md` barrier. For that node, launch and
monitoring are pre-authorized: **do not stop to ask for Kaggle access.** An *approved* node is
one the owner has named; the standing authorization removes the access prompt, not the
requirement that a node be approved.

Submission is authorized by the same sentence but only under its own predicate — see
[`C4_SUBMISSION_PREDICATE.md`](C4_SUBMISSION_PREDICATE.md). Evaluate the predicate; do not
substitute your judgement for it in either direction.

**RunPod remains forbidden.** No other lane is reopened.

## The doctrine: bounded command plans, never free-form shell

You are driving the **owner's real authenticated session**. The rule, which
`run_cellect_cmux_roundtrip.py` already established in this repo and which you must follow:

> Never type ad-hoc Kaggle commands into the cmux shell. Invoke a script whose complete
> external command surface is enumerated in advance as a `command_plan()` dict, and which
> writes a receipt for every invocation.

Why this is not bureaucracy: an enumerated plan is reviewable before it runs, reproducible
after, and impossible to typo into a destructive variant. Free-form typing into an
authenticated session is how a one-run authorization became two byte-identical versions and
a `0.74` GPU-h debit.

## Auth facts you must not rediscover the hard way

- The **agent-shell token** (`~/.kaggle/credentials.json`) is not the owner session. Use cmux.
- `ListDatasetFiles` and `GetDatasetMetadata` return **403 even under a refreshed owner
  session**. This is endpoint-specific OAuth scope, **not** entitlement and **not** rules.
  **Never tell the owner to accept the competition rules** — that was done ~2026-07-08 and
  asserting otherwise contradicts banked evidence (`Q-APNCC-PUBLIC` in `LEDGER.md`).
- Therefore: **never gate a workflow on a file listing.** Verify remote assets by
  **downloading and hashing** (SHA-256 of container and member). `datasets status`,
  `datasets version`, and authenticated SDK **download** all work.
- Kaggle may **expand an uploaded uncompressed tar** during materialization. Resolve files by
  **content hash, not filename**. A literal `+` in a filename mangles on the mount path and
  already cost three kernel launches.

## Node order

```
C0  cmux ops module + serial lock + receipts     no external calls
C1  asset upload and round-trip verification     datasets version/status/download
C2  launch                                        guard receipts, quota, push, version read-back
C3  monitor to terminal                           poll, hard kill, output, quota, debit
C4  submission predicate                          evaluate; act only if it evaluates true
```

`C0` first, always. `C1`..`C3` are the autonomous loop. Each node writes its receipts before
the next may start.

## Non-negotiables

1. **One in-flight kernel, enforced by a lockfile**, not by intention (`C0`).
2. **Fresh guard receipt per push** — `check_registry_freshness.py`, then
   `kaggle_prelaunch_guard.py --phase static`, then `--phase launch` bound to the exact
   kernel ID. Exit `0` and `"status": "PASS"`. Cite path **and SHA-256**.
3. **Quota re-queried immediately before and after every launch**, owner-authenticated. The
   **pre/post delta is the debit** — never your own timing.
4. **Exact version accounting.** Read back the kernel ID and version Kaggle actually assigned
   and assert it matches what you pushed (`KFG-15`/`KFG-16`: Kaggle has silently accepted a
   run under a drifted slug).
5. **Hard runtime kill** at the preregistered wall clock. A trip means quarantined partials
   and no verdict — not a restart.
6. **Same-signature retry is forbidden.** New signature → add the `KFG-N` family *and* a
   regression fixture in the change that fixes it, before retrying.
7. **Never** touch the sealed `PREREGISTRATION.json` (`0d96b336…1ad8`), the frozen scorer,
   the N1 compiler, the sealed B0 (`a9ce8c0b…b4ff2`), or the population manifest
   (`4ab7bb23…76f0c`).
8. **Secrets never leave cmux.** No credential, token, or key may be echoed into a receipt,
   log, report, or commit. Receipts store command **argv and output hashes**, and stdout only
   after scrubbing.

## Reporting

Every node appends to `LIVE_STATE.md` (dated, newest first, short) and writes machine-readable
receipts next to the node. State outcomes plainly: if a push failed, say so with the real
output; if a step was skipped, say that. A terminal FAIL on this node is a legitimate result
that closes the plan's last named escalation — report it as such, not as a setback to retry
around.
