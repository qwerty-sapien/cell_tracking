# Autonomous limits and scope (LMT)

Status: **BINDING — sole active authority for autonomous operational and resource limits**  
Effective: 2026-08-25  
Owner amendment: this document supersedes every older autonomous runtime, wall-clock,
polling, launch, external-action, local-memory, and workspace-growth limit in campaign
plans, DAG prompts, skills, and guards. The active envelope applies **only** to
`cell_tracking`-initiated local CPU runs and is based on the owner's reported 200 GB
remaining workspace and 24 GiB RAM: at most one quarter (50 GB) of that workspace and
one third (8 GiB) of RAM. It imposes no memory, storage, or artifact-size upper bound on
Kaggle kernel pushes or Kaggle compute. Older values remain historical evidence only.

## Precedence and boundary

1. A current explicit user instruction wins for the action it addresses.
2. This file controls autonomous scope, external effects, and compute/resource limits.
3. Project plans and preregistrations control scientific eligibility and acceptance criteria.
4. Historical receipts and logs describe what happened; they do not create current limits.

Scientific gates—score thresholds, leakage rules, held-out ordering, DEAD mechanisms, and
observation-distinctness—are not autonomy limits and remain where they are preregistered.
Physical platform capacities and local test timeouts are measurements, not grants of authority.

## Canonical machine policy

The JSON block below is the machine-readable authority consumed by launch and certification
tools. Change limits here, then update generated receipts/tests; do not copy numeric authority
into another active file.

<!-- LMT_POLICY_JSON_BEGIN -->
```json
{
  "schema": "cell-tracking.autonomy-limits.v1",
  "effective_date": "2026-08-25",
  "kaggle": {
    "lmt_memory_and_storage_upper_bounds": null,
    "local_resource_safety_policy_applies": false,
    "autonomous_projection_strictly_below_hours": 8.0,
    "weekly_gpu_budget_hours": 30.0,
    "auto_approved_wall_clock_kill_seconds": null,
    "projection_at_or_above_cap_requires_explicit_human_approval": true,
    "productive_candidate_required": true,
    "serial_launch_lock_required": true,
    "authoritative_quota_reconciliation_required": true,
    "exact_version_accounting_required": true,
    "one_shot_receipt_required": true,
    "threshold_evasion_by_sharding_forbidden": true
  },
  "external_scope": {
    "kaggle_dataset_and_kernel_actions_when_productive": true,
    "kaggle_competition_submission_when_scientifically_authorized": true,
    "paid_compute_auto_approved_if_conservative_max_usd_strictly_below": 3.0,
    "paid_compute_at_or_above_usd_requires_explicit_human_approval": true,
    "runpod_requires_explicit_human_approval": false,
    "paid_compute_threshold_evasion_by_sharding_forbidden": true,
    "other_external_writes_require_explicit_human_approval": true,
    "destructive_or_irrecoverable_actions_require_explicit_targeted_authority": true
  },
  "monitoring": {
    "default_mode": "terminal_or_atomic_wait",
    "maximum_unchanged_status_checks_per_hour": 1,
    "explicit_send_without_poll_instruction_overrides_default": true
  },
  "local_resource_safety": {
    "applies_only_to": "cell_tracking_initiated_local_cpu_runs",
    "applies_to_kaggle_kernel_push_or_compute": false,
    "owner_reported_free_workspace_gb": 200.0,
    "owner_reported_physical_ram_gib": 24.0,
    "maximum_workspace_growth_gb": 50.0,
    "idle_calibration_seconds": 45.0,
    "sample_interval_seconds": 0.25,
    "minimum_growth_allowance_mib": 6144.0,
    "measured_idle_growth_multiplier": 5.0,
    "sustained_growth_abort_seconds": 15.0,
    "immediate_growth_abort_mib": 8192.0,
    "maximum_combined_rss_mib": 8192.0
  }
}
```
<!-- LMT_POLICY_JSON_END -->

## Autonomous Kaggle scope

Codex may autonomously create or update Kaggle datasets, push kernels, monitor runs, retrieve
outputs, and make competition submissions only when the action is productive: it advances an
observation-distinct candidate or an approved scientific node whose preregistered gates have
passed. Quota expiry alone is never productive.

The local CPU memory and workspace-growth envelope does **not** apply to pushing a Kaggle kernel,
its packaged artifacts, or execution on Kaggle compute. LMT sets no upper bound on Kaggle memory,
storage, or artifact size. Physical Kaggle platform constraints may still determine whether an
artifact can be accepted or a run can complete, but those are provider capabilities—not LMT gates
and not reasons to block an otherwise authorized push.

For a whole-run authenticated projection **strictly below 8 hours**, Kaggle launch is
auto-approved. Such a run has **no agent-imposed per-run wall-clock kill**; an overrun is logged
and reconciled rather than terminated solely because an old estimate was crossed.

A projection **at or above 8 hours** is outside autonomous launch scope. Prepare a human approval
request containing expected effect size, compute necessity, failure modes, falsifying evidence,
and primary literature. Explicit approval defines that run's scope; do not split or shard work to
evade this boundary.

Every autonomous launch still requires the shared serial lock, authoritative quota refresh,
coverage by the 30 GPU-hour weekly ledger, exact package/input hashes, fresh-version accounting,
prelaunch certification, and a one-shot predicate/receipt. These integrity requirements are not
runtime sub-caps.

## Submission and monitoring scope

Competition submission is autonomous only when the project has already authorized that exact
scientific artifact and all local/hidden validation gates pass. Never substitute a merely
complete kernel output for submission eligibility.

Default monitoring waits on terminal or atomic evidence and performs at most one unchanged status
check per hour. If the user explicitly says to send without polling, stop after authenticated
launch confirmation. A later task may resume monitoring.

## Local and repository scope

Within the user-designated workspace, Codex may inspect, simulate, test, profile, edit, and create
reversible artifacts needed for the task. Preserve unrelated dirty-worktree changes and immutable
historical launch packages/receipts. A new remote version must use new certified bytes and, when
fresh-Version-1 accounting is required, a fresh slug.

For every `cell_tracking`-initiated **local CPU run**, local resource guards first measure 45 seconds
of idle growth, then use a 6 GiB sustained-growth allowance or five times measured idle growth,
whichever is greater. Sample every 0.25 seconds and abort after 15 seconds of sustained growth beyond
that allowance. Independently abort immediately if either combined live parent-plus-owned-child RSS
reaches 8 GiB or post-calibration growth reaches 8 GiB. The absolute combined-RSS ceiling controls
if the adaptive allowance would otherwise exceed it. Count concurrently live processes; do not add
nonconcurrent historical parent and child peaks or knowingly double-count shared mappings merely to
manufacture a breach.

A `cell_tracking`-initiated local CPU run may grow the user-designated workspace by at most 50 GB
(decimal) relative to its authenticated pre-operation baseline. This is an attributable-growth
limit, not permission to delete unrelated data or consume reserved space. Guards for materially
writing local CPU runs must record starting free space and attributable workspace size, check before
large writes, fail closed before crossing the limit, and clean disposable partial artifacts when
safe. The 50 GB workspace limit and 8 GiB combined-RSS ceiling implement the owner's
one-quarter-of-200-GB and one-third-of-24-GiB envelopes for those local CPU runs only.

These are the active host safety values because they are centralized above; no plan or skill may
shadow them. They do not apply to Kaggle kernel packaging/pushes or Kaggle compute, and they do not
create a universal runtime cap. Test scenario timeouts may remain local diagnostics but cannot
become Kaggle launch authority.

A single bounded paid-compute action is auto-approved only when its conservative maximum charge is
strictly below US$3, including Thunder or RunPod. Spend at or above that threshold requires explicit
human approval. Related work may not be split, sharded, or relabelled to evade the threshold; use the
conservative aggregate maximum for one scientific or operational objective. Provider credentials,
scientific eligibility, preregistration, integrity, and runtime/resource guards remain binding.
Other external writes, messages, purchases, deployments, and account mutations require explicit
authority unless a separate standing project authorization names them. Destructive or
irrecoverable actions require exact targets and targeted authority.

## Superseded limits

The following are revoked as active autonomous authority wherever they appear outside this file:

- OCC sliced-W1's 1-hour projection stop and 3,500-second soft stop;
- legacy 5-hour and 6-hour Kaggle launch/kill gates;
- fractional 60%, 70%, 75%, or 80% runtime/memory eligibility sub-caps;
- the 256 MiB sustained-growth and 384 MiB immediate-growth local RSS guards effective
  from 2026-08-10 through 2026-08-11;
- per-lane wall-clock limits used only to manage now-completed historical experiments; and
- any autonomous limit inferred solely from a historical reservation or receipt.

Do not mutate historical evidence to erase these values. Mark the old rule superseded, build any
future run as a new hash-bound package, and apply this file at certification and launch time.
