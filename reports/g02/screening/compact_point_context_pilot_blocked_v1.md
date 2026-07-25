# Compact point-context local pilot — runtime blocked

Status: **BLOCKED-RUNTIME; no model evidence**.

The candidate manifest and dry-data audit passed. The guarded 500-step
fit-only pilot was then attempted twice, but the locked zero-swap-growth rule
terminated both launches before optimization began:

| attempt | elapsed | peak child RSS | free-memory floor | swap growth | stop |
|---|---:|---:|---:|---:|---|
| 1 | 10.53 s | 0.728 GiB | 32% | +1.50 MiB | swap growth |
| 2 | 2.13 s | 0.217 GiB | 32% | +80.75 MiB | swap growth |

The host already had about 21.5 GiB of swap in use. A read-only process audit
showed unrelated active repository-diff, Next server, and desktop workloads.
They were not interrupted. No model report or checkpoint was written, no
optimization step completed, and no accuracy decision is admissible.

The candidate remains `PASS` for the local pilot. Retry only after host swap
pressure is stable; do not relax `swap_growth_allowed: false`. CV57, Kaggle,
GPU, and RunPod remain closed.
