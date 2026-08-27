# OCC sliced-W1 repair2 — adversarial prelaunch review

Date: 2026-08-10  
Kernel: `onebitbrain/g02-occ-sliced-w1-repair2-v1`  
Decision: **ELIGIBLE FOR ONE VERSION-1 SEND; DO NOT POLL IN THIS TURN**

## Exact identity

- Entrypoint SHA-256: `a15ad1a84f1c42e3ab34d592e6ef56b01704461006c0e11d865ca0eae111264d`
- Metadata SHA-256: `426d04e1eabc18909da10ba1554837403e460a2fcd11cb70f122d75b795f0024`
- Package inventory SHA-256: `58fe85185a10f107478eea15de5d0aa0a745ed590a06adcdb35df7ccd9040b65`
- Profile SHA-256: `81bbb076323d052d92cbeebca45fda2a984254d305efa41970708fa769c85cd5`
- Root `LMT.md` SHA-256: `10c6eb8536fdbe2210f25cabc50784aed397b68e82b076d48ebbbdd58ddee787`
- Certificate content SHA-256: `808587da174f6733a374f2f6491f1d43bfebb90a1c0abfaa79b93b284dbd8c62`
- Inline guard raw-file SHA-256: `98e76ae37574e58e89af85a722e350948f7bb9816ec92a2fac8ef0870f1acd7c`

## Evidence

The complete `ship-kaggle-kernels` ladder reports `ELIGIBLE` and autonomous launch
eligibility. Eight normal simulations—tiny, stratified, largest-video stress, and a
six-video expansive sample, each repeated twice—returned code 0, the declared
workload count, the complete marker, and byte-identical semantic receipts within each
scenario. Four injected launch faults passed: missing mount, missing required file,
and corrupted required file all failed before heavy work; a duplicate basename did
not perturb exact direct-path resolution.

The projected remote upper bound is `3,208.536531 s = 0.891261 h`, with no internal
wall-clock kill. Projected peak RSS is `4,514,723,080 bytes` against a 16-GiB profile.
The only declared residual platform risk is the local macOS versus remote Linux ABI;
repair1 already proved the exact pinned manylinux wheels install and Step 0 execute on
Kaggle Python 3.12. This package keeps those wheels and scientific calculations exact.

The independent breaker review found and repaired a successful `SystemExit(0)` that
the broad logger had mislabeled as `_run_failure`. The exact recertification has zero
normal-run `Traceback` hits and zero normal-run `_run_failure` hits. Fourteen focused
scientific/parity checks, three exact Step-0 carrier/tamper checks, Ruff, JSON parsing,
`git diff --check`, the certifier's post-run `check`, and the inline launch guard pass.

## Scope

The output is a fit-side feature supplier only: 45,961 frozen occupied-swap actions,
14 unchanged raw 3-D sliced-W1 features plus the existing A0 control. It does not fit
a selector, score a graph, create a competition artifact, or authorize submission.
The one-shot CMUX action must use a fresh Version-1 slug, a `0.01 h` administrative
ledger reservation despite CPU-only metadata, and authenticated projection
`0.891261 h`. Stop after the push receipt; do not poll.

