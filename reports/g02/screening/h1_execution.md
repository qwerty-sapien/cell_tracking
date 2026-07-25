# H1 execution — launch-integrity gate

Verdict: **STOP / RESERVE QUOTA — prerequisite gate failed before upload or
timing smoke.**

The human authorization was recorded, but H1 could not enter the Kaggle timing
smoke. The attached execution contract requires the authenticated H3 real-mask
or aligned `wrfeat-regionprops2` arm, verified remote datasets, and a verified
kernel before the smoke. Those prerequisites are absent. No dataset, kernel,
GPU job, feature cache, graph gate, CV57 run, or public submission was launched.

## Gate evidence

- Scope audit of the authenticated local manifests:
  - raw fit-train105: 10,710 files, top-level scope `train`, zero paths matching
    `mask`, `wrfeat`, `regionprop`, `label`, or `segment`;
  - sparse cache: 210 files, zero such paths;
  - Option-C assets overlay: 3 files, zero such paths.
- The banked H1-PREP evidence independently states that neither authenticated
  per-detection instance masks nor aligned 12-D `wrfeat-regionprops2` arrays
  exist for both 44b6 and 6bba. Pseudo-masks are forbidden.
- Authenticated read-only Kaggle checks with CLI 2.2.3:
  - `onebitbrain/trackastra-6bba-assets-v1`: `ready`;
  - `onebitbrain/trackastra-h1-option-c-assets-v1`: unavailable, and an
    owner-scoped `trackastra-h1` dataset listing returned `No datasets found`;
  - `onebitbrain/trackastra-h1-fit105-cache-v1`: unavailable;
  - `onebitbrain/trackastra-h1-feature-cache`: owner-scoped kernel listing
    returned `Not found`.
- Fresh 45-second clean-idle preflights observed swap
  `0.00 MiB -> 0.00 MiB`.
- Focused fail-closed package verification passed:
  `PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python -m unittest -v
  tests.test_h1_feature_cache_prep` — 12/12 tests.

## Preregistered metrics

No held-fit graph arm was legally reachable:

| family / arm | adjusted edge-Jaccard delta | raw delta | node-recall delta | verdict |
|---|---:|---:|---:|---|
| 44b6 zero-wrfeat | not run | not run | not run | prerequisite STOP |
| 44b6 real wrfeat | not run | not run | not run | authenticated input absent |
| 6bba zero-wrfeat | not run | not run | not run | 44b6-first gate not reached |
| 6bba real wrfeat | not run | not run | not run | 44b6-first gate not reached |

CV57 and the public probe were not reached. Kaggle GPU-hours are `0`; new
submissions are `0`; RunPod usage is `0 h / $0`.

## Frozen package identities

- kernel:
  `4106c0cb7d53f192b1788f23fb9009a4cc41af806dc9c4c0f88fbc10913bb71a`
- kernel metadata:
  `c950cae960214b08a1d1e7f7eedd88ba549eadfc51637b3626c98ce6d31bfbf8`
- Option-C contract:
  `9f2247cf69cd431840395162ec057770081689a11e964ccb27ff4bd4aee6f6b7`
- assets manifest:
  `2744214e9298f7091aa7b7618e6188c88ae6fe6550306b3544b70167a7ce4e03`
- sparse-cache manifest:
  `ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28`
- raw fit-train105 manifest:
  `665c505b4e38a03f5a501527fde58a064d3ce8fac413898656a4f3f3e6cf781e`
- source-dependency manifest:
  `fadc47e81d7290e5d723ae5da279382f1218e4ce138aa0d5712cb88990feab05`
- H1-PREP JSON:
  `2e1aad547febdc9212f257ac038b555d30b390666132b9d1f4d734d3f429da82`
- focused test source:
  `9ecbce8338bd95b875a7fec7e12a7e436c33ea045b5c1a97d097cecad39c4f02`

## Resume condition

H1 remains PARKed at the launch-integrity boundary. A future execution requires
a new authenticated, node-aligned real H3 mask/wrfeat artifact covering all
fit-train105 identities in both families, plus created and version-verified H1
remote datasets and kernel. Only then may a fresh two-family T4 timing smoke
establish repeatability, a frozen M5-to-T4 tolerance, and the <=7-hour
projection. This goal ends at the mandated STOP; H2 remains closed.
