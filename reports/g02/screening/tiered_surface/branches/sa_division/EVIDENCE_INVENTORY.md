# SA-DIVISION frozen evidence inventory

Date: 2026-07-28  
Status at preregistration freeze: execution pins authenticated; candidate-arm
outcomes not yet inspected.

## Frozen execution inputs

| Input | Absolute frozen root path | Expected SHA-256 | Authentication |
|---|---|---|---|
| N1 interface | `/Users/agent/Desktop/cell_tracking/reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json` | `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865` | MATCH |
| N1 implementation | `/Users/agent/Desktop/cell_tracking/src/celltrack_alpha/action_compiler.py` | `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5` | MATCH |
| grouped folds | `/Users/agent/Desktop/cell_tracking/reports/g02/grouped_3fold_audit20_v1.json` | `ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769` | MATCH |
| P0 oracle | `/Users/agent/Desktop/cell_tracking/reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json` | `8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5f18c8889c06` | MATCH |
| fit provenance | `/Users/agent/Desktop/cell_tracking/reports/g02/fit_provenance_v1.json` | `6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f` | MATCH |

## Governance pins read before preregistration

| Artifact | SHA-256 |
|---|---|
| human-locked plan | `a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116` |
| `SCOPE_MANIFEST.md` | `066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49` |
| `ALIGNMENT.md` | `eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37` |
| `TIERED_EXPLORATION_DAG.md` | `e1e0f45cb6b85cab75ea898a9857bb3e6cebb4fe288a2b60466efb734c2a8671` |
| `N4_EVENT_NODE_ACTIONS.md` | `c938316a51a19c2fc1b713d01ebd1d20c040c17342002a336f24f74a7ff00571` |
| event/node register | `6df46ab0b3744a89b588d55d24e71a687f70e261667491adf6e2552c1b5fb5b3` |
| `LIVE_STATE.md` at lane start | `d6c7ca832109876b17cfb56483502a9dd022cc50b52f38105362db8561b9eab8` |

## Outcome-bearing inventory

Inspected only after `DEPENDENCY_PREREGISTRATION.md` was frozen.

| Immutable arm | Banked artifact @ SHA-256 | `44b6` adjusted/raw/node | `6bba` adjusted/raw/node | Eligible |
|---|---|---|---|---|
| `GA-H1-ZWRFEAT-GRAPH/points-only-zero-wrfeat-12` | `reports/g02/screening/ga_h1_zero_wrfeat_graph/report.json` @ `94a3cd006cd40aeb45f1773f6e198767133c3724eed720f573b1e3acd6851c9a` | `-0.011148076581064648 / -0.011057532129446046 / -0.00028790090722474737` | `null / null / null` (not run) | no |
| `C1-TA-HOCT-LITE` | `reports/g02/screening/top50_parallel_dags/branches/c1_ta_hoct_lite/outputs/c1_n3_44b6_graph.json` @ `6a202b209c9c1b7851ca1e43e4c67e5dc3772e78426fdc267f1c01cfe5953a10` | `-0.007602934501881209 / -0.007591069106931747 / -0.00018539307341891575` | `null / null / null` (not run) | no |
| `U0-T4CANON` | `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/u0_t4canon/terminal_report.md` @ `c29d52abd4b48c349818b0e761d8fd2b338da3e23b49164d31c86e11486f0151` | `null / null / null` (governance only) | `null / null / null` | no |
| `C2-KBEST-STABILITY` | `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/c2_kbest_stability/terminal_report.md` @ `259edb515125bd5616c3df8fcc07c4aa93a35afed895fbb61151612378a0b0a6` | `null / null / null` (not scored) | `null / null / null` | no |

APNCC is the unchanged graph beta, not an incremental qualifying arm. The frozen
D1 dependency artifact independently records the one-arm indivisibility rule:
`reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/d1_persistent_fork/N2_association_dependency.md`
@ `9ead4f6dff91c5eadf651ee711a3d73bea31b41f0528d7393a116f0a79418113`.

**Qualifying banked association arms: 0.** Division fitting therefore remains
closed. Memo C2/K-best values and Branch-B `66/223` remain quarantined and were not
used.

## Fixture audit artifacts

| Artifact | SHA-256 | Result |
|---|---|---|
| `DEPENDENCY_PREREGISTRATION.md` | `e7ab847530d72ca69a5b55b3e480c26e62a89599d98f5c40abff384a3528f1c1` | frozen before arm outcomes |
| `fixture_audit.py` | `717ee98343b47ca5781175b99e48424fe53b1c1ae97815241025a4ea1a253123` | fixture-only implementation |
| `guarded_results.json` | `38990d30d4b30ea7afbf99e0a655b0150c34a73fdd4918f93178316d9d9e3c76` | `PASS_PARK_DEPENDENCY`, 7/7 |
| `fixture_preflight_retry1.json` | `66ccd0287153fa7d6d0547a72993f35d450746fdee6cfd90871115a42f27b9f7` | 45.638 s, zero swap growth |
| `fixture_guard_retry1.json` | `3cf2644ce0ef19ada6c53510dbe1d42d01a81aa9aff934d20acf778091793522` | PASS, 1.833 s, peak RSS 0.075 GiB, zero swap growth |

The first guarded attempt failed closed because the test compared the runtime N1
interface to the frozen interface including freeze-time metadata fields. Five
structural tests had passed and swap growth was zero. The corrected comparison checks
the runtime field subset plus the pinned implementation hash; the fresh-preflight
retry passed all tests.

