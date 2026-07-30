# SA-SWAP wave-1 terminal report

Status: **KILL_44B6**. Tier proposal: **PARK-C** for this exact instantiation; no subagent DEAD promotion.

The preregistered capacity-limited exact action-versus-no-op regret head did not improve the frozen APNCC 44b6 graph. Alpha changed all 36 videos but reduced adjusted edge Jaccard by `-0.0003102875275311501` and raw edge Jaccard by `-0.00029659222678224406`; node recall was unchanged. This crosses the preregistered kill line `adjusted delta <= 0`. The capacity-matched beta produced the same aggregate graph vector despite a different fragment aggregate and 14 fewer selected actions.

## Official graph vector

| Arm | Adjusted edge Jaccard | Delta | Raw edge Jaccard | Delta | Node recall | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Frozen APNCC | 0.8665345260361912 | — | 0.8625960574674240 | — | 0.9926180002844743 | — |
| Alpha, 49 features | 0.8662242385086600 | -0.0003102875275311501 | 0.8622994652406417 | -0.00029659222678224406 | 0.9926180002844743 | 0.0 |
| Beta, Trackastra margin zeroed | 0.8662242385086600 | -0.0003102875275311501 | 0.8622994652406417 | -0.00029659222678224406 | 0.9926180002844743 | 0.0 |

Alpha selected 3,322 actions; beta selected 3,308. Their fragment aggregates are respectively `ed5dcc062f41d6cd936ea4952262a294965a9be9d4898c52ad8ef0013a308485` and `d5398910b6c152de3752cc27f8a68eaf019ac615a0669bd8e7c002fd688f8df6`. Both changed all 36 graphs. Alpha/beta edge counts were identical (`TP=10320`, `FP=954`, `FN=694`) versus control (`TP=10327`, `FP=958`, `FN=687`).

## Population and N1 closure

- Source proposals: 84,883.
- Complete exact occupied-conflict actions: 45,961.
- Every candidate was compiled with frozen public `compile_action` on a locality-equivalent complete witness: 45,961 compiles, 0 rejects.
- Every nonempty final alpha and beta batch was compiled with frozen `compile_batch` against the full APNCC baseline.
- Invalid single actions: 0. Invalid batches: 0. Merges: 0. Dangling endpoints: 0.
- Exclusions: 5,261 already-present no-ops, 17,122 missing endpoints, 8,444 non-singly-occupied sources, and 8,095 non-singly-occupied targets.

The locality proof is frozen in `N1_LOCALITY_EQUIVALENCE.json` (`836812c72a51e915e74b9e57b987c21e3f915290feff76b01121481d664dc55c`). The streamed repair was bitwise-equivalent to prior in-memory fit semantics on deterministic fixtures and passed 18/18 guarded tests.

## Execution and scope

The accepted run followed the post-repair barrier exactly:

1. Repaired test preflight passed with zero swap growth.
2. All 18 tests passed under the continuous guard with zero swap growth.
3. Implementation/test/proof hashes were frozen in `POST_REPAIR_EXECUTION_MANIFEST.json`.
4. A later fresh 45-second family preflight passed.
5. The 44b6 family run completed under the continuous guard in 4,594.99 seconds, with zero swap growth and 0.8614 GiB peak process-tree RSS.

No sibling live output, SA-REPR output, C1, C2/K-best, Branch-B, CV57, validation12, audit, test, GPU, Kaggle, submission, RunPod, or external mutation was consumed or performed.

Because 44b6 did not GO, 6bba was not authorized and was not run. CV57 remains forbidden.

## Evidence

- Terminal result: `result_44b6_postrepair_v1/report.json`, SHA-256 `62953d270ae383c9bacdea9ab33a3f6980358a31a329b219447460abd5604864`.
- Family guard: `guard_44b6_postrepair_v1.json`, SHA-256 `a7379869a7ce9f3a2180f7ebe5c223a58dd1a792b6530ad2419f8d7398eafb2d`.
- Family preflight: `preflight_44b6_postrepair_v1.json`, SHA-256 `d6ac2f03a0c34c7b9382f396aa10db07d0eeea8f240ce8fdf78360dc6f366f71`.
- Preregistration: `PREREGISTRATION.json`, SHA-256 `00e6036ef076482a39266d797c5fea3e2e4845ba4262c8532cfe70bb8ccb4628`.

The exact wave-1 SA-SWAP instantiation should not be rerun. A future phase may revisit association expansion only through a newly named, independently frozen mechanism.
