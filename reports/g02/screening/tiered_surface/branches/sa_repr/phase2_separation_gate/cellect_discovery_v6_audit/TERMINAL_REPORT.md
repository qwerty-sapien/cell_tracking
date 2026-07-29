# CELLECT dataset-v6 discovery N2 — independent terminal audit

Status: **FAIL CLOSED — shared-kernel repair required before N3**

The owner-downloaded v6 payload is present, positive, and complete. Independent
content-addressed discovery found exactly one bundle manifest at
`ae3bc9b2…7fecd3`, exactly one request manifest at `82111af5…bc83`, and 77
unique request payloads totaling 1,391,458 actions:

| Frozen class | Request files | Actions |
|---|---:|---:|
| delete | 41 | 1,345,497 |
| occupied swap | 36 | 45,961 |

The prototype returns structured records with `action_class`, `identity`,
`action_count`, byte count, SHA-256, and resolved path. It passes after every
downloaded object is exposed only through opaque filenames and arbitrary
parents. Its seven tests cover the positive case and missing, empty, duplicate,
and wrong-hash manifests/payloads. The unchanged shared suite also passes
21/21.

N2 does **not** yet clear N3. The current shared step 0:

1. finds `consolidated_bundle_manifest.json` by exact filename;
2. resolves request payloads as `bundle_root / manifest_record["path"]`; and
3. returns bare paths, after which the kernel derives `action_class` from the
   parent directory and `identity` from the filename stem.

On the current owner download those canonical names happen to work and the
shared step 0 returns all 77 requests before heavy imports. On the opaque
name/parent layout, the independent prototype still passes but shared step 0
fails with `expected exactly one consolidated_bundle_manifest.json, found 0`.
This is exactly the transport dependency N2 was required to eliminate.

The orchestrator patch is mechanical and science-preserving: port the
content-addressed unique SHA+byte resolver into shared pure-stdlib step 0; pin
and resolve the request manifest by hash; return the structured records; consume
their frozen class/identity in the kernel; and add the independent opaque-name,
missing, empty, duplicate, and wrong-hash fixtures to the shared suite. Then
rerun both actual-v6 and opaque-layout preflights.

No authenticated or network command, Kaggle authorization/launch, GPU debit,
training, full cache, graph, 6bba, CV57, submission, RunPod, tier move,
threshold change, plan amendment, or DEAD action occurred. No authenticated v6
end-to-end runtime projection exists yet; the historical 0.0545-second receipt
is step-0 timing only, not a `<5 h` GPU projection.
