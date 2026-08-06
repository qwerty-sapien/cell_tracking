# C1 — asset upload and round-trip verification (autonomous, mutating)

**Precondition: `C0` complete and its tests pass. `K1`/`K2` have produced the staging
directory and the kernel file.** Runs through cmux. No GPU is spent here.

## What goes in the dataset

Slug: pick one **new** slug for this lane and register it **before** creating it —
`dataset_ownership` in `KAGGLE_FAILURE_REGISTRY.json`, in the same change (`KFG-17`), then:

```bash
python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
```

`FAIL_REGISTRY_STALE` must be fixed, never pushed around.

Payload:

| Object | Identity |
|---|---|
| Trackastra backbone | `assets/trackastra-authentic-v1/model.pt`, `24c290ce…981f`, `109,906,058` B, `27,456,880` params |
| DELETE population | manifest `4ab7bb23…76f0c`, `1,345,497` actions / `1,118` beneficial |
| Sealed B0 | `a9ce8c0b…b4ff2` |
| Outer split | the five frozen hash-fold labels carried by the population manifest (H0 Option A) |
| Frozen N1 compiler + scorer | `457f6607…24f5`, interface `2a72950c…f865` |
| Wheels | **canonical PEP-427 filenames** |

## Transport rules that have already cost this campaign runs

- **Wheel basenames must be canonical.** A content-hash-prefixed basename made pip reject the
  install *after* step 0 passed — `0.74` GPU-h, no science.
- **Never a bare `.pth`/`.pt` where byte identity matters.** Kaggle's pipeline transformed a
  checkpoint by `959,197` bytes. Use the **uncompressed tar** pattern that already recovered
  `f3e8e730…89ce5` exactly, with safe-ASCII member names.
- **No `+` or exotic characters in any filename.** Three launches died to mount-path mangling.
- **Resolve by content hash, not filename**, everywhere — Kaggle may expand archives during
  materialization.

## Sequence

1. Build the staging directory locally. Emit `ASSET_MANIFEST.json`: every member's path, byte
   size, SHA-256, and for tarred objects the member hash too.
2. `dataset_create` (first version) or `dataset_version` via `cmux_ops`, `-r skip`.
3. Poll `dataset_status` until `READY`. Bound the wait (30 min is the existing precedent) and
   fail closed on timeout rather than looping forever.
4. **Verify by download and hash — this is mandatory and non-substitutable.**
   `dataset_download --unzip` into a scratch dir, then hash **every** member against
   `ASSET_MANIFEST.json`.
   - **Do not use `datasets files` to verify.** It returns 403 under the owner session by
     design (endpoint scope). A listing is not verification anyway.
   - Account for Kaggle expansion: if a tar was expanded, verify the **members**; if a gzip
     was expanded, verify the deterministic recompression identity, normalizing the Python
     3.12+ gzip **OS byte** (that exact mismatch killed `SELECTOR-REQUIREMENT-CPU-V2`).
   - Any mismatch → **stop.** Do not push a kernel against an unverified mount. CELLECT v7
     died at step 0 because the payload at run time was not what a prior attestation claimed.
5. Write `ASSET_ROUNDTRIP_RECEIPT.json`: dataset ID, version, manifest hash, per-member
   verification results, the `dataset_status` receipt, and every `cmux_ops` receipt hash.

## Do not

- Do not upload the `ctc.zip` companion unless a node actually needs it — it is `101,957,234`
  bytes of payload the PEFT lane does not consume.
- Do not upload anything derived from a held fold, GT repair bundles, or scorer outcome
  fields. The leakage clause is `information_leakage_exact: 0`.
- Do not push a kernel in this node.

## Done when

Dataset is `READY`, **every** member verified byte-exact by download-and-hash, registry
freshness passes, and the round-trip receipt is written with all `cmux_ops` receipt hashes
cited. Then `C2` may start.
