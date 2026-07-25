# K — Build & push the 6bba Trackastra feature-generation Kaggle kernel

> Load `_shared_context.md`, `DAG.md`, and `campaign_narrowing_and_hypothesis_dag_v1.md`.
> This node produces the H1 feature cache on Kaggle GPU to escape the local host-swap
> deadlock. It is **frozen inference, not training**, and **not a competition
> submission** — it consumes Kaggle GPU quota, not a 5/day leaderboard slot.

## Authorization & scope (read first)

- **Human authorization is granted (2026-07-25) for this one Kaggle GPU
  feature-generation push.** It does not authorize any competition submission,
  RunPod launch, or GPU *training*; every other gate stays closed.
- **6bba only.** The pipeline is fail-closed to `train:6bba_` (69 videos). Do **not**
  touch 44b6 in this node — 44b6 has no feature pipeline yet and is a separate task.
- Kaggle free GPU quota (~8 h) resets in 1 h. Frozen inference is projected at
  **1.213 h** (`GA-TRACKASTRA-FULLPOOL`), so it fits — but budget dataset-upload and
  queue time. If the job cannot plausibly finish before the reset, **stop and report**;
  do not burn quota on a run the reset will guillotine.

## Hard facts about the pipeline (do not rediscover)

- The combined `scripts/build_trackastra_full_pool_feature_cache.py` is **disabled**
  (`two_stage_required: true` in the contract → it raises). Use the two-stage scripts:
  1. `scripts/build_trackastra_apncc_base_cache.py` — Stage A, V45 → 48-D base cache.
  2. `scripts/enrich_trackastra_apncc_cache.py` — Stage B, frozen Trackastra → 49-D.
- On Kaggle (16 GB RAM) the two stacks *can* co-reside, so the local zero-swap reason
  for isolation is moot — but **keep the two-stage sequence** because the contract
  (`e060b284…8ac`) requires it and the first 48 columns must stay byte-exact vs local
  v1. Run Stage A fully, then Stage B.
- Parity reference: identity `train:6bba_edf14583` (contract `parity_reference`). Its
  enriched feature SHA must match the authenticated local v1 first video exactly.

## Required inputs → Kaggle datasets (no internet in-kernel)

Enumerate the exact pinned inputs and SHAs the two local scripts consume, then make
each available on Kaggle. Reuse existing datasets where possible; create the rest with
`kaggle datasets create -p <dir>` and record every slug:

- frozen Trackastra checkpoint zip + member (the 27.46M zero-wrfeat checkpoint under
  `external/Tracking_Support/`) → `--checkpoint-zip`, `--checkpoint-member`.
- source tree: `src/celltrack_alpha`, `scripts/` (run_bh, sweep_bh_graph_weights,
  trackastra_checkpoint_loader, the two builders), and the Trackastra repo src on the
  path → `--source-root`.
- support pack (already on Kaggle: `pilkwang/biohub-tracking-support-pack-50ep-v1`) →
  `--support-root`.
- sparse edge scorer artifacts → `--edge-artifact`, `--edge-auxiliary`.
- raw proposal archive `--zip` + `prepare.json`.
- fit-train105 6bba zarr caches → `--cache`.
- the two-stage contract JSON + its SHA-256 → `--contract`, `--contract-sha256`.

## Build steps

1. **Manifest.** Record every input path, SHA-256, and the exact local two-stage
   invocation (read both scripts' argparse). Write it to
   `reports/g02/trackastra_6bba_feature_kernel_manifest_v1.json`.
2. **Datasets.** Create/verify the Kaggle datasets above; capture slugs and versions.
3. **Kernel script** `kaggle_kernel/trackastra-6bba-feat/trackastra_feat.py`:
   set `sys.path` to the mounted dataset roots; run Stage A over all 69 6bba videos,
   then Stage B; write the 49-D cache + per-video authenticated checkpoints to
   `/kaggle/working`; print a final manifest (69/69 count, per-video SHAs, elapsed,
   parity-reference SHA). Fail closed on any identity/contract mismatch.
4. **kernel-metadata.json** in that dir:
   ```json
   {
     "id": "onebitbrain/trackastra-6bba-feat",
     "title": "trackastra-6bba-feat",
     "code_file": "trackastra_feat.py",
     "language": "python",
     "kernel_type": "script",
     "is_private": true,
     "enable_gpu": true,
     "enable_internet": false,
     "machine_shape": "NvidiaTeslaT4",
     "competition_sources": ["biohub-cell-tracking-during-development"],
     "dataset_sources": ["<all created slugs>"]
   }
   ```
5. **Local parity gate (before push).** Run the two-stage scripts locally on the single
   parity-reference video `train:6bba_edf14583` via the *same code path* the kernel
   uses; confirm byte/numeric parity of its 49-D feature vs authenticated local v1.
   Add a focused test. **Do not push if parity fails.**
6. **Push:** `kaggle kernels push -p kaggle_kernel/trackastra-6bba-feat`. Poll status;
   capture the version id and logs.
7. **Retrieve & verify.** Download the output; assert 69/69 videos, first-video parity
   vs local v1, and all contract/identity SHAs. Preserve the run archive **without
   expanding it** unless analysis requires it.
8. **Bank.** Add LEDGER row `GA-TRACKASTRA-6BBA-FEAT-20260725` with runtime, GPU-hours
   consumed, parity result, and dataset/version identities; commit locally. This cache
   feeds H1's held-fit graph gate (the 44b6 extension is a separate downstream node).

## Fail-closed / stop conditions

- Any identity, contract, or first-video parity mismatch → abort; do not use the cache.
- Kaggle run errors, or cannot finish before the 1 h quota reset → stop and report;
  never spend quota on an un-finishable run.
- Requires configured Kaggle API credentials; if absent, stop and report — do not
  attempt alternative upload paths.
- Nothing here authorizes a competition submission or GPU training.

## Output

`reports/g02/screening/trackastra_6bba_feature_kernel_v1.md` — the manifest, dataset
slugs, kernel id/version, runtime/GPU-hours, parity result, and the LEDGER entry.
