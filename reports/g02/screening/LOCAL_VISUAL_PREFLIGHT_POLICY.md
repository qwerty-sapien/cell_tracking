# Local visual preflight for every planned tracking-model kernel

Effective 2026-08-14, a Kaggle kernel that executes or changes a tracking model is not launch-eligible until the exact candidate has run locally and its tracking behaviour has been reviewed in MP4 form. This gate is additional to the repository LMT, preregistration gates, serial launch lock, quota reconciliation, package certification, and terminal monitoring. It cannot open a branch that those controls keep closed.

## Binding order

1. Freeze the exact kernel package tree, code, checkpoint, input manifest, local execution environment, and preregistered identity set as repository-relative paths plus full lowercase SHA-256 values. The input manifest must use `celltrack-alpha.local-visual-input-manifest.v1` and carry the exact `preregistered_visual_identities` list. Identities use the strict `split:dataset` grammar with nonempty alphanumeric/underscore/hyphen tokens; separators, traversal, extra colons, whitespace, and control characters are rejected.
2. Run that exact candidate locally on the preregistered representative identity or identities. A shortened temporal slice is allowed for a cheap visual canary, but the receipt must say which slice was used.
3. Persist the candidate prediction artifact before rendering.
4. Seal a separate local-execution receipt with `execution_origin: local`, exit code 0, ordered start/end timestamps, and exact package/code/checkpoint/input/environment/prediction bindings. It is accepted only when it names a centrally trusted repository runner with an exact registered path and code hash and binds its argv hash, stdout/stderr paths and hashes, process timestamps/status, and exact prediction outputs. The central trusted-runner allowlist is intentionally empty until a runner receives explicit code review and hash registration; contract fields cannot grant trust. A self-authored JSON receipt, synthetic clip, or recovered remote output cannot substitute.
5. Render at least one MP4 with a JSON sidecar that repeats the model ID, local run ID, package/code/checkpoint/input/environment, prediction, identity, execution receipt, render timestamp, and MP4 bindings. Every MP4 must expose a nonzero video stream to `ffprobe` and complete a full `ffmpeg` decode; headers, truncated files, audio-only containers, and zero-frame media are rejected.
6. Visually review every video. Metrics-only inspection does not count. Each review is keyed by the exact `(identity, video_sha256)` pair, occurs after rendering, and records concrete observations for identity switches, fragmentation, hallucinated links, missed links, missed and false divisions, jump/drift failures, and border failures. The preregistered identity set must match exactly—without extras, omissions, or duplicates—across the bound input manifest, contract, execution and process prediction outputs, video records, sidecars, and reviews.
7. Seal `LOCAL_VISUAL_PREFLIGHT.json` after all reviews with verdict `PASS`. A visible defect may still pass when the experiment is deliberately testing that defect and the promotion rationale says why; corrupted, incoherent, stale, remotely substituted, or candidate-mismatched output cannot pass.
8. Run `kaggle_prelaunch_guard.py` from repository-relative contract and registry files. It re-hashes the complete package and all bound assets and rejects non-model claims unless the central allowlist contains the exact allowlist ID, kernel ID, execution profile, and code hash and static inspection finds no model-execution marker.
9. Only then may the normal Kaggle package certifier and serialized `cmux_ops.py kernel_push` boundary be considered. Under the shared lock, `cmux_ops.py` requires the guard project root to resolve exactly to the physical checkout containing `cmux_ops.py` and the registry to resolve exactly to that checkout's canonical `KAGGLE_FAILURE_REGISTRY.json`; this visual trust anchor never follows `CELLTRACK_SHARED_ROOT` or another environment variable. Arbitrary temporary roots and substitute registries are rejected. It then loads the real guard, replays it from current contract/registry bytes, requires exact equality with the stored predicate, re-hashes every package and visual artifact, checks remote status, and repeats the replay immediately before push. The legacy direct shell push wrapper is fail-closed.

## Coverage

- A baseline video cannot certify a changed candidate.
- A remote result cannot substitute for the preceding local run.
- A score, plot, CSV, or graph receipt without a reviewed MP4 cannot satisfy the gate.
- Infrastructure kernels that execute the tracker, including the V32 cache exporter, are model kernels and require this gate.
- A profile name or arbitrary `model_execution: false` flag is never an exemption. Only an exact centrally maintained allowlist entry bound to allowlist ID, kernel ID, profile, and code SHA is eligible, and any static model-execution marker still rejects it.
- For a scientific candidate, the default review set should include the named canary plus one high-residual `6bba` identity and one high-residual `44b6` identity. A one-video receipt is the mechanical minimum, not evidence of cross-family transfer.

## Receipt shape

The kernel contract declares:

```json
{
  "model_execution": true,
  "checkpoint_path": "assets/model.pt",
  "checkpoint_sha256": "<sha256>",
  "input_manifest_path": "assets/input-manifest.json",
  "input_manifest_sha256": "<sha256>",
  "execution_environment_path": "reports/.../environment.json",
  "execution_environment_sha256": "<sha256>",
  "preregistered_visual_identities": ["train:44b6_example"],
  "local_execution_command_sha256": "<sha256-of-canonical-argv>",
  "local_execution_receipt": {
    "path": "reports/.../LOCAL_EXECUTION.json",
    "sha256": "<sha256>"
  },
  "local_visual_preflight_receipt": {
    "path": "reports/.../LOCAL_VISUAL_PREFLIGHT.json",
    "sha256": "<sha256>"
  }
}
```

The visual receipt uses schema `celltrack-alpha.local-visual-preflight.v1`, status `PASS`, ordered timestamps, package/code/checkpoint/input/environment paths and hashes, the preregistered identity set, `model_id`, `local_run_id`, the local-execution receipt hash, a nonempty `videos` list, and an exact-set `reviews` list keyed by identity and video hash. The execution receipt uses `celltrack-alpha.local-model-execution.v1`, repeats the identity set, and embeds a `celltrack-alpha.local-process-output.v1` receipt plus trusted runner and argv bindings. Each video record binds repository-relative prediction, MP4, and sidecar paths and hashes. Each `celltrack-alpha.local-visual-sidecar.v1` sidecar repeats all candidate, run, asset, prediction, video, and execution-receipt bindings.

## Current video evidence

The canonical inventory is `reports/g04/TRACKING_VIDEO_INVENTORY.json`:

- 180 physical MP4 files;
- 57 distinct dataset identities: 21 `44b6` and 36 `6bba`;
- three complete 57-video encodes and one curated 9-video set;
- 171 distinct MP4 byte hashes because the curated nine are byte-identical copies from the older post-scoring full catalogue.

The inventory validator recognizes only the four registered catalogue/schema/classification tuples above, requires the manifest, index, physical MP4, metadata sidecar, and prediction-fragment identity sets to agree exactly, recomputes the fragment-set hash, validates sidecar frame/duration claims with `ffprobe`, and keeps all current catalogues classified as historical visual evidence (`local_execution: false`). These counts are derived from the authenticated exact sets rather than trusted manifest summaries.

The original 123 videos bind predictions from `reports/g02/selected_graph_appearance_direct_cv57_v1/fragments`; their complete index is `reports/g04/post_scoring_visual_review/full_catalog_20260804/index.html`. The 57 new videos in `reports/g04/v32_exact_visual_review_20260814/index.html` bind the hash-authenticated frozen V32 submission (`c61784c3…ab446`) and are the canonical visual diagnosis of the 0.911 model.

The V32 predictions were recovered from its historical exact Kaggle Version 1, not newly executed locally. They therefore diagnose V32 but cannot certify a changed candidate or satisfy this policy's local-execution requirement. No new V32-derived model kernel may launch until its own local-run receipt and candidate-specific MP4 evidence pass.
