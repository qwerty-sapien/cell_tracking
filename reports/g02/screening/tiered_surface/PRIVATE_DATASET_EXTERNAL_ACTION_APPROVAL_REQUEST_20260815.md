# Private Kaggle dataset external-action approval request

Date: 2026-08-15

## Requested scope

Approve exactly these two private Kaggle dataset mutations and their read-only
status/file/download checks for byte-exact round-trip validation:

1. Create private Dataset Version 1 at
   `onebitbrain/g02-cellect-zarr-cv2-runtime-deps-v1` from
   `dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/division/d2k5_zarr_cv2_dependency_carrier/payload`.
   The payload is approximately 60.7 MB and contains ten CPython-3.12 Linux
   wheels, their license texts, an exact requirements lock, and a carrier
   manifest. It contains no competition images, labels, scores, submissions,
   predictions, or model checkpoints. Payload inventory SHA-256:
   `e9932c698fdbd50529a1330c8fa0e60b41d552d680a24a0639065778054423cf`.

2. Create private Dataset Version 2 at the existing slug
   `onebitbrain/g02-linajea-zf-tf1-gpu-runtime-v3` from
   `dag_prompts/2026-08-14-multisubstrate-alpha-portfolio-v2/exploration/linajea_zf_gpu_visual_v2/gpu_v3/out/carrier_dataset_v4_r3`.
   The upload contains exactly the outer manifest plus the opaque
   `1,137,123,888`-byte runtime payload. Payload SHA-256:
   `b25bc3ed999ac105c5b7c22bb7d5ac1cdd4eb032e2b3ad7e83cdde362a39cd28`;
   outer-manifest SHA-256:
   `fb322e27f58180c64605c7b56123814b2febc9bb415ce8d9fda1afda469cfc13`.
   This is the frozen released zebrafish Linajea runtime, graphs, and weights;
   it contains no competition images, competition labels, scores,
   submissions, or predictions.

Both datasets remain private. Mutations must use the repository serial lock,
authenticated Kaggle CLI, exact version accounting, receipted pre/post quota
queries, and fresh downloaded round-trip verification of paths, byte counts,
and SHA-256. Any upload drift, archive expansion, missing file, unexpected
extra file, or non-ready status stops the launch sequence.

## Downstream boundary

This approval does not authorize scoring or competition submission. After both
round-trips pass, the already-authorized kernel-launch workflow may seal the
exact central allowlist records and push these two visual-only kernels serially:

- `onebitbrain/g02-cellect-native-gpu-visual-v2`
- `onebitbrain/g02-linajea-zf-tf1-gpu-visual-v3`

Their GPU executions may overlap. CELLECT must return two decoded MP4s and
Linajea six decoded MP4s for local visual review before any scorer, full38,
hybrid, or promotion gate opens.

