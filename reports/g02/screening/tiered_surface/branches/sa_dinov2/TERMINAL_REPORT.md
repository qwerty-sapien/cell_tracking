# SA-DINOV2 Wave-3 terminal report

Date: 2026-07-29 (Asia/Singapore)  
Verdict: **R4 AUTHENTICATED; R5 XY-BETA STAGED; NO SCIENTIFIC OUTCOME**

The official DINOv2 ViT-S/14 bundle is provenance-complete. Source is pinned to
official `facebookresearch/dinov2` commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` and tree
`2a27257b79b0633b027a21014bc9360e3c1b3f43`. The exact Apache-2.0
license is SHA-256
`600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7`.
The pinned source enumerates the official checkpoint URL used here.

The 88,283,115-byte `dinov2_vits14_pretrain.pth` is pinned at SHA-256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
It is stored outside Git at the content-addressed, read-only local path recorded in
`WEIGHT_PIN_MANIFEST.json`; the large checkpoint is not committed.

Validation hashes the checkpoint before deserialization and then uses
`torch.load(..., weights_only=True, mmap=True, map_location="cpu")`. The root is a
string-to-tensor dictionary with exactly 175 keys and 22,056,576 tensor elements.
The pinned official model also has exactly 22,056,576 parameters. Strict loading
has zero missing, unexpected, or shape-mismatched keys, and zero nonfinite tensors.
A repeated synthetic CPU forward is byte-exact and yields the frozen R5 interface:
384-D CLS, 256×384 patch tokens, and a 768-D CLS-plus-central-2×2 embedding.

This satisfies R4's previously missing encoder-provenance dependency. R5 is staged
as supplier `F0-DINOV2-VITS14-XY-V1`, independently of CELLECT, against the
existing exact 44b6 delete/occupied-swap population, sealed B0, five folds,
per-class verdicts, A1/A0 alignment control, deterministic ridge probe, and all
unchanged AUC/retention floors. Its outcome-blind eight-scalar endpoint-pair
featurizer and permutation control pass 6/6 focused tests; the safe-loader passes
6/6 focused tests.

R4 and R5 remain PARK-B with no tier move proposed. That is deliberate: provenance
is now banked, but no authentic G-02 endpoint feature, separation metric, graph,
6bba, or CV57 outcome was opened. R6 remains blocked on an actual frozen R5 XY
beta cache.

## Newly unblocked

- R4 authenticated provenance now permits a separately authorized bounded R5
  endpoint-feature materialization.
- DINOv2 is registered as the second independent supplier for the unchanged
  separation gate alongside CELLECT.

## Artifacts

- `PREREGISTRATION.json` — `ae481d29c62a3c5e11305230b582e7c9ab6492b665f32b319c43b007ae38611c`
- `SOURCE_LICENSE_RECEIPT.json` — `fd9edd0102aed6c4c1e8e53c7cbee6dc1e322c416f4873be9bc539d70aa4ba24`
- `WEIGHT_PIN_MANIFEST.json` — `424071779e1e8ddea305fc5b499a41a88ab348e95eeef7364bb09129f5a94c4b`
- `SAFE_LOAD_RESULT.json` — `b7efbda4031c91a410a8f022350330c6b03f8a254703fce8578dcd990e723ee5`
- `validate_dinov2_vits14.py` — `8481d598aaecc27c49a79b1b9bb60184a3c873d214f117b8e8524a0534996566`
- `test_validate_dinov2_vits14.py` — `0efed86c8862252d793f46d7ff57ec818cd1e1ed4712517b067ceeb7984f40a9`
- `R5_XY_BETA_STAGING_CONTRACT.json` — `c8e3df7739a3691b6ad9c93234ecc8db9696992c126de3ff343def38a93d5610`
- `dinov2_xy_action_featurizer.py` — `f93a202a7e662859f92442a4a6932ad371af513c65627838f2ebfd7bfecb9b49`
- `test_dinov2_xy_action_featurizer.py` — `bff9231949a01d0bc6372d5611070b1d0ba661cf913215dc740b688a7b36bc86`

External usage: one explicitly authorized official source clone and one explicitly
authorized official checkpoint download. GPU 0 h; Kaggle kernel 0 h; budget debit
0; upload 0; submission 0; RunPod 0. No DEAD action, plan amendment, or shared-state
write occurred.
