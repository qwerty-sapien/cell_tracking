# DPRC-O0 N3 — source and weight receipt

Date: 2026-07-30 (Asia/Singapore)
Status: **OWNER AUTHORIZATION RECORDED; inbound provenance acquisition only; no Kaggle asset, kernel, or compute run**

## Owner authorization

The owner wrote **"DINOv2: go ahead"** in the campaign coordination task. It
authorizes the single DPRC-O0 route already bounded in
`N2_KAGGLE_PROTOCOL_DRAFT.md`; it does not authorize an R5 repair/retry,
parallel launch, or any other DINO route.

## Fresh official provenance

| Object | Exact identity |
|---|---|
| Upstream source | `https://github.com/facebookresearch/dinov2.git` at `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| Source license | Apache-2.0; `LICENSE` SHA-256 `600cc67cc4cb2f5ea3171fce7c873ad174dc4bec8782bbe9db0afd83513b935b7` |
| Model constructor | `dinov2_vits14_reg`, exactly four register tokens, from `dinov2/hub/backbones.py` |
| Official checkpoint URL | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth` |
| Checkpoint bytes | `88,291,785` (official HTTP `Content-Length`) |
| Checkpoint SHA-256 | `f433177089a681826af819f1c4ee3bc48fd463fb38d32fc837e3dc74e5641fb` |

The checkpoint was fetched only into an ignored temporary directory for hash
authentication. It is distinct from, and may never be combined with, the R5
checkpoint `b938bf1b…0cd9`, R5 source archive, requests, rows, package,
quarantined fragments, or launch lineage.

## Input facts observed without a model run

- Kaggle competition source: `biohub-cell-tracking-during-development`.
- The local source archive is named
  `biohub-cell-tracking-during-development.zip`; it is not uploaded or copied
  by this route.
- Predeclared calibration candidates are
  `test/44b6_0113de3b.zarr` and `test/44b6_0b24845f.zarr`. Their outputs, if
  a calibration runs, are timing-only and must be discarded without association
  analysis.
- The first candidate's array is Zarr v3 `0`, shape `[100, 64, 256, 256]`,
  `uint16`, chunks `[1, 64, 256, 256]`. Its current physical scales are
  `[1.0, 1.625, 0.40625, 0.40625]` (T/Z/Y/X).

## Remaining launch blockers

No `DPRC-O0` Kaggle asset, kernel, or run exists. Before any such creation,
the new route still needs a clean, independently derived action/materialization
manifest; an exact foreground/channel/crop rule; and a local preflight that
can enforce the five-second/R5-exclusion contract. The two-video calibration
must then establish a conservative complete-44b6 projection of at most 4.0
GPU-h. Failure of any one item terminalizes this exact route with no retry.
