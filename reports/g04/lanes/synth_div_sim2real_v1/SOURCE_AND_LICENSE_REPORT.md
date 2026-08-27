# SYNTH-DIV-SIM2REAL-v1 source and licence record

Frozen on 2026-08-03 before model fitting.

## Synthetic supervision

The dataset and generator were published by **José Freitas**
(`@josefreitasalvesneto`) in Kaggle discussion
[732103](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103).
The author declares the generated data **CC0-1.0**.  The two public Kaggle
notebooks identify their code as **Apache-2.0**.  This distinction matters:
the 18.5 GiB payload is notebook output, not a separately licensed Kaggle
Dataset record, so the CC0 conclusion is based on the author's explicit
discussion statement and is preserved here with exact source provenance.

Registered public artifacts:

- Generator/output notebook, version 3, Kaggle version ID `339394219`:
  [biohub-synthetic-dataset](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset).
- Explorer notebook, version 6, Kaggle version ID `339530198`:
  [Synthetic 3D Microscopy Data for Cell Tracking](https://www.kaggle.com/code/josefreitasalvesneto/synthetic-3d-microscopy-data-for-cell-tracking).

Authenticated local source bytes:

| artifact | bytes | SHA-256 |
|---|---:|---|
| generator v3 notebook | 177,661 | `04ed49b48c896dea94b2d47d5f59c00479d53a1cba109925642be0dccb9a97e5` |
| generator v3 metadata | 634 | `f6d043f1656be40339c527b84f4c046583d535b0a9325ad6df0b310097ad925e` |
| explorer v6 notebook | 21,441 | `f038937e995f7d007a4d1d6723e2e7595744be41b7ad3a569c8205a13f0748bd` |
| explorer v6 metadata | 755 | `5894b64c519d18825a80b8b0515c72d9bb2e6f5aa349731915533d7a55ed1a42` |

The generator metadata and manifest report 1,539 static volumes, 2,174
six-frame sequences, 4,056,226 labelled nodes, and 165,267 labelled division
parents.  Their exact event prevalence is `0.0407440315209261`, compared with
the preregistered real target prior `0.0026`.  The model therefore uses the
frozen event-loss weights `w_pos=0.06381302740413997` and
`w_neg=1.0397641847164156`; it does not apply a second logit shift.

The acquired v3 output authenticated as 3,716 files and 19,868,008,592 bytes.
Its canonical streamed tree SHA-256 is
`9a87684aecd0426a355c99dfe1ee7b7dfbd2da2eb713b9bc3d250cdc029865ec`;
the complete per-file hash and structural report is
`PAYLOAD_AUTHENTICATION.json` (SHA-256
`0fd643c8045f5bcf9b223658d73da7a3836cff4b91be148e8cc45c395aa65b0f`).
Validation reproduced every published aggregate and did not materialize any
image array.

The author explicitly describes the distribution match as partial and
texture/contrast as the weakest axes.  Accordingly, raw pixels, texture, and
contrast are excluded from this transfer lane.  Only geometry, lineage, and
motion labels are used, including the published 1.86 µm/frame median step,
`+0.30` lag-1 directional persistence, and 7.24 µm sister separation.

## Cell Tracking Challenge datasets

The locally present CTC archives are registered, but they are **not used** by
this lane.  The authoritative [CTC conditions of
use](https://celltrackingchallenge.net/datasets/) allow use related to the
Cell Tracking Challenge without separate consent.  They require explicit
organizer permission for public, non-CTC scientific use and forbid cloning or
mirroring.  Training a Biohub Kaggle competition model is not CTC-related, so
citation alone does not clear that use.  Written permission must be attached
to the registry before any CTC archive is loaded for training or uploaded.

If permission is obtained, the required general citation is:

> Maška, M. et al. The Cell Tracking Challenge: 10 years of objective
> benchmarking. *Nature Methods* **20**, 1010–1020 (2023).
> [doi:10.1038/s41592-023-01879-y](https://doi.org/10.1038/s41592-023-01879-y)

Dataset-specific provider attribution remains additional; CTC attributes the
SIM+ material relevant here to V. Ulman and D. Svoboda.

The machine-readable authority record is `EXTERNAL_DATA_REGISTRY.json`.
