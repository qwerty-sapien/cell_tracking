# CRW-HPO1000 private carrier — explicit Kaggle egress approval request

The host policy reviewer stopped `dataset_create` before any external mutation because
the standing Kaggle kernel authorization does not independently authorize uploading this
specific held-fit-derived payload.

## Exact proposed upload

- Destination: private Kaggle dataset
  `onebitbrain/g02-crw-hpo1000-carrier-v1`, exactly one Version 1.
- Local directory: `input_dataset/`, total `5.6 MiB`.
- Carrier NPZ: `c46afff71557c657d97814baa3ba452101502103f56ade550909bbd158b0b02b`.
- Carrier manifest: `3fce7917c2e4eb713b94cb5b936599806901813039c7e58214bc64b065c416a9`.
- Parameter bank: `21ff9a481d0dc01f9bd6063a7595ea3e56a0ba112b11ffaeb0d5a7dbf49b6997`.
- Preregistration: `5ed8a44693ab9921a5bab4240a5aa570a8e9dc31e92ccaf18884df5ab7c2dd55`.

The payload contains the frozen 92,905 × 18 float64 CRW feature matrix, binary
positive-net target, five integer attribution counters, held-fit video identity codes,
five frozen fold codes, canonical candidate IDs, conflict-component/slot masks, and the
1,000-tuple parameter bank. It therefore contains held-fit-derived supervision and must
be treated as private experimental data.

It does **not** contain raw images, raw ground-truth graphs, the official scorer,
scorer outputs, CV57 or 6bba data, competition-test data, a submission artifact, or
credentials. The Kaggle kernel is private, offline, CPU-only, and not
submission-authorized.

## What approval would authorize

Approval authorizes only:

1. upload of the exact directory above as private dataset Version 1;
2. mandatory READY polling and byte-exact round-trip download verification;
3. one exact Version-1 push of private kernel `onebitbrain/g02-crw-hpo1000-v1` after
   round-trip PASS, under the existing serial/quota/prelaunch gates;
4. terminal monitoring and exact-version output retrieval.

It would not authorize making the dataset public, another dataset or kernel version,
competition submission, CV57/6bba access, post-outcome retuning, or deletion of the
remote dataset.
