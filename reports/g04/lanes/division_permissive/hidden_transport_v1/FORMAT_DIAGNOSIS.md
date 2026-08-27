# G-04 Lane A submission-format diagnosis

The promoted CSV is not malformed. It is byte-identical to the downloaded public kernel output, matches the scored APNCC header/dialect/order, has valid frame-adjacent lineage topology, and passes the organizer's published CSV-to-graph conversion path.

The failed transport's exact local contract defect is that it copied a frozen CSV containing only the four public test identities and deliberately made zero competition-input visits. Kaggle code submissions rerun the selected kernel against hidden test data. A syntactically valid public CSV therefore cannot cover the hidden dataset identities. Kaggle returned submission ref `55201829` as `COMPLETE` with blank scores, `total_bytes=0`, and generic “incorrect format” text. No hidden traceback is exposed, so the causal classification remains evidence-bound inference rather than traceback verification.

This signature has a campaign-local precedent: ref `54890576` also reached `COMPLETE` with blank scores and `total_bytes=0` after a public-only registry failed during the hidden rerun. The relevant evidence is `reports/g02/scorer_submission_path_findings_v1.md` (SHA-256 `444c8363…114c`) and `reports/g02/kaggle_submission_failure_54890576_v1.json` (SHA-256 `85b51571…d3ae`).

The corrected transport is not another CSV copy. `onebitbrain/g04divcap5-v1` embeds the exact scored APNCC monolith (package `779aa8d3…606b`, ref `54942864`) and first computes a fresh submission from the mounted test Zarrs. It then requires an exact match between those freshly produced dataset identities and the mounted raw test inventory, computes the frozen label-blind intensity summaries, and applies `unfiltered_geometry_cap5`. The public raw-Zarr validation reproduced the promoted `13,713,344`-byte / `263,297`-row CSV exactly, SHA-256 `da73c754…4509`.

The old frozen-copy predicate is superseded by `code_transport_v1/TRANSPORT_INVALIDATION.json` and must never be reused. No upload, kernel run, or competition submission was performed while producing this correction.
