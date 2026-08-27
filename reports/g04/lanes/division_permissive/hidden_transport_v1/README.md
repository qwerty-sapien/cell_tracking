# G-04 Lane A hidden-rerun transport v1

Status: **local package and raw-Zarr parity PASS; no Kaggle mutation performed**.

Package: `kaggle_kernel/g04divcap5-v1`

- Kernel ID: `onebitbrain/g04divcap5-v1`
- Main code: `411,398` bytes, SHA-256 `63f60d5c…1bcf`
- Metadata: SHA-256 `b28d10d4…0fd5`
- Package identity: `11256286…eb2`
- Exact base: scored APNCC package `779aa8d3…606b`, public ref `54942864`
- Postprocessor: `4b10b179…d356`
- Public target: `13,713,344` bytes / `263,297` rows / SHA-256 `da73c754…4509`

The kernel runs exact APNCC inference against the currently mounted test data, then applies the frozen label-blind `unfiltered_geometry_cap5` transform to that fresh parent. Public dataset names are not part of the postprocessor contract. The only attached dataset is the same public support pack used by scored APNCC; there is no private public-CSV carrier.

Before a kernel push, regenerate a fresh CMUX-compatible prelaunch receipt and require the package hashes above. After one exact kernel version reaches `COMPLETE`, download its outputs and require `g04_apncc_parent.csv`, `submission.csv`, and `G04_DIVISION_POSTPROCESS_RECEIPT.json` to satisfy every terminal gate in `CODE_SUBMISSION_PREDICATE.json`. Only then may root create an eligible version-bound predicate and submit once with `-k/-v/-f submission.csv` under the shared serial lock.

The prior `g04-division-permissive-submit-v1` copy transport is terminally invalidated and must not be retried.
