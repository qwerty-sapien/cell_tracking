# G-03 Lane 0 h005 terminal report

## Outcome

Lane 0 is **infrastructure-terminal with no h005 measurement**. The sole
dependency-repaired Kaggle version 2 exited after `3.5865 s` inside top-level
STEP0 because the runtime dependency manifest at the four-source GPU-worker
mount did not match the frozen SHA-256. The prior CPU release-path probe had
authenticated the same dataset and manifest successfully when mounted as its
only private dataset source. V2's frozen check did not emit observed presence,
size, or SHA before raising, so the terminal evidence cannot distinguish a
missing mount from a byte-divergent mount. It does establish that the mismatch
preceded dependency installation and all scientific work.

There was no scientific-core import, CUDA model allocation, raw/GEFF access,
optimizer step, partial checkpoint, immutable h005 checkpoint, held-fit36
evaluation, graph score, slope, CV57 access, hybrid, submission, or RunPod
action. Kaggle reports no output files; the downloaded terminal artifact is the
log only. Authoritative quota remains `1.02/28.98 T4-h` used/remaining, so V2's
rounded debit is `0.00 h` and the two failed launch versions together cost a
rounded `0.01 h` of infrastructure time.

## Allocation decision

The h030 gate is **not evaluable**: it requires a fixed h005 score on
held-fit36, and neither checkpoint nor score exists. Therefore h030 is not
funded, without treating the infrastructure failure as a flat or negative
scientific trajectory. The entire planned 25-hour continuation remains
preserved. No subtraction uses the CV57 campaign best; the only eligible
reference would have been APNCC `0.8665345260361912` on held-fit36.

The frozen V2 repair contract explicitly allowed no V3. Lane 0 therefore closes
here. Any future retry would require an owner-approved supersession plus a new
probe using the exact four-source composition and a STEP0 receipt that records
manifest presence, size, and observed hash before comparison.

Evidence: `V2_TERMINAL_DIAGNOSIS.json`; terminal log SHA-256
`ed03693f6ed51b4405cd4e3f6d46d939d59111943614533868f9ee374269fe73`.
