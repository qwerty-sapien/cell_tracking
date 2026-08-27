# G04 Lane-A public-copy transport v1 — terminally invalid

Status: `TERMINAL_INVALIDATED_NO_RETRY`.

This historical package copied a CSV computed from four public identities and
explicitly made zero visits to the competition input. Kaggle code submissions
are rerun against the hidden test mount; consequently Version 1 produced no
hidden-population rows and submission ref `55201829` completed with a blank
score and zero bytes. The CSV's local format was valid. The failure was a
hidden-rerun transport mismatch, not a CSV-format defect.

Do not run the former launch sequence or reuse its predicate. Its scripts and
tests are archived under `_ARCHIVE/g04_public_copy_transport_invalid/`; the
kernel package and receipts remain here as immutable failure evidence.

The corrected lineage is `onebitbrain/g04divcap5-v1`: it discovers the mounted
test dynamically, performs fresh APNCC inference, applies the frozen G04
postprocess, and emits a version-bound output. All future code-output
submissions must use the shared one-shot lifecycle, an exact-version CMUX
output receipt, and a predicate carrying the hidden-rerun compatibility
invariants.
