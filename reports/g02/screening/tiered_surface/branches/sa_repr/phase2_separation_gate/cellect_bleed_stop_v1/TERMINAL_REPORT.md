# CELLECT bleed-stop v1 — local mounted-layout PASS, no launch

The v6 transport dependency on `consolidated_bundle_manifest.json` is removed.
The generated single-file kernel embeds the exact 77 request records and every
required SHA-256/byte pin in code. At runtime it resolves 77 requests,
population, safe loader, three wheels, and the checkpoint solely by content.
The request and bundle manifests are not required or consulted.

The local Kaggle-mounted fixture contains 83 opaque files and no external
manifest. It reproduces tar materialization by replacing the checkpoint tar
with its raw byte-exact member under an opaque name. Step 0 passes that layout,
also passes the owner-downloaded layout, and fails closed for one missing or
duplicated required object. Canonical filenames for the safe loader, wheels,
and checkpoint are created only under the working directory after content
authentication, so downstream import/pip behavior is independent of mounted
names.

The post-step0 science suffix is byte-identical after normalizing only the
dataset transport-version literal (`7583c9b6…4046`). The independent discovery
tests pass 7/7 and the current shared CELLECT suite passes 24/24. Population,
folds, probe, thresholds, A1/A0 construction, feature math, and evaluator are
unchanged.

This is a local transport PASS only. No CELLECT dataset upload, authorization,
kernel, GPU debit, outcome, verdict, PARK-C move, or graph unlock occurred.
CELLECT remains launch-closed until the DINOv2 Lane-A run reports.
