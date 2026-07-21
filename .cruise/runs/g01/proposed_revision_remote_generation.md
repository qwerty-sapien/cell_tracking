# Proposed Revision: Remote Kaggle Kernel Generation Before Scored Submission

- Status: Proposed, inactive
- Trigger: T-03 local full test generation blocked after two approved attempts; T-03R classified the likely cause as local resource/runtime and recommended Kaggle-kernel remote generation.
- Preserves: G-01 endpoint, low-compute envelope, package parity, score provenance, submission cap, and no paid compute.

## Evidence

- T-02 package parity passed: `reports/kaggle_g01/package_parity_manifest.json`
- T-03 local full generation failed to produce `reports/kaggle_g01/submission.csv`.
- T-03R passed ZIP/Zarr/package checks and one-dataset one-timepoint alpha smoke, then recommended plan revision.
- No Kaggle submission has been made and no Kaggle quota has been consumed.

## Minimal Contract Change

Add a replacement path for the blocked T-03 dependency:

### D-05 - Kaggle Kernel Remote Generation

- Status: Proposed
- Allowed: push/update the private Kaggle kernel `onebitbrain/biohub-celltrack-alpha`; run it on Kaggle free CPU with GPU and internet disabled; read kernel run logs/status; download generated `submission.csv` and logs into `reports/kaggle_g01/`.
- Prohibited: paid compute, GPU, internet-enabled runtime, public kernel publication, unrelated Kaggle account changes, credential changes, submissions to other competitions, printing secret values, or triggering a scored competition submission during this generation step.
- Budget: one remote generation run plus one packaging/runtime repair rerun if the first run fails before producing a CSV.
- Evidence: kernel version/status/log, output artifact digest, downloaded CSV digest, and cleanup/retained-state note.
- Reopening trigger: Kaggle asks for a payment/credential/account action, requires GPU/paid runtime, or the run cannot complete within free Kaggle limits.

### T-03B - Remote Generate Full Test Submission

- Depends on: T-02 and T-03/T-03R evidence.
- Inputs: alpha package identity `7ed7c9949918cebd777cb4e3e300f89ed320006aa79504909f3f795982540620`; `kaggle_kernel/**`.
- Output: downloaded `reports/kaggle_g01/submission.csv`, kernel log/status evidence, and artifact digest.
- Focused evidence: Kaggle kernel run completed and wrote a CSV; no scored competition submission was triggered.

### T-03C - Locally Validate Downloaded Remote CSV

- Depends on: T-03B.
- Output: local schema/graph/dataset coverage validation report.
- Focused evidence: AC-03 and AC-04 become ready for T-04, or block with exact CSV/kernel defect.

### T-04 Adjustment

- T-04 consumes the T-03C-validated downloaded CSV and performs the first scored Kaggle competition submission.
- D-04 scored submission cap remains unchanged: maximum three scored submissions.

## Approval Choice Needed

Approve this revision before any Kaggle kernel push/run. Without it, the active Goal remains at the T-03 block because no locally validated full submission CSV exists.
