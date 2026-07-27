# GA-H1-ZWRFEAT-GRAPH — parity repair and 44b6 graph kill

Date: 2026-07-27
Verdict: **DEAD / 44b6 GRAPH KILL**

## Parity repair

The human-authorized Option-C repair makes Tesla T4 the canonical and deployed
platform. Both attested T4 repeats are `array_equal`; M5 is retired from pass/fail
and retained only for this one-time diagnostic. The existing 105-video cache was
accepted without regeneration.

The M5 comparison is **BENIGN_FP**. All integer identity/geometry fields are exact;
shapes, dtypes, feature order, and finiteness match; there is no whole-column
offset, permutation, or sign crossing. The geometry drift reaches 15 float32 ULPs,
not one ULP, but remains below `8.940696716308594e-7`.

| field / column | max abs drift | `atol` | `rtol` | classification |
|---|---:|---:|---:|---|
| geometry confidences | `8.9406967e-7` | `1e-6` | `0` | BENIGN_FP |
| geometry probabilities | `8.9406967e-7` | `1e-6` | `0` | BENIGN_FP |
| context 00 | `2.8610230e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| context 15 | `3.8146973e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| context 16 | `2.8610230e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| context 31 | `3.8146973e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| context 39 | `3.8146973e-6` | `2e-6` | `5e-5` | BENIGN_FP |
| context 40 | `2.6226044e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| context 41 | `2.8610230e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| context 44 | `1.4901161e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| context 45 | `1.3709068e-6` | `1e-6` | `5e-5` | BENIGN_FP |
| Trackastra margin | `1.7642975e-5` | `2e-5` | `5e-5` | BENIGN_FP |

The remaining 39 neural columns are exact. Full mismatch counts, quantiles, and
residual-sign evidence are in `h1_parity_repair_contract.json`.

## Fit-only execution repairs

Before scoring, the runner hash-verified all 210 fit-cache files and all 105
fit-label files. The exact-pair join audit found 295 labeled cached proposals on
34/36 44b6 videos and 922 on 66/69 6bba videos; five videos legitimately have no
overlap. Empty video groups remain valid grouped-OOF groups.

The first runnable contract inherited a proposal threshold requiring at least
100 accepted calibration examples per fold. That is impossible on 44b6 and
contradicts H0's ban on proposal-only gates. Before any graph result existed, the
parity-repair overlay was corrected to require support 1 at the original 0.95
floor and to use the maximum finite calibration score as a deterministic,
report-only fallback. Proposal metrics can select an operating point but cannot
prevent graph construction.

## 44b6 result

| metric | APNCC control | H1 candidate | delta |
|---|---:|---:|---:|
| adjusted edge Jaccard | `0.8665345260` | `0.8553864495` | `−0.0111480766` |
| raw edge Jaccard | `0.8625960575` | `0.8515385253` | `−0.0110575321` |
| node recall | `0.9926180003` | `0.9923300994` | `−0.0002879009` |

All 36 fragments are observation-distinct. Report-only held proposal precision
is `0.9565217391`: 242 correct rescues, 11 harmful swaps, 253 accepted from 295
eligible labeled proposals.

The adjusted graph delta is nonpositive and all three objective guards regress.
The locked action is therefore **KILL**. The runner did not call 6bba and did not
access CV57, validation12, or holdout paths. No GPU, Kaggle, RunPod, submission,
or prediction action occurred.

`H3-MASK-SOURCE` is now eligible only as a separate human-authorized provenance
node to establish authentic per-detection masks. It remains unstarted and
pseudo-masks remain forbidden.

## Pinned artifacts

- parity repair contract: `f0beba3c34479cc8ff40a17bd22331c4b0191ca55779e77fa43f04cdf421aa24`
- rotated graph contract: `09b67cdd2dd35fa67b968bd3aa4f53c569da4706ec545c709bc655de750cd4fb`
- coordinator result: `94a3cd006cd40aeb45f1773f6e198767133c3724eed720f573b1e3acd6851c9a`
- 44b6 family report: `530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d`
- final resource guard: `3538f97e714645d5527349cb7f1eaf90f734a2cd0fe7d1db1a581ab23b5cda71`
- label-join audit: `09465eaaa70f1e88b217c6395e1f8705a49b08aa3a756d076d40e2056b9280fc`
- accepted feature report: `d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf`
- accepted run manifest: `548bbac49af6e2170d4ce8f562a26ac7ce9b14433d29d8e10673e0797d4e8d10`

The final guarded graph process ran for `2092.347636938095` seconds, peaked at
`1.1191253662109375` GiB RSS under the 4 GiB cap, and recorded `0.0` MiB swap
growth. Its child exit code `3` is the coordinator's intentional KILL verdict.
