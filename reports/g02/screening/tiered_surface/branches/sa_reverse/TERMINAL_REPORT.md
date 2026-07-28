# SA-REVERSE-SUPPLIER terminal report

Disposition: **PARK_NOT_READY — BOUNDED CPU COMPUTE REQUEST**
Tier proposal: **none; retain `AE-P6` and `R8` at `PARK-B`**

This lane froze an exact genuine reverse contract but correctly stopped before
inference. The existing fit-only `44b6` supplier is authentic and forward-only:
all **36/36** feature files (**36,225,668 bytes**) cover **3,539**
transitions, **84,883** proposals, and **121,534** conflict rows. The matching
sparse detection input is also local and authentic: **36 `.bhc` + 36 `.json`**
files, **678,781,416 bytes**. The pinned Trackastra source at commit
`fb265888c8fce2ac16a362ebf182ccd5ea48e936` and BSD-3-Clause license match every
banked file hash.

The exact checkpoint does not exist at any known pinned local path, and no
immutable reverse cache exists. The required member is SHA-256
`24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f`
(27,456,880 parameters / 280 tensors / 109,827,520 tensor bytes). A reverse
score therefore cannot be obtained without a new model pass. This lane did not
negate, transpose, duplicate, or recalibrate the forward margin.

`ENDPOINT_RETURN_CONTRACT.json` now specifies the exact future supplier:
preserve node identities and packing, map `t_rev=min(t)+max(t)-t`, execute a
separate reversed-time Trackastra pass, and return canonical `u_t→v_t+1` as
`A_reverse[v,u]`. It freezes the quiet-softmax null residual, missing-endpoint
failure, division exception, row schema, disabled-cycle beta, permutation
control, and array-exact repeat witness. This contract is an enabler, not a
supplier or scientific outcome.

The bounded request is CPU/local only: **3,638 calls** including one repeated
99-call witness, **2,728.5 seconds / 0.758 CPU-h projected**, **1.5 h hard
wall**, **4 GiB RAM cap**, **512 MiB scratch cap**, and **0 GPU-h**. It first
requires the orchestrator or human custodian to provide the already
authenticated checkpoint locally. The request is not authorized and was not
launched.

Evidence:

- preregistration:
  `0c559bac5725cda4f81d28e6804eb125eaa9420dc2c8bb7614fe4948ca3a5b06`
- capability audit:
  `8d4affd049515b26446c98ad8d2eb3a155396f8dacca127d0118734197c9a777`
- endpoint-return contract:
  `4eaf9d8f13eb0121d804be4252e6b53667ae65f7955c3eadf858d257804fd294`
- bounded compute request:
  `1c7cf49122c88826fc44128184ff27bfbc28a9c32e5004adf33e92d4e342927f`
- 45-second preflight:
  `828cb55ac207d93dd04868a0a62b2d93019c4902e553bee6c718aaad89944c92`
- focused tests:
  `1d68fbefb3f465a3bb5baa9de6a6d930369912eee637ad6fc786b73538ed5ea1`
- guarded audit:
  `e891c1f75e200b025221042be0a322541d9d6cd4b6a296733ed1bdb725236da4`

The idle preflight ran **45.087 seconds**; four focused tests and the full
capability audit passed with **0 MiB swap growth**. Reverse inference,
separation, graph, `6bba`, CV57, GPU, Kaggle, budget debit, CELLECT, RunPod,
submission, DEAD, plan amendment, and shared single-writer mutations were all
false.
