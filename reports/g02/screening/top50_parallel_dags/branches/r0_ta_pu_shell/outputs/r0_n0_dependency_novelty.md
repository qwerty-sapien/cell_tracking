# R0 N0 — dependency and novelty lock

Date: 2026-07-27  
Verdict: **PARK**  
Seeking-alpha classification: **research bet**  
External usage: **0**

## Falsifiable gate

`R0-TA-PU-SHELL` may advance only when one immutable pretrained Trackastra/HOCT
association arm has already emitted exact fit-only graph vectors satisfying all of:

- adjusted graph delta `>=+0.005` on 44b6;
- adjusted graph delta `>=+0.005` on 6bba;
- raw edge-Jaccard delta `>=0` on both families; and
- node-recall delta `>=0` on both families.

The gate is indivisible. Proposal metrics, cache coverage, governance checks,
preregistration, dry validation, and evidence combined across different arms do not
qualify.

## Result

**PARK. Zero qualifying upstream arms exist.**

| Candidate | Authenticated state | 44b6 adjusted/raw/node | 6bba adjusted/raw/node | Qualifies |
|---|---|---|---|---|
| old `H1-ZWRFEAT` | DEAD on structural M5↔T4 drift; graph never ran | `null/null/null` | `null/null/null` | no |
| `U0-T4CANON` | N0/N1 GO as scorer-free governance/interface work; no association graph | `null/null/null` | `null/null/null` | no |
| `C1-TA-HOCT-LITE` | N0/N1 GO for mechanism and preregistration only; its graph gates remain unopened | `null/null/null` | `null/null/null` | no |
| `C2-KBEST-STABILITY` | N0 PARK at its dependency gate; no implementation or graph result | `null/null/null` | `null/null/null` | no |
| APNCC | frozen baseline, not a pretrained incremental association winner | baseline | baseline | no |

The frozen Trackastra cache is evidence that a pretrained representation was
materialized, not evidence that it improves the graph objective. No qualifying
association-model hash can therefore be attached to an upstream passing graph. This
is a missing prerequisite, not an identity ambiguity, so the correct verdict is PARK
rather than ESCALATE.

## Evidence authentication

| Artifact | SHA-256 | What it establishes |
|---|---|---|
| U0 N0 report | `b5bb9747f77fb4aaed74cd8c7d41dc6541b5621f589492233fa61791062e5592` | separate T4-native contract preregistered, not scored |
| U0 N1 report | `be419cb9236d8015d03ec066578cdc9f34ab29a7841b4012866c9dbd0e676b76` | interface verification 10/10, not scored |
| U0 draft contract | `08cd90e5d67f0bf9c7822caa1a057f767b5e7e66e0889fcde04ef317cfc016c1` | T4-only governance identity |
| U0 implementation manifest | `12fe8874965ee1da05b726e99fd901a12e945afa2e4b336f7311acf91a999180` | implementation/test identity |
| C1 N0 machine manifest | `f1227f3d221c01a23179afa9291791b8dee2b6429e9ccc9a3b4ee9181d1e770e` | mechanism/data manifest only |
| C1 N1 preregistration | `a66a6866a94adf774be1e58669076b7474c45e24f019bd655bdb2d19c5d1be7f` | frozen future gates, not their outcomes |
| C2 terminal report | `259edb515125bd5616c3df8fcc07c4aa93a35afed895fbb61151612378a0b0a6` | dependency PARK, no graph access |
| D1 association audit | `9ead4f6dff91c5eadf651ee711a3d73bea31b41f0528d7393a116f0a79418113` | independent zero-qualifying-arm audit |
| banked CG1 report | `ab20461dfdd2c05f2e9525e3d1e50898bf8b49ad5ba9403c240e5f7519c69867` | generic PU/CRC mechanisms failed the cheap screen |
| banked CG1 JSON | `2826f3da991991c8b0276969a5de6a576bd3696d23185d75a91105dcccb84a88` | complete negative-screen evidence |

The exact current-state file is mutable by the concurrent campaign and is therefore
referenced by path and decision content rather than treated as the immutable
association artifact required by this gate.

## Novelty and DEAD-mechanism separation

The proposed R0 conjunction remains materially named and distinct, but the
distinction is only a **research-bet design** until its upstream representation wins:

| Prior negative | What failed | R0-only delta that would be required |
|---|---|---|
| `CG1-PU` | sampled deployment mixture; class-prior mismatch; zero retained correct edges in the tested instantiation | qualifying pretrained association representation, complete unlabeled deployment census, cross-fit nnPU risk, and no unlabeled-as-negative conversion |
| `CG1-CRC` | separate CRC rules retained `<1%` correct links and could not certify the 44b6 tail | group-fixed calibration on the complete population after pretrained association lift; no FDR claim without super-uniform null evidence |
| `R12SHELL` | geometry/NCC could not abstain against 5.34M background candidates | learned pretrained link-vs-no-link representation rather than geometry/NCC |
| `MGSHELL` | constant velocity retained the background burden and achieved about 0.015 validation precision | no motion-only admission; explicit no-link probability plus risk-controlled admission |
| `TEMPBG` | compact from-scratch background gate over-abstained and lost recall | frozen pretrained association signal plus full-population nnPU; no from-scratch representation |

The component basis remains literature-grounded by
[Trackastra (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf),
[nnPU risk (NeurIPS 2017)](https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html),
[Conformal Risk Control (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html),
and [Benjamini–Yekutieli dependent-test FDR control](https://doi.org/10.1214/aos/1013699998).
Those sources ground components, not the campaign-specific conjunction or its validity
under these videos. R0 must still establish its PU identifiability, groupwise
exchangeability, and super-uniform null evidence in N1 if it ever unlocks.

## Stop and resume condition

N1–N5 were not opened. No banked negative was rerun; no full-pool census, model,
training, graph score, CV57 access, GPU, Kaggle, RunPod, submission, push, or outward
mutation occurred.

Resume from N0 only after **one named association arm** publishes immutable 44b6 and
6bba graph artifacts meeting every numeric gate above. On resume, authenticate that
single arm's model/representation hash, scorer hash, observation identities, and full
score vectors. Do not combine partial family evidence or relax the threshold.

## Proposed LEDGER row

`R0-TA-PU-SHELL-N0 | DEPENDENCY/LITERATURE | 0 qualifying pretrained association
arms; all family graph vectors null | U0/C1/C2/D1 + banked CG1 identities above |
0 external usage | PARK | materially new four-part design retained as research bet;
resume only after one arm passes >=+0.005 adjusted on both families with raw/node
nonnegative | r0_n0_dependency_novelty.md`

## Cascade

PARK → halt. Do not dispatch `N1_population_preregistration.md`.
