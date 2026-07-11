# Evaluation Record

## Audit baseline

Audit date: 2026-07-11 (Asia/Singapore).

- Repository: `/Users/agent/Desktop/cell_tracking`.
- Git has an unborn `HEAD`; there is no commit history to use as evidence. Existing-file modification times and direct filesystem inspection were used instead.
- No `AGENTS.md` was present.
- No `pyproject.toml`, requirements file, pytest configuration, or original test directory was present.
- Runtime used for verification: local `.venv`, Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0.
- `tracksdata` and `polars` are installed; `ultrack` and `pytest` are not installed.
- Existing reports, derived plots, the 81 GB competition archive, external source, and production outputs were preserved.
- Secret-bearing files were excluded from reads. No credential value was printed, copied, or evaluated.

## Before-change evidence

The original alpha linker sorted all candidate edges by local cost and greedily accepted the first degree-compatible edge. A deterministic two-by-two counterexample produced only one edge:

```text
original selected edges: [(1, 3, 0.1)]
```

The globally optimal feasible assignment is two edges, `(1, 4)` and `(2, 3)`, with total cost `0.31`. This became a regression test.

The original two-timepoint competition smoke command was run to temporary output:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_alpha_routed_zip.py \
  --split train --datasets 44b6_0113de3b \
  --clip-start 0 --clip-length 2 --evaluate --route clean_point \
  --output /private/tmp/celltrack_alpha_before.csv \
  --metrics-output /private/tmp/celltrack_alpha_before.json
```

Observed baseline: 240 nodes, 96 edges, no divisions or merges, bounded edge Jaccard `1.0`, node recall `1.0`, and node-count-adjusted edge score `1.0534` (implementation field `adj_edge_jaccard`). This short sparse clip is a compatibility smoke, not a representative accuracy estimate.

## Automated tests

The repository now uses standard-library `unittest`, because pytest is absent:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

The final count is recorded in the final-verification section below. Coverage includes:

- OME-NGFF named-axis calibration and malformed metadata;
- anisotropic physical-distance gates;
- global assignment versus a greedy dead end;
- deterministic ties under candidate permutation;
- tracklet gap closing and adjacent-predecessor protection;
- merge, mixed-time division, `dt`, and degree invariants;
- opt-in intensity cost behavior;
- strict array/window/offset/volume validation;
- controlled missing/incompatible Ultrack errors;
- deterministic synthetic 4D association benchmark.

## Synthetic fixed-detection benchmark

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/benchmark_synthetic_tracking.py --repetitions 10
```

Observed result:

| Field | Value |
|---|---:|
| Shape | `6 x 3 x 16 x 16` (`T/Z/Y/X`) |
| Physical scale | `(2.0, 1.0, 1.0)` (`Z/Y/X`) |
| Nodes | 14 |
| Truth / predicted edges | 10 / 10 |
| Edge precision / recall / F1 | `1.0 / 1.0 / 1.0` |
| Gap edges recovered | 1 |
| Repeated outputs identical | true |
| Fixture/config digest | `51b98139c2650ee2b0b2e57fe2049d8e510e177aad51e74077c1c06c484c1569` |
| Edge digest | `eac380a9a04d3155b222df8c557d9840d90274c9dd545912ad6364b1fbceb99b` |
| Anisotropic z distractor | correct scale matches truth; isotropic scale does not |
| Median / maximum runtime | Recorded by each run; not used as a cross-machine gate |

This benchmark bypasses segmentation and uses engineered point truth. It demonstrates deterministic association and gap behavior only.

## Competition compatibility checks

The same two-timepoint alpha smoke was re-run after the change. It retained 240 nodes, 96 edges, no divisions/merges, bounded edge Jaccard `1.0`, recall `1.0`, and node-count-adjusted edge score `1.0534`. Nodes were identical; 16 of 96 selected edges changed in each direction because global assignment legitimately differs from greedy selection. The sparse metric on this small clip did not distinguish them. No claim of real-data improvement is made from this result.

The repository metric defines `r = (N_pred - N_estimated) / N_estimated` and `J_adj = max(0, J_edge * (1 - 0.1 * r))`. Because under-prediction makes `r` negative, `J_adj` can exceed `1` and is not itself a Jaccard index (nominal range `0..1.1` for non-negative counts). Its combined score is `J_adj + 0.1 * J_division` when divisions are present (nominal range `0..1.2`). Formula and field provenance: `external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py`.

The production duplicate-data route was rebuilt to temporary paths:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/build_duplicate_geff_submission.py \
  --output /private/tmp/duplicate_geff_after.csv \
  --report-output /private/tmp/duplicate_geff_after.json
```

Observed:

- 2,193 nodes and 2,127 edges across four datasets;
- schema valid, no endpoint errors;
- edge Jaccard `1.0`, division Jaccard `1.0` where defined;
- combined repository score `1.1936358875688189` (`J_adj + 0.1 * J_division`);
- SHA-256 `22e67a1ac1b414b7ee0f188a8fdecc0bbe4cfa5ca93fffe318bbb923d30efefc`, byte-identical to the existing production CSV.

This shows that the solver upgrade did not alter the currently generated duplicate-route CSV. The upstream duplicate gate matched member names, uncompressed sizes, and CRC32; CRC32 is not collision-resistant proof of source byte identity.

## Backend status

`celltrack_baseline`, `celltrack_alpha`, and the Ultrack integration module import successfully in the current environment. Full Ultrack solving was not run because the optional package is absent and the project's documented release is incompatible with Python 3.14. The controlled-error tests cover both missing-package and import-time type-crash paths; they do not substitute for a real Ultrack run on a supported Python version.

## Security and privacy audit boundary

The application is a local Python CLI/scientific stack, not a web server, so no framework-specific web-security guidance applies. The filename-only inventory found `.env` and `gurobi.lic`; neither file was opened. Source-code credential names are `GUROBI_API_KEY`, `GRB_LICENSE_FILE`, `WLSACCESSID`, `WLSSECRET`, `LICENSEID`, `GUROBI_WLSACCESSID`, `GUROBI_WLSSECRET`, and `GUROBI_LICENSEID`; no values were inspected. Network download and Kaggle push scripts were inventoried but never executed. Microscopy data was not uploaded, extracted, or copied.

Release blockers remain: complete closed validation for every `RouteConfig`/`UltrackConfig` field, Ultrack array/resource preflight before allocation, atomic CSV/JSON writes, explicit overwrite policy, decompressed-size limits for untrusted archives, and packaging/dependency audit infrastructure. The safe-path scan found argument-list subprocess calls and no `shell=True`, pickle loading, or archive extraction in the core execution paths, but that inventory is not a full adversarial audit.

## Final verification after independent review

An independent read-only review found unit/axis ambiguity, incomplete invariant and unmatched/gap coverage, a non-discriminating synthetic digest, CRC wording, metric terminology, and large gap-matrix risk. The implementation and documents were revised: units normalize to micrometers, raw metadata must be `T/Z/Y/X`, edge `dt` is integral, invariant/birth-death/global-gap/nested-import tables were added, the benchmark now golden-gates an anisotropy-discriminating manifest, gap LAPs decompose by connected component, and claims were narrowed.

Final gates:

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -q`: **40 tests passed**.
- The assignment suite includes 200 seeded random 1–4 by 1–4 sparse cases checked against exhaustive partial-matching optimum. A supplemental review run checked 500 randomized cases with the same result.
- Sparse-component test: 200 disconnected gap candidates produced 200 independent 2x2 augmented LAPs rather than one 400x400 matrix.
- `py_compile` over all Python files in `src`, `scripts`, `tests`, and `kaggle_kernel`: passed, with bytecode redirected outside the repository.
- AST parsing: **36 Python files passed**.
- Runtime type-hint resolution: **75 project functions/classes passed** across baseline, alpha, synthetic, and Ultrack integration modules.
- Actual archive metadata: dataset `44b6_0113de3b`, shape `(100, 64, 256, 256)`, physical scale `(1.625, 0.40625, 0.40625)` micrometers.
- Final synthetic run: 14 nodes, 10/10 edges, precision/recall/F1 `1.0`, one gap, stable fixture and output digests over 10 repetitions; median/max runtime `0.577/0.994 ms` on this machine.
- Final two-frame alpha smoke: 240 nodes, 96 edges, zero divisions/merges, bounded edge Jaccard `1.0`, recall `1.0`, node-count-adjusted score `1.0534`.
- Final duplicate-route regeneration: 2,193 nodes, 2,127 edges, valid schema, output SHA-256 `22e67a1ac1b414b7ee0f188a8fdecc0bbe4cfa5ca93fffe318bbb923d30efefc`, byte-for-byte equal to the existing production CSV.
- Documentation directory inventory: exactly four files—`EVALUATION.md`, `RESEARCH.md`, `ARCHITECTURE.md`, and `TODO.md`; all are Markdown.
- Documentation inventory found the seven pre-existing runbook/research Markdown files and no root README, LICENSE, CHANGELOG, CONTRIBUTING, SECURITY, CITATION, or packaging manifest.
- Filesystem Git check: `.git/HEAD` still points to unborn `refs/heads/main`; no main ref file and no Git index exist. No commit, stage, push, reset, or branch mutation occurred.

Residual review findings are explicit release blockers rather than hidden successes: one connected bipartite component can still allocate a large dense LAP; configuration validation is not closed across every public field; Ultrack input allocation lacks full resource preflight; writes are not atomic; and there is no reproducible environment/package manifest.

## Scientific limitations

- No representative held-out non-duplicate biological sequence was used to calibrate link, appearance, motion, birth, death, or division costs.
- The intensity term is therefore disabled by default.
- The fallback has point detections, not instance masks; shape and segmentation metrics are unavailable.
- The synthetic benchmark cannot establish biological lineage accuracy, generalization, CTC performance, or state-of-the-art status.
- The two-frame competition smoke has sparse labels and is too easy to rank linkers.
- The existing full-video report remains the best local fallback evidence: score `0.6283` after dataset-specific coordinate correction, below the project's `0.8` gate.
- Division accuracy, long-gap recovery, dense crossings, memory scaling, and streaming/batch equivalence remain unvalidated on biological data.
