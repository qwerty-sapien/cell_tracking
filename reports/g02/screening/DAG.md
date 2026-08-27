# Execution DAG — revised 2026-07-24

This DAG is the controlling screening/calibration sub-DAG for active G-02. The root
coordinator is the sole writer of `LEDGER.md`, shared gate state, and integrated
calibration artifacts. Workers own disjoint outputs and return evidence.

Scorer semantic correctness and empirical local-to-Kaggle transfer are separate gates:

- `P0A` gates every local score claim.
- `P0B/CAL` has a passed coarse high-range subgate and an unresolved fine
  transfer subgate. Fine public-score forecasting and final selection remain
  gated.
- An unresolved transfer model does **not** halt leakage-safe local decomposition,
  diagnostics, candidate development, or the association/detection GPU gates.

```mermaid
graph TD
    CTRL["CTRL: authority, ledger, artifact identities"]
    P0A["P0A: scorer semantic parity"]
    P0B["P0B: observation diff + transfer calibration"]
    E["E: consolidate existing B1/B2/E1 evidence"]
    V["V: split/audit integrity"]
    L["L: local mechanism screens"]
    Q["Q: preregister sequential public probes"]
    H["H: human approval for next submission"]
    CAL["CAL: interval-aware calibration verdict"]
    GA{"GA: association GPU gate"}
    GD{"GD: detection GPU gate"}
    M["Learned association work"]
    D["Detection-front-end work"]
    S["S1: final portfolio and hedge"]

    CTRL --> P0A
    P0A --> P0B
    P0A --> E
    P0A --> V
    E --> L
    V --> L
    P0B --> Q
    L --> Q
    Q --> H
    H --> CAL
    L --> GA
    L --> GD
    GA --> M
    GD --> D
    CAL --> S
    M --> S
    D --> S
```

## Node rules

- **CTRL:** reconcile the active G-02 authority, initialize/own `LEDGER.md`, freeze
  artifact identities, and serialize shared writes.
- **P0A:** use the attested scorer and fixtures. It has already passed locally; rerun only
  when scorer/source identity changes.
- **P0B:** build canonical observation comparison and audit all existing pairs. It now
  includes the three historical graph-cost artifacts and the measured APNCC-direct
  high-range artifact.
- **E:** consolidate and validate existing division, decomposition, fragmentation/oracle,
  and error-taxonomy evidence before rerunning anything.
- **V:** preserve grouped folds and untouched audit membership.
- **L:** run CPU/cached local mechanism screens independently of transfer calibration.
- **Q/H:** public probes are observation-distinct, preregistered, submitted one at a time,
  and count against the five-per-competition-day cap. Two final selections remain reserved.
- **CAL:** treat displayed scores as rounding intervals. APNCC-direct now
  establishes coarse high-range alignment (`0.881522` local → `0.881` public),
  but no decision-grade global affine exists. Do not spend a calibration-only
  follow-up; attach the next measurement to a materially stronger candidate.
- **GA/GD:** GPU escalation depends on measured headroom, error mechanism, proxy
  improvement, runtime, and budget—not on which calibration model wins.
- **LIT/M5:** every new learned candidate first passes the primary-literature/mechanism
  manifest, then the cheapest guarded fit-only M5 falsification. One local lane may run
  alongside eligible external work; it stops at 18 GiB, any swap growth, or its
  preregistered timeout (at most seven hours).
- **S1:** final selection waits for sufficient transfer evidence and hedges
  generalization versus family/mechanism-tagged upside.

## GPU branches

Association and detection are independent:

- `GA` requires association headroom plus a preregistered CPU/proxy improvement large
  enough to justify the planned compute.
- `GD` requires detection-bound evidence plus a corresponding front-end proxy.
- Initial training is capped at 30 cumulative Kaggle T4 GPU-hours per model; archive
  candidates that cannot become viable inside that ceiling.
- The local literature-first fast-fail lane precedes external initial training and does
  not consume or weaken the Kaggle ceiling. Holdout access is forbidden during it.
- RunPod is unavailable before an accepted score reaches `0.89`. Thereafter it is limited
  to a higher-quality-GPU-dependent optimization with literature/theory-backed expected
  score improvement greater than `5%`, a `$10` total cap, exact-job accounting, and
  explicit human approval. The ordinary hard improvement gate is `3.5%`.

## Stop conditions

- Scorer semantic parity failure halts local scoring until repaired.
- Split/audit leakage halts affected local claims.
- Transfer calibration failure keeps public forecasting and S1 unresolved but leaves
  local development active.
- No Kaggle submission or RunPod launch occurs without explicit human approval.
- Do not exceed five Kaggle submissions per competition day or commit either final
  selection prematurely.
