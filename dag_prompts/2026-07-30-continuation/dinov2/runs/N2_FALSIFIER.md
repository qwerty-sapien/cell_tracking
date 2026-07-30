# N2 — DPRC-O0 fixture falsifier result

Date: 2026-07-30 (Asia/Singapore)
Verdict: **GO**

After the corrected contract was preregistered in
[N2_PREREGISTRATION.md](N2_PREREGISTRATION.md), running

    python3 N2_dprc_fixture_test.py

passed all four focused tests:

1. same-mean spatial permutation: global pooling tied at 1.0, identity cycle
   score was 1.0, and reversed-grid cycle score was 0.25;
2. uniform token control: exact zero incumbent-versus-alternative margin;
3. empty foreground: fail-closed ValueError;
4. repeat: score-identical result.

The test is wholly synthetic and proves only the predeclared local-correspondence
contract. It does not load a model or campaign data, cannot assess 44b6, and
creates no source, feature, data, Kaggle, or R5 artifact. The independently
specified, bounded, no-retry protocol is in
[N2_KAGGLE_PROTOCOL_DRAFT.md](N2_KAGGLE_PROTOCOL_DRAFT.md).

The contract is deterministic and the protocol is materially distinct from R5;
therefore N2 meets its GO gate. Its limitation is carried forward: a real
microscopy/graph effect remains unmeasured.

## Proposed ledger row — not written

DPRC-O0 | N2 GO | deterministic synthetic contract passes; fresh-supplier
protocol requires owner gate | N3 HUMAN-DECISION

**Cascade:** GO to N3 human gate.
