# Kaggle standing authorization

Owner decision: 2026-07-31.

The owner authorizes authenticated Kaggle kernel launches, competition
submissions, and monitoring for G-02 without a new permission request whenever
the root coordinator determines that the action is productive.

Productivity means one of:

1. a preregistered, scientifically approved node is launch-ready and the run
   can change a binding campaign decision; or
2. an observation-distinct, locally qualified submission can change the public
   score or calibrate a still-open transfer question.

The following remain mandatory:

- the human-locked plan and any plan-amendment barrier;
- the shared exclusive launch lock and remote zero-in-flight check;
- authoritative Kaggle quota reconciliation before and after;
- exact prelaunch input, code, version-delta, and transport checks;
- the node's frozen scientific kill line, plus the current per-run projection
  gate: an authenticated projection below 8 h is auto-approved with no
  wall-clock kill; an 8 h-or-longer projection needs explicit human approval
  following a literature-cited pitch that states effect size, necessity,
  failure modes, and falsification evidence;
- terminal monitoring and banking in `LIVE_STATE.md` and `LEDGER.md`.

Quota expiry alone is not productivity. A byte-identical repeat of an already
scored submission is not productive. RunPod remains separately human-gated.

The 30 h/week shared GPU ledger remains the hard budget protection and is
reconciled before every launch. This amendment supersedes only the former
per-launch five-hour gate and runtime kill: six-way sharding of a 6.8 h job
added a reassembly layer that produced invalid outdegree-three parents, buying
no additional human oversight.

## Authorization-time remote facts

The owner-authenticated Kaggle CLI reported:

- GPU quota: `5.06 h` used, `24.94 h` remaining, `30.00 h` total;
- quota refresh: `2026-08-01T00:00:00Z` (`2026-08-01 08:00` Singapore);
- competition deadline: `2026-09-29T23:59:00`;
- recent G-02 kernels: all terminal, with no observed in-flight campaign run;
- latest scored submission: reference `54942864`, public `0.881`, submitted
  `2026-07-24T04:26:39.277000`.

The only currently submission-ready file,
`apncc_transport_smoke_v2/remote_output/submission.csv`, has SHA-256
`6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3`,
which is the exact already-scored public-`0.881` APNCC artifact. It was not
resubmitted because it is observation-identical and therefore not productive.
