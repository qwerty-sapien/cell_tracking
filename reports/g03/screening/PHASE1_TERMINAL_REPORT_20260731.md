# G-03 Phase 1 terminal report — hard fail before checkpointing

**Verdict: `G03_STOPPED_PER_PHASE1_HARD_KILL`.**

The only authorized Phase 1 A attempt, private Kaggle kernel
`onebitbrain/g03-phase1-resume-a-v1` v1, was pushed once from the committed
worktree package `05c1c6de566abad84257bbdb33ff258c6ecf6f9a`. Its source and
metadata hashes were verified locally; the authenticated quota bracket was
`5.43 / 24.57` T4-hours before and after the push.

Kaggle terminalized the kernel as `ERROR`. The downloaded log has SHA-256
`e1c5370d1bba55d7f6aa65257616de19459cf0d60045222d6743eab56be7caed` and
shows at 0.70 seconds:

```
ModuleNotFoundError: No module named 'g03_resume_torch'
```

Kaggle executed only the declared `code_file` as `/kaggle/src/script.py`; its
adjacent helper modules were not vendored into that script. The failure occurred
before Torch import, CUDA initialization, timed training, checkpoint emission,
data access, graph-emission receipt, handoff dataset creation, or Kernel B. The
post-terminal authoritative GPU quota remains `5.43 h` used / `24.57 h`
remaining, so debit is exactly `0.0` T4-hours.

The Phase 1 preregistration defines any inability to demonstrate the new-kernel
handoff, including an integrity/runtime failure, as a no-retry hard fail that
stops the campaign. Therefore no package correction, A retry, handoff dataset,
Kernel B, Phase 2 training, CV57, submission, or final-slot action is authorized
under G-03. A future campaign needs a new owner charter and preregistration; the
obvious repair is compiling helpers into one Kaggle `code_file`, but it is not
executed here.

Evidence: `PHASE1_PRELAUNCH_20260731.json` and `phase1_ops/` receipts.
