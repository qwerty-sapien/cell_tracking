# Reproducibility

The accepted execution is defined by:

- preregistration `00e6036ef076482a39266d797c5fea3e2e4845ba4262c8532cfe70bb8ccb4628`;
- execution repair `1996f4a8096d5cb858ec0bf42ec06786f6910c5921a4fd11d57b24a4f64f827b`;
- N1 locality proof `836812c72a51e915e74b9e57b987c21e3f915290feff76b01121481d664dc55c`;
- implementation `1441f2e49ada926b4b7b28a63b1a43849823c835c1e3f36a05212193afdef695`;
- tests `2758e0bdfc4ccdeb6e7c24efb86f1b16953f518a8f257287c05f128c5eb21575`;
- post-repair manifest `0f02903fe19ea86f79ee22059ae67def473ded92aaac61d2efa830ee52d2282a`.

The exact accepted family command is banked in `guard_44b6_postrepair_v1.json`. It must be preceded by a new 45-second zero-swap preflight and run through `scripts/run_u0_t4canon_guard.py`; direct unguarded execution fails closed.

Do not rerun this exact instantiation for screening. Reproduction is for audit only. The terminal family decision is `KILL_44B6`; 6bba and CV57 remain closed.
