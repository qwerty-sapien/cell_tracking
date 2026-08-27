# APNCC transport smoke v2 — terminal adjudication

The single authorized remote version completed its banked runner and produced a
downloadable `submission.csv`. Step 0 passed in 2.715378342 seconds with all 62
declared wheel byte hashes verified and zero competition-path visits.

The 263,282-row CSV passes the independent raw-test-derived validator: all four
expected datasets, 134,409 nodes, 128,873 edges, and no missing endpoints,
merges, duplicate graph objects, or out-degree violations.

The wrapper's retained `ERROR` is a benign wrong-reference false positive. It
compared the APNCC output against `kaggle_kernel/bhsubmitout_v3`, a
pre-appearance-swap V45 baseline. Against the authenticated public-0.881 T4
archive `results-bhap.zip`
(`031aa5f64fd6eaf918213fc072a098657e2a6bc3d3e5a2d2ae0642d4037b7219`),
the remote submission and all four fragments are byte-identical; all non-timing
`run.json` fields and proposal/cap counts also match.

Terminal classification:
**PASS end-to-end submission-CSV transport / PASS authenticated public-archive
identity**.

No scientific re-evaluation, scored submission, graph action, CPU/T4 fallback,
or retry follows. This run debited 0.16 T4-hours and moved Kaggle's
authoritative balance from 25.47 to 25.31 hours remaining. The later campaign
balance is 24.98 hours after the separately governed CELLECT closeout.

Authoritative receipt:
`PUBLIC_ARCHIVE_PARITY_ADJUDICATION.json`
(`d0cf4c80ece8cb442e232531b80e908fc34ecd06603eadc670819294766e3e15`).
