# SA-SWAP wave-1 DAG

Status: **FROZEN PHASE-1 EXECUTION CONTRACT**

```text
S0 authenticate root inputs and immutable APNCC/Trackastra suppliers
  |
  +--> S1 freeze exact occupied-conflict/no-op preregistration
         |
         +--> S2 unit + adversarial compiler checks
                |
                +--> S3 fresh 45 s zero-swap preflight
                       |
                       +--> S4 44b6 grouped-OOF exact-regret graph falsifier
                              |
                              +-- FAIL/PARK --> stop; 6bba and CV57 closed
                              |
                              +-- GO (adj >= +.005, raw/node >= 0)
                                      |
                                      +--> fresh 45 s preflight
                                             |
                                             +--> S5 identical 6bba gate
                                                    |
                                                    +-- FAIL --> stop; CV57 closed
                                                    +-- GO --> report only;
                                                        CV57 remains a root gate
```

The alpha consumes only the frozen 49-column APNCC/Trackastra feature cache. The
capacity-matched beta zeros column 48 (`trackastra_margin`). Both predict the exact
single-action adjusted-edge regret against an explicit no-op. Every emitted action
and final batch must compile under the frozen N1 interface with complete incident
conflict ownership.

Quarantined C2/K-best and Branch-B claims are outside this DAG. SA-REPR outputs are
also outside wave 1; integration is permitted only after the root barrier freezes a
new supplier for Phase 2.
