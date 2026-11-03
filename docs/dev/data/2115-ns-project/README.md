# #2115 `DISCOPT_NS_PROJECT` graduation panel data

Produced by `discopt_benchmarks/scripts/issue2115_graduation_panel.py`
on the 74-instance in-repo corpus (`python/tests/data/minlplib_nl`) plus 17 MINLPLib
`.nl` files fetched from minlplib.org (`m3 m6 m7 fo7 fo7_2 fo8 o7 o7_2 flay03m
flay04m slay04m slay07m clay0203h clay0204hfsg rsyn0820m netmod_dol1 sssd08-04`),
oracle `minlplib.solu` with `docs/dev/data/cert-optima.json` as fallback. Arms in
separate processes, interleaved per instance, order alternating; 2 instances at a
time on a 4-core container.

| file | configuration | role |
|---|---|---|
| `panel_default_noscip_60s.*` | default route, `pyscipopt` blocked, 60 s | **the graduation run** |
| `panel_default_scip_45s.*` | default route, `pyscipopt` importable, 45 s | superseded: SCIP does the #2069 confirmation, so the in-house driver barely runs |
| `panel_deterministic_scip_30s.*` | `deterministic=True`, `pyscipopt` importable, 30 s | superseded: that route never reaches the MILP driver on the m6/m7 class |

Graduation run: 952 executed checks, 0 violations; projection rescued 3885 node
bounds on 6 instances; certified 63 -> 64 (`m6`); wall where it ran 223.6 -> 161.6 s.
The two superseded runs are cert-clean too (881 and 946 checks, 0 violations) and
outcome-neutral (58 -> 58, 65 -> 65 certified), which is what they should be in a
configuration where the flag's code path is not on the critical path.

## `DISCOPT_SUBTOL_NOISE_GLOBAL` (#2115 follow-up)

| file | configuration | role |
|---|---|---|
| `panel_subtol_noscip_60s.*` | `--flag DISCOPT_SUBTOL_NOISE_GLOBAL`, default route, `pyscipopt` blocked, 60 s, `DISCOPT_NS_PROJECT` on in both arms | the graduation run |

951 executed checks, 0 violations; 22 noise-driven sub-`tol` pivots suppressed on
4 instances (`m6`, `m7`, `prolog`, `ex14_1_7`); certified 64 -> 64, wall flat --
cert-clean but outcome-neutral. Graduated by owner decision as a verdict fix.
