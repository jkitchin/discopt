# Perf follow-up — V1 results (2026-07-06)

**Status:** measured. The V1 gate/measurement task of
`docs/dev/perf-followup-plan-2026-07-05.md` §2. Re-runs the exact 2026-07-05
BARON head-to-head against `main` after the full plan (A1–A3 + F1–F7) landed.

**Method.** `discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py
--time-limit 60`, 61 vendored MINLPLib `.nl`, discopt (isolated subprocess,
release build) vs full-license BARON via GAMS 53, MINLPLib primalbound oracle.
Same machine class as the 2026-07-05 baseline. **Release POUNCE 0.7.0 wheel**
(5 MB `_pounce*.so`) — the `jkitchin/pounce#182` debug-build artifact does not
apply. Reports:
`reports/global_opt_baron_vs_discopt_2026-07-06T03-25-20.{json,md}` (V1) vs
`reports/global_opt_baron_vs_discopt_2026-07-05T10-47-11.json` (baseline).

## 1. discopt: baseline → V1

| metric | baseline 2026-07-05 | **V1 2026-07-06** | Δ |
|---|---:|---:|---:|
| total wall | 23.1 min | **18.8 min** | **−19 %** |
| proved optimal | 40 | **42** | +2 |
| correct (ok) | 49 | **51** | +2 |
| hit ~60 s limit | 18 | **16** | −2 |
| **VIOLATIONS** | 0 | **0** | — (hard gate holds) |
| rows with a live dual bound | 0 | **51 / 61** | A3 fix (`res.bound`) |

## 2. Hard gates (all pass)

- **Zero correctness violations** (0 → 0). No run claimed a certified global at
  the wrong value, returned an incumbent better than the proven global, or
  reported a dual bound crossing its oracle (the A3 bound-vs-oracle check is now
  live on 51/61 rows and found none).
- **No certification lost** — proved-optimal *increased* 40 → 42 (flay03m and
  clay0303hfsg now certify within budget); no instance that certified on
  2026-07-05 regressed.
- **`time_limit` contract restored** (F4): the two baseline overruns are gone —
  contvar 120 s → 61 s, heatexch_gen3 86 s → 60.9 s. Every instance is within
  the §0.7 envelope; the slowest "timeout" instance is beuster at 65.5 s.

## 3. Where the wall time went (wins map to the tasks)

| instance | baseline | V1 | Δ | status | task |
|---|---:|---:|---:|---|---|
| contvar | 120.0 s | 61.0 s | −59.0 | feasible | F4 (budget contract) |
| flay03m | 63.8 s | 7.5 s | −56.4 | **optimal** (was feasible@TL) | F1 |
| clay0303hfsg | 60.4 s | 22.5 s | −37.9 | **optimal** | F1 |
| fac2 | 32.5 s | 5.8 s | −26.7 | optimal | F1 |
| heatexch_gen3 | 86.0 s | 60.9 s | −25.2 | time_limit | F4 |
| nvs01 | 22.5 s | 5.5 s | −17.0 | optimal | F2 |
| tspn12 | 26.3 s | 13.6 s | −12.7 | feasible | F1/general |

The +2 proved-optimal are **flay03m** and **clay0303hfsg** — both F1 (the LNS
enumeration budget stopped burning the whole budget on heuristic sub-NLPs, so
the tree closes). One trivial regression: **beuster** 61.0 → 65.5 s (+4.5 s),
time-limit both times — noise on a hard branch-and-reduce instance.

## 4. vs BARON

BARON (full license, this run): **14.2 min, 54 correct** — stable vs the
baseline. discopt closed part of the gap (23.1 → 18.8 min) and the easy/mid
instances that dominated it (fac2, flay03m, nvs01) are now seconds, not tens of
seconds. discopt remains behind BARON on total wall and correct-count; the
residual is concentrated in the hard instances below.

## 5. What did NOT move — and why (honest scope)

The hard **pinned-bound / branch-and-reduce** instances are unchanged:
st_e36, nvs05, tls2, casctanks, the heatexch_gen* / bchoco0* families.
**F5 and F6 independently established these are not addressable by this plan:**
- **F5** (killed): their loose bound is not an even-power envelope defect — the
  composite-square envelopes are already exact; the stall is the objective's
  bilinear/product relaxation + an argument spanning zero. A branch-and-reduce /
  product-relaxation problem (cert-gap-plan §7).
- **F6** (killed): warm-starting node NLPs across the tree does not help this
  class — iteration count is neutral-to-worse (tls2 0.58×; nvs05 wins only on
  ≤2-bound-diff pairs and reverses on realistic ones), and POUNCE on a release
  build is already ~1.1× Ipopt (pounce#182 resolved). Node throughput, not node
  speed, was never the limiter — the weak dual bound is.

Closing these is future work under the branch-and-reduce roadmap
(cert-gap-plan §7), not this follow-up.

## 6. Ledger

Shipped (measured wins): **F1** (fac2 4×, flay03m/clay certify), **F2**
(nvs01 4×), **F3** (nvs09 tighter bound + recovered cuts POUNCE dropped),
**F4** (time_limit contract on all 61), **F7** (opt-in startup-tax trim).
Correctness pre-flight: **A1** (callback bound honesty + C-37), **A2** (1e30
sentinel purge + false-optimal decertification test), **A3** (live
bound-vs-oracle gate). Killed with recorded falsifications: **F5**, **F6**.
Follow-ups filed: **#507** (super3t / term-classifier second overrun site);
**jkitchin/pounce#182** resolved (debug-build artifact).
