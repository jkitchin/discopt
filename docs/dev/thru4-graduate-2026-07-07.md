# THRU-4-graduate — cut-pool inheritance held-out validation: **stays default-OFF** (2026-07-07)

**Status: NOT graduated.** `DISCOPT_CUT_INHERIT` / `SolverTuning.cut_inherit`
remains default **OFF**. The broad held-out validation (Part 1) is fully clean —
0 soundness violations, root integrity, no lost certs, no incumbent harm, no
slowdowns — but the graduation's cert-panel arm (the flag ON against the
committed 41-instance `cert-baseline.jsonl`) exposes **two deterministic
flag-ON certificate losses** (`nvs06`, `tspn05`) that the held-out draw and
THRU-4's own probe/control set (#551) did not contain. Per the kill criterion
(`docs/dev/flag-graduation-protocol.md`: "any lost incumbent → that flag stays
OFF with the instance recorded") and CLAUDE.md §1, the flip does not ship.
**This PR changes no solver code** — it is the findings record. The
default-OFF path on this branch is byte-identical to `origin/main`.

**Branch:** `thru4-graduate` from `origin/main` @ `cd6e199d` (has #551).
**Predecessor:** THRU-4 shipped the flag default-OFF (#551,
`docs/dev/thru4-cut-inheritance-2026-07-07.md`). This is its (attempted)
graduation, mirroring ILS (#532) / governor (#541).

> **Method.** Apple M-series (arm64), Python 3.12, release build
> (`maturin develop --release`; pounce `_pounce.abi3.so` = 4.73 MB, not debug),
> `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`. Corpus:
> `~/Dropbox/projects/discopt-minlp-benchmark/` (oracle `minlplib.solu`,
> `=best=`/`=bestdual=` bracket). All A/B solves `time_limit=30,
> gap_tolerance=1e-4`, one instance per subprocess (fresh interpreter per
> config; flag propagated via env: OFF = unset, ON = `DISCOPT_CUT_INHERIT=1`),
> outer kill budget TL+120 s.

---

## 0. Executive summary

| gate | result |
|---|---|
| **Part 1 — held-out (49 scored A/B pairs, 11 classes)** | |
| soundness (hard) | **PASS** — 0 dual bounds crossing `=best=`, 0 false-optimal vs the `[=bestdual=, =best=]` bracket, 0 false-infeasible, in either config |
| root integrity | **PASS** — root bound ON ≡ OFF (≤1e-9) on all 47 rows where both configs surface one; the 2 remaining rows are OFF=None → ON=value, the *documented* flag-ON cold-path pool capture (#551 design), both values sound (≤ oracle) |
| cert preservation (held-out) | **PASS** — 5 instances certify `optimal` in both configs, objectives identical; 0 lost; 0 gained on this draw |
| incumbent no-harm | **PASS** — ON objective-at-exit same-or-better on all 49; 2 strict incumbent *gains* ON (powerflow0039r finds the oracle `=best=` 41869.05 that OFF misses; nuclearvb finds −1.02005 where OFF exits empty) |
| wall no-harm | **PASS** — 0 instances >10 % slower ON; sonet25v6 is 3.1× *faster* ON (144.7 s → 46.9 s to the same 3-node state); OFF (not ON) blows the 150 s kill budget on graphpart_clique-70 and ngone |
| fires | **PASS** — pool populates + inherits on 11/38 solved quadratic-slice instances (`pool/size` 4–69, `skipped_separations` up to 61); geomean wall ratio on the firing slice 1.004×, node ratio 0.998× (throughput-neutral here — the 2–5× is the dense integer-QP class, not this draw); true non-quadratic controls all inert with ~0 overhead |
| **Part 2 — cert-panel arm (flag ON vs `cert-baseline.jsonl`, 41 instances)** | |
| cert preservation (panel) | **FAIL** — `nvs06` and `tspn05` lose their certificates flag-ON, deterministically (solo-reproduced); `dispatch` (3→15) and `nvs13` (19→53) shift nodes with |Δobj| ≤ 5e-13 (benign perf notes) |

→ **NOT CLEAN. Default stays OFF. No rebaseline** (no default-path change; the
committed baseline still describes the shipped default, including its known
pre-existing nvs13 19→49 drift, #550).

---

## 1. Part 1 — held-out selection

Seeded (`seed=20260707`), stratified round-robin over MINLPLib probtypes,
`vars ≤ 400`, `=best=` oracle present, **excluding** the 61 vendored panel
instances and THRU-3/4's tuning probes (nvs10/13/17/19/23/24). Two strata:

- **quadratic slice** (48 drawn): QP, QCP, QCQP, BQP, BQCP, BQCQP, MIQCP,
  MBQP-family (MBQCP, MBQCQP) — where the square/PSD pool can fire;
- **control slice** (12 drawn + 10 supplemental): MBNLP / MINLP — where the
  pool should be inert.

21 of the 70 attempted rows are unscoreable: 17 blow the outer kill budget in
*both* configs (config-symmetric — the sonet/color_lab/qspp/ising BQP-family
roots and the eg_*/autocorr_bern* controls run >150 s before the first
node-loop deadline check; a pre-existing TL-enforcement gap orthogonal to this
flag), 2 are binary-`.nl` unreadable (color_lab2_4x0, portfol_roundlot), and 2
are asymmetric **in ON's favour** (graphpart_clique-70, ngone: OFF is killed
at 150 s, ON returns; ON side clean — no bound claims, no false certs). The
degenerate first control draw (cheapest-first eg_*/autocorr_bern* all
kill-budget-bound) was replaced by a supplemental held-out control draw of 10
known-30 s-solvable MBNLP/MINLP instances (G2's held-out set, none vendored,
none probes): ravempb, syn10m02m, enpro56pb, fo7_2, syn05m04hfsg,
cvxnonsep_{psig20,nsig20,normcon20,pcon20}, m7_ar3_1.

**Scored: 49 A/B pairs** across QCP 12, MBQCP 13, QCQP 4, BQCQP 3, BQCP 2,
MIQCP 1, MBQCQP 1, QP 1, BQP 1 (38 quadratic) + MINLP 6, MBNLP 5 (11
controls).

## 2. Part 1 — per-instance table (OFF vs ON)

`pool sz/skip` = flag-ON `pool/size` / `pool/skipped_separations` (— = no
pool). Every optimal-in-both row is objective-identical; every row where both
configs report a bound has bound ON ≡ OFF.

| instance | class | slice | OFF status | OFF obj | OFF bound | OFF n | OFF s | ON status | ON obj | ON bound | ON n | ON s | pool sz/skip |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| ex5_3_3 | QCQP | quad | feasible | 3.23402 | 1.63132 | 363 | 30.49 | feasible | 3.23402 | 1.63132 | 389 | 30.28 | - |
| elf | MBQCP | quad | feasible | 0.328 | 0 | 1341 | 30.12 | feasible | 0.328 | 0 | 1341 | 30.19 | - |
| crudeoil_pooling_ct1 | MBQCQP | quad | time_limit | - | 50013 | 31 | 32.62 | time_limit | - | 50013 | 31 | 32.45 | - |
| sonet22v4 | BQCP | quad | time_limit | - | - | 1 | 84.82 | time_limit | - | - | 1 | 84.38 | - |
| sonet22v5 | BQCQP | quad | time_limit | - | -78215.2 | 7 | 38.95 | time_limit | - | -78215.2 | 7 | 39.03 | - |
| tln12 | MIQCP | quad | time_limit | - | 16.0415 | 767 | 31.99 | time_limit | - | 16.0415 | 735 | 31.35 | - |
| hybriddynamic_fixedcc | QP | quad | feasible | 1.47352 | 0.768 | 825 | 30.25 | feasible | 1.47352 | 0.768 | 825 | 30.34 | - |
| kall_congruentcircles_c61 | QCP | quad | feasible | 1.28761 | -1e-09 | 59 | 30.02 | feasible | 1.28761 | -1e-09 | 59 | 30.09 | 44/34 |
| powerflow0014r | QCQP | quad | feasible | 8082.58 | 3.15e-11 | 571 | 31.87 | feasible | 8082.58 | 3.15e-11 | 571 | 31.61 | - |
| ringpack_10_1 | MBQCP | quad | feasible | -8.69299 | -20.8582 | 31 | 30.46 | feasible | -8.69299 | -20.8582 | 31 | 30.17 | 27/15 |
| sonet23v6 | BQCP | quad | time_limit | - | - | 1 | 109.24 | time_limit | - | - | 1 | 107.93 | - |
| sonet23v4 | BQCQP | quad | time_limit | - | -53974.4 | 3 | 40.25 | time_limit | - | -53974.4 | 3 | 39.92 | - |
| kall_circles_c6b | QCP | quad | feasible | 2.24292 | -1.01e-09 | 75 | 30.63 | feasible | 2.24292 | -1.01e-09 | 75 | 30.54 | 31/46 |
| ringpack_10_2 | MBQCP | quad | feasible | -8.69299 | -20.8582 | 15 | 31.2 | feasible | -8.69299 | -20.8582 | 15 | 31.11 | 69/5 |
| kall_circles_c6c | QCP | quad | feasible | 3.05424 | -1e-09 | 115 | 30.15 | feasible | 3.05424 | -1e-09 | 113 | 30.62 | 30/57 |
| powerflow0030r | QCQP | quad | feasible | 576.893 | 0 | 185 | 33.33 | feasible | 576.893 | 0 | 181 | 32.85 | - |
| sssd18-08persp | MBQCP | quad | feasible | 857521 | 4459.55 | 95 | 31.67 | feasible | 857521 | 4459.55 | 95 | 31.77 | - |
| sonet25v6 | BQCQP | quad | time_limit | - | - | 3 | 144.66 | time_limit | - | - | 3 | 46.93 | - |
| pointpack10 | QCP | quad | feasible | 0.177467 | 1.0625 | 607 | 30.12 | feasible | 0.177467 | 1.0625 | 607 | 30.06 | - |
| powerflow0039r | QCQP | quad | time_limit | - | 27033.8 | 3 | 48.39 | **feasible** | **41869.1** | 27033.8 | 45 | 33.6 | - |
| ringpack_20_1 | MBQCP | quad | time_limit | - | -41.7164 | 3 | 37.74 | time_limit | - | -41.7164 | 3 | 37.49 | 34/0 |
| kall_circles_c8a | QCP | quad | feasible | 3.74344 | -1e-09 | 3 | 8.5 | feasible | 3.74344 | -1e-09 | 3 | 8.26 | 4/3 |
| sssd20-08persp | MBQCP | quad | feasible | 476047 | 4673.92 | 159 | 30.06 | feasible | 476047 | 4673.92 | 191 | 30.7 | - |
| pointpack12 | QCP | quad | feasible | 0.151111 | 1.0625 | 95 | 32.42 | feasible | 0.151111 | 1.0625 | 95 | 32.45 | - |
| ringpack_20_2 | MBQCP | quad | time_limit | - | -41.7164 | 1 | 31.84 | time_limit | - | -41.7164 | 1 | 31.85 | 23/0 |
| pointpack14 | QCP | quad | feasible | 0.116727 | 1.0625 | 95 | 37.17 | feasible | 0.116727 | 1.0625 | 95 | 37.01 | - |
| color_lab3_4x0 | BQP | quad | time_limit | - | - | 1 | 64.46 | time_limit | - | - | 1 | 64.23 | - |
| ringpack_20_3 | MBQCP | quad | time_limit | - | -41.7164 | 1 | 51.4 | time_limit | - | -41.7164 | 1 | 51.17 | - |
| knp3-12 | QCP | quad | feasible | 1.10557 | 2.64438 | 127 | 30.86 | feasible | 1.10557 | 2.64438 | 127 | 30.66 | 64/61 |
| gabriel02 | MBQCP | quad | time_limit | - | 48.3906 | 7 | 33.49 | time_limit | - | 48.3906 | 7 | 33.5 | - |
| waterund01 | QCP | quad | feasible | 86.8333 | 83.4375 | 15 | 31.23 | feasible | 86.8333 | 83.4375 | 15 | 31.21 | - |
| hydroenergy1 | MBQCP | quad | feasible | 209418 | 213983 | 127 | 31.98 | feasible | 209418 | 213983 | 127 | 31.98 | - |
| kall_circlespolygons_c1p12 | QCP | quad | feasible | 0.339602 | -1e-09 | 3 | 31.39 | feasible | 0.339602 | -1e-09 | 3 | 31.14 | 63/3 |
| crudeoil_li01 | MBQCP | quad | time_limit | - | 5245.51 | 7 | 32.45 | time_limit | - | 5245.51 | 7 | 32.54 | - |
| kall_circlespolygons_c1p13 | QCP | quad | feasible | 0.339602 | -1e-09 | 5 | 64.77 | feasible | 0.339602 | -1e-09 | 5 | 64.74 | 46/2 |
| nuclearva | MBQCP | quad | time_limit | - | - | 31 | 31.23 | time_limit | - | - | 31 | 31.35 | - |
| kall_circlesrectangles_c1r12 | QCP | quad | feasible | 0.339602 | 0 | 661 | 30.2 | feasible | 0.339602 | 0 | 659 | 30.19 | - |
| nuclearvb | MBQCP | quad | time_limit | - | - | 63 | 33.91 | **feasible** | **-1.02005** | - | 3 | 31.12 | - |
| spring | MINLP | ctrl | optimal | 0.846244 | 0.846226 | 49 | 9.78 | optimal | 0.846244 | 0.846226 | 49 | 9.78 | 6/47 |
| ravempb | MBNLP | ctrl | optimal | 269590 | 269590 | 189 | 21.37 | optimal | 269590 | 269590 | 189 | 21.53 | - |
| syn10m02m | MBNLP | ctrl | feasible | 1453.9 | 3163.6 | 551 | 30.48 | feasible | 1453.9 | 3163.6 | 553 | 30.21 | - |
| enpro56pb | MBNLP | ctrl | feasible | 263428 | 261679 | 603 | 31.09 | feasible | 263428 | 261679 | 603 | 31.08 | - |
| fo7_2 | MBNLP | ctrl | time_limit | - | - | 1951 | 30.18 | time_limit | - | - | 1951 | 30.2 | - |
| syn05m04hfsg | MBNLP | ctrl | feasible | 5510.39 | 10254.4 | 411 | 30.1 | feasible | 5510.39 | 10254.4 | 411 | 30.36 | - |
| cvxnonsep_psig20 | MINLP | ctrl | optimal | 93.8114 | 93.8023 | 33 | 0.77 | optimal | 93.8114 | 93.8023 | 33 | 0.65 | - |
| cvxnonsep_nsig20 | MINLP | ctrl | optimal | 80.9493 | 80.9435 | 67 | 1.06 | optimal | 80.9493 | 80.9435 | 67 | 1.1 | - |
| cvxnonsep_normcon20 | MINLP | ctrl | optimal | -21.7491 | -21.7491 | 25 | 0.67 | optimal | -21.7491 | -21.7491 | 25 | 0.67 | - |
| cvxnonsep_pcon20 | MINLP | ctrl | feasible | -21.5123 | -49.75 | 2527 | 30.02 | feasible | -21.5123 | -49.75 | 2527 | 30.02 | - |
| m7_ar3_1 | MINLP | ctrl | time_limit | - | - | 1183 | 30.32 | time_limit | - | - | 1183 | 30.19 | - |

(hydroenergy1 / pointpack* / knp3-12 are maximize instances — bound above the
incumbent is the correct sense; identical in both configs regardless.)

Notes:

- **Root integrity, precisely:** the two OFF=None → ON=value root-bound rows
  are sonet25v6 (ON root −1337280.0 ≤ oracle −30660) and powerflow0039r
  (ON root 2.0 ≤ oracle 41869) — the flag-ON cold-path pool probe *surfaces* a
  (loose, valid) root bound where the OFF path exits without reporting one.
  On every row where OFF reports a root bound, ON is identical to ≤1e-9.
- **The firing slice is throughput-neutral here** (geomean wall 1.004×, nodes
  0.998×, n=10): on these held-out QCP/MBQCP instances the per-node square/PSD
  loops are *not* the bottleneck the way they are on the dense integer-QP
  probes, so skipping them neither helps nor harms — bounds and node counts
  come out identical. The flag's 2–5× is class-specific (nvs-style dense
  integer QP) plus the cold-path root-wall rescues visible on sonet25v6
  (3.1×), powerflow0039r (1.44× + incumbent), graphpart_clique-70/ngone
  (OFF killed at 150 s, ON returns).
- **`spring` (control) fires soundly**: MINLP by probtype but carries
  univariate squares, so the pool correctly keys on structure, not label —
  6 rows pooled, 47 skips, byte-identical certificate (0.846244, 49 nodes).

## 3. Part 2 — the blocker: cert-panel losses flag-ON

`check_cert_neutrality.py` in a fresh interpreter with the flag ON against the
committed 41-instance `docs/dev/data/cert-baseline.jsonl` (the graduation
gate's cert-panel arm): 37/41 byte-identical; 4 rows shift, of which 2 are
disqualifying:

| instance | baseline | flag-ON | classification |
|---|---|---|---|
| dispatch | optimal, 3 nodes | optimal, 15 nodes, \|Δobj\|=4.6e-13 | benign perf note (pool fires; objective identical) |
| nvs13 | optimal, 19 nodes | optimal, 53 nodes, \|Δobj\|=0 | 19→49 is the known pre-existing main drift (#550); 49→53 is the flag's own, objective identical (matches #551's control table) |
| **nvs06** | **optimal 1.7703125, 5 nodes, 7.0 s/10 s** | **feasible 231.70004, 1 node, 1.5 s** | **CERT LOST + incumbent degraded 130×, deterministic** |
| **tspn05** | **optimal 191.25521, 39 nodes, 27.8 s/60 s** | **feasible 191.25521 (same incumbent), bound 190.279, 205 nodes, 60 s TL** | **CERT LOST — cannot close without per-node re-separation** |

Solo reproductions (fresh subprocess each, same budgets):

- `nvs06` (pure-integer nonconvex, budget 10 s): `=0` → optimal 1.7703125002,
  bound 1.7703124984, 5 nodes, 1.36 s. Flag-ON → **feasible 231.70004342,
  bound 1.10001500, 1 node, 1.52 s** — it terminates after ONE node with
  8.5 s of budget unused. The flag-ON cold-path capture reroutes the root
  flow: instead of the OFF path's feasibility pump → NLP-relaxation pump →
  integer local search (which finds 1.77031 pre-tree), the ON path seeds
  `SubNLP incumbent obj=251.3` then `Box-search incumbent obj=231.7` and the
  node loop ends after node 1 without exhausting the budget or closing the
  gap (bound 1.10 vs incumbent 231.7). The bound itself stays *valid*
  (1.10 ≤ 1.77) — this is not a false certificate, it is a **premature exit
  with a lost certificate and a badly degraded incumbent-at-exit**, i.e. a
  bug in the flag-ON path on the pure-integer cold-path class, hidden until
  now because #551's probe/control set (nvs10/13/17/19/23/24) happens not to
  contain an instance that routes this way.
- `tspn05` (spatial, budget 60 s): `=0` → optimal 191.2552077, 39 nodes,
  18.9 s. Flag-ON → feasible 191.2552077 (identical incumbent), bound
  190.2786, 201 nodes, 60 s TL. Here the per-node square/PSD re-separation is
  **load-bearing for closure**: the inherited root pool alone leaves each node
  relaxation too loose to fathom, the tree balloons 39→201+ nodes, and the
  certificate is lost at budget. Skipping re-separation is sound (bound stays
  valid, incumbent identical) but not free on this class — the THRU-3
  "re-separation not needed" measurement was made on nvs19/nvs24 and does not
  transfer to tspn05's structure.

## 4. Verdict + disposition

- **Default stays OFF.** Both panel losses are deterministic, within-budget,
  and class-general (a root-flow reroute on pure-integer cold-path models; a
  closure dependency on per-node separation for tspn05-shape spatial models) —
  not tolerance flakes. The kill criterion of the graduation protocol fires.
- **No rebaseline.** The shipped default path is untouched; the committed
  `cert-baseline.jsonl` still describes it (including the known, pre-existing
  nvs13 19→49 drift recorded by #550/#541 — refreshing that stale row remains
  out of scope here for the same reason it was in #551: it is main-drift, not
  this branch's).
- **Follow-ups before any future flip attempt** (recorded, not shipped):
  1. **Fix the nvs06-class premature exit** under `cut_inherit` on the
     pure-integer cold path — root-cause why the pool probe reroutes the
     pre-tree incumbent search (feasibility pump skipped) and why the node
     loop ends at node 1 with budget remaining. This is a bug in the flag
     path itself and should be filed on the correctness backlog (#396 list)
     even while the flag is OFF — an opt-in user can hit it today.
  2. **Hybrid re-separation trigger** for the tspn05 class: inherit the pool
     but re-separate when a node's LP bound stalls against its parent (the
     "lazy trigger" THRU-4 §1 skipped on nvs-only evidence) — the measurement
     here falsifies "re-separation never needed" as a *class* claim.
  3. The 17 both-config outer-timeout instances expose a **TL-enforcement
     gap** (root/separation work not deadline-checked for >5× the time
     limit) — orthogonal to this flag, worth its own probe.

## 5. Gates run (this branch: docs-only, no solver code)

| gate | result |
|---|---|
| `pytest -m smoke` (python/tests) | 620 passed, 14 skipped |
| `pytest -m slow python/tests/test_adversarial_recent_fixes.py` | 10 passed |
| `ruff check` / `ruff format --check` (v0.14.6) | clean |
| pre-commit `mypy` (whole `python/discopt/`) | Passed |
| `cargo test -p discopt-core` | n/a — no Rust touched |
| cert-neutrality, default path (flag unset = OFF) | unchanged — this branch ships no code |
