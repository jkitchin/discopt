# Flag graduation — verdicts 2 & 3 (follow-on to BR-3 #602, 2026-07-10)

**Task:** run graduation **verdicts 2 and 3** (two independent held-out gate runs)
for the three flags BR-3 (`docs/dev/br3-regate-2026-07-10.md`) found
GRADUATE-ELIGIBLE at **verdict 1/3**. Per T2.6, a bound-changing flag needs **3
consecutive green verdicts** before any default-ON flip. Each flag is composed
with `DISCOPT_LU_DENSITY_ROUTE=1` (the #591 / A-2 dense retry), as BR-3
established:

1. `DISCOPT_LU_DENSITY_ROUTE` — the engine retry itself (#74 line).
2. `DISCOPT_OBJ_BRANCH_PRIORITY` — BR-3: nvs21 cured, 1144→1102 nodes.
3. `DISCOPT_LIFT_LOOSE_PRODUCTS` — TD-A; BR-3: nvs09 gap tightens.

**This PR flips no defaults.** Outcome: **none of the three flags graduate** — a
**new, deterministic certificate loss on `nvs22`, introduced by the density route
itself**, surfaces at verdict 3 and fails all three arms (they all compose with the
route). Details and root-cause below.

## Build & method

- **Build:** `maturin develop --release` in this worktree
  (`agent-a045b99cdfea826ff`). Verified worktree-local:
  `discopt.__file__ = …/agent-a045b99cdfea826ff/python/discopt/__init__.py`,
  `discopt._rust.__file__ = …/python/discopt/_rust.cpython-312-darwin.so`.
  `pounce-solver 0.7.0`, `jax 0.10.2`, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`.
  Isolated subprocess per solve (env flags read fresh at import).
- **Methodology:** the BR-3 / #581 house pattern. Shared all-OFF baseline; each
  flag ON = *flag + `DISCOPT_LU_DENSITY_ROUTE=1`*. Two **independent** verdicts with
  **different** held-out panels / conditions.
- **Verdict 2:** 28-instance panel = BR-3's structure with a **different instance
  draw**. Blocker probes kept (nvs21, st_e36, nvs09, alkyl, tls2); fresh
  integer-NLP (nvs03/05/08/10/11/12/16/18), fresh QP/MIQP (st_miqp1/4/5, st_test2,
  st_testgr1), fresh spatial/bilinear (ex5_2_2_case2, ex5_2_4, ex7_2_2, ex4_1_2,
  ex4_1_8, ex8_1_1, st_bpaf1a), fresh tree (st_e29, st_e31, gbd). **TL = 40 s.**
- **Verdict 3:** 35-instance **larger + longer** panel. Blockers kept; broader mix
  including nvs14/20/22/01/04/06/13/15/23, st_miqp2/3/5, st_test2, meanvarx, alan,
  ex5_2_2_case1, ex5_2_4, ex7_2_2/2_3, ex4_1_3/1_8/1_9, ex8_1_1/1_6, st_bpaf1a/1b,
  st_e29/e31, gbd, gkocis. **TL = 60 s.**
- **Criteria (per flag per verdict):** `incorrect_count = 0` (certified objective
  vs oracle, abs 1e-4 / rel 1e-3, zero slack); **zero certificate losses** (OFF
  optimal → ON non-optimal = hard fail); zero oracle crosses; objective drift ≤ abs
  1e-4 / rel 1e-3; node-count delta (deterministic — the graded metric). A run that
  never engages the flag is INCONCLUSIVE, not green.
- **Oracle:** `~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu` (`=opt=`,
  or the `[=bestdual=, =best=]` bracket). Full snapshot corpus
  (`minlplib/nl/`, 4 830 instances) — not the in-repo 61-file subset.
- **Concurrency hygiene (§0.7):** a co-tenant agent (arescue) shared the machine;
  load avg ranged ≈ 3.5–11 during the runs. **All verdicts rest on deterministic
  node counts / status / bound / objective, not wall time.** The nvs21 OFF baseline
  reproduced byte-identically to BR-3 (195 n), and every liveness delta matched
  BR-3 exactly (below), confirming the build and determinism. Wall times recorded
  but used for **no** verdict.
- **Artifacts:** `docs/dev/data/flag-graduation-verdicts23-2026-07-10/`
  (per-arm JSON for v2 & v3, `liveness.json`, `nvs22_rootcause.json`, and the
  harness `gate.py` / `verdict.py` / `liveness.py` / `solve_one.py`).

## Flag-engagement proof (setup step 2 — each flag LIVE in THIS build)

Every liveness delta matched BR-3's table exactly, confirming a correct,
deterministic build before any measurement (`liveness.json`):

| flag | probe | OFF | ON (flag alone) | engaged? |
|---|---|---|---|---|
| `DISCOPT_LU_DENSITY_ROUTE` | nvs21 | optimal 195 n | optimal **191 n** | **yes** |
| `DISCOPT_OBJ_BRANCH_PRIORITY` | nvs01 | optimal 17 n | optimal **11 n** | **yes** |
| `DISCOPT_OBJ_BRANCH_PRIORITY` | nvs13 | optimal 45 n | optimal **35 n** | **yes** |
| `DISCOPT_LIFT_LOOSE_PRODUCTS` | nvs09 | feasible bnd −58.65 | feasible bnd **−47.06** | **yes** |

Additional in-panel engagement proof for `lift_loose_products` (v2, nvs09): the
route alone gives bound −58.73 / 159 n; **lift + route gives −45.78 / 117 n** — the
tighter bound is the lift mechanism, not the route. TAIL-1 reproduces.

## Verdict 2 (28 instances, TL = 40 s) — all three GREEN

Shared OFF baseline: every certified objective matches the oracle; nvs21 195 n,
st_e36 153 n, alkyl optimal 29 n (no square/route interaction here). nvs09, nvs05,
tls2, ex5_2_2_case2, ex7_2_2, st_e31 are non-optimal at OFF (feasible/time_limit),
so they cannot register a cert loss.

| flag (ON = flag + route) | engaged? | incorrect | oracle-cross | cert-loss | node Δ (both-certified) | verdict |
|---|:-:|:-:|:-:|:-:|---|---|
| **lu_density_route** | yes | 0 | 0 | **0** | 1050 → 1046 (nvs21 195→191) | **GREEN** |
| **obj_branch_priority** | yes | 0 | 0 | **0** (alkyl stays optimal 29 n) | 1050 → 1038 (nvs21 195→191, nvs18 105→97) | **GREEN** |
| **lift_loose_products** | yes | 0 | 0 | **0** | 1050 → 1046 (nvs21 195→191); nvs09 bnd −58.65→−45.78, 191→117 n | **GREEN** |

All three flags are green on verdict 2. No composition loss on this panel (the
alkyl loss BR-3 saw was `square_cost_gate`×route, not tested here).

## Verdict 3 (35 instances, TL = 60 s) — all three RED (nvs22 route cert loss)

Shared OFF baseline: nvs22 solves **optimal, 35 n** (obj 6.05821994, bound
6.058219999). All ON arms compose with the density route, and the route flips
nvs22 to **feasible, 37 n**:

| flag (ON = flag + route) | engaged? | incorrect | oracle-cross | cert-loss | node Δ (both-certified) | verdict |
|---|:-:|:-:|:-:|---|---|---|
| **lu_density_route** | yes | 0 | 0 | **1: nvs22** optimal→feasible | 1057 → 1055 (nvs21 195→191, nvs13 45→47) | **RED** |
| **obj_branch_priority** | yes | 0 | 0 | **1: nvs22** optimal→feasible | 1057 → 1037 (nvs21 195→191, nvs01 17→11, nvs13 45→35) | **RED** |
| **lift_loose_products** | yes | 0 | 0 | **1: nvs22** optimal→feasible | 1057 → 1055 (nvs21 195→191, nvs13 45→47) | **RED** |

`incorrect_count = 0` and no oracle cross on every arm — the nvs22 answer is
numerically correct. But **OFF optimal → ON non-optimal is a hard certificate loss
(§0.1, zero slack)**, so all three arms fail verdict 3.

Note the loss appears in the **`lu_density_route` (route-alone) arm** — it is the
route's own behavior, inherited by the other two arms because they compose with the
route. It is *distinct* from anything BR-3 saw: not the nvs21 loss the route
**cured** (#591), not the `square_cost_gate`×route alkyl loss, not the
`node_reduce` st_e36 loss.

## Root-cause: the density route regresses nvs22 (new finding)

Deterministic, isolated re-runs (`nvs22_rootcause.json`), 2–3 reps each,
byte-identical:

| nvs22 | route **OFF** | route **ON** |
|---|---|---|
| TL = 40 s | **optimal, 35 n**, bound 6.058219998999999 | **feasible, 37 n**, bound 6.0582199951512 |
| TL = 60 s | **optimal, 35 n**, bound 6.058219998999999 | **feasible, 37 n**, bound 6.0582199951512 |

**Not TL-sensitive** — route ON gives the identical `feasible, 37 n` result at both
40 s and 60 s, so this is a **hard certificate loss, not a slower-certify** (which
would *not* block graduation). The route changes the LU factorization path at nodes
that qualify for the density-aware route (#557), which changes the node-LP
solutions → a slightly different search tree (37 vs 35 nodes) whose **final dual
bound never formally closes the gap**: route's bound 6.0582199951512 sits ~9.8e-9
(relative) above its incumbent 6.0582199356118, so the solver terminates
`feasible`; OFF's 35-node tree closes the same gap and terminates `optimal`. Both
incumbents match the oracle 6.05822 to ~6e-8 (well within abs 1e-4). The certificate
— not the answer — is lost.

This is the same *class* as the nvs21 issue #591 targets (LU-route change perturbs
the per-node LP and therefore the fathoming), but here the perturbation goes the
**wrong** way: the retry that *rescued* nvs21's stuck bound *introduces* a
non-closing tree on nvs22. It is a concrete, reproducible blocker that keeps the
density route **opt-in**, and it blocks every flag that composes with the route.

## Graduation outcome

| flag | verdict 1 (BR-3) | verdict 2 | verdict 3 | consecutive greens | GRADUATE? |
|---|:-:|:-:|:-:|:-:|:-:|
| `DISCOPT_LU_DENSITY_ROUTE` | GREEN | GREEN | **RED (nvs22)** | 2 (broken) | **NO** |
| `DISCOPT_OBJ_BRANCH_PRIORITY` | GREEN | GREEN | **RED (nvs22 via route)** | 2 (broken) | **NO** |
| `DISCOPT_LIFT_LOOSE_PRODUCTS` | GREEN | GREEN | **RED (nvs22 via route)** | 2 (broken) | **NO** |

**None of the three flags graduate.** Each reached 2 consecutive greens (BR-3
verdict 1 + this verdict 2) but broke the streak at verdict 3 on the nvs22 route
loss. T2.6 requires 3 **consecutive** greens; the count resets. **No default-ON
flip is prepared or committed** — deliberately, because the shared composition
(*flag + route*) is not clean on the verdict-3 panel.

The obj_branch_priority and lift_loose_products flags are **not themselves** the
cause of the verdict-3 failure (both are clean of new losses on their own; nvs22
loses in the route-alone arm too). But because the graduation configuration under
test is *flag + route* — and the route is the mechanism BR-3 required to cure the
old blockers — a route regression blocks the composite. Any future re-gate of these
two flags must either (a) first fix the density-route nvs22 regression, or (b)
re-scope the composition.

## Recommended next step (not done here)

- **Entry experiment for the density-route nvs22 regression:** instrument which
  nvs22 nodes take the density-aware route (#557) vs the historical dense-preferring
  path, and why the route tree (37 n) fails to fathom the last open node that OFF
  (35 n) fathoms. Kill criterion: if the route can be made to reproduce OFF's
  fathoming on nvs22 without reintroducing the nvs21 stuck-bound (#591), the route
  becomes a clean graduation candidate again and this verdict can be re-run. Until
  then the density route — and therefore all three composed flags — stays opt-in.

## Verification for the PR

- `import discopt._rust` verified worktree-local (setup step 1).
- Every flag proven LIVE against BR-3's exact liveness numbers (setup step 2).
- Verdict data + oracle-checked analysis committed under
  `docs/dev/data/flag-graduation-verdicts23-2026-07-10/`.
- No solver math changed (docs + data only); no defaults flipped.
