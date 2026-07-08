# C-43 — nvs22 cut-inherit false-optimal fixed (pool-infeasible re-verify);
# broad graduation KILLED by an nvs19 cert loss the fix exposes. Flag stays opt-in. (2026-07-08)

**Status.** Phase 1 (the soundness fix) SHIPS. Phase 2 (the default-ON flip) is
**KILLED** and cut-inheritance stays **opt-in** (default force-off, byte-identical
to `origin/main`). The nvs22 false-optimal (#564, the CUT-INHERIT-GRAD blocker)
is fixed and nvs22 flag-ON now *certifies the oracle optimum*. But root-causing it
showed the pre-fix flag path was resting on **unsound sub-box fathoms** across the
class; making them sound costs `nvs19` its certificate (a Phase-2 gate failure),
so the flip is killed and the deeper source fix is filed as #567.

> **Method.** Apple M-series arm64, Python 3.12, release build
> (`maturin develop --release`; pounce `_pounce.abi3.so` = 4.51 MB, not debug),
> `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, corpus
> `~/Dropbox/projects/discopt-minlp-benchmark/`, oracle `minlplib.solu`. Single
> serial runs per arm, fresh interpreter per solve.

---

## Part 1 — root cause (file:line + mechanism)

### 1a. Reproduce (confirmed)

`nvs22` (MINLP, pure-integer nonconvex; oracle `6.0582200`), TL 25 s:

| arm | status | objective | bound |
|---|---|---:|---:|
| default (force-off) | optimal | **6.0582** (correct) | 6.0582 |
| `DISCOPT_CUT_INHERIT=1` (force-on) | optimal | **33.55166** (FALSE) | 33.55166 |
| `=gated` (structure gate) | optimal | **33.55166** (FALSE) | 33.55166 |

Deterministic at TL 25/60/120 s — the pre-fix tree *stably* (but falsely) closes
at 33.55. `pool/dropped_nodes` is unset: C-42's cold-path drop-retry never fires.

### 1b. The mechanism — a FALSE Farkas-infeasible fathom (not a "reroute")

#564 framed this as an incumbent-search reroute on the pool-*success* path. The
measurement (CLAUDE.md §4 — the measurement wins) found something more direct and
more serious: a **false fathom**. Instrumenting node solves by whether the node
box contains the true optimum integer assignment `(x0,x1,x2,x3)=(5,1,1,2)`:

| arm | OPT-containing nodes | verdicts |
|---|---:|---|
| pool-inherited (`both`) | 3 | `optimal −4.37`, `optimal 1.84`, **`infeasible`** |
| pool-free (`skip_only`) | 7 | all `optimal`, down to `6.058` |

A node whose box **contains the feasible optimum 6.0582** is certified
**Farkas-infeasible** once the inherited pool is appended, while the pool-free
relaxation of the *same box* solves to `optimal`. The Farkas ray certifies the
*pool-augmented* polytope empty — a real proof only if every pool row is valid on
the box. Here a pool row is **not valid on that sub-box**, so it is a false
fathom: the region holding 6.0582 is pruned, the tree closes around the
suboptimal incumbent 33.55, and a false `optimal` is certified. The dual bound
33.55 crosses the oracle 6.0582 — a false certificate (CLAUDE.md §1 hard stop).

Isolation A/B (monkeypatched `solve_at_node`, TL 25 s):

| config | inherited cuts | skip separators | result |
|---|---|---|---|
| `skip_only` | no | yes | **optimal 6.0582** (correct) |
| `cuts_only` | yes | no | feasible 33.55, bound **−4.37** (valid, no cross) |
| `both` | yes | yes | **optimal 33.55** (FALSE) |

So the inherited **cuts** are the cause; the per-node separator *skip* is
harmless. `cuts_only` keeps a valid bound (−4.37) but the wrong incumbent; `both`
adds enough tightening that the false fathoms let the tree close falsely.

### 1c. Why the pool row is invalid on the sub-box — column remapping

The pool for nvs22 (10 rows) is separated on the general-spatial cold path
(`solver.py:5217` region) and is dominated by the **`convex` + `univariate_square`**
families (measured by per-family separation-timer delta during the root capture).
Both are nominally *box-independent* (a square tangent `s ≥ 2x₀x − x₀²`
under-estimates `x²` everywhere; a convex supporting hyperplane is global). So the
invalidity is **not** cut invalidity — it is **column remapping**. The pool row is
stated over the root's lifted column layout (`n_total` columns); the per-node
incremental `assemble` / lifted-FBBT rebuild (`mccormick_lp.py:842`,
`_try_incremental_node`:`545`) can produce the *same column count* with *different
column semantics* (a re-lifted / pinned aux column), so the row addresses the
wrong lifted variables and cuts a feasible point. The append guard
(`mccormick_lp.py:537`, `:927`) checks only `sparse_cols == n_total` (count), not
column identity — insufficient. The source fix is filed as **#567** (#396 backlog).

## Part 2 — the fix (general, C-42-consistent)

**The inherited pool is an accelerator, never a dependency — extended to the
Farkas-infeasible branch, on both the fast and cold paths.** In `solve_at_node`
(now a thin wrapper over `_solve_at_node_impl`, `mccormick_lp.py`): when a
non-empty pool was on offer and the pool-augmented solve returns `infeasible`,
**re-verify pool-free** before trusting the fathom:

1. Cheap first — solve the **base** McCormick relaxation (no separators, no pool,
   `separate=False`), the loosest valid outer approximation. If it is *also*
   infeasible, the node's subtree is genuinely empty (a valid relaxation with an
   empty feasible set is a rigorous fathom) — keep the `infeasible` verdict, one
   loose LP. This is the hot path for a *sound* pool.
2. Only if the base relaxation is **feasible** was the fathom pool-induced — pay
   the full pool-free re-solve (byte-identical to the default path's node solve)
   and adopt that valid result (recover the node on its sound bound). Counted as
   `pool/dropped_nodes` (same role as C-42's).

Soundness: re-solving pool-free only *loosens* the relaxation, so the guard can
never introduce a false fathom of its own; it only forgoes a possible (and
possibly-invalid) prune. C-42 (#553) covered the *no-certified-verdict* branch of
the cold path; C-43 covers the *Farkas-infeasible* branch across the incremental
fast path AND the cold path — the branch C-42's cold-path-only, verdict-gated
retry did not reach.

### 2a. The bug behind the bug — `len()` on a sparse pool

The pool `A_rows` is a scipy CSR matrix, whose `len()` is *ambiguous and raises*.
A naive `len(A_rows) > 0` row check throws on every pool node; the driver's
node-solve `try/except` swallows it and silently *skips* the node — which happened
to mask the false-optimal behind a crash instead of fixing it (and left
`pool/dropped_nodes` at 0 while behaviour changed — a red flag we chased down).
The fix uses a sparse-safe `_pool_has_rows` (`.shape[0]` via `_sparse_rows`).
Regression-pinned by `test_pool_has_rows_is_sparse_safe`.

### 2b. nvs22 before/after (flag-ON, TL 25 s)

| arm | status | objective | bound | `dropped_nodes` |
|---|---|---:|---:|---:|
| before | optimal | **33.55166** (FALSE) | 33.55166 | 0 |
| **after** | **optimal** | **6.05821994** (= oracle) | 6.0582 | **21** |

Deterministic. The 21 recoveries are the false fathoms the guard catches.

## Part 3 — verification

### 3a. C-42 regression cases retained (flag-ON)

| instance | status | objective | oracle | verdict |
|---|---|---:|---:|---|
| nvs06 (TL 20) | optimal | 1.7703125 | 1.7703125 | **retained** (`dropped=1`) |
| tspn05 (TL 60) | optimal | 191.25521 | 191.25521 | **retained** |
| nvs19 (TL 60) | *feasible* | −1098.4 (= oracle incumbent) | −1098.4 | cert LOST (see 3c) |

nvs06 and tspn05 hold. `test_c42_cut_inherit_coldpath.py`: 3 slow + 4 smoke pass.

### 3b. HiGHS/oracle battery — pool-firing instances, flag-ON (TL 40 s)

Every certified `optimal` cross-checked against `minlplib.solu`; every bound
checked against the oracle with the correct sense (min: bound ≤ opt; max: bound ≥
opt):

| instance | sense | status | obj | bound | oracle | sound? |
|---|---|---|---:|---:|---:|---|
| nvs17 | min | optimal | −1100.4 | −1100.40 | −1100.4 | ✓ |
| nvs19 | min | feasible | −1098.4 | −1100.24 | −1098.4 | ✓ |
| nvs22 | min | **optimal** | **6.0582** | 6.0582 | 6.0582 | ✓ |
| nvs23 | min | feasible | −1124.8 | −1130.41 | −1125.2 | ✓ |
| nvs24 | min | feasible | −1031.8 | −1035.66 | −1033.2 | ✓ |
| knp3-12 | MAX | feasible | 1.1056 | 2.6444 | 1.1056 | ✓ (2.64 ≥ opt) |
| dispatch | min | optimal | 3155.29 | 3155.01 | 3155.29 | ✓ |
| kall_circles_c6a…c8a | min | feasible | — | ≤ opt | — | ✓ |

**0 false-optima, 0 dual-bound-crosses-oracle.** Phase-1 hard gate MET.

Broad held-out (20 instances: st_e31/e05, ex122x, nvs0x, alkyl, st_miqp1/2, gear,
…), flag-ON TL 30 s: all sound, every `optimal` = oracle, and the guard is
**inert** (`dropped=None` on all 20 — it fires only where a pool row is actually
invalid). 0 violations.

### 3c. Why the flip is KILLED — nvs19 (flag-ON, TL 60 s, serial)

| instance | PRE-FIX | POST-FIX | Δ |
|---|---|---|---|
| nvs17 | optimal 17.1 s | optimal 17.2 s | identical |
| **nvs19** | **optimal 48.5 s** | **feasible @ 60 s** | **cert LOST** |
| nvs23 | feasible | feasible | identical |
| nvs24 | feasible | feasible | identical |
| dispatch | optimal 0.7 s | optimal 0.9 s | identical |
| knp3-12 / kall_c6b / kall_c6c | feasible | feasible | identical |

Only **nvs19 regresses**. Pre-fix nvs19 certified −1098.4 (the correct oracle) —
but with `dropped=0`, i.e. by *trusting 5 unsound sub-box fathoms* that happened to
prune only non-improving regions (sound-by-luck). Post-fix refuses them
(`dropped=5`), re-opens those nodes, and the extra exploration pushes the cert past
60 s. This is the **correct** behaviour (the fathoms were unsound), but it is a
*lost certificate* on a documented win instance → the CUT-INHERIT-GRAD default-ON
flip's "certs preserved-or-gained" gate FAILS. Per the graduation kill criterion,
**the flip is killed; the flag stays opt-in.**

A cheaper recovery (return the loose base relaxation instead of the full pool-free
re-solve) was tried and *falsified*: it makes nvs22 fail to certify (bound stuck at
−4.37) and does not recover nvs19 — the tight recovered bound is load-bearing.

### 3d. Default path unchanged — cert-neutrality

The fix is gated on `_pool_has_rows(inherited_cuts)`, which is empty on the
default (force-off) path, so the default path is untouched. `solver_tuning.py`
default is unchanged (force-off). `check_cert_neutrality.py` (41-instance panel,
default): **NEUTRAL — 41/41, `|Δobj|=0` on every row, node counts unchanged**
(tspn05 39→39, st_testgr3 549→549). No rebaseline (the gate does not fire on the
shipped default; the "sanctioned bound-changing rebaseline" clause is moot because
the flip is not made).

## Part 4 — decision

**Phase 1 ships. Phase 2 (default-ON flip) killed.** The nvs22 false certificate
is fixed and nvs22 flag-ON certifies the oracle optimum; the fix is sound across
the HiGHS/oracle battery and the broad held-out slice, and byte-identical on the
default path. But the fix exposes that cut-inheritance's sub-box fathoms are
**broadly unsound** (column remapping, #567) and making them sound costs nvs19 its
certificate — a Phase-2 gate failure. The flag stays opt-in until #567 fixes the
invalidity at the source (which would retire the runtime re-verify, restore
nvs19's cert, and re-open the graduation).

## Follow-ups

1. **#567 — column-identity-safe pool inheritance.** Tag pool rows with the lifted
   variables they reference and re-map/drop per node when the layout changes (or,
   conservatively, refuse to append the pool unless the node's lifted layout is
   provably identical to the root's). Removes the false fathoms → the C-43
   re-verify goes inert → nvs19 re-certifies → re-attempt the flip. #396 backlog.

## Gates

| gate | result |
|---|---|
| nvs22 flag-ON (was xfail) | **optimal 6.0582** (= oracle); regression test flipped xfail→passing |
| `pytest -m smoke` (python/tests) | **633 passed, 12 skipped** (incl. 2 new C-43 smoke tests) |
| `pytest -m slow test_adversarial_recent_fixes.py` | **10 passed** |
| `pytest -m slow test_c42_cut_inherit_coldpath.py` | **3 passed** (nvs06, tspn05, nvs22 sound) |
| HiGHS/oracle battery (pool-firing + broad held-out) | **0 false-optima, 0 bound-crosses-oracle** |
| `check_cert_neutrality.py` (default force-off, 41-panel) | **NEUTRAL**, `|Δobj|=0`, node counts unchanged |
| `ruff check` / `ruff format --check` (v0.14.6) | clean |
| pre-commit `mypy` (v2.1.0, whole package) | **Passed** |
| `cargo test -p discopt-core` | n/a — no Rust touched |
