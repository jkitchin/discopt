# #1154 — `SumOverExpression` in the GDP walkers: the §5 differential panel (2026-09-04)

**Status: GRADUATED DEFAULT-ON.** `DISCOPT_GDP_SUMOVER` /
`SolverTuning.gdp_sumover`, default `True`; `=0` is a live opt-out that restores
the pre-#1154 walkers byte for byte. The panel below meets both §5 bars —
cert-clean (`invalid_bounds = 0` over 648 solves on the firing class, 0 bound and
0 primal violations over 52 oracle-scored corpus instances) and net-positive
(three loud refusals become three certified optima; hull certifies 46/54 against
29/54, sd 0.00 over 3 interleaved reps).

**The one judgment call, stated plainly.** §5's net-positive bar is written for a
mechanism that fires across the `.nl` corpus. This one *cannot* fire there at
all: `SumOverExpression` is created only by the Python modeling API's `dm.sum`,
and the corpus contains zero of them (measured, §2a). So "net-positive" is
measured on the class where the mechanism does fire, and the corpus arm is used
for what it *can* establish — that nothing existing moves. If the owner reads §5
as requiring corpus-measured benefit, the flip back is one word in
`solver_tuning.py` (`default=True` → `default=False`); nothing else in the change
depends on the default.

> **Method.** Linux x86-64, Python 3.12, release build (`make build`;
> `python/discopt/_rust.abi3.so`, `maturin develop --release`). Corpus: the 66
> vendored `.nl` instances in `python/tests/data/minlplib_nl/`. Oracle:
> `discopt_benchmarks.utils.reference_optima` (resolves through
> `known_optima.toml` / `cert-optima.json`; no `.solu` snapshot is installed on
> this host, so 14 of 66 instances are reported **unscored** rather than
> silently skipped). Probes in `scratchpad/issue1154/`; every one prints an
> executed-assertion count and exits non-zero at zero (CLAUDE.md §6), and none
> swallows an exception (§7). Load gate (§9): `uptime` 0.04–0.12 at the start of
> every arm; the one round run under self-inflicted contention was discarded and
> re-run.

---

## 0. What the change is

Six walkers in `python/discopt/_relax/gdp_reformulate.py` had no case for
`SumOverExpression` — the n-ary node `dm.sum(f(i) for i in S)` builds, holding
its already-expanded term list in `.terms`:

| walker | on `main` | consequence |
|---|---|---|
| `_collect_variables` | returns `{}` | hull disaggregates nothing |
| `_is_linear` | `False` | hull takes the perspective route |
| `_bound_expression` | `(-inf, +inf)` fallback | **big-M refuses**: no finite `M` |
| `_body_at_zero` | raises | **hull refuses**: no `g(0)` |
| `_substitute_vars` | passes the node through | substitution silently does not happen |
| `_hull_linear_substitute` | falls through | never reached (`_is_linear` is `False`) |

The fix gives each one a case, under a single rule: **`Σ[t1, …, tn]` is handled
exactly as the left-folded chain `t1 + … + tn`** — the desugaring the modeling
layer could equally have produced. Nothing about the reformulation mathematics
is new; only the node type is newly recognised. That equivalence is the whole
contract and is machine-checked per walker in
`python/tests/test_1154_gdp_sumover_hull.py`.

One piece is **not** behind the flag, because it is a refusal and not a
capability: `_assert_hull_saw_every_variable`, an independent-walker cross-check
run unconditionally in `_reformulate_disjunction_hull`. See §4.

---

## 1. Entry experiment — the three refusals on `main` (§4)

`scratchpad/issue1154/repro.py`, the model from the issue:

```python
x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(3)]
m.either_or([[dm.sum(x[i] - 1 for i in range(3)) <= 0.0], [x[0] >= 8.0]])
m.minimize(-(x[0] + x[1] + x[2]))          # true optimum -30.0
```

| sources | marker (`SumOverExpression` count in `gdp_reformulate.py`) | `auto` | `big-m` | `hull` |
|---|---|---|---|---|
| `main` (1e75919) | 2 | refuses (no finite big-M) | refuses (no finite big-M) | refuses (`g(0)`) |
| HEAD, flag OFF | 11 | refuses (no finite big-M) | refuses (no finite big-M) | refuses (**coverage**) |
| HEAD, flag ON | 11 | `optimal -30.0 / -30.0` | `optimal -30.0 / -30.0` | `optimal -30.0 / -30.0` |

Three loud refusals become three certified optima. The OFF arm still refuses on
every route; only the hull refusal moves, from `HullPerspectiveOriginError`
(raised later, at `g(0)`) to the new, earlier and more specific
`HullVariableCoverageError`, which names `x1` and `x2` — precisely the variables
that would otherwise have been emitted un-disaggregated.

**Fail-before / pass-after**, run in a `git worktree` at `origin/main` with §8
marker counts confirming which sources each arm loaded (`SumOverExpression`
count 2 vs 11; `HullVariableCoverageError` count 0 vs present):

| sources | `test_1154_gdp_sumover_hull.py` |
|---|---|
| `origin/main` (import of the new error shimmed so the assertions actually run) | **17 failed, 4 passed** |
| HEAD | **21 passed** |

The 4 that pass on `main` are the three `test_flag_off_still_refuses_loudly`
parametrizations and the NaN guard — i.e. exactly the tests that assert the OFF
arm *is* `main`.

---

## 2. Corpus arm — the flag is structurally inert, and the corpus is cert-clean

### 2a. Inertness, measured rather than argued (`panel_inertness.py`)

`SumOverExpression` is built only by the Python modeling API's `dm.sum`; the
`.nl` reader never emits one. Measured over all 66 vendored instances, walking
every constraint body and the objective:

```
instances_loaded=66
nodes_walked=33376
sumover_nodes=0
load_failures=0
```

Zero. The flag's read sites are therefore *unreachable* on this corpus — the
corpus arm can measure **no risk to existing work**, and cannot measure benefit.

### 2b. A/B differential anyway (`panel_corpus_diff.py`)

Every instance solved twice, interleaved, `deterministic=True`, `time_limit=10`,
comparing `status` / `objective` / `bound` / `node_count` exactly (§5's
bound-neutral regime):

```
instances_compared=66
mismatches=3          (syn05hfsg, tanksize, tls2)
errored=0
```

63/66 byte-identical.

### 2c. The three mismatches are wall-truncation, not the flag (`panel_corpus_control.py`)

§2a says a flag effect is impossible here, but that is an argument, not a
measurement. The control repeats the **same** arm against itself, 3 reps each,
on exactly those three instances, on a quiet machine (load 0.04):

```
syn05hfsg/arm0: 3 distinct outcome(s) in 3 reps
syn05hfsg/arm1: 3 distinct outcome(s) in 3 reps
tanksize/arm0:  2 distinct outcome(s) in 3 reps
tanksize/arm1:  3 distinct outcome(s) in 3 reps
tls2/arm0:      2 distinct outcome(s) in 3 reps
tls2/arm1:      1 distinct outcome(s) in 3 reps
arms_whose_own_repeats_disagree=5/6
```

The within-arm spread *contains* every between-arm difference the differential
reported. `tls2` with the flag **OFF** produced both `feasible / 2.099999990166769 /
229 nodes` and `time_limit / 2.003761171751075 / 205 nodes` — which is exactly the
pair the A/B flagged as a mismatch. `deterministic=True` neutralizes role-2
budgets but leaves the user's role-1 `time_limit` and the phase-entry gates
wall-dependent (#1116), so an instance cut by the wall is cut at a
machine-speed-dependent point. This is the control PR #1150 established when
`beuster` produced both outcomes in both arms.

### 2d. Cert-clean scoring against the oracle (`panel_corpus_cert.py`)

An A/B says the flag changes nothing; it does not say what it leaves unchanged is
sound. The ON arm scored against the reference optimum, sense taken from the
loaded model (5 vendored instances are MAXIMIZE and assuming min manufactures
false violations):

```
instances_scored=52
instances_unscored=14   ['4stufen','bchoco06','bchoco07','bchoco08','beuster',
                         'casctanks','contvar','heatexch_gen1','heatexch_gen2',
                         'heatexch_gen3','tspn05','tspn08','tspn10','tspn12']
instances_with_unproven_oracle=0
bound_violations=0
primal_violations=0
```

`incorrect_count = 0`. No dual bound above (below, for a MAXIMIZE instance) its
proven oracle; no incumbent better than its oracle. The 14 unscored instances
have no oracle in any in-repo source on this host and are reported as unscored,
not counted as passes.

---

## 3. Capability arm — the class where the mechanism can fire

The corpus cannot measure benefit (§2a), so the net-positive arm is a generated
family of Python-API GDP models with a `dm.sum` disjunct body: **108 cases** over
`n_terms ∈ {2,3,5}` × `n_disjuncts ∈ {2,3}` × sense `{<=, >=, ==}` × three
coefficient patterns × `{affine, exp}` bodies, each solved on all three GDP
routes in **two arms** — the body written as `Σ[...]` and the identical body
written as the folded chain `t1 + … + tn`. 648 solves.

### 3a. The node reaches the relaxer (`probe_node_survives.py`)

Comparing arms only says something about the downstream layers if the node
*survives* the GDP reformulation. It does: the reformulated model carries 1
surviving `SumOverExpression` on both `big-m` and `hull`. So the relaxer, the
evaluator and the B&B see the n-ary node, not a rebuilt chain.

### 3b. Retraction of the v1 scoring (CLAUDE.md §11)

**`panel_capability.py` (v1) reported 1 bound violation, 2 certification
regressions and 5 objective mismatches. All eight are withdrawn.** v1 scored the
Σ arm against *the chain arm's own result on the same route* — a solver output
used as an oracle. On the `hull` route with an `exp` body that oracle is not
trustworthy: the chain arm routinely stops at `feasible` short of the optimum
`auto` and `big-m` both certify, so v1 flagged the Σ arm for being **closer to
the truth**. Reading the same log against the `auto`/`big-m` columns:

| case (hull, nl=1) | `auto`/`big-m` certify | chain hull | Σ hull |
|---|---|---|---|
| n=3 d=2 `<=` | −8.630462164 | `feasible` −8.603386 | `optimal` −8.630462 |
| n=3 d=3 `==` | −8.630462174 | `feasible` −8.603386 | `optimal` −8.630462 |
| n=5 d=2 `==` | −9.116077840 | `feasible` −8.932426 | `feasible` −9.116078 |
| n=5 d=3 `<=` | −9.116077831 | `feasible` −8.932426 | `feasible` −9.116078 |
| n=5 d=3 `==` | −9.116077840 | `feasible` −8.367016 | `feasible` −9.116078 |

v1's single "bound violation" is the same story: on `n=2 d=3 <= c=(0.5,2.0,−1.5)
nl=1`, `auto` and `big-m` both certify **−11.505256737**, the Σ hull arm agrees to
1.8e−8, and it is the *chain* hull arm that reports −11.505563 — see §3d.

### 3c. v2, scored without an oracle (`panel_capability_v2.py`)

v2 replaces the oracle with a gate that cannot have that failure mode. Every
incumbent, in every arm and route, is feasibility-verified in numpy against the
**original** disjunction; the best verified-feasible objective across all six
(route, arm) pairs is a valid upper bound on the true minimum; therefore any
reported dual bound above it is invalid. No solver is assumed correct.

```
cases_scored=108
invalid_bounds=0
infeasible_incumbents=2      (both in the pre-existing hull/CHAIN arm -- §3d)
refusals=0

certification rate (optimal / answered), per route and arm:
  auto   chain   : 108 / 108        auto   sumover : 108 / 108
  big-m  chain   : 108 / 108        big-m  sumover : 108 / 108
  hull   chain   :  81 / 108        hull   sumover :  99 / 108

cases where the arm reaches the best verified-feasible incumbent:
  chain: 199        sumover: 236
```

**`invalid_bounds = 0` over 648 solves** — the §1 bar, and the one that cannot be
traded. `auto` and `big-m` are bit-identical between the arms on all 108 cases;
every difference is on the `hull` route with a nonlinear body, where the Σ arm
certifies more and reaches the better verified incumbent more often. The
plausible mechanism is that the n-ary node presents the factorable
reformulation with one linear aggregation of `n` children instead of `n−1`
nested `BinaryOp`s, each of which can pick up its own auxiliary; that is a
hypothesis, not a measured claim, and nothing here depends on it.

### 3d. The two infeasible incumbents are pre-existing, and within tolerance

Both are in the `hull`/**chain** arm — the `BinaryOp` path `main` already has —
and both reproduce **bit-for-bit in a `git worktree` at `origin/main`** (§8
marker: `_sumover_terms` count 0 there, 7 in the patched tree)
(`probe_hull_chain_infeasible.py`):

| case | reported | residual on the nearest disjunct |
|---|---|---|
| n=2 d=3 `<=` c=(0.5,2.0,−1.5) | `optimal` −11.505563 | 8.29e−05 on rhs 3.0 → **2.8e−05 relative** |
| n=3 d=2 `<=` c=(1.0,) | `feasible` −8.603386 | 1.50e−05 on rhs 4.0 → **3.8e−06 relative** |

Both sit inside the project's declared feasibility tolerance (`rel = 1e-4`,
`conftest.py`); v2's checker used a flat absolute `1e-5`, which is stricter than
the contract. They are the FSG ε-perspective's documented O(ε) residual, not a
new defect, and not attributable to #1154. Note the direction this cuts: the
strict checker *excluded* those two points from the `ref` upper bound, which
makes the `invalid_bounds` gate **stricter**, so `invalid_bounds = 0` holds a
fortiori.

### 3e. Repetition control on the one timing-shaped claim (`panel_hull_cert_reps.py`)

The hull certification rate is measured under a 15 s wall limit, so §9 applies:
interleaved A/B, load gate, spread. Re-running only the subset that carries the
difference — the `hull` route on the 54 nonlinear cases — three times, arms
interleaved *within* each case, load 0.71 at start:

| arm | rep 0 | rep 1 | rep 2 | mean | sd |
|---|---|---|---|---|---|
| chain | 29/54 | 29/54 | 29/54 | 29.00 | **0.00** |
| `Σ[...]` | 46/54 | 46/54 | 46/54 | 46.00 | **0.00** |

`executed_comparisons=324`. +17 certifications out of 54, with zero spread in
either arm across three reps. Not a draw of the noise.

---

## 4. The unconditional part: an independent-walker coverage guard

`_assert_hull_saw_every_variable` is **not** behind the flag, because it is a
refusal and not a capability. `_reformulate_disjunction_hull` keys everything —
the disaggregated variables, the aggregation rows, the bound-linking rows, the
substitution map — on `_collect_variables`. A variable that walker misses is not
merely under-disaggregated: it survives into the reformulated row
**un-substituted**, so the disjunct body is imposed *globally*. That is exactly
what PR #1150 shipped and had to revert:

```
_hull_modes_d0_c0: Σ[3 terms] - (0 * y0) <= 0
```

The guard cross-checks `_collect_variables` against a walker this module does not
own — `discopt.modeling.core._iter_model_leaves`, which already handles
`SumExpression`, `SumOverExpression`, `MatMulExpression` and `CustomCall`. A
cross-check against the *same* walker would be a no-op (§6): if the collector
cannot see a node, neither could the probe.

It costs no model that reformulates today. Every node type that can trip it is
one `main` already refuses on the nonlinear route (`_body_at_zero` raises
`HullPerspectiveOriginError` for the same set), and `_is_linear` is `False` for
all of them, so the linear route was never reachable either. Measured: the 731
GDP / OA / Benders / GBD / MPEC tests are identical with and without it, and the
guard fires on exactly one thing in the whole suite — the #1154 repro with the
flag OFF, where it names `x1` and `x2`.

---

## 5. What this does NOT ship

* **`SumExpression`** (the axis reduction `dm.sum(X, axis=0)`) is still `False`
  in `_is_linear` and still unhandled by the walkers. It is a different node with
  array semantics; the constant-scaling rule in `_hull_linear_substitute` is not
  obviously the same for it. Left conservative on purpose — a `False` here only
  forgoes a capability, and the new §4 guard turns the silent-miss hazard into a
  refusal.
* **`_extract_body_coeffs`** keeps declining the node. That is PR #1150's fix
  (#1039), deliberately not duplicated here; the two changes compose (every caller
  pairs `_is_linear` with the extractor and falls back on the extractor's `None`,
  so a `True` from `_is_linear` cannot over-promise at those sites). The two PRs
  do conflict textually in `_is_linear` — whichever lands second resolves it.
* **The hull FSG ε-residual** (§3d) — pre-existing, within the declared
  `rel = 1e-4` feasibility tolerance, reproduces on `main`, out of scope here.
* **`test_oa.py::TestOAEdgeCases::test_infeasible_model`** fails identically in
  both arms and on a `git worktree` at `origin/main` (`assert None == {}`).
  Pre-existing; PR #1150 documented the same and also left it alone.

---

## 6. Verification

| suite | flag OFF | flag ON |
|---|---|---|
| `pytest -m smoke` | 1119 passed, 20 skipped, 2 xpassed | 1119 passed, 20 skipped, 2 xpassed |
| `pytest -m slow test_adversarial_recent_fixes.py` | 19 passed | 19 passed |
| GDP selection, `-m "slow or not slow"` (11 files) | 311 passed, 12 skipped, 2 xpassed | 311 passed, 12 skipped, 2 xpassed |
| OA / Benders / GBD / MPEC / hull selection (19 files) | 420 passed, 1 failed (pre-existing) | 420 passed, 1 failed (pre-existing) |
| `python/tests/test_1154_gdp_sumover_hull.py` | — | 22 passed (17 failed / 4 passed on `origin/main`) |
| `cargo test -p discopt-core` | not run — no Rust touched (`git diff origin/main...HEAD -- crates/` is empty) | |

The 2 xpassed are pre-existing (PR #1150 reports the same pair).
