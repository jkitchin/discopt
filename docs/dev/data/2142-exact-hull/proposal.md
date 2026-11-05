# #2142 Direction 2: a hull lowering that keeps `1/yhat` out of the products

Status: analysis + design + entry experiment (no solver code changed). Scripts, logs and the
pre-registration are in this directory (`README.md`). Performance-plan §114 is the premise;
§111 is the kill this must not repeat.

## 0. Summary

- **Root cause (confirmed on the lowered model, `p0_structure.log`).** The shipped hull
  lowers cstr's five rate-law rows `rate*Q*Q + 2*FA*FB == 0` as
  `(v_r/yhat)(v_Q/yhat)(v_Q/yhat) + 2 (v_FA/yhat)(v_FB/yhat)` times `yhat`. The lifted
  spatial spec the kernel is handed carries |global bound| up to **1e27** (`arm_hull_1.log`),
  above the LP's 1e20 sentinel. The model's own variables are all within [-10, 10].
- **Design chosen: the lifted hull (LH).** Factor every nonconvex nonlinear disjunct row into
  atoms (products, squares, univariates) *before* hulling; hull the now-linear row exactly
  (Balas); define each atom's disaggregated copy by a *homogenized* equality
  `w*y = p*q` (no division anywhere); add the perspective of each atom's envelope on the
  disjunct's box as static linear rows (the exact hull of the per-disjunct McCormick
  relaxation, Lee & Grossmann 2001 / Ruiz & Grossmann 2010). Exact at `y in {0,1}`; a
  relaxation of the disjunction at fractional `y`; every column bounded by its atom's range
  on the disjunct's box. Proof sketch in §3.
- **Entry experiment (cstr, deterministic, `PYTHONHASHSEED=0`; §5 below).** The spec's
  largest bound drops from 1e27 to 1e3 for every candidate; all three candidates move the
  10k-node bound off 0.094. Numbers and the kill-criterion verdict are in §5.
- **Implementation plan** (§6): `DISCOPT_HULL_EXACT_PERSPECTIVE`, default OFF, in
  `gdp_reformulate.py` next to the #1917 origin-free hook, polynomial rows first, convex
  rows keep the eps-perspective; tests and the §5 panel plan.

## 1. What the shipped hull does, and why it is vacuous on cstr

`_reformulate_disjunction_hull` (python/discopt/_relax/gdp_reformulate.py:2831) lowers a
nonlinear disjunct row `g(x) <sense> rhs` by the Furman–Sawaya–Grossmann eps-perspective
(`Furman2020`): `yhat g(v/yhat) - eps g(0)(1-y) <sense> rhs*yhat`, `yhat = (1-eps) y + eps`,
`eps = 1e-8`. For a *convex* `g` this is the closed convex hull of the disjunct (up to eps),
which is what the formula is for. cstr's five nonlinear rows are nonconvex **equalities**
(degree-3 polynomials, `rate*Q*Q + 2*FA*FB = 0`); the detector proves none convex (§111). The
relaxation stack then lifts every `v/yhat` as a column of magnitude `10/1e-8 = 1e9`, every
pairwise product to 1e18 and the triple product to 1e27, and relaxes each by McCormick on
those boxes. Measured consequences (§114): node LP bound <= 0.0941 everywhere, 6,089
numerical LP verdicts per 1,000 nodes, bound 0.0940 at 10k nodes.

The perspective is the right *object*; the division is the wrong *representation* of it for
a nonconvex `g`. Three candidate representations keep `1/yhat` out of every product.

## 2. Candidates

Notation: disjunct `k` with selector `y` (binary), the disjunct's own box `B_k`, the hull's
disaggregated copies `v_i` of each variable (bound links `min(L,0) y <= v_i <= max(U,0) y`,
so `v = 0` at `y = 0`), and the ratio `t = v / y` for `y > 0` (`t in B_k` by the links).

### (a) Homogenized perspective (`homog`)

For a polynomial row `sum_m c_m prod_{i in m} x_i <sense> rhs` of degree `d`:

    sum_m c_m prod_{i in m} v_i * y^(d - |m|)  <sense>  rhs * y^d .

No division, no aux. For `y > 0` it is `y^d g(t) <sense> y^d rhs`, i.e. `g(t) <sense> rhs`:
the perspective's solution set exactly. At `y = 0` it reads `0 <sense> 0`. For cstr:
`v_r v_Q v_Q + 2 v_FA v_FB y = 0`. The relaxation stack factors it on the copies' global
boxes (square, two BLF products, one BLF with `y`): every kernel term kind already exists.
Weakness: the LP relaxation is plain McCormick on the copies' *global* boxes; it is strictly
weaker than the hull of the per-disjunct McCormick (shown in §3.4). Applies to polynomial
rows only (a non-polynomial atom cannot be homogenized).

### (b) Lift-then-hull with homogenized atom definitions (`lifted`, the chosen design)

1. **Lift.** Factor the row into atoms `z_j = phi_j(children)` on `B_k` (products, squares;
   univariates in the follow-up) so the row is linear: `a^T x + b^T z <sense> rhs`.
2. **Hull the linear row** on the copies: `a^T v + b^T w <sense> rhs * y`, where `w_j` is the
   disaggregated copy of `z_j` (`w_j = y z_j(t)`), boxed to `[min(Z_j,0), max(Z_j,0)]` with
   `Z_j` the atom's interval on `B_k`, plus the links `Z_j^lo y <= w_j <= Z_j^hi y`.
3. **Define each atom exactly, homogenized.** Product `z = p q` (with copies `w_p`, `w_q`;
   a variable is its own copy): `w_z * y = w_p * w_q`. Square: `w_z * y = w_p^2`. Both sides
   are degree 2 in `(w, v, y)`; no division. Magnitudes: the operands' ranges on `B_k`.
4. **Hull the envelopes.** Every envelope row of the atom on `B_k` (McCormick's four rows,
   the square's secant and tangents, a univariate's secant/tangents) is linear in
   `(z, children)` with a constant; its perspective — the row with the constant scaled by
   `y` — is added as a static linear row: `w_z >= L_q w_p + L_p w_q - L_p L_q y`, etc.

The row set of step 4 is the Balas (`Balas1985`, `Balas1998`) hull of the union of the
per-disjunct polyhedra `P_k = {x in B_k, linear rows, envelope rows}`: the convex hull of the
Lee–Grossmann / Ruiz–Grossmann "relax each disjunct, then hull" relaxation
(`Lee2001`, `Ruiz2010`), which is the tightest polyhedral relaxation obtainable from
per-disjunct McCormick. Steps 2–3 make the lowering exact at integer points, which that
relaxation alone is not. On cstr: 15 auxes and 60 hull-envelope rows (`arm_lifted_1.log`).

### (b') The shipped eps-free lifted perspective applied to every nonlinear row (`ratio`)

`_origin_free_hull_row` (#1917) already emits `y * g(t) <sense> y * rhs` with `v - y t = 0`,
`t` boxed to `B_k`, for rows whose `g(0)` is not finite. Applying it to every nonlinear row
is a three-line change. It is exact and bounded, but the McCormick relaxation of `y * s`
with `s = g(t)` is `s in [S^lo (1-y), S^hi (1-y)]`: big-M on the ratio copy's residual, so
at fractional `y` the row is no tighter than big-M. Measured in §5 as a control.

### (c) Conic / exact perspective for convex rows

`_conic_hull_body` (`DISCOPT_HULL_CONIC_QUADRATIC`, default ON) already lowers a separable
convex quadratic inequality as a smoothed cone with no division. cstr's nonlinear rows are
nonconvex equalities, so (c) does not apply to the rows that pin cstr. In the general design
(§6) a row the detector proves convex keeps the eps-perspective / cone — the exact convex
hull — and only the nonconvex rows go to (b). Nothing in §111's kill applies: §111 sent the
nonconvex rows to big-M; (b) sends them to the hull of their McCormick relaxation.

### (d) Literature not already in the catalog

- `Ceria1999` (Ceria & Soares): the lifted formulation `x = sum y_k t_k`, `t_k in S_k` with
  `v_k = y_k t_k` — the ratio form (b'); for convex `S_k` its projection is the hull.
- `Stubbs1999` (Stubbs & Mehrotra): the perspective hull for 0-1 convex programs, the
  ancestor of the eps-perspective. Both are exact only for convex disjuncts.
- `Lee2001`, `Ruiz2010`: for nonconvex GDP, replace each disjunct by a convex (McCormick)
  relaxation and take the hull of the union; `Ruiz2010` adds basic steps to strengthen it on
  bilinear GDPs. (b) is this relaxation plus exact homogenized definitions.
- `Vielma2019` (Cayley embedding): small/strong formulations of unions of convex sets that
  avoid the perspective's division — the conic route of (c), convex-only.
- `Kronqvist2021`, `Kronqvist2025` (P-split): intermediate formulations between big-M and
  hull by splitting a convex row into partitions; convex-only, and the partition is on the
  row, not on the atoms. Not a route for cstr's nonconvex equalities.
- `Trespalacios2015` (improved big-M), `Sawaya2005`, `Trespalacios2016` (cutting planes for
  convex GDP), `Grossmann2013` (review): no representation of a nonconvex disjunct row other
  than big-M or relax-then-hull.
- `Furman2020`: the shipped eps-perspective; its own scope is convex disjuncts.

None gives a nonconvex-row hull beyond "relax, then hull". (b) is that construction made
exact by the homogenized atom definitions, which is the part the literature leaves to the
global solver's spatial branching.

## 3. Soundness of (b)

Let `S_k = {x in B_k : rows_k(x)}` be disjunct `k`'s set, `Z_k = {(x, z): x in S_k, z =
atoms(x)}` its lift, `P_k = {(x, z): x in B_k, linear rows, envelope rows}` its McCormick
polyhedron, `Z_k subset P_k`. The lowered model has rows (i) `x = sum_k v_k`, `sum y_k = 1`,
links; (ii) hulled linear rows; (iii) homogenized atom equalities; (iv) hulled envelopes.

**3.1 Exactness at integer `y`.** If `y_k = 1`: the links pin every other disjunct's copies
and auxes to 0, so `x = v_k`; (iii) reads `w_z = w_p w_q` recursively, so `w = atoms(x)`;
(ii) is the original row; (iv) holds because the envelopes are valid at the true atom
values. If `y_k = 0`: `v_k = 0`, `w_k = 0` (links), and every row of (ii)–(iv) reads
`0 <sense> 0` because every constant is scaled by `y`. Equalities are two inequalities with
the same argument. So the integer points of the lowered model project exactly onto the
original feasible set (no feasible point cut, no infeasible point admitted). An empty
disjunct forces nothing: its rows are `0 <sense> 0` whenever it is off.

**3.2 Relaxation at fractional `y`.** Take `(x^k, z^k) in Z_k` and `lambda in Delta`. Set
`y_k = lambda_k`, `v_k = lambda_k x^k`, `w_k = lambda_k z^k`. (i), (ii) and (iv) hold because
they are linear with constants scaled by `y` (Balas). (iii) holds:
`w_z y = lambda^2 z = lambda x_p lambda x_q = w_p w_q`. Hence every convex combination with
one point per disjunct is feasible, the lowered set contains the original, and any valid
relaxation of the lowered rows bounds the original. The LP relaxation of the lowered model
contains the rows (iv), so it is at least as tight as the hull of `union_k P_k`.

**3.3 Magnitudes.** Every column is a copy (box within `B_k` and 0), an atom copy (box
within the atom's interval on `B_k` and 0) or `y in [0,1]`; every product is of two such
columns. On cstr the lifted spec's largest bound is 1e3 (the range of `rate*Q*Q`).

**3.4 Why (b) is tighter than (a) at fractional `y`.** On cstr's `FA*FB` at the root
(`v in [0,10]`, `y in [0,1]`): (b)'s rows give `w <= 10 v_FA`, `w >= 10 v_FA + 10 v_FB -
100 y`. (a) lifts `pi = v_FA v_FB` with McCormick on `[0,10]^2` and `pi' = pi y`: the best
it implies is `pi' <= 10 v_FA + 100 (1 - y)` and `pi' >= 10 v_FA + 10 v_FB - 100`, both
weaker by `100 (1-y)`. Every (a) row is implied by (b)'s, so (b) dominates (a) at the LP.

**3.5 What is not claimed.** (iv) is built on `B_k` at lowering time; the kernel's per-node
envelopes of (iii) use the copies' node boxes (which contain 0), so after branching on a
copy, (iii)'s McCormick tightens while (iv) does not. A perspective-aware kernel term
(envelope of `w_z y = w_p w_q` on the box of `t = v/y`) would tighten (iv) per node; that is a
follow-up, not part of this design.

## 4. What the relaxation stack represents today

Kernel term kinds (`spatial_producer.py`): BLF products `w = A*B` of affine forms (McCormick,
regenerated per box), kind 1 monomial powers, kind 2 affine squares, kind 3 sqrt, kind 4
negative affine powers. (a) and (b) emit only products of two columns, squares and linear
rows, so the native kernel takes them without change (`arm_*_1.log`: the kernel ran on all
three). The eps-perspective needs kind 4 (`1/yhat`) with a 1e8 range inside every product.

## 5. Entry experiment

Pre-registered in `PREREG.md` before any run past 1 node. cstr, deterministic,
`PYTHONHASHSEED=0`, `DISCOPT_PROFILE=1`, 4 arms concurrent on 4 cores (walls recorded, not
compared). Driver `arm.py`; prototype `exact_hull_proto.py` (monkeypatch of the hull's
origin-free hook; the rest of the hull is the shipped code).

RESULTS_TABLE

Exactness checks (`check.py`, `check.log`): CHECK_SUMMARY

VERDICT

## 6. Implementation plan

Flag `DISCOPT_HULL_EXACT_PERSPECTIVE` (default OFF; `solver_tuning.py` field
`hull_exact_perspective`, `_env_flag(..., default=False)`, read through
`_tuning()` like `hull_conic_quadratic`). Row added to `docs/dev/flag-retirement-audit.md`
in the same PR.

Files and functions (`python/discopt/_relax/gdp_reformulate.py`):

1. `_polynomial_monomials(body) -> dict[tuple[str, ...], float] | None` — the prototype's
   `monomials()`: scalar variables, constants, `+ - *`, `neg`, `SumOver`; `None` on anything
   else (division, `**`, function calls, array variables, a `Parameter` — its coefficients
   must stay live, #1862). Pure function, unit-tested.
2. `_lifted_hull_row(con, y_k, owned, bounds_k, cache, model, stem, new_vars, new_cons)
   -> tuple[Expression, Expression]` — the prototype's `lifted` arm: per monomial of degree
   >= 2 a left-to-right chain of atom copies (cached per disjunct and monomial prefix so
   `Q*Q` is shared by every row of the disjunct), each with its homogenized equality, its
   links and its hulled envelope rows (`_hull_envelope_rows`: McCormick for two distinct
   columns; secant plus three tangents for a square). Returns the linear body and
   `rhs * y_k`, in the row's own sense (an equality becomes `_le`/`_ge` as today).
3. In `_reformulate_disjunction_hull`, nonlinear branch (line ~3190): when the flag is on,
   the row is not proven convex (`classify_constraint`, as §111's probe used; a convex row
   keeps the eps-perspective / cone, which is its exact hull) and `_polynomial_monomials`
   succeeds, call `_lifted_hull_row`; else fall through to today's code unchanged. The
   origin-free (#1917) path is untouched. With the flag off the lowering is byte-identical.
4. Nesting: the new rows are homogeneous of degree 1 in the disjunct's locals for fixed
   `y` only through `w`; a parent hulling this disjunction as a child re-hulls them like any
   other local rows (the atom copies are locals, zero when `y_k` is, which is what the nested
   hull needs). Covered by the existing nesting tests under the flag.
5. Counters: `_HULL_EXACT_STATS` (rows lifted, atoms, envelope rows), printed by the solve
   inventory, so a run can prove the branch fired (CLAUDE.md §6).

Tests (`python/tests/test_2142_hull_exact_perspective.py`):

- flag OFF: `reformulate_gdp(cstr, "hull")` row/variable counts and a 200-node
  deterministic `node_count`/bound bit-identical to no flag;
- `gdp_differential.run_shape` battery (`python/tests/gdp_differential.py`) on three new
  shapes with bilinear and cubic disjunct rows (an equality, an inequality, a shared
  product across rows): no feasible point cut, no infeasible point admitted — the §5
  feasible-point sampling;
- fixed-selector exactness: each configuration of the shapes solved under the flag, big-M
  and the shipped hull gives the same optimum; the cstr optimum 3.0620073 is reproduced;
- fixed-box differential: on 200 random sub-boxes of the shapes, the flag-ON LP bound is
  <= the enumerated true box optimum (soundness) and the comparison with the flag-OFF bound
  is recorded — ON >= OFF is *not* asserted, because the two are different formulations,
  not a strengthening; on cstr-like boxes ON is expected far above OFF;
- convex-combination containment: convex combinations of lowered feasible points of
  different configurations satisfy every linear lowered row;
- magnitude: the lifted spec's largest |global bound| on cstr under the flag is <= 1e4;
- cstr regression: deterministic 2,000 nodes, flag ON bound > flag OFF bound, <= oracle.

Panel (§5, both bars): flag ON vs OFF, `--assert-unfired-identical`, over the in-repo
corpus (74, `max_nodes=2000`) and the GDP parity set (24 x hull, `max_nodes=5000`), plus
the `requires-hull` GDPlib instances whose disjuncts carry polynomial nonconvex rows (the
fire screen lists them). Cert-clean: `incorrect_count = 0`, no bound above its reference,
no certification lost, incumbents feasibility-verified. Net-positive: a certification gained
or a tighter node-limited bound where it fires, and bit-identical trees where it does not.
Graduation flips the default; failure retires the flag and the implementation in one PR.

## 7. Risks

- **Model growth.** On cstr the lowered model goes from 236 to 251 columns and 597 to ~690
  rows; the mixed hull of §111 showed that a larger model can cost more nodes than the
  tighter rows gain. The entry numbers in §5 measure this directly.
- **Static envelopes (3.5).** The hulled envelope rows do not tighten with the node box;
  on instances whose gap closes only by spatial branching on the copies, the kernel's own
  McCormick of the homogenized rows carries the tightening, and it is weaker than a
  perspective-aware term would be.
- **Scope.** Polynomial rows only in the first increment; a disjunct row with `exp`, `log`,
  a division or a fractional power keeps the eps-perspective and its magnitudes. The
  univariate extension (atom copy `w = y f(t)` with the envelope rows of `f` on `B_k`
  scaled by `y`, and the ratio link `v = y t` for the argument only) is the same
  construction; it is listed as the follow-up because cstr does not exercise it.
- **Rounding at `y = 0`.** Every row must be exactly 0 at the off disjunct (the #1043
  class). The lifted rows have no additive constant, so they are exactly 0 in floating
  point when every column is 0; the test asserts it.
- **Interaction with `bound_pretransformation`.** big-M gets the #1617 bound rows; the hull
  does not. If the entry numbers show (b) below big-M at equal nodes, the next lever is to
  give the hull those rows, not to change the perspective again.
