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
- **Design chosen: the homogenized perspective.** For a polynomial disjunct row of degree
  `d`, replace `yhat g(v/yhat)` by `sum_m c_m prod_{i in m} v_i * y^(d-|m|) <sense> rhs y^d`:
  the perspective multiplied through by `y^(d-1)`, so no division anywhere and every product
  is between disaggregated copies (box <= the disjunct's) and `y in [0,1]`. Exact at
  `y in {0,1}`; a relaxation of the disjunction at fractional `y`; a special case of the
  lift-then-hull construction whose proof is in §3. A row the detector proves convex keeps
  the eps-perspective (its exact convex hull); only nonconvex rows change.
- **Entry experiment (cstr, deterministic, `PYTHONHASHSEED=0`; §5).** Every candidate
  drops the spec's largest bound from 1e27 to 1e3. The homogenized perspective (`homog`,
  candidate (a)) is the best: 10k-node bound 0.094 -> **1.611**, 20k-node bound **2.189**,
  numerical LP verdicts ~6,000 per 1k nodes -> ~110 per 1k. The lift-then-hull design
  (`lifted`, (b)) is tighter at the LP and *worse* in the tree (0.47 at 10k; 0.11 with both
  row sets), so the chosen design is `homog`, with (b)'s construction retired by measurement.
- **Kill criterion (pre-registered): not met.** `homog` at 20k (2.189) is below big-M's
  2.788, and it does not certify on the default path in 300 s (30k nodes, bound 1.58). The
  lowering fixes the hull's magnitude pathology; it does not make the hull route certify
  cstr. The remaining gap is the hull's larger model per node, not its relaxation.
- **Implementation plan** (§6): `DISCOPT_HULL_EXACT_PERSPECTIVE`, default OFF, in
  `gdp_reformulate.py` next to the #1917 origin-free hook, polynomial rows first, convex
  rows keep the eps-perspective; tests and the §5 panel plan. Recommended as a hull-route
  fix in its own right (the panel measures ON against hull OFF), explicitly *not* as the
  close of #2142.

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

### (b) Lift-then-hull with homogenized atom definitions (`lifted`; retired by §5)

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
hull — and only the nonconvex rows are homogenized. Nothing in §111's kill applies: §111
sent the nonconvex rows to big-M; (a) and (b) keep them in the hull, division-free.

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
global solver's spatial branching; (a) is (b) with the atoms substituted back (§3.4a).

## 3. Soundness of (b), and of (a) as its special case

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

**3.4a (a) is (b) without the auxes.** Substituting `w_z y = w_p w_q` recursively into (b)'s
linear row and multiplying through by `y^(d-1)` gives exactly (a)'s row, so (a)'s lowered set
is the projection of (b)'s minus the envelope rows (iv): 3.1 and 3.2 hold for (a) verbatim.
At the LP, (a) is weaker (3.4); in the tree, (a) is better (§5).

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

| arm (cstr, deterministic, `PYTHONHASHSEED=0`, dual proofs ON unless noted) | 1 node | 10k nodes | 20k nodes | spec max \|bound\| | `LpVerdictNumerical` | node statuses (optimal / infeasible / numerical) |
|---|---|---|---|---|---|---|
| `hull` (shipped eps-perspective) | 0.0 | 0.0940 (§111/§114, 855-1022 s) | not run | **1e27** | 6,089 per 1k nodes (§114 d1) | 753 / 20 / 167 per 1k (§114) |
| `bigm` (shipped default) | -1e-12 | **2.2486** (51.5 s) | **2.7875** (84.3 s; = §112 seed 0) | 1e3 | 176 @10k, 216 @20k | 5,570 / 3,362 / 0 @10k |
| `homog` (candidate (a)) | -1e-12 | **1.6108** (183 s) | **2.1886** (405 s) | 1e3 | 1,122 @10k, 4,920 @20k | 5,640 / 3,395 / 0 @10k |
| `homog`, dual proofs OFF | | | 0.7481 (231 s) | 1e3 | 1,441 | 10,542 / 7,741 / 17 @20k |
| `lifted` (candidate (b)) | -1e-12 | 0.4705 (305 s) | not run (below `homog`) | 1e3 | 2,772 | 5,550 / 1,107 / 37 |
| `both` (`lifted` rows + `homog` row) | -1e-12 | 0.1056 (303 s) | not run | 1e3 | 1,558 | 6,544 / 1,945 / 4 |
| `ratio` (shipped #1917 hook on every row) | -1e-12 | 0.2574 (550 s) | not run | 1.2e3 | 12,064 | 6,090 / 1,046 / 11 |
| `homog`, default path, `time_limit=300` (criterion 2; load 3) | | 30,246 nodes: bound 1.5799, incumbent 3.1302, **not certified** | | 1e3 | 838 | 18,631 / 9,431 / 0 |

Reference for the dual-proof-OFF row: big-M OFF at 20k is 1.3432 (§108, §111). Walls: up to 4
arms concurrent on 4 cores (§9: node counts are the comparison, not walls); `bigm` and `homog`
10k ran side by side, so their walls are comparable to each other only roughly (`homog` is
~3.5x slower per node on a model of 285 columns against 105).

**Falsified expectation, recorded (§4 house style).** `lifted` dominates `homog` at the LP
(§3.4, by construction) and reaches a *lower* tree bound at 10k nodes (0.47 against 1.61);
adding the lifted rows to `homog` (`both`) drops it to 0.11. The static hull-envelope rows and
the atom auxes do not pay in the tree: both arms fathom far fewer nodes as infeasible
(1,107 and 1,945 against 3,395) and their bound stays low. The per-node McCormick of the
homogenized row on the branched copies is what moves the bound, and the extra product terms
`w*y = p*q` dilute it (the kernel's envelope of a product with `y in [0,1]` is loose until `y`
is fixed). The chosen design is therefore the simplest one, `homog`; the lifted construction
is retired by this measurement (its exactness proof in §3 stands and is reused for `homog`,
which is its special case without auxes).

Exactness checks (`check.py`, `check.log`; 63 executed assertions, both `homog` and
`lifted`):

- the big-M certified optimum x* (29,551 nodes, obj 3.0620146), mapped to the lowered
  columns, violates no lowered row by more than **2.2e-16**;
- four fixed selector configurations ((5,5) = x*'s, (1,1), (2,1), (3,3)): the candidate's
  incumbent equals big-M's to 1e-14 in every case (3.0620146, 9.8947368, 4.0618817,
  3.3148454), and each candidate solution projects onto the original rows plus the selected
  disjuncts' rows with violation <= **8.9e-16**. (The fixed configurations are nonconvex
  NLPs that no arm certifies in the 120 s / 2,000-node budget, so this compares incumbents,
  not certified optima; the (5,5) incumbent is the oracle.)
- 18 convex combinations of the four lowered points satisfy every linear lowered row (575
  rows under `homog`, which adds none to today's 597; 675 under `lifted`).

`homog` leaves the lowered model at today's 236 columns and 597 rows; only the five
nonlinear rows' expressions change.

**Verdict against the pre-registered kill criterion: killed as the lever for #2142's
"hull certifies cstr".** Criterion 1 fails: `homog` at 20k nodes is 2.1886 against big-M's
2.7875 (and 0.7481 against 1.3432 with dual proofs off in both). Criterion 2 fails: on the
default path at 300 s `homog` reaches 30k nodes and bound 1.58, uncertified. The secondary
bar passes by a wide margin: 0.094 -> 1.61 at 10k and 0.094 -> 2.19 at 20k, the spec's
largest bound 1e27 -> 1e3, numerical LP verdicts from ~6,000 per 1k nodes to ~110 per 1k.

What the data says about the issue: with the magnitude pathology removed, the hull route on
cstr behaves like big-M's tree (the same node-status mix, dual proofs firing at the same
rate) on a model 2.7x larger, and reaches ~78 % of big-M's bound at equal nodes. The hull
lowering is no longer the reason the route does not certify; the remaining gap is the
larger model per node (the copies and their links are branched and propagated instead of
`x`), which is the hull's structural cost on a model whose nonlinear rows are all nonconvex
equalities. Pyomo's `gdp.hull` + SCIP does not certify cstr either (the issue's premise).

## 6. Implementation plan

Flag `DISCOPT_HULL_EXACT_PERSPECTIVE` (default OFF; `solver_tuning.py` field
`hull_exact_perspective`, `_env_flag(..., default=False)`, read through `_tuning()` like
`hull_conic_quadratic`). A row is added to `docs/dev/flag-retirement-audit.md` in the same
PR, naming the panel below as its exit.

Files and functions (`python/discopt/_relax/gdp_reformulate.py`):

1. `_polynomial_monomials(body) -> dict[tuple[str, ...], float] | None` — the prototype's
   `monomials()` (`exact_hull_proto.py`): scalar variables, constants, `+ - *`, `neg`,
   `SumOver`; `None` on anything else (division, `**`, function calls, array variables,
   `IndexExpression`, a `Parameter` — its coefficients must stay live, #1862). A pure
   function with its own unit test (distributivity, cancellation to an empty dict, the
   `None` cases).
2. `_homogenized_hull_row(con, y_k, copies, mono) -> tuple[Expression, Expression]` — the
   prototype's `homog` arm: `d = max |m|`, body `sum_m c_m prod_{i in m} v_i * y^(d-|m|)`,
   rhs `rhs * y^d`, in the row's own sense (an equality becomes `_le`/`_ge` as today). No
   new column, no new row beyond today's; the lowered model keeps today's 236 columns and
   597 rows on cstr (`check.log`).
3. In `_reformulate_disjunction_hull`, nonlinear branch (line ~3190, `if not origin_free`):
   when the flag is on, `_polynomial_monomials` succeeds, and the row is not proven convex
   (`classify_constraint`, as §111's `e0_classify.py` used — a convex row keeps the
   eps-perspective / cone, which is its exact convex hull), emit the homogenized row
   instead of `subst_body * y_clamp - eps g0 (1-y)`; otherwise fall through unchanged. The
   origin-free (#1917) path and the cone path are untouched. With the flag off the
   lowering is byte-identical.
4. Nesting (#1775 G2.1): the homogenized row is a polynomial in disjunct `k`'s locals and
   `y_k`; a parent re-hulls it like any other nonlinear local row (its `g(0)` is 0, so the
   eps-perspective applies, and under the flag the parent homogenizes it again, which is
   exact by the same argument with `y_k` as one more local). Covered by the nesting tests
   run under the flag.
5. Counter `_HULL_EXACT_STATS` (rows homogenized, rows declined and why), surfaced by the
   solve inventory so a run can prove the branch fired (CLAUDE.md §6).

Not built, recorded: the lifted construction (b) (`exact_hull_proto.py`, `lifted`/`both`),
retired by §5's measurement; the univariate extension (`exp`, `log`, fractional powers in a
disjunct row keep the eps-perspective and its magnitudes — the same homogenization is not
available, the ratio form (b') measured worst of all).

Tests (`python/tests/test_2142_hull_exact_perspective.py`):

- flag OFF: `reformulate_gdp(cstr, "hull")` byte-identical rows (compare `str(body)` of
  every row) and a 200-node deterministic `node_count`/bound identical to no flag;
- `_polynomial_monomials` unit test;
- `gdp_differential.run_shape` battery (`python/tests/gdp_differential.py`) on three new
  shapes with bilinear and cubic disjunct rows (an equality, an inequality, a product shared
  by two rows): no feasible point cut, no infeasible point admitted — the §5 feasible-point
  sampling (the battery enumerates the integer auxiliaries and decides the continuous ones
  by LP; the hull's copies are pinned, so the nonlinear verdict path applies);
- fixed-selector exactness: every configuration of the shapes solved under the flag, big-M
  and the shipped hull gives the same certified optimum; cstr's 3.0620073 reproduced under
  the flag (deterministic, `max_nodes` cap, backstop time limit);
- fixed-box differential: on 200 random sub-boxes of the shapes, the flag-ON LP bound is <=
  the enumerated true box optimum (soundness). The comparison with the flag-OFF bound is
  recorded, not asserted as `>=`: the two are different formulations, not a strengthening;
- the off-disjunct rows are exactly 0 in floating point at `v = 0, y = 0` (the #1043 class);
- magnitude: the lifted spec's largest finite |global bound| on cstr under the flag <= 1e4
  (through `solve_spatial_tree_py`'s `global_lo`/`global_hi`, as `arm.py` reads them);
- cstr regression: deterministic 2,000 nodes, flag ON bound > flag OFF bound and <= oracle.

Panel (§5, both bars, ON vs hull OFF): `--assert-unfired-identical`, over the in-repo corpus
(74, `max_nodes=2000`) and the GDP parity set (24 x hull, `max_nodes=5000`), plus the GDPlib
instances whose disjuncts carry polynomial nonconvex rows (a fire screen over
`python/tests/data` GDP cases lists them). Cert-clean: `incorrect_count = 0`, no bound above
its reference, no certification lost, incumbents feasibility-verified. Net-positive: a
certification gained or a tighter node-limited bound where it fires, bit-identical trees
where it does not. A pass flips the default (keep `=0` and today's path); a fail retires the
flag and the implementation in one PR.

## 7. Risks

- **It does not close #2142.** Both pre-registered criteria failed (§5). Building it is
  justified only as a fix of the hull route's magnitude pathology, measured ON against hull
  OFF by the panel; whoever builds it must not present it as the cstr certification.
- **Per-node tightness.** The homogenized row's McCormick is on the copies' node boxes, which
  contain 0 while `y` is fractional; a perspective-aware kernel term (the envelope on the box
  of `t = v/y`) would be tighter per node and is the one lever left on the relaxation side.
  The static version of it (`lifted`) made the tree worse, so this is not a safe bet either.
- **Model size per node** is the hull's structural cost (285 vs 105 kernel columns on
  cstr; ~3.5x the wall per node). Nothing in this design reduces it.
- **Scope.** Polynomial rows only in the first increment; a disjunct row with `exp`, `log`,
  a division or a fractional power keeps the eps-perspective and its magnitudes. The
  univariate extension (atom copy `w = y f(t)` with the envelope rows of `f` on `B_k`
  scaled by `y`, and the ratio link `v = y t` for the argument only) is the same
  construction; it is listed as the follow-up because cstr does not exercise it.
- **Rounding at `y = 0`.** Every row must be exactly 0 at the off disjunct (the #1043
  class). The homogenized row has no additive constant and no `eps`, so it is exactly 0 in
  floating point when every column is 0; the test asserts it.
- **`bound_pretransformation` is not the difference.** big-M gets the #1775 G2.4 bound rows
  `x <= sum_k ub_k y_k`; the hull's links and aggregation imply them, so giving the hull
  those rows is not a lever. The difference measured in §5 is the model per node.
