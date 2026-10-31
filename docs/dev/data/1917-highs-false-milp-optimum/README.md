# HiGHS returns a false "optimal" on an OA master (found under #1917, 2026-10-31)

**Where it showed.** `pyomo.constrained_layout_clay0203` (a `gdp_parity` instance)
under `gdp_method="big-m"` on main `4517c7b`. The convex route's `lp_nlp_bb`
logs, on every run:

> lp_nlp_bb: master dual bound 42098.844879 is above the incumbent 41573.2625223
> by more than rounding -- one of the two is wrong

The inversion guard withholds that bound. The solve then falls back to NLP-BB and
certifies the correct optimum, 41573.26 (the GAMS oracle). So **no wrong answer
reached the user**.

**What is wrong.** `clay_cuts.py` captures the HiGHS lazy master: its base rows,
bounds and all 393 lazy and node cuts.
- The certified optimum `x*` satisfies every captured row to 1.8e-15, and
  `c·x* = 41573.26`. The fixed-integer LP at `x*`'s assignment is optimal at
  41573.26.
- Yet the master claims `OPTIMAL` at 42098.84. Its presolve-free cross-check
  claims 43285.35.

So the bound is wrong with respect to the master's **own** rows. The rows are
fine; the MILP solve is not.

`clay0203_master.mps` is that master (30 columns, 18 integer, 525 rows):

| solver | default | presolve off | random_seed=1 | random_seed=2 |
|---|---|---|---|---|
| **SCIP 6.3.0** | **41420.27 (optimal, proven)** | | | |
| HiGHS 1.15.1 / 1.14.0 / 1.13.1 | 56725.53 | 41907.46 | 55015.19 | 41907.46 |
| HiGHS 1.12.0 / 1.11.0 / 1.10.0 | 55239.57 | 41420.27 | 55239.57 | 55239.57 |

Every HiGHS entry except two reports `Optimal` at a value above the proven
optimum. Ruled out:
- Tiny coefficients: dropping the four `|a| < 1e-7` terms with outward rhs
  correction changes nothing.
- Feasibility tolerances: tightening them makes it worse (54422).
With `x*` given as a start (`setSolution`), HiGHS returns 41420.27.

**Exposure in discopt.**
- *The pure-MILP route* (`lp_milp_highs`, "verified HiGHS route"): the same
  MILP rebuilt through `Model.solve` certifies the correct 41420.27 on 7 of 7 row
  orderings (`mps_to_discopt.py`). Its own presolve and coefficient tightening
  sidestep the failure here; nothing guarantees they always will.
- *The `lp_nlp_bb` OA master*: the bound is used as given, guarded only by the
  inversion check (bound > incumbent). A false master bound that lands *within the
  gap tolerance of a suboptimal incumbent* would pass that guard. HiGHS's false
  "optima" are objective values of master-feasible points, and the NLP at such a
  configuration can produce that incumbent. **That is a false-certificate path.**
  It did not fire on this instance.

Not fixed here; the remedy is a design decision (owner).
