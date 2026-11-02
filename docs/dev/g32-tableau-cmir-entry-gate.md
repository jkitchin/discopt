# #1917 G3.2 — tableau c-MIR separator: pre-registered entry gate

Registered **before** any implementation (CLAUDE.md §4). It is a gate, not a report.

## Measured premise

On the first LOA master of `med_term_purchasing` (n=498, 216 binaries, 924 rows;
LP root −8358.999245, optimum −6797.5397), the Rust MILP root bound is the bare
LP under every cut profile. Root GMI cuts leave the objective unchanged (the LP
is dual-degenerate), and 60 s of B&B reaches bound −7698 without proving
optimality. We built `ref/HiGHS` with each separator env-gated and ran it at 1
thread, 60 s:

| HiGHS separators     | root bound | nodes   | wall   |
|----------------------|-----------:|--------:|-------:|
| all off              | −8358.999  | 438,498 | 60 s (unsolved, bound −7507.7) |
| tableau only         | −7061.0    | 1,433   | 0.73 s |
| path only            | −7512.6    | 2,010   | 0.69 s |
| default (all)        | −6908.9    | 65      | 0.45 s |

## Hypothesis

HiGHS's tableau separator pipeline alone closes the gap. The pipeline is
`HighsTableauSeparator.cpp`, then `HighsTransformedLp.cpp` (bound substitution,
simple and variable bounds), then `HighsCutGeneration.cpp` (c-MIR with
complementation and lifting heuristics, plus integral-support strengthening).
A faithful port of it closes the gap on the same master, where our GMI does not.
Probing implications are added only if the gate shows they are needed.

## Gate (on the fixture, prototype in Python or Rust)

1. **Pass:** the root bound after the root cut loop is **≥ −7100**.
2. **Soundness (non-negotiable):** every cut must keep the known optimal point
   feasible, and it must also keep feasible points sampled from the MILP
   (incumbents from fixed-binary LP solves) feasible.
3. **Kill:** a faithful port of the tableau path that stays **above −7800**
   (i.e. bound < −7800) is killed. The kill is reported together with its delta
   against HiGHS's cut-by-cut log.

## After the gate passes

1. Put it in the Rust root cut loop behind a default-OFF `DISCOPT_*` flag, with
   a row in `flag-retirement-audit.md`.
2. Add `cargo test -p discopt-core` unit tests for cut validity.
3. Run a differential panel (LOA master replay set, in-repo MILP corpus, MTP)
   that is cert-clean with no valid point cut, and report the node and wall
   ratios.
4. Graduate or retire the flag per CLAUDE.md §5.

## Result (2026-11-01): the premise is falsified, the kill criterion fires

**Retraction (§11).** The "tableau only" row above ran with HiGHS **presolve on**.
It does not measure the tableau separator alone, and the hypothesis built on it
("the tableau pipeline alone closes the gap") is wrong. The same patched HiGHS
binary, `threads=1`, `G32_OFF=path,modk,implbound,clique` (only the tableau
separator on), gives:

| model fed to HiGHS (presolve off unless noted)          | root after cuts | nodes | wall   |
|---------------------------------------------------------|----------------:|------:|-------:|
| raw fixture, HiGHS presolve **on** (the row above)      | −7061.0         | 1,433 | 0.73 s |
| raw fixture                                             | **−8357.4**     | 4,159 | 3.63 s |
| HiGHS's own presolved model (`write_presolved_model_file`) | −7233.2      | 1,355 | 0.62 s |
| raw + FBBT only                                         | −8281.7         | 2,964 | 2.26 s |
| raw + FBBT + OBBT, no coefficient tightening            | −8224.8         | 2,782 | 2.09 s |
| raw + FBBT + big-M coefficient tightening               | −7166.3         | 1,099 | 1.12 s |
| raw + FBBT + coef tightening + OBBT + coef again        | −7125.0         |   550 | 0.82 s |

Without presolve, HiGHS's own tableau separator stays at −8357.4. That is above
−7800, so a *faithful* port of the tableau path alone on this master is killed
by gate 3. It cannot do better than the code it ports. The lever is **big-M
coefficient tightening**. It removes 624 coefficients ≥ 1000 in HiGHS presolve
(max |a| 10100 → 171.9). For example, `x261 ≤ 1000·y76` becomes
`x261 ≤ 1.59·y76`. Coefficient tightening does not move the LP root (−8358.999
in every row). It changes which cuts the tableau rows yield. FBBT or OBBT bound
changes without it are worth almost nothing (−8282, −8225).

Our own separators gain less from the same reformulation. On the same tightened
models, under default kwargs, with the 60 s bound at 1 thread:

| model                        | Rust root | Rust 60 s bound      |
|------------------------------|----------:|---------------------:|
| raw                          | −8359.0   | −7763.6 (unsolved)   |
| FBBT + coef                  | −8219.3   | −7372.1 (unsolved)   |
| FBBT + coef + OBBT + coef    | −8057.1   | −7149.9 (unsolved)   |

So both pieces are needed. Coefficient tightening comes first (presolve), and
HiGHS-quality tableau c-MIR comes on top of it. Neither suffices alone. The
known optimum (−6797.5397) is feasible in both tightened models, with max row
violation 1.2e-10. Scripts are `g32/coef_entry.py` (with `ABL=` for the
ablation), `g32/rtight.py` and `g32/feas_chk.py` in the session scratchpad.

## Step 1 (2026-11-01): Rust big-M coefficient tightening, `DISCOPT_RUST_MILP_COEF_TIGHTEN`

The pass lives in `crates/discopt-core/src/bnb/coef_tighten.rs`. It runs in the
Rust MILP driver after the bound presolve, on "≤" rows only, and changes
integer/binary coefficients only. The formula is HiGHS `HPresolve::rowPresolve`
coefficient strengthening and SCIP `consdataTightenCoefs`. The flag is
default OFF. Measured on `mtp.mps` at 1 thread, load ~12:

| arm | coefs tightened | Rust root | Rust 60 s bound | incumbent |
|-----|----------------:|----------:|----------------:|----------:|
| OFF | 0               | −8358.999 | −7763.12 (89,823 nodes) | none |
| ON  | 390 (282 rows)  | −8304.98  | −7440.46 (97,791 nodes) | −6783.96 |

**Retraction of the step-1 prediction (§11).** I predicted −8219 for the root and
−7372 for the 60 s bound. Both are missed, and the pass is not the cause. On the
same fixture with bounds pre-tightened by the Python 20-sweep FBBT
(`mtp_fbbt_only.mps`), the identical Rust pass gives root −8219.14 (381 coefs),
which is the prediction exactly. The difference comes from the strength of the
Rust root bound presolve. `tighten_bounds_csc` (`lp/simplex/presolve.rs`) stops
at `max_rounds = 8`, where the Python FBBT runs 20 sweeps, so the pass sees
looser `u_j`, and therefore a looser max activity, on chained big-M rows. Raising
that cap is a *separate* bound-changing candidate. It is deliberately not part
of this flag, because it would confound the measurement. Use it only as its own
measured, flagged candidate if step 2 needs it.

Feasible-point check: 121 points feasible in the original model (the optimum,
30 random-objective MILP optima and their continuous re-solves at fixed
integers) are accepted as seeds in both arms (`g32/ct_seed.py`).

### Default-box guard, and why the flag ON can report a weaker root (2026-11-01)

The panel (below) gives a weaker Rust root with the flag ON on three instances:
clay0203_master, jobshop_small and modprodnet. The first hypothesis was that the
pass reads the Python default box `±9.999e19` (`constants.DEFAULT_VARIABLE_BOUND`)
as a finite bound. The pass now refuses such a row anyway:

* a side with `|bound| >= EFFECTIVE_INF` (`crate::numeric::EFFECTIVE_INF = 1e19`,
  shared with FBBT's `EXTENDED_MUL_INF`) counts as unbounded, in the spirit of
  HiGHS `impliedRowUpper != kHighsInf` and SCIP `hugeval`;
* a row whose rounding bound `gamma(k+2)·(Σ|a_j b_j| + |rhs|)` exceeds
  `COEF_TOL·d` is refused (`rows_skipped_precision`), standing in for HiGHS's
  `HighsCDouble` activity.

Both have unit tests (`default_box_sides_are_unbounded`,
`cancelled_huge_activity_is_refused_on_precision`). Each fails with its guard
reverted and passes with it in place. The hypothesis was **falsified for the
corpus**, though:

* On all four instances both skip counters are 0, and the guarded roots match the
  unguarded roots exactly: jobshop −5542.31, clay0203 −1.04007e6, modprodnet
  −1.07169e6, rsyn0830m −2060.33.
* None of the rows the pass tightens contains a default-box column
  (`g32/replay/hugechk.py`, 1,225 rows checked).
* The HiGHS LP optimum of the tightened rows equals that of the original rows:
  jobshop 8.0 / 8.0, clay0203 0.0 / 0.0, modprodnet −1246402.4 / −1246402.4. The
  same containment is now pinned on the in-repo fixtures where the pass fires
  (`python/tests/test_1917_milp_coef_tighten.py`).

**The cause is a pre-existing defect on main, outside this flag.** The Rust root
is the Neumaier–Shcherbina safe bound over the node box
(`DISCOPT_MILP_NS_NODE_BOUND`, default ON). On a column the model leaves at the
default box, it multiplies the reduced-cost residual by ~1e20. jobshop_small's
objective column has bounds `±9.999e19`, and clay0203's cost columns 24–29 have
`lb = −9.999e19`. Any equivalent but perturbed LP, such as one with tightened
coefficients, moves that noise. Two measurements isolate it (max_nodes = 2,
deterministic):

| instance | arm | NS bound (default) | `DISCOPT_MILP_NS_NODE_BOUND=0` | default box → 1e20 sentinel |
|---|---|---:|---:|---:|
| jobshop_small | OFF | 8.2475 | 8.247524752 | 8.2475 |
| jobshop_small | ON  | −5542.31 | 8.25 | 8.25 |
| clay0203_master | OFF | −1e-12 | 0.0 | −1e-12 |
| clay0203_master | ON  | −1.04007e6 | −1.42e-12 | ~−4e-12 |

With the vertex bound, or with the default box written as the sentinel, ON ≥ OFF
on both instances. **What OFF does:** on these instances the flag-OFF NS bound
happens to land near the LP value. It is still the same `residual × 9.999e19`
product and is only as good as that arm's rounding noise. This doc records the
defect and does not fix it. A fix belongs in the NS bound, which should treat
`|bound| >= EFFECTIVE_INF` as infinite, as the pass now does, and it is its own
bound-changing candidate.

modprodnet is different. Both arms are above the plain LP, and the gap after two
nodes comes from cuts and branching (OFF emitted 5 Gomory cuts and ON 118, with 0
kept in each), so it is a search effect.

**rsyn0830m:** the root is better ON (−2318.77 → −2060.33,
deterministic, max_nodes = 2). Its weaker 30 s bound is therefore a search effect,
which is acceptable by the stated criterion.
