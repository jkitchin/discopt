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

### NS default-box fix: entry experiment KILLED (2026-11-02)

**Hypothesis.** The Rust MILP root bound, the NS safe bound (`DISCOPT_MILP_NS_NODE_BOUND`,
default ON), loses to the vertex bound on enough big-M masters, through the `±9.999e19`
default box, to mask the G3.2 levers. A fix would take the width from a valid finite
source (FBBT implied bounds, or a dual projection that zeroes the residual).

**Pre-registered kill.** Fewer than 3 instances where the NS root differs from the
vertex root by more than 1e-6 relative.

**Measurement.** `g32/replay/nsentry.py` covered all 36 captured Rust-MILP inputs of
the step-1 panel (7 in-repo, 23 LOA masters, 6 big-M). The bound was taken at
`max_nodes=2`, once with NS default and once with `DISCOPT_MILP_NS_NODE_BOUND=0`, each
in its own process (the flag is read once). Each run covered both coef-tighten arms:
72 comparisons per run, `PROBE_DONE 72` twice.

| coef tighten | instances with NS ≠ vertex (> 1e-6 rel) |
|---|---|
| OFF (default) | **0** of 36 |
| ON | **2** of 36: clay0203_master (−1.04e6 vs −1.4e-12), jobshop_small (−5542.31 vs 8.25) |

Every other row agrees to ≤ 1.4e-8 relative. That includes the 7 instances that
carry default-box columns but never show the effect: med_term_purchasing (78),
batch_processing (10), spectralog (8), constrained_layout_clay0203 (6) and
strip_packing_concrete (1). hda is `−inf` in both arms. spectralog's `−7.9992e20`
is its own optimum (HiGHS agrees), so it is a modelling artifact, not a bound gap.

**Verdict: killed.** At default settings the defect affects no instance in this set,
and it shows up only on a perturbed LP (the coef-tightened rows of two masters). It
stays recorded above as a known fragility of NS over the default box, not as a G3.2
lever. It does not explain the mixed step-1 panel either. Neither of the two node
regressions (disease_model, strip_packing_concrete) is in the affected set.

## Step 2 entry experiment: attribution grid (2026-11-02)

The lead asked that step 2 attribute its gain: does c-MIR need coefficient
tightening, or is coefficient tightening only riding along? This section answers
that with a prototype, before any Rust c-MIR is written (§4).

**Instrument.** `g32/tab_cmir_entry2.py` (session scratchpad). Each round it solves
the LP with HiGHS, takes `λ = B⁻ᵀe_i` for every fractional basic integer, and
aggregates `λᵀ(Ax − r) = 0` over structural columns and row activities. It then
generates a cut in one of three modes:

* `gmi`: δ = 1 only, which is GMI on the tableau row;
* `cmir_novub`: HiGHS-style δ search with complementation;
* `cmir`: the same plus variable-upper-bound substitution `x ≤ Σ U_v y_v`.

It keeps the 200 most efficacious cuts per round and runs 15 rounds. Every kept
cut is checked against the HiGHS-proven optimum (−6797.5397), and the run asserts
that the check count is positive. Inputs:

* `mtp_raw` is the MTP master as captured.
* `mtp_rcoef8` approximates what the Rust driver sees with
  `DISCOPT_RUST_MILP_COEF_TIGHTEN=1`: 8 FBBT sweeps (the Rust presolve cap), then
  the Rust pass through `milp_coef_tighten_csc_py`. It tightens 390 coefficients
  in 282 rows, the same count the driver reports in step 1.

| model \ separator (prototype, 15 rounds) | `gmi` (δ=1) | `cmir_novub` | `cmir` (+VUB) | validity checks |
|---|---:|---:|---:|---:|
| coef OFF (`mtp_raw`)    | −8308.5 | −8235.0 | −8358.5 | 830 / 875 / 871 |
| coef ON (`mtp_rcoef8`)  | **−7198.9** | −7211.9 | **−7111.1** | 2022 / 2052 / 2006 |
| FBBT-20 + coef (HiGHS ref. −7166.3) | — | −7159.7 | −7145.4 | 1672 / 1798 |

No cut removed the known optimum in any arm.

The same master through the **Rust driver** (`g32/rustgrid.py`, root bound at
`max_nodes=2`, its existing GMI + cover loop, `PROBE_DONE 10`):

| Rust root cut budget (cuts/rounds/select) | coef OFF | coef ON |
|---|---:|---:|
| 16 / 1 / no (legacy probe profile) | −8359.0 | −8322.1 |
| 200 / 10 / yes (`_STRONG_CUT_PROFILE`, the escalation) | −8359.0 | −8099.1 |
| 200 / 15 / yes | −8359.0 | −8099.1 |
| 2000 / 15 / no | −8359.0 | −7547.2 |
| 2000 / 30 / no | −8359.0 | −7508.6 (30 rounds, 1678 GMI cuts emitted) |

With coef OFF the Rust loop tails off after round 1: 28 cuts, no bound movement.

**Attribution.**

1. *Coefficient tightening is the lever, not a passenger.* Without it, no
   separator moves the root by more than 124 (best −8235.0), in the prototype or
   in Rust. With it, plain tableau GMI in the prototype reaches −7198.9.
2. *c-MIR needs coefficient tightening.* With coef OFF, c-MIR+VUB is −8358.5,
   no better than the bare LP. Its increment over GMI exists only with coef ON,
   and there it is **+87.8** (−7198.9 → −7111.1).
3. *δ search without VUB substitution is worth nothing here* (−7211.9 against
   −7198.9 for GMI). The c-MIR increment is VUB substitution.
4. *The largest gap is in the Rust cut loop, not the separator.* The same
   separator family (tableau GMI) on the same tightened model reaches −7198.9 in
   the prototype loop. The Rust loop reaches −7508.6 at a 2000/30 budget and
   −8099.1 at the budget the solver actually escalates to. That is a 310 gap, or
   900 at the escalation budget, against c-MIR's 88.

**Against the pre-registered gate** (pass ≥ −7200, kill < −7800). On coef ON,
prototype c-MIR+VUB passes at −7111.1, and prototype GMI sits on the line at
−7198.9. No Rust configuration passes today. The best Rust run is −7508.6, and
the escalation budget gives −8099.1, which is past the kill line.

## Step 2 result: Rust tableau c-MIR in the driver (2026-11-02)

The separator is `crates/discopt-core/src/lp/tableau_cmir.rs`, default OFF behind
`DISCOPT_RUST_MILP_TABLEAU_CMIR`. It is a port of the prototype's `cmir` mode. It
detects variable bounds once on the base rows, with one continuous column, the rest
integer, and a singleton logical. On MTP it finds 522. For each fractional-basic
integer, it runs c-MIR on the refined tableau row `w = B^-T e_i`, with both signs.

Probe: `scratchpad/loa/g32/grid2x2.py` and `prof2.py`, on the MTP replay input. It
measures the root bound at `max_nodes=2`, and it asserts both that `root <= -6797.54`
and that each c-MIR arm separated (12/12 arms ran).

| budget (total cuts / rounds / select) | coef OFF, c-MIR OFF | OFF, ON | ON, OFF | ON, ON |
|---|---:|---:|---:|---:|
| 16 / 1 / no (probe) | -8359.0 | -8359.0 | -8322.1 | -8322.1 |
| 200 / 10 / select (escalation default) | -8359.0 | -8359.0 | -8099.1 | -8119.3 |
| unbounded / 15 / no | | | -7547.2 | -7395.1 |
| unbounded / 30 / no | -8359.0 | -8359.0 | -7508.6 | -7291.9 |
| unbounded / 15 / select | | | -7446.1 | -7481.7 |
| unbounded / 30 / select | | | -7383.3 | -7308.8 |

The two unbounded/30/no ON/ON figures differ by version. The first build refused
1791 of 4398 rows on dynamism and gave -7270.8. The current build relaxes small
coefficients into the rhs (the HiGHS way) instead of refusing, which leaves 197
dynamism refusals and gives -7291.9. The table shows the current build.

What the grid says:

1. **Coefficient tightening remains the precondition.** With coef OFF, the loop
   tails off after round 1 for both separators, and neither moves the root. The
   prototype found the same with VUB substitution: -8358.5.
2. **c-MIR's gain in the Rust loop is small and budget-dependent.** It ranges from
   +238 to +75 at 30 rounds, and from +152 to -36 at 15 rounds. That is about the
   prototype's +88 over GMI, and the sign is not stable.
3. **At the escalation default, the budget binds, not the separator.** 200 total
   cuts are spent by round 3 with or without c-MIR.
4. **The entry gate is not met.** The best root is -7270.8 against a pass mark of
   -7200, and that figure needs 10x the default budget. No configuration reaches the
   kill threshold of -7800 at an unbounded budget. At the default budget, both
   separators sit at about -8100, which is below that threshold.
5. **The dominant lever is still the loop, not the separator.** The prototype's
   plain GMI, at 15 rounds × 200 cuts per round, reached -7198.9. Rust's best GMI
   is -7383.3, at 30 rounds with selection.
