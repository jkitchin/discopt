# #1008 — hypotheses, pre-registered (CLAUDE.md §4)

## Entry profile (the shared evidence)

`samply --rate 999` on the RLT-on QPLIB_1157 root relaxation LP (`m = 3937`),
in-house dual simplex 6.25 s vs HiGHS 0.216 s on the identical matrix:

| region                          | share |
|---------------------------------|------:|
| `FeralLU::factorize_sparse`     | 59.5% |
| — numeric `SparseLu::factor`    | 45.7% |
| — symbolic `SparseLuSymbolic::analyze` | 13.5% |
| Forrest–Tomlin `update`         | 18.7% |
| ftran / btran                   | 16.4% |
| pricing + ratio test            |  1.7% |

Iteration counts: in-house 4965 pivots vs HiGHS 2153 — **2.3× the pivots but
12.5× the time per pivot**. So the gap is the per-iteration linear algebra, not
the pivot rule. Pricing and the ratio test (two of the issue's named suspects)
are 1.7% and are ruled out as the primary cause.

---

## H1 — refactorization cadence *(FALSIFIED — see below)*

**Hypothesis.** Both pivot loops refactorize after a hardcoded `updates >= 48`,
a constant that does not scale with `m`. On these `m ≈ 4000–8000` lifted bases
that window is far too short, so 59.5% of the solve is rebuilding a factor that
was still perfectly usable. Lengthening the cadence closes a large part of the
gap.

**Experiment.** Sweep `DISCOPT_LP_REFACTOR_INTERVAL ∈ {48, 100, 200, 400, 800,
1600, 3200}` over 12 manifest-selected continuous nonconvex QPLIB relaxation
LPs, recording wall, iterations, status and objective vs HiGHS.

**Kill criterion (pre-registered).** H1 is falsified if either (1) wall time is
flat or worse as the interval grows past 48, or (2) the objective drifts from
HiGHS by more than 1e-6 relative at any interval, or the status stops being
`optimal`.

### Result: FALSIFIED, on both arms of the kill criterion.

| instance        |    m | HiGHS |    48 |    100 |    200 | best |
|-----------------|-----:|------:|------:|-------:|-------:|-----:|
| QPLIB_0911_rlt0 | 5150 | 0.049 | 1.948 | 46.99! | 46.91! |   48 |
| QPLIB_0911_rlt1 | 5150 | 0.048 | 1.930 | 47.04! |      – |   48 |
| QPLIB_1157_rlt0 | 3273 | 0.030 | 0.280 |  0.254 |      – |  100 |
| QPLIB_1157_rlt1 | 3937 | 0.200 | 6.341 |  5.039 |      – |  100 |
| QPLIB_1619_rlt0 | 5135 | 0.119 | 1.196 |  1.594 |      – |   48 |
| QPLIB_1619_rlt1 | 5635 | 0.542 |14.851 | 11.481 |      – |  100 |
| QPLIB_1745_rlt0 | 5160 | 0.085 |47.02! | 46.37! |      – |    – |
| QPLIB_1745_rlt1 | 5660 | 0.466 |41.113 | 31.025 |      – |  100 |
| QPLIB_1886_rlt0 | 2550 | 0.030 | 0.488 |  0.431 |      – |  100 |
| QPLIB_1886_rlt1 | 2550 | 0.031 | 0.499 |  0.432 |      – |  100 |
| QPLIB_1967_rlt0 | 5175 | 0.120 | 2.006 |  1.796 |      – |  100 |
| QPLIB_1967_rlt1 | 5175 | 0.118 | 2.004 |  1.788 |      – |  100 |

(`!` = hit the 45 s per-LP limit, status `iter_limit`.)

Best-interval speedup vs 48: n=11, min 1.00×, **median 1.12×**, max 1.33× — and
QPLIB_0911 (both arms) *regressed* from a 1.9 s `optimal` to a 47 s `iter_limit`,
tripping arm (2) of the kill criterion outright.

Two independent reasons the mechanism cannot work, both visible in the same data:

1. **Amdahl.** Refactorization is 59.5% of wall, so eliminating it *entirely*
   caps the speedup at 2.47× against a measured 8.7–29× gap. The lever is too
   small even at its theoretical limit.
2. **The work only moves.** Halving the refactorization count lengthens the eta
   chain, and ftran/btran/update all replay it. Predicted 1.42× on
   QPLIB_1157_rlt1 if refactorization cost halved with nothing else changing;
   measured 1.26×.

**Retraction (§11).** An earlier draft of `simplex/refac.rs` stated "the fix is
to stop treating 48 updates as the amortization window". That claim is
withdrawn: the measurement above shows the cadence is worth ~1.1× and
destabilises one instance in six. The adaptive-cadence implementation has been
reverted; only the profiling counters it motivated are kept.

---

## H2 — LU fill-in from strict partial pivoting

**Evidence.** The profile says three *different* LU hot spots are slow at once —
refactorize (59.5%), FT update (18.7%), triangular solves (16.4%), 94.6%
together. One quantity sets the cost of all three: the size of the factor.
Measuring it (`profile::Ctr::LuFactorNnz / LuBasisNnz`, `DISCOPT_PROFILE=1`,
same 12 LPs) shows the fill ratio `nnz(L+U)/nnz(B)` ranging 2.57–17.54 and
tracking wall almost monotonically:

| instance        |    m | nnz(B) | nnz(L+U) |  fill | wall (s) |
|-----------------|-----:|-------:|---------:|------:|---------:|
| QPLIB_1619_rlt0 | 5135 |  25780 |    66349 |  2.57 |     1.17 |
| QPLIB_1157_rlt0 | 3273 |   8361 |    21682 |  2.59 |     0.28 |
| QPLIB_1967_rlt* | 5175 |  54564 |   179777 |  3.29 |     1.92 |
| QPLIB_1886_rlt* | 2550 |  19824 |    70550 |  3.56 |     0.51 |
| QPLIB_1157_rlt1 | 3937 |  26286 |   151360 |  5.76 |     6.23 |
| QPLIB_1619_rlt1 | 5635 |  43965 |   286682 |  6.52 |    15.57 |
| QPLIB_0911_rlt* | 5150 |  24525 |   166480 |  6.79 |     1.92 |
| QPLIB_1745_rlt1 | 5660 |  50277 |   474539 |  9.44 |    42.23 |
| QPLIB_1745_rlt0 | 5160 |  35542 |   623564 | 17.54 |   >45 (limit) |

**Hypothesis.** The fill is inflated by the pivoting rule, not by the problem.
feral's `LuParams::pivot_threshold` defaults to `1.0` — strict partial pivoting —
and discopt inherits that default. But the sparse path first computes a
fill-reducing column ordering (AMD on the AᵀA pattern); strict partial pivoting
overrides that plan at every step, so the ordering's fill estimate is discarded.
Relaxing the threshold to the standard LP value (`u ≈ 0.1`, Davis §6.5 `cs_lu`
`tol`; Suhl & Suhl 1990) lets the ordering hold and shrinks `nnz(L+U)`, which
cuts refactorization, FT update and every triangular solve **simultaneously** —
94.6% of the wall, not 59.5%.

**Experiment.** Sweep `DISCOPT_LU_PIVOT_THRESHOLD ∈ {1.0 (baseline), 0.5, 0.1,
0.01}` over the same 12 LPs; record fill, iterations, wall, status, and the
objective against HiGHS on every cell.

**Kill criterion (pre-registered).** H2 is falsified if *any* of:

1. Fill (`nnz(L+U)/nnz(B)`) does not fall by at least 20% at `u = 0.1` on the
   high-fill instances (the ones the hypothesis says are mis-pivoted);
2. Median wall does not improve versus `u = 1.0`;
3. Any cell's objective drifts from HiGHS by more than `1e-6` relative, or a
   status regresses from `optimal` (a sparser factor bought with a wrong or
   undecided answer is not a speedup — CLAUDE.md §1).

**Regime.** Loosening the threshold changes the pivot sequence, so this is
*bound-changing* under CLAUDE.md §5: default `1.0` (byte-identical to today),
shipped behind `DISCOPT_LU_PIVOT_THRESHOLD`, default-OFF pending the corpus
differential panel.

### Result: FALSIFIED, on arms 1 and 3 of the kill criterion.

Sweep completed: 4 thresholds x 12 LPs = 48 cells, all measured
(`scratchpad/i1008/pivsweep.jsonl`, report `pivreport.py`).

**Arm 1 (fill drop >= 20% at u=0.1 on the high-fill instances) — FAILED.**
6 high-fill instances evaluated; not one reached 20%, and the worst-filling
instance in the corpus got *worse*:

| instance        | fill u=1.0 | fill u=0.1 |  drop |
|-----------------|-----------:|-----------:|------:|
| QPLIB_0911_rlt* |       6.79 |       7.39 | -8.8% |
| QPLIB_1157_rlt1 |       5.76 |       5.39 |  6.4% |
| QPLIB_1619_rlt1 |       6.52 |       5.58 | 14.4% |
| QPLIB_1745_rlt0 |      17.45 |      14.34 | 17.8% |
| QPLIB_1745_rlt1 |       9.44 |       9.24 |  2.1% |

**Arm 3 (no status regression) — FAILED.** QPLIB_0911 solves in 1279 iterations
and 1.9 s at u=1.0 but blows up to 5376 iterations / 46.5 s `iter_limit` at
u=0.5 and 6656 iterations / 47.0 s at u=0.01. Objective drift was clean
(0 cells above 1e-6 vs HiGHS), but an instance that stops finishing is a
regression under CLAUDE.md §1 regardless.

**Arm 2 (median wall) — passed, and it does not rescue the hypothesis.**
Best-per-instance speedup vs u=1.0: n=11, min 1.00x, median 1.14x, max 1.61x —
and "best per instance" is an oracle that cherry-picks u after the fact; there
is no single u that wins broadly (best-u is 0.01 on four, 0.5 on four, 0.1 on
one, and 1.0 on two). Against an 8.7x-29x gap this is noise.

**Why it failed (root cause).** feral's sparse factorization is CSparse `cs_lu`:
at step k it prefers the *diagonal* entry `w[qcol[k]]` when
`|w[qcol[k]]| >= u * amax`, otherwise the largest entry. That rule only has
something to prefer if the identity is already a good transversal of the matrix.
For an LP basis — columns are basis slots in whatever order the ratio test
assigned them, rows are constraints — there is no such natural diagonal, and
feral computes no maximum transversal before factorizing. Production codes pair
threshold pivoting with exactly that pre-pass (MC64/HSL, MUMPS, SuperLU_DIST;
HiGHS's Suhl-Suhl triangularization). The threshold and the transversal are one
mechanism; the threshold alone has nothing meaningful to prefer, which is
precisely what the fill table shows.

**Incidental finding (not H2, worth its own issue).** QPLIB_0911 degrades under
*any* perturbation of the pivot path — refactor interval 100 (H1): 6400 iters /
timeout; u=0.5: 5376 / timeout; u=0.01: 6656 / timeout — while sitting at 1279
iterations on the baseline path. HiGHS does the same LP in 1050. That is a
degeneracy/stall pathology in the dual simplex's ratio test, independent of both
levers tested here.

### Re-scope experiment: is there ANY fill headroom? — NO.

Before building the transversal that H2's root cause pointed at, measure the
headroom directly (`scratchpad/i1008/headroom.py`): take the basis discopt's
engine *ends* on, and factor that same matrix with SuperLU
(`scipy.sparse.linalg.splu`, COLAMD ordering, `diag_pivot_thresh=0.1`) — i.e. a
production sparse LU that does have the fill-reducing ordering plus threshold
pivoting plus a transversal. Same matrix on both sides, so the pivot path is
factored out. Fill ratios (`nnz(L+U)/nnz(B)`):

| instance        | feral | SuperLU | headroom |
|-----------------|------:|--------:|---------:|
| QPLIB_0911_rlt0 |  6.79 |    8.73 |    0.78x |
| QPLIB_0911_rlt1 |  6.79 |    8.73 |    0.78x |
| QPLIB_1157_rlt0 |  2.59 |    1.39 |    1.86x |
| QPLIB_1157_rlt1 |  5.76 |    6.65 |    0.87x |
| QPLIB_1619_rlt0 |  2.57 |    6.25 |    0.41x |
| QPLIB_1619_rlt1 |  6.52 |    6.45 |    1.01x |
| QPLIB_1745_rlt0 | 17.36 |   13.45 |    1.29x |
| QPLIB_1745_rlt1 |  9.27 |    9.55 |    0.97x |
| QPLIB_1886_rlt* |  3.56 |    4.15 |    0.86x |
| QPLIB_1967_rlt* |  3.29 |    3.50 |    0.94x |

n = 12, min 0.41x, **median 0.94x**, max 1.86x. feral's factor is already at
parity with SuperLU's and is *smaller* on 8 of 12. The fill on these bases is
intrinsic to the bases, not an artifact of our factorization, and no ordering /
transversal / threshold change can recover it. **The fill theory is dead, and
the transversal work H2's post-mortem proposed is cancelled** — building it would
have chased a headroom of 0.94x. (CLAUDE.md §4: the measurement wins.)


## H3 — which HALF of "refactorize" is the 59.5%? (attribution, not a fix)

The entry flamegraph said "refactorize 59.5%". A sparse refactorization is two
unrelated things with unrelated fixes, so the split was measured directly by
adding `profile::Phase::LuSymbolic` (`SparseLuSymbolic::analyze` — AMD on the
`AᵀA` pattern) and `LuNumeric` (`SparseLu::factor`). 18 captured QPLIB
relaxation LPs, `scratchpad/i1008/h3.py`:

| instance        | total ms | symb% | num% | other% | n facs |
|-----------------|---------:|------:|-----:|-------:|-------:|
| QPLIB_0911_rlt0 |   2424.3 | 11.1% | 51.2%|  37.7% |     27 |
| QPLIB_1157_rlt0 |    331.1 | 20.6% | 18.3%|  61.1% |     31 |
| QPLIB_1157_rlt1 |   7385.2 | 13.5% | 47.0%|  39.5% |    104 |
| QPLIB_1451_rlt0 |  47704.8 |  4.4% | 73.9%|  21.7% |     75 |
| QPLIB_1451_rlt1 |  51830.4 |  5.5% | 69.5%|  25.0% |    107 |
| QPLIB_1535_rlt0 |   2841.0 | 19.8% | 22.5%|  57.6% |     76 |
| QPLIB_1535_rlt1 |  46216.3 |  3.3% | 64.2%|  32.5% |    134 |
| QPLIB_1619_rlt0 |   1211.9 | 25.5% | 22.8%|  51.8% |     49 |
| QPLIB_1619_rlt1 |  16664.4 |  5.3% | 56.7%|  38.0% |    128 |
| QPLIB_1703_rlt0 |   3703.5 | 11.8% | 37.1%|  51.1% |     91 |
| QPLIB_1703_rlt1 |  46991.4 |  2.7% | 66.7%|  30.6% |    118 |
| QPLIB_1745_rlt0 |  46831.8 |  4.5% | 71.7%|  23.8% |    129 |
| QPLIB_1745_rlt1 |  42256.6 |  6.1% | 65.2%|  28.7% |    171 |
| QPLIB_1886_rlt* |    535.5 | 16.2% | 33.2%|  50.6% |     20 |
| QPLIB_1967_rlt* |   1984.4 |  9.9% | 45.5%|  44.6% |     33 |

Numeric ≈50% (median), symbolic ≈11% (median, range 2.7–25.5%), everything else
(ftran/btran, FT update, pricing) ≈38%. On the *sparse* bases the symbolic
analysis costs **more than the numeric factorization it prepares**
(QPLIB_1157_rlt0 68.1 vs 60.7 ms; QPLIB_1619_rlt0 308.6 vs 275.7 ms).

## H4 — is the numeric LU slow per unit of work? Somewhat, and it is not ours.

`scratchpad/i1008/h4.py`: feral's ms per numeric factorization against
`scipy.sparse.linalg.splu` (COLAMD, `diag_pivot_thresh=0.1`, best of 3) on the
basis the engine ends on. n=18, min 0.6x, **median 2.1x**, max 4.1x — and the
comparison is biased *in feral's favour* (feral's figure averages over its whole
path, whose early bases are sparser than the final one).

So the numeric kernel is a genuine ~2x constant-factor gap, but it lives in the
external `feral` crate, not in this repository, and at ~50% of wall a perfect fix
would be worth at most `1/(0.5/2.1 + 0.5) = 1.36x` — against a measured 8.7x–29x
gap. Recorded, not actionable here.

## H5 — the symbolic ordering is recomputed from scratch every refactorization

**Evidence.** H3: symbolic analysis is a median 11% of LP wall (up to 25.5%), and
it is redundant by construction — a simplex basis changes by exactly one column
per pivot and discopt refactorizes roughly every 48 pivots, so successive bases
differ in ~1% of their columns while the ordering is rebuilt from nothing every
time. `ata_pattern()` is quadratic in the length of the densest basis row.

**Soundness.** `SparseLuSymbolic` is `{ m, qcol, qcol_inv }` — a column
permutation and nothing else (no elimination tree, no predicted L/U pattern);
`SparseLu::factor` validates only `symbolic.m == a.m`; the numeric factorization
picks its own row pivots. So *any* permutation factors the matrix handed in
correctly, and a stale ordering can only cost fill. feral's own module doc calls
the handle reusable. The guard is therefore on fill, not on correctness.

**Hypothesis.** Caching the ordering inside `FeralLU` and reusing it while the
resulting `nnz(L+U)` stays within 1.25x of what a fresh analysis achieved removes
most of the symbolic cost, for a median speedup of roughly 1.10x — small, but the
only lever of the five examined that is contained, sound, and not falsified.

**Kill criterion (pre-registered).** H5 is falsified if any of:

1. Median speedup over the 18 LPs is below 1.05x (i.e. the cache does not pay for
   the guard's occasional wasted factorization);
2. Any instance's objective drifts from the flag-off arm by more than 1e-9
   relative, or a status regresses from `optimal`;
3. Reuse rate is under 50% of sparse factorizations (the guard trips so often
   that the mechanism is not actually operating).

**Regime.** A different column order changes rounding and hence the pivot
sequence, so *bound-changing* under CLAUDE.md §5: default **off**
(`DISCOPT_LU_SYMBOLIC_REUSE=1` to opt in), byte-identical to today when off,
pending the corpus-wide differential panel §5 requires for graduation.

---

## H6 — the DUAL ratio test has no stability tie-break; the PRIMAL one does

**Evidence (three independent measurements, all already in this file).**

1. The entry profile: in-house 4965 pivots vs HiGHS 2153 on QPLIB_1157_rlt1 —
   **2.3x the iterations**. H1–H5 all attacked the *cost per iteration*; none
   touched the iteration count, which is the other factor in the 8.7x–29x gap.
2. QPLIB_0911 solves in 1279 pivots / 1.9 s on the default path and goes to
   `iter_limit` (5376–6656 pivots, >45 s) under **five** different rounding-level
   perturbations of the pivot path — refactor interval 100 and 200 (H1), LU pivot
   threshold 0.5 and 0.01 (H2), and symbolic-ordering reuse (H5). A perturbation
   that changes no tolerance, guard or bound formula cannot multiply the iteration
   count by 5x unless the pivot path is marginally stable to begin with.
3. `DISCOPT_PROFILE=1` on the QPLIB_0911 baseline: `DualDegeneratePivots` = 553
   of 1279 pivots (43% zero-length steps), `DualStallTrips` = 0.

**The asymmetry.** `primal.rs` implements a Harris two-pass bounded ratio test
with an EXPAND anti-degeneracy shift (`primal.rs:1397` "Pass 2: among rows that
block within `t` (Harris), take the LARGEST pivot"; `primal.rs:1721` "EXPAND
anti-degeneracy (Gill et al.)"). `dual.rs` has **no** occurrence of "Harris" and
no pass 2: its bound-flipping walk (`dual.rs:764-783`) takes whichever breakpoint
happens to close `delta`, with no reference to the pivot magnitude `|α_rj|` —
even though `build_candidates` already returns `|α_rj|` as the third field of
every candidate and it is otherwise used only for the slope accumulation. Under
degeneracy the leading breakpoints are a cluster of near-equal (typically zero)
ratios, so *which* of them enters is arbitrary, and taking a tiny `|α_rj|`
produces the zero-length step, the long eta chain, and — at `dual.rs:804` — the
`piv.abs() <= tol` bail-out to the cold fallback.

**Hypothesis.** Adding the missing pass 2 — among candidates at or before the
chosen breakpoint whose ratio is within `harris_tol` of it, enter the one with the
largest `|α_rj|` — cuts the dual iteration count on degenerate LPs and removes the
marginal stability that makes QPLIB_0911 collapse under any perturbation.

**Soundness.** The search is restricted to sorted indices **≤** the walk's own
stopping index, i.e. to breakpoints with ratio *no larger* than the one the
unmodified rule picked. The dual step length can therefore only get **shorter**,
and a shorter dual step cannot make any reduced cost dual-infeasible — so unlike
textbook Harris this variant needs no relaxed dual-feasibility tolerance and
introduces no dual excursion at all. The flip set is the same rule applied at the
new stopping index (`flips = cand[0..idx(q*)]`, all finite-gap by construction).
The existing exact-`dvec` `dual_feasible()` gate before `Optimal`, the
Neumaier–Shcherbina safe bound, and the `piv.abs() <= tol` guard are untouched.
Bland mode is untouched (it must keep its provably non-cycling smallest-index
rule).

**Experiment.** (a) Entry: the five perturbation cells of QPLIB_0911 above, with
the flag on — does the stall clear? (b) Panel: flag ON vs OFF, interleaved, over
the same 18 captured QPLIB relaxation LPs as H5, recording iterations, wall,
status and objective.

**Kill criterion (pre-registered).** H6 is falsified if any of:

1. The QPLIB_0911 entry cell does not improve — i.e. iterations with the flag on
   are not below the 1279-pivot baseline AND the `DISCOPT_LU_SYMBOLIC_REUSE=1`
   perturbation still goes to `iter_limit` with the flag on;
2. Median iteration count over the 18 LPs is not reduced by at least 5%;
3. Any instance's objective drifts from the flag-off arm by more than 1e-9
   relative, or a status regresses from `optimal`.

**Regime.** Choosing a different entering column changes the pivot sequence:
*bound-changing* under CLAUDE.md §5. Default **off**
(`DISCOPT_LP_DUAL_HARRIS=1` to opt in), byte-identical to today when off, pending
the corpus-wide differential panel §5 requires for graduation.

### H5 Result: FALSIFIED, on criteria 1 and 2.

Interleaved A/B (`scratchpad/i1008/ab.py`, 3 reps per arm, arms alternated within
each instance, 45 s per-LP limit), 13 of the 18 captured LPs completed before the
run was cut; `uptime` load average **4.57** at launch — see the caveat below.

| instance        | OFF (s)        | ON (s)         | speedup | facs | reuse | refill |
|-----------------|---------------:|---------------:|--------:|-----:|------:|-------:|
| QPLIB_0911_rlt0 |  2.092 ± 0.042 | 45.660 ± 0.845 |  0.05x! |  129 | 89.9% |     12 |
| QPLIB_0911_rlt1 |  2.072 ± 0.299 | 46.857 ± 0.663 |  0.04x! |  123 | 89.4% |     12 |
| QPLIB_1157_rlt0 |  0.317 ± 0.006 |  0.263 ± 0.003 |   1.21x |   30 | 70.0% |      8 |
| QPLIB_1157_rlt1 |  6.619 ± 0.500 |  6.655 ± 0.370 |   0.99x |  105 | 75.2% |     25 |
| QPLIB_1451_rlt0 | 47.511 ± 0.861 | 46.150 ± 0.647 |   1.03x |   86 | 67.4% |     27 |
| QPLIB_1451_rlt1 | 45.954 ± 0.102 | 45.674 ± 0.295 |   1.01x |  107 | 57.0% |     45 |
| QPLIB_1535_rlt0 |  2.298 ± 0.018 |  2.140 ± 0.032 |   1.07x |   74 | 75.7% |     17 |
| QPLIB_1535_rlt1 | 45.296 ± 0.062 | 46.706 ± 0.053 |   0.97x |  145 | 86.2% |     19 |
| QPLIB_1619_rlt0 |  1.132 ± 0.012 |  1.121 ± 0.008 |   1.01x |   50 | 72.0% |     13 |
| QPLIB_1619_rlt1 | 15.345 ± 0.035 | 14.455 ± 0.071 |   1.06x |  119 | 83.2% |     19 |
| QPLIB_1703_rlt0 |  3.214 ± 0.019 |  2.584 ± 0.055 |   1.24x |   90 | 76.7% |     20 |
| QPLIB_1703_rlt1 | 47.070 ± 0.021 | 47.073 ± 0.116 |   1.00x |  113 | 55.8% |     49 |
| QPLIB_1745_rlt0 | 46.763 ± 1.155 | 45.666 ± 0.738 |   1.02x |  145 | 84.8% |     21 |

(`!` = the ON arm hit the 45 s limit with status `iter_limit`.)

**Criterion 2 (no status regression) — FAILED.** QPLIB_0911, both arms:
`optimal` (2.1 s, 1279 pivots) → `iter_limit` (>45 s, 6144 pivots). An
`iter_limit` on a node LP is an uncertified node; under CLAUDE.md §1 that ends
the discussion regardless of anything else in the table. Objective agreement was
otherwise clean: 0 of 11 `optimal`/`optimal` pairs drifted above 1e-9 relative.

**Criterion 1 (median ≥ 1.05x) — FAILED.** Median over all 13 is **1.009x**;
excluding the two QPLIB_0911 rows (i.e. giving the mechanism every benefit of the
doubt) it is **1.024x**, still under the 1.05x bar. Two instances gain
meaningfully (1.21x, 1.24x) and one loses (0.97x); the rest are ~1.0x.

**Criterion 3 (reuse ≥ 50%) — passed.** 1016 of 1316 sparse factorizations
(**77.2%**) reused the cached ordering; 287 refreshes on the fill cap, 0 on a
factorization failure. So the mechanism *operated* — it simply is not worth
anything. This is the `DISCOPT_CUT_INHERIT` shape: sound, and not net-positive.

**Load caveat (§9).** `uptime` reported load average **4.57** when the A/B
started. The 0.05x/0.04x regressions are three orders of magnitude outside any
plausible noise and stand. The ~1.0x cluster does **not**: at that load the
per-arm ± spreads understate the true variance, so 0.97x–1.07x should be read as
"indistinguishable from no effect", not as eleven separate measurements. That
does not rescue the hypothesis — H5 needed ≥ 1.05x *median* and the cluster is
centred on 1.0 — but no individual cell in that range should be cited.

**Retraction (§11).** Commit `c67fa62f` ("perf(lp): #1008 reuse the LU symbolic
ordering across refactorizations") shipped this flag with the *prediction* of "a
median speedup of roughly 1.10x" and no measurement. That prediction is
withdrawn: measured median is 1.009x with two status regressions.
`DISCOPT_LU_SYMBOLIC_REUSE` **must not graduate** and stays default-OFF. The
implementation is retained only because it is byte-identical when off and because
its `Phase::LuSymbolic`/`LuNumeric` split is what produced the H3 attribution —
not because it is a speedup.

### H6 entry cell: the stall CLEARS. Criterion 1 not tripped.

Four arms, interleaved within each rep, 2 reps, `uptime` load average 3.94–4.26
(`scratchpad/i1008/h6.py`, log `h6_entry.log`). Both arms of QPLIB_0911:

| instance        | arm    | wall (s)       | iters | status       | repivots |
|-----------------|--------|---------------:|------:|--------------|---------:|
| QPLIB_0911_rlt0 | base   |  1.936 ± 0.003 |  1279 | `optimal`    |        0 |
| QPLIB_0911_rlt0 | harris |  2.507 ± 0.057 |  1260 | `optimal`    |       51 |
| QPLIB_0911_rlt0 | sym    | 46.225 ± 0.744 |  6400 | `iter_limit` |        0 |
| QPLIB_0911_rlt0 | both   |  4.395 ± 0.118 |  1655 | `optimal`    |      102 |
| QPLIB_0911_rlt1 | base   |  2.140 ± 0.245 |  1279 | `optimal`    |        0 |
| QPLIB_0911_rlt1 | harris |  2.792 ± 0.318 |  1260 | `optimal`    |       51 |
| QPLIB_0911_rlt1 | sym    | 46.731 ± 0.602 |  6400 | `iter_limit` |        0 |
| QPLIB_0911_rlt1 | both   |  4.374 ± 0.027 |  1655 | `optimal`    |      102 |

Identical on both arms of the instance, so this is the LP's structure, not one
captured matrix. Objectives agree to 3e-12 relative across all four arms.

**Criterion 1 — not tripped, on both of its clauses.** Iterations with the flag
on are 1260 < 1279 baseline, and the `sym` perturbation that produced a >45 s
`iter_limit` becomes `optimal` in 4.4 s once the stability pass is on: **10.5x**
on that cell, and an uncertified node becomes a certified one. `DualStallTrips`
stays 0 throughout, confirming the existing anti-cycling guard is not what
rescued it.

**The cost, stated plainly.** On the *unperturbed* path the pass is a wall
**regression**: 1.936 → 2.507 s (0.77x) for 1.5% fewer pivots, i.e. ~31% more
time per pivot. 51 repivots out of 1260 pivots (4%) is not enough scanning to
explain that, so the cost is in what the repivots *do* — a shorter dual step
means fewer bound flips per pivot, so each pivot buys less progress. Whether that
trade is net-positive is exactly what the 18-LP panel decides; the entry cell
does not settle it and is not claimed to.

### H6 Result: FALSIFIED, on criteria 2 and 3.

Panel: `base` vs `harris`, interleaved within each rep, 2 reps, 45 s per-LP limit,
all 18 captured QPLIB relaxation LPs (`scratchpad/i1008/h6_panel.log`).
`uptime` load average **9.37** at launch, 3.55 at finish — see the caveat.

| instance        | base (s) | harris (s) | speedup | iters base | iters harris | Δiter | repivots | status base/harris |
|-----------------|---------:|-----------:|--------:|-----------:|-------------:|------:|---------:|--------------------|
| QPLIB_0911_rlt0 |    1.914 |      2.451 |    0.78 |       1279 |         1260 | −1.5% |       51 | optimal/optimal    |
| QPLIB_0911_rlt1 |    1.920 |      2.481 |    0.77 |       1279 |         1260 | −1.5% |       51 | optimal/optimal    |
| QPLIB_1157_rlt0 |    0.275 |      0.276 |    1.00 |       1481 |         1481 |  0.0% |        0 | optimal/optimal    |
| QPLIB_1157_rlt1 |    6.217 |      6.153 |    1.01 |       4965 |         4965 |  0.0% |        0 | optimal/optimal    |
| QPLIB_1451_rlt0 |   48.275 |     45.918 |    1.05 |       3840 |         3328 |−13.3% |      397 | iter_limit/iter_limit |
| QPLIB_1451_rlt1 |   46.124 |     48.546 |    0.95 |       5120 |         4096 |−20.0% |      863 | iter_limit/iter_limit |
| QPLIB_1535_rlt0 |    2.305 |      2.553 |    0.90 |       3613 |         3574 | −1.1% |        8 | optimal/optimal    |
| QPLIB_1535_rlt1 |   45.117 |     45.361 |    0.99 |       6912 |         6912 |  0.0% |        0 | iter_limit/iter_limit |
| QPLIB_1619_rlt0 |    1.148 |      1.266 |    0.91 |       2319 |         2209 | −4.7% |        9 | optimal/optimal    |
| QPLIB_1619_rlt1 |   15.316 |     12.468 |    1.23 |       6121 |         5866 | −4.2% |        1 | optimal/optimal    |
| QPLIB_1703_rlt0 |    3.261 |      3.248 |    1.00 |       4321 |         4298 | −0.5% |        5 | optimal/optimal    |
| QPLIB_1703_rlt1 |   47.534 |     47.015 |    1.01 |       5888 |         5888 |  0.0% |        0 | iter_limit/iter_limit |
| QPLIB_1745_rlt0 |   45.725 |     46.637 |    0.98 |       6144 |         6656 | +8.3% |      540 | iter_limit/iter_limit |
| QPLIB_1745_rlt1 |   42.505 |     46.039 |    0.92 |       8183 |         7424 | −9.3% |      289 | **optimal/iter_limit** |
| QPLIB_1886_rlt0 |    0.517 |      0.517 |    1.00 |        947 |          947 |  0.0% |        0 | optimal/optimal    |
| QPLIB_1886_rlt1 |    0.520 |      0.515 |    1.01 |        947 |          947 |  0.0% |        0 | optimal/optimal    |
| QPLIB_1967_rlt0 |    1.909 |      1.908 |    1.00 |       1552 |         1552 |  0.0% |        0 | optimal/optimal    |
| QPLIB_1967_rlt1 |    1.944 |      1.946 |    1.00 |       1552 |         1552 |  0.0% |        0 | optimal/optimal    |

**Criterion 2 (median iteration reduction ≥ 5%) — FAILED.** Median Δiter is
**−0.3%** over all 18 and **−0.5%** over the 13 that reach `optimal`. Pass 2
fires at all on only 10 of 18, and on 8 of those it moves fewer than 60 of
several thousand pivots. Wall speedup median **0.999x** (min 0.774x, max 1.228x)
— no effect on throughput.

**Criterion 3 (no status regression) — FAILED.** QPLIB_1745_rlt1 goes from
`optimal` (42.5 s, 8183 pivots) to `iter_limit` (46.0 s). Its objective then
differs by 3.2e-02 relative, which is simply the mid-loop value of an unfinished
solve, not a wrong optimum — but the uncertified node stands. Every other
`optimal`/`optimal` pair agrees to ≤ 1e-9 relative.

**The iteration counts on time-limited rows are NOT convergence evidence.** On a
row that hits the 45 s limit in both arms, fewer iterations means *slower per
iteration*, not *closer to optimal*. QPLIB_1451_rlt1's headline −20% is a 20%
throughput loss, not a 20% improvement. Reading it the other way would have been
the most attractive misreading available here, so it is called out explicitly.

**Where the cost comes from.** On QPLIB_0911, 51 repivots out of 1260 pivots (4%)
buy a 1.5% iteration reduction and cost 28% wall — ~31% more time per pivot. The
scan itself cannot cost that. The mechanism is that a different (larger-|α_rj|)
entering column is not a *sparser* one, so the bases that follow carry more LU
fill; the same tradeoff that H2 and the SuperLU headroom experiment already
showed is not ours to control. QPLIB_1619_rlt1 makes the point from the other
side: **one** repivot in 6121 pivots moves wall by 1.23x. That is a butterfly
effect on the pivot path, not a mechanism, and it is why no cell in this table
should be read as an effect size.

**Load caveat (§9).** `uptime` load average was 9.37 at launch. Every arm was
interleaved within each rep so the load hits both equally, and the qualitative
findings (median ≈ 1.00x, the 0.78x on QPLIB_0911, the QPLIB_1745_rlt1 status
regression) are robust to it. Individual cells between 0.95x and 1.05x are not
distinguishable from noise at that load and are not cited as results.

**Verdict.** H6 is falsified as a throughput fix. It survives its *entry*
criterion — the pass genuinely removes the QPLIB_0911 stall (46.2 s `iter_limit`
→ 4.4 s `optimal`, 10.5x, on the arm where the stall occurs) — but a benefit
confined to the instances that already stall is not a general fix under
CLAUDE.md §2, and criterion 3 shows the same mechanism *creates* a stall
elsewhere (QPLIB_1745_rlt1). `DISCOPT_LP_DUAL_HARRIS` must not graduate, and
under CLAUDE.md §3 (no dead flags) it is removed rather than shipped OFF, exactly
as H2's `pivot.rs` was.
