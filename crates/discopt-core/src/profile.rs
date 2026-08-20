//! Env-gated phase/pivot profiling for the MILP driver and simplex.
//!
//! A permanent, near-zero-overhead instrumentation facility for engine
//! performance work (issue #332). When `DISCOPT_PROFILE` is **unset**, every hook
//! is a single relaxed atomic-bool load plus a no-op (the timers never call
//! `Instant::now`, the counters never touch their atomics) — so the production
//! hot path is unaffected. When it is set, [`init_from_env`] flips a global flag
//! and the hooks accumulate per-phase wall time and per-category pivot counts that
//! [`dump`] prints (and resets) at the end of a solve.
//!
//! Thread-safe (relaxed atomics) so the rayon-parallel node loop is safe to
//! instrument. Call [`init_from_env`] once at the start of a solve, [`reset`] to
//! clear between solves, and [`dump`] to print.
#![allow(missing_docs)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable profiling iff `DISCOPT_PROFILE` is present in the environment. Cheap to
/// call repeatedly; the first call per process fixes the flag.
pub fn init_from_env() {
    ENABLED.store(
        std::env::var_os("DISCOPT_PROFILE").is_some(),
        Ordering::Relaxed,
    );
}

/// Whether profiling is currently active.
#[inline(always)]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

macro_rules! timed_phases {
    ($($name:ident),* $(,)?) => {
        #[derive(Clone, Copy)]
        pub enum Phase { $($name),* }
        const NP: usize = { let mut c = 0; $( let _ = stringify!($name); c += 1; )* c };
        static PNAMES: &[&str] = &[$(stringify!($name)),*];
        static PCOUNT: [AtomicU64; NP] = [$( { let _ = stringify!($name); AtomicU64::new(0) }),*];
        static PNANOS: [AtomicU64; NP] = [$( { let _ = stringify!($name); AtomicU64::new(0) }),*];
    };
}

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        #[derive(Clone, Copy)]
        pub enum Ctr { $($name),* }
        const NC: usize = { let mut c = 0; $( let _ = stringify!($name); c += 1; )* c };
        static CNAMES: &[&str] = &[$(stringify!($name)),*];
        static CVALS: [AtomicU64; NC] = [$( { let _ = stringify!($name); AtomicU64::new(0) }),*];
        /// Monotonic totals. [`dump`] zeroes `CVALS` (it is a "print since last
        /// dump" view) and is called from per-solve binding sites, so a caller
        /// measuring a whole run cannot read `CVALS` — on the Python spatial path
        /// every node LP wipes them. These accumulate in parallel and are cleared
        /// only by an explicit [`reset`].
        static CTOTALS: [AtomicU64; NC] = [$( { let _ = stringify!($name); AtomicU64::new(0) }),*];
    };
}

// Coarse MILP-driver phases and fine simplex-internal phases.
timed_phases!(
    RootCutLoop,
    RootSolve,
    SepCover,
    SepGomory,
    Augment,
    DiveRepair,
    NodeLpSolve,
    StrongBranch,
    SearchLoop,
    // Bound-reduction phases (cert:T0.3): FBBT/constraint-propagation at the
    // node (`Fbbt`) and at the root presolve (`NodeReduce`).
    Fbbt,
    NodeReduce,
    PriceBtran,
    PriceSweep,
    AlphaFtran,
    FtUpdate,
    Refactorize,
    // Warm dual-simplex node-LP phases (THRU-5): split the pure-LP node re-solve
    // (`solve_lp_warm_csc`) into the one-time basis factorize + dual-feasibility
    // verify (`DualPrepare`), the exact basic-value / reduced-cost recompute the
    // loop seeds and periodically refreshes from (`DualRecompute`), and the dual
    // pivot loop itself (`DualPivotLoop`). Instrumentation only; the existing
    // simplex phases above cover the cold/primal path but not this warm dual one,
    // which is the dominant per-node cost on the pure-LP node-bound path.
    DualPrepare,
    DualRecompute,
    DualPivotLoop,
    // #1008: the two halves of a SPARSE basis refactorization. `LuSymbolic` is
    // `SparseLuSymbolic::analyze` (AMD on the AᵀA pattern — a fill-reducing
    // ordering recomputed from scratch on every refactorization even though an
    // LP basis changes by one column per pivot); `LuNumeric` is `SparseLu::factor`
    // (the actual elimination). "Refactorization is 59.5% of wall" does not say
    // which half, and the two have completely different fixes.
    LuSymbolic,
    LuNumeric,
);

// Pivot categorization for the cold-primal simplex (degeneracy analysis).
counters!(
    // #1060 continuous-repair dive schedule. `DiveOffRoot` counts dives run away
    // from the root (the no-incumbent schedule; always 0 at the default stride 0),
    // `DiveOffRootHits` how many of those returned a repaired incumbent. The pair
    // is the audit: a schedule that fires and never repairs is pure overhead and
    // must not graduate, and without a counter that case reads exactly like a
    // schedule that never fired (CLAUDE.md §6).
    DiveOffRoot,
    DiveOffRootHits,
    Phase1Pivots,
    Phase2Pivots,
    DegeneratePivots,
    BoundFlips,
    BlandActivations,
    Refactorizations,
    // Numeric-focus iterative-refinement recovery (discopt#364), split by path so
    // the two very different triggers can be told apart when measuring.
    //
    // Primal (audit-failure driven): a drifted "Optimal" failed the feasibility
    // audit and triggered a fresh refined refactorization (Attempts); Rescues =
    // how many the refined point pulled back to a sound Optimal (the rest stayed
    // Numerical and fell back to cold, as before).
    RefinedRecoveryAttemptsPrimal,
    RefinedRecoveryRescuesPrimal,
    // Dual (growth-gated): the working factor's growth signal flagged possible
    // digit loss at the optimality gate, triggering a fresh refined recompute
    // (Attempts); Rescues = how many revealed a hidden infeasibility and so
    // *prevented* a wrong "Optimal" (returning None → cold solve). A non-rescue
    // Attempt still certified Optimal, but with the sharper x_B values.
    RefinedRecoveryAttemptsDual,
    RefinedRecoveryRescuesDual,
    // Dual-simplex anti-cycling (discopt#364): degenerate dual pivots (entering
    // reduced cost ≈ 0 → no dual-objective progress) that accumulate the stall
    // count, and how often that stall crossed the threshold and switched the dual
    // to Bland's smallest-index rule to break a potential cycle.
    DualDegeneratePivots,
    DualBlandActivations,
    // Primal EXPAND anti-degeneracy (discopt#364): degenerate blocking steps that
    // were bumped up to the guaranteed EXPAND minimum step (breaking the stall in
    // place instead of accumulating toward the Bland switch).
    ExpandMinSteps,
    // Warm dual-simplex reoptimizations (DualWarmSolves) and how many of them the
    // dual could not solve and fell back to a cold primal re-solve (DualColdFallbacks
    // — numerical breakdown / iteration cap on a *valid* warm start, i.e. the
    // "engine swap" the framework-LP-error-handling policy of #376 would try to
    // pre-empt by escalating in place). The ratio is the escalation headroom.
    DualWarmSolves,
    DualColdFallbacks,
    // Warm dual-simplex stall-guard trips (discopt F2): a warm re-solve that hit
    // the size-derived stall cap (K·(m+n)+C ≤ max_iter) and abandoned the warm
    // basis for a cold re-solve of the *same* LP. A subset of DualColdFallbacks
    // (those caused specifically by the stall guard, not a numerical breakdown).
    // > 0 on the pathological append-and-re-solve class (nvs01), = 0 on the
    // healthy majority — so the guard's action is auditable and its
    // bound-neutrality (same optimum, cold path) is measurable.
    DualStallTrips,
    // Degeneracy-stall detection in the warm dual loop (#1013). `DualStallTrips`
    // above fires only when the F2 *cap* is reached, so an LP that grinds below
    // the cap reads 0 there while 43% of its pivots take a zero-length step —
    // a detector that cannot see the stall it is named for (CLAUDE.md §6).
    // These measure the degeneracy directly:
    //   `DualDegenerateRunArms` — episodes where the run of consecutive
    //     degenerate pivots crossed the arming threshold (the stall signal);
    //   `DualDegenerateRunMax` — the longest such run in the process (a MAX, not
    //     a sum: written with `record_max`);
    //   `DualDegenerateStallBails` — warm solves abandoned for the trusted cold
    //     solve because a run reached `SimplexOptions::dual_stall_patience`.
    DualDegenerateRunArms,
    DualDegenerateRunMax,
    DualDegenerateStallBails,
    // THRU-5: split the primal refactorization trigger into its three causes so the
    // wide-McCormick refactor thrash can be attributed. RefacFtFail = the FT
    // (product-form) update returned Err (numerical bump breakdown → forced
    // refactor); RefacCap = the fixed refactorization interval (DISCOPT_LP_REFAC_INTERVAL); RefacWorkGate = the adaptive
    // work gate (accumulated update work exceeded factor nnz × mult).
    RefacFtFail,
    RefacCap,
    RefacWorkGate,
    // ENGINE-1 (#557): split the `RefacFtFail` (feral `update` returned Err) trigger
    // by its feral `RefactorCause`, so the decision fork "is the refactor
    // accuracy-necessary or is the bail over-conservative?" is measured, not
    // assumed. `Growth` = element-growth high-water exceeded `max_growth` (1e8) —
    // the candidate over-conservative bail (a large but bounded growth may still
    // give an accurate updated factor); `TinyPivot` = a bump/final diagonal pivot
    // at/below `zero_pivot_tol·‖U₀‖∞` — genuine numerical breakdown, refactor
    // necessary; `Singular` = the replacement column is rank-order dependent — a
    // structural (not tunable) breakdown. Instrumentation only; read off
    // `FeralLU::last_refactor()` at the same trigger site.
    RefacFtGrowth,
    RefacFtTinyPivot,
    RefacFtSingular,
    // #85 failure-triggered dense retry (density-aware LU route, #557): a node LP
    // that failed (Numerical/IterLimit) on the sparse route was re-solved once,
    // cold, with the route suppressed (Retries), and how many of those retries
    // reached a terminal certificate — Optimal/Infeasible/Unbounded — instead of
    // failing again (Rescues). The gap Retries − Rescues falls through to the
    // existing fallback chain exactly as before the retry existed.
    LpDenseRetries,
    LpDenseRetryRescues,
    // #956 follow-through: the TERMINAL verdict histogram of the cold primal
    // simplex, counted once per solve at its single exit point (`assemble`, after
    // the feasibility audit has had its say). This is the instrument that decides
    // whether an undecided node LP is an uncertified infeasibility, a drifted
    // optimum, or a factorization breakdown — the three have completely different
    // fixes and were previously indistinguishable from outside.
    LpVerdictOptimal,
    LpVerdictInfeasible,
    LpVerdictNumerical,
    LpVerdictIterLimit,
    LpVerdictUnbounded,
    // The load-bearing bucket: phase 1 left a real residual (so the LP looks
    // infeasible) but the Farkas ray did NOT certify, so the honest — and
    // useless — `Numerical` is returned instead of a proof. Since #927 the spatial
    // tree branches on that verdict, and `lp_spatial_bb` folds it into
    // `unresolved_lb`, which is why such a tree can never conclude `infeasible`.
    LpInfeasUncertified,
    // Why `assemble`'s audit downgraded an `Optimal`: a column outside its bounds
    // (refinement cannot help) vs a row residual (refinement can, and Rescues
    // above counts when it did).
    LpAuditBoundsFail,
    LpAuditRowsFail,
    // Why `farkas_ray_certifies_cols` rejected a ray: it needed a finite bound on a
    // column the ray selected and had none (`Open` — it recovers such bounds for
    // SLACK columns only, so a ray touching an unbounded structural/auxiliary
    // column is rejected outright), or the Neumaier-Shcherbina margin was not met
    // (`Margin`). Splitting these is what separates "the certificate is incomplete"
    // from "the LP is not actually infeasible".
    FarkasRejectOpen,
    FarkasRejectMargin,
    // The subset of `Margin` rejections that the pre-#1017 margin would have
    // ACCEPTED: `bᵀy − contrib` cleared the `1e-9·(1 + |by| + Σ|boxmax|)` floor but not
    // the rigorous bound on the rounding of the accumulations that produced it. Each
    // one is a certificate #1017 removed — the whole cost side of that change, and the
    // only way to see it in a corpus run without an A/B rebuild.
    FarkasRejectCancellation,
    // #1008: the same treatment for the PRIMAL unbounded ray, which had none. An
    // `Infeasible` verdict must clear `farkas_ray_certifies_cols`; an `Unbounded`
    // verdict was taken on faith from "the ratio test found no blocking row",
    // which is also what a silently broken ftran looks like. `Certified` counts
    // rays that pass; the three `Reject` counters say which condition failed —
    // `A d = 0` (the ray is not a recession direction at all), `cᵀd < 0` (the
    // objective does not improve along it), or box recession (it leaves the box).
    // On the captured QPLIB_2170 relaxation the ray had ONE nonzero, |A d| = 1.0
    // and cᵀd = 0: every basic α came back zero for a column that is not zero.
    UnboundedRayCertified,
    UnboundedRejectRowResidual,
    UnboundedRejectObjective,
    UnboundedRejectBox,
    // #1008 R1: the warm dual's unstable- (near-zero-) pivot exit, split by which
    // way it went. `Recoveries` counts in-place refactorize+recompute+re-select,
    // which keeps the monotone dual loop alive; `Bails` counts the hand-off to the
    // cold primal. The recovery is gated on `bank_deadline_duals`, which
    // `lp_bindings` sets to `deadline.is_some()` — so a caller passing no
    // `time_limit` gets the bail on the same LP and the same starting basis. These
    // two counters are what makes that a measurement rather than a reading of the
    // control flow.
    DualUnstablePivotRecoveries,
    DualUnstablePivotBails,
    // The WARM path's own terminal histogram. `solve_lp_warm_csc` is the entry the
    // Python spatial engine's per-node LPs come through, and it can return without
    // ever reaching the cold primal's `assemble`, so the `LpVerdict*` counters above
    // do not see it (measured: witness W1 runs 2495 nodes and registers ONE cold
    // verdict). Counted separately so the two engines can be told apart.
    WarmVerdictOptimal,
    WarmVerdictInfeasible,
    WarmVerdictNumerical,
    WarmVerdictIterLimit,
    WarmVerdictUnbounded,
    // How FAR outside its box the audit found the offending column, as a multiple
    // of the audit's own relative tolerance `1e-6·(1+|bound|)`. A near-miss (a few
    // ×) is a drift the basis could be cleaned up from; a gross one means the basis
    // is simply wrong and no tolerance change is admissible. Decides whether the
    // dominant `LpAuditBoundsFail` bucket is repairable at all.
    AuditExcursionLt10x,
    AuditExcursionLt1e3x,
    AuditExcursionLt1e6x,
    AuditExcursionGe1e6x,
    // #956 T2': is the column the audit rejected BASIC (so `x_B` is wrong — either
    // drift or a genuinely primal-infeasible basis) or nonbasic (which should be
    // impossible, since a nonbasic sits exactly at a bound by construction)? And
    // does recomputing `x_B` from a refinement-polished factorization pull it back
    // inside? `assemble` currently asserts it cannot ("a Harris ratio-test artefact,
    // not a solve-accuracy problem", measured for #364 on a different corpus) — these
    // counters re-test that claim on the spatial McCormick LPs.
    AuditBoundsOnBasic,
    AuditBoundsOnNonbasic,
    AuditBoundsRefineFixes,
    AuditBoundsRefinePersists,
    // Does the LP handed to the simplex have a CROSSED box (`l[j] > u[j]`)? Such an
    // LP is empty by inspection — no basis can satisfy it, phase 1 cannot repair it,
    // and no amount of refinement will pull the basic variable back inside a box
    // that has no inside. `assemble_node_lp` can produce one: it intersects each
    // auxiliary column's incoming bounds with the box-derived envelope range
    // (`l[a].max(alo)` / `u[a].min(ahi)`) with nothing checking that the result is
    // non-empty. Counted per solve, split by whether the solve then failed.
    LpCrossedBox,
    LpCrossedBoxAndFailed,
    // Is the basis primal-feasible when phase 1 hands off to phase 2? `assemble`'s
    // sibling comment asserts it is ("the LP is FEASIBLE and, because the cleanup
    // used only phase-1-feasible ratio-test pivots, the basis is primal-feasible"),
    // but phase 1 measures feasibility as `sum|artificials| <= 1e-6` — an ABSOLUTE
    // test on artificials, which says nothing about whether the structural basics
    // are inside their boxes. These counters test the claim.
    Phase1EndBoxOk,
    Phase1EndBoxViolated,
    // ... and how big the violation is, measured against EXPAND's own per-pivot
    // ceiling (1e-7). At-or-below that scale it is one pivot's sanctioned Harris
    // excursion; far above it, an accumulation across pivots or update drift.
    // #956 T3': nodes fathomed on a rigorous emptiness certificate rather than by
    // bound. Zero on an instance means the T3' arm never fired there, which is the
    // difference between "the change is neutral" and "the experiment measured
    // nothing" (CLAUDE.md §6).
    TreeCertInfeasPrunes,
    Phase1ViolLe1Expand,
    Phase1ViolLe100Expand,
    Phase1ViolGt100Expand,
    // ... and, for a violating solve, whether the ratio test's OWN (incrementally
    // maintained) x_B was already outside the box, or was clean and had merely
    // drifted away from what the basis actually holds. Different repairs.
    // #956 T2': mid-window exact re-derivations of x_B (DISCOPT_PRIMAL_XB_REFRESH).
    XbMidRefresh,
    Phase1IncrAlsoViolates,
    Phase1DriftDominates,
    Phase1ViolUnexplained,
    Phase1NoIncrXb,
    // #956 T2': the EXPAND-free re-solve of a cold primal that failed to decide,
    // and how many of those reached a terminal certificate instead of failing again.
    EntryDense,
    EntryCols,
    EntryColsWarm,
    ExpandResetArmed,
    ExpandResetRetries,
    ExpandResetRescues,
    // #1008 LU fill-in. Accumulated over every SPARSE `factorize_sparse` call:
    // the nonzero count of the basis handed in (`LuBasisNnz`), the nonzero count
    // of the `L`+`U` that came out (`LuFactorNnz`), and the call count
    // (`LuSparseFactorizations`). `LuFactorNnz / LuBasisNnz` is the fill ratio —
    // the quantity that sets the cost of ALL THREE of the LU hot spots at once
    // (refactorize, each FT update, and every ftran/btran), so it separates "we
    // refactorize too often" from "each factor is far denser than it should be".
    LuSparseFactorizations,
    LuBasisNnz,
    LuFactorNnz,
    // #1008 D3: the DUAL loop's refactorizations. Previously invisible — the
    // `Refactorizations`/`Refac*` counters above are incremented only by the
    // primal loop, so on an LP the dual solved outright the profile showed 100+
    // `LuSparseFactorizations` against zero refactorization events and the cost
    // could not be attributed to a trigger. `DualRefacCap` is the fixed-interval
    // cap (`DISCOPT_LP_REFAC_INTERVAL`), `DualRefacFtFail` a failed FT update.
    DualRefactorizations,
    DualRefacCap,
    DualRefacFtFail,
);

/// Add `n` to a counter (for accumulated quantities such as nonzero counts,
/// where [`incr`]'s +1 is not the datum). Same enable-gating as [`incr`].
#[inline(always)]
pub fn incr_by(c: Ctr, n: u64) {
    if enabled() {
        CVALS[c as usize].fetch_add(n, Ordering::Relaxed);
        CTOTALS[c as usize].fetch_add(n, Ordering::Relaxed);
    }
}

/// Keep the **maximum** `v` ever recorded on a counter, rather than a sum. For
/// extremal quantities (the longest degenerate pivot run, #1013) a total is
/// meaningless — summing run lengths across solves says nothing about whether any
/// single solve stalled. Same enable-gating as [`incr`].
#[inline(always)]
pub fn record_max(c: Ctr, v: u64) {
    if enabled() {
        CVALS[c as usize].fetch_max(v, Ordering::Relaxed);
        CTOTALS[c as usize].fetch_max(v, Ordering::Relaxed);
    }
}

#[inline(always)]
pub fn incr(c: Ctr) {
    if enabled() {
        CVALS[c as usize].fetch_add(1, Ordering::Relaxed);
        CTOTALS[c as usize].fetch_add(1, Ordering::Relaxed);
    }
}

#[inline(always)]
fn record(p: Phase, nanos: u64) {
    let i = p as usize;
    PCOUNT[i].fetch_add(1, Ordering::Relaxed);
    PNANOS[i].fetch_add(nanos, Ordering::Relaxed);
}

/// RAII timer that accumulates elapsed time to `phase` on drop. A no-op (no
/// `Instant::now`) when profiling is disabled.
pub struct Timer {
    phase: Phase,
    start: Option<Instant>,
}

impl Timer {
    #[inline(always)]
    pub fn new(phase: Phase) -> Self {
        Self {
            phase,
            start: if enabled() {
                Some(Instant::now())
            } else {
                None
            },
        }
    }
}

impl Drop for Timer {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(s) = self.start {
            record(self.phase, s.elapsed().as_nanos() as u64);
        }
    }
}

/// Reset the per-`dump` accumulators (e.g. between solves).
///
/// Deliberately leaves `CTOTALS` alone. `milp_driver` calls this at the start of
/// every MILP sub-solve, so clearing the run totals here silently zeroed them
/// mid-run: a counter incremented before the last sub-solve read back as 0, and
/// "0" reads as "this path never executed". That cost a wrong conclusion twice
/// (#956 T1 and again in the T6 fired-check), which is why the totals now survive
/// both [`dump`] and this. Use [`reset_totals`] to clear them on purpose.
pub fn reset() {
    for a in PCOUNT.iter().chain(PNANOS.iter()).chain(CVALS.iter()) {
        a.store(0, Ordering::Relaxed);
    }
}

/// Clear the run totals as well — the deliberate "start a new measurement" reset,
/// called only by an instrument that owns the whole process.
pub fn reset_totals() {
    reset();
    for a in CTOTALS.iter() {
        a.store(0, Ordering::Relaxed);
    }
}

/// Current value of a counter. Mainly for tests/observability: lets a caller
/// read a counter (e.g. [`Ctr::DualStallTrips`]) without going through [`dump`]
/// (which prints to stderr and resets). Reads 0 when profiling was never enabled,
/// since [`incr`] only accumulates while [`enabled`] holds.
#[inline]
pub fn counter(c: Ctr) -> u64 {
    CVALS[c as usize].load(Ordering::Relaxed)
}

/// Every counter as `(name, value)`, without printing or resetting.
///
/// [`dump`] writes to stderr and zeroes the accumulators, which makes it unusable
/// as a measurement instrument for a caller that wants the numbers back (the #956
/// follow-through needs the verdict histogram in Python). This returns them.
pub fn counter_snapshot() -> Vec<(&'static str, u64)> {
    (0..NC)
        .map(|i| (CNAMES[i], CTOTALS[i].load(Ordering::Relaxed)))
        .collect()
}

/// Force the profiling flag on/off. Test-only: production toggles it exactly once
/// via [`init_from_env`]. Exposed so a Rust test can deterministically observe a
/// [`counter`] without setting the `DISCOPT_PROFILE` env var process-wide.
#[cfg(test)]
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Process-wide lock every test that calls [`set_enabled`] or reads a [`counter`]
/// must hold.
///
/// `ENABLED` and the counter arrays are global, and `cargo test` runs the suite
/// on a thread pool: without this, one test's closing `set_enabled(false)` lands
/// in the middle of another's measurement and the second reads 0. That is exactly
/// the CLAUDE.md #6 failure mode — an instrument that silently measures nothing —
/// and it is why it is a lock rather than a retry.
#[cfg(test)]
pub static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`TEST_LOCK`], ignoring poisoning (a failed test elsewhere must not
/// cascade into unrelated failures here).
#[cfg(test)]
pub fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Print the accumulated table to stderr when profiling is enabled, then reset.
pub fn dump() {
    if !enabled() {
        return;
    }
    eprintln!("--- MILP phase profile (count, total ms) ---");
    for i in 0..NP {
        let c = PCOUNT[i].swap(0, Ordering::Relaxed);
        let ns = PNANOS[i].swap(0, Ordering::Relaxed);
        if c == 0 {
            continue;
        }
        eprintln!(
            "  {:<14} {:>9} calls {:>10.2} ms",
            PNAMES[i],
            c,
            ns as f64 / 1e6
        );
    }
    eprintln!("--- simplex pivot categorization ---");
    for i in 0..NC {
        let v = CVALS[i].swap(0, Ordering::Relaxed);
        eprintln!("  {:<18} {:>10}", CNAMES[i], v);
    }
}
