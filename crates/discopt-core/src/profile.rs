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
);

// Pivot categorization for the cold-primal simplex (degeneracy analysis).
counters!(
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
    // THRU-5: split the primal refactorization trigger into its three causes so the
    // wide-McCormick refactor thrash can be attributed. RefacFtFail = the FT
    // (product-form) update returned Err (numerical bump breakdown → forced
    // refactor); RefacCap = the hard 48-update cap; RefacWorkGate = the adaptive
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
);

#[inline(always)]
pub fn incr(c: Ctr) {
    if enabled() {
        CVALS[c as usize].fetch_add(1, Ordering::Relaxed);
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

/// Reset all accumulators (e.g. between solves).
pub fn reset() {
    for a in PCOUNT.iter().chain(PNANOS.iter()).chain(CVALS.iter()) {
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

/// Force the profiling flag on/off. Test-only: production toggles it exactly once
/// via [`init_from_env`]. Exposed so a Rust test can deterministically observe a
/// [`counter`] without setting the `DISCOPT_PROFILE` env var process-wide.
#[cfg(test)]
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
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
