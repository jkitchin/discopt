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
    PriceBtran,
    PriceSweep,
    AlphaFtran,
    FtUpdate,
    Refactorize,
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
