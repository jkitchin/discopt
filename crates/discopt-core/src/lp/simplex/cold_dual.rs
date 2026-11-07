//! #2212: the dual simplex from a sign-matched logical basis as the engine's first
//! cold attempt, ahead of the cold two-phase primal.
//!
//! **Why.** The cold primal's phase 1 has no term for box infeasibility of the
//! basics: on degenerate relaxations its `x_B` drifts out of the box, phase 2 starts
//! from a basis the final audit refuses, and the LP comes back `Numerical`
//! (#956 T2′: `Phase1EndBoxViolated` + `Phase1DriftDominates` + `LpAuditBoundsFail`).
//! Two callers each patched this for themselves: #2163's third `ScaledProblem` stage
//! in the MILP driver and #2202's `node_lp_slack_retry` in the spatial kernel. Both
//! re-solve with the dual simplex from a logical basis, the way HiGHS starts. This
//! module puts the same start under every cold solve instead.
//!
//! **Evidence** (`docs/dev/data/2212-cold-path/`). Over the 19 captured LP fixtures
//! the cascade here decides 18 (all but the tolerance-dependent `cstr`) against
//! the cold primal's 12, with no wrong verdict. On 148 in-repo root relaxations it
//! decides 126 against 124 and is 31 % faster on the 92 both decide.
//!
//! **The cascade**, each step accepted only with a certificate checked here on the
//! data the caller passed:
//!
//! 1. the LP's **own** logical columns as the basis (a zero-cost unit column per
//!    row), each nonbasic structural at the bound its cost sign selects. It returns
//!    a full basis in the caller's own columns, so a B&B child can warm-start from it;
//! 2. **#2163's augmented** form, a logical column `-e_i` per row fixed at
//!    `[b_i, b_i]` (so the logical basis always exists), when the LP has no logical
//!    column on some row. Its basis comes back restricted to the structural columns,
//!    so it can be short, and warm starts refuse a short basis and solve cold;
//! 3. otherwise `None`: the caller runs the cold primal unchanged.
//!
//! The start is dual-feasible exactly when every costed nonbasic column has a finite
//! bound on the side its cost selects (`y = 0`, so `d = c`). When one does not, the
//! step is refused rather than attempted, and the count is kept
//! (`LogicalDualColdRefused`): those LPs are the coverage gap the README records.
//!
//! **Soundness.** The warm dual loop runs no feasibility audit of its own (the #1287
//! retraction in `dual.rs`), so nothing it returns is trusted as-is. An `Optimal`
//! must pass [`point_passes_original_audit`] and an `Infeasible` must carry a ray
//! that [`farkas_ray_certifies_cols`] verifies. Anything else (a verdict that fails
//! its check, `Numerical`, `IterLimit`, `Unbounded`) is discarded and the primal
//! runs. The augmented rows are the original rows and the logical columns
//! contribute exactly `bᵀy` to the Farkas bound, so a ray of the augmented LP is a
//! ray of the original unchanged.
//!
//! **No recursion, no wasted primal.** The attempt calls the warm entry, whose
//! breakdown fallback is a cold solve, i.e. this function's caller. While an attempt
//! runs, [`in_attempt`] is set and `solve_lp_cols` returns a `Numerical` stub at
//! once instead of running the primal on the attempt's (possibly augmented) LP.
//! A failed attempt therefore costs the dual attempt plus one primal, on the
//! caller's LP.
//!
//! Gated by `DISCOPT_LP_LOGICAL_DUAL_COLD`, default **OFF** pending its CLAUDE.md §5
//! panels (row in `docs/dev/flag-retirement-audit.md`).

use super::primal::{farkas_ray_certifies_cols, point_passes_original_audit};
use super::{LpSolve, LpStatus, SimplexOptions, SparseCols};
use crate::lp::basis::{Basis, AT_LOWER, AT_UPPER, BASIC};

const INF: f64 = 1e20;

/// Whether the cascade is on: `DISCOPT_LP_LOGICAL_DUAL_COLD`, read once. Unset, `0`,
/// `false` or `off` is OFF (the default); `1`, `true` or `on` is ON. Anything else
/// is **refused**, not defaulted: an A/B arm whose spelling silently reads as the
/// default measures one arm twice (the `parse_stall_patience` lesson).
pub(crate) fn logical_dual_cold_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = test_override::get() {
        return v;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("DISCOPT_LP_LOGICAL_DUAL_COLD").unwrap_or_default();
        parse_logical_dual_cold(&raw).unwrap_or_else(|e| panic!("{e}"))
    })
}

fn parse_logical_dual_cold(raw: &str) -> Result<bool, String> {
    match raw.trim() {
        "" | "0" | "false" | "False" | "off" | "OFF" => Ok(false),
        "1" | "true" | "True" | "on" | "ON" => Ok(true),
        other => Err(format!(
            "DISCOPT_LP_LOGICAL_DUAL_COLD={other:?} is not a recognized value. \
             Use 1/true/on, or 0/false/off (the default)."
        )),
    }
}

std::thread_local! {
    static IN_ATTEMPT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SCOPE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SCOPE_FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run one caller-level cold solve (`f`) as a scope in which the cascade is tried
/// at most once **to failure**.
///
/// The solve entries retry on the exactly-unscaled matrix after a bound-losing
/// verdict (#649 in `solve_lp_warm_csc_inner`, #1935 in `solve_lp_cols_scaled`), and
/// each retry reaches `solve_lp_cols` again. The cascade's own warm solve already
/// runs that scaled-then-unscaled pair, so a second attempt on the same LP repeats
/// work that has just failed. Measured on `hda` in the first §5 panel: 13 of 21
/// attempts failed and every one was paid twice, leaving 1 node against 3 in the
/// 120 s budget (`docs/dev/data/2212-cold-path/README.md`). Inside a scope, once an
/// attempt has failed or been refused, later attempts are skipped and counted
/// (`LogicalDualColdSkippedRepeat`). A decided attempt returns, so nothing follows it.
/// Scopes nest; only the outermost one resets the state.
pub(crate) fn cold_scope<T>(f: impl FnOnce() -> T) -> T {
    struct Exit;
    impl Drop for Exit {
        fn drop(&mut self) {
            SCOPE_DEPTH.with(|d| d.set(d.get() - 1));
        }
    }
    if SCOPE_DEPTH.with(|d| d.get()) == 0 {
        SCOPE_FAILED.with(|c| c.set(false));
    }
    SCOPE_DEPTH.with(|d| d.set(d.get() + 1));
    let _exit = Exit;
    f()
}

fn scope_failed() -> bool {
    SCOPE_DEPTH.with(|d| d.get()) > 0 && SCOPE_FAILED.with(|c| c.get())
}

fn mark_scope_failed() {
    if SCOPE_DEPTH.with(|d| d.get()) > 0 {
        SCOPE_FAILED.with(|c| c.set(true));
    }
}

/// Pivot budget for one cascade attempt: `min(max_iter, 2(m+n) + 1000)`.
///
/// Measured over every attempt that decided an LP in the entry experiment and the
/// throughput panel (121 LPs): at most `0.65(m+n)` pivots, median `0.18(m+n)`, so
/// this is over 3x the largest. An attempt that has not finished inside it is not
/// going to, and the primal gets the rest of the caller's budget.
fn attempt_budget(m: usize, n: usize, opts: &SimplexOptions) -> SimplexOptions {
    let mut o = opts.clone();
    o.max_iter = opts.max_iter.min(
        2usize
            .saturating_mul(m.saturating_add(n))
            .saturating_add(1000),
    );
    o
}

/// Whether a logical-dual attempt is running on this thread (see the module doc).
pub(crate) fn in_attempt() -> bool {
    IN_ATTEMPT.with(|c| c.get())
}

/// Run `f` with [`in_attempt`] set; restored on exit, including on unwind.
fn attempting<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_ATTEMPT.with(|c| c.set(self.0));
        }
    }
    let _g = Restore(IN_ATTEMPT.with(|c| c.replace(true)));
    f()
}

/// The `Numerical` result `solve_lp_cols` returns inside an attempt instead of
/// running the primal: no point, no certificate, nothing a caller can use.
pub(crate) fn declined(n: usize) -> LpSolve {
    LpSolve {
        status: LpStatus::Numerical,
        x: vec![0.0; n],
        obj: f64::NAN,
        basis: Basis {
            col_status: vec![AT_LOWER; n],
            basic_vars: Vec::new(),
        },
        iters: 0,
        dual: Vec::new(),
        ray: Vec::new(),
    }
}

/// Bound side each nonbasic column takes so that, with `y = 0`, every reduced cost
/// `d_j = c_j` has the right sign. `None` when a column's selected side is open
/// (a free zero-cost column included: it would have to be basic).
fn sign_matched(j: usize, c: &[f64], l: &[f64], u: &[f64]) -> Option<i8> {
    if c[j] >= 0.0 && l[j] > -INF {
        Some(AT_LOWER)
    } else if c[j] <= 0.0 && u[j] < INF {
        Some(AT_UPPER)
    } else {
        None
    }
}

/// Step 1's start: one zero-cost unit column per row (the trailing one, where
/// slacks are appended), every other column at its sign-matched bound.
fn own_logical_start(
    sp: &SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
) -> Option<Basis> {
    let mut logical = vec![usize::MAX; m];
    for j in (0..n).rev() {
        let (rows, vals) = sp.col(j);
        if rows.len() == 1 && vals[0] != 0.0 && c[j] == 0.0 && logical[rows[0]] == usize::MAX {
            logical[rows[0]] = j;
        }
    }
    if logical.contains(&usize::MAX) {
        return None;
    }
    let mut col_status = vec![AT_LOWER; n];
    for &j in &logical {
        col_status[j] = BASIC;
    }
    for (j, st) in col_status.iter_mut().enumerate() {
        if *st != BASIC {
            *st = sign_matched(j, c, l, u)?;
        }
    }
    Some(Basis {
        col_status,
        basic_vars: logical,
    })
}

/// Accept `sol` only with a certificate checked on the caller's data.
#[allow(clippy::too_many_arguments)]
fn certified(
    sol: &LpSolve,
    sp: &SparseCols,
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> bool {
    match sol.status {
        LpStatus::Optimal => {
            sol.dual.len() == m && point_passes_original_audit(&sol.x, sp, m, n, b, l, u)
        }
        LpStatus::Infeasible => {
            sol.dual.len() == m && farkas_ray_certifies_cols(&sol.dual, sp, n, m, b, l, u)
        }
        _ => false,
    }
}

/// The cascade (module doc). `Some` is a certified verdict on the caller's LP;
/// `None` means run the cold primal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn logical_dual_cold(
    sp: &SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    opts: &SimplexOptions,
) -> Option<LpSolve> {
    if m == 0 || in_attempt() {
        return None;
    }
    if scope_failed() {
        crate::profile::incr(crate::profile::Ctr::LogicalDualColdSkippedRepeat);
        return None;
    }
    crate::profile::incr(crate::profile::Ctr::LogicalDualColdAttempts);
    let opts = &attempt_budget(m, n, opts);

    // Step 1: the LP's own logical columns.
    if let Some(start) = own_logical_start(sp, m, n, c, l, u) {
        let sol = attempting(|| {
            super::dual::solve_lp_warm_csc(sp.clone(), m, n, c, l, u, b, Some(&start), opts)
        });
        if certified(&sol, sp, m, n, b, l, u) {
            crate::profile::incr(crate::profile::Ctr::LogicalDualColdOwnDecided);
            return Some(sol);
        }
    }

    // Step 2: #2163's augmented form.
    let mut col_status = Vec::with_capacity(n + m);
    for j in 0..n {
        match sign_matched(j, c, l, u) {
            Some(s) => col_status.push(s),
            None => {
                crate::profile::incr(crate::profile::Ctr::LogicalDualColdRefused);
                mark_scope_failed();
                return None;
            }
        }
    }
    col_status.extend(std::iter::repeat_n(BASIC, m));
    let (cp, ri, vals) = sp.raw();
    let (mut col_ptr, mut row_idx, mut vv) = (cp.to_vec(), ri.to_vec(), vals.to_vec());
    for i in 0..m {
        row_idx.push(i);
        vv.push(-1.0);
        col_ptr.push(row_idx.len());
    }
    let aug = SparseCols::from_csc(col_ptr, row_idx, vv);
    let mut c_a = c.to_vec();
    c_a.resize(n + m, 0.0);
    let (mut l_a, mut u_a) = (l.to_vec(), u.to_vec());
    l_a.extend_from_slice(b);
    u_a.extend_from_slice(b);
    let zero = vec![0.0; m];
    let start = Basis {
        col_status,
        basic_vars: (n..n + m).collect(),
    };
    let w = attempting(|| {
        super::dual::solve_lp_warm_csc(aug, m, n + m, &c_a, &l_a, &u_a, &zero, Some(&start), opts)
    });
    let x: Vec<f64> = w.x.iter().take(n).copied().collect();
    let obj = if w.status == LpStatus::Optimal {
        x.iter().zip(c).map(|(xj, cj)| xj * cj).sum()
    } else {
        w.obj
    };
    let sol = LpSolve {
        status: w.status,
        x,
        obj,
        basis: Basis {
            col_status: w.basis.col_status.iter().take(n).copied().collect(),
            basic_vars: w
                .basis
                .basic_vars
                .iter()
                .copied()
                .filter(|&j| j < n)
                .collect(),
        },
        iters: w.iters,
        dual: w.dual,
        ray: w.ray,
    };
    if certified(&sol, sp, m, n, b, l, u) {
        crate::profile::incr(crate::profile::Ctr::LogicalDualColdAugDecided);
        return Some(sol);
    }
    crate::profile::incr(crate::profile::Ctr::LogicalDualColdFellBack);
    mark_scope_failed();
    None
}

#[cfg(test)]
pub(crate) mod test_override {
    use std::cell::Cell;
    std::thread_local! {
        static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    }
    pub(crate) fn get() -> Option<bool> {
        OVERRIDE.with(|c| c.get())
    }
    /// Forces [`super::logical_dual_cold_enabled`] on this thread until dropped.
    pub(crate) struct Scoped(Option<bool>);
    impl Scoped {
        pub(crate) fn new(v: bool) -> Self {
            Scoped(OVERRIDE.with(|c| c.replace(Some(v))))
        }
    }
    impl Drop for Scoped {
        fn drop(&mut self) {
            OVERRIDE.with(|c| c.set(self.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::simplex::dual::solve_lp_warm_csc;

    /// `{m, n, col_ptr, row_idx, vals, c, l, u, b}` of a captured `*_lp.json`.
    struct Lp {
        m: usize,
        n: usize,
        sp: SparseCols,
        c: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        b: Vec<f64>,
    }

    fn parse(json: &str) -> Lp {
        fn tail<'a>(json: &'a str, key: &str) -> &'a str {
            let k = format!("\"{key}\":");
            let at = json
                .find(&k)
                .unwrap_or_else(|| panic!("fixture lacks {key}"));
            json[at + k.len()..].trim_start()
        }
        let nums = |key: &str| -> Vec<f64> {
            let s = tail(json, key);
            let close = s.find(']').expect("array");
            s[1..close]
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().parse::<f64>().expect("number"))
                .collect()
        };
        let ints = |key: &str| -> Vec<usize> { nums(key).iter().map(|&v| v as usize).collect() };
        let scalar = |key: &str| -> usize {
            let s = tail(json, key);
            let end = s.find([',', '}']).expect("scalar");
            s[..end].trim().parse().expect("integer")
        };
        Lp {
            m: scalar("m"),
            n: scalar("n"),
            sp: SparseCols::from_csc(ints("col_ptr"), ints("row_idx"), nums("vals")),
            c: nums("c"),
            l: nums("l"),
            u: nums("u"),
            b: nums("b"),
        }
    }

    /// The engine's cold path on `lp`: the warm entry with no start, as a B&B node
    /// with nothing to inherit and the entry experiment's `cold` arm run it.
    fn cold(lp: &Lp, on: bool) -> (LpSolve, [u64; 5]) {
        let _flag = test_override::Scoped::new(on);
        let _guard = crate::profile::test_guard();
        crate::profile::reset();
        crate::profile::set_enabled(true);
        let sol = solve_lp_warm_csc(
            lp.sp.clone(),
            lp.m,
            lp.n,
            &lp.c,
            &lp.l,
            &lp.u,
            &lp.b,
            None,
            &SimplexOptions::default(),
        );
        use crate::profile::{counter, Ctr};
        let ctr = [
            counter(Ctr::LogicalDualColdAttempts),
            counter(Ctr::LogicalDualColdOwnDecided),
            counter(Ctr::LogicalDualColdAugDecided),
            counter(Ctr::LogicalDualColdFellBack),
            counter(Ctr::LogicalDualColdSkippedRepeat),
        ];
        crate::profile::set_enabled(false);
        (sol, ctr)
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * (1.0 + b.abs())
    }

    #[test]
    fn attempt_budget_is_capped_by_size_and_by_the_callers_cap() {
        let o = SimplexOptions::default();
        assert_eq!(attempt_budget(100, 300, &o).max_iter, 1800);
        let tight = SimplexOptions {
            max_iter: 500,
            ..SimplexOptions::default()
        };
        assert_eq!(attempt_budget(100, 300, &tight).max_iter, 500);
    }

    /// The scope state never leaks: outside any scope nothing is skipped, and the
    /// outermost scope resets a failure left by an earlier solve.
    #[test]
    fn scope_failure_does_not_outlive_its_solve() {
        assert!(!scope_failed());
        cold_scope(|| {
            mark_scope_failed();
            assert!(scope_failed());
            cold_scope(|| assert!(scope_failed(), "nested scopes share the state"));
        });
        assert!(!scope_failed(), "outside a scope nothing is skipped");
        cold_scope(|| assert!(!scope_failed(), "a new solve starts clean"));
    }

    #[test]
    fn parse_table_refuses_unknown_spellings() {
        assert_eq!(parse_logical_dual_cold(""), Ok(false));
        assert_eq!(parse_logical_dual_cold("0"), Ok(false));
        assert_eq!(parse_logical_dual_cold("on"), Ok(true));
        assert!(parse_logical_dual_cold("yes please").is_err());
    }

    /// The #2202 captured nous1 node LPs: the cold primal ends `Numerical` on both
    /// (phase-1 drift); the logical-basis dual from the LP's own slack columns
    /// decides both, matching HiGHS (presolve off, 1e-10: optimal 1.2060754585 /
    /// infeasible), with a certificate checked on the original data.
    #[test]
    fn nous1_node_lps_are_decided_from_their_own_logical_basis_2212() {
        let mut checked = 0;
        for (json, highs) in [
            (
                include_str!("../../bnb/testdata/nous1_inherited_basis_numerical_lp.json"),
                Some(1.2060754585257176),
            ),
            (
                include_str!(
                    "../../bnb/testdata/nous1_inherited_basis_numerical_infeasible_lp.json"
                ),
                None,
            ),
        ] {
            let lp = parse(json);
            let (off, ctr_off) = cold(&lp, false);
            assert_eq!(
                off.status,
                LpStatus::Numerical,
                "the captured defect (flag off)"
            );
            assert_eq!(ctr_off[0], 0, "flag off must not attempt");
            let (on, ctr_on) = cold(&lp, true);
            assert_eq!(ctr_on[1], 1, "decided by step 1 (own logical columns)");
            match highs {
                Some(obj) => {
                    assert_eq!(on.status, LpStatus::Optimal);
                    assert!(near(on.obj, obj), "obj {} vs HiGHS {obj}", on.obj);
                    assert_eq!(on.basis.basic_vars.len(), lp.m, "a full basis for children");
                }
                None => assert_eq!(on.status, LpStatus::Infeasible),
            }
            assert!(!in_attempt(), "guard restored");
            checked += 1;
        }
        assert_eq!(checked, 2);
    }

    /// #2163's ReLU-with-Anderson-cuts LPs have equality rows with no logical
    /// column, so step 1 does not apply; step 2's augmented form decides them.
    #[test]
    fn relu_cut_lps_are_decided_by_the_augmented_start_2212() {
        let mut checked = 0;
        for (json, obj) in [
            (
                include_str!("testdata/relu_anderson_cuts_numerical_lp.json"),
                -254.27360235760088,
            ),
            (
                include_str!("testdata/relu_anderson_cuts_false_optimal_lp.json"),
                22.368165413029715,
            ),
        ] {
            let lp = parse(json);
            let (off, _) = cold(&lp, false);
            assert_ne!(
                off.status,
                LpStatus::Optimal,
                "the captured defect (flag off)"
            );
            let (on, ctr) = cold(&lp, true);
            assert_eq!(ctr[2], 1, "decided by step 2 (augmented)");
            assert_eq!(on.status, LpStatus::Optimal);
            assert!(near(on.obj, obj), "obj {} vs HiGHS {obj}", on.obj);
            checked += 1;
        }
        assert_eq!(checked, 2);
    }

    /// The #2142 cstr big-M node LP stays undecided with the flag on: HiGHS's own
    /// verdict on it flips with its tolerance (infeasible at 1e-10, optimal
    /// 0.2397 at its defaults), and no start here decides it. Pinned so a change
    /// that "fixes" it is looked at, not assumed.
    #[test]
    fn cstr_node_lp_stays_undecided_with_the_flag_on_2212() {
        let lp = parse(include_str!("testdata/cstr_bigm_dual_rebuild_lp.json"));
        let (on, ctr) = cold(&lp, true);
        // Two cold solves reach `solve_lp_cols` here, the scaled one and the #649
        // retry on the unscaled matrix. The cascade is tried once; the retry's
        // attempt is skipped (`cold_scope`) instead of repeating work that just failed.
        assert_eq!(ctr[0], 1, "attempted once");
        assert_eq!(ctr[3], 1, "not certified, so the primal ran");
        assert!(ctr[4] >= 1, "the unscaled retry's attempt was skipped");
        assert!(
            !matches!(on.status, LpStatus::Optimal | LpStatus::Infeasible),
            "cstr decided as {:?}",
            on.status
        );
    }
}
