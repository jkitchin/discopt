//! One LP matrix equilibrated once and solved many times, with every `Optimal`
//! verified against the ORIGINAL data before it leaves this module (#1945).
//!
//! The auto-scaling entry points ([`super::solve_lp`], [`super::solve_lp_cols_scaled`],
//! [`super::solve_lp_warm`], [`super::solve_lp_warm_csc`], [`super::solve_lp_batch`])
//! equilibrate internally and, since #1942, re-audit a scaled `Optimal` against the
//! original LP: the simplex's own final audit runs on the equilibrated system with a
//! floor of `1e-6` in SCALED units, so a column scaled by `~1e6` admits `~0.1` of
//! violation in original units. A caller that scales the matrix itself and calls the
//! solve core on scaled data bypassed that check, because the core never sees the
//! original units. The MILP driver did exactly that (it equilibrates once per batch
//! and shares the factors across the batch's node LPs), and on real instances it
//! returned `Optimal` node LPs whose unscaled point missed a bound or row by up to
//! `1e-3` relative, some of them integral vertices it adopted as incumbents (#1945:
//! 314 of 18,930 driver `Optimal`s over the in-repo corpus on the AMP route).
//!
//! [`ScaledProblem`] is the way to scale once and solve many times without losing
//! that guarantee. It owns the original matrix, the [`Scaling`] and the scaled
//! matrix; its solves take ORIGINAL bounds and return results in SOLVE coordinates
//! (scaled when the matrix is scaled -- exactly what the caller's existing
//! unscale-after-solve code expects), and they all pass through [`ScaledProblem::verify`]:
//!
//! * an `Optimal` whose unscaled point fails the original-space audit is re-solved on
//!   the original data (#649's recovery, warm from the same start when there is one);
//! * a retry that decides the LP is mapped back into solve coordinates -- exactly,
//!   since every factor is a power of two -- so a Farkas ray or row duals verify and
//!   unscale exactly as a scaled solve's would;
//! * a retry that decides nothing turns the verdict into `Numerical` with no
//!   certificate, which callers treat as an undecided LP (branch, never fathom).
//!
//! The check can only refuse or re-derive an `Optimal`, never create one. Outside
//! `lp/simplex` no code builds a [`Scaling`] or scales a matrix; the structural test
//! below fails if one does, so a new caller cannot reopen the bypass silently.

use super::dual::PreparedDual;
use super::linsolve::{node_feral_lu, LinearSolver};
use super::primal::{
    downgrade_false_optimal, farkas_ray_certifies_cols, point_fails_original_audit,
    point_passes_original_audit, retry_decided,
};
use super::scaling::Scaling;
use super::sparse::SparseCols;
use super::{solve_lp_cols, solve_lp_warm_scaled_csc, LpSolve, LpStatus, SimplexOptions};
use crate::lp::basis::{Basis, AT_LOWER, AT_UPPER, BASIC};
use crate::lp::crossover::LpView;

/// `DISCOPT_LP_DUAL_COLD` (D2 engine item W5(a), default OFF): solve a cold LP
/// with the dual simplex from the all-logical basis (HiGHS's cold start, with its
/// auxiliary-LP dual phase 1 when the logical basis is not dual feasible) before
/// the two-phase primal, and export a FULL basis (basic logicals exchanged out), so
/// the next warm start is not refused for its shape. Read once.
fn dual_cold_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("DISCOPT_LP_DUAL_COLD").unwrap_or_default();
        parse_dual_cold(&raw)
    })
}

/// The parse table behind [`dual_cold_enabled`]; garbage is refused.
fn parse_dual_cold(raw: &str) -> bool {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "no" => false,
        "1" | "true" | "yes" => true,
        other => panic!(
            "DISCOPT_LP_DUAL_COLD={other:?} is not recognized (expected 1/true/yes or \
             0/false/no). Refusing rather than silently picking an arm."
        ),
    }
}

/// HiGHS's dual phase-1 box for a column with bounds `[l, u]` (`HEkk.cpp` 2665):
/// free -> [-1000, 1000], lower-only -> [0, 1], upper-only -> [-1, 0], boxed or
/// fixed -> [0, 0]. Every box holds 0, so with a zero rhs the phase-1 LP is
/// feasible at `x = 0` and bounded.
fn phase1_box(l: f64, u: f64) -> (f64, f64) {
    const INF: f64 = 1e20;
    match (l > -INF, u < INF) {
        (false, false) => (-1000.0, 1000.0),
        (true, false) => (0.0, 1.0),
        (false, true) => (-1.0, 0.0),
        (true, true) => (0.0, 0.0),
    }
}

/// Exchange every basic logical of an optimal basis of the logical-augmented LP
/// (`n` structural columns, then the `m` logicals `-e_i`) for a nonbasic
/// structural column, so the exported basis is a full basis of the original LP.
///
/// A logical is fixed (`[b_i, b_i]`), and the entering structural keeps the value
/// of its nonbasic bound, so the point is unchanged: only the labels move. A
/// logical whose row holds its own nonbasic structural singleton is relabelled to
/// it directly (the two columns are parallel); the rest take the largest-|alpha|
/// nonbasic structural of their row of `B^-1 A`, one LU update each. Returns `None`
/// when a row admits no entering column (a redundant row) or the LU fails; the
/// caller then exports no full basis.
fn exchange_out_logicals(aug: &SparseCols, m: usize, n: usize, basis: &Basis) -> Option<Basis> {
    const PIVOT_MIN: f64 = 1e-7;
    let mut basic: Vec<usize> = basis.basic_vars.clone();
    if basic.len() != m {
        return None;
    }
    let mut stat: Vec<i8> = basis.col_status.clone();
    if !basic.iter().any(|&j| j >= n) {
        return Some(Basis {
            col_status: stat[..n].to_vec(),
            basic_vars: basic,
        });
    }
    // Relabel: row i's logical -> a nonbasic structural singleton of row i.
    let mut single: Vec<Option<(usize, f64)>> = vec![None; m];
    for j in 0..n {
        if stat[j] == BASIC {
            continue;
        }
        let (rows, vals) = aug.col(j);
        if rows.len() == 1 {
            let i = rows[0];
            if single[i].is_none_or(|(_, v)| vals[0].abs() > v.abs()) {
                single[i] = Some((j, vals[0]));
            }
        }
    }
    for slot in 0..m {
        let jl = basic[slot];
        if jl >= n {
            if let Some((j, _)) = single[jl - n] {
                if stat[j] != BASIC {
                    basic[slot] = j;
                    stat[j] = BASIC;
                    stat[jl] = AT_LOWER;
                    crate::profile::incr(crate::profile::Ctr::DualColdRelabels);
                }
            }
        }
    }
    if !basic.iter().any(|&j| j >= n) {
        return Some(Basis {
            col_status: stat[..n].to_vec(),
            basic_vars: basic,
        });
    }
    let col_of = |j: usize| -> Vec<(usize, f64)> {
        let (rows, vals) = aug.col(j);
        rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
    };
    let mut lu = node_feral_lu();
    let cols: Vec<Vec<(usize, f64)>> = basic.iter().map(|&j| col_of(j)).collect();
    lu.factorize_sparse(m, &cols).ok()?;
    for slot in 0..m {
        if basic[slot] < n {
            continue;
        }
        let mut rho = vec![0.0; m];
        rho[slot] = 1.0;
        lu.btran(&mut rho).ok()?;
        let mut best: Option<(usize, f64)> = None;
        for j in 0..n {
            if stat[j] == BASIC {
                continue;
            }
            let a = aug.dot(j, &rho);
            if a.abs() > PIVOT_MIN && best.is_none_or(|(_, v)| a.abs() > v.abs()) {
                best = Some((j, a));
            }
        }
        let (q, _) = best?;
        let mut dense = vec![0.0; m];
        for (r, v) in col_of(q) {
            dense[r] = v;
        }
        lu.update(slot, &dense).ok()?;
        stat[basic[slot]] = AT_LOWER;
        basic[slot] = q;
        stat[q] = BASIC;
        crate::profile::incr(crate::profile::Ctr::DualColdExchanges);
    }
    Some(Basis {
        col_status: stat[..n].to_vec(),
        basic_vars: basic,
    })
}

/// An LP matrix with its objective and rhs, equilibrated once when ill-conditioned.
/// See the module documentation.
pub struct ScaledProblem {
    m: usize,
    n: usize,
    scaling: Option<Scaling>,
    original: SparseCols,
    /// The equilibrated matrix; `None` when the matrix is well-conditioned (solves
    /// then run on `original`, and the simplex's own audit is in original units).
    scaled: Option<SparseCols>,
    c: Vec<f64>,
    b: Vec<f64>,
    c_s: Vec<f64>,
    b_s: Vec<f64>,
}

impl ScaledProblem {
    /// Equilibrate `original` (an `m × n` CSC) when its dynamic range warrants it
    /// ([`Scaling::from_sparse`]); otherwise keep it as is.
    pub fn new(original: SparseCols, m: usize, n: usize, c: &[f64], b: &[f64]) -> Self {
        let scaling = Scaling::from_sparse(&original, m, n);
        let (scaled, c_s, b_s) = match &scaling {
            Some(s) => {
                let mut a = original.clone();
                s.scale_cols(&mut a);
                (Some(a), s.scale_c(c), s.scale_b(b))
            }
            None => (None, Vec::new(), Vec::new()),
        };
        ScaledProblem {
            m,
            n,
            scaling,
            original,
            scaled,
            c: c.to_vec(),
            b: b.to_vec(),
            c_s,
            b_s,
        }
    }

    /// The equilibration, or `None` when solves run on the original data.
    pub fn scaling(&self) -> Option<&Scaling> {
        self.scaling.as_ref()
    }

    /// The matrix solves run on (scaled when [`Self::scaling`] is `Some`).
    pub fn solve_matrix(&self) -> &SparseCols {
        self.scaled.as_ref().unwrap_or(&self.original)
    }

    /// The original, unscaled matrix.
    pub fn original_matrix(&self) -> &SparseCols {
        &self.original
    }

    /// Solve-space objective (scaled when [`Self::scaling`] is `Some`).
    pub fn solve_c(&self) -> &[f64] {
        if self.scaling.is_some() {
            &self.c_s
        } else {
            &self.c
        }
    }

    /// Solve-space rhs (scaled when [`Self::scaling`] is `Some`).
    pub fn solve_b(&self) -> &[f64] {
        if self.scaling.is_some() {
            &self.b_s
        } else {
            &self.b
        }
    }

    /// Solve-space copies of ORIGINAL bounds `l`, `u` (infinite sides preserved).
    pub fn solve_bounds(&self, l: &[f64], u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match &self.scaling {
            Some(s) => (s.scale_lower(l), s.scale_upper(u)),
            None => (l.to_vec(), u.to_vec()),
        }
    }

    /// Warm re-solve from `start` over ORIGINAL bounds `l`, `u` (dual simplex, cold
    /// fallback on any difficulty). The result is in solve coordinates and verified
    /// ([`Self::verify`]).
    pub fn solve_warm(
        &self,
        l: &[f64],
        u: &[f64],
        start: &Basis,
        opts: &SimplexOptions,
    ) -> LpSolve {
        // A short basis (a cold primal solve's, with zero artificials dropped) is
        // refused by the warm dual for its shape and would cold-solve through the
        // primal; under W5(a) it cold-solves through the dual instead.
        if dual_cold_enabled() && start.basic_vars.len() != self.m {
            crate::profile::incr(crate::profile::Ctr::DualColdShortStart);
            return self.solve_cold(l, u, opts);
        }
        let (sl, su) = self.solve_bounds(l, u);
        let view = self.view(self.solve_c(), &sl, &su);
        let sol = solve_lp_warm_scaled_csc(&view, self.solve_b(), start, opts, self.solve_matrix());
        let sol = self.recover_numerical(sol, l, u, Some(start), opts);
        self.verify(sol, l, u, Some(start), opts)
    }

    /// Cold solve over ORIGINAL bounds `l`, `u`. The result is in solve coordinates
    /// and verified ([`Self::verify`]).
    pub fn solve_cold(&self, l: &[f64], u: &[f64], opts: &SimplexOptions) -> LpSolve {
        if dual_cold_enabled() {
            if let Some(sol) = self.dual_cold(l, u, opts) {
                return self.verify(sol, l, u, None, opts);
            }
        }
        let (sl, su) = self.solve_bounds(l, u);
        let sol = solve_lp_cols(
            self.solve_matrix().clone(),
            self.m,
            self.n,
            self.solve_c(),
            &sl,
            &su,
            self.solve_b(),
            opts,
        );
        let sol = self.recover_numerical(sol, l, u, None, opts);
        self.verify(sol, l, u, None, opts)
    }

    /// #2163: the #649 recovery the other auto-scaling entry points already apply
    /// ([`super::solve_lp_warm_csc`] and friends), which this type did not. Pow2
    /// geometric-mean equilibration is a heuristic, and on some LPs it is
    /// counterproductive: on trained ReLU big-M MILPs (hidden weights from ~1e-8 to
    /// ~10) it sets column factors up to 2^16. That squeezes the scaled box widths to a
    /// median of ~4e-3 and a minimum of ~8e-6, and the scaled solve ends `Numerical`:
    /// a phase-1 residual whose Farkas ray does not verify, or a basic column outside
    /// its box at the audit. The UNSCALED solve of the same LP decides it, matching
    /// HiGHS. On network A of #2163 that held for 203 of the 207 node LPs the driver
    /// handed back undecided (197 optimal, 6 infeasible, as HiGHS says), and those
    /// undecided nodes are what drained the tree into `unresolved_floor`.
    ///
    /// So when a scaled solve returns `Numerical`, re-solve on the original data
    /// (warm from the same start when there is one, else cold) and, if that decides
    /// the LP, return it mapped into solve coordinates exactly as
    /// [`Self::verify`]'s retry is. Sound: the unscaled solve carries its own
    /// feasibility audit in original units, so a returned `Optimal` / `Infeasible` /
    /// `Unbounded` is genuine (and a returned `Optimal` still goes through
    /// [`Self::verify`], a no-op on an unscaled-space point that passed it); a retry
    /// that decides nothing leaves the original `Numerical` untouched. It can only
    /// recover a verdict scaling lost, never change one scaling reached.
    ///
    /// `IterLimit` is deliberately NOT retried: it is a budget verdict (strong
    /// branching and the driver's pivot-bounded warm root solve impose it on
    /// purpose), so re-running the same budget on the other matrix would double a
    /// cost the caller chose to cap.
    fn retry_numerical_unscaled(
        &self,
        sol: LpSolve,
        l: &[f64],
        u: &[f64],
        start: Option<&Basis>,
        opts: &SimplexOptions,
    ) -> LpSolve {
        let Some(s) = &self.scaling else {
            return sol;
        };
        if sol.status != LpStatus::Numerical {
            return sol;
        }
        crate::profile::incr(crate::profile::Ctr::ScaledNumericalUnscaledRetry);
        let view = self.view(&self.c, l, u);
        let mut alt = match start {
            Some(basis) => solve_lp_warm_scaled_csc(&view, &self.b, basis, opts, &self.original),
            None => solve_lp_cols(
                self.original.clone(),
                self.m,
                self.n,
                &self.c,
                l,
                u,
                &self.b,
                opts,
            ),
        };
        if !self.decided_on_original(&alt, l, u) {
            return sol;
        }
        crate::profile::incr(crate::profile::Ctr::ScaledNumericalUnscaledRetryDecided);
        // Into solve coordinates, as every other result of this problem is. Exact:
        // power-of-two factors.
        s.scale_x(&mut alt.x);
        s.scale_dual(&mut alt.dual);
        s.scale_ray(&mut alt.ray);
        alt
    }

    /// Whether a re-solve `alt` on the ORIGINAL data (original coordinates) decided
    /// the LP: `Infeasible` / `Unbounded` (their certificates are checked by the
    /// caller, as for any solve), or an `Optimal` whose point passes the original-LP
    /// audit. The audit is needed because a warm re-solve runs the dual
    /// re-optimize, which audits nothing (the #1287 retraction in `dual.rs`); a
    /// cold re-solve's own audit already passed, so for it this is a re-check.
    fn decided_on_original(&self, alt: &LpSolve, l: &[f64], u: &[f64]) -> bool {
        match alt.status {
            LpStatus::Optimal => {
                point_passes_original_audit(&alt.x, &self.original, self.m, self.n, &self.b, l, u)
            }
            LpStatus::Infeasible | LpStatus::Unbounded => true,
            LpStatus::Numerical | LpStatus::IterLimit => false,
        }
    }

    /// W5(a) (`DISCOPT_LP_DUAL_COLD`): the cold solve through the dual simplex, in
    /// SOLVE coordinates, from the all-logical basis of `A x - s = 0`, `s` fixed at
    /// `b` (the #2163 augmentation, [`Self::logical_basis_dual`]).
    ///
    /// - Structural columns sit at the bound their cost sign wants. When one cannot
    ///   (a one-sided or free column with the wrong cost sign), HiGHS's dual phase 1
    ///   runs first: the same LP over [`phase1_box`] boxes with `s` fixed at 0, which
    ///   is feasible at 0 and bounded; its optimal basis is the phase-2 start.
    /// - Phase 2 goes through [`PreparedDual`], which re-verifies dual feasibility
    ///   on the true bounds (flipping boxed columns); a start it refuses means the
    ///   LP may be dual infeasible, and this returns `None` (the primal decides).
    /// - Accepted only with a certificate checked HERE (the warm dual audits
    ///   nothing, #1287): an `Optimal` whose unscaled point passes the original-LP
    ///   audit, or an `Infeasible` whose ray verifies as a Farkas certificate.
    ///   Anything else is `None`.
    /// - Basic logicals are exchanged out ([`exchange_out_logicals`]) so the
    ///   exported basis is full; when that fails the basis is the structural part
    ///   (short), as [`Self::logical_basis_dual`] exports.
    fn dual_cold(&self, l: &[f64], u: &[f64], opts: &SimplexOptions) -> Option<LpSolve> {
        const INF: f64 = 1e20;
        let (m, n) = (self.m, self.n);
        let (sl, su) = self.solve_bounds(l, u);
        let (sc, sb) = (self.solve_c(), self.solve_b());
        crate::profile::incr(crate::profile::Ctr::DualColdAttempts);
        let _t = crate::profile::Timer::new(crate::profile::Phase::DualCold);
        let (cp, ri, vals) = self.solve_matrix().raw();
        let mut col_ptr = cp.to_vec();
        let mut row_idx = ri.to_vec();
        let mut vv = vals.to_vec();
        for i in 0..m {
            row_idx.push(i);
            vv.push(-1.0);
            col_ptr.push(row_idx.len());
        }
        let aug = SparseCols::from_csc(col_ptr, row_idx, vv);
        let mut c_a = sc.to_vec();
        c_a.resize(n + m, 0.0);
        let mut l_a = sl.clone();
        l_a.extend_from_slice(sb);
        let mut u_a = su.clone();
        u_a.extend_from_slice(sb);
        let b_a = vec![0.0; m];
        let mut wopts = opts.clone();
        wopts.warm_flip_boxed = true;
        // #350: a dual solve that stalls on an ill-conditioned LP must hand over to
        // the cold primal, not grind to the deadline. Same cap as the driver's
        // slack-basis dual start (`warm_root_opts`); an `IterLimit` is `None` below.
        wopts.max_iter = wopts.max_iter.min((8 * (m + n)).max(512));
        let mut iters = 0usize;
        // Logical basis, structurals at their cost-preferred bound.
        let mut need_phase1 = false;
        let mut col_status: Vec<_> = (0..n)
            .map(|j| {
                let (lo, hi, c) = (sl[j] > -INF, su[j] < INF, sc[j]);
                if c >= 0.0 && lo {
                    AT_LOWER
                } else if c <= 0.0 && hi {
                    AT_UPPER
                } else {
                    need_phase1 = true;
                    if lo {
                        AT_LOWER
                    } else if hi {
                        AT_UPPER
                    } else {
                        AT_LOWER
                    }
                }
            })
            .collect();
        col_status.extend(std::iter::repeat_n(BASIC, m));
        let mut start = Basis {
            col_status,
            basic_vars: (n..n + m).collect(),
        };
        if need_phase1 {
            crate::profile::incr(crate::profile::Ctr::DualColdPhase1);
            let mut l1 = Vec::with_capacity(n + m);
            let mut u1 = Vec::with_capacity(n + m);
            for j in 0..n {
                let (a, b) = phase1_box(sl[j], su[j]);
                l1.push(a);
                u1.push(b);
            }
            l1.resize(n + m, 0.0);
            u1.resize(n + m, 0.0);
            let mut st1 = start.clone();
            for j in 0..n {
                st1.col_status[j] = if sc[j] >= 0.0 { AT_LOWER } else { AT_UPPER };
            }
            let p1 = PreparedDual::prepare_cols(&aug, m, n + m, &c_a, &l1, &u1, &st1, &wopts)?;
            let w1 = p1.reoptimize(&l1, &u1, &b_a, &wopts);
            iters += w1.iters;
            if w1.status != LpStatus::Optimal || w1.basis.basic_vars.len() != m {
                return None;
            }
            // Phase-1 statuses on the true bounds: boxed keep theirs (prepare flips
            // a dual-infeasible one), one-sided take their finite bound, free sit at 0.
            let mut st2 = w1.basis.clone();
            for j in 0..n {
                if st2.col_status[j] == BASIC {
                    continue;
                }
                let (lo, hi) = (sl[j] > -INF, su[j] < INF);
                st2.col_status[j] = match (lo, hi) {
                    (true, true) => st2.col_status[j],
                    (false, true) => AT_UPPER,
                    _ => AT_LOWER,
                };
            }
            for j in n..n + m {
                if st2.col_status[j] != BASIC {
                    st2.col_status[j] = AT_LOWER;
                }
            }
            start = st2;
        }
        let Some(p2) = PreparedDual::prepare_cols(&aug, m, n + m, &c_a, &l_a, &u_a, &start, &wopts)
        else {
            crate::profile::incr(crate::profile::Ctr::DualColdPrepRefused);
            return None;
        };
        let w = p2.reoptimize(&l_a, &u_a, &b_a, &wopts);
        iters += w.iters;
        let x: Vec<f64> = w.x.iter().take(n).copied().collect();
        let accepted = match w.status {
            LpStatus::Optimal => {
                let mut xo = x.clone();
                if let Some(s) = &self.scaling {
                    s.unscale_x(&mut xo);
                }
                w.dual.len() == m
                    && point_passes_original_audit(&xo, &self.original, m, n, &self.b, l, u)
            }
            LpStatus::Infeasible => {
                farkas_ray_certifies_cols(&w.dual, self.solve_matrix(), n, m, sb, &sl, &su)
            }
            _ => false,
        };
        if !accepted {
            crate::profile::incr(crate::profile::Ctr::DualColdUndecided);
            return None;
        }
        crate::profile::incr(crate::profile::Ctr::DualColdDecided);
        let basis = match exchange_out_logicals(&aug, m, n, &w.basis) {
            Some(b) => b,
            None => {
                crate::profile::incr(crate::profile::Ctr::DualColdShortBasis);
                Basis {
                    col_status: w.basis.col_status.iter().take(n).copied().collect(),
                    basic_vars: w
                        .basis
                        .basic_vars
                        .iter()
                        .copied()
                        .filter(|&j| j < n)
                        .collect(),
                }
            }
        };
        let obj = if w.status == LpStatus::Optimal {
            x.iter().zip(sc).map(|(xj, cj)| xj * cj).sum()
        } else {
            w.obj
        };
        Some(LpSolve {
            status: w.status,
            x,
            obj,
            basis,
            iters,
            dual: w.dual,
            ray: w.ray,
        })
    }

    /// #2163: everything this problem does with a `Numerical` solve, in order. The
    /// #649 unscaled retry first ([`Self::retry_numerical_unscaled`]); if the LP is
    /// still undecided, the logical-basis dual solve ([`Self::logical_basis_dual`]).
    /// Each stage runs only when every stage before it failed to decide, so an LP
    /// the solver already decides takes exactly the path it took before.
    fn recover_numerical(
        &self,
        sol: LpSolve,
        l: &[f64],
        u: &[f64],
        start: Option<&Basis>,
        opts: &SimplexOptions,
    ) -> LpSolve {
        let sol = self.retry_numerical_unscaled(sol, l, u, start, opts);
        if sol.status != LpStatus::Numerical {
            return sol;
        }
        match self.logical_basis_dual(l, u, opts) {
            Some(mut alt) => {
                if let Some(s) = &self.scaling {
                    s.scale_x(&mut alt.x);
                    s.scale_dual(&mut alt.dual);
                    s.scale_ray(&mut alt.ray);
                }
                alt
            }
            None => sol,
        }
    }

    /// #2163: solve the ORIGINAL LP with the dual simplex from an all-logical basis,
    /// the way HiGHS starts: every row gets a logical column, `A x - s = 0` with
    /// `s` fixed at `[b, b]`, so a basis of logicals always exists, and it is dual
    /// feasible once each structural column sits at the bound its cost sign wants.
    ///
    /// Why a third stage: on trained ReLU big-M MILPs with Anderson cut rows
    /// (entries ~6e-8..1.4e3, equality rows with no logical of their own), the
    /// primal cold solve -- scaled or not -- stops at a basis the final audit
    /// refuses (28 of 43 captured node LPs with a basic column off its box, 16 with
    /// a row residual), with phase-1 artificials still basic, so no full basis is
    /// left to warm-start from. From the logical basis the dual simplex decided all
    /// 43, each audit-clean in original units (worst 5e-10) and matching HiGHS;
    /// on the 207 plain big-M node LPs of #2163's network A it decided all 207
    /// (201 optimal matching HiGHS, 6 infeasible as HiGHS says), including the 4
    /// the unscaled retry could not.
    ///
    /// Accepted only with a certificate checked HERE, because the warm dual path
    /// runs no feasibility audit of its own (the #1287 retraction in `dual.rs`):
    /// an `Optimal` whose point passes the original-LP audit, or an `Infeasible`
    /// whose ray verifies as a Farkas certificate of the original LP. The rows of
    /// the augmented LP are the original rows, and the logical columns contribute
    /// exactly `bᵀy` to the Farkas bound, so the ray is a ray of the original LP
    /// unchanged. Anything else is `None` and the caller keeps its `Numerical`.
    ///
    /// Returns ORIGINAL coordinates: `x` and the basis restricted to the
    /// structural columns (a basic logical has no structural counterpart, so the
    /// basis can come back short; warm starts already refuse a short basis and
    /// solve cold), row duals and ray over the original rows.
    fn logical_basis_dual(&self, l: &[f64], u: &[f64], opts: &SimplexOptions) -> Option<LpSolve> {
        const INF: f64 = 1e20;
        let (m, n) = (self.m, self.n);
        let mut col_status = Vec::with_capacity(n + m);
        for j in 0..n {
            let lo = l[j] > -INF;
            let hi = u[j] < INF;
            let c = self.c[j];
            col_status.push(if c >= 0.0 && lo {
                AT_LOWER
            } else if c <= 0.0 && hi {
                AT_UPPER
            } else {
                // No bound on the side the cost needs: the logical basis is not dual
                // feasible and this stage does not apply.
                return None;
            });
        }
        col_status.extend(std::iter::repeat_n(BASIC, m));
        crate::profile::incr(crate::profile::Ctr::LogicalBasisDualAttempts);
        let (cp, ri, vals) = self.original.raw();
        let mut col_ptr = cp.to_vec();
        let mut row_idx = ri.to_vec();
        let mut vv = vals.to_vec();
        for i in 0..m {
            row_idx.push(i);
            vv.push(-1.0);
            col_ptr.push(row_idx.len());
        }
        let aug = SparseCols::from_csc(col_ptr, row_idx, vv);
        let mut c_a = self.c.clone();
        c_a.resize(n + m, 0.0);
        let mut l_a = l.to_vec();
        l_a.extend_from_slice(&self.b);
        let mut u_a = u.to_vec();
        u_a.extend_from_slice(&self.b);
        let b_a = vec![0.0; m];
        let start = Basis {
            col_status,
            basic_vars: (n..n + m).collect(),
        };
        let view = LpView {
            a: &[],
            m,
            n: n + m,
            c: &c_a,
            l: &l_a,
            u: &u_a,
        };
        let w = solve_lp_warm_scaled_csc(&view, &b_a, &start, opts, &aug);
        let x: Vec<f64> = w.x.iter().take(n).copied().collect();
        let accepted = match w.status {
            LpStatus::Optimal => {
                w.dual.len() == m
                    && point_passes_original_audit(&x, &self.original, m, n, &self.b, l, u)
            }
            LpStatus::Infeasible => {
                farkas_ray_certifies_cols(&w.dual, &self.original, n, m, &self.b, l, u)
            }
            _ => false,
        };
        if !accepted {
            return None;
        }
        crate::profile::incr(crate::profile::Ctr::LogicalBasisDualDecided);
        let basis = Basis {
            col_status: w.basis.col_status.iter().take(n).copied().collect(),
            basic_vars: w
                .basis
                .basic_vars
                .iter()
                .copied()
                .filter(|&j| j < n)
                .collect(),
        };
        let obj = if w.status == LpStatus::Optimal {
            x.iter().zip(&self.c).map(|(xj, cj)| xj * cj).sum()
        } else {
            w.obj
        };
        Some(LpSolve {
            status: w.status,
            x,
            obj,
            basis,
            iters: w.iters,
            dual: w.dual,
            ray: w.ray,
        })
    }

    /// The choke point: check a solve-space result `sol` of this problem over
    /// ORIGINAL bounds `l`, `u` against the original data. Returns `sol` unchanged
    /// unless it is an `Optimal` of a scaled solve whose unscaled point fails the
    /// original-space audit; then the LP is re-solved on the original data (warm
    /// from `start` when given, else cold) and a decided retry is returned in solve
    /// coordinates, or the verdict becomes `Numerical` with no certificate.
    ///
    /// For a caller that runs its own solve on [`Self::solve_matrix`] (e.g. a
    /// prepared dual re-optimization); [`Self::solve_warm`] / [`Self::solve_cold`]
    /// call it already.
    pub fn verify(
        &self,
        sol: LpSolve,
        l: &[f64],
        u: &[f64],
        start: Option<&Basis>,
        opts: &SimplexOptions,
    ) -> LpSolve {
        // Unscaled: the simplex's own final audit already ran in original units.
        let Some(s) = &self.scaling else {
            return sol;
        };
        if sol.status != LpStatus::Optimal {
            return sol;
        }
        let mut x = sol.x.clone();
        s.unscale_x(&mut x);
        if !point_fails_original_audit(&x, &self.original, self.m, self.n, &self.b, l, u) {
            return sol;
        }
        let view = self.view(&self.c, l, u);
        let mut alt = match start {
            Some(basis) => solve_lp_warm_scaled_csc(&view, &self.b, basis, opts, &self.original),
            None => solve_lp_cols(
                self.original.clone(),
                self.m,
                self.n,
                &self.c,
                l,
                u,
                &self.b,
                opts,
            ),
        };
        if !(self.decided_on_original(&alt, l, u) && retry_decided(&alt)) {
            // #2163: before refusing, the logical-basis dual solve. Measured: on 4 of
            // the 43 cut-augmented node LPs of #2163 the scaled solve returned a false
            // `Optimal`, this re-solve decided nothing, and the LP came back
            // `Numerical`; the logical-basis dual decides all 4 (matching HiGHS).
            return match self.logical_basis_dual(l, u, opts) {
                Some(mut alt) => {
                    s.scale_x(&mut alt.x);
                    s.scale_dual(&mut alt.dual);
                    s.scale_ray(&mut alt.ray);
                    alt
                }
                None => downgrade_false_optimal(sol),
            };
        }
        // The retry ran on the original data; express it in solve coordinates, as
        // every other result of this problem is. Exact: power-of-two factors.
        s.scale_x(&mut alt.x);
        s.scale_dual(&mut alt.dual);
        s.scale_ray(&mut alt.ray);
        alt
    }

    fn view<'a>(&'a self, c: &'a [f64], l: &'a [f64], u: &'a [f64]) -> LpView<'a> {
        LpView {
            a: &[], // the CSC solvers read the matrix from their CSC argument
            m: self.m,
            n: self.n,
            c,
            l,
            u,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::simplex::{
        solve_lp, solve_lp_batch, solve_lp_cols_scaled, solve_lp_warm, solve_lp_warm_csc,
        LpInstance,
    };

    const INF: f64 = 1e20;
    const FIXTURE: &str = include_str!("testdata/adhya1stp_tiny_coef_false_optimal_lp.json");

    /// W5(a): the dual cold solve reaches the primal cold solve's optimum and exports
    /// a FULL basis, on LPs that need each of its parts. Each case is
    /// `(dense A, m, n, c, l, u, b)`:
    /// 1. equality rows with no slack (logicals must be exchanged out by LU pivots);
    /// 2. a free column with a nonzero cost (the auxiliary dual phase 1);
    /// 3. a lower-only column whose cost has the wrong sign (phase 1 again);
    /// 4. an inequality row with its own slack singleton (the relabel).
    /// An infeasible LP must come back `Infeasible` with a verified Farkas ray.
    /// Executed comparisons are counted (CLAUDE.md §6).
    #[test]
    fn dual_cold_matches_primal_cold_and_exports_full_basis() {
        type Case = (
            Vec<f64>,
            usize,
            usize,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        );
        let cases: Vec<Case> = vec![
            // x0 + x1 + x2 = 2, x0 - x1 = 0.5; min x0 + 2x1 + 3x2, x in [0, 5]
            (
                vec![1.0, 1.0, 1.0, 1.0, -1.0, 0.0],
                2,
                3,
                vec![1.0, 2.0, 3.0],
                vec![0.0; 3],
                vec![5.0; 3],
                vec![2.0, 0.5],
            ),
            // free x0 with c = 1: x0 - x1 = 0, x1 in [1, 3]  -> x0 = 1
            (
                vec![1.0, -1.0],
                1,
                2,
                vec![1.0, 0.0],
                vec![-INF, 1.0],
                vec![INF, 3.0],
                vec![0.0],
            ),
            // lower-only x0 with c = -1: x0 + x1 = 4, x1 in [0, 4] -> x0 = 4, obj -4
            (
                vec![1.0, 1.0],
                1,
                2,
                vec![-1.0, 0.0],
                vec![0.0, 0.0],
                vec![INF, 4.0],
                vec![4.0],
            ),
            // 2x0 + x1 - s = 3, s in [0, inf) (slack singleton); min x0 + x1
            (
                vec![2.0, 1.0, -1.0],
                1,
                3,
                vec![1.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![10.0, 10.0, INF],
                vec![3.0],
            ),
        ];
        let opts = SimplexOptions::default();
        let mut checked = 0usize;
        for (k, (a, m, n, c, l, u, b)) in cases.iter().enumerate() {
            let sp = SparseCols::from_dense(a, *m, *n);
            let p = ScaledProblem::new(sp.clone(), *m, *n, c, b);
            let primal = solve_lp_cols(sp.clone(), *m, *n, c, l, u, b, &opts);
            assert_eq!(primal.status, LpStatus::Optimal, "case {k}: premise");
            let dual = p
                .dual_cold(l, u, &opts)
                .unwrap_or_else(|| panic!("case {k}: undecided"));
            assert_eq!(dual.status, LpStatus::Optimal, "case {k}");
            assert!(
                (dual.obj - primal.obj).abs() <= 1e-9 * (1.0 + primal.obj.abs()),
                "case {k}: dual {} vs primal {}",
                dual.obj,
                primal.obj
            );
            assert_eq!(dual.basis.basic_vars.len(), *m, "case {k}: short basis");
            assert_eq!(dual.basis.col_status.len(), *n, "case {k}");
            assert!(dual.basis.basic_vars.iter().all(|&j| j < *n), "case {k}");
            checked += 1;
        }
        // x0 + x1 = 5 with x in [0, 1]: infeasible.
        let sp = SparseCols::from_dense(&[1.0, 1.0], 1, 2);
        let (c, l, u, b) = (vec![1.0, 1.0], vec![0.0; 2], vec![1.0; 2], vec![5.0]);
        let p = ScaledProblem::new(sp.clone(), 1, 2, &c, &b);
        let inf = p
            .dual_cold(&l, &u, &opts)
            .expect("decides the infeasible LP");
        assert_eq!(inf.status, LpStatus::Infeasible);
        assert!(farkas_ray_certifies_cols(&inf.dual, &sp, 2, 1, &b, &l, &u));
        checked += 1;
        assert_eq!(checked, cases.len() + 1);
    }

    #[test]
    fn parse_dual_cold_table() {
        assert!(!parse_dual_cold(""));
        assert!(!parse_dual_cold("0"));
        assert!(parse_dual_cold("1"));
        assert!(parse_dual_cold(" Yes "));
        assert!(std::panic::catch_unwind(|| parse_dual_cold("maybe")).is_err());
    }

    struct Lp {
        m: usize,
        n: usize,
        sp: SparseCols,
        c: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        b: Vec<f64>,
    }

    /// The fixture's LP and its captured basis (the flat-JSON format of
    /// `dual.rs`'s `parse_stall_fixture`).
    fn fixture() -> (Lp, Basis) {
        fixture_from(FIXTURE)
    }

    /// [`fixture`] for any captured LP in the same flat-JSON format.
    fn fixture_from(json: &str) -> (Lp, Basis) {
        fn field<'a>(json: &'a str, key: &str) -> &'a str {
            let k = format!("\"{key}\":");
            &json[json.find(&k).expect("key present") + k.len()..]
        }
        fn scalar(json: &str, key: &str) -> usize {
            let s = field(json, key);
            s[..s.find([',', '}']).unwrap()].trim().parse().unwrap()
        }
        fn arr(json: &str, key: &str) -> Vec<f64> {
            let s = field(json, key);
            let open = s.find('[').unwrap() + 1;
            let close = s[open..].find(']').unwrap() + open;
            s[open..close]
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().parse().unwrap())
                .collect()
        }
        let us = |k: &str| {
            arr(json, k)
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>()
        };
        let lp = Lp {
            m: scalar(json, "m"),
            n: scalar(json, "n"),
            sp: SparseCols::from_csc(us("col_ptr"), us("row_idx"), arr(json, "vals")),
            c: arr(json, "c"),
            l: arr(json, "l"),
            u: arr(json, "u"),
            b: arr(json, "b"),
        };
        let basis = Basis {
            col_status: arr(json, "col_status")
                .into_iter()
                .map(|v| v as i8)
                .collect(),
            basic_vars: us("basic_vars"),
        };
        (lp, basis)
    }

    /// Max relative violation of `l <= x <= u` and `A x = b` in ORIGINAL units.
    fn original_violation(lp: &Lp, x: &[f64]) -> f64 {
        let mut worst = 0.0f64;
        for j in 0..lp.n {
            if lp.l[j] > -INF {
                worst = worst.max((lp.l[j] - x[j]) / (1.0 + lp.l[j].abs()));
            }
            if lp.u[j] < INF {
                worst = worst.max((x[j] - lp.u[j]) / (1.0 + lp.u[j].abs()));
            }
        }
        let mut ax = vec![0.0; lp.m];
        for j in 0..lp.n {
            let (rows, vals) = lp.sp.col(j);
            for (&i, &a) in rows.iter().zip(vals) {
                ax[i] += a * x[j];
            }
        }
        (0..lp.m).fold(worst, |w, i| {
            w.max((ax[i] - lp.b[i]).abs() / (1.0 + lp.b[i].abs()))
        })
    }

    /// #2163 regression: two node LPs the Rust MILP engine handed back undecided on
    /// a trained 2-32-32-1 ReLU big-M MILP (network A of the issue), captured with
    /// the node's warm-start basis. Equilibration sets column factors up to 2^16 on
    /// this matrix (entries ~1.5e-8 .. ~12), and the solve on the scaled data ends
    /// `Numerical`, warm and cold (asserted first, so the fixtures provably still
    /// exercise the failure). HiGHS decides both: one optimal at
    /// -6.806923340932464, one infeasible. `ScaledProblem` must decide them too,
    /// through the unscaled retry, and with certificates that hold where the
    /// driver checks them (the point in original units; the Farkas ray in solve
    /// coordinates, where the driver verifies it).
    #[test]
    fn scaled_numerical_is_recovered_by_the_unscaled_retry_2163() {
        let opts = SimplexOptions::default();
        for (json, want_optimal) in [
            (
                include_str!("testdata/relu_bigm_scaled_numerical_optimal_lp.json"),
                true,
            ),
            (
                include_str!("testdata/relu_bigm_scaled_numerical_infeasible_lp.json"),
                false,
            ),
        ] {
            let (lp, basis) = fixture_from(json);
            let p = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
            assert!(p.scaling().is_some(), "fixture must be equilibrated");
            let (sl, su) = p.solve_bounds(&lp.l, &lp.u);

            // Precondition: the scaled solves fail, warm and cold.
            let view = p.view(p.solve_c(), &sl, &su);
            let warm_scaled =
                solve_lp_warm_scaled_csc(&view, p.solve_b(), &basis, &opts, p.solve_matrix());
            let cold_scaled = solve_lp_cols(
                p.solve_matrix().clone(),
                lp.m,
                lp.n,
                p.solve_c(),
                &sl,
                &su,
                p.solve_b(),
                &opts,
            );
            assert_eq!(
                warm_scaled.status,
                LpStatus::Numerical,
                "precondition (warm)"
            );
            assert_eq!(
                cold_scaled.status,
                LpStatus::Numerical,
                "precondition (cold)"
            );

            let warm = p.solve_warm(&lp.l, &lp.u, &basis, &opts);
            let cold = p.solve_cold(&lp.l, &lp.u, &opts);
            for (label, sol) in [("warm", warm), ("cold", cold)] {
                if want_optimal {
                    assert_eq!(sol.status, LpStatus::Optimal, "{label}");
                    let hi = -6.806923340932464f64;
                    assert!(
                        (sol.obj - hi).abs() <= 1e-7 * (1.0 + hi.abs()),
                        "{label}: obj {} vs HiGHS {hi}",
                        sol.obj
                    );
                    let mut x = sol.x.clone();
                    p.scaling().unwrap().unscale_x(&mut x);
                    let viol = original_violation(&lp, &x);
                    assert!(viol <= 1e-6, "{label}: original violation {viol:e}");
                } else {
                    assert_eq!(sol.status, LpStatus::Infeasible, "{label}");
                    assert!(
                        super::super::primal::farkas_ray_certifies_cols(
                            &sol.dual,
                            p.solve_matrix(),
                            lp.n,
                            lp.m,
                            p.solve_b(),
                            &sl,
                            &su,
                        ),
                        "{label}: the ray must verify in solve coordinates"
                    );
                }
            }
        }
    }

    /// #2163 regression, cut-augmented class: two node LPs captured from the Rust
    /// MILP engine on #2163's network B with native Anderson cuts (#2164). Both end
    /// undecided on every pre-existing stage: the cold primal on the ORIGINAL data
    /// is `Numerical` (asserted first, so the fixtures still exercise the failure),
    /// and through `ScaledProblem` one reaches the logical stage from a scaled
    /// `Numerical`, the other from a scaled false `Optimal` that `verify` refuses.
    /// HiGHS solves both. The logical-basis dual stage must decide them, with a point
    /// that is feasible in original units and HiGHS's objective, and the counter
    /// proves that stage (not some other path) is what decided them.
    #[test]
    fn cut_augmented_numerical_is_decided_by_the_logical_basis_dual_2163() {
        let _guard = crate::profile::test_guard();
        crate::profile::set_enabled(true);
        crate::profile::reset();
        let opts = SimplexOptions::default();
        let mut checks = 0usize;
        for (json, highs) in [
            (
                include_str!("testdata/relu_anderson_cuts_numerical_lp.json"),
                -254.27360315225872f64,
            ),
            (
                include_str!("testdata/relu_anderson_cuts_false_optimal_lp.json"),
                22.36816528521379f64,
            ),
        ] {
            let (lp, basis) = fixture_from(json);
            let raw = solve_lp_cols(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.l, &lp.u, &lp.b, &opts);
            assert_eq!(raw.status, LpStatus::Numerical, "precondition: cold primal");

            let p = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
            let warm = p.solve_warm(&lp.l, &lp.u, &basis, &opts);
            let cold = p.solve_cold(&lp.l, &lp.u, &opts);
            for (label, sol) in [("warm", warm), ("cold", cold)] {
                assert_eq!(sol.status, LpStatus::Optimal, "{label}");
                assert!(
                    (sol.obj - highs).abs() <= 1e-7 * (1.0 + highs.abs()),
                    "{label}: obj {} vs HiGHS {highs}",
                    sol.obj
                );
                let mut x = sol.x.clone();
                if let Some(s) = p.scaling() {
                    s.unscale_x(&mut x);
                }
                let viol = original_violation(&lp, &x);
                assert!(viol <= 1e-6, "{label}: original violation {viol:e}");
                checks += 1;
            }
        }
        // Under `DISCOPT_LP_DUAL_COLD=1` the cold solve is the W5(a) dual cold solve,
        // the same logical-basis dual with its own audit; count either stage.
        let decided = crate::profile::counter(crate::profile::Ctr::LogicalBasisDualDecided)
            + crate::profile::counter(crate::profile::Ctr::DualColdDecided);
        crate::profile::set_enabled(false);
        assert_eq!(checks, 4);
        assert!(
            decided >= 4,
            "the logical-basis dual decided {decided} solves, want >= 4"
        );
    }

    /// #2163: the logical-basis dual on an infeasible LP returns `Infeasible` only
    /// with a ray that certifies the ORIGINAL LP empty (the augmented rows are the
    /// original rows), and on a feasible one an audited optimum matching HiGHS.
    #[test]
    fn logical_basis_dual_certificates_hold_on_the_original_lp_2163() {
        let opts = SimplexOptions::default();
        let (lp, _) = fixture_from(include_str!(
            "testdata/relu_bigm_scaled_numerical_infeasible_lp.json"
        ));
        let p = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
        let inf = p
            .logical_basis_dual(&lp.l, &lp.u, &opts)
            .expect("decides the infeasible fixture");
        assert_eq!(inf.status, LpStatus::Infeasible);
        assert!(super::super::primal::farkas_ray_certifies_cols(
            &inf.dual, &lp.sp, lp.n, lp.m, &lp.b, &lp.l, &lp.u
        ));

        let (lp, _) = fixture_from(include_str!(
            "testdata/relu_bigm_scaled_numerical_optimal_lp.json"
        ));
        let p = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
        let opt = p
            .logical_basis_dual(&lp.l, &lp.u, &opts)
            .expect("decides the optimal fixture");
        assert_eq!(opt.status, LpStatus::Optimal);
        let hi = -6.806923340932464f64;
        assert!(
            (opt.obj - hi).abs() <= 1e-7 * (1.0 + hi.abs()),
            "obj {}",
            opt.obj
        );
        assert!(original_violation(&lp, &opt.x) <= 1e-6);
        assert_eq!(opt.dual.len(), lp.m);
    }

    /// #1945 regression. The MILP driver's old recipe -- equilibrate once, solve the
    /// SCALED matrix with the core solvers, unscale `x` -- returns `Optimal` on this
    /// infeasible LP with column 0 off its bound by 0.0179 (asserted first, so the
    /// fixture provably still exercises the mechanism). Through `ScaledProblem` the
    /// same solves never report an `Optimal` that fails the original LP.
    #[test]
    fn scaled_problem_refuses_the_false_optimal_the_old_driver_recipe_returned() {
        let (lp, basis) = fixture();
        let opts = SimplexOptions::default();
        let mut checked = 0usize;

        // The old recipe, cold and warm, on the driver's scaled data.
        let scaling = Scaling::from_sparse(&lp.sp, lp.m, lp.n).expect("fixture is ill-scaled");
        let mut a_s = lp.sp.clone();
        scaling.scale_cols(&mut a_s);
        let (c_s, b_s) = (scaling.scale_c(&lp.c), scaling.scale_b(&lp.b));
        let (l_s, u_s) = (scaling.scale_lower(&lp.l), scaling.scale_upper(&lp.u));
        let view = LpView {
            a: &[],
            m: lp.m,
            n: lp.n,
            c: &c_s,
            l: &l_s,
            u: &u_s,
        };
        let old = [
            solve_lp_cols(a_s.clone(), lp.m, lp.n, &c_s, &l_s, &u_s, &b_s, &opts),
            solve_lp_warm_scaled_csc(&view, &b_s, &basis, &opts, &a_s),
        ];
        for mut sol in old {
            assert_eq!(
                sol.status,
                LpStatus::Optimal,
                "the fixture no longer reproduces #1945"
            );
            scaling.unscale_x(&mut sol.x);
            let v = original_violation(&lp, &sol.x);
            assert!(
                v > 1e-3,
                "old recipe's Optimal was feasible after all ({v:e})"
            );
            checked += 2;
        }

        // The same solves through the choke point.
        let problem = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
        assert!(problem.scaling().is_some());
        let _guard = crate::profile::test_guard();
        for warm in [false, true] {
            crate::profile::reset_totals();
            crate::profile::set_enabled(true);
            let mut sol = if warm {
                problem.solve_warm(&lp.l, &lp.u, &basis, &opts)
            } else {
                problem.solve_cold(&lp.l, &lp.u, &opts)
            };
            let snap = crate::profile::counter_snapshot();
            let ctr = |name: &str| snap.iter().find(|(k, _)| *k == name).map_or(0, |(_, v)| *v);
            let refused = ctr("ScaledOptimalOriginalAuditFail");
            // `DISCOPT_LP_DUAL_COLD=1`: the cold solve is decided by the dual cold
            // solve, which audits its Optimal on the original data itself (and the
            // checks below re-audit it), so the scaled solve's audit never runs.
            let dual_cold = if warm { 0 } else { ctr("DualColdDecided") };
            crate::profile::set_enabled(false);
            assert!(
                refused >= 1 || dual_cold >= 1,
                "warm={warm}: the original-space audit never fired"
            );
            checked += 1;
            if sol.status == LpStatus::Optimal {
                problem.scaling().unwrap().unscale_x(&mut sol.x);
                let v = original_violation(&lp, &sol.x);
                assert!(
                    v <= 1e-6,
                    "warm={warm}: Optimal still violates the original LP ({v:e})"
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 8, "probe did not run every assertion");
    }

    /// Deterministic generator (no `rand` dependency): xorshift64*.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A random standard-form LP `[A | I] x = b` whose columns mix entries of
    /// ~1e-14..1e-9 with O(1) ones (the #1942/#1945 class), feasible by construction
    /// (`b = A x0` for a point `x0` inside the box) unless `perturb_infeasible`.
    fn random_ill_scaled_lp(rng: &mut Rng, m: usize, k: usize, perturb_infeasible: bool) -> Lp {
        let n = k + m;
        let mut dense = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..k {
                let r = rng.next();
                if r < 0.15 {
                    dense[i * n + j] = 10f64.powf(-14.0 + 5.0 * rng.next()); // tiny
                } else if r < 0.6 {
                    dense[i * n + j] = 4.0 * rng.next() - 2.0;
                }
            }
            dense[i * n + k + i] = 1.0; // slack
        }
        let mut l = vec![0.0; n];
        let mut u = vec![0.0; n];
        let mut x0 = vec![0.0; n];
        for j in 0..k {
            l[j] = 10.0 * rng.next() - 5.0;
            u[j] = l[j] + 0.5 + 10.0 * rng.next();
            x0[j] = l[j] + (u[j] - l[j]) * rng.next();
        }
        for j in k..n {
            l[j] = 0.0;
            u[j] = INF;
            x0[j] = 3.0 * rng.next();
        }
        let mut b: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| dense[i * n + j] * x0[j]).sum())
            .collect();
        if perturb_infeasible {
            b[0] -= 1e3; // slack 0 would have to go negative
        }
        let c = (0..n)
            .map(|j| if j < k { 2.0 * rng.next() - 1.0 } else { 0.0 })
            .collect();
        Lp {
            m,
            n,
            sp: SparseCols::from_dense(&dense, m, n),
            c,
            l,
            u,
            b,
        }
    }

    fn dense(lp: &Lp) -> Vec<f64> {
        let mut a = vec![0.0; lp.m * lp.n];
        for j in 0..lp.n {
            let (rows, vals) = lp.sp.col(j);
            for (&i, &v) in rows.iter().zip(vals) {
                a[i * lp.n + j] = v;
            }
        }
        a
    }

    /// #1945 invariant: whichever public entry point solves an LP, an `Optimal` holds
    /// on the ORIGINAL data (relative 1e-6 plus roundoff slack). Covers the
    /// auto-scaling entry points #1942 patched and both `ScaledProblem` solves, over
    /// the fixture and a family of random ill-scaled LPs.
    #[test]
    fn every_public_solve_returns_only_original_feasible_optima() {
        let opts = SimplexOptions::default();
        let mut rng = Rng(0x1945_1942_dead_beef);
        let mut lps: Vec<Lp> = vec![fixture().0];
        for t in 0..60 {
            let (m, k) = (3 + t % 7, 4 + (t * 5) % 9);
            lps.push(random_ill_scaled_lp(&mut rng, m, k, t % 10 == 9));
        }
        let (mut optima, mut solves) = (0usize, 0usize);
        for lp in &lps {
            let a = dense(lp);
            let view = LpView {
                a: &a,
                m: lp.m,
                n: lp.n,
                c: &lp.c,
                l: &lp.l,
                u: &lp.u,
            };
            // Slack basis start for the warm entry points (the last m columns of the
            // generated LPs; for the fixture any shape-valid start exercises the path).
            let start = {
                let mut col_status = vec![crate::lp::basis::AT_LOWER; lp.n];
                let basic_vars: Vec<usize> = (lp.n - lp.m..lp.n).collect();
                for &j in &basic_vars {
                    col_status[j] = crate::lp::basis::BASIC;
                }
                Basis {
                    col_status,
                    basic_vars,
                }
            };
            let problem = ScaledProblem::new(lp.sp.clone(), lp.m, lp.n, &lp.c, &lp.b);
            let mut results: Vec<(&str, LpSolve)> = vec![
                ("solve_lp", solve_lp(&view, &lp.b, &opts)),
                (
                    "solve_lp_cols_scaled",
                    solve_lp_cols_scaled(
                        lp.sp.clone(),
                        lp.m,
                        lp.n,
                        &lp.c,
                        &lp.l,
                        &lp.u,
                        &lp.b,
                        &opts,
                    ),
                ),
                ("solve_lp_warm", solve_lp_warm(&view, &lp.b, &start, &opts)),
                (
                    "solve_lp_warm_csc",
                    solve_lp_warm_csc(
                        lp.sp.clone(),
                        lp.m,
                        lp.n,
                        &lp.c,
                        &lp.l,
                        &lp.u,
                        &lp.b,
                        None,
                        &opts,
                    ),
                ),
            ];
            let inst = [LpInstance {
                b: lp.b.clone(),
                l: lp.l.clone(),
                u: lp.u.clone(),
            }];
            results.push((
                "solve_lp_batch",
                solve_lp_batch(&a, lp.m, lp.n, &lp.c, &inst, &opts).remove(0),
            ));
            // `ScaledProblem` returns solve coordinates: unscale before checking.
            for (name, mut sol) in [
                (
                    "ScaledProblem::solve_cold",
                    problem.solve_cold(&lp.l, &lp.u, &opts),
                ),
                (
                    "ScaledProblem::solve_warm",
                    problem.solve_warm(&lp.l, &lp.u, &start, &opts),
                ),
            ] {
                if let Some(s) = problem.scaling() {
                    s.unscale_x(&mut sol.x);
                }
                results.push((name, sol));
            }
            for (name, sol) in results {
                solves += 1;
                if sol.status == LpStatus::Optimal {
                    optima += 1;
                    let v = original_violation(lp, &sol.x);
                    assert!(
                        v <= 1e-5,
                        "{name}: Optimal violates the original LP by {v:e}"
                    );
                }
            }
        }
        assert_eq!(solves, lps.len() * 7);
        assert!(
            optima >= 200,
            "too few Optimal verdicts checked ({optima}) -- generator degenerate?"
        );
    }

    /// #1945 structure: outside `lp/simplex` no code may build a `Scaling` or scale a
    /// matrix -- that is how a caller ends up solving scaled data with no
    /// original-space audit. Scale once through `ScaledProblem`, or call an
    /// auto-scaling entry point. Test modules (after `#[cfg(test)]`) are skipped.
    #[test]
    fn nothing_outside_lp_simplex_scales_a_matrix_itself() {
        // A measurement binary, not a solve path: it times the scaled warm solve.
        const ALLOW: &[&str] = &["bin/e0_warm_bench.rs"];
        const FORBIDDEN: &[&str] = &[
            "Scaling::from_sparse(",
            "Scaling::from_matrix(",
            ".scale_cols(",
            "ScaledLp::maybe_new(",
        ];
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut stack = vec![src.clone()];
        let (mut files, mut offenders) = (0usize, Vec::new());
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                if p.extension().is_none_or(|e| e != "rs") {
                    continue;
                }
                let rel = p
                    .strip_prefix(&src)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if rel.starts_with("lp/simplex/") || ALLOW.contains(&rel.as_str()) {
                    continue;
                }
                files += 1;
                let text = std::fs::read_to_string(&p).unwrap();
                let prod = text.split("#[cfg(test)]").next().unwrap_or("");
                for pat in FORBIDDEN {
                    if prod.contains(pat) {
                        offenders.push(format!("{rel}: {pat}"));
                    }
                }
            }
        }
        assert!(
            files > 20,
            "scanned only {files} files -- wrong source root?"
        );
        assert!(
            offenders.is_empty(),
            "scales outside lp/simplex (use ScaledProblem): {offenders:?}"
        );
    }
}
