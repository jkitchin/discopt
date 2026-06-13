//! Bounded-variable two-phase **primal** revised simplex (cold start).
//!
//! Solves `min cᵀx s.t. A x = b, l ≤ x ≤ u` (the crate's standard form via
//! [`LpView`]). Phase 1 adds one artificial per row to form a trivial identity
//! basis and minimizes total artificial value to reach a feasible basis (or
//! certify infeasibility); Phase 2 optimizes the real objective. The basis is
//! factorized by [`FeralLU`] (`ftran`/`btran` + product-form updates).
//!
//! This is the correctness-first cold solver (roadmap P1). It recomputes the
//! basic solution by `ftran` each iteration (clear over fast) and uses Dantzig
//! pricing with a Bland's-rule fallback after a stall to prevent cycling.
//! Warm-start (dual simplex) and performance pricing/factorization come later.
//!
// Several simplex loops index `cost`/`stat`/columns by the same index `j`, so a
// range loop is clearer than zipping multiple slices.
#![allow(clippy::needless_range_loop)]

use super::linsolve::{FeralLU, LinearSolver};
use super::{LpSolve, LpStatus, SimplexOptions};
use crate::lp::basis::{Basis, AT_LOWER, AT_UPPER, BASIC};
use crate::lp::crossover::LpView;

const INF: f64 = 1e20;

/// Solve the LP `min cᵀx s.t. A x = b, l ≤ x ≤ u` by two-phase primal simplex.
pub fn solve_lp(lp: &LpView<'_>, b: &[f64], opts: &SimplexOptions) -> LpSolve {
    Simplex::new(lp, b, opts).run()
}

struct Simplex<'a> {
    a: &'a [f64], // m×n row-major (structural+slack columns)
    b: &'a [f64], // length m
    c: &'a [f64], // length n
    m: usize,
    n: usize,  // real columns
    na: usize, // n + m (with artificials appended)
    tol: f64,
    max_iter: usize,
    lu: FeralLU,
    basis: Vec<usize>, // slot -> column (len m)
    slot_of: Vec<i64>, // column -> slot, or -1
    stat: Vec<i8>,     // column -> BASIC/AT_LOWER/AT_UPPER (len na)
    lb: Vec<f64>,      // len na
    ub: Vec<f64>,      // len na
}

impl<'a> Simplex<'a> {
    fn new(lp: &'a LpView<'a>, b: &'a [f64], opts: &SimplexOptions) -> Self {
        let (m, n) = (lp.m, lp.n);
        let na = n + m;
        let mut lb = vec![0.0; na];
        let mut ub = vec![0.0; na];
        lb[..n].copy_from_slice(lp.l);
        ub[..n].copy_from_slice(lp.u);
        // Artificials: bounds set per phase (Phase 1 [0,inf], Phase 2 [0,0]).
        Self {
            a: lp.a,
            b,
            c: lp.c,
            m,
            n,
            na,
            tol: opts.tol,
            max_iter: opts.max_iter,
            lu: FeralLU::new(),
            basis: Vec::new(),
            slot_of: vec![-1; na],
            stat: vec![AT_LOWER; na],
            lb,
            ub,
        }
    }

    /// Dense column `A_j` (length m). Real columns from `a`; artificial `n+i` is
    /// `sign_i · e_i`, sign chosen at init so the starting residual is ≥ 0.
    fn column(&self, j: usize, art_sign: &[f64]) -> Vec<f64> {
        let mut col = vec![0.0; self.m];
        if j < self.n {
            for (i, ci) in col.iter_mut().enumerate() {
                *ci = self.a[i * self.n + j];
            }
        } else {
            let i = j - self.n;
            col[i] = art_sign[i];
        }
        col
    }

    /// Bound value a nonbasic column sits at (lower or upper).
    fn nb_value(&self, j: usize) -> f64 {
        match self.stat[j] {
            AT_UPPER => self.ub[j],
            _ => {
                // at lower; if lower is -inf (free), sit at 0
                if self.lb[j] <= -INF {
                    0.0
                } else {
                    self.lb[j]
                }
            }
        }
    }

    /// Recompute basic-variable values: x_B = B⁻¹ (b − Σ_{nonbasic} A_j x_j).
    fn basic_values(&mut self, art_sign: &[f64]) -> Result<Vec<f64>, ()> {
        let mut rhs = self.b.to_vec();
        for j in 0..self.na {
            if self.stat[j] != BASIC {
                let v = self.nb_value(j);
                if v != 0.0 {
                    let col = self.column(j, art_sign);
                    for (i, ci) in col.iter().enumerate() {
                        rhs[i] -= ci * v;
                    }
                }
            }
        }
        self.lu.ftran(&mut rhs).map_err(|_| ())?;
        Ok(rhs)
    }

    fn refactorize(&mut self, art_sign: &[f64]) -> Result<(), ()> {
        let cols: Vec<Vec<f64>> = self
            .basis
            .iter()
            .map(|&j| self.column(j, art_sign))
            .collect();
        self.lu.factorize(self.m, &cols).map_err(|_| ())
    }

    fn run(mut self) -> LpSolve {
        let m = self.m;
        // --- initialize: all real vars nonbasic at a finite bound (or 0 if free)
        for j in 0..self.n {
            self.stat[j] = if self.lb[j] > -INF {
                AT_LOWER
            } else if self.ub[j] < INF {
                AT_UPPER
            } else {
                AT_LOWER // free → treated as at 0
            };
        }
        // residual r = b − Σ A_j x_j over real nonbasic vars
        let mut r = self.b.to_vec();
        for j in 0..self.n {
            let v = self.nb_value(j);
            if v != 0.0 {
                for i in 0..m {
                    r[i] -= self.a[i * self.n + j] * v;
                }
            }
        }
        let art_sign: Vec<f64> = (0..m)
            .map(|i| if r[i] >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        // artificial basis: slot i = column n+i
        self.basis = (0..m).map(|i| self.n + i).collect();
        for i in 0..m {
            let col = self.n + i;
            self.slot_of[col] = i as i64;
            self.stat[col] = BASIC;
            self.lb[col] = 0.0;
            self.ub[col] = INF; // phase 1
        }
        if self.refactorize(&art_sign).is_err() {
            return self.failed();
        }

        // --- Phase 1: minimize sum of artificials ---
        let mut cost1 = vec![0.0; self.na];
        for i in 0..m {
            cost1[self.n + i] = 1.0;
        }
        match self.simplex_loop(&cost1, &art_sign) {
            Ok(_) => {}
            Err(st) => return self.assemble(st, &art_sign),
        }
        let xb = match self.basic_values(&art_sign) {
            Ok(v) => v,
            Err(()) => return self.failed(),
        };
        let infeas: f64 = self
            .basis
            .iter()
            .enumerate()
            .filter(|(_, &j)| j >= self.n)
            .map(|(slot, _)| xb[slot].abs())
            .sum();
        if infeas > 1e-6 {
            return self.assemble(LpStatus::Infeasible, &art_sign);
        }

        // --- Phase 2: pin artificials to 0, optimize real objective ---
        for i in 0..m {
            self.ub[self.n + i] = 0.0; // fix artificials at 0
        }
        let mut cost2 = vec![0.0; self.na];
        cost2[..self.n].copy_from_slice(self.c);
        let st = match self.simplex_loop(&cost2, &art_sign) {
            Ok(()) => LpStatus::Optimal,
            Err(s) => s,
        };
        self.assemble(st, &art_sign)
    }

    /// Primal simplex iterations for the given `cost`. Returns Ok(()) at
    /// optimality, Err(Unbounded/IterLimit/Numerical) otherwise.
    fn simplex_loop(&mut self, cost: &[f64], art_sign: &[f64]) -> Result<(), LpStatus> {
        let m = self.m;
        let mut updates = 0usize;
        let mut stall = 0usize;
        // Devex reference weights γⱼ ≥ 1 for nonbasic pricing, reset per loop
        // (the basis composition differs between Phase 1 and Phase 2). Devex
        // selects the entering column maximizing dⱼ²/γⱼ — a cheap steepest-edge
        // approximation that only changes *which* improving column enters, so it
        // never affects correctness (Bland's rule still guards against cycling).
        let mut gamma = vec![1.0f64; self.na];
        // Maintain x_B incrementally across pivots (rank-1 update per step)
        // instead of recomputing it by a full ftran each iteration; refreshed
        // exactly on every refactorization to bound floating-point drift.
        let mut xb = self
            .basic_values(art_sign)
            .map_err(|_| LpStatus::Numerical)?;
        for _iter in 0..self.max_iter {
            // price: y = B⁻ᵀ c_B ; reduced cost d_j = c_j − yᵀA_j
            let mut y: Vec<f64> = self.basis.iter().map(|&j| cost[j]).collect();
            if self.lu.btran(&mut y).is_err() {
                return Err(LpStatus::Numerical);
            }
            let bland = stall > 2 * (self.na + 1);
            // choose entering: Devex (max dⱼ²/γⱼ over improving cols), with a
            // Bland's-rule fallback (first improving) once a stall is detected.
            let mut enter: Option<usize> = None;
            let mut best_score = 0.0f64;
            for j in 0..self.na {
                if self.stat[j] == BASIC {
                    continue;
                }
                let col = self.column(j, art_sign);
                let dj = cost[j] - dot(&y, &col);
                let improving = (self.stat[j] == AT_LOWER && dj < -self.tol)
                    || (self.stat[j] == AT_UPPER && dj > self.tol);
                if improving {
                    if bland {
                        enter = Some(j);
                        break;
                    }
                    let score = dj * dj / gamma[j];
                    if score > best_score {
                        best_score = score;
                        enter = Some(j);
                    }
                }
            }
            let q = match enter {
                Some(q) => q,
                None => return Ok(()), // optimal
            };
            let dir = if self.stat[q] == AT_LOWER { 1.0 } else { -1.0 };

            // direction α = B⁻¹ A_q
            let mut alpha = self.column(q, art_sign);
            if self.lu.ftran(&mut alpha).is_err() {
                return Err(LpStatus::Numerical);
            }

            // ratio test: entering moves by t≥0; basic i: v_i(t) = xb[i] − dir·t·α[i]
            let mut t_max = if self.ub[q] < INF && self.lb[q] > -INF {
                self.ub[q] - self.lb[q]
            } else {
                INF
            };
            let mut leave_slot: Option<usize> = None;
            let mut leave_to_upper = false;
            for i in 0..m {
                let delta = -dir * alpha[i]; // d v_i / d t
                let bi = self.basis[i];
                if delta < -self.tol {
                    // decreasing → hits lower bound
                    let lb = self.lb[bi];
                    if lb > -INF {
                        let t = (xb[i] - lb) / (-delta);
                        if t < t_max - self.tol || (leave_slot.is_none() && t <= t_max + self.tol) {
                            t_max = t.max(0.0);
                            leave_slot = Some(i);
                            leave_to_upper = false;
                        }
                    }
                } else if delta > self.tol {
                    // increasing → hits upper bound
                    let ub = self.ub[bi];
                    if ub < INF {
                        let t = (ub - xb[i]) / delta;
                        if t < t_max - self.tol || (leave_slot.is_none() && t <= t_max + self.tol) {
                            t_max = t.max(0.0);
                            leave_slot = Some(i);
                            leave_to_upper = true;
                        }
                    }
                }
            }

            if t_max >= INF {
                return Err(LpStatus::Unbounded);
            }
            if t_max <= self.tol {
                stall += 1;
            } else {
                stall = 0;
            }

            // Incremental x_B step: basic values move along −dir·α by t_max.
            let q_val = self.nb_value(q); // entering's value before the move
            for (i, xbi) in xb.iter_mut().enumerate() {
                *xbi -= dir * t_max * alpha[i];
            }

            match leave_slot {
                None => {
                    // bound flip: entering goes to its other bound, no basis change
                    self.stat[q] = if self.stat[q] == AT_LOWER {
                        AT_UPPER
                    } else {
                        AT_LOWER
                    };
                }
                Some(slot) => {
                    // Devex reference-weight update — uses the OLD basis (the LU
                    // still factorizes it until `update` below). Pivot element is
                    // α at the leaving row; it is nonzero by the ratio test.
                    let pivot = alpha[slot];
                    if !bland && pivot.abs() > self.tol {
                        let gamma_q = gamma[q];
                        let mut rho = vec![0.0; m];
                        rho[slot] = 1.0;
                        if self.lu.btran(&mut rho).is_ok() {
                            for j in 0..self.na {
                                if self.stat[j] != BASIC && j != q {
                                    let arj = dot(&rho, &self.column(j, art_sign));
                                    let cand = (arj / pivot) * (arj / pivot) * gamma_q;
                                    if cand > gamma[j] {
                                        gamma[j] = cand;
                                    }
                                }
                            }
                            // The leaving variable becomes nonbasic with a fresh weight.
                            let leaving0 = self.basis[slot];
                            gamma[leaving0] = (gamma_q / (pivot * pivot)).max(1.0);
                            // Reframe (reset weights) if drift makes them blow up.
                            if gamma[leaving0] > 1e10 {
                                for g in gamma.iter_mut() {
                                    *g = 1.0;
                                }
                            }
                        }
                    }
                    let leaving = self.basis[slot];
                    self.stat[leaving] = if leave_to_upper { AT_UPPER } else { AT_LOWER };
                    self.slot_of[leaving] = -1;
                    self.basis[slot] = q;
                    self.slot_of[q] = slot as i64;
                    self.stat[q] = BASIC;
                    // the entering variable now occupies this slot
                    xb[slot] = q_val + dir * t_max;
                    // factorization update with the entering column
                    let col = self.column(q, art_sign);
                    let need_refac = self.lu.update(slot, &col).is_err();
                    updates += 1;
                    if need_refac || updates >= 48 {
                        if self.refactorize(art_sign).is_err() {
                            return Err(LpStatus::Numerical);
                        }
                        xb = self
                            .basic_values(art_sign)
                            .map_err(|_| LpStatus::Numerical)?;
                        updates = 0;
                    }
                }
            }
        }
        Err(LpStatus::IterLimit)
    }

    fn assemble(mut self, status: LpStatus, art_sign: &[f64]) -> LpSolve {
        let xb = self
            .basic_values(art_sign)
            .unwrap_or_else(|_| vec![0.0; self.m]);
        let mut x = vec![0.0; self.n];
        for j in 0..self.n {
            x[j] = if self.stat[j] == BASIC {
                xb[self.slot_of[j] as usize]
            } else {
                self.nb_value(j)
            };
        }
        let obj: f64 = (0..self.n).map(|j| self.c[j] * x[j]).sum();
        // basis over the real columns (artificials excluded; if a real-var basis
        // is wanted the caller post-processes — here basic_vars lists real basics)
        let basic_vars: Vec<usize> = self.basis.iter().copied().filter(|&j| j < self.n).collect();
        let col_status: Vec<i8> = (0..self.n).map(|j| self.stat[j]).collect();
        LpSolve {
            status,
            x,
            obj,
            basis: Basis {
                col_status,
                basic_vars,
            },
            iters: 0,
        }
    }

    fn failed(self) -> LpSolve {
        let n = self.n;
        LpSolve {
            status: LpStatus::Numerical,
            x: vec![0.0; n],
            obj: 0.0,
            basis: Basis {
                col_status: vec![AT_LOWER; n],
                basic_vars: Vec::new(),
            },
            iters: 0,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(a: &[f64], m: usize, n: usize, b: &[f64], c: &[f64], l: &[f64], u: &[f64]) -> LpSolve {
        let lp = LpView { a, m, n, c, l, u };
        solve_lp(&lp, b, &SimplexOptions::default())
    }

    #[test]
    fn knapsack_lp_relaxation() {
        // max 16(x0+x1+x2+x3) s.t. 5Σx + s = 9, x∈[0,1], s∈[0,inf]
        // min -16Σx. Optimum: Σx = 9/5 = 1.8 → obj -28.8.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let c = [-16.0, -16.0, -16.0, -16.0, 0.0];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, 1.0, INF];
        let r = solve(&a, 1, 5, &[9.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - (-28.8)).abs() < 1e-6, "obj {}", r.obj);
        assert!((5.0 * r.x[..4].iter().sum::<f64>() + r.x[4] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn two_constraint_lp() {
        // min -x0 - 2 x1 s.t. x0+x1+s0=4, x0+3x1+s1=6, x∈[0,inf], s∈[0,inf]
        // Optimum at x0=3,x1=1 → -5? check: x0+x1=4, x0+3x1=6 → x1=1,x0=3, obj -5.
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [INF; 4];
        let r = solve(&a, 2, 4, &[4.0, 6.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - (-5.0)).abs() < 1e-6, "obj {}", r.obj);
        assert!((r.x[0] - 3.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn equality_constraint_feasible() {
        // min x0+x1 s.t. x0+x1=2 (equality, no slack), x∈[0,inf]. obj 2.
        let a = [1.0, 1.0];
        let c = [1.0, 1.0];
        let l = [0.0, 0.0];
        let u = [INF, INF];
        let r = solve(&a, 1, 2, &[2.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - 2.0).abs() < 1e-6, "obj {}", r.obj);
    }

    #[test]
    fn infeasible_detected() {
        // x0 + s = 1, s∈[0,inf], x0∈[2,inf]: x0≥2 but x0≤1 → infeasible.
        let a = [1.0, 1.0];
        let c = [1.0, 0.0];
        let l = [2.0, 0.0];
        let u = [INF, INF];
        let r = solve(&a, 1, 2, &[1.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Infeasible);
    }

    #[test]
    fn unbounded_detected() {
        // min -x0 s.t. x0 - s = 0, x0∈[0,inf], s∈[0,inf]  → x0 can grow → unbounded.
        let a = [1.0, -1.0];
        let c = [-1.0, 0.0];
        let l = [0.0, 0.0];
        let u = [INF, INF];
        let r = solve(&a, 1, 2, &[0.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Unbounded);
    }

    #[test]
    fn upper_bound_active() {
        // min -x0 s.t. x0 + s = 10, x0∈[0,3], s∈[0,inf]. Optimum x0=3, obj -3.
        let a = [1.0, 1.0];
        let c = [-1.0, 0.0];
        let l = [0.0, 0.0];
        let u = [3.0, INF];
        let r = solve(&a, 1, 2, &[10.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - (-3.0)).abs() < 1e-6, "obj {}", r.obj);
        assert!((r.x[0] - 3.0).abs() < 1e-6);
    }
}
