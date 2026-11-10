//! MILP presolve on the row form `lo <= A x <= up`, `l <= x <= u` (gap plan §12.34).
//!
//! Provenance: the reductions follow HiGHS's presolve (`HPresolve`, MIT License,
//! <https://github.com/ERGO-Code/HiGHS>, Copyright (c) HiGHS): its rule set and its
//! accept tests guide this implementation, which is written for discopt's row form and
//! postsolve. See `THIRD_PARTY_NOTICES.md` at the repository root, and
//! `lp/highs_cutgen/mod.rs` for why discopt carries such code in Rust rather than
//! linking a native solver.
//!
//! Phase 1 (this module): reductions whose only primal postsolve is "this column was
//! fixed at this value" --
//! - fixed columns (`l == u`) are substituted out;
//! - empty rows are checked and dropped; singleton rows become column bounds;
//! - redundant rows (activity range inside the row range) are dropped;
//! - forcing rows (activity range touching the row range at one end) fix their columns;
//! - empty columns are fixed at their best bound by cost;
//! - dual fixing: a column whose cost and every row push the same way is fixed at that
//!   bound (keeps at least one optimal solution);
//! - parallel rows (found by hashing the normalized row, not pairwise) are merged;
//! - doubleton equations `a x_s + b x_t = rhs` substitute `x_s` out (HiGHS
//!   `doubletonEq`: a continuous `x_s` preferred; two integers only when `b/a` and
//!   `rhs/a` are integral), `x_s`'s bounds becoming implied bounds on `x_t`.
//! - the aggregator (HiGHS `aggregator`): an implied-free column of an equality row
//!   (the row alone keeps it within its bounds) is substituted out, subject to
//!   HiGHS's pivot threshold (0.01 of the row's largest coefficient) and fill-in limit
//!   (10); an integer column only when every other column of the row is integer and
//!   every ratio and the right-hand side divided by its coefficient are integral.
//!
//! - bound propagation: integer bounds tightened from row activities; continuous
//!   columns keep implied bounds that feed only the aggregator's implied-free test;
//! - dominated columns (HiGHS `detectDominatedCol`): the reduced cost is bounded over a
//!   box of row duals (row signs, tightened one step by the dual constraints of continuous
//!   columns with an infinite bound) and a (weakly) dominated column is fixed; see
//!   `Work::dominated_cols` for why this is valid for a MILP.
//!
//! Postsolve replays a stack of fixings and substitutions in reverse.
//!
//! Every step is exact up to `feastol` on the row activity, and the caller re-verifies
//! the postsolved point on the original rows before it may count toward a certificate.

use std::collections::HashMap;

const INF: f64 = f64::INFINITY;

/// A MILP in row form.
#[derive(Clone, Debug)]
pub struct RowForm {
    /// Columns.
    pub n: usize,
    /// Rows as `(columns, values)`.
    pub rows: Vec<(Vec<usize>, Vec<f64>)>,
    /// Row lower sides (`-inf` when absent).
    pub lo: Vec<f64>,
    /// Row upper sides (`+inf` when absent).
    pub up: Vec<f64>,
    /// Objective (minimize).
    pub c: Vec<f64>,
    /// Column lower bounds.
    pub l: Vec<f64>,
    /// Column upper bounds.
    pub u: Vec<f64>,
    /// Integrality.
    pub is_int: Vec<bool>,
}

/// Reduction counts.
#[derive(Clone, Debug, Default)]
pub struct PresolveStats {
    /// Fixed columns removed.
    pub fixed_cols: usize,
    /// Empty rows removed.
    pub empty_rows: usize,
    /// Singleton rows turned into bounds.
    pub singleton_rows: usize,
    /// Redundant rows removed.
    pub redundant_rows: usize,
    /// Forcing rows (their columns fixed).
    pub forcing_rows: usize,
    /// Empty columns fixed.
    pub empty_cols: usize,
    /// Columns fixed by dual arguments.
    pub dual_fixed: usize,
    /// Parallel rows merged.
    pub parallel_rows: usize,
    /// Doubleton equations substituted out.
    pub doubleton_eqs: usize,
    /// Implied-free columns aggregated out of equality rows.
    pub aggregated: usize,
    /// Integer bound tightenings by row-activity propagation.
    pub propagated: usize,
    /// Columns fixed as (weakly) dominated by row-dual bounds.
    pub dominated_cols: usize,
    /// Of those, fixings that needed a row-dual bound derived from a column.
    pub dominated_by_derived: usize,
    /// Passes over the rule set.
    pub passes: usize,
}

/// One postsolve step.
#[derive(Clone, Debug)]
pub enum PostOp {
    /// Column `j` fixed at `v`.
    Fix(usize, f64),
    /// `x_s = (rhs - b x_t) / a` (original indices).
    Subst {
        /// Substituted column.
        s: usize,
        /// Staying column.
        t: usize,
        /// Coefficient of `x_s`.
        a: f64,
        /// Coefficient of `x_t`.
        b: f64,
        /// Right-hand side.
        rhs: f64,
    },
    /// `x_s = (rhs - sum_k a_k x_k) / a` (original indices).
    Agg {
        /// Substituted column.
        s: usize,
        /// Its coefficient.
        a: f64,
        /// Right-hand side.
        rhs: f64,
        /// The row's other entries.
        terms: Vec<(usize, f64)>,
    },
}

/// The presolved problem and how to map its solutions back.
#[derive(Clone, Debug)]
pub struct Presolved {
    /// The reduced problem (columns renumbered `0..keep_cols.len()`).
    pub form: RowForm,
    /// Original index of each reduced column.
    pub keep_cols: Vec<usize>,
    /// Original index of each reduced row.
    pub keep_rows: Vec<usize>,
    /// Per reduced row, the constant moved out of its activity (fixed columns,
    /// substitutions): reduced `lo/up` = original `lo/up` minus this, unless a merge or
    /// tightening also changed the range.
    pub row_shift: Vec<f64>,
    /// Postsolve steps, in the order they were made.
    pub ops: Vec<PostOp>,
    /// Objective constant of the removed columns.
    pub offset: f64,
    /// Presolve proved the problem infeasible.
    pub infeasible: bool,
    /// Counts.
    pub stats: PresolveStats,
}

impl Presolved {
    /// Original-space point from a reduced-space point.
    pub fn postsolve(&self, x_red: &[f64], n_orig: usize) -> Vec<f64> {
        let mut x = vec![0.0; n_orig];
        for (k, &j) in self.keep_cols.iter().enumerate() {
            x[j] = x_red[k];
        }
        for op in self.ops.iter().rev() {
            match *op {
                PostOp::Fix(j, v) => x[j] = v,
                PostOp::Subst { s, t, a, b, rhs } => x[s] = (rhs - b * x[t]) / a,
                PostOp::Agg {
                    s,
                    a,
                    rhs,
                    ref terms,
                } => {
                    x[s] = (rhs - terms.iter().map(|&(k, v)| v * x[k]).sum::<f64>()) / a;
                }
            }
        }
        x
    }
}

struct Work {
    n: usize,
    rows: Vec<(Vec<usize>, Vec<f64>)>,
    lo: Vec<f64>,
    up: Vec<f64>,
    shift: Vec<f64>,
    row_alive: Vec<bool>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    is_int: Vec<bool>,
    col_alive: Vec<bool>,
    /// Implied lower/upper bounds of continuous columns with the row implying them
    /// (HiGHS `implColLower/Upper` + source). Never written into `l/u`: they only feed
    /// the aggregator's implied-free test, whose bound drops make the reduced problem a
    /// relaxation the postsolve check guards.
    impl_l: Vec<(f64, usize)>,
    impl_u: Vec<(f64, usize)>,
    /// Rows each column appears in (may include dead rows).
    col_rows: Vec<Vec<usize>>,
    ops: Vec<PostOp>,
    offset: f64,
    infeasible: bool,
    feastol: f64,
    stats: PresolveStats,
}

impl Work {
    fn activity(&self, r: usize) -> (f64, usize, f64, usize) {
        let (ri, rv) = &self.rows[r];
        let (mut mn, mut nmn, mut mx, mut nmx) = (0.0, 0usize, 0.0, 0usize);
        for (&j, &a) in ri.iter().zip(rv) {
            let (lj, uj) = (self.l[j], self.u[j]);
            let (lo_c, hi_c) = if a > 0.0 {
                (a * lj, a * uj)
            } else {
                (a * uj, a * lj)
            };
            if lo_c.is_finite() {
                mn += lo_c;
            } else {
                nmn += 1;
            }
            if hi_c.is_finite() {
                mx += hi_c;
            } else {
                nmx += 1;
            }
        }
        (mn, nmn, mx, nmx)
    }

    /// Relative tolerance at `v`. An infinite `v` gets the absolute `feastol`: a
    /// tolerance of `inf` turns every comparison against it into `-inf < -inf`, which
    /// once let the aggregator take an unbounded implied range for an implied-free one.
    fn tol(&self, v: f64) -> f64 {
        if v.is_finite() {
            self.feastol * (1.0 + v.abs())
        } else {
            self.feastol
        }
    }

    fn fix_col(&mut self, j: usize, v: f64) {
        if !self.col_alive[j] {
            return;
        }
        let v = if self.is_int[j] { v.round() } else { v };
        self.col_alive[j] = false;
        self.ops.push(PostOp::Fix(j, v));
        self.offset += self.c[j] * v;
        for &r in &self.col_rows[j] {
            if !self.row_alive[r] {
                continue;
            }
            let (ri, rv) = &mut self.rows[r];
            if let Some(p) = ri.iter().position(|&q| q == j) {
                let a = rv[p];
                ri.swap_remove(p);
                rv.swap_remove(p);
                if self.lo[r].is_finite() {
                    self.lo[r] -= a * v;
                }
                if self.up[r].is_finite() {
                    self.up[r] -= a * v;
                }
                self.shift[r] += a * v;
            }
        }
    }

    fn tighten(&mut self, j: usize, lo: f64, up: f64) {
        let ft = self.feastol;
        let (mut lo, mut up) = (lo, up);
        if self.is_int[j] {
            if lo.is_finite() {
                lo = (lo - ft).ceil();
            }
            if up.is_finite() {
                up = (up + ft).floor();
            }
        }
        if lo > self.l[j] {
            self.l[j] = lo;
        }
        if up < self.u[j] {
            self.u[j] = up;
        }
        if self.l[j] > self.u[j] + self.tol(self.u[j]) {
            self.infeasible = true;
        } else if self.l[j] > self.u[j] {
            // Within tolerance: collapse to one value.
            let v = if self.is_int[j] {
                self.l[j].round()
            } else {
                0.5 * (self.l[j] + self.u[j])
            };
            self.l[j] = v;
            self.u[j] = v;
        }
    }

    /// `doubletonEq` on equality row `r` with exactly two entries. Returns whether it
    /// substituted.
    fn doubleton_eq(&mut self, r: usize) -> bool {
        let (ri, rv) = self.rows[r].clone();
        let rhs0 = self.up[r];
        let ((j1, a1), (j2, a2)) = ((ri[0], rv[0]), (ri[1], rv[1]));
        // Which column to substitute out: continuous first; among two continuous the
        // larger |coefficient| (better conditioned); two integers: the smaller |coef|.
        let (i1, i2) = (self.is_int[j1], self.is_int[j2]);
        let first = match (i1, i2) {
            (false, true) => true,
            (true, false) => false,
            (false, false) => a1.abs() >= a2.abs(),
            (true, true) => a1.abs() <= a2.abs(),
        };
        let (s, a, t, mut b) = if first {
            (j1, a1, j2, a2)
        } else {
            (j2, a2, j1, a1)
        };
        let mut rhs = rhs0;
        if self.is_int[s] {
            // Both integer: x_s = rhs/a - (b/a) x_t must be integral for integral x_t.
            let q = (b / a).round();
            if (q * a - b).abs() > 1e-12 * b.abs().max(1.0) {
                return false;
            }
            b = q * a;
            let qr = (rhs / a).round();
            if (qr * a - rhs).abs() > self.tol(rhs) {
                self.infeasible = true;
                return false;
            }
            rhs = qr * a;
        }
        // x_s's bounds as implied bounds on x_t: l_s <= (rhs - b x_t)/a <= u_s.
        let (ls, us) = (self.l[s], self.u[s]);
        let at_l = if ls.is_finite() {
            (rhs - a * ls) / b
        } else if (a / b) > 0.0 {
            INF
        } else {
            -INF
        };
        let at_u = if us.is_finite() {
            (rhs - a * us) / b
        } else if (a / b) > 0.0 {
            -INF
        } else {
            INF
        };
        let (tlo, tup) = if at_l <= at_u {
            (at_l, at_u)
        } else {
            (at_u, at_l)
        };
        self.tighten(t, tlo, tup);
        if self.infeasible {
            return false;
        }
        self.ops.push(PostOp::Subst { s, t, a, b, rhs });
        self.col_alive[s] = false;
        self.row_alive[r] = false;
        // x_s = rhs/a + f x_t, f = -b/a.
        let (c0, f) = (rhs / a, -b / a);
        self.offset += self.c[s] * c0;
        self.c[t] += self.c[s] * f;
        let srows = self.col_rows[s].clone();
        for q in srows {
            if !self.row_alive[q] {
                continue;
            }
            let Some(p) = self.rows[q].0.iter().position(|&k| k == s) else {
                continue;
            };
            let coef = self.rows[q].1[p];
            self.rows[q].0.swap_remove(p);
            self.rows[q].1.swap_remove(p);
            if self.lo[q].is_finite() {
                self.lo[q] -= coef * c0;
            }
            if self.up[q].is_finite() {
                self.up[q] -= coef * c0;
            }
            self.shift[q] += coef * c0;
            let add = coef * f;
            if let Some(pt) = self.rows[q].0.iter().position(|&k| k == t) {
                self.rows[q].1[pt] += add;
                if self.rows[q].1[pt].abs() <= 1e-12 {
                    self.rows[q].0.swap_remove(pt);
                    self.rows[q].1.swap_remove(pt);
                }
            } else if add != 0.0 {
                self.rows[q].0.push(t);
                self.rows[q].1.push(add);
                self.col_rows[t].push(q);
            }
        }
        self.stats.doubleton_eqs += 1;
        true
    }

    /// HiGHS `aggregator` on equality row `r`: substitute out an implied-free column.
    fn aggregate(&mut self, r: usize) -> bool {
        const PIVOT_THRESHOLD: f64 = 0.01;
        const MAX_FILLIN: i64 = 10;
        const MAX_ROW_LEN: usize = 16;
        let (ri, rv) = self.rows[r].clone();
        if ri.len() < 3 || ri.len() > MAX_ROW_LEN {
            return false;
        }
        let rhs = self.up[r];
        let maxabs = rv.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let all_int = ri.iter().all(|&k| self.is_int[k]);
        // Candidate order: continuous columns first, then by column length.
        let mut cand: Vec<usize> = (0..ri.len()).collect();
        cand.sort_by_key(|&p| (self.is_int[ri[p]], self.col_rows[ri[p]].len()));
        for p in cand {
            let (s, a) = (ri[p], rv[p]);
            if a.abs() < PIVOT_THRESHOLD * maxabs {
                continue;
            }
            if self.is_int[s] {
                // Implied integral: every other column integer, every ratio integral.
                if !all_int {
                    continue;
                }
                let ok = rv
                    .iter()
                    .all(|&v| ((v / a) - (v / a).round()).abs() <= 1e-12)
                    && ((rhs / a) - (rhs / a).round()).abs() <= 1e-12;
                if !ok {
                    continue;
                }
            }
            // Implied free (HiGHS `isImpliedFree`): the rows containing x_s, with the
            // other columns' bounds (and their implied bounds from other rows), keep x_s
            // within [l_s, u_s]. Implications can chain (as in HiGHS), so a wrong "implied"
            // is possible in principle; it only drops x_s's bounds, which makes the reduced
            // problem a relaxation: its infeasibility and (verified) optimum carry over, and
            // `solve_presolved` re-checks the postsolved point on the original rows and
            // bounds and falls back to the original problem when it fails or when the
            // reduced problem is unbounded.
            let (mut il, mut iu) = (f64::NEG_INFINITY, f64::INFINITY);
            for &q in &self.col_rows[s] {
                if !self.row_alive[q] {
                    continue;
                }
                let Some(pq) = self.rows[q].0.iter().position(|&k| k == s) else {
                    continue;
                };
                let aq = self.rows[q].1[pq];
                let (mut rmin, mut rmax) = (0.0f64, 0.0f64);
                let (mut nmin, mut nmax) = (0usize, 0usize);
                for (&k, &v) in self.rows[q].0.iter().zip(&self.rows[q].1) {
                    if k == s {
                        continue;
                    }
                    // Other columns' implied bounds, except those this row implies.
                    let lk = if self.impl_l[k].1 != q {
                        self.l[k].max(self.impl_l[k].0)
                    } else {
                        self.l[k]
                    };
                    let uk = if self.impl_u[k].1 != q {
                        self.u[k].min(self.impl_u[k].0)
                    } else {
                        self.u[k]
                    };
                    let (lo_c, hi_c) = if v > 0.0 {
                        (v * lk, v * uk)
                    } else {
                        (v * uk, v * lk)
                    };
                    if lo_c.is_finite() {
                        rmin += lo_c
                    } else {
                        nmin += 1
                    }
                    if hi_c.is_finite() {
                        rmax += hi_c
                    } else {
                        nmax += 1
                    }
                }
                // aq x_s in [lo_q - rest_max, up_q - rest_min].
                let low = if self.lo[q].is_finite() && nmax == 0 {
                    self.lo[q] - rmax
                } else {
                    f64::NEG_INFINITY
                };
                let high = if self.up[q].is_finite() && nmin == 0 {
                    self.up[q] - rmin
                } else {
                    f64::INFINITY
                };
                let (b1, b2) = if aq > 0.0 {
                    (low / aq, high / aq)
                } else {
                    (high / aq, low / aq)
                };
                if b1.is_finite() || b1 == f64::INFINITY {
                    il = il.max(b1);
                }
                if b2.is_finite() || b2 == f64::NEG_INFINITY {
                    iu = iu.min(b2);
                }
            }
            // Tolerances at the column's own bounds, never at the implied ones (which may
            // be infinite).
            let tl = self.tol(self.l[s]);
            let tu = self.tol(self.u[s]);
            if (self.l[s].is_finite() && (il < self.l[s] - tl || il.is_nan()))
                || (self.u[s].is_finite() && (iu > self.u[s] + tu || iu.is_nan()))
            {
                continue;
            }
            // Fill-in: new entries in the other rows of x_s.
            let mut fill: i64 = -((ri.len() + self.col_rows[s].len()) as i64 - 1);
            for &q in &self.col_rows[s] {
                if q == r || !self.row_alive[q] {
                    continue;
                }
                for &k in &ri {
                    if k != s && !self.rows[q].0.contains(&k) {
                        fill += 1;
                    }
                }
            }
            if fill > MAX_FILLIN {
                continue;
            }
            // Substitute x_s = rhs/a - sum_k (a_k/a) x_k.
            let terms: Vec<(usize, f64)> = ri
                .iter()
                .zip(&rv)
                .filter(|(&k, _)| k != s)
                .map(|(&k, &v)| (k, v))
                .collect();
            self.ops.push(PostOp::Agg {
                s,
                a,
                rhs,
                terms: terms.clone(),
            });
            self.col_alive[s] = false;
            self.row_alive[r] = false;
            let c0 = rhs / a;
            self.offset += self.c[s] * c0;
            for &(k, v) in &terms {
                self.c[k] -= self.c[s] * v / a;
            }
            let srows = self.col_rows[s].clone();
            for q in srows {
                if !self.row_alive[q] {
                    continue;
                }
                let Some(pq) = self.rows[q].0.iter().position(|&k| k == s) else {
                    continue;
                };
                let coef = self.rows[q].1[pq];
                self.rows[q].0.swap_remove(pq);
                self.rows[q].1.swap_remove(pq);
                if self.lo[q].is_finite() {
                    self.lo[q] -= coef * c0;
                }
                if self.up[q].is_finite() {
                    self.up[q] -= coef * c0;
                }
                self.shift[q] += coef * c0;
                for &(k, v) in &terms {
                    let add = -coef * v / a;
                    if let Some(pk) = self.rows[q].0.iter().position(|&j| j == k) {
                        self.rows[q].1[pk] += add;
                        if self.rows[q].1[pk].abs() <= 1e-12 {
                            self.rows[q].0.swap_remove(pk);
                            self.rows[q].1.swap_remove(pk);
                        }
                    } else if add != 0.0 {
                        self.rows[q].0.push(k);
                        self.rows[q].1.push(add);
                        self.col_rows[k].push(q);
                    }
                }
            }
            self.stats.aggregated += 1;
            return true;
        }
        false
    }

    /// Bound propagation over the rows (HiGHS presolve's activity-based bound
    /// tightening): integer columns get their bounds tightened (rounded with HiGHS's
    /// `1e-6` integrality slack, so only points violating a row are removed); continuous
    /// columns record implied bounds in `impl_l/impl_u` and keep their bounds.
    fn propagate(&mut self) -> bool {
        const INT_SLACK: f64 = 1e-6;
        let mut changed = false;
        for j in 0..self.n {
            self.impl_l[j] = (-INF, usize::MAX);
            self.impl_u[j] = (INF, usize::MAX);
        }
        for r in 0..self.rows.len() {
            if !self.row_alive[r] || self.rows[r].0.len() < 2 {
                continue;
            }
            let (mn, nmn, mx, nmx) = self.activity(r);
            if nmn > 1 && nmx > 1 {
                continue;
            }
            let (lo, up) = (self.lo[r], self.up[r]);
            let (ri, rv) = self.rows[r].clone();
            for (&j, &a) in ri.iter().zip(&rv) {
                if a.abs() < 1e-9 || !self.col_alive[j] {
                    continue;
                }
                let (lj, uj) = (self.l[j], self.u[j]);
                let (cmin, cmax) = if a > 0.0 {
                    (a * lj, a * uj)
                } else {
                    (a * uj, a * lj)
                };
                // Residual activity ranges without column j.
                let rmin = if cmin.is_finite() {
                    if nmn == 0 {
                        Some(mn - cmin)
                    } else {
                        None
                    }
                } else if nmn == 1 {
                    Some(mn)
                } else {
                    None
                };
                let rmax = if cmax.is_finite() {
                    if nmx == 0 {
                        Some(mx - cmax)
                    } else {
                        None
                    }
                } else if nmx == 1 {
                    Some(mx)
                } else {
                    None
                };
                // a x_j <= up - rmin and a x_j >= lo - rmax.
                let hi_ax = match rmin {
                    Some(v) if up.is_finite() => up - v,
                    _ => INF,
                };
                let lo_ax = match rmax {
                    Some(v) if lo.is_finite() => lo - v,
                    _ => -INF,
                };
                let (bl, bu) = if a > 0.0 {
                    (lo_ax / a, hi_ax / a)
                } else {
                    (hi_ax / a, lo_ax / a)
                };
                let bl = if bl.is_nan() { -INF } else { bl };
                let bu = if bu.is_nan() { INF } else { bu };
                if self.is_int[j] {
                    let nl = if bl.is_finite() {
                        (bl - INT_SLACK * (1.0 + bl.abs())).ceil()
                    } else {
                        -INF
                    };
                    let nu = if bu.is_finite() {
                        (bu + INT_SLACK * (1.0 + bu.abs())).floor()
                    } else {
                        INF
                    };
                    // Integral steps only; huge implied values are left alone.
                    let tl = nl > self.l[j] && nl.abs() < 1e9;
                    let tu = nu < self.u[j] && nu.abs() < 1e9;
                    if tl || tu {
                        self.tighten(
                            j,
                            if tl { nl } else { self.l[j] },
                            if tu { nu } else { self.u[j] },
                        );
                        self.stats.propagated += 1;
                        changed = true;
                        if self.infeasible {
                            return false;
                        }
                    }
                } else {
                    if bl > self.impl_l[j].0 {
                        self.impl_l[j] = (bl, r);
                    }
                    if bu < self.impl_u[j].0 {
                        self.impl_u[j] = (bu, r);
                    }
                }
            }
        }
        changed
    }

    /// Row-dual bounds from the row sense alone (minimization, reduced cost
    /// `z = c - A^T y`): a `>=` row has `y >= 0`, a `<=` row `y <= 0`, a ranged or
    /// equality row a free `y`.
    fn sign_dual_bounds(&self, r: usize) -> (f64, f64) {
        match (self.lo[r].is_finite(), self.up[r].is_finite()) {
            (true, false) => (0.0, INF),
            (false, true) => (-INF, 0.0),
            _ => (-INF, INF),
        }
    }

    /// Dominated columns (HiGHS `detectDominatedCol`, strong and weak): bound the
    /// reduced cost `z_j = c_j - a_j^T y` over a box of row duals and fix `x_j` at the
    /// bound the sign of `z_j` points to.
    ///
    /// Why this is valid for a MILP. Take an optimal `x*`, fix the integer columns other
    /// than `j` at their `x*` values and move `x_j` by `delta` toward `l_j`; the remaining
    /// LP over the continuous columns has value function `v`, and LP duality gives
    /// `v(b - delta a_j) <= v(b) - delta min_{y in D} a_j^T y` over its dual region `D`
    /// (bounded in that direction means primal feasible). So the objective changes by at
    /// most `-delta (c_j - max_{y in D} a_j^T y)`, which is `<= 0` when the lower bound
    /// on `z_j` over a box containing `D` is `>= 0`. The box must therefore contain the
    /// dual region of the LP *without* column `j`:
    /// - row signs bound every `y`;
    /// - a continuous column `k != j` whose upper (lower) bound is infinite adds the dual
    ///   constraint `a_k^T y <= c_k` (`>= c_k`), from which a bound on one `y_i` follows
    ///   using only the sign bounds of the other rows (one step, so no derived bound
    ///   rests on a constraint of `j`);
    /// - integer columns contribute nothing (their duals are not constraints of the LP).
    ///
    /// Each derived bound records its source column; for `j` it falls back to the sign
    /// bound when the source is `j`, and fixing `j` drops the bounds it sourced.
    fn dominated_cols(&mut self) -> bool {
        let m = self.rows.len();
        // Best derived bound per row side, with its source column.
        let mut ylo = vec![(-INF, usize::MAX); m];
        let mut yup = vec![(INF, usize::MAX); m];
        for r in 0..m {
            if self.row_alive[r] {
                let (a, b) = self.sign_dual_bounds(r);
                ylo[r].0 = a;
                yup[r].0 = b;
            }
        }
        // Column entries over alive rows.
        let col_entries = |w: &Work, k: usize| -> Vec<(usize, f64)> {
            let mut e = Vec::new();
            for &q in &w.col_rows[k] {
                if !w.row_alive[q] {
                    continue;
                }
                if let Some(p) = w.rows[q].0.iter().position(|&t| t == k) {
                    e.push((q, w.rows[q].1[p]));
                }
            }
            e
        };
        for k in 0..self.n {
            if !self.col_alive[k] || self.is_int[k] {
                continue;
            }
            let ek = col_entries(self, k);
            if ek.is_empty() {
                continue;
            }
            // Dual constraints: `sense = +1` for `a^T y <= c` (upper bound infinite),
            // `-1` for `a^T y >= c` (lower bound infinite), i.e. `sense a^T y <= sense c`.
            for sense in [1.0f64, -1.0] {
                let unbounded = if sense > 0.0 {
                    self.u[k] == INF
                } else {
                    self.l[k] == -INF
                };
                if !unbounded {
                    continue;
                }
                // Minimum of each term sense*a_i*y_i over the sign box.
                let term_min = |q: usize, a: f64| -> f64 {
                    let (lo, up) = self.sign_dual_bounds(q);
                    let s = sense * a;
                    let v = if s > 0.0 { s * lo } else { s * up };
                    if v.is_nan() {
                        0.0
                    } else {
                        v
                    }
                };
                let mins: Vec<f64> = ek.iter().map(|&(q, a)| term_min(q, a)).collect();
                let ninf = mins.iter().filter(|v| v.is_infinite()).count();
                if ninf > 1 {
                    continue;
                }
                let finite_sum: f64 = mins.iter().filter(|v| v.is_finite()).sum();
                let rhs = sense * self.c[k];
                for (p, &(q, a)) in ek.iter().enumerate() {
                    // sense*a*y_q <= rhs - sum_{others} min.
                    let rest = if mins[p].is_infinite() {
                        finite_sum
                    } else if ninf == 0 {
                        finite_sum - mins[p]
                    } else {
                        continue;
                    };
                    let s = sense * a;
                    let bound = (rhs - rest) / s;
                    // Slack against rounding: loosen by a relative 1e-9.
                    let slack = 1e-9 * (1.0 + bound.abs());
                    if s > 0.0 {
                        let b = bound + slack;
                        if b < yup[q].0 {
                            yup[q] = (b, k);
                        }
                    } else {
                        let b = bound - slack;
                        if b > ylo[q].0 {
                            ylo[q] = (b, k);
                        }
                    }
                }
            }
        }
        let mut changed = false;
        for j in 0..self.n {
            if !self.col_alive[j] {
                continue;
            }
            let ej = col_entries(self, j);
            if ej.is_empty() {
                continue;
            }
            // z_j lower = c_j - max a^T y; upper = c_j - min a^T y.
            let (mut amax, mut amin) = (0.0f64, 0.0f64);
            // The same sums over the sign box alone (to count what the derived bounds add).
            let (mut smax, mut smin) = (0.0f64, 0.0f64);
            for &(q, a) in &ej {
                let (sl, su) = self.sign_dual_bounds(q);
                let (mx, mn) = if a > 0.0 {
                    (a * su, a * sl)
                } else {
                    (a * sl, a * su)
                };
                smax += mx;
                smin += mn;
                let lo = if ylo[q].1 == j { sl } else { ylo[q].0 };
                let up = if yup[q].1 == j { su } else { yup[q].0 };
                let (mx, mn) = if a > 0.0 {
                    (a * up, a * lo)
                } else {
                    (a * lo, a * up)
                };
                amax += if mx.is_nan() { 0.0 } else { mx };
                amin += if mn.is_nan() { 0.0 } else { mn };
            }
            let zlo = self.c[j] - amax;
            let zup = self.c[j] - amin;
            const TOL: f64 = 1e-9;
            let target = if zlo >= -TOL * (1.0 + self.c[j].abs()) && self.l[j].is_finite() {
                Some(self.l[j])
            } else if zup <= TOL * (1.0 + self.c[j].abs()) && self.u[j].is_finite() {
                Some(self.u[j])
            } else {
                None
            };
            if let Some(v) = target {
                let tol = TOL * (1.0 + self.c[j].abs());
                if !(self.c[j] - smax >= -tol || self.c[j] - smin <= tol) {
                    self.stats.dominated_by_derived += 1;
                }
                self.fix_col(j, v);
                self.stats.dominated_cols += 1;
                changed = true;
                // Bounds sourced by j no longer hold for the problem without j.
                for q in 0..m {
                    if ylo[q].1 == j {
                        ylo[q] = (self.sign_dual_bounds(q).0, usize::MAX);
                    }
                    if yup[q].1 == j {
                        yup[q] = (self.sign_dual_bounds(q).1, usize::MAX);
                    }
                }
            }
        }
        changed
    }

    fn pass(&mut self) -> bool {
        let mut changed = false;
        // Fixed columns.
        for j in 0..self.n {
            if self.col_alive[j] && self.l[j] == self.u[j] {
                let v = self.l[j];
                self.fix_col(j, v);
                self.stats.fixed_cols += 1;
                changed = true;
            }
        }
        if self.propagate() {
            changed = true;
        }
        if self.infeasible {
            return false;
        }
        // Rows: empty, singleton, redundant, forcing.
        for r in 0..self.rows.len() {
            if !self.row_alive[r] || self.infeasible {
                continue;
            }
            let len = self.rows[r].0.len();
            if len == 0 {
                if 0.0 < self.lo[r] - self.tol(self.lo[r])
                    || 0.0 > self.up[r] + self.tol(self.up[r])
                {
                    self.infeasible = true;
                    return false;
                }
                self.row_alive[r] = false;
                self.stats.empty_rows += 1;
                changed = true;
                continue;
            }
            if len == 1 {
                let (j, a) = (self.rows[r].0[0], self.rows[r].1[0]);
                let (lo, up) = (self.lo[r] / a, self.up[r] / a);
                let (blo, bup) = if a > 0.0 { (lo, up) } else { (up, lo) };
                self.tighten(j, blo, bup);
                self.row_alive[r] = false;
                self.stats.singleton_rows += 1;
                changed = true;
                continue;
            }
            if len == 2 && self.lo[r] == self.up[r] && self.doubleton_eq(r) {
                changed = true;
                continue;
            }
            if len >= 3 && self.lo[r] == self.up[r] && self.aggregate(r) {
                changed = true;
                continue;
            }
            if self.infeasible {
                return false;
            }
            let (mn, nmn, mx, nmx) = self.activity(r);
            let (lo, up) = (self.lo[r], self.up[r]);
            if (nmn == 0 && up.is_finite() && mn > up + self.tol(up))
                || (nmx == 0 && lo.is_finite() && mx < lo - self.tol(lo))
            {
                self.infeasible = true;
                return false;
            }
            let lo_ok = !lo.is_finite() || (nmn == 0 && mn >= lo - self.tol(lo) * 1e-3);
            let up_ok = !up.is_finite() || (nmx == 0 && mx <= up + self.tol(up) * 1e-3);
            if lo_ok && up_ok {
                self.row_alive[r] = false;
                self.stats.redundant_rows += 1;
                changed = true;
                continue;
            }
            // Forcing: the row can only be met at one end of its activity range.
            let force_max = nmx == 0 && lo.is_finite() && mx <= lo + self.tol(lo) * 1e-3;
            let force_min = nmn == 0 && up.is_finite() && mn >= up - self.tol(up) * 1e-3;
            if force_max || force_min {
                let (ri, rv) = self.rows[r].clone();
                for (&j, &a) in ri.iter().zip(&rv) {
                    let at_upper = (a > 0.0) == force_max;
                    let v = if at_upper { self.u[j] } else { self.l[j] };
                    self.l[j] = v;
                    self.u[j] = v;
                }
                self.row_alive[r] = false;
                self.stats.forcing_rows += 1;
                changed = true;
            }
        }
        if self.infeasible {
            return false;
        }
        // Columns: empty, dual fixing.
        for j in 0..self.n {
            if !self.col_alive[j] {
                continue;
            }
            // Directions in which the rows allow x_j to move freely.
            let (mut down_ok, mut up_ok, mut empty) = (true, true, true);
            for &r in &self.col_rows[j] {
                if !self.row_alive[r] {
                    continue;
                }
                let Some(p) = self.rows[r].0.iter().position(|&q| q == j) else {
                    continue;
                };
                empty = false;
                let a = self.rows[r].1[p];
                let (lo_f, up_f) = (self.lo[r].is_finite(), self.up[r].is_finite());
                // Decreasing x_j lowers a x when a > 0: safe iff the row has no lower side.
                if (a > 0.0 && lo_f) || (a < 0.0 && up_f) {
                    down_ok = false;
                }
                if (a > 0.0 && up_f) || (a < 0.0 && lo_f) {
                    up_ok = false;
                }
            }
            let cj = self.c[j];
            let target = if empty {
                if cj > 0.0 {
                    Some(self.l[j])
                } else if cj < 0.0 {
                    Some(self.u[j])
                } else if self.l[j].is_finite() {
                    Some(self.l[j])
                } else if self.u[j].is_finite() {
                    Some(self.u[j])
                } else {
                    Some(0.0)
                }
            } else if cj >= 0.0 && down_ok {
                Some(self.l[j])
            } else if cj <= 0.0 && up_ok {
                Some(self.u[j])
            } else {
                None
            };
            if let Some(v) = target {
                // An infinite target means the objective is unbounded in that direction
                // (or the column is free and unused): leave it to the search.
                if v.is_finite() {
                    if empty {
                        self.stats.empty_cols += 1;
                    } else {
                        self.stats.dual_fixed += 1;
                    }
                    self.fix_col(j, v);
                    changed = true;
                }
            }
        }
        if self.dominated_cols() {
            changed = true;
        }
        // Parallel rows: hash the row normalized by its first coefficient.
        let mut seen: HashMap<Vec<(usize, i64)>, usize> = HashMap::new();
        for r in 0..self.rows.len() {
            if !self.row_alive[r] || self.rows[r].0.len() < 2 {
                continue;
            }
            let mut ent: Vec<(usize, f64)> = self.rows[r]
                .0
                .iter()
                .copied()
                .zip(self.rows[r].1.iter().copied())
                .collect();
            ent.sort_by_key(|e| e.0);
            let s = ent[0].1;
            let key: Vec<(usize, i64)> = ent
                .iter()
                .map(|&(j, a)| (j, ((a / s) * 1e9).round() as i64))
                .collect();
            if let Some(&r0) = seen.get(&key) {
                // r = lambda r0 exactly (checked entrywise, not just by hash).
                let mut e0: Vec<(usize, f64)> = self.rows[r0]
                    .0
                    .iter()
                    .copied()
                    .zip(self.rows[r0].1.iter().copied())
                    .collect();
                e0.sort_by_key(|e| e.0);
                let lam = s / e0[0].1;
                let exact = e0.len() == ent.len()
                    && e0.iter().zip(&ent).all(|(&(j0, a0), &(j1, a1))| {
                        j0 == j1 && (a1 - lam * a0).abs() <= 1e-12 * a1.abs().max(1.0)
                    });
                if !exact {
                    continue;
                }
                // lo_r <= lam a0 x <= up_r  =>  a0 x in [lo_r, up_r] / lam.
                let (a, b) = (self.lo[r] / lam, self.up[r] / lam);
                let (nlo, nup) = if lam > 0.0 { (a, b) } else { (b, a) };
                let nlo = if nlo.is_nan() { -INF } else { nlo };
                let nup = if nup.is_nan() { INF } else { nup };
                self.lo[r0] = self.lo[r0].max(nlo);
                self.up[r0] = self.up[r0].min(nup);
                if self.lo[r0] > self.up[r0] + self.tol(self.up[r0]) {
                    self.infeasible = true;
                    return false;
                }
                self.row_alive[r] = false;
                self.stats.parallel_rows += 1;
                changed = true;
            } else {
                seen.insert(key, r);
            }
        }
        changed
    }
}

/// Presolve `f` with feasibility tolerance `feastol`, running the rule set to a
/// fixpoint (at most `max_passes` passes).
pub fn presolve(f: &RowForm, feastol: f64, max_passes: usize) -> Presolved {
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); f.n];
    for (r, (ri, _)) in f.rows.iter().enumerate() {
        for &j in ri {
            col_rows[j].push(r);
        }
    }
    let mut w = Work {
        n: f.n,
        rows: f.rows.clone(),
        lo: f.lo.clone(),
        up: f.up.clone(),
        shift: vec![0.0; f.rows.len()],
        row_alive: vec![true; f.rows.len()],
        c: f.c.clone(),
        l: f.l.clone(),
        u: f.u.clone(),
        is_int: f.is_int.clone(),
        col_alive: vec![true; f.n],
        impl_l: vec![(-INF, usize::MAX); f.n],
        impl_u: vec![(INF, usize::MAX); f.n],
        col_rows,
        ops: Vec::new(),
        offset: 0.0,
        infeasible: false,
        feastol,
        stats: PresolveStats::default(),
    };
    // Integer bounds rounded inward first.
    for j in 0..f.n {
        if w.is_int[j] {
            w.tighten(j, w.l[j], w.u[j]);
        }
    }
    while !w.infeasible && w.stats.passes < max_passes {
        w.stats.passes += 1;
        if !w.pass() {
            break;
        }
    }
    let keep_cols: Vec<usize> = (0..f.n).filter(|&j| w.col_alive[j]).collect();
    let mut newidx = vec![usize::MAX; f.n];
    for (k, &j) in keep_cols.iter().enumerate() {
        newidx[j] = k;
    }
    let mut rows = Vec::new();
    let (mut lo, mut up) = (Vec::new(), Vec::new());
    let (mut keep_rows, mut row_shift) = (Vec::new(), Vec::new());
    for r in 0..w.rows.len() {
        if !w.row_alive[r] {
            continue;
        }
        keep_rows.push(r);
        row_shift.push(w.shift[r]);
        let (ri, rv) = &w.rows[r];
        rows.push((ri.iter().map(|&j| newidx[j]).collect(), rv.clone()));
        lo.push(w.lo[r]);
        up.push(w.up[r]);
    }
    let pick = |v: &Vec<f64>| keep_cols.iter().map(|&j| v[j]).collect::<Vec<f64>>();
    Presolved {
        form: RowForm {
            n: keep_cols.len(),
            rows,
            lo,
            up,
            c: pick(&w.c),
            l: pick(&w.l),
            u: pick(&w.u),
            is_int: keep_cols.iter().map(|&j| w.is_int[j]).collect(),
        },
        keep_cols,
        keep_rows,
        row_shift,
        ops: w.ops,
        offset: w.offset,
        infeasible: w.infeasible,
        stats: w.stats,
    }
}

/// Whether `x` satisfies `f`'s rows, bounds and integrality within `tol` (relative).
#[allow(clippy::needless_range_loop)]
pub fn row_form_feasible(f: &RowForm, x: &[f64], tol: f64) -> bool {
    for j in 0..f.n {
        let t = tol * (1.0 + x[j].abs());
        if x[j] < f.l[j] - t || x[j] > f.u[j] + t {
            return false;
        }
        if f.is_int[j] && (x[j] - x[j].round()).abs() > 1e-5 {
            return false;
        }
    }
    for (r, (ri, rv)) in f.rows.iter().enumerate() {
        let act: f64 = ri.iter().zip(rv).map(|(&j, &a)| a * x[j]).sum();
        let t = tol * (1.0 + act.abs());
        if act < f.lo[r] - t || act > f.up[r] + t {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn int(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i64
        }
    }

    /// Minimum of `f` by enumeration over its (integer, bounded) box.
    fn brute_min(f: &RowForm) -> Option<(f64, Vec<f64>)> {
        if f.n == 0 {
            let x: Vec<f64> = Vec::new();
            return row_form_feasible(f, &x, 1e-9).then(|| (0.0, x));
        }
        let mut x: Vec<f64> = f.l.clone();
        let mut best: Option<(f64, Vec<f64>)> = None;
        loop {
            if row_form_feasible(f, &x, 1e-9) {
                let v: f64 = f.c.iter().zip(&x).map(|(c, x)| c * x).sum();
                if best.as_ref().is_none_or(|b| v < b.0 - 1e-9) {
                    best = Some((v, x.clone()));
                }
            }
            let mut j = 0;
            while j < f.n {
                x[j] += 1.0;
                if x[j] <= f.u[j] {
                    break;
                }
                x[j] = f.l[j];
                j += 1;
            }
            if j == f.n {
                break;
            }
        }
        best
    }

    fn random_ip(r: &mut Lcg) -> RowForm {
        let n = r.int(1, 5) as usize;
        let m = r.int(0, 5) as usize;
        let l: Vec<f64> = (0..n).map(|_| r.int(-2, 1) as f64).collect();
        let u: Vec<f64> = l.iter().map(|&v| v + r.int(0, 3) as f64).collect();
        let c: Vec<f64> = (0..n).map(|_| r.int(-4, 4) as f64).collect();
        let mut rows = Vec::new();
        let (mut lo, mut up) = (Vec::new(), Vec::new());
        for _ in 0..m {
            let k = r.int(1, n as i64) as usize;
            let mut cols: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let s = r.int(i as i64, n as i64 - 1) as usize;
                cols.swap(i, s);
            }
            let ri: Vec<usize> = cols[..k].to_vec();
            let rv: Vec<f64> = ri
                .iter()
                .map(|_| {
                    let v = r.int(-3, 3);
                    if v == 0 {
                        1.0
                    } else {
                        v as f64
                    }
                })
                .collect();
            let mid = r.int(-4, 4) as f64;
            match r.int(0, 3) {
                0 => {
                    lo.push(-INF);
                    up.push(mid);
                }
                1 => {
                    lo.push(mid);
                    up.push(INF);
                }
                2 => {
                    lo.push(mid);
                    up.push(mid);
                }
                _ => {
                    lo.push(mid);
                    up.push(mid + r.int(0, 3) as f64);
                }
            }
            rows.push((ri.clone(), rv.clone()));
            if r.int(0, 4) == 0 {
                // A parallel copy, scaled.
                let s = [2.0, -1.0, 0.5, -3.0][r.int(0, 3) as usize];
                rows.push((ri, rv.iter().map(|v| v * s).collect()));
                let (a, b) = (lo[lo.len() - 1] * s, up[up.len() - 1] * s);
                if s > 0.0 {
                    lo.push(a);
                    up.push(b);
                } else {
                    lo.push(b);
                    up.push(a);
                }
            }
        }
        let mut f = RowForm {
            n,
            rows,
            lo,
            up,
            c,
            l,
            u,
            is_int: vec![true; n],
        };
        if n >= 3 && r.int(0, 2) == 0 {
            // An aggregation row: x_s = rhs - sum of unit terms, x_s's box wide enough for
            // the row alone to imply it (an implied-free, implied-integral column).
            let s = r.int(0, n as i64 - 1) as usize;
            let others: Vec<usize> = (0..n).filter(|&k| k != s).collect();
            let rv: Vec<f64> = others
                .iter()
                .map(|_| if r.int(0, 1) == 0 { 1.0 } else { -1.0 })
                .collect();
            let rhs = r.int(-2, 2) as f64;
            f.l[s] = -20.0;
            f.u[s] = 20.0;
            let mut ri = others.clone();
            ri.push(s);
            let mut vals = rv.clone();
            vals.push(1.0);
            f.rows.push((ri, vals));
            f.lo.push(rhs);
            f.up.push(rhs);
        }
        f
    }

    #[test]
    fn presolve_preserves_the_optimum_and_postsolve_is_feasible() {
        let mut r = Lcg(2024);
        let (mut cases, mut reduced, mut infeasible, mut feasible) =
            (0usize, 0usize, 0usize, 0usize);
        let mut doubletons = 0usize;
        let mut aggregated = 0usize;
        let mut propagated = 0usize;
        for _ in 0..3000 {
            let f = random_ip(&mut r);
            let want = brute_min(&f);
            let p = presolve(&f, 1e-9, 20);
            cases += 1;
            doubletons += p.stats.doubleton_eqs;
            aggregated += p.stats.aggregated;
            propagated += p.stats.propagated;
            if p.keep_cols.len() < f.n || p.form.rows.len() < f.rows.len() {
                reduced += 1;
            }
            if p.infeasible {
                infeasible += 1;
                assert!(
                    want.is_none(),
                    "presolve declared a feasible problem infeasible: {f:?}"
                );
                continue;
            }
            let got = brute_min(&p.form);
            match (&want, &got) {
                (None, None) => {}
                (Some((wv, _)), Some((gv, gx))) => {
                    feasible += 1;
                    assert!(
                        (wv - (gv + p.offset)).abs() <= 1e-9,
                        "optimum changed: {wv} vs {} for {f:?}",
                        gv + p.offset
                    );
                    let x = p.postsolve(gx, f.n);
                    assert!(
                        row_form_feasible(&f, &x, 1e-9),
                        "postsolved point infeasible: {x:?} for {f:?}"
                    );
                    let v: f64 = f.c.iter().zip(&x).map(|(c, x)| c * x).sum();
                    assert!((v - wv).abs() <= 1e-9);
                }
                _ => panic!("feasibility changed: original {want:?} reduced {got:?} for {f:?}"),
            }
        }
        assert!(
            cases == 3000
                && reduced > 1000
                && infeasible > 50
                && feasible > 500
                && doubletons > 50
                && aggregated > 20
                && propagated > 100,
            "probe fired too little: reduced {reduced} infeasible {infeasible} feasible {feasible} doubletons {doubletons} aggregated {aggregated} propagated {propagated}"
        );
    }
}
