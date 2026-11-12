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
//! - dominated column pairs (Achterberg et al. 2020 §6.4; Gamrath et al. 2015 Thm. 3;
//!   HiGHS `dominatedColumns`): fixings and integer bound tightenings; see
//!   `Work::dominated_pairs`.
//!
//! - parallel columns with proportional costs merged (`PostOp::Split` undoes it).
//!
//! - sparsify (HiGHS `sparsify`, opt-in via `presolve_opts(.., implied_int, sparsify = true)`):
//!   multiples of equality rows added to other rows to cancel nonzeros; see
//!   `Work::sparsify`.
//!
//! Postsolve replays a stack of fixings and substitutions in reverse.
//!
//! Every step is exact up to `feastol` on the row activity, and the caller re-verifies
//! the postsolved point on the original rows before it may count toward a certificate.

use std::collections::HashMap;

const INF: f64 = f64::INFINITY;

/// A coefficient at most this fraction of its row's largest is noise to the presolve
/// rules ([`Work::is_noise`]); and a singleton row's lone coefficient at most this in
/// absolute value is left alone.
const NOISE_REL: f64 = 1e-9;

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
    /// Rows redundant only over implied bounds of continuous columns.
    pub redundant_rows_implied: usize,
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
    /// Dominated column pairs that fixed or tightened a column.
    pub dominated_pairs: usize,
    /// Parallel columns merged.
    pub parallel_cols: usize,
    /// Continuous columns marked integral because an equality row implies it
    /// (HiGHS `isImpliedInteger`, primal detection).
    pub implied_int_primal: usize,
    /// Continuous columns marked integral because some optimum has them integral
    /// (HiGHS `isImpliedInteger`, dual detection).
    pub implied_int_dual: usize,
    /// Sparsify (`DISCOPT_MILP_SPARSIFY`): row additions `row += lambda * eq` made.
    pub sparsified_rows: usize,
    /// Sparsify: net nonzeros removed by those additions (cancelled minus fill-in;
    /// the singleton / doubleton clean-up after them is counted under its own rule).
    pub sparsify_nz_removed: usize,
    /// Sparsify: additions HiGHS would make that were refused because they leave a
    /// round-off entry that cannot be removed soundly (see `Work::sparsify`).
    pub sparsify_refused_roundoff: usize,
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
    /// Parallel columns merged: column `j` carried `x' = x_j + s x_k`; split it back
    /// within the original bounds (`x_k` integral when `k_int`, `x_j` when `j_int`).
    Split {
        /// Kept column (holds `x'`).
        j: usize,
        /// Removed column.
        k: usize,
        /// Scale: column `k` = `s` times column `j`.
        s: f64,
        /// Bounds of `x_j` before the merge.
        lj: f64,
        /// Upper bound of `x_j` before the merge.
        uj: f64,
        /// Bounds of `x_k`.
        lk: f64,
        /// Upper bound of `x_k`.
        uk: f64,
        /// Integrality of `x_j`.
        j_int: bool,
        /// Integrality of `x_k`.
        k_int: bool,
    },
}

/// Split `x'` into `(x_j, x_k)` with `x_j + s x_k = x'` inside the bounds (see
/// `PostOp::Split`); the merge conditions make such a split exist.
#[allow(clippy::too_many_arguments)]
fn split_parallel(
    xp: f64,
    s: f64,
    lj: f64,
    uj: f64,
    lk: f64,
    uk: f64,
    j_int: bool,
    k_int: bool,
) -> (f64, f64) {
    // x_k in [(x' - u_j)/s, (x' - l_j)/s] (ordered by the sign of s), within [l_k, u_k].
    let (a, b) = ((xp - uj) / s, (xp - lj) / s);
    let (mut lo, mut hi) = if s > 0.0 { (a, b) } else { (b, a) };
    lo = if lo.is_nan() { -INF } else { lo.max(lk) };
    hi = if hi.is_nan() { INF } else { hi.min(uk) };
    if k_int {
        lo = (lo - 1e-9).ceil();
        hi = (hi + 1e-9).floor();
    }
    let xk = if lo.is_finite() {
        lo
    } else if hi.is_finite() {
        hi
    } else {
        0.0f64.max(lk).min(uk)
    };
    let mut xj = xp - s * xk;
    if j_int {
        xj = xj.round();
    }
    (xj, xk)
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
                PostOp::Split {
                    j,
                    k,
                    s,
                    lj,
                    uj,
                    lk,
                    uk,
                    j_int,
                    k_int,
                } => {
                    let (xj, xk) = split_parallel(x[j], s, lj, uj, lk, uk, j_int, k_int);
                    x[j] = xj;
                    x[k] = xk;
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
    fn new(f: &RowForm, feastol: f64) -> Work {
        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); f.n];
        for (r, (ri, _)) in f.rows.iter().enumerate() {
            for &j in ri {
                col_rows[j].push(r);
            }
        }
        Work {
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
        }
    }

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

    /// Whether coefficient `a` of row `r` is noise no rule may divide by, pivot on or
    /// derive a bound through: at most `NOISE_REL` of the row's largest coefficient.
    ///
    /// discopt certifies under tolerance semantics (a point satisfying the rows within
    /// the feasibility tolerance is feasible, as for the oracle solvers), but a rule
    /// dividing by such a coefficient reasons in exact arithmetic: a substitution
    /// `x_s = (rhs - b x_t) / 4.4e-16` or a bound derived through it amplifies the
    /// tolerance (and every rounding in the other terms) by `1 / |a|`, and cut off
    /// tolerance-feasible points. Measured (`presolve_never_divides_by_a_noise_coefficient`):
    /// with one 1e-16-scale entry on an equality row, the reduced optimum came out 0
    /// against a true -1 -- a bound above the optimum -- and feasible problems were
    /// declared infeasible. Cancellation leaves exactly such entries on equality rows,
    /// where `cancel_entry` must keep them (an equality has no slack to relax), so the
    /// rules themselves refuse them. Refusing a reduction is always sound.
    fn is_noise(&self, r: usize, a: f64) -> bool {
        let mx = self.rows[r].1.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.abs() <= NOISE_REL * mx
    }

    /// Entry `pos` of row `q` has just been updated by a substitution and may have
    /// cancelled to round-off. An exact zero is removed. A nonzero `|v| <= 1e-12` is
    /// removed only from an inequality, whose sides are then relaxed by the term's reach
    /// `|v| max(|l_k|, |u_k|)` over the column's box; otherwise (an equality, which must
    /// stay one for the rebuild, or an unbounded column) it is kept.
    ///
    /// Removing it outright is not a relaxation: on the captured portfol_roundlot OA
    /// master (`x_j = n_j / a_j` aggregated, integers `n_j` up to 1e5) the leftovers of
    /// the cancellation in `aggregate` carried 4.6e-8 of activity, the reduced problem
    /// cut off a point verified feasible on the original rows, and the solve certified
    /// a bound 4.6e-8 above it (0.0282902567 vs 0.0282902109).
    fn cancel_entry(&mut self, q: usize, pos: usize) {
        let v = self.rows[q].1[pos];
        if v != 0.0 {
            if v.abs() > 1e-12 || self.lo[q] == self.up[q] {
                return;
            }
            let k = self.rows[q].0[pos];
            let reach = self.l[k].abs().max(self.u[k].abs());
            if !reach.is_finite() || reach >= INF {
                return;
            }
            let slack = v.abs() * reach * (1.0 + 1e-12) + f64::MIN_POSITIVE;
            if self.lo[q].is_finite() {
                self.lo[q] -= slack;
            }
            if self.up[q].is_finite() {
                self.up[q] += slack;
            }
        }
        self.rows[q].0.swap_remove(pos);
        self.rows[q].1.swap_remove(pos);
    }

    /// `doubletonEq` on equality row `r` with exactly two entries. Returns whether it
    /// substituted.
    fn doubleton_eq(&mut self, r: usize) -> bool {
        let (ri, rv) = self.rows[r].clone();
        let rhs0 = self.up[r];
        let ((j1, a1), (j2, a2)) = ((ri[0], rv[0]), (ri[1], rv[1]));
        if self.is_noise(r, a1) || self.is_noise(r, a2) {
            return false;
        }
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
                self.cancel_entry(q, pt);
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
        // HiGHS `presolve_pivot_threshold` and `presolve_substitution_maxfillin`.
        const PIVOT_THRESHOLD: f64 = 0.01;
        const MAX_FILLIN: i64 = 10;
        let (ri, rv) = self.rows[r].clone();
        if ri.len() < 3 {
            return false;
        }
        let rhs = self.up[r];
        let row_max = rv.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        // Candidate order: continuous columns first, then by column length.
        let mut cand: Vec<usize> = (0..ri.len()).collect();
        cand.sort_by_key(|&p| (self.is_int[ri[p]], self.col_rows[ri[p]].len()));
        for p in cand {
            let (s, a) = (ri[p], rv[p]);
            if self.is_noise(r, a) {
                continue;
            }
            // Pivot: large enough against the row's or the column's largest entry.
            if a.abs() < PIVOT_THRESHOLD * row_max {
                let col_max = self.col_rows[s]
                    .iter()
                    .filter(|&&q| self.row_alive[q])
                    .filter_map(|&q| {
                        let rq = &self.rows[q];
                        rq.0.iter().position(|&k| k == s).map(|pq| rq.1[pq].abs())
                    })
                    .fold(0.0f64, f64::max);
                if a.abs() < PIVOT_THRESHOLD * col_max {
                    continue;
                }
            }
            if self.is_int[s] && !self.implied_integral(s) {
                continue;
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
                if self.is_noise(q, aq) {
                    // Row q cannot bound x_s through a noise coefficient: it implies
                    // nothing here, so x_s is not shown implied free by it.
                    continue;
                }
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
                        self.cancel_entry(q, pk);
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

    /// HiGHS `detectImpliedIntegers` (`HPresolve.cpp:7261-7265`, with `isImpliedInteger`
    /// `:358-418`): mark continuous columns integral when integrality costs nothing.
    ///
    /// * Primal: an alive equality row has every other column integer with `a_k / a_s`
    ///   and `rhs / a_s` integral ([`Self::implied_integral`]), so `x_s` is integral at
    ///   every point where the integer columns are. Repeated to a fixpoint, as each
    ///   marked column can complete another's row.
    /// * Dual: every alive row of `x_s` has at least one other column, all of them
    ///   integer, with `a_k / a_s` and each finite side `/ a_s` integral, and `x_s`'s
    ///   bounds are integral. With the integer columns fixed, `x_s` then ranges over an
    ///   interval with integral ends and touches no other row, so an optimal value of
    ///   `x_s` is integral: some optimal solution has `x_s` integral.
    ///
    /// Two departures from HiGHS, both for soundness by construction. The dual test
    /// counts as integer only original and primal-detected columns, never another
    /// dual-detected one (HiGHS counts all), so no two dual-detected columns share a
    /// row and each interval argument stands alone. And integrality is tested at
    /// `1e-12` relative, not HiGHS's feasibility tolerance. It runs after the last
    /// pass, so no later rule rewrites the rows a dual detection relied on.
    fn detect_implied_integers(&mut self) {
        let integral = |v: f64| (v - v.round()).abs() <= 1e-12 * v.abs().max(1.0);
        loop {
            let mut changed = false;
            for s in 0..self.n {
                if !self.col_alive[s] || self.is_int[s] || !self.implied_integral(s) {
                    continue;
                }
                // Integral at every feasible point: the bounds round inward.
                let lo = if self.l[s].is_finite() {
                    (self.l[s] - 1e-9).ceil()
                } else {
                    self.l[s]
                };
                let up = if self.u[s].is_finite() {
                    (self.u[s] + 1e-9).floor()
                } else {
                    self.u[s]
                };
                if lo > up {
                    // A real infeasibility; left to the solve rather than claimed here.
                    continue;
                }
                self.l[s] = lo;
                self.u[s] = up;
                self.is_int[s] = true;
                self.stats.implied_int_primal += 1;
                changed = true;
            }
            if !changed {
                break;
            }
        }
        let base = self.is_int.clone();
        for s in 0..self.n {
            if !self.col_alive[s] || base[s] {
                continue;
            }
            let bound_ok = |v: f64| !v.is_finite() || integral(v);
            if !bound_ok(self.l[s]) || !bound_ok(self.u[s]) {
                continue;
            }
            let ok = self.col_rows[s].iter().all(|&q| {
                if !self.row_alive[q] {
                    return true;
                }
                let (qi, qv) = &self.rows[q];
                let Some(pq) = qi.iter().position(|&k| k == s) else {
                    return true;
                };
                let a = qv[pq];
                qi.len() >= 2
                    && a != 0.0
                    && (!self.lo[q].is_finite() || integral(self.lo[q] / a))
                    && (!self.up[q].is_finite() || integral(self.up[q] / a))
                    && qi
                        .iter()
                        .zip(qv)
                        .all(|(&k, &v)| k == s || (base[k] && integral(v / a)))
            });
            if ok {
                self.l[s] = self.l[s].round();
                self.u[s] = self.u[s].round();
                self.is_int[s] = true;
                self.stats.implied_int_dual += 1;
            }
        }
    }

    /// HiGHS `isImpliedIntegral`: some alive equality row of `x_s` has every other
    /// column integer with `a_k / a_s` and `rhs / a_s` integral, so `x_s` is integral
    /// whenever that row holds. Substituting `x_s` out through any row then drops a
    /// constraint the kept row still enforces (and the postsolve check re-verifies
    /// integrality on the original problem).
    fn implied_integral(&self, s: usize) -> bool {
        let integral = |v: f64| (v - v.round()).abs() <= 1e-12 * v.abs().max(1.0);
        self.col_rows[s].iter().any(|&q| {
            if !self.row_alive[q] || self.lo[q] != self.up[q] {
                return false;
            }
            let (qi, qv) = &self.rows[q];
            let Some(pq) = qi.iter().position(|&k| k == s) else {
                return false;
            };
            let a = qv[pq];
            integral(self.up[q] / a)
                && qi
                    .iter()
                    .zip(qv)
                    .all(|(&k, &v)| k == s || (self.is_int[k] && integral(v / a)))
        })
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
                if a.abs() < 1e-9 || self.is_noise(r, a) || !self.col_alive[j] {
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

    /// Residual activity range of row `r` without column `col`, with column `fixed.0`
    /// (if present in the row) counted at the value `fixed.1`. `None` for an infinite
    /// side.
    fn residual(
        &self,
        r: usize,
        col: usize,
        fixed: Option<(usize, f64)>,
    ) -> (Option<f64>, Option<f64>) {
        let (mut mn, mut mx) = (Some(0.0f64), Some(0.0f64));
        for (&k, &a) in self.rows[r].0.iter().zip(&self.rows[r].1) {
            if k == col {
                continue;
            }
            let (lk, uk) = match fixed {
                Some((f, v)) if f == k => (v, v),
                _ => (self.l[k], self.u[k]),
            };
            let (cmin, cmax) = if a > 0.0 {
                (a * lk, a * uk)
            } else {
                (a * uk, a * lk)
            };
            mn = mn.and_then(|s| {
                if cmin.is_finite() {
                    Some(s + cmin)
                } else {
                    None
                }
            });
            mx = mx.and_then(|s| {
                if cmax.is_finite() {
                    Some(s + cmax)
                } else {
                    None
                }
            });
        }
        (mn, mx)
    }

    /// Bounds on `x_col` from its rows with `fixed` substituted. `worst = false`: the
    /// implied range (every feasible point lies in it). `worst = true`: the range in
    /// which every row holds whatever the other columns are (residual taken at its
    /// least favourable end).
    fn col_range(&self, col: usize, fixed: Option<(usize, f64)>, worst: bool) -> (f64, f64) {
        let (mut lo, mut hi) = (-INF, INF);
        for &r in &self.col_rows[col] {
            if !self.row_alive[r] {
                continue;
            }
            let Some(p) = self.rows[r].0.iter().position(|&k| k == col) else {
                continue;
            };
            let a = self.rows[r].1[p];
            if self.is_noise(r, a) {
                // No bound through a noise coefficient (`is_noise`).
                continue;
            }
            let (rmin, rmax) = self.residual(r, col, fixed);
            // a x <= up - (rmin | rmax), a x >= lo - (rmax | rmin).
            let (ru, rl) = if worst { (rmax, rmin) } else { (rmin, rmax) };
            let ax_hi = match ru {
                Some(v) if self.up[r].is_finite() => self.up[r] - v,
                _ if self.up[r].is_finite() && worst => -INF,
                _ => INF,
            };
            let ax_lo = match rl {
                Some(v) if self.lo[r].is_finite() => self.lo[r] - v,
                _ if self.lo[r].is_finite() && worst => INF,
                _ => -INF,
            };
            let (bl, bu) = if a > 0.0 {
                (ax_lo / a, ax_hi / a)
            } else {
                (ax_hi / a, ax_lo / a)
            };
            if !bl.is_nan() {
                lo = lo.max(bl);
            }
            if !bu.is_nan() {
                hi = hi.min(bu);
            }
        }
        (lo, hi)
    }

    /// Whether `sj x_j` dominates `sk x_k` (`sj, sk` in `{1, -1}`): no larger cost, and
    /// in every row a coefficient that moves each finite side no worse (equal on rows
    /// with both sides finite). Then some optimal solution has `x_j` at its `sj`-upper
    /// bound or `x_k` at its `sk`-lower bound: moving `delta` from `sk x_k` to `sj x_j`
    /// changes each row by `delta (alpha - beta)` on the safe side and the objective by
    /// `delta (sj c_j - sk c_k) <= 0`, and keeps integrality since both have the same
    /// type.
    fn dominates(&self, j: usize, sj: f64, k: usize, sk: f64) -> bool {
        const EPS: f64 = 1e-9;
        if sj * self.c[j] > sk * self.c[k] + EPS {
            return false;
        }
        let entry = |c: usize, r: usize| -> f64 {
            self.rows[r]
                .0
                .iter()
                .position(|&q| q == c)
                .map_or(0.0, |p| self.rows[r].1[p])
        };
        for &(c, s) in &[(j, true), (k, false)] {
            for &r in &self.col_rows[c] {
                if !self.row_alive[r] {
                    continue;
                }
                // Rows holding both are checked once, from j's side.
                if !s && self.rows[r].0.contains(&j) {
                    continue;
                }
                let alpha = sj * entry(j, r);
                let beta = sk * entry(k, r);
                if self.up[r].is_finite() && alpha > beta + EPS {
                    return false;
                }
                if self.lo[r].is_finite() && alpha < beta - EPS {
                    return false;
                }
            }
        }
        true
    }

    /// Dominated columns, pairwise (Achterberg et al. 2020, section 6.4; Gamrath et
    /// al. 2015, Theorem 3, "predictive bound analysis"; HiGHS `dominatedColumns`).
    /// For `sj x_j` dominating `sk x_k`, with `J` the `sj`-upper bound of `x_j` and `K`
    /// the `sk`-lower bound of `x_k`, some optimal solution has `x_j = J` or `x_k = K`
    /// (see `dominates`). Hence, for every such solution:
    /// - `J` infinite: `x_k = K` (fixed);
    /// - `sj x_j >= min(sj J, implied lower of sj x_j given x_k = K)`;
    /// - `sk x_k <= max(sk K, implied upper of sk x_k given x_j = J)`;
    ///
    /// and one of them can be modified, keeping it optimal and of that form, to also have
    /// - if `sj c_j <= 0`: `sj x_j >= min(sj J, worst-case upper of sj x_j given x_k = K)`
    ///   (raising it there keeps every row whatever the others are);
    /// - if `sk c_k >= 0`: `sk x_k <= max(sk K, worst-case lower of sk x_k given x_j = J)`.
    ///
    /// Each step is argued on the current (already tightened) problem. As in HiGHS,
    /// bounds change only on integer columns (or when a column becomes fixed).
    /// Candidates `k` are drawn from `j`'s shortest row, as HiGHS does.
    fn dominated_pairs(&mut self) -> bool {
        const INT_SLACK: f64 = 1e-6;
        let mut changed = false;
        for j in 0..self.n {
            if !self.col_alive[j] || self.l[j] == self.u[j] {
                continue;
            }
            // Shortest alive row of j.
            let mut best: Option<usize> = None;
            for &r in &self.col_rows[j] {
                if self.row_alive[r]
                    && self.rows[r].0.contains(&j)
                    && best.is_none_or(|b| self.rows[r].0.len() < self.rows[b].0.len())
                {
                    best = Some(r);
                }
            }
            let Some(br) = best else { continue };
            let cands: Vec<usize> = self.rows[br].0.clone();
            for k in cands {
                if k == j || !self.col_alive[k] || !self.col_alive[j] {
                    continue;
                }
                if self.is_int[j] != self.is_int[k] {
                    continue;
                }
                for &(sj, sk) in &[(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    if !self.col_alive[k] || self.l[k] == self.u[k] || self.l[j] == self.u[j] {
                        break;
                    }
                    if !self.dominates(j, sj, k, sk) {
                        continue;
                    }
                    let jstar = if sj > 0.0 { self.u[j] } else { self.l[j] };
                    let kstar = if sk > 0.0 { self.l[k] } else { self.u[k] };
                    if !jstar.is_finite() {
                        if kstar.is_finite() {
                            self.l[k] = kstar;
                            self.u[k] = kstar;
                            self.stats.dominated_pairs += 1;
                            changed = true;
                        }
                        continue;
                    }
                    if !kstar.is_finite() {
                        continue;
                    }
                    // Tighten j away from J (its sj-lower side).
                    let (il, iu) = self.col_range(j, Some((k, kstar)), false);
                    let mut nb = if sj > 0.0 { il } else { -iu };
                    if sj * self.c[j] <= 0.0 {
                        let (wl, wu) = self.col_range(j, Some((k, kstar)), true);
                        let w = if sj > 0.0 { wu } else { -wl };
                        // The value x_j can be raised to: for an integer column the
                        // integer at or below it (rounding the other way would claim a
                        // value the rows forbid).
                        let w = if self.is_int[j] && w.is_finite() {
                            (w + INT_SLACK * (1.0 + w.abs())).floor()
                        } else {
                            w
                        };
                        if w > nb {
                            nb = w;
                        }
                    }
                    let nb = nb.min(sj * jstar);
                    let jfix = self.set_side_bound(j, sj, nb, INT_SLACK);
                    // Tighten k away from K (its sk-upper side).
                    let (il, iu) = self.col_range(k, Some((j, jstar)), false);
                    let mut nb = if sk > 0.0 { iu } else { -il };
                    if sk * self.c[k] >= 0.0 {
                        let (wl, wu) = self.col_range(k, Some((j, jstar)), true);
                        let w = if sk > 0.0 { wl } else { -wu };
                        let w = if self.is_int[k] && w.is_finite() {
                            (w - INT_SLACK * (1.0 + w.abs())).ceil()
                        } else {
                            w
                        };
                        if w < nb {
                            nb = w;
                        }
                    }
                    let nb = nb.max(sk * kstar);
                    let kfix = self.set_side_bound_upper(k, sk, nb, INT_SLACK);
                    if jfix || kfix {
                        self.stats.dominated_pairs += 1;
                        changed = true;
                    }
                    if self.infeasible {
                        return false;
                    }
                }
            }
        }
        changed
    }

    /// `s x_j >= nb` (`s` in `{1, -1}`): applied to an integer column (rounded with the
    /// integrality slack), or to any column when it fixes it. Returns whether it applied.
    fn set_side_bound(&mut self, j: usize, s: f64, nb: f64, slack: f64) -> bool {
        if !nb.is_finite() {
            return false;
        }
        // In x space: s = 1 -> x >= nb; s = -1 -> x <= -nb.
        let (mut lo, mut up) = (self.l[j], self.u[j]);
        if s > 0.0 {
            let v = if self.is_int[j] {
                (nb - slack * (1.0 + nb.abs())).ceil()
            } else {
                nb
            };
            if v <= lo || !(self.is_int[j] || v >= up) {
                return false;
            }
            lo = v.min(up);
        } else {
            let v = if self.is_int[j] {
                (-nb + slack * (1.0 + nb.abs())).floor()
            } else {
                -nb
            };
            if v >= up || !(self.is_int[j] || v <= lo) {
                return false;
            }
            up = v.max(lo);
        }
        self.tighten(j, lo, up);
        true
    }

    /// `s x_k <= nb`: the mirror of `set_side_bound`.
    fn set_side_bound_upper(&mut self, k: usize, s: f64, nb: f64, slack: f64) -> bool {
        self.set_side_bound(k, -s, -nb, slack)
    }

    /// Parallel columns (HiGHS `detectParallelRowsAndCols`, column part): when column
    /// `k` is `s` times column `j` and `c_k = s c_j`, only `x' = x_j + s x_k` matters, so
    /// `k` is removed and `j` carries `x'` with the summed bounds. Postsolve must split
    /// `x'` back inside the bounds, which is always possible when: both are continuous;
    /// `j` is continuous and `u_j - l_j >= |s|` (an interval of `x_k` of length `>= 1`);
    /// both are integer, `s` integral and `u_j - l_j >= |s| - 1`. The roles are swapped
    /// when that admits the pair. Unequal costs are a domination, left to
    /// `dominated_pairs`.
    #[allow(clippy::needless_range_loop)]
    fn parallel_cols(&mut self) -> bool {
        let mut groups: HashMap<Vec<(usize, i64)>, Vec<usize>> = HashMap::new();
        let mut entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.n];
        for (r, (ri, rv)) in self.rows.iter().enumerate() {
            if !self.row_alive[r] {
                continue;
            }
            for (&j, &a) in ri.iter().zip(rv) {
                entries[j].push((r, a));
            }
        }
        for j in 0..self.n {
            if !self.col_alive[j] || entries[j].is_empty() {
                continue;
            }
            entries[j].sort_by_key(|e| e.0);
            let s0 = entries[j][0].1;
            let key: Vec<(usize, i64)> = entries[j]
                .iter()
                .map(|&(r, a)| (r, ((a / s0) * 1e9).round() as i64))
                .collect();
            groups.entry(key).or_default().push(j);
        }
        let mut changed = false;
        let mut keys: Vec<_> = groups.into_iter().filter(|g| g.1.len() > 1).collect();
        keys.sort_by_key(|g| g.1[0]);
        for (_, cols) in keys {
            let j0 = cols[0];
            for &k0 in &cols[1..] {
                if !self.col_alive[j0] || !self.col_alive[k0] {
                    continue;
                }
                let lam = entries[k0][0].1 / entries[j0][0].1;
                let exact = entries[j0].len() == entries[k0].len()
                    && entries[j0]
                        .iter()
                        .zip(&entries[k0])
                        .all(|(&(r0, a0), &(r1, a1))| {
                            r0 == r1 && (a1 - lam * a0).abs() <= 1e-12 * a1.abs().max(1.0)
                        });
                if !exact
                    || (self.c[k0] - lam * self.c[j0]).abs() > 1e-12 * self.c[k0].abs().max(1.0)
                {
                    continue;
                }
                // Choose which column carries x' (j) and which is split off (k).
                let admits = |w: &Work, j: usize, k: usize, s: f64| -> bool {
                    let width = w.u[j] - w.l[j];
                    match (w.is_int[j], w.is_int[k]) {
                        (false, false) => true,
                        (false, true) => width >= s.abs(),
                        (true, true) => (s - s.round()).abs() <= 1e-12 && width >= s.abs() - 1.0,
                        (true, false) => false,
                    }
                };
                let (j, k, s) = if admits(self, j0, k0, lam) {
                    (j0, k0, lam)
                } else if admits(self, k0, j0, 1.0 / lam) {
                    (k0, j0, 1.0 / lam)
                } else {
                    continue;
                };
                // The columns are only CLOSE to parallel (the test above has an absolute
                // floor, `max(|a1|, 1)`): merging drops `d_r = a_kr - s a_jr` from each row
                // and `d_c = c_k - s c_j` from the objective. The objective term would move
                // the reduced problem's bound (its objective is reported as is), so `d_c`
                // must be exactly zero. A row term is absorbed by widening an inequality by
                // its reach `|d_r| max(|l_k|, |u_k|)`; on an equality, or with no finite
                // reach, the columns are not merged. (Same class as `cancel_entry`.)
                if self.c[k] - s * self.c[j] != 0.0 {
                    continue;
                }
                let reach_k = self.l[k].abs().max(self.u[k].abs());
                let mut widen: Vec<(usize, f64)> = Vec::new();
                let mut refuse = false;
                for (&(r, aj), &(_, ak)) in entries[j].iter().zip(&entries[k]) {
                    let d = (ak - s * aj).abs();
                    if d == 0.0 {
                        continue;
                    }
                    if self.lo[r] == self.up[r] || !reach_k.is_finite() || reach_k >= INF {
                        refuse = true;
                        break;
                    }
                    widen.push((r, d * reach_k * (1.0 + 1e-12) + f64::MIN_POSITIVE));
                }
                if refuse {
                    continue;
                }
                for (r, w) in widen {
                    if self.lo[r].is_finite() {
                        self.lo[r] -= w;
                    }
                    if self.up[r].is_finite() {
                        self.up[r] += w;
                    }
                }
                self.ops.push(PostOp::Split {
                    j,
                    k,
                    s,
                    lj: self.l[j],
                    uj: self.u[j],
                    lk: self.l[k],
                    uk: self.u[k],
                    j_int: self.is_int[j],
                    k_int: self.is_int[k],
                });
                let (lo_k, up_k) = if s > 0.0 {
                    (s * self.l[k], s * self.u[k])
                } else {
                    (s * self.u[k], s * self.l[k])
                };
                let nl = self.l[j] + lo_k;
                let nu = self.u[j] + up_k;
                self.l[j] = if nl.is_nan() { -INF } else { nl };
                self.u[j] = if nu.is_nan() { INF } else { nu };
                // x' is integral only when both are.
                self.is_int[j] = self.is_int[j] && self.is_int[k];
                self.col_alive[k] = false;
                for &r in &self.col_rows[k] {
                    let (ri, rv) = &mut self.rows[r];
                    if let Some(p) = ri.iter().position(|&q| q == k) {
                        ri.swap_remove(p);
                        rv.swap_remove(p);
                    }
                }
                self.stats.parallel_cols += 1;
                changed = true;
            }
        }
        changed
    }

    /// Whether row `r` is redundant over the columns' bounds tightened by the implied
    /// bounds of continuous columns from *other* rows (HiGHS's row activity bounds over
    /// `implColLower/Upper` with source exclusion). A wrong drop could only relax the
    /// problem, which the postsolve check guards; `drop_sourced_by` keeps a dropped row
    /// from supporting any later decision.
    fn redundant_by_implied(&self, r: usize) -> bool {
        let (lo, up) = (self.lo[r], self.up[r]);
        let (mut mn, mut mx) = (0.0f64, 0.0f64);
        let mut used = false;
        for (&k, &a) in self.rows[r].0.iter().zip(&self.rows[r].1) {
            let lk = if self.impl_l[k].1 != r && self.impl_l[k].0 > self.l[k] {
                used = true;
                self.impl_l[k].0
            } else {
                self.l[k]
            };
            let uk = if self.impl_u[k].1 != r && self.impl_u[k].0 < self.u[k] {
                used = true;
                self.impl_u[k].0
            } else {
                self.u[k]
            };
            let (cmin, cmax) = if a > 0.0 {
                (a * lk, a * uk)
            } else {
                (a * uk, a * lk)
            };
            mn += cmin;
            mx += cmax;
        }
        if !used || mn.is_nan() || mx.is_nan() {
            return false;
        }
        let lo_ok = !lo.is_finite() || mn >= lo - self.tol(lo) * 1e-3;
        let up_ok = !up.is_finite() || mx <= up + self.tol(up) * 1e-3;
        lo_ok && up_ok
    }

    /// Forget the implied bounds row `r` was the source of (it is gone).
    fn drop_sourced_by(&mut self, r: usize) {
        for k in 0..self.n {
            if self.impl_l[k].1 == r {
                self.impl_l[k] = (-INF, usize::MAX);
            }
            if self.impl_u[k].1 == r {
                self.impl_u[k] = (INF, usize::MAX);
            }
        }
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

    /// Alive nonzeros per column (HiGHS `colsize`).
    fn col_sizes(&self) -> Vec<usize> {
        let mut cs = vec![0usize; self.n];
        for (r, (ri, _)) in self.rows.iter().enumerate() {
            if self.row_alive[r] {
                for &j in ri {
                    cs[j] += 1;
                }
            }
        }
        cs
    }

    /// Alive rows holding column `j`, each once (`col_rows` may list dead rows, rows
    /// the column has left, and repeats).
    fn rows_of(&self, j: usize) -> Vec<usize> {
        let mut out: Vec<usize> = self.col_rows[j]
            .iter()
            .copied()
            .filter(|&q| self.row_alive[q] && self.rows[q].0.contains(&j))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Sparsify: a port of HiGHS's `HPresolve::sparsify`
    /// (`highs/presolve/HPresolve.cpp`, HiGHS 1.15.1, MIT License, Copyright (c)
    /// HiGHS; license text in `THIRD_PARTY_NOTICES.md` at the repository root).
    ///
    /// For each equality row `eq` (`a_eq x = r`, in HiGHS's order: by row length, then
    /// index, as of the start of the rule), every other row `q` holding `eq`'s sparsest
    /// column is a candidate for `q += lambda * eq` (sides `+= lambda r`). The
    /// multipliers on offer are `lambda = -a_qj / a_eqj` over `eq`'s columns `j` (those
    /// with `|lambda| > 1e3` skipped); a multiplier's count is the entries it cancels,
    /// a near-identical one (within `sqrt(feastol) / |a_eqj|`) adding to it only when
    /// its residual is at most `small_matrix_value` and otherwise marking it unusable.
    /// The row is changed when the best count (ties: smaller `|lambda|`) exceeds the
    /// fill-in `misses` (columns of `eq` absent from `q`), at most one fill-in, none
    /// when `eq` has no integer column and `q` has one, and never a fill-in of a general
    /// integer (an integer column whose box is wider than 1.5). After each equality's
    /// additions the touched rows that became singletons or doubleton equations are
    /// removed, as HiGHS does (`removeRowSingletons`, `removeDoubletonEquations`).
    ///
    /// Why it is exact for a MILP: on every point with `a_eq x = r`, `q + lambda eq`
    /// takes the value `a_q x + lambda r`, so the feasible set (integrality included)
    /// is unchanged and the primal postsolve is the identity; nothing in HiGHS's rule
    /// restricts `lambda` by integrality, and none is needed. The row's constant moves
    /// with it (`shift -= lambda r`), so the driver's rebuild `b - row_shift` with the
    /// row's own slack keeps the slack's value `(b - a_q x) / sigma`.
    ///
    /// Departures from HiGHS, each only ever less aggressive:
    /// - HiGHS drops a combined entry with `|v| <= small_matrix_value` outright. Here
    ///   every touched entry goes through `cancel_entry` (an exact zero is removed, a
    ///   leftover `|v| <= 1e-12` only from an inequality that is relaxed by its reach),
    ///   the soundness rule of this module, and an addition that would leave a nonzero
    ///   `|v| <= small_matrix_value` that `cancel_entry` cannot remove is not made
    ///   (`stats.sparsify_refused_roundoff`). With exact data and multipliers (integer
    ///   rows, `lambda = +-1, +-2`) every cancellation is an exact zero and nothing
    ///   differs from HiGHS.
    /// - HiGHS's second scan (rows holding the second-sparsest column but not the
    ///   sparsest) is not ported: in 1.15.1 it sets `skip` for every stored-row column
    ///   missing from the candidate except the second-sparsest one, and the sparsest
    ///   column is missing from every row it scans by construction, so it never adds a
    ///   row. Porting it would be dead code.
    /// - The singleton / doubleton clean-up looks at the touched rows only; the next
    ///   pass handles anything else.
    ///
    /// Returns whether any row was changed.
    fn sparsify(&mut self) -> bool {
        // HiGHS defaults: `small_matrix_value` and `primal_feasibility_tolerance` (whose
        // square root is the scale-clustering width `minNonzeroVal`).
        const SMALL_MATRIX_VALUE: f64 = 1e-9;
        const PRIMAL_FEASTOL: f64 = 1e-7;
        const MAX_SCALE: f64 = 1e3;
        let min_nonzero_val = PRIMAL_FEASTOL.sqrt();
        // Entries the additions removed and created (net = the nonzeros saved).
        let (mut nz_removed, mut nz_filled) = (0usize, 0usize);
        let mut eqs: Vec<usize> = (0..self.rows.len())
            .filter(|&r| self.row_alive[r] && self.lo[r] == self.up[r] && self.lo[r].is_finite())
            .collect();
        eqs.sort_by_key(|&r| (self.rows[r].0.len(), r));
        let mut colsize = self.col_sizes();
        // Position of each column in the stored equality row (usize::MAX: absent).
        let mut eqpos = vec![usize::MAX; self.n];
        let mut changed = false;
        let n_int = |w: &Work, q: usize| w.rows[q].0.iter().filter(|&&k| w.is_int[k]).count();
        for eq in eqs {
            if self.infeasible {
                return changed;
            }
            if !self.row_alive[eq] || self.lo[eq] != self.up[eq] || self.rows[eq].0.len() < 2 {
                continue;
            }
            // HiGHS `storeRow`: the row in column order.
            let mut stored: Vec<(usize, f64)> = self.rows[eq]
                .0
                .iter()
                .copied()
                .zip(self.rows[eq].1.iter().copied())
                .collect();
            stored.sort_by_key(|e| e.0);
            let mut sparsest = usize::MAX;
            let mut sparsest_len = usize::MAX;
            for &(j, _) in &stored {
                if colsize[j] < sparsest_len {
                    sparsest = j;
                    sparsest_len = colsize[j];
                }
            }
            for (p, &(j, _)) in stored.iter().enumerate() {
                eqpos[j] = p;
            }
            let eq_int = n_int(self, eq);
            let mut adds: Vec<(usize, f64)> = Vec::new();
            // Candidate's value per stored-row position.
            let mut cand_val: Vec<Option<f64>> = vec![None; stored.len()];
            // (scale, count); count -1 marks a numerically bad scale.
            let mut scales: Vec<(f64, i64)> = Vec::new();
            for q in self.rows_of(sparsest) {
                if q == eq {
                    continue;
                }
                cand_val.iter_mut().for_each(|v| *v = None);
                for (&k, &v) in self.rows[q].0.iter().zip(&self.rows[q].1) {
                    if eqpos[k] != usize::MAX {
                        cand_val[eqpos[k]] = Some(v);
                    }
                }
                scales.clear();
                let mut misses: i64 = 0;
                // No fill-in when a continuous equality cancels in a row with integers
                // (HiGHS: "a huge deterioration of cut performance").
                let mut max_misses: i64 = 1;
                if eq_int == 0 && n_int(self, q) != 0 {
                    max_misses -= 1;
                }
                for (p, &(j, a_eq)) in stored.iter().enumerate() {
                    let Some(cv) = cand_val[p] else {
                        if self.is_int[j] && self.u[j] - self.l[j] > 1.5 {
                            // No fill-in of general integers.
                            misses = 2;
                            break;
                        }
                        misses += 1;
                        if misses > max_misses {
                            break;
                        }
                        continue;
                    };
                    let scale = -cv / a_eq;
                    if scale.abs() > MAX_SCALE {
                        continue;
                    }
                    let tol = min_nonzero_val / a_eq.abs();
                    // `std::map::lower_bound(scale - tol)`.
                    let it = scales.partition_point(|e| e.0 < scale - tol);
                    if it < scales.len() && (scales[it].0 - scale).abs() <= tol {
                        if scales[it].1 == -1 {
                            continue;
                        }
                        if (scales[it].0 * a_eq + cv).abs() <= SMALL_MATRIX_VALUE {
                            scales[it].1 += 1;
                        } else {
                            scales[it].1 = -1;
                        }
                    } else {
                        let at = scales.partition_point(|e| e.0 < scale);
                        scales.insert(at, (scale, 1));
                    }
                }
                if misses > max_misses || scales.is_empty() {
                    continue;
                }
                let (mut num_cancel, mut scale) = (0i64, 0.0f64);
                for &(s, cnt) in &scales {
                    if cnt <= misses {
                        continue;
                    }
                    if cnt > num_cancel || (cnt == num_cancel && s.abs() < scale.abs()) {
                        scale = s;
                        num_cancel = cnt;
                    }
                }
                if num_cancel > misses {
                    adds.push((q, scale));
                }
            }
            for &(j, _) in &stored {
                eqpos[j] = usize::MAX;
            }
            if adds.is_empty() {
                continue;
            }
            let rhs = self.lo[eq];
            // HiGHS drops a combined entry with `|v| <= small_matrix_value` without
            // compensating the row. Here such a round-off leftover must be removable by
            // `cancel_entry` (an inequality, `|v| <= 1e-12`, a finite reach); otherwise the
            // row is not changed at all. Keeping it is not an option either: measured on a
            // random integer program, multipliers -1/6 and -4/3 left `1.1e-16 x2` in an
            // equality, which the doubleton rule then divided by (coefficients of 1.5e16)
            // and turned a feasible problem infeasible.
            let leftover_ok = |w: &Work, q: usize, scale: f64| -> bool {
                stored.iter().all(|&(j, a_eq)| {
                    let Some(p) = w.rows[q].0.iter().position(|&k| k == j) else {
                        return true;
                    };
                    let v = w.rows[q].1[p] + scale * a_eq;
                    if v == 0.0 || v.abs() > SMALL_MATRIX_VALUE {
                        return true;
                    }
                    let reach = w.l[j].abs().max(w.u[j].abs());
                    w.lo[q] != w.up[q] && v.abs() <= 1e-12 && reach.is_finite()
                })
            };
            let n_adds = adds.len();
            adds.retain(|&(q, scale)| leftover_ok(self, q, scale));
            self.stats.sparsify_refused_roundoff += n_adds - adds.len();
            for &(q, scale) in &adds {
                if self.lo[q].is_finite() {
                    self.lo[q] += scale * rhs;
                }
                if self.up[q].is_finite() {
                    self.up[q] += scale * rhs;
                }
                // The constant moved out of the row's activity changes by the opposite
                // amount (see `Presolved::row_shift`).
                self.shift[q] -= scale * rhs;
                for &(j, a_eq) in &stored {
                    let add = scale * a_eq;
                    if let Some(p) = self.rows[q].0.iter().position(|&k| k == j) {
                        self.rows[q].1[p] += add;
                        let before = self.rows[q].0.len();
                        self.cancel_entry(q, p);
                        if self.rows[q].0.len() < before {
                            colsize[j] -= 1;
                            nz_removed += 1;
                        }
                    } else if add != 0.0 {
                        self.rows[q].0.push(j);
                        self.rows[q].1.push(add);
                        if !self.col_rows[j].contains(&q) {
                            self.col_rows[j].push(q);
                        }
                        colsize[j] += 1;
                        nz_filled += 1;
                    }
                }
                self.stats.sparsified_rows += 1;
                changed = true;
            }
            // HiGHS `removeRowSingletons` / `removeDoubletonEquations` on what changed.
            let mut cleaned = false;
            for &(q, _) in &adds {
                if !self.row_alive[q] || self.infeasible {
                    continue;
                }
                match self.rows[q].0.len() {
                    1 => {
                        let (j, a) = (self.rows[q].0[0], self.rows[q].1[0]);
                        let (lo, up) = (self.lo[q] / a, self.up[q] / a);
                        let (blo, bup) = if a > 0.0 { (lo, up) } else { (up, lo) };
                        self.tighten(j, blo, bup);
                        self.row_alive[q] = false;
                        self.stats.singleton_rows += 1;
                        cleaned = true;
                    }
                    2 if self.lo[q] == self.up[q] && self.doubleton_eq(q) => cleaned = true,
                    _ => {}
                }
            }
            if cleaned {
                colsize = self.col_sizes();
            }
        }
        self.stats.sparsify_nz_removed += nz_removed.saturating_sub(nz_filled);
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
                if a.abs() <= NOISE_REL {
                    // A lone noise coefficient bounds nothing it can be trusted for
                    // (`is_noise`): the row stays as it is.
                    continue;
                }
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
            if self.redundant_by_implied(r) {
                self.row_alive[r] = false;
                self.drop_sourced_by(r);
                self.stats.redundant_rows_implied += 1;
                changed = true;
                continue;
            }
            // Forcing: the row can only be met at one end of its activity range.
            let force_max = nmx == 0 && lo.is_finite() && mx <= lo + self.tol(lo) * 1e-3;
            let force_min = nmn == 0 && up.is_finite() && mn >= up - self.tol(up) * 1e-3;
            if force_max || force_min {
                let (ri, rv) = self.rows[r].clone();
                for (&j, &a) in ri.iter().zip(&rv) {
                    if self.is_noise(r, a) {
                        // A noise column is not forced (`is_noise`): it stays free, so
                        // removing the row relaxes rather than cuts.
                        continue;
                    }
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
        if self.dominated_pairs() {
            changed = true;
        }
        if self.parallel_cols() {
            changed = true;
        }
        if self.infeasible {
            return false;
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
                let close = e0.len() == ent.len()
                    && e0.iter().zip(&ent).all(|(&(j0, a0), &(j1, a1))| {
                        j0 == j1 && (a1 - lam * a0).abs() <= 1e-12 * a1.abs().max(1.0)
                    });
                if !close {
                    continue;
                }
                // `r` is `lam r0 + d` with `d = a1 - lam a0` only CLOSE to zero, so its
                // range moves to `r0` widened by `|d x|`'s reach over the box: `delta =
                // sum |d_j| max(|l_j|, |u_j|)`. Taking the stricter side as if `d` were 0
                // is not a relaxation: the closeness test above has an absolute floor
                // (`max(|a1|, 1)`), so on small coefficients over large columns (OA cuts
                // after aggregation: ~1e-6 on integers up to 1e5) a difference it accepts
                // is worth ~1e-7 of activity. A nonzero `d_j` on an unbounded column has
                // no finite reach: the rows are not merged. (The same class as
                // `cancel_entry`; on the captured portfol_roundlot master the measured
                // false bound came from there, not from this rule.)
                let mut delta = 0.0f64;
                let mut unbounded = false;
                for (&(_, a0), &(j, a1)) in e0.iter().zip(&ent) {
                    let d = (a1 - lam * a0).abs();
                    if d == 0.0 {
                        continue;
                    }
                    let reach = self.l[j].abs().max(self.u[j].abs());
                    if !reach.is_finite() || reach >= INF {
                        unbounded = true;
                        break;
                    }
                    delta += d * reach;
                }
                if unbounded {
                    continue;
                }
                // Outward for the rounding of the sum; exactly parallel rows (every
                // `d_j` zero) merge as before, so an equality stays an equality.
                let delta = if delta > 0.0 {
                    delta * (1.0 + 1e-12) + f64::MIN_POSITIVE
                } else {
                    0.0
                };
                // lo_r - delta <= lam a0 x <= up_r + delta  =>  a0 x in that / lam.
                let (a, b) = ((self.lo[r] - delta) / lam, (self.up[r] + delta) / lam);
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
    presolve_opts(f, feastol, max_passes, false, false)
}

/// `presolve` with the opt-in rules: `implied_int` runs HiGHS's implied-integer
/// detection after the last pass ([`Work::detect_implied_integers`]); `sparsify` runs
/// HiGHS's sparsify (`Work::sparsify`) once, at the first pass that removes at most
/// 0.05 % of the rows and of the columns (HiGHS runs it when its reduction since the last
/// size checkpoint is at most 0.05 %), and the passes continue after it when it changed a
/// row.
pub fn presolve_opts(
    f: &RowForm,
    feastol: f64,
    max_passes: usize,
    implied_int: bool,
    sparsify: bool,
) -> Presolved {
    let mut w = Work::new(f, feastol);
    // Integer bounds rounded inward first.
    for j in 0..f.n {
        if w.is_int[j] {
            w.tighten(j, w.l[j], w.u[j]);
        }
    }
    let alive = |w: &Work| {
        (
            w.row_alive.iter().filter(|&&a| a).count(),
            w.col_alive.iter().filter(|&&a| a).count(),
        )
    };
    let mut try_sparsify = sparsify;
    while !w.infeasible && w.stats.passes < max_passes {
        w.stats.passes += 1;
        let (r0, c0) = alive(&w);
        let mut changed = w.pass();
        if try_sparsify && !w.infeasible {
            // HiGHS `problemSizeReduction`: the larger of the row and column reductions,
            // in percent.
            let (r1, c1) = alive(&w);
            let pct = |a: usize, b: usize| {
                if a == 0 {
                    0.0
                } else {
                    100.0 * (a - b) as f64 / a as f64
                }
            };
            if pct(r0, r1).max(pct(c0, c1)) <= 0.05 {
                try_sparsify = false;
                if w.sparsify() {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if implied_int && !w.infeasible {
        w.detect_implied_integers();
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

/// `f` as free-format MPS (minimization), for diagnostics.
#[allow(clippy::needless_range_loop)]
pub fn to_free_mps(f: &RowForm) -> String {
    use std::fmt::Write;
    let mut o = String::from("NAME reduced\nROWS\n N obj\n");
    for r in 0..f.rows.len() {
        let t = match (f.lo[r].is_finite(), f.up[r].is_finite()) {
            (true, true) if f.lo[r] == f.up[r] => "E",
            (true, _) => "G",
            (false, true) => "L",
            (false, false) => "N",
        };
        let _ = writeln!(o, " {t} r{r}");
    }
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); f.n];
    for (r, (ri, rv)) in f.rows.iter().enumerate() {
        for (&j, &a) in ri.iter().zip(rv) {
            cols[j].push((r, a));
        }
    }
    o.push_str("COLUMNS\n");
    let mut in_int = false;
    for j in 0..f.n {
        if f.is_int[j] != in_int {
            let m = if f.is_int[j] { "INTORG" } else { "INTEND" };
            let _ = writeln!(o, " M{j} 'MARKER' '{m}'");
            in_int = f.is_int[j];
        }
        let _ = writeln!(o, " x{j} obj {:e}", f.c[j]);
        for &(r, a) in &cols[j] {
            let _ = writeln!(o, " x{j} r{r} {a:e}");
        }
    }
    if in_int {
        o.push_str(" Mend 'MARKER' 'INTEND'\n");
    }
    o.push_str("RHS\n");
    for r in 0..f.rows.len() {
        let v = if f.lo[r].is_finite() {
            f.lo[r]
        } else {
            f.up[r]
        };
        if v.is_finite() && v != 0.0 {
            let _ = writeln!(o, " rhs r{r} {v:e}");
        }
    }
    o.push_str("RANGES\n");
    for r in 0..f.rows.len() {
        if f.lo[r].is_finite() && f.up[r].is_finite() && f.lo[r] != f.up[r] {
            let _ = writeln!(o, " rng r{r} {:e}", f.up[r] - f.lo[r]);
        }
    }
    o.push_str("BOUNDS\n");
    for j in 0..f.n {
        let (l, u) = (f.l[j], f.u[j]);
        if l == u {
            let _ = writeln!(o, " FX bnd x{j} {l:e}");
            continue;
        }
        if l.is_finite() {
            let _ = writeln!(o, " LO bnd x{j} {l:e}");
        } else {
            let _ = writeln!(o, " MI bnd x{j}");
        }
        if u.is_finite() {
            let _ = writeln!(o, " UP bnd x{j} {u:e}");
        } else {
            let _ = writeln!(o, " PL bnd x{j}");
        }
    }
    o.push_str("ENDATA\n");
    o
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

    /// A substitution whose coefficients cancel to round-off on a large column must not
    /// drop the leftover from an inequality without relaxing it (`cancel_entry`; the
    /// class behind the portfol_roundlot OA master's false bound). Aggregating `x` out of
    /// `x - a n - b m = 0` leaves `5e-13 n` in `t - x + (a + 5e-13) n >= 0`, worth 5e-8
    /// at `n = 1e5`; the point below is feasible and must stay feasible after presolve.
    #[test]
    fn a_cancelled_coefficient_is_not_dropped_without_relaxing_its_row() {
        let a = 1e-5;
        // Columns: 0 = x (continuous), 1 = n, 2 = m (integers), 3 = t (continuous).
        let f = RowForm {
            n: 4,
            rows: vec![
                (vec![0, 1, 2], vec![1.0, -a, -a]),
                (vec![3, 0, 1], vec![1.0, -1.0, a + 5e-13]),
            ],
            lo: vec![0.0, 0.0],
            up: vec![0.0, INF],
            c: vec![0.0, 0.0, 0.0, 1.0],
            l: vec![-100.0, 0.0, 0.0, -10.0],
            u: vec![100.0, 1e5, 1e5, 10.0],
            is_int: vec![false, true, true, false],
        };
        let point = [a * 1e5, 1e5, 0.0, 1.0 - (a + 5e-13) * 1e5];
        assert!(
            row_form_feasible(&f, &point, 1e-12),
            "the point must be feasible"
        );
        let p = presolve(&f, 1e-9, 20);
        // Probe (CLAUDE.md section 6): the rule under test must have fired.
        assert!(
            p.stats.aggregated + p.stats.doubleton_eqs > 0,
            "{:?}",
            p.stats
        );
        assert!(!p.infeasible);
        let point_obj: f64 = f.c.iter().zip(&point).map(|(c, x)| c * x).sum();
        if p.keep_cols.is_empty() {
            // Presolve solved it outright (measured: `t` fixed by dual fixing): the
            // reduced optimum is the offset, and it must not be above the point's
            // objective. The pre-fix rule dropped the 5e-13 term and fixed t at 0.
            assert!(
                p.offset <= point_obj + 1e-15,
                "presolve optimum {:e} above a feasible point's {point_obj:e}",
                p.offset
            );
            return;
        }
        let red: Vec<f64> = p.keep_cols.iter().map(|&j| point[j]).collect();
        let mut checked = 0;
        for (r, (ri, rv)) in p.form.rows.iter().enumerate() {
            let act: f64 = ri.iter().zip(rv).map(|(&k, &v)| v * red[k]).sum();
            assert!(
                act <= p.form.up[r] + 1e-12 && act >= p.form.lo[r] - 1e-12,
                "reduced row {r}: activity {act:e} outside [{:e}, {:e}] at a feasible point",
                p.form.lo[r],
                p.form.up[r]
            );
            checked += 1;
        }
        // Rows that collapsed to a singleton became column bounds: judge those too.
        for (k, &x) in red.iter().enumerate() {
            assert!(
                x >= p.form.l[k] - 1e-12 && x <= p.form.u[k] + 1e-12,
                "reduced column {k} (original {}): {x:e} outside [{:e}, {:e}]",
                p.keep_cols[k],
                p.form.l[k],
                p.form.u[k]
            );
            checked += 1;
        }
        assert!(checked > 0);
    }

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
        if r.int(0, 2) == 0 {
            // A parallel column: s times an existing column, with s times its cost (and
            // sometimes an unrelated cost, a domination instead).
            let j = r.int(0, f.n as i64 - 1) as usize;
            let sc = [1.0, 2.0, -1.0, -2.0][r.int(0, 3) as usize];
            let k = f.n;
            f.n += 1;
            for (ri, rv) in f.rows.iter_mut() {
                if let Some(p) = ri.iter().position(|&q| q == j) {
                    let v = rv[p] * sc;
                    ri.push(k);
                    rv.push(v);
                }
            }
            let ck = if r.int(0, 3) == 0 {
                r.int(-4, 4) as f64
            } else {
                sc * f.c[j]
            };
            f.c.push(ck);
            let lk = r.int(-2, 1) as f64;
            f.l.push(lk);
            f.u.push(lk + r.int(0, 3) as f64);
            f.is_int.push(true);
        }
        f
    }

    #[test]
    fn presolve_preserves_the_optimum_and_postsolve_is_feasible() {
        // `XSEED=<u64>` reruns the 3,000 cases from another seed (test-only).
        let mut r = Lcg(std::env::var("XSEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2024));
        let (mut cases, mut reduced, mut infeasible, mut feasible) =
            (0usize, 0usize, 0usize, 0usize);
        let mut doubletons = 0usize;
        let mut aggregated = 0usize;
        let mut propagated = 0usize;
        let mut pairs = 0usize;
        let mut par_cols = 0usize;
        for _ in 0..3000 {
            let f = random_ip(&mut r);
            let want = brute_min(&f);
            let p = presolve(&f, 1e-9, 20);
            cases += 1;
            doubletons += p.stats.doubleton_eqs;
            aggregated += p.stats.aggregated;
            propagated += p.stats.propagated;
            pairs += p.stats.dominated_pairs;
            par_cols += p.stats.parallel_cols;
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
                && propagated > 100
                && pairs > 100
                && par_cols > 20,
            "probe fired too little: reduced {reduced} infeasible {infeasible} feasible {feasible} doubletons {doubletons} aggregated {aggregated} propagated {propagated} dominated pairs {pairs} parallel cols {par_cols}"
        );
    }

    /// Exact optimum of a problem whose last column (`h`) is continuous and all others
    /// integer with finite bounds: per integer point, `h` ranges over an interval.
    fn exact_min_one_continuous(f: &RowForm) -> Option<f64> {
        let h = f.n - 1;
        let mut x: Vec<f64> = f.l[..h].to_vec();
        let mut best: Option<f64> = None;
        loop {
            let (mut lo_h, mut up_h) = (f.l[h], f.u[h]);
            let mut ok = true;
            for (r, (ri, rv)) in f.rows.iter().enumerate() {
                let (mut s, mut a_h) = (0.0, 0.0);
                for (&j, &v) in ri.iter().zip(rv) {
                    if j == h {
                        a_h += v;
                    } else {
                        s += v * x[j];
                    }
                }
                if a_h == 0.0 {
                    ok &= s >= f.lo[r] - 1e-9 && s <= f.up[r] + 1e-9;
                } else {
                    let (p, q) = ((f.lo[r] - s) / a_h, (f.up[r] - s) / a_h);
                    lo_h = lo_h.max(p.min(q));
                    up_h = up_h.min(p.max(q));
                }
            }
            if ok && lo_h <= up_h + 1e-9 {
                let ch = f.c[h];
                let vh = if ch > 0.0 {
                    ch * lo_h
                } else if ch < 0.0 {
                    ch * up_h
                } else {
                    0.0
                };
                let v: f64 = f.c[..h].iter().zip(&x).map(|(c, x)| c * x).sum::<f64>() + vh;
                if best.is_none_or(|b| v < b - 1e-9) {
                    best = Some(v);
                }
            }
            let mut j = 0;
            while j < h {
                x[j] += 1.0;
                if x[j] <= f.u[j] {
                    break;
                }
                x[j] = f.l[j];
                j += 1;
            }
            if j == h {
                break;
            }
        }
        best
    }

    /// HiGHS's implied-integer rule (`presolve_opts(.., true, false)`) never changes the
    /// optimum: on random problems with one continuous column `h` (integer, half-
    /// integral and fractional data mixed so the rule both fires and refuses), whenever
    /// presolve marks `h` integral the all-integer reduced problem has the exact
    /// optimum of the original (computed with `h` continuous), and its postsolved
    /// optimum is feasible for the original.
    #[test]
    fn implied_integer_marking_preserves_the_optimum() {
        let mut r = Lcg(77);
        let pick = |r: &mut Lcg| -> f64 {
            match r.int(0, 9) {
                0 => 0.5,
                1 => -1.5,
                _ => r.int(-3, 3) as f64,
            }
        };
        let (mut cases, mut primal, mut dual, mut refused, mut checked) = (0, 0, 0, 0, 0);
        for _ in 0..20000 {
            let ni = r.int(1, 3) as usize;
            let n = ni + 1;
            let mut l: Vec<f64> = (0..ni).map(|_| r.int(-2, 1) as f64).collect();
            let mut u: Vec<f64> = l.iter().map(|&v| v + r.int(0, 3) as f64).collect();
            let lh = r.int(-4, 0) as f64;
            let uh = lh + r.int(0, 6) as f64 + if r.int(0, 4) == 0 { 0.5 } else { 0.0 };
            l.push(lh);
            u.push(uh);
            let c: Vec<f64> = (0..n).map(|_| r.int(-4, 4) as f64).collect();
            let m = r.int(1, 3) as usize;
            let (mut rows, mut lo, mut up) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..m {
                let mut ri = Vec::new();
                let mut rv = Vec::new();
                for j in 0..n {
                    if j == ni || r.int(0, 2) > 0 {
                        let v = if j == ni {
                            [1.0, -1.0, 2.0, 0.5][r.int(0, 3) as usize]
                        } else {
                            pick(&mut r)
                        };
                        if v != 0.0 {
                            ri.push(j);
                            rv.push(v);
                        }
                    }
                }
                if ri.is_empty() {
                    continue;
                }
                let base = r.int(-4, 4) as f64 + if r.int(0, 5) == 0 { 0.5 } else { 0.0 };
                match r.int(0, 2) {
                    0 => {
                        lo.push(base);
                        up.push(base);
                    }
                    1 => {
                        lo.push(-INF);
                        up.push(base);
                    }
                    _ => {
                        lo.push(base);
                        up.push(INF);
                    }
                }
                rows.push((ri, rv));
            }
            let mut is_int = vec![true; n];
            is_int[ni] = false;
            let f = RowForm {
                n,
                rows,
                lo,
                up,
                c,
                l,
                u,
                is_int,
            };
            let want = exact_min_one_continuous(&f);
            let p = presolve_opts(&f, 1e-9, 20, true, false);
            cases += 1;
            primal += p.stats.implied_int_primal;
            dual += p.stats.implied_int_dual;
            if p.infeasible {
                assert!(
                    want.is_none(),
                    "presolve declared a feasible problem infeasible: {f:?}"
                );
                continue;
            }
            if p.form.is_int.iter().any(|&b| !b) {
                refused += 1;
                continue;
            }
            if p.stats.implied_int_primal + p.stats.implied_int_dual == 0 {
                continue; // h was removed by another rule, not marked
            }
            checked += 1;
            let got = brute_min(&p.form);
            match (want, &got) {
                (None, None) => {}
                (Some(wv), Some((gv, gx))) => {
                    assert!(
                        (wv - (gv + p.offset)).abs() <= 1e-9,
                        "optimum changed: {wv} vs {} for {f:?} (stats {:?})",
                        gv + p.offset,
                        p.stats
                    );
                    let x = p.postsolve(gx, f.n);
                    assert!(
                        row_form_feasible(&f, &x, 1e-9),
                        "postsolved point infeasible: {x:?} for {f:?}"
                    );
                }
                _ => panic!("feasibility changed: original {want:?} reduced {got:?} for {f:?}"),
            }
        }
        eprintln!(
            "implied-int: cases {cases} primal {primal} dual {dual} refused {refused} checked {checked}"
        );
        assert!(
            cases == 20000 && primal > 100 && dual > 100 && refused > 300 && checked > 250,
            "probe fired too little: primal {primal} dual {dual} refused {refused} checked {checked}"
        );
    }

    /// Presolve on integer programs with FRACTIONAL row coefficients (thirds): the
    /// optimum must be preserved and the postsolved optimum feasible, as for integer
    /// data. Cancellation in such rows leaves round-off entries (~4e-16) that the
    /// doubleton/singleton rules must never divide by (found by the sparsify port's
    /// census; reproduced here with sparsify absent). `XSEED` reruns other seeds.
    #[test]
    fn presolve_with_fractional_coefficients_preserves_the_optimum() {
        let base: u64 = std::env::var("XSEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4242);
        let mut r = Lcg(base);
        let fr = [
            -5.0 / 3.0,
            -4.0 / 3.0,
            -2.0 / 3.0,
            -1.0 / 3.0,
            1.0 / 3.0,
            2.0 / 3.0,
            4.0 / 3.0,
            5.0 / 3.0,
            -3.0,
            -2.0,
            -1.0,
            1.0,
            2.0,
            3.0,
        ];
        let (mut cases, mut feasible, mut reduced) = (0usize, 0usize, 0usize);
        for _ in 0..3000 {
            let mut f = random_ip(&mut r);
            for (_, rv) in f.rows.iter_mut() {
                for v in rv.iter_mut() {
                    *v = fr[r.int(0, fr.len() as i64 - 1) as usize];
                }
            }
            let want = brute_min(&f);
            let p = presolve(&f, 1e-9, 20);
            cases += 1;
            if p.keep_cols.len() < f.n || p.form.rows.len() < f.rows.len() {
                reduced += 1;
            }
            if p.infeasible {
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
                        (wv - (gv + p.offset)).abs() <= 1e-6,
                        "optimum changed: {wv} vs {} for {f:?}",
                        gv + p.offset
                    );
                    let x = p.postsolve(gx, f.n);
                    assert!(
                        row_form_feasible(&f, &x, 1e-6),
                        "postsolved point infeasible: {x:?} for {f:?}"
                    );
                }
                _ => panic!("feasibility changed: original {want:?} reduced {got:?} for {f:?}"),
            }
        }
        assert!(
            cases == 3000 && feasible > 300 && reduced > 1000,
            "probe fired too little: feasible {feasible} reduced {reduced}"
        );
    }

    /// Presolve never divides by a noise coefficient (`Work::is_noise`). Random
    /// integer programs get one 1e-16-scale entry (the round-off cancellation leaves)
    /// on an equality row; the reduced problem must keep the optimum (brute force over
    /// both, at the feasibility tolerance the solver certifies under) and the
    /// postsolved optimum must be feasible, and presolve may not call a feasible
    /// problem infeasible. Before the fix: reduced optima above the true one (0 vs -1),
    /// wrong optima, and false infeasibility, on the default presolve.
    #[test]
    fn presolve_never_divides_by_a_noise_coefficient() {
        let nz = |st: &PresolveStats| {
            format!("{st:?}")
                .split(", ")
                .filter(|t| !t.ends_with(": 0") && !t.contains("passes"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut r = Lcg(5);
        let (mut bad, mut cases, mut exact, mut tolerated) = (0, 0, 0, 0);
        for _ in 0..20000 {
            let mut f = random_ip(&mut r);
            if f.rows.is_empty() {
                continue;
            }
            // Put a round-off coefficient on one entry of an equality row.
            let k = r.int(0, f.rows.len() as i64 - 1) as usize;
            if f.rows[k].0.len() < 2 {
                continue;
            }
            f.lo[k] = f.up[k].min(f.lo[k].max(-4.0));
            f.up[k] = f.lo[k];
            if !f.lo[k].is_finite() {
                continue;
            }
            let e = r.int(0, f.rows[k].1.len() as i64 - 1) as usize;
            f.rows[k].1[e] = [4.4e-16, -4.4e-16, 1e-13, -2.2e-16][r.int(0, 3) as usize];
            cases += 1;
            let want = brute_min(&f);
            let p = presolve(&f, 1e-9, 20);
            // The driver's contract (`solve_presolved`): an infeasibility verdict is never
            // trusted, a reduced problem that is infeasible or whose postsolved point
            // fails the original rows (at 1e-6) sends the solve back to the original.
            // What nothing downstream catches -- and is therefore unsafe -- is a reduced
            // optimum above the true one (a false bound), or a postsolved point that
            // passes the check while its objective is not the reduced one.
            if p.infeasible {
                // Right on an infeasible problem; untrusted (so tolerated) on a feasible one.
                if want.is_some() {
                    tolerated += 1;
                } else {
                    exact += 1;
                }
                continue;
            }
            let got = brute_min(&p.form);
            match (&want, &got) {
                (Some((wv, _)), Some((gv, gx))) => {
                    let rv = gv + p.offset;
                    let x = p.postsolve(gx, f.n);
                    let passes = row_form_feasible(&f, &x, 1e-6);
                    let xv: f64 = f.c.iter().zip(&x).map(|(c, x)| c * x).sum();
                    if rv > wv + 1e-6 {
                        bad += 1;
                        eprintln!("FALSE BOUND want {wv} reduced {rv} {}", nz(&p.stats));
                    } else if passes && (xv - rv).abs() > 1e-6 {
                        bad += 1;
                        eprintln!("MISREPORT x worth {xv} reported {rv} {}", nz(&p.stats));
                    } else if (rv - wv).abs() <= 1e-6 && passes {
                        exact += 1;
                    } else {
                        tolerated += 1;
                    }
                }
                (None, Some((_, gx))) => {
                    if row_form_feasible(&f, &p.postsolve(gx, f.n), 1e-6) {
                        bad += 1;
                        eprintln!("INFEASIBLE MADE FEASIBLE {}", nz(&p.stats));
                    } else {
                        tolerated += 1;
                    }
                }
                (Some(_), None) => tolerated += 1,
                (None, None) => exact += 1,
            }
        }
        eprintln!(
            "noise-coefficient probe: cases {cases} exact {exact} tolerated {tolerated} unsafe {bad}"
        );
        assert!(
            cases > 10_000 && exact > 10_000,
            "probe fired too little: {cases} cases, {exact} exact"
        );
        assert_eq!(
            bad, 0,
            "presolve produced an unsafe reduction on {bad} cases"
        );
    }
    /// Random integer programs with the structure sparsify cancels: one or two equality
    /// rows, rows holding a multiple of one of them (sometimes with one of its columns
    /// missing) plus other terms, and unrelated rows. Sides are drawn around the activity
    /// of a reference point in the box, sometimes past it, so most cases are feasible
    /// and some are not.
    fn random_ip_sparsify(r: &mut Lcg) -> RowForm {
        let n = r.int(4, 6) as usize;
        let l: Vec<f64> = (0..n).map(|_| r.int(-1, 0) as f64).collect();
        let u: Vec<f64> = l.iter().map(|&v| v + r.int(1, 2) as f64).collect();
        let c: Vec<f64> = (0..n).map(|_| r.int(-4, 4) as f64).collect();
        let xr: Vec<f64> = (0..n)
            .map(|j| r.int(l[j] as i64, u[j] as i64) as f64)
            .collect();
        let mut f = RowForm {
            n,
            rows: Vec::new(),
            lo: Vec::new(),
            up: Vec::new(),
            c,
            l,
            u,
            is_int: vec![true; n],
        };
        // Dyadic coefficients and multipliers only (see the test below for why).
        let coef = |r: &mut Lcg| [1.0, -1.0, 2.0, 1.0, -2.0][r.int(0, 4) as usize];
        let push_row = |f: &mut RowForm, r: &mut Lcg, ri: Vec<usize>, rv: Vec<f64>, eq| {
            let act: f64 = ri.iter().zip(&rv).map(|(&j, &a)| a * xr[j]).sum();
            let off = r.int(-1, 2) as f64;
            let (lo, up) = match (eq, r.int(0, 2)) {
                (true, _) | (_, 2) => (act, act),
                (_, 0) => (-INF, act + off),
                _ => (act - off, INF),
            };
            f.rows.push((ri, rv));
            f.lo.push(lo);
            f.up.push(up);
        };
        let shuffled = |r: &mut Lcg| {
            let mut cols: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let s = r.int(i as i64, n as i64 - 1) as usize;
                cols.swap(i, s);
            }
            cols
        };
        for _ in 0..r.int(1, 2) {
            let cols = shuffled(r);
            let k = r.int(3, (n as i64 - 1).min(4)) as usize;
            let eqc: Vec<usize> = cols[..k].to_vec();
            let eqv: Vec<f64> = eqc.iter().map(|_| coef(r)).collect();
            push_row(&mut f, r, eqc.clone(), eqv.clone(), true);
            for _ in 0..r.int(1, 3) {
                let lam = [1.0, -1.0, 2.0, -2.0, 0.5, 4.0][r.int(0, 5) as usize];
                let drop = if r.int(0, 2) == 0 {
                    Some(r.int(0, k as i64 - 1) as usize)
                } else {
                    None
                };
                let (mut ri, mut rv) = (Vec::new(), Vec::new());
                for (p, (&j, &a)) in eqc.iter().zip(&eqv).enumerate() {
                    if Some(p) != drop {
                        ri.push(j);
                        rv.push(lam * a);
                    }
                }
                // At least one other column (else the row is parallel to the equality
                // and merged before sparsify's turn).
                let first = r.int(k as i64, n as i64 - 1) as usize;
                for (p, &j) in cols.iter().enumerate().skip(k) {
                    if p == first || r.int(0, 1) == 0 {
                        ri.push(j);
                        rv.push(coef(r));
                    }
                }
                push_row(&mut f, r, ri, rv, false);
            }
        }
        for _ in 0..r.int(0, 2) {
            let cols = shuffled(r);
            let k = r.int(2, n as i64) as usize;
            let ri: Vec<usize> = cols[..k].to_vec();
            let rv: Vec<f64> = ri.iter().map(|_| coef(r)).collect();
            push_row(&mut f, r, ri, rv, false);
        }
        f
    }

    /// The live problem of `w` (alive rows and columns) and its objective constant.
    fn snapshot(w: &Work) -> (RowForm, f64) {
        let keep: Vec<usize> = (0..w.n).filter(|&j| w.col_alive[j]).collect();
        let mut idx = vec![usize::MAX; w.n];
        for (k, &j) in keep.iter().enumerate() {
            idx[j] = k;
        }
        let pick = |v: &Vec<f64>| keep.iter().map(|&j| v[j]).collect::<Vec<f64>>();
        let mut f = RowForm {
            n: keep.len(),
            rows: Vec::new(),
            lo: Vec::new(),
            up: Vec::new(),
            c: pick(&w.c),
            l: pick(&w.l),
            u: pick(&w.u),
            is_int: keep.iter().map(|&j| w.is_int[j]).collect(),
        };
        for r in 0..w.rows.len() {
            if w.row_alive[r] {
                f.rows.push((
                    w.rows[r].0.iter().map(|&j| idx[j]).collect(),
                    w.rows[r].1.clone(),
                ));
                f.lo.push(w.lo[r]);
                f.up.push(w.up[r]);
            }
        }
        (f, w.offset)
    }

    /// The sparsify step alone is exact: on random integer programs, run the other
    /// rules to a fixpoint, then compare the live problem's optimum (plus the objective
    /// constant) before and after `Work::sparsify`, by enumeration. Counts prove the
    /// rule fired (CLAUDE.md section 6).
    #[test]
    fn sparsify_step_keeps_the_live_problem_exact() {
        let mut r = Lcg(std::env::var("XSEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7));
        let (mut fired, mut compared, mut both_infeasible) = (0usize, 0usize, 0usize);
        let mut sparsified = 0usize;
        for _ in 0..3000 {
            let f = random_ip_sparsify(&mut r);
            let mut w = Work::new(&f, 1e-9);
            for j in 0..f.n {
                if w.is_int[j] {
                    w.tighten(j, w.l[j], w.u[j]);
                }
            }
            for _ in 0..20 {
                if w.infeasible || !w.pass() {
                    break;
                }
            }
            if w.infeasible {
                continue;
            }
            let (g0, o0) = snapshot(&w);
            let before = brute_min(&g0);
            let rows_before = w.stats.sparsified_rows;
            if !w.sparsify() {
                continue;
            }
            fired += 1;
            sparsified += w.stats.sparsified_rows - rows_before;
            if w.infeasible {
                assert!(before.is_none(), "sparsify declared infeasible: {g0:?}");
                continue;
            }
            let (g1, o1) = snapshot(&w);
            let after = brute_min(&g1);
            match (&before, &after) {
                (None, None) => both_infeasible += 1,
                (Some((v0, _)), Some((v1, _))) => {
                    compared += 1;
                    assert!(
                        ((v0 + o0) - (v1 + o1)).abs() <= 1e-9,
                        "sparsify changed the optimum: {} -> {} on {g0:?} -> {g1:?}",
                        v0 + o0,
                        v1 + o1
                    );
                }
                _ => panic!(
                    "sparsify changed feasibility: {before:?} -> {after:?} on {g0:?} -> {g1:?}"
                ),
            }
        }
        eprintln!(
            "sparsify step probe: fired in {fired} cases ({sparsified} rows), optimum compared \
             {compared}, both infeasible {both_infeasible}"
        );
        assert!(
            fired > 400 && sparsified > 1000 && compared > 300,
            "probe fired too little: fired {fired} rows {sparsified} compared {compared}"
        );
    }

    /// `solve_presolved`'s contract on one presolve result: presolve never calls a
    /// feasible problem infeasible; the reduced optimum (plus offset) is never above the
    /// true optimum; at the true optimum the postsolved point is feasible with that
    /// value; below it (a relaxation) the postsolved point fails the check on the
    /// original rows, so the driver solves the original instead. Returns whether the
    /// result was exact (`false`: a relaxation the check catches).
    fn driver_contract(f: &RowForm, p: &Presolved, want: &Option<(f64, Vec<f64>)>) -> bool {
        if p.infeasible {
            assert!(
                want.is_none(),
                "presolve declared a feasible problem infeasible: {f:?}"
            );
            return true;
        }
        let got = brute_min(&p.form);
        match (want, &got) {
            (_, None) => {
                assert!(
                    want.is_none(),
                    "reduced problem infeasible, original {want:?}: {f:?}"
                );
                true
            }
            (None, Some((_, gx))) => {
                // A relaxation of an infeasible problem: the check must reject it.
                let x = p.postsolve(gx, f.n);
                assert!(!row_form_feasible(f, &x, 1e-9), "{x:?} feasible for {f:?}");
                false
            }
            (Some((wv, _)), Some((gv, gx))) => {
                let red = gv + p.offset;
                assert!(
                    red <= wv + 1e-9,
                    "reduced optimum {red} above {wv} for {f:?}"
                );
                let x = p.postsolve(gx, f.n);
                if (red - wv).abs() <= 1e-9 {
                    assert!(
                        row_form_feasible(f, &x, 1e-9),
                        "postsolved point infeasible: {x:?} for {f:?}"
                    );
                    let v: f64 = f.c.iter().zip(&x).map(|(c, x)| c * x).sum();
                    assert!((v - wv).abs() <= 1e-9);
                    true
                } else {
                    assert!(
                        !row_form_feasible(f, &x, 1e-9),
                        "relaxed optimum {red} < {wv} with a feasible postsolved point \
                         {x:?} for {f:?}"
                    );
                    false
                }
            }
        }
    }

    /// The whole presolve with sparsify on random integer programs, held to the
    /// contract `solve_presolved` relies on (`driver_contract`). The probe counts rows
    /// actually sparsified (CLAUDE.md section 6).
    ///
    /// The defect this test first exposed (a cancellation leaving a ~4e-16 entry on an
    /// equality row that `doubleton_eq` / the singleton rule then divided by: 12
    /// contract violations in 180,000 cases over seeds 200-259 with sparsify, 3
    /// without) was in presolve itself, not in sparsify, and is fixed by
    /// `Work::is_noise` (`presolve_never_divides_by_a_noise_coefficient`); the same
    /// 60-seed census is now clean (`XSEED` reruns any seed).
    #[test]
    fn sparsify_presolve_keeps_the_driver_contract() {
        let mut r = Lcg(std::env::var("XSEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7));
        let (mut cases, mut feasible, mut infeasible) = (0usize, 0usize, 0usize);
        let (mut sparsified, mut fired_cases, mut nz_removed) = (0usize, 0usize, 0usize);
        let (mut fired_exact, mut relaxed, mut relaxed_off) = (0usize, 0usize, 0usize);
        for _ in 0..3000 {
            let f = random_ip_sparsify(&mut r);
            let want = brute_min(&f);
            cases += 1;
            if want.is_some() {
                feasible += 1;
            } else {
                infeasible += 1;
            }
            let p = presolve_opts(&f, 1e-9, 20, false, true);
            sparsified += p.stats.sparsified_rows;
            nz_removed += p.stats.sparsify_nz_removed;
            let fired = p.stats.sparsified_rows > 0;
            fired_cases += usize::from(fired);
            if !driver_contract(&f, &p, &want) {
                relaxed += 1;
            } else if fired && want.is_some() {
                fired_exact += 1;
            }
            if !driver_contract(&f, &presolve_opts(&f, 1e-9, 20, false, false), &want) {
                relaxed_off += 1;
            }
        }
        eprintln!(
            "sparsify probe: cases {cases} feasible {feasible} infeasible {infeasible} \
             sparsified rows {sparsified} in {fired_cases} cases, net nonzeros removed \
             {nz_removed}; exact with sparsify on feasible cases {fired_exact}; relaxed \
             (caught by the postsolve check) {relaxed} on vs {relaxed_off} off"
        );
        assert!(
            cases == 3000
                && feasible > 1000
                && infeasible > 100
                && sparsified > 1000
                && fired_cases > 400
                && fired_exact > 250
                && nz_removed > 2000,
            "probe fired too little: cases {cases} feasible {feasible} sparsified {sparsified} \
             fired cases {fired_cases} exact {fired_exact} nz removed {nz_removed}"
        );
    }

    /// Sparsify by hand, on the rule alone (no other presolve pass first).
    /// `x0 + x1 + x2 = 1` (binaries) cancels all three of its entries in
    /// `x0 + x1 + x2 + x3 <= 2` (`x3` integer in [0, 3]; the row becomes the singleton
    /// `x3 <= 1`, a bound), and two in `2 x0 + 2 x2 + x4 >= 1` at the price of one
    /// fill-in (`-2 x1 + x4 >= -1`). A continuous equality may not fill in a row with
    /// integers, and no row may receive a general-integer fill-in.
    #[test]
    fn sparsify_cancels_entries_by_hand() {
        let f = RowForm {
            n: 6,
            rows: vec![
                (vec![0, 1, 2], vec![1.0, 1.0, 1.0]),
                (vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]),
                (vec![0, 2, 4], vec![2.0, 2.0, 1.0]),
                (vec![1, 5], vec![1.0, -1.0]),
            ],
            lo: vec![1.0, -INF, 1.0, -1.0],
            up: vec![1.0, 2.0, INF, INF],
            c: vec![1.0, 2.0, 3.0, -1.0, 1.0, 1.0],
            l: vec![0.0; 6],
            u: vec![1.0, 1.0, 1.0, 3.0, 4.0, 1.0],
            is_int: vec![true; 6],
        };
        let mut w = Work::new(&f, 1e-9);
        assert!(w.sparsify());
        assert_eq!(w.stats.sparsified_rows, 2, "{:?}", w.stats);
        // Row 1: three cancelled, no fill-in; row 2: two cancelled, one fill-in.
        assert_eq!(w.stats.sparsify_nz_removed, 4, "{:?}", w.stats);
        // Row 1 became `x3 <= 1` and was turned into a bound.
        assert!(!w.row_alive[1]);
        assert_eq!(w.u[3], 1.0);
        assert_eq!(w.stats.singleton_rows, 1);
        // Row 2: -2 x1 + x4 >= -1; its constant moved by +2 (`shift -= lambda r`).
        let mut e: Vec<(usize, f64)> = w.rows[2]
            .0
            .iter()
            .copied()
            .zip(w.rows[2].1.iter().copied())
            .collect();
        e.sort_by_key(|t| t.0);
        assert_eq!(e, vec![(1, -2.0), (4, 1.0)]);
        assert_eq!((w.lo[2], w.up[2], w.shift[2]), (-1.0, INF, 2.0));
        // The equality and the unrelated row are untouched.
        assert_eq!(w.rows[0], f.rows[0]);
        assert_eq!(w.rows[3], f.rows[3]);
        // The whole presolve with sparsify keeps the optimum.
        let (want, _) = brute_min(&f).expect("feasible");
        // (Measured: here the other rules solve the problem before sparsify's turn; the
        // randomized test above is the end-to-end probe.)
        let p = presolve_opts(&f, 1e-9, 20, false, true);
        assert!(!p.infeasible);
        let (gv, gx) = brute_min(&p.form).expect("reduced feasible");
        assert!((want - (gv + p.offset)).abs() <= 1e-9);
        assert!(row_form_feasible(&f, &p.postsolve(&gx, f.n), 1e-9));

        // A continuous equality may not fill in a row with an integer column. (Row 2
        // only gives z2 a second entry, so that z0 -- not z2 -- is the sparsest column:
        // candidates must hold the sparsest column.)
        let g = RowForm {
            n: 4,
            rows: vec![
                (vec![0, 1, 2], vec![1.0, 1.0, 1.0]),
                (vec![0, 1, 3], vec![1.0, 1.0, 1.0]),
                (vec![2, 3], vec![1.0, -1.0]),
            ],
            lo: vec![1.0, 0.0, -INF],
            up: vec![1.0, INF, 1.0],
            c: vec![0.0; 4],
            l: vec![0.0; 4],
            u: vec![1.0; 4],
            is_int: vec![false, false, false, true],
        };
        let mut w = Work::new(&g, 1e-9);
        assert!(!w.sparsify(), "{:?}", w.rows);
        // ... but it may when the row has no integer column: z0 + z1 + z3 -> -z2 + z3.
        let mut g2 = g.clone();
        g2.is_int[3] = false;
        let mut w = Work::new(&g2, 1e-9);
        assert!(w.sparsify());
        assert_eq!(w.stats.sparsified_rows, 1);
        assert_eq!((w.lo[1], w.shift[1]), (-1.0, 1.0));
        // No fill-in of a general integer: z2 integer in [0, 5] is missing from row 1.
        let mut g3 = g.clone();
        g3.is_int = vec![true; 4];
        g3.u[2] = 5.0;
        let mut w = Work::new(&g3, 1e-9);
        assert!(!w.sparsify(), "{:?}", w.rows);
        // A binary z2 may be filled in.
        g3.u[2] = 1.0;
        let mut w = Work::new(&g3, 1e-9);
        assert!(w.sparsify());
    }
}
