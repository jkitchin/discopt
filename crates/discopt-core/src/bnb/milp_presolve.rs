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
//! - parallel rows (found by hashing the normalized row, not pairwise) are merged.
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
    /// Passes over the rule set.
    pub passes: usize,
}

/// The presolved problem and how to map its solutions back.
#[derive(Clone, Debug)]
pub struct Presolved {
    /// The reduced problem (columns renumbered `0..keep_cols.len()`).
    pub form: RowForm,
    /// Original index of each reduced column.
    pub keep_cols: Vec<usize>,
    /// Columns removed, with their fixed values (original indices).
    pub fixed: Vec<(usize, f64)>,
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
        for &(j, v) in &self.fixed {
            x[j] = v;
        }
        x
    }
}

struct Work {
    n: usize,
    rows: Vec<(Vec<usize>, Vec<f64>)>,
    lo: Vec<f64>,
    up: Vec<f64>,
    row_alive: Vec<bool>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    is_int: Vec<bool>,
    col_alive: Vec<bool>,
    /// Rows each column appears in (may include dead rows).
    col_rows: Vec<Vec<usize>>,
    fixed: Vec<(usize, f64)>,
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

    fn tol(&self, v: f64) -> f64 {
        self.feastol * (1.0 + v.abs())
    }

    fn fix_col(&mut self, j: usize, v: f64) {
        if !self.col_alive[j] {
            return;
        }
        let v = if self.is_int[j] { v.round() } else { v };
        self.col_alive[j] = false;
        self.fixed.push((j, v));
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
        row_alive: vec![true; f.rows.len()],
        c: f.c.clone(),
        l: f.l.clone(),
        u: f.u.clone(),
        is_int: f.is_int.clone(),
        col_alive: vec![true; f.n],
        col_rows,
        fixed: Vec::new(),
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
    for r in 0..w.rows.len() {
        if !w.row_alive[r] {
            continue;
        }
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
        fixed: w.fixed,
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
        RowForm {
            n,
            rows,
            lo,
            up,
            c,
            l,
            u,
            is_int: vec![true; n],
        }
    }

    #[test]
    fn presolve_preserves_the_optimum_and_postsolve_is_feasible() {
        let mut r = Lcg(2024);
        let (mut cases, mut reduced, mut infeasible, mut feasible) =
            (0usize, 0usize, 0usize, 0usize);
        for _ in 0..3000 {
            let f = random_ip(&mut r);
            let want = brute_min(&f);
            let p = presolve(&f, 1e-9, 20);
            cases += 1;
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
            cases == 3000 && reduced > 1000 && infeasible > 50 && feasible > 500,
            "probe fired too little: reduced {reduced} infeasible {infeasible} feasible {feasible}"
        );
    }
}
