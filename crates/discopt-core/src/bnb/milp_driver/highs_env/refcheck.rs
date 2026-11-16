//! Debug instrument for the HiGHS-design search (`DISCOPT_HIGHS_SEARCH_CHECK_POINT`).
//!
//! The file names one value per structural column of a known feasible point (for a
//! false-certificate hunt: an optimal one), in the driver's column space (the
//! structural columns `solve_milp_csc` receives, after Python's fixed-column
//! removal). The top-level search then checks, against that point:
//!
//! * every cut entering its pool or its own LP, before and after the rewrite over the
//!   base columns, and the whole LP after every row change (rows, column bounds);
//! * every bound given to a box holding the point (node evaluation, separation lift,
//!   strong-branching commits and updates, children, queued nodes) -- with the
//!   incumbent no better than the point, none may exceed the point's value;
//! * every reduction of a domain holding the point (propagation after branching,
//!   backtracking or queueing, reduced-cost fixing in an evaluation, install of a
//!   queued node): it must keep the point;
//! * every queued node holding the point: it leaves the queue only by being popped.
//!
//! Each failure prints one `HIGHS_SEARCH_INVALID` line on stderr, and the final bound
//! is judged too (`kind=final`); each event on the point's path prints a
//! `HIGHS_SEARCH_TRACE` line, so the last one names where the path ended. A summary
//! with the counts prints at the end of the search, so a run that checked nothing
//! says so. Off when unset; it never changes what the search does. A point that does
//! not satisfy the search's base rows is a mapping error and panics (the instrument
//! must not silently check nothing). Used to find the suboptimal-node bound defect
//! fixed in [`super::HighsEnv::dual_bound`].

use super::*;

pub(super) struct RefCheck {
    /// The point over the base working columns (structural, then the slacks solved
    /// from the rows).
    pub x: Vec<f64>,
    /// Its objective value, `obj_const` included.
    pub obj: f64,
    pub cuts_checked: usize,
    pub cuts_bad: usize,
    pub lp_checks: usize,
    pub lp_bad: usize,
    pub nodes_checked: usize,
    pub nodes_containing: usize,
    pub bounds_bad: usize,
    pub queue_checked: usize,
    pub queue_containing: usize,
    /// Queue slots holding a node whose box contains the point.
    pub slots: std::collections::HashSet<i64>,
}

/// The box `l..u` with a node's queued bound changes applied.
pub(super) fn stack_box(l: &[f64], u: &[f64], stack: &[QDomainChange]) -> (Vec<f64>, Vec<f64>) {
    let (mut lo, mut hi) = (l.to_vec(), u.to_vec());
    for c in stack {
        let j = c.col as usize;
        if c.upper {
            hi[j] = hi[j].min(c.val);
        } else {
            lo[j] = lo[j].max(c.val);
        }
    }
    (lo, hi)
}

fn tol_of(v: f64) -> f64 {
    1e-6 * (1.0 + v.abs())
}

impl RefCheck {
    /// The instrument for this search, or `None` (unset, or a sub-MIP whose columns
    /// are not the point's).
    pub fn load(ctx: &NodeCtx<'_>) -> Option<RefCheck> {
        let path = std::env::var("DISCOPT_HIGHS_SEARCH_CHECK_POINT").ok()?;
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("check point {path}: {e}"));
        let xs: Vec<f64> = text
            .split_ascii_whitespace()
            .map(|t| t.parse::<f64>().expect("check point value"))
            .collect();
        // Only the top-level solve's search has the point's box: a sub-MIP (depth > 0,
        // fixings, no ported separators) does not.
        let top = ctx.highs_node.as_ref().is_some_and(|h| h.top_level);
        if !top {
            eprintln!("HIGHS_SEARCH_CHECK skip: not the top-level search");
            return None;
        }
        assert_eq!(
            xs.len(),
            ctx.ns,
            "HIGHS_SEARCH_CHECK: point length vs the search's structural columns"
        );
        let (m, n, ns) = (ctx.m_w, ctx.n_w, ctx.ns);
        let csc = ctx.csc_rc;
        let mut x = xs.clone();
        x.resize(n, f64::NAN);
        // Solve the slacks from the rows: repeatedly, a row with one unknown column.
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for j in 0..n {
            let (ri, rv) = csc.col(j);
            for (&i, &v) in ri.iter().zip(rv) {
                if v != 0.0 {
                    rows[i].push((j, v));
                }
            }
        }
        loop {
            let mut progress = false;
            for (i, row) in rows.iter().enumerate() {
                let unknown: Vec<(usize, f64)> = row
                    .iter()
                    .copied()
                    .filter(|&(j, _)| x[j].is_nan())
                    .collect();
                if unknown.len() != 1 {
                    continue;
                }
                let (k, a) = unknown[0];
                let act: f64 = row
                    .iter()
                    .filter(|&&(j, _)| j != k)
                    .map(|&(j, v)| v * x[j])
                    .sum();
                x[k] = (ctx.b_w[i] - act) / a;
                progress = true;
            }
            if !progress {
                break;
            }
        }
        if let Some(j) = (0..n).find(|&j| x[j].is_nan()) {
            panic!(
                "HIGHS_SEARCH_CHECK: base column {j} (of {n}, ns {ns}) not determined by the rows"
            );
        }
        let mut resid = 0.0f64;
        for (i, row) in rows.iter().enumerate() {
            let act: f64 = row.iter().map(|&(j, v)| v * x[j]).sum();
            resid = resid.max((act - ctx.b_w[i]).abs() / (1.0 + ctx.b_w[i].abs()));
        }
        let bviol = x
            .iter()
            .zip(&ctx.l_w[..n])
            .zip(&ctx.u_w[..n])
            .map(|((&v, &l), &u)| (l - v).max(v - u))
            .fold(0.0f64, f64::max);
        let iviol = (0..ns)
            .filter(|&j| ctx.is_int[j])
            .map(|j| (x[j] - x[j].round()).abs())
            .fold(0.0f64, f64::max);
        let obj: f64 = (0..n).map(|j| ctx.c_w[j] * x[j]).sum::<f64>() + ctx.obj_const;
        eprintln!(
            "HIGHS_SEARCH_CHECK base: ns={ns} n_w={n} m_w={m} row_resid={resid:e} \
             box_viol={bviol:e} int_viol={iviol:e} obj={obj:.12}"
        );
        if resid > 1e-7 || iviol > 1e-6 {
            panic!("HIGHS_SEARCH_CHECK: the point does not satisfy the search's base rows (wrong column map?)");
        }
        if bviol > 1e-7 {
            eprintln!("HIGHS_SEARCH_INVALID kind=base-box viol={bviol:e}");
        }
        Some(RefCheck {
            x,
            obj,
            cuts_checked: 0,
            cuts_bad: 0,
            lp_checks: 0,
            lp_bad: 0,
            nodes_checked: 0,
            nodes_containing: 0,
            bounds_bad: 0,
            queue_checked: 0,
            queue_containing: 0,
            slots: std::collections::HashSet::new(),
        })
    }

    /// The point over the search LP's current columns: the base columns, then each cut
    /// row's surplus `a·x - rhs`.
    pub fn current(&self, lp: Option<&SearchLp>) -> Vec<f64> {
        let Some(lp) = lp else { return self.x.clone() };
        let mut x = self.x.clone();
        x.resize(lp.n, f64::NAN);
        for r in &lp.rows {
            let act: f64 = r.cut.coeffs.iter().zip(&self.x).map(|(a, v)| a * v).sum();
            x[r.col] = act - r.cut.rhs;
        }
        x
    }

    /// Check one `coeffs·x >= rhs` cut over the columns of `x`.
    pub fn cut(&mut self, src: &str, node: i64, cut: &GomoryCut, x: &[f64]) {
        self.cuts_checked += 1;
        let mut act = 0.0;
        let mut len = 0;
        for (j, &a) in cut.coeffs.iter().enumerate() {
            if a != 0.0 {
                len += 1;
                let v = x.get(j).copied().unwrap_or(f64::NAN);
                act += a * v;
            }
        }
        if act.is_nan() || act < cut.rhs - tol_of(cut.rhs) {
            self.cuts_bad += 1;
            let beyond = cut
                .coeffs
                .iter()
                .enumerate()
                .filter(|(j, &a)| a != 0.0 && *j >= self.x.len())
                .count();
            eprintln!(
                "HIGHS_SEARCH_INVALID kind=cut src={src} node={node} len={len} \
                 nz_beyond_base={beyond} ncoef={} act={act:e} rhs={:e} viol={:e}",
                cut.coeffs.len(),
                cut.rhs,
                cut.rhs - act
            );
        }
    }

    /// Check the search LP's whole data at the point: every row's equality and every
    /// column's bounds.
    pub fn lp(&mut self, src: &str, node: i64, lp: &SearchLp) {
        self.lp_checks += 1;
        let x = self.current(Some(lp));
        let mut act = vec![0.0f64; lp.m];
        for (j, &xj) in x.iter().enumerate().take(lp.n) {
            let (ri, rv) = lp.csc.col(j);
            for (&i, &v) in ri.iter().zip(rv) {
                act[i] += v * xj;
            }
        }
        let mut worst = (0.0f64, usize::MAX);
        for (i, (&a, &b)) in act.iter().zip(&lp.b).enumerate() {
            let r = (a - b).abs() / (1.0 + b.abs());
            if r > worst.0 {
                worst = (r, i);
            }
        }
        let mut bworst = (0.0f64, usize::MAX);
        for (j, ((&v, &l), &u)) in x.iter().zip(&lp.l).zip(&lp.u).enumerate() {
            let v = (l - v).max(v - u);
            if v > bworst.0 {
                bworst = (v, j);
            }
        }
        if worst.0 > 1e-7 || bworst.0 > 1e-6 {
            self.lp_bad += 1;
            eprintln!(
                "HIGHS_SEARCH_INVALID kind=lp src={src} node={node} m={} n={} rows={} \
                 row_resid={:e}@{} box_viol={:e}@{}",
                lp.m,
                lp.n,
                lp.rows.len(),
                worst.0,
                worst.1,
                bworst.0,
                bworst.1
            );
        }
    }

    /// Whether node bounds are judged at this cutoff: the point (and every node box
    /// holding it) stays in the search unless an incumbent strictly better than the
    /// point exists, which for an optimal point never happens. A cutoff at the
    /// point's own value keeps the checks on: a valid LP bound on a box holding the
    /// point never exceeds its value, whatever the incumbent.
    pub fn active(&self, cutoff: f64) -> bool {
        cutoff > self.obj - 1e-9 * (1.0 + self.obj.abs())
    }

    /// Whether `lo..hi` (structural) contains the point.
    pub fn contains(&self, lo: &[f64], hi: &[f64]) -> bool {
        lo.iter()
            .zip(hi)
            .zip(&self.x)
            .all(|((&l, &u), &v)| v >= l - 1e-9 && v <= u + 1e-9)
    }

    /// A bound on a box containing the point must not exceed the point's value.
    pub fn bound(&mut self, kind: &str, node: i64, bound: f64, cutoff: f64, what: &str) {
        if bound > self.obj + 1e-9 * (1.0 + self.obj.abs()) {
            self.bounds_bad += 1;
            eprintln!(
                "HIGHS_SEARCH_INVALID kind={kind} node={node} bound={bound:.12} ref={:.12} \
                 excess={:e} cutoff={cutoff:.12} {what}",
                self.obj,
                bound - self.obj
            );
        }
    }

    pub fn summary(&self, final_bound: f64) {
        eprintln!(
            "HIGHS_SEARCH_CHECK summary: ref={:.12} final_bound={final_bound:.12} cuts={} \
             cuts_bad={} lp_checks={} lp_bad={} nodes={} nodes_containing={} queued={} \
             queued_containing={} bounds_bad={}",
            self.obj,
            self.cuts_checked,
            self.cuts_bad,
            self.lp_checks,
            self.lp_bad,
            self.nodes_checked,
            self.nodes_containing,
            self.queue_checked,
            self.queue_containing,
            self.bounds_bad
        );
        if final_bound > self.obj + 1e-9 * (1.0 + self.obj.abs()) {
            eprintln!(
                "HIGHS_SEARCH_INVALID kind=final bound={final_bound:.12} ref={:.12}",
                self.obj
            );
        }
    }
}
