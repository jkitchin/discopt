//! Probing on binary columns and the implications it records: a translation of the
//! probing half of HiGHS's `HighsImplications` (`computeImplications`, `runProbing`,
//! `addImplication`, `applyImplications`; `highs/mip/HighsImplications.cpp`).
//!
//! Derived from HiGHS 1.16 (commit `f3f113d`), Copyright (c) 2016-2025 the HiGHS
//! authors, MIT License; see `THIRD_PARTY_NOTICES.md`.
//!
//! Probing fixes a binary column on the *global* domain, propagates, records what the
//! fixing implied and backtracks. An implied fixing of another binary becomes a size-2
//! clique (`cliquetable.addClique`); a bound implied on a non-binary column is stored
//! and re-applied whenever the probed literal is fixed later
//! (`applyImplications`, called from [`Domain::change_bound`]). A probe that empties the
//! domain fixes the column to its other value (`cliquetable.vertexInfeasible`), and a
//! column whose two probes both bound another column tightens that column globally to
//! the hull of the two.
//!
//! Every conclusion is model-implied over the global box at the time of the probe, so it
//! holds for every point of that box satisfying the rows (and, with an objective cutoff
//! on the domain, for every improving one, as in HiGHS). What is *not* translated:
//! `dualFixProbing` (presolve only), lifting opportunities, the variable bounds
//! `addVLB`/`addVUB` derive from the implications (discopt's search keeps its own
//! `VarBoundStore`), and the substitution `runProbing` records when a column is fixed
//! to different values by the two probes (the search cannot substitute a column; the
//! hull tightening it would otherwise do is a no-op on such a column, so it is
//! skipped).

use std::collections::BTreeMap;
use std::sync::Arc;

use super::clique::{CliqueTable, CliqueVar, GlobalDomainView, K_HIGHS_IINF};
use super::domain::{BoundType, Domain, DomainChange, Reason};

/// `HighsImplications::Implication`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Implication {
    /// Implied lower bound (`-inf` for none).
    pub lb: f64,
    /// Implied upper bound (`+inf` for none).
    pub ub: f64,
}

/// `HighsImplications` (the probing half).
#[derive(Clone, Debug)]
pub struct Implications {
    /// `implications[ImplIdx{col, val}]`, index `2 * col + val`: implied column ->
    /// implied bounds.
    implications: Vec<BTreeMap<usize, Implication>>,
    /// `hasProbed`, index `2 * col + val`.
    has_probed: Vec<bool>,
    implications_up: Vec<DomainChange>,
    implications_down: Vec<DomainChange>,
    /// `numImplications`.
    pub num_implications: i64,
}

/// What a probing pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProbeStats {
    /// Columns `runProbing` probed.
    pub probed: usize,
    /// Global bound changes (infeasible probes and hull tightenings).
    pub reductions: usize,
    /// Clique entries the probes added.
    pub new_entries: i64,
}

impl Implications {
    /// `HighsImplications(ncols)`.
    pub fn new(ncols: usize) -> Self {
        Implications {
            implications: vec![BTreeMap::new(); 2 * ncols],
            has_probed: vec![false; 2 * ncols],
            implications_up: Vec::new(),
            implications_down: Vec::new(),
            num_implications: 0,
        }
    }

    /// `probedBefore(col, val)`.
    pub fn probed_before(&self, col: usize, val: u8) -> bool {
        self.has_probed[2 * col + val as usize]
    }

    /// The implications of `col = val`.
    pub fn implications_of(&self, col: usize, val: u8) -> &BTreeMap<usize, Implication> {
        &self.implications[2 * col + val as usize]
    }

    /// `addImplication(idx, implCol, implic)`.
    fn add_implication(&mut self, idx: usize, implcol: usize, implic: Implication) {
        match self.implications[idx].entry(implcol) {
            std::collections::btree_map::Entry::Occupied(mut e) => {
                let cur = e.get_mut();
                if cur.lb == f64::NEG_INFINITY && implic.lb != f64::NEG_INFINITY {
                    self.num_implications += 1;
                }
                if cur.ub == f64::INFINITY && implic.ub != f64::INFINITY {
                    self.num_implications += 1;
                }
                cur.lb = cur.lb.max(implic.lb);
                cur.ub = cur.ub.min(implic.ub);
            }
            std::collections::btree_map::Entry::Vacant(e) => {
                if implic.lb != f64::NEG_INFINITY {
                    self.num_implications += 1;
                }
                if implic.ub != f64::INFINITY {
                    self.num_implications += 1;
                }
                e.insert(implic);
            }
        }
    }

    /// `applyImplications(domain, col, val)`: `col` was fixed to `val`; tighten every
    /// column the probe of that literal bounded, with reason `cliqueTable(col, val)`
    /// (conflict analysis explains it by the fixing of `col`). Integer columns take
    /// every bound tighter by more than `feastol`; a continuous column only one that
    /// fixes it.
    pub fn apply_implications(&self, dom: &mut Domain, col: usize, val: u8) {
        debug_assert!(dom.col_lower[col] == dom.col_upper[col]);
        let feastol = dom.feastol();
        let reason = Reason::CliqueTable(col, val as usize);
        for (&implcol, implic) in &self.implications[2 * col + val as usize] {
            debug_assert!(!dom.infeasible());
            if dom.col_lower[implcol] == dom.col_upper[implcol] {
                continue;
            }
            let isint = dom.is_int(implcol);
            if (!isint && implic.lb > dom.col_upper[implcol] - feastol)
                || (isint && implic.lb > dom.col_lower[implcol] + feastol)
            {
                dom.change_bound(
                    DomainChange {
                        boundval: implic.lb,
                        column: implcol,
                        boundtype: BoundType::Lower,
                    },
                    reason,
                );
                if dom.infeasible() {
                    return;
                }
            }
            if (!isint && implic.ub < dom.col_lower[implcol] + feastol)
                || (isint && implic.ub < dom.col_upper[implcol] - feastol)
            {
                dom.change_bound(
                    DomainChange {
                        boundval: implic.ub,
                        column: implcol,
                        boundtype: BoundType::Upper,
                    },
                    reason,
                );
            }
            if dom.infeasible() {
                return;
            }
        }
    }
}

/// A clique view of `dom` (the table's domain callbacks), with `is_int` borrowed apart.
fn view<'a>(dom: &'a mut Domain, is_int: &'a [bool]) -> GlobalDomainView<'a> {
    let feastol = dom.feastol();
    let nnz = dom.num_model_nonzeros();
    GlobalDomainView {
        dom,
        is_int,
        feastol,
        num_model_nonzeros: nnz,
    }
}

/// The global domain, the clique table and the implications probing works on. The
/// domain must be at its global baseline (an empty change stack): probing undoes each
/// probe with `backtrack()`, and commits the global reductions it finds as the new
/// baseline.
pub struct Prober<'a> {
    /// The global domain.
    pub dom: &'a mut Domain,
    /// `mipdata_->cliquetable`.
    pub table: &'a mut CliqueTable,
    /// `mipdata_->implications`.
    pub implics: &'a mut Implications,
    is_int: Vec<bool>,
}

impl<'a> Prober<'a> {
    /// Probing over `dom`, which must carry no clique table or implications of its
    /// own (they are `table` and `implics` here, and grow while probing).
    pub fn new(
        dom: &'a mut Domain,
        table: &'a mut CliqueTable,
        implics: &'a mut Implications,
    ) -> Self {
        assert!(
            dom.domain_change_stack().is_empty(),
            "probing needs the domain at its global baseline"
        );
        assert!(
            !dom.has_clique_table(),
            "probing owns the clique table; detach it from the domain first"
        );
        let n = dom.col_lower.len();
        let is_int = (0..n).map(|j| dom.is_int(j)).collect();
        Prober {
            dom,
            table,
            implics,
            is_int,
        }
    }

    fn is_binary(&self, col: usize) -> bool {
        self.dom.is_binary(col)
    }

    fn is_fixed(&self, col: usize) -> bool {
        self.dom.col_lower[col] == self.dom.col_upper[col]
    }

    /// Make the global reductions found so far the domain's baseline, so the next
    /// probe's `backtrack()` stops above them.
    fn commit(&mut self) {
        if !self.dom.infeasible() && !self.dom.domain_change_stack().is_empty() {
            self.dom.commit_as_global();
        }
    }

    /// `computeImplications(col, val)`: true when probing ended early (domain
    /// infeasible, `col` fixed, or the probe infeasible and `col` fixed the other way).
    pub fn compute_implications(&mut self, col: usize, val: u8) -> bool {
        self.dom.propagate();
        self.commit();
        if self.dom.infeasible() || self.is_fixed(col) {
            return true;
        }
        let changedend = self.dom.changed_cols().len();
        let stackimplicstart = self.dom.domain_change_stack().len() + 1;
        let probe = if val == 1 {
            DomainChange {
                boundval: 1.0,
                column: col,
                boundtype: BoundType::Lower,
            }
        } else {
            DomainChange {
                boundval: 0.0,
                column: col,
                boundtype: BoundType::Upper,
            }
        };
        // The probe is a branching change, so `backtrack()` undoes exactly it and what
        // it implied.
        self.dom.change_bound(probe, Reason::Branching);
        if !self.dom.infeasible() {
            self.dom.propagate();
        }
        if self.dom.infeasible() {
            self.dom.backtrack();
            self.dom.clear_changed_cols_from(changedend);
            let mut v = view(self.dom, &self.is_int);
            self.table.vertex_infeasible(&mut v, col, val);
            self.commit();
            return true;
        }
        let stackimplicend = self.dom.domain_change_stack().len();
        let num_entries = self.table.num_entries() as i64;
        let max_entries = 100_000 + self.dom.num_model_nonzeros() as i64;
        let mut implics: Vec<DomainChange> = Vec::new();
        {
            let stack = self.dom.domain_change_stack();
            let reasons = self.dom.domain_change_reasons();
            for i in stackimplicstart..stackimplicend {
                if let Reason::CliqueTable(c, _) = reasons[i] {
                    if c == col || num_entries >= max_entries {
                        continue;
                    }
                }
                implics.push(stack[i]);
            }
        }
        self.dom.backtrack();
        self.dom.clear_changed_cols_from(changedend);
        debug_assert!(self.dom.domain_change_stack().is_empty());

        // Implications on binaries become size-2 cliques; `partition` keeps the
        // non-binary ones first.
        let (mut nonbin, bin): (Vec<DomainChange>, Vec<DomainChange>) =
            implics.into_iter().partition(|a| !self.is_binary(a.column));
        for i in &bin {
            let other = if i.boundtype == BoundType::Lower {
                CliqueVar::new(i.column, 0)
            } else {
                CliqueVar::new(i.column, 1)
            };
            let clique = [CliqueVar::new(col, val), other];
            {
                let mut v = view(self.dom, &self.is_int);
                self.table.add_clique(&mut v, &clique, false, K_HIGHS_IINF);
            }
            self.commit();
            if self.dom.infeasible() || self.is_fixed(col) {
                return true;
            }
        }

        let idx = 2 * col + val as usize;
        self.implics.has_probed[idx] = true;
        for i in &nonbin {
            let implication = if i.boundtype == BoundType::Lower {
                Implication {
                    lb: i.boundval,
                    ub: f64::INFINITY,
                }
            } else {
                Implication {
                    lb: f64::NEG_INFINITY,
                    ub: i.boundval,
                }
            };
            self.implics.add_implication(idx, i.column, implication);
        }
        // `pdqsort` by column; stable here, so equal columns keep stack order.
        nonbin.sort_by_key(|a| a.column);
        if val == 1 {
            self.implics.implications_up = nonbin;
        } else {
            self.implics.implications_down = nonbin;
        }
        false
    }

    /// `runProbing(col, numReductions)`: probe both values of binary `col` and tighten
    /// every column both probes bound to the hull of the two. True when `col` was
    /// probed (or probing ended early).
    pub fn run_probing(&mut self, col: usize, num_reductions: &mut usize) -> bool {
        if !self.is_binary(col)
            || self.implics.probed_before(col, 1)
            || self.implics.probed_before(col, 0)
            || self.table.substitution(col).is_some()
        {
            return false;
        }
        let fixed_before = self.dom.infeasible();
        if self.compute_implications(col, 1) || self.dom.infeasible() {
            if !fixed_before && self.is_fixed(col) {
                *num_reductions += 1;
            }
            return true;
        }
        if self.table.substitution(col).is_some() {
            return true;
        }
        if self.compute_implications(col, 0) || self.dom.infeasible() {
            if self.is_fixed(col) {
                *num_reductions += 1;
            }
            return true;
        }
        if self.table.substitution(col).is_some() {
            return true;
        }
        let up = std::mem::take(&mut self.implics.implications_up);
        let down = std::mem::take(&mut self.implics.implications_down);
        let (nup, ndown) = (up.len(), down.len());
        let (mut u, mut d) = (0usize, 0usize);
        while u < nup && d < ndown {
            if up[u].column < down[d].column {
                u += 1;
            } else if down[d].column < up[u].column {
                d += 1;
            } else {
                let implcol = up[u].column;
                let mut lb_down = self.dom.col_lower[implcol];
                let mut ub_down = self.dom.col_upper[implcol];
                let mut lb_up = lb_down;
                let mut ub_up = ub_down;
                while d < ndown && down[d].column == implcol {
                    if down[d].boundtype == BoundType::Lower {
                        lb_down = lb_down.max(down[d].boundval);
                    } else {
                        ub_down = ub_down.min(down[d].boundval);
                    }
                    d += 1;
                }
                while u < nup && up[u].column == implcol {
                    if up[u].boundtype == BoundType::Lower {
                        lb_up = lb_up.max(up[u].boundval);
                    } else {
                        ub_up = ub_up.min(up[u].boundval);
                    }
                    u += 1;
                }
                if self.table.substitution(implcol).is_some() || self.is_fixed(implcol) {
                    continue;
                }
                if lb_down == ub_down
                    && lb_up == ub_up
                    && (lb_down - lb_up).abs() > self.dom.feastol()
                {
                    // `implcol = lb_down + (lb_up - lb_down) * col`: HiGHS substitutes
                    // the column out; the search keeps it (see the module docs).
                    continue;
                }
                let lb = lb_down.min(lb_up);
                let ub = ub_down.max(ub_up);
                if lb > self.dom.col_lower[implcol] {
                    self.dom.change_bound(
                        DomainChange {
                            boundval: lb,
                            column: implcol,
                            boundtype: BoundType::Lower,
                        },
                        Reason::Unspecified,
                    );
                    *num_reductions += 1;
                    if self.dom.infeasible() {
                        return true;
                    }
                }
                if ub < self.dom.col_upper[implcol] {
                    self.dom.change_bound(
                        DomainChange {
                            boundval: ub,
                            column: implcol,
                            boundtype: BoundType::Upper,
                        },
                        Reason::Unspecified,
                    );
                    *num_reductions += 1;
                    if self.dom.infeasible() {
                        return true;
                    }
                }
                self.commit();
            }
        }
        self.implics.has_probed[2 * col] = true;
        self.implics.has_probed[2 * col + 1] = true;
        true
    }

    /// The probing loop of `separateImpliedBounds`: probe every candidate binary not
    /// probed yet, until the clique table is full or the domain infeasible, then
    /// `runCliqueMerging` when `next_cleanup_call` drops below zero (it is decreased by
    /// the entries added). Candidates are the LP's fractional integers in HiGHS.
    /// `entries_after_presolve` is `numCliqueEntriesAfterFirstPresolve`.
    pub fn probe_candidates(
        &mut self,
        cands: &[usize],
        next_cleanup_call: &mut i64,
        entries_after_presolve: i64,
    ) -> ProbeStats {
        let mut st = ProbeStats::default();
        if self.table.is_full() {
            return st;
        }
        let old_entries = self.table.num_entries() as i64;
        for &col in cands {
            if self.dom.col_lower[col] != 0.0
                || self.dom.col_upper[col] != 1.0
                || (self.implics.probed_before(col, 0) && self.implics.probed_before(col, 1))
            {
                continue;
            }
            if self.run_probing(col, &mut st.reductions) {
                st.probed += 1;
                if self.dom.infeasible() {
                    return st;
                }
            }
            if self.table.is_full() {
                break;
            }
        }
        st.new_entries = self.table.num_entries() as i64 - old_entries;
        *next_cleanup_call -= st.new_entries.max(0);
        if *next_cleanup_call < 0 {
            {
                let mut v = view(self.dom, &self.is_int);
                self.table.run_clique_merging(&mut v);
            }
            self.commit();
            *next_cleanup_call = entries_after_presolve.min(self.table.num_entries() as i64);
        }
        st
    }
}

/// Probe `cands` on `dom` (at its global baseline), extending `table` and recording
/// implications, then attach both to `dom` so its `change_bound` uses them. Returns
/// what probing did; the domain may come back infeasible (the problem has no point,
/// or none better than the cutoff).
pub fn probe_and_attach(dom: &mut Domain, mut table: CliqueTable, cands: &[usize]) -> ProbeStats {
    let n = dom.col_lower.len();
    let mut implics = Implications::new(n);
    // `nextCleanupCall = numNonzero()`.
    let mut next_cleanup = dom.num_model_nonzeros() as i64;
    let entries0 = table.num_entries() as i64;
    let st = {
        let mut p = Prober::new(dom, &mut table, &mut implics);
        p.probe_candidates(cands, &mut next_cleanup, entries0)
    };
    if !dom.infeasible() {
        dom.set_clique_table(Some(Arc::new(table)));
        dom.set_implications(Some(Arc::new(implics)));
    }
    st
}

#[cfg(test)]
mod tests {
    use super::super::clique::{CliqueRows, CliqueTable};
    use super::super::domain::{Domain, DomainModel};
    use super::super::random::HighsRandom;
    use super::*;
    use std::sync::Arc;

    const INF: f64 = f64::INFINITY;

    /// A random small program: `nb` binaries plus, sometimes, one integer or
    /// continuous column; packing/partitioning rows and general knapsack-like rows.
    #[allow(clippy::type_complexity)]
    fn random_instance(
        rng: &mut HighsRandom,
    ) -> (
        usize,
        Vec<bool>,
        Vec<f64>,
        Vec<f64>,
        Vec<(Vec<usize>, Vec<f64>)>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let nb = 5 + rng.integer_below(4) as usize;
        let has_extra = rng.bit();
        let n = nb + has_extra as usize;
        let mut is_int = vec![true; n];
        let lb0 = vec![0.0; n];
        let mut ub0 = vec![1.0; n];
        if has_extra {
            ub0[nb] = 3.0;
            if rng.bit() {
                is_int[nb] = false;
                ub0[nb] = 2.0;
            }
        }
        let nrows = 2 + rng.integer_below(4) as usize;
        let mut rows: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
        let (mut lo, mut up) = (Vec::new(), Vec::new());
        for _ in 0..nrows {
            let kind = rng.integer_below(4);
            let len = 2 + rng.integer_below(3) as usize;
            let mut cols: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut cols);
            cols.truncate(len.min(n));
            if kind <= 1 {
                cols.retain(|&c| c < nb);
                let mut vals = Vec::new();
                let mut rhs = 1.0;
                for _ in &cols {
                    if rng.integer_below(4) == 0 {
                        vals.push(-1.0);
                        rhs -= 1.0;
                    } else {
                        vals.push(1.0);
                    }
                }
                let eq = kind == 1 && rng.bit();
                lo.push(if eq { rhs } else { -INF });
                up.push(rhs);
                rows.push((cols, vals));
            } else {
                let vals: Vec<f64> = cols
                    .iter()
                    .map(|_| {
                        let a = 1 + rng.integer_below(6);
                        if rng.integer_below(3) == 0 {
                            -(a as f64)
                        } else {
                            a as f64
                        }
                    })
                    .collect();
                let pos: f64 = vals.iter().filter(|v| **v > 0.0).sum();
                let r = (pos * rng.fraction()).floor();
                if rng.integer_below(4) == 0 {
                    lo.push(r - 2.0);
                    up.push(INF);
                } else {
                    lo.push(-INF);
                    up.push(r);
                }
                rows.push((cols, vals));
            }
        }
        (nb, is_int, lb0, ub0, rows, lo, up)
    }

    /// Every feasible point (the extra column on a grid: integers, or steps of 0.25).
    fn feasible_points(
        nb: usize,
        is_int: &[bool],
        ub0: &[f64],
        rows: &[(Vec<usize>, Vec<f64>)],
        lo: &[f64],
        up: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = is_int.len();
        let extra: Vec<f64> = if n > nb {
            if is_int[nb] {
                (0..=ub0[nb] as usize).map(|k| k as f64).collect()
            } else {
                (0..=(4.0 * ub0[nb]) as usize)
                    .map(|k| 0.25 * k as f64)
                    .collect()
            }
        } else {
            vec![0.0]
        };
        let mut points = Vec::new();
        for mask in 0..(1u32 << nb) {
            for &e in &extra {
                let mut x: Vec<f64> = (0..nb).map(|j| ((mask >> j) & 1) as f64).collect();
                if n > nb {
                    x.push(e);
                }
                let ok = rows
                    .iter()
                    .zip(lo.iter().zip(up))
                    .all(|((c, v), (&l, &u))| {
                        let a: f64 = c.iter().zip(v).map(|(&j, &w)| w * x[j]).sum();
                        a >= l - 1e-9 && a <= u + 1e-9
                    });
                if ok {
                    points.push(x);
                }
            }
        }
        points
    }

    /// Soundness of probing and of the fixing hook it feeds, by enumeration over 400
    /// random small programs. After `probe_and_attach` over every binary:
    /// - no feasible point leaves the global box (infeasible probes, hull tightening);
    /// - every clique in the table holds at every feasible point;
    /// - every stored implication of `col = val` holds at every feasible point with
    ///   `x[col] = val`;
    /// - fixing the binaries of a feasible point one by one on the domain (each fixing
    ///   runs `addImplications` and `applyImplications`, then propagation) never makes
    ///   the domain infeasible and never moves a bound past the point.
    ///
    /// And when probing reports the domain infeasible, there is no feasible point.
    /// The executed-check counts are asserted (CLAUDE.md §6).
    #[test]
    fn probing_never_cuts_a_feasible_point() {
        let mut rng = HighsRandom::new(4242);
        let (mut instances, mut infeasible_proofs, mut clique_checks) = (0usize, 0usize, 0usize);
        let (mut impl_checks, mut fix_checks, mut hook_fixings, mut reductions) =
            (0usize, 0usize, 0usize, 0usize);
        let mut strictly_tighter = 0usize;
        for _ in 0..400 {
            let (nb, is_int, lb0, ub0, rows, lo, up) = random_instance(&mut rng);
            let n = is_int.len();
            let points = feasible_points(nb, &is_int, &ub0, &rows, &lo, &up);
            let model = Arc::new(DomainModel::from_rows(
                n,
                &rows,
                lo.clone(),
                up.clone(),
                is_int.clone(),
            ));
            let nnz: usize = rows.iter().map(|r| r.0.len()).sum();
            let mut dom = Domain::new(Arc::clone(&model), &lb0, &ub0, 1e-6, 1e-9);
            dom.propagate();
            if dom.infeasible() {
                assert!(points.is_empty(), "propagation cut a feasible point");
                continue;
            }
            dom.commit_as_global();
            let mut table = CliqueTable::new(n);
            table.set_max_entries(nnz as i32);
            {
                let cr = CliqueRows {
                    ar_start: &model.ar_start,
                    ar_index: &model.ar_index,
                    ar_value: &model.ar_value,
                    row_lower: &lo,
                    row_upper: &up,
                    num_orig_rows: lo.len(),
                    feastol: 1e-6,
                    epsilon: 1e-9,
                };
                let mut v = GlobalDomainView {
                    dom: &mut dom,
                    is_int: &is_int,
                    feastol: 1e-6,
                    num_model_nonzeros: nnz,
                };
                table.extract_cliques(&mut v, &cr, None, true);
            }
            if dom.infeasible() {
                assert!(points.is_empty(), "clique extraction cut a feasible point");
                continue;
            }
            if !dom.domain_change_stack().is_empty() {
                dom.commit_as_global();
            }
            let cands: Vec<usize> = (0..nb).collect();
            let st = probe_and_attach(&mut dom, table, &cands);
            reductions += st.reductions;
            if dom.infeasible() {
                assert!(
                    points.is_empty(),
                    "probing proved a feasible program infeasible"
                );
                infeasible_proofs += 1;
                continue;
            }
            instances += 1;
            let table = dom.take_clique_table().expect("probing attached the table");
            let implics = dom
                .take_implications()
                .expect("probing attached implications");
            for x in &points {
                for j in 0..n {
                    assert!(
                        x[j] >= dom.col_lower[j] - 1e-9 && x[j] <= dom.col_upper[j] + 1e-9,
                        "probing cut feasible point {x:?} at column {j}"
                    );
                }
                for (_, members, _, _) in table.cliques() {
                    let s: usize = members
                        .iter()
                        .map(|v| (x[v.c()] == v.val as f64) as usize)
                        .sum();
                    assert!(s <= 1, "clique {members:?} cut feasible point {x:?}");
                    clique_checks += 1;
                }
                for col in 0..nb {
                    let val = x[col] as u8;
                    for (&ic, im) in implics.implications_of(col, val) {
                        assert!(
                            x[ic] >= im.lb - 1e-6 && x[ic] <= im.ub + 1e-6,
                            "implication of x{col}={val} on x{ic} {im:?} cut {x:?}"
                        );
                        impl_checks += 1;
                    }
                }
            }
            // The same global domain without the table or the implications: the
            // hook may only add to what row propagation derives.
            let mut plain = dom.clone();
            dom.set_clique_table(Some(table));
            dom.set_implications(Some(implics));
            for x in &points {
                for col in 0..nb {
                    if dom.infeasible() {
                        break;
                    }
                    if dom.col_lower[col] == dom.col_upper[col] {
                        continue;
                    }
                    let chg = if x[col] > 0.5 {
                        DomainChange {
                            boundval: 1.0,
                            column: col,
                            boundtype: BoundType::Lower,
                        }
                    } else {
                        DomainChange {
                            boundval: 0.0,
                            column: col,
                            boundtype: BoundType::Upper,
                        }
                    };
                    dom.change_bound(chg, Reason::Branching);
                    if !dom.infeasible() {
                        dom.propagate();
                    }
                    assert!(
                        !dom.infeasible(),
                        "the fixing hook cut feasible point {x:?}"
                    );
                    let moves = match chg.boundtype {
                        BoundType::Lower => chg.boundval > plain.col_lower[col],
                        BoundType::Upper => chg.boundval < plain.col_upper[col],
                    };
                    if moves {
                        plain.change_bound(chg, Reason::Branching);
                        plain.propagate();
                    }
                    assert!(
                        !plain.infeasible(),
                        "row propagation cut feasible point {x:?}"
                    );
                    let mut tighter = false;
                    for j in 0..n {
                        assert!(
                            dom.col_lower[j] >= plain.col_lower[j] - 1e-9
                                && dom.col_upper[j] <= plain.col_upper[j] + 1e-9,
                            "the hook loosened column {j}"
                        );
                        tighter |= dom.col_lower[j] > plain.col_lower[j] + 1e-9
                            || dom.col_upper[j] < plain.col_upper[j] - 1e-9;
                    }
                    strictly_tighter += tighter as usize;
                    for j in 0..n {
                        assert!(
                            x[j] >= dom.col_lower[j] - 1e-9 && x[j] <= dom.col_upper[j] + 1e-9,
                            "the fixing hook cut feasible point {x:?} at column {j}"
                        );
                    }
                    fix_checks += 1;
                }
                hook_fixings += dom
                    .domain_change_reasons()
                    .iter()
                    .filter(|r| matches!(r, Reason::CliqueTable(..)))
                    .count();
                dom.backtrack_to_global();
                plain.backtrack_to_global();
            }
        }
        eprintln!(
            "instances={instances} infeasible_proofs={infeasible_proofs} \
             clique_checks={clique_checks} impl_checks={impl_checks} fix_checks={fix_checks} \
             hook_fixings={hook_fixings} reductions={reductions} \
             strictly_tighter={strictly_tighter}"
        );
        // What the hook adds: after some fixing, the box is strictly tighter than row
        // propagation alone leaves it (a clique's contrapositive, an implication
        // stored from a probe). Without the hook this count is 0.
        assert!(
            strictly_tighter > 100,
            "the hook never tightened: {strictly_tighter}"
        );
        assert!(instances >= 100, "too few feasible instances: {instances}");
        assert!(clique_checks > 1000, "clique checks: {clique_checks}");
        assert!(impl_checks > 100, "implication checks: {impl_checks}");
        assert!(fix_checks > 1000, "fixing checks: {fix_checks}");
        assert!(
            hook_fixings > 100,
            "the hook fixed too little: {hook_fixings}"
        );
        assert!(reductions > 0, "probing never reduced the global box");
    }
}
