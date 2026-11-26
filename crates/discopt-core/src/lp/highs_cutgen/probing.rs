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
    /// `nextCleanupCall`: entries probing may add before `runCliqueMerging`.
    pub next_cleanup_call: i64,
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
            next_cleanup_call: 0,
        }
    }

    /// `probedBefore(col, val)`.
    pub fn probed_before(&self, col: usize, val: u8) -> bool {
        self.has_probed[2 * col + val as usize]
    }

    /// `hasProbed` reset, keeping the implications: `HighsImplications::rebuild` after
    /// presolve clears `hasProbed` and re-adds the implications, so the root's
    /// `separateImpliedBounds` probes the LP's fractional binaries again.
    pub fn clear_probed(&mut self) {
        self.has_probed.iter_mut().for_each(|p| *p = false);
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
                crate::profile::incr(crate::profile::Ctr::MilpImplApplyFixings);
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
                crate::profile::incr(crate::profile::Ctr::MilpImplApplyFixings);
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

impl Prober<'_> {
    fn num_fixed_cols(&self) -> usize {
        (0..self.dom.col_lower.len())
            .filter(|&j| self.dom.col_lower[j] == self.dom.col_upper[j])
            .count()
    }

    /// `HPresolve::runProbing` (`HPresolveProbing.cpp`), first call, between its
    /// `prepareProbing` and `finaliseProbing`: probe every binary in the order of
    /// `computeProbingScore` (`min(5000, up * down) / (1 + numProbes)` then
    /// `min(100, up + down)`, `getNumImplications` of each literal; ties by
    /// `HighsRandom(random_seed).integer()`, then column), under the same contingents
    /// (`probingContingent` from 1000, the neighbourhood-query `splayContingent`, the
    /// early abort on deleted columns, the caps on new cliques and implications).
    ///
    /// Not ported, all presolve-only model edits the search cannot make on a fixed
    /// column space: `dualFixProbing` (it needs the dual-fix-probing locks
    /// propagation), the substitutions `runProbing` would apply, lifting for probing,
    /// and `finaliseProbing`'s row deletions and clique extensions. A substitution
    /// still counts towards `numDel`, as in HiGHS. `nrows` / `nnz` are the model's.
    pub fn presolve_probing(
        &mut self,
        submip: bool,
        random_seed: u32,
        nrows: usize,
        nnz: i64,
    ) -> ProbeStats {
        let mut st = ProbeStats::default();
        let ncols = self.dom.col_lower.len();
        // `PresolveProbingScope`.
        let old_flag = self.table.presolve_flag();
        self.table.set_presolve_flag(true);
        // `prepareProbing`: `cleanupFixed` after propagation.
        self.dom.propagate();
        self.commit();
        if !self.dom.infeasible() {
            let mut v = view(self.dom, &self.is_int);
            self.table.cleanup_fixed(&mut v);
        }
        self.commit();
        let old_entries = self.table.num_entries() as i64;
        let mut binaries: Vec<(i64, i64, i32, usize)> = Vec::new();
        if !self.dom.infeasible() && !self.table.is_full() {
            let mut random = super::random::HighsRandom::new(random_seed);
            for i in 0..ncols {
                if self.is_binary(i) {
                    let up = self.table.num_implications_val(i, true) as i64;
                    let down = self.table.num_implications_val(i, false) as i64;
                    // `numProbes[i]` is 0 on the first call.
                    let s1 = (up * down).min(5000);
                    let s2 = (up + down).min(100);
                    binaries.push((-s1, -s2, random.integer(), i));
                }
            }
        }
        binaries.sort();
        let fixed_start = self.num_fixed_cols();
        let calc_num_del = |p: &Self| -> i64 {
            (p.num_fixed_cols() - fixed_start) as i64 + p.table.substitutions().len() as i64
        };
        let num_cliques_start = self.table.num_cliques() as i64;
        let num_implics_start = self.implics.num_implications;
        let mut num_del = calc_num_del(self);
        let num_del_start = 0i64;
        let mut splay_contingent =
            self.table.num_neighbourhood_queries + (if submip { 0 } else { 100_000 }).max(10 * nnz);
        let mut probing_contingent: i64 = 1000;
        let mut num_probed: i64 = 0;
        let mut num_fail: i64 = 0;
        let cap = 1_000_000i64.max(2 * nnz);
        let del_lim = (nrows + ncols) as i64 / 20;
        for &(_, _, _, i) in &binaries {
            if self.table.substitution(i).is_some() || !self.is_binary(i) {
                continue;
            }
            let tighten_limits = num_probed >= 2500;
            let early_abort = if tighten_limits {
                num_del > del_lim.min(1000)
            } else {
                num_del > del_lim.max(1000)
            };
            if early_abort
                || self.table.is_full()
                || self.table.num_cliques() as i64 - num_cliques_start > cap
                || self.implics.num_implications - num_implics_start > cap
                || self.table.num_neighbourhood_queries > splay_contingent
                || probing_contingent - num_probed < 0
            {
                break;
            }
            let mut num_bound_chgs = 0usize;
            let cliques_before = self.table.num_cliques() as i64;
            if !self.run_probing(i, &mut num_bound_chgs) {
                continue;
            }
            st.probed += 1;
            st.reductions += num_bound_chgs;
            probing_contingent += num_bound_chgs as i64;
            let num_new_cliques = (self.table.num_cliques() as i64 - cliques_before).max(0);
            let new_num_del = calc_num_del(self);
            if new_num_del > num_del {
                probing_contingent += num_del;
                if !submip {
                    splay_contingent += 100 * (new_num_del + num_del_start);
                    splay_contingent += 1000 * num_new_cliques;
                }
                num_del = new_num_del;
                num_fail = 0;
            } else if submip || num_new_cliques == 0 {
                splay_contingent -= (if tighten_limits { 250 } else { 100 }) * num_fail;
                num_fail += 1;
            } else {
                splay_contingent += 1000 * num_new_cliques;
                num_fail = 0;
            }
            num_probed += 1;
            if self.dom.infeasible() {
                break;
            }
        }
        // `finaliseProbing`: `cleanupFixed`, then `runCliqueMerging`.
        if !self.dom.infeasible() {
            let mut v = view(self.dom, &self.is_int);
            self.table.cleanup_fixed(&mut v);
        }
        self.commit();
        if !self.dom.infeasible() {
            let mut v = view(self.dom, &self.is_int);
            self.table.run_clique_merging(&mut v);
        }
        self.commit();
        self.table.set_presolve_flag(old_flag);
        st.new_entries = self.table.num_entries() as i64 - old_entries;
        st
    }
}

/// HiGHS's presolve probing schedule (`HPresolve::runProbing`), on in
/// [`probe_and_attach`] when given.
#[derive(Clone, Copy, Debug)]
pub struct PresolveProbing {
    /// `mipsolver->submip`.
    pub submip: bool,
    /// `options->random_seed`.
    pub random_seed: u32,
    /// Model rows.
    pub nrows: usize,
}

/// Probe on `dom` (at its global baseline), extending `table` and recording
/// implications, then attach both to `dom` so its `change_bound` uses them. With
/// `presolve`, first HiGHS's presolve probing of every binary
/// ([`Prober::presolve_probing`]), after which `hasProbed` is reset as
/// `HighsImplications::rebuild` does; then the root's `separateImpliedBounds` probing
/// of `cands` (the root LP's fractional binaries). Returns what probing did (both
/// passes); the domain may come back infeasible (the problem has no point, or none
/// better than the cutoff).
pub fn probe_and_attach(
    dom: &mut Domain,
    table: CliqueTable,
    cands: &[usize],
    presolve: Option<PresolveProbing>,
) -> ProbeStats {
    probe_and_attach_with(dom, table, cands, presolve, None)
}

/// What the root separation's probing (`DISCOPT_HIGHS_SEPA_ROOT_PROBE`) leaves for the
/// search: HiGHS's `mipdata_->implications` and the global domain's bounds after
/// presolve probing and the root rounds' `separateImpliedBounds`. HiGHS runs presolve
/// probing once and the tree reads the root's implications; a search handed this runs
/// neither presolve probing nor `HighsImplications::rebuild` again.
#[derive(Clone, Debug)]
pub struct RootProbed {
    /// The implications, with `hasProbed` for the columns the root probed.
    pub implics: Implications,
    /// The root probing domain's global lower bounds.
    pub col_lower: Vec<f64>,
    /// The root probing domain's global upper bounds.
    pub col_upper: Vec<f64>,
}

/// [`probe_and_attach`], or, with `prior`, the search start after the root separation
/// probed: its global reductions are applied to `dom` (then propagated and made the
/// baseline), its implications (with `hasProbed`) are kept, `table` keeps its
/// `numCliqueEntriesAfterFirstPresolve`, `presolve` must be `None`, and only `cands`
/// not probed yet are probed.
pub fn probe_and_attach_with(
    dom: &mut Domain,
    mut table: CliqueTable,
    cands: &[usize],
    presolve: Option<PresolveProbing>,
    prior: Option<RootProbed>,
) -> ProbeStats {
    let n = dom.col_lower.len();
    let nnz = dom.num_model_nonzeros() as i64;
    let had_prior = prior.is_some();
    let mut implics = match prior {
        None => Implications::new(n),
        Some(rp) => {
            assert!(
                presolve.is_none(),
                "presolve probing ran at the root; HiGHS runs it once"
            );
            assert_eq!(
                rp.col_lower.len(),
                n,
                "root probing's domain has the search's columns"
            );
            for j in 0..n {
                if dom.infeasible() {
                    break;
                }
                if rp.col_lower[j] > dom.col_lower[j] {
                    dom.change_bound(
                        DomainChange {
                            boundval: rp.col_lower[j],
                            column: j,
                            boundtype: BoundType::Lower,
                        },
                        Reason::Unspecified,
                    );
                }
                if !dom.infeasible() && rp.col_upper[j] < dom.col_upper[j] {
                    dom.change_bound(
                        DomainChange {
                            boundval: rp.col_upper[j],
                            column: j,
                            boundtype: BoundType::Upper,
                        },
                        Reason::Unspecified,
                    );
                }
            }
            if !dom.infeasible() {
                dom.propagate();
            }
            if !dom.infeasible() {
                dom.commit_as_global();
            }
            rp.implics
        }
    };
    crate::profile::record_max(
        crate::profile::Ctr::MilpCliqueTableCliques0,
        table.num_cliques() as u64,
    );
    crate::profile::record_max(
        crate::profile::Ctr::MilpCliqueTableEntries0,
        table.num_entries().max(0) as u64,
    );
    let mut st = ProbeStats::default();
    if let Some(pp) = presolve {
        let mut p = Prober::new(dom, &mut table, &mut implics);
        st = p.presolve_probing(pp.submip, pp.random_seed, pp.nrows, nnz);
        crate::profile::incr_by(crate::profile::Ctr::MilpProbePresolve, st.probed as u64);
        implics.clear_probed();
    }
    // `numCliqueEntriesAfterFirstPresolve`, and `nextCleanupCall = numNonzero()`
    // (`HighsImplications::rebuild`).
    let entries0 = if had_prior {
        table.num_entries_after_presolve() as i64
    } else {
        table.num_entries() as i64
    };
    if !had_prior {
        table.set_num_entries_after_presolve(entries0 as i32);
        implics.next_cleanup_call = nnz;
    }
    if !dom.infeasible() {
        let mut p = Prober::new(dom, &mut table, &mut implics);
        let mut next = p.implics.next_cleanup_call;
        let s2 = p.probe_candidates(cands, &mut next, entries0);
        p.implics.next_cleanup_call = next;
        st.probed += s2.probed;
        st.reductions += s2.reductions;
        st.new_entries += s2.new_entries;
    }
    crate::profile::record_max(
        crate::profile::Ctr::MilpCliqueTableCliques1,
        table.num_cliques() as u64,
    );
    crate::profile::record_max(
        crate::profile::Ctr::MilpCliqueTableEntries1,
        table.num_entries().max(0) as u64,
    );
    if !dom.infeasible() {
        dom.set_clique_table(Some(Arc::new(table)));
        dom.set_implications(Some(Arc::new(implics)));
    }
    st
}

/// `separateImpliedBounds` in a search node's separation round: detach the shared
/// clique table and implications from `gdom` (the global domain) and `local` (the
/// node's), probe the node LP's fractional binaries not probed yet on `gdom`
/// ([`Prober::probe_candidates`], `runCliqueMerging` by `nextCleanupCall`), attach the
/// grown table and implications to both again, and return the implied-bound cuts
/// `vals . x <= rhs` the point `x` violates by more than `feastol`, each with whether
/// it is integral (`!isColContinuous(implCol)`). Probing commits its global
/// reductions to `gdom`'s baseline. Neither domain is touched without an attached
/// table and implications.
pub fn separate_implied_bounds(
    gdom: &mut Domain,
    local: &mut Domain,
    x: &[f64],
    frac: &[usize],
) -> (ProbeStats, Vec<([usize; 2], [f64; 2], f64, bool)>) {
    let mut st = ProbeStats::default();
    let mut cuts = Vec::new();
    let (Some(t), Some(im)) = (gdom.take_clique_table(), gdom.take_implications()) else {
        return (st, cuts);
    };
    let lt = local.take_clique_table();
    let li = local.take_implications();
    drop((lt, li));
    let mut table = Arc::try_unwrap(t).unwrap_or_else(|a| (*a).clone());
    let mut implics = Arc::try_unwrap(im).unwrap_or_else(|a| (*a).clone());
    if !gdom.infeasible() {
        if !gdom.domain_change_stack().is_empty() {
            gdom.commit_as_global();
        }
        st = probe_lp_candidates(gdom, &mut table, &mut implics, frac);
    }
    if !gdom.infeasible() {
        cuts = implied_bound_cuts(gdom, &implics, x, frac);
    }
    let table = Arc::new(table);
    let implics = Arc::new(implics);
    local.set_clique_table(Some(table.clone()));
    local.set_implications(Some(implics.clone()));
    gdom.set_clique_table(Some(table));
    gdom.set_implications(Some(implics));
    (st, cuts)
}

/// The probing loop of `separateImpliedBounds` on `gdom` (at its global baseline, no
/// table or implications attached): probe the candidates `frac` not probed yet,
/// `runCliqueMerging` by `nextCleanupCall`, and leave the table's neighbourhood-query
/// count as it was (HiGHS restores `numNeighbourhoodQueries`). Commits the global
/// reductions probing finds to `gdom`'s baseline.
pub fn probe_lp_candidates(
    gdom: &mut Domain,
    table: &mut CliqueTable,
    implics: &mut Implications,
    frac: &[usize],
) -> ProbeStats {
    let entries_after = table.num_entries_after_presolve() as i64;
    let old_queries = table.num_neighbourhood_queries;
    let mut p = Prober::new(gdom, table, implics);
    let mut next = p.implics.next_cleanup_call;
    let st = p.probe_candidates(frac, &mut next, entries_after);
    p.implics.next_cleanup_call = next;
    table.num_neighbourhood_queries = old_queries;
    st
}

/// The cut loop of `separateImpliedBounds`: for each binary in `frac` still free in
/// `gdom`, the implied-bound cuts `vals . x <= rhs` from its cached implications that
/// `x` violates by more than `gdom`'s feasibility tolerance, each with whether it is
/// integral (`!isColContinuous(implCol)`).
pub fn implied_bound_cuts(
    gdom: &Domain,
    implics: &Implications,
    x: &[f64],
    frac: &[usize],
) -> Vec<([usize; 2], [f64; 2], f64, bool)> {
    let mut cuts = Vec::new();
    let feastol = gdom.feastol();
    for &col in frac {
        if gdom.col_lower[col] != 0.0 || gdom.col_upper[col] != 1.0 {
            continue;
        }
        for val in 0..2u8 {
            for (&ic, imp) in implics.implications_of(col, val) {
                let (glb, gub) = (gdom.col_lower[ic], gdom.col_upper[ic]);
                let integral = gdom.is_int(ic);
                let mut cand: Vec<([usize; 2], [f64; 2], f64)> = Vec::new();
                // HiGHS's list is sorted by `HighsDomainChange::operator<`, (column,
                // boundtype) with kLower < kUpper, so a column's lower-bound cut comes
                // first; the order fixes the pool's row indices and so its tie-breaks.
                if val == 1 {
                    if imp.lb - feastol > glb {
                        cand.push(([ic, col], [-1.0, imp.lb - glb], -glb));
                    }
                    if imp.ub + feastol < gub {
                        cand.push(([ic, col], [1.0, gub - imp.ub], gub));
                    }
                } else {
                    if imp.lb - feastol > glb {
                        cand.push(([ic, col], [-1.0, glb - imp.lb], -imp.lb));
                    }
                    if imp.ub + feastol < gub {
                        cand.push(([ic, col], [1.0, imp.ub - gub], imp.ub));
                    }
                }
                for (inds, vals, rhs) in cand {
                    let viol = x[inds[0]] * vals[0] + x[inds[1]] * vals[1] - rhs;
                    if viol > feastol && rhs.is_finite() && vals.iter().all(|v| v.is_finite()) {
                        cuts.push((inds, vals, rhs, integral));
                    }
                }
            }
        }
    }
    cuts
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
            // Every other instance runs presolve probing (`HPresolve::runProbing`)
            // first, which resets `hasProbed` before the candidate pass.
            let pp = (instances % 2 == 1).then_some(PresolveProbing {
                submip: false,
                random_seed: 0,
                nrows: lo.len(),
            });
            let st = probe_and_attach(&mut dom, table, &cands, pp);
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

    /// `separate_implied_bounds` (node separation's `separateImpliedBounds`): on random
    /// small programs, probing on a global domain and the implied-bound cuts it returns
    /// never cut a feasible point of the program (enumerated), every cut is violated
    /// by the point it was separated for, and both domains get the table back.
    #[test]
    fn implied_bound_cuts_never_cut_a_feasible_point() {
        let mut rng = HighsRandom::new(777);
        let (mut cut_checks, mut cuts_found, mut instances) = (0usize, 0usize, 0usize);
        for _ in 0..1600 {
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
                continue;
            }
            dom.commit_as_global();
            let mut table = CliqueTable::new(n);
            table.set_max_entries(nnz as i32);
            // Attach with no probing, so the node pass does all of it.
            probe_and_attach(&mut dom, table, &[], None);
            if dom.infeasible() {
                continue;
            }
            let mut local = dom.clone();
            let x: Vec<f64> = (0..n)
                .map(|j| {
                    if j < nb {
                        rng.fraction()
                    } else {
                        lb0[j] + (ub0[j] - lb0[j]) * rng.fraction()
                    }
                })
                .collect();
            let fr: Vec<usize> = (0..nb).collect();
            let (_, cuts) = separate_implied_bounds(&mut dom, &mut local, &x, &fr);
            assert!(dom.has_clique_table() && local.has_clique_table());
            if dom.infeasible() {
                assert!(
                    points.is_empty(),
                    "node probing proved a feasible program infeasible"
                );
                continue;
            }
            instances += 1;
            for (inds, vals, rhs, _) in &cuts {
                cuts_found += 1;
                let at_x = x[inds[0]] * vals[0] + x[inds[1]] * vals[1];
                assert!(
                    at_x > rhs + 1e-6,
                    "cut {inds:?} {vals:?} <= {rhs} not violated at x"
                );
                for p in &points {
                    let a = p[inds[0]] * vals[0] + p[inds[1]] * vals[1];
                    assert!(
                        a <= rhs + 1e-6,
                        "implied-bound cut {inds:?} {vals:?} <= {rhs} cut {p:?}"
                    );
                    cut_checks += 1;
                }
            }
            for p in &points {
                for j in 0..n {
                    assert!(p[j] >= dom.col_lower[j] - 1e-9 && p[j] <= dom.col_upper[j] + 1e-9);
                }
            }
        }
        eprintln!("instances={instances} cuts={cuts_found} cut_checks={cut_checks}");
        assert!(instances > 1000 && cuts_found > 100 && cut_checks > 5000);
    }
}
