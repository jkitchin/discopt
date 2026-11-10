//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! `HighsDomain`'s incremental bound propagation over model rows (gap plan §12.30).
//!
//! Each row keeps its minimum and maximum activity over the current column bounds
//! (double-double, with the count of infinite contributions kept apart), updated in
//! O(column length) per bound change instead of being recomputed. A row is queued
//! for propagation only when a change could let it tighten a bound
//! (`markPropagate`'s capacity-threshold test), and propagation processes only the
//! queue. Every change goes on a stack with its previous value, so a dive
//! backtracks in O(changes).
//!
//! Faithful to HiGHS v1.x `HighsDomain.cpp`: `computeMinActivity`,
//! `computeMaxActivity`, `adjustedUb/Lb`, `propagateRowUpper/Lower`,
//! `updateActivityLb/UbChange`, `updateThresholdLb/UbChange`,
//! `recomputeCapacityThreshold`, `markPropagate`, `computeRowActivities`,
//! `doChangeBound`, `changeBound`, `backtrack`, `backtrackToGlobal`, `propagate`,
//! `getColLower/UpperPos`.
//!
//! Not yet ported (later layers of the search port): objective propagation,
//! cut-pool and conflict-pool propagation, conflict analysis, clique-table and
//! implication fixings on binaries, redundant-row recording.

use super::cdouble::CDouble;
use std::sync::Arc;

const INF: f64 = f64::INFINITY;
/// `kHighsTiny`.
const K_HIGHS_TINY: f64 = 1e-14;

/// `HighsBoundType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundType {
    /// Lower bound.
    Lower,
    /// Upper bound.
    Upper,
}

/// `HighsDomainChange`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainChange {
    /// New bound value.
    pub boundval: f64,
    /// Column.
    pub column: usize,
    /// Which bound.
    pub boundtype: BoundType,
}

/// `HighsDomain::Reason` (the kinds this port produces).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// `kBranching`.
    Branching,
    /// `kUnknown` (`Reason::unspecified()`).
    Unspecified,
    /// `kModelRowUpper`.
    ModelRowUpper(usize),
    /// `kModelRowLower`.
    ModelRowLower(usize),
    /// `kConflictingBounds`.
    ConflictingBounds(usize),
    /// Reduced-cost fixing (HiGHS passes `unspecified()`; kept distinct for counts).
    RedCost,
}

/// The rows a domain propagates: `row_lower <= a x <= row_upper` over the columns,
/// infinite sides as `f64::INFINITY`.
pub struct DomainModel {
    /// Columns.
    pub ncols: usize,
    /// Row starts (row-wise matrix), length `nrows + 1`.
    pub ar_start: Vec<usize>,
    /// Column per row entry.
    pub ar_index: Vec<usize>,
    /// Value per row entry.
    pub ar_value: Vec<f64>,
    /// Column starts (column-wise matrix), length `ncols + 1`.
    pub a_start: Vec<usize>,
    /// Row per column entry.
    pub a_index: Vec<usize>,
    /// Value per column entry.
    pub a_value: Vec<f64>,
    /// Row lower sides.
    pub row_lower: Vec<f64>,
    /// Row upper sides.
    pub row_upper: Vec<f64>,
    /// Integrality per column.
    pub is_int: Vec<bool>,
}

impl DomainModel {
    /// Build from rows given as `(columns, values)` pairs.
    pub fn from_rows(
        ncols: usize,
        rows: &[(Vec<usize>, Vec<f64>)],
        row_lower: Vec<f64>,
        row_upper: Vec<f64>,
        is_int: Vec<bool>,
    ) -> Self {
        let mut ar_start = vec![0usize];
        let (mut ar_index, mut ar_value) = (Vec::new(), Vec::new());
        let mut colcount = vec![0usize; ncols];
        for (ri, rv) in rows {
            for (&c, &v) in ri.iter().zip(rv) {
                if v != 0.0 {
                    ar_index.push(c);
                    ar_value.push(v);
                    colcount[c] += 1;
                }
            }
            ar_start.push(ar_index.len());
        }
        let mut a_start = vec![0usize; ncols + 1];
        for c in 0..ncols {
            a_start[c + 1] = a_start[c] + colcount[c];
        }
        let mut fill = a_start.clone();
        let mut a_index = vec![0usize; ar_index.len()];
        let mut a_value = vec![0.0; ar_index.len()];
        for r in 0..rows.len() {
            for k in ar_start[r]..ar_start[r + 1] {
                let c = ar_index[k];
                a_index[fill[c]] = r;
                a_value[fill[c]] = ar_value[k];
                fill[c] += 1;
            }
        }
        DomainModel {
            ncols,
            ar_start,
            ar_index,
            ar_value,
            a_start,
            a_index,
            a_value,
            row_lower,
            row_upper,
            is_int,
        }
    }

    /// Rows.
    pub fn nrows(&self) -> usize {
        self.row_lower.len()
    }
}

fn activity_contribution_min(coef: f64, lb: f64, ub: f64) -> f64 {
    if coef < 0.0 {
        if ub == INF {
            return -INF;
        }
        coef * ub
    } else {
        if lb == -INF {
            return -INF;
        }
        coef * lb
    }
}

fn activity_contribution_max(coef: f64, lb: f64, ub: f64) -> f64 {
    if coef < 0.0 {
        if lb == -INF {
            return INF;
        }
        coef * lb
    } else {
        if ub == INF {
            return INF;
        }
        coef * ub
    }
}

fn compute_delta(val: f64, oldbound: f64, newbound: f64, inf: f64, numinfs: &mut i32) -> CDouble {
    if oldbound == inf {
        *numinfs -= 1;
        CDouble::from(newbound) * val
    } else if newbound == inf {
        *numinfs += 1;
        CDouble::from(-oldbound) * val
    } else {
        (CDouble::from(newbound) - CDouble::from(oldbound)) * val
    }
}

fn bound_range(upper: f64, lower: f64, tolerance: f64, continuous: bool) -> f64 {
    let range = upper - lower;
    range
        - if continuous {
            (0.3 * range).max(1000.0 * tolerance)
        } else {
            tolerance
        }
}

/// `HighsDomain` (model-row propagation).
pub struct Domain {
    model: Arc<DomainModel>,
    feastol: f64,
    epsilon: f64,
    /// Current column lower bounds.
    pub col_lower: Vec<f64>,
    /// Current column upper bounds.
    pub col_upper: Vec<f64>,
    col_lower_pos: Vec<i64>,
    col_upper_pos: Vec<i64>,
    activitymin: Vec<CDouble>,
    activitymininf: Vec<i32>,
    activitymax: Vec<CDouble>,
    activitymaxinf: Vec<i32>,
    capacity_threshold: Vec<f64>,
    propagateflags: Vec<u8>,
    propagateinds: Vec<usize>,
    domchgstack: Vec<DomainChange>,
    prevboundval: Vec<(f64, i64)>,
    domchgreason: Vec<Reason>,
    branch_pos: Vec<usize>,
    changedcolsflags: Vec<bool>,
    changedcols: Vec<usize>,
    infeasible: bool,
    infeasible_pos: usize,
    infeasible_reason: Reason,
}

impl Domain {
    /// `HighsDomain(mipsolver)` + `computeRowActivities()`, over the global bounds.
    pub fn new(
        model: Arc<DomainModel>,
        lb: &[f64],
        ub: &[f64],
        feastol: f64,
        epsilon: f64,
    ) -> Self {
        let n = model.ncols;
        let m = model.nrows();
        let mut d = Domain {
            model,
            feastol,
            epsilon,
            col_lower: lb[..n].to_vec(),
            col_upper: ub[..n].to_vec(),
            col_lower_pos: vec![-1; n],
            col_upper_pos: vec![-1; n],
            activitymin: vec![CDouble::from(0.0); m],
            activitymininf: vec![0; m],
            activitymax: vec![CDouble::from(0.0); m],
            activitymaxinf: vec![0; m],
            capacity_threshold: vec![0.0; m],
            propagateflags: vec![0; m],
            propagateinds: Vec::with_capacity(m),
            domchgstack: Vec::new(),
            prevboundval: Vec::new(),
            domchgreason: Vec::new(),
            branch_pos: Vec::new(),
            changedcolsflags: vec![false; n],
            changedcols: Vec::new(),
            infeasible: false,
            infeasible_pos: 0,
            infeasible_reason: Reason::Unspecified,
        };
        d.compute_row_activities();
        d
    }

    /// Whether the domain is infeasible.
    pub fn infeasible(&self) -> bool {
        self.infeasible
    }

    /// The reason recorded with the infeasibility.
    pub fn infeasible_reason(&self) -> Reason {
        self.infeasible_reason
    }

    /// Columns changed since the last [`Self::clear_changed_cols`].
    pub fn changed_cols(&self) -> &[usize] {
        &self.changedcols
    }

    /// `clearChangedCols()`.
    pub fn clear_changed_cols(&mut self) {
        for &c in &self.changedcols {
            self.changedcolsflags[c] = false;
        }
        self.changedcols.clear();
    }

    /// The domain change stack.
    pub fn domain_change_stack(&self) -> &[DomainChange] {
        &self.domchgstack
    }

    /// Branching positions on the stack.
    pub fn branching_positions(&self) -> &[usize] {
        &self.branch_pos
    }

    /// Row activity bounds `(min, ninfmin, max, ninfmax)`.
    pub fn row_activity(&self, row: usize) -> (f64, i32, f64, i32) {
        (
            self.activitymin[row].to_f64(),
            self.activitymininf[row],
            self.activitymax[row].to_f64(),
            self.activitymaxinf[row],
        )
    }

    fn is_continuous(&self, col: usize) -> bool {
        !self.model.is_int[col]
    }

    /// `getColLowerPos(col, stackpos, pos)`.
    pub fn col_lower_pos(&self, col: usize, stackpos: i64) -> (f64, i64) {
        let mut lb = self.col_lower[col];
        let mut pos = self.col_lower_pos[col];
        while pos > stackpos || (pos != -1 && self.prevboundval[pos as usize].0 == lb) {
            lb = self.prevboundval[pos as usize].0;
            pos = self.prevboundval[pos as usize].1;
        }
        (lb, pos)
    }

    /// `getColUpperPos(col, stackpos, pos)`.
    pub fn col_upper_pos(&self, col: usize, stackpos: i64) -> (f64, i64) {
        let mut ub = self.col_upper[col];
        let mut pos = self.col_upper_pos[col];
        while pos > stackpos || (pos != -1 && self.prevboundval[pos as usize].0 == ub) {
            ub = self.prevboundval[pos as usize].0;
            pos = self.prevboundval[pos as usize].1;
        }
        (ub, pos)
    }

    fn bounds_for_activity(&self, col: usize) -> (f64, f64) {
        if self.infeasible {
            let p = self.infeasible_pos as i64 - 1;
            (self.col_lower_pos(col, p).0, self.col_upper_pos(col, p).0)
        } else {
            (self.col_lower[col], self.col_upper[col])
        }
    }

    /// `computeMinActivity` over row `row`.
    pub fn compute_min_activity(&self, row: usize) -> (i32, CDouble) {
        let mut act = CDouble::from(0.0);
        let mut ninf = 0;
        for k in self.model.ar_start[row]..self.model.ar_start[row + 1] {
            let col = self.model.ar_index[k];
            let (lb, ub) = self.bounds_for_activity(col);
            let c = activity_contribution_min(self.model.ar_value[k], lb, ub);
            if c == -INF {
                ninf += 1;
            } else {
                act += c;
            }
        }
        act.renormalize();
        (ninf, act)
    }

    /// `computeMaxActivity` over row `row`.
    pub fn compute_max_activity(&self, row: usize) -> (i32, CDouble) {
        let mut act = CDouble::from(0.0);
        let mut ninf = 0;
        for k in self.model.ar_start[row]..self.model.ar_start[row + 1] {
            let col = self.model.ar_index[k];
            let (lb, ub) = self.bounds_for_activity(col);
            let c = activity_contribution_max(self.model.ar_value[k], lb, ub);
            if c == INF {
                ninf += 1;
            } else {
                act += c;
            }
        }
        act.renormalize();
        (ninf, act)
    }

    fn adjusted_ub(&self, col: usize, bound_val: CDouble) -> (f64, bool) {
        let ft = self.feastol;
        if self.model.is_int[col] {
            let bound = (bound_val + ft).floor().to_f64();
            let accept = bound < self.col_upper[col]
                && self.col_upper[col] - bound > 1000.0 * ft * bound.abs();
            (bound, accept)
        } else {
            let bv = bound_val.to_f64();
            let bound = if (bv - self.col_lower[col]).abs() <= self.epsilon {
                self.col_lower[col]
            } else {
                bv
            };
            let accept = if self.col_upper[col] == INF {
                true
            } else if bound + 1000.0 * ft < self.col_upper[col] {
                let mut rel = self.col_upper[col] - bound;
                if self.col_lower[col] != -INF {
                    rel /= self.col_upper[col] - self.col_lower[col];
                } else {
                    rel /= self.col_upper[col].abs().max(bound.abs());
                }
                rel >= 0.3
            } else {
                false
            };
            (bound, accept)
        }
    }

    fn adjusted_lb(&self, col: usize, bound_val: CDouble) -> (f64, bool) {
        let ft = self.feastol;
        if self.model.is_int[col] {
            let bound = (bound_val - ft).ceil().to_f64();
            let accept = bound > self.col_lower[col]
                && bound - self.col_lower[col] > 1000.0 * ft * bound.abs();
            (bound, accept)
        } else {
            let bv = bound_val.to_f64();
            let bound = if (self.col_upper[col] - bv).abs() <= self.epsilon {
                self.col_upper[col]
            } else {
                bv
            };
            let accept = if self.col_lower[col] == -INF {
                true
            } else if bound - 1000.0 * ft > self.col_lower[col] {
                let mut rel = bound - self.col_lower[col];
                if self.col_upper[col] != INF {
                    rel /= self.col_upper[col] - self.col_lower[col];
                } else {
                    rel /= self.col_lower[col].abs().max(bound.abs());
                }
                rel >= 0.3
            } else {
                false
            };
            (bound, accept)
        }
    }

    fn propagate_row_upper(&self, row: usize, out: &mut Vec<DomainChange>) {
        let ninfmin = self.activitymininf[row];
        if ninfmin > 1 {
            return;
        }
        let minactivity = self.activitymin[row];
        let rupper = self.model.row_upper[row];
        for k in self.model.ar_start[row]..self.model.ar_start[row + 1] {
            let col = self.model.ar_index[k];
            let val = self.model.ar_value[k];
            let contrib = activity_contribution_min(val, self.col_lower[col], self.col_upper[col]);
            let minresact = if ninfmin == 1 {
                if contrib != -INF {
                    continue;
                }
                minactivity
            } else {
                minactivity - contrib
            };
            let bound_val = (rupper - minresact) / val;
            if (bound_val.to_f64() * K_HIGHS_TINY).abs() > self.feastol {
                continue;
            }
            if val > 0.0 {
                let (b, ok) = self.adjusted_ub(col, bound_val);
                if ok {
                    out.push(DomainChange {
                        boundval: b,
                        column: col,
                        boundtype: BoundType::Upper,
                    });
                }
            } else {
                let (b, ok) = self.adjusted_lb(col, bound_val);
                if ok {
                    out.push(DomainChange {
                        boundval: b,
                        column: col,
                        boundtype: BoundType::Lower,
                    });
                }
            }
        }
    }

    fn propagate_row_lower(&self, row: usize, out: &mut Vec<DomainChange>) {
        let ninfmax = self.activitymaxinf[row];
        if ninfmax > 1 {
            return;
        }
        let maxactivity = self.activitymax[row];
        let rlower = self.model.row_lower[row];
        for k in self.model.ar_start[row]..self.model.ar_start[row + 1] {
            let col = self.model.ar_index[k];
            let val = self.model.ar_value[k];
            let contrib = activity_contribution_max(val, self.col_lower[col], self.col_upper[col]);
            let maxresact = if ninfmax == 1 {
                if contrib != INF {
                    continue;
                }
                maxactivity
            } else {
                maxactivity - contrib
            };
            let bound_val = (rlower - maxresact) / val;
            if (bound_val.to_f64() * K_HIGHS_TINY).abs() > self.feastol {
                continue;
            }
            if val < 0.0 {
                let (b, ok) = self.adjusted_ub(col, bound_val);
                if ok {
                    out.push(DomainChange {
                        boundval: b,
                        column: col,
                        boundtype: BoundType::Upper,
                    });
                }
            } else {
                let (b, ok) = self.adjusted_lb(col, bound_val);
                if ok {
                    out.push(DomainChange {
                        boundval: b,
                        column: col,
                        boundtype: BoundType::Lower,
                    });
                }
            }
        }
    }

    fn update_threshold_lb_change(
        &self,
        col: usize,
        newbound: f64,
        val: f64,
        threshold: f64,
    ) -> f64 {
        if newbound != self.col_upper[col] {
            let tn = val.abs()
                * bound_range(
                    self.col_upper[col],
                    newbound,
                    self.feastol,
                    self.is_continuous(col),
                );
            threshold.max(tn).max(self.feastol)
        } else {
            threshold
        }
    }

    fn update_threshold_ub_change(
        &self,
        col: usize,
        newbound: f64,
        val: f64,
        threshold: f64,
    ) -> f64 {
        if newbound != self.col_lower[col] {
            let tn = val.abs()
                * bound_range(
                    newbound,
                    self.col_lower[col],
                    self.feastol,
                    self.is_continuous(col),
                );
            threshold.max(tn).max(self.feastol)
        } else {
            threshold
        }
    }

    fn recompute_capacity_threshold(&mut self, row: usize) {
        let mut t = -self.feastol;
        for k in self.model.ar_start[row]..self.model.ar_start[row + 1] {
            let col = self.model.ar_index[k];
            if self.col_upper[col] == self.col_lower[col] {
                continue;
            }
            let th = self.model.ar_value[k].abs()
                * bound_range(
                    self.col_upper[col],
                    self.col_lower[col],
                    self.feastol,
                    self.is_continuous(col),
                );
            t = t.max(th).max(self.feastol);
        }
        self.capacity_threshold[row] = t;
    }

    fn mark_propagate(&mut self, row: usize) {
        if self.propagateflags[row] != 0 {
            return;
        }
        let ft = self.feastol;
        let (rl, ru) = (self.model.row_lower[row], self.model.row_upper[row]);
        let proplower = rl != -INF
            && (self.activitymininf[row] != 0 || self.activitymin[row] < rl - ft)
            && (self.activitymaxinf[row] == 1
                || (self.activitymax[row].to_f64() - rl) <= self.capacity_threshold[row]);
        let propupper = ru != INF
            && (self.activitymaxinf[row] != 0 || self.activitymax[row] > ru + ft)
            && (self.activitymininf[row] == 1
                || (ru - self.activitymin[row].to_f64()) <= self.capacity_threshold[row]);
        if proplower || propupper {
            self.propagateinds.push(row);
            self.propagateflags[row] = 1;
        }
    }

    fn compute_row_activities(&mut self) {
        for i in 0..self.model.nrows() {
            let (ni, a) = self.compute_min_activity(i);
            self.activitymininf[i] = ni;
            self.activitymin[i] = a;
            let (ni, a) = self.compute_max_activity(i);
            self.activitymaxinf[i] = ni;
            self.activitymax[i] = a;
            self.recompute_capacity_threshold(i);
            if (self.activitymininf[i] <= 1 && self.model.row_upper[i] != INF)
                || (self.activitymaxinf[i] <= 1 && self.model.row_lower[i] != -INF)
            {
                self.mark_propagate(i);
            }
        }
    }

    /// `updateActivityLbChange` / `updateActivityUbChange` (`lower` selects which).
    fn update_activity(&mut self, col: usize, oldbound: f64, newbound: f64, lower: bool) {
        let model = Arc::clone(&self.model);
        let (start, mut end) = (model.a_start[col], model.a_start[col + 1]);
        let ft = self.feastol;
        let inf = if lower { -INF } else { INF };
        let mut i = start;
        while i < end {
            let row = model.a_index[i];
            let val = model.a_value[i];
            // Lower bound with a positive coefficient (or upper with a negative one)
            // moves the minimum activity; otherwise the maximum.
            let moves_min = (val > 0.0) == lower;
            if moves_min {
                let mut ninf = self.activitymininf[row];
                let delta = compute_delta(val, oldbound, newbound, inf, &mut ninf);
                self.activitymininf[row] = ninf;
                self.activitymin[row] += delta;
                if delta <= 0.0 {
                    self.capacity_threshold[row] = if lower {
                        self.update_threshold_lb_change(
                            col,
                            newbound,
                            val,
                            self.capacity_threshold[row],
                        )
                    } else {
                        self.update_threshold_ub_change(
                            col,
                            newbound,
                            val,
                            self.capacity_threshold[row],
                        )
                    };
                    i += 1;
                    continue;
                }
                if model.row_upper[row] != INF
                    && self.activitymininf[row] == 0
                    && self.activitymin[row] - model.row_upper[row] > ft
                {
                    self.infeasible = true;
                    self.infeasible_pos = self.domchgstack.len();
                    self.infeasible_reason = Reason::ModelRowUpper(row);
                    end = i + 1;
                    break;
                }
                if self.activitymininf[row] <= 1
                    && self.propagateflags[row] == 0
                    && model.row_upper[row] != INF
                {
                    self.mark_propagate(row);
                }
            } else {
                let mut ninf = self.activitymaxinf[row];
                let delta = compute_delta(val, oldbound, newbound, inf, &mut ninf);
                self.activitymaxinf[row] = ninf;
                self.activitymax[row] += delta;
                if delta >= 0.0 {
                    self.capacity_threshold[row] = if lower {
                        self.update_threshold_lb_change(
                            col,
                            newbound,
                            val,
                            self.capacity_threshold[row],
                        )
                    } else {
                        self.update_threshold_ub_change(
                            col,
                            newbound,
                            val,
                            self.capacity_threshold[row],
                        )
                    };
                    i += 1;
                    continue;
                }
                if model.row_lower[row] != -INF
                    && self.activitymaxinf[row] == 0
                    && model.row_lower[row] - self.activitymax[row] > ft
                {
                    self.infeasible = true;
                    self.infeasible_pos = self.domchgstack.len();
                    self.infeasible_reason = Reason::ModelRowLower(row);
                    end = i + 1;
                    break;
                }
                if self.activitymaxinf[row] <= 1
                    && self.propagateflags[row] == 0
                    && model.row_lower[row] != -INF
                {
                    self.mark_propagate(row);
                }
            }
            i += 1;
        }
        if self.infeasible {
            // Reverse the activity changes made before the infeasibility was found.
            for i in start..end {
                let row = model.a_index[i];
                let val = model.a_value[i];
                let moves_min = (val > 0.0) == lower;
                if moves_min {
                    let mut ninf = self.activitymininf[row];
                    let d = compute_delta(val, newbound, oldbound, inf, &mut ninf);
                    self.activitymininf[row] = ninf;
                    self.activitymin[row] += d;
                } else {
                    let mut ninf = self.activitymaxinf[row];
                    let d = compute_delta(val, newbound, oldbound, inf, &mut ninf);
                    self.activitymaxinf[row] = ninf;
                    self.activitymax[row] += d;
                }
            }
        }
    }

    fn do_change_bound(&mut self, chg: DomainChange) -> f64 {
        let col = chg.column;
        let lower = chg.boundtype == BoundType::Lower;
        let oldbound = if lower {
            std::mem::replace(&mut self.col_lower[col], chg.boundval)
        } else {
            std::mem::replace(&mut self.col_upper[col], chg.boundval)
        };
        if oldbound != chg.boundval {
            if !self.infeasible {
                self.update_activity(col, oldbound, chg.boundval, lower);
            }
            if !self.changedcolsflags[col] {
                self.changedcolsflags[col] = true;
                self.changedcols.push(col);
            }
        }
        oldbound
    }

    /// `changeBound(boundchg, reason)`.
    pub fn change_bound(&mut self, mut chg: DomainChange, reason: Reason) {
        let col = chg.column;
        let ft = self.feastol;
        let prev_pos;
        if chg.boundtype == BoundType::Lower {
            if chg.boundval <= self.col_lower[col] {
                if reason != Reason::Branching {
                    return;
                }
                chg.boundval = self.col_lower[col];
            }
            if chg.boundval > self.col_upper[col] {
                if chg.boundval - self.col_upper[col] > ft {
                    if !self.infeasible {
                        self.infeasible_pos = self.domchgstack.len();
                        self.infeasible = true;
                        self.infeasible_reason = Reason::ConflictingBounds(self.domchgstack.len());
                    }
                } else {
                    chg.boundval = self.col_upper[col];
                    if chg.boundval == self.col_lower[col] {
                        return;
                    }
                }
            }
            prev_pos = self.col_lower_pos[col];
            self.col_lower_pos[col] = self.domchgstack.len() as i64;
        } else {
            if chg.boundval >= self.col_upper[col] {
                if reason != Reason::Branching {
                    return;
                }
                chg.boundval = self.col_upper[col];
            }
            if chg.boundval < self.col_lower[col] {
                if self.col_lower[col] - chg.boundval > ft {
                    if !self.infeasible {
                        self.infeasible_pos = self.domchgstack.len();
                        self.infeasible = true;
                        self.infeasible_reason = Reason::ConflictingBounds(self.domchgstack.len());
                    }
                } else {
                    chg.boundval = self.col_lower[col];
                    if chg.boundval == self.col_upper[col] {
                        return;
                    }
                }
            }
            prev_pos = self.col_upper_pos[col];
            self.col_upper_pos[col] = self.domchgstack.len() as i64;
        }
        if reason == Reason::Branching {
            self.branch_pos.push(self.domchgstack.len());
        }
        let oldbound = self.do_change_bound(chg);
        self.prevboundval.push((oldbound, prev_pos));
        self.domchgstack.push(chg);
        self.domchgreason.push(reason);
    }

    /// `backtrack()`: undo to (and including) the last branching change, which is
    /// returned; `None` when the stack empties.
    pub fn backtrack(&mut self) -> Option<DomainChange> {
        let mut k = self.domchgstack.len() as i64 - 1;
        let old_infeasible = self.infeasible;
        if self.infeasible && self.infeasible_pos == self.domchgstack.len() {
            self.infeasible = false;
            self.infeasible_reason = Reason::Unspecified;
        }
        while k >= 0 {
            let ku = k as usize;
            let (prevbound, prevpos) = self.prevboundval[ku];
            let chg = self.domchgstack[ku];
            if chg.boundtype == BoundType::Lower {
                self.col_lower_pos[chg.column] = prevpos;
            } else {
                self.col_upper_pos[chg.column] = prevpos;
            }
            self.do_change_bound(DomainChange {
                boundval: prevbound,
                ..chg
            });
            if self.infeasible && self.infeasible_pos == ku {
                self.infeasible = false;
                self.infeasible_reason = Reason::Unspecified;
            }
            if self.domchgreason[ku] == Reason::Branching {
                self.branch_pos.pop();
                break;
            }
            k -= 1;
        }
        if old_infeasible {
            self.infeasible_reason = Reason::Unspecified;
            self.infeasible = false;
        }
        if k < 0 {
            self.domchgstack.clear();
            self.prevboundval.clear();
            self.domchgreason.clear();
            self.branch_pos.clear();
            return None;
        }
        let ku = k as usize;
        let chg = self.domchgstack[ku];
        self.domchgstack.truncate(ku);
        self.domchgreason.truncate(ku);
        self.prevboundval.truncate(ku);
        Some(chg)
    }

    /// `backtrackToGlobal()`.
    pub fn backtrack_to_global(&mut self) {
        let mut k = self.domchgstack.len() as i64 - 1;
        let old_infeasible = self.infeasible;
        if self.infeasible && self.infeasible_pos == self.domchgstack.len() {
            self.infeasible = false;
            self.infeasible_reason = Reason::Unspecified;
        }
        while k >= 0 {
            let ku = k as usize;
            let (prevbound, prevpos) = self.prevboundval[ku];
            let chg = self.domchgstack[ku];
            if chg.boundtype == BoundType::Lower {
                self.col_lower_pos[chg.column] = prevpos;
            } else {
                self.col_upper_pos[chg.column] = prevpos;
            }
            if prevbound != chg.boundval {
                self.do_change_bound(DomainChange {
                    boundval: prevbound,
                    ..chg
                });
            }
            if self.infeasible && self.infeasible_pos == ku {
                self.infeasible = false;
                self.infeasible_reason = Reason::Unspecified;
            }
            k -= 1;
        }
        if old_infeasible {
            self.infeasible_reason = Reason::Unspecified;
            self.infeasible = false;
        }
        self.domchgstack.clear();
        self.prevboundval.clear();
        self.domchgreason.clear();
        self.branch_pos.clear();
    }

    /// `propagate()`: true when there was anything to propagate.
    pub fn propagate(&mut self) -> bool {
        if self.propagateinds.is_empty() {
            return false;
        }
        let model = Arc::clone(&self.model);
        let ft = self.feastol;
        let mut inds: Vec<usize> = Vec::new();
        let mut buf: Vec<(usize, Vec<DomainChange>, Vec<DomainChange>)> = Vec::new();
        while !self.propagateinds.is_empty() {
            std::mem::swap(&mut inds, &mut self.propagateinds);
            for &row in &inds {
                self.propagateflags[row] = 0;
            }
            if !self.infeasible {
                buf.clear();
                for &i in &inds {
                    let mut up = Vec::new();
                    let mut lo = Vec::new();
                    let mut recompute = false;
                    if model.row_upper[i] != INF
                        && (self.activitymaxinf[i] != 0
                            || self.activitymax[i] > model.row_upper[i] + ft)
                    {
                        self.activitymin[i].renormalize();
                        self.propagate_row_upper(i, &mut up);
                        recompute = true;
                    }
                    if model.row_lower[i] != -INF
                        && (self.activitymininf[i] != 0
                            || self.activitymin[i] < model.row_lower[i] - ft)
                    {
                        self.activitymax[i].renormalize();
                        self.propagate_row_lower(i, &mut lo);
                        recompute = true;
                    }
                    if recompute {
                        self.recompute_capacity_threshold(i);
                    }
                    buf.push((i, up, lo));
                }
                'apply: for (i, up, lo) in buf.drain(..) {
                    for c in up {
                        if self.infeasible {
                            break 'apply;
                        }
                        self.change_bound(c, Reason::ModelRowUpper(i));
                    }
                    if self.infeasible {
                        break;
                    }
                    for c in lo {
                        if self.infeasible {
                            break 'apply;
                        }
                        self.change_bound(c, Reason::ModelRowLower(i));
                    }
                    if self.infeasible {
                        break;
                    }
                }
            }
            inds.clear();
        }
        true
    }

    /// Debug check (HiGHS's `#ifndef NDEBUG` asserts): the incremental activities
    /// agree with recomputation. Returns the worst absolute disagreement and whether
    /// every infinity count matches.
    pub fn check_activities(&self) -> (f64, bool) {
        let mut worst = 0.0f64;
        let mut counts_ok = true;
        for i in 0..self.model.nrows() {
            let (ni, a) = self.compute_min_activity(i);
            counts_ok &= ni == self.activitymininf[i];
            if ni == 0 {
                worst = worst.max((a - self.activitymin[i]).to_f64().abs());
            }
            let (ni, a) = self.compute_max_activity(i);
            counts_ok &= ni == self.activitymaxinf[i];
            if ni == 0 {
                worst = worst.max((a - self.activitymax[i]).to_f64().abs());
            }
        }
        (worst, counts_ok)
    }
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

    /// A random small integer program, its feasible points by enumeration.
    fn random_ip(r: &mut Lcg) -> (DomainModel, Vec<f64>, Vec<f64>, Vec<Vec<i64>>) {
        let n = r.int(2, 4) as usize;
        let m = r.int(1, 4) as usize;
        let lb: Vec<f64> = (0..n).map(|_| r.int(-2, 0) as f64).collect();
        let ub: Vec<f64> = lb.iter().map(|&l| l + r.int(1, 3) as f64).collect();
        let mut rows = Vec::new();
        let (mut rl, mut ru) = (Vec::new(), Vec::new());
        for _ in 0..m {
            let mut ri = Vec::new();
            let mut rv = Vec::new();
            for j in 0..n {
                let v = r.int(-3, 3);
                if v != 0 {
                    ri.push(j);
                    rv.push(v as f64);
                }
            }
            let mid = r.int(-3, 3) as f64;
            match r.int(0, 2) {
                0 => {
                    rl.push(-INF);
                    ru.push(mid);
                }
                1 => {
                    rl.push(mid);
                    ru.push(INF);
                }
                _ => {
                    rl.push(mid);
                    ru.push(mid + r.int(0, 2) as f64);
                }
            }
            rows.push((ri, rv));
        }
        let model = DomainModel::from_rows(n, &rows, rl.clone(), ru.clone(), vec![true; n]);
        let mut pts = Vec::new();
        let mut x: Vec<i64> = lb.iter().map(|&v| v as i64).collect();
        loop {
            let ok = (0..m).all(|i| {
                let a: f64 = rows[i]
                    .0
                    .iter()
                    .zip(&rows[i].1)
                    .map(|(&j, &v)| v * x[j] as f64)
                    .sum();
                a >= rl[i] - 1e-9 && a <= ru[i] + 1e-9
            });
            if ok {
                pts.push(x.clone());
            }
            let mut j = 0;
            while j < n {
                x[j] += 1;
                if x[j] as f64 <= ub[j] {
                    break;
                }
                x[j] = lb[j] as i64;
                j += 1;
            }
            if j == n {
                break;
            }
        }
        (model, lb, ub, pts)
    }

    #[test]
    fn propagation_never_removes_a_feasible_point_and_activities_stay_exact() {
        let mut r = Lcg(7);
        let (mut checked, mut propagated, mut infeasible_proofs) = (0usize, 0usize, 0usize);
        for _ in 0..400 {
            let (model, lb, ub, pts) = random_ip(&mut r);
            let model = Arc::new(model);
            let mut d = Domain::new(Arc::clone(&model), &lb, &ub, 1e-6, 1e-9);
            // A random branching, then propagation.
            let col = r.int(0, model.ncols as i64 - 1) as usize;
            let up = r.int(0, 1) == 1;
            let val = r.int(lb[col] as i64, ub[col] as i64) as f64;
            let chg = if up {
                DomainChange {
                    boundval: val,
                    column: col,
                    boundtype: BoundType::Lower,
                }
            } else {
                DomainChange {
                    boundval: val,
                    column: col,
                    boundtype: BoundType::Upper,
                }
            };
            d.change_bound(chg, Reason::Branching);
            if d.propagate() {
                propagated += 1;
            }
            let (w, counts) = d.check_activities();
            if !d.infeasible() {
                assert!(
                    counts && w <= 1e-6,
                    "activities drifted: {w} counts {counts}"
                );
            }
            let in_branch: Vec<&Vec<i64>> = pts
                .iter()
                .filter(|p| {
                    if up {
                        p[col] as f64 >= val
                    } else {
                        p[col] as f64 <= val
                    }
                })
                .collect();
            if d.infeasible() {
                infeasible_proofs += 1;
                assert!(
                    in_branch.is_empty(),
                    "infeasible verdict with {} feasible points",
                    in_branch.len()
                );
            } else {
                for p in &in_branch {
                    for j in 0..model.ncols {
                        checked += 1;
                        assert!(
                            p[j] as f64 >= d.col_lower[j] - 1e-9
                                && p[j] as f64 <= d.col_upper[j] + 1e-9,
                            "feasible point cut off at column {j}"
                        );
                    }
                }
            }
            // Backtracking restores the global bounds and activities exactly.
            d.backtrack_to_global();
            assert_eq!(d.col_lower, lb);
            assert_eq!(d.col_upper, ub);
            let (w, counts) = d.check_activities();
            assert!(counts && w <= 1e-9);
        }
        assert!(checked > 1000 && propagated > 100 && infeasible_proofs > 10,
            "probe fired too little: checked {checked} propagated {propagated} infeasible {infeasible_proofs}");
    }

    #[test]
    fn backtrack_pops_to_the_last_branching() {
        let rows = vec![(vec![0, 1], vec![1.0, 1.0])];
        let model = DomainModel::from_rows(2, &rows, vec![-INF], vec![3.0], vec![true, true]);
        let mut d = Domain::new(Arc::new(model), &[0.0, 0.0], &[5.0, 5.0], 1e-6, 1e-9);
        assert!(d.propagate());
        assert_eq!(d.col_upper, vec![3.0, 3.0]);
        d.change_bound(
            DomainChange {
                boundval: 2.0,
                column: 0,
                boundtype: BoundType::Lower,
            },
            Reason::Branching,
        );
        d.propagate();
        assert_eq!(d.col_upper[1], 1.0);
        let b = d.backtrack().expect("branching change");
        assert_eq!(b.column, 0);
        assert_eq!(d.col_lower[0], 0.0);
        assert_eq!(d.col_upper[1], 3.0);
    }
}
