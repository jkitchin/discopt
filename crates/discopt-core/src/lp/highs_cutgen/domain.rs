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
//! `getColLower/UpperPos`, `set/getReducedDomainChangeStack`; and
//! `CutpoolPropagation` (`cutAdded`, `cutDeleted`, `markPropagateCut`,
//! `updateActivityLb/UbChange`, `recomputeCapacityThreshold`), with the domain's
//! `markPropagateCut(reason)`, `getMinCutActivity` and the cut-pool section of
//! `propagate`; and `ObjectivePropagation` (constructor, `updateActivityLb/UbChange`,
//! `recomputeCapacityThreshold`, `shouldBePropagated`, `propagate`,
//! `getPropagationConstraint`) with `checkChangeBound`; and `ConflictPoolPropagation`
//! (`conflictAdded`, `conflictDeleted`, `markPropagateConflict`,
//! `updateActivityLb/UbChange`, `propagateConflict`, the watched-literal lists) with
//! the conflict-pool sections of `updateActivityLb/UbChange`, `propagate`,
//! `markPropagateCut(reason)` and `havePropagationRows`, `isActive` and `flip`; and
//! conflict analysis: `ConflictSet` (`explainInfeasibility*`, `explainBoundChange*`,
//! `resolveLinearLeq/Geq`, the resolve queue, `resolveDepth`, `computeCuts`, both
//! `conflictAnalysis`), `HighsDomain::conflictAnalysis` (both overloads) and
//! `conflictAnalyzeReconvergence`. The conflict pool is [`super::conflict_pool`].
//!
//! Not yet ported (later layers of the search port): clique-table and implication
//! fixings on binaries (this port never produces a `kCliqueTable` reason; the
//! explanation of one needs no clique table and is translated), redundant-row
//! recording, `tightenCoefficients`, `fixCol`, the debug-solution checks.
//!
//! Conflict analysis reads two things HiGHS takes from other components: pseudocost
//! conflict scores ([`ConflictScores`], `HighsPseudocost`) and the open-node counts per
//! column ([`OpenNodeCounts`], `HighsNodeQueue::numNodesUp/Down`), both as traits so
//! the translations of those components plug in.

use super::cdouble::CDouble;
use super::conflict_pool::{ConflictDomainObservers, ConflictPool};
use super::objective::ObjectiveFunction;
use super::stdalgo::pdqsort;
use std::collections::{BTreeMap, BinaryHeap};
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
    /// `Reason::cut(cutpoolindex, cut)`: row `cut` of the domain's cut pool `pool`.
    Cut(usize, usize),
    /// `kObjective`.
    Objective,
    /// `Reason::cut(cutpoolpropagation.size() + conflictpoolindex, conflict)`:
    /// conflict `conflict` of the domain's conflict pool `pool`.
    Conflict(usize, usize),
    /// `Reason::cliqueTable(col, val)`: implied by column `col` fixed to `val`
    /// (never produced until the clique table is ported).
    CliqueTable(usize, usize),
}

/// `HighsPseudocost`'s conflict scores, as conflict analysis updates them
/// (`increaseConflictWeight`, `increaseConflictScoreUp/Down`). A trait so the
/// pseudocost translation plugs in without this module depending on it.
pub trait ConflictScores {
    /// `increaseConflictWeight()`.
    fn increase_conflict_weight(&mut self);
    /// `increaseConflictScoreUp(col)`.
    fn increase_conflict_score_up(&mut self, col: usize);
    /// `increaseConflictScoreDown(col)`.
    fn increase_conflict_score_down(&mut self, col: usize);
}

/// `HighsNodeQueue::numNodesUp/Down(col)`: open nodes with a lower (up) or upper
/// (down) bound change on `col`; conflict analysis weighs its resolution candidates
/// by them (`computePrio`).
pub trait OpenNodeCounts {
    /// `numNodesUp(col)`.
    fn num_nodes_up(&self, col: usize) -> i64;
    /// `numNodesDown(col)`.
    fn num_nodes_down(&self, col: usize) -> i64;
}

/// No open nodes (an empty node queue).
pub struct NoOpenNodes;
impl OpenNodeCounts for NoOpenNodes {
    fn num_nodes_up(&self, _col: usize) -> i64 {
        0
    }
    fn num_nodes_down(&self, _col: usize) -> i64 {
        0
    }
}

/// Which domains in a conflict analysis propagate the conflict pool, with the
/// pool's index among each one's conflict pools (`addConflictPool`). HiGHS's pool
/// notifies its `propagationDomains`; with the global domain registered first and the
/// search's local domain (a copy) after it, a new conflict reaches the global domain
/// first. Other domains propagating the pool are notified by the caller with the
/// indices conflict analysis returns.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConflictLinks {
    /// The pool's index in the local domain (the one analysed), if it propagates it.
    pub local: Option<usize>,
    /// The pool's index in the global domain, if it propagates it.
    pub global: Option<usize>,
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

/// `HighsDomain::CutpoolPropagation`: one cut pool's rows as a domain tracks them.
///
/// HiGHS reads the rows through a pointer to the pool's `HighsDynamicRowMatrix`;
/// here the pool hands each row to the domain when it is added
/// ([`Domain::cut_added`]) and the domain keeps a copy, with per-column lists that
/// mirror the pool matrix's column links. HiGHS inserts a linked row at the head of
/// each column's list, so its traversals visit the newest row first; the lists here
/// are pushed in insertion order and walked in reverse, which is the same order.
#[derive(Clone, Default)]
struct CutpoolProp {
    /// Mirror of the pool matrix: row columns and values per cut slot.
    rows: Vec<(Vec<usize>, Vec<f64>)>,
    /// `cutpool->getRhs()`.
    rhs: Vec<f64>,
    /// `columnsLinked(cut)`.
    linked: Vec<bool>,
    /// Column links of the positive (`AheadPos_`) and negative (`AheadNeg_`) entries.
    col_pos: Vec<Vec<(usize, f64)>>,
    col_neg: Vec<Vec<(usize, f64)>>,
    /// `activitycuts_`, `activitycutsinf_`, `propagatecutflags_`,
    /// `propagatecutinds_`, `capacityThreshold_`.
    activity: Vec<CDouble>,
    activity_inf: Vec<i32>,
    flags: Vec<u8>,
    inds: Vec<usize>,
    threshold: Vec<f64>,
}

impl CutpoolProp {
    fn new(ncols: usize) -> Self {
        CutpoolProp {
            col_pos: vec![Vec::new(); ncols],
            col_neg: vec![Vec::new(); ncols],
            ..Default::default()
        }
    }

    /// `HighsDynamicRowMatrix::unlinkColumns` on the mirror.
    fn unlink(&mut self, cut: usize) {
        if !self.linked[cut] {
            return;
        }
        self.linked[cut] = false;
        let (inds, vals) = &self.rows[cut];
        for (&c, &v) in inds.iter().zip(vals) {
            let list = if v > 0.0 {
                &mut self.col_pos[c]
            } else {
                &mut self.col_neg[c]
            };
            let p = list
                .iter()
                .position(|&(r, _)| r == cut)
                .expect("linked cut missing from its column list");
            list.remove(p);
        }
    }

    /// `markPropagateCut(cut)`.
    fn mark_propagate_cut(&mut self, cut: usize) {
        if self.flags[cut] == 0
            && (self.activity_inf[cut] == 1
                || self.rhs[cut] - self.activity[cut].to_f64() <= self.threshold[cut])
        {
            self.inds.push(cut);
            self.flags[cut] |= 1;
        }
    }

    fn ensure_tracked(&mut self, cut: usize) {
        if self.activity.len() <= cut {
            self.activity.resize(cut + 1, CDouble::from(0.0));
            self.activity_inf.resize(cut + 1, 0);
            self.flags.resize(cut + 1, 2);
            self.threshold.resize(cut + 1, 0.0);
        }
    }
}

/// `ConflictPoolPropagation::WatchedLiteral`: a watched conflict literal (column -1
/// when unused) and its links in the column's watch list.
#[derive(Clone, Copy, Debug)]
struct WatchedLiteral {
    boundval: f64,
    column: i64,
    boundtype: BoundType,
    prev: i64,
    next: i64,
}

impl Default for WatchedLiteral {
    /// `{0.0, -1, kLower}`, unlinked.
    fn default() -> Self {
        WatchedLiteral {
            boundval: 0.0,
            column: -1,
            boundtype: BoundType::Lower,
            prev: -1,
            next: -1,
        }
    }
}

impl WatchedLiteral {
    /// `watched.domchg != entry` (`HighsDomainChange::operator!=`).
    fn differs(&self, d: &DomainChange) -> bool {
        self.boundtype != d.boundtype
            || self.column != d.column as i64
            || self.boundval != d.boundval
    }
}

/// `HighsDomain::ConflictPoolPropagation`: one conflict pool's conflicts as a domain
/// tracks them, by two watched literals per conflict.
///
/// HiGHS reads the conflicts through a pointer to the pool; here the pool hands each
/// conflict to the domain when it is added ([`Domain::conflict_added`]) and the domain
/// keeps a copy (`None` once deleted), as [`CutpoolProp`] keeps the cut rows.
///
/// The watch lists reproduce a quirk of `linkWatchedLiteral` (HiGHS 1.15.1 and
/// master): the list head is only moved when the list is non-empty
/// (`if (head != -1) { ...; head = linkPos; }`), so a literal is never put into an
/// empty list and every list stays empty. `updateActivityLb/UbChange` therefore never
/// visits a conflict, and a conflict is propagated only when it is marked on
/// `conflictAdded` (fewer than two literals inactive at that point) or through
/// `markPropagateCut(reason)` when a change it explains is backtracked. Kept as is for
/// replay; missing propagation is sound.
#[derive(Clone, Default)]
struct ConflictPoolProp {
    /// Mirror of the pool's conflicts.
    conflicts: Vec<Option<Vec<DomainChange>>>,
    /// `colLowerWatched_`, `colUpperWatched_`: list heads per column.
    col_lower_watched: Vec<i64>,
    col_upper_watched: Vec<i64>,
    /// `conflictFlag_`: number of inactive watched literals (bits 0-1), queued for
    /// propagation (4), deleted (8).
    conflict_flag: Vec<u8>,
    /// `propagateConflictInds_`.
    propagate_conflict_inds: Vec<usize>,
    /// `watchedLiterals_`: two per conflict.
    watched: Vec<WatchedLiteral>,
}

impl ConflictPoolProp {
    fn new(ncols: usize) -> Self {
        ConflictPoolProp {
            col_lower_watched: vec![-1; ncols],
            col_upper_watched: vec![-1; ncols],
            ..Default::default()
        }
    }

    /// The watch-list head of literal `link_pos`'s column and bound type.
    fn head(&mut self, link_pos: usize) -> &mut i64 {
        let w = self.watched[link_pos];
        let col = w.column as usize;
        if w.boundtype == BoundType::Lower {
            &mut self.col_lower_watched[col]
        } else {
            &mut self.col_upper_watched[col]
        }
    }

    /// `linkWatchedLiteral(linkPos)`, with HiGHS's head update (see the type's doc).
    fn link_watched_literal(&mut self, link_pos: usize) {
        debug_assert!(self.watched[link_pos].column != -1);
        let head = *self.head(link_pos);
        self.watched[link_pos].prev = -1;
        self.watched[link_pos].next = head;
        if head != -1 {
            self.watched[head as usize].prev = link_pos as i64;
            *self.head(link_pos) = link_pos as i64;
        }
    }

    /// `unlinkWatchedLiteral(linkPos)`.
    fn unlink_watched_literal(&mut self, link_pos: usize) {
        if self.watched[link_pos].column == -1 {
            return;
        }
        let (prev, next) = (self.watched[link_pos].prev, self.watched[link_pos].next);
        if prev != -1 {
            self.watched[prev as usize].next = next;
        } else {
            *self.head(link_pos) = next;
        }
        self.watched[link_pos].column = -1;
        if next != -1 {
            self.watched[next as usize].prev = prev;
        }
    }

    /// `markPropagateConflict(conflict)`.
    fn mark_propagate_conflict(&mut self, conflict: usize) {
        if self.conflict_flag[conflict] < 2 {
            self.propagate_conflict_inds.push(conflict);
            self.conflict_flag[conflict] |= 4;
        }
    }

    fn set_watched(&mut self, watch_pos: usize, d: &DomainChange) {
        let w = &mut self.watched[watch_pos];
        w.boundval = d.boundval;
        w.column = d.column as i64;
        w.boundtype = d.boundtype;
    }
}

/// A contribution-tree key, ordered like HiGHS's `std::pair<double, HighsInt>`
/// (`-contribution`, `col`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct ContribKey(f64, usize);
impl Eq for ContribKey {}
impl PartialOrd for ContribKey {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for ContribKey {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&o.0)
            .expect("NaN objective contribution")
            .then(self.1.cmp(&o.1))
    }
}

/// `HighsDomain::ObjectivePropagation`: propagation of `cost . x <= upper_limit`
/// with the objective's lower bound kept incrementally. Binaries in a clique
/// partition contribute through `ObjectiveContributionTree`s (HiGHS's cached
/// red-black trees keyed by `(-contribution, col)`; ordered maps here, with
/// `first()` the smallest key and `last()` the largest).
#[derive(Clone)]
struct ObjProp {
    obj: Arc<ObjectiveFunction>,
    objective_lower: CDouble,
    num_inf_obj_lower: i32,
    capacity_threshold: f64,
    is_propagated: bool,
    /// `objectiveLowerContributions`: (contribution, col, partition) per position.
    contributions: Vec<(f64, usize, usize)>,
    /// One contribution tree per partition: key -> position.
    trees: Vec<BTreeMap<ContribKey, usize>>,
    propagation_cons_buffer: Vec<f64>,
    /// `partitionCliqueData`: (multiplier, rhs).
    partition_clique_data: Vec<(f64, i64)>,
}

impl ObjProp {
    fn key(&self, pos: usize) -> ContribKey {
        ContribKey(-self.contributions[pos].0, self.contributions[pos].1)
    }
    fn link(&mut self, pos: usize) {
        let k = self.key(pos);
        let part = self.contributions[pos].2;
        self.trees[part].insert(k, pos);
    }
    fn unlink(&mut self, pos: usize) {
        let k = self.key(pos);
        let part = self.contributions[pos].2;
        self.trees[part].remove(&k);
    }
    fn first(&self, part: usize) -> Option<usize> {
        self.trees[part].values().next().copied()
    }
    fn last(&self, part: usize) -> Option<usize> {
        self.trees[part].values().next_back().copied()
    }
    fn successor(&self, pos: usize) -> Option<usize> {
        let part = self.contributions[pos].2;
        let k = self.key(pos);
        self.trees[part]
            .range((std::ops::Bound::Excluded(k), std::ops::Bound::Unbounded))
            .next()
            .map(|(_, &p)| p)
    }
}

/// `HighsDomain`: model-row and cut-pool propagation.
#[derive(Clone)]
pub struct Domain {
    model: Arc<DomainModel>,
    feastol: f64,
    epsilon: f64,
    /// Minimum relative improvement to accept a continuous column's new bound
    /// (`adjustedUb/Lb`: HiGHS uses 0.3).
    cont_accept: f64,
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
    /// `cutpoolpropagation`.
    cutpools: Vec<CutpoolProp>,
    /// Whether this is the global domain (`domain == &mipdata_->getDomain()`), which
    /// also tracks cuts added without propagation and ignores unlinking.
    is_global: bool,
    /// `cutpool->resetAge(cut)` calls made during propagation, for the caller to
    /// apply to its pools (the domain does not own them): `(pool, cut)`.
    cut_age_resets: Vec<(usize, usize)>,
    /// `objProp_` (inactive until [`Domain::setup_objective_propagation`]).
    obj_prop: Option<Box<ObjProp>>,
    /// `mipdata_->upper_limit`: the objective cutoff objective propagation uses.
    upper_limit: f64,
    /// `conflictPoolPropagation`.
    conflictpools: Vec<ConflictPoolProp>,
    /// `conflictpool_->resetAge(conflict)` calls made during propagation, as
    /// `(pool, conflict)`, for the caller to apply.
    conflict_age_resets: Vec<(usize, usize)>,
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
            cont_accept: 0.3,
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
            cutpools: Vec::new(),
            is_global: false,
            cut_age_resets: Vec::new(),
            obj_prop: None,
            upper_limit: INF,
            conflictpools: Vec::new(),
            conflict_age_resets: Vec::new(),
        };
        d.compute_row_activities();
        d
    }

    /// Set the minimum relative improvement for continuous bound changes (HiGHS: 0.3).
    pub fn set_continuous_accept(&mut self, v: f64) {
        self.cont_accept = v;
    }

    /// Mark this as the global domain (HiGHS compares against
    /// `mipdata_->getDomain()`): it tracks the activity of cuts added without
    /// propagation and keeps tracking a cut its pool unlinks for propagation.
    pub fn set_global(&mut self, global: bool) {
        self.is_global = global;
    }

    /// `addCutpool(cutpool)`: register a cut pool; returns its index in reasons.
    pub fn add_cutpool(&mut self) -> usize {
        self.cutpools.push(CutpoolProp::new(self.model.ncols));
        self.cutpools.len() - 1
    }

    /// `CutpoolPropagation::cutAdded(cut, propagate)`, with the row handed over
    /// (HiGHS reads it from the pool). Row `inds`/`vals` are the cut `vals . x <= rhs`.
    pub fn cut_added(
        &mut self,
        pool: usize,
        cut: usize,
        inds: &[usize],
        vals: &[f64],
        rhs: f64,
        propagate: bool,
    ) {
        {
            let cp = &mut self.cutpools[pool];
            if cp.rows.len() <= cut {
                cp.rows.resize(cut + 1, (Vec::new(), Vec::new()));
                cp.rhs.resize(cut + 1, INF);
                cp.linked.resize(cut + 1, false);
            }
            debug_assert!(!cp.linked[cut], "cut slot reused while still linked");
            cp.rows[cut] = (inds.to_vec(), vals.to_vec());
            cp.rhs[cut] = rhs;
            cp.linked[cut] = propagate;
            if propagate {
                for (&c, &v) in inds.iter().zip(vals) {
                    if v > 0.0 {
                        cp.col_pos[c].push((cut, v));
                    } else {
                        cp.col_neg[c].push((cut, v));
                    }
                }
            }
        }
        if !propagate && !self.is_global {
            return;
        }
        self.cutpools[pool].ensure_tracked(cut);
        self.cutpools[pool].flags[cut] &= !2u8;
        let (ninf, act) = self.compute_min_activity_of(inds, vals);
        let cp = &mut self.cutpools[pool];
        cp.activity_inf[cut] = ninf;
        cp.activity[cut] = act;
        if propagate {
            self.recompute_cut_capacity_threshold(pool, cut);
            self.cutpools[pool].mark_propagate_cut(cut);
        }
    }

    /// `CutpoolPropagation::cutDeleted(cut, deletedOnlyForPropagation)`. The mirror
    /// of the pool matrix unlinks the row in every domain, as the pool's matrix does;
    /// only a local domain marks an unlinked cut deleted.
    pub fn cut_deleted(&mut self, pool: usize, cut: usize, deleted_only_for_propagation: bool) {
        let is_global = self.is_global;
        let cp = &mut self.cutpools[pool];
        if cut < cp.linked.len() {
            cp.unlink(cut);
            if !deleted_only_for_propagation {
                cp.rows[cut] = (Vec::new(), Vec::new());
                cp.rhs[cut] = INF;
            }
        }
        if deleted_only_for_propagation && is_global {
            return;
        }
        if cut < cp.flags.len() {
            cp.flags[cut] |= 2;
        }
    }

    /// `getMinCutActivity(cutpool, cut)`.
    pub fn min_cut_activity(&self, pool: usize, cut: usize) -> f64 {
        let cp = &self.cutpools[pool];
        if cut < cp.flags.len() && cp.flags[cut] & 2 == 0 && cp.activity_inf[cut] == 0 {
            cp.activity[cut].to_f64()
        } else {
            -INF
        }
    }

    /// The `cutpool->resetAge(cut)` calls propagation made since the last call, as
    /// `(pool, cut)`; the caller applies them to its pools in order.
    pub fn take_cut_age_resets(&mut self) -> Vec<(usize, usize)> {
        std::mem::take(&mut self.cut_age_resets)
    }

    /// `addConflictPool(conflictPool)`: register a conflict pool; returns its index in
    /// [`Reason::Conflict`]. Conflicts the pool holds already are not handed over (as
    /// in HiGHS, a new `ConflictPoolPropagation` starts empty).
    pub fn add_conflict_pool(&mut self) -> usize {
        self.conflictpools
            .push(ConflictPoolProp::new(self.model.ncols));
        self.conflictpools.len() - 1
    }

    /// `ConflictPoolPropagation::conflictAdded(conflict)`, with the conflict's literals
    /// handed over (HiGHS reads them from the pool).
    pub fn conflict_added(&mut self, pool: usize, conflict: usize, entries: &[DomainChange]) {
        let active: Vec<bool> = entries.iter().map(|e| self.is_active(e)).collect();
        let positions: Vec<i64> = entries
            .iter()
            .map(|e| match e.boundtype {
                BoundType::Lower => self.col_lower_pos[e.column],
                BoundType::Upper => self.col_upper_pos[e.column],
            })
            .collect();
        let cp = &mut self.conflictpools[pool];
        if cp.conflicts.len() <= conflict {
            cp.conflicts.resize(conflict + 1, None);
        }
        cp.conflicts[conflict] = Some(entries.to_vec());
        if cp.conflict_flag.len() <= conflict {
            cp.watched
                .resize(2 * conflict + 2, WatchedLiteral::default());
            cp.conflict_flag.resize(conflict + 1, 0);
        }
        let mut num_watched = 0usize;
        for (i, e) in entries.iter().enumerate() {
            if active[i] {
                continue;
            }
            let watch_pos = 2 * conflict + num_watched;
            cp.set_watched(watch_pos, e);
            cp.link_watched_literal(watch_pos);
            num_watched += 1;
            if num_watched == 2 {
                break;
            }
        }
        match num_watched {
            0 => {
                // Every literal holds: watch the two set most recently.
                let mut latest_active = [(0i64, 0usize); 2];
                let mut num_active = 0usize;
                for (i, &pos) in positions.iter().enumerate() {
                    match num_active {
                        0 => {
                            latest_active[0] = (pos, i);
                            num_active = 1;
                        }
                        1 => {
                            latest_active[1] = (pos, i);
                            num_active = 2;
                            if latest_active[0].0 < latest_active[1].0 {
                                latest_active.swap(0, 1);
                            }
                        }
                        _ => {
                            if pos > latest_active[1].0 {
                                latest_active[1] = (pos, i);
                                if latest_active[0].0 < latest_active[1].0 {
                                    latest_active.swap(0, 1);
                                }
                            }
                        }
                    }
                }
                for (i, la) in latest_active.iter().enumerate().take(num_active) {
                    let watch_pos = 2 * conflict + i;
                    cp.set_watched(watch_pos, &entries[la.1]);
                    cp.link_watched_literal(watch_pos);
                }
            }
            1 => {
                let mut latest = -1i64;
                let mut latest_pos = -1i64;
                for (i, &pos) in positions.iter().enumerate() {
                    if pos > latest_pos {
                        latest = i as i64;
                        latest_pos = pos;
                    }
                }
                if latest != -1 {
                    let watch_pos = 2 * conflict + 1;
                    cp.set_watched(watch_pos, &entries[latest as usize]);
                    cp.link_watched_literal(watch_pos);
                }
            }
            _ => {}
        }
        cp.conflict_flag[conflict] = num_watched as u8 | (cp.conflict_flag[conflict] & 4);
        cp.mark_propagate_conflict(conflict);
    }

    /// `ConflictPoolPropagation::conflictDeleted(conflict)`; the mirror drops the
    /// literals, as the pool's range becomes `(-1, -1)`.
    pub fn conflict_deleted(&mut self, pool: usize, conflict: usize) {
        let cp = &mut self.conflictpools[pool];
        if conflict >= cp.conflict_flag.len() {
            // Never handed to this domain (HiGHS indexes out of range here).
            return;
        }
        cp.conflict_flag[conflict] |= 8;
        cp.unlink_watched_literal(2 * conflict);
        cp.unlink_watched_literal(2 * conflict + 1);
        cp.conflicts[conflict] = None;
    }

    /// `ConflictPoolPropagation::updateActivityLbChange` (`lower`) or
    /// `updateActivityUbChange`: count the watched literals the change makes
    /// inactive (or active again) and queue the conflicts that reach fewer than two.
    fn update_conflict_activity(
        &mut self,
        pool: usize,
        col: usize,
        oldbound: f64,
        newbound: f64,
        lower: bool,
    ) {
        debug_assert!(!self.infeasible);
        let cp = &mut self.conflictpools[pool];
        let mut i = if lower {
            cp.col_lower_watched[col]
        } else {
            cp.col_upper_watched[col]
        };
        while i != -1 {
            let iu = i as usize;
            let conflict = iu >> 1;
            let bv = cp.watched[iu].boundval;
            let delta = if lower {
                ((bv > newbound) as u8).wrapping_sub((bv > oldbound) as u8)
            } else {
                ((bv < newbound) as u8).wrapping_sub((bv < oldbound) as u8)
            };
            if delta != 0 {
                cp.conflict_flag[conflict] = cp.conflict_flag[conflict].wrapping_add(delta);
                cp.mark_propagate_conflict(conflict);
            }
            i = cp.watched[iu].next;
        }
    }

    /// `ConflictPoolPropagation::propagateConflict(conflict)`.
    fn propagate_conflict(&mut self, pool: usize, conflict: usize) {
        // The propagate flag goes; the watched and deleted information stays.
        self.conflictpools[pool].conflict_flag[conflict] &= 3 | 8;
        // Two inactive literals watched, or the conflict deleted.
        if self.conflictpools[pool].conflict_flag[conflict] >= 2 {
            return;
        }
        if self.infeasible {
            return;
        }
        let entries = match &self.conflictpools[pool].conflicts[conflict] {
            Some(e) => e.clone(),
            None => {
                let cp = &mut self.conflictpools[pool];
                cp.unlink_watched_literal(2 * conflict);
                cp.unlink_watched_literal(2 * conflict + 1);
                return;
            }
        };
        let mut inactive = [0usize; 2];
        let mut num_inactive = 0usize;
        for (i, e) in entries.iter().enumerate() {
            if self.is_active(e) {
                continue;
            }
            inactive[num_inactive] = i;
            num_inactive += 1;
            if num_inactive == 2 {
                break;
            }
        }
        self.conflictpools[pool].conflict_flag[conflict] = num_inactive as u8;
        match num_inactive {
            0 => {
                self.infeasible = true;
                self.infeasible_reason = Reason::Conflict(pool, conflict);
                self.infeasible_pos = self.domchgstack.len();
                self.conflict_age_resets.push((pool, conflict));
            }
            1 => {
                let domchg = self.flip(&entries[inactive[0]]);
                if !self.is_active(&domchg) {
                    self.change_bound(domchg, Reason::Conflict(pool, conflict));
                    self.conflict_age_resets.push((pool, conflict));
                }
            }
            _ => {
                let cp = &mut self.conflictpools[pool];
                if cp.watched[2 * conflict].differs(&entries[inactive[0]]) {
                    cp.unlink_watched_literal(2 * conflict);
                    cp.set_watched(2 * conflict, &entries[inactive[0]]);
                    cp.link_watched_literal(2 * conflict);
                }
                if cp.watched[2 * conflict + 1].differs(&entries[inactive[1]]) {
                    cp.unlink_watched_literal(2 * conflict + 1);
                    cp.set_watched(2 * conflict + 1, &entries[inactive[1]]);
                    cp.link_watched_literal(2 * conflict + 1);
                }
            }
        }
    }

    /// The `conflictpool_->resetAge(conflict)` calls propagation made since the last
    /// call, as `(pool, conflict)`; the caller applies them to its pools in order.
    pub fn take_conflict_age_resets(&mut self) -> Vec<(usize, usize)> {
        std::mem::take(&mut self.conflict_age_resets)
    }

    /// `isActive(domchg)`: the literal holds in the current bounds.
    pub fn is_active(&self, d: &DomainChange) -> bool {
        match d.boundtype {
            BoundType::Lower => d.boundval <= self.col_lower[d.column],
            BoundType::Upper => d.boundval >= self.col_upper[d.column],
        }
    }

    /// `flip(domchg)`: the literal's negation (by `feastol`, rounded for integers).
    pub fn flip(&self, d: &DomainChange) -> DomainChange {
        if d.boundtype == BoundType::Lower {
            let mut v = d.boundval - self.feastol;
            if self.model.is_int[d.column] {
                v = v.floor();
            }
            DomainChange {
                boundval: v,
                column: d.column,
                boundtype: BoundType::Upper,
            }
        } else {
            let mut v = d.boundval + self.feastol;
            if self.model.is_int[d.column] {
                v = v.ceil();
            }
            DomainChange {
                boundval: v,
                column: d.column,
                boundtype: BoundType::Lower,
            }
        }
    }

    /// The literals `addConflictCut` stores for a frontier: its changes in stack
    /// order, a continuous column's bound moved by `feastol` away from the conflict
    /// (so the stored literal is strictly weaker).
    pub fn conflict_cut_entries<'a>(
        &self,
        frontier: impl IntoIterator<Item = &'a DomainChange>,
    ) -> Vec<DomainChange> {
        frontier
            .into_iter()
            .map(|d| {
                let mut e = *d;
                if !self.model.is_int[e.column] {
                    if e.boundtype == BoundType::Lower {
                        e.boundval += self.feastol;
                    } else {
                        e.boundval -= self.feastol;
                    }
                }
                e
            })
            .collect()
    }

    /// The literals `addReconvergenceCut` stores: the flipped reconvergence change,
    /// then the frontier as in [`Self::conflict_cut_entries`].
    pub fn reconvergence_cut_entries<'a>(
        &self,
        frontier: impl IntoIterator<Item = &'a DomainChange>,
        reconvergence_domchg: &DomainChange,
    ) -> Vec<DomainChange> {
        let mut out = vec![self.flip(reconvergence_domchg)];
        out.extend(self.conflict_cut_entries(frontier));
        out
    }

    /// `getMinActivity(row)`.
    pub fn min_activity(&self, row: usize) -> f64 {
        if self.activitymininf[row] == 0 {
            self.activitymin[row].to_f64()
        } else {
            -INF
        }
    }

    /// `getMaxActivity(row)`.
    pub fn max_activity(&self, row: usize) -> f64 {
        if self.activitymaxinf[row] == 0 {
            self.activitymax[row].to_f64()
        } else {
            INF
        }
    }

    /// `HighsDomain::conflictAnalysis(conflictPool, globaldom, pseudocost)`: explain
    /// this (local, infeasible) domain's infeasibility over the global domain and
    /// store the conflicts found in `pool`. Returns the indices of the conflicts
    /// added, in order. The two domains must share their cut-pool indices (HiGHS
    /// finds a pool in the global domain by its address).
    pub fn conflict_analysis(
        &mut self,
        pool: &mut ConflictPool,
        links: ConflictLinks,
        globaldom: &mut Domain,
        scores: &mut dyn ConflictScores,
        nodes: &dyn OpenNodeCounts,
    ) -> Vec<usize> {
        if globaldom.infeasible() || !self.infeasible {
            return Vec::new();
        }
        globaldom.propagate();
        if globaldom.infeasible() {
            return Vec::new();
        }
        let mut cs = ConflictSet::new(self, globaldom, links, nodes);
        cs.conflict_analysis(pool, scores);
        cs.added
    }

    /// `HighsDomain::conflictAnalysis(proofinds, proofvals, prooflen, proofrhs,
    /// conflictPool, globaldom, pseudocost)`: a globally valid proof row
    /// `vals . x[inds] <= rhs` that the local bounds violate (e.g. a dual ray of an
    /// infeasible LP) is explained instead of the domain's infeasibility.
    pub fn conflict_analysis_proof(
        &mut self,
        proofinds: &[usize],
        proofvals: &[f64],
        proofrhs: f64,
        pool: &mut ConflictPool,
        links: ConflictLinks,
        globaldom: &mut Domain,
        scores: &mut dyn ConflictScores,
        nodes: &dyn OpenNodeCounts,
    ) -> Vec<usize> {
        if globaldom.infeasible() {
            return Vec::new();
        }
        globaldom.propagate();
        if globaldom.infeasible() {
            return Vec::new();
        }
        let mut cs = ConflictSet::new(self, globaldom, links, nodes);
        cs.conflict_analysis_proof(proofinds, proofvals, proofrhs, pool, scores);
        cs.added
    }

    /// `conflictAnalyzeReconvergence(domchg, proofinds, proofvals, prooflen,
    /// proofrhs, conflictPool, globaldom, pseudocost)`: explain the bound change
    /// `domchg` (not on the stack) by the proof row and store the reconvergence cut.
    pub fn conflict_analyze_reconvergence(
        &mut self,
        domchg: &DomainChange,
        proofinds: &[usize],
        proofvals: &[f64],
        proofrhs: f64,
        pool: &mut ConflictPool,
        links: ConflictLinks,
        globaldom: &mut Domain,
        scores: &mut dyn ConflictScores,
        nodes: &dyn OpenNodeCounts,
    ) -> Vec<usize> {
        if globaldom.infeasible() {
            return Vec::new();
        }
        globaldom.propagate();
        if globaldom.infeasible() {
            return Vec::new();
        }
        let stack_len = self.domchgstack.len() as i64;
        let mut cs = ConflictSet::new(self, globaldom, links, nodes);
        let (ninfmin, activitymin) = cs.globaldom.compute_min_activity_of(proofinds, proofvals);
        if ninfmin != 0 {
            return Vec::new();
        }
        let frontier = Frontier::new();
        if !cs.explain_bound_change_leq(
            &frontier,
            (stack_len, *domchg),
            proofinds,
            proofvals,
            proofrhs,
            activitymin.to_f64(),
        ) {
            return Vec::new();
        }
        if 10 * cs.resolved_domain_changes.len() > 1000 + 3 * cs.localdom.num_integral_cols() {
            return Vec::new();
        }
        let mut frontier = frontier;
        for &(p, d) in &cs.resolved_domain_changes {
            frontier.entry(p).or_insert(d);
        }
        let mut depth = cs.localdom.branch_pos.len();
        while depth > 0 {
            let bp = cs.localdom.branch_pos[depth - 1];
            if cs.localdom.domchgstack[bp].boundval != cs.localdom.prevboundval[bp].0 {
                break;
            }
            depth -= 1;
        }
        cs.resolve_depth(&mut frontier, depth, 0, scores, 0, false);
        let entries = cs
            .localdom
            .reconvergence_cut_entries(frontier.values(), domchg);
        cs.add_to_pool(pool, &entries);
        cs.added
    }

    /// `mipsolver->mipdata_->integral_cols.size()`.
    fn num_integral_cols(&self) -> usize {
        self.model.is_int.iter().filter(|&&b| b).count()
    }

    /// Set `mipdata_->upper_limit`, the cutoff objective propagation works against.
    pub fn set_upper_limit(&mut self, upper_limit: f64) {
        self.upper_limit = upper_limit;
    }

    /// `setupObjectivePropagation()`: `ObjectivePropagation(domain)` over the
    /// current bounds.
    pub fn setup_objective_propagation(&mut self, obj: Arc<ObjectiveFunction>) {
        let nz = obj.nonzeros().to_vec();
        let starts = obj.partition_starts().to_vec();
        let np = obj.num_partitions();
        let cost = &obj.cost;
        let mut op = ObjProp {
            obj: Arc::clone(&obj),
            objective_lower: CDouble::from(0.0),
            num_inf_obj_lower: 0,
            capacity_threshold: INF,
            is_propagated: false,
            contributions: vec![(0.0, 0, 0); starts[np]],
            trees: vec![BTreeMap::new(); np],
            propagation_cons_buffer: if np != 0 {
                obj.values_packed().to_vec()
            } else {
                Vec::new()
            },
            partition_clique_data: if np != 0 {
                vec![(0.0, 0); np]
            } else {
                Vec::new()
            },
        };
        // Each clique partition's columns first contribute with their larger
        // value, then the largest contribution of the partition is removed.
        for i in 0..np {
            op.partition_clique_data[i].1 = 1;
            for (j, &col) in nz.iter().enumerate().take(starts[i + 1]).skip(starts[i]) {
                if cost[col] > 0.0 {
                    op.objective_lower += cost[col];
                    op.contributions[j] = (cost[col], col, i);
                    op.partition_clique_data[i].1 -= 1;
                    if self.col_lower[col] == 0.0 {
                        op.link(j);
                    }
                } else {
                    op.contributions[j] = (-cost[col], col, i);
                    if self.col_upper[col] == 1.0 {
                        op.link(j);
                    }
                }
            }
            if let Some(w) = op.first(i) {
                op.objective_lower -= op.contributions[w].0;
            }
        }
        for &col in &nz[starts[np]..] {
            if cost[col] > 0.0 {
                if self.col_lower[col] == -INF {
                    op.num_inf_obj_lower += 1;
                } else {
                    op.objective_lower += self.col_lower[col] * cost[col];
                }
            } else if self.col_upper[col] == INF {
                op.num_inf_obj_lower += 1;
            } else {
                op.objective_lower += self.col_upper[col] * cost[col];
            }
        }
        self.obj_prop = Some(Box::new(op));
        self.obj_recompute_capacity_threshold();
    }

    /// The objective's lower bound over the current bounds, `(value, ninf)`
    /// (`objectiveLower`, `numInfObjLower`); `None` without objective propagation.
    pub fn objective_lower(&self) -> Option<(f64, i32)> {
        self.obj_prop
            .as_ref()
            .map(|op| (op.objective_lower.to_f64(), op.num_inf_obj_lower))
    }

    /// `ObjectivePropagation::getPropagationConstraint(domchgStackSize, ...,
    /// domchgCol)`: the objective cutoff row `vals . x[inds] <= rhs` as it stood
    /// when the stack had `stack_size` changes.
    pub fn objective_propagation_constraint(
        &mut self,
        stack_size: i64,
        domchg_col: Option<usize>,
    ) -> (Vec<usize>, Vec<f64>, f64) {
        let op = self
            .obj_prop
            .as_ref()
            .expect("objective propagation inactive");
        let obj = Arc::clone(&op.obj);
        let np = obj.num_partitions();
        let inds = obj.nonzeros().to_vec();
        if np == 0 {
            return (inds, obj.values_packed().to_vec(), self.upper_limit);
        }
        let starts = obj.partition_starts();
        let cost = &obj.cost;
        let mut tmp_rhs = CDouble::from(self.upper_limit);
        let mut largests = Vec::with_capacity(np);
        for i in 0..np {
            let mut largest = 0.0f64;
            for &c in &inds[starts[i]..starts[i + 1]] {
                if Some(c) == domchg_col {
                    continue;
                }
                if cost[c] > 0.0 {
                    if self.col_lower_pos(c, stack_size).0 < 1.0 {
                        largest = largest.max(cost[c]);
                    }
                } else if self.col_upper_pos(c, stack_size).0 > 0.0 {
                    largest = largest.max(-cost[c]);
                }
            }
            largests.push(largest);
        }
        let op = self.obj_prop.as_mut().unwrap();
        let packed = obj.values_packed();
        for (i, &largest) in largests.iter().enumerate() {
            tmp_rhs += largest * op.partition_clique_data[i].1 as f64;
            if op.partition_clique_data[i].0 != largest {
                op.partition_clique_data[i].0 = largest;
                for j in starts[i]..starts[i + 1] {
                    op.propagation_cons_buffer[j] = packed[j] - largest.copysign(packed[j]);
                }
            }
        }
        (inds, op.propagation_cons_buffer.clone(), tmp_rhs.to_f64())
    }

    /// `ObjectivePropagation::recomputeCapacityThreshold()`.
    fn obj_recompute_capacity_threshold(&mut self) {
        let mut op = self
            .obj_prop
            .take()
            .expect("objective propagation inactive");
        let obj = Arc::clone(&op.obj);
        let np = obj.num_partitions();
        let ft = self.feastol;
        op.capacity_threshold = -ft;
        for i in 0..np {
            let Some(worst) = op.first(i) else { continue };
            let wcol = op.contributions[worst].1;
            if self.col_lower[wcol] == self.col_upper[wcol] {
                continue;
            }
            let mut contribution = op.contributions[worst].0;
            let best = op.last(i).unwrap();
            if best != worst {
                contribution -= op.contributions[best].0;
            }
            op.capacity_threshold = op.capacity_threshold.max(contribution * (1.0 - ft));
        }
        for &col in &obj.nonzeros()[obj.partition_starts()[np]..] {
            op.capacity_threshold = op.capacity_threshold.max(
                obj.cost[col].abs()
                    * bound_range(
                        self.col_upper[col],
                        self.col_lower[col],
                        ft,
                        self.is_continuous(col),
                    ),
            );
        }
        self.obj_prop = Some(op);
    }

    /// `ObjectivePropagation::updateActivityLbChange` (`lower`) or
    /// `updateActivityUbChange`.
    fn obj_update(&mut self, col: usize, oldbound: f64, newbound: f64, lower: bool) {
        let mut op = self
            .obj_prop
            .take()
            .expect("objective propagation inactive");
        let cost = op.obj.cost[col];
        let ft = self.feastol;
        let cont = self.is_continuous(col);
        // The bound that moves the objective's lower bound: a lower bound for a
        // positive cost, an upper bound for a negative one.
        let moves_lower = if lower { cost > 0.0 } else { cost < 0.0 };
        if !moves_lower {
            let loosened = if lower {
                newbound < oldbound
            } else {
                newbound > oldbound
            };
            if cost != 0.0 && loosened {
                let range = if lower {
                    bound_range(self.col_upper[col], newbound, ft, cont)
                } else {
                    bound_range(newbound, self.col_lower[col], ft, cont)
                };
                op.capacity_threshold = op.capacity_threshold.max(cost.abs() * range);
                op.is_propagated = false;
            }
            self.obj_prop = Some(op);
            return;
        }
        op.is_propagated = false;
        let inf = if lower { -INF } else { INF };
        let partition_pos = op.obj.col_partition_pos(col);
        let mut infeasible = false;
        if partition_pos == -1 {
            if oldbound == inf {
                op.num_inf_obj_lower -= 1;
            } else {
                op.objective_lower -= oldbound * cost;
            }
            if newbound == inf {
                op.num_inf_obj_lower += 1;
            } else {
                op.objective_lower += newbound * cost;
            }
            let loosened = if lower {
                newbound < oldbound
            } else {
                newbound > oldbound
            };
            if loosened {
                op.capacity_threshold = op.capacity_threshold.max(
                    cost.abs() * bound_range(self.col_upper[col], self.col_lower[col], ft, cont),
                );
            } else if op.num_inf_obj_lower == 0 && op.objective_lower > self.upper_limit {
                infeasible = true;
            }
        } else {
            let pos = partition_pos as usize;
            let part = op.contributions[pos].2;
            // Relaxing the binary back to the bound that contributes its larger value.
            let relaxed = if lower {
                newbound == 0.0
            } else {
                newbound == 1.0
            };
            if relaxed {
                let curr_first = op.first(part);
                op.link(pos);
                let old_contribution = curr_first.map_or(0.0, |f| op.contributions[f].0);
                if Some(pos) == op.first(part) && op.contributions[pos].0 != old_contribution {
                    op.objective_lower += old_contribution;
                    op.objective_lower -= op.contributions[pos].0;
                    let best = op.last(part).unwrap();
                    let mut delta = op.contributions[pos].0;
                    if best != pos {
                        delta -= op.contributions[best].0;
                    }
                    op.capacity_threshold = op.capacity_threshold.max(delta * (1.0 - ft));
                } else {
                    op.capacity_threshold = op
                        .capacity_threshold
                        .max((old_contribution - op.contributions[pos].0) * (1.0 - ft));
                }
            } else {
                let was_first = op.first(part) == Some(pos);
                if was_first {
                    op.objective_lower += op.contributions[pos].0;
                }
                op.unlink(pos);
                if was_first {
                    if let Some(w) = op.first(part) {
                        op.objective_lower -= op.contributions[w].0;
                    }
                }
                if op.num_inf_obj_lower == 0 && op.objective_lower > self.upper_limit {
                    infeasible = true;
                }
            }
        }
        self.obj_prop = Some(op);
        if infeasible {
            self.infeasible = true;
            self.infeasible_pos = self.domchgstack.len();
            self.infeasible_reason = Reason::Objective;
            self.obj_update(col, newbound, oldbound, lower);
        }
    }

    /// `ObjectivePropagation::shouldBePropagated()`.
    fn obj_should_be_propagated(&self) -> bool {
        let Some(op) = self.obj_prop.as_ref() else {
            return false;
        };
        !(op.is_propagated
            || op.num_inf_obj_lower > 1
            || self.infeasible
            || self.upper_limit == INF
            || self.upper_limit - op.objective_lower.to_f64() > op.capacity_threshold)
    }

    /// `checkChangeBound(boundtype, col, boundval, reason)`.
    fn check_change_bound(
        &mut self,
        boundtype: BoundType,
        col: usize,
        boundval: CDouble,
        reason: Reason,
    ) -> bool {
        if (boundval.to_f64() * K_HIGHS_TINY).abs() > self.feastol {
            return false;
        }
        let (bound, accept) = match boundtype {
            BoundType::Lower => self.adjusted_lb(col, boundval),
            BoundType::Upper => self.adjusted_ub(col, boundval),
        };
        if !accept {
            return false;
        }
        self.change_bound(
            DomainChange {
                boundval: bound,
                column: col,
                boundtype,
            },
            reason,
        );
        true
    }

    /// `ObjectivePropagation::propagate()`.
    fn obj_propagate(&mut self) {
        if !self.obj_should_be_propagated() {
            return;
        }
        let obj = Arc::clone(&self.obj_prop.as_ref().unwrap().obj);
        let cost = &obj.cost;
        let upper_limit = self.upper_limit;
        let (ol, ninf) = {
            let op = self.obj_prop.as_ref().unwrap();
            (op.objective_lower, op.num_inf_obj_lower)
        };
        if ninf == 0 && ol > upper_limit {
            self.infeasible = true;
            self.infeasible_pos = self.domchgstack.len();
            self.infeasible_reason = Reason::Objective;
            return;
        }
        let nz = obj.nonzeros();
        let mut capacity = upper_limit - ol;
        if ninf == 1 {
            // Only the one column contributing with an infinite bound can be bounded.
            for &col in &nz[obj.num_binary()..] {
                if (cost[col] > 0.0 && self.col_lower[col] != -INF)
                    || (cost[col] < 0.0 && self.col_upper[col] != INF)
                {
                    continue;
                }
                let bt = if cost[col] > 0.0 {
                    BoundType::Upper
                } else {
                    BoundType::Lower
                };
                self.check_change_bound(bt, col, capacity / cost[col], Reason::Objective);
                break;
            }
        } else {
            let np = obj.num_partitions();
            loop {
                let mut num_bound_changes = 0usize;
                for i in 0..np {
                    let op = self.obj_prop.as_ref().unwrap();
                    let Some(worst) = op.first(i) else { continue };
                    let mut contribution = op.contributions[worst].0;
                    let second = op.successor(worst);
                    if let Some(s2) = second {
                        contribution -= op.contributions[s2].0;
                    }
                    // The upper limit already has a tolerance: a hard cutoff.
                    if contribution > capacity {
                        let col = op.contributions[worst].1;
                        if cost[col] > 0.0 {
                            if self.col_upper[col] > 0.0 {
                                num_bound_changes += 1;
                                self.change_bound(
                                    DomainChange {
                                        boundval: 0.0,
                                        column: col,
                                        boundtype: BoundType::Upper,
                                    },
                                    Reason::Objective,
                                );
                                if self.infeasible {
                                    break;
                                }
                            }
                        } else if self.col_lower[col] < 1.0 {
                            num_bound_changes += 1;
                            self.change_bound(
                                DomainChange {
                                    boundval: 1.0,
                                    column: col,
                                    boundtype: BoundType::Lower,
                                },
                                Reason::Objective,
                            );
                            if self.infeasible {
                                break;
                            }
                        }
                    } else if second.is_some() {
                        // The column with the smallest contribution may be fixed to
                        // the bound that gives its larger objective value.
                        loop {
                            let op = self.obj_prop.as_ref().unwrap();
                            let best = op.last(i).unwrap();
                            let first = op.first(i).unwrap();
                            if best == first {
                                break;
                            }
                            if CDouble::from(op.contributions[first].0 - op.contributions[best].0)
                                > capacity
                            {
                                let col = op.contributions[best].1;
                                let chg = if cost[col] > 0.0 {
                                    debug_assert!(self.col_lower[col] < 1.0);
                                    DomainChange {
                                        boundval: 1.0,
                                        column: col,
                                        boundtype: BoundType::Lower,
                                    }
                                } else {
                                    debug_assert!(self.col_upper[col] > 0.0);
                                    DomainChange {
                                        boundval: 0.0,
                                        column: col,
                                        boundtype: BoundType::Upper,
                                    }
                                };
                                num_bound_changes += 1;
                                self.change_bound(chg, Reason::Objective);
                                if self.infeasible {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        if self.infeasible {
                            break;
                        }
                    }
                }
                if self.infeasible {
                    break;
                }
                for &col in &nz[obj.partition_starts()[np]..] {
                    let changed = if cost[col] > 0.0 {
                        self.check_change_bound(
                            BoundType::Upper,
                            col,
                            (capacity + self.col_lower[col] * cost[col]) / cost[col],
                            Reason::Objective,
                        )
                    } else {
                        self.check_change_bound(
                            BoundType::Lower,
                            col,
                            (capacity + self.col_upper[col] * cost[col]) / cost[col],
                            Reason::Objective,
                        )
                    };
                    if changed {
                        num_bound_changes += 1;
                    }
                    if self.infeasible {
                        break;
                    }
                }
                if self.infeasible || num_bound_changes == 0 {
                    break;
                }
                capacity = upper_limit - self.obj_prop.as_ref().unwrap().objective_lower;
            }
        }
        self.obj_recompute_capacity_threshold();
        self.obj_prop.as_mut().unwrap().is_propagated = true;
    }

    /// `markPropagateCut(reason)`.
    fn mark_propagate_cut_reason(&mut self, reason: Reason) {
        match reason {
            Reason::Cut(pool, cut) => self.cutpools[pool].mark_propagate_cut(cut),
            Reason::Conflict(pool, conflict) => {
                self.conflictpools[pool].mark_propagate_conflict(conflict)
            }
            _ => {}
        }
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

    /// `clearChangedCols(start)`: forget the changed columns recorded from `start` on.
    pub fn clear_changed_cols_from(&mut self, start: usize) {
        for &c in &self.changedcols[start.min(self.changedcols.len())..] {
            self.changedcolsflags[c] = false;
        }
        self.changedcols.truncate(start);
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

    /// `computeMinActivity(start, end, ARindex, ARvalue, ...)` over a given row.
    fn compute_min_activity_of(&self, inds: &[usize], vals: &[f64]) -> (i32, CDouble) {
        let mut act = CDouble::from(0.0);
        let mut ninf = 0;
        for (&col, &val) in inds.iter().zip(vals) {
            let (lb, ub) = self.bounds_for_activity(col);
            let c = activity_contribution_min(val, lb, ub);
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
                rel >= self.cont_accept
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
                rel >= self.cont_accept
            } else {
                false
            };
            (bound, accept)
        }
    }

    fn propagate_row_upper(&self, row: usize, out: &mut Vec<DomainChange>) {
        let (s, e) = (self.model.ar_start[row], self.model.ar_start[row + 1]);
        self.propagate_row_upper_of(
            &self.model.ar_index[s..e],
            &self.model.ar_value[s..e],
            self.model.row_upper[row],
            self.activitymin[row],
            self.activitymininf[row],
            out,
        );
    }

    /// `propagateRowUpper(Rindex, Rvalue, Rlen, Rupper, minactivity, ninfmin, ...)`.
    fn propagate_row_upper_of(
        &self,
        inds: &[usize],
        vals: &[f64],
        rupper: f64,
        minactivity: CDouble,
        ninfmin: i32,
        out: &mut Vec<DomainChange>,
    ) {
        if ninfmin > 1 {
            return;
        }
        for (&col, &val) in inds.iter().zip(vals) {
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

    /// `CutpoolPropagation::recomputeCapacityThreshold(cut)`.
    fn recompute_cut_capacity_threshold(&mut self, pool: usize, cut: usize) {
        let mut t = -self.feastol;
        let (inds, vals) = &self.cutpools[pool].rows[cut];
        for (&col, &val) in inds.iter().zip(vals) {
            if self.col_upper[col] == self.col_lower[col] {
                continue;
            }
            let th = val.abs()
                * bound_range(
                    self.col_upper[col],
                    self.col_lower[col],
                    self.feastol,
                    self.is_continuous(col),
                );
            t = t.max(th).max(self.feastol);
        }
        self.cutpools[pool].threshold[cut] = t;
    }

    /// `CutpoolPropagation::updateActivityLbChange` (`lower`) or
    /// `updateActivityUbChange`, with HiGHS's `threshold`, `activity` and
    /// `infeasdomain` arguments.
    #[allow(clippy::too_many_arguments)]
    fn update_cut_activity(
        &mut self,
        pool: usize,
        col: usize,
        oldbound: f64,
        newbound: f64,
        lower: bool,
        threshold: bool,
        activity: bool,
        infeasdomain: bool,
    ) {
        // A loosened bound widens the capacity of the cuts it bounds from the other
        // side: negative entries for a lower bound, positive for an upper bound.
        let loosened = if lower {
            newbound < oldbound
        } else {
            newbound > oldbound
        };
        if loosened && threshold {
            let n = if lower {
                self.cutpools[pool].col_neg[col].len()
            } else {
                self.cutpools[pool].col_pos[col].len()
            };
            for j in (0..n).rev() {
                let (row, val) = if lower {
                    self.cutpools[pool].col_neg[col][j]
                } else {
                    self.cutpools[pool].col_pos[col][j]
                };
                let t = self.cutpools[pool].threshold[row];
                self.cutpools[pool].threshold[row] = if lower {
                    self.update_threshold_lb_change(col, newbound, val, t)
                } else {
                    self.update_threshold_ub_change(col, newbound, val, t)
                };
            }
        }
        if !activity {
            return;
        }
        // The entries whose minimum activity the bound moves: positive for a lower
        // bound, negative for an upper bound.
        let n = if lower {
            self.cutpools[pool].col_pos[col].len()
        } else {
            self.cutpools[pool].col_neg[col].len()
        };
        let entry = |cp: &CutpoolProp, j: usize| {
            if lower {
                cp.col_pos[col][j]
            } else {
                cp.col_neg[col][j]
            }
        };
        let inf = if lower { -INF } else { INF };
        if !infeasdomain {
            for j in (0..n).rev() {
                let (row, val) = entry(&self.cutpools[pool], j);
                let cp = &mut self.cutpools[pool];
                let delta = compute_delta(val, oldbound, newbound, inf, &mut cp.activity_inf[row]);
                cp.activity[row] += delta;
                if delta <= 0.0 {
                    let t = self.cutpools[pool].threshold[row];
                    self.cutpools[pool].threshold[row] = if lower {
                        self.update_threshold_lb_change(col, newbound, val, t)
                    } else {
                        self.update_threshold_ub_change(col, newbound, val, t)
                    };
                    continue;
                }
                let cp = &mut self.cutpools[pool];
                if cp.activity_inf[row] == 0
                    && (cp.activity[row] - cp.rhs[row]).to_f64() > self.feastol
                {
                    self.infeasible = true;
                    self.infeasible_pos = self.domchgstack.len();
                    self.infeasible_reason = Reason::Cut(pool, row);
                    break;
                }
                cp.mark_propagate_cut(row);
            }
        }
        if self.infeasible {
            // Reverse this pool's changes, up to and including the culprit cut.
            for j in (0..n).rev() {
                let (row, val) = entry(&self.cutpools[pool], j);
                let cp = &mut self.cutpools[pool];
                let d = compute_delta(val, newbound, oldbound, inf, &mut cp.activity_inf[row]);
                cp.activity[row] += d;
                if self.infeasible_reason == Reason::Cut(pool, row) {
                    break;
                }
            }
        }
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
        if self.obj_prop.is_some() {
            self.obj_update(col, oldbound, newbound, lower);
            if self.infeasible {
                return;
            }
        }
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
        if !self.infeasible {
            // If cut pool i finds the domain infeasible, pools after it still update
            // their thresholds, and pools before it reverse their activity changes.
            let mut infeas_pool: Option<usize> = None;
            for p in 0..self.cutpools.len() {
                if !self.infeasible {
                    self.update_cut_activity(p, col, oldbound, newbound, lower, true, true, false);
                    if self.infeasible {
                        infeas_pool = Some(p);
                    }
                } else {
                    self.update_cut_activity(p, col, oldbound, newbound, lower, true, false, true);
                }
            }
            if let Some(ip) = infeas_pool {
                for p in 0..ip {
                    self.update_cut_activity(p, col, oldbound, newbound, lower, false, true, true);
                }
            }
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
            if self.obj_prop.is_some() {
                self.obj_update(col, newbound, oldbound, lower);
            }
        } else {
            for p in 0..self.conflictpools.len() {
                self.update_conflict_activity(p, col, oldbound, newbound, lower);
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
        let old_reason = self.infeasible_reason;
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
            self.mark_propagate_cut_reason(old_reason);
            self.infeasible_reason = Reason::Unspecified;
            self.infeasible = false;
        }
        for i in (k + 1) as usize..self.domchgreason.len() {
            self.mark_propagate_cut_reason(self.domchgreason[i]);
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

    /// `setDomainChangeStack(domchgstack, branchingPositions)`: install a node's bound
    /// changes on a domain that is at its global bounds (HiGHS installs only after the
    /// search backtracked to the global domain). Changes already implied are skipped;
    /// each applied change is propagated; a branching change is kept as a branching
    /// position unless it is strictly redundant or a local change already implies it.
    pub fn set_domain_change_stack(
        &mut self,
        stack: &[DomainChange],
        branching_positions: &[usize],
    ) {
        self.infeasible = false;
        for c in &self.domchgstack {
            match c.boundtype {
                BoundType::Lower => self.col_lower_pos[c.column] = -1,
                BoundType::Upper => self.col_upper_pos[c.column] = -1,
            }
        }
        self.prevboundval.clear();
        self.domchgstack.clear();
        self.domchgreason.clear();
        self.branch_pos.clear();
        let implied = |d: &Domain, c: &DomainChange| match c.boundtype {
            BoundType::Lower => c.boundval <= d.col_lower[c.column],
            BoundType::Upper => c.boundval >= d.col_upper[c.column],
        };
        let mut k = 0usize;
        for &bp in branching_positions {
            while k < bp {
                if !implied(self, &stack[k]) {
                    self.change_bound(stack[k], Reason::Unspecified);
                    if !self.infeasible {
                        self.propagate();
                    }
                    if self.infeasible {
                        return;
                    }
                }
                k += 1;
            }
            if k == stack.len() {
                return;
            }
            let c = stack[k];
            let skip = match c.boundtype {
                BoundType::Lower => {
                    c.boundval <= self.col_lower[c.column]
                        && (c.boundval < self.col_lower[c.column]
                            || self.col_lower_pos[c.column] != -1)
                }
                BoundType::Upper => {
                    c.boundval >= self.col_upper[c.column]
                        && (c.boundval > self.col_upper[c.column]
                            || self.col_upper_pos[c.column] != -1)
                }
            };
            if !skip {
                self.change_bound(c, Reason::Branching);
                if !self.infeasible {
                    self.propagate();
                }
                if self.infeasible {
                    return;
                }
            }
            k += 1;
        }
        while k < stack.len() {
            if !implied(self, &stack[k]) {
                self.change_bound(stack[k], Reason::Unspecified);
                if !self.infeasible {
                    self.propagate();
                }
                if self.infeasible {
                    break;
                }
            }
            k += 1;
        }
    }

    /// `getReducedDomainChangeStack(branchingPositions)`: the tightest change per column
    /// and bound, in stack order, with the positions of those that are (or descend from)
    /// branching changes.
    pub fn reduced_domain_change_stack(&self) -> (Vec<DomainChange>, Vec<usize>) {
        let mut reduced = Vec::with_capacity(self.domchgstack.len());
        let mut positions = Vec::with_capacity(self.branch_pos.len());
        for (i, c) in self.domchgstack.iter().enumerate() {
            let last = match c.boundtype {
                BoundType::Lower => self.col_lower_pos[c.column],
                BoundType::Upper => self.col_upper_pos[c.column],
            };
            if last != i as i64 {
                continue;
            }
            if self.domchgreason[i] == Reason::Branching {
                positions.push(reduced.len());
            } else {
                let mut k = i;
                while self.prevboundval[k].1 != -1 {
                    k = self.prevboundval[k].1 as usize;
                    if self.domchgreason[k] == Reason::Branching {
                        positions.push(reduced.len());
                        break;
                    }
                }
            }
            reduced.push(*c);
        }
        (reduced, positions)
    }

    /// `backtrackToGlobal()`.
    pub fn backtrack_to_global(&mut self) {
        let mut k = self.domchgstack.len() as i64 - 1;
        let old_infeasible = self.infeasible;
        let old_reason = self.infeasible_reason;
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
            self.mark_propagate_cut_reason(old_reason);
            self.infeasible_reason = Reason::Unspecified;
            self.infeasible = false;
        }
        for i in 0..self.domchgreason.len() {
            self.mark_propagate_cut_reason(self.domchgreason[i]);
        }
        self.domchgstack.clear();
        self.prevboundval.clear();
        self.domchgreason.clear();
        self.branch_pos.clear();
    }

    /// Make the current bounds the global baseline: clear the change stack so
    /// [`Self::backtrack_to_global`] returns here (HiGHS keeps global-domain changes off
    /// a local domain's stack: the local domain starts as a copy of the propagated
    /// global one). Only valid when the current bounds hold globally, e.g. after
    /// propagating the global box.
    pub fn commit_as_global(&mut self) {
        assert!(!self.infeasible, "cannot commit an infeasible domain");
        for c in &self.domchgstack {
            match c.boundtype {
                BoundType::Lower => self.col_lower_pos[c.column] = -1,
                BoundType::Upper => self.col_upper_pos[c.column] = -1,
            }
        }
        self.domchgstack.clear();
        self.prevboundval.clear();
        self.domchgreason.clear();
        self.branch_pos.clear();
    }

    /// `propagate()`: true when there was anything to propagate.
    pub fn propagate(&mut self) -> bool {
        if !self.have_propagation_rows() {
            return false;
        }
        let model = Arc::clone(&self.model);
        let ft = self.feastol;
        let mut inds: Vec<usize> = Vec::new();
        let mut buf: Vec<(usize, Vec<DomainChange>, Vec<DomainChange>)> = Vec::new();
        while self.have_propagation_rows() {
            if self.obj_prop.is_some() {
                self.obj_propagate();
            }
            for pool in 0..self.conflictpools.len() {
                while !self.conflictpools[pool].propagate_conflict_inds.is_empty() {
                    let conflicts =
                        std::mem::take(&mut self.conflictpools[pool].propagate_conflict_inds);
                    for conflict in conflicts {
                        self.propagate_conflict(pool, conflict);
                    }
                }
            }
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
            for pool in 0..self.cutpools.len() {
                if self.cutpools[pool].inds.is_empty() {
                    continue;
                }
                std::mem::swap(&mut inds, &mut self.cutpools[pool].inds);
                for &cut in &inds {
                    self.cutpools[pool].flags[cut] &= 2;
                }
                if !self.infeasible {
                    let mut changes: Vec<(usize, Vec<DomainChange>)> = Vec::new();
                    for &cut in &inds {
                        let mut up = Vec::new();
                        // A cut marked deleted is skipped.
                        if self.cutpools[pool].flags[cut] & 2 == 0 {
                            self.cutpools[pool].activity[cut].renormalize();
                            let cp = &self.cutpools[pool];
                            let (ri, rv) = &cp.rows[cut];
                            self.propagate_row_upper_of(
                                ri,
                                rv,
                                cp.rhs[cut],
                                cp.activity[cut],
                                cp.activity_inf[cut],
                                &mut up,
                            );
                            self.recompute_cut_capacity_threshold(pool, cut);
                        }
                        changes.push((cut, up));
                    }
                    for (cut, up) in changes {
                        if !up.is_empty() {
                            self.cut_age_resets.push((pool, cut));
                            for c in up {
                                if self.infeasible {
                                    break;
                                }
                                self.change_bound(c, Reason::Cut(pool, cut));
                            }
                        }
                        if self.infeasible {
                            break;
                        }
                    }
                }
                inds.clear();
            }
        }
        true
    }

    /// `havePropagationRows()`.
    fn have_propagation_rows(&self) -> bool {
        !self.propagateinds.is_empty()
            || self.obj_should_be_propagated()
            || self.cutpools.iter().any(|cp| !cp.inds.is_empty())
            || self
                .conflictpools
                .iter()
                .any(|cp| !cp.propagate_conflict_inds.is_empty())
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

/// A conflict-analysis frontier: `std::set<LocalDomChg>`, ordered by stack position,
/// each with its (possibly relaxed) bound change.
type Frontier = BTreeMap<i64, DomainChange>;

/// `ConflictSet::ResolveCandidate`.
#[derive(Clone, Copy, Debug, Default)]
struct ResolveCandidate {
    delta: f64,
    base_bound: f64,
    prio: f64,
    bound_pos: i64,
    value_pos: usize,
}

/// `ResolveCandidate::operator<`: higher priority first, then the earlier position.
fn candidate_less(a: &ResolveCandidate, b: &ResolveCandidate) -> bool {
    if a.prio > b.prio {
        return true;
    }
    if b.prio > a.prio {
        return false;
    }
    a.bound_pos < b.bound_pos
}

/// `computePrio(val, bound, globalbound, numnodes)`.
fn compute_prio(val: f64, bound: f64, globalbound: f64, numnodes: i64) -> f64 {
    (val * (bound - globalbound) * (1 + numnodes) as f64).abs()
}

/// `HighsDomain::ConflictSet`: the frontiers, resolve queue and buffers of one
/// conflict analysis of `localdom` against `globaldom`.
///
/// HiGHS keeps the queue as a heap of iterators into a frontier set ordered by
/// position; positions in a frontier are unique and an entry is queued at most once
/// while it is in the frontier, so a max-heap of positions pops in the same order.
struct ConflictSet<'a> {
    localdom: &'a mut Domain,
    /// Read only, except that the conflict pool notifies it of new conflicts.
    globaldom: &'a mut Domain,
    links: ConflictLinks,
    /// `mipdata_->nodequeue`.
    nodes: &'a dyn OpenNodeCounts,
    reason_side_frontier: Frontier,
    reconvergence_frontier: Frontier,
    resolve_queue: BinaryHeap<i64>,
    resolved_domain_changes: Vec<(i64, DomainChange)>,
    resolve_buffer: Vec<ResolveCandidate>,
    /// The conflicts added to the pool, in order.
    added: Vec<usize>,
}

impl<'a> ConflictSet<'a> {
    fn new(
        localdom: &'a mut Domain,
        globaldom: &'a mut Domain,
        links: ConflictLinks,
        nodes: &'a dyn OpenNodeCounts,
    ) -> Self {
        let cap = localdom.domchgstack.len();
        ConflictSet {
            localdom,
            globaldom,
            links,
            nodes,
            reason_side_frontier: Frontier::new(),
            reconvergence_frontier: Frontier::new(),
            resolve_queue: BinaryHeap::new(),
            resolved_domain_changes: Vec::with_capacity(cap),
            resolve_buffer: Vec::new(),
            added: Vec::new(),
        }
    }

    fn feastol(&self) -> f64 {
        self.localdom.feastol
    }

    fn epsilon(&self) -> f64 {
        self.localdom.epsilon
    }

    fn is_continuous(&self, col: usize) -> bool {
        !self.localdom.model.is_int[col]
    }

    /// The pool's `addConflictCut`/`addReconvergenceCut` storage and notification:
    /// the global domain first, then the local one (`propagationDomains` order).
    fn add_to_pool(&mut self, pool: &mut ConflictPool, entries: &[DomainChange]) {
        let mut obs: Vec<(&mut Domain, usize)> = Vec::with_capacity(2);
        if let Some(g) = self.links.global {
            obs.push((&mut *self.globaldom, g));
        }
        if let Some(l) = self.links.local {
            obs.push((&mut *self.localdom, l));
        }
        let c = pool.add_conflict_cut(entries, &mut ConflictDomainObservers(obs));
        self.added.push(c);
    }

    /// The rows a reason or the infeasibility refers to: model row `row`.
    fn model_row(&self, row: usize) -> (Vec<usize>, Vec<f64>) {
        let m = &self.localdom.model;
        let (s, e) = (m.ar_start[row], m.ar_start[row + 1]);
        (m.ar_index[s..e].to_vec(), m.ar_value[s..e].to_vec())
    }

    /// Cut `cut` of cut pool `pool` (`cutpool->getCut`, `getRhs`), from the local
    /// domain's mirror of the pool.
    fn cut_row(&self, pool: usize, cut: usize) -> (Vec<usize>, Vec<f64>, f64) {
        let cp = &self.localdom.cutpools[pool];
        let (i, v) = &cp.rows[cut];
        (i.clone(), v.clone(), cp.rhs[cut])
    }

    /// The conflict behind a [`Reason::Conflict`], `None` when the local domain has
    /// it flagged deleted (`conflictFlag_ & 8`).
    fn conflict_entries(&self, pool: usize, conflict: usize) -> Option<Vec<DomainChange>> {
        let cp = &self.localdom.conflictpools[pool];
        if cp.conflict_flag[conflict] & 8 != 0 {
            return None;
        }
        cp.conflicts[conflict].clone()
    }

    /// `explainBoundChangeGeq(currentFrontier, domchg, inds, vals, len, rhs, maxAct)`.
    fn explain_bound_change_geq(
        &mut self,
        current_frontier: &Frontier,
        domchg: (i64, DomainChange),
        inds: &[usize],
        vals: &[f64],
        rhs: f64,
        max_act: f64,
    ) -> bool {
        if max_act == INF {
            return false;
        }
        let mut domchg_val = 0.0;
        self.resolve_buffer.clear();
        let mut m = CDouble::from(max_act);
        for i in 0..inds.len() {
            let col = inds[i];
            if col == domchg.1.column {
                domchg_val = vals[i];
                continue;
            }
            let mut cand = ResolveCandidate {
                value_pos: i,
                ..Default::default()
            };
            if vals[i] > 0.0 {
                let (ub, bp) = self.localdom.col_upper_pos(col, domchg.0);
                cand.bound_pos = bp;
                let gub = self.globaldom.col_upper[col];
                if gub <= ub || bp == -1 {
                    continue;
                }
                if let Some(fd) = current_frontier.get(&bp) {
                    cand.base_bound = fd.boundval;
                    if cand.base_bound != gub {
                        m += vals[i] * (cand.base_bound - gub);
                    }
                    if cand.base_bound <= ub {
                        continue;
                    }
                } else {
                    cand.base_bound = gub;
                }
                cand.delta = vals[i] * (ub - cand.base_bound);
                cand.prio = compute_prio(vals[i], ub, gub, self.nodes.num_nodes_down(col));
            } else {
                let (lb, bp) = self.localdom.col_lower_pos(col, domchg.0);
                cand.bound_pos = bp;
                let glb = self.globaldom.col_lower[col];
                if glb >= lb || bp == -1 {
                    continue;
                }
                if let Some(fd) = current_frontier.get(&bp) {
                    cand.base_bound = fd.boundval;
                    if cand.base_bound != glb {
                        m += vals[i] * (cand.base_bound - glb);
                    }
                    if cand.base_bound >= lb {
                        continue;
                    }
                } else {
                    cand.base_bound = glb;
                }
                cand.delta = vals[i] * (lb - cand.base_bound);
                cand.prio = compute_prio(vals[i], lb, glb, self.nodes.num_nodes_up(col));
            }
            self.resolve_buffer.push(cand);
        }
        if domchg_val == 0.0 {
            return false;
        }
        pdqsort(&mut self.resolve_buffer, candidate_less);
        // The bound constraint a0 x0 >= b0 is implied by a0 x0 + sum ai xi >= b when
        // the residual maximum activity M is at most b - b0; M is lowered with the
        // stronger local bounds until that holds.
        let col = domchg.1.column;
        let mut b0 = domchg.1.boundval;
        if !self.is_continuous(col) {
            // An integer bound was rounded and can be relaxed by 1 - 10 feastol.
            if domchg.1.boundtype == BoundType::Lower {
                b0 -= 1.0 - 10.0 * self.feastol();
            } else {
                b0 += 1.0 - 10.0 * self.feastol();
            }
        } else if domchg.1.boundtype == BoundType::Lower {
            b0 -= self.epsilon();
        } else {
            b0 += self.epsilon();
        }
        b0 *= domchg_val;
        let mupper = rhs - b0;
        if domchg_val < 0.0 {
            m -= domchg_val * self.globaldom.col_lower[col];
        } else {
            m -= domchg_val * self.globaldom.col_upper[col];
        }
        self.resolve_linear_geq(m, mupper, vals)
    }

    /// `explainBoundChangeLeq(currentFrontier, domchg, inds, vals, len, rhs, minAct)`.
    fn explain_bound_change_leq(
        &mut self,
        current_frontier: &Frontier,
        domchg: (i64, DomainChange),
        inds: &[usize],
        vals: &[f64],
        rhs: f64,
        min_act: f64,
    ) -> bool {
        if min_act == -INF {
            return false;
        }
        let mut domchg_val = 0.0;
        self.resolve_buffer.clear();
        let mut m = CDouble::from(min_act);
        for i in 0..inds.len() {
            let col = inds[i];
            if col == domchg.1.column {
                domchg_val = vals[i];
                continue;
            }
            let mut cand = ResolveCandidate {
                value_pos: i,
                ..Default::default()
            };
            if vals[i] > 0.0 {
                let (lb, bp) = self.localdom.col_lower_pos(col, domchg.0);
                cand.bound_pos = bp;
                let glb = self.globaldom.col_lower[col];
                if glb >= lb || bp == -1 {
                    continue;
                }
                if let Some(fd) = current_frontier.get(&bp) {
                    cand.base_bound = fd.boundval;
                    if cand.base_bound != glb {
                        m += vals[i] * (cand.base_bound - glb);
                    }
                    if cand.base_bound >= lb {
                        continue;
                    }
                } else {
                    cand.base_bound = glb;
                }
                cand.delta = vals[i] * (lb - cand.base_bound);
                cand.prio = compute_prio(vals[i], lb, glb, self.nodes.num_nodes_up(col));
            } else {
                let (ub, bp) = self.localdom.col_upper_pos(col, domchg.0);
                cand.bound_pos = bp;
                let gub = self.globaldom.col_upper[col];
                if gub <= ub || bp == -1 {
                    continue;
                }
                if let Some(fd) = current_frontier.get(&bp) {
                    cand.base_bound = fd.boundval;
                    if cand.base_bound != gub {
                        m += vals[i] * (cand.base_bound - gub);
                    }
                    if cand.base_bound <= ub {
                        continue;
                    }
                } else {
                    cand.base_bound = gub;
                }
                cand.delta = vals[i] * (ub - cand.base_bound);
                cand.prio = compute_prio(vals[i], ub, gub, self.nodes.num_nodes_down(col));
            }
            self.resolve_buffer.push(cand);
        }
        if domchg_val == 0.0 {
            return false;
        }
        pdqsort(&mut self.resolve_buffer, candidate_less);
        // The bound constraint a0 x0 <= b0 is implied by a0 x0 + sum ai xi <= b when
        // the residual minimum activity M is at least b - b0; M is raised with the
        // stronger local bounds until that holds.
        let col = domchg.1.column;
        let mut b0 = domchg.1.boundval;
        if !self.is_continuous(col) {
            if domchg.1.boundtype == BoundType::Lower {
                b0 -= 1.0 - 10.0 * self.feastol();
            } else {
                b0 += 1.0 - 10.0 * self.feastol();
            }
        } else if domchg.1.boundtype == BoundType::Lower {
            b0 -= self.epsilon();
        } else {
            b0 += self.epsilon();
        }
        b0 *= domchg_val;
        let mlower = rhs - b0;
        if domchg_val < 0.0 {
            m -= domchg_val * self.globaldom.col_upper[col];
        } else {
            m -= domchg_val * self.globaldom.col_lower[col];
        }
        self.resolve_linear_leq(m, mlower, vals)
    }

    /// Walk a relaxed lower bound back to the earliest stack position that still
    /// implies it (`while (relaxLb <= prevboundval_[pos].first) pos = ...`). HiGHS
    /// would index position -1 if the walk ran off the stack; it stops at the
    /// earliest change instead, which still implies the relaxed bound.
    fn walk_back_lower(&self, mut pos: i64, relax: f64) -> i64 {
        let prev = &self.localdom.prevboundval;
        while relax <= prev[pos as usize].0 && prev[pos as usize].1 != -1 {
            pos = prev[pos as usize].1;
        }
        pos
    }

    /// The upper-bound counterpart of [`Self::walk_back_lower`].
    fn walk_back_upper(&self, mut pos: i64, relax: f64) -> i64 {
        let prev = &self.localdom.prevboundval;
        while relax >= prev[pos as usize].0 && prev[pos as usize].1 != -1 {
            pos = prev[pos as usize].1;
        }
        pos
    }

    /// `resolveLinearGeq(M, Mupper, vals)`.
    fn resolve_linear_geq(&mut self, mut m: CDouble, mupper: f64, vals: &[f64]) -> bool {
        self.resolved_domain_changes.clear();
        let ft = self.feastol();
        let eps = self.epsilon();
        let mut covered = (m - mupper).to_f64();
        if covered > 0.0 {
            for k in 0..self.resolve_buffer.len() {
                let cand = self.resolve_buffer[k];
                let p = cand.bound_pos;
                self.resolved_domain_changes
                    .push((p, self.localdom.domchgstack[p as usize]));
                m += cand.delta;
                covered = (m - mupper).to_f64();
                if covered <= 0.0 {
                    break;
                }
            }
            if covered > 0.0 {
                return false;
            }
            if covered < -ft {
                // Room to relax bounds or drop bound changes from the explanation.
                let mut k = self.resolved_domain_changes.len() as i64 - 1;
                while k >= 0 {
                    let ku = k as usize;
                    let cand = self.resolve_buffer[ku];
                    let i = cand.value_pos;
                    let (pos, d) = self.resolved_domain_changes[ku];
                    let col = d.column;
                    if d.boundtype == BoundType::Lower {
                        let lb = d.boundval;
                        let glb = cand.base_bound;
                        let mut relax_lb = ((mupper - (m - cand.delta)) / vals[i] + glb).to_f64();
                        if !self.is_continuous(col) {
                            relax_lb = relax_lb.ceil();
                        }
                        if relax_lb - lb >= -ft {
                            k -= 1;
                            continue;
                        }
                        self.resolved_domain_changes[ku].1.boundval = relax_lb;
                        if relax_lb - glb <= eps {
                            // Dropped from the explanation altogether.
                            let last = self.resolved_domain_changes.len() - 1;
                            self.resolved_domain_changes.swap(last, ku);
                            self.resolved_domain_changes.truncate(last);
                            m -= cand.delta;
                        } else {
                            self.resolved_domain_changes[ku].0 =
                                self.walk_back_lower(pos, relax_lb);
                            m += vals[i] * (relax_lb - lb);
                        }
                        covered = (m - mupper).to_f64();
                        if covered >= -ft {
                            break;
                        }
                    } else {
                        let ub = d.boundval;
                        let gub = cand.base_bound;
                        let mut relax_ub = ((mupper - (m - cand.delta)) / vals[i] + gub).to_f64();
                        if !self.is_continuous(col) {
                            relax_ub = relax_ub.floor();
                        }
                        if relax_ub - ub <= ft {
                            k -= 1;
                            continue;
                        }
                        self.resolved_domain_changes[ku].1.boundval = relax_ub;
                        if relax_ub - gub >= -eps {
                            let last = self.resolved_domain_changes.len() - 1;
                            self.resolved_domain_changes.swap(last, ku);
                            self.resolved_domain_changes.truncate(last);
                            m -= cand.delta;
                        } else {
                            self.resolved_domain_changes[ku].0 =
                                self.walk_back_upper(pos, relax_ub);
                            m += vals[i] * (relax_ub - ub);
                        }
                        covered = (m - mupper).to_f64();
                        if covered >= -ft {
                            break;
                        }
                    }
                    k -= 1;
                }
                debug_assert!(covered <= ft);
            }
        }
        true
    }

    /// `resolveLinearLeq(M, Mlower, vals)`.
    fn resolve_linear_leq(&mut self, mut m: CDouble, mlower: f64, vals: &[f64]) -> bool {
        self.resolved_domain_changes.clear();
        let ft = self.feastol();
        let eps = self.epsilon();
        let mut covered = (m - mlower).to_f64();
        if covered < 0.0 {
            for k in 0..self.resolve_buffer.len() {
                let cand = self.resolve_buffer[k];
                let p = cand.bound_pos;
                self.resolved_domain_changes
                    .push((p, self.localdom.domchgstack[p as usize]));
                m += cand.delta;
                covered = (m - mlower).to_f64();
                if covered >= 0.0 {
                    break;
                }
            }
            if covered < 0.0 {
                return false;
            }
            if covered > ft {
                let mut k = self.resolved_domain_changes.len() as i64 - 1;
                while k >= 0 {
                    let ku = k as usize;
                    let cand = self.resolve_buffer[ku];
                    let i = cand.value_pos;
                    let (pos, d) = self.resolved_domain_changes[ku];
                    let col = d.column;
                    if d.boundtype == BoundType::Lower {
                        let lb = d.boundval;
                        let glb = cand.base_bound;
                        let mut relax_lb = ((mlower - (m - cand.delta)) / vals[i] + glb).to_f64();
                        if !self.is_continuous(col) {
                            relax_lb = relax_lb.ceil();
                        }
                        if relax_lb - lb >= -ft {
                            k -= 1;
                            continue;
                        }
                        self.resolved_domain_changes[ku].1.boundval = relax_lb;
                        if relax_lb - glb <= eps {
                            let last = self.resolved_domain_changes.len() - 1;
                            self.resolved_domain_changes.swap(last, ku);
                            self.resolved_domain_changes.truncate(last);
                            m -= cand.delta;
                        } else {
                            self.resolved_domain_changes[ku].0 =
                                self.walk_back_lower(pos, relax_lb);
                            m += vals[i] * (relax_lb - lb);
                        }
                        covered = (m - mlower).to_f64();
                        if covered <= ft {
                            break;
                        }
                    } else {
                        let ub = d.boundval;
                        let gub = cand.base_bound;
                        let mut relax_ub = ((mlower - (m - cand.delta)) / vals[i] + gub).to_f64();
                        if !self.is_continuous(col) {
                            relax_ub = relax_ub.floor();
                        }
                        if relax_ub - ub <= ft {
                            k -= 1;
                            continue;
                        }
                        self.resolved_domain_changes[ku].1.boundval = relax_ub;
                        if relax_ub - gub >= -eps {
                            let last = self.resolved_domain_changes.len() - 1;
                            self.resolved_domain_changes.swap(last, ku);
                            self.resolved_domain_changes.truncate(last);
                            m -= cand.delta;
                        } else {
                            self.resolved_domain_changes[ku].0 =
                                self.walk_back_upper(pos, relax_ub);
                            m += vals[i] * (relax_ub - ub);
                        }
                        covered = (m - mlower).to_f64();
                        if covered <= ft {
                            break;
                        }
                    }
                    k -= 1;
                }
                debug_assert!(covered >= -ft);
            }
        }
        true
    }

    /// `explainInfeasibility()`.
    fn explain_infeasibility(&mut self) -> bool {
        match self.localdom.infeasible_reason {
            Reason::Unspecified | Reason::Branching | Reason::RedCost => false,
            Reason::ConflictingBounds(pos) => {
                self.resolved_domain_changes.clear();
                let p = pos as i64;
                let chg = self.localdom.domchgstack[pos];
                self.resolved_domain_changes.push((p, chg));
                let other = if chg.boundtype == BoundType::Lower {
                    self.localdom.col_upper_pos(chg.column, p).1
                } else {
                    self.localdom.col_lower_pos(chg.column, p).1
                };
                if other != -1 {
                    self.resolved_domain_changes
                        .push((other, self.localdom.domchgstack[other as usize]));
                }
                true
            }
            // HiGHS asserts here: a clique-table infeasibility is never recorded.
            Reason::CliqueTable(..) => false,
            Reason::ModelRowLower(row) => {
                let (inds, vals) = self.model_row(row);
                let max_act = self.globaldom.max_activity(row);
                let rl = self.localdom.model.row_lower[row];
                self.explain_infeasibility_geq(&inds, &vals, rl, max_act)
            }
            Reason::ModelRowUpper(row) => {
                let (inds, vals) = self.model_row(row);
                let min_act = self.globaldom.min_activity(row);
                let ru = self.localdom.model.row_upper[row];
                self.explain_infeasibility_leq(&inds, &vals, ru, min_act)
            }
            Reason::Objective => {
                let ip = self.localdom.infeasible_pos as i64;
                let (inds, vals, rhs) = self.localdom.objective_propagation_constraint(ip, None);
                let (ninfmin, min_act) = self.globaldom.compute_min_activity_of(&inds, &vals);
                // A globally unbounded column bounded locally (e.g. by a heuristic's
                // branching).
                if ninfmin > 0 {
                    return false;
                }
                self.explain_infeasibility_leq(&inds, &vals, rhs, min_act.to_f64())
            }
            Reason::Cut(pool, cut) => {
                let (inds, vals, rhs) = self.cut_row(pool, cut);
                let min_act = self.globaldom.min_cut_activity(pool, cut);
                self.explain_infeasibility_leq(&inds, &vals, rhs, min_act)
            }
            Reason::Conflict(pool, conflict) => match self.conflict_entries(pool, conflict) {
                None => false,
                Some(entries) => self.explain_infeasibility_conflict(&entries),
            },
        }
    }

    /// `explainInfeasibilityConflict(conflict, len)`.
    fn explain_infeasibility_conflict(&mut self, conflict: &[DomainChange]) -> bool {
        self.resolved_domain_changes.clear();
        let ip = self.localdom.infeasible_pos as i64;
        for e in conflict {
            if self.globaldom.is_active(e) {
                continue;
            }
            let prev = &self.localdom.prevboundval;
            let mut pos;
            if e.boundtype == BoundType::Lower {
                let (lb, p) = self.localdom.col_lower_pos(e.column, ip);
                pos = p;
                if pos == -1 || lb < e.boundval {
                    return false;
                }
                while prev[pos as usize].0 >= e.boundval {
                    pos = prev[pos as usize].1;
                    // HiGHS asserts this cannot happen (the literal is inactive
                    // globally); a local domain whose start is tighter than the
                    // global domain can reach it, and then there is no explanation.
                    if pos == -1 {
                        return false;
                    }
                }
            } else {
                let (ub, p) = self.localdom.col_upper_pos(e.column, ip);
                pos = p;
                if pos == -1 || ub > e.boundval {
                    return false;
                }
                while prev[pos as usize].0 <= e.boundval {
                    pos = prev[pos as usize].1;
                    if pos == -1 {
                        return false;
                    }
                }
            }
            self.resolved_domain_changes.push((pos, *e));
        }
        true
    }

    /// The local bound positions an infeasibility explanation starts from:
    /// `infeasible_pos` when the domain is infeasible, `kHighsIInf` otherwise.
    fn infeasible_stack_pos(&self) -> i64 {
        if self.localdom.infeasible {
            self.localdom.infeasible_pos as i64
        } else {
            i64::MAX
        }
    }

    /// `explainInfeasibilityGeq(inds, vals, len, rhs, maxActivity)`.
    fn explain_infeasibility_geq(
        &mut self,
        inds: &[usize],
        vals: &[f64],
        rhs: f64,
        max_act: f64,
    ) -> bool {
        if max_act == INF {
            return false;
        }
        let ip = self.infeasible_stack_pos();
        self.resolve_buffer.clear();
        for i in 0..inds.len() {
            let col = inds[i];
            let mut cand = ResolveCandidate {
                value_pos: i,
                ..Default::default()
            };
            if vals[i] > 0.0 {
                let (ub, bp) = self.localdom.col_upper_pos(col, ip);
                cand.bound_pos = bp;
                cand.base_bound = self.globaldom.col_upper[col];
                if cand.base_bound <= ub || bp == -1 {
                    continue;
                }
                cand.delta = vals[i] * (ub - cand.base_bound);
                cand.prio = compute_prio(
                    vals[i],
                    ub,
                    self.globaldom.col_upper[col],
                    self.nodes.num_nodes_down(col),
                );
            } else {
                let (lb, bp) = self.localdom.col_lower_pos(col, ip);
                cand.bound_pos = bp;
                cand.base_bound = self.globaldom.col_lower[col];
                if cand.base_bound >= lb || bp == -1 {
                    continue;
                }
                cand.delta = vals[i] * (lb - cand.base_bound);
                cand.prio = compute_prio(
                    vals[i],
                    lb,
                    self.globaldom.col_lower[col],
                    self.nodes.num_nodes_up(col),
                );
            }
            self.resolve_buffer.push(cand);
        }
        pdqsort(&mut self.resolve_buffer, candidate_less);
        let mupper = rhs - 10.0f64.max(rhs.abs()) * self.feastol();
        self.resolve_linear_geq(CDouble::from(max_act), mupper, vals)
    }

    /// `explainInfeasibilityLeq(inds, vals, len, rhs, minActivity)`.
    fn explain_infeasibility_leq(
        &mut self,
        inds: &[usize],
        vals: &[f64],
        rhs: f64,
        min_act: f64,
    ) -> bool {
        if min_act == -INF {
            return false;
        }
        let ip = self.infeasible_stack_pos();
        self.resolve_buffer.clear();
        for i in 0..inds.len() {
            let col = inds[i];
            let mut cand = ResolveCandidate {
                value_pos: i,
                ..Default::default()
            };
            if vals[i] > 0.0 {
                let (lb, bp) = self.localdom.col_lower_pos(col, ip);
                cand.bound_pos = bp;
                cand.base_bound = self.globaldom.col_lower[col];
                if cand.base_bound >= lb || bp == -1 {
                    continue;
                }
                cand.delta = vals[i] * (lb - cand.base_bound);
                cand.prio = compute_prio(
                    vals[i],
                    lb,
                    self.globaldom.col_lower[col],
                    self.nodes.num_nodes_up(col),
                );
            } else {
                let (ub, bp) = self.localdom.col_upper_pos(col, ip);
                cand.bound_pos = bp;
                cand.base_bound = self.globaldom.col_upper[col];
                if cand.base_bound <= ub || bp == -1 {
                    continue;
                }
                cand.delta = vals[i] * (ub - cand.base_bound);
                cand.prio = compute_prio(
                    vals[i],
                    ub,
                    self.globaldom.col_upper[col],
                    self.nodes.num_nodes_down(col),
                );
            }
            self.resolve_buffer.push(cand);
        }
        pdqsort(&mut self.resolve_buffer, candidate_less);
        let mlower = rhs + 10.0f64.max(rhs.abs()) * self.feastol();
        self.resolve_linear_leq(CDouble::from(min_act), mlower, vals)
    }

    /// `explainBoundChange(currentFrontier, domchg)`.
    fn explain_bound_change(
        &mut self,
        current_frontier: &Frontier,
        domchg: (i64, DomainChange),
    ) -> bool {
        let pos = domchg.0;
        match self.localdom.domchgreason[pos as usize] {
            Reason::Unspecified
            | Reason::Branching
            | Reason::ConflictingBounds(_)
            | Reason::RedCost => false,
            Reason::CliqueTable(col, val) => {
                // Implied by `col` fixed to `val`: that fixing explains it.
                self.resolved_domain_changes.clear();
                let bound_pos = if val != 0 {
                    self.localdom.col_lower_pos(col, pos).1
                } else {
                    self.localdom.col_upper_pos(col, pos).1
                };
                if bound_pos != -1 {
                    self.resolved_domain_changes
                        .push((bound_pos, self.localdom.domchgstack[bound_pos as usize]));
                }
                true
            }
            Reason::ModelRowLower(row) => {
                let (inds, vals) = self.model_row(row);
                let max_act = self.globaldom.max_activity(row);
                let rl = self.localdom.model.row_lower[row];
                self.explain_bound_change_geq(current_frontier, domchg, &inds, &vals, rl, max_act)
            }
            Reason::ModelRowUpper(row) => {
                let (inds, vals) = self.model_row(row);
                let min_act = self.globaldom.min_activity(row);
                let ru = self.localdom.model.row_upper[row];
                self.explain_bound_change_leq(current_frontier, domchg, &inds, &vals, ru, min_act)
            }
            Reason::Objective => {
                let (inds, vals, rhs) = self
                    .localdom
                    .objective_propagation_constraint(pos, Some(domchg.1.column));
                let (ninfmin, min_act) = self.globaldom.compute_min_activity_of(&inds, &vals);
                // HiGHS asserts at most one infinite contribution and gives up on
                // one ("todo: treat case with a single infinite contribution"); more
                // than one gives up too, rather than treat a partial sum as the
                // minimum activity.
                if ninfmin >= 1 {
                    return false;
                }
                self.explain_bound_change_leq(
                    current_frontier,
                    domchg,
                    &inds,
                    &vals,
                    rhs,
                    min_act.to_f64(),
                )
            }
            Reason::Cut(pool, cut) => {
                let (inds, vals, rhs) = self.cut_row(pool, cut);
                let min_act = self.globaldom.min_cut_activity(pool, cut);
                self.explain_bound_change_leq(current_frontier, domchg, &inds, &vals, rhs, min_act)
            }
            Reason::Conflict(pool, conflict) => match self.conflict_entries(pool, conflict) {
                None => false,
                Some(entries) => self.explain_bound_change_conflict(domchg, &entries),
            },
        }
    }

    /// `explainBoundChangeConflict(domchg, conflict, len)`.
    fn explain_bound_change_conflict(
        &mut self,
        locdomchg: (i64, DomainChange),
        conflict: &[DomainChange],
    ) -> bool {
        self.resolved_domain_changes.clear();
        let domchg = self.localdom.flip(&locdomchg.1);
        let mut found_domchg = false;
        for e in conflict {
            if !found_domchg && e.column == domchg.column && e.boundtype == domchg.boundtype {
                let covers = if e.boundtype == BoundType::Lower {
                    e.boundval <= domchg.boundval
                } else {
                    e.boundval >= domchg.boundval
                };
                if covers {
                    found_domchg = true;
                    continue;
                }
            }
            if self.globaldom.is_active(e) {
                continue;
            }
            let prev = &self.localdom.prevboundval;
            let mut pos;
            if e.boundtype == BoundType::Lower {
                let (lb, p) = self.localdom.col_lower_pos(e.column, locdomchg.0 - 1);
                pos = p;
                if pos == -1 || lb < e.boundval {
                    return false;
                }
                while prev[pos as usize].0 >= e.boundval {
                    pos = prev[pos as usize].1;
                    // See `explain_infeasibility_conflict`.
                    if pos == -1 {
                        return false;
                    }
                }
            } else {
                let (ub, p) = self.localdom.col_upper_pos(e.column, locdomchg.0 - 1);
                pos = p;
                if pos == -1 || ub > e.boundval {
                    return false;
                }
                while prev[pos as usize].0 <= e.boundval {
                    pos = prev[pos as usize].1;
                    if pos == -1 {
                        return false;
                    }
                }
            }
            self.resolved_domain_changes
                .push((pos, self.localdom.domchgstack[pos as usize]));
        }
        found_domchg
    }

    /// `resolvable(domChgPos)`: a change with a reason other than branching or
    /// unknown (reduced-cost fixing is HiGHS's `unspecified()`).
    fn resolvable(&self, pos: i64) -> bool {
        !matches!(
            self.localdom.domchgreason[pos as usize],
            Reason::Branching | Reason::Unspecified | Reason::RedCost
        )
    }

    /// `resolveDepth(frontier, depthLevel, stopSize, pseudocost, minResolve,
    /// increaseConflictScore)`: resolve the frontier's changes at depth
    /// `depth_level` (newest first) until at most `stop_size` resolvable ones are
    /// left; -1 when the depth has no frontier change.
    fn resolve_depth(
        &mut self,
        frontier: &mut Frontier,
        depth_level: usize,
        stop_size: usize,
        scores: &mut dyn ConflictScores,
        min_resolve: i64,
        increase_conflict_score: bool,
    ) -> i64 {
        self.resolve_queue.clear();
        let branch_pos = self.localdom.branch_pos.clone();
        let start_pos = if depth_level == 0 {
            0
        } else {
            branch_pos[depth_level - 1] as i64 + 1
        };
        let mut depth_level = depth_level;
        while depth_level < branch_pos.len() {
            let bp = branch_pos[depth_level];
            if self.localdom.domchgstack[bp].boundval != self.localdom.prevboundval[bp].0 {
                break;
            }
            // A redundant depth is skipped.
            depth_level += 1;
        }
        let in_range: Vec<i64> = if depth_level == branch_pos.len() {
            frontier.range(start_pos..).map(|(&p, _)| p).collect()
        } else {
            let end = branch_pos[depth_level] as i64;
            if start_pos > end {
                Vec::new()
            } else {
                frontier.range(start_pos..=end).map(|(&p, _)| p).collect()
            }
        };
        if in_range.is_empty() {
            return -1;
        }
        for p in in_range {
            if self.resolvable(p) {
                self.resolve_queue.push(p);
            }
        }
        let mut num_resolved = 0i64;
        while self.resolve_queue.len() > stop_size
            || (!self.resolve_queue.is_empty() && num_resolved < min_resolve)
        {
            let p = self.resolve_queue.pop().unwrap();
            let d = frontier[&p];
            if !self.explain_bound_change(frontier, (p, d)) {
                continue;
            }
            num_resolved += 1;
            frontier.remove(&p);
            let resolved = std::mem::take(&mut self.resolved_domain_changes);
            for &(ip, id) in &resolved {
                match frontier.entry(ip) {
                    std::collections::btree_map::Entry::Vacant(v) => {
                        v.insert(id);
                        if increase_conflict_score {
                            let s = self.localdom.domchgstack[ip as usize];
                            if s.boundtype == BoundType::Lower {
                                scores.increase_conflict_score_up(s.column);
                            } else {
                                scores.increase_conflict_score_down(s.column);
                            }
                        }
                        if ip >= start_pos && self.resolvable(ip) {
                            self.resolve_queue.push(ip);
                        }
                    }
                    std::collections::btree_map::Entry::Occupied(mut o) => {
                        let cur = o.get_mut();
                        if id.boundtype == BoundType::Lower {
                            cur.boundval = cur.boundval.max(id.boundval);
                        } else {
                            cur.boundval = cur.boundval.min(id.boundval);
                        }
                    }
                }
            }
            self.resolved_domain_changes = resolved;
        }
        num_resolved
    }

    /// `computeCuts(depthLevel, conflictPool, pseudocost)`: the conflict cut of a
    /// depth and, at a unique implication point other than the branching, its
    /// reconvergence cut; -1 for an empty depth.
    fn compute_cuts(
        &mut self,
        depth_level: usize,
        pool: &mut ConflictPool,
        scores: &mut dyn ConflictScores,
    ) -> i64 {
        let mut frontier = std::mem::take(&mut self.reason_side_frontier);
        let min_resolve = if depth_level == self.localdom.branch_pos.len() {
            1
        } else {
            0
        };
        let num_resolved =
            self.resolve_depth(&mut frontier, depth_level, 1, scores, min_resolve, true);
        self.reason_side_frontier = frontier;
        if num_resolved == -1 {
            return -1;
        }
        let mut num_conflicts = 0;
        if num_resolved > 0 {
            let entries = self
                .localdom
                .conflict_cut_entries(self.reason_side_frontier.values());
            self.add_to_pool(pool, &entries);
            num_conflicts += 1;
        }
        // A queue of one: a resolvable unique implication point other than the
        // branching.
        if self.resolve_queue.len() == 1 {
            let p = self.resolve_queue.pop().unwrap();
            let uip = (p, self.reason_side_frontier[&p]);
            self.resolve_queue.clear();
            let mut rf = Frontier::new();
            rf.insert(uip.0, uip.1);
            let nr = self.resolve_depth(&mut rf, depth_level, 0, scores, 0, false);
            let keep_uip = rf.contains_key(&uip.0);
            self.reconvergence_frontier = rf;
            if nr > 0 && !keep_uip {
                let entries = self
                    .localdom
                    .reconvergence_cut_entries(self.reconvergence_frontier.values(), &uip.1);
                self.add_to_pool(pool, &entries);
                num_conflicts += 1;
            }
        }
        num_conflicts
    }

    /// The depth loop shared by both `conflictAnalysis` overloads, after the
    /// infeasibility is explained into `resolved_domain_changes`.
    fn analyse_depths(&mut self, pool: &mut ConflictPool, scores: &mut dyn ConflictScores) {
        scores.increase_conflict_weight();
        for &(_, d) in &self.resolved_domain_changes {
            if d.boundtype == BoundType::Lower {
                scores.increase_conflict_score_up(d.column);
            } else {
                scores.increase_conflict_score_down(d.column);
            }
        }
        if 10 * self.resolved_domain_changes.len() > 1000 + 3 * self.localdom.num_integral_cols() {
            return;
        }
        for &(p, d) in &self.resolved_domain_changes {
            self.reason_side_frontier.entry(p).or_insert(d);
        }
        let mut num_conflicts = 0i64;
        let mut last_depth = self.localdom.branch_pos.len() as i64;
        let mut curr_depth = last_depth;
        while curr_depth >= 0 {
            if curr_depth > 0 {
                // Redundant branching changes (added for symmetry handling) are
                // skipped.
                let bp = self.localdom.branch_pos[curr_depth as usize - 1];
                if self.localdom.domchgstack[bp].boundval == self.localdom.prevboundval[bp].0 {
                    last_depth -= 1;
                    curr_depth -= 1;
                    continue;
                }
            }
            let num_new = self.compute_cuts(curr_depth as usize, pool, scores);
            if num_new == -1 {
                // An empty depth does not count.
                last_depth -= 1;
                curr_depth -= 1;
                continue;
            }
            num_conflicts += num_new;
            // Nothing in the first non-empty depth: stop.
            if num_conflicts == 0 {
                break;
            }
            // Nothing at this depth, and the first five non-empty depths done: stop.
            if last_depth - curr_depth >= 4 && num_new == 0 {
                break;
            }
            curr_depth -= 1;
        }
        // Stopped in the highest non-empty depth: no conflict yet. The frontier is
        // added anyway: the last branching itself led to the infeasibility and
        // should have been propagated one depth up (e.g. an integer column's
        // threshold was not reached).
        if curr_depth == last_depth {
            let entries = self
                .localdom
                .conflict_cut_entries(self.reason_side_frontier.values());
            self.add_to_pool(pool, &entries);
        }
    }

    /// `ConflictSet::conflictAnalysis(conflictPool, pseudocost)`.
    fn conflict_analysis(&mut self, pool: &mut ConflictPool, scores: &mut dyn ConflictScores) {
        if !self.explain_infeasibility() {
            return;
        }
        self.analyse_depths(pool, scores);
    }

    /// `ConflictSet::conflictAnalysis(proofinds, proofvals, prooflen, proofrhs,
    /// conflictPool, pseudocost)`.
    fn conflict_analysis_proof(
        &mut self,
        proofinds: &[usize],
        proofvals: &[f64],
        proofrhs: f64,
        pool: &mut ConflictPool,
        scores: &mut dyn ConflictScores,
    ) {
        let (ninfmin, activitymin) = self.globaldom.compute_min_activity_of(proofinds, proofvals);
        if ninfmin != 0 {
            return;
        }
        if !self.explain_infeasibility_leq(proofinds, proofvals, proofrhs, activitymin.to_f64()) {
            return;
        }
        self.analyse_depths(pool, scores);
    }
}

#[cfg(test)]
mod tests {
    use super::super::objective::ObjectiveFunction;
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

    /// Worst disagreement between each tracked, linked, live cut's incremental
    /// minimum activity and its recomputation, and whether the infinity counts match.
    fn check_cut_activities(d: &Domain) -> (f64, bool, usize) {
        let (mut worst, mut ok, mut n) = (0.0f64, true, 0usize);
        for cp in &d.cutpools {
            for cut in 0..cp.flags.len() {
                if cp.flags[cut] & 2 != 0 || !cp.linked[cut] {
                    continue;
                }
                let (ri, rv) = &cp.rows[cut];
                let (ni, a) = d.compute_min_activity_of(ri, rv);
                n += 1;
                ok &= ni == cp.activity_inf[cut];
                if ni == 0 {
                    worst = worst.max((a - cp.activity[cut]).to_f64().abs());
                }
            }
        }
        (worst, ok, n)
    }

    /// `debugCheckObjectiveLower`: |incremental - from scratch| and whether the
    /// infinity counts agree.
    fn check_objective_lower(d: &Domain) -> (f64, bool) {
        let op = d.obj_prop.as_ref().unwrap();
        let obj = &op.obj;
        let cost = &obj.cost;
        let nz = obj.nonzeros();
        let starts = obj.partition_starts();
        let np = obj.num_partitions();
        let mut lower = CDouble::from(0.0);
        let mut ninf = 0;
        for i in 0..np {
            let mut largest = 0.0f64;
            for &c in &nz[starts[i]..starts[i + 1]] {
                if cost[c] > 0.0 {
                    lower += cost[c];
                    if d.col_lower[c] < 1.0 {
                        largest = largest.max(cost[c]);
                    }
                } else if d.col_upper[c] > 0.0 {
                    largest = largest.max(-cost[c]);
                }
            }
            lower -= largest;
        }
        for &c in &nz[starts[np]..] {
            if cost[c] > 0.0 {
                if d.col_lower[c] > -INF {
                    lower += d.col_lower[c] * cost[c];
                } else {
                    ninf += 1;
                }
            } else if d.col_upper[c] < INF {
                lower += d.col_upper[c] * cost[c];
            } else {
                ninf += 1;
            }
        }
        (
            (lower - op.objective_lower).to_f64().abs(),
            ninf == op.num_inf_obj_lower,
        )
    }

    /// Random binaries with one clique row over a subset `S` (at most one literal
    /// `x_j = [cost_j < 0]` true), the clique partition `S` + singletons, and a
    /// random second row. Returns the model, costs, the partition and the points.
    #[allow(clippy::type_complexity)]
    fn random_clique_ip(
        r: &mut Lcg,
    ) -> (DomainModel, Vec<f64>, Vec<usize>, Vec<usize>, Vec<Vec<i64>>) {
        let n = r.int(3, 6) as usize;
        let cost: Vec<f64> = (0..n)
            .map(|_| {
                let c = r.int(-4, 4);
                (if c == 0 { 1 } else { c }) as f64
            })
            .collect();
        let ns = r.int(2, 3.min(n as i64)) as usize;
        let mut cols: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = r.int(0, i as i64) as usize;
            cols.swap(i, j);
        }
        let set = cols[..ns].to_vec();
        let (mut ri, mut rv) = (Vec::new(), Vec::new());
        let mut npos = 0;
        for &j in &set {
            ri.push(j);
            if cost[j] < 0.0 {
                rv.push(1.0);
            } else {
                rv.push(-1.0);
                npos += 1;
            }
        }
        let mut rows = vec![(ri, rv)];
        let (mut rl, mut ru) = (vec![-INF], vec![1.0 - npos as f64]);
        let (mut r2i, mut r2v) = (Vec::new(), Vec::new());
        for j in 0..n {
            let v = r.int(-2, 2);
            if v != 0 {
                r2i.push(j);
                r2v.push(v as f64);
            }
        }
        if !r2i.is_empty() {
            rows.push((r2i, r2v));
            rl.push(-INF);
            ru.push(r.int(0, 3) as f64);
        }
        let model = DomainModel::from_rows(n, &rows, rl.clone(), ru.clone(), vec![true; n]);
        let mut clqvars = set.clone();
        clqvars.extend(cols[ns..].iter().copied());
        let mut starts = vec![0, ns];
        for k in ns..n {
            starts.push(k + 1);
        }
        let mut pts = Vec::new();
        for mask in 0..(1u32 << n) {
            let x: Vec<i64> = (0..n).map(|j| ((mask >> j) & 1) as i64).collect();
            let ok = (0..rows.len()).all(|i| {
                let a: f64 = rows[i]
                    .0
                    .iter()
                    .zip(&rows[i].1)
                    .map(|(&j, &v)| v * x[j] as f64)
                    .sum();
                a >= rl[i] - 1e-9 && a <= ru[i] + 1e-9
            });
            if ok {
                pts.push(x);
            }
        }
        (model, cost, clqvars, starts, pts)
    }

    #[test]
    fn objective_propagation_never_removes_a_point_under_the_cutoff() {
        let mut r = Lcg(23);
        let (mut checked, mut obj_changes, mut obj_infeasible, mut with_partition) =
            (0usize, 0usize, 0usize, 0usize);
        for it in 0..3000 {
            let clique = it % 2 == 1;
            let (model, cost, lb, ub, pts, part) = if clique {
                let (m, c, clq, st, pts) = random_clique_ip(&mut r);
                let n = m.ncols;
                (m, c, vec![0.0; n], vec![1.0; n], pts, Some((clq, st)))
            } else {
                let (m, lb, ub, pts) = random_ip(&mut r);
                let c: Vec<f64> = (0..m.ncols).map(|_| r.int(-3, 3) as f64).collect();
                (m, c, lb, ub, pts, None)
            };
            if pts.is_empty() {
                continue;
            }
            let n = model.ncols;
            let objv = |p: &Vec<i64>| (0..n).map(|j| cost[j] * p[j] as f64).sum::<f64>();
            let best = pts.iter().map(objv).fold(INF, f64::min);
            let upper_limit = best + r.int(0, 3) as f64 + 0.5;
            let model = Arc::new(model);
            let mut obj = ObjectiveFunction::new(&cost, &model.is_int, &lb, &ub);
            if let Some((clq, st)) = &part {
                obj.setup_clique_partition(clq, st);
                if obj.num_partitions() > 0 {
                    with_partition += 1;
                }
            }
            let mut d = Domain::new(Arc::clone(&model), &lb, &ub, 1e-6, 1e-9);
            d.set_upper_limit(upper_limit);
            d.setup_objective_propagation(Arc::new(obj));
            let (w, ok) = check_objective_lower(&d);
            assert!(ok && w <= 1e-9, "initial objective lower wrong: {w}");
            let base = d.objective_lower().unwrap();
            let mut box_lb = lb.clone();
            let mut box_ub = ub.clone();
            for _ in 0..3 {
                let col = r.int(0, n as i64 - 1) as usize;
                if d.col_lower[col] >= d.col_upper[col] {
                    continue;
                }
                let up = r.int(0, 1) == 1;
                let val = r.int(d.col_lower[col] as i64, d.col_upper[col] as i64) as f64;
                let boundtype = if up {
                    box_lb[col] = box_lb[col].max(val);
                    BoundType::Lower
                } else {
                    box_ub[col] = box_ub[col].min(val);
                    BoundType::Upper
                };
                d.change_bound(
                    DomainChange {
                        boundval: val,
                        column: col,
                        boundtype,
                    },
                    Reason::Branching,
                );
                if r.int(0, 2) != 0 {
                    d.propagate();
                }
                obj_changes += d
                    .domchgreason
                    .iter()
                    .filter(|r| **r == Reason::Objective)
                    .count();
                let under: Vec<&Vec<i64>> = pts
                    .iter()
                    .filter(|p| {
                        objv(p) <= upper_limit
                            && (0..n).all(|j| p[j] as f64 >= box_lb[j] && p[j] as f64 <= box_ub[j])
                    })
                    .collect();
                if d.infeasible() {
                    if d.infeasible_reason() == Reason::Objective {
                        obj_infeasible += 1;
                    }
                    assert!(
                        under.is_empty(),
                        "infeasible verdict with points under the cutoff"
                    );
                    break;
                }
                let (w, ok) = check_objective_lower(&d);
                assert!(ok && w <= 1e-9, "objective lower drifted: {w}");
                for p in &under {
                    for j in 0..n {
                        checked += 1;
                        assert!(
                            p[j] as f64 >= d.col_lower[j] - 1e-9
                                && p[j] as f64 <= d.col_upper[j] + 1e-9,
                            "point under the cutoff removed at column {j}"
                        );
                    }
                }
            }
            d.backtrack_to_global();
            assert_eq!(d.col_lower, lb);
            assert_eq!(d.col_upper, ub);
            let (w, ok) = check_objective_lower(&d);
            assert!(ok && w <= 1e-9);
            let now = d.objective_lower().unwrap();
            assert!((now.0 - base.0).abs() <= 1e-9 && now.1 == base.1);
        }
        assert!(
            checked > 2000 && obj_changes > 200 && obj_infeasible > 20 && with_partition > 500,
            "probe fired too little: checked {checked} objective changes {obj_changes} \
             objective infeasible {obj_infeasible} with partition {with_partition}"
        );
    }

    #[test]
    fn objective_propagation_bounds_the_one_infinite_column() {
        // min x0 + x1, x0 in [0, 10], x1 in [-inf, 10]: cutoff 3 bounds nothing
        // (one infinite contribution, it is the column itself) until x1 gets a lower
        // bound; the cutoff then bounds both from above.
        let model = Arc::new(DomainModel::from_rows(
            2,
            &[],
            vec![],
            vec![],
            vec![false; 2],
        ));
        let (lb, ub) = ([0.0, -INF], [10.0, 10.0]);
        let obj = ObjectiveFunction::new(&[1.0, 1.0], &[false; 2], &lb, &ub);
        let mut d = Domain::new(Arc::clone(&model), &lb, &ub, 1e-6, 1e-9);
        d.set_upper_limit(3.0);
        d.setup_objective_propagation(Arc::new(obj));
        assert_eq!(d.objective_lower(), Some((0.0, 1)));
        // One infinite contribution: x1 (the infinite one) is bounded by 3 - 0.
        assert!(d.propagate());
        assert_eq!(d.col_upper[1], 3.0);
        d.change_bound(
            DomainChange {
                boundval: 1.0,
                column: 1,
                boundtype: BoundType::Lower,
            },
            Reason::Branching,
        );
        d.propagate();
        assert_eq!(d.objective_lower(), Some((1.0, 0)));
        assert_eq!(d.col_upper[0], 2.0);
        assert_eq!(d.domchgreason.last(), Some(&Reason::Objective));
    }

    #[test]
    fn cut_propagation_fixes_and_proves_infeasible_with_cut_reasons() {
        // No model rows: only the cut x0 + x1 <= 1 over binaries links them.
        let model = Arc::new(DomainModel::from_rows(
            2,
            &[],
            vec![],
            vec![],
            vec![true; 2],
        ));
        let mut d = Domain::new(Arc::clone(&model), &[0.0, 0.0], &[1.0, 1.0], 1e-6, 1e-9);
        let pool = d.add_cutpool();
        d.cut_added(pool, 0, &[0, 1], &[1.0, 1.0], 1.0, true);
        assert_eq!(d.min_cut_activity(pool, 0), 0.0);
        let one = |col| DomainChange {
            boundval: 1.0,
            column: col,
            boundtype: BoundType::Lower,
        };
        d.change_bound(one(0), Reason::Branching);
        assert!(d.propagate());
        assert_eq!(d.col_upper[1], 0.0);
        assert_eq!(d.domchgreason.last(), Some(&Reason::Cut(pool, 0)));
        assert_eq!(d.take_cut_age_resets(), vec![(pool, 0)]);
        assert_eq!(d.min_cut_activity(pool, 0), 1.0);
        d.backtrack_to_global();
        assert_eq!(d.col_upper, vec![1.0, 1.0]);
        assert_eq!(d.min_cut_activity(pool, 0), 0.0);
        // Both at one: the cut proves the node infeasible, and the activity change
        // that found it is rolled back.
        d.change_bound(one(0), Reason::Branching);
        d.change_bound(one(1), Reason::Branching);
        assert!(d.infeasible());
        assert_eq!(d.infeasible_reason(), Reason::Cut(pool, 0));
        assert_eq!(check_cut_activities(&d).1, true);
        d.backtrack_to_global();
        assert!(!d.infeasible());
        let (w, ok, n) = check_cut_activities(&d);
        assert!(ok && w == 0.0 && n == 1);
        // A deleted cut no longer propagates or reports an activity.
        d.cut_deleted(pool, 0, false);
        assert_eq!(d.min_cut_activity(pool, 0), -INF);
        d.change_bound(one(0), Reason::Branching);
        d.propagate();
        assert_eq!(d.col_upper[1], 1.0);
    }

    #[test]
    fn pool_notifies_its_propagation_domains() {
        use super::super::pool::{CutPool, DomainObservers, PoolObserver};
        let model = Arc::new(DomainModel::from_rows(
            3,
            &[],
            vec![],
            vec![],
            vec![true; 3],
        ));
        let mut global = Domain::new(Arc::clone(&model), &[0.0; 3], &[1.0; 3], 1e-6, 1e-9);
        global.set_global(true);
        let gp = global.add_cutpool();
        let mut local = global.clone();
        local.set_global(false);
        let mut pool = CutPool::new(3, 10, 100, 10, 5);
        let c0 = {
            let mut obs = DomainObservers(vec![(&mut global, gp), (&mut local, gp)]);
            pool.add_cut(&mut obs, &[2, 0], &[1.0, 1.0], 1.0, true, true, false)
        };
        assert!(c0 >= 0);
        let c0 = c0 as usize;
        // Both domains track it; the pool sorted the row.
        assert_eq!(global.min_cut_activity(gp, c0), 0.0);
        assert_eq!(local.min_cut_activity(gp, c0), 0.0);
        assert_eq!(local.cutpools[gp].rows[c0].0, vec![0, 2]);
        local.change_bound(
            DomainChange {
                boundval: 1.0,
                column: 2,
                boundtype: BoundType::Lower,
            },
            Reason::Branching,
        );
        local.propagate();
        assert_eq!(local.col_upper[0], 0.0);
        for (p, c) in local.take_cut_age_resets() {
            assert_eq!(p, gp);
            pool.reset_age(c as i32);
        }
        // Unlinking for propagation: the local domain drops the cut, the global
        // domain keeps its activity.
        {
            let mut obs = DomainObservers(vec![(&mut global, gp), (&mut local, gp)]);
            obs.cut_deleted(c0 as i32, true);
        }
        assert_eq!(local.min_cut_activity(gp, c0), -INF);
        assert_eq!(global.min_cut_activity(gp, c0), 0.0);
    }

    #[test]
    fn cut_propagation_never_removes_a_feasible_point_and_activities_stay_exact() {
        let mut r = Lcg(11);
        let (mut checked, mut cut_fixings, mut cut_infeasible, mut dives) = (0usize, 0, 0, 0);
        for _ in 0..4000 {
            let (model, lb, ub, pts) = random_ip(&mut r);
            if pts.is_empty() {
                continue;
            }
            let n = model.ncols;
            // Half the time the model rows are dropped, so the cuts alone carry the
            // structure (the points stay feasible for every valid cut).
            let model = if r.int(0, 1) == 0 {
                Arc::new(DomainModel::from_rows(
                    n,
                    &[],
                    vec![],
                    vec![],
                    vec![true; n],
                ))
            } else {
                Arc::new(model)
            };
            let mut d = Domain::new(Arc::clone(&model), &lb, &ub, 1e-6, 1e-9);
            d.set_global(r.int(0, 1) == 1);
            let pool = d.add_cutpool();
            // Valid cuts: random integer rows with the rhs at the feasible maximum.
            let ncuts = r.int(1, 4) as usize;
            for cut in 0..ncuts {
                let (mut ri, mut rv) = (Vec::new(), Vec::new());
                for j in 0..n {
                    let v = r.int(-3, 3);
                    if v != 0 {
                        ri.push(j);
                        rv.push(v as f64);
                    }
                }
                if ri.is_empty() {
                    ri.push(0);
                    rv.push(1.0);
                }
                let rhs = pts
                    .iter()
                    .map(|p| {
                        ri.iter()
                            .zip(&rv)
                            .map(|(&j, &v)| v * p[j] as f64)
                            .sum::<f64>()
                    })
                    .fold(-INF, f64::max);
                let propagate = r.int(0, 3) != 0;
                d.cut_added(pool, cut, &ri, &rv, rhs, propagate);
            }
            if r.int(0, 3) == 0 {
                let c = r.int(0, ncuts as i64 - 1) as usize;
                let only_prop = r.int(0, 1) == 1;
                d.cut_deleted(pool, c, only_prop);
            }
            d.propagate();
            assert!(
                !d.infeasible(),
                "valid cuts proved a feasible box infeasible"
            );
            d.commit_as_global();
            let (glb, gub) = (d.col_lower.clone(), d.col_upper.clone());
            dives += 1;
            // A dive of up to three branchings, propagating after each.
            let mut box_lb = glb.clone();
            let mut box_ub = gub.clone();
            for _ in 0..3 {
                let col = r.int(0, n as i64 - 1) as usize;
                if d.col_lower[col] >= d.col_upper[col] {
                    continue;
                }
                let up = r.int(0, 1) == 1;
                let val = r.int(d.col_lower[col] as i64, d.col_upper[col] as i64) as f64;
                let boundtype = if up {
                    box_lb[col] = box_lb[col].max(val);
                    BoundType::Lower
                } else {
                    box_ub[col] = box_ub[col].min(val);
                    BoundType::Upper
                };
                d.change_bound(
                    DomainChange {
                        boundval: val,
                        column: col,
                        boundtype,
                    },
                    Reason::Branching,
                );
                // Sometimes branch twice before propagating, so a bound change can
                // drive a cut's activity past its rhs directly.
                if r.int(0, 1) == 0 {
                    d.propagate();
                }
                cut_fixings += d
                    .domchgreason
                    .iter()
                    .filter(|r| matches!(r, Reason::Cut(..)))
                    .count();
                let in_box: Vec<&Vec<i64>> = pts
                    .iter()
                    .filter(|p| {
                        (0..n).all(|j| p[j] as f64 >= box_lb[j] && p[j] as f64 <= box_ub[j])
                    })
                    .collect();
                if d.infeasible() {
                    if matches!(d.infeasible_reason(), Reason::Cut(..)) {
                        cut_infeasible += 1;
                    }
                    assert!(in_box.is_empty(), "infeasible verdict with feasible points");
                    break;
                }
                let (w, ok, _) = check_cut_activities(&d);
                assert!(ok && w <= 1e-6, "cut activities drifted: {w} counts {ok}");
                for p in &in_box {
                    for j in 0..n {
                        checked += 1;
                        assert!(
                            p[j] as f64 >= d.col_lower[j] - 1e-9
                                && p[j] as f64 <= d.col_upper[j] + 1e-9,
                            "feasible point cut off at column {j}"
                        );
                    }
                }
            }
            // One-level backtrack, then to the global box: activities stay exact.
            d.backtrack();
            let (w, ok, _) = check_cut_activities(&d);
            assert!(ok && w <= 1e-6);
            d.backtrack_to_global();
            assert_eq!(d.col_lower, glb);
            assert_eq!(d.col_upper, gub);
            let (w, ok, _) = check_cut_activities(&d);
            assert!(ok && w <= 1e-9);
            d.propagate();
            assert!(!d.infeasible());
        }
        assert!(
            dives > 2000 && checked > 1000 && cut_fixings > 200 && cut_infeasible > 15,
            "probe fired too little: dives {dives} checked {checked} cut fixings \
             {cut_fixings} cut infeasible {cut_infeasible}"
        );
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
    fn reduced_stack_reinstalls_to_the_same_bounds() {
        // Branch down a random path with propagation, take HiGHS's reduced stack, and
        // install it on a fresh domain at the global bounds: the same box, and no more
        // branching positions, on every node of the path.
        let mut r = Lcg(11);
        let (mut checked, mut with_implied) = (0usize, 0usize);
        for _ in 0..300 {
            let (model, lb, ub, _) = random_ip(&mut r);
            let model = Arc::new(model);
            let mut d = Domain::new(model.clone(), &lb, &ub, 1e-6, 1e-9);
            d.propagate();
            if d.infeasible() {
                continue;
            }
            d.commit_as_global();
            let (glb, gub) = (d.col_lower.clone(), d.col_upper.clone());
            for _ in 0..4 {
                let free: Vec<usize> = (0..lb.len())
                    .filter(|&j| d.col_upper[j] > d.col_lower[j])
                    .collect();
                if free.is_empty() {
                    break;
                }
                let j = free[r.int(0, free.len() as i64 - 1) as usize];
                let up = r.int(0, 1) == 1;
                let chg = if up {
                    DomainChange {
                        boundval: d.col_lower[j] + 1.0,
                        column: j,
                        boundtype: BoundType::Lower,
                    }
                } else {
                    DomainChange {
                        boundval: d.col_upper[j] - 1.0,
                        column: j,
                        boundtype: BoundType::Upper,
                    }
                };
                d.change_bound(chg, Reason::Branching);
                d.propagate();
                if d.infeasible() {
                    break;
                }
                let (stack, pos) = d.reduced_domain_change_stack();
                if stack.len() > pos.len() {
                    with_implied += 1;
                }
                let mut e = Domain::new(model.clone(), &glb, &gub, 1e-6, 1e-9);
                e.set_domain_change_stack(&stack, &pos);
                assert!(!e.infeasible());
                assert_eq!(e.col_lower, d.col_lower, "lower bounds after reinstall");
                assert_eq!(e.col_upper, d.col_upper, "upper bounds after reinstall");
                // A branching change that earlier changes already imply is dropped on
                // reinstall (HiGHS's rule), so positions can only shrink.
                assert!(e.reduced_domain_change_stack().1.len() <= pos.len());
                checked += 1;
            }
        }
        // §6: the check ran, also on stacks carrying propagated (non-branching) changes.
        assert!(
            checked > 200 && with_implied > 50,
            "checked {checked}, with implied {with_implied}"
        );
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

    // ------------------------------------------------------------ conflicts

    use super::super::conflict_pool::{ConflictDomainObservers, ConflictPool};

    fn lit(col: usize, lower: bool, v: f64) -> DomainChange {
        DomainChange {
            boundval: v,
            column: col,
            boundtype: if lower {
                BoundType::Lower
            } else {
                BoundType::Upper
            },
        }
    }

    /// Counts the pseudocost conflict-score calls.
    #[derive(Default)]
    struct Scores {
        weight: usize,
        up: Vec<usize>,
        down: Vec<usize>,
    }
    impl ConflictScores for Scores {
        fn increase_conflict_weight(&mut self) {
            self.weight += 1;
        }
        fn increase_conflict_score_up(&mut self, col: usize) {
            self.up.push(col);
        }
        fn increase_conflict_score_down(&mut self, col: usize) {
            self.down.push(col);
        }
    }

    /// Open-node counts per column.
    struct Nodes(Vec<i64>, Vec<i64>);
    impl OpenNodeCounts for Nodes {
        fn num_nodes_up(&self, col: usize) -> i64 {
            self.0[col]
        }
        fn num_nodes_down(&self, col: usize) -> i64 {
            self.1[col]
        }
    }

    /// Does point `p` satisfy every literal of the conflict (so the conflict would
    /// cut it off)?
    fn satisfies_all(p: &[i64], conflict: &[DomainChange]) -> bool {
        conflict.iter().all(|e| match e.boundtype {
            BoundType::Lower => p[e.column] as f64 >= e.boundval - 1e-9,
            BoundType::Upper => p[e.column] as f64 <= e.boundval + 1e-9,
        })
    }

    /// The conflict-propagation bookkeeping is coherent: bit 4 exactly for queued
    /// conflicts (each queued once), bit 8 exactly for deleted ones, watched literals
    /// belong to their conflict, and (HiGHS's `linkWatchedLiteral` quirk) every watch
    /// list is empty. Returns the number of tracked conflicts checked.
    fn check_conflict_tracking(d: &Domain) -> usize {
        let mut n = 0;
        for cp in &d.conflictpools {
            assert!(cp.col_lower_watched.iter().all(|&h| h == -1));
            assert!(cp.col_upper_watched.iter().all(|&h| h == -1));
            let mut queued = vec![0usize; cp.conflict_flag.len()];
            for &c in &cp.propagate_conflict_inds {
                queued[c] += 1;
            }
            for c in 0..cp.conflict_flag.len() {
                n += 1;
                let f = cp.conflict_flag[c];
                assert!(queued[c] <= 1, "conflict {c} queued twice");
                assert_eq!(f & 4 != 0, queued[c] == 1, "queue bit of conflict {c}");
                // Deleted conflicts are dropped from the mirror (an index never
                // handed to this domain has neither).
                if f & 8 != 0 {
                    assert!(cp.conflicts[c].is_none(), "deleted conflict {c} kept");
                }
                if cp.conflicts[c].is_some() {
                    assert_eq!(f & 8, 0, "live conflict {c} flagged deleted");
                }
                for w in &cp.watched[2 * c..2 * c + 2] {
                    if w.column == -1 {
                        continue;
                    }
                    assert_eq!((w.prev, w.next), (-1, -1));
                    let entries = cp.conflicts[c].as_ref().expect("watched deleted conflict");
                    assert!(
                        entries.iter().any(|e| !w.differs(e)),
                        "watched literal not in conflict {c}"
                    );
                }
            }
        }
        n
    }

    #[test]
    fn conflict_propagates_a_fixing_proves_infeasibility_and_is_deleted() {
        let model = Arc::new(DomainModel::from_rows(
            3,
            &[],
            vec![],
            vec![],
            vec![true; 3],
        ));
        let mut d = Domain::new(Arc::clone(&model), &[0.0; 3], &[1.0; 3], 1e-6, 1e-9);
        let cp = d.add_conflict_pool();
        let mut pool = ConflictPool::new(10, 100);
        d.change_bound(lit(0, true, 1.0), Reason::Branching);
        // Not both x0 = 1 and x1 = 1: with x0 = 1 the conflict has one inactive
        // literal, is queued when added, and fixes x1 = 0.
        let c = pool.add_conflict_cut(
            &[lit(0, true, 1.0), lit(1, true, 1.0)],
            &mut ConflictDomainObservers(vec![(&mut d, cp)]),
        );
        assert_eq!(c, 0);
        assert!(d.propagate());
        assert_eq!(d.col_upper[1], 0.0);
        assert_eq!(d.domchgreason.last(), Some(&Reason::Conflict(cp, c)));
        assert_eq!(d.take_conflict_age_resets(), vec![(cp, c)]);
        check_conflict_tracking(&d);
        // Backtracking the fixing re-queues its conflict (`markPropagateCut`); with
        // both literals inactive it then watches them and stays quiet.
        d.backtrack_to_global();
        assert_eq!(d.conflictpools[cp].propagate_conflict_inds, vec![c]);
        d.propagate();
        assert_eq!(d.col_upper, vec![1.0; 3]);
        assert_eq!(d.conflictpools[cp].conflict_flag[c], 2);
        // HiGHS's watch lists stay empty (see `ConflictPoolProp`), so x0 = 1 alone no
        // longer wakes the conflict: x1 keeps its bound.
        d.change_bound(lit(0, true, 1.0), Reason::Branching);
        assert!(!d.propagate());
        assert_eq!(d.col_upper[1], 1.0);
        check_conflict_tracking(&d);
        d.backtrack_to_global();

        // Both literals already true when the conflict arrives: infeasible.
        let mut e = Domain::new(Arc::clone(&model), &[0.0; 3], &[1.0; 3], 1e-6, 1e-9);
        let ecp = e.add_conflict_pool();
        e.change_bound(lit(0, true, 1.0), Reason::Branching);
        e.change_bound(lit(2, false, 0.0), Reason::Branching);
        let c2 = pool.add_conflict_cut(
            &[lit(2, false, 0.0), lit(0, true, 1.0)],
            &mut ConflictDomainObservers(vec![(&mut d, cp), (&mut e, ecp)]),
        );
        assert_eq!(c2, 1);
        e.propagate();
        assert!(e.infeasible());
        assert_eq!(e.infeasible_reason(), Reason::Conflict(ecp, c2));
        e.backtrack_to_global();
        assert!(!e.infeasible());
        check_conflict_tracking(&e);

        // Deletion: the domains drop the conflict; a re-mark does not queue it.
        pool.remove_conflict(
            c,
            &mut ConflictDomainObservers(vec![(&mut d, cp), (&mut e, ecp)]),
        );
        assert!(d.conflictpools[cp].conflicts[c].is_none());
        assert_eq!(d.conflictpools[cp].conflict_flag[c] & 8, 8);
        d.mark_propagate_cut_reason(Reason::Conflict(cp, c));
        assert!(d.conflictpools[cp].propagate_conflict_inds.is_empty());
        check_conflict_tracking(&d);
        // The freed index is reused by the next conflict, which the domains track
        // afresh.
        let c3 = pool.add_conflict_cut(
            &[lit(1, false, 0.0)],
            &mut ConflictDomainObservers(vec![(&mut d, cp)]),
        );
        assert_eq!(c3, c);
        d.propagate();
        assert_eq!(d.col_lower[1], 1.0);
        check_conflict_tracking(&d);
    }

    /// Binaries x0, x1, x2 with x0 <= x2 and x0 + x1 + x2 <= 2, a global domain and a
    /// local copy, both propagating one conflict pool.
    fn implication_chain() -> (Domain, Domain, usize) {
        let rows = vec![
            (vec![0, 2], vec![1.0, -1.0]),
            (vec![0, 1, 2], vec![1.0, 1.0, 1.0]),
        ];
        let model =
            DomainModel::from_rows(3, &rows, vec![-INF, -INF], vec![0.0, 2.0], vec![true; 3]);
        let mut global = Domain::new(Arc::new(model), &[0.0; 3], &[1.0; 3], 1e-6, 1e-9);
        global.set_global(true);
        let cp = global.add_conflict_pool();
        let mut local = global.clone();
        local.set_global(false);
        (global, local, cp)
    }

    #[test]
    fn conflict_analysis_finds_the_conflict_of_an_implication() {
        let (mut global, mut local, cp) = implication_chain();
        let mut pool = ConflictPool::new(10, 100);
        let mut scores = Scores::default();
        let links = ConflictLinks {
            local: Some(cp),
            global: Some(cp),
        };
        // x1 = 1 propagates nothing; x0 = 1 implies x2 = 1 (row 0), and then row 1
        // is violated.
        local.change_bound(lit(1, true, 1.0), Reason::Branching);
        local.propagate();
        local.change_bound(lit(0, true, 1.0), Reason::Branching);
        local.propagate();
        assert!(local.infeasible());
        assert_eq!(local.domchgreason[2], Reason::ModelRowUpper(0));
        assert_eq!(local.infeasible_reason(), Reason::ModelRowUpper(1));
        let added =
            local.conflict_analysis(&mut pool, links, &mut global, &mut scores, &NoOpenNodes);
        // Row 1 is explained by x1 >= 1, x0 >= 1, x2 >= 1; at the last depth x2 >= 1
        // resolves to x0 >= 1: the conflict "not (x1 >= 1 and x0 >= 1)". The first
        // depth has nothing to resolve.
        assert_eq!(added, vec![0]);
        assert_eq!(
            pool.conflict(0).unwrap(),
            &[lit(1, true, 1.0), lit(0, true, 1.0)]
        );
        assert_eq!(scores.weight, 1);
        assert_eq!(scores.up, vec![1, 0, 2]);
        assert!(scores.down.is_empty());
        // Both domains were told; the local one, infeasible, queued it.
        assert!(global.conflictpools[cp].conflicts[0].is_some());
        assert_eq!(local.conflictpools[cp].propagate_conflict_inds, vec![0]);
        // Backtracking x0 = 1 leaves x1 = 1: the conflict fixes x0 = 0.
        local.backtrack();
        local.propagate();
        assert!(!local.infeasible());
        assert_eq!(local.col_upper[0], 0.0);
        assert_eq!(local.domchgreason.last(), Some(&Reason::Conflict(cp, 0)));
        check_conflict_tracking(&local);

        // The proof-row overload on a feasible domain: 2 x0 + x1 <= 2 is valid (x0 = 1
        // forces x1 = 0) and violated by x1 = 1, x0 = 1. Nothing at the last depth
        // resolves, so the frontier itself is the conflict.
        let (mut global, mut local, cp) = implication_chain();
        let mut pool = ConflictPool::new(10, 100);
        local.change_bound(lit(1, true, 1.0), Reason::Branching);
        local.change_bound(lit(0, true, 1.0), Reason::Branching);
        assert!(!local.infeasible());
        let added = local.conflict_analysis_proof(
            &[0, 1],
            &[2.0, 1.0],
            2.0,
            &mut pool,
            ConflictLinks {
                local: Some(cp),
                global: Some(cp),
            },
            &mut global,
            &mut Scores::default(),
            &NoOpenNodes,
        );
        assert_eq!(added, vec![0]);
        assert_eq!(
            pool.conflict(0).unwrap(),
            &[lit(1, true, 1.0), lit(0, true, 1.0)]
        );
        // Not infeasible yet: the queued conflict proves it on propagation.
        local.propagate();
        assert!(local.infeasible());
        assert_eq!(local.infeasible_reason(), Reason::Conflict(cp, 0));

        // Reconvergence: x1 <= 0 is implied by the proof row given x0 = 1 (and so
        // x2 = 1); the cut is "not (x1 >= 1 and x0 >= 1)".
        let (mut global, mut local, cp) = implication_chain();
        let mut pool = ConflictPool::new(10, 100);
        local.change_bound(lit(0, true, 1.0), Reason::Branching);
        local.propagate();
        let added = local.conflict_analyze_reconvergence(
            &lit(1, false, 0.0),
            &[0, 1, 2],
            &[1.0, 1.0, 1.0],
            2.0,
            &mut pool,
            ConflictLinks {
                local: Some(cp),
                global: None,
            },
            &mut global,
            &mut Scores::default(),
            &NoOpenNodes,
        );
        assert_eq!(added, vec![0]);
        assert_eq!(
            pool.conflict(0).unwrap(),
            &[lit(1, true, 1.0), lit(0, true, 1.0)]
        );
        assert!(global.conflictpools[cp].conflicts.is_empty());
    }

    /// Random valid rows for an IP: integer coefficients, rhs at the feasible
    /// maximum (so every feasible point satisfies them).
    fn random_valid_rows(
        r: &mut Lcg,
        n: usize,
        pts: &[Vec<i64>],
        k: usize,
    ) -> Vec<(Vec<usize>, Vec<f64>, f64)> {
        let mut out = Vec::new();
        for _ in 0..k {
            let (mut ri, mut rv) = (Vec::new(), Vec::new());
            for j in 0..n {
                let v = r.int(-3, 3);
                if v != 0 {
                    ri.push(j);
                    rv.push(v as f64);
                }
            }
            if ri.is_empty() {
                ri.push(0);
                rv.push(1.0);
            }
            let rhs = pts
                .iter()
                .map(|p| {
                    ri.iter()
                        .zip(&rv)
                        .map(|(&j, &v)| v * p[j] as f64)
                        .sum::<f64>()
                })
                .fold(-INF, f64::max);
            out.push((ri, rv, rhs));
        }
        out
    }

    fn check_conflict_valid(pts: &[Vec<i64>], entries: &[DomainChange]) -> usize {
        for p in pts {
            assert!(
                !satisfies_all(p, entries),
                "conflict {entries:?} cuts off feasible point {p:?}"
            );
        }
        pts.len()
    }

    #[test]
    fn conflict_analysis_and_propagation_never_cut_a_feasible_point() {
        let mut r = Lcg(31);
        let (mut instances, mut analyses, mut proof_analyses) = (0usize, 0usize, 0usize);
        let (mut conflicts, mut conflict_fixings, mut conflict_infeasible) = (0usize, 0, 0);
        let (mut points_checked, mut deleted, mut global_fixings, mut tracked) =
            (0usize, 0usize, 0usize, 0usize);
        // Infeasibility reasons analysed: model row, cut, conflict, objective,
        // conflicting bounds.
        let mut reasons = [0usize; 5];
        let mut with_objective = 0usize;
        for _ in 0..3000 {
            let (model, lb, ub, pts) = random_ip(&mut r);
            if pts.is_empty() {
                continue;
            }
            instances += 1;
            let n = model.ncols;
            let model = Arc::new(model);
            let mut global = Domain::new(Arc::clone(&model), &lb, &ub, 1e-6, 1e-9);
            global.set_global(true);
            // Half the instances carry an objective cutoff: the points that count
            // are then those under it (objective propagation removes the others).
            let pts = if r.int(0, 1) == 1 {
                let cost: Vec<f64> = (0..n).map(|_| r.int(-3, 3) as f64).collect();
                let objv = |p: &Vec<i64>| (0..n).map(|j| cost[j] * p[j] as f64).sum::<f64>();
                let best = pts.iter().map(objv).fold(INF, f64::min);
                let upper_limit = best + r.int(0, 2) as f64 + 0.5;
                let obj = ObjectiveFunction::new(&cost, &model.is_int, &lb, &ub);
                global.set_upper_limit(upper_limit);
                global.setup_objective_propagation(Arc::new(obj));
                with_objective += 1;
                pts.iter()
                    .filter(|p| objv(p) <= upper_limit)
                    .cloned()
                    .collect()
            } else {
                pts
            };
            // Valid cuts in a cut pool both domains propagate (cut reasons), and
            // valid rows kept back as LP-style proofs.
            let gp = global.add_cutpool();
            let ncuts = r.int(0, 3) as usize;
            for (c, (ri, rv, rhs)) in random_valid_rows(&mut r, n, &pts, ncuts)
                .into_iter()
                .enumerate()
            {
                global.cut_added(gp, c, &ri, &rv, rhs, true);
            }
            let proofs = random_valid_rows(&mut r, n, &pts, 6);
            let gcp = global.add_conflict_pool();
            global.propagate();
            assert!(
                !global.infeasible(),
                "valid rows proved a feasible model infeasible"
            );
            let nodes = Nodes(
                (0..n).map(|_| r.int(0, 3)).collect(),
                (0..n).map(|_| r.int(0, 3)).collect(),
            );
            // Small age limits, so conflicts are deleted while stacks still cite them.
            let mut pool = ConflictPool::new(r.int(1, 4) as i32, r.int(2, 20) as i32);
            let mut scores = Scores::default();
            let links = ConflictLinks {
                local: Some(gcp),
                global: Some(gcp),
            };
            for _dive in 0..8 {
                let mut local = global.clone();
                local.set_global(false);
                // The box of the current node, per branching depth.
                let mut boxes: Vec<(Vec<f64>, Vec<f64>)> =
                    vec![(global.col_lower.clone(), global.col_upper.clone())];
                let mut seen_on_stack = local.domchgreason.len();
                for _step in 0..12 {
                    if local.infeasible() {
                        let (blb, bub) = boxes.last().unwrap();
                        assert!(
                            !pts.iter()
                                .any(|p| (0..n)
                                    .all(|j| p[j] as f64 >= blb[j] && p[j] as f64 <= bub[j])),
                            "infeasible verdict with a feasible point in the box"
                        );
                        reasons[match local.infeasible_reason() {
                            Reason::ModelRowLower(_) | Reason::ModelRowUpper(_) => 0,
                            Reason::Cut(..) => 1,
                            Reason::Conflict(..) => 2,
                            Reason::Objective => 3,
                            Reason::ConflictingBounds(_) => 4,
                            other => panic!("unexpected infeasibility reason {other:?}"),
                        }] += 1;
                        if matches!(local.infeasible_reason(), Reason::Conflict(..)) {
                            conflict_infeasible += 1;
                        }
                        analyses += 1;
                        let added = local.conflict_analysis(
                            &mut pool,
                            links,
                            &mut global,
                            &mut scores,
                            &nodes,
                        );
                        for &c in &added {
                            conflicts += 1;
                            points_checked += check_conflict_valid(&pts, pool.conflict(c).unwrap());
                        }
                        for (_, c) in local
                            .take_conflict_age_resets()
                            .into_iter()
                            .chain(global.take_conflict_age_resets())
                        {
                            pool.reset_age(c);
                        }
                        if local.backtrack().is_none() {
                            break;
                        }
                        boxes.pop();
                        seen_on_stack = seen_on_stack.min(local.domchgreason.len());
                        local.propagate();
                    } else {
                        let free: Vec<usize> = (0..n)
                            .filter(|&j| local.col_lower[j] < local.col_upper[j])
                            .collect();
                        if free.is_empty() {
                            break;
                        }
                        let j = free[r.int(0, free.len() as i64 - 1) as usize];
                        let up = r.int(0, 1) == 1;
                        let mut v =
                            r.int(local.col_lower[j] as i64, local.col_upper[j] as i64) as f64;
                        // Now and then a bound past the other one (conflicting bounds).
                        if r.int(0, 11) == 0 {
                            v = if up {
                                local.col_upper[j] + 1.0
                            } else {
                                local.col_lower[j] - 1.0
                            };
                        }
                        let (mut blb, mut bub) = boxes.last().unwrap().clone();
                        let chg = if up {
                            blb[j] = blb[j].max(v);
                            lit(j, true, v)
                        } else {
                            bub[j] = bub[j].min(v);
                            lit(j, false, v)
                        };
                        boxes.push((blb, bub));
                        local.change_bound(chg, Reason::Branching);
                        // A proof row the local bounds violate is explained (as the
                        // dual ray of an infeasible LP would be).
                        if !local.infeasible() {
                            for (pi, pv, prhs) in &proofs {
                                let (ninf, act) = local.compute_min_activity_of(pi, pv);
                                if ninf == 0 && act.to_f64() > prhs + 1e-6 {
                                    proof_analyses += 1;
                                    let added = local.conflict_analysis_proof(
                                        pi,
                                        pv,
                                        *prhs,
                                        &mut pool,
                                        links,
                                        &mut global,
                                        &mut scores,
                                        &nodes,
                                    );
                                    for &c in &added {
                                        conflicts += 1;
                                        points_checked +=
                                            check_conflict_valid(&pts, pool.conflict(c).unwrap());
                                    }
                                    break;
                                }
                            }
                        }
                        if r.int(0, 3) != 0 {
                            local.propagate();
                        }
                    }
                    // The conflict fixings new on the stack.
                    seen_on_stack = seen_on_stack.min(local.domchgreason.len());
                    conflict_fixings += local.domchgreason[seen_on_stack..]
                        .iter()
                        .filter(|r| matches!(r, Reason::Conflict(..)))
                        .count();
                    seen_on_stack = local.domchgreason.len();
                    tracked += check_conflict_tracking(&local);
                    if !local.infeasible() {
                        // No feasible point of the node's box is removed, and the
                        // row and cut activities stay exact.
                        let (blb, bub) = boxes.last().unwrap();
                        for p in &pts {
                            if (0..n).all(|j| p[j] as f64 >= blb[j] && p[j] as f64 <= bub[j]) {
                                for j in 0..n {
                                    points_checked += 1;
                                    assert!(
                                        p[j] as f64 >= local.col_lower[j] - 1e-9
                                            && p[j] as f64 <= local.col_upper[j] + 1e-9,
                                        "propagation removed feasible point {p:?}"
                                    );
                                }
                            }
                        }
                        let (w, ok) = local.check_activities();
                        assert!(ok && w <= 1e-6, "row activities drifted");
                        let (w, ok, _) = check_cut_activities(&local);
                        assert!(ok && w <= 1e-6, "cut activities drifted");
                    }
                }
                // Global propagation of the conflicts keeps every feasible point.
                let before = global.domchgreason.len();
                global.propagate();
                assert!(
                    !global.infeasible(),
                    "conflicts proved a feasible model infeasible"
                );
                global_fixings += global.domchgreason[before..]
                    .iter()
                    .filter(|r| matches!(r, Reason::Conflict(..)))
                    .count();
                for p in &pts {
                    for j in 0..n {
                        points_checked += 1;
                        assert!(
                            p[j] as f64 >= global.col_lower[j] - 1e-9
                                && p[j] as f64 <= global.col_upper[j] + 1e-9,
                            "global conflict propagation removed feasible point {p:?}"
                        );
                    }
                }
                tracked += check_conflict_tracking(&global);
                let live = pool.num_conflicts();
                pool.perform_aging(
                    false,
                    &mut ConflictDomainObservers(vec![(&mut global, gcp)]),
                );
                deleted += live - pool.num_conflicts();
            }
        }
        eprintln!(
            "instances {instances} analyses {analyses} proof analyses {proof_analyses} \
             conflicts {conflicts} conflict fixings {conflict_fixings} conflict \
             infeasible {conflict_infeasible} global conflict fixings {global_fixings} \
             deleted {deleted} points checked {points_checked} tracked {tracked} \
             infeasibility reasons (row, cut, conflict, objective, conflicting bounds) \
             {reasons:?} with objective {with_objective}"
        );
        assert!(
            instances > 500
                && analyses > 1000
                && proof_analyses > 100
                && conflicts > 1000
                && conflict_fixings > 200
                && conflict_infeasible > 20
                && global_fixings > 50
                && deleted > 200
                && with_objective > 500
                && reasons.iter().all(|&k| k > 20)
                && points_checked > 100_000,
            "probe fired too little"
        );
    }
}
