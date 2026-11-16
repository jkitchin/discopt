//! Provenance: the control flow of `HighsSearch::evaluateNodeImpl` from HiGHS
//! (https://github.com/ERGO-Code/HiGHS, Copyright (c) HiGHS, MIT License), translated
//! into Rust and wired to discopt's components, for deep integration with discopt's
//! LP, safety layer and certificate machinery and a pure-Rust build. See
//! `THIRD_PARTY_NOTICES.md` at the repository root for the license text.
//!
//! Stage 1d-2 of `docs/dev/highs-mip-replacement-plan.md`: the component part of
//! node evaluation behind `SearchEnv::evaluate`. The node's local domain
//! ([`crate::lp::highs_cutgen::domain::Domain`]) is propagated, its box is flushed to
//! the working LP, and the LP is solved through [`super::node_lp`] -- the same
//! safety layer `solve_node` uses (NS safe bound, Farkas verification, point and
//! snap verification), so the new search trusts exactly the LP outcomes the batched
//! search trusts. An optimal bounded node then gets reduced-cost fixing against the
//! cutoff and propagation, and is re-evaluated when that changed columns (HiGHS's
//! `return evaluateNode()`).
//!
//! Pseudocost observations (inference, cutoff, objective change) go through
//! [`super::highs_branch::BranchPseudocost`]. Not yet HiGHS's (later stages): the
//! estimate is the LP objective until `computeBestEstimate` lands with the pseudocost
//! port (stage 5); no symmetry, no conflict analysis (stage 7), no basic degenerate
//! duals (stage 2).

// Wired into the driver by stage 1d-5 (the search loop switch); until then only the
// stage 1d-2 differential gate (`diff`, tests) runs it.
#![cfg_attr(not(test), allow(dead_code))]

use super::highs_branch::BranchPseudocost;
use super::*;
use crate::bnb::highs_search::search::{NodeData, NodeResult as SearchResult};
use crate::lp::highs_cutgen::domain::{BoundType, Domain, DomainChange, Reason};

/// Counters of [`HighsEval`]'s outcomes (the probe of its tests).
#[derive(Clone, Debug, Default)]
pub(super) struct EvalStats {
    pub evaluations: usize,
    pub domain_infeasible: usize,
    pub lp_infeasible: usize,
    pub bound_exceeding: usize,
    pub integral: usize,
    pub untrusted: usize,
    pub open: usize,
    pub redcost_fixings: usize,
    pub reevaluations: usize,
}

/// The search's node evaluation over the local domain.
pub(super) struct HighsEval {
    /// The local domain (its global baseline is the working LP's propagated box).
    pub dom: Domain,
    /// Row -> singleton slack `(column, coefficient)` of the working LP.
    slacks: Vec<Option<(usize, f64)>>,
    /// The LP's current basis (HiGHS's LP keeps its basis across evaluations); `None`
    /// solves as the root.
    pub basis: Option<Basis>,
    /// The cutoff in objective units (with `obj_const`): a node whose bound exceeds
    /// it is pruned (`getCutoffBound`), and reduced-cost fixing and objective
    /// propagation work against it.
    pub cutoff: f64,
    /// Verified incumbent candidates found by evaluation: `(x over structurals, value)`.
    pub found: Vec<(Vec<f64>, f64)>,
    /// When the LP vertex's integral realisation missed a row (#2163): the column to
    /// branch on.
    pub branch_hint: Option<usize>,
    /// The last trusted LP point (structurals), for branching.
    pub point: Option<Vec<f64>>,
    /// The same point over all working columns (slacks included), for heuristics.
    pub point_full: Option<Vec<f64>>,
    /// Whether the last LP was unbounded.
    pub unbounded: bool,
    /// Whether any LP of this search was unbounded (sticky): the search then gives
    /// the problem back to the driver, whose own handling decides Unbounded.
    pub unbounded_seen: bool,
    /// Reduced costs of the structurals at the last trusted LP (`col_dual`).
    pub redcost: Option<Vec<f64>>,
    /// Unscaled row duals of the last trusted LP (`row_dual`).
    pub last_dual: Option<Vec<f64>>,
    /// Bumped each time an evaluation stores fresh duals (`last_dual`): the search
    /// LP's cut-row ageing runs only on a dual-feasible solve, as HiGHS's
    /// `performAging` returns early without valid duals.
    pub dual_epoch: u64,
    /// Reduced costs of all working columns at [`Self::basis`] (for refitting it to
    /// another node's box).
    pub basis_d: Option<Vec<f64>>,
    /// The last trusted LP's duals as the solve returned them (solve space), and its
    /// working-LP box: what node separation re-solves from.
    pub last_dual_raw: Option<Vec<f64>>,
    pub last_box: Option<(Vec<f64>, Vec<f64>)>,
    /// A bound from node separation for the node evaluated next (the same node: the
    /// dive's first evaluation after `separateAndStoreBasis`), with `obj_const`.
    pub pending_lift: Option<f64>,
    /// LP iterations: all node and strong-branching solves, and the strong-branching
    /// share (`lpiterations`, `sblpiterations`); heuristics' share (`heurlpiterations`).
    pub lp_iters: i64,
    pub sb_iters: i64,
    pub heur_iters: i64,
    /// The root LP solution (`getRootLpSol`), set by the search after the root.
    pub root_sol: Vec<f64>,
    /// `subrootsol`: the LP solution at the subtree root (cleared at `installNode`).
    pub subrootsol: Vec<f64>,
    /// `reliableatnode`: per column, `1` up reliable, `2` down reliable; HiGHS clears
    /// it when a dive starts (`diveImpl`), which the search does.
    pub reliable_at_node: std::collections::HashMap<usize, u8>,
    /// Nodes that could be neither branched nor resolved: the smallest bound among
    /// them (the certified dual bound must stay at or below it; HiGHS declares such
    /// a node infeasible, discopt keeps its valid bound).
    pub unresolved_floor: f64,
    /// Whether incumbents found tighten [`Self::cutoff`] (HiGHS: `addIncumbent`
    /// lowers the upper limit). Off for a fixed external cutoff.
    pub tighten_cutoff: bool,
    pub stats: EvalStats,
}

impl HighsEval {
    /// A local domain over the working LP (`None` when its rows are infeasible over
    /// the global box).
    pub fn new(ctx: &NodeCtx<'_>) -> Option<Self> {
        let (dom, slacks) = domain_from_lp(
            ctx.csc_rc,
            ctx.m_w,
            ctx.n_w,
            ctx.ns,
            ctx.b_w,
            ctx.l_w,
            ctx.u_w,
            ctx.is_int_full,
        )?;
        Some(HighsEval {
            dom,
            slacks,
            basis: None,
            cutoff: f64::INFINITY,
            found: Vec::new(),
            branch_hint: None,
            point: None,
            point_full: None,
            unbounded: false,
            unbounded_seen: false,
            redcost: None,
            last_dual: None,
            dual_epoch: 0,
            basis_d: None,
            last_dual_raw: None,
            last_box: None,
            pending_lift: None,
            lp_iters: 0,
            sb_iters: 0,
            heur_iters: 0,
            root_sol: Vec::new(),
            subrootsol: Vec::new(),
            reliable_at_node: std::collections::HashMap::new(),
            unresolved_floor: f64::INFINITY,
            tighten_cutoff: true,
            stats: EvalStats::default(),
        })
    }

    /// Install a box on the local domain as branching changes from its global
    /// baseline (without propagating); whether the domain stayed feasible.
    pub fn install_box(&mut self, lb: &[f64], ub: &[f64]) -> bool {
        self.dom.backtrack_to_global();
        let ns = lb.len();
        for j in 0..ns {
            let tl = if lb[j] <= -INF {
                f64::NEG_INFINITY
            } else {
                lb[j]
            };
            let tu = if ub[j] >= INF { f64::INFINITY } else { ub[j] };
            if tl > self.dom.col_lower[j] && !self.dom.infeasible() {
                self.dom.change_bound(
                    DomainChange {
                        boundval: tl,
                        column: j,
                        boundtype: BoundType::Lower,
                    },
                    Reason::Branching,
                );
            }
            if tu < self.dom.col_upper[j] && !self.dom.infeasible() {
                self.dom.change_bound(
                    DomainChange {
                        boundval: tu,
                        column: j,
                        boundtype: BoundType::Upper,
                    },
                    Reason::Branching,
                );
            }
        }
        !self.dom.infeasible()
    }

    /// The working-LP box of the local domain.
    pub fn lp_box(&self, ctx: &NodeCtx<'_>) -> (Vec<f64>, Vec<f64>) {
        domain_lp_box(
            &self.dom,
            &self.slacks,
            &ctx.b_w[..ctx.m_w],
            &ctx.l_w[..ctx.n_w],
            &ctx.u_w[..ctx.n_w],
            ctx.ns,
        )
    }

    fn offer(&mut self, inc: Option<(Vec<f64>, f64)>) {
        if let Some((x, v)) = inc {
            self.offer_found(x, v);
        }
    }

    /// `addIncumbent` for a verified point: record it and lower the cutoff.
    pub fn offer_found(&mut self, x: Vec<f64>, v: f64) {
        if self.tighten_cutoff && v < self.cutoff {
            self.cutoff = v;
        }
        self.found.push((x, v));
    }

    /// The post-LP part of `evaluateNodeImpl` for one LP outcome: updates `node` and
    /// returns the result, or `None` when reduced-cost fixing and propagation changed
    /// columns and the node must be evaluated again.
    #[allow(clippy::too_many_arguments)]
    pub fn classify(
        &mut self,
        ctx: &NodeCtx<'_>,
        nl: NodeLp,
        node: &mut NodeData,
        parent: Option<&NodeData>,
        pc: &mut dyn BranchPseudocost,
        full_l: &[f64],
        full_u: &[f64],
        redcost: bool,
    ) -> Option<SearchResult> {
        let oc = ctx.obj_const;
        self.branch_hint = None;
        self.unbounded = false;
        self.lp_iters += nl.sol.iters as i64;
        match nl.verdict {
            NodeLpVerdict::InfeasibleProved => {
                self.stats.lp_infeasible += 1;
                cutoff_observation(parent, pc);
                Some(SearchResult::LpInfeasible)
            }
            NodeLpVerdict::Unbounded => {
                self.unbounded = true;
                self.unbounded_seen = true;
                self.point = None;
                self.point_full = None;
                self.stats.open += 1;
                Some(SearchResult::Open)
            }
            // No trusted bound or point (HiGHS: not `scaledOptimal`): the node stays
            // open on the bound it inherited; the safety layer's incumbent candidate,
            // if any, is still offered.
            NodeLpVerdict::Untrusted | NodeLpVerdict::UnverifiedPoint => {
                self.point = None;
                self.point_full = None;
                self.stats.untrusted += 1;
                Some(SearchResult::Open)
            }
            NodeLpVerdict::Abstain { incumbent } => {
                self.offer(incumbent);
                self.point = None;
                self.point_full = None;
                self.stats.untrusted += 1;
                Some(SearchResult::Open)
            }
            NodeLpVerdict::SnapRefused {
                bound,
                col,
                ref incumbent,
            } => {
                self.offer(incumbent.clone());
                self.basis = Some(nl.sol.basis.clone());
                node.lp_objective = nl.sol.obj + oc;
                node.estimate = node.lp_objective;
                self.redcost = Some(structural_redcost(ctx, &nl));
                self.last_dual = Some(nl.rc_dual().to_vec());
                self.dual_epoch += 1;
                self.basis_d = Some(full_redcost(ctx, &nl));
                self.last_dual_raw = Some(nl.sol.dual.clone());
                self.last_box = Some((full_l.to_vec(), full_u.to_vec()));
                objective_observation(parent, node, pc);
                node.lower_bound = node.lower_bound.max(bound + oc);
                self.point = Some(nl.sol.x[..ctx.ns].to_vec());
                self.point_full = Some(nl.sol.x.clone());
                self.branch_hint = Some(col);
                self.stats.open += 1;
                Some(SearchResult::Open)
            }
            NodeLpVerdict::Bound { bound, integral } => {
                self.basis = Some(nl.sol.basis.clone());
                node.lp_objective = nl.sol.obj + oc;
                node.estimate = node.lp_objective;
                self.redcost = Some(structural_redcost(ctx, &nl));
                self.last_dual = Some(nl.rc_dual().to_vec());
                self.dual_epoch += 1;
                self.basis_d = Some(full_redcost(ctx, &nl));
                self.last_dual_raw = Some(nl.sol.dual.clone());
                self.last_box = Some((full_l.to_vec(), full_u.to_vec()));
                objective_observation(parent, node, pc);
                self.point = Some(nl.sol.x[..ctx.ns].to_vec());
                self.point_full = Some(nl.sol.x.clone());
                if integral {
                    // `addIncumbent`; the LP is dual feasible, so the node is done. The
                    // point offered is the vertex's integral realisation, which the
                    // safety layer verified on the original rows and bounds (#2163),
                    // valued exactly (`c . x`, #2136).
                    let (x, v) = integral_incumbent(ctx, &nl.sol.x);
                    self.offer(Some((x, v)));
                    node.lower_bound = node.lower_bound.max(bound + oc);
                    self.stats.integral += 1;
                    self.stats.bound_exceeding += 1;
                    return Some(SearchResult::BoundExceeding);
                }
                node.lower_bound = node.lower_bound.max(bound + oc);
                if node.lower_bound > self.cutoff {
                    self.stats.bound_exceeding += 1;
                    return Some(SearchResult::BoundExceeding);
                }
                if redcost && self.cutoff.is_finite() {
                    // `HighsRedcostFixing::propagateRedCost` (discopt's fixing, whose
                    // safety margin only loosens), then `localdom.propagate()`.
                    if let Some((rl, ru)) = reduced_cost_fix(
                        ctx.csc_rc,
                        ctx.m_w,
                        ctx.c_w,
                        &nl.sol.basis,
                        nl.rc_dual(),
                        bound + oc,
                        self.cutoff,
                        full_l,
                        full_u,
                        ctx.ns,
                        ctx.is_int_full,
                        ctx.opts.simplex.tol,
                    ) {
                        for j in 0..ctx.ns {
                            if self.dom.infeasible() {
                                break;
                            }
                            if rl[j] > full_l[j] && rl[j] > self.dom.col_lower[j] {
                                self.stats.redcost_fixings += 1;
                                self.dom.change_bound(
                                    DomainChange {
                                        boundval: rl[j],
                                        column: j,
                                        boundtype: BoundType::Lower,
                                    },
                                    Reason::RedCost,
                                );
                            }
                            if ru[j] < full_u[j]
                                && ru[j] < self.dom.col_upper[j]
                                && !self.dom.infeasible()
                            {
                                self.stats.redcost_fixings += 1;
                                self.dom.change_bound(
                                    DomainChange {
                                        boundval: ru[j],
                                        column: j,
                                        boundtype: BoundType::Upper,
                                    },
                                    Reason::RedCost,
                                );
                            }
                        }
                    }
                    if !self.dom.infeasible() {
                        self.dom.propagate();
                    }
                    if self.dom.infeasible() {
                        self.dom.clear_changed_cols();
                        self.stats.domain_infeasible += 1;
                        cutoff_observation(parent, pc);
                        return Some(SearchResult::DomainInfeasible);
                    }
                    if !self.dom.changed_cols().is_empty() {
                        self.stats.reevaluations += 1;
                        return None;
                    }
                }
                self.stats.open += 1;
                Some(SearchResult::Open)
            }
        }
    }

    /// `evaluateNodeImpl` (the component part; the control's early suboptimality
    /// test, tree weight and dump events stay in `Search::evaluate_node`). A
    /// re-evaluation that finds the node above the optimality limit returns `Open`:
    /// the control's own test then makes it suboptimal, as HiGHS's inner call does.
    pub fn evaluate(
        &mut self,
        ctx: &NodeCtx<'_>,
        node: &mut NodeData,
        parent: Option<&NodeData>,
        pc: &mut dyn BranchPseudocost,
        inheuristic: bool,
        optimality_limit: f64,
    ) -> SearchResult {
        let oc = ctx.obj_const;
        if let Some(lift) = self.pending_lift.take() {
            node.lower_bound = node.lower_bound.max(lift);
        }
        let mut first = true;
        loop {
            if !first && !inheuristic && node.lower_bound > optimality_limit {
                return SearchResult::Open;
            }
            first = false;
            self.stats.evaluations += 1;
            self.dom.set_upper_limit(self.cutoff - oc);
            self.dom.propagate();
            if let Some(p) = parent {
                // `addInferenceObservation`: the changes since the node's branching.
                if p.decision.col >= 0 {
                    let inferences = self.dom.domain_change_stack().len() as i64
                        - (node.domchg_stack_pos as i64 + 1);
                    pc.add_inference_observation(
                        p.decision.col as usize,
                        inferences,
                        !p.decision.upper,
                    );
                }
            }
            if self.dom.infeasible() {
                self.dom.clear_changed_cols();
                self.stats.domain_infeasible += 1;
                cutoff_observation(parent, pc);
                return SearchResult::DomainInfeasible;
            }
            // `lp->flushDomain(localdom)`.
            self.dom.clear_changed_cols();
            let (fl, fu) = self.lp_box(ctx);
            let nl = node_lp(
                ctx,
                fit_basis_to_box(self.basis.clone(), self.basis_d.as_deref(), &fl, &fu),
                &fl,
                &fu,
            );
            if let Some((v, 0)) = self.dom.objective_lower() {
                node.lower_bound = node.lower_bound.max(v + oc);
            }
            if let Some(r) = self.classify(ctx, nl, node, parent, pc, &fl, &fu, true) {
                return r;
            }
        }
    }
}

/// A start basis for the LP over the box `l..u`, from a basis kept from another node
/// and its reduced costs `d` (all working columns): every nonbasic column with both
/// sides finite goes to the side its reduced cost makes dual feasible (`d > 0` lower,
/// `d < 0` upper), and one on an infinite side moves to its finite side -- what
/// HiGHS's dual simplex does with a kept basis when bounds change (bound flipping).
/// A basis carried across nodes otherwise starts dual infeasible wherever a column
/// fixed at the earlier node (`l = u`, any status) is boxed again, and the solve
/// repairs that in dual phase 1. Only the start changes: the LP solved, and its
/// safety layer, are the same.
pub(super) fn fit_basis_to_box(
    basis: Option<Basis>,
    d: Option<&[f64]>,
    l: &[f64],
    u: &[f64],
) -> Option<Basis> {
    let mut b = basis?;
    let mut moved = 0u64;
    for (j, st) in b.col_status.iter_mut().enumerate() {
        if j >= l.len() || *st == BASIC {
            continue;
        }
        let (lo, hi) = (l[j] > -INF, u[j] < INF);
        let want = match (lo, hi) {
            (true, true) => match d.and_then(|d| d.get(j)) {
                Some(&dj) if dj > 0.0 => AT_LOWER,
                Some(&dj) if dj < 0.0 => AT_UPPER,
                _ => *st,
            },
            (true, false) => AT_LOWER,
            (false, true) => AT_UPPER,
            (false, false) => *st,
        };
        if want != *st && (*st == AT_LOWER || *st == AT_UPPER) {
            *st = want;
            moved += 1;
        }
    }
    crate::profile::incr_by(crate::profile::Ctr::HighsMipSearchBasisFits, moved);
    Some(b)
}

/// The incumbent an integral vertex offers: its integral realisation (which the
/// safety layer verified on the original rows and bounds, #2163) over the
/// structurals, valued exactly as `c . x + obj_const` (#2136).
pub(super) fn integral_incumbent(ctx: &NodeCtx<'_>, x_full: &[f64]) -> (Vec<f64>, f64) {
    let x = snap_integral(ctx, x_full).unwrap_or_else(|| x_full.to_vec());
    let v = x.iter().zip(ctx.c_w).map(|(xj, cj)| xj * cj).sum::<f64>() + ctx.obj_const;
    (x[..ctx.ns].to_vec(), v)
}

/// Reduced costs `c_j - a_j^T y` of every working column at an LP.
fn full_redcost(ctx: &NodeCtx<'_>, nl: &NodeLp) -> Vec<f64> {
    let y = nl.rc_dual();
    (0..ctx.n_w)
        .map(|j| ctx.c_w[j] - ctx.csc_rc.dot(j, y))
        .collect()
}

/// The structurals' reduced costs `c_j - a_j^T y` at an LP (`col_dual`).
fn structural_redcost(ctx: &NodeCtx<'_>, nl: &NodeLp) -> Vec<f64> {
    let y = nl.rc_dual();
    (0..ctx.ns)
        .map(|j| ctx.c_w[j] - ctx.csc_rc.dot(j, y))
        .collect()
}

/// `pseudocost.addCutoffObservation` for the parent's branching, when the parent had
/// an LP and the branching was not a fallback (`branching_point != boundval`).
fn cutoff_observation(parent: Option<&NodeData>, pc: &mut dyn BranchPseudocost) {
    if let Some(p) = parent {
        if p.decision.col >= 0
            && p.lp_objective != f64::NEG_INFINITY
            && p.branching_point != p.decision.val
        {
            pc.add_cutoff_observation(p.decision.col as usize, !p.decision.upper);
        }
    }
}

/// `pseudocost.addObservation(col, boundval - branching_point, max(0, lp - parent lp))`.
fn objective_observation(
    parent: Option<&NodeData>,
    node: &NodeData,
    pc: &mut dyn BranchPseudocost,
) {
    if let Some(p) = parent {
        if p.decision.col >= 0
            && p.lp_objective != f64::NEG_INFINITY
            && p.branching_point != p.decision.val
        {
            let delta = p.decision.val - p.branching_point;
            let objdelta = (node.lp_objective - p.lp_objective).max(0.0);
            pc.add_observation(p.decision.col as usize, delta, objdelta);
        }
    }
}

/// The stage 1d-2 gate: on every node the batched search evaluates, run the new
/// evaluation too and compare.
///
/// * **Same box**: the node LP over exactly `solve_node`'s box and start basis,
///   classified by [`HighsEval::classify`] (no reduced-cost fixing, no cutoff), must
///   give `solve_node`'s verdict and bound bit for bit. Only taken when the node's box
///   was not changed by `solve_node`'s own propagation.
/// * **Full evaluation**: install the node's box on the local domain and run
///   [`HighsEval::evaluate`] (propagation, LP, reduced-cost fixing against a cutoff
///   just above a reference optimum `x*`, re-evaluation). When `x*` lies in the box,
///   the evaluation must not prune it (no infeasible or bound-exceeding verdict
///   unless it found a point at least as good), its bound must not exceed `obj*`,
///   and the local domain must still contain `x*`.
#[cfg(test)]
pub(super) mod diff {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default, Debug)]
    pub struct DiffStats {
        pub nodes: usize,
        pub same_box: usize,
        pub same_box_mismatch: Vec<String>,
        pub full: usize,
        pub ref_in_box: usize,
        pub unsound: Vec<String>,
        /// Full evaluations that pruned a node `solve_node` left open.
        pub stronger_prunes: usize,
        pub eval: EvalStats,
    }

    pub struct EvalDiff {
        /// `(x*, obj*)` over the structurals, objective with `obj_const`.
        pub reference: Option<(Vec<f64>, f64)>,
        pub stats: DiffStats,
    }

    thread_local! {
        pub static DIFF: RefCell<Option<EvalDiff>> = const { RefCell::new(None) };
    }

    fn same_box(
        ctx: &NodeCtx<'_>,
        id: NodeId,
        lb: &[f64],
        ub: &[f64],
        out: &NodeOutput,
        st: &mut DiffStats,
    ) {
        let mut full_l = lb.to_vec();
        let mut full_u = ub.to_vec();
        full_l.extend_from_slice(ctx.slack_l);
        full_u.extend_from_slice(ctx.slack_u);
        let nl = node_lp(ctx, ctx.tm.node_basis(id), &full_l, &full_u);
        let Some(mut e) = HighsEval::new(ctx) else {
            return;
        };
        e.tighten_cutoff = false;
        let mut node = NodeData::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        let r = e
            .classify(
                ctx,
                nl,
                &mut node,
                None,
                &mut super::super::highs_branch::NoPseudocost,
                &full_l,
                &full_u,
                false,
            )
            .expect("no re-evaluation without reduced-cost fixing");
        st.same_box += 1;
        let res = &out.result;
        let ok = match r {
            SearchResult::LpInfeasible => {
                res.lower_bound == INFEAS_SENTINEL && res.sentinel_is_exclusion
            }
            SearchResult::BoundExceeding => res.is_feasible && out.lp_bound == node.lower_bound,
            SearchResult::Open if e.unbounded => out.unbounded,
            SearchResult::Open if e.point.is_none() => {
                res.lower_bound == f64::NEG_INFINITY && !res.is_feasible
            }
            // `solve_node` may lift its bound past the LP's (node GMI), never below.
            SearchResult::Open => {
                !res.is_feasible
                    && out.lp_bound == node.lower_bound
                    && res.lower_bound >= out.lp_bound
                    && (e.branch_hint.is_none() || out.branch_hint == e.branch_hint)
            }
            other => {
                st.same_box_mismatch.push(format!("unexpected {other:?}"));
                return;
            }
        };
        if !ok {
            st.same_box_mismatch.push(format!(
                "node {id:?}: new {r:?} bound {} point {} | solve_node lp bound {} bound {} feasible {} unbounded {}",
                node.lower_bound,
                e.point.is_some(),
                out.lp_bound,
                res.lower_bound,
                res.is_feasible,
                out.unbounded
            ) + &format!(
                " excl {} iters {} has_basis {} dpre {}",
                res.sentinel_is_exclusion,
                out.iters,
                ctx.tm.node_basis(id).is_some(),
                ctx.domain_pre.is_some()
            ));
        }
    }

    fn full(
        ctx: &NodeCtx<'_>,
        id: NodeId,
        lb: &[f64],
        ub: &[f64],
        out: &NodeOutput,
        reference: &(Vec<f64>, f64),
        st: &mut DiffStats,
    ) {
        let Some(mut e) = HighsEval::new(ctx) else {
            return;
        };
        let (xr, objr) = reference;
        let tol = 1e-6 * (1.0 + objr.abs());
        e.cutoff = objr + tol;
        e.tighten_cutoff = false;
        e.basis = ctx.tm.node_basis(id);
        let in_box = (0..ctx.ns).all(|j| xr[j] >= lb[j] - 1e-9 && xr[j] <= ub[j] + 1e-9);
        let mut node = NodeData::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        let r = if e.install_box(lb, ub) {
            e.evaluate(
                ctx,
                &mut node,
                None,
                &mut super::super::highs_branch::NoPseudocost,
                false,
                f64::INFINITY,
            )
        } else {
            SearchResult::DomainInfeasible
        };
        st.full += 1;
        let pruned = matches!(
            r,
            SearchResult::DomainInfeasible
                | SearchResult::LpInfeasible
                | SearchResult::BoundExceeding
        );
        let old_pruned = out.result.lower_bound == INFEAS_SENTINEL || out.result.is_feasible;
        if pruned && !old_pruned {
            st.stronger_prunes += 1;
        }
        let ev = &e.stats;
        st.eval.evaluations += ev.evaluations;
        st.eval.domain_infeasible += ev.domain_infeasible;
        st.eval.lp_infeasible += ev.lp_infeasible;
        st.eval.bound_exceeding += ev.bound_exceeding;
        st.eval.integral += ev.integral;
        st.eval.untrusted += ev.untrusted;
        st.eval.open += ev.open;
        st.eval.redcost_fixings += ev.redcost_fixings;
        st.eval.reevaluations += ev.reevaluations;
        if !in_box {
            return;
        }
        st.ref_in_box += 1;
        let found_as_good = e.found.iter().any(|(_, v)| *v <= objr + tol);
        if pruned && !(r == SearchResult::BoundExceeding && found_as_good) {
            st.unsound.push(format!(
                "node {id:?}: {r:?} prunes a box holding x* (bound {})",
                node.lower_bound
            ));
        }
        if node.lower_bound > objr + tol {
            st.unsound.push(format!(
                "node {id:?}: bound {} above obj* {objr}",
                node.lower_bound
            ));
        }
        if !pruned
            && !(0..ctx.ns)
                .all(|j| xr[j] >= e.dom.col_lower[j] - 1e-6 && xr[j] <= e.dom.col_upper[j] + 1e-6)
        {
            st.unsound
                .push(format!("node {id:?}: the local domain lost x*"));
        }
    }

    /// Run both comparisons on a batch (no-op unless a test armed [`DIFF`]).
    pub fn batch(
        ctx: &NodeCtx<'_>,
        ids: &[NodeId],
        lbs: &[Vec<f64>],
        ubs: &[Vec<f64>],
        outs: &[NodeOutput],
    ) {
        // The top-level solve only (a sub-MIP's batches are not the problem's).
        if MILP_DEPTH.with(|d| d.get()) != 1 {
            return;
        }
        DIFF.with(|d| {
            let mut d = d.borrow_mut();
            let Some(diff) = d.as_mut() else { return };
            for k in 0..ids.len() {
                let out = &outs[k];
                if out.deferred {
                    continue;
                }
                diff.stats.nodes += 1;
                // Only when `solve_node` itself did not propagate: its FBBT can tighten
                // slack bounds without reporting it, and an FBBT-infeasible node
                // reports no tightening either.
                if !ctx.opts.node_propagation && ctx.domain_pre.is_none() {
                    same_box(ctx, ids[k], &lbs[k], &ubs[k], out, &mut diff.stats);
                }
                if let Some(reference) = diff.reference.clone() {
                    full(
                        ctx,
                        ids[k],
                        &lbs[k],
                        &ubs[k],
                        out,
                        &reference,
                        &mut diff.stats,
                    );
                }
            }
        });
    }
}
