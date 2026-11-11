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
//! Not yet HiGHS's (later stages): the estimate is the LP objective until the
//! pseudocost port (stage 5); no symmetry, no conflict analysis (stage 7), no basic
//! degenerate duals (stage 2).

// Wired into the driver by stage 1d-5 (the search loop switch); until then only the
// stage 1d-2 differential gate (`diff`, tests) runs it.
#![cfg_attr(not(test), allow(dead_code))]

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
    /// Whether the last LP was unbounded.
    pub unbounded: bool,
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
            unbounded: false,
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
            if self.tighten_cutoff && v < self.cutoff {
                self.cutoff = v;
            }
            self.found.push((x, v));
        }
    }

    /// The post-LP part of `evaluateNodeImpl` for one LP outcome: updates `node` and
    /// returns the result, or `None` when reduced-cost fixing and propagation changed
    /// columns and the node must be evaluated again.
    pub fn classify(
        &mut self,
        ctx: &NodeCtx<'_>,
        nl: NodeLp,
        node: &mut NodeData,
        full_l: &[f64],
        full_u: &[f64],
        redcost: bool,
    ) -> Option<SearchResult> {
        let oc = ctx.obj_const;
        self.branch_hint = None;
        self.unbounded = false;
        match nl.verdict {
            NodeLpVerdict::InfeasibleProved => {
                self.stats.lp_infeasible += 1;
                Some(SearchResult::LpInfeasible)
            }
            NodeLpVerdict::Unbounded => {
                self.unbounded = true;
                self.point = None;
                self.stats.open += 1;
                Some(SearchResult::Open)
            }
            // No trusted bound or point (HiGHS: not `scaledOptimal`): the node stays
            // open on the bound it inherited; the safety layer's incumbent candidate,
            // if any, is still offered.
            NodeLpVerdict::Untrusted | NodeLpVerdict::UnverifiedPoint => {
                self.point = None;
                self.stats.untrusted += 1;
                Some(SearchResult::Open)
            }
            NodeLpVerdict::Abstain { incumbent } => {
                self.offer(incumbent);
                self.point = None;
                self.stats.untrusted += 1;
                Some(SearchResult::Open)
            }
            NodeLpVerdict::SnapRefused {
                bound,
                col,
                incumbent,
            } => {
                self.offer(incumbent);
                self.basis = Some(nl.sol.basis.clone());
                node.lp_objective = nl.sol.obj + oc;
                node.estimate = node.lp_objective;
                node.lower_bound = node.lower_bound.max(bound + oc);
                self.point = Some(nl.sol.x[..ctx.ns].to_vec());
                self.branch_hint = Some(col);
                self.stats.open += 1;
                Some(SearchResult::Open)
            }
            NodeLpVerdict::Bound { bound, integral } => {
                self.basis = Some(nl.sol.basis.clone());
                node.lp_objective = nl.sol.obj + oc;
                node.estimate = node.lp_objective;
                self.point = Some(nl.sol.x[..ctx.ns].to_vec());
                if integral {
                    // `addIncumbent`; the LP is dual feasible, so the node is done.
                    self.offer(Some((nl.sol.x[..ctx.ns].to_vec(), nl.sol.obj + oc)));
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
        inheuristic: bool,
        optimality_limit: f64,
    ) -> SearchResult {
        let oc = ctx.obj_const;
        let mut first = true;
        loop {
            if !first && !inheuristic && node.lower_bound > optimality_limit {
                return SearchResult::Open;
            }
            first = false;
            self.stats.evaluations += 1;
            self.dom.set_upper_limit(self.cutoff - oc);
            self.dom.propagate();
            if self.dom.infeasible() {
                self.dom.clear_changed_cols();
                self.stats.domain_infeasible += 1;
                return SearchResult::DomainInfeasible;
            }
            // `lp->flushDomain(localdom)`.
            self.dom.clear_changed_cols();
            let (fl, fu) = self.lp_box(ctx);
            let nl = node_lp(ctx, self.basis.clone(), &fl, &fu);
            if let Some((v, 0)) = self.dom.objective_lower() {
                node.lower_bound = node.lower_bound.max(v + oc);
            }
            if let Some(r) = self.classify(ctx, nl, node, &fl, &fu, true) {
                return r;
            }
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
            .classify(ctx, nl, &mut node, &full_l, &full_u, false)
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
            e.evaluate(ctx, &mut node, false, f64::INFINITY)
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
