//! Provenance: the control flow of `HighsSearch::branchImpl` and
//! `HighsSearch::selectBranchingCandidate`, and `HighsLpRelaxation::computeLPDegneracy`,
//! from HiGHS (https://github.com/ERGO-Code/HiGHS, Copyright (c) HiGHS, MIT License),
//! translated into Rust and wired to discopt's components, for deep integration with
//! discopt's LP, safety layer and certificate machinery and a pure-Rust build with no
//! C/C++ compiler to ship. See `THIRD_PARTY_NOTICES.md` at the repository root.
//!
//! Stage 1d-3 of `docs/dev/highs-mip-replacement-plan.md`: the component part of
//! `branch` behind `SearchEnv::branch`. Candidates are the fractional integer columns
//! of the node's last trusted LP point; reliability strong branching solves each
//! child LP on the local domain through [`super::node_lp`] (the safety layer), so a
//! child is only pruned on a verified Farkas ray or a safe bound, and an integral
//! child vertex becomes an incumbent only once verified. Commitments to one child
//! (the other side infeasible or bound-exceeding) and the re-evaluations they cause
//! are reported to the control as [`BranchStep`]s; the final branching changes the
//! local domain and is reported in [`BranchOutcome::branched`].
//!
//! Deviations from HiGHS, all on the side of safety or recorded:
//! * Child bounds from strong branching (`upbound`/`downbound`) and every prune
//!   decision use the NS safe bound, not the vertex objective (`checkSol`); the
//!   pseudocost observation uses the vertex objective, as HiGHS does.
//! * Fractional columns are taken in column order (HiGHS iterates a hash table).
//! * A node that can be neither branched nor resolved (all integer columns fixed, LP
//!   unresolved) is not declared infeasible, as HiGHS does after its fresh-LP
//!   retries: its valid bound goes into [`HighsEval::unresolved_floor`].
//! * Conflict analysis (stage 7) and symmetry are not yet called.

// Wired into the driver by stage 1d-5; until then only tests run it.
#![cfg_attr(not(test), allow(dead_code))]

use super::highs_eval::HighsEval;
use super::*;
use crate::bnb::highs_search::node_queue::DomainChange as QDomainChange;
use crate::bnb::highs_search::search::{
    BranchOutcome, BranchStep, NodeData, NodeResult as SearchResult,
};
use crate::lp::highs_cutgen::domain::{BoundType, DomainChange, Reason};

/// The `HighsPseudocost` interface branching and node evaluation use (implemented
/// by the stage-5 translation; [`NoPseudocost`] for callers without one).
pub(super) trait BranchPseudocost {
    /// `getMinReliable`.
    fn min_reliable(&self) -> i64;
    /// `setMinReliable`.
    fn set_min_reliable(&mut self, v: i64);
    /// `setDegeneracyFactor`.
    fn set_degeneracy_factor(&mut self, f: f64);
    /// `isReliable`.
    fn is_reliable(&self, col: usize) -> bool;
    /// `getPseudocostUp(col, frac)`.
    fn pseudocost_up(&self, col: usize, frac: f64) -> f64;
    /// `getPseudocostDown(col, frac)`.
    fn pseudocost_down(&self, col: usize, frac: f64) -> f64;
    /// `getPseudocostUp(col, frac, offset)`.
    fn pseudocost_up_offset(&self, col: usize, frac: f64, offset: f64) -> f64;
    /// `getPseudocostDown(col, frac, offset)`.
    fn pseudocost_down_offset(&self, col: usize, frac: f64, offset: f64) -> f64;
    /// `getScore(col, upcost, downcost)`.
    fn score_costs(&self, col: usize, up: f64, down: f64) -> f64;
    /// `getScore(col, frac)`.
    fn score_frac(&self, col: usize, frac: f64) -> f64;
    /// `addObservation`.
    fn add_observation(&mut self, col: usize, delta: f64, objdelta: f64);
    /// `addInferenceObservation`.
    fn add_inference_observation(&mut self, col: usize, inferences: i64, upbranch: bool);
    /// `addCutoffObservation`.
    fn add_cutoff_observation(&mut self, col: usize, upbranch: bool);
    /// `getScoreUp(col, frac)`.
    fn score_up(&self, col: usize, frac: f64) -> f64;
    /// `getScoreDown(col, frac)`.
    fn score_down(&self, col: usize, frac: f64) -> f64;
    /// `getAvgInferencesUp`.
    fn avg_inferences_up(&self, col: usize) -> f64;
    /// `getAvgInferencesDown`.
    fn avg_inferences_down(&self, col: usize) -> f64;
}

/// A pseudocost that records nothing and has no opinion (every column reliable,
/// every score zero): branching on it picks by queue balance and column order.
pub(super) struct NoPseudocost;

impl BranchPseudocost for NoPseudocost {
    fn min_reliable(&self) -> i64 {
        0
    }
    fn set_min_reliable(&mut self, _v: i64) {}
    fn set_degeneracy_factor(&mut self, _f: f64) {}
    fn is_reliable(&self, _col: usize) -> bool {
        true
    }
    fn pseudocost_up(&self, _col: usize, _frac: f64) -> f64 {
        0.0
    }
    fn pseudocost_down(&self, _col: usize, _frac: f64) -> f64 {
        0.0
    }
    fn pseudocost_up_offset(&self, _col: usize, _frac: f64, _offset: f64) -> f64 {
        0.0
    }
    fn pseudocost_down_offset(&self, _col: usize, _frac: f64, _offset: f64) -> f64 {
        0.0
    }
    fn score_costs(&self, _col: usize, _up: f64, _down: f64) -> f64 {
        0.0
    }
    fn score_frac(&self, _col: usize, _frac: f64) -> f64 {
        0.0
    }
    fn add_observation(&mut self, _col: usize, _delta: f64, _objdelta: f64) {}
    fn add_inference_observation(&mut self, _col: usize, _inferences: i64, _upbranch: bool) {}
    fn add_cutoff_observation(&mut self, _col: usize, _upbranch: bool) {}
    fn score_up(&self, _col: usize, _frac: f64) -> f64 {
        0.0
    }
    fn score_down(&self, _col: usize, _frac: f64) -> f64 {
        0.0
    }
    fn avg_inferences_up(&self, _col: usize) -> f64 {
        0.0
    }
    fn avg_inferences_down(&self, _col: usize) -> f64 {
        0.0
    }
}

impl BranchPseudocost for crate::bnb::highs_search::pseudocost::Pseudocost {
    fn min_reliable(&self) -> i64 {
        self.get_min_reliable() as i64
    }
    fn set_min_reliable(&mut self, v: i64) {
        crate::bnb::highs_search::pseudocost::Pseudocost::set_min_reliable(self, v as i32);
    }
    fn set_degeneracy_factor(&mut self, f: f64) {
        crate::bnb::highs_search::pseudocost::Pseudocost::set_degeneracy_factor(self, f);
    }
    fn is_reliable(&self, col: usize) -> bool {
        crate::bnb::highs_search::pseudocost::Pseudocost::is_reliable(self, col)
    }
    fn pseudocost_up(&self, col: usize, frac: f64) -> f64 {
        self.get_pseudocost_up(col, frac)
    }
    fn pseudocost_down(&self, col: usize, frac: f64) -> f64 {
        self.get_pseudocost_down(col, frac)
    }
    fn pseudocost_up_offset(&self, col: usize, frac: f64, offset: f64) -> f64 {
        self.get_pseudocost_up_offset(col, frac, offset)
    }
    fn pseudocost_down_offset(&self, col: usize, frac: f64, offset: f64) -> f64 {
        self.get_pseudocost_down_offset(col, frac, offset)
    }
    fn score_costs(&self, col: usize, up: f64, down: f64) -> f64 {
        self.get_score_costs(col, up, down)
    }
    fn score_frac(&self, col: usize, frac: f64) -> f64 {
        self.get_score(col, frac)
    }
    fn score_up(&self, col: usize, frac: f64) -> f64 {
        self.get_score_up(col, frac)
    }
    fn score_down(&self, col: usize, frac: f64) -> f64 {
        self.get_score_down(col, frac)
    }
    fn add_observation(&mut self, col: usize, delta: f64, objdelta: f64) {
        crate::bnb::highs_search::pseudocost::Pseudocost::add_observation(
            self, col, delta, objdelta,
        );
    }
    fn add_inference_observation(&mut self, col: usize, inferences: i64, upbranch: bool) {
        // `HighsInt` in HiGHS.
        let n = inferences.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        crate::bnb::highs_search::pseudocost::Pseudocost::add_inference_observation(
            self, col, n, upbranch,
        );
    }
    fn add_cutoff_observation(&mut self, col: usize, upbranch: bool) {
        crate::bnb::highs_search::pseudocost::Pseudocost::add_cutoff_observation(
            self, col, upbranch,
        );
    }
    fn avg_inferences_up(&self, col: usize) -> f64 {
        self.get_avg_inferences_up(col)
    }
    fn avg_inferences_down(&self, col: usize) -> f64 {
        self.get_avg_inferences_down(col)
    }
}

/// `HighsSearch::ChildSelectionRule` (the rules `branchImpl` implements; HiGHS uses
/// `RootSol`, and `HybridInferenceCost` in sub-MIPs -- the rest are kept for parity).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChildSelectionRule {
    Up,
    Down,
    RootSol,
    Obj,
    BestCost,
    WorstCost,
    Disjunction,
    HybridInferenceCost,
}

const UP_RELIABLE: u8 = 1;
const DOWN_RELIABLE: u8 = 2;

/// What the search tells branching about its state.
pub(super) struct BranchCtx<'a> {
    /// `getOptimalityLimit`.
    pub optimality_limit: f64,
    /// `getUpperLimit` (the incumbent cutoff with its tolerance).
    pub upper_limit: f64,
    /// `getFeasTol`, `getEpsilon`.
    pub feastol: f64,
    pub epsilon: f64,
    /// `getNodeQueue().numNodesUp/Down(col)`.
    pub nodes_up_down: &'a dyn Fn(usize) -> (i64, i64),
    /// `checkLimits()` during strong branching.
    pub check_limits: &'a dyn Fn() -> bool,
    /// `childselrule`.
    pub child_rule: ChildSelectionRule,
    /// `inheuristic`.
    pub inheuristic: bool,
}

fn frac(v: f64) -> f64 {
    (v - v.round()).abs()
}

impl HighsEval {
    /// `lp->getFractionalIntegers()` of the last trusted LP point.
    fn fractional_integers(&self, ctx: &NodeCtx<'_>, feastol: f64) -> Vec<(usize, f64)> {
        let Some(x) = &self.point else {
            return Vec::new();
        };
        (0..ctx.ns)
            .filter(|&j| ctx.is_int[j] && frac(x[j]) > feastol)
            .map(|j| (j, x[j]))
            .collect()
    }

    /// `computeLPDegneracy(localdom)` on the working LP `A x = b` (HiGHS's rows are
    /// discopt's rows; a row's basis status is its slack's, a row is an equality when
    /// it has no slack or a fixed one; the dual tolerance is the LP's own maximum
    /// dual infeasibility).
    pub fn lp_degeneracy(&self, ctx: &NodeCtx<'_>, basis: &Basis, y: &[f64]) -> f64 {
        let (m, ns) = (ctx.m_w, ctx.ns);
        let mut slack_of: Vec<Option<usize>> = vec![None; m];
        for j in ns..ctx.n_w {
            let (ri, rv) = ctx.csc_rc.col(j);
            if ri.len() == 1 && rv[0] != 0.0 && ri[0] < m {
                slack_of[ri[0]] = Some(j);
            }
        }
        let dual_of = |j: usize| ctx.c_w[j] - ctx.csc_rc.dot(j, y);
        // `max_dual_infeasibility` over the nonbasic columns.
        let mut dualtol = 0.0f64;
        for j in 0..ctx.n_w {
            let st = basis.col_status.get(j).copied().unwrap_or(BASIC);
            let d = dual_of(j);
            if st == AT_LOWER {
                dualtol = dualtol.max(-d);
            } else if st == AT_UPPER {
                dualtol = dualtol.max(d);
            }
        }
        let (mut fixed_rows, mut inequalities, mut basic_equalities) = (0i64, 0i64, 0i64);
        for i in 0..m {
            match slack_of[i] {
                Some(s) if ctx.l_w[s] != ctx.u_w[s] => {
                    inequalities += 1;
                    if basis.col_status.get(s).copied().unwrap_or(BASIC) != BASIC
                        && y[i].abs() > dualtol
                    {
                        fixed_rows += 1;
                    }
                }
                Some(s) => {
                    basic_equalities +=
                        (basis.col_status.get(s).copied().unwrap_or(BASIC) == BASIC) as i64
                }
                None => {}
            }
        }
        let (mut already_fixed, mut fixed_cols) = (0i64, 0i64);
        for j in 0..ns {
            if basis.col_status.get(j).copied().unwrap_or(BASIC) != BASIC {
                if dual_of(j).abs() > dualtol {
                    fixed_cols += 1;
                } else if self.dom.col_lower[j] == self.dom.col_upper[j] {
                    already_fixed += 1;
                }
            }
        }
        let (nc, nr) = (ns as i64, m as i64);
        let base = nc - already_fixed + inequalities + basic_equalities - nr;
        let share = if base > 0 {
            1.0 - (fixed_cols + fixed_rows) as f64 / base as f64
        } else {
            0.0
        };
        let ratio = if nr > 0 {
            (nc + inequalities + basic_equalities - fixed_cols - fixed_rows - already_fixed) as f64
                / nr as f64
        } else {
            1.0
        };
        let fac1 = if share < 0.8 {
            1.0
        } else {
            10f64.powf(10.0 * (share - 0.7))
        };
        let fac2 = if ratio < 2.0 { 1.0 } else { 10.0 * ratio };
        fac1 * fac2
    }

    /// A branching change on the local domain (`localdom.changeBound(domchg)`, reason
    /// branching).
    fn branch_change(&mut self, col: usize, upper: bool, val: f64) {
        self.dom.change_bound(
            DomainChange {
                boundval: val,
                column: col,
                boundtype: if upper {
                    BoundType::Upper
                } else {
                    BoundType::Lower
                },
            },
            Reason::Branching,
        );
    }

    /// The tail of `analyzeSolution` for one other candidate: try the bound change the
    /// strong-branching point satisfies, propagate, and say whether the point
    /// survives (the change and its propagation undone afterwards).
    fn solution_survives(
        &mut self,
        col: usize,
        upper: bool,
        val: f64,
        sol: &[f64],
        feastol: f64,
    ) -> bool {
        let nchanged = self.dom.changed_cols().len();
        let stack0 = self.dom.domain_change_stack().len();
        self.branch_change(col, upper, val);
        if !self.dom.infeasible() {
            self.dom.propagate();
        }
        if self.dom.infeasible() {
            self.dom.backtrack();
            self.dom.clear_changed_cols_from(nchanged);
            return false;
        }
        let stack = self.dom.domain_change_stack();
        let valid = stack[stack0 + 1..].iter().all(|c| match c.boundtype {
            BoundType::Lower => c.boundval <= sol[c.column] + feastol,
            BoundType::Upper => c.boundval >= sol[c.column] - feastol,
        });
        self.dom.backtrack();
        self.dom.clear_changed_cols_from(nchanged);
        valid
    }

    /// `branchImpl`'s component part: select a candidate (reliability strong
    /// branching, which may commit to a child and re-evaluate), then branch.
    pub fn branch(
        &mut self,
        ctx: &NodeCtx<'_>,
        pc: &mut dyn BranchPseudocost,
        top: &NodeData,
        parent: Option<&NodeData>,
        bc: &BranchCtx<'_>,
    ) -> BranchOutcome {
        debug_assert!(self.dom.changed_cols().is_empty());
        let mut steps: Vec<BranchStep> = Vec::new();
        // The node on top of the control's stack, as the control will see it after
        // each step, and the node under it.
        let mut cur = top.clone();
        cur.decision.col = -1;
        let mut under: Option<NodeData> = parent.cloned();
        let minrel = pc.min_reliable();
        let mut child_lb = cur.lower_bound;
        let mut result = SearchResult::Open;
        let mut down_lb;
        let mut up_lb;
        loop {
            let fracints = self.fractional_integers(ctx, bc.feastol);
            if !(cur.opensubtrees == 2 && self.point.is_some() && !fracints.is_empty()) {
                break;
            }
            let mut sbmaxiters = 0i64;
            if minrel > 0 {
                let sbiters = self.sb_iters;
                sbmaxiters = 100_000 + ((self.lp_iters - self.heur_iters - self.sb_iters) >> 1);
                if sbiters > sbmaxiters {
                    pc.set_min_reliable(0);
                } else if sbiters > (sbmaxiters >> 1) {
                    let ratio = (sbiters - (sbmaxiters >> 1)) as f64
                        / (sbmaxiters - (sbmaxiters >> 1)) as f64;
                    let reduced = (minrel as f64 - ratio * (minrel - 1) as f64) as i64;
                    pc.set_min_reliable(minrel.min(reduced));
                }
            }
            let degeneracy = match (&self.basis, &self.last_dual) {
                (Some(b), Some(y)) => self.lp_degeneracy(ctx, &b.clone(), &y.clone()),
                _ => 1.0,
            };
            pc.set_degeneracy_factor(degeneracy);
            if degeneracy >= 10.0 {
                pc.set_min_reliable(0);
            }
            down_lb = cur.lower_bound;
            up_lb = cur.lower_bound;
            let sel = self.select_candidate(
                ctx,
                pc,
                &mut cur,
                &mut under,
                &mut steps,
                &fracints,
                sbmaxiters,
                bc,
                &mut down_lb,
                &mut up_lb,
            );
            child_lb = cur.lower_bound;
            if let Some(k) = sel {
                let (col, value) = fracints[k];
                cur.branching_point = value;
                let upward = self.child_direction(ctx, pc, &cur, col, bc);
                cur.decision = QDomainChange {
                    col: col as i32,
                    upper: !upward,
                    val: if upward { value.ceil() } else { value.floor() },
                };
                if upward {
                    cur.other_child_lb = down_lb;
                    child_lb = up_lb;
                } else {
                    cur.other_child_lb = up_lb;
                    child_lb = down_lb;
                }
                result = SearchResult::Branched;
                break;
            }
            // A commitment changed the domain: evaluate the new top node.
            debug_assert!(!self.dom.changed_cols().is_empty());
            result = self.evaluate_in_branch(ctx, pc, &mut cur, under.as_ref(), bc);
            steps.push(BranchStep::Update {
                lower_bound: cur.lower_bound,
                estimate: cur.estimate,
                lp_objective: cur.lp_objective,
                opensubtrees: cur.opensubtrees,
            });
            if result == SearchResult::SubOptimal {
                break;
            }
        }
        pc.set_min_reliable(minrel);
        pc.set_degeneracy_factor(1.0);
        if cur.opensubtrees != 2 || result == SearchResult::SubOptimal {
            return BranchOutcome {
                steps,
                result,
                branched: None,
                closed: Some((
                    cur.lower_bound,
                    cur.estimate,
                    cur.opensubtrees,
                    cur.decision.col,
                )),
            };
        }
        if cur.decision.col == -1 {
            self.fallback_branching(ctx, pc, &mut cur, bc);
        }
        if cur.decision.col == -1 {
            // All integer columns fixed and still open. HiGHS declares the node
            // infeasible after re-solving with fresh LPs; discopt keeps its valid bound
            // instead (the certified dual bound stays at or below it).
            self.unresolved_floor = self.unresolved_floor.min(cur.lower_bound);
            cur.opensubtrees = 0;
            return BranchOutcome {
                steps,
                result: SearchResult::LpInfeasible,
                branched: None,
                closed: Some((cur.lower_bound, cur.estimate, 0, -1)),
            };
        }
        // Open the child with the branching decision.
        let pos = self.dom.domain_change_stack().len() as i32;
        let d = cur.decision;
        self.branch_change(d.col as usize, d.upper, d.val);
        BranchOutcome {
            steps,
            result: SearchResult::Branched,
            branched: Some((
                d,
                cur.branching_point,
                cur.other_child_lb,
                cur.lower_bound,
                cur.estimate,
                child_lb,
                pos,
            )),
            closed: None,
        }
    }

    /// The child selection rule of `branchImpl` for candidate `col`: whether to take
    /// the up branch first.
    fn child_direction(
        &mut self,
        ctx: &NodeCtx<'_>,
        pc: &mut dyn BranchPseudocost,
        cur: &NodeData,
        col: usize,
        bc: &BranchCtx<'_>,
    ) -> bool {
        let bp = cur.branching_point;
        match bc.child_rule {
            ChildSelectionRule::Up => true,
            ChildSelectionRule::Down => false,
            ChildSelectionRule::RootSol => {
                let mut down_prio = pc.avg_inferences_down(col) + bc.epsilon;
                let mut up_prio = pc.avg_inferences_up(col) + bc.epsilon;
                let (down_val, up_val) = (bp.floor(), bp.ceil());
                let clamp = |r: f64| r.max(down_val).min(up_val);
                if !self.subrootsol.is_empty() {
                    let rootsol = clamp(self.subrootsol[col]);
                    up_prio *= 1.0 + (bp - rootsol);
                    down_prio *= 1.0 + (rootsol - bp);
                } else {
                    if cur.lp_objective != f64::NEG_INFINITY {
                        if let Some(x) = &self.point {
                            self.subrootsol = x.clone();
                        }
                    }
                    if !self.root_sol.is_empty() {
                        let rootsol = clamp(self.root_sol[col]);
                        up_prio *= 1.0 + (bp - rootsol);
                        down_prio *= 1.0 + (rootsol - bp);
                    }
                }
                up_prio + bc.epsilon >= down_prio
            }
            ChildSelectionRule::Obj => ctx.c_w[col] >= 0.0,
            ChildSelectionRule::BestCost => {
                pc.pseudocost_up_offset(col, bp, bc.feastol)
                    <= pc.pseudocost_down_offset(col, bp, bc.feastol)
            }
            ChildSelectionRule::WorstCost => {
                pc.pseudocost_up(col, bp) >= pc.pseudocost_down(col, bp)
            }
            ChildSelectionRule::Disjunction => {
                let (up, down) = (bc.nodes_up_down)(col);
                if up != down {
                    up > down
                } else {
                    ctx.c_w[col] >= 0.0
                }
            }
            ChildSelectionRule::HybridInferenceCost => {
                let up_score = (1.0 + pc.avg_inferences_up(col))
                    / pc.pseudocost_up_offset(col, bp, bc.feastol);
                let down_score = (1.0 + pc.avg_inferences_down(col))
                    / pc.pseudocost_down_offset(col, bp, bc.feastol);
                up_score >= down_score
            }
        }
    }

    /// The fallback of `branchImpl` when no candidate came out of selection (the LP
    /// has no trusted fractional point): any unfixed integer column by score.
    fn fallback_branching(
        &mut self,
        ctx: &NodeCtx<'_>,
        pc: &mut dyn BranchPseudocost,
        cur: &mut NodeData,
        bc: &BranchCtx<'_>,
    ) {
        let mut bestscore = -1.0f64;
        pc.set_degeneracy_factor(1e6);
        for i in 0..ctx.ns {
            if !ctx.is_int[i] {
                continue;
            }
            let (lo, hi) = (self.dom.col_lower[i], self.dom.col_upper[i]);
            if hi - lo < 0.5 {
                continue;
            }
            let fracval = if lo.is_finite() && hi.is_finite() {
                (0.5 * (lo + hi + 0.5)).floor() + 0.5
            } else if lo.is_finite() {
                lo + 0.5
            } else if hi.is_finite() {
                hi - 0.5
            } else {
                0.5
            };
            let score = pc.score_frac(i, fracval);
            if score > bestscore {
                bestscore = score;
                let cost = match &self.redcost {
                    Some(rc) if self.point.is_some() => rc[i],
                    _ => ctx.c_w[i],
                };
                let upward = if cost.abs() > bc.feastol && bc.upper_limit < f64::INFINITY {
                    cost > 0.0
                } else if pc.avg_inferences_up(i) > pc.avg_inferences_down(i) + bc.feastol {
                    true
                } else if pc.avg_inferences_up(i) < pc.avg_inferences_down(i) - bc.feastol {
                    false
                } else {
                    let n = self.dom.domain_change_stack().len() as i64;
                    let (_, lpos) = self.dom.col_lower_pos(i, n);
                    let (_, upos) = self.dom.col_upper_pos(i, n);
                    lpos <= upos
                };
                let v = if upward {
                    fracval.ceil()
                } else {
                    fracval.floor()
                };
                cur.branching_point = v;
                cur.decision = QDomainChange {
                    col: i as i32,
                    upper: !upward,
                    val: v,
                };
            }
        }
        pc.set_degeneracy_factor(1.0);
    }

    /// `evaluateNode` inside branching, with the control's bookkeeping for the node's
    /// open subtrees (the control applies the reported `Update`).
    fn evaluate_in_branch(
        &mut self,
        ctx: &NodeCtx<'_>,
        pc: &mut dyn BranchPseudocost,
        cur: &mut NodeData,
        parent: Option<&NodeData>,
        bc: &BranchCtx<'_>,
    ) -> SearchResult {
        let early = !bc.inheuristic && cur.lower_bound > bc.optimality_limit;
        let mut r = if early {
            SearchResult::SubOptimal
        } else {
            self.evaluate(ctx, cur, parent, pc, bc.inheuristic, bc.optimality_limit)
        };
        if !early {
            if r != SearchResult::Open {
                cur.opensubtrees = 0;
            } else if !bc.inheuristic && cur.lower_bound > bc.optimality_limit {
                r = SearchResult::SubOptimal;
            }
        }
        r
    }

    /// Commit the top node to the child on the other side of a strong-branching
    /// probe (`branchDownwards`/`branchUpwards` and the parent adjustment), with the
    /// local domain already backtracked from the probe.
    #[allow(clippy::too_many_arguments)]
    fn commit_child(
        &mut self,
        cur: &mut NodeData,
        under: &mut Option<NodeData>,
        steps: &mut Vec<BranchStep>,
        col: usize,
        upper: bool,
        val: f64,
        branchpoint: f64,
        parent_open: u8,
        parent_other_lb: Option<f64>,
    ) {
        let pos = self.dom.domain_change_stack().len() as i32;
        cur.opensubtrees = 1;
        cur.branching_point = branchpoint;
        cur.decision = QDomainChange {
            col: col as i32,
            upper,
            val,
        };
        self.branch_change(col, upper, val);
        let mut child = NodeData::new(cur.lower_bound, cur.estimate);
        child.domchg_stack_pos = pos;
        let mut parent = cur.clone();
        parent.opensubtrees = parent_open;
        if let Some(v) = parent_other_lb {
            parent.other_child_lb = v;
        }
        parent.skip_depth_count = 1;
        steps.push(BranchStep::Commit {
            decision: cur.decision,
            branching_point: branchpoint,
            pos,
            parent_open,
            parent_other_lb: parent.other_child_lb,
        });
        *under = Some(parent);
        *cur = child;
    }

    /// `selectBranchingCandidate`: the index into `fracints` of the chosen candidate,
    /// or `None` when strong branching committed to a child (the domain changed; the
    /// caller re-evaluates).
    #[allow(clippy::too_many_arguments)]
    fn select_candidate(
        &mut self,
        ctx: &NodeCtx<'_>,
        pc: &mut dyn BranchPseudocost,
        cur: &mut NodeData,
        under: &mut Option<NodeData>,
        steps: &mut Vec<BranchStep>,
        fracints: &[(usize, f64)],
        max_sb_iters: i64,
        bc: &BranchCtx<'_>,
        down_lb: &mut f64,
        up_lb: &mut f64,
    ) -> Option<usize> {
        let nf = fracints.len();
        let lb0 = cur.lower_bound;
        let mut upscore = vec![f64::INFINITY; nf];
        let mut downscore = vec![f64::INFINITY; nf];
        let mut upbound = vec![lb0; nf];
        let mut downbound = vec![lb0; nf];
        let mut uprel = vec![false; nf];
        let mut downrel = vec![false; nf];
        for (k, &(col, fracval)) in fracints.iter().enumerate() {
            if pc.is_reliable(col) {
                upscore[k] = pc.pseudocost_up(col, fracval);
                downscore[k] = pc.pseudocost_down(col, fracval);
                uprel[k] = true;
                downrel[k] = true;
            } else {
                let flags = self.reliable_at_node.get(&col).copied().unwrap_or(0);
                if flags & UP_RELIABLE != 0 {
                    upscore[k] = pc.pseudocost_up(col, fracval);
                    uprel[k] = true;
                }
                if flags & DOWN_RELIABLE != 0 {
                    downscore[k] = pc.pseudocost_down(col, fracval);
                    downrel[k] = true;
                }
            }
        }
        let mut min_score = bc.feastol;
        // The parent LP's objective for `objdelta` (`lp->getObjective()` before the
        // probe), its basis to warm-start each probe (the playground restores it).
        let parent_obj = cur.lp_objective;
        let parent_basis = self.basis.clone();
        let parent_point = self.point.clone();
        let parent_redcost = self.redcost.clone();
        let parent_dual = self.last_dual.clone();
        let parent_basis_d = self.basis_d.clone();
        loop {
            let must_stop = self.sb_iters >= max_sb_iters || (bc.check_limits)();
            // `selectBestScore(mustStop)`.
            let mut best: Option<usize> = None;
            let (mut bestscore, mut bestnodes, mut bestnum) = (-1.0f64, -1.0f64, 0i64);
            let oldmin = min_score;
            for k in 0..nf {
                if upscore[k] <= oldmin {
                    uprel[k] = true;
                }
                if downscore[k] <= oldmin {
                    downrel[k] = true;
                }
                let s = 1e-3
                    * (if uprel[k] { upscore[k] } else { 0.0 }).min(if downrel[k] {
                        downscore[k]
                    } else {
                        0.0
                    });
                min_score = min_score.max(s);
                let col = fracints[k].0;
                let score = if upscore[k] <= oldmin || downscore[k] <= oldmin {
                    pc.score_costs(col, upscore[k].min(oldmin), downscore[k].min(oldmin))
                } else if upscore[k] == f64::INFINITY || downscore[k] == f64::INFINITY {
                    if must_stop {
                        pc.score_frac(col, fracints[k].1)
                    } else {
                        f64::INFINITY
                    }
                } else {
                    pc.score_costs(col, upscore[k], downscore[k])
                };
                let (upn, downn) = (bc.nodes_up_down)(col);
                let numnodes = upn + downn;
                let nodes = if upn != 0 || downn != 0 {
                    (downn as f64 / numnodes as f64) * (upn as f64 / numnodes as f64)
                } else {
                    0.0
                };
                if score > bestscore
                    || (score > bestscore - bc.feastol && (nodes, numnodes) > (bestnodes, bestnum))
                {
                    bestscore = score;
                    best = Some(k);
                    bestnodes = nodes;
                    bestnum = numnodes;
                }
            }
            let cand = best.expect("no branching candidate scored");
            if (uprel[cand] && downrel[cand]) || must_stop {
                *down_lb = downbound[cand];
                *up_lb = upbound[cand];
                // Restore the parent LP state the probes overwrote.
                self.basis = parent_basis;
                self.basis_d = parent_basis_d;
                self.point = parent_point;
                self.redcost = parent_redcost;
                self.last_dual = parent_dual;
                return Some(cand);
            }
            let (col, fracval) = fracints[cand];
            let (upval, downval) = (fracval.ceil(), fracval.floor());
            let upbranch = !(!downrel[cand]
                && (uprel[cand]
                    || (downscore[cand], pc.avg_inferences_down(col))
                        >= (upscore[cand], pc.avg_inferences_up(col))));
            // `strongBranch(upbranch)`.
            let mut inferences = -(self.dom.domain_change_stack().len() as i64) - 1;
            let boundval = if upbranch { upval } else { downval };
            self.branch_change(col, !upbranch, boundval);
            if !self.dom.infeasible() {
                self.dom.propagate();
            }
            inferences += self.dom.domain_change_stack().len() as i64;
            // The other side, should this probe prove its side empty or too costly.
            let (oupper, oval) = if upbranch {
                (true, downval)
            } else {
                (false, upval)
            };
            if self.dom.infeasible() {
                pc.add_cutoff_observation(col, upbranch);
                self.dom.backtrack();
                self.dom.clear_changed_cols();
                self.restore(&parent_basis, &parent_point, &parent_redcost, &parent_dual);
                self.basis_d = parent_basis_d.clone();
                self.commit_child(cur, under, steps, col, oupper, oval, fracval, 0, None);
                return None;
            }
            pc.add_inference_observation(col, inferences, upbranch);
            self.dom.clear_changed_cols();
            let (fl, fu) = self.lp_box(ctx);
            let nl = node_lp(
                ctx,
                super::highs_eval::fit_basis_to_box(
                    parent_basis.clone(),
                    parent_basis_d.as_deref(),
                    &fl,
                    &fu,
                ),
                &fl,
                &fu,
            );
            let iters = nl.sol.iters as i64;
            self.lp_iters += iters;
            self.sb_iters += iters;
            let oc = ctx.obj_const;
            match nl.verdict {
                NodeLpVerdict::Bound { bound, integral } => {
                    let solobj = nl.sol.obj + oc;
                    let mut objdelta = (solobj - parent_obj).max(0.0);
                    if objdelta <= bc.epsilon {
                        objdelta = 0.0;
                    }
                    if upbranch {
                        upscore[cand] = objdelta;
                        uprel[cand] = true;
                        *self.reliable_at_node.entry(col).or_insert(0) |= UP_RELIABLE;
                    } else {
                        downscore[cand] = objdelta;
                        downrel[cand] = true;
                        *self.reliable_at_node.entry(col).or_insert(0) |= DOWN_RELIABLE;
                    }
                    let delta = if upbranch {
                        upval - fracval
                    } else {
                        downval - fracval
                    };
                    pc.add_observation(col, delta, objdelta);
                    // `analyzeSolution`: other candidates the probe point already
                    // rounds give their own observation for free.
                    let sol = nl.sol.x[..ctx.ns].to_vec();
                    for k in 0..nf {
                        let (kcol, kval) = fracints[k];
                        if kcol == col {
                            continue;
                        }
                        let (kdown, kup) = (kval.floor(), kval.ceil());
                        if sol[kcol] <= kdown + bc.feastol {
                            if self.dom.col_upper[kcol] > kdown + bc.feastol
                                && !self.solution_survives(kcol, true, kdown, &sol, bc.feastol)
                            {
                                continue;
                            }
                            if objdelta <= bc.feastol {
                                pc.add_observation(kcol, kdown - kval, objdelta);
                                *self.reliable_at_node.entry(kcol).or_insert(0) |= DOWN_RELIABLE;
                            }
                            downscore[k] = downscore[k].min(objdelta);
                        } else if sol[kcol] >= kup - bc.feastol {
                            if self.dom.col_lower[kcol] < kup - bc.feastol
                                && !self.solution_survives(kcol, false, kup, &sol, bc.feastol)
                            {
                                continue;
                            }
                            if objdelta <= bc.feastol {
                                pc.add_observation(kcol, kup - kval, objdelta);
                                *self.reliable_at_node.entry(kcol).or_insert(0) |= UP_RELIABLE;
                            }
                            upscore[k] = upscore[k].min(objdelta);
                        }
                    }
                    if integral {
                        let (x, v) = super::highs_eval::integral_incumbent(ctx, &nl.sol.x);
                        self.offer_found(x, v);
                    }
                    // The child's safe bound (HiGHS: `solobj`, the vertex value).
                    let child_bound = bound + oc;
                    if upbranch {
                        upbound[cand] = child_bound;
                    } else {
                        downbound[cand] = child_bound;
                    }
                    if child_bound > bc.optimality_limit {
                        let pruned = child_bound > self.cutoff;
                        self.dom.backtrack();
                        self.dom.clear_changed_cols();
                        self.restore(&parent_basis, &parent_point, &parent_redcost, &parent_dual);
                        self.basis_d = parent_basis_d.clone();
                        self.commit_child(
                            cur,
                            under,
                            steps,
                            col,
                            oupper,
                            oval,
                            fracval,
                            if pruned { 0 } else { 1 },
                            Some(child_bound),
                        );
                        return None;
                    }
                }
                NodeLpVerdict::InfeasibleProved => {
                    pc.add_cutoff_observation(col, upbranch);
                    self.dom.backtrack();
                    self.dom.clear_changed_cols();
                    self.restore(&parent_basis, &parent_point, &parent_redcost, &parent_dual);
                    self.basis_d = parent_basis_d.clone();
                    self.commit_child(cur, under, steps, col, oupper, oval, fracval, 0, None);
                    return None;
                }
                // An LP failure: score the column zero so it is not chosen if
                // anything else can be (HiGHS's `todo2` arm). A snap-refused or
                // abstained probe offers its verified incumbent candidate first.
                other => {
                    if let NodeLpVerdict::SnapRefused {
                        incumbent: Some((x, v)),
                        ..
                    }
                    | NodeLpVerdict::Abstain {
                        incumbent: Some((x, v)),
                    } = other
                    {
                        self.offer_found(x, v);
                    }
                    downscore[cand] = 0.0;
                    upscore[cand] = 0.0;
                    downrel[cand] = true;
                    uprel[cand] = true;
                    *self.reliable_at_node.entry(col).or_insert(0) |= UP_RELIABLE | DOWN_RELIABLE;
                }
            }
            self.dom.backtrack();
            self.dom.clear_changed_cols();
        }
    }

    fn restore(
        &mut self,
        basis: &Option<Basis>,
        point: &Option<Vec<f64>>,
        redcost: &Option<Vec<f64>>,
        dual: &Option<Vec<f64>>,
    ) {
        self.basis = basis.clone();
        self.point = point.clone();
        self.redcost = redcost.clone();
        self.last_dual = dual.clone();
    }
}

/// Stage 1d-3/1d-5 gate: the production HiGHS-design search ([`super::highs_env::run`])
/// run on the working LP the batched search has at its root, so its certified result
/// can be compared with the batched search's.
#[cfg(test)]
pub(super) mod e2e {
    use super::super::highs_env::{self, Limits};
    use super::*;
    use std::cell::RefCell;

    /// The search's outcome.
    #[derive(Debug, Default, Clone)]
    pub struct E2eResult {
        pub ran: bool,
        /// Best verified point and its value (with `obj_const`).
        pub best: Option<(Vec<f64>, f64)>,
        /// Certified lower bound at the end.
        pub bound: f64,
        pub complete: bool,
        pub nodes: i64,
        pub commits: usize,
        pub branched: usize,
        pub sb_iters: i64,
        pub stopped_by_limit: bool,
        /// Inputs a test may set when arming: the search's relative gap tolerance
        /// (default `1e-9`), and whether to start without the tree's incumbent.
        pub cfg_gap_tol: Option<f64>,
        pub cfg_no_initial: bool,
    }

    thread_local! {
        /// Armed by a test (`Some` with `ran == false`); the search runs once at the
        /// driver's first batch.
        pub static E2E: RefCell<Option<E2eResult>> = const { RefCell::new(None) };
    }

    /// Run the search once if a test armed [`E2E`].
    pub fn maybe_run(ctx: &NodeCtx<'_>) {
        // The top-level solve only: a sub-MIP (RENS/RINS neighbourhood, columns
        // fixed) runs its own driver, whose first batch is not the problem's root.
        if MILP_DEPTH.with(|d| d.get()) != 1 {
            return;
        }
        let armed = E2E.with(|e| e.borrow().as_ref().is_some_and(|r| !r.ran));
        if !armed {
            return;
        }
        let (cfg_gap_tol, cfg_no_initial) = E2E.with(|e| {
            let e = e.borrow();
            let r = e.as_ref().unwrap();
            (r.cfg_gap_tol, r.cfg_no_initial)
        });
        let lim = Limits {
            max_nodes: 200_000,
            max_open_nodes: None,
            deadline: None,
            gap_tol: cfg_gap_tol.unwrap_or(1e-9),
            rel_gap_max_abs: false,
            abs_gap_tol: None,
            heuristics: false,
            strong_branch: true,
            stall_at: None,
        };
        let mut res = E2eResult {
            ran: true,
            cfg_gap_tol,
            cfg_no_initial,
            ..Default::default()
        };
        // Seeded with the tree's incumbent, as the production route is: the root may
        // already have fixed columns by reduced cost against it.
        let initial = ctx
            .tm
            .incumbent()
            .filter(|(_, v)| !cfg_no_initial && *v < INFEAS_SENTINEL - 1.0)
            .map(|(x, v)| (x.to_vec(), v));
        match highs_env::run(
            ctx,
            &lim,
            initial,
            f64::NEG_INFINITY,
            None,
            None,
            Vec::new(),
        ) {
            Some(o) => {
                res.best = o.best;
                res.bound = o.bound;
                res.complete = o.complete;
                res.nodes = o.nodes;
                res.commits = o.commits;
                res.branched = o.branched;
                res.sb_iters = o.sb_iters;
                res.stopped_by_limit = o.stop.is_some();
            }
            None => res.bound = f64::INFINITY,
        }
        E2E.with(|e| *e.borrow_mut() = Some(res));
    }
}
