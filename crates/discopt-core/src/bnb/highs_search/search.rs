//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `lp/highs_cutgen/mod.rs` for why
//! HiGHS is ported rather than linked.
//!
//! The *control* of HiGHS's search (`mip/HighsSearch.cpp`: `installNode`,
//! `evaluateNode`'s bookkeeping, `branch`'s child creation, `backtrack`,
//! `backtrackPlunge`, `dive`, `currentNodeToQueue`, `openNodesToQueue`) and of the main
//! loop's node processing (`mip/HighsMipSolver.cpp`: `installNodes`, `processNodes`,
//! `pruneNode`, `separateAndStoreBasis`, `runHeuristics`, `dive`, `backtrackPlunge`).
//!
//! Everything the control *calls* (the LP and domain inside `evaluateNode`, branching
//! selection, local-domain operations, pseudocost scores, separation, heuristics, the
//! node queue) goes through [`SearchEnv`], so the same control runs against discopt's
//! components in the driver and against a recorded HiGHS solve in the replay
//! (`replay_search.rs`), which feeds HiGHS's component outcomes and requires HiGHS's
//! decisions. What is HiGHS's here and what stays discopt's is set out in
//! `docs/dev/highs-mip-replacement-plan.md` §6.
//!
//! Not translated (not used by the stage-1 configuration, or owned by components):
//! symmetry/orbital fixing, conflict analysis, LP basis storage and recovery, worker
//! parallelism, restarts. Each is a call into `SearchEnv` or a recorded gap.

use super::node_queue::{DomainChange, OpenNode};

/// `HighsSearch::NodeResult` (same order as HiGHS's enum, which the dumps record).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeResult {
    /// `kBoundExceeding`.
    BoundExceeding = 0,
    /// `kDomainInfeasible`.
    DomainInfeasible = 1,
    /// `kLpInfeasible`.
    LpInfeasible = 2,
    /// `kBranched`.
    Branched = 3,
    /// `kSubOptimal`.
    SubOptimal = 4,
    /// `kOpen`.
    Open = 5,
}

impl NodeResult {
    /// From HiGHS's integer value.
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => NodeResult::BoundExceeding,
            1 => NodeResult::DomainInfeasible,
            2 => NodeResult::LpInfeasible,
            3 => NodeResult::Branched,
            4 => NodeResult::SubOptimal,
            5 => NodeResult::Open,
            _ => panic!("not a NodeResult: {v}"),
        }
    }
}

/// `HighsSearch::NodeData` (without the basis and stabilizer orbits, which belong to
/// the LP and symmetry components).
#[derive(Clone, Debug, PartialEq)]
pub struct NodeData {
    /// `lower_bound`.
    pub lower_bound: f64,
    /// `estimate`.
    pub estimate: f64,
    /// `branching_point`.
    pub branching_point: f64,
    /// `lp_objective`.
    pub lp_objective: f64,
    /// `other_child_lb`.
    pub other_child_lb: f64,
    /// `branchingdecision` (column -1: none).
    pub decision: DomainChange,
    /// `domgchgStackPos`.
    pub domchg_stack_pos: i32,
    /// `skipDepthCount`.
    pub skip_depth_count: u8,
    /// `opensubtrees`.
    pub opensubtrees: u8,
}

impl NodeData {
    /// `NodeData(parentlb, parentestimate)`.
    pub fn new(parent_lb: f64, parent_estimate: f64) -> Self {
        NodeData {
            lower_bound: parent_lb,
            estimate: parent_estimate,
            branching_point: 0.0,
            lp_objective: f64::NEG_INFINITY,
            other_child_lb: parent_lb,
            decision: DomainChange {
                col: -1,
                upper: false,
                val: 0.0,
            },
            domchg_stack_pos: -1,
            skip_depth_count: 0,
            opensubtrees: 2,
        }
    }
}

/// One step branch selection takes on the node stack before the final branching
/// (HiGHS's `selectBranchingCandidate` commits to a child when strong branching proves
/// the other side infeasible or bound-exceeding, and re-evaluates nodes).
#[derive(Clone, Debug, PartialEq)]
pub enum BranchStep {
    /// `branchDownwards`/`branchUpwards` on the top node, then the parent's
    /// adjustment (`opensubtrees`, `other_child_lb`, `skipDepthCount = 1`,
    /// `depthoffset -= 1`).
    Commit {
        /// The bound change (`upper`: `branchDownwards`).
        decision: DomainChange,
        /// `branchpoint`.
        branching_point: f64,
        /// Domain stack position before the bound change.
        pos: i32,
        /// The parent's `opensubtrees` after the adjustment.
        parent_open: u8,
        /// The parent's `other_child_lb` after the adjustment.
        parent_other_lb: f64,
    },
    /// A re-evaluation of the top node inside selection (its bounds and state after).
    Update {
        /// Lower bound.
        lower_bound: f64,
        /// Estimate.
        estimate: f64,
        /// LP objective.
        lp_objective: f64,
        /// Open subtrees.
        opensubtrees: u8,
    },
}

/// What branch selection returns to the control.
#[derive(Clone, Debug, PartialEq)]
pub struct BranchOutcome {
    /// Steps taken on the stack, in order.
    pub steps: Vec<BranchStep>,
    /// The result.
    pub result: NodeResult,
    /// When branched: the top node's decision, branching point and other child's
    /// bound, its bound and estimate, the chosen child's bound from selection
    /// (`childLb`), and the domain stack position before the child's bound change.
    pub branched: Option<(DomainChange, f64, f64, f64, f64, f64, i32)>,
    /// When not branched: the top node's bound, estimate and open subtrees.
    pub closed: Option<(f64, f64, u8, i32)>,
}

/// How the main loop took a node from the queue (`MI` in the dump).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallKind {
    /// The root node.
    Root,
    /// `popBestBoundNode`.
    BestBound,
    /// `popBestNode` (hybrid estimate).
    BestEstimate,
}

/// Which queue hand-off (`SQ C|O` in the dump).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueHandoff {
    /// `currentNodeToQueue`.
    Current,
    /// `openNodesToQueue`.
    Open,
}

/// Everything the search control calls. The `note_*` methods mark the points where
/// HiGHS's dump records a decision; the driver's environment ignores them, the replay
/// checks them.
pub trait SearchEnv {
    // --- node queue (`HighsNodeQueue`) ---
    /// `emplaceNode`: returns the tree weight.
    fn queue_emplace(
        &mut self,
        d: Vec<DomainChange>,
        b: Vec<i32>,
        lb: f64,
        est: f64,
        depth: i32,
    ) -> f64;
    /// `popBestNode`.
    fn queue_pop_best(&mut self) -> OpenNode;
    /// `popBestBoundNode`.
    fn queue_pop_best_bound(&mut self) -> OpenNode;
    /// `empty`.
    fn queue_is_empty(&self) -> bool;
    /// `getBestLowerBound`.
    fn queue_best_lower_bound(&self) -> f64;
    /// `getBestBoundDomchgStackSize`.
    fn queue_best_bound_stack_size(&self) -> i32;

    // --- limits ---
    /// `getCutoffBound`.
    fn cutoff_bound(&self) -> f64;
    /// `getOptimalityLimit`.
    fn optimality_limit(&self) -> f64;
    /// `mipdata_->feastol`.
    fn feastol(&self) -> f64;
    /// `HighsSearch::checkLimits(nodeOffset)` (time, node limits).
    fn search_check_limits(&mut self, nnodes: i64) -> bool;
    /// `mipdata_->checkLimits()`.
    fn mip_check_limits(&mut self) -> bool;

    // --- local domain (`HighsDomain` on the search's local domain) ---
    /// `localdom.setDomainChangeStack(stack, branchings)`.
    fn domain_install(&mut self, node: &OpenNode);
    /// `localdom.getDomainChangeStack().size()`.
    fn domain_stack_len(&self) -> i32;
    /// `localdom.backtrack()` (one branching level).
    fn domain_backtrack(&mut self);
    /// `localdom.backtrackToGlobal()` (and flushing it to the LP).
    fn domain_backtrack_to_global(&mut self);
    /// After backtracking into an ancestor with open subtrees: `localdom.propagate()`;
    /// whether the domain became infeasible.
    fn domain_propagate_after_backtrack(&mut self, depth: i32) -> bool;
    /// `localdom.changeBound(decision)`, then (unless `skip_propagate`) `propagate`
    /// (and conflict analysis on infeasibility); whether the domain is infeasible.
    fn domain_try_sibling(&mut self, decision: DomainChange, skip_propagate: bool) -> bool;
    /// Undo `domain_try_sibling` (`localdom.backtrack()`, `clearChangedCols`).
    fn domain_undo_sibling(&mut self);
    /// Before a queue hand-off: `localdom.propagate()`; whether infeasible.
    fn domain_propagate_for_queue(&mut self) -> bool;
    /// `localdom.getReducedDomainChangeStack(branchPositions)`.
    fn domain_reduced_stack(&mut self) -> (Vec<DomainChange>, Vec<i32>);
    /// `localdom.getObjectiveLowerBound()`.
    fn domain_objective_lower_bound(&mut self) -> f64;
    /// `mipdata_->getDomain().propagate()` after a pruned node (`pruneNode`).
    fn global_propagate(&mut self);
    /// `getGlobalDomain().infeasible()`.
    fn global_infeasible(&self) -> bool;

    // --- components ---
    /// The component part of `evaluateNode`: propagation, LP solve, incumbent and
    /// reduced-cost fixing, pseudocost observations, with the recursion when
    /// propagation changed columns. Updates `node` (lower bound, estimate, LP
    /// objective) and returns HiGHS's result *before* the control's suboptimality
    /// test. `parent` is the parent's node data.
    fn evaluate(
        &mut self,
        node: &mut NodeData,
        parent: Option<&NodeData>,
        depth: i32,
    ) -> NodeResult;
    /// The component part of `branch`: candidate selection (strong branching, which
    /// may commit to children and re-evaluate), as steps for the control to apply, and
    /// the outcome. `top` is the node being branched (read-only).
    fn branch(&mut self, top: &NodeData, depth: i32) -> BranchOutcome;
    /// Pseudocost scores for `backtrackPlunge`: the sibling's
    /// (`getScoreDown`/`getScoreUp` of `col` at `point`, `up` for the up branch) and, for
    /// the deepest open ancestor `(col, point, ancestor decision is a lower-bound
    /// change)`, `(inactive, active)`.
    fn plunge_scores(
        &mut self,
        node: (i32, f64, bool),
        ancestor: Option<(i32, f64, bool)>,
    ) -> (f64, Option<(f64, f64)>);
    /// `separateAndStoreBasis`'s separation; whether the global domain became
    /// infeasible.
    fn separate(&mut self) -> bool;
    /// `worker.getAllowHeuristics() && mipdata_->moreHeuristicsAllowed()`.
    fn heuristics_allowed(&mut self) -> bool;
    /// `runHeuristics` after its node evaluation: randomized rounding, RENS/RINS.
    fn run_heuristics(&mut self);

    // --- recorded decision points (dump events) ---
    /// `MI`.
    fn note_install(&mut self, _kind: InstallKind) {}
    /// `SI`.
    fn note_installed(&mut self, _depth: i32, _node: &OpenNode) {}
    /// `SE`: after `evaluateNode` (every call, including recursive ones).
    fn note_evaluated(&mut self, _r: NodeResult, _depth: i32, _node: &NodeData) {}
    /// `SB` (+ `SBP` with the parent when branched).
    fn note_branched(
        &mut self,
        _r: NodeResult,
        _depth: i32,
        _top: &NodeData,
        _parent: Option<&NodeData>,
        _lb0: f64,
    ) {
    }
    /// `SKP`.
    fn note_ancestor_propagated(&mut self, _depth: i32, _infeasible: bool) {}
    /// `SKS`.
    fn note_sibling(&mut self, _depth: i32, _decision: DomainChange, _nodelb: f64, _pruned: bool) {}
    /// `SPQ`.
    fn note_plunge(
        &mut self,
        _depth: i32,
        _decision: DomainChange,
        _nodelb: f64,
        _score: f64,
        _to_queue: bool,
        _ancestor: i32,
    ) {
    }
    /// `SK` / `SP`: after `backtrack` / `backtrackPlunge`.
    fn note_backtracked(&mut self, _plunge: bool, _r: bool, _depth: i32, _top: Option<&NodeData>) {}
    /// `SQ`.
    fn note_queued(&mut self, _kind: QueueHandoff, _depth: i32, _node: &NodeData, _pruned: bool) {}
    /// `SD0` / `SD`.
    fn note_dive(&mut self, _end: Option<(NodeResult, i32, &NodeData)>) {}
    /// `MS`.
    fn note_separate(&mut self) {}
    /// `MH`.
    fn note_heuristics(&mut self) {}
    /// `ML`.
    fn note_plunge_limit(&mut self) {}
    /// After a main-loop round: bounds updated, solutions synced, limits checked,
    /// global domain propagated and the queue pruned (`MG` in the dump). Returns
    /// whether the search stops (`checkLimits`: gap closed, time or node limit).
    fn after_round(&mut self) -> bool;
}

/// `HighsSearch`'s control state.
#[derive(Debug, Default)]
pub struct Search {
    /// `nodestack`.
    pub nodestack: Vec<NodeData>,
    /// `depthoffset`.
    pub depthoffset: i32,
    /// `nnodes`.
    pub nnodes: i64,
    /// `nleaves`.
    pub nleaves: i64,
    /// `treeweight` (the fraction of the tree closed).
    pub treeweight: f64,
    /// `countTreeWeight`.
    pub count_tree_weight: bool,
    /// `inheuristic`.
    pub inheuristic: bool,
}

impl Search {
    /// `getCurrentDepth`.
    pub fn current_depth(&self) -> i32 {
        self.nodestack.len() as i32 + self.depthoffset
    }
    /// `hasNode`.
    pub fn has_node(&self) -> bool {
        !self.nodestack.is_empty()
    }
    /// `currentNodePruned`.
    pub fn current_node_pruned(&self) -> bool {
        self.nodestack.last().unwrap().opensubtrees == 0
    }
    /// `cutoffNode`.
    pub fn cutoff_node(&mut self) {
        self.nodestack.last_mut().unwrap().opensubtrees = 0;
    }

    /// `installNode`.
    pub fn install_node<E: SearchEnv>(&mut self, env: &mut E, node: OpenNode) {
        env.domain_install(&node);
        self.nodestack
            .push(NodeData::new(node.lower_bound, node.estimate));
        self.depthoffset = node.depth - 1;
        let d = self.current_depth();
        env.note_installed(d, &node);
    }

    /// `evaluateNode` (control around the component call).
    pub fn evaluate_node<E: SearchEnv>(&mut self, env: &mut E) -> NodeResult {
        let depth = self.current_depth();
        let optlim = env.optimality_limit();
        let n = self.nodestack.len();
        let early = !self.inheuristic && self.nodestack[n - 1].lower_bound > optlim;
        let mut r = if early {
            NodeResult::SubOptimal
        } else {
            let (head, top) = self.nodestack.split_at_mut(n - 1);
            env.evaluate(&mut top[0], head.last(), depth)
        };
        if !early {
            let top = &mut self.nodestack[n - 1];
            if r != NodeResult::Open {
                self.treeweight += 2f64.powi(1 - depth);
                top.opensubtrees = 0;
            } else if !self.inheuristic && top.lower_bound > optlim {
                r = NodeResult::SubOptimal;
            }
        }
        env.note_evaluated(r, depth, &self.nodestack[n - 1]);
        r
    }

    /// `branchDownwards` (`upper`) / `branchUpwards`: commit the top node to one child.
    fn branch_to(&mut self, decision: DomainChange, branching_point: f64, pos: i32) {
        let top = self.nodestack.last_mut().unwrap();
        top.opensubtrees = 1;
        top.branching_point = branching_point;
        top.decision = decision;
        let mut child = NodeData::new(top.lower_bound, top.estimate);
        child.domchg_stack_pos = pos;
        self.nodestack.push(child);
    }

    /// `branch` (the control around branch selection).
    pub fn branch<E: SearchEnv>(&mut self, env: &mut E) -> NodeResult {
        let depth = self.current_depth();
        let lb0 = self.nodestack.last().unwrap().lower_bound;
        self.nodestack.last_mut().unwrap().decision.col = -1;
        let out = env.branch(self.nodestack.last().unwrap(), depth);
        for step in &out.steps {
            match *step {
                BranchStep::Commit {
                    decision,
                    branching_point,
                    pos,
                    parent_open,
                    parent_other_lb,
                } => {
                    self.branch_to(decision, branching_point, pos);
                    let n = self.nodestack.len();
                    let parent = &mut self.nodestack[n - 2];
                    parent.opensubtrees = parent_open;
                    parent.other_child_lb = parent_other_lb;
                    parent.skip_depth_count = 1;
                    self.depthoffset -= 1;
                }
                BranchStep::Update {
                    lower_bound,
                    estimate,
                    lp_objective,
                    opensubtrees,
                } => {
                    let top = self.nodestack.last_mut().unwrap();
                    top.lower_bound = lower_bound;
                    top.estimate = estimate;
                    top.lp_objective = lp_objective;
                    top.opensubtrees = opensubtrees;
                }
            }
        }
        let r = out.result;
        if let Some((decision, bp, ocl, lb, est, child_lb, pos)) = out.branched {
            let top = self.nodestack.last_mut().unwrap();
            top.decision = decision;
            top.branching_point = bp;
            top.other_child_lb = ocl;
            top.lower_bound = lb;
            top.estimate = est;
            top.opensubtrees = 1;
            let mut child = NodeData::new(child_lb.max(lb), est);
            child.domchg_stack_pos = pos;
            self.nodestack.push(child);
        } else if let Some((lb, est, open, col)) = out.closed {
            let top = self.nodestack.last_mut().unwrap();
            top.lower_bound = lb;
            top.estimate = est;
            top.opensubtrees = open;
            top.decision.col = col;
        }
        let n = self.nodestack.len();
        let parent = (r == NodeResult::Branched && n >= 2).then(|| self.nodestack[n - 2].clone());
        env.note_branched(
            r,
            self.current_depth(),
            &self.nodestack[n - 1],
            parent.as_ref(),
            lb0,
        );
        r
    }

    /// Flip a node's branching decision to the other child (HiGHS's arithmetic).
    fn flip(node: &mut NodeData) -> bool {
        let fallback = node.decision.val == node.branching_point;
        if !node.decision.upper {
            node.decision.upper = true;
            node.decision.val = (node.decision.val - 0.5).floor();
        } else {
            node.decision.upper = false;
            node.decision.val = (node.decision.val + 0.5).ceil();
        }
        if fallback {
            node.branching_point = node.decision.val;
        }
        fallback
    }

    /// The common head of `backtrack`/`backtrackPlunge`: pop closed nodes. Returns
    /// `false` when the stack empties (then the domain is back at the global domain).
    fn pop_closed<E: SearchEnv>(&mut self, env: &mut E) -> bool {
        while self.nodestack.last().unwrap().opensubtrees == 0 {
            self.count_tree_weight = true;
            self.depthoffset += self.nodestack.last().unwrap().skip_depth_count as i32;
            if self.nodestack.len() == 1 {
                self.nodestack.pop();
                env.domain_backtrack_to_global();
                return false;
            }
            self.nodestack.pop();
            env.domain_backtrack();
            if self.nodestack.last().unwrap().opensubtrees != 0 {
                self.count_tree_weight = self.nodestack.last().unwrap().skip_depth_count == 0;
                let depth = self.current_depth();
                let inf = env.domain_propagate_after_backtrack(depth);
                env.note_ancestor_propagated(depth, inf);
                if inf {
                    if self.count_tree_weight {
                        self.treeweight += 2f64.powi(-depth);
                    }
                    self.nodestack.last_mut().unwrap().opensubtrees = 0;
                }
            }
        }
        true
    }

    /// `backtrack`.
    pub fn backtrack<E: SearchEnv>(&mut self, env: &mut E) -> bool {
        let r = self.backtrack_impl(env);
        env.note_backtracked(false, r, self.current_depth(), self.nodestack.last());
        r
    }

    fn backtrack_impl<E: SearchEnv>(&mut self, env: &mut E) -> bool {
        if self.nodestack.is_empty() {
            return false;
        }
        loop {
            if !self.pop_closed(env) {
                return false;
            }
            let depth = self.current_depth();
            let cutoff = env.cutoff_bound();
            let pos = env.domain_stack_len();
            let (decision, nodelb, est) = {
                let curr = self.nodestack.last_mut().unwrap();
                curr.opensubtrees = 0;
                Self::flip(curr);
                (
                    curr.decision,
                    curr.lower_bound.max(curr.other_child_lb),
                    curr.estimate,
                )
            };
            let over = nodelb > cutoff;
            let inf = env.domain_try_sibling(decision, over);
            let prune = over || inf;
            env.note_sibling(depth, decision, nodelb, prune);
            if prune {
                env.domain_undo_sibling();
                if self.count_tree_weight {
                    self.treeweight += 2f64.powi(-depth);
                }
                continue;
            }
            let mut child = NodeData::new(nodelb, est);
            child.domchg_stack_pos = pos;
            self.nodestack.push(child);
            return true;
        }
    }

    /// `backtrackPlunge`.
    pub fn backtrack_plunge<E: SearchEnv>(&mut self, env: &mut E) -> bool {
        let r = self.backtrack_plunge_impl(env);
        env.note_backtracked(true, r, self.current_depth(), self.nodestack.last());
        r
    }

    fn backtrack_plunge_impl<E: SearchEnv>(&mut self, env: &mut E) -> bool {
        if self.nodestack.is_empty() {
            return false;
        }
        loop {
            if !self.pop_closed(env) {
                return false;
            }
            let depth = self.current_depth();
            let cutoff = env.cutoff_bound();
            let (decision, mut nodelb, est, fallback, bp) = {
                let curr = self.nodestack.last_mut().unwrap();
                curr.opensubtrees = 0;
                let fallback = Self::flip(curr);
                (
                    curr.decision,
                    curr.lower_bound.max(curr.other_child_lb),
                    curr.estimate,
                    fallback,
                    curr.branching_point,
                )
            };
            let pos = env.domain_stack_len();
            let over = nodelb > cutoff;
            let inf = env.domain_try_sibling(decision, over);
            let prune = over || inf;
            env.note_sibling(depth, decision, nodelb, prune);
            if prune {
                env.domain_undo_sibling();
                if self.count_tree_weight {
                    self.treeweight += 2f64.powi(-depth);
                }
                continue;
            }
            nodelb = nodelb.max(env.domain_objective_lower_bound());
            let mut to_queue = nodelb > env.optimality_limit();
            // The deepest open ancestor (HiGHS reads only that one, then `break`s).
            let n = self.nodestack.len();
            let ancestor = (0..n - 1)
                .rev()
                .find(|&i| self.nodestack[i].opensubtrees != 0);
            // The new sibling's direction: flipped to an upper-bound change scores down.
            let node_q = (
                decision.col,
                if fallback { 0.5 } else { bp },
                !decision.upper,
            );
            let anc_q = ancestor.map(|i| {
                let a = &self.nodestack[i];
                let afb = a.decision.val == a.branching_point;
                (
                    a.decision.col,
                    if afb { 0.5 } else { a.branching_point },
                    !a.decision.upper,
                )
            });
            let (score, anc_scores) =
                env.plunge_scores(node_q, if to_queue { None } else { anc_q });
            if !to_queue {
                if let Some((inactive, active)) = anc_scores {
                    to_queue = inactive - active > score + env.feastol();
                }
            }
            env.note_plunge(
                depth,
                decision,
                nodelb,
                score,
                to_queue,
                if to_queue && anc_scores.is_none() {
                    -1
                } else {
                    ancestor.map_or(-1, |i| i as i32)
                },
            );
            if to_queue {
                let (stack, br) = env.domain_reduced_stack();
                let w = env.queue_emplace(stack, br, nodelb, est, depth + 1);
                if self.count_tree_weight {
                    self.treeweight += w;
                }
                env.domain_undo_sibling();
                continue;
            }
            let mut child = NodeData::new(nodelb, est);
            child.domchg_stack_pos = pos;
            self.nodestack.push(child);
            return true;
        }
    }

    /// The common body of the queue hand-offs: queue the top node unless it prunes.
    fn top_to_queue<E: SearchEnv>(&mut self, env: &mut E, kind: QueueHandoff) {
        let depth = self.current_depth();
        let lb = self.nodestack.last().unwrap().lower_bound;
        let mut prune = lb > env.cutoff_bound();
        if !prune {
            prune = env.domain_propagate_for_queue();
        }
        env.note_queued(kind, depth, self.nodestack.last().unwrap(), prune);
        if !prune {
            let (stack, br) = env.domain_reduced_stack();
            let olb = env.domain_objective_lower_bound();
            let est = self.nodestack.last().unwrap().estimate;
            let w = env.queue_emplace(stack, br, lb.max(olb), est, depth);
            if self.count_tree_weight {
                self.treeweight += w;
            }
        } else if self.count_tree_weight {
            self.treeweight += 2f64.powi(1 - depth);
        }
        self.nodestack.last_mut().unwrap().opensubtrees = 0;
    }

    /// `currentNodeToQueue`.
    pub fn current_node_to_queue<E: SearchEnv>(&mut self, env: &mut E) {
        self.top_to_queue(env, QueueHandoff::Current);
    }

    /// `openNodesToQueue`.
    pub fn open_nodes_to_queue<E: SearchEnv>(&mut self, env: &mut E) {
        if self.nodestack.is_empty() {
            return;
        }
        if self.nodestack.last().unwrap().opensubtrees == 0 {
            self.backtrack(env);
        }
        while !self.nodestack.is_empty() {
            self.top_to_queue(env, QueueHandoff::Open);
            self.backtrack(env);
        }
    }

    /// `dive`.
    pub fn dive<E: SearchEnv>(&mut self, env: &mut E, node_lim: i64) -> NodeResult {
        env.note_dive(None);
        let r = self.dive_impl(env, node_lim);
        let d = self.current_depth();
        match self.nodestack.last() {
            Some(top) => env.note_dive(Some((r, d, top))),
            None => env.note_dive(Some((r, d, &NodeData::new(f64::NAN, f64::NAN)))),
        }
        r
    }

    fn dive_impl<E: SearchEnv>(&mut self, env: &mut E, node_lim: i64) -> NodeResult {
        loop {
            self.nnodes += 1;
            let r = self.evaluate_node(env);
            if env.search_check_limits(self.nnodes) {
                return r;
            }
            if r != NodeResult::Open {
                return r;
            }
            let r = self.branch(env);
            if r != NodeResult::Branched {
                return r;
            }
            if self.nnodes >= node_lim {
                return r;
            }
        }
    }
}

/// The main loop's node selection state (`numQueueLeaves`, `lastLbLeave`).
#[derive(Debug, Default)]
pub struct MainLoop {
    /// `numQueueLeaves`.
    pub num_queue_leaves: i64,
    /// `lastLbLeave`.
    pub last_lb_leave: i64,
    /// Nodes processed (`mipdata_->num_nodes`, flushed from the search).
    pub num_nodes: i64,
}

/// `kHighsIInf` as a node limit.
pub const NO_NODE_LIMIT: i64 = i64::MAX;

impl MainLoop {
    /// `installNodes` for one worker (and the root's `popBestBoundNode`).
    pub fn install<E: SearchEnv>(&mut self, env: &mut E, search: &mut Search, root: bool) {
        if root {
            env.note_install(InstallKind::Root);
            let n = env.queue_pop_best_bound();
            search.install_node(env, n);
            return;
        }
        if self.num_queue_leaves - self.last_lb_leave >= 10 {
            env.note_install(InstallKind::BestBound);
            let n = env.queue_pop_best_bound();
            search.install_node(env, n);
            self.last_lb_leave = self.num_queue_leaves;
        } else {
            let size = env.queue_best_bound_stack_size();
            let lb = env.queue_best_lower_bound();
            env.note_install(InstallKind::BestEstimate);
            let n = env.queue_pop_best();
            if n.lower_bound == lb && n.domchgstack.len() as i32 == size {
                self.last_lb_leave = self.num_queue_leaves;
            }
            search.install_node(env, n);
        }
        self.num_queue_leaves += 1;
    }

    /// `processNode` for one worker (`processNodes`' body).
    pub fn process_node<E: SearchEnv>(
        &mut self,
        env: &mut E,
        search: &mut Search,
        skip_separation: bool,
        node_lim: i64,
        plunge_limit: i64,
    ) {
        let mut nodes_explored: i64 = 0;
        if !skip_separation {
            // evaluateNode lambda
            if search.evaluate_node(env) == NodeResult::SubOptimal {
                search.current_node_to_queue(env);
            }
            // pruneNode lambda
            if search.current_node_pruned() {
                search.backtrack(env);
                env.global_propagate();
                search.nleaves += 1;
                search.nnodes += 1;
                return;
            }
            if env.global_infeasible() {
                return;
            }
            if env.mip_check_limits() {
                return;
            }
            // separateAndStoreBasis lambda
            env.note_separate();
            if env.separate() {
                search.cutoff_node();
                search.open_nodes_to_queue(env);
                return;
            }
        }
        let mut consider_heuristics = true;
        loop {
            if consider_heuristics
                && (skip_separation || node_lim == NO_NODE_LIMIT)
                && env.heuristics_allowed()
                && self.run_heuristics(env, search)
            {
                break;
            }
            consider_heuristics = false;
            if env.global_infeasible() {
                break;
            }
            if self.dive(env, search, node_lim) {
                break;
            }
            if env.search_check_limits(search.nnodes) {
                break;
            }
            if search.nnodes + nodes_explored >= plunge_limit {
                env.note_plunge_limit();
                break;
            }
            nodes_explored += search.nnodes;
            if !search.backtrack_plunge(env) {
                break;
            }
            // flushStatistics
            self.num_nodes += search.nnodes;
            search.nnodes = 0;
        }
    }

    /// `runHeuristics` lambda.
    fn run_heuristics<E: SearchEnv>(&mut self, env: &mut E, search: &mut Search) -> bool {
        env.note_heuristics();
        if search.evaluate_node(env) == NodeResult::SubOptimal {
            return true;
        }
        if search.current_node_pruned() {
            search.nleaves += 1;
            return false;
        }
        env.run_heuristics();
        env.global_infeasible()
    }

    /// `dive` lambda.
    fn dive<E: SearchEnv>(&mut self, env: &mut E, search: &mut Search, node_lim: i64) -> bool {
        let lim = if node_lim == NO_NODE_LIMIT {
            i64::MAX
        } else {
            search.nnodes + node_lim
        };
        if !search.current_node_pruned() {
            if search.dive(env, lim) == NodeResult::SubOptimal {
                return true;
            }
            search.nleaves += 1;
        }
        node_lim != NO_NODE_LIMIT && search.nnodes >= lim
    }

    /// One main-loop round for a single worker: install, process, return open nodes.
    /// Returns whether the search stops.
    pub fn round<E: SearchEnv>(
        &mut self,
        env: &mut E,
        search: &mut Search,
        root: bool,
        node_lim: i64,
    ) -> bool {
        self.install(env, search, root);
        self.process_node(env, search, root, node_lim, 100);
        search.open_nodes_to_queue(env);
        self.num_nodes += search.nnodes;
        search.nnodes = 0;
        env.after_round()
    }
}

#[cfg(test)]
mod tests {
    use super::super::node_queue::NodeQueue;
    use super::*;

    /// A minimal environment: a real queue, a domain whose queue-time propagation
    /// result is fixed, and nothing else.
    struct Stub {
        q: NodeQueue,
        prune_on_queue: bool,
        emplaced: Vec<(f64, f64, i32)>,
    }
    impl SearchEnv for Stub {
        fn queue_emplace(
            &mut self,
            d: Vec<DomainChange>,
            b: Vec<i32>,
            lb: f64,
            est: f64,
            depth: i32,
        ) -> f64 {
            self.emplaced.push((lb, est, depth));
            self.q.emplace_node(d, b, lb, est, depth).1
        }
        fn queue_pop_best(&mut self) -> OpenNode {
            self.q.pop_best_node().1
        }
        fn queue_pop_best_bound(&mut self) -> OpenNode {
            self.q.pop_best_bound_node().1
        }
        fn queue_is_empty(&self) -> bool {
            self.q.is_empty()
        }
        fn queue_best_lower_bound(&self) -> f64 {
            self.q.best_lower_bound()
        }
        fn queue_best_bound_stack_size(&self) -> i32 {
            self.q.best_bound_domchg_stack_size()
        }
        fn cutoff_bound(&self) -> f64 {
            f64::INFINITY
        }
        fn optimality_limit(&self) -> f64 {
            f64::INFINITY
        }
        fn feastol(&self) -> f64 {
            1e-6
        }
        fn search_check_limits(&mut self, _n: i64) -> bool {
            false
        }
        fn mip_check_limits(&mut self) -> bool {
            false
        }
        fn domain_install(&mut self, _n: &OpenNode) {}
        fn domain_stack_len(&self) -> i32 {
            0
        }
        fn domain_backtrack(&mut self) {}
        fn domain_backtrack_to_global(&mut self) {}
        fn domain_propagate_after_backtrack(&mut self, _d: i32) -> bool {
            false
        }
        fn domain_try_sibling(&mut self, _d: DomainChange, _s: bool) -> bool {
            false
        }
        fn domain_undo_sibling(&mut self) {}
        fn domain_propagate_for_queue(&mut self) -> bool {
            self.prune_on_queue
        }
        fn domain_reduced_stack(&mut self) -> (Vec<DomainChange>, Vec<i32>) {
            (Vec::new(), Vec::new())
        }
        fn domain_objective_lower_bound(&mut self) -> f64 {
            f64::NEG_INFINITY
        }
        fn global_propagate(&mut self) {}
        fn global_infeasible(&self) -> bool {
            false
        }
        fn evaluate(&mut self, _n: &mut NodeData, _p: Option<&NodeData>, _d: i32) -> NodeResult {
            NodeResult::Open
        }
        fn branch(&mut self, _t: &NodeData, _d: i32) -> BranchOutcome {
            unreachable!()
        }
        fn plunge_scores(
            &mut self,
            _n: (i32, f64, bool),
            _a: Option<(i32, f64, bool)>,
        ) -> (f64, Option<(f64, f64)>) {
            (0.0, None)
        }
        fn separate(&mut self) -> bool {
            false
        }
        fn heuristics_allowed(&mut self) -> bool {
            false
        }
        fn run_heuristics(&mut self) {}
        fn after_round(&mut self) -> bool {
            false
        }
    }

    /// A root (one child explored, the other open) with an open child on top.
    fn two_level() -> Search {
        let mut root = NodeData::new(1.0, 2.0);
        root.decision = DomainChange {
            col: 0,
            upper: false,
            val: 1.0,
        };
        root.branching_point = 0.5;
        root.opensubtrees = 1;
        let child = NodeData::new(1.5, 2.5);
        Search {
            nodestack: vec![root, child],
            count_tree_weight: true,
            ..Default::default()
        }
    }

    /// `openNodesToQueue` with every hand-off pruned by propagation: nothing reaches the
    /// queue, each pruned node adds `2^(1 - depth)` to the tree weight, and the stack
    /// empties (the path HiGHS never took on the population, `SQ O` pruned = 0 of 8,926).
    #[test]
    fn open_nodes_to_queue_prunes_without_queueing() {
        let mut env = Stub {
            q: NodeQueue::new(1),
            prune_on_queue: true,
            emplaced: Vec::new(),
        };
        let mut s = two_level();
        s.open_nodes_to_queue(&mut env);
        assert!(env.emplaced.is_empty(), "{:?}", env.emplaced);
        assert!(s.nodestack.is_empty());
        // The child (depth 2) and the root's other child (depth 2) were pruned.
        assert!((s.treeweight - 2.0 * 0.5).abs() < 1e-15, "{}", s.treeweight);

        // Control: without the prune both reach the queue, at depth 2, with their bounds.
        let mut env = Stub {
            q: NodeQueue::new(1),
            prune_on_queue: false,
            emplaced: Vec::new(),
        };
        let mut s = two_level();
        s.open_nodes_to_queue(&mut env);
        assert_eq!(env.emplaced, vec![(1.5, 2.5, 2), (1.0, 2.0, 2)]);
        assert!(s.nodestack.is_empty());
    }
}
