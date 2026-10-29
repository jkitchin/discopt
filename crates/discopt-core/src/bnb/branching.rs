//! Branching strategies for the B&B tree.
//!
//! Supports most-fractional branching (default) and pseudocost-based branching.
//! Pseudocosts track the historical per-unit bound improvement when branching
//! on each variable, enabling reliability branching (use pseudocosts when
//! reliable, fall back to strong branching otherwise).

use crate::bnb::node::{BranchRecord, Node, NodeId, NodeStatus};

/// Minimal variable info for branching decisions.
///
/// Decoupled from expr module (built separately). Each entry describes one
/// variable group: its flat offset in the variable vector, size (1 for scalars),
/// and whether it requires integrality.
#[derive(Debug, Clone)]
pub struct VarBranchInfo {
    /// Flat offset into the variable vector.
    pub offset: usize,
    /// Number of elements (1 for scalar, >1 for vector variables).
    pub size: usize,
    /// True for Binary and Integer variable types.
    pub is_integer: bool,
}

/// A decision about which variable to branch on and the split point.
#[derive(Debug, Clone)]
pub struct BranchDecision {
    /// Flat index into the variable vector.
    pub var_index: usize,
    /// Value to branch on (typically floor of fractional value for integers).
    pub branch_point: f64,
}

/// Integrality tolerance: values within this distance of an integer are
/// considered integral.
const INTEGRALITY_TOL: f64 = 1e-5;

// ---------------------------------------------------------------------------
// Pseudocost tracking
// ---------------------------------------------------------------------------

/// Per-variable pseudocost tracker.
///
/// Pseudocosts measure the average bound improvement per unit of variable
/// change when branching on a variable:
///   - `down_cost`: average (parent_lb - child_lb) / (frac_part) for down branches
///   - `up_cost`: average (parent_lb - child_lb) / (1 - frac_part) for up branches
///
/// where child_lb is the relaxation lower bound of the child node.
#[derive(Debug, Clone)]
pub struct Pseudocosts {
    /// Sum of per-unit bound changes for down branches, per flat variable index.
    down_sum: Vec<f64>,
    /// Count of down-branch observations.
    down_count: Vec<u32>,
    /// Sum of per-unit bound changes for up branches.
    up_sum: Vec<f64>,
    /// Count of up-branch observations.
    up_count: Vec<u32>,
    /// Default pseudocost for variables with no observations.
    default_cost: f64,
}

impl Pseudocosts {
    /// Create a new pseudocost tracker for `n_vars` variables.
    pub fn new(n_vars: usize) -> Self {
        Self {
            down_sum: vec![0.0; n_vars],
            down_count: vec![0; n_vars],
            up_sum: vec![0.0; n_vars],
            up_count: vec![0; n_vars],
            default_cost: 1.0,
        }
    }

    /// Record a pseudocost observation.
    ///
    /// - `var_index`: which variable was branched on.
    /// - `parent_lb`: parent node's relaxation lower bound.
    /// - `child_lb`: child node's relaxation lower bound.
    /// - `frac_part`: fractional part of the variable at the parent.
    /// - `is_down`: true if this was the down branch (x <= floor).
    pub fn update(
        &mut self,
        var_index: usize,
        parent_lb: f64,
        child_lb: f64,
        frac_part: f64,
        is_down: bool,
    ) {
        if var_index >= self.down_sum.len() {
            return;
        }
        let delta_bound = child_lb - parent_lb;
        // delta_bound should be >= 0 (child is at least as tight as parent)
        let delta_bound = delta_bound.max(0.0);

        if is_down {
            let delta_var = frac_part.max(1e-10);
            self.down_sum[var_index] += delta_bound / delta_var;
            self.down_count[var_index] += 1;
        } else {
            let delta_var = (1.0 - frac_part).max(1e-10);
            self.up_sum[var_index] += delta_bound / delta_var;
            self.up_count[var_index] += 1;
        }
    }

    /// Get the average down pseudocost for a variable.
    pub fn down_cost(&self, var_index: usize) -> f64 {
        if var_index >= self.down_count.len() || self.down_count[var_index] == 0 {
            return self.default_cost;
        }
        self.down_sum[var_index] / self.down_count[var_index] as f64
    }

    /// Get the average up pseudocost for a variable.
    pub fn up_cost(&self, var_index: usize) -> f64 {
        if var_index >= self.up_count.len() || self.up_count[var_index] == 0 {
            return self.default_cost;
        }
        self.up_sum[var_index] / self.up_count[var_index] as f64
    }

    /// Total number of observations (down + up) for a variable.
    pub fn observation_count(&self, var_index: usize) -> u32 {
        if var_index >= self.down_count.len() {
            return 0;
        }
        self.down_count[var_index] + self.up_count[var_index]
    }

    /// Compute the pseudocost score for branching on a variable.
    ///
    /// Uses the product scoring rule: score = down_gain * up_gain,
    /// where gain = pseudocost * delta_var. This is the standard
    /// SCIP/Gurobi scoring heuristic.
    pub fn score(&self, var_index: usize, frac_part: f64) -> f64 {
        let d = self.down_cost(var_index) * frac_part;
        let u = self.up_cost(var_index) * (1.0 - frac_part);
        // Product score with small epsilon to avoid zero
        (1e-6 + d) * (1e-6 + u)
    }
}

// ---------------------------------------------------------------------------
// Branching variable selection
// ---------------------------------------------------------------------------

/// Select the most-fractional variable for branching.
///
/// Among all integer variables with fractional values, selects the one whose
/// fractional part is closest to 0.5 (i.e., most ambiguous / most fractional).
/// Returns `None` if all integer variables are at integral values.
pub fn select_branch_variable(
    solution: &[f64],
    variables: &[VarBranchInfo],
) -> Option<BranchDecision> {
    let mut best: Option<BranchDecision> = None;
    let mut best_fractionality = f64::NEG_INFINITY;

    for var in variables {
        if !var.is_integer {
            continue;
        }
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx >= solution.len() {
                continue;
            }
            let val = solution[idx];
            let frac = val - val.floor();

            // Skip if effectively integral.
            if !(INTEGRALITY_TOL..=1.0 - INTEGRALITY_TOL).contains(&frac) {
                continue;
            }

            // Fractionality metric: closeness to 0.5 (higher = more fractional).
            // We use 0.5 - |frac - 0.5|, which is maximized when frac == 0.5.
            let score = 0.5 - (frac - 0.5).abs();

            if score > best_fractionality {
                best_fractionality = score;
                best = Some(BranchDecision {
                    var_index: idx,
                    branch_point: val.floor(),
                });
            }
        }
    }

    best
}

/// Select the branching variable using pseudocost scores.
///
/// Among all integer variables with fractional values, selects the one with
/// the highest pseudocost product score. Falls back to most-fractional if
/// no variable has pseudocost observations.
///
/// Returns a tuple of (decision, unreliable_candidates):
/// - `decision`: the best variable to branch on (or None).
/// - `unreliable_candidates`: flat indices of fractional variables with fewer
///   than `reliability_threshold` pseudocost observations. These are candidates
///   for strong branching.
pub fn select_branch_variable_pseudocost(
    solution: &[f64],
    variables: &[VarBranchInfo],
    pseudocosts: &Pseudocosts,
    reliability_threshold: u32,
) -> (Option<BranchDecision>, Vec<usize>) {
    let mut best: Option<BranchDecision> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut unreliable: Vec<usize> = Vec::new();

    for var in variables {
        if !var.is_integer {
            continue;
        }
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx >= solution.len() {
                continue;
            }
            let val = solution[idx];
            let frac = val - val.floor();

            if !(INTEGRALITY_TOL..=1.0 - INTEGRALITY_TOL).contains(&frac) {
                continue;
            }

            let obs = pseudocosts.observation_count(idx);
            if obs < reliability_threshold {
                unreliable.push(idx);
            }

            let score = pseudocosts.score(idx, frac);
            if score > best_score {
                best_score = score;
                best = Some(BranchDecision {
                    var_index: idx,
                    branch_point: val.floor(),
                });
            }
        }
    }

    (best, unreliable)
}

/// Check if a solution is integer-feasible for all integer variables.
pub fn is_integer_feasible(solution: &[f64], variables: &[VarBranchInfo]) -> bool {
    for var in variables {
        if !var.is_integer {
            continue;
        }
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx >= solution.len() {
                return false;
            }
            let val = solution[idx];
            let frac = val - val.floor();
            if frac > INTEGRALITY_TOL && frac < 1.0 - INTEGRALITY_TOL {
                return false;
            }
        }
    }
    true
}

/// Create two child nodes from a parent node and a branch decision.
///
/// Left child: upper bound on branch variable tightened to floor(value).
/// Right child: lower bound on branch variable tightened to ceil(value).
///
/// The `next_id` closure is called twice to assign IDs to the children.
pub fn create_children(
    parent: &Node,
    decision: &BranchDecision,
    mut next_id: impl FnMut() -> NodeId,
) -> (Node, Node) {
    let idx = decision.var_index;
    let bp = decision.branch_point;

    // Warm-start: pass parent's stored solution and optimal LP basis to
    // children. The parent's basis is dual-feasible for each child (the branch
    // only tightens one bound), so the dual simplex re-optimizes in few pivots.
    let parent_sol = parent.parent_solution.clone();
    let parent_basis = parent.basis.clone();

    // Inherit parent's lower bound: the child's feasible region is a subset
    // of the parent's, so the parent's lower bound is valid for the child.
    // This prevents the global lower bound from dropping to -inf when
    // unsolved children enter the pool.
    let inherited_lb = parent.local_lower_bound;

    // Left child: x_i <= floor(val)
    let mut left_ub = parent.ub.clone();
    left_ub[idx] = bp; // bp is already floor(val)
    let left = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        branch: Some(BranchRecord {
            var: idx,
            up: false,
            value: bp,
        }),
        lb: parent.lb.clone(),
        ub: left_ub,
        local_lower_bound: inherited_lb,
        best_estimate: inherited_lb,
        status: NodeStatus::Pending,
        parent_solution: parent_sol.clone(),
        basis: parent_basis.clone(),
    };

    // Right child: x_i >= ceil(val)
    let mut right_lb = parent.lb.clone();
    right_lb[idx] = bp + 1.0; // ceil(val) = floor(val) + 1
    let right = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        branch: Some(BranchRecord {
            var: idx,
            up: true,
            value: bp + 1.0,
        }),
        lb: right_lb,
        ub: parent.ub.clone(),
        local_lower_bound: inherited_lb,
        best_estimate: inherited_lb,
        status: NodeStatus::Pending,
        parent_solution: parent_sol,
        basis: parent_basis,
    };

    (left, right)
}

/// Create two child nodes by spatial bisection on a continuous variable.
///
/// Left child: x_i <= midpoint.  Right child: x_i >= midpoint.
/// Unlike integer branching, there is no +1 gap between children.
pub fn create_children_spatial(
    parent: &Node,
    decision: &BranchDecision,
    mut next_id: impl FnMut() -> NodeId,
) -> (Node, Node) {
    let idx = decision.var_index;
    let bp = decision.branch_point; // midpoint

    let parent_sol = parent.parent_solution.clone();
    let inherited_lb = parent.local_lower_bound;

    // Left child: x_i <= midpoint
    let mut left_ub = parent.ub.clone();
    left_ub[idx] = bp;
    let left = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        branch: Some(BranchRecord {
            var: idx,
            up: false,
            value: bp,
        }),
        lb: parent.lb.clone(),
        ub: left_ub,
        local_lower_bound: inherited_lb,
        best_estimate: inherited_lb,
        status: NodeStatus::Pending,
        parent_solution: parent_sol.clone(),
        basis: None, // spatial branching is MINLP-side; no simplex basis
    };

    // Right child: x_i >= midpoint (no gap)
    let mut right_lb = parent.lb.clone();
    right_lb[idx] = bp;
    let right = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        branch: Some(BranchRecord {
            var: idx,
            up: true,
            value: bp,
        }),
        lb: right_lb,
        ub: parent.ub.clone(),
        local_lower_bound: inherited_lb,
        best_estimate: inherited_lb,
        status: NodeStatus::Pending,
        parent_solution: parent_sol,
        basis: None, // spatial branching is MINLP-side; no simplex basis
    };

    (left, right)
}

/// Minimum relative domain width for a continuous variable to be eligible
/// for spatial branching.  Prevents infinite bisection.
const SPATIAL_MIN_WIDTH: f64 = 1e-6;

/// Select a continuous variable for spatial branching (longest-edge bisection).
///
/// Picks the variable with the widest domain relative to its global range.
/// Only considers variables whose current domain width exceeds
/// [`SPATIAL_MIN_WIDTH`] times the global width.
///
/// `deprioritized` (length `n_vars`, or empty to disable) flags continuous
/// variables that are *functionally dependent* — each is pinned by a single
/// equality `x_i = f(others)`, so branching on it merely re-splits a value the
/// relaxation already determines from the independent variables. Spatial B&B
/// makes no progress bisecting such a variable: the McCormick gap is driven by
/// the independent inputs, not the dependent output.
///
/// `allow_dependent` controls whether those deprioritized columns are eligible.
/// The caller branches in stages — independent continuous first
/// (`allow_dependent = false`), then *independent integer* domain-partition
/// branching (see [`select_spatial_integer_branch_variable`]), and only if both
/// of those are exhausted does it call again with `allow_dependent = true` as a
/// completeness-preserving last resort. This ordering matters: on the
/// welded-beam class (nvs05) the objective gap is driven by an integer product
/// (`i1*i2`), so the *integer* domain partition must run before any dependent
/// continuous output is bisected — otherwise the search wastes branches on the
/// pinned stress intermediates and the bound never climbs. The dependent
/// fallback never refuses to branch when a branchable dimension remains, so the
/// search stays complete.
pub fn select_spatial_branch_variable(
    node_lb: &[f64],
    node_ub: &[f64],
    global_lb: &[f64],
    global_ub: &[f64],
    integer_vars: &[VarBranchInfo],
    deprioritized: &[bool],
    allow_dependent: bool,
) -> Option<(BranchDecision, f64)> {
    let n = node_lb.len();

    // Build a quick lookup of which flat indices are integer variables.
    let mut is_int = vec![false; n];
    for var in integer_vars {
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx < n {
                is_int[idx] = true;
            }
        }
    }

    let mut best_idx: Option<usize> = None;
    let mut best_width = 0.0_f64;
    for idx in 0..n {
        if is_int[idx] {
            continue;
        }
        if !allow_dependent && idx < deprioritized.len() && deprioritized[idx] {
            continue; // Dependent variable, skip unless this is the fallback pass.
        }
        let width = node_ub[idx] - node_lb[idx];
        let global_width = global_ub[idx] - global_lb[idx];
        if global_width < 1e-15 {
            continue; // Fixed variable, skip.
        }
        let relative_width = width / global_width;
        if !relative_width.is_finite() {
            // An unbounded dimension (infinite node width, or an infinite global
            // range) yields NaN/inf here. Such a dimension is NOT spatially
            // branchable: bisecting an infinite interval never shrinks its
            // normalized width. Skip it deterministically. Without this, a NaN
            // relative width silently defeats both the `< SPATIAL_MIN_WIDTH`
            // "too tight" filter (NaN < x is false) and the `> best_width`
            // selection (NaN > x is false), so the variable is neither skipped
            // as fixed nor selected — the root dead-ends unbranched and the
            // tree falsely certifies (issue #467).
            continue;
        }
        if relative_width < SPATIAL_MIN_WIDTH {
            continue; // Domain already very tight.
        }
        // Select on the same relative width used for the threshold so the
        // branch falls on the dimension that has shrunk least relative to its
        // root domain (longest-edge in normalized coordinates). Selecting on
        // the absolute width instead just tracks whichever variable happens to
        // have the largest raw range. Strict `>` with the ascending scan keeps
        // tie-breaking deterministic (lowest index wins).
        if relative_width > best_width {
            best_width = relative_width;
            best_idx = Some(idx);
        }
    }

    best_idx.map(|idx| {
        let mid = 0.5 * (node_lb[idx] + node_ub[idx]);
        (
            BranchDecision {
                var_index: idx,
                branch_point: mid,
            },
            best_width,
        )
    })
}

/// A nonlinear term of ORIGINAL columns the Python spatial tree scores branching
/// candidates on (#1978): a bilinear product `x_i * x_j` (`i == j` is a square) or
/// a power `x_i ^ p`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreTerm {
    /// `x_i * x_j`.
    Bilinear(usize, usize),
    /// `x_i ^ p`.
    Power(usize, f64),
}

/// Fraction of the way the split point is pulled from the node point to the box
/// midpoint (SCIP's `branching/midpull`). `1.0` = the midpoint, the legacy rule's
/// split: #1970's native `0.75` cost `test_sign_envelope` two certificates here,
/// because a discontinuity at a box midpoint (`sign(x) == 0` on `[-2, 2]`) then
/// never lands on a child boundary; on #1975's prolog piece the native score
/// certifies without midpull too (57 nodes).
const SCORE_MIDPULL: f64 = 1.0;

/// Gap between a term and its McCormick / secant envelope at `x` over the box,
/// computed from the point and the box alone (the tree sees only original
/// columns). `None` for a term with an infinite or crossed side.
fn score_term_gap(t: &ScoreTerm, x: &[f64], lb: &[f64], ub: &[f64]) -> Option<f64> {
    let clamp = |i: usize| x[i].max(lb[i]).min(ub[i]);
    let finite = |i: usize| lb[i].is_finite() && ub[i].is_finite() && lb[i] <= ub[i];
    match *t {
        ScoreTerm::Bilinear(i, j) => {
            if !(finite(i) && finite(j)) {
                return None;
            }
            let (xi, xj) = (clamp(i), clamp(j));
            if i == j {
                // Secant gap of x^2: (x - l)(u - x).
                return Some((xi - lb[i]) * (ub[i] - xi));
            }
            let (dli, dui, dlj, duj) = (xi - lb[i], ub[i] - xi, xj - lb[j], ub[j] - xj);
            let under = (dli * dlj).min(dui * duj);
            let over = (dli * duj).min(dui * dlj);
            Some(under + over)
        }
        ScoreTerm::Power(i, p) => {
            if !finite(i) || !p.is_finite() {
                return None;
            }
            let integral = (p - p.round()).abs() < 1e-12;
            if !integral && lb[i] < 0.0 {
                return None;
            }
            let (l, u, xi) = (lb[i], ub[i], clamp(i));
            if u - l <= 0.0 {
                return Some(0.0);
            }
            let (fl, fu, fx) = (l.powf(p), u.powf(p), xi.powf(p));
            if !(fl.is_finite() && fu.is_finite() && fx.is_finite()) {
                return None;
            }
            let secant = fl + (fu - fl) / (u - l) * (xi - l);
            Some((secant - fx).abs())
        }
    }
}

/// #1978: choose the continuous column to bisect by its share of the envelope gap
/// of the nonlinear terms it appears in at the node point `x` -- #1970's native
/// rule (after SCIP's nonlinear-handler rule, Bestuzheva et al. 2023), ported to
/// the Python spatial tree, which sees only original columns and so measures each
/// term's gap from the point and the box (see [`score_term_gap`]).
///
/// The legacy rule ([`select_spatial_branch_variable`]) bisects the widest column
/// relative to its root range across EVERY continuous column, including ones that
/// appear only linearly and whose split shrinks no envelope. On #1975's rescaled
/// prolog piece that never isolates the narrow factors `u0, u1 in [0, ~5]` against
/// their wide partners `x2..x5 in [0, ~7000]`, and the bound stalls at 0.
///
/// Candidates are the continuous, non-deprioritized columns of terms with a
/// positive gap whose width relative to `ref` is at least [`SPATIAL_MIN_WIDTH`].
/// Each term's gap is split evenly over its candidate operands; the largest total
/// wins (ties: wider relative width, then lower index). The split point is
/// pulled [`SCORE_MIDPULL`] of the way from the point to the midpoint (all the way:
/// the midpoint). `None` when
/// no term has a gap on a candidate -- the caller then falls back to the legacy
/// rule, so the search stays complete. Branching order only: both children still
/// cover the parent box.
#[allow(clippy::too_many_arguments)]
pub fn select_spatial_score_variable(
    node_lb: &[f64],
    node_ub: &[f64],
    ref_lb: &[f64],
    ref_ub: &[f64],
    x: &[f64],
    integer_vars: &[VarBranchInfo],
    deprioritized: &[bool],
    terms: &[ScoreTerm],
) -> Option<(BranchDecision, f64)> {
    let n = node_lb.len();
    if x.len() < n {
        return None;
    }
    let mut is_int = vec![false; n];
    for var in integer_vars {
        for i in 0..var.size {
            if var.offset + i < n {
                is_int[var.offset + i] = true;
            }
        }
    }
    let rel_width = |i: usize| -> Option<f64> {
        let w = node_ub[i] - node_lb[i];
        let g = ref_ub[i] - ref_lb[i];
        if g < 1e-15 {
            return None;
        }
        let r = w / g;
        (r.is_finite() && r >= SPATIAL_MIN_WIDTH).then_some(r)
    };
    let eligible = |i: usize| -> bool {
        i < n
            && !is_int[i]
            && !(i < deprioritized.len() && deprioritized[i])
            && rel_width(i).is_some()
    };
    let mut viol = vec![0.0f64; n];
    for t in terms {
        let ops: [Option<usize>; 2] = match *t {
            ScoreTerm::Bilinear(i, j) if i != j => [Some(i), Some(j)],
            ScoreTerm::Bilinear(i, _) | ScoreTerm::Power(i, _) => [Some(i), None],
        };
        if ops.iter().flatten().any(|&i| i >= n) {
            continue;
        }
        let gap = match score_term_gap(t, x, node_lb, node_ub) {
            Some(g) if g > 0.0 && g.is_finite() => g,
            _ => continue,
        };
        let cands: Vec<usize> = ops
            .iter()
            .flatten()
            .copied()
            .filter(|&i| eligible(i))
            .collect();
        let share = gap / cands.len().max(1) as f64;
        for i in cands {
            viol[i] += share;
        }
    }
    let mut best: Option<(usize, f64, f64)> = None; // (idx, viol, rel width)
    for (i, &v) in viol.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        let r = rel_width(i).unwrap_or(0.0);
        let better = match best {
            None => true,
            Some((_, bv, br)) => v > bv || (v == bv && r > br),
        };
        if better {
            best = Some((i, v, r));
        }
    }
    let (idx, _, r) = best?;
    let (l, u) = (node_lb[idx], node_ub[idx]);
    let mid = 0.5 * (l + u);
    let xc = x[idx].max(l).min(u);
    let mut bp = SCORE_MIDPULL * mid + (1.0 - SCORE_MIDPULL) * xc;
    if !(bp > l && bp < u) {
        bp = mid;
    }
    Some((
        BranchDecision {
            var_index: idx,
            branch_point: bp,
        },
        r,
    ))
}

/// Select an *integer* variable for spatial (domain-partition) branching.
///
/// Fallback used by nonconvex spatial B&B when no continuous variable is
/// available to bisect, yet the convex relaxation still has an open gap on a
/// product/power involving an integer variable (e.g. gear4's `i1*i2/(i3*i4)`,
/// whose integers are integer-feasible at the node but whose McCormick envelope
/// is not tight). Without this the node is fathomed with no incumbent and a
/// feasible problem can be falsely certified infeasible / falsely optimal.
///
/// Only columns flagged in `spatial_int_cols` (integer variables appearing in a
/// nonlinear term) are eligible. A column qualifies only when its integer domain
/// can still split into two non-empty halves (`ub - lb >= 1`). Returns
/// `(var_index, branch_point, relative_width)` with disjoint children `[lb, bp]`
/// and `[bp+1, ub]`.
///
/// The returned `relative_width` (current width / global width) lets the caller
/// compare an integer candidate against a continuous one *on equal footing*:
/// nonconvex spatial B&B branches on whichever dimension — integer or continuous
/// — has shrunk least relative to its root domain. This is essential on the
/// welded-beam class (nvs05), where the objective gap is driven by an integer
/// product `i1*i2` with `i1,i2 ∈ [1,200]`: if integers were always relegated
/// behind continuous bisection, the continuous variables (infinitely bisectable)
/// would never exhaust and the integer partition would never fire, freezing the
/// bound.
pub fn select_spatial_integer_branch_variable(
    node_lb: &[f64],
    node_ub: &[f64],
    global_lb: &[f64],
    global_ub: &[f64],
    spatial_int_cols: &[bool],
) -> Option<(usize, f64, f64)> {
    let n = node_lb.len();
    let mut best_idx: Option<usize> = None;
    let mut best_rel_width = 0.0_f64;

    for idx in 0..n {
        if idx >= spatial_int_cols.len() || !spatial_int_cols[idx] {
            continue;
        }
        let width = node_ub[idx] - node_lb[idx];
        if width < 1.0 - 1e-9 {
            continue; // Cannot split into two non-empty integer halves.
        }
        let global_width = global_ub[idx] - global_lb[idx];
        if global_width < 1e-15 {
            continue; // Fixed at root, skip.
        }
        let relative_width = width / global_width;
        if !relative_width.is_finite() {
            // Unbounded integer dimension (infinite node or global width): not
            // spatially branchable. Skip deterministically so a NaN/inf relative
            // width cannot defeat the `> best_rel_width` selection (issue #467).
            continue;
        }
        if relative_width > best_rel_width {
            best_rel_width = relative_width;
            best_idx = Some(idx);
        }
    }

    best_idx.map(|idx| {
        let mut bp = (0.5 * (node_lb[idx] + node_ub[idx])).floor();
        if bp >= node_ub[idx] {
            bp = node_ub[idx] - 1.0;
        }
        if bp < node_lb[idx] {
            bp = node_lb[idx];
        }
        (idx, bp, best_rel_width)
    })
}

/// Whether every spatially-relevant dimension of this node's box has finite
/// width (issue #1301).
///
/// [`select_spatial_branch_variable`] and [`select_spatial_integer_branch_variable`]
/// both deliberately skip a dimension whose width is infinite (or whose global
/// range is infinite, making the relative width non-finite) rather than treat
/// it as a branch candidate — an infinite interval cannot be shrunk by
/// bisection. That skip is correct for branching, but a caller that sees "no
/// branch direction found" cannot conclude the box is *tight*: an unbounded
/// dimension is skipped for the opposite reason a tight one is. Conflating the
/// two let a node with a genuinely open, unbounded dimension be fathomed as
/// resolved whenever its relaxation happened to report a finite bound —
/// `steenbrf` (468 unbounded variables) certified `optimal` at 398.36 against
/// a true optimum of 282.68 this way. This function is the discriminator: it
/// returns `false` (not tight) the moment any dimension select_spatial_*
/// would have skipped is unbounded, so the caller can tell "unbranchable
/// because unbounded" apart from "unbranchable because tight".
///
/// Covers exactly the dimensions the two selectors consider: continuous
/// (non-integer) variables, independent or dependent, plus any integer column
/// marked in `spatial_int_cols`. An ordinary integer variable outside
/// `spatial_int_cols` is excluded — its box width does not bear on whether the
/// *relaxation* over this box is exact, only its own (separately handled)
/// fractional branching does.
pub fn spatial_box_is_tight(
    node_lb: &[f64],
    node_ub: &[f64],
    integer_vars: &[VarBranchInfo],
    spatial_int_cols: &[bool],
) -> bool {
    let n = node_lb.len();
    let mut is_int = vec![false; n];
    for var in integer_vars {
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx < n {
                is_int[idx] = true;
            }
        }
    }
    for idx in 0..n {
        let spatial_int_col = idx < spatial_int_cols.len() && spatial_int_cols[idx];
        if is_int[idx] && !spatial_int_col {
            continue; // Ordinary integer variable: handled by fractional branching.
        }
        let width = node_ub[idx] - node_lb[idx];
        if !width.is_finite() {
            return false;
        }
    }
    true
}

/// #1869: pick an UNBOUNDED spatially-relevant dimension and a FINITE point to split
/// it at, or `None` when every such dimension has finite width.
///
/// The spatial selectors skip a dimension of infinite width (#467: bisecting an
/// infinite interval never shrinks it), so a node whose only open direction is
/// unbounded used to be fathomed with the tree bound pinned at `-inf`. Splitting at
/// a finite point instead gives one child a finite box, which a relaxation can
/// bound, and leaves the other unbounded, so it keeps its inherited bound and the
/// tree can never certify through it until a relaxation bounds it. The point grows
/// geometrically on re-splits of the unbounded child, `l + 2 max(1, |l|, |x|)` for
/// `[l, inf)` (mirrored for `(-inf, u]`; the relaxation value, or 0, for a free
/// column), so the finite children cover any bounded region in finitely many
/// splits. Returns `(index, point, is_integer)`; an integer column's point is an
/// integer and the caller splits it disjointly. Among candidates the one with the
/// largest finite relaxation magnitude is chosen (lowest index on ties).
pub fn select_unbounded_branch(
    node_lb: &[f64],
    node_ub: &[f64],
    integer_vars: &[VarBranchInfo],
    spatial_int_cols: &[bool],
    solution: &[f64],
) -> Option<(usize, f64, bool)> {
    let n = node_lb.len();
    let mut is_int = vec![false; n];
    for var in integer_vars {
        for i in 0..var.size {
            let idx = var.offset + i;
            if idx < n {
                is_int[idx] = true;
            }
        }
    }
    let mut best: Option<(usize, f64)> = None;
    for idx in 0..n {
        let spatial_int_col = idx < spatial_int_cols.len() && spatial_int_cols[idx];
        if is_int[idx] && !spatial_int_col {
            continue;
        }
        if (node_ub[idx] - node_lb[idx]).is_finite() {
            continue;
        }
        let mag = solution
            .get(idx)
            .copied()
            .filter(|v| v.is_finite())
            .map(f64::abs)
            .unwrap_or(0.0);
        if best.is_none_or(|(_, m)| mag > m) {
            best = Some((idx, mag));
        }
    }
    let (idx, mag) = best?;
    let (lo, hi) = (node_lb[idx], node_ub[idx]);
    let x = solution.get(idx).copied().filter(|v| v.is_finite());
    let mut bp = if lo.is_finite() {
        lo + 2.0 * 1f64.max(lo.abs()).max(mag)
    } else if hi.is_finite() {
        hi - 2.0 * 1f64.max(hi.abs()).max(mag)
    } else {
        x.unwrap_or(0.0)
    };
    if !bp.is_finite() {
        return None;
    }
    if is_int[idx] {
        bp = bp.floor();
        // The disjoint split is `<= bp` / `>= bp + 1`; both sides must be non-empty.
        if bp < lo || bp + 1.0 > hi {
            return None;
        }
    }
    Some((idx, bp, is_int[idx]))
}

/// Create two children partitioning an integer variable's domain disjointly:
/// left `x_idx <= bp`, right `x_idx >= bp + 1`. Unlike
/// [`create_children_spatial`] (shared branch point, valid for continuous
/// bisection), integer ranges must not overlap or the right child could fail to
/// shrink and the search would loop.
pub fn create_children_spatial_int(
    parent: &Node,
    idx: usize,
    bp: f64,
    mut next_id: impl FnMut() -> NodeId,
) -> (Node, Node) {
    let parent_sol = parent.parent_solution.clone();
    let inherited_lb = parent.local_lower_bound;

    let mut left_ub = parent.ub.clone();
    left_ub[idx] = bp;
    let left = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        branch: Some(BranchRecord {
            var: idx,
            up: false,
            value: bp,
        }),
        lb: parent.lb.clone(),
        ub: left_ub,
        local_lower_bound: inherited_lb,
        best_estimate: inherited_lb,
        status: NodeStatus::Pending,
        parent_solution: parent_sol.clone(),
        basis: None,
    };

    let mut right_lb = parent.lb.clone();
    right_lb[idx] = bp + 1.0;
    let right = Node {
        id: next_id(),
        parent: Some(parent.id),
        depth: parent.depth + 1,
        branch: Some(BranchRecord {
            var: idx,
            up: true,
            value: bp + 1.0,
        }),
        lb: right_lb,
        ub: parent.ub.clone(),
        local_lower_bound: inherited_lb,
        best_estimate: inherited_lb,
        status: NodeStatus::Pending,
        parent_solution: parent_sol,
        basis: None,
    };

    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::basis::Basis;

    #[test]
    fn children_inherit_parent_basis() {
        // A parent carrying an optimal basis branches; both children must
        // inherit that basis as their dual-simplex warm-start state.
        let mut parent = Node::new(NodeId(0), None, 0, vec![0.0, 0.0], vec![5.0, 5.0]);
        let basis = Basis::from_basic(4, vec![2, 3]);
        parent.basis = Some(basis.clone());
        parent.local_lower_bound = -10.0;
        let decision = BranchDecision {
            var_index: 0,
            branch_point: 2.0,
        };
        let mut counter = 1usize;
        let (left, right) = create_children(&parent, &decision, || {
            let id = NodeId(counter);
            counter += 1;
            id
        });
        assert_eq!(left.basis.as_ref(), Some(&basis));
        assert_eq!(right.basis.as_ref(), Some(&basis));
    }

    #[test]
    fn test_most_fractional_selects_closest_to_half() {
        // x0=0.3 (frac=0.3), x1=0.5 (frac=0.5), x2=0.9 (frac=0.9)
        let solution = vec![0.3, 0.5, 0.9];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        let decision = select_branch_variable(&solution, &vars).unwrap();
        assert_eq!(decision.var_index, 1); // x1=0.5 is most fractional
        assert_eq!(decision.branch_point, 0.0); // floor(0.5) = 0
    }

    #[test]
    fn test_no_branch_when_all_integral() {
        let solution = vec![1.0, 2.0, 3.0];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        assert!(select_branch_variable(&solution, &vars).is_none());
    }

    #[test]
    fn test_no_branch_on_continuous() {
        let solution = vec![0.5, 0.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: false, // continuous, should be skipped
        }];
        assert!(select_branch_variable(&solution, &vars).is_none());
    }

    #[test]
    fn test_mixed_integer_continuous() {
        // x0 continuous=0.5, x1 integer=0.7
        let solution = vec![0.5, 0.7];
        let vars = vec![
            VarBranchInfo {
                offset: 0,
                size: 1,
                is_integer: false,
            },
            VarBranchInfo {
                offset: 1,
                size: 1,
                is_integer: true,
            },
        ];
        let decision = select_branch_variable(&solution, &vars).unwrap();
        assert_eq!(decision.var_index, 1); // only x1 is integer
        assert_eq!(decision.branch_point, 0.0); // floor(0.7)
    }

    #[test]
    fn test_near_integral_skipped() {
        let solution = vec![1.0 + 1e-7]; // nearly integral
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 1,
            is_integer: true,
        }];
        assert!(select_branch_variable(&solution, &vars).is_none());
    }

    #[test]
    fn test_is_integer_feasible_true() {
        let solution = vec![1.0, 2.0, 0.0];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        assert!(is_integer_feasible(&solution, &vars));
    }

    #[test]
    fn test_is_integer_feasible_false() {
        let solution = vec![1.0, 2.5, 0.0];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        assert!(!is_integer_feasible(&solution, &vars));
    }

    #[test]
    fn test_is_integer_feasible_continuous_ignored() {
        let solution = vec![1.5, 2.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: false,
        }];
        assert!(is_integer_feasible(&solution, &vars));
    }

    #[test]
    fn test_create_children_bounds() {
        let parent = Node::new(NodeId(0), None, 0, vec![0.0, 0.0], vec![10.0, 10.0]);
        let decision = BranchDecision {
            var_index: 0,
            branch_point: 3.0, // branching x0 at floor(3.7) = 3
        };
        let mut counter = 1usize;
        let (left, right) = create_children(&parent, &decision, || {
            let id = NodeId(counter);
            counter += 1;
            id
        });

        // Left: x0 <= 3
        assert_eq!(left.id, NodeId(1));
        assert_eq!(left.parent, Some(NodeId(0)));
        assert_eq!(left.depth, 1);
        assert_eq!(left.lb, vec![0.0, 0.0]);
        assert_eq!(left.ub, vec![3.0, 10.0]);
        assert_eq!(left.status, NodeStatus::Pending);

        // Right: x0 >= 4
        assert_eq!(right.id, NodeId(2));
        assert_eq!(right.parent, Some(NodeId(0)));
        assert_eq!(right.depth, 1);
        assert_eq!(right.lb, vec![4.0, 0.0]);
        assert_eq!(right.ub, vec![10.0, 10.0]);
        assert_eq!(right.status, NodeStatus::Pending);
    }

    // ----- Pseudocost tests -----

    #[test]
    fn test_pseudocosts_default_cost() {
        let pc = Pseudocosts::new(3);
        // No observations → default cost of 1.0.
        assert_eq!(pc.down_cost(0), 1.0);
        assert_eq!(pc.up_cost(0), 1.0);
        assert_eq!(pc.observation_count(0), 0);
    }

    #[test]
    fn test_pseudocosts_update_down() {
        let mut pc = Pseudocosts::new(3);
        // Branch on var 0 with frac=0.3, parent_lb=1.0, child_lb=2.0
        // delta_bound = 1.0, delta_var = 0.3, per-unit = 1.0/0.3 ≈ 3.333
        pc.update(0, 1.0, 2.0, 0.3, true);
        assert_eq!(pc.observation_count(0), 1);
        let cost = pc.down_cost(0);
        assert!((cost - 1.0 / 0.3).abs() < 1e-10);
        // Up cost should still be default.
        assert_eq!(pc.up_cost(0), 1.0);
    }

    #[test]
    fn test_pseudocosts_update_up() {
        let mut pc = Pseudocosts::new(3);
        // Branch on var 1 with frac=0.7, parent_lb=5.0, child_lb=6.5
        // delta_bound = 1.5, delta_var = 1-0.7 = 0.3, per-unit = 5.0
        pc.update(1, 5.0, 6.5, 0.7, false);
        assert_eq!(pc.observation_count(1), 1);
        let cost = pc.up_cost(1);
        assert!((cost - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pseudocosts_average_multiple_observations() {
        let mut pc = Pseudocosts::new(2);
        // Two down observations on var 0:
        // obs1: delta_bound=2.0, frac=0.5 → per-unit = 4.0
        // obs2: delta_bound=1.0, frac=0.5 → per-unit = 2.0
        // average = 3.0
        pc.update(0, 0.0, 2.0, 0.5, true);
        pc.update(0, 1.0, 2.0, 0.5, true);
        assert_eq!(pc.observation_count(0), 2);
        assert!((pc.down_cost(0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pseudocosts_negative_delta_clamped() {
        let mut pc = Pseudocosts::new(2);
        // child_lb < parent_lb → delta clamped to 0.
        pc.update(0, 5.0, 3.0, 0.5, true);
        assert!((pc.down_cost(0)).abs() < 1e-10);
    }

    #[test]
    fn test_pseudocosts_score_product() {
        let mut pc = Pseudocosts::new(2);
        // Set known costs: down=2.0, up=3.0 for var 0.
        pc.update(0, 0.0, 1.0, 0.5, true); // per-unit = 2.0
        pc.update(0, 0.0, 1.5, 0.5, false); // per-unit = 3.0
        let score = pc.score(0, 0.5);
        // d = 2.0 * 0.5 = 1.0, u = 3.0 * 0.5 = 1.5
        // score = (1e-6 + 1.0) * (1e-6 + 1.5) ≈ 1.5
        assert!((score - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_pseudocosts_out_of_range() {
        let pc = Pseudocosts::new(2);
        // Out-of-range variable index → default.
        assert_eq!(pc.down_cost(99), 1.0);
        assert_eq!(pc.up_cost(99), 1.0);
        assert_eq!(pc.observation_count(99), 0);
    }

    #[test]
    fn test_select_branch_variable_pseudocost_basic() {
        let solution = vec![0.3, 0.7, 0.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 3,
            is_integer: true,
        }];
        let mut pc = Pseudocosts::new(3);
        // Give var 2 a high pseudocost.
        pc.update(2, 0.0, 10.0, 0.5, true); // per-unit = 20.0
        pc.update(2, 0.0, 10.0, 0.5, false); // per-unit = 20.0

        let (decision, _unreliable) = select_branch_variable_pseudocost(&solution, &vars, &pc, 8);
        let d = decision.unwrap();
        assert_eq!(d.var_index, 2); // var 2 has highest score
        assert_eq!(d.branch_point, 0.0);
    }

    #[test]
    fn test_select_branch_variable_pseudocost_unreliable() {
        let solution = vec![0.5, 0.5];
        let vars = vec![VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: true,
        }];
        let pc = Pseudocosts::new(2); // no observations
        let (_decision, unreliable) = select_branch_variable_pseudocost(&solution, &vars, &pc, 8);
        // Both variables have 0 observations < 8 threshold → unreliable.
        assert_eq!(unreliable.len(), 2);
        assert!(unreliable.contains(&0));
        assert!(unreliable.contains(&1));
    }

    #[test]
    fn test_create_children_second_variable() {
        let parent = Node::new(
            NodeId(5),
            Some(NodeId(2)),
            3,
            vec![1.0, 2.0, 0.0],
            vec![5.0, 8.0, 1.0],
        );
        let decision = BranchDecision {
            var_index: 1,
            branch_point: 4.0,
        };
        let mut counter = 10usize;
        let (left, right) = create_children(&parent, &decision, || {
            let id = NodeId(counter);
            counter += 1;
            id
        });

        // Left: x1 <= 4
        assert_eq!(left.ub[1], 4.0);
        assert_eq!(left.lb[1], 2.0); // unchanged

        // Right: x1 >= 5
        assert_eq!(right.lb[1], 5.0);
        assert_eq!(right.ub[1], 8.0); // unchanged
    }

    #[test]
    fn test_spatial_branch_skips_unbounded_dimension() {
        // Issue #467: an unbounded continuous dimension yields a NaN relative
        // width (inf/inf) that silently defeats both filters; it must be skipped,
        // and a coexisting FINITE dimension must still be selected.
        // x0 = [-5, inf] (unbounded), x1 = [-5, 5] (finite).
        let node_lb = vec![-5.0, -5.0];
        let node_ub = vec![f64::INFINITY, 5.0];
        let global_lb = vec![-5.0, -5.0];
        let global_ub = vec![f64::INFINITY, 5.0];
        let sel = select_spatial_branch_variable(
            &node_lb,
            &node_ub,
            &global_lb,
            &global_ub,
            &[],
            &[],
            false,
        );
        assert!(sel.is_some(), "the finite dimension x1 must be branchable");
        assert_eq!(
            sel.unwrap().0.var_index,
            1,
            "must branch the finite dim, not the unbounded one"
        );

        // Both dimensions unbounded → no branchable dimension at all.
        let none = select_spatial_branch_variable(
            &[-5.0, f64::NEG_INFINITY],
            &[f64::INFINITY, 5.0],
            &[-5.0, f64::NEG_INFINITY],
            &[f64::INFINITY, 5.0],
            &[],
            &[],
            false,
        );
        assert!(
            none.is_none(),
            "an all-unbounded box has no spatial branch direction"
        );
    }

    #[test]
    fn test_spatial_integer_branch_skips_unbounded_dimension() {
        // Issue #467: an unbounded integer dimension must not be selected via a
        // NaN/inf relative width.
        let cols = vec![true];
        let none = select_spatial_integer_branch_variable(
            &[0.0],
            &[f64::INFINITY],
            &[0.0],
            &[f64::INFINITY],
            &cols,
        );
        assert!(
            none.is_none(),
            "an unbounded integer dimension is not branchable"
        );
    }

    // #1869: an unbounded dimension gets a FINITE split point inside its domain.
    #[test]
    fn unbounded_branch_point_is_finite_and_inside() {
        let none: Vec<VarBranchInfo> = vec![];
        // [2, inf): l + 2 max(1, |l|, |x|) with x = 3 -> 2 + 6 = 8.
        let (i, bp, int) =
            select_unbounded_branch(&[0.0, 2.0], &[1.0, f64::INFINITY], &none, &[], &[0.5, 3.0])
                .expect("an unbounded column");
        assert_eq!((i, int), (1, false));
        assert!((bp - 8.0).abs() < 1e-12 && bp > 2.0, "{bp}");
        // (-inf, -4]: mirrored, u - 2 max(1, |u|, |x|) = -4 - 8 = -12.
        let (_, bp, _) =
            select_unbounded_branch(&[f64::NEG_INFINITY], &[-4.0], &none, &[], &[f64::NAN])
                .unwrap();
        assert!((bp + 12.0).abs() < 1e-12, "{bp}");
        // Free column: the relaxation value (0 when it is not finite).
        let (_, bp, _) =
            select_unbounded_branch(&[f64::NEG_INFINITY], &[f64::INFINITY], &none, &[], &[-7.5])
                .unwrap();
        assert_eq!(bp, -7.5);
        let (_, bp, _) = select_unbounded_branch(
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            &none,
            &[],
            &[f64::INFINITY],
        )
        .unwrap();
        assert_eq!(bp, 0.0);
        // All finite: nothing to split.
        assert!(select_unbounded_branch(&[0.0], &[1.0], &none, &[], &[0.5]).is_none());
    }

    #[test]
    fn unbounded_branch_skips_ordinary_integers_and_floors_spatial_ones() {
        let ints = vec![VarBranchInfo {
            offset: 0,
            size: 1,
            is_integer: true,
        }];
        // An ordinary integer column is the fractional brancher's, not this one's.
        assert!(
            select_unbounded_branch(&[0.0], &[f64::INFINITY], &ints, &[false], &[2.5]).is_none()
        );
        // A spatial-integer column gets an integer point, split disjointly.
        let (i, bp, int) =
            select_unbounded_branch(&[0.5], &[f64::INFINITY], &ints, &[true], &[2.5]).unwrap();
        assert_eq!((i, int), (0, true));
        assert_eq!(bp, bp.floor());
        assert!(bp >= 0.5);
    }
}

#[cfg(test)]
mod tests_1978 {
    //! #1978: the Python tree's spatial score rule (`select_spatial_score_variable`).
    use super::*;

    fn pick(
        lb: &[f64],
        ub: &[f64],
        x: &[f64],
        terms: &[ScoreTerm],
        ints: &[VarBranchInfo],
    ) -> Option<(BranchDecision, f64)> {
        select_spatial_score_variable(lb, ub, lb, ub, x, ints, &[], terms)
    }

    /// The #1978 shape: a narrow factor `u in [0, 5]` against a wide partner
    /// `x in [0, 7000]`, plus a wide column that only appears linearly. The legacy
    /// longest-edge rule ties every column at relative width 1 and takes the
    /// lowest index (the linear one); the score rule takes a term operand, and of
    /// those the narrow factor shared by two terms.
    #[test]
    fn score_rule_prefers_the_shared_narrow_factor_over_linear_columns() {
        // cols: 0 = linear-only, 1 = u, 2 = x_a, 3 = x_b
        let lb = [0.0, 0.0, 0.0, 0.0];
        let ub = [9000.0, 5.0, 7000.0, 7000.0];
        let x = [100.0, 2.3, 3000.0, 2500.0];
        let terms = [ScoreTerm::Bilinear(1, 2), ScoreTerm::Bilinear(1, 3)];
        let legacy = select_spatial_branch_variable(&lb, &ub, &lb, &ub, &[], &[], false);
        assert_eq!(
            legacy.unwrap().0.var_index,
            0,
            "control: legacy splits the linear column"
        );
        let (d, _) = pick(&lb, &ub, &x, &terms, &[]).expect("a term has a gap");
        assert_eq!(d.var_index, 1, "the score rule splits u: {d:?}");
        // The split point is the midpoint (SCORE_MIDPULL = 1), strictly inside.
        let want = 2.5;
        assert!((d.branch_point - want).abs() < 1e-12, "{}", d.branch_point);
        assert!(d.branch_point > lb[1] && d.branch_point < ub[1]);
    }

    /// No gap (the point sits on a box corner of every term), integer operands, and
    /// deprioritized operands all leave nothing to score: `None`, so the caller's
    /// legacy fallback keeps the search complete.
    #[test]
    fn score_rule_declines_when_nothing_scores() {
        let lb = [0.0, 0.0];
        let ub = [4.0, 4.0];
        let terms = [ScoreTerm::Bilinear(0, 1)];
        assert!(
            pick(&lb, &ub, &[0.0, 4.0], &terms, &[]).is_none(),
            "corner: no gap"
        );
        let ints = [VarBranchInfo {
            offset: 0,
            size: 2,
            is_integer: true,
        }];
        assert!(
            pick(&lb, &ub, &[1.0, 2.0], &terms, &ints).is_none(),
            "integers are not candidates"
        );
        let dep = select_spatial_score_variable(
            &lb,
            &ub,
            &lb,
            &ub,
            &[1.0, 2.0],
            &[],
            &[true, true],
            &terms,
        );
        assert!(dep.is_none(), "deprioritized operands are not candidates");
        // An unbounded side has no envelope gap to measure.
        let inf_ub = [4.0, f64::INFINITY];
        assert!(pick(&lb, &inf_ub, &[1.0, 2.0], &terms, &[]).is_none());
    }

    /// The bilinear gap is the McCormick under- plus over-estimation at the point;
    /// a square and a power use their secant gap.
    #[test]
    fn score_term_gaps_are_the_envelope_gaps() {
        let (lb, ub) = ([0.0, 0.0], [2.0, 4.0]);
        // x0 = 1, x1 = 1: under = min(1*1, 1*3) = 1, over = min(1*3, 1*1) = 1.
        let g = score_term_gap(&ScoreTerm::Bilinear(0, 1), &[1.0, 1.0], &lb, &ub).unwrap();
        assert!((g - 2.0).abs() < 1e-15, "{g}");
        let sq = score_term_gap(&ScoreTerm::Bilinear(0, 0), &[0.5, 0.0], &lb, &ub).unwrap();
        assert!((sq - 0.75).abs() < 1e-15, "(x-l)(u-x) = 0.5*1.5: {sq}");
        let p3 = score_term_gap(&ScoreTerm::Power(1, 3.0), &[0.0, 2.0], &lb, &ub).unwrap();
        // secant of x^3 on [0, 4] at 2 is 32, x^3 = 8.
        assert!((p3 - 24.0).abs() < 1e-12, "{p3}");
        // A non-integer power on a box reaching below 0 has no real envelope.
        assert!(
            score_term_gap(&ScoreTerm::Power(0, 0.5), &[1.0, 0.0], &[-1.0, 0.0], &ub).is_none()
        );
    }

    /// Branching order only: the two children of a score pick cover the parent box.
    #[test]
    fn score_pick_children_cover_the_parent() {
        let lb = [0.0, 0.0, 0.0];
        let ub = [1.0, 5.0, 7000.0];
        let x = [0.5, 1.0, 6999.0];
        let terms = [ScoreTerm::Bilinear(1, 2), ScoreTerm::Power(0, 2.0)];
        let (d, _) = pick(&lb, &ub, &x, &terms, &[]).unwrap();
        let i = d.var_index;
        assert!(lb[i] < d.branch_point && d.branch_point < ub[i]);
        // [lb, bp] U [bp, ub] = [lb, ub]: create_children_spatial splits at bp.
        let parent = Node {
            id: NodeId(0),
            parent: None,
            depth: 0,
            branch: None,
            lb: lb.to_vec(),
            ub: ub.to_vec(),
            local_lower_bound: 0.0,
            best_estimate: 0.0,
            status: NodeStatus::Pending,
            parent_solution: None,
            basis: None,
        };
        let mut next = 1usize;
        let (l, r) = create_children_spatial(&parent, &d, || {
            next += 1;
            NodeId(next)
        });
        assert_eq!(
            (l.lb[i], l.ub[i], r.lb[i], r.ub[i]),
            (lb[i], d.branch_point, d.branch_point, ub[i])
        );
    }
}
