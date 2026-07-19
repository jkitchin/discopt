//! Native spatial branch-and-bound tree loop (issue #764, C1 build-order item 4).
//!
//! Composes [`solve_spatial_node`] into a full spatial B&B: pop a box, solve the
//! node relaxation (rigorous safe bound), OBBT-tighten the box, and either accept a
//! feasible point as the incumbent or branch. Runs entirely in `discopt-core`.
//!
//! Soundness (the non-negotiable, correctness-first contract):
//! * **Pruning** uses only the [`ns_safe_bound_csc`](crate::lp::simplex::refine)
//!   safe lower bound (`<=` the true node optimum), so a node is fathomed only when
//!   its relaxation *provably* cannot beat the incumbent.
//! * **Incumbent acceptance** requires a *sufficient* feasibility condition — every
//!   integer candidate integral AND every lifted term tight (`|x_aux − f(operands)|
//!   <= mccormick_tol`, i.e. the McCormick relaxation is exact at the point) — so an
//!   accepted point is genuinely feasible for the original nonconvex problem and its
//!   linear objective value is valid. A looser check could bless an infeasible point
//!   (`incorrect_count > 0`); this never does.
//! * **Branching** partitions the box into two covering children (`ub = p` and
//!   `lo = p` for a split point `p in [lo,hi]`), whose union is the parent — so no
//!   feasible point is ever lost.

use crate::bnb::spatial_kernel::{solve_spatial_node, EnvTerm, SpatialKernelSpec};
use crate::lp::simplex::{LpStatus, SimplexOptions};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// An open B&B node in the best-bound frontier: its box and the inherited lower
/// bound `pb` (the parent's rigorous bound, a valid lower bound for this region).
/// Ordered so a max-heap yields the SMALLEST `pb` first (best-bound search): the
/// lowest-bound region is explored + tightened first, which is what lifts the global
/// frontier minimum — pure DFS leaves low-bound siblings unexplored and the reported
/// bound stuck at the root value.
struct QNode {
    pb: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

impl PartialEq for QNode {
    fn eq(&self, o: &Self) -> bool {
        self.pb == o.pb
    }
}
impl Eq for QNode {}
impl PartialOrd for QNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for QNode {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reverse: BinaryHeap is a max-heap, we want the min `pb` on top.
        o.pb.partial_cmp(&self.pb).unwrap_or(Ordering::Equal)
    }
}

/// Termination status of the tree solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeStatus {
    /// The gap closed to `gap_tol` (global bound met the incumbent).
    Optimal,
    /// The node budget was exhausted with the gap still open.
    NodeLimit,
    /// No feasible point exists (the root relaxation was infeasible).
    Infeasible,
}

/// Tunables for [`solve_spatial_tree`].
#[derive(Clone, Copy, Debug)]
pub struct SpatialTreeConfig {
    /// Maximum nodes to process before returning [`TreeStatus::NodeLimit`].
    pub max_nodes: usize,
    /// Absolute gap `incumbent − global_bound` at/below which the solve stops.
    pub gap_tol: f64,
    /// Integrality tolerance for incumbent acceptance / integer branching.
    pub int_tol: f64,
    /// McCormick-exactness tolerance for incumbent acceptance (`|x_aux − f|`).
    pub mccormick_tol: f64,
    /// Minimum box width worth spatial-branching (avoids infinite splitting).
    pub min_box_width: f64,
    /// Whether to run the in-kernel OBBT sweep at each node.
    pub run_obbt: bool,
}

impl Default for SpatialTreeConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            gap_tol: 1e-6,
            int_tol: 1e-5,
            mccormick_tol: 1e-6,
            min_box_width: 1e-9,
            run_obbt: true,
        }
    }
}

/// Result of the tree solve.
#[derive(Clone, Debug)]
pub struct SpatialTreeResult {
    /// Termination status.
    pub status: TreeStatus,
    /// Best feasible objective found, or `None` if no incumbent was accepted.
    pub incumbent: Option<f64>,
    /// The incumbent point (structural columns), empty if none.
    pub incumbent_x: Vec<f64>,
    /// Global lower bound (min sense): `<=` the true optimum.
    pub bound: f64,
    /// Nodes processed.
    pub node_count: usize,
    /// Total LP solves across all nodes (relaxation + OBBT probes).
    pub n_lp_solves: usize,
}

/// True value of a lifted term at the point `x` (structural columns), for the
/// McCormick-exactness feasibility test. `None` for a sqrt of a negative argument
/// (infeasible point — never accepted).
fn term_true_value(t: &EnvTerm, x: &[f64]) -> Option<f64> {
    Some(match *t {
        EnvTerm::Bilinear { i, j, .. } => x[i] * x[j],
        EnvTerm::Monomial { i, p, .. } => x[i].powi(p),
        EnvTerm::AffineSquare { j, coeff, cst, .. } => {
            let t = coeff * x[j] + cst;
            t * t
        }
        EnvTerm::Sqrt { x: xc, coeff, cst, .. } => {
            let arg = coeff * x[xc] + cst;
            if arg < 0.0 {
                return None;
            }
            arg.sqrt()
        }
    })
}

/// The operand column a spatial branch on this term should split, and the term's
/// current McCormick gap `|x_aux − f(operands)|`. For a bilinear term we split the
/// operand with the wider box (passed in via `width`); for the 1-D terms the sole
/// operand.
fn term_gap_and_branch_col(t: &EnvTerm, x: &[f64], width: &dyn Fn(usize) -> f64) -> (f64, usize) {
    let aux = t.aux_col();
    let true_v = term_true_value(t, x);
    let gap = match true_v {
        Some(v) => (x[aux] - v).abs(),
        None => f64::INFINITY, // infeasible operand: force a branch here
    };
    let col = match *t {
        EnvTerm::Bilinear { i, j, .. } => {
            if width(i) >= width(j) {
                i
            } else {
                j
            }
        }
        EnvTerm::Monomial { i, .. } => i,
        EnvTerm::AffineSquare { j, .. } => j,
        EnvTerm::Sqrt { x: xc, .. } => xc,
    };
    (gap, col)
}

/// Solve `spec` by native spatial branch-and-bound. `spec.global_lo/global_hi` is
/// the root box; `spec.integrality` marks integer columns; `spec.obbt_candidates`
/// (if `config.run_obbt`) are probed per node.
pub fn solve_spatial_tree(
    spec: &SpatialKernelSpec,
    config: &SpatialTreeConfig,
    opts: &SimplexOptions,
) -> SpatialTreeResult {
    // Best-bound frontier: a min-heap on the inherited lower bound. Exploring the
    // lowest-bound region first lifts the global frontier minimum (the reported dual
    // bound) as fast as possible — the key to certifying instances like tanksize
    // whose bound climbs only slowly.
    let mut heap: BinaryHeap<QNode> = BinaryHeap::new();
    heap.push(QNode {
        pb: f64::NEG_INFINITY,
        lo: spec.global_lo.clone(),
        hi: spec.global_hi.clone(),
    });
    let mut incumbent: Option<f64> = None;
    let mut incumbent_x: Vec<f64> = Vec::new();
    let mut node_count = 0usize;
    let mut n_lp_solves = 0usize;

    // Global lower bound = min, over every region that leaves the tree WITHOUT being
    // subdivided (pruned / infeasible / feasible-leaf / width-exhausted), of a valid
    // lower bound for that region. Each contribution is a safe bound (`<=` the
    // region's true optimum), so the accumulated min is `<=` the true global optimum
    // — a rigorous lower bound. Branched regions contribute nothing (their children
    // do). Open frontier nodes carry their `pb` as a valid region lower bound; under
    // best-bound the heap top is exactly that frontier minimum.
    let mut global_lb_closed = f64::INFINITY;

    while let Some(QNode { pb: parent_bound, lo, hi }) = heap.pop() {
        // Fathom by the parent bound if the incumbent already dominates it. The
        // region's valid lower bound is `parent_bound`.
        if let Some(inc) = incumbent {
            if parent_bound >= inc - config.gap_tol {
                global_lb_closed = global_lb_closed.min(parent_bound);
                continue;
            }
        }
        if node_count >= config.max_nodes {
            // Global bound = min(closed regions, open frontier, this unpopped node).
            // Under best-bound the heap top is the frontier minimum, but take the full
            // min defensively.
            let frontier = heap.iter().map(|n| n.pb).fold(f64::INFINITY, f64::min);
            let gb = global_lb_closed.min(frontier).min(parent_bound);
            return SpatialTreeResult {
                status: TreeStatus::NodeLimit,
                incumbent,
                incumbent_x,
                bound: gb,
                node_count,
                n_lp_solves,
            };
        }
        node_count += 1;

        let node = solve_spatial_node(spec, &lo, &hi, config.run_obbt, opts);
        n_lp_solves += node.n_lp_solves;

        // Infeasible node: empty region, contributes +inf (nothing).
        if node.status != LpStatus::Optimal {
            continue;
        }
        // Rigorous safe lower bound for this region, inheriting the parent's bound as
        // a floor: `parent_bound` is a valid lower bound for the parent region, which
        // CONTAINS this child, so the child's region optimum is `>= parent_bound`.
        // Taking the max keeps the bound finite and monotone when the node's own safe
        // bound is looser or uncertifiable (`-inf` — e.g. non-finite duals from an
        // ill-conditioned McCormick LP on tanksize), which would otherwise poison the
        // global lower bound. Sound: `max(safe, parent)` is still `<=` the true region
        // optimum since both terms are.
        let bound = node.bound.max(parent_bound);
        // Fathom by bound vs incumbent. The region's valid lower bound is `bound`.
        if let Some(inc) = incumbent {
            if bound >= inc - config.gap_tol {
                global_lb_closed = global_lb_closed.min(bound);
                continue;
            }
        }

        // Apply OBBT-tightened bounds to the box (tighten-only, sound).
        let mut lo = lo;
        let mut hi = hi;
        if config.run_obbt {
            for (k, &cand) in spec.obbt_candidates.iter().enumerate() {
                if k < node.tightened.len() {
                    let (glo, ghi) = node.tightened[k];
                    lo[cand] = lo[cand].max(glo);
                    hi[cand] = hi[cand].min(ghi);
                }
            }
        }

        let x = &node.x;

        // --- Feasibility test (sufficient condition for a valid incumbent) --- //
        // (a) integer candidates integral.
        let mut int_ok = true;
        let mut frac_int: Option<usize> = None;
        for (j, &is_int) in spec.integrality.iter().enumerate().take(spec.n_cols) {
            if is_int && (x[j] - x[j].round()).abs() > config.int_tol {
                int_ok = false;
                frac_int = Some(j);
                break;
            }
        }
        // (b) every lifted term McCormick-tight — fixed-width EnvTerms and the
        //     affine-form product (BlfTerm) terms alike.
        let width = |c: usize| hi[c] - lo[c];
        let mut worst_gap = 0.0f64;
        let mut branch_col: Option<usize> = None;
        for t in &spec.terms {
            let (gap, col) = term_gap_and_branch_col(t, x, &width);
            if gap > worst_gap {
                worst_gap = gap;
                branch_col = Some(col);
            }
        }
        for t in &spec.blf_terms {
            let a_val = t.a_const + dot_form(&t.a_cols, &t.a_coeffs, x);
            let b_val = t.b_const + dot_form(&t.b_cols, &t.b_coeffs, x);
            let gap = (x[t.w] - a_val * b_val).abs();
            if gap > worst_gap {
                worst_gap = gap;
                // Spatial-branch the widest operand column across A ∪ B.
                let mut best = None;
                let mut best_w = -1.0f64;
                for &c in t.a_cols.iter().chain(t.b_cols.iter()) {
                    let cw = width(c);
                    if cw > best_w {
                        best_w = cw;
                        best = Some(c);
                    }
                }
                branch_col = best;
            }
        }
        let terms_tight = worst_gap <= config.mccormick_tol;

        if int_ok && terms_tight {
            // Feasible leaf: accept if it improves the incumbent. The objective is
            // linear over the (now-tight) lifted columns, so `cᵀx` is the true
            // objective at this feasible point. The region contributes the SAFE node
            // bound (`<=` its true optimum), NOT the feasible value cᵀx — an upper
            // bound would corrupt the global lower bound.
            let obj = dot(&spec.c, x);
            if incumbent.map(|inc| obj < inc - 1e-12).unwrap_or(true) {
                incumbent = Some(obj);
                incumbent_x = x[..spec.n_cols].to_vec();
            }
            global_lb_closed = global_lb_closed.min(bound);
            continue;
        }

        // --- Branch --- //
        // Prefer closing an integer infeasibility; else spatial-branch the worst
        // McCormick gap. Fall back to the widest branchable box if neither picks.
        let (split_col, split_at) = if let Some(j) = frac_int {
            (j, x[j].floor() + 0.5) // integer branch: <= floor, >= ceil
        } else if let Some(col) = branch_col {
            if width(col) <= config.min_box_width {
                // Term is un-splittable (already pinned) yet not tight — numerical;
                // accept the box as exhausted to avoid an infinite loop. The region's
                // valid lower bound is the safe node `bound`.
                global_lb_closed = global_lb_closed.min(bound);
                continue;
            }
            // Spatial branch at the LP value, pulled to the interior.
            let p = clamp_interior(x[col], lo[col], hi[col]);
            (col, p)
        } else {
            global_lb_closed = global_lb_closed.min(bound);
            continue;
        };

        // Two covering children.
        if spec.integrality[split_col] {
            // integer: child1 x<=floor, child2 x>=ceil
            let f = (split_at - 0.5).floor();
            let lo1 = lo.clone();
            let mut hi1 = hi.clone();
            hi1[split_col] = f;
            let mut lo2 = lo.clone();
            let hi2 = hi.clone();
            lo2[split_col] = f + 1.0;
            if hi1[split_col] >= lo1[split_col] - 1e-12 {
                heap.push(QNode { pb: bound, lo: lo1, hi: hi1 });
            }
            if hi2[split_col] >= lo2[split_col] - 1e-12 {
                heap.push(QNode { pb: bound, lo: lo2, hi: hi2 });
            }
        } else {
            let mut hi1 = hi.clone();
            hi1[split_col] = split_at;
            let mut lo2 = lo.clone();
            lo2[split_col] = split_at;
            heap.push(QNode { pb: bound, lo: lo.clone(), hi: hi1 });
            heap.push(QNode { pb: bound, lo: lo2, hi: hi.clone() });
        }
    }

    // Worklist empty: every region was explored. The reported bound is the min safe
    // bound over all closed regions — a rigorous global lower bound (`<=` the true
    // optimum), never the incumbent (an upper bound). It is `<=` the incumbent
    // because the feasible leaf that set the incumbent contributed its own safe bound.
    match incumbent {
        Some(inc) => SpatialTreeResult {
            status: TreeStatus::Optimal,
            incumbent,
            incumbent_x,
            // Clamp to the incumbent so a loose safe bound never reports a gap wider
            // than the true one; still `<=` the true optimum.
            bound: global_lb_closed.min(inc),
            node_count,
            n_lp_solves,
        },
        None => SpatialTreeResult {
            status: TreeStatus::Infeasible,
            incumbent: None,
            incumbent_x: Vec::new(),
            bound: f64::INFINITY,
            node_count,
            n_lp_solves,
        },
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `Σ coeffs[k] * x[cols[k]]` — the value of a sparse affine form's linear part.
fn dot_form(cols: &[usize], coeffs: &[f64], x: &[f64]) -> f64 {
    cols.iter().zip(coeffs.iter()).map(|(&c, &a)| a * x[c]).sum()
}

/// Pull a split point strictly inside `(lo, hi)` so both children are nonempty; if
/// `p` sits at a bound, use the midpoint.
fn clamp_interior(p: f64, lo: f64, hi: f64) -> f64 {
    let eps = 1e-9 * (1.0 + (hi - lo).abs());
    if p <= lo + eps || p >= hi - eps {
        0.5 * (lo + hi)
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bnb::spatial_kernel::FixedRow;

    // minimize w = x*y  s.t.  x + y >= 3,  x,y in [0,2].
    // True feasible region on [0,2]^2 with x+y>=3: the min of x*y is 2 (corners
    // (2,1),(1,2)); the interior x=y=1.5 gives 2.25 > 2. McCormick underestimates
    // at the root, so B&B must branch to certify 2.0.
    fn xy_min_spec() -> SpatialKernelSpec {
        SpatialKernelSpec {
            n_cols: 3,
            n_orig: 2,
            c: vec![0.0, 0.0, 1.0], // minimize w
            integrality: vec![false, false, false],
            global_lo: vec![0.0, 0.0, -1e20],
            global_hi: vec![2.0, 2.0, 1e20],
            // x + y >= 3  ==>  -x - y <= -3
            fixed_rows: vec![FixedRow {
                cols: vec![0, 1],
                coeffs: vec![-1.0, -1.0],
                rhs: -3.0,
            }],
            terms: vec![EnvTerm::Bilinear { i: 0, j: 1, w: 2 }],
            blf_terms: vec![],
            obbt_candidates: vec![0, 1],
        }
    }

    #[test]
    fn branches_to_certify_bilinear_min() {
        let spec = xy_min_spec();
        let opts = SimplexOptions::default();
        let cfg = SpatialTreeConfig {
            max_nodes: 5000,
            gap_tol: 1e-5,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &opts);
        assert_eq!(res.status, TreeStatus::Optimal, "did not converge: {res:?}");
        let inc = res.incumbent.expect("incumbent found");
        // Global optimum is 2.0.
        assert!((inc - 2.0).abs() < 1e-3, "incumbent {inc} != 2.0");
        // Soundness: global bound never above the true optimum.
        assert!(res.bound <= 2.0 + 1e-6, "bound {} above optimum 2.0", res.bound);
        // The incumbent point is feasible: x*y == w and x+y>=3.
        let x = &res.incumbent_x;
        assert!((x[0] * x[1] - x[2]).abs() < 1e-4, "w != x*y at incumbent");
        assert!(x[0] + x[1] >= 3.0 - 1e-4, "x+y>=3 violated");
    }

    // A pure integer-branch case: minimize -x s.t. x in [0,2] integer, no terms.
    // Optimum x=2, obj -2. Exercises integer branching + acceptance.
    #[test]
    fn integer_branch_finds_optimum() {
        let spec = SpatialKernelSpec {
            n_cols: 1,
            n_orig: 1,
            c: vec![-1.0],
            integrality: vec![true],
            global_lo: vec![0.0],
            global_hi: vec![2.0],
            fixed_rows: vec![],
            terms: vec![],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        let cfg = SpatialTreeConfig {
            run_obbt: false,
            ..SpatialTreeConfig::default()
        };
        let res = solve_spatial_tree(&spec, &cfg, &SimplexOptions::default());
        assert_eq!(res.status, TreeStatus::Optimal);
        assert!((res.incumbent.unwrap() - (-2.0)).abs() < 1e-9);
        assert!((res.incumbent_x[0] - 2.0).abs() < 1e-9);
    }
}
