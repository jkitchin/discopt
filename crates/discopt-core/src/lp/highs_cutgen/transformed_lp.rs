//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! Faithful port of `mip/HighsTransformedLp.cpp` (bound substitution for single-row
//! cut generation) and of the variable-bound choice it relies on,
//! `HighsImplications::getBestVub`/`getBestVlb`/`cleanupVub`/`cleanupVlb`.
//!
//! A `TransformedLp` is built once per separation round from the LP solution and
//! then transforms/untransforms many rows. It keeps state between calls (each
//! column's last bound type, and variable bounds tightened in place), so replay
//! tests drive whole call sequences, not single calls.
//!
//! Transformed column space: structural columns `0..num_cols`, then one slack per
//! LP row at `num_cols + row`.

use super::cdouble::CDouble;
use super::integers::nearest_integer;
use super::sparse_vec::SparseVectorSum;

const INF: f64 = f64::INFINITY;

/// `HighsImplications::VarBound`: `x <= coef * y + constant` (VUB) or
/// `x >= coef * y + constant` (VLB) with `y` binary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarBound {
    /// Coefficient of the binary column.
    pub coef: f64,
    /// Constant term.
    pub constant: f64,
}

impl VarBound {
    /// `minValue()`: the bound's value at its smaller end.
    pub fn min_value(&self) -> f64 {
        (CDouble::new(self.constant) + self.coef.min(0.0)).to_f64()
    }
    /// `maxValue()`: the bound's value at its larger end.
    pub fn max_value(&self) -> f64 {
        (CDouble::new(self.constant) + self.coef.max(0.0)).to_f64()
    }
}

/// One stored variable bound of a column, in the store's iteration order, with
/// the node-queue counts `getBestVub/Vlb` use as a tiebreak.
#[derive(Clone, Copy, Debug)]
pub struct VbEntry {
    /// The binary column `y`.
    pub col: i32,
    /// The bound.
    pub vb: VarBound,
    /// `nodequeue.numNodesDown(col)`.
    pub nodes_down: i64,
    /// `nodequeue.numNodesUp(col)`.
    pub nodes_up: i64,
}

/// What a `TransformedLp` reads from the LP relaxation and the global domain.
/// Bounds are read through it at every call because the global domain can
/// tighten between calls.
pub trait TlContext {
    /// Number of LP columns (also the slack offset).
    fn num_cols(&self) -> usize;
    /// Number of LP rows.
    fn num_rows(&self) -> usize;
    /// `mipdata_->feastol`.
    fn feastol(&self) -> f64;
    /// `options_mip_->small_matrix_value`.
    fn small_matrix_value(&self) -> f64;
    /// Global lower bound of a structural column.
    fn col_lower(&self, col: i32) -> f64;
    /// Global upper bound of a structural column.
    fn col_upper(&self, col: i32) -> f64;
    /// `lprelaxation.slackLower(row, globaldom)`.
    fn slack_lower(&self, row: i32) -> f64;
    /// `lprelaxation.slackUpper(row, globaldom)`.
    fn slack_upper(&self, row: i32) -> f64;
    /// `lprelaxation.isColIntegral(col)` on the transformed column space.
    fn is_col_integral(&self, col: i32) -> bool;
    /// `lprelaxation.getRow(row)`.
    fn row(&self, row: i32) -> (&[i32], &[f64]);
    /// Called at the start of every `transform`/`untransform`. A no-op by default;
    /// the replay tests use it to load the global bounds HiGHS read at that call,
    /// since the cut pool can tighten the global domain between calls.
    fn begin_call(&self) {}
}

/// LP-solution data and variable-bound lists read only by the constructor.
pub struct TlSetup<'a> {
    /// LP primal values of the structural columns.
    pub col_value: &'a [f64],
    /// LP reduced costs (`col_dual`) of the structural columns.
    pub col_dual: &'a [f64],
    /// LP row activities.
    pub row_value: &'a [f64],
    /// `mipdata_->continuous_cols`.
    pub continuous_cols: &'a [i32],
    /// `mipdata_->integral_cols`.
    pub integral_cols: &'a [i32],
    /// The VUBs of a column after `cleanupVarbounds`, in iteration order.
    pub vubs: &'a dyn Fn(i32) -> Vec<VbEntry>,
    /// The VLBs of a column after `cleanupVarbounds`, in iteration order.
    pub vlbs: &'a dyn Fn(i32) -> Vec<VbEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum BoundType {
    // C++ value-initialises the enum to its first enumerator.
    #[default]
    SimpleUb,
    SimpleLb,
    VariableUb,
    VariableLb,
}

/// `HighsTransformedLp`.
pub struct TransformedLp {
    best_vub: Vec<(i32, VarBound)>,
    best_vlb: Vec<(i32, VarBound)>,
    simple_lb_dist: Vec<f64>,
    simple_ub_dist: Vec<f64>,
    lb_dist: Vec<f64>,
    ub_dist: Vec<f64>,
    bound_dist: Vec<f64>,
    bound_types: Vec<BoundType>,
    vectorsum: SparseVectorSum,
    /// The constructor stopped early because the global domain was infeasible.
    pub aborted: bool,
}

fn is_fixed(ctx: &dyn TlContext, col: i32) -> bool {
    ctx.col_lower(col) == ctx.col_upper(col)
}

/// `HighsImplications::getBestVub`.
fn get_best_vub(
    ctx: &dyn TlContext,
    setup: &TlSetup,
    col: i32,
    list: &[VbEntry],
    best_ub: &mut f64,
) -> (i32, VarBound) {
    let feastol = ctx.feastol();
    let colval = setup.col_value;
    let dual = setup.col_dual;
    let mut best = (
        -1i32,
        VarBound {
            coef: 0.0,
            constant: INF,
        },
    );
    let mut minbest_ub = *best_ub;
    let mut best_ub_dist = INF;
    let mut bestvubnodes: i64 = 0;
    let mut scale = ctx.col_upper(col) - ctx.col_lower(col);
    scale = if scale == INF { 1.0 } else { 1.0 / scale };
    for e in list {
        let (vub_col, vub) = (e.col, e.vb);
        if vub.coef == INF || is_fixed(ctx, vub_col) {
            continue;
        }
        let vubval = colval[vub_col as usize] * vub.coef + vub.constant;
        let mut ub_dist = 0.0f64.max(vubval - colval[col as usize]);
        let y_dist = feastol
            + if vub.coef > 0.0 {
                1.0 - colval[vub_col as usize]
            } else {
                colval[vub_col as usize]
            };
        let norm2 = 1.0 + vub.coef * vub.coef;
        if ub_dist * ub_dist > y_dist * y_dist * norm2 {
            continue;
        }
        ub_dist *= scale;
        if ub_dist <= best_ub_dist + feastol {
            let minvubval = vub.min_value();
            let vubnodes = if vub.coef > 0.0 {
                e.nodes_down
            } else {
                e.nodes_up
            };
            let better = if ub_dist < best_ub_dist - feastol {
                true
            } else if vubnodes > bestvubnodes {
                true
            } else if vubnodes < bestvubnodes {
                false
            } else if minvubval < minbest_ub - feastol {
                true
            } else if minvubval > minbest_ub + feastol {
                false
            } else {
                // HiGHS indexes col_dual[-1] here when no bound is held yet; that
                // branch is unreachable because a finite distance wins above.
                best.0 >= 0
                    && dual[vub_col as usize] / vub.coef - dual[best.0 as usize] / best.1.coef
                        > feastol
            };
            if better {
                *best_ub = vubval;
                minbest_ub = minvubval;
                best = (vub_col, vub);
                bestvubnodes = vubnodes;
                best_ub_dist = ub_dist;
            }
        }
    }
    best
}

/// `HighsImplications::getBestVlb`.
fn get_best_vlb(
    ctx: &dyn TlContext,
    setup: &TlSetup,
    col: i32,
    list: &[VbEntry],
    best_lb: &mut f64,
) -> (i32, VarBound) {
    let feastol = ctx.feastol();
    let colval = setup.col_value;
    let dual = setup.col_dual;
    let mut best = (
        -1i32,
        VarBound {
            coef: 0.0,
            constant: -INF,
        },
    );
    let mut maxbest_lb = *best_lb;
    let mut best_lb_dist = INF;
    let mut bestvlbnodes: i64 = 0;
    let mut scale = ctx.col_upper(col) - ctx.col_lower(col);
    scale = if scale == INF { 1.0 } else { 1.0 / scale };
    for e in list {
        let (vlb_col, vlb) = (e.col, e.vb);
        if vlb.coef == -INF || is_fixed(ctx, vlb_col) {
            continue;
        }
        let vlbval = colval[vlb_col as usize] * vlb.coef + vlb.constant;
        let mut lb_dist = 0.0f64.max(colval[col as usize] - vlbval);
        let y_dist = feastol
            + if vlb.coef > 0.0 {
                colval[vlb_col as usize]
            } else {
                1.0 - colval[vlb_col as usize]
            };
        let norm2 = 1.0 + vlb.coef * vlb.coef;
        if lb_dist * lb_dist > y_dist * y_dist * norm2 {
            continue;
        }
        lb_dist *= scale;
        if lb_dist <= best_lb_dist + feastol {
            let maxvlbval = vlb.max_value();
            let vlbnodes = if vlb.coef > 0.0 {
                e.nodes_up
            } else {
                e.nodes_down
            };
            let better = if lb_dist < best_lb_dist - feastol {
                true
            } else if vlbnodes > bestvlbnodes {
                true
            } else if vlbnodes < bestvlbnodes {
                false
            } else if maxvlbval > maxbest_lb + feastol {
                true
            } else if maxvlbval < maxbest_lb - feastol {
                false
            } else {
                best.0 >= 0
                    && dual[vlb_col as usize] / vlb.coef - dual[best.0 as usize] / best.1.coef
                        < -feastol
            };
            if better {
                *best_lb = vlbval;
                maxbest_lb = maxvlbval;
                best = (vlb_col, vlb);
                bestvlbnodes = vlbnodes;
                best_lb_dist = lb_dist;
            }
        }
    }
    best
}

/// `HighsImplications::cleanupVub` with `allowBoundChanges = false`.
fn cleanup_vub(vub_col: i32, vub: &mut VarBound, ub: f64, feastol: f64, epsilon: f64) -> bool {
    if vub_col == -1 {
        return false;
    }
    let maxub = CDouble::new(vub.max_value());
    let minub = CDouble::new(vub.min_value());
    if minub >= ub - feastol {
        return true; // redundant
    } else if maxub > ub + epsilon {
        let newcoef = (ub - minub).to_f64();
        if vub.coef > 0.0 {
            vub.coef = newcoef;
        } else {
            vub.constant = ub;
            vub.coef = -newcoef;
        }
    }
    false
}

/// `HighsImplications::cleanupVlb` with `allowBoundChanges = false`.
fn cleanup_vlb(vlb_col: i32, vlb: &mut VarBound, lb: f64, feastol: f64, epsilon: f64) -> bool {
    if vlb_col == -1 {
        return false;
    }
    let maxlb = CDouble::new(vlb.max_value());
    let minlb = CDouble::new(vlb.min_value());
    if maxlb <= lb + feastol {
        return true;
    } else if minlb < lb - epsilon {
        let newcoef = (lb - maxlb).to_f64();
        if vlb.coef < 0.0 {
            vlb.coef = newcoef;
        } else {
            vlb.constant = lb;
            vlb.coef = -newcoef;
        }
    }
    false
}

/// A transformed row, as `transform` returns it.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformedRow {
    /// Transformed column per position.
    pub inds: Vec<i32>,
    /// Coefficient per position.
    pub vals: Vec<f64>,
    /// Upper bound of each shifted/complemented variable (`ub - lb`).
    pub upper: Vec<f64>,
    /// LP value of each shifted/complemented variable.
    pub solval: Vec<f64>,
    /// Right-hand side.
    pub rhs: f64,
    /// Every integral column ended with a positive coefficient.
    pub integers_positive: bool,
}

impl TransformedLp {
    /// The `HighsTransformedLp` constructor. `epsilon` is `mipdata_->epsilon`
    /// (used later by the in-place variable-bound tightening in `transform`).
    pub fn new(ctx: &dyn TlContext, setup: &TlSetup) -> Self {
        let n = ctx.num_cols() + ctx.num_rows();
        let mut t = TransformedLp {
            best_vub: vec![
                (
                    -1,
                    VarBound {
                        coef: 0.0,
                        constant: 0.0
                    }
                );
                n
            ],
            best_vlb: vec![
                (
                    -1,
                    VarBound {
                        coef: 0.0,
                        constant: 0.0
                    }
                );
                n
            ],
            simple_lb_dist: vec![0.0; n],
            simple_ub_dist: vec![0.0; n],
            lb_dist: vec![0.0; n],
            ub_dist: vec![0.0; n],
            bound_dist: vec![0.0; n],
            bound_types: vec![BoundType::default(); n],
            vectorsum: SparseVectorSum::new(n),
            aborted: false,
        };
        let feastol = ctx.feastol();
        let colval = setup.col_value;
        for &col in setup.continuous_cols {
            let c = col as usize;
            if is_fixed(ctx, col) {
                continue;
            }
            let mut bestub = ctx.col_upper(col);
            t.simple_ub_dist[c] = bestub - colval[c];
            if t.simple_ub_dist[c] <= feastol {
                t.simple_ub_dist[c] = 0.0;
            }
            t.best_vub[c] = get_best_vub(ctx, setup, col, &(setup.vubs)(col), &mut bestub);
            let mut bestlb = ctx.col_lower(col);
            t.simple_lb_dist[c] = colval[c] - bestlb;
            if t.simple_lb_dist[c] <= feastol {
                t.simple_lb_dist[c] = 0.0;
            }
            t.best_vlb[c] = get_best_vlb(ctx, setup, col, &(setup.vlbs)(col), &mut bestlb);
            t.lb_dist[c] = colval[c] - bestlb;
            if t.lb_dist[c] <= feastol {
                t.lb_dist[c] = 0.0;
            }
            t.ub_dist[c] = bestub - colval[c];
            if t.ub_dist[c] <= feastol {
                t.ub_dist[c] = 0.0;
            }
            t.bound_dist[c] = t.lb_dist[c].min(t.ub_dist[c]);
        }
        for &col in setup.integral_cols {
            let c = col as usize;
            let mut bestub = ctx.col_upper(col);
            let mut bestlb = ctx.col_lower(col);
            t.simple_ub_dist[c] = bestub - colval[c];
            if t.simple_ub_dist[c] <= feastol {
                t.simple_ub_dist[c] = 0.0;
            }
            t.simple_lb_dist[c] = colval[c] - bestlb;
            if t.simple_lb_dist[c] <= feastol {
                t.simple_lb_dist[c] = 0.0;
            }
            let simple_bnd_dist = t.simple_lb_dist[c].min(t.simple_ub_dist[c]);
            if simple_bnd_dist > 0.0
                && (nearest_integer(colval[c]) as f64 - colval[c]).abs() < feastol
            {
                t.best_vub[c] = get_best_vub(ctx, setup, col, &(setup.vubs)(col), &mut bestub);
                t.best_vlb[c] = get_best_vlb(ctx, setup, col, &(setup.vlbs)(col), &mut bestlb);
                t.lb_dist[c] = colval[c] - bestlb;
                if t.lb_dist[c] <= feastol {
                    t.lb_dist[c] = 0.0;
                }
                t.ub_dist[c] = bestub - colval[c];
                if t.ub_dist[c] <= feastol {
                    t.ub_dist[c] = 0.0;
                }
                t.bound_dist[c] = t.lb_dist[c].min(t.ub_dist[c]);
                if t.bound_dist[c] > simple_bnd_dist + feastol {
                    t.lb_dist[c] = t.simple_lb_dist[c];
                    t.ub_dist[c] = t.simple_ub_dist[c];
                    t.bound_dist[c] = simple_bnd_dist;
                    t.best_vub[c].0 = -1;
                    t.best_vlb[c].0 = -1;
                }
            } else {
                t.lb_dist[c] = t.simple_lb_dist[c];
                t.ub_dist[c] = t.simple_ub_dist[c];
                t.bound_dist[c] = simple_bnd_dist;
            }
        }
        let offset = ctx.num_cols();
        for row in 0..ctx.num_rows() {
            let s = offset + row;
            let bestub = ctx.slack_upper(row as i32);
            let bestlb = ctx.slack_lower(row as i32);
            if bestlb == bestub {
                continue;
            }
            t.lb_dist[s] = setup.row_value[row] - bestlb;
            if t.lb_dist[s] <= feastol {
                t.lb_dist[s] = 0.0;
            }
            t.simple_lb_dist[s] = t.lb_dist[s];
            t.ub_dist[s] = bestub - setup.row_value[row];
            if t.ub_dist[s] <= feastol {
                t.ub_dist[s] = 0.0;
            }
            t.simple_ub_dist[s] = t.ub_dist[s];
            t.bound_dist[s] = t.lb_dist[s].min(t.ub_dist[s]);
        }
        t
    }

    /// `boundDistance(col)`.
    pub fn bound_distance(&self, col: i32) -> f64 {
        self.bound_dist[col as usize]
    }

    /// Constructor outputs, for the replay tests: per transformed column
    /// (lbDist, ubDist, simpleLbDist, simpleUbDist, boundDist, bestVlb, bestVub).
    #[allow(clippy::type_complexity)]
    pub fn debug_state(
        &self,
    ) -> (
        &[f64],
        &[f64],
        &[f64],
        &[f64],
        &[f64],
        &[(i32, VarBound)],
        &[(i32, VarBound)],
    ) {
        (
            &self.lb_dist,
            &self.ub_dist,
            &self.simple_lb_dist,
            &self.simple_ub_dist,
            &self.bound_dist,
            &self.best_vlb,
            &self.best_vub,
        )
    }

    /// `HighsTransformedLp::transform`. `None` where HiGHS returns `false`.
    pub fn transform(
        &mut self,
        ctx: &dyn TlContext,
        epsilon: f64,
        mut inds: Vec<i32>,
        mut vals: Vec<f64>,
        rhs: f64,
        mut integers_positive: bool,
        prefer_vbds: bool,
    ) -> Option<TransformedRow> {
        ctx.begin_call();
        let feastol = ctx.feastol();
        let smv = ctx.small_matrix_value();
        let slack_offset = ctx.num_cols() as i32;
        let get_lb = |col: i32| {
            if col < slack_offset {
                ctx.col_lower(col)
            } else {
                ctx.slack_lower(col - slack_offset)
            }
        };
        let get_ub = |col: i32| {
            if col < slack_offset {
                ctx.col_upper(col)
            } else {
                ctx.slack_upper(col - slack_offset)
            }
        };

        let mut tmp_rhs = CDouble::new(rhs);
        let mut num_nz = inds.len();
        let remove = |inds: &mut Vec<i32>, vals: &mut Vec<f64>, num_nz: &mut usize, pos: usize| {
            *num_nz -= 1;
            inds[pos] = inds[*num_nz];
            vals[pos] = vals[*num_nz];
            inds[*num_nz] = 0;
            vals[*num_nz] = 0.0;
        };

        let mut i = 0usize;
        while i < num_nz {
            let col = inds[i];
            let c = col as usize;
            let lb = get_lb(col);
            let ub = get_ub(col);
            if ub - lb < smv {
                tmp_rhs -= lb.min(ub) * vals[i];
                remove(&mut inds, &mut vals, &mut num_nz, i);
                continue;
            }
            if lb == -INF && ub == INF {
                self.vectorsum.clear();
                return None;
            }
            if self.best_vub[c].0 != -1 && self.best_vub[c].1.max_value() > ub + feastol {
                let (vc, mut vb) = self.best_vub[c];
                cleanup_vub(vc, &mut vb, ub, feastol, epsilon);
                self.best_vub[c].1 = vb;
            }
            if self.best_vlb[c].0 != -1 && self.best_vlb[c].1.min_value() < lb - feastol {
                let (vc, mut vb) = self.best_vlb[c];
                cleanup_vlb(vc, &mut vb, lb, feastol, epsilon);
                self.best_vlb[c].1 = vb;
            }
            let old_bound_type = self.bound_types[c];

            if ctx.is_col_integral(col) {
                if ub - lb <= 1.5
                    || self.bound_dist[c] != 0.0
                    || self.simple_lb_dist[c] == 0.0
                    || self.simple_ub_dist[c] == 0.0
                {
                    self.bound_types[c] =
                        if self.simple_lb_dist[c] < self.simple_ub_dist[c] - feastol {
                            BoundType::SimpleLb
                        } else if self.simple_ub_dist[c] < self.simple_lb_dist[c] - feastol {
                            BoundType::SimpleUb
                        } else if vals[i] > 0.0 {
                            BoundType::SimpleLb
                        } else {
                            BoundType::SimpleUb
                        };
                    i += 1;
                    continue;
                }
                self.bound_types[c] = if self.best_vlb[c].0 == -1
                    || self.ub_dist[c] < self.lb_dist[c] - feastol
                {
                    BoundType::VariableUb
                } else if self.best_vub[c].0 == -1 || self.lb_dist[c] < self.ub_dist[c] - feastol {
                    BoundType::VariableLb
                } else if vals[i] > 0.0 {
                    BoundType::VariableUb
                } else {
                    BoundType::VariableLb
                };
            } else {
                let (vlb, vub) = (self.best_vlb[c].0 != -1, self.best_vub[c].0 != -1);
                self.bound_types[c] = if self.lb_dist[c] < self.ub_dist[c] - feastol {
                    if !vlb {
                        BoundType::SimpleLb
                    } else if prefer_vbds
                        || vals[i] > 0.0
                        || self.simple_lb_dist[c] > self.lb_dist[c] + feastol
                    {
                        BoundType::VariableLb
                    } else {
                        BoundType::SimpleLb
                    }
                } else if self.ub_dist[c] < self.lb_dist[c] - feastol {
                    if !vub {
                        BoundType::SimpleUb
                    } else if prefer_vbds
                        || vals[i] < 0.0
                        || self.simple_ub_dist[c] > self.ub_dist[c] + feastol
                    {
                        BoundType::VariableUb
                    } else {
                        BoundType::SimpleUb
                    }
                } else if vals[i] > 0.0 {
                    if vlb {
                        BoundType::VariableLb
                    } else if prefer_vbds && vub {
                        BoundType::VariableUb
                    } else {
                        BoundType::SimpleLb
                    }
                } else if vub {
                    BoundType::VariableUb
                } else if prefer_vbds && vlb {
                    BoundType::VariableLb
                } else {
                    BoundType::SimpleUb
                };
            }

            match self.bound_types[c] {
                BoundType::SimpleLb => {
                    if vals[i] > 0.0 {
                        tmp_rhs -= lb * vals[i];
                        self.bound_types[c] = old_bound_type;
                        remove(&mut inds, &mut vals, &mut num_nz, i);
                        continue;
                    }
                }
                BoundType::SimpleUb => {
                    if vals[i] < 0.0 {
                        tmp_rhs -= ub * vals[i];
                        self.bound_types[c] = old_bound_type;
                        remove(&mut inds, &mut vals, &mut num_nz, i);
                        continue;
                    }
                }
                BoundType::VariableLb => {
                    let (vc, vb) = self.best_vlb[c];
                    tmp_rhs -= vb.constant * vals[i];
                    self.vectorsum.add(vc, vals[i] * vb.coef);
                    if vals[i] > 0.0 {
                        self.bound_types[c] = old_bound_type;
                        remove(&mut inds, &mut vals, &mut num_nz, i);
                        continue;
                    }
                }
                BoundType::VariableUb => {
                    let (vc, vb) = self.best_vub[c];
                    tmp_rhs -= vb.constant * vals[i];
                    self.vectorsum.add(vc, vals[i] * vb.coef);
                    vals[i] = -vals[i];
                    if vals[i] > 0.0 {
                        self.bound_types[c] = old_bound_type;
                        remove(&mut inds, &mut vals, &mut num_nz, i);
                        continue;
                    }
                }
            }
            i += 1;
        }

        if !self.vectorsum.nonzeros().is_empty() {
            for k in 0..num_nz {
                if vals[k] != 0.0 {
                    self.vectorsum.add(inds[k], vals[k]);
                }
            }
            self.vectorsum.cleanup(|_, v| v.abs() <= smv);
            inds = self.vectorsum.nonzeros().to_vec();
            num_nz = inds.len();
            vals = inds.iter().map(|&k| self.vectorsum.value(k)).collect();
            self.vectorsum.clear();
        } else {
            vals.truncate(num_nz);
            inds.truncate(num_nz);
        }

        for j in 0..num_nz {
            let col = inds[j];
            let c = col as usize;
            let lb = get_lb(col);
            let ub = get_ub(col);
            if !ctx.is_col_integral(col) {
                continue;
            }
            if matches!(
                self.bound_types[c],
                BoundType::VariableLb | BoundType::VariableUb
            ) {
                continue;
            }
            self.bound_types[c] = if integers_positive {
                if (lb != -INF && vals[j] > 0.0) || ub == INF {
                    BoundType::SimpleLb
                } else {
                    BoundType::SimpleUb
                }
            } else if self.lb_dist[c] < self.ub_dist[c] {
                BoundType::SimpleLb
            } else {
                BoundType::SimpleUb
            };
        }

        let mut upper = vec![0.0; num_nz];
        let mut solval = vec![0.0; num_nz];
        for j in 0..num_nz {
            let col = inds[j];
            let c = col as usize;
            let lb = get_lb(col);
            let ub = get_ub(col);
            upper[j] = ub - lb;
            match self.bound_types[c] {
                BoundType::SimpleLb => {
                    tmp_rhs -= lb * vals[j];
                    solval[j] = self.lb_dist[c];
                }
                BoundType::SimpleUb => {
                    tmp_rhs -= ub * vals[j];
                    vals[j] = -vals[j];
                    solval[j] = self.ub_dist[c];
                }
                BoundType::VariableLb => solval[j] = self.lb_dist[c],
                BoundType::VariableUb => solval[j] = self.ub_dist[c],
            }
            if ctx.is_col_integral(col) {
                integers_positive = integers_positive && vals[j] > 0.0;
            }
        }
        let rhs = tmp_rhs.to_f64();
        if num_nz == 0 && rhs >= -feastol {
            return None;
        }
        Some(TransformedRow {
            inds,
            vals,
            upper,
            solval,
            rhs,
            integers_positive,
        })
    }

    /// `HighsTransformedLp::untransform`. `None` where HiGHS returns `false`.
    pub fn untransform(
        &mut self,
        ctx: &dyn TlContext,
        inds: &[i32],
        vals_in: &[f64],
        rhs: f64,
        integral: bool,
    ) -> Option<(Vec<i32>, Vec<f64>, f64)> {
        ctx.begin_call();
        let mut tmp_rhs = CDouble::new(rhs);
        let slack_offset = ctx.num_cols() as i32;
        let mut vals = vals_in.to_vec();
        for i in 0..inds.len() {
            if vals[i] == 0.0 {
                continue;
            }
            let col = inds[i];
            let c = col as usize;
            match self.bound_types[c] {
                BoundType::VariableLb => {
                    let (vc, vb) = self.best_vlb[c];
                    tmp_rhs += vb.constant * vals[i];
                    self.vectorsum.add(vc, -vals[i] * vb.coef);
                    self.vectorsum.add(col, vals[i]);
                }
                BoundType::VariableUb => {
                    let (vc, vb) = self.best_vub[c];
                    tmp_rhs -= vb.constant * vals[i];
                    self.vectorsum.add(vc, vals[i] * vb.coef);
                    self.vectorsum.add(col, -vals[i]);
                }
                BoundType::SimpleLb => {
                    if col < slack_offset {
                        tmp_rhs += vals[i] * ctx.col_lower(col);
                        self.vectorsum.add(col, vals[i]);
                    } else {
                        let row = col - slack_offset;
                        tmp_rhs += vals[i] * ctx.slack_lower(row);
                        let (ri, rv) = ctx.row(row);
                        for j in 0..ri.len() {
                            self.vectorsum.add(ri[j], vals[i] * rv[j]);
                        }
                    }
                }
                BoundType::SimpleUb => {
                    if col < slack_offset {
                        tmp_rhs -= vals[i] * ctx.col_upper(col);
                        self.vectorsum.add(col, -vals[i]);
                    } else {
                        let row = col - slack_offset;
                        tmp_rhs -= vals[i] * ctx.slack_upper(row);
                        vals[i] = -vals[i];
                        let (ri, rv) = ctx.row(row);
                        for j in 0..ri.len() {
                            self.vectorsum.add(ri[j], vals[i] * rv[j]);
                        }
                    }
                }
            }
        }
        let rhs_out;
        if integral {
            self.vectorsum.cleanup(|_, v| v.abs() < 0.5);
            rhs_out = tmp_rhs.to_f64().round();
        } else {
            let smv = ctx.small_matrix_value();
            let feastol = ctx.feastol();
            let mut abort = false;
            self.vectorsum.cleanup(|col, val| {
                let absval = val.abs();
                if absval <= smv {
                    return true;
                }
                if absval <= feastol {
                    if val > 0.0 {
                        let l = ctx.col_lower(col);
                        if l == -INF {
                            abort = true;
                        } else {
                            tmp_rhs -= val * l;
                        }
                    } else {
                        let u = ctx.col_upper(col);
                        if u == INF {
                            abort = true;
                        } else {
                            tmp_rhs -= val * u;
                        }
                    }
                    return true;
                }
                false
            });
            if abort {
                self.vectorsum.clear();
                return None;
            }
            rhs_out = tmp_rhs.to_f64();
        }
        let out_inds = self.vectorsum.nonzeros().to_vec();
        let out_vals: Vec<f64> = if integral {
            out_inds
                .iter()
                .map(|&k| self.vectorsum.value(k).round())
                .collect()
        } else {
            out_inds.iter().map(|&k| self.vectorsum.value(k)).collect()
        };
        self.vectorsum.clear();
        Some((out_inds, out_vals, rhs_out))
    }
}
