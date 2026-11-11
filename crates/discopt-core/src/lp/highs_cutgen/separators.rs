//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! Faithful ports of `HighsLpAggregator`, the `generateCut`/`finalizeAndAddCut`
//! wrappers of `HighsCutGeneration`, `HighsDomain::tightenCoefficients`, and the
//! path-aggregation (`HighsPathSeparator`) and tableau (`HighsTableauSeparator`)
//! separators.
//!
//! The separators talk to the LP relaxation through [`SepContext`] and to the cut
//! pool through [`CutSink`], so the same code runs against discopt's LP and pool
//! and, in the replay tests, against HiGHS's recorded LP data and `addCut` results.

use std::collections::HashMap;

use super::cdouble::CDouble;
use super::core::{CoreCut, CutGenCore, CutGenParams};
use super::random::{pair_hash, HighsRandom};
use super::sparse_vec::SparseVectorSum;
use super::stdalgo::{pdqsort, pdqsort_branchless};
use super::transformed_lp::{drop_with_bound, TlContext, TransformedLp};

const INF: f64 = f64::INFINITY;
const K_HIGHS_TINY: f64 = 1e-14;

/// LP-relaxation data the separators read, beyond what a transformed LP reads.
pub trait SepContext: TlContext {
    /// `mipdata_->epsilon`.
    fn epsilon(&self) -> f64;
    /// LP primal values of the structural columns.
    fn col_value(&self) -> &[f64];
    /// LP row activities.
    fn row_value(&self) -> &[f64];
    /// LP row duals.
    fn row_dual(&self) -> &[f64];
    /// LP row lower bound.
    fn row_lower(&self, row: i32) -> f64;
    /// LP row upper bound.
    fn row_upper(&self, row: i32) -> f64;
    /// Column-wise LP matrix: (start, index, value).
    fn col_matrix(&self) -> (&[i32], &[i32], &[f64]);
    /// `mipdata_->continuous_cols`.
    fn continuous_cols(&self) -> &[i32];
    /// `mipdata_->integral_cols.size()`.
    fn num_integral_cols(&self) -> usize;
    /// `lprelaxation.getMaxAbsRowVal(row)`.
    fn max_abs_row_val(&self, row: i32) -> f64;
}

/// The tableau separator's view of the LP basis.
pub trait TableauContext: SepContext {
    /// `lpSolver.hasInvert()`.
    fn has_invert(&self) -> bool;
    /// `getBasicVariablesArray()`: per basis position, a column (`< num_cols`) or
    /// `num_cols + row`.
    fn basic_variables(&self) -> &[i32];
    /// `getDualEdgeWeights()`, if available.
    fn dual_edge_weights(&self) -> Option<&[f64]>;
    /// `getBasisInverseRowSparse(basis_index)`: the nonzeros of row `basis_index`
    /// of B^-1 as (row, weight), or `None` on failure.
    fn basis_inverse_row(&mut self, basis_index: i32) -> Option<Vec<(i32, f64)>>;
    /// `total_lp_iterations - heuristic_lp_iterations`.
    fn sepa_lp_iterations(&self) -> i64;
    /// `options_mip_->mip_pool_soft_limit`.
    fn pool_soft_limit(&self) -> i64;
}

/// The cut pool as the separators use it.
pub trait CutSink {
    /// `cutpool.addCut(...)`: the new cut's index, or -1 when it is rejected
    /// (a duplicate).
    fn add_cut(&mut self, inds: &[i32], vals: &[f64], rhs: f64, integral: bool) -> i32;
    /// `getNumCuts()`.
    fn num_cuts(&self) -> i64;
    /// `getNumAvailableCuts()`.
    fn num_available_cuts(&self) -> i64;
}

// --------------------------------------------------------------- aggregator

/// `HighsLpAggregator`.
pub struct LpAggregator {
    vectorsum: SparseVectorSum,
    /// Entries `current_aggregation` removed from `vectorsum` as small, since the last
    /// `clear()`: the aggregation is `vectorsum + dropped`, which every later call must
    /// still account for (HiGHS forgets them).
    dropped: Vec<(i32, f64)>,
}

impl LpAggregator {
    /// `HighsLpAggregator(lprelaxation)`.
    pub fn new(ctx: &dyn TlContext) -> Self {
        LpAggregator {
            vectorsum: SparseVectorSum::new(ctx.num_rows() + ctx.num_cols()),
            dropped: Vec::new(),
        }
    }
    /// `addRow(row, weight)`: add `weight * row`, with the row's slack.
    pub fn add_row(&mut self, ctx: &dyn TlContext, row: i32, weight: f64) {
        let (ri, rv) = ctx.row(row);
        for i in 0..ri.len() {
            self.vectorsum.add(ri[i], weight * rv[i]);
        }
        self.vectorsum.add(ctx.num_cols() as i32 + row, -weight);
    }
    /// `getCurrentAggregation(inds, vals, negate)`, returning the base row
    /// `Σ vals·x ≤ rhs` it implies.
    ///
    /// HiGHS drops structural entries with `|val| ≤ small_matrix_value` and keeps the
    /// right-hand side at 0, which assumes a dropped term is negligible. It is not when
    /// the column is large: on the captured fac2 OA master a 6e-11 coefficient on the
    /// epigraph column (value 3.3e8, no upper bound) carried 0.02 of activity, and the
    /// tableau and path separators emitted cuts violated by a verified feasible point,
    /// which certified 376262250 against a feasible 331837498. Here a dropped entry
    /// relaxes `rhs` by its bound on the safe side (`v > 0`: `-v·lb`; `v < 0`: `-v·ub`),
    /// as HiGHS's own `postprocessCut` does, and an entry whose bound on that side is
    /// infinite is kept rather than dropped (keeping a term is exact).
    ///
    /// The sum itself is cleaned exactly as HiGHS cleans it (same entries removed, same
    /// order), so the returned row equals HiGHS's whenever every dropped entry has a
    /// finite bound on its side; only `rhs` then differs from HiGHS's 0. The removed
    /// entries are remembered until `clear()`, because the path and tableau separators
    /// read the same sum again (negated, or after adding rows), and the relaxation is
    /// direction-dependent.
    pub fn current_aggregation(
        &mut self,
        ctx: &dyn TlContext,
        negate: bool,
    ) -> (Vec<i32>, Vec<f64>, f64) {
        let droptol = ctx.small_matrix_value();
        let num_col = ctx.num_cols() as i32;
        let dropped = &mut self.dropped;
        self.vectorsum.cleanup(|col, val| {
            if col < num_col && val.abs() <= droptol {
                dropped.push((col, val));
                true
            } else {
                false
            }
        });
        let sign = if negate { -1.0 } else { 1.0 };
        let mut inds = self.vectorsum.nonzeros().to_vec();
        let mut vals: Vec<f64> = inds
            .iter()
            .map(|&i| sign * self.vectorsum.value(i))
            .collect();
        let mut rhs = CDouble::new(0.0);
        for &(col, val) in &self.dropped {
            let v = sign * val;
            if !drop_with_bound(ctx, col, v, &mut rhs) {
                match inds.iter().position(|&c| c == col) {
                    Some(k) => vals[k] += v,
                    None => {
                        inds.push(col);
                        vals.push(v);
                    }
                }
            }
        }
        (inds, vals, rhs.to_f64())
    }
    /// `clear()`.
    pub fn clear(&mut self) {
        self.vectorsum.clear();
        self.dropped.clear();
    }
    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.vectorsum.nonzeros().is_empty()
    }
}

// ------------------------------------------------------------ cut generation

/// `HighsDomain::tightenCoefficients` on the global domain.
pub fn tighten_coefficients(ctx: &dyn TlContext, inds: &[i32], vals: &mut [f64], rhs: &mut f64) {
    let mut maxactivity = CDouble::new(0.0);
    for i in 0..inds.len() {
        if vals[i] > 0.0 {
            let u = ctx.col_upper(inds[i]);
            if u == INF {
                return;
            }
            maxactivity += u * vals[i];
        } else {
            let l = ctx.col_lower(inds[i]);
            if l == -INF {
                return;
            }
            maxactivity += l * vals[i];
        }
    }
    let maxabscoef = maxactivity - *rhs;
    if maxabscoef > ctx.feastol() {
        let mut upper = CDouble::new(*rhs);
        let mut tightened = 0;
        for i in 0..inds.len() {
            if !ctx.is_col_integral(inds[i]) {
                continue;
            }
            if vals[i] > maxabscoef.to_f64() {
                let delta = vals[i] - maxabscoef;
                upper -= delta * ctx.col_upper(inds[i]);
                vals[i] = maxabscoef.to_f64();
                tightened += 1;
            } else if vals[i] < -maxabscoef.to_f64() {
                let delta = -vals[i] - maxabscoef;
                upper += delta * ctx.col_lower(inds[i]);
                vals[i] = -maxabscoef.to_f64();
                tightened += 1;
            }
        }
        if tightened != 0 {
            *rhs = upper.to_f64();
        }
    }
}

/// One `HighsCutGeneration` object (one per separator call).
pub struct CutGenerator {
    core: CutGenCore,
}

impl CutGenerator {
    /// `HighsCutGeneration(lpRelaxation, cutpool)`: `seed` is `random_seed + LP
    /// iterations + pool size`, truncated to 32 bits as HiGHS does.
    pub fn new(ctx: &dyn SepContext, seed: i64) -> Self {
        let p = CutGenParams {
            feastol: ctx.feastol(),
            epsilon: ctx.epsilon(),
            num_cols: ctx.num_cols(),
        };
        CutGenerator {
            core: CutGenCore::new(p, HighsRandom::new(seed as u32)),
        }
    }

    /// The core (for the random state in tests).
    pub fn core(&self) -> &CutGenCore {
        &self.core
    }

    fn violation(ctx: &dyn SepContext, inds: &[i32], vals: &[f64], rhs: f64) -> f64 {
        let sol = ctx.col_value();
        let mut v = CDouble::new(-rhs);
        for i in 0..inds.len() {
            v += sol[inds[i] as usize] * vals[i];
        }
        v.to_f64()
    }

    fn finish(&mut self, ctx: &dyn SepContext, pool: &mut dyn CutSink, cut: CoreCut) -> bool {
        let lb = |c: i32| ctx.col_lower(c);
        let ub = |c: i32| ctx.col_upper(c);
        let isint = |c: i32| ctx.is_col_integral(c);
        let Some(pp) = self.core.postprocess_cut(cut, &lb, &ub, &isint) else {
            return false;
        };
        let mut rhs = pp.rhs.to_f64();
        let inds = pp.inds;
        let mut vals = pp.vals;
        if Self::violation(ctx, &inds, &vals, rhs) <= 10.0 * ctx.feastol() {
            return false;
        }
        tighten_coefficients(ctx, &inds, &mut vals, &mut rhs);
        pool.add_cut(
            &inds,
            &vals,
            rhs,
            pp.integral_support && pp.integral_coefficients,
        ) != -1
    }

    /// `generateCut(transLp, inds, vals, rhs, onlyInitialCMIRScale)`.
    pub fn generate_cut(
        &mut self,
        ctx: &dyn SepContext,
        tl: &mut TransformedLp,
        pool: &mut dyn CutSink,
        inds: Vec<i32>,
        vals: Vec<f64>,
        rhs: f64,
        only_initial_cmir_scale: bool,
    ) -> bool {
        let Some(row) = tl.transform(ctx, ctx.epsilon(), inds, vals, rhs, true, false) else {
            return false;
        };
        let base = super::core::BaseRow {
            col_integral: row.inds.iter().map(|&c| ctx.is_col_integral(c)).collect(),
            inds: row.inds,
            vals: row.vals,
            rhs: row.rhs,
            upper: row.upper,
            solval: row.solval,
            ints_positive: row.integers_positive,
        };
        let Some(cut) = self.core.generate_cut_core(&base, only_initial_cmir_scale) else {
            return false;
        };
        let Some((ui, uv, urhs)) =
            tl.untransform(ctx, &cut.inds, &cut.vals, cut.rhs.to_f64(), false)
        else {
            return false;
        };
        let cut = CoreCut {
            inds: ui,
            vals: uv,
            rhs: CDouble::new(urhs),
            integral_support: cut.integral_support,
            integral_coefficients: cut.integral_coefficients,
        };
        self.finish(ctx, pool, cut)
    }

    /// `finalizeAndAddCut(globaldom, inds, vals, rhs)`.
    pub fn finalize_and_add_cut(
        &mut self,
        ctx: &dyn SepContext,
        pool: &mut dyn CutSink,
        mut inds: Vec<i32>,
        mut vals: Vec<f64>,
        rhs: f64,
    ) -> bool {
        let mut integral_support = true;
        let mut rowlen = inds.len();
        let mut i = rowlen;
        while i > 0 {
            i -= 1;
            if vals[i] == 0.0 {
                rowlen -= 1;
                inds[i] = inds[rowlen];
                vals[i] = vals[rowlen];
            } else {
                integral_support &= ctx.is_col_integral(inds[i]);
            }
        }
        inds.truncate(rowlen);
        vals.truncate(rowlen);
        let cut = CoreCut {
            inds,
            vals,
            rhs: CDouble::new(rhs),
            integral_support,
            integral_coefficients: false,
        };
        self.finish(ctx, pool, cut)
    }
}

// ----------------------------------------------------------- path separator

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowType {
    Unusable,
    Geq,
    Eq,
    Leq,
}

/// `HighsPathSeparator`: its random generator persists across calls.
pub struct PathSeparator {
    randgen: HighsRandom,
    /// `numCalls`.
    pub num_calls: i64,
}

impl PathSeparator {
    /// `HighsPathSeparator(mipsolver)`: `randgen.initialise(random_seed)`.
    pub fn new(random_seed: i32) -> Self {
        PathSeparator {
            randgen: HighsRandom::new(random_seed as u32),
            num_calls: 0,
        }
    }
    /// Resume with a recorded generator state (replay tests).
    pub fn with_state(state: u64) -> Self {
        PathSeparator {
            randgen: HighsRandom::from_state(state),
            num_calls: 0,
        }
    }
    /// The generator state.
    pub fn rng_state(&self) -> u64 {
        self.randgen.state()
    }

    /// `separateLpSolution`. `cg_seed` seeds this call's `HighsCutGeneration`.
    pub fn separate(
        &mut self,
        ctx: &dyn SepContext,
        agg: &mut LpAggregator,
        tl: &mut TransformedLp,
        pool: &mut dyn CutSink,
        cg_seed: i64,
    ) {
        let feastol = ctx.feastol();
        let epsilon = ctx.epsilon();
        let num_row = ctx.num_rows();
        let num_col = ctx.num_cols();
        let row_value = ctx.row_value();
        let row_dual = ctx.row_dual();
        let (astart, aindex, avalue) = ctx.col_matrix();

        let mut rowtype = vec![RowType::Unusable; num_row];
        for i in 0..num_row {
            let (lo, up) = (ctx.row_lower(i as i32), ctx.row_upper(i as i32));
            if lo == up {
                rowtype[i] = RowType::Eq;
                continue;
            }
            let mut lowerslack = INF;
            let mut upperslack = INF;
            if lo != -INF {
                lowerslack = row_value[i] - lo;
            }
            if up != INF {
                upperslack = up - row_value[i];
            }
            rowtype[i] = if lowerslack > feastol && upperslack > feastol {
                RowType::Unusable
            } else if lowerslack < upperslack {
                RowType::Geq
            } else {
                RowType::Leq
            };
        }

        let mut num_continuous = vec![0i32; num_row];
        for &col in ctx.continuous_cols() {
            if tl.bound_distance(col) == 0.0 {
                continue;
            }
            for k in astart[col as usize]..astart[col as usize + 1] {
                num_continuous[aindex[k as usize] as usize] += 1;
            }
        }

        let mut col_substitutions: Vec<(i32, f64)> = vec![(-1, 0.0); num_col];
        for i in 0..num_row {
            if rowtype[i] != RowType::Eq || num_continuous[i] != 1 {
                continue;
            }
            let (ri, rv) = ctx.row(i as i32);
            let mut col = -1i32;
            let mut val = 0.0;
            for j in 0..ri.len() {
                if ctx.is_col_integral(ri[j]) {
                    continue;
                }
                if tl.bound_distance(ri[j]) == 0.0 {
                    continue;
                }
                col = ri[j];
                val = rv[j];
                break;
            }
            debug_assert!(col != -1);
            if col == -1 || col_substitutions[col as usize].0 != -1 {
                continue;
            }
            col_substitutions[col as usize] = (i as i32, val);
            rowtype[i] = RowType::Unusable;
        }

        let mut in_arc_rows: Vec<(i32, f64)> = Vec::new();
        let mut col_in_arcs: Vec<(usize, usize)> = vec![(0, 0); num_col];
        let mut out_arc_rows: Vec<(i32, f64)> = Vec::new();
        let mut col_out_arcs: Vec<(usize, usize)> = vec![(0, 0); num_col];
        for &col in ctx.continuous_cols() {
            let c = col as usize;
            if tl.bound_distance(col) == 0.0 || col_substitutions[c].0 != -1 {
                continue;
            }
            col_in_arcs[c].0 = in_arc_rows.len();
            col_out_arcs[c].0 = out_arc_rows.len();
            for k in astart[c]..astart[c + 1] {
                let (r, v) = (aindex[k as usize], avalue[k as usize]);
                match rowtype[r as usize] {
                    RowType::Unusable => continue,
                    RowType::Leq => {
                        if v < 0.0 {
                            in_arc_rows.push((r, v));
                        } else {
                            out_arc_rows.push((r, v));
                        }
                    }
                    RowType::Geq => {
                        if v > 0.0 {
                            in_arc_rows.push((r, v));
                        } else {
                            out_arc_rows.push((r, v));
                        }
                    }
                    RowType::Eq => {
                        in_arc_rows.push((r, v));
                        out_arc_rows.push((r, v));
                    }
                }
            }
            col_in_arcs[c].1 = in_arc_rows.len();
            col_out_arcs[c].1 = out_arc_rows.len();
        }

        let mut cut_gen = CutGenerator::new(ctx, cg_seed);
        const MAX_PATH_LEN: usize = 6;
        let mut current_path = [0i32; MAX_PATH_LEN];
        let mut aggregated_path: Vec<(Vec<i32>, Vec<f64>, f64)> = Vec::new();
        let max_weight = 1.0 / feastol;
        let min_weight = feastol;
        let check_weight = |w: f64| {
            let w = w.abs();
            w <= max_weight && w >= min_weight
        };

        for i in 0..num_row {
            let scales: [f64; 2] = match rowtype[i] {
                RowType::Unusable => continue,
                RowType::Eq => {
                    if row_dual[i] <= epsilon {
                        [1.0, -1.0]
                    } else {
                        [-1.0, 1.0]
                    }
                }
                RowType::Leq => [1.0, -1.0],
                RowType::Geq => [-1.0, 1.0],
            };
            let mut success = false;
            for &scale_s in &scales {
                agg.add_row(ctx, i as i32, scale_s);
                current_path[0] = i as i32;
                let mut curr_path_len = 1usize;
                let mut try_negated_scale = false;
                aggregated_path.clear();

                while curr_path_len != MAX_PATH_LEN {
                    let (base_inds, base_vals, base_rhs) = agg.current_aggregation(ctx, false);
                    let mut added_substitution_rows = false;
                    let mut best_out_arc_col = -1i32;
                    let mut out_arc_col_val = 0.0;
                    let mut out_arc_col_bound_dist = 0.0;
                    let mut best_in_arc_col = -1i32;
                    let mut in_arc_col_val = 0.0;
                    let mut in_arc_col_bound_dist = 0.0;
                    let in_path = |row: i32, len: usize, path: &[i32; MAX_PATH_LEN]| {
                        path[..len].contains(&row)
                    };

                    for j in 0..base_inds.len() {
                        let col = base_inds[j];
                        if col as usize >= num_col
                            || tl.bound_distance(col) == 0.0
                            || ctx.is_col_integral(col)
                        {
                            continue;
                        }
                        let c = col as usize;
                        if col_substitutions[c].0 != -1 {
                            added_substitution_rows = true;
                            agg.add_row(
                                ctx,
                                col_substitutions[c].0,
                                -base_vals[j] / col_substitutions[c].1,
                            );
                            continue;
                        }
                        if added_substitution_rows {
                            continue;
                        }
                        // skipCol(col, colArcs, arcRows, otherColArcs, otherArcRows)
                        let skip_col = |col_arcs: &[(usize, usize)],
                                        arc_rows: &[(i32, f64)],
                                        other_col_arcs: &[(usize, usize)],
                                        other_arc_rows: &[(i32, f64)],
                                        tns: &mut bool|
                         -> bool {
                            if curr_path_len == 1 && !*tns {
                                if col_arcs[c].1 - col_arcs[c].0 <= curr_path_len {
                                    for k in col_arcs[c].0..col_arcs[c].1 {
                                        if arc_rows[k].0 != i as i32 {
                                            *tns = true;
                                            break;
                                        }
                                    }
                                } else {
                                    *tns = true;
                                }
                            }
                            if other_col_arcs[c].0 == other_col_arcs[c].1 {
                                return true;
                            }
                            if other_col_arcs[c].1 - other_col_arcs[c].0 <= curr_path_len {
                                for k in other_col_arcs[c].0..other_col_arcs[c].1 {
                                    if !in_path(other_arc_rows[k].0, curr_path_len, &current_path) {
                                        return false;
                                    }
                                }
                                return true;
                            }
                            false
                        };
                        if base_vals[j] < 0.0 {
                            if skip_col(
                                &col_out_arcs,
                                &out_arc_rows,
                                &col_in_arcs,
                                &in_arc_rows,
                                &mut try_negated_scale,
                            ) {
                                continue;
                            }
                            if best_out_arc_col == -1
                                || tl.bound_distance(col) > out_arc_col_bound_dist
                            {
                                best_out_arc_col = col;
                                out_arc_col_val = base_vals[j];
                                out_arc_col_bound_dist = tl.bound_distance(col);
                            }
                        } else {
                            if skip_col(
                                &col_in_arcs,
                                &in_arc_rows,
                                &col_out_arcs,
                                &out_arc_rows,
                                &mut try_negated_scale,
                            ) {
                                continue;
                            }
                            if best_in_arc_col == -1
                                || tl.bound_distance(col) > in_arc_col_bound_dist
                            {
                                best_in_arc_col = col;
                                in_arc_col_val = base_vals[j];
                                in_arc_col_bound_dist = tl.bound_distance(col);
                            }
                        }
                    }
                    if added_substitution_rows {
                        continue;
                    }

                    success =
                        cut_gen.generate_cut(ctx, tl, pool, base_inds, base_vals, base_rhs, false);
                    let (neg_inds, neg_vals, neg_rhs) = agg.current_aggregation(ctx, true);
                    if !aggregated_path.is_empty()
                        || best_out_arc_col != -1
                        || best_in_arc_col != -1
                    {
                        aggregated_path.push((neg_inds.clone(), neg_vals.clone(), neg_rhs));
                    }
                    success |=
                        cut_gen.generate_cut(ctx, tl, pool, neg_inds, neg_vals, neg_rhs, false);
                    if success || (best_out_arc_col == -1 && best_in_arc_col == -1) {
                        break;
                    }

                    let find_row = |best_arc_col: i32,
                                    val: f64,
                                    col_arcs: &[(usize, usize)],
                                    arc_rows: &[(i32, f64)],
                                    randgen: &mut HighsRandom|
                     -> Option<(i32, f64)> {
                        let (lo, hi) = col_arcs[best_arc_col as usize];
                        let arc_row = lo + randgen.integer_below((hi - lo) as i32) as usize;
                        let try_k = |k: usize| -> Option<(i32, f64)> {
                            let r = arc_rows[k].0;
                            let w = -val / arc_rows[k].1;
                            if !in_path(r, curr_path_len, &current_path) && check_weight(w) {
                                Some((r, w))
                            } else {
                                None
                            }
                        };
                        if let Some(x) = try_k(arc_row) {
                            return Some(x);
                        }
                        for k in arc_row + 1..hi {
                            if let Some(x) = try_k(k) {
                                return Some(x);
                            }
                        }
                        for k in lo..arc_row {
                            if let Some(x) = try_k(k) {
                                return Some(x);
                            }
                        }
                        None
                    };
                    let found = if best_in_arc_col == -1
                        || (best_out_arc_col != -1
                            && out_arc_col_bound_dist >= in_arc_col_bound_dist - feastol)
                    {
                        match find_row(
                            best_out_arc_col,
                            out_arc_col_val,
                            &col_in_arcs,
                            &in_arc_rows,
                            &mut self.randgen,
                        ) {
                            Some(x) => Some(x),
                            None => {
                                if best_in_arc_col == -1 {
                                    None
                                } else {
                                    find_row(
                                        best_in_arc_col,
                                        in_arc_col_val,
                                        &col_out_arcs,
                                        &out_arc_rows,
                                        &mut self.randgen,
                                    )
                                }
                            }
                        }
                    } else {
                        find_row(
                            best_in_arc_col,
                            in_arc_col_val,
                            &col_out_arcs,
                            &out_arc_rows,
                            &mut self.randgen,
                        )
                    };
                    let Some((row, weight)) = found else { break };
                    current_path[curr_path_len] = row;
                    agg.add_row(ctx, row, weight);
                    curr_path_len += 1;
                }

                let mut path_len = aggregated_path.len();
                if path_len > 1 {
                    success |= self.path_mixing_cut(
                        ctx,
                        tl,
                        pool,
                        &mut cut_gen,
                        &mut aggregated_path,
                        &mut path_len,
                    );
                }
                agg.clear();
                if !try_negated_scale {
                    break;
                }
            }
            let _ = success;
        }
    }

    fn path_mixing_cut(
        &mut self,
        ctx: &dyn SepContext,
        tl: &mut TransformedLp,
        pool: &mut dyn CutSink,
        cut_gen: &mut CutGenerator,
        aggregated_path: &mut [(Vec<i32>, Vec<f64>, f64)],
        path_len: &mut usize,
    ) -> bool {
        let feastol = ctx.feastol();
        let epsilon = ctx.epsilon();
        let mut success = false;
        let mut index_pos: HashMap<i32, usize> = HashMap::new();
        let mut inds: Vec<i32> = Vec::new();
        let mut solval: Vec<f64> = Vec::new();
        let mut upper: Vec<f64> = Vec::new();
        let mut is_integral: Vec<bool> = Vec::new();
        // Each aggregation's own right-hand side (0 in HiGHS; nonzero here only where
        // `current_aggregation` relaxed it for a dropped entry).
        let mut rhs: Vec<f64> = aggregated_path[..*path_len].iter().map(|a| a.2).collect();
        let mut delta = 1.0f64;
        let n0 = *path_len;
        for k in 0..n0 {
            let (ai, av) = (aggregated_path[k].0.clone(), aggregated_path[k].1.clone());
            let Some(row) = tl.transform(ctx, epsilon, ai, av, rhs[k], false, false) else {
                *path_len = k;
                break;
            };
            aggregated_path[k] = (row.inds.clone(), row.vals.clone(), row.rhs);
            rhs[k] = row.rhs;
            if k == 0 {
                if rhs[k] > K_HIGHS_TINY {
                    *path_len = k;
                    break;
                }
                if rhs[k] >= -feastol {
                    rhs[k] = 0.0;
                }
            } else if rhs[k] >= rhs[k - 1] - feastol {
                *path_len = k;
                break;
            }
            delta = rhs[k].abs().max(delta);
            for j in 0..row.inds.len() {
                let index = row.inds[j];
                let pos = index_pos.entry(index).or_insert(0);
                if *pos == 0 {
                    inds.push(index);
                    solval.push(row.solval[j]);
                    upper.push(row.upper[j]);
                    is_integral.push(ctx.is_col_integral(index));
                    if *is_integral.last().unwrap() {
                        delta = row.vals[j].abs().max(delta);
                    }
                    *pos = inds.len();
                } else if is_integral[*pos - 1] {
                    delta = row.vals[j].abs().max(delta);
                }
            }
        }
        let path_len = *path_len;
        if path_len <= 1 {
            return false;
        }
        delta = (delta + 1.0).log2().ceil().exp2();
        let mut num_inds = inds.len();
        let mut cut_rhs = CDouble::new(0.0);
        let mut cut_vals = vec![0.0f64; num_inds];
        let mut max_frac = vec![0.0f64; num_inds];
        let mut down_sum = vec![CDouble::new(0.0); num_inds];
        let mut f_sum = vec![CDouble::new(0.0); num_inds];
        let mut f_last = 0.0f64;
        let mut scale = -1.0 / delta;
        for k in 0..path_len {
            let f = rhs[k] * scale;
            let f_diff = CDouble::new(f) - f_last;
            cut_rhs += f_diff;
            for j in 0..aggregated_path[k].0.len() {
                let i = index_pos[&aggregated_path[k].0[j]] - 1;
                let gj = aggregated_path[k].1[j] * scale;
                if !is_integral[i] {
                    cut_vals[i] = cut_vals[i].max(gj);
                } else {
                    let gjdown = gj.floor();
                    let hj = gj - gjdown;
                    max_frac[i] = max_frac[i].max(hj);
                    down_sum[i] += f_diff * gjdown;
                    f_sum[i] += f_diff;
                    cut_vals[i] = if f_sum[i] < max_frac[i] {
                        (down_sum[i] + f_sum[i]).to_f64()
                    } else {
                        (down_sum[i] + max_frac[i]).to_f64()
                    };
                }
            }
            if k > 0 {
                let mut viol = cut_rhs.to_f64();
                for j in 0..num_inds {
                    viol -= solval[j] * cut_vals[j];
                }
                viol *= delta;
                if viol > 10.0 * feastol {
                    scale = -delta;
                    let r = cut_rhs.to_f64() * scale;
                    for v in cut_vals.iter_mut() {
                        *v *= scale;
                    }
                    let mut j = num_inds;
                    while j > 0 {
                        j -= 1;
                        if cut_vals[j].abs() <= epsilon {
                            num_inds -= 1;
                            cut_vals.swap(j, num_inds);
                            inds.swap(j, num_inds);
                        }
                    }
                    cut_vals.truncate(num_inds);
                    inds.truncate(num_inds);
                    if let Some((ui, uv, ur)) = tl.untransform(ctx, &inds, &cut_vals, r, false) {
                        success |= cut_gen.finalize_and_add_cut(ctx, pool, ui, uv, ur);
                    }
                    break;
                }
            }
            f_last = f;
        }
        success
    }
}

// --------------------------------------------------------- tableau separator

#[derive(Clone, Debug)]
struct FractionalInteger {
    fractionality: f64,
    row_ep_norm2: f64,
    score: f64,
    basis_index: i32,
    row_ep: Vec<(i32, f64)>,
}

/// `fractionality(x)`: distance to the nearest integer.
fn fractionality(x: f64) -> f64 {
    (x - x.round()).abs()
}

/// `HighsHashHelpers::hash(int64_t)`.
fn hash_i64(v: i64) -> u64 {
    let (b0, b1) = (v as u64 as u32, ((v as u64) >> 32) as u32);
    pair_hash(1, b0, b1) ^ (pair_hash(0, b0, b1) >> 32)
}

/// `std::make_pair(a1, b1) > std::make_pair(a2, b2)` (via `operator<`).
fn pair_gt(a1: f64, b1: u64, a2: f64, b2: u64) -> bool {
    a2 < a1 || (!(a1 < a2) && b2 < b1)
}

/// `HighsTableauSeparator`: `numTries` persists across calls.
#[derive(Default)]
pub struct TableauSeparator {
    /// `numTries`.
    pub num_tries: i64,
    /// `numCalls` (incremented by `HighsSeparator::run` before each call).
    pub num_calls: i64,
}

impl TableauSeparator {
    /// `separateLpSolution`.
    pub fn separate(
        &mut self,
        ctx: &mut dyn TableauContext,
        agg: &mut LpAggregator,
        tl: &mut TransformedLp,
        pool: &mut dyn CutSink,
        cg_seed: i64,
    ) {
        if !ctx.has_invert() {
            return;
        }
        if pool.num_available_cuts() > ctx.pool_soft_limit() {
            return;
        }
        let feastol = ctx.feastol();
        let num_row = ctx.num_rows();
        let num_col = ctx.num_cols() as i32;
        let mut cut_gen = CutGenerator::new(ctx, cg_seed);
        let basis = ctx.basic_variables().to_vec();
        let mut frac: Vec<FractionalInteger> = Vec::new();
        for (i, &b) in basis.iter().enumerate().take(num_row) {
            let my_frac = if b >= num_col {
                let row = b - num_col;
                if !ctx.is_col_integral(b) {
                    continue;
                }
                fractionality(ctx.row_value()[row as usize])
            } else {
                if !ctx.is_col_integral(b) {
                    continue;
                }
                fractionality(ctx.col_value()[b as usize])
            };
            if my_frac < 1000.0 * feastol {
                continue;
            }
            frac.push(FractionalInteger {
                fractionality: my_frac,
                row_ep_norm2: 0.0,
                score: -1.0,
                basis_index: i as i32,
                row_ep: Vec::new(),
            });
        }
        if frac.is_empty() {
            return;
        }
        let mut max_tries: i64 = 5000 + self.num_calls * 50 + ctx.sepa_lp_iterations() / 10;
        if self.num_tries >= max_tries {
            return;
        }
        max_tries -= self.num_tries;
        let cap = 200 + (0.1 * (num_row.min(ctx.num_integral_cols()) as f64)) as i64;
        max_tries = max_tries.min(cap);
        if frac.len() as i64 > max_tries {
            let nt = self.num_tries;
            // pdqsort's result depends only on its comparison sequence, so sorting
            // positions with the same comparator gives the same permutation.
            let mut order: Vec<usize> = (0..frac.len()).collect();
            if let Some(ew) = ctx.dual_edge_weights() {
                let ew = ew.to_vec();
                let fr = &frac;
                pdqsort(&mut order, |&ia: &usize, &ib: &usize| {
                    let (a, b) = (&fr[ia], &fr[ib]);
                    let s1 = a.fractionality * (1.0 - a.fractionality) / ew[a.basis_index as usize];
                    let s2 = b.fractionality * (1.0 - b.fractionality) / ew[b.basis_index as usize];
                    pair_gt(
                        s1,
                        hash_i64(nt + a.basis_index as i64),
                        s2,
                        hash_i64(nt + b.basis_index as i64),
                    )
                });
            } else {
                let fr = &frac;
                pdqsort(&mut order, |&ia: &usize, &ib: &usize| {
                    let (a, b) = (&fr[ia], &fr[ib]);
                    pair_gt(
                        a.fractionality,
                        hash_i64(nt + a.basis_index as i64),
                        b.fractionality,
                        hash_i64(nt + b.basis_index as i64),
                    )
                });
            }
            order.truncate(max_tries as usize);
            frac = order.iter().map(|&k| frac[k].clone()).collect();
        }
        self.num_tries += frac.len() as i64;

        for fv in frac.iter_mut() {
            let Some(row_ep) = ctx.basis_inverse_row(fv.basis_index) else {
                continue;
            };
            if row_ep.len() == 1 {
                continue;
            }
            fv.row_ep_norm2 = 0.0;
            let mut min_weight = INF;
            let mut max_weight = 0.0f64;
            for &(row, weight) in &row_ep {
                let scaled = ctx.max_abs_row_val(row) * weight.abs();
                if scaled <= feastol {
                    continue;
                }
                min_weight = min_weight.min(scaled);
                max_weight = max_weight.max(scaled);
                fv.row_ep_norm2 += scaled * scaled;
                fv.row_ep.push((row, weight));
            }
            if fv.row_ep.len() <= 1 {
                continue;
            }
            if max_weight / min_weight <= 1e4 {
                fv.score = fv.fractionality * (1.0 - fv.fractionality) / fv.row_ep_norm2;
            }
        }
        frac.retain(|f| !(f.score <= feastol));
        if frac.is_empty() {
            return;
        }
        // FractionalInteger::operator< is `score > other.score`.
        let mut order: Vec<usize> = (0..frac.len()).collect();
        {
            let fr = &frac;
            pdqsort_branchless(&mut order, |&ia: &usize, &ib: &usize| {
                fr[ia].score > fr[ib].score
            });
        }
        let frac: Vec<FractionalInteger> = order.iter().map(|&k| frac[k].clone()).collect();
        let mut best_score = -1.0f64;
        let num_cuts0 = pool.num_cuts();
        let best_score_fac = [0.0025, 0.01];
        let mip_cols = ctx.num_cols();
        for fv in &frac {
            if pool.num_cuts() - num_cuts0 >= 1000 {
                break;
            }
            if fv.score < best_score_fac[(pool.num_cuts() - num_cuts0 >= 50) as usize] * best_score
            {
                break;
            }
            for &(row, w) in &fv.row_ep {
                agg.add_row(ctx, row, w);
            }
            let (inds, vals, arhs) = agg.current_aggregation(ctx, false);
            if 10usize.wrapping_mul(inds.len().wrapping_sub(fv.row_ep.len())) > 10000 + mip_cols {
                agg.clear();
                continue;
            }
            if inds.len() > fv.row_ep.len() {
                let mut max_abs = 0.0f64;
                let mut min_abs = INF;
                for k in 0..inds.len() {
                    if (inds[k] as usize) < mip_cols {
                        max_abs = vals[k].abs().max(max_abs);
                        min_abs = vals[k].abs().min(min_abs);
                    }
                }
                if max_abs / min_abs > 1e6 {
                    agg.clear();
                    continue;
                }
            }
            cut_gen.generate_cut(ctx, tl, pool, inds, vals, arhs, false);
            let (ninds, nvals, nrhs) = agg.current_aggregation(ctx, true);
            cut_gen.generate_cut(ctx, tl, pool, ninds, nvals, nrhs, false);
            agg.clear();
            if best_score == -1.0 && pool.num_cuts() != num_cuts0 {
                best_score = fv.score;
            }
        }
    }
}
