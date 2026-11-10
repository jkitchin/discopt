//! Bridge from discopt's standard-form root LP to the ported HiGHS separation
//! subsystem (gap plan P3, option A, phase 4b).
//!
//! discopt's LP is `A x = b, l <= x <= u` over structural columns `0..ns` followed by
//! slack columns. HiGHS's view is one row `lo <= a x <= up` per LP row over the
//! structural columns, with the row activity as an implicit column `ns + row`.
//! Each round this module derives that view (a row's range from its singleton slack
//! column, or an equality when it has none), runs the ported tableau and path
//! separators into a persistent ported [`CutPool`], and returns the pool's
//! selection as `>=` cuts over the structural columns.
//!
//! Soundness: every cut comes from HiGHS's cut generators applied to valid
//! aggregations of LP rows over the global box, so it is valid for every integer
//! point of the box. A row whose slack column is not a clean singleton is given
//! infinite bounds, which makes `transform` reject any aggregation that uses it, so
//! no cut leans on a range this module could not read. The tableau separator is
//! skipped in a round whose basis holds such a column.
//!
//! Not ported (measured deferrals, §12.16-§12.19): the implied-bound and clique
//! separators. Variable bounds (§12.21), root propagation (§12.20), node separation
//! (§12.23) and mod-k (§12.24) are ported behind their levers.

use super::pool::{CutPool, NoObserver};
use super::separators::{
    CutSink, LpAggregator, PathSeparator, SepContext, TableauContext, TableauSeparator,
};
use super::transformed_lp::{TlContext, TlSetup, TransformedLp};
use super::vbound::{Domain, VarBoundStore};
use crate::lp::gomory::GomoryCut;
use crate::lp::simplex::linsolve::{FeralLU, LinearSolver};
use crate::lp::simplex::sparse::SparseCols;

const INF: f64 = f64::INFINITY;
/// discopt's LP layer marks an infinite bound with the sentinel `1e20`, not
/// `f64::INFINITY` (CLAUDE.md "Key Constraints"). HiGHS's code tests `== kHighsInf`,
/// so every bound crossing into the port is mapped: a finite-looking 1e20 would be
/// complemented against and the cancellation produces invalid cuts (measured: 104 of
/// 104 round-1 tableau cuts on nexp-50-20-4-2 cut off a feasible point).
const LP_INF: f64 = 1e20;

fn lo_bound(v: f64) -> f64 {
    if v <= -LP_INF {
        -INF
    } else {
        v
    }
}

fn up_bound(v: f64) -> f64 {
    if v >= LP_INF {
        INF
    } else {
        v
    }
}
/// `mip_feasibility_tolerance` (HiGHS default).
pub const FEASTOL: f64 = 1e-6;
/// `mipdata_->epsilon` (HiGHS default).
pub const EPSILON: f64 = 1e-9;
/// `small_matrix_value` (HiGHS default).
pub const SMALL_MATRIX_VALUE: f64 = 1e-9;
/// `mip_pool_age_limit`.
pub const POOL_AGE_LIMIT: i32 = 30;
/// `mip_pool_soft_limit`.
pub const POOL_SOFT_LIMIT: i64 = 10000;

/// The standard-form root LP of one round.
pub struct StdFormRound<'a> {
    /// Column-wise matrix, `m` rows by `n` columns.
    pub csc: &'a SparseCols,
    /// Rows.
    pub m: usize,
    /// Columns (structural then slacks).
    pub n: usize,
    /// Structural columns (`0..ns`).
    pub ns: usize,
    /// Right-hand sides.
    pub b: &'a [f64],
    /// Column lower bounds (the global box for structural columns).
    pub l: &'a [f64],
    /// Column upper bounds.
    pub u: &'a [f64],
    /// Integrality per column.
    pub is_int: &'a [bool],
    /// Optimal LP point (all `n` columns).
    pub x: &'a [f64],
    /// Row duals.
    pub y: &'a [f64],
    /// Basic column per basis position.
    pub basic_vars: &'a [usize],
}

/// HiGHS's view of the round's LP.
struct View<'a> {
    r: &'a StdFormRound<'a>,
    ns: usize,
    m: usize,
    rows: Vec<(Vec<i32>, Vec<f64>)>,
    rowlo: Vec<f64>,
    rowup: Vec<f64>,
    slb: Vec<f64>,
    sub: Vec<f64>,
    rowval: Vec<f64>,
    maxabs: Vec<f64>,
    isint: Vec<bool>,
    astart: Vec<i32>,
    aindex: Vec<i32>,
    avalue: Vec<f64>,
    contcols: Vec<i32>,
    nintcols: usize,
    colval: Vec<f64>,
    /// Per basis position: HiGHS column and the B^-1 row rescale, or None.
    basis_map: Vec<Option<(i32, f64)>>,
    /// `basis_map`'s HiGHS columns (-1 where unmapped).
    basic_cols: Vec<i32>,
    lu: Option<FeralLU>,
    /// Global column bounds, integer columns' rounded to integers (HiGHS's domain
    /// keeps them integral; the generator shifts integers by these).
    cl: Vec<f64>,
    cu: Vec<f64>,
}

impl<'a> View<'a> {
    fn build(r: &'a StdFormRound<'a>) -> Self {
        let (ns, m, n) = (r.ns, r.m, r.n);
        let mut rows: Vec<(Vec<i32>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); m];
        let (mut astart, mut aindex, mut avalue) = (vec![0i32], Vec::new(), Vec::new());
        for j in 0..ns {
            let (ri, rv) = r.csc.col(j);
            for (&i, &v) in ri.iter().zip(rv) {
                if v != 0.0 {
                    rows[i].0.push(j as i32);
                    rows[i].1.push(v);
                    aindex.push(i as i32);
                    avalue.push(v);
                }
            }
            astart.push(aindex.len() as i32);
        }
        // Slack columns: a column >= ns that is a singleton in a row with no other
        // slack is that row's slack.
        let mut slack_of_row: Vec<Option<(usize, f64)>> = vec![None; m];
        let mut slack_count = vec![0usize; m];
        let mut col_slack_row: Vec<Option<(usize, f64)>> = vec![None; n];
        for j in ns..n {
            let (ri, rv) = r.csc.col(j);
            let nz: Vec<(usize, f64)> = ri
                .iter()
                .zip(rv)
                .filter(|(_, v)| **v != 0.0)
                .map(|(&i, &v)| (i, v))
                .collect();
            for &(i, _) in &nz {
                slack_count[i] += 1;
            }
            if nz.len() == 1 {
                col_slack_row[j] = Some(nz[0]);
                slack_of_row[nz[0].0] = Some((j, nz[0].1));
            }
        }
        let mut rowlo = vec![-INF; m];
        let mut rowup = vec![INF; m];
        for i in 0..m {
            match (slack_count[i], slack_of_row[i]) {
                (0, _) => {
                    rowlo[i] = r.b[i];
                    rowup[i] = r.b[i];
                }
                (1, Some((j, sigma)))
                    if lo_bound(r.l[j]).is_finite() || up_bound(r.u[j]).is_finite() =>
                {
                    // a x + sigma s = b  =>  a x = b - sigma s
                    let (sl, su) = (lo_bound(r.l[j]), up_bound(r.u[j]));
                    let (e1, e2) = (r.b[i] - sigma * sl, r.b[i] - sigma * su);
                    let (lo, up) = if sigma > 0.0 { (e2, e1) } else { (e1, e2) };
                    rowlo[i] = if lo.is_nan() { -INF } else { lo_bound(lo) };
                    rowup[i] = if up.is_nan() { INF } else { up_bound(up) };
                }
                _ => {}
            }
        }
        for j in ns..n {
            if let Some((i, _)) = col_slack_row[j] {
                if slack_count[i] != 1 {
                    col_slack_row[j] = None;
                }
            }
        }
        let colval: Vec<f64> = r.x[..ns].to_vec();
        let mut rowval = vec![0.0f64; m];
        let mut maxabs = vec![0.0f64; m];
        let mut isint = vec![false; ns + m];
        isint[..ns].copy_from_slice(&r.is_int[..ns]);
        let mut slb = vec![-INF; m];
        let mut sub = vec![INF; m];
        for i in 0..m {
            let (ri, rv) = &rows[i];
            let (mut act, mut minact, mut maxact) = (0.0f64, 0.0f64, 0.0f64);
            let mut integral = !ri.is_empty();
            for (&c, &v) in ri.iter().zip(rv) {
                let c = c as usize;
                act += v * colval[c];
                maxabs[i] = maxabs[i].max(v.abs());
                integral = integral && r.is_int[c] && (v - v.round()).abs() <= EPSILON;
                let (cl, cu) = (lo_bound(r.l[c]), up_bound(r.u[c]));
                let (lo, hi) = if v > 0.0 {
                    (v * cl, v * cu)
                } else {
                    (v * cu, v * cl)
                };
                minact += lo;
                maxact += hi;
            }
            rowval[i] = act;
            isint[ns + i] = integral;
            slb[i] = if rowlo[i] > -INF {
                rowlo[i]
            } else if minact.is_nan() {
                -INF
            } else {
                minact
            };
            sub[i] = if rowup[i] < INF {
                rowup[i]
            } else if maxact.is_nan() {
                INF
            } else {
                maxact
            };
            if integral {
                // HiGHS rounds an integral row's bounds at setup
                // (`HighsMipSolverData.cpp:964-971`); the generator shifts the row's
                // (integer) activity by these bounds, so they must be integral.
                if slb[i] > -INF {
                    slb[i] = (slb[i] - FEASTOL).ceil();
                }
                if sub[i] < INF {
                    sub[i] = (sub[i] + FEASTOL).floor();
                }
            }
        }
        let contcols: Vec<i32> = (0..ns)
            .filter(|&j| !r.is_int[j] && r.l[j] != r.u[j])
            .map(|j| j as i32)
            .collect();
        let nintcols = (0..ns).filter(|&j| r.is_int[j] && r.l[j] != r.u[j]).count();
        let basis_map: Vec<Option<(i32, f64)>> = r
            .basic_vars
            .iter()
            .map(|&bv| {
                if bv < ns {
                    Some((bv as i32, 1.0))
                } else {
                    // ours: sigma e_q; HiGHS: -e_q  =>  B_H^-1 row = -sigma * ours.
                    col_slack_row[bv].map(|(q, sigma)| ((ns + q) as i32, -sigma))
                }
            })
            .collect();
        let basic_cols: Vec<i32> = basis_map
            .iter()
            .map(|b| b.map(|p| p.0).unwrap_or(-1))
            .collect();
        let cl: Vec<f64> = (0..ns)
            .map(|j| {
                let l = lo_bound(r.l[j]);
                if r.is_int[j] && l > -INF {
                    (l - FEASTOL).ceil()
                } else {
                    l
                }
            })
            .collect();
        let cu: Vec<f64> = (0..ns)
            .map(|j| {
                let u = up_bound(r.u[j]);
                if r.is_int[j] && u < INF {
                    (u + FEASTOL).floor()
                } else {
                    u
                }
            })
            .collect();
        View {
            r,
            ns,
            m,
            rows,
            rowlo,
            rowup,
            slb,
            sub,
            rowval,
            maxabs,
            isint,
            astart,
            aindex,
            avalue,
            contcols,
            nintcols,
            colval,
            basis_map,
            basic_cols,
            lu: None,
            cl,
            cu,
        }
    }

    fn basis_is_clean(&self) -> bool {
        self.r.basic_vars.len() == self.m && self.basis_map.iter().all(|b| b.is_some())
    }
}

impl TlContext for View<'_> {
    fn num_cols(&self) -> usize {
        self.ns
    }
    fn num_rows(&self) -> usize {
        self.m
    }
    fn feastol(&self) -> f64 {
        FEASTOL
    }
    fn small_matrix_value(&self) -> f64 {
        SMALL_MATRIX_VALUE
    }
    fn col_lower(&self, col: i32) -> f64 {
        self.cl[col as usize]
    }
    fn col_upper(&self, col: i32) -> f64 {
        self.cu[col as usize]
    }
    fn slack_lower(&self, row: i32) -> f64 {
        self.slb[row as usize]
    }
    fn slack_upper(&self, row: i32) -> f64 {
        self.sub[row as usize]
    }
    fn is_col_integral(&self, col: i32) -> bool {
        self.isint[col as usize]
    }
    fn row(&self, row: i32) -> (&[i32], &[f64]) {
        let r = &self.rows[row as usize];
        (&r.0, &r.1)
    }
}

impl SepContext for View<'_> {
    fn epsilon(&self) -> f64 {
        EPSILON
    }
    fn col_value(&self) -> &[f64] {
        &self.colval
    }
    fn row_value(&self) -> &[f64] {
        &self.rowval
    }
    fn row_dual(&self) -> &[f64] {
        self.r.y
    }
    fn row_lower(&self, row: i32) -> f64 {
        self.rowlo[row as usize]
    }
    fn row_upper(&self, row: i32) -> f64 {
        self.rowup[row as usize]
    }
    fn col_matrix(&self) -> (&[i32], &[i32], &[f64]) {
        (&self.astart, &self.aindex, &self.avalue)
    }
    fn continuous_cols(&self) -> &[i32] {
        &self.contcols
    }
    fn num_integral_cols(&self) -> usize {
        self.nintcols
    }
    fn max_abs_row_val(&self, row: i32) -> f64 {
        self.maxabs[row as usize]
    }
}

impl TableauContext for View<'_> {
    fn has_invert(&self) -> bool {
        self.basis_is_clean()
    }
    fn basic_variables(&self) -> &[i32] {
        &self.basic_cols
    }
    fn dual_edge_weights(&self) -> Option<&[f64]> {
        None
    }
    fn basis_inverse_row(&mut self, basis_index: i32) -> Option<Vec<(i32, f64)>> {
        let m = self.m;
        if self.lu.is_none() {
            let bcols: Vec<Vec<(usize, f64)>> = self
                .r
                .basic_vars
                .iter()
                .map(|&bv| {
                    let (ri, rv) = self.r.csc.col(bv);
                    ri.iter().zip(rv).map(|(&i, &a)| (i, a)).collect()
                })
                .collect();
            let mut lu = FeralLU::new();
            lu.factorize_sparse(m, &bcols).ok()?;
            self.lu = Some(lu);
        }
        let (_, scale) = self.basis_map[basis_index as usize]?;
        let mut e = vec![0.0; m];
        e[basis_index as usize] = 1.0;
        let w = crate::lp::gomory::btran_refined(
            self.lu.as_mut()?,
            self.r.csc,
            self.r.basic_vars,
            &e,
            FEASTOL,
        )?;
        Some(
            w.iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.0)
                .map(|(i, &v)| (i as i32, v * scale))
                .collect(),
        )
    }
    fn sepa_lp_iterations(&self) -> i64 {
        0
    }
    fn pool_soft_limit(&self) -> i64 {
        POOL_SOFT_LIMIT
    }
}

/// Diagnostic instrument: `DISCOPT_HIGHS_SEPA_CHECK_POINT=<file>` names a file with
/// one value per structural column of a known feasible (e.g. optimal) point. Every
/// cut entering the pool is checked against it and each violated one is reported
/// on stderr with its source; the counts print at the end of each round. Off when
/// unset; it never changes what is separated.
fn check_point() -> Option<&'static [f64]> {
    static P: std::sync::OnceLock<Option<Vec<f64>>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let path = std::env::var("DISCOPT_HIGHS_SEPA_CHECK_POINT").ok()?;
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("check point {path}: {e}"));
        Some(
            text.split_ascii_whitespace()
                .map(|t| t.parse::<f64>().expect("check point value"))
                .collect(),
        )
    })
    .as_deref()
}

struct PoolSink<'a> {
    pool: &'a mut CutPool,
    source: &'static str,
    /// The reference point, at the top level only.
    probe: Option<&'static [f64]>,
    checked: usize,
    violated: usize,
    /// HiGHS extracts variable bounds from every accepted pool cut of length
    /// <= 100 (`HighsCutPool::addCut` -> `extractCliquesFromCut`).
    vb: Option<(&'a mut VarBoundStore, &'a [f64], &'a [f64], &'a [bool])>,
}

impl CutSink for PoolSink<'_> {
    fn add_cut(&mut self, inds: &[i32], vals: &[f64], rhs: f64, integral: bool) -> i32 {
        if let Some(xr) = self.probe {
            self.checked += 1;
            let act: f64 = inds
                .iter()
                .zip(vals)
                .map(|(&j, &v)| v * xr[j as usize])
                .sum();
            if act > rhs + 1e-6 * (1.0 + rhs.abs()) {
                self.violated += 1;
                eprintln!(
                    "HIGHS_SEPA_INVALID source={} len={} act={act:e} rhs={rhs:e} integral={integral}",
                    self.source,
                    inds.len()
                );
            }
        }
        let idx = self
            .pool
            .add_cut(&mut NoObserver, inds, vals, rhs, integral, true, false);
        if idx >= 0 && inds.len() <= 100 {
            if let Some((store, lb, ub, is_int)) = self.vb.as_mut() {
                store.extract_from_cut(&Domain { lb, ub, is_int }, inds, vals, rhs);
            }
        }
        idx
    }
    fn num_cuts(&self) -> i64 {
        self.pool.num_cuts()
    }
    fn num_available_cuts(&self) -> i64 {
        self.pool.num_available_cuts()
    }
}

/// The persistent separation state of one root (HiGHS's `HighsSeparation` with
/// its separators and the global cut pool).
pub struct RootSeparation {
    pool: CutPool,
    path: PathSeparator,
    tableau: TableauSeparator,
    random_seed: i32,
    lp_iterations: i64,
    model_rows: usize,
    /// The top-level solve (not a sub-MIP): only there is a reference point of the
    /// whole problem meaningful for `DISCOPT_HIGHS_SEPA_CHECK_POINT`.
    top_level: bool,
    /// Variable bounds (HiGHS's clique-table extraction from model rows and pool
    /// cuts); `None` until the first round, or always under
    /// `DISCOPT_HIGHS_SEPA_VBOUNDS=0`.
    vbounds: Option<VarBoundStore>,
    /// Cuts added to the pool across rounds.
    pub cuts_generated: usize,
}

/// Under `DISCOPT_MILP_ROOT_CMIR=highs`: whether the transformed LP substitutes
/// variable bounds (gap plan §12.21). ON with the `highs` arm;
/// `DISCOPT_HIGHS_SEPA_VBOUNDS=0` is the A/B lever. Read per solve, refusing garbage.
pub fn vbounds_enabled() -> bool {
    match std::env::var("DISCOPT_HIGHS_SEPA_VBOUNDS") {
        Err(_) => true,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" | "" => true,
            "0" | "false" | "no" | "off" => false,
            other => {
                panic!("DISCOPT_HIGHS_SEPA_VBOUNDS={other:?} is not recognized (expected 1 or 0)")
            }
        },
    }
}

/// Under `DISCOPT_MILP_ROOT_CMIR=highs`: whether the ported mod-k separator
/// (`HighsModkSeparator`, gap plan §12.24) runs after the path separator, at the
/// root and (with [`node_sepa_enabled`]) at nodes. Default OFF;
/// `DISCOPT_HIGHS_SEPA_MODK=1` opts in. Read per call, refusing garbage.
pub fn modk_enabled() -> bool {
    match std::env::var("DISCOPT_HIGHS_SEPA_MODK") {
        Err(_) => false,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => {
                panic!("DISCOPT_HIGHS_SEPA_MODK={other:?} is not recognized (expected 1 or 0)")
            }
        },
    }
}

/// One round's outcome.
pub struct RoundCuts {
    /// Selected cuts as `coeffs . x >= rhs` over the structural columns, each with
    /// its pool index (report it back through [`RootSeparation::lp_cut_removed`]
    /// when the LP drops the row).
    pub cuts: Vec<(i32, GomoryCut)>,
    /// Cuts the separators added to the pool this round.
    pub generated: usize,
}

impl RootSeparation {
    /// A fresh separation state for a model with `ns` structural columns and the
    /// given original row count and nonzero count (the pool's density limits).
    pub fn new(ns: usize, model_rows: usize, model_nnz: usize, top_level: bool) -> Self {
        RootSeparation {
            pool: CutPool::new(
                ns,
                POOL_AGE_LIMIT,
                POOL_SOFT_LIMIT,
                model_nnz as i64,
                model_rows.max(1) as i64,
            ),
            path: PathSeparator::new(0),
            tableau: TableauSeparator::default(),
            random_seed: 0,
            lp_iterations: 0,
            model_rows,
            top_level,
            vbounds: None,
            cuts_generated: 0,
        }
    }

    /// Variable bounds held (0 when off).
    pub fn num_vbounds(&self) -> i64 {
        self.vbounds.as_ref().map_or(0, |v| v.len())
    }

    /// The LP dropped the row of pool cut `index`.
    pub fn lp_cut_removed(&mut self, index: i32) {
        self.pool.lp_cut_removed(index);
    }

    /// Record LP iterations (they seed each round's cut generators, as in HiGHS).
    pub fn add_lp_iterations(&mut self, iters: i64) {
        self.lp_iterations += iters;
    }

    /// One separation round: tableau then path separator into the pool, then the
    /// pool's selection for the LP.
    pub fn round(&mut self, r: &StdFormRound<'_>) -> RoundCuts {
        let mut view = View::build(r);
        if self.vbounds.is_none() && vbounds_enabled() {
            // HiGHS extracts from the model rows once (`extractCliques`, run by its
            // presolve's probing), over the global box at that point.
            let mut store = VarBoundStore::new(r.ns, FEASTOL);
            let d = Domain {
                lb: &view.cl,
                ub: &view.cu,
                is_int: &view.isint[..r.ns],
            };
            for i in 0..self.model_rows.min(view.m) {
                let (ri, rv) = &view.rows[i];
                store.extract_from_model_row(&d, ri, rv, view.rowlo[i], view.rowup[i], EPSILON);
            }
            self.vbounds = Some(store);
        }
        let empty = Vec::new();
        let vstore = self.vbounds.take();
        let vubs = |c: i32| vstore.as_ref().map_or_else(|| empty.clone(), |v| v.vubs(c));
        let vlbs = |c: i32| vstore.as_ref().map_or_else(|| empty.clone(), |v| v.vlbs(c));
        let intcols: Vec<i32> = (0..r.ns)
            .filter(|&j| r.is_int[j])
            .map(|j| j as i32)
            .collect();
        let setup = TlSetup {
            col_value: &view.colval.clone(),
            col_dual: &vec![0.0; r.ns],
            row_value: &view.rowval.clone(),
            continuous_cols: &view.contcols.clone(),
            integral_cols: &intcols,
            vubs: &vubs,
            vlbs: &vlbs,
        };
        let probe = if self.top_level { check_point() } else { None };
        if let Some(xr) = probe {
            // The instrument is only meaningful if the point satisfies this LP's rows
            // and the view's column bounds; say so before judging any cut by it.
            let (mut bad_rows, mut bad_cols) = (0usize, 0usize);
            for i in 0..view.m {
                let (ri, rv) = &view.rows[i];
                let act: f64 = ri.iter().zip(rv).map(|(&j, &v)| v * xr[j as usize]).sum();
                let tol = 1e-6 * (1.0 + act.abs());
                if act < view.slb[i] - tol || act > view.sub[i] + tol {
                    if bad_rows < 3 {
                        let slack: Vec<String> = (r.ns..r.n)
                            .filter_map(|j| {
                                let (rr, rv) = r.csc.col(j);
                                rr.iter().position(|&q| q == i).map(|p| {
                                    format!(
                                        "col {j} coef {:e} [{:e},{:e}] nnz {}",
                                        rv[p],
                                        r.l[j],
                                        r.u[j],
                                        rr.len()
                                    )
                                })
                            })
                            .collect();
                        eprintln!(
                            "HIGHS_SEPA_POINT row {i} act {act:e} not in [{:e}, {:e}] (lo {:e} up {:e}, b {:e}, len {}, int {}) slack {:?}",
                            view.slb[i], view.sub[i], view.rowlo[i], view.rowup[i], r.b[i], ri.len(), view.isint[view.ns + i], slack
                        );
                    }
                    bad_rows += 1;
                }
            }
            for j in 0..view.ns {
                if xr[j] < view.cl[j] - 1e-6 || xr[j] > view.cu[j] + 1e-6 {
                    if bad_cols < 3 {
                        eprintln!(
                            "HIGHS_SEPA_POINT col {j} x {:e} not in [{:e}, {:e}] int {}",
                            xr[j], view.cl[j], view.cu[j], view.isint[j]
                        );
                    }
                    bad_cols += 1;
                }
            }
            let fr: Vec<usize> = (0..view.ns)
                .filter(|&j| view.isint[j] && (xr[j] - xr[j].round()).abs() > 1e-6)
                .collect();
            for &j in fr.iter().take(4) {
                eprintln!(
                    "HIGHS_SEPA_POINT frac col {j} x {:e} bounds [{:e}, {:e}] lp {:e}",
                    xr[j], view.cl[j], view.cu[j], view.colval[j]
                );
            }
            let frac = fr.len();
            eprintln!(
                "HIGHS_SEPA_POINT ns {} m {} n {} bad_rows {bad_rows} bad_cols {bad_cols} fractional_ints {frac}",
                view.ns, view.m, r.n
            );
        }
        let mut tl = TransformedLp::new(&view, &setup);
        let mut vstore = vstore;
        let (cl, cu, vint) = (
            view.cl.clone(),
            view.cu.clone(),
            view.isint[..r.ns].to_vec(),
        );
        let mut agg = LpAggregator::new(&view);
        let before = self.pool.num_cuts();
        {
            self.tableau.num_calls += 1;
            let seed = self.random_seed as i64 + self.lp_iterations + self.pool.num_cuts();
            let mut sink = PoolSink {
                pool: &mut self.pool,
                source: "tableau",
                probe,
                checked: 0,
                violated: 0,
                vb: vstore.as_mut().map(|v| (v, &cl[..], &cu[..], &vint[..])),
            };
            self.tableau
                .separate(&mut view, &mut agg, &mut tl, &mut sink, seed);
            if probe.is_some() {
                eprintln!(
                    "HIGHS_SEPA_CHECK tableau checked={} violated={}",
                    sink.checked, sink.violated
                );
            }
        }
        {
            self.path.num_calls += 1;
            let seed = self.random_seed as i64 + self.lp_iterations + self.pool.num_cuts();
            let mut sink = PoolSink {
                pool: &mut self.pool,
                source: "path",
                probe,
                checked: 0,
                violated: 0,
                vb: vstore.as_mut().map(|v| (v, &cl[..], &cu[..], &vint[..])),
            };
            self.path
                .separate(&view, &mut agg, &mut tl, &mut sink, seed);
            if probe.is_some() {
                eprintln!(
                    "HIGHS_SEPA_CHECK path checked={} violated={}",
                    sink.checked, sink.violated
                );
            }
        }
        if modk_enabled() {
            let seed = self.random_seed as i64 + self.lp_iterations + self.pool.num_cuts();
            let mut sink = PoolSink {
                pool: &mut self.pool,
                source: "modk",
                probe,
                checked: 0,
                violated: 0,
                vb: vstore.as_mut().map(|v| (v, &cl[..], &cu[..], &vint[..])),
            };
            super::modk::separate_modk(&view, &mut agg, &mut tl, &mut sink, seed);
            if probe.is_some() {
                eprintln!(
                    "HIGHS_SEPA_CHECK modk checked={} violated={}",
                    sink.checked, sink.violated
                );
            }
        }
        self.vbounds = vstore;
        crate::profile::record_max(
            crate::profile::Ctr::HighsSepaVarBounds,
            self.num_vbounds().max(0) as u64,
        );
        let generated = (self.pool.num_cuts() - before).max(0) as usize;
        self.cuts_generated += generated;
        let mut cutset = Vec::new();
        let sel = self.pool.separate(
            &mut NoObserver,
            &view.colval,
            &view.cl,
            &view.cu,
            FEASTOL,
            &mut cutset,
        );
        let cuts = sel
            .into_iter()
            .map(|c| {
                let mut coeffs = vec![0.0; r.ns];
                for (&j, &v) in c.inds.iter().zip(&c.vals) {
                    coeffs[j as usize] = -v;
                }
                (
                    c.index,
                    GomoryCut {
                        coeffs,
                        rhs: -c.rhs,
                    },
                )
            })
            .collect();
        RoundCuts { cuts, generated }
    }
}

impl RootSeparation {
    /// Hand over the variable-bound store after the root (for node separation, which
    /// reads it shared and never adds to it).
    pub fn take_vbounds(&mut self) -> Option<VarBoundStore> {
        self.vbounds.take()
    }
}

/// How tree nodes use the ported separators under `DISCOPT_MILP_ROOT_CMIR=highs`
/// (gap plan §12.23, §12.26).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeSepaMode {
    /// discopt's node GMI only (default).
    Off,
    /// The ported tableau/path separators in place of node GMI (`=1`).
    Replace,
    /// Both: the ported cuts plus node GMI's selection (`=add`).
    Add,
}

/// `DISCOPT_HIGHS_SEPA_NODE`: unset/`0` Off, `1` Replace, `add` Add. Read per call,
/// refusing garbage.
pub fn node_sepa_mode() -> NodeSepaMode {
    match std::env::var("DISCOPT_HIGHS_SEPA_NODE") {
        Err(_) => NodeSepaMode::Off,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" | "replace" => NodeSepaMode::Replace,
            "add" => NodeSepaMode::Add,
            "0" | "false" | "no" | "off" | "" => NodeSepaMode::Off,
            other => {
                panic!("DISCOPT_HIGHS_SEPA_NODE={other:?} is not recognized (expected 0, 1 or add)")
            }
        },
    }
}

/// Whether node separation under the port is on in either mode.
pub fn node_sepa_enabled() -> bool {
    node_sepa_mode() != NodeSepaMode::Off
}

/// The inputs of one node separation call beyond its LP.
pub struct NodeSepa<'a> {
    /// The node's structural column bounds (the pool scores efficacy in the node's
    /// box; validity never reads them).
    pub node_l: &'a [f64],
    /// Upper counterpart of `node_l`.
    pub node_u: &'a [f64],
    /// Original model rows and their nonzeros (the pool's density limits).
    pub model_rows: usize,
    /// Nonzeros of the original rows over the structural columns.
    pub model_nnz: usize,
    /// The root's variable bounds, read only.
    pub vbounds: Option<&'a VarBoundStore>,
    /// Seeds the cut generators (deterministic per node).
    pub seed: i64,
    /// The top-level solve: enables `DISCOPT_HIGHS_SEPA_CHECK_POINT`.
    pub top_level: bool,
    /// Cuts returned at most.
    pub max_cuts: usize,
}

/// One node separation round: HiGHS's tableau and path separators on the node LP
/// `r`, whose `l`/`u` must be the **global** box for every column (structural and
/// slack). The node's own bounds can be tighter (branching, node FBBT on slacks), and
/// a cut derived from them would be valid only in the subtree; deriving every cut
/// over the global box keeps it valid everywhere, as HiGHS's transformed LP does by
/// reading the global domain. Stateless: a fresh pool and fresh separators per call,
/// so nodes evaluated in parallel share nothing. Returns `>=` cuts over the structural
/// columns, in the pool's selection order.
pub fn separate_node(r: &StdFormRound<'_>, p: &NodeSepa<'_>) -> Vec<GomoryCut> {
    let mut view = View::build(r);
    let empty = Vec::new();
    let vubs = |c: i32| p.vbounds.map_or_else(|| empty.clone(), |v| v.vubs(c));
    let vlbs = |c: i32| p.vbounds.map_or_else(|| empty.clone(), |v| v.vlbs(c));
    let intcols: Vec<i32> = (0..r.ns)
        .filter(|&j| r.is_int[j])
        .map(|j| j as i32)
        .collect();
    let colval = view.colval.clone();
    let rowval = view.rowval.clone();
    let contcols = view.contcols.clone();
    let coldual = vec![0.0; r.ns];
    let setup = TlSetup {
        col_value: &colval,
        col_dual: &coldual,
        row_value: &rowval,
        continuous_cols: &contcols,
        integral_cols: &intcols,
        vubs: &vubs,
        vlbs: &vlbs,
    };
    let probe = if p.top_level { check_point() } else { None };
    let mut pool = CutPool::new(
        r.ns,
        POOL_AGE_LIMIT,
        POOL_SOFT_LIMIT,
        p.model_nnz as i64,
        p.model_rows.max(1) as i64,
    );
    let mut tl = TransformedLp::new(&view, &setup);
    let mut agg = LpAggregator::new(&view);
    let (mut checked, mut violated) = (0usize, 0usize);
    {
        let mut tableau = TableauSeparator {
            num_calls: 1,
            ..Default::default()
        };
        let mut sink = PoolSink {
            pool: &mut pool,
            source: "node-tableau",
            probe,
            checked: 0,
            violated: 0,
            vb: None,
        };
        tableau.separate(&mut view, &mut agg, &mut tl, &mut sink, p.seed);
        checked += sink.checked;
        violated += sink.violated;
    }
    {
        let mut path = PathSeparator::new(0);
        path.num_calls += 1;
        let seed = p.seed + pool.num_cuts();
        let mut sink = PoolSink {
            pool: &mut pool,
            source: "node-path",
            probe,
            checked: 0,
            violated: 0,
            vb: None,
        };
        path.separate(&view, &mut agg, &mut tl, &mut sink, seed);
        checked += sink.checked;
        violated += sink.violated;
    }
    if modk_enabled() {
        let seed = p.seed + pool.num_cuts();
        let mut sink = PoolSink {
            pool: &mut pool,
            source: "node-modk",
            probe,
            checked: 0,
            violated: 0,
            vb: None,
        };
        super::modk::separate_modk(&view, &mut agg, &mut tl, &mut sink, seed);
        checked += sink.checked;
        violated += sink.violated;
    }
    if probe.is_some() {
        eprintln!("HIGHS_SEPA_NODE_CHECK checked={checked} violated={violated}");
    }
    let nl: Vec<f64> = p.node_l[..r.ns].iter().map(|&v| lo_bound(v)).collect();
    let nu: Vec<f64> = p.node_u[..r.ns].iter().map(|&v| up_bound(v)).collect();
    let mut cutset = Vec::new();
    let sel = pool.separate(
        &mut NoObserver,
        &view.colval,
        &nl,
        &nu,
        FEASTOL,
        &mut cutset,
    );
    sel.into_iter()
        .take(p.max_cuts)
        .map(|c| {
            let mut coeffs = vec![0.0; r.ns];
            for (&j, &v) in c.inds.iter().zip(&c.vals) {
                coeffs[j as usize] = -v;
            }
            GomoryCut {
                coeffs,
                rhs: -c.rhs,
            }
        })
        .collect()
}
