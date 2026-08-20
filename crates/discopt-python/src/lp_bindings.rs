//! PyO3 bindings for the LP crossover + basis recovery (`discopt_core::lp`).
//!
//! The LP is passed in standard form `min cᵀx s.t. A x = b, l ≤ x ≤ u` with `A`
//! a C-contiguous `m × n` array (the Python side builds this via
//! `_decompose_eq_slack_form`). These let the Python B&B cut loop push an
//! interior POUNCE optimum to a vertex and recover a simplex basis for
//! basis-derived cuts.
//!
// PyO3 entry points are necessarily flat (one parameter per Python argument)
// and return Python-owned array tuples, so the argument-count and
// type-complexity lints don't meaningfully apply to these binding shims.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use discopt_core::bnb::milp_driver::{
    solve_milp_lazy_hooked as core_solve_milp_lazy_hooked, MilpCheckpoint, MilpDebugControl,
    MilpDebugHook, MilpDebugState, MilpLazyHook, MilpLazyVerdict, MilpOptions, MilpStatus,
};
use discopt_core::lp::aggregation::separate_aggregation_mir;
use discopt_core::lp::basis::{recover_basis, Basis, BASIC};
use discopt_core::lp::crossover::{crossover_to_vertex, LpView};
use discopt_core::lp::gomory::{separate_gomory, GomoryCut};
use discopt_core::lp::mir::separate_mir;
use discopt_core::lp::simplex::{
    solve_lp as simplex_solve_lp, solve_lp_batch, solve_lp_warm, solve_lp_warm_csc,
    unstable_pivot_recovery_default, LpInstance, LpStatus, SimplexOptions, SparseCols,
};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Turn an optional per-solve wall-clock budget (seconds from now) into the absolute
/// [`SimplexOptions::deadline`] the simplex loops poll.
///
/// `None` and `+inf` both mean "no limit" — `+inf` is the natural way for a caller
/// with an uncapped budget to spell it, and rejecting it would push a `ValueError`
/// into a defensive `except` and silently disable the caller's fast path. `Some(0.0)`
/// is an already-elapsed deadline, i.e. "my budget is spent, return immediately",
/// which is exactly what a caller whose outer deadline has passed means. Mirrors the
/// validation in `spatial_bindings.rs`.
fn parse_deadline(time_limit_s: Option<f64>) -> PyResult<Option<std::time::Instant>> {
    match time_limit_s {
        None => Ok(None),
        Some(seconds) if seconds == f64::INFINITY => Ok(None),
        Some(seconds) if seconds.is_nan() || seconds < 0.0 => Err(PyValueError::new_err(
            "time_limit_s must be non-negative and not NaN (use None or +inf for no limit)",
        )),
        Some(seconds) => {
            let duration = std::time::Duration::try_from_secs_f64(seconds)
                .map_err(|_| PyValueError::new_err("time_limit_s is too large"))?;
            Ok(Some(
                std::time::Instant::now()
                    .checked_add(duration)
                    .ok_or_else(|| PyValueError::new_err("time_limit_s is too large"))?,
            ))
        }
    }
}

/// The [`parse_deadline`] contract in *seconds*, for callees that build their own
/// deadline from a duration ([`MilpOptions::time_limit_s`] is relative to the driver's
/// own `t_start`, so it cannot take an absolute `Instant`).
///
/// The distinction this exists to preserve is between an **expired** budget and **no**
/// budget. Both MILP entries used to take a bare `f64` and map it with
/// `if time_limit_s > 0.0 { Some(t) } else { None }`, which collapses `0.0` — "my budget
/// is spent" — onto the same wire value as "run forever". That is the sentinel-collapse
/// class documented for `INF` in the LP layer, one level up: a caller that had *already*
/// exhausted a shared budget across earlier attempts launched an **unbounded** MILP
/// B&B at exactly the moment it should not have started at all. Measured on issue #928:
/// AMP's `solve(time_limit=3.0)` ran past 350 s. `Some(0.0)` now reaches the driver
/// intact, where it is an already-elapsed deadline that stops at the first poll with
/// `gap_certified = false` — the same well-trodden path as a deadline that expires at
/// node 1, so the bound it reports stays sound.
fn parse_budget_secs(time_limit_s: Option<f64>) -> PyResult<Option<f64>> {
    match time_limit_s {
        None => Ok(None),
        Some(seconds) if seconds == f64::INFINITY => Ok(None),
        Some(seconds) if seconds.is_nan() || seconds < 0.0 => Err(PyValueError::new_err(
            "time_limit_s must be non-negative and not NaN (use None or +inf for no limit)",
        )),
        // Reject what the driver's `Duration::from_secs_f64` would panic on.
        Some(seconds) => match std::time::Duration::try_from_secs_f64(seconds) {
            Ok(_) => Ok(Some(seconds)),
            Err(_) => Err(PyValueError::new_err("time_limit_s is too large")),
        },
    }
}

/// Refuse a NaN entry in an LP's variable box.
///
/// A NaN bound has no meaning in an LP, and — worse — it reads *differently
/// depending on which way the comparison is written*, because every comparison
/// against NaN is false. The simplex asks "is this side open?" both ways:
/// `ub < INF` (the ratio test's blocking check) calls a NaN upper bound **open**,
/// while `ub >= INF` (the unbounded-ray box-recession check) calls the same bound
/// **closed**. On issue #1008 that split produced an LP the ratio test walked to
/// `t = INF` on and reported `unbounded`, over a box it could not certify as
/// recessive. This is the `INF`-is-`1e20` hazard from CLAUDE.md in its other
/// guise: there the sentinel silently survived a multiplication, here it is
/// silently absent.
///
/// The modeling layer spells "no bound" as NaN (`Model.continuous(ub=None)` →
/// `array(nan)`); the LP layer spells it as the sentinel `±1e20`. Translating
/// between the two is the caller's job — `lp_simplex._finite_box` does it — and
/// this refuses loudly rather than let an untranslated box reach the simplex and
/// be read two ways. `±inf` is *not* rejected: it satisfies `>= INF` and fails
/// `< INF`, so both readings agree it is open.
fn check_box_not_nan(lb: &[f64], ub: &[f64]) -> PyResult<()> {
    for (name, v) in [("lb", lb), ("ub", ub)] {
        if let Some(j) = v.iter().position(|x| x.is_nan()) {
            return Err(PyValueError::new_err(format!(
                "{name}[{j}] is NaN; an LP bound must be finite or the ±1e20 \
                 unbounded sentinel (NaN is read as both open and closed by \
                 different guards — see issue #1008)"
            )));
        }
    }
    Ok(())
}

/// Push an interior LP optimum `x` to a vertex of the optimal face.
///
/// `a` is the C-contiguous `m × n` equality-constraint matrix; `c`, `lb`, `ub`
/// are length `n`. Returns the vertex as a new length-`n` array (same objective
/// and feasibility as `x`). `max_iter = 0` selects the `n + 1` default.
#[pyfunction]
#[pyo3(signature = (x, a, c, lb, ub, tol=1e-7, max_iter=0))]
pub fn crossover_to_vertex_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dims = a.shape();
    let (m, n) = (dims[0], dims[1]);
    let a_flat = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?;
    let lp = LpView {
        a: a_flat,
        m,
        n,
        c: c.as_slice()?,
        l: lb.as_slice()?,
        u: ub.as_slice()?,
    };
    let xv = crossover_to_vertex(x.as_slice()?, &lp, tol, max_iter);
    Ok(PyArray1::from_vec(py, xv))
}

/// Recover a simplex basis at the vertex `x` of the standard-form LP.
///
/// Returns `(col_status, basic_vars)` — `col_status` is a length-`n` `int8`
/// array of HiGHS `HighsBasisStatus` codes (`0`=AtLower, `1`=Basic,
/// `2`=AtUpper) and `basic_vars` the `m` basic column indices — or `None` when
/// `x` is not a basic feasible solution (see `recover_basis`).
#[pyfunction]
#[pyo3(signature = (x, a, c, lb, ub, tol=1e-7))]
pub fn recover_basis_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    tol: f64,
) -> PyResult<Option<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>)>> {
    let dims = a.shape();
    let (m, n) = (dims[0], dims[1]);
    let a_flat = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?;
    let lp = LpView {
        a: a_flat,
        m,
        n,
        c: c.as_slice()?,
        l: lb.as_slice()?,
        u: ub.as_slice()?,
    };
    match recover_basis(x.as_slice()?, &lp, tol) {
        Some(b) => {
            let status = PyArray1::from_vec(py, b.col_status);
            let basic: Vec<i64> = b.basic_vars.iter().map(|&v| v as i64).collect();
            Ok(Some((status, PyArray1::from_vec(py, basic))))
        }
        None => Ok(None),
    }
}

/// Separate Gomory mixed-integer cuts at the vertex `x` of the standard-form LP.
///
/// Recovers a basis at `x`, reconstructs the vertex from the exact basis +
/// bounds + `b` (the length-`m` rhs of `A x = b`), and derives one GMI cut per
/// fractional integer basic variable. `integrality` is a length-`n` bool array.
/// Returns `(coeffs, rhs)` — `coeffs` is a `k × n` array and `rhs` length `k`,
/// the cuts `coeffs[i] · x ≥ rhs[i]` over the standard-form variables — or
/// `None` when `x` is not a basic feasible solution (basis recovery declined).
#[pyfunction]
#[pyo3(signature = (x, a, b, c, lb, ub, integrality, tol=1e-7, max_dynamism=1e7))]
pub fn gomory_cuts_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, bool>,
    tol: f64,
    max_dynamism: f64,
) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)>> {
    let dims = a.shape();
    let (m, n) = (dims[0], dims[1]);
    let a_flat = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?;
    let lp = LpView {
        a: a_flat,
        m,
        n,
        c: c.as_slice()?,
        l: lb.as_slice()?,
        u: ub.as_slice()?,
    };
    let xs = x.as_slice()?;
    let basis = match recover_basis(xs, &lp, tol) {
        Some(bs) => bs,
        None => return Ok(None),
    };
    let cuts = separate_gomory(
        &lp,
        b.as_slice()?,
        &basis,
        integrality.as_slice()?,
        tol,
        max_dynamism,
    );

    let k = cuts.len();
    let mut flat = Vec::with_capacity(k * n);
    let mut rhs = Vec::with_capacity(k);
    for cut in &cuts {
        flat.extend_from_slice(&cut.coeffs);
        rhs.push(cut.rhs);
    }
    let coeffs = PyArray1::from_vec(py, flat).reshape([k, n])?;
    Ok(Some((coeffs, PyArray1::from_vec(py, rhs))))
}

/// Separate MIR cuts from the `≤` rows `a_ub · x ≤ b_ub` at point `x`.
///
/// `a_ub` is C-contiguous `m × n`; `lb`/`ub` the length-`n` lower/upper bounds
/// (used for the lower-shift / upper-complement bound substitution — pass `+inf`
/// in `ub[j]` to disable complementation for column `j`); `integrality` a
/// length-`n` bool array. Returns `(coeffs, rhs)` — a `k × n` array and length-`k`
/// rhs, the cuts `coeffs[i] · x ≤ rhs[i]` over the structural variables — or
/// `None` when no cut is produced.
#[pyfunction]
#[pyo3(signature = (a_ub, b_ub, lb, ub, integrality, x, tol=1e-7, max_dynamism=1e7))]
pub fn mir_cuts_py<'py>(
    py: Python<'py>,
    a_ub: PyReadonlyArray2<'py, f64>,
    b_ub: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, bool>,
    x: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_dynamism: f64,
) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)>> {
    let dims = a_ub.shape();
    let n = dims[1];
    let a_flat = a_ub
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a_ub` must be C-contiguous"))?;
    let cuts = separate_mir(
        a_flat,
        b_ub.as_slice()?,
        lb.as_slice()?,
        ub.as_slice()?,
        integrality.as_slice()?,
        x.as_slice()?,
        tol,
        max_dynamism,
    );
    let k = cuts.len();
    if k == 0 {
        return Ok(None);
    }
    let mut flat = Vec::with_capacity(k * n);
    let mut rhs = Vec::with_capacity(k);
    for cut in &cuts {
        flat.extend_from_slice(&cut.coeffs);
        rhs.push(cut.rhs);
    }
    let coeffs = PyArray1::from_vec(py, flat).reshape([k, n])?;
    Ok(Some((coeffs, PyArray1::from_vec(py, rhs))))
}

/// Separate Marchand–Wolsey aggregation c-MIR cuts from the `≤` rows
/// `a_ub · x ≤ b_ub` at point `x`.
///
/// Pairs rows with nonnegative weights to cancel a continuous variable, forms the
/// valid implied aggregate row, and applies the same complemented MIR as
/// [`mir_cuts_py`] to it — so every cut is valid for the original feasible set (a
/// nonnegative row combination of `≤` rows plus a valid MIR; see
/// `discopt_core::lp::aggregation`). `a_ub` is C-contiguous `m × n`; `lb`/`ub` are
/// length-`n` bounds (`+inf` in `ub[j]` disables complementation for column `j`);
/// `integrality` a length-`n` bool array. Returns `(coeffs, rhs)` — a `k × n`
/// array and length-`k` rhs, the cuts `coeffs[i] · x ≤ rhs[i]` over the structural
/// variables, ordered most-violated-first — or `None` when no cut is produced.
#[pyfunction]
#[pyo3(signature = (a_ub, b_ub, lb, ub, integrality, x, tol=1e-7, max_dynamism=1e7))]
pub fn aggregation_mir_cuts_py<'py>(
    py: Python<'py>,
    a_ub: PyReadonlyArray2<'py, f64>,
    b_ub: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, bool>,
    x: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_dynamism: f64,
) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)>> {
    let dims = a_ub.shape();
    let n = dims[1];
    let a_flat = a_ub
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a_ub` must be C-contiguous"))?;
    let mut cuts = separate_aggregation_mir(
        a_flat,
        b_ub.as_slice()?,
        lb.as_slice()?,
        ub.as_slice()?,
        integrality.as_slice()?,
        x.as_slice()?,
        tol,
        max_dynamism,
    );
    if cuts.is_empty() {
        return Ok(None);
    }
    // Most-violated first, deterministic tie-break by insertion order.
    cuts.sort_by(|p, q| {
        q.violation
            .partial_cmp(&p.violation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = cuts.len();
    let mut flat = Vec::with_capacity(k * n);
    let mut rhs = Vec::with_capacity(k);
    for ac in &cuts {
        flat.extend_from_slice(&ac.cut.coeffs);
        rhs.push(ac.cut.rhs);
    }
    let coeffs = PyArray1::from_vec(py, flat).reshape([k, n])?;
    Ok(Some((coeffs, PyArray1::from_vec(py, rhs))))
}

/// Solve a standard-form LP `min cᵀx s.t. A x = b, lb ≤ x ≤ ub` with the
/// warm-startable revised simplex (cold start). `a` is C-contiguous `m × n`.
/// Returns `(status, x, obj, iters)` where status is one of `optimal`,
/// `infeasible`, `unbounded`, `iter_limit`, `numerical`. For validation against
/// HiGHS / Netlib.
#[pyfunction]
#[pyo3(signature = (c, a, b, lb, ub, tol=1e-9, max_iter=100_000))]
pub fn solve_lp_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(String, Bound<'py, PyArray1<f64>>, f64, usize)> {
    let dims = a.shape();
    check_box_not_nan(lb.as_slice()?, ub.as_slice()?)?;
    let (m, n) = (dims[0], dims[1]);
    let a_flat = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?;
    let lp = LpView {
        a: a_flat,
        m,
        n,
        c: c.as_slice()?,
        l: lb.as_slice()?,
        u: ub.as_slice()?,
    };
    let opts = SimplexOptions {
        tol,
        max_iter,
        deadline: None,
        // F2: warm dual-simplex stall guard on by default (size-derived cap →
        // cold fallback on trip; bound-neutral). Cold-only entry points ignore it.
        warm_stall_guard: true,
        warm_stall_cap_override: None,
        expel_zero_artificials: false,
        bank_deadline_duals: false,
        recover_unstable_pivot: false,
        // #1013: hand a STALLED warm dual re-solve to the cold solve (default from
        // `DISCOPT_LP_DUAL_STALL_BAIL`). Inert on the cold-only entry points.
        dual_stall_patience: SimplexOptions::default().dual_stall_patience,
        cold_dual_start: false,
    };
    let sol = simplex_solve_lp(&lp, b.as_slice()?, &opts);
    let status = match sol.status {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
        LpStatus::IterLimit => "iter_limit",
        LpStatus::Numerical => "numerical",
    };
    Ok((
        status.to_string(),
        PyArray1::from_vec(py, sol.x),
        sol.obj,
        sol.iters,
    ))
}

/// Build a dual-simplex warm-start basis from a previous solve's
/// `(col_status, basic_vars)`, extending it for rows/columns appended since.
///
/// The cutting-plane loop re-solves the SAME structural columns with only rows
/// (cuts) appended, and the standard form is `[A_ub | I]` — one slack column per
/// row — so each appended row adds exactly one trailing slack column. A starting
/// basis with `n_old` columns / `m_old` rows is therefore valid for the current
/// `n`/`m` iff `n - n_old == m - m_old >= 0`: the new columns are the appended
/// slacks, which we make basic (the previous vertex stays a basis of the larger
/// system, dual-feasible, so the dual simplex re-optimizes). Any inconsistency
/// returns `None`, and the caller cold-starts — so this only affects speed.
fn build_extended_basis(cs: &[i8], bv: &[i64], n: usize, m: usize) -> Option<Basis> {
    let n_old = cs.len();
    let m_old = bv.len();
    if n_old > n || m_old > m {
        return None;
    }
    let dn = n - n_old;
    let dm = m - m_old;
    if dn != dm {
        return None; // not a clean one-slack-per-appended-row growth
    }
    let mut col_status: Vec<i8> = Vec::with_capacity(n);
    col_status.extend_from_slice(cs);
    col_status.resize(n, BASIC); // appended slacks (cols n_old..n) enter the basis
    let mut basic_vars: Vec<usize> = Vec::with_capacity(m);
    for &v in bv {
        if v < 0 || (v as usize) >= n_old {
            return None;
        }
        basic_vars.push(v as usize);
    }
    for j in n_old..n {
        basic_vars.push(j);
    }
    // Enforce col_status/basic_vars consistency in case the incoming pair was
    // slightly stale; `PreparedDual::prepare` validates further and the warm
    // solver cold-falls-back on any residual inconsistency.
    for &v in &basic_vars {
        col_status[v] = BASIC;
    }
    Some(Basis {
        col_status,
        basic_vars,
    })
}

/// Warm-startable standard-form LP solve: same problem as [`solve_lp_py`]
/// (`min cᵀx s.t. A x = b, lb ≤ x ≤ ub`), but it accepts an optional starting
/// basis (`start_col_status` length `n'`, `start_basic_vars` length `m'`) and
/// returns the final basis alongside the solution. When the starting basis has
/// fewer rows/columns than the current LP (rows appended since — the
/// cutting-plane case), it is extended by making the appended slack columns
/// basic so the dual simplex re-optimizes from the previous vertex.
///
/// Soundness: a missing/mismatched/singular basis is silently ignored
/// (`solve_lp_warm` cold-falls-back internally), and the dual simplex converges
/// to the LP optimum just like a cold solve — so the returned objective (hence
/// any relaxation bound built on it) is unchanged; the basis only changes speed.
/// Returns `(status, x, obj, iters, col_status, basic_vars, dual, ray)`.
///
/// `dual` (length `m`) and `ray` (length `n`) are *certificate candidates* a
/// caller verifies before trusting (see [`LpSolve::dual`]/[`LpSolve::ray`]): at
/// `optimal` `dual` is the row duals (feed to a Neumaier–Shcherbina safe bound),
/// at `infeasible` `dual` is a Farkas ray, at `unbounded` `ray` is a primal ray.
/// They are mapped back from any internal equilibration, so they are consistent
/// with the `a`/`b`/`lb`/`ub` passed in. Each is empty when not applicable.
#[pyfunction]
#[pyo3(signature = (c, a, b, lb, ub, start_col_status=None, start_basic_vars=None,
                    tol=1e-9, max_iter=100_000))]
#[allow(clippy::too_many_arguments)]
pub fn solve_lp_warm_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    start_col_status: Option<PyReadonlyArray1<'py, i8>>,
    start_basic_vars: Option<PyReadonlyArray1<'py, i64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(
    String,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let dims = a.shape();
    check_box_not_nan(lb.as_slice()?, ub.as_slice()?)?;
    let (m, n) = (dims[0], dims[1]);
    let a_flat = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?;
    let lp = LpView {
        a: a_flat,
        m,
        n,
        c: c.as_slice()?,
        l: lb.as_slice()?,
        u: ub.as_slice()?,
    };
    let opts = SimplexOptions {
        tol,
        max_iter,
        deadline: None,
        // F2: warm dual-simplex stall guard on by default (size-derived cap →
        // cold fallback on trip; bound-neutral). Cold-only entry points ignore it.
        warm_stall_guard: true,
        warm_stall_cap_override: None,
        expel_zero_artificials: false,
        bank_deadline_duals: false,
        recover_unstable_pivot: false,
        // #1013: hand a STALLED warm dual re-solve to the cold solve (default from
        // `DISCOPT_LP_DUAL_STALL_BAIL`). Inert on the cold-only entry points.
        dual_stall_patience: SimplexOptions::default().dual_stall_patience,
        cold_dual_start: false,
    };
    let b_slice = b.as_slice()?;

    let start = match (start_col_status, start_basic_vars) {
        (Some(cs), Some(bv)) => build_extended_basis(cs.as_slice()?, bv.as_slice()?, n, m),
        _ => None,
    };
    let sol = match start {
        Some(basis) => solve_lp_warm(&lp, b_slice, &basis, &opts),
        None => simplex_solve_lp(&lp, b_slice, &opts),
    };
    let status = match sol.status {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
        LpStatus::IterLimit => "iter_limit",
        LpStatus::Numerical => "numerical",
    };
    let basic_vars_i64: Vec<i64> = sol.basis.basic_vars.iter().map(|&v| v as i64).collect();
    Ok((
        status.to_string(),
        PyArray1::from_vec(py, sol.x),
        sol.obj,
        sol.iters,
        PyArray1::from_vec(py, sol.basis.col_status),
        PyArray1::from_vec(py, basic_vars_i64),
        PyArray1::from_vec(py, sol.dual),
        PyArray1::from_vec(py, sol.ray),
    ))
}

/// Sparse-native counterpart of [`solve_lp_warm_py`]: the constraint matrix is
/// passed as CSC (`col_ptr`/`row_idx`/`vals`, column-major) instead of a dense
/// `m × n` array, so the ~0.3%-dense lifted relaxations are never materialized
/// dense on either side of the boundary. `n` is the total column count (structural
/// + slacks, matching the CSC). Returns the same
/// `(status, x, obj, iters, col_status, basic_vars, dual, ray)` 8-tuple as
/// [`solve_lp_warm_py`], with the certificates mapped back from any equilibration.
///
/// `time_limit_s` is an optional wall-clock budget for this one LP, in seconds:
/// `None` (the default) means no limit, and the simplex runs to convergence. It maps
/// straight onto [`SimplexOptions::deadline`], which the dual pivot loop polls every
/// 256 pivots and the primal likewise, so a stalling LP yields instead of running
/// unbounded. Note this is `Option`-typed on purpose and does NOT share
/// [`solve_milp_csc_py`]'s convention, where a bare `0.0` spells "no limit": here
/// `Some(0.0)` is an already-elapsed deadline that returns immediately, which is
/// what a caller with a spent budget means. Without this parameter every caller that
/// computed a per-LP budget had it silently dropped — on nvs24 one node LP ran 47 s
/// against the 0.2 s its caller passed, 59 494 degenerate dual pivots deep, turning
/// a 3.9 s solve budget into 53 s.
#[pyfunction]
#[pyo3(signature = (c, m, n, col_ptr, row_idx, vals, b, lb, ub,
                    start_col_status=None, start_basic_vars=None, tol=1e-9, max_iter=100_000,
                    time_limit_s=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_lp_warm_csc_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    m: usize,
    n: usize,
    col_ptr: PyReadonlyArray1<'py, i64>,
    row_idx: PyReadonlyArray1<'py, i64>,
    vals: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    start_col_status: Option<PyReadonlyArray1<'py, i8>>,
    start_basic_vars: Option<PyReadonlyArray1<'py, i64>>,
    tol: f64,
    max_iter: usize,
    time_limit_s: Option<f64>,
) -> PyResult<(
    String,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    // THRU-5: honor DISCOPT_PROFILE on the pure-LP node-bound path too. Instances
    // that never call solve_milp_py (their node bound is a pure warm LP) would
    // otherwise leave profiling uninitialized and dump() a no-op. Cheap; first call
    // per process fixes the flag.
    discopt_core::profile::init_from_env();
    check_box_not_nan(lb.as_slice()?, ub.as_slice()?)?;
    let col_ptr: Vec<usize> = col_ptr.as_slice()?.iter().map(|&x| x as usize).collect();
    let row_idx: Vec<usize> = row_idx.as_slice()?.iter().map(|&x| x as usize).collect();
    let vals_v: Vec<f64> = vals.as_slice()?.to_vec();
    let sp = SparseCols::from_csc(col_ptr, row_idx, vals_v);
    let deadline = parse_deadline(time_limit_s)?;
    let opts = SimplexOptions {
        tol,
        max_iter,
        deadline,
        // F2: warm dual-simplex stall guard on by default (size-derived cap →
        // cold fallback on trip; bound-neutral). Cold-only entry points ignore it.
        warm_stall_guard: true,
        warm_stall_cap_override: None,
        expel_zero_artificials: false,
        // #928: this is the one entry point whose deadline-cut solves bank the
        // dual loop's anytime floor (see `SimplexOptions::bank_deadline_duals`).
        // Engaged exactly when the caller bounded this LP in time; the MILP
        // driver's own deadline route keeps the default `false`, so the default
        // B&B pivot path is untouched.
        bank_deadline_duals: deadline.is_some(),
        // #1008 R1: the near-zero-pivot recovery used to ride on the line above,
        // so a caller who passed no `time_limit` lost it — measured on
        // QPLIB_2170, that is the difference between `optimal 0` and no bound at
        // all. It now has its own gate. `deadline.is_some()` keeps the deadline
        // path bit-identical to what its own graduation panel judged; the env
        // opt-in extends it to deadline-free callers and is default-OFF until the
        // §5 panel graduates it.
        recover_unstable_pivot: deadline.is_some() || unstable_pivot_recovery_default(),
        // #1013: hand a STALLED warm dual re-solve to the cold solve (default from
        // `DISCOPT_LP_DUAL_STALL_BAIL`). Inert on the cold-only entry points.
        dual_stall_patience: SimplexOptions::default().dual_stall_patience,
        cold_dual_start: false,
    };
    let start = match (start_col_status, start_basic_vars) {
        (Some(cs), Some(bv)) => build_extended_basis(cs.as_slice()?, bv.as_slice()?, n, m),
        _ => None,
    };
    let sol = solve_lp_warm_csc(
        sp,
        m,
        n,
        c.as_slice()?,
        lb.as_slice()?,
        ub.as_slice()?,
        b.as_slice()?,
        start.as_ref(),
        &opts,
    );
    // THRU-5: emit the per-node warm-dual phase profile to stderr when
    // DISCOPT_PROFILE is set (no-op otherwise), so the pure-LP node-bound path
    // (the dominant per-node cost) is visible alongside the solve_milp_py driver.
    discopt_core::profile::dump();
    let status = match sol.status {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
        LpStatus::IterLimit => "iter_limit",
        LpStatus::Numerical => "numerical",
    };
    let basic_vars_i64: Vec<i64> = sol.basis.basic_vars.iter().map(|&v| v as i64).collect();
    Ok((
        status.to_string(),
        PyArray1::from_vec(py, sol.x),
        sol.obj,
        sol.iters,
        PyArray1::from_vec(py, sol.basis.col_status),
        PyArray1::from_vec(py, basic_vars_i64),
        PyArray1::from_vec(py, sol.dual),
        PyArray1::from_vec(py, sol.ray),
    ))
}

/// Solve a batch of LPs that share the constraint matrix `a` (C-contiguous
/// `m × n`) and objective `c`, each with its own right-hand side and bounds. The
/// per-instance data are stacked: `b` is `k × m`, `lb`/`ub` are `k × n`. The
/// equilibration scaling and scaled matrix are computed once and reused across
/// the batch, and the instances are solved in parallel.
///
/// Returns `(statuses, x, objs)` where `statuses` is a length-`k` list of
/// Read every profiling counter as a dict, without printing or resetting.
///
/// `profile::dump()` writes to stderr and zeroes the accumulators, so it cannot
/// serve as a measurement instrument for a caller that wants the numbers back. The
/// #956 follow-through needs the simplex's terminal-verdict histogram in Python to
/// tell an uncertified infeasibility apart from a drifted optimum or a
/// factorization breakdown. Counters only accumulate while `DISCOPT_PROFILE` is
/// set, so this returns zeros otherwise.
#[pyfunction]
pub fn profile_counters_py(py: Python<'_>) -> PyResult<PyObject> {
    discopt_core::profile::init_from_env();
    let out = PyDict::new(py);
    for (name, value) in discopt_core::profile::counter_snapshot() {
        out.set_item(name, value)?;
    }
    Ok(out.into())
}

/// Zero every profiling counter, so a caller can bracket one measurement.
#[pyfunction]
pub fn profile_reset_py() {
    discopt_core::profile::init_from_env();
    discopt_core::profile::reset_totals();
}

/// strings (`optimal`/`infeasible`/`unbounded`/`iter_limit`/`numerical`), `x` is
/// a `k × n` array of solutions, and `objs` is length `k`.
#[pyfunction]
#[pyo3(signature = (c, a, b, lb, ub, tol=1e-9, max_iter=100_000))]
pub fn solve_lp_batch_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    lb: PyReadonlyArray2<'py, f64>,
    ub: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(
    Vec<String>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let dims = a.shape();
    check_box_not_nan(lb.as_slice()?, ub.as_slice()?)?;
    let (m, n) = (dims[0], dims[1]);
    let k = b.shape()[0];
    if b.shape()[1] != m {
        return Err(PyValueError::new_err("`b` must be k × m"));
    }
    if lb.shape() != [k, n] || ub.shape() != [k, n] {
        return Err(PyValueError::new_err("`lb`/`ub` must be k × n"));
    }
    let a_owned: Vec<f64> = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?
        .to_vec();
    let c_owned: Vec<f64> = c.as_slice()?.to_vec();
    let b_flat = b
        .as_slice()
        .map_err(|_| PyValueError::new_err("`b` must be C-contiguous"))?;
    let lb_flat = lb
        .as_slice()
        .map_err(|_| PyValueError::new_err("`lb` must be C-contiguous"))?;
    let ub_flat = ub
        .as_slice()
        .map_err(|_| PyValueError::new_err("`ub` must be C-contiguous"))?;
    let instances: Vec<LpInstance> = (0..k)
        .map(|t| LpInstance {
            b: b_flat[t * m..(t + 1) * m].to_vec(),
            l: lb_flat[t * n..(t + 1) * n].to_vec(),
            u: ub_flat[t * n..(t + 1) * n].to_vec(),
        })
        .collect();
    let opts = SimplexOptions {
        tol,
        max_iter,
        deadline: None,
        // F2: warm dual-simplex stall guard on by default (size-derived cap →
        // cold fallback on trip; bound-neutral). Cold-only entry points ignore it.
        warm_stall_guard: true,
        warm_stall_cap_override: None,
        expel_zero_artificials: false,
        bank_deadline_duals: false,
        recover_unstable_pivot: false,
        // #1013: hand a STALLED warm dual re-solve to the cold solve (default from
        // `DISCOPT_LP_DUAL_STALL_BAIL`). Inert on the cold-only entry points.
        dual_stall_patience: SimplexOptions::default().dual_stall_patience,
        cold_dual_start: false,
    };
    // The solve touches no Python objects, so release the GIL to let the core's
    // rayon workers run the batch concurrently without contending on it.
    let sols = py.allow_threads(|| solve_lp_batch(&a_owned, m, n, &c_owned, &instances, &opts));

    let mut statuses = Vec::with_capacity(k);
    let mut x_flat = vec![0.0f64; k * n];
    let mut objs = vec![0.0f64; k];
    for (t, sol) in sols.iter().enumerate() {
        statuses.push(
            match sol.status {
                LpStatus::Optimal => "optimal",
                LpStatus::Infeasible => "infeasible",
                LpStatus::Unbounded => "unbounded",
                LpStatus::IterLimit => "iter_limit",
                LpStatus::Numerical => "numerical",
            }
            .to_string(),
        );
        x_flat[t * n..(t + 1) * n].copy_from_slice(&sol.x[..n.min(sol.x.len())]);
        objs[t] = sol.obj;
    }
    let x_arr = PyArray1::from_vec(py, x_flat).reshape([k, n])?;
    Ok((statuses, x_arr, PyArray1::from_vec(py, objs)))
}

/// Solve a pure MILP `min cᵀx + obj_const s.t. A x = b, lb ≤ x ≤ ub`, with the
/// columns in `integer_cols` integer-constrained, by the Rust-internal
/// warm-started-simplex branch and bound. `a` is C-contiguous `m × n` standard
/// form (structural columns `[0, n_struct)`, slacks after). Returns
/// `(status, x[n_struct], obj, bound, nodes, lp_iters)` where status is one of
/// `optimal`, `feasible`, `infeasible`, `unbounded`, `node_limit`.
/// Adapter that lets an attached Python debugger drive the Rust MILP search.
///
/// Holds a GIL-independent handle (`Py<PyAny>`, which is `Send + Sync`) to a
/// Python callable. `checkpoint` re-acquires the GIL — the solve runs under
/// `py.allow_threads`, so the search thread does not hold it — builds a plain
/// state dict, calls the callable, and maps a truthy return to `Stop`. A hook
/// that raises is reported and treated as `Continue`, so a buggy debugger can
/// never corrupt the solve — EXCEPT `KeyboardInterrupt`, which stops the
/// search and is stashed in `pending` so `solve_milp_py` re-raises it after
/// the solve returns (Ctrl-C must abort the solve, not be swallowed).
struct PyMilpHook {
    callback: Py<PyAny>,
    pending: std::sync::Mutex<Option<PyErr>>,
}

impl MilpDebugHook for PyMilpHook {
    fn checkpoint(&self, state: &MilpDebugState<'_>) -> MilpDebugControl {
        Python::with_gil(|py| {
            let cp = match state.checkpoint {
                MilpCheckpoint::IterStart => "iter_start",
                MilpCheckpoint::AfterSelect => "after_select",
                MilpCheckpoint::AfterProcess => "after_process",
                MilpCheckpoint::IncumbentFound => "incumbent_found",
                MilpCheckpoint::Terminated => "terminated",
            };
            let d = PyDict::new(py);
            let _ = d.set_item("checkpoint", cp);
            let _ = d.set_item("iteration", state.iteration);
            let _ = d.set_item("nodes", state.total_nodes);
            let _ = d.set_item("open_nodes", state.open_nodes);
            let _ = d.set_item("incumbent", state.incumbent);
            let _ = d.set_item("bound", state.bound);
            let _ = d.set_item("gap", state.gap);
            let _ = d.set_item("elapsed", state.elapsed);
            // Node-box inspection at AfterSelect: marshal the batch's boxes/ids
            // as plain nested lists (best-effort; skipped on any conversion
            // error since inspection is non-critical).
            if let (Some(lbs), Some(ubs), Some(ids)) =
                (state.batch_lb, state.batch_ub, state.batch_ids)
            {
                let rows = |boxes: &[Vec<f64>]| -> Option<Bound<'_, PyList>> {
                    let r: Vec<Bound<'_, PyList>> = boxes
                        .iter()
                        .map(|row| PyList::new(py, row.iter().copied()).ok())
                        .collect::<Option<Vec<_>>>()?;
                    PyList::new(py, r).ok()
                };
                if let (Some(py_lb), Some(py_ub)) = (rows(lbs), rows(ubs)) {
                    let _ = d.set_item("batch_lb", py_lb);
                    let _ = d.set_item("batch_ub", py_ub);
                    let id_vec: Vec<usize> = ids.iter().map(|n| n.0).collect();
                    if let Ok(id_list) = PyList::new(py, id_vec) {
                        let _ = d.set_item("batch_ids", id_list);
                    }
                    let _ = d.set_item("n_vars", state.n_vars);
                }
            }
            match self.callback.bind(py).call1((d,)) {
                Ok(ret) => {
                    if ret.is_truthy().unwrap_or(false) {
                        MilpDebugControl::Stop
                    } else {
                        MilpDebugControl::Continue
                    }
                }
                Err(e) => {
                    if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                        *self.pending.lock().unwrap() = Some(e);
                        MilpDebugControl::Stop
                    } else {
                        e.print(py);
                        MilpDebugControl::Continue
                    }
                }
            }
        })
    }
}

/// Python-side lazy-constraint separator for the MILP driver (#1060).
///
/// This is what makes single-tree LP/NLP branch-and-bound possible without a
/// commercial MILP backend: the driver offers every integer-feasible point to
/// `callback` before it can become the incumbent, and folds the rows a veto
/// returns into the shared matrix.
///
/// **Callback contract.** `callback(x: np.ndarray) -> list[(coeffs, rhs)]`,
/// where `x` has length `n_struct` and each returned row means
/// `coeffs · x <= rhs` and must be **globally valid** — it is added to the
/// shared relaxation, not to one node. An empty list accepts the point.
///
/// The driver's internal cut form is `coeffs · x >= rhs`, so each row is
/// negated on the way in.
///
/// A callback that raises, or returns something that does not match the
/// contract, yields [`MilpLazyVerdict::Failed`]: the search stops uncertified
/// and the error is stashed in `pending` for `run_milp_hooked` to re-raise.
/// Constraints the caller asked for but that were never enforced must not be
/// silently dropped (CLAUDE.md §7) — unlike the debug hook, which is
/// inspection-only and can safely continue after an error.
struct PyMilpLazyHook {
    callback: Py<PyAny>,
    /// Structural width; a returned row may not be longer (a longer row would
    /// spill into the driver's slack columns and mean something else entirely).
    n_struct: usize,
    pending: std::sync::Mutex<Option<PyErr>>,
}

impl PyMilpLazyHook {
    /// Record the first failure only — later ones are consequences of it.
    fn fail(&self, e: PyErr) -> MilpLazyVerdict {
        let mut slot = self.pending.lock().unwrap();
        if slot.is_none() {
            *slot = Some(e);
        }
        MilpLazyVerdict::Failed
    }
}

impl MilpLazyHook for PyMilpLazyHook {
    fn separate(&self, x: &[f64]) -> MilpLazyVerdict {
        Python::with_gil(|py| {
            let arr = PyArray1::from_slice(py, x);
            let ret = match self.callback.bind(py).call1((arr,)) {
                Ok(r) => r,
                Err(e) => return self.fail(e),
            };
            // `list[(coeffs, rhs)]`; a numpy row extracts through the sequence
            // protocol, so both plain lists and arrays are accepted.
            let rows: Vec<(Vec<f64>, f64)> = match ret.extract() {
                Ok(r) => r,
                Err(e) => {
                    return self.fail(PyValueError::new_err(format!(
                        "lazy_callback must return a sequence of (coeffs, rhs) pairs \
                         meaning `coeffs @ x <= rhs`; got {ret:?} ({e})"
                    )))
                }
            };
            if rows.is_empty() {
                return MilpLazyVerdict::Accept;
            }
            let mut cuts = Vec::with_capacity(rows.len());
            for (coeffs, rhs) in rows {
                if coeffs.len() > self.n_struct {
                    return self.fail(PyValueError::new_err(format!(
                        "lazy_callback returned a row of length {} over {} structural \
                         columns; a longer row would address the driver's slack columns",
                        coeffs.len(),
                        self.n_struct
                    )));
                }
                if !rhs.is_finite() || coeffs.iter().any(|v| !v.is_finite()) {
                    return self.fail(PyValueError::new_err(
                        "lazy_callback returned a row with a non-finite entry; such a \
                         row cannot be added to the relaxation soundly",
                    ));
                }
                // `coeffs @ x <= rhs`  ==>  `(-coeffs) @ x >= -rhs`.
                cuts.push(GomoryCut {
                    coeffs: coeffs.iter().map(|v| -v).collect(),
                    rhs: -rhs,
                });
            }
            MilpLazyVerdict::Reject(cuts)
        })
    }
}

#[pyfunction]
#[pyo3(signature = (c, a, b, lb, ub, integer_cols, n_struct, obj_const=0.0,
                    max_nodes=1_000_000, gap_tol=1e-6, tol=1e-9, root_cuts=16,
                    cut_rounds=1, gmi_cuts=true, cut_select=false, node_cuts=false,
                    max_pool_cuts=128, heuristics=true, presolve=true, strong_branch=true,
                    node_propagation=false, reduced_cost_fixing=true,
                    sb_max_cands=6, sb_node_budget=48,
                    initial_incumbent=None, time_limit_s=None, debug_hook=None))]
pub fn solve_milp_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integer_cols: PyReadonlyArray1<'py, i64>,
    n_struct: usize,
    obj_const: f64,
    max_nodes: usize,
    gap_tol: f64,
    tol: f64,
    root_cuts: usize,
    cut_rounds: usize,
    gmi_cuts: bool,
    cut_select: bool,
    node_cuts: bool,
    max_pool_cuts: usize,
    heuristics: bool,
    presolve: bool,
    strong_branch: bool,
    node_propagation: bool,
    reduced_cost_fixing: bool,
    sb_max_cands: usize,
    sb_node_budget: usize,
    initial_incumbent: Option<PyReadonlyArray1<'py, f64>>,
    time_limit_s: Option<f64>,
    debug_hook: Option<Py<PyAny>>,
) -> PyResult<(String, Bound<'py, PyArray1<f64>>, f64, f64, usize, usize)> {
    let dims = a.shape();
    check_box_not_nan(lb.as_slice()?, ub.as_slice()?)?;
    let (m, n) = (dims[0], dims[1]);
    // Materialize owned copies of the borrowed numpy inputs. `PyReadonlyArray`
    // borrows are only valid while the GIL is held, so we copy before releasing
    // it; the copies (not the numpy buffers) feed the solve. The solve is long
    // and touches no Python objects, so it runs under `py.allow_threads` — this
    // unblocks the interpreter and lets the core's rayon workers (when built with
    // `parallel`) run without contending on the GIL.
    let a_owned: Vec<f64> = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?
        .to_vec();
    let c_owned: Vec<f64> = c.as_slice()?.to_vec();
    let b_owned: Vec<f64> = b.as_slice()?.to_vec();
    let l_owned: Vec<f64> = lb.as_slice()?.to_vec();
    let u_owned: Vec<f64> = ub.as_slice()?.to_vec();
    let int_cols: Vec<usize> = integer_cols
        .as_slice()?
        .iter()
        .map(|&v| v as usize)
        .collect();
    // Fully sparse driver (T3b5): build the working CSC from the dense numpy matrix
    // once and delegate. The CSC entry `solve_milp_csc_py` skips this densify.
    let csc = SparseCols::from_dense(&a_owned, m, n);
    let (status, x, obj, bound, nodes, lp_iters, _lazy_calls, _lazy_requeues) = run_milp_hooked(
        py,
        csc,
        m,
        n,
        c_owned,
        l_owned,
        u_owned,
        b_owned,
        int_cols,
        n_struct,
        obj_const,
        max_nodes,
        gap_tol,
        tol,
        root_cuts,
        cut_rounds,
        gmi_cuts,
        cut_select,
        node_cuts,
        max_pool_cuts,
        heuristics,
        presolve,
        strong_branch,
        node_propagation,
        reduced_cost_fixing,
        sb_max_cands,
        sb_node_budget,
        initial_incumbent,
        time_limit_s,
        debug_hook,
        // No lazy separator on this entry point: it keeps the historical
        // 6-tuple return and the driver's untouched path. `solve_milp_lazy_csc_py`
        // is the entry that takes one.
        None,
    )?;
    Ok((status, x, obj, bound, nodes, lp_iters))
}

/// CSC-input MILP entry (docs/dev/sparse-milp-plan.md T4): the constraint matrix is
/// passed column-major (`col_ptr`/`row_idx`/`vals`, `m` rows × `n` cols) so a large
/// sparse relaxation is NEVER densified — not on the Python side (no `A.toarray()`)
/// and not in Rust (the driver is fully sparse). Same options and return tuple as
/// [`solve_milp_py`].
#[pyfunction]
#[pyo3(signature = (c, m, n, col_ptr, row_idx, vals, b, lb, ub, integer_cols, n_struct,
                    obj_const=0.0, max_nodes=1_000_000, gap_tol=1e-6, tol=1e-9, root_cuts=16,
                    cut_rounds=1, gmi_cuts=true, cut_select=false, node_cuts=false,
                    max_pool_cuts=128, heuristics=true, presolve=true, strong_branch=true,
                    node_propagation=false, reduced_cost_fixing=true,
                    sb_max_cands=6, sb_node_budget=48,
                    initial_incumbent=None, time_limit_s=None, debug_hook=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_milp_csc_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    m: usize,
    n: usize,
    col_ptr: PyReadonlyArray1<'py, i64>,
    row_idx: PyReadonlyArray1<'py, i64>,
    vals: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integer_cols: PyReadonlyArray1<'py, i64>,
    n_struct: usize,
    obj_const: f64,
    max_nodes: usize,
    gap_tol: f64,
    tol: f64,
    root_cuts: usize,
    cut_rounds: usize,
    gmi_cuts: bool,
    cut_select: bool,
    node_cuts: bool,
    max_pool_cuts: usize,
    heuristics: bool,
    presolve: bool,
    strong_branch: bool,
    node_propagation: bool,
    reduced_cost_fixing: bool,
    sb_max_cands: usize,
    sb_node_budget: usize,
    initial_incumbent: Option<PyReadonlyArray1<'py, f64>>,
    time_limit_s: Option<f64>,
    debug_hook: Option<Py<PyAny>>,
) -> PyResult<(String, Bound<'py, PyArray1<f64>>, f64, f64, usize, usize)> {
    let col_ptr_v: Vec<usize> = col_ptr.as_slice()?.iter().map(|&x| x as usize).collect();
    check_box_not_nan(lb.as_slice()?, ub.as_slice()?)?;
    let row_idx_v: Vec<usize> = row_idx.as_slice()?.iter().map(|&x| x as usize).collect();
    let vals_v: Vec<f64> = vals.as_slice()?.to_vec();
    let c_owned: Vec<f64> = c.as_slice()?.to_vec();
    let b_owned: Vec<f64> = b.as_slice()?.to_vec();
    let l_owned: Vec<f64> = lb.as_slice()?.to_vec();
    let u_owned: Vec<f64> = ub.as_slice()?.to_vec();
    let int_cols: Vec<usize> = integer_cols
        .as_slice()?
        .iter()
        .map(|&v| v as usize)
        .collect();
    let csc = SparseCols::from_csc(col_ptr_v, row_idx_v, vals_v);
    let (status, x, obj, bound, nodes, lp_iters, _lazy_calls, _lazy_requeues) = run_milp_hooked(
        py,
        csc,
        m,
        n,
        c_owned,
        l_owned,
        u_owned,
        b_owned,
        int_cols,
        n_struct,
        obj_const,
        max_nodes,
        gap_tol,
        tol,
        root_cuts,
        cut_rounds,
        gmi_cuts,
        cut_select,
        node_cuts,
        max_pool_cuts,
        heuristics,
        presolve,
        strong_branch,
        node_propagation,
        reduced_cost_fixing,
        sb_max_cands,
        sb_node_budget,
        initial_incumbent,
        time_limit_s,
        debug_hook,
        // No lazy separator on this entry point: it keeps the historical
        // 6-tuple return and the driver's untouched path. `solve_milp_lazy_csc_py`
        // is the entry that takes one.
        None,
    )?;
    Ok((status, x, obj, bound, nodes, lp_iters))
}

/// CSC-input MILP entry **with a lazy-constraint separator** (#1060).
///
/// Identical to [`solve_milp_csc_py`] except that `lazy_callback` is called with
/// every integer-feasible point the search finds, before that point can become
/// the incumbent:
///
/// ```text
/// lazy_callback(x: np.ndarray) -> list[(coeffs, rhs)]      # coeffs @ x <= rhs
/// ```
///
/// An empty return accepts the point. A non-empty return **rejects** it: the
/// rows are added to the shared relaxation (they must be globally valid) and the
/// node is put back in the search — not fathomed, since a veto is not a proof
/// that the box is empty. This is what lets the Python OA layer run a
/// Quesada-Grossmann single tree on the in-house simplex instead of requiring a
/// commercial backend's lazy-constraint callbacks.
///
/// A callback that raises propagates out of this function, and the search stops
/// without a certificate rather than returning a result built against
/// constraints that were never enforced.
///
/// Returns [`solve_milp_csc_py`]'s tuple plus `(lazy_calls, lazy_requeues)` —
/// the anti-vacuity counters (CLAUDE.md §6). `lazy_calls == 0` means the
/// separator never saw a point, which is NOT the same as it having accepted
/// everything and must not be reported as convergence.
#[pyfunction]
#[pyo3(signature = (c, m, n, col_ptr, row_idx, vals, b, lb, ub, integer_cols, n_struct,
                    lazy_callback,
                    obj_const=0.0, max_nodes=1_000_000, gap_tol=1e-6, tol=1e-9, root_cuts=16,
                    cut_rounds=1, gmi_cuts=true, cut_select=false, node_cuts=false,
                    max_pool_cuts=128, heuristics=true, presolve=true, strong_branch=true,
                    node_propagation=false, reduced_cost_fixing=true,
                    sb_max_cands=6, sb_node_budget=48,
                    initial_incumbent=None, time_limit_s=None, debug_hook=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve_milp_lazy_csc_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    m: usize,
    n: usize,
    col_ptr: PyReadonlyArray1<'py, i64>,
    row_idx: PyReadonlyArray1<'py, i64>,
    vals: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integer_cols: PyReadonlyArray1<'py, i64>,
    n_struct: usize,
    lazy_callback: Py<PyAny>,
    obj_const: f64,
    max_nodes: usize,
    gap_tol: f64,
    tol: f64,
    root_cuts: usize,
    cut_rounds: usize,
    gmi_cuts: bool,
    cut_select: bool,
    node_cuts: bool,
    max_pool_cuts: usize,
    heuristics: bool,
    presolve: bool,
    strong_branch: bool,
    node_propagation: bool,
    reduced_cost_fixing: bool,
    sb_max_cands: usize,
    sb_node_budget: usize,
    initial_incumbent: Option<PyReadonlyArray1<'py, f64>>,
    time_limit_s: Option<f64>,
    debug_hook: Option<Py<PyAny>>,
) -> PyResult<(
    String,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    usize,
    usize,
    usize,
    usize,
)> {
    if !lazy_callback.bind(py).is_callable() {
        return Err(PyValueError::new_err("lazy_callback must be callable"));
    }
    let col_ptr_v: Vec<usize> = col_ptr.as_slice()?.iter().map(|&x| x as usize).collect();
    check_box_not_nan(lb.as_slice()?, ub.as_slice()?)?;
    let row_idx_v: Vec<usize> = row_idx.as_slice()?.iter().map(|&x| x as usize).collect();
    let vals_v: Vec<f64> = vals.as_slice()?.to_vec();
    let c_owned: Vec<f64> = c.as_slice()?.to_vec();
    let b_owned: Vec<f64> = b.as_slice()?.to_vec();
    let l_owned: Vec<f64> = lb.as_slice()?.to_vec();
    let u_owned: Vec<f64> = ub.as_slice()?.to_vec();
    let int_cols: Vec<usize> = integer_cols
        .as_slice()?
        .iter()
        .map(|&v| v as usize)
        .collect();
    let csc = SparseCols::from_csc(col_ptr_v, row_idx_v, vals_v);
    run_milp_hooked(
        py,
        csc,
        m,
        n,
        c_owned,
        l_owned,
        u_owned,
        b_owned,
        int_cols,
        n_struct,
        obj_const,
        max_nodes,
        gap_tol,
        tol,
        root_cuts,
        cut_rounds,
        gmi_cuts,
        cut_select,
        node_cuts,
        max_pool_cuts,
        heuristics,
        presolve,
        strong_branch,
        node_propagation,
        reduced_cost_fixing,
        sb_max_cands,
        sb_node_budget,
        initial_incumbent,
        time_limit_s,
        debug_hook,
        Some(lazy_callback),
    )
}

/// Shared MILP solve: builds `MilpOptions`, wraps an optional debug hook, runs the
/// fully sparse driver under `allow_threads`, and marshals the result back. Both the
/// dense entry (`solve_milp_py`) and the CSC entry (`solve_milp_csc_py`) funnel here
/// after producing the working `SparseCols`.
#[allow(clippy::too_many_arguments)]
fn run_milp_hooked<'py>(
    py: Python<'py>,
    csc: SparseCols,
    m: usize,
    n: usize,
    c_owned: Vec<f64>,
    l_owned: Vec<f64>,
    u_owned: Vec<f64>,
    b_owned: Vec<f64>,
    int_cols: Vec<usize>,
    n_struct: usize,
    obj_const: f64,
    max_nodes: usize,
    gap_tol: f64,
    tol: f64,
    root_cuts: usize,
    cut_rounds: usize,
    gmi_cuts: bool,
    cut_select: bool,
    node_cuts: bool,
    max_pool_cuts: usize,
    heuristics: bool,
    presolve: bool,
    strong_branch: bool,
    node_propagation: bool,
    reduced_cost_fixing: bool,
    sb_max_cands: usize,
    sb_node_budget: usize,
    initial_incumbent: Option<PyReadonlyArray1<'py, f64>>,
    time_limit_s: Option<f64>,
    debug_hook: Option<Py<PyAny>>,
    lazy_callback: Option<Py<PyAny>>,
) -> PyResult<(
    String,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    usize,
    usize,
    usize,
    usize,
)> {
    let opts = MilpOptions {
        n_struct,
        integer_cols: int_cols,
        max_nodes,
        time_limit_s: parse_budget_secs(time_limit_s)?,
        gap_tol,
        root_cuts,
        cut_rounds,
        gmi_cuts,
        cut_select,
        node_cuts,
        max_pool_cuts,
        heuristics,
        presolve,
        strong_branch,
        node_propagation,
        reduced_cost_fixing,
        sb_max_cands,
        sb_node_budget,
        initial_incumbent: initial_incumbent
            .map(|arr| arr.as_slice().map(|s| s.to_vec()))
            .transpose()?,
        simplex: SimplexOptions {
            tol,
            max_iter: 100_000,
            // The MILP driver clones this and injects its own wall-clock deadline
            // from `time_limit_s`, so the base options leave it unset.
            deadline: None,
            // F2: warm dual-simplex stall guard (size-derived cap → cold fallback).
            warm_stall_guard: true,
            warm_stall_cap_override: None,
            // P1.0 (measured OFF, deliberately): turning this on makes the primal
            // emit a FULL length-`m` basis of real columns instead of dropping the
            // slot a degenerate basic artificial occupies, which stops
            // `PreparedDual::prepare` rejecting the node basis on shape (measured
            // 619/620 rejections on the rsyn0840m OA master). But once the root cut
            // loop stopped cold-solving every round, flipping it was *exactly*
            // neutral on node count and wall — so per CLAUDE.md §5 it stays OFF
            // (sound but not net-positive is not a graduation).
            expel_zero_artificials: false,
            bank_deadline_duals: false,
            recover_unstable_pivot: false,
            // #1013: hand a STALLED warm dual re-solve to the cold solve (default from
            // `DISCOPT_LP_DUAL_STALL_BAIL`). Inert on the cold-only entry points.
            dual_stall_patience: SimplexOptions::default().dual_stall_patience,
            cold_dual_start: false,
        },
    };
    // Wrap an attached Python debugger (if any) as a core hook. It is created
    // before releasing the GIL and re-acquires it per checkpoint; when absent,
    // the search runs the untouched (bound-neutral) path.
    let hook = debug_hook.map(|cb| PyMilpHook {
        callback: cb,
        pending: std::sync::Mutex::new(None),
    });
    let hook_ref: Option<&dyn MilpDebugHook> = hook.as_ref().map(|h| h as &dyn MilpDebugHook);
    // Same shape for the lazy separator (#1060): absent means the driver runs
    // its untouched, bound-neutral path.
    let lazy = lazy_callback.map(|cb| PyMilpLazyHook {
        callback: cb,
        n_struct,
        pending: std::sync::Mutex::new(None),
    });
    let lazy_ref: Option<&dyn MilpLazyHook> = lazy.as_ref().map(|h| h as &dyn MilpLazyHook);
    let res = py.allow_threads(|| {
        let r = core_solve_milp_lazy_hooked(
            csc, m, n, &c_owned, &l_owned, &u_owned, &b_owned, obj_const, &opts, hook_ref, lazy_ref,
        );
        // Emit the per-phase / pivot profile to stderr when DISCOPT_PROFILE is set
        // (no-op otherwise). solve_milp has returned, so its function-scoped phase
        // timers have recorded. Engine perf work (issue #332) reads this.
        discopt_core::profile::dump();
        r
    });
    // Re-raise a KeyboardInterrupt captured by the debug hook: Ctrl-C during an
    // attached debug session aborts the solve (graceful search stop above) and
    // then propagates as the exception the caller expects, instead of being
    // silently converted into a normal-looking partial result.
    if let Some(h) = hook.as_ref() {
        if let Some(err) = h.pending.lock().unwrap().take() {
            return Err(err);
        }
    }
    // A separator that raised (or broke its contract) propagates: the search
    // already stopped uncertified, and returning its partial result as a normal
    // one would hide constraints that were never enforced (CLAUDE.md §7).
    if let Some(h) = lazy.as_ref() {
        if let Some(err) = h.pending.lock().unwrap().take() {
            return Err(err);
        }
    }
    let status = match res.status {
        MilpStatus::Optimal => "optimal",
        MilpStatus::Feasible => "feasible",
        MilpStatus::Infeasible => "infeasible",
        MilpStatus::Unbounded => "unbounded",
        MilpStatus::NodeLimit => "node_limit",
    };
    Ok((
        status.to_string(),
        PyArray1::from_vec(py, res.x),
        res.obj,
        res.bound,
        res.nodes,
        res.lp_iters,
        res.lazy_calls,
        res.lazy_requeues,
    ))
}
