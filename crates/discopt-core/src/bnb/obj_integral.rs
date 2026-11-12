//! Objective-integrality detection: the *granularity* of a MILP objective.
//!
//! When every variable carrying a nonzero objective coefficient is
//! integer-constrained and the coefficients share a rational scale, every
//! attainable objective value lies on a lattice `k·g + obj_const`. The spacing
//! `g` is the **granularity**, and it is worth a surprising amount:
//!
//! * A node whose lower bound `lb` satisfies `lb > U - g` (for incumbent `U`)
//!   cannot contain a *strictly better* solution — the next attainable value
//!   below `U` is `U - g`. Pruning on `U - g` rather than `U` is what lets a
//!   fractional dual bound close an integral gap. On `mk_binpack` the root LP
//!   bound is 7.72 against an optimum of 8 and **no cut moves it** (measured:
//!   SCIP applies 140 cuts and stays at 7.72); the entire gap is closed by this
//!   rule, and without it the search runs 551,849 nodes with the bound frozen.
//! * Symmetrically the reported dual bound may be rounded *up* onto the lattice.
//!
//! Prior art: SCIP computes the ceiling once, onto the primal cutoff
//! (`scip/src/scip/prob.c` `SCIPprobCheckObjIntegral`, applied at
//! `scip/src/scip/primal.c:428`), so a single value feeds node fathoming, the
//! leaf queue, the LP objective limit and conflict analysis. HiGHS gates on an
//! integral *scale* rather than integral coefficients
//! (`highs/mip/HighsObjectiveFunction.h` `isIntegral()`, detector
//! `highs/mip/HighsMipSolverData.cpp` `checkObjIntegrality`), which is strictly
//! more general — a `0.5`-spaced objective still qualifies. We follow HiGHS.
//!
//! # Soundness
//! The detector is the load-bearing part: a granularity that is too LARGE prunes
//! nodes that could hold the optimum, i.e. produces a false bound — the one
//! failure mode this whole file must not have. So it **refuses** (returns
//! `None`) on anything it cannot prove: a nonzero cost on a continuous column, a
//! coefficient that is not a clean rational at the working precision, or a
//! denominator beyond `MAX_DENOM`. Refusing costs a missed speedup; guessing
//! costs a wrong answer.

/// Largest denominator admitted when rationalizing a coefficient. Beyond this
/// the coefficient is treated as irrational and the objective as non-integral.
///
/// Deliberately small. The cap is not a performance guard, it is the soundness
/// guard: continued fractions will rationalize *anything* if allowed a big
/// enough denominator — π is 103993/33102 to well inside `RATIONAL_TOL` — and a
/// lattice inferred that way is an artifact of floating point, not a property of
/// the model. Capping at 1000 rejects π (the best convergent under the cap,
/// 355/113, is off by 2.7e-7) while still admitting every spacing a real
/// objective uses (halves, thirds, 0.01, 0.001). It also floors the granularity
/// at 1e-3, below which the cutoff `U - g` is indistinguishable from `U` and the
/// rule buys nothing anyway.
use crate::lp::simplex::sparse::SparseCols;
use crate::presolve::directed::{add_up, div_up, mul_up};

const MAX_DENOM: i64 = 1_000;

/// Relative tolerance for accepting `d·c` as an integer.
const RATIONAL_TOL: f64 = 1e-9;

/// Coefficients below this magnitude are treated as exactly zero (they carry no
/// objective contribution, so they impose no lattice constraint).
const ZERO_TOL: f64 = 1e-12;

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn lcm_i64(a: i64, b: i64) -> Option<i64> {
    if a == 0 || b == 0 {
        return None;
    }
    let g = gcd_i64(a, b);
    a.checked_div(g).and_then(|q| q.checked_mul(b))
}

/// The continued-fraction convergent `p/q` (`1 <= q <= MAX_DENOM`) nearest `v`
/// to `RATIONAL_TOL`, else `None`. Continued-fraction expansion, so a value like
/// `1/3` is found at `q = 3` rather than by scanning.
///
/// The convergent is a *candidate*, not a proof: `1 - 2^-32` is accepted as
/// `1/1` here (#1953). What makes the lattice sound is [`rational_error`], which
/// measures how far the stored coefficient really is from `p/q`, and the drift
/// [`spacing_of`] builds from it.
fn rationalize(v: f64) -> Option<(i64, i64)> {
    if !v.is_finite() {
        return None;
    }
    // Continued-fraction convergents p/q -> v.
    let (mut p0, mut q0, mut p1, mut q1) = (0i64, 1i64, 1i64, 0i64);
    let mut x = v;
    for _ in 0..64 {
        let a = x.floor();
        if !a.is_finite() || a.abs() > MAX_DENOM as f64 {
            return None;
        }
        let ai = a as i64;
        let p2 = ai.checked_mul(p1)?.checked_add(p0)?;
        let q2 = ai.checked_mul(q1)?.checked_add(q0)?;
        if q2.abs() > MAX_DENOM {
            return None;
        }
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        if q1 != 0 {
            let approx = p1 as f64 / q1 as f64;
            if (approx - v).abs() <= RATIONAL_TOL * v.abs().max(1.0) {
                // q > 0: only the first partial quotient can be negative, and q
                // is still 1 when it is consumed.
                return if q1 <= 0 { None } else { Some((p1, q1)) };
            }
        }
        let frac = x - a;
        if frac.abs() <= 1e-15 {
            return None; // exhausted without meeting the tolerance
        }
        x = 1.0 / frac;
    }
    None
}

/// An upper bound on `|v - p/q|`, exact zero when `v == p/q` exactly (#1953).
///
/// `v.mul_add(q, -p)` is the correctly rounded value of the exact `v*q - p`
/// (`p` and `q` are small integers, exact in binary64). That exact value is a
/// multiple of `ulp(v) >= 2^-1074`, so it rounds to zero only when it is zero,
/// and otherwise its magnitude is at most `|e|*(1 + 2^-52)`.
fn rational_error(v: f64, p: i64, q: i64) -> f64 {
    let e = v.mul_add(q as f64, -(p as f64));
    if e == 0.0 {
        return 0.0;
    }
    div_up(mul_up(e.abs(), 1.0 + f64::EPSILON), q as f64)
}

/// One objective term: coefficient `t` on a column whose values satisfy
/// `|x| <= mag` (`mag` may be `+inf`), plus `err`, an upper bound on how far `t`
/// itself is from the coefficient the model really induces (zero for a cost read
/// straight from `c`; the rounding of `-s*v` for a substituted term).
#[derive(Debug, Clone, Copy)]
struct Term {
    t: f64,
    mag: f64,
    err: f64,
}

/// `|x| <= mag` over `[lo, up]`.
fn magnitude(lo: f64, up: f64) -> f64 {
    lo.abs().max(up.abs())
}

/// `a * mag` rounded up, with `0 * inf = 0`: a term that is exactly zero
/// contributes nothing however wide its column.
fn drift_of(a: f64, mag: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else {
        mul_up(a, mag)
    }
}

/// The objective lattice, or `None` when it cannot be proven.
///
/// `c`, `is_int`, `lo` and `up` are indexed alike over the structural columns.
/// `obj_const` is deliberately **not** a parameter: it shifts the lattice but not
/// its spacing, and every use here is a *difference* of objective values.
pub fn objective_granularity(
    c: &[f64],
    is_int: &[bool],
    lo: &[f64],
    up: &[f64],
) -> Option<ObjLattice> {
    debug_assert_eq!(c.len(), is_int.len());
    if lo.len() < c.len() || up.len() < c.len() {
        return None;
    }
    let mut terms: Vec<Term> = Vec::new();
    for (j, &cj) in c.iter().enumerate() {
        let mag = magnitude(lo[j], up[j]);
        if cj.abs() <= ZERO_TOL {
            // Treated as zero, so its whole contribution is drift.
            terms.push(Term {
                t: cj,
                mag,
                err: 0.0,
            });
            continue;
        }
        // A nonzero cost on a continuous column destroys the lattice outright.
        if !is_int.get(j).copied().unwrap_or(false) {
            return None;
        }
        terms.push(Term {
            t: cj,
            mag,
            err: 0.0,
        });
    }
    spacing_of(&terms, 0.0).map(|(spacing, drift)| ObjLattice {
        spacing,
        shift: 0.0,
        drift,
        optimum_only: false,
    })
}

/// Largest admitted drift, relative to the spacing.
///
/// Not a soundness tolerance: every consumer accounts for the drift it is handed
/// (see `TreeManager::cutoff_value`, `round_bound_up` and the global bound). It
/// only declines lattices whose drift would eat the spacing they claim -- a
/// lattice with `drift` comparable to `spacing` prunes nothing it may prune.
const DRIFT_MAX_REL: f64 = 1e-9;

/// The spacing `g` of a lattice `{ k*g }` and a drift `D` such that every value of
/// `Sum_j t_j x_j` (over integer `x_j` within the terms' magnitudes) lies within
/// `D` of a lattice point, or `None` when it cannot be proven.
///
/// The drift is what makes this sound (#1953). Each coefficient is replaced by a
/// rational `r_j = p_j/q_j`; the lattice is exact for `Sum r_j x_j`, and the
/// stored objective differs from it by at most
/// `Sum_j (|t_j - r_j| + err_j) * mag_j`, summed with upward rounding, plus the
/// caller's `anchor_err` (the rounding of a computed shift). Coefficients below
/// `ZERO_TOL` are treated as zero and their whole `|t_j| * mag_j` is drift. Before
/// this, `1 - 2^-32` was read as `1` with no account of the `2^-32 * 2^31 = 0.5`
/// it moves the objective across a `[2^30, 2^31]` box, and a spacing of 1 pruned
/// the optimum. A drift above `DRIFT_MAX_REL * g` (an infinite one included)
/// declines.
///
/// Shared by both entry points so the rationalization and gcd -- the part whose
/// failure mode is a false bound -- exists once.
fn spacing_of(terms: &[Term], anchor_err: f64) -> Option<(f64, f64)> {
    let mut den: i64 = 1;
    let mut drift = anchor_err;
    let mut rats: Vec<(i64, i64)> = Vec::new();
    for &Term { t, mag, err } in terms {
        if !t.is_finite() || !err.is_finite() {
            return None;
        }
        if t.abs() <= ZERO_TOL {
            drift = add_up(drift, drift_of(add_up(t.abs(), err), mag));
            continue;
        }
        let (p, q) = rationalize(t)?;
        drift = add_up(drift, drift_of(add_up(rational_error(t, p, q), err), mag));
        den = lcm_i64(den, q)?;
        if den > MAX_DENOM {
            return None;
        }
        rats.push((p, q));
    }
    if rats.is_empty() {
        // A constant objective: every value is the same. Sound to call granular,
        // but there is nothing to prune, so decline and keep the downstream
        // arithmetic free of a degenerate case.
        return None;
    }
    // With `r_j = p_j/q_j` and `den` a multiple of every `q_j`, `den*r_j` is the
    // integer `p_j*(den/q_j)` -- exact integer arithmetic, no rounding test -- so
    // `Sum r_j x_j` is a multiple of `gcd_j(den*r_j) / den`.
    let mut g_num: i64 = 0;
    for &(p, q) in &rats {
        g_num = gcd_i64(g_num, p.checked_mul(den / q)?);
    }
    if g_num == 0 {
        return None;
    }
    let g = g_num as f64 / den as f64;
    if !g.is_finite() || g <= 0.0 {
        return None;
    }
    if !drift.is_finite() || drift > DRIFT_MAX_REL * g {
        return None;
    }
    Some((g, drift))
}

/// A proven objective lattice: every attainable objective value lies within
/// `drift` of `k*spacing + shift + obj_const` for some integer `k`.
///
/// `shift` is the constant the substitution below contributes; it is zero for a
/// directly-integral objective. It exists because the anchor is load-bearing:
/// rounding a dual bound onto a lattice whose offset is wrong lifts it past the
/// optimum, which is a false certificate (CLAUDE.md §1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjLattice {
    /// Lattice spacing `g > 0`.
    pub spacing: f64,
    /// Constant the substitution contributes; zero for a directly-integral
    /// objective. Added to the model's objective constant to anchor the lattice.
    pub shift: f64,
    /// Every attainable objective lies within `drift` of a lattice point, not on
    /// one: the stored coefficients are only within rounding of the rationals the
    /// lattice is built from (#1953). Zero when they are exactly those rationals.
    pub drift: f64,
    /// The lattice holds at *some* optimal solution, not at every feasible point:
    /// a term came from [`interval_lattice`] (a continuous column bounded by
    /// inequalities against lattice columns). Rounding a valid dual bound up stays
    /// sound (the optimal value is a lattice point), but an incumbent may sit off
    /// the lattice, so fathoming must cut below the largest lattice point under it
    /// rather than at `incumbent - spacing` (`TreeManager::cutoff_for`).
    pub optimum_only: bool,
}

/// True only for a bit-exactly fixed column.
///
/// No tolerance, deliberately. A column declared fixed when it is merely narrow
/// lets the objective vary by `c_j * width` off the claimed lattice, and the
/// bound rounding would then lift past the optimum. Fixed MPS columns and the
/// slacks of equality rows come out bit-identical, so exactness costs nothing.
fn is_fixed(lo: f64, up: f64) -> bool {
    lo.is_finite() && up.is_finite() && lo == up
}

/// The equality system a lattice derivation reads, in the column-major form the
/// driver already holds.
///
/// Grouped rather than passed as four parameters so that the row view travels as
/// one thing: `m` and `n` must agree with `csc`, and splitting them across an
/// argument list is how a caller ends up passing the structural count where the
/// total was meant.
pub struct LatticeRows<'a> {
    /// The constraint matrix `A`, column-major over all `n` columns.
    pub csc: &'a SparseCols,
    /// The right-hand side `b`, one entry per row.
    pub b: &'a [f64],
    /// Row count.
    pub m: usize,
    /// Total column count: structural columns followed by slacks.
    pub n: usize,
}

/// The objective lattice, resolving a costed *continuous* column through an
/// equality row that pins it to integer columns.
///
/// [`objective_granularity`] refuses the moment a continuous column carries
/// cost, and that pattern is common: a model written as `min z  s.t.  z = <integer
/// expression>` routes its whole objective through one continuous column. Measured
/// over a 104-instance MIPLIB draw, 20 instances present a directly-integral
/// objective and a further **13** are of exactly this shape -- the whole
/// `neos-361*` family, `misc07` (spacing 5), `markshare*`, `rout` (spacing 0.01).
/// All 13 were verified to have their true optimum on the derived lattice.
///
/// The model is **not** rewritten. This only reads the row to compute a spacing,
/// so there is no substitution to undo and no postsolve obligation -- the
/// distinction from the free-column-singleton substitution HiGHS performs in
/// presolve (`highs/src/presolve/HPresolve.cpp` `freeColSubstitution`), which
/// removes the column and must restore it. SCIP reaches the same lattice from the
/// other side, aggregating the variable away before `SCIPprobCheckObjIntegral`
/// runs (`scip/src/scip/prob.c`). Ours is the cheaper, narrower move: a detector,
/// not a transformation.
///
/// # Soundness
/// If row `i` is `Sum_k a_ik z_k = b_i` and column `j` has `a_ij != 0`, then at
/// *every* feasible point `z_j = (b_i - Sum_{k!=j} a_ik z_k) / a_ij`. So when every
/// other column of that row is integer-constrained or bit-exactly fixed, the term
/// `c_j z_j` is itself a lattice value. Other constraints only remove feasible
/// points, which cannot invalidate a claim quantified over all of them. Every
/// unproven case returns `None`.
///
/// Arguments are the engine form: `c`/`lo`/`up` span all `n` columns (structural
/// then slack), `is_int` only the `ns` structural ones, and `rows` is the equality
/// system `A z = b`.
pub fn objective_lattice(
    c: &[f64],
    is_int: &[bool],
    lo: &[f64],
    up: &[f64],
    rows: &LatticeRows<'_>,
) -> Option<ObjLattice> {
    objective_lattice_opts(c, is_int, lo, up, rows, None)
}

/// [`objective_lattice`], and with `interval = Some((lo, up))` a costed continuous
/// column no equality row pins may still resolve through [`interval_lattice`] over
/// those bounds; the result is then [`ObjLattice::optimum_only`].
///
/// The interval bounds are the model's own, not root-propagated ones: propagation
/// only adds bounds the rows already imply (the feasible set is unchanged, so the
/// interval argument holds over it), and it rounds them outward (`4 + 1ulp`), which
/// the exact-rational test would rightly refuse.
pub fn objective_lattice_opts(
    c: &[f64],
    is_int: &[bool],
    lo: &[f64],
    up: &[f64],
    rows: &LatticeRows<'_>,
    interval: Option<(&[f64], &[f64])>,
) -> Option<ObjLattice> {
    let (csc, b, m, n) = (rows.csc, rows.b, rows.m, rows.n);
    let mut optimum_only = false;
    let ns = is_int.len();
    if lo.len() < n || up.len() < n || c.len() < ns || b.len() < m {
        return None;
    }
    let integral = |k: usize| k < ns && is_int[k];

    // Rounding of the computed terms and shift (#1953). A product or quotient of
    // floats is within `u = 2^-53` relative of exact, so `fl(-fl(c/a)*v)` is within
    // `(2u + u^2)|t|` of `-c*v/a`; `REL` = `4u` covers that and the `1/(1-u)` of
    // relating the error to the computed `t`. The shift is a running float sum;
    // each step adds at most `u*|partial sum|` plus the error of its addend.
    const REL: f64 = 2.0 * f64::EPSILON;
    let mut terms: Vec<Term> = Vec::new();
    let mut shift = 0.0f64;
    let mut shift_err = 0.0f64;
    let add_shift = |shift: &mut f64, shift_err: &mut f64, v: f64, v_err: f64| {
        *shift += v;
        *shift_err = add_up(add_up(*shift_err, v_err), mul_up(shift.abs(), REL));
    };
    let mut pending: Vec<usize> = Vec::new();
    for j in 0..ns {
        let cj = c[j];
        let mag = magnitude(lo[j], up[j]);
        if cj.abs() <= ZERO_TOL {
            terms.push(Term {
                t: cj,
                mag,
                err: 0.0,
            });
            continue;
        }
        if integral(j) {
            terms.push(Term {
                t: cj,
                mag,
                err: 0.0,
            });
        } else if is_fixed(lo[j], up[j]) {
            let v = cj * lo[j];
            add_shift(&mut shift, &mut shift_err, v, mul_up(v.abs(), REL));
        } else {
            pending.push(j);
        }
    }
    // Nothing to resolve: identical to `objective_granularity`, and no row work.
    if pending.is_empty() {
        return spacing_of(&terms, shift_err).map(|(spacing, drift)| ObjLattice {
            spacing,
            shift,
            drift,
            optimum_only: false,
        });
    }

    // Row-wise view, built only over the rows the pending columns touch. One
    // O(nnz) pass at the root, and only for models that need it.
    let mut want = vec![interval.is_some(); m];
    for &j in &pending {
        for &i in csc.col(j).0 {
            if i < m {
                want[i] = true;
            }
        }
    }
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for k in 0..n {
        let (ri, va) = csc.col(k);
        for (t, &i) in ri.iter().enumerate() {
            if i < m && want[i] {
                rows[i].push((k, va[t]));
            }
        }
    }

    for &j in &pending {
        let mut resolved = false;
        for &i in csc.col(j).0 {
            if i >= m {
                continue;
            }
            let a_ij = rows[i]
                .iter()
                .find(|&&(k, _)| k == j)
                .map(|&(_, v)| v)
                .unwrap_or(0.0);
            if a_ij.abs() <= ZERO_TOL {
                continue;
            }
            // Usable only if every *other* column of the row is integral or
            // fixed. A second free continuous column -- another pending
            // objective column included -- makes the row prove nothing, so no
            // pair of pending columns can resolve through the same row and no
            // resolution can depend on another. A coefficient below `ZERO_TOL`
            // counts: its column still moves `z_j`.
            if rows[i]
                .iter()
                .any(|&(k, v)| k != j && v != 0.0 && !integral(k) && !is_fixed(lo[k], up[k]))
            {
                continue;
            }
            let s = c[j] / a_ij;
            if !s.is_finite() {
                continue;
            }
            let sb = s * b[i];
            add_shift(&mut shift, &mut shift_err, sb, mul_up(sb.abs(), REL));
            for &(k, v) in &rows[i] {
                if k == j || v == 0.0 {
                    continue;
                }
                let t = -s * v;
                if integral(k) {
                    terms.push(Term {
                        t,
                        mag: magnitude(lo[k], up[k]),
                        err: mul_up(t.abs(), REL),
                    });
                } else {
                    let w = t * lo[k];
                    add_shift(&mut shift, &mut shift_err, w, mul_up(w.abs(), 2.0 * REL));
                }
            }
            resolved = true;
            break;
        }
        if let (false, Some((ilo, iup))) = (resolved, interval) {
            if ilo.len() < n || iup.len() < n {
                return None;
            }
            let integral_dyn = |k: usize| integral(k);
            if let Some((lp, lq)) =
                interval_lattice(j, &rows, csc, m, b, c, ilo, iup, &integral_dyn)
            {
                // `c_j z_j = (c_j L) * (z_j / L)` with `z_j / L` an integer at the
                // optimum the argument picks.
                let lf = lp as f64 / lq as f64;
                let l_err = rational_error(lf, lp, lq);
                let t = c[j] * lf;
                let prod_err = if c[j].mul_add(lf, -t) == 0.0 {
                    0.0
                } else {
                    mul_up(t.abs(), REL)
                };
                let mag = magnitude(lo[j], up[j]).min(magnitude(ilo[j], iup[j]));
                terms.push(Term {
                    t,
                    mag: if mag.is_finite() {
                        mul_up(div_up(mag, lp as f64), lq as f64)
                    } else {
                        mag
                    },
                    err: add_up(mul_up(c[j].abs(), l_err), prod_err),
                });
                optimum_only = true;
                resolved = true;
            }
        }
        if !resolved {
            return None;
        }
    }
    if !shift.is_finite() {
        return None;
    }
    spacing_of(&terms, shift_err).map(|(spacing, drift)| ObjLattice {
        spacing,
        shift,
        drift,
        optimum_only,
    })
}

/// `v` as an exact rational `p/q` (`q <= MAX_DENOM`): only when the stored float is
/// that rational bit for bit (#1953's error is then zero, so no drift arises).
fn exact_rational(v: f64) -> Option<(i64, i64)> {
    let (p, q) = rationalize(v)?;
    (rational_error(v, p, q) == 0.0).then_some((p, q))
}

fn rat_reduce(p: i64, q: i64) -> Option<(i64, i64)> {
    if q == 0 {
        return None;
    }
    let g = gcd_i64(p, q).max(1);
    let (p, q) = (p / g, q / g);
    let (p, q) = if q < 0 {
        (p.checked_neg()?, -q)
    } else {
        (p, q)
    };
    (q <= MAX_DENOM).then_some((p, q))
}

/// `|a / b|` for exact rationals.
fn rat_abs_div(a: (i64, i64), b: (i64, i64)) -> Option<(i64, i64)> {
    if b.0 == 0 {
        return None;
    }
    rat_reduce(a.0.checked_mul(b.1)?.abs(), a.1.checked_mul(b.0)?.abs())
}

/// `|a * b|` for exact rationals.
fn rat_abs_mul(a: (i64, i64), b: (i64, i64)) -> Option<(i64, i64)> {
    rat_reduce(a.0.checked_mul(b.0)?.abs(), a.1.checked_mul(b.1)?)
}

/// The largest rational dividing both (`gcd(0, x) = x`).
fn rat_gcd(a: (i64, i64), b: (i64, i64)) -> Option<(i64, i64)> {
    if a.0 == 0 {
        return rat_reduce(b.0.abs(), b.1);
    }
    if b.0 == 0 {
        return rat_reduce(a.0.abs(), a.1);
    }
    let num = gcd_i64(a.0.checked_mul(b.1)?, b.0.checked_mul(a.1)?);
    rat_reduce(num, a.1.checked_mul(b.1)?)
}

/// A side or bound value: infinite (the `1e20` sentinel included) is no endpoint.
fn finite_side(v: f64) -> bool {
    v.is_finite() && v.abs() < 1e19
}

/// The lattice `L` a continuous column is *pinned* to at every feasible point: an
/// equality row (every other column integral or bit-exactly fixed, no free slack)
/// gives `z_k = (b - Sum a z) / a_k`, a multiple of the gcd of `|a/a_k|` over the
/// integer columns and `|(b - fixed part)/a_k|`. Exact rationals only.
fn pinned_lattice(
    k: usize,
    rows: &[Vec<(usize, f64)>],
    col_rows: &[usize],
    b: &[f64],
    lo: &[f64],
    up: &[f64],
    integral: &dyn Fn(usize) -> bool,
) -> Option<(i64, i64)> {
    'rows: for &i in col_rows {
        let Some(&(_, a_k)) = rows[i].iter().find(|&&(q, _)| q == k) else {
            continue;
        };
        let Some(ak) = exact_rational(a_k).filter(|r| r.0 != 0) else {
            continue;
        };
        let mut l = (0i64, 1i64);
        let mut rhs = exact_rational(b[i]);
        for &(q, v) in &rows[i] {
            if q == k || v == 0.0 {
                continue;
            }
            let Some(rv) = exact_rational(v) else {
                continue 'rows;
            };
            if integral(q) {
                let Some(g) = rat_abs_div(rv, ak).and_then(|t| rat_gcd(l, t)) else {
                    continue 'rows;
                };
                l = g;
            } else if is_fixed(lo[q], up[q]) {
                let Some(fv) = exact_rational(lo[q]) else {
                    continue 'rows;
                };
                let prod = v * lo[q];
                if v.mul_add(lo[q], -prod) != 0.0 {
                    continue 'rows;
                }
                let Some(pr) = exact_rational(prod) else {
                    continue 'rows;
                };
                let _ = fv;
                let Some(g) = rat_abs_div(pr, ak).and_then(|t| rat_gcd(l, t)) else {
                    continue 'rows;
                };
                l = g;
            } else {
                continue 'rows;
            }
        }
        let Some(rb) = rhs.take() else {
            continue;
        };
        let Some(g) = rat_abs_div(rb, ak).and_then(|t| rat_gcd(l, t)) else {
            continue;
        };
        if g.0 != 0 {
            return Some(g);
        }
    }
    None
}

/// The lattice `L` of a costed continuous column `j` that no equality row pins,
/// when every row it appears in bounds it against lattice values (b-ball's
/// `max x12 s.t. x12 <= x_k`, each `x_k` half a sum of binaries).
///
/// Each row of `j` is `a_j z_j + Sum a_k z_k + s t = b` with every other column
/// integral, bit-exactly fixed or [`pinned_lattice`], plus at most one slack `t`
/// that appears in no other row and carries no cost; `t`'s finite bounds are the
/// row's sides. With the integer columns fixed, `z_j` therefore ranges over an
/// interval whose finite ends are multiples of `L` (the gcd of every side, bound
/// and coefficient contribution, divided by `|a_j|`), and `z_j` appears in no other
/// row: moving it to the end its cost prefers (to a multiple of `L` inside the
/// interval when its cost is zero) keeps the point feasible and does not worsen the
/// objective. So *some* optimal solution has `z_j` on the lattice -- not every
/// feasible point, which is why the result is flagged `optimum_only`. Columns found
/// this way are never used as "lattice" neighbours of another (only integral,
/// fixed and pinned ones are), so two of them never share a row and each interval
/// argument stands alone. Exact rationals only; anything else refuses.
#[allow(clippy::too_many_arguments)]
fn interval_lattice(
    j: usize,
    rows: &[Vec<(usize, f64)>],
    csc: &SparseCols,
    m: usize,
    b: &[f64],
    c: &[f64],
    lo: &[f64],
    up: &[f64],
    integral: &dyn Fn(usize) -> bool,
) -> Option<(i64, i64)> {
    let mut l = (0i64, 1i64);
    for bound in [lo[j], up[j]] {
        if finite_side(bound) {
            l = rat_gcd(l, exact_rational(bound)?)?;
        }
    }
    let rows_of =
        |k: usize| -> Vec<usize> { csc.col(k).0.iter().copied().filter(|&i| i < m).collect() };
    for i in rows_of(j) {
        let a_j = rows[i].iter().find(|&&(q, _)| q == j)?.1;
        let aj = exact_rational(a_j).filter(|r| r.0 != 0)?;
        let mut slack: Option<(f64, usize)> = None;
        for &(k, v) in &rows[i] {
            if k == j || v == 0.0 {
                continue;
            }
            let rv = exact_rational(v)?;
            if integral(k) {
                l = rat_gcd(l, rat_abs_div(rv, aj)?)?;
            } else if is_fixed(lo[k], up[k]) {
                let prod = v * lo[k];
                if v.mul_add(lo[k], -prod) != 0.0 {
                    return None;
                }
                l = rat_gcd(l, rat_abs_div(exact_rational(prod)?, aj)?)?;
            } else if slack.is_none() && c.get(k).is_some_and(|&ck| ck == 0.0) && rows_of(k) == [i]
            {
                slack = Some((v, k));
            } else {
                let lk = pinned_lattice(k, rows, &rows_of(k), b, lo, up, integral)?;
                l = rat_gcd(l, rat_abs_mul(rat_abs_div(rv, aj)?, lk)?)?;
            }
        }
        // The row's sides: `b - s*t` at each finite bound of the slack, or `b` itself
        // for an equality row.
        let sides: Vec<f64> = match slack {
            None => vec![b[i]],
            Some((sv, k)) => {
                let mut v = Vec::new();
                for t in [lo[k], up[k]] {
                    if finite_side(t) {
                        let prod = sv * t;
                        if sv.mul_add(t, -prod) != 0.0 {
                            return None;
                        }
                        let side = b[i] - prod;
                        if (b[i] - side) != prod {
                            return None; // the subtraction rounded
                        }
                        v.push(side);
                    }
                }
                v
            }
        };
        for side in sides {
            l = rat_gcd(l, rat_abs_div(exact_rational(side)?, aj)?)?;
        }
    }
    (l.0 != 0).then_some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spacing over a `[-1000, 1000]` box per column: wide enough that a
    /// rounded rational like `1/3` still has a negligible drift.
    fn gran(c: &[f64], is_int: &[bool]) -> Option<f64> {
        let lo = vec![-1000.0; c.len()];
        let up = vec![1000.0; c.len()];
        objective_granularity(c, is_int, &lo, &up).map(|l| l.spacing)
    }

    /// Build the engine form `A z = b` from dense row-major `A`.
    fn csc(a: &[f64], m: usize, n: usize) -> SparseCols {
        SparseCols::from_dense(a, m, n)
    }

    /// The shape the lever exists for: `min z` with `z` continuous and pinned by
    /// `z - 2x - 4y = 0` over integers. The base detector must refuse; the
    /// substitution must find spacing 2.
    #[test]
    fn resolves_a_continuous_objective_column_through_its_defining_row() {
        // columns: z(0, continuous), x(1, int), y(2, int)
        let c = vec![1.0, 0.0, 0.0];
        let is_int = vec![false, true, true];
        let a = vec![1.0, -2.0, -4.0];
        let lo = vec![-100.0, 0.0, 0.0];
        let up = vec![100.0, 10.0, 10.0];
        let b = vec![0.0];

        assert_eq!(
            gran(&c, &is_int),
            None,
            "base detector must refuse a costed continuous column"
        );
        let got = objective_lattice(
            &c,
            &is_int,
            &lo,
            &up,
            &LatticeRows {
                csc: &csc(&a, 1, 3),
                b: &b,
                m: 1,
                n: 3,
            },
        )
        .expect("row pins z to an integer lattice");
        assert!((got.spacing - 2.0).abs() < 1e-12, "spacing {}", got.spacing);
        assert!((got.shift - 0.0).abs() < 1e-12, "shift {}", got.shift);
    }

    /// A nonzero right-hand side shifts the lattice but not its spacing, and the
    /// shift is what anchors the rounding. Getting it wrong lifts a dual bound
    /// past the optimum, so it is asserted on its own.
    #[test]
    fn a_nonzero_rhs_moves_the_anchor_not_the_spacing() {
        let c = vec![1.0, 0.0];
        let is_int = vec![false, true];
        // z - 3x = 7  =>  z = 7 + 3x, lattice {3k + 7}
        let a = vec![1.0, -3.0];
        let lo = vec![-100.0, 0.0];
        let up = vec![100.0, 10.0];
        let b = vec![7.0];
        let got = objective_lattice(
            &c,
            &is_int,
            &lo,
            &up,
            &LatticeRows {
                csc: &csc(&a, 1, 2),
                b: &b,
                m: 1,
                n: 2,
            },
        )
        .expect("pinned by the row");
        assert!((got.spacing - 3.0).abs() < 1e-12, "spacing {}", got.spacing);
        assert!((got.shift - 7.0).abs() < 1e-12, "shift {}", got.shift);
    }

    /// The refusal that keeps it sound: the defining row carries a *second* free
    /// continuous column, so it proves nothing about `z`.
    #[test]
    fn refuses_when_the_defining_row_holds_another_free_continuous_column() {
        // z - 2x - w = 0, with w continuous and free
        let c = vec![1.0, 0.0, 0.0];
        let is_int = vec![false, true, false];
        let a = vec![1.0, -2.0, -1.0];
        let lo = vec![-100.0, 0.0, 0.0];
        let up = vec![100.0, 10.0, 5.0];
        let b = vec![0.0];
        assert_eq!(
            objective_lattice(
                &c,
                &is_int,
                &lo,
                &up,
                &LatticeRows {
                    csc: &csc(&a, 1, 3),
                    b: &b,
                    m: 1,
                    n: 3
                }
            ),
            None
        );
    }

    /// A *fixed* continuous column in the row is a constant, so it is admissible
    /// -- and it lands in the shift, not the spacing.
    #[test]
    fn a_fixed_continuous_column_in_the_row_is_a_constant() {
        // z - 2x - w = 0 with w fixed at 1.5  =>  z = 2x + 1.5
        let c = vec![1.0, 0.0, 0.0];
        let is_int = vec![false, true, false];
        let a = vec![1.0, -2.0, -1.0];
        let lo = vec![-100.0, 0.0, 1.5];
        let up = vec![100.0, 10.0, 1.5];
        let b = vec![0.0];
        let got = objective_lattice(
            &c,
            &is_int,
            &lo,
            &up,
            &LatticeRows {
                csc: &csc(&a, 1, 3),
                b: &b,
                m: 1,
                n: 3,
            },
        )
        .expect("a fixed column is a constant, not a free variable");
        assert!((got.spacing - 2.0).abs() < 1e-12, "spacing {}", got.spacing);
        assert!((got.shift - 1.5).abs() < 1e-12, "shift {}", got.shift);
    }

    /// `is_fixed` takes no tolerance on purpose: a merely-narrow column lets the
    /// objective drift off the claimed lattice, which is a false bound.
    #[test]
    fn a_nearly_fixed_column_is_not_treated_as_fixed() {
        let c = vec![1.0, 0.0, 0.0];
        let is_int = vec![false, true, false];
        let a = vec![1.0, -2.0, -1.0];
        let lo = vec![-100.0, 0.0, 1.5];
        let up = vec![100.0, 10.0, 1.5 + 1e-9];
        let b = vec![0.0];
        assert_eq!(
            objective_lattice(
                &c,
                &is_int,
                &lo,
                &up,
                &LatticeRows {
                    csc: &csc(&a, 1, 3),
                    b: &b,
                    m: 1,
                    n: 3
                }
            ),
            None,
            "a 1e-9-wide column is not fixed"
        );
    }

    /// An inequality row cannot pin anything. In engine form that is a row whose
    /// slack column is free, so the slack itself is the disqualifying continuous
    /// column -- the same test, reached from the model's own encoding.
    #[test]
    fn an_inequality_row_does_not_pin_the_objective_column() {
        // z - 2x - s = 0 with s in [0, 10]: the row is `z - 2x >= 0`, not an equality.
        let c = vec![1.0, 0.0, 0.0];
        let is_int = vec![false, true];
        let a = vec![1.0, -2.0, -1.0];
        let lo = vec![-100.0, 0.0, 0.0];
        let up = vec![100.0, 10.0, 10.0];
        let b = vec![0.0];
        assert_eq!(
            objective_lattice(
                &c,
                &is_int,
                &lo,
                &up,
                &LatticeRows {
                    csc: &csc(&a, 1, 3),
                    b: &b,
                    m: 1,
                    n: 3
                }
            ),
            None
        );
    }

    /// With no continuous cost column the result must agree with the base
    /// detector exactly -- that agreement is what makes the OFF arm and the ON arm
    /// identical on the 91 % of models the substitution never touches.
    #[test]
    fn agrees_with_the_base_detector_when_nothing_needs_resolving() {
        let c = vec![4.0, 6.0];
        let is_int = vec![true, true];
        let a = vec![1.0, 1.0];
        let lo = vec![0.0, 0.0];
        let up = vec![10.0, 10.0];
        let b = vec![3.0];
        let base = gran(&c, &is_int).expect("4x + 6y");
        let got = objective_lattice(
            &c,
            &is_int,
            &lo,
            &up,
            &LatticeRows {
                csc: &csc(&a, 1, 2),
                b: &b,
                m: 1,
                n: 2,
            },
        )
        .expect("same lattice");
        assert!((got.spacing - base).abs() < 1e-12);
        assert!(got.shift == 0.0, "shift {}", got.shift);
    }

    /// The soundness property for the substituted path, stated as a test: for
    /// random integer points the *implied* objective must land on the derived
    /// lattice, anchored by the shift. A spacing too large or an anchor off the
    /// grid is a false bound, so this is the guard that matters.
    #[test]
    fn substituted_lattice_contains_every_implied_objective() {
        let mut state = 0xD1B54A32D192ED03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut checked = 0usize;
        for _ in 0..300 {
            let nx = 1 + (next() % 4) as usize; // integer columns
            let n = 1 + nx; // z at column 0
                            // z*a0 + sum a_k x_k = rhs, cost only on z
            let a0 = {
                let v = ((next() % 7) as i64) - 3;
                if v == 0 {
                    1
                } else {
                    v
                }
            } as f64;
            let ak: Vec<f64> = (0..nx)
                .map(|_| (((next() % 13) as i64) - 6) as f64)
                .collect();
            let rhs = (((next() % 21) as i64) - 10) as f64;
            let cz = {
                let v = ((next() % 5) as i64) - 2;
                if v == 0 {
                    1
                } else {
                    v
                }
            } as f64;

            let mut a = vec![a0];
            a.extend_from_slice(&ak);
            let mut c = vec![cz];
            c.extend(std::iter::repeat(0.0).take(nx));
            let mut is_int = vec![false];
            is_int.extend(std::iter::repeat(true).take(nx));
            let mut lo = vec![-1e6];
            lo.extend(std::iter::repeat(-20.0).take(nx));
            let mut up = vec![1e6];
            up.extend(std::iter::repeat(20.0).take(nx));
            let b = vec![rhs];

            let Some(l) = objective_lattice(
                &c,
                &is_int,
                &lo,
                &up,
                &LatticeRows {
                    csc: &csc(&a, 1, n),
                    b: &b,
                    m: 1,
                    n,
                },
            ) else {
                continue;
            };
            for _ in 0..15 {
                let x: Vec<f64> = (0..nx)
                    .map(|_| (((next() % 41) as i64) - 20) as f64)
                    .collect();
                // The row forces this value of z at every feasible point.
                let z = (rhs - ak.iter().zip(&x).map(|(a, x)| a * x).sum::<f64>()) / a0;
                let obj = cz * z;
                let k = (obj - l.shift) / l.spacing;
                assert!(
                    (k - k.round()).abs() < 1e-6,
                    "implied objective {obj} is off the lattice (spacing {}, shift {}, a0 {a0}, ak {ak:?}, rhs {rhs}, cz {cz})",
                    l.spacing,
                    l.shift
                );
                checked += 1;
            }
        }
        // CLAUDE.md §6: a probe that asserted nothing must fail, not pass.
        assert!(
            checked > 400,
            "substituted-lattice property never exercised ({checked})"
        );
    }

    #[test]
    fn unit_binary_objective_has_granularity_one() {
        let c = vec![1.0, 1.0, 1.0];
        let is_int = vec![true; 3];
        assert_eq!(gran(&c, &is_int), Some(1.0));
    }

    #[test]
    fn integer_coefficients_use_their_gcd() {
        // 4x + 6y over integers hits only multiples of 2.
        let c = vec![4.0, 6.0];
        let is_int = vec![true, true];
        assert_eq!(gran(&c, &is_int), Some(2.0));
    }

    #[test]
    fn half_integral_objective_is_detected() {
        let c = vec![0.5, 1.5];
        let is_int = vec![true, true];
        let g = gran(&c, &is_int).expect("0.5-spaced lattice");
        assert!((g - 0.5).abs() < 1e-12, "got {g}");
    }

    #[test]
    fn zero_cost_columns_are_ignored_even_when_continuous() {
        let c = vec![1.0, 0.0, 2.0];
        let is_int = vec![true, false, true];
        assert_eq!(gran(&c, &is_int), Some(1.0));
    }

    #[test]
    fn refuses_when_a_continuous_column_carries_cost() {
        let c = vec![1.0, 3.0];
        let is_int = vec![true, false];
        assert_eq!(gran(&c, &is_int), None);
    }

    #[test]
    fn refuses_an_irrational_coefficient() {
        // Left uncapped, continued fractions would hand back 1/33102 here.
        let c = vec![1.0, std::f64::consts::PI];
        let is_int = vec![true, true];
        assert_eq!(gran(&c, &is_int), None);
    }

    #[test]
    fn handles_negative_and_mixed_sign_coefficients() {
        let c = vec![-2.0, 4.0, -6.0];
        let is_int = vec![true; 3];
        assert_eq!(gran(&c, &is_int), Some(2.0));
    }

    #[test]
    fn refuses_a_denominator_past_the_cap() {
        let c = vec![1.0, 1.0 / 4001.0];
        let is_int = vec![true, true];
        assert_eq!(gran(&c, &is_int), None);
    }

    #[test]
    fn refuses_a_constant_objective() {
        let c = vec![0.0, 0.0];
        let is_int = vec![true, true];
        assert_eq!(gran(&c, &is_int), None);
    }

    /// The soundness property, stated as a test: for random integer points the
    /// objective must actually land on the detected lattice. A granularity that
    /// is too large would produce a false bound, so this is the guard that
    /// matters.
    #[test]
    fn detected_granularity_never_overstates_the_lattice() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut checked = 0usize;
        for _ in 0..400 {
            let n = 1 + (next() % 6) as usize;
            let c: Vec<f64> = (0..n)
                .map(|_| {
                    let num = (next() % 21) as i64 - 10;
                    let den = 1 + (next() % 4) as i64;
                    num as f64 / den as f64
                })
                .collect();
            let is_int = vec![true; n];
            let Some(g) = gran(&c, &is_int) else {
                continue;
            };
            for _ in 0..20 {
                let obj: f64 = c
                    .iter()
                    .map(|&cj| cj * (((next() % 41) as i64) - 20) as f64)
                    .sum();
                let k = obj / g;
                assert!(
                    (k - k.round()).abs() < 1e-6,
                    "objective {obj} is not a multiple of granularity {g} (c={c:?})"
                );
                checked += 1;
            }
        }
        // CLAUDE.md §6: a probe that asserted nothing must fail, not pass.
        assert!(
            checked > 500,
            "granularity property never exercised ({checked})"
        );
    }

    /// #1953 item 1: `1 - 2^-32` is exact in binary64 and is *not* 1. Over a
    /// `[2^30, 2^31]` box it moves the objective by up to 0.5, so a unit lattice
    /// would prune the optimum. Smaller boxes too (the issue's `N = 2^16`).
    #[test]
    fn issue_1953_near_integer_coefficient_is_not_a_unit_lattice() {
        let eps = 2f64.powi(-32);
        for n in [2f64.powi(30), 2f64.powi(16), 1e6] {
            let c = [1.0, -(1.0 - eps)];
            let lo = [n, n];
            let up = [2.0 * n, 2.0 * n];
            assert_eq!(
                objective_granularity(&c, &[true, true], &lo, &up),
                None,
                "N = {n}"
            );
        }
        // Unbounded columns: an inexact rational has unbounded drift.
        let inf = f64::INFINITY;
        assert_eq!(
            objective_granularity(&[1.0 / 3.0], &[true], &[0.0], &[inf]),
            None
        );
        // ... while an exact one needs no bound at all.
        assert_eq!(
            objective_granularity(&[0.5, 1.5], &[true, true], &[-inf, 0.0], &[inf, inf])
                .map(|l| (l.spacing, l.drift)),
            Some((0.5, 0.0))
        );
    }

    /// A decimal spacing the model means (`0.01`, stored inexactly) is still
    /// detected on a realistic box, with the drift reported rather than ignored,
    /// and the drift really bounds the distance of attainable values from the
    /// lattice.
    #[test]
    fn issue_1953_decimal_spacing_carries_its_drift() {
        let c = [0.01, 0.03];
        let lo = [0.0, -500.0];
        let up = [1e4, 500.0];
        let lat = objective_granularity(&c, &[true, true], &lo, &up).expect("0.01 lattice");
        assert_eq!(lat.spacing, 0.01);
        assert!(lat.drift > 0.0 && lat.drift < 1e-12, "drift {}", lat.drift);
        let mut checked = 0;
        for x in [0i64, 1, 7, 9999, 10_000] {
            for y in [-500i64, -3, 0, 499, 500] {
                let v = c[0] * x as f64 + c[1] * y as f64;
                let exact = (x + 3 * y) as f64 / 100.0; // the lattice point
                                                        // f64 evaluation adds its own rounding; allow it on top of drift.
                assert!((v - exact).abs() <= lat.drift + 4.0 * f64::EPSILON * 1e2);
                checked += 1;
            }
        }
        assert_eq!(checked, 25);
    }

    /// A coefficient below `ZERO_TOL` is not free: over a wide column its whole
    /// contribution is drift, and a wide enough column declines the lattice.
    #[test]
    fn issue_1953_dropped_tiny_coefficient_counts_as_drift() {
        let c = [1.0, 1e-13];
        let is_int = [true, false];
        assert!(objective_granularity(&c, &is_int, &[0.0, 0.0], &[10.0, 1.0]).is_some());
        assert_eq!(
            objective_granularity(&c, &is_int, &[0.0, 0.0], &[10.0, 1e6]),
            None
        );
    }

    /// The substitution path accounts for the same drift: `z = (1 - 2^-32) y`
    /// with `y` integer on `[2^30, 2^31]` is not a unit lattice.
    #[test]
    fn issue_1953_substituted_near_integer_term_is_declined() {
        let eps = 2f64.powi(-32);
        let n = 2f64.powi(30);
        // columns: z (continuous, cost 1), y (int); row z - (1 - eps) y = 0
        let a = vec![1.0, -(1.0 - eps)];
        let got = objective_lattice(
            &[1.0, 0.0],
            &[false, true],
            &[-1e12, n],
            &[1e12, 2.0 * n],
            &LatticeRows {
                csc: &csc(&a, 1, 2),
                b: &[0.0],
                m: 1,
                n: 2,
            },
        );
        assert_eq!(got, None);
    }

    /// Engine form `A z = b` for the interval tests: structural columns first, then
    /// one slack per inequality row (`row - slack = rhs` with the slack in
    /// `[0, 1e20]`, i.e. `row >= rhs`; a `<=` row is written negated).
    struct EngineLp {
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        lo: Vec<f64>,
        up: Vec<f64>,
        c: Vec<f64>,
        is_int: Vec<bool>,
    }

    fn engine(
        ns_lo: &[f64],
        ns_up: &[f64],
        c: &[f64],
        is_int: &[bool],
        eqs: &[(Vec<f64>, f64)],
        ges: &[(Vec<f64>, f64)],
    ) -> EngineLp {
        let ns = ns_lo.len();
        let n = ns + ges.len();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (row, rhs) in eqs {
            let mut r = row.clone();
            r.resize(n, 0.0);
            a.push(r);
            b.push(*rhs);
        }
        for (k, (row, rhs)) in ges.iter().enumerate() {
            let mut r = row.clone();
            r.resize(n, 0.0);
            r[ns + k] = -1.0;
            a.push(r);
            b.push(*rhs);
        }
        let mut lo = ns_lo.to_vec();
        let mut up = ns_up.to_vec();
        let mut cc = c.to_vec();
        for _ in ges {
            lo.push(0.0);
            up.push(1e20);
            cc.push(0.0);
        }
        EngineLp {
            a,
            b,
            lo,
            up,
            c: cc,
            is_int: is_int.to_vec(),
        }
    }

    fn lattice_of(e: &EngineLp, interval: bool) -> Option<ObjLattice> {
        let (m, n) = (e.a.len(), e.lo.len());
        let dense: Vec<f64> = e.a.iter().flatten().copied().collect();
        objective_lattice_opts(
            &e.c,
            &e.is_int,
            &e.lo,
            &e.up,
            &LatticeRows {
                csc: &csc(&dense, m, n),
                b: &e.b,
                m,
                n,
            },
            interval.then_some((&e.lo[..], &e.up[..])),
        )
    }

    /// b-ball's shape: `max z s.t. z <= x_k`, `2 x_k = sum of binaries`. Each `x_k`
    /// is pinned to the half-integers and `z` lies between them, so the optimum is a
    /// half-integer: spacing 1/2, flagged `optimum_only`. The equality-only
    /// detector refuses (no row pins `z`).
    #[test]
    fn interval_rule_finds_b_balls_half_lattice() {
        // Columns: b0..b3 (binary), x0, x1, z.
        let lo = [0.0, 0.0, 0.0, 0.0, -1e20, -1e20, 0.0];
        let up = [1.0, 1.0, 1.0, 1.0, 1e20, 1e20, 1e20];
        let c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let is_int = [true, true, true, true, false, false, false];
        let eqs = vec![
            (vec![-1.0, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0], 0.0),
            (vec![0.0, 0.0, -1.0, -1.0, 0.0, 2.0, 0.0], 0.0),
        ];
        // x_k - z >= 0
        let ges = vec![
            (vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0], 0.0),
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0], 0.0),
        ];
        let e = engine(&lo, &up, &c, &is_int, &eqs, &ges);
        assert_eq!(lattice_of(&e, false), None);
        let l = lattice_of(&e, true).expect("interval rule resolves z");
        assert_eq!(l.spacing, 0.5);
        assert_eq!(l.shift, 0.0);
        assert_eq!(l.drift, 0.0);
        assert!(l.optimum_only);

        // Refusals. A second free continuous column in z's row proves nothing.
        let mut lo2 = lo.to_vec();
        let mut up2 = up.to_vec();
        let mut c2 = c.to_vec();
        let mut int2 = is_int.to_vec();
        lo2.push(-1e20);
        up2.push(1e20);
        c2.push(0.0);
        int2.push(false);
        let mut ges2 = ges.clone();
        ges2[0].0.push(1.0);
        let eqs2: Vec<(Vec<f64>, f64)> = eqs
            .iter()
            .map(|(r, b)| {
                let mut r = r.clone();
                r.push(0.0);
                (r, *b)
            })
            .collect();
        let mut ges2b = ges2.clone();
        ges2b[1].0.push(0.0);
        let e2 = engine(&lo2, &up2, &c2, &int2, &eqs2, &ges2b);
        assert_eq!(lattice_of(&e2, true), None, "free neighbour must refuse");
        // An inexact coefficient (0.1 is no binary rational) refuses.
        let mut ges3 = ges.clone();
        ges3[0].0[6] = -0.1;
        let e3 = engine(&lo, &up, &c, &is_int, &eqs, &ges3);
        assert_eq!(
            lattice_of(&e3, true),
            None,
            "inexact coefficient must refuse"
        );
        // A bound off the lattice (z <= 0.3) shrinks the spacing to a common divisor
        // or refuses; it must never keep 1/2.
        let mut up4 = up;
        up4[6] = 0.3;
        let e4 = engine(&lo, &up4, &c, &is_int, &eqs, &ges);
        assert!(lattice_of(&e4, true)
            .is_none_or(|l| (0.3f64 / l.spacing - (0.3f64 / l.spacing).round()).abs() < 1e-9));
    }

    /// Exact oracle: random problems of the interval shape -- binaries, continuous
    /// columns pinned by equalities, and a costed continuous `z` bounded by
    /// inequalities against them. The true optimum (binaries enumerated, `z`'s
    /// interval solved exactly) must lie on every lattice the detector returns, and
    /// with the lattice's `optimum_only` cutoff no incumbent value above the optimum
    /// -- on the lattice or off it -- may fathom a node bounded at the optimum.
    #[test]
    fn interval_lattice_holds_at_the_true_optimum() {
        use crate::bnb::branching::VarBranchInfo;
        use crate::bnb::pool::SelectionStrategy;
        use crate::bnb::tree_manager::TreeManager;
        let mut seed = 99u64;
        let mut rnd = |lo: i64, hi: i64| -> i64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((seed >> 33) as i64).rem_euclid(hi - lo + 1)
        };
        let (mut cases, mut found, mut checked, mut cut_checks) = (0, 0, 0, 0);
        let mut plain_violations = 0usize;
        for _ in 0..3000 {
            let groups = rnd(1, 2) as usize;
            let per = rnd(1, 3) as usize;
            let nb = groups * per;
            let ns = nb + groups + 1;
            let z = ns - 1;
            let mut lo = vec![0.0; nb];
            let mut up = vec![1.0; nb];
            lo.extend(std::iter::repeat_n(-1e20, groups));
            up.extend(std::iter::repeat_n(1e20, groups));
            let zl = [0.0, -3.0][rnd(0, 1) as usize];
            let zu = [1e20, 7.0, 6.5][rnd(0, 2) as usize];
            lo.push(zl);
            up.push(zu);
            let mut c: Vec<f64> = (0..nb).map(|_| rnd(-2, 2) as f64).collect();
            c.extend(std::iter::repeat_n(0.0, groups));
            let cz = [-1.0, 1.0, -2.0, 0.5, -0.5][rnd(0, 4) as usize];
            c.push(cz);
            let mut is_int = vec![true; nb];
            is_int.extend(std::iter::repeat_n(false, groups + 1));
            // Pins: d_k x_k - sum a_i b_i = r_k.
            let mut pins = Vec::new();
            let mut eqs = Vec::new();
            for g in 0..groups {
                let d = rnd(1, 4) as f64;
                let r = [-1.0, 0.0, 1.0, 2.0, 0.5][rnd(0, 4) as usize];
                let coefs: Vec<f64> = (0..per).map(|_| rnd(1, 3) as f64).collect();
                let mut row = vec![0.0; ns];
                for (t, &a) in coefs.iter().enumerate() {
                    row[g * per + t] = -a;
                }
                row[nb + g] = d;
                eqs.push((row, r));
                pins.push((d, r, coefs));
            }
            // z rows: e*x_g - z >= -s (z <= s + e x_g)  or  z + e*x_g >= s.
            let mut ges = Vec::new();
            let mut zrows = Vec::new();
            for g in 0..groups {
                let e = rnd(1, 2) as f64;
                let s = [0.0, 1.0, 0.5, -1.0][rnd(0, 3) as usize];
                let upper = rnd(0, 1) == 0;
                let mut row = vec![0.0; ns];
                if upper {
                    row[nb + g] = e;
                    row[z] = -1.0;
                    ges.push((row, -s));
                } else {
                    row[nb + g] = e;
                    row[z] = 1.0;
                    ges.push((row, s));
                }
                zrows.push((g, e, s, upper));
            }
            let lp = engine(&lo, &up, &c, &is_int, &eqs, &ges);
            cases += 1;
            // Oracle.
            let mut opt = f64::INFINITY;
            let mut attained: Vec<f64> = Vec::new();
            let mut unbounded = false;
            for mask in 0..(1u32 << nb) {
                let bins: Vec<f64> = (0..nb).map(|i| ((mask >> i) & 1) as f64).collect();
                let xs: Vec<f64> = (0..groups)
                    .map(|g| {
                        let (d, r, ref co) = pins[g];
                        (r + co
                            .iter()
                            .enumerate()
                            .map(|(t, a)| a * bins[g * per + t])
                            .sum::<f64>())
                            / d
                    })
                    .collect();
                let (mut zlo, mut zhi) = (zl, if zu >= 1e19 { f64::INFINITY } else { zu });
                for &(g, e, s, upper) in &zrows {
                    if upper {
                        zhi = zhi.min(s + e * xs[g]);
                    } else {
                        zlo = zlo.max(s - e * xs[g]);
                    }
                }
                if zlo > zhi + 1e-12 {
                    continue;
                }
                let zstar = if cz > 0.0 { zlo } else { zhi };
                if !zstar.is_finite() {
                    unbounded = true;
                    break;
                }
                let base: f64 = (0..nb).map(|i| c[i] * bins[i]).sum();
                let v = base + cz * zstar;
                opt = opt.min(v);
                // Also an off-interval-end point (midpoint) as a possible incumbent.
                attained.push(v);
                if zhi.is_finite() {
                    attained.push(base + cz * 0.5 * (zlo + zhi));
                }
            }
            if unbounded || !opt.is_finite() {
                continue;
            }
            let Some(l) = lattice_of(&lp, true) else {
                continue;
            };
            found += 1;
            let k = (opt - l.shift) / l.spacing;
            assert!(
                (k - k.round()).abs() <= 1e-9 + l.drift / l.spacing,
                "optimum {opt} off the lattice {l:?}"
            );
            checked += 1;
            // Cutoff: no incumbent above the optimum may fathom a node bounded at it.
            let mut tm = TreeManager::new(
                1,
                vec![0.0],
                vec![1.0],
                vec![VarBranchInfo {
                    offset: 0,
                    size: 1,
                    is_integer: true,
                }],
                SelectionStrategy::BestFirst,
            );
            tm.set_objective_lattice(Some(l.spacing), l.shift, l.drift);
            tm.set_objective_lattice_optimum_only(l.optimum_only);
            let mut plain = tm.clone_lattice_probe();
            plain.set_objective_lattice_optimum_only(false);
            for &u in &attained {
                if u > opt + 1e-9 && plain.cutoff_for(u) < opt - 1e-9 {
                    plain_violations += 1;
                }
                if u > opt + 1e-9 {
                    assert!(
                        tm.cutoff_for(u) >= opt - 1e-9,
                        "incumbent {u} cuts off the optimum {opt} ({l:?})"
                    );
                    cut_checks += 1;
                }
            }
        }
        eprintln!("interval lattice: cases {cases} found {found} checked {checked} cutoff checks {cut_checks} plain-rule violations {plain_violations}");
        // The off-lattice cutoff is load-bearing: `incumbent - spacing` alone would
        // have fathomed the optimum on these.
        assert!(
            plain_violations > 0,
            "no case exercises the off-lattice incumbent"
        );
        assert!(
            found > 200 && checked == found && cut_checks > 500,
            "probe fired too little: found {found} checked {checked} cutoff checks {cut_checks}"
        );
    }
}
