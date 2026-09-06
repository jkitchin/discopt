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

const MAX_DENOM: i64 = 1_000;

/// Relative tolerance for accepting `d·c` as an integer.
const RATIONAL_TOL: f64 = 1e-9;

/// Coefficients below this magnitude are treated as exactly zero (they carry no
/// objective contribution, so they impose no lattice constraint).
const ZERO_TOL: f64 = 1e-12;

/// The engine's "unbounded" sentinel. `f64::is_finite` is TRUE at this value, so
/// every bound test in this module must compare against it rather than calling
/// `is_finite` (CLAUDE.md's `INF` note; #1189 review, finding 2).
const INF: f64 = 1e20;

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

/// Smallest `d in 1..=MAX_DENOM` with `d·v` integral to `RATIONAL_TOL`, else
/// `None`. Continued-fraction expansion, so a value like `1/3` is found at
/// `d = 3` rather than by scanning.
fn denominator_of(v: f64) -> Option<i64> {
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
                let q = q1.abs();
                return if q == 0 { None } else { Some(q) };
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

/// The objective granularity `g > 0` when every attainable objective value lies
/// on the lattice `k·g + obj_const`, or `None` when that cannot be proven.
///
/// `c` and `is_int` are indexed alike over the structural columns. `obj_const`
/// is deliberately **not** a parameter: it shifts the lattice but not its
/// spacing, and every use here is a *difference* of objective values.
pub fn objective_granularity(c: &[f64], is_int: &[bool]) -> Option<f64> {
    debug_assert_eq!(c.len(), is_int.len());
    let mut terms: Vec<f64> = Vec::new();
    for (j, &cj) in c.iter().enumerate() {
        if cj.abs() <= ZERO_TOL {
            continue;
        }
        // A nonzero cost on a continuous column destroys the lattice outright.
        if !is_int.get(j).copied().unwrap_or(false) {
            return None;
        }
        terms.push(cj);
    }
    spacing_of(&terms)
}

/// The spacing of the lattice `{ k*g : k integer }` containing every value of
/// `Sum_j t_j x_j` over integer `x_j`, or `None` when it cannot be proven.
///
/// Shared by both entry points so the rationalization and gcd — the part whose
/// failure mode is a false bound — exists once.
fn spacing_of(terms: &[f64]) -> Option<f64> {
    let mut den: i64 = 1;
    let mut any = false;
    for &t in terms {
        if t.abs() <= ZERO_TOL {
            continue;
        }
        any = true;
        den = lcm_i64(den, denominator_of(t)?)?;
        if den > MAX_DENOM {
            return None;
        }
    }
    if !any {
        // A constant objective: every value is the same. Sound to call granular,
        // but there is nothing to prune, so decline and keep the downstream
        // arithmetic free of a degenerate case.
        return None;
    }
    // With `d*t_j` all integral, the sum is `(1/d)*Sum (d*t_j) x_j` over integer
    // `x_j`, hence a multiple of `gcd_j(d*t_j) / d`.
    let mut g_num: i64 = 0;
    for &t in terms {
        if t.abs() <= ZERO_TOL {
            continue;
        }
        let scaled = t * den as f64;
        let r = scaled.round();
        if (scaled - r).abs() > RATIONAL_TOL * scaled.abs().max(1.0) {
            return None;
        }
        if r.abs() > i64::MAX as f64 {
            return None;
        }
        g_num = gcd_i64(g_num, r as i64);
    }
    if g_num == 0 {
        return None;
    }
    let g = g_num as f64 / den as f64;
    if !g.is_finite() || g <= 0.0 {
        return None;
    }
    Some(g)
}

/// A proven objective lattice: every attainable objective value equals
/// `k*spacing + shift + obj_const` for some integer `k`.
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
}

/// True only for a bit-exactly fixed column.
///
/// No tolerance, deliberately. A column declared fixed when it is merely narrow
/// lets the objective vary by `c_j * width` off the claimed lattice, and the
/// bound rounding would then lift past the optimum. Fixed MPS columns and the
/// slacks of equality rows come out bit-identical, so exactness costs nothing.
fn is_fixed(lo: f64, up: f64) -> bool {
    // NOT `is_finite`: `1e20_f64.is_finite()` is true, so a column pinned at the
    // engine's INF sentinel used to read as fixed and anchor `shift` at 1e20.
    // `round_bound_up` then computes `(v - 1e20)/g`, and the cancellation on the
    // way back to objective units makes the lift numeric garbage that can land
    // ABOVE `v` and be published as a dual bound (#1189 review, finding 2).
    lo > -INF && up < INF && lo == up
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
    let (csc, b, m, n) = (rows.csc, rows.b, rows.m, rows.n);
    let ns = is_int.len();
    if lo.len() < n || up.len() < n || c.len() < ns || b.len() < m {
        return None;
    }
    let integral = |k: usize| k < ns && is_int[k];

    let mut terms: Vec<f64> = Vec::new();
    let mut shift = 0.0f64;
    let mut pending: Vec<usize> = Vec::new();
    for j in 0..ns {
        let cj = c[j];
        if cj.abs() <= ZERO_TOL {
            continue;
        }
        if integral(j) {
            terms.push(cj);
        } else if is_fixed(lo[j], up[j]) {
            shift += cj * lo[j];
        } else {
            pending.push(j);
        }
    }
    // Nothing to resolve: identical to `objective_granularity`, and no row work.
    if pending.is_empty() {
        return spacing_of(&terms).map(|spacing| ObjLattice { spacing, shift });
    }

    // Row-wise view, built only over the rows the pending columns touch. One
    // O(nnz) pass at the root, and only for models that need it.
    let mut want = vec![false; m];
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
            // resolution can depend on another.
            //
            // The magnitude test is `v != 0.0`, NOT `|v| > ZERO_TOL`. A free
            // continuous column contributes `-(v / a_ij) * z_k` to `z_j` at every
            // feasible point, and with `INF = 1e20` that is enormous however small
            // `v` is: at `v = 1e-13` the objective still moves by ~1e7. Judging the
            // coefficient instead of the coefficient TIMES THE COLUMN'S WIDTH is
            // exactly the gear4 mistake CLAUDE.md records -- "test unboundedness on
            // the bound, never on a product" -- and it published a lattice the
            // optimum does not sit on, i.e. a false certificate (#1189 review,
            // finding 1). An exact structural zero contributes nothing at any
            // width, so it stays skippable.
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
            shift += s * b[i];
            for &(k, v) in &rows[i] {
                // Exact zeros only. Dropping a small-but-nonzero coefficient here
                // would remove a term from the gcd and hand back a COARSER lattice
                // than the truth -- unsound in the same direction as finding 1. A
                // tiny term instead flows into `spacing_of`, where `denominator_of`
                // refuses it; refusal is the safe outcome.
                if k == j || v == 0.0 {
                    continue;
                }
                if integral(k) {
                    terms.push(-s * v);
                } else {
                    shift -= s * v * lo[k];
                }
            }
            resolved = true;
            break;
        }
        if !resolved {
            return None;
        }
    }
    if !shift.is_finite() {
        return None;
    }
    spacing_of(&terms).map(|spacing| ObjLattice { spacing, shift })
}

#[cfg(test)]
mod tests {
    use super::*;

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
            objective_granularity(&c, &is_int),
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
        let base = objective_granularity(&c, &is_int).expect("4x + 6y");
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
                    n: n,
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
        assert_eq!(objective_granularity(&c, &is_int), Some(1.0));
    }

    #[test]
    fn integer_coefficients_use_their_gcd() {
        // 4x + 6y over integers hits only multiples of 2.
        let c = vec![4.0, 6.0];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), Some(2.0));
    }

    #[test]
    fn half_integral_objective_is_detected() {
        let c = vec![0.5, 1.5];
        let is_int = vec![true, true];
        let g = objective_granularity(&c, &is_int).expect("0.5-spaced lattice");
        assert!((g - 0.5).abs() < 1e-12, "got {g}");
    }

    #[test]
    fn zero_cost_columns_are_ignored_even_when_continuous() {
        let c = vec![1.0, 0.0, 2.0];
        let is_int = vec![true, false, true];
        assert_eq!(objective_granularity(&c, &is_int), Some(1.0));
    }

    #[test]
    fn refuses_when_a_continuous_column_carries_cost() {
        let c = vec![1.0, 3.0];
        let is_int = vec![true, false];
        assert_eq!(objective_granularity(&c, &is_int), None);
    }

    #[test]
    fn refuses_an_irrational_coefficient() {
        // Left uncapped, continued fractions would hand back 1/33102 here.
        let c = vec![1.0, std::f64::consts::PI];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), None);
    }

    #[test]
    fn handles_negative_and_mixed_sign_coefficients() {
        let c = vec![-2.0, 4.0, -6.0];
        let is_int = vec![true; 3];
        assert_eq!(objective_granularity(&c, &is_int), Some(2.0));
    }

    #[test]
    fn refuses_a_denominator_past_the_cap() {
        let c = vec![1.0, 1.0 / 4001.0];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), None);
    }

    #[test]
    fn refuses_a_constant_objective() {
        let c = vec![0.0, 0.0];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), None);
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
            let Some(g) = objective_granularity(&c, &is_int) else {
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

    /// #1189 review, finding 1: a tiny coefficient on a FREE CONTINUOUS column
    /// must disqualify the row, whatever its magnitude.
    ///
    /// `|v| <= ZERO_TOL` said nothing about how far the column can move the
    /// objective: the contribution is `-(v/a_ij) * z_k`, and with the engine's
    /// `INF = 1e20` sentinel `z_k` is not small. Judging the coefficient instead
    /// of the coefficient-times-width is the gear4 mistake CLAUDE.md records --
    /// "test unboundedness on the bound, never on a product" -- and it published
    /// a lattice the optimum does not sit on.
    #[test]
    fn a_tiny_coefficient_on_an_unbounded_column_does_not_make_a_lattice() {
        // columns: z(0, cont, costed), x(1, int), w(2, cont, free at the sentinel)
        // row: z - 2x - 1e-13 w = 0
        let c = vec![1.0, 0.0, 0.0];
        let is_int = vec![false, true, false];
        let a = vec![1.0, -2.0, -1e-13];
        let lo = vec![-1e20, 0.0, -1e20];
        let up = vec![1e20, 10.0, 1e20];
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
        );
        assert_eq!(
            got,
            None,
            "a free continuous column with |v|={} moves the objective by up to \
             {} -- refusing is the only sound answer, got {:?}",
            1e-13,
            1e-13 * 1e20,
            got
        );
    }

    /// #1189 review, finding 2: the `1e20` INF sentinel is not a finite bound.
    ///
    /// `f64::is_finite` is true at 1e20, so a column "fixed" at the sentinel
    /// anchored the lattice there. `round_bound_up` then evaluates
    /// `(v - 1e20)/g`, whose cancellation makes the lift numeric garbage that
    /// can land above `v` and be published as a dual bound.
    #[test]
    fn a_column_pinned_at_the_inf_sentinel_is_not_fixed() {
        assert!(
            !is_fixed(1e20, 1e20),
            "1e20 is the engine's INF sentinel, not a finite bound"
        );
        assert!(!is_fixed(-1e20, -1e20), "the negative sentinel likewise");
        assert!(is_fixed(3.5, 3.5), "an ordinary fixed column still counts");

        // and end to end: the sentinel must not anchor a shift.
        let c = vec![1.0, 1.0];
        let is_int = vec![true, false];
        let a = vec![1.0, 1.0];
        let lo = vec![0.0, 1e20];
        let up = vec![10.0, 1e20];
        let b = vec![0.0];
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
        );
        assert!(
            got.is_none_or(|l| l.shift.abs() < 1e19),
            "the lattice must not be anchored at the INF sentinel: {got:?}"
        );
    }

    /// #1189 review, finding 1 (companion): an exact structural zero is still
    /// skippable -- it contributes nothing at any width -- so the fix must not
    /// refuse rows it previously resolved for a good reason.
    #[test]
    fn an_exact_zero_coefficient_still_resolves_the_row() {
        let c = vec![1.0, 0.0, 0.0];
        let is_int = vec![false, true, false];
        let a = vec![1.0, -2.0, 0.0];
        let lo = vec![-100.0, 0.0, -1e20];
        let up = vec![100.0, 10.0, 1e20];
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
        );
        assert_eq!(
            got.map(|l| l.spacing),
            Some(2.0),
            "an exact zero contributes nothing and must not disqualify the row"
        );
    }
}
