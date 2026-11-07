//! Aggregated complemented mixed-integer rounding (c-MIR) cuts with variable-bound
//! substitution (Marchand & Wolsey 2001), separated from the model's own rows (#2107).
//!
//! # Why this exists
//!
//! The single-row MIR of [`super::mir`] needs every continuous column to have a finite
//! simple bound to substitute. On big-M OA masters most continuous columns have none:
//! their only upper bound is a *variable* bound row `x_j <= beta + M y_k` coupling them
//! to a binary. Measured on the #2107 masters (`syn20m03m`: 183 of 315 columns open on
//! one side, 249 such two-column rows; `rsyn0820m`: 127 of 215, 67 rows), single-row
//! MIR found 0 cuts and the in-house GMI loop at any budget stopped at the bound SCIP's
//! GMI-only root reaches (-4096 vs -4045, claim -2647). SCIP's c-MIR alone closes
//! 99.6 % / 80 % of the same root gaps, and a prototype of this separator let the
//! in-house engine confirm both masters (159 and 1171 nodes) where it could not in
//! 20 s before.
//!
//! # The construction
//!
//! The driver's matrix is standard form `A z = b`, every row an equality over
//! structural and slack columns, so any combination of rows is an equality and is in
//! particular a valid `<=` row. A separation step:
//!
//! 1. **Aggregation.** Start from an original row (either sign). Up to `max_aggr`
//!    times, pick the continuous column farthest (at the LP point) from its closest
//!    bound and eliminate it with another original row containing it, preferring the
//!    row that is tightest at the LP point.
//! 2. **Bound substitution.** Each continuous column `x_j` is replaced by a
//!    nonnegative `s_j` through its closest bound at the LP point: `x_j = l_j + s_j`,
//!    `x_j = u_j - s_j`, or a variable bound `x_j = beta + M y_k -/+ s_j`, which moves
//!    `M` onto the integer `y_k`. Each integer column is shifted to its lower bound or
//!    complemented to its upper bound (whichever is nearer), giving `y' >= 0`.
//! 3. **MIR** at scalings `delta` in `{1} u {1/|a_k|}` (and 2x, 4x, 8x of each once a
//!    cut is found): with `f0 = frac(delta beta')`,
//!    `sum floor(delta a_k) + max(0, f_k - f0)/(1 - f0) y'_k
//!     + sum min(delta c_j, 0)/(1 - f0) s_j <= floor(delta beta')`.
//! 4. **Back-substitution** to `z`, and the most efficacious cut per starting row.
//!
//! # Soundness
//!
//! Every row used is an original model row, aggregated with real multipliers (all rows
//! are equalities). A variable bound is derived from one original row with at most one
//! other continuous column, bounded over its box, so it holds at every feasible point.
//! Each substitution is an affine identity whose new variable is nonnegative at every
//! feasible point, and integer-valued exactly when the shifted bound is integral (the
//! C-4 premise of [`super::mir`]; a fractional bound takes the continuous coefficient
//! instead). MIR is valid for `{y' in Z^+, s >= 0 : row}`. So each cut is satisfied by
//! every integer-feasible `z`, whatever LP point it was separated at.
//!
//! Floating point: multipliers and coefficients are computed in `f64`, as GMI's are;
//! the rhs is relaxed by [`RHS_SAFETY`] relative to its terms. A term is removed from a
//! cut only by charging its maximum over the box to the rhs (a relaxation), never by
//! deleting it (a strengthening): the #1236 rule. The cut row then goes through the
//! driver's `substitute_slacks_to_structural` like every other root cut.

use std::collections::BTreeMap;

use super::gomory::GomoryCut;
use super::simplex::sparse::SparseCols;

/// The LP layer's unbounded sentinel; bounds are tested against it, never products.
const INF: f64 = 1e20;
/// Minimum fractionality of the scaled rhs (avoids tiny `1 - f0` denominators).
const FRAC_MIN: f64 = 1e-3;
/// HiGHS's f0 window (`HighsCutGeneration.cpp:526`), under `DISCOPT_MILP_CMIR_HIGHS`.
const FRAC_MIN_HIGHS: f64 = 0.005;
/// HiGHS's feasibility tolerance for "off its bound" and delta deduplication.
const HIGHS_FEASTOL: f64 = 1e-6;

/// D2 W4 (`DISCOPT_MILP_CMIR_HIGHS`, default OFF): HiGHS's c-MIR delta selection
/// (`HighsCutGeneration.cpp:529-636`): a candidate for every integer column off its
/// bound at the LP point (not the first `MAX_DELTAS` by index), f0 in [0.005, 0.995],
/// and the best delta retried as a coarser divisor (x2, x4, x8).
/// `DISCOPT_MILP_CMIR_FLIP=1` adds HiGHS's complementation flip pass (`:638-677`).
fn highs_cmir_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("DISCOPT_MILP_CMIR_HIGHS").unwrap_or_default();
        env_bool(&raw, "DISCOPT_MILP_CMIR_HIGHS")
    })
}

/// D2 W4 (`DISCOPT_MILP_LIFTED_MIC`, default OFF): also try HiGHS's lifted
/// mixed-integer cover (`HighsCutGeneration.cpp:25-129, 280-500`) on every c-MIR base
/// row and keep whichever of the two cuts is more efficacious.
fn lifted_mic_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("DISCOPT_MILP_LIFTED_MIC").unwrap_or_default();
        env_bool(&raw, "DISCOPT_MILP_LIFTED_MIC")
    })
}

fn cmir_flip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("DISCOPT_MILP_CMIR_FLIP").unwrap_or_default();
        env_bool(&raw, "DISCOPT_MILP_CMIR_FLIP")
    })
}

fn env_bool(raw: &str, name: &str) -> bool {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "no" => false,
        "1" | "true" | "yes" => true,
        other => panic!("{name}={other:?} is not recognized (expected 1 or 0)"),
    }
}
/// Absolute cap on a cut coefficient or rhs.
const MAX_ABS_COEFF: f64 = 1e7;
/// Largest coefficient ratio a cut may carry.
const MAX_DYNAMISM: f64 = 1e7;
/// Least efficacy (violation over norm) worth returning.
const MIN_EFFICACY: f64 = 1e-4;
/// Integrality of a shifted integer column's active bound.
const INT_BOUND_TOL: f64 = 1e-6;
/// Scalings tried per row (beyond `delta = 1`).
const MAX_DELTAS: usize = 8;
/// Relative relaxation of every cut's rhs, covering `f64` rounding of the multipliers
/// and coefficients that produced it.
const RHS_SAFETY: f64 = 1e-9;
/// A starting row is used only when it is this tight at the LP point, relative to its
/// largest coefficient (SCIP's `maxslackroot` idea).
const MAX_BASE_SLACK: f64 = 0.1;

/// `x_j <= beta + m * y_k` (an upper variable bound) or `>=` (lower), on integer `k`.
#[derive(Clone, Copy, Debug)]
struct VarBound {
    k: usize,
    beta: f64,
    m: f64,
}

/// How a continuous column is substituted.
#[derive(Clone, Copy, Debug)]
enum Sub {
    Lb,
    Ub,
    Vlb(VarBound),
    Vub(VarBound),
}

/// The problem a separation round reads.
struct Ctx<'a> {
    l: &'a [f64],
    u: &'a [f64],
    x: &'a [f64],
    is_int: &'a [bool],
    tol: f64,
    vlb: Vec<Vec<VarBound>>,
    vub: Vec<Vec<VarBound>>,
}

impl Ctx<'_> {
    /// The closest bound of continuous `j` at the LP point: `(distance, substitution)`.
    /// Ties prefer a variable bound (it carries the integer information). `None` when
    /// the column has no finite bound of any kind.
    fn closest(&self, j: usize) -> Option<(f64, Sub)> {
        let x = self.x;
        let mut best: Option<(f64, u8, Sub)> = None;
        let mut offer = |d: f64, rank: u8, s: Sub| {
            let d = d.max(0.0);
            if best.is_none_or(|(bd, br, _)| d < bd || (d == bd && rank < br)) {
                best = Some((d, rank, s));
            }
        };
        for vb in &self.vlb[j] {
            offer(x[j] - (vb.beta + vb.m * x[vb.k]), 0, Sub::Vlb(*vb));
        }
        for vb in &self.vub[j] {
            offer(vb.beta + vb.m * x[vb.k] - x[j], 0, Sub::Vub(*vb));
        }
        if self.l[j] > -INF {
            offer(x[j] - self.l[j], 1, Sub::Lb);
        }
        if self.u[j] < INF {
            offer(self.u[j] - x[j], 1, Sub::Ub);
        }
        best.map(|(d, _, s)| (d, s))
    }

    /// Distance of continuous `j` to its closest bound (`INF` when it has none).
    fn dist(&self, j: usize) -> f64 {
        self.closest(j).map_or(INF, |(d, _)| d)
    }
}

/// Variable bounds from the original rows with two or three nonzeros: one continuous
/// column `j`, one integer column `k`, and at most one other continuous column, which
/// is bounded over its box.
fn variable_bounds(
    rows: &[Vec<(usize, f64)>],
    b: &[f64],
    l: &[f64],
    u: &[f64],
    is_int: &[bool],
    n: usize,
) -> (Vec<Vec<VarBound>>, Vec<Vec<VarBound>>) {
    let mut vlb = vec![Vec::new(); n];
    let mut vub = vec![Vec::new(); n];
    for (i, nz) in rows.iter().enumerate() {
        if nz.len() < 2 || nz.len() > 3 {
            continue;
        }
        for &(j, aj) in nz {
            if is_int[j] || aj == 0.0 {
                continue;
            }
            for &(k, ak) in nz {
                if k == j || !is_int[k] || ak == 0.0 {
                    continue;
                }
                // The remaining entry (if any) must be continuous; its contribution
                // `a_o z_o` ranges over `[rmin, rmax]` on its box.
                let mut rmin = 0.0f64;
                let mut rmax = 0.0f64;
                let mut rmin_inf = false;
                let mut rmax_inf = false;
                let mut ok = true;
                for &(o, ao) in nz {
                    if o == j || o == k {
                        continue;
                    }
                    if is_int[o] {
                        ok = false;
                        break;
                    }
                    let (lo_inf, hi_inf) = (l[o] <= -INF, u[o] >= INF);
                    if ao > 0.0 {
                        rmin += ao * l[o];
                        rmax += ao * u[o];
                        rmin_inf |= lo_inf;
                        rmax_inf |= hi_inf;
                    } else {
                        rmin += ao * u[o];
                        rmax += ao * l[o];
                        rmin_inf |= hi_inf;
                        rmax_inf |= lo_inf;
                    }
                }
                if !ok {
                    continue;
                }
                let m = -ak / aj;
                // a_j x_j = b - a_k y_k - rest, rest in [rmin, rmax].
                if !rmin_inf {
                    let vb = VarBound {
                        k,
                        beta: (b[i] - rmin) / aj,
                        m,
                    };
                    if aj > 0.0 {
                        vub[j].push(vb);
                    } else {
                        vlb[j].push(vb);
                    }
                }
                if !rmax_inf {
                    let vb = VarBound {
                        k,
                        beta: (b[i] - rmax) / aj,
                        m,
                    };
                    if aj > 0.0 {
                        vlb[j].push(vb);
                    } else {
                        vub[j].push(vb);
                    }
                }
            }
        }
    }
    (vlb, vub)
}

/// A cut `coef . z <= rhs` (sparse) with its efficacy at the LP point.
struct Cut {
    coef: BTreeMap<usize, f64>,
    rhs: f64,
    efficacy: f64,
}

/// One integer term of the substituted row: column, complemented?, coefficient on
/// `y'`, and whether `y'` is integer-valued (the C-4 premise).
#[derive(Clone, Copy, Debug)]
struct IntTerm {
    k: usize,
    comp: bool,
    a: f64,
    int_ok: bool,
}

/// c-MIR on one aggregated row `sum coef_j z_j <= rhs`.
fn cmir_row(ctx: &Ctx<'_>, coef: &BTreeMap<usize, f64>, rhs: f64) -> Option<Cut> {
    let (l, u, x) = (ctx.l, ctx.u, ctx.x);
    let mut icoef: BTreeMap<usize, f64> = BTreeMap::new();
    let mut conts: Vec<(usize, Sub, f64)> = Vec::new();
    let mut rhs2 = rhs;
    for (&j, &aj) in coef {
        if ctx.is_int[j] {
            *icoef.entry(j).or_insert(0.0) += aj;
        }
    }
    for (&j, &aj) in coef {
        if ctx.is_int[j] {
            continue;
        }
        let (_, sub) = ctx.closest(j)?;
        match sub {
            Sub::Lb => {
                rhs2 -= aj * l[j];
                conts.push((j, sub, aj));
            }
            Sub::Ub => {
                rhs2 -= aj * u[j];
                conts.push((j, sub, -aj));
            }
            Sub::Vlb(vb) => {
                rhs2 -= aj * vb.beta;
                *icoef.entry(vb.k).or_insert(0.0) += aj * vb.m;
                conts.push((j, sub, aj));
            }
            Sub::Vub(vb) => {
                rhs2 -= aj * vb.beta;
                *icoef.entry(vb.k).or_insert(0.0) += aj * vb.m;
                conts.push((j, sub, -aj));
            }
        }
    }
    let mut ints: Vec<IntTerm> = Vec::with_capacity(icoef.len());
    for (&k, &ak) in &icoef {
        if ak == 0.0 {
            continue;
        }
        if l[k] <= -INF {
            return None;
        }
        let comp = u[k] < INF && x[k] - l[k] > 0.5 * (u[k] - l[k]);
        let bound = if comp { u[k] } else { l[k] };
        let int_ok = (bound - bound.round()).abs() <= INT_BOUND_TOL;
        if comp {
            rhs2 -= ak * u[k];
            ints.push(IntTerm {
                k,
                comp,
                a: -ak,
                int_ok,
            });
        } else {
            rhs2 -= ak * l[k];
            ints.push(IntTerm {
                k,
                comp,
                a: ak,
                int_ok,
            });
        }
    }
    if ints.iter().all(|t| !t.int_ok) || !rhs2.is_finite() {
        return None; // no integer structure to round
    }
    let lifted = if lifted_mic_enabled() {
        lifted_mic_cut(ctx, &ints, &conts, rhs2)
    } else {
        None
    };
    let mir = if highs_cmir_enabled() {
        cmir_row_highs(ctx, ints, &conts, rhs2)
    } else {
        cmir_row_legacy(ctx, &ints, &conts, rhs2)
    };
    match (mir, lifted) {
        (Some(a), Some(b)) => Some(if b.efficacy > a.efficacy { b } else { a }),
        (a, b) => a.or(b),
    }
}

/// The pre-W4 delta selection (the first `MAX_DELTAS` integer coefficients by index).
fn cmir_row_legacy(
    ctx: &Ctx<'_>,
    ints: &[IntTerm],
    conts: &[(usize, Sub, f64)],
    rhs2: f64,
) -> Option<Cut> {
    let (ints, conts) = (ints.to_vec(), conts.to_vec());
    let mut deltas: Vec<f64> = vec![1.0];
    for t in &ints {
        if t.int_ok && t.a.abs() > ctx.tol && deltas.len() <= MAX_DELTAS {
            let d = 1.0 / t.a.abs();
            if d.is_finite() && !deltas.iter().any(|&e| (e - d).abs() <= 1e-12 * d) {
                deltas.push(d);
            }
        }
    }
    let mut best: Option<Cut> = None;
    for &d in &deltas {
        try_delta(ctx, &ints, &conts, rhs2, d, &mut best);
    }
    if best.is_some() {
        for mult in [2.0, 4.0, 8.0] {
            for &d in &deltas {
                try_delta(ctx, &ints, &conts, rhs2, d * mult, &mut best);
            }
        }
    }
    best
}

/// HiGHS's lifted mixed-integer cover on a `<=` knapsack in transformed space
/// (`HighsCutGeneration.cpp:25-129` `determineCover(lpSol = true)` and `:280-500`
/// `separateLiftedMixedIntegerCover`): `sum vals_i x_i <= rhs` with every integer
/// `x_i in [0, upper_i]` (`upper_i` integral, `vals_i > 0`) and continuous `x_i >= 0`.
/// Returns the cut's coefficients (same indexing) and rhs, `sum c_i x_i <= rhs_cut`.
/// Continuous entries keep a negative coefficient and drop a positive one (a
/// relaxation at their zero lower bound). Pure, so its validity is brute-forced
/// directly in the tests.
fn lifted_mic_knapsack(
    vals: &[f64],
    upper: &[f64],
    solval: &[f64],
    is_int: &[bool],
    rhs: f64,
) -> Option<(Vec<f64>, f64)> {
    let feastol = HIGHS_FEASTOL;
    if rhs <= 10.0 * feastol {
        return None;
    }
    // determineCover(lpSol = true)
    let mut cover: Vec<usize> = (0..vals.len())
        .filter(|&j| is_int[j] && vals[j] > 0.0 && solval[j] > feastol)
        .collect();
    let max_cover = cover.len();
    let at_upper = |j: usize| solval[j] >= upper[j] - feastol;
    cover.sort_by_key(|&j| !at_upper(j));
    let mut coversize = cover.iter().take_while(|&&j| at_upper(j)).count();
    let mut coverweight: f64 = cover[..coversize].iter().map(|&j| vals[j] * upper[j]).sum();
    cover[coversize..].sort_by(|&i, &j| {
        let (bi, bj) = (upper[i] < 1.5, upper[j] < 1.5);
        if bi != bj {
            return bj.cmp(&bi);
        }
        let (ca, cb) = (solval[i] * vals[i], solval[j] * vals[j]);
        if ca > cb + feastol {
            return std::cmp::Ordering::Less;
        }
        if ca < cb - feastol {
            return std::cmp::Ordering::Greater;
        }
        vals[j].total_cmp(&vals[i]).then(i.cmp(&j))
    });
    let minlambda = (10.0 * feastol).max(feastol * rhs.abs());
    while coversize != max_cover {
        if coverweight - rhs > minlambda {
            break;
        }
        let j = cover[coversize];
        coverweight += vals[j] * upper[j];
        coversize += 1;
    }
    if coversize == 0 {
        return None;
    }
    let lambda = coverweight - rhs;
    if lambda <= minlambda {
        return None;
    }
    cover.truncate(coversize);
    let in_cover: Vec<bool> = {
        let mut f = vec![false; vals.len()];
        for &j in &cover {
            f[j] = true;
        }
        f
    };
    // separateLiftedMixedIntegerCover
    cover.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]).then(a.cmp(&b)));
    let mut a: Vec<f64> = Vec::with_capacity(coversize);
    let mut u: Vec<f64> = Vec::with_capacity(coversize + 1);
    let mut m: Vec<f64> = Vec::with_capacity(coversize + 1);
    let (mut usum, mut msum) = (0.0f64, 0.0f64);
    for &i in &cover {
        u.push(usum);
        m.push(msum);
        a.push(vals[i]);
        usum += upper[i];
        msum += upper[i] * vals[i];
    }
    u.push(usum);
    m.push(msum);
    let cplus_end = |thr: f64| cover.iter().take_while(|&&i| thr <= vals[i]).count();
    let mut lpos: Option<usize> = None;
    let mut best_cplusend = 0usize;
    let mut best_lval = 0.0f64;
    let mut best_at_upper = true;
    for (i, &j) in cover.iter().enumerate() {
        let ub = upper[j];
        let atu = solval[j] >= ub - feastol;
        if atu && !best_at_upper {
            continue;
        }
        let mju = ub * vals[j];
        let mu = mju - lambda;
        if mu <= 10.0 * feastol || vals[j].abs() < 1000.0 * feastol {
            continue;
        }
        let mudival = mu / vals[j];
        if (mudival - mudival.round()).abs() <= feastol {
            continue;
        }
        let eta = mudival.ceil();
        let thr = (ub - eta + 1.0) * vals[j];
        let ce = cplus_end(thr);
        let mut mcplus = m[ce];
        if i < ce {
            mcplus -= mju;
        }
        let jl = mcplus + eta * vals[j];
        if jl > best_lval || (!atu && best_at_upper) {
            lpos = Some(i);
            best_cplusend = ce;
            best_lval = jl;
            best_at_upper = atu;
        }
    }
    let lpos = lpos?;
    let l = cover[lpos];
    let al = vals[l];
    let upperl = upper[l];
    let mlu = upperl * al;
    let mu = mlu - lambda;
    let mut cv: Vec<usize> = cover[..best_cplusend].to_vec();
    a.truncate(best_cplusend);
    u.truncate(best_cplusend + 1);
    m.truncate(best_cplusend + 1);
    if lpos < best_cplusend {
        a.remove(lpos);
        cv.remove(lpos);
        u.remove(lpos + 1);
        m.remove(lpos + 1);
        for i in (lpos + 1)..best_cplusend {
            u[i] -= upperl;
            m[i] -= mlu;
        }
    }
    let cplussize = a.len();
    let mudival = mu / al;
    let eta = mudival.ceil();
    let r = (mu - mudival.floor() * al).max(0.0);
    let ulme1 = upperl - eta + 1.0;
    let thr = ulme1 * al;
    let kmin = (eta - upperl - 0.5).floor() as i64;
    let phi_l = |aa: f64| -> f64 {
        let mut k = ((aa / al) as i64).min(-1);
        while k >= kmin {
            let kf = k as f64;
            if aa >= kf * al + r {
                return aa - (kf + 1.0) * r;
            }
            if aa >= kf * al {
                return kf * (al - r);
            }
            k -= 1;
        }
        kmin as f64 * (al - r)
    };
    let kmax = (upperl - eta + 0.5).floor() as i64;
    let gamma_l = |z: f64| -> f64 {
        for i in 0..cplussize {
            let upi = upper[cv[i]] as i64;
            for h in 0..=upi {
                let hf = h as f64;
                let mih = m[i] + hf * a[i];
                let uih = u[i] + hf;
                let mpd = mih + a[i] - thr;
                if z <= mpd {
                    return uih * ulme1 * (al - r);
                }
                let mut k = ((z - mpd) / al) as i64 - 1;
                while k <= kmax {
                    let kf = k as f64;
                    if z <= mpd + kf * al + r {
                        return (uih * ulme1 + kf) * (al - r);
                    }
                    if z <= mpd + (kf + 1.0) * al {
                        return (uih * ulme1) * (al - r) + z - mih - a[i] + thr - (kf + 1.0) * r;
                    }
                    k += 1;
                }
            }
        }
        let mc = m[cplussize];
        let mut p = ((z - mc) / al) as i64 - 1;
        loop {
            let pf = p as f64;
            if z <= mc + pf * al + r {
                return (u[cplussize] * ulme1 + pf) * (al - r);
            }
            if z <= mc + (pf + 1.0) * al {
                return (u[cplussize] * ulme1) * (al - r) + z - mc - (pf + 1.0) * r;
            }
            p += 1;
        }
    };
    let mut out = vec![0.0; vals.len()];
    let mut cut_rhs = (upperl - eta) * r - lambda;
    for i in 0..vals.len() {
        let v = vals[i];
        if v == 0.0 {
            continue;
        }
        if !is_int[i] {
            if v < 0.0 {
                out[i] = v;
            }
            continue;
        }
        if in_cover[i] {
            out[i] = -phi_l(-v);
            cut_rhs += out[i] * upper[i];
        } else {
            out[i] = gamma_l(v);
        }
    }
    if !cut_rhs.is_finite() || out.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some((out, cut_rhs))
}

/// `DISCOPT_MILP_LIFTED_MIC`: the lifted mixed-integer cover of one substituted c-MIR
/// base row (`sum a y' + sum cs s <= rhs2`, `y'` complemented integers, `s >= 0`),
/// mapped back to `z` exactly as [`try_delta`] maps a MIR. Every integer must have a
/// finite integral range; integers with a negative coefficient are complemented the
/// other way first (an affine identity on a finite bound).
fn lifted_mic_cut(
    ctx: &Ctx<'_>,
    ints: &[IntTerm],
    conts: &[(usize, Sub, f64)],
    rhs2: f64,
) -> Option<Cut> {
    let (l, u, x) = (ctx.l, ctx.u, ctx.x);
    let mut ints: Vec<IntTerm> = ints.to_vec();
    let mut rhs = rhs2;
    for t in ints.iter_mut() {
        let k = t.k;
        let range = u[k] - l[k];
        if l[k] <= -INF || u[k] >= INF || (range - range.round()).abs() > INT_BOUND_TOL || !t.int_ok
        {
            return None;
        }
        if t.a < 0.0 {
            let ak = if t.comp { -t.a } else { t.a };
            let (old_b, new_b) = if t.comp { (u[k], l[k]) } else { (l[k], u[k]) };
            rhs += ak * old_b - ak * new_b;
            t.comp = !t.comp;
            t.a = -t.a;
        }
    }
    let n_i = ints.len();
    let mut vals = Vec::with_capacity(n_i + conts.len());
    let mut upper = Vec::with_capacity(n_i + conts.len());
    let mut solval = Vec::with_capacity(n_i + conts.len());
    let mut is_int = Vec::with_capacity(n_i + conts.len());
    for t in &ints {
        vals.push(t.a);
        upper.push((u[t.k] - l[t.k]).round());
        solval.push(if t.comp {
            u[t.k] - x[t.k]
        } else {
            x[t.k] - l[t.k]
        });
        is_int.push(true);
    }
    for &(_, _, cs) in conts {
        vals.push(cs);
        upper.push(INF);
        solval.push(0.0);
        is_int.push(false);
    }
    let (cv, crhs) = lifted_mic_knapsack(&vals, &upper, &solval, &is_int, rhs)?;
    // Back to z (same identities as `try_delta`).
    let mut cx: BTreeMap<usize, f64> = BTreeMap::new();
    let mut rx = crhs;
    let add = |cx: &mut BTreeMap<usize, f64>, j: usize, v: f64| {
        if v != 0.0 {
            *cx.entry(j).or_insert(0.0) += v;
        }
    };
    for (t, &g) in ints.iter().zip(&cv[..n_i]) {
        if g == 0.0 {
            continue;
        }
        if t.comp {
            add(&mut cx, t.k, -g);
            rx -= g * u[t.k];
        } else {
            add(&mut cx, t.k, g);
            rx += g * l[t.k];
        }
    }
    for (&(j, sub, _), &g) in conts.iter().zip(&cv[n_i..]) {
        if g == 0.0 {
            continue;
        }
        match sub {
            Sub::Lb => {
                add(&mut cx, j, g);
                rx += g * l[j];
            }
            Sub::Ub => {
                add(&mut cx, j, -g);
                rx -= g * u[j];
            }
            Sub::Vlb(vb) => {
                add(&mut cx, j, g);
                add(&mut cx, vb.k, -g * vb.m);
                rx += g * vb.beta;
            }
            Sub::Vub(vb) => {
                add(&mut cx, j, -g);
                add(&mut cx, vb.k, g * vb.m);
                rx -= g * vb.beta;
            }
        }
    }
    if !drop_small_coeffs(&mut cx, &mut rx, l, u) {
        return None;
    }
    let mut out: BTreeMap<usize, f64> = BTreeMap::new();
    let (mut max_c, mut min_c, mut mag) = (0.0f64, f64::INFINITY, rx.abs());
    for (&j, &v) in &cx {
        if v == 0.0 {
            continue;
        }
        if !v.is_finite() {
            return None;
        }
        max_c = max_c.max(v.abs());
        min_c = min_c.min(v.abs());
        mag += (v * x[j]).abs();
        out.insert(j, v);
    }
    if out.is_empty()
        || max_c > MAX_ABS_COEFF
        || rx.abs() > MAX_ABS_COEFF
        || max_c / min_c > MAX_DYNAMISM
    {
        return None;
    }
    rx += RHS_SAFETY * (1.0 + mag);
    let f = (2.0f64).powi(-(max_c.log2().round() as i32));
    if f.is_finite() && f > 0.0 {
        for v in out.values_mut() {
            *v *= f;
        }
        rx *= f;
    }
    let act: f64 = out.iter().map(|(&j, &v)| v * x[j]).sum();
    let norm = out.values().map(|v| v * v).sum::<f64>().sqrt();
    let efficacy = (act - rx) / norm;
    (efficacy > MIN_EFFICACY).then_some(Cut {
        coef: out,
        rhs: rx,
        efficacy,
    })
}

/// HiGHS's small-coefficient removal (`HighsCutGeneration.cpp:784-805`) on a cut
/// `sum v_j z_j <= rx`: a coefficient at most `100 * feastol * max(max|v|, 1e-3)`
/// leaves the cut by charging its minimum over the box to the rhs (#1236: `v > 0`
/// charges `v * l_j`, `v < 0` charges `v * u_j`). A small coefficient on a column
/// with an infinite bound in that direction rejects the cut (`false`), as in HiGHS.
fn drop_small_coeffs(cx: &mut BTreeMap<usize, f64>, rx: &mut f64, l: &[f64], u: &[f64]) -> bool {
    let maxabs = cx.values().fold(0.0f64, |m, v| m.max(v.abs()));
    let minc = 100.0 * HIGHS_FEASTOL * maxabs.max(1e-3);
    let mut drop: Vec<usize> = Vec::new();
    for (&j, &v) in cx.iter() {
        if v != 0.0 && v.abs() <= minc {
            let pin = if v > 0.0 { l[j] } else { u[j] };
            if pin <= -INF || pin >= INF {
                return false;
            }
            *rx -= v * pin;
            drop.push(j);
        }
    }
    for j in drop {
        cx.remove(&j);
    }
    true
}

/// The cut [`try_delta`] produces at one scaling, if any.
fn cut_at(
    ctx: &Ctx<'_>,
    ints: &[IntTerm],
    conts: &[(usize, Sub, f64)],
    rhs2: f64,
    d: f64,
) -> Option<Cut> {
    let mut c = None;
    try_delta(ctx, ints, conts, rhs2, d, &mut c);
    c
}

/// `DISCOPT_MILP_CMIR_HIGHS`: HiGHS's delta selection, then (with
/// `DISCOPT_MILP_CMIR_FLIP`) its complementation flip pass. Scalings here are
/// `1 / delta` in HiGHS's divisor convention.
fn cmir_row_highs(
    ctx: &Ctx<'_>,
    mut ints: Vec<IntTerm>,
    conts: &[(usize, Sub, f64)],
    mut rhs2: f64,
) -> Option<Cut> {
    let (l, u, x) = (ctx.l, ctx.u, ctx.x);
    // Off-bound value of an integer term in its complemented space.
    let off = |t: &IntTerm| {
        if t.comp {
            u[t.k] - x[t.k]
        } else {
            x[t.k] - l[t.k]
        }
    };
    let mut divisors: Vec<f64> = vec![1.0];
    for t in &ints {
        let a = t.a.abs();
        if t.int_ok
            && a > ctx.tol
            && off(t) > HIGHS_FEASTOL
            && !divisors
                .iter()
                .any(|&e| (e - a).abs() <= 10.0 * HIGHS_FEASTOL)
        {
            divisors.push(a);
        }
    }
    let mut best: Option<Cut> = None;
    let mut best_div = 0.0;
    for &dv in &divisors {
        if let Some(c) = cut_at(ctx, &ints, conts, rhs2, 1.0 / dv) {
            if best.as_ref().is_none_or(|b| c.efficacy > b.efficacy) {
                best = Some(c);
                best_div = dv;
            }
        }
    }
    best.as_ref()?;
    for mult in [2.0, 4.0, 8.0] {
        if let Some(c) = cut_at(ctx, &ints, conts, rhs2, 1.0 / (best_div * mult)) {
            if best.as_ref().is_none_or(|b| c.efficacy > b.efficacy) {
                best = Some(c);
                best_div *= mult;
            }
        }
    }
    if cmir_flip_enabled() {
        // Flip each integer with a finite upper bound whose LP value is off both
        // bounds; keep the flip when the cut's efficacy improves. Complementing is an
        // affine identity on the column's finite bound, so every flip is sound.
        let d = 1.0 / best_div;
        for i in 0..ints.len() {
            let k = ints[i].k;
            if u[k] >= INF || l[k] <= -INF {
                continue;
            }
            if x[k] - l[k] <= HIGHS_FEASTOL || u[k] - x[k] <= HIGHS_FEASTOL {
                continue;
            }
            let ak = if ints[i].comp { -ints[i].a } else { ints[i].a };
            let saved = (ints[i].comp, ints[i].a, ints[i].int_ok, rhs2);
            let (old_b, new_b) = if ints[i].comp {
                (u[k], l[k])
            } else {
                (l[k], u[k])
            };
            rhs2 += ak * old_b - ak * new_b;
            ints[i].comp = !ints[i].comp;
            ints[i].a = -ints[i].a;
            ints[i].int_ok = (new_b - new_b.round()).abs() <= INT_BOUND_TOL;
            match cut_at(ctx, &ints, conts, rhs2, d) {
                Some(c) if best.as_ref().is_none_or(|b| c.efficacy > b.efficacy) => {
                    best = Some(c);
                }
                _ => {
                    ints[i].comp = saved.0;
                    ints[i].a = saved.1;
                    ints[i].int_ok = saved.2;
                    rhs2 = saved.3;
                }
            }
        }
    }
    best
}

/// The MIR inequality of the substituted row at scaling `delta`, back in `z`.
fn try_delta(
    ctx: &Ctx<'_>,
    ints: &[IntTerm],
    conts: &[(usize, Sub, f64)],
    rhs2: f64,
    delta: f64,
    best: &mut Option<Cut>,
) {
    let (l, u, x) = (ctx.l, ctx.u, ctx.x);
    let bt = delta * rhs2;
    let f0 = bt - bt.floor();
    let fmin = if highs_cmir_enabled() {
        FRAC_MIN_HIGHS
    } else {
        FRAC_MIN
    };
    if !(fmin..=1.0 - fmin).contains(&f0) {
        return;
    }
    let mut cx: BTreeMap<usize, f64> = BTreeMap::new();
    let mut rx = bt.floor();
    let add = |cx: &mut BTreeMap<usize, f64>, j: usize, v: f64| {
        if v != 0.0 {
            *cx.entry(j).or_insert(0.0) += v;
        }
    };
    for t in ints {
        let ad = delta * t.a;
        let g = if t.int_ok {
            let fk = ad - ad.floor();
            ad.floor() + (fk - f0).max(0.0) / (1.0 - f0)
        } else {
            ad.min(0.0) / (1.0 - f0)
        };
        if g == 0.0 {
            continue;
        }
        if t.comp {
            // y' = u - y
            add(&mut cx, t.k, -g);
            rx -= g * u[t.k];
        } else {
            // y' = y - l
            add(&mut cx, t.k, g);
            rx += g * l[t.k];
        }
    }
    for &(j, sub, cs) in conts {
        let g = (delta * cs).min(0.0) / (1.0 - f0);
        if g == 0.0 {
            continue;
        }
        match sub {
            Sub::Lb => {
                add(&mut cx, j, g);
                rx += g * l[j];
            }
            Sub::Ub => {
                add(&mut cx, j, -g);
                rx -= g * u[j];
            }
            Sub::Vlb(vb) => {
                // s = x - beta - m y
                add(&mut cx, j, g);
                add(&mut cx, vb.k, -g * vb.m);
                rx += g * vb.beta;
            }
            Sub::Vub(vb) => {
                // s = beta + m y - x
                add(&mut cx, j, -g);
                add(&mut cx, vb.k, g * vb.m);
                rx -= g * vb.beta;
            }
        }
    }
    if highs_cmir_enabled() && !drop_small_coeffs(&mut cx, &mut rx, l, u) {
        return;
    }
    // #1236: a term leaves the cut only by charging its maximum over the box to the
    // rhs (`v z_j <= ...` dropped needs `rx -= min_box(v z_j)`); otherwise it stays.
    let mut max_c = 0.0f64;
    let mut min_c = f64::INFINITY;
    let mut mag = rx.abs();
    let mut out: BTreeMap<usize, f64> = BTreeMap::new();
    for (&j, &v) in &cx {
        if v == 0.0 {
            continue;
        }
        if !v.is_finite() {
            return;
        }
        if v.abs() <= ctx.tol {
            let pin = if v > 0.0 { l[j] } else { u[j] };
            if pin > -INF && pin < INF && (v * pin).abs() <= ctx.tol {
                rx -= v * pin;
                continue;
            }
        } else {
            min_c = min_c.min(v.abs());
        }
        max_c = max_c.max(v.abs());
        mag += (v * x[j]).abs();
        out.insert(j, v);
    }
    if out.is_empty()
        || !min_c.is_finite()
        || max_c > MAX_ABS_COEFF
        || rx.abs() > MAX_ABS_COEFF
        || max_c / min_c > MAX_DYNAMISM
    {
        return;
    }
    rx += RHS_SAFETY * (1.0 + mag);
    if highs_cmir_enabled() {
        // HiGHS scales every cut (`postprocessCut`); a power of two keeps it exact.
        let f = (2.0f64).powi(-(max_c.log2().round() as i32));
        if f.is_finite() && f > 0.0 {
            for v in out.values_mut() {
                *v *= f;
            }
            rx *= f;
        }
    }
    let act: f64 = out.iter().map(|(&j, &v)| v * x[j]).sum();
    let norm = out.values().map(|v| v * v).sum::<f64>().sqrt();
    let efficacy = (act - rx) / norm;
    if efficacy > MIN_EFFICACY && best.as_ref().is_none_or(|b| efficacy > b.efficacy) {
        *best = Some(Cut {
            coef: out,
            rhs: rx,
            efficacy,
        });
    }
}

/// How far row `i` is from tight at the LP point: the distance of its singleton
/// (slack-like) continuous columns to their closest bounds, weighted by `|a|`.
fn row_slack(ctx: &Ctx<'_>, nz: &[(usize, f64)], col_rows: &[Vec<usize>], scale: f64) -> f64 {
    let mut s = 0.0;
    for &(o, ao) in nz {
        if !ctx.is_int[o] && col_rows[o].len() == 1 {
            s += (scale * ao).abs() * ctx.dist(o);
        }
    }
    s
}

/// Separate aggregated c-MIR cuts at the LP point `x` from the first `n_orig_rows`
/// rows of the standard-form matrix (`csc`, `m` rows by `n` columns, `A z = b`).
///
/// Returns at most `max_cuts` cuts (most efficacious first) in the driver's
/// `coeffs . z >= rhs` convention over all `n` columns; the driver rewrites slack
/// references over structural columns. `max_aggr` bounds the aggregation steps per
/// starting row (0 is single-row c-MIR with variable bounds).
#[allow(clippy::too_many_arguments)]
pub fn separate_cmir_csc(
    csc: &SparseCols,
    n: usize,
    m: usize,
    l: &[f64],
    u: &[f64],
    b: &[f64],
    x: &[f64],
    is_int: &[bool],
    n_orig_rows: usize,
    max_aggr: usize,
    max_cuts: usize,
    tol: f64,
) -> Vec<GomoryCut> {
    let rows_n = n_orig_rows.min(m);
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); rows_n];
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    let (col_ptr, row_idx, vals) = csc.raw();
    for j in 0..n {
        for idx in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[idx];
            if i < rows_n && vals[idx] != 0.0 {
                rows[i].push((j, vals[idx]));
                col_rows[j].push(i);
            }
        }
    }
    let (vlb, vub) = variable_bounds(&rows, b, l, u, is_int, n);
    let ctx = Ctx {
        l,
        u,
        x,
        is_int,
        tol,
        vlb,
        vub,
    };
    let mut found: Vec<Cut> = Vec::new();
    for (i, nz) in rows.iter().enumerate() {
        // A starting row needs integer structure: an integer column, or a continuous
        // one whose variable bound will bring an integer in.
        let has_int_structure = nz
            .iter()
            .any(|&(j, _)| is_int[j] || !ctx.vlb[j].is_empty() || !ctx.vub[j].is_empty());
        if !has_int_structure {
            continue;
        }
        let amax = nz.iter().fold(0.0f64, |a, &(_, v)| a.max(v.abs()));
        if row_slack(&ctx, nz, &col_rows, 1.0) > MAX_BASE_SLACK * amax + 1e-6 {
            continue;
        }
        for sign in [1.0f64, -1.0] {
            let mut cur: BTreeMap<usize, f64> = nz.iter().map(|&(j, v)| (j, sign * v)).collect();
            let mut crhs = sign * b[i];
            let mut used = vec![i];
            let mut row_best: Option<Cut> = None;
            for level in 0..=max_aggr {
                if let Some(c) = cmir_row(&ctx, &cur, crhs) {
                    if row_best.as_ref().is_none_or(|b| c.efficacy > b.efficacy) {
                        row_best = Some(c);
                    }
                }
                if level == max_aggr {
                    break;
                }
                // Eliminate the continuous column farthest from its closest bound.
                let mut bad: Option<(usize, f64)> = None;
                for (&j, &aj) in &cur {
                    if is_int[j] || aj == 0.0 {
                        continue;
                    }
                    let d = ctx.dist(j);
                    if d > 1e-6 && bad.is_none_or(|(_, bd)| d > bd) {
                        bad = Some((j, d));
                    }
                }
                let Some((j, _)) = bad else { break };
                let aj = cur[&j];
                let mut cand: Option<(f64, usize, f64)> = None;
                for &r2 in &col_rows[j] {
                    if used.contains(&r2) {
                        continue;
                    }
                    let Some(&(_, a2j)) = rows[r2].iter().find(|&&(c, _)| c == j) else {
                        continue;
                    };
                    let lam = -aj / a2j;
                    if !lam.is_finite() {
                        continue;
                    }
                    let score = row_slack(&ctx, &rows[r2], &col_rows, lam);
                    if cand.is_none_or(|(s, _, _)| score < s) {
                        cand = Some((score, r2, lam));
                    }
                }
                let Some((_, r2, lam)) = cand else { break };
                for &(c, v) in &rows[r2] {
                    *cur.entry(c).or_insert(0.0) += lam * v;
                }
                // The eliminated column cancels by construction; the other entries
                // keep their exact (rounded) sums, zeros included only if exact.
                cur.remove(&j);
                cur.retain(|_, v| *v != 0.0);
                crhs += lam * b[r2];
                used.push(r2);
            }
            if let Some(c) = row_best {
                found.push(c);
            }
        }
    }
    found.sort_by(|a, b| b.efficacy.total_cmp(&a.efficacy));
    found.truncate(max_cuts);
    found
        .into_iter()
        .map(|c| {
            let mut coeffs = vec![0.0; n];
            for (j, v) in c.coef {
                coeffs[j] = -v;
            }
            GomoryCut {
                coeffs,
                rhs: -c.rhs,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! #2107 soundness: feasible-point sampling on random big-M fixed-charge models
    //! (every cut, separated at arbitrary points, must hold at every sampled
    //! integer-feasible point) and a differential solve (c-MIR on vs off, same
    //! certified optimum).
    use super::*;
    use crate::bnb::milp_driver::{solve_milp_hooked, MilpOptions, MilpStatus};
    use crate::lp::simplex::SimplexOptions;

    /// Deterministic LCG in [0, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// A fixed-charge model in standard form. Columns: `ny` binaries `y`, `ny`
    /// continuous `x` declared `[0, inf)`, then one slack per row. Rows (`<=`, slack
    /// in `[0, inf)`): `x_j - M_j y_j <= 0`; `-(sum_{j in S_r} x_j) <= -d_r`;
    /// `sum w_j y_j <= W`.
    struct Model {
        ny: usize,
        m: usize,
        n: usize,
        a: Vec<f64>, // dense row-major m x n
        b: Vec<f64>,
        c: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        is_int: Vec<bool>,
        big_m: Vec<f64>,
    }

    fn model(seed: u64, ny: usize, ndem: usize) -> Model {
        let mut r = Rng(seed);
        let ns = 2 * ny;
        let m = ny + ndem + 1;
        let n = ns + m;
        let mut a = vec![0.0; m * n];
        let mut b = vec![0.0; m];
        let big_m: Vec<f64> = (0..ny).map(|_| (2.0 + 8.0 * r.next()).round()).collect();
        for j in 0..ny {
            a[j * n + ny + j] = 1.0;
            a[j * n + j] = -big_m[j];
        }
        for k in 0..ndem {
            let row = ny + k;
            let mut tot = 0.0;
            for j in 0..ny {
                if r.next() < 0.6 {
                    a[row * n + ny + j] = -1.0;
                    tot += big_m[j];
                }
            }
            b[row] = -(0.15 + 0.5 * r.next()) * tot - 0.37; // fractional demand
        }
        let row = ny + ndem;
        let mut wsum = 0.0;
        for j in 0..ny {
            let w = (1.0 + 4.0 * r.next()).round();
            a[row * n + j] = w;
            wsum += w;
        }
        b[row] = (0.7 * wsum).floor() + 0.5;
        for i in 0..m {
            a[i * n + ns + i] = 1.0;
        }
        let mut c = vec![0.0; n];
        for j in 0..ny {
            c[j] = 3.0 + 10.0 * r.next();
            c[ny + j] = 0.5 + r.next();
        }
        let mut l = vec![0.0; n];
        let mut u = vec![INF; n];
        for v in u.iter_mut().take(ny) {
            *v = 1.0;
        }
        for v in l.iter_mut().skip(ns) {
            *v = 0.0;
        }
        let mut is_int = vec![false; n];
        for v in is_int.iter_mut().take(ny) {
            *v = true;
        }
        Model {
            ny,
            m,
            n,
            a,
            b,
            c,
            l,
            u,
            is_int,
            big_m,
        }
    }

    /// Full standard-form point (slacks from the rows) for structural `xs`.
    fn with_slacks(md: &Model, xs: &[f64]) -> Vec<f64> {
        let ns = 2 * md.ny;
        let mut z = xs.to_vec();
        z.resize(md.n, 0.0);
        for i in 0..md.m {
            let act: f64 = (0..ns).map(|j| md.a[i * md.n + j] * xs[j]).sum();
            z[ns + i] = md.b[i] - act;
        }
        z
    }

    fn feasible(md: &Model, z: &[f64]) -> bool {
        (0..md.n).all(|j| z[j] >= md.l[j] - 1e-12 && z[j] <= md.u[j] + 1e-12)
    }

    #[test]
    fn lifted_mic_cut_holds_at_every_integer_point() {
        // Brute force over random mixed-integer knapsacks `sum v_i y_i + sum w_c s_c <= b`
        // with general integers y_i in [0, u_i] (u_i <= 4) and continuous s_c >= 0 of
        // either sign. The cut keeps negative continuous coefficients unchanged and drops
        // positive ones, so its worst case at an integer point is the smallest feasible
        // continuous contribution: lhs = c.y - max(0, v.y - b) when some w_c < 0 (else
        // the point must satisfy v.y <= b on its own).
        let mut r = Rng(0x1d4c);
        let (mut cuts, mut checks) = (0usize, 0usize);
        for _ in 0..3000 {
            let ni = 2 + (r.next() * 4.0) as usize; // 2..5 integers
            let nc = (r.next() * 3.0) as usize; // 0..2 continuous
            let mut vals = Vec::new();
            let mut upper = Vec::new();
            let mut is_int = Vec::new();
            for _ in 0..ni {
                vals.push((1.0 + r.next() * 19.0).round() + if r.next() < 0.3 { 0.5 } else { 0.0 });
                upper.push((1.0 + r.next() * 3.99).floor());
                is_int.push(true);
            }
            let mut has_neg = false;
            for _ in 0..nc {
                let w = if r.next() < 0.7 {
                    -(1.0 + r.next() * 9.0)
                } else {
                    1.0 + r.next() * 5.0
                };
                has_neg |= w < 0.0;
                vals.push(w);
                upper.push(INF);
                is_int.push(false);
            }
            let cap: f64 = (0..ni).map(|i| vals[i] * upper[i]).sum();
            let b = (0.2 + 0.6 * r.next()) * cap;
            let solval: Vec<f64> = (0..vals.len())
                .map(|i| if is_int[i] { r.next() * upper[i] } else { 0.0 })
                .collect();
            let Some((c, crhs)) = lifted_mic_knapsack(&vals, &upper, &solval, &is_int, b) else {
                continue;
            };
            cuts += 1;
            // Enumerate every integer point of the box.
            let mut y = vec![0usize; ni];
            loop {
                let vy: f64 = (0..ni).map(|i| vals[i] * y[i] as f64).sum();
                let cy: f64 = (0..ni).map(|i| c[i] * y[i] as f64).sum();
                let feasible = has_neg || vy <= b + 1e-9;
                if feasible {
                    let lhs = if has_neg { cy - (vy - b).max(0.0) } else { cy };
                    checks += 1;
                    assert!(
                        lhs <= crhs + 1e-7 * (1.0 + crhs.abs()),
                        "cut {c:?} <= {crhs} violated at y={y:?} (lhs {lhs}); vals {vals:?} \
                         upper {upper:?} b {b}"
                    );
                }
                let mut i = 0;
                while i < ni {
                    y[i] += 1;
                    if y[i] as f64 <= upper[i] {
                        break;
                    }
                    y[i] = 0;
                    i += 1;
                }
                if i == ni {
                    break;
                }
            }
        }
        eprintln!("lifted MIC brute force: {cuts} cuts, {checks} point checks");
        assert!(cuts > 200, "only {cuts} lifted cuts produced");
        assert!(checks > 10_000, "only {checks} checks");
    }

    #[test]
    fn every_cut_holds_at_every_sampled_feasible_point() {
        let mut checks = 0usize;
        let mut cuts_seen = 0usize;
        for seed in 1..=40u64 {
            let md = model(seed, 5, 2);
            let csc = SparseCols::from_dense(&md.a, md.m, md.n);
            let mut r = Rng(seed ^ 0x9e37);
            // Integer-feasible points: every binary pattern, continuous parts at
            // random interior points, at their big-M caps, and at zero.
            let mut pts: Vec<Vec<f64>> = Vec::new();
            for mask in 0..(1u32 << md.ny) {
                for t in 0..30 {
                    let mut xs = vec![0.0; 2 * md.ny];
                    for j in 0..md.ny {
                        let on = (mask >> j) & 1 == 1;
                        xs[j] = if on { 1.0 } else { 0.0 };
                        let cap = if on { md.big_m[j] } else { 0.0 };
                        xs[md.ny + j] = match t {
                            0 => cap,
                            1 => 0.0,
                            _ => cap * r.next(),
                        };
                    }
                    let z = with_slacks(&md, &xs);
                    if feasible(&md, &z) {
                        pts.push(z);
                    }
                }
            }
            assert!(!pts.is_empty(), "seed {seed}: no feasible point sampled");
            // Separate at arbitrary fractional points: validity cannot depend on it.
            for _ in 0..25 {
                let mut xs = vec![0.0; 2 * md.ny];
                for j in 0..md.ny {
                    xs[j] = r.next();
                    xs[md.ny + j] = md.big_m[j] * (xs[j] + 0.3 * (r.next() - 0.5)).max(0.0);
                }
                let z = with_slacks(&md, &xs);
                let cuts = separate_cmir_csc(
                    &csc, md.n, md.m, &md.l, &md.u, &md.b, &z, &md.is_int, md.m, 2, 1000, 1e-9,
                );
                cuts_seen += cuts.len();
                for cut in &cuts {
                    for p in &pts {
                        let act: f64 = cut.coeffs.iter().zip(p).map(|(a, v)| a * v).sum();
                        checks += 1;
                        assert!(
                            act >= cut.rhs - 1e-9 * (1.0 + cut.rhs.abs()),
                            "seed {seed}: cut {:?} >= {} violated at feasible {p:?} ({act})",
                            cut.coeffs,
                            cut.rhs
                        );
                    }
                }
            }
        }
        // CLAUDE.md section 6: the probe fired.
        eprintln!("cmir sampling: {cuts_seen} cuts, {checks} point checks");
        assert!(cuts_seen > 100, "only {cuts_seen} cuts separated");
        assert!(checks > 10_000, "only {checks} checks");
    }

    #[test]
    fn variable_bounds_are_read_from_big_m_rows() {
        let md = model(7, 4, 1);
        let rows: Vec<Vec<(usize, f64)>> = (0..md.m)
            .map(|i| {
                (0..md.n)
                    .filter(|&j| md.a[i * md.n + j] != 0.0)
                    .map(|j| (j, md.a[i * md.n + j]))
                    .collect()
            })
            .collect();
        let (_, vub) = variable_bounds(&rows, &md.b, &md.l, &md.u, &md.is_int, md.n);
        for j in 0..md.ny {
            let x = md.ny + j;
            assert!(
                vub[x]
                    .iter()
                    .any(|vb| vb.k == j && (vb.m - md.big_m[j]).abs() < 1e-12 && vb.beta == 0.0),
                "x{x} <= {} y{j} not found: {:?}",
                md.big_m[j],
                vub[x]
            );
        }
    }

    fn opts(ns: usize, ints: Vec<usize>, cmir: bool) -> MilpOptions {
        MilpOptions {
            n_struct: ns,
            integer_cols: ints,
            max_nodes: 200_000,
            max_open_nodes: None,
            time_limit_s: None,
            gap_tol: 1e-9,
            abs_gap_tol: None,
            root_cuts: 200,
            cut_rounds: 10,
            gmi_cuts: true,
            cmir_cuts: cmir,
            cut_select: true,
            node_cuts: false,
            max_pool_cuts: 128,
            heuristics: true,
            presolve: true,
            strong_branch: true,
            node_propagation: true,
            reduced_cost_fixing: true,
            sb_max_cands: 6,
            sb_node_budget: 48,
            initial_incumbent: None,
            node_hook_rounds: 0,
            node_hook_cut_cap: 0,
            root_cut_time_s: None,
            root_cut_prune: true,
            simplex: SimplexOptions::default(),
        }
    }

    #[test]
    fn cmir_on_and_off_certify_the_same_optimum() {
        let _guard = crate::profile::test_guard();
        crate::profile::set_enabled(true);
        crate::profile::reset();
        let mut pairs = Vec::new();
        for seed in 1..=30u64 {
            let md = model(seed, 8, 3);
            let ns = 2 * md.ny;
            let solve = |cmir: bool| {
                solve_milp_hooked(
                    SparseCols::from_dense(&md.a, md.m, md.n),
                    md.m,
                    md.n,
                    &md.c,
                    &md.l,
                    &md.u,
                    &md.b,
                    0.0,
                    &opts(ns, (0..md.ny).collect(), cmir),
                    None,
                )
            };
            pairs.push((seed, solve(false), solve(true)));
        }
        let fired = crate::profile::counter(crate::profile::Ctr::SepCmirCuts);
        crate::profile::set_enabled(false);
        let mut compared = 0usize;
        for (seed, off, on) in &pairs {
            if off.status != MilpStatus::Optimal {
                continue; // infeasible draw
            }
            assert_eq!(on.status, MilpStatus::Optimal, "seed {seed}: {on:?}");
            let tol = 1e-6 * (1.0 + off.obj.abs());
            assert!(
                (on.obj - off.obj).abs() <= tol,
                "seed {seed}: on {} off {}",
                on.obj,
                off.obj
            );
            assert!(
                on.bound <= off.obj + tol,
                "seed {seed}: bound {} > optimum {}",
                on.bound,
                off.obj
            );
            compared += 1;
        }
        eprintln!("cmir differential: {compared} instances, {fired} c-MIR candidates");
        // CLAUDE.md section 6: compared enough, and the ON arm really separated cuts.
        assert!(compared >= 15, "only {compared} instances compared");
        assert!(fired > 0, "c-MIR never separated a cut in the ON arm");
    }
}
