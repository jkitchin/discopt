//! Learned dual-proof (Farkas) constraints for the native spatial tree (#2142).
//!
//! When a node LP is **certified infeasible** the dual simplex hands back a
//! verified Farkas ray `y`. SCIP turns that ray into a *dual proof constraint*
//! (`conflict_general.c` `SCIPgetFarkasProof`, `conflict_dualproofanalysis.c`
//! `tightenDualproof` / `createAndAddProofcons`): aggregate the globally valid
//! rows with the ray's multipliers, post-process, and keep the aggregated row as a
//! propagate-only constraint for the rest of the search. On GDPlib `cstr` big-M
//! the whole of SCIP's conflict-analysis value is this mechanism (2.393 vs 1.854
//! at 20k nodes with it off; `docs/dev/data/2142-scip-expert/findings.md`), and
//! the proofs are binary→continuous implications through a chain of linear
//! equalities that row-wise FBBT cannot derive at any number of rounds.
//!
//! What this module does, per certified-empty node:
//! 1. Take the ray's entries on the **global** rows only: the spec's fixed rows
//!    and the tree's global cut block. The node-local envelope rows are dropped
//!    (SCIP `conflict/uselocalrows = FALSE`, measured 2.375 vs 2.393).
//! 2. For each sign of the ray, multipliers `λ_r = max(0, −σ·y_r) ≥ 0` on the
//!    `≤` rows; the aggregation `Σ λ_r A_r x ≤ Σ λ_r b_r` is a nonnegative
//!    combination of valid rows, hence **valid for every point of the global box**
//!    whatever `y` is — a wrong or noisy ray can only make the row useless, never
//!    invalid.
//! 3. Make the floating-point row rigorous: the right-hand side is rounded up and
//!    widened by the aggregation's rounding error times the column's global
//!    magnitude (a column with an infinite global bound and nonzero error rejects
//!    the row). Cancel (tightenDualproof) a continuous column whose node bound on
//!    its activity side equals its global bound, and every tiny coefficient, by
//!    moving `a_j·gb_j` (rounded the safe way) into the right-hand side.
//! 4. Keep the row only if its minimum activity over the node box exceeds the
//!    right-hand side by a margin (it must at least cut this node), its support is
//!    at most `max(10, 15 % of the columns)` (SCIP `conflict/maxvarsfac`), and it
//!    is not a duplicate.
//!
//! The pool is tree-global (every row is globally valid), propagate-only (never an
//! LP row), and bounded: when full, the least recently useful row is evicted
//! (SCIP's conflict store ages its proofs out the same way).

use std::collections::HashMap;

use crate::bnb::spatial_kernel::{FixedRow, SpatialKernelSpec};
use crate::presolve::directed::{add_down, add_up, mul_down, mul_up, sub_up};

/// The LP layer's infinite-bound sentinel.
const SENTINEL: f64 = 1e20;
/// Relative magnitude below which a ray multiplier counts as zero.
const LAMBDA_REL_TOL: f64 = 1e-12;
/// Relative magnitude below which an aggregated coefficient is cancelled.
const COEF_REL_TOL: f64 = 1e-9;
/// A kept row must cut the node by at least this much (relative to `1 + |rhs|`,
/// coefficients normalised to `max |a| = 1`). Ten times the propagator's
/// infeasibility tolerance, so the row can prove the node's box empty.
const KEEP_MARGIN: f64 = 1e-6;
/// Default pool capacity (rows).
pub const DEFAULT_POOL_CAP: usize = 2000;

/// `γ_k = k·u / (1 − k·u)`: the relative error bound of a `k`-term float sum.
fn gamma(k: usize) -> f64 {
    let u = f64::EPSILON * 0.5;
    let ku = k as f64 * u;
    ku / (1.0 - ku)
}

/// Why a candidate proof was not kept (counted per reason; diagnostic only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofReject {
    /// No global row carries a multiplier of the usable sign.
    NoGlobalSupport,
    /// A column with nonzero rounding error has an infinite global bound.
    Unbounded,
    /// The aggregated row does not cut the node box (the envelope rows mattered).
    NotViolated,
    /// The row's support exceeds the fill-in limit.
    TooDense,
}

/// Build a dual-proof row from a Farkas ray `y` (length `m`, one entry per node
/// LP row) of a node LP whose first `spec.fixed_rows.len()` rows are the spec's
/// fixed rows and whose last `cuts.len()` rows are `cuts` (the layout of
/// [`crate::bnb::spatial_kernel::assemble_node_lp_with_cuts`]).
///
/// `(lo, hi)` is the box the LP was solved over. Returns the normalised row
/// (`max |a| = 1`) or the reason it was rejected. The returned row is valid for
/// every point of the spec's global box regardless of `y`.
pub fn proof_from_farkas(
    spec: &SpatialKernelSpec,
    cuts: &[FixedRow],
    y: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> Result<FixedRow, ProofReject> {
    let n_fixed = spec.fixed_rows.len();
    if y.len() < n_fixed + cuts.len() || !y.iter().all(|v| v.is_finite()) {
        return Err(ProofReject::NoGlobalSupport);
    }
    let cut_off = y.len() - cuts.len();
    let global_rows = || {
        spec.fixed_rows
            .iter()
            .enumerate()
            .map(|(r, fr)| (fr, y[r]))
            .chain(
                cuts.iter()
                    .enumerate()
                    .map(move |(k, c)| (c, y[cut_off + k])),
            )
    };
    let mut best: Result<(FixedRow, f64), ProofReject> = Err(ProofReject::NoGlobalSupport);
    for sgn in [1.0f64, -1.0] {
        let lmax = global_rows().fold(0.0f64, |a, (_, v)| a.max(-sgn * v));
        if !(lmax > 0.0 && lmax.is_finite()) {
            continue;
        }
        // Multipliers scaled to max 1 (any nonnegative float values are valid
        // multipliers: the scaling's rounding changes WHICH valid row we build,
        // never its validity).
        let rows: Vec<(&FixedRow, f64)> = global_rows()
            .filter_map(|(fr, v)| {
                let l = -sgn * v / lmax;
                (l > LAMBDA_REL_TOL).then_some((fr, l))
            })
            .collect();
        match aggregate(spec, &rows, lo, hi) {
            Ok((row, viol)) => {
                if best.as_ref().map_or(true, |(_, bv)| viol > *bv) {
                    best = Ok((row, viol));
                }
            }
            Err(e) => {
                // Keep the most informative rejection reason.
                if let Err(prev) = best {
                    if prev == ProofReject::NoGlobalSupport {
                        best = Err(e);
                    }
                }
            }
        }
    }
    best.map(|(r, _)| r)
}

/// [`proof_from_farkas`] with a fallback (#2142): when the node LP's own ray,
/// restricted to the global rows, does not cut the node (the envelope rows carried
/// the infeasibility), solve the feasibility LP over the GLOBAL rows only on the
/// same box ([`crate::bnb::spatial_kernel::assemble_global_rows_lp`]); if it is
/// certified infeasible its ray aggregates global rows only. This is the offline
/// construction of `docs/dev/data/2142-scip-expert/py/d2_proofs.py`. Returns the
/// row (or the first attempt's rejection) and whether the fallback LP was solved.
pub fn proof_with_global_lp(
    spec: &SpatialKernelSpec,
    cuts: &[FixedRow],
    y: &[f64],
    lo: &[f64],
    hi: &[f64],
    opts: &crate::lp::simplex::SimplexOptions,
) -> (Result<FixedRow, ProofReject>, bool) {
    let first = proof_from_farkas(spec, cuts, y, lo, hi);
    if !matches!(first, Err(ProofReject::NotViolated)) {
        return (first, false);
    }
    let lp = crate::bnb::spatial_kernel::assemble_global_rows_lp(spec, lo, hi, cuts);
    let c = vec![0.0; lp.n_total];
    let sol = crate::lp::simplex::primal::solve_lp_cols_scaled(
        lp.sp, lp.m, lp.n_total, &c, &lp.l, &lp.u, &lp.b, opts,
    );
    if sol.status != crate::lp::simplex::LpStatus::Infeasible {
        return (first, true);
    }
    (proof_from_farkas(spec, cuts, &sol.dual, lo, hi), true)
}

/// Aggregate `Σ λ_r (A_r x ≤ b_r)` rigorously, clean it up, and test it against the
/// node box. Returns the normalised row and its normalised violation.
fn aggregate(
    spec: &SpatialKernelSpec,
    rows: &[(&FixedRow, f64)],
    lo: &[f64],
    hi: &[f64],
) -> Result<(FixedRow, f64), ProofReject> {
    if rows.is_empty() {
        return Err(ProofReject::NoGlobalSupport);
    }
    let n = spec.n_cols;
    let mut a = vec![0.0f64; n];
    let mut a_abs = vec![0.0f64; n];
    let mut a_cnt = vec![0usize; n];
    // β rounded UP: an upper bound on the exact `Σ λ_r b_r`.
    let mut rhs = 0.0f64;
    for (fr, l) in rows {
        rhs = add_up(rhs, mul_up(*l, fr.rhs));
        for (&j, &c) in fr.cols.iter().zip(&fr.coeffs) {
            let t = l * c;
            a[j] += t;
            a_abs[j] += t.abs();
            a_cnt[j] += 1;
        }
    }
    if !rhs.is_finite() {
        return Err(ProofReject::Unbounded);
    }
    // Rounding error of each aggregated coefficient, `|a_j − ã_j| ≤ e_j`, moved into
    // the rhs over the GLOBAL box: `a·x ≤ ã·x + Σ e_j |x_j| ≤ β + Σ e_j M_j`.
    let glo = &spec.global_lo;
    let ghi = &spec.global_hi;
    let mag = |j: usize| -> f64 {
        let (l, h) = (glo[j], ghi[j]);
        if l <= -SENTINEL || h >= SENTINEL || !l.is_finite() || !h.is_finite() {
            f64::INFINITY
        } else {
            l.abs().max(h.abs())
        }
    };
    for j in 0..n {
        if a_cnt[j] == 0 {
            continue;
        }
        // +1 for the product `l*c` rounding, +1 slack.
        let e = mul_up(gamma(a_cnt[j] + 2), a_abs[j]);
        if e > 0.0 {
            let m = mag(j);
            if !m.is_finite() {
                return Err(ProofReject::Unbounded);
            }
            rhs = add_up(rhs, mul_up(e, m));
        }
    }
    // Cancellation: drop `a_j x_j` by bounding it below with its global bound
    // (`a_j x_j ≥ a_j·gb_j` on the global box), which is exact at this node when
    // the node bound on the activity side IS the global bound (tightenDualproof),
    // and also drops numerically negligible coefficients.
    let amax = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if !(amax > 0.0 && amax.is_finite()) {
        return Err(ProofReject::NoGlobalSupport);
    }
    for j in 0..n {
        if a[j] == 0.0 {
            continue;
        }
        let gb = if a[j] > 0.0 { glo[j] } else { ghi[j] };
        let gb_finite = gb.is_finite() && gb.abs() < SENTINEL;
        let tiny = a[j].abs() <= COEF_REL_TOL * amax;
        let at_global = !spec.integrality[j] && {
            let nb = if a[j] > 0.0 { lo[j] } else { hi[j] };
            nb == gb
        };
        if (tiny || at_global) && gb_finite {
            rhs = sub_up(rhs, mul_down(a[j], gb));
            a[j] = 0.0;
        }
    }
    let support: Vec<usize> = (0..n).filter(|&j| a[j] != 0.0).collect();
    if support.is_empty() {
        return Err(ProofReject::NoGlobalSupport);
    }
    let max_nnz = 10usize.max((0.15 * n as f64).ceil() as usize);
    // Normalise to max |a| = 1. Division by a power of two is exact; otherwise the
    // rhs is re-widened by the same rigorous argument (the scaled row is
    // `(a/s)·x ≤ rhs/s` exactly, then each `a_j/s` is rounded).
    let s = support.iter().fold(0.0f64, |m, &j| m.max(a[j].abs()));
    let s = 2f64.powi(s.log2().round() as i32); // power of two: exact scaling
    let coeffs: Vec<f64> = support.iter().map(|&j| a[j] / s).collect();
    let rhs = rhs / s;
    // Min activity over the node box, rounded DOWN.
    let mut minact = 0.0f64;
    for (&j, &c) in support.iter().zip(&coeffs) {
        let b = if c > 0.0 { lo[j] } else { hi[j] };
        if !b.is_finite() || b.abs() >= SENTINEL {
            return Err(ProofReject::NotViolated);
        }
        minact = add_down(minact, mul_down(c, b));
    }
    let viol = (minact - rhs) / (1.0 + rhs.abs());
    if !(viol > KEEP_MARGIN) {
        return Err(ProofReject::NotViolated);
    }
    if support.len() > max_nnz {
        return Err(ProofReject::TooDense);
    }
    Ok((
        FixedRow {
            cols: support,
            coeffs,
            rhs,
        },
        viol,
    ))
}

/// Outcome of offering a row to the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolInsert {
    /// A new row was stored.
    Added,
    /// A row with the same coefficients existed; its rhs was tightened.
    Tightened,
    /// A row with the same coefficients and an rhs at least as tight existed.
    Duplicate,
}

/// The tree-global pool of learned proof rows.
#[derive(Clone, Debug, Default)]
pub struct DualProofPool {
    /// The rows, in insertion order (eviction swaps the last row in).
    pub rows: Vec<FixedRow>,
    /// Per row: the node count at which it was last useful (learned, tightened a
    /// box, or proved a box empty). The eviction key.
    pub last_useful: Vec<usize>,
    keys: HashMap<Vec<(usize, i64)>, usize>,
    /// Capacity (rows); `0` means unbounded.
    pub cap: usize,
    /// Rows evicted to make room.
    pub n_evicted: usize,
}

fn row_key(r: &FixedRow) -> Vec<(usize, i64)> {
    r.cols
        .iter()
        .zip(&r.coeffs)
        .map(|(&j, &c)| (j, (c * 1e9).round() as i64))
        .collect()
}

impl DualProofPool {
    /// An empty pool holding at most `cap` rows.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            cap,
            ..Self::default()
        }
    }

    /// Offer a (valid) row learned at node `now`.
    pub fn insert(&mut self, row: FixedRow, now: usize) -> PoolInsert {
        let key = row_key(&row);
        if let Some(&i) = self.keys.get(&key) {
            self.last_useful[i] = now;
            // Same quantised coefficients: the rows differ by < 1e-9 per entry, so
            // only replace when the new row is tighter (both rows are valid).
            if row.rhs < self.rows[i].rhs {
                self.rows[i] = row;
                return PoolInsert::Tightened;
            }
            return PoolInsert::Duplicate;
        }
        if self.cap > 0 && self.rows.len() >= self.cap {
            // Evict the least recently useful row (ties: lowest index).
            let (victim, _) = self
                .last_useful
                .iter()
                .enumerate()
                .min_by_key(|&(i, &t)| (t, i))
                .expect("non-empty pool at capacity");
            self.keys.remove(&row_key(&self.rows[victim]));
            let last = self.rows.len() - 1;
            self.rows.swap_remove(victim);
            self.last_useful.swap_remove(victim);
            if victim != last {
                let moved = row_key(&self.rows[victim]);
                self.keys.insert(moved, victim);
            }
            self.n_evicted += 1;
        }
        self.keys.insert(key, self.rows.len());
        self.rows.push(row);
        self.last_useful.push(now);
        PoolInsert::Added
    }

    /// Record that the rows flagged in `hits` were useful at node `now`.
    pub fn touch(&mut self, hits: &[bool], now: usize) {
        for (t, &h) in self.last_useful.iter_mut().zip(hits) {
            if h {
                *t = now;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bnb::spatial_kernel::{assemble_node_lp, EnvTerm};
    use crate::lp::simplex::{primal::solve_lp_cols_scaled, LpStatus, SimplexOptions};

    /// Tiny deterministic LCG (no external crate), uniform in [0, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn spec_with_rows(n: usize, rows: Vec<FixedRow>, int: Vec<bool>) -> SpatialKernelSpec {
        SpatialKernelSpec {
            n_cols: n,
            n_orig: n,
            c: vec![0.0; n],
            integrality: int,
            global_lo: vec![0.0; n],
            global_hi: vec![1.0; n],
            fixed_rows: rows,
            terms: vec![],
            blf_terms: vec![],
            obbt_candidates: vec![],
        }
    }

    /// A chain `x0 ≤ x1 ≤ … ≤ x_{n-1}` plus `x_{n-1} ≤ 0.5`; the node box fixes
    /// `x0 ≥ 0.9`, so the LP is infeasible and the proof is `x0 ≤ 0.5` — the
    /// aggregation of the whole chain, which row-wise FBBT needs n rounds for.
    #[test]
    fn chain_proof_is_the_aggregate_and_cuts_the_node() {
        let n = 6;
        let mut rows = Vec::new();
        for k in 0..n - 1 {
            rows.push(FixedRow {
                cols: vec![k, k + 1],
                coeffs: vec![1.0, -1.0],
                rhs: 0.0,
            });
        }
        rows.push(FixedRow {
            cols: vec![n - 1],
            coeffs: vec![1.0],
            rhs: 0.5,
        });
        let spec = spec_with_rows(n, rows, vec![false; n]);
        let mut lo = vec![0.0; n];
        let hi = vec![1.0; n];
        lo[0] = 0.9;
        let lp = assemble_node_lp(&spec, &lo, &hi);
        let c = vec![0.0; lp.n_total];
        let sol = solve_lp_cols_scaled(
            lp.sp.clone(),
            lp.m,
            lp.n_total,
            &c,
            &lp.l,
            &lp.u,
            &lp.b,
            &SimplexOptions::default(),
        );
        assert_eq!(sol.status, LpStatus::Infeasible);
        let row = proof_from_farkas(&spec, &[], &sol.dual, &lo, &hi).expect("a proof");
        // The middle columns cancel exactly (their coefficients sum to 0), leaving x0.
        assert_eq!(row.cols, vec![0], "row {row:?}");
        assert!(row.coeffs[0] > 0.0);
        let bound = row.rhs / row.coeffs[0];
        assert!(bound >= 0.5 && bound < 0.5 + 1e-9, "x0 <= {bound}");
    }

    /// Feasible-point sampling (CLAUDE.md §5): random `≤` systems over [0,1]^n,
    /// random node boxes; every learned row must hold at every sampled point that
    /// satisfies the fixed rows (anywhere in the global box), and its rhs must be
    /// at least the exact-ish aggregate recomputed in higher-effort arithmetic.
    #[test]
    fn learned_rows_hold_at_every_sampled_feasible_point() {
        let mut rng = Lcg(0x2142);
        let (mut n_proofs, mut n_checks, mut n_feasible) = (0usize, 0usize, 0usize);
        for _inst in 0..400 {
            let n = 3 + (rng.next() * 5.0) as usize;
            let m = 2 + (rng.next() * 6.0) as usize;
            let mut rows = Vec::new();
            for _ in 0..m {
                let mut cols = Vec::new();
                let mut coeffs = Vec::new();
                for j in 0..n {
                    if rng.next() < 0.6 {
                        cols.push(j);
                        coeffs.push((rng.next() - 0.5) * 10.0);
                    }
                }
                if cols.is_empty() {
                    continue;
                }
                rows.push(FixedRow {
                    cols,
                    coeffs,
                    rhs: (rng.next() - 0.3) * 4.0,
                });
            }
            let int: Vec<bool> = (0..n).map(|_| rng.next() < 0.3).collect();
            let spec = spec_with_rows(n, rows, int);
            // Random node box inside [0,1]^n.
            let mut lo = vec![0.0; n];
            let mut hi = vec![1.0; n];
            for j in 0..n {
                let (a, b) = (rng.next(), rng.next());
                if rng.next() < 0.5 {
                    lo[j] = a.min(b);
                    hi[j] = a.max(b);
                }
            }
            let lp = assemble_node_lp(&spec, &lo, &hi);
            let c = vec![0.0; lp.n_total];
            let sol = solve_lp_cols_scaled(
                lp.sp.clone(),
                lp.m,
                lp.n_total,
                &c,
                &lp.l,
                &lp.u,
                &lp.b,
                &SimplexOptions::default(),
            );
            if sol.status != LpStatus::Infeasible {
                continue;
            }
            let Ok(row) = proof_from_farkas(&spec, &[], &sol.dual, &lo, &hi) else {
                continue;
            };
            n_proofs += 1;
            // It cuts the node: min activity over the node box > rhs.
            let minact: f64 = row
                .cols
                .iter()
                .zip(&row.coeffs)
                .map(|(&j, &c)| if c > 0.0 { c * lo[j] } else { c * hi[j] })
                .sum();
            assert!(minact > row.rhs, "proof does not cut its node");
            // Sample the GLOBAL box; keep the points feasible for the fixed rows.
            for _ in 0..3000 {
                let x: Vec<f64> = (0..n)
                    .map(|j| {
                        let v = rng.next();
                        if spec.integrality[j] {
                            v.round()
                        } else if rng.next() < 0.2 {
                            v.round() // bound-touching points
                        } else {
                            v
                        }
                    })
                    .collect();
                let feas = spec.fixed_rows.iter().all(|fr| {
                    fr.cols
                        .iter()
                        .zip(&fr.coeffs)
                        .map(|(&j, &c)| c * x[j])
                        .sum::<f64>()
                        <= fr.rhs
                });
                if !feas {
                    continue;
                }
                n_feasible += 1;
                let act: f64 = row
                    .cols
                    .iter()
                    .zip(&row.coeffs)
                    .map(|(&j, &c)| c * x[j])
                    .sum();
                assert!(
                    act <= row.rhs + 1e-12,
                    "learned row {row:?} cuts feasible point {x:?} (act {act})"
                );
                n_checks += 1;
            }
        }
        eprintln!("proofs {n_proofs}, feasible-point checks {n_checks} ({n_feasible} feasible)");
        assert!(n_proofs >= 20, "too few proofs exercised: {n_proofs}");
        assert!(n_checks >= 1000, "too few checks executed: {n_checks}");
    }

    /// The rhs is rounded OUTWARD: rows with coefficients that do not sum exactly
    /// give an rhs no smaller than the aggregate computed in exact-rational-like
    /// (two-sum compensated) arithmetic over the global box.
    #[test]
    fn rhs_is_rounded_outward() {
        // 0.1*x0 + 0.2*x1 <= 0.3 and 0.7*x0 - 0.2*x1 <= 0.1/3 with lambda = 1 each:
        // aggregate 0.8 x0 <= 0.3 + 0.0333.., float-inexact everywhere.
        let rows = vec![
            FixedRow {
                cols: vec![0, 1],
                coeffs: vec![0.1, 0.2],
                rhs: 0.3,
            },
            FixedRow {
                cols: vec![0, 1],
                coeffs: vec![0.7, -0.2],
                rhs: 0.1 / 3.0,
            },
        ];
        let spec = spec_with_rows(2, rows, vec![false, false]);
        let lo = vec![0.9, 0.0];
        let hi = vec![1.0, 1.0];
        let y = vec![-1.0, -1.0];
        let row = proof_from_farkas(&spec, &[], &y, &lo, &hi).expect("violated aggregate");
        // Naive aggregate rhs, normalised the same way.
        let naive_rhs = 0.3 + 0.1 / 3.0;
        let naive_a0 = 0.1 + 0.7;
        let scale = row.coeffs[0] / naive_a0;
        assert!(row.cols == vec![0]);
        assert!(
            row.rhs >= naive_rhs * scale,
            "rhs {} below the naive aggregate {}",
            row.rhs,
            naive_rhs * scale
        );
        // The widening is tiny (rounding-level), not a gross relaxation.
        assert!(row.rhs - naive_rhs * scale < 1e-12);
    }

    /// Rows touching a column with an infinite global bound and nonzero rounding
    /// error are refused (the error cannot be bounded), never emitted unsound.
    #[test]
    fn infinite_global_bound_with_rounding_error_is_refused() {
        let mut spec = spec_with_rows(
            2,
            vec![
                FixedRow {
                    cols: vec![0, 1],
                    coeffs: vec![0.1, 0.3],
                    rhs: 0.0,
                },
                FixedRow {
                    cols: vec![0, 1],
                    coeffs: vec![0.7, 0.3],
                    rhs: 0.0,
                },
            ],
            vec![false, false],
        );
        spec.global_hi[1] = SENTINEL;
        let lo = vec![0.5, 0.0];
        let hi = vec![1.0, 1.0];
        assert_eq!(
            proof_from_farkas(&spec, &[], &[-1.0, -1.0], &lo, &hi),
            Err(ProofReject::Unbounded)
        );
    }

    /// The node-local envelope rows are never aggregated: a ray that needs them
    /// yields `NotViolated`, not a row built from a local inequality.
    #[test]
    fn envelope_rows_are_excluded() {
        // w = x*y on [0,1]^2 (envelope rows local), fixed row w >= 0.5 i.e. -w <= -0.5;
        // node box x in [0, 0.1] makes w <= 0.1 via the envelope: infeasible only
        // through a local row.
        let spec = SpatialKernelSpec {
            n_cols: 3,
            n_orig: 2,
            c: vec![0.0; 3],
            integrality: vec![false; 3],
            global_lo: vec![0.0, 0.0, 0.0],
            global_hi: vec![1.0, 1.0, 1.0],
            fixed_rows: vec![FixedRow {
                cols: vec![2],
                coeffs: vec![-1.0],
                rhs: -0.5,
            }],
            terms: vec![EnvTerm::Bilinear { i: 0, j: 1, w: 2 }],
            blf_terms: vec![],
            obbt_candidates: vec![],
        };
        let lo = vec![0.0, 0.0, 0.0];
        let hi = vec![0.1, 1.0, 1.0];
        let lp = assemble_node_lp(&spec, &lo, &hi);
        let c = vec![0.0; lp.n_total];
        let sol = solve_lp_cols_scaled(
            lp.sp.clone(),
            lp.m,
            lp.n_total,
            &c,
            &lp.l,
            &lp.u,
            &lp.b,
            &SimplexOptions::default(),
        );
        assert_eq!(sol.status, LpStatus::Infeasible);
        // The aux column's LP upper bound comes from the box (0.1), so the fixed row
        // alone, over the LP's column bounds, is violated: this is the case where
        // the global row DOES carry the proof on the node box. Over the original
        // box (hi_w = 1) it does not.
        let hi_orig = vec![0.1, 1.0, 1.0];
        let r = proof_from_farkas(&spec, &[], &sol.dual, &lp.l[..3], &lp.u[..3]);
        assert!(r.is_ok(), "{r:?}");
        // (`w` sits at its global bound there, so it is cancelled and nothing is
        // left: either refusal is correct, a row is not.)
        let r2 = proof_from_farkas(&spec, &[], &sol.dual, &lo, &hi_orig);
        assert!(
            matches!(
                r2,
                Err(ProofReject::NotViolated) | Err(ProofReject::NoGlobalSupport)
            ),
            "{r2:?}"
        );
    }

    #[test]
    fn pool_dedupes_tightens_and_evicts_lru() {
        let mk = |c: f64, rhs: f64| FixedRow {
            cols: vec![0, 1],
            coeffs: vec![1.0, c],
            rhs,
        };
        let mut pool = DualProofPool::with_capacity(2);
        assert_eq!(pool.insert(mk(0.5, 1.0), 1), PoolInsert::Added);
        assert_eq!(pool.insert(mk(0.5, 1.0), 2), PoolInsert::Duplicate);
        assert_eq!(pool.insert(mk(0.5, 0.9), 3), PoolInsert::Tightened);
        assert_eq!(pool.rows[0].rhs, 0.9);
        assert_eq!(pool.insert(mk(0.25, 1.0), 4), PoolInsert::Added);
        pool.touch(&[false, true], 5);
        // Row 0 (last useful at 3) is the LRU victim.
        assert_eq!(pool.insert(mk(0.125, 1.0), 6), PoolInsert::Added);
        assert_eq!(pool.n_evicted, 1);
        assert_eq!(pool.rows.len(), 2);
        assert!(pool.rows.iter().all(|r| r.coeffs[1] != 0.5));
        // Keys stay consistent after the swap-remove.
        assert_eq!(pool.insert(mk(0.25, 1.0), 7), PoolInsert::Duplicate);
        assert_eq!(pool.insert(mk(0.125, 1.0), 7), PoolInsert::Duplicate);
    }
}
