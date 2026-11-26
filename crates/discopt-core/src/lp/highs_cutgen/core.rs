//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! Faithful port of the single-row core of `mip/HighsCutGeneration.cpp`: everything
//! `generateCut` does between `HighsTransformedLp::transform` and `untransform`
//! (`preprocessBaseInequality`, `determineCover`, the three lifted-cover functions,
//! `cmirCutGenerationHeuristic`, `tryGenerateCut`, complementation), plus
//! `postprocessCut`, which runs on the untransformed cut.
//!
//! Port contract: same operations in the same order on the same types (`CDouble`
//! where HiGHS uses `HighsCDouble`, the same sort/partition algorithms, the same
//! random draws), so that a cut replayed from a HiGHS dump comes out bit-identical.
//! The replay tests at the bottom of this file enforce that against records dumped
//! from HiGHS itself. Not ported: the `lpSol = false` cover order, which is used
//! only by conflict generation and reads the node queue.
//!
//! The C++ works through raw pointers into the caller's buffers and swaps them;
//! here each buffer is an owned `Vec` truncated to `rowlen`, which is equivalent
//! because HiGHS never reads past `rowlen`.

use super::cdouble::CDouble;
use super::domain::OpenNodeCounts;
use super::integers::{frexp_exp, integral_scale, is_integral, nearest_integer};
use super::random::{hash_pair_i32, HighsRandom};
use super::stdalgo::{
    partial_sort, partition, pdqsort, pdqsort_branchless, pdqsort_f64, upper_bound,
};

/// `kHighsTiny`.
pub const K_HIGHS_TINY: f64 = 1e-14;
/// `kHighsInf`.
pub const K_HIGHS_INF: f64 = f64::INFINITY;

/// A base inequality `sum vals[i] x[inds[i]] <= rhs` in the transformed space
/// (every variable shifted to a zero lower bound), as `generateCut` holds it right
/// after `transform`.
#[derive(Clone, Debug)]
pub struct BaseRow {
    /// Column (or implicit slack) index per position.
    pub inds: Vec<i32>,
    /// Coefficient per position.
    pub vals: Vec<f64>,
    /// Right-hand side.
    pub rhs: f64,
    /// Transformed upper bound per position (`kHighsInf` when unbounded).
    pub upper: Vec<f64>,
    /// Transformed LP solution value per position.
    pub solval: Vec<f64>,
    /// `lpRelaxation.isColIntegral(inds[i])` per position.
    pub col_integral: Vec<bool>,
    /// The `intsPositive` flag `transform` returned.
    pub ints_positive: bool,
}

/// What `generateCut` holds just before `untransform`.
#[derive(Clone, Debug)]
pub struct CoreCut {
    /// Column index per position.
    pub inds: Vec<i32>,
    /// Coefficient per position.
    pub vals: Vec<f64>,
    /// Right-hand side, compensated.
    pub rhs: CDouble,
    /// Every variable in the support is integral.
    pub integral_support: bool,
    /// Every coefficient (and the rhs) is integral.
    pub integral_coefficients: bool,
}

/// Solver context for one `HighsCutGeneration` object.
#[derive(Clone, Copy, Debug)]
pub struct CutGenParams {
    /// `mipdata_->feastol`.
    pub feastol: f64,
    /// `mipdata_->epsilon`.
    pub epsilon: f64,
    /// `lpRelaxation.numCols()`, which sizes the base-row length limit.
    pub num_cols: usize,
}

/// Which generator produced the last core cut (for coverage accounting).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CutPath {
    /// No cut.
    #[default]
    None,
    /// c-MIR, forced because the row has an unbounded integer.
    CmirUnboundedInt,
    /// c-MIR, which beat (or replaced) the lifted cover cut.
    Cmir,
    /// Lifted knapsack cover (pure binary row).
    KnapsackCover,
    /// Lifted mixed-binary cover.
    MixedBinaryCover,
    /// Lifted mixed-integer cover.
    MixedIntegerCover,
}

/// One `HighsCutGeneration` object: its random generator and work buffers.
pub struct CutGenCore {
    p: CutGenParams,
    path: CutPath,
    rng: HighsRandom,
    cover: Vec<usize>,
    coverweight: CDouble,
    lambda: CDouble,
    upper: Vec<f64>,
    solval: Vec<f64>,
    complementation: Vec<u8>,
    isintegral: Vec<u8>,
    inds: Vec<i32>,
    vals: Vec<f64>,
    rhs: CDouble,
    integral_support: bool,
    integral_coefficients: bool,
    rowlen: usize,
    initial_scale: f64,
    integerinds: Vec<usize>,
    deltas: Vec<f64>,
    /// Step budget of the mixed-integer cover lifting (`MAX_LIFT_WORK`; tests lower
    /// it to drive the refusal branch).
    lift_work_limit: u64,
}

/// `2^52`: every `f64` at or above it in magnitude is already an integer.
const TWO_POW_52: f64 = 4_503_599_627_370_496.0;

/// Magnitude bound under which an integer-valued `f64` converts to `i64` exactly and
/// leaves room for the `+-1` steps the lifting loops take (#2232).
const EXACT_I64_BOUND: f64 = 4_611_686_018_427_387_904.0; // 2^62

/// `floor(x)` through an integer cast where the cast is exact, `f64::floor` otherwise.
///
/// The cast `x as i64` saturates at `|x| >= 2^63` and maps NaN to 0, and the old
/// unconditional form then computed `i64::MIN - 1`: a panic under overflow checks
/// and a sign-flipped floor (`fast_floor(-1e20) = +9.2e18`) in a release build
/// (#2232). Below `2^52` the result is bit-identical to the old code. `NaN` now
/// stays `NaN` (the old code returned 0): in `cmir_efficacy` every comparison
/// with it is false, so the candidate scale is dropped, which is the sound outcome.
#[inline(always)]
fn fast_floor(x: f64) -> f64 {
    if !(x.abs() < TWO_POW_52) {
        return x.floor();
    }
    let t = x as i64;
    (t - ((x < t as f64) as i64)) as f64
}

/// `x as i64` (truncation toward zero) when it does not saturate and leaves headroom
/// for the `+-1` steps of the lifting loops; `None` for NaN, infinities and
/// `|x| >= 2^62`, where `as` would saturate and the lifting function would be
/// evaluated at a different integer than the one the math asks for (#2232).
#[inline(always)]
fn exact_i64(x: f64) -> Option<i64> {
    if x.abs() < EXACT_I64_BOUND {
        Some(x as i64)
    } else {
        None
    }
}

/// Step budget for evaluating the lifted mixed-integer cover's lifting functions
/// over one row. HiGHS's `gamma_l` walks `h = 0..=upper` of each cover variable,
/// which with `exact_i64` can now be ~2^62 steps (the old saturating `i32` cast
/// capped it near 2^31, then overflowed). A work limit HiGHS lacks: past it the cut
/// is refused, which is sound. Measured: a 200k-row random search at
/// `feastol = 1e-30` reached no row past it, and the 38-instance in-repo MILP panel
/// is unchanged with it (#2232 review). The test lowers it to drive the branch.
const MAX_LIFT_WORK: u64 = 1 << 24;

/// Which cover-lifting refusal fired, for tests that must prove a refusal branch
/// actually ran (CLAUDE.md §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiftRefusal {
    KnapsackRange,
    MixedIntegerRange,
    MixedIntegerWork,
}

#[cfg(test)]
thread_local! {
    static LIFT_REFUSALS: std::cell::RefCell<Vec<LiftRefusal>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[inline(always)]
fn note_refusal(_kind: LiftRefusal) {
    #[cfg(test)]
    LIFT_REFUSALS.with(|r| r.borrow_mut().push(_kind));
}

impl CutGenCore {
    /// A generator resuming from the given random state.
    pub fn new(p: CutGenParams, rng: HighsRandom) -> Self {
        CutGenCore {
            p,
            path: CutPath::None,
            rng,
            cover: Vec::new(),
            coverweight: CDouble::new(0.0),
            lambda: CDouble::new(0.0),
            upper: Vec::new(),
            solval: Vec::new(),
            complementation: Vec::new(),
            isintegral: Vec::new(),
            inds: Vec::new(),
            vals: Vec::new(),
            rhs: CDouble::new(0.0),
            integral_support: false,
            integral_coefficients: false,
            rowlen: 0,
            initial_scale: 1.0,
            integerinds: Vec::new(),
            deltas: Vec::new(),
            lift_work_limit: MAX_LIFT_WORK,
        }
    }

    /// HiGHS seeds the generator with `random_seed + LP iterations + pool size`.
    pub fn seeded(
        p: CutGenParams,
        random_seed: i32,
        lp_iterations: i64,
        num_pool_cuts: i32,
    ) -> Self {
        let seed = (random_seed as i64)
            .wrapping_add(lp_iterations)
            .wrapping_add(num_pool_cuts as i64);
        CutGenCore::new(p, HighsRandom::new(seed as u32))
    }

    /// The random generator (its state after the last call).
    pub fn rng(&self) -> &HighsRandom {
        &self.rng
    }

    /// The generator that produced the last `generate_cut_core` cut.
    pub fn last_path(&self) -> CutPath {
        self.path
    }

    // ---------------------------------------------------------------- helpers

    fn flip_complementation(&mut self, index: usize) {
        debug_assert!(self.upper[index] != K_HIGHS_INF);
        self.complementation[index] = 1 - self.complementation[index];
        self.solval[index] = self.upper[index] - self.solval[index];
        self.rhs -= self.upper[index] * self.vals[index];
        self.vals[index] = -self.vals[index];
    }

    fn remove_complementation(&mut self) {
        if self.complementation.is_empty() {
            return;
        }
        for i in 0..self.rowlen {
            if self.complementation[i] != 0 {
                self.flip_complementation(i);
            }
        }
    }

    #[inline(always)]
    fn update_violation_and_norm(
        &self,
        index: usize,
        aj: f64,
        violation: &mut f64,
        norm: &mut f64,
    ) {
        *violation += aj * self.solval[index];
        if aj > 0.0 && self.solval[index] <= self.p.feastol {
            return;
        }
        if aj < 0.0 && self.solval[index] >= self.upper[index] - self.p.feastol {
            return;
        }
        *norm += aj * aj;
    }

    /// `HighsCutGeneration::scale`: shift exponents so |val| lands near 1 (at most
    /// a 2^10 enlargement), returning the factor applied.
    fn scale(&mut self, val: f64) -> f64 {
        let mut expshift = -frexp_exp(val);
        expshift = expshift.min(10);
        self.rhs = self.rhs.ldexp(expshift);
        for i in 0..self.rowlen {
            self.vals[i] = libm::ldexp(self.vals[i], expshift);
        }
        libm::ldexp(1.0, expshift)
    }

    // ------------------------------------------------------------ preprocess

    fn preprocess_base_inequality(
        &mut self,
        col_integral: &[bool],
        has_unbounded_ints: &mut bool,
        has_general_ints: &mut bool,
        has_continuous: &mut bool,
    ) -> bool {
        let feastol = self.p.feastol;
        *has_unbounded_ints = false;
        *has_continuous = false;
        *has_general_ints = false;
        let mut num_zeros: usize = 0;

        let mut maxact = -feastol;
        let mut max_abs_val = 0.0f64;
        for i in 0..self.rowlen {
            max_abs_val = self.vals[i].abs().max(max_abs_val);
        }
        self.initial_scale = self.scale(max_abs_val);

        self.isintegral.resize(self.rowlen, 0);
        for i in 0..self.rowlen {
            self.isintegral[i] = (col_integral[i] && self.vals[i].abs() > 10.0 * feastol) as u8;
            if self.isintegral[i] == 0 {
                if self.upper[i] < 2.0 * self.solval[i] {
                    if self.complementation.is_empty() {
                        self.complementation.resize(self.rowlen, 0);
                    }
                    self.flip_complementation(i);
                }
                if self.vals[i] > 0.0 || self.vals[i].abs() * self.upper[i] <= 10.0 * feastol {
                    if self.vals[i] < 0.0 {
                        if self.upper[i] == K_HIGHS_INF {
                            return false;
                        }
                        self.rhs -= self.vals[i] * self.upper[i];
                    }
                    num_zeros += 1;
                    self.vals[i] = 0.0;
                    continue;
                }
                *has_continuous = true;
            } else {
                if self.upper[i] == K_HIGHS_INF {
                    *has_unbounded_ints = true;
                    *has_general_ints = true;
                } else if self.upper[i] != 1.0 {
                    *has_general_ints = true;
                }
                if self.vals[i] > 0.0 {
                    maxact += self.vals[i] * self.upper[i];
                }
            }
        }

        let max_len = (100.0 + 0.15 * (self.p.num_cols as f64)) as i64;
        if self.rowlen as i64 - num_zeros as i64 > max_len {
            let num_cancel = (self.rowlen as i64 - num_zeros as i64 - max_len) as usize;
            let mut cancel_nzs: Vec<usize> = Vec::new();
            for i in 0..self.rowlen {
                let cancel_slack = if self.vals[i] > 0.0 {
                    self.solval[i]
                } else {
                    self.upper[i] - self.solval[i]
                };
                if cancel_slack <= feastol {
                    cancel_nzs.push(i);
                }
            }
            if cancel_nzs.len() < num_cancel {
                return false;
            }
            if cancel_nzs.len() > num_cancel {
                let vals = &self.vals;
                partial_sort(&mut cancel_nzs, num_cancel, |&a: &usize, &b: &usize| {
                    vals[a].abs() < vals[b].abs()
                });
            }
            for &j in cancel_nzs.iter().take(num_cancel) {
                if self.vals[j] < 0.0 {
                    self.rhs -= self.vals[j] * self.upper[j];
                } else {
                    maxact -= self.vals[j] * self.upper[j];
                }
                self.vals[j] = 0.0;
            }
            num_zeros += num_cancel;
        }

        if num_zeros != 0 {
            let has_compl = !self.complementation.is_empty();
            let mut i = self.rowlen;
            while i > 0 {
                i -= 1;
                if self.vals[i] == 0.0 {
                    self.rowlen -= 1;
                    let r = self.rowlen;
                    self.inds[i] = self.inds[r];
                    self.vals[i] = self.vals[r];
                    self.upper[i] = self.upper[r];
                    self.solval[i] = self.solval[r];
                    self.isintegral[i] = self.isintegral[r];
                    if has_compl {
                        self.complementation[i] = self.complementation[r];
                    }
                    num_zeros -= 1;
                    if num_zeros == 0 {
                        break;
                    }
                }
            }
        }
        self.truncate_to_rowlen();
        maxact > self.rhs
    }

    fn truncate_to_rowlen(&mut self) {
        let n = self.rowlen;
        self.inds.truncate(n);
        self.vals.truncate(n);
        self.upper.truncate(n);
        self.solval.truncate(n);
        self.isintegral.truncate(n);
        if !self.complementation.is_empty() {
            self.complementation.truncate(n);
        }
    }

    // ------------------------------------------------------------------ cover

    /// `determineCover(lpSol)`: `nodes` is `None` for `lpSol = true` (an LP point's
    /// cover) and the node queue's open-node counts for `lpSol = false` (a conflict
    /// proof's cover, `generateConflict`).
    fn determine_cover(&mut self, nodes: Option<&dyn OpenNodeCounts>) -> bool {
        let feastol = self.p.feastol;
        if self.rhs <= 10.0 * feastol {
            return false;
        }
        self.cover.clear();
        for j in 0..self.rowlen {
            if self.isintegral[j] == 0 {
                continue;
            }
            if nodes.is_none() && self.solval[j] <= feastol {
                continue;
            }
            self.cover.push(j);
        }
        let max_cover_size = self.cover.len();
        let r = self.rng.integer();
        self.coverweight = CDouble::new(0.0);

        let (solval, upper, vals, inds) = (&self.solval, &self.upper, &self.vals, &self.inds);
        let mut coversize = 0;
        if let Some(nq) = nodes {
            let compl = &self.complementation;
            let count = |i: usize| -> i64 {
                let col = inds[i] as usize;
                if compl.get(i).copied().unwrap_or(0) != 0 {
                    nq.num_nodes_down(col)
                } else {
                    nq.num_nodes_up(col)
                }
            };
            pdqsort(
                &mut self.cover[..max_cover_size],
                |&i: &usize, &j: &usize| {
                    if solval[i] > feastol && solval[j] <= feastol {
                        return true;
                    }
                    if solval[i] <= feastol && solval[j] > feastol {
                        return false;
                    }
                    let (na, nb) = (count(i), count(j));
                    if na > nb {
                        return true;
                    }
                    if na < nb {
                        return false;
                    }
                    hash_pair_i32(inds[i], r) > hash_pair_i32(inds[j], r)
                },
            );
        } else {
            coversize = partition(&mut self.cover, |&j: &usize| {
                solval[j] >= upper[j] - feastol
            });
            for i in 0..coversize {
                let j = self.cover[i];
                self.coverweight += vals[j] * upper[j];
            }
            pdqsort(
                &mut self.cover[coversize..max_cover_size],
                |&i: &usize, &j: &usize| {
                    if upper[i] < 1.5 && upper[j] > 1.5 {
                        return true;
                    }
                    if upper[i] > 1.5 && upper[j] < 1.5 {
                        return false;
                    }
                    let contribution_a = solval[i] * vals[i];
                    let contribution_b = solval[j] * vals[j];
                    if contribution_a > contribution_b + feastol {
                        return true;
                    }
                    if contribution_a < contribution_b - feastol {
                        return false;
                    }
                    if (vals[i] - vals[j]).abs() <= feastol {
                        return hash_pair_i32(inds[i], r) > hash_pair_i32(inds[j], r);
                    }
                    vals[i] > vals[j]
                },
            );
        }

        let minlambda = (10.0 * feastol).max(feastol * self.rhs.to_f64().abs());
        while coversize != max_cover_size {
            let lambda = (self.coverweight - self.rhs).to_f64();
            if lambda > minlambda {
                break;
            }
            let j = self.cover[coversize];
            self.coverweight += self.vals[j] * self.upper[j];
            coversize += 1;
        }
        if coversize == 0 {
            return false;
        }
        self.coverweight.renormalize();
        self.lambda = self.coverweight - self.rhs;
        if self.lambda <= minlambda {
            return false;
        }
        self.cover.truncate(coversize);
        true
    }

    /// Returns `false`, leaving `rhs` unchanged, when a lifting coefficient is out
    /// of the range where the integer arithmetic is exact (#2232).
    fn separate_lifted_knapsack_cover(&mut self) -> bool {
        let feastol = self.p.feastol;
        let epsilon = self.p.epsilon;
        let coversize = self.cover.len();
        let mut s = vec![0.0f64; coversize];
        let mut coverflag = vec![0i8; self.rowlen];
        {
            let vals = &self.vals;
            pdqsort_branchless(&mut self.cover, |&a: &usize, &b: &usize| vals[a] > vals[b]);
        }
        let vals = &self.vals;
        let cover = &self.cover;
        let mut abartmp = CDouble::new(vals[cover[0]]);
        let mut sigma = self.lambda;
        for i in 1..coversize {
            let delta = abartmp - vals[cover[i]];
            let kdelta = (i as f64) * delta;
            if kdelta.to_f64() < sigma.to_f64() {
                abartmp = CDouble::new(vals[cover[i]]);
                sigma -= kdelta;
            } else {
                abartmp -= sigma * (1.0 / i as f64);
                sigma = CDouble::new(0.0);
                break;
            }
        }
        if sigma.to_f64() > 0.0 {
            abartmp = self.rhs / (coversize as f64);
        }
        let abar = abartmp.to_f64();

        let mut sum = CDouble::new(0.0);
        let mut cplussize: i32 = 0;
        for i in 0..coversize {
            sum += abar.min(vals[cover[i]]);
            s[i] = sum.to_f64();
            if vals[cover[i]] > abar + feastol {
                cplussize += 1;
                coverflag[cover[i]] = 1;
            } else {
                coverflag[cover[i]] = -1;
            }
        }
        let mut halfintegral = false;
        let mut out_of_range = false;
        let mut g = |z: f64| -> f64 {
            let hfrac = z / abar;
            let mut coef = 0.0;
            // HiGHS: `HighsInt h = std::floor(hfrac + 0.5)`, undefined behaviour out of
            // range. A saturating `as i32` made `h - 1` overflow for `hfrac <= -2^31`,
            // which wraps to `i32::MAX` in a release build: a huge coefficient on a
            // variable whose lifted coefficient is 0, i.e. an invalid cut (#2232).
            // `i64` is exact wherever the old `i32` was; beyond 2^62 the cut is refused.
            let Some(mut h) = exact_i64((hfrac + 0.5).floor()) else {
                out_of_range = true;
                return 0.0;
            };
            if h != 0
                && (hfrac - h as f64).abs() * 1.0f64.max(abar) <= epsilon
                && h <= cplussize as i64 - 1
            {
                halfintegral = true;
                coef = 0.5;
            }
            h = (h - 1).max(0);
            while (h as usize) < coversize {
                if z <= s[h as usize] + feastol {
                    break;
                }
                h += 1;
            }
            coef + h as f64
        };

        let mut newvals = std::mem::take(&mut self.vals);
        for i in 0..self.rowlen {
            if newvals[i] == 0.0 {
                continue;
            }
            if coverflag[i] == -1 {
                newvals[i] = 1.0;
            } else {
                newvals[i] = g(newvals[i]);
            }
        }
        // `try_generate_cut` restores `vals` from its copy either way; `rhs` must
        // be untouched on refusal because the c-MIR fallback reads it.
        self.vals = newvals;
        if out_of_range {
            note_refusal(LiftRefusal::KnapsackRange);
            return false;
        }
        self.rhs = CDouble::new((coversize as i64 - 1) as f64);
        if halfintegral {
            self.rhs *= 2.0;
            for i in 0..self.rowlen {
                self.vals[i] *= 2.0;
            }
        }
        self.integral_support = true;
        self.integral_coefficients = true;
        true
    }

    fn separate_lifted_mixed_binary_cover(&mut self) -> bool {
        let epsilon = self.p.epsilon;
        self.integral_support = false;
        self.integral_coefficients = false;
        let coversize = self.cover.len();
        let mut s = vec![0.0f64; coversize];
        let mut coverflag = vec![0u8; self.rowlen];
        if coversize == 0 {
            return false;
        }
        for i in 0..coversize {
            coverflag[self.cover[i]] = 1;
        }
        {
            let vals = &self.vals;
            pdqsort_branchless(&mut self.cover, |&a: &usize, &b: &usize| vals[a] > vals[b]);
        }
        let lambda = self.lambda;
        let mut sum = CDouble::new(0.0);
        let mut p = coversize;
        for i in 0..coversize {
            if self.vals[self.cover[i]] - lambda <= epsilon {
                p = i;
                break;
            }
            sum += self.vals[self.cover[i]];
            s[i] = sum.to_f64();
        }
        if p == 0 {
            return false;
        }
        let phi = |a: f64| -> f64 {
            for i in 0..p {
                if a <= s[i] - lambda {
                    return ((i as f64) * lambda).to_f64();
                }
                if a <= s[i] {
                    return (((i + 1) as f64) * lambda + (CDouble::new(a) - s[i])).to_f64();
                }
            }
            ((p as f64) * lambda + (CDouble::new(a) - s[p - 1])).to_f64()
        };

        self.rhs = -lambda;
        self.integral_coefficients = false;
        self.integral_support = true;
        for i in 0..self.rowlen {
            if self.isintegral[i] == 0 {
                if self.vals[i] < 0.0 {
                    self.integral_support = false;
                } else {
                    self.vals[i] = 0.0;
                }
                continue;
            }
            if coverflag[i] != 0 {
                self.vals[i] = self.vals[i].min(lambda.to_f64());
                self.rhs += self.vals[i];
            } else {
                self.vals[i] = phi(self.vals[i]);
            }
        }
        true
    }

    fn separate_lifted_mixed_integer_cover(&mut self) -> bool {
        let feastol = self.p.feastol;
        self.integral_support = false;
        self.integral_coefficients = false;
        let coversize = self.cover.len();
        let mut coverflag = vec![0u8; self.rowlen];
        for &i in &self.cover {
            coverflag[i] = 1;
        }
        {
            let vals = &self.vals;
            pdqsort_branchless(&mut self.cover, |&a: &usize, &b: &usize| vals[a] > vals[b]);
        }
        let lambda = self.lambda;
        let mut a: Vec<CDouble> = vec![CDouble::new(0.0); coversize];
        let mut u: Vec<CDouble> = vec![CDouble::new(0.0); coversize + 1];
        let mut m: Vec<CDouble> = vec![CDouble::new(0.0); coversize + 1];
        let mut usum = CDouble::new(0.0);
        let mut msum = CDouble::new(0.0);
        for c in 0..coversize {
            let i = self.cover[c];
            u[c] = usum;
            m[c] = msum;
            a[c] = CDouble::new(self.vals[i]);
            let ub = self.upper[i];
            usum += ub;
            msum += ub * a[c];
        }
        u[coversize] = usum;
        m[coversize] = msum;

        let mut lpos: i64 = -1;
        let mut bestl_cplusend: usize = 0;
        let mut bestl_val = 0.0f64;
        let mut bestl_at_upper = true;
        for i in 0..coversize {
            let j = self.cover[i];
            let ub = self.upper[j];
            let at_upper = self.solval[j] >= ub - feastol;
            if at_upper && !bestl_at_upper {
                continue;
            }
            let mju = ub * self.vals[j];
            let mu = mju - lambda;
            if mu <= 10.0 * feastol {
                continue;
            }
            if self.vals[j].abs() < 1000.0 * feastol {
                continue;
            }
            let mudival = (mu / self.vals[j]).to_f64();
            if is_integral(mudival, feastol) {
                continue;
            }
            let eta = mudival.ceil();
            let ulminusetaplusone = CDouble::new(ub) - eta + 1.0;
            let cplusthreshold = ulminusetaplusone * self.vals[j];
            let thr = cplusthreshold.to_f64();
            let vals = &self.vals;
            let cplusend = upper_bound(&self.cover, &thr, |t: &f64, &ci: &usize| *t > vals[ci]);
            let mut mcplus = m[cplusend];
            if i < cplusend {
                mcplus -= mju;
            }
            let jl_val = (mcplus + eta * self.vals[j]).to_f64();
            if jl_val > bestl_val || (!at_upper && bestl_at_upper) {
                lpos = i as i64;
                bestl_cplusend = cplusend;
                bestl_val = jl_val;
                bestl_at_upper = at_upper;
            }
        }
        if lpos == -1 {
            return false;
        }
        let lpos = lpos as usize;
        let l = self.cover[lpos];
        let al = CDouble::new(self.vals[l]);
        let upperl = self.upper[l];
        let mlu = upperl * al;
        let mu = mlu - lambda;

        a.truncate(bestl_cplusend);
        self.cover.truncate(bestl_cplusend);
        u.truncate(bestl_cplusend + 1);
        m.truncate(bestl_cplusend + 1);
        if lpos < bestl_cplusend {
            a.remove(lpos);
            self.cover.remove(lpos);
            u.remove(lpos + 1);
            m.remove(lpos + 1);
            for i in lpos + 1..bestl_cplusend {
                u[i] -= upperl;
                m[i] -= mlu;
            }
        }
        let cplussize = a.len();

        let mudival = (mu / al).to_f64();
        let eta = mudival.ceil();
        let mut r = mu - mudival.floor() * al;
        if r < 0.0 {
            r = CDouble::new(0.0);
        }
        let ulminusetaplusone = CDouble::new(upperl) - eta + 1.0;
        let cplusthreshold = ulminusetaplusone * al;
        // HiGHS converts these to `HighsInt`, undefined behaviour out of range; the
        // saturating `as` casts this replaces computed the lifting function at the
        // wrong integer and could overflow `k - 1` / `p - 1`. Every conversion below
        // goes through `exact_i64`, and an out-of-range one refuses the cut (#2232).
        // The values are identical to the old casts wherever those did not saturate
        // (|value| < 2^31 for the old `i32` ones).
        let (Some(kmin), Some(kmax)) = (
            exact_i64((eta - upperl - 0.5).floor()),
            exact_i64((upperl - eta + 0.5).floor()),
        ) else {
            note_refusal(LiftRefusal::MixedIntegerRange);
            return false;
        };
        let out_of_range = std::cell::Cell::new(false);
        let trunc_i64 = |x: f64| -> i64 {
            exact_i64(x).unwrap_or_else(|| {
                out_of_range.set(true);
                0
            })
        };
        // Steps taken by the lifting loops; past `MAX_LIFT_WORK` the cut is refused.
        let work = std::cell::Cell::new(0u64);
        let work_limit = self.lift_work_limit;
        let over_budget = || {
            work.set(work.get() + 1);
            work.get() > work_limit
        };

        let phi_l = |av: f64| -> f64 {
            let mut k: i64 = trunc_i64(av / al.to_f64()).min(-1);
            if out_of_range.get() {
                return 0.0;
            }
            while k >= kmin {
                if over_budget() {
                    return 0.0;
                }
                if av >= (k as f64) * al + r {
                    return (av - ((k + 1) as f64) * r).to_f64();
                }
                if av >= (k as f64) * al {
                    return ((k as f64) * (al - r)).to_f64();
                }
                k -= 1;
            }
            ((kmin as f64) * (al - r)).to_f64()
        };

        let upper = &self.upper;
        let cover = &self.cover;
        let gamma_l = |z: f64| -> f64 {
            for i in 0..cplussize {
                let upperi = trunc_i64(upper[cover[i]]);
                if out_of_range.get() {
                    return 0.0;
                }
                let mut h: i64 = 0;
                while h <= upperi {
                    if over_budget() {
                        return 0.0;
                    }
                    let mih = m[i] + (h as f64) * a[i];
                    let uih = u[i] + h as f64;
                    let mihplusdeltai = mih + a[i] - cplusthreshold;
                    if z <= mihplusdeltai {
                        return (uih * ulminusetaplusone * (al - r)).to_f64();
                    }
                    let mut k: i64 = trunc_i64(((z - mihplusdeltai) / al).to_f64()) - 1;
                    if out_of_range.get() {
                        return 0.0;
                    }
                    while k <= kmax {
                        if over_budget() {
                            return 0.0;
                        }
                        if z <= mihplusdeltai + (k as f64) * al + r {
                            return ((uih * ulminusetaplusone + k as f64) * (al - r)).to_f64();
                        }
                        if z <= mihplusdeltai + ((k + 1) as f64) * al {
                            return ((uih * ulminusetaplusone) * (al - r) + z - mih - a[i]
                                + cplusthreshold
                                - ((k + 1) as f64) * r)
                                .to_f64();
                        }
                        k += 1;
                    }
                    h += 1;
                }
            }
            let mut p: i64 = trunc_i64(((z - m[cplussize]) / al).to_f64()) - 1;
            if out_of_range.get() {
                return 0.0;
            }
            loop {
                if over_budget() {
                    return 0.0;
                }
                if z <= m[cplussize] + (p as f64) * al + r {
                    return ((u[cplussize] * ulminusetaplusone + p as f64) * (al - r)).to_f64();
                }
                if z <= m[cplussize] + ((p + 1) as f64) * al {
                    return ((u[cplussize] * ulminusetaplusone) * (al - r) + z
                        - m[cplussize]
                        - ((p + 1) as f64) * r)
                        .to_f64();
                }
                p += 1;
            }
        };

        let mut rhs = (CDouble::new(upperl) - eta) * r - lambda;
        let mut integral_support = true;
        let mut newvals = std::mem::take(&mut self.vals);
        for i in 0..self.rowlen {
            if newvals[i] == 0.0 {
                continue;
            }
            if self.isintegral[i] == 0 {
                if newvals[i] < 0.0 {
                    integral_support = false;
                } else {
                    newvals[i] = 0.0;
                }
                continue;
            }
            if coverflag[i] != 0 {
                newvals[i] = -phi_l(-newvals[i]);
                rhs += newvals[i] * upper[i];
            } else {
                newvals[i] = gamma_l(newvals[i]);
            }
        }
        self.vals = newvals;
        // `rhs` is untouched on refusal; `try_generate_cut` restores `vals`.
        if out_of_range.get() {
            note_refusal(LiftRefusal::MixedIntegerRange);
            return false;
        }
        if work.get() > work_limit {
            note_refusal(LiftRefusal::MixedIntegerWork);
            return false;
        }
        self.rhs = rhs;
        self.integral_support = integral_support;
        self.integral_coefficients = false;
        true
    }

    // ------------------------------------------------------------------- cmir

    /// One trial scale of the c-MIR heuristic: the efficacy of the c-MIR of the
    /// current row scaled by `1/delta`, or `None` where HiGHS skips the scale.
    fn cmir_efficacy(
        &self,
        integerinds: &[usize],
        delta: f64,
        continuouscontribution: f64,
        continuoussqrnorm: f64,
    ) -> Option<f64> {
        const MAX_CMIR_SCALE: f64 = 1e6;
        const F0MIN: f64 = 0.005;
        const F0MAX: f64 = 0.995;
        let scale = 1.0 / delta;
        let scalrhs = self.rhs.to_f64() * scale;
        let downrhs = fast_floor(scalrhs);
        let f0 = scalrhs - downrhs;
        if !(F0MIN..=F0MAX).contains(&f0) {
            return None;
        }
        let oneoveroneminusf0 = 1.0 / (1.0 - f0);
        if oneoveroneminusf0 > MAX_CMIR_SCALE {
            return None;
        }
        let contscale = scale * oneoveroneminusf0;
        let mut sqrnorm = contscale * contscale * continuoussqrnorm;
        let mut viol = contscale * continuouscontribution - downrhs;
        for &j in integerinds {
            let scalaj = self.vals[j] * scale;
            let downaj = fast_floor(scalaj + K_HIGHS_TINY);
            let fj = scalaj - downaj;
            let aj = downaj + 0.0f64.max((fj - f0) * oneoveroneminusf0);
            self.update_violation_and_norm(j, aj, &mut viol, &mut sqrnorm);
        }
        Some(viol / sqrnorm.sqrt())
    }

    fn cmir_cut_generation_heuristic(
        &mut self,
        min_efficacy: f64,
        only_initial_cmir_scale: bool,
    ) -> bool {
        const F0MIN: f64 = 0.005;
        const F0MAX: f64 = 0.995;
        let feastol = self.p.feastol;
        self.integral_support = false;
        self.integral_coefficients = false;
        let mut continuouscontribution = 0.0f64;
        let mut continuoussqrnorm = 0.0f64;
        self.deltas.clear();
        self.integerinds.clear();
        let mut maxabsdelta = 0.0f64;
        self.complementation.resize(self.rowlen, 0);

        for i in 0..self.rowlen {
            if self.isintegral[i] != 0 {
                self.integerinds.push(i);
                if self.upper[i] < 2.0 * self.solval[i] {
                    self.flip_complementation(i);
                }
                if only_initial_cmir_scale {
                    continue;
                }
                if self.solval[i] > feastol {
                    let delta = self.vals[i].abs();
                    if delta <= 1e-4 || delta == maxabsdelta {
                        continue;
                    }
                    maxabsdelta = maxabsdelta.max(delta);
                    self.deltas.push(delta);
                }
            } else {
                let v = self.vals[i];
                self.update_violation_and_norm(
                    i,
                    v,
                    &mut continuouscontribution,
                    &mut continuoussqrnorm,
                );
            }
        }

        if continuoussqrnorm == 0.0 && self.deltas.len() > 1 {
            let int_scale = integral_scale(&self.deltas, feastol, K_HIGHS_TINY);
            if int_scale != 0.0 && int_scale <= 1e4 {
                let scalrhs = self.rhs.to_f64() * int_scale;
                let downrhs = fast_floor(scalrhs);
                let f0 = scalrhs - downrhs;
                if (F0MIN..=F0MAX).contains(&f0) {
                    self.deltas.push(1.0 / int_scale);
                }
            }
        }
        self.deltas.push(1.0f64.min(self.initial_scale));
        if !only_initial_cmir_scale {
            self.deltas
                .push(maxabsdelta + 1.0f64.min(self.initial_scale));
        }
        if self.deltas.is_empty() {
            return false;
        }
        pdqsort_f64(&mut self.deltas);
        let mut curdelta = self.deltas[0];
        for i in 1..self.deltas.len() {
            if self.deltas[i] - curdelta <= 10.0 * feastol {
                self.deltas[i] = 0.0;
            } else {
                curdelta = self.deltas[i];
            }
        }
        self.deltas.retain(|&d| d != 0.0);
        let mut bestdelta = -1.0f64;
        let mut bestefficacy = min_efficacy;
        let deltas = std::mem::take(&mut self.deltas);
        let integerinds = std::mem::take(&mut self.integerinds);
        for &delta in &deltas {
            if let Some(efficacy) = self.cmir_efficacy(
                &integerinds,
                delta,
                continuouscontribution,
                continuoussqrnorm,
            ) {
                if efficacy > bestefficacy {
                    bestdelta = delta;
                    bestefficacy = efficacy;
                }
            }
        }
        self.deltas = deltas;
        if bestdelta == -1.0 {
            self.integerinds = integerinds;
            return false;
        }
        let mut k = 1;
        while !only_initial_cmir_scale && k <= 3 {
            let delta = bestdelta * ((1i32 << k) as f64);
            if let Some(efficacy) = self.cmir_efficacy(
                &integerinds,
                delta,
                continuouscontribution,
                continuoussqrnorm,
            ) {
                if efficacy > bestefficacy {
                    bestdelta = delta;
                    bestefficacy = efficacy;
                }
            }
            k += 1;
        }

        for &k in &integerinds {
            if self.upper[k] == K_HIGHS_INF {
                continue;
            }
            if self.solval[k] <= feastol {
                continue;
            }
            self.flip_complementation(k);
            let eff = self.cmir_efficacy(
                &integerinds,
                bestdelta,
                continuouscontribution,
                continuoussqrnorm,
            );
            match eff {
                Some(efficacy) if efficacy > bestefficacy => bestefficacy = efficacy,
                _ => self.flip_complementation(k),
            }
        }
        self.integerinds = integerinds;

        let scale = 1.0 / CDouble::new(bestdelta);
        let scalrhs = self.rhs * scale;
        let downrhs = scalrhs.to_f64().floor();
        let f0 = scalrhs - downrhs;
        let oneoveroneminusf0 = 1.0 / (1.0 - f0);

        self.rhs = CDouble::new(downrhs * bestdelta);
        self.integral_support = true;
        self.integral_coefficients = false;
        for j in 0..self.rowlen {
            if self.vals[j] == 0.0 {
                continue;
            }
            if self.isintegral[j] == 0 {
                if self.vals[j] > 0.0 {
                    self.vals[j] = 0.0;
                } else {
                    self.vals[j] = (self.vals[j] * oneoveroneminusf0).to_f64();
                    self.integral_support = false;
                }
            } else {
                let scalaj = scale * self.vals[j];
                let downaj = (scalaj + K_HIGHS_TINY).to_f64().floor();
                let fj = scalaj - downaj;
                let mut aj = CDouble::new(downaj);
                if fj > f0 {
                    aj += (fj - f0) * oneoveroneminusf0;
                }
                self.vals[j] = (aj * bestdelta).to_f64();
            }
        }
        true
    }

    // ------------------------------------------------------------ try / main

    #[allow(clippy::too_many_arguments)]
    fn try_generate_cut(
        &mut self,
        has_unbounded_ints: bool,
        has_general_ints: bool,
        has_continuous: bool,
        min_efficacy: f64,
        only_initial_cmir_scale: bool,
        allow_reject_cut: bool,
        nodes: Option<&dyn OpenNodeCounts>,
    ) -> bool {
        if has_unbounded_ints {
            self.path = CutPath::CmirUnboundedInt;
            return self.cmir_cut_generation_heuristic(min_efficacy, only_initial_cmir_scale);
        }
        let tmp_vals = self.vals.clone();
        let tmp_inds = self.inds.clone();
        let tmp_complementation = self.complementation.clone();
        let tmp_solval = self.solval.clone();
        let mut tmp_rhs = self.rhs;

        let mut success = false;
        let mut save_integral_support = false;
        let mut save_integral_coefficients = false;
        let mut lifted_path = CutPath::None;
        if self.determine_cover(nodes) {
            if !has_continuous && !has_general_ints {
                success = self.separate_lifted_knapsack_cover();
                lifted_path = CutPath::KnapsackCover;
            } else if has_general_ints {
                success = self.separate_lifted_mixed_integer_cover();
                lifted_path = CutPath::MixedIntegerCover;
            } else {
                success = self.separate_lifted_mixed_binary_cover();
                lifted_path = CutPath::MixedBinaryCover;
            }
        }
        let mut min_mir_efficacy = min_efficacy;
        if success {
            save_integral_support = self.integral_support;
            save_integral_coefficients = self.integral_coefficients;
            let mut violation = -self.rhs.to_f64();
            let mut sqrnorm = 0.0f64;
            for i in 0..self.rowlen {
                let v = self.vals[i];
                self.update_violation_and_norm(i, v, &mut violation, &mut sqrnorm);
            }
            let efficacy = violation / sqrnorm.sqrt();
            if allow_reject_cut && efficacy <= min_efficacy {
                success = false;
                self.rhs = tmp_rhs;
            } else {
                min_mir_efficacy += efficacy;
                std::mem::swap(&mut tmp_rhs, &mut self.rhs);
            }
        }
        // Run c-MIR on the saved copy; keep the lifted cut aside.
        let lifted_vals = std::mem::replace(&mut self.vals, tmp_vals);
        let lifted_inds = std::mem::replace(&mut self.inds, tmp_inds);

        if self.cmir_cut_generation_heuristic(min_mir_efficacy, only_initial_cmir_scale) {
            self.path = CutPath::Cmir;
            true
        } else if success {
            self.path = lifted_path;
            self.rhs = tmp_rhs;
            self.complementation = tmp_complementation;
            self.solval = tmp_solval;
            self.vals = lifted_vals;
            self.inds = lifted_inds;
            self.integral_support = save_integral_support;
            self.integral_coefficients = save_integral_coefficients;
            true
        } else {
            false
        }
    }

    /// The part of `HighsCutGeneration::generateCut` between `transform` and
    /// `untransform`. `None` where HiGHS returns `false` at that stage.
    pub fn generate_cut_core(
        &mut self,
        row: &BaseRow,
        only_initial_cmir_scale: bool,
    ) -> Option<CoreCut> {
        self.path = CutPath::None;
        self.rowlen = row.inds.len();
        self.inds = row.inds.clone();
        self.vals = row.vals.clone();
        self.upper = row.upper.clone();
        self.solval = row.solval.clone();
        self.rhs = CDouble::new(row.rhs);
        self.complementation.clear();
        self.isintegral.clear();
        let mut has_unbounded_ints = false;
        let mut has_general_ints = false;
        let mut has_continuous = false;
        // `col_integral` is permuted together with `inds` by the in-place zero
        // removal; only `isintegral` (computed from it first) is read afterwards.
        if !self.preprocess_base_inequality(
            &row.col_integral,
            &mut has_unbounded_ints,
            &mut has_general_ints,
            &mut has_continuous,
        ) {
            return None;
        }
        if !has_unbounded_ints && !row.ints_positive {
            self.complementation.resize(self.rowlen, 0);
            for i in 0..self.rowlen {
                if self.vals[i] > 0.0 || self.isintegral[i] == 0 {
                    continue;
                }
                self.flip_complementation(i);
            }
        }
        if !self.try_generate_cut(
            has_unbounded_ints,
            has_general_ints,
            has_continuous,
            10.0 * self.p.feastol,
            only_initial_cmir_scale,
            true,
            None,
        ) {
            return None;
        }
        self.remove_complementation();
        let mut i = self.rowlen;
        while i > 0 {
            i -= 1;
            if self.vals[i] == 0.0 {
                self.rowlen -= 1;
                let r = self.rowlen;
                self.inds[i] = self.inds[r];
                self.vals[i] = self.vals[r];
            }
        }
        self.inds.truncate(self.rowlen);
        self.vals.truncate(self.rowlen);
        Some(CoreCut {
            inds: self.inds.clone(),
            vals: self.vals.clone(),
            rhs: self.rhs,
            integral_support: self.integral_support,
            integral_coefficients: self.integral_coefficients,
        })
    }

    /// The part of `HighsCutGeneration::generateConflict` (`HighsCutGeneration.cpp:
    /// 1208-1280`) before `postprocessCut`: the dual proof `vals . x[inds] <= rhs`
    /// complemented against the GLOBAL bounds (`glb`, `gub`), the local bounds
    /// (`llb`, `lub`) as the point the cover is built around, then
    /// `preprocessBaseInequality` and `tryGenerateCut(..., feastol, false, false,
    /// false)`, with the complementation removed again. `nodes` are the open-node
    /// counts `determineCover(lpSol = false)` orders the cover by. `None` where HiGHS
    /// returns `false`. Every input slice is indexed by position in `inds`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_conflict_core(
        &mut self,
        inds: &[i32],
        vals: &[f64],
        rhs: f64,
        glb: &[f64],
        gub: &[f64],
        llb: &[f64],
        lub: &[f64],
        col_integral: &[bool],
        nodes: &dyn OpenNodeCounts,
    ) -> Option<CoreCut> {
        self.path = CutPath::None;
        self.rowlen = inds.len();
        self.inds = inds.to_vec();
        self.vals = vals.to_vec();
        self.rhs = CDouble::new(rhs);
        self.complementation = vec![0; self.rowlen];
        self.isintegral.clear();
        self.upper = vec![0.0; self.rowlen];
        self.solval = vec![0.0; self.rowlen];
        // Global bounds by column, for removing the complementation after the
        // in-place reordering of `preprocessBaseInequality`.
        let mut gbounds: std::collections::HashMap<i32, (f64, f64)> =
            std::collections::HashMap::with_capacity(self.rowlen);
        let mut activity = 0.0f64;
        for i in 0..self.rowlen {
            let (gl, gu) = (glb[i], gub[i]);
            gbounds.insert(inds[i], (gl, gu));
            self.upper[i] = gu - gl;
            let mut sv = if self.vals[i] < 0.0 {
                gu.min(lub[i])
            } else {
                gl.max(llb[i])
            };
            if self.vals[i] < 0.0 && gu != K_HIGHS_INF {
                self.rhs -= gu * self.vals[i];
                self.vals[i] = -self.vals[i];
                self.complementation[i] = 1;
                sv = gu - sv;
            } else {
                self.rhs -= gl * self.vals[i];
                self.complementation[i] = 0;
                sv -= gl;
            }
            self.solval[i] = sv;
            activity += sv * self.vals[i];
        }
        let r = self.rhs.to_f64();
        if activity > r {
            let sol_scale = r / activity;
            for v in self.solval.iter_mut() {
                *v *= sol_scale;
            }
        }
        let mut has_unbounded_ints = false;
        let mut has_general_ints = false;
        let mut has_continuous = false;
        if !self.preprocess_base_inequality(
            col_integral,
            &mut has_unbounded_ints,
            &mut has_general_ints,
            &mut has_continuous,
        ) {
            return None;
        }
        let feastol = self.p.feastol;
        if !self.try_generate_cut(
            has_unbounded_ints,
            has_general_ints,
            has_continuous,
            feastol,
            false,
            false,
            Some(nodes),
        ) {
            return None;
        }
        if !self.complementation.is_empty() {
            for i in 0..self.rowlen {
                let (gl, gu) = gbounds[&self.inds[i]];
                if self.complementation[i] != 0 {
                    self.rhs -= gu * self.vals[i];
                    self.vals[i] = -self.vals[i];
                } else {
                    self.rhs += gl * self.vals[i];
                }
            }
        }
        self.inds.truncate(self.rowlen);
        self.vals.truncate(self.rowlen);
        Some(CoreCut {
            inds: self.inds.clone(),
            vals: self.vals.clone(),
            rhs: self.rhs,
            integral_support: self.integral_support,
            integral_coefficients: self.integral_coefficients,
        })
    }

    /// `HighsCutGeneration::postprocessCut` on an untransformed cut. `lb`, `ub` and
    /// `is_int` read the global domain / integrality of a column. Returns the final
    /// cut and its integrality flags, or `None` where HiGHS rejects it.
    pub fn postprocess_cut(
        &mut self,
        cut: CoreCut,
        lb: &dyn Fn(i32) -> f64,
        ub: &dyn Fn(i32) -> f64,
        is_int: &dyn Fn(i32) -> bool,
    ) -> Option<CoreCut> {
        let feastol = self.p.feastol;
        let epsilon = self.p.epsilon;
        self.rowlen = cut.inds.len();
        self.inds = cut.inds;
        self.vals = cut.vals;
        self.rhs = cut.rhs;
        self.integral_support = cut.integral_support;
        self.integral_coefficients = cut.integral_coefficients;
        let out = |s: &mut Self| {
            s.inds.truncate(s.rowlen);
            s.vals.truncate(s.rowlen);
            Some(CoreCut {
                inds: s.inds.clone(),
                vals: s.vals.clone(),
                rhs: s.rhs,
                integral_support: s.integral_support,
                integral_coefficients: s.integral_coefficients,
            })
        };

        if self.rhs < 0.0 && self.rhs > -epsilon {
            self.rhs = CDouble::new(0.0);
        }
        if self.integral_support && self.integral_coefficients {
            self.remove_zeros();
            return out(self);
        }
        let mut max_abs_value = 0.0f64;
        for i in 0..self.rowlen {
            max_abs_value = self.vals[i].abs().max(max_abs_value);
        }
        let min_coefficient_value = 100.0 * feastol * max_abs_value.max(1e-3);
        self.integral_support = true;
        let mut i = self.rowlen;
        while i > 0 {
            i -= 1;
            if self.vals[i] == 0.0 {
                continue;
            }
            if self.vals[i].abs() <= min_coefficient_value {
                if self.vals[i] < 0.0 {
                    let u = ub(self.inds[i]);
                    if u == K_HIGHS_INF {
                        return None;
                    }
                    self.rhs -= u * self.vals[i];
                } else {
                    let l = lb(self.inds[i]);
                    if l == -K_HIGHS_INF {
                        return None;
                    }
                    self.rhs -= l * self.vals[i];
                }
                self.vals[i] = 0.0;
                continue;
            }
            if self.integral_support && !is_int(self.inds[i]) {
                self.integral_support = false;
            }
        }
        self.remove_zeros();
        if self.rowlen == 0 {
            return None;
        }
        if self.integral_support {
            let intscale = integral_scale(&self.vals[..self.rowlen], feastol, epsilon);
            let mut scale_smallest_val_to_one = true;
            if intscale != 0.0 && intscale * 1.0f64.max(max_abs_value) <= (1u64 << 52) as f64 {
                self.rhs.renormalize();
                self.rhs *= intscale;
                max_abs_value = nearest_integer(max_abs_value * intscale) as f64;
                for i in 0..self.rowlen {
                    let scaleval = intscale * CDouble::new(self.vals[i]);
                    let intval = nearest_integer(scaleval.to_f64()) as f64;
                    let delta = (scaleval - intval).to_f64();
                    self.vals[i] = intval;
                    if delta < 0.0 {
                        let u = ub(self.inds[i]);
                        if u == K_HIGHS_INF {
                            return None;
                        }
                        self.rhs -= delta * u;
                    } else {
                        let l = lb(self.inds[i]);
                        if l == -K_HIGHS_INF {
                            return None;
                        }
                        self.rhs -= delta * l;
                    }
                }
                self.rhs = (self.rhs + feastol).floor();
                if intscale * max_abs_value * feastol < 0.5 {
                    scale_smallest_val_to_one = false;
                    self.integral_coefficients = true;
                }
            }
            if scale_smallest_val_to_one {
                let mut min_abs_value = K_HIGHS_INF;
                for i in 0..self.rowlen {
                    min_abs_value = self.vals[i].abs().min(min_abs_value);
                }
                self.scale(min_abs_value - epsilon);
            }
        } else {
            self.scale(max_abs_value - epsilon);
        }
        out(self)
    }

    fn remove_zeros(&mut self) {
        let mut i = self.rowlen;
        while i > 0 {
            i -= 1;
            if self.vals[i] == 0.0 {
                self.rowlen -= 1;
                let r = self.rowlen;
                self.inds[i] = self.inds[r];
                self.vals[i] = self.vals[r];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counts;
    impl OpenNodeCounts for Counts {
        fn num_nodes_up(&self, col: usize) -> i64 {
            (col % 3) as i64
        }
        fn num_nodes_down(&self, col: usize) -> i64 {
            (col % 2) as i64
        }
    }

    /// `generate_conflict_core` + `postprocess_cut` (HiGHS `generateConflict`) on random
    /// proof rows `a . x <= b` over a mixed box: the cut must hold at every grid point of
    /// the global box that satisfies the proof (integers exhaustively, continuous on a
    /// fine grid), whatever the local box the node had.
    #[test]
    fn generate_conflict_cut_is_valid_for_the_proof_over_the_global_box() {
        let mut rng = HighsRandom::new(7);
        let (mut cuts, mut checks) = (0usize, 0usize);
        for _case in 0..400 {
            let n = 2 + (rng.integer_below(4) as usize);
            let mut inds = Vec::new();
            let mut vals = Vec::new();
            let (mut glb, mut gub, mut llb, mut lub, mut isi) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for j in 0..n {
                inds.push(j as i32);
                let v = (rng.integer_below(13) as f64 - 6.0) * 0.5
                    + 0.25 * (rng.integer_below(2) as f64);
                vals.push(if v == 0.0 { 1.0 } else { v });
                let int = rng.integer_below(4) != 0;
                isi.push(int);
                let lo = rng.integer_below(2) as f64 - 1.0;
                let hi = lo + 1.0 + rng.integer_below(3) as f64;
                glb.push(lo);
                gub.push(hi);
                // A local box inside the global one.
                let l2 = if int {
                    lo + rng.integer_below((hi - lo) as i32 + 1) as f64
                } else {
                    lo
                };
                let l2 = l2.min(hi);
                llb.push(l2);
                lub.push(hi.max(l2));
            }
            // A proof the local box violates: min activity over the local box exceeds rhs.
            let minact: f64 = (0..n)
                .map(|j| vals[j] * if vals[j] > 0.0 { llb[j] } else { lub[j] })
                .sum();
            let rhs = minact - 0.5 - 0.25 * rng.integer_below(3) as f64;
            let p = CutGenParams {
                feastol: 1e-6,
                epsilon: 1e-9,
                num_cols: n,
            };
            let mut gen = CutGenCore::seeded(p, 0, 0, 0);
            let Some(core) = gen
                .generate_conflict_core(&inds, &vals, rhs, &glb, &gub, &llb, &lub, &isi, &Counts)
            else {
                continue;
            };
            let lbf = |c: i32| glb[c as usize];
            let ubf = |c: i32| gub[c as usize];
            let isf = |c: i32| isi[c as usize];
            let Some(cut) = gen.postprocess_cut(core, &lbf, &ubf, &isf) else {
                continue;
            };
            cuts += 1;
            let crhs = cut.rhs.to_f64();
            // Enumerate the global box.
            let steps: Vec<Vec<f64>> = (0..n)
                .map(|j| {
                    let k = if isi[j] {
                        (gub[j] - glb[j]) as usize
                    } else {
                        8 * (gub[j] - glb[j]) as usize
                    };
                    (0..=k)
                        .map(|t| glb[j] + (gub[j] - glb[j]) * t as f64 / k as f64)
                        .collect()
                })
                .collect();
            let mut idx = vec![0usize; n];
            loop {
                let x: Vec<f64> = (0..n).map(|j| steps[j][idx[j]]).collect();
                let pa: f64 = (0..n).map(|j| vals[j] * x[j]).sum();
                if pa <= rhs + 1e-9 {
                    let ca: f64 = cut
                        .inds
                        .iter()
                        .zip(&cut.vals)
                        .map(|(&c, &v)| v * x[c as usize])
                        .sum();
                    checks += 1;
                    assert!(
                        ca <= crhs + 1e-6 * (1.0 + crhs.abs()),
                        "cut {:?} {:?} <= {crhs} cuts off {x:?} (proof {vals:?} <= {rhs}, \
                         box {glb:?}..{gub:?}, int {isi:?})",
                        cut.inds,
                        cut.vals
                    );
                }
                let mut j = 0;
                while j < n {
                    idx[j] += 1;
                    if idx[j] < steps[j].len() {
                        break;
                    }
                    idx[j] = 0;
                    j += 1;
                }
                if j == n {
                    break;
                }
            }
        }
        eprintln!("generate_conflict: {cuts} cuts, {checks} point checks");
        assert!(
            cuts > 20 && checks > 200,
            "probe did not fire: {cuts} cuts, {checks} checks"
        );
    }

    /// #2232: `fast_floor` must equal `f64::floor` in value everywhere, including where the
    /// `as i64` cast saturates (|x| >= 2^63) or maps NaN to 0. Before the fix,
    /// `fast_floor(-1e20)` overflowed `i64::MIN - 1` (a panic under overflow checks,
    /// `+9.2e18` in a release build).
    #[test]
    fn fast_floor_matches_floor_everywhere() {
        let two63 = 9_223_372_036_854_775_808.0f64;
        let mut xs = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            3.7,
            -2.5,
            -1.0,
            1e15 + 0.5,
            -(1e15 + 0.5),
            TWO_POW_52,
            -TWO_POW_52,
            TWO_POW_52 - 0.5,
            -(TWO_POW_52 - 0.5),
            1e20,
            -1e20,
            -9.3e18,
            two63,
            -two63,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
        ];
        // Every power of two either side of the cast's range edges.
        for e in 0..1024 {
            let v = 2f64.powi(e);
            xs.extend([v, -v, v * 1.5, -v * 1.5]);
        }
        let mut checked = 0usize;
        for &x in &xs {
            // By value: on `-0.0` the integer path returns `+0.0` (as it always did);
            // the two zeros compare equal and give the same `f0`.
            assert_eq!(fast_floor(x), x.floor(), "x = {x:e}");
            checked += 1;
        }
        assert!(fast_floor(f64::NAN).is_nan());
        checked += 1;
        assert_eq!(checked, xs.len() + 1);
        assert!(checked > 4000, "only {checked} values checked");
    }

    #[test]
    fn exact_i64_refuses_where_the_cast_would_saturate() {
        assert_eq!(exact_i64(-3.7), Some(-3));
        assert_eq!(exact_i64(1e18), Some(1_000_000_000_000_000_000));
        assert_eq!(
            exact_i64(-(EXACT_I64_BOUND - 1024.0)),
            Some(-(1i64 << 62) + 1024)
        );
        for x in [
            EXACT_I64_BOUND,
            -EXACT_I64_BOUND,
            1e20,
            -1e20,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            assert_eq!(exact_i64(x), None, "x = {x:e}");
        }
    }

    /// #2232: on badly scaled pure-integer rows (coefficient magnitudes 1e-3..1e20)
    /// the cut generator must neither panic nor return a cut that some integer point
    /// satisfying the base row violates. Checked by enumeration on small boxes.
    #[test]
    fn badly_scaled_rows_give_valid_cuts_or_none() {
        let mut seed = 0x2232_u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 11
        };
        let p = CutGenParams {
            feastol: 1e-6,
            epsilon: 1e-9,
            num_cols: 8,
        };
        let mut rows_tried = 0usize;
        let mut cuts = 0usize;
        let mut points_checked = 0usize;
        let mut paths = std::collections::HashSet::new();
        for _ in 0..20_000 {
            let n = 2 + (rnd() % 3) as usize;
            let general = rnd() % 2 == 0;
            let upper: Vec<f64> = (0..n)
                .map(|_| {
                    if general {
                        1.0 + (rnd() % 3) as f64
                    } else {
                        1.0
                    }
                })
                .collect();
            let vals: Vec<f64> = (0..n)
                .map(|_| {
                    let e = (rnd() % 24) as i32 - 3;
                    let mant = 1.0 + (rnd() % 1000) as f64 / 997.0;
                    let s = if rnd() % 4 == 0 { -1.0 } else { 1.0 };
                    s * mant * 10f64.powi(e)
                })
                .collect();
            let solval: Vec<f64> = (0..n)
                .map(|i| upper[i] * (rnd() % 1001) as f64 / 1000.0)
                .collect();
            let maxact: f64 = (0..n).map(|i| vals[i].max(0.0) * upper[i]).sum();
            let minact: f64 = (0..n).map(|i| vals[i].min(0.0) * upper[i]).sum();
            let rhs = minact + (maxact - minact) * (rnd() % 1000) as f64 / 1000.0;
            let row = BaseRow {
                inds: (0..n as i32).collect(),
                vals: vals.clone(),
                rhs,
                upper: upper.clone(),
                solval,
                col_integral: vec![true; n],
                ints_positive: false,
            };
            rows_tried += 1;
            let mut g = CutGenCore::new(p, HighsRandom::new(rnd() as u32));
            let Some(cut) = g.generate_cut_core(&row, false) else {
                continue;
            };
            cuts += 1;
            paths.insert(format!("{:?}", g.last_path()));
            let crhs = cut.rhs.to_f64();
            // Enumerate the integer box.
            let mut x = vec![0i64; n];
            loop {
                let (mut act, mut scale) = (0.0f64, rhs.abs());
                for i in 0..n {
                    act += vals[i] * x[i] as f64;
                    scale += (vals[i] * x[i] as f64).abs();
                }
                // Only points clearly feasible for the base row count.
                if act <= rhs - 1e-9 * scale.max(1.0) {
                    let (mut cact, mut cscale) = (0.0f64, crhs.abs() + 1.0);
                    for (k, &j) in cut.inds.iter().enumerate() {
                        let t = cut.vals[k] * x[j as usize] as f64;
                        cact += t;
                        cscale += t.abs();
                    }
                    assert!(
                        cact <= crhs + 1e-6 * cscale,
                        "invalid cut {cut:?} ({:?}) cuts off x = {x:?} of row \
                         vals={vals:?} rhs={rhs:e} upper={upper:?}",
                        g.last_path()
                    );
                    points_checked += 1;
                }
                let mut i = 0;
                while i < n {
                    x[i] += 1;
                    if x[i] as f64 <= upper[i] {
                        break;
                    }
                    x[i] = 0;
                    i += 1;
                }
                if i == n {
                    break;
                }
            }
        }
        eprintln!("rows {rows_tried} cuts {cuts} points {points_checked} paths {paths:?}");
        assert_eq!(paths.len(), 4, "not every cut path was reached: {paths:?}");
        assert!(cuts > 500, "only {cuts} cuts generated");
        assert!(
            points_checked > 5000,
            "only {points_checked} points checked"
        );
    }

    /// Runs the cut generator on one integer row and returns the cut, the path it
    /// took and the cover-lifting refusals recorded on the way.
    fn cut_of(
        feastol: f64,
        vals: &[f64],
        rhs: f64,
        upper: &[f64],
        solval: &[f64],
        lift_work_limit: u64,
    ) -> (Option<CoreCut>, CutPath, Vec<LiftRefusal>) {
        let n = vals.len();
        let row = BaseRow {
            inds: (0..n as i32).collect(),
            vals: vals.to_vec(),
            rhs,
            upper: upper.to_vec(),
            solval: solval.to_vec(),
            col_integral: vec![true; n],
            ints_positive: false,
        };
        let p = CutGenParams {
            feastol,
            epsilon: feastol * 1e-3,
            num_cols: 8,
        };
        LIFT_REFUSALS.with(|r| r.borrow_mut().clear());
        let mut g = CutGenCore::new(p, HighsRandom::new(7));
        g.lift_work_limit = lift_work_limit;
        let cut = g.generate_cut_core(&row, false);
        let refusals = LIFT_REFUSALS.with(|r| r.borrow().clone());
        (cut, g.last_path(), refusals)
    }

    /// Asserts `cut` keeps every point of `grid` (one list of values per variable)
    /// that satisfies the base row; returns how many feasible points were checked.
    fn assert_keeps_feasible_points(
        cut: &CoreCut,
        vals: &[f64],
        rhs: f64,
        grid: &[Vec<f64>],
    ) -> usize {
        let n = vals.len();
        let crhs = cut.rhs.to_f64();
        let mut idx = vec![0usize; n];
        let mut checked = 0usize;
        loop {
            let x: Vec<f64> = (0..n).map(|i| grid[i][idx[i]]).collect();
            let act: f64 = (0..n).map(|i| vals[i] * x[i]).sum();
            if act <= rhs {
                let (mut cact, mut cscale) = (0.0f64, crhs.abs() + 1.0);
                for (k, &j) in cut.inds.iter().enumerate() {
                    let t = cut.vals[k] * x[j as usize];
                    cact += t;
                    cscale += t.abs();
                }
                assert!(cact <= crhs + 1e-9 * cscale, "{cut:?} cuts off x = {x:?}");
                checked += 1;
            }
            let mut i = 0;
            while i < n {
                idx[i] += 1;
                if idx[i] < grid[i].len() {
                    break;
                }
                idx[i] = 0;
                i += 1;
            }
            if i == n {
                return checked;
            }
        }
    }

    /// #2232 review: the knapsack-cover refusal must actually run. After the row is
    /// scaled (largest |coefficient| in [0.5, 1)) only coefficients above
    /// `10 * feastol` count as integral, so `z / abar <= 1 / (10 * feastol)`: about
    /// 1e5 at `feastol = 1e-6`, far below 2^62. The refusal (like the old `i32`
    /// saturation it replaces) is reachable only with a tiny `feastol`, used here.
    #[test]
    fn knapsack_cover_refuses_out_of_range_lifting() {
        let vals = [1.0, 1.0, 1.0, 1e19];
        let (cut, path, refusals) = cut_of(
            1e-30,
            &vals,
            2.5,
            &[1.0; 4],
            &[0.9, 0.9, 0.7, 0.0],
            MAX_LIFT_WORK,
        );
        assert_eq!(refusals, vec![LiftRefusal::KnapsackRange]);
        assert_ne!(path, CutPath::KnapsackCover);
        let cut = cut.expect("the c-MIR fallback still produces a cut on this row");
        assert_eq!(path, CutPath::Cmir);
        let binary = vec![0.0, 1.0];
        let checked = assert_keeps_feasible_points(&cut, &vals, 2.5, &vec![binary; 4]);
        assert_eq!(
            checked, 7,
            "x3 = 0 and at most two of x0..x2 at 1: 1 + 3 + 3"
        );

        // Control: the same row at the ordinary tolerance never reaches the lifting.
        let (_, _, refusals) = cut_of(
            1e-6,
            &vals,
            2.5,
            &[1.0; 4],
            &[0.9, 0.9, 0.7, 0.0],
            MAX_LIFT_WORK,
        );
        assert!(refusals.is_empty(), "{refusals:?}");
    }

    /// #2232 review: a general-integer cover variable with an upper bound >= 2^62
    /// (here 1e19, at the ordinary `feastol`) makes the mixed-integer cover refuse;
    /// the old `as i32` cast saturated it to 2^31 - 1. The c-MIR cut that replaces it
    /// must keep every feasible point.
    #[test]
    fn mixed_integer_cover_refuses_out_of_range_upper_bound() {
        let vals = [1.0, 1.0, 1.0];
        let upper = [1e19, 3.0, 3.0];
        let solval = [1.5, 1.4, 0.0];
        let (cut, path, refusals) = cut_of(1e-6, &vals, 2.5, &upper, &solval, MAX_LIFT_WORK);
        assert_eq!(refusals, vec![LiftRefusal::MixedIntegerRange]);
        assert_eq!(path, CutPath::Cmir);
        let cut = cut.expect("the c-MIR fallback still produces a cut on this row");
        let mut x0: Vec<f64> = (0..=5).map(f64::from).collect();
        x0.extend([1e18, 1e19]);
        let small: Vec<f64> = (0..=3).map(f64::from).collect();
        let checked = assert_keeps_feasible_points(&cut, &vals, 2.5, &[x0, small.clone(), small]);
        assert_eq!(checked, 10, "integer points with x0 + x1 + x2 <= 2");

        // Control: with a small upper bound the same row is lifted, not refused.
        let (_, path, refusals) =
            cut_of(1e-6, &vals, 2.5, &[3.0, 3.0, 3.0], &solval, MAX_LIFT_WORK);
        assert!(refusals.is_empty(), "{refusals:?}");
        assert_ne!(path, CutPath::None);
    }

    /// #2232 review: the lifting-work budget. A 200k-row random search at
    /// `feastol = 1e-30` found no row that exceeds `MAX_LIFT_WORK`, so the budget is
    /// lowered here to drive the branch: the refusal must fire, and the cut that
    /// replaces the lifted one must stay valid.
    #[test]
    fn mixed_integer_cover_refuses_past_the_work_budget() {
        let vals = [1.0, 1.0, 1.0];
        let upper = [2.0, 2.0, 2.0];
        let solval = [0.9, 0.9, 0.7];
        let grid: Vec<Vec<f64>> = vec![(0..=2).map(f64::from).collect(); 3];

        let (cut, path, refusals) = cut_of(1e-6, &vals, 2.5, &upper, &solval, MAX_LIFT_WORK);
        assert!(refusals.is_empty(), "{refusals:?}");
        assert_eq!(
            path,
            CutPath::MixedIntegerCover,
            "control must take the lifted path"
        );
        let checked = assert_keeps_feasible_points(&cut.unwrap(), &vals, 2.5, &grid);
        assert_eq!(checked, 10);

        let (cut, path, refusals) = cut_of(1e-6, &vals, 2.5, &upper, &solval, 0);
        assert_eq!(refusals, vec![LiftRefusal::MixedIntegerWork]);
        assert_ne!(path, CutPath::MixedIntegerCover);
        if let Some(cut) = cut {
            assert_eq!(assert_keeps_feasible_points(&cut, &vals, 2.5, &grid), 10);
        }
    }
}
