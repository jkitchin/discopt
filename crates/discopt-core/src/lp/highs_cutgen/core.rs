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
}

#[inline(always)]
fn fast_floor(x: f64) -> f64 {
    let t = x as i64;
    (t - ((x < t as f64) as i64)) as f64
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

    fn determine_cover(&mut self) -> bool {
        let feastol = self.p.feastol;
        if self.rhs <= 10.0 * feastol {
            return false;
        }
        self.cover.clear();
        for j in 0..self.rowlen {
            if self.isintegral[j] == 0 {
                continue;
            }
            if self.solval[j] <= feastol {
                continue;
            }
            self.cover.push(j);
        }
        let max_cover_size = self.cover.len();
        let r = self.rng.integer();
        self.coverweight = CDouble::new(0.0);

        let (solval, upper, vals, inds) = (&self.solval, &self.upper, &self.vals, &self.inds);
        let mut coversize = partition(&mut self.cover, |&j: &usize| {
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

    fn separate_lifted_knapsack_cover(&mut self) {
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
        let mut g = |z: f64| -> f64 {
            let hfrac = z / abar;
            let mut coef = 0.0;
            let mut h = (hfrac + 0.5).floor() as i32;
            if h != 0
                && (hfrac - h as f64).abs() * 1.0f64.max(abar) <= epsilon
                && h <= cplussize - 1
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

        self.rhs = CDouble::new((coversize as i64 - 1) as f64);
        for i in 0..self.rowlen {
            if self.vals[i] == 0.0 {
                continue;
            }
            if coverflag[i] == -1 {
                self.vals[i] = 1.0;
            } else {
                self.vals[i] = g(self.vals[i]);
            }
        }
        if halfintegral {
            self.rhs *= 2.0;
            for i in 0..self.rowlen {
                self.vals[i] *= 2.0;
            }
        }
        self.integral_support = true;
        self.integral_coefficients = true;
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
        let kmin = (eta - upperl - 0.5).floor() as i32;

        let phi_l = |av: f64| -> f64 {
            let mut k: i64 = ((av / al.to_f64()) as i64).min(-1);
            while k >= kmin as i64 {
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

        let kmax: i64 = (upperl - eta + 0.5).floor() as i64;
        let upper = &self.upper;
        let cover = &self.cover;
        let gamma_l = |z: f64| -> f64 {
            for i in 0..cplussize {
                let upperi = upper[cover[i]] as i32;
                let mut h: i32 = 0;
                while h <= upperi {
                    let mih = m[i] + (h as f64) * a[i];
                    let uih = u[i] + h as f64;
                    let mihplusdeltai = mih + a[i] - cplusthreshold;
                    if z <= mihplusdeltai {
                        return (uih * ulminusetaplusone * (al - r)).to_f64();
                    }
                    let mut k: i64 = (((z - mihplusdeltai) / al).to_f64() as i64) - 1;
                    while k <= kmax {
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
            let mut p: i64 = (((z - m[cplussize]) / al).to_f64() as i64) - 1;
            loop {
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
        if self.determine_cover() {
            if !has_continuous && !has_general_ints {
                self.separate_lifted_knapsack_cover();
                success = true;
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
