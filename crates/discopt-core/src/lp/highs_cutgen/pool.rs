//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text.
//!
//! Faithful port of `mip/HighsCutPool.cpp` (single-threaded paths) and the parts of
//! `HighsDynamicRowMatrix` and `HighsHashHelpers` it depends on: row-slot reuse
//! (which fixes the row index every selection tiebreak hashes), the bucketed
//! duplicate test (`compute_cut_hash` = `vector_hash` + `double_hash_code`), the
//! scoring and parallelism-filtered selection of `separate`, ageing, and the
//! propagation-row bookkeeping.
//!
//! Domain propagation from pool cuts (HiGHS's `CutpoolPropagation`) is outside this
//! module: a [`PoolObserver`] is told when cuts are added or deleted, and the caller
//! reports propagated cuts back through [`CutPool::reset_age`].

use std::collections::{BTreeSet, HashMap};

use super::cdouble::CDouble;
use super::random::pair_hash;
use super::stdalgo::{pdqsort, pdqsort_branchless, upper_bound};

const INF: f64 = f64::INFINITY;
const M61: u64 = 0x1fff_ffff_ffff_ffff;
const GOLDEN_RATIO_RECIPROCAL: f64 = 0.61803398874989484;

/// `HighsHashHelpers::multiply_modM61`.
fn multiply_mod_m61(a: u64, b: u64) -> u64 {
    let (ahi, bhi) = (a >> 32, b >> 32);
    let (alo, blo) = (a & 0xffff_ffff, b & 0xffff_ffff);
    let term_64 = ahi.wrapping_mul(bhi);
    let term_32 = ahi.wrapping_mul(blo).wrapping_add(bhi.wrapping_mul(alo));
    let mut term_0 = alo.wrapping_mul(blo);
    term_0 = (term_0 & M61).wrapping_add(term_0 >> 61);
    term_0 = term_0.wrapping_add(((term_32 >> 29).wrapping_add(term_32 << 32)) & M61);
    let ab61 = (term_64 << 3) | (term_0 >> 61);
    let mut result = (term_0 & M61).wrapping_add(ab61);
    if result >= M61 {
        result -= M61;
    }
    result
}

const C64: [u64; 64] = [
    0xc8497d2a400d9551,
    0x80c8963be3e4c2f3,
    0x042d8680e260ae5b,
    0x8a183895eeac1536,
    0xa94e9c75f80ad6de,
    0x7e92251dec62835e,
    0x07294165cb671455,
    0x89b0f6212b0a4292,
    0x31900011b96bf554,
    0xa44540f8eee2094f,
    0xce7ffd372e4c64fc,
    0x51c9d471bfe6a10f,
    0x758c2a674483826f,
    0xf91a20abe63f8b02,
    0xc2a069024a1fcc6f,
    0xd5bb18b70c5dbd59,
    0xd510adac6d1ae289,
    0x571d069b23050a79,
    0x60873b8872933e06,
    0x780481cc19670350,
    0x7a48551760216885,
    0xb5d68b918231e6ca,
    0xa7e5571699aa5274,
    0x7b6d309b2cfdcf01,
    0x04e77c3d474daeff,
    0x4dbf099fd7247031,
    0x5d70dca901130beb,
    0x9f8b5f0df4182499,
    0x293a74c9686092da,
    0xd09bdab6840f52b3,
    0xc05d47f3ab302263,
    0x6b79e62b884b65d6,
    0xa581106fc980c34d,
    0xf081b7145ea2293e,
    0xfb27243dd7c3f5ad,
    0x5211bf8860ea667f,
    0x9455e65cb2385e7f,
    0x0dfaf6731b449b33,
    0x4ec98b3c6f5e68c7,
    0x007bfd4a42ae936b,
    0x65c93061f8674518,
    0x640816f17127c5d1,
    0x6dd4bab17b7c3a74,
    0x34d9268c256fa1ba,
    0x0b4d0c6b5b50d7f4,
    0x30aa965bc9fadaff,
    0xc0ac1d0c2771404d,
    0xc5e64509abb76ef2,
    0xd606b11990624a36,
    0x0d3f05d242ce2fb7,
    0x469a803cb276fe32,
    0xa4a44d177a3e23f4,
    0xb9d9a120dcc1ca03,
    0x2e15af8165234a2e,
    0x10609ba2720573d4,
    0xaa4191b60368d1d5,
    0x333dd2300bc57762,
    0xdf6ec48f79fb402f,
    0x5ed20fcef1b734fa,
    0x4c94924ec8be21ee,
    0x5abe6ad9d131e631,
    0xbe10136a522e602d,
    0x53671115c340e779,
    0x9f392fe43e2144da,
];

/// `HighsHashHelpers::vector_hash` over the raw bytes of `bytes`. The pair buffer
/// persists across pairs, so a short trailing pair keeps the previous pair's high
/// bytes, exactly as the C++ `memcpy` into a reused `std::array` does.
pub fn vector_hash_bytes(bytes: &[u8]) -> u64 {
    let mut pair = [0u8; 8];
    let mut hash: u64 = 0;
    let mut k: usize = 0;
    let mut ptr = 0usize;
    let end = bytes.len();
    while ptr != end {
        let num_bytes = (end - ptr).min(256);
        let num_pairs = num_bytes.div_ceil(8);
        let last_pair_bytes = num_bytes - (num_pairs - 1) * 8;
        let mut chunkhash = [0u64; 2];
        if num_pairs == 32 && hash != 0 {
            if hash >= M61 {
                hash -= M61;
            }
            hash = multiply_mod_m61(hash, C64[k & 63] & M61);
            k += 1;
        }
        let mut n = num_pairs;
        while n >= 1 {
            let b = if n == 1 { last_pair_bytes } else { 8 };
            pair[..b].copy_from_slice(&bytes[ptr..ptr + b]);
            let p0 = u32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]);
            let p1 = u32::from_le_bytes([pair[4], pair[5], pair[6], pair[7]]);
            chunkhash[n & 1] = chunkhash[n & 1].wrapping_add(pair_hash(32 - n, p0, p1));
            ptr += b;
            n -= 1;
        }
        hash = hash.wrapping_add((chunkhash[0] >> 3) ^ (chunkhash[1] >> 32));
    }
    // `return hash * fibonacci_muliplier();`
    hash.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// `HighsHashHelpers::double_hash_code`.
pub fn double_hash_code(val: f64) -> u32 {
    let (hashbits, exponent) = libm::frexp(val * GOLDEN_RATIO_RECIPROCAL);
    let mut h = (exponent as i16) as u16 as u32;
    h = (h << 16) | ((libm::ldexp(hashbits, 15) as i16) as u16 as u32);
    h
}

/// `compute_cut_hash` (HighsCutPool.cpp).
pub fn compute_cut_hash(inds: &[i32], vals: &[f64], maxabscoef: f64) -> u64 {
    let scale = 1.0 / maxabscoef;
    let codes: Vec<u8> = vals
        .iter()
        .flat_map(|&v| double_hash_code(scale * v).to_le_bytes())
        .collect();
    let ib: Vec<u8> = inds.iter().flat_map(|&i| i.to_le_bytes()).collect();
    vector_hash_bytes(&ib) ^ (vector_hash_bytes(&codes) >> 32)
}

/// `HighsHashHelpers::hash(uint64_t)`.
fn hash_u64(v: u64) -> u64 {
    let (b0, b1) = (v as u32, (v >> 32) as u32);
    pair_hash(1, b0, b1) ^ (pair_hash(0, b0, b1) >> 32)
}

// ------------------------------------------------------------ row storage

/// The storage half of `HighsDynamicRowMatrix`: rows in reusable slots and
/// reusable row indices (deleted indices are reused last-in first-out).
/// Column linking is kept as a per-row flag; the column lists themselves are only
/// used by HiGHS's propagation, which lives outside this module.
#[derive(Default, Clone)]
pub struct DynamicRowMatrix {
    ar_index: Vec<i32>,
    ar_value: Vec<f64>,
    ar_range: Vec<(i64, i64)>,
    cols_linked: Vec<bool>,
    freespaces: BTreeSet<(i64, i64)>,
    deleted_rows: Vec<i32>,
}

impl DynamicRowMatrix {
    /// `addRow` (indices must already be sorted).
    pub fn add_row(&mut self, inds: &[i32], vals: &[f64], link_cols: bool) -> i32 {
        let len = inds.len() as i64;
        let start;
        let slot = self.freespaces.range((len, -1)..).next().copied();
        match slot {
            None => {
                start = self.ar_index.len() as i64;
                let end = start + len;
                self.ar_index.resize(end as usize, 0);
                self.ar_value.resize(end as usize, 0.0);
            }
            Some(free) => {
                self.freespaces.remove(&free);
                start = free.1;
                if free.0 > len {
                    self.freespaces.insert((free.0 - len, start + len));
                }
            }
        }
        let end = start + len;
        let rowindex = match self.deleted_rows.pop() {
            None => {
                self.ar_range.push((start, end));
                self.cols_linked.push(link_cols);
                (self.ar_range.len() - 1) as i32
            }
            Some(r) => {
                self.ar_range[r as usize] = (start, end);
                self.cols_linked[r as usize] = link_cols;
                r
            }
        };
        for k in 0..inds.len() {
            self.ar_index[start as usize + k] = inds[k];
            self.ar_value[start as usize + k] = vals[k];
        }
        rowindex
    }
    /// `removeRow`.
    pub fn remove_row(&mut self, row: i32) {
        let (start, end) = self.ar_range[row as usize];
        self.deleted_rows.push(row);
        self.freespaces.insert((end - start, start));
        self.ar_range[row as usize] = (-1, -1);
    }
    /// `unlinkColumns`.
    pub fn unlink_columns(&mut self, row: i32) {
        self.cols_linked[row as usize] = false;
    }
    /// `columnsLinked`.
    pub fn columns_linked(&self, row: i32) -> bool {
        self.cols_linked[row as usize]
    }
    /// `getNumRows` (including deleted slots).
    pub fn num_rows(&self) -> usize {
        self.ar_range.len()
    }
    /// `getNumDelRows`.
    pub fn num_del_rows(&self) -> usize {
        self.deleted_rows.len()
    }
    /// The row's (indices, values); empty for a deleted row.
    pub fn row(&self, row: i32) -> (&[i32], &[f64]) {
        let (s, e) = self.ar_range[row as usize];
        if s < 0 {
            return (&[], &[]);
        }
        (
            &self.ar_index[s as usize..e as usize],
            &self.ar_value[s as usize..e as usize],
        )
    }
}

// --------------------------------------------------------------- the pool

/// Told about pool changes (HiGHS's `CutpoolPropagation::cutAdded/cutDeleted`).
pub trait PoolObserver {
    /// A cut was added; `propagate` says whether it is linked for propagation.
    fn cut_added(&mut self, _row: i32, _propagate: bool) {}
    /// A cut was deleted (`deleted_only_for_propagation`: unlinked, still in pool).
    fn cut_deleted(&mut self, _row: i32, _deleted_only_for_propagation: bool) {}
}

/// A no-op observer.
pub struct NoObserver;
impl PoolObserver for NoObserver {}

/// `HighsCutPool` (one global pool, single-threaded).
pub struct CutPool {
    matrix: DynamicRowMatrix,
    rhs: Vec<f64>,
    ages: Vec<i16>,
    num_lps: Vec<i16>,
    rownormalization: Vec<f64>,
    maxabscoef: Vec<f64>,
    rowintegral: Vec<bool>,
    hash_to_cut: HashMap<u64, Vec<i32>>,
    prop_rows: BTreeSet<(i16, i32)>,
    best_observed_score: f64,
    min_score_factor: f64,
    min_density_lim: f64,
    agelim: i32,
    softlimit: i64,
    num_lp_cuts: i64,
    num_prop_nzs: i64,
    num_prop_rows: i64,
    age_distribution: Vec<i64>,
    model_nnz: i64,
    model_rows: i64,
}

/// One cut selected by `separate`: its pool index, row and rhs.
#[derive(Clone, Debug)]
pub struct SelectedCut {
    /// Pool row index.
    pub index: i32,
    /// Column indices (sorted).
    pub inds: Vec<i32>,
    /// Coefficients.
    pub vals: Vec<f64>,
    /// Right-hand side (`<=`).
    pub rhs: f64,
}

impl CutPool {
    /// `HighsCutPool(ncols, agelim, softlimit, index)`; `model_nnz`/`model_rows`
    /// are the original model's (`mipsolver.numNonzero()`, `numRow()`).
    pub fn new(ncols: usize, agelim: i32, softlimit: i64, model_nnz: i64, model_rows: i64) -> Self {
        CutPool {
            matrix: DynamicRowMatrix::default(),
            rhs: Vec::new(),
            ages: Vec::new(),
            num_lps: Vec::new(),
            rownormalization: Vec::new(),
            maxabscoef: Vec::new(),
            rowintegral: Vec::new(),
            hash_to_cut: HashMap::new(),
            prop_rows: BTreeSet::new(),
            best_observed_score: 0.0,
            min_score_factor: 0.9,
            min_density_lim: 0.1 * ncols as f64,
            agelim,
            softlimit,
            num_lp_cuts: 0,
            num_prop_nzs: 0,
            num_prop_rows: 0,
            age_distribution: vec![0; agelim as usize + 1],
            model_nnz,
            model_rows,
        }
    }

    /// `getNumCuts()`.
    pub fn num_cuts(&self) -> i64 {
        (self.matrix.num_rows() - self.matrix.num_del_rows()) as i64
    }
    /// `getNumAvailableCuts()`.
    pub fn num_available_cuts(&self) -> i64 {
        self.num_cuts() - self.num_lp_cuts
    }
    /// The stored cut (sorted indices, values) and its rhs.
    pub fn cut(&self, row: i32) -> (&[i32], &[f64], f64) {
        let (i, v) = self.matrix.row(row);
        (i, v, self.rhs[row as usize])
    }
    /// `getMaxAbsCutCoef`.
    pub fn max_abs_cut_coef(&self, row: i32) -> f64 {
        self.maxabscoef[row as usize]
    }
    /// `cutIsIntegral`.
    pub fn cut_is_integral(&self, row: i32) -> bool {
        self.rowintegral[row as usize]
    }
    /// Number of row slots (including deleted ones).
    pub fn num_slots(&self) -> usize {
        self.matrix.num_rows()
    }
    /// `columnsLinked(row)`: is the cut used for propagation?
    pub fn is_propagated(&self, row: i32) -> bool {
        self.matrix.columns_linked(row)
    }
    /// Is this slot a live cut?
    pub fn is_live(&self, row: i32) -> bool {
        self.rhs[row as usize] != INF
    }
    /// The (bestObservedScore, minScoreFactor) pair, for replay checks.
    pub fn score_state(&self) -> (f64, f64) {
        (self.best_observed_score, self.min_score_factor)
    }

    fn row_len(&self, row: i32) -> i64 {
        self.matrix.row(row).0.len() as i64
    }

    fn is_duplicate(&self, hash: u64, norm: f64, inds: &[i32], vals: &[f64]) -> bool {
        let Some(bucket) = self.hash_to_cut.get(&hash) else {
            return false;
        };
        for &row in bucket {
            let (ri, rv) = self.matrix.row(row);
            if ri.len() != inds.len() || ri != inds {
                continue;
            }
            let mut dotprod = 0.0f64;
            for i in 0..inds.len() {
                dotprod += vals[i] * rv[i];
            }
            let parallelism = dotprod * self.rownormalization[row as usize] * norm;
            if parallelism >= 1.0 - 1e-6 {
                return true;
            }
        }
        false
    }

    /// `getParallelism(row1, row2)`.
    pub fn parallelism(&self, row1: i32, row2: i32) -> f64 {
        let (i1, v1) = self.matrix.row(row1);
        let (i2, v2) = self.matrix.row(row2);
        let (mut a, mut b) = (0usize, 0usize);
        let mut dotprod = 0.0f64;
        while a != i1.len() && b != i2.len() {
            if i1[a] < i2[b] {
                a += 1;
            } else if i2[b] < i1[a] {
                b += 1;
            } else {
                dotprod += v1[a] * v2[b];
                a += 1;
                b += 1;
            }
        }
        dotprod * self.rownormalization[row1 as usize] * self.rownormalization[row2 as usize]
    }

    /// `addCut(...)`: the new cut's index, or -1 for a duplicate.
    pub fn add_cut(
        &mut self,
        obs: &mut dyn PoolObserver,
        inds: &[i32],
        vals: &[f64],
        rhs: f64,
        integral: bool,
        mut propagate: bool,
        is_conflict: bool,
    ) -> i32 {
        let rlen = inds.len();
        let mut norm = 0.0f64;
        let mut maxabscoef = 0.0f64;
        let mut sort_buffer: Vec<(i32, f64)> = Vec::with_capacity(rlen);
        for i in 0..rlen {
            norm += vals[i] * vals[i];
            maxabscoef = maxabscoef.max(vals[i].abs());
            sort_buffer.push((inds[i], vals[i]));
        }
        pdqsort_branchless(&mut sort_buffer, |a: &(i32, f64), b: &(i32, f64)| a.0 < b.0);
        let sinds: Vec<i32> = sort_buffer.iter().map(|p| p.0).collect();
        let svals: Vec<f64> = sort_buffer.iter().map(|p| p.1).collect();
        let h = compute_cut_hash(&sinds, &svals, maxabscoef);
        let normalization = 1.0 / norm.sqrt();
        if self.is_duplicate(h, normalization, &sinds, &svals) {
            return -1;
        }
        if propagate {
            let new_prop_nzs = self.num_prop_nzs + rlen as i64;
            let avg_model_nzs = self.model_nnz as f64 / self.model_rows as f64;
            let new_avg_prop_nzs = new_prop_nzs as f64 / (self.num_prop_rows + 1) as f64;
            const ALPHA: f64 = 2.0;
            let lim = (ALPHA * avg_model_nzs).max(self.min_density_lim);
            let too_dense = if is_conflict {
                new_avg_prop_nzs > lim
            } else {
                rlen as f64 >= lim
            };
            if too_dense {
                propagate = false;
            } else {
                self.num_prop_rows += 1;
                self.num_prop_nzs = new_prop_nzs;
            }
        }
        let mut excess = self.num_prop_nzs - 2 * self.model_nnz;
        if excess > 0 {
            let mut drop: Vec<(i16, i32)> = Vec::new();
            for &(age, row) in self.prop_rows.iter().rev() {
                if excess <= 0 {
                    break;
                }
                let len = self.row_len(row);
                excess -= len;
                self.num_prop_nzs -= len;
                self.num_prop_rows -= 1;
                drop.push((age, row));
            }
            for &(age, row) in &drop {
                self.matrix.unlink_columns(row);
                obs.cut_deleted(row, true);
                self.prop_rows.remove(&(age, row));
            }
        }
        let rowindex = self.matrix.add_row(&sinds, &svals, propagate);
        self.hash_to_cut.entry(h).or_default().push(rowindex);
        let r = rowindex as usize;
        if r == self.rhs.len() {
            self.rhs.push(0.0);
            self.ages.push(0);
            self.num_lps.push(0);
            self.rownormalization.push(0.0);
            self.maxabscoef.push(0.0);
            self.rowintegral.push(false);
        }
        self.rhs[r] = rhs;
        self.ages[r] = 0i32.max(self.agelim - 5) as i16;
        self.age_distribution[self.ages[r] as usize] += 1;
        self.rowintegral[r] = integral;
        self.num_lps[r] = 0;
        if propagate {
            self.prop_rows.insert((self.ages[r], rowindex));
        }
        self.rownormalization[r] = normalization;
        self.maxabscoef[r] = maxabscoef;
        obs.cut_added(rowindex, propagate);
        rowindex
    }

    /// `resetAge(cut)`.
    pub fn reset_age(&mut self, cut: i32) {
        let c = cut as usize;
        if self.ages[c] > 0 {
            if self.matrix.columns_linked(cut) {
                self.prop_rows.remove(&(self.ages[c], cut));
                self.prop_rows.insert((0, cut));
            }
            self.age_distribution[self.ages[c] as usize] -= 1;
            self.age_distribution[0] += 1;
            self.ages[c] = 0;
        }
    }

    /// `lpCutRemoved(cut)`.
    pub fn lp_cut_removed(&mut self, cut: i32) {
        let c = cut as usize;
        let n = self.num_lps[c];
        self.num_lps[c] -= 1;
        if n > 1 {
            return;
        }
        if self.matrix.columns_linked(cut) {
            self.prop_rows.remove(&(self.ages[c], cut));
            self.prop_rows.insert((1, cut));
        }
        self.ages[c] = 1;
        self.num_lp_cuts -= 1;
        self.age_distribution[1] += 1;
    }

    fn delete_cut(&mut self, obs: &mut dyn PoolObserver, i: i32, is_propagated: bool) {
        obs.cut_deleted(i, false);
        if is_propagated {
            self.num_prop_rows -= 1;
            self.num_prop_nzs -= self.row_len(i);
        }
        self.matrix.remove_row(i);
        self.ages[i as usize] = -1;
        self.rhs[i as usize] = INF;
    }

    /// `performAging()`.
    pub fn perform_aging(&mut self, obs: &mut dyn PoolObserver) {
        let end = self.matrix.num_rows();
        let mut agelim = self.agelim;
        let mut num_available = self.num_available_cuts();
        while agelim > 5 && num_available > self.softlimit {
            num_available -= self.age_distribution[agelim as usize];
            agelim -= 1;
        }
        for i in 0..end {
            let row = i as i32;
            if self.num_lps[i] > 0 && self.ages[i] >= 0 {
                self.age_distribution[self.ages[i] as usize] -= 1;
                if self.matrix.columns_linked(row) {
                    self.prop_rows.remove(&(self.ages[i], row));
                    self.prop_rows.insert((-1, row));
                }
                self.ages[i] = -1;
                self.num_lp_cuts += 1;
            } else if self.num_lps[i] == 0 && self.ages[i] == -1 && self.rhs[i] != INF {
                if self.matrix.columns_linked(row) {
                    self.prop_rows.remove(&(self.ages[i], row));
                    self.prop_rows.insert((1, row));
                }
                self.ages[i] = 1;
                self.num_lp_cuts -= 1;
                self.age_distribution[1] += 1;
                continue;
            }
            if self.ages[i] < 0 {
                continue;
            }
            let is_propagated = self.matrix.columns_linked(row);
            if is_propagated {
                self.prop_rows.remove(&(self.ages[i], row));
            }
            self.age_distribution[self.ages[i] as usize] -= 1;
            self.ages[i] += 1;
            if self.ages[i] as i32 > agelim {
                self.delete_cut(obs, row, is_propagated);
            } else {
                if is_propagated {
                    self.prop_rows.insert((self.ages[i], row));
                }
                self.age_distribution[self.ages[i] as usize] += 1;
            }
        }
    }

    /// `separate(sol, domain, cutset, feastol)` for one pool: score the pool's
    /// violated cuts and select from them, skipping any too parallel (> 0.1) to a cut
    /// already in `cutset` (pool indices selected earlier this round). Selected cuts
    /// are appended to `cutset` and returned with their rows.
    pub fn separate(
        &mut self,
        obs: &mut dyn PoolObserver,
        sol: &[f64],
        col_lower: &[f64],
        col_upper: &[f64],
        feastol: f64,
        cutset: &mut Vec<i32>,
    ) -> Vec<SelectedCut> {
        let nrows = self.matrix.num_rows();
        let mut efficacious: Vec<(f64, i32)> = Vec::new();
        let mut agelim = self.agelim;
        let mut num_cuts = self.num_cuts() - self.num_lp_cuts;
        while agelim > 1 && num_cuts > self.softlimit {
            num_cuts -= self.age_distribution[agelim as usize];
            agelim -= 1;
        }
        for i in 0..nrows {
            let row = i as i32;
            if self.ages[i] < 0 {
                continue;
            }
            let (ri, rv) = self.matrix.row(row);
            let mut viol = -self.rhs[i];
            for k in 0..ri.len() {
                viol += rv[k] * sol[ri[k] as usize];
            }
            let is_propagated = self.matrix.columns_linked(row);
            self.age_distribution[self.ages[i] as usize] -= 1;
            if is_propagated {
                self.prop_rows.remove(&(self.ages[i], row));
            }
            if viol <= feastol {
                self.ages[i] += 1;
                if self.ages[i] as i32 >= agelim {
                    let (ri, rv) = self.matrix.row(row);
                    let h = compute_cut_hash(ri, rv, self.maxabscoef[i]);
                    self.delete_cut(obs, row, is_propagated);
                    if let Some(b) = self.hash_to_cut.get_mut(&h) {
                        if let Some(p) = b.iter().position(|&x| x == row) {
                            b.remove(p);
                        }
                    }
                } else {
                    if is_propagated {
                        self.prop_rows.insert((self.ages[i], row));
                    }
                    self.age_distribution[self.ages[i] as usize] += 1;
                }
                continue;
            }
            let (ri, rv) = self.matrix.row(row);
            let mut rownorm = CDouble::new(0.0);
            let mut num_active: i32 = 0;
            for k in 0..ri.len() {
                let col = ri[k] as usize;
                let s = sol[col];
                if rv[k] > 0.0 {
                    if s > col_lower[col] + feastol {
                        rownorm += rv[k] * rv[k];
                        num_active += 1;
                    }
                } else if s < col_upper[col] - feastol {
                    rownorm += rv[k] * rv[k];
                    num_active += 1;
                }
            }
            self.ages[i] = 0;
            self.age_distribution[0] += 1;
            if is_propagated {
                self.prop_rows.insert((0, row));
            }
            let score = viol / (num_active as f64 * rownorm.to_f64().sqrt());
            efficacious.push((score, row));
        }
        if efficacious.is_empty() {
            return Vec::new();
        }
        let n = efficacious.len() as u64;
        pdqsort(&mut efficacious, |a: &(f64, i32), b: &(f64, i32)| {
            if a.0 > b.0 {
                return true;
            }
            if a.0 < b.0 {
                return false;
            }
            let ha = hash_u64(((a.1 as u64) << 32).wrapping_add(n));
            let hb = hash_u64(((b.1 as u64) << 32).wrapping_add(n));
            // std::make_pair(ha, a.1) > std::make_pair(hb, b.1)
            hb < ha || (!(ha < hb) && b.1 < a.1)
        });
        // std::max(a, b) == (a < b) ? b : a
        let best = if efficacious[0].0 < self.best_observed_score {
            self.best_observed_score
        } else {
            efficacious[0].0
        };
        self.best_observed_score = best;
        let min_score = self.min_score_factor * best;
        let mut numeff = upper_bound(&efficacious, &min_score, |ms: &f64, c: &(f64, i32)| {
            *ms > c.0
        });
        let lower = efficacious.len() / 20;
        let upper = efficacious.len() - 1;
        if numeff <= lower {
            numeff = (efficacious.len() / 2).max(1);
            self.min_score_factor = efficacious[numeff - 1].0 / best;
        } else if numeff > upper {
            self.min_score_factor = efficacious[upper].0 / best;
        }
        efficacious.truncate(numeff);
        let mut out = Vec::new();
        for &(_, cut) in &efficacious {
            let mut discard = false;
            for &other in cutset.iter() {
                if self.parallelism(other, cut) > 0.1 {
                    discard = true;
                    break;
                }
            }
            if discard {
                continue;
            }
            let c = cut as usize;
            self.num_lps[c] += 1;
            self.age_distribution[self.ages[c] as usize] -= 1;
            self.num_lp_cuts += 1;
            if self.matrix.columns_linked(cut) {
                self.prop_rows.remove(&(self.ages[c], cut));
                self.prop_rows.insert((-1, cut));
            }
            self.ages[c] = -1;
            cutset.push(cut);
            let (ri, rv) = self.matrix.row(cut);
            out.push(SelectedCut {
                index: cut,
                inds: ri.to_vec(),
                vals: rv.to_vec(),
                rhs: self.rhs[c],
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_slots_and_indices_are_reused_like_highs() {
        let mut m = DynamicRowMatrix::default();
        let a = m.add_row(&[0, 1, 2], &[1.0, 2.0, 3.0], true);
        let b = m.add_row(&[3], &[1.0], false);
        assert_eq!((a, b), (0, 1));
        m.remove_row(0);
        // A shorter row reuses the freed index 0 and the freed slot.
        let c = m.add_row(&[5, 6], &[1.0, 1.0], true);
        assert_eq!(c, 0);
        assert_eq!(m.row(0).0, &[5, 6]);
        assert_eq!(m.num_rows(), 2);
        assert_eq!(m.num_del_rows(), 0);
    }
}

#[cfg(test)]
mod hash_tests {
    use super::*;

    /// Reference values from a C++ program built against HiGHS's `HighsHash.h`
    /// (`p3_highs_vector_hash_reference.cpp`).
    #[test]
    fn vector_hash_and_double_hash_code_match_highs() {
        let want: [(usize, u64, u64); 9] = [
            (1, 2991392633619243010, 4184337800404310407),
            (2, 2574845380624260901, 8143067179758130621),
            (3, 3975202055887256781, 18308897704616663529),
            (7, 17401165802475764621, 6460167240443044437),
            (8, 12297328762915772131, 3420768701321748498),
            (63, 18090444257249596473, 12422252017345781273),
            (64, 15578138274156662755, 3578098807783585491),
            (65, 1951689278951451635, 12347858070305657115),
            (100, 1435027129425998760, 15881221248456063818),
        ];
        for (n, wi, wc) in want {
            let v: Vec<u8> = (0..n as i32)
                .flat_map(|i| (3 * i + 1).to_le_bytes())
                .collect();
            let c: Vec<u8> = (0..n)
                .flat_map(|i| double_hash_code(0.37 * (i as f64 + 1.0) - 1.0).to_le_bytes())
                .collect();
            assert_eq!(vector_hash_bytes(&v), wi, "ints n={n}");
            assert_eq!(vector_hash_bytes(&c), wc, "codes n={n}");
        }
        for (d, w) in [
            (1.0, 20251u32),
            (0.5, 4294922011),
            (-0.25, 4294881509),
            (3.7, 149804),
            (1e-9, 4293022961),
            (0.0, 0),
        ] {
            assert_eq!(double_hash_code(d), w, "{d}");
        }
    }
}
