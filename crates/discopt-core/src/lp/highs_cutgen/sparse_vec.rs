//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text.
//!
//! Faithful port of `util/HighsSparseVectorSum.h`: a dense accumulator of
//! compensated sums with a list of touched indices in first-touch order. The
//! nonzero order is part of every row HiGHS builds with it (aggregations,
//! transformed and untransformed cuts), so the port keeps it exactly.

use super::cdouble::CDouble;
use super::stdalgo::partition;

/// `HighsSparseVectorSum`.
#[derive(Clone, Debug, Default)]
pub struct SparseVectorSum {
    /// Accumulated value per index.
    pub values: Vec<CDouble>,
    /// Touched indices, in first-touch order (then permuted by `cleanup`).
    pub nonzeroinds: Vec<i32>,
}

impl SparseVectorSum {
    /// `HighsSparseVectorSum(dimension)`.
    pub fn new(dimension: usize) -> Self {
        let mut s = SparseVectorSum::default();
        s.set_dimension(dimension);
        s
    }
    /// `setDimension`.
    pub fn set_dimension(&mut self, dimension: usize) {
        self.values.resize(dimension, CDouble::new(0.0));
        self.nonzeroinds.reserve(dimension);
    }
    /// `add(index, double)`. An exact cancellation leaves the smallest normal
    /// double so the index stays registered, as in HiGHS.
    #[inline]
    pub fn add(&mut self, index: i32, value: f64) {
        let i = index as usize;
        if self.values[i] != 0.0 {
            self.values[i] += value;
        } else {
            self.values[i] = CDouble::new(value);
            self.nonzeroinds.push(index);
        }
        if self.values[i] == 0.0 {
            self.values[i] = CDouble::new(f64::MIN_POSITIVE);
        }
    }
    /// `add(index, HighsCDouble)`.
    #[inline]
    pub fn add_cd(&mut self, index: i32, value: CDouble) {
        let i = index as usize;
        if self.values[i] != 0.0 {
            self.values[i] += value;
        } else {
            self.values[i] = value;
            self.nonzeroinds.push(index);
        }
        if self.values[i] == 0.0 {
            self.values[i] = CDouble::new(f64::MIN_POSITIVE);
        }
    }
    /// `getNonzeros()`.
    pub fn nonzeros(&self) -> &[i32] {
        &self.nonzeroinds
    }
    /// `getValue(index)`.
    #[inline]
    pub fn value(&self, index: i32) -> f64 {
        self.values[index as usize].to_f64()
    }
    /// `clear()`.
    pub fn clear(&mut self) {
        if 10 * self.nonzeroinds.len() < 3 * self.values.len() {
            for &i in &self.nonzeroinds {
                self.values[i as usize] = CDouble::new(0.0);
            }
        } else {
            for v in self.values.iter_mut() {
                *v = CDouble::new(0.0);
            }
        }
        self.nonzeroinds.clear();
    }
    /// `partition(pred)` over the nonzero list (libstdc++ order).
    pub fn partition<P: FnMut(&i32) -> bool>(&mut self, pred: P) -> usize {
        partition(&mut self.nonzeroinds, pred)
    }
    /// `cleanup(isZero)`: drop entries for which `is_zero(index, value)` holds,
    /// swapping each dropped one behind the kept range (back to front).
    pub fn cleanup<F: FnMut(i32, f64) -> bool>(&mut self, mut is_zero: F) {
        let mut num_nz = self.nonzeroinds.len();
        let mut i = num_nz;
        while i > 0 {
            i -= 1;
            let pos = self.nonzeroinds[i];
            let val = self.values[pos as usize].to_f64();
            if is_zero(pos, val) {
                self.values[pos as usize] = CDouble::new(0.0);
                num_nz -= 1;
                self.nonzeroinds.swap(num_nz, i);
            }
        }
        self.nonzeroinds.truncate(num_nz);
    }
}
