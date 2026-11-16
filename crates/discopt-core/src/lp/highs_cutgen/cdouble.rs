//! Provenance: a Rust translation of HiGHS source (https://github.com/ERGO-Code/HiGHS,
//! Copyright (c) HiGHS, MIT License), modified for discopt. See `THIRD_PARTY_NOTICES.md`
//! at the repository root for the license text, and `mod.rs` for why HiGHS is ported
//! rather than linked.
//!
//! Faithful port of `util/HighsCDouble.h`: a double-double ("compensated double")
//! value `hi + lo`, with every operator reproducing HiGHS's exact operation order so
//! that a replayed cut is bit-identical to the C++ one.
//!
//! The asymmetries are deliberate and are HiGHS's, not typos: `CDouble * CDouble`
//! adds `hi * v.lo` and drops `lo * v.lo`/`lo * v.hi` (except through
//! `two_product`), `f64 - CDouble` is computed as `(-b) + a`, and so on. Do not
//! "simplify" any of these; the replay tests in `core.rs` fail if the bits change.

use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// `HighsCDouble`: the value `hi + lo` carried in two doubles.
#[derive(Clone, Copy, Debug, Default)]
pub struct CDouble {
    /// Leading part.
    pub hi: f64,
    /// Compensation term.
    pub lo: f64,
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let z = x - a;
    let y = (a - (x - z)) + (b - z);
    (x, y)
}

#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    const FACTOR: f64 = ((1u64 << 27) + 1) as f64;
    let c = FACTOR * a;
    let x = c - (c - a);
    let y = a - x;
    (x, y)
}

#[inline(always)]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (a1, a2) = split(a);
    let (b1, b2) = split(b);
    let y = a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2);
    (x, y)
}

impl CDouble {
    /// `HighsCDouble(double)`.
    #[inline(always)]
    pub const fn new(v: f64) -> Self {
        CDouble { hi: v, lo: 0.0 }
    }
    /// Both parts as given (HiGHS's private two-argument constructor).
    #[inline(always)]
    pub const fn from_parts(hi: f64, lo: f64) -> Self {
        CDouble { hi, lo }
    }
    /// `explicit operator double()`.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
    /// `renormalize()`: re-split so `lo` is the rounding error of `hi`.
    #[inline(always)]
    pub fn renormalize(&mut self) {
        let (h, l) = two_sum(self.hi, self.lo);
        self.hi = h;
        self.lo = l;
    }
    /// `abs(HighsCDouble)`.
    pub fn abs(self) -> Self {
        if self.to_f64() < 0.0 {
            -self
        } else {
            self
        }
    }
    /// `sqrt(HighsCDouble)`.
    pub fn sqrt(self) -> Self {
        let c = (self.hi + self.lo).sqrt();
        if c == 0.0 {
            return CDouble::new(0.0);
        }
        let mut res = self / c;
        res += c;
        res.hi *= 0.5;
        res.lo *= 0.5;
        res
    }
    /// `floor(HighsCDouble)`.
    pub fn floor(self) -> Self {
        if self.abs().to_f64() < 1.0 {
            let d = self.to_f64();
            if d == 0.0 || d > 0.0 {
                return CDouble::new(0.0);
            }
            return CDouble::new(-1.0);
        }
        let floor_x = self.to_f64().floor();
        let (h, l) = two_sum(floor_x, (self - floor_x).to_f64().floor());
        CDouble { hi: h, lo: l }
    }
    /// `ceil(HighsCDouble)`.
    pub fn ceil(self) -> Self {
        if self.abs().to_f64() < 1.0 {
            let d = self.to_f64();
            if d == 0.0 || d < 0.0 {
                return CDouble::new(0.0);
            }
            return CDouble::new(1.0);
        }
        let ceil_x = self.to_f64().ceil();
        let (h, l) = two_sum(ceil_x, (self - ceil_x).to_f64().ceil());
        CDouble { hi: h, lo: l }
    }
    /// `round(HighsCDouble)`: `floor(x + 0.5)`.
    pub fn round(self) -> Self {
        (self + 0.5).floor()
    }
    /// `ldexp(HighsCDouble, exp)` on both parts.
    pub fn ldexp(self, exp: i32) -> Self {
        CDouble {
            hi: libm::ldexp(self.hi, exp),
            lo: libm::ldexp(self.lo, exp),
        }
    }
}

impl From<f64> for CDouble {
    #[inline(always)]
    fn from(v: f64) -> Self {
        CDouble::new(v)
    }
}

impl AddAssign<f64> for CDouble {
    #[inline(always)]
    fn add_assign(&mut self, v: f64) {
        let (h, c) = two_sum(v, self.hi);
        self.hi = h;
        self.lo += c;
    }
}
impl AddAssign<CDouble> for CDouble {
    #[inline(always)]
    fn add_assign(&mut self, v: CDouble) {
        *self += v.hi;
        self.lo += v.lo;
    }
}
impl SubAssign<f64> for CDouble {
    #[inline(always)]
    fn sub_assign(&mut self, v: f64) {
        *self += -v;
    }
}
impl SubAssign<CDouble> for CDouble {
    #[inline(always)]
    fn sub_assign(&mut self, v: CDouble) {
        *self -= v.hi;
        self.lo -= v.lo;
    }
}
impl MulAssign<f64> for CDouble {
    #[inline(always)]
    fn mul_assign(&mut self, v: f64) {
        let c = self.lo * v;
        let (h, l) = two_product(self.hi, v);
        self.hi = h;
        self.lo = l;
        *self += c;
    }
}
impl MulAssign<CDouble> for CDouble {
    #[inline(always)]
    fn mul_assign(&mut self, v: CDouble) {
        let c1 = self.hi * v.lo;
        let c2 = self.lo * v.hi;
        let (h, l) = two_product(self.hi, v.hi);
        self.hi = h;
        self.lo = l;
        *self += c1;
        *self += c2;
    }
}
impl DivAssign<f64> for CDouble {
    #[inline(always)]
    fn div_assign(&mut self, v: f64) {
        let d = CDouble {
            hi: self.hi / v,
            lo: self.lo / v,
        };
        let mut c = d * v - *self;
        c.hi /= v;
        c.lo /= v;
        *self = d - c;
    }
}
impl DivAssign<CDouble> for CDouble {
    #[inline(always)]
    fn div_assign(&mut self, v: CDouble) {
        let vdbl = v.hi + v.lo;
        let d = CDouble {
            hi: self.hi / vdbl,
            lo: self.lo / vdbl,
        };
        let mut c = d * v - *self;
        c.hi /= vdbl;
        c.lo /= vdbl;
        *self = d - c;
    }
}
impl Neg for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn neg(self) -> CDouble {
        CDouble {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}
impl Add<f64> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn add(self, v: f64) -> CDouble {
        let (h, l) = two_sum(self.hi, v);
        CDouble {
            hi: h,
            lo: l + self.lo,
        }
    }
}
impl Add<CDouble> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn add(self, v: CDouble) -> CDouble {
        let mut res = self + v.hi;
        res.lo += v.lo;
        res
    }
}
impl Add<CDouble> for f64 {
    type Output = CDouble;
    #[inline(always)]
    fn add(self, b: CDouble) -> CDouble {
        b + self
    }
}
impl Sub<f64> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn sub(self, v: f64) -> CDouble {
        let (h, l) = two_sum(self.hi, -v);
        CDouble {
            hi: h,
            lo: l + self.lo,
        }
    }
}
impl Sub<CDouble> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn sub(self, v: CDouble) -> CDouble {
        let mut res = self - v.hi;
        res.lo -= v.lo;
        res
    }
}
impl Sub<CDouble> for f64 {
    type Output = CDouble;
    #[inline(always)]
    fn sub(self, b: CDouble) -> CDouble {
        -b + self
    }
}
impl Mul<f64> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn mul(self, v: f64) -> CDouble {
        let (h, l) = two_product(self.hi, v);
        let mut res = CDouble { hi: h, lo: l };
        res += self.lo * v;
        res
    }
}
impl Mul<CDouble> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn mul(self, v: CDouble) -> CDouble {
        let mut res = self * v.hi;
        res += self.hi * v.lo;
        res
    }
}
impl Mul<CDouble> for f64 {
    type Output = CDouble;
    #[inline(always)]
    fn mul(self, b: CDouble) -> CDouble {
        b * self
    }
}
impl Div<f64> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn div(self, v: f64) -> CDouble {
        let mut res = self;
        res /= v;
        res
    }
}
impl Div<CDouble> for CDouble {
    type Output = CDouble;
    #[inline(always)]
    fn div(self, v: CDouble) -> CDouble {
        let mut res = self;
        res /= v;
        res
    }
}
impl Div<CDouble> for f64 {
    type Output = CDouble;
    #[inline(always)]
    fn div(self, b: CDouble) -> CDouble {
        CDouble::new(self) / b
    }
}

// Comparisons are on the rounded value, as in HiGHS.
impl PartialEq for CDouble {
    fn eq(&self, o: &CDouble) -> bool {
        self.to_f64() == o.to_f64()
    }
}
impl PartialEq<f64> for CDouble {
    fn eq(&self, o: &f64) -> bool {
        self.to_f64() == *o
    }
}
impl PartialOrd for CDouble {
    fn partial_cmp(&self, o: &CDouble) -> Option<Ordering> {
        self.to_f64().partial_cmp(&o.to_f64())
    }
}
impl PartialOrd<f64> for CDouble {
    fn partial_cmp(&self, o: &f64) -> Option<Ordering> {
        self.to_f64().partial_cmp(o)
    }
}
impl PartialEq<CDouble> for f64 {
    fn eq(&self, o: &CDouble) -> bool {
        *self == o.to_f64()
    }
}
impl PartialOrd<CDouble> for f64 {
    fn partial_cmp(&self, o: &CDouble) -> Option<Ordering> {
        self.partial_cmp(&o.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sum_recovers_the_bits_plain_double_loses() {
        // 1 + 1e-17 is 1 in double; the compensated value keeps the 1e-17.
        let mut s = CDouble::new(1.0);
        s += 1e-17;
        assert_eq!(s.hi, 1.0);
        assert_eq!(s.lo, 1e-17);
        let back = s - 1.0;
        assert_eq!(back.to_f64(), 1e-17);
    }

    #[test]
    fn floor_and_ceil_follow_the_highs_small_value_branch() {
        assert_eq!(CDouble::new(0.3).floor().to_f64(), 0.0);
        assert_eq!(CDouble::new(-0.3).floor().to_f64(), -1.0);
        assert_eq!(CDouble::new(-0.3).ceil().to_f64(), 0.0);
        assert_eq!(CDouble::new(0.3).ceil().to_f64(), 1.0);
        let x = CDouble::from_parts(5.0, -1e-20);
        assert_eq!(x.floor().to_f64(), 4.0);
        assert_eq!(x.ceil().to_f64(), 5.0);
    }
}
