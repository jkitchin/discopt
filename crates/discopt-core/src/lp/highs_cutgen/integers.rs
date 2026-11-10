//! Faithful port of the `util/HighsIntegers.h` functions the cut generator uses.

use super::cdouble::CDouble;

/// `std::frexp` exponent.
#[inline]
pub(crate) fn frexp_exp(x: f64) -> i32 {
    libm::frexp(x).1
}

/// `HighsIntegers::nearestInteger`: `(int64_t)(x + copysign(0.5, x))`.
#[inline]
pub fn nearest_integer(x: f64) -> i64 {
    (x + 0.5f64.copysign(x)) as i64
}

/// `HighsIntegers::isIntegral`.
#[inline]
pub fn is_integral(x: f64, eps: f64) -> bool {
    let y = (x - x.trunc()).abs();
    y.min(1.0 - y) <= eps
}

/// `HighsIntegers::gcd`.
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    if a < 0 {
        a = a.wrapping_neg();
    }
    if b < 0 {
        b = b.wrapping_neg();
    }
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    loop {
        let h = a % b;
        a = b;
        b = h;
        if b == 0 {
            break;
        }
    }
    a
}

/// `HighsIntegers::denominator`: continued-fraction denominator <= `maxdenom`.
pub fn denominator(x: f64, eps: f64, maxdenom: i64) -> i64 {
    let mut ai = x as i64;
    let mut m = [ai, 1i64, 1, 0];
    let mut xi = CDouble::new(x);
    let mut fraction = xi - ai as f64;
    while fraction > eps {
        xi = 1.0 / fraction;
        if xi.to_f64() > (1i64 << 53) as f64 {
            break;
        }
        ai = xi.to_f64() as i64;
        let t = m[2].wrapping_mul(ai).wrapping_add(m[3]);
        if t > maxdenom {
            break;
        }
        m[3] = m[2];
        m[2] = t;
        let t = m[0].wrapping_mul(ai).wrapping_add(m[1]);
        m[1] = m[0];
        m[0] = t;
        fraction = xi - ai as f64;
    }
    ai = (maxdenom - m[3]) / m[2];
    m[1] = m[1].wrapping_add(m[0].wrapping_mul(ai));
    m[3] = m[3].wrapping_add(m[2].wrapping_mul(ai));
    let x0 = m[0] as f64 / m[2] as f64;
    let x1 = m[1] as f64 / m[3] as f64;
    let x = x.abs();
    let err0 = (x - x0).abs();
    let err1 = (x - x1).abs();
    if err0 < err1 {
        return m[2];
    }
    m[3]
}

/// `HighsIntegers::integralScale(vals, n, deltadown, deltaup)`.
pub fn integral_scale(vals: &[f64], deltadown: f64, deltaup: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    // std::minmax_element by |.|: first smallest, LAST largest.
    let mut minval = vals[0];
    let mut maxval = vals[0];
    for &v in &vals[1..] {
        if v.abs() < minval.abs() {
            minval = v;
        }
        // `!comp(v, max)`; inputs are finite, so `>=` is the same test.
        if v.abs() >= maxval.abs() {
            maxval = v;
        }
    }
    let mut expshift = 0i32;
    if minval < -deltadown || minval > deltaup {
        expshift = frexp_exp(minval);
    }
    expshift = (-expshift).max(0) + 3;
    let mut exp_max_val = frexp_exp(maxval);
    exp_max_val = exp_max_val.min(32);
    if exp_max_val + expshift > 32 {
        expshift = 32 - exp_max_val;
    }
    let mut denom: u64 = 75u64 << expshift;
    let mut startdenom: i64 = denom as i64;

    let mut val = (startdenom as f64) * CDouble::new(vals[0]);
    let mut downval = (val + deltaup).floor();
    let mut fraction = val - downval;
    if fraction > deltadown {
        denom = denom.wrapping_mul(denominator(fraction.to_f64(), deltaup, 1000) as u64);
        val = (denom as f64) * CDouble::new(vals[0]);
        downval = (val + deltaup).floor();
        fraction = val - downval;
        if fraction > deltadown {
            return 0.0;
        }
    }
    let mut currgcd: u64 = downval.to_f64().abs() as u64;
    for &vi in &vals[1..] {
        val = (denom as f64) * CDouble::new(vi);
        downval = (val + deltaup).floor();
        fraction = val - downval;
        if fraction > deltadown {
            val = (startdenom as f64) * CDouble::new(vi);
            fraction = val - val.floor();
            denom = denom.wrapping_mul(denominator(fraction.to_f64(), deltaup, 1000) as u64);
            val = (denom as f64) * CDouble::new(vi);
            downval = (val + deltaup).floor();
            fraction = val - downval;
            if fraction > deltadown {
                return 0.0;
            }
        }
        if currgcd != 1 {
            currgcd = gcd(currgcd as i64, downval.to_f64() as i64) as u64;
            if denom > u32::MAX as u64 {
                // C++ divides unconditionally (a zero gcd would trap there); a zero
                // gcd means every value so far is 0, so dividing is skipped.
                if let Some(q) = denom.checked_div(currgcd) {
                    denom = q;
                    if startdenom != 1 {
                        startdenom /= gcd(currgcd as i64, startdenom);
                    }
                }
                currgcd = 1;
            }
        }
    }
    denom as f64 / currgcd as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_scale_finds_common_denominators() {
        assert_eq!(integral_scale(&[0.5, 1.5, 2.0], 1e-6, 1e-9), 2.0);
        assert_eq!(
            integral_scale(&[1.0 / 3.0, 2.0 / 3.0, 1.0], 1e-6, 1e-9),
            3.0
        );
        assert_eq!(integral_scale(&[2.0, 4.0, 6.0], 1e-6, 1e-9), 0.5);
        assert_eq!(
            integral_scale(&[std::f64::consts::PI, 1.0], 1e-9, 1e-12),
            0.0
        );
        assert_eq!(nearest_integer(-2.5), -3);
        assert!(is_integral(3.0000001, 1e-6));
    }
}
