//! Feasibility-Based Bound Tightening (FBBT).
//!
//! Implements interval arithmetic and forward/backward propagation
//! through the expression DAG to tighten variable bounds.

use crate::expr::{
    BinOp, ConstraintSense, ExprArena, ExprId, ExprNode, MathFunc, ModelRepr, ObjectiveSense, UnOp,
    VarType,
};
use std::f64::consts::PI;

/// Feasibility tolerance for declaring a constraint infeasible during FBBT.
///
/// A forward-propagated constraint body that misses its required output bound
/// by less than this is treated as feasible (numerical noise), not as proof of
/// infeasibility. Matches the solver's absolute feasibility tolerance (1e-6)
/// and is deliberately larger than the FBBT convergence tolerance (~1e-8) so
/// that eps-scale residuals from approximate reformulations (e.g. GDP hull
/// perspective forms) cannot fabricate an unsound infeasibility certificate.
pub const FEAS_TOL: f64 = 1e-6;

// ─────────────────────────────────────────────────────────────
// Interval type
// ─────────────────────────────────────────────────────────────

/// A closed interval `[lo, hi]`.
///
/// An interval with `lo > hi` is empty, representing infeasibility.
#[derive(Debug, Clone, Copy)]
pub struct Interval {
    /// Lower bound of the interval.
    pub lo: f64,
    /// Upper bound of the interval.
    pub hi: f64,
}

impl Interval {
    /// Create a new interval.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// The entire real line.
    pub fn entire() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// A point interval `[v, v]`.
    pub fn point(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    /// An empty interval.
    pub fn empty() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Whether the interval is empty (lo > hi).
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// Whether the interval is empty by more than a feasibility tolerance,
    /// i.e. `lo - hi > tol`.
    ///
    /// Use this (not [`is_empty`]) when deciding that a *constraint* is
    /// genuinely infeasible. Approximate reformulations — notably the GDP
    /// hull perspective form `y * f(v / y)` with a clamp `y + eps` — leave
    /// eps-scale (~1e-8) residuals at integer faces. A strict `lo > hi`
    /// check mistakes that numerical noise for infeasibility and can fix a
    /// disjunction's selector incorrectly, producing an unsound bound. A
    /// tolerance-aware check declares infeasibility only when the violation
    /// exceeds the feasibility tolerance.
    pub fn is_empty_beyond(&self, tol: f64) -> bool {
        self.lo - self.hi > tol
    }

    /// Whether `x` is contained in the interval.
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }

    /// Intersect two intervals.
    pub fn intersect(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Interval arithmetic
// ─────────────────────────────────────────────────────────────

/// `[a,b] + [c,d] = [a+c, b+d]`
pub fn interval_add(a: &Interval, b: &Interval) -> Interval {
    Interval::new(a.lo + b.lo, a.hi + b.hi)
}

/// `[a,b] - [c,d] = [a-d, b-c]`
pub fn interval_sub(a: &Interval, b: &Interval) -> Interval {
    Interval::new(a.lo - b.hi, a.hi - b.lo)
}

/// `[a,b] * [c,d]` using all four endpoint products.
///
/// A finite `0` endpoint multiplied by an infinite endpoint yields `0 * ±∞ =
/// NaN` in IEEE-754. By the interval-multiplication convention the product of a
/// zero factor with any interval (including an unbounded one) is `0`, so we map
/// those NaN corner products to `0` (C-22). This keeps `interval_mul` a sound,
/// finite outer enclosure — e.g. `[0,0] * [-∞,∞] = [0,0]` rather than the
/// `[NaN,NaN]` that would silently discard downstream tightening. NaN can arise
/// here *only* from `0 * ±∞`; every other operand pair is finite×finite (never
/// NaN) or a genuine ±∞ product, so the substitution never masks a real value.
pub fn interval_mul(a: &Interval, b: &Interval) -> Interval {
    let prod = |x: f64, y: f64| {
        let p = x * y;
        if p.is_nan() {
            0.0
        } else {
            p
        }
    };
    let p1 = prod(a.lo, b.lo);
    let p2 = prod(a.lo, b.hi);
    let p3 = prod(a.hi, b.lo);
    let p4 = prod(a.hi, b.hi);
    Interval::new(p1.min(p2).min(p3).min(p4), p1.max(p2).max(p3).max(p4))
}

/// `[a,b] / [c,d]` with division-by-zero handling.
pub fn interval_div(a: &Interval, b: &Interval) -> Interval {
    if b.lo <= 0.0 && b.hi >= 0.0 {
        // Denominator contains zero — result is the entire real line.
        Interval::entire()
    } else {
        let inv_b = Interval::new(1.0 / b.hi, 1.0 / b.lo);
        interval_mul(a, &inv_b)
    }
}

/// `[a,b]^n` for integer exponent.
pub fn interval_pow_int(base: &Interval, n: i64) -> Interval {
    if n == 0 {
        return Interval::point(1.0);
    }
    if n == 1 {
        return *base;
    }
    if n < 0 {
        let pos = interval_pow_int(base, -n);
        return interval_div(&Interval::point(1.0), &pos);
    }
    if n % 2 == 0 {
        // Even power: result is non-negative.
        if base.lo >= 0.0 {
            Interval::new(base.lo.powi(n as i32), base.hi.powi(n as i32))
        } else if base.hi <= 0.0 {
            Interval::new(base.hi.powi(n as i32), base.lo.powi(n as i32))
        } else {
            // Interval straddles zero.
            let max_val = base.lo.abs().max(base.hi.abs()).powi(n as i32);
            Interval::new(0.0, max_val)
        }
    } else {
        // Odd power: monotone increasing.
        Interval::new(base.lo.powi(n as i32), base.hi.powi(n as i32))
    }
}

/// `[a,b]^[c,d]` for general power.
pub fn interval_pow(base: &Interval, exp: &Interval) -> Interval {
    // If exponent is a point and integer, use int version.
    if (exp.hi - exp.lo).abs() < 1e-12 {
        let e = exp.lo;
        let e_int = e.round() as i64;
        if (e - e_int as f64).abs() < 1e-12 {
            return interval_pow_int(base, e_int);
        }
    }
    // General case: base must be non-negative for real-valued power.
    let b = Interval::new(base.lo.max(0.0), base.hi.max(0.0));
    if b.is_empty() || b.hi < 0.0 {
        return Interval::entire();
    }
    let vals = [
        b.lo.powf(exp.lo),
        b.lo.powf(exp.hi),
        b.hi.powf(exp.lo),
        b.hi.powf(exp.hi),
    ];
    let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Interval::new(lo, hi)
}

/// `neg([a,b]) = [-b, -a]`
pub fn interval_neg(a: &Interval) -> Interval {
    Interval::new(-a.hi, -a.lo)
}

/// `abs([a,b])`
pub fn interval_abs(a: &Interval) -> Interval {
    if a.lo >= 0.0 {
        *a
    } else if a.hi <= 0.0 {
        Interval::new(-a.hi, -a.lo)
    } else {
        Interval::new(0.0, a.lo.abs().max(a.hi.abs()))
    }
}

/// `exp([a,b]) = [exp(a), exp(b)]`
pub fn interval_exp(a: &Interval) -> Interval {
    Interval::new(a.lo.exp(), a.hi.exp())
}

/// `log([a,b]) = [log(max(a, eps)), log(b)]`
pub fn interval_log(a: &Interval) -> Interval {
    let lo = a.lo.max(f64::MIN_POSITIVE);
    if a.hi <= 0.0 {
        return Interval::empty();
    }
    Interval::new(lo.ln(), a.hi.ln())
}

/// `log2([a,b])`
pub fn interval_log2(a: &Interval) -> Interval {
    let lo = a.lo.max(f64::MIN_POSITIVE);
    if a.hi <= 0.0 {
        return Interval::empty();
    }
    Interval::new(lo.log2(), a.hi.log2())
}

/// `log10([a,b])`
pub fn interval_log10(a: &Interval) -> Interval {
    let lo = a.lo.max(f64::MIN_POSITIVE);
    if a.hi <= 0.0 {
        return Interval::empty();
    }
    Interval::new(lo.log10(), a.hi.log10())
}

/// `sqrt([a,b]) = [sqrt(max(a,0)), sqrt(b)]`
pub fn interval_sqrt(a: &Interval) -> Interval {
    if a.hi < 0.0 {
        return Interval::empty();
    }
    Interval::new(a.lo.max(0.0).sqrt(), a.hi.sqrt())
}

/// `sin([a,b])` with periodicity handling.
pub fn interval_sin(a: &Interval) -> Interval {
    if a.width() >= 2.0 * PI {
        return Interval::new(-1.0, 1.0);
    }
    // Normalize to [0, 2*PI) range.
    let lo_norm = a.lo.rem_euclid(2.0 * PI);
    let hi_norm = lo_norm + (a.hi - a.lo);

    let lo_sin = a.lo.sin();
    let hi_sin = a.hi.sin();
    let mut min_val = lo_sin.min(hi_sin);
    let mut max_val = lo_sin.max(hi_sin);

    // Check if interval contains a maximum (pi/2 + 2*k*pi).
    let peak = PI / 2.0;
    if contains_angle(lo_norm, hi_norm, peak) {
        max_val = 1.0;
    }
    // Check if interval contains a minimum (3*pi/2 + 2*k*pi).
    let trough = 3.0 * PI / 2.0;
    if contains_angle(lo_norm, hi_norm, trough) {
        min_val = -1.0;
    }

    Interval::new(min_val, max_val)
}

/// `cos([a,b])` with periodicity handling.
pub fn interval_cos(a: &Interval) -> Interval {
    // cos(x) = sin(x + pi/2)
    interval_sin(&Interval::new(a.lo + PI / 2.0, a.hi + PI / 2.0))
}

/// `tan([a,b])` with branch handling.
///
/// `tan` is increasing and continuous on each branch `((k-1/2)pi, (k+1/2)pi)`
/// with vertical asymptotes at `(k+1/2)pi`. When `[a,b]` lies within a single
/// branch (no asymptote strictly inside), `tan` is monotone, so the image is
/// `[tan(a), tan(b)]`. Otherwise the image is unbounded and we return `entire`.
pub fn interval_tan(a: &Interval) -> Interval {
    // The branch index of x is round(x / pi): branch k spans
    // ((k-1/2)pi, (k+1/2)pi), with asymptotes at the half-integer multiples.
    let branch_lo = (a.lo / PI).round();
    let branch_hi = (a.hi / PI).round();
    if branch_lo == branch_hi {
        // Same branch: tan is increasing and finite here.
        Interval::new(a.lo.tan(), a.hi.tan())
    } else {
        // Spans at least one asymptote: unbounded.
        Interval::entire()
    }
}

/// Check if the angle `target` (mod 2*pi) is in [lo_norm, hi_norm].
fn contains_angle(lo_norm: f64, hi_norm: f64, target: f64) -> bool {
    // Check if any 2*k*pi + target falls in [lo_norm, hi_norm].
    let mut t = target;
    while t < lo_norm {
        t += 2.0 * PI;
    }
    t <= hi_norm
}

// ─────────────────────────────────────────────────────────────
// Forward propagation
// ─────────────────────────────────────────────────────────────

/// Forward-propagate interval bounds from leaves to root.
///
/// Returns a vector of intervals, one per arena node.
pub fn forward_propagate(arena: &ExprArena, _id: ExprId, var_bounds: &[Interval]) -> Vec<Interval> {
    let n = arena.len();
    let mut bounds = vec![Interval::entire(); n];

    // Walk nodes in topological order (0..n). Because the arena adds
    // children before parents, indices 0..n are already topologically sorted.
    for i in 0..n {
        let eid = ExprId(i);
        bounds[i] = eval_node_interval(arena, eid, var_bounds, &bounds);
    }
    bounds
}

/// Compute the interval for a single node given its children's intervals.
fn eval_node_interval(
    arena: &ExprArena,
    id: ExprId,
    var_bounds: &[Interval],
    node_bounds: &[Interval],
) -> Interval {
    match arena.get(id) {
        ExprNode::Constant(v) => Interval::point(*v),
        ExprNode::ConstantArray(data, _) => {
            if data.len() == 1 {
                Interval::point(data[0])
            } else {
                // For arrays, compute the range of all elements.
                let lo = data.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                Interval::new(lo, hi)
            }
        }
        ExprNode::Variable { index, size, .. } => {
            if *size == 1 {
                var_bounds[*index]
            } else {
                // Array variable — union of all element bounds.
                // Typically each element is accessed via Index nodes.
                var_bounds[*index]
            }
        }
        ExprNode::Parameter { value, .. } => {
            if value.len() == 1 {
                Interval::point(value[0])
            } else {
                let lo = value.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = value.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                Interval::new(lo, hi)
            }
        }
        ExprNode::BinaryOp { op, left, right } => {
            let l = node_bounds[left.0];
            let r = node_bounds[right.0];
            match op {
                BinOp::Add => interval_add(&l, &r),
                BinOp::Sub => interval_sub(&l, &r),
                BinOp::Mul => interval_mul(&l, &r),
                BinOp::Div => interval_div(&l, &r),
                BinOp::Pow => interval_pow(&l, &r),
            }
        }
        ExprNode::UnaryOp { op, operand } => {
            let a = node_bounds[operand.0];
            match op {
                UnOp::Neg => interval_neg(&a),
                UnOp::Abs => interval_abs(&a),
            }
        }
        ExprNode::FunctionCall { func, args } => {
            if args.is_empty() {
                return Interval::entire();
            }
            let a0 = node_bounds[args[0].0];
            match func {
                MathFunc::Exp => interval_exp(&a0),
                MathFunc::Log => interval_log(&a0),
                MathFunc::Log2 => interval_log2(&a0),
                MathFunc::Log10 => interval_log10(&a0),
                MathFunc::Sqrt => interval_sqrt(&a0),
                MathFunc::Sin => interval_sin(&a0),
                MathFunc::Cos => interval_cos(&a0),
                MathFunc::Tan => interval_tan(&a0),
                MathFunc::Atan => {
                    // atan is monotonically increasing, range (-pi/2, pi/2)
                    Interval::new(a0.lo.atan(), a0.hi.atan())
                }
                MathFunc::Sinh => {
                    // sinh is monotonically increasing
                    Interval::new(a0.lo.sinh(), a0.hi.sinh())
                }
                MathFunc::Cosh => {
                    // cosh is convex, minimum at 0
                    if a0.lo >= 0.0 {
                        Interval::new(a0.lo.cosh(), a0.hi.cosh())
                    } else if a0.hi <= 0.0 {
                        Interval::new(a0.hi.cosh(), a0.lo.cosh())
                    } else {
                        Interval::new(1.0, a0.lo.cosh().max(a0.hi.cosh()))
                    }
                }
                MathFunc::Asin => {
                    // asin defined on [-1, 1], monotonically increasing
                    let lo = a0.lo.max(-1.0).asin();
                    let hi = a0.hi.min(1.0).asin();
                    Interval::new(lo, hi)
                }
                MathFunc::Acos => {
                    // acos defined on [-1, 1], monotonically decreasing
                    let lo = a0.hi.min(1.0).acos();
                    let hi = a0.lo.max(-1.0).acos();
                    Interval::new(lo, hi)
                }
                MathFunc::Tanh => {
                    // tanh is monotonically increasing, range (-1, 1)
                    Interval::new(a0.lo.tanh(), a0.hi.tanh())
                }
                MathFunc::Asinh => {
                    // asinh is monotonically increasing on all of R
                    Interval::new(a0.lo.asinh(), a0.hi.asinh())
                }
                MathFunc::Acosh => {
                    // acosh defined on [1, inf), monotonically increasing
                    let lo = a0.lo.max(1.0).acosh();
                    let hi = a0.hi.max(1.0).acosh();
                    Interval::new(lo, hi)
                }
                MathFunc::Atanh => {
                    // atanh defined on (-1, 1), monotonically increasing.
                    // Clamp just inside the domain to avoid +/-inf.
                    const EPS: f64 = 1e-12;
                    let lo = a0.lo.clamp(-1.0 + EPS, 1.0 - EPS).atanh();
                    let hi = a0.hi.clamp(-1.0 + EPS, 1.0 - EPS).atanh();
                    Interval::new(lo, hi)
                }
                MathFunc::Erf => {
                    // erf is monotonically increasing, range (-1, 1)
                    Interval::new(libm::erf(a0.lo), libm::erf(a0.hi))
                }
                MathFunc::Log1p => {
                    // log1p(x) = ln(1 + x), defined on (-1, inf), increasing.
                    let lo = (a0.lo.max(-1.0)).ln_1p();
                    let hi = (a0.hi.max(-1.0)).ln_1p();
                    Interval::new(lo, hi)
                }
                MathFunc::Sigmoid => {
                    // sigmoid is monotonically increasing, range (0, 1)
                    let sig = |x: f64| 0.5 + 0.5 * (0.5 * x).tanh();
                    Interval::new(sig(a0.lo), sig(a0.hi))
                }
                MathFunc::Softplus => {
                    // softplus is monotonically increasing, range (0, inf)
                    let sp = |x: f64| x.max(0.0) + (-x.abs()).exp().ln_1p();
                    Interval::new(sp(a0.lo), sp(a0.hi))
                }
                MathFunc::Abs => interval_abs(&a0),
                MathFunc::Sign => Interval::new(-1.0, 1.0),
                MathFunc::Min => {
                    if args.len() > 1 {
                        let a1 = node_bounds[args[1].0];
                        Interval::new(a0.lo.min(a1.lo), a0.hi.min(a1.hi))
                    } else {
                        a0
                    }
                }
                MathFunc::Max => {
                    if args.len() > 1 {
                        let a1 = node_bounds[args[1].0];
                        Interval::new(a0.lo.max(a1.lo), a0.hi.max(a1.hi))
                    } else {
                        a0
                    }
                }
                MathFunc::Prod => {
                    // Single-arg prod is identity; multi-arg is a product chain.
                    if args.len() == 1 {
                        a0
                    } else {
                        let mut result = a0;
                        for arg in &args[1..] {
                            result = interval_mul(&result, &node_bounds[arg.0]);
                        }
                        result
                    }
                }
                MathFunc::Norm1 | MathFunc::Norm2 | MathFunc::NormInf | MathFunc::NormP(_) => {
                    // A p-norm is non-negative. The forward pass collapses an
                    // array argument to a single node interval, so a tight
                    // component-wise bound is not available here; return the
                    // sound non-negative enclosure. (Tightening for norms comes
                    // from the McCormick relaxation, not FBBT.)
                    Interval::new(0.0, f64::INFINITY)
                }
            }
        }
        ExprNode::Index { base, .. } => {
            // The interval of an indexed expression is the interval of the base.
            node_bounds[base.0]
        }
        ExprNode::MatMul { .. } => {
            // Conservative bound for matmul.
            Interval::entire()
        }
        ExprNode::Sum { operand, .. } => {
            // Sum of an array — conservative. For scalar, same as operand.
            node_bounds[operand.0]
        }
        ExprNode::SumOver { terms } => {
            let mut result = Interval::point(0.0);
            for t in terms {
                result = interval_add(&result, &node_bounds[t.0]);
            }
            result
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Backward propagation
// ─────────────────────────────────────────────────────────────

/// Inverse error function `erf^{-1}(y)` for `y in (-1, 1)`.
///
/// A Winitzki closed-form initial guess refined by Newton iterations against
/// `libm::erf` (which converges to ~machine precision). Backward propagation
/// widens the result outward by a small margin so the preimage stays a sound
/// superset despite any residual error.
fn erfinv(y: f64) -> f64 {
    use std::f64::consts::PI;
    let y = y.clamp(-1.0 + 1e-12, 1.0 - 1e-12);
    let a = 0.147_f64;
    let ln = (1.0 - y * y).ln();
    let t1 = 2.0 / (PI * a) + 0.5 * ln;
    let mut x = y.signum() * ((t1 * t1 - ln / a).sqrt() - t1).sqrt();
    // Newton refinement: x <- x - (erf(x) - y) / (2/sqrt(pi) * e^{-x^2}).
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..3 {
        let deriv = two_over_sqrt_pi * (-x * x).exp();
        if deriv.abs() < 1e-300 {
            break;
        }
        x -= (libm::erf(x) - y) / deriv;
    }
    x
}

/// Backward-propagate an output bound through the expression DAG to
/// tighten variable bounds.
///
/// `output_bound` is the feasible range for the root node `id`.
/// `node_bounds` are the forward-propagated bounds.
/// `var_bounds` is updated in place with tightened bounds.
pub fn backward_propagate(
    arena: &ExprArena,
    id: ExprId,
    output_bound: Interval,
    node_bounds: &[Interval],
    var_bounds: &mut [Interval],
) {
    // Intersect the output bound with the forward-propagated bound.
    let tightened = output_bound.intersect(&node_bounds[id.0]);
    if tightened.is_empty() {
        return;
    }

    match arena.get(id) {
        ExprNode::Variable { index, size, .. } => {
            if *size == 1 {
                var_bounds[*index] = var_bounds[*index].intersect(&tightened);
            }
        }
        ExprNode::BinaryOp { op, left, right } => {
            let l = node_bounds[left.0];
            let r = node_bounds[right.0];
            match op {
                BinOp::Add => {
                    // a + b in [lo, hi]
                    // a in [lo - b_hi, hi - b_lo]
                    // b in [lo - a_hi, hi - a_lo]
                    let new_l = Interval::new(tightened.lo - r.hi, tightened.hi - r.lo);
                    let new_r = Interval::new(tightened.lo - l.hi, tightened.hi - l.lo);
                    backward_propagate(arena, *left, new_l, node_bounds, var_bounds);
                    backward_propagate(arena, *right, new_r, node_bounds, var_bounds);
                }
                BinOp::Sub => {
                    // a - b in [lo, hi]
                    // a in [lo + b_lo, hi + b_hi]
                    // b in [a_lo - hi, a_hi - lo]
                    let new_l = Interval::new(tightened.lo + r.lo, tightened.hi + r.hi);
                    let new_r = Interval::new(l.lo - tightened.hi, l.hi - tightened.lo);
                    backward_propagate(arena, *left, new_l, node_bounds, var_bounds);
                    backward_propagate(arena, *right, new_r, node_bounds, var_bounds);
                }
                BinOp::Mul => {
                    // a * b in [lo, hi]
                    // a in [lo, hi] / b (if b doesn't contain 0)
                    if r.lo > 0.0 || r.hi < 0.0 {
                        let new_l = interval_div(&tightened, &r);
                        backward_propagate(arena, *left, new_l, node_bounds, var_bounds);
                    }
                    if l.lo > 0.0 || l.hi < 0.0 {
                        let new_r = interval_div(&tightened, &l);
                        backward_propagate(arena, *right, new_r, node_bounds, var_bounds);
                    }
                }
                BinOp::Div => {
                    // a / b in [lo, hi]
                    // a in [lo, hi] * b
                    let new_l = interval_mul(&tightened, &r);
                    backward_propagate(arena, *left, new_l, node_bounds, var_bounds);
                    // b in a / [lo, hi] (if [lo,hi] doesn't contain 0)
                    if tightened.lo > 0.0 || tightened.hi < 0.0 {
                        let new_r = interval_div(&l, &tightened);
                        backward_propagate(arena, *right, new_r, node_bounds, var_bounds);
                    }
                }
                BinOp::Pow => {
                    // If exponent is constant integer, we can invert.
                    if let Some(exp_val) = arena.try_constant_value_pub(*right) {
                        let exp_int = exp_val.round() as i64;
                        if (exp_val - exp_int as f64).abs() < 1e-12 && exp_int > 0 {
                            let inv = 1.0 / exp_int as f64;
                            if exp_int % 2 == 1 {
                                // Odd power is monotone: base^n in [lo, hi]
                                // => base in [lo^(1/n), hi^(1/n)], preserving sign.
                                let new_lo = if tightened.lo >= 0.0 {
                                    tightened.lo.powf(inv)
                                } else {
                                    -((-tightened.lo).powf(inv))
                                };
                                let new_hi = if tightened.hi >= 0.0 {
                                    tightened.hi.powf(inv)
                                } else {
                                    -((-tightened.hi).powf(inv))
                                };
                                let new_base = Interval::new(new_lo, new_hi);
                                backward_propagate(arena, *left, new_base, node_bounds, var_bounds);
                            } else {
                                // Even power: u^n in [lo, hi] with u^n >= 0 always.
                                // |u| in [root_lo, root_hi] where
                                //   root_hi = hi^(1/n), root_lo = max(0, lo)^(1/n).
                                // A negative upper bound on the output is infeasible.
                                if tightened.hi < 0.0 {
                                    backward_propagate(
                                        arena,
                                        *left,
                                        Interval::empty(),
                                        node_bounds,
                                        var_bounds,
                                    );
                                    return;
                                }
                                let root_hi = tightened.hi.powf(inv);
                                let root_lo = tightened.lo.max(0.0).powf(inv);
                                // Use the forward base bounds to resolve the sign of u.
                                let new_base = if l.lo >= 0.0 {
                                    // Base known nonnegative: u in [root_lo, root_hi].
                                    Interval::new(root_lo, root_hi)
                                } else if l.hi <= 0.0 {
                                    // Base known nonpositive: u in [-root_hi, -root_lo].
                                    Interval::new(-root_hi, -root_lo)
                                } else {
                                    // Base straddles zero: the feasible set is
                                    // [-root_hi, -root_lo] U [root_lo, root_hi]; we
                                    // soundly relax to the hull [-root_hi, root_hi].
                                    Interval::new(-root_hi, root_hi)
                                };
                                backward_propagate(arena, *left, new_base, node_bounds, var_bounds);
                            }
                        }
                    }
                }
            }
        }
        ExprNode::UnaryOp { op, operand } => {
            match op {
                UnOp::Neg => {
                    // -a in [lo, hi] => a in [-hi, -lo]
                    let new = interval_neg(&tightened);
                    backward_propagate(arena, *operand, new, node_bounds, var_bounds);
                }
                UnOp::Abs => {
                    // |a| in [lo, hi] => a in [-hi, -lo] union [lo, hi]
                    // Conservative: a in [-hi, hi]
                    let new = Interval::new(-tightened.hi, tightened.hi);
                    backward_propagate(arena, *operand, new, node_bounds, var_bounds);
                }
            }
        }
        ExprNode::FunctionCall { func, args } => {
            if args.is_empty() {
                return;
            }
            match func {
                MathFunc::Exp => {
                    // exp(a) in [lo, hi] => a in [log(lo), log(hi)]
                    let new_lo = if tightened.lo > 0.0 {
                        tightened.lo.ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    let new_hi = if tightened.hi > 0.0 {
                        tightened.hi.ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    let new = Interval::new(new_lo, new_hi);
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Log => {
                    // log(a) in [lo, hi] => a in [exp(lo), exp(hi)]
                    let new = Interval::new(tightened.lo.exp(), tightened.hi.exp());
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Sqrt => {
                    // sqrt(a) in [lo, hi] => a in [lo^2, hi^2] (lo >= 0)
                    let lo = tightened.lo.max(0.0);
                    let new = Interval::new(lo * lo, tightened.hi * tightened.hi);
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Log2 => {
                    // log2(a) in [lo, hi] => a in [2^lo, 2^hi]
                    let new = Interval::new(tightened.lo.exp2(), tightened.hi.exp2());
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Log10 => {
                    // log10(a) in [lo, hi] => a in [10^lo, 10^hi]
                    let new = Interval::new(10f64.powf(tightened.lo), 10f64.powf(tightened.hi));
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Log1p => {
                    // log1p(a)=ln(1+a) in [lo, hi] => a in [e^lo - 1, e^hi - 1]
                    let new = Interval::new(tightened.lo.exp_m1(), tightened.hi.exp_m1());
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Sinh => {
                    // sinh increasing, inverse asinh.
                    let new = Interval::new(tightened.lo.asinh(), tightened.hi.asinh());
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Asinh => {
                    // asinh increasing, inverse sinh.
                    let new = Interval::new(tightened.lo.sinh(), tightened.hi.sinh());
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Tanh => {
                    // tanh increasing onto (-1, 1), inverse atanh. Clamp inside domain.
                    const EPS: f64 = 1e-12;
                    let lo = tightened.lo.clamp(-1.0 + EPS, 1.0 - EPS).atanh();
                    let hi = tightened.hi.clamp(-1.0 + EPS, 1.0 - EPS).atanh();
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(lo, hi),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Atanh => {
                    // atanh increasing on (-1, 1), inverse tanh.
                    let new = Interval::new(tightened.lo.tanh(), tightened.hi.tanh());
                    backward_propagate(arena, args[0], new, node_bounds, var_bounds);
                }
                MathFunc::Tan => {
                    // tan is monotone within a single branch. If the forward
                    // input interval lies in one branch, invert via atan with
                    // the branch offset: x = atan(y) + k*pi.
                    use std::f64::consts::PI;
                    let inp = node_bounds[args[0].0];
                    let branch_lo = (inp.lo / PI).round();
                    let branch_hi = (inp.hi / PI).round();
                    if branch_lo == branch_hi {
                        let k_pi = branch_lo * PI;
                        let new_lo = tightened.lo.atan() + k_pi;
                        let new_hi = tightened.hi.atan() + k_pi;
                        backward_propagate(
                            arena,
                            args[0],
                            Interval::new(new_lo, new_hi),
                            node_bounds,
                            var_bounds,
                        );
                    }
                }
                MathFunc::Atan => {
                    // atan increasing onto (-pi/2, pi/2), inverse tan. Clamp range.
                    use std::f64::consts::FRAC_PI_2;
                    const EPS: f64 = 1e-12;
                    let lo = tightened.lo.clamp(-FRAC_PI_2 + EPS, FRAC_PI_2 - EPS).tan();
                    let hi = tightened.hi.clamp(-FRAC_PI_2 + EPS, FRAC_PI_2 - EPS).tan();
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(lo, hi),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Asin => {
                    // asin increasing onto [-pi/2, pi/2], inverse sin; preimage in [-1, 1].
                    use std::f64::consts::FRAC_PI_2;
                    let lo = tightened.lo.clamp(-FRAC_PI_2, FRAC_PI_2).sin();
                    let hi = tightened.hi.clamp(-FRAC_PI_2, FRAC_PI_2).sin();
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(lo, hi),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Acos => {
                    // acos decreasing onto [0, pi], inverse cos; preimage in [-1, 1].
                    use std::f64::consts::PI;
                    let lo_in = tightened.lo.clamp(0.0, PI);
                    let hi_in = tightened.hi.clamp(0.0, PI);
                    // decreasing: a in [cos(hi_in), cos(lo_in)]
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(hi_in.cos(), lo_in.cos()),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Acosh => {
                    // acosh increasing onto [0, inf), inverse cosh; preimage in [1, inf).
                    let lo = tightened.lo.max(0.0).cosh();
                    let hi = tightened.hi.max(0.0).cosh();
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(lo, hi),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Cosh => {
                    // cosh(a) in [lo, hi] (hi >= 1) is even; conservative symmetric preimage
                    // a in [-acosh(hi), acosh(hi)].
                    let r = tightened.hi.max(1.0).acosh();
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(-r, r),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Sigmoid => {
                    // sigmoid increasing onto (0, 1), inverse logit a = ln(p/(1-p)).
                    const EPS: f64 = 1e-12;
                    let logit = |p: f64| {
                        let p = p.clamp(EPS, 1.0 - EPS);
                        p.ln() - (1.0 - p).ln()
                    };
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(logit(tightened.lo), logit(tightened.hi)),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Softplus => {
                    // softplus increasing onto (0, inf), inverse a = s + ln(1 - e^{-s}), s > 0.
                    const EPS: f64 = 1e-12;
                    let inv = |s: f64| {
                        let s = s.max(EPS);
                        s + (-(-s).exp()).ln_1p()
                    };
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(inv(tightened.lo), inv(tightened.hi)),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Erf => {
                    // erf increasing onto (-1, 1); invert with erfinv, clamped
                    // to the open domain and widened by a small margin so the
                    // preimage is a sound superset despite erfinv's tiny error.
                    const M: f64 = 1e-9;
                    let lo = erfinv(tightened.lo) - M;
                    let hi = erfinv(tightened.hi) + M;
                    backward_propagate(
                        arena,
                        args[0],
                        Interval::new(lo, hi),
                        node_bounds,
                        var_bounds,
                    );
                }
                MathFunc::Sin => {
                    // sin is monotone on each piece between consecutive extrema
                    // at pi/2 + k*pi. If the forward input lies in one piece,
                    // invert via asin with the right 2*m*pi offset; otherwise the
                    // preimage is a union of intervals and we conservatively skip.
                    use std::f64::consts::{FRAC_PI_2, PI};
                    let inp = node_bounds[args[0].0];
                    // Monotone iff no sin-extremum (pi/2 + k*pi) is strictly
                    // interior to [inp.lo, inp.hi]; extrema at the endpoints are
                    // fine. Take the smallest extremum >= inp.lo.
                    let ext = FRAC_PI_2 + ((inp.lo - FRAC_PI_2) / PI).ceil() * PI;
                    let monotone = !(ext > inp.lo + 1e-12 && ext < inp.hi - 1e-12);
                    if monotone && (inp.hi - inp.lo) <= PI + 1e-9 {
                        let mid = 0.5 * (inp.lo + inp.hi);
                        let ylo = tightened.lo.clamp(-1.0, 1.0);
                        let yhi = tightened.hi.clamp(-1.0, 1.0);
                        let (new_lo, new_hi) = if mid.cos() >= 0.0 {
                            // increasing piece centered at 2*m*pi
                            let m = (mid / (2.0 * PI)).round();
                            (ylo.asin() + 2.0 * m * PI, yhi.asin() + 2.0 * m * PI)
                        } else {
                            // decreasing piece centered at pi + 2*m*pi
                            let m = ((mid - PI) / (2.0 * PI)).round();
                            (
                                PI - yhi.asin() + 2.0 * m * PI,
                                PI - ylo.asin() + 2.0 * m * PI,
                            )
                        };
                        backward_propagate(
                            arena,
                            args[0],
                            Interval::new(new_lo, new_hi),
                            node_bounds,
                            var_bounds,
                        );
                    }
                }
                MathFunc::Cos => {
                    // cos is monotone on each piece [k*pi, (k+1)*pi]: decreasing
                    // where sin>0, increasing where sin<0. acos(y) in [0, pi].
                    use std::f64::consts::PI;
                    let inp = node_bounds[args[0].0];
                    // Monotone iff no cos-extremum (k*pi) is strictly interior.
                    let ext = (inp.lo / PI).ceil() * PI;
                    let monotone = !(ext > inp.lo + 1e-12 && ext < inp.hi - 1e-12);
                    if monotone && (inp.hi - inp.lo) <= PI + 1e-9 {
                        let mid = 0.5 * (inp.lo + inp.hi);
                        let ylo = tightened.lo.clamp(-1.0, 1.0);
                        let yhi = tightened.hi.clamp(-1.0, 1.0);
                        let m = (mid / (2.0 * PI)).round();
                        let (new_lo, new_hi) = if mid.sin() > 0.0 {
                            // decreasing piece [2m*pi, pi+2m*pi]: x = acos(y) + 2m*pi,
                            // larger y -> smaller x.
                            (yhi.acos() + 2.0 * m * PI, ylo.acos() + 2.0 * m * PI)
                        } else {
                            // increasing piece [pi+2m'*pi, 2pi+2m'*pi]:
                            // x = 2*pi*m - acos(y) (m rounds to the piece's right end).
                            (2.0 * PI * m - ylo.acos(), 2.0 * PI * m - yhi.acos())
                        };
                        backward_propagate(
                            arena,
                            args[0],
                            Interval::new(new_lo, new_hi),
                            node_bounds,
                            var_bounds,
                        );
                    }
                }
                _ => {
                    // No backward propagation for the remaining functions
                    // (sign, min/max, prod, norm).
                }
            }
        }
        ExprNode::SumOver { terms } => {
            // For a sum t1 + t2 + ... + tn in [lo, hi],
            // each ti in [lo - sum_others_hi, hi - sum_others_lo].
            for (i, t) in terms.iter().enumerate() {
                let mut others_lo = 0.0;
                let mut others_hi = 0.0;
                for (j, s) in terms.iter().enumerate() {
                    if i != j {
                        others_lo += node_bounds[s.0].lo;
                        others_hi += node_bounds[s.0].hi;
                    }
                }
                let new = Interval::new(tightened.lo - others_hi, tightened.hi - others_lo);
                backward_propagate(arena, *t, new, node_bounds, var_bounds);
            }
        }
        ExprNode::Index { base, .. } => {
            backward_propagate(arena, *base, tightened, node_bounds, var_bounds);
        }
        ExprNode::Sum { operand, .. } => {
            backward_propagate(arena, *operand, tightened, node_bounds, var_bounds);
        }
        ExprNode::Constant(_)
        | ExprNode::ConstantArray(_, _)
        | ExprNode::Parameter { .. }
        | ExprNode::MatMul { .. } => {}
    }
}

// ─────────────────────────────────────────────────────────────
// Helper: public constant-value extraction
// ─────────────────────────────────────────────────────────────

impl ExprArena {
    /// Public wrapper for `try_constant_value` (which is private in expr.rs).
    pub fn try_constant_value_pub(&self, id: ExprId) -> Option<f64> {
        match self.get(id) {
            ExprNode::Constant(v) => Some(*v),
            ExprNode::Parameter { value, shape, .. } => {
                if shape.is_empty() || (shape.len() == 1 && shape[0] == 1) {
                    value.first().copied()
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Integrality-aware snapping (binary-indicator propagation)
// ─────────────────────────────────────────────────────────────

/// Integrality margin for snapping FBBT-derived bounds on integer and binary
/// variables. A derived bound must cross an integer by more than this before
/// we round inward, so eps-scale residuals (GDP hull perspective / McCormick
/// noise at integer faces, ~1e-8) cannot fix a variable wrongly and cut off a
/// feasible point. Matches the soundness margin used elsewhere ([`FEAS_TOL`]).
pub(crate) const INTEGRALITY_SNAP_TOL: f64 = FEAS_TOL;

/// Round one FBBT-derived interval inward to integrality.
///
/// For an integer-constrained variable, a continuous lower bound `lo` implies
/// the integer bound `ceil(lo)` and an upper bound `hi` implies `floor(hi)`,
/// since every feasible value is integral. This is always a sound tightening:
/// it can only discard non-integer slack, never a feasible integer point. The
/// `INTEGRALITY_SNAP_TOL` pullback makes it conservative — a bound a hair past
/// an integer is treated as that integer rather than rounded to the next one,
/// so eps-scale residuals can't fix a variable wrongly. `ceil`/`floor` of ±inf
/// stay ±inf, so unbounded sides pass through. An empty input is returned
/// unchanged; a value squeezed to neither 0 nor 1 (for a binary) yields an
/// empty interval — a genuine integer infeasibility the caller detects.
pub(crate) fn snap_integral_interval(iv: Interval) -> Interval {
    if iv.is_empty() {
        return iv;
    }
    Interval::new(
        (iv.lo - INTEGRALITY_SNAP_TOL).ceil(),
        (iv.hi + INTEGRALITY_SNAP_TOL).floor(),
    )
}

/// Snap derived bounds to integrality for every integer and binary variable.
///
/// This is what makes FBBT *indicator-aware*. A big-M guard
/// `g(x) ≤ M·(1 − b)` back-propagates an interval onto the binary `b`; snapping
/// that interval to `{0, 1}` fixes `b` whenever the guarded body is forced
/// feasible/infeasible (the backward indicator rule). Once `b` is fixed, the
/// next forward sweep evaluates `M·(1 − b)` exactly, activating or deactivating
/// the guard and tightening the guarded continuous variables (the forward
/// indicator rule).
fn snap_integral_bounds(model: &ModelRepr, var_bounds: &mut [Interval]) {
    for (i, v) in model.variables.iter().enumerate() {
        if v.var_type != VarType::Continuous {
            var_bounds[i] = snap_integral_interval(var_bounds[i]);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Fixed-point FBBT
// ─────────────────────────────────────────────────────────────

/// Seed a variable BLOCK's FBBT interval from its element-wise bounds.
///
/// The FBBT engine carries **one interval per variable block** (an array
/// variable of `size > 1` is a single `var_bounds` slot). An `Index` node that
/// selects element `k` resolves — in both forward and backward propagation — to
/// this single shared block interval, so the interval must be a valid *outer*
/// bound for **every** element of the block, not any one element's.
///
/// C-31: the previous seed used `v.lb.first()/v.ub.first()` — element 0's bounds
/// — and stamped them onto the whole block. On heterogeneous per-element bounds
/// that interval EXCLUDES feasible points of the other elements: a forward
/// `Index` on element `k != 0` then evaluates against element 0's (wrong)
/// interval, cutting feasible arguments and, on a genuine mismatch, declaring a
/// feasible model infeasible. That collapsed box reaches the certified LP dual
/// bound via `_fbbt_argument_box` (`milp_relaxation.py`), so the envelope built
/// over it can be invalid. Seeding from the element-wise UNION
/// (`min` lower, `max` upper) restores soundness: the block interval then
/// contains every element's feasible interval, so interval arithmetic over it is
/// a superset — FBBT can only *lose* tightening for the block, never cut a
/// feasible point. For a homogeneous block the union equals element 0, so this
/// is a no-op there (no regression for the common case).
pub(crate) fn seed_block_interval(v: &crate::expr::VarInfo) -> Interval {
    if v.lb.is_empty() || v.ub.is_empty() {
        return Interval::new(f64::NEG_INFINITY, f64::INFINITY);
    }
    let lo = v.lb.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.ub.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Interval::new(lo, hi)
}

/// Run FBBT to fixed-point on a model with an optional incumbent cutoff.
///
/// When `incumbent_bound` is `Some(bound)`, an additional synthetic constraint
/// is injected: `objective <= bound` (for minimize) or `objective >= bound`
/// (for maximize). This allows FBBT to exploit incumbent information for
/// tighter bounds without LP solves.
///
/// Returns tightened variable bounds (indexed by variable index, not offset).
pub fn fbbt_with_cutoff(
    model: &ModelRepr,
    max_iter: usize,
    tol: f64,
    incumbent_bound: Option<f64>,
) -> Vec<Interval> {
    let n_vars = model.variables.len();
    let mut var_bounds: Vec<Interval> = model.variables.iter().map(seed_block_interval).collect();

    // Determine the objective cutoff constraint (if any).
    let obj_cutoff: Option<(ExprId, Interval)> = incumbent_bound.map(|bound| {
        let output_bound = match model.objective_sense {
            ObjectiveSense::Minimize => Interval::new(f64::NEG_INFINITY, bound),
            ObjectiveSense::Maximize => Interval::new(bound, f64::INFINITY),
        };
        (model.objective, output_bound)
    });

    for _ in 0..max_iter {
        let old_bounds = var_bounds.clone();

        for constr in &model.constraints {
            let node_bounds = forward_propagate(&model.arena, constr.body, &var_bounds);

            let output_bound = match constr.sense {
                ConstraintSense::Le => Interval::new(f64::NEG_INFINITY, constr.rhs),
                ConstraintSense::Ge => Interval::new(constr.rhs, f64::INFINITY),
                ConstraintSense::Eq => Interval::point(constr.rhs),
            };

            let body_bound = node_bounds[constr.body.0];
            if body_bound
                .intersect(&output_bound)
                .is_empty_beyond(FEAS_TOL)
            {
                for b in &mut var_bounds {
                    *b = Interval::empty();
                }
                return var_bounds;
            }

            backward_propagate(
                &model.arena,
                constr.body,
                output_bound,
                &node_bounds,
                &mut var_bounds,
            );
        }

        // Propagate the objective cutoff constraint.
        if let Some((obj_expr, ref cutoff_bound)) = obj_cutoff {
            let node_bounds = forward_propagate(&model.arena, obj_expr, &var_bounds);
            let obj_bound = node_bounds[obj_expr.0];
            if obj_bound.intersect(cutoff_bound).is_empty_beyond(FEAS_TOL) {
                for b in &mut var_bounds {
                    *b = Interval::empty();
                }
                return var_bounds;
            }
            backward_propagate(
                &model.arena,
                obj_expr,
                *cutoff_bound,
                &node_bounds,
                &mut var_bounds,
            );
        }

        // Snap derived bounds to integrality (indicator-aware propagation).
        // Done before the convergence check so a freshly-fixed binary is fed
        // back into the next forward sweep within this same call.
        snap_integral_bounds(model, &mut var_bounds);

        let mut max_change = 0.0_f64;
        for i in 0..n_vars {
            let dlo = (var_bounds[i].lo - old_bounds[i].lo).abs();
            let dhi = (var_bounds[i].hi - old_bounds[i].hi).abs();
            max_change = max_change.max(dlo).max(dhi);
        }
        if max_change < tol {
            break;
        }
    }

    var_bounds
}

/// Run FBBT to fixed-point on a model.
///
/// Returns tightened variable bounds (indexed by variable index, not offset).
pub fn fbbt(model: &ModelRepr, max_iter: usize, tol: f64) -> Vec<Interval> {
    let n_vars = model.variables.len();
    // C-31: seed each block from the element-wise union of its bounds (a valid
    // outer bound for every element), NOT element 0 — see `seed_block_interval`.
    let mut var_bounds: Vec<Interval> = model.variables.iter().map(seed_block_interval).collect();

    for _ in 0..max_iter {
        let old_bounds = var_bounds.clone();

        for constr in &model.constraints {
            // Forward propagation.
            let node_bounds = forward_propagate(&model.arena, constr.body, &var_bounds);

            // Determine the output bound from the constraint sense and rhs.
            let output_bound = match constr.sense {
                ConstraintSense::Le => Interval::new(f64::NEG_INFINITY, constr.rhs),
                ConstraintSense::Ge => Interval::new(constr.rhs, f64::INFINITY),
                ConstraintSense::Eq => Interval::point(constr.rhs),
            };

            // Check feasibility: if the forward bound is incompatible
            // with the constraint, the problem is infeasible.
            let body_bound = node_bounds[constr.body.0];
            if body_bound
                .intersect(&output_bound)
                .is_empty_beyond(FEAS_TOL)
            {
                // Infeasible — mark all bounds as empty.
                for b in &mut var_bounds {
                    *b = Interval::empty();
                }
                return var_bounds;
            }

            // Backward propagation.
            backward_propagate(
                &model.arena,
                constr.body,
                output_bound,
                &node_bounds,
                &mut var_bounds,
            );
        }

        // Snap derived bounds to integrality (indicator-aware propagation).
        snap_integral_bounds(model, &mut var_bounds);

        // Check convergence.
        let mut max_change = 0.0_f64;
        for i in 0..n_vars {
            let dlo = (var_bounds[i].lo - old_bounds[i].lo).abs();
            let dhi = (var_bounds[i].hi - old_bounds[i].hi).abs();
            max_change = max_change.max(dlo).max(dhi);
        }
        if max_change < tol {
            break;
        }
    }

    var_bounds
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::*;

    // -- Interval arithmetic tests --

    #[test]
    fn test_interval_add() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 5.0);
        let r = interval_add(&a, &b);
        assert!((r.lo - 3.0).abs() < 1e-15);
        assert!((r.hi - 8.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_sub() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 5.0);
        let r = interval_sub(&a, &b);
        assert!((r.lo - (-4.0)).abs() < 1e-15);
        assert!((r.hi - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_mul_positive() {
        let a = Interval::new(2.0, 3.0);
        let b = Interval::new(4.0, 5.0);
        let r = interval_mul(&a, &b);
        assert!((r.lo - 8.0).abs() < 1e-15);
        assert!((r.hi - 15.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_mul_mixed() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(-1.0, 4.0);
        let r = interval_mul(&a, &b);
        assert!((r.lo - (-8.0)).abs() < 1e-15);
        assert!((r.hi - 12.0).abs() < 1e-15);
    }

    #[test]
    fn c22_interval_mul_zero_times_entire_is_zero() {
        // C-22: [0,0] * [-inf, inf]. Each corner product is 0 * (±inf) = NaN.
        // Before the fix, f64::min/max propagate NaN and the result is [NaN, NaN]
        // — a "lost tightening" bug (any variable intersected with a NaN interval
        // keeps its stale bound). By the interval convention 0 * anything = 0, so
        // the sound, informative enclosure is [0, 0].
        let a = Interval::point(0.0);
        let b = Interval::entire();
        let r = interval_mul(&a, &b);
        assert!(!r.lo.is_nan() && !r.hi.is_nan(), "result must not be NaN");
        assert_eq!(r.lo, 0.0);
        assert_eq!(r.hi, 0.0);
    }

    #[test]
    fn c22_interval_mul_never_nan_and_encloses_true_product() {
        // C-22 property test: over a grid of intervals including infinite
        // endpoints and zero-width [0,0] factors, interval_mul must (a) never
        // produce NaN endpoints and (b) contain the true product of every pair
        // of representative points drawn from the two intervals (rigorous
        // outer enclosure). A NaN endpoint fails containment silently, so we
        // check both explicitly.
        let ninf = f64::NEG_INFINITY;
        let pinf = f64::INFINITY;
        let bounds = [ninf, -3.0, -1.0, 0.0, 1.0, 2.5, pinf];
        for &alo in &bounds {
            for &ahi in &bounds {
                if alo > ahi {
                    continue;
                }
                for &blo in &bounds {
                    for &bhi in &bounds {
                        if blo > bhi {
                            continue;
                        }
                        let a = Interval::new(alo, ahi);
                        let b = Interval::new(blo, bhi);
                        let r = interval_mul(&a, &b);
                        assert!(
                            !r.lo.is_nan() && !r.hi.is_nan(),
                            "interval_mul({a:?}, {b:?}) produced NaN: {r:?}"
                        );
                        assert!(r.lo <= r.hi, "interval_mul({a:?}, {b:?}) => {r:?}");
                        // Containment: probe *finite* points from each factor and
                        // require the product lies within the result interval.
                        // A finite point drawn from an unbounded factor is still
                        // a member of that interval, so its product must be
                        // enclosed. (Degenerate point-at-infinity intervals like
                        // [-inf,-inf] have no finite members and are excluded by
                        // finite_probes returning empty.)
                        let a_pts = finite_probes(alo, ahi);
                        let b_pts = finite_probes(blo, bhi);
                        for &x in &a_pts {
                            for &y in &b_pts {
                                let p = x * y;
                                assert!(
                                    p >= r.lo - 1e-6 && p <= r.hi + 1e-6,
                                    "product {p} of {x}*{y} escaped {r:?} \
                                     for a={a:?} b={b:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Genuine finite members of `[lo, hi]` used as containment witnesses in
    /// the C-22 property test. Every returned value satisfies `lo <= v <= hi`.
    /// Unbounded sides are probed with a large finite magnitude (still a member);
    /// a degenerate point-at-infinity interval (e.g. `[-inf,-inf]`) has no finite
    /// members and returns empty.
    fn finite_probes(lo: f64, hi: f64) -> Vec<f64> {
        let mut pts = Vec::new();
        if lo.is_finite() {
            pts.push(lo);
        } else if lo == f64::NEG_INFINITY && hi > f64::NEG_INFINITY {
            // A large-magnitude negative member, but never above hi.
            let cap = if hi.is_finite() { hi } else { 1e6 };
            pts.push((-1e6_f64).min(cap));
        }
        if hi.is_finite() {
            pts.push(hi);
        } else if hi == f64::INFINITY && lo < f64::INFINITY {
            // A large-magnitude positive member, but never below lo.
            let floor = if lo.is_finite() { lo } else { -1e6 };
            pts.push((1e6_f64).max(floor));
        }
        if lo.is_finite() && hi.is_finite() && (hi - lo).abs() > 1e-15 {
            pts.push(lo + (hi - lo) * 0.5);
        }
        // Include 0 when it is a member — exercises the 0 * ±∞ corner.
        if lo <= 0.0 && hi >= 0.0 {
            pts.push(0.0);
        }
        pts
    }

    #[test]
    fn test_interval_div_no_zero() {
        let a = Interval::new(6.0, 12.0);
        let b = Interval::new(2.0, 3.0);
        let r = interval_div(&a, &b);
        assert!((r.lo - 2.0).abs() < 1e-15);
        assert!((r.hi - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_div_contains_zero() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(-1.0, 1.0);
        let r = interval_div(&a, &b);
        assert!(r.lo.is_infinite() && r.lo < 0.0);
        assert!(r.hi.is_infinite() && r.hi > 0.0);
    }

    #[test]
    fn test_interval_pow_even() {
        // [-2, 3]^2 = [0, 9]
        let a = Interval::new(-2.0, 3.0);
        let r = interval_pow_int(&a, 2);
        assert!((r.lo - 0.0).abs() < 1e-15);
        assert!((r.hi - 9.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_pow_odd() {
        // [-2, 3]^3 = [-8, 27]
        let a = Interval::new(-2.0, 3.0);
        let r = interval_pow_int(&a, 3);
        assert!((r.lo - (-8.0)).abs() < 1e-15);
        assert!((r.hi - 27.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_neg() {
        let a = Interval::new(1.0, 5.0);
        let r = interval_neg(&a);
        assert!((r.lo - (-5.0)).abs() < 1e-15);
        assert!((r.hi - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_interval_abs_positive() {
        let a = Interval::new(2.0, 5.0);
        let r = interval_abs(&a);
        assert!((r.lo - 2.0).abs() < 1e-15);
        assert!((r.hi - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_abs_negative() {
        let a = Interval::new(-5.0, -2.0);
        let r = interval_abs(&a);
        assert!((r.lo - 2.0).abs() < 1e-15);
        assert!((r.hi - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_abs_mixed() {
        let a = Interval::new(-3.0, 5.0);
        let r = interval_abs(&a);
        assert!((r.lo - 0.0).abs() < 1e-15);
        assert!((r.hi - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_exp() {
        let a = Interval::new(0.0, 1.0);
        let r = interval_exp(&a);
        assert!((r.lo - 1.0).abs() < 1e-14);
        assert!((r.hi - 1.0_f64.exp()).abs() < 1e-14);
    }

    #[test]
    fn test_interval_log() {
        let a = Interval::new(1.0, 10.0);
        let r = interval_log(&a);
        assert!((r.lo - 0.0).abs() < 1e-14);
        assert!((r.hi - 10.0_f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn test_interval_sqrt() {
        let a = Interval::new(4.0, 16.0);
        let r = interval_sqrt(&a);
        assert!((r.lo - 2.0).abs() < 1e-14);
        assert!((r.hi - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_interval_sin_small() {
        let a = Interval::new(0.0, PI / 2.0);
        let r = interval_sin(&a);
        assert!((r.lo - 0.0).abs() < 1e-14);
        assert!((r.hi - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_interval_sin_full() {
        let a = Interval::new(0.0, 2.0 * PI);
        let r = interval_sin(&a);
        assert!((r.lo - (-1.0)).abs() < 1e-14);
        assert!((r.hi - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_interval_cos() {
        let a = Interval::new(0.0, PI);
        let r = interval_cos(&a);
        assert!((r.lo - (-1.0)).abs() < 1e-14);
        assert!((r.hi - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_interval_empty() {
        let a = Interval::empty();
        assert!(a.is_empty());
        assert!(!a.contains(0.0));
    }

    #[test]
    fn test_interval_contains() {
        let a = Interval::new(1.0, 5.0);
        assert!(a.contains(3.0));
        assert!(a.contains(1.0));
        assert!(a.contains(5.0));
        assert!(!a.contains(0.0));
        assert!(!a.contains(6.0));
    }

    #[test]
    fn test_interval_width() {
        let a = Interval::new(1.0, 5.0);
        assert!((a.width() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_intersect() {
        let a = Interval::new(1.0, 5.0);
        let b = Interval::new(3.0, 7.0);
        let r = a.intersect(&b);
        assert!((r.lo - 3.0).abs() < 1e-15);
        assert!((r.hi - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_interval_intersect_empty() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(5.0, 7.0);
        let r = a.intersect(&b);
        assert!(r.is_empty());
    }

    // -- Forward propagation tests --

    fn make_simple_add_model() -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        // x (index 0) + y (index 1)
        let _x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let _y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: ExprId(0),
            right: ExprId(1),
        });
        (arena, sum)
    }

    #[test]
    fn test_forward_propagate_add() {
        let (arena, sum) = make_simple_add_model();
        let var_bounds = vec![Interval::new(1.0, 3.0), Interval::new(2.0, 5.0)];
        let bounds = forward_propagate(&arena, sum, &var_bounds);
        let result = bounds[sum.0];
        assert!((result.lo - 3.0).abs() < 1e-15);
        assert!((result.hi - 8.0).abs() < 1e-15);
    }

    #[test]
    fn test_forward_propagate_exp() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let exp_x = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Exp,
            args: vec![x],
        });
        let var_bounds = vec![Interval::new(0.0, 1.0)];
        let bounds = forward_propagate(&arena, exp_x, &var_bounds);
        let result = bounds[exp_x.0];
        assert!((result.lo - 1.0).abs() < 1e-14);
        assert!((result.hi - 1.0_f64.exp()).abs() < 1e-14);
    }

    // -- FBBT tests --

    fn make_linear_model() -> ModelRepr {
        // x + y <= 10, x in [0, 100], y in [0, 100]
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        // Objective: x (dummy)
        ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Continuous,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
            ],
            n_vars: 2,
        }
    }

    #[test]
    fn test_fbbt_linear_bound_tightening() {
        let model = make_linear_model();
        let bounds = fbbt(&model, 10, 1e-8);
        // x + y <= 10 with x >= 0, y >= 0
        // => x_ub should be tightened to 10 (when y = 0)
        // => y_ub should be tightened to 10 (when x = 0)
        assert!((bounds[0].lo - 0.0).abs() < 1e-10);
        assert!((bounds[0].hi - 10.0).abs() < 1e-10);
        assert!((bounds[1].lo - 0.0).abs() < 1e-10);
        assert!((bounds[1].hi - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fbbt_exp_bound_tightening() {
        // exp(x) <= 10 with x in [0, 100]
        // => x_ub should be tightened to ln(10) ≈ 2.302
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let exp_x = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Exp,
            args: vec![x],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: exp_x,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![100.0],
            }],
            n_vars: 1,
        };
        let bounds = fbbt(&model, 10, 1e-8);
        assert!((bounds[0].lo - 0.0).abs() < 1e-10);
        assert!((bounds[0].hi - 10.0_f64.ln()).abs() < 1e-8);
    }

    #[test]
    fn test_fbbt_equality_constraint() {
        // x + y = 5, x in [0, 10], y in [0, 10]
        // => x in [0, 5], y in [0, 5]
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Eq,
                rhs: 5.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![10.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Continuous,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![10.0],
                },
            ],
            n_vars: 2,
        };
        let bounds = fbbt(&model, 10, 1e-8);
        assert!((bounds[0].lo - 0.0).abs() < 1e-10);
        assert!((bounds[0].hi - 5.0).abs() < 1e-10);
        assert!((bounds[1].lo - 0.0).abs() < 1e-10);
        assert!((bounds[1].hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fbbt_mul_constraint() {
        // 2*x <= 10, x in [0, 100]
        // => x_ub should be tightened to 5
        let mut arena = ExprArena::new();
        let c2 = arena.add(ExprNode::Constant(2.0));
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c2,
            right: x,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(1),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: prod,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![100.0],
            }],
            n_vars: 1,
        };
        let bounds = fbbt(&model, 10, 1e-8);
        assert!((bounds[0].lo - 0.0).abs() < 1e-10);
        assert!((bounds[0].hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fbbt_ge_constraint() {
        // x >= 5, x in [0, 100]
        // => x_lb should be tightened to 5
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x,
                sense: ConstraintSense::Ge,
                rhs: 5.0,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![100.0],
            }],
            n_vars: 1,
        };
        let bounds = fbbt(&model, 10, 1e-8);
        assert!((bounds[0].lo - 5.0).abs() < 1e-10);
        assert!((bounds[0].hi - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_fbbt_sqrt_constraint() {
        // sqrt(x) <= 3, x in [0, 100]
        // => x_ub should be tightened to 9
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let sqrt_x = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Sqrt,
            args: vec![x],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sqrt_x,
                sense: ConstraintSense::Le,
                rhs: 3.0,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![100.0],
            }],
            n_vars: 1,
        };
        let bounds = fbbt(&model, 10, 1e-8);
        assert!((bounds[0].lo - 0.0).abs() < 1e-10);
        assert!((bounds[0].hi - 9.0).abs() < 1e-8);
    }

    #[test]
    fn test_fbbt_convergence_one_iteration() {
        // Simple enough that one iteration suffices.
        let model = make_linear_model();
        let bounds = fbbt(&model, 1, 1e-8);
        assert!((bounds[0].hi - 10.0).abs() < 1e-10);
        assert!((bounds[1].hi - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fbbt_sum_over() {
        // x + y + z <= 15, all in [0, 100]
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let z = arena.add(ExprNode::Variable {
            name: "z".into(),
            index: 2,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::SumOver {
            terms: vec![x, y, z],
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0),
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 15.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Continuous,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
                VarInfo {
                    name: "z".into(),
                    var_type: VarType::Continuous,
                    offset: 2,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
            ],
            n_vars: 3,
        };
        let bounds = fbbt(&model, 10, 1e-8);
        // Each variable should be tightened to [0, 15].
        for b in &bounds {
            assert!((b.lo - 0.0).abs() < 1e-10);
            assert!((b.hi - 15.0).abs() < 1e-10);
        }
    }

    // -- fbbt_with_cutoff tests --

    #[test]
    fn test_fbbt_with_cutoff_basic_tightening() {
        // min x s.t. x + y <= 10, x in [0,100], y in [0,100]
        // With cutoff=7: objective x <= 7
        // => x in [0, 7], y in [0, 10]
        let model = make_linear_model();
        let bounds = fbbt_with_cutoff(&model, 10, 1e-8, Some(7.0));
        assert!((bounds[0].lo - 0.0).abs() < 1e-10);
        assert!((bounds[0].hi - 7.0).abs() < 1e-10);
        assert!((bounds[1].lo - 0.0).abs() < 1e-10);
        assert!((bounds[1].hi - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fbbt_with_cutoff_none_matches_fbbt() {
        // None cutoff should match plain fbbt
        let model = make_linear_model();
        let bounds_plain = fbbt(&model, 10, 1e-8);
        let bounds_cutoff = fbbt_with_cutoff(&model, 10, 1e-8, None);
        for (a, b) in bounds_plain.iter().zip(bounds_cutoff.iter()) {
            assert!((a.lo - b.lo).abs() < 1e-14);
            assert!((a.hi - b.hi).abs() < 1e-14);
        }
    }

    #[test]
    fn test_fbbt_with_cutoff_infeasibility() {
        // min x s.t. x + y <= 10, x >= 0, y >= 0
        // With cutoff=-1 (x <= -1), infeasible since x >= 0
        let model = make_linear_model();
        let bounds = fbbt_with_cutoff(&model, 10, 1e-8, Some(-1.0));
        for b in &bounds {
            assert!(b.is_empty(), "Expected infeasible (empty bounds)");
        }
    }

    #[test]
    fn test_fbbt_with_cutoff_maximize() {
        // max x s.t. x + y <= 10, x in [0,100], y in [0,100]
        // With cutoff=3: objective x >= 3 => x in [3, 10]
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        let model = ModelRepr {
            arena,
            objective: ExprId(0), // x
            objective_sense: ObjectiveSense::Maximize,
            constraints: vec![ConstraintRepr {
                body: sum,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: Some("c1".into()),
            }],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Continuous,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![0.0],
                    ub: vec![100.0],
                },
            ],
            n_vars: 2,
        };
        let bounds = fbbt_with_cutoff(&model, 10, 1e-8, Some(3.0));
        assert!((bounds[0].lo - 3.0).abs() < 1e-10);
        assert!((bounds[0].hi - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fbbt_with_cutoff_nonlinear_obj() {
        // min exp(x) s.t. x in [-10, 10]
        // With cutoff=e^2 (~7.389): exp(x) <= e^2 => x <= 2
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let exp_x = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Exp,
            args: vec![x],
        });
        let model = ModelRepr {
            arena,
            objective: exp_x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![-10.0],
                ub: vec![10.0],
            }],
            n_vars: 1,
        };
        let cutoff = 2.0_f64.exp(); // e^2
        let bounds = fbbt_with_cutoff(&model, 10, 1e-8, Some(cutoff));
        assert!((bounds[0].lo - (-10.0)).abs() < 1e-8);
        assert!((bounds[0].hi - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_fbbt_with_cutoff_even_power_straddling_base() {
        // min (1 - x)^2 over x in [-4, 11]. The base (1 - x) straddles zero
        // (forward range [-10, 5]), so the even-power backward rule must still
        // invert the cutoff: (1-x)^2 <= 0.01 => |1-x| <= 0.1 => x in [0.9, 1.1].
        // This is the Rosenbrock-style shifted-square pattern: without the fix
        // the box never collapses and certification is pathologically slow.
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let one = arena.add(ExprNode::Constant(1.0));
        let diff = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: one,
            right: x,
        });
        let two = arena.add(ExprNode::Constant(2.0));
        let sq = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: diff,
            right: two,
        });
        let model = ModelRepr {
            arena,
            objective: sq,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![-4.0],
                ub: vec![11.0],
            }],
            n_vars: 1,
        };
        let bounds = fbbt_with_cutoff(&model, 10, 1e-8, Some(0.01));
        assert!(
            (bounds[0].lo - 0.9).abs() < 1e-6,
            "expected lo ~0.9, got {}",
            bounds[0].lo
        );
        assert!(
            (bounds[0].hi - 1.1).abs() < 1e-6,
            "expected hi ~1.1, got {}",
            bounds[0].hi
        );
    }

    #[test]
    fn test_backward_even_power_negative_base() {
        // For a known-nonpositive base, u^2 in [4, 9] => u in [-3, -2].
        // Model: objective = x^2, x in [-5, -1] (so base x is nonpositive),
        // cutoff 9 => x^2 <= 9 => x in [-3, -1] (intersect with forward [-5,-1]).
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let two = arena.add(ExprNode::Constant(2.0));
        let sq = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: two,
        });
        let model = ModelRepr {
            arena,
            objective: sq,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![-5.0],
                ub: vec![-1.0],
            }],
            n_vars: 1,
        };
        let bounds = fbbt_with_cutoff(&model, 10, 1e-8, Some(9.0));
        assert!(
            (bounds[0].lo - (-3.0)).abs() < 1e-6,
            "expected lo ~-3, got {}",
            bounds[0].lo
        );
        assert!(
            (bounds[0].hi - (-1.0)).abs() < 1e-6,
            "expected hi ~-1, got {}",
            bounds[0].hi
        );
    }

    // -- feasibility-tolerance regression tests (issue #27a) --

    #[test]
    fn test_is_empty_beyond_tolerance() {
        // An eps-scale inverted interval is empty in the strict sense but
        // feasible within tolerance — it must not be treated as infeasible.
        let eps = Interval::new(1.0, 1.0 - 1e-9);
        assert!(eps.is_empty());
        assert!(!eps.is_empty_beyond(FEAS_TOL));

        // A genuinely inverted interval is empty beyond tolerance.
        let real = Interval::new(1.0, 0.9);
        assert!(real.is_empty());
        assert!(real.is_empty_beyond(FEAS_TOL));

        // The canonical empty interval is empty beyond any finite tolerance.
        assert!(Interval::empty().is_empty_beyond(FEAS_TOL));
    }

    /// Build a one-variable model `x` fixed to `[0, 0]` with a single `x >= rhs`
    /// constraint. With `rhs` a small positive number the constraint is violated
    /// by exactly `rhs` — the shape of a GDP hull perspective residual at an
    /// integer face.
    fn make_eps_violated_model(rhs: f64) -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x,
                sense: ConstraintSense::Ge,
                rhs,
                name: Some("c1".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![0.0],
            }],
            n_vars: 1,
        }
    }

    #[test]
    fn test_fbbt_eps_violation_not_infeasible() {
        // x = 0, constraint x >= 1e-9. The body misses the bound by 1e-9, well
        // within the feasibility tolerance: FBBT must NOT collapse all bounds to
        // empty (which would fabricate an unsound infeasibility certificate).
        let model = make_eps_violated_model(1e-9);
        let bounds = fbbt(&model, 5, 1e-8);
        assert!(
            bounds[0].lo.is_finite() && bounds[0].hi.is_finite(),
            "eps-scale violation must not be treated as infeasible"
        );
        // The variable stays pinned near its fixed value, not blown up to empty.
        assert!(bounds[0].lo <= 1e-6 && bounds[0].hi >= -1e-6);
    }

    #[test]
    fn test_fbbt_real_violation_is_infeasible() {
        // x = 0, constraint x >= 0.5. The violation (0.5) exceeds the feasibility
        // tolerance, so FBBT must still detect infeasibility and empty the bounds.
        let model = make_eps_violated_model(0.5);
        let bounds = fbbt(&model, 5, 1e-8);
        assert!(
            bounds.iter().all(|b| b.is_empty()),
            "a violation beyond the feasibility tolerance must be infeasible"
        );
    }

    // ── Integrality-aware (binary-indicator) propagation ──────────

    fn scalar(arena: &mut ExprArena, name: &str, index: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.into(),
            index,
            size: 1,
            shape: vec![],
        })
    }

    fn ivar(name: &str, vt: VarType, lb: f64, ub: f64) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: vt,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    /// Build `x - coeff*b ≤ 0` with x continuous and b binary.
    /// With coeff = M this is the big-M guard `x ≤ M·b`.
    fn make_indicator_model(x_lb: f64, x_ub: f64, coeff: f64) -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = scalar(&mut arena, "x", 0);
        let b = scalar(&mut arena, "b", 1);
        let c = arena.add(ExprNode::Constant(coeff));
        let mb = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c,
            right: b,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x,
            right: mb,
        });
        ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: Some("guard".into()),
            }],
            variables: vec![
                ivar("x", VarType::Continuous, x_lb, x_ub),
                ivar("b", VarType::Binary, 0.0, 1.0),
            ],
            n_vars: 2,
        }
    }

    #[test]
    fn test_indicator_backward_infers_binary() {
        // Guard x ≤ 10·b, with branching already forcing x ∈ [3, 10].
        // Then b ≥ x/10 ≥ 0.3, and since b ∈ {0,1}, b must be 1.
        let model = make_indicator_model(3.0, 10.0, 10.0);
        let bounds = fbbt(&model, 8, 1e-9);
        assert!(!bounds.iter().any(|b| b.is_empty()));
        assert!(
            (bounds[1].lo - 1.0).abs() < 1e-9 && (bounds[1].hi - 1.0).abs() < 1e-9,
            "binary should be inferred = 1, got {:?}",
            bounds[1]
        );
    }

    #[test]
    fn test_indicator_forward_activates_guard() {
        // Guard x ≤ 10·b with b fixed to 0 (e.g. by branching). The guard then
        // forces x ≤ 0; combined with x ≥ 0 this pins x to 0.
        let mut model = make_indicator_model(0.0, 10.0, 10.0);
        model.variables[1].lb = vec![0.0];
        model.variables[1].ub = vec![0.0]; // b = 0
        let bounds = fbbt(&model, 8, 1e-9);
        assert!(!bounds.iter().any(|b| b.is_empty()));
        assert!(
            bounds[0].hi <= 1e-6,
            "deactivated guard should force x ≤ 0, got {:?}",
            bounds[0]
        );
    }

    #[test]
    fn test_indicator_infeasible_when_binary_squeezed_out() {
        // A binary forced to the fractional value 0.5 has no integer realisation:
        // snapping yields the empty interval [1, 0], a genuine infeasibility.
        let mut arena = ExprArena::new();
        let b = scalar(&mut arena, "b", 0);
        let model = ModelRepr {
            arena,
            objective: b,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: b,
                sense: ConstraintSense::Eq,
                rhs: 0.5,
                name: Some("pin".into()),
            }],
            variables: vec![ivar("b", VarType::Binary, 0.0, 1.0)],
            n_vars: 1,
        };
        let bounds = fbbt(&model, 8, 1e-9);
        assert!(
            bounds.iter().any(|b| b.is_empty()),
            "a binary squeezed to neither 0 nor 1 must be infeasible, got {:?}",
            bounds
        );
    }

    #[test]
    fn test_integer_bounds_snapped_inward() {
        // A general integer variable: 3·n ∈ [7, 17] ⇒ n ∈ [2.33, 5.67],
        // which snaps to the integer hull [3, 5].
        let mut arena = ExprArena::new();
        let n = scalar(&mut arena, "n", 0);
        let c = arena.add(ExprNode::Constant(3.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c,
            right: n,
        });
        let model = ModelRepr {
            arena,
            objective: n,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body,
                    sense: ConstraintSense::Ge,
                    rhs: 7.0,
                    name: Some("lo".into()),
                },
                ConstraintRepr {
                    body,
                    sense: ConstraintSense::Le,
                    rhs: 17.0,
                    name: Some("hi".into()),
                },
            ],
            variables: vec![ivar("n", VarType::Integer, 0.0, 100.0)],
            n_vars: 1,
        };
        let bounds = fbbt(&model, 8, 1e-9);
        assert!((bounds[0].lo - 3.0).abs() < 1e-9, "lo {:?}", bounds[0]);
        assert!((bounds[0].hi - 5.0).abs() < 1e-9, "hi {:?}", bounds[0]);
    }

    #[test]
    fn test_eps_residual_does_not_fix_binary() {
        // Guard x ≤ 1e-9·b with x ∈ [0, 0]. Backward leaves b only an eps-scale
        // lower bound; the integrality pullback must NOT fix b to 1 — doing so
        // would wrongly eliminate the b = 0 disjunct and yield an unsound bound.
        let model = make_indicator_model(0.0, 0.0, 1e-9);
        let bounds = fbbt(&model, 8, 1e-9);
        assert!(!bounds.iter().any(|b| b.is_empty()));
        assert!(
            bounds[1].lo <= 1e-9 && bounds[1].hi >= 1.0 - 1e-9,
            "eps residual must leave the binary free, got {:?}",
            bounds[1]
        );
    }

    // ── C-31 (=TG-1) — FBBT array-block seeding (FIXED) ──
    //
    // `fbbt()` carries ONE `Interval` per variable *block*, and `eval_node_interval`
    // resolves every `Index{base,col}` node to that single shared block interval
    // (the column is ignored). The old seed used `v.lb.first()`/`v.ub.first()` —
    // element 0's bounds — so an array variable with heterogeneous per-element
    // bounds had element 0's (tighter) bounds illegally propagated onto every
    // other element, cutting feasible points and (on a genuine mismatch) declaring
    // a feasible model infeasible. FIX (`seed_block_interval`): seed each block
    // from the element-wise UNION [min lb, max ub], a valid outer bound for every
    // element — so the block interval never excludes a feasible argument. These
    // two tests assert the FIXED behaviour (no feasible cut; no false infeasible);
    // they FAIL on the pre-fix element-0 seed. See also the Python-side consumer
    // test `test_c31_fbbt_argument_box_envelope_contains_feasible` which pins the
    // certified-LP-relaxation reach (`_fbbt_argument_box` / `milp_relaxation.py`).

    /// Build a single continuous array variable block `x` of `size` with the given
    /// element-wise bounds, plus a constraint on element `col`: `x[col] {sense} rhs`.
    fn array_var_model(
        lb: Vec<f64>,
        ub: Vec<f64>,
        col: usize,
        sense: ConstraintSense,
        rhs: f64,
    ) -> ModelRepr {
        let size = lb.len();
        let mut arena = ExprArena::new();
        let xvar = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size,
            shape: vec![size],
        });
        let idx = arena.add(ExprNode::Index {
            base: xvar,
            index: IndexSpec::Scalar(col),
        });
        ModelRepr {
            arena,
            objective: idx,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: idx,
                sense,
                rhs,
                name: Some("c".into()),
            }],
            variables: vec![VarInfo {
                name: "x".into(),
                var_type: VarType::Continuous,
                offset: 0,
                size,
                shape: vec![size],
                lb,
                ub,
            }],
            n_vars: size,
        }
    }

    #[test]
    fn c31_array_block_seeds_from_element_union_not_element0() {
        // x is a length-2 continuous array with heterogeneous per-element bounds:
        // lb=[8,0], ub=[10,10]. Element 1 is genuinely free in [0,10]. Constraint
        // touches element 1 trivially: `x[1] >= 0` (always satisfiable). The old
        // C-31 collapse seeded the whole block from element 0 → [8,10], erasing
        // the feasible region x[1] ∈ [0,8). The fix seeds each block from the
        // element-wise UNION [min lb, max ub] = [0,10], a valid outer bound for
        // every element, so the feasible region is preserved.
        let model = array_var_model(
            vec![8.0, 0.0],
            vec![10.0, 10.0],
            1,
            ConstraintSense::Ge,
            0.0,
        );
        let bounds = fbbt(&model, 8, 1e-9);
        // One interval per BLOCK (n_vars here = variables.len() == 1).
        assert_eq!(
            bounds.len(),
            1,
            "fbbt returns one interval per block, not per element"
        );
        // C-31 FIXED: the block interval must be the element-wise union outer
        // bound, so its lower bound is element 1's 0.0 — NOT element-0's 8.0.
        // A lower bound above 0.0 would cut the feasible region x[1] ∈ [0,8).
        assert!(
            bounds[0].lo <= 0.0 + 1e-9,
            "C-31: block must not collapse to element-0 lb=8; feasible x[1]∈[0,8) \
             would be cut. got {:?}",
            bounds[0]
        );
        assert!(
            bounds[0].hi >= 10.0 - 1e-9,
            "C-31: block upper bound must cover every element (10.0), got {:?}",
            bounds[0]
        );
    }

    #[test]
    fn c31_heterogeneous_block_no_false_infeasible() {
        // x length-2: lb=[5,0], ub=[5,3]. Element 0 is fixed at 5; element 1 is
        // free in [0,3]. Constraint `x[1] <= 3` is trivially satisfiable
        // (x=[5, 0..3] is feasible), so FBBT must NOT report infeasible.
        // The old C-31 collapse seeded the block from element 0 → [5,5]; the
        // Index on element 1 resolved to [5,5]; intersecting with (-inf,3] was
        // empty → false infeasible. The union seed [0,5] intersected with
        // (-inf,3] is [0,3] → feasible.
        let model = array_var_model(vec![5.0, 0.0], vec![5.0, 3.0], 1, ConstraintSense::Le, 3.0);
        let bounds = fbbt(&model, 8, 1e-9);
        // C-31 FIXED: a feasible model must NOT be reported infeasible.
        // "FBBT never reports feasible as infeasible."
        assert!(
            !bounds.iter().any(|b| b.is_empty()),
            "C-31: feasible model x=[5, 0..3] must not be declared infeasible, \
             got {:?}",
            bounds
        );
    }
}
