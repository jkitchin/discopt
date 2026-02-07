//! Feasibility-Based Bound Tightening (FBBT).
//!
//! Implements interval arithmetic and forward/backward propagation
//! through the expression DAG to tighten variable bounds.

use crate::expr::{
    BinOp, ConstraintSense, ExprArena, ExprId, ExprNode, MathFunc, ModelRepr, UnOp,
};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────
// Interval type
// ─────────────────────────────────────────────────────────────

/// A closed interval [lo, hi].
///
/// An interval with `lo > hi` is empty, representing infeasibility.
#[derive(Debug, Clone, Copy)]
pub struct Interval {
    pub lo: f64,
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

    /// A point interval [v, v].
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

/// [a,b] + [c,d] = [a+c, b+d]
pub fn interval_add(a: &Interval, b: &Interval) -> Interval {
    Interval::new(a.lo + b.lo, a.hi + b.hi)
}

/// [a,b] - [c,d] = [a-d, b-c]
pub fn interval_sub(a: &Interval, b: &Interval) -> Interval {
    Interval::new(a.lo - b.hi, a.hi - b.lo)
}

/// [a,b] * [c,d] using all four endpoint products.
pub fn interval_mul(a: &Interval, b: &Interval) -> Interval {
    let p1 = a.lo * b.lo;
    let p2 = a.lo * b.hi;
    let p3 = a.hi * b.lo;
    let p4 = a.hi * b.hi;
    Interval::new(
        p1.min(p2).min(p3).min(p4),
        p1.max(p2).max(p3).max(p4),
    )
}

/// [a,b] / [c,d] with division-by-zero handling.
pub fn interval_div(a: &Interval, b: &Interval) -> Interval {
    if b.lo <= 0.0 && b.hi >= 0.0 {
        // Denominator contains zero — result is the entire real line.
        Interval::entire()
    } else {
        let inv_b = Interval::new(1.0 / b.hi, 1.0 / b.lo);
        interval_mul(a, &inv_b)
    }
}

/// [a,b]^n for integer exponent.
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

/// [a,b]^[c,d] for general power.
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

/// neg([a,b]) = [-b, -a]
pub fn interval_neg(a: &Interval) -> Interval {
    Interval::new(-a.hi, -a.lo)
}

/// abs([a,b])
pub fn interval_abs(a: &Interval) -> Interval {
    if a.lo >= 0.0 {
        *a
    } else if a.hi <= 0.0 {
        Interval::new(-a.hi, -a.lo)
    } else {
        Interval::new(0.0, a.lo.abs().max(a.hi.abs()))
    }
}

/// exp([a,b]) = [exp(a), exp(b)]
pub fn interval_exp(a: &Interval) -> Interval {
    Interval::new(a.lo.exp(), a.hi.exp())
}

/// log([a,b]) = [log(max(a, eps)), log(b)]
pub fn interval_log(a: &Interval) -> Interval {
    let lo = a.lo.max(f64::MIN_POSITIVE);
    if a.hi <= 0.0 {
        return Interval::empty();
    }
    Interval::new(lo.ln(), a.hi.ln())
}

/// log2([a,b])
pub fn interval_log2(a: &Interval) -> Interval {
    let lo = a.lo.max(f64::MIN_POSITIVE);
    if a.hi <= 0.0 {
        return Interval::empty();
    }
    Interval::new(lo.log2(), a.hi.log2())
}

/// log10([a,b])
pub fn interval_log10(a: &Interval) -> Interval {
    let lo = a.lo.max(f64::MIN_POSITIVE);
    if a.hi <= 0.0 {
        return Interval::empty();
    }
    Interval::new(lo.log10(), a.hi.log10())
}

/// sqrt([a,b]) = [sqrt(max(a,0)), sqrt(b)]
pub fn interval_sqrt(a: &Interval) -> Interval {
    if a.hi < 0.0 {
        return Interval::empty();
    }
    Interval::new(a.lo.max(0.0).sqrt(), a.hi.sqrt())
}

/// sin([a,b]) with periodicity handling.
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

/// cos([a,b]) with periodicity handling.
pub fn interval_cos(a: &Interval) -> Interval {
    // cos(x) = sin(x + pi/2)
    interval_sin(&Interval::new(a.lo + PI / 2.0, a.hi + PI / 2.0))
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
                MathFunc::Tan => {
                    // Conservative: tan can diverge, return entire.
                    Interval::entire()
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
                MathFunc::Norm2 => {
                    // |x| for single arg.
                    if args.len() == 1 {
                        interval_abs(&a0)
                    } else {
                        // sqrt(sum(x_i^2)) — conservative.
                        Interval::new(0.0, f64::INFINITY)
                    }
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
                            // base^n in [lo, hi] => base in [lo^(1/n), hi^(1/n)]
                            // (for odd n or base >= 0)
                            if exp_int % 2 == 1 || l.lo >= 0.0 {
                                let inv = 1.0 / exp_int as f64;
                                let new_lo = if tightened.lo >= 0.0 {
                                    tightened.lo.powf(inv)
                                } else if exp_int % 2 == 1 {
                                    -((-tightened.lo).powf(inv))
                                } else {
                                    0.0
                                };
                                let new_hi = if tightened.hi >= 0.0 {
                                    tightened.hi.powf(inv)
                                } else if exp_int % 2 == 1 {
                                    -((-tightened.hi).powf(inv))
                                } else {
                                    // Even power can't be negative.
                                    return;
                                };
                                let new_base = Interval::new(new_lo, new_hi);
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
                _ => {
                    // No backward propagation for other functions.
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
// Fixed-point FBBT
// ─────────────────────────────────────────────────────────────

/// Run FBBT to fixed-point on a model.
///
/// Returns tightened variable bounds (indexed by variable index, not offset).
pub fn fbbt(model: &ModelRepr, max_iter: usize, tol: f64) -> Vec<Interval> {
    let n_vars = model.variables.len();
    let mut var_bounds: Vec<Interval> = model
        .variables
        .iter()
        .map(|v| {
            // Use the first element's bounds (scalar variables).
            Interval::new(
                v.lb.first().copied().unwrap_or(f64::NEG_INFINITY),
                v.ub.first().copied().unwrap_or(f64::INFINITY),
            )
        })
        .collect();

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
            if body_bound.intersect(&output_bound).is_empty() {
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
}
