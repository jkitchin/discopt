//! Feasibility-Based Bound Tightening (FBBT).
//!
//! Implements interval arithmetic and forward/backward propagation
//! through the expression DAG to tighten variable bounds.

use crate::expr::{
    xlogx, BinOp, ConstraintSense, ExprArena, ExprId, ExprNode, MathFunc, ModelRepr,
    ObjectiveSense, UnOp, VarType,
};
use std::f64::consts::PI;
use std::time::Instant;

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

    /// Whether the interval is *formally* inverted (`lo > hi`), with **zero**
    /// tolerance.
    ///
    /// This is a syntactic predicate, **not** a feasibility verdict. Do not use
    /// it to conclude that a constraint, a node, or a presolve sweep is
    /// infeasible — use [`is_empty_beyond`] for that. It is deliberately *not*
    /// named `is_empty` (#907): under that name it read as a legitimate
    /// emptiness test and was used to fathom B&B nodes and abort presolve
    /// sweeps, so a crossing of 8.5e-14 discarded a live region.
    ///
    /// Legitimate uses are guards where either answer is sound — early-returning
    /// an already-degenerate interval unchanged, or filtering it out of a
    /// candidate list.
    pub fn is_formally_inverted(&self) -> bool {
        self.lo > self.hi
    }

    /// Whether the interval is empty by more than a feasibility tolerance,
    /// i.e. `lo - hi > tol`.
    ///
    /// **This is the only predicate that may be used to conclude
    /// infeasibility.** Approximate reformulations — notably the GDP hull
    /// perspective form `y * f(v / y)` with a clamp `y + eps` — leave eps-scale
    /// (~1e-8) residuals at integer faces. A strict `lo > hi` check mistakes
    /// that numerical noise for infeasibility and can fix a disjunction's
    /// selector incorrectly, producing an unsound bound. A tolerance-aware check
    /// declares infeasibility only when the violation exceeds the feasibility
    /// tolerance.
    ///
    /// The threshold is measured, not assumed. Instrumenting both acting sites
    /// over the in-repo corpus (10,105 predicate evaluations, #907 experiment A)
    /// found the crossing-magnitude distribution degenerate at two points:
    /// rounding noise at 1e-14 absolute / 1e-16 relative, and genuine
    /// infeasibility at exactly 1.0 (binary domain wipeout, `[1.0, 0.0]`) plus
    /// the explicit `[inf, -inf]` sentinel of [`Interval::empty`]. `FEAS_TOL`
    /// sits ~7 orders above the largest noise crossing and ~6 orders below the
    /// smallest genuine one, and nothing was observed in between. Note the
    /// genuine population is unsampled in `(FEAS_TOL, 1.0)`.
    pub fn is_empty_beyond(&self, tol: f64) -> bool {
        self.lo - self.hi > tol
    }

    /// Whether `x` is contained in the interval.
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }

    /// Repair a sub-tolerance inversion in place, returning `true` if it did.
    ///
    /// A crossing of `lo - hi` in `(0, tol]` is two derivations of the same
    /// quantity disagreeing in their last ulps, not an infeasibility. Widen to
    /// `[min(lo, hi), max(lo, hi)]` — the smallest interval containing **both**
    /// endpoints, so whichever derivation was the sound one keeps its endpoint
    /// and no feasible point is cut.
    ///
    /// Widening (not snapping to a midpoint) is the whole point: a midpoint
    /// collapse is *tighter* than at least one sound endpoint and could cut the
    /// optimum. A crossing beyond `tol`, and the explicit `[inf, -inf]` empty
    /// sentinel, are left untouched — they are real verdicts.
    ///
    /// Repairing is the necessary second half of the #907 fix. Declining to
    /// *declare* emptiness is not enough if an inverted interval still escapes
    /// to a consumer: `lo > hi` on an LP column bound reproduces the same false
    /// infeasibility one layer down.
    pub fn repair_if_subtol_inverted(&mut self, tol: f64) -> bool {
        let cross = self.lo - self.hi;
        if cross > 0.0 && cross <= tol {
            let (lo, hi) = (self.lo.min(self.hi), self.lo.max(self.hi));
            self.lo = lo;
            self.hi = hi;
            true
        } else {
            false
        }
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

/// Whether any interval in `bounds` is empty by more than `tol`.
///
/// The single sanctioned way for a caller to conclude "this box is infeasible".
/// See [`Interval::is_empty_beyond`] for why the strict `lo > hi` form is not.
pub fn any_empty_beyond(bounds: &[Interval], tol: f64) -> bool {
    bounds.iter().any(|b| b.is_empty_beyond(tol))
}

/// Repair every sub-`tol` inversion in `bounds` in place; returns how many.
///
/// Pair this with [`any_empty_beyond`] at any site that acts on an emptiness
/// verdict: repair first so no inverted interval escapes to a consumer, then
/// test. A non-zero return is a numerical smell worth surfacing — it means some
/// pass produced a formally-empty interval that was pure floating-point noise —
/// so callers thread the count out rather than absorbing it silently.
pub fn repair_subtol_crossings(bounds: &mut [Interval], tol: f64) -> usize {
    bounds
        .iter_mut()
        .map(|b| usize::from(b.repair_if_subtol_inverted(tol)))
        .sum()
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
    // Formal inversion (#907): this clamps the base to its non-negative part and
    // asks whether anything survived. Returning `entire()` is the maximally
    // conservative answer, so either verdict is sound; no infeasibility is
    // concluded.
    let b = Interval::new(base.lo.max(0.0), base.hi.max(0.0));
    if b.is_formally_inverted() || b.hi < 0.0 {
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

/// Push the direct children of `id` onto `out`.
fn push_children(arena: &ExprArena, id: ExprId, out: &mut Vec<ExprId>) {
    match arena.get(id) {
        ExprNode::BinaryOp { left, right, .. } | ExprNode::MatMul { left, right } => {
            out.push(*left);
            out.push(*right);
        }
        ExprNode::UnaryOp { operand, .. } | ExprNode::Sum { operand, .. } => out.push(*operand),
        ExprNode::FunctionCall { args, .. } => out.extend(args.iter().copied()),
        ExprNode::Index { base, .. } => out.push(*base),
        ExprNode::SumOver { terms } => out.extend(terms.iter().copied()),
        ExprNode::Variable { .. }
        | ExprNode::Constant(_)
        | ExprNode::ConstantArray(_, _)
        | ExprNode::Parameter { .. } => {}
    }
}

/// Reusable buffers for [`forward_propagate_into`], allocated once per FBBT
/// pass instead of once per constraint.
///
/// This type exists because forward propagation used to evaluate **every node
/// in the arena** for **every constraint**, making one FBBT sweep
/// `O(n_constraints x arena_len)` rather than `O(sum of constraint sizes)`.
/// On a set-covering model with ~900 rows that is ~1e8 interval evaluations
/// per sweep, which is why root probing — five sweeps per probe, two probes
/// per binary — consumed its entire time budget and tightened nothing
/// (measured: probing 60,362 ms, 27,900 work units, 0 bounds tightened).
///
/// The `seen`/`epoch` stamp is what keeps the per-call cost proportional to
/// the subtree: clearing an `O(arena_len)` visited array per constraint would
/// re-introduce the same quadratic, just with a cheaper constant.
#[derive(Debug, Default)]
pub struct FwdScratch {
    bounds: Vec<Interval>,
    seen: Vec<u32>,
    epoch: u32,
    stack: Vec<ExprId>,
    order: Vec<usize>,
}

impl FwdScratch {
    /// An empty scratch buffer; it sizes itself on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare for a propagation over an arena of `n` nodes.
    fn begin(&mut self, n: usize) {
        if self.bounds.len() != n {
            self.bounds = vec![Interval::entire(); n];
            self.seen = vec![0u32; n];
            self.epoch = 0;
        }
        self.epoch = match self.epoch.checked_add(1) {
            Some(e) => e,
            None => {
                // Wrapped. A stale stamp could alias the new epoch and make a
                // node look already-evaluated, so clear before reusing 1.
                self.seen.iter_mut().for_each(|s| *s = 0);
                1
            }
        };
    }
}

/// Forward-propagate interval bounds over the subtree rooted at `id`, reusing
/// `scratch`.
///
/// Returns the whole node-bounds slice, but **only the slots reachable from
/// `id` are written by this call**; every other slot holds whatever the
/// previous call left there. That is sound for the one thing these bounds are
/// used for — [`backward_propagate`] descends from the same `id`, so it reads
/// only that subtree — and it is the difference between `O(arena_len)` and
/// `O(subtree)` per constraint.
///
/// **Do not read a slot outside the subtree of `id`.** If you need bounds for
/// two unrelated expressions, propagate twice; forward propagation is a pure
/// function of `var_bounds`, so the second call cannot disagree with the
/// first. (`polynomial.rs` did read across subtrees, relying on the old
/// whole-arena behaviour, and now calls twice.)
pub fn forward_propagate_into<'a>(
    arena: &ExprArena,
    id: ExprId,
    var_bounds: &[Interval],
    scratch: &'a mut FwdScratch,
) -> &'a [Interval] {
    scratch.begin(arena.len());

    // Collect the reachable node set. The arena appends children before
    // parents, so ascending index order over that set is a valid topological
    // order — the same order the whole-arena walk used.
    scratch.order.clear();
    scratch.stack.clear();
    scratch.stack.push(id);
    while let Some(nid) = scratch.stack.pop() {
        let i = nid.0;
        if scratch.seen[i] == scratch.epoch {
            continue;
        }
        scratch.seen[i] = scratch.epoch;
        scratch.order.push(i);
        push_children(arena, nid, &mut scratch.stack);
    }
    scratch.order.sort_unstable();

    // `Sum` needs to know how many elements it folds; nothing else does, so the
    // shape pass is paid only by arenas that contain a reduction (#1364).
    let reduce_counts = if scratch
        .order
        .iter()
        .any(|i| matches!(arena.get(ExprId(*i)), ExprNode::Sum { .. }))
    {
        sum_reduce_counts(arena)
    } else {
        Vec::new()
    };
    let empty: [Option<usize>; 0] = [];
    let counts: &[Option<usize>] = if reduce_counts.is_empty() {
        &empty
    } else {
        &reduce_counts
    };

    for k in 0..scratch.order.len() {
        let i = scratch.order[k];
        let v = eval_node_interval(arena, ExprId(i), var_bounds, &scratch.bounds, counts);
        scratch.bounds[i] = v;
    }
    &scratch.bounds
}

/// Forward-propagate interval bounds from leaves to root.
///
/// Returns a vector of intervals, one per arena node. Nodes outside the
/// subtree rooted at `id` are left at [`Interval::entire`] — see
/// [`forward_propagate_into`], which this wraps, for why that is sound and
/// for the one call site that had to change.
pub fn forward_propagate(arena: &ExprArena, id: ExprId, var_bounds: &[Interval]) -> Vec<Interval> {
    let mut scratch = FwdScratch::new();
    forward_propagate_into(arena, id, var_bounds, &mut scratch);
    scratch.bounds
}

/// How many elements each `Sum` node folds, or `None` where it cannot be said.
///
/// `eval_node_interval` carries ONE scalar `Interval` per node, holding the hull
/// over an array node's elements. The enclosure of a reduction is therefore
/// `[n*lo, n*hi]`, and `n` is the only missing piece -- hence this table. Shapes
/// come from `expand::shapes_of`, the single definition of shape inference for
/// this arena; a shape error (a model this arena cannot shape) leaves every entry
/// `None`, which the `Sum` rule reads as "abstain".
///
/// Indexed by node id; non-`Sum` nodes are `None` and never read.
fn sum_reduce_counts(arena: &ExprArena) -> Vec<Option<usize>> {
    let n = arena.len();
    let mut out = vec![None; n];

    // `shapes_of` is all-or-nothing: one node it cannot shape fails the whole
    // arena. That happens on ordinary models -- the PyO3 converter collapses a
    // 1x1 constant array to a scalar `Constant`, so `sum(C * x, axis=1)` reaches
    // here with an operand of rank 1 and the axis out of range -- and letting it
    // disable the rule for every reduction in the arena cost the `discopt.ml`
    // full-space models their certificate (measured: a pinned 1-1-1 sigmoid net
    // went from `optimal` to `feasible`, 55 nodes). So the shape table is an
    // optimization, and `single_element_fold` below is the fallback that does
    // not need it.
    let shapes = crate::expand::shapes_of(arena).ok();

    for (i, slot) in out.iter_mut().enumerate() {
        let ExprNode::Sum { operand, axis } = arena.get(ExprId(i)) else {
            continue;
        };
        let from_shapes = shapes.as_ref().and_then(|sh| {
            let shape = &sh[operand.0];
            match axis {
                // Full reduction: every element folds into one scalar.
                None => Some(shape.iter().product::<usize>().max(1)),
                // Axis reduction: each output element folds that axis's length.
                // An out-of-range axis means this arena lost a length-1
                // dimension; `single_element_fold` answers that case exactly, and
                // guessing a count here would not be sound (over-counting
                // RAISES the lower bound of `[n*lo, n*hi]` when `lo > 0`).
                Some(ax) => shape.get(*ax).copied(),
            }
        });
        *slot = from_shapes.or_else(|| single_element_fold(arena, *operand));
    }
    out
}

/// `Some(1)` when `id` provably stands for exactly one scalar element.
///
/// A structural walk that needs no shape table: a constant, a size-1 variable or
/// parameter, a full reduction, and any element-wise combination of those are
/// single elements, because broadcasting single elements yields a single element.
/// Anything else answers `None` -- the caller then abstains, which is sound.
/// A wrong `Some(1)` would not be, so every arm here is a case where the node
/// cannot stand for more than one value.
fn single_element_fold(arena: &ExprArena, id: ExprId) -> Option<usize> {
    let single = match arena.get(id) {
        ExprNode::Constant(_) => true,
        ExprNode::ConstantArray(data, _) => data.len() == 1,
        ExprNode::Variable { size, .. } => *size == 1,
        ExprNode::Parameter { shape, .. } => shape.iter().product::<usize>() == 1,
        // A full reduction is one scalar whatever it reduced.
        ExprNode::Sum { axis: None, .. } => true,
        ExprNode::SumOver { terms } => terms
            .iter()
            .all(|t| single_element_fold(arena, *t).is_some()),
        ExprNode::BinaryOp { left, right, .. } => {
            single_element_fold(arena, *left).is_some()
                && single_element_fold(arena, *right).is_some()
        }
        ExprNode::UnaryOp { operand, .. } => single_element_fold(arena, *operand).is_some(),
        ExprNode::FunctionCall { args, .. } => args
            .iter()
            .all(|a| single_element_fold(arena, *a).is_some()),
        _ => false,
    };
    single.then_some(1)
}

/// Sound preimage for one element of a reduction whose sum lies in `sum_bound`.
///
/// Every element of the operand lies in its hull `[lo, hi]`, so
/// `e_i = S - sum_{j != i} e_j` lies in `[L - (n-1)*hi, U - (n-1)*lo]`. `None`
/// means "cannot say" -- an unknown fold count, or an arithmetic that went
/// non-finite -- and the caller then declines to tighten, which is always sound.
fn sum_backward_preimage(
    sum_bound: Interval,
    hull: Interval,
    n: Option<usize>,
) -> Option<Interval> {
    let k = n?;
    if k <= 1 {
        // A one-element fold IS the element: the sum's interval transfers exactly.
        return Some(sum_bound);
    }
    let km1 = (k - 1) as f64;
    let lo = sum_bound.lo - km1 * hull.hi;
    let hi = sum_bound.hi - km1 * hull.lo;
    if lo.is_nan() || hi.is_nan() || lo > hi {
        return None;
    }
    Some(Interval::new(lo, hi))
}

/// Sound enclosure of a reduction that folds `n` elements, each enclosed by `a`.
///
/// Every element lies in `[a.lo, a.hi]`, so their sum lies in `[n*a.lo, n*a.hi]`.
/// Returning `a` itself -- which this code did until #1364 -- is NOT conservative:
/// it is narrow, an enclosure that does not contain the value. On
/// `out == sum(C * z, axis=1)` with `C = [[-1, -1]]` and `z in [0, 4]^2` it gave
/// `out >= -4` where the true floor is `-8`, an invalid FBBT tightening that cut
/// the optimum out of the box and let the tree certify a false optimum. The
/// Python-side evaluator had the same defect and fixed it in #1158; this is the
/// Rust half.
fn interval_sum_of(a: Interval, n: Option<usize>) -> Interval {
    match n {
        Some(0) => Interval::point(0.0),
        Some(1) => a,
        // `n` unknown: the hull says nothing about how many terms are folded, so
        // abstain rather than guess. Loose is sound; narrow is not.
        None => Interval::entire(),
        Some(k) => {
            let k = k as f64;
            let lo = a.lo * k;
            let hi = a.hi * k;
            if lo.is_nan() || hi.is_nan() {
                Interval::entire()
            } else {
                Interval::new(lo, hi)
            }
        }
    }
}

/// Compute the interval for a single node given its children's intervals.
fn eval_node_interval(
    arena: &ExprArena,
    id: ExprId,
    var_bounds: &[Interval],
    node_bounds: &[Interval],
    reduce_counts: &[Option<usize>],
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
                MathFunc::Entropy => entropy_interval(&a0),
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
            // A reduction folds `n` elements, each enclosed by the operand's hull,
            // so the sum lies in `[n*lo, n*hi]`. Passing the hull straight through
            // (what this did until #1364) is narrow, not conservative.
            interval_sum_of(
                node_bounds[operand.0],
                reduce_counts.get(id.0).copied().flatten(),
            )
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
    // The `Sum` inversion needs each reduction's fold count (#1364). Compute the
    // table ONCE per walk here rather than inside the recursion, and only when the
    // arena actually contains a reduction -- most constraint bodies do not.
    let counts = if (0..arena.len()).any(|i| matches!(arena.get(ExprId(i)), ExprNode::Sum { .. })) {
        sum_reduce_counts(arena)
    } else {
        Vec::new()
    };
    backward_propagate_with(arena, id, output_bound, node_bounds, var_bounds, &counts);
}

fn backward_propagate_with(
    arena: &ExprArena,
    id: ExprId,
    output_bound: Interval,
    node_bounds: &[Interval],
    var_bounds: &mut [Interval],
    reduce_counts: &[Option<usize>],
) {
    // Intersect the output bound with the forward-propagated bound.
    //
    // Formal inversion is the right test here (#907): bailing out declines to
    // *tighten*, which is the conservative direction — it can only leave the box
    // looser, never cut a feasible point. It is also the guard that stops a
    // noise-inverted interval being written into `var_bounds` and surfacing as a
    // spurious emptiness verdict upstream. No infeasibility is concluded.
    let tightened = output_bound.intersect(&node_bounds[id.0]);
    if tightened.is_formally_inverted() {
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
                    backward_propagate_with(
                        arena,
                        *left,
                        new_l,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                    backward_propagate_with(
                        arena,
                        *right,
                        new_r,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                BinOp::Sub => {
                    // a - b in [lo, hi]
                    // a in [lo + b_lo, hi + b_hi]
                    // b in [a_lo - hi, a_hi - lo]
                    let new_l = Interval::new(tightened.lo + r.lo, tightened.hi + r.hi);
                    let new_r = Interval::new(l.lo - tightened.hi, l.hi - tightened.lo);
                    backward_propagate_with(
                        arena,
                        *left,
                        new_l,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                    backward_propagate_with(
                        arena,
                        *right,
                        new_r,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                BinOp::Mul => {
                    // a * b in [lo, hi]
                    // a in [lo, hi] / b (if b doesn't contain 0)
                    if r.lo > 0.0 || r.hi < 0.0 {
                        let new_l = interval_div(&tightened, &r);
                        backward_propagate_with(
                            arena,
                            *left,
                            new_l,
                            node_bounds,
                            var_bounds,
                            reduce_counts,
                        );
                    }
                    if l.lo > 0.0 || l.hi < 0.0 {
                        let new_r = interval_div(&tightened, &l);
                        backward_propagate_with(
                            arena,
                            *right,
                            new_r,
                            node_bounds,
                            var_bounds,
                            reduce_counts,
                        );
                    }
                }
                BinOp::Div => {
                    // a / b in [lo, hi]
                    // a in [lo, hi] * b
                    let new_l = interval_mul(&tightened, &r);
                    backward_propagate_with(
                        arena,
                        *left,
                        new_l,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                    // b in a / [lo, hi] (if [lo,hi] doesn't contain 0)
                    if tightened.lo > 0.0 || tightened.hi < 0.0 {
                        let new_r = interval_div(&l, &tightened);
                        backward_propagate_with(
                            arena,
                            *right,
                            new_r,
                            node_bounds,
                            var_bounds,
                            reduce_counts,
                        );
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
                                backward_propagate_with(
                                    arena,
                                    *left,
                                    new_base,
                                    node_bounds,
                                    var_bounds,
                                    reduce_counts,
                                );
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
                                backward_propagate_with(
                                    arena,
                                    *left,
                                    new_base,
                                    node_bounds,
                                    var_bounds,
                                    reduce_counts,
                                );
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
                    backward_propagate_with(
                        arena,
                        *operand,
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                UnOp::Abs => {
                    // |a| in [lo, hi] => a in [-hi, -lo] union [lo, hi]
                    // Conservative: a in [-hi, hi]
                    let new = Interval::new(-tightened.hi, tightened.hi);
                    backward_propagate_with(
                        arena,
                        *operand,
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
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
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                MathFunc::Log => {
                    // log(a) in [lo, hi] => a in [exp(lo), exp(hi)]
                    let new = Interval::new(tightened.lo.exp(), tightened.hi.exp());
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                MathFunc::Sqrt => {
                    // sqrt(a) in [lo, hi] => a in [lo^2, hi^2] (lo >= 0)
                    let lo = tightened.lo.max(0.0);
                    let new = Interval::new(lo * lo, tightened.hi * tightened.hi);
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                MathFunc::Log2 => {
                    // log2(a) in [lo, hi] => a in [2^lo, 2^hi]
                    let new = Interval::new(tightened.lo.exp2(), tightened.hi.exp2());
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                MathFunc::Log10 => {
                    // log10(a) in [lo, hi] => a in [10^lo, 10^hi]
                    let new = Interval::new(10f64.powf(tightened.lo), 10f64.powf(tightened.hi));
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                MathFunc::Log1p => {
                    // log1p(a)=ln(1+a) in [lo, hi] => a in [e^lo - 1, e^hi - 1]
                    let new = Interval::new(tightened.lo.exp_m1(), tightened.hi.exp_m1());
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                MathFunc::Sinh => {
                    // sinh increasing, inverse asinh.
                    let new = Interval::new(tightened.lo.asinh(), tightened.hi.asinh());
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
                }
                MathFunc::Asinh => {
                    // asinh increasing, inverse sinh.
                    let new = Interval::new(tightened.lo.sinh(), tightened.hi.sinh());
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
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
                    backward_propagate_with(
                        arena,
                        args[0],
                        new,
                        node_bounds,
                        var_bounds,
                        reduce_counts,
                    );
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
                MathFunc::Entropy => {
                    // `x*ln(x)` is not monotone: it falls on [0, 1/e] and rises
                    // after. Invert only when the forward input sits inside one
                    // branch; a box straddling 1/e has a preimage that is a
                    // union of two intervals, and the enclosing hull of that
                    // union is the input box itself -- no tightening, so skip
                    // rather than pretend.
                    let inp = node_bounds[args[0].0];
                    if let Some(pre) = entropy_preimage(&inp, &tightened) {
                        backward_propagate_with(
                            arena,
                            args[0],
                            pre,
                            node_bounds,
                            var_bounds,
                            reduce_counts,
                        );
                    }
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
                backward_propagate_with(arena, *t, new, node_bounds, var_bounds, reduce_counts);
            }
        }
        ExprNode::Index { base, .. } => {
            backward_propagate_with(
                arena,
                *base,
                tightened,
                node_bounds,
                var_bounds,
                reduce_counts,
            );
        }
        ExprNode::Sum { operand, .. } => {
            // Handing `tightened` straight to the operand asserts that EVERY element
            // lies in the SUM's interval -- `sum(x) <= 10` implies `x_i <= 10` -- which
            // is false unless every other term is known non-negative. That was the
            // backward half of #1364.
            //
            // The sound inversion uses the operand's own hull. With every element in
            // `[lo, hi]` and the sum in `[L, U]`,
            //     e_i = S - sum_{j != i} e_j  in  [L - (n-1)*hi, U - (n-1)*lo],
            // which collapses to `[L, U]` at `n == 1` (the old rule, where it was
            // right) and stays valid for any wider fold. An unknown `n`, or an
            // arithmetic that goes non-finite, declines to tighten -- the
            // conservative direction.
            let hull = node_bounds[operand.0];
            let preimage =
                sum_backward_preimage(tightened, hull, reduce_counts.get(id.0).copied().flatten());
            if let Some(pre) = preimage {
                backward_propagate_with(
                    arena,
                    *operand,
                    pre,
                    node_bounds,
                    var_bounds,
                    reduce_counts,
                );
            }
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
    // Formal inversion (#907): an already-degenerate interval is returned
    // unchanged rather than fed to `ceil`/`floor`. Pass-through concludes
    // nothing; the caller still applies its own tolerance-aware verdict.
    if iv.is_formally_inverted() {
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
/// # `bound` is in the MODEL's objective space, not the solver's
///
/// The row is built against `model.objective` under `model.objective_sense`, so
/// `bound` must be the value of the model's *own* objective at the incumbent --
/// `f(x_inc)`, positive-as-written for a maximize. The Python solver, the B&B
/// tree and the LP relaxation rows all carry the objective in the internal
/// *minimization* space (`-f` for a maximize), so a caller holding
/// `tree.incumbent()[1]` must convert before calling in
/// (`discopt.modeling.core.repr_space_cutoff`).
///
/// Passing the internal value for a maximize model builds `f >= -f(x_inc)`,
/// which is *stricter* than valid whenever `f(x_inc) < 0`: it empties boxes that
/// contain the optimum, and because callers consume the empty verdict as a
/// rigorous fathom the tree then certifies a false `optimal` (issue #1373).
///
/// Returns tightened variable bounds (indexed by variable index, not offset).
pub fn fbbt_with_cutoff(
    model: &ModelRepr,
    max_iter: usize,
    tol: f64,
    incumbent_bound: Option<f64>,
) -> Vec<Interval> {
    fbbt_with_cutoff_until(model, max_iter, tol, incumbent_bound, None)
}

/// Number of constraints between deadline polls inside a single FBBT sweep.
/// `Instant::now()` is ~20 ns, so polling every constraint would cost a few
/// milliseconds per sweep on a 100k-row model -- negligible against the propagation
/// itself, but the stride keeps it free on models with many trivial rows.
const FBBT_DEADLINE_POLL_STRIDE: usize = 64;

/// Like [`fbbt_with_cutoff`] but stops once `deadline` passes, returning the bounds
/// tightened so far.
///
/// FBBT is an anytime algorithm: `backward_propagate` only ever *tightens*
/// `var_bounds`, and each constraint's inference is independently valid, so stopping
/// after any prefix of sweeps -- or after any prefix of constraints *within* a sweep
/// -- yields a valid, merely looser box. It is never a fixed point, which FBBT's
/// callers do not require.
///
/// Without this, a single call runs `max_iter` full forward/backward sweeps over
/// every constraint regardless of any budget, and the presolve orchestrator only
/// checks its own budget *between passes*. Two passes overran because of it on
/// `watercontamination0202` (106,711 vars / 107,209 constraints, issue #863), each
/// >90 s against a 7.5 s budget:
///
///   * `fbbt` directly;
///   * `probing`, which polls its deadline once per binary variable but calls `fbbt`
///     twice per binary. That instance has only **7** binaries, so the per-binary
///     poll granularity was ~13 s of unpollable work each -- the poll was there and
///     still could not help.
pub fn fbbt_with_cutoff_until(
    model: &ModelRepr,
    max_iter: usize,
    tol: f64,
    incumbent_bound: Option<f64>,
    deadline: Option<Instant>,
) -> Vec<Interval> {
    let n_vars = model.variables.len();
    let mut var_bounds: Vec<Interval> = model.variables.iter().map(seed_block_interval).collect();

    // Determine the objective cutoff constraint (if any).
    // One buffer for the whole call, not one per constraint per sweep.
    let mut scratch = FwdScratch::new();

    let obj_cutoff: Option<(ExprId, Interval)> = incumbent_bound.map(|bound| {
        let output_bound = match model.objective_sense {
            ObjectiveSense::Minimize => Interval::new(f64::NEG_INFINITY, bound),
            ObjectiveSense::Maximize => Interval::new(bound, f64::INFINITY),
        };
        (model.objective, output_bound)
    });

    for _ in 0..max_iter {
        if let Some(dl) = deadline {
            if Instant::now() >= dl {
                return var_bounds;
            }
        }
        let old_bounds = var_bounds.clone();

        for (ci, constr) in model.constraints.iter().enumerate() {
            // A single sweep over a 100k-row model is itself far longer than a tight
            // budget, so poll within the sweep as well (#863).
            if let Some(dl) = deadline {
                if ci % FBBT_DEADLINE_POLL_STRIDE == 0 && Instant::now() >= dl {
                    return var_bounds;
                }
            }
            let node_bounds =
                forward_propagate_into(&model.arena, constr.body, &var_bounds, &mut scratch);

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
                node_bounds,
                &mut var_bounds,
            );
        }

        // Propagate the objective cutoff constraint.
        if let Some((obj_expr, ref cutoff_bound)) = obj_cutoff {
            let node_bounds =
                forward_propagate_into(&model.arena, obj_expr, &var_bounds, &mut scratch);
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
                node_bounds,
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
    fbbt_until(model, max_iter, tol, None)
}

/// Like [`fbbt`] but stops once `deadline` passes, returning the bounds tightened so
/// far. See [`fbbt_with_cutoff_until`] for why this is sound (FBBT is anytime: it
/// only ever tightens, and each constraint's inference is independently valid) and
/// for the measurement that motivated it (#863).
pub fn fbbt_until(
    model: &ModelRepr,
    max_iter: usize,
    tol: f64,
    deadline: Option<Instant>,
) -> Vec<Interval> {
    let n_vars = model.variables.len();
    // C-31: seed each block from the element-wise union of its bounds (a valid
    // outer bound for every element), NOT element 0 — see `seed_block_interval`.
    let mut var_bounds: Vec<Interval> = model.variables.iter().map(seed_block_interval).collect();
    // One buffer for the whole call, not one per constraint per sweep.
    let mut scratch = FwdScratch::new();

    for _ in 0..max_iter {
        if let Some(dl) = deadline {
            if Instant::now() >= dl {
                return var_bounds;
            }
        }
        let old_bounds = var_bounds.clone();

        for (ci, constr) in model.constraints.iter().enumerate() {
            if let Some(dl) = deadline {
                if ci % FBBT_DEADLINE_POLL_STRIDE == 0 && Instant::now() >= dl {
                    return var_bounds;
                }
            }
            // Forward propagation.
            let node_bounds =
                forward_propagate_into(&model.arena, constr.body, &var_bounds, &mut scratch);

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
                node_bounds,
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

// ─────────────────────────────────────────────────────────────
// entropy(x) = x*ln(x): interval rules
// ─────────────────────────────────────────────────────────────

/// Minimizer of `x*ln(x)`.
const ENTROPY_ARGMIN: f64 = 0.367_879_441_171_442_33; // 1/e
/// Minimum value of `x*ln(x)`, `-1/e`.
const ENTROPY_MIN: f64 = -0.367_879_441_171_442_33;

/// Outward margin on an inverted `entropy` endpoint, absorbing the rounding of
/// `x*ln(x)` in the bisection below. Applied to the *preimage*, so it can only
/// widen the interval -- never cut a feasible point out of the box.
const ENTROPY_INV_MARGIN: f64 = 1e-9;

/// Forward interval enclosure of `entropy([lo, hi])`.
///
/// Mirrors `_relax/convexity/interval.py::entropy` up to the `XLOG_FLOOR` clamp
/// and this module's usual absence of outward rounding, including the
/// continuous extension `f(0) = 0` that makes a site-fraction box starting at 0
/// enclose finitely (#1242). `lo < 0` is outside the domain and abstains.
///
/// "Exactly" would be too strong, in two ways, and a differential test written
/// against that word would report both as bugs:
///
/// 1. This rule computes through [`expr::xlogx`], which floors its argument at
///    `XLOG_FLOOR = 1e-300`; the Python rule uses the exact continuous
///    extension with no floor. They agree at 0 and everywhere at or above
///    1e-300, and differ on `x` in `(0, 1e-300)` -- at `x = 1e-310` this
///    returns `1e-310*ln(1e-300)`, about 2.4e-309 ABOVE the true image.
/// 2. The Python rule rounds its endpoints outward; this one does not. That is
///    the convention throughout this module (`interval_exp`, `interval_log`,
///    `interval_sqrt`, ... are all bare `Interval::new`), not something new
///    here, but it does mean the Python enclosure is ~1 ULP wider on each side.
fn entropy_interval(a: &Interval) -> Interval {
    if a.lo < 0.0 {
        return Interval::new(f64::NEG_INFINITY, f64::INFINITY);
    }
    let f_lo = xlogx(a.lo);
    let f_hi = xlogx(a.hi);
    let lo = if a.lo <= ENTROPY_ARGMIN && a.hi >= ENTROPY_ARGMIN {
        ENTROPY_MIN
    } else {
        f_lo.min(f_hi)
    };
    Interval::new(lo, f_lo.max(f_hi))
}

/// Preimage of `out` under `entropy`, restricted to the forward box `inp`.
///
/// Returns `None` when no sound tightening is available: `inp` outside the
/// domain, or straddling the minimizer `1/e` (where the preimage is a union of
/// two intervals whose hull is `inp` itself).
fn entropy_preimage(inp: &Interval, out: &Interval) -> Option<Interval> {
    if inp.lo < 0.0 || !inp.lo.is_finite() || !inp.hi.is_finite() || inp.lo > inp.hi {
        return None;
    }
    if inp.hi <= ENTROPY_ARGMIN {
        // Decreasing branch: f(lo) >= f(hi), so the preimage of [out.lo, out.hi]
        // is [f^-1(out.hi), f^-1(out.lo)] with the decreasing inverse.
        let lo = entropy_inv(out.hi, inp, false)?;
        let hi = entropy_inv(out.lo, inp, false)?;
        Some(Interval::new(
            lo - ENTROPY_INV_MARGIN,
            hi + ENTROPY_INV_MARGIN,
        ))
    } else if inp.lo >= ENTROPY_ARGMIN {
        // Increasing branch.
        let lo = entropy_inv(out.lo, inp, true)?;
        let hi = entropy_inv(out.hi, inp, true)?;
        Some(Interval::new(
            lo - ENTROPY_INV_MARGIN,
            hi + ENTROPY_INV_MARGIN,
        ))
    } else {
        None
    }
}

/// Solve `xlogx(t) = y` for `t` in the monotone box `inp`, by bisection.
///
/// `increasing` says which branch `inp` lies on. A target outside
/// `entropy(inp)` is clamped to the nearer endpoint of `inp`, which is the
/// correct preimage endpoint for a monotone function on a closed box.
///
/// Bisection is used rather than a Lambert-W: the root is bracketed by
/// construction, every iteration keeps the bracket, and 200 halvings drive it
/// to the last representable digit -- a rigorous enclosure, not a fitted
/// approximation. The caller widens by [`ENTROPY_INV_MARGIN`] on top.
fn entropy_inv(y: f64, inp: &Interval, increasing: bool) -> Option<f64> {
    if y.is_nan() {
        return None;
    }
    let (mut a, mut b) = (inp.lo, inp.hi);
    let (fa, fb) = (xlogx(a), xlogx(b));
    // Endpoint values bracket the whole attainable range on a monotone branch.
    let (ylo, yhi) = if increasing { (fa, fb) } else { (fb, fa) };
    if y <= ylo {
        return Some(if increasing { a } else { b });
    }
    if y >= yhi {
        return Some(if increasing { b } else { a });
    }
    for _ in 0..200 {
        let mid = 0.5 * (a + b);
        if mid <= a || mid >= b {
            break; // adjacent floats: bracket is as tight as f64 allows
        }
        let fm = xlogx(mid);
        let go_right = if increasing { fm < y } else { fm > y };
        if go_right {
            a = mid;
        } else {
            b = mid;
        }
    }
    // Either endpoint of the final bracket is within one ulp of the root; the
    // caller's outward margin covers the difference.
    Some(0.5 * (a + b))
}

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
        assert!(a.is_formally_inverted());
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
        assert!(r.is_formally_inverted());
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

    // -- subtree-restricted forward propagation (the O(rows x arena) fix) --

    /// Reference implementation: the whole-arena walk that `forward_propagate`
    /// used to be. Every subtree-restricted result must agree with this on the
    /// subtree it evaluated.
    fn forward_propagate_whole_arena(arena: &ExprArena, var_bounds: &[Interval]) -> Vec<Interval> {
        let n = arena.len();
        let mut bounds = vec![Interval::entire(); n];
        let counts = sum_reduce_counts(arena);
        for i in 0..n {
            bounds[i] = eval_node_interval(arena, ExprId(i), var_bounds, &bounds, &counts);
        }
        bounds
    }

    /// Three unrelated expressions in one arena, so a subtree walk provably
    /// skips work: `x*y`, `exp(z)`, `x + z`.
    fn make_three_tree_arena() -> (ExprArena, Vec<ExprId>) {
        let mut arena = ExprArena::new();
        let mut v = |a: &mut ExprArena, name: &str, index: usize| {
            a.add(ExprNode::Variable {
                name: name.into(),
                index,
                size: 1,
                shape: vec![],
            })
        };
        let x = v(&mut arena, "x", 0);
        let y = v(&mut arena, "y", 1);
        let z = v(&mut arena, "z", 2);
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: y,
        });
        let expz = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Exp,
            args: vec![z],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: z,
        });
        (arena, vec![prod, expz, sum])
    }

    #[test]
    fn subtree_propagation_agrees_with_the_whole_arena_walk() {
        let (arena, roots) = make_three_tree_arena();
        let var_bounds = vec![
            Interval::new(1.0, 3.0),
            Interval::new(2.0, 5.0),
            Interval::new(0.0, 1.0),
        ];
        let reference = forward_propagate_whole_arena(&arena, &var_bounds);

        let mut checked = 0usize;
        let mut scratch = FwdScratch::new();
        for &root in &roots {
            let got = forward_propagate_into(&arena, root, &var_bounds, &mut scratch);
            // Every node reachable from `root` must match the reference exactly.
            let mut stack = vec![root];
            let mut seen = std::collections::HashSet::new();
            while let Some(nid) = stack.pop() {
                if !seen.insert(nid.0) {
                    continue;
                }
                assert_eq!(
                    got[nid.0].lo, reference[nid.0].lo,
                    "lo differs at node {} under root {}",
                    nid.0, root.0
                );
                assert_eq!(
                    got[nid.0].hi, reference[nid.0].hi,
                    "hi differs at node {} under root {}",
                    nid.0, root.0
                );
                checked += 1;
                push_children(&arena, nid, &mut stack);
            }
        }
        // Anti-vacuity (CLAUDE.md section 6): a walk that visited nothing would
        // pass every assertion above.
        // 3 nodes under x*y + 2 under exp(z) + 3 under x+z.
        assert!(checked >= 8, "comparisons executed: {checked}");
    }

    #[test]
    fn the_fresh_vec_wrapper_leaves_other_subtrees_entire() {
        let (arena, roots) = make_three_tree_arena();
        let var_bounds = vec![
            Interval::new(1.0, 3.0),
            Interval::new(2.0, 5.0),
            Interval::new(0.0, 1.0),
        ];
        // Propagate only `exp(z)`. `x*y` shares no node with it.
        let bounds = forward_propagate(&arena, roots[1], &var_bounds);
        assert_eq!(bounds[roots[1].0].lo, 1.0);
        let prod = bounds[roots[0].0];
        assert!(
            prod.lo == f64::NEG_INFINITY && prod.hi == f64::INFINITY,
            "an unvisited root must stay entire, got {prod:?}"
        );
    }

    #[test]
    fn propagation_evaluates_only_the_subtree() {
        // The whole point of the change. `exp(z)` is 2 nodes out of 6; the old
        // implementation evaluated all 6 for it, and did so once per
        // constraint per sweep.
        let (arena, roots) = make_three_tree_arena();
        assert_eq!(arena.len(), 6, "arena shape changed; update this test");
        let var_bounds = vec![
            Interval::new(1.0, 3.0),
            Interval::new(2.0, 5.0),
            Interval::new(0.0, 1.0),
        ];
        let mut scratch = FwdScratch::new();
        forward_propagate_into(&arena, roots[1], &var_bounds, &mut scratch);
        assert_eq!(
            scratch.order.len(),
            2,
            "expected exp(z) and z, evaluated {:?}",
            scratch.order
        );
        forward_propagate_into(&arena, roots[0], &var_bounds, &mut scratch);
        assert_eq!(scratch.order.len(), 3, "expected x*y, x, y");
    }

    #[test]
    fn a_reused_scratch_does_not_leak_a_stale_epoch() {
        // The stamp is what keeps the per-call cost proportional to the
        // subtree. If a stale stamp aliased the current epoch, nodes would be
        // silently skipped and read back at their previous values.
        let (arena, roots) = make_three_tree_arena();
        let mut scratch = FwdScratch::new();
        let tight = vec![
            Interval::new(1.0, 1.0),
            Interval::new(2.0, 2.0),
            Interval::new(0.0, 0.0),
        ];
        let loose = vec![
            Interval::new(1.0, 3.0),
            Interval::new(2.0, 5.0),
            Interval::new(0.0, 1.0),
        ];
        let mut checked = 0usize;
        for _ in 0..3 {
            let b = forward_propagate_into(&arena, roots[0], &tight, &mut scratch);
            assert_eq!((b[roots[0].0].lo, b[roots[0].0].hi), (2.0, 2.0));
            checked += 1;
            let b = forward_propagate_into(&arena, roots[0], &loose, &mut scratch);
            assert_eq!((b[roots[0].0].lo, b[roots[0].0].hi), (2.0, 15.0));
            checked += 1;
        }
        assert_eq!(checked, 6, "assertions executed: {checked}");
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
    fn fbbt_until_honors_past_deadline() {
        // FBBT runs `max_iter` full forward/backward sweeps over every constraint and
        // the presolve orchestrator only checks its time budget BETWEEN passes, so a
        // single call overran a 7.5 s budget by >12x on watercontamination0202 -- both
        // directly as the `fbbt` pass and indirectly through `probing`, which calls
        // FBBT twice per binary and could only poll between binaries (#863).
        //
        // With the deadline already past, no propagation may happen: the returned
        // bounds must be the declared box. That is sound -- FBBT is anytime and only
        // ever tightens, so an untightened box is valid, merely looser.
        let model = make_linear_model();
        let tightened = fbbt_until(&model, 10, 1e-8, None);
        assert!(
            (tightened[0].hi - 10.0).abs() < 1e-10,
            "control must tighten"
        );

        let past = std::time::Instant::now();
        let bailed = fbbt_until(&model, 10, 1e-8, Some(past));
        assert!(
            (bailed[0].hi - 100.0).abs() < 1e-10,
            "fbbt must bail before propagating once the deadline has passed; got {:?}",
            bailed[0]
        );
        assert!((bailed[1].hi - 100.0).abs() < 1e-10);
        // Still a valid enclosure of the tightened box.
        assert!(bailed[0].lo <= tightened[0].lo && bailed[0].hi >= tightened[0].hi);
        assert!(bailed[1].lo <= tightened[1].lo && bailed[1].hi >= tightened[1].hi);
    }

    #[test]
    fn fbbt_until_with_a_future_deadline_matches_no_deadline() {
        // The poll must be the ONLY difference: a deadline far in the future has to
        // give bit-identical bounds, or every presolve result silently changed.
        let model = make_linear_model();
        let a = fbbt_until(&model, 10, 1e-8, None);
        let future = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        let b = fbbt_until(&model, 10, 1e-8, Some(future));
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.lo, y.lo, "var {i} lo differs");
            assert_eq!(x.hi, y.hi, "var {i} hi differs");
        }
    }

    #[test]
    fn fbbt_with_cutoff_until_honors_past_deadline() {
        let model = make_linear_model();
        let tightened = fbbt_with_cutoff_until(&model, 10, 1e-8, None, None);
        assert!(
            (tightened[0].hi - 10.0).abs() < 1e-10,
            "control must tighten"
        );

        let past = std::time::Instant::now();
        let bailed = fbbt_with_cutoff_until(&model, 10, 1e-8, None, Some(past));
        assert!((bailed[0].hi - 100.0).abs() < 1e-10);
        assert!((bailed[1].hi - 100.0).abs() < 1e-10);

        // And a future deadline is indistinguishable from none.
        let future = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        let same = fbbt_with_cutoff_until(&model, 10, 1e-8, None, Some(future));
        for (x, y) in tightened.iter().zip(same.iter()) {
            assert_eq!(x.lo, y.lo);
            assert_eq!(x.hi, y.hi);
        }
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
            assert!(
                b.is_formally_inverted(),
                "Expected infeasible (empty bounds)"
            );
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
        assert!(eps.is_formally_inverted());
        assert!(!eps.is_empty_beyond(FEAS_TOL));

        // A genuinely inverted interval is empty beyond tolerance.
        let real = Interval::new(1.0, 0.9);
        assert!(real.is_formally_inverted());
        assert!(real.is_empty_beyond(FEAS_TOL));

        // The canonical empty interval is empty beyond any finite tolerance.
        assert!(Interval::empty().is_empty_beyond(FEAS_TOL));
    }

    // -- sub-tolerance crossing repair (issue #907) --

    #[test]
    fn test_repair_widens_subtol_crossing_without_cutting_either_endpoint() {
        // The exact crossing measured on `heatexch_gen3` var4 on the default
        // path: two derivations of the same fixed value disagreeing by 8.5e-14.
        let mut iv = Interval::new(226.7, 226.699_999_999_999_9);
        assert!(iv.is_formally_inverted());
        assert!(iv.repair_if_subtol_inverted(FEAS_TOL));

        // Repair must WIDEN to contain both endpoints. A midpoint collapse
        // would be tighter than one sound endpoint and could cut the optimum.
        assert!(!iv.is_formally_inverted());
        assert!(iv.contains(226.7), "repair cut the upper derivation");
        assert!(
            iv.contains(226.699_999_999_999_9),
            "repair cut the lower one"
        );
        assert_eq!(iv.lo, 226.699_999_999_999_9);
        assert_eq!(iv.hi, 226.7);
    }

    /// ANTI-PERMISSIVENESS CONTROL. Repair must not launder a real infeasibility
    /// into a feasible box. Without this the fix would be the tolerance-tweak
    /// CLAUDE.md §3 forbids rather than a correctness fix.
    #[test]
    fn test_repair_leaves_genuine_and_sentinel_emptiness_alone() {
        // A crossing beyond FEAS_TOL is a real verdict — untouched.
        let mut real = Interval::new(1.0, 0.9);
        assert!(!real.repair_if_subtol_inverted(FEAS_TOL));
        assert!(real.is_empty_beyond(FEAS_TOL));

        // The exact genuine population observed at the in-tree site: a binary
        // whose domain was wiped out, `[1.0, 0.0]`, crossing exactly 1.0.
        let mut binary = Interval::new(1.0, 0.0);
        assert!(!binary.repair_if_subtol_inverted(FEAS_TOL));
        assert!(binary.is_empty_beyond(FEAS_TOL));

        // The explicit `[inf, -inf]` sentinel survives any finite tolerance.
        let mut sentinel = Interval::empty();
        assert!(!sentinel.repair_if_subtol_inverted(FEAS_TOL));
        assert!(sentinel.is_empty_beyond(FEAS_TOL));

        // A well-formed interval is not disturbed.
        let mut ok = Interval::new(0.0, 1.0);
        assert!(!ok.repair_if_subtol_inverted(FEAS_TOL));
        assert_eq!((ok.lo, ok.hi), (0.0, 1.0));
    }

    #[test]
    fn test_slice_helpers_count_and_verdict() {
        let mut bounds = vec![
            Interval::new(0.0, 1.0),                     // fine
            Interval::new(226.7, 226.699_999_999_999_9), // noise
            Interval::new(5.0, 5.0 - 1e-12),             // noise
        ];
        // Verdict BEFORE repair: no genuine emptiness present.
        assert!(!any_empty_beyond(&bounds, FEAS_TOL));
        assert_eq!(repair_subtol_crossings(&mut bounds, FEAS_TOL), 2);
        assert!(bounds.iter().all(|b| !b.is_formally_inverted()));
        // Idempotent: a second pass repairs nothing.
        assert_eq!(repair_subtol_crossings(&mut bounds, FEAS_TOL), 0);

        // ANTI-PERMISSIVENESS: a genuine crossing still reports empty, and
        // repair declines to touch it.
        bounds.push(Interval::new(1.0, 0.0));
        assert_eq!(repair_subtol_crossings(&mut bounds, FEAS_TOL), 0);
        assert!(any_empty_beyond(&bounds, FEAS_TOL));
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
            bounds.iter().all(|b| b.is_formally_inverted()),
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
        assert!(!bounds.iter().any(|b| b.is_formally_inverted()));
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
        assert!(!bounds.iter().any(|b| b.is_formally_inverted()));
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
            bounds.iter().any(|b| b.is_formally_inverted()),
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
        assert!(!bounds.iter().any(|b| b.is_formally_inverted()));
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
            !bounds.iter().any(|b| b.is_formally_inverted()),
            "C-31: feasible model x=[5, 0..3] must not be declared infeasible, \
             got {:?}",
            bounds
        );
    }
}

#[cfg(test)]
mod entropy_tests {
    use super::*;

    /// Reference `x*ln(x)` sampled densely over a box; used to prove the
    /// analytic enclosure actually contains the function (#1242).
    fn sampled_range(lo: f64, hi: f64) -> (f64, f64) {
        let n = 20_001;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for i in 0..n {
            let t = lo + (hi - lo) * (i as f64) / ((n - 1) as f64);
            let v = xlogx(t);
            mn = mn.min(v);
            mx = mx.max(v);
        }
        (mn, mx)
    }

    #[test]
    fn entropy_value_at_zero_is_the_limit() {
        assert_eq!(xlogx(0.0), 0.0);
        assert_eq!(xlogx(1.0), 0.0);
        assert!((xlogx(std::f64::consts::E.recip()) + std::f64::consts::E.recip()).abs() < 1e-15);
        assert!(xlogx(-0.5).is_nan(), "x < 0 is outside the domain");
        // The clamp keeps a denormal argument finite rather than -inf * 0.
        assert!(xlogx(1e-320).is_finite());
    }

    #[test]
    fn entropy_interval_is_exact_on_closed_domain() {
        // (lo, hi, expected lo, expected hi)
        let cases: &[(f64, f64, f64, f64)] = &[
            // u < 1/e: decreasing branch, so [f(u), f(0)] = [u ln u, 0].
            (0.0, 0.2, 0.2 * 0.2_f64.ln(), 0.0),
            // u == 1/e: the minimizer is the right endpoint.
            (0.0, ENTROPY_ARGMIN, ENTROPY_MIN, 0.0),
            // u > 1/e: interior minimum, max at an endpoint (both are 0 here).
            (0.0, 1.0, ENTROPY_MIN, 0.0),
            // u > 1: max at the right endpoint.
            (0.0, 2.0, ENTROPY_MIN, 2.0 * 2.0_f64.ln()),
            // strictly-interior box, minimizer inside.
            (0.1, 0.9, ENTROPY_MIN, 0.9 * 0.9_f64.ln()),
            // strictly-increasing box: both endpoints past the minimizer.
            (0.5, 0.9, 0.5 * 0.5_f64.ln(), 0.9 * 0.9_f64.ln()),
        ];
        let mut checked = 0usize;
        for &(lo, hi, elo, ehi) in cases {
            let got = entropy_interval(&Interval::new(lo, hi));
            assert!(
                (got.lo - elo).abs() < 1e-12 && (got.hi - ehi).abs() < 1e-12,
                "entropy([{lo}, {hi}]) = [{}, {}], expected [{elo}, {ehi}]",
                got.lo,
                got.hi
            );
            // And it really encloses the function, not just the formula.
            let (smn, smx) = sampled_range(lo, hi);
            assert!(
                got.lo <= smn + 1e-12 && got.hi >= smx - 1e-12,
                "entropy([{lo}, {hi}]) = [{}, {}] does not enclose sampled [{smn}, {smx}]",
                got.lo,
                got.hi
            );
            checked += 1;
        }
        assert_eq!(checked, cases.len(), "probe executed no comparisons");
    }

    #[test]
    fn entropy_interval_abstains_below_zero() {
        let got = entropy_interval(&Interval::new(-0.1, 1.0));
        assert_eq!(got.lo, f64::NEG_INFINITY);
        assert_eq!(got.hi, f64::INFINITY);
    }

    #[test]
    fn entropy_preimage_is_sound_on_each_branch() {
        // Decreasing branch [0, 1/e]: require entropy(x) <= -0.2.
        //
        // f(x) = -0.2 has roots at 0.07865836 and 0.77169097 (re-derived with
        // brentq on each side of the minimizer). An earlier version of this
        // comment quoted ~0.0712 and ~0.3070, which solve nothing nearby --
        // f(0.0712) = -0.1881, f(0.3070) = -0.3625 -- and then argued from the
        // fake second root that "both roots are below 1/e". They are not: the
        // real one is 0.7717. The branch is still unambiguous here, for the
        // other reason -- 0.7717 lies OUTSIDE `inp`, so on [0, 1/e] the
        // constraint f(x) <= -0.2 is exactly x >= 0.07865836, and the expected
        // preimage is [0.07865836, 1/e].
        let inp = Interval::new(0.0, ENTROPY_ARGMIN);
        let out = Interval::new(f64::NEG_INFINITY, -0.2);
        let pre = entropy_preimage(&inp, &out).expect("decreasing branch inverts");
        // Every point of `inp` with f(x) <= -0.2 must survive.
        let mut kept = 0usize;
        for i in 0..=10_000 {
            let x = inp.lo + (inp.hi - inp.lo) * (i as f64) / 10_000.0;
            if xlogx(x) <= -0.2 {
                assert!(
                    x >= pre.lo && x <= pre.hi,
                    "feasible x={x} cut out by preimage [{}, {}]",
                    pre.lo,
                    pre.hi
                );
                kept += 1;
            }
        }
        assert!(kept > 0, "probe found no feasible points to check");
        // ...and it must be a real tightening. Without this, every assertion
        // above is satisfied by `pre == inp` (or anything wider), so the arm
        // would pass against an `entropy_preimage` that hands the input box
        // straight back -- sound, and a complete no-op. Only the increasing
        // arm below had this check.
        assert!(
            pre.lo > inp.lo,
            "decreasing branch did not tighten: {pre:?}"
        );
        assert!(
            (pre.lo - 0.078_658_360_286_855_77).abs() < 1e-6,
            "decreasing branch cut at the wrong root: {pre:?}"
        );

        // Increasing branch [1/e, 3].
        let inp = Interval::new(ENTROPY_ARGMIN, 3.0);
        let out = Interval::new(f64::NEG_INFINITY, 1.0);
        let pre = entropy_preimage(&inp, &out).expect("increasing branch inverts");
        let mut kept2 = 0usize;
        for i in 0..=10_000 {
            let x = inp.lo + (inp.hi - inp.lo) * (i as f64) / 10_000.0;
            if xlogx(x) <= 1.0 {
                assert!(
                    x >= pre.lo && x <= pre.hi,
                    "feasible x={x} cut out by preimage [{}, {}]",
                    pre.lo,
                    pre.hi
                );
                kept2 += 1;
            }
        }
        assert!(kept2 > 0, "probe found no feasible points to check");
        // It must also be a real tightening, not the input box back.
        assert!(pre.hi < 3.0, "increasing branch did not tighten: {pre:?}");
    }

    #[test]
    fn entropy_preimage_skips_a_box_straddling_the_minimizer() {
        // The preimage there is a union of two intervals whose hull is the
        // input box; returning it would be sound but useless, and returning
        // one branch would be UNSOUND. Skip is the only correct answer.
        let inp = Interval::new(0.0, 1.0);
        let out = Interval::new(f64::NEG_INFINITY, -0.2);
        assert!(entropy_preimage(&inp, &out).is_none());
    }

    #[test]
    fn entropy_forward_fbbt_bounds_a_variable_expression() {
        let mut arena = ExprArena::new();
        let x = arena.intern(ExprNode::Variable {
            name: "y".to_string(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let e = arena.intern(ExprNode::FunctionCall {
            func: MathFunc::Entropy,
            args: vec![x],
        });
        let var_bounds = vec![Interval::new(0.0, 1.0)];
        let node_bounds = forward_propagate(&arena, e, &var_bounds);
        let got = node_bounds[e.0];
        assert!(
            (got.lo - ENTROPY_MIN).abs() < 1e-12 && got.hi.abs() < 1e-12,
            "forward FBBT gave [{}, {}], expected [-1/e, 0]",
            got.lo,
            got.hi
        );
    }
}
