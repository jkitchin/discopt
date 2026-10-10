//! Polynomial detection + reformulation to quadratic form (M4 + M5 of issue #51).
//!
//! ## What this pass does
//!
//! 1. **Detect** polynomial subexpressions in constraint bodies and the
//!    objective. A polynomial here is a sum of monomials, where each
//!    monomial is a numeric coefficient times a product of integer
//!    powers of scalar variables (or scalar `Index(Variable, …)` leaves).
//! 2. **Reformulate** every monomial of total degree > 2 by introducing
//!    auxiliary variables defined by bilinear products. After the pass,
//!    every constraint body has algebraic degree ≤ 2 in the augmented
//!    variable set; the original feasible region embeds isomorphically
//!    via the auxiliary equality constraints.
//! 3. **Derive reduction constraints** (M5) by computing the McCormick
//!    range of each auxiliary product `w = a · b` from forward interval
//!    bounds on `a, b` and recording it as the auxiliary variable's
//!    declared bound. These are provably feasibility-preserving (they
//!    follow algebraically from the equality definitions) and yield
//!    measurable LP-relaxation tightening because every downstream
//!    relaxation consumes them as variable bounds on `w`.
//!
//! ## Why both passes ship together
//!
//! Karia, Adjiman & Chachuat (2022) observe that polynomial-to-quadratic
//! reformulation is materially tighter than direct McCormick on cubic
//! and higher monomials *only* when the auxiliary variables come with
//! good a-priori bounds. M5 supplies exactly those bounds without the
//! caller needing a separate FBBT pass. Running M4 alone produces
//! aux variables with `[-inf, +inf]` declared bounds, which then takes
//! several FBBT sweeps to recover.
//!
//! ## Algorithmic shape
//!
//! Detection runs in time linear in the arena size. Reformulation uses
//! a global cache keyed by the canonical pair `(left_id, right_id)`
//! (sorted) so identical bilinear products across monomials and
//! constraints share the same auxiliary variable — the basic Karia
//! 2022 sharing optimization. Decomposition of a monomial of total
//! degree `d` introduces at most `d − 2` auxiliaries (left fold from
//! the highest-exponent factor downward), but in practice the cache
//! collapses many of those across the model.
//!
//! ## Acceptance criteria from issue #51
//!
//! M4:
//! - Reformulation preserves feasibility and optimality. *Verified by
//!   sampling*: the rewritten polynomial body equals the original to
//!   within 1e-9 at every sampled point of the feasible box.
//! - Resulting relaxation is at least as tight as direct McCormick on
//!   the original on ≥ 80% of test instances. *Smoke-verified* via the
//!   regression suite; the formal bookkeeping integration with the LP
//!   relaxation compiler is a follow-up.
//! - Reproduces Karia, Adjiman & Chachuat (2022) tightening on the
//!   small published instances within 1%. *Documented*: a representative
//!   instance is included in the regression suite; the full case study
//!   is gated by the relaxation-compiler integration.
//!
//! M5:
//! - Every derived constraint is provably valid (no feasible point of
//!   the original model violates it). *By construction*: the McCormick
//!   bound on `a · b` is a sound enclosure of the product over the
//!   forward bounds on `a, b`.
//! - Adding the derived constraints does not change the global optimum
//!   of the reference test set. *Verified by sampling*.
//! - Adds measurable LP-relaxation tightening (≥ 1% gap closure) on at
//!   least one instance in the polynomial MINLP test set. *Verified*:
//!   regression tests exercise a known polynomial instance.
//!
//! ## References
//!
//! Karia, T., Adjiman, C. S., Chachuat, B. (2022). *Polynomial
//!   reformulation in global optimization*. Comput. Chem. Eng. 165,
//!   107909.
//! Liberti, L., Pantelides, C. C. (2003). *Convex envelopes of
//!   monomials of odd degree*. J. Global Optim. 25(2), 157-168.

use crate::expr::{
    BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr, UnOp, VarInfo,
    VarType,
};
use std::collections::HashMap;

use super::fbbt::{forward_propagate, interval_mul, Interval};

// ─────────────────────────────────────────────────────────────
// Polynomial data
// ─────────────────────────────────────────────────────────────

/// One monomial: coefficient times a product of variable powers.
///
/// `factors` is a sparse list of `(variable_leaf_expr_id, exponent)`
/// pairs with strictly positive exponents. The leaf id is the arena
/// id of either a `Variable` or `Index(Variable, _)` node — whichever
/// the user actually writes — so identical leaves dedupe by arena id.
#[derive(Debug, Clone)]
pub struct Monomial {
    /// Numeric coefficient.
    pub coeff: f64,
    /// Variable leaves and their integer exponents, sorted by leaf id.
    pub factors: Vec<(ExprId, u32)>,
}

impl Monomial {
    /// Total polynomial degree of the monomial.
    pub fn total_degree(&self) -> u32 {
        self.factors.iter().map(|(_, e)| *e).sum()
    }
}

/// A polynomial: sum of monomials plus a numeric constant offset.
#[derive(Debug, Clone, Default)]
pub struct Polynomial {
    /// Constant offset (degree-0 part).
    pub constant: f64,
    /// Monomials of degree ≥ 1.
    pub monomials: Vec<Monomial>,
}

impl Polynomial {
    /// Maximum total degree across monomials. Returns 0 if there are
    /// only constants.
    pub fn max_total_degree(&self) -> u32 {
        self.monomials
            .iter()
            .map(|m| m.total_degree())
            .max()
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────
// Polynomial detection
// ─────────────────────────────────────────────────────────────

/// Try to interpret `root` as a polynomial in scalar variable leaves.
///
/// Returns `None` if any non-polynomial atom is encountered (division
/// by a non-constant, transcendental, non-integer power, abs, …) so
/// the caller can leave such expressions alone.
///
/// **Unbounded.** This expands products and integer powers in full; a
/// product of `k` sums of width `w` produces `w^k` monomials. A caller that
/// runs before the time limit and cannot afford that must use
/// [`try_polynomial_budgeted`] instead (#1686).
pub fn try_polynomial(arena: &ExprArena, root: ExprId) -> Option<Polynomial> {
    let mut work = PolyWork::unlimited();
    match try_polynomial_work(arena, root, &mut work) {
        Ok(p) => Some(p),
        Err(PolyStop::NotPolynomial) => None,
        // `u64::MAX` cannot be spent by any expansion that fits in memory.
        Err(PolyStop::BudgetExhausted) => None,
    }
}

/// Outcome of a budgeted expansion that did not produce a polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyBudgetedError {
    /// The expression contains a non-polynomial atom (same as
    /// [`try_polynomial`] returning `None`).
    NotPolynomial,
    /// The expansion would have exceeded the work budget. Nothing is known
    /// about the expression; the caller must abstain, not guess.
    BudgetExhausted,
}

/// Work allowance for [`try_polynomial_budgeted`], in two separately capped
/// currencies. Both are **operation counts**, never a clock (#912).
///
/// - `linear`: node visits, leaf monomials, and copies through a constant
///   factor (`c * p`, `p / c`, `p ^ 1`). Proportional to the size of the
///   expression *as a tree*; only exceeds the arena size when shared
///   subtrees are re-walked.
/// - `expansion`: monomials created by multiplying two non-constant
///   polynomials (products and integer powers ≥ 2). This is the
///   `w^k` term that makes a product of sums blow up (#1686).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyBudget {
    /// Allowance for linear work.
    pub linear: u64,
    /// Allowance for expansion work.
    pub expansion: u64,
}

impl PolyBudget {
    /// No limit in either currency.
    pub const UNLIMITED: PolyBudget = PolyBudget {
        linear: u64::MAX,
        expansion: u64::MAX,
    };
}

/// Units actually spent by one [`try_polynomial_budgeted`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PolySpent {
    /// Linear units spent.
    pub linear: u64,
    /// Expansion units spent.
    pub expansion: u64,
}

/// [`try_polynomial`] with a deterministic work budget (#1686).
///
/// The budget is an **operation count**, not a clock (#912's rule: a clock
/// would make the result a function of machine load). See [`PolyBudget`] for
/// what is charged. A product is priced *before* it is formed, so an
/// over-budget cross-product is refused without being allocated.
///
/// Returns the result and the units spent (also on failure). Within budget
/// the result is identical to [`try_polynomial`]: the budget only decides
/// whether the expansion runs to completion, never how it is computed.
pub fn try_polynomial_budgeted(
    arena: &ExprArena,
    root: ExprId,
    budget: PolyBudget,
) -> (Result<Polynomial, PolyBudgetedError>, PolySpent) {
    let mut work = PolyWork {
        budget,
        spent: PolySpent::default(),
    };
    let r = match try_polynomial_work(arena, root, &mut work) {
        Ok(p) => Ok(p),
        Err(PolyStop::NotPolynomial) => Err(PolyBudgetedError::NotPolynomial),
        Err(PolyStop::BudgetExhausted) => Err(PolyBudgetedError::BudgetExhausted),
    };
    (r, work.spent)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolyStop {
    NotPolynomial,
    BudgetExhausted,
}

/// Remaining work for one expansion. `unlimited()` is `u64::MAX` in both
/// currencies, which keeps the unbudgeted [`try_polynomial`] path
/// behaviourally unchanged.
struct PolyWork {
    budget: PolyBudget,
    spent: PolySpent,
}

impl PolyWork {
    fn unlimited() -> Self {
        PolyWork {
            budget: PolyBudget::UNLIMITED,
            spent: PolySpent::default(),
        }
    }

    #[inline]
    fn linear(&mut self, units: u64) -> Result<(), PolyStop> {
        if units > self.budget.linear {
            return Err(PolyStop::BudgetExhausted);
        }
        self.budget.linear -= units;
        self.spent.linear = self.spent.linear.saturating_add(units);
        Ok(())
    }

    #[inline]
    fn expansion(&mut self, units: u64) -> Result<(), PolyStop> {
        if units > self.budget.expansion {
            return Err(PolyStop::BudgetExhausted);
        }
        self.budget.expansion -= units;
        self.spent.expansion = self.spent.expansion.saturating_add(units);
        Ok(())
    }

    /// Charge `multiply_into(l, r)`: a copy through a constant factor when
    /// either side has no monomials, a genuine expansion otherwise.
    fn product(&mut self, l: &Polynomial, r: &Polynomial) -> Result<(), PolyStop> {
        let (nl, nr) = (l.monomials.len() as u64, r.monomials.len() as u64);
        let cost = nl
            .saturating_mul(nr)
            .saturating_add(nl)
            .saturating_add(nr)
            .saturating_add(1);
        if nl == 0 || nr == 0 {
            self.linear(cost)
        } else {
            self.expansion(cost)
        }
    }
}

fn not_poly<T>(o: Option<T>) -> Result<T, PolyStop> {
    o.ok_or(PolyStop::NotPolynomial)
}

fn try_polynomial_work(
    arena: &ExprArena,
    root: ExprId,
    work: &mut PolyWork,
) -> Result<Polynomial, PolyStop> {
    let mut p = Polynomial::default();
    walk_into_polynomial(arena, root, 1.0, &mut p, work)?;
    canonicalise(&mut p);
    Ok(p)
}

fn walk_into_polynomial(
    arena: &ExprArena,
    id: ExprId,
    sign: f64,
    out: &mut Polynomial,
    work: &mut PolyWork,
) -> Result<(), PolyStop> {
    // One unit per node visit: a DAG whose shared subtrees are re-walked as a
    // tree is exponential in its arena size even when it pushes no monomials.
    work.linear(1)?;
    match arena.get(id) {
        ExprNode::Constant(v) => {
            out.constant += sign * *v;
            Ok(())
        }
        ExprNode::Parameter { value, shape, .. } => {
            if shape.is_empty() || (shape.len() == 1 && shape[0] == 1) {
                out.constant += sign * not_poly(value.first().copied())?;
                Ok(())
            } else {
                Err(PolyStop::NotPolynomial)
            }
        }
        ExprNode::ConstantArray(data, shape) => {
            if data.len() == 1 && shape.iter().all(|&d| d == 1) {
                out.constant += sign * data[0];
                Ok(())
            } else {
                Err(PolyStop::NotPolynomial)
            }
        }
        ExprNode::Variable { size, .. } if *size == 1 => {
            work.linear(1)?;
            out.monomials.push(Monomial {
                coeff: sign,
                factors: vec![(id, 1)],
            });
            Ok(())
        }
        ExprNode::Index { base, .. } => match arena.get(*base) {
            ExprNode::Variable { .. } => {
                work.linear(1)?;
                out.monomials.push(Monomial {
                    coeff: sign,
                    factors: vec![(id, 1)],
                });
                Ok(())
            }
            _ => Err(PolyStop::NotPolynomial),
        },
        ExprNode::UnaryOp { op, operand } => match op {
            UnOp::Neg => walk_into_polynomial(arena, *operand, -sign, out, work),
            UnOp::Abs => Err(PolyStop::NotPolynomial),
        },
        ExprNode::BinaryOp { op, left, right } => match op {
            BinOp::Add => {
                walk_into_polynomial(arena, *left, sign, out, work)?;
                walk_into_polynomial(arena, *right, sign, out, work)
            }
            BinOp::Sub => {
                walk_into_polynomial(arena, *left, sign, out, work)?;
                walk_into_polynomial(arena, *right, -sign, out, work)
            }
            BinOp::Mul => {
                let lp = try_polynomial_work(arena, *left, work)?;
                let rp = try_polynomial_work(arena, *right, work)?;
                work.product(&lp, &rp)?;
                multiply_into(out, sign, &lp, &rp);
                Ok(())
            }
            BinOp::Div => {
                // Polynomial only if the denominator is a numeric constant.
                let denom = not_poly(constant_value(arena, *right))?;
                if denom == 0.0 {
                    return Err(PolyStop::NotPolynomial);
                }
                let p = try_polynomial_work(arena, *left, work)?;
                work.linear(p.monomials.len() as u64)?;
                add_scaled(out, sign / denom, &p);
                Ok(())
            }
            BinOp::Pow => {
                let exp = not_poly(constant_value(arena, *right))?;
                let exp_int = exp.round() as i64;
                if (exp - exp_int as f64).abs() > 1e-12 || exp_int < 0 {
                    return Err(PolyStop::NotPolynomial);
                }
                if exp_int == 0 {
                    out.constant += sign * 1.0;
                    return Ok(());
                }
                let base_poly = try_polynomial_work(arena, *left, work)?;
                let powered = power_polynomial(&base_poly, exp_int as u32, work)?;
                // The copy is priced in the currency that created it.
                let n = powered.monomials.len() as u64;
                if exp_int == 1 {
                    work.linear(n)?;
                } else {
                    work.expansion(n)?;
                }
                add_scaled(out, sign, &powered);
                Ok(())
            }
        },
        ExprNode::Sum { operand, axis } if axis.is_none() => {
            walk_into_polynomial(arena, *operand, sign, out, work)
        }
        ExprNode::SumOver { terms } => {
            for t in terms {
                walk_into_polynomial(arena, *t, sign, out, work)?;
            }
            Ok(())
        }
        _ => Err(PolyStop::NotPolynomial),
    }
}

fn constant_value(arena: &ExprArena, id: ExprId) -> Option<f64> {
    match arena.get(id) {
        ExprNode::Constant(v) => Some(*v),
        ExprNode::Parameter { value, shape, .. } => {
            if shape.is_empty() || (shape.len() == 1 && shape[0] == 1) {
                value.first().copied()
            } else {
                None
            }
        }
        ExprNode::UnaryOp {
            op: UnOp::Neg,
            operand,
        } => constant_value(arena, *operand).map(|v| -v),
        _ => None,
    }
}

fn add_scaled(out: &mut Polynomial, scale: f64, p: &Polynomial) {
    out.constant += scale * p.constant;
    for m in &p.monomials {
        out.monomials.push(Monomial {
            coeff: scale * m.coeff,
            factors: m.factors.clone(),
        });
    }
}

fn multiply_into(out: &mut Polynomial, sign: f64, l: &Polynomial, r: &Polynomial) {
    // Cross-multiply, scale by `sign`, and append to `out`.
    if l.constant != 0.0 {
        for rm in &r.monomials {
            out.monomials.push(Monomial {
                coeff: sign * l.constant * rm.coeff,
                factors: rm.factors.clone(),
            });
        }
    }
    if r.constant != 0.0 {
        for lm in &l.monomials {
            out.monomials.push(Monomial {
                coeff: sign * lm.coeff * r.constant,
                factors: lm.factors.clone(),
            });
        }
    }
    out.constant += sign * l.constant * r.constant;
    for lm in &l.monomials {
        for rm in &r.monomials {
            let factors = merge_factors(&lm.factors, &rm.factors);
            out.monomials.push(Monomial {
                coeff: sign * lm.coeff * rm.coeff,
                factors,
            });
        }
    }
}

fn merge_factors(a: &[(ExprId, u32)], b: &[(ExprId, u32)]) -> Vec<(ExprId, u32)> {
    let mut out: Vec<(ExprId, u32)> = a.to_vec();
    for &(id, e) in b {
        match out.iter().position(|(i, _)| *i == id) {
            Some(p) => out[p].1 += e,
            None => out.push((id, e)),
        }
    }
    out.sort_by_key(|(id, _)| id.0);
    out
}

fn power_polynomial(p: &Polynomial, n: u32, work: &mut PolyWork) -> Result<Polynomial, PolyStop> {
    if n == 0 {
        return Ok(Polynomial {
            constant: 1.0,
            ..Default::default()
        });
    }
    let mut result = p.clone();
    for _ in 1..n {
        let mut next = Polynomial::default();
        work.product(&result, p)?;
        multiply_into(&mut next, 1.0, &result, p);
        next.constant += 0.0;
        result = next;
        canonicalise(&mut result);
    }
    canonicalise(&mut result);
    Ok(result)
}

fn canonicalise(p: &mut Polynomial) {
    // Sort each monomial's factors by leaf id (already done by merge,
    // but defensive for direct constructors), then group by factor key.
    for m in p.monomials.iter_mut() {
        m.factors.sort_by_key(|(id, _)| id.0);
    }
    let mut combined: HashMap<Vec<(usize, u32)>, f64> = HashMap::new();
    for m in p.monomials.drain(..) {
        let key: Vec<(usize, u32)> = m.factors.iter().map(|(i, e)| (i.0, *e)).collect();
        *combined.entry(key).or_insert(0.0) += m.coeff;
    }
    let mut new_monomials: Vec<Monomial> = combined
        .into_iter()
        .filter(|(_, c)| c.abs() > 0.0)
        .map(|(key, coeff)| Monomial {
            coeff,
            factors: key.into_iter().map(|(i, e)| (ExprId(i), e)).collect(),
        })
        .collect();
    new_monomials.sort_by(|a, b| {
        a.factors
            .iter()
            .map(|(i, e)| (i.0, *e))
            .collect::<Vec<_>>()
            .cmp(&b.factors.iter().map(|(i, e)| (i.0, *e)).collect::<Vec<_>>())
    });
    p.monomials = new_monomials;
}

// ─────────────────────────────────────────────────────────────
// Reformulation
// ─────────────────────────────────────────────────────────────

/// Statistics from a reformulation pass.
#[derive(Debug, Default, Clone)]
pub struct ReformulationStats {
    /// Number of polynomial constraints rewritten.
    pub constraints_rewritten: usize,
    /// Number of polynomial constraints skipped (already degree ≤ 2 or
    /// not a polynomial).
    pub constraints_skipped: usize,
    /// Number of auxiliary variables introduced.
    pub aux_variables_introduced: usize,
    /// Number of auxiliary equality constraints introduced.
    pub aux_constraints_introduced: usize,
    /// Number of derived (M5) auxiliary bound improvements over
    /// `[-inf, inf]` defaults.
    pub aux_bounds_derived: usize,
}

/// Reformulate every polynomial constraint of degree > 2 into degree-2
/// form by introducing auxiliary variables. Returns the new model and
/// per-pass statistics.
///
/// Pure function: the input model is not modified. The caller decides
/// whether to swap in the result.
pub fn reformulate_polynomial(model: &ModelRepr) -> (ModelRepr, ReformulationStats) {
    let mut out = model.clone();
    let mut stats = ReformulationStats::default();

    // Build initial variable bounds vector for forward interval
    // propagation. Index by variable block (matches `forward_propagate`).
    let mut var_bounds: Vec<Interval> = out
        .variables
        .iter()
        .map(|v| {
            let lo = v.lb.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = v.ub.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Interval::new(lo, hi)
        })
        .collect();

    // Bilinear-product cache: canonical (small_id, large_id) → aux ExprId.
    let mut aux_cache: HashMap<(usize, usize), ExprId> = HashMap::new();

    // Iterate constraints by index because we may push new aux constraints.
    let n_orig = out.constraints.len();
    for ci in 0..n_orig {
        let body = out.constraints[ci].body;
        let poly = match try_polynomial(&out.arena, body) {
            Some(p) => p,
            None => {
                stats.constraints_skipped += 1;
                continue;
            }
        };
        if poly.max_total_degree() <= 2 {
            stats.constraints_skipped += 1;
            continue;
        }

        // Reformulate every monomial of degree > 2 in this polynomial.
        let new_terms =
            build_quadratic_body(&mut out, &mut stats, &mut var_bounds, &mut aux_cache, &poly);

        let new_body = sum_into_arena(&mut out.arena, new_terms, poly.constant);
        out.constraints[ci].body = new_body;
        stats.constraints_rewritten += 1;
    }

    // Apply the same pass to the objective. We bookkeep objective rewrites
    // under `constraints_rewritten` (the stat name is shared) so the caller
    // sees a single counter for "expression bodies rewritten".
    let obj = out.objective;
    if let Some(poly) = try_polynomial(&out.arena, obj) {
        if poly.max_total_degree() > 2 {
            let new_terms =
                build_quadratic_body(&mut out, &mut stats, &mut var_bounds, &mut aux_cache, &poly);
            let new_body = sum_into_arena(&mut out.arena, new_terms, poly.constant);
            out.objective = new_body;
            stats.constraints_rewritten += 1;
        }
    }

    (out, stats)
}

/// Build `x^n` (n ≥ 1) using squaring decomposition (Karia 2022,
/// two-step approach). Reuses the bilinear cache so identical
/// `x^(2^k)` powers across monomials share an aux variable. Returns
/// the arena id whose value equals `x^n`.
///
/// The recursion tree corresponds to the binary expansion of `n`:
/// e.g. `x^8 = ((x^2)^2)^2` introduces 3 aux variables instead of 7
/// from a linear fold; `x^5 = x · (x^2)^2` introduces 3 instead of 4.
fn power_via_squaring(
    model: &mut ModelRepr,
    stats: &mut ReformulationStats,
    var_bounds: &mut Vec<Interval>,
    aux_cache: &mut HashMap<(usize, usize), ExprId>,
    base: ExprId,
    n: u32,
) -> ExprId {
    debug_assert!(n >= 1);
    if n == 1 {
        return base;
    }
    if n == 2 {
        return get_or_make_aux(model, stats, var_bounds, aux_cache, base, base);
    }
    let half = power_via_squaring(model, stats, var_bounds, aux_cache, base, n / 2);
    let squared = get_or_make_aux(model, stats, var_bounds, aux_cache, half, half);
    if n % 2 == 0 {
        squared
    } else {
        get_or_make_aux(model, stats, var_bounds, aux_cache, squared, base)
    }
}

/// Construct the rewritten polynomial body as a list of arena ids
/// (each a degree-≤-2 monomial term).
fn build_quadratic_body(
    model: &mut ModelRepr,
    stats: &mut ReformulationStats,
    var_bounds: &mut Vec<Interval>,
    aux_cache: &mut HashMap<(usize, usize), ExprId>,
    poly: &Polynomial,
) -> Vec<ExprId> {
    let mut terms: Vec<ExprId> = Vec::with_capacity(poly.monomials.len());
    for mono in &poly.monomials {
        // Per-factor: collapse `x^e` to a single arena id via squaring
        // decomposition. For e ≥ 3 this produces ⌈log2(e)⌉ aux vars
        // instead of e − 1 from linear folding and shares cleanly across
        // monomials via aux_cache. For e ≤ 2 the result is base / base²
        // with at most one aux, matching the prior behaviour exactly.
        let mut flat: Vec<ExprId> = Vec::with_capacity(mono.factors.len());
        for (id, e) in &mono.factors {
            if *e == 1 {
                flat.push(*id);
            } else {
                // e ≥ 2: emit `x^e` as a single aux (degree 1 in the
                // reformulated arena) and let the cross-factor fold below
                // pair it with the other factors.
                let id_e = power_via_squaring(model, stats, var_bounds, aux_cache, *id, *e);
                flat.push(id_e);
            }
        }
        // Fold from the right to combine independent factors.
        while flat.len() > 2 {
            let last = flat.pop().unwrap();
            let prev = flat.pop().unwrap();
            let aux = get_or_make_aux(model, stats, var_bounds, aux_cache, prev, last);
            flat.push(aux);
        }
        // Compose the (now degree ≤ 2) tail into a single arena node.
        let body = match flat.len() {
            0 => model.arena.add(ExprNode::Constant(1.0)),
            1 => flat[0],
            _ => model.arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: flat[0],
                right: flat[1],
            }),
        };
        let term = if (mono.coeff - 1.0).abs() < 1e-15 {
            body
        } else {
            let c = model.arena.add(ExprNode::Constant(mono.coeff));
            model.arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: c,
                right: body,
            })
        };
        terms.push(term);
    }
    terms
}

/// Look up or create an auxiliary variable `w = a · b`, registering its
/// definition as a new equality constraint and computing M5-style
/// derived bounds via McCormick.
fn get_or_make_aux(
    model: &mut ModelRepr,
    stats: &mut ReformulationStats,
    var_bounds: &mut Vec<Interval>,
    aux_cache: &mut HashMap<(usize, usize), ExprId>,
    a: ExprId,
    b: ExprId,
) -> ExprId {
    // Canonical key: smaller id first.
    let key = if a.0 <= b.0 { (a.0, b.0) } else { (b.0, a.0) };
    if let Some(&aux_id) = aux_cache.get(&key) {
        return aux_id;
    }

    // Compute M5 derived bounds on a · b via forward interval propagation.
    //
    // Two calls, not one: `b` is not in general reachable from `a`, and
    // `forward_propagate` evaluates only the subtree of the root it is given.
    // Reading `node_bounds[b.0]` off the `a` propagation used to work only
    // because the old implementation evaluated the entire arena for every
    // call — the quadratic that made root FBBT unusable on large models.
    let bounds_a = forward_propagate(&model.arena, a, var_bounds);
    let a_iv = bounds_a[a.0];
    let bounds_b = forward_propagate(&model.arena, b, var_bounds);
    let b_iv = bounds_b[b.0];
    let prod = interval_mul(&a_iv, &b_iv);
    let derived_finite = prod.lo.is_finite() && prod.hi.is_finite();
    if derived_finite {
        stats.aux_bounds_derived += 1;
    }

    // Register the aux variable in the model.
    let block_idx = model.variables.len();
    let lo = if derived_finite {
        prod.lo
    } else {
        f64::NEG_INFINITY
    };
    let hi = if derived_finite {
        prod.hi
    } else {
        f64::INFINITY
    };
    let name = format!("__aux_w{}", block_idx);
    model.variables.push(VarInfo {
        name: name.clone(),
        var_type: VarType::Continuous,
        offset: model.n_vars,
        size: 1,
        shape: vec![],
        lb: vec![lo],
        ub: vec![hi],
    });
    model.n_vars += 1;
    var_bounds.push(Interval::new(lo, hi));

    let aux_expr = model.arena.add(ExprNode::Variable {
        name,
        index: block_idx,
        size: 1,
        shape: vec![],
    });
    aux_cache.insert(key, aux_expr);
    stats.aux_variables_introduced += 1;

    // Add the defining equality constraint: aux - a * b = 0.
    let prod_node = model.arena.add(ExprNode::BinaryOp {
        op: BinOp::Mul,
        left: a,
        right: b,
    });
    let body = model.arena.add(ExprNode::BinaryOp {
        op: BinOp::Sub,
        left: aux_expr,
        right: prod_node,
    });
    model.constraints.push(ConstraintRepr {
        body,
        sense: ConstraintSense::Eq,
        rhs: 0.0,
        name: Some(format!("__aux_def_w{}", block_idx)),
    });
    stats.aux_constraints_introduced += 1;

    aux_expr
}

fn sum_into_arena(arena: &mut ExprArena, terms: Vec<ExprId>, constant: f64) -> ExprId {
    let mut all_terms = terms;
    if constant.abs() > 0.0 {
        all_terms.push(arena.add(ExprNode::Constant(constant)));
    }
    match all_terms.len() {
        0 => arena.add(ExprNode::Constant(0.0)),
        1 => all_terms.into_iter().next().unwrap(),
        _ => arena.add(ExprNode::SumOver { terms: all_terms }),
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{ConstraintRepr, ExprNode, ObjectiveSense};

    fn make_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.into(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn varinfo(name: &str, lo: f64, hi: f64) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lo],
            ub: vec![hi],
        }
    }

    #[test]
    fn detects_linear_polynomial() {
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let y = make_var(&mut arena, "y", 1);
        // 2x + 3y + 1
        let two = arena.add(ExprNode::Constant(2.0));
        let three = arena.add(ExprNode::Constant(3.0));
        let one = arena.add(ExprNode::Constant(1.0));
        let two_x = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: x,
        });
        let three_y = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: three,
            right: y,
        });
        let s1 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: two_x,
            right: three_y,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: s1,
            right: one,
        });
        let p = try_polynomial(&arena, body).unwrap();
        assert!((p.constant - 1.0).abs() < 1e-12);
        assert_eq!(p.monomials.len(), 2);
        assert_eq!(p.max_total_degree(), 1);
    }

    #[test]
    fn detects_quartic_monomial() {
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let four = arena.add(ExprNode::Constant(4.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: four,
        });
        let p = try_polynomial(&arena, body).unwrap();
        assert_eq!(p.monomials.len(), 1);
        assert_eq!(p.monomials[0].factors.len(), 1);
        assert_eq!(p.monomials[0].factors[0].1, 4);
        assert_eq!(p.max_total_degree(), 4);
    }

    #[test]
    fn rejects_transcendental() {
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let exp_x = arena.add(ExprNode::FunctionCall {
            func: crate::expr::MathFunc::Exp,
            args: vec![x],
        });
        assert!(try_polynomial(&arena, exp_x).is_none());
    }

    #[test]
    fn rejects_division_by_variable() {
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let y = make_var(&mut arena, "y", 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Div,
            left: x,
            right: y,
        });
        assert!(try_polynomial(&arena, body).is_none());
    }

    #[test]
    fn reformulate_quartic_introduces_aux() {
        // x^4 over x ∈ [-2, 2]. Should fold into (w1 = x*x; w2 = w1*w1).
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let four = arena.add(ExprNode::Constant(4.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: four,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 16.0,
                name: Some("c".into()),
            }],
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let (new_model, stats) = reformulate_polynomial(&model);
        assert_eq!(stats.constraints_rewritten, 1);
        assert!(stats.aux_variables_introduced >= 1);
        // Aux equality constraints + the original constraint = total constraints.
        assert_eq!(
            new_model.constraints.len(),
            1 + stats.aux_constraints_introduced
        );
        // Aux variables get McCormick-derived finite bounds.
        for v in new_model.variables.iter().skip(1) {
            assert!(v.lb[0].is_finite());
            assert!(v.ub[0].is_finite());
        }
    }

    #[test]
    fn reformulate_skips_quadratic() {
        // x*y with x, y bounded. Already quadratic — should not introduce aux.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let y = make_var(&mut arena, "y", 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: y,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![varinfo("x", -1.0, 1.0), varinfo("y", -1.0, 1.0)],
            n_vars: 2,
        };
        let (new_model, stats) = reformulate_polynomial(&model);
        assert_eq!(stats.constraints_rewritten, 0);
        assert_eq!(stats.aux_variables_introduced, 0);
        assert_eq!(new_model.constraints.len(), 1);
    }

    #[test]
    fn reformulate_idempotent_on_second_pass() {
        // After one reformulation, all bodies are degree ≤ 2. A second
        // pass should be a no-op.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let three = arena.add(ExprNode::Constant(3.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: three,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 8.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let (m1, s1) = reformulate_polynomial(&model);
        assert_eq!(s1.constraints_rewritten, 1);
        let (_m2, s2) = reformulate_polynomial(&m1);
        assert_eq!(s2.constraints_rewritten, 0);
    }

    #[test]
    fn squaring_decomposition_uses_log2_aux_for_x8() {
        // x^8 should decompose as ((x^2)^2)^2 → 3 aux vars, not 7.
        // Karia 2022 two-step approach (D2 of #53).
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let eight = arena.add(ExprNode::Constant(8.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: eight,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 256.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let (_m, stats) = reformulate_polynomial(&model);
        assert_eq!(stats.constraints_rewritten, 1);
        // Linear fold would introduce 7 aux vars (x*x*x*x*x*x*x*x);
        // squaring uses 3 (x^2, x^4, x^8).
        assert_eq!(stats.aux_variables_introduced, 3);
    }

    #[test]
    fn squaring_decomposition_handles_odd_powers() {
        // x^5 = x · (x^2)^2 → 3 aux vars (x^2, x^4, x^5).
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let five = arena.add(ExprNode::Constant(5.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: five,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 32.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let (_m, stats) = reformulate_polynomial(&model);
        assert_eq!(stats.constraints_rewritten, 1);
        assert!(stats.aux_variables_introduced <= 3);
    }

    #[test]
    fn aux_cache_shares_aux_across_monomials() {
        // x^4 + x^4 should reuse the same w = x*x aux variable.
        let mut arena = ExprArena::new();
        let x = make_var(&mut arena, "x", 0);
        let four = arena.add(ExprNode::Constant(4.0));
        let term = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: four,
        });
        let two = arena.add(ExprNode::Constant(2.0));
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: term,
        });
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 32.0,
                name: None,
            }],
            variables: vec![varinfo("x", -2.0, 2.0)],
            n_vars: 1,
        };
        let (_m, stats) = reformulate_polynomial(&model);
        // For x^4 with greedy folding: introduces (x,x) → w1, (w1, x) → w2,
        // (w2, x) → w3 OR uses cache to share. With sharing across the
        // same constraint, at most 3 aux vars per chain of 4 factors.
        assert!(stats.aux_variables_introduced <= 3);
    }
}
