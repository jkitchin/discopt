//! Batch substitution-graph aggregator — P2(a′) item (i) (issue #844).
//!
//! ## Why this exists
//!
//! [`super::aggregate::aggregate_variables`] and
//! [`super::eliminate::eliminate_variables`] both require the eliminated
//! variable to appear in **exactly one expression** (its defining equality and
//! nowhere else), and both rescan every constraint per aggregation. The
//! P2(a) entry experiment (`docs/dev/sota-parity-analysis-2026-07-27.md`)
//! measured the consequences on `watercontamination0202` (106,711 vars,
//! 106,201 linear equalities): 342 aggregations in 30 s (≈11/s, projecting
//! ~2.6 h) for a 1.00× reduction, against SCIP's 189× in 0.19 s.
//!
//! This pass removes both limitations at once:
//!
//! 1. **Substitute-everywhere semantics.** An eliminated variable is rewritten
//!    out of *every* expression it appears in — inequalities, nonlinear rows
//!    and the objective included — not just its defining equality.
//! 2. **One rewrite pass.** All definitions are resolved *first* (into affine
//!    maps onto a surviving representative), then a single walk over the
//!    expression DAG applies every substitution and the variable renumbering
//!    together. Cost is O(nnz + nodes + eliminations·α) rather than
//!    O(eliminations × rows).
//!
//! ## The transform
//!
//! Every `Eq` row with a linear (total-degree-1) body is resolved into
//! *representative space* by the union-find below. A row that lands on
//!
//! ```text
//!   one representative:   A·x_r == R          ⇒  x_r fixed to R/A
//!   two representatives:  A·x_p + B·x_q == R  ⇒  x_p = (−B/A)·x_q + R/A
//! ```
//!
//! contributes an elimination. The union-find carries an **affine potential**:
//! each non-root block `i` stores `x_i = coeff_i · x_parent(i) + offset_i`, and
//! `find` composes the maps along the path (with path compression). Chains of
//! arbitrary depth therefore collapse to a *direct* definition on the class
//! representative, which is why postsolve needs no topological ordering (see
//! [`postsolve_point`]).
//!
//! A row whose variables all resolve to representatives that are already
//! related — the **cycle** case — carries no new elimination: it is rejected
//! and left in the model, where FBBT/redundancy can act on it. Rejecting
//! rather than "simplifying" keeps this pass an exact reformulation.
//!
//! ## Exactness
//!
//! The transform is an equivalence, not a relaxation. For every eliminated
//! block `e` with `x_e = c·x_s + o`:
//!
//! - the defining row is dropped, and it holds by construction at any point of
//!   the reduced model lifted through the definition;
//! - `x_e ∈ [l_e, u_e]` is an interval constraint on a *bijective* affine
//!   image of `x_s`, so it transfers **exactly** onto `x_s`'s bounds
//!   (`scale_interval` below). Nothing is dropped and nothing is relaxed.
//!
//! Consequently: #eliminated blocks == #dropped rows exactly (asserted), the
//! objective value is preserved, and a feasible point of the reduced model
//! lifts to a feasible point of the original.
//!
//! ## Numerical guards (why these thresholds)
//!
//! Substituting `x = m·y + q` multiplies every coefficient on `x` by `m` and
//! injects `q` into every row's constant. Unchecked, a near-zero pivot turns a
//! well-scaled model into an ill-conditioned one, so a candidate is rejected
//! (the row is *kept* — nothing is lost but the reduction) when:
//!
//! - the pivot is below [`MIN_PIVOT_ABS`] in absolute value: at that magnitude
//!   a coefficient is indistinguishable from a structural zero produced by
//!   cancellation while resolving the row into representative space;
//! - the pivot is below [`PIVOT_REL`] times the row's largest resolved
//!   coefficient, which caps the per-step multiplier `|m|` at `1/PIVOT_REL`;
//! - the *composed* multiplier or offset anywhere in the merged class would
//!   exceed [`MAX_ABS_COEFF`] / [`MAX_ABS_OFFSET`]. Chains compose
//!   multiplicatively, so a per-step cap alone does not bound a chain; the
//!   union-find carries a per-class running maximum and a union is rejected
//!   *before* it is applied when the merge would breach the cap.
//!
//! Both caps are on measurable quantities and every rejection is counted in
//! [`SubstitutionStats`], so a disappointing reduction can be attributed
//! rather than guessed at.
//!
//! ## Scope (v0)
//!
//! - Eliminated blocks must be **continuous**, **scalar** (`size == 1`) and
//!   have exactly one `Variable` leaf in the arena. Integer/binary blocks are
//!   never eliminated (substituting them out would move an integrality
//!   requirement onto an affine image of another variable); they may serve as
//!   *sources*, which is sound precisely because the eliminated variable is
//!   continuous.
//! - Only rows resolving to ≤ 2 representatives define a substitution.
//!   General ≥3-variable Gaussian elimination is item (ii) of the P2(a′)
//!   scope and is not done here.
//! - Rows with more than [`MAX_ROW_TERMS`] linear monomials are skipped as
//!   definition candidates (resolving them is quadratic in the term count and
//!   they are never doubletons); they are of course still *rewritten*.
//!
//! ## Determinism
//!
//! Rows are processed in `model.constraints` order; ties in pivot choice break
//! on the lower block index; eliminated blocks are emitted in ascending block
//! order; `HashMap`s are only point-queried, never iterated on a result path.

use std::collections::{HashMap, HashSet};

use super::polynomial::try_polynomial;
use crate::expr::{BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr};
use crate::expr::{VarInfo, VarType};

/// Smallest absolute pivot coefficient accepted for a substitution.
pub const MIN_PIVOT_ABS: f64 = 1e-9;
/// Smallest pivot accepted *relative* to the row's largest resolved
/// coefficient. Caps the per-step substitution multiplier at `1 / PIVOT_REL`.
pub const PIVOT_REL: f64 = 1e-6;
/// Cap on the composed multiplier `|c|` of any eliminated block's definition.
pub const MAX_ABS_COEFF: f64 = 1e6;
/// Cap on the composed offset `|o|` of any eliminated block's definition.
pub const MAX_ABS_OFFSET: f64 = 1e9;
/// Maximum linear monomials in a row considered as a definition candidate.
pub const MAX_ROW_TERMS: usize = 64;
/// Relative slack when testing whether a transferred interval is empty or a
/// fixed value violates its declared bounds.
const FEAS_TOL: f64 = 1e-9;

/// How an eliminated variable is recovered from the reduced solution.
#[derive(Debug, Clone, PartialEq)]
pub enum SubstDef {
    /// `x = value` — determined by a row that resolved to a single
    /// representative (possibly after chain resolution).
    Fixed(f64),
    /// `x = coeff · x[source_block] + offset`, where `source_block` is a
    /// **surviving** block index in the *input* model's numbering.
    Affine {
        /// Input-model block index of the surviving source variable.
        source_block: usize,
        /// Multiplier.
        coeff: f64,
        /// Additive offset.
        offset: f64,
    },
}

/// One recorded elimination, in input-model block numbering.
#[derive(Debug, Clone, PartialEq)]
pub struct SubstitutionRecord {
    /// Block index of the eliminated variable in the *input* model.
    pub eliminated_block: usize,
    /// How to recover it.
    pub def: SubstDef,
}

/// Statistics and the postsolve payload from one substitution run.
#[derive(Debug, Clone, Default)]
pub struct SubstitutionStats {
    /// Number of variable blocks removed.
    pub variables_eliminated: usize,
    /// Number of equalities dropped. Always equals `variables_eliminated`.
    pub equalities_dropped: usize,
    /// Linear equality rows examined as definition candidates.
    pub candidate_rows: usize,
    /// Rows rejected because they resolved to 0 or >2 representatives.
    pub cycles_rejected: usize,
    /// Rows rejected by the pivot-magnitude guards.
    pub pivots_rejected: usize,
    /// Rows rejected because the merge would breach the coefficient/offset caps.
    pub growth_rejected: usize,
    /// Rows rejected because no resolved representative was eliminable.
    pub ineligible_rejected: usize,
    /// Set when a transferred interval is empty or a fixed value falls outside
    /// its declared bounds. The model is returned **unchanged** in that case,
    /// leaving the infeasibility for FBBT to report.
    pub infeasible_detected: bool,
    /// Set when the pass declined to apply an otherwise-computed reduction
    /// (non-finite composed definition, broken invariant). Model unchanged.
    pub aborted: Option<String>,
    /// Recovery records, one per eliminated block, ascending by block index.
    pub records: Vec<SubstitutionRecord>,
    /// `old_block -> Some(new_block)` for survivors, `None` for eliminated.
    pub block_map: Vec<Option<usize>>,
}

// ─────────────────────────────────────────────────────────────
// Union-find over variable blocks with an affine potential
// ─────────────────────────────────────────────────────────────

/// `x_i = coeff[i] · x_parent[i] + offset[i]`; roots are their own parent with
/// the identity map. `max_coeff`/`max_offset` are maintained **on roots only**
/// and upper-bound `|c|`/`|o|` over every member of that root's class.
struct AffineUf {
    parent: Vec<usize>,
    coeff: Vec<f64>,
    offset: Vec<f64>,
    max_coeff: Vec<f64>,
    max_offset: Vec<f64>,
    fixed: Vec<Option<f64>>,
}

impl AffineUf {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            coeff: vec![1.0; n],
            offset: vec![0.0; n],
            max_coeff: vec![1.0; n],
            max_offset: vec![0.0; n],
            fixed: vec![None; n],
        }
    }

    /// Resolve `i` to its representative, returning `(root, c, o)` with
    /// `x_i = c · x_root + o`. Iterative — substitution chains reach 10^4 on
    /// the target instances, so a recursive `find` would risk a stack overflow
    /// — with path compression: after the call, every node on the path points
    /// directly at the root carrying its composed map.
    fn find(&mut self, i: usize) -> (usize, f64, f64) {
        let mut path: Vec<usize> = Vec::new();
        let mut cur = i;
        while self.parent[cur] != cur {
            path.push(cur);
            cur = self.parent[cur];
        }
        let root = cur;
        let mut c = 1.0f64;
        let mut o = 0.0f64;
        for &node in path.iter().rev() {
            // x_node = coeff[node]·x_parent + offset[node]; the parent already
            // composes to (c, o) against the root.
            let nc = self.coeff[node] * c;
            let no = self.coeff[node] * o + self.offset[node];
            c = nc;
            o = no;
            self.parent[node] = root;
            self.coeff[node] = nc;
            self.offset[node] = no;
        }
        if i == root {
            (root, 1.0, 0.0)
        } else {
            (root, self.coeff[i], self.offset[i])
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────

/// Run the batch substitution aggregator once. Pure function.
///
/// Returns the reduced model and the stats/postsolve payload. When
/// `infeasible_detected` or `aborted` is set the **input model is returned
/// unchanged** with an empty record list, so the caller can always treat the
/// result as a drop-in replacement.
pub fn substitute_variables(model: &ModelRepr) -> (ModelRepr, SubstitutionStats) {
    let n_blocks = model.variables.len();
    let mut stats = SubstitutionStats {
        block_map: (0..n_blocks).map(Some).collect(),
        ..Default::default()
    };

    let (leaf_to_block, block_to_leaf, aliased) = scalar_leaf_maps(model);
    if leaf_to_block.is_empty() {
        return (model.clone(), stats);
    }

    let eliminable = |b: usize| -> bool {
        model.variables[b].var_type == VarType::Continuous
            && model.variables[b].size == 1
            && !model.variables[b].lb.is_empty()
            && !aliased.contains(&b)
            && block_to_leaf.contains_key(&b)
    };

    let mut uf = AffineUf::new(n_blocks);
    let mut consumed_rows: Vec<usize> = Vec::new();
    let mut eliminated: HashSet<usize> = HashSet::new();

    for (ci, c) in model.constraints.iter().enumerate() {
        if c.sense != ConstraintSense::Eq {
            continue;
        }
        let Some(row) = linear_row(model, c, &leaf_to_block) else {
            continue;
        };
        stats.candidate_rows += 1;

        // Resolve the row into representative space, folding already-fixed
        // representatives into the constant.
        let mut rterms: Vec<(usize, f64)> = Vec::with_capacity(2);
        let mut lhs_const = row.constant;
        let mut row_max = 0.0f64;
        for &(b, cb) in &row.terms {
            let (root, pc, po) = uf.find(b);
            lhs_const += cb * po;
            let a = cb * pc;
            row_max = row_max.max(a.abs());
            if let Some(v) = uf.fixed[root] {
                lhs_const += a * v;
            } else if let Some(slot) = rterms.iter_mut().find(|(r, _)| *r == root) {
                slot.1 += a;
            } else {
                rterms.push((root, a));
            }
        }
        let r = c.rhs - lhs_const;
        if !r.is_finite() || !row_max.is_finite() {
            stats.pivots_rejected += 1;
            continue;
        }
        // A merged coefficient of *exactly* zero is a structural absence and is
        // safe to drop; a near-zero one is a cancellation residual that must be
        // kept so the relative pivot guard below can reject it.
        rterms.retain(|(_, a)| *a != 0.0);

        match rterms.len() {
            // Constant relation: redundant or infeasible; either way the row
            // stays and FBBT decides.
            0 => stats.cycles_rejected += 1,
            1 => {
                let (root, a) = rterms[0];
                if a.abs() < MIN_PIVOT_ABS || a.abs() < PIVOT_REL * row_max {
                    stats.pivots_rejected += 1;
                    continue;
                }
                let v = r / a;
                if !v.is_finite() || v.abs() > MAX_ABS_OFFSET {
                    stats.growth_rejected += 1;
                    continue;
                }
                if !eliminable(root) || eliminated.contains(&root) {
                    stats.ineligible_rejected += 1;
                    continue;
                }
                uf.fixed[root] = Some(v);
                eliminated.insert(root);
                consumed_rows.push(ci);
            }
            2 => {
                let (root_p, a) = rterms[0];
                let (root_q, b) = rterms[1];
                let p_ok = eliminable(root_p) && !eliminated.contains(&root_p);
                let q_ok = eliminable(root_q) && !eliminated.contains(&root_q);
                // Prefer eliminating the block with the larger coefficient
                // (multiplier ≤ 1), subject to eligibility; ties on block index.
                let pick_p = match (p_ok, q_ok) {
                    (true, true) => match a.abs().partial_cmp(&b.abs()) {
                        Some(std::cmp::Ordering::Greater) => true,
                        Some(std::cmp::Ordering::Less) => false,
                        _ => root_p < root_q,
                    },
                    (true, false) => true,
                    (false, true) => false,
                    (false, false) => {
                        stats.ineligible_rejected += 1;
                        continue;
                    }
                };
                let (elim_root, pivot, keep_root, other) = if pick_p {
                    (root_p, a, root_q, b)
                } else {
                    (root_q, b, root_p, a)
                };
                if pivot.abs() < MIN_PIVOT_ABS || pivot.abs() < PIVOT_REL * row_max {
                    stats.pivots_rejected += 1;
                    continue;
                }
                // x_elim_root = m·x_keep_root + q
                let m = -other / pivot;
                let q = r / pivot;
                if !m.is_finite() || !q.is_finite() || m == 0.0 {
                    stats.pivots_rejected += 1;
                    continue;
                }
                let new_max_coeff = uf.max_coeff[elim_root] * m.abs();
                let new_max_offset = uf.max_offset[elim_root] * m.abs() + q.abs();
                if new_max_coeff > MAX_ABS_COEFF || new_max_offset > MAX_ABS_OFFSET {
                    stats.growth_rejected += 1;
                    continue;
                }
                uf.parent[elim_root] = keep_root;
                uf.coeff[elim_root] = m;
                uf.offset[elim_root] = q;
                uf.max_coeff[keep_root] = uf.max_coeff[keep_root].max(new_max_coeff);
                uf.max_offset[keep_root] = uf.max_offset[keep_root].max(new_max_offset);
                eliminated.insert(elim_root);
                consumed_rows.push(ci);
            }
            _ => stats.cycles_rejected += 1,
        }
    }

    if eliminated.is_empty() {
        return (model.clone(), stats);
    }

    // ── Resolve every eliminated block to a direct definition ──────────
    let mut lb: Vec<f64> = Vec::with_capacity(n_blocks);
    let mut ub: Vec<f64> = Vec::with_capacity(n_blocks);
    for v in &model.variables {
        if v.size == 1 && !v.lb.is_empty() && !v.ub.is_empty() {
            lb.push(v.lb[0]);
            ub.push(v.ub[0]);
        } else {
            lb.push(f64::NEG_INFINITY);
            ub.push(f64::INFINITY);
        }
    }

    let mut elim_sorted: Vec<usize> = eliminated.iter().copied().collect();
    elim_sorted.sort_unstable();
    let mut records: Vec<SubstitutionRecord> = Vec::with_capacity(elim_sorted.len());
    for &e in &elim_sorted {
        let (root, c, o) = uf.find(e);
        let (elo, ehi) = (model.variables[e].lb[0], model.variables[e].ub[0]);
        let def = if let Some(v) = uf.fixed[root] {
            let val = c * v + o;
            if !val.is_finite() {
                return abort(model, stats, "non-finite fixed value");
            }
            let slack = FEAS_TOL * (1.0 + val.abs());
            if val < elo - slack || val > ehi + slack {
                stats.infeasible_detected = true;
                return (model.clone(), stats);
            }
            SubstDef::Fixed(val)
        } else {
            if !c.is_finite() || !o.is_finite() || c == 0.0 {
                return abort(model, stats, "non-finite affine definition");
            }
            // Transfer x_e ∈ [l_e, u_e] onto the source exactly:
            //   x_root = (x_e − o) / c
            let (tlo, thi) = scale_interval(1.0 / c, elo - o, ehi - o);
            if tlo.is_nan() || thi.is_nan() {
                return abort(model, stats, "non-finite transferred bound");
            }
            if tlo > lb[root] {
                lb[root] = tlo;
            }
            if thi < ub[root] {
                ub[root] = thi;
            }
            if lb[root] > ub[root] + FEAS_TOL * (1.0 + lb[root].abs()) {
                stats.infeasible_detected = true;
                return (model.clone(), stats);
            }
            SubstDef::Affine {
                source_block: root,
                coeff: c,
                offset: o,
            }
        };
        records.push(SubstitutionRecord {
            eliminated_block: e,
            def,
        });
    }

    // A representative named as an Affine source must survive. Check, don't
    // trust: a violation would make postsolve read an unassigned slot.
    for rec in &records {
        if let SubstDef::Affine { source_block, .. } = rec.def {
            if eliminated.contains(&source_block) {
                return abort(model, stats, "affine source was itself eliminated");
            }
        }
    }

    let mut block_map: Vec<Option<usize>> = Vec::with_capacity(n_blocks);
    let mut next = 0usize;
    for b in 0..n_blocks {
        if eliminated.contains(&b) {
            block_map.push(None);
        } else {
            block_map.push(Some(next));
            next += 1;
        }
    }

    let drop_rows: HashSet<usize> = consumed_rows.iter().copied().collect();
    assert_eq!(
        drop_rows.len(),
        records.len(),
        "substitution invariant: exactly one dropped equality per eliminated block"
    );

    let new_model = rewrite_model(
        model,
        &records,
        &drop_rows,
        &block_map,
        &block_to_leaf,
        &lb,
        &ub,
    );

    stats.variables_eliminated = records.len();
    stats.equalities_dropped = drop_rows.len();
    stats.records = records;
    stats.block_map = block_map;
    (new_model, stats)
}

fn abort(
    model: &ModelRepr,
    mut stats: SubstitutionStats,
    why: &str,
) -> (ModelRepr, SubstitutionStats) {
    stats.aborted = Some(why.to_string());
    stats.variables_eliminated = 0;
    stats.equalities_dropped = 0;
    stats.records.clear();
    stats.block_map = (0..model.variables.len()).map(Some).collect();
    (model.clone(), stats)
}

/// Iterate [`substitute_variables`] until no further reduction, at most
/// `max_sweeps` times.
///
/// Later sweeps find definitions the first cannot: a row with three variables
/// two of which land in the same class resolves to a doubleton only after the
/// first sweep's rewrite.
///
/// Returns `(models, chain)` where `chain[i]` is sweep `i`'s stats, `models[i]`
/// is the model that sweep consumed, and `models[chain.len()]` is the final
/// reduced model. [`postsolve_chain`] inverts the sweeps in reverse order.
pub fn substitute_to_fixpoint(
    model: &ModelRepr,
    max_sweeps: usize,
) -> (Vec<ModelRepr>, Vec<SubstitutionStats>) {
    let mut models: Vec<ModelRepr> = vec![model.clone()];
    let mut chain: Vec<SubstitutionStats> = Vec::new();
    for _ in 0..max_sweeps {
        let (next, st) = substitute_variables(models.last().expect("seeded above"));
        if st.variables_eliminated == 0 {
            break;
        }
        models.push(next);
        chain.push(st);
    }
    (models, chain)
}

/// Lift a reduced-model point back to the original variable space.
///
/// `x_red` is a flat scalar vector over `reduced.n_vars`; the result is a flat
/// vector over `original.n_vars`. Because every `Affine` source is a
/// *surviving* block, one pass suffices — no topological ordering is needed.
pub fn postsolve_point(
    original: &ModelRepr,
    reduced: &ModelRepr,
    stats: &SubstitutionStats,
    x_red: &[f64],
) -> Result<Vec<f64>, String> {
    if x_red.len() != reduced.n_vars {
        return Err(format!(
            "postsolve: point has {} entries, reduced model has {} scalar vars",
            x_red.len(),
            reduced.n_vars
        ));
    }
    if stats.block_map.len() != original.variables.len() {
        return Err("postsolve: block_map length does not match the original model".to_string());
    }
    let mut out = vec![f64::NAN; original.n_vars];
    for (b, slot) in stats.block_map.iter().enumerate() {
        let Some(nb) = slot else { continue };
        let ov = &original.variables[b];
        let nv = reduced
            .variables
            .get(*nb)
            .ok_or_else(|| format!("postsolve: reduced block {nb} is out of range"))?;
        if ov.size != nv.size {
            return Err(format!(
                "postsolve: block {b} changed size during substitution"
            ));
        }
        out[ov.offset..ov.offset + ov.size].copy_from_slice(&x_red[nv.offset..nv.offset + nv.size]);
    }
    for rec in &stats.records {
        let ov = &original.variables[rec.eliminated_block];
        let v = match rec.def {
            SubstDef::Fixed(v) => v,
            SubstDef::Affine {
                source_block,
                coeff,
                offset,
            } => {
                let s = out[original.variables[source_block].offset];
                if s.is_nan() {
                    return Err(format!(
                        "postsolve: source block {source_block} was not recovered before use"
                    ));
                }
                coeff * s + offset
            }
        };
        out[ov.offset] = v;
    }
    if let Some(i) = out.iter().position(|v| v.is_nan()) {
        return Err(format!("postsolve: scalar variable {i} was never assigned"));
    }
    Ok(out)
}

/// Invert a whole chain of sweeps produced by [`substitute_to_fixpoint`].
///
/// `models[i]` is the model that sweep `i` consumed; `models[chain.len()]` is
/// the final reduced model. Sweeps are inverted in reverse order.
pub fn postsolve_chain(
    models: &[ModelRepr],
    chain: &[SubstitutionStats],
    x_red: &[f64],
) -> Result<Vec<f64>, String> {
    if models.len() != chain.len() + 1 {
        return Err(format!(
            "postsolve_chain: {} models for {} sweeps (expected {})",
            models.len(),
            chain.len(),
            chain.len() + 1
        ));
    }
    let mut x = x_red.to_vec();
    for i in (0..chain.len()).rev() {
        x = postsolve_point(&models[i], &models[i + 1], &chain[i], &x)?;
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────
// Row extraction
// ─────────────────────────────────────────────────────────────

struct LinearRow {
    /// `(block_index, coefficient)` in polynomial order, distinct blocks.
    terms: Vec<(usize, f64)>,
    constant: f64,
}

/// Extract a linear row over scalar variable blocks, or `None`.
fn linear_row(
    model: &ModelRepr,
    c: &ConstraintRepr,
    leaf_to_block: &HashMap<ExprId, usize>,
) -> Option<LinearRow> {
    if !c.rhs.is_finite() {
        return None;
    }
    let poly = try_polynomial(&model.arena, c.body)?;
    if poly.max_total_degree() != 1 {
        return None;
    }
    if poly.monomials.is_empty() || poly.monomials.len() > MAX_ROW_TERMS {
        return None;
    }
    if !poly.constant.is_finite() {
        return None;
    }
    let mut terms: Vec<(usize, f64)> = Vec::with_capacity(poly.monomials.len());
    for m in &poly.monomials {
        if m.factors.len() != 1 || m.factors[0].1 != 1 {
            return None;
        }
        let block = *leaf_to_block.get(&m.factors[0].0)?;
        if m.coeff == 0.0 || !m.coeff.is_finite() {
            return None;
        }
        terms.push((block, m.coeff));
    }
    Some(LinearRow {
        terms,
        constant: poly.constant,
    })
}

/// `(leaf -> block, block -> canonical leaf, aliased blocks)` for scalar
/// variable blocks. A block with more than one `Variable` leaf is *aliased*:
/// it is still renumbered, but never eliminated. Its canonical (lowest-id)
/// leaf is retained so it can still serve as a substitution source.
type LeafMaps = (
    HashMap<ExprId, usize>,
    HashMap<usize, ExprId>,
    HashSet<usize>,
);

fn scalar_leaf_maps(model: &ModelRepr) -> LeafMaps {
    let mut leaf_to_block: HashMap<ExprId, usize> = HashMap::new();
    let mut block_to_leaf: HashMap<usize, ExprId> = HashMap::new();
    let mut aliased: HashSet<usize> = HashSet::new();
    for nid in 0..model.arena.len() {
        let id = ExprId(nid);
        if let ExprNode::Variable { index, size, .. } = model.arena.get(id) {
            if *size != 1 || *index >= model.variables.len() || model.variables[*index].size != 1 {
                continue;
            }
            leaf_to_block.insert(id, *index);
            // Ascending scan ⇒ the first insert is the lowest node id.
            if block_to_leaf.insert(*index, id).is_some() {
                aliased.insert(*index);
                // Restore the lowest-id canonical leaf.
                block_to_leaf.insert(*index, first_leaf_of(model, *index));
            }
        }
    }
    (leaf_to_block, block_to_leaf, aliased)
}

fn first_leaf_of(model: &ModelRepr, block: usize) -> ExprId {
    for nid in 0..model.arena.len() {
        if let ExprNode::Variable { index, size, .. } = model.arena.get(ExprId(nid)) {
            if *index == block && *size == 1 {
                return ExprId(nid);
            }
        }
    }
    unreachable!("block {block} has at least one scalar leaf by construction")
}

// ─────────────────────────────────────────────────────────────
// Single-pass rewrite
// ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rewrite_model(
    model: &ModelRepr,
    records: &[SubstitutionRecord],
    drop_rows: &HashSet<usize>,
    block_map: &[Option<usize>],
    block_to_leaf: &HashMap<usize, ExprId>,
    lb: &[f64],
    ub: &[f64],
) -> ModelRepr {
    let old = &model.arena;
    let mut new = ExprArena::with_capacity(old.len());
    let mut map: Vec<Option<ExprId>> = vec![None; old.len()];

    // Pre-create the `Variable` nodes for surviving scalar blocks so that a
    // substitution can reference its source regardless of node ordering.
    let mut survivor_node: HashMap<usize, ExprId> = HashMap::new();
    for (b, slot) in block_map.iter().enumerate() {
        let Some(nb) = slot else { continue };
        let Some(leaf) = block_to_leaf.get(&b) else {
            continue;
        };
        if let ExprNode::Variable { name, shape, .. } = old.get(*leaf) {
            let id = new.add(ExprNode::Variable {
                name: name.clone(),
                index: *nb,
                size: 1,
                shape: shape.clone(),
            });
            survivor_node.insert(b, id);
            map[leaf.0] = Some(id);
        }
    }

    // Build each eliminated block's replacement subtree exactly once. This is
    // what makes the rewrite O(1) per elimination instead of O(rows).
    let mut replacement: HashMap<usize, ExprId> = HashMap::new();
    for rec in records {
        let repl = match rec.def {
            SubstDef::Fixed(v) => new.add(ExprNode::Constant(v)),
            SubstDef::Affine {
                source_block,
                coeff,
                offset,
            } => {
                let src = survivor_node[&source_block];
                let mut t = if coeff == 1.0 {
                    src
                } else {
                    let k = new.add(ExprNode::Constant(coeff));
                    new.add(ExprNode::BinaryOp {
                        op: BinOp::Mul,
                        left: k,
                        right: src,
                    })
                };
                if offset != 0.0 {
                    let k = new.add(ExprNode::Constant(offset));
                    t = new.add(ExprNode::BinaryOp {
                        op: BinOp::Add,
                        left: t,
                        right: k,
                    });
                }
                t
            }
        };
        replacement.insert(rec.eliminated_block, repl);
        if let Some(leaf) = block_to_leaf.get(&rec.eliminated_block) {
            map[leaf.0] = Some(repl);
        }
    }

    let mut rewriter = Rewriter {
        old,
        new,
        map,
        block_map,
        replacement: &replacement,
    };

    let objective = rewriter.rewrite(model.objective);
    let mut constraints: Vec<ConstraintRepr> = Vec::with_capacity(model.constraints.len());
    for (ci, c) in model.constraints.iter().enumerate() {
        if drop_rows.contains(&ci) {
            continue;
        }
        constraints.push(ConstraintRepr {
            body: rewriter.rewrite(c.body),
            sense: c.sense,
            rhs: c.rhs,
            name: c.name.clone(),
        });
    }

    let mut variables: Vec<VarInfo> = Vec::with_capacity(model.variables.len());
    let mut running = 0usize;
    for (b, slot) in block_map.iter().enumerate() {
        if slot.is_none() {
            continue;
        }
        let mut v = model.variables[b].clone();
        if v.size == 1 && !v.lb.is_empty() && !v.ub.is_empty() {
            v.lb[0] = v.lb[0].max(lb[b]);
            v.ub[0] = v.ub[0].min(ub[b]);
        }
        v.offset = running;
        running += v.size;
        variables.push(v);
    }

    ModelRepr {
        arena: rewriter.new,
        objective,
        objective_sense: model.objective_sense,
        constraints,
        variables,
        n_vars: running,
    }
}

struct Rewriter<'a> {
    old: &'a ExprArena,
    new: ExprArena,
    map: Vec<Option<ExprId>>,
    block_map: &'a [Option<usize>],
    replacement: &'a HashMap<usize, ExprId>,
}

impl Rewriter<'_> {
    /// Iterative post-order rebuild of `root` into the new arena. Does not
    /// assume children carry lower ids than their parents, memoises shared
    /// subexpressions, and drops nodes unreachable from the roots.
    fn rewrite(&mut self, root: ExprId) -> ExprId {
        let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];
        while let Some((id, expanded)) = stack.pop() {
            if self.map[id.0].is_some() {
                continue;
            }
            if !expanded {
                stack.push((id, true));
                for ch in children(self.old.get(id)) {
                    if self.map[ch.0].is_none() {
                        stack.push((ch, false));
                    }
                }
                continue;
            }
            let node = self.old.get(id);
            let built = match node {
                ExprNode::Variable {
                    name,
                    index,
                    size,
                    shape,
                } => {
                    if let Some(r) = self.replacement.get(index) {
                        *r
                    } else {
                        let ni = self.block_map[*index]
                            .expect("surviving variable must have a new block index");
                        self.new.add(ExprNode::Variable {
                            name: name.clone(),
                            index: ni,
                            size: *size,
                            shape: shape.clone(),
                        })
                    }
                }
                ExprNode::Constant(v) => self.new.add(ExprNode::Constant(*v)),
                ExprNode::ConstantArray(d, s) => {
                    self.new.add(ExprNode::ConstantArray(d.clone(), s.clone()))
                }
                ExprNode::Parameter { name, value, shape } => self.new.add(ExprNode::Parameter {
                    name: name.clone(),
                    value: value.clone(),
                    shape: shape.clone(),
                }),
                ExprNode::BinaryOp { op, left, right } => {
                    let (l, r) = (self.map[left.0].unwrap(), self.map[right.0].unwrap());
                    self.new.add(ExprNode::BinaryOp {
                        op: *op,
                        left: l,
                        right: r,
                    })
                }
                ExprNode::MatMul { left, right } => {
                    let (l, r) = (self.map[left.0].unwrap(), self.map[right.0].unwrap());
                    self.new.add(ExprNode::MatMul { left: l, right: r })
                }
                ExprNode::UnaryOp { op, operand } => {
                    let o = self.map[operand.0].unwrap();
                    self.new.add(ExprNode::UnaryOp {
                        op: *op,
                        operand: o,
                    })
                }
                ExprNode::Sum { operand, axis } => {
                    let o = self.map[operand.0].unwrap();
                    self.new.add(ExprNode::Sum {
                        operand: o,
                        axis: *axis,
                    })
                }
                ExprNode::FunctionCall { func, args } => {
                    let a: Vec<ExprId> = args.iter().map(|x| self.map[x.0].unwrap()).collect();
                    self.new.add(ExprNode::FunctionCall {
                        func: *func,
                        args: a,
                    })
                }
                ExprNode::Index { base, index } => {
                    let b = self.map[base.0].unwrap();
                    self.new.add(ExprNode::Index {
                        base: b,
                        index: index.clone(),
                    })
                }
                ExprNode::SumOver { terms } => {
                    let t: Vec<ExprId> = terms.iter().map(|x| self.map[x.0].unwrap()).collect();
                    self.new.add(ExprNode::SumOver { terms: t })
                }
            };
            self.map[id.0] = Some(built);
        }
        self.map[root.0].expect("root must be rebuilt")
    }
}

fn children(node: &ExprNode) -> Vec<ExprId> {
    match node {
        ExprNode::Constant(_)
        | ExprNode::ConstantArray(_, _)
        | ExprNode::Parameter { .. }
        | ExprNode::Variable { .. } => Vec::new(),
        ExprNode::BinaryOp { left, right, .. } | ExprNode::MatMul { left, right } => {
            vec![*left, *right]
        }
        ExprNode::UnaryOp { operand, .. } | ExprNode::Sum { operand, .. } => vec![*operand],
        ExprNode::FunctionCall { args, .. } => args.clone(),
        ExprNode::Index { base, .. } => vec![*base],
        ExprNode::SumOver { terms } => terms.clone(),
    }
}

/// Scale `[lo, hi]` by `c`, preserving ordering.
fn scale_interval(c: f64, lo: f64, hi: f64) -> (f64, f64) {
    let a = c * lo;
    let b = c * hi;
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{ConstraintRepr, ObjectiveSense};

    fn scalar_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.to_string(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo(name: &str, lb: f64, ub: f64, ty: VarType) -> VarInfo {
        VarInfo {
            name: name.to_string(),
            var_type: ty,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    /// Build `sum_i coeffs[i] * leaves[i]` as an arena expression.
    fn lin(arena: &mut ExprArena, terms: &[(f64, ExprId)]) -> ExprId {
        let mut acc: Option<ExprId> = None;
        for (c, leaf) in terms {
            let k = arena.add(ExprNode::Constant(*c));
            let t = arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: k,
                right: *leaf,
            });
            acc = Some(match acc {
                None => t,
                Some(a) => arena.add(ExprNode::BinaryOp {
                    op: BinOp::Add,
                    left: a,
                    right: t,
                }),
            });
        }
        acc.expect("at least one term")
    }

    fn finalize(mut variables: Vec<VarInfo>) -> Vec<VarInfo> {
        let mut off = 0;
        for v in variables.iter_mut() {
            v.offset = off;
            off += v.size;
        }
        variables
    }

    /// x0 = 2·x1 + 1, x1 = 3·x2 + 2, x2 = 4·x3 + 3 — a chain of THREE
    /// substitutions — plus one inequality and an objective in which every
    /// chained variable also appears. `aggregate`/`eliminate` reject all three
    /// definitions because each defined variable appears in more than one
    /// expression; this pass must eliminate all three and collapse the chain
    /// onto a single representative.
    fn chain_model() -> ModelRepr {
        let mut arena = ExprArena::new();
        let x: Vec<ExprId> = (0..5)
            .map(|i| scalar_var(&mut arena, &format!("x{i}"), i))
            .collect();
        let e0 = lin(&mut arena, &[(1.0, x[0]), (-2.0, x[1])]);
        let e1 = lin(&mut arena, &[(1.0, x[1]), (-3.0, x[2])]);
        let e2 = lin(&mut arena, &[(1.0, x[2]), (-4.0, x[3])]);
        let ineq = lin(
            &mut arena,
            &[(1.0, x[0]), (1.0, x[1]), (1.0, x[2]), (1.0, x[3])],
        );
        let obj = lin(&mut arena, &[(1.0, x[0]), (1.0, x[4])]);
        let variables = finalize(
            (0..5)
                .map(|i| vinfo(&format!("x{i}"), -1e3, 1e3, VarType::Continuous))
                .collect(),
        );
        ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: e0,
                    sense: ConstraintSense::Eq,
                    rhs: 1.0,
                    name: None,
                },
                ConstraintRepr {
                    body: e1,
                    sense: ConstraintSense::Eq,
                    rhs: 2.0,
                    name: None,
                },
                ConstraintRepr {
                    body: e2,
                    sense: ConstraintSense::Eq,
                    rhs: 3.0,
                    name: None,
                },
                ConstraintRepr {
                    body: ineq,
                    sense: ConstraintSense::Le,
                    rhs: 100.0,
                    name: None,
                },
            ],
            variables,
            n_vars: 5,
        }
    }

    #[test]
    fn legacy_aggregate_cannot_touch_the_chain_model() {
        // Fails-before evidence: the pre-existing pass eliminates nothing here,
        // because every defined variable also appears in the inequality.
        let m = chain_model();
        let (out, stats) = super::super::aggregate::aggregate_variables(&m);
        assert_eq!(stats.variables_aggregated, 0);
        assert_eq!(out.variables.len(), 5);
    }

    #[test]
    fn chain_of_three_substitutions_collapses_to_one_representative() {
        let m = chain_model();
        let (red, st) = substitute_variables(&m);
        assert_eq!(st.variables_eliminated, 3, "x0, x1 and x2 are determined");
        assert_eq!(st.equalities_dropped, 3);
        assert_eq!(red.variables.len(), 2, "x3 and the unused x4 survive");
        assert_eq!(red.constraints.len(), 1, "only the inequality remains");

        // Every definition must name the SAME surviving representative — that
        // is what "the chain collapsed" means, and it is why postsolve needs no
        // topological order.
        let sources: Vec<usize> = st
            .records
            .iter()
            .map(|r| match r.def {
                SubstDef::Affine { source_block, .. } => source_block,
                SubstDef::Fixed(_) => usize::MAX,
            })
            .collect();
        assert_eq!(sources.len(), 3);
        assert!(
            sources.iter().all(|s| *s == sources[0] && *s != usize::MAX),
            "definitions must share one representative, got {sources:?}"
        );
    }

    #[test]
    fn postsolve_inverts_the_chain_exactly() {
        let m = chain_model();
        let (red, st) = substitute_variables(&m);
        for t in [-3.5_f64, 0.0, 2.25, 7.0] {
            let mut x_red = vec![0.0; red.n_vars];
            for (i, v) in red.variables.iter().enumerate() {
                x_red[v.offset] = t + i as f64 * 0.5;
            }
            let full = postsolve_point(&m, &red, &st, &x_red).expect("invertible");
            assert_eq!(full.len(), m.n_vars);
            // Every dropped equality must hold exactly at the lifted point.
            for (k, c) in m.constraints.iter().enumerate() {
                if c.sense != ConstraintSense::Eq {
                    continue;
                }
                let r = m.evaluate_expr(c.body, &full) - c.rhs;
                assert!(r.abs() < 1e-9, "row {k} residual {r} at t={t}");
            }
            // The objective and the surviving inequality body must agree.
            let o_full = m.evaluate_objective(&full);
            let o_red = red.evaluate_objective(&x_red);
            assert!((o_full - o_red).abs() < 1e-9, "{o_full} vs {o_red}");
            let b_full = m.evaluate_expr(m.constraints[3].body, &full);
            let b_red = red.evaluate_expr(red.constraints[0].body, &x_red);
            assert!((b_full - b_red).abs() < 1e-9, "{b_full} vs {b_red}");
        }
    }

    #[test]
    fn cycle_is_rejected_and_its_row_is_kept() {
        // x0 − x1 = 0, x1 − x2 = 0, x2 − x0 = 0: the third row closes a cycle
        // and defines nothing new. It must be rejected, not "simplified away".
        let mut arena = ExprArena::new();
        let x: Vec<ExprId> = (0..3)
            .map(|i| scalar_var(&mut arena, &format!("x{i}"), i))
            .collect();
        let r0 = lin(&mut arena, &[(1.0, x[0]), (-1.0, x[1])]);
        let r1 = lin(&mut arena, &[(1.0, x[1]), (-1.0, x[2])]);
        let r2 = lin(&mut arena, &[(1.0, x[2]), (-1.0, x[0])]);
        let obj = lin(&mut arena, &[(1.0, x[0])]);
        let variables = finalize(
            (0..3)
                .map(|i| vinfo(&format!("x{i}"), -10.0, 10.0, VarType::Continuous))
                .collect(),
        );
        let m = ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: [r0, r1, r2]
                .into_iter()
                .map(|body| ConstraintRepr {
                    body,
                    sense: ConstraintSense::Eq,
                    rhs: 0.0,
                    name: None,
                })
                .collect(),
            variables,
            n_vars: 3,
        };
        let (red, st) = substitute_variables(&m);
        assert_eq!(st.variables_eliminated, 2);
        assert_eq!(st.cycles_rejected, 1, "the closing row must be rejected");
        assert_eq!(red.variables.len(), 1);
        assert_eq!(red.constraints.len(), 1, "the cycle row stays in the model");
        // And it is still satisfied after the rewrite, for any survivor value.
        let full = postsolve_point(&m, &red, &st, &[4.25]).unwrap();
        assert_eq!(full, vec![4.25, 4.25, 4.25]);
        let resid = red.evaluate_expr(red.constraints[0].body, &[4.25]);
        assert!(resid.abs() < 1e-12, "kept cycle row residual {resid}");
    }

    #[test]
    fn integer_blocks_are_never_eliminated_but_may_be_sources() {
        // y integer, x continuous, x − 2·y = 1 ⇒ x must go, y must stay.
        let mut arena = ExprArena::new();
        let xv = scalar_var(&mut arena, "x", 0);
        let yv = scalar_var(&mut arena, "y", 1);
        let row = lin(&mut arena, &[(1.0, xv), (-2.0, yv)]);
        let obj = lin(&mut arena, &[(1.0, xv)]);
        let variables = finalize(vec![
            vinfo("x", -100.0, 100.0, VarType::Continuous),
            vinfo("y", 0.0, 10.0, VarType::Integer),
        ]);
        let m = ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: row,
                sense: ConstraintSense::Eq,
                rhs: 1.0,
                name: None,
            }],
            variables,
            n_vars: 2,
        };
        let (red, st) = substitute_variables(&m);
        assert_eq!(st.variables_eliminated, 1);
        assert_eq!(st.records[0].eliminated_block, 0, "the continuous x goes");
        assert_eq!(red.variables.len(), 1);
        assert_eq!(red.variables[0].var_type, VarType::Integer);
        let full = postsolve_point(&m, &red, &st, &[3.0]).unwrap();
        assert!(
            (full[0] - 7.0).abs() < 1e-12,
            "x = 2·3 + 1 = 7, got {full:?}"
        );
    }

    #[test]
    fn empty_transferred_interval_reports_infeasible_and_changes_nothing() {
        // x ∈ [0, 1], y ∈ [5, 6], x − y = 0 has no solution.
        let mut arena = ExprArena::new();
        let xv = scalar_var(&mut arena, "x", 0);
        let yv = scalar_var(&mut arena, "y", 1);
        let row = lin(&mut arena, &[(1.0, xv), (-1.0, yv)]);
        let obj = lin(&mut arena, &[(1.0, xv)]);
        let variables = finalize(vec![
            vinfo("x", 0.0, 1.0, VarType::Continuous),
            vinfo("y", 5.0, 6.0, VarType::Continuous),
        ]);
        let m = ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: row,
                sense: ConstraintSense::Eq,
                rhs: 0.0,
                name: None,
            }],
            variables,
            n_vars: 2,
        };
        let (red, st) = substitute_variables(&m);
        assert!(st.infeasible_detected);
        assert_eq!(st.variables_eliminated, 0);
        assert_eq!(red.variables.len(), 2, "model returned unchanged");
        assert_eq!(red.constraints.len(), 1);
    }

    #[test]
    fn tiny_pivot_is_rejected() {
        // 1e-12·x + 1·y = 1: eliminating y is fine (pivot 1), but the row's
        // pivot on x would be 1e-12 against a row max of 1. Force the bad
        // choice by making y integer, so only x is eliminable.
        let mut arena = ExprArena::new();
        let xv = scalar_var(&mut arena, "x", 0);
        let yv = scalar_var(&mut arena, "y", 1);
        let row = lin(&mut arena, &[(1e-12, xv), (1.0, yv)]);
        let obj = lin(&mut arena, &[(1.0, xv)]);
        let variables = finalize(vec![
            vinfo("x", -1e6, 1e6, VarType::Continuous),
            vinfo("y", 0.0, 10.0, VarType::Integer),
        ]);
        let m = ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: row,
                sense: ConstraintSense::Eq,
                rhs: 1.0,
                name: None,
            }],
            variables,
            n_vars: 2,
        };
        let (red, st) = substitute_variables(&m);
        assert_eq!(st.variables_eliminated, 0);
        assert_eq!(st.pivots_rejected, 1);
        assert_eq!(red.variables.len(), 2);
    }

    #[test]
    fn fixpoint_sweeps_chain_and_postsolve_chain_inverts_them() {
        let m = chain_model();
        let (models, chain) = substitute_to_fixpoint(&m, 4);
        assert_eq!(models.len(), chain.len() + 1);
        assert!(!chain.is_empty());
        let red = models.last().unwrap();
        let x_red: Vec<f64> = (0..red.n_vars).map(|i| 1.5 + i as f64).collect();
        let full = postsolve_chain(&models, &chain, &x_red).expect("invertible");
        assert_eq!(full.len(), m.n_vars);
        for c in m
            .constraints
            .iter()
            .filter(|c| c.sense == ConstraintSense::Eq)
        {
            assert!((m.evaluate_expr(c.body, &full) - c.rhs).abs() < 1e-9);
        }
        assert!((m.evaluate_objective(&full) - red.evaluate_objective(&x_red)).abs() < 1e-9);
    }

    #[test]
    fn postsolve_rejects_a_mis_sized_point() {
        let m = chain_model();
        let (red, st) = substitute_variables(&m);
        assert!(postsolve_point(&m, &red, &st, &[1.0]).is_err());
    }
}
