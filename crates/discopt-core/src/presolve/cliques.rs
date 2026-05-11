//! Implied-clique extraction across binary variables (F2 of the
//! presolve roadmap).
//!
//! ## What this pass does
//!
//! Detects pairs of binary variables `(b_i, b_j)` that cannot
//! simultaneously equal 1 because some linear constraint forbids it.
//! Each such pair is a 2-clique (edge) in the binary conflict graph;
//! the orchestrator records them on the pass delta for downstream
//! consumers (relaxation compiler, branching, primal heuristic).
//!
//! For a `≤` constraint `Σ a_k x_k + c0  ≤  rhs`, where every term
//! involves a binary variable, two binaries `b_i` and `b_j` conflict
//! if assigning both to 1 (and every other variable at its activity-
//! minimising value) already violates the bound:
//!
//! ```text
//!     a_i + a_j + min_activity_of_rest + c0  >  rhs
//! ```
//!
//! For `≥` and `=` the dual / two-sided test is applied analogously.
//!
//! ## What this pass does *not* do (yet)
//!
//! - It does not enumerate maximal cliques. Only pairwise edges are
//!   produced. Maximal-clique extension over the edge graph is a
//!   future enhancement and is what gives the full reduction in LP
//!   relaxation strength reported by Achterberg et al. (2020).
//! - It does not rewrite or replace the source constraint. The clique
//!   data appears in the delta's `structure.cliques` field; the
//!   model is unchanged.
//! - It only inspects rows whose body is a polynomial of degree ≤ 1
//!   and whose only variable factors are scalar binary variables.
//!   Mixed continuous/binary rows are skipped to keep the activity
//!   computation simple.
//!
//! ## Determinism
//!
//! Constraints are scanned in `model.constraints` order; within each
//! constraint, binary leaves are sorted by variable-block index. Pair
//! detection runs over `(i, j)` with `i < j`, so the resulting clique
//! list is canonical. No `HashMap` iteration on the hot path.

use std::collections::BTreeSet;

use super::polynomial::try_polynomial;
use crate::expr::{ConstraintSense, ExprNode, ModelRepr, VarType};

/// Per-pass diagnostics for clique extraction.
#[derive(Debug, Clone, Default)]
pub struct CliqueStats {
    /// Number of constraints inspected.
    pub linear_rows_scanned: usize,
    /// Number of distinct conflict edges discovered (after dedup).
    pub edges_found: usize,
}

/// Result of [`extract_cliques`]: a list of conflict edges between
/// binary variable blocks. Each edge is `(i, j)` with `i < j`.
#[derive(Debug, Clone, Default)]
pub struct CliqueSet {
    /// Edges, sorted lexicographically by `(i, j)`.
    pub edges: Vec<(usize, usize)>,
}

/// Run binary clique extraction. Pure function; never modifies the
/// model.
pub fn extract_cliques(model: &ModelRepr) -> (CliqueSet, CliqueStats) {
    let mut stats = CliqueStats::default();
    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();

    // Pre-build a map from variable-block index to "is binary".
    let is_binary: Vec<bool> = model
        .variables
        .iter()
        .map(|v| matches!(v.var_type, VarType::Binary))
        .collect();

    for c in &model.constraints {
        let poly = match try_polynomial(&model.arena, c.body) {
            Some(p) => p,
            None => continue,
        };
        if poly.max_total_degree() > 1 {
            continue;
        }
        // Collect (block_index, coeff) pairs. All factors must resolve
        // to binary scalar variables; otherwise skip the row.
        let mut row: Vec<(usize, f64)> = Vec::with_capacity(poly.monomials.len());
        let mut ok = true;
        for m in &poly.monomials {
            if m.factors.len() != 1 || m.factors[0].1 != 1 {
                ok = false;
                break;
            }
            let leaf = m.factors[0].0;
            let block = match model.arena.get(leaf) {
                ExprNode::Variable { index, size, .. } if *size == 1 => *index,
                _ => {
                    ok = false;
                    break;
                }
            };
            if !is_binary.get(block).copied().unwrap_or(false) {
                ok = false;
                break;
            }
            if m.coeff.abs() > 1e-15 {
                row.push((block, m.coeff));
            }
        }
        if !ok || row.len() < 2 {
            continue;
        }
        stats.linear_rows_scanned += 1;

        // Sort by block index so output is canonical.
        row.sort_by_key(|(b, _)| *b);

        // Compute the activity-extremum baseline: each binary at the
        // value that *helps* the constraint (minimises LHS for Le,
        // maximises for Ge). We then test pairs by re-flipping two
        // binaries to 1 and checking whether the constraint still
        // holds.
        scan_pairs(c.sense, c.rhs, poly.constant, &row, &mut edge_set);
    }

    let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    edges.sort_unstable();
    stats.edges_found = edges.len();
    (CliqueSet { edges }, stats)
}

/// Add every pair of `(b_i, b_j)` from `row` that conflicts under
/// constraint `sense / rhs` with constant offset `c0` to `edges`.
///
/// `row` is sorted ascending by block index.
fn scan_pairs(
    sense: ConstraintSense,
    rhs: f64,
    c0: f64,
    row: &[(usize, f64)],
    edges: &mut BTreeSet<(usize, usize)>,
) {
    // For each binary k, the "helping" assignment minimises (Le case)
    // or maximises (Ge case) the contribution `a_k * b_k`. With
    // `b_k ∈ {0, 1}`:
    //   Le helping: 0 if a_k ≥ 0, 1 if a_k < 0.
    //   Ge helping: 1 if a_k ≥ 0, 0 if a_k < 0.
    // Eq sense triggers both checks.
    let test_le = matches!(sense, ConstraintSense::Le | ConstraintSense::Eq);
    let test_ge = matches!(sense, ConstraintSense::Ge | ConstraintSense::Eq);

    if test_le {
        // Baseline LHS at "all helping for Le" assignment.
        let baseline_le: f64 = c0 + row.iter().map(|(_, a)| a.min(0.0)).sum::<f64>();
        for i in 0..row.len() {
            for j in (i + 1)..row.len() {
                let (bi, ai) = row[i];
                let (bj, aj) = row[j];
                // Switching i and j to 1 contributes (a_i - a_i.min(0)) + (a_j - a_j.min(0)).
                let delta_i = ai - ai.min(0.0); // = max(ai, 0)
                let delta_j = aj - aj.min(0.0);
                let test = baseline_le + delta_i + delta_j;
                if test > rhs + 1e-9 {
                    let (lo, hi) = if bi < bj { (bi, bj) } else { (bj, bi) };
                    if lo != hi {
                        edges.insert((lo, hi));
                    }
                }
            }
        }
    }
    if test_ge {
        // Baseline LHS at "all helping for Ge" assignment.
        let baseline_ge: f64 = c0 + row.iter().map(|(_, a)| a.max(0.0)).sum::<f64>();
        for i in 0..row.len() {
            for j in (i + 1)..row.len() {
                let (bi, ai) = row[i];
                let (bj, aj) = row[j];
                // Switching i and j to 0 changes contribution by
                // -(a_k - a_k.min(0)) = -max(a_k, 0).
                let delta_i = -ai.max(0.0);
                let delta_j = -aj.max(0.0);
                let test = baseline_ge + delta_i + delta_j;
                if test < rhs - 1e-9 {
                    let (lo, hi) = if bi < bj { (bi, bj) } else { (bj, bi) };
                    if lo != hi {
                        edges.insert((lo, hi));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr,
        ObjectiveSense, VarInfo, VarType,
    };

    fn binary_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.into(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo_bin(name: &str, offset: usize) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Binary,
            offset,
            size: 1,
            shape: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        }
    }

    fn vinfo_cont(name: &str, offset: usize) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Continuous,
            offset,
            size: 1,
            shape: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        }
    }

    fn lin(arena: &mut ExprArena, c: f64, var: ExprId) -> ExprId {
        let cn = arena.add(ExprNode::Constant(c));
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cn,
            right: var,
        })
    }

    fn add(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: a,
            right: b,
        })
    }

    /// Set-packing: `b0 + b1 + b2 ≤ 1` ⇒ all 3 pairs conflict.
    #[test]
    fn set_packing_three_edges() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let b1 = binary_var(&mut arena, "b1", 1);
        let b2 = binary_var(&mut arena, "b2", 2);
        let body = {
            let s01 = add(&mut arena, b0, b1);
            add(&mut arena, s01, b2)
        };
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_bin("b1", 1), vinfo_bin("b2", 2)],
            n_vars: 3,
        };
        let (c, s) = extract_cliques(&model);
        assert_eq!(s.edges_found, 3);
        assert_eq!(c.edges, vec![(0, 1), (0, 2), (1, 2)]);
    }

    /// Coefficient pair: `2 b0 + 2 b1 ≤ 3` forbids both = 1
    /// (4 > 3) but allows either alone.
    #[test]
    fn coeff_pair_forbidden() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let b1 = binary_var(&mut arena, "b1", 1);
        let body = {
            let a = lin(&mut arena, 2.0, b0);
            let b = lin(&mut arena, 2.0, b1);
            add(&mut arena, a, b)
        };
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 3.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_bin("b1", 1)],
            n_vars: 2,
        };
        let (c, _) = extract_cliques(&model);
        assert_eq!(c.edges, vec![(0, 1)]);
    }

    /// Loose constraint: `b0 + b1 ≤ 5` ⇒ no edges.
    #[test]
    fn loose_constraint_no_edges() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let b1 = binary_var(&mut arena, "b1", 1);
        let body = add(&mut arena, b0, b1);
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_bin("b1", 1)],
            n_vars: 2,
        };
        let (c, _) = extract_cliques(&model);
        assert!(c.edges.is_empty());
    }

    /// Mixed continuous/binary row is skipped (v0 scope).
    #[test]
    fn mixed_row_skipped() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let body = add(&mut arena, b0, x);
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_cont("x", 1)],
            n_vars: 2,
        };
        let (c, s) = extract_cliques(&model);
        assert!(c.edges.is_empty());
        assert_eq!(s.linear_rows_scanned, 0);
    }

    /// Ge sense: `b0 + b1 + b2 ≥ 2` ⇒ at most one zero ⇒ each pair
    /// of zeros conflicts. With baseline at all=1 (helping), test
    /// "two zeros" subtracts 2 from baseline 3, giving 1 < 2. So
    /// every pair of zeros violates ⇒ 3 edges.
    #[test]
    fn ge_sense_zero_pair() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let b1 = binary_var(&mut arena, "b1", 1);
        let b2 = binary_var(&mut arena, "b2", 2);
        let body = {
            let s01 = add(&mut arena, b0, b1);
            add(&mut arena, s01, b2)
        };
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Ge,
                rhs: 2.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_bin("b1", 1), vinfo_bin("b2", 2)],
            n_vars: 3,
        };
        let (c, _) = extract_cliques(&model);
        assert_eq!(c.edges, vec![(0, 1), (0, 2), (1, 2)]);
    }
}
