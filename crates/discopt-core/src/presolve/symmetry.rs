//! Symmetry detection (D4 of issue #51).
//!
//! Detects variable permutation symmetries in the model by comparing
//! structural signatures of each scalar variable's "column" — its
//! coefficient pattern across constraints and the objective, together
//! with its type and bounds.
//!
//! ## Soundness (MVP)
//!
//! This MVP restricts to **fully-polynomial** models (every constraint
//! body and the objective expressible as a polynomial). For each
//! variable `v` we extract the multiset of monomials *that touch v*
//! across each constraint and the objective; we replace `v` itself by
//! a sentinel marker in the monomial signature so that two variables
//! `v` and `w` with structurally identical neighbourhoods produce
//! identical fingerprints.
//!
//! Two variables with identical fingerprints AND identical type, lb,
//! ub generate a candidate orbit. We do NOT verify the *full*
//! permutation symmetry of the constraint system (that requires graph
//! isomorphism); we only emit orbits where the local structure is
//! sound for swap. For linear models this is exact (the column
//! signature uniquely determines participation up to permutation of
//! identical columns). For nonlinear models, the local equality is a
//! necessary but not always sufficient condition — orbits emitted
//! here are *candidates* the caller can choose to break.
//!
//! ## What this pass produces
//!
//! - A list of `Orbit { vars: Vec<usize> }` of size ≥ 2, where each
//!   `vars` lists variable indices believed to be exchangeable.
//! - Statistics on number of orbits and total variables involved.
//!
//! ## What this pass does *not* do
//!
//! - It does not add lex-ordering constraints to the model. That is
//!   the consumer's responsibility (and is unsafe to do for nonlinear
//!   orbits without further verification). The detection is reported
//!   in the [`SymmetryDelta`] alone.

use std::collections::BTreeMap;

use super::polynomial::try_polynomial;
use crate::expr::{ExprArena, ExprId, ExprNode, ModelRepr, VarType};

/// A candidate orbit: a set of variable indices believed to be
/// exchangeable under the constraint and objective structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orbit {
    /// Variable indices in this orbit (sorted, length ≥ 2).
    pub vars: Vec<usize>,
}

/// Statistics from symmetry detection.
#[derive(Debug, Clone, Default)]
pub struct SymmetryStats {
    /// Number of variables inspected as candidates.
    pub variables_examined: usize,
    /// Number of orbits found (each of size ≥ 2).
    pub orbits_found: usize,
    /// Total variables across all orbits.
    pub total_orbit_members: usize,
}

/// Run D4 symmetry detection on `model`.
///
/// Returns a list of candidate orbits and per-pass statistics. The
/// model itself is not modified.
pub fn detect_symmetries(model: &ModelRepr) -> (Vec<Orbit>, SymmetryStats) {
    let mut stats = SymmetryStats::default();

    // 1. Find every scalar variable in the arena and map its var-block
    //    index to its single arena leaf id.
    let mut leaf_of: BTreeMap<usize, ExprId> = BTreeMap::new();
    for nid in 0..model.arena.len() {
        if let ExprNode::Variable { index, size, .. } = model.arena.get(ExprId(nid)) {
            if *size == 1 && !leaf_of.contains_key(index) {
                leaf_of.insert(*index, ExprId(nid));
            }
        }
    }
    stats.variables_examined = leaf_of.len();

    // 2. Try to extract polynomials for every constraint body and
    //    the objective. If any of those is not polynomial, abort
    //    detection — we don't trust the structural signature.
    let mut constraint_polys: Vec<super::polynomial::Polynomial> =
        Vec::with_capacity(model.constraints.len());
    for c in &model.constraints {
        match try_polynomial(&model.arena, c.body) {
            Some(p) => constraint_polys.push(p),
            None => return (Vec::new(), stats),
        }
    }
    let obj_poly = try_polynomial(&model.arena, model.objective);
    if obj_poly.is_none() && !is_constant_or_singleton_var(&model.arena, model.objective) {
        return (Vec::new(), stats);
    }

    // 3. Build a fingerprint per variable and bucket.
    let mut buckets: BTreeMap<Vec<u8>, Vec<usize>> = BTreeMap::new();
    for (vidx, &leaf) in &leaf_of {
        if *vidx >= model.variables.len() {
            continue;
        }
        let vinfo = &model.variables[*vidx];
        if vinfo.size != 1 {
            continue;
        }
        let lb = vinfo.lb.first().copied().unwrap_or(f64::NEG_INFINITY);
        let ub = vinfo.ub.first().copied().unwrap_or(f64::INFINITY);
        let var_type_tag = match vinfo.var_type {
            VarType::Continuous => 0u8,
            VarType::Binary => 1,
            VarType::Integer => 2,
        };

        let mut fp: Vec<u8> = Vec::new();
        fp.push(var_type_tag);
        fp.extend_from_slice(&lb.to_bits().to_le_bytes());
        fp.extend_from_slice(&ub.to_bits().to_le_bytes());

        // Per-constraint contribution.
        for (ci, poly) in constraint_polys.iter().enumerate() {
            let sense_tag = match model.constraints[ci].sense {
                crate::expr::ConstraintSense::Le => 0u8,
                crate::expr::ConstraintSense::Ge => 1,
                crate::expr::ConstraintSense::Eq => 2,
            };
            let mut row_terms: Vec<(Vec<(u8, u32)>, u64)> = Vec::new();
            for m in &poly.monomials {
                let touches = m.factors.iter().any(|(fid, _)| *fid == leaf);
                if !touches {
                    continue;
                }
                // Build a sentinel-ised factor list: replace the
                // candidate's leaf with marker 0xFE; other variable
                // factors get marker 0x01..0xFD by their *type*
                // signature, ignoring identity. For an orbit detection
                // we want two columns to compare equal regardless of
                // which other variables they coexist with — but we
                // also want different *non-orbit* neighbours to make
                // them differ. This MVP signature uses the var-block's
                // type/bounds tuple as the neighbour token.
                let mut tokens: Vec<(u8, u32)> = Vec::new();
                for (fid, deg) in &m.factors {
                    if *fid == leaf {
                        tokens.push((0xFE, *deg));
                    } else if let ExprNode::Variable {
                        index: other_idx,
                        size: 1,
                        ..
                    } = model.arena.get(*fid)
                    {
                        let oi = *other_idx;
                        if oi < model.variables.len() && model.variables[oi].size == 1 {
                            let ot = match model.variables[oi].var_type {
                                VarType::Continuous => 1u8,
                                VarType::Binary => 2,
                                VarType::Integer => 3,
                            };
                            tokens.push((ot, *deg));
                        } else {
                            tokens.push((0xFD, *deg));
                        }
                    } else {
                        tokens.push((0xFD, *deg));
                    }
                }
                tokens.sort();
                row_terms.push((tokens, m.coeff.to_bits()));
            }
            row_terms.sort();
            fp.push(sense_tag);
            fp.extend_from_slice(&model.constraints[ci].rhs.to_bits().to_le_bytes());
            fp.extend_from_slice(&(row_terms.len() as u32).to_le_bytes());
            for (toks, cb) in &row_terms {
                fp.extend_from_slice(&(toks.len() as u32).to_le_bytes());
                for (t, d) in toks {
                    fp.push(*t);
                    fp.extend_from_slice(&d.to_le_bytes());
                }
                fp.extend_from_slice(&cb.to_le_bytes());
            }
        }

        // Objective contribution (if polynomial).
        if let Some(poly) = obj_poly.as_ref() {
            let mut row_terms: Vec<(Vec<(u8, u32)>, u64)> = Vec::new();
            for m in &poly.monomials {
                let touches = m.factors.iter().any(|(fid, _)| *fid == leaf);
                if !touches {
                    continue;
                }
                let mut tokens: Vec<(u8, u32)> = Vec::new();
                for (fid, deg) in &m.factors {
                    if *fid == leaf {
                        tokens.push((0xFE, *deg));
                    } else {
                        tokens.push((0xFD, *deg));
                    }
                }
                tokens.sort();
                row_terms.push((tokens, m.coeff.to_bits()));
            }
            row_terms.sort();
            fp.extend_from_slice(b"OBJ");
            fp.extend_from_slice(&(row_terms.len() as u32).to_le_bytes());
            for (toks, cb) in &row_terms {
                fp.extend_from_slice(&(toks.len() as u32).to_le_bytes());
                for (t, d) in toks {
                    fp.push(*t);
                    fp.extend_from_slice(&d.to_le_bytes());
                }
                fp.extend_from_slice(&cb.to_le_bytes());
            }
        }

        buckets.entry(fp).or_default().push(*vidx);
    }

    // 4. Emit orbits of size ≥ 2.
    let mut orbits: Vec<Orbit> = Vec::new();
    for (_fp, mut vars) in buckets {
        if vars.len() < 2 {
            continue;
        }
        vars.sort();
        stats.total_orbit_members += vars.len();
        orbits.push(Orbit { vars });
    }
    orbits.sort_by(|a, b| a.vars.cmp(&b.vars));
    stats.orbits_found = orbits.len();

    (orbits, stats)
}

fn is_constant_or_singleton_var(arena: &ExprArena, root: ExprId) -> bool {
    matches!(
        arena.get(root),
        ExprNode::Constant(_) | ExprNode::Variable { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprNode, ModelRepr, ObjectiveSense,
        VarInfo,
    };

    fn scalar_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.to_string(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo(name: &str, t: VarType, lb: f64, ub: f64) -> VarInfo {
        VarInfo {
            name: name.to_string(),
            var_type: t,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    #[test]
    fn detects_two_interchangeable_binaries() {
        // x1, x2 binary, identical bounds; constraint x1 + x2 <= 1
        // (a packing constraint). Objective: minimise x1 + x2.
        let mut arena = ExprArena::new();
        let x1 = scalar_var(&mut arena, "x1", 0);
        let x2 = scalar_var(&mut arena, "x2", 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x1,
            right: x2,
        });
        let model = ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![
                vinfo("x1", VarType::Binary, 0.0, 1.0),
                vinfo("x2", VarType::Binary, 0.0, 1.0),
            ],
            n_vars: 2,
        };
        let (orbits, stats) = detect_symmetries(&model);
        assert_eq!(stats.orbits_found, 1);
        assert_eq!(orbits[0].vars, vec![0, 1]);
    }

    #[test]
    fn no_orbit_when_bounds_differ() {
        // x1 ∈ [0,1], x2 ∈ [0,2] — different bounds break the symmetry.
        let mut arena = ExprArena::new();
        let x1 = scalar_var(&mut arena, "x1", 0);
        let x2 = scalar_var(&mut arena, "x2", 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x1,
            right: x2,
        });
        let model = ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![
                vinfo("x1", VarType::Continuous, 0.0, 1.0),
                vinfo("x2", VarType::Continuous, 0.0, 2.0),
            ],
            n_vars: 2,
        };
        let (orbits, stats) = detect_symmetries(&model);
        assert_eq!(stats.orbits_found, 0);
        assert!(orbits.is_empty());
    }

    #[test]
    fn no_orbit_when_coefficients_differ() {
        // x1 + 2*x2 <= 10 — different coefficients break the symmetry.
        let mut arena = ExprArena::new();
        let x1 = scalar_var(&mut arena, "x1", 0);
        let x2 = scalar_var(&mut arena, "x2", 1);
        let two = arena.add(ExprNode::Constant(2.0));
        let two_x2 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: x2,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x1,
            right: two_x2,
        });
        let model = ModelRepr {
            arena,
            objective: x1,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 10.0,
                name: None,
            }],
            variables: vec![
                vinfo("x1", VarType::Continuous, 0.0, 5.0),
                vinfo("x2", VarType::Continuous, 0.0, 5.0),
            ],
            n_vars: 2,
        };
        let (orbits, stats) = detect_symmetries(&model);
        assert_eq!(stats.orbits_found, 0);
        assert!(orbits.is_empty());
    }

    #[test]
    fn detects_orbit_of_three() {
        // x1+x2+x3 <= 2; min x1+x2+x3.
        let mut arena = ExprArena::new();
        let x1 = scalar_var(&mut arena, "x1", 0);
        let x2 = scalar_var(&mut arena, "x2", 1);
        let x3 = scalar_var(&mut arena, "x3", 2);
        let s12 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x1,
            right: x2,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: s12,
            right: x3,
        });
        let model = ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 2.0,
                name: None,
            }],
            variables: vec![
                vinfo("x1", VarType::Binary, 0.0, 1.0),
                vinfo("x2", VarType::Binary, 0.0, 1.0),
                vinfo("x3", VarType::Binary, 0.0, 1.0),
            ],
            n_vars: 3,
        };
        let (orbits, stats) = detect_symmetries(&model);
        assert_eq!(stats.orbits_found, 1);
        assert_eq!(orbits[0].vars, vec![0, 1, 2]);
        assert_eq!(stats.total_orbit_members, 3);
    }

    #[test]
    fn deterministic_orbit_ordering() {
        // Two separate orbits — confirm orbit list is sorted.
        let mut arena = ExprArena::new();
        let x1 = scalar_var(&mut arena, "x1", 0);
        let x2 = scalar_var(&mut arena, "x2", 1);
        let y1 = scalar_var(&mut arena, "y1", 2);
        let y2 = scalar_var(&mut arena, "y2", 3);
        let xs = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x1,
            right: x2,
        });
        let ys = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: y1,
            right: y2,
        });
        let two = arena.add(ExprNode::Constant(2.0));
        let two_ys = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: two,
            right: ys,
        });
        let obj = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: xs,
            right: two_ys,
        });
        let model = ModelRepr {
            arena,
            objective: obj,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: xs,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: None,
                },
                ConstraintRepr {
                    body: ys,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: None,
                },
            ],
            variables: vec![
                vinfo("x1", VarType::Binary, 0.0, 1.0),
                vinfo("x2", VarType::Binary, 0.0, 1.0),
                vinfo("y1", VarType::Binary, 0.0, 1.0),
                vinfo("y2", VarType::Binary, 0.0, 1.0),
            ],
            n_vars: 4,
        };
        let (orbits, stats) = detect_symmetries(&model);
        assert_eq!(stats.orbits_found, 2);
        // x's appear in xs constraint only; y's in ys only — so the
        // x-orbit and y-orbit are distinct and both detected.
        assert_eq!(orbits[0].vars, vec![0, 1]);
        assert_eq!(orbits[1].vars, vec![2, 3]);

        // Re-running yields identical output (determinism check).
        let (orbits2, _) = detect_symmetries(&model);
        assert_eq!(orbits, orbits2);
    }
}
