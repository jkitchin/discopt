//! AMP helpers backed by the Rust expression representation.
//!
//! These routines keep the high-level AMP algorithm in Python while moving
//! repeated expression-tree walks into Rust.

use std::collections::{BTreeMap, BTreeSet};

use crate::expr::{index_spec_collect_flat, BinOp, ExprId, ExprNode, ModelRepr, UnOp};

/// Classified nonlinear terms used by AMP partition selection and relaxation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AmpNonlinearTerms {
    /// Distinct bilinear terms as sorted pairs of flat variable indices.
    pub bilinear: Vec<(usize, usize)>,
    /// Distinct trilinear terms as sorted triples of flat variable indices.
    pub trilinear: Vec<(usize, usize, usize)>,
    /// Distinct higher-order multilinear terms as sorted flat variable indices.
    pub multilinear: Vec<Vec<usize>>,
    /// Distinct monomial terms as `(flat_variable_index, exponent)`.
    pub monomial: Vec<(usize, usize)>,
    /// Number of general nonlinear expression nodes encountered.
    pub general_nl_count: usize,
    /// Variable-to-product-term incidence map.
    ///
    /// Product term ids are assigned in discovery order across bilinear,
    /// trilinear, and higher-order multilinear terms.
    pub term_incidence: BTreeMap<usize, BTreeSet<usize>>,
    /// Sorted variables that appear in bilinear or multilinear product terms.
    pub partition_candidates: Vec<usize>,
}

/// Classify nonlinear terms in a model using the Rust expression arena.
pub fn classify_nonlinear_terms(model: &ModelRepr) -> AmpNonlinearTerms {
    let mut classifier = TermClassifier::new(model);
    classifier.classify_node(model.objective);
    for constraint in &model.constraints {
        classifier.classify_node(constraint.body);
    }
    classifier.finish()
}

struct TermClassifier<'a> {
    model: &'a ModelRepr,
    result: AmpNonlinearTerms,
    seen_bilinear: BTreeSet<(usize, usize)>,
    seen_trilinear: BTreeSet<(usize, usize, usize)>,
    seen_multilinear: BTreeSet<Vec<usize>>,
    seen_monomial: BTreeSet<(usize, usize)>,
}

impl<'a> TermClassifier<'a> {
    fn new(model: &'a ModelRepr) -> Self {
        Self {
            model,
            result: AmpNonlinearTerms::default(),
            seen_bilinear: BTreeSet::new(),
            seen_trilinear: BTreeSet::new(),
            seen_multilinear: BTreeSet::new(),
            seen_monomial: BTreeSet::new(),
        }
    }

    fn finish(mut self) -> AmpNonlinearTerms {
        let mut candidates = BTreeSet::new();
        for (i, j) in &self.result.bilinear {
            candidates.insert(*i);
            candidates.insert(*j);
        }
        for (i, j, k) in &self.result.trilinear {
            candidates.insert(*i);
            candidates.insert(*j);
            candidates.insert(*k);
        }
        for term in &self.result.multilinear {
            candidates.extend(term.iter().copied());
        }
        self.result.partition_candidates = candidates.into_iter().collect();
        self.result
    }

    fn next_product_term_idx(&self) -> usize {
        self.result.bilinear.len() + self.result.trilinear.len() + self.result.multilinear.len()
    }

    fn record_bilinear(&mut self, i: usize, j: usize) {
        let key = if i <= j { (i, j) } else { (j, i) };
        if self.seen_bilinear.insert(key) {
            let term_idx = self.next_product_term_idx();
            self.result.bilinear.push(key);
            for var in [key.0, key.1] {
                self.result
                    .term_incidence
                    .entry(var)
                    .or_default()
                    .insert(term_idx);
            }
        }
    }

    fn record_trilinear(&mut self, i: usize, j: usize, k: usize) {
        let mut values = [i, j, k];
        values.sort_unstable();
        let key = (values[0], values[1], values[2]);
        if self.seen_trilinear.insert(key) {
            let term_idx = self.next_product_term_idx();
            self.result.trilinear.push(key);
            for var in [key.0, key.1, key.2] {
                self.result
                    .term_incidence
                    .entry(var)
                    .or_default()
                    .insert(term_idx);
            }
        }
    }

    fn record_multilinear(&mut self, indices: &[usize]) {
        let mut key = indices.to_vec();
        key.sort_unstable();
        if key.len() < 4 {
            return;
        }
        if self.seen_multilinear.insert(key.clone()) {
            let term_idx = self.next_product_term_idx();
            self.result.multilinear.push(key.clone());
            for var in key {
                self.result
                    .term_incidence
                    .entry(var)
                    .or_default()
                    .insert(term_idx);
            }
        }
    }

    fn record_monomial(&mut self, var_idx: usize, exp: usize) {
        let key = (var_idx, exp);
        if self.seen_monomial.insert(key) {
            self.result.monomial.push(key);
        }
    }

    fn classify_node(&mut self, id: ExprId) {
        match self.model.arena.get(id) {
            ExprNode::Constant(_) | ExprNode::ConstantArray(_, _) | ExprNode::Parameter { .. } => {}
            ExprNode::Variable { .. } => {}
            ExprNode::Index { base, .. } => {
                let base = *base;
                if !matches!(self.model.arena.get(base), ExprNode::Variable { .. }) {
                    self.classify_node(base);
                }
            }
            ExprNode::BinaryOp { op, left, right } => match op {
                BinOp::Pow => self.classify_power(*left, *right),
                BinOp::Mul => self.classify_product(id, *left, *right),
                BinOp::Add | BinOp::Sub => {
                    self.classify_node(*left);
                    self.classify_node(*right);
                }
                BinOp::Div => {
                    if self.constant_value(*right).is_some() {
                        self.classify_node(*left);
                    } else {
                        self.result.general_nl_count += 1;
                    }
                }
            },
            ExprNode::UnaryOp { op, operand } => match op {
                UnOp::Neg => self.classify_node(*operand),
                UnOp::Abs => self.result.general_nl_count += 1,
            },
            ExprNode::FunctionCall { .. } => {
                self.result.general_nl_count += 1;
                let args_len = match self.model.arena.get(id) {
                    ExprNode::FunctionCall { args, .. } => args.len(),
                    _ => unreachable!(),
                };
                for arg_idx in 0..args_len {
                    let arg = match self.model.arena.get(id) {
                        ExprNode::FunctionCall { args, .. } => args[arg_idx],
                        _ => unreachable!(),
                    };
                    self.classify_node(arg);
                }
            }
            ExprNode::Sum { operand, .. } => self.classify_node(*operand),
            ExprNode::SumOver { .. } => {
                let terms_len = match self.model.arena.get(id) {
                    ExprNode::SumOver { terms } => terms.len(),
                    _ => unreachable!(),
                };
                for term_idx in 0..terms_len {
                    let term = match self.model.arena.get(id) {
                        ExprNode::SumOver { terms } => terms[term_idx],
                        _ => unreachable!(),
                    };
                    self.classify_node(term);
                }
            }
            ExprNode::MatMul { left, right } => {
                self.classify_node(*left);
                self.classify_node(*right);
            }
        }
    }

    fn classify_power(&mut self, left: ExprId, right: ExprId) {
        if let Some(flat) = self.flat_index(left) {
            if let Some(exp_val) = self.constant_value(right) {
                if exp_val.abs() <= 1e-12 {
                    return;
                }
                if exp_val >= 2.0
                    && exp_val.is_finite()
                    && (exp_val - exp_val.round()).abs() <= 1e-12
                {
                    self.record_monomial(flat, exp_val.round() as usize);
                    return;
                }
                if (exp_val - 1.0).abs() > 1e-12 {
                    self.result.general_nl_count += 1;
                    return;
                }
            }
        }

        self.classify_node(left);
        self.classify_node(right);
    }

    fn classify_product(&mut self, id: ExprId, left: ExprId, right: ExprId) {
        if let Some(factors) = self.collect_product_factors(id) {
            let mut unique_vars = Vec::new();
            for factor in &factors {
                if !unique_vars.contains(factor) {
                    unique_vars.push(*factor);
                }
            }

            if unique_vars.len() == 1 {
                let exp = factors.iter().filter(|idx| **idx == unique_vars[0]).count();
                self.record_monomial(unique_vars[0], exp);
                return;
            }

            if unique_vars
                .iter()
                .any(|var| factors.iter().filter(|idx| *idx == var).count() >= 2)
            {
                self.result.general_nl_count += 1;
                return;
            }

            match unique_vars.len() {
                2 => self.record_bilinear(unique_vars[0], unique_vars[1]),
                3 => self.record_trilinear(unique_vars[0], unique_vars[1], unique_vars[2]),
                _ => self.record_multilinear(&unique_vars),
            }
            return;
        }

        self.classify_node(left);
        self.classify_node(right);
    }

    fn collect_product_factors(&self, id: ExprId) -> Option<Vec<usize>> {
        let mut indices = Vec::new();
        if self.collect_product_factors_inner(id, &mut indices) && indices.len() >= 2 {
            Some(indices)
        } else {
            None
        }
    }

    fn collect_product_factors_inner(&self, id: ExprId, out: &mut Vec<usize>) -> bool {
        match self.model.arena.get(id) {
            ExprNode::BinaryOp {
                op: BinOp::Mul,
                left,
                right,
            } => {
                self.collect_product_factors_inner(*left, out)
                    && self.collect_product_factors_inner(*right, out)
            }
            ExprNode::Constant(_) => true,
            _ => {
                if let Some(flat) = self.flat_index(id) {
                    out.push(flat);
                    true
                } else {
                    false
                }
            }
        }
    }

    fn flat_index(&self, id: ExprId) -> Option<usize> {
        match self.model.arena.get(id) {
            ExprNode::Variable { index, size, .. } => {
                if *size == 1 {
                    self.model.variables.get(*index).map(|var| var.offset)
                } else {
                    None
                }
            }
            ExprNode::Index { base, index } => {
                if let ExprNode::Variable {
                    index: var_block_idx,
                    ..
                } = self.model.arena.get(*base)
                {
                    let var = self.model.variables.get(*var_block_idx)?;
                    let selected = index_spec_collect_flat(index, &var.shape);
                    match selected.as_slice() {
                        [local] => Some(var.offset + *local),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn constant_value(&self, id: ExprId) -> Option<f64> {
        match self.model.arena.get(id) {
            ExprNode::Constant(value) => Some(*value),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        ConstraintRepr, ConstraintSense, ExprArena, ExprNode, IndexElem, IndexSpec, MathFunc,
        ModelRepr, ObjectiveSense, VarInfo, VarType,
    };

    fn bilinear_model() -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".to_string(),
            index: 0,
            size: 3,
            shape: vec![3],
        });
        let x0 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(0),
        });
        let x1 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(1),
        });
        let x2 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(2),
        });
        let prod01 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0,
            right: x1,
        });
        let prod02 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0,
            right: x2,
        });
        let objective = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: prod01,
            right: prod02,
        });

        ModelRepr {
            arena,
            objective,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body: x0,
                sense: ConstraintSense::Ge,
                rhs: 0.0,
                name: None,
            }],
            variables: vec![VarInfo {
                name: "x".to_string(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 3,
                shape: vec![3],
                lb: vec![0.0; 3],
                ub: vec![10.0; 3],
            }],
            n_vars: 3,
        }
    }

    #[test]
    fn classifies_bilinear_terms_and_incidence() {
        let terms = classify_nonlinear_terms(&bilinear_model());

        assert_eq!(terms.bilinear, vec![(0, 1), (0, 2)]);
        assert_eq!(terms.partition_candidates, vec![0, 1, 2]);
        assert_eq!(terms.term_incidence.get(&0).unwrap().len(), 2);
        assert_eq!(terms.term_incidence.get(&1).unwrap().len(), 1);
        assert_eq!(terms.term_incidence.get(&2).unwrap().len(), 1);
    }

    #[test]
    fn classifies_multidimensional_indices_with_row_major_offsets() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".to_string(),
            index: 0,
            size: 8,
            shape: vec![2, 4],
        });
        let x12 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Tuple(vec![1, 2]),
        });
        let x13 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Tuple(vec![1, 3]),
        });
        let x00 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Multi(vec![IndexElem::Scalar(0), IndexElem::Scalar(0)]),
        });
        let x01 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Multi(vec![IndexElem::Scalar(0), IndexElem::Scalar(1)]),
        });
        let prod_last_row = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x12,
            right: x13,
        });
        let prod_first_row = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x00,
            right: x01,
        });
        let objective = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: prod_last_row,
            right: prod_first_row,
        });

        let model = ModelRepr {
            arena,
            objective,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".to_string(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 8,
                shape: vec![2, 4],
                lb: vec![0.0; 8],
                ub: vec![10.0; 8],
            }],
            n_vars: 8,
        };

        let terms = classify_nonlinear_terms(&model);

        assert_eq!(terms.bilinear, vec![(6, 7), (0, 1)]);
        assert_eq!(terms.partition_candidates, vec![0, 1, 6, 7]);
        assert_eq!(terms.term_incidence.get(&6), Some(&BTreeSet::from([0])));
        assert_eq!(terms.term_incidence.get(&7), Some(&BTreeSet::from([0])));
        assert_eq!(terms.term_incidence.get(&0), Some(&BTreeSet::from([1])));
        assert_eq!(terms.term_incidence.get(&1), Some(&BTreeSet::from([1])));
    }

    #[test]
    fn classifies_trilinear_multilinear_and_monomial_terms() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".to_string(),
            index: 0,
            size: 5,
            shape: vec![5],
        });
        let refs: Vec<_> = (0..5)
            .map(|i| {
                arena.add(ExprNode::Index {
                    base: x,
                    index: IndexSpec::Scalar(i),
                })
            })
            .collect();
        let tri01 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: refs[0],
            right: refs[1],
        });
        let tri = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: tri01,
            right: refs[2],
        });
        let multi01 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: refs[0],
            right: refs[1],
        });
        let multi012 = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: multi01,
            right: refs[2],
        });
        let multi = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: multi012,
            right: refs[3],
        });
        let pow_exp = arena.add(ExprNode::Constant(3.0));
        let monomial = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: refs[4],
            right: pow_exp,
        });
        let sum = arena.add(ExprNode::SumOver {
            terms: vec![tri, multi, monomial],
        });

        let model = ModelRepr {
            arena,
            objective: sum,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".to_string(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 5,
                shape: vec![5],
                lb: vec![0.0; 5],
                ub: vec![10.0; 5],
            }],
            n_vars: 5,
        };

        let terms = classify_nonlinear_terms(&model);

        assert_eq!(terms.trilinear, vec![(0, 1, 2)]);
        assert_eq!(terms.multilinear, vec![vec![0, 1, 2, 3]]);
        assert_eq!(terms.monomial, vec![(4, 3)]);
        assert_eq!(terms.partition_candidates, vec![0, 1, 2, 3]);
    }

    #[test]
    fn ignores_zero_power_as_constant() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".to_string(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let zero = arena.add(ExprNode::Constant(0.0));
        let objective = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x,
            right: zero,
        });
        let model = ModelRepr {
            arena,
            objective,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".to_string(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 1,
                shape: vec![],
                lb: vec![0.0],
                ub: vec![10.0],
            }],
            n_vars: 1,
        };

        let terms = classify_nonlinear_terms(&model);

        assert!(terms.monomial.is_empty());
        assert_eq!(terms.general_nl_count, 0);
        assert!(terms.partition_candidates.is_empty());
    }

    #[test]
    fn counts_general_nonlinear_div_abs_and_function_call() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".to_string(),
            index: 0,
            size: 2,
            shape: vec![2],
        });
        let x0 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(0),
        });
        let x1 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(1),
        });
        let div = arena.add(ExprNode::BinaryOp {
            op: BinOp::Div,
            left: x0,
            right: x1,
        });
        let abs = arena.add(ExprNode::UnaryOp {
            op: UnOp::Abs,
            operand: x0,
        });
        let sin = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Sin,
            args: vec![x1],
        });
        let objective = arena.add(ExprNode::SumOver {
            terms: vec![div, abs, sin],
        });
        let model = ModelRepr {
            arena,
            objective,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".to_string(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 2,
                shape: vec![2],
                lb: vec![0.0; 2],
                ub: vec![10.0; 2],
            }],
            n_vars: 2,
        };

        let terms = classify_nonlinear_terms(&model);

        assert_eq!(terms.general_nl_count, 3);
    }

    #[test]
    fn repeated_factor_product_is_single_general_nonlinearity() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".to_string(),
            index: 0,
            size: 2,
            shape: vec![2],
        });
        let x0 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(0),
        });
        let x1 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(1),
        });
        let x0_squared = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0,
            right: x0,
        });
        let objective = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0_squared,
            right: x1,
        });
        let model = ModelRepr {
            arena,
            objective,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![VarInfo {
                name: "x".to_string(),
                var_type: VarType::Continuous,
                offset: 0,
                size: 2,
                shape: vec![2],
                lb: vec![0.0; 2],
                ub: vec![10.0; 2],
            }],
            n_vars: 2,
        };

        let terms = classify_nonlinear_terms(&model);

        assert_eq!(terms.general_nl_count, 1);
        assert!(terms.monomial.is_empty());
        assert!(terms.bilinear.is_empty());
    }
}
