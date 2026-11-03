//! PyO3 bindings for the compact CDCL SAT solver (`discopt_core::sat`).
//!
//! Literals are DIMACS-style nonzero integers: `+v` / `-v` for variable `v` in
//! `1..=n_vars`. Used by the MILP IIS engine to get cheap unsatisfiable cores
//! over the rows of pure-binary clause models.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use discopt_core::sat::{mk_lit, Lit, SolveResult, Solver};

/// Incremental CDCL SAT solver with assumptions and final-conflict cores.
#[pyclass]
pub struct PySatSolver {
    inner: Solver,
}

fn to_lit(n: usize, x: i64) -> PyResult<Lit> {
    let v = x.unsigned_abs() as usize;
    if x == 0 || v > n {
        return Err(PyValueError::new_err(format!(
            "literal {x} is not in +-1..={n}"
        )));
    }
    Ok(mk_lit(v - 1, x < 0))
}

fn from_lit(l: Lit) -> i64 {
    let v = (l >> 1) as i64 + 1;
    if l & 1 == 1 {
        -v
    } else {
        v
    }
}

#[pymethods]
impl PySatSolver {
    /// A solver over variables `1..=n_vars` with no clauses.
    #[new]
    fn new(n_vars: usize) -> Self {
        PySatSolver {
            inner: Solver::new(n_vars),
        }
    }

    /// Add a clause; returns False once the clause set is unsatisfiable outright.
    fn add_clause(&mut self, lits: Vec<i64>) -> PyResult<bool> {
        let n = self.inner.n_vars();
        let c: Vec<Lit> = lits
            .into_iter()
            .map(|x| to_lit(n, x))
            .collect::<PyResult<_>>()?;
        Ok(self.inner.add_clause(&c))
    }

    /// Solve under `assumptions`: True (SAT), False (UNSAT) or None when
    /// `conflict_limit` conflicts of this call ran out first.
    #[pyo3(signature = (assumptions, conflict_limit=None))]
    fn solve(
        &mut self,
        assumptions: Vec<i64>,
        conflict_limit: Option<u64>,
    ) -> PyResult<Option<bool>> {
        let n = self.inner.n_vars();
        let a: Vec<Lit> = assumptions
            .into_iter()
            .map(|x| to_lit(n, x))
            .collect::<PyResult<_>>()?;
        Ok(match self.inner.solve(&a, conflict_limit) {
            SolveResult::Sat => Some(true),
            SolveResult::Unsat => Some(false),
            SolveResult::Unknown => None,
        })
    }

    /// After an UNSAT `solve`: the failed assumptions (as passed) whose
    /// conjunction is unsatisfiable. Empty if the clauses alone are.
    fn core(&self) -> Vec<i64> {
        self.inner.core().iter().map(|&l| from_lit(l)).collect()
    }

    /// After a SAT `solve`: the value of variables 1..=n_vars.
    fn model(&self) -> Vec<bool> {
        self.inner.model().to_vec()
    }

    /// Cumulative (conflicts, decisions, propagations).
    fn stats(&self) -> (u64, u64, u64) {
        let s = self.inner.stats();
        (s.conflicts, s.decisions, s.propagations)
    }
}
