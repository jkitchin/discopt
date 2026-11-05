//! A persistent MILP held in Rust and re-solved after cheap edits (highs-parity P3.1).
//!
//! The MILP IIS engine (`_iis_milp._Engine`) builds one model and then toggles
//! elements -- a row's bounds, a column's bounds, a column's integrality -- solving
//! after each toggle. On HiGHS that is one `Highs` object; this is the in-house
//! counterpart. It owns the equality standard form the driver consumes
//! (`A x = b`, `l <= x <= u`, structural columns `[0, n_struct)` followed by the
//! logicals, see `python/discopt/_relax/std_form.py`) and edits it in place.
//!
//! **Bound-neutral by construction.** [`MilpHandle::solve`] calls
//! [`solve_milp_node_hooked`] -- the function `solve_milp_csc_py` runs -- with no
//! hooks, on the handle's current arrays. A handle after any sequence of edits is
//! therefore exactly a fresh solve of the arrays those edits produced, and the
//! tests below check that bit for bit (status, objective, bound, nodes, `x`).
//! Nothing here changes the driver, the simplex or the tree: no warm start is
//! carried between solves (each solve is a fresh driver run), so a handle buys a
//! caller one marshaling and in-place edits, not a different search.
//!
//! [`MilpHandle::presolve_only`] runs the driver's own root preparation --
//! #2107 entry relaxation, #1287 integer rounding and root presolve -- through
//! [`milp_presolve_verdict`] and reports whether it proves the box empty, which is
//! the verdict a `presolve = true` solve returns as `Infeasible` with zero nodes.

use super::milp_driver::{milp_presolve_verdict, solve_milp_node_hooked, MilpOptions, MilpResult};
use crate::lp::simplex::sparse::SparseCols;

/// One MILP in equality standard form, editable in place. See the module docs.
#[derive(Clone)]
pub struct MilpHandle {
    csc: SparseCols,
    m: usize,
    n: usize,
    c: Vec<f64>,
    b: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    n_struct: usize,
    /// Integrality of the structural columns (length `n_struct`).
    is_int: Vec<bool>,
    obj_const: f64,
    /// Seed for the next solves' incumbent ([`MilpOptions::initial_incumbent`]);
    /// validated by the driver, never trusted.
    incumbent: Option<Vec<f64>>,
}

fn check_box(lo: f64, hi: f64, what: &str) -> Result<(), String> {
    if lo.is_nan() || hi.is_nan() {
        return Err(format!("{what}: NaN bound ({lo}, {hi})"));
    }
    Ok(())
}

impl MilpHandle {
    /// Build a handle. `integer_cols` must index structural columns. Shapes are
    /// checked here, since the driver only `debug_assert`s them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        csc: SparseCols,
        m: usize,
        n: usize,
        c: Vec<f64>,
        b: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        n_struct: usize,
        integer_cols: &[usize],
        obj_const: f64,
    ) -> Result<Self, String> {
        if c.len() != n || l.len() != n || u.len() != n || b.len() != m || n_struct > n {
            return Err(format!(
                "MilpHandle: shape mismatch (m={m}, n={n}, n_struct={n_struct}, c={}, l={}, \
                 u={}, b={})",
                c.len(),
                l.len(),
                u.len(),
                b.len()
            ));
        }
        let (_, row_idx, _) = csc.raw();
        if csc.ncols() != n || row_idx.iter().any(|&r| r >= m) {
            return Err(format!(
                "MilpHandle: CSC matrix is not {m} x {n} (has {} columns, or a row index >= m)",
                csc.ncols()
            ));
        }
        for j in 0..n {
            check_box(l[j], u[j], &format!("MilpHandle: column {j}"))?;
        }
        let mut is_int = vec![false; n_struct];
        for &j in integer_cols {
            if j >= n_struct {
                return Err(format!(
                    "MilpHandle: integer column {j} is not structural (n_struct={n_struct})"
                ));
            }
            is_int[j] = true;
        }
        Ok(Self {
            csc,
            m,
            n,
            c,
            b,
            l,
            u,
            n_struct,
            is_int,
            obj_const,
            incumbent: None,
        })
    }

    /// Set column `j`'s box (structural or logical).
    pub fn change_col_bounds(&mut self, j: usize, lo: f64, hi: f64) -> Result<(), String> {
        if j >= self.n {
            return Err(format!(
                "change_col_bounds: column {j} out of range (n={})",
                self.n
            ));
        }
        check_box(lo, hi, &format!("change_col_bounds: column {j}"))?;
        self.l[j] = lo;
        self.u[j] = hi;
        Ok(())
    }

    /// Make structural column `j` integer or continuous.
    pub fn change_integrality(&mut self, j: usize, is_int: bool) -> Result<(), String> {
        if j >= self.n_struct {
            return Err(format!(
                "change_integrality: column {j} is not structural (n_struct={})",
                self.n_struct
            ));
        }
        self.is_int[j] = is_int;
        Ok(())
    }

    /// Replace every column's box at once (the IIS engine re-configures a whole
    /// probe; one call instead of `n` keeps the marshaling O(n) in Rust).
    pub fn set_col_bounds_all(&mut self, l: &[f64], u: &[f64]) -> Result<(), String> {
        if l.len() != self.n || u.len() != self.n {
            return Err(format!(
                "set_col_bounds_all: got {} / {} bounds for n={}",
                l.len(),
                u.len(),
                self.n
            ));
        }
        for j in 0..self.n {
            check_box(l[j], u[j], &format!("set_col_bounds_all: column {j}"))?;
        }
        self.l.copy_from_slice(l);
        self.u.copy_from_slice(u);
        Ok(())
    }

    /// Replace the structural columns' integrality at once (length `n_struct`).
    pub fn set_integrality_all(&mut self, is_int: &[bool]) -> Result<(), String> {
        if is_int.len() != self.n_struct {
            return Err(format!(
                "set_integrality_all: got {} flags for n_struct={}",
                is_int.len(),
                self.n_struct
            ));
        }
        self.is_int.copy_from_slice(is_int);
        Ok(())
    }

    /// Seed (or clear) the incumbent the next solves start from. The driver
    /// validates it against the bounds, integrality and rows of that solve and
    /// ignores it otherwise ([`MilpOptions::initial_incumbent`]).
    pub fn set_solution(&mut self, x: Option<Vec<f64>>) -> Result<(), String> {
        if let Some(v) = x.as_ref() {
            if v.len() != self.n_struct {
                return Err(format!(
                    "set_solution: point has {} entries, n_struct={}",
                    v.len(),
                    self.n_struct
                ));
            }
        }
        self.incumbent = x;
        Ok(())
    }

    /// Current column boxes.
    pub fn col_bounds(&self) -> (&[f64], &[f64]) {
        (&self.l, &self.u)
    }

    /// Current integer columns, ascending.
    pub fn integer_cols(&self) -> Vec<usize> {
        (0..self.n_struct).filter(|&j| self.is_int[j]).collect()
    }

    /// `base` with this handle's structure, integrality and seed filled in.
    pub fn options(&self, base: &MilpOptions) -> MilpOptions {
        let mut o = base.clone();
        o.n_struct = self.n_struct;
        o.integer_cols = self.integer_cols();
        o.initial_incumbent = self.incumbent.clone();
        o
    }

    /// Solve the current problem: [`solve_milp_node_hooked`] with no hooks, on the
    /// handle's arrays, under `base`'s options (see [`Self::options`]).
    pub fn solve(&self, base: &MilpOptions) -> MilpResult {
        let opts = self.options(base);
        solve_milp_node_hooked(
            self.csc.clone(),
            self.m,
            self.n,
            &self.c,
            &self.l,
            &self.u,
            &self.b,
            self.obj_const,
            &opts,
            None,
            None,
            None,
        )
    }

    /// Whether the driver's root preparation and presolve prove the current box
    /// empty ([`milp_presolve_verdict`]). `false` decides nothing.
    pub fn presolve_only(&self, base: &MilpOptions) -> bool {
        let opts = self.options(base);
        milp_presolve_verdict(
            self.csc.clone(),
            self.m,
            self.n,
            &self.c,
            &self.l,
            &self.u,
            &self.b,
            &opts,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::milp_driver::sparse_milp_diff::{panel, Case};
    use super::super::milp_driver::MilpStatus;
    use super::*;

    /// The Rust LP layer's infinity sentinel (CLAUDE.md "Key Constraints").
    const INF: f64 = 1e20;

    fn fresh(
        case_csc: &SparseCols,
        m: usize,
        n: usize,
        c: &[f64],
        l: &[f64],
        u: &[f64],
        b: &[f64],
        opts: &MilpOptions,
    ) -> MilpResult {
        solve_milp_node_hooked(
            case_csc.clone(),
            m,
            n,
            c,
            l,
            u,
            b,
            0.0,
            opts,
            None,
            None,
            None,
        )
    }

    fn same(a: &MilpResult, b: &MilpResult) -> bool {
        a.status == b.status
            && a.obj.to_bits() == b.obj.to_bits()
            && a.bound.to_bits() == b.bound.to_bits()
            && a.nodes == b.nodes
            && a.lp_iters == b.lp_iters
            && a.x.len() == b.x.len()
            && a.x
                .iter()
                .zip(&b.x)
                .all(|(p, q)| p.to_bits() == q.to_bits())
    }

    fn handle_of(case: &Case) -> MilpHandle {
        MilpHandle::new(
            SparseCols::from_dense(&case.a, case.m, case.n),
            case.m,
            case.n,
            case.c.clone(),
            case.b.clone(),
            case.l.clone(),
            case.u.clone(),
            case.ns,
            &case.int_cols,
            0.0,
        )
        .unwrap()
    }

    /// A deterministic xorshift, so the toggle sequences are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, k: usize) -> usize {
            (self.next() % k as u64) as usize
        }
    }

    /// A random small MILP in equality standard form: `m` rows `A_s x + s = b`,
    /// `s >= 0` (one logical per row), structural columns boxed in `[0, 3]`, about
    /// half of them integer.
    fn random_case(
        rng: &mut Rng,
        m: usize,
        ns: usize,
    ) -> (
        Vec<f64>,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        usize,
        Vec<usize>,
    ) {
        let n = ns + m;
        let mut a = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..ns {
                if rng.below(3) != 0 {
                    a[i * n + j] = (rng.below(9) as f64) - 3.0;
                }
            }
            a[i * n + ns + i] = 1.0;
        }
        let c: Vec<f64> = (0..n)
            .map(|j| {
                if j < ns {
                    (rng.below(11) as f64) - 6.0
                } else {
                    0.0
                }
            })
            .collect();
        let b: Vec<f64> = (0..m).map(|_| (rng.below(12) as f64) + 1.0).collect();
        let l = vec![0.0; n];
        let mut u = vec![3.0; n];
        for v in u.iter_mut().skip(ns) {
            *v = INF;
        }
        let ints: Vec<usize> = (0..ns).filter(|_| rng.below(2) == 0).collect();
        (a, m, n, c, l, u, b, ns, ints)
    }

    /// P3.1 Done: a handle re-solved after each toggle equals a fresh solve of the
    /// same arrays bit for bit, on the `sparse_milp_diff` panel and on 20 random
    /// toggle sequences. Prints the comparison count and fails on zero (§6).
    #[test]
    fn handle_after_toggles_matches_a_fresh_solve_bit_for_bit() {
        let mut compared = 0usize;
        let mut statuses = std::collections::BTreeSet::new();
        let mut cases: Vec<(
            String,
            Vec<f64>,
            usize,
            usize,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            usize,
            Vec<usize>,
            MilpOptions,
        )> = panel()
            .into_iter()
            .map(|cs| {
                let o = cs.opts();
                (
                    cs.name.to_string(),
                    cs.a,
                    cs.m,
                    cs.n,
                    cs.c,
                    cs.l,
                    cs.u,
                    cs.b,
                    cs.ns,
                    cs.int_cols,
                    o,
                )
            })
            .collect();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for k in 0..20 {
            let (a, m, n, c, l, u, b, ns, ints) = random_case(&mut rng, 2 + k % 3, 3 + k % 4);
            let mut o = panel()[0].opts();
            o.n_struct = ns;
            o.integer_cols = ints.clone();
            cases.push((format!("random{k}"), a, m, n, c, l, u, b, ns, ints, o));
        }
        for (name, a, m, n, c, l, u, b, ns, ints, base) in cases {
            let csc = SparseCols::from_dense(&a, m, n);
            let mut h = MilpHandle::new(
                csc.clone(),
                m,
                n,
                c.clone(),
                b.clone(),
                l.clone(),
                u.clone(),
                ns,
                &ints,
                0.0,
            )
            .unwrap();
            let (mut lw, mut uw) = (l.clone(), u.clone());
            let mut iw = vec![false; ns];
            for &j in &ints {
                iw[j] = true;
            }
            for step in 0..6 {
                if step > 0 {
                    // One toggle: a column's box (tightened, relaxed or fixed) or a
                    // structural column's integrality.
                    let j = rng.below(n);
                    match rng.below(4) {
                        0 if j < ns => {
                            iw[j] = !iw[j];
                            h.change_integrality(j, iw[j]).unwrap();
                        }
                        1 => {
                            let v = (rng.below(4)) as f64;
                            lw[j] = v;
                            uw[j] = v.max(u[j].min(v + 1.0));
                            h.change_col_bounds(j, lw[j], uw[j]).unwrap();
                        }
                        2 => {
                            lw[j] = l[j];
                            uw[j] = u[j];
                            h.change_col_bounds(j, lw[j], uw[j]).unwrap();
                        }
                        _ => {
                            lw[j] = -INF;
                            uw[j] = INF;
                            h.change_col_bounds(j, lw[j], uw[j]).unwrap();
                        }
                    }
                }
                let mut o = base.clone();
                o.integer_cols = (0..ns).filter(|&j| iw[j]).collect();
                o.max_nodes = 2_000;
                let want = fresh(&csc, m, n, &c, &lw, &uw, &b, &o);
                let got = h.solve(&o);
                assert!(
                    same(&got, &want),
                    "{name} step {step}: handle {:?} != fresh {:?}",
                    (got.status, got.obj, got.bound, got.nodes),
                    (want.status, want.obj, want.bound, want.nodes)
                );
                // presolve_only agrees with a presolve=true solve's zero-node verdict.
                let mut op = o.clone();
                op.presolve = true;
                let pre = h.presolve_only(&op);
                let full = fresh(&csc, m, n, &c, &lw, &uw, &b, &op);
                if pre {
                    assert_eq!(full.status, MilpStatus::Infeasible, "{name} step {step}");
                    assert_eq!(full.nodes, 0, "{name} step {step}");
                }
                statuses.insert(format!("{:?}", got.status));
                compared += 1;
            }
        }
        eprintln!("milp_handle: compared={compared} statuses={statuses:?}");
        assert!(compared >= 100, "only {compared} comparisons");
        // The panel exercises more than one verdict, or the comparison is vacuous.
        assert!(statuses.len() >= 3, "statuses seen: {statuses:?}");
    }

    #[test]
    fn presolve_only_proves_an_empty_integer_box() {
        // x0 integer in [2, 5] with x0 + s = 1, s >= 0: presolve proves it empty.
        let case = &panel()[3];
        assert_eq!(case.name, "infeasible");
        let h = handle_of(case);
        assert!(h.presolve_only(&case.opts()));
        // The feasible knapsack is not refuted.
        let k = &panel()[1];
        assert!(!handle_of(k).presolve_only(&k.opts()));
    }

    #[test]
    fn bulk_setters_equal_one_edit_per_column() {
        let case = &panel()[5];
        let base = case.opts();
        let (mut a, mut b) = (handle_of(case), handle_of(case));
        let mut l = case.l.clone();
        let mut u = case.u.clone();
        l[0] = 1.0;
        u[2] = 0.0;
        a.set_col_bounds_all(&l, &u).unwrap();
        b.change_col_bounds(0, 1.0, case.u[0]).unwrap();
        b.change_col_bounds(2, case.l[2], 0.0).unwrap();
        let mut ints = vec![true; case.ns];
        ints[1] = false;
        a.set_integrality_all(&ints).unwrap();
        b.change_integrality(1, false).unwrap();
        let (ra, rb) = (a.solve(&base), b.solve(&base));
        assert_eq!(ra.status, rb.status);
        assert_eq!(ra.obj.to_bits(), rb.obj.to_bits());
        assert_eq!(ra.nodes, rb.nodes);
        assert!(a.set_col_bounds_all(&l[1..], &u).is_err());
        assert!(a.set_integrality_all(&ints[1..]).is_err());
    }

    #[test]
    fn edits_are_validated() {
        let mut h = handle_of(&panel()[1]);
        assert!(h.change_col_bounds(99, 0.0, 1.0).is_err());
        assert!(h.change_col_bounds(0, f64::NAN, 1.0).is_err());
        assert!(h.change_integrality(4, true).is_err()); // a logical column
        assert!(h.set_solution(Some(vec![0.0; 3])).is_err());
        assert!(h.set_solution(Some(vec![0.0; 4])).is_ok());
    }

    #[test]
    fn a_seed_is_validated_not_trusted() {
        // Seeding the knapsack's optimum leaves the verdict unchanged; seeding a point
        // that violates the row is ignored (the driver validates it).
        let case = &panel()[1];
        let base = case.opts();
        let mut h = handle_of(case);
        let plain = h.solve(&base);
        h.set_solution(Some(vec![1.0, 1.0, 1.0, 1.0])).unwrap(); // 5*4 = 20 > 9
        let seeded_bad = h.solve(&base);
        assert_eq!(seeded_bad.status, plain.status);
        assert_eq!(seeded_bad.obj.to_bits(), plain.obj.to_bits());
    }
}
