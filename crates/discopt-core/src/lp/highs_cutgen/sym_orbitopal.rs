//! Dynamic orbitopal fixing in the branching order (`DISCOPT_MILP_ORBITOPAL`).
//!
//! Derived from SCIP's `src/scip/symmetry_orbitopal.c` (SCIP Optimization Suite
//! 10.0.2, Copyright (c) 2002-2026 Zuse Institute Berlin, Apache License 2.0; see
//! `THIRD_PARTY_NOTICES.md`). This reimplements the method; it does not translate the
//! code line by line. Changes from the original:
//!
//! - The column order is fixed (SCIP's `SCIP_COLUMNORDERING_NONE`); SCIP's default
//!   reorders columns around the branched one.
//! - The rows come from the node's path in the search's own shadow tree, not from
//!   SCIP's per-node event data.
//! - Propagation reuses this crate's column-lex propagator
//!   ([`super::orbitope::Orbitope::propagate`]), restricted to the chosen rows.
//!
//! The method is due to Bendotti, Fouilhoux & Rottner (`bendotti2021orbitopal`), for
//! full orbitopes with a dynamic row order. Van Doornmalen & Hojny
//! (`vandoornmalen2023unified`, Section 4.2) cast it in the dynamic-ordering framework
//! that `DISCOPT_MILP_SYM_DYNAMIC` and `DISCOPT_MILP_SYM_ORBITAL` also use.
//!
//! # What it does
//!
//! An orbitope is a `p x k` matrix of distinct bounded integer columns whose column
//! permutations are verified automorphisms (detected by
//! `bnb::milp_symmetry::orbitopes`). At a node `β`, its rows are taken in the order
//! in which `β`'s path from the root first branched on any entry of that row. Rows
//! the path never touched are left out. Within a row, the entries stay in column
//! order. The node then propagates the chain `C_0 >=_lex C_1 >=_lex ...` over those
//! rows only.
//!
//! # Why it is sound
//!
//! - **The order.** The row-wise order `σ_β` (the rows above, then the rest) grows by
//!   appending along every path: a child's touched rows extend its parent's.
//! - **The lex maximum.** Under `σ_β`, the lexicographic maximum of an orbit under
//!   the column group is the point whose columns are sorted lex-descending over that
//!   row order.
//! - **What is propagated.** The chain over the touched rows is a consequence of that
//!   sorting, so the propagation only relaxes the full reduction.
//! - **Retention.** The framework then keeps a representative of every orbit at a
//!   leaf. The exhaustive test below checks this on random trees, alone and combined
//!   with the other dynamic methods. Its mutant reverses the row order, which breaks
//!   the append property, and must lose an orbit.
//!
//! # Combining with the other methods
//!
//! The driver hands `DISCOPT_MILP_SYM_DYNAMIC` and `DISCOPT_MILP_SYM_ORBITAL` only
//! the generators that move no orbitope entry. The two groups then act on disjoint
//! columns, as SCIP's components do.
//!
//! Every change carries `Reason::Unspecified` and is made only at a node's own
//! evaluation.

use std::sync::Arc;

use super::orbitope::{Orbitope, Orbitopes, Tightening};

/// The model's verified orbitopes, with each entry's `(orbitope, row)`.
#[derive(Debug, Clone)]
pub struct DynamicOrbitopes {
    /// The orbitopes (with their model fingerprint).
    pub orbitopes: Arc<Orbitopes>,
    row_of: Vec<Option<(usize, usize)>>,
}

impl DynamicOrbitopes {
    /// Index the entries of `orbitopes` by row.
    pub fn new(orbitopes: Arc<Orbitopes>) -> Arc<Self> {
        let mut row_of = vec![None; orbitopes.ns];
        for (o, orb) in orbitopes.list.iter().enumerate() {
            for col in &orb.cols {
                for (r, &j) in col.iter().enumerate() {
                    row_of[j] = Some((o, r));
                }
            }
        }
        Arc::new(DynamicOrbitopes { orbitopes, row_of })
    }

    /// Whether these were derived on a model with `ns` structural columns and costs
    /// `c[..ns]`.
    pub fn matches(&self, ns: usize, c: &[f64]) -> bool {
        self.orbitopes.matches(ns, c)
    }

    /// Whether structural column `j` is an orbitope entry.
    pub fn member(&self, j: usize) -> bool {
        self.orbitopes.member.get(j).copied().unwrap_or(false)
    }

    /// Propagate every orbitope over the rows `order` (the node's branched columns,
    /// first occurrences, root to node) touched, in that order. Appends tightenings to
    /// `out` (an infeasible chain appends a lower bound above its column's upper
    /// bound) and returns how many.
    pub fn propagate(
        &self,
        order: &[usize],
        col_lower: &[f64],
        col_upper: &[f64],
        feastol: f64,
        out: &mut Vec<Tightening>,
    ) -> usize {
        self.propagate_rows(order, col_lower, col_upper, feastol, out, false)
    }

    /// [`Self::propagate`]; `reverse` (the tests' negative control only) reverses
    /// each orbitope's row order.
    fn propagate_rows(
        &self,
        order: &[usize],
        col_lower: &[f64],
        col_upper: &[f64],
        feastol: f64,
        out: &mut Vec<Tightening>,
        reverse: bool,
    ) -> usize {
        let start = out.len();
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); self.orbitopes.list.len()];
        for &j in order {
            if let Some(&Some((o, r))) = self.row_of.get(j) {
                if !rows[o].contains(&r) {
                    rows[o].push(r);
                }
            }
        }
        for (o, rs) in rows.iter_mut().enumerate() {
            if reverse {
                rs.reverse();
            }
            if rs.is_empty() {
                continue;
            }
            let full = &self.orbitopes.list[o];
            let sub = Orbitope {
                cols: full
                    .cols
                    .iter()
                    .map(|c| rs.iter().map(|&r| c[r]).collect())
                    .collect(),
            };
            let before = out.len();
            sub.propagate(col_lower, col_upper, feastol, out);
            if out[before..]
                .iter()
                .any(|&(j, is_lo, v)| is_lo && v > col_upper[j] + feastol)
            {
                break;
            }
        }
        out.len() - start
    }
}

#[cfg(test)]
mod tests {
    use super::super::sym_dynamic::DynamicLex;
    use super::super::sym_orbital::OrbitalReduction;
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    fn all_points(lo: &[f64], hi: &[f64]) -> Vec<Vec<f64>> {
        let mut pts = vec![vec![]];
        for (l, h) in lo.iter().zip(hi) {
            let mut nxt = Vec::new();
            for p in &pts {
                let mut v = *l;
                while v <= *h {
                    let mut q = p.clone();
                    q.push(v);
                    nxt.push(q);
                    v += 1.0;
                }
            }
            pts = nxt;
        }
        pts
    }

    fn closure(gens: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let n = gens[0].len();
        let mut g: Vec<Vec<usize>> = vec![(0..n).collect()];
        let mut i = 0;
        while i < g.len() {
            for h in gens {
                let c: Vec<usize> = (0..n).map(|j| g[i][h[j]]).collect();
                if !g.contains(&c) {
                    g.push(c);
                }
            }
            i += 1;
        }
        g
    }

    fn apply(out: &[Tightening], lo: &mut [f64], hi: &mut [f64]) {
        for &(j, is_lo, v) in out {
            if is_lo {
                lo[j] = lo[j].max(v);
            } else {
                hi[j] = hi[j].min(v);
            }
        }
    }

    /// The column swap `C_a <-> C_b` of a `p x k` orbitope laid out column-major
    /// (`cols[j][r] = j * p + r`) inside `n` columns.
    fn swap(n: usize, p: usize, a: usize, b: usize) -> Vec<usize> {
        let mut g: Vec<usize> = (0..n).collect();
        for r in 0..p {
            g[a * p + r] = b * p + r;
            g[b * p + r] = a * p + r;
        }
        g
    }

    struct Setup<'a> {
        dor: &'a DynamicOrbitopes,
        reverse: bool,
        lex: Option<&'a DynamicLex>,
        orb: Option<&'a OrbitalReduction>,
        feas: &'a [Vec<f64>],
        root_lo: &'a [f64],
        root_hi: &'a [f64],
        broken: &'a [bool],
    }

    fn tree(
        s: &Setup,
        lo: Vec<f64>,
        hi: Vec<f64>,
        path: Vec<(usize, bool, f64)>,
        rng: &mut Rng,
        stats: &mut (usize, usize),
        kept: &mut Vec<Vec<f64>>,
    ) {
        let n = lo.len();
        let (mut lo, mut hi) = (lo, hi);
        let mut sigma: Vec<usize> = Vec::new();
        for &(j, _, _) in &path {
            if !sigma.contains(&j) {
                sigma.push(j);
            }
        }
        for _ in 0..3 {
            let mut out = Vec::new();
            let k = s
                .dor
                .propagate_rows(&sigma, &lo, &hi, 1e-6, &mut out, s.reverse);
            stats.0 += k;
            if let Some(o) = s.orb {
                o.propagate(
                    &path, s.root_lo, s.root_hi, s.broken, &lo, &hi, 1e-6, &mut out,
                );
            }
            if let Some(d) = s.lex {
                d.propagate(&sigma, &lo, &hi, 1e-6, &mut out);
            }
            apply(&out, &mut lo, &mut hi);
            if (0..n).any(|j| lo[j] > hi[j]) {
                stats.1 += 1;
                return;
            }
            if out.is_empty() {
                break;
            }
        }
        let inside: Vec<&Vec<f64>> = s
            .feas
            .iter()
            .filter(|x| (0..n).all(|j| lo[j] <= x[j] && x[j] <= hi[j]))
            .collect();
        if inside.is_empty() {
            return;
        }
        if rng.next(3) == 0 {
            for j in 0..n {
                lo[j] = inside.iter().map(|x| x[j]).fold(f64::INFINITY, f64::min);
                hi[j] = inside
                    .iter()
                    .map(|x| x[j])
                    .fold(f64::NEG_INFINITY, f64::max);
            }
        }
        let free: Vec<usize> = (0..n).filter(|&j| lo[j] < hi[j]).collect();
        if free.is_empty() {
            kept.push(lo.clone());
            return;
        }
        let j = free[rng.next(free.len() as u64) as usize];
        let v = lo[j] + rng.next((hi[j] - lo[j]) as u64) as f64;
        let (mut hd, mut lu) = (hi.clone(), lo.clone());
        hd[j] = v;
        lu[j] = v + 1.0;
        let mut pd = path.clone();
        pd.push((j, true, v));
        let mut pu = path;
        pu.push((j, false, v + 1.0));
        tree(s, lo.clone(), hd, pd, rng, stats, kept);
        tree(s, lu, hi, pu, rng, stats, kept);
    }

    /// Orbit retention over random trees. `mutant` reverses the row order.
    /// `others_all` hands the other methods every generator, orbitope swaps included
    /// (the combination the driver does not make). Returns
    /// `(trees, fired, pruned, orbits)` or the first lost orbit.
    fn experiment(mutant: bool, others_all: bool) -> Result<(usize, usize, usize, usize), String> {
        // (n, ub, p, k, extra generators on the columns past the orbitope, feasible)
        type Case = (
            usize,
            f64,
            usize,
            usize,
            Vec<Vec<usize>>,
            fn(&[f64]) -> bool,
        );
        let cases: Vec<Case> = vec![
            // A 2 x 3 orbitope over {0,1,2}.
            (6, 2.0, 2, 3, vec![], |x| x.iter().sum::<f64>() <= 7.0),
            // A 3 x 2 binary orbitope (two "bins", three "items").
            (6, 1.0, 3, 2, vec![], |x| {
                x[0] + x[3] <= 1.0 && x[1] + x[4] <= 1.0 && x[2] + x[5] <= 1.0
            }),
            // A 2 x 2 orbitope next to a swap of columns 4 and 5.
            (6, 1.0, 2, 2, vec![vec![0, 1, 2, 3, 5, 4]], |x| {
                x[0] + x[1] + x[4] >= 1.0 && x[2] + x[3] + x[5] <= 2.0
            }),
        ];
        let mut rng = Rng(11);
        let (mut trees, mut fired, mut pruned, mut orbits) = (0usize, 0usize, 0usize, 0usize);
        for (n, ub, p, k, extra, pred) in &cases {
            let (n, p, k) = (*n, *p, *k);
            let swaps: Vec<Vec<usize>> = (0..k - 1).map(|j| swap(n, p, j, j + 1)).collect();
            let mut gens = swaps.clone();
            gens.extend(extra.iter().cloned());
            let group = closure(&gens);
            let act = |g: &Vec<usize>, x: &[f64]| -> Vec<f64> { (0..n).map(|j| x[g[j]]).collect() };
            let root_lo = vec![0.0; n];
            let root_hi = vec![*ub; n];
            let feas: Vec<Vec<f64>> = all_points(&root_lo, &root_hi)
                .into_iter()
                .filter(|x| group.iter().all(|g| pred(&act(g, x))))
                .collect();
            assert!(!feas.is_empty());
            let orbitope = Orbitope {
                cols: (0..k)
                    .map(|j| (0..p).map(|r| j * p + r).collect())
                    .collect(),
            };
            let c = vec![0.0; n];
            let dor = DynamicOrbitopes::new(Orbitopes::new(vec![orbitope], n, &c));
            let others: Vec<Vec<usize>> = if others_all {
                gens.clone()
            } else {
                gens.iter()
                    .filter(|g| (0..n).all(|j| g[j] == j || !dor.member(j)))
                    .cloned()
                    .collect()
            };
            let lex = (!others.is_empty()).then(|| DynamicLex::new(&others, n, &c, &vec![true; n]));
            let orb = (!others.is_empty()).then(|| OrbitalReduction::new(&others, n, &c));
            let broken = orb
                .as_ref()
                .map(|o| o.broken(&root_lo, &root_hi))
                .unwrap_or_else(|| vec![false; n]);
            for (with_lex, with_orb) in [(false, false), (true, false), (false, true), (true, true)]
            {
                if (with_lex || with_orb) && others.is_empty() {
                    continue;
                }
                let s = Setup {
                    dor: &dor,
                    reverse: mutant,
                    lex: lex.as_deref().filter(|_| with_lex),
                    orb: orb.as_deref().filter(|_| with_orb),
                    feas: &feas,
                    root_lo: &root_lo,
                    root_hi: &root_hi,
                    broken: &broken,
                };
                for _ in 0..60 {
                    let mut kept = Vec::new();
                    let mut stats = (0usize, 0usize);
                    tree(
                        &s,
                        root_lo.clone(),
                        root_hi.clone(),
                        Vec::new(),
                        &mut rng,
                        &mut stats,
                        &mut kept,
                    );
                    fired += stats.0;
                    pruned += stats.1;
                    trees += 1;
                    let mut seen = vec![false; feas.len()];
                    for (i, x) in feas.iter().enumerate() {
                        if seen[i] {
                            continue;
                        }
                        let orbit: Vec<Vec<f64>> = group.iter().map(|g| act(g, x)).collect();
                        for (t, y) in feas.iter().enumerate() {
                            if orbit.contains(y) {
                                seen[t] = true;
                            }
                        }
                        orbits += 1;
                        if !kept.iter().any(|y| orbit.contains(y)) {
                            return Err(format!(
                                "orbit of {x:?} lost (p {p}, k {k}, lex {with_lex}, \
                                 orbital {with_orb})"
                            ));
                        }
                    }
                }
            }
        }
        Ok((trees, fired, pruned, orbits))
    }

    /// Every orbit keeps a leaf representative, with orbitopal fixing alone and
    /// combined with dynamic lex and orbital reduction on the remaining generators.
    #[test]
    fn every_orbit_keeps_a_representative() {
        let (trees, fired, pruned, orbits) = experiment(false, false).unwrap();
        eprintln!("orbitopal retention: {trees} {fired} {pruned} {orbits}");
        assert!(
            trees == 360 && fired > 500 && pruned > 40 && orbits > 1000,
            "{trees} {fired} {pruned} {orbits}"
        );
    }

    /// Negative control: rows in reverse first-branching order (not extended by
    /// appending) must lose an orbit.
    #[test]
    fn mutant_with_reversed_rows_loses_an_orbit() {
        let e = experiment(true, false);
        assert!(e.is_err(), "the mutant passed: {e:?}");
    }

    /// Negative control for the driver's generator filter: handing the orbitope's own
    /// swaps to dynamic lex or orbital reduction as well (their branching order is not
    /// the row-wise one) must lose an orbit.
    #[test]
    fn sharing_the_orbitope_generators_loses_an_orbit() {
        let e = experiment(false, true);
        assert!(e.is_err(), "sharing passed: {e:?}");
    }

    /// Branching on an entry of row 1 first orders row 1 before row 0.
    #[test]
    fn rows_follow_the_branching_order() {
        // 2 x 2: C_0 = (0, 1), C_1 = (2, 3).
        let o = Orbitope {
            cols: vec![vec![0, 1], vec![2, 3]],
        };
        let d = DynamicOrbitopes::new(Orbitopes::new(vec![o], 4, &[0.0; 4]));
        // x1 = 0 branched: over row 1 alone, x1 >= x3 forces x3 = 0.
        let mut out = Vec::new();
        d.propagate(&[1], &[0.0; 4], &[1.0, 0.0, 1.0, 1.0], 1e-6, &mut out);
        assert_eq!(out, vec![(3, false, 0.0)]);
        // Nothing branched: no rows, no change.
        out.clear();
        d.propagate(&[], &[0.0; 4], &[1.0, 0.0, 1.0, 1.0], 1e-6, &mut out);
        assert!(out.is_empty());
    }
}
