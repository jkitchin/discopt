//! Dynamic lexicographic reduction in the branching order
//! (`DISCOPT_MILP_SYM_DYNAMIC`).
//!
//! Independent implementation (no translated code) of the dynamic lexicographic
//! reduction of van Doornmalen & Hojny's unified symmetry-handling framework
//! (`vandoornmalen2023unified`). The pairwise lex propagator is the one of
//! [`super::orbitope`] (Frisch et al. 2002, `frisch2002global`), generalised to two
//! sequences that may share variables.
//!
//! # What is claimed, and why it is sound
//!
//! Let `G` be a set of **verified** formulation automorphisms of the model being
//! solved (`bnb::milp_symmetry::detect`), closed under inverses here, and `Γ` the
//! group they generate. Write `γ(x)_j = x_{π(j)}`. At a node `β` of the search tree,
//! let `σ_β = (b_1, ..., b_m)` be the columns branched on along the path from the
//! root, each at its first branching, in that order. For every `π ∈ G` the node
//! propagates
//!
//! ```text
//!     (x_{b_1}, ..., x_{b_m})  >=_lex  (x_{π(b_1)}, ..., x_{π(b_m)}).
//! ```
//!
//! Let `F` be the improving feasible set (feasible, objective below the incumbent).
//! It is `Γ`-invariant: the automorphisms preserve rows, bounds, costs and
//! integrality. Fix an orbit `O ⊆ F` and follow the tree from the root. Keep, at each
//! visited node `β`, a point `y ∈ O` in `β`'s region whose `σ_β`-values are the
//! lexicographic maximum over `O`. At the root `σ` is empty and any `y` works. A
//! node's region is cut out by its branching bounds, which constrain `σ_β`-columns
//! only. Every other reduction removes either points outside `F` (propagation,
//! reduced cost, conflicts, cutoff) or points that violate a constraint above, which
//! `y` satisfies by maximality. So `y` survives all of them.
//!
//! When `β` branches on a column already in `σ_β`, the children keep `σ_β`, and `y`
//! lies in one of them. When `β` branches on a new column `i`, take `y'` maximising
//! `x_i` among the points of `O` whose `σ_β`-values equal `y`'s. `y'` has the same
//! `σ_β`-values as `y`, so it is in `β`'s region and in one child, and its
//! `σ_β + i` values are the lex maximum over `O`. Hence every orbit keeps a
//! representative at some leaf, and the reduction removes points, never an orbit.
//! The argument needs no symmetric node box. Conflicts learned with these reductions
//! are statements over real bound changes (`Reason::Unspecified` leaves) and stay
//! valid for `F`.
//!
//! The propagator derives only consequences of one such lex constraint and the
//! current bounds, as [`super::orbitope`] does. Skip the positions where both entries
//! are fixed equal or are the same column. At the first remaining position `α`, a
//! fixed `a < b` is infeasible, and a fixed `a > b` satisfies the constraint.
//! Otherwise `a_α >= b_α + d`, where `d = 1` exactly when a tie at `α` cannot be
//! completed. That holds when the two intervals do not meet, or when
//! `ub(a_suffix) <_lex lb(b_suffix)`. The suffix test is a necessary condition for any
//! point of the box, shared columns included, since `x_a <= ub(a)` and
//! `x_b >= lb(b)` componentwise imply `x_a <=_lex ub(a) <_lex lb(b) <=_lex x_b`.
//!
//! Only integer columns enter `σ`. The order is cut at the first column that is not
//! a verified integer, and a lex constraint over a prefix is implied by the full one.

use std::sync::Arc;

pub use super::orbitope::Tightening;

/// Verified automorphisms of one model, for dynamic lex reduction.
#[derive(Debug, Clone)]
pub struct DynamicLex {
    /// Non-identity structural column maps `π` (with `γ(x)_j = x_{π(j)}`), closed
    /// under inverses and deduplicated.
    pub perms: Vec<Vec<usize>>,
    /// Structural columns of the model they were verified on.
    pub ns: usize,
    /// Per structural column: a verified integer (the only columns `σ` may hold).
    pub is_int: Vec<bool>,
    /// Fingerprint of the model's structural costs (see [`Self::matches`]).
    cost_bits: Vec<u64>,
}

fn cost_key(v: f64) -> u64 {
    if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

impl DynamicLex {
    /// Bundle `generators` (column maps over at least the `ns` structural columns,
    /// mapping structural columns to structural columns) for a model with costs
    /// `c[..ns]` and integrality `is_int[..ns]`.
    pub fn new(generators: &[Vec<usize>], ns: usize, c: &[f64], is_int: &[bool]) -> Arc<Self> {
        let mut perms: Vec<Vec<usize>> = Vec::new();
        for g in generators {
            assert!(g.len() >= ns, "a generator covers every structural column");
            let p: Vec<usize> = g[..ns].to_vec();
            assert!(
                p.iter().all(|&j| j < ns),
                "a verified generator maps structural columns to structural columns"
            );
            let mut inv = vec![usize::MAX; ns];
            for (j, &t) in p.iter().enumerate() {
                assert_eq!(inv[t], usize::MAX, "a generator is a permutation");
                inv[t] = j;
            }
            if p.iter().enumerate().all(|(j, &t)| j == t) {
                continue;
            }
            for q in [p, inv] {
                if !perms.contains(&q) {
                    perms.push(q);
                }
            }
        }
        Arc::new(DynamicLex {
            perms,
            ns,
            is_int: is_int[..ns].to_vec(),
            cost_bits: c[..ns].iter().map(|v| cost_key(*v)).collect(),
        })
    }

    /// Whether these automorphisms were verified on a model with `ns` structural
    /// columns and costs `c[..ns]`.
    pub fn matches(&self, ns: usize, c: &[f64]) -> bool {
        ns == self.ns
            && c.len() >= ns
            && c[..ns]
                .iter()
                .zip(&self.cost_bits)
                .all(|(v, &k)| cost_key(*v) == k)
    }

    /// The usable prefix of a branching order: distinct columns, cut at the first
    /// one that is not a verified integer structural column.
    pub fn usable_order(&self, order: &[usize]) -> usize {
        order
            .iter()
            .position(|&j| j >= self.ns || !self.is_int[j])
            .unwrap_or(order.len())
    }

    /// Propagate `x_order >=_lex π(x)_order` for every map, over the bounds
    /// `col_lower`/`col_upper` (rounded inward first). `order` must be distinct
    /// columns. Appends the tightenings to `out`. When a constraint cannot hold, it
    /// appends one lower bound above its column's upper bound, which makes the domain
    /// infeasible. Returns the number appended.
    pub fn propagate(
        &self,
        order: &[usize],
        col_lower: &[f64],
        col_upper: &[f64],
        feastol: f64,
        out: &mut Vec<Tightening>,
    ) -> usize {
        let start = out.len();
        let order = &order[..self.usable_order(order)];
        if order.is_empty() {
            return 0;
        }
        // The maps that move a column of the order, as pairs of local sequences.
        let active: Vec<&Vec<usize>> = self
            .perms
            .iter()
            .filter(|p| order.iter().any(|&j| p[j] != j))
            .collect();
        if active.is_empty() {
            return 0;
        }
        let mut idx: Vec<usize> = order.to_vec();
        for p in &active {
            idx.extend(order.iter().map(|&j| p[j]));
        }
        idx.sort_unstable();
        idx.dedup();
        let pos = |j: usize| idx.binary_search(&j).expect("a propagated column");
        let a: Vec<usize> = order.iter().map(|&j| pos(j)).collect();
        let bs: Vec<Vec<usize>> = active
            .iter()
            .map(|p| order.iter().map(|&j| pos(p[j])).collect())
            .collect();
        let mut lo: Vec<f64> = idx
            .iter()
            .map(|&j| (col_lower[j] - feastol).ceil())
            .collect();
        let mut hi: Vec<f64> = idx
            .iter()
            .map(|&j| (col_upper[j] + feastol).floor())
            .collect();
        let (lo0, hi0) = (lo.clone(), hi.clone());
        if let Some(t) = (0..lo.len()).find(|&t| lo[t] > hi[t]) {
            out.push((idx[t], true, col_upper[idx[t]] + 1.0));
            return out.len() - start;
        }
        // Each sweep tightens an integer bound or stops; the cap only bounds the work
        // (stopping early is sound: fewer consequences, none wrong).
        for _ in 0..8 {
            let mut changed = false;
            for b in &bs {
                match propagate_lex(&a, b, &mut lo, &mut hi) {
                    Err(t) => {
                        out.push((idx[t], true, col_upper[idx[t]] + 1.0));
                        return out.len() - start;
                    }
                    Ok(c) => changed |= c,
                }
            }
            if !changed {
                break;
            }
        }
        for t in 0..lo.len() {
            if lo[t] > lo0[t] {
                out.push((idx[t], true, lo[t]));
            }
            if hi[t] < hi0[t] {
                out.push((idx[t], false, hi[t]));
            }
        }
        out.len() - start
    }
}

/// Some point of the box has `a >=_lex b` (a necessary condition; see the module doc).
fn suffix_can_reach(a: &[usize], b: &[usize], lo: &[f64], hi: &[f64]) -> bool {
    for (&x, &y) in a.iter().zip(b) {
        if hi[x] > lo[y] {
            return true;
        }
        if hi[x] < lo[y] {
            return false;
        }
    }
    true
}

/// Propagate `a >=_lex b` on integer bounds `lo`/`hi` over local columns, which `a`
/// and `b` may share. `Err(t)` names a column whose domain the constraint empties;
/// otherwise returns whether a bound changed.
fn propagate_lex(a: &[usize], b: &[usize], lo: &mut [f64], hi: &mut [f64]) -> Result<bool, usize> {
    let p = a.len();
    let mut r = 0;
    while r < p {
        let (x, y) = (a[r], b[r]);
        if x == y {
            r += 1;
            continue;
        }
        if lo[x] == hi[x] && lo[y] == hi[y] {
            if lo[x] > lo[y] {
                return Ok(false);
            }
            if lo[x] < lo[y] {
                return Err(x);
            }
            r += 1;
            continue;
        }
        break;
    }
    if r == p {
        return Ok(false);
    }
    let (x, y) = (a[r], b[r]);
    let tie =
        lo[x].max(lo[y]) <= hi[x].min(hi[y]) && suffix_can_reach(&a[r + 1..], &b[r + 1..], lo, hi);
    let d = if tie { 0.0 } else { 1.0 };
    let nlo = lo[x].max(lo[y] + d);
    let nhi = hi[y].min(hi[x] - d);
    let changed = nlo > lo[x] || nhi < hi[y];
    lo[x] = nlo;
    hi[y] = nhi;
    if lo[x] > hi[x] {
        return Err(x);
    }
    if lo[y] > hi[y] {
        return Err(y);
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
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

    fn lex_holds(order: &[usize], p: &[usize], x: &[f64]) -> bool {
        for &j in order {
            if x[j] > x[p[j]] {
                return true;
            }
            if x[j] < x[p[j]] {
                return false;
            }
        }
        true
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

    /// The group generated by `gens` (small `n`), every element as a column map.
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

    /// Exhaustive on small boxes: the propagated box keeps every point that satisfies
    /// all the lex constraints (no valid point is cut), and an infeasible verdict means
    /// none exists. Shared columns between the two sequences are the normal case.
    #[test]
    fn propagation_keeps_every_lex_point() {
        let n = 5;
        let gens = vec![vec![1, 2, 3, 4, 0], vec![1, 0, 2, 3, 4]];
        let d = DynamicLex::new(&gens, n, &[0.0; 5], &[true; 5]);
        let mut rng = Rng(11);
        let (mut checked, mut tightened, mut infeasible) = (0usize, 0usize, 0usize);
        for _ in 0..600 {
            let (mut lo, mut hi) = (vec![0.0; n], vec![0.0; n]);
            for j in 0..n {
                lo[j] = rng.next(2) as f64;
                hi[j] = lo[j] + rng.next(3) as f64;
            }
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                order.swap(i, rng.next(i as u64 + 1) as usize);
            }
            order.truncate(1 + rng.next(n as u64) as usize);
            let mut out = Vec::new();
            d.propagate(&order, &lo, &hi, 1e-6, &mut out);
            let (mut nlo, mut nhi) = (lo.clone(), hi.clone());
            apply(&out, &mut nlo, &mut nhi);
            let empty = (0..n).any(|j| nlo[j] > nhi[j]);
            tightened += usize::from(!out.is_empty());
            infeasible += usize::from(empty);
            for x in all_points(&lo, &hi) {
                if d.perms.iter().all(|p| lex_holds(&order, p, &x)) {
                    checked += 1;
                    assert!(!empty, "infeasible verdict with a lex point {x:?}");
                    for j in 0..n {
                        assert!(nlo[j] <= x[j] && x[j] <= nhi[j], "cut {x:?} at {j}");
                    }
                }
            }
        }
        assert!(
            checked > 1000 && tightened > 100 && infeasible > 0,
            "{checked} {tightened} {infeasible}"
        );
    }

    /// One random branch-and-bound tree with dynamic lex reduction at every node and
    /// a valid "propagation" (shrink to the hull of the feasible points left). Returns
    /// the feasible points at the leaves.
    #[allow(clippy::too_many_arguments)]
    fn tree(
        d: &DynamicLex,
        feas: &[Vec<f64>],
        lo: Vec<f64>,
        hi: Vec<f64>,
        sigma: Vec<usize>,
        rng: &mut Rng,
        stats: &mut (usize, usize),
        kept: &mut Vec<Vec<f64>>,
    ) {
        let n = lo.len();
        let (mut lo, mut hi) = (lo, hi);
        let mut out = Vec::new();
        d.propagate(&sigma, &lo, &hi, 1e-6, &mut out);
        stats.0 += out.len();
        apply(&out, &mut lo, &mut hi);
        if (0..n).any(|j| lo[j] > hi[j]) {
            stats.1 += 1;
            return;
        }
        let inside: Vec<&Vec<f64>> = feas
            .iter()
            .filter(|x| (0..n).all(|j| lo[j] <= x[j] && x[j] <= hi[j]))
            .collect();
        if inside.is_empty() {
            return;
        }
        if rng.next(3) == 0 {
            // A valid reduction: the hull of the feasible points in the box.
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
        // Prefer re-branching on a column already in `σ` now and then.
        let j = free[rng.next(free.len() as u64) as usize];
        let v = lo[j] + rng.next((hi[j] - lo[j]) as u64) as f64;
        let mut s = sigma.clone();
        if !s.contains(&j) {
            s.push(j);
        }
        let (mut hd, mut lu) = (hi.clone(), lo.clone());
        hd[j] = v;
        lu[j] = v + 1.0;
        tree(d, feas, lo.clone(), hd, s.clone(), rng, stats, kept);
        tree(d, feas, lu, hi, s, rng, stats, kept);
    }

    /// The framework's claim, exhaustively: over random trees with random branching
    /// orders and split values, on symmetric feasible sets, every orbit of feasible
    /// points keeps at least one representative at a leaf.
    #[test]
    fn every_orbit_keeps_a_representative() {
        type Case = (usize, f64, Vec<Vec<usize>>, fn(&[f64]) -> bool);
        let cases: Vec<Case> = vec![
            // S_4 on 4 columns in {0,1,2}, with a symmetric knapsack.
            (4, 2.0, vec![vec![1, 2, 3, 0], vec![1, 0, 2, 3]], |x| {
                x.iter().sum::<f64>() <= 5.0
            }),
            // Two interchangeable "bins" of three binaries each (column swap) and
            // a cyclic shift of the rows: a product group.
            (
                6,
                1.0,
                vec![vec![3, 4, 5, 0, 1, 2], vec![1, 2, 0, 4, 5, 3]],
                |x| x[0] + x[1] + x[2] >= 1.0 && x[3] + x[4] + x[5] <= 2.0 || x[0] == x[3],
            ),
            // A cyclic group C_5 only (no transpositions), general integers.
            (5, 2.0, vec![vec![1, 2, 3, 4, 0]], |x| {
                x.iter().sum::<f64>() >= 3.0
            }),
        ];
        let mut rng = Rng(3);
        let (mut trees, mut fired, mut pruned, mut orbits) = (0usize, 0usize, 0usize, 0usize);
        for (n, ub, gens, pred) in &cases {
            let n = *n;
            let group = closure(gens);
            let all = all_points(&vec![0.0; n], &vec![*ub; n]);
            let feas_raw: Vec<Vec<f64>> = all.into_iter().filter(|x| pred(x)).collect();
            // The feasible set used must be invariant: take its group closure's core
            // (the points whose whole orbit satisfies the predicate).
            let act = |g: &Vec<usize>, x: &[f64]| -> Vec<f64> { (0..n).map(|j| x[g[j]]).collect() };
            let feas: Vec<Vec<f64>> = feas_raw
                .iter()
                .filter(|x| group.iter().all(|g| pred(&act(g, x))))
                .cloned()
                .collect();
            assert!(!feas.is_empty());
            for use_group in [false, true] {
                let maps = if use_group {
                    group.clone()
                } else {
                    gens.clone()
                };
                let d = DynamicLex::new(&maps, n, &vec![0.0; n], &vec![true; n]);
                for _ in 0..60 {
                    let mut kept = Vec::new();
                    let mut stats = (0usize, 0usize);
                    tree(
                        &d,
                        &feas,
                        vec![0.0; n],
                        vec![*ub; n],
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
                        for (k, y) in feas.iter().enumerate() {
                            if orbit.contains(y) {
                                seen[k] = true;
                            }
                        }
                        orbits += 1;
                        assert!(
                            kept.iter().any(|y| orbit.contains(y)),
                            "orbit of {x:?} lost (gens {gens:?}, full group {use_group})"
                        );
                    }
                }
            }
        }
        assert!(
            trees == 360 && fired > 1000 && pruned > 100 && orbits > 1000,
            "{trees} {fired} {pruned} {orbits}"
        );
    }

    #[test]
    fn order_stops_at_a_continuous_column() {
        let d = DynamicLex::new(&[vec![1, 0, 2]], 3, &[1.0, 1.0, 0.0], &[true, true, false]);
        assert_eq!(d.usable_order(&[0, 2, 1]), 1);
        assert_eq!(d.usable_order(&[2, 0]), 0);
        assert_eq!(d.perms.len(), 1, "an involution is its own inverse");
        assert!(d.matches(3, &[1.0, 1.0, 0.0]));
        assert!(!d.matches(3, &[1.0, 2.0, 0.0]));
        // x0 >= x1 at order (0): fixes x0 = 0 to force x1 = 0.
        let mut out = Vec::new();
        d.propagate(&[0], &[0.0, 0.0, 0.0], &[0.0, 1.0, 5.0], 1e-6, &mut out);
        assert_eq!(out, vec![(1, false, 0.0)]);
    }
}
