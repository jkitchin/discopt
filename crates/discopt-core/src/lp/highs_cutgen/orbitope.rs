//! Column-lexicographic propagation of full integer orbitopes
//! (`DISCOPT_MILP_ORBITOPE`).
//!
//! Independent implementation (no translated code). The method is the propagation of
//! a chain of lexicographic ordering constraints `C_0 >=_lex C_1 >=_lex ...` between
//! the columns of an orbitope (Kaibel & Pfetsch 2008, `kaibel2008packing`, for
//! orbitopes; Frisch et al. 2002, `frisch2002global`, for the pairwise lex
//! propagator; Bendotti et al. 2021, `bendotti2021orbitopal`, for general integer
//! full orbitopes).
//!
//! # What is claimed, and why it is sound
//!
//! An [`Orbitope`] is a `p x k` matrix of distinct integer structural columns with
//! finite bounds. The detector (`bnb::milp_symmetry::orbitopes`) builds it only from
//! **verified** formulation automorphisms `g_0, ..., g_{k-2}`, where `g_j` maps column
//! `C_j` of the matrix elementwise onto `C_{j+1}` and back, and moves no other
//! structural column. It also requires that the indices increase along every row,
//! and that every column is ascending in the one common row order.
//!
//! For such `g_j`, in the global column-index order that the lex rows of
//! `DISCOPT_MILP_SYM_LEX` also use, the first position where `x` and `g_j(x)` differ
//! is `C_j[alpha]`, with `alpha` the first row where `x_{C_j} != x_{C_{j+1}}`. So
//! `x >=_lex g_j(x)` is exactly `x_{C_j} >=_lex x_{C_{j+1}}` in row order. The
//! lexicographic maximum of every orbit of feasible points satisfies
//! `x >=_lex g(x)` for every `g` in the group, so the column constraints and the lex
//! rows hold together: they remove points, never an orbit.
//!
//! The propagator ([`Orbitope::propagate`]) derives only consequences of
//! `a >=_lex b` and the current bounds, for adjacent columns `a`, `b`. Skip the rows
//! where both entries are fixed equal. At the first remaining row `alpha`, a fixed
//! `a < b` is infeasible, and a fixed `a > b` satisfies the constraint. Otherwise
//! `a_alpha >= b_alpha + d`, where `d = 1` exactly when a tie at `alpha` cannot be
//! completed: either the two intervals do not meet, or the suffix cannot reach
//! `>=_lex` (`ub(a_suffix) <_lex lb(b_suffix)`). The suffix test is exact, because
//! the entries are distinct columns with independent bounds. So
//! `lb(a_alpha) >= lb(b_alpha) + d` and `ub(b_alpha) <= ub(a_alpha) - d`.
//!
//! Every change is made with `Reason::Unspecified`, which is a non-resolvable leaf of
//! conflict analysis. A conflict that contains one is a statement over real bound
//! changes, valid for the reduced problem of this run, and it never leaves the run.

use std::sync::Arc;

/// One full integer orbitope: `cols[j][r]` is the structural column in row `r` of
/// orbitope column `j`.
#[derive(Debug, Clone, PartialEq)]
pub struct Orbitope {
    /// The columns, in chain order (`cols[j] >=_lex cols[j+1]`).
    pub cols: Vec<Vec<usize>>,
}

/// The orbitopes of one model, with what the domain checks before it uses them.
#[derive(Debug, Clone)]
pub struct Orbitopes {
    /// Disjoint orbitopes.
    pub list: Vec<Orbitope>,
    /// Structural columns of the model they were detected on.
    pub ns: usize,
    /// Per structural column: whether it is an entry of some orbitope.
    pub member: Vec<bool>,
    /// The bits of the detection model's structural costs (a fingerprint the domain
    /// checks, so the orbitopes never reach a model they were not derived on).
    pub cost_bits: Vec<u64>,
}

impl Orbitopes {
    /// Bundle `list` for a model with `ns` structural columns and costs `c[..ns]`.
    pub fn new(list: Vec<Orbitope>, ns: usize, c: &[f64]) -> Arc<Self> {
        let mut member = vec![false; ns];
        for o in &list {
            for col in &o.cols {
                for &j in col {
                    assert!(j < ns, "orbitope entries are structural columns");
                    assert!(!member[j], "orbitope entries are distinct");
                    member[j] = true;
                }
            }
        }
        Arc::new(Orbitopes {
            list,
            ns,
            member,
            cost_bits: c[..ns].iter().map(|v| cost_key(*v)).collect(),
        })
    }

    /// Whether these orbitopes were derived on a model with `ns` structural columns
    /// and costs `c[..ns]`.
    pub fn matches(&self, ns: usize, c: &[f64]) -> bool {
        ns == self.ns
            && c.len() >= ns
            && c[..ns]
                .iter()
                .zip(&self.cost_bits)
                .all(|(v, &k)| cost_key(*v) == k)
    }
}

fn cost_key(v: f64) -> u64 {
    if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// A bound tightening found by [`Orbitope::propagate`]: `(column, is_lower, value)`.
pub type Tightening = (usize, bool, f64);

/// `ub(a) >=_lex lb(b)`: some point of the box has `a >=_lex b`.
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

/// Propagate `a >=_lex b` on the integer bounds `lo`/`hi`. Returns `Err(())` when the
/// constraint cannot hold, and otherwise whether a bound changed.
fn propagate_pair(a: &[usize], b: &[usize], lo: &mut [f64], hi: &mut [f64]) -> Result<bool, ()> {
    let p = a.len();
    let mut r = 0;
    while r < p {
        let (x, y) = (a[r], b[r]);
        if lo[x] == hi[x] && lo[y] == hi[y] {
            if lo[x] > lo[y] {
                return Ok(false);
            }
            if lo[x] < lo[y] {
                return Err(());
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
    if lo[x] > hi[x] || lo[y] > hi[y] {
        return Err(());
    }
    Ok(changed)
}

impl Orbitope {
    /// Propagate the column chain over the bounds `col_lower`/`col_upper` (integer
    /// columns, so they are rounded inward first). Appends the tightenings to `out`.
    /// When the chain cannot hold, it appends one lower bound above its column's upper
    /// bound, which makes the domain infeasible. Returns the tightenings found.
    pub fn propagate(
        &self,
        col_lower: &[f64],
        col_upper: &[f64],
        feastol: f64,
        out: &mut Vec<Tightening>,
    ) -> usize {
        let start = out.len();
        let k = self.cols.len();
        // Local copies over the orbitope's own columns.
        let mut idx: Vec<usize> = self.cols.iter().flatten().copied().collect();
        idx.sort_unstable();
        let pos = |j: usize| idx.binary_search(&j).expect("an orbitope column");
        let mut lo: Vec<f64> = idx
            .iter()
            .map(|&j| (col_lower[j] - feastol).ceil())
            .collect();
        let mut hi: Vec<f64> = idx
            .iter()
            .map(|&j| (col_upper[j] + feastol).floor())
            .collect();
        let lo0 = lo.clone();
        let hi0 = hi.clone();
        let local: Vec<Vec<usize>> = self
            .cols
            .iter()
            .map(|c| c.iter().map(|&j| pos(j)).collect())
            .collect();
        if let Some(t) = (0..lo.len()).find(|&t| lo[t] > hi[t]) {
            out.push((idx[t], true, col_upper[idx[t]] + 1.0));
            return out.len() - start;
        }
        // Each sweep tightens an integer bound or stops; the cap only bounds the work
        // (stopping early is sound: fewer consequences, none wrong).
        for _ in 0..(2 * k + 2) {
            let mut changed = false;
            for j in 0..k.saturating_sub(1) {
                match propagate_pair(&local[j], &local[j + 1], &mut lo, &mut hi) {
                    Err(()) => {
                        let x = local[j]
                            .iter()
                            .copied()
                            .find(|&t| lo[t] > hi[t])
                            .unwrap_or(local[j][0]);
                        out.push((idx[x], true, col_upper[idx[x]] + 1.0));
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

#[cfg(test)]
mod tests {
    use super::*;

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

    fn chain_holds(o: &Orbitope, x: &[f64]) -> bool {
        o.cols.windows(2).all(|w| {
            for (&a, &b) in w[0].iter().zip(&w[1]) {
                if x[a] > x[b] {
                    return true;
                }
                if x[a] < x[b] {
                    return false;
                }
            }
            true
        })
    }

    /// Exhaustive on small boxes: the propagated box keeps every point that satisfies
    /// the chain (no valid point is cut), and an infeasible verdict means none exists.
    #[test]
    fn propagation_keeps_every_chain_point() {
        let o = Orbitope {
            cols: vec![vec![0, 3], vec![1, 4], vec![2, 5]],
        };
        let mut checked = 0usize;
        let mut tightened = 0usize;
        let mut infeasible = 0usize;
        let mut seed = 7u64;
        for _ in 0..400 {
            let mut lo = vec![0.0; 6];
            let mut hi = vec![0.0; 6];
            for j in 0..6 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let a = ((seed >> 33) % 3) as f64;
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let w = ((seed >> 33) % 3) as f64;
                lo[j] = a;
                hi[j] = a + w;
            }
            let mut out = Vec::new();
            o.propagate(&lo, &hi, 1e-6, &mut out);
            let (mut nlo, mut nhi) = (lo.clone(), hi.clone());
            for &(j, is_lo, v) in &out {
                if is_lo {
                    nlo[j] = nlo[j].max(v);
                } else {
                    nhi[j] = nhi[j].min(v);
                }
            }
            let empty = (0..6).any(|j| nlo[j] > nhi[j]);
            tightened += usize::from(!out.is_empty());
            infeasible += usize::from(empty);
            for x in all_points(&lo, &hi) {
                if chain_holds(&o, &x) {
                    checked += 1;
                    assert!(!empty, "infeasible verdict with a chain point {x:?}");
                    for j in 0..6 {
                        assert!(nlo[j] <= x[j] && x[j] <= nhi[j], "cut {x:?} at {j}");
                    }
                }
            }
        }
        assert!(
            checked > 1000 && tightened > 50 && infeasible > 0,
            "{checked} {tightened} {infeasible}"
        );
    }

    #[test]
    fn strict_when_the_tie_cannot_complete() {
        // a = (x0, x2), b = (x1, x3); x2 <= 0 and x3 >= 1, so a tie at row 0 fails.
        let o = Orbitope {
            cols: vec![vec![0, 2], vec![1, 3]],
        };
        let lo = [0.0, 0.0, 0.0, 1.0];
        let hi = [3.0, 3.0, 0.0, 1.0];
        let mut out = Vec::new();
        o.propagate(&lo, &hi, 1e-6, &mut out);
        assert!(out.contains(&(0, true, 1.0)), "{out:?}");
        assert!(out.contains(&(1, false, 2.0)), "{out:?}");
    }
}
