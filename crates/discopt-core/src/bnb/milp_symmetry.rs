//! Formulation symmetry of a MILP and static lexicographic-leader rows
//! (`DISCOPT_MILP_SYM_LEX`).
//!
//! Independent implementation (no translated code). The methods are the standard ones:
//! colour refinement of the coefficient graph, individualisation to propose
//! automorphisms (McKay & Piperno 2014, `mckay2014practical`), and symmetry-breaking
//! lexicographic-leader constraints (Crawford et al. 1996, `crawford1996symmetry`;
//! Margot 2010, `margot2010symmetry`, for the MILP setting; Pfetsch & Rehn 2019,
//! `pfetsch2019computational`, for their use in a MIP solver).
//!
//! # What is claimed, and why it is sound
//!
//! The input is the driver's working form `A [x; s] = b`, columns `0..ns` structural
//! and `ns..n` slacks, bounds `l <= [x; s] <= u`, cost `c`, integrality `is_int`.
//!
//! * A permutation `pi` of the columns is a **formulation automorphism** when it maps
//!   each column to one with the same integrality, bounds, cost and structural/slack
//!   kind, and maps the multiset of rows onto itself (each row's image, as a set of
//!   `(pi(col), coef)` with the same `b`, is a row, with multiplicities). Every
//!   candidate is checked exactly against those conditions ([`verify`]); colour
//!   refinement only *proposes* candidates, so a hash collision or a wrong guess costs
//!   a rejected candidate, never a wrong reduction. Only automorphisms verified on the
//!   model being solved are used.
//! * An automorphism maps feasible points to feasible points of equal cost, so the
//!   group `G` they generate does too. Fix ONE order of the structural columns
//!   ([`Symmetry::order`], column index). Every `G`-orbit of feasible points contains
//!   its lexicographic maximum `x*` in that order, and `x* >=_lex gamma(x*)` for every
//!   `gamma` in `G`, where `gamma(x)_j = x_{pi(j)}`. Positions `pi(j) = j` compare
//!   equal, and `x >=_lex y` implies the same on any prefix of the moved positions.
//!   Because every row uses the same order, the rows are jointly satisfied by `x*`:
//!   they remove points, never an orbit (`assert_common_order`).
//! * A prefix of bounded integer positions `p_1 < ... < p_k` is encoded as
//!   `sum_t w_t (x_{p_t} - x_{pi(p_t)}) >= 0` with `w_t = prod_{t' > t} r_{t'}`,
//!   `r = floor(u) - ceil(l) + 1` (an integer, so all weights are integers below
//!   [`WEIGHT_CAP`] and the row's activity is exact in `f64`). If the prefixes differ first at `t`, the left side is at least
//!   `w_t - sum_{t' > t} w_{t'} (r_{t'} - 1) = 1 > 0` (telescoping); if they are
//!   equal it is `0`. So the row is exactly non-strict lex on the prefix. The prefix
//!   stops before the first moved continuous or unbounded column and before the
//!   weight product would exceed [`WEIGHT_CAP`]; a shorter prefix is still valid.
//!
//! The rows rest on the invariance of the whole model: any infeasibility proof that
//! uses one has the whole model as its support (closure under the permutations
//! used). [`Symmetry::generators`] records them for that closure.

use crate::lp::simplex::sparse::SparseCols;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// The largest product of column ranges a lex row's weights may reach. Kept small:
/// the weights are LP coefficients (measured on no-ip-65059, HiGHS on the
/// augmented model: 2088 nodes at 1e6, 1713 at 1e3, 2418 at 1e2).
pub const WEIGHT_CAP: f64 = 1e3;

/// Bounds at or beyond this magnitude are infinite (the driver's sentinel).
const INF: f64 = 1e20;

/// `DISCOPT_MILP_SYM_LEX`: append verified symmetry's lexicographic-leader rows to
/// the top-level MILP before the search. Default OFF until its §5 panel; garbage
/// refused.
pub fn sym_lex_enabled() -> bool {
    match std::env::var("DISCOPT_MILP_SYM_LEX") {
        Err(_) => false,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => panic!("DISCOPT_MILP_SYM_LEX={other:?} is not recognized (expected 1 or 0)"),
        },
    }
}

/// `DISCOPT_MILP_ORBITOPE`: propagate the column-lex order of full integer orbitopes
/// built from verified automorphisms (see [`crate::lp::highs_cutgen::orbitope`]).
/// Default OFF (bound-changing, §5 panel owed). Panics on an unrecognized value.
pub fn orbitope_enabled() -> bool {
    match std::env::var("DISCOPT_MILP_ORBITOPE") {
        Err(_) => false,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => panic!("DISCOPT_MILP_ORBITOPE={other:?} is not recognized (expected 1 or 0)"),
        },
    }
}

/// One lexicographic-leader row `sum coef_j x_j >= 0` over structural columns.
#[derive(Debug, Clone, PartialEq)]
pub struct LexRow {
    /// The generator it was derived from (index into [`Symmetry::generators`]).
    pub generator: usize,
    /// The prefix of moved positions, in [`Symmetry::order`].
    pub positions: Vec<usize>,
    /// Nonzero coefficients, ascending column.
    pub coefs: Vec<(usize, f64)>,
}

/// Verified automorphisms and the rows derived from them.
#[derive(Debug, Clone, Default)]
pub struct Symmetry {
    /// The one structural-column order every row uses.
    pub order: Vec<usize>,
    /// Verified automorphisms, as column maps over all `n` columns.
    pub generators: Vec<Vec<usize>>,
    /// The lex-leader rows, one per generator whose first moved position encodes.
    pub rows: Vec<LexRow>,
    /// Candidates proposed and checked.
    pub tried: usize,
}

/// The model the detection reads (the driver's working form).
#[allow(missing_docs)]
pub struct Form<'a> {
    pub csc: &'a SparseCols,
    pub m: usize,
    pub n: usize,
    pub ns: usize,
    pub c: &'a [f64],
    pub l: &'a [f64],
    pub u: &'a [f64],
    pub b: &'a [f64],
    pub is_int: &'a [bool],
}

fn bits(v: f64) -> u64 {
    // -0.0 and 0.0 are the same coefficient; every infinite bound is the same.
    if v == 0.0 {
        0
    } else if v >= INF {
        f64::INFINITY.to_bits()
    } else if v <= -INF {
        f64::NEG_INFINITY.to_bits()
    } else {
        v.to_bits()
    }
}

fn h<T: Hash>(t: T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

type ColKey = (bool, bool, u64, u64, u64);

struct Graph<'a> {
    f: &'a Form<'a>,
    /// CSR of the rows: `(col, coef bits)`.
    rows: Vec<Vec<(usize, u64)>>,
    colkey: Vec<ColKey>,
    /// Row signature (sorted `(col, coef)`, `b`) -> multiplicity.
    sigs: HashMap<(Vec<(usize, u64)>, u64), usize>,
}

impl<'a> Graph<'a> {
    fn new(f: &'a Form<'a>) -> Self {
        let mut rows = vec![Vec::new(); f.m];
        for j in 0..f.n {
            let (ri, vs) = f.csc.col(j);
            for (&i, &v) in ri.iter().zip(vs) {
                if v != 0.0 {
                    rows[i].push((j, bits(v)));
                }
            }
        }
        let colkey = (0..f.n)
            .map(|j| {
                (
                    j < f.ns,
                    f.is_int[j],
                    bits(f.l[j]),
                    bits(f.u[j]),
                    bits(f.c[j]),
                )
            })
            .collect();
        let mut sigs = HashMap::new();
        for (i, r) in rows.iter().enumerate() {
            let mut s = r.clone();
            s.sort_unstable();
            *sigs.entry((s, bits(f.b[i]))).or_insert(0) += 1;
        }
        Graph {
            f,
            rows,
            colkey,
            sigs,
        }
    }

    /// Colour refinement to a fixpoint: `(column colours, row colours, rounds)`.
    fn refine(&self, mut cc: Vec<u64>, mut rc: Vec<u64>) -> (Vec<u64>, Vec<u64>, usize) {
        let f = self.f;
        let mut ncls = usize::MAX;
        let mut buf: Vec<(u64, u64)> = Vec::new();
        for it in 0..1000 {
            let nr: Vec<u64> = (0..f.m)
                .map(|i| {
                    buf.clear();
                    buf.extend(self.rows[i].iter().map(|&(j, v)| (v, cc[j])));
                    buf.sort_unstable();
                    h((rc[i], &buf))
                })
                .collect();
            let nc: Vec<u64> = (0..f.n)
                .map(|j| {
                    let (ri, vs) = f.csc.col(j);
                    buf.clear();
                    buf.extend(
                        ri.iter()
                            .zip(vs)
                            .filter(|(_, &v)| v != 0.0)
                            .map(|(&i, &v)| (bits(v), nr[i])),
                    );
                    buf.sort_unstable();
                    h((cc[j], &buf))
                })
                .collect();
            let k = distinct(&nc) + distinct(&nr);
            cc = nc;
            rc = nr;
            if k == ncls {
                return (cc, rc, it);
            }
            ncls = k;
        }
        panic!("colour refinement did not reach a fixpoint in 1000 rounds");
    }

    /// Exact check: `pi` is a bijection preserving column attributes and the row
    /// multiset.
    fn verify(&self, pi: &[usize]) -> bool {
        let n = self.f.n;
        if pi.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for &k in pi {
            if k >= n || seen[k] {
                return false;
            }
            seen[k] = true;
        }
        if (0..n).any(|j| self.colkey[j] != self.colkey[pi[j]]) {
            return false;
        }
        let mut need: HashMap<(Vec<(usize, u64)>, u64), usize> = HashMap::new();
        for (i, r) in self.rows.iter().enumerate() {
            let mut s: Vec<(usize, u64)> = r.iter().map(|&(j, v)| (pi[j], v)).collect();
            s.sort_unstable();
            let key = (s, bits(self.f.b[i]));
            let have = match self.sigs.get(&key) {
                Some(&c) => c,
                None => return false,
            };
            let e = need.entry(key).or_insert(0);
            *e += 1;
            if *e > have {
                return false;
            }
        }
        true
    }

    /// A candidate `pi` with `pi(v) = w`: individualise `v` in one copy of the base
    /// partition and `w` in the other, refine both, and pair the cells.
    fn find(&self, base: &(Vec<u64>, Vec<u64>, usize), v: usize, w: usize) -> Option<Vec<usize>> {
        let mut a = base.0.clone();
        a[v] = h(("individualised", a[v]));
        let mut b = base.0.clone();
        b[w] = h(("individualised", b[w]));
        let r1 = self.refine(a, base.1.clone());
        let r2 = self.refine(b, base.1.clone());
        if r1.2 != r2.2 {
            return None;
        }
        let mut g1: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut g2: HashMap<u64, Vec<usize>> = HashMap::new();
        for j in 0..self.f.n {
            g1.entry(r1.0[j]).or_default().push(j);
            g2.entry(r2.0[j]).or_default().push(j);
        }
        if g1.len() != g2.len() {
            return None;
        }
        let mut pi = vec![usize::MAX; self.f.n];
        for (k, x) in &g1 {
            let y = g2.get(k)?;
            if x.len() != y.len() {
                return None;
            }
            // Common members stay; the rest pair in ascending order (a block swap comes
            // out as a transposition, not a shift).
            let iny: std::collections::HashSet<usize> = y.iter().copied().collect();
            let inx: std::collections::HashSet<usize> = x.iter().copied().collect();
            let xs: Vec<usize> = x.iter().copied().filter(|j| !iny.contains(j)).collect();
            let ys: Vec<usize> = y.iter().copied().filter(|j| !inx.contains(j)).collect();
            for &j in x.iter().filter(|j| iny.contains(j)) {
                pi[j] = j;
            }
            for (&p, &q) in xs.iter().zip(&ys) {
                pi[p] = q;
            }
        }
        self.verify(&pi).then_some(pi)
    }
}

fn distinct(v: &[u64]) -> usize {
    let mut s = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s.len()
}

/// Detect verified automorphisms (at most `max_tries` candidates, stopping at
/// `deadline`) and derive the lex-leader rows.
pub fn detect(f: &Form<'_>, max_tries: usize, deadline: Option<std::time::Instant>) -> Symmetry {
    assert_eq!(f.is_int.len(), f.n);
    let g = Graph::new(f);
    let cc0: Vec<u64> = g.colkey.iter().map(h).collect();
    let rc0: Vec<u64> = (0..f.m).map(|i| h(("row", bits(f.b[i])))).collect();
    let base = g.refine(cc0, rc0);
    let mut cells: HashMap<u64, Vec<usize>> = HashMap::new();
    for j in 0..f.ns {
        if f.is_int[j] {
            cells.entry(base.0[j]).or_default().push(j);
        }
    }
    let mut cells: Vec<Vec<usize>> = cells.into_values().filter(|c| c.len() > 1).collect();
    // Deterministic: largest cells first, ties by first column.
    cells.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
    let mut sym = Symmetry {
        order: (0..f.ns).collect(),
        ..Default::default()
    };
    'cells: for cell in &cells {
        for w in cell.windows(2) {
            if sym.tried >= max_tries || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break 'cells;
            }
            let (a, b) = (w[0], w[1]);
            if sym.generators.iter().any(|p| p[a] == b) {
                continue;
            }
            sym.tried += 1;
            if let Some(pi) = g.find(&base, a, b) {
                sym.generators.push(pi);
            }
        }
    }
    for (k, pi) in sym.generators.iter().enumerate() {
        if let Some(r) = lex_row(f, &sym.order, pi, k) {
            sym.rows.push(r);
        }
    }
    assert_common_order(&sym);
    sym
}

/// The lex-leader row of `pi` in `order` (see the module docs), or `None` when its
/// first moved position cannot be encoded.
fn lex_row(f: &Form<'_>, order: &[usize], pi: &[usize], generator: usize) -> Option<LexRow> {
    let mut positions = Vec::new();
    let mut weight = 1.0f64;
    for &j in order {
        if pi[j] == j {
            continue;
        }
        if !f.is_int[j] || f.l[j] <= -INF || f.u[j] >= INF {
            break;
        }
        // Integer range, so every weight is an integer below WEIGHT_CAP and the row's
        // activity is exact in f64: propagation on it (and an infeasible domain it
        // yields) involves no rounding.
        let r = f.u[j].floor() - f.l[j].ceil() + 1.0;
        if r < 1.0 || weight * r > WEIGHT_CAP {
            break;
        }
        positions.push(j);
        weight *= r;
    }
    if positions.is_empty() {
        return None;
    }
    let mut coef: HashMap<usize, f64> = HashMap::new();
    let mut w = 1.0f64;
    for &j in positions.iter().rev() {
        *coef.entry(j).or_insert(0.0) += w;
        *coef.entry(pi[j]).or_insert(0.0) -= w;
        w *= f.u[j].floor() - f.l[j].ceil() + 1.0;
    }
    let mut coefs: Vec<(usize, f64)> = coef.into_iter().filter(|&(_, v)| v != 0.0).collect();
    coefs.sort_unstable_by_key(|&(j, _)| j);
    (!coefs.is_empty()).then_some(LexRow {
        generator,
        positions,
        coefs,
    })
}

/// Every row's prefix is an increasing subsequence of the ONE common order: the
/// condition that makes the rows jointly valid (they all keep the same orbit
/// representative).
fn assert_common_order(sym: &Symmetry) {
    let rank: HashMap<usize, usize> = sym.order.iter().enumerate().map(|(k, &j)| (j, k)).collect();
    for r in &sym.rows {
        let ranks: Vec<usize> = r.positions.iter().map(|j| rank[j]).collect();
        assert!(
            ranks.windows(2).all(|w| w[0] < w[1]),
            "lex row of generator {} does not follow the common column order",
            r.generator
        );
    }
}

/// Full integer orbitopes from the verified generators: chains `C_0, ..., C_{k-1}` of
/// disjoint structural column sets in which a verified involution `g_j` swaps `C_j` and
/// `C_{j+1}` elementwise and moves no other structural column. Every entry is a bounded
/// integer, the indices increase along each row, and every column is ascending in the
/// one row order: the conditions [`crate::lp::highs_cutgen::orbitope`] rests on.
pub fn orbitopes(f: &Form<'_>, sym: &Symmetry) -> Vec<crate::lp::highs_cutgen::orbitope::Orbitope> {
    use crate::lp::highs_cutgen::orbitope::Orbitope;
    let ns = f.ns;
    let ok_col = |j: usize| f.is_int[j] && f.l[j] > -INF && f.u[j] < INF;
    // Eligible generators: involutions whose moved structurals are bounded integers.
    let gens: Vec<&Vec<usize>> = sym
        .generators
        .iter()
        .filter(|g| {
            (0..f.n).all(|j| g[g[j]] == j)
                && (0..ns).any(|j| g[j] != j)
                && (0..ns).all(|j| g[j] == j || (g[j] < ns && ok_col(j)))
        })
        .collect();
    let moved = |g: &Vec<usize>| (0..ns).filter(|&j| g[j] != j).count();
    let valid = |cols: &[Vec<usize>]| {
        let p = cols[0].len();
        cols.iter()
            .all(|c| c.len() == p && c.windows(2).all(|w| w[0] < w[1]))
            && (0..p).all(|r| cols.windows(2).all(|w| w[0][r] < w[1][r]))
    };
    let mut taken = vec![false; ns];
    let mut used = vec![false; gens.len()];
    let mut out = Vec::new();
    for i in 0..gens.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let g = gens[i];
        if (0..ns).any(|j| g[j] != j && taken[j]) {
            continue;
        }
        let a: Vec<usize> = (0..ns).filter(|&j| g[j] > j).collect();
        let mut cols = vec![a.clone(), a.iter().map(|&j| g[j]).collect::<Vec<_>>()];
        if !valid(&cols) {
            continue;
        }
        let mut mine = taken.clone();
        for c in &cols {
            for &j in c {
                mine[j] = true;
            }
        }
        loop {
            let mut grew = false;
            for t in 0..gens.len() {
                if used[t] {
                    continue;
                }
                let h = gens[t];
                for front in [false, true] {
                    let end = if front {
                        &cols[0]
                    } else {
                        &cols[cols.len() - 1]
                    };
                    if moved(h) != 2 * end.len() || end.iter().any(|&j| h[j] == j || mine[h[j]]) {
                        continue;
                    }
                    let new: Vec<usize> = end.iter().map(|&j| h[j]).collect();
                    let mut cand = cols.clone();
                    if front {
                        cand.insert(0, new.clone());
                    } else {
                        cand.push(new.clone());
                    }
                    if !valid(&cand) {
                        continue;
                    }
                    cols = cand;
                    for &j in &new {
                        mine[j] = true;
                    }
                    used[t] = true;
                    grew = true;
                    break;
                }
            }
            if !grew {
                break;
            }
        }
        taken = mine;
        out.push(Orbitope { cols });
    }
    out
}

/// The working form with `rows` appended as `sum coef x - s = 0`, `s in [0, INF)`
/// (a new slack column per row, after the existing columns). Returns
/// `(csc, m, n, c, l, u, b)`.
#[allow(clippy::type_complexity)]
pub fn augment(
    f: &Form<'_>,
    rows: &[LexRow],
) -> (
    SparseCols,
    usize,
    usize,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let k = rows.len();
    let mut extra: Vec<Vec<(usize, f64)>> = vec![Vec::new(); f.n];
    for (t, r) in rows.iter().enumerate() {
        for &(j, v) in &r.coefs {
            assert!(j < f.ns, "lex rows are over structural columns");
            extra[j].push((f.m + t, v));
        }
    }
    let (cp, ri, vs) = f.csc.raw();
    let mut col_ptr = vec![0usize];
    let mut row_idx = Vec::with_capacity(ri.len() + 2 * k);
    let mut vals = Vec::with_capacity(ri.len() + 2 * k);
    for j in 0..f.n {
        row_idx.extend_from_slice(&ri[cp[j]..cp[j + 1]]);
        vals.extend_from_slice(&vs[cp[j]..cp[j + 1]]);
        for &(i, v) in &extra[j] {
            row_idx.push(i);
            vals.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    for t in 0..k {
        row_idx.push(f.m + t);
        vals.push(-1.0);
        col_ptr.push(row_idx.len());
    }
    let mut c = f.c.to_vec();
    let mut l = f.l.to_vec();
    let mut u = f.u.to_vec();
    let mut b = f.b.to_vec();
    c.resize(f.n + k, 0.0);
    l.resize(f.n + k, 0.0);
    u.resize(f.n + k, INF);
    b.resize(f.m + k, 0.0);
    (
        SparseCols::from_csc(col_ptr, row_idx, vals),
        f.m + k,
        f.n + k,
        c,
        l,
        u,
        b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test model: `(csc, m, n, c, l, u, b, is_int)`.
    type Model = (
        SparseCols,
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<bool>,
    );

    /// `k` interchangeable blocks of `d` general integers in `[0, r-1]`, one shared
    /// row `sum x >= 1` (with surplus) and per-block rows `x_b1 + x_b2 <= cap`.
    fn blocks(k: usize, d: usize, r: f64, cap: f64) -> Model {
        let ns = k * d;
        let m = 1 + k;
        let n = ns + m;
        let mut dense = vec![0.0; m * n];
        dense[..ns].iter_mut().for_each(|v| *v = 1.0);
        dense[ns] = -1.0;
        for bk in 0..k {
            let i = 1 + bk;
            dense[i * n + bk * d] = 1.0;
            dense[i * n + bk * d + 1] = 1.0;
            dense[i * n + ns + i] = -1.0;
        }
        let csc = SparseCols::from_dense(&dense, m, n);
        let mut l = vec![0.0; n];
        let mut u = vec![r - 1.0; n];
        u[ns] = INF;
        l[ns] = 1.0;
        for i in 1..m {
            l[ns + i] = -INF;
            u[ns + i] = cap;
        }
        let c = vec![0.0; n];
        let b = vec![0.0; m];
        let mut is_int = vec![false; n];
        is_int[..ns].iter_mut().for_each(|v| *v = true);
        (csc, m, n, c, l, u, b, is_int)
    }

    fn form<'a>(t: &'a Model, ns: usize) -> Form<'a> {
        Form {
            csc: &t.0,
            m: t.1,
            n: t.2,
            ns,
            c: &t.3,
            l: &t.4,
            u: &t.5,
            b: &t.6,
            is_int: &t.7,
        }
    }

    #[test]
    fn finds_verified_block_and_coordinate_symmetry() {
        let t = blocks(4, 2, 3.0, 3.0);
        let f = form(&t, 8);
        let s = detect(&f, 1000, None);
        let g = Graph::new(&f);
        assert!(!s.generators.is_empty());
        for p in &s.generators {
            assert!(g.verify(p));
        }
        assert!(!s.rows.is_empty());
    }

    #[test]
    fn rejects_a_non_automorphism() {
        let t = blocks(3, 2, 3.0, 3.0);
        let f = form(&t, 6);
        let g = Graph::new(&f);
        // Swapping x_0 with x_2 alone (different blocks' rows) is not an automorphism.
        let mut pi: Vec<usize> = (0..f.n).collect();
        pi.swap(0, 2);
        assert!(!g.verify(&pi));
        // A non-bijection is refused.
        let mut pi: Vec<usize> = (0..f.n).collect();
        pi[1] = 0;
        assert!(!g.verify(&pi));
    }

    /// Feasible-point sampling: enumerate every integer point of a small symmetric
    /// model; each orbit's lex-max (in the common order) satisfies every row, and the
    /// set kept by the rows meets every orbit.
    #[test]
    fn rows_keep_every_orbit_lex_max() {
        let (k, d, r) = (3usize, 2usize, 3.0f64);
        let t = blocks(k, d, r, 3.0);
        let ns = k * d;
        let f = form(&t, ns);
        let s = detect(&f, 1000, None);
        assert!(!s.rows.is_empty());
        let total = (r as usize).pow(ns as u32);
        let mut checked = 0usize;
        let feasible = |x: &[f64]| {
            x.iter().sum::<f64>() >= 1.0 && (0..k).all(|bk| x[bk * d] + x[bk * d + 1] <= 3.0)
        };
        let point = |mut code: usize| {
            let mut x = vec![0.0; ns];
            for v in x.iter_mut() {
                *v = (code % r as usize) as f64;
                code /= r as usize;
            }
            x
        };
        let keep = |x: &[f64]| {
            s.rows
                .iter()
                .all(|row| row.coefs.iter().map(|&(j, v)| v * x[j]).sum::<f64>() >= 0.0)
        };
        // Orbit of x under the generated group (closure by BFS over generators).
        let orbit = |x: &[f64]| {
            let mut seen = vec![x.to_vec()];
            let mut k = 0;
            while k < seen.len() {
                let y = seen[k].clone();
                for p in &s.generators {
                    let z: Vec<f64> = (0..ns).map(|j| y[p[j]]).collect();
                    if !seen.contains(&z) {
                        seen.push(z);
                    }
                }
                k += 1;
            }
            seen
        };
        for code in 0..total {
            let x = point(code);
            if !feasible(&x) {
                continue;
            }
            let orb = orbit(&x);
            let lexmax = orb.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            assert!(keep(lexmax), "lex-max {lexmax:?} of an orbit is cut");
            assert!(orb.iter().any(|y| keep(y)));
            checked += 1;
        }
        assert!(checked > 100, "probe fired on {checked} points");
    }

    /// Joint soundness of the lex rows and the orbitope chains (feasible-point
    /// sampling): every orbit of feasible points keeps its lex-max under BOTH, every
    /// orbitope's adjacent swap is a verified generator, and orbitopes were found.
    #[test]
    fn orbitopes_and_rows_keep_every_orbit_lex_max() {
        let mut probes = 0usize;
        let mut found = 0usize;
        for &(k, d, r) in &[(3usize, 2usize, 3.0f64), (4, 2, 2.0), (2, 3, 3.0)] {
            let t = blocks(k, d, r, 3.0);
            let ns = k * d;
            let f = form(&t, ns);
            let s = detect(&f, 1000, None);
            let g = Graph::new(&f);
            let orbs = orbitopes(&f, &s);
            found += orbs.len();
            for o in &orbs {
                for w in o.cols.windows(2) {
                    // The swap of the two columns is (on structurals) a verified generator.
                    assert!(s.generators.iter().any(|p| {
                        g.verify(p)
                            && (0..ns).all(|j| {
                                let want = match (
                                    w[0].iter().position(|&a| a == j),
                                    w[1].iter().position(|&b| b == j),
                                ) {
                                    (Some(r), _) => w[1][r],
                                    (_, Some(r)) => w[0][r],
                                    _ => j,
                                };
                                p[j] == want
                            })
                    }));
                }
            }
            let chains = |x: &[f64]| {
                orbs.iter().all(|o| {
                    o.cols.windows(2).all(|w| {
                        for (&a, &b) in w[0].iter().zip(&w[1]) {
                            if x[a] != x[b] {
                                return x[a] > x[b];
                            }
                        }
                        true
                    })
                })
            };
            let keep = |x: &[f64]| {
                s.rows
                    .iter()
                    .all(|row| row.coefs.iter().map(|&(j, v)| v * x[j]).sum::<f64>() >= 0.0)
                    && chains(x)
            };
            let feasible = |x: &[f64]| {
                x.iter().sum::<f64>() >= 1.0 && (0..k).all(|bk| x[bk * d] + x[bk * d + 1] <= 3.0)
            };
            let total = (r as usize).pow(ns as u32);
            for mut code in 0..total {
                let mut x = vec![0.0; ns];
                for v in x.iter_mut() {
                    *v = (code % r as usize) as f64;
                    code /= r as usize;
                }
                if !feasible(&x) {
                    continue;
                }
                let mut seen = vec![x.clone()];
                let mut q = 0;
                while q < seen.len() {
                    let y = seen[q].clone();
                    for p in &s.generators {
                        let z: Vec<f64> = (0..ns).map(|j| y[p[j]]).collect();
                        if !seen.contains(&z) {
                            seen.push(z);
                        }
                    }
                    q += 1;
                }
                let lexmax = seen
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                assert!(keep(lexmax), "lex-max {lexmax:?} of an orbit is cut");
                probes += 1;
            }
        }
        assert!(
            probes > 300 && found >= 3,
            "probes {probes}, orbitopes {found}"
        );
    }

    #[test]
    fn weights_encode_exact_lex_on_the_prefix() {
        let t = blocks(2, 3, 4.0, 5.0);
        let f = form(&t, 6);
        let s = detect(&f, 1000, None);
        for row in &s.rows {
            for &j in &row.positions {
                assert!(f.is_int[j]);
            }
            let prod: f64 = row
                .positions
                .iter()
                .map(|&j| f.u[j] - f.l[j] + 1.0)
                .product();
            assert!(prod <= WEIGHT_CAP);
        }
    }

    #[test]
    fn augment_appends_rows_with_surplus() {
        let t = blocks(2, 2, 3.0, 3.0);
        let f = form(&t, 4);
        let rows = vec![LexRow {
            generator: 0,
            positions: vec![0],
            coefs: vec![(0, 1.0), (2, -1.0)],
        }];
        let (csc, m, n, c, l, u, b) = augment(&f, &rows);
        assert_eq!((m, n), (f.m + 1, f.n + 1));
        assert_eq!(csc.col(n - 1), (&[m - 1][..], &[-1.0][..]));
        assert_eq!(*csc.col(0).0.last().unwrap(), m - 1);
        assert_eq!(
            (c[n - 1], l[n - 1], u[n - 1], b[m - 1]),
            (0.0, 0.0, INF, 0.0)
        );
    }
}
