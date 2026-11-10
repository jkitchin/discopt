//! Variable-bound discovery, ported from HiGHS: the variable-upper/lower-bound
//! halves of `HighsCliqueTable::extractCliques` (model rows, run once at setup with
//! `transformRows`) and `HighsCliqueTable::extractCliquesFromCut` (each pool cut of
//! length <= 100), stored with `HighsImplications::addVUB`/`addVLB` semantics (MIR
//! strengthening for integer columns, keep the tighter bound per (column, binary),
//! the `5e6 + 10 n` cap).
//!
//! A variable bound `x <= coef * y + constant` (VUB) or `x >= coef * y + constant`
//! (VLB), `y` binary, is derived from one valid row by fixing `y` to either value and
//! bounding the rest of the row by the global box, so it is valid for every point
//! of the box satisfying that row. `HighsTransformedLp` substitutes a column by such
//! a bound when that tightens the base inequality; `untransform` substitutes it
//! back, so cut validity rests on the variable bound's validity alone.
//!
//! Not faithful in one respect, recorded: HiGHS keeps each column's bounds in a
//! hash tree whose iteration order is not reproduced here (insertion order is), so
//! `getBestVub`/`getBestVlb` can break exact ties differently.

use super::cdouble::CDouble;
use super::stdalgo::partition;
use super::transformed_lp::{VarBound, VbEntry};

const INF: f64 = f64::INFINITY;
const K_HIGHS_TINY: f64 = 1e-14;

/// Per column, its VUBs and VLBs in insertion order.
pub struct VarBoundStore {
    vubs: Vec<Vec<(i32, VarBound)>>,
    vlbs: Vec<Vec<(i32, VarBound)>>,
    num: i64,
    max: i64,
    feastol: f64,
}

/// The global box and integrality the discovery reads.
pub struct Domain<'a> {
    /// Column lower bounds.
    pub lb: &'a [f64],
    /// Column upper bounds.
    pub ub: &'a [f64],
    /// Column integrality.
    pub is_int: &'a [bool],
}

impl Domain<'_> {
    fn is_binary(&self, c: usize) -> bool {
        self.is_int[c] && self.lb[c] == 0.0 && self.ub[c] == 1.0
    }
    fn is_fixed(&self, c: usize) -> bool {
        self.lb[c] == self.ub[c]
    }
}

/// `HighsImplications::strengthenVarBound`.
fn strengthen(vb: &mut VarBound, multiplier: f64) {
    if vb.coef.abs() == INF || vb.constant.abs() == INF {
        return;
    }
    const F0MIN: f64 = 0.005;
    const F0MAX: f64 = 0.995;
    let downrhs = (multiplier * vb.constant).floor();
    let f0 = multiplier * vb.constant - downrhs;
    if !(F0MIN..=F0MAX).contains(&f0) {
        return;
    }
    let downaj = (-multiplier * vb.coef + K_HIGHS_TINY).floor();
    let fj = -multiplier * vb.coef - downaj;
    vb.constant = multiplier * downrhs;
    vb.coef = -multiplier * (downaj + (fj - f0).max(0.0) / (1.0 - f0));
}

impl VarBoundStore {
    /// An empty store for `ncols` columns (`maxVarBounds = 5e6 + 10 ncols`).
    pub fn new(ncols: usize, feastol: f64) -> Self {
        VarBoundStore {
            vubs: vec![Vec::new(); ncols],
            vlbs: vec![Vec::new(); ncols],
            num: 0,
            max: 5_000_000 + 10 * ncols as i64,
            feastol,
        }
    }
    /// Stored bounds.
    pub fn len(&self) -> i64 {
        self.num
    }
    /// Whether no bound is stored.
    pub fn is_empty(&self) -> bool {
        self.num == 0
    }
    fn too_many(&self) -> bool {
        self.num >= self.max
    }
    /// A column's VUBs as `getBestVub` reads them (node counts 0 at the root).
    pub fn vubs(&self, col: i32) -> Vec<VbEntry> {
        self.vubs[col as usize]
            .iter()
            .map(|&(c, vb)| VbEntry {
                col: c,
                vb,
                nodes_down: 0,
                nodes_up: 0,
            })
            .collect()
    }
    /// A column's VLBs.
    pub fn vlbs(&self, col: i32) -> Vec<VbEntry> {
        self.vlbs[col as usize]
            .iter()
            .map(|&(c, vb)| VbEntry {
                col: c,
                vb,
                nodes_down: 0,
                nodes_up: 0,
            })
            .collect()
    }

    /// `addVUB(col, vubcol, coef, constant)`.
    pub fn add_vub(&mut self, d: &Domain, col: usize, vubcol: i32, coef: f64, constant: f64) {
        if self.too_many() || (coef.abs() == INF && constant.abs() == INF) {
            return;
        }
        let mut vb = VarBound { coef, constant };
        if d.is_int[col] {
            strengthen(&mut vb, 1.0);
            if vb.coef == 0.0 {
                return;
            }
        }
        let min_bound = vb.min_value();
        if min_bound >= d.ub[col] - self.feastol {
            return;
        }
        let list = &mut self.vubs[col];
        match list.iter_mut().find(|(c, _)| *c == vubcol) {
            Some((_, cur)) => {
                if min_bound < cur.min_value() - self.feastol {
                    *cur = vb;
                }
            }
            None => {
                list.push((vubcol, vb));
                self.num += 1;
            }
        }
    }

    /// `addVLB(col, vlbcol, coef, constant)`.
    pub fn add_vlb(&mut self, d: &Domain, col: usize, vlbcol: i32, coef: f64, constant: f64) {
        if self.too_many() || (coef.abs() == INF && constant.abs() == INF) {
            return;
        }
        let mut vb = VarBound { coef, constant };
        if d.is_int[col] {
            strengthen(&mut vb, -1.0);
            if vb.coef == 0.0 {
                return;
            }
        }
        let max_bound = vb.max_value();
        if max_bound <= d.lb[col] + self.feastol {
            return;
        }
        let list = &mut self.vlbs[col];
        match list.iter_mut().find(|(c, _)| *c == vlbcol) {
            Some((_, cur)) => {
                if max_bound > cur.max_value() + self.feastol {
                    *cur = vb;
                }
            }
            None => {
                list.push((vlbcol, vb));
                self.num += 1;
            }
        }
    }

    /// The variable-bound half of `HighsCliqueTable::extractCliques` on one row
    /// `sum vals x <= rhs` already complemented to positive coefficients
    /// (`complementation[k] = -1` where `x` was replaced by `ub - x`, `+1` where
    /// shifted by `lb`), `rhs` adjusted accordingly.
    fn extract_from_complemented(
        &mut self,
        d: &Domain,
        inds: &[i32],
        vals: &[f64],
        complementation: &[i8],
        rhs: f64,
    ) {
        let feastol = self.feastol;
        let mut perm: Vec<usize> = (0..inds.len()).collect();
        let nbin = partition(&mut perm, |&p| d.is_binary(inds[p] as usize));
        let ntotal = perm.len();
        if nbin >= ntotal {
            return;
        }
        for i in 0..nbin {
            let bincol = inds[perm[i]];
            let impliedub = CDouble::new(rhs) - vals[perm[i]];
            if self.too_many() {
                break;
            }
            for j in nbin..ntotal {
                let col = inds[perm[j]] as usize;
                if d.is_fixed(col) {
                    continue;
                }
                let colub = CDouble::new(d.ub[col]) - d.lb[col];
                let mut implcolub = impliedub / vals[perm[j]];
                if d.is_int[col] {
                    implcolub = CDouble::new((implcolub.to_f64() + feastol).floor());
                }
                if implcolub < colub - feastol {
                    let (coef, mut constant) = if complementation[perm[i]] == -1 {
                        (colub - implcolub, implcolub)
                    } else {
                        (implcolub - colub, colub)
                    };
                    if complementation[perm[j]] == -1 {
                        constant -= d.ub[col];
                        self.add_vlb(d, col, bincol, -coef.to_f64(), -constant.to_f64());
                    } else {
                        constant += d.lb[col];
                        self.add_vub(d, col, bincol, coef.to_f64(), constant.to_f64());
                    }
                }
            }
        }
    }

    /// `extractCliques(mipsolver, transformRows = true)`'s per-row `checkRow` for
    /// a model row `lo <= sum vals x <= up` (both directions).
    pub fn extract_from_model_row(
        &mut self,
        d: &Domain,
        inds: &[i32],
        vals: &[f64],
        lo: f64,
        up: f64,
        epsilon: f64,
    ) {
        for (bound, direction) in [(up, 1.0f64), (lo, -1.0f64)] {
            if direction * bound == INF {
                continue;
            }
            let mut rhs = direction * bound;
            let (mut ci, mut cv, mut cc) = (Vec::new(), Vec::new(), Vec::new());
            let mut freevar = false;
            let mut nbin = 0usize;
            for (&col, &v0) in inds.iter().zip(vals) {
                let c = col as usize;
                let val = direction * v0;
                if val.abs() < epsilon {
                    continue;
                }
                if d.is_binary(c) {
                    nbin += 1;
                }
                if val < 0.0 {
                    if d.ub[c] == INF {
                        freevar = true;
                        break;
                    }
                    cv.push(-val);
                    ci.push(col);
                    cc.push(-1i8);
                    rhs -= val * d.ub[c];
                } else {
                    if d.lb[c] == -INF {
                        freevar = true;
                        break;
                    }
                    cv.push(val);
                    ci.push(col);
                    cc.push(1i8);
                    rhs -= val * d.lb[c];
                }
            }
            if !freevar && nbin != 0 {
                self.extract_from_complemented(d, &ci, &cv, &cc, rhs);
            }
        }
    }

    /// The variable-bound half of `extractCliquesFromCut` for a cut `sum vals x <= rhs`.
    pub fn extract_from_cut(&mut self, d: &Domain, inds: &[i32], vals: &[f64], rhs: f64) {
        let feastol = self.feastol;
        let mut minact = CDouble::new(0.0);
        let mut nbin = 0usize;
        for (&col, &v) in inds.iter().zip(vals) {
            let c = col as usize;
            if d.is_binary(c) {
                nbin += 1;
            }
            if v > 0.0 {
                if d.lb[c] == -INF {
                    return;
                }
                minact += v * d.lb[c];
            } else {
                if d.ub[c] == INF {
                    return;
                }
                minact += v * d.ub[c];
            }
        }
        if rhs - minact < 0.0 {
            minact = CDouble::new(rhs);
        }
        if nbin <= 1 {
            return;
        }
        let len = inds.len();
        let mut perm: Vec<usize> = (0..len).collect();
        let nbin = partition(&mut perm, |&p| d.is_binary(inds[p] as usize));
        if nbin >= len {
            return;
        }
        for i in 0..nbin {
            let bincol = inds[perm[i]];
            let implied_activity = CDouble::new(rhs) - minact - vals[perm[i]].abs();
            for j in nbin..len {
                let col = inds[perm[j]] as usize;
                if d.is_fixed(col) {
                    continue;
                }
                let vj = vals[perm[j]];
                if vj > 0.0 {
                    let mut implcolub = (implied_activity + vj * d.lb[col]).to_f64() / vj;
                    if d.is_int[col] {
                        implcolub = (implcolub + feastol).floor();
                    }
                    if implcolub < d.ub[col] - feastol {
                        let (coef, constant) = if vals[perm[i]] < 0.0 {
                            (d.ub[col] - implcolub, implcolub)
                        } else {
                            if d.ub[col] == INF {
                                continue;
                            }
                            (implcolub - d.ub[col], d.ub[col])
                        };
                        self.add_vub(d, col, bincol, coef, constant);
                    }
                } else {
                    let mut implcollb = (implied_activity + vj * d.ub[col]).to_f64() / vj;
                    if d.is_int[col] {
                        implcollb = (implcollb - feastol).ceil();
                    }
                    if implcollb > d.lb[col] + feastol {
                        let (coef, constant) = if vals[perm[i]] < 0.0 {
                            (d.lb[col] - implcollb, implcollb)
                        } else {
                            if d.lb[col] == -INF {
                                continue;
                            }
                            (implcollb - d.lb[col], d.lb[col])
                        };
                        self.add_vlb(d, col, bincol, coef, constant);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every derived bound must hold at every integer point of a small box that
    /// satisfies the source row (CLAUDE.md §5 feasible-point sampling, exhaustive).
    #[test]
    fn derived_variable_bounds_hold_at_every_feasible_point() {
        // x0 binary, x1 integer in [0, 7], x2 continuous in [0, 5] (sampled on a grid).
        let lb = [0.0, 0.0, 0.0];
        let ub = [1.0, 7.0, 5.0];
        let is_int = [true, true, false];
        let d = Domain {
            lb: &lb,
            ub: &ub,
            is_int: &is_int,
        };
        let rows: Vec<(Vec<f64>, f64, f64)> = vec![
            (vec![4.0, 1.0, 0.0], -INF, 7.0),  // x1 <= 7 - 4 y
            (vec![-3.0, 1.0, 1.0], -INF, 2.5), // x1 + x2 <= 2.5 + 3 y
            (vec![2.0, -1.0, 0.0], -INF, 0.0), // x1 >= 2 y
            (vec![5.0, 0.0, 1.0], 1.0, 6.0),   // 1 <= 5 y + x2 <= 6
        ];
        let mut store = VarBoundStore::new(3, 1e-6);
        let inds = [0i32, 1, 2];
        for (vals, lo, up) in &rows {
            store.extract_from_model_row(&d, &inds, vals, *lo, *up, 1e-9);
            store.extract_from_cut(&d, &inds, vals, *up);
        }
        assert!(store.len() >= 4, "derived only {}", store.len());
        let mut checks = 0usize;
        for y in [0.0, 1.0] {
            for x1 in 0..=7 {
                for k in 0..=50 {
                    let x = [y, x1 as f64, 0.1 * k as f64];
                    let feasible = rows.iter().all(|(v, lo, up)| {
                        let a: f64 = v.iter().zip(&x).map(|(p, q)| p * q).sum();
                        a >= lo - 1e-9 && a <= up + 1e-9
                    });
                    if !feasible {
                        continue;
                    }
                    for col in 1..3 {
                        for e in store.vubs(col) {
                            checks += 1;
                            let bound = e.vb.coef * x[e.col as usize] + e.vb.constant;
                            assert!(
                                x[col as usize] <= bound + 1e-9,
                                "VUB x{col} <= {:?} cut off {x:?}",
                                e.vb
                            );
                        }
                        for e in store.vlbs(col) {
                            checks += 1;
                            let bound = e.vb.coef * x[e.col as usize] + e.vb.constant;
                            assert!(
                                x[col as usize] >= bound - 1e-9,
                                "VLB x{col} >= {:?} cut off {x:?}",
                                e.vb
                            );
                        }
                    }
                }
            }
        }
        assert!(checks > 100, "only {checks} checks");
    }
}
