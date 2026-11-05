//! Feasibility Jump (Luteberget & Sartor 2023), an LP-free MILP primal heuristic
//! (highs-parity P4.6).
//!
//! Rows `lo <= a x <= hi` are split into `g x <= b` rows with weights `w`. The search
//! minimises the weighted violation `sum_i w_i max(0, g_i x - b_i)` by moving one
//! variable at a time to its *jump value*: the value in its box (integral for an
//! integer column) that minimises the weighted violation of its rows with every other
//! variable held. That function of one variable is convex and piecewise linear, so the
//! jump value is found by a slope walk over the rows' breakpoints. When no sampled
//! candidate improves, the weights of the violated rows are increased by one (the
//! paper's weighting step), which reshapes the landscape so the walk can leave the
//! local minimum.
//!
//! Selection is the paper's sampled variant: candidates are the columns of up to
//! [`ROW_SAMPLE`] violated rows plus a few random columns, and the best strictly
//! improving one moves. The objective is not used: the module finds *a* feasible
//! point, nothing else, and the caller must verify any point before trusting it
//! (the MILP driver re-validates seeded incumbents).
//!
//! Experimental: no production path calls this yet. P4.6's entry experiment calls it
//! through `discopt._rust.fj_search_py`; it is wired into the driver only if that
//! experiment passes its bar.

#![allow(clippy::needless_range_loop)]

use std::time::Instant;

/// Infinity sentinel of the Rust LP layer (CLAUDE.md "Key Constraints").
const INF: f64 = 1e20;
/// Violated rows sampled per step.
const ROW_SAMPLE: usize = 25;
/// Random columns added to each step's candidates.
const RANDOM_COLS: usize = 5;
/// Feasibility tolerance on a `<=` row, relative to `1 + |b|`.
const FEAS_TOL: f64 = 1e-6;

/// Outcome of one search.
#[derive(Debug, Clone)]
pub struct FjResult {
    /// The point, when every row is satisfied within tolerance.
    pub x: Option<Vec<f64>>,
    /// Variable moves made.
    pub moves: u64,
    /// Weighting steps (local minima met).
    pub weight_bumps: u64,
}

/// Small deterministic PRNG (xorshift64*), so the search is reproducible per seed.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Problem {
    // `g x <= b`, row-wise and column-wise.
    row_ptr: Vec<usize>,
    row_col: Vec<usize>,
    col_ptr: Vec<usize>,
    col_row: Vec<usize>,
    col_val: Vec<f64>,
    b: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    is_int: Vec<bool>,
}

impl Problem {
    /// From `A` in CSR (`m` x `n`) with row range `[row_lo, row_hi]`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        m: usize,
        n: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        row_lo: &[f64],
        row_hi: &[f64],
        lb: &[f64],
        ub: &[f64],
        is_int: &[bool],
    ) -> Self {
        let mut row_ptr = vec![0usize];
        let mut row_col = Vec::new();
        let mut row_val = Vec::new();
        let mut b = Vec::new();
        for i in 0..m {
            let (s, e) = (indptr[i], indptr[i + 1]);
            for (sign, side) in [(1.0, row_hi[i]), (-1.0, -row_lo[i])] {
                if side.is_finite() && side.abs() < INF {
                    for k in s..e {
                        row_col.push(indices[k]);
                        row_val.push(sign * data[k]);
                    }
                    row_ptr.push(row_col.len());
                    b.push(side);
                }
            }
        }
        // Transpose to columns.
        let nr = b.len();
        let mut cnt = vec![0usize; n + 1];
        for &j in &row_col {
            cnt[j + 1] += 1;
        }
        for j in 0..n {
            cnt[j + 1] += cnt[j];
        }
        let col_ptr = cnt.clone();
        let mut fill = cnt;
        let mut col_row = vec![0usize; row_col.len()];
        let mut col_val = vec![0f64; row_col.len()];
        for r in 0..nr {
            for k in row_ptr[r]..row_ptr[r + 1] {
                let j = row_col[k];
                col_row[fill[j]] = r;
                col_val[fill[j]] = row_val[k];
                fill[j] += 1;
            }
        }
        Problem {
            row_ptr,
            row_col,
            col_ptr,
            col_row,
            col_val,
            b,
            lb: lb.iter().map(|&v| v.max(-INF)).collect(),
            ub: ub.iter().map(|&v| v.min(INF)).collect(),
            is_int: is_int.to_vec(),
        }
    }
}

struct State<'a> {
    p: &'a Problem,
    x: Vec<f64>,
    act: Vec<f64>,
    w: Vec<f64>,
    // Violated rows, maintained incrementally.
    viol: Vec<usize>,
    viol_pos: Vec<usize>,
    breaks: Vec<(f64, f64)>,
}

const NOT_VIOL: usize = usize::MAX;

impl<'a> State<'a> {
    fn new(p: &'a Problem) -> Self {
        let n = p.lb.len();
        let mut x = vec![0.0; n];
        for j in 0..n {
            let mut v = 0f64.clamp(p.lb[j], p.ub[j]);
            if p.is_int[j] {
                v = (v - 1e-9).ceil();
            }
            x[j] = v;
        }
        let nr = p.b.len();
        let mut act = vec![0.0; nr];
        for j in 0..n {
            if x[j] != 0.0 {
                for k in p.col_ptr[j]..p.col_ptr[j + 1] {
                    act[p.col_row[k]] += p.col_val[k] * x[j];
                }
            }
        }
        let mut s = State {
            p,
            x,
            act,
            w: vec![1.0; nr],
            viol: Vec::new(),
            viol_pos: vec![NOT_VIOL; nr],
            breaks: Vec::new(),
        };
        for r in 0..nr {
            s.refresh(r);
        }
        s
    }

    fn violated(&self, r: usize) -> bool {
        self.act[r] - self.p.b[r] > FEAS_TOL * (1.0 + self.p.b[r].abs())
    }

    fn refresh(&mut self, r: usize) {
        let v = self.violated(r);
        let pos = self.viol_pos[r];
        if v && pos == NOT_VIOL {
            self.viol_pos[r] = self.viol.len();
            self.viol.push(r);
        } else if !v && pos != NOT_VIOL {
            let last = *self.viol.last().unwrap();
            self.viol.swap_remove(pos);
            if last != r {
                self.viol_pos[last] = pos;
            }
            self.viol_pos[r] = NOT_VIOL;
        }
    }

    /// Weighted violation over column `j`'s rows with `x_j = t`.
    fn f(&self, j: usize, t: f64) -> f64 {
        let p = self.p;
        let d = t - self.x[j];
        let mut s = 0.0;
        for k in p.col_ptr[j]..p.col_ptr[j + 1] {
            let r = p.col_row[k];
            let v = self.act[r] + p.col_val[k] * d - p.b[r];
            if v > 0.0 {
                s += self.w[r] * v;
            }
        }
        s
    }

    /// `(jump value, score)`: the score is the decrease in weighted violation.
    fn jump(&mut self, j: usize) -> (f64, f64) {
        let p = self.p;
        let xj = self.x[j];
        let (lo, hi) = (p.lb[j], p.ub[j]);
        // Slope walk over breakpoints: row r's term w|a|·max(0, ±(t - t_r)) adds
        // w|a| to the slope once t passes t_r (a > 0) or removes it (a < 0).
        self.breaks.clear();
        let mut slope = 0.0;
        for k in p.col_ptr[j]..p.col_ptr[j + 1] {
            let r = p.col_row[k];
            let a = p.col_val[k];
            let tr = xj + (p.b[r] - self.act[r]) / a;
            let wa = self.w[r] * a.abs();
            if a < 0.0 {
                slope -= wa; // active to the left of t_r
            }
            self.breaks.push((tr, wa));
        }
        self.breaks
            .sort_by(|u, v| u.0.partial_cmp(&v.0).unwrap_or(std::cmp::Ordering::Equal));
        // The minimiser is the first breakpoint at which the slope turns >= 0.
        let mut t_star = lo;
        if slope < 0.0 {
            t_star = hi;
            for &(tr, wa) in &self.breaks {
                slope += wa;
                if slope >= 0.0 {
                    t_star = tr;
                    break;
                }
            }
        }
        let t_star = t_star.clamp(lo, hi);
        let f0 = self.f(j, xj);
        let mut best = (xj, 0.0);
        let cands: [f64; 2] = if p.is_int[j] {
            [(t_star + 1e-9).floor(), (t_star - 1e-9).ceil()]
        } else {
            [t_star, t_star]
        };
        for t in cands {
            let t = t.clamp(lo, hi);
            if !t.is_finite() || t.abs() >= INF || t == xj {
                continue;
            }
            if p.is_int[j] && (t - t.round()).abs() > 1e-9 {
                continue;
            }
            let sc = f0 - self.f(j, t);
            if sc > best.1 + 1e-12 {
                best = (t, sc);
            }
        }
        best
    }

    fn apply(&mut self, j: usize, t: f64) {
        let p = self.p;
        let d = t - self.x[j];
        self.x[j] = t;
        for k in p.col_ptr[j]..p.col_ptr[j + 1] {
            let r = p.col_row[k];
            self.act[r] += p.col_val[k] * d;
            self.refresh(r);
        }
    }
}

/// Run one Feasibility Jump search on `row_lo <= A x <= row_hi`, `lb <= x <= ub`,
/// `x_j` integer where `is_int[j]`; `A` is CSR (`m` x `n`). Stops on the first
/// feasible point, after `max_moves` steps (moves plus weighting steps), or at
/// `deadline`.
#[allow(clippy::too_many_arguments)]
pub fn fj_search(
    m: usize,
    n: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    row_lo: &[f64],
    row_hi: &[f64],
    lb: &[f64],
    ub: &[f64],
    is_int: &[bool],
    seed: u64,
    max_moves: u64,
    deadline: Option<Instant>,
) -> FjResult {
    let p = Problem::new(m, n, indptr, indices, data, row_lo, row_hi, lb, ub, is_int);
    let mut s = State::new(&p);
    let mut rng = Rng::new(seed);
    let (mut moves, mut bumps) = (0u64, 0u64);
    let mut cand: Vec<usize> = Vec::new();
    let mut mark = vec![false; n];
    let mut step = 0u64;
    loop {
        if s.viol.is_empty() {
            return FjResult {
                x: Some(s.x),
                moves,
                weight_bumps: bumps,
            };
        }
        // A step is a move or a weighting step; both count, or a search that only
        // ever re-weights (no improving move exists, e.g. an infeasible system)
        // would never stop without a deadline.
        if moves + bumps >= max_moves {
            break;
        }
        step += 1;
        if step % 256 == 0 {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            }
        }
        cand.clear();
        let nv = s.viol.len();
        let take = nv.min(ROW_SAMPLE);
        for q in 0..take {
            let r = if nv <= ROW_SAMPLE {
                s.viol[q]
            } else {
                s.viol[rng.below(nv)]
            };
            for k in p.row_ptr[r]..p.row_ptr[r + 1] {
                let j = p.row_col[k];
                if !mark[j] {
                    mark[j] = true;
                    cand.push(j);
                }
            }
        }
        for _ in 0..RANDOM_COLS.min(n) {
            let j = rng.below(n);
            if !mark[j] {
                mark[j] = true;
                cand.push(j);
            }
        }
        let mut best: Option<(usize, f64, f64)> = None;
        for idx in 0..cand.len() {
            let j = cand[idx];
            mark[j] = false;
            let (t, sc) = s.jump(j);
            if sc > 1e-12 && best.is_none_or(|b| sc > b.2) {
                best = Some((j, t, sc));
            }
        }
        match best {
            Some((j, t, _)) => {
                s.apply(j, t);
                moves += 1;
            }
            None => {
                for q in 0..s.viol.len() {
                    let r = s.viol[q];
                    s.w[r] += 1.0;
                }
                bumps += 1;
            }
        }
    }
    FjResult {
        x: None,
        moves,
        weight_bumps: bumps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(rows: &[Vec<(usize, f64)>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut p = vec![0];
        let mut i = Vec::new();
        let mut d = Vec::new();
        for r in rows {
            for &(j, v) in r {
                i.push(j);
                d.push(v);
            }
            p.push(i.len());
        }
        (p, i, d)
    }

    fn feasible(rows: &[Vec<(usize, f64)>], lo: &[f64], hi: &[f64], x: &[f64]) -> bool {
        rows.iter().zip(lo.iter().zip(hi)).all(|(r, (&l, &h))| {
            let a: f64 = r.iter().map(|&(j, v)| v * x[j]).sum();
            a >= l - 1e-6 && a <= h + 1e-6
        })
    }

    #[test]
    fn finds_a_point_of_a_small_knapsack_system() {
        // x0 + x1 + x2 = 2, 3x0 + 5x1 + 4x2 <= 8, x integer in [0, 1].
        let rows = vec![
            vec![(0, 1.0), (1, 1.0), (2, 1.0)],
            vec![(0, 3.0), (1, 5.0), (2, 4.0)],
        ];
        let (p, i, d) = csr(&rows);
        let lo = [2.0, f64::NEG_INFINITY];
        let hi = [2.0, 8.0];
        let r = fj_search(
            2, 3, &p, &i, &d, &lo, &hi, &[0.0; 3], &[1.0; 3], &[true; 3], 1, 10_000, None,
        );
        let x = r.x.expect("a feasible point exists and is easy");
        assert!(feasible(&rows, &lo, &hi, &x), "{x:?}");
        assert!(x.iter().all(|v| (v - v.round()).abs() < 1e-9));
    }

    #[test]
    fn an_infeasible_system_returns_no_point() {
        // x0 + x1 >= 3 with x binary: infeasible.
        let rows = vec![vec![(0, 1.0), (1, 1.0)]];
        let (p, i, d) = csr(&rows);
        let r = fj_search(
            1,
            2,
            &p,
            &i,
            &d,
            &[3.0],
            &[f64::INFINITY],
            &[0.0; 2],
            &[1.0; 2],
            &[true; 2],
            7,
            2_000,
            None,
        );
        assert!(r.x.is_none());
        assert!(r.moves + r.weight_bumps > 0, "the search ran");
    }

    #[test]
    fn continuous_columns_jump_to_the_breakpoint() {
        // x0 continuous in [0, 10], x0 >= 3.5.
        let rows = vec![vec![(0, 1.0)]];
        let (p, i, d) = csr(&rows);
        let r = fj_search(
            1,
            1,
            &p,
            &i,
            &d,
            &[3.5],
            &[f64::INFINITY],
            &[0.0],
            &[10.0],
            &[false],
            3,
            100,
            None,
        );
        let x = r.x.unwrap();
        assert!((x[0] - 3.5).abs() < 1e-9, "{x:?}");
    }
}
