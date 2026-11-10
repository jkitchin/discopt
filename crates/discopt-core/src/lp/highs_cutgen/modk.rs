//! `HighsModkSeparator` with `HighsGFkSolve` and `HighsSplay` (gap plan P3,
//! option A, §12.24): mod-k cuts from integer combinations of tight LP rows.
//!
//! Each tight row whose continuous columns all sit at a bound (in the transformed
//! LP) is scaled to integer coefficients over the integer columns with a positive
//! transformed value, and the system `sum_r w_r a_r = 0, sum_r w_r b_r = 1 (mod k)`
//! is solved in GF(k) for k = 2, 3, 5, 7. Each solution `w` names an aggregation of
//! rows; `generateCut` (with `onlyInitialCMIRScale`) runs on it twice: weighted
//! `(w (k-1) mod k) / k` and negated, then `w / k`. Validity is `generateCut`'s, as
//! for the path separator: the GF(k) solve only chooses the aggregation weights.
//!
//! Faithful to HiGHS v1.x including its tie-breaking: the elimination order comes
//! from a `std::priority_queue` comparing only column sizes (ported with the
//! libstdc++ heap), and the row iteration order from the splay trees.

use std::collections::HashSet;

use super::separators::{CutGenerator, CutSink, LpAggregator, SepContext};
use super::stdalgo::{pop_heap, push_heap};
use super::transformed_lp::TransformedLp;

// ------------------------------------------------------------------- splay

#[derive(Clone, Copy)]
enum Slot {
    NLeft,
    NRight,
    L(usize),
    R(usize),
}

/// The left/right child arrays of a forest of splay trees keyed by `key[pos]`.
struct Splay<'a> {
    left: &'a mut [i32],
    right: &'a mut [i32],
    key: &'a [i32],
}

impl Splay<'_> {
    fn set(&mut self, s: Slot, v: i32, nl: &mut i32, nr: &mut i32) {
        match s {
            Slot::NLeft => *nl = v,
            Slot::NRight => *nr = v,
            Slot::L(p) => self.left[p] = v,
            Slot::R(p) => self.right[p] = v,
        }
    }

    /// `highs_splay(key, root, ...)`: the new root.
    fn splay(&mut self, key: i32, mut root: i32) -> i32 {
        if root == -1 {
            return -1;
        }
        let (mut n_left, mut n_right) = (-1i32, -1i32);
        let mut lright = Slot::NRight;
        let mut rleft = Slot::NLeft;
        loop {
            let r = root as usize;
            if key < self.key[r] {
                let left = self.left[r];
                if left == -1 {
                    break;
                }
                if key < self.key[left as usize] {
                    let y = left as usize;
                    self.left[r] = self.right[y];
                    self.right[y] = root;
                    root = y as i32;
                    if self.left[y] == -1 {
                        break;
                    }
                }
                self.set(rleft, root, &mut n_left, &mut n_right);
                rleft = Slot::L(root as usize);
                root = self.left[root as usize];
            } else if key > self.key[r] {
                let right = self.right[r];
                if right == -1 {
                    break;
                }
                if key > self.key[right as usize] {
                    let y = right as usize;
                    self.right[r] = self.left[y];
                    self.left[y] = root;
                    root = y as i32;
                    if self.right[y] == -1 {
                        break;
                    }
                }
                self.set(lright, root, &mut n_left, &mut n_right);
                lright = Slot::R(root as usize);
                root = self.right[root as usize];
            } else {
                break;
            }
        }
        let r = root as usize;
        let (lr, rr) = (self.left[r], self.right[r]);
        self.set(lright, lr, &mut n_left, &mut n_right);
        self.set(rleft, rr, &mut n_left, &mut n_right);
        self.left[r] = n_right;
        self.right[r] = n_left;
        root
    }

    /// `highs_splay_link`.
    fn link(&mut self, node: usize, root: &mut i32) {
        if *root == -1 {
            self.left[node] = -1;
            self.right[node] = -1;
            *root = node as i32;
            return;
        }
        *root = self.splay(self.key[node], *root);
        let r = *root as usize;
        if self.key[node] < self.key[r] {
            self.left[node] = self.left[r];
            self.right[node] = *root;
            self.left[r] = -1;
        } else {
            debug_assert!(self.key[node] > self.key[r]);
            self.right[node] = self.right[r];
            self.left[node] = *root;
            self.right[r] = -1;
        }
        *root = node as i32;
    }

    /// `highs_splay_unlink`: the new root of the tree that held `node`.
    fn unlink(&mut self, node: usize, root: i32) -> i32 {
        debug_assert!(root != -1);
        let root = self.splay(self.key[node], root);
        debug_assert!(self.key[root as usize] == self.key[node]);
        if root != node as i32 {
            let sub = self.right[root as usize];
            let nr = self.unlink(node, sub);
            self.right[root as usize] = nr;
            return root;
        }
        if self.left[node] == -1 {
            self.right[node]
        } else {
            let nr = self.splay(self.key[node], self.left[node]);
            self.right[nr as usize] = self.right[node];
            nr
        }
    }
}

// ------------------------------------------------------------------- GF(k)

/// `HighsGFkSolve::SolutionEntry`: GF(k) column (an LP row slot) and its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SolutionEntry {
    /// Index into the integral system's rows (`integralScales`).
    pub index: i32,
    /// Weight in `1..k`.
    pub weight: u32,
}

fn gfk_inverse(k: u32, a: u32) -> u32 {
    match k {
        2 => 1,
        3 => a,
        _ => ((a as u64).pow(k - 2) % k as u64) as u32,
    }
}

/// `HighsGFkSolve` with one right-hand side.
#[derive(Default)]
pub struct GfkSolve {
    num_col: usize,
    num_row: usize,
    arow: Vec<i32>,
    acol: Vec<i32>,
    avalue: Vec<u32>,
    rowsize: Vec<i32>,
    colsize: Vec<i32>,
    colhead: Vec<i32>,
    anext: Vec<i32>,
    aprev: Vec<i32>,
    rowroot: Vec<i32>,
    arleft: Vec<i32>,
    arright: Vec<i32>,
    rhs: Vec<u32>,
    factor_col_perm: Vec<i32>,
    factor_row_perm: Vec<i32>,
    col_basis_status: Vec<i8>,
    row_used: Vec<i8>,
    iterstack: Vec<i32>,
    rowpositions: Vec<i32>,
    rowpos_colsizes: Vec<i32>,
    /// `std::priority_queue<HighsInt, vector, greater>`: distinct values, so any
    /// min-heap pops the same sequence.
    freeslots: std::collections::BinaryHeap<std::cmp::Reverse<i32>>,
}

impl GfkSolve {
    fn splay_of(&mut self) -> Splay<'_> {
        Splay {
            left: &mut self.arleft,
            right: &mut self.arright,
            key: &self.acol,
        }
    }

    fn link(&mut self, pos: usize) {
        let c = self.acol[pos] as usize;
        self.anext[pos] = self.colhead[c];
        self.aprev[pos] = -1;
        self.colhead[c] = pos as i32;
        if self.anext[pos] != -1 {
            let n = self.anext[pos] as usize;
            self.aprev[n] = pos as i32;
        }
        self.colsize[c] += 1;
        let r = self.arow[pos] as usize;
        let mut root = self.rowroot[r];
        self.splay_of().link(pos, &mut root);
        self.rowroot[r] = root;
        self.rowsize[r] += 1;
    }

    fn unlink(&mut self, pos: usize) {
        let (next, prev) = (self.anext[pos], self.aprev[pos]);
        if next != -1 {
            self.aprev[next as usize] = prev;
        }
        let c = self.acol[pos] as usize;
        if prev != -1 {
            self.anext[prev as usize] = next;
        } else {
            self.colhead[c] = next;
        }
        self.colsize[c] -= 1;
        let r = self.arow[pos] as usize;
        let root = self.rowroot[r];
        let nr = self.splay_of().unlink(pos, root);
        self.rowroot[r] = nr;
        self.rowsize[r] -= 1;
        self.avalue[pos] = 0;
        self.freeslots.push(std::cmp::Reverse(pos as i32));
    }

    fn drop_if_zero(&mut self, pos: usize) {
        if self.avalue[pos] == 0 {
            self.unlink(pos);
        }
    }

    fn store_row_positions(&mut self, pos: i32) {
        if pos == -1 {
            return;
        }
        debug_assert!(self.iterstack.is_empty());
        self.iterstack.push(pos);
        while let Some(p) = self.iterstack.pop() {
            let pu = p as usize;
            self.rowpositions.push(p);
            self.rowpos_colsizes
                .push(self.colsize[self.acol[pu] as usize]);
            if self.arleft[pu] != -1 {
                self.iterstack.push(self.arleft[pu]);
            }
            if self.arright[pu] != -1 {
                self.iterstack.push(self.arright[pu]);
            }
        }
    }

    /// `findNonzero(row, col)`: splays the row's tree.
    fn find_nonzero(&mut self, row: usize, col: i32) -> i32 {
        if self.rowroot[row] == -1 {
            return -1;
        }
        let root = self.rowroot[row];
        let nr = self.splay_of().splay(col, root);
        self.rowroot[row] = nr;
        if self.acol[nr as usize] == col {
            nr
        } else {
            -1
        }
    }

    fn add_nonzero(&mut self, row: usize, col: i32, val: u32) {
        debug_assert!(self.find_nonzero(row, col) == -1);
        let pos = if let Some(std::cmp::Reverse(p)) = self.freeslots.pop() {
            let p = p as usize;
            self.avalue[p] = val;
            self.arow[p] = row as i32;
            self.acol[p] = col;
            self.aprev[p] = -1;
            p
        } else {
            self.avalue.push(val);
            self.arow.push(row as i32);
            self.acol.push(col);
            self.anext.push(-1);
            self.aprev.push(-1);
            self.arleft.push(-1);
            self.arright.push(-1);
            self.avalue.len() - 1
        };
        self.link(pos);
    }

    /// `fromCSC<k>(Aval, Aindex, Astart, numRow)`.
    pub fn from_csc(
        &mut self,
        k: u32,
        aval: &[i64],
        aindex: &[i32],
        astart: &[i32],
        num_row: usize,
    ) {
        self.avalue.clear();
        self.acol.clear();
        self.arow.clear();
        self.freeslots.clear();
        self.num_col = astart.len() - 1;
        self.num_row = num_row;
        self.colhead = vec![-1; self.num_col];
        self.colsize = vec![0; self.num_col];
        self.rhs = vec![0; num_row];
        self.rowroot = vec![-1; num_row];
        self.rowsize = vec![0; num_row];
        for i in 0..self.num_col {
            for j in astart[i] as usize..astart[i + 1] as usize {
                let mut val = aval[j] % k as i64;
                if val == 0 {
                    continue;
                }
                if val < 0 {
                    val += k as i64;
                }
                self.avalue.push(val as u32);
                self.acol.push(i as i32);
                self.arow.push(aindex[j]);
            }
        }
        let nnz = self.avalue.len();
        self.anext = vec![0; nnz];
        self.aprev = vec![0; nnz];
        self.arleft = vec![0; nnz];
        self.arright = vec![0; nnz];
        for pos in 0..nnz {
            self.link(pos);
        }
    }

    /// `setRhs<k>(row, val)`.
    pub fn set_rhs(&mut self, k: u32, row: usize, val: i64) {
        self.rhs[row] = (val.unsigned_abs() as u32) % k;
    }

    /// `solve<k>(reportSolution)`.
    pub fn solve<F: FnMut(&mut Vec<SolutionEntry>)>(&mut self, k: u32, mut report: F) {
        // std::priority_queue with `a.first > b.first`.
        let mut pq: Vec<(i32, i32)> = Vec::new();
        let mut cmp = |a: &(i32, i32), b: &(i32, i32)| a.0 > b.0;
        for i in 0..self.num_col {
            pq.push((self.colsize[i], i as i32));
            push_heap(&mut pq, &mut cmp);
        }
        let max_pivot = self.num_row.min(self.num_col);
        self.factor_col_perm.clear();
        self.factor_row_perm.clear();
        self.col_basis_status = vec![0; self.num_col];
        self.row_used = vec![0; self.num_row];
        let mut num_pivot = 0usize;

        while !pq.is_empty() {
            let (old_col_size, pivot_col) = pq[0];
            pop_heap(&mut pq, &mut cmp);
            pq.pop();
            let pc = pivot_col as usize;
            if self.colsize[pc] == 0 {
                continue;
            }
            debug_assert!(self.col_basis_status[pc] == 0);
            if self.colsize[pc] != old_col_size {
                pq.push((self.colsize[pc], pivot_col));
                push_heap(&mut pq, &mut cmp);
                continue;
            }
            let mut pivot = -1i32;
            let mut pivot_row = -1i32;
            let mut pivot_row_len = i32::MAX;
            let mut it = self.colhead[pc];
            while it != -1 {
                let row = self.arow[it as usize];
                if self.row_used[row as usize] == 0 && self.rowsize[row as usize] < pivot_row_len {
                    pivot_row_len = self.rowsize[row as usize];
                    pivot_row = row;
                    pivot = it;
                }
                it = self.anext[it as usize];
            }
            debug_assert!(pivot != -1);
            let pivot_inverse = gfk_inverse(k, self.avalue[pivot as usize]);
            self.rowpositions.clear();
            self.rowpos_colsizes.clear();
            let pr = pivot_row as usize;
            self.store_row_positions(self.rowroot[pr]);
            let mut coliter = self.colhead[pc];
            while coliter != -1 {
                let next = self.anext[coliter as usize];
                if coliter == pivot {
                    coliter = next;
                    continue;
                }
                let row = self.arow[coliter as usize] as usize;
                if self.row_used[row] != 0 {
                    coliter = next;
                    continue;
                }
                let scale = pivot_inverse * (k - self.avalue[coliter as usize]);
                self.rhs[row] = (self.rhs[row] + scale * self.rhs[pr]) % k;
                for idx in 0..self.rowpositions.len() {
                    let prp = self.rowpositions[idx] as usize;
                    let pcol = self.acol[prp];
                    let nzpos = self.find_nonzero(row, pcol);
                    if nzpos == -1 {
                        let val = (scale * self.avalue[prp]) % k;
                        if val != 0 {
                            self.add_nonzero(row, pcol, val);
                        }
                    } else {
                        let np = nzpos as usize;
                        self.avalue[np] = (self.avalue[np] + scale * self.avalue[prp]) % k;
                        self.drop_if_zero(np);
                    }
                }
                coliter = next;
            }
            num_pivot += 1;
            self.factor_col_perm.push(pivot_col);
            self.factor_row_perm.push(pivot_row);
            self.col_basis_status[pc] = 1;
            self.row_used[pr] = 1;
            if num_pivot == max_pivot {
                break;
            }
            for i in 0..pivot_row_len as usize {
                let col = self.acol[self.rowpositions[i] as usize] as usize;
                let oldsize = self.rowpos_colsizes[i];
                self.colsize[col] -= 1;
                if self.colsize[col] == 0 {
                    continue;
                }
                if self.colsize[col] < oldsize {
                    pq.push((self.colsize[col], col as i32));
                    push_heap(&mut pq, &mut cmp);
                }
            }
        }

        for i in 0..self.num_row {
            if self.row_used[i] == 0 && self.rhs[i] != 0 {
                return;
            }
        }

        let mut solution: Vec<SolutionEntry> = Vec::with_capacity(self.num_col);
        let num_factor_rows = self.factor_row_perm.len();
        let mut basis_swaps: Vec<(usize, i32)> = Vec::new();
        debug_assert!(self.iterstack.is_empty());
        for i in (0..num_factor_rows).rev() {
            let row = self.factor_row_perm[i] as usize;
            self.iterstack.push(self.rowroot[row]);
            while let Some(rowpos) = self.iterstack.pop() {
                let rp = rowpos as usize;
                if self.arleft[rp] != -1 {
                    self.iterstack.push(self.arleft[rp]);
                }
                if self.arright[rp] != -1 {
                    self.iterstack.push(self.arright[rp]);
                }
                let col = self.acol[rp] as usize;
                if self.col_basis_status[col] != 0 {
                    continue;
                }
                self.col_basis_status[col] = -1;
                basis_swaps.push((i, col as i32));
            }
        }

        let mut swap_pos = 0usize;
        loop {
            solution.clear();
            for i in (0..num_factor_rows).rev() {
                let row = self.factor_row_perm[i] as usize;
                let mut solval: u32 = 0;
                for e in 0..solution.len() {
                    let se = solution[e];
                    let pos = self.find_nonzero(row, se.index);
                    if pos != -1 {
                        solval += self.avalue[pos as usize] * se.weight;
                    }
                }
                solval = self.rhs[row] + k - (solval % k);
                let col = self.factor_col_perm[i];
                let pos = self.find_nonzero(row, col);
                debug_assert!(pos != -1);
                let inv = gfk_inverse(k, self.avalue[pos as usize]);
                solval = (solval * inv) % k;
                if solval != 0 {
                    solution.push(SolutionEntry {
                        index: col,
                        weight: solval,
                    });
                }
            }
            report(&mut solution);
            if swap_pos < basis_swaps.len() {
                let (bi, entering) = basis_swaps[swap_pos];
                let leaving = self.factor_col_perm[bi];
                self.factor_col_perm[bi] = entering;
                self.col_basis_status[entering as usize] = 1;
                self.col_basis_status[leaving as usize] = 0;
                swap_pos += 1;
            } else {
                break;
            }
        }
    }
}

// --------------------------------------------------------------- separator

/// `HighsModkSeparator::separateLpSolution`. `cg_seed` seeds the call's
/// `HighsCutGeneration` (`random_seed + LP iterations + pool size`).
pub fn separate_modk(
    ctx: &dyn SepContext,
    agg: &mut LpAggregator,
    tl: &mut TransformedLp,
    pool: &mut dyn CutSink,
    cg_seed: i64,
) {
    let num_row = ctx.num_rows();
    let num_col = ctx.num_cols();
    let feastol = ctx.feastol();
    let epsilon = ctx.epsilon();
    let (astart, aindex, _) = ctx.col_matrix();
    let mut skip_row = vec![false; num_row];
    for &col in ctx.continuous_cols() {
        if tl.bound_distance(col) == 0.0 {
            continue;
        }
        for kk in astart[col as usize]..astart[col as usize + 1] {
            skip_row[aindex[kk as usize] as usize] = true;
        }
    }
    let mut cut_gen = CutGenerator::new(ctx, cg_seed);
    let is_continuous = |c: i32| !ctx.is_col_integral(c);

    let mut integral_scales: Vec<(i32, f64)> = Vec::new();
    let mut int_value: Vec<i64> = Vec::new();
    let mut int_index: Vec<i32> = Vec::new();
    let mut int_start: Vec<i32> = vec![0];
    let row_value = ctx.row_value();
    let mut num_nonzero_rhs = 0usize;
    let max_int_row_len = (1000.0 + 0.1 * num_col as f64) as usize;

    for row in 0..num_row {
        if skip_row[row] {
            continue;
        }
        let (lo, up) = (ctx.row_lower(row as i32), ctx.row_upper(row as i32));
        let leq = if up - row_value[row] <= feastol {
            true
        } else if row_value[row] - lo <= feastol {
            false
        } else {
            continue;
        };
        let (ri, rv) = ctx.row(row as i32);
        let (rhs, vals): (f64, Vec<f64>) = if leq {
            (up, rv.to_vec())
        } else {
            (-lo, rv.iter().map(|x| -x).collect())
        };
        let Some(t) = tl.transform(ctx, epsilon, ri.to_vec(), vals, rhs, false, true) else {
            continue;
        };
        let rowlen = t.inds.len();
        if rowlen > max_int_row_len {
            let mut int_row_len = 0usize;
            for i in 0..rowlen {
                if t.solval[i] <= feastol || is_continuous(t.inds[i]) {
                    continue;
                }
                int_row_len += 1;
            }
            if int_row_len > max_int_row_len || (int_row_len == 0 && t.rhs.abs() <= epsilon) {
                continue;
            }
        }
        let intscale;
        let intrhs;
        if !ctx.is_col_integral((num_col + row) as i32) {
            let mut scale_vals: Vec<f64> = Vec::new();
            for i in 0..rowlen {
                if is_continuous(t.inds[i]) {
                    continue;
                }
                if t.solval[i] > feastol {
                    scale_vals.push(t.vals[i]);
                }
            }
            if t.rhs.abs() > epsilon {
                scale_vals.push(-t.rhs);
            }
            if scale_vals.is_empty() {
                continue;
            }
            intscale = super::integers::integral_scale(&scale_vals, feastol, epsilon);
            if intscale == 0.0 || intscale > 1e6 {
                continue;
            }
            intrhs = super::integers::nearest_integer(intscale * t.rhs);
            for i in 0..rowlen {
                if is_continuous(t.inds[i]) {
                    continue;
                }
                if t.solval[i] > feastol {
                    int_index.push(t.inds[i]);
                    int_value.push(super::integers::nearest_integer(intscale * t.vals[i]));
                }
            }
        } else {
            intscale = 1.0;
            intrhs = super::integers::nearest_integer(t.rhs);
            for i in 0..rowlen {
                if t.solval[i] > feastol {
                    int_index.push(t.inds[i]);
                    int_value.push(super::integers::nearest_integer(t.vals[i]));
                }
            }
        }
        if intrhs != 0 {
            num_nonzero_rhs += 1;
        }
        int_index.push(num_col as i32);
        int_value.push(intrhs);
        int_start.push(int_value.len() as i32);
        integral_scales.push((row as i32, intscale));
    }

    if integral_scales.is_empty() || num_nonzero_rhs == 0 {
        return;
    }

    for k in [2u32, 3, 5, 7] {
        let num_cuts = pool.num_cuts();
        let mut used: HashSet<Vec<SolutionEntry>> = HashSet::new();
        let mut gfk = GfkSolve::default();
        gfk.from_csc(k, &int_value, &int_index, &int_start, num_col + 1);
        gfk.set_rhs(k, num_col, 1);
        gfk.solve(k, |weights: &mut Vec<SolutionEntry>| {
            if weights.is_empty() {
                return;
            }
            // pdqsort by index; indices are distinct, so any sort agrees.
            weights.sort_unstable_by_key(|w| w.index);
            if !used.insert(weights.clone()) {
                return;
            }
            debug_assert!(agg.is_empty());
            for w in weights.iter() {
                let (row, scale) = integral_scales[w.index as usize];
                let weight = scale * (((w.weight * (k - 1)) % k) as f64 / k as f64);
                agg.add_row(ctx, row, weight);
            }
            let (inds, vals) = agg.current_aggregation(ctx, false);
            cut_gen.generate_cut(ctx, tl, pool, inds, vals, 0.0, true);
            if k != 2 {
                agg.clear();
                for w in weights.iter() {
                    let (row, scale) = integral_scales[w.index as usize];
                    agg.add_row(ctx, row, scale * (w.weight as f64 / k as f64));
                }
            }
            let (inds, vals) = agg.current_aggregation(ctx, true);
            cut_gen.generate_cut(ctx, tl, pool, inds, vals, 0.0, true);
            agg.clear();
        });
        if pool.num_cuts() != num_cuts {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force check: every reported GF(k) solution satisfies the system.
    #[test]
    fn gfk_solutions_satisfy_the_system() {
        let mut checked = 0usize;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as i64
        };
        for k in [2u32, 3, 5, 7] {
            for _ in 0..200 {
                let ncols = 1 + (rnd() % 8) as usize; // system columns = LP rows
                let nrows = 1 + (rnd() % 6) as usize; // + rhs row
                let mut aval = Vec::new();
                let mut aidx = Vec::new();
                let mut astart = vec![0i32];
                let mut dense = vec![vec![0i64; ncols]; nrows + 1];
                for c in 0..ncols {
                    for r in 0..=nrows {
                        if rnd() % 3 == 0 {
                            let v = rnd() % 11 - 5;
                            if v != 0 {
                                aval.push(v);
                                aidx.push(r as i32);
                                dense[r][c] = v;
                            }
                        }
                    }
                    astart.push(aval.len() as i32);
                }
                let mut g = GfkSolve::default();
                g.from_csc(k, &aval, &aidx, &astart, nrows + 1);
                g.set_rhs(k, nrows, 1);
                g.solve(k, |w: &mut Vec<SolutionEntry>| {
                    for r in 0..=nrows {
                        let mut s: i64 = 0;
                        for e in w.iter() {
                            s += dense[r][e.index as usize] * e.weight as i64;
                        }
                        let want = if r == nrows { 1 } else { 0 };
                        assert_eq!(s.rem_euclid(k as i64), want, "k={k} row {r}");
                    }
                    checked += 1;
                });
            }
        }
        assert!(checked > 100, "only {checked} solutions checked");
    }
}
