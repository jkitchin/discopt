//! Pre-root primal heuristics: HiGHS's trivial points and the Feasibility Jump
//! heuristic.
//!
//! Provenance. `FeasJump` and its helpers are a translation of HiGHS's
//! `highs/mip/feasibilityjump.hh` (the Feasibility Jump heuristic; the file's
//! header places it in the HiGHS suite under the MIT License) and of the driver
//! half of `highs/mip/HighsFeasibilityJump.cpp`; [`trivial_points`] translates
//! `HighsMipSolverData::trivialHeuristics` (`highs/mip/HighsMipSolverData.cpp`).
//! HiGHS: https://github.com/ERGO-Code/HiGHS, MIT License, "Copyright (c) 2026
//! HiGHS"; the license text is in `THIRD_PARTY_NOTICES.md`. The Mersenne Twister
//! and the canonical-uniform draw are written from the published MT19937
//! algorithm (Matsumoto & Nishimura 1998) and the C++ standard's specification
//! of `generate_canonical` ([rand.util.canonical]), not from any C++ library
//! source, so the random sequence HiGHS's solver consumes can be replayed.
//!
//! Why here: HiGHS runs both before the root LP (`HighsMipSolver.cpp:187-210`).
//! Measured on the 120-row MIPLIB population at 20 s (2026-11-14,
//! `highs-mip-replacement-plan.md` §9): of the 29 rows where HiGHS has a first
//! incumbent and discopt has none, feasibility jump found it on 12 and a trivial
//! point on 6, most within 0.1-3.4 s; discopt had neither heuristic.
//!
//! Soundness: nothing here is trusted. A point is a candidate only; the caller
//! re-verifies every row, bound and integrality (`validate_seed_incumbent`) before
//! it becomes an incumbent.

/// MT19937, the 32-bit Mersenne Twister (`std::mt19937`).
pub(crate) struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub(crate) fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
            let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.mt[i] = v;
        }
        self.idx = 0;
    }

    /// The next 32-bit output (`rng()`).
    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `rng() % n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    /// `std::uniform_real_distribution<double>(0, 1)(rng)`: the standard's
    /// `generate_canonical<double, 53>` over a 32-bit engine takes
    /// `k = ceil(53 / 32) = 2` draws, `S = g0 + g1·2³²`, and returns `S / 2⁶⁴`
    /// (clamped below 1, the LWG 2524 resolution).
    fn uniform01(&mut self) -> f64 {
        let g0 = self.next_u32() as f64;
        let g1 = self.next_u32() as f64;
        let r = 4_294_967_296.0_f64;
        let s = g0 + g1 * r;
        let v = s / (r * r);
        if v >= 1.0 {
            1.0 - f64::EPSILON / 2.0
        } else {
            v
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowType {
    Lte,
    Gte,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VarType {
    Continuous,
    Integer,
}

fn eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

struct Var {
    vartype: VarType,
    lb: f64,
    ub: f64,
    obj: f64,
    /// `(constraint, coefficient)`, in the order constraints were added.
    coeffs: Vec<(u32, f64)>,
}

struct Constraint {
    sense: RowType,
    rhs: f64,
    coeffs: Vec<(u32, f64)>,
    weight: f64,
    lhs: f64,
    violated_idx: i32,
}

impl Constraint {
    /// 0 when `lhs` satisfies the row, `-|violation|` otherwise.
    fn score(&self, lhs: f64) -> f64 {
        match self.sense {
            RowType::Lte => -(0.0f64.max(lhs - self.rhs)),
            RowType::Gte => -(0.0f64.max(self.rhs - lhs)),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Move {
    value: f64,
    score: f64,
}

/// Outcome of a [`FeasJump`] run.
pub(crate) struct FjResult {
    /// The best feasible assignment found (all variables), if any.
    pub(crate) solution: Option<Vec<f64>>,
    /// Effort units spent (the solver's own work counter).
    pub(crate) effort: usize,
}

/// The feasibility-jump solver (`external_feasibilityjump::FeasibilityJumpSolver`
/// with the `Problem` and `JumpMove` it owns), for `RowType::{Lte, Gte}` rows with
/// `relax_continuous = 0`, the only configuration HiGHS uses.
pub(crate) struct FeasJump {
    eq_tol: f64,
    viol_tol: f64,
    vars: Vec<Var>,
    cons: Vec<Constraint>,
    x: Vec<f64>,
    violated: Vec<u32>,
    nnz: usize,
    objective: f64,
    moves: Vec<Move>,
    shift_buf: Vec<(f64, f64)>,
    good: Vec<u32>,
    good_idx: Vec<i32>,
    rng: Mt19937,
    best_objective: f64,
    objective_weight: f64,
    best_violation: usize,
    effort_at_last_callback: usize,
    effort_at_last_improvement: usize,
    total_effort: usize,
    weight_update_increment: f64,
}

const RANDOM_VAR_PROBABILITY: f64 = 0.001;
const RANDOM_CELL_PROBABILITY: f64 = 0.01;
const MAX_MOVES_TO_EVALUATE: usize = 25;
const MIN_RELATIVE_OBJECTIVE_IMPROVEMENT: f64 = 1e-4;
const CALLBACK_EFFORT: usize = 500_000;
const WEIGHT_UPDATE_DECAY: f64 = 1.0;

/// What the solve's control callback decides (`CallbackControlFlow`).
pub(crate) enum Flow {
    Continue,
    Terminate,
}

impl FeasJump {
    pub(crate) fn new(seed: u32, eq_tol: f64, viol_tol: f64) -> Self {
        FeasJump {
            eq_tol,
            viol_tol,
            vars: Vec::new(),
            cons: Vec::new(),
            x: Vec::new(),
            violated: Vec::new(),
            nnz: 0,
            objective: f64::NAN,
            moves: Vec::new(),
            shift_buf: Vec::new(),
            good: Vec::new(),
            good_idx: Vec::new(),
            rng: Mt19937::new(seed),
            best_objective: f64::INFINITY,
            objective_weight: 0.0,
            best_violation: usize::MAX,
            effort_at_last_callback: 0,
            effort_at_last_improvement: 0,
            total_effort: 0,
            weight_update_increment: 1.0,
        }
    }

    pub(crate) fn add_var(&mut self, integer: bool, lb: f64, ub: f64, obj: f64) {
        self.good_idx.push(-1);
        self.vars.push(Var {
            vartype: if integer {
                VarType::Integer
            } else {
                VarType::Continuous
            },
            lb,
            ub,
            obj,
            coeffs: Vec::new(),
        });
        self.x.push(lb);
    }

    /// `addConstraint(sense, rhs, ...)` for `a·x <= rhs` (`upper`) or
    /// `a·x >= rhs`. Returns `false` when the row is empty and violated (the
    /// caller's model is infeasible), as the C++ returns `INT_MIN`.
    pub(crate) fn add_constraint(
        &mut self,
        upper: bool,
        rhs: f64,
        idx: &[usize],
        val: &[f64],
    ) -> bool {
        let sense = if upper { RowType::Lte } else { RowType::Gte };
        let coeffs: Vec<(u32, f64)> = idx.iter().zip(val).map(|(&j, &v)| (j as u32, v)).collect();
        if coeffs.is_empty() {
            return match sense {
                RowType::Lte => 0.0 <= rhs + self.eq_tol,
                RowType::Gte => self.eq_tol >= rhs,
            };
        }
        let ci = self.cons.len() as u32;
        for &(j, v) in &coeffs {
            self.vars[j as usize].coeffs.push((ci, v));
        }
        self.nnz += coeffs.len();
        self.cons.push(Constraint {
            sense,
            rhs,
            coeffs,
            weight: 1.0,
            lhs: f64::NAN,
            violated_idx: -1,
        });
        true
    }

    fn reset_incumbent(&mut self, initial: &[f64]) {
        self.x.copy_from_slice(initial);
        self.objective = 0.0;
        for (v, &xv) in self.vars.iter().zip(&self.x) {
            self.objective += v.obj * xv;
        }
        self.violated.clear();
        for ci in 0..self.cons.len() {
            let mut lhs = 0.0;
            for &(j, a) in &self.cons[ci].coeffs {
                lhs += a * self.x[j as usize];
            }
            let c = &mut self.cons[ci];
            c.lhs = lhs;
            if c.score(lhs) < -self.viol_tol {
                c.violated_idx = self.violated.len() as i32;
                self.violated.push(ci as u32);
            } else {
                c.violated_idx = -1;
            }
        }
    }

    /// Run until the callback says stop. `initial` is the starting assignment.
    /// The callback receives `(total effort, effort since last improvement,
    /// new solution if any, its objective)`.
    pub(crate) fn solve(
        &mut self,
        initial: &[f64],
        mut callback: impl FnMut(usize, usize, Option<&[f64]>, f64) -> Flow,
    ) -> FjResult {
        let mut best: Option<Vec<f64>> = None;
        self.init(initial);
        let mut step: u64 = 0;
        while step < i32::MAX as u64 {
            if self.user_terminate(&mut callback, None) {
                break;
            }
            if self.violated.len() < self.best_violation {
                self.effort_at_last_improvement = self.total_effort;
                self.best_violation = self.violated.len();
            }
            if self.violated.is_empty() && self.objective < self.best_objective {
                let rel = (self.best_objective - self.objective) / 1.0f64.max(self.objective.abs());
                if rel > MIN_RELATIVE_OBJECTIVE_IMPROVEMENT {
                    self.effort_at_last_improvement = self.total_effort;
                    self.best_objective = self.objective;
                    best = Some(self.x.clone());
                    let sol = self.x.clone();
                    if self.user_terminate(&mut callback, Some(&sol)) {
                        break;
                    }
                }
            }
            if self.vars.is_empty() {
                break;
            }
            let v = self.select_variable();
            self.do_variable_move(v);
            step += 1;
        }
        FjResult {
            solution: best,
            effort: self.total_effort,
        }
    }

    fn init(&mut self, initial: &[f64]) {
        self.reset_incumbent(initial);
        self.moves = vec![Move::default(); self.vars.len()];
        self.total_effort += self.nnz;
        self.good.clear();
        for i in 0..self.vars.len() {
            self.reset_moves(i);
        }
    }

    fn user_terminate(
        &mut self,
        callback: &mut impl FnMut(usize, usize, Option<&[f64]>, f64) -> Flow,
        solution: Option<&[f64]>,
    ) -> bool {
        if solution.is_some() || self.total_effort - self.effort_at_last_callback > CALLBACK_EFFORT
        {
            self.effort_at_last_callback = self.total_effort;
            let since = self.total_effort - self.effort_at_last_improvement;
            matches!(
                callback(self.total_effort, since, solution, self.objective),
                Flow::Terminate
            )
        } else {
            false
        }
    }

    fn select_variable(&mut self) -> usize {
        if !self.good.is_empty() {
            if self.rng.uniform01() < RANDOM_VAR_PROBABILITY {
                let k = self.rng.below(self.good.len());
                return self.good[k] as usize;
            }
            let sample = MAX_MOVES_TO_EVALUATE.min(self.good.len());
            self.total_effort += sample;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_var = usize::MAX;
            for _ in 0..sample {
                let k = self.rng.below(self.good.len());
                let v = self.good[k] as usize;
                let s = self.moves[v].score;
                if s > best_score {
                    best_score = s;
                    best_var = v;
                }
            }
            debug_assert!(best_var != usize::MAX);
            return best_var;
        }
        self.update_weights();
        if !self.violated.is_empty() {
            let ci = self.violated[self.rng.below(self.violated.len())] as usize;
            if self.rng.uniform01() < RANDOM_CELL_PROBABILITY {
                let k = self.rng.below(self.cons[ci].coeffs.len());
                return self.cons[ci].coeffs[k].0 as usize;
            }
            let mut best_score = f64::NEG_INFINITY;
            // The C++ leaves `UINT_MAX` when no cell scores above -inf; the
            // first cell is the defined stand-in for that unreachable case.
            let mut best_var = self.cons[ci].coeffs[0].0 as usize;
            for &(j, _) in &self.cons[ci].coeffs {
                let s = self.moves[j as usize].score;
                if s > best_score {
                    best_score = s;
                    best_var = j as usize;
                }
            }
            return best_var;
        }
        self.rng.below(self.vars.len())
    }

    fn update_weights(&mut self) {
        let mut rescale = false;
        let mut dt = 0usize;
        if self.violated.is_empty() {
            self.objective_weight += self.weight_update_increment;
            if self.objective_weight > 1.0e20 {
                rescale = true;
            }
            dt += self.vars.len();
            for (vi, var) in self.vars.iter().enumerate() {
                let m = &mut self.moves[vi];
                m.score += self.weight_update_increment * var.obj * (m.value - self.x[vi]);
            }
        } else {
            for k in 0..self.violated.len() {
                let ci = self.violated[k] as usize;
                self.cons[ci].weight += self.weight_update_increment;
                if self.cons[ci].weight > 1.0e20 {
                    rescale = true;
                }
                dt += self.cons[ci].coeffs.len();
                for p in 0..self.cons[ci].coeffs.len() {
                    let (j, a) = self.cons[ci].coeffs[p];
                    let j = j as usize;
                    let c = &self.cons[ci];
                    let m = &mut self.moves[j];
                    let cand = c.lhs + a * (m.value - self.x[j]);
                    m.score += self.weight_update_increment * (c.score(cand) - c.score(c.lhs));
                    self.update_good_moves(j);
                }
            }
        }
        self.weight_update_increment /= WEIGHT_UPDATE_DECAY;
        if rescale {
            self.weight_update_increment *= 1.0e-20;
            self.objective_weight *= 1.0e-20;
            for c in &mut self.cons {
                c.weight *= 1.0e-20;
            }
            dt += self.cons.len();
            for i in 0..self.vars.len() {
                self.reset_moves(i);
            }
        }
        self.total_effort += dt;
    }

    fn do_variable_move(&mut self, vi: usize) {
        let new_value = self.moves[vi].value;
        // `Problem::setValue` with the `modifyMove` / `updateGoodMoves` callback.
        let old_value = self.x[vi];
        let delta = new_value - old_value;
        self.x[vi] = new_value;
        self.objective += self.vars[vi].obj * delta;
        let mut dt = 0usize;
        for p in 0..self.vars[vi].coeffs.len() {
            let (ci, a) = self.vars[vi].coeffs[p];
            let ci = ci as usize;
            let old_lhs = self.cons[ci].lhs;
            let new_lhs = old_lhs + a * delta;
            self.cons[ci].lhs = new_lhs;
            let new_cost = self.cons[ci].score(new_lhs);
            if new_cost < -self.viol_tol && self.cons[ci].violated_idx == -1 {
                self.cons[ci].violated_idx = self.violated.len() as i32;
                self.violated.push(ci as u32);
            }
            if new_cost >= -self.viol_tol && self.cons[ci].violated_idx != -1 {
                let last = self.violated.len() - 1;
                let last_ci = self.violated[last] as usize;
                let this = self.cons[ci].violated_idx as usize;
                self.violated.swap(this, last);
                self.cons[last_ci].violated_idx = this as i32;
                self.cons[ci].violated_idx = -1;
                self.violated.pop();
            }
            dt += self.cons[ci].coeffs.len();
            for q in 0..self.cons[ci].coeffs.len() {
                let (j, cj) = self.cons[ci].coeffs[q];
                let j = j as usize;
                if j != vi {
                    // `modifyMove`.
                    let c = &self.cons[ci];
                    let inc = self.x[j];
                    let m = &mut self.moves[j];
                    let old_mod = old_lhs + cj * (m.value - inc);
                    let old_term = c.weight * (c.score(old_mod) - c.score(old_lhs));
                    let new_mod = new_lhs + cj * (m.value - inc);
                    let new_term = c.weight * (c.score(new_mod) - c.score(new_lhs));
                    m.score += new_term - old_term;
                    self.update_good_moves(j);
                }
            }
        }
        self.total_effort += dt;
        self.reset_moves(vi);
    }

    fn update_good_moves(&mut self, vi: usize) {
        let any_good = self.moves[vi].score > 0.0;
        if any_good && self.good_idx[vi] == -1 {
            self.good_idx[vi] = self.good.len() as i32;
            self.good.push(vi as u32);
        } else if !any_good && self.good_idx[vi] != -1 {
            let last = self.good.len() - 1;
            let last_v = self.good[last] as usize;
            let this = self.good_idx[vi] as usize;
            self.good.swap(this, last);
            self.good_idx[last_v] = this as i32;
            self.good_idx[vi] = -1;
            self.good.pop();
        }
    }

    fn reset_moves(&mut self, vi: usize) {
        self.total_effort += self.vars[vi].coeffs.len();
        self.update_jump_value(vi);
        let var = &self.vars[vi];
        let m = &mut self.moves[vi];
        m.score = 0.0;
        m.score += self.objective_weight * var.obj * (m.value - self.x[vi]);
        for &(ci, a) in &var.coeffs {
            let c = &self.cons[ci as usize];
            let cand = c.lhs + a * (m.value - self.x[vi]);
            m.score += c.weight * (c.score(cand) - c.score(c.lhs));
        }
        self.update_good_moves(vi);
    }

    /// `JumpMove::updateValue`: the value minimizing the weighted violation of
    /// `vi`'s rows over its range, by a sweep of the piecewise-linear breakpoints.
    ///
    /// Faithful to HiGHS, including that a row's valid range is not swapped for a
    /// negative coefficient (an inverted range is skipped, `first > second`).
    fn update_jump_value(&mut self, vi: usize) {
        self.shift_buf.clear();
        let var = &self.vars[vi];
        let inc = self.x[vi];
        let (lower, upper) = (var.lb, var.ub);
        let current_start = lower;
        let mut slope = 0.0;
        for &(ci, a) in &var.coeffs {
            let c = &self.cons[ci as usize];
            let bound = match c.sense {
                RowType::Lte => (f64::NEG_INFINITY, c.rhs),
                RowType::Gte => (c.rhs, f64::INFINITY),
            };
            let residual = c.lhs - a * inc;
            let mut lo = (1.0 / a) * (bound.0 - residual);
            let mut hi = (1.0 / a) * (bound.1 - residual);
            if var.vartype == VarType::Integer {
                lo = (lo - self.eq_tol).ceil();
                hi = (hi + self.eq_tol).floor();
            }
            if lo > hi {
                continue;
            }
            if lo > current_start {
                slope -= c.weight;
                if lo < upper {
                    self.shift_buf.push((lo, c.weight));
                }
            }
            if hi <= current_start {
                slope += c.weight;
            } else if hi < upper {
                self.shift_buf.push((hi, c.weight));
            }
        }
        if lower.is_finite() {
            self.shift_buf.push((lower, 0.0));
        }
        if upper.is_finite() {
            self.shift_buf.push((upper, 0.0));
        }
        self.shift_buf.sort_by(|p, q| {
            p.0.partial_cmp(&q.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(p.1.partial_cmp(&q.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut score = 0.0;
        let mut value = if self.shift_buf.is_empty() {
            inc
        } else {
            self.shift_buf[0].0
        };
        let mut best_score = score;
        let mut best_value = value;
        for &(at, w) in &self.shift_buf {
            score += (at - value) * slope;
            slope += w;
            value = at;
            if eq(best_value, inc, self.eq_tol)
                || (!eq(value, inc, self.eq_tol) && score < best_score)
            {
                best_score = score;
                best_value = value;
            }
            if !eq(best_value, inc, self.eq_tol) && slope >= 0.0 {
                break;
            }
        }
        self.moves[vi].value = best_value;
    }
}

/// A row-form MILP: `lo_i <= Σ a_ij x_j <= up_i` over `ncols` columns with
/// bounds `[l_j, u_j]`, row-major.
pub(crate) struct RowModel<'a> {
    pub(crate) row_ptr: &'a [usize],
    pub(crate) row_idx: &'a [usize],
    pub(crate) row_val: &'a [f64],
    pub(crate) lo: &'a [f64],
    pub(crate) up: &'a [f64],
    pub(crate) l: &'a [f64],
    pub(crate) u: &'a [f64],
    pub(crate) c: &'a [f64],
    pub(crate) is_int: &'a [bool],
}

/// `HighsMipSolverData::feasibilityJump`: build the solver from `model`, start
/// from `incumbent` (clamped into the bounds) or each column's finite bound, and
/// stop at `nnz << 10` total effort or `nnz << 8` without improvement, as HiGHS
/// does, or when `stop()` says so (discopt's addition: the solve deadline, checked
/// at the solver's callback points). Returns the best point found, if any.
pub(crate) fn feasibility_jump(
    model: &RowModel<'_>,
    incumbent: Option<&[f64]>,
    mut stop: impl FnMut() -> bool,
) -> Option<(Vec<f64>, usize)> {
    const EPSILON: f64 = 1e-9; // HiGHS `small_matrix_value`
    const FEASTOL: f64 = 1e-6; // HiGHS `mip_feasibility_tolerance`
    let n = model.l.len();
    let mut fj = FeasJump::new(0, EPSILON, FEASTOL);
    let mut x0 = vec![0.0; n];
    for j in 0..n {
        let (mut lower, mut upper) = (model.l[j], model.u[j]);
        if model.is_int[j] {
            lower = (lower - FEASTOL).ceil();
            upper = (upper + FEASTOL).floor();
        }
        let legal = lower <= upper && lower < f64::INFINITY && upper > f64::NEG_INFINITY;
        if !legal {
            return None;
        }
        fj.add_var(model.is_int[j], lower, upper, model.c[j]);
        x0[j] = match incumbent.map(|s| s[j]).filter(|v| v.is_finite()) {
            Some(v) => lower.max(upper.min(v)),
            None if lower.is_finite() => lower,
            None if upper.is_finite() => upper,
            None => 0.0,
        };
    }
    for i in 0..model.lo.len() {
        let (s, e) = (model.row_ptr[i], model.row_ptr[i + 1]);
        let (idx, val) = (&model.row_idx[s..e], &model.row_val[s..e]);
        if model.lo[i].is_finite() && !fj.add_constraint(false, model.lo[i], idx, val) {
            return None;
        }
        if model.up[i].is_finite() && !fj.add_constraint(true, model.up[i], idx, val) {
            return None;
        }
    }
    let nnz = model.row_val.len();
    let max_total = nnz << 10;
    let max_since = nnz << 8;
    let res = fj.solve(&x0, |total, since, _sol, _obj| {
        if since > max_since || total > max_total || stop() {
            Flow::Terminate
        } else {
            Flow::Continue
        }
    });
    res.solution.map(|x| (x, res.effort))
}

/// `HighsMipSolverData::trivialHeuristics`: for a pure-integer model, the
/// all-zero point (when every integer lower bound is <= 0 and every row admits a
/// zero activity), all-lower (when distinct from zero), all-upper (when every upper
/// bound is finite) and the lock point (each column at the bound its coefficient
/// signs favour), each kept when its rows are feasible to `FEASTOL`. Returns the
/// candidates in that order; the caller validates and keeps the best.
pub(crate) fn trivial_points(model: &RowModel<'_>) -> Vec<Vec<f64>> {
    const FEASTOL: f64 = 1e-6;
    let n = model.l.len();
    if model.is_int.iter().any(|&b| !b) {
        return Vec::new();
    }
    let mut l = model.l.to_vec();
    let mut u = model.u.to_vec();
    let (mut lower_non_pos, mut lower_zero, mut lower_finite, mut upper_finite) =
        (true, true, true, true);
    for j in 0..n {
        l[j] = l[j].ceil();
        u[j] = u[j].floor();
        if !(l[j] <= u[j] && l[j] < f64::INFINITY && u[j] > f64::NEG_INFINITY) {
            return Vec::new();
        }
        if l[j] > 0.0 {
            lower_non_pos = false;
        }
        if l[j] != 0.0 {
            lower_zero = false;
        }
        if l[j] == f64::NEG_INFINITY {
            lower_finite = false;
        }
        if u[j] == f64::INFINITY {
            upper_finite = false;
        }
        if !(lower_non_pos || lower_zero || upper_finite) {
            break;
        }
    }
    let rows_ok = |x: &[f64]| {
        (0..model.lo.len()).all(|i| {
            let (s, e) = (model.row_ptr[i], model.row_ptr[i + 1]);
            let act: f64 = (s..e).map(|k| model.row_val[k] * x[model.row_idx[k]]).sum();
            act >= model.lo[i] - FEASTOL && act <= model.up[i] + FEASTOL
        })
    };
    let mut out = Vec::new();
    // All zero: a zero activity must be feasible for every row.
    if lower_non_pos
        && model
            .lo
            .iter()
            .zip(model.up)
            .all(|(&lo, &up)| lo <= FEASTOL && up >= -FEASTOL)
    {
        out.push(vec![0.0; n]);
    }
    if !lower_zero && rows_ok(&l) {
        out.push(l.clone());
    }
    if upper_finite && rows_ok(&u) {
        out.push(u.clone());
    }
    if lower_finite && upper_finite {
        // Column-wise coefficient signs from the row-major matrix.
        let mut pos = vec![0u32; n];
        let mut neg = vec![0u32; n];
        for (k, &j) in model.row_idx.iter().enumerate() {
            if model.row_val[k] > 0.0 {
                pos[j] += 1;
            } else {
                neg[j] += 1;
            }
        }
        let p: Vec<f64> = (0..n)
            .map(|j| if pos[j] > neg[j] { l[j] } else { u[j] })
            .collect();
        if rows_ok(&p) {
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_reference_sequence() {
        // C++ [rand.predef]: the 10000th output of a default-seeded mt19937.
        let mut g = Mt19937::new(5489);
        let first = g.next_u32();
        let mut last = first;
        for _ in 1..10_000 {
            last = g.next_u32();
        }
        assert_eq!(first, 3_499_211_612);
        assert_eq!(last, 4_123_659_995);
        // Seed 0, first five outputs (MT19937 init_genrand reference values).
        let mut g = Mt19937::new(0);
        let v: Vec<u32> = (0..5).map(|_| g.next_u32()).collect();
        assert_eq!(
            v,
            [
                2_357_136_044,
                2_546_248_239,
                3_071_714_933,
                3_626_093_760,
                2_588_848_963
            ]
        );
        // The canonical draw: (g0 + g1·2³²) / 2⁶⁴ from the next two outputs.
        let mut g = Mt19937::new(0);
        let u = g.uniform01();
        let want: f64 =
            (2_357_136_044.0 + 2_546_248_239.0 * 4_294_967_296.0) / 18_446_744_073_709_551_616.0;
        assert_eq!(u.to_bits(), want.to_bits());
    }

    fn knapsack_cover() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Set cover over 6 binaries: each row needs >= 1 of its columns.
        let rows: [&[usize]; 5] = [&[0, 1], &[1, 2, 3], &[3, 4], &[0, 4, 5], &[2, 5]];
        let mut ptr = vec![0];
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for r in rows {
            for &j in r {
                idx.push(j);
                val.push(1.0);
            }
            ptr.push(idx.len());
        }
        (ptr, idx, val, vec![1.0; 5], vec![f64::INFINITY; 5])
    }

    #[test]
    fn feasibility_jump_finds_a_feasible_cover() {
        let (ptr, idx, val, lo, up) = knapsack_cover();
        let l = vec![0.0; 6];
        let u = vec![1.0; 6];
        let c = vec![3.0, 2.0, 4.0, 1.0, 5.0, 2.0];
        let is_int = vec![true; 6];
        let m = RowModel {
            row_ptr: &ptr,
            row_idx: &idx,
            row_val: &val,
            lo: &lo,
            up: &up,
            l: &l,
            u: &u,
            c: &c,
            is_int: &is_int,
        };
        let (x, effort) = feasibility_jump(&m, None, || false).expect("a cover exists");
        assert!(effort > 0);
        for i in 0..lo.len() {
            let act: f64 = (ptr[i]..ptr[i + 1]).map(|k| val[k] * x[idx[k]]).sum();
            assert!(act >= lo[i] - 1e-9, "row {i} violated: {act}");
        }
        assert!(x.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn trivial_points_respect_rows_and_purity() {
        let (ptr, idx, val, lo, up) = knapsack_cover();
        let l = vec![0.0; 6];
        let u = vec![1.0; 6];
        let c = vec![1.0; 6];
        let is_int = vec![true; 6];
        let m = RowModel {
            row_ptr: &ptr,
            row_idx: &idx,
            row_val: &val,
            lo: &lo,
            up: &up,
            l: &l,
            u: &u,
            c: &c,
            is_int: &is_int,
        };
        // Zero violates every `>= 1` row; all-upper (all ones) covers; the lock
        // point puts each column (all coefficients positive) at its lower bound.
        let pts = trivial_points(&m);
        assert_eq!(pts, vec![vec![1.0; 6]]);
        // A continuous column turns the heuristic off, as in HiGHS.
        let mut mixed = is_int.clone();
        mixed[2] = false;
        let m2 = RowModel {
            is_int: &mixed,
            ..m
        };
        assert!(trivial_points(&m2).is_empty());
    }
}
