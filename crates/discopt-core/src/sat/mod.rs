//! A compact CDCL SAT solver with assumptions and final-conflict cores.
//!
//! Built for one job: give the MILP IIS engine (`python/discopt/_iis_milp.py`) a
//! cheap unsatisfiable core over the rows of a pure-binary clause model, so the
//! deletion filter starts from a small candidate instead of discovering it one
//! MILP probe at a time. Measured on MIPcc27's smallset with an external solver
//! as the instrument (the motivation, not a dependency): the first UNSAT proof on
//! the `bf*`/`ssa`/`jnh` clause models needs 18-50 conflicts and a whole
//! core-refined row-MUS under ~200, so this is deliberately small -- no clause
//! deletion, no inprocessing.
//!
//! Classic MiniSat structure (Een & Sorensson 2003): two-watched-literal
//! propagation, first-UIP learning, VSIDS on a binary heap, phase saving, Luby
//! restarts. Assumptions are decided first, one per decision level; when one is
//! found false, `analyze_final` collects the assumptions its falsity rests on --
//! the core. Learnt clauses are implied by the clause database, so they are kept
//! across incremental `solve` calls.
//!
//! The solver's answers are never trusted by the IIS engine on their own: a core
//! only *proposes* rows, and the engine still proves the returned subsystem
//! infeasible with HiGHS (presolve on and off) and certifies every member.

/// A literal: `2 * var + sign`, sign 1 meaning negated.
pub type Lit = u32;

const NO_REASON: u32 = u32::MAX;

/// Positive (`neg == false`) or negative literal of `var`.
#[inline]
pub fn mk_lit(var: usize, neg: bool) -> Lit {
    ((var as u32) << 1) | (neg as u32)
}

#[inline]
fn lit_var(l: Lit) -> usize {
    (l >> 1) as usize
}

#[inline]
fn lit_neg(l: Lit) -> Lit {
    l ^ 1
}

#[inline]
fn lit_is_neg(l: Lit) -> bool {
    l & 1 == 1
}

/// Outcome of [`Solver::solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveResult {
    /// Satisfiable under the assumptions; see [`Solver::model`].
    Sat,
    /// Unsatisfiable under the assumptions; see [`Solver::core`].
    Unsat,
    /// The conflict budget ran out first.
    Unknown,
}

/// Search counters (cumulative over all `solve` calls).
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Conflicts analysed.
    pub conflicts: u64,
    /// Branching decisions (assumption decisions included).
    pub decisions: u64,
    /// Literals propagated.
    pub propagations: u64,
}

/// Binary max-heap of variables keyed by activity.
struct VarHeap {
    heap: Vec<usize>,
    pos: Vec<usize>, // usize::MAX when absent
}

impl VarHeap {
    fn new(n: usize) -> Self {
        VarHeap {
            heap: Vec::with_capacity(n),
            pos: vec![usize::MAX; n],
        }
    }
    fn contains(&self, v: usize) -> bool {
        self.pos[v] != usize::MAX
    }
    fn up(&mut self, mut i: usize, act: &[f64]) {
        let v = self.heap[i];
        while i > 0 {
            let p = (i - 1) / 2;
            if act[self.heap[p]] >= act[v] {
                break;
            }
            self.heap[i] = self.heap[p];
            self.pos[self.heap[i]] = i;
            i = p;
        }
        self.heap[i] = v;
        self.pos[v] = i;
    }
    fn down(&mut self, mut i: usize, act: &[f64]) {
        let v = self.heap[i];
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            if l >= n {
                break;
            }
            let r = l + 1;
            let c = if r < n && act[self.heap[r]] > act[self.heap[l]] {
                r
            } else {
                l
            };
            if act[self.heap[c]] <= act[v] {
                break;
            }
            self.heap[i] = self.heap[c];
            self.pos[self.heap[i]] = i;
            i = c;
        }
        self.heap[i] = v;
        self.pos[v] = i;
    }
    fn insert(&mut self, v: usize, act: &[f64]) {
        if self.contains(v) {
            return;
        }
        self.heap.push(v);
        let i = self.heap.len() - 1;
        self.pos[v] = i;
        self.up(i, act);
    }
    fn increased(&mut self, v: usize, act: &[f64]) {
        if self.contains(v) {
            let i = self.pos[v];
            self.up(i, act);
        }
    }
    fn pop(&mut self, act: &[f64]) -> Option<usize> {
        if self.heap.is_empty() {
            return None;
        }
        let top = self.heap[0];
        let last = self.heap.pop().unwrap();
        self.pos[top] = usize::MAX;
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.pos[last] = 0;
            self.down(0, act);
        }
        Some(top)
    }
}

/// CDCL solver over variables `0..n_vars`.
pub struct Solver {
    n: usize,
    clauses: Vec<Vec<Lit>>,
    watches: Vec<Vec<u32>>, // by literal: clauses watching it (visited when it becomes false)
    assigns: Vec<i8>,       // -1 unassigned, 0 false, 1 true
    level: Vec<u32>,
    reason: Vec<u32>,
    trail: Vec<Lit>,
    trail_lim: Vec<usize>,
    qhead: usize,
    activity: Vec<f64>,
    var_inc: f64,
    heap: VarHeap,
    polarity: Vec<bool>, // saved phase: true = last value was false
    seen: Vec<bool>,
    ok: bool,
    core: Vec<Lit>,
    model: Vec<bool>,
    stats: Stats,
}

impl Solver {
    /// A solver with `n_vars` variables and no clauses.
    pub fn new(n_vars: usize) -> Self {
        let mut s = Solver {
            n: n_vars,
            clauses: Vec::new(),
            watches: vec![Vec::new(); 2 * n_vars],
            assigns: vec![-1; n_vars],
            level: vec![0; n_vars],
            reason: vec![NO_REASON; n_vars],
            trail: Vec::with_capacity(n_vars),
            trail_lim: Vec::new(),
            qhead: 0,
            activity: vec![0.0; n_vars],
            var_inc: 1.0,
            heap: VarHeap::new(n_vars),
            polarity: vec![true; n_vars],
            seen: vec![false; n_vars],
            ok: true,
            core: Vec::new(),
            model: Vec::new(),
            stats: Stats::default(),
        };
        for v in 0..n_vars {
            s.heap.insert(v, &s.activity);
        }
        s
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize {
        self.n
    }

    /// Cumulative search counters.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    #[inline]
    fn value(&self, l: Lit) -> i8 {
        let a = self.assigns[lit_var(l)];
        if a < 0 {
            -1
        } else if (a == 1) != lit_is_neg(l) {
            1
        } else {
            0
        }
    }

    #[inline]
    fn decision_level(&self) -> usize {
        self.trail_lim.len()
    }

    fn enqueue(&mut self, l: Lit, reason: u32) {
        let v = lit_var(l);
        self.assigns[v] = if lit_is_neg(l) { 0 } else { 1 };
        self.level[v] = self.decision_level() as u32;
        self.reason[v] = reason;
        self.trail.push(l);
    }

    fn cancel_until(&mut self, lvl: usize) {
        if self.decision_level() <= lvl {
            return;
        }
        let stop = self.trail_lim[lvl];
        for i in (stop..self.trail.len()).rev() {
            let l = self.trail[i];
            let v = lit_var(l);
            self.assigns[v] = -1;
            self.reason[v] = NO_REASON;
            self.polarity[v] = lit_is_neg(l);
            self.heap.insert(v, &self.activity);
        }
        self.trail.truncate(stop);
        self.trail_lim.truncate(lvl);
        self.qhead = stop;
    }

    /// Add a clause (DIMACS-free form: literals from [`mk_lit`]). Returns `false`
    /// once the database is unsatisfiable at level 0. Must be called between
    /// `solve` calls (the solver always returns to level 0).
    pub fn add_clause(&mut self, lits: &[Lit]) -> bool {
        assert_eq!(self.decision_level(), 0, "add_clause above level 0");
        if !self.ok {
            return false;
        }
        let mut c: Vec<Lit> = lits.to_vec();
        for &l in &c {
            assert!(lit_var(l) < self.n, "literal of an unknown variable");
        }
        c.sort_unstable();
        c.dedup();
        // tautology or already satisfied at level 0 -> nothing to add
        for w in c.windows(2) {
            if w[1] == lit_neg(w[0]) {
                return true;
            }
        }
        if c.iter().any(|&l| self.value(l) == 1) {
            return true;
        }
        c.retain(|&l| self.value(l) != 0);
        match c.len() {
            0 => {
                self.ok = false;
                false
            }
            1 => {
                self.enqueue(c[0], NO_REASON);
                if self.propagate().is_some() {
                    self.ok = false;
                }
                self.ok
            }
            _ => {
                let ci = self.clauses.len() as u32;
                self.watches[c[0] as usize].push(ci);
                self.watches[c[1] as usize].push(ci);
                self.clauses.push(c);
                true
            }
        }
    }

    /// Unit propagation; returns a conflicting clause index.
    fn propagate(&mut self) -> Option<u32> {
        while self.qhead < self.trail.len() {
            let p = self.trail[self.qhead];
            self.qhead += 1;
            self.stats.propagations += 1;
            let false_lit = lit_neg(p);
            let ws = std::mem::take(&mut self.watches[false_lit as usize]);
            let mut kept: Vec<u32> = Vec::with_capacity(ws.len());
            let mut conflict = None;
            let mut idx = 0;
            while idx < ws.len() {
                let ci = ws[idx];
                idx += 1;
                let c = &mut self.clauses[ci as usize];
                if c[0] == false_lit {
                    c.swap(0, 1);
                }
                let first = c[0];
                // value() borrows self immutably; read assignment directly
                let first_val = {
                    let a = self.assigns[lit_var(first)];
                    if a < 0 {
                        -1
                    } else if (a == 1) != lit_is_neg(first) {
                        1
                    } else {
                        0
                    }
                };
                if first_val == 1 {
                    kept.push(ci);
                    continue;
                }
                let mut moved = false;
                for k in 2..c.len() {
                    let l = c[k];
                    let a = self.assigns[lit_var(l)];
                    let lv = if a < 0 {
                        -1
                    } else if (a == 1) != lit_is_neg(l) {
                        1
                    } else {
                        0
                    };
                    if lv != 0 {
                        c.swap(1, k);
                        let nw = c[1] as usize;
                        self.watches[nw].push(ci);
                        moved = true;
                        break;
                    }
                }
                if moved {
                    continue;
                }
                kept.push(ci);
                if first_val == 0 {
                    conflict = Some(ci);
                    kept.extend_from_slice(&ws[idx..]);
                    break;
                }
                self.enqueue(first, ci);
            }
            // pushes to watches[false_lit] cannot happen above (a new watch is never false)
            let mut existing = std::mem::take(&mut self.watches[false_lit as usize]);
            kept.append(&mut existing);
            self.watches[false_lit as usize] = kept;
            if conflict.is_some() {
                self.qhead = self.trail.len();
                return conflict;
            }
        }
        None
    }

    fn bump(&mut self, v: usize) {
        self.activity[v] += self.var_inc;
        if self.activity[v] > 1e100 {
            for a in self.activity.iter_mut() {
                *a *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }
        self.heap.increased(v, &self.activity);
    }

    /// First-UIP conflict analysis: (learnt clause with the asserting literal
    /// first, backtrack level).
    fn analyze(&mut self, mut confl: u32) -> (Vec<Lit>, usize) {
        let mut learnt: Vec<Lit> = vec![0];
        let mut path_c = 0usize;
        let mut p: Option<Lit> = None;
        let mut index = self.trail.len();
        let dl = self.decision_level() as u32;
        loop {
            let start = if p.is_none() { 0 } else { 1 };
            let c = self.clauses[confl as usize].clone();
            for &q in &c[start..] {
                let v = lit_var(q);
                if !self.seen[v] && self.level[v] > 0 {
                    self.bump(v);
                    self.seen[v] = true;
                    if self.level[v] >= dl {
                        path_c += 1;
                    } else {
                        learnt.push(q);
                    }
                }
            }
            loop {
                index -= 1;
                if self.seen[lit_var(self.trail[index])] {
                    break;
                }
            }
            let pl = self.trail[index];
            p = Some(pl);
            let v = lit_var(pl);
            self.seen[v] = false;
            path_c -= 1;
            if path_c == 0 {
                break;
            }
            confl = self.reason[v];
            debug_assert!(confl != NO_REASON);
        }
        learnt[0] = lit_neg(p.unwrap());
        for &l in &learnt[1..] {
            self.seen[lit_var(l)] = false;
        }
        let mut bt = 0usize;
        if learnt.len() > 1 {
            let mut max_i = 1;
            for i in 2..learnt.len() {
                if self.level[lit_var(learnt[i])] > self.level[lit_var(learnt[max_i])] {
                    max_i = i;
                }
            }
            learnt.swap(1, max_i);
            bt = self.level[lit_var(learnt[1])] as usize;
        }
        self.var_inc /= 0.95;
        (learnt, bt)
    }

    /// Assumptions responsible for `p` (the negation of a failed assumption)
    /// being true: the final conflict, as the failed assumption literals.
    fn analyze_final(&mut self, p: Lit) {
        self.core.clear();
        self.core.push(lit_neg(p));
        if self.decision_level() == 0 {
            return;
        }
        self.seen[lit_var(p)] = true;
        for i in (self.trail_lim[0]..self.trail.len()).rev() {
            let x = lit_var(self.trail[i]);
            if !self.seen[x] {
                continue;
            }
            let r = self.reason[x];
            if r == NO_REASON {
                // a decision at an assumption level is an assumption
                debug_assert!(self.level[x] > 0);
                self.core.push(self.trail[i]);
            } else {
                let c = &self.clauses[r as usize];
                for &q in &c[1..] {
                    if self.level[lit_var(q)] > 0 {
                        self.seen[lit_var(q)] = true;
                    }
                }
            }
            self.seen[x] = false;
        }
        self.seen[lit_var(p)] = false;
        self.core.sort_unstable();
        self.core.dedup();
    }

    fn luby(y: f64, mut x: u64) -> f64 {
        let (mut size, mut seq) = (1u64, 0u32);
        while size < x + 1 {
            seq += 1;
            size = 2 * size + 1;
        }
        while size - 1 != x {
            size = (size - 1) >> 1;
            seq -= 1;
            x %= size;
        }
        y.powi(seq as i32)
    }

    /// Solve under `assumptions`. `conflict_limit` bounds the conflicts of THIS
    /// call (`None`: unbounded). The solver returns to level 0 afterwards.
    pub fn solve(&mut self, assumptions: &[Lit], conflict_limit: Option<u64>) -> SolveResult {
        self.core.clear();
        self.model.clear();
        if !self.ok {
            return SolveResult::Unsat;
        }
        for &a in assumptions {
            assert!(lit_var(a) < self.n, "assumption of an unknown variable");
        }
        let start_conflicts = self.stats.conflicts;
        let mut restarts = 0u64;
        let mut until_restart = (Self::luby(2.0, 0) * 100.0) as u64;
        let mut since_restart = 0u64;
        let result = loop {
            if let Some(confl) = self.propagate() {
                self.stats.conflicts += 1;
                since_restart += 1;
                if self.decision_level() == 0 {
                    self.ok = false;
                    break SolveResult::Unsat;
                }
                let (learnt, bt) = self.analyze(confl);
                self.cancel_until(bt);
                if learnt.len() == 1 {
                    self.enqueue(learnt[0], NO_REASON);
                } else {
                    let ci = self.clauses.len() as u32;
                    self.watches[learnt[0] as usize].push(ci);
                    self.watches[learnt[1] as usize].push(ci);
                    let asserting = learnt[0];
                    self.clauses.push(learnt);
                    self.enqueue(asserting, ci);
                }
                if let Some(lim) = conflict_limit {
                    if self.stats.conflicts - start_conflicts >= lim {
                        break SolveResult::Unknown;
                    }
                }
            } else {
                if since_restart >= until_restart {
                    restarts += 1;
                    since_restart = 0;
                    until_restart = (Self::luby(2.0, restarts) * 100.0) as u64;
                    self.cancel_until(0);
                    continue;
                }
                let mut next: Option<Lit> = None;
                let mut failed: Option<Lit> = None;
                while self.decision_level() < assumptions.len() {
                    let a = assumptions[self.decision_level()];
                    match self.value(a) {
                        1 => self.trail_lim.push(self.trail.len()), // dummy level
                        0 => {
                            failed = Some(a);
                            break;
                        }
                        _ => {
                            next = Some(a);
                            break;
                        }
                    }
                }
                if let Some(a) = failed {
                    self.analyze_final(lit_neg(a));
                    break SolveResult::Unsat;
                }
                if next.is_none() {
                    while let Some(v) = self.heap.pop(&self.activity) {
                        if self.assigns[v] < 0 {
                            next = Some(mk_lit(v, self.polarity[v]));
                            break;
                        }
                    }
                }
                match next {
                    None => {
                        self.model = (0..self.n).map(|v| self.assigns[v] == 1).collect();
                        break SolveResult::Sat;
                    }
                    Some(l) => {
                        self.stats.decisions += 1;
                        self.trail_lim.push(self.trail.len());
                        self.enqueue(l, NO_REASON);
                    }
                }
            }
        };
        self.cancel_until(0);
        result
    }

    /// After `Unsat`: the failed assumptions (a subset of the assumptions passed
    /// to `solve`) whose conjunction is already unsatisfiable. Empty when the
    /// clause database itself is unsatisfiable.
    pub fn core(&self) -> &[Lit] {
        &self.core
    }

    /// After `Sat`: the value of every variable.
    pub fn model(&self) -> &[bool] {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG so the tests need no crate.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn satisfies(clauses: &[Vec<Lit>], x: &[bool]) -> bool {
        clauses
            .iter()
            .all(|c| c.iter().any(|&l| x[lit_var(l)] != lit_is_neg(l)))
    }

    /// Brute force: is `clauses` satisfiable with the `fixed` literals true?
    fn brute(n: usize, clauses: &[Vec<Lit>], fixed: &[Lit]) -> bool {
        (0u64..(1u64 << n)).any(|bits| {
            let x: Vec<bool> = (0..n).map(|v| bits >> v & 1 == 1).collect();
            fixed.iter().all(|&l| x[lit_var(l)] != lit_is_neg(l)) && satisfies(clauses, &x)
        })
    }

    fn random_cnf(rng: &mut Rng, n: usize, m: usize, k: usize) -> Vec<Vec<Lit>> {
        (0..m)
            .map(|_| {
                (0..k)
                    .map(|_| mk_lit(rng.below(n as u64) as usize, rng.below(2) == 1))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn trivial_sat_and_unsat() {
        let mut s = Solver::new(2);
        assert!(s.add_clause(&[mk_lit(0, false), mk_lit(1, false)]));
        assert_eq!(s.solve(&[], None), SolveResult::Sat);
        assert!(s.model()[0] || s.model()[1]);
        s.add_clause(&[mk_lit(0, true)]);
        s.add_clause(&[mk_lit(1, true)]);
        assert_eq!(s.solve(&[], None), SolveResult::Unsat);
        assert!(s.core().is_empty());
    }

    fn pigeonhole(p: usize, h: usize) -> (usize, Vec<Vec<Lit>>) {
        let var = |i: usize, j: usize| i * h + j;
        let mut cls = Vec::new();
        for i in 0..p {
            cls.push((0..h).map(|j| mk_lit(var(i, j), false)).collect());
        }
        for j in 0..h {
            for a in 0..p {
                for b in a + 1..p {
                    cls.push(vec![mk_lit(var(a, j), true), mk_lit(var(b, j), true)]);
                }
            }
        }
        (p * h, cls)
    }

    #[test]
    fn pigeonhole_needs_real_search() {
        let (n, cls) = pigeonhole(6, 5);
        let mut s = Solver::new(n);
        for c in &cls {
            s.add_clause(c);
        }
        assert_eq!(s.solve(&[], None), SolveResult::Unsat);
        assert!(s.stats().conflicts > 0);
        let (n, cls) = pigeonhole(5, 5);
        let mut s = Solver::new(n);
        for c in &cls {
            s.add_clause(c);
        }
        assert_eq!(s.solve(&[], None), SolveResult::Sat);
        assert!(satisfies(&cls, s.model()));
    }

    #[test]
    fn random_3sat_agrees_with_brute_force() {
        let mut rng = Rng(7);
        let mut checked = 0;
        for _ in 0..400 {
            let n = 4 + rng.below(9) as usize; // 4..12
            let m = (n as f64 * (3.0 + rng.below(30) as f64 / 10.0)) as usize;
            let cls = random_cnf(&mut rng, n, m, 3);
            let mut s = Solver::new(n);
            for c in &cls {
                s.add_clause(c);
            }
            let r = s.solve(&[], None);
            let truth = brute(n, &cls, &[]);
            assert_eq!(r == SolveResult::Sat, truth);
            if r == SolveResult::Sat {
                assert!(satisfies(&cls, s.model()));
            }
            checked += 1;
        }
        assert_eq!(checked, 400);
    }

    /// Selector-encoded clause sets, incremental solving under assumptions:
    /// verdicts agree with brute force, SAT models satisfy the assumptions, and
    /// every UNSAT core is a subset of the assumptions that is itself UNSAT.
    #[test]
    fn assumptions_and_cores_agree_with_brute_force() {
        let mut rng = Rng(2093);
        let (mut cores, mut sats) = (0, 0);
        for _ in 0..150 {
            let n = 4 + rng.below(5) as usize; // 4..8 problem vars
            let m = 6 + rng.below(10) as usize; // clauses, each with a selector
            let k = 1 + rng.below(3) as usize;
            let cls = random_cnf(&mut rng, n, m, k);
            let mut s = Solver::new(n + m);
            let mut sel_cls = Vec::new();
            for (i, c) in cls.iter().enumerate() {
                let mut cc = c.clone();
                cc.push(mk_lit(n + i, true)); // c OR not s_i
                s.add_clause(&cc);
                sel_cls.push(cc);
            }
            for _trial in 0..4 {
                let assumed: Vec<Lit> = (0..m)
                    .filter(|_| rng.below(3) != 0)
                    .map(|i| mk_lit(n + i, false))
                    .collect();
                let r = s.solve(&assumed, None);
                let truth = brute(n + m, &sel_cls, &assumed);
                assert_eq!(r == SolveResult::Sat, truth);
                if r == SolveResult::Sat {
                    let x = s.model();
                    assert!(satisfies(&sel_cls, x));
                    assert!(assumed.iter().all(|&a| x[lit_var(a)] != lit_is_neg(a)));
                    sats += 1;
                } else {
                    let core = s.core().to_vec();
                    assert!(
                        core.iter().all(|l| assumed.contains(l)),
                        "core outside assumptions"
                    );
                    assert!(!brute(n + m, &sel_cls, &core), "core is satisfiable");
                    cores += 1;
                }
            }
        }
        assert!(cores > 50 && sats > 50, "cores {cores} sats {sats}");
    }

    #[test]
    fn conflict_budget_returns_unknown() {
        let (n, cls) = pigeonhole(9, 8);
        let mut s = Solver::new(n);
        for c in &cls {
            s.add_clause(c);
        }
        assert_eq!(s.solve(&[], Some(10)), SolveResult::Unknown);
        // the solver stays usable after a budget stop
        assert_eq!(s.solve(&[], Some(10)), SolveResult::Unknown);
    }
}
