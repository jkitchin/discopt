//! A deterministic work counter for solver effort that LP iterations do not see.
//!
//! **Why.** A heuristic sub-MIP is bounded by work, not by the wall clock, so that a
//! certifying solve explores the same tree whatever its time limit and machine load
//! (#1116 role 2; `test_1236_determinism_contract[nvs02]` failed in #2220's CI with
//! node counts `[2, 4, 2]` under a wall-share budget). LP iterations alone are not
//! that work: measured on `gen-ip054` (#2220), one RENS sub-MIP spent 3.46 s on
//! 1,140 LP iterations, and a `sample` of it put nearly every frame in the exact
//! rational Farkas check (`lp::simplex::farkas_exact`), which costs nothing in
//! iterations. An iteration cap loose enough for the rest let that sub-MIP take the
//! parent's time (parent nodes 112,085 -> 97,527, dual bounds worse on 22 of 112
//! MIPLIB rows). Nor is an iteration a fixed cost: it is set by the fill of the
//! basis factor. A `reblock166` sub-MIP spent 6.3 s factorizing its root basis with
//! no iteration counted, so the LU layer counts the entries it touches.
//!
//! **What.** Code that does uncounted work calls [`add`] with a [`Kind`] and a count of
//! its elementary operations (64-bit limb operations for the exact arithmetic,
//! entries touched for the LU). The count goes to the *sink* of the solve running on
//! this thread, if any. A solve that wants its work measured opens a sink with
//! [`Scope::new_solve`] (optionally with a [`Limit`]); its total rolls up into the
//! enclosing sink when it closes.
//!
//! **Parallel tasks.** The parallel node loop and the batch LP solver capture a
//! [`Handle`] with [`current`] before they fan out, and each rayon task runs under
//! [`Scope::enter`]: a private child sink that starts from the parent's total *as
//! captured* and adds its own work to the parent when the task ends. So what a task
//! reads (and [`over_limit`] decides) is its own work plus a fixed snapshot, never
//! another task's progress, and the parent's total after the join is the same sum
//! whatever thread ran which task.
//!
//! The count is a function of the inputs only (no clock), so a budget on it is
//! reproducible.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The kinds of uncounted work, each a separate total (they are weighted
/// differently by the budget that reads them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 64-bit limb operations of exact rational arithmetic.
    ExactLimb = 0,
    /// Entries touched by LU factorization, solves and updates.
    Lu = 1,
    /// Splay-tree steps of the mod-k separator's GF(k) elimination
    /// (`lp::highs_cutgen::modk`): 71% of a `reblock166` sub-MIP that overran its
    /// budget 5x was there, with no LU or iteration count to show for it.
    Sepa = 2,
}

/// Number of [`Kind`]s.
pub const KINDS: usize = 3;

/// A budget on a solve's [`Kind`] totals: stop once `Σ weight[k] · total[k] ≥ units`.
/// Checked inside long-running loops (the simplex) through [`over_limit`], so a
/// single LP cannot run far past its solve's budget.
#[derive(Clone, Copy, Debug)]
pub struct Limit {
    /// The budget, in the caller's work units.
    pub units: u64,
    /// Units per count of each [`Kind`], indexed by `Kind as usize`.
    pub weight: [u64; KINDS],
}

/// One solve's (or one task's) work totals, by [`Kind`].
#[derive(Debug)]
pub struct Totals {
    own: [AtomicU64; KINDS],
    /// The parent's totals when this task's [`Handle`] was captured (zero for a
    /// solve's own sink).
    base: [u64; KINDS],
    limit: Option<Limit>,
}

impl Totals {
    fn new(base: [u64; KINDS], limit: Option<Limit>) -> Self {
        Totals {
            own: Default::default(),
            base,
            limit,
        }
    }
    /// The total of `kind` seen from here: the snapshot plus this sink's own work.
    pub fn get(&self, kind: Kind) -> u64 {
        self.base[kind as usize] + self.own[kind as usize].load(Ordering::Relaxed)
    }
    fn own(&self) -> [u64; KINDS] {
        std::array::from_fn(|k| self.own[k].load(Ordering::Relaxed))
    }
    fn add(&self, kind: Kind, units: u64) {
        self.own[kind as usize].fetch_add(units, Ordering::Relaxed);
    }
    fn weighted(&self, lim: &Limit) -> u64 {
        (0..KINDS)
            .map(|k| {
                lim.weight[k].saturating_mul(self.base[k] + self.own[k].load(Ordering::Relaxed))
            })
            .fold(0u64, u64::saturating_add)
    }
    fn over_limit(&self) -> bool {
        self.limit
            .is_some_and(|lim| self.weighted(&lim) >= lim.units)
    }
    fn remaining(&self) -> Option<u64> {
        self.limit
            .map(|lim| lim.units.saturating_sub(self.weighted(&lim)))
    }
}

/// One solve's work totals.
pub type Sink = Arc<Totals>;

/// A sink captured on the spawning thread for its rayon tasks ([`current`]).
#[derive(Clone, Debug)]
pub struct Handle {
    parent: Sink,
    base: [u64; KINDS],
}

thread_local! {
    static CURRENT: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

/// Credit `units` of `kind` to the solve running on this thread (a no-op outside
/// any scope).
#[inline]
pub fn add(kind: Kind, units: u64) {
    CURRENT.with(|c| {
        if let Some(s) = c.borrow().as_ref() {
            s.add(kind, units);
        }
    });
}

/// The current solve's total of `kind` (0 outside any scope).
pub fn current_total(kind: Kind) -> u64 {
    CURRENT.with(|c| c.borrow().as_ref().map_or(0, |s| s.get(kind)))
}

/// Whether the solve running on this thread has spent its [`Limit`] (false outside
/// any scope or without a limit).
#[inline]
pub fn over_limit() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(|s| s.over_limit()))
}

/// What is left of the [`Limit`] of the solve running on this thread, in its units
/// (`None` outside any scope or without a limit). A nested budgeted solve sizes its
/// own limit from this.
pub fn remaining() -> Option<u64> {
    CURRENT.with(|c| c.borrow().as_ref().and_then(|s| s.remaining()))
}

/// The current sink and a snapshot of its totals, to hand to rayon tasks.
pub fn current() -> Option<Handle> {
    CURRENT.with(|c| {
        c.borrow().as_ref().map(|s| Handle {
            parent: s.clone(),
            base: std::array::from_fn(|k| s.base[k] + s.own[k].load(Ordering::Relaxed)),
        })
    })
}

/// A sink installed on this thread until dropped; the previous one is restored and
/// this sink's own work is added to its parent.
pub struct Scope {
    prev: Option<Sink>,
    mine: Option<Sink>,
    parent: Option<Sink>,
}

impl Scope {
    /// Open a new sink for a solve, with an optional [`Limit`] on its own work.
    pub fn new_solve(limit: Option<Limit>) -> (Scope, Sink) {
        let s: Sink = Arc::new(Totals::new([0; KINDS], limit));
        let parent = CURRENT.with(|c| c.borrow().clone());
        (Scope::install(Some(s.clone()), parent), s)
    }

    /// Run a task under a child of the captured sink (see the module docs).
    pub fn enter(h: Option<Handle>) -> Scope {
        match h {
            None => Scope::install(None, None),
            Some(h) => {
                let child = Arc::new(Totals::new(h.base, h.parent.limit));
                Scope::install(Some(child), Some(h.parent))
            }
        }
    }

    fn install(sink: Option<Sink>, parent: Option<Sink>) -> Scope {
        let prev = CURRENT.with(|c| std::mem::replace(&mut *c.borrow_mut(), sink.clone()));
        Scope {
            prev,
            mine: sink,
            parent,
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let (Some(p), Some(m)) = (self.parent.as_ref(), self.mine.as_ref()) {
            for (k, v) in m.own().into_iter().enumerate() {
                p.own[k].fetch_add(v, Ordering::Relaxed);
            }
        }
        CURRENT.with(|c| *c.borrow_mut() = self.prev.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_goes_to_the_innermost_scope_and_rolls_up() {
        add(Kind::Lu, 5); // outside any scope: dropped
        let (outer, o) = Scope::new_solve(None);
        add(Kind::Lu, 3);
        {
            let (_inner, i) = Scope::new_solve(None);
            add(Kind::Lu, 10);
            add(Kind::ExactLimb, 2);
            assert_eq!(i.get(Kind::Lu), 10);
            assert_eq!(o.get(Kind::Lu), 3);
        }
        assert_eq!(o.get(Kind::Lu), 13);
        assert_eq!(o.get(Kind::ExactLimb), 2);
        drop(outer);
        assert!(current().is_none());
    }

    #[test]
    fn a_task_sees_its_own_work_and_the_snapshot_only() {
        let lim = Limit {
            units: 100,
            weight: [0, 1, 0],
        };
        let (_s, o) = Scope::new_solve(Some(lim));
        add(Kind::Lu, 60);
        let h = current();
        add(Kind::Lu, 1000); // after the capture: invisible to the tasks
        std::thread::scope(|t| {
            for _ in 0..2 {
                let h = h.clone();
                t.spawn(move || {
                    let _g = Scope::enter(h);
                    add(Kind::Lu, 30);
                    assert_eq!(current_total(Kind::Lu), 90);
                    assert!(!over_limit());
                    add(Kind::Lu, 10);
                    assert!(over_limit());
                });
            }
        });
        assert_eq!(o.get(Kind::Lu), 60 + 1000 + 2 * 40);
        assert!(over_limit());
    }

    #[test]
    fn no_limit_is_never_over() {
        let (_s, _o) = Scope::new_solve(None);
        add(Kind::Lu, u64::MAX / 2);
        assert!(!over_limit());
        assert_eq!(remaining(), None);
    }

    #[test]
    fn remaining_counts_down_and_saturates() {
        let lim = Limit {
            units: 100,
            weight: [0, 2, 0],
        };
        let (_s, _o) = Scope::new_solve(Some(lim));
        assert_eq!(remaining(), Some(100));
        add(Kind::Lu, 30);
        add(Kind::ExactLimb, 1000); // weight 0
        assert_eq!(remaining(), Some(40));
        add(Kind::Lu, 30);
        assert_eq!(remaining(), Some(0));
    }
}
