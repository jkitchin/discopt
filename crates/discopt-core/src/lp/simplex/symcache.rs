//! #1008 — reuse of the sparse LU's fill-reducing column ordering.
//!
//! # Why
//!
//! Splitting `factorize_sparse` into its two halves (`profile::Phase::LuSymbolic`
//! / `LuNumeric`, added by the same issue) showed that the *symbolic* half —
//! `SparseLuSymbolic::analyze`, i.e. AMD on the `AᵀA` pattern — is 2.7%–25.5% of
//! total LP wall across 18 captured QPLIB relaxation LPs, median ≈10%, and on the
//! sparser bases it costs **more than the numeric factorization it prepares**
//! (QPLIB_1157_rlt0: 68.1 ms symbolic vs 60.7 ms numeric over 31
//! refactorizations; QPLIB_1619_rlt0: 308.6 vs 275.7 over 49).
//!
//! That work is almost entirely redundant. A simplex basis changes by exactly one
//! column per pivot and discopt refactorizes roughly every 48 pivots, so
//! successive bases differ in ~1% of their columns — yet the ordering is
//! recomputed from scratch every time, and `ata_pattern()` is quadratic in the
//! length of the densest row of the basis.
//!
//! # Why reuse is sound
//!
//! [`feral::lu::SparseLuSymbolic`] is *only* `{ m, qcol, qcol_inv }` — a column
//! permutation, nothing structural (no elimination tree, no predicted L/U
//! pattern). `SparseLu::factor` validates only `symbolic.m == a.m`, and the
//! numeric factorization does its own row pivoting on top of whatever column
//! order it is handed. **Any** permutation therefore yields a correct
//! factorization of the matrix actually passed in; feral's own module doc says
//! the handle is "reusable ... across numerically different but structurally
//! identical bases". Reusing a slightly stale ordering can only cost *fill*, not
//! correctness — so the guard here is on fill, and every downstream numeric
//! safeguard (feral's `max_growth`, `zero_pivot_tol`, the periodic exact
//! recompute, `assemble`'s feasibility audit, the Neumaier–Shcherbina safe bound)
//! is untouched.
//!
//! # Regime
//!
//! A different column order changes rounding, hence the pivot sequence, so this
//! is *bound-changing* under CLAUDE.md §5 even though no bound formula moves:
//! default **off**, opt in with `DISCOPT_LU_SYMBOLIC_REUSE=1`, pending the
//! corpus-wide differential panel §5 requires for graduation.

use std::sync::OnceLock;

/// Fill the reused ordering is allowed to add before it is thrown away and a
/// fresh analysis is run. 1.25 = accept up to 25% more `nnz(L+U)` than the fresh
/// analysis achieved on the basis the cache was built from. Tripping the guard
/// costs one wasted numeric factorization, so this trades a rare double-factor
/// against a bounded fill regression; the trip rate is counted
/// (`Ctr::LuSymbolicRefreshFill`) rather than assumed.
pub const MAX_FILL_GROWTH: f64 = 1.25;

const ENV: &str = "DISCOPT_LU_SYMBOLIC_REUSE";

fn parse(raw: &str) -> bool {
    match raw.trim() {
        "" => false,
        "0" => false,
        "1" => true,
        other => panic!("{ENV} must be 0 or 1, got {other:?}"),
    }
}

fn from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var(ENV).map(|v| parse(&v)).unwrap_or(false))
}

std::thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether the ordering may be reused. Default `false` — byte-identical to the
/// pre-#1008 engine.
pub fn enabled() -> bool {
    OVERRIDE.with(|o| o.get()).unwrap_or_else(from_env)
}

/// Run `f` with reuse forced on/off on this thread, restoring the previous
/// setting afterwards (including on panic).
pub fn with_enabled<R>(on: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|o| o.set(self.0));
        }
    }
    let _restore = Restore(OVERRIDE.with(|o| o.replace(Some(on))));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off() {
        // No override set on this fresh test thread, and the env var is unset in
        // CI: the engine must behave exactly as it did before #1008.
        assert!(!enabled(), "symbolic reuse must default to OFF");
    }

    #[test]
    fn scoped_override_applies_and_is_restored() {
        assert!(!enabled());
        with_enabled(true, || assert!(enabled()));
        assert!(!enabled(), "override leaked out of its scope");
    }

    #[test]
    fn override_is_restored_even_when_the_body_panics() {
        let r = std::panic::catch_unwind(|| with_enabled(true, || panic!("boom")));
        assert!(r.is_err());
        assert!(!enabled(), "override survived a panic");
    }

    #[test]
    fn out_of_range_values_are_refused() {
        // Loud refusal, not a silent default (CLAUDE.md §3).
        let mut refused = 0;
        for bad in ["2", "true", "yes", "-1"] {
            assert!(
                std::panic::catch_unwind(|| parse(bad)).is_err(),
                "{bad:?} should be refused"
            );
            refused += 1;
        }
        assert_eq!(refused, 4, "refusal checks executed");
    }

    #[test]
    fn recognised_values_parse() {
        assert!(parse("1"));
        assert!(!parse("0"));
        assert!(!parse(""), "an empty value must not switch the feature on");
        assert!(
            parse(" 1 "),
            "surrounding whitespace must not change the meaning"
        );
    }
}
