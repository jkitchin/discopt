//! `DISCOPT_HIGHS_CONFIG`: one switch for the HiGHS configuration of the MILP engine.
//!
//! The HiGHS-design components are judged together, as HiGHS's own components are
//! (its knock-outs on the hard rows of the MIPLIB-38+M panel show no single component
//! it cannot lose; they compound -- `docs/dev/highs-mip-replacement-plan.md`). Each
//! component keeps its own flag; when that flag is unset it takes this switch's value,
//! so `DISCOPT_HIGHS_CONFIG=1` turns the whole configuration on and a component's own
//! `=0` still opts it out (and `=1` on, with the switch unset).
//!
//! Members: `DISCOPT_HIGHS_SEARCH_CUT_LP`, `DISCOPT_HIGHS_SEPA_NODE` (as `1`, replace),
//! `DISCOPT_HIGHS_SEPA_CUTOFF_BOX`, `DISCOPT_LP_DUAL_NOISE_ROWS`,
//! `DISCOPT_MILP_IMPLIED_INT`, `DISCOPT_MILP_SPARSIFY`, `DISCOPT_LP_DUAL_PERTURB_FIRST`,
//! `DISCOPT_LP_WARM_FLIP_BOXED`, `DISCOPT_HIGHS_HEUR_EFFORT`, `DISCOPT_HIGHS_SUBMIP_PRESOLVE`,
//! `DISCOPT_LP_SUBTOL_LADDER` (HiGHS
//! has no primal repair of sub-tolerance reduced costs: it treats them as optimal at its
//! dual tolerance; the ladder is the rigorous counterpart -- skip the repair when the
//! node-bound ladder certifies the vertex).
//!
//! **Graduated default-ON for the MILP engine** (2026-11-12, `highs-mip-replacement-plan.md`:
//! two 62-row panels, 53 vs 52 and 54 vs 52 certified against the previous default, 0
//! violations). `DISCOPT_HIGHS_CONFIG=0` restores the previous engine. The LP-level members
//! stay off for LPs solved outside the MILP engine unless the switch is set to `1`
//! ([`lp_default`]). Garbage is refused.

/// `DISCOPT_HIGHS_CONFIG` as set: `None` when unset. Read once; garbage refused.
fn raw() -> Option<bool> {
    static V: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("DISCOPT_HIGHS_CONFIG") {
        Err(_) => None,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" | "" => Some(false),
            other => panic!("DISCOPT_HIGHS_CONFIG={other:?} is not recognized (expected 1 or 0)"),
        },
    })
}

/// Whether the MILP engine runs the HiGHS configuration. **Graduated default-ON**
/// (2026-11-12): `DISCOPT_HIGHS_CONFIG=0` is the opt-out.
pub fn enabled() -> bool {
    // A test pinned to the pre-graduation defaults (`cmir::legacy_defaults`) sees the
    // switch's old default.
    raw().unwrap_or(!crate::lp::cmir::legacy_defaults())
}

/// The default of the configuration's LP-level members (`DISCOPT_LP_DUAL_NOISE_ROWS`,
/// `DISCOPT_LP_DUAL_PERTURB_FIRST`, `DISCOPT_LP_WARM_FLIP_BOXED`,
/// `DISCOPT_LP_SUBTOL_LADDER`) in `SimplexOptions::default()`, i.e. for every LP solved
/// outside the MILP engine (spatial B&B, NLP-BB, the Python LP bindings): ON only when
/// `DISCOPT_HIGHS_CONFIG=1` is set explicitly. The graduation was measured on MILP
/// solves only; the MILP driver turns them on itself (`dual::milp_lp_members`).
pub fn lp_default() -> bool {
    raw().unwrap_or(false)
}
