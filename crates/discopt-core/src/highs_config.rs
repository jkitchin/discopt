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
//! `DISCOPT_MILP_IMPLIED_INT`, `DISCOPT_LP_DUAL_PERTURB_FIRST`.
//!
//! Default OFF. Garbage is refused.

/// Whether `DISCOPT_HIGHS_CONFIG` turns the HiGHS configuration on. Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("DISCOPT_HIGHS_CONFIG") {
        Err(_) => false,
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" | "" => false,
            other => panic!("DISCOPT_HIGHS_CONFIG={other:?} is not recognized (expected 1 or 0)"),
        },
    })
}
