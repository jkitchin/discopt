//! One truth table for every `DISCOPT_*` environment flag read from Rust.
//!
//! The Python side has [`discopt._env`](../../../python/discopt/_env.py); this is
//! its Rust twin, with the *same* table so a flag means the same thing on both
//! sides of the FFI boundary:
//!
//! | value (case-insensitive, trimmed) | result |
//! |---|---|
//! | `1` / `true` / `yes` / `on` | `true` |
//! | `0` / `false` / `no` / `off` | `false` |
//! | unset, or empty/whitespace-only | the caller's `default` |
//! | anything else | the caller's `default`, after a warning on stderr |
//!
//! Before this module three Rust flags were *presence* tests
//! (`std::env::var_os(..).is_some()`), so `DISCOPT_PROFILE=0` **enabled**
//! profiling and `DISCOPT_DISABLE_CSE=0` **disabled** CSE — the exact opposite of
//! the repo's documented `=0` escape-hatch convention (architecture review
//! 2026-07-28 §2.4). They now honour `=0`.
//!
//! Unlike the Python helper an unparseable value does not abort: a solver kernel
//! has no exception channel and killing a solve over a typo'd debug flag is worse
//! than falling back. It is loud on stderr instead, which is the same signal a
//! `ValueError` traceback gives, and the Python entry points (which own every
//! documented workflow) refuse loudly before the Rust core is ever reached.

/// Accepted true spellings.
pub const TRUE_VALUES: [&str; 4] = ["1", "true", "yes", "on"];
/// Accepted false spellings.
pub const FALSE_VALUES: [&str; 4] = ["0", "false", "no", "off"];

/// Parse one string against the shared truth table.
///
/// `None` means "not a recognized boolean"; callers decide what to do about it
/// (the env readers below fall back to the default and warn).
pub fn parse_bool(raw: &str) -> Option<bool> {
    let low = raw.trim().to_ascii_lowercase();
    if low.is_empty() {
        return None;
    }
    if TRUE_VALUES.contains(&low.as_str()) {
        return Some(true);
    }
    if FALSE_VALUES.contains(&low.as_str()) {
        return Some(false);
    }
    None
}

/// `name` as a boolean, per the module truth table.
///
/// Unset, empty, or unparseable ⇒ `default` (unparseable also warns on stderr).
pub fn env_bool(name: &str, default: bool) -> bool {
    let raw = match std::env::var(name) {
        Ok(v) => v,
        Err(_) => return default,
    };
    if raw.trim().is_empty() {
        return default;
    }
    match parse_bool(&raw) {
        Some(v) => v,
        None => {
            eprintln!(
                "discopt: {}={:?} is not a boolean (accepted: {} / {}); using the default {}.",
                name,
                raw,
                TRUE_VALUES.join("/"),
                FALSE_VALUES.join("/"),
                default
            );
            default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truth_table_matches_python() {
        for t in ["1", "true", "TRUE", "Yes", "on", " on "] {
            assert_eq!(parse_bool(t), Some(true), "{t:?} should be true");
        }
        for f in ["0", "false", "FALSE", "No", "off", " off "] {
            assert_eq!(parse_bool(f), Some(false), "{f:?} should be false");
        }
        for bad in ["", "   ", "2", "ture", "yes please"] {
            assert_eq!(parse_bool(bad), None, "{bad:?} should not parse");
        }
    }

    #[test]
    fn unset_returns_default() {
        // A name no test or shell would ever set.
        let name = "DISCOPT_ENV_RS_UNSET_PROBE_9df1";
        assert!(std::env::var(name).is_err());
        assert!(env_bool(name, true));
        assert!(!env_bool(name, false));
    }
}
