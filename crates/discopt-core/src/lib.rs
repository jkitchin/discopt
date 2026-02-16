//! discopt-core: Core MINLP solver engine
//!
//! This crate provides the Branch-and-Bound engine, expression IR,
//! and preprocessing for the discopt MINLP solver.

#![deny(missing_docs)]

pub mod expr;
pub mod bnb;
pub mod nl_parser;
pub mod presolve;

/// Returns the version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.2.0");
    }
}
