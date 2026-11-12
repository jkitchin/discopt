//! The translation of HiGHS's MIP search core (`mip/HighsSearch`, `HighsNodeQueue`,
//! `HighsLpRelaxation`, `HighsMipSolver`, ...) that replaces the MILP driver's search,
//! stage by stage (`docs/dev/highs-mip-replacement-plan.md`).
//!
//! Provenance: translated from HiGHS's C++ source (Copyright (c) HiGHS, MIT License);
//! see `THIRD_PARTY_NOTICES.md` and `lp/highs_cutgen/mod.rs` for why discopt carries
//! HiGHS's design in Rust rather than linking it. Each layer is checked against dumps of
//! HiGHS itself before the driver uses it.

pub mod heur_effort;
pub mod node_queue;
pub mod pseudocost;
pub mod search;

#[cfg(test)]
mod replay_queue;
#[cfg(test)]
mod replay_search;
