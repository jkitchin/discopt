//! A faithful Rust port of HiGHS's cut-generation subsystem (MIT-licensed HiGHS,
//! <https://github.com/ERGO-Code/HiGHS>), built bottom-up so each layer can be
//! checked bit-for-bit against HiGHS itself before the next is ported.
//!
//! Why a port and not more tuning of `lp/cmir.rs`: on the MIPLIB-38+M population
//! HiGHS's root separation is the one stage whose removal costs it certifications
//! (60 -> 47 at 20 s), and a same-LP differential showed its generation and its
//! pool selection only work as a pair (docs/dev/highs-parity-d2-gap-plan.md §12.13-
//! §12.15). Porting pieces into our own pipeline was measured not to transfer.
//!
//! Layers, bottom-up:
//! - [`cdouble`]  `HighsCDouble`, the double-double type the generator computes in;
//! - [`random`]   `HighsRandom` and the hash helpers it is built on;
//! - [`stdalgo`]  `pdqsort` and the libstdc++ algorithms whose tie order the cut
//!   depends on;
//! - [`integers`] `HighsIntegers` (integral scales, continued fractions);
//! - [`core`]     `HighsCutGeneration`'s single-row core and `postprocessCut`;
//! - [`sparse_vec`] `HighsSparseVectorSum`;
//! - [`transformed_lp`] `HighsTransformedLp` and the variable-bound choice it uses;
//! - [`separators`] `HighsLpAggregator`, the cut-generation wrappers, and the path
//!   and tableau separators.
//!
//! The replay fixture `tests/data/highs_cutgen_replay.txt` holds records dumped by
//! a HiGHS build patched to print every `generateCut` call's inputs and outputs
//! (`docs/dev/data/highs-parity/gap-plan-p3/p3_highs_generatecut_dump.patch`). The
//! tests in [`core`] replay each record and require identical bits.

// The modules mirror the C++ statement for statement (index loops over parallel
// arrays, `x == 0 || x > 0`, `h <= n - 1`) so a reader can check them against
// HiGHS line by line; clippy's rewrites would break that correspondence.
#![allow(
    clippy::needless_range_loop,
    clippy::int_plus_one,
    clippy::double_comparisons,
    clippy::if_same_then_else,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::type_complexity,
    // `!(a < b)` is HiGHS's NaN behaviour (std::pair / remove_if); keep it.
    clippy::neg_cmp_op_on_partial_ord
)]

pub mod bridge;
pub mod cdouble;
pub mod core;
pub mod integers;
pub mod modk;
pub mod pool;
pub mod random;
pub mod separators;
pub mod sparse_vec;
pub mod stdalgo;
pub mod transformed_lp;
pub mod vbound;

#[cfg(test)]
mod replay;
#[cfg(test)]
mod replay_pool;
#[cfg(test)]
mod replay_tl;
