//! Linear-programming support for the B&B engine.
//!
//! discopt solves LP/QP relaxations with POUNCE (an interior-point method), so
//! a relaxation optimum is the *analytic center* of the optimal face rather
//! than a vertex. This module hosts the pure-Rust machinery that recovers the
//! sharp vertex/basis structure an interior point smears — starting with the
//! [`crossover`] from an interior optimum to a vertex of the optimal face.
//!
//! Everything here is dependency-free dense linear algebra: the systems are
//! small (the number of variables still strictly inside their bounds at an
//! LP optimum), so a hand-written rank-revealing elimination is both adequate
//! and avoids pulling a BLAS/LAPACK stack into the wheel build.

pub mod basis;
pub mod crossover;
pub mod gomory;
pub mod mir;
pub mod simplex;
