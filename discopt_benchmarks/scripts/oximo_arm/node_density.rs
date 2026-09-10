//! Where oximo's speed comes from, measured rather than assumed (#1215, §49).
//!
//! oximo is **not** vectorised: `constraint!(m, c[i in 0..n], ...)` expands to
//! `__add_constraints_over(name, &set, |i| ...)`, which is a plain
//! `for key in set { rule(key) }` loop building one `Constraint` per row. So the
//! question is what makes each *element* cheap. This probe answers it by reading
//! the arena at each stage instead of guessing which pushes the operator
//! overloads make.
//!
//! Build it in the same scratch crate as `main.rs` (see README.md).

use oximo::prelude::*;

const N: usize = 1000;

fn main() {
    let mut checks = 0usize;

    // Stage 1: variable declaration. One `Var` node each; `x[i]` afterwards is a
    // 16-byte `Copy` handle, so *using* a variable pushes nothing.
    let m = Model::new("linear");
    let before = m.arena().len();
    variable!(m, x[i in 0..N] >= 0.0);
    variable!(m, y[i in 0..N] >= 0.0);
    variable!(m, z[i in 0..N] >= 0.0);
    let after_vars = m.arena().len();
    println!("declaring 3x{N} variables: arena {before} -> {after_vars}");
    checks += 1;

    // Stage 2-3: a linear row. The `+` overload checks whether both sides are
    // linear and fuses them into ONE `Linear { coeffs, constant }` node, so
    // `x + y` costs a single node -- not two `Var`s and an `Add`.
    constraint!(m, c0, x[0] + 2.0 * y[0] + 3.0 * z[0] <= 10.0);
    let one_linear = m.arena().len();
    println!("+1 linear row  (x + 2y + 3z <= c): +{} nodes", one_linear - after_vars);
    checks += 1;

    constraint!(m, c1, x[1] + y[1] <= 10.0);
    let two = m.arena().len();
    println!("+1 linear row  (x + y <= c):       +{} nodes", two - one_linear);
    checks += 1;

    // Stage 4-5: no fusion is available for a nonlinear body, so these are the
    // ordinary tree costs.
    constraint!(m, c2, x[2].exp() + y[2] <= 10.0);
    let three = m.arena().len();
    println!("+1 nonlinear   (exp(x) + y <= c):  +{} nodes", three - two);
    checks += 1;

    constraint!(m, c3, x[3] * y[3] + (x[3] + 1.0).log() <= 10.0);
    let four = m.arena().len();
    println!("+1 coupled     (x*y + log(x+1)):   +{} nodes", four - three);
    checks += 1;

    // Stage 6: the row's surviving node already IS the coefficient vector, and
    // the RHS constant is folded into the bound. A linear row costs its writer
    // and its backends no DAG walk at all.
    let cons = m.constraints();
    let c = cons.algebraic().iter().find(|c| c.name == "c0").expect("c0");
    let arena = m.arena();
    println!("\nfinal node of the linear row: {:?}", arena.get(c.lhs));
    println!("its bounds: [{}, {}]", c.lower, c.upper);
    checks += 1;

    // Stage 7: the same fusion makes a long sum flat -- 100 terms, one node.
    drop(arena);
    drop(cons);
    let before_sum = m.arena().len();
    constraint!(m, c4, sum!(x[i] for i in 0..100) <= 10.0);
    println!("\n+1 row summing 100 terms:          +{} nodes", m.arena().len() - before_sum);
    checks += 1;

    eprintln!("# executed: {checks} stage measurements");
    assert!(checks > 0, "probe measured nothing");
}
